"""
Objective intraday strategy lab for 2-year crypto perp data.

What this script does
---------------------
1. Builds one common universe from locally cached 5m/15m/1h/1d/funding data.
2. Generates signals for several distinct intraday hypotheses:
   - BBS_1H_5M_PULLBACK: compression breakout -> 5m EMA21 pullback entry
   - VWAP_PULLBACK_15M: intraday VWAP/EMA pullback continuation with 1h trend filter
   - FUNDING_FADE_15M: funding-crowding exhaustion fade back toward VWAP
3. Simulates them in one consistent portfolio engine:
   - next-bar entry only
   - taker fees both sides
   - slippage both sides
   - actual funding history per symbol
   - capped concurrent positions
4. Saves a JSON report with metrics, monthly returns, and assumptions.

This is intentionally conservative. It is meant for research ranking, not
for claiming future monthly profitability.
"""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multi_agent.agents.bb_squeeze_agent import (  # noqa: E402
    compute_squeeze_indicators,
    detect_squeeze_signal,
)


CACHE_DIR = Path(__file__).parent / "data_cache"
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START_BALANCE = 1_000.0
RISK_PER_TRADE = 0.01
MAX_POSITIONS = 4
NOTIONAL_CAP_FRAC = 1.50
TAKER_FEE = 0.00055
SLIPPAGE = 0.00035
SYMBOL_COOLDOWN_MS = 4 * 60 * 60 * 1000

BBS_FUNDING_BLOCK_LONG = 0.0005
BBS_FUNDING_BLOCK_SHORT = -0.0005


@dataclass
class Signal:
    symbol: str
    strategy: str
    direction: str
    signal_ts: int
    next_open_ts: int
    sl: float
    tp: float
    max_hold_bars: int
    note: dict


def safe_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def discover_common_universe() -> list[str]:
    groups: dict[str, set[str]] = defaultdict(set)
    for path in CACHE_DIR.glob("*.parquet"):
        name = path.stem
        for suffix in ("_5m_730d", "_15m_730d", "_1h_730d", "_1d_730d", "_funding_730d"):
            if name.endswith(suffix):
                groups[suffix].add(name[: -len(suffix)])
                break
    required = ["_5m_730d", "_15m_730d", "_1h_730d", "_1d_730d", "_funding_730d"]
    common = set.intersection(*(groups[s] for s in required))
    return sorted(common)


def load_df(base: str, suffix: str) -> pd.DataFrame | None:
    path = CACHE_DIR / f"{base}{suffix}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if df is None or df.empty:
        return None
    for col in ("open", "high", "low", "close", "volume", "funding_rate"):
        if col in df.columns:
            df[col] = df[col].astype(float)
    if "timestamp" in df.columns:
        df["timestamp"] = df["timestamp"].astype(np.int64)
    return df.reset_index(drop=True)


def ema(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < period:
        return out
    vals = pd.Series(arr, dtype=float).ewm(span=period, adjust=False).mean().to_numpy()
    out[period - 1 :] = vals[period - 1 :]
    return out


def sma(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < period:
        return out
    vals = pd.Series(arr, dtype=float).rolling(period).mean().to_numpy()
    out[period - 1 :] = vals[period - 1 :]
    return out


def stdev(arr: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(arr), np.nan, dtype=float)
    if len(arr) < period:
        return out
    vals = pd.Series(arr, dtype=float).rolling(period).std(ddof=0).to_numpy()
    out[period - 1 :] = vals[period - 1 :]
    return out


def atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(closes), np.nan, dtype=float)
    if len(closes) < period:
        return out
    tr = np.empty(len(closes), dtype=float)
    tr[0] = highs[0] - lows[0]
    prev_close = closes[:-1]
    tr[1:] = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(np.abs(highs[1:] - prev_close), np.abs(lows[1:] - prev_close)),
    )
    vals = pd.Series(tr).ewm(com=period - 1, adjust=False).mean().to_numpy()
    out[period - 1 :] = vals[period - 1 :]
    return out


def rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    out = np.full(len(closes), np.nan, dtype=float)
    if len(closes) < period + 1:
        return out
    diff = np.diff(closes, prepend=closes[0])
    up = np.where(diff > 0, diff, 0.0)
    down = np.where(diff < 0, -diff, 0.0)
    avg_up = pd.Series(up).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    avg_down = pd.Series(down).ewm(alpha=1 / period, adjust=False).mean().to_numpy()
    rs = np.divide(avg_up, avg_down, out=np.zeros_like(avg_up), where=avg_down > 0)
    vals = 100 - (100 / (1 + rs))
    out[period:] = vals[period:]
    return out


def session_vwap(df: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    tp = (df["high"].to_numpy() + df["low"].to_numpy() + df["close"].to_numpy()) / 3.0
    vol = df["volume"].to_numpy()
    day = ts.dt.floor("D")
    out = np.full(len(df), np.nan, dtype=float)
    pv = pd.Series(tp * vol, index=day).groupby(level=0).cumsum().to_numpy()
    cv = pd.Series(vol, index=day).groupby(level=0).cumsum().to_numpy()
    mask = cv > 0
    out[mask] = pv[mask] / cv[mask]
    return out


def align_hourly_trend(ltf_ts: np.ndarray, df_1h: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    ts1 = df_1h["timestamp"].to_numpy()
    close1 = df_1h["close"].to_numpy(dtype=float)
    ema50 = ema(close1, 50)
    ema200 = ema(close1, 200)
    trend = np.zeros(len(ts1), dtype=int)
    long_mask = (close1 > ema50) & (ema50 > ema200)
    short_mask = (close1 < ema50) & (ema50 < ema200)
    trend[long_mask] = 1
    trend[short_mask] = -1
    idx = np.searchsorted(ts1, ltf_ts, side="right") - 1
    mapped_trend = np.zeros(len(ltf_ts), dtype=int)
    mapped_ret8 = np.full(len(ltf_ts), np.nan, dtype=float)
    valid = idx >= 0
    mapped_trend[valid] = trend[idx[valid]]
    ret8 = np.full(len(ts1), np.nan, dtype=float)
    for i in range(8, len(ts1)):
        if close1[i - 8] > 0:
            ret8[i] = close1[i] / close1[i - 8] - 1.0
    mapped_ret8[valid] = ret8[idx[valid]]
    return mapped_trend, mapped_ret8


def align_daily_dir(ltf_ts: np.ndarray, df_1d: pd.DataFrame) -> np.ndarray:
    tsd = df_1d["timestamp"].to_numpy()
    closed = df_1d["close"].to_numpy(dtype=float)
    ema200 = ema(closed, 200)
    daily_dir = np.zeros(len(tsd), dtype=int)
    daily_dir[(closed > ema200)] = 1
    daily_dir[(closed < ema200)] = -1
    idx = np.searchsorted(tsd, ltf_ts, side="right") - 1
    mapped = np.zeros(len(ltf_ts), dtype=int)
    valid = idx >= 0
    mapped[valid] = daily_dir[idx[valid]]
    return mapped


def funding_features(df_f: pd.DataFrame, bar_ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    fts = df_f["timestamp"].to_numpy(dtype=np.int64)
    fr = df_f["funding_rate"].to_numpy(dtype=float)
    roll_mean = pd.Series(fr).rolling(45, min_periods=12).mean().to_numpy()
    roll_std = pd.Series(fr).rolling(45, min_periods=12).std(ddof=0).to_numpy()
    z = np.divide(fr - roll_mean, roll_std, out=np.full_like(fr, np.nan), where=roll_std > 0)
    idx = np.searchsorted(fts, bar_ts, side="right") - 1
    rate_aligned = np.full(len(bar_ts), np.nan, dtype=float)
    z_aligned = np.full(len(bar_ts), np.nan, dtype=float)
    valid = idx >= 0
    rate_aligned[valid] = fr[idx[valid]]
    z_aligned[valid] = z[idx[valid]]
    return rate_aligned, z_aligned


def funding_pnl(df_f: pd.DataFrame, entry_ts: int, exit_ts: int, notional: float, direction: str) -> float:
    if notional <= 0 or exit_ts <= entry_ts:
        return 0.0
    f = df_f
    mask = (f["timestamp"] > entry_ts) & (f["timestamp"] <= exit_ts)
    if not mask.any():
        return 0.0
    signed = -1.0 if direction == "LONG" else 1.0
    return float((f.loc[mask, "funding_rate"] * notional * signed).sum())


def gen_bbs_pullback(symbol: str, df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_1d: pd.DataFrame, df_f: pd.DataFrame) -> list[Signal]:
    signals: list[Signal] = []
    if min(len(df_5m), len(df_1h), len(df_1d)) < 300:
        return signals

    ts5 = df_5m["timestamp"].to_numpy(dtype=np.int64)
    op5 = df_5m["open"].to_numpy(dtype=float)
    hi5 = df_5m["high"].to_numpy(dtype=float)
    lo5 = df_5m["low"].to_numpy(dtype=float)
    cl5 = df_5m["close"].to_numpy(dtype=float)
    ema21_5 = ema(cl5, 21)
    atr14_5 = atr(hi5, lo5, cl5, 14)

    ts1 = df_1h["timestamp"].to_numpy(dtype=np.int64)
    daily_dir = align_daily_dir(ts1, df_1d)
    funding_1h, _ = funding_features(df_f, ts1 + 60 * 60 * 1000)
    ind = compute_squeeze_indicators(
        df_1h,
        bb_len=20,
        bb_mult=2.0,
        kc_len=20,
        kc_mult=1.5,
        mom_len=20,
        adx_period=14,
    )

    for i in range(220, len(df_1h) - 1):
        sig = detect_squeeze_signal(ind, bar_idx=i, min_squeeze_bars=8, adx_threshold=15.0, vol_mult=1.0)
        if sig is None:
            continue
        direction = sig["direction"]
        if daily_dir[i] != 0 and ((direction == "LONG" and daily_dir[i] < 0) or (direction == "SHORT" and daily_dir[i] > 0)):
            continue
        fr = funding_1h[i]
        if not np.isnan(fr):
            if direction == "LONG" and fr > BBS_FUNDING_BLOCK_LONG:
                continue
            if direction == "SHORT" and fr < BBS_FUNDING_BLOCK_SHORT:
                continue

        signal_end_ts = int(ts1[i] + 60 * 60 * 1000)
        start_idx = int(np.searchsorted(ts5, signal_end_ts, side="left"))
        max_wait = 20

        for k in range(max(start_idx + 1, 22), min(start_idx + max_wait, len(ts5) - 2)):
            if np.isnan(ema21_5[k]) or np.isnan(atr14_5[k]):
                continue
            touched = False
            if direction == "LONG":
                touched = lo5[k] <= ema21_5[k] * 1.002 and cl5[k] >= ema21_5[k] * 0.998
            else:
                touched = hi5[k] >= ema21_5[k] * 0.998 and cl5[k] <= ema21_5[k] * 1.002
            if not touched:
                continue
            entry_open = float(op5[k + 1])
            if entry_open <= 0:
                continue
            current_ema = ema21_5[k + 1] if not np.isnan(ema21_5[k + 1]) else ema21_5[k]
            if direction == "LONG" and entry_open < current_ema * 0.993:
                continue
            if direction == "SHORT" and entry_open > current_ema * 1.007:
                continue

            sl_dist = float(np.clip(atr14_5[k] * 1.5, entry_open * 0.003, entry_open * 0.05))
            sl = entry_open - sl_dist if direction == "LONG" else entry_open + sl_dist
            risk = abs(entry_open - sl)
            tp = entry_open + risk * 3.0 if direction == "LONG" else entry_open - risk * 3.0
            signals.append(
                Signal(
                    symbol=symbol,
                    strategy="BBS_1H_5M_PULLBACK",
                    direction=direction,
                    signal_ts=int(signal_end_ts),
                    next_open_ts=int(ts5[k + 1]),
                    sl=float(sl),
                    tp=float(tp),
                    max_hold_bars=96,
                    note={
                        "squeeze_bars": int(sig["squeeze_bars"]),
                        "adx": float(sig["adx"]),
                        "funding_rate": None if np.isnan(fr) else float(fr),
                    },
                )
            )
            break
    return signals


def gen_vwap_pullback(symbol: str, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_f: pd.DataFrame) -> list[Signal]:
    signals: list[Signal] = []
    if min(len(df_15m), len(df_1h)) < 250:
        return signals

    ts = df_15m["timestamp"].to_numpy(dtype=np.int64)
    op = df_15m["open"].to_numpy(dtype=float)
    hi = df_15m["high"].to_numpy(dtype=float)
    lo = df_15m["low"].to_numpy(dtype=float)
    cl = df_15m["close"].to_numpy(dtype=float)
    vol = df_15m["volume"].to_numpy(dtype=float)

    ema21 = ema(cl, 21)
    atr14 = atr(hi, lo, cl, 14)
    rsi14 = rsi(cl, 14)
    vwap = session_vwap(df_15m)
    vol_sma20 = sma(vol, 20)
    trend_1h, _ = align_hourly_trend(ts, df_1h)
    fr_rate, _ = funding_features(df_f, ts)

    for i in range(220, len(df_15m) - 2):
        if any(np.isnan(x) for x in (ema21[i], atr14[i], rsi14[i], vwap[i], vol_sma20[i])):
            continue
        direction: str | None = None

        if trend_1h[i] > 0:
            above = cl[i] > vwap[i] and cl[i] > ema21[i]
            recent_pullback = any(lo[j] <= min(ema21[j], vwap[j]) * 1.002 for j in range(max(1, i - 4), i))
            if above and recent_pullback and cl[i] > cl[i - 1] and 52 <= rsi14[i] <= 72 and vol[i] >= vol_sma20[i] * 0.9:
                if np.isnan(fr_rate[i]) or fr_rate[i] <= 0.0007:
                    direction = "LONG"
        elif trend_1h[i] < 0:
            below = cl[i] < vwap[i] and cl[i] < ema21[i]
            recent_pullback = any(hi[j] >= max(ema21[j], vwap[j]) * 0.998 for j in range(max(1, i - 4), i))
            if below and cl[i] < cl[i - 1] and recent_pullback and 28 <= rsi14[i] <= 48 and vol[i] >= vol_sma20[i] * 0.9:
                if np.isnan(fr_rate[i]) or fr_rate[i] >= -0.0007:
                    direction = "SHORT"

        if direction is None:
            continue

        entry = float(op[i + 1])
        if entry <= 0:
            continue
        if direction == "LONG":
            base_sl = min(np.min(lo[max(0, i - 4) : i + 1]), entry - atr14[i] * 1.15)
        else:
            base_sl = max(np.max(hi[max(0, i - 4) : i + 1]), entry + atr14[i] * 1.15)

        dist = abs(entry - base_sl)
        dist = float(np.clip(dist, entry * 0.0035, entry * 0.03))
        sl = entry - dist if direction == "LONG" else entry + dist
        tp = entry + dist * 2.0 if direction == "LONG" else entry - dist * 2.0

        signals.append(
            Signal(
                symbol=symbol,
                strategy="VWAP_PULLBACK_15M",
                direction=direction,
                signal_ts=int(ts[i]),
                next_open_ts=int(ts[i + 1]),
                sl=float(sl),
                tp=float(tp),
                max_hold_bars=48,
                note={
                    "rsi": float(rsi14[i]),
                    "trend_1h": int(trend_1h[i]),
                    "funding_rate": None if np.isnan(fr_rate[i]) else float(fr_rate[i]),
                },
            )
        )
    return signals


def gen_funding_fade(symbol: str, df_15m: pd.DataFrame, df_1h: pd.DataFrame, df_f: pd.DataFrame) -> list[Signal]:
    signals: list[Signal] = []
    if min(len(df_15m), len(df_1h), len(df_f)) < 120:
        return signals

    ts = df_15m["timestamp"].to_numpy(dtype=np.int64)
    op = df_15m["open"].to_numpy(dtype=float)
    hi = df_15m["high"].to_numpy(dtype=float)
    lo = df_15m["low"].to_numpy(dtype=float)
    cl = df_15m["close"].to_numpy(dtype=float)

    atr14 = atr(hi, lo, cl, 14)
    rsi14 = rsi(cl, 14)
    vwap = session_vwap(df_15m)
    fr_rate, fr_z = funding_features(df_f, ts)
    _, ret8_1h = align_hourly_trend(ts, df_1h)

    for i in range(120, len(df_15m) - 2):
        if any(np.isnan(x) for x in (atr14[i], rsi14[i], vwap[i], fr_rate[i], fr_z[i], ret8_1h[i])):
            continue
        direction: str | None = None
        vwap_dev = (cl[i] / vwap[i] - 1.0) if vwap[i] > 0 else 0.0

        if (
            fr_rate[i] > 0
            and fr_z[i] >= 1.5
            and vwap_dev >= 0.008
            and ret8_1h[i] >= 0.02
            and rsi14[i - 1] > 66
            and rsi14[i] <= 66
            and cl[i] < cl[i - 1]
        ):
            direction = "SHORT"
        elif (
            fr_rate[i] < 0
            and fr_z[i] <= -1.5
            and vwap_dev <= -0.008
            and ret8_1h[i] <= -0.02
            and rsi14[i - 1] < 34
            and rsi14[i] >= 34
            and cl[i] > cl[i - 1]
        ):
            direction = "LONG"

        if direction is None:
            continue

        entry = float(op[i + 1])
        if entry <= 0:
            continue
        if direction == "LONG":
            local_extreme = float(np.min(lo[max(0, i - 6) : i + 1]))
            sl = min(entry - atr14[i] * 1.2, local_extreme - atr14[i] * 0.35)
            risk = entry - sl
            target_vwap = vwap[i]
            target_rr = entry + risk * 1.8
            tp = min(target_vwap, target_rr) if target_vwap > entry else target_rr
            if tp <= entry:
                tp = target_rr
        else:
            local_extreme = float(np.max(hi[max(0, i - 6) : i + 1]))
            sl = max(entry + atr14[i] * 1.2, local_extreme + atr14[i] * 0.35)
            risk = sl - entry
            target_vwap = vwap[i]
            target_rr = entry - risk * 1.8
            tp = max(target_vwap, target_rr) if target_vwap < entry else target_rr
            if tp >= entry:
                tp = target_rr

        risk = abs(entry - sl)
        if risk <= 0:
            continue
        rr = abs(tp - entry) / risk
        if rr < 1.15:
            continue

        signals.append(
            Signal(
                symbol=symbol,
                strategy="FUNDING_FADE_15M",
                direction=direction,
                signal_ts=int(ts[i]),
                next_open_ts=int(ts[i + 1]),
                sl=float(sl),
                tp=float(tp),
                max_hold_bars=32,
                note={
                    "funding_rate": float(fr_rate[i]),
                    "funding_z": float(fr_z[i]),
                    "ret8_1h": float(ret8_1h[i]),
                    "vwap_dev_pct": float(vwap_dev * 100.0),
                },
            )
        )
    return signals


def generate_signals_for_symbol(symbol: str) -> tuple[dict[str, list[Signal]], dict[str, pd.DataFrame], pd.DataFrame]:
    base = safe_name(symbol)
    df_5m = load_df(base, "_5m_730d")
    df_15m = load_df(base, "_15m_730d")
    df_1h = load_df(base, "_1h_730d")
    df_1d = load_df(base, "_1d_730d")
    df_f = load_df(base, "_funding_730d")
    if any(df is None for df in (df_5m, df_15m, df_1h, df_1d, df_f)):
        return {}, {}, pd.DataFrame()

    sigs = {
        "BBS_1H_5M_PULLBACK": gen_bbs_pullback(symbol, df_5m, df_1h, df_1d, df_f),
        "VWAP_PULLBACK_15M": gen_vwap_pullback(symbol, df_15m, df_1h, df_f),
        "FUNDING_FADE_15M": gen_funding_fade(symbol, df_15m, df_1h, df_f),
    }
    sim_dfs = {
        "BBS_1H_5M_PULLBACK": df_5m,
        "VWAP_PULLBACK_15M": df_15m,
        "FUNDING_FADE_15M": df_15m,
    }
    return sigs, sim_dfs, df_f


def summarize_monthly(eq_points: list[tuple[int, float]], start_balance: float, start_ts: int, end_ts: int) -> list[dict]:
    idx = pd.to_datetime([ts for ts, _ in eq_points], unit="ms", utc=True)
    ser = pd.Series([bal for _, bal in eq_points], index=idx, dtype=float)
    ser = ser.groupby(level=0).last().sort_index()
    start_idx = pd.to_datetime(start_ts, unit="ms", utc=True)
    end_idx = pd.to_datetime(end_ts, unit="ms", utc=True)
    full_idx = pd.date_range(start=start_idx, end=end_idx, freq="D", tz="UTC")
    daily = ser.reindex(ser.index.union(full_idx)).sort_index().ffill()
    daily = daily.reindex(full_idx, method="ffill").fillna(start_balance)
    monthly_end = daily.resample("ME").last()
    prev = monthly_end.shift(1).fillna(start_balance)
    monthly = []
    for ts, end_bal in monthly_end.items():
        start_bal = float(prev.loc[ts])
        ret_pct = (end_bal / start_bal - 1.0) * 100.0 if start_bal > 0 else 0.0
        monthly.append(
            {
                "month": ts.strftime("%Y-%m"),
                "start": round(start_bal, 2),
                "end": round(float(end_bal), 2),
                "ret_pct": round(float(ret_pct), 2),
            }
        )
    return monthly


def simulate_strategy(
    strategy_name: str,
    all_signals: list[Signal],
    sim_dfs: dict[tuple[str, str], pd.DataFrame],
    funding_map: dict[str, pd.DataFrame],
) -> dict:
    sigs = sorted(all_signals, key=lambda s: s.next_open_ts)
    balance = START_BALANCE
    open_positions: dict[str, dict] = {}
    cooldown_until: dict[str, int] = {}
    trades: list[dict] = []
    eq_points: list[tuple[int, float]] = []

    if not sigs:
        return {
            "strategy": strategy_name,
            "metrics": {
                "trades": 0,
                "wr_pct": 0.0,
                "profit_factor": 0.0,
                "return_pct": 0.0,
                "max_dd_pct": 0.0,
                "final_balance": START_BALANCE,
                "avg_monthly_pct": 0.0,
                "profitable_months": 0,
                "months": 0,
            },
            "monthly_returns": [],
            "trades_sample": [],
        }

    start_ts = min(s.next_open_ts for s in sigs)
    last_df_end = []
    for df in sim_dfs.values():
        if df is not None and not df.empty:
            last_df_end.append(int(df["timestamp"].iloc[-1]))
    end_ts = max(last_df_end) if last_df_end else max(s.next_open_ts for s in sigs)
    eq_points.append((start_ts, balance))

    def realize_due(until_ts: int) -> None:
        nonlocal balance
        due = [sym for sym, pos in open_positions.items() if pos["exit_ts"] <= until_ts]
        for sym in sorted(due, key=lambda x: open_positions[x]["exit_ts"]):
            pos = open_positions.pop(sym)
            balance += pos["pnl_usd"]
            trades.append(pos)
            eq_points.append((pos["exit_ts"], balance))
            if pos["reason"] == "SL":
                cooldown_until[pos["symbol"]] = pos["exit_ts"] + SYMBOL_COOLDOWN_MS

    for sig in sigs:
        realize_due(sig.next_open_ts)
        if sig.symbol in open_positions:
            continue
        if cooldown_until.get(sig.symbol, 0) > sig.next_open_ts:
            continue
        if len(open_positions) >= MAX_POSITIONS:
            continue

        df = sim_dfs.get((sig.symbol, sig.strategy))
        if df is None or df.empty:
            continue
        ts = df["timestamp"].to_numpy(dtype=np.int64)
        op = df["open"].to_numpy(dtype=float)
        hi = df["high"].to_numpy(dtype=float)
        lo = df["low"].to_numpy(dtype=float)
        cl = df["close"].to_numpy(dtype=float)
        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= len(ts) - 1:
            continue

        if sig.direction == "LONG":
            entry = op[j] * (1.0 + SLIPPAGE)
        else:
            entry = op[j] * (1.0 - SLIPPAGE)
        if entry <= 0:
            continue

        dist = abs(entry - sig.sl)
        if dist <= 0:
            continue

        risk_usd = balance * RISK_PER_TRADE
        qty = risk_usd / dist
        max_notional = balance * NOTIONAL_CAP_FRAC
        if entry * qty > max_notional:
            qty = max_notional / entry
            risk_usd = qty * dist
        if qty <= 0 or risk_usd <= 0:
            continue
        notional = entry * qty

        exit_price = None
        exit_ts = None
        reason = "timeout"
        last_bar = min(j + sig.max_hold_bars, len(ts) - 1)
        for k in range(j, last_bar + 1):
            if sig.direction == "LONG":
                if lo[k] <= sig.sl:
                    exit_price = sig.sl
                    exit_ts = int(ts[k])
                    reason = "SL"
                    break
                if hi[k] >= sig.tp:
                    exit_price = sig.tp
                    exit_ts = int(ts[k])
                    reason = "TP"
                    break
            else:
                if hi[k] >= sig.sl:
                    exit_price = sig.sl
                    exit_ts = int(ts[k])
                    reason = "SL"
                    break
                if lo[k] <= sig.tp:
                    exit_price = sig.tp
                    exit_ts = int(ts[k])
                    reason = "TP"
                    break
        if exit_price is None or exit_ts is None:
            exit_ts = int(ts[last_bar])
            exit_price = float(cl[last_bar])
            reason = "timeout"

        if sig.direction == "LONG":
            slip_exit = exit_price * (1.0 - SLIPPAGE)
            gross = (slip_exit - entry) * qty
        else:
            slip_exit = exit_price * (1.0 + SLIPPAGE)
            gross = (entry - slip_exit) * qty
        fees = entry * qty * TAKER_FEE + slip_exit * qty * TAKER_FEE
        funding = funding_pnl(funding_map[sig.symbol], int(ts[j]), exit_ts, notional, sig.direction)
        pnl = gross - fees + funding

        open_positions[sig.symbol] = {
            "symbol": sig.symbol,
            "strategy": sig.strategy,
            "direction": sig.direction,
            "entry_ts": int(ts[j]),
            "exit_ts": exit_ts,
            "entry": float(entry),
            "exit": float(slip_exit),
            "qty": float(qty),
            "notional": float(notional),
            "risk_usd": float(risk_usd),
            "pnl_usd": float(pnl),
            "reason": reason,
            "funding_pnl": float(funding),
            "note": sig.note,
        }

    realize_due(end_ts + 1)

    gross_wins = sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0)
    gross_losses = -sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0)
    win_count = sum(1 for t in trades if t["pnl_usd"] > 0)
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf") if gross_wins > 0 else 0.0

    peak = START_BALANCE
    max_dd_pct = 0.0
    for _, bal in sorted(eq_points, key=lambda x: x[0]):
        peak = max(peak, bal)
        if peak > 0:
            dd = (peak - bal) / peak * 100.0
            max_dd_pct = max(max_dd_pct, dd)

    monthly = summarize_monthly(eq_points, START_BALANCE, start_ts, end_ts)
    profitable_months = sum(1 for m in monthly if m["ret_pct"] > 0)
    avg_monthly = float(np.mean([m["ret_pct"] for m in monthly])) if monthly else 0.0

    metrics = {
        "trades": len(trades),
        "wr_pct": round((win_count / len(trades) * 100.0) if trades else 0.0, 2),
        "profit_factor": round(float(pf), 3) if math.isfinite(pf) else None,
        "return_pct": round((balance / START_BALANCE - 1.0) * 100.0, 2),
        "max_dd_pct": round(max_dd_pct, 2),
        "final_balance": round(balance, 2),
        "avg_monthly_pct": round(avg_monthly, 2),
        "profitable_months": profitable_months,
        "months": len(monthly),
    }

    trades_sample = sorted(trades, key=lambda t: abs(t["pnl_usd"]), reverse=True)[:20]
    return {
        "strategy": strategy_name,
        "metrics": metrics,
        "monthly_returns": monthly,
        "trades_sample": trades_sample,
    }


def main() -> None:
    universe = discover_common_universe()
    if not universe:
        raise RuntimeError(f"No common 2-year universe found in {CACHE_DIR}")

    all_signals: dict[str, list[Signal]] = defaultdict(list)
    sim_dfs: dict[tuple[str, str], pd.DataFrame] = {}
    funding_map: dict[str, pd.DataFrame] = {}

    for base in universe:
        symbol = base.replace("_USDT_USDT", "/USDT:USDT")
        sigs_by_strategy, symbol_sim_dfs, df_f = generate_signals_for_symbol(symbol)
        if df_f.empty:
            continue
        funding_map[symbol] = df_f
        for strat, sigs in sigs_by_strategy.items():
            all_signals[strat].extend(sigs)
            sim_dfs[(symbol, strat)] = symbol_sim_dfs[strat]

    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "assumptions": {
            "start_balance": START_BALANCE,
            "risk_per_trade": RISK_PER_TRADE,
            "max_positions": MAX_POSITIONS,
            "notional_cap_frac": NOTIONAL_CAP_FRAC,
            "taker_fee": TAKER_FEE,
            "slippage_each_side": SLIPPAGE,
            "symbol_cooldown_ms": SYMBOL_COOLDOWN_MS,
            "common_universe_count": len(universe),
            "common_universe": universe,
            "note": "All strategies use next-bar entries and actual cached funding history. No orderbook history is used.",
        },
        "results": [],
    }

    for strat in ("BBS_1H_5M_PULLBACK", "VWAP_PULLBACK_15M", "FUNDING_FADE_15M"):
        result = simulate_strategy(strat, all_signals[strat], sim_dfs, funding_map)
        result["signal_count"] = len(all_signals[strat])
        report["results"].append(result)

    report["results"].sort(key=lambda x: (x["metrics"]["profit_factor"] or 0.0), reverse=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"intraday_edges_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({"report_path": str(out_path), "strategies": report["results"]}, indent=2))


if __name__ == "__main__":
    main()
