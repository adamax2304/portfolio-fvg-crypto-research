"""
backtest_session_v3.py -- Session V3 (2026-04-16)

Builds on V2 (NY-ARB found PF=4.01, +630% PnL, DD=6.92%).

NEW in V3:
  1. Maximally realistic execution:
     - Entry at NEXT BAR OPEN after signal (no same-bar TP/SL look-ahead)
     - Explicit slippage (0.05% per side) + taker fee (0.055%)
     - Intra-bar SL/TP priority: if both in same bar, assume SL first (conservative)
     - Session close forces exit at close of session-end bar
  2. Top-20 market-cap expansion (20 symbols)
  3. Grid entry (scale-in) with 2-3 legs
  4. Asian-session strategies (4 variants: pre-LO breakout, Tokyo-range breakout,
     Asian-ARB mean-reversion, London-open Asian-range breakout)

Run: py -3 multi_agent/backtests/backtest_session_v3.py
"""
from __future__ import annotations
import json, logging, sys, time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("session_bt_v3")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

# Top-20 market-cap (approx, April 2026), excluding stablecoins
TOP20 = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "TON/USDT:USDT", "ADA/USDT:USDT",
    "TRX/USDT:USDT", "AVAX/USDT:USDT", "LINK/USDT:USDT", "DOT/USDT:USDT",
    "BCH/USDT:USDT", "NEAR/USDT:USDT", "UNI/USDT:USDT", "ICP/USDT:USDT",
    "APT/USDT:USDT", "XLM/USDT:USDT", "ATOM/USDT:USDT", "HBAR/USDT:USDT",
]

START_BALANCE = 1000.0
TAKER_FEE = 0.00055
SLIPPAGE  = 0.0005
MAX_LEVERAGE = 50
RISK_PER_TRADE = 0.01
MIN_NOTIONAL = 5.0  # Bybit dust limit


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def _load(symbol, tf, days_tag="365d"):
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in (days_tag, "180d", "90d"):
        for ext in (".parquet", ".csv"):
            fp = CACHE_DIR / f"{safe}_{tf}_{tag}{ext}"
            if fp.exists():
                try:
                    df = pd.read_parquet(fp) if ext == ".parquet" else pd.read_csv(fp)
                    for c in ("open", "high", "low", "close", "volume"):
                        if c in df.columns:
                            df[c] = df[c].astype(float)
                    if "timestamp" in df.columns:
                        df["timestamp"] = df["timestamp"].astype(np.int64)
                    return df
                except Exception:
                    continue
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

def _atr(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def _ema(x, period):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n)
    if n == 0:
        return out
    k = 2 / (period + 1)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out


def _supertrend(hi, lo, cl, period=10, mult=3.0):
    atr = _atr(hi, lo, cl, period)
    n = len(cl)
    med = (hi + lo) / 2.0
    upper = med + mult * atr
    lower = med - mult * atr
    fub = np.copy(upper)
    flb = np.copy(lower)
    for i in range(1, n):
        if upper[i] < fub[i-1] or cl[i-1] > fub[i-1]:
            fub[i] = upper[i]
        else:
            fub[i] = fub[i-1]
        if lower[i] > flb[i-1] or cl[i-1] < flb[i-1]:
            flb[i] = lower[i]
        else:
            flb[i] = flb[i-1]
    st = np.zeros(n, dtype=np.int32)
    st[0] = 1
    for i in range(1, n):
        if st[i-1] == 1 and cl[i] < flb[i]:
            st[i] = -1
        elif st[i-1] == -1 and cl[i] > fub[i]:
            st[i] = 1
        else:
            st[i] = st[i-1]
    return st


def _adx(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = hi[i] - hi[i-1]
        dn = lo[i-1] - lo[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr_w = np.zeros(n)
    plus_s = np.zeros(n)
    minus_s = np.zeros(n)
    if n <= period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    atr_w[period] = np.sum(tr[1:period+1])
    plus_s[period] = np.sum(plus_dm[1:period+1])
    minus_s[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        atr_w[i] = atr_w[i-1] - atr_w[i-1]/period + tr[i]
        plus_s[i] = plus_s[i-1] - plus_s[i-1]/period + plus_dm[i]
        minus_s[i] = minus_s[i-1] - minus_s[i-1]/period + minus_dm[i]
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_w[i] > 0:
            plus_di[i] = 100 * plus_s[i] / atr_w[i]
            minus_di[i] = 100 * minus_s[i] / atr_w[i]
    dx = np.zeros(n)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s
    adx = np.zeros(n)
    if n >= 2*period:
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx, plus_di, minus_di


# ═══════════════════════════════════════════════════════════════════════════
#  PREP & SESSION RANGES
# ═══════════════════════════════════════════════════════════════════════════

def prep_15m_arrays(df):
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    ts = df["timestamp"].values.astype(np.int64)
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    hours = dt.hour.values.astype(np.int32)
    dates = np.array(dt.strftime("%Y-%m-%d"))
    atr = _atr(h, l, c, 14)
    volma = pd.Series(v).rolling(20, min_periods=1).mean().values
    return {
        "o": o, "h": h, "l": l, "c": c, "v": v, "ts": ts,
        "hour": hours, "date": dates,
        "atr": atr, "volma": volma, "n": len(c),
    }


def build_session_ranges(arr):
    """Per-date dict: Tokyo (00-04), PreLO (00-07), London (07-13), NY (13-22) + PDH/PDL."""
    n = arr["n"]
    h, l = arr["h"], arr["l"]
    date = arr["date"]
    hour = arr["hour"]
    by_date = defaultdict(list)
    for i in range(n):
        by_date[date[i]].append(i)
    ranges = {}
    dates_sorted = sorted(by_date.keys())
    prev_hi = None
    prev_lo = None
    prev_close = None
    for d in dates_sorted:
        idxs = by_date[d]
        t_i  = [i for i in idxs if 0 <= hour[i] < 4]    # Tokyo 00-04
        a_i  = [i for i in idxs if 0 <= hour[i] < 7]    # Asian 00-07 (for NY strat)
        l_i  = [i for i in idxs if 7 <= hour[i] < 13]
        n_i  = [i for i in idxs if 13 <= hour[i] < 22]
        thi = max((h[i] for i in t_i), default=np.nan)
        tlo = min((l[i] for i in t_i), default=np.nan)
        ahi = max((h[i] for i in a_i), default=np.nan)
        alo = min((l[i] for i in a_i), default=np.nan)
        lhi = max((h[i] for i in l_i), default=np.nan)
        llo = min((l[i] for i in l_i), default=np.nan)
        nhi = max((h[i] for i in n_i), default=np.nan)
        nlo = min((l[i] for i in n_i), default=np.nan)
        ranges[d] = {
            "tok_hi": thi, "tok_lo": tlo,
            "asian_hi": ahi, "asian_lo": alo,
            "london_hi": lhi, "london_lo": llo,
            "ny_hi": nhi, "ny_lo": nlo,
            "pd_hi": prev_hi, "pd_lo": prev_lo, "pd_close": prev_close,
        }
        day_hi = max((h[i] for i in idxs), default=prev_hi)
        day_lo = min((l[i] for i in idxs), default=prev_lo)
        day_cl = (arr["c"][idxs[-1]] if idxs else prev_close)
        prev_hi = day_hi
        prev_lo = day_lo
        prev_close = day_cl
    return ranges


def compute_htf_trends(df_4h, df_1h):
    out = {}
    if df_4h is not None and len(df_4h) > 60:
        cl = df_4h["close"].values.astype(float)
        hi = df_4h["high"].values.astype(float)
        lo = df_4h["low"].values.astype(float)
        ts = df_4h["timestamp"].values.astype(np.int64)
        ema50 = _ema(cl, 50)
        st = _supertrend(hi, lo, cl, 10, 3.0)
        out["ema50_4h"] = {int(ts[i]): (1 if cl[i] > ema50[i] else -1) for i in range(50, len(cl))}
        out["st_4h"] = {int(ts[i]): int(st[i]) for i in range(len(cl))}
    if df_1h is not None and len(df_1h) > 60:
        cl = df_1h["close"].values.astype(float)
        hi = df_1h["high"].values.astype(float)
        lo = df_1h["low"].values.astype(float)
        ts = df_1h["timestamp"].values.astype(np.int64)
        ema50 = _ema(cl, 50)
        st = _supertrend(hi, lo, cl, 10, 3.0)
        adx, pdi, mdi = _adx(hi, lo, cl, 14)
        out["ema50_1h"] = {int(ts[i]): (1 if cl[i] > ema50[i] else -1) for i in range(50, len(cl))}
        out["st_1h"] = {int(ts[i]): int(st[i]) for i in range(len(cl))}
        def adir(i):
            if adx[i] < 20: return 0
            return 1 if pdi[i] > mdi[i] else -1
        out["adx_1h"] = {int(ts[i]): adir(i) for i in range(30, len(cl))}
    return out


def align_htf(htf_map, ts_15m):
    if not htf_map:
        return np.zeros(len(ts_15m), dtype=np.int32)
    ks = np.array(sorted(htf_map.keys()), dtype=np.int64)
    vs = np.array([htf_map[int(k)] for k in ks], dtype=np.int32)
    idx = np.searchsorted(ks, ts_15m, side="right") - 1
    idx = np.clip(idx, 0, len(vs)-1)
    aligned = vs[idx]
    aligned[ts_15m < ks[0]] = 0
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL GEN — NY-ARB V3 (realistic next-bar entry)
# ═══════════════════════════════════════════════════════════════════════════

def gen_arb_v3_retest(arr, sess_ranges, params, htf_aligned_map=None):
    """
    ARB at NY open with LIMIT at retest of breakout level.
    Signal = bar i closes beyond ahi/alo. Entry = limit at ahi/alo.
    If price doesn't retest within N bars OR session ends, order expires.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr, volma, v = arr["atr"], arr["volma"], arr["v"]
    n = arr["n"]

    open_hr = params.get("open_hr", 13)
    close_hr = params.get("close_hr", 22)
    retest_bars = params.get("retest_bars", 8)  # how many bars to wait for retest
    sl_mode = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 2.5)
    tp_mode = params.get("tp_mode", "range_mult")
    tp_r = params.get("tp_r", 2.0)
    tp_range_mult = params.get("tp_range_mult", 2.0)
    htf_filter = params.get("htf_filter", None)
    max_entry_hr = params.get("max_entry_hr", 4)
    min_range_atr = params.get("min_range_atr", 0.3)
    max_range_atr = params.get("max_range_atr", 5.0)

    signals = []
    last_sig_date = None
    for i in range(50, n-retest_bars-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo): continue
        if hour[i] < open_hr or hour[i] >= close_hr: continue
        if hour[i] >= open_hr + max_entry_hr: continue
        if last_sig_date == d: continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0: continue
        aR = a_range / atr[i]
        if aR < min_range_atr or aR > max_range_atr: continue

        direction = None; target = None
        if h[i] > ahi and c[i] > ahi:
            direction = "LONG"; target = ahi
        elif l[i] < alo and c[i] < alo:
            direction = "SHORT"; target = alo
        if direction is None: continue

        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        # Wait for retest: price touches target within retest_bars
        filled_bar = None
        for k in range(1, retest_bars + 1):
            bi = i + k
            if bi >= n or date[bi] != d or hour[bi] >= close_hr: break
            if direction == "LONG" and l[bi] <= target and h[bi] >= target:
                filled_bar = bi; break
            if direction == "SHORT" and h[bi] >= target and l[bi] <= target:
                filled_bar = bi; break
        if filled_bar is None: continue

        entry_ref = target  # limit fill exactly at break level

        if sl_mode == "atr":
            sl = entry_ref - atr[i] * sl_atr_mult if direction == "LONG" else entry_ref + atr[i] * sl_atr_mult
        else:
            sl = entry_ref * 0.985 if direction == "LONG" else entry_ref * 1.015
        risk_px = abs(entry_ref - sl)
        if risk_px <= 0: continue

        if tp_mode == "fixed_r":
            tp = entry_ref + risk_px * tp_r if direction == "LONG" else entry_ref - risk_px * tp_r
        elif tp_mode == "range_mult":
            tp = entry_ref + a_range * tp_range_mult if direction == "LONG" else entry_ref - a_range * tp_range_mult
        elif tp_mode == "nearest_level":
            cand = []
            if direction == "LONG":
                if r["pd_hi"] and not np.isnan(r["pd_hi"]) and r["pd_hi"] > entry_ref + risk_px: cand.append(r["pd_hi"])
                proj = ahi + a_range
                if proj > entry_ref + risk_px: cand.append(proj)
                tp = min(cand) if cand else entry_ref + risk_px * 2.0
            else:
                if r["pd_lo"] and not np.isnan(r["pd_lo"]) and r["pd_lo"] < entry_ref - risk_px: cand.append(r["pd_lo"])
                proj = alo - a_range
                if proj < entry_ref - risk_px: cand.append(proj)
                tp = max(cand) if cand else entry_ref - risk_px * 2.0
        else:
            tp = entry_ref + risk_px * 2.0 if direction == "LONG" else entry_ref - risk_px * 2.0

        signals.append({
            "i_signal": i, "i_entry": filled_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "trigger": float(target),
            "date": d, "is_limit_retest": True,
        })
        last_sig_date = d
    return signals


def gen_arb_v3(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Asian Range Breakout at NY session.
    Signal fires on bar close if:
      - high[i] > ahi and close[i] > ahi (LONG) or corresponding SHORT
      - entry recorded at bar i+1 OPEN (next bar — realistic)
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr, volma, v = arr["atr"], arr["volma"], arr["v"]
    n = arr["n"]

    open_hr = params.get("open_hr", 13)
    close_hr = params.get("close_hr", 22)
    sl_mode = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 2.5)
    sl_pct = params.get("sl_pct", 0.02)
    tp_mode = params.get("tp_mode", "range_mult")
    tp_r = params.get("tp_r", 2.0)
    tp_range_mult = params.get("tp_range_mult", 2.0)
    htf_filter = params.get("htf_filter", None)
    max_entry_hr = params.get("max_entry_hr", 3)
    vol_mult = params.get("vol_mult", 0.0)
    min_range_atr = params.get("min_range_atr", 0.3)
    max_range_atr = params.get("max_range_atr", 5.0)

    signals = []
    last_sig_date = None

    for i in range(50, n-1):  # -1 so i+1 exists
        d = date[i]
        if d not in sess_ranges:
            continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo):
            continue
        if hour[i] < open_hr or hour[i] >= close_hr:
            continue
        if max_entry_hr is not None and hour[i] >= open_hr + max_entry_hr:
            continue
        if last_sig_date == d:
            continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0:
            continue
        a_range_atr = a_range / atr[i]
        if a_range_atr < min_range_atr or a_range_atr > max_range_atr:
            continue

        direction = None
        trigger = None
        if h[i] > ahi and c[i] > ahi:
            direction = "LONG"
            trigger = ahi
        elif l[i] < alo and c[i] < alo:
            direction = "SHORT"
            trigger = alo
        if direction is None:
            continue

        # HTF filter — evaluated at signal bar (end of bar i, BEFORE entry at i+1)
        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        if vol_mult > 0:
            ma = volma[i]
            if ma > 0 and v[i] < ma * vol_mult:
                continue

        # REALISTIC ENTRY: at bar i+1 open
        entry_bar = i + 1
        if entry_bar >= n:
            continue
        if date[entry_bar] != d:
            continue  # next bar already different session/day
        entry_ref = o[entry_bar]

        # SL computed from entry_ref using ATR at signal bar
        if sl_mode == "atr":
            sl = entry_ref - atr[i] * sl_atr_mult if direction == "LONG" else entry_ref + atr[i] * sl_atr_mult
        elif sl_mode == "pct":
            sl = entry_ref * (1 - sl_pct) if direction == "LONG" else entry_ref * (1 + sl_pct)
        else:
            sl = entry_ref - atr[i] * 2.5 if direction == "LONG" else entry_ref + atr[i] * 2.5

        risk_px = abs(entry_ref - sl)
        if risk_px <= 0:
            continue

        # TP
        if tp_mode == "fixed_r":
            tp = entry_ref + risk_px * tp_r if direction == "LONG" else entry_ref - risk_px * tp_r
        elif tp_mode == "range_mult":
            tp = entry_ref + a_range * tp_range_mult if direction == "LONG" else entry_ref - a_range * tp_range_mult
        elif tp_mode == "nearest_level":
            cand = []
            if direction == "LONG":
                if r["pd_hi"] and not np.isnan(r["pd_hi"]) and r["pd_hi"] > entry_ref + risk_px: cand.append(r["pd_hi"])
                if not np.isnan(r["london_hi"]) and r["london_hi"] > entry_ref + risk_px: cand.append(r["london_hi"])
                proj = ahi + a_range
                if proj > entry_ref + risk_px: cand.append(proj)
                tp = min(cand) if cand else entry_ref + risk_px * 2.0
            else:
                if r["pd_lo"] and not np.isnan(r["pd_lo"]) and r["pd_lo"] < entry_ref - risk_px: cand.append(r["pd_lo"])
                if not np.isnan(r["london_lo"]) and r["london_lo"] < entry_ref - risk_px: cand.append(r["london_lo"])
                proj = alo - a_range
                if proj < entry_ref - risk_px: cand.append(proj)
                tp = max(cand) if cand else entry_ref - risk_px * 2.0
        else:
            tp = entry_ref + risk_px * 2.0 if direction == "LONG" else entry_ref - risk_px * 2.0

        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "trigger": float(trigger),
            "date": d,
        })
        last_sig_date = d

    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  ASIAN SESSION STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def gen_asia_tokyobreak(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Tokyo-range breakout:
      range formed 00:00-04:00 UTC
      entry window 04:00-07:00 UTC (before London)
      break of Tokyo high/low → enter on next bar
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]

    sl_mode = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 2.0)
    sl_pct = params.get("sl_pct", 0.015)
    tp_mode = params.get("tp_mode", "range_mult")
    tp_r = params.get("tp_r", 2.0)
    tp_range_mult = params.get("tp_range_mult", 2.0)
    htf_filter = params.get("htf_filter", None)
    min_range_atr = params.get("min_range_atr", 0.3)
    max_range_atr = params.get("max_range_atr", 4.0)

    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        thi, tlo = r["tok_hi"], r["tok_lo"]
        if np.isnan(thi) or np.isnan(tlo): continue
        if hour[i] < 4 or hour[i] >= 7: continue
        if last_sig_date == d: continue
        a_range = thi - tlo
        if a_range <= 0 or atr[i] <= 0: continue
        aR = a_range / atr[i]
        if aR < min_range_atr or aR > max_range_atr: continue

        direction = None
        if h[i] > thi and c[i] > thi:
            direction = "LONG"
        elif l[i] < tlo and c[i] < tlo:
            direction = "SHORT"
        if direction is None: continue

        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]

        if sl_mode == "atr":
            sl = entry_ref - atr[i] * sl_atr_mult if direction == "LONG" else entry_ref + atr[i] * sl_atr_mult
        elif sl_mode == "pct":
            sl = entry_ref * (1 - sl_pct) if direction == "LONG" else entry_ref * (1 + sl_pct)

        risk_px = abs(entry_ref - sl)
        if risk_px <= 0: continue

        if tp_mode == "fixed_r":
            tp = entry_ref + risk_px * tp_r if direction == "LONG" else entry_ref - risk_px * tp_r
        elif tp_mode == "range_mult":
            tp = entry_ref + a_range * tp_range_mult if direction == "LONG" else entry_ref - a_range * tp_range_mult

        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_asia_londonopen(arr, sess_ranges, params, htf_aligned_map=None):
    """Asian range (00-07) breakout at LONDON open (07:00 UTC), window 07-13."""
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]

    sl_mode = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 2.5)
    tp_mode = params.get("tp_mode", "range_mult")
    tp_r = params.get("tp_r", 2.0)
    tp_range_mult = params.get("tp_range_mult", 2.0)
    htf_filter = params.get("htf_filter", None)
    max_entry_hr = params.get("max_entry_hr", 3)  # 07:00-09:59
    min_range_atr = params.get("min_range_atr", 0.3)
    max_range_atr = params.get("max_range_atr", 5.0)

    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo): continue
        if hour[i] < 7 or hour[i] >= 13: continue
        if hour[i] >= 7 + max_entry_hr: continue
        if last_sig_date == d: continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0: continue
        aR = a_range / atr[i]
        if aR < min_range_atr or aR > max_range_atr: continue

        direction = None
        if h[i] > ahi and c[i] > ahi:
            direction = "LONG"
        elif l[i] < alo and c[i] < alo:
            direction = "SHORT"
        if direction is None: continue

        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]

        if sl_mode == "atr":
            sl = entry_ref - atr[i] * sl_atr_mult if direction == "LONG" else entry_ref + atr[i] * sl_atr_mult
        else:
            sl = entry_ref * 0.985 if direction == "LONG" else entry_ref * 1.015
        risk_px = abs(entry_ref - sl)
        if risk_px <= 0: continue
        if tp_mode == "fixed_r":
            tp = entry_ref + risk_px * tp_r if direction == "LONG" else entry_ref - risk_px * tp_r
        else:
            tp = entry_ref + a_range * tp_range_mult if direction == "LONG" else entry_ref - a_range * tp_range_mult
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_asia_pdhbreak(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Pre-London breakout of PDH/PDL.
    Window: 00:00-07:00 UTC. Trigger: 15m close beyond PDH or PDL.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]

    sl_mode = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 2.0)
    tp_mode = params.get("tp_mode", "fixed_r")
    tp_r = params.get("tp_r", 2.5)
    tp_range_mult = params.get("tp_range_mult", 1.5)
    htf_filter = params.get("htf_filter", None)

    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        pdh, pdl = r["pd_hi"], r["pd_lo"]
        if pdh is None or pdl is None: continue
        if hour[i] < 0 or hour[i] >= 7: continue
        if last_sig_date == d: continue
        if atr[i] <= 0: continue

        direction = None
        if h[i] > pdh and c[i] > pdh:
            direction = "LONG"
        elif l[i] < pdl and c[i] < pdl:
            direction = "SHORT"
        if direction is None: continue

        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl = entry_ref - atr[i] * sl_atr_mult if direction == "LONG" else entry_ref + atr[i] * sl_atr_mult
        risk_px = abs(entry_ref - sl)
        if risk_px <= 0: continue
        pd_range = pdh - pdl if pdh and pdl else atr[i] * 2
        if tp_mode == "fixed_r":
            tp = entry_ref + risk_px * tp_r if direction == "LONG" else entry_ref - risk_px * tp_r
        else:
            tp = entry_ref + pd_range * tp_range_mult if direction == "LONG" else entry_ref - pd_range * tp_range_mult
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(pd_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_asia_tokyo_mr(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Tokyo mean-reversion: fade Tokyo-range extremes toward mid during Asian session.
    Entry window: 04:00-07:00 UTC, when price touches tok_hi or tok_lo.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]

    sl_atr_mult = params.get("sl_atr_mult", 1.0)
    tp_to = params.get("tp_to", "mid")  # "mid" or "opp"
    htf_filter = params.get("htf_filter", None)
    min_range_atr = params.get("min_range_atr", 0.8)
    max_range_atr = params.get("max_range_atr", 4.0)

    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        thi, tlo = r["tok_hi"], r["tok_lo"]
        if np.isnan(thi) or np.isnan(tlo): continue
        if hour[i] < 4 or hour[i] >= 7: continue
        if last_sig_date == d: continue
        a_range = thi - tlo
        if a_range <= 0 or atr[i] <= 0: continue
        aR = a_range / atr[i]
        if aR < min_range_atr or aR > max_range_atr: continue
        mid = (thi + tlo) / 2.0

        direction = None
        trigger = None
        # Fade: price touches tok_hi from below -> SHORT, touches tok_lo from above -> LONG
        if h[i] >= thi and c[i] < thi:  # rejected at top
            direction = "SHORT"
            trigger = thi
        elif l[i] <= tlo and c[i] > tlo:  # rejected at bottom
            direction = "LONG"
            trigger = tlo
        if direction is None: continue

        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl = entry_ref - atr[i] * sl_atr_mult if direction == "LONG" else entry_ref + atr[i] * sl_atr_mult
        risk_px = abs(entry_ref - sl)
        if risk_px <= 0: continue

        if tp_to == "mid":
            tp = mid
        else:
            tp = tlo if direction == "SHORT" else thi  # opposite side
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
        })
        last_sig_date = d
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATE (V3 — realistic)
# ═══════════════════════════════════════════════════════════════════════════

def simulate(sigs_by_sym, arrays_by_sym, session_end_hr=22, trail_cfg=None,
              grid_cfg=None):
    """
    Realistic execution:
      - Entry at bar i_entry OPEN (+ slippage)
      - Exit checks from bar i_entry onwards
      - Conservative: if SL and TP both hit same bar, assume SL first (worst case)
      - Force close at session_end_hr

    grid_cfg: {"legs": [(weight, pullback_atr), ...], "timeout_bars": 12}
      Each leg fires independently if price pulls back to leg level.
      Untilled legs cancelled at timeout or exit.
    """
    balance = START_BALANCE
    completed = []
    peak = balance
    max_dd = 0.0

    for sym, sigs in sigs_by_sym.items():
        arr = arrays_by_sym[sym]
        h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
        hour, date = arr["hour"], arr["date"]
        atr = arr["atr"]
        n = arr["n"]

        for sig in sigs:
            i_entry = sig["i_entry"]
            entry_ref = sig["entry_ref"]
            sl_init = sig["sl"]
            tp = sig["tp"]
            direction = sig["direction"]
            risk_px = sig["risk_px"]
            entry_date = date[i_entry]

            # Compute per-leg plan
            if grid_cfg is None:
                legs = [{"weight": 1.0, "price": entry_ref, "bar": i_entry,
                         "filled": False, "pullback_atr": 0.0}]
            else:
                legs = []
                atr_ref = atr[i_entry-1] if i_entry > 0 else atr[i_entry]
                for (wt, pullback_atr) in grid_cfg["legs"]:
                    if pullback_atr == 0:
                        lp = entry_ref
                    elif direction == "LONG":
                        lp = entry_ref - pullback_atr * atr_ref
                    else:
                        lp = entry_ref + pullback_atr * atr_ref
                    legs.append({"weight": wt, "price": lp, "bar": None,
                                  "filled": False, "pullback_atr": pullback_atr})

            grid_timeout = grid_cfg.get("timeout_bars", 12) if grid_cfg else 0

            risk_usd = balance * RISK_PER_TRADE
            if risk_px <= 0 or entry_ref <= 0:
                continue

            # Placeholder qty_usd — exact qty depends on avg fill price, computed post-hoc.
            # We pre-size based on entry_ref and risk_px:
            qty_usd_nominal = risk_usd / (risk_px / entry_ref)
            if qty_usd_nominal > balance * MAX_LEVERAGE:
                qty_usd_nominal = balance * MAX_LEVERAGE

            trail_active = False
            sl_cur = sl_init
            mfe = 0.0
            mae = 0.0
            exit_reason = "EOS"
            exit_px = c[n-1]
            exit_bar = n - 1
            filled_count = 0
            avg_entry = 0.0
            total_weight = 0.0

            for j in range(i_entry, n):
                # Time stop
                if date[j] != entry_date:
                    # Already past session day
                    if filled_count == 0:
                        break
                    exit_reason = "EOS"
                    exit_px = c[j-1] if j > i_entry else c[j]
                    exit_bar = j - 1 if j > i_entry else j
                    break
                if hour[j] >= session_end_hr:
                    if filled_count == 0:
                        break
                    exit_reason = "EOS"
                    exit_px = c[j]
                    exit_bar = j
                    break

                bar_o = o[j]; bar_h = h[j]; bar_l = l[j]

                # Fill legs
                for leg in legs:
                    if leg["filled"]: continue
                    fill_now = False
                    fill_price = None
                    if leg["pullback_atr"] == 0:
                        # Market / taker at entry bar open OR limit at entry_ref for retest
                        if j == i_entry:
                            fill_now = True
                            if sig.get("is_limit_retest", False):
                                fill_price = entry_ref  # limit fills exactly
                            else:
                                fill_price = bar_o * (1 + SLIPPAGE) if direction == "LONG" else bar_o * (1 - SLIPPAGE)
                    else:
                        # Limit order leg — expires after timeout
                        if j - i_entry > grid_timeout:
                            continue
                        target = leg["price"]
                        if direction == "LONG" and bar_l <= target:
                            fill_now = True
                            fill_price = target
                        elif direction == "SHORT" and bar_h >= target:
                            fill_now = True
                            fill_price = target
                    if fill_now:
                        leg["filled"] = True
                        leg["bar"] = j
                        leg["fill_price"] = fill_price
                        new_total = total_weight + leg["weight"]
                        if new_total > 0:
                            avg_entry = (avg_entry * total_weight + fill_price * leg["weight"]) / new_total
                        total_weight = new_total
                        filled_count += 1

                if filled_count == 0:
                    continue  # nothing filled yet

                # MFE/MAE based on avg_entry
                if direction == "LONG":
                    cur_mfe_r = (bar_h - avg_entry) / risk_px
                    cur_mae_r = (avg_entry - bar_l) / risk_px
                else:
                    cur_mfe_r = (avg_entry - bar_l) / risk_px
                    cur_mae_r = (bar_h - avg_entry) / risk_px
                mfe = max(mfe, cur_mfe_r)
                mae = max(mae, cur_mae_r)

                # Trail activation
                if trail_cfg is not None:
                    act_r = trail_cfg.get("activate_at_r", 1.0)
                    if not trail_active and cur_mfe_r >= act_r:
                        trail_active = True
                    if trail_active:
                        mode = trail_cfg.get("trail_mode", "swing")
                        if mode == "breakeven":
                            if direction == "LONG":
                                sl_cur = max(sl_cur, avg_entry)
                            else:
                                sl_cur = min(sl_cur, avg_entry)
                        elif mode == "atr":
                            mlt = trail_cfg.get("atr_mult", 2.0)
                            if direction == "LONG":
                                sl_cur = max(sl_cur, c[j] - atr[j] * mlt)
                            else:
                                sl_cur = min(sl_cur, c[j] + atr[j] * mlt)
                        elif mode == "swing":
                            k = trail_cfg.get("swing_bars", 5)
                            buf = trail_cfg.get("buffer_atr", 0.2)
                            kk = min(k, j - i_entry)
                            if kk >= 1:
                                if direction == "LONG":
                                    new_sl = float(np.min(l[j-kk:j+1])) - atr[j] * buf
                                    sl_cur = max(sl_cur, new_sl)
                                else:
                                    new_sl = float(np.max(h[j-kk:j+1])) + atr[j] * buf
                                    sl_cur = min(sl_cur, new_sl)

                # Check SL/TP — conservative: SL checked first
                if direction == "LONG":
                    if bar_l <= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur
                        exit_bar = j
                        break
                    if bar_h >= tp:
                        exit_reason = "TP"
                        exit_px = tp
                        exit_bar = j
                        break
                else:
                    if bar_h >= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur
                        exit_bar = j
                        break
                    if bar_l <= tp:
                        exit_reason = "TP"
                        exit_px = tp
                        exit_bar = j
                        break

            if filled_count == 0:
                continue

            # Compute PnL per leg
            qty_total_usd = qty_usd_nominal * total_weight  # since total_weight may be < 1 if some legs unfilled
            pnl_usd = 0.0
            for leg in legs:
                if not leg["filled"]: continue
                leg_qty_usd = qty_usd_nominal * leg["weight"]
                fill = leg["fill_price"]
                eff_exit = exit_px * (1 - SLIPPAGE) if direction == "LONG" else exit_px * (1 + SLIPPAGE)
                entry_fee = leg_qty_usd * TAKER_FEE
                exit_fee = leg_qty_usd * TAKER_FEE
                if direction == "LONG":
                    leg_pnl = leg_qty_usd * (eff_exit - fill) / fill - entry_fee - exit_fee
                else:
                    leg_pnl = leg_qty_usd * (fill - eff_exit) / fill - entry_fee - exit_fee
                pnl_usd += leg_pnl

            balance += pnl_usd
            peak = max(peak, balance)
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            exit_r = ((exit_px - avg_entry) if direction == "LONG" else (avg_entry - exit_px)) / risk_px

            completed.append({
                "sym": sym, "direction": direction,
                "i_entry": i_entry, "exit_bar": exit_bar,
                "bars": exit_bar - i_entry,
                "reason": exit_reason,
                "pnl_usd": pnl_usd,
                "won": pnl_usd > 0,
                "exit_r": exit_r, "mfe": mfe, "mae": mae,
                "avg_entry": avg_entry, "exit": exit_px,
                "sl_init": sl_init, "tp": tp,
                "filled_legs": filled_count, "total_weight": total_weight,
                "date": entry_date,
            })

    return _aggregate(completed, balance, max_dd)


def _aggregate(completed, balance, max_dd):
    if not completed:
        return {"trades": 0, "wins": 0, "losses": 0, "wr_pct": 0.0,
                "pnl_usd": 0.0, "pnl_pct": 0.0, "profit_factor": 0.0,
                "max_dd_pct": 0.0, "avg_r": 0.0, "avg_mfe": 0.0,
                "avg_bars": 0.0, "final_bal": balance, "exit_reasons": {}}
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / len(completed) * 100
    reasons = defaultdict(int)
    for t in completed: reasons[t["reason"]] += 1
    return {
        "trades": len(completed), "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "pnl_pct": round((balance - START_BALANCE) / START_BALANCE * 100, 2),
        "profit_factor": round(pf, 3),
        "max_dd_pct": round(max_dd, 2),
        "avg_r": round(float(np.mean([t["exit_r"] for t in completed])), 3),
        "avg_mfe": round(float(np.mean([t["mfe"] for t in completed])), 3),
        "avg_bars": round(float(np.mean([t["bars"] for t in completed])), 1),
        "final_bal": round(balance, 2),
        "exit_reasons": dict(reasons),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  RUN WRAPPER
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy(name, gen_fn, arrays, ranges, htf_aligned_bysym, params,
                  trail_cfg=None, grid_cfg=None, session_end_hr=22,
                  compute_per_sym=False):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_fn(arr, ranges[sym], params, htf_aligned_bysym.get(sym))
    res = simulate(sigs, arrays, session_end_hr=session_end_hr,
                    trail_cfg=trail_cfg, grid_cfg=grid_cfg)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    if compute_per_sym:
        per_sym = {}
        for sym in sigs:
            per_sym[sym] = simulate({sym: sigs[sym]}, {sym: arrays[sym]},
                                    session_end_hr=session_end_hr,
                                    trail_cfg=trail_cfg, grid_cfg=grid_cfg)
        res["per_sym"] = per_sym
    return res


def log_r(name, r):
    logger.info("%-32s trades=%4d WR=%5.1f%% PF=%5.2f PnL=%+8.1f%% DD=%5.2f%% avgR=%+.2f",
                 name, r["trades"], r["wr_pct"], r["profit_factor"],
                 r["pnl_pct"], r["max_dd_pct"], r["avg_r"])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    logger.info("=== Session V3 backtest (next-bar entry, top-20, grid, Asia) ===")

    arrays = {}
    ranges = {}
    htf_aligned_bysym = {}
    loaded = []
    for sym in TOP20:
        df15 = _load(sym, "15m", "365d")
        if df15 is None or len(df15) < 5000:
            logger.warning("Skipping %s (no 15m)", sym)
            continue
        df1h = _load(sym, "1h", "365d")
        df4h = _load(sym, "4h", "365d")
        arr = prep_15m_arrays(df15)
        rng = build_session_ranges(arr)
        htfs = compute_htf_trends(df4h, df1h)
        aligned = {nm: align_htf(m, arr["ts"]) for nm, m in htfs.items()}
        arrays[sym] = arr
        ranges[sym] = rng
        htf_aligned_bysym[sym] = aligned
        loaded.append(sym)
    logger.info("Loaded %d/%d symbols: %s", len(loaded), len(TOP20),
                 [s.replace("/USDT:USDT","") for s in loaded])

    results = []

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: 3 best configs from V2 on top-20
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("--- TEST 1: 3 best configs on top-20 ---")

    base = {
        "open_hr": 13, "close_hr": 22,
        "min_range_atr": 0.3, "max_range_atr": 5.0,
        "sl_mode": "atr", "sl_atr_mult": 2.5,
        "max_entry_hr": 3,
    }
    # Variant A: ADX+RngM2+sw5 (best PF)
    cfg_A = dict(base); cfg_A.update({
        "htf_filter": "adx_1h", "tp_mode": "range_mult", "tp_range_mult": 2.0,
    })
    trail_A = {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}

    # Variant B: ST+RngM2+sw5 (best PnL, lowest DD)
    cfg_B = dict(base); cfg_B.update({
        "htf_filter": "st_1h", "tp_mode": "range_mult", "tp_range_mult": 2.0,
    })
    trail_B = {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}

    # Variant C: ADX+LvlTP+sw5 (highest WR)
    cfg_C = dict(base); cfg_C.update({
        "htf_filter": "adx_1h", "tp_mode": "nearest_level",
    })
    trail_C = {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}

    for nm, p, tr in [("A_ADX_RngM2_sw5", cfg_A, trail_A),
                       ("B_ST_RngM2_sw5",  cfg_B, trail_B),
                       ("C_ADX_LvlTP_sw5", cfg_C, trail_C)]:
        r = run_strategy(f"TOP20_{nm}", gen_arb_v3, arrays, ranges, htf_aligned_bysym,
                          p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = f"TOP20_{nm}"; r["params"] = p; r["trail_cfg"] = tr
        results.append(r)
        log_r(f"TOP20_{nm}", r)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: Grid entry variants on best config (A)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("--- TEST 2: Grid entry variants (on config A) ---")
    grid_variants = [
        ("Grid_2leg_p03", {"legs": [(0.5, 0.0), (0.5, 0.3)], "timeout_bars": 8}),
        ("Grid_2leg_p05", {"legs": [(0.5, 0.0), (0.5, 0.5)], "timeout_bars": 8}),
        ("Grid_2leg_p08", {"legs": [(0.5, 0.0), (0.5, 0.8)], "timeout_bars": 12}),
        ("Grid_3leg_030507", {"legs": [(0.4, 0.0), (0.3, 0.3), (0.3, 0.7)], "timeout_bars": 12}),
        ("Grid_3leg_conservative", {"legs": [(0.25, 0.0), (0.35, 0.4), (0.40, 0.8)], "timeout_bars": 16}),
        ("Grid_heavy_pullback", {"legs": [(0.3, 0.0), (0.7, 0.5)], "timeout_bars": 10}),
    ]
    for nm, gcfg in grid_variants:
        r = run_strategy(f"GRID_{nm}", gen_arb_v3, arrays, ranges, htf_aligned_bysym,
                          cfg_A, trail_cfg=trail_A, grid_cfg=gcfg, compute_per_sym=False)
        r["config_id"] = f"GRID_{nm}"; r["params"] = cfg_A; r["grid_cfg"] = gcfg
        results.append(r)
        log_r(f"GRID_{nm}", r)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: Asian session strategies
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("--- TEST 3: Asian session strategies ---")

    asia_configs = [
        # Tokyo breakout (00-04 range, 04-07 entry, close by 09:00)
        ("Asia_Tokyo_none_R2",    gen_asia_tokyobreak,
            {"sl_atr_mult": 2.0, "tp_mode": "range_mult", "tp_range_mult": 2.0, "min_range_atr": 0.3},
            None, 9),
        ("Asia_Tokyo_adx_R2",     gen_asia_tokyobreak,
            {"sl_atr_mult": 2.0, "tp_mode": "range_mult", "tp_range_mult": 2.0,
             "htf_filter": "adx_1h", "min_range_atr": 0.3},
            None, 9),
        ("Asia_Tokyo_st_R2",      gen_asia_tokyobreak,
            {"sl_atr_mult": 2.0, "tp_mode": "range_mult", "tp_range_mult": 2.0,
             "htf_filter": "st_1h", "min_range_atr": 0.3},
            None, 9),
        ("Asia_Tokyo_st_sw5",     gen_asia_tokyobreak,
            {"sl_atr_mult": 2.5, "tp_mode": "fixed_r", "tp_r": 3.0,
             "htf_filter": "st_1h", "min_range_atr": 0.3},
            {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 9),
        # London open ARB (07-10 entry)
        ("Asia_LOopen_adx_R2",    gen_asia_londonopen,
            {"sl_atr_mult": 2.5, "tp_mode": "range_mult", "tp_range_mult": 2.0,
             "htf_filter": "adx_1h", "max_entry_hr": 3},
            None, 13),
        ("Asia_LOopen_st_R2",     gen_asia_londonopen,
            {"sl_atr_mult": 2.5, "tp_mode": "range_mult", "tp_range_mult": 2.0,
             "htf_filter": "st_1h", "max_entry_hr": 3},
            None, 13),
        ("Asia_LOopen_st_sw5",    gen_asia_londonopen,
            {"sl_atr_mult": 2.5, "tp_mode": "range_mult", "tp_range_mult": 2.0,
             "htf_filter": "st_1h", "max_entry_hr": 3},
            {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 13),
        ("Asia_LOopen_adx_sw5",   gen_asia_londonopen,
            {"sl_atr_mult": 2.5, "tp_mode": "range_mult", "tp_range_mult": 2.0,
             "htf_filter": "adx_1h", "max_entry_hr": 3},
            {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 13),
        # PDH/PDL breakout 00-07
        ("Asia_PDH_adx_R25",      gen_asia_pdhbreak,
            {"sl_atr_mult": 2.0, "tp_mode": "fixed_r", "tp_r": 2.5, "htf_filter": "adx_1h"},
            None, 13),
        ("Asia_PDH_st_R3_sw5",    gen_asia_pdhbreak,
            {"sl_atr_mult": 2.0, "tp_mode": "fixed_r", "tp_r": 3.0, "htf_filter": "st_1h"},
            {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 13),
        # Tokyo mean-reversion (fade extremes)
        ("Asia_TokyoMR_mid",      gen_asia_tokyo_mr,
            {"sl_atr_mult": 1.0, "tp_to": "mid", "min_range_atr": 0.8},
            None, 9),
        ("Asia_TokyoMR_opp",      gen_asia_tokyo_mr,
            {"sl_atr_mult": 1.0, "tp_to": "opp", "min_range_atr": 0.8},
            None, 9),
    ]
    for item in asia_configs:
        nm, fn, p, tr, sess_end = item
        r = run_strategy(nm, fn, arrays, ranges, htf_aligned_bysym,
                          p, trail_cfg=tr, session_end_hr=sess_end)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r)
        log_r(nm, r)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: BTC+ETH only with realistic V3 execution (compare vs V2)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("--- TEST 4: BTC+ETH only with realistic next-bar entry ---")
    be_arrays = {s: arrays[s] for s in ["BTC/USDT:USDT", "ETH/USDT:USDT"] if s in arrays}
    be_ranges = {s: ranges[s] for s in be_arrays}
    be_htf = {s: htf_aligned_bysym[s] for s in be_arrays}
    for nm, p, tr in [("BTC_ETH_A_ADX_RngM2", cfg_A, trail_A),
                       ("BTC_ETH_B_ST_RngM2",  cfg_B, trail_B),
                       ("BTC_ETH_C_ADX_LvlTP", cfg_C, trail_C)]:
        r = run_strategy(nm, gen_arb_v3, be_arrays, be_ranges, be_htf,
                          p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r)
        log_r(nm, r)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: Retest/limit entry on top-20
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("--- TEST 5: Retest (limit at break level) entry ---")
    retest_configs = [
        ("Retest_ADX_RngM2",  {**cfg_A, "retest_bars": 8}, trail_A),
        ("Retest_ST_RngM2",   {**cfg_B, "retest_bars": 8}, trail_B),
        ("Retest_ADX_LvlTP",  {**cfg_C, "retest_bars": 8}, trail_C),
        ("Retest_ADX_R3_noTr",{**cfg_A, "retest_bars": 8, "tp_mode":"fixed_r", "tp_r":3.0}, None),
        ("Retest_ADX_wait12", {**cfg_A, "retest_bars": 12}, trail_A),
        ("Retest_ADX_wait4",  {**cfg_A, "retest_bars": 4}, trail_A),
    ]
    for nm, p, tr in retest_configs:
        r = run_strategy(nm, gen_arb_v3_retest, arrays, ranges, htf_aligned_bysym,
                          p, trail_cfg=tr, compute_per_sym=False)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r)
        log_r(nm, r)

    # Retest + BTC+ETH only
    logger.info("--- TEST 5B: Retest on BTC+ETH only ---")
    for nm, p, tr in [("Retest_BE_ADX_RngM2", {**cfg_A, "retest_bars": 8}, trail_A),
                       ("Retest_BE_ST_RngM2",  {**cfg_B, "retest_bars": 8}, trail_B)]:
        r = run_strategy(nm, gen_arb_v3_retest, be_arrays, be_ranges, be_htf,
                          p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r)
        log_r(nm, r)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 6: Hour-specific entry windows (only first N hours)
    # ═══════════════════════════════════════════════════════════════════════
    logger.info("--- TEST 6: Narrow entry windows ---")
    for max_hr in [1, 2, 3, 4]:
        p = {**cfg_A, "max_entry_hr": max_hr}
        r = run_strategy(f"Hr{max_hr}_ADX_RngM2", gen_arb_v3, arrays, ranges, htf_aligned_bysym,
                          p, trail_cfg=trail_A)
        r["config_id"] = f"Hr{max_hr}_ADX_RngM2"; r["params"] = p; r["trail_cfg"] = trail_A
        results.append(r)
        log_r(f"Hr{max_hr}_ADX_RngM2", r)
    for max_hr in [1, 2, 3, 4]:
        p = {**cfg_A, "max_entry_hr": max_hr, "retest_bars": 8}
        r = run_strategy(f"Hr{max_hr}_Retest_ADX", gen_arb_v3_retest, arrays, ranges, htf_aligned_bysym,
                          p, trail_cfg=trail_A)
        r["config_id"] = f"Hr{max_hr}_Retest_ADX"; r["params"] = p; r["trail_cfg"] = trail_A
        results.append(r)
        log_r(f"Hr{max_hr}_Retest_ADX", r)

    # ═══════════════════════════════════════════════════════════════════════
    # SAVE + REPORT
    # ═══════════════════════════════════════════════════════════════════════
    out_path = RESULTS_DIR / f"backtest_session_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(out_path, "w") as f:
            json.dump({"symbols": loaded, "results": results}, f, indent=2, default=str)
        logger.info("Saved %s (%.1fs)", out_path.name, _time.time()-t0)
    except Exception as e:
        logger.warning("Save failed: %s", e)

    # Reports
    def _fmt(r):
        return (f"{r['config_id'][:32]:32s} {r['trades']:>5} {r['wr_pct']:>5.1f} "
                f"{r['profit_factor']:>5.2f} {r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f} "
                f"{r['avg_r']:>+6.2f}")

    print("\n" + "="*90)
    print("TEST 1 — TOP-20 with 3 best V2 configs")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    t1 = [r for r in results if r["config_id"].startswith("TOP20_")]
    for r in t1: print(_fmt(r))

    print("\n--- Per-symbol breakdown (TEST 1) ---")
    print(f"{'config':20s} {'sym':6s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8}")
    for r in t1:
        cfg = r["config_id"].replace("TOP20_", "")
        per = r.get("per_sym", {})
        # sort by PnL desc
        sorted_sym = sorted(per.items(), key=lambda kv: kv[1]["pnl_pct"], reverse=True)
        for sym, sr in sorted_sym:
            s = sym.replace("/USDT:USDT", "")
            print(f"{cfg[:20]:20s} {s:6s} {sr['trades']:>5} {sr['wr_pct']:>5.1f} {sr['profit_factor']:>5.2f} {sr['pnl_pct']:>+8.1f}")

    print("\n" + "="*90)
    print("TEST 2 — Grid entry variants (config A)")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    for r in [x for x in results if x["config_id"].startswith("GRID_")]:
        print(_fmt(r))

    print("\n" + "="*90)
    print("TEST 3 — Asian session strategies")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    for r in [x for x in results if x["config_id"].startswith("Asia_")]:
        print(_fmt(r))

    print("\n" + "="*90)
    print("TEST 4 — BTC+ETH only with REALISTIC execution")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    for r in [x for x in results if x["config_id"].startswith("BTC_ETH_")]:
        print(_fmt(r))
        per = r.get("per_sym", {})
        for sym, sr in per.items():
            s = sym.replace("/USDT:USDT", "")
            print(f"    {s:6s} trades={sr['trades']:>4} WR={sr['wr_pct']:>5.1f} PF={sr['profit_factor']:>5.2f} PnL={sr['pnl_pct']:>+7.1f}%")

    print("\n" + "="*90)
    print("TEST 5 — Retest (limit) entry on top-20")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    for r in [x for x in results if x["config_id"].startswith("Retest_") and not x["config_id"].startswith("Retest_BE_")]:
        print(_fmt(r))

    print("\n--- TEST 5B: Retest on BTC+ETH ---")
    for r in [x for x in results if x["config_id"].startswith("Retest_BE_")]:
        print(_fmt(r))
        per = r.get("per_sym", {})
        for sym, sr in per.items():
            s = sym.replace("/USDT:USDT", "")
            print(f"    {s:6s} trades={sr['trades']:>4} WR={sr['wr_pct']:>5.1f} PF={sr['profit_factor']:>5.2f} PnL={sr['pnl_pct']:>+7.1f}%")

    print("\n" + "="*90)
    print("TEST 6 — Narrow entry windows")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    for r in [x for x in results if x["config_id"].startswith("Hr")]:
        print(_fmt(r))

    print("\n" + "="*90)
    print("OVERALL Top-15 by PF (min 40 trades)")
    print(f"{'config':32s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    filtered = [r for r in results if r.get("trades", 0) >= 40]
    for r in sorted(filtered, key=lambda x: x["profit_factor"], reverse=True)[:15]:
        print(_fmt(r))

    print("\nOVERALL Top-10 by PnL (min 40 trades)")
    for r in sorted(filtered, key=lambda x: x["pnl_pct"], reverse=True)[:10]:
        print(_fmt(r))


if __name__ == "__main__":
    main()
