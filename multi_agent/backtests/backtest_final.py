"""
backtest_final.py  --  Final clean engine (2026-05-02)
======================================================

Цель: исправить баги bbtest_level_filters.py + сделать движок гибким и
корректным для финальной валидации торговой стратегии.

Найденные баги в backtest_level_filters.py (объясняют WR 30% vs v12 WR 61%):
  1. **Break-even @ 1.5R при slippage+комиссиях = маленький лосс, а не "0R win"**
     Когда BE срабатывает (entry*1.001 для LONG), exit_price = SL*(1-slip),
     gross gain ≈ +0.0005, fees+slip ≈ -0.00075 → pnl < 0 → засчитан LOSS.
     Многие выигрышные трейды (которые прошли 1.5R, потом откатились)
     теперь зачислены в лосы. Это срезает WR с 60%+ до 30%.
  2. **Двойной slippage**: применяется к exit_price И учитывается в taker fee.
  3. **Entry at CLOSE вместо OPEN** в match_15m_entries: hasattr(dict, "df")
     всегда False, поэтому fallback к close всегда срабатывает. Но в v12
     то же самое — оба входят на close. Но это пессимистично:
     к моменту close 15m бара после HTF-сигнала большая часть импульса
     ушла, SL рядом, PF падает.

Фиксы:
  • Опциональный BE (по умолчанию OFF — как в v12)
  • Чистая модель издержек (slippage + fee, без двойного счёта)
  • Опциональный entry на OPEN следующего бара (более реалистично)
  • Учёт теней (high/low) для SL/TP — уже было, но теперь корректно
  • Поддержка 365d / 730d данных
  • Несколько стратегий: BBS_1H_5M, BBS_1H_15M, KC_4H_15M
  • Опциональный obstacle-ahead фильтр

Запуск:
  py -3 multi_agent/backtests/backtest_final.py
  py -3 multi_agent/backtests/backtest_final.py --tag 365d
  py -3 multi_agent/backtests/backtest_final.py --tag 730d --strategies BBS_1H_15M,KC_4H_15M
"""
from __future__ import annotations
import argparse
import logging
import sys
import time as _time
from pathlib import Path
from itertools import product

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal, check_head_fake,
)

logger = logging.getLogger("final")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger.setLevel(logging.INFO)

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── Портфель / risk ──────────────────────────────────────────────────────────
START_BALANCE   = 500.0
RISK_PER_TRADE  = 0.02
MAKER_FEE       = 0.00020   # 0.020% post-only вход (next_bar)
TAKER_FEE       = 0.00055   # 0.055% market выход (SL/TP)
SLIPPAGE        = 0.0005    # 0.05% — применяется ОДИН раз (на exit)
MIN_MARGIN      = 5.0
MAX_LEVERAGE    = 50
MAX_MARGIN_FRAC = 0.05
MAX_POSITIONS   = 4
SL_COOLDOWN_MS  = 4 * 3_600_000

# ── Стратегии ────────────────────────────────────────────────────────────────
STRATEGIES = {
    "BBS_1H_5M": {
        "type": "bbs", "htf": "1h", "ltf": "5m", "entry_mode": "pullback_ema21",
        "min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0,
        "headfake": True, "ema_stack": True, "daily_ema200": True,
        "sl_atr": 1.5, "tp_r": 3.0, "max_wait": 20,  # 20 баров 5m = 100 мин
        "fee_in": MAKER_FEE,  # market при pullback (тоже maker если pre-positioned, но обычно taker)
    },
    "BBS_1H_15M": {
        "type": "bbs", "htf": "1h", "ltf": "15m", "entry_mode": "next_bar",
        "min_squeeze": 4, "adx_thresh": 15.0, "vol_mult": 1.0,
        "headfake": True, "ema_stack": True, "daily_ema200": False,
        "sl_atr": 1.5, "tp_r": 2.5, "max_wait": 1,
        "fee_in": MAKER_FEE,
    },
    "KC_4H_15M": {
        "type": "kc", "htf": "4h", "ltf": "15m", "entry_mode": "next_bar",
        "min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0,
        "mom_bars": 6, "ema_stack": False, "daily_ema200": False,
        "sl_atr": 1.5, "tp_r": 2.0, "max_wait": 1,
        "fee_in": MAKER_FEE,
    },
}

# ── Символы ──────────────────────────────────────────────────────────────────
# 27 символов с 1h+15m+5m+4h на 730d
SYMBOLS_730D = [
    "ADA/USDT:USDT",  "ALGO/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",
    "ATOM/USDT:USDT", "AVAX/USDT:USDT", "BCH/USDT:USDT",  "BNB/USDT:USDT",
    "BTC/USDT:USDT",  "DOGE/USDT:USDT", "DOT/USDT:USDT",  "ETH/USDT:USDT",
    "FIL/USDT:USDT",  "HBAR/USDT:USDT", "ICP/USDT:USDT",  "INJ/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT",  "NEAR/USDT:USDT", "OP/USDT:USDT",
    "SOL/USDT:USDT",  "SUI/USDT:USDT",  "TON/USDT:USDT",  "TRX/USDT:USDT",
    "UNI/USDT:USDT",  "XLM/USDT:USDT",  "XRP/USDT:USDT",
]

# 36 символов на 365d (включая мемкоины, новые листинги)
SYMBOLS_365D = SYMBOLS_730D + [
    "1000PEPE/USDT:USDT", "POL/USDT:USDT", "SEI/USDT:USDT",
    "ONDO/USDT:USDT", "MNT/USDT:USDT", "CRV/USDT:USDT",
    "DEXE/USDT:USDT", "DASH/USDT:USDT", "XMR/USDT:USDT",
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load(symbol: str, tf: str, tag: str) -> pd.DataFrame | None:
    safe = symbol.replace("/", "_").replace(":", "_")
    # Try requested tag first, then fallback
    fallbacks = [tag]
    for alt in ("730d", "365d", "180d", "90d"):
        if alt not in fallbacks:
            fallbacks.append(alt)
    for ttag in fallbacks:
        for ext in (".parquet", ".csv"):
            p = CACHE_DIR / f"{safe}_{tf}_{ttag}{ext}"
            if p.exists():
                try:
                    df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
                    if df is None or df.empty:
                        continue
                    for c in ("open", "high", "low", "close", "volume"):
                        if c in df.columns:
                            df[c] = df[c].astype(float)
                    if "timestamp" in df.columns:
                        df["timestamp"] = df["timestamp"].astype(int)
                    return df
                except Exception:
                    continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    s = pd.Series(arr)
    res = s.ewm(span=p, adjust=False).mean().to_numpy()
    out[p - 1:] = res[p - 1:]
    return out

def _sma(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    cs = np.nancumsum(arr)
    out[p - 1:] = (cs[p - 1:] - np.concatenate(([0.0], cs[:-p]))) / p
    return out

def _stdev(arr: np.ndarray, p: int) -> np.ndarray:
    return pd.Series(arr).rolling(p).std(ddof=0).to_numpy()

def _atr(hi, lo, cl, p=14):
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    out = np.full(n, np.nan)
    if n < p:
        return out
    out[p - 1] = np.mean(tr[:p])
    for i in range(p, n):
        out[i] = out[i-1]*(p-1)/p + tr[i]/p
    return out

def _adx(hi, lo, cl, p=14):
    n = len(cl)
    adx = np.full(n, 20.0)
    pdm = np.zeros(n); ndm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up = hi[i]-hi[i-1]; dn = lo[i-1]-lo[i]
        pdm[i] = up if up > dn and up > 0 else 0
        ndm[i] = dn if dn > up and dn > 0 else 0
    if n < p*3:
        return adx
    atr_s = np.mean(tr[1:p+1]); pdm_s = np.mean(pdm[1:p+1]); ndm_s = np.mean(ndm[1:p+1])
    dx_buf = []
    for i in range(p, n):
        atr_s = atr_s - atr_s/p + tr[i]
        pdm_s = pdm_s - pdm_s/p + pdm[i]
        ndm_s = ndm_s - ndm_s/p + ndm[i]
        pdi = 100*pdm_s/atr_s if atr_s > 0 else 0
        ndi = 100*ndm_s/atr_s if atr_s > 0 else 0
        s = pdi + ndi
        dx = abs(pdi-ndi)/s*100 if s > 0 else 0
        dx_buf.append(dx)
        if len(dx_buf) == p:
            adx[i] = np.mean(dx_buf)
        elif len(dx_buf) > p:
            adx[i] = (adx[i-1]*(p-1)+dx)/p
    return adx


def prepare(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 60:
        return None
    op = df["open"].to_numpy(float) if "open" in df.columns else df["close"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    vo = df["volume"].to_numpy(float)
    ts = df["timestamp"].to_numpy(int)
    sma20 = _sma(cl, 20); std20 = _stdev(cl, 20)
    vol_ma = _sma(vo, 20)
    return {
        "op": op, "cl": cl, "hi": hi, "lo": lo, "ts": ts, "n": len(cl),
        "atr": _atr(hi, lo, cl, 14),
        "adx": _adx(hi, lo, cl, 14),
        "ema21": _ema(cl, 21), "ema50": _ema(cl, 50), "ema200": _ema(cl, 200),
        "bb_upper": sma20 + 2*std20, "bb_lower": sma20 - 2*std20, "bb_mid": sma20,
        "kc_mid": _ema(cl, 20),
        "kc_atr": _atr(hi, lo, cl, 10),
        "vol_ratio": np.where(vol_ma > 0, vo / vol_ma, 1.0),
        "df": df,
    }


def get_daily_dir(d1d, ts: int):
    if d1d is None:
        return None
    idx = int(np.searchsorted(d1d["ts"], ts, side="right")) - 1
    if idx < 200 or np.isnan(d1d["ema200"][idx]):
        return None
    return "LONG" if d1d["cl"][idx] > d1d["ema200"][idx] else "SHORT"


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE-AHEAD filter (Variant C)
# ─────────────────────────────────────────────────────────────────────────────

def is_obstacle_ahead(dhtf: dict, bar_idx: int, direction: str,
                      level_pct: float, level_bars: int) -> bool:
    cl = dhtf["cl"]; hi = dhtf["hi"]; lo = dhtf["lo"]
    close = cl[bar_idx]
    if close <= 0:
        return False
    start = max(0, bar_idx - level_bars)
    if direction == "LONG":
        highs = hi[start:bar_idx]
        highs_above = highs[highs > close]
        if len(highs_above) == 0:
            return False
        nearest = float(np.min(highs_above))
        return (nearest - close) / close <= level_pct
    else:
        lows = lo[start:bar_idx]
        lows_below = lows[lows < close]
        if len(lows_below) == 0:
            return False
        nearest = float(np.max(lows_below))
        return (close - nearest) / close <= level_pct


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_bbs(dhtf: dict, params: dict, obs_pct: float = 0.0, obs_bars: int = 0) -> dict:
    """{bar_idx: {direction, entry_price_htf, sl_atr_value}}"""
    try:
        sq_ind = compute_squeeze_indicators(dhtf["df"])
    except Exception:
        return {}
    if sq_ind is None:
        return {}
    cl = dhtf["cl"]; hi = dhtf["hi"]; lo = dhtf["lo"]
    n = dhtf["n"]
    adx = dhtf["adx"]; vol_r = dhtf["vol_ratio"]
    e21 = dhtf["ema21"]; e50 = dhtf["ema50"]; e200 = dhtf["ema200"]
    atr_arr = dhtf["atr"]
    min_sq = params["min_squeeze"]; adx_th = params["adx_thresh"]
    vol_m  = params["vol_mult"];   use_hf = params["headfake"]
    use_st = params.get("ema_stack", True)
    sigs = {}
    warmup = max(250, obs_bars + 10)
    for i in range(warmup, n - 1):
        if np.isnan(adx[i]) or adx[i] < adx_th:
            continue
        if vol_r[i] < vol_m:
            continue
        sig = detect_squeeze_signal(sq_ind, bar_idx=i, min_squeeze_bars=min_sq,
                                    adx_threshold=adx_th, vol_mult=vol_m)
        if sig is None:
            continue
        if use_hf and check_head_fake(sq_ind, i, 1):
            continue
        d = sig["direction"]
        if use_st:
            if d == "LONG":
                if not (cl[i] > e21[i] > e50[i] > e200[i]):
                    continue
            else:
                if not (cl[i] < e21[i] < e50[i] < e200[i]):
                    continue
        atr_v = atr_arr[i] if not np.isnan(atr_arr[i]) else cl[i]*0.01
        if atr_v <= 0:
            continue
        # Optional obstacle-ahead filter
        if obs_pct > 0 and obs_bars > 0:
            if is_obstacle_ahead(dhtf, i, d, obs_pct, obs_bars):
                continue
        sigs[i] = {"direction": d, "atr_htf": atr_v, "close_htf": cl[i]}
    return sigs


def detect_kc(dhtf: dict, params: dict, obs_pct: float = 0.0, obs_bars: int = 0) -> dict:
    """KC squeeze on HTF."""
    cl = dhtf["cl"]; hi = dhtf["hi"]; lo = dhtf["lo"]
    n = dhtf["n"]
    adx = dhtf["adx"]; vol_r = dhtf["vol_ratio"]
    bb_u = dhtf["bb_upper"]; bb_l = dhtf["bb_lower"]
    kc_mid = dhtf["kc_mid"]; kc_atr = dhtf["kc_atr"]
    atr_arr = dhtf["atr"]
    min_sq = params["min_squeeze"]; adx_th = params["adx_thresh"]
    vol_m  = params["vol_mult"];    mom_b  = params["mom_bars"]

    kc_u = kc_mid + 1.5 * kc_atr
    kc_l = kc_mid - 1.5 * kc_atr

    sq = np.zeros(n, dtype=bool); sq_len = np.zeros(n, dtype=int)
    for i in range(20, n):
        if not np.isnan(bb_u[i]) and not np.isnan(kc_u[i]):
            sq[i] = (bb_l[i] > kc_l[i]) and (bb_u[i] < kc_u[i])
    for i in range(1, n):
        sq_len[i] = sq_len[i-1] + 1 if sq[i] else 0

    sigs = {}
    warmup = max(250, mom_b + 10, obs_bars + 10)
    for i in range(warmup, n - 1):
        if sq[i]:
            continue
        if sq_len[i-1] < min_sq:
            continue
        if np.isnan(adx[i]) or adx[i] < adx_th:
            continue
        if vol_r[i] < vol_m:
            continue
        seg = cl[i-mom_b:i+1]
        if len(seg) < mom_b or np.any(np.isnan(seg)):
            continue
        slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
        atr_v = atr_arr[i] if not np.isnan(atr_arr[i]) else cl[i]*0.01
        if slope > 0:
            if np.isnan(bb_u[i]) or bb_u[i] <= kc_u[i]:
                continue
            d = "LONG"
        else:
            if np.isnan(bb_l[i]) or bb_l[i] >= kc_l[i]:
                continue
            d = "SHORT"
        if obs_pct > 0 and obs_bars > 0:
            if is_obstacle_ahead(dhtf, i, d, obs_pct, obs_bars):
                continue
        sigs[i] = {"direction": d, "atr_htf": atr_v, "close_htf": cl[i]}
    return sigs


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY MATCHING (HTF -> LTF)
# ─────────────────────────────────────────────────────────────────────────────

def match_entries(htf_sigs: dict, dhtf: dict, dltf: dict,
                  d1d, params: dict, entry_at_open: bool = True,
                  anchor: str = "close") -> dict:
    """
    Convert HTF signals → LTF entries.
    {ltf_bar_idx: {direction, entry, sl, tp, dist, source}}
    """
    if dltf is None or not htf_sigs:
        return {}
    entry_mode = params["entry_mode"]
    max_wait   = params["max_wait"]
    sl_atr     = params["sl_atr"]
    tp_r       = params["tp_r"]
    use_daily  = params.get("daily_ema200", False)
    ts_htf = dhtf["ts"]
    ts_ltf = dltf["ts"]
    op_ltf = dltf["op"]; cl_ltf = dltf["cl"]
    hi_ltf = dltf["hi"]; lo_ltf = dltf["lo"]
    atr_ltf = dltf["atr"]; e21_ltf = dltf["ema21"]
    n_ltf = dltf["n"]

    # HTF bar-period (in ms): infer from medians of ts diffs
    htf_period_ms = int(np.median(np.diff(ts_htf[:200]))) if len(ts_htf) > 1 else 3_600_000

    entries = {}
    for hbar, sig in htf_sigs.items():
        d = sig["direction"]
        if use_daily and d1d is not None:
            dd = get_daily_dir(d1d, ts_htf[hbar])
            if dd is not None and dd != d:
                continue
        # ANCHOR:
        #   "close" — search from HTF close (correct, no look-ahead)
        #   "open"  — search from HTF open (v12-style, has 45-min look-ahead!)
        if anchor == "close":
            anchor_ms = int(ts_htf[hbar]) + htf_period_ms
            ltf_start = int(np.searchsorted(ts_ltf, anchor_ms, side="left"))
        else:  # "open" — v12 style (look-ahead bias)
            anchor_ms = int(ts_htf[hbar])
            ltf_start = int(np.searchsorted(ts_ltf, anchor_ms, side="left"))
        if ltf_start >= n_ltf:
            continue

        for j in range(ltf_start, min(ltf_start + max_wait, n_ltf)):
            if j in entries:
                continue
            atr_v = atr_ltf[j] if not np.isnan(atr_ltf[j]) else cl_ltf[j]*0.01
            if atr_v <= 0:
                continue

            # Entry price selection: OPEN of bar j (more realistic for next_bar
            # post-only limit), or CLOSE if explicitly chosen
            entry_p = op_ltf[j] if entry_at_open else cl_ltf[j]

            if entry_mode == "next_bar":
                if d == "LONG":
                    sl = entry_p - sl_atr * atr_v
                    dist = entry_p - sl
                    if dist <= 0: continue
                    tp = entry_p + tp_r * dist
                else:
                    sl = entry_p + sl_atr * atr_v
                    dist = sl - entry_p
                    if dist <= 0: continue
                    tp = entry_p - tp_r * dist
                entries[j] = {"direction": d, "entry": entry_p,
                              "sl": sl, "tp": tp, "dist": dist}
                break

            elif entry_mode == "pullback_ema21":
                if np.isnan(e21_ltf[j]):
                    continue
                ema = e21_ltf[j]
                # Use bar's wick to check pullback
                if d == "LONG" and lo_ltf[j] <= ema * 1.002 and cl_ltf[j] > ema:
                    entry_p = cl_ltf[j]  # market entry on touch
                    sl = entry_p - sl_atr * atr_v
                    dist = entry_p - sl
                    if dist <= 0: continue
                    tp = entry_p + tp_r * dist
                    entries[j] = {"direction": d, "entry": entry_p,
                                  "sl": sl, "tp": tp, "dist": dist}
                    break
                elif d == "SHORT" and hi_ltf[j] >= ema * 0.998 and cl_ltf[j] < ema:
                    entry_p = cl_ltf[j]
                    sl = entry_p + sl_atr * atr_v
                    dist = sl - entry_p
                    if dist <= 0: continue
                    tp = entry_p - tp_r * dist
                    entries[j] = {"direction": d, "entry": entry_p,
                                  "sl": sl, "tp": tp, "dist": dist}
                    break
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def simulate(all_entries: dict[str, dict[str, dict]],
             all_ltf:     dict[str, dict[str, dict]],
             strat_params: dict,
             start_frac: float = 0.0,
             end_frac:   float = 1.0,
             use_be:     bool  = False,
             be_trigger: float = 1.5,
             be_offset:  float = 0.001) -> dict:
    """
    Симуляция портфеля. SL первым (пессимистично), TP вторым.
    Для SL/TP fill: цена SL/TP — РЕАЛЬНАЯ цена triggered, +1× slippage на exit_price.
    Komissii: maker fee in (post-only) + taker fee out (market on SL/TP).

    use_be=False (по умолчанию) — как в v12, чистый SL/TP.
    use_be=True  — BE при движении в сторону трейда на be_trigger × dist.
    """
    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    trades  = []
    open_pos: dict[tuple, dict] = {}
    sym_cool: dict[str, int] = {}

    timeline = []
    for sym, sm in all_entries.items():
        for sid, ents in sm.items():
            dltf = all_ltf.get(sym, {}).get(sid)
            if dltf is None:
                continue
            ts_arr = dltf["ts"]; n_ltf = dltf["n"]
            i_start = int(n_ltf * start_frac); i_end = int(n_ltf * end_frac)
            for bar, td in ents.items():
                if bar < i_start or bar >= i_end:
                    continue
                timeline.append((int(ts_arr[bar]), sym, sid, bar, td))
    timeline.sort(key=lambda x: x[0])

    for ts_e, sym, sid, bar, td in timeline:
        dltf = all_ltf[sym][sid]
        cl = dltf["cl"]; hi = dltf["hi"]; lo = dltf["lo"]; n = dltf["n"]
        key = (sym, sid)
        if key in open_pos:
            continue
        if sym_cool.get(sym, 0) > ts_e:
            continue
        if len(open_pos) >= MAX_POSITIONS:
            continue

        d     = td["direction"]
        entry = td["entry"]
        sl    = td["sl"]
        tp    = td["tp"]
        dist  = td["dist"]
        risk_usd = balance * RISK_PER_TRADE
        if dist <= 0 or risk_usd < 0.01:
            continue
        qty = risk_usd / dist
        raw_margin = qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        lev = 1
        if raw_margin > max_margin and max_margin > 0:
            lev = min(int(np.ceil(raw_margin / max_margin)), MAX_LEVERAGE)
        eff_margin = raw_margin / max(lev, 1)
        if eff_margin < MIN_MARGIN:
            continue

        open_pos[key] = True
        cur_sl = sl
        be_done = False

        exit_price = None; exit_bar = None; reason = "timeout"
        for j in range(bar + 1, n):
            h_j = hi[j]; l_j = lo[j]
            if d == "LONG":
                # BE check: favorable high
                if use_be and not be_done:
                    if (h_j - entry) >= be_trigger * dist:
                        cur_sl = entry * (1 + be_offset)
                        be_done = True
                # SL first (pessimistic)
                if l_j <= cur_sl:
                    exit_price = cur_sl; reason = "SL"; exit_bar = j; break
                if h_j >= tp:
                    exit_price = tp; reason = "TP"; exit_bar = j; break
            else:
                if use_be and not be_done:
                    if (entry - l_j) >= be_trigger * dist:
                        cur_sl = entry * (1 - be_offset)
                        be_done = True
                if h_j >= cur_sl:
                    exit_price = cur_sl; reason = "SL"; exit_bar = j; break
                if l_j <= tp:
                    exit_price = tp; reason = "TP"; exit_bar = j; break

        del open_pos[key]
        if exit_bar is None:
            continue

        # Применяем slippage к exit (один раз; вход — лимитный без slip)
        if d == "LONG":
            slip_exit = exit_price * (1 - SLIPPAGE)
        else:
            slip_exit = exit_price * (1 + SLIPPAGE)

        gross = (slip_exit - entry) * qty if d == "LONG" else (entry - slip_exit) * qty
        fee_in_pct = strat_params.get(sid, {}).get("fee_in", MAKER_FEE)
        entry_fee = entry * qty * fee_in_pct
        exit_fee  = slip_exit * qty * TAKER_FEE
        pnl = gross - entry_fee - exit_fee

        balance += pnl
        balance = max(balance, 0.01)
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

        trades.append({
            "sym": sym, "strat": sid, "direction": d,
            "entry": entry, "exit": slip_exit, "reason": reason,
            "pnl_usd": round(pnl, 4),
            "pnl_r":   round(pnl / risk_usd, 2) if risk_usd > 0 else 0,
            "win": pnl > 0,
        })
        if reason == "SL":
            sym_cool[sym] = int(dltf["ts"][exit_bar]) + SL_COOLDOWN_MS

    n_t = len(trades)
    wins = [t for t in trades if t["win"]]
    loss = [t for t in trades if not t["win"]]
    gw = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in loss))
    pf = gw / gl if gl > 0 else float("inf")
    wr = len(wins)/n_t*100 if n_t > 0 else 0.0
    avg_r = float(np.mean([t["pnl_r"] for t in trades])) if trades else 0.0

    # Per strat
    by_s = {}
    for t in trades:
        by_s.setdefault(t["strat"], []).append(t)
    per_strat = {}
    for s, ts_ in by_s.items():
        w = [x for x in ts_ if x["win"]]
        gw_s = sum(x["pnl_usd"] for x in w)
        gl_s = abs(sum(x["pnl_usd"] for x in ts_ if not x["win"]))
        per_strat[s] = {
            "trades": len(ts_),
            "wr": round(len(w)/len(ts_)*100, 1),
            "pf": round(gw_s/gl_s, 2) if gl_s > 0 else float("inf"),
            "pnl_r": round(float(np.mean([x["pnl_r"] for x in ts_])), 2),
        }

    return {
        "trades": n_t, "wins": len(wins), "losses": len(loss),
        "wr": round(wr, 1), "pf": round(pf, 2),
        "total_pnl": round(sum(t["pnl_usd"] for t in trades), 2),
        "avg_r": round(avg_r, 2),
        "max_dd": round(max_dd, 1),
        "final_bal": round(balance, 2),
        "per_strat": per_strat,
        "trade_list": trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# RUN HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def load_all(symbols, tag):
    data = {}
    for sym in symbols:
        cell = {}
        for tf in ("1h", "4h", "15m", "5m", "1d"):
            df = _load(sym, tf, tag)
            if df is not None:
                cell[tf] = prepare(df)
            else:
                cell[tf] = None
        # build d1d compact
        if cell["1d"] is not None:
            cl1d = cell["1d"]["cl"]; ts1d = cell["1d"]["ts"]
            cell["d1d"] = {"cl": cl1d, "ts": ts1d, "ema200": _ema(cl1d, 200)}
        else:
            cell["d1d"] = None
        if cell["1h"] is not None and cell["1h"]["n"] >= 300:
            data[sym] = cell
    logger.info("Loaded %d/%d symbols (tag=%s)", len(data), len(symbols), tag)
    return data


def build_entries(data, strategies, obs_pct=0.0, obs_bars=0,
                  entry_at_open=True, anchor="close"):
    """
    Returns {sym: {sid: entries_dict}}, {sym: {sid: dltf}}
    """
    all_e = {}; all_l = {}
    sig_count = {sid: 0 for sid in strategies}
    for sym, c in data.items():
        all_e[sym] = {}; all_l[sym] = {}
        for sid, p in strategies.items():
            dhtf = c.get(p["htf"]); dltf = c.get(p["ltf"])
            if dhtf is None or dltf is None:
                continue
            if p["type"] == "bbs":
                sigs = detect_bbs(dhtf, p, obs_pct=obs_pct, obs_bars=obs_bars)
            else:
                sigs = detect_kc(dhtf, p, obs_pct=obs_pct, obs_bars=obs_bars)
            sig_count[sid] += len(sigs)
            ents = match_entries(sigs, dhtf, dltf, c.get("d1d"), p,
                                 entry_at_open=entry_at_open, anchor=anchor)
            all_e[sym][sid] = ents
            all_l[sym][sid] = dltf
    return all_e, all_l, sig_count


def fmt(label, r):
    return (f"  {label:<46} T={r['trades']:>4}  WR={r['wr']:>5.1f}%  PF={r['pf']:>5.2f}  "
            f"DD={r['max_dd']:>4.1f}%  PnL=${r['total_pnl']:>10.2f}  avgR={r['avg_r']:>+5.2f}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="365d", choices=["180d", "365d", "730d"])
    ap.add_argument("--strategies", default="BBS_1H_15M",
                    help="Comma-separated list of strategy IDs")
    ap.add_argument("--mode", default="audit",
                    choices=["audit", "grid", "portfolio", "year"])
    ap.add_argument("--symbols", default="730d",
                    choices=["730d", "365d"])
    args = ap.parse_args()

    t0 = _time.time()
    SYMS = SYMBOLS_730D if args.symbols == "730d" else SYMBOLS_365D
    strats_used = {sid: STRATEGIES[sid] for sid in args.strategies.split(",") if sid in STRATEGIES}

    logger.info("=== Backtest FINAL ===")
    logger.info("Tag=%s symbols=%d strategies=%s mode=%s",
                args.tag, len(SYMS), list(strats_used), args.mode)

    data = load_all(SYMS, args.tag)
    if not data:
        logger.error("No data!")
        return

    SEP = "=" * 130
    print(f"\n{SEP}")
    print(f"  BACKTEST FINAL  --  tag={args.tag}  syms={len(data)}  strats={','.join(strats_used)}")
    print(f"  Bal=${START_BALANCE} Risk={RISK_PER_TRADE*100:.0f}%  "
          f"Maker={MAKER_FEE*100:.3f}% Taker={TAKER_FEE*100:.3f}% Slip={SLIPPAGE*100:.2f}%")
    print(f"  Max positions={MAX_POSITIONS}  SL cooldown=4h")
    print(f"{SEP}\n")

    if args.mode == "audit":
        # AUDIT MODE: матрица anchor × entry_price × BE
        print("== AUDIT: матрица anchor × entry × BE")
        print("    anchor=close (REALISTIC: HTF closed, then LTF entry — no look-ahead)")
        print("    anchor=open  (V12-STYLE: LTF bar inside HTF — 45-min LOOK-AHEAD bias)")
        print()
        for anc in ("close", "open"):
            for ent_mode_name, eo in [("OPEN", True), ("CLOSE", False)]:
                ae, al, sc = build_entries(data, strats_used, entry_at_open=eo, anchor=anc)
                for be_label, be_on in [("BE_OFF", False), ("BE_ON", True)]:
                    full = simulate(ae, al, strats_used, 0, 1, use_be=be_on)
                    is_  = simulate(ae, al, strats_used, 0, 0.8, use_be=be_on)
                    oos  = simulate(ae, al, strats_used, 0.8, 1, use_be=be_on)
                    tag  = f"anc={anc:>5} entry={ent_mode_name:<5} {be_label}"
                    print(f"\n  ── {tag}  HTF sigs={sum(sc.values())}")
                    print(fmt("FULL", full))
                    print(fmt("IS  (80%)", is_))
                    print(fmt("OOS (20%)", oos))

    elif args.mode == "grid":
        # GRID MODE: obstacle filter grid + BE off, REALISTIC anchor=close
        print("== GRID: obstacle-ahead фильтр (anchor=close, BE OFF, entry=OPEN)")
        ae0, al0, sc0 = build_entries(data, strats_used, entry_at_open=True,
                                       obs_pct=0, obs_bars=0, anchor="close")
        is_b = simulate(ae0, al0, strats_used, 0, 0.8, use_be=False)
        oos_b = simulate(ae0, al0, strats_used, 0.8, 1, use_be=False)
        print(fmt("BASELINE     IS", is_b))
        print(fmt("BASELINE     OOS", oos_b))
        print()

        best = None
        best_score = -1
        for pct in [0.003, 0.005, 0.010, 0.015, 0.020, 0.030]:
            for bars in [50, 100, 200, 500]:
                ae, al, sc = build_entries(data, strats_used, entry_at_open=True,
                                            obs_pct=pct, obs_bars=bars, anchor="close")
                is_ = simulate(ae, al, strats_used, 0, 0.8, use_be=False)
                oos = simulate(ae, al, strats_used, 0.8, 1, use_be=False)
                lbl = f"pct={pct*100:.1f}% bars={bars}"
                if oos["trades"] >= 8:
                    score = oos["pf"] * (oos["wr"]/100) - oos["max_dd"]/100
                    if score > best_score:
                        best_score = score
                        best = {"pct": pct, "bars": bars, "is": is_, "oos": oos}
                print(fmt(f"{lbl} IS",  is_))
                print(fmt(f"{lbl} OOS", oos))
        if best:
            print(f"\n  BEST: pct={best['pct']*100:.1f}% bars={best['bars']}  "
                  f"OOS PF={best['oos']['pf']} WR={best['oos']['wr']}% DD={best['oos']['max_dd']}%")

    elif args.mode == "portfolio":
        # PORTFOLIO MODE: combine multiple strategies
        print("== PORTFOLIO: BBS_1H_5M + BBS_1H_15M + KC_4H_15M (BE OFF, entry=OPEN)")
        all_strats = STRATEGIES
        ae, al, sc = build_entries(data, all_strats, entry_at_open=True)
        for sid, n in sc.items():
            print(f"  {sid:15s}: {n:>4} HTF signals")

        is_  = simulate(ae, al, all_strats, 0, 0.8, use_be=False)
        oos  = simulate(ae, al, all_strats, 0.8, 1, use_be=False)
        full = simulate(ae, al, all_strats, 0, 1, use_be=False)
        print(fmt("FULL portfolio", full))
        print(fmt("IS  portfolio", is_))
        print(fmt("OOS portfolio", oos))

        for label, r in (("FULL", full), ("IS", is_), ("OOS", oos)):
            print(f"\n  Per-strategy ({label}):")
            for s, v in sorted(r["per_strat"].items(), key=lambda x: -x[1]["trades"]):
                print(f"    {s:15s}: T={v['trades']:>3}  WR={v['wr']:>5.1f}%  "
                      f"PF={v['pf']:>5.2f}  avgR={v['pnl_r']:>+.2f}")

    elif args.mode == "year":
        # YEAR MODE: best obstacle config on full year, no BE
        print("== YEAR: финальная валидация (BE OFF, entry=OPEN, obstacle=optional)")
        # Run baseline + best obstacle config
        configs = [
            ("baseline (no obs)", 0.0, 0),
            ("obs 0.5% × 200h",  0.005, 200),
            ("obs 1.0% × 200h",  0.010, 200),
            ("obs 1.5% × 200h",  0.015, 200),
            ("obs 1.0% × 100h",  0.010, 100),
        ]
        for label, pct, bars in configs:
            ae, al, sc = build_entries(data, strats_used, entry_at_open=True,
                                        obs_pct=pct, obs_bars=bars)
            full = simulate(ae, al, strats_used, 0, 1, use_be=False)
            is_  = simulate(ae, al, strats_used, 0, 0.8, use_be=False)
            oos  = simulate(ae, al, strats_used, 0.8, 1, use_be=False)
            print(f"\n  {label}  HTF sigs={sum(sc.values())}")
            print(fmt("FULL", full))
            print(fmt("IS  ", is_))
            print(fmt("OOS ", oos))

    print(f"\n  Время: {_time.time()-t0:.1f}s\n{SEP}")


if __name__ == "__main__":
    main()
