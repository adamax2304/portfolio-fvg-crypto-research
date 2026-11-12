"""
backtest_v10.py  --  Deep MTF Optimization + Swing Strategy + 365d (2026-04-13)

Goals:
  1. Optimize current MTF intraday strategies (fine-tune around v9 optimum)
  2. New swing strategy: 4h signal -> 1h/15m entry
  3. Global trend filter (Daily EMA200) on all strategies
  4. Extended SL/TP optimization grid
  5. Walk-forward on full 365d data
  6. Full portfolio simulation

Architecture:
  Phase 0   Load 365d data (5m, 15m, 1h, 4h, 1d)
  Phase 1   Optimize MTF BBS 1h->5m (24-point grid around v9 best)
  Phase 2   Optimize MTF BBS 1h->15m (24-point grid)
  Phase 3   Optimize MTF KC 1h->15m (18-point grid)
  Phase 4   New: Swing BBS 4h->1h and 4h->15m
  Phase 5   New: Swing KC 4h->1h and 4h->15m
  Phase 6   Global trend filter test (Daily EMA200)
  Phase 7   SL/TP deep optimization (20+ combos) for top configs
  Phase 8   Walk-forward 67/33 on 365d
  Phase 9   Portfolio simulation
  Phase 10  Final report
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

from multi_agent.shared.regime_detector import (
    SoftRegimeDetector,
    precompute_1h_indicators, precompute_4h_indicators,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
import multi_agent.config as cfg

logger = logging.getLogger("v10")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

START_BALANCE = 1000.0
TAKER_FEE = 0.00055
SLIPPAGE  = 0.0005
MAX_LEVERAGE = 50

# ===========================================================================
#  DATA LOADING
# ===========================================================================

def _load(symbol, tf, days_tag="365d"):
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in (days_tag, "180d", "90d"):
        for ext in (".parquet", ".csv"):
            fpath = CACHE_DIR / f"{safe}_{tf}_{tag}{ext}"
            if fpath.exists():
                try:
                    df = pd.read_parquet(fpath) if ext == ".parquet" else pd.read_csv(fpath)
                    if df is not None and not df.empty:
                        for c in ("open", "high", "low", "close", "volume"):
                            if c in df.columns:
                                df[c] = df[c].astype(float)
                        if "timestamp" in df.columns:
                            df["timestamp"] = df["timestamp"].astype(int)
                        return df
                except Exception:
                    continue
    return None


def resample_to_30m(df15):
    if df15 is None or len(df15) < 4:
        return None
    df = df15.copy()
    ms_30m = 1_800_000
    df["ts_30m"] = (df["timestamp"] // ms_30m) * ms_30m
    agg = df.groupby("ts_30m").agg({
        "timestamp": "first", "open": "first", "high": "max",
        "low": "min", "close": "last", "volume": "sum",
    }).reset_index(drop=True)
    return agg


def load_all_tf(days_tag="365d"):
    all_data = {}
    for sym in SYMBOLS:
        d = {}
        d["df5"]  = _load(sym, "5m", days_tag)
        d["df15"] = _load(sym, "15m", days_tag)
        d["df30"] = resample_to_30m(d["df15"])
        d["df1h"] = _load(sym, "1h", days_tag)
        d["df4h"] = _load(sym, "4h", days_tag)
        d["df1d"] = _load(sym, "1d", days_tag)

        if d["df15"] is None or len(d["df15"]) < 300:
            continue
        if d["df1h"] is None or len(d["df1h"]) < 210:
            continue

        d["ind_1h"] = precompute_1h_indicators(d["df1h"])
        d["ind_4h"] = precompute_4h_indicators(d["df4h"]) if d["df4h"] is not None and len(d["df4h"]) > 50 else {}
        d["ts1h"] = d["df1h"]["timestamp"].astype(int).to_numpy()
        d["ts4h"] = d["df4h"]["timestamp"].astype(int).to_numpy() if d["df4h"] is not None else None

        all_data[sym] = d
    return all_data


# ===========================================================================
#  INDICATOR HELPERS
# ===========================================================================

def _sma(arr, period):
    out = np.full_like(arr, np.nan, dtype=float)
    cs = np.cumsum(arr)
    out[period-1:] = (cs[period-1:] - np.concatenate(([0], cs[:-period]))) / period
    return out

def _ema(arr, period):
    out = np.full_like(arr, np.nan, dtype=float)
    k = 2.0 / (period + 1)
    out[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out

def _stdev(arr, period):
    out = np.full_like(arr, np.nan, dtype=float)
    for i in range(period-1, len(arr)):
        out[i] = np.std(arr[i-period+1:i+1])
    return out

def _rsi(cl, period=14):
    n = len(cl)
    rsi = np.full(n, 50.0)
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    if n <= period:
        return rsi
    ag = np.mean(gain[1:period+1])
    al = np.mean(loss[1:period+1])
    for i in range(period, n):
        ag = (ag * (period-1) + gain[i]) / period
        al = (al * (period-1) + loss[i]) / period
        rsi[i] = 100.0 - 100.0 / (1.0 + ag / al) if al > 0 else 100.0
    return rsi

def _atr(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i-1] * (period-1) + tr[i]) / period
    return out

def _adx(hi, lo, cl, period=14):
    n = len(cl)
    adx = np.full(n, 20.0)
    pdm = np.zeros(n); ndm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up = hi[i]-hi[i-1]; down = lo[i-1]-lo[i]
        pdm[i] = up if (up > down and up > 0) else 0
        ndm[i] = down if (down > up and down > 0) else 0
    if n < period * 3:
        return adx
    atr_s = np.mean(tr[1:period+1])
    pdm_s = np.mean(pdm[1:period+1])
    ndm_s = np.mean(ndm[1:period+1])
    dx_arr = []
    for i in range(period, n):
        atr_s = atr_s - atr_s/period + tr[i]
        pdm_s = pdm_s - pdm_s/period + pdm[i]
        ndm_s = ndm_s - ndm_s/period + ndm[i]
        pdi = 100*pdm_s/atr_s if atr_s > 0 else 0
        ndi = 100*ndm_s/atr_s if atr_s > 0 else 0
        s = pdi + ndi
        dx = abs(pdi-ndi)/s*100 if s > 0 else 0
        dx_arr.append(dx)
        if len(dx_arr) == period:
            adx[i] = np.mean(dx_arr)
        elif len(dx_arr) > period:
            adx[i] = (adx[i-1]*(period-1)+dx)/period
    return adx


def prepare_tf_arrays(df):
    if df is None or len(df) < 50:
        return None
    cl = df["close"].astype(float).to_numpy()
    hi = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    op = df["open"].astype(float).to_numpy()
    vo = df["volume"].astype(float).to_numpy()
    ts = df["timestamp"].astype(int).to_numpy()
    n = len(cl)
    atr14 = _atr(hi, lo, cl, 14)
    adx14 = _adx(hi, lo, cl, 14)
    rsi14 = _rsi(cl, 14)
    ema200 = _ema(cl, 200)

    sma20 = _sma(cl, 20)
    std20 = _stdev(cl, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_mid = sma20

    kc_mid = _ema(cl, 20)
    atr_kc = _atr(hi, lo, cl, 10)
    kc_upper = kc_mid + 1.5 * atr_kc
    kc_lower = kc_mid - 1.5 * atr_kc

    kc_squeeze = np.zeros(n, dtype=bool)
    kc_sq_len = np.zeros(n, dtype=int)
    for i in range(20, n):
        if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
            kc_squeeze[i] = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])
    for i in range(1, n):
        if kc_squeeze[i]:
            kc_sq_len[i] = kc_sq_len[i-1] + 1

    vol_ma20 = _sma(vo, 20)
    vol_ratio = np.where(vol_ma20 > 0, vo / vol_ma20, 1.0)

    ema21 = _ema(cl, 21)
    ema50 = _ema(cl, 50)

    return {
        "df": df, "cl": cl, "hi": hi, "lo": lo, "op": op, "vo": vo, "ts": ts, "n": n,
        "atr": atr14, "adx": adx14, "rsi": rsi14,
        "ema200": ema200, "ema50": ema50, "ema21": ema21,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
        "kc_upper": kc_upper, "kc_lower": kc_lower, "kc_mid": kc_mid,
        "kc_squeeze": kc_squeeze, "kc_sq_len": kc_sq_len,
        "vol_ratio": vol_ratio,
    }


def prepare_daily_trend(df_daily):
    """Compute daily EMA200 for global trend filter."""
    if df_daily is None or len(df_daily) < 210:
        return None
    cl = df_daily["close"].astype(float).to_numpy()
    ts = df_daily["timestamp"].astype(int).to_numpy()
    ema200 = _ema(cl, 200)
    ema50 = _ema(cl, 50)
    return {"cl": cl, "ts": ts, "ema200": ema200, "ema50": ema50, "n": len(cl)}


def get_daily_trend(daily_data, bar_ts):
    """Get trend direction from daily data at given timestamp.
    Returns: 'LONG', 'SHORT', or None (no data / warmup)."""
    if daily_data is None:
        return None
    idx = int(np.searchsorted(daily_data["ts"], bar_ts, side="right")) - 1
    if idx < 200 or np.isnan(daily_data["ema200"][idx]):
        return None
    if daily_data["cl"][idx] > daily_data["ema200"][idx]:
        return "LONG"
    return "SHORT"


# ===========================================================================
#  STRATEGY: BBS (multi-TF)
# ===========================================================================

def gen_bbs_signals_tf(tf_data, htf_data, params):
    if tf_data is None:
        return {}
    df = tf_data["df"]
    try:
        squeeze_ind = compute_squeeze_indicators(df)
    except Exception:
        return {}
    if squeeze_ind is None:
        return {}

    min_sq     = params.get("min_squeeze", 10)
    adx_th     = params.get("adx_thresh", 22.0)
    vol_m      = params.get("vol_mult", 1.0)
    use_htf    = params.get("htf_filter", True)
    use_hf     = params.get("headfake_filter", True)
    rsi_filter = params.get("rsi_filter", False)
    rsi_ob     = params.get("rsi_ob", 70)
    rsi_os     = params.get("rsi_os", 30)

    cl = tf_data["cl"]; hi = tf_data["hi"]; lo = tf_data["lo"]
    n = tf_data["n"]
    adx = tf_data["adx"]
    vol_r = tf_data["vol_ratio"]
    rsi = tf_data["rsi"]

    sigs = {}
    warmup = max(300, int(n * 0.03))

    for i in range(warmup, n):
        if not np.isnan(adx[i]) and adx[i] < adx_th:
            continue
        if vol_m > 0 and vol_r[i] < vol_m:
            continue

        bbs_s = detect_squeeze_signal(squeeze_ind, bar_idx=i,
                                      min_squeeze_bars=min_sq,
                                      adx_threshold=adx_th,
                                      vol_mult=vol_m)
        if not bbs_s:
            continue

        # RSI overbought/oversold filter
        if rsi_filter:
            if bbs_s["direction"] == "LONG" and rsi[i] > rsi_ob:
                continue
            if bbs_s["direction"] == "SHORT" and rsi[i] < rsi_os:
                continue

        if use_htf and htf_data is not None:
            htf_ema = htf_data.get("ema200")
            htf_cl = htf_data.get("cl")
            htf_ts = htf_data.get("ts")
            if htf_ema is not None and htf_cl is not None:
                ts_i = tf_data["ts"][i]
                h_idx = int(np.searchsorted(htf_ts, ts_i, side="right")) - 1
                if h_idx < 200 or np.isnan(htf_ema[h_idx]):
                    continue
                htf_d = "LONG" if htf_cl[h_idx] > htf_ema[h_idx] else "SHORT"
                if bbs_s["direction"] != htf_d:
                    continue

        if use_hf and check_head_fake(squeeze_ind, i, 2):
            continue

        ce_sl = chandelier_exit_sl(hi, lo, squeeze_ind["ce_atr"],
                                   bbs_s["direction"], i,
                                   ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                                   ce_mult=cfg.BB_SQUEEZE_CE_MULT)
        rv = abs(cl[i] - ce_sl)
        if rv <= 0 or rv / cl[i] > 0.05:
            continue

        sigs[i] = {"direction": bbs_s["direction"],
                    "entry_price": float(cl[i]),
                    "default_sl": float(ce_sl)}
    return sigs


# ===========================================================================
#  STRATEGY: KC Squeeze (multi-TF)
# ===========================================================================

def gen_kc_squeeze_signals_tf(tf_data, htf_data, params):
    if tf_data is None:
        return {}

    min_sq     = params.get("min_squeeze", 10)
    adx_min    = params.get("adx_min", 15.0)
    vol_mult   = params.get("vol_mult", 1.0)
    mom_bars   = params.get("mom_bars", 12)
    use_htf    = params.get("htf_filter", False)
    sl_mult    = params.get("sl_atr_mult", 2.0)

    cl = tf_data["cl"]; hi = tf_data["hi"]; lo = tf_data["lo"]; n = tf_data["n"]
    atr = tf_data["atr"]; adx = tf_data["adx"]
    vol_r = tf_data["vol_ratio"]
    kc_sq = tf_data["kc_squeeze"]
    kc_sq_len = tf_data["kc_sq_len"]
    bb_upper = tf_data["bb_upper"]; bb_lower = tf_data["bb_lower"]
    kc_upper = tf_data["kc_upper"]; kc_lower = tf_data["kc_lower"]

    sigs = {}
    warmup = max(300, mom_bars + 2)

    for i in range(warmup, n):
        if kc_sq[i]:
            continue
        if kc_sq_len[i-1] < min_sq:
            continue
        if adx_min > 0 and (np.isnan(adx[i]) or adx[i] < adx_min):
            continue
        if vol_mult > 0 and vol_r[i] < vol_mult:
            continue

        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01

        seg = cl[i-mom_bars:i+1]
        if len(seg) < mom_bars or np.isnan(seg).any():
            continue
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0]

        if slope > 0:
            direction = "LONG"
            if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
                if bb_upper[i] <= kc_upper[i]:
                    continue
            if use_htf and htf_data is not None:
                htf_ema = htf_data.get("ema200")
                htf_cl = htf_data.get("cl")
                htf_ts = htf_data.get("ts")
                if htf_ema is not None and htf_cl is not None:
                    ts_i = tf_data["ts"][i]
                    h_idx = int(np.searchsorted(htf_ts, ts_i, side="right")) - 1
                    if h_idx < 200 or htf_cl[h_idx] <= htf_ema[h_idx]:
                        continue
            sl_v = cl[i] - atr_v * sl_mult
        else:
            direction = "SHORT"
            if not np.isnan(bb_lower[i]) and not np.isnan(kc_lower[i]):
                if bb_lower[i] >= kc_lower[i]:
                    continue
            if use_htf and htf_data is not None:
                htf_ema = htf_data.get("ema200")
                htf_cl = htf_data.get("cl")
                htf_ts = htf_data.get("ts")
                if htf_ema is not None and htf_cl is not None:
                    ts_i = tf_data["ts"][i]
                    h_idx = int(np.searchsorted(htf_ts, ts_i, side="right")) - 1
                    if h_idx < 200 or htf_cl[h_idx] >= htf_ema[h_idx]:
                        continue
            sl_v = cl[i] + atr_v * sl_mult

        sigs[i] = {"direction": direction, "entry_price": float(cl[i]),
                    "default_sl": float(sl_v)}
    return sigs


# ===========================================================================
#  MULTI-TF: Signal on HTF -> Entry on LTF
# ===========================================================================

def gen_mtf_signals(htf_sigs, htf_data, ltf_data, params,
                    daily_data=None, use_daily_filter=False):
    if ltf_data is None or htf_data is None:
        return {}

    entry_mode   = params.get("entry_mode", "pullback_ema21")
    max_wait     = params.get("max_wait_bars", 12)
    sl_atr_mult  = params.get("sl_atr_mult", 2.5)

    cl_ltf = ltf_data["cl"]; hi_ltf = ltf_data["hi"]; lo_ltf = ltf_data["lo"]
    n_ltf = ltf_data["n"]
    ts_ltf = ltf_data["ts"]; ts_htf = htf_data["ts"]
    atr_ltf = ltf_data["atr"]
    ema21_ltf = ltf_data["ema21"]

    sigs = {}

    for htf_bar, htf_sig in htf_sigs.items():
        htf_ts = htf_data["ts"][htf_bar]

        # Global trend filter
        if use_daily_filter and daily_data is not None:
            daily_trend = get_daily_trend(daily_data, htf_ts)
            if daily_trend is not None and daily_trend != htf_sig["direction"]:
                continue

        ltf_start = int(np.searchsorted(ts_ltf, htf_ts, side="left"))
        if ltf_start >= n_ltf:
            continue

        direction = htf_sig["direction"]
        for j in range(ltf_start, min(ltf_start + max_wait, n_ltf)):
            if j in sigs:
                continue
            atr_v = atr_ltf[j] if not np.isnan(atr_ltf[j]) else cl_ltf[j] * 0.01

            if entry_mode == "next_bar":
                sl_v = (cl_ltf[j] - atr_v * sl_atr_mult) if direction == "LONG" else (cl_ltf[j] + atr_v * sl_atr_mult)
                sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                            "default_sl": float(sl_v), "htf_bar": htf_bar}
                break

            elif entry_mode == "pullback_ema21":
                if np.isnan(ema21_ltf[j]):
                    continue
                if direction == "LONG" and lo_ltf[j] <= ema21_ltf[j] * 1.002 and cl_ltf[j] > ema21_ltf[j]:
                    sl_v = cl_ltf[j] - atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    break
                elif direction == "SHORT" and hi_ltf[j] >= ema21_ltf[j] * 0.998 and cl_ltf[j] < ema21_ltf[j]:
                    sl_v = cl_ltf[j] + atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    break

            elif entry_mode == "pullback_bb_mid":
                bb_m = ltf_data["bb_mid"]
                if np.isnan(bb_m[j]):
                    continue
                if direction == "LONG" and lo_ltf[j] <= bb_m[j] * 1.002 and cl_ltf[j] > bb_m[j]:
                    sl_v = cl_ltf[j] - atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    break
                elif direction == "SHORT" and hi_ltf[j] >= bb_m[j] * 0.998 and cl_ltf[j] < bb_m[j]:
                    sl_v = cl_ltf[j] + atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    break

    return sigs


# ===========================================================================
#  SL / SIMULATION ENGINE (from v9, slightly extended)
# ===========================================================================

def compute_sl(sl_cfg, d, entry, atr_v, hi, lo, i, default_sl):
    t = sl_cfg["type"]
    if t == "default":
        sl = default_sl
    elif t == "atr":
        sl = (entry - atr_v * sl_cfg["mult"]) if d == "LONG" else (entry + atr_v * sl_cfg["mult"])
    elif t == "structure":
        s = max(0, i - sl_cfg.get("n_bars", 5) + 1)
        sl = float(np.min(lo[s:i+1])) if d == "LONG" else float(np.max(hi[s:i+1]))
    else:
        sl = default_sl
    if d == "LONG"  and sl >= entry: sl = entry * 0.98
    if d == "SHORT" and sl <= entry: sl = entry * 1.02
    return float(sl)


def run_sim_tf(signals_by_sym, tf_arrays_by_sym, sl_cfg, tp_cfg,
               risk_pt=0.010, cap_frac=0.50, max_pos=4,
               start_frac=0.0, end_frac=1.0):
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    open_pos = {}
    acnt = 0
    completed = []

    max_n = 0
    for sym, arr in tf_arrays_by_sym.items():
        if arr is not None:
            max_n = max(max_n, arr["n"])
    if max_n == 0:
        return _empty_result()

    i_start = max(300, int(max_n * start_frac))
    i_end = int(max_n * end_frac)

    for i in range(i_start, i_end):
        for sym, arr in tf_arrays_by_sym.items():
            if arr is None or i >= arr["n"]:
                continue
            cl_i = arr["cl"][i]; hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            if sym in open_pos:
                pos = open_pos[sym]; d = pos["d"]; entry = pos["entry"]
                sl = pos["sl"]; r = pos["risk"]; q = pos["qty"]
                bars = i - pos["bar_i"]
                if d == "LONG":
                    pos["mfe"] = max(pos["mfe"], (hi_i - entry) / r)
                    pos["mae"] = max(pos["mae"], (entry - lo_i) / r)
                else:
                    pos["mfe"] = max(pos["mfe"], (entry - lo_i) / r)
                    pos["mae"] = max(pos["mae"], (hi_i - entry) / r)
                pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = False; tp_price = 0.0; exit_rsn = "sl_hit" if hit_sl else ""

                if not hit_sl:
                    tp_type = tp_cfg["type"]
                    if tp_type == "fixed_r":
                        if pnl_r >= tp_cfg["r"]:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_fixed"
                    elif tp_type == "trail_atr":
                        if pnl_r >= tp_cfg.get("trail_start_r", 1.5):
                            tm = tp_cfg.get("trail_mult", 2.5)
                            if d == "LONG":
                                pos["best"] = max(pos.get("best", entry), hi_i)
                                nt = pos["best"] - atr_i * tm
                                pos["tsl"] = max(pos.get("tsl", sl), nt)
                            else:
                                pos["best"] = min(pos.get("best", entry), lo_i)
                                nt = pos["best"] + atr_i * tm
                                ct = pos.get("tsl", 0)
                                pos["tsl"] = min(ct, nt) if ct > 0 else nt
                        eff = pos.get("tsl", sl)
                        if d == "LONG":
                            eff = max(sl, eff)
                            if lo_i <= eff: hit_sl = True; sl = eff; exit_rsn = "trail"
                        else:
                            eff = min(sl, eff) if eff > 0 else sl
                            if hi_i >= eff: hit_sl = True; sl = eff; exit_rsn = "trail"
                    elif tp_type == "partial_fixed":
                        r1 = tp_cfg["r1"]; r2 = tp_cfg["r2"]; frac = tp_cfg["frac"]
                        if not pos.get("partial") and pnl_r >= r1:
                            pq = q * frac
                            gp = ((cl_i - entry) if d == "LONG" else (entry - cl_i)) * pq
                            np_ = gp - cl_i * pq * (TAKER_FEE + SLIPPAGE)
                            balance += np_; pos["qty"] -= pq; q = pos["qty"]
                            pos["partial"] = True; pos["ppnl"] = np_
                            if d == "LONG" and entry > sl: pos["sl"] = entry; sl = entry
                            elif d == "SHORT" and entry < sl: pos["sl"] = entry; sl = entry
                        if pos.get("partial") and pnl_r >= r2:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_partial"

                if hit_sl or hit_tp:
                    if not exit_rsn: exit_rsn = "tp" if hit_tp else "sl"
                    ep = tp_price if hit_tp else sl
                    qo = pos["qty"]
                    gross = ((ep - entry) if d == "LONG" else (entry - ep)) * qo
                    fee = ep * qo * (TAKER_FEE + SLIPPAGE)
                    net = gross - fee
                    total_net = net + pos.get("ppnl", 0.0)
                    balance += net
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    completed.append({"sym": sym, "d": d, "exit_r": round(exit_r, 3),
                        "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                        "bars": bars, "won": total_net > 0, "pnl_usd": round(total_net, 4),
                        "reason": exit_rsn})
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            if acnt >= max_pos:
                continue
            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig = sym_sigs[i]
            new_sl = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                atr_i, arr["hi"], arr["lo"], i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue
            ab = balance * cap_frac
            ru = ab * risk_pt
            qty = ru / rd
            mg = sig["entry_price"] * qty / MAX_LEVERAGE
            if mg > ab * 0.4:
                qty = ab * 0.4 * MAX_LEVERAGE / sig["entry_price"]
            if qty * sig["entry_price"] < 5.0:
                continue
            ef = sig["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
            balance -= ef
            open_pos[sym] = {"d": sig["direction"], "entry": sig["entry_price"],
                "sl": new_sl, "risk": rd, "qty": qty, "bar_i": i,
                "mfe": 0.0, "mae": 0.0, "partial": False, "ppnl": 0.0}
            acnt += 1

    return _aggregate(completed, balance)


def _empty_result():
    return {"trades": 0, "wr_pct": 0, "pnl_pct": 0, "profit_factor": 0,
            "max_dd_pct": 0, "mfe_cap": 0, "giveback": 0, "avg_bars": 0,
            "exit_reasons": {}}


def _aggregate(completed, balance):
    if not completed:
        return _empty_result()
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    n = len(completed)
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / n * 100
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100
    mv = [t for t in completed if t["mfe"] >= 0.5]
    mfe_cap = float(np.mean([max(0, t["exit_r"]) / t["mfe"] for t in mv]) * 100) if mv else 0
    r1 = [t for t in completed if t["mfe"] >= 1.0]
    gb = len([t for t in r1 if t["exit_r"] < 0.5]) / len(r1) * 100 if r1 else 0
    reasons = defaultdict(int)
    for t in completed:
        reasons[t["reason"]] += 1
    bal = START_BALANCE; peak = START_BALANCE; max_dd = 0.0
    for t in completed:
        bal += t["pnl_usd"]; peak = max(peak, bal)
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 1), "pnl_pct": round(pnl_pct, 2),
        "profit_factor": round(min(pf, 99), 3), "max_dd_pct": round(max_dd, 2),
        "avg_exit_r": round(float(np.mean([t["exit_r"] for t in completed])), 3),
        "avg_mfe": round(float(np.mean([t["mfe"] for t in completed])), 3),
        "avg_mae": round(float(np.mean([t["mae"] for t in completed])), 3),
        "mfe_cap": round(mfe_cap, 1), "giveback": round(gb, 1),
        "avg_bars": round(float(np.mean([t["bars"] for t in completed])), 1),
        "exit_reasons": dict(reasons), "final_bal": round(balance, 2),
    }


def _score(r):
    if r["trades"] < 8: return -999.0
    pf  = min(r["profit_factor"], 5.0)
    mfe = r.get("mfe_cap", 0) / 100.0
    gb  = r.get("giveback", 0) / 100.0
    dd  = r["max_dd_pct"] / 100.0
    pnl = max(r["pnl_pct"], -50) / 100.0
    return 0.35*pf + 0.25*mfe - 0.10*gb - 0.15*dd + 0.15*pnl


# ===========================================================================
#  SL/TP combos (extended grid)
# ===========================================================================

SLTP_COMBOS = [
    ("atr1.5_2.0r", {"type": "atr", "mult": 1.5}, {"type": "fixed_r", "r": 2.0}),
    ("atr1.5_2.5r", {"type": "atr", "mult": 1.5}, {"type": "fixed_r", "r": 2.5}),
    ("atr1.5_3.0r", {"type": "atr", "mult": 1.5}, {"type": "fixed_r", "r": 3.0}),
    ("atr2.0_2.5r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 2.5}),
    ("atr2.0_3.0r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 3.0}),
    ("atr2.0_3.5r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 3.5}),
    ("atr2.0_4.0r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 4.0}),
    ("atr2.5_3.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.0}),
    ("atr2.5_3.5r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5}),
    ("atr2.5_4.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 4.0}),
    ("atr2.5_5.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 5.0}),
    ("atr3.0_3.5r", {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 3.5}),
    ("atr3.0_4.0r", {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 4.0}),
    ("atr3.0_5.0r", {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 5.0}),
    ("atr3.5_4.0r", {"type": "atr", "mult": 3.5}, {"type": "fixed_r", "r": 4.0}),
    ("atr3.5_5.0r", {"type": "atr", "mult": 3.5}, {"type": "fixed_r", "r": 5.0}),
    ("def_3.5r",    {"type": "default"},           {"type": "fixed_r", "r": 3.5}),
    ("def_4.0r",    {"type": "default"},           {"type": "fixed_r", "r": 4.0}),
    ("atr2.0_part1535", {"type": "atr", "mult": 2.0}, {"type": "partial_fixed", "r1": 1.5, "r2": 3.5, "frac": 0.5}),
    ("atr2.5_part2040", {"type": "atr", "mult": 2.5}, {"type": "partial_fixed", "r1": 2.0, "r2": 4.0, "frac": 0.5}),
    ("atr2.5_trail25",  {"type": "atr", "mult": 2.5}, {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 2.5}),
    ("atr3.0_trail30",  {"type": "atr", "mult": 3.0}, {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 3.0}),
]


# ===========================================================================
#  RUNNER HELPERS
# ===========================================================================

def build_mtf_sigs_and_arrays(all_data, htf_key, ltf_key, htf_tf,
                               gen_fn, sig_params, entry_params,
                               daily_filter=False):
    """Generate MTF signals for all symbols. Returns (sigs_by_sym, ltf_arrays)."""
    ltf_arrays = {}
    all_sigs = {}
    for sym, d in all_data.items():
        htf_arr = prepare_tf_arrays(d.get(htf_key))
        ltf_arr = prepare_tf_arrays(d.get(ltf_key))
        if htf_arr is None or ltf_arr is None:
            continue
        ltf_arrays[sym] = ltf_arr
        htf_trend = prepare_tf_arrays(d.get("df4h")) if htf_tf in ("1h",) else prepare_tf_arrays(d.get("df1h"))
        htf_sigs = gen_fn(htf_arr, htf_trend, sig_params)
        daily_d = prepare_daily_trend(d.get("df1d")) if daily_filter else None
        mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, entry_params,
                                   daily_data=daily_d, use_daily_filter=daily_filter)
        all_sigs[sym] = mtf_sigs
    return all_sigs, ltf_arrays


def run_and_report(label, all_sigs, tf_arrays, sl_cfg, tp_cfg, **kwargs):
    total = sum(len(s) for s in all_sigs.values())
    res = run_sim_tf(all_sigs, tf_arrays, sl_cfg, tp_cfg, **kwargs)
    res["total_sigs"] = total
    res["score"] = _score(res)
    print("  %-40s sigs=%4d n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%"
          % (label, total, res["trades"], res["profit_factor"],
             res["pnl_pct"], res["max_dd_pct"]), flush=True)
    return res


def run_walk_forward(name, all_sigs, tf_arrays, sl_cfg, tp_cfg):
    train = run_sim_tf(all_sigs, tf_arrays, sl_cfg, tp_cfg,
                       risk_pt=0.010, cap_frac=0.60, max_pos=4,
                       start_frac=0.0, end_frac=0.67)
    test = run_sim_tf(all_sigs, tf_arrays, sl_cfg, tp_cfg,
                      risk_pt=0.010, cap_frac=0.60, max_pos=4,
                      start_frac=0.67, end_frac=1.0)
    passed = test["profit_factor"] >= 1.0 and test["pnl_pct"] > -5.0
    tag = "PASS" if passed else "FAIL"
    print("  WF %-35s train=%+6.1f%%(PF=%.2f n=%d) test=%+6.1f%%(PF=%.2f n=%d) [%s]"
          % (name, train["pnl_pct"], train["profit_factor"], train["trades"],
             test["pnl_pct"], test["profit_factor"], test["trades"], tag), flush=True)
    return {"name": name, "train_pf": train["profit_factor"],
            "test_pf": test["profit_factor"],
            "train_pnl": round(train["pnl_pct"], 2),
            "test_pnl": round(test["pnl_pct"], 2),
            "train_n": train["trades"], "test_n": test["trades"],
            "passed": passed, "tag": tag}


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("backtest_v10.py -- Deep MTF Optimization + Swing + 365d", flush=True)
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    t0 = _time.time()

    # ---- Phase 0: Load ----
    print("\n  Phase 0 -- Loading multi-TF data (365d preferred)...", flush=True)
    all_data = load_all_tf("365d")
    print("  Symbols loaded: %d" % len(all_data), flush=True)

    # Verify data lengths
    for sym in list(all_data.keys())[:3]:
        d = all_data[sym]
        lens = {k: len(d[k]) if d.get(k) is not None else 0 for k in ("df5","df15","df1h","df4h","df1d")}
        print("    %s: %s" % (sym.split("/")[0], lens), flush=True)

    has_5m = any(d.get("df5") is not None and len(d["df5"]) > 1000 for d in all_data.values())
    has_daily = any(d.get("df1d") is not None and len(d["df1d"]) > 200 for d in all_data.values())
    print("  5m data: %s, Daily data: %s" % (has_5m, has_daily), flush=True)

    all_results = {}

    # ==================================================================
    # Phase 1: Optimize MTF BBS 1h->5m (around v9 optimum)
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 1 -- Optimize MTF BBS 1h->5m pullback", flush=True)
    print("=" * 72, flush=True)

    bbs_1h_5m_grid = []
    for sq in (4, 6, 8, 10):
        for adx in (15, 18, 22):
            for vol in (0.8, 1.0):
                bbs_1h_5m_grid.append({
                    "id": "bbs1h5m_sq%d_adx%d_v%.1f" % (sq, adx, vol),
                    "sig_params": {"min_squeeze": sq, "adx_thresh": adx, "vol_mult": vol,
                                   "htf_filter": False, "headfake_filter": True},
                    "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
                })
    # Also vary wait bars and entry mode
    for wait in (10, 15, 25, 30):
        bbs_1h_5m_grid.append({
            "id": "bbs1h5m_w%d" % wait,
            "sig_params": {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0,
                           "htf_filter": False, "headfake_filter": True},
            "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": wait, "sl_atr_mult": 2.5},
        })
    bbs_1h_5m_grid.append({
        "id": "bbs1h5m_bb_pb",
        "sig_params": {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_bb_mid", "max_wait_bars": 20, "sl_atr_mult": 2.5},
    })
    # RSI filter variant
    bbs_1h_5m_grid.append({
        "id": "bbs1h5m_rsi",
        "sig_params": {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True, "rsi_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
    })

    p1_results = []
    if has_5m:
        for cfg_entry in bbs_1h_5m_grid:
            sigs, ltf_arr = build_mtf_sigs_and_arrays(
                all_data, "df1h", "df5", "1h",
                gen_bbs_signals_tf, cfg_entry["sig_params"], cfg_entry["entry_params"])
            if not ltf_arr:
                continue
            res = run_and_report(cfg_entry["id"], sigs, ltf_arr,
                                {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5},
                                risk_pt=0.010, cap_frac=0.60, max_pos=4)
            res["config_id"] = cfg_entry["id"]
            res["params"] = cfg_entry
            p1_results.append(res)
        p1_results.sort(key=lambda x: x["score"], reverse=True)
    else:
        print("  SKIP (no 5m data)", flush=True)
    all_results["p1_bbs1h5m"] = p1_results

    # ==================================================================
    # Phase 2: Optimize MTF BBS 1h->15m
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 2 -- Optimize MTF BBS 1h->15m", flush=True)
    print("=" * 72, flush=True)

    bbs_1h_15m_grid = []
    for sq in (4, 6, 8, 10):
        for adx in (15, 18, 22):
            for entry_m in ("next_bar", "pullback_ema21"):
                bbs_1h_15m_grid.append({
                    "id": "bbs1h15m_sq%d_a%d_%s" % (sq, adx, entry_m[:4]),
                    "sig_params": {"min_squeeze": sq, "adx_thresh": adx, "vol_mult": 1.0,
                                   "htf_filter": False, "headfake_filter": True},
                    "entry_params": {"entry_mode": entry_m, "max_wait_bars": 8, "sl_atr_mult": 2.5},
                })

    p2_results = []
    for cfg_entry in bbs_1h_15m_grid:
        sigs, ltf_arr = build_mtf_sigs_and_arrays(
            all_data, "df1h", "df15", "1h",
            gen_bbs_signals_tf, cfg_entry["sig_params"], cfg_entry["entry_params"])
        if not ltf_arr:
            continue
        res = run_and_report(cfg_entry["id"], sigs, ltf_arr,
                            {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5},
                            risk_pt=0.010, cap_frac=0.60, max_pos=4)
        res["config_id"] = cfg_entry["id"]
        res["params"] = cfg_entry
        p2_results.append(res)
    p2_results.sort(key=lambda x: x["score"], reverse=True)
    all_results["p2_bbs1h15m"] = p2_results

    # ==================================================================
    # Phase 3: Optimize MTF KC 1h->15m
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 3 -- Optimize MTF KC 1h->15m", flush=True)
    print("=" * 72, flush=True)

    kc_1h_15m_grid = []
    for sq in (4, 6, 8, 10):
        for adx in (12, 15, 18):
            for mom in (6, 8, 10):
                kc_1h_15m_grid.append({
                    "id": "kc1h15m_sq%d_a%d_m%d" % (sq, adx, mom),
                    "sig_params": {"min_squeeze": sq, "adx_min": adx, "vol_mult": 1.0,
                                   "mom_bars": mom, "sl_atr_mult": 2.0},
                    "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
                })
    # Pullback variant for top params
    for sq in (4, 6, 8):
        kc_1h_15m_grid.append({
            "id": "kc1h15m_sq%d_pb" % sq,
            "sig_params": {"min_squeeze": sq, "adx_min": 15, "vol_mult": 1.0,
                           "mom_bars": 8, "sl_atr_mult": 2.0},
            "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 8, "sl_atr_mult": 2.5},
        })

    p3_results = []
    for cfg_entry in kc_1h_15m_grid:
        sigs, ltf_arr = build_mtf_sigs_and_arrays(
            all_data, "df1h", "df15", "1h",
            gen_kc_squeeze_signals_tf, cfg_entry["sig_params"], cfg_entry["entry_params"])
        if not ltf_arr:
            continue
        res = run_and_report(cfg_entry["id"], sigs, ltf_arr,
                            {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5},
                            risk_pt=0.010, cap_frac=0.60, max_pos=4)
        res["config_id"] = cfg_entry["id"]
        res["params"] = cfg_entry
        p3_results.append(res)
    p3_results.sort(key=lambda x: x["score"], reverse=True)
    all_results["p3_kc1h15m"] = p3_results

    # ==================================================================
    # Phase 4: NEW Swing BBS 4h->1h and 4h->15m
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 4 -- Swing BBS 4h->1h/15m", flush=True)
    print("=" * 72, flush=True)

    swing_bbs_grid = []
    for ltf, ltf_key, wait in [("1h", "df1h", 6), ("15m", "df15", 15)]:
        for sq in (3, 4, 6, 8):
            for adx in (12, 15, 18):
                for entry_m in ("next_bar", "pullback_ema21"):
                    swing_bbs_grid.append({
                        "id": "sw_bbs4h%s_sq%d_a%d_%s" % (ltf, sq, adx, entry_m[:4]),
                        "htf_key": "df4h", "ltf_key": ltf_key, "htf_tf": "4h",
                        "sig_params": {"min_squeeze": sq, "adx_thresh": adx, "vol_mult": 1.0,
                                       "htf_filter": False, "headfake_filter": True},
                        "entry_params": {"entry_mode": entry_m, "max_wait_bars": wait, "sl_atr_mult": 3.0},
                    })

    p4_results = []
    for cfg_entry in swing_bbs_grid:
        sigs, ltf_arr = build_mtf_sigs_and_arrays(
            all_data, cfg_entry["htf_key"], cfg_entry["ltf_key"], cfg_entry["htf_tf"],
            gen_bbs_signals_tf, cfg_entry["sig_params"], cfg_entry["entry_params"])
        if not ltf_arr:
            continue
        res = run_and_report(cfg_entry["id"], sigs, ltf_arr,
                            {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 3.5},
                            risk_pt=0.010, cap_frac=0.50, max_pos=3)
        res["config_id"] = cfg_entry["id"]
        res["params"] = cfg_entry
        p4_results.append(res)
    p4_results.sort(key=lambda x: x["score"], reverse=True)
    all_results["p4_swing_bbs"] = p4_results

    # ==================================================================
    # Phase 5: NEW Swing KC 4h->1h/15m
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 5 -- Swing KC 4h->1h/15m", flush=True)
    print("=" * 72, flush=True)

    swing_kc_grid = []
    for ltf, ltf_key, wait in [("1h", "df1h", 6), ("15m", "df15", 15)]:
        for sq in (3, 4, 6, 8):
            for adx in (12, 15, 18):
                for mom in (4, 6, 8):
                    swing_kc_grid.append({
                        "id": "sw_kc4h%s_sq%d_a%d_m%d" % (ltf, sq, adx, mom),
                        "htf_key": "df4h", "ltf_key": ltf_key, "htf_tf": "4h",
                        "sig_params": {"min_squeeze": sq, "adx_min": adx, "vol_mult": 1.0,
                                       "mom_bars": mom, "sl_atr_mult": 2.5},
                        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 3.0},
                    })

    p5_results = []
    for cfg_entry in swing_kc_grid:
        sigs, ltf_arr = build_mtf_sigs_and_arrays(
            all_data, cfg_entry["htf_key"], cfg_entry["ltf_key"], cfg_entry["htf_tf"],
            gen_kc_squeeze_signals_tf, cfg_entry["sig_params"], cfg_entry["entry_params"])
        if not ltf_arr:
            continue
        res = run_and_report(cfg_entry["id"], sigs, ltf_arr,
                            {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 3.5},
                            risk_pt=0.010, cap_frac=0.50, max_pos=3)
        res["config_id"] = cfg_entry["id"]
        res["params"] = cfg_entry
        p5_results.append(res)
    p5_results.sort(key=lambda x: x["score"], reverse=True)
    all_results["p5_swing_kc"] = p5_results

    # ==================================================================
    # Phase 6: Global trend filter test
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 6 -- Global Trend Filter (Daily EMA200)", flush=True)
    print("=" * 72, flush=True)

    p6_results = {}
    if has_daily:
        # Test daily filter on best configs from each phase
        best_configs = []
        if p1_results:
            best_configs.append(("BBS_1h->5m_daily", "df1h", "df5", "1h",
                                 gen_bbs_signals_tf,
                                 p1_results[0]["params"]["sig_params"],
                                 p1_results[0]["params"]["entry_params"]))
        if p2_results:
            best_configs.append(("BBS_1h->15m_daily", "df1h", "df15", "1h",
                                 gen_bbs_signals_tf,
                                 p2_results[0]["params"]["sig_params"],
                                 p2_results[0]["params"]["entry_params"]))
        if p3_results:
            best_configs.append(("KC_1h->15m_daily", "df1h", "df15", "1h",
                                 gen_kc_squeeze_signals_tf,
                                 p3_results[0]["params"]["sig_params"],
                                 p3_results[0]["params"]["entry_params"]))
        if p4_results and p4_results[0]["profit_factor"] > 1.0:
            best_configs.append(("SwBBS_4h_daily", p4_results[0]["params"]["htf_key"],
                                 p4_results[0]["params"]["ltf_key"],
                                 p4_results[0]["params"]["htf_tf"],
                                 gen_bbs_signals_tf,
                                 p4_results[0]["params"]["sig_params"],
                                 p4_results[0]["params"]["entry_params"]))
        if p5_results and p5_results[0]["profit_factor"] > 1.0:
            best_configs.append(("SwKC_4h_daily", p5_results[0]["params"]["htf_key"],
                                 p5_results[0]["params"]["ltf_key"],
                                 p5_results[0]["params"]["htf_tf"],
                                 gen_kc_squeeze_signals_tf,
                                 p5_results[0]["params"]["sig_params"],
                                 p5_results[0]["params"]["entry_params"]))

        for label, htf_k, ltf_k, htf_tf, gen_fn, sig_p, entry_p in best_configs:
            # Without daily filter
            sigs_no, arr_no = build_mtf_sigs_and_arrays(
                all_data, htf_k, ltf_k, htf_tf, gen_fn, sig_p, entry_p, daily_filter=False)
            res_no = run_and_report(label + "_NO_DAILY", sigs_no, arr_no,
                                   {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5},
                                   risk_pt=0.010, cap_frac=0.60, max_pos=4)
            # With daily filter
            sigs_d, arr_d = build_mtf_sigs_and_arrays(
                all_data, htf_k, ltf_k, htf_tf, gen_fn, sig_p, entry_p, daily_filter=True)
            res_d = run_and_report(label + "_DAILY", sigs_d, arr_d,
                                  {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5},
                                  risk_pt=0.010, cap_frac=0.60, max_pos=4)
            p6_results[label] = {"no_daily": res_no, "daily": res_d,
                                 "improvement": res_d["profit_factor"] - res_no["profit_factor"]}
    else:
        print("  SKIP (no daily data)", flush=True)
    all_results["p6_daily_filter"] = p6_results

    # ==================================================================
    # Phase 7: SL/TP deep optimization for top configs
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 7 -- SL/TP Deep Optimization", flush=True)
    print("=" * 72, flush=True)

    p7_results = {}

    # Gather top configs (best from each phase with PF > 1.1)
    top_configs = []
    for phase_name, phase_res, htf_k, ltf_k, htf_tf, gen_fn in [
        ("BBS_1h5m",  p1_results, "df1h", "df5",  "1h", gen_bbs_signals_tf),
        ("BBS_1h15m", p2_results, "df1h", "df15", "1h", gen_bbs_signals_tf),
        ("KC_1h15m",  p3_results, "df1h", "df15", "1h", gen_kc_squeeze_signals_tf),
    ]:
        if phase_res and phase_res[0]["profit_factor"] > 1.1:
            top_configs.append((phase_name, phase_res[0], htf_k, ltf_k, htf_tf, gen_fn))

    # Add best swing configs
    if p4_results and p4_results[0]["profit_factor"] > 1.1:
        best4 = p4_results[0]
        top_configs.append(("SwBBS4h", best4, best4["params"]["htf_key"],
                           best4["params"]["ltf_key"], best4["params"]["htf_tf"],
                           gen_bbs_signals_tf))
    if p5_results and p5_results[0]["profit_factor"] > 1.1:
        best5 = p5_results[0]
        top_configs.append(("SwKC4h", best5, best5["params"]["htf_key"],
                           best5["params"]["ltf_key"], best5["params"]["htf_tf"],
                           gen_kc_squeeze_signals_tf))

    for name, best_r, htf_k, ltf_k, htf_tf, gen_fn in top_configs:
        # Check if daily filter helps
        use_daily = False
        if name in p6_results and p6_results[name].get("improvement", 0) > 0.05:
            use_daily = True
            print("  [%s] Using daily filter (improvement %.2f)" % (name, p6_results[name]["improvement"]), flush=True)

        sigs, arr = build_mtf_sigs_and_arrays(
            all_data, htf_k, ltf_k, htf_tf, gen_fn,
            best_r["params"]["sig_params"], best_r["params"]["entry_params"],
            daily_filter=use_daily)
        if not arr:
            continue

        print("  SL/TP opt: %s" % name, flush=True)
        combo_results = []
        for combo_name, sl_c, tp_c in SLTP_COMBOS:
            res = run_sim_tf(sigs, arr, sl_c, tp_c,
                             risk_pt=0.010, cap_frac=0.60, max_pos=4)
            res["sltp_id"] = combo_name
            res["score"] = _score(res)
            combo_results.append(res)
            print("    %-20s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%"
                  % (combo_name, res["trades"], res["profit_factor"],
                     res["pnl_pct"], res["max_dd_pct"]), flush=True)
        combo_results.sort(key=lambda x: x["score"], reverse=True)
        p7_results[name] = combo_results
    all_results["p7_sltp"] = {k: v[:5] for k, v in p7_results.items()}

    # ==================================================================
    # Phase 8: Walk-Forward on 365d
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 8 -- Walk-Forward Validation (365d)", flush=True)
    print("=" * 72, flush=True)

    wf_results = []
    for name, best_r, htf_k, ltf_k, htf_tf, gen_fn in top_configs:
        use_daily = name in p6_results and p6_results[name].get("improvement", 0) > 0.05
        sigs, arr = build_mtf_sigs_and_arrays(
            all_data, htf_k, ltf_k, htf_tf, gen_fn,
            best_r["params"]["sig_params"], best_r["params"]["entry_params"],
            daily_filter=use_daily)
        if not arr:
            continue

        # Best SL/TP from Phase 7
        best_sl = {"type": "atr", "mult": 2.5}
        best_tp = {"type": "fixed_r", "r": 3.5}
        if name in p7_results and p7_results[name]:
            best_combo = p7_results[name][0]
            for cn, sl_c, tp_c in SLTP_COMBOS:
                if cn == best_combo["sltp_id"]:
                    best_sl = sl_c; best_tp = tp_c; break

        wfr = run_walk_forward("WF_%s" % name, sigs, arr, best_sl, best_tp)
        wfr["daily_filter"] = use_daily
        wfr["best_sltp"] = p7_results.get(name, [{}])[0].get("sltp_id", "default")
        wf_results.append(wfr)

        # Also WF with default SL/TP for comparison
        if best_sl != {"type": "atr", "mult": 2.5} or best_tp["r"] != 3.5:
            wfr2 = run_walk_forward("WF_%s_defSLTP" % name, sigs, arr,
                                    {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5})
            wf_results.append(wfr2)

    all_results["p8_walkforward"] = wf_results

    # ==================================================================
    # Phase 9: Portfolio simulation
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 9 -- Portfolio Simulation", flush=True)
    print("=" * 72, flush=True)

    passed_wf = [w for w in wf_results if w["passed"] and "defSLTP" not in w["name"]]
    print("  Configs that passed WF: %d" % len(passed_wf), flush=True)
    for w in passed_wf:
        print("    %s  test_PF=%.2f test_PnL=%+.1f%%" % (w["name"], w["test_pf"], w["test_pnl"]), flush=True)

    if passed_wf:
        # Calculate weighted portfolio
        n_strats = len(passed_wf)
        equal_alloc = 1.0 / n_strats
        w_pnl = 0; w_dd = 0
        for name, best_r, htf_k, ltf_k, htf_tf, gen_fn in top_configs:
            matching_wf = [w for w in passed_wf if name in w["name"]]
            if not matching_wf:
                continue
            # Re-run full sim for accurate PnL
            use_daily = name in p6_results and p6_results[name].get("improvement", 0) > 0.05
            sigs, arr = build_mtf_sigs_and_arrays(
                all_data, htf_k, ltf_k, htf_tf, gen_fn,
                best_r["params"]["sig_params"], best_r["params"]["entry_params"],
                daily_filter=use_daily)
            if not arr:
                continue
            best_sl = {"type": "atr", "mult": 2.5}
            best_tp = {"type": "fixed_r", "r": 3.5}
            if name in p7_results and p7_results[name]:
                for cn, sl_c, tp_c in SLTP_COMBOS:
                    if cn == p7_results[name][0]["sltp_id"]:
                        best_sl = sl_c; best_tp = tp_c; break
            res = run_sim_tf(sigs, arr, best_sl, best_tp,
                             risk_pt=0.010, cap_frac=0.60, max_pos=4)
            w_pnl += equal_alloc * res["pnl_pct"]
            w_dd += equal_alloc * res["max_dd_pct"]
            print("  %-30s alloc=%d%%  n=%3d  PF=%.2f  PnL=%+7.1f%%  DD=%4.1f%%"
                  % (name, int(equal_alloc*100), res["trades"], res["profit_factor"],
                     res["pnl_pct"], res["max_dd_pct"]), flush=True)

        final_bal = START_BALANCE * (1 + w_pnl / 100)
        print("\n  Portfolio PnL:     %+.1f%%" % w_pnl, flush=True)
        print("  Portfolio DD:      %.1f%% (weighted)" % w_dd, flush=True)
        print("  $1000 -> $%.0f" % final_bal, flush=True)
        data_days = 365  # or actual from data
        print("  Annualized:        %+.0f%%" % (w_pnl / data_days * 365), flush=True)
        all_results["p9_portfolio"] = {
            "pnl_pct": round(w_pnl, 2), "dd_pct": round(w_dd, 2),
            "final_bal": round(final_bal, 2), "n_strategies": n_strats}

    # ==================================================================
    # Phase 10: Final Report
    # ==================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 10 -- FINAL REPORT", flush=True)
    print("=" * 72, flush=True)

    # Summary per phase
    for phase_name, phase_key in [
        ("MTF BBS 1h->5m", "p1_bbs1h5m"),
        ("MTF BBS 1h->15m", "p2_bbs1h15m"),
        ("MTF KC 1h->15m", "p3_kc1h15m"),
        ("Swing BBS 4h", "p4_swing_bbs"),
        ("Swing KC 4h", "p5_swing_kc"),
    ]:
        results = all_results.get(phase_key, [])
        if not results:
            print("  %-25s NO DATA" % phase_name, flush=True)
            continue
        best = results[0]
        tag = "VIABLE" if best["profit_factor"] >= 1.15 else ("MARGINAL" if best["profit_factor"] >= 1.05 else "DEAD")
        print("  %-25s [%s] best=%-25s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%"
              % (phase_name, tag, best["config_id"], best["trades"],
                 best["profit_factor"], best["pnl_pct"], best["max_dd_pct"]),
              flush=True)
        for r in results[1:3]:
            print("    %-30s n=%3d PF=%.2f PnL=%+7.1f%%" % (
                r["config_id"], r["trades"], r["profit_factor"], r["pnl_pct"]), flush=True)

    print("\n  == Daily Filter Impact ==", flush=True)
    for label, data in p6_results.items():
        nd = data["no_daily"]; d = data["daily"]
        better = "BETTER" if d["profit_factor"] > nd["profit_factor"] else "WORSE"
        print("    %-25s NO_DAILY PF=%.2f -> DAILY PF=%.2f [%s]"
              % (label, nd["profit_factor"], d["profit_factor"], better), flush=True)

    print("\n  == Walk-Forward Results ==", flush=True)
    for w in wf_results:
        print("    %-35s train=%+6.1f%%(PF=%.2f) test=%+6.1f%%(PF=%.2f) [%s]"
              % (w["name"], w["train_pnl"], w["train_pf"],
                 w["test_pnl"], w["test_pf"], w["tag"]), flush=True)

    if p7_results:
        print("\n  == Best SL/TP per Strategy ==", flush=True)
        for name, combos in p7_results.items():
            if combos:
                print("    %-25s %s PF=%.2f PnL=%+.1f%%" % (
                    name, combos[0]["sltp_id"], combos[0]["profit_factor"],
                    combos[0]["pnl_pct"]), flush=True)

    # ---- Save ----
    out_path = RESULTS_DIR / ("backtest_v10_%s.json" % ts)
    # Clean params for JSON serialization
    clean = {}
    for k, v in all_results.items():
        if isinstance(v, list):
            clean[k] = [{kk: vv for kk, vv in r.items() if kk != "params"} for r in v[:10]]
        elif isinstance(v, dict):
            clean[k] = v
        else:
            clean[k] = v
    clean["metadata"] = {"timestamp": ts, "symbols": len(all_data),
                          "runtime_sec": round(_time.time() - t0, 1)}
    with open(out_path, "w") as f:
        json.dump(clean, f, indent=2, default=str)
    print("\n  Results saved:", out_path, flush=True)
    print("\nTotal runtime: %d min %d sec" % divmod(int(_time.time() - t0), 60), flush=True)


if __name__ == "__main__":
    main()
