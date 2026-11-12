"""
backtest_v11.py  --  v11 Filter Impact Analysis (2026-04-14)

Goal: Test all v11 live-system improvements on the 3 best v10 configs.
      Compare each filter individually and combined vs v10 baseline.

Filters tested:
  1. Symbol blacklist (ARB, OP, POL removed)
  2. Time-of-day gate UTC 07-20 (EU/US overlap)
  3. LTF bar confirmation (entry bar body must align with direction)
  4. Break-even SL at 1.5R (move SL to entry + buffer)
  5. SL cooldown (4h / 24h after 2 consecutive SLs)
  6. VWAP filter for BBS_1H_5M pullback (EMA21 touch above daily VWAP for LONG)

v10 baselines (from saved_configs):
  BBS 1h->5m:  PF=9.66  trades=93   WR=77.4%  DD=1.9%
  BBS 1h->15m: PF=3.92  trades=288  WR=59.7%  DD=4.1%
  KC  1h->15m: PF=5.76  trades=83   WR=68.7%  DD=2.0%

Decision rule: Keep v11 combined if:
  - PF doesn't drop below v10 PF * 0.85
  - Trades don't drop below v10 trades * 0.60
  - Max DD doesn't worsen by more than 2%
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

from multi_agent.shared.regime_detector import precompute_1h_indicators, precompute_4h_indicators
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
import multi_agent.config as cfg

logger = logging.getLogger("v11")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

# All 20 symbols (same as v10)
SYMBOLS_ALL = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

# v11 blacklist
BLACKLIST = {"ARB/USDT:USDT", "OP/USDT:USDT", "POL/USDT:USDT"}
SYMBOLS_V11 = [s for s in SYMBOLS_ALL if s not in BLACKLIST]

START_BALANCE = 1000.0
TAKER_FEE  = 0.00055
SLIPPAGE   = 0.0005
MAX_LEVERAGE = 50

# v10 best params (from saved configs + MEMORY.md)
V10_CONFIGS = {
    "bbs_1h_5m": {
        "sig_params": {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 1.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 3.0},
        "daily_filter": True,   # BBS_1H_5M: daily EMA200 filter ON
        "htf_key": "df1h", "ltf_key": "df5", "htf_tf": "1h",
        "gen_fn": "bbs",
        "cap_frac": 0.35, "max_pos": 4,
    },
    "bbs_1h_15m": {
        "sig_params": {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 1.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
        "htf_key": "df1h", "ltf_key": "df15", "htf_tf": "1h",
        "gen_fn": "bbs",
        "cap_frac": 0.30, "max_pos": 4,
    },
    "kc_1h_15m": {
        "sig_params": {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                       "mom_bars": 6, "sl_atr_mult": 1.5},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 1.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.0},
        "daily_filter": False,
        "htf_key": "df1h", "ltf_key": "df15", "htf_tf": "1h",
        "gen_fn": "kc",
        "cap_frac": 0.30, "max_pos": 4,
    },
}

# v10 published results for comparison
V10_PUBLISHED = {
    "bbs_1h_5m":  {"pf": 9.66, "trades": 93,  "wr": 77.4, "dd": 1.9, "pnl": 216.1},
    "bbs_1h_15m": {"pf": 3.92, "trades": 288, "wr": 59.7, "dd": 4.1, "pnl": 592.5},
    "kc_1h_15m":  {"pf": 5.76, "trades": 83,  "wr": 68.7, "dd": 2.0, "pnl": 93.0},
}

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


def load_all_tf(symbols, days_tag="365d"):
    all_data = {}
    for sym in symbols:
        d = {}
        d["df5"]  = _load(sym, "5m", days_tag)
        d["df15"] = _load(sym, "15m", days_tag)
        d["df1h"] = _load(sym, "1h", days_tag)
        d["df4h"] = _load(sym, "4h", days_tag)
        d["df1d"] = _load(sym, "1d", days_tag)

        if d["df15"] is None or len(d["df15"]) < 300:
            continue
        if d["df1h"] is None or len(d["df1h"]) < 210:
            continue

        d["ind_1h"] = precompute_1h_indicators(d["df1h"])
        d["ts1h"] = d["df1h"]["timestamp"].astype(int).to_numpy()
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

    # Daily VWAP (reset at each UTC day boundary)
    vwap = _compute_daily_vwap(ts, cl, hi, lo, vo)

    return {
        "df": df, "cl": cl, "hi": hi, "lo": lo, "op": op, "vo": vo, "ts": ts, "n": n,
        "atr": atr14, "adx": adx14, "rsi": rsi14,
        "ema200": ema200, "ema50": ema50, "ema21": ema21,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
        "kc_upper": kc_upper, "kc_lower": kc_lower, "kc_mid": kc_mid,
        "kc_squeeze": kc_squeeze, "kc_sq_len": kc_sq_len,
        "vol_ratio": vol_ratio, "vwap": vwap,
    }


def _compute_daily_vwap(ts, cl, hi, lo, vo):
    """Compute daily VWAP, resetting at UTC midnight."""
    n = len(ts)
    vwap = np.full(n, np.nan)
    typical = (hi + lo + cl) / 3.0
    day_ms = 86_400_000
    cum_tp_vol = 0.0
    cum_vol = 0.0
    current_day = -1
    for i in range(n):
        day = ts[i] // day_ms
        if day != current_day:
            cum_tp_vol = 0.0
            cum_vol = 0.0
            current_day = day
        cum_tp_vol += typical[i] * vo[i]
        cum_vol += vo[i]
        if cum_vol > 0:
            vwap[i] = cum_tp_vol / cum_vol
    return vwap


def prepare_daily_trend(df_daily):
    if df_daily is None or len(df_daily) < 210:
        return None
    cl = df_daily["close"].astype(float).to_numpy()
    ts = df_daily["timestamp"].astype(int).to_numpy()
    ema200 = _ema(cl, 200)
    return {"cl": cl, "ts": ts, "ema200": ema200, "n": len(cl)}


def get_daily_trend(daily_data, bar_ts):
    if daily_data is None:
        return None
    idx = int(np.searchsorted(daily_data["ts"], bar_ts, side="right")) - 1
    if idx < 200 or np.isnan(daily_data["ema200"][idx]):
        return None
    return "LONG" if daily_data["cl"][idx] > daily_data["ema200"][idx] else "SHORT"


# ===========================================================================
#  SIGNAL GENERATORS (same as v10)
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

    cl = tf_data["cl"]; hi = tf_data["hi"]; lo = tf_data["lo"]
    n = tf_data["n"]
    adx = tf_data["adx"]
    vol_r = tf_data["vol_ratio"]

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


def gen_kc_squeeze_signals_tf(tf_data, htf_data, params):
    if tf_data is None:
        return {}

    min_sq  = params.get("min_squeeze", 10)
    adx_min = params.get("adx_min", 15.0)
    vol_mult = params.get("vol_mult", 1.0)
    mom_bars = params.get("mom_bars", 12)
    sl_mult  = params.get("sl_atr_mult", 2.0)

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
            sl_v = cl[i] - atr_v * sl_mult
        else:
            direction = "SHORT"
            if not np.isnan(bb_lower[i]) and not np.isnan(kc_lower[i]):
                if bb_lower[i] >= kc_lower[i]:
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

    entry_mode  = params.get("entry_mode", "pullback_ema21")
    max_wait    = params.get("max_wait_bars", 12)
    sl_atr_mult = params.get("sl_atr_mult", 2.5)

    cl_ltf = ltf_data["cl"]; hi_ltf = ltf_data["hi"]; lo_ltf = ltf_data["lo"]
    n_ltf = ltf_data["n"]
    ts_ltf = ltf_data["ts"]; ts_htf = htf_data["ts"]
    atr_ltf = ltf_data["atr"]
    ema21_ltf = ltf_data["ema21"]

    sigs = {}

    for htf_bar, htf_sig in htf_sigs.items():
        htf_ts = htf_data["ts"][htf_bar]

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
                sl_v = (cl_ltf[j] - atr_v * sl_atr_mult) if direction == "LONG" else \
                       (cl_ltf[j] + atr_v * sl_atr_mult)
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

    return sigs


# ===========================================================================
#  SL COMPUTATION
# ===========================================================================

def compute_sl(sl_cfg, d, entry, atr_v, hi, lo, i, default_sl):
    t = sl_cfg["type"]
    if t == "default":
        sl = default_sl
    elif t == "atr":
        sl = (entry - atr_v * sl_cfg["mult"]) if d == "LONG" else (entry + atr_v * sl_cfg["mult"])
    else:
        sl = default_sl
    if d == "LONG"  and sl >= entry: sl = entry * 0.98
    if d == "SHORT" and sl <= entry: sl = entry * 1.02
    return float(sl)


# ===========================================================================
#  SIMULATION ENGINE v11 (with optional filter flags)
# ===========================================================================

def run_sim_v11(signals_by_sym, tf_arrays_by_sym, sl_cfg, tp_cfg,
                risk_pt=0.010, cap_frac=0.50, max_pos=4,
                start_frac=0.0, end_frac=1.0,
                # v11 filter flags
                use_time_gate=False,
                use_ltf_confirm=False,
                use_break_even=False, be_r=1.5, be_buffer=0.001,
                use_cooldown=False,
                cooldown_bars_4h=16,   # bars equivalent to 4h on 15m (16 x 15m = 4h)
                cooldown_bars_5m=48,   # bars equivalent to 4h on 5m  (48 x 5m = 4h)
                sl_streak_limit=2,
                cooldown_extended_factor=6,  # extended = 6x normal (24h)
                use_vwap=False,
                ):
    """Extended simulation engine with v11 filters."""
    balance = START_BALANCE
    peak = START_BALANCE
    open_pos: dict = {}
    acnt = 0
    completed = []

    # v11 per-symbol state
    cooldown_until: dict[str, int] = {}
    sl_streak: dict[str, int] = {}

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

            # ---- Position management ----
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

                # Break-even SL
                if use_break_even and not pos.get("be_applied", False):
                    if pnl_r >= be_r:
                        if d == "LONG":
                            new_be = entry * (1.0 + be_buffer)
                            if new_be > pos["sl"]:
                                pos["sl"] = new_be; sl = new_be
                        else:
                            new_be = entry * (1.0 - be_buffer)
                            if new_be < pos["sl"]:
                                pos["sl"] = new_be; sl = new_be
                        pos["be_applied"] = True

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

                if hit_sl or hit_tp:
                    if not exit_rsn: exit_rsn = "tp" if hit_tp else "sl"
                    ep = tp_price if hit_tp else sl
                    qo = pos["qty"]
                    gross = ((ep - entry) if d == "LONG" else (entry - ep)) * qo
                    fee = ep * qo * (TAKER_FEE + SLIPPAGE)
                    net = gross - fee
                    balance += net
                    peak = max(peak, balance)
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    completed.append({"sym": sym, "d": d, "exit_r": round(exit_r, 3),
                        "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                        "bars": bars, "won": net > 0, "pnl_usd": round(net, 4),
                        "reason": exit_rsn})
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]

                    # Cooldown tracking
                    if use_cooldown:
                        if hit_sl and not hit_tp:
                            streak = sl_streak.get(sym, 0) + 1
                            sl_streak[sym] = streak
                            # Determine cooldown duration
                            n_bars = arr["n"]
                            # Estimate bar size: if many bars (~52560/y for 5m vs ~17520 for 15m)
                            est_bars_per_day = n_bars / 365.0 if n_bars > 1000 else 96
                            cd_normal = int(est_bars_per_day / 6)   # 4h worth of bars
                            if streak >= sl_streak_limit:
                                cd = cd_normal * cooldown_extended_factor  # 24h
                            else:
                                cd = cd_normal
                            cooldown_until[sym] = i + cd
                        else:
                            sl_streak[sym] = 0
                continue

            # ---- Entry logic ----
            if acnt >= max_pos:
                continue

            # Cooldown gate
            if use_cooldown and sym in cooldown_until and i < cooldown_until[sym]:
                continue

            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig = sym_sigs[i]
            direction = sig["direction"]

            # Time-of-day gate UTC 07-20
            if use_time_gate:
                ts_ms = int(arr["ts"][i])
                hour_utc = (ts_ms // 3_600_000) % 24
                if not (7 <= hour_utc < 20):
                    continue

            # LTF bar confirmation (entry bar body aligns with direction)
            if use_ltf_confirm:
                bar_body = float(arr["cl"][i]) - float(arr["op"][i])
                if direction == "LONG" and bar_body <= 0:
                    continue
                if direction == "SHORT" and bar_body >= 0:
                    continue

            # VWAP filter (for pullback entries: EMA21 touch must be above VWAP for LONG)
            if use_vwap:
                vwap_v = arr["vwap"][i]
                if not np.isnan(vwap_v):
                    if direction == "LONG" and arr["cl"][i] < vwap_v:
                        continue
                    if direction == "SHORT" and arr["cl"][i] > vwap_v:
                        continue

            new_sl = compute_sl(sl_cfg, direction, sig["entry_price"],
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
            open_pos[sym] = {"d": direction, "entry": sig["entry_price"],
                "sl": new_sl, "risk": rd, "qty": qty, "bar_i": i,
                "mfe": 0.0, "mae": 0.0, "be_applied": False}
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


# ===========================================================================
#  BUILD SIGNALS HELPER
# ===========================================================================

def build_sigs(cfg_name, all_data, filter_name="", extra_entry_params=None):
    """Build MTF signals for a named v10 config."""
    c = V10_CONFIGS[cfg_name]
    ltf_arrays = {}
    all_sigs = {}
    for sym, d in all_data.items():
        htf_arr = prepare_tf_arrays(d.get(c["htf_key"]))
        ltf_arr = prepare_tf_arrays(d.get(c["ltf_key"]))
        if htf_arr is None or ltf_arr is None:
            continue
        ltf_arrays[sym] = ltf_arr

        gen_fn = gen_bbs_signals_tf if c["gen_fn"] == "bbs" else gen_kc_squeeze_signals_tf
        htf_trend = prepare_tf_arrays(d.get("df4h")) if c["htf_tf"] == "1h" else prepare_tf_arrays(d.get("df1h"))
        htf_sigs = gen_fn(htf_arr, htf_trend, c["sig_params"])

        daily_d = prepare_daily_trend(d.get("df1d")) if c["daily_filter"] else None
        entry_p = dict(c["entry_params"])
        if extra_entry_params:
            entry_p.update(extra_entry_params)

        mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, entry_p,
                                   daily_data=daily_d, use_daily_filter=c["daily_filter"])
        all_sigs[sym] = mtf_sigs
    return all_sigs, ltf_arrays


def run_variant(label, cfg_name, all_data, **filter_kwargs):
    c = V10_CONFIGS[cfg_name]
    sigs, arrs = build_sigs(cfg_name, all_data)
    res = run_sim_v11(sigs, arrs, c["sl_cfg"], c["tp_cfg"],
                      risk_pt=0.010, cap_frac=c["cap_frac"], max_pos=c["max_pos"],
                      **filter_kwargs)
    pub = V10_PUBLISHED[cfg_name]
    delta_pf = res["profit_factor"] - pub["pf"]
    delta_n  = res["trades"] - pub["trades"]
    print("  %-45s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%  [dPF=%+.2f  dN=%+d]"
          % (label, res["trades"], res["profit_factor"],
             res["pnl_pct"], res["max_dd_pct"], delta_pf, delta_n), flush=True)
    return res


def run_walk_forward(name, cfg_name, all_data, **filter_kwargs):
    c = V10_CONFIGS[cfg_name]
    sigs, arrs = build_sigs(cfg_name, all_data)
    train = run_sim_v11(sigs, arrs, c["sl_cfg"], c["tp_cfg"],
                        risk_pt=0.010, cap_frac=c["cap_frac"], max_pos=c["max_pos"],
                        start_frac=0.0, end_frac=0.67, **filter_kwargs)
    test = run_sim_v11(sigs, arrs, c["sl_cfg"], c["tp_cfg"],
                       risk_pt=0.010, cap_frac=c["cap_frac"], max_pos=c["max_pos"],
                       start_frac=0.67, end_frac=1.0, **filter_kwargs)
    passed = test["profit_factor"] >= 1.0 and test["pnl_pct"] > -5.0
    tag = "PASS" if passed else "FAIL"
    print("  WF %-38s train=%+6.1f%%(PF=%.2f n=%d) test=%+6.1f%%(PF=%.2f n=%d) [%s]"
          % (name, train["pnl_pct"], train["profit_factor"], train["trades"],
             test["pnl_pct"], test["profit_factor"], test["trades"], tag), flush=True)
    return {"name": name, "train_pf": train["profit_factor"],
            "test_pf": test["profit_factor"],
            "train_pnl": round(train["pnl_pct"], 2), "test_pnl": round(test["pnl_pct"], 2),
            "train_n": train["trades"], "test_n": test["trades"],
            "passed": passed, "tag": tag}


# ===========================================================================
#  DECISION HELPER
# ===========================================================================

def evaluate_v11(cfg_name, baseline, v11_combined):
    pub = V10_PUBLISHED[cfg_name]
    min_pf = pub["pf"] * 0.85
    min_n  = pub["trades"] * 0.60
    max_dd = pub["dd"] + 2.0

    ok_pf = v11_combined["profit_factor"] >= min_pf
    ok_n  = v11_combined["trades"] >= min_n
    ok_dd = v11_combined["max_dd_pct"] <= max_dd

    verdict = "KEEP v11" if (ok_pf and ok_n and ok_dd) else "REVERT to v10"
    print("  Decision %-15s: PF%.2f>=%.2f[%s]  n%d>=%d[%s]  DD%.1f%%<=%.1f%%[%s]  => %s"
          % (cfg_name,
             v11_combined["profit_factor"], min_pf, "OK" if ok_pf else "FAIL",
             v11_combined["trades"], int(min_n), "OK" if ok_n else "FAIL",
             v11_combined["max_dd_pct"], max_dd, "OK" if ok_dd else "FAIL",
             verdict), flush=True)
    return verdict, ok_pf and ok_n and ok_dd


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("backtest_v11.py -- v11 Filter Impact Analysis (365d)", flush=True)
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    t0 = _time.time()

    # ---- Load data (all 20 symbols for baseline, 17 for blacklist variants) ----
    print("\n  Loading data (365d)...", flush=True)
    all_data_full = load_all_tf(SYMBOLS_ALL, "365d")
    all_data_bl   = {k: v for k, v in all_data_full.items() if k not in BLACKLIST}
    print("  Full: %d symbols  |  Blacklisted: %d symbols" % (len(all_data_full), len(all_data_bl)), flush=True)

    has_5m = any(d.get("df5") is not None and len(d["df5"]) > 1000 for d in all_data_full.values())
    print("  5m data available: %s" % has_5m, flush=True)

    all_results = {}

    # =====================================================================
    # PHASE 1: v10 Baseline (no filters, all 20 symbols)
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 1 -- v10 BASELINE (20 symbols, no v11 filters)", flush=True)
    print("=" * 72, flush=True)
    print("  Published v10 reference:", flush=True)
    for cn, p in V10_PUBLISHED.items():
        print("    %-15s PF=%.2f  n=%d  WR=%.1f%%  DD=%.1f%%  PnL=%+.1f%%"
              % (cn, p["pf"], p["trades"], p["wr"], p["dd"], p["pnl"]), flush=True)
    print(flush=True)

    p1_results = {}
    configs_to_test = ["bbs_1h_15m", "kc_1h_15m"]
    if has_5m:
        configs_to_test = ["bbs_1h_5m"] + configs_to_test

    for cfg_name in configs_to_test:
        print("  [%s] baseline:" % cfg_name, flush=True)
        res = run_variant("  baseline (20sym)", cfg_name, all_data_full)
        p1_results[cfg_name] = res

    all_results["p1_baseline"] = p1_results

    # =====================================================================
    # PHASE 2: Filter Impact (17 symbols after blacklist = common base)
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 2 -- INDIVIDUAL FILTER IMPACT (17 sym, blacklist applied)", flush=True)
    print("=" * 72, flush=True)

    p2_results = {}

    for cfg_name in configs_to_test:
        print("\n  --- %s ---" % cfg_name, flush=True)
        c = V10_CONFIGS[cfg_name]
        pub = V10_PUBLISHED[cfg_name]

        variants = [
            ("bl_only (17sym)",         {"use_time_gate": False, "use_ltf_confirm": False, "use_break_even": False, "use_cooldown": False}),
            ("bl + time_gate",           {"use_time_gate": True,  "use_ltf_confirm": False, "use_break_even": False, "use_cooldown": False}),
            ("bl + ltf_confirm",         {"use_time_gate": False, "use_ltf_confirm": True,  "use_break_even": False, "use_cooldown": False}),
            ("bl + break_even",          {"use_time_gate": False, "use_ltf_confirm": False, "use_break_even": True,  "use_cooldown": False}),
            ("bl + cooldown",            {"use_time_gate": False, "use_ltf_confirm": False, "use_break_even": False, "use_cooldown": True}),
            ("bl + ALL filters",         {"use_time_gate": True,  "use_ltf_confirm": True,  "use_break_even": True,  "use_cooldown": True}),
        ]

        cfg_results = []
        for label, flags in variants:
            sigs, arrs = build_sigs(cfg_name, all_data_bl)
            res = run_sim_v11(sigs, arrs, c["sl_cfg"], c["tp_cfg"],
                              risk_pt=0.010, cap_frac=c["cap_frac"], max_pos=c["max_pos"],
                              **flags)
            delta_pf = res["profit_factor"] - pub["pf"]
            delta_n  = res["trades"] - pub["trades"]
            print("  %-45s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%  [dPF=%+.2f  dN=%+d]"
                  % (label, res["trades"], res["profit_factor"],
                     res["pnl_pct"], res["max_dd_pct"], delta_pf, delta_n), flush=True)
            cfg_results.append({"label": label, **res})
        p2_results[cfg_name] = cfg_results

    all_results["p2_filter_impact"] = p2_results

    # =====================================================================
    # PHASE 3: VWAP filter for BBS_1H_5M pullback
    # =====================================================================
    if has_5m:
        print("\n" + "=" * 72, flush=True)
        print("  PHASE 3 -- VWAP FILTER (BBS_1H_5M only)", flush=True)
        print("=" * 72, flush=True)

        sigs, arrs = build_sigs("bbs_1h_5m", all_data_bl)
        c = V10_CONFIGS["bbs_1h_5m"]
        pub = V10_PUBLISHED["bbs_1h_5m"]

        p3_variants = [
            ("bl_only (baseline)",  {"use_vwap": False, "use_break_even": False}),
            ("+ VWAP filter",       {"use_vwap": True,  "use_break_even": False}),
            ("+ VWAP + break_even", {"use_vwap": True,  "use_break_even": True}),
            ("+ ALL + VWAP",        {"use_time_gate": True, "use_ltf_confirm": True,
                                     "use_break_even": True, "use_cooldown": True, "use_vwap": True}),
        ]

        p3_results = []
        for label, flags in p3_variants:
            res = run_sim_v11(sigs, arrs, c["sl_cfg"], c["tp_cfg"],
                              risk_pt=0.010, cap_frac=c["cap_frac"], max_pos=c["max_pos"],
                              **flags)
            delta_pf = res["profit_factor"] - pub["pf"]
            delta_n  = res["trades"] - pub["trades"]
            print("  %-45s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%  [dPF=%+.2f  dN=%+d]"
                  % (label, res["trades"], res["profit_factor"],
                     res["pnl_pct"], res["max_dd_pct"], delta_pf, delta_n), flush=True)
            p3_results.append({"label": label, **res})
        all_results["p3_vwap"] = p3_results
    else:
        print("\n  PHASE 3 -- VWAP FILTER: SKIP (no 5m data)", flush=True)
        all_results["p3_vwap"] = []

    # =====================================================================
    # PHASE 4: Walk-forward on v11 combined (best variant per strategy)
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 4 -- WALK-FORWARD on v11 combined (67/33 split)", flush=True)
    print("=" * 72, flush=True)

    wf_results = []
    for cfg_name in configs_to_test:
        wfr = run_walk_forward(
            "WF_%s_v11all" % cfg_name, cfg_name, all_data_bl,
            use_time_gate=True, use_ltf_confirm=True,
            use_break_even=True, use_cooldown=True,
        )
        wf_results.append(wfr)

        # Also WF without filters for comparison
        wfr_base = run_walk_forward(
            "WF_%s_base" % cfg_name, cfg_name, all_data_bl,
        )
        wf_results.append(wfr_base)

    all_results["p4_walkforward"] = wf_results

    # =====================================================================
    # PHASE 5: Final Decision Summary
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 5 -- FINAL DECISION SUMMARY", flush=True)
    print("=" * 72, flush=True)

    decisions = {}
    for cfg_name in configs_to_test:
        # Find "bl_only" baseline and "ALL filters" result from phase 2
        cfg_p2 = p2_results.get(cfg_name, [])
        baseline_r = next((r for r in cfg_p2 if "bl_only" in r["label"]), None)
        v11_r = next((r for r in cfg_p2 if "ALL filters" in r["label"]), None)
        if baseline_r and v11_r:
            verdict, ok = evaluate_v11(cfg_name, baseline_r, v11_r)
            decisions[cfg_name] = {"verdict": verdict, "ok": ok}

    # Print final summary
    print("\n  === v11 Filter Summary ===", flush=True)
    print("  (negative dN means fewer trades - check if acceptable)", flush=True)
    print(flush=True)
    for cfg_name in configs_to_test:
        cfg_p2 = p2_results.get(cfg_name, [])
        print("  %s:" % cfg_name, flush=True)
        for r in cfg_p2:
            print("    %-40s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%%"
                  % (r["label"], r["trades"], r["profit_factor"],
                     r["pnl_pct"], r["max_dd_pct"]), flush=True)
        if cfg_name in decisions:
            print("    >> %s" % decisions[cfg_name]["verdict"], flush=True)
        print(flush=True)

    print("  === Walk-Forward Results ===", flush=True)
    for w in wf_results:
        print("    %-40s train=%+6.1f%%(PF=%.2f n=%d) test=%+6.1f%%(PF=%.2f n=%d) [%s]"
              % (w["name"], w["train_pnl"], w["train_pf"], w["train_n"],
                 w["test_pnl"], w["test_pf"], w["test_n"], w["tag"]), flush=True)

    # ---- Save results ----
    out_path = RESULTS_DIR / ("backtest_v11_%s.json" % ts)
    save_data = {
        "metadata": {
            "timestamp": ts,
            "symbols_full": len(all_data_full),
            "symbols_blacklisted": len(all_data_bl),
            "runtime_sec": round(_time.time() - t0, 1),
            "v10_published": V10_PUBLISHED,
            "blacklist": sorted(BLACKLIST),
        },
        "p1_baseline": {k: {kk: vv for kk, vv in v.items() if kk not in ("df",)}
                        for k, v in p1_results.items()},
        "p2_filter_impact": {
            k: [{kk: vv for kk, vv in r.items() if kk not in ("df", "exit_reasons")} for r in v]
            for k, v in p2_results.items()
        },
        "p3_vwap": [{kk: vv for kk, vv in r.items() if kk not in ("df",)}
                    for r in all_results.get("p3_vwap", [])],
        "p4_walkforward": wf_results,
        "p5_decisions": decisions,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\n  Results saved:", out_path, flush=True)
    print("Total runtime: %d min %d sec" % divmod(int(_time.time() - t0), 60), flush=True)


if __name__ == "__main__":
    main()
