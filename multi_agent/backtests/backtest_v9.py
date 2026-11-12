"""
backtest_v9.py  --  Comprehensive Multi-TF Strategy Research  (2026-04-13)

3 strategies × 4 timeframes (5m, 15m, 30m, 1h):
  1. FVG (BigBeluga style) — large gaps (0.6%+), zone retest, trend filter
  2. BBS — squeeze breakout, params scaled per timeframe
  3. KC Squeeze — Keltner squeeze expansion, params scaled per timeframe

Additional tests:
  - Multi-TF signal->entry: signal on higher TF, entry on lower TF
  - SL/TP optimization for all viable configs
  - Walk-forward validation

Architecture:
  Phase 0   Load data for all timeframes (5m, 15m, 30m, 1h) per symbol
  Phase 1   FVG on all 4 timeframes  (entry quality)
  Phase 2   BBS on all 4 timeframes  (entry quality)
  Phase 3   KC Squeeze on all 4 timeframes  (entry quality)
  Phase 4   Multi-TF signal->entry (signal on HTF, entry on LTF)
  Phase 5   SL/TP optimization for top configs
  Phase 6   Walk-forward on all viable
  Phase 7   Final report
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

logger = logging.getLogger("v9")
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

SOFT_CFG = {"confirm_bars": 5, "min_hold_bars": 12, "smooth_alpha": 0.30,
            "min_confidence": 1.0, "transition_margin": 0.5,
            "undefined_threshold": 0.5}

# ===========================================================================
#  DATA LOADING — multi-timeframe
# ===========================================================================

def _load(symbol, tf, days_tag="180d"):
    safe = symbol.replace("/", "_").replace(":", "_")
    for ext in (".parquet", ".csv"):
        fpath = CACHE_DIR / f"{safe}_{tf}_{days_tag}{ext}"
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


def resample_15m_to_30m(df15):
    """Resample 15m OHLCV to 30m by aggregating pairs of rows."""
    if df15 is None or len(df15) < 4:
        return None
    # Round timestamps to 30m boundaries
    df = df15.copy()
    ms_30m = 1_800_000
    df["ts_30m"] = (df["timestamp"] // ms_30m) * ms_30m
    agg = df.groupby("ts_30m").agg({
        "timestamp": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).reset_index(drop=True)
    return agg


def load_all_tf(days_tag="180d"):
    """Load 5m, 15m, 30m, 1h, 4h data for all symbols."""
    all_data = {}
    for sym in SYMBOLS:
        d = {}
        d["df5"]  = _load(sym, "5m", days_tag)
        d["df15"] = _load(sym, "15m", days_tag)
        d["df30"] = resample_15m_to_30m(d["df15"])
        d["df1h"] = _load(sym, "1h", days_tag)
        d["df4h"] = _load(sym, "4h", days_tag)

        if d["df15"] is None or len(d["df15"]) < 300:
            continue
        if d["df1h"] is None or len(d["df1h"]) < 210:
            continue

        # Pre-compute HTF indicators (shared by all strategies)
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
    """Extract OHLCV arrays + compute base indicators for any timeframe."""
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

    # Bollinger Bands
    sma20 = _sma(cl, 20)
    std20 = _stdev(cl, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_mid = sma20

    # Keltner Channel
    kc_mid = _ema(cl, 20)
    atr_kc = _atr(hi, lo, cl, 10)
    kc_upper = kc_mid + 1.5 * atr_kc
    kc_lower = kc_mid - 1.5 * atr_kc

    # Squeeze detection
    kc_squeeze = np.zeros(n, dtype=bool)
    kc_sq_len = np.zeros(n, dtype=int)
    for i in range(20, n):
        if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
            kc_squeeze[i] = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])
    for i in range(1, n):
        if kc_squeeze[i]:
            kc_sq_len[i] = kc_sq_len[i-1] + 1

    # Volume
    vol_ma20 = _sma(vo, 20)
    vol_ratio = np.where(vol_ma20 > 0, vo / vol_ma20, 1.0)

    # EMAs for trend
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


# ===========================================================================
#  STRATEGY 1: FVG (BigBeluga style)
# ===========================================================================

def gen_fvg_signals(tf_data, htf_data, params):
    """
    BigBeluga-style FVG: detect large gaps, trade retest with trend filter.

    FVG detection:
    - Bullish: candle[i].low > candle[i-2].high (gap between candle 1 and 3)
    - Bearish: candle[i].high < candle[i-2].low
    - Gap must be >= min_gap_pct of price

    Entry on retest:
    - Price must touch gap zone
    - Rejection candle: close back in trend direction
    - Optional: volume confirmation
    - Trend filter: price vs EMA200

    SL: just beyond gap zone
    TP: fixed R-multiple
    """
    if tf_data is None:
        return {}

    min_gap_pct  = params.get("min_gap_pct", 0.006)   # 0.6%
    max_age      = params.get("max_age", 100)
    min_age      = params.get("min_age", 3)
    need_reject  = params.get("rejection", True)
    trend_filter = params.get("trend_filter", True)   # EMA200 alignment
    vol_filter   = params.get("vol_filter", False)
    vol_mult     = params.get("vol_mult", 1.2)
    htf_trend    = params.get("htf_trend", False)     # Use 1h EMA200 for trend
    sl_buffer_pct = params.get("sl_buffer_pct", 0.3)  # SL buffer beyond gap (% of ATR)
    max_gap_pct  = params.get("max_gap_pct", 0.05)    # Max 5% gap (filter noise)

    cl = tf_data["cl"]; hi = tf_data["hi"]; lo = tf_data["lo"]
    op = tf_data["op"]; n = tf_data["n"]
    atr = tf_data["atr"]; adx = tf_data["adx"]
    vol_r = tf_data["vol_ratio"]
    ema200 = tf_data["ema200"]

    sigs = {}
    active_fvgs = []
    warmup = max(250, int(n * 0.05))  # min warmup bars

    for i in range(2, n):
        # ---- Detect new FVGs ----
        if cl[i] > 0:
            # Bullish FVG: gap UP (candle i low > candle i-2 high)
            if lo[i] > hi[i-2]:
                gap_pct = (lo[i] - hi[i-2]) / cl[i]
                if min_gap_pct <= gap_pct <= max_gap_pct:
                    active_fvgs.append({
                        "type": "bull", "top": float(lo[i]), "bot": float(hi[i-2]),
                        "mid": float((lo[i] + hi[i-2]) / 2), "bar": i,
                        "gap_pct": gap_pct,
                    })

            # Bearish FVG: gap DOWN (candle i high < candle i-2 low)
            if hi[i] < lo[i-2]:
                gap_pct = (lo[i-2] - hi[i]) / cl[i]
                if min_gap_pct <= gap_pct <= max_gap_pct:
                    active_fvgs.append({
                        "type": "bear", "top": float(lo[i-2]), "bot": float(hi[i]),
                        "mid": float((lo[i-2] + hi[i]) / 2), "bar": i,
                        "gap_pct": gap_pct,
                    })

        # Prune expired gaps
        active_fvgs = [f for f in active_fvgs if i - f["bar"] <= max_age]

        if i < warmup:
            continue
        if i in sigs:
            continue

        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01

        # ---- Check retests ----
        for fvg in list(active_fvgs):
            age = i - fvg["bar"]
            if age < min_age:
                continue

            # Bullish FVG retest: price dips into gap zone, closes above bottom
            if fvg["type"] == "bull":
                if lo[i] <= fvg["top"] and cl[i] >= fvg["bot"]:
                    # Trend filter: price above EMA200
                    if trend_filter and not np.isnan(ema200[i]) and cl[i] < ema200[i]:
                        continue
                    # HTF trend filter
                    if htf_trend and htf_data is not None:
                        htf_ema = htf_data.get("ema200")
                        htf_cl = htf_data.get("cl")
                        htf_ts = htf_data.get("ts")
                        if htf_ema is not None and htf_cl is not None:
                            ts_i = tf_data["ts"][i]
                            h_idx = int(np.searchsorted(htf_ts, ts_i, side="right")) - 1
                            if h_idx >= 200 and not np.isnan(htf_ema[h_idx]):
                                if htf_cl[h_idx] < htf_ema[h_idx]:
                                    continue

                    # Rejection candle: bullish close (close > open) and close above gap mid
                    if need_reject:
                        if not (cl[i] > op[i] and cl[i] > fvg["mid"]):
                            continue

                    # Volume filter
                    if vol_filter and vol_r[i] < vol_mult:
                        continue

                    # SL: below gap bottom
                    sl_v = fvg["bot"] - atr_v * sl_buffer_pct
                    risk = cl[i] - sl_v
                    if risk <= 0 or risk / cl[i] > 0.05 or risk / cl[i] < 0.001:
                        continue

                    sigs[i] = {"direction": "LONG", "entry_price": float(cl[i]),
                               "default_sl": float(sl_v), "gap_pct": fvg["gap_pct"],
                               "age": age}
                    active_fvgs.remove(fvg)
                    break

            # Bearish FVG retest: price wicks into gap zone, closes below top
            elif fvg["type"] == "bear":
                if hi[i] >= fvg["bot"] and cl[i] <= fvg["top"]:
                    if trend_filter and not np.isnan(ema200[i]) and cl[i] > ema200[i]:
                        continue
                    if htf_trend and htf_data is not None:
                        htf_ema = htf_data.get("ema200")
                        htf_cl = htf_data.get("cl")
                        htf_ts = htf_data.get("ts")
                        if htf_ema is not None and htf_cl is not None:
                            ts_i = tf_data["ts"][i]
                            h_idx = int(np.searchsorted(htf_ts, ts_i, side="right")) - 1
                            if h_idx >= 200 and not np.isnan(htf_ema[h_idx]):
                                if htf_cl[h_idx] > htf_ema[h_idx]:
                                    continue

                    if need_reject:
                        if not (cl[i] < op[i] and cl[i] < fvg["mid"]):
                            continue
                    if vol_filter and vol_r[i] < vol_mult:
                        continue

                    sl_v = fvg["top"] + atr_v * sl_buffer_pct
                    risk = sl_v - cl[i]
                    if risk <= 0 or risk / cl[i] > 0.05 or risk / cl[i] < 0.001:
                        continue

                    sigs[i] = {"direction": "SHORT", "entry_price": float(cl[i]),
                               "default_sl": float(sl_v), "gap_pct": fvg["gap_pct"],
                               "age": age}
                    active_fvgs.remove(fvg)
                    break

    return sigs


# ===========================================================================
#  STRATEGY 2: BBS (adapted for any timeframe)
# ===========================================================================

def gen_bbs_signals_tf(tf_data, htf_data, params):
    """BBS: detect squeeze on entry TF, use HTF for trend confirmation."""
    if tf_data is None:
        return {}

    # Need squeeze indicators from bb_squeeze_agent
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
    warmup = max(300, int(n * 0.05))

    for i in range(warmup, n):
        # ADX filter
        if not np.isnan(adx[i]) and adx[i] < adx_th:
            continue

        # Volume filter
        if vol_m > 0 and vol_r[i] < vol_m:
            continue

        # Squeeze detection
        bbs_s = detect_squeeze_signal(squeeze_ind, bar_idx=i,
                                      min_squeeze_bars=min_sq,
                                      adx_threshold=adx_th,
                                      vol_mult=vol_m)
        if not bbs_s:
            continue

        # HTF trend filter
        if use_htf and htf_data is not None:
            htf_ema = htf_data.get("ema200")
            htf_cl = htf_data.get("cl")
            htf_ts = htf_data.get("ts")
            if htf_ema is not None and htf_cl is not None:
                ts_i = tf_data["ts"][i]
                h_idx = int(np.searchsorted(htf_ts, ts_i, side="right")) - 1
                if h_idx < 200:
                    continue
                if np.isnan(htf_ema[h_idx]):
                    continue
                htf_d = "LONG" if htf_cl[h_idx] > htf_ema[h_idx] else "SHORT"
                if bbs_s["direction"] != htf_d:
                    continue

        # Head-fake filter
        if use_hf and check_head_fake(squeeze_ind, i, 2):
            continue

        # Chandelier Exit SL (as default)
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
#  STRATEGY 3: KC Squeeze (adapted for any timeframe)
# ===========================================================================

def gen_kc_squeeze_signals_tf(tf_data, htf_data, params):
    """KC Squeeze: BB inside KC -> expansion + momentum direction."""
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

def gen_mtf_signals(htf_sigs, htf_data, ltf_data, params):
    """
    Signal fires on HTF (e.g. 1h BBS signal).
    Then look for good entry on LTF (e.g. 15m pullback to EMA21 or squeeze release).
    """
    if ltf_data is None or htf_data is None:
        return {}

    entry_mode   = params.get("entry_mode", "pullback_ema21")  # pullback_ema21, next_bar, squeeze_release
    max_wait     = params.get("max_wait_bars", 12)  # max LTF bars to wait for entry
    sl_atr_mult  = params.get("sl_atr_mult", 2.5)

    cl_ltf = ltf_data["cl"]; hi_ltf = ltf_data["hi"]; lo_ltf = ltf_data["lo"]
    n_ltf = ltf_data["n"]
    ts_ltf = ltf_data["ts"]; ts_htf = htf_data["ts"]
    atr_ltf = ltf_data["atr"]
    ema21_ltf = ltf_data["ema21"]

    sigs = {}

    for htf_bar, htf_sig in htf_sigs.items():
        htf_ts = htf_data["ts"][htf_bar]
        # Find corresponding LTF bar
        ltf_start = int(np.searchsorted(ts_ltf, htf_ts, side="left"))
        if ltf_start >= n_ltf:
            continue

        direction = htf_sig["direction"]
        found = False

        for j in range(ltf_start, min(ltf_start + max_wait, n_ltf)):
            atr_v = atr_ltf[j] if not np.isnan(atr_ltf[j]) else cl_ltf[j] * 0.01

            if entry_mode == "next_bar":
                # Just enter on next LTF bar
                sl_v = (cl_ltf[j] - atr_v * sl_atr_mult) if direction == "LONG" else (cl_ltf[j] + atr_v * sl_atr_mult)
                sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                            "default_sl": float(sl_v), "htf_bar": htf_bar}
                found = True
                break

            elif entry_mode == "pullback_ema21":
                # Wait for pullback to EMA21, then enter
                if np.isnan(ema21_ltf[j]):
                    continue
                if direction == "LONG" and lo_ltf[j] <= ema21_ltf[j] * 1.002 and cl_ltf[j] > ema21_ltf[j]:
                    sl_v = cl_ltf[j] - atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    found = True
                    break
                elif direction == "SHORT" and hi_ltf[j] >= ema21_ltf[j] * 0.998 and cl_ltf[j] < ema21_ltf[j]:
                    sl_v = cl_ltf[j] + atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    found = True
                    break

            elif entry_mode == "pullback_bb_mid":
                # Wait for pullback to BB mid
                bb_m = ltf_data["bb_mid"]
                if np.isnan(bb_m[j]):
                    continue
                if direction == "LONG" and lo_ltf[j] <= bb_m[j] * 1.002 and cl_ltf[j] > bb_m[j]:
                    sl_v = cl_ltf[j] - atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    found = True
                    break
                elif direction == "SHORT" and hi_ltf[j] >= bb_m[j] * 0.998 and cl_ltf[j] < bb_m[j]:
                    sl_v = cl_ltf[j] + atr_v * sl_atr_mult
                    sigs[j] = {"direction": direction, "entry_price": float(cl_ltf[j]),
                                "default_sl": float(sl_v), "htf_bar": htf_bar}
                    found = True
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
    elif t == "structure":
        s = max(0, i - sl_cfg.get("n_bars", 5) + 1)
        sl = float(np.min(lo[s:i+1])) if d == "LONG" else float(np.max(hi[s:i+1]))
    else:
        sl = default_sl
    if d == "LONG"  and sl >= entry: sl = entry * 0.98
    if d == "SHORT" and sl <= entry: sl = entry * 1.02
    return float(sl)


# ===========================================================================
#  SIMULATION ENGINE
# ===========================================================================

def run_sim_tf(signals_by_sym, tf_arrays_by_sym, sl_cfg, tp_cfg,
               risk_pt=0.010, cap_frac=0.50, max_pos=4,
               start_frac=0.0, end_frac=1.0):
    """
    Simulate trades for a single timeframe.
    signals_by_sym: {sym: {bar_idx: signal}}
    tf_arrays_by_sym: {sym: tf_arrays_dict}
    """
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    open_pos = {}
    acnt = 0
    completed = []

    # Find max bars
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

            # ---- manage open position ----
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

            # ---- signal lookup ----
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
    # Max DD
    bal = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    for t in completed:
        bal += t["pnl_usd"]
        peak = max(peak, bal)
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
#  CONFIG GRIDS
# ===========================================================================

# ---- FVG configs per timeframe ----
FVG_CFGS_BY_TF = {
    "5m": [
        {"id": "fvg5_base",  "params": {"min_gap_pct": 0.003, "max_age": 60, "min_age": 3, "rejection": True, "trend_filter": True, "htf_trend": True}},
        {"id": "fvg5_06pct", "params": {"min_gap_pct": 0.006, "max_age": 60, "min_age": 3, "rejection": True, "trend_filter": True, "htf_trend": True}},
        {"id": "fvg5_loose", "params": {"min_gap_pct": 0.003, "max_age": 100, "min_age": 2, "rejection": False, "trend_filter": True}},
        {"id": "fvg5_vol",   "params": {"min_gap_pct": 0.003, "max_age": 60, "min_age": 3, "rejection": True, "trend_filter": True, "vol_filter": True, "vol_mult": 1.5}},
        {"id": "fvg5_htf_only", "params": {"min_gap_pct": 0.004, "max_age": 80, "min_age": 3, "rejection": True, "trend_filter": False, "htf_trend": True}},
    ],
    "15m": [
        {"id": "fvg15_06pct",   "params": {"min_gap_pct": 0.006, "max_age": 50, "min_age": 3, "rejection": True, "trend_filter": True}},
        {"id": "fvg15_06_htf",  "params": {"min_gap_pct": 0.006, "max_age": 50, "min_age": 3, "rejection": True, "trend_filter": True, "htf_trend": True}},
        {"id": "fvg15_04pct",   "params": {"min_gap_pct": 0.004, "max_age": 80, "min_age": 3, "rejection": True, "trend_filter": True}},
        {"id": "fvg15_08pct",   "params": {"min_gap_pct": 0.008, "max_age": 40, "min_age": 3, "rejection": True, "trend_filter": True}},
        {"id": "fvg15_loose",   "params": {"min_gap_pct": 0.004, "max_age": 100, "min_age": 2, "rejection": False, "trend_filter": True}},
        {"id": "fvg15_vol",     "params": {"min_gap_pct": 0.006, "max_age": 50, "min_age": 3, "rejection": True, "trend_filter": True, "vol_filter": True, "vol_mult": 1.3}},
        {"id": "fvg15_no_trend","params": {"min_gap_pct": 0.006, "max_age": 50, "min_age": 3, "rejection": True, "trend_filter": False}},
        {"id": "fvg15_10pct",   "params": {"min_gap_pct": 0.010, "max_age": 30, "min_age": 5, "rejection": True, "trend_filter": True}},
    ],
    "30m": [
        {"id": "fvg30_06pct",  "params": {"min_gap_pct": 0.006, "max_age": 40, "min_age": 2, "rejection": True, "trend_filter": True}},
        {"id": "fvg30_08pct",  "params": {"min_gap_pct": 0.008, "max_age": 40, "min_age": 2, "rejection": True, "trend_filter": True}},
        {"id": "fvg30_10pct",  "params": {"min_gap_pct": 0.010, "max_age": 30, "min_age": 3, "rejection": True, "trend_filter": True}},
        {"id": "fvg30_loose",  "params": {"min_gap_pct": 0.005, "max_age": 60, "min_age": 2, "rejection": False, "trend_filter": True}},
        {"id": "fvg30_htf",    "params": {"min_gap_pct": 0.008, "max_age": 40, "min_age": 2, "rejection": True, "trend_filter": True, "htf_trend": True}},
    ],
    "1h": [
        {"id": "fvg1h_08pct",  "params": {"min_gap_pct": 0.008, "max_age": 30, "min_age": 2, "rejection": True, "trend_filter": True}},
        {"id": "fvg1h_10pct",  "params": {"min_gap_pct": 0.010, "max_age": 24, "min_age": 2, "rejection": True, "trend_filter": True}},
        {"id": "fvg1h_15pct",  "params": {"min_gap_pct": 0.015, "max_age": 20, "min_age": 2, "rejection": True, "trend_filter": True}},
        {"id": "fvg1h_loose",  "params": {"min_gap_pct": 0.006, "max_age": 50, "min_age": 1, "rejection": False, "trend_filter": True}},
        {"id": "fvg1h_vol",    "params": {"min_gap_pct": 0.010, "max_age": 24, "min_age": 2, "rejection": True, "trend_filter": True, "vol_filter": True, "vol_mult": 1.5}},
    ],
}

# ---- BBS configs per timeframe (params scaled) ----
BBS_CFGS_BY_TF = {
    "5m": [
        {"id": "bbs5_base",   "params": {"min_squeeze": 15, "adx_thresh": 22, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs5_loose",  "params": {"min_squeeze": 8,  "adx_thresh": 18, "vol_mult": 0.8, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs5_tight",  "params": {"min_squeeze": 20, "adx_thresh": 25, "vol_mult": 1.2, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs5_nohf",   "params": {"min_squeeze": 15, "adx_thresh": 22, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": False}},
    ],
    "15m": [
        {"id": "bbs15_v8",    "params": {"min_squeeze": 10, "adx_thresh": 22, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs15_loose", "params": {"min_squeeze": 8,  "adx_thresh": 18, "vol_mult": 0.8, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs15_tight", "params": {"min_squeeze": 15, "adx_thresh": 25, "vol_mult": 1.2, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs15_mid",   "params": {"min_squeeze": 12, "adx_thresh": 20, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}},
    ],
    "30m": [
        {"id": "bbs30_base",  "params": {"min_squeeze": 8,  "adx_thresh": 20, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs30_loose", "params": {"min_squeeze": 5,  "adx_thresh": 15, "vol_mult": 0.8, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs30_tight", "params": {"min_squeeze": 12, "adx_thresh": 22, "vol_mult": 1.2, "htf_filter": True, "headfake_filter": True}},
        {"id": "bbs30_nohf",  "params": {"min_squeeze": 8,  "adx_thresh": 20, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": False}},
    ],
    "1h": [
        {"id": "bbs1h_base",  "params": {"min_squeeze": 6,  "adx_thresh": 18, "vol_mult": 1.0, "htf_filter": False, "headfake_filter": True}},
        {"id": "bbs1h_loose", "params": {"min_squeeze": 4,  "adx_thresh": 15, "vol_mult": 0.8, "htf_filter": False, "headfake_filter": False}},
        {"id": "bbs1h_tight", "params": {"min_squeeze": 8,  "adx_thresh": 22, "vol_mult": 1.2, "htf_filter": False, "headfake_filter": True}},
        {"id": "bbs1h_4h",    "params": {"min_squeeze": 6,  "adx_thresh": 18, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}},
    ],
}

# ---- KC Squeeze configs per timeframe ----
KC_CFGS_BY_TF = {
    "5m": [
        {"id": "kc5_base",  "params": {"min_squeeze": 15, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12, "sl_atr_mult": 2.0}},
        {"id": "kc5_loose", "params": {"min_squeeze": 8,  "adx_min": 12, "vol_mult": 0,   "mom_bars": 12, "sl_atr_mult": 2.0}},
        {"id": "kc5_tight", "params": {"min_squeeze": 20, "adx_min": 20, "vol_mult": 1.2, "mom_bars": 15, "sl_atr_mult": 2.0}},
    ],
    "15m": [
        {"id": "kc15_base",  "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12, "sl_atr_mult": 2.0}},
        {"id": "kc15_loose", "params": {"min_squeeze": 6,  "adx_min": 12, "vol_mult": 0,   "mom_bars": 12, "sl_atr_mult": 2.0}},
        {"id": "kc15_htf",   "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12, "htf_filter": True, "sl_atr_mult": 2.0}},
    ],
    "30m": [
        {"id": "kc30_base",  "params": {"min_squeeze": 8,  "adx_min": 15, "vol_mult": 1.0, "mom_bars": 10, "sl_atr_mult": 2.0}},
        {"id": "kc30_loose", "params": {"min_squeeze": 5,  "adx_min": 12, "vol_mult": 0,   "mom_bars": 10, "sl_atr_mult": 2.0}},
        {"id": "kc30_tight", "params": {"min_squeeze": 12, "adx_min": 18, "vol_mult": 1.2, "mom_bars": 12, "sl_atr_mult": 2.5}},
    ],
    "1h": [
        {"id": "kc1h_base",  "params": {"min_squeeze": 6,  "adx_min": 15, "vol_mult": 1.0, "mom_bars": 8,  "sl_atr_mult": 2.0}},
        {"id": "kc1h_loose", "params": {"min_squeeze": 4,  "adx_min": 12, "vol_mult": 0,   "mom_bars": 6,  "sl_atr_mult": 2.0}},
        {"id": "kc1h_tight", "params": {"min_squeeze": 10, "adx_min": 18, "vol_mult": 1.2, "mom_bars": 10, "sl_atr_mult": 2.5}},
    ],
}

# SL/TP combos for optimization
SLTP_COMBOS = [
    ("atr1.5_2r",   {"type": "atr", "mult": 1.5}, {"type": "fixed_r", "r": 2.0}),
    ("atr1.5_2.5r", {"type": "atr", "mult": 1.5}, {"type": "fixed_r", "r": 2.5}),
    ("atr2.0_2.5r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 2.5}),
    ("atr2.0_3.0r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 3.0}),
    ("atr2.0_3.5r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 3.5}),
    ("atr2.5_3.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.0}),
    ("atr2.5_3.5r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5}),
    ("atr2.5_4.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 4.0}),
    ("atr3.0_3.5r", {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 3.5}),
    ("def_2.5r",    {"type": "default"},           {"type": "fixed_r", "r": 2.5}),
    ("def_3.5r",    {"type": "default"},           {"type": "fixed_r", "r": 3.5}),
    ("atr2.0_part", {"type": "atr", "mult": 2.0}, {"type": "partial_fixed", "r1": 1.5, "r2": 3.5, "frac": 0.5}),
    ("atr2.5_trail",{"type": "atr", "mult": 2.5}, {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 2.5}),
]


# ===========================================================================
#  PHASE RUNNERS
# ===========================================================================

def run_strategy_phase(name, strat_fn, cfgs_by_tf, all_data, tf_map):
    """Run a strategy across all timeframes.
    tf_map: {tf_name: df_key} e.g. {"5m": "df5", "15m": "df15", "30m": "df30", "1h": "df1h"}
    """
    print("\n" + "=" * 70, flush=True)
    print("  %s — all timeframes" % name, flush=True)
    print("=" * 70, flush=True)

    all_results = {}

    for tf_name, df_key in tf_map.items():
        cfgs = cfgs_by_tf.get(tf_name, [])
        if not cfgs:
            continue

        # Build tf_arrays for all symbols
        tf_arrays = {}
        htf_arrays = {}
        for sym, d in all_data.items():
            arr = prepare_tf_arrays(d.get(df_key))
            if arr is not None:
                tf_arrays[sym] = arr
                # HTF: use 1h for 5m/15m/30m, use 4h for 1h
                if tf_name == "1h":
                    htf_arr = prepare_tf_arrays(d.get("df4h"))
                else:
                    htf_arr = prepare_tf_arrays(d.get("df1h"))
                htf_arrays[sym] = htf_arr

        if not tf_arrays:
            print("  [%s] No data available, skipping." % tf_name, flush=True)
            continue

        tf_results = []
        for cfg_entry in cfgs:
            # Generate signals
            all_sigs = {}
            total_sigs = 0
            for sym in tf_arrays:
                s = strat_fn(tf_arrays[sym], htf_arrays.get(sym), cfg_entry["params"])
                all_sigs[sym] = s
                total_sigs += len(s)

            # Sim with default SL/TP (ATR2.5, 3.5R)
            res = run_sim_tf(all_sigs, tf_arrays,
                             {"type": "atr", "mult": 2.5},
                             {"type": "fixed_r", "r": 3.5},
                             risk_pt=0.010, cap_frac=0.60, max_pos=4)
            res["config_id"] = cfg_entry["id"]
            res["timeframe"] = tf_name
            res["total_sigs"] = total_sigs
            res["score"] = _score(res)
            res["params"] = cfg_entry["params"]
            tf_results.append(res)

            print("  [%s] %-22s sigs=%5d n=%3d PF=%.2f PnL=%+6.1f%% DD=%.1f%%"
                  % (tf_name, cfg_entry["id"], total_sigs, res["trades"],
                     res["profit_factor"], res["pnl_pct"], res["max_dd_pct"]),
                  flush=True)

        tf_results.sort(key=lambda x: x["score"], reverse=True)
        all_results[tf_name] = tf_results

    return all_results


def run_sltp_optimization(name, strat_fn, best_params, tf_arrays, htf_arrays, tf_name):
    """SL/TP optimization for a single strategy+timeframe combo."""
    print("  SL/TP opt: %s [%s]" % (name, tf_name), flush=True)

    # Generate signals once
    all_sigs = {}
    for sym in tf_arrays:
        all_sigs[sym] = strat_fn(tf_arrays[sym], htf_arrays.get(sym), best_params)

    results = []
    for combo_name, sl_c, tp_c in SLTP_COMBOS:
        res = run_sim_tf(all_sigs, tf_arrays, sl_c, tp_c,
                         risk_pt=0.010, cap_frac=0.60, max_pos=4)
        res["sltp_id"] = combo_name
        res["score"] = _score(res)
        results.append(res)
        print("    %-20s n=%3d PF=%.2f PnL=%+6.1f%% DD=%.1f%%"
              % (combo_name, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"]), flush=True)

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def run_walk_forward(name, all_sigs, tf_arrays, sl_cfg, tp_cfg):
    """67/33 walk-forward."""
    train = run_sim_tf(all_sigs, tf_arrays, sl_cfg, tp_cfg,
                       risk_pt=0.010, cap_frac=0.60, max_pos=4,
                       start_frac=0.0, end_frac=0.67)
    test = run_sim_tf(all_sigs, tf_arrays, sl_cfg, tp_cfg,
                      risk_pt=0.010, cap_frac=0.60, max_pos=4,
                      start_frac=0.67, end_frac=1.0)
    passed = test["profit_factor"] >= 1.0 and test["pnl_pct"] > -5.0
    tag = "PASS" if passed else "FAIL"
    print("  WF %-20s train=%+.1f%%(PF=%.2f n=%d) test=%+.1f%%(PF=%.2f n=%d) [%s]"
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
    print("backtest_v9.py -- Comprehensive Multi-TF Strategy Research", flush=True)
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    t0 = _time.time()

    # ---- Phase 0: Load data ----
    print("\n  Phase 0 -- Loading multi-TF data...", flush=True)
    all_data = load_all_tf("180d")
    print("  Symbols loaded: %d" % len(all_data), flush=True)

    # Check which TFs are available
    tf_map = {}
    has_5m = any(d.get("df5") is not None and len(d["df5"]) > 1000 for d in all_data.values())
    has_30m = any(d.get("df30") is not None and len(d["df30"]) > 200 for d in all_data.values())
    if has_5m:
        tf_map["5m"] = "df5"
    tf_map["15m"] = "df15"
    if has_30m:
        tf_map["30m"] = "df30"
    tf_map["1h"] = "df1h"
    print("  Available timeframes: %s" % list(tf_map.keys()), flush=True)

    # ---- Phase 1: FVG ----
    fvg_results = run_strategy_phase("PHASE 1 -- FVG (BigBeluga style)",
                                      gen_fvg_signals, FVG_CFGS_BY_TF,
                                      all_data, tf_map)

    # ---- Phase 2: BBS ----
    bbs_results = run_strategy_phase("PHASE 2 -- BBS (multi-TF)",
                                      gen_bbs_signals_tf, BBS_CFGS_BY_TF,
                                      all_data, tf_map)

    # ---- Phase 3: KC Squeeze ----
    kc_results = run_strategy_phase("PHASE 3 -- KC Squeeze (multi-TF)",
                                     gen_kc_squeeze_signals_tf, KC_CFGS_BY_TF,
                                     all_data, tf_map)

    # ---- Phase 4: Multi-TF signal->entry ----
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 4 -- Multi-TF Signal->Entry", flush=True)
    print("=" * 70, flush=True)

    mtf_results = {}
    # Test: BBS signal on 1h -> entry on 15m
    htf_configs = [
        ("BBS_1h->15m_next",     "1h", "15m", gen_bbs_signals_tf, {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0, "htf_filter": False, "headfake_filter": True}, {"entry_mode": "next_bar", "sl_atr_mult": 2.5}),
        ("BBS_1h->15m_pullback", "1h", "15m", gen_bbs_signals_tf, {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0, "htf_filter": False, "headfake_filter": True}, {"entry_mode": "pullback_ema21", "max_wait_bars": 8, "sl_atr_mult": 2.5}),
        ("BBS_1h->15m_bb_pb",    "1h", "15m", gen_bbs_signals_tf, {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0, "htf_filter": False, "headfake_filter": True}, {"entry_mode": "pullback_bb_mid", "max_wait_bars": 8, "sl_atr_mult": 2.5}),
        ("BBS_1h->5m_pullback",  "1h", "5m",  gen_bbs_signals_tf, {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0, "htf_filter": False, "headfake_filter": True}, {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5}),
        ("KC_1h->15m_next",      "1h", "15m", gen_kc_squeeze_signals_tf, {"min_squeeze": 6, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 8, "sl_atr_mult": 2.0}, {"entry_mode": "next_bar", "sl_atr_mult": 2.5}),
        ("KC_1h->15m_pullback",  "1h", "15m", gen_kc_squeeze_signals_tf, {"min_squeeze": 6, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 8, "sl_atr_mult": 2.0}, {"entry_mode": "pullback_ema21", "max_wait_bars": 8, "sl_atr_mult": 2.5}),
        ("BBS_30m->15m_pullback","30m", "15m", gen_bbs_signals_tf, {"min_squeeze": 8, "adx_thresh": 20, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}, {"entry_mode": "pullback_ema21", "max_wait_bars": 6, "sl_atr_mult": 2.5}),
        ("BBS_30m->5m_pullback", "30m", "5m",  gen_bbs_signals_tf, {"min_squeeze": 8, "adx_thresh": 20, "vol_mult": 1.0, "htf_filter": True, "headfake_filter": True}, {"entry_mode": "pullback_ema21", "max_wait_bars": 15, "sl_atr_mult": 2.5}),
    ]

    htf_df_map = {"5m": "df5", "15m": "df15", "30m": "df30", "1h": "df1h"}

    for mtf_name, htf_tf, ltf_tf, gen_fn, sig_params, entry_params in htf_configs:
        htf_key = htf_df_map.get(htf_tf)
        ltf_key = htf_df_map.get(ltf_tf)
        if htf_key is None or ltf_key is None:
            continue

        # Check data availability
        if htf_key not in tf_map.values() and htf_tf not in tf_map:
            print("  %-30s SKIP (no %s data)" % (mtf_name, htf_tf), flush=True)
            continue
        if ltf_key not in tf_map.values() and ltf_tf not in tf_map:
            print("  %-30s SKIP (no %s data)" % (mtf_name, ltf_tf), flush=True)
            continue

        # Generate HTF signals, then find LTF entries
        all_mtf_sigs = {}
        total_sigs = 0
        ltf_arrays = {}
        for sym, d in all_data.items():
            htf_arr = prepare_tf_arrays(d.get(htf_key))
            ltf_arr = prepare_tf_arrays(d.get(ltf_key))
            if htf_arr is None or ltf_arr is None:
                continue
            ltf_arrays[sym] = ltf_arr
            # Get HTF data for trend filter
            if htf_tf == "1h":
                htf_trend = prepare_tf_arrays(d.get("df4h"))
            else:
                htf_trend = prepare_tf_arrays(d.get("df1h"))
            htf_sigs = gen_fn(htf_arr, htf_trend, sig_params)
            # Map HTF signals -> LTF entries
            mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, entry_params)
            all_mtf_sigs[sym] = mtf_sigs
            total_sigs += len(mtf_sigs)

        if not ltf_arrays:
            continue

        res = run_sim_tf(all_mtf_sigs, ltf_arrays,
                         {"type": "atr", "mult": 2.5},
                         {"type": "fixed_r", "r": 3.5},
                         risk_pt=0.010, cap_frac=0.60, max_pos=4)
        res["config_name"] = mtf_name
        res["total_sigs"] = total_sigs
        res["score"] = _score(res)
        mtf_results[mtf_name] = res
        print("  %-30s sigs=%4d n=%3d PF=%.2f PnL=%+6.1f%% DD=%.1f%%"
              % (mtf_name, total_sigs, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"]), flush=True)

    # ---- Phase 5: SL/TP optimization for viable configs ----
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 5 -- SL/TP Optimization (viable configs, PF >= 1.1)", flush=True)
    print("=" * 70, flush=True)

    sltp_results = {}
    viable_threshold = 1.10

    gen_fn_map = {"FVG": gen_fvg_signals, "BBS": gen_bbs_signals_tf, "KC": gen_kc_squeeze_signals_tf}
    strat_results_map = {"FVG": fvg_results, "BBS": bbs_results, "KC": kc_results}

    for strat_name, strat_tf_results in strat_results_map.items():
        for tf_name, tf_results in strat_tf_results.items():
            best = tf_results[0] if tf_results else None
            if best and best["profit_factor"] >= viable_threshold and best["trades"] >= 15:
                df_key = htf_df_map.get(tf_name)
                if df_key is None:
                    continue
                # Build arrays
                tf_arrays = {}
                htf_arrays = {}
                for sym, d in all_data.items():
                    arr = prepare_tf_arrays(d.get(df_key))
                    if arr is not None:
                        tf_arrays[sym] = arr
                        if tf_name == "1h":
                            htf_arrays[sym] = prepare_tf_arrays(d.get("df4h"))
                        else:
                            htf_arrays[sym] = prepare_tf_arrays(d.get("df1h"))

                key = "%s_%s" % (strat_name, tf_name)
                sltp_results[key] = run_sltp_optimization(
                    strat_name, gen_fn_map[strat_name],
                    best["params"], tf_arrays, htf_arrays, tf_name)

    # ---- Phase 6: Walk-Forward ----
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 6 -- Walk-Forward Validation", flush=True)
    print("=" * 70, flush=True)

    wf_results = []
    for strat_name, strat_tf_results in strat_results_map.items():
        for tf_name, tf_results in strat_tf_results.items():
            top_viable = [r for r in tf_results[:3]
                          if r["profit_factor"] >= 1.05 and r["trades"] >= 12]
            if not top_viable:
                continue

            df_key = htf_df_map.get(tf_name)
            if df_key is None:
                continue

            tf_arrays = {}
            htf_arrays = {}
            for sym, d in all_data.items():
                arr = prepare_tf_arrays(d.get(df_key))
                if arr is not None:
                    tf_arrays[sym] = arr
                    if tf_name == "1h":
                        htf_arrays[sym] = prepare_tf_arrays(d.get("df4h"))
                    else:
                        htf_arrays[sym] = prepare_tf_arrays(d.get("df1h"))

            for vr in top_viable[:2]:
                all_sigs = {}
                for sym in tf_arrays:
                    all_sigs[sym] = gen_fn_map[strat_name](
                        tf_arrays[sym], htf_arrays.get(sym), vr["params"])

                # Best SL/TP for this strategy+TF
                best_sl = {"type": "atr", "mult": 2.5}
                best_tp = {"type": "fixed_r", "r": 3.5}
                key = "%s_%s" % (strat_name, tf_name)
                if key in sltp_results and sltp_results[key]:
                    best_combo = sltp_results[key][0]
                    # Find matching SL/TP
                    for cn, sl_c, tp_c in SLTP_COMBOS:
                        if cn == best_combo["sltp_id"]:
                            best_sl = sl_c
                            best_tp = tp_c
                            break

                wfr = run_walk_forward(
                    "%s_%s_%s" % (strat_name, tf_name, vr["config_id"]),
                    all_sigs, tf_arrays, best_sl, best_tp)
                wf_results.append(wfr)

    # Also WF for best MTF configs
    for mtf_name, mtf_res in mtf_results.items():
        if mtf_res["profit_factor"] >= 1.05 and mtf_res["trades"] >= 12:
            # Re-generate signals for WF
            for cfg_entry in htf_configs:
                if cfg_entry[0] == mtf_name:
                    _, htf_tf, ltf_tf, gen_fn, sig_params, entry_params = cfg_entry
                    htf_key = htf_df_map[htf_tf]
                    ltf_key = htf_df_map[ltf_tf]
                    all_mtf_sigs = {}
                    ltf_arrays = {}
                    for sym, d in all_data.items():
                        htf_arr = prepare_tf_arrays(d.get(htf_key))
                        ltf_arr = prepare_tf_arrays(d.get(ltf_key))
                        if htf_arr is None or ltf_arr is None:
                            continue
                        ltf_arrays[sym] = ltf_arr
                        if htf_tf == "1h":
                            htf_trend = prepare_tf_arrays(d.get("df4h"))
                        else:
                            htf_trend = prepare_tf_arrays(d.get("df1h"))
                        htf_sigs = gen_fn(htf_arr, htf_trend, sig_params)
                        all_mtf_sigs[sym] = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, entry_params)
                    if ltf_arrays:
                        wfr = run_walk_forward(
                            "MTF_%s" % mtf_name, all_mtf_sigs, ltf_arrays,
                            {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5})
                        wf_results.append(wfr)
                    break

    # ---- Phase 7: Final Report ----
    print("\n" + "=" * 70, flush=True)
    print("  FINAL REPORT -- Multi-TF Strategy Research v9", flush=True)
    print("=" * 70, flush=True)

    for strat_name, strat_tf_results in [("FVG", fvg_results), ("BBS", bbs_results), ("KC", kc_results)]:
        print("\n  == %s ==" % strat_name, flush=True)
        for tf_name in ["5m", "15m", "30m", "1h"]:
            tf_results = strat_tf_results.get(tf_name, [])
            if not tf_results:
                print("    [%s] no data" % tf_name, flush=True)
                continue
            best = tf_results[0]
            tag = "VIABLE" if best["profit_factor"] >= 1.15 else ("MARGINAL" if best["profit_factor"] >= 1.05 else "DEAD")
            print("    [%s] [%s] best=%-18s n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%%"
                  % (tf_name, tag, best["config_id"], best["trades"],
                     best["profit_factor"], best["pnl_pct"], best["max_dd_pct"]),
                  flush=True)
            for i, r in enumerate(tf_results[:3]):
                if i == 0:
                    continue
                print("           %-25s n=%3d PF=%.3f PnL=%+6.1f%%"
                      % (r["config_id"], r["trades"], r["profit_factor"], r["pnl_pct"]),
                      flush=True)

    print("\n  == MULTI-TF Signal->Entry ==", flush=True)
    for mtf_name, res in sorted(mtf_results.items(),
                                 key=lambda x: x[1].get("profit_factor", 0), reverse=True):
        tag = "VIABLE" if res["profit_factor"] >= 1.15 else ("MARGINAL" if res["profit_factor"] >= 1.05 else "DEAD")
        print("    %-30s [%s] n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%%"
              % (mtf_name, tag, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"]), flush=True)

    if sltp_results:
        print("\n  == SL/TP Optimization ==", flush=True)
        for key, combos in sltp_results.items():
            if combos:
                print("    %-20s best=%s PF=%.3f" % (key, combos[0]["sltp_id"],
                      combos[0]["profit_factor"]), flush=True)

    print("\n  == WALK-FORWARD ==", flush=True)
    passed = [w for w in wf_results if w.get("passed")]
    failed = [w for w in wf_results if not w.get("passed")]
    for w in wf_results:
        print("    %-35s train=%+.1f%%(PF=%.2f) test=%+.1f%%(PF=%.2f) [%s]"
              % (w["name"], w["train_pnl"], w["train_pf"],
                 w["test_pnl"], w["test_pf"], w["tag"]), flush=True)

    if passed:
        print("\n  ** WALK-FORWARD PASSED: **", flush=True)
        for w in passed:
            print("    %s  test_PF=%.2f test_PnL=%+.1f%%"
                  % (w["name"], w["test_pf"], w["test_pnl"]), flush=True)

    # ---- Save ----
    out = {
        "fvg_results": {k: [{kk: vv for kk, vv in r.items() if kk != "params"} for r in v]
                        for k, v in fvg_results.items()},
        "bbs_results": {k: [{kk: vv for kk, vv in r.items() if kk != "params"} for r in v]
                        for k, v in bbs_results.items()},
        "kc_results": {k: [{kk: vv for kk, vv in r.items() if kk != "params"} for r in v]
                       for k, v in kc_results.items()},
        "mtf_results": mtf_results,
        "sltp_optimization": {k: v[:5] for k, v in sltp_results.items()},
        "walk_forward": wf_results,
        "metadata": {"timestamp": ts, "symbols": len(all_data),
                      "timeframes": list(tf_map.keys()),
                      "runtime_sec": round(_time.time() - t0, 1)},
    }
    out_path = RESULTS_DIR / ("backtest_v9_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n  Results saved:", out_path, flush=True)
    print("\nTotal runtime: %d min %d sec" % divmod(int(_time.time() - t0), 60), flush=True)


if __name__ == "__main__":
    main()
