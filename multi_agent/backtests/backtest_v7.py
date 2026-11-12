"""
backtest_v7.py  --  Trend-Following Research Engine  (2026-04-13)

Lesson from v6: Mean reversion + fakeout + FVG retest = no edge on 15m crypto.
ONLY trend/momentum strategies survive. This version tests 4 trend-following approaches.

Strategies:
  BBS      -- Volatility compression → expansion breakout  (core, enhanced grid)
  EMA_MOM  -- EMA crossover + ADX + volume momentum        (new)
  DONCHIAN -- N-bar high/low breakout                      (new, turtle-style)
  KC_SQ    -- Keltner Channel squeeze → expansion          (new, BBS cousin)

Architecture (same as v6):
  Phase 0  Pre-compute indicators + regime   (runs once)
  Phase 1  BBS enhanced grid                 (10 configs)
  Phase 2  EMA Momentum grid                 (12 configs)
  Phase 3  Donchian Breakout grid            (10 configs)
  Phase 4  Keltner Squeeze grid              (10 configs)
  Phase 5  SL/TP optimization for top-3 of each
  Phase 6  Combined portfolio + regime allocation
  Phase 7  Walk-forward validation (ALL top-3 per strategy)
  Phase 8  Final report
"""
from __future__ import annotations
import json, logging, sys, time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

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
from multi_agent.backtests.backtest_v4 import (
    preload_all, _check_180d_available,
    SYMBOLS, BBS_BLACKLIST_CANDIDATES,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
)
import multi_agent.config as cfg

logger = logging.getLogger("v7")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLACKLIST = BBS_BLACKLIST_CANDIDATES
SOFT_CFG = {"confirm_bars": 5, "min_hold_bars": 12, "smooth_alpha": 0.30,
            "min_confidence": 1.0, "transition_margin": 0.5,
            "undefined_threshold": 0.5}

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
    out[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i-1] * (period-1) + tr[i]) / period
    return out

def _adx(hi, lo, cl, period=14):
    n = len(cl)
    adx = np.full(n, 20.0)
    pdm = np.zeros(n); ndm = np.zeros(n)
    tr  = np.zeros(n)
    for i in range(1, n):
        tr[i]  = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up = hi[i] - hi[i-1]; down = lo[i-1] - lo[i]
        pdm[i] = up   if (up > down and up > 0)   else 0
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
        if atr_s > 0:
            pdi = 100 * pdm_s / atr_s
            ndi = 100 * ndm_s / atr_s
        else:
            pdi = ndi = 0
        s = pdi + ndi
        dx = abs(pdi - ndi) / s * 100 if s > 0 else 0
        dx_arr.append(dx)
        if len(dx_arr) == period:
            adx[i] = np.mean(dx_arr)
        elif len(dx_arr) > period:
            adx[i] = (adx[i-1] * (period-1) + dx) / period
    return adx


def compute_extra(D: dict) -> dict:
    """Compute extra indicators for all strategies. Called once per symbol."""
    cl = D["cl15"]; hi = D["hi15"]; lo = D["lo15"]; vo = D["vo15"]
    n = len(cl)
    op = D["df15"]["open"].astype(float).to_numpy()

    # ---- Bollinger Bands ----
    sma20 = _sma(cl, 20)
    std20 = _stdev(cl, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_mid   = sma20
    bb_bw    = np.where(sma20 > 0, (bb_upper - bb_lower) / sma20, 0)

    # ---- Keltner Channel ----
    kc_mid   = _ema(cl, 20)
    atr_kc   = _atr(hi, lo, cl, 10)
    kc_upper = kc_mid + 1.5 * atr_kc
    kc_lower = kc_mid - 1.5 * atr_kc
    # Squeeze: BB inside KC
    kc_squeeze = np.zeros(n, dtype=bool)
    for i in range(20, n):
        if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
            kc_squeeze[i] = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])

    # ---- EMAs for momentum ----
    ema8   = _ema(cl, 8)
    ema13  = _ema(cl, 13)
    ema21  = _ema(cl, 21)
    ema50  = _ema(cl, 50)
    ema100 = _ema(cl, 100)
    ema200 = _ema(cl, 200)

    # ---- RSI, ADX, ATR, Volume ----
    rsi14    = _rsi(cl, 14)
    adx14    = D["adx15"] if "adx15" in D else _adx(hi, lo, cl, 14)
    atr14    = D["atr15"] if "atr15" in D else _atr(hi, lo, cl, 14)
    vol_ma20 = _sma(vo, 20)
    vol_ratio = np.where(vol_ma20 > 0, vo / vol_ma20, 1.0)

    # ---- Donchian Channel (multiple periods) ----
    don = {}
    for period in [20, 30, 50, 55, 100]:
        dc_hi = np.full(n, np.nan)
        dc_lo = np.full(n, np.nan)
        for i in range(period, n):
            dc_hi[i] = np.max(hi[i-period:i])
            dc_lo[i] = np.min(lo[i-period:i])
        don[period] = {"hi": dc_hi, "lo": dc_lo}

    # ---- Keltner Squeeze run-length ----
    kc_sq_len = np.zeros(n, dtype=int)
    for i in range(1, n):
        if kc_squeeze[i]:
            kc_sq_len[i] = kc_sq_len[i-1] + 1

    return {
        "open": op, "n": n,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid, "bb_bw": bb_bw,
        "kc_upper": kc_upper, "kc_lower": kc_lower, "kc_mid": kc_mid,
        "kc_squeeze": kc_squeeze, "kc_sq_len": kc_sq_len,
        "ema8": ema8, "ema13": ema13, "ema21": ema21,
        "ema50": ema50, "ema100": ema100, "ema200": ema200,
        "rsi": rsi14, "adx": adx14, "atr": atr14,
        "vol_ma": vol_ma20, "vol_ratio": vol_ratio,
        "donchian": don,
    }


# ===========================================================================
#  SIGNAL DETECTORS
# ===========================================================================

def gen_bbs_signals(D, extra, regime_arr, params):
    """BBS signals with parameterized thresholds."""
    sigs = {}
    if D["squeeze"] is None:
        return sigs
    min_sq   = params.get("min_squeeze", 15)
    adx_th   = params.get("adx_thresh", 25.0)
    vol_m    = params.get("vol_mult", 1.5)
    use_htf  = params.get("htf_filter", True)
    use_hf   = params.get("headfake_filter", True)
    atr_min_pct = params.get("atr_min_pct", 0.0)
    regimes  = params.get("regimes", {"TREND": 1.0, "TRANSITION": 0.85})
    n = D["n_bars"]
    for i in range(300, n):
        r = regime_arr[i]
        sm = regimes.get(r, 0.0)
        if sm < 0.05:
            continue
        if atr_min_pct > 0:
            atr_pct = extra["atr"][i] / D["cl15"][i] if D["cl15"][i] > 0 else 0
            if np.isnan(atr_pct) or atr_pct < atr_min_pct:
                continue
        bbs_s = detect_squeeze_signal(D["squeeze"], bar_idx=i,
                                      min_squeeze_bars=min_sq,
                                      adx_threshold=adx_th,
                                      vol_mult=vol_m)
        if not bbs_s:
            continue
        if use_htf:
            ema200 = D["ind_1h"]["ema200_1h"]
            cl1h   = D["ind_1h"]["closes_1h_arr"]
            ts_i   = D["ts15"][i]
            htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
            if htf_idx < 210:
                continue
            htf_d = "LONG" if cl1h[htf_idx] > ema200[htf_idx] else "SHORT"
            if bbs_s["direction"] != htf_d:
                continue
        if use_hf and check_head_fake(D["squeeze"], i, 2):
            continue
        ce_sl = chandelier_exit_sl(D["hi15"], D["lo15"], D["squeeze"]["ce_atr"],
                                   bbs_s["direction"], i,
                                   ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                                   ce_mult=cfg.BB_SQUEEZE_CE_MULT)
        rv = abs(D["cl15"][i] - ce_sl)
        if rv <= 0 or rv / D["cl15"][i] > 0.05:
            continue
        sigs[i] = {"direction": bbs_s["direction"],
                    "entry_price": float(D["cl15"][i]),
                    "default_sl": float(ce_sl), "regime": r, "size_m": sm}
    return sigs


def gen_ema_mom_signals(D, extra, regime_arr, params):
    """EMA crossover momentum: fast EMA crosses slow EMA with ADX + volume."""
    sigs = {}
    fast_p    = params.get("fast", 8)
    slow_p    = params.get("slow", 21)
    adx_min   = params.get("adx_min", 20.0)     # 0 = no filter
    vol_mult  = params.get("vol_mult", 1.2)      # 0 = no filter
    use_htf   = params.get("htf_filter", False)
    rsi_lo    = params.get("rsi_lo", 30.0)        # avoid overbought short / oversold long
    rsi_hi    = params.get("rsi_hi", 70.0)
    regimes   = params.get("regimes", None)
    sl_mult   = params.get("sl_atr_mult", 2.0)

    cl = D["cl15"]; n = extra["n"]
    atr = extra["atr"]; adx = extra["adx"]; rsi = extra["rsi"]
    vol_r = extra["vol_ratio"]

    # Select EMA arrays based on params
    ema_map = {8: extra["ema8"], 13: extra["ema13"], 21: extra["ema21"],
               50: extra["ema50"], 100: extra["ema100"], 200: extra["ema200"]}
    fast_ema = ema_map.get(fast_p)
    slow_ema = ema_map.get(slow_p)
    if fast_ema is None or slow_ema is None:
        return sigs

    for i in range(max(300, slow_p + 2), n):
        if np.isnan(fast_ema[i]) or np.isnan(slow_ema[i]):
            continue
        if np.isnan(fast_ema[i-1]) or np.isnan(slow_ema[i-1]):
            continue

        r = regime_arr[i] if i < len(regime_arr) else "UNDEFINED"
        if regimes is not None:
            sm = regimes.get(r, 0.0)
            if sm < 0.05:
                continue
        else:
            sm = 1.0

        # ADX filter
        if adx_min > 0 and (np.isnan(adx[i]) or adx[i] < adx_min):
            continue

        # Volume filter
        if vol_mult > 0 and vol_r[i] < vol_mult:
            continue

        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01

        # LONG: fast crosses above slow
        if fast_ema[i] > slow_ema[i] and fast_ema[i-1] <= slow_ema[i-1]:
            # Don't enter overbought
            if rsi_hi > 0 and not np.isnan(rsi[i]) and rsi[i] > rsi_hi:
                continue
            # HTF alignment
            if use_htf:
                ema200_1h = D["ind_1h"]["ema200_1h"]
                cl1h = D["ind_1h"]["closes_1h_arr"]
                ts_i = D["ts15"][i]
                htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
                if htf_idx < 210 or cl1h[htf_idx] <= ema200_1h[htf_idx]:
                    continue
            sl_v = cl[i] - atr_v * sl_mult
            sigs[i] = {"direction": "LONG", "entry_price": float(cl[i]),
                        "default_sl": float(sl_v), "regime": r, "size_m": sm}

        # SHORT: fast crosses below slow
        elif fast_ema[i] < slow_ema[i] and fast_ema[i-1] >= slow_ema[i-1]:
            if rsi_lo > 0 and not np.isnan(rsi[i]) and rsi[i] < rsi_lo:
                continue
            if use_htf:
                ema200_1h = D["ind_1h"]["ema200_1h"]
                cl1h = D["ind_1h"]["closes_1h_arr"]
                ts_i = D["ts15"][i]
                htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
                if htf_idx < 210 or cl1h[htf_idx] >= ema200_1h[htf_idx]:
                    continue
            sl_v = cl[i] + atr_v * sl_mult
            sigs[i] = {"direction": "SHORT", "entry_price": float(cl[i]),
                        "default_sl": float(sl_v), "regime": r, "size_m": sm}

    return sigs


def gen_donchian_signals(D, extra, regime_arr, params):
    """Donchian channel breakout (turtle-trading style)."""
    sigs = {}
    period    = params.get("period", 55)
    adx_min   = params.get("adx_min", 20.0)
    vol_mult  = params.get("vol_mult", 0.0)     # 0 = no filter
    use_htf   = params.get("htf_filter", False)
    cooldown  = params.get("cooldown", 10)       # min bars between signals same sym
    regimes   = params.get("regimes", None)
    sl_mult   = params.get("sl_atr_mult", 2.0)

    cl = D["cl15"]; hi = D["hi15"]; lo = D["lo15"]; n = extra["n"]
    atr = extra["atr"]; adx = extra["adx"]
    vol_r = extra["vol_ratio"]

    don_data = extra["donchian"].get(period)
    if don_data is None:
        # Compute on the fly for non-standard periods
        dc_hi = np.full(n, np.nan)
        dc_lo = np.full(n, np.nan)
        for i in range(period, n):
            dc_hi[i] = np.max(hi[i-period:i])
            dc_lo[i] = np.min(lo[i-period:i])
        don_data = {"hi": dc_hi, "lo": dc_lo}

    dc_hi = don_data["hi"]
    dc_lo = don_data["lo"]
    last_sig = -999

    for i in range(max(300, period + 1), n):
        if np.isnan(dc_hi[i]) or np.isnan(dc_lo[i]):
            continue

        r = regime_arr[i] if i < len(regime_arr) else "UNDEFINED"
        if regimes is not None:
            sm = regimes.get(r, 0.0)
            if sm < 0.05:
                continue
        else:
            sm = 1.0

        if i - last_sig < cooldown:
            continue

        if adx_min > 0 and (np.isnan(adx[i]) or adx[i] < adx_min):
            continue

        if vol_mult > 0 and vol_r[i] < vol_mult:
            continue

        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01

        # LONG: close above N-bar high
        if cl[i] > dc_hi[i]:
            if use_htf:
                ema200_1h = D["ind_1h"]["ema200_1h"]
                cl1h = D["ind_1h"]["closes_1h_arr"]
                ts_i = D["ts15"][i]
                htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
                if htf_idx < 210 or cl1h[htf_idx] <= ema200_1h[htf_idx]:
                    continue
            sl_v = cl[i] - atr_v * sl_mult
            sigs[i] = {"direction": "LONG", "entry_price": float(cl[i]),
                        "default_sl": float(sl_v), "regime": r, "size_m": sm}
            last_sig = i

        # SHORT: close below N-bar low
        elif cl[i] < dc_lo[i]:
            if use_htf:
                ema200_1h = D["ind_1h"]["ema200_1h"]
                cl1h = D["ind_1h"]["closes_1h_arr"]
                ts_i = D["ts15"][i]
                htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
                if htf_idx < 210 or cl1h[htf_idx] >= ema200_1h[htf_idx]:
                    continue
            sl_v = cl[i] + atr_v * sl_mult
            sigs[i] = {"direction": "SHORT", "entry_price": float(cl[i]),
                        "default_sl": float(sl_v), "regime": r, "size_m": sm}
            last_sig = i

    return sigs


def gen_kc_squeeze_signals(D, extra, regime_arr, params):
    """Keltner Channel squeeze: BB contracts inside KC, then expands out.
    Similar to BBS but uses KC for squeeze detection + momentum for direction."""
    sigs = {}
    min_sq    = params.get("min_squeeze", 10)    # min bars in squeeze
    use_htf   = params.get("htf_filter", False)
    adx_min   = params.get("adx_min", 15.0)
    vol_mult  = params.get("vol_mult", 1.0)
    mom_bars  = params.get("mom_bars", 12)       # momentum lookback for direction
    regimes   = params.get("regimes", None)
    sl_mult   = params.get("sl_atr_mult", 2.0)

    cl = D["cl15"]; hi = D["hi15"]; lo = D["lo15"]; n = extra["n"]
    atr = extra["atr"]; adx = extra["adx"]
    vol_r = extra["vol_ratio"]
    kc_sq = extra["kc_squeeze"]
    kc_sq_len = extra["kc_sq_len"]
    bb_upper = extra["bb_upper"]; bb_lower = extra["bb_lower"]
    kc_upper = extra["kc_upper"]; kc_lower = extra["kc_lower"]

    for i in range(max(300, mom_bars + 2), n):
        # Detect squeeze END: was in squeeze, now out
        if kc_sq[i]:
            continue  # still in squeeze
        if kc_sq_len[i-1] < min_sq:
            continue  # previous squeeze too short

        r = regime_arr[i] if i < len(regime_arr) else "UNDEFINED"
        if regimes is not None:
            sm = regimes.get(r, 0.0)
            if sm < 0.05:
                continue
        else:
            sm = 1.0

        if adx_min > 0 and (np.isnan(adx[i]) or adx[i] < adx_min):
            continue

        if vol_mult > 0 and vol_r[i] < vol_mult:
            continue

        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01

        # Determine direction via momentum (linear regression slope over mom_bars)
        if i < mom_bars:
            continue
        seg = cl[i-mom_bars:i+1]
        if len(seg) < mom_bars or np.isnan(seg).any():
            continue
        x = np.arange(len(seg))
        slope = np.polyfit(x, seg, 1)[0]

        if slope > 0:
            direction = "LONG"
            # Confirm: BB expanding upward
            if not np.isnan(bb_upper[i]) and not np.isnan(kc_upper[i]):
                if bb_upper[i] <= kc_upper[i]:
                    continue  # BB hasn't expanded above KC yet
            if use_htf:
                ema200_1h = D["ind_1h"]["ema200_1h"]
                cl1h = D["ind_1h"]["closes_1h_arr"]
                ts_i = D["ts15"][i]
                htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
                if htf_idx < 210 or cl1h[htf_idx] <= ema200_1h[htf_idx]:
                    continue
            sl_v = cl[i] - atr_v * sl_mult
        else:
            direction = "SHORT"
            if not np.isnan(bb_lower[i]) and not np.isnan(kc_lower[i]):
                if bb_lower[i] >= kc_lower[i]:
                    continue
            if use_htf:
                ema200_1h = D["ind_1h"]["ema200_1h"]
                cl1h = D["ind_1h"]["closes_1h_arr"]
                ts_i = D["ts15"][i]
                htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
                if htf_idx < 210 or cl1h[htf_idx] >= ema200_1h[htf_idx]:
                    continue
            sl_v = cl[i] + atr_v * sl_mult

        sigs[i] = {"direction": direction, "entry_price": float(cl[i]),
                    "default_sl": float(sl_v), "regime": r, "size_m": sm}

    return sigs


# ===========================================================================
#  SL / TP COMPUTATION
# ===========================================================================

def _sl_atr(d, entry, atr, mult):
    return (entry - atr * mult) if d == "LONG" else (entry + atr * mult)

def compute_sl(sl_cfg, d, entry, atr, hi, lo, i, default_sl):
    t = sl_cfg["type"]
    if t == "default":
        sl = default_sl
    elif t == "atr":
        sl = _sl_atr(d, entry, atr, sl_cfg["mult"])
    elif t == "structure":
        s = max(0, i - sl_cfg.get("n_bars", 5) + 1)
        sl = float(np.min(lo[s:i+1])) if d == "LONG" else float(np.max(hi[s:i+1]))
    else:
        sl = default_sl
    if d == "LONG"  and sl >= entry: sl = entry * 0.98
    if d == "SHORT" and sl <= entry: sl = entry * 1.02
    return float(sl)


# ===========================================================================
#  GENERIC SIMULATION ENGINE (from v6, unchanged)
# ===========================================================================

def run_sim(signals_by_sym, all_data, extra_ind, sl_cfg, tp_cfg,
            mgt_cfg=None, risk_pt=0.010, cap_frac=0.50, max_pos=4,
            start_frac=0.0, end_frac=1.0):
    """Generic simulation."""
    mgt_cfg = mgt_cfg or {"be_r": 0.0}
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    open_pos = {}
    acnt = 0
    completed = []
    max_n = max(d["n_bars"] for d in all_data.values())
    i_start = max(300, int(max_n * start_frac))
    i_end = int(max_n * end_frac)

    for i in range(i_start, i_end):
        for sym, D in all_data.items():
            if i >= D["n_bars"]:
                continue
            cl_i = D["cl15"][i]; hi_i = D["hi15"][i]; lo_i = D["lo15"][i]
            atr_i = D["atr15"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01
            ex = extra_ind.get(sym, {})

            # ---- manage open position ----
            if sym in open_pos:
                pos = open_pos[sym]; d = pos["d"]; entry = pos["entry"]
                sl = pos["sl"]; r = pos["risk"]; q = pos["qty"]
                bars = i - pos["bar_i"]
                # MFE / MAE
                if d == "LONG":
                    pos["mfe"] = max(pos["mfe"], (hi_i - entry) / r)
                    pos["mae"] = max(pos["mae"], (entry - lo_i) / r)
                else:
                    pos["mfe"] = max(pos["mfe"], (entry - lo_i) / r)
                    pos["mae"] = max(pos["mae"], (hi_i - entry) / r)
                pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)

                # BE management
                be_r = mgt_cfg.get("be_r", 0.0)
                if be_r > 0 and pnl_r >= be_r and not pos.get("be"):
                    buf = entry * 0.0005
                    nb = (entry + buf) if d == "LONG" else (entry - buf)
                    if (d == "LONG" and nb > sl) or (d == "SHORT" and nb < sl):
                        pos["sl"] = nb; sl = nb
                    pos["be"] = True

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = False; tp_price = 0.0; exit_rsn = "sl_hit" if hit_sl else ""

                tp_type = tp_cfg["type"]
                if not hit_sl:
                    if tp_type == "fixed_r":
                        if pnl_r >= tp_cfg["r"]:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_fixed"
                    elif tp_type == "trail_atr":
                        if pnl_r >= tp_cfg.get("trail_start_r", 1.0):
                            tm = tp_cfg.get("trail_mult", 2.0)
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
                            if lo_i <= eff: hit_sl = True; sl = eff; exit_rsn = "trail_atr"
                        else:
                            eff = min(sl, eff) if eff > 0 else sl
                            if hi_i >= eff: hit_sl = True; sl = eff; exit_rsn = "trail_atr"
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
                    elif tp_type == "time_exit":
                        if bars >= tp_cfg.get("max_bars", 50):
                            hit_tp = True; tp_price = cl_i; exit_rsn = "time_exit"
                        elif tp_cfg.get("min_r", 0) > 0 and pnl_r >= tp_cfg["min_r"]:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_min_r"

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
                    completed.append({"sym": sym, "d": d,
                        "exit_r": round(exit_r, 3),
                        "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                        "bars": bars, "won": total_net > 0,
                        "pnl_usd": round(total_net, 4),
                        "reason": exit_rsn, "regime": pos.get("regime", "?")})
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
            sm = sig.get("size_m", 1.0)
            if sm < 0.05:
                continue
            new_sl = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                atr_i, D["hi15"], D["lo15"], i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue
            ab = balance * cap_frac
            ru = ab * risk_pt * sm
            qty = ru / rd
            mg = sig["entry_price"] * qty / MAX_LEVERAGE
            if mg > ab * 0.4:
                qty = ab * 0.4 * MAX_LEVERAGE / sig["entry_price"]
            if qty * sig["entry_price"] < 5.0:
                continue
            ef = sig["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
            balance -= ef
            open_pos[sym] = {"d": sig["direction"],
                "entry": sig["entry_price"],
                "sl": new_sl, "risk": rd, "qty": qty, "bar_i": i,
                "mfe": 0.0, "mae": 0.0, "partial": False, "ppnl": 0.0,
                "be": False, "regime": sig.get("regime", "?")}
            acnt += 1

    # ---- aggregate ----
    if not completed:
        return {"trades": 0, "wr_pct": 0, "pnl_pct": 0, "profit_factor": 0,
                "max_dd_pct": 0, "mfe_cap": 0, "giveback": 0, "avg_bars": 0,
                "exit_reasons": {}, "per_regime": {}}
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    n_ = len(completed)
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / n_ * 100
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100
    mv = [t for t in completed if t["mfe"] >= 0.5]
    mfe_cap = float(np.mean([max(0, t["exit_r"]) / t["mfe"]
                              for t in mv]) * 100) if mv else 0
    r1 = [t for t in completed if t["mfe"] >= 1.0]
    gb = len([t for t in r1 if t["exit_r"] < 0.5]) / len(r1) * 100 if r1 else 0
    reasons = defaultdict(int)
    for t in completed:
        reasons[t["reason"]] += 1
    regimes_seen = set(t["regime"] for t in completed)
    per_regime = {}
    for rg in regimes_seen:
        rt = [t for t in completed if t["regime"] == rg]
        rw = [t for t in rt if t["won"]]
        rl = [t for t in rt if not t["won"]]
        rgp = sum(t["pnl_usd"] for t in rw)
        rgl = abs(sum(t["pnl_usd"] for t in rl))
        rpf = rgp / rgl if rgl > 0 else (999.0 if rgp > 0 else 0)
        per_regime[rg] = {"n": len(rt),
                          "wr": round(len(rw)/len(rt)*100, 1) if rt else 0,
                          "pf": round(rpf, 2)}
    return {
        "trades": n_, "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 1), "pnl_pct": round(pnl_pct, 2),
        "profit_factor": round(min(pf, 99), 3), "max_dd_pct": round(max_dd, 2),
        "avg_exit_r": round(float(np.mean([t["exit_r"] for t in completed])), 3),
        "avg_mfe": round(float(np.mean([t["mfe"] for t in completed])), 3),
        "avg_mae": round(float(np.mean([t["mae"] for t in completed])), 3),
        "mfe_cap": round(mfe_cap, 1), "giveback": round(gb, 1),
        "avg_bars": round(float(np.mean([t["bars"] for t in completed])), 1),
        "exit_reasons": dict(reasons), "per_regime": per_regime,
        "final_bal": round(balance, 2), "completed": completed,
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

# ---- BBS (10 configs: wider grid around known sweet spot) ----
BBS_CFGS = [
    {"id": "bbs_prod", "desc": "current production",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_atr25", "desc": "ATR2.5 SL",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_loose", "desc": "loose sq8 adx15",
     "params": {"min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_loose_atr25", "desc": "loose + ATR2.5 SL",
     "params": {"min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_mid", "desc": "mid sq10 adx20",
     "params": {"min_squeeze": 10, "adx_thresh": 20.0, "vol_mult": 1.2,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_mid_atr25", "desc": "mid + ATR2.5 SL",
     "params": {"min_squeeze": 10, "adx_thresh": 20.0, "vol_mult": 1.2,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_nohtf", "desc": "no HTF filter",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": False, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_partial", "desc": "partial TP: 50%@2R rest@4R",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "atr", "mult": 2.5},
     "tp": {"type": "partial_fixed", "r1": 2.0, "r2": 4.0, "frac": 0.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_trail", "desc": "ATR trail after 2R",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "atr", "mult": 2.5},
     "tp": {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 3.0},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_allreg_atr25", "desc": "all regimes + ATR2.5",
     "params": {"min_squeeze": 12, "adx_thresh": 20.0, "vol_mult": 1.2,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85, "SQUEEZE": 0.5, "RANGE": 0.3}},
     "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
]

# ---- EMA Momentum (12 configs) ----
EMA_CFGS = [
    {"id": "ema_8_21", "desc": "EMA8/21 adx20 vol1.2",
     "params": {"fast": 8, "slow": 21, "adx_min": 20, "vol_mult": 1.2, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_8_21_htf", "desc": "EMA8/21 +HTF",
     "params": {"fast": 8, "slow": 21, "adx_min": 20, "vol_mult": 1.2,
                "htf_filter": True, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_8_21_loose", "desc": "EMA8/21 adx15 no vol",
     "params": {"fast": 8, "slow": 21, "adx_min": 15, "vol_mult": 0, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_13_50", "desc": "EMA13/50 adx20 vol1.2",
     "params": {"fast": 13, "slow": 50, "adx_min": 20, "vol_mult": 1.2, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_13_50_htf", "desc": "EMA13/50 +HTF",
     "params": {"fast": 13, "slow": 50, "adx_min": 20, "vol_mult": 1.2,
                "htf_filter": True, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_13_50_trend", "desc": "EMA13/50 TREND only",
     "params": {"fast": 13, "slow": 50, "adx_min": 20, "vol_mult": 1.2,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_21_50", "desc": "EMA21/50 adx20 vol1.2",
     "params": {"fast": 21, "slow": 50, "adx_min": 20, "vol_mult": 1.2, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_21_100", "desc": "EMA21/100 adx20",
     "params": {"fast": 21, "slow": 100, "adx_min": 20, "vol_mult": 0, "sl_atr_mult": 2.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_8_21_3r", "desc": "EMA8/21 +3.5R target",
     "params": {"fast": 8, "slow": 21, "adx_min": 20, "vol_mult": 1.2, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_8_21_trail", "desc": "EMA8/21 +trail",
     "params": {"fast": 8, "slow": 21, "adx_min": 20, "vol_mult": 1.2, "sl_atr_mult": 2.0},
     "sl": {"type": "default"},
     "tp": {"type": "trail_atr", "trail_start_r": 1.5, "trail_mult": 2.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_8_50_htf", "desc": "EMA8/50 +HTF wide SL",
     "params": {"fast": 8, "slow": 50, "adx_min": 20, "vol_mult": 1.0,
                "htf_filter": True, "sl_atr_mult": 2.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "ema_13_21_fast", "desc": "EMA13/21 fast adx25",
     "params": {"fast": 13, "slow": 21, "adx_min": 25, "vol_mult": 1.5, "sl_atr_mult": 1.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
]

# ---- Donchian Breakout (10 configs) ----
DON_CFGS = [
    {"id": "don_55", "desc": "55-bar breakout",
     "params": {"period": 55, "adx_min": 20, "cooldown": 10, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_55_htf", "desc": "55-bar +HTF",
     "params": {"period": 55, "adx_min": 20, "cooldown": 10, "htf_filter": True,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_55_trend", "desc": "55-bar TREND only",
     "params": {"period": 55, "adx_min": 20, "cooldown": 10,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_100", "desc": "100-bar breakout",
     "params": {"period": 100, "adx_min": 20, "cooldown": 20, "sl_atr_mult": 2.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_30", "desc": "30-bar breakout",
     "params": {"period": 30, "adx_min": 20, "cooldown": 8, "sl_atr_mult": 1.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_20", "desc": "20-bar breakout",
     "params": {"period": 20, "adx_min": 20, "cooldown": 6, "sl_atr_mult": 1.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_55_vol", "desc": "55-bar +vol filter",
     "params": {"period": 55, "adx_min": 15, "vol_mult": 1.5, "cooldown": 10,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_55_trail", "desc": "55-bar +trail",
     "params": {"period": 55, "adx_min": 20, "cooldown": 10, "sl_atr_mult": 2.0},
     "sl": {"type": "default"},
     "tp": {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 2.5},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_55_noadx", "desc": "55-bar no ADX",
     "params": {"period": 55, "adx_min": 0, "cooldown": 10, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
    {"id": "don_50_htf_vol", "desc": "50-bar +HTF +vol",
     "params": {"period": 50, "adx_min": 20, "vol_mult": 1.3, "cooldown": 10,
                "htf_filter": True, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.50, "max_pos": 4},
]

# ---- Keltner Squeeze (10 configs) ----
KC_CFGS = [
    {"id": "kc_base", "desc": "sq10 adx15 vol1.0",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_strict", "desc": "sq20 adx20 vol1.3",
     "params": {"min_squeeze": 20, "adx_min": 20, "vol_mult": 1.3, "mom_bars": 12,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_loose", "desc": "sq6 adx12 no vol",
     "params": {"min_squeeze": 6, "adx_min": 12, "vol_mult": 0, "mom_bars": 12,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_htf", "desc": "sq10 +HTF",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                "htf_filter": True, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_trend", "desc": "sq10 TREND only",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}, "sl_atr_mult": 2.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_long_sq", "desc": "sq30 adx15",
     "params": {"min_squeeze": 30, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 20,
                "sl_atr_mult": 2.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_atr25", "desc": "sq10 ATR2.5 SL",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                "sl_atr_mult": 2.0},
     "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_partial", "desc": "sq10 partial 50%@2R rest@4R",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"},
     "tp": {"type": "partial_fixed", "r1": 2.0, "r2": 4.0, "frac": 0.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_trail", "desc": "sq10 trail after 2R",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                "sl_atr_mult": 2.0},
     "sl": {"type": "default"},
     "tp": {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 2.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
    {"id": "kc_fast_mom", "desc": "sq10 fast momentum (6 bars)",
     "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 6,
                "sl_atr_mult": 1.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.60, "max_pos": 4},
]


# ===========================================================================
#  PHASE 0 -- PRE-COMPUTE
# ===========================================================================

def phase0(all_data):
    extra_ind = {}
    regime_cache = {}
    total = len(all_data)
    for k, (sym, D) in enumerate(all_data.items()):
        print("  [%d/%d] %s ..." % (k+1, total, sym), flush=True)
        extra_ind[sym] = compute_extra(D)
        n = D["n_bars"]
        rarr = ["UNDEFINED"] * n
        det = SoftRegimeDetector(SOFT_CFG)
        for i in range(300, n):
            ts_i = D["ts15"][i]
            htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
            if htf_idx < 210:
                continue
            rs = det.update(sym, D["df1h"], htf_idx, D["df4h"],
                            **(D["ind_1h"]), **(D["ind_4h"]))
            rarr[i] = rs.regime.value
        regime_cache[sym] = rarr
    return extra_ind, regime_cache


# ===========================================================================
#  PHASE RUNNERS
# ===========================================================================

def _run_phase(name, cfgs, gen_fn, all_data, extra_ind, regime_cache):
    print("\n" + "=" * 70, flush=True)
    print("  PHASE -- %s  (%d configs)" % (name, len(cfgs)), flush=True)
    print("=" * 70, flush=True)
    results = []
    for k, c in enumerate(cfgs):
        # Generate signals for all symbols
        all_sigs = {}
        total_sigs = 0
        for sym, D in all_data.items():
            s = gen_fn(D, extra_ind[sym], regime_cache[sym], c["params"])
            all_sigs[sym] = s
            total_sigs += len(s)
        # Run simulation
        res = run_sim(all_sigs, all_data, extra_ind,
                      c["sl"], c["tp"],
                      mgt_cfg=c.get("mgt", {}),
                      risk_pt=c.get("risk", 0.010),
                      cap_frac=c.get("frac", 0.50),
                      max_pos=c.get("max_pos", 4))
        res["config_id"] = c["id"]
        res["desc"] = c["desc"]
        res["total_sigs"] = total_sigs
        res["score"] = _score(res)
        # Remove completed trades from JSON (save space)
        res.pop("completed", None)
        results.append(res)
        pf_s = "PF=%.2f" % res["profit_factor"]
        pnl_s = "PnL=%+.1f%%" % res["pnl_pct"]
        dd_s = "DD=%.1f%%" % res["max_dd_pct"]
        print("  [%02d/%02d] %-40s sigs=%5d n=%3d %s %s %s score=%.3f"
              % (k+1, len(cfgs), c["id"] + " " + c["desc"],
                 total_sigs, res["trades"], pf_s, pnl_s, dd_s, res["score"]),
              flush=True)
    # Sort by score
    results.sort(key=lambda x: x["score"], reverse=True)
    # Print top-3
    print("\n  Top-3:", flush=True)
    for i, r in enumerate(results[:3]):
        regime_str = ""
        for rg, v in sorted(r["per_regime"].items()):
            regime_str += "\n       %s: n=%d WR=%d%% PF=%.2f" % (rg, v["n"], v["wr"], v["pf"])
        print("    %d. %-36s PF=%.3f PnL=%+.1f%% WR=%d%% DD=%.1f%% n=%d MFEcap=%.1f%%%s"
              % (i+1, r["config_id"], r["profit_factor"], r["pnl_pct"],
                 r["wr_pct"], r["max_dd_pct"], r["trades"],
                 r["mfe_cap"], regime_str), flush=True)
    return results


def _run_sltp_opt(name, gen_fn, base_params, all_data, extra_ind, regime_cache):
    """Phase 5: SL/TP deep optimization for a given strategy's top config."""
    print("\n  -- SL/TP optimization for %s --" % name, flush=True)
    sl_tp_combos = [
        {"id": "atr1.5_2.5r", "sl": {"type": "atr", "mult": 1.5}, "tp": {"type": "fixed_r", "r": 2.5}},
        {"id": "atr2.0_3.0r", "sl": {"type": "atr", "mult": 2.0}, "tp": {"type": "fixed_r", "r": 3.0}},
        {"id": "atr2.0_3.5r", "sl": {"type": "atr", "mult": 2.0}, "tp": {"type": "fixed_r", "r": 3.5}},
        {"id": "atr2.5_3.0r", "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.0}},
        {"id": "atr2.5_3.5r", "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5}},
        {"id": "atr2.5_4.0r", "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 4.0}},
        {"id": "atr3.0_3.5r", "sl": {"type": "atr", "mult": 3.0}, "tp": {"type": "fixed_r", "r": 3.5}},
        {"id": "default_2.5r", "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5}},
        {"id": "default_3.0r", "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0}},
        {"id": "default_3.5r", "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5}},
        {"id": "atr2.0_partial", "sl": {"type": "atr", "mult": 2.0},
         "tp": {"type": "partial_fixed", "r1": 2.0, "r2": 4.0, "frac": 0.5}},
        {"id": "atr2.5_trail", "sl": {"type": "atr", "mult": 2.5},
         "tp": {"type": "trail_atr", "trail_start_r": 2.0, "trail_mult": 2.5}},
    ]
    # Generate signals once
    all_sigs = {}
    for sym, D in all_data.items():
        all_sigs[sym] = gen_fn(D, extra_ind[sym], regime_cache[sym], base_params)
    results = []
    for combo in sl_tp_combos:
        res = run_sim(all_sigs, all_data, extra_ind,
                      combo["sl"], combo["tp"],
                      risk_pt=0.010, cap_frac=0.60, max_pos=4)
        res["config_id"] = combo["id"]
        res["score"] = _score(res)
        res.pop("completed", None)
        results.append(res)
        print("    %-25s n=%3d PF=%.2f PnL=%+.1f%% DD=%.1f%%"
              % (combo["id"], res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"]), flush=True)
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


# ===========================================================================
#  WALK-FORWARD
# ===========================================================================

def walk_forward(name, gen_fn, cfg_entry, all_data, extra_ind, regime_cache):
    """67/33 walk-forward split."""
    all_sigs = {}
    for sym, D in all_data.items():
        all_sigs[sym] = gen_fn(D, extra_ind[sym], regime_cache[sym], cfg_entry["params"])
    train = run_sim(all_sigs, all_data, extra_ind,
                    cfg_entry["sl"], cfg_entry["tp"],
                    mgt_cfg=cfg_entry.get("mgt", {}),
                    risk_pt=cfg_entry.get("risk", 0.010),
                    cap_frac=cfg_entry.get("frac", 0.50),
                    max_pos=cfg_entry.get("max_pos", 4),
                    start_frac=0.0, end_frac=0.67)
    test = run_sim(all_sigs, all_data, extra_ind,
                   cfg_entry["sl"], cfg_entry["tp"],
                   mgt_cfg=cfg_entry.get("mgt", {}),
                   risk_pt=cfg_entry.get("risk", 0.010),
                   cap_frac=cfg_entry.get("frac", 0.50),
                   max_pos=cfg_entry.get("max_pos", 4),
                   start_frac=0.67, end_frac=1.0)
    train.pop("completed", None)
    test.pop("completed", None)
    train_pnl = train["pnl_pct"]
    test_pnl  = test["pnl_pct"]
    delta = test_pnl - train_pnl
    degrad = delta / abs(train_pnl) * 100 if abs(train_pnl) > 0.1 else 0
    passed = test["profit_factor"] >= 1.0 and test_pnl > -5.0
    tag = "PASS" if passed else "FAIL"
    print("  %-10s %-25s train=%+6.1f%% test=%+6.1f%% delta=%+.1f%% PF_test=%.2f n_test=%d [%s]"
          % (name, cfg_entry["id"], train_pnl, test_pnl, delta,
             test["profit_factor"], test["trades"], tag), flush=True)
    return {"name": name, "id": cfg_entry["id"],
            "train_pnl": round(train_pnl, 2), "test_pnl": round(test_pnl, 2),
            "train_pf": train["profit_factor"], "test_pf": test["profit_factor"],
            "train_n": train["trades"], "test_n": test["trades"],
            "delta": round(delta, 2), "passed": passed, "tag": tag}


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("backtest_v7.py -- Trend-Following Research Engine", flush=True)
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    print("Data: 180d", flush=True)
    t0 = _time.time()

    # ---- load data ----
    all_data = preload_all("180d", blacklist=set())
    print("Symbols loaded:", len(all_data), flush=True)

    # ---- Phase 0 ----
    print("\n  Phase 0 -- Pre-computing indicators + regime...", flush=True)
    p0_t = _time.time()
    extra_ind, regime_cache = phase0(all_data)
    print("  Phase 0 done in %d sec" % int(_time.time() - p0_t), flush=True)

    # ---- Phase 1: BBS ----
    bbs_results = _run_phase("PHASE 1 -- BBS", BBS_CFGS,
                              gen_bbs_signals, all_data, extra_ind, regime_cache)

    # ---- Phase 2: EMA Momentum ----
    ema_results = _run_phase("PHASE 2 -- EMA_MOM", EMA_CFGS,
                              gen_ema_mom_signals, all_data, extra_ind, regime_cache)

    # ---- Phase 3: Donchian ----
    don_results = _run_phase("PHASE 3 -- DONCHIAN", DON_CFGS,
                              gen_donchian_signals, all_data, extra_ind, regime_cache)

    # ---- Phase 4: Keltner Squeeze ----
    kc_results = _run_phase("PHASE 4 -- KC_SQUEEZE", KC_CFGS,
                             gen_kc_squeeze_signals, all_data, extra_ind, regime_cache)

    all_strategy_results = {
        "BBS": bbs_results, "EMA_MOM": ema_results,
        "DONCHIAN": don_results, "KC_SQUEEZE": kc_results,
    }

    # ---- Phase 5: SL/TP optimization for viable top configs ----
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 5 -- SL/TP Deep Optimization (viable strategies)", flush=True)
    print("=" * 70, flush=True)
    sltp_results = {}
    viable_threshold = 1.05
    gen_fns = {
        "BBS": gen_bbs_signals, "EMA_MOM": gen_ema_mom_signals,
        "DONCHIAN": gen_donchian_signals, "KC_SQUEEZE": gen_kc_squeeze_signals,
    }
    cfg_lists = {
        "BBS": BBS_CFGS, "EMA_MOM": EMA_CFGS,
        "DONCHIAN": DON_CFGS, "KC_SQUEEZE": KC_CFGS,
    }
    for sname, sresults in all_strategy_results.items():
        best = sresults[0] if sresults else None
        if best and best["profit_factor"] >= viable_threshold and best["trades"] >= 15:
            # Find the config entry for the best result
            best_id = best["config_id"]
            base_cfg = None
            for c in cfg_lists[sname]:
                if c["id"] == best_id:
                    base_cfg = c
                    break
            if base_cfg:
                sltp_results[sname] = _run_sltp_opt(
                    sname, gen_fns[sname], base_cfg["params"],
                    all_data, extra_ind, regime_cache)
        else:
            pf_str = "%.2f" % best["profit_factor"] if best else "N/A"
            n_str = str(best["trades"]) if best else "0"
            print("  Skipping %s (best PF=%s n=%s)" % (sname, pf_str, n_str), flush=True)

    # ---- Phase 6: Combined Portfolio ----
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 6 -- Combined Portfolio", flush=True)
    print("=" * 70, flush=True)
    # Pick best viable config per strategy
    viable = {}
    for sname, sresults in all_strategy_results.items():
        for r in sresults:
            if r["profit_factor"] >= 1.15 and r["trades"] >= 15:
                viable[sname] = r
                break
    if viable:
        # Merge signals from all viable strategies
        combined_sigs = {}
        for sname, vr in viable.items():
            vid = vr["config_id"]
            best_cfg = None
            for c in cfg_lists[sname]:
                if c["id"] == vid:
                    best_cfg = c
                    break
            if best_cfg is None:
                continue
            print("  Viable: %s -> %s (PF=%.2f, PnL=%+.1f%%)"
                  % (sname, vid, vr["profit_factor"], vr["pnl_pct"]), flush=True)
            for sym, D in all_data.items():
                ss = gen_fns[sname](D, extra_ind[sym], regime_cache[sym], best_cfg["params"])
                if sym not in combined_sigs:
                    combined_sigs[sym] = {}
                for bar_i, sig in ss.items():
                    if bar_i not in combined_sigs[sym]:
                        combined_sigs[sym][bar_i] = sig
        total_csigs = sum(len(v) for v in combined_sigs.values())
        print("  Combined signals: %d" % total_csigs, flush=True)
        # Use average SL/TP from viable configs
        comb_res = run_sim(combined_sigs, all_data, extra_ind,
                           {"type": "atr", "mult": 2.5},
                           {"type": "fixed_r", "r": 3.5},
                           risk_pt=0.008, cap_frac=0.60, max_pos=5)
        comb_res.pop("completed", None)
        print("  Combined: n=%d PF=%.3f PnL=%+.1f%% DD=%.1f%% WR=%d%%"
              % (comb_res["trades"], comb_res["profit_factor"],
                 comb_res["pnl_pct"], comb_res["max_dd_pct"], comb_res["wr_pct"]),
              flush=True)
        for rg, v in sorted(comb_res.get("per_regime", {}).items()):
            print("    %s: n=%d WR=%d%% PF=%.2f" % (rg, v["n"], v["wr"], v["pf"]),
                  flush=True)
    else:
        comb_res = {"trades": 0, "note": "no viable strategies"}
        print("  No viable strategies for combination.", flush=True)

    for sname, sresults in all_strategy_results.items():
        if sname not in viable:
            best = sresults[0] if sresults else {"profit_factor": 0, "trades": 0}
            print("  Excluded: %s -> best PF=%.2f n=%d"
                  % (sname, best.get("profit_factor", 0), best.get("trades", 0)),
                  flush=True)

    # ---- Phase 7: Walk-Forward ----
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 7 -- Walk-Forward Validation (67/33) -- ALL top-3 per viable", flush=True)
    print("=" * 70, flush=True)
    wf_results = []
    for sname, sresults in all_strategy_results.items():
        top3_viable = [r for r in sresults[:5]
                       if r["profit_factor"] >= 1.05 and r["trades"] >= 12]
        for vr in top3_viable[:3]:
            vid = vr["config_id"]
            best_cfg = None
            for c in cfg_lists[sname]:
                if c["id"] == vid:
                    best_cfg = c
                    break
            if best_cfg:
                wfr = walk_forward(sname, gen_fns[sname], best_cfg,
                                   all_data, extra_ind, regime_cache)
                wf_results.append(wfr)

    if not wf_results:
        print("  No configs qualified for walk-forward.", flush=True)

    # ---- Phase 8: Final Report ----
    print("\n" + "=" * 70, flush=True)
    print("  FINAL REPORT -- Trend-Following Research v7", flush=True)
    print("=" * 70, flush=True)

    for sname in ["BBS", "EMA_MOM", "DONCHIAN", "KC_SQUEEZE"]:
        sresults = all_strategy_results[sname]
        best_pf = sresults[0]["profit_factor"] if sresults else 0
        tag = "VIABLE" if best_pf >= 1.15 else ("MARGINAL" if best_pf >= 1.05 else "REJECTED")
        print("\n  -- %s [%s] --" % (sname, tag), flush=True)
        for i, r in enumerate(sresults[:5]):
            print("    %d. %-30s n=%3d PF=%.3f PnL=%+6.1f%% WR=%d%% DD=%.1f%% MFE=%.1f%% sigs=%d"
                  % (i+1, r["config_id"][:22] + " " + r["desc"][:8],
                     r["trades"], r["profit_factor"], r["pnl_pct"],
                     r["wr_pct"], r["max_dd_pct"], r["mfe_cap"],
                     r["total_sigs"]), flush=True)

    if sltp_results:
        print("\n  -- SL/TP Optimization Results --", flush=True)
        for sname, combos in sltp_results.items():
            print("    %s: best=%s PF=%.3f" % (sname, combos[0]["config_id"],
                  combos[0]["profit_factor"]), flush=True)

    print("\n  -- COMBINED PORTFOLIO --", flush=True)
    if comb_res.get("trades", 0) > 0:
        print("    Strategies: %s" % list(viable.keys()), flush=True)
        print("    n=%d PF=%.3f PnL=%+.1f%% WR=%d%% DD=%.1f%%"
              % (comb_res["trades"], comb_res["profit_factor"],
                 comb_res["pnl_pct"], comb_res["wr_pct"],
                 comb_res["max_dd_pct"]), flush=True)
    else:
        print("    No viable combination.", flush=True)

    print("\n  -- WALK-FORWARD --", flush=True)
    if wf_results:
        for wfr in wf_results:
            print("    %s %-25s train=%+.1f%% test=%+.1f%% PF_test=%.2f n_test=%d [%s]"
                  % (wfr["name"], wfr["id"], wfr["train_pnl"], wfr["test_pnl"],
                     wfr["test_pf"], wfr["test_n"], wfr["tag"]), flush=True)
    else:
        print("    No configs tested.", flush=True)

    passed_wf = [w for w in wf_results if w["passed"]]
    if passed_wf:
        print("\n  ** WALK-FORWARD PASSED: **", flush=True)
        for w in passed_wf:
            print("    %s / %s  test_PF=%.2f test_PnL=%+.1f%%"
                  % (w["name"], w["id"], w["test_pf"], w["test_pnl"]), flush=True)
    else:
        print("\n  ** NO configs passed walk-forward. **", flush=True)

    # ---- Save results ----
    out = {
        "strategy_results": {k: v for k, v in all_strategy_results.items()},
        "sltp_optimization": sltp_results,
        "combined": comb_res,
        "walk_forward": wf_results,
        "metadata": {"timestamp": ts, "symbols": len(all_data),
                      "runtime_sec": round(_time.time() - t0, 1)},
    }
    out_path = RESULTS_DIR / ("backtest_v7_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n  Results saved:", out_path, flush=True)
    print("\nTotal runtime: %d min %d sec" % divmod(int(_time.time() - t0), 60), flush=True)


if __name__ == "__main__":
    main()
