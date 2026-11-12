"""
multi_agent/backtests/backtest_forensic.py  — Phase 1: Trade Forensics Audit

Deep trade-quality audit for every strategy:
  1. Rejection waterfall (per filter stage)
  2. MFE / MAE analysis (max favorable/adverse excursion in R-multiples)
  3. Monetization analysis (% reaching +0.5R, +1R, +1.5R, +2R, +3R)
  4. Counterfactual exit comparison (fixed R, partial+trail, structure)
  5. Regime interaction audit (trades & PnL by regime)
  6. Symbol concentration (top/worst contributors)
  7. Stop placement quality (stop hit vs later reaching target)

Phase 2: Portfolio rebuild with evidence-based redesign.
"""
from __future__ import annotations
import json, logging, time as _time, math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.shared.regime_detector import (
    SoftRegimeDetector, SoftRegimeState, Regime,
    precompute_1h_indicators, precompute_4h_indicators,
    _ema, _sma, _stdev, _atr, _adx, DEFAULT_SOFT_CFG,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake, _adx as bbs_adx,
)
from multi_agent.agents.density_agent import (
    compute_volume_profile, compute_rolling_vwap as density_vwap,
    detect_density_signal, _rsi as density_rsi, _atr as density_atr,
    _adx as density_adx,
)
from multi_agent.agents.mean_reversion_agent import (
    compute_mr_indicators, detect_mr_signal,
    _rsi as mr_rsi, _atr as mr_atr, _sma as mr_sma, _stdev as mr_stdev,
)
from multi_agent.shared.fvg_core import (
    detect_fvgs_v2, _select_active_fvgs,
    composite_signal_score, evaluate_entry_reaction,
)
import multi_agent.config as cfg

logger = logging.getLogger("forensic")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

BACKTEST_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

START_BALANCE = 1000.0
TAKER_FEE = 0.00055
SLIPPAGE = 0.00025
MAX_LEVERAGE = 25


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    safe = symbol.replace("/", "_").replace(":", "_")
    for ext in (".parquet", ".csv"):
        fpath = CACHE_DIR / f"{safe}_{tf}_90d{ext}"
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath) if ext == ".parquet" else pd.read_csv(fpath)
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


# ═══════════════════════════════════════════════════════════════════════════════
#  MFE / MAE tracker — for every trade, track bar-by-bar excursion
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mfe_mae(highs, lows, closes, entry_bar, direction, entry_price,
                    sl, risk, max_bars=500):
    """
    Scan forward from entry_bar to compute MFE/MAE in R-multiples.
    Also returns: what R-level was reached at each bar, whether various R
    targets were hit, and what the exit would have been under different schemes.
    """
    n = len(closes)
    end = min(entry_bar + max_bars, n)
    if risk <= 0:
        return None

    mfe_price = entry_price  # best favorable price
    mae_price = entry_price  # worst adverse price
    mfe_r = 0.0
    mae_r = 0.0

    # Track R-levels reached
    reached = {0.5: False, 1.0: False, 1.5: False, 2.0: False, 3.0: False, 4.0: False}
    reached_bar = {0.5: -1, 1.0: -1, 1.5: -1, 2.0: -1, 3.0: -1, 4.0: -1}

    # Counterfactual exits: what would exit at each R give?
    cf_exits = {}  # {R: exit_price}

    sl_hit_bar = -1
    best_r_before_sl = 0.0

    for j in range(entry_bar + 1, end):
        h = highs[j]
        l = lows[j]

        if direction == "LONG":
            fav = h - entry_price
            adv = entry_price - l
            cur_fav_r = fav / risk
            cur_adv_r = adv / risk

            if h > mfe_price:
                mfe_price = h
                mfe_r = (h - entry_price) / risk
            if l < mae_price:
                mae_price = l
                mae_r = (entry_price - l) / risk

            # Check R targets
            for rt in reached:
                if not reached[rt]:
                    target = entry_price + rt * risk
                    if h >= target:
                        reached[rt] = True
                        reached_bar[rt] = j
                        if rt not in cf_exits:
                            cf_exits[rt] = target

            # SL hit
            if sl_hit_bar < 0 and l <= sl:
                sl_hit_bar = j
                best_r_before_sl = mfe_r

        else:  # SHORT
            fav = entry_price - l
            adv = h - entry_price
            cur_fav_r = fav / risk
            cur_adv_r = adv / risk

            if l < mfe_price:
                mfe_price = l
                mfe_r = (entry_price - l) / risk
            if h > mae_price:
                mae_price = h
                mae_r = (h - entry_price) / risk

            for rt in reached:
                if not reached[rt]:
                    target = entry_price - rt * risk
                    if l <= target:
                        reached[rt] = True
                        reached_bar[rt] = j
                        if rt not in cf_exits:
                            cf_exits[rt] = target

            if sl_hit_bar < 0 and h >= sl:
                sl_hit_bar = j
                best_r_before_sl = mfe_r

    return {
        "mfe_r": round(mfe_r, 3),
        "mae_r": round(mae_r, 3),
        "mfe_price": mfe_price,
        "mae_price": mae_price,
        "mfe_pct": round((mfe_price - entry_price) / entry_price * 100 if direction == "LONG"
                         else (entry_price - mfe_price) / entry_price * 100, 4),
        "mae_pct": round((entry_price - mae_price) / entry_price * 100 if direction == "LONG"
                         else (mae_price - entry_price) / entry_price * 100, 4),
        "reached": reached,
        "reached_bar": reached_bar,
        "cf_exits": cf_exits,
        "sl_hit_bar": sl_hit_bar,
        "best_r_before_sl": round(best_r_before_sl, 3),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  BBS forensic — instrumented rejection waterfall
# ═══════════════════════════════════════════════════════════════════════════════

def bbs_rejection_waterfall(ind, bar_idx, min_squeeze_bars=6,
                             adx_threshold=20.0, vol_mult=1.2):
    """Return (signal_or_None, rejection_reason_or_None)."""
    n = len(ind["closes"])
    i = bar_idx
    if i < 2 or i >= n:
        return None, "insufficient_data"

    for key in ("ema21", "ema50", "ema200", "adx", "momentum"):
        if np.isnan(ind[key][i]):
            return None, "nan_indicators"

    # Must not be in squeeze
    if ind["squeeze_on"][i]:
        return None, "still_in_squeeze"

    # Count squeeze bars
    squeeze_count = 0
    for j in range(i - 1, -1, -1):
        if ind["squeeze_on"][j]:
            squeeze_count += 1
        else:
            break

    if squeeze_count < min_squeeze_bars:
        return None, "squeeze_too_short"

    # Must be first release bar
    if not ind["squeeze_on"][i - 1]:
        return None, "not_first_release"

    mom = ind["momentum"][i]
    mom_prev = ind["momentum"][i - 1]
    if np.isnan(mom) or np.isnan(mom_prev):
        return None, "nan_momentum"

    if ind["adx"][i] < adx_threshold:
        return None, "adx_too_low"

    vol = ind["volumes"][i]
    vol_avg = ind["vol_sma20"][i]
    if np.isnan(vol_avg) or vol_avg <= 0 or vol < vol_mult * vol_avg:
        return None, "volume_too_low"

    close = ind["closes"][i]
    ema21 = ind["ema21"][i]
    ema50 = ind["ema50"][i]
    ema200 = ind["ema200"][i]

    # LONG
    if mom > 0 and mom > mom_prev:
        if close > ema21 > ema50 > ema200:
            return {"direction": "LONG", "entry_price": close, "momentum": float(mom),
                    "adx": float(ind["adx"][i]), "squeeze_bars": squeeze_count,
                    "volume_ratio": float(vol / max(vol_avg, 1e-9)), "bar_idx": i}, None
        return None, "ema_stack_misaligned"

    # SHORT
    if mom < 0 and mom < mom_prev:
        if close < ema21 < ema50 < ema200:
            return {"direction": "SHORT", "entry_price": close, "momentum": float(mom),
                    "adx": float(ind["adx"][i]), "squeeze_bars": squeeze_count,
                    "volume_ratio": float(vol / max(vol_avg, 1e-9)), "bar_idx": i}, None
        return None, "ema_stack_misaligned"

    return None, "momentum_direction"


# ═══════════════════════════════════════════════════════════════════════════════
#  MR forensic — instrumented rejection waterfall
# ═══════════════════════════════════════════════════════════════════════════════

def mr_rejection_waterfall(ind, bar_idx, consec_losses):
    """Return (signal_or_None, rejection_reason_or_None)."""
    n = len(ind["closes"])
    i = bar_idx
    if i < 30 or i >= n:
        return None, "insufficient_data"

    close = ind["closes"][i]
    bb_upper = ind["bb_upper"][i]
    bb_lower = ind["bb_lower"][i]
    bb_mid = ind["bb_mid"][i]
    rsi_val = ind["rsi"][i]
    adx_val = ind["adx"][i]
    atr_val = ind["atr"][i]
    vwap_val = ind["vwap"][i]

    for v in (bb_upper, bb_lower, bb_mid, rsi_val, adx_val, atr_val, vwap_val):
        if np.isnan(v):
            return None, "nan_indicators"

    if adx_val > cfg.MR_ADX_MAX:
        return None, "adx_too_high"

    # Not outside BB at all
    if close >= bb_lower and close <= bb_upper:
        return None, "price_inside_bb"

    # LONG setup
    if close < bb_lower:
        if rsi_val > cfg.MR_RSI_OVERSOLD:
            return None, "rsi_not_extreme"

        vwap_dev = (vwap_val - close) / vwap_val * 100.0 if vwap_val > 0 else 0
        if vwap_dev < cfg.MR_VWAP_MIN_DEVIATION_PCT:
            return None, "vwap_dev_insufficient"

        if consec_losses.get("LONG", 0) >= cfg.MR_MAX_CONSECUTIVE_LOSSES_SAME_DIR:
            return None, "anti_revenge"

        if i >= 5:
            recent_vol = np.mean(ind["volumes"][i - 5:i])
            if ind["volumes"][i] > recent_vol * 3.0:
                return None, "volume_surge"

        sl = close - atr_val * cfg.MR_SL_ATR_MULT
        tp = bb_mid
        risk = close - sl
        reward = tp - close
        if risk <= 0:
            return None, "invalid_risk"
        if reward / risk < cfg.MR_MIN_RR:
            return None, "rr_too_low"

        return {"direction": "LONG", "entry_price": float(close), "sl": float(sl),
                "tp": float(tp), "risk": float(risk), "bar_idx": i}, None

    # SHORT
    if close > bb_upper:
        if rsi_val < cfg.MR_RSI_OVERBOUGHT:
            return None, "rsi_not_extreme"

        vwap_dev = (close - vwap_val) / vwap_val * 100.0 if vwap_val > 0 else 0
        if vwap_dev < cfg.MR_VWAP_MIN_DEVIATION_PCT:
            return None, "vwap_dev_insufficient"

        if consec_losses.get("SHORT", 0) >= cfg.MR_MAX_CONSECUTIVE_LOSSES_SAME_DIR:
            return None, "anti_revenge"

        if i >= 5:
            recent_vol = np.mean(ind["volumes"][i - 5:i])
            if ind["volumes"][i] > recent_vol * 3.0:
                return None, "volume_surge"

        sl = close + atr_val * cfg.MR_SL_ATR_MULT
        tp = bb_mid
        risk = sl - close
        reward = close - tp
        if risk <= 0:
            return None, "invalid_risk"
        if reward / risk < cfg.MR_MIN_RR:
            return None, "rr_too_low"

        return {"direction": "SHORT", "entry_price": float(close), "sl": float(sl),
                "tp": float(tp), "risk": float(risk), "bar_idx": i}, None

    return None, "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Density forensic — instrumented rejection waterfall
# ═══════════════════════════════════════════════════════════════════════════════

def density_rejection_waterfall(closes, highs, lows, volumes, bar_idx,
                                 vp, vwap, rsi, adx, atr_arr):
    """Return (signal_or_None, rejection_reason_or_None)."""
    i = bar_idx
    n = len(closes)
    if i < 30 or i >= n:
        return None, "insufficient_data"

    close = closes[i]
    poc = vp.get("poc", 0)
    vah = vp.get("vah", 0)
    val_low = vp.get("val", 0)
    vwap_val = vwap[i] if not np.isnan(vwap[i]) else 0
    rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
    adx_val = adx[i] if not np.isnan(adx[i]) else 30
    atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 0

    if poc <= 0 or vah <= 0 or val_low <= 0 or vwap_val <= 0 or atr_val <= 0:
        return None, "invalid_vp_data"

    if adx_val > cfg.DENSITY_ADX_MAX:
        return None, "adx_too_high"

    vol_avg = np.mean(volumes[max(0, i - 20):i]) if i >= 20 else np.mean(volumes[:max(i, 1)])
    if vol_avg <= 0:
        return None, "zero_volume"

    # Not at extremes of value area
    at_val = close <= val_low * 1.002
    at_vah = close >= vah * 0.998

    if not at_val and not at_vah:
        return None, "price_inside_value_area"

    if at_val:
        if rsi_val > cfg.DENSITY_RSI_OVERSOLD:
            return None, "rsi_not_extreme"
        vwap_dev = (vwap_val - close) / vwap_val * 100.0 if vwap_val > 0 else 0
        if vwap_dev < cfg.DENSITY_VWAP_DEVIATION_PCT:
            return None, "vwap_dev_insufficient"
        if i >= 3:
            recent_move = (closes[i - 3] - close) / closes[i - 3] * 100.0
            if recent_move > 2.0:
                return None, "impulse_breakdown"
        recent_low = float(np.min(lows[max(0, i - 10):i + 1]))
        sl = recent_low - atr_val * cfg.DENSITY_SL_ATR_MULT
        tp1 = vwap_val
        risk = close - sl
        if risk <= 0:
            return None, "invalid_risk"
        reward = tp1 - close
        rr = reward / risk
        if rr < cfg.DENSITY_MIN_RR:
            return None, "rr_too_low"
        return {"direction": "LONG", "entry_price": float(close), "sl": float(sl),
                "tp1": float(tp1), "tp2": float(poc), "risk": float(risk),
                "rr": float(rr), "bar_idx": i}, None

    if at_vah:
        if rsi_val < cfg.DENSITY_RSI_OVERBOUGHT:
            return None, "rsi_not_extreme"
        vwap_dev = (close - vwap_val) / vwap_val * 100.0 if vwap_val > 0 else 0
        if vwap_dev < cfg.DENSITY_VWAP_DEVIATION_PCT:
            return None, "vwap_dev_insufficient"
        if i >= 3:
            recent_move = (close - closes[i - 3]) / closes[i - 3] * 100.0
            if recent_move > 2.0:
                return None, "impulse_breakdown"
        recent_high = float(np.max(highs[max(0, i - 10):i + 1]))
        sl = recent_high + atr_val * cfg.DENSITY_SL_ATR_MULT
        tp1 = vwap_val
        risk = sl - close
        if risk <= 0:
            return None, "invalid_risk"
        reward = close - tp1
        rr = reward / risk
        if rr < cfg.DENSITY_MIN_RR:
            return None, "rr_too_low"
        return {"direction": "SHORT", "entry_price": float(close), "sl": float(sl),
                "tp1": float(tp1), "tp2": float(poc), "risk": float(risk),
                "rr": float(rr), "bar_idx": i}, None

    return None, "unknown"


# ═══════════════════════════════════════════════════════════════════════════════
#  Counterfactual exit simulator
# ═══════════════════════════════════════════════════════════════════════════════

def simulate_counterfactual_exits(highs, lows, closes, entry_bar, direction,
                                   entry_price, sl, risk, atr_arr,
                                   vwap_arr=None, bb_mid_arr=None,
                                   max_bars=500):
    """
    For a single trade, simulate many exit schemes and return results.
    """
    n = len(closes)
    end = min(entry_bar + max_bars, n)
    results = {}

    if risk <= 0:
        return results

    # Fixed-R exits
    for r_target in [1.0, 1.5, 2.0, 3.0]:
        if direction == "LONG":
            tp = entry_price + r_target * risk
        else:
            tp = entry_price - r_target * risk

        exit_price = None
        exit_bar = -1
        for j in range(entry_bar + 1, end):
            # Check SL first
            if direction == "LONG":
                if lows[j] <= sl:
                    exit_price = sl
                    exit_bar = j
                    break
                if highs[j] >= tp:
                    exit_price = tp
                    exit_bar = j
                    break
            else:
                if highs[j] >= sl:
                    exit_price = sl
                    exit_bar = j
                    break
                if lows[j] <= tp:
                    exit_price = tp
                    exit_bar = j
                    break

        if exit_price is None:
            exit_price = closes[end - 1] if end > entry_bar + 1 else entry_price
            exit_bar = end - 1

        if direction == "LONG":
            pnl_r = (exit_price - entry_price) / risk
        else:
            pnl_r = (entry_price - exit_price) / risk
        results[f"fixed_{r_target}R"] = {
            "exit_price": exit_price, "pnl_r": round(pnl_r, 3),
            "bars_held": exit_bar - entry_bar, "win": pnl_r > 0}

    # Partial at 1R + trail rest with 1.5 ATR
    for partial_r in [1.0, 1.5, 2.0]:
        partial_frac = 0.5
        partial_done = False
        trail_sl = sl
        pnl_parts = 0.0
        remaining = 1.0
        exit_bar_p = -1

        for j in range(entry_bar + 1, end):
            atr_j = atr_arr[j] if j < len(atr_arr) and not np.isnan(atr_arr[j]) else risk
            if direction == "LONG":
                # Check trail SL
                if lows[j] <= trail_sl:
                    pnl_parts += (trail_sl - entry_price) / risk * remaining
                    exit_bar_p = j
                    break
                # Check partial
                tp_partial = entry_price + partial_r * risk
                if not partial_done and highs[j] >= tp_partial:
                    pnl_parts += partial_r * partial_frac
                    remaining -= partial_frac
                    partial_done = True
                # Update trailing SL
                if partial_done:
                    new_trail = highs[j] - 1.5 * atr_j
                    trail_sl = max(trail_sl, new_trail)
            else:
                if highs[j] >= trail_sl:
                    pnl_parts += (entry_price - trail_sl) / risk * remaining
                    exit_bar_p = j
                    break
                tp_partial = entry_price - partial_r * risk
                if not partial_done and lows[j] <= tp_partial:
                    pnl_parts += partial_r * partial_frac
                    remaining -= partial_frac
                    partial_done = True
                if partial_done:
                    new_trail = lows[j] + 1.5 * atr_j
                    trail_sl = min(trail_sl, new_trail) if trail_sl > 0 else new_trail

        if exit_bar_p < 0:
            if direction == "LONG":
                pnl_parts += (closes[end - 1] - entry_price) / risk * remaining
            else:
                pnl_parts += (entry_price - closes[end - 1]) / risk * remaining
            exit_bar_p = end - 1

        results[f"partial_{partial_r}R_trail"] = {
            "pnl_r": round(pnl_parts, 3),
            "bars_held": exit_bar_p - entry_bar,
            "win": pnl_parts > 0, "partial_taken": partial_done}

    # Chandelier trailing (no fixed TP)
    trail_sl_ce = sl
    exit_bar_ce = -1
    for j in range(entry_bar + 1, end):
        atr_j = atr_arr[j] if j < len(atr_arr) and not np.isnan(atr_arr[j]) else risk
        if direction == "LONG":
            if lows[j] <= trail_sl_ce:
                exit_bar_ce = j
                break
            pnl_now = (closes[j] - entry_price) / risk
            if pnl_now >= 1.0:
                hh = float(np.max(highs[max(entry_bar, j - 22 + 1):j + 1]))
                new_ce = hh - 3.0 * atr_j
                trail_sl_ce = max(trail_sl_ce, new_ce)
        else:
            if highs[j] >= trail_sl_ce:
                exit_bar_ce = j
                break
            pnl_now = (entry_price - closes[j]) / risk
            if pnl_now >= 1.0:
                ll = float(np.min(lows[max(entry_bar, j - 22 + 1):j + 1]))
                new_ce = ll + 3.0 * atr_j
                trail_sl_ce = min(trail_sl_ce, new_ce) if trail_sl_ce > 0 else new_ce

    ce_exit = trail_sl_ce if exit_bar_ce >= 0 else closes[end - 1]
    if exit_bar_ce < 0:
        exit_bar_ce = end - 1
    if direction == "LONG":
        pnl_ce = (ce_exit - entry_price) / risk
    else:
        pnl_ce = (entry_price - ce_exit) / risk
    results["chandelier_trail"] = {
        "pnl_r": round(pnl_ce, 3), "bars_held": exit_bar_ce - entry_bar, "win": pnl_ce > 0}

    # VWAP exit (mean reversion target)
    if vwap_arr is not None:
        exit_bar_v = -1
        for j in range(entry_bar + 1, end):
            if direction == "LONG":
                if lows[j] <= sl:
                    exit_bar_v = j
                    break
                if not np.isnan(vwap_arr[j]) and highs[j] >= vwap_arr[j]:
                    exit_bar_v = j
                    break
            else:
                if highs[j] >= sl:
                    exit_bar_v = j
                    break
                if not np.isnan(vwap_arr[j]) and lows[j] <= vwap_arr[j]:
                    exit_bar_v = j
                    break

        if exit_bar_v < 0:
            v_exit = closes[end - 1]
            exit_bar_v = end - 1
        else:
            if direction == "LONG":
                v_exit = vwap_arr[exit_bar_v] if highs[exit_bar_v] >= vwap_arr[exit_bar_v] else sl
            else:
                v_exit = vwap_arr[exit_bar_v] if lows[exit_bar_v] <= vwap_arr[exit_bar_v] else sl

        pnl_v = ((v_exit - entry_price) / risk) if direction == "LONG" else ((entry_price - v_exit) / risk)
        results["vwap_exit"] = {
            "pnl_r": round(pnl_v, 3), "bars_held": exit_bar_v - entry_bar, "win": pnl_v > 0}

    # BB midline exit
    if bb_mid_arr is not None:
        exit_bar_bb = -1
        for j in range(entry_bar + 1, end):
            if direction == "LONG":
                if lows[j] <= sl:
                    exit_bar_bb = j
                    break
                if not np.isnan(bb_mid_arr[j]) and highs[j] >= bb_mid_arr[j]:
                    exit_bar_bb = j
                    break
            else:
                if highs[j] >= sl:
                    exit_bar_bb = j
                    break
                if not np.isnan(bb_mid_arr[j]) and lows[j] <= bb_mid_arr[j]:
                    exit_bar_bb = j
                    break

        if exit_bar_bb < 0:
            bb_exit = closes[end - 1]
            exit_bar_bb = end - 1
        else:
            if direction == "LONG":
                bb_exit = bb_mid_arr[exit_bar_bb] if highs[exit_bar_bb] >= bb_mid_arr[exit_bar_bb] else sl
            else:
                bb_exit = bb_mid_arr[exit_bar_bb] if lows[exit_bar_bb] <= bb_mid_arr[exit_bar_bb] else sl

        pnl_bb = ((bb_exit - entry_price) / risk) if direction == "LONG" else ((entry_price - bb_exit) / risk)
        results["bb_mid_exit"] = {
            "pnl_r": round(pnl_bb, 3), "bars_held": exit_bar_bb - entry_bar, "win": pnl_bb > 0}

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  FVG forensic — instrumented rejection waterfall
# ═══════════════════════════════════════════════════════════════════════════════

def fvg_rejection_waterfall(df_15m, i, closes, highs, lows, volumes, timestamps_15m,
                             closes_1h, timestamps_1h, ema200_1h,
                             closes_4h, timestamps_4h, ema200_4h,
                             adx_15m):
    """
    FVG signal detection with rejection tracking.
    Returns (signal_or_None, rejection_reason_or_None).
    """
    if i < 250:
        return None, "insufficient_data"

    ts_i = int(timestamps_15m[i])
    htf_idx = int(np.searchsorted(timestamps_1h, ts_i, side="right")) - 1
    if htf_idx < 200 or htf_idx >= len(ema200_1h) or np.isnan(ema200_1h[htf_idx]):
        return None, "htf_data_missing"

    htf_close = closes_1h[htf_idx]
    htf_ema = ema200_1h[htf_idx]
    htf_bullish = htf_close > htf_ema
    htf_bearish = htf_close < htf_ema

    # 4h confirmation
    if ema200_4h is not None and timestamps_4h is not None:
        j4 = int(np.searchsorted(timestamps_4h, ts_i, side="right")) - 1
        if 24 <= j4 < len(ema200_4h) and not np.isnan(ema200_4h[j4]):
            if htf_bullish and closes_4h[j4] < ema200_4h[j4] * 0.995:
                return None, "4h_contradicts_1h"
            if htf_bearish and closes_4h[j4] > ema200_4h[j4] * 1.005:
                return None, "4h_contradicts_1h"

    # ADX filter
    adx_val = adx_15m[i]
    if not np.isnan(adx_val) and adx_val < 15.0:
        return None, "adx_too_low"

    # Volume filter
    if i >= 10:
        vol_avg = np.mean(volumes[i - 10:i])
        if vol_avg > 0 and volumes[i] < 0.6 * vol_avg:
            return None, "volume_too_low"

    # Detect FVGs
    ws = max(0, i - 384)
    df_w = df_15m.iloc[ws:i + 1].copy().reset_index(drop=True)
    if len(df_w) < 50:
        return None, "insufficient_window"

    all_fvgs = detect_fvgs_v2(df_w, lookback=min(384, len(df_w)),
                               percentile_threshold=cfg.FVG_PERCENTILE_THRESHOLD,
                               percentile_lookback=min(500, len(df_w)))
    active = _select_active_fvgs(all_fvgs)

    if not active:
        return None, "no_active_fvg"

    cv = closes[i]
    best_signal = None
    last_reject = "no_matching_fvg"

    for fvg in active:
        if fvg.type == "bullish" and not htf_bullish:
            last_reject = "fvg_htf_direction_mismatch"
            continue
        if fvg.type == "bearish" and not htf_bearish:
            last_reject = "fvg_htf_direction_mismatch"
            continue
        if fvg.quality_score < 40.0:
            last_reject = "fvg_quality_too_low"
            continue

        gap = max(0.0, fvg.high - fvg.low)
        if gap <= 0:
            last_reject = "fvg_zero_gap"
            continue
        lo_band = fvg.low + gap * 0.20
        hi_band = fvg.low + gap * 0.80
        if not (lo_band <= cv <= hi_band):
            last_reject = "price_outside_entry_zone"
            continue

        reaction_ok, reaction_score = evaluate_entry_reaction(df_w.tail(6), fvg)
        fvg.reaction_score = float(reaction_score)
        comp = composite_signal_score(fvg, reaction_score)

        # Fib bonus
        swing_lo = float(np.min(lows[max(0, i - 120):i + 1]))
        swing_hi = float(np.max(highs[max(0, i - 120):i + 1]))
        fib_range = swing_hi - swing_lo
        fib_bonus = 0.0
        if fib_range > 0:
            f382 = swing_hi - fib_range * 0.618
            f618 = swing_hi - fib_range * 0.382
            if f382 <= cv <= f618:
                fib_bonus = 5.0

        effective_comp = comp + fib_bonus

        if effective_comp < 50.0:
            last_reject = "composite_too_low"
            continue
        if effective_comp > 90.0:
            last_reject = "composite_too_high"
            continue

        direction = "LONG" if fvg.type == "bullish" else "SHORT"
        if direction == "LONG":
            sl = float(np.min(lows[max(0, i - 60):i + 1])) * 0.9985
        else:
            sl = float(np.max(highs[max(0, i - 60):i + 1])) * 1.0015
        risk = abs(cv - sl)
        if risk <= 0 or risk / cv > 0.05:
            last_reject = "invalid_risk_distance"
            continue

        # TP: structural — 2R target (simplified for forensic; actual uses structure levels)
        if direction == "LONG":
            tp = cv + risk * 2.0
        else:
            tp = cv - risk * 2.0

        best_signal = {
            "direction": direction,
            "entry_price": float(cv),
            "sl": float(sl),
            "tp": float(tp),
            "risk": float(risk),
            "composite": float(effective_comp),
            "quality": float(fvg.quality_score),
            "fib_bonus": fib_bonus,
            "bar_idx": i,
        }
        break

    if best_signal is not None:
        return best_signal, None
    return None, last_reject


# ═══════════════════════════════════════════════════════════════════════════════
#  Core forensic engine — runs one strategy at a time
# ═══════════════════════════════════════════════════════════════════════════════

def run_forensic_audit(agent_name: str, use_regime: bool = True) -> dict:
    """
    Run a comprehensive forensic audit for one agent.
    Returns detailed diagnostics.
    """
    t0 = _time.time()
    logger.info(f"=== FORENSIC AUDIT: {agent_name.upper()} (regime={'ON' if use_regime else 'OFF'}) ===")

    rejection_counts = defaultdict(int)
    total_bars_scanned = 0
    raw_candidates = 0
    regime_rejected = 0
    htf_rejected = 0

    trades = []  # detailed trade records
    per_symbol = defaultdict(lambda: {"trades": 0, "pnl_r": 0.0, "wins": 0})
    per_regime = defaultdict(lambda: {"trades": 0, "pnl_r": 0.0, "wins": 0})

    soft_cfg = {
        "confirm_bars": 5, "min_hold_bars": 12,
        "smooth_alpha": 0.30, "min_confidence": 1.0,
        "transition_margin": 0.5, "undefined_threshold": 0.5,
    }

    for sym_idx, symbol in enumerate(BACKTEST_SYMBOLS):
        df_15m = _load(symbol, "15m")
        df_1h = _load(symbol, "1h")
        df_4h = _load(symbol, "4h")
        if df_15m is None or df_1h is None:
            continue
        if len(df_15m) < 250 or len(df_1h) < 210:
            continue

        cl15 = df_15m["close"].astype(float).to_numpy()
        hi15 = df_15m["high"].astype(float).to_numpy()
        lo15 = df_15m["low"].astype(float).to_numpy()
        vo15 = df_15m["volume"].astype(float).to_numpy()
        ts15 = df_15m["timestamp"].astype(int).to_numpy()
        ts1h = df_1h["timestamp"].astype(int).to_numpy()
        n15 = len(cl15)

        ind_1h = precompute_1h_indicators(df_1h)
        ema200_1h = ind_1h["ema200_1h"]
        cl1h = ind_1h["closes_1h_arr"]

        ind_4h = None
        if df_4h is not None and len(df_4h) > 50:
            ind_4h = precompute_4h_indicators(df_4h)

        # Agent-specific indicators
        squeeze_ind = None
        mr_ind = None
        d_rsi = d_adx = d_atr = d_vwap = None
        atr_15m = _atr(hi15, lo15, cl15, 14)
        vwap_15m = density_vwap(hi15, lo15, cl15, vo15, 96)
        bb_basis_15m = mr_sma(cl15, 20)
        bb_std_15m = mr_stdev(cl15, 20)
        bb_mid_15m = bb_basis_15m

        if agent_name == "bbs":
            squeeze_ind = compute_squeeze_indicators(df_15m)
        elif agent_name == "mr":
            mr_ind = compute_mr_indicators(df_15m)
        elif agent_name == "density":
            d_rsi = density_rsi(cl15, cfg.DENSITY_RSI_PERIOD if hasattr(cfg, 'DENSITY_RSI_PERIOD') else 14)
            d_adx = density_adx(hi15, lo15, cl15, 14)
            d_atr = density_atr(hi15, lo15, cl15, 14)
            d_vwap = density_vwap(hi15, lo15, cl15, vo15, cfg.DENSITY_VP_LOOKBACK_BARS)
        elif agent_name == "fvg":
            adx_15m_arr = _adx(hi15, lo15, cl15, 14)
            cl4h = df_4h["close"].astype(float).to_numpy() if df_4h is not None and len(df_4h) > 50 else None
            ts4h = df_4h["timestamp"].astype(int).to_numpy() if df_4h is not None and len(df_4h) > 50 else None
            ema200_4h_arr = ind_4h.get("ema200_4h") if ind_4h else None

        detector = SoftRegimeDetector(soft_cfg)
        prev_regime = Regime.UNDEFINED
        position = None
        mr_consec = {"LONG": 0, "SHORT": 0}

        i_start = 250
        i_end = n15

        for i in range(i_start, i_end):
            total_bars_scanned += 1
            close_i = cl15[i]
            high_i = hi15[i]
            low_i = lo15[i]

            # Regime
            htf_idx = int(np.searchsorted(ts1h, int(ts15[i]), side="right")) - 1
            if htf_idx < 210:
                continue

            rs = detector.update(symbol, df_1h, htf_idx, df_4h,
                                 **(ind_1h), **(ind_4h or {}))

            # ── Manage open position ──
            if position is not None:
                pos = position
                d = pos["direction"]
                entry = pos["entry"]
                sl = pos["sl"]
                r = pos["risk"]
                tp = pos.get("tp", 0)

                hit_sl = (d == "LONG" and low_i <= sl) or (d == "SHORT" and high_i >= sl)
                hit_tp = False
                if tp > 0:
                    hit_tp = (d == "LONG" and high_i >= tp) or (d == "SHORT" and low_i <= tp)

                # BBS: chandelier trailing
                if pos["agent"] == "bbs" and squeeze_ind is not None:
                    pnl_r_now = ((close_i - entry) / r) if d == "LONG" else ((entry - close_i) / r)
                    if pnl_r_now >= 1.0:
                        new_ce = chandelier_exit_sl(
                            hi15, lo15, squeeze_ind["ce_atr"], d, i,
                            ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                        if d == "LONG":
                            pos["trailing_sl"] = max(pos.get("trailing_sl", sl), new_ce)
                        else:
                            pos["trailing_sl"] = min(pos.get("trailing_sl", sl), new_ce) if pos.get("trailing_sl", 0) > 0 else new_ce

                    eff_sl = pos.get("trailing_sl", sl)
                    if d == "LONG":
                        eff_sl = max(sl, eff_sl)
                        hit_sl = low_i <= eff_sl
                        if hit_sl:
                            sl = eff_sl
                    else:
                        if eff_sl > 0:
                            eff_sl = min(sl, eff_sl)
                        hit_sl = high_i >= eff_sl if eff_sl > 0 else high_i >= sl
                        if hit_sl:
                            sl = eff_sl if eff_sl > 0 else sl

                    if pnl_r_now >= 3.0:
                        hit_tp = True
                        tp = close_i

                if hit_sl or hit_tp:
                    exit_p = tp if hit_tp else sl
                    if d == "LONG":
                        pnl_r = (exit_p - entry) / r
                    else:
                        pnl_r = (entry - exit_p) / r
                    bars_held = i - pos["entry_bar"]

                    # MFE/MAE
                    excursion = compute_mfe_mae(
                        hi15, lo15, cl15, pos["entry_bar"], d, entry, pos["orig_sl"], r)

                    # Counterfactual exits
                    cf = simulate_counterfactual_exits(
                        hi15, lo15, cl15, pos["entry_bar"], d, entry, pos["orig_sl"], r,
                        atr_15m, vwap_arr=vwap_15m, bb_mid_arr=bb_mid_15m)

                    trade_rec = {
                        "symbol": symbol,
                        "agent": agent_name,
                        "direction": d,
                        "entry_price": entry,
                        "exit_price": exit_p,
                        "sl": pos["orig_sl"],
                        "tp": pos.get("tp", 0),
                        "risk": r,
                        "pnl_r": round(pnl_r, 3),
                        "pnl_pct": round((exit_p - entry) / entry * 100 if d == "LONG"
                                         else (entry - exit_p) / entry * 100, 4),
                        "win": pnl_r > 0,
                        "bars_held": bars_held,
                        "regime_at_entry": pos.get("regime", "?"),
                        "entry_bar": pos["entry_bar"],
                        "excursion": excursion,
                        "counterfactual": cf,
                        "exit_type": "tp" if hit_tp else "sl",
                    }
                    trades.append(trade_rec)
                    per_symbol[symbol]["trades"] += 1
                    per_symbol[symbol]["pnl_r"] += pnl_r
                    if pnl_r > 0:
                        per_symbol[symbol]["wins"] += 1
                    per_regime[pos.get("regime", "?")]["trades"] += 1
                    per_regime[pos.get("regime", "?")]["pnl_r"] += pnl_r
                    if pnl_r > 0:
                        per_regime[pos.get("regime", "?")]["wins"] += 1

                    if agent_name == "mr" and pnl_r <= 0:
                        mr_consec[d] = mr_consec.get(d, 0) + 1
                    elif agent_name == "mr" and pnl_r > 0:
                        mr_consec[d] = 0

                    position = None
                    continue

                continue  # still in position

            # ── No position: try signal ──
            signal = None
            reject_reason = None

            if agent_name == "bbs" and squeeze_ind is not None:
                signal, reject_reason = bbs_rejection_waterfall(
                    squeeze_ind, i,
                    min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                    adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                    vol_mult=cfg.BB_SQUEEZE_VOL_MULT)

                if signal is not None:
                    raw_candidates += 1
                    # HTF filter
                    htf_dir = "LONG" if cl1h[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                    if signal["direction"] != htf_dir:
                        signal = None
                        reject_reason = "htf_bias_mismatch"
                        htf_rejected += 1
                    # Head fake check
                    elif check_head_fake(squeeze_ind, i, 2):
                        signal = None
                        reject_reason = "head_fake"

                    if signal is not None:
                        ce_sl = chandelier_exit_sl(
                            hi15, lo15, squeeze_ind["ce_atr"], signal["direction"], i,
                            ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                        risk_val = abs(close_i - ce_sl)
                        if 0 < risk_val / close_i <= 0.05:
                            signal["sl"] = float(ce_sl)
                            signal["risk"] = float(risk_val)
                        else:
                            signal = None
                            reject_reason = "invalid_risk_distance"

            elif agent_name == "mr" and mr_ind is not None:
                signal, reject_reason = mr_rejection_waterfall(
                    mr_ind, i, mr_consec)
                if signal is not None:
                    raw_candidates += 1

            elif agent_name == "density":
                vp_start = max(0, i - cfg.DENSITY_VP_LOOKBACK_BARS)
                vp = compute_volume_profile(hi15, lo15, cl15, vo15,
                                             num_bins=cfg.DENSITY_VP_NUM_BINS,
                                             start_idx=vp_start, end_idx=i + 1)
                signal, reject_reason = density_rejection_waterfall(
                    cl15, hi15, lo15, vo15, i, vp, d_vwap, d_rsi, d_adx, d_atr)
                if signal is not None:
                    raw_candidates += 1

            elif agent_name == "fvg" and adx_15m_arr is not None:
                signal, reject_reason = fvg_rejection_waterfall(
                    df_15m, i, cl15, hi15, lo15, vo15, ts15,
                    cl1h, ts1h, ema200_1h, cl4h, ts4h, ema200_4h_arr,
                    adx_15m_arr)
                if signal is not None:
                    raw_candidates += 1

            if reject_reason:
                rejection_counts[reject_reason] += 1

            if signal is None:
                continue

            # Regime gating
            if use_regime:
                if not rs.trade_ok.get(agent_name, False):
                    rejection_counts["regime_blocked"] += 1
                    regime_rejected += 1
                    continue
                size_m = rs.size_mult.get(agent_name, 0.0)
                if size_m < 0.05:
                    rejection_counts["regime_size_too_small"] += 1
                    regime_rejected += 1
                    continue

            # Open position
            risk_val = signal.get("risk", abs(close_i - signal.get("sl", close_i * 0.98)))
            if risk_val <= 0:
                continue

            position = {
                "agent": agent_name,
                "direction": signal["direction"],
                "entry": signal["entry_price"],
                "sl": signal.get("sl", 0),
                "orig_sl": signal.get("sl", 0),
                "tp": signal.get("tp", signal.get("tp1", 0)),
                "risk": risk_val,
                "entry_bar": i,
                "regime": rs.regime.value,
                "trailing_sl": signal.get("sl", 0),
            }

        if sym_idx % 5 == 4:
            logger.info(f"  {symbol} done ({sym_idx+1}/{len(BACKTEST_SYMBOLS)}) "
                        f"trades={len(trades)} t={_time.time()-t0:.0f}s")

    elapsed = _time.time() - t0
    logger.info(f"  Forensic audit complete: {len(trades)} trades, {elapsed:.0f}s")

    # ── Compile diagnostics ──
    return _compile_diagnostics(agent_name, trades, rejection_counts,
                                 total_bars_scanned, raw_candidates,
                                 regime_rejected, htf_rejected,
                                 per_symbol, per_regime, use_regime)


def _compile_diagnostics(agent_name, trades, rejection_counts,
                          total_bars_scanned, raw_candidates,
                          regime_rejected, htf_rejected,
                          per_symbol, per_regime, use_regime):
    """Compile all forensic metrics from trade records."""
    n_trades = len(trades)
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]

    # Basic stats
    wr = len(wins) / n_trades * 100 if n_trades > 0 else 0
    pnl_r_total = sum(t["pnl_r"] for t in trades)
    avg_win_r = np.mean([t["pnl_r"] for t in wins]) if wins else 0
    avg_loss_r = np.mean([t["pnl_r"] for t in losses]) if losses else 0
    avg_hold = np.mean([t["bars_held"] for t in trades]) if trades else 0

    # MFE/MAE
    mfe_rs = [t["excursion"]["mfe_r"] for t in trades if t["excursion"]]
    mae_rs = [t["excursion"]["mae_r"] for t in trades if t["excursion"]]
    mfe_pcts = [t["excursion"]["mfe_pct"] for t in trades if t["excursion"]]
    mae_pcts = [t["excursion"]["mae_pct"] for t in trades if t["excursion"]]

    # Monetization
    monetization = {}
    for rt in [0.5, 1.0, 1.5, 2.0, 3.0]:
        reached = sum(1 for t in trades if t["excursion"] and t["excursion"]["reached"][rt])
        monetization[f"reached_{rt}R"] = reached
        monetization[f"reached_{rt}R_pct"] = round(reached / n_trades * 100, 1) if n_trades > 0 else 0

    # Trades that reached +1R but ended as losers
    reached_1r_but_lost = sum(1 for t in trades if t["excursion"] and
                               t["excursion"]["reached"][1.0] and t["pnl_r"] <= 0)
    reached_2r_but_gave_back = sum(1 for t in trades if t["excursion"] and
                                    t["excursion"]["reached"][2.0] and t["pnl_r"] < 1.0)

    # Stop hit before any meaningful favorable move
    stop_before_move = sum(1 for t in trades if t["excursion"] and
                            t["excursion"]["sl_hit_bar"] >= 0 and
                            t["excursion"]["best_r_before_sl"] < 0.3)

    # Counterfactual exit comparison
    cf_summary = defaultdict(lambda: {"trades": 0, "wins": 0, "pnl_r": 0.0, "bars": []})
    for t in trades:
        for scheme, result in t.get("counterfactual", {}).items():
            cf_summary[scheme]["trades"] += 1
            if result["win"]:
                cf_summary[scheme]["wins"] += 1
            cf_summary[scheme]["pnl_r"] += result["pnl_r"]
            cf_summary[scheme]["bars"].append(result["bars_held"])

    cf_report = {}
    for scheme, data in cf_summary.items():
        n = data["trades"]
        cf_report[scheme] = {
            "trades": n,
            "wr_pct": round(data["wins"] / n * 100, 1) if n > 0 else 0,
            "total_r": round(data["pnl_r"], 2),
            "avg_r": round(data["pnl_r"] / n, 3) if n > 0 else 0,
            "avg_bars": round(np.mean(data["bars"]), 1) if data["bars"] else 0,
        }

    # Regime breakdown
    regime_report = {}
    for reg, data in per_regime.items():
        n = data["trades"]
        regime_report[reg] = {
            "trades": n,
            "wr_pct": round(data["wins"] / n * 100, 1) if n > 0 else 0,
            "total_r": round(data["pnl_r"], 2),
            "avg_r": round(data["pnl_r"] / n, 3) if n > 0 else 0,
        }

    # Symbol breakdown — top 5 winners and losers
    sym_list = [(s, d["trades"], d["pnl_r"], d["wins"]) for s, d in per_symbol.items()]
    sym_list.sort(key=lambda x: x[2], reverse=True)

    # Stop quality analysis
    sl_tight = sum(1 for t in trades if t["excursion"] and
                    t["excursion"]["mae_r"] > 0 and t["excursion"]["mae_r"] < 0.5 and
                    not t["win"] and t["excursion"]["mfe_r"] > 1.0)  # stop hit, but MFE was >1R
    sl_wide = sum(1 for t in trades if t["excursion"] and
                   t["excursion"]["mae_r"] > 2.0 and not t["win"])  # SL very far, still lost

    return {
        "agent": agent_name,
        "use_regime": use_regime,
        "total_bars_scanned": total_bars_scanned,
        "raw_candidates": raw_candidates,
        "trades": n_trades,
        "wins": len(wins),
        "losses": len(losses),
        "wr_pct": round(wr, 1),
        "total_pnl_r": round(pnl_r_total, 2),
        "avg_win_r": round(avg_win_r, 3),
        "avg_loss_r": round(avg_loss_r, 3),
        "avg_hold_bars": round(avg_hold, 1),
        "rejection_waterfall": dict(rejection_counts),
        "regime_rejected": regime_rejected,
        "htf_rejected": htf_rejected,
        "mfe_mae": {
            "avg_mfe_r": round(np.mean(mfe_rs), 3) if mfe_rs else 0,
            "median_mfe_r": round(np.median(mfe_rs), 3) if mfe_rs else 0,
            "avg_mae_r": round(np.mean(mae_rs), 3) if mae_rs else 0,
            "median_mae_r": round(np.median(mae_rs), 3) if mae_rs else 0,
            "avg_mfe_pct": round(np.mean(mfe_pcts), 3) if mfe_pcts else 0,
            "avg_mae_pct": round(np.mean(mae_pcts), 3) if mae_pcts else 0,
            "mfe_r_buckets": {
                "<0.5R": sum(1 for x in mfe_rs if x < 0.5),
                "0.5-1R": sum(1 for x in mfe_rs if 0.5 <= x < 1.0),
                "1-2R": sum(1 for x in mfe_rs if 1.0 <= x < 2.0),
                "2-3R": sum(1 for x in mfe_rs if 2.0 <= x < 3.0),
                ">3R": sum(1 for x in mfe_rs if x >= 3.0),
            },
            "mae_r_buckets": {
                "<0.5R": sum(1 for x in mae_rs if x < 0.5),
                "0.5-1R": sum(1 for x in mae_rs if 0.5 <= x < 1.0),
                "1-2R": sum(1 for x in mae_rs if 1.0 <= x < 2.0),
                ">2R": sum(1 for x in mae_rs if x >= 2.0),
            },
        },
        "monetization": {
            **monetization,
            "reached_1R_but_lost": reached_1r_but_lost,
            "reached_1R_but_lost_pct": round(reached_1r_but_lost / n_trades * 100, 1) if n_trades else 0,
            "reached_2R_but_gave_back": reached_2r_but_gave_back,
            "stop_before_any_move": stop_before_move,
            "stop_before_any_move_pct": round(stop_before_move / n_trades * 100, 1) if n_trades else 0,
        },
        "counterfactual_exits": cf_report,
        "regime_breakdown": regime_report,
        "symbol_top5": sym_list[:5],
        "symbol_bottom5": sym_list[-5:] if len(sym_list) >= 5 else sym_list,
        "stop_quality": {
            "tight_stop_lost_good_trade": sl_tight,
            "wide_stop_still_lost": sl_wide,
            "tight_pct": round(sl_tight / n_trades * 100, 1) if n_trades else 0,
            "wide_pct": round(sl_wide / n_trades * 100, 1) if n_trades else 0,
        },
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: Portfolio engine with evidence-based redesign
# ═══════════════════════════════════════════════════════════════════════════════

def run_portfolio_v3(params: dict, agents_enabled: list = None,
                     start_frac: float = 0.0, end_frac: float = 1.0,
                     skip_fvg: bool = True) -> dict:
    """
    Evidence-based portfolio backtest.
    Agents: fvg, bbs, mr (density excluded by default).
    """
    P = params
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    all_trades = []

    if agents_enabled is None:
        agents_enabled = ["bbs", "mr"]
        if not skip_fvg:
            agents_enabled.insert(0, "fvg")

    per_agent = {a: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []}
                 for a in agents_enabled}

    soft_cfg = {
        "confirm_bars": P.get("regime_confirm_bars", 5),
        "min_hold_bars": P.get("regime_min_hold", 12),
        "smooth_alpha": P.get("regime_smooth_alpha", 0.30),
        "min_confidence": P.get("regime_min_confidence", 1.0),
        "transition_margin": P.get("regime_transition_margin", 0.5),
        "undefined_threshold": P.get("regime_undefined_threshold", 0.5),
    }

    # BBS soft gating by regime
    bbs_regime_mult = {
        "SQUEEZE": P.get("bbs_mult_squeeze", 1.0),
        "TRANSITION": P.get("bbs_mult_transition", 0.85),
        "TREND": P.get("bbs_mult_trend", 0.65),
        "RANGE": P.get("bbs_mult_range", 0.0),
        "UNDEFINED": P.get("bbs_mult_undefined", 0.0),
    }

    # MR gating — WIDENED (forensic: TRANSITION best +5.17R, RANGE secondary, TREND marginal)
    mr_regime_mult = {
        "TRANSITION": P.get("mr_mult_transition", 1.0),
        "RANGE": P.get("mr_mult_range", 0.85),
        "TREND": P.get("mr_mult_trend", 0.3),
        "SQUEEZE": P.get("mr_mult_squeeze", 0.0),
        "UNDEFINED": 0.0,
    }

    risk_map = {
        "fvg": P.get("fvg_risk", 0.011),
        "bbs": P.get("bbs_risk", 0.009),
        "mr": P.get("mr_risk", 0.005),
    }
    frac_map = {
        "fvg": P.get("fvg_frac", 0.50),
        "bbs": P.get("bbs_frac", 0.35),
        "mr": P.get("mr_frac", 0.15),
    }

    for symbol in BACKTEST_SYMBOLS:
        df_15m = _load(symbol, "15m")
        df_1h = _load(symbol, "1h")
        df_4h = _load(symbol, "4h")
        if df_15m is None or df_1h is None:
            continue
        if len(df_15m) < 250 or len(df_1h) < 210:
            continue

        n_total = len(df_15m)
        i_start = max(250, int(n_total * start_frac))
        i_end = int(n_total * end_frac)

        cl15 = df_15m["close"].astype(float).to_numpy()
        hi15 = df_15m["high"].astype(float).to_numpy()
        lo15 = df_15m["low"].astype(float).to_numpy()
        vo15 = df_15m["volume"].astype(float).to_numpy()
        ts15 = df_15m["timestamp"].astype(int).to_numpy()
        ts1h = df_1h["timestamp"].astype(int).to_numpy()

        ind_1h = precompute_1h_indicators(df_1h)
        ema200_1h = ind_1h["ema200_1h"]
        cl1h = ind_1h["closes_1h_arr"]

        ind_4h = None
        ts4h = cl4h = ema200_4h = None
        if df_4h is not None and len(df_4h) > 50:
            ind_4h = precompute_4h_indicators(df_4h)
            ema200_4h = ind_4h["ema200_4h"]
            ts4h = df_4h["timestamp"].astype(int).to_numpy()
            cl4h = df_4h["close"].astype(float).to_numpy()

        squeeze_ind = compute_squeeze_indicators(df_15m) if "bbs" in agents_enabled else None
        mr_ind = compute_mr_indicators(df_15m) if "mr" in agents_enabled else None
        atr_15m = _atr(hi15, lo15, cl15, 14)
        adx_15m = _adx(hi15, lo15, cl15, 14) if "fvg" in agents_enabled else None

        detector = SoftRegimeDetector(soft_cfg)
        position = None
        mr_consec = {"LONG": 0, "SHORT": 0}

        for i in range(i_start, i_end):
            close_i = cl15[i]
            high_i = hi15[i]
            low_i = lo15[i]

            htf_idx = int(np.searchsorted(ts1h, int(ts15[i]), side="right")) - 1
            if htf_idx < 210:
                continue

            rs = detector.update(symbol, df_1h, htf_idx, df_4h,
                                 **(ind_1h), **(ind_4h or {}))
            regime_str = rs.regime.value

            # Position management
            if position is not None:
                pos = position
                d = pos["direction"]
                entry = pos["entry"]
                sl = pos["sl"]
                r = pos["risk"]
                q = pos["qty"]
                agent = pos["agent"]

                hit_sl = (d == "LONG" and low_i <= sl) or (d == "SHORT" and high_i >= sl)
                hit_tp = False
                tp = pos.get("tp", 0)

                # BBS trailing
                if agent == "bbs" and squeeze_ind is not None:
                    pnl_r_now = ((close_i - entry) / r) if d == "LONG" else ((entry - close_i) / r)

                    # Partial TP at specified R
                    partial_r = P.get("bbs_partial_tp_r", 1.5)
                    if partial_r > 0 and not pos.get("partial_taken") and pnl_r_now >= partial_r:
                        frac = P.get("bbs_partial_frac", 0.5)
                        pq = q * frac
                        pnl_partial = ((close_i - entry) if d == "LONG" else (entry - close_i)) * pq
                        fee = close_i * pq * (TAKER_FEE + SLIPPAGE)
                        balance += pnl_partial - fee
                        pos["qty"] -= pq
                        pos["partial_taken"] = True
                        q = pos["qty"]

                    if pnl_r_now >= 1.0:
                        new_ce = chandelier_exit_sl(
                            hi15, lo15, squeeze_ind["ce_atr"], d, i,
                            ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                        if d == "LONG":
                            pos["trailing_sl"] = max(pos.get("trailing_sl", sl), new_ce)
                        else:
                            pos["trailing_sl"] = min(pos.get("trailing_sl", sl), new_ce) if pos.get("trailing_sl", 0) > 0 else new_ce

                    eff_sl = pos.get("trailing_sl", sl)
                    if d == "LONG":
                        eff_sl = max(sl, eff_sl)
                        hit_sl = low_i <= eff_sl
                        if hit_sl: sl = eff_sl
                    else:
                        if eff_sl > 0: eff_sl = min(sl, eff_sl)
                        hit_sl = high_i >= eff_sl if eff_sl > 0 else high_i >= sl
                        if hit_sl: sl = eff_sl if eff_sl > 0 else sl

                    # Fixed R exit (evidence: 3R = +49R vs chandelier 4R = +32.64R)
                    bbs_max_r = P.get("bbs_fixed_tp_r", 3.0)
                    if pnl_r_now >= bbs_max_r:
                        hit_tp = True
                        tp = close_i

                # MR / FVG: simple SL/TP
                elif tp > 0:
                    hit_tp = (d == "LONG" and high_i >= tp) or (d == "SHORT" and low_i <= tp)

                if hit_sl or hit_tp:
                    exit_p = tp if hit_tp else sl
                    pnl = ((exit_p - entry) if d == "LONG" else (entry - exit_p)) * q
                    fee = exit_p * q * (TAKER_FEE + SLIPPAGE)
                    net = pnl - fee
                    balance += net
                    all_trades.append(net)
                    pa = per_agent[agent]
                    pa["trades"].append(net)
                    pa["pnl"] += net
                    if net > 0: pa["wins"] += 1
                    else: pa["losses"] += 1
                    if agent == "mr":
                        if net <= 0: mr_consec[d] = mr_consec.get(d, 0) + 1
                        else: mr_consec[d] = 0
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    position = None
                    continue

                continue

            # Try signals
            for agent in agents_enabled:
                # Get regime multiplier
                if agent == "bbs":
                    size_m = bbs_regime_mult.get(regime_str, 0.0)
                elif agent == "mr":
                    size_m = mr_regime_mult.get(regime_str, 0.0)
                else:  # fvg
                    size_m = rs.size_mult.get("fvg", 0.0)
                    if not rs.trade_ok.get("fvg", False):
                        size_m = 0.0

                if size_m < 0.05:
                    continue

                signal = None

                if agent == "bbs" and squeeze_ind is not None:
                    sig = detect_squeeze_signal(
                        squeeze_ind, bar_idx=i,
                        min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                        adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                        vol_mult=cfg.BB_SQUEEZE_VOL_MULT)
                    if sig:
                        htf_dir = "LONG" if cl1h[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                        if sig["direction"] == htf_dir and not check_head_fake(squeeze_ind, i, 2):
                            ce_sl = chandelier_exit_sl(
                                hi15, lo15, squeeze_ind["ce_atr"], sig["direction"], i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            risk_v = abs(close_i - ce_sl)
                            if 0 < risk_v / close_i <= 0.05:
                                signal = {
                                    "direction": sig["direction"],
                                    "entry_price": float(close_i),
                                    "sl": float(ce_sl), "risk": float(risk_v),
                                    "tp": 0, "bar_idx": i,
                                }

                elif agent == "mr" and mr_ind is not None:
                    sig = detect_mr_signal(mr_ind, i, mr_consec)
                    if sig:
                        signal = sig

                elif agent == "fvg" and adx_15m is not None:
                    # Simplified FVG using backtest_v2 logic
                    if i >= 250:
                        from multi_agent.backtests.backtest_v2 import _fvg_signal
                        fvg_params = {
                            "fvg_adx_min": P.get("fvg_adx_min", 15.0),
                            "fvg_volume_mult": P.get("fvg_volume_mult", 0.6),
                            "fvg_quality_min": P.get("fvg_quality_min", 40.0),
                            "fvg_entry_lo": P.get("fvg_entry_lo", 0.20),
                            "fvg_entry_hi": P.get("fvg_entry_hi", 0.80),
                            "fvg_composite_lo": P.get("fvg_composite_lo", 55.0),
                            "fvg_composite_hi": P.get("fvg_composite_hi", 90.0),
                            "fvg_fib_mandatory": P.get("fvg_fib_mandatory", False),
                            "fvg_fib_bonus": P.get("fvg_fib_bonus", 5.0),
                        }
                        sig = _fvg_signal(
                            df_15m, i, cl15, hi15, lo15, vo15, ts15,
                            cl1h, ts1h, ema200_1h, cl4h, ts4h, ema200_4h,
                            adx_15m, fvg_params)
                        if sig:
                            # FVG: add TP at 2R (backtest_v2 returns no TP)
                            fvg_risk = sig.get("risk", 0)
                            if fvg_risk > 0 and "tp" not in sig:
                                fvg_tp_r = P.get("fvg_tp_r", 2.0)
                                if sig["direction"] == "LONG":
                                    sig["tp"] = sig["entry_price"] + fvg_risk * fvg_tp_r
                                else:
                                    sig["tp"] = sig["entry_price"] - fvg_risk * fvg_tp_r
                            signal = sig

                if signal is None:
                    continue

                # Sizing
                agent_bal = balance * frac_map.get(agent, 0.25)
                base_risk = risk_map.get(agent, 0.008)
                effective_risk = base_risk * size_m
                risk_usd = agent_bal * effective_risk
                risk_dist = signal.get("risk", abs(signal["entry_price"] - signal.get("sl", 0)))
                if risk_dist <= 0:
                    continue
                qty = risk_usd / risk_dist
                margin = signal["entry_price"] * qty / MAX_LEVERAGE
                if margin > agent_bal * 0.3:
                    qty = agent_bal * 0.3 * MAX_LEVERAGE / signal["entry_price"]
                if qty * signal["entry_price"] < 5.0:
                    continue

                fee = signal["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
                balance -= fee

                position = {
                    "agent": agent,
                    "direction": signal["direction"],
                    "entry": signal["entry_price"],
                    "sl": signal.get("sl", 0),
                    "tp": signal.get("tp", 0),
                    "risk": risk_dist,
                    "qty": qty,
                    "bar_idx": i,
                    "trailing_sl": signal.get("sl", 0),
                    "partial_taken": False,
                }
                break

    # Stats
    total_trades = sum(pa["wins"] + pa["losses"] for pa in per_agent.values())
    total_wins = sum(pa["wins"] for pa in per_agent.values())
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100.0
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    win_t = [t for t in all_trades if t > 0]
    loss_t = [t for t in all_trades if t <= 0]
    gp = sum(win_t)
    gl = abs(sum(loss_t))
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0

    per_agent_out = {}
    for a, pa in per_agent.items():
        t = pa["wins"] + pa["losses"]
        w_t = [x for x in pa["trades"] if x > 0]
        l_t = [x for x in pa["trades"] if x <= 0]
        gp_a = sum(w_t)
        gl_a = abs(sum(l_t))
        per_agent_out[a] = {
            "trades": t, "wins": pa["wins"], "losses": pa["losses"],
            "wr_pct": round(pa["wins"] / t * 100, 1) if t > 0 else 0,
            "pnl_usd": round(pa["pnl"], 2),
            "pf": round(gp_a / gl_a, 2) if gl_a > 0 else 0,
        }

    return {
        "trades": total_trades, "wins": total_wins,
        "losses": total_trades - total_wins,
        "wr_pct": round(wr, 1), "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "final_balance": round(balance, 2),
        "per_agent": per_agent_out,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — Phase 1 then Phase 2
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 100)
    logger.info("TRADE FORENSICS AUDIT + PORTFOLIO REBUILD — Phase 1 & 2")
    logger.info("=" * 100)

    all_results = {}

    # ═══ PHASE 1: Forensic audit per strategy ═══
    logger.info("\n### PHASE 1: FORENSIC AUDIT ###")

    for agent in ["fvg", "bbs", "mr", "density"]:
        for use_reg in [True, False]:
            key = f"{agent}_{'regime' if use_reg else 'noreg'}"
            logger.info(f"\n--- Auditing {key} ---")
            result = run_forensic_audit(agent, use_regime=use_reg)
            all_results[key] = result

            # Print summary
            r = result
            print(f"\n{'='*80}")
            print(f"  {key.upper()}")
            print(f"{'='*80}")
            print(f"  Bars scanned: {r['total_bars_scanned']:,}")
            print(f"  Raw candidates: {r['raw_candidates']:,}")
            print(f"  Trades taken: {r['trades']}  WR: {r['wr_pct']}%")
            print(f"  Total PnL (R): {r['total_pnl_r']}")
            print(f"  Avg win: {r['avg_win_r']}R  Avg loss: {r['avg_loss_r']}R")
            print(f"  Avg hold: {r['avg_hold_bars']} bars")
            if r['regime_rejected']:
                print(f"  Regime rejected: {r['regime_rejected']}")
            print(f"\n  Rejection waterfall:")
            for reason, count in sorted(r['rejection_waterfall'].items(), key=lambda x: -x[1])[:15]:
                print(f"    {reason:<30} {count:>8}")
            print(f"\n  MFE/MAE:")
            m = r['mfe_mae']
            print(f"    Avg MFE: {m['avg_mfe_r']}R ({m['avg_mfe_pct']}%)")
            print(f"    Med MFE: {m['median_mfe_r']}R")
            print(f"    Avg MAE: {m['avg_mae_r']}R ({m['avg_mae_pct']}%)")
            print(f"    Med MAE: {m['median_mae_r']}R")
            print(f"    MFE buckets: {m['mfe_r_buckets']}")
            print(f"    MAE buckets: {m['mae_r_buckets']}")
            print(f"\n  Monetization:")
            mon = r['monetization']
            for rt in [0.5, 1.0, 1.5, 2.0, 3.0]:
                print(f"    Reached {rt}R: {mon[f'reached_{rt}R_pct']}% ({mon[f'reached_{rt}R']} trades)")
            print(f"    Reached 1R but lost: {mon['reached_1R_but_lost']} ({mon['reached_1R_but_lost_pct']}%)")
            print(f"    Reached 2R but gave back: {mon['reached_2R_but_gave_back']}")
            print(f"    Stop before any move: {mon['stop_before_any_move']} ({mon['stop_before_any_move_pct']}%)")
            print(f"\n  Counterfactual exits (vs actual):")
            for scheme, data in sorted(r['counterfactual_exits'].items()):
                print(f"    {scheme:<25} WR={data['wr_pct']:>5.1f}%  Total R={data['total_r']:>+7.2f}  "
                      f"Avg R={data['avg_r']:>+6.3f}  Bars={data['avg_bars']:>5.1f}")
            print(f"\n  Stop quality:")
            sq = r['stop_quality']
            print(f"    Too tight (hit SL, MFE>1R): {sq['tight_stop_lost_good_trade']} ({sq['tight_pct']}%)")
            print(f"    Too wide (MAE>2R, still lost): {sq['wide_stop_still_lost']} ({sq['wide_pct']}%)")
            print(f"\n  Regime breakdown:")
            for reg, data in sorted(r['regime_breakdown'].items()):
                print(f"    {reg:<15} T={data['trades']:>4}  WR={data['wr_pct']:>5.1f}%  "
                      f"Total R={data['total_r']:>+7.2f}  Avg R={data['avg_r']:>+6.3f}")
            print(f"\n  Top symbols: {r['symbol_top5'][:3]}")
            print(f"  Worst symbols: {r['symbol_bottom5'][:3]}")

    # ═══ PHASE 2: Portfolio rebuild ═══
    logger.info("\n### PHASE 2: PORTFOLIO REBUILD ###")

    base_params = {
        "regime_confirm_bars": 5, "regime_min_hold": 12,
        "regime_smooth_alpha": 0.30, "regime_transition_margin": 0.5,
        "regime_undefined_threshold": 0.5, "regime_min_confidence": 1.0,
        # BBS regime multipliers — FLIPPED (forensic evidence: TREND best, SQUEEZE worst)
        "bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85,
        "bbs_mult_squeeze": 0.5, "bbs_mult_range": 0.0,
        "bbs_mult_undefined": 0.0,
        # BBS exit — fixed 3R (forensic: +49R vs chandelier +32.64R)
        "bbs_fixed_tp_r": 3.0, "bbs_partial_tp_r": 0, "bbs_partial_frac": 0,
        # MR — WIDENED (forensic: TRANSITION best, RANGE secondary, TREND marginal)
        "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.3,
        "mr_mult_squeeze": 0.0,
        # Risk
        "fvg_risk": 0.011, "bbs_risk": 0.010, "mr_risk": 0.006,
        "fvg_frac": 0.45, "bbs_frac": 0.35, "mr_frac": 0.15,
        # FVG params
        "fvg_adx_min": 15.0, "fvg_volume_mult": 0.6, "fvg_quality_min": 40.0,
        "fvg_entry_lo": 0.20, "fvg_entry_hi": 0.80,
        "fvg_composite_lo": 55.0, "fvg_composite_hi": 90.0,
        "fvg_fib_mandatory": False, "fvg_fib_bonus": 5.0,
        "fvg_tp_r": 2.0,
    }

    portfolio_tests = {}

    # Test 1: FVG only (core strategy baseline)
    logger.info("--- Portfolio: FVG only ---")
    r = run_portfolio_v3(base_params, agents_enabled=["fvg"], skip_fvg=False)
    portfolio_tests["fvg_only"] = r
    logger.info(f"  FVG only: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 2: BBS only
    logger.info("--- Portfolio: BBS only ---")
    r = run_portfolio_v3(base_params, agents_enabled=["bbs"])
    portfolio_tests["bbs_only"] = r
    logger.info(f"  BBS only: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 3: MR only
    logger.info("--- Portfolio: MR only ---")
    r = run_portfolio_v3(base_params, agents_enabled=["mr"])
    portfolio_tests["mr_only"] = r
    logger.info(f"  MR only: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 4: FVG + BBS (main directional combo)
    logger.info("--- Portfolio: FVG + BBS ---")
    r = run_portfolio_v3(base_params, agents_enabled=["fvg", "bbs"], skip_fvg=False)
    portfolio_tests["fvg_bbs"] = r
    logger.info(f"  FVG+BBS: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 5: FVG + MR
    logger.info("--- Portfolio: FVG + MR ---")
    r = run_portfolio_v3(base_params, agents_enabled=["fvg", "mr"], skip_fvg=False)
    portfolio_tests["fvg_mr"] = r
    logger.info(f"  FVG+MR: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 6: BBS + MR
    logger.info("--- Portfolio: BBS + MR ---")
    r = run_portfolio_v3(base_params, agents_enabled=["bbs", "mr"])
    portfolio_tests["bbs_mr"] = r
    logger.info(f"  BBS+MR: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 7: FVG + BBS + MR (full portfolio)
    logger.info("--- Portfolio: FVG + BBS + MR ---")
    r = run_portfolio_v3(base_params, agents_enabled=["fvg", "bbs", "mr"], skip_fvg=False)
    portfolio_tests["fvg_bbs_mr"] = r
    logger.info(f"  FVG+BBS+MR: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Test 4-8: BBS parameter variants based on forensic evidence
    bbs_variants = [
        {"bbs_partial_tp_r": 1.0, "bbs_partial_frac": 0.5, "bbs_mult_transition": 0.90},
        {"bbs_partial_tp_r": 1.5, "bbs_partial_frac": 0.5, "bbs_mult_transition": 0.90, "bbs_mult_trend": 0.75},
        {"bbs_partial_tp_r": 2.0, "bbs_partial_frac": 0.4, "bbs_mult_transition": 0.85},
        {"bbs_partial_tp_r": 0, "bbs_mult_transition": 0.90, "bbs_mult_trend": 0.70},  # pure trail
        {"bbs_partial_tp_r": 1.5, "bbs_partial_frac": 0.5, "bbs_risk": 0.010,
         "bbs_mult_transition": 0.90, "bbs_mult_trend": 0.75},
    ]
    for vi, bv in enumerate(bbs_variants):
        p = {**base_params, **bv}
        logger.info(f"--- Portfolio: BBS+MR variant {vi} ---")
        r = run_portfolio_v3(p, agents_enabled=["bbs", "mr"])
        portfolio_tests[f"bbs_mr_v{vi}"] = r
        logger.info(f"  v{vi}: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # MR variants
    mr_variants = [
        {"mr_risk": 0.004, "mr_mult_range": 1.0, "mr_mult_transition": 0.3},
        {"mr_risk": 0.006, "mr_mult_range": 1.0, "mr_mult_transition": 0.6},
    ]
    for vi, mv in enumerate(mr_variants):
        p = {**base_params, **mv}
        logger.info(f"--- Portfolio: BBS+MR mr_v{vi} ---")
        r = run_portfolio_v3(p, agents_enabled=["bbs", "mr"])
        portfolio_tests[f"bbs_mr_mrv{vi}"] = r
        logger.info(f"  mr_v{vi}: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Allocation variants
    alloc_variants = [
        {"bbs_frac": 0.60, "mr_frac": 0.15, "bbs_risk": 0.010, "mr_risk": 0.005},
        {"bbs_frac": 0.70, "mr_frac": 0.10, "bbs_risk": 0.010, "mr_risk": 0.004},
        {"bbs_frac": 0.50, "mr_frac": 0.20, "bbs_risk": 0.009, "mr_risk": 0.006},
    ]
    for vi, av in enumerate(alloc_variants):
        p = {**base_params, **av}
        logger.info(f"--- Portfolio: BBS+MR alloc_v{vi} ---")
        r = run_portfolio_v3(p, agents_enabled=["bbs", "mr"])
        portfolio_tests[f"bbs_mr_alloc{vi}"] = r
        logger.info(f"  alloc{vi}: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f}")

    # Walk-forward for best
    logger.info("--- Walk-forward validation ---")
    # Find best config
    def _score(r):
        return r["pnl_pct"] - 0.5 * r["max_dd_pct"] + 0.3 * math.log(r["trades"] + 1)

    best_key = max(portfolio_tests, key=lambda k: _score(portfolio_tests[k]))
    logger.info(f"Best config: {best_key}")

    # Walk-forward using full portfolio (FVG + BBS + MR)
    r_train = run_portfolio_v3(base_params, agents_enabled=["fvg", "bbs", "mr"],
                                skip_fvg=False, start_frac=0.0, end_frac=0.67)
    r_test = run_portfolio_v3(base_params, agents_enabled=["fvg", "bbs", "mr"],
                               skip_fvg=False, start_frac=0.67, end_frac=1.0)
    portfolio_tests["wf_train"] = r_train
    portfolio_tests["wf_test"] = r_test
    logger.info(f"  Train: T={r_train['trades']} PnL={r_train['pnl_pct']:+.2f}% DD={r_train['max_dd_pct']:.2f}%")
    logger.info(f"  Test:  T={r_test['trades']} PnL={r_test['pnl_pct']:+.2f}% DD={r_test['max_dd_pct']:.2f}%")

    # ═══ FINAL REPORT ═══
    print("\n" + "=" * 110)
    print("PHASE 2: PORTFOLIO COMPARISON")
    print("=" * 110)
    header = f"{'Config':<25} {'Trades':>6} {'WR%':>6} {'PnL%':>9} {'MaxDD%':>8} {'PF':>6} {'Score':>8}"
    print(header)
    print("-" * 80)

    sorted_tests = sorted(portfolio_tests.items(), key=lambda x: _score(x[1]), reverse=True)
    for key, r in sorted_tests:
        sc = _score(r)
        print(f"{key:<25} {r['trades']:>6} {r['wr_pct']:>5.1f}% {r['pnl_pct']:>+8.2f}% "
              f"{r['max_dd_pct']:>7.2f}% {r['profit_factor']:>5.2f} {sc:>+7.2f}")
        if "per_agent" in r:
            for a, s in r["per_agent"].items():
                if s["trades"] > 0:
                    print(f"   - {a.upper():<10} T={s['trades']:>3} WR={s['wr_pct']:>5.1f}% PnL=${s['pnl_usd']:>+8.2f} PF={s['pf']:>5.2f}")

    # Save
    save_data = {
        "phase1_forensics": {k: {kk: vv for kk, vv in v.items() if kk != "trades_raw"}
                              for k, v in all_results.items()},
        "phase2_portfolio": portfolio_tests,
        "best_config": best_key,
    }
    path = RESULTS_DIR / f"forensic_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")

    return save_data


if __name__ == "__main__":
    main()
