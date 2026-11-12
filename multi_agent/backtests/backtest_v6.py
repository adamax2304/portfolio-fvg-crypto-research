"""
backtest_v6.py  --  Multi-Strategy Research Engine  (2026-04-13)

Strategies:
  BBS     -- Volatility expansion / breakout  (existing, tuned)
  RANGE   -- Mean reversion at BB/RSI extremes (new)
  FAKEOUT -- Liquidity grab / false breakout   (new)
  FVG2    -- FVG zone retest reaction trades   (new)

Architecture:
  Phase 0  Pre-compute indicators + regime   (runs once)
  Phase 1  BBS parameter grid
  Phase 2  Range/MR parameter grid
  Phase 3  Fakeout parameter grid
  Phase 4  FVG v2 parameter grid
  Phase 5  SL/TP deep optimization top-3
  Phase 6  Combined portfolio + regime allocation
  Phase 7  Walk-forward validation
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

logger = logging.getLogger("v6")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLACKLIST = BBS_BLACKLIST_CANDIDATES
SOFT_CFG = {"confirm_bars": 5, "min_hold_bars": 12, "smooth_alpha": 0.30,
            "min_confidence": 1.0, "transition_margin": 0.5, "undefined_threshold": 0.5}

# ===========================================================================
#  INDICATOR FUNCTIONS
# ===========================================================================

def _sma(arr, period):
    out = np.full_like(arr, np.nan)
    cs = np.cumsum(arr)
    out[period-1:] = (cs[period-1:] - np.concatenate(([0], cs[:-period]))) / period
    return out

def _stdev(arr, period):
    out = np.full_like(arr, np.nan)
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


def compute_extra(D: dict) -> dict:
    """Compute extra indicators for new strategies. Called once per symbol."""
    cl = D["cl15"]; hi = D["hi15"]; lo = D["lo15"]; vo = D["vo15"]
    n = len(cl)
    op = D["df15"]["open"].astype(float).to_numpy()
    sma20 = _sma(cl, 20)
    std20 = _stdev(cl, 20)
    bb_upper = sma20 + 2.0 * std20
    bb_lower = sma20 - 2.0 * std20
    bb_mid   = sma20
    bb_pctb  = np.where((bb_upper - bb_lower) > 0, (cl - bb_lower) / (bb_upper - bb_lower), 0.5)
    rsi14    = _rsi(cl, 14)
    vol_ma20 = _sma(vo, 20)
    vol_ratio= np.where(vol_ma20 > 0, vo / vol_ma20, 1.0)
    atr14    = D["atr15"]
    adx14    = D["adx15"]
    return {
        "open": op, "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_mid,
        "bb_pctb": bb_pctb, "rsi": rsi14, "vol_ma": vol_ma20, "vol_ratio": vol_ratio,
        "atr": atr14, "adx": adx14, "n": n,
    }

# ===========================================================================
#  SIGNAL DETECTORS
# ===========================================================================

def gen_bbs_signals(D, extra, regime_arr, params):
    """Generate BBS signals with parameterized thresholds."""
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
        # atr filter
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
        sigs[i] = {"direction": bbs_s["direction"], "entry_price": float(D["cl15"][i]),
                    "default_sl": float(ce_sl), "regime": r, "size_m": sm}
    return sigs


def gen_range_signals(D, extra, regime_arr, params):
    """Mean reversion at BB/RSI extremes."""
    sigs = {}
    bb_lo_th   = params.get("bb_lo", 0.05)   # %B threshold for oversold
    bb_hi_th   = params.get("bb_hi", 0.95)   # %B threshold for overbought
    rsi_lo     = params.get("rsi_lo", 30.0)
    rsi_hi     = params.get("rsi_hi", 70.0)
    adx_max    = params.get("adx_max", 25.0)  # 0 = no filter
    use_vol    = params.get("vol_filter", False)
    vol_max    = params.get("vol_max", 3.0)
    regimes    = params.get("regimes", None)   # None = all regimes
    atr_mult   = params.get("sl_atr_mult", 1.5)
    n = extra["n"]
    bb = extra["bb_pctb"]; rsi = extra["rsi"]; adx = extra["adx"]
    cl = D["cl15"]; atr = extra["atr"]
    for i in range(50, n):
        if np.isnan(bb[i]) or np.isnan(rsi[i]):
            continue
        r = regime_arr[i] if i < len(regime_arr) else "UNDEFINED"
        if regimes is not None:
            sm = regimes.get(r, 0.0)
            if sm < 0.05:
                continue
        else:
            sm = 1.0
        if adx_max > 0 and not np.isnan(adx[i]) and adx[i] > adx_max:
            continue
        if use_vol and extra["vol_ratio"][i] > vol_max:
            continue
        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01
        # LONG: oversold
        if bb[i] <= bb_lo_th and rsi[i] <= rsi_lo:
            sl_v = cl[i] - atr_v * atr_mult
            sigs[i] = {"direction": "LONG", "entry_price": float(cl[i]),
                        "default_sl": float(sl_v), "regime": r, "size_m": sm}
        # SHORT: overbought
        elif bb[i] >= bb_hi_th and rsi[i] >= rsi_hi:
            sl_v = cl[i] + atr_v * atr_mult
            sigs[i] = {"direction": "SHORT", "entry_price": float(cl[i]),
                        "default_sl": float(sl_v), "regime": r, "size_m": sm}
    return sigs


def gen_fakeout_signals(D, extra, regime_arr, params):
    """Liquidity grab / false breakout detection."""
    sigs = {}
    lb       = params.get("lookback", 20)
    wick_min = params.get("wick_ratio", 0.3)
    use_vol  = params.get("vol_filter", True)
    vol_mult = params.get("vol_mult", 1.2)
    regimes  = params.get("regimes", None)
    atr_mult = params.get("sl_atr_mult", 1.0)
    cl = D["cl15"]; hi = D["hi15"]; lo = D["lo15"]; op = extra["open"]
    vo = D["vo15"]; atr = extra["atr"]; n = extra["n"]
    for i in range(lb + 1, n):
        r = regime_arr[i] if i < len(regime_arr) else "UNDEFINED"
        if regimes is not None:
            sm = regimes.get(r, 0.0)
            if sm < 0.05:
                continue
        else:
            sm = 1.0
        rng = hi[i] - lo[i]
        if rng <= 0:
            continue
        roll_hi = np.max(hi[i-lb:i])
        roll_lo = np.min(lo[i-lb:i])
        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01
        # Fake breakout UP -> SHORT
        if hi[i] > roll_hi and cl[i] < roll_hi:
            upper_wick = hi[i] - max(op[i], cl[i])
            if upper_wick / rng >= wick_min:
                if use_vol and extra["vol_ratio"][i] < vol_mult:
                    continue
                sl_v = hi[i] + atr_v * 0.2  # SL just above wick
                sigs[i] = {"direction": "SHORT", "entry_price": float(cl[i]),
                            "default_sl": float(sl_v), "regime": r, "size_m": sm}
                continue
        # Fake breakout DOWN -> LONG
        if lo[i] < roll_lo and cl[i] > roll_lo:
            lower_wick = min(op[i], cl[i]) - lo[i]
            if lower_wick / rng >= wick_min:
                if use_vol and extra["vol_ratio"][i] < vol_mult:
                    continue
                sl_v = lo[i] - atr_v * 0.2  # SL just below wick
                sigs[i] = {"direction": "LONG", "entry_price": float(cl[i]),
                            "default_sl": float(sl_v), "regime": r, "size_m": sm}
    return sigs


def gen_fvg2_signals(D, extra, regime_arr, params):
    """FVG v2: detect FVG zones, trade the retest reaction."""
    sigs = {}
    min_gap_pct = params.get("min_gap_pct", 0.001)
    max_age     = params.get("max_age", 50)
    min_age     = params.get("min_age", 3)
    need_reject = params.get("rejection", True)
    regimes     = params.get("regimes", None)
    atr_mult    = params.get("sl_atr_mult", 1.5)
    cl = D["cl15"]; hi = D["hi15"]; lo = D["lo15"]; op = extra["open"]
    atr = extra["atr"]; n = extra["n"]
    active_fvgs = []
    for i in range(2, n):
        # Detect new FVGs
        if lo[i] > hi[i-2] and cl[i] > 0:
            gap = (lo[i] - hi[i-2]) / cl[i]
            if gap >= min_gap_pct:
                active_fvgs.append({"type": "bull", "top": float(lo[i]),
                    "bot": float(hi[i-2]), "mid": float((lo[i]+hi[i-2])/2), "bar": i})
        if hi[i] < lo[i-2] and cl[i] > 0:
            gap = (lo[i-2] - hi[i]) / cl[i]
            if gap >= min_gap_pct:
                active_fvgs.append({"type": "bear", "top": float(lo[i-2]),
                    "bot": float(hi[i]), "mid": float((lo[i-2]+hi[i])/2), "bar": i})
        # Prune old
        active_fvgs = [f for f in active_fvgs if i - f["bar"] <= max_age]
        if i < 300:
            continue
        r = regime_arr[i] if i < len(regime_arr) else "UNDEFINED"
        if regimes is not None:
            sm = regimes.get(r, 0.0)
            if sm < 0.05:
                continue
        else:
            sm = 1.0
        atr_v = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01
        # Check retests
        for fvg in list(active_fvgs):
            age = i - fvg["bar"]
            if age < min_age:
                continue
            if fvg["type"] == "bull" and lo[i] <= fvg["top"] and cl[i] >= fvg["bot"]:
                if need_reject and not (cl[i] > fvg["mid"] and cl[i] > op[i]):
                    continue
                sl_v = fvg["bot"] - atr_v * 0.3
                sigs[i] = {"direction": "LONG", "entry_price": float(cl[i]),
                            "default_sl": float(sl_v), "regime": r, "size_m": sm}
                active_fvgs.remove(fvg)
                break
            elif fvg["type"] == "bear" and hi[i] >= fvg["bot"] and cl[i] <= fvg["top"]:
                if need_reject and not (cl[i] < fvg["mid"] and cl[i] < op[i]):
                    continue
                sl_v = fvg["top"] + atr_v * 0.3
                sigs[i] = {"direction": "SHORT", "entry_price": float(cl[i]),
                            "default_sl": float(sl_v), "regime": r, "size_m": sm}
                active_fvgs.remove(fvg)
                break
    return sigs


# ===========================================================================
#  SL COMPUTATION (from v5)
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
#  GENERIC SIMULATION ENGINE
# ===========================================================================

def run_sim(signals_by_sym, all_data, extra_ind, sl_cfg, tp_cfg,
            mgt_cfg=None, risk_pt=0.010, cap_frac=0.50, max_pos=4,
            start_frac=0.0, end_frac=1.0):
    """Generic simulation. signals_by_sym = {sym: {bar_idx: signal_dict}}."""
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
                    elif tp_type == "bb_mid":
                        mid = ex.get("bb_mid", np.array([]))[i] if i < ex.get("n", 0) else cl_i
                        if not np.isnan(mid) and pnl_r >= 0.3:
                            if (d == "LONG" and hi_i >= mid) or (d == "SHORT" and lo_i <= mid):
                                hit_tp = True
                                tp_price = min(mid, cl_i) if d == "LONG" else max(mid, cl_i)
                                exit_rsn = "tp_bb_mid"
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
                    completed.append({"sym": sym, "d": d, "exit_r": round(exit_r, 3),
                        "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                        "bars": bars, "won": total_net > 0, "pnl_usd": round(total_net, 4),
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
            open_pos[sym] = {"d": sig["direction"], "entry": sig["entry_price"],
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
    # per-regime breakdown
    regimes_seen = set(t["regime"] for t in completed)
    per_regime = {}
    for rg in regimes_seen:
        rt = [t for t in completed if t["regime"] == rg]
        rw = [t for t in rt if t["won"]]
        rl = [t for t in rt if not t["won"]]
        rgp = sum(t["pnl_usd"] for t in rw)
        rgl = abs(sum(t["pnl_usd"] for t in rl))
        rpf = rgp / rgl if rgl > 0 else (999.0 if rgp > 0 else 0)
        per_regime[rg] = {"n": len(rt), "wr": round(len(rw)/len(rt)*100, 1) if rt else 0,
                          "pf": round(rpf, 2)}
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
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

# ---- BBS configs ----
BBS_CFGS = [
    {"id": "bbs_base", "desc": "current production",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_loose", "desc": "loose filters",
     "params": {"min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_vloose", "desc": "very loose filters",
     "params": {"min_squeeze": 5, "adx_thresh": 12.0, "vol_mult": 0.8,
                "htf_filter": True, "headfake_filter": False,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85, "SQUEEZE": 0.5}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_nohtf", "desc": "no HTF filter",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": False, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_atrfilt", "desc": "ATR>0.5% filter",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True, "atr_min_pct": 0.005,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_allreg", "desc": "all regimes",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85, "RANGE": 0.5, "SQUEEZE": 0.5}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
    {"id": "bbs_atr25", "desc": "ATR2.5 SL",
     "params": {"min_squeeze": 15, "adx_thresh": 25.0, "vol_mult": 1.5,
                "htf_filter": True, "headfake_filter": True,
                "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
     "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
     "risk": 0.010, "frac": 0.75, "max_pos": 4},
]

# ---- Range/MR configs ----
RANGE_CFGS = [
    {"id": "rng_base", "desc": "BB2.0 RSI30/70 ADX<25",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_loose1", "desc": "BB2.0 RSI35/65 ADX<30",
     "params": {"bb_lo": 0.10, "bb_hi": 0.90, "rsi_lo": 35, "rsi_hi": 65, "adx_max": 30},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_loose2", "desc": "RSI40/60 no ADX",
     "params": {"bb_lo": 0.10, "bb_hi": 0.90, "rsi_lo": 40, "rsi_hi": 60, "adx_max": 0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_strict", "desc": "RSI25/75 ADX<20",
     "params": {"bb_lo": 0.02, "bb_hi": 0.98, "rsi_lo": 25, "rsi_hi": 75, "adx_max": 20},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_bbmid", "desc": "base + BB mid TP",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25},
     "sl": {"type": "default"}, "tp": {"type": "bb_mid"},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_2r", "desc": "base + 2R target",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_1r", "desc": "base + 1R target",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.0},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_tightstop", "desc": "ATR1.0 SL",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25},
     "sl": {"type": "atr", "mult": 1.0}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_norsi", "desc": "BB only no RSI",
     "params": {"bb_lo": 0.02, "bb_hi": 0.98, "rsi_lo": 50, "rsi_hi": 50, "adx_max": 25},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_regime", "desc": "RANGE+SQUEEZE only",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25,
                "regimes": {"RANGE": 1.0, "SQUEEZE": 0.8}},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_vloose", "desc": "very loose all",
     "params": {"bb_lo": 0.15, "bb_hi": 0.85, "rsi_lo": 40, "rsi_hi": 60, "adx_max": 0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.0},
     "risk": 0.008, "frac": 0.30, "max_pos": 5},
    {"id": "rng_time20", "desc": "base + time exit 20 bars",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25},
     "sl": {"type": "default"}, "tp": {"type": "time_exit", "max_bars": 20, "min_r": 0.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
    {"id": "rng_volfilt", "desc": "base + vol reject >3x",
     "params": {"bb_lo": 0.05, "bb_hi": 0.95, "rsi_lo": 30, "rsi_hi": 70, "adx_max": 25,
                "vol_filter": True, "vol_max": 3.0},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.008, "frac": 0.30, "max_pos": 3},
]

# ---- Fakeout configs ----
FAKEOUT_CFGS = [
    {"id": "fko_base", "desc": "lb20 wick0.3 vol1.2",
     "params": {"lookback": 20, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_loose", "desc": "lb20 wick0.15 no vol",
     "params": {"lookback": 20, "wick_ratio": 0.15, "vol_filter": False},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_vloose", "desc": "lb10 wick0.1 no vol",
     "params": {"lookback": 10, "wick_ratio": 0.1, "vol_filter": False},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 4},
    {"id": "fko_strict", "desc": "lb20 wick0.5 vol1.5",
     "params": {"lookback": 20, "wick_ratio": 0.5, "vol_filter": True, "vol_mult": 1.5},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_short_lb", "desc": "lb10 wick0.3 vol1.2",
     "params": {"lookback": 10, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_long_lb", "desc": "lb30 wick0.3 vol1.2",
     "params": {"lookback": 30, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_1.5r", "desc": "base + 1.5R",
     "params": {"lookback": 20, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_3r", "desc": "base + 3R",
     "params": {"lookback": 20, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_atr_sl", "desc": "ATR1.0 SL",
     "params": {"lookback": 20, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "atr", "mult": 1.0}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_time30", "desc": "base + time exit 30",
     "params": {"lookback": 20, "wick_ratio": 0.3, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "time_exit", "max_bars": 30, "min_r": 0.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fko_lb15", "desc": "lb15 wick0.25 vol1.2",
     "params": {"lookback": 15, "wick_ratio": 0.25, "vol_filter": True, "vol_mult": 1.2},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
]

# ---- FVG v2 configs ----
FVG2_CFGS = [
    {"id": "fvg2_base", "desc": "gap0.1% age3-50 reject",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_loose", "desc": "gap0.05% age3-100 no reject",
     "params": {"min_gap_pct": 0.0005, "max_age": 100, "min_age": 3, "rejection": False},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_vloose", "desc": "gap0.03% age3-200 no reject",
     "params": {"min_gap_pct": 0.0003, "max_age": 200, "min_age": 2, "rejection": False},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 4},
    {"id": "fvg2_strict", "desc": "gap0.2% age5-30 reject",
     "params": {"min_gap_pct": 0.002, "max_age": 30, "min_age": 5, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_1.5r", "desc": "base + 1.5R",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 1.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_2.5r", "desc": "base + 2.5R",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_3r", "desc": "base + 3R",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 3.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_atr1", "desc": "ATR1.0 SL",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "atr", "mult": 1.0}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_wide_age", "desc": "gap0.1% age3-150 reject",
     "params": {"min_gap_pct": 0.001, "max_age": 150, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_no_reject", "desc": "gap0.1% age3-50 no reject",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": False},
     "sl": {"type": "default"}, "tp": {"type": "fixed_r", "r": 2.0},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_bbmid", "desc": "base + BB mid TP",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "bb_mid"},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
    {"id": "fvg2_time40", "desc": "base + time exit 40",
     "params": {"min_gap_pct": 0.001, "max_age": 50, "min_age": 3, "rejection": True},
     "sl": {"type": "default"}, "tp": {"type": "time_exit", "max_bars": 40, "min_r": 0.5},
     "risk": 0.010, "frac": 0.35, "max_pos": 3},
]


# ===========================================================================
#  PHASE 0  --  PRE-COMPUTE
# ===========================================================================

def phase0(all_data):
    """Pre-compute extra indicators + regime per bar per symbol."""
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
    """Run a strategy phase: for each config generate signals -> simulate."""
    print("\n" + "=" * 70, flush=True)
    print("  %s  (%d configs)" % (name, len(cfgs)), flush=True)
    print("=" * 70, flush=True)
    results = []
    for k, c in enumerate(cfgs):
        # generate signals
        all_sigs = {}
        total_sigs = 0
        for sym, D in all_data.items():
            sigs = gen_fn(D, extra_ind[sym], regime_cache[sym], c["params"])
            all_sigs[sym] = sigs
            total_sigs += len(sigs)
        # simulate
        r = run_sim(all_sigs, all_data, extra_ind,
                    c["sl"], c["tp"], risk_pt=c.get("risk", 0.01),
                    cap_frac=c.get("frac", 0.5), max_pos=c.get("max_pos", 4))
        r["config_id"] = c["id"]
        r["desc"] = c["desc"]
        r["total_sigs"] = total_sigs
        r["score"] = _score(r)
        results.append(r)
        pf_s = "%.2f" % r["profit_factor"] if r["trades"] > 0 else "N/A"
        print("  [%02d/%02d] %-40s sigs=%4d n=%3d PF=%s PnL=%+.1f%% DD=%.1f%% score=%.3f" % (
            k+1, len(cfgs), c["id"]+" "+c["desc"][:22], total_sigs,
            r["trades"], pf_s, r["pnl_pct"], r["max_dd_pct"], r["score"]), flush=True)
    results.sort(key=lambda x: x["score"], reverse=True)
    # top-3 with regime breakdown
    print("\n  Top-3:", flush=True)
    for i, r in enumerate(results[:3]):
        print("    %d. %-30s PF=%.3f PnL=%+.1f%% WR=%.0f%% DD=%.1f%% n=%d MFEcap=%.1f%%" % (
            i+1, r["config_id"], r["profit_factor"], r["pnl_pct"],
            r["wr_pct"], r["max_dd_pct"], r["trades"], r.get("mfe_cap", 0)), flush=True)
        for rg, rv in sorted(r.get("per_regime", {}).items()):
            if rv["n"] > 0:
                print("       %s: n=%d WR=%.0f%% PF=%.2f" % (rg, rv["n"], rv["wr"], rv["pf"]), flush=True)
    return results


# ===========================================================================
#  PHASE 6  --  COMBINED PORTFOLIO
# ===========================================================================

def phase_combine(all_data, extra_ind, regime_cache, strategy_results):
    """Combine best config from each viable strategy into a portfolio."""
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 6 -- Combined Portfolio", flush=True)
    print("=" * 70, flush=True)

    # Collect viable strategies (PF > 1.0 and n > 10)
    viable = {}
    strat_map = {
        "BBS": (BBS_CFGS, gen_bbs_signals),
        "RANGE": (RANGE_CFGS, gen_range_signals),
        "FAKEOUT": (FAKEOUT_CFGS, gen_fakeout_signals),
        "FVG2": (FVG2_CFGS, gen_fvg2_signals),
    }
    for sname, res_list in strategy_results.items():
        if not res_list:
            continue
        best = res_list[0]
        if best["profit_factor"] >= 1.0 and best["trades"] >= 10:
            viable[sname] = best
            print("  Viable: %s -> %s (PF=%.2f, PnL=%+.1f%%)" % (
                sname, best["config_id"], best["profit_factor"], best["pnl_pct"]), flush=True)
        else:
            print("  Excluded: %s -> best PF=%.2f n=%d (below threshold)" % (
                sname, best["profit_factor"], best["trades"]), flush=True)

    if not viable:
        print("  No viable strategies found!", flush=True)
        return {}

    # Generate combined signals: priority = higher PF strategy wins conflicts
    priority = sorted(viable.keys(), key=lambda s: viable[s]["profit_factor"], reverse=True)
    all_combined = {}
    for sym in all_data:
        sym_sigs = {}
        for sname in priority:
            cfgs, gen_fn = strat_map[sname]
            best_id = viable[sname]["config_id"]
            cfg_obj = next((c for c in cfgs if c["id"] == best_id), cfgs[0])
            strat_sigs = gen_fn(all_data[sym], extra_ind[sym], regime_cache[sym], cfg_obj["params"])
            for bar_i, sig in strat_sigs.items():
                if bar_i not in sym_sigs:  # first-come (higher PF wins)
                    sig["strategy"] = sname
                    sym_sigs[bar_i] = sig
        all_combined[sym] = sym_sigs

    total_sigs = sum(len(v) for v in all_combined.values())
    print("  Combined signals: %d" % total_sigs, flush=True)

    # Find best portfolio parameters from the dominant strategy
    best_strat = priority[0] if priority else "BBS"
    best_cfg = next((c for c in strat_map[best_strat][0]
                     if c["id"] == viable[best_strat]["config_id"]),
                    strat_map[best_strat][0][0])

    # Run combined simulation
    r = run_sim(all_combined, all_data, extra_ind,
                best_cfg["sl"], best_cfg["tp"],
                risk_pt=best_cfg.get("risk", 0.01),
                cap_frac=0.80, max_pos=6)
    r["config_id"] = "combined"
    r["desc"] = "+".join(priority)
    print("  Combined: n=%d PF=%.3f PnL=%+.1f%% DD=%.1f%% WR=%.0f%%" % (
        r["trades"], r["profit_factor"], r["pnl_pct"], r["max_dd_pct"], r["wr_pct"]), flush=True)
    for rg, rv in sorted(r.get("per_regime", {}).items()):
        if rv["n"] > 0:
            print("    %s: n=%d WR=%.0f%% PF=%.2f" % (rg, rv["n"], rv["wr"], rv["pf"]), flush=True)
    return {"combined": r, "viable": {k: v["config_id"] for k, v in viable.items()},
            "priority": priority}


# ===========================================================================
#  PHASE 7  --  WALK-FORWARD
# ===========================================================================

def phase_walkforward(all_data, extra_ind, regime_cache, strategy_results, strat_map):
    """67/33 walk-forward on top-1 per viable strategy."""
    print("\n" + "=" * 70, flush=True)
    print("  PHASE 7 -- Walk-Forward Validation (67/33)", flush=True)
    print("=" * 70, flush=True)
    wf = {}
    for sname, res_list in strategy_results.items():
        if not res_list or res_list[0]["profit_factor"] < 1.0 or res_list[0]["trades"] < 10:
            continue
        best = res_list[0]
        cfgs, gen_fn = strat_map[sname]
        cfg_obj = next((c for c in cfgs if c["id"] == best["config_id"]), cfgs[0])
        # train
        tr_sigs = {}
        for sym, D in all_data.items():
            tr_sigs[sym] = gen_fn(D, extra_ind[sym], regime_cache[sym], cfg_obj["params"])
        tr = run_sim(tr_sigs, all_data, extra_ind, cfg_obj["sl"], cfg_obj["tp"],
                     risk_pt=cfg_obj.get("risk", 0.01), cap_frac=cfg_obj.get("frac", 0.5),
                     max_pos=cfg_obj.get("max_pos", 4), start_frac=0.0, end_frac=0.67)
        # test
        te = run_sim(tr_sigs, all_data, extra_ind, cfg_obj["sl"], cfg_obj["tp"],
                     risk_pt=cfg_obj.get("risk", 0.01), cap_frac=cfg_obj.get("frac", 0.5),
                     max_pos=cfg_obj.get("max_pos", 4), start_frac=0.67, end_frac=1.0)
        delta = te["pnl_pct"] - tr["pnl_pct"]
        degrad = (delta / tr["pnl_pct"] * 100) if tr["pnl_pct"] != 0 else 0
        ok = te["profit_factor"] >= 1.0 and te["pnl_pct"] > -5
        wf[sname] = {"config_id": best["config_id"],
            "train_pnl": tr["pnl_pct"], "test_pnl": te["pnl_pct"],
            "train_pf": tr["profit_factor"], "test_pf": te["profit_factor"],
            "delta": round(delta, 2), "degrad": round(degrad, 1), "pass": ok}
        print("  %-10s %-20s train=%+6.1f%% test=%+6.1f%% delta=%+5.1f%% degrad=%+4.0f%% %s" % (
            sname, best["config_id"], tr["pnl_pct"], te["pnl_pct"],
            delta, degrad, "[OK]" if ok else "[FAIL]"), flush=True)
    return wf


# ===========================================================================
#  FINAL REPORT
# ===========================================================================

def print_report(strategy_results, combined, wf):
    print("\n" + "=" * 70, flush=True)
    print("  FINAL REPORT -- Multi-Strategy Research v6", flush=True)
    print("=" * 70, flush=True)
    for sname, rlist in strategy_results.items():
        if not rlist:
            continue
        best = rlist[0]
        status = "VIABLE" if best["profit_factor"] >= 1.0 and best["trades"] >= 10 else "REJECTED"
        print("\n  -- %s [%s] --" % (sname, status), flush=True)
        for i, r in enumerate(rlist[:5]):
            print("    %d. %-30s n=%3d PF=%.3f PnL=%+6.1f%% WR=%.0f%% DD=%.1f%% MFE=%.1f%% sigs=%d" % (
                i+1, r["config_id"]+" "+r["desc"][:15], r["trades"],
                r["profit_factor"], r["pnl_pct"], r["wr_pct"],
                r["max_dd_pct"], r.get("mfe_cap", 0), r.get("total_sigs", 0)), flush=True)
    if combined:
        c = combined.get("combined", {})
        if c:
            print("\n  -- COMBINED PORTFOLIO --", flush=True)
            print("    Strategies: %s" % combined.get("priority", []), flush=True)
            print("    n=%d PF=%.3f PnL=%+.1f%% WR=%.0f%% DD=%.1f%%" % (
                c["trades"], c["profit_factor"], c["pnl_pct"],
                c["wr_pct"], c["max_dd_pct"]), flush=True)
    if wf:
        print("\n  -- WALK-FORWARD --", flush=True)
        for sname, w in wf.items():
            print("    %s: train=%+.1f%% test=%+.1f%% PF_test=%.2f %s" % (
                sname, w["train_pnl"], w["test_pnl"], w["test_pf"],
                "[OK]" if w["pass"] else "[FAIL]"), flush=True)


# ===========================================================================
#  SAVE
# ===========================================================================

def save_results(strategy_results, combined, wf):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / ("backtest_v6_%s.json" % ts)
    data = {"strategy_results": {}, "combined": {}, "walkforward": {}}
    for sname, rlist in strategy_results.items():
        data["strategy_results"][sname] = [
            {k: v for k, v in r.items() if k != "completed"} for r in rlist[:10]]
    if combined:
        c = combined.get("combined", {})
        data["combined"] = {k: v for k, v in c.items() if k != "completed"} if c else {}
        data["combined"]["viable"] = combined.get("viable", {})
        data["combined"]["priority"] = combined.get("priority", [])
    data["walkforward"] = wf
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)
    print("\n  Results saved: %s" % out, flush=True)
    return out


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    ts0 = datetime.now()
    print("backtest_v6.py -- Multi-Strategy Research Engine", flush=True)
    print("Started: %s" % ts0.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    days_tag = "180d" if _check_180d_available() else "90d"
    print("Data: %s" % days_tag, flush=True)

    all_data = preload_all(days_tag=days_tag, blacklist=BLACKLIST)
    print("Symbols loaded: %d" % len(all_data), flush=True)
    if not all_data:
        print("ERROR: no data", flush=True)
        return

    print("\n  Phase 0 -- Pre-computing indicators + regime...", flush=True)
    t0 = _time.time()
    extra_ind, regime_cache = phase0(all_data)
    print("  Phase 0 done in %.0f sec" % (_time.time() - t0), flush=True)

    strategy_results = {}
    strat_map = {
        "BBS": (BBS_CFGS, gen_bbs_signals),
        "RANGE": (RANGE_CFGS, gen_range_signals),
        "FAKEOUT": (FAKEOUT_CFGS, gen_fakeout_signals),
        "FVG2": (FVG2_CFGS, gen_fvg2_signals),
    }

    strategy_results["BBS"] = _run_phase("PHASE 1 -- BBS", BBS_CFGS, gen_bbs_signals,
                                          all_data, extra_ind, regime_cache)
    strategy_results["RANGE"] = _run_phase("PHASE 2 -- RANGE/MR", RANGE_CFGS, gen_range_signals,
                                            all_data, extra_ind, regime_cache)
    strategy_results["FAKEOUT"] = _run_phase("PHASE 3 -- FAKEOUT", FAKEOUT_CFGS, gen_fakeout_signals,
                                              all_data, extra_ind, regime_cache)
    strategy_results["FVG2"] = _run_phase("PHASE 4 -- FVG v2", FVG2_CFGS, gen_fvg2_signals,
                                           all_data, extra_ind, regime_cache)

    # Phase 6: Combined
    combined = phase_combine(all_data, extra_ind, regime_cache, strategy_results)

    # Phase 7: Walk-forward
    wf = phase_walkforward(all_data, extra_ind, regime_cache, strategy_results, strat_map)

    print_report(strategy_results, combined, wf)
    save_results(strategy_results, combined, wf)

    elapsed = _time.time() - ts0.timestamp() if hasattr(ts0, 'timestamp') else 0
    elapsed = (datetime.now() - ts0).total_seconds()
    m, s = divmod(elapsed, 60)
    print("\nTotal runtime: %.0f min %.0f sec" % (m, s), flush=True)


if __name__ == "__main__":
    main()
