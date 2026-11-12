"""
backtest_v5.py  --  Position Management Optimization  (2026-04-13)

ARCHITECTURE FIX vs naive approach:
  WRONG: run_sltp() x 70 combos = signal_detection x 70 (massive waste)
  RIGHT: pre-compute signals ONCE, then 70 combos reuse cached results

  BBS pre-compute: O(n) per symbol  -- fast
  FVG pre-compute: O(n^2) per symbol -- slow once, not 35x

  Regime cache: computed once, reused across all combos

PHASES
  0  Pre-compute regime + BBS + FVG signals (runs once)
  1  BBS -- 35 SL x TP x management combos
  2  FVG -- 35 SL x TP x management combos
  3  Walk-forward validation of top-5 per strategy
  4  Final report vs current production

KEY METRICS
  1. Profit Factor
  2. MFE capture %  (exit_R / max_favorable_R)
  3. Profit giveback %  (reached 1R, exited < 0.5R)
  4. Max DD
  5. PnL %
"""
from __future__ import annotations
import json, logging, sys
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
    _atr, _adx,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
from multi_agent.backtests.backtest_v4 import (
    preload_all, _check_180d_available, _fvg_signal_v4,
    SYMBOLS, BBS_BLACKLIST_CANDIDATES,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
)
import multi_agent.config as cfg

logger = logging.getLogger("v5")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---- risk / capacity ---------------------------------------------------
BBS_RISK    = 0.010
BBS_FRAC    = 0.75
BBS_MAX_POS = 4

FVG_RISK    = 0.010
FVG_FRAC    = 0.50
FVG_MAX_POS = 2

BLACKLIST = BBS_BLACKLIST_CANDIDATES   # ARB, OP, POL

# ---- regime gates (no_squeeze from v4) --------------------------------
BBS_REGIME_MULT = {
    "TREND":      1.0,
    "TRANSITION": 0.85,
    "RANGE":      0.0,
    "SQUEEZE":    0.0,
    "UNDEFINED":  0.0,
}
FVG_REGIME_MULT = {
    "SQUEEZE":    1.0,
    "TRANSITION": 0.5,
    "TREND":      0.0,
    "RANGE":      0.0,
    "UNDEFINED":  0.0,
}

# ---- FVG signal params (v4 optimized) ---------------------------------
FVG_P = {
    "fvg_sl_atr_mult":    2.0,
    "fvg_sl_atr_max_pct": 2.5,
    "fvg_exit_mode":      "fixed",
    "fvg_fixed_tp_r":     3.0,
    "fvg_adx_min":        15.0,
    "fvg_volume_mult":    0.6,
    "fvg_quality_min":    40.0,
    "fvg_entry_lo":       0.20,
    "fvg_entry_hi":       0.80,
    "fvg_composite_lo":   55.0,
    "fvg_composite_hi":   90.0,
    "fvg_fib_mandatory":  False,
    "fvg_fib_bonus":      5.0,
    "fvg_mult_squeeze":   1.0,
    "fvg_mult_transition":0.5,
}

SOFT_CFG = {
    "confirm_bars":        5,
    "min_hold_bars":       12,
    "smooth_alpha":        0.30,
    "min_confidence":      1.0,
    "transition_margin":   0.5,
    "undefined_threshold": 0.5,
}


# ============================================================
#  PHASE 0 -- PRE-COMPUTE SIGNALS & REGIME  (runs once)
# ============================================================

def precompute_all(all_data: dict) -> dict:
    """
    Pre-compute for each symbol:
      - regime_arr[i]     : regime string per 15m bar
      - bbs_signals[i]    : BBS signal dict if fires at bar i, else None
      - fvg_signals[i]    : FVG signal dict if fires at bar i, else None

    Returns cache dict: {sym: {"regime": [...], "bbs": {i: sig}, "fvg": {i: sig}}}
    FVG is slow (O(n^2)) but runs only ONCE here.
    """
    cache = {}
    total = len(all_data)
    for k, (sym, D) in enumerate(all_data.items()):
        n = D["n_bars"]
        print("  [%d/%d] Pre-computing %s (%d bars)..." % (k+1, total, sym, n), flush=True)
        regime_arr = ["UNDEFINED"] * n
        bbs_sigs   = {}   # bar_idx -> signal dict
        fvg_sigs   = {}   # bar_idx -> signal dict

        detector = SoftRegimeDetector(SOFT_CFG)

        for i in range(300, n):
            cl_i = D["cl15"][i]
            ts_i = D["ts15"][i]

            htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
            if htf_idx < 210:
                continue

            rs = detector.update(
                sym, D["df1h"], htf_idx, D["df4h"],
                **(D["ind_1h"]), **(D["ind_4h"]))
            regime_str = rs.regime.value
            regime_arr[i] = regime_str

            # ---- BBS signal ----------------------------------------
            bbs_m = BBS_REGIME_MULT.get(regime_str, 0.0)
            if bbs_m >= 0.05 and D["squeeze"] is not None:
                bbs_s = detect_squeeze_signal(
                    D["squeeze"], bar_idx=i,
                    min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                    adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                    vol_mult=cfg.BB_SQUEEZE_VOL_MULT)
                if bbs_s:
                    ema200 = D["ind_1h"]["ema200_1h"]
                    cl1h   = D["ind_1h"]["closes_1h_arr"]
                    htf_d  = "LONG" if cl1h[htf_idx] > ema200[htf_idx] else "SHORT"
                    if bbs_s["direction"] == htf_d and not check_head_fake(D["squeeze"], i, 2):
                        ce_sl = chandelier_exit_sl(
                            D["hi15"], D["lo15"], D["squeeze"]["ce_atr"],
                            bbs_s["direction"], i,
                            ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                            ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                        risk_v = abs(cl_i - ce_sl)
                        if 0 < risk_v / cl_i <= 0.05:
                            bbs_sigs[i] = {
                                "direction":    bbs_s["direction"],
                                "entry_price":  float(cl_i),
                                "default_sl":   float(ce_sl),
                                "regime":       regime_str,
                                "size_m":       bbs_m,
                            }

            # ---- FVG signal (O(n^2) -- done only once here) --------
            fvg_m = FVG_REGIME_MULT.get(regime_str, 0.0)
            if fvg_m >= 0.05:
                sig = _fvg_signal_v4(D, i, FVG_P)
                if sig:
                    fvg_sigs[i] = {
                        "direction":   sig["direction"],
                        "entry_price": float(sig.get("entry_price", cl_i)),
                        "default_sl":  float(sig["sl"]),
                        "regime":      regime_str,
                        "size_m":      fvg_m,
                    }

        cache[sym] = {
            "regime": regime_arr,
            "bbs":    bbs_sigs,
            "fvg":    fvg_sigs,
        }
        nb = len(bbs_sigs)
        nf = len(fvg_sigs)
        print("     -> BBS signals: %d  FVG signals: %d" % (nb, nf), flush=True)

    return cache


# ============================================================
#  SL COMPUTATION
# ============================================================

def _sl_atr(d: str, entry: float, atr: float, mult: float) -> float:
    return (entry - atr * mult) if d == "LONG" else (entry + atr * mult)

def _sl_struct(d: str, hi: np.ndarray, lo: np.ndarray, i: int, n: int) -> float:
    s = max(0, i - n + 1)
    return float(np.min(lo[s:i+1])) if d == "LONG" else float(np.max(hi[s:i+1]))

def _sl_hybrid(d: str, a: float, b: float, mode: str) -> float:
    if d == "LONG":
        return max(a, b) if mode == "tight" else min(a, b)
    return min(a, b) if mode == "tight" else max(a, b)

def compute_sl(sl_cfg: dict, d: str, entry: float,
               atr: float, hi: np.ndarray, lo: np.ndarray,
               i: int, default_sl: float) -> float:
    t = sl_cfg["type"]
    if t == "default":
        sl = default_sl
    elif t == "atr":
        sl = _sl_atr(d, entry, atr, sl_cfg["mult"])
    elif t == "structure":
        sl = _sl_struct(d, hi, lo, i, sl_cfg["n_bars"])
    elif t == "hybrid_tight":
        sl = _sl_hybrid(d, _sl_atr(d, entry, atr, sl_cfg["atr_mult"]),
                        _sl_struct(d, hi, lo, i, sl_cfg["struct_bars"]), "tight")
    elif t == "hybrid_wide":
        sl = _sl_hybrid(d, _sl_atr(d, entry, atr, sl_cfg["atr_mult"]),
                        _sl_struct(d, hi, lo, i, sl_cfg["struct_bars"]), "wide")
    elif t == "adaptive":
        atr_pct = atr / entry if entry > 0 else 0.01
        mult = sl_cfg["mult_low"] if atr_pct < sl_cfg["thresh"] else sl_cfg["mult_high"]
        sl = _sl_atr(d, entry, atr, mult)
    else:
        sl = default_sl
    # guard
    if d == "LONG"  and sl >= entry: sl = entry * 0.98
    if d == "SHORT" and sl <= entry: sl = entry * 1.02
    return float(sl)


# ============================================================
#  ROLLING VWAP
# ============================================================

def _vwap(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray,
          vo: np.ndarray, i: int, lb: int = 96) -> float:
    s = max(0, i - lb + 1)
    tp  = (hi[s:i+1] + lo[s:i+1] + cl[s:i+1]) / 3.0
    vol = vo[s:i+1]
    sv  = np.sum(vol)
    return float(np.sum(tp * vol) / sv) if sv > 0 else float(cl[i])


# ============================================================
#  CORE SIMULATION  (uses pre-cached signals)
# ============================================================

def run_sltp(combo: dict, agent_name: str,
             all_data: dict, cache: dict,
             start_frac: float = 0.0,
             end_frac:   float = 1.0) -> dict:
    """
    Run one SL/TP/management combo using pre-cached signals.
    Bar-by-bar only for OPEN POSITION management (fast).
    Signal lookup is O(1) per bar.
    """
    sl_cfg  = combo["sl"]
    tp_cfg  = combo["tp"]
    mgt_cfg = combo.get("mgt", {"be_r": 0.0})

    regime_mult = BBS_REGIME_MULT if agent_name == "bbs" else FVG_REGIME_MULT
    risk_pt     = BBS_RISK  if agent_name == "bbs" else FVG_RISK
    cap_frac    = BBS_FRAC  if agent_name == "bbs" else FVG_FRAC
    max_pos     = BBS_MAX_POS if agent_name == "bbs" else FVG_MAX_POS

    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    open_pos  = {}
    agent_cnt = 0
    completed = []

    max_n   = max(d["n_bars"] for d in all_data.values())
    i_start = max(300, int(max_n * start_frac))
    i_end   = int(max_n * end_frac)

    for i in range(i_start, i_end):
        for sym, D in all_data.items():
            if i >= D["n_bars"]:
                continue

            cl_i  = D["cl15"][i]
            hi_i  = D["hi15"][i]
            lo_i  = D["lo15"][i]
            atr_i = D["atr15"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            sym_cache  = cache.get(sym, {})
            regime_str = sym_cache.get("regime", [])[i] if i < len(sym_cache.get("regime", [])) else "UNDEFINED"
            size_m     = regime_mult.get(regime_str, 0.0)

            # ── Manage open position ──────────────────────────────────
            if sym in open_pos:
                pos   = open_pos[sym]
                d     = pos["direction"]
                entry = pos["entry"]
                sl    = pos["sl"]
                r     = pos["risk"]
                q     = pos["qty"]
                bars  = i - pos["bar_idx"]

                # MFE / MAE (using bar high/low)
                if d == "LONG":
                    cur_mfe = (hi_i - entry) / r
                    cur_mae = (entry - lo_i) / r
                else:
                    cur_mfe = (entry - lo_i) / r
                    cur_mae = (hi_i - entry) / r
                pos["mfe"] = max(pos["mfe"], cur_mfe)
                pos["mae"] = max(pos["mae"], cur_mae)

                # P&L in R at close
                pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)

                # Break-even management
                be_r = mgt_cfg.get("be_r", 0.0)
                if be_r > 0 and pnl_r >= be_r and not pos.get("be_applied"):
                    buf = entry * 0.0005
                    new_be = (entry + buf) if d == "LONG" else (entry - buf)
                    better = (d == "LONG" and new_be > pos["sl"]) or \
                             (d == "SHORT" and (new_be < pos["sl"] or pos["sl"] == 0))
                    if better:
                        pos["sl"] = new_be
                        sl = new_be
                    pos["be_applied"] = True

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp   = False
                tp_price = 0.0
                exit_rsn = "sl_hit" if hit_sl else ""

                tp_type = tp_cfg["type"]

                if not hit_sl:
                    # ---- fixed R --------------------------------------
                    if tp_type == "fixed_r":
                        if pnl_r >= tp_cfg["r"]:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_fixed"

                    # ---- partial fixed --------------------------------
                    elif tp_type == "partial_fixed":
                        r1, r2, frac = tp_cfg["r1"], tp_cfg["r2"], tp_cfg["frac"]
                        if not pos.get("partial_done") and pnl_r >= r1:
                            pq      = q * frac
                            gross_p = ((cl_i - entry) if d == "LONG" else (entry - cl_i)) * pq
                            net_p   = gross_p - cl_i * pq * (TAKER_FEE + SLIPPAGE)
                            balance += net_p
                            pos["qty"] -= pq
                            q = pos["qty"]
                            pos["partial_done"] = True
                            pos["partial_pnl"]  = net_p
                            # BE after partial
                            if d == "LONG" and entry > pos["sl"]:
                                pos["sl"] = entry; sl = entry
                            elif d == "SHORT" and (entry < pos["sl"] or pos["sl"] == 0):
                                pos["sl"] = entry; sl = entry
                        if pos.get("partial_done") and pnl_r >= r2:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_partial_r2"

                    # ---- chandelier trail ----------------------------
                    elif tp_type == "trail_chandelier":
                        if pnl_r >= tp_cfg.get("trail_start_r", 1.0) and D["squeeze"] is not None:
                            new_ce = chandelier_exit_sl(
                                D["hi15"], D["lo15"], D["squeeze"]["ce_atr"], d, i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                                ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            if d == "LONG":
                                pos["trail_sl"] = max(pos.get("trail_sl", sl), new_ce)
                            else:
                                ct = pos.get("trail_sl", 0)
                                pos["trail_sl"] = min(ct, new_ce) if ct > 0 else new_ce
                        eff = pos.get("trail_sl", sl)
                        if d == "LONG":
                            eff = max(sl, eff)
                            if lo_i <= eff: hit_sl = True; sl = eff; exit_rsn = "trail_ce"
                        else:
                            eff = min(sl, eff) if eff > 0 else sl
                            if hi_i >= eff: hit_sl = True; sl = eff; exit_rsn = "trail_ce"
                        cap = tp_cfg.get("cap_r", 0)
                        if cap > 0 and pnl_r >= cap and not hit_sl:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "trail_cap"

                    # ---- ATR trail -----------------------------------
                    elif tp_type == "trail_atr":
                        if pnl_r >= tp_cfg.get("trail_start_r", 1.0):
                            tm = tp_cfg.get("trail_mult", 2.0)
                            if d == "LONG":
                                pos["best_px"] = max(pos.get("best_px", entry), hi_i)
                                nt = pos["best_px"] - atr_i * tm
                                pos["trail_sl"] = max(pos.get("trail_sl", sl), nt)
                            else:
                                pos["best_px"] = min(pos.get("best_px", entry), lo_i)
                                nt = pos["best_px"] + atr_i * tm
                                ct = pos.get("trail_sl", 0)
                                pos["trail_sl"] = min(ct, nt) if ct > 0 else nt
                        eff = pos.get("trail_sl", sl)
                        if d == "LONG":
                            eff = max(sl, eff)
                            if lo_i <= eff: hit_sl = True; sl = eff; exit_rsn = "trail_atr"
                        else:
                            eff = min(sl, eff) if eff > 0 else sl
                            if hi_i >= eff: hit_sl = True; sl = eff; exit_rsn = "trail_atr"
                        cap = tp_cfg.get("cap_r", 0)
                        if cap > 0 and pnl_r >= cap and not hit_sl:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "trail_atr_cap"

                    # ---- structure trail -----------------------------
                    elif tp_type == "trail_struct":
                        if pnl_r >= tp_cfg.get("trail_start_r", 1.0):
                            nb = tp_cfg.get("n_bars", 10)
                            st = max(0, i - nb + 1)
                            if d == "LONG":
                                nt = float(np.min(D["lo15"][st:i+1]))
                                pos["trail_sl"] = max(pos.get("trail_sl", sl), nt)
                            else:
                                nt = float(np.max(D["hi15"][st:i+1]))
                                ct = pos.get("trail_sl", 0)
                                pos["trail_sl"] = min(ct, nt) if ct > 0 else nt
                        eff = pos.get("trail_sl", sl)
                        if d == "LONG":
                            eff = max(sl, eff)
                            if lo_i <= eff: hit_sl = True; sl = eff; exit_rsn = "trail_struct"
                        else:
                            eff = min(sl, eff) if eff > 0 else sl
                            if hi_i >= eff: hit_sl = True; sl = eff; exit_rsn = "trail_struct"

                    # ---- VWAP exits ----------------------------------
                    elif tp_type == "vwap_full":
                        vwap = _vwap(D["hi15"], D["lo15"], D["cl15"], D["vo15"], i)
                        if pnl_r >= tp_cfg.get("min_r", 0.5):
                            if (d == "LONG" and hi_i >= vwap) or (d == "SHORT" and lo_i <= vwap):
                                hit_tp = True
                                tp_price = min(vwap, cl_i) if d == "LONG" else max(vwap, cl_i)
                                exit_rsn = "tp_vwap"

                    elif tp_type == "vwap_partial":
                        vwap  = _vwap(D["hi15"], D["lo15"], D["cl15"], D["vo15"], i)
                        min_r = tp_cfg.get("min_r", 0.5)
                        frac  = tp_cfg.get("frac",  0.5)
                        cap_r = tp_cfg.get("cap_r", 3.0)
                        if not pos.get("partial_done") and pnl_r >= min_r:
                            hit_vwap = (d == "LONG" and hi_i >= vwap) or (d == "SHORT" and lo_i <= vwap)
                            if hit_vwap:
                                ep      = min(vwap, cl_i) if d == "LONG" else max(vwap, cl_i)
                                pq      = q * frac
                                gross_p = ((ep - entry) if d == "LONG" else (entry - ep)) * pq
                                net_p   = gross_p - ep * pq * (TAKER_FEE + SLIPPAGE)
                                balance += net_p
                                pos["qty"] -= pq; q = pos["qty"]
                                pos["partial_done"] = True
                                pos["partial_pnl"]  = net_p
                                pos["sl"] = entry; sl = entry
                        if pos.get("partial_done") and pnl_r >= cap_r:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_vwap_cap"

                    # ---- time exit -----------------------------------
                    elif tp_type == "time_exit":
                        min_r = tp_cfg.get("min_r", 0.0)
                        if min_r > 0 and pnl_r >= min_r:
                            hit_tp = True; tp_price = cl_i; exit_rsn = "tp_min_r"
                        elif bars >= tp_cfg.get("max_bars", 50):
                            hit_tp = True; tp_price = cl_i; exit_rsn = "time_exit"

                # ---- Execute exit ------------------------------------
                if hit_sl or hit_tp:
                    if not exit_rsn:
                        exit_rsn = "tp_hit" if hit_tp else "sl_hit"
                    ep    = tp_price if hit_tp else sl
                    q_out = pos["qty"]
                    gross = ((ep - entry) if d == "LONG" else (entry - ep)) * q_out
                    fee   = ep * q_out * (TAKER_FEE + SLIPPAGE)
                    net   = gross - fee
                    balance += net
                    peak   = max(peak, balance)
                    dd     = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    total_net = net + pos.get("partial_pnl", 0.0)
                    completed.append({
                        "sym":         sym,
                        "direction":   d,
                        "exit_r":      round(exit_r, 3),
                        "mfe":         round(pos["mfe"], 3),
                        "mae":         round(pos["mae"], 3),
                        "bars":        bars,
                        "won":         total_net > 0,
                        "pnl_usd":     round(total_net, 4),
                        "exit_reason": exit_rsn,
                        "regime":      pos["regime"],
                    })
                    agent_cnt = max(0, agent_cnt - 1)
                    del open_pos[sym]
                continue   # skip signal scan while position was open

            # ── Signal lookup (O(1) -- pre-cached) ──────────────────────
            if agent_cnt >= max_pos:
                continue
            if size_m < 0.05:
                continue

            sig_cache = sym_cache.get("bbs" if agent_name == "bbs" else "fvg", {})
            if i not in sig_cache:
                continue

            sig = sig_cache[i]
            new_sl    = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                   atr_i, D["hi15"], D["lo15"], i, sig["default_sl"])
            risk_dist = abs(sig["entry_price"] - new_sl)
            if risk_dist <= 0: continue
            if risk_dist / sig["entry_price"] > 0.05: continue
            if risk_dist / sig["entry_price"] < 0.001: continue

            agent_bal = balance * cap_frac
            risk_usd  = agent_bal * risk_pt * size_m
            qty       = risk_usd / risk_dist
            margin    = sig["entry_price"] * qty / MAX_LEVERAGE
            if margin > agent_bal * 0.4:
                qty = agent_bal * 0.4 * MAX_LEVERAGE / sig["entry_price"]
            if qty * sig["entry_price"] < 5.0:
                continue

            entry_fee = sig["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
            balance  -= entry_fee

            open_pos[sym] = {
                "direction":    sig["direction"],
                "entry":        sig["entry_price"],
                "sl":           new_sl,
                "risk":         risk_dist,
                "qty":          qty,
                "bar_idx":      i,
                "mfe":          0.0,
                "mae":          0.0,
                "partial_done": False,
                "partial_pnl":  0.0,
                "be_applied":   False,
                "regime":       sig["regime"],
            }
            agent_cnt += 1

    # ── Aggregate ─────────────────────────────────────────────────────────────
    if not completed:
        return {
            "combo_id": combo["id"], "desc": combo["desc"],
            "trades": 0, "wr_pct": 0, "pnl_pct": 0,
            "profit_factor": 0, "max_dd_pct": 0,
            "mfe_capture_pct": 0, "giveback_pct": 0,
            "stop_eff_pct": 0, "avg_mfe": 0, "avg_mae": 0,
            "avg_exit_r": 0, "avg_bars": 0, "exit_reasons": {},
        }

    wins   = [t for t in completed if     t["won"]]
    losses = [t for t in completed if not t["won"]]
    n      = len(completed)
    gp     = sum(t["pnl_usd"] for t in wins)
    gl     = abs(sum(t["pnl_usd"] for t in losses))
    pf     = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0.0)
    wr     = len(wins) / n * 100
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100

    mfe_v   = [t for t in completed if t["mfe"] >= 0.5]
    mfe_cap = (np.mean([max(0, t["exit_r"]) / t["mfe"] for t in mfe_v]) * 100
               if mfe_v else 0.0)
    r1      = [t for t in completed if t["mfe"] >= 1.0]
    gb      = len([t for t in r1 if t["exit_r"] < 0.5]) / len(r1) * 100 if r1 else 0.0
    early   = [t for t in losses if t["mfe"] < 0.5]
    stop_eff = (1.0 - len(early) / len(losses)) * 100 if losses else 100.0

    reasons = defaultdict(int)
    for t in completed:
        reasons[t["exit_reason"]] += 1

    return {
        "combo_id":        combo["id"],
        "desc":            combo["desc"],
        "trades":          n,
        "wins":            len(wins),
        "losses":          len(losses),
        "wr_pct":          round(wr,      1),
        "pnl_pct":         round(pnl_pct, 2),
        "pnl_usd":         round(balance - START_BALANCE, 2),
        "profit_factor":   round(pf,      3),
        "max_dd_pct":      round(max_dd,  2),
        "avg_exit_r":      round(float(np.mean([t["exit_r"] for t in completed])), 3),
        "avg_win_r":       round(float(np.mean([t["exit_r"] for t in wins]))   if wins   else 0, 3),
        "avg_loss_r":      round(float(np.mean([t["exit_r"] for t in losses])) if losses else 0, 3),
        "avg_mfe":         round(float(np.mean([t["mfe"]    for t in completed])), 3),
        "avg_mae":         round(float(np.mean([t["mae"]    for t in completed])), 3),
        "mfe_capture_pct": round(mfe_cap,  1),
        "giveback_pct":    round(gb,       1),
        "stop_eff_pct":    round(stop_eff, 1),
        "avg_bars":        round(float(np.mean([t["bars"]   for t in completed])), 1),
        "exit_reasons":    dict(reasons),
        "final_balance":   round(balance,  2),
        "completed_trades": completed,
    }


# ============================================================
#  SCORE
# ============================================================

def _score(r: dict) -> float:
    if r["trades"] < 10: return -999.0
    pf  = min(r["profit_factor"], 5.0)
    mfe = r["mfe_capture_pct"] / 100.0
    gb  = r["giveback_pct"]    / 100.0
    dd  = r["max_dd_pct"]      / 100.0
    pnl = max(r["pnl_pct"], -50) / 100.0
    return 0.35*pf + 0.30*mfe - 0.15*gb - 0.10*dd + 0.10*pnl


# ============================================================
#  COMBO DEFINITIONS
# ============================================================

def _c(cid, desc, sl, tp, mgt=None):
    return {"id": cid, "desc": desc, "sl": sl, "tp": tp,
            "mgt": mgt or {"be_r": 0.0}}

_BE1  = {"be_r": 1.0}
_BE15 = {"be_r": 1.5}

BBS_COMBOS = [
    _c("bbs_01", "chandelier+fixed3.5R [CURRENT]",
       {"type":"default"}, {"type":"fixed_r","r":3.5}),
    _c("bbs_02", "chandelier+fixed3.0R",
       {"type":"default"}, {"type":"fixed_r","r":3.0}),
    _c("bbs_03", "chandelier+fixed4.0R",
       {"type":"default"}, {"type":"fixed_r","r":4.0}),
    _c("bbs_04", "chandelier+fixed2.5R",
       {"type":"default"}, {"type":"fixed_r","r":2.5}),
    _c("bbs_05", "chandelier+trail_ce",
       {"type":"default"}, {"type":"trail_chandelier","trail_start_r":1.0}),
    _c("bbs_06", "chandelier+trail_ce_cap4R",
       {"type":"default"}, {"type":"trail_chandelier","trail_start_r":1.0,"cap_r":4.0}),
    _c("bbs_07", "chandelier+partial50@1.5R->3.5R",
       {"type":"default"}, {"type":"partial_fixed","r1":1.5,"frac":0.5,"r2":3.5}),
    _c("bbs_08", "chandelier+partial50@2R->3.5R",
       {"type":"default"}, {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.5}),
    _c("bbs_09", "chandelier+fixed3.5R+BE@1R",
       {"type":"default"}, {"type":"fixed_r","r":3.5}, _BE1),
    _c("bbs_10", "chandelier+fixed3.5R+BE@1.5R",
       {"type":"default"}, {"type":"fixed_r","r":3.5}, _BE15),
    _c("bbs_11", "ATR1.0+fixed3.5R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":3.5}),
    _c("bbs_12", "ATR1.5+fixed3.5R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":3.5}),
    _c("bbs_13", "ATR2.0+fixed3.5R",
       {"type":"atr","mult":2.0}, {"type":"fixed_r","r":3.5}),
    _c("bbs_14", "ATR2.5+fixed3.5R",
       {"type":"atr","mult":2.5}, {"type":"fixed_r","r":3.5}),
    _c("bbs_15", "ATR1.5+fixed3.0R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":3.0}),
    _c("bbs_16", "ATR1.5+fixed4.0R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":4.0}),
    _c("bbs_17", "ATR1.5+trail_ce",
       {"type":"atr","mult":1.5}, {"type":"trail_chandelier","trail_start_r":1.0}),
    _c("bbs_18", "ATR1.5+trail_atr2.0",
       {"type":"atr","mult":1.5}, {"type":"trail_atr","trail_start_r":1.0,"trail_mult":2.0}),
    _c("bbs_19", "ATR1.5+partial50@2R->3.5R",
       {"type":"atr","mult":1.5}, {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.5}),
    _c("bbs_20", "ATR1.5+fixed3.5R+BE@1R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":3.5}, _BE1),
    _c("bbs_21", "ATR2.0+trail_atr1.5",
       {"type":"atr","mult":2.0}, {"type":"trail_atr","trail_start_r":1.0,"trail_mult":1.5}),
    _c("bbs_22", "ATR2.0+partial50@2R->3.5R+BE@1R",
       {"type":"atr","mult":2.0}, {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.5}, _BE1),
    _c("bbs_23", "struct5+fixed3.5R",
       {"type":"structure","n_bars":5}, {"type":"fixed_r","r":3.5}),
    _c("bbs_24", "struct10+fixed3.5R",
       {"type":"structure","n_bars":10}, {"type":"fixed_r","r":3.5}),
    _c("bbs_25", "struct5+trail_ce",
       {"type":"structure","n_bars":5}, {"type":"trail_chandelier","trail_start_r":1.0}),
    _c("bbs_26", "hybrid_tight(ATR1.5+str5)+fixed3.5R",
       {"type":"hybrid_tight","atr_mult":1.5,"struct_bars":5}, {"type":"fixed_r","r":3.5}),
    _c("bbs_27", "hybrid_wide(ATR2.0+str10)+fixed3.5R",
       {"type":"hybrid_wide","atr_mult":2.0,"struct_bars":10}, {"type":"fixed_r","r":3.5}),
    _c("bbs_28", "hybrid_tight+partial50@2R->3.5R+BE@1R",
       {"type":"hybrid_tight","atr_mult":1.5,"struct_bars":5},
       {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.5}, _BE1),
    _c("bbs_29", "adaptive(1.5x<1%,2.5x>=1%)+fixed3.5R",
       {"type":"adaptive","thresh":0.01,"mult_low":1.5,"mult_high":2.5},
       {"type":"fixed_r","r":3.5}),
    _c("bbs_30", "adaptive+trail_ce",
       {"type":"adaptive","thresh":0.01,"mult_low":1.5,"mult_high":2.5},
       {"type":"trail_chandelier","trail_start_r":1.0}),
    _c("bbs_31", "ATR1.5+vwap_partial(50%)+cap3.5R",
       {"type":"atr","mult":1.5},
       {"type":"vwap_partial","min_r":0.5,"frac":0.5,"cap_r":3.5}),
    _c("bbs_32", "chandelier+vwap_full(min0.5R)",
       {"type":"default"}, {"type":"vwap_full","min_r":0.5}),
    _c("bbs_33", "ATR1.5+trail_struct10",
       {"type":"atr","mult":1.5},
       {"type":"trail_struct","trail_start_r":1.0,"n_bars":10}),
    _c("bbs_34", "ATR1.5+partial30@2R->5R",
       {"type":"atr","mult":1.5},
       {"type":"partial_fixed","r1":2.0,"frac":0.3,"r2":5.0}),
    _c("bbs_35", "chandelier+partial50@1.5R->3.5R+BE@1R",
       {"type":"default"},
       {"type":"partial_fixed","r1":1.5,"frac":0.5,"r2":3.5}, _BE1),
]

FVG_COMBOS = [
    _c("fvg_01", "ATR2.0+fixed3.0R [CURRENT]",
       {"type":"default"}, {"type":"fixed_r","r":3.0}),
    _c("fvg_02", "ATR1.5+fixed3.0R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":3.0}),
    _c("fvg_03", "ATR1.0+fixed3.0R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":3.0}),
    _c("fvg_04", "ATR1.0+fixed2.5R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":2.5}),
    _c("fvg_05", "ATR1.5+fixed2.5R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":2.5}),
    _c("fvg_06", "ATR2.0+fixed2.5R",
       {"type":"atr","mult":2.0}, {"type":"fixed_r","r":2.5}),
    _c("fvg_07", "ATR1.0+fixed3.5R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":3.5}),
    _c("fvg_08", "ATR1.5+fixed3.5R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":3.5}),
    _c("fvg_09", "ATR1.0+fixed2.0R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":2.0}),
    _c("fvg_10", "ATR2.0+fixed3.5R",
       {"type":"atr","mult":2.0}, {"type":"fixed_r","r":3.5}),
    _c("fvg_11", "struct3+fixed3.0R",
       {"type":"structure","n_bars":3}, {"type":"fixed_r","r":3.0}),
    _c("fvg_12", "struct5+fixed3.0R",
       {"type":"structure","n_bars":5}, {"type":"fixed_r","r":3.0}),
    _c("fvg_13", "struct3+fixed2.5R",
       {"type":"structure","n_bars":3}, {"type":"fixed_r","r":2.5}),
    _c("fvg_14", "hybrid_tight(ATR1.0+str3)+fixed3.0R",
       {"type":"hybrid_tight","atr_mult":1.0,"struct_bars":3},
       {"type":"fixed_r","r":3.0}),
    _c("fvg_15", "hybrid_tight(ATR1.5+str5)+fixed3.0R",
       {"type":"hybrid_tight","atr_mult":1.5,"struct_bars":5},
       {"type":"fixed_r","r":3.0}),
    _c("fvg_16", "ATR1.0+fixed3.0R+BE@1R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":3.0}, _BE1),
    _c("fvg_17", "ATR1.5+fixed3.0R+BE@1R",
       {"type":"atr","mult":1.5}, {"type":"fixed_r","r":3.0}, _BE1),
    _c("fvg_18", "ATR1.0+fixed3.0R+BE@1.5R",
       {"type":"atr","mult":1.0}, {"type":"fixed_r","r":3.0}, _BE15),
    _c("fvg_19", "struct3+fixed2.5R+BE@1R",
       {"type":"structure","n_bars":3}, {"type":"fixed_r","r":2.5}, _BE1),
    _c("fvg_20", "ATR1.0+partial50@1.5R->3.0R",
       {"type":"atr","mult":1.0},
       {"type":"partial_fixed","r1":1.5,"frac":0.5,"r2":3.0}),
    _c("fvg_21", "ATR1.5+partial50@2R->3.0R",
       {"type":"atr","mult":1.5},
       {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.0}),
    _c("fvg_22", "ATR1.0+partial50@1.5R->3.0R+BE@1R",
       {"type":"atr","mult":1.0},
       {"type":"partial_fixed","r1":1.5,"frac":0.5,"r2":3.0}, _BE1),
    _c("fvg_23", "struct5+partial50@2R->3.0R",
       {"type":"structure","n_bars":5},
       {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.0}),
    _c("fvg_24", "ATR1.0+partial50@1.5R->2.5R",
       {"type":"atr","mult":1.0},
       {"type":"partial_fixed","r1":1.5,"frac":0.5,"r2":2.5}),
    _c("fvg_25", "ATR1.0+trail_atr1.5",
       {"type":"atr","mult":1.0},
       {"type":"trail_atr","trail_start_r":1.0,"trail_mult":1.5}),
    _c("fvg_26", "ATR1.5+trail_ce",
       {"type":"atr","mult":1.5},
       {"type":"trail_chandelier","trail_start_r":1.0}),
    _c("fvg_27", "ATR1.5+trail_struct5",
       {"type":"atr","mult":1.5},
       {"type":"trail_struct","trail_start_r":1.0,"n_bars":5}),
    _c("fvg_28", "ATR2.0+trail_atr1.5",
       {"type":"atr","mult":2.0},
       {"type":"trail_atr","trail_start_r":1.0,"trail_mult":1.5}),
    _c("fvg_29", "adaptive(0.8x<0.8%,1.5x>=0.8%)+fixed3.0R",
       {"type":"adaptive","thresh":0.008,"mult_low":0.8,"mult_high":1.5},
       {"type":"fixed_r","r":3.0}),
    _c("fvg_30", "adaptive+trail_atr1.5",
       {"type":"adaptive","thresh":0.008,"mult_low":0.8,"mult_high":1.5},
       {"type":"trail_atr","trail_start_r":1.0,"trail_mult":1.5}),
    _c("fvg_31", "ATR1.5+vwap_partial(50%)+cap3.0R",
       {"type":"atr","mult":1.5},
       {"type":"vwap_partial","min_r":0.5,"frac":0.5,"cap_r":3.0}),
    _c("fvg_32", "ATR1.0+time_exit@40bars",
       {"type":"atr","mult":1.0},
       {"type":"time_exit","max_bars":40,"min_r":0.0}),
    _c("fvg_33", "ATR1.0+time_exit@40bars+min1R",
       {"type":"atr","mult":1.0},
       {"type":"time_exit","max_bars":40,"min_r":1.0}),
    _c("fvg_34", "ATR1.5+time_exit@60bars",
       {"type":"atr","mult":1.5},
       {"type":"time_exit","max_bars":60,"min_r":0.0}),
    _c("fvg_35", "hybrid_tight(ATR1.0+str3)+partial50@2R->3.0R+BE@1R",
       {"type":"hybrid_tight","atr_mult":1.0,"struct_bars":3},
       {"type":"partial_fixed","r1":2.0,"frac":0.5,"r2":3.0}, _BE1),
]


# ============================================================
#  PHASE RUNNERS
# ============================================================

def phase_bbs(all_data: dict, cache: dict, results: dict) -> None:
    print("\n" + "="*70, flush=True)
    print("  PHASE 1 -- BBS Position Management (%d combos)" % len(BBS_COMBOS), flush=True)
    print("="*70, flush=True)
    bbs_res = []
    for k, combo in enumerate(BBS_COMBOS):
        r = run_sltp(combo, "bbs", all_data, cache)
        r["score"] = _score(r)
        bbs_res.append(r)
        print("  [%02d/%02d] %-45s n=%3d PF=%.2f PnL=%+.1f%% MFEcap=%4.1f%% DD=%.1f%% score=%.3f" % (
            k+1, len(BBS_COMBOS),
            combo["id"]+" "+combo["desc"][:28],
            r["trades"], r["profit_factor"], r["pnl_pct"],
            r["mfe_capture_pct"], r["max_dd_pct"], r["score"]), flush=True)
    bbs_res.sort(key=lambda x: x["score"], reverse=True)
    results["bbs"] = bbs_res
    print("\n  Top 5 BBS:", flush=True)
    for i, r in enumerate(bbs_res[:5]):
        print("    %d. %-35s PF=%.3f MFEcap=%.1f%% gb=%.1f%% PnL=%+.1f%% DD=%.1f%%" % (
            i+1, r["combo_id"]+" "+r["desc"][:20],
            r["profit_factor"], r["mfe_capture_pct"],
            r["giveback_pct"], r["pnl_pct"], r["max_dd_pct"]), flush=True)


def phase_fvg(all_data: dict, cache: dict, results: dict) -> None:
    print("\n" + "="*70, flush=True)
    print("  PHASE 2 -- FVG Position Management (%d combos)" % len(FVG_COMBOS), flush=True)
    print("="*70, flush=True)
    fvg_res = []
    for k, combo in enumerate(FVG_COMBOS):
        r = run_sltp(combo, "fvg", all_data, cache)
        r["score"] = _score(r)
        fvg_res.append(r)
        print("  [%02d/%02d] %-45s n=%3d PF=%.2f PnL=%+.1f%% MFEcap=%4.1f%% DD=%.1f%% score=%.3f" % (
            k+1, len(FVG_COMBOS),
            combo["id"]+" "+combo["desc"][:28],
            r["trades"], r["profit_factor"], r["pnl_pct"],
            r["mfe_capture_pct"], r["max_dd_pct"], r["score"]), flush=True)
    fvg_res.sort(key=lambda x: x["score"], reverse=True)
    results["fvg"] = fvg_res
    print("\n  Top 5 FVG:", flush=True)
    for i, r in enumerate(fvg_res[:5]):
        print("    %d. %-35s PF=%.3f MFEcap=%.1f%% gb=%.1f%% PnL=%+.1f%% DD=%.1f%%" % (
            i+1, r["combo_id"]+" "+r["desc"][:20],
            r["profit_factor"], r["mfe_capture_pct"],
            r["giveback_pct"], r["pnl_pct"], r["max_dd_pct"]), flush=True)


def phase_walkforward(all_data: dict, cache: dict, results: dict) -> None:
    print("\n" + "="*70, flush=True)
    print("  PHASE 3 -- Walk-Forward (67/33) top-3 BBS + top-3 FVG", flush=True)
    print("="*70, flush=True)
    wf = {}
    for agent_name, key, combos in [("bbs","bbs",BBS_COMBOS), ("fvg","fvg",FVG_COMBOS)]:
        top3 = results.get(key, [])[:3]
        for rf in top3:
            cid   = rf["combo_id"]
            combo = next((c for c in combos if c["id"] == cid), None)
            if combo is None: continue
            tr = run_sltp(combo, agent_name, all_data, cache, 0.0, 0.67)
            te = run_sltp(combo, agent_name, all_data, cache, 0.67, 1.0)
            delta = te["pnl_pct"] - tr["pnl_pct"]
            degrad = (delta / tr["pnl_pct"] * 100) if tr["pnl_pct"] != 0 else 0
            wf[cid] = {
                "agent": agent_name, "combo_id": cid, "desc": rf["desc"],
                "train_pnl": tr["pnl_pct"], "test_pnl": te["pnl_pct"],
                "train_pf":  tr["profit_factor"], "test_pf": te["profit_factor"],
                "delta_pct": round(delta, 2), "degradation": round(degrad, 1),
            }
            ok = te["profit_factor"] >= 1.0 and te["pnl_pct"] > -5
            print("  %-20s train=%+6.1f%% test=%+6.1f%% delta=%+6.1f%% degrad=%+5.0f%% %s" % (
                cid, tr["pnl_pct"], te["pnl_pct"], delta, degrad,
                "[OK]" if ok else "[FAIL]"), flush=True)
    results["walkforward"] = wf


def print_report(results: dict) -> None:
    print("\n" + "="*70, flush=True)
    print("  FINAL REPORT -- Position Management Optimization v5", flush=True)
    print("="*70, flush=True)
    for key, label, base_id in [("bbs","BBS","bbs_01"), ("fvg","FVG","fvg_01")]:
        rlist = results.get(key, [])
        if not rlist: continue
        bl   = next((r for r in rlist if r["combo_id"] == base_id), None)
        top5 = rlist[:5]
        print("\n  -- %s (baseline: %s) --" % (label, base_id), flush=True)
        if bl:
            print("  Baseline PF=%.3f PnL=%+.1f%% MFEcap=%.1f%% gb=%.1f%% DD=%.1f%%" % (
                bl["profit_factor"], bl["pnl_pct"],
                bl["mfe_capture_pct"], bl["giveback_pct"], bl["max_dd_pct"]), flush=True)
        print("  %-5s %-50s %5s %7s %6s %7s %6s %5s" % (
            "Rank","Config","n","PnL%","PF","MFEcap","gb%","DD%"), flush=True)
        print("  " + "-"*90, flush=True)
        for i, r in enumerate(top5):
            mark = " [BEST]" if i == 0 else ""
            print("  %-5d %-50s %5d %+6.1f%% %6.3f %6.1f%% %5.1f%% %4.1f%%%s" % (
                i+1, r["combo_id"]+" "+r["desc"][:30],
                r["trades"], r["pnl_pct"], r["profit_factor"],
                r["mfe_capture_pct"], r["giveback_pct"], r["max_dd_pct"],
                mark), flush=True)
        if bl and top5:
            best = top5[0]
            print("\n  Improvement vs baseline:", flush=True)
            print("    PF:      %+.3f" % (best["profit_factor"]  - bl["profit_factor"]), flush=True)
            print("    PnL:     %+.1f%%" % (best["pnl_pct"]       - bl["pnl_pct"]), flush=True)
            print("    MFEcap:  %+.1f%%" % (best["mfe_capture_pct"]- bl["mfe_capture_pct"]), flush=True)
            print("    Giveback:%+.1f%%" % (best["giveback_pct"]  - bl["giveback_pct"]), flush=True)
            print("    MaxDD:   %+.1f%%" % (best["max_dd_pct"]    - bl["max_dd_pct"]), flush=True)


def save_results(results: dict) -> Path:
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / ("backtest_v5_%s.json" % ts)
    slim = {}
    for k, v in results.items():
        if isinstance(v, list):
            slim[k] = [{kk: vv for kk, vv in r.items()
                        if kk != "completed_trades"} for r in v]
        else:
            slim[k] = v
    with open(out, "w", encoding="utf-8") as f:
        json.dump(slim, f, indent=2, default=str)
    print("\n  Results saved: %s" % out, flush=True)
    return out


# ============================================================
#  MAIN
# ============================================================

def main():
    ts0 = datetime.now()
    print("backtest_v5.py -- Position Management Optimization", flush=True)
    print("Started: %s" % ts0.strftime("%Y-%m-%d %H:%M:%S"), flush=True)

    days_tag = "180d" if _check_180d_available() else "90d"
    print("Data: %s" % days_tag, flush=True)

    all_data = preload_all(days_tag=days_tag, blacklist=BLACKLIST)
    print("Symbols loaded: %d" % len(all_data), flush=True)
    if not all_data:
        print("ERROR: no data -- run fetch_data.py first", flush=True)
        return

    print("\n" + "="*70, flush=True)
    print("  PHASE 0 -- Pre-computing signals & regime (runs once)", flush=True)
    print("  BBS: O(n)  FVG: O(n^2) slow -- but only here, not in combos", flush=True)
    print("="*70, flush=True)
    t0 = datetime.now()
    cache = precompute_all(all_data)
    t1 = datetime.now()
    total_bbs = sum(len(v["bbs"]) for v in cache.values())
    total_fvg = sum(len(v["fvg"]) for v in cache.values())
    print("Pre-compute done in %.0f sec. BBS signals: %d  FVG signals: %d" % (
        (t1-t0).total_seconds(), total_bbs, total_fvg), flush=True)

    results = {
        "meta": {
            "started": ts0.isoformat(),
            "data_tag": days_tag,
            "symbols": len(all_data),
            "blacklist": list(BLACKLIST),
            "total_bbs_signals": total_bbs,
            "total_fvg_signals": total_fvg,
        }
    }

    phase_bbs(all_data, cache, results)
    phase_fvg(all_data, cache, results)
    phase_walkforward(all_data, cache, results)
    print_report(results)
    save_results(results)

    elapsed = (datetime.now() - ts0).total_seconds()
    print("\nTotal runtime: %.0f min %.0f sec" % divmod(elapsed, 60), flush=True)


if __name__ == "__main__":
    main()
