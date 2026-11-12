"""
backtest_session_v6.py -- Session V6 (2026-04-16)

Goal: push AL-break PnL past +135% of V5 champion using:
  1. STRUCTURE-BASED TRAIL (user idea): when price has moved N*R in our direction
     AND forms a tight consolidation, move stop under the consolidation low
     (for LONG) / above its high (for SHORT). Guarantees locking in profit
     while letting the trade breathe vs a fixed trail.
  2. TP extensions: R=5, R=6, R=8 with structure-trail
  3. Wide-range-only filter: trade only when AL range >= N*ATR (skip narrow days)
  4. First-2h-NY filter: only enter in 13-15 UTC (freshest liquidity)
  5. Day-of-week analysis
  6. Direction-bias: LONG-only / SHORT-only
  7. Re-entry on retest of breakout level within the same day
  8. Walk-forward on the best new config

Base config (V5 champion): SL=2.0*ATR, TP=4R, HTF=trend_combo_1h, GOOD10 symbols.

V6 imports helpers from V4 and AL-break generator from V5 without modifying them.
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

from multi_agent.backtests.backtest_session_v4 import (
    _load, _atr, _ema, _adx, _supertrend,
    prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf,
    _place_sl_tp, _check_htf,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE, RISK_PER_TRADE,
    TOP14,
)
from multi_agent.backtests.backtest_session_v5 import (
    GOOD10, BLACKLIST, _aggregate_v5,
)

logger = logging.getLogger("session_bt_v6")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  AL-break V6 signal generator (adds filters: min_range_atr, first-2h,
#  direction-bias, dow-filter, allow_retest_reentry)
# ═══════════════════════════════════════════════════════════════════════════

def gen_al_break_v6(arr, sess_ranges, params, htf_aligned_map=None):
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    dow = arr["dow"]
    atr = arr["atr"]
    n = arr["n"]

    max_entry_hr = params.get("max_entry_hr", 4)   # hours after NY open
    min_range_atr = params.get("min_range_atr", 0.5)
    max_range_atr = params.get("max_range_atr", 5.0)
    direction_filter = params.get("direction_filter")  # None, "LONG", "SHORT"
    allowed_dows = params.get("allowed_dows")  # None or set of ints 0-6
    allow_retest = params.get("allow_retest_reentry", False)
    retest_tol = params.get("retest_tol_atr", 0.15)

    signals = []
    # Track per-day: first signal index & direction (for retest re-entry)
    sigs_by_date: dict = {}

    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        if allowed_dows is not None and int(dow[i]) not in allowed_dows: continue
        r = sess_ranges[d]
        alhi, allo = r["al_hi"], r["al_lo"]
        if np.isnan(alhi) or np.isnan(allo): continue
        if hour[i] < 13 or hour[i] >= 13 + max_entry_hr: continue
        al_range = alhi - allo
        if al_range <= 0 or atr[i] <= 0: continue
        rR = al_range / atr[i]
        if rR < min_range_atr or rR > max_range_atr: continue

        direction = None
        lvl = None
        if h[i] > alhi and c[i] > alhi:
            direction = "LONG"; lvl = alhi
        elif l[i] < allo and c[i] < allo:
            direction = "SHORT"; lvl = allo
        if direction is None: continue
        if direction_filter and direction != direction_filter: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        # First signal of day
        already = sigs_by_date.get(d)
        is_primary = already is None
        is_retest = False
        if not is_primary:
            # Only allow retest re-entry if enabled AND same direction AND
            # price has pulled back to breakout level (within retest_tol ATR)
            if not allow_retest: continue
            if already["direction"] != direction: continue
            lvl_prev = already["level"]
            tol = retest_tol * atr[i]
            if direction == "LONG":
                if l[i] > lvl_prev + tol: continue  # didn't pull back enough
                if c[i] < lvl_prev: continue  # must close above level again
            else:
                if h[i] < lvl_prev - tol: continue
                if c[i] > lvl_prev: continue
            is_retest = True

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], al_range)
        if risk_px <= 0: continue

        sig = {
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(al_range), "date": d,
            "level": float(lvl),
            "is_retest": is_retest,
        }
        signals.append(sig)
        if is_primary:
            sigs_by_date[d] = {"direction": direction, "level": float(lvl), "i_entry": entry_bar}
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATE v6 — adds STRUCTURE-BASED TRAIL
# ═══════════════════════════════════════════════════════════════════════════

def _detect_consolidation(arr, j, i_entry, direction,
                          lookback=6, range_atr_max=0.9, min_bars_since_entry=3):
    """
    Detect a tight consolidation on last `lookback` bars.
    Returns (ok, cons_hi, cons_lo).

    Conditions:
      - Need ≥ min_bars_since_entry bars since entry (let trade develop)
      - last `lookback` bar range / ATR <= range_atr_max
      - For LONG: consolidation must be ABOVE entry_ref (already in profit)
      - For SHORT: consolidation must be BELOW entry_ref
    """
    if j - i_entry < min_bars_since_entry:
        return False, 0.0, 0.0
    k = min(lookback, j - i_entry + 1)
    if k < 3:
        return False, 0.0, 0.0
    h_slice = arr["h"][j-k+1:j+1]
    l_slice = arr["l"][j-k+1:j+1]
    cons_hi = float(np.max(h_slice))
    cons_lo = float(np.min(l_slice))
    atr_i = arr["atr"][j]
    if atr_i <= 0:
        return False, 0.0, 0.0
    rng = cons_hi - cons_lo
    if rng / atr_i > range_atr_max:
        return False, cons_hi, cons_lo
    return True, cons_hi, cons_lo


def simulate_v6(sigs_by_sym, arrays_by_sym, session_end_hr=22,
                trail_cfg=None, scale_out=None):
    """
    V6 simulate. trail_cfg supports mode="structure":
      - activate_at_r: MFE threshold to start looking for structure
      - lookback: bars to inspect for consolidation
      - range_atr_max: max range/ATR ratio to accept as consolidation
      - buffer_atr: extra buffer below low (LONG) / above high (SHORT)
      - min_bars_since_entry: min bars trade must have been open
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
            sess_end = sig.get("session_end_hr", session_end_hr)

            risk_usd = balance * RISK_PER_TRADE
            if risk_px <= 0 or entry_ref <= 0: continue
            qty_usd = risk_usd / (risk_px / entry_ref)
            if qty_usd > balance * MAX_LEVERAGE: qty_usd = balance * MAX_LEVERAGE
            eff_entry = entry_ref * (1 + SLIPPAGE) if direction == "LONG" else entry_ref * (1 - SLIPPAGE)
            entry_fee = qty_usd * TAKER_FEE

            sl_cur = sl_init
            trail_active = False
            mfe = 0.0; mae = 0.0
            exit_reason = "EOS"
            exit_px = c[n-1]; exit_bar = n - 1

            partial_done = False
            partial_pnl = 0.0
            partial_fee = 0.0
            qty_remain = qty_usd

            # Structure-trail state
            struct_locked = False  # have we identified a structure at least once?

            for j in range(i_entry, n):
                if date[j] != entry_date:
                    exit_reason = "EOS"
                    exit_px = c[j-1]; exit_bar = j - 1
                    break
                if hour[j] >= sess_end:
                    exit_reason = "EOS"
                    exit_px = c[j]; exit_bar = j
                    break
                bar_h = h[j]; bar_l = l[j]

                if direction == "LONG":
                    mfe = max(mfe, (bar_h - eff_entry) / risk_px)
                    mae = max(mae, (eff_entry - bar_l) / risk_px)
                else:
                    mfe = max(mfe, (eff_entry - bar_l) / risk_px)
                    mae = max(mae, (bar_h - eff_entry) / risk_px)

                # Scale-out (kept available)
                if scale_out is not None and not partial_done:
                    at_r = scale_out.get("at_r", 1.0)
                    frac = scale_out.get("frac", 0.5)
                    partial_target = eff_entry + risk_px * at_r if direction == "LONG" else eff_entry - risk_px * at_r
                    hit = (bar_h >= partial_target) if direction == "LONG" else (bar_l <= partial_target)
                    if hit:
                        qty_taken_now = qty_usd * frac
                        qty_remain = qty_usd - qty_taken_now
                        eff_pt = partial_target * (1 - SLIPPAGE) if direction == "LONG" else partial_target * (1 + SLIPPAGE)
                        if direction == "LONG":
                            partial_pnl += qty_taken_now * (eff_pt - eff_entry) / eff_entry
                        else:
                            partial_pnl += qty_taken_now * (eff_entry - eff_pt) / eff_entry
                        partial_fee += qty_taken_now * TAKER_FEE
                        partial_done = True
                        if direction == "LONG":
                            sl_cur = max(sl_cur, eff_entry)
                        else:
                            sl_cur = min(sl_cur, eff_entry)

                # Trailing stop
                if trail_cfg is not None:
                    act_r = trail_cfg.get("activate_at_r", 1.0)
                    if not trail_active and mfe >= act_r: trail_active = True
                    if trail_active:
                        mode = trail_cfg.get("trail_mode", "swing")
                        if mode == "breakeven":
                            if direction == "LONG": sl_cur = max(sl_cur, eff_entry)
                            else: sl_cur = min(sl_cur, eff_entry)
                        elif mode == "atr":
                            mlt = trail_cfg.get("atr_mult", 2.0)
                            if direction == "LONG": sl_cur = max(sl_cur, c[j] - atr[j] * mlt)
                            else: sl_cur = min(sl_cur, c[j] + atr[j] * mlt)
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
                        elif mode == "structure":
                            # --- USER'S IDEA ---
                            # Once price moved in our direction AND consolidated,
                            # put SL under/above the consolidation structure.
                            lb = trail_cfg.get("lookback", 6)
                            ratmax = trail_cfg.get("range_atr_max", 0.9)
                            buf = trail_cfg.get("buffer_atr", 0.15)
                            minbars = trail_cfg.get("min_bars_since_entry", 3)
                            ok, cons_hi, cons_lo = _detect_consolidation(
                                arr, j, i_entry, direction,
                                lookback=lb, range_atr_max=ratmax,
                                min_bars_since_entry=minbars,
                            )
                            if ok:
                                if direction == "LONG":
                                    new_sl = cons_lo - atr[j] * buf
                                    # Only move up (never down) and never below
                                    # breakeven once we've locked structure once
                                    if struct_locked:
                                        new_sl = max(new_sl, eff_entry)
                                    sl_cur = max(sl_cur, new_sl)
                                else:
                                    new_sl = cons_hi + atr[j] * buf
                                    if struct_locked:
                                        new_sl = min(new_sl, eff_entry)
                                    sl_cur = min(sl_cur, new_sl)
                                struct_locked = True
                            else:
                                # Fallback swing while waiting for consolidation
                                fb = trail_cfg.get("fallback_swing", 5)
                                fb_buf = trail_cfg.get("fallback_buffer_atr", 0.2)
                                kk = min(fb, j - i_entry)
                                if kk >= 1:
                                    if direction == "LONG":
                                        new_sl = float(np.min(l[j-kk:j+1])) - atr[j] * fb_buf
                                        sl_cur = max(sl_cur, new_sl)
                                    else:
                                        new_sl = float(np.max(h[j-kk:j+1])) + atr[j] * fb_buf
                                        sl_cur = min(sl_cur, new_sl)

                # Exit checks
                if direction == "LONG":
                    if bar_l <= sl_cur:
                        if struct_locked: exit_reason = "STRUCT_TRAIL"
                        elif trail_active: exit_reason = "TRAIL"
                        elif partial_done and sl_cur == eff_entry: exit_reason = "BE"
                        else: exit_reason = "SL"
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_h >= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break
                else:
                    if bar_h >= sl_cur:
                        if struct_locked: exit_reason = "STRUCT_TRAIL"
                        elif trail_active: exit_reason = "TRAIL"
                        elif partial_done and sl_cur == eff_entry: exit_reason = "BE"
                        else: exit_reason = "SL"
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_l <= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break

            eff_exit = exit_px * (1 - SLIPPAGE) if direction == "LONG" else exit_px * (1 + SLIPPAGE)
            exit_fee = qty_remain * TAKER_FEE
            if direction == "LONG":
                main_pnl = qty_remain * (eff_exit - eff_entry) / eff_entry
            else:
                main_pnl = qty_remain * (eff_entry - eff_exit) / eff_entry
            pnl_usd = main_pnl + partial_pnl - entry_fee - exit_fee - partial_fee
            balance += pnl_usd
            peak = max(peak, balance)
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            exit_r = ((eff_exit - eff_entry) if direction == "LONG" else (eff_entry - eff_exit)) / risk_px
            completed.append({
                "sym": sym, "direction": direction,
                "i_entry": i_entry, "exit_bar": exit_bar,
                "bars": exit_bar - i_entry,
                "reason": exit_reason, "pnl_usd": pnl_usd, "won": pnl_usd > 0,
                "exit_r": exit_r, "mfe": mfe, "mae": mae,
                "entry": entry_ref, "exit": exit_px, "sl": sl_init, "tp": tp, "date": entry_date,
                "struct_locked": struct_locked,
            })
    return _aggregate_v5(completed, balance, max_dd)


def run_config(name, gen_fn, arrays, ranges, htf_map, params,
               trail_cfg=None, scale_out=None,
               session_end_hr=22, compute_per_sym=False):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_fn(arr, ranges[sym], params, htf_map.get(sym))
    res = simulate_v6(sigs, arrays, session_end_hr=session_end_hr,
                      trail_cfg=trail_cfg, scale_out=scale_out)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    if compute_per_sym:
        per_sym = {}
        for sym in sigs:
            per_sym[sym] = simulate_v6({sym: sigs[sym]}, {sym: arrays[sym]},
                                        session_end_hr=session_end_hr,
                                        trail_cfg=trail_cfg, scale_out=scale_out)
        res["per_sym"] = per_sym
    return {"config_id": name, "params": params, "trail": trail_cfg,
            "scale_out": scale_out, **res}


def log_r(name, r):
    logger.info(f"{name:44s} trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+7.1f}%  "
                f"DD={r['max_dd_pct']:>5.2f}% avgR={r['avg_r']:>+.2f}")


def split_by_quarters(arr):
    ts = arr["ts"]
    tmin, tmax = int(ts[0]), int(ts[-1])
    step = (tmax - tmin) // 4
    bounds = [tmin + step * k for k in range(5)]
    quarters = []
    for q in range(4):
        lo, hi = bounds[q], bounds[q+1]
        mask = (ts >= lo) & (ts < hi)
        if not mask.any():
            quarters.append(None); continue
        idxs = np.where(mask)[0]
        i0, i1 = idxs[0], idxs[-1] + 1
        sub = {}
        for k, v in arr.items():
            if isinstance(v, np.ndarray):
                sub[k] = v[i0:i1]
            else:
                sub[k] = v
        sub["n"] = i1 - i0
        quarters.append(sub)
    return quarters


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

BASE_PARAMS_V5_CHAMPION = {
    "sl_mode": "atr", "sl_atr_mult": 2.0,
    "tp_mode": "fixed_r", "tp_r": 4.0,
    "max_entry_hr": 4, "min_range_atr": 0.5, "max_range_atr": 5.0,
    "htf_filter": "trend_combo_1h",
}


def main():
    t_start = _time.time()
    logger.info("=== Session V6 (structure-trail + PnL push) ===")

    arrays = {}
    ranges = {}
    htf_by_sym = {}
    for sym in TOP14:
        df15 = _load(sym, "15m")
        if df15 is None: continue
        df1h = _load(sym, "1h")
        if df1h is None: df1h = _load(sym, "1H")
        df4h = _load(sym, "4h")
        if df4h is None: df4h = _load(sym, "4H")
        arr = prep_15m_arrays(df15)
        ranges[sym] = build_session_ranges(arr)
        htf = compute_htf_trends(df4h, df1h)
        htf_aligned = {k: align_htf(m, arr["ts"]) for k, m in htf.items()}
        arrays[sym] = arr
        htf_by_sym[sym] = htf_aligned
    logger.info(f"Loaded {len(arrays)} symbols")

    good10_arrays = {s: arrays[s] for s in GOOD10 if s in arrays}
    results = []

    # ═══════════════════════════════════════════════════════════════
    # SECTION 0: V5 champion baseline replay (sanity check)
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 0: V5 champion replay (baseline) ===")
    r = run_config("S0_V5_champion", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                   BASE_PARAMS_V5_CHAMPION)
    log_r("S0_V5_champion", r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1: Structure-based trail (USER'S IDEA)
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 1: STRUCTURE-trail variants (new idea) ===")
    # Use wide TP so structure-trail has room
    struct_variants = [
        # (name, activate_r, lookback, range_atr_max, buffer, min_bars, tp_r)
        ("struct_a1_lb6_rmx0.9_b0.15",      1.0, 6,  0.9, 0.15, 3, 5.0),
        ("struct_a1_lb8_rmx0.8_b0.2",       1.0, 8,  0.8, 0.20, 3, 5.0),
        ("struct_a1.5_lb6_rmx1.0_b0.2",     1.5, 6,  1.0, 0.20, 3, 6.0),
        ("struct_a1.5_lb8_rmx0.7_b0.25",    1.5, 8,  0.7, 0.25, 4, 6.0),
        ("struct_a2_lb6_rmx0.8_b0.2",       2.0, 6,  0.8, 0.20, 4, 8.0),
        ("struct_a2_lb10_rmx0.6_b0.3",      2.0, 10, 0.6, 0.30, 5, 8.0),
        ("struct_a1_lb4_rmx1.0_b0.1",       1.0, 4,  1.0, 0.10, 2, 5.0),
    ]
    for name, act, lb, rmx, buf, mbars, tp_r in struct_variants:
        params = dict(BASE_PARAMS_V5_CHAMPION); params["tp_r"] = tp_r
        tcfg = {
            "activate_at_r": act, "trail_mode": "structure",
            "lookback": lb, "range_atr_max": rmx,
            "buffer_atr": buf, "min_bars_since_entry": mbars,
            "fallback_swing": 5, "fallback_buffer_atr": 0.2,
        }
        r = run_config(f"S1_{name}", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg)
        log_r(f"S1_{name}", r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: TP extensions (5R / 6R / 8R) without trail
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 2: TP extensions (no trail, fixed TP) ===")
    for tp_r in [5.0, 6.0, 8.0, 10.0]:
        for sl_m in [1.5, 2.0, 2.5]:
            params = dict(BASE_PARAMS_V5_CHAMPION)
            params["tp_r"] = tp_r; params["sl_atr_mult"] = sl_m
            name = f"S2_SL{sl_m}_TP{tp_r}"
            r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
            log_r(name, r); results.append(r)

    # TP extensions WITH best structure-trail found
    # (plugged in dynamically after S1 finishes)
    s1_best = max([r for r in results if r["config_id"].startswith("S1_") and r["trades"] >= 40],
                  key=lambda x: x["pnl_pct"], default=None)
    if s1_best:
        logger.info(f"Best S1 struct: {s1_best['config_id']} PnL={s1_best['pnl_pct']}% "
                    f"PF={s1_best['profit_factor']}")
        best_tcfg = s1_best["trail"]
        logger.info("")
        logger.info("=== SECTION 2b: Extreme TP with best structure-trail ===")
        for tp_r in [5.0, 6.0, 8.0, 10.0, 15.0]:
            params = dict(BASE_PARAMS_V5_CHAMPION); params["tp_r"] = tp_r
            name = f"S2b_TP{tp_r}_struct"
            r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                           params, trail_cfg=best_tcfg)
            log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Wide-range-only filter
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 3: Wide-range filter (only high-conviction days) ===")
    for min_rr in [0.8, 1.0, 1.2, 1.5]:
        for tp_r in [3.0, 4.0, 5.0]:
            params = dict(BASE_PARAMS_V5_CHAMPION)
            params["min_range_atr"] = min_rr; params["tp_r"] = tp_r
            name = f"S3_minRR{min_rr}_TP{tp_r}"
            r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
            log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: First 2h NY only
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 4: First 2h NY only (13-15 UTC) ===")
    for tp_r in [3.0, 4.0, 5.0]:
        for max_hr in [2, 3]:
            params = dict(BASE_PARAMS_V5_CHAMPION)
            params["max_entry_hr"] = max_hr; params["tp_r"] = tp_r
            name = f"S4_maxhr{max_hr}_TP{tp_r}"
            r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
            log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Day-of-week analysis (test each day separately)
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 5: Day-of-week performance ===")
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for d_i, d_name in enumerate(dow_names):
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["allowed_dows"] = {d_i}
        name = f"S5_DOW_{d_name}"
        r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
        log_r(name, r); results.append(r)

    # Best combined days (weekdays vs weekend)
    for dows, dname in [
        ({0, 1, 2, 3, 4}, "MonFri"),   # weekdays
        ({0, 1, 2, 3}, "MonThu"),       # M-Th
        ({1, 2, 3}, "TueThu"),          # mid-week
        ({5, 6}, "SatSun"),             # weekend
    ]:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["allowed_dows"] = dows
        name = f"S5_DOW_{dname}"
        r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
        log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: LONG-only vs SHORT-only
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 6: LONG-only vs SHORT-only ===")
    for d in ["LONG", "SHORT"]:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["direction_filter"] = d
        name = f"S6_{d}_only"
        r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
        log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: Retest re-entry
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 7: Allow retest re-entry within same day ===")
    for tol in [0.1, 0.2, 0.3]:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["allow_retest_reentry"] = True
        params["retest_tol_atr"] = tol
        name = f"S7_retest_tol{tol}"
        r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
        log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 8: ULTIMATE stacking — combine best elements from each section
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 8: ULTIMATE combo — stack best knobs ===")

    # Auto-pick best from each earlier section
    def best_in(prefix, min_trades=40):
        cs = [r for r in results if r["config_id"].startswith(prefix) and r["trades"] >= min_trades]
        return max(cs, key=lambda x: x["pnl_pct"], default=None)

    best_s1 = best_in("S1_")
    best_s3 = best_in("S3_")
    best_s4 = best_in("S4_")
    best_s5_dows = best_in("S5_DOW_")
    best_s6 = best_in("S6_")

    if best_s5_dows and best_s5_dows["config_id"].startswith("S5_DOW_") and "params" in best_s5_dows:
        best_dows = best_s5_dows["params"].get("allowed_dows")
    else:
        best_dows = None

    # Combo A: wide TP + struct-trail + min_range filter
    if best_s1 and best_s3:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["min_range_atr"] = best_s3["params"].get("min_range_atr", 0.5)
        params["tp_r"] = 6.0
        tcfg = best_s1["trail"]
        r = run_config("S8_A_struct_minRR", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg)
        log_r("S8_A_struct_minRR", r); results.append(r)

    # Combo B: struct-trail + first 2h filter
    if best_s1:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["max_entry_hr"] = 2
        params["tp_r"] = 6.0
        tcfg = best_s1["trail"]
        r = run_config("S8_B_struct_first2h", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg)
        log_r("S8_B_struct_first2h", r); results.append(r)

    # Combo C: struct-trail + best DOW + min_range
    if best_s1 and best_s3 and best_dows:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["min_range_atr"] = best_s3["params"].get("min_range_atr", 0.5)
        params["allowed_dows"] = best_dows
        params["tp_r"] = 6.0
        tcfg = best_s1["trail"]
        r = run_config("S8_C_struct_minRR_DOW", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg)
        log_r("S8_C_struct_minRR_DOW", r); results.append(r)

    # Combo D: struct-trail + retest + wide TP
    if best_s1:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["allow_retest_reentry"] = True
        params["retest_tol_atr"] = 0.2
        params["tp_r"] = 6.0
        tcfg = best_s1["trail"]
        r = run_config("S8_D_struct_retest", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg, compute_per_sym=True)
        log_r("S8_D_struct_retest", r); results.append(r)

    # Combo MAX: everything stacked
    if best_s1 and best_s3:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["min_range_atr"] = best_s3["params"].get("min_range_atr", 0.5)
        params["max_entry_hr"] = 3
        params["allow_retest_reentry"] = True
        params["retest_tol_atr"] = 0.2
        params["tp_r"] = 8.0
        if best_dows and best_dows != {0, 1, 2, 3, 4, 5, 6}:
            params["allowed_dows"] = best_dows
        tcfg = best_s1["trail"]
        r = run_config("S8_MAX_all_stacked", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg, compute_per_sym=True)
        log_r("S8_MAX_all_stacked", r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 9: Walk-forward on the ULTIMATE best config
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 9: Walk-forward 4Q on overall best ===")
    # Exclude per-symbol-computed configs' duplicates by filter on trades ≥ 80
    candidates = [r for r in results if r["trades"] >= 80
                  and not r["config_id"].startswith("S0_")
                  and not r["config_id"].startswith("S5_DOW_")  # DOW subsets too sparse
                  and not r["config_id"].startswith("S6_")]     # direction-only too sparse
    overall_best = max(candidates, key=lambda x: x["pnl_pct"], default=None)
    if overall_best:
        logger.info(f"Overall best: {overall_best['config_id']} "
                    f"PF={overall_best['profit_factor']} PnL={overall_best['pnl_pct']}%")
        per_sym_q = {s: split_by_quarters(arrays[s]) for s in GOOD10 if s in arrays}
        for qi in range(4):
            q_arrays = {s: per_sym_q[s][qi] for s in per_sym_q if per_sym_q[s][qi] is not None}
            bp = overall_best["params"]
            tcfg = overall_best.get("trail")
            scfg = overall_best.get("scale_out")
            r = run_config(f"S9_WF_Q{qi+1}", gen_al_break_v6, q_arrays, ranges, htf_by_sym,
                           bp, trail_cfg=tcfg, scale_out=scfg)
            log_r(f"S9_WF_Q{qi+1}", r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SAVE + SUMMARY
    # ═══════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v6_{ts}.json"
    def _json_default(x):
        if isinstance(x, set):
            return sorted(x)
        try:
            return float(x)
        except Exception:
            return str(x)
    payload = {"symbols": list(arrays.keys()), "good10": GOOD10, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t_start:.1f}s)")

    # Print TOP-25 by PnL
    print("\n" + "="*120)
    print("OVERALL TOP-25 by PnL (min 40 trades)")
    print(f"{'config':44s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s}  {'avgR':>5s}")
    flt = [r for r in results if r.get("trades", 0) >= 40]
    flt.sort(key=lambda r: r["pnl_pct"], reverse=True)
    for r in flt[:25]:
        print(f"{r['config_id']:44s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f}  "
              f"{r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f}  {r['avg_r']:>+5.2f}")

    # Section-summary
    print("\n" + "-"*120)
    print("Best per section (PnL-wise, min 40 trades)")
    section_prefixes = [("S0_", "V5 baseline"), ("S1_", "Structure trail"),
                        ("S2_", "TP extensions"), ("S2b_", "Ext TP+struct"),
                        ("S3_", "Wide-range filter"), ("S4_", "First 2h NY"),
                        ("S5_DOW_", "Day-of-week"), ("S6_", "Direction only"),
                        ("S7_", "Retest re-entry"), ("S8_", "Stacked combos")]
    for pfx, title in section_prefixes:
        cs = [r for r in results if r["config_id"].startswith(pfx) and r["trades"] >= 40]
        if not cs: continue
        b = max(cs, key=lambda x: x["pnl_pct"])
        print(f"  {title:24s} {b['config_id']:36s} PF={b['profit_factor']:>5.2f} "
              f"PnL={b['pnl_pct']:>+7.1f}% DD={b['max_dd_pct']:>5.2f}% trades={b['trades']}")

    # Walk-forward summary
    wf = [r for r in results if r["config_id"].startswith("S9_WF_")]
    if wf:
        print("\nWALK-FORWARD quarterly:")
        for r in wf:
            pf_ok = r["profit_factor"] >= 1.0
            pnl_ok = r["pnl_pct"] > 0
            tag = "PASS" if (pf_ok and pnl_ok) else "FAIL"
            print(f"  {r['config_id']:16s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                  f"PnL={r['pnl_pct']:>+7.1f}% DD={r['max_dd_pct']:>5.2f}%  [{tag}]")

    # Per-symbol of the overall best configs
    top_with_per = [r for r in results if r.get("per_sym")]
    if top_with_per:
        top_with_per.sort(key=lambda r: r["pnl_pct"], reverse=True)
        for r in top_with_per[:2]:
            print(f"\nPer-symbol for {r['config_id']} "
                  f"PF={r['profit_factor']} PnL={r['pnl_pct']}% DD={r['max_dd_pct']}%:")
            per = r.get("per_sym", {})
            for sym, s in sorted(per.items(), key=lambda kv: kv[1]["pnl_pct"], reverse=True):
                sn = sym.replace("/USDT:USDT", "")
                mk = "+" if s["pnl_pct"] > 0 else "-"
                print(f"  {mk} {sn:6s} n={s['trades']:>4} WR={s['wr_pct']:>5.1f}% "
                      f"PF={s['profit_factor']:>5.2f} PnL={s['pnl_pct']:>+7.1f}%  avgR={s['avg_r']:>+.2f}")


if __name__ == "__main__":
    main()
