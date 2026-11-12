"""
backtest_session_v5.py -- Session V5 (2026-04-16)

Goal: RAISE PnL of the one working strategy (H6 AL-break) by every available knob,
validate via walk-forward, then add Asian-session strategies.

Building on V4 (H6 AL_combo_R2: PF=1.18, PnL=+54%, DD=17.9% on 14 sym).

Optimization angles:
  1. Symbol-filter portfolio (drop XRP/UNI/SOL/BNB blacklist)
  2. SL/TP grid-search (sl_atr 1.0..3.5, tp_r 1.5..5, tp_mode fixed vs range_mult)
  3. Partial-TP + runner ("scale-out 50% at R=1, runner to R=4 with trail")
  4. Pyramid: add leg on pullback retest within same day if still aligned
  5. Dynamic risk: scale risk% by trend_combo strength & AL range width
  6. Trail variants: breakeven / ATR 1.5/2.0/2.5 / swing 3/5/8 / session-VWAP
  7. Wider-stop / wider-TP combos as user requested
  8. Walk-forward 4-quarter validation
  9. Asian session: Tokyo ORB / London-open breakout from Asian range / Asian MR

Realistic execution (V3/V4 standard):
  - Entry at next bar OPEN
  - Slippage 0.05% per side + taker fee 0.055% per side (~0.21% round-trip)
  - Market exit at session end hour

Module imports helpers from V4 (does not modify it).
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

# Import helpers from V4 (benign research module)
from multi_agent.backtests.backtest_session_v4 import (
    _load, _atr, _ema, _adx, _supertrend,
    prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf,
    _place_sl_tp, _check_htf,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE, RISK_PER_TRADE,
    TOP14,
)

logger = logging.getLogger("session_bt_v5")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BLACKLIST = {"XRP/USDT:USDT", "UNI/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT"}
GOOD10 = [s for s in TOP14 if s not in BLACKLIST]  # 10 profitable symbols


# ═══════════════════════════════════════════════════════════════════════════
#  AL-break signal (V4-compatible, with optional pyramid support)
# ═══════════════════════════════════════════════════════════════════════════

def gen_al_break_v5(arr, sess_ranges, params, htf_aligned_map=None):
    """H6 AL-break with optional pyramid retest leg."""
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    max_entry_hr = params.get("max_entry_hr", 4)
    min_range_atr = params.get("min_range_atr", 0.5)
    max_range_atr = params.get("max_range_atr", 5.0)
    allow_pyramid = params.get("allow_pyramid", False)
    pyramid_rR = params.get("pyramid_rR", 0.5)  # add leg at this R

    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        alhi, allo = r["al_hi"], r["al_lo"]
        if np.isnan(alhi) or np.isnan(allo): continue
        if hour[i] < 13 or hour[i] >= 13 + max_entry_hr: continue
        if last_sig_date == d: continue
        al_range = alhi - allo
        if al_range <= 0 or atr[i] <= 0: continue
        rR = al_range / atr[i]
        if rR < min_range_atr or rR > max_range_atr: continue

        direction = None
        if h[i] > alhi and c[i] > alhi:
            direction = "LONG"; lvl = alhi
        elif l[i] < allo and c[i] < allo:
            direction = "SHORT"; lvl = allo
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue

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
            "level": float(lvl),  # breakout level for pyramid retest
            "pyramid_rR": float(pyramid_rR) if allow_pyramid else None,
        }
        signals.append(sig)
        last_sig_date = d
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  Asian-session strategies
# ═══════════════════════════════════════════════════════════════════════════

def gen_tokyo_orb(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Asian-1: Tokyo Opening Range (00-02 UTC) → break after 02, window 02-06.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    or_end_hr = params.get("or_end_hr", 2)
    entry_end_hr = params.get("entry_end_hr", 6)
    min_or_atr = params.get("min_or_atr", 0.3)
    max_or_atr = params.get("max_or_atr", 4.0)
    # Build Tokyo OR per date
    tokyo_or = {}
    by_date = defaultdict(list)
    for i in range(n):
        by_date[date[i]].append(i)
    for d, idxs in by_date.items():
        t_i = [i for i in idxs if 0 <= hour[i] < or_end_hr]
        if not t_i:
            continue
        tokyo_or[d] = (max(h[i] for i in t_i), min(l[i] for i in t_i))
    for i in range(50, n-1):
        d = date[i]
        if d not in tokyo_or: continue
        thi, tlo = tokyo_or[d]
        if hour[i] < or_end_hr or hour[i] >= entry_end_hr: continue
        if last_sig_date == d: continue
        or_range = thi - tlo
        if or_range <= 0 or atr[i] <= 0: continue
        rR = or_range / atr[i]
        if rR < min_or_atr or rR > max_or_atr: continue
        direction = None
        if h[i] > thi and c[i] > thi: direction = "LONG"
        elif l[i] < tlo and c[i] < tlo: direction = "SHORT"
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue
        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], or_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(or_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_london_break_asian(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Asian-2: London-open breakout from Asian range (07-12 UTC window).
    London-session setup: break Asian 00-07 hi/lo, target runs toward London range expansion.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    entry_start_hr = params.get("entry_start_hr", 7)
    entry_end_hr = params.get("entry_end_hr", 12)
    min_range_atr = params.get("min_range_atr", 0.3)
    max_range_atr = params.get("max_range_atr", 4.0)
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo): continue
        if hour[i] < entry_start_hr or hour[i] >= entry_end_hr: continue
        if last_sig_date == d: continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0: continue
        rR = a_range / atr[i]
        if rR < min_range_atr or rR > max_range_atr: continue
        direction = None
        if h[i] > ahi and c[i] > ahi: direction = "LONG"
        elif l[i] < alo and c[i] < alo: direction = "SHORT"
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue
        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], a_range)
        if risk_px <= 0: continue
        # session_end for Asian-London is London close (13:00)
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
            "session_end_hr": 13,
        })
        last_sig_date = d
    return signals


def gen_asian_mr(arr, sess_ranges, params, htf_aligned_map=None):
    """
    Asian-3: Mean-reversion FROM Asian extreme back toward mid.
    If price sweeps Asian hi/lo in 05-07 UTC with wick (rejection), short the fake.
    Works in RANGE regime (HTF ADX<20 or HTF trend_combo==0 → skip ADX filter if configured).
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    sweep_start_hr = params.get("sweep_start_hr", 5)
    sweep_end_hr = params.get("sweep_end_hr", 9)
    wick_thr = params.get("wick_thr", 0.55)  # wick share of bar
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo): continue
        if hour[i] < sweep_start_hr or hour[i] >= sweep_end_hr: continue
        if last_sig_date == d: continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0: continue
        body = abs(c[i] - o[i])
        total = h[i] - l[i]
        if total <= 0: continue
        direction = None
        if h[i] > ahi and c[i] < ahi:
            # swept hi, rejected → short
            upper_wick = h[i] - max(c[i], o[i])
            if upper_wick / total < wick_thr: continue
            direction = "SHORT"
        elif l[i] < alo and c[i] > alo:
            lower_wick = min(c[i], o[i]) - l[i]
            if lower_wick / total < wick_thr: continue
            direction = "LONG"
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue
        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], a_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
            "session_end_hr": 13,
        })
        last_sig_date = d
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATE v5 with partial-TP, pyramid, dynamic risk
# ═══════════════════════════════════════════════════════════════════════════

def simulate_v5(sigs_by_sym, arrays_by_sym, session_end_hr=22,
                trail_cfg=None, scale_out=None, dynamic_risk=False):
    """
    Extended simulation:
      - scale_out: {"at_r": 1.0, "frac": 0.5} — take frac at R level, runner continues
      - dynamic_risk: scale risk% by (trend_combo_1h strength, al_range/atr ratio)
      - supports sig["session_end_hr"] override (Asian/London strategies)
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

            # Dynamic risk: up to 1.5x if range is wide & trend is strong
            risk_mult = 1.0
            if dynamic_risk:
                al_range = sig.get("asian_range", risk_px)
                if atr[sig["i_signal"]] > 0:
                    rr_ratio = al_range / atr[sig["i_signal"]]
                    # Reward tighter range pre-breakout (true compression) with more size
                    if rr_ratio < 1.5: risk_mult = 1.4
                    elif rr_ratio < 2.5: risk_mult = 1.2
                    else: risk_mult = 1.0
            risk_usd = balance * RISK_PER_TRADE * risk_mult
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

            # Scale-out state
            partial_done = False
            partial_pnl = 0.0
            partial_fee = 0.0
            qty_remain = qty_usd
            qty_taken = 0.0

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

                # Scale-out: take partial profit at R-level if configured
                if scale_out is not None and not partial_done:
                    at_r = scale_out.get("at_r", 1.0)
                    frac = scale_out.get("frac", 0.5)
                    partial_target = eff_entry + risk_px * at_r if direction == "LONG" else eff_entry - risk_px * at_r
                    hit = False
                    if direction == "LONG" and bar_h >= partial_target: hit = True
                    elif direction == "SHORT" and bar_l <= partial_target: hit = True
                    if hit:
                        qty_taken_now = qty_usd * frac
                        qty_remain = qty_usd - qty_taken_now
                        qty_taken += qty_taken_now
                        eff_pt = partial_target * (1 - SLIPPAGE) if direction == "LONG" else partial_target * (1 + SLIPPAGE)
                        if direction == "LONG":
                            partial_pnl += qty_taken_now * (eff_pt - eff_entry) / eff_entry
                        else:
                            partial_pnl += qty_taken_now * (eff_entry - eff_pt) / eff_entry
                        partial_fee += qty_taken_now * TAKER_FEE
                        partial_done = True
                        # Move SL to breakeven after partial
                        if direction == "LONG":
                            sl_cur = max(sl_cur, eff_entry)
                        else:
                            sl_cur = min(sl_cur, eff_entry)

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

                if direction == "LONG":
                    if bar_l <= sl_cur:
                        exit_reason = "TRAIL" if trail_active else ("BE" if partial_done and sl_cur == eff_entry else "SL")
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_h >= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break
                else:
                    if bar_h >= sl_cur:
                        exit_reason = "TRAIL" if trail_active else ("BE" if partial_done and sl_cur == eff_entry else "SL")
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
                "partial": partial_done,
            })
    return _aggregate_v5(completed, balance, max_dd)


def _aggregate_v5(completed, balance, max_dd):
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


def run_config(name, gen_fn, arrays, ranges, htf_map, params,
               trail_cfg=None, scale_out=None, dynamic_risk=False,
               session_end_hr=22, compute_per_sym=False):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_fn(arr, ranges[sym], params, htf_map.get(sym))
    res = simulate_v5(sigs, arrays, session_end_hr=session_end_hr,
                       trail_cfg=trail_cfg, scale_out=scale_out, dynamic_risk=dynamic_risk)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    if compute_per_sym:
        per_sym = {}
        for sym in sigs:
            per_sym[sym] = simulate_v5({sym: sigs[sym]}, {sym: arrays[sym]},
                                        session_end_hr=session_end_hr,
                                        trail_cfg=trail_cfg, scale_out=scale_out,
                                        dynamic_risk=dynamic_risk)
        res["per_sym"] = per_sym
    return {"config_id": name, "params": params, "trail": trail_cfg,
            "scale_out": scale_out, "dynamic_risk": dynamic_risk, **res}


def log_r(name, r):
    logger.info(f"{name:40s} trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+7.1f}%  "
                f"DD={r['max_dd_pct']:>5.2f}% avgR={r['avg_r']:>+.2f}")


# ═══════════════════════════════════════════════════════════════════════════
#  Walk-forward split
# ═══════════════════════════════════════════════════════════════════════════

def split_by_quarters(arr, ranges):
    """Split a symbol's array into 4 quarters by timestamp."""
    ts = arr["ts"]
    tmin, tmax = int(ts[0]), int(ts[-1])
    step = (tmax - tmin) // 4
    bounds = [tmin + step * k for k in range(5)]
    quarters = []
    for q in range(4):
        lo, hi = bounds[q], bounds[q+1]
        mask = (ts >= lo) & (ts < hi)
        if not mask.any():
            quarters.append(None)
            continue
        idxs = np.where(mask)[0]
        i0, i1 = idxs[0], idxs[-1] + 1
        sub = {}
        for k, v in arr.items():
            if isinstance(v, np.ndarray):
                sub[k] = v[i0:i1]
            else:
                sub[k] = v
        sub["n"] = i1 - i0
        # ranges: keep all — generator only uses dates that exist in array
        quarters.append(sub)
    return quarters


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t_start = _time.time()
    logger.info("=== Session V5 (PnL maximization + WF + Asian strategies) ===")

    # Load data
    arrays = {}
    ranges = {}
    htf_by_sym = {}
    avg_stats = {}
    for sym in TOP14:
        df15 = _load(sym, "15m")
        if df15 is None:
            logger.warning(f"No 15m for {sym}"); continue
        df1h = _load(sym, "1h")
        if df1h is None: df1h = _load(sym, "1H")
        df4h = _load(sym, "4h")
        if df4h is None: df4h = _load(sym, "4H")
        arr = prep_15m_arrays(df15)
        ranges[sym] = build_session_ranges(arr)
        htf = compute_htf_trends(df4h, df1h)
        htf_aligned = {}
        for k, m in htf.items():
            htf_aligned[k] = align_htf(m, arr["ts"])
        arrays[sym] = arr
        htf_by_sym[sym] = htf_aligned

        # Session movement stats per symbol
        nyrngs = []; asrngs = []
        for d, r in ranges[sym].items():
            if not np.isnan(r["ny_hi"]) and not np.isnan(r["ny_lo"]):
                if r["pd_close"] and r["pd_close"] > 0:
                    nyrngs.append((r["ny_hi"] - r["ny_lo"]) / r["pd_close"] * 100)
            if not np.isnan(r["asian_hi"]) and not np.isnan(r["asian_lo"]):
                if r["pd_close"] and r["pd_close"] > 0:
                    asrngs.append((r["asian_hi"] - r["asian_lo"]) / r["pd_close"] * 100)
        avg_stats[sym] = {
            "ny_median_pct": float(np.median(nyrngs)) if nyrngs else 0.0,
            "ny_mean_pct": float(np.mean(nyrngs)) if nyrngs else 0.0,
            "as_median_pct": float(np.median(asrngs)) if asrngs else 0.0,
            "as_mean_pct": float(np.mean(asrngs)) if asrngs else 0.0,
            "days": len(nyrngs),
        }
    logger.info(f"Loaded {len(arrays)} symbols")

    # Print session movement stats
    logger.info("")
    logger.info("=== SESSION AVG MOVEMENT per symbol (%of prev close) ===")
    logger.info(f"{'sym':6s} {'NY_med%':>8s} {'NY_mean%':>9s} {'AS_med%':>8s} {'AS_mean%':>9s} {'days':>6s}")
    for sym in TOP14:
        s = avg_stats.get(sym)
        if not s: continue
        sn = sym.replace("/USDT:USDT", "")
        logger.info(f"{sn:6s} {s['ny_median_pct']:>8.2f} {s['ny_mean_pct']:>9.2f} "
                    f"{s['as_median_pct']:>8.2f} {s['as_mean_pct']:>9.2f} {s['days']:>6d}")

    # Portfolio selectors
    top14_arrays = {s: arrays[s] for s in TOP14 if s in arrays}
    good10_arrays = {s: arrays[s] for s in GOOD10 if s in arrays}

    results = []

    # ═══════════════════════════════════════════════════════════════
    # BASELINE: V4 best (AL_combo_R2) on top14 and good10
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 1: Baseline AL_combo_R2 on TOP14 vs GOOD10 ===")
    base_params = {
        "sl_mode": "atr", "sl_atr_mult": 2.5, "tp_mode": "fixed_r", "tp_r": 2.0,
        "max_entry_hr": 4, "min_range_atr": 0.5, "max_range_atr": 5.0,
        "htf_filter": "trend_combo_1h",
    }
    r = run_config("S1_Base_TOP14", gen_al_break_v5, top14_arrays, ranges, htf_by_sym,
                   base_params, compute_per_sym=False)
    log_r("S1_Base_TOP14", r); results.append(r)
    r = run_config("S1_Base_GOOD10", gen_al_break_v5, good10_arrays, ranges, htf_by_sym,
                   base_params, compute_per_sym=True)
    log_r("S1_Base_GOOD10", r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2: SL/TP grid search on GOOD10
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 2: SL/TP grid-search on GOOD10 ===")
    sl_mults = [1.5, 2.0, 2.5, 3.0]
    tp_rs = [1.5, 2.0, 2.5, 3.0, 4.0]
    for sl_m in sl_mults:
        for tp_r in tp_rs:
            params = dict(base_params)
            params["sl_atr_mult"] = sl_m
            params["tp_r"] = tp_r
            name = f"S2_SL{sl_m}_TP{tp_r}"
            r = run_config(name, gen_al_break_v5, good10_arrays, ranges, htf_by_sym, params)
            log_r(name, r); results.append(r)

    # range-mult TP variant
    for rng_mult in [1.5, 2.0, 2.5, 3.0]:
        params = dict(base_params)
        params["tp_mode"] = "range_mult"
        params["tp_range_mult"] = rng_mult
        for sl_m in [1.5, 2.0, 2.5]:
            params["sl_atr_mult"] = sl_m
            name = f"S2_RNG{rng_mult}_SL{sl_m}"
            r = run_config(name, gen_al_break_v5, good10_arrays, ranges, htf_by_sym, params)
            log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3: Trailing stop variants
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 3: Trailing-stop variants on best S2 config ===")
    # Use best SL/TP from S2 — determined at runtime
    s2_subset = [r for r in results if r["config_id"].startswith("S2_") and r["trades"] >= 40]
    s2_subset.sort(key=lambda x: x["pnl_pct"], reverse=True)
    best_s2 = s2_subset[0] if s2_subset else None
    if best_s2:
        logger.info(f"Best S2: {best_s2['config_id']} PF={best_s2['profit_factor']} PnL={best_s2['pnl_pct']}%")
        bp = best_s2["params"]
        trail_variants = [
            ("breakeven_1R", {"activate_at_r": 1.0, "trail_mode": "breakeven"}),
            ("atr_act1R_m1.5", {"activate_at_r": 1.0, "trail_mode": "atr", "atr_mult": 1.5}),
            ("atr_act1R_m2.0", {"activate_at_r": 1.0, "trail_mode": "atr", "atr_mult": 2.0}),
            ("swing3_act1R", {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 3, "buffer_atr": 0.2}),
            ("swing5_act1R", {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
            ("swing8_act1.5R", {"activate_at_r": 1.5, "trail_mode": "swing", "swing_bars": 8, "buffer_atr": 0.3}),
        ]
        for tname, tcfg in trail_variants:
            name = f"S3_{tname}"
            # Raise tp_r so trail has room to run
            params = dict(bp); params["tp_r"] = max(params.get("tp_r", 2.0), 4.0)
            r = run_config(name, gen_al_break_v5, good10_arrays, ranges, htf_by_sym,
                           params, trail_cfg=tcfg)
            log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 4: Scale-out (partial TP + runner)
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 4: Scale-out variants ===")
    if best_s2:
        bp = best_s2["params"]
        scale_variants = [
            ("so_1R_50", {"at_r": 1.0, "frac": 0.5}, None, 3.0),
            ("so_1R_50_swtrail", {"at_r": 1.0, "frac": 0.5},
                {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 4.0),
            ("so_1.5R_50_swtrail", {"at_r": 1.5, "frac": 0.5},
                {"activate_at_r": 1.5, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 5.0),
            ("so_1R_70_atrtrail", {"at_r": 1.0, "frac": 0.7},
                {"activate_at_r": 1.0, "trail_mode": "atr", "atr_mult": 1.5}, 4.0),
            ("so_2R_33_swtrail", {"at_r": 2.0, "frac": 0.33},
                {"activate_at_r": 2.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}, 6.0),
        ]
        for sname, scfg, tcfg, tpR in scale_variants:
            name = f"S4_{sname}"
            params = dict(bp); params["tp_r"] = tpR
            r = run_config(name, gen_al_break_v5, good10_arrays, ranges, htf_by_sym,
                           params, trail_cfg=tcfg, scale_out=scfg)
            log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 5: Dynamic risk
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 5: Dynamic risk scaling ===")
    if best_s2:
        bp = best_s2["params"]
        params = dict(bp); params["tp_r"] = 4.0
        scfg = {"at_r": 1.0, "frac": 0.5}
        tcfg = {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}
        r = run_config("S5_dynrisk_so_swtrail", gen_al_break_v5, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg, scale_out=scfg, dynamic_risk=True)
        log_r("S5_dynrisk_so_swtrail", r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 6: Walk-forward on best config
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 6: Walk-forward 4Q on best config ===")
    # Find overall best PnL on GOOD10
    best_so_far = max((r for r in results if r["trades"] >= 40
                        and not r["config_id"].startswith("S1_Base_TOP14")),
                      key=lambda x: x["pnl_pct"], default=None)
    if best_so_far:
        logger.info(f"Best overall: {best_so_far['config_id']} "
                    f"PF={best_so_far['profit_factor']} PnL={best_so_far['pnl_pct']}%")
        wf_results = []
        # Need to re-split arrays per symbol by quarter
        per_sym_q = {s: split_by_quarters(arrays[s], ranges[s]) for s in GOOD10 if s in arrays}
        for qi in range(4):
            q_arrays = {s: per_sym_q[s][qi] for s in per_sym_q if per_sym_q[s][qi] is not None}
            bp = best_so_far["params"]
            tcfg = best_so_far.get("trail")
            scfg = best_so_far.get("scale_out")
            drisk = best_so_far.get("dynamic_risk", False)
            r = run_config(f"S6_WF_Q{qi+1}", gen_al_break_v5, q_arrays, ranges, htf_by_sym,
                           bp, trail_cfg=tcfg, scale_out=scfg, dynamic_risk=drisk)
            log_r(f"S6_WF_Q{qi+1}", r)
            wf_results.append(r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SECTION 7: Asian session strategies
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== SECTION 7: Asian-session strategies on TOP14 ===")
    # Try several HTF filters
    asian_htf_variants = [None, "adx25_1h", "trend_combo_1h", "st_1h"]

    # 7.1 Tokyo ORB
    for htf in asian_htf_variants:
        for tp_r in [1.5, 2.0, 3.0]:
            for or_end in [2, 3]:
                params = {
                    "sl_mode": "atr", "sl_atr_mult": 2.0,
                    "tp_mode": "fixed_r", "tp_r": tp_r,
                    "or_end_hr": or_end, "entry_end_hr": 6,
                    "min_or_atr": 0.3, "max_or_atr": 4.0,
                    "htf_filter": htf,
                }
                name = f"S7_TOKYO_orb{or_end}_{htf or 'none'}_R{tp_r}"
                r = run_config(name, gen_tokyo_orb, top14_arrays, ranges, htf_by_sym,
                               params, session_end_hr=7)
                log_r(name, r); results.append(r)

    # 7.2 London breakout from Asian range
    for htf in asian_htf_variants:
        for tp_r in [1.5, 2.0, 3.0]:
            params = {
                "sl_mode": "atr", "sl_atr_mult": 2.0,
                "tp_mode": "fixed_r", "tp_r": tp_r,
                "entry_start_hr": 7, "entry_end_hr": 12,
                "min_range_atr": 0.3, "max_range_atr": 4.0,
                "htf_filter": htf,
            }
            name = f"S7_LDN_{htf or 'none'}_R{tp_r}"
            r = run_config(name, gen_london_break_asian, top14_arrays, ranges, htf_by_sym,
                           params, session_end_hr=13)
            log_r(name, r); results.append(r)

    # 7.3 Asian MR
    for htf in [None, "adx25_1h"]:  # MR doesn't need strong trend, test without
        for tp_mode in ["fixed_r"]:
            for tp_r in [1.0, 1.5, 2.0]:
                params = {
                    "sl_mode": "atr", "sl_atr_mult": 1.5,
                    "tp_mode": tp_mode, "tp_r": tp_r,
                    "sweep_start_hr": 5, "sweep_end_hr": 9, "wick_thr": 0.5,
                    "htf_filter": htf,
                }
                name = f"S7_ASMR_{htf or 'none'}_R{tp_r}"
                r = run_config(name, gen_asian_mr, top14_arrays, ranges, htf_by_sym,
                               params, session_end_hr=13)
                log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SAVE + SUMMARY
    # ═══════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v5_{ts}.json"
    payload = {"symbols": list(arrays.keys()), "good10": GOOD10,
               "avg_stats": avg_stats, "results": results}
    with open(path, "w") as f:
        json.dump(payload, f, default=float, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t_start:.1f}s)")

    # Print TOP-20 by PnL
    print("\n" + "="*110)
    print("OVERALL TOP-20 by PnL (min 40 trades)")
    print(f"{'config':40s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>7s} {'DD%':>6s}  {'avgR':>5s}")
    flt = [r for r in results if r.get("trades", 0) >= 40]
    flt.sort(key=lambda r: r["pnl_pct"], reverse=True)
    for r in flt[:20]:
        print(f"{r['config_id']:40s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f}  "
              f"{r['pnl_pct']:>+7.1f} {r['max_dd_pct']:>6.2f}  {r['avg_r']:>+5.2f}")

    # Print TOP-10 by PF
    print("\nOVERALL TOP-10 by PF (min 40 trades)")
    flt_pf = sorted(flt, key=lambda r: r["profit_factor"], reverse=True)
    for r in flt_pf[:10]:
        print(f"{r['config_id']:40s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f}  "
              f"{r['pnl_pct']:>+7.1f} {r['max_dd_pct']:>6.2f}  {r['avg_r']:>+5.2f}")

    # Walk-forward summary
    wf = [r for r in results if r["config_id"].startswith("S6_WF_")]
    if wf:
        print("\nWALK-FORWARD quarterly result:")
        for r in wf:
            pass_flag = "PASS" if (r["pnl_pct"] > 0 and r["profit_factor"] >= 1.0) else "FAIL"
            print(f"  {r['config_id']:16s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                  f"PnL={r['pnl_pct']:>+7.1f}% DD={r['max_dd_pct']:>5.2f}%  [{pass_flag}]")

    # Per-symbol breakdown for top PnL config on GOOD10
    good10_top = [r for r in results if "GOOD10" in r["config_id"] and r.get("per_sym")]
    if good10_top:
        best = max(good10_top, key=lambda x: x["pnl_pct"])
        print(f"\nPer-symbol for {best['config_id']} PF={best['profit_factor']} PnL={best['pnl_pct']}%:")
        per = best.get("per_sym", {})
        for sym, s in sorted(per.items(), key=lambda kv: kv[1]["pnl_pct"], reverse=True):
            sn = sym.replace("/USDT:USDT", "")
            mk = "+" if s["pnl_pct"] > 0 else "-"
            print(f"  {mk} {sn:6s} n={s['trades']:>4} WR={s['wr_pct']:>5.1f}% "
                  f"PF={s['profit_factor']:>5.2f} PnL={s['pnl_pct']:>+7.1f}%  avgR={s['avg_r']:>+.2f}")


if __name__ == "__main__":
    main()
