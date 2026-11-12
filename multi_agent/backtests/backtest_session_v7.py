"""
backtest_session_v7.py -- Session V7 (2026-04-16)

V6 FINAL champion: F1_exWed = SL2.0/TP4R + exclude Wednesday
  PF=1.73, PnL=+220.4%, DD=9.18%, 306 trades/year

V7 goal: squeeze more PnL / lower DD via:
  1. BTC-regime filter: skip days when BTC ATR% is too low (chop) or extreme (panic)
  2. Correlation-diversification: when many symbols signal same direction on same day,
     scale per-trade risk DOWN (we're over-exposed to one macro move)
  3. Calendar filter: first-Friday-of-month (NFP) skip, day-after-Wed (post-FOMC wake) keep
  4. NY VWAP distance filter at entry (skip when price is > N*ATR from VWAP = stretched)
  5. BTC 1h trend alignment: require BTC 1h ADX/EMA trend in entry direction (macro confirm)
  6. Compound: try 1.5% risk-per-trade (base was 1%)

Base stack: V6 champion (SL2.0 TP4R, exclude Wed, GOOD10 symbols).

Note: "News calendar" — we don't have live FOMC/CPI dates in data. Using proxies:
  - BTC vol regime (captures news-day vol expansion)
  - First Friday of month (NFP)
  - Skip day-before-FOMC? (Wed already excluded so this is redundant)
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
    _load, _atr, _ema, _adx,
    prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf,
    _place_sl_tp, _check_htf,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE, RISK_PER_TRADE,
    TOP14,
)
from multi_agent.backtests.backtest_session_v5 import GOOD10, _aggregate_v5
from multi_agent.backtests.backtest_session_v6 import (
    BASE_PARAMS_V5_CHAMPION, split_by_quarters, _detect_consolidation,
)

logger = logging.getLogger("session_bt_v7")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  BTC regime map (built once, shared across symbols)
# ═══════════════════════════════════════════════════════════════════════════

def build_btc_regime(arrays):
    """
    Returns per-date regime dict: date → {atr_pct, regime, bias}
      regime: 'chop' (low vol), 'normal', 'panic' (high vol)
      bias: -1/0/1 based on BTC 1h EMA50 + ADX
    """
    if "BTC/USDT:USDT" not in arrays:
        return {}
    arr = arrays["BTC/USDT:USDT"]
    c = arr["c"]; h = arr["h"]; l = arr["l"]; atr = arr["atr"]
    dates = arr["date"]
    # Daily ATR% = rolling 24h range / close, on end-of-day bars
    by_date = defaultdict(list)
    for i in range(len(c)):
        by_date[dates[i]].append(i)
    regime_by_date = {}
    daily_vols = []
    dates_sorted = sorted(by_date.keys())
    for d in dates_sorted:
        idxs = by_date[d]
        if not idxs: continue
        day_hi = max(h[i] for i in idxs)
        day_lo = min(l[i] for i in idxs)
        day_cl = c[idxs[-1]]
        if day_cl <= 0: continue
        vol_pct = (day_hi - day_lo) / day_cl * 100
        daily_vols.append((d, vol_pct))
    if not daily_vols: return {}
    vols = np.array([v for _, v in daily_vols])
    p25 = float(np.percentile(vols, 25))
    p75 = float(np.percentile(vols, 75))
    p90 = float(np.percentile(vols, 90))
    logger.info(f"BTC daily vol%: p25={p25:.2f} p75={p75:.2f} p90={p90:.2f} "
                f"median={float(np.median(vols)):.2f}")
    for d, vp in daily_vols:
        if vp < p25: regime = "chop"
        elif vp > p90: regime = "panic"
        elif vp > p75: regime = "high"
        else: regime = "normal"
        regime_by_date[d] = {"vol_pct": vp, "regime": regime}
    return regime_by_date


def count_signals_per_day(sigs_by_sym):
    """Return dict: date → {direction: count}. Used for correlation-dedupe."""
    counts = defaultdict(lambda: {"LONG": 0, "SHORT": 0})
    for sym, sigs in sigs_by_sym.items():
        for s in sigs:
            counts[s["date"]][s["direction"]] += 1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════════
#  V7 signal generator (inherits V6 + regime/calendar filters)
# ═══════════════════════════════════════════════════════════════════════════

def gen_al_break_v7(arr, sess_ranges, params, htf_aligned_map=None,
                    btc_regime=None):
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    dow = arr["dow"]
    atr = arr["atr"]
    ny_vwap = arr["ny_vwap"]
    n = arr["n"]
    ts = arr["ts"]

    max_entry_hr = params.get("max_entry_hr", 4)
    min_range_atr = params.get("min_range_atr", 0.5)
    max_range_atr = params.get("max_range_atr", 5.0)
    allowed_dows = params.get("allowed_dows", {0, 1, 3, 4, 5, 6})  # default: exWed
    skip_first_friday = params.get("skip_first_friday", False)
    skip_regimes = params.get("skip_regimes", set())
    max_vwap_dist_atr = params.get("max_vwap_dist_atr", 0.0)  # 0 = off

    signals = []
    sigs_by_date: dict = {}

    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        if int(dow[i]) not in allowed_dows: continue

        # First-Friday (NFP): dow==4, day 1-7 of month
        if skip_first_friday and int(dow[i]) == 4:
            try:
                day_num = int(d.split("-")[2])
                if day_num <= 7: continue
            except Exception:
                pass

        # BTC regime filter
        if btc_regime and skip_regimes:
            reg = btc_regime.get(d, {}).get("regime")
            if reg in skip_regimes: continue

        r = sess_ranges[d]
        alhi, allo = r["al_hi"], r["al_lo"]
        if np.isnan(alhi) or np.isnan(allo): continue
        if hour[i] < 13 or hour[i] >= 13 + max_entry_hr: continue
        al_range = alhi - allo
        if al_range <= 0 or atr[i] <= 0: continue
        rR = al_range / atr[i]
        if rR < min_range_atr or rR > max_range_atr: continue

        direction = None; lvl = None
        if h[i] > alhi and c[i] > alhi:
            direction = "LONG"; lvl = alhi
        elif l[i] < allo and c[i] < allo:
            direction = "SHORT"; lvl = allo
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        # VWAP-stretch filter: skip if price already too far from NY VWAP
        if max_vwap_dist_atr > 0 and ny_vwap[i] > 0:
            dist = abs(c[i] - ny_vwap[i]) / atr[i] if atr[i] > 0 else 999
            if dist > max_vwap_dist_atr: continue

        already = sigs_by_date.get(d)
        if already: continue

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
        }
        signals.append(sig)
        sigs_by_date[d] = {"direction": direction, "level": float(lvl)}
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATE v7 — adds correlation-diversification
# ═══════════════════════════════════════════════════════════════════════════

def simulate_v7(sigs_by_sym, arrays_by_sym, session_end_hr=22,
                trail_cfg=None, risk_per_trade=RISK_PER_TRADE,
                corr_dedupe=False, corr_threshold=4, corr_scale=0.5):
    """
    corr_dedupe: if N+ symbols signal same direction on same day,
                 scale each position's risk by corr_scale.
    """
    balance = START_BALANCE
    completed = []
    peak = balance
    max_dd = 0.0

    # Pre-compute same-day same-direction counts
    day_counts = count_signals_per_day(sigs_by_sym) if corr_dedupe else {}

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

            risk_mult = 1.0
            if corr_dedupe:
                cnt = day_counts.get(entry_date, {}).get(direction, 0)
                if cnt >= corr_threshold:
                    risk_mult = corr_scale
            risk_usd = balance * risk_per_trade * risk_mult
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

            for j in range(i_entry, n):
                if date[j] != entry_date:
                    exit_reason = "EOS"; exit_px = c[j-1]; exit_bar = j - 1; break
                if hour[j] >= sess_end:
                    exit_reason = "EOS"; exit_px = c[j]; exit_bar = j; break
                bar_h = h[j]; bar_l = l[j]

                if direction == "LONG":
                    mfe = max(mfe, (bar_h - eff_entry) / risk_px)
                    mae = max(mae, (eff_entry - bar_l) / risk_px)
                else:
                    mfe = max(mfe, (eff_entry - bar_l) / risk_px)
                    mae = max(mae, (bar_h - eff_entry) / risk_px)

                if trail_cfg is not None:
                    act_r = trail_cfg.get("activate_at_r", 1.0)
                    if not trail_active and mfe >= act_r: trail_active = True
                    if trail_active:
                        mode = trail_cfg.get("trail_mode", "swing")
                        if mode == "swing":
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
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_h >= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break
                else:
                    if bar_h >= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_l <= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break

            eff_exit = exit_px * (1 - SLIPPAGE) if direction == "LONG" else exit_px * (1 + SLIPPAGE)
            exit_fee = qty_usd * TAKER_FEE
            if direction == "LONG":
                pnl_usd = qty_usd * (eff_exit - eff_entry) / eff_entry - entry_fee - exit_fee
            else:
                pnl_usd = qty_usd * (eff_entry - eff_exit) / eff_entry - entry_fee - exit_fee
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
                "entry": entry_ref, "exit": exit_px, "sl": sl_init, "tp": tp,
                "date": entry_date, "risk_mult": risk_mult,
            })
    return _aggregate_v5(completed, balance, max_dd)


def run_config(name, arrays, ranges, htf_map, params, btc_regime=None,
               trail_cfg=None, risk_per_trade=RISK_PER_TRADE,
               corr_dedupe=False, corr_threshold=4, corr_scale=0.5,
               session_end_hr=22, compute_per_sym=False):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_al_break_v7(arr, ranges[sym], params, htf_map.get(sym),
                                      btc_regime=btc_regime)
    res = simulate_v7(sigs, arrays, session_end_hr=session_end_hr,
                      trail_cfg=trail_cfg, risk_per_trade=risk_per_trade,
                      corr_dedupe=corr_dedupe, corr_threshold=corr_threshold,
                      corr_scale=corr_scale)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    if compute_per_sym:
        per_sym = {}
        for sym in sigs:
            per_sym[sym] = simulate_v7({sym: sigs[sym]}, {sym: arrays[sym]},
                                        session_end_hr=session_end_hr,
                                        trail_cfg=trail_cfg, risk_per_trade=risk_per_trade,
                                        corr_dedupe=corr_dedupe,
                                        corr_threshold=corr_threshold,
                                        corr_scale=corr_scale)
        res["per_sym"] = per_sym
    return {"config_id": name, "params": params, "trail": trail_cfg,
            "risk_per_trade": risk_per_trade,
            "corr_dedupe": corr_dedupe, "corr_threshold": corr_threshold,
            "corr_scale": corr_scale, **res}


def log_r(name, r):
    logger.info(f"{name:44s} trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+8.1f}%  "
                f"DD={r['max_dd_pct']:>5.2f}% avgR={r['avg_r']:>+.2f}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    logger.info("=== V7: BTC regime + correlation + calendar filters ===")

    arrays, ranges, htf_by_sym = {}, {}, {}
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
    btc_regime = build_btc_regime(arrays)
    logger.info(f"BTC regime map built for {len(btc_regime)} days")

    # Breakdown of regime days
    by_reg = defaultdict(int)
    for d, v in btc_regime.items(): by_reg[v["regime"]] += 1
    logger.info(f"Regime counts: {dict(by_reg)}")

    results = []

    # V6 baseline replay
    logger.info("")
    logger.info("=== V0: V6 champion replay (baseline) ===")
    base = dict(BASE_PARAMS_V5_CHAMPION)
    base["allowed_dows"] = {0, 1, 3, 4, 5, 6}  # exWed
    r = run_config("V0_V6_champion", good10_arrays, ranges, htf_by_sym, base)
    log_r("V0_V6_champion", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section R: BTC regime filter
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section R: BTC regime filters ===")
    regime_tests = [
        ("R_skip_chop",      {"chop"}),
        ("R_skip_panic",     {"panic"}),
        ("R_skip_chop_panic",{"chop", "panic"}),
        ("R_skip_high_panic",{"high", "panic"}),
        ("R_skip_chop_high", {"chop", "high"}),
        ("R_only_normal",    {"chop", "high", "panic"}),  # keep normal only
    ]
    for name, skip in regime_tests:
        p = dict(base); p["skip_regimes"] = skip
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p, btc_regime=btc_regime)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section C: Correlation diversification
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section C: Correlation-diversification (scale risk when aligned) ===")
    for thr in [3, 4, 5, 6]:
        for scale in [0.3, 0.5, 0.7]:
            p = dict(base)
            r = run_config(f"C_thr{thr}_scl{scale}", good10_arrays, ranges, htf_by_sym, p,
                           corr_dedupe=True, corr_threshold=thr, corr_scale=scale)
            log_r(f"C_thr{thr}_scl{scale}", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section N: NFP / first-Friday skip
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section N: First-Friday (NFP) skip ===")
    p = dict(base); p["skip_first_friday"] = True
    r = run_config("N_skip_nfp_friday", good10_arrays, ranges, htf_by_sym, p)
    log_r("N_skip_nfp_friday", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section V: NY VWAP-stretch filter
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section V: VWAP-stretch filter (don't chase far from VWAP) ===")
    for dist in [1.0, 1.5, 2.0, 3.0]:
        p = dict(base); p["max_vwap_dist_atr"] = dist
        name = f"V_maxdist{dist}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section K: Risk-per-trade scaling (compound harder)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section K: Risk-per-trade scaling ===")
    for rpt in [0.005, 0.01, 0.015, 0.02]:
        p = dict(base)
        name = f"K_rpt{rpt}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p, risk_per_trade=rpt)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section U: ULTIMATE stacking — take best of each section
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section U: ULTIMATE stacking ===")

    def best_in(pfx, min_tr=50):
        cs = [r for r in results if r["config_id"].startswith(pfx) and r["trades"] >= min_tr]
        return max(cs, key=lambda x: x["pnl_pct"], default=None)

    br = best_in("R_")
    bc = best_in("C_")
    bv = best_in("V_")
    bk = best_in("K_")

    for b, label in [(br, "regime"), (bc, "corr"), (bv, "vwap"), (bk, "risk")]:
        if b: logger.info(f"  best {label}: {b['config_id']} PnL={b['pnl_pct']}%")

    # Stack configs
    stack_configs = []

    # U1: base + best regime
    if br and "params" in br:
        skip_r = br["params"].get("skip_regimes", set())
        if skip_r:
            pp = dict(base); pp["skip_regimes"] = skip_r
            stack_configs.append(("U1_base_regime", pp, None, RISK_PER_TRADE, False, 4, 0.5))

    # U2: base + best regime + best corr
    if br and bc and "params" in br:
        skip_r = br["params"].get("skip_regimes", set())
        if skip_r:
            pp = dict(base); pp["skip_regimes"] = skip_r
            thr = bc.get("corr_threshold", 4)
            scl = bc.get("corr_scale", 0.5)
            stack_configs.append(("U2_regime_corr", pp, None, RISK_PER_TRADE, True, thr, scl))

    # U3: base + best regime + best vwap
    if br and bv and "params" in bv and "params" in br:
        skip_r = br["params"].get("skip_regimes", set())
        mvd = bv["params"].get("max_vwap_dist_atr", 0)
        pp = dict(base); pp["skip_regimes"] = skip_r; pp["max_vwap_dist_atr"] = mvd
        stack_configs.append(("U3_regime_vwap", pp, None, RISK_PER_TRADE, False, 4, 0.5))

    # U4: ALL stacked
    if br and bc and bv:
        skip_r = br["params"].get("skip_regimes", set())
        mvd = bv["params"].get("max_vwap_dist_atr", 0)
        thr = bc.get("corr_threshold", 4)
        scl = bc.get("corr_scale", 0.5)
        pp = dict(base); pp["skip_regimes"] = skip_r; pp["max_vwap_dist_atr"] = mvd
        pp["skip_first_friday"] = True
        stack_configs.append(("U4_ALL_stacked", pp, None, RISK_PER_TRADE, True, thr, scl))

    # U5: ALL + wider TP
    if br:
        skip_r = br["params"].get("skip_regimes", set())
        pp = dict(base); pp["skip_regimes"] = skip_r; pp["tp_r"] = 5.0
        stack_configs.append(("U5_regime_TP5", pp, None, RISK_PER_TRADE, False, 4, 0.5))

    # U6: ALL + tight swing-trail to lock profit
    if br:
        skip_r = br["params"].get("skip_regimes", set())
        pp = dict(base); pp["skip_regimes"] = skip_r
        tcfg = {"activate_at_r": 2.0, "trail_mode": "swing",
                "swing_bars": 8, "buffer_atr": 0.25}
        stack_configs.append(("U6_regime_swingtrail", pp, tcfg, RISK_PER_TRADE, False, 4, 0.5))

    for name, p, tcfg, rpt, cdd, cth, csc in stack_configs:
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime, trail_cfg=tcfg,
                       risk_per_trade=rpt, corr_dedupe=cdd,
                       corr_threshold=cth, corr_scale=csc,
                       compute_per_sym=(name in ("U4_ALL_stacked", "U5_regime_TP5", "U1_base_regime")))
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Section W: Walk-forward on overall best
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Section W: Walk-forward 4Q on overall best ===")
    cands = [r for r in results if r["trades"] >= 80]
    best = max(cands, key=lambda x: x["pnl_pct"], default=None)
    if best:
        logger.info(f"Overall best: {best['config_id']} PF={best['profit_factor']} "
                    f"PnL={best['pnl_pct']}% DD={best['max_dd_pct']}%")
        per_sym_q = {s: split_by_quarters(arrays[s]) for s in GOOD10 if s in arrays}
        for qi in range(4):
            q_arrays = {s: per_sym_q[s][qi] for s in per_sym_q if per_sym_q[s][qi] is not None}
            r = run_config(f"W_WF_Q{qi+1}", q_arrays, ranges, htf_by_sym,
                           best["params"], btc_regime=btc_regime,
                           trail_cfg=best.get("trail"),
                           risk_per_trade=best.get("risk_per_trade", RISK_PER_TRADE),
                           corr_dedupe=best.get("corr_dedupe", False),
                           corr_threshold=best.get("corr_threshold", 4),
                           corr_scale=best.get("corr_scale", 0.5))
            log_r(f"W_WF_Q{qi+1}", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # SAVE + SUMMARY
    # ═════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v7_{ts}.json"
    def _json_default(x):
        if isinstance(x, set): return sorted(x)
        try: return float(x)
        except Exception: return str(x)
    with open(path, "w") as f:
        json.dump({"results": results, "btc_regime_counts": dict(by_reg)},
                  f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t0:.1f}s)")

    print("\n" + "="*120)
    print("TOP-25 by PnL (V7, min 40 trades)")
    print(f"{'config':44s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s}  {'avgR':>5s}")
    flt = [r for r in results if r.get("trades", 0) >= 40]
    flt.sort(key=lambda r: r["pnl_pct"], reverse=True)
    for r in flt[:25]:
        print(f"{r['config_id']:44s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f}  "
              f"{r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f}  {r['avg_r']:>+5.2f}")

    # Section bests
    print("\n" + "-"*120)
    print("Best per section")
    for pfx, title in [("V0_", "V6 baseline"), ("R_", "BTC regime"),
                        ("C_", "Correlation"), ("N_", "NFP skip"),
                        ("V_", "VWAP stretch"), ("K_", "Risk/trade"),
                        ("U", "Stacked combos")]:
        cs = [r for r in results if r["config_id"].startswith(pfx) and r["trades"] >= 40]
        if not cs: continue
        b = max(cs, key=lambda x: x["pnl_pct"])
        print(f"  {title:18s} {b['config_id']:36s} PF={b['profit_factor']:>5.2f} "
              f"PnL={b['pnl_pct']:>+8.1f}% DD={b['max_dd_pct']:>5.2f}% trades={b['trades']}")

    # WF
    wf = [r for r in results if r["config_id"].startswith("W_WF_")]
    if wf:
        print("\nWALK-FORWARD:")
        for r in wf:
            pf_ok = r["profit_factor"] >= 1.0
            pnl_ok = r["pnl_pct"] > 0
            tag = "PASS" if (pf_ok and pnl_ok) else "FAIL"
            print(f"  {r['config_id']:16s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                  f"PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%  [{tag}]")

    # Per-symbol breakdowns
    ps = [r for r in results if r.get("per_sym")]
    if ps:
        ps.sort(key=lambda r: r["pnl_pct"], reverse=True)
        for r in ps[:2]:
            print(f"\nPer-symbol for {r['config_id']} "
                  f"PF={r['profit_factor']} PnL={r['pnl_pct']}% DD={r['max_dd_pct']}%:")
            for sym, s in sorted(r["per_sym"].items(), key=lambda kv: kv[1]["pnl_pct"], reverse=True):
                sn = sym.replace("/USDT:USDT", "")
                mk = "+" if s["pnl_pct"] > 0 else "-"
                print(f"  {mk} {sn:6s} n={s['trades']:>4} WR={s['wr_pct']:>5.1f}% "
                      f"PF={s['profit_factor']:>5.2f} PnL={s['pnl_pct']:>+7.1f}%  avgR={s['avg_r']:>+.2f}")


if __name__ == "__main__":
    main()
