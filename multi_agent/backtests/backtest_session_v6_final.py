"""
backtest_session_v6_final.py -- V6 follow-up (2026-04-16)

Main V6 revealed DoW asymmetry:
  - Thu: PF=3.11 PnL=+62% DD=3.7%  ** best
  - Sat: PF=2.44 PnL=+39% DD=6.1%
  - Wed: PF=0.43 PnL=-26.5% DD=30.5% ** DRAG
  - Mon/Tue/Fri/Sun: mildly positive (PF 1.1-1.5)

Goal: bump V5 champion PnL (+135%) by pruning bad DoW(s),
then walk-forward to ensure the DoW pattern isn't just in-sample noise.

Tests:
  F1. Exclude Wed only
  F2. Exclude Wed + Sun
  F3. Top 3 DoW only (Thu + Sat + a third)
  F4. All except Wed (6 days)
  F5. Best F-combo on TP=5R, TP=6R (check wider-TP interaction)
  F6. WF 4Q on the winner

Plus: structure-trail re-run with LATER activation (2.5R, 3R) and wider range
allowance — the user's idea might work better if it kicks in later when the
trade is already +2.5R in profit and has more room to consolidate.
"""
from __future__ import annotations
import json, logging, sys, time as _time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from multi_agent.backtests.backtest_session_v4 import (
    _load, prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf, TOP14,
)
from multi_agent.backtests.backtest_session_v5 import GOOD10
from multi_agent.backtests.backtest_session_v6 import (
    gen_al_break_v6, simulate_v6, run_config, log_r, split_by_quarters,
    BASE_PARAMS_V5_CHAMPION,
)

logger = logging.getLogger("session_bt_v6f")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
RESULTS_DIR = Path(__file__).parent / "results"


def main():
    t0 = _time.time()
    logger.info("=== V6 FINAL: DoW exclusion + late-activation structure-trail ===")

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
    good10_arrays = {s: arrays[s] for s in GOOD10 if s in arrays}
    results = []

    # ═════════════════════════════════════════════════════════════
    # F1-F4: DoW exclusion with V5 champion (SL2.0 TP4R)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== F1-F4: DoW exclusion on V5 champion ===")

    # DoW: 0=Mon,1=Tue,2=Wed,3=Thu,4=Fri,5=Sat,6=Sun
    dow_sets = [
        ("F1_exWed",         {0, 1, 3, 4, 5, 6},     "exclude Wed"),
        ("F2_exWedSun",      {0, 1, 3, 4, 5},        "exclude Wed+Sun"),
        ("F3_exWedMon",      {1, 3, 4, 5, 6},        "exclude Wed+Mon"),
        ("F3b_exWedMonSun",  {1, 3, 4, 5},           "exclude Wed+Mon+Sun"),
        ("F4_ThuSat",        {3, 5},                 "Thu+Sat only (top 2)"),
        ("F4b_ThuSatMonTue", {0, 1, 3, 5},           "Thu+Sat+Mon+Tue"),
        ("F4c_all_good",     {0, 1, 3, 4, 5, 6},     "all except Wed"),  # same as F1
    ]
    for name, dset, desc in dow_sets:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["allowed_dows"] = dset
        r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
        log_r(f"{name} ({desc})", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # F5: Combine best DoW exclusion with wider TP
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== F5: best DoW ex + wider TP ===")
    # Need F1 (excl Wed) result to seed TP expansion
    for tp_r in [3.0, 4.0, 5.0, 6.0]:
        for sl_m in [1.5, 2.0, 2.5]:
            params = dict(BASE_PARAMS_V5_CHAMPION)
            params["allowed_dows"] = {0, 1, 3, 4, 5, 6}  # excl Wed
            params["sl_atr_mult"] = sl_m; params["tp_r"] = tp_r
            name = f"F5_exWed_SL{sl_m}_TP{tp_r}"
            r = run_config(name, gen_al_break_v6, good10_arrays, ranges, htf_by_sym, params)
            log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # F6: Late-activation structure-trail (user's idea, but activates later)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== F6: Late-activation structure-trail (activate=2.5R, 3R) ===")
    late_variants = [
        ("late_a2.5_lb8_tp6",  2.5, 8,  0.8, 0.2,  4, 6.0),
        ("late_a2.5_lb10_tp8", 2.5, 10, 0.7, 0.25, 5, 8.0),
        ("late_a3_lb8_tp8",    3.0, 8,  0.8, 0.25, 5, 8.0),
        ("late_a3_lb12_tp10",  3.0, 12, 0.6, 0.3,  6, 10.0),
        ("late_a2_lb6_tp5",    2.0, 6,  0.8, 0.2,  4, 5.0),
    ]
    for name, act, lb, rmx, buf, mbars, tp_r in late_variants:
        params = dict(BASE_PARAMS_V5_CHAMPION); params["tp_r"] = tp_r
        tcfg = {
            "activate_at_r": act, "trail_mode": "structure",
            "lookback": lb, "range_atr_max": rmx,
            "buffer_atr": buf, "min_bars_since_entry": mbars,
            "fallback_swing": 5, "fallback_buffer_atr": 0.2,
        }
        r = run_config(f"F6_{name}", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg)
        log_r(f"F6_{name}", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # F7: Late-struct-trail + DoW-exWed combo
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== F7: Late-struct + DoW-exWed combo ===")
    for name, act, lb, rmx, buf, mbars, tp_r in late_variants:
        params = dict(BASE_PARAMS_V5_CHAMPION)
        params["tp_r"] = tp_r
        params["allowed_dows"] = {0, 1, 3, 4, 5, 6}
        tcfg = {
            "activate_at_r": act, "trail_mode": "structure",
            "lookback": lb, "range_atr_max": rmx,
            "buffer_atr": buf, "min_bars_since_entry": mbars,
            "fallback_swing": 5, "fallback_buffer_atr": 0.2,
        }
        r = run_config(f"F7_exWed_{name}", gen_al_break_v6, good10_arrays, ranges, htf_by_sym,
                       params, trail_cfg=tcfg, compute_per_sym=(name == "late_a2.5_lb8_tp6"))
        log_r(f"F7_exWed_{name}", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # F8: Overall best — walk-forward 4Q
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== F8: Walk-forward 4Q on overall best ===")
    cands = [r for r in results if r["trades"] >= 60]
    best = max(cands, key=lambda x: x["pnl_pct"], default=None)
    if best:
        logger.info(f"Overall best: {best['config_id']} PF={best['profit_factor']} "
                    f"PnL={best['pnl_pct']}% DD={best['max_dd_pct']}%")
        per_sym_q = {s: split_by_quarters(arrays[s]) for s in GOOD10 if s in arrays}
        for qi in range(4):
            q_arrays = {s: per_sym_q[s][qi] for s in per_sym_q if per_sym_q[s][qi] is not None}
            r = run_config(f"F8_WF_Q{qi+1}", gen_al_break_v6, q_arrays, ranges, htf_by_sym,
                           best["params"], trail_cfg=best.get("trail"))
            log_r(f"F8_WF_Q{qi+1}", r); results.append(r)

        # also compute per-symbol for the winner
        r_ps = run_config(f"F8_PS_{best['config_id']}", gen_al_break_v6, good10_arrays,
                          ranges, htf_by_sym, best["params"], trail_cfg=best.get("trail"),
                          compute_per_sym=True)
        log_r("F8_per_sym", r_ps); results.append(r_ps)

    # ═════════════════════════════════════════════════════════════
    # SAVE & SUMMARY
    # ═════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v6final_{ts}.json"
    def _json_default(x):
        if isinstance(x, set): return sorted(x)
        try: return float(x)
        except Exception: return str(x)
    with open(path, "w") as f:
        json.dump({"results": results}, f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t0:.1f}s)")

    # Summary
    print("\n" + "="*120)
    print("TOP-20 by PnL (V6 FINAL, min 40 trades)")
    print(f"{'config':44s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s}  {'avgR':>5s}")
    flt = [r for r in results if r.get("trades", 0) >= 40]
    flt.sort(key=lambda r: r["pnl_pct"], reverse=True)
    for r in flt[:20]:
        print(f"{r['config_id']:44s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f}  "
              f"{r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f}  {r['avg_r']:>+5.2f}")

    # Walk-forward
    wf = [r for r in results if r["config_id"].startswith("F8_WF_")]
    if wf:
        print("\nWALK-FORWARD of best:")
        for r in wf:
            pf_ok = r["profit_factor"] >= 1.0
            pnl_ok = r["pnl_pct"] > 0
            tag = "PASS" if (pf_ok and pnl_ok) else "FAIL"
            print(f"  {r['config_id']:16s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                  f"PnL={r['pnl_pct']:>+7.1f}% DD={r['max_dd_pct']:>5.2f}%  [{tag}]")

    # Per-symbol of winner
    ps = [r for r in results if r.get("per_sym")]
    if ps:
        b = max(ps, key=lambda x: x["pnl_pct"])
        print(f"\nPer-symbol for {b['config_id']} "
              f"PF={b['profit_factor']} PnL={b['pnl_pct']}% DD={b['max_dd_pct']}%:")
        for sym, s in sorted(b["per_sym"].items(), key=lambda kv: kv[1]["pnl_pct"], reverse=True):
            sn = sym.replace("/USDT:USDT", "")
            mk = "+" if s["pnl_pct"] > 0 else "-"
            print(f"  {mk} {sn:6s} n={s['trades']:>4} WR={s['wr_pct']:>5.1f}% "
                  f"PF={s['profit_factor']:>5.2f} PnL={s['pnl_pct']:>+7.1f}%  avgR={s['avg_r']:>+.2f}")


if __name__ == "__main__":
    main()
