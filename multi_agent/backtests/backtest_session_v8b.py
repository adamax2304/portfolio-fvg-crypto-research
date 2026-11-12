"""
backtest_session_v8b.py -- 2026-04-17

Follow-up to V8:
  - Walk-forward H9_losers2_bars96 (best PF cooldown config — not WF'd in V8)
  - Test "super-champion" stack: V7 base + cooldown + SL1.75/TP5
  - Test cooldown variants on super-champion

V8 had WF Q4 fail on H9_losers4_bars288 (-5.3%). Does H9_losers2_bars96 hold up?

Also: what if the Q4 failure is due to a regime-change, and cooldown helps
recover? Test super-champion on full + per-quarter.
"""
from __future__ import annotations
import sys, logging, time as _time, json
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np

from multi_agent.backtests.backtest_session_v4 import (
    _load, prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf, TOP14, RISK_PER_TRADE,
)
from multi_agent.backtests.backtest_session_v5 import GOOD10
from multi_agent.backtests.backtest_session_v6 import (
    BASE_PARAMS_V5_CHAMPION, split_by_quarters,
)
from multi_agent.backtests.backtest_session_v7 import build_btc_regime
from multi_agent.backtests.backtest_session_v8 import (
    run_config, log_r,
    build_hourly_btc_regime, build_range_expansion_map,
    build_pre_break_momentum, build_vol_ratio_15m, build_symbol_atr_pct,
)

logger = logging.getLogger("bt_v8b")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
RESULTS_DIR = Path(__file__).parent / "results"


def main():
    t0 = _time.time()
    logger.info("=== V8b: WF on cooldown variants + super-champion stack ===")

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

    base = dict(BASE_PARAMS_V5_CHAMPION)
    base["allowed_dows"] = {0, 1, 3, 4, 5, 6}
    base["skip_regimes"] = {"chop"}

    results = []

    # ═══════════════════════════════════════════════════════════════
    # Define candidates
    # ═══════════════════════════════════════════════════════════════
    candidates = []

    # 1. V7 baseline
    candidates.append(("V7_base",             dict(base), None, 0, 0, RISK_PER_TRADE))

    # 2. Cooldown variants
    candidates.append(("H9_cd_2_96",          dict(base), None, 2, 96, RISK_PER_TRADE))
    candidates.append(("H9_cd_3_96",          dict(base), None, 3, 96, RISK_PER_TRADE))
    candidates.append(("H9_cd_2_192",         dict(base), None, 2, 192, RISK_PER_TRADE))
    candidates.append(("H9_cd_3_192",         dict(base), None, 3, 192, RISK_PER_TRADE))
    candidates.append(("H9_cd_4_288",         dict(base), None, 4, 288, RISK_PER_TRADE))

    # 3. Super-champion: V7 + SL1.75/TP5 + cooldown(2,96)
    p = dict(base); p["sl_atr_mult"] = 1.75; p["tp_r"] = 5.0
    candidates.append(("SC_sl1.75tp5_cd2_96", dict(p), None, 2, 96, RISK_PER_TRADE))
    candidates.append(("SC_sl1.75tp5_cd3_96", dict(p), None, 3, 96, RISK_PER_TRADE))
    candidates.append(("SC_sl1.75tp5_cd4_288",dict(p), None, 4, 288, RISK_PER_TRADE))

    # 4. Additional SL/TP combos
    for sl in [1.5, 1.75, 2.0, 2.25]:
        for tp in [4.0, 5.0, 6.0]:
            pp = dict(base); pp["sl_atr_mult"] = sl; pp["tp_r"] = tp
            candidates.append((f"SLTP_sl{sl}_tp{tp}_cd2_96", pp, None, 2, 96, RISK_PER_TRADE))

    # 5. Aggressive risk scaling on cooldown(2,96) stack
    p = dict(base); p["sl_atr_mult"] = 1.75; p["tp_r"] = 5.0
    candidates.append(("SC_rpt0.015", dict(p), None, 2, 96, 0.015))
    candidates.append(("SC_rpt0.02",  dict(p), None, 2, 96, 0.02))

    # 6. Swing trail variants on cooldown(2,96) stack
    for act_r in [1.0, 1.5, 2.0, 2.5]:
        for sw in [5, 8, 10]:
            tcfg = {"activate_at_r": act_r, "trail_mode": "swing",
                    "swing_bars": sw, "buffer_atr": 0.25}
            p = dict(base); p["tp_r"] = 6.0
            candidates.append((f"TR_act{act_r}_sw{sw}_cd2_96", p, tcfg, 2, 96, RISK_PER_TRADE))

    # ═══════════════════════════════════════════════════════════════
    # Run all candidates
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info(f"=== Running {len(candidates)} candidate configs ===")
    for name, p, tcfg, cd_M, cd_bars, rpt in candidates:
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime, trail_cfg=tcfg,
                       risk_per_trade=rpt,
                       cooldown_after_losers=cd_M, cooldown_bars=cd_bars)
        r["cd_M"] = cd_M; r["cd_bars"] = cd_bars
        log_r(name, r); results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # Walk-forward 4Q on top 5 candidates by risk-adjusted PnL
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== WF 4Q on TOP-5 by PnL/DD ratio (min 80 trades) ===")

    cands = [r for r in results if r["trades"] >= 80 and r["max_dd_pct"] > 0]
    cands.sort(key=lambda r: r["pnl_pct"] / max(r["max_dd_pct"], 1.0), reverse=True)
    top5 = cands[:5]
    for b in top5:
        logger.info(f"  -> {b['config_id']:30s} PF={b['profit_factor']:.2f} "
                    f"PnL={b['pnl_pct']:.1f}% DD={b['max_dd_pct']:.2f}% score={b['pnl_pct']/b['max_dd_pct']:.1f}")

    per_sym_q = {s: split_by_quarters(arrays[s]) for s in GOOD10 if s in arrays}
    for b in top5:
        logger.info(f"")
        logger.info(f"--- WF on {b['config_id']} ---")
        for qi in range(4):
            q_arrays = {s: per_sym_q[s][qi] for s in per_sym_q if per_sym_q[s][qi] is not None}
            q_ranges = {s: build_session_ranges(q_arrays[s]) for s in q_arrays}
            r = run_config(f"WF_{b['config_id']}_Q{qi+1}", q_arrays, q_ranges, htf_by_sym,
                           b["params"], btc_regime=btc_regime,
                           trail_cfg=b.get("trail"),
                           risk_per_trade=b.get("risk_per_trade", RISK_PER_TRADE),
                           cooldown_after_losers=b.get("cd_M", 0),
                           cooldown_bars=b.get("cd_bars", 0))
            tag = "PASS" if (r["pnl_pct"] > 0 and r["profit_factor"] >= 1.0) else "FAIL"
            logger.info(f"  Q{qi+1}  trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                        f"PnL={r['pnl_pct']:>+7.1f}% DD={r['max_dd_pct']:>5.2f}% [{tag}]")
            r["wf_tag"] = tag; results.append(r)

    # ═══════════════════════════════════════════════════════════════
    # SAVE + SUMMARY
    # ═══════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v8b_{ts}.json"
    def _json_default(x):
        if isinstance(x, set): return sorted(x)
        try: return float(x)
        except Exception: return str(x)
    with open(path, "w") as f:
        json.dump({"results": results}, f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t0:.1f}s)")

    # SUMMARY
    print("\n" + "="*120)
    print(f"TOP-20 by PnL (V8b, min 40 trades)")
    print(f"{'config':35s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s} {'score':>6s}")
    flt = [r for r in results if r.get("trades", 0) >= 40 and not r["config_id"].startswith("WF_")]
    flt.sort(key=lambda r: r["pnl_pct"], reverse=True)
    for r in flt[:20]:
        score = r["pnl_pct"] / max(r["max_dd_pct"], 1.0)
        print(f"{r['config_id']:35s} {r['trades']:>6} {r['wr_pct']:>5.1f} "
              f"{r['profit_factor']:>5.2f}  {r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f} "
              f"{score:>6.1f}")

    print("\nWALK-FORWARD RESULTS PER CONFIG:")
    wf = [r for r in results if r["config_id"].startswith("WF_")]
    # Group by config prefix
    groups = {}
    for r in wf:
        key = "_".join(r["config_id"].split("_")[1:-1])
        groups.setdefault(key, []).append(r)
    for key, qs in groups.items():
        passes = sum(1 for q in qs if q.get("wf_tag") == "PASS")
        print(f"  {key:30s}: {passes}/{len(qs)} PASS, quarters: "
              + " | ".join(f"Q{i+1}={'P' if q.get('wf_tag')=='PASS' else 'F'}({q['pnl_pct']:+.0f}%)"
                            for i, q in enumerate(sorted(qs, key=lambda x: x['config_id']))))


if __name__ == "__main__":
    main()
