"""
portfolio_v6_final.py — Final V6 with corrected pyramid + funding tuning + best combo
Then run winner on 730d for 2y validation.
"""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, START_BALANCE,
)
from multi_agent.backtests.portfolio_v6 import (
    BASELINE, run_test, build_correlation_clusters, simulate_v6, collect_v6_signals,
    print_res,
)


def main():
    tag = "365d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    sym_cluster = build_correlation_clusters(BASELINE, tag, threshold=0.85)

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V6 FIXES: pyramid 2R+SL=+0.3R + funding lower threshold")
    print(f"══════════════════════════════════════════════════════════════════")

    all_results = {}

    # Test fixed pyramid (2R trigger, SL = BE + 0.3R)
    for at_r, sl_off in [(2.0, 0.3), (2.0, 0.5), (1.5, 0.5), (2.5, 0.5), (3.0, 0.7)]:
        # We need to call simulate_v6 directly with these params
        sigs, sim_dfs = collect_v6_signals(
            BASELINE, regime, tag,
            vol_smacross=1.3, vol_supertrend=1.7, vol_rsi=1.5, vol_fib=1.5,
            vol_window=30,
        )
        if not sigs: continue
        res = simulate_v6(sigs, sim_dfs, regime,
                            start_balance=START_BALANCE,
                            base_risk=0.03,
                            use_pyramid=True,
                            pyramid_at_r=at_r,
                            pyramid_sl_offset_r=sl_off)
        m = res["metrics"]; rets = [r["ret_pct"] for r in res["monthly_returns"]]
        avgmo = float(np.mean(rets)) if rets else 0
        winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
        name = f"PYR_{at_r}R_SL+{sl_off}"
        all_results[name] = {"metrics": m, "rets": rets}
        print(f"  {name:20s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
              f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+7.1f}% DD={m['max_dd_pct']:>5.1f}% "
              f"Avg/mo={avgmo:>+5.1f}% Win={winmo}/{totmo}")

    # Funding with various thresholds
    print(f"\n  Funding skip threshold tuning:")
    for fund_thresh in [0.0001, 0.0002, 0.0005, 0.001]:
        # Custom funding: pass to collect_v6
        sigs, sim_dfs = collect_v6_signals(
            BASELINE, regime, tag,
            vol_smacross=1.3, vol_supertrend=1.7, vol_rsi=1.5, vol_fib=1.5,
            vol_window=30,
            use_funding=True, funding_thresh=fund_thresh,
        )
        if not sigs: continue
        res = simulate_v6(sigs, sim_dfs, regime,
                            start_balance=START_BALANCE, base_risk=0.03)
        m = res["metrics"]; rets = [r["ret_pct"] for r in res["monthly_returns"]]
        avgmo = float(np.mean(rets)) if rets else 0
        winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
        name = f"FUND_{fund_thresh*10000:.1f}bp"
        all_results[name] = {"metrics": m, "rets": rets}
        print(f"  {name:20s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
              f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+7.1f}% DD={m['max_dd_pct']:>5.1f}% "
              f"Avg/mo={avgmo:>+5.1f}% Win={winmo}/{totmo}")

    # Best combo: per-strat vol + best pyramid + best funding
    print(f"\n  Best combinations:")
    best_pyr_at_r, best_pyr_sl = 2.0, 0.5  # default safe
    best_fund_thresh = 0.0002  # 0.02%

    combos = [
        ("V5_BASELINE", {}),
        ("V6_per_strat_vol", {}),
        ("V6_per_strat+fund", {"use_funding": True, "funding_thresh": best_fund_thresh}),
        ("V6_per_strat+pyr2R", {"use_pyramid": True, "pyramid_at_r": 2.0, "pyramid_sl_offset_r": 0.5}),
        ("V6_per_strat+pyr+fund", {"use_pyramid": True, "pyramid_at_r": 2.0,
                                     "pyramid_sl_offset_r": 0.5,
                                     "use_funding": True, "funding_thresh": best_fund_thresh}),
        ("V6_ALL_safe", {"use_pyramid": True, "pyramid_at_r": 2.5, "pyramid_sl_offset_r": 0.5,
                          "use_correlation_cap": True, "sym_cluster": sym_cluster,
                          "use_funding": True, "funding_thresh": best_fund_thresh}),
    ]
    for name, kwargs in combos:
        sim_kwargs = {k: kwargs.get(k) for k in
                       ["use_pyramid", "pyramid_at_r", "pyramid_sl_offset_r",
                        "use_correlation_cap", "sym_cluster", "use_vol_scaled"]
                       if k in kwargs}
        col_kwargs = {k: kwargs.get(k) for k in
                       ["use_funding", "funding_thresh"] if k in kwargs}
        if name == "V5_BASELINE":
            vsm, vst = 1.5, 1.5
        else:
            vsm, vst = 1.3, 1.7
        sigs, sim_dfs = collect_v6_signals(
            BASELINE, regime, tag,
            vol_smacross=vsm, vol_supertrend=vst, vol_rsi=1.5, vol_fib=1.5,
            vol_window=30, **col_kwargs,
        )
        if not sigs: continue
        res = simulate_v6(sigs, sim_dfs, regime,
                            start_balance=START_BALANCE, base_risk=0.03,
                            **sim_kwargs)
        m = res["metrics"]; rets = [r["ret_pct"] for r in res["monthly_returns"]]
        avgmo = float(np.mean(rets)) if rets else 0
        winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
        all_results[name] = {"metrics": m, "rets": rets, "trades": res["trades"]}
        print(f"  {name:30s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
              f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+7.1f}% DD={m['max_dd_pct']:>5.1f}% "
              f"Avg/mo={avgmo:>+5.1f}% Win={winmo}/{totmo}")

    # FINAL: best of best on 730d
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  RUN BEST CONFIG ON 730d (2-year validation)")
    print(f"══════════════════════════════════════════════════════════════════")

    btc730 = load("BTC", "1d", "730d")
    regime730 = build_regime(btc730)

    # Best winners on 730d (only includes 27 baseline syms; need to verify 730d has them)
    syms_730 = [s for s in BASELINE if (Path(__file__).parent / "data_cache" /
                                           f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n  730d symbols available: {len(syms_730)}")

    final_combos = [
        ("V6_FINAL_per_strat", {}),
        ("V6_FINAL_per_strat+fund", {"use_funding": True, "funding_thresh": best_fund_thresh}),
        ("V6_FINAL_per_strat_4pct", {"base_risk": 0.04}),
    ]
    for name, kwargs in final_combos:
        risk = kwargs.get("base_risk", 0.03)
        col_kwargs = {k: kwargs.get(k) for k in
                       ["use_funding", "funding_thresh"] if k in kwargs}
        sigs, sim_dfs = collect_v6_signals(
            syms_730, regime730, "730d",
            vol_smacross=1.3, vol_supertrend=1.7, vol_rsi=1.5, vol_fib=1.5,
            vol_window=30, **col_kwargs,
        )
        if not sigs: continue
        res = simulate_v6(sigs, sim_dfs, regime730,
                            start_balance=START_BALANCE, base_risk=risk)
        m = res["metrics"]; rets = [r["ret_pct"] for r in res["monthly_returns"]]
        avgmo = float(np.mean(rets)) if rets else 0
        medmo = float(np.median(rets)) if rets else 0
        winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
        print(f"  {name:30s}")
        print(f"    Trades: {m['n']}  PF: {m['pf']:.2f}  WR: {m['wr']:.1f}%")
        print(f"    avgR: {m['avg_r']:+.3f}  Return: {m['return_pct']:+.1f}%  DD: {m['max_dd_pct']:.1f}%")
        print(f"    Avg/mo: {avgmo:+.1f}%  Median: {medmo:+.1f}%  Win-mo: {winmo}/{totmo}")
        # Per-strategy
        per = defaultdict(list)
        for t in res["trades"]: per[t["strat"]].append(t)
        for sn, tlist in sorted(per.items()):
            from multi_agent.backtests.strategy_lab import metrics_dict_from_trades
            mm = metrics_dict_from_trades(tlist, START_BALANCE)
            print(f"      {sn:15s}: n={mm.get('n',0):4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f}")


if __name__ == "__main__":
    main()
