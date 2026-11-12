"""
v_730d_audit.py — Compare V3 / V5 / V6 on 730d to find what works
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import defaultdict
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from multi_agent.backtests.strategy_lab import (
    load, build_regime, simulate_portfolio, START_BALANCE,
    metrics_dict_from_trades,
)
from multi_agent.backtests.portfolio_v3 import collect_v3_signals  # original V3
from multi_agent.backtests.portfolio_v6 import (
    BASELINE, collect_v6_signals, simulate_v6,
)


def run(name, sigs, sim_dfs, regime, risk=0.03, max_pos=5):
    if not sigs:
        print(f"  {name}: no signals"); return None
    res = simulate_portfolio(sigs, sim_dfs, regime,
                              start_balance=START_BALANCE, risk=risk, max_pos=max_pos)
    m = res["metrics"]
    rets = [r["ret_pct"] for r in res["monthly_returns"]]
    avgmo = float(np.mean(rets)) if rets else 0
    medmo = float(np.median(rets)) if rets else 0
    winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
    print(f"  {name:35s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
          f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+7.1f}% DD={m['max_dd_pct']:>5.1f}% "
          f"Avg/mo={avgmo:+.1f}% Med={medmo:+.1f}% Win={winmo}/{totmo}")
    per = defaultdict(list)
    for t in res["trades"]: per[t["strat"]].append(t)
    for sn, tlist in sorted(per.items()):
        mm = metrics_dict_from_trades(tlist, START_BALANCE)
        print(f"      {sn:15s}: n={mm.get('n',0):4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f}")
    return {"name": name, "metrics": m, "rets": rets}


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms_730 = [s for s in BASELINE if (Path(__file__).parent / "data_cache" /
                                          f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n730d symbols: {len(syms_730)}")
    print(f"Available: {syms_730}")

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  AUDIT: V3 vs V5 vs V6 on 730d (BASELINE)")
    print(f"══════════════════════════════════════════════════════════════════")

    # 1. V3 ORIGINAL (no vol filter, no per-strat vol)
    sigs, dfs = collect_v3_signals(syms_730, regime, "730d")
    run("V3_ORIGINAL_3pct", sigs, dfs, regime, risk=0.03)

    # 2. V3 4% risk (V3 winner from earlier)
    run("V3_ORIGINAL_4pct", sigs, dfs, regime, risk=0.04)

    # 3. V5: vol filter @ 1.5x, window 30 (same TPs as V3 — no per-strat)
    sigs, dfs = collect_v6_signals(syms_730, regime, "730d",
                                       vol_smacross=1.5, vol_supertrend=1.5,
                                       vol_rsi=1.5, vol_fib=1.5, vol_window=30)
    run("V5_VOL1.5_W30_3pct", sigs, dfs, regime, risk=0.03)

    # 4. V5 4% risk
    run("V5_VOL1.5_W30_4pct", sigs, dfs, regime, risk=0.04)

    # 5. V5 with window 20 (the original winner)
    sigs, dfs = collect_v6_signals(syms_730, regime, "730d",
                                       vol_smacross=1.5, vol_supertrend=1.5,
                                       vol_rsi=1.5, vol_fib=1.5, vol_window=20)
    run("V5_VOL1.5_W20_3pct", sigs, dfs, regime, risk=0.03)

    # 6. V5 with vol_window=10 (more responsive)
    sigs, dfs = collect_v6_signals(syms_730, regime, "730d",
                                       vol_smacross=1.5, vol_supertrend=1.5,
                                       vol_rsi=1.5, vol_fib=1.5, vol_window=10)
    run("V5_VOL1.5_W10_3pct", sigs, dfs, regime, risk=0.03)

    # 7. NO VOL FILTER (vol_thresh = 0 effectively)
    sigs, dfs = collect_v6_signals(syms_730, regime, "730d",
                                       vol_smacross=0.0, vol_supertrend=0.0,
                                       vol_rsi=0.0, vol_fib=0.0, vol_window=30)
    run("NO_VOL_FILTER_3pct", sigs, dfs, regime, risk=0.03)

    # 8. NO VOL FILTER 4%
    run("NO_VOL_FILTER_4pct", sigs, dfs, regime, risk=0.04)

    # 9. V6 per-strat vol (1.3/1.7)
    sigs, dfs = collect_v6_signals(syms_730, regime, "730d",
                                       vol_smacross=1.3, vol_supertrend=1.7,
                                       vol_rsi=1.5, vol_fib=1.5, vol_window=30)
    run("V6_PER_STRAT_3pct", sigs, dfs, regime, risk=0.03)

    # 10. V6 per-strat vol 4%
    run("V6_PER_STRAT_4pct", sigs, dfs, regime, risk=0.04)


if __name__ == "__main__":
    main()
