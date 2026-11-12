"""V7 winner: BEAR=long_only — full validation with walk-forward + risk grid."""
from __future__ import annotations
import sys
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
from multi_agent.backtests.portfolio_v7 import (
    BASELINE, collect_v7_signals, simulate_v7, print_res, print_per_strat,
    CACHE_DIR,
)


def main():
    tag = "730d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    syms = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_{tag}.parquet").exists()]

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V7 WINNER VALIDATION (BEAR=long_only on 730d)")
    print(f"══════════════════════════════════════════════════════════════════")

    sigs, dfs, info = collect_v7_signals(syms, regime, tag, bear_mode="long_only")
    print(f"  Bear filter: {info['bear_filtered']}/{info['bear_signals']} bear signals filtered (SHORT in bear)")

    print(f"\n  Risk level grid:")
    for risk in [0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
        res = simulate_v7(sigs, dfs, regime, base_risk=risk)
        print_res(f"BEAR_LONG_ONLY_R{int(risk*1000)/10:.1f}pct", res)

    print(f"\n  Combined: BEAR_LONG_ONLY + bear_risk_mult variations:")
    for risk in [0.04]:
        for bear_mult in [0.5, 0.75, 1.0, 1.25]:
            res = simulate_v7(sigs, dfs, regime, base_risk=risk, bear_risk_mult=bear_mult)
            print_res(f"R{risk*100:.0f}pct_bear_mult={bear_mult}", res)

    # Walk-forward on best
    print(f"\n══════ Walk-forward on BEAR=long_only @ 4% risk ══════")
    all_ts = sorted([s.next_open_ts for s in sigs])
    if not all_ts: return
    first = all_ts[0]; last = all_ts[-1]
    train_ms = 6 * 30 * 86_400_000
    test_ms = 3 * 30 * 86_400_000
    folds = []
    cursor = first + train_ms
    while cursor + test_ms <= last:
        train_end = cursor; test_end = cursor + test_ms
        test_sigs = [s for s in sigs if train_end <= s.next_open_ts < test_end]
        if test_sigs:
            res = simulate_v7(test_sigs, dfs, regime, base_risk=0.04, start_balance=1000)
            m = res["metrics"]
            t0 = pd.Timestamp(train_end, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            t1 = pd.Timestamp(test_end, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            folds.append({"start": t0, "end": t1, **m})
        cursor += test_ms
    print(f"  {'Fold':12s} {'End':12s} {'n':>4s} {'PF':>5s} {'WR%':>5s} {'Return':>8s} {'DD%':>5s}")
    pos_folds = 0
    for f in folds:
        is_pos = "✓" if f.get("return_pct", 0) > 0 else "✗"
        pos_folds += 1 if f.get("return_pct", 0) > 0 else 0
        print(f"  {is_pos} {f['start']:10s} {f['end']:10s} {f['n']:>4d} {f['pf']:>5.2f} {f['wr']:>5.1f} "
              f"{f['return_pct']:>+7.1f}% {f['max_dd_pct']:>5.1f}%")
    print(f"\n  Positive folds: {pos_folds}/{len(folds)} = {pos_folds/max(1,len(folds))*100:.0f}%")

    # Best risk level confirmed — show full breakdown
    print(f"\n══════ FINAL: BEAR=long_only @ 4% risk full breakdown ══════")
    res = simulate_v7(sigs, dfs, regime, base_risk=0.04)
    print_res("V7_FINAL", res)
    print_per_strat(res)
    print(f"\n  Monthly returns:")
    print(f"  {'Month':10s} {'Return':>9s}")
    for mr in res["monthly"]:
        print(f"  {mr['month']:10s} {mr['ret_pct']:>+8.1f}%")

    # Sensitivity check
    print(f"\n══════ Sensitivity (BEAR_MODE alternatives near long_only) ══════")
    for mode in ["all", "long_only", "skip"]:
        sigs2, dfs2, info2 = collect_v7_signals(syms, regime, tag, bear_mode=mode)
        res = simulate_v7(sigs2, dfs2, regime, base_risk=0.04)
        print_res(f"  bear={mode}", res)


if __name__ == "__main__":
    main()
