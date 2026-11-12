"""
pivot_dist_filter.py — Test pivot-to-entry distance filter on V7+L4

Concept: skip signal when entry price is too far above the pivot used for SL.
This catches "late" entries where price already had a big run from swing low.

Filter: if (entry - pivot) / ATR > N → skip.
Equivalently: SL distance in ATR units (sig.dist / ATR(14)) > N.
"""
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
    load, build_regime, atr as _atr,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import (
    load_all_events, build_sentiment_series, simulate_news_aware,
)
from multi_agent.backtests.filter_tests_v7 import simulate_filtered


def apply_pivot_dist_filter(sigs, sim_dfs, max_dist_atr: float):
    """
    Skip signals where (entry - SL) / ATR > max_dist_atr.
    For pivot-based strategies, this ≈ pivot-to-entry distance in ATR.
    For ATR-based strategies (rsi_rev), distance is fixed, so they pass.
    """
    if max_dist_atr <= 0:
        return sigs, 0, {}

    kept = []
    removed = 0
    atr_cache = {}
    removed_by_strat = defaultdict(int)
    distance_dist = []  # for histogram

    for sig in sigs:
        key = (sig.sym, sig.strat)
        df  = sim_dfs.get(key)
        if df is None:
            kept.append(sig); continue
        if key not in atr_cache:
            atr_cache[key] = _atr(
                df["high"].to_numpy(float),
                df["low"].to_numpy(float),
                df["close"].to_numpy(float), 14
            )
        a_arr = atr_cache[key]
        ts_arr = df["timestamp"].to_numpy(int)
        idx = int(np.searchsorted(ts_arr, sig.ts, side="left"))
        if idx >= len(a_arr) or np.isnan(a_arr[idx]) or a_arr[idx] <= 0:
            kept.append(sig); continue

        dist_atr = sig.dist / a_arr[idx]
        distance_dist.append((sig.strat, dist_atr))

        if dist_atr > max_dist_atr:
            removed += 1
            removed_by_strat[sig.strat] += 1
        else:
            kept.append(sig)

    return kept, removed, dict(removed_by_strat)


def fmt(name, res, baseline_m=None, removed_info=""):
    m = res["metrics"]
    if m["n"] == 0:
        print(f"  {name:48s} NO TRADES"); return
    d_wr  = f" ({m['wr'] - baseline_m['wr']:+.1f})"   if baseline_m else ""
    d_ret = f" ({m['return_pct'] - baseline_m['return_pct']:+.1f})" if baseline_m else ""
    d_pf  = f" ({m['pf'] - baseline_m['pf']:+.2f})"   if baseline_m else ""
    d_dd  = f" ({m['max_dd_pct'] - baseline_m['max_dd_pct']:+.1f})" if baseline_m else ""
    print(
        f"  {name:48s}  n={m['n']:3d}  WR={m['wr']:5.1f}%{d_wr:9s}  "
        f"PF={m['pf']:5.2f}{d_pf:8s}  AvgR={m['avg_r']:+5.2f}  "
        f"Ret={m['return_pct']:+8.1f}%{d_ret:9s}  DD={m['max_dd_pct']:5.1f}%{d_dd:8s}"
        f"{('  ' + removed_info) if removed_info else ''}"
    )


def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  PIVOT-DISTANCE FILTER on V7+L4")
    print("══════════════════════════════════════════════════════════════════════")

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n  Symbols: {len(syms)}  |  Loading signals...")

    sigs_raw, sim_dfs, info = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    events    = load_all_events()
    print(f"  Signals: {len(sigs_raw)}  |  Events: {len(events)}\n")

    # ── Distribution of pivot distances ──────────────────────────────────────
    print("──────────────────────────────────────────────────────────────────────")
    print("  Distribution of (entry-SL)/ATR across signals (per strategy)")
    print("──────────────────────────────────────────────────────────────────────")
    _, _, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, max_dist_atr=999)
    by_strat = defaultdict(list)
    atr_cache = {}
    for sig in sigs_raw:
        key = (sig.sym, sig.strat)
        df  = sim_dfs.get(key)
        if df is None: continue
        if key not in atr_cache:
            atr_cache[key] = _atr(df["high"].to_numpy(float),
                                    df["low"].to_numpy(float),
                                    df["close"].to_numpy(float), 14)
        a_arr = atr_cache[key]
        ts_arr = df["timestamp"].to_numpy(int)
        idx = int(np.searchsorted(ts_arr, sig.ts, side="left"))
        if idx >= len(a_arr) or np.isnan(a_arr[idx]) or a_arr[idx] <= 0: continue
        by_strat[sig.strat].append(sig.dist / a_arr[idx])

    print(f"  {'Strategy':18s} {'n':>5s} {'min':>6s} {'p25':>6s} {'p50':>6s} "
          f"{'p75':>6s} {'p90':>6s} {'p95':>6s} {'max':>6s}")
    for strat, dists in sorted(by_strat.items()):
        a = np.array(dists)
        print(f"  {strat:18s} {len(a):>5d} {a.min():>6.2f} "
              f"{np.percentile(a,25):>6.2f} {np.percentile(a,50):>6.2f} "
              f"{np.percentile(a,75):>6.2f} {np.percentile(a,90):>6.2f} "
              f"{np.percentile(a,95):>6.2f} {a.max():>6.2f}")

    # ── BASELINE ─────────────────────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  BASELINE (V7+L4, no filter)")
    print("──────────────────────────────────────────────────────────────────────")
    base_res = simulate_filtered(sigs_raw, sim_dfs, regime, events,
                                  base_risk=0.04, enable_l4=True, range_bars=0)
    base_m = base_res["metrics"]
    fmt("BASELINE", base_res)

    # ── Filter grid ──────────────────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  Pivot-distance grid: skip if (entry-SL)/ATR > max_dist")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  {'Config':48s}  {'n':>3s}  {'WR%':14s}  {'PF':13s}  {'AvgR':6s}  "
          f"{'Ret%':16s}  {'DD%':12s}  removed")

    best = None; best_ret = base_m["return_pct"]; best_dd_thresh = None

    for max_d in [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 12.0]:
        sigs_f, removed, by_strat_rem = apply_pivot_dist_filter(
            sigs_raw, sim_dfs, max_dist_atr=max_d
        )
        res = simulate_filtered(sigs_f, sim_dfs, regime, events,
                                 base_risk=0.04, enable_l4=True, range_bars=0)
        rem_info = f"removed {removed} ({removed/len(sigs_raw)*100:.1f}%)"
        # by-strat breakdown
        by_str = ", ".join(f"{k[:5]}={v}" for k, v in by_strat_rem.items())
        rem_full = f"{rem_info:25s}  [{by_str}]"
        fmt(f"max_dist={max_d:.1f}×ATR", res, base_m, rem_full)

        m = res["metrics"]
        # Track best by Ret only if DD doesn't deteriorate too much
        if m["return_pct"] > best_ret and m["n"] >= 80 and m["max_dd_pct"] <= base_m["max_dd_pct"] + 5:
            best_ret = m["return_pct"]; best = max_d

    # ── Combination with F2 (range exit) ──────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  COMBINED: pivot-distance filter + F2 range-exit")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  {'Config':48s}  {'n':>3s}  {'WR%':14s}  {'PF':13s}  {'AvgR':6s}  "
          f"{'Ret%':16s}  {'DD%':12s}")

    for max_d in [4.0, 5.0, 6.0]:
        for rb, ra in [(12, 1.5), (8, 1.5)]:
            sigs_f, removed, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, max_d)
            res = simulate_filtered(sigs_f, sim_dfs, regime, events,
                                     base_risk=0.04, enable_l4=True,
                                     range_bars=rb, range_atr_mult=ra,
                                     range_min_progress=0.20,
                                     range_close_mode="market")
            fmt(f"max_dist={max_d:.1f}×ATR + F2(bars≥{rb} rng<{ra:.1f})",
                res, base_m, f"removed {removed}")

    # ── Walk-forward on best config ──────────────────────────────────────────
    if best is not None:
        print(f"\n──────────────────────────────────────────────────────────────────────")
        print(f"  WALK-FORWARD: max_dist={best:.1f}×ATR  (3-month chronological folds)")
        print(f"──────────────────────────────────────────────────────────────────────")
        sigs_f, removed, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, best)

        all_ts = sorted([s.next_open_ts for s in sigs_f])
        first = all_ts[0]; last = all_ts[-1]
        train_ms = 6 * 30 * 86_400_000
        test_ms  = 3 * 30 * 86_400_000
        cursor   = first + train_ms
        folds_filt = []; folds_base = []
        while cursor + test_ms <= last:
            train_end = cursor; test_end = cursor + test_ms
            test_filt = [s for s in sigs_f   if train_end <= s.next_open_ts < test_end]
            test_base = [s for s in sigs_raw if train_end <= s.next_open_ts < test_end]
            r_filt = simulate_filtered(test_filt, sim_dfs, regime, events,
                                        base_risk=0.04, enable_l4=True, range_bars=0)
            r_base = simulate_filtered(test_base, sim_dfs, regime, events,
                                        base_risk=0.04, enable_l4=True, range_bars=0)
            t1 = pd.Timestamp(test_end, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            folds_filt.append((t1, r_filt["metrics"]))
            folds_base.append((t1, r_base["metrics"]))
            cursor += test_ms

        print(f"  {'End':12s} {'BASELINE Ret':>14s} {'BASELINE WR':>13s} | "
              f"{'FILTER Ret':>14s} {'FILTER WR':>13s} | {'ΔRet':>8s} {'ΔWR':>7s}")
        wins_base = 0; wins_filt = 0
        for (te, mb), (_, mf) in zip(folds_base, folds_filt):
            d_ret = mf["return_pct"] - mb["return_pct"]
            d_wr  = mf["wr"] - mb["wr"]
            if mb["return_pct"] > 0: wins_base += 1
            if mf["return_pct"] > 0: wins_filt += 1
            print(f"  {te:12s} {mb['return_pct']:>+12.1f}% {mb['wr']:>11.1f}% | "
                  f"{mf['return_pct']:>+12.1f}% {mf['wr']:>11.1f}% | "
                  f"{d_ret:>+7.1f}% {d_wr:>+6.1f}")
        print(f"\n  Positive folds: BASELINE {wins_base}/{len(folds_base)}  →  FILTER {wins_filt}/{len(folds_filt)}")

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  VERDICT")
    print("══════════════════════════════════════════════════════════════════════")
    if best is not None:
        sigs_f, removed, by_strat_rem = apply_pivot_dist_filter(sigs_raw, sim_dfs, best)
        res = simulate_filtered(sigs_f, sim_dfs, regime, events,
                                 base_risk=0.04, enable_l4=True, range_bars=0)
        m = res["metrics"]
        print(f"\n  Best filter: skip if pivot distance > {best:.1f}×ATR")
        print(f"  Removed: {removed}/{len(sigs_raw)} signals ({removed/len(sigs_raw)*100:.0f}%)")
        print(f"\n  Baseline:  Ret={base_m['return_pct']:+.1f}%  WR={base_m['wr']:.1f}%  "
              f"PF={base_m['pf']:.2f}  DD={base_m['max_dd_pct']:.1f}%")
        print(f"  Filtered:  Ret={m['return_pct']:+.1f}%  WR={m['wr']:.1f}%  "
              f"PF={m['pf']:.2f}  DD={m['max_dd_pct']:.1f}%")
        print(f"\n  Δ Ret: {m['return_pct'] - base_m['return_pct']:+.1f}%")
        print(f"  Δ WR : {m['wr'] - base_m['wr']:+.1f}pp")
        print(f"  Δ PF : {m['pf'] - base_m['pf']:+.2f}")
        print(f"  Δ DD : {m['max_dd_pct'] - base_m['max_dd_pct']:+.1f}pp")
    else:
        print(f"\n  → No threshold improves return without inflating DD by >5pp.")
        print(f"  Baseline remains optimal: Ret={base_m['return_pct']:+.1f}%  "
              f"WR={base_m['wr']:.1f}%  PF={base_m['pf']:.2f}  DD={base_m['max_dd_pct']:.1f}%")
    print()


if __name__ == "__main__":
    main()
