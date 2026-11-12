"""
portfolio_stats.py — Daily trade count + simultaneous exposure statistics
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
    load, build_regime, START_BALANCE,
    MAX_MARGIN_FRAC, TAKER_FEE, SLIPPAGE,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import load_all_events
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter
from multi_agent.backtests.risk_controls_v7 import simulate_rc


def analyse(trades, label=""):
    if not trades:
        print(f"  {label}: no trades"); return

    # ── 1. Trades per day ────────────────────────────────────────────────────
    by_day = defaultdict(int)
    for t in trades:
        day = pd.Timestamp(t["entry_ts"], unit="ms", tz="UTC").date()
        by_day[day] += 1

    daily_counts = np.array(list(by_day.values()))
    all_days = pd.date_range(
        start=pd.Timestamp(min(t["entry_ts"] for t in trades), unit="ms", tz="UTC").date(),
        end  =pd.Timestamp(max(t["entry_ts"] for t in trades), unit="ms", tz="UTC").date(),
        freq ="D"
    )
    trading_days  = len(by_day)
    calendar_days = len(all_days)
    zero_days     = calendar_days - trading_days

    print(f"\n{'━'*68}")
    print(f"  {label}")
    print(f"{'━'*68}")
    print(f"\n  ── Сделки в день ──────────────────────────────────────────────")
    print(f"  Всего трейдов       : {len(trades)}")
    print(f"  Период              : {all_days[0].date()} → {all_days[-1].date()}"
          f"  ({calendar_days} дней)")
    print(f"  Дней с трейдами     : {trading_days}")
    print(f"  Дней без трейдов    : {zero_days}  ({zero_days/calendar_days*100:.0f}%)")
    print(f"  Среднее (торг.дни)  : {daily_counts.mean():.2f} сделок/день")
    print(f"  Среднее (все дни)   : {len(trades)/calendar_days:.3f} сделок/день")
    print(f"  Медиана             : {np.median(daily_counts):.1f}")
    print(f"  Макс. в один день   : {daily_counts.max()}")

    dist = defaultdict(int)
    for v in daily_counts:
        dist[v] += 1
    print(f"\n  Распределение (кол-во дней с N сделками):")
    for k in sorted(dist):
        bar = "█" * dist[k]
        print(f"    {k} сделок: {dist[k]:3d} дней  {bar}")

    # ── 2. Simultaneous open positions ───────────────────────────────────────
    # Build timeline of opens/closes
    events_tl = []
    for t in trades:
        events_tl.append((t["entry_ts"], +1, t))
        events_tl.append((t["exit_ts"],  -1, t))
    events_tl.sort(key=lambda x: x[0])

    open_now = set()
    max_concurrent = 0
    snapshots = []   # (ts, n_open, total_risk_frac, total_notional_frac)

    # We need balance at each point — approximate using START_BALANCE for simplicity
    # (for margin/notional fractions relative to starting capital)
    balance_approx = START_BALANCE
    eq = START_BALANCE
    trade_pnl = {id(t): t["pnl"] for t in trades}

    for ts, delta, t in events_tl:
        if delta == +1:
            open_now.add(id(t))
        else:
            open_now.discard(id(t))
            eq = max(0.01, eq + t["pnl"])

        n = len(open_now)
        if n > max_concurrent:
            max_concurrent = n

        # Compute current open risk & notional
        open_trades = [t2 for t2 in trades if id(t2) in open_now]
        total_risk_usd = sum(t2["risk_usd"] for t2 in open_trades)
        # Notional = risk_usd / dist × entry  ≈ entry × qty
        # dist / entry ≈ SL% → notional ≈ risk_usd / (sl_pct)
        # We have entry and sl stored
        total_notional = 0
        for t2 in open_trades:
            entry = t2.get("entry", 0)
            sl    = t2.get("sl", entry)
            dist  = abs(entry - sl)
            if dist > 0 and entry > 0:
                qty       = t2["risk_usd"] / dist
                notional  = entry * qty
                total_notional += notional

        snapshots.append({
            "ts": ts,
            "n_open": n,
            "risk_usd": total_risk_usd,
            "notional": total_notional,
            "eq": eq,
        })

    if not snapshots:
        return

    df_snap = pd.DataFrame(snapshots)
    # risk fraction vs START_BALANCE (fixed denominator for fair comparison)
    df_snap["risk_frac"]     = df_snap["risk_usd"]  / START_BALANCE * 100
    df_snap["notional_frac"] = df_snap["notional"]  / START_BALANCE * 100

    # Filter to moments when at least 1 position is open
    df_open = df_snap[df_snap["n_open"] > 0]

    print(f"\n  ── Одновременно открытые позиции ──────────────────────────────")
    print(f"  Макс. одновременно  : {max_concurrent}")
    if not df_open.empty:
        print(f"  Среднее (когда >0)  : {df_open['n_open'].mean():.2f}")
        print(f"  Медиана (когда >0)  : {df_open['n_open'].median():.1f}")

    dist2 = df_snap["n_open"].value_counts().sort_index()
    print(f"\n  Распределение моментов времени по числу открытых позиций:")
    total_snaps = len(df_snap)
    for k, cnt in dist2.items():
        bar = "█" * min(cnt, 40)
        print(f"    {k} позиций: {cnt:4d} моментов  ({cnt/total_snaps*100:4.1f}%)  {bar}")

    print(f"\n  ── Задействованный депозит ────────────────────────────────────")
    print(f"  (относительно стартового капитала {START_BALANCE:.0f}$)")
    print(f"\n  По РИСКУ (4% за сделку × N открытых):")
    if not df_open.empty:
        print(f"    Среднее  : {df_open['risk_frac'].mean():5.1f}%  депозита в риске")
        print(f"    Медиана  : {df_open['risk_frac'].median():5.1f}%")
        print(f"    P90      : {df_open['risk_frac'].quantile(0.90):5.1f}%")
        print(f"    Макс     : {df_open['risk_frac'].max():5.1f}%")

    print(f"\n  По НОТИОНАЛЬНОМУ объёму (плечо × маржа):")
    if not df_open.empty:
        print(f"    Среднее  : {df_open['notional_frac'].mean():6.1f}%  от депозита")
        print(f"    Медиана  : {df_open['notional_frac'].median():6.1f}%")
        print(f"    P90      : {df_open['notional_frac'].quantile(0.90):6.1f}%")
        print(f"    Макс     : {df_open['notional_frac'].max():6.1f}%")

    # Per simultaneous position count breakdown
    print(f"\n  Средний риск % по числу одновременных позиций:")
    print(f"  {'N поз':>8s}  {'Случаев':>8s}  {'Риск%':>8s}  {'Ntnl%':>8s}")
    for n in sorted(df_snap["n_open"].unique()):
        sub = df_snap[df_snap["n_open"] == n]
        print(f"  {n:>8d}  {len(sub):>8d}  {sub['risk_frac'].mean():>7.1f}%  "
              f"{sub['notional_frac'].mean():>7.1f}%")


def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  PORTFOLIO STATS — сделок в день + задействованный депозит")
    print("══════════════════════════════════════════════════════════════════════")

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE
              if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n  Symbols: {len(syms)}  |  Loading...")
    sigs_raw, sim_dfs, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    events = load_all_events()

    sigs_base, _, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, 7.0)
    print(f"  Signals: {len(sigs_base)}  |  Events: {len(events)}\n")

    # Run baseline simulation
    res = simulate_rc(sigs_base, sim_dfs, regime, events)
    analyse(res["trades"], "BASELINE (V7+L4+PivotFilter, max_pos=5, risk=4%)")

    # Also show RC5 version (corr filter)
    from multi_agent.backtests.risk_controls_v7 import build_corr_matrix
    corr_mx = build_corr_matrix(sim_dfs)
    res5 = simulate_rc(sigs_base, sim_dfs, regime, events,
                       max_corr_open=1, corr_threshold=0.80, corr_matrix=corr_mx)
    analyse(res5["trades"], "+RC5 (corr max=1 @0.80)")

    print(f"\n{'━'*68}\n")


if __name__ == "__main__":
    main()
