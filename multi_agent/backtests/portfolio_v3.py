"""
portfolio_v3.py — Final v3 portfolio with all winning components
==================================================================
Components:
  • smacross_pivot_1d (PF=1.57, DD=12.9R)  — pivot-stop golden cross
  • supertrend_4h (PF=1.05, n=3056)         — high-frequency baseline
  • rsi_reversion_4h_v2 (PF=2.06)           — wider SL (sl=2.0/tp=3.0)
  • fib_pump_4h_optim (PF=1.20)             — niche pump retracement

Compounding portfolio sim with:
  • TAKER fees both sides
  • Slippage both sides
  • Funding cost
  • Max 5 concurrent positions
  • SL cooldown 4h per symbol
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
    load, build_regime, simulate_portfolio,
    SYMBOLS_730D, START_BALANCE, RISK_PER_TRADE, MAX_POSITIONS,
    PortfolioSignal, _ts_period_ms, atr, ema, sma, rsi, supertrend,
    _emit, Signal, regime_at, metrics_dict_from_trades, _monthly_returns,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
)


def collect_v3_signals(syms, regime, tag="730d"):
    sigs: list[PortfolioSignal] = []
    sim_dfs = {}

    # 1. smacross_pivot_1d
    for sym in syms:
        df = load(sym, "1d", tag)
        if df is None: continue
        slist = strat_smacross_pivot_1d(df, regime, tp_r=4.0)  # use 4R TP from grid
        sim_dfs[(sym, "smacross_pivot_1d")] = df
        for s in slist:
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts,
                sym=sym, strat="smacross_pivot_1d", direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    # 2. supertrend_pivot_4h
    for sym in syms:
        df = load(sym, "4h", tag)
        if df is None: continue
        slist = strat_supertrend_pivot_4h(df, regime, tp_r=2.5)
        sim_dfs[(sym, "supertrend_pivot_4h")] = df
        for s in slist:
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts,
                sym=sym, strat="supertrend_pivot_4h", direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    # 3. rsi_reversion_4h with optimized sl=2.0 tp=3.0
    for sym in syms:
        df = load(sym, "4h", tag)
        if df is None: continue
        slist = _strat_rsi_rev_v2(df, regime, sl_atr=2.0, tp_r=3.0)
        sim_dfs[(sym, "rsi_reversion_v2")] = df
        for s in slist:
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts,
                sym=sym, strat="rsi_reversion_v2", direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    # 4. fib_pump_4h optimized
    for sym in syms:
        df = load(sym, "4h", tag)
        if df is None: continue
        slist = strat_fib_pump_1h(df, regime,
                                   consolidation_bars=18,
                                   pump_min_atr=3.0,
                                   pump_max_bars=6,
                                   retrace_min=0.382,
                                   retrace_max=0.618,
                                   tp_extension=1.0,
                                   sl_buffer_atr=0.5)
        sim_dfs[(sym, "fib_pump_4h")] = df
        for s in slist:
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts,
                sym=sym, strat="fib_pump_4h", direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


def _strat_rsi_rev_v2(df, regime, sl_atr, tp_r, os=20, ob=80):
    if df is None or len(df) < 30: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    r = rsi(cl, 14)
    sigs = []
    for i in range(20, len(cl) - 1):
        if np.isnan(a[i]) or np.isnan(r[i]) or np.isnan(r[i-1]): continue
        trend, _ = regime_at(regime, int(ts[i]))
        if trend != 0: continue  # sideways only
        next_open = op[i+1]
        if r[i-1] <= os and r[i] > os:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif r[i-1] >= ob and r[i] < ob:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = SYMBOLS_730D

    print("Collecting V3 signals across 4 strategies...")
    sigs, sim_dfs = collect_v3_signals(syms, regime, "730d")
    print(f"Total signals: {len(sigs)}")
    counts = defaultdict(int)
    for s in sigs: counts[s.strat] += 1
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # Run portfolio at multiple risk levels
    for risk, max_pos in [(0.02, 4), (0.03, 5), (0.04, 5), (0.05, 6)]:
        print(f"\n══════════════════════════════════════════════════════════════════")
        print(f"  V3 PORTFOLIO  risk={risk*100:.0f}%  max_pos={max_pos}")
        print(f"══════════════════════════════════════════════════════════════════")
        res = simulate_portfolio(sigs, sim_dfs, regime,
                                  start_balance=START_BALANCE,
                                  risk=risk, max_pos=max_pos)
        m = res["metrics"]
        print(f"  Trades: {m['n']}  WR: {m['wr']}%  PF: {m['pf']}  avgR: {m['avg_r']:+.3f}")
        print(f"  Final: ${m['final_balance']:,.2f}  Return: {m['return_pct']:+.1f}%  DD: {m['max_dd_pct']:.1f}%")

        per = defaultdict(list)
        for t in res["trades"]:
            per[t["strat"]].append(t)
        print(f"  Per-strategy:")
        for sn, tlist in sorted(per.items()):
            mm = metrics_dict_from_trades(tlist, START_BALANCE)
            print(f"    {sn:25s}: n={mm['n']:4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f} sumR={mm.get('sum_r',0):+7.1f}")

        # Monthly stats
        rets = [mr["ret_pct"] for mr in res["monthly_returns"]]
        if rets:
            print(f"  Avg monthly: {np.mean(rets):+.1f}%   Median: {np.median(rets):+.1f}%   Std: {np.std(rets):.1f}%")
            print(f"  Profitable months: {sum(1 for r in rets if r > 0)}/{len(rets)}")
            print(f"  Best month: {max(rets):+.1f}%   Worst: {min(rets):+.1f}%")

    # Show monthly breakdown for the best config
    print(f"\n══════ Monthly equity curve (risk=3%, max_pos=5) ══════")
    res = simulate_portfolio(sigs, sim_dfs, regime,
                              start_balance=START_BALANCE,
                              risk=0.03, max_pos=5)
    print(f"  {'Month':10s} {'Start':>10s} {'End':>10s} {'Return':>9s}")
    for mr in res["monthly_returns"]:
        print(f"  {mr['month']:10s} ${mr['start']:>9,.2f} ${mr['end']:>9,.2f} {mr['ret_pct']:>+8.1f}%")


if __name__ == "__main__":
    main()
