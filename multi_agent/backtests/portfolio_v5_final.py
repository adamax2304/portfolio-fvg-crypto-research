"""
portfolio_v5_final.py — Final tuning around the volume-spike filter
=====================================================================
Confirmed: Volume @ 1.5× threshold + NEW TPs is the winning combo.
Now test:
  • Threshold 1.2 / 1.5 / 1.7 / 2.0
  • Risk levels 2% / 3% / 4% / 5%
  • Different windows for vol average (10/20/30 bars)
  • Combine with COMPOSITE universe at the same threshold
"""
from __future__ import annotations
import sys
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from collections import defaultdict

from multi_agent.backtests.strategy_lab import (
    load, build_regime, simulate_portfolio, atr,
    PortfolioSignal, START_BALANCE,
    metrics_dict_from_trades,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
)
from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2
from multi_agent.backtests.universe_filters import list_available_symbols


def _vol_ok(df, j, threshold, window):
    if j < window + 1: return True
    vo = df["volume"].to_numpy()
    avg = np.nanmean(vo[max(0, j-window):j])
    return vo[j] >= threshold * avg if avg > 0 else True


def collect(syms, regime, tag, vol_thresh, vol_window,
             smacross_tp=4.0, supertrend_tp=5.0, rsi_tp=3.0):
    sigs = []; sim_dfs = {}
    def add(strat, df, slist, sym):
        if df is None: return
        sim_dfs[(sym, strat)] = df
        ts = df["timestamp"].to_numpy()
        for s in slist:
            j = int(np.searchsorted(ts, s.next_open_ts, side="left"))
            if j >= len(ts): continue
            if not _vol_ok(df, j, vol_thresh, vol_window): continue
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts, sym=sym,
                strat=strat, direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))
    for sym in syms:
        df1 = load(sym, "1d", tag)
        if df1 is not None:
            add("smacross", df1, strat_smacross_pivot_1d(df1, regime, tp_r=smacross_tp), sym)
        df4 = load(sym, "4h", tag)
        if df4 is not None:
            add("supertrend", df4, strat_supertrend_pivot_4h(df4, regime, tp_r=supertrend_tp), sym)
            add("rsi_rev", df4, _strat_rsi_rev_v2(df4, regime, sl_atr=2.0, tp_r=rsi_tp), sym)
            add("fib_pump", df4, strat_fib_pump_1h(df4, regime, consolidation_bars=18,
                                                     pump_min_atr=3.0, pump_max_bars=6,
                                                     retrace_min=0.382, retrace_max=0.618,
                                                     tp_extension=1.0, sl_buffer_atr=0.5), sym)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


def run(name, syms, regime, tag, **kwargs):
    risk = kwargs.pop("risk", 0.03)
    max_pos = kwargs.pop("max_pos", 5)
    sigs, sim_dfs = collect(syms, regime, tag, **kwargs)
    if not sigs: return None
    res = simulate_portfolio(sigs, sim_dfs, regime,
                              start_balance=START_BALANCE,
                              risk=risk, max_pos=max_pos)
    m = res["metrics"]
    rets = [mr["ret_pct"] for mr in res["monthly_returns"]]
    return {"name": name, "metrics": m, "rets": rets,
            "trades": res["trades"]}


def main():
    tag = "365d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    BASELINE = ["ADA","ALGO","APT","ARB","ATOM","AVAX","BCH","BNB","BTC",
                "DOGE","DOT","ETH","FIL","HBAR","ICP","INJ","LINK","LTC",
                "NEAR","OP","SOL","SUI","TON","TRX","UNI","XLM","XRP"]
    avail = list_available_symbols(tag, ("1d","4h","1h"))
    BASELINE = [s for s in BASELINE if s in avail]

    results = []
    # Vol threshold grid × risk grid
    for vt in [1.0, 1.2, 1.5, 1.7, 2.0]:
        for risk in [0.02, 0.03, 0.04, 0.05]:
            for vw in [10, 20, 30]:
                r = run(f"VOL{vt}_R{int(risk*100)}_W{vw}", BASELINE, regime, tag,
                          vol_thresh=vt, vol_window=vw, risk=risk)
                if r: results.append(r)

    # Print sorted by return
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V5 VOLUME FILTER × RISK GRID  (BASELINE 27, V4 TPs, 365d)")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Config':22s} {'n':>4s} {'PF':>5s} {'WR%':>5s} {'avgR':>6s} {'Return':>8s} {'DD%':>5s} {'Avg/mo':>7s} {'Win-mo':>7s}")
    results.sort(key=lambda r: -r["metrics"]["return_pct"])
    for r in results[:30]:
        m = r["metrics"]
        avgmo = float(np.mean(r["rets"])) if r["rets"] else 0
        winmo = sum(1 for x in r["rets"] if x > 0) if r["rets"] else 0
        totmo = len(r["rets"]) if r["rets"] else 0
        print(f"  {r['name']:22s} {m['n']:>4d} {m['pf']:>5.2f} {m['wr']:>5.1f} "
              f"{m['avg_r']:>+5.2f} {m['return_pct']:>+7.1f}% {m['max_dd_pct']:>4.1f}% "
              f"{avgmo:>+6.1f}% {winmo}/{totmo}")

    # Output the per-strategy breakdown of the absolute winner
    if results:
        winner = results[0]
        print(f"\n══════ WINNER: {winner['name']} ══════")
        per = defaultdict(list)
        for t in winner["trades"]: per[t["strat"]].append(t)
        for sn, tlist in sorted(per.items()):
            mm = metrics_dict_from_trades(tlist, START_BALANCE)
            print(f"  {sn:25s}: n={mm.get('n',0):4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f} sumR={mm.get('sum_r',0):+7.1f}")
        print(f"\n  Monthly returns:")
        # Recompute monthlies
        # Already in winner["rets"] but no month names here
        m = winner["metrics"]
        print(f"  Final: ${m['final_balance']:,.2f}  Return: {m['return_pct']:+.1f}%  DD: {m['max_dd_pct']:.1f}%")
        print(f"  Avg/mo: {np.mean(winner['rets']):+.1f}%   Med: {np.median(winner['rets']):+.1f}%")


if __name__ == "__main__":
    main()
