"""
fib_grid.py — Optimize Fibonacci pump retracement parameters
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from multi_agent.backtests.strategy_lab import (
    load, build_regime, simulate_signals, metrics, SYMBOLS_730D,
)
from multi_agent.backtests.strategy_v3 import strat_fib_pump_1h


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = SYMBOLS_730D

    # Param grid for 4h
    grids = []
    for tf in ("1h", "4h"):
        if tf == "1h":
            cons_grid = [12, 24, 36]
            pump_max_bars = 8
        else:
            cons_grid = [12, 18, 24]
            pump_max_bars = 6
        for cons in cons_grid:
            for cons_max_atr in [1.5, 2.0, 2.5]:
                for pump_min_atr in [1.5, 2.0, 2.5, 3.0]:
                    for retrace_min, retrace_max in [(0.382, 0.618), (0.236, 0.618), (0.382, 0.786), (0.5, 0.786)]:
                        for tp_ext in [0.5, 1.0, 1.5]:
                            for sl_buf in [0.3, 0.5, 1.0]:
                                grids.append({
                                    "tf": tf, "cons": cons, "cons_max_atr": cons_max_atr,
                                    "pump_min_atr": pump_min_atr, "pump_max_bars": pump_max_bars,
                                    "retrace_min": retrace_min, "retrace_max": retrace_max,
                                    "tp_ext": tp_ext, "sl_buf": sl_buf,
                                })

    # Just sample a focused subset (full grid too big)
    focus_grid = []
    for tf in ("1h", "4h"):
        for cons in [12, 24] if tf == "1h" else [12, 18]:
            for pump_min_atr in [1.5, 2.0, 3.0]:
                for retrace_min, retrace_max in [(0.382, 0.618), (0.236, 0.786)]:
                    for tp_ext in [0.5, 1.0]:
                        for sl_buf in [0.3, 0.5]:
                            focus_grid.append({
                                "tf": tf, "cons": cons, "cons_max_atr": 2.0,
                                "pump_min_atr": pump_min_atr,
                                "pump_max_bars": 6 if tf == "4h" else 8,
                                "retrace_min": retrace_min, "retrace_max": retrace_max,
                                "tp_ext": tp_ext, "sl_buf": sl_buf,
                            })

    print(f"Grid size: {len(focus_grid)}")
    rows = []
    for g in focus_grid:
        all_trades = []
        for sym in syms:
            df = load(sym, g["tf"], "730d")
            if df is None: continue
            sigs = strat_fib_pump_1h(df, regime,
                                      sl_buffer_atr=g["sl_buf"],
                                      tp_extension=g["tp_ext"],
                                      consolidation_bars=g["cons"],
                                      consolidation_max_range_atr=g["cons_max_atr"],
                                      pump_min_atr=g["pump_min_atr"],
                                      pump_max_bars=g["pump_max_bars"],
                                      retrace_min=g["retrace_min"],
                                      retrace_max=g["retrace_max"],
                                      regime_filter=False)
            if sigs:
                trades = simulate_signals(sym, "fib", sigs, df, regime)
                all_trades.extend(trades)
        m = metrics(all_trades)
        rows.append({**g, **m})

    rows.sort(key=lambda r: -r["sum_r"])
    print(f"\n  {'TF':3s} {'cons':>4s} {'pump':>4s} {'rmin':>4s} {'rmax':>4s} {'tp_e':>4s} {'sl_b':>4s} {'n':>4s} {'WR%':>5s} {'PF':>5s} {'avgR':>6s} {'sumR':>7s}")
    for r in rows[:30]:
        print(f"  {r['tf']:3s} {r['cons']:>4d} {r['pump_min_atr']:>4.1f} {r['retrace_min']:>4.2f} {r['retrace_max']:>4.2f}"
              f" {r['tp_ext']:>4.1f} {r['sl_buf']:>4.1f}  {r['n']:>4d} {r['wr']:>4.1f} {r['pf']:>5.2f} {r['avg_r']:>+5.2f} {r['sum_r']:>+6.1f}")


if __name__ == "__main__":
    main()
