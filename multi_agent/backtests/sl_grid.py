"""
sl_grid.py — Test wider SL/TP combinations on top strategies
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

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, sma, ema, rsi, supertrend, _ts_period_ms, _emit, Signal,
    simulate_signals, metrics, regime_at, SYMBOLS_730D,
)


def strat_supertrend_4h_v(df, regime, sl_atr, tp_r, st_p=10, st_mult=3.0):
    if df is None or len(df) < 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    st, sdir = supertrend(hi, lo, cl, st_p, st_mult)
    sigs = []
    for i in range(st_p + 14, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if sdir[i] == 1 and sdir[i-1] == -1:
            next_open = op[i+1]
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif sdir[i] == -1 and sdir[i-1] == 1:
            next_open = op[i+1]
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


def strat_smacross_1d_v(df, regime, sl_atr, tp_r):
    if df is None or len(df) < 220: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    s50 = sma(cl, 50); s200 = sma(cl, 200); a = atr(hi, lo, cl, 14)
    sigs = []
    for i in range(201, len(cl) - 1):
        if np.isnan(s50[i]) or np.isnan(s200[i]) or np.isnan(a[i]): continue
        if np.isnan(s50[i-1]) or np.isnan(s200[i-1]): continue
        next_open = op[i+1]
        if s50[i-1] <= s200[i-1] and s50[i] > s200[i]:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif s50[i-1] >= s200[i-1] and s50[i] < s200[i]:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


def strat_rsi_rev_4h_v(df, regime, sl_atr, tp_r, os=20, ob=80, regime_filter=True):
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
        if regime_filter and trend != 0: continue
        next_open = op[i+1]
        if r[i-1] <= os and r[i] > os:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif r[i-1] >= ob and r[i] < ob:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


def grid_test(strat_fn, strat_name, tf, sl_grid, tp_grid, regime, syms, tag="730d",
              extra_kwargs=None):
    extra_kwargs = extra_kwargs or {}
    rows = []
    for sl_atr in sl_grid:
        for tp_r in tp_grid:
            all_trades = []
            for sym in syms:
                df = load(sym, tf, tag)
                if df is None: continue
                sigs = strat_fn(df, regime, sl_atr, tp_r, **extra_kwargs)
                if sigs:
                    trades = simulate_signals(sym, strat_name, sigs, df, regime)
                    all_trades.extend(trades)
            m = metrics(all_trades)
            rows.append({"sl_atr": sl_atr, "tp_r": tp_r, **m})
    return rows


def print_grid(rows, title):
    print(f"\n{title}")
    print(f"  {'SL×ATR':>7s} {'TP×R':>5s} {'n':>5s} {'WR%':>6s} {'PF':>6s} {'avgR':>7s} {'sumR':>8s} {'ddR':>6s}")
    rows_sorted = sorted(rows, key=lambda r: -r["pf"] if r["pf"] != 99.0 else -r["sum_r"])
    for r in rows_sorted:
        if r["n"] == 0:
            print(f"  {r['sl_atr']:>5.1f}   {r['tp_r']:>4.1f}  {r['n']:>5d} {0:>5.1f}  {0:>5.2f}  {0:+6.3f}  {0:+7.1f}  {0:>5.1f}")
        else:
            print(f"  {r['sl_atr']:>5.1f}   {r['tp_r']:>4.1f}  {r['n']:>5d} {r['wr']:>5.1f}  {r['pf']:>5.2f}  "
                  f"{r['avg_r']:+6.3f}  {r['sum_r']:+7.1f}  {r['max_dd_r']:>5.1f}")


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = SYMBOLS_730D

    # supertrend_4h grid
    rows = grid_test(strat_supertrend_4h_v, "supertrend_4h", "4h",
                     sl_grid=[1.5, 2.0, 2.5, 3.0, 4.0],
                     tp_grid=[1.5, 2.0, 2.5, 3.0, 4.0],
                     regime=regime, syms=syms)
    print_grid(rows, "── supertrend_4h SL×TP grid (sorted by PF)")

    # smacross_1d grid
    rows = grid_test(strat_smacross_1d_v, "smacross_1d", "1d",
                     sl_grid=[2.0, 3.0, 4.0, 5.0],
                     tp_grid=[2.0, 3.0, 4.0, 5.0, 6.0],
                     regime=regime, syms=syms)
    print_grid(rows, "── smacross_1d SL×TP grid")

    # rsi_reversion_4h grid (KEY: 37.7% false stops → wider SL hypothesis)
    rows = grid_test(strat_rsi_rev_4h_v, "rsi_reversion_4h", "4h",
                     sl_grid=[1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
                     tp_grid=[1.0, 1.5, 2.0, 2.5, 3.0],
                     regime=regime, syms=syms)
    print_grid(rows, "── rsi_reversion_4h SL×TP grid (key for false-stop hypothesis)")


if __name__ == "__main__":
    main()
