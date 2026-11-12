"""
tp_variants.py — Test TP/exit variations on top strategies
============================================================
Tested exit modes:
  • fixed_R       — pre-grid (1.5/2/2.5/3/4/5R)
  • partial_50_50 — 50% at 1R, 50% at N×R, residual SL→BE after partial
  • partial_33    — 33% at 1R, 33% at 2R, 33% at 4R (3 levels)
  • trailing_atr  — after +1R, SL trails at price - 1.5×ATR (LONG)
  • hybrid        — 50% at 1.5R partial + 50% trailing ATR

Each tested on smacross_pivot_1d and supertrend_pivot_4h on 730d.
Uses TAKER fees both sides + slippage + funding.
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
from dataclasses import dataclass

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, _ts_period_ms, regime_at,
    SYMBOLS_730D, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H, RISK_PER_TRADE,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d,
)


@dataclass
class ExitResult:
    sym: str
    strat: str
    direction: str
    entry_ts: int
    exit_ts: int
    pnl_r: float
    win: bool
    bars_held: int
    reason: str


def _fees_funding(entry_p, exit_p, qty, bars_held, period_ms):
    """Compute taker fees + funding for a fill segment."""
    funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))
    entry_fee = entry_p * qty * TAKER_FEE
    exit_fee  = exit_p  * qty * TAKER_FEE
    funding   = entry_p * qty * funding_per_bar * bars_held
    return entry_fee + exit_fee + funding


def simulate_exits(sym, strat_name, sigs, df, regime, mode="fixed_R", **kwargs):
    """Simulate signals with chosen exit mode."""
    if not sigs or df is None: return []
    ts = df["timestamp"].to_numpy()
    op = df["open"].to_numpy()
    hi = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    cl = df["close"].to_numpy()
    a = atr(hi, lo, cl, 14)
    n = len(ts)
    period_ms = _ts_period_ms(ts)
    out = []
    for sig in sigs:
        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1: continue
        d = sig.direction
        entry_p = op[j] * (1 + SLIPPAGE) if d == "LONG" else op[j] * (1 - SLIPPAGE)
        sl0 = sig.sl
        if d == "LONG":
            dist = entry_p - sl0
            if dist <= 0: continue
            tp_full = entry_p + (sig.tp - sig.entry) * (dist / sig.dist) if sig.dist > 0 else sig.tp
        else:
            dist = sl0 - entry_p
            if dist <= 0: continue
            tp_full = entry_p - (sig.entry - sig.tp) * (dist / sig.dist) if sig.dist > 0 else sig.tp

        # Use unit risk per trade so pnl_r is direct
        qty = 1.0 / dist  # so 1 R move = $1

        # ---------------------- MODE ROUTING ----------------------
        if mode == "fixed_R":
            tp_r_target = kwargs.get("tp_r", 3.0)
            if d == "LONG":
                tp_p = entry_p + tp_r_target * dist
            else:
                tp_p = entry_p - tp_r_target * dist
            exit_p, exit_bar, reason = _walk_exit_simple(
                d, j, n, hi, lo, cl, sl0, tp_p)
            if exit_p is None: continue
            slip_exit = exit_p * (1 - SLIPPAGE) if d == "LONG" else exit_p * (1 + SLIPPAGE)
            gross = (slip_exit - entry_p) * qty if d == "LONG" else (entry_p - slip_exit) * qty
            costs = _fees_funding(entry_p, slip_exit, qty, exit_bar - j + 1, period_ms)
            pnl = gross - costs
            out.append(ExitResult(sym, strat_name, d, int(ts[j]), int(ts[exit_bar]),
                                   pnl, pnl > 0, exit_bar - j + 1, reason))

        elif mode == "partial_50_50":
            # 50% at +1R, 50% at +N R, after partial1 SL moves to entry (BE)
            tp1_r = 1.0
            tp2_r = kwargs.get("tp2_r", 3.0)
            if d == "LONG":
                tp1 = entry_p + tp1_r * dist; tp2 = entry_p + tp2_r * dist
            else:
                tp1 = entry_p - tp1_r * dist; tp2 = entry_p - tp2_r * dist
            cur_sl = sl0
            partial1_done = False
            seg_pnl = 0.0
            cum_costs = 0.0
            exit_bar = None; reason = "timeout"
            for k in range(j, min(j + 500, n)):
                h_k = hi[k]; l_k = lo[k]
                if d == "LONG":
                    # SL hit
                    if l_k <= cur_sl:
                        # remaining qty exits at SL
                        rem_qty = qty * (0.5 if partial1_done else 1.0)
                        slip_exit = cur_sl * (1 - SLIPPAGE)
                        seg_pnl += (slip_exit - entry_p) * rem_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, rem_qty, k - j + 1, period_ms)
                        exit_bar = k; reason = "SL" if not partial1_done else "BE/SL"
                        break
                    # TP1 partial
                    if not partial1_done and h_k >= tp1:
                        partial_qty = qty * 0.5
                        slip_exit = tp1 * (1 - SLIPPAGE)
                        seg_pnl += (slip_exit - entry_p) * partial_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, partial_qty, k - j + 1, period_ms)
                        partial1_done = True
                        cur_sl = entry_p  # move to BE
                    # TP2 final
                    if h_k >= tp2:
                        rem_qty = qty * (0.5 if partial1_done else 1.0)
                        slip_exit = tp2 * (1 - SLIPPAGE)
                        seg_pnl += (slip_exit - entry_p) * rem_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, rem_qty, k - j + 1, period_ms)
                        exit_bar = k; reason = "TP2"
                        break
                else:  # SHORT
                    if h_k >= cur_sl:
                        rem_qty = qty * (0.5 if partial1_done else 1.0)
                        slip_exit = cur_sl * (1 + SLIPPAGE)
                        seg_pnl += (entry_p - slip_exit) * rem_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, rem_qty, k - j + 1, period_ms)
                        exit_bar = k; reason = "SL" if not partial1_done else "BE/SL"
                        break
                    if not partial1_done and l_k <= tp1:
                        partial_qty = qty * 0.5
                        slip_exit = tp1 * (1 + SLIPPAGE)
                        seg_pnl += (entry_p - slip_exit) * partial_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, partial_qty, k - j + 1, period_ms)
                        partial1_done = True
                        cur_sl = entry_p
                    if l_k <= tp2:
                        rem_qty = qty * (0.5 if partial1_done else 1.0)
                        slip_exit = tp2 * (1 + SLIPPAGE)
                        seg_pnl += (entry_p - slip_exit) * rem_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, rem_qty, k - j + 1, period_ms)
                        exit_bar = k; reason = "TP2"
                        break
            if exit_bar is None: continue
            pnl = seg_pnl - cum_costs
            out.append(ExitResult(sym, strat_name, d, int(ts[j]), int(ts[exit_bar]),
                                   pnl, pnl > 0, exit_bar - j + 1, reason))

        elif mode == "trailing_atr":
            # After price hits +1R, SL trails at high - 1.5*ATR(at trail bar)
            trigger_r = kwargs.get("trigger_r", 1.0)
            atr_mult = kwargs.get("atr_mult", 1.5)
            cur_sl = sl0
            trail_active = False
            best_price = entry_p
            exit_p = None; exit_bar = None; reason = "timeout"
            for k in range(j, min(j + 500, n)):
                h_k = hi[k]; l_k = lo[k]
                a_k = a[k] if not np.isnan(a[k]) else 0
                if d == "LONG":
                    if h_k > best_price:
                        best_price = h_k
                    if not trail_active and h_k >= entry_p + trigger_r * dist:
                        trail_active = True
                    if trail_active and a_k > 0:
                        new_sl = best_price - atr_mult * a_k
                        if new_sl > cur_sl:
                            cur_sl = new_sl
                    if l_k <= cur_sl:
                        exit_p = cur_sl; exit_bar = k
                        reason = "trail_SL" if trail_active else "SL"
                        break
                else:
                    if l_k < best_price:
                        best_price = l_k
                    if not trail_active and l_k <= entry_p - trigger_r * dist:
                        trail_active = True
                    if trail_active and a_k > 0:
                        new_sl = best_price + atr_mult * a_k
                        if new_sl < cur_sl:
                            cur_sl = new_sl
                    if h_k >= cur_sl:
                        exit_p = cur_sl; exit_bar = k
                        reason = "trail_SL" if trail_active else "SL"
                        break
            if exit_bar is None:
                exit_p = cl[min(j + 499, n - 1)]
                exit_bar = min(j + 499, n - 1)
            slip_exit = exit_p * (1 - SLIPPAGE) if d == "LONG" else exit_p * (1 + SLIPPAGE)
            gross = (slip_exit - entry_p) * qty if d == "LONG" else (entry_p - slip_exit) * qty
            costs = _fees_funding(entry_p, slip_exit, qty, exit_bar - j + 1, period_ms)
            pnl = gross - costs
            out.append(ExitResult(sym, strat_name, d, int(ts[j]), int(ts[exit_bar]),
                                   pnl, pnl > 0, exit_bar - j + 1, reason))

        elif mode == "hybrid":
            # 50% partial at 1.5R + 50% trailing
            partial_r = kwargs.get("partial_r", 1.5)
            atr_mult = kwargs.get("atr_mult", 1.5)
            if d == "LONG":
                tp_partial = entry_p + partial_r * dist
            else:
                tp_partial = entry_p - partial_r * dist
            cur_sl = sl0
            partial_done = False
            trail_active = False
            best_price = entry_p
            seg_pnl = 0.0
            cum_costs = 0.0
            exit_bar = None; reason = "timeout"
            for k in range(j, min(j + 500, n)):
                h_k = hi[k]; l_k = lo[k]
                a_k = a[k] if not np.isnan(a[k]) else 0
                if d == "LONG":
                    if h_k > best_price: best_price = h_k
                    if not partial_done and h_k >= tp_partial:
                        partial_qty = qty * 0.5
                        slip_exit = tp_partial * (1 - SLIPPAGE)
                        seg_pnl += (slip_exit - entry_p) * partial_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, partial_qty, k - j + 1, period_ms)
                        partial_done = True
                        cur_sl = entry_p  # move to BE
                        trail_active = True
                    if trail_active and a_k > 0:
                        new_sl = best_price - atr_mult * a_k
                        if new_sl > cur_sl:
                            cur_sl = new_sl
                    if l_k <= cur_sl:
                        rem_qty = qty * (0.5 if partial_done else 1.0)
                        slip_exit = cur_sl * (1 - SLIPPAGE)
                        seg_pnl += (slip_exit - entry_p) * rem_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, rem_qty, k - j + 1, period_ms)
                        exit_bar = k
                        reason = "trail" if partial_done else "SL"
                        break
                else:
                    if l_k < best_price: best_price = l_k
                    if not partial_done and l_k <= tp_partial:
                        partial_qty = qty * 0.5
                        slip_exit = tp_partial * (1 + SLIPPAGE)
                        seg_pnl += (entry_p - slip_exit) * partial_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, partial_qty, k - j + 1, period_ms)
                        partial_done = True
                        cur_sl = entry_p
                        trail_active = True
                    if trail_active and a_k > 0:
                        new_sl = best_price + atr_mult * a_k
                        if new_sl < cur_sl:
                            cur_sl = new_sl
                    if h_k >= cur_sl:
                        rem_qty = qty * (0.5 if partial_done else 1.0)
                        slip_exit = cur_sl * (1 + SLIPPAGE)
                        seg_pnl += (entry_p - slip_exit) * rem_qty
                        cum_costs += _fees_funding(entry_p, slip_exit, rem_qty, k - j + 1, period_ms)
                        exit_bar = k
                        reason = "trail" if partial_done else "SL"
                        break
            if exit_bar is None: continue
            pnl = seg_pnl - cum_costs
            out.append(ExitResult(sym, strat_name, d, int(ts[j]), int(ts[exit_bar]),
                                   pnl, pnl > 0, exit_bar - j + 1, reason))
    return out


def _walk_exit_simple(d, j, n, hi, lo, cl, sl, tp):
    for k in range(j, min(j + 500, n)):
        h = hi[k]; l = lo[k]
        if d == "LONG":
            if l <= sl: return sl, k, "SL"
            if h >= tp: return tp, k, "TP"
        else:
            if h >= sl: return sl, k, "SL"
            if l <= tp: return tp, k, "TP"
    return cl[min(j + 499, n - 1)], min(j + 499, n - 1), "timeout"


def metrics_r(trades):
    if not trades: return {"n": 0, "wr": 0, "pf": 0, "avg_r": 0, "sum_r": 0, "max_dd_r": 0}
    wins = [t for t in trades if t.pnl_r > 0]
    losses = [t for t in trades if t.pnl_r <= 0]
    gw = sum(t.pnl_r for t in wins)
    gl = abs(sum(t.pnl_r for t in losses))
    pf = gw / gl if gl > 0 else 99
    wr = len(wins) / len(trades) * 100
    sum_r = sum(t.pnl_r for t in trades)
    avg_r = sum_r / len(trades)
    eq = np.cumsum([t.pnl_r for t in sorted(trades, key=lambda x: x.entry_ts)])
    peak = np.maximum.accumulate(eq) if len(eq) else np.array([0])
    dd = (peak - eq).max() if len(eq) else 0
    return {"n": len(trades), "wr": round(wr, 1), "pf": round(pf, 2),
            "avg_r": round(avg_r, 3), "sum_r": round(sum_r, 1),
            "max_dd_r": round(dd, 1)}


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = SYMBOLS_730D

    # Generate signals once for each strategy
    print("Generating signals...")
    smacross_sigs = {}
    supertrend_sigs = {}
    for sym in syms:
        df1d = load(sym, "1d", "730d")
        df4h = load(sym, "4h", "730d")
        if df1d is not None:
            smacross_sigs[sym] = (strat_smacross_pivot_1d(df1d, regime, tp_r=4.0), df1d)
        if df4h is not None:
            supertrend_sigs[sym] = (strat_supertrend_pivot_4h(df4h, regime, tp_r=2.5), df4h)
    print(f"smacross syms: {len(smacross_sigs)}")
    print(f"supertrend syms: {len(supertrend_sigs)}")

    # Test each TP variant on each strategy
    test_cases = []

    # Fixed-R grid
    for tp_r in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        test_cases.append(("smacross_pivot_1d", "fixed_R", {"tp_r": tp_r}))
        test_cases.append(("supertrend_pivot_4h", "fixed_R", {"tp_r": tp_r}))

    # Partial 50/50 with various TP2
    for tp2_r in [2.0, 2.5, 3.0, 4.0]:
        test_cases.append(("smacross_pivot_1d", "partial_50_50", {"tp2_r": tp2_r}))
        test_cases.append(("supertrend_pivot_4h", "partial_50_50", {"tp2_r": tp2_r}))

    # Trailing
    for trigger_r in [0.5, 1.0, 1.5]:
        for atr_mult in [1.5, 2.5, 3.0]:
            test_cases.append(("smacross_pivot_1d", "trailing_atr",
                                {"trigger_r": trigger_r, "atr_mult": atr_mult}))
            test_cases.append(("supertrend_pivot_4h", "trailing_atr",
                                {"trigger_r": trigger_r, "atr_mult": atr_mult}))

    # Hybrid
    for partial_r in [1.0, 1.5, 2.0]:
        for atr_mult in [1.5, 2.5]:
            test_cases.append(("smacross_pivot_1d", "hybrid",
                                {"partial_r": partial_r, "atr_mult": atr_mult}))
            test_cases.append(("supertrend_pivot_4h", "hybrid",
                                {"partial_r": partial_r, "atr_mult": atr_mult}))

    print(f"\nRunning {len(test_cases)} test combinations...")
    results = []
    for strat, mode, kwargs in test_cases:
        all_trades = []
        sig_map = smacross_sigs if strat == "smacross_pivot_1d" else supertrend_sigs
        for sym, (sigs, df) in sig_map.items():
            trades = simulate_exits(sym, strat, sigs, df, regime, mode=mode, **kwargs)
            all_trades.extend(trades)
        m = metrics_r(all_trades)
        results.append({"strat": strat, "mode": mode, "kwargs": kwargs, **m})

    # Print sorted by pf for each strategy
    for strat in ("smacross_pivot_1d", "supertrend_pivot_4h"):
        print(f"\n══════════════════════════════════════════════════════════════════")
        print(f"  {strat}  — TP variants sorted by sumR")
        print(f"══════════════════════════════════════════════════════════════════")
        rows = [r for r in results if r["strat"] == strat]
        rows.sort(key=lambda r: -r["sum_r"])
        print(f"  {'Mode':16s} {'Params':35s} {'n':>4s} {'WR%':>5s} {'PF':>5s} {'avgR':>6s} {'sumR':>7s} {'ddR':>5s}")
        for r in rows:
            kw = ", ".join(f"{k}={v}" for k, v in r["kwargs"].items())
            print(f"  {r['mode']:16s} {kw:35s} {r['n']:>4d} {r['wr']:>5.1f} {r['pf']:>5.2f} "
                  f"{r['avg_r']:>+5.2f} {r['sum_r']:>+6.1f} {r['max_dd_r']:>5.1f}")


if __name__ == "__main__":
    main()
