"""
alloc_optimizer.py -- Find best capital allocation across 4 strategies.

Uses pre-built signals from portfolio_monthly.py infra.
Tests ~100 allocation combos, reports Calmar / PnL / DD.
"""
from __future__ import annotations
import sys, itertools
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from multi_agent.backtests.backtest_v10 import (
    load_all_tf, prepare_tf_arrays, prepare_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
    compute_sl,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
)

STRATEGIES = [
    {
        "name": "BBS_5m",
        "htf_key": "df1h", "ltf_key": "df5", "gen_fn": "bbs",
        "sig_params":   {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                         "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 3.0},
        "daily_filter": True,
        "max_pos": 4,
    },
    {
        "name": "BBS_15m",
        "htf_key": "df1h", "ltf_key": "df15", "gen_fn": "bbs",
        "sig_params":   {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                         "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
        "max_pos": 4,
    },
    {
        "name": "KC_15m",
        "htf_key": "df1h", "ltf_key": "df15", "gen_fn": "kc",
        "sig_params":   {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                         "mom_bars": 6, "sl_atr_mult": 1.5},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.0},
        "daily_filter": False,
        "max_pos": 4,
    },
    {
        "name": "SwKC",
        "htf_key": "df4h", "ltf_key": "df15", "gen_fn": "kc",
        "sig_params":   {"min_squeeze": 8, "adx_min": 15, "vol_mult": 1.0,
                         "mom_bars": 6, "sl_atr_mult": 2.0},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 2.0},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
        "max_pos": 2,
    },
]


# ── Build signals once ────────────────────────────────────────────────────────

def build_signals(all_data):
    strat_trades = {}   # name -> list of {ts, pnl_pct, won}
    for strat in STRATEGIES:
        name = strat["name"]
        gen_fn = gen_bbs_signals_tf if strat["gen_fn"] == "bbs" else gen_kc_squeeze_signals_tf
        all_sigs = {}
        ltf_arrs = {}
        for sym, d in all_data.items():
            htf_arr = prepare_tf_arrays(d.get(strat["htf_key"]))
            ltf_arr = prepare_tf_arrays(d.get(strat["ltf_key"]))
            if htf_arr is None or ltf_arr is None:
                continue
            ltf_arrs[sym] = ltf_arr
            htf_trend = prepare_tf_arrays(d.get("df4h") if strat["htf_key"] == "df1h" else d.get("df1h"))
            htf_sigs = gen_fn(htf_arr, htf_trend, strat["sig_params"])
            daily_d = prepare_daily_trend(d.get("df1d")) if strat["daily_filter"] else None
            mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, strat["entry_params"],
                                       daily_data=daily_d, use_daily_filter=strat["daily_filter"])
            all_sigs[sym] = mtf_sigs

        # Simulate with unit risk (risk_pt=1.0 = 100%) to get pure % returns per trade
        trades = _sim_unit(all_sigs, ltf_arrs, strat["sl_cfg"], strat["tp_cfg"],
                           strat["max_pos"])
        strat_trades[name] = trades
        print(f"  {name}: {len(trades)} trades")
    return strat_trades


def _sim_unit(signals_by_sym, tf_arrays_by_sym, sl_cfg, tp_cfg, max_pos):
    """Simulate and return list of {ts_ms, r_multiple} per trade."""
    balance = 1.0   # unit balance
    open_pos = {}
    acnt = 0
    results = []

    max_n = max((a["n"] for a in tf_arrays_by_sym.values() if a), default=0)
    if max_n == 0:
        return results

    for i in range(max(300, 0), max_n):
        for sym, arr in tf_arrays_by_sym.items():
            if arr is None or i >= arr["n"]:
                continue
            cl_i = arr["cl"][i]; hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            if sym in open_pos:
                pos = open_pos[sym]; d = pos["d"]; entry = pos["entry"]
                sl = pos["sl"]; r = pos["risk"]

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = False; tp_price = 0.0

                if not hit_sl:
                    pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)
                    if tp_cfg["type"] == "fixed_r" and pnl_r >= tp_cfg["r"]:
                        hit_tp = True; tp_price = cl_i

                if hit_sl or hit_tp:
                    ep = tp_price if hit_tp else sl
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    # Net R after fees (approximate: fee ~ 0.11% round trip = ~0.1%/entry)
                    fee_r = (TAKER_FEE + SLIPPAGE) * 2 * entry / r
                    net_r = exit_r - fee_r
                    results.append({
                        "ts":    int(arr["ts"][i]),
                        "r":     net_r,
                        "sym":   sym,
                    })
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            if acnt >= max_pos:
                continue
            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig = sym_sigs[i]
            new_sl = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                atr_i, arr["hi"], arr["lo"], i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue
            open_pos[sym] = {"d": sig["direction"], "entry": sig["entry_price"],
                             "sl": new_sl, "risk": rd}
            acnt += 1
    return results


# ── Portfolio sim from trade R-multiples ──────────────────────────────────────

def sim_portfolio(strat_trades, allocs, risk_pt=0.01):
    """
    allocs: dict {strat_name: fraction_of_capital}
    risk_pt: % of equity risked per trade
    Each trade's PnL = balance * risk_pt * alloc * R_multiple
    """
    # Merge all trades with strat label and sort by ts
    all_trades = []
    for name, trades in strat_trades.items():
        alloc = allocs.get(name, 0.0)
        if alloc <= 0:
            continue
        for t in trades:
            all_trades.append({"ts": t["ts"], "name": name, "r": t["r"], "alloc": alloc})
    all_trades.sort(key=lambda x: x["ts"])

    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0

    for t in all_trades:
        pnl = balance * risk_pt * t["alloc"] * t["r"]
        balance += pnl
        if balance > peak:
            peak = balance
        dd = (peak - balance) / peak * 100
        if dd > max_dd:
            max_dd = dd

    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100
    calmar  = pnl_pct / max_dd if max_dd > 0.01 else pnl_pct
    return {
        "final": round(balance, 0),
        "pnl_pct": round(pnl_pct, 1),
        "max_dd":  round(max_dd, 2),
        "calmar":  round(calmar, 1),
    }


# ── Grid search ───────────────────────────────────────────────────────────────

def grid_search(strat_trades, risk_pt=0.01):
    names = ["BBS_5m", "BBS_15m", "KC_15m", "SwKC"]

    # Allocations must sum to 1.0
    # Grid: each strategy gets 0%, 10%, 20%, ..., 80%
    results = []
    steps = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
             0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    for a, b, c in itertools.product(steps, steps, steps):
        d = round(1.0 - a - b - c, 2)
        if d < 0 or d > 0.80:
            continue
        allocs = {"BBS_5m": a, "BBS_15m": b, "KC_15m": c, "SwKC": d}
        res = sim_portfolio(strat_trades, allocs, risk_pt)
        res["allocs"] = (a, b, c, d)
        results.append(res)

    results.sort(key=lambda x: x["pnl_pct"], reverse=True)
    return results


def main():
    print("Allocation Optimizer -- 4 strategies, 365d")
    print("Loading data...")
    all_data = load_all_tf("365d")
    has_5m = any(d.get("df5") is not None and len(d["df5"]) > 1000 for d in all_data.values())
    print(f"Symbols: {len(all_data)},  5m: {has_5m}")
    if not has_5m:
        STRATEGIES[:] = [s for s in STRATEGIES if s["ltf_key"] != "df5"]

    print("\nBuilding strategy trade logs...")
    strat_trades = build_signals(all_data)

    total = sum(len(v) for v in strat_trades.values())
    print(f"Total trades across all strategies: {total}")

    # ── Quick check: individual strategy stats ────────────────────────────────
    print("\nIndividual strategy stats (100% alloc, 1% risk):")
    for name, trades in strat_trades.items():
        res = sim_portfolio(strat_trades, {name: 1.0}, risk_pt=0.01)
        wins = sum(1 for t in trades if t["r"] > 0)
        avg_r = sum(t["r"] for t in trades) / len(trades) if trades else 0
        print(f"  {name:<12}  n={len(trades):>3}  WR={wins/len(trades)*100:.0f}%"
              f"  avgR={avg_r:>+.2f}  PnL={res['pnl_pct']:>+.0f}%"
              f"  DD={res['max_dd']:.1f}%  Calmar={res['calmar']:.1f}")

    # ── Grid search with 1% risk ───────────────────────────────────────────────
    print("\nGrid searching allocations (1% risk per trade)...")
    grid = grid_search(strat_trades, risk_pt=0.01)

    # By max PnL
    print("\n--- Top 15 by MAX PnL ---")
    print(f"  {'BBS_5m':>6}  {'BBS_15m':>7}  {'KC_15m':>6}  {'SwKC':>5}  "
          f"{'PnL%':>7}  {'DD%':>5}  {'Calmar':>6}  {'$1k->':>7}")
    for r in grid[:15]:
        a, b, c, d = r["allocs"]
        print(f"  {a*100:>5.0f}%  {b*100:>6.0f}%  {c*100:>5.0f}%  {d*100:>4.0f}%  "
              f"  {r['pnl_pct']:>+6.0f}%  {r['max_dd']:>4.1f}%"
              f"  {r['calmar']:>6.0f}  ${r['final']:>6,.0f}")

    # By max Calmar (return/DD)
    grid_by_calmar = sorted(grid, key=lambda x: x["calmar"] if x["pnl_pct"] > 50 else -1, reverse=True)
    print("\n--- Top 15 by CALMAR ratio (PnL/DD) -- best risk-adjusted ---")
    print(f"  {'BBS_5m':>6}  {'BBS_15m':>7}  {'KC_15m':>6}  {'SwKC':>5}  "
          f"{'PnL%':>7}  {'DD%':>5}  {'Calmar':>6}  {'$1k->':>7}")
    for r in grid_by_calmar[:15]:
        a, b, c, d = r["allocs"]
        print(f"  {a*100:>5.0f}%  {b*100:>6.0f}%  {c*100:>5.0f}%  {d*100:>4.0f}%  "
              f"  {r['pnl_pct']:>+6.0f}%  {r['max_dd']:>4.1f}%"
              f"  {r['calmar']:>6.0f}  ${r['final']:>6,.0f}")

    # ── Specific named scenarios ───────────────────────────────────────────────
    scenarios = {
        "v10 original":     {"BBS_5m": 0.35, "BBS_15m": 0.30, "KC_15m": 0.20, "SwKC": 0.15},
        "BBS15m heavy":     {"BBS_5m": 0.15, "BBS_15m": 0.60, "KC_15m": 0.15, "SwKC": 0.10},
        "BBS15m dominant":  {"BBS_5m": 0.10, "BBS_15m": 0.70, "KC_15m": 0.15, "SwKC": 0.05},
        "BBS15m only":      {"BBS_5m": 0.00, "BBS_15m": 1.00, "KC_15m": 0.00, "SwKC": 0.00},
        "BBS5m only":       {"BBS_5m": 1.00, "BBS_15m": 0.00, "KC_15m": 0.00, "SwKC": 0.00},
        "equal 4 strats":   {"BBS_5m": 0.25, "BBS_15m": 0.25, "KC_15m": 0.25, "SwKC": 0.25},
        "equal 3 (no Sw)":  {"BBS_5m": 0.33, "BBS_15m": 0.34, "KC_15m": 0.33, "SwKC": 0.00},
        "BBS15m+5m 50/50":  {"BBS_5m": 0.50, "BBS_15m": 0.50, "KC_15m": 0.00, "SwKC": 0.00},
        "best grid found":  {"BBS_5m": grid[0]["allocs"][0], "BBS_15m": grid[0]["allocs"][1],
                             "KC_15m": grid[0]["allocs"][2], "SwKC": grid[0]["allocs"][3]},
    }

    print("\n--- Named scenarios comparison (1% risk) ---")
    print(f"  {'Scenario':<22}  {'5m':>4}  {'15m':>4}  {'KC':>4}  {'Sw':>4}  "
          f"{'PnL%':>7}  {'DD%':>5}  {'Calmar':>7}  {'$1k->':>7}")
    for label, allocs in scenarios.items():
        r = sim_portfolio(strat_trades, allocs, risk_pt=0.01)
        a = allocs.get("BBS_5m", 0)
        b = allocs.get("BBS_15m", 0)
        c = allocs.get("KC_15m", 0)
        d = allocs.get("SwKC", 0)
        print(f"  {label:<22}  {a*100:>3.0f}%  {b*100:>3.0f}%  {c*100:>3.0f}%  {d*100:>3.0f}%"
              f"  {r['pnl_pct']:>+6.0f}%  {r['max_dd']:>4.1f}%"
              f"  {r['calmar']:>7.0f}  ${r['final']:>6,.0f}")

    # ── Best for 2% risk ─────────────────────────────────────────────────────
    print("\n--- Same scenarios with 2% risk per trade ---")
    print(f"  {'Scenario':<22}  {'PnL%':>10}  {'DD%':>5}  {'$1k->':>10}")
    for label, allocs in scenarios.items():
        r = sim_portfolio(strat_trades, allocs, risk_pt=0.02)
        print(f"  {label:<22}  {r['pnl_pct']:>+9.0f}%  {r['max_dd']:>4.1f}%  ${r['final']:>9,.0f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
