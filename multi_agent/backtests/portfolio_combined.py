"""
portfolio_combined.py -- Shared-capital portfolio sim, 1% and 2% risk per trade.

All 4 strategies share one $1000 capital pool.
No allocations -- every trade risks risk_pt% of current equity.
Max concurrent positions = max_pos (portfolio-wide).
Monthly PnL breakdown.
"""
from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.backtest_v10 import (
    load_all_tf, prepare_tf_arrays, prepare_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
    compute_sl,
    SYMBOLS, START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
)

# ── Strategy configs (v10 best params) ───────────────────────────────────────
STRATEGIES = [
    {
        "name": "BBS_1H_5M",
        "htf_key": "df1h", "ltf_key": "df5",
        "gen_fn": "bbs",
        "sig_params":   {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                         "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 3.0},
        "daily_filter": True,
    },
    {
        "name": "BBS_1H_15M",
        "htf_key": "df1h", "ltf_key": "df15",
        "gen_fn": "bbs",
        "sig_params":   {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                         "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
    },
    {
        "name": "KC_1H_15M",
        "htf_key": "df1h", "ltf_key": "df15",
        "gen_fn": "kc",
        "sig_params":   {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                         "mom_bars": 6, "sl_atr_mult": 1.5},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.0},
        "daily_filter": False,
    },
    {
        "name": "SwKC_4H_15M",
        "htf_key": "df4h", "ltf_key": "df15",
        "gen_fn": "kc",
        "sig_params":   {"min_squeeze": 8, "adx_min": 15, "vol_mult": 1.0,
                         "mom_bars": 6, "sl_atr_mult": 2.0},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 2.0},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
    },
]


def build_all_signals(all_data):
    """Returns dict: (strat_name, sym) -> {bar_i: sig_dict}."""
    all_sigs = {}
    ltf_cache = {}

    for strat in STRATEGIES:
        name    = strat["name"]
        gen_fn  = gen_bbs_signals_tf if strat["gen_fn"] == "bbs" else gen_kc_squeeze_signals_tf
        htf_key = strat["htf_key"]
        ltf_key = strat["ltf_key"]

        for sym, d in all_data.items():
            htf_arr = prepare_tf_arrays(d.get(htf_key))
            ltf_arr = prepare_tf_arrays(d.get(ltf_key))
            if htf_arr is None or ltf_arr is None:
                continue

            # Cache ltf arrays per (sym, ltf_key) to avoid re-preparing
            cache_key = (sym, ltf_key)
            if cache_key not in ltf_cache:
                ltf_cache[cache_key] = ltf_arr
            ltf_arr = ltf_cache[cache_key]

            htf_trend = prepare_tf_arrays(d.get("df4h") if htf_key == "df1h" else d.get("df1h"))
            htf_sigs  = gen_fn(htf_arr, htf_trend, strat["sig_params"])

            daily_d = prepare_daily_trend(d.get("df1d")) if strat["daily_filter"] else None
            mtf_sigs = gen_mtf_signals(
                htf_sigs, htf_arr, ltf_arr, strat["entry_params"],
                daily_data=daily_d, use_daily_filter=strat["daily_filter"]
            )
            if mtf_sigs:
                all_sigs[(name, sym)] = mtf_sigs

    return all_sigs, ltf_cache


def run_combined(all_sigs, ltf_cache, risk_pt=0.010, max_pos=6):
    """
    Shared-capital portfolio sim.
    - One equity pool ($1000 start)
    - Each trade risks risk_pt * current_equity
    - max_pos concurrent positions across ALL strategies + symbols
    """
    balance = START_BALANCE
    peak    = START_BALANCE
    open_pos = {}   # key: (strat_name, sym) -> pos dict
    completed = []

    # Build sorted event list: (bar_i, strat_name, sym)
    # We iterate by bar index. Need a common time axis.
    # Use 15m bars as base (most strategies use it), 5m where needed.

    # Collect all ltf arrays keyed by (strat_name, sym)
    strat_arrays = {}
    for (name, sym), sigs in all_sigs.items():
        strat = next(s for s in STRATEGIES if s["name"] == name)
        ltf_key = strat["ltf_key"]
        arr = ltf_cache.get((sym, ltf_key))
        if arr is not None:
            strat_arrays[(name, sym)] = arr

    # Find max bar count
    max_n = max((a["n"] for a in strat_arrays.values()), default=0)
    if max_n == 0:
        return [], 0.0

    # We'll iterate by 15m-equivalent index using timestamp
    # Sort all signals into a flat list: (ts_ms, strat, sym, bar_i)
    events = []
    for (name, sym), sigs in all_sigs.items():
        arr = strat_arrays.get((name, sym))
        if arr is None:
            continue
        for bar_i, sig in sigs.items():
            if bar_i < arr["n"]:
                ts = int(arr["ts"][bar_i])
                events.append((ts, name, sym, bar_i, sig))
    events.sort(key=lambda x: x[0])

    # Also need to check open positions at every bar of their ltf
    # Build per-(strat,sym) timeline for position management
    # We'll do a bar-by-bar loop on all strat_arrays sorted by timestamp

    # Simpler approach: iterate events in timestamp order
    # For each event, first manage open positions at that timestamp
    # then try to open new position

    # Collect all timestamps from all ltf arrays
    all_ts_set = set()
    for arr in strat_arrays.values():
        for ts in arr["ts"]:
            all_ts_set.add(int(ts))
    all_ts_sorted = sorted(all_ts_set)

    # Build lookup: ts -> list of (name, sym, bar_i)
    ts_to_bars = defaultdict(list)
    for (name, sym), arr in strat_arrays.items():
        for i, ts in enumerate(arr["ts"]):
            ts_to_bars[int(ts)].append((name, sym, i))

    # Signal lookup
    sig_lookup = {}
    for (name, sym), sigs in all_sigs.items():
        for bar_i, sig in sigs.items():
            arr = strat_arrays.get((name, sym))
            if arr is not None and bar_i < arr["n"]:
                ts = int(arr["ts"][bar_i])
                sig_lookup[(ts, name, sym)] = (bar_i, sig)

    warmup_ts = all_ts_sorted[min(300, len(all_ts_sorted)-1)]

    for ts in all_ts_sorted:
        if ts < warmup_ts:
            continue

        bars_at_ts = ts_to_bars[ts]

        # ── Manage open positions ─────────────────────────────────────────
        to_close = []
        for pos_key, pos in open_pos.items():
            p_name, p_sym = pos_key
            arr = strat_arrays.get(pos_key)
            if arr is None:
                continue
            # Find bar index for this timestamp in this strat's ltf
            idx = int(np.searchsorted(arr["ts"], ts, side="left"))
            if idx >= arr["n"]:
                continue
            if arr["ts"][idx] != ts:
                continue

            cl_i  = arr["cl"][idx]
            hi_i  = arr["hi"][idx]
            lo_i  = arr["lo"][idx]
            atr_i = arr["atr"][idx]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            d      = pos["d"]
            entry  = pos["entry"]
            sl     = pos["sl"]
            r      = pos["risk"]
            q      = pos["qty"]
            strat  = next(s for s in STRATEGIES if s["name"] == p_name)
            tp_cfg = strat["tp_cfg"]

            pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)
            hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
            hit_tp = False
            tp_price = 0.0
            reason = ""

            if not hit_sl:
                if tp_cfg["type"] == "fixed_r" and pnl_r >= tp_cfg["r"]:
                    hit_tp = True
                    tp_price = cl_i
                    reason = "tp"
            else:
                reason = "sl"

            if hit_sl or hit_tp:
                ep    = tp_price if hit_tp else sl
                gross = ((ep - entry) if d == "LONG" else (entry - ep)) * q
                fee   = ep * q * (TAKER_FEE + SLIPPAGE)
                net   = gross - fee
                to_close.append((pos_key, net, reason, ts, p_name))

        for pos_key, net, reason, exit_ts, strat_name in to_close:
            balance += net
            peak = max(peak, balance)
            dd = (peak - balance) / peak * 100
            completed.append({
                "ts":     exit_ts,
                "strat":  strat_name,
                "pnl":    round(net, 4),
                "reason": reason,
                "won":    net > 0,
                "dd_pct": round(dd, 2),
            })
            del open_pos[pos_key]

        # ── Open new positions ────────────────────────────────────────────
        for name, sym, bar_i in bars_at_ts:
            if len(open_pos) >= max_pos:
                break
            pos_key = (name, sym)
            if pos_key in open_pos:
                continue

            sig_key = (ts, name, sym)
            if sig_key not in sig_lookup:
                continue
            bar_i_sig, sig = sig_lookup[sig_key]

            arr = strat_arrays[pos_key]
            atr_i = arr["atr"][bar_i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = arr["cl"][bar_i] * 0.01

            strat = next(s for s in STRATEGIES if s["name"] == name)
            new_sl = compute_sl(strat["sl_cfg"], sig["direction"],
                                sig["entry_price"], atr_i,
                                arr["hi"], arr["lo"], bar_i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue

            # Risk-based sizing on SHARED equity
            ru  = balance * risk_pt
            qty = ru / rd
            mg  = sig["entry_price"] * qty / MAX_LEVERAGE
            if mg > balance * 0.4:
                qty = balance * 0.4 * MAX_LEVERAGE / sig["entry_price"]
            if qty * sig["entry_price"] < 5.0:
                continue

            fee = sig["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
            balance -= fee
            open_pos[pos_key] = {
                "d":      sig["direction"],
                "entry":  sig["entry_price"],
                "sl":     new_sl,
                "risk":   rd,
                "qty":    qty,
                "ts_in":  ts,
            }

    return completed, balance


def month_key(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m")


def print_monthly(trades, final_bal, risk_label):
    if not trades:
        print("No trades.")
        return

    monthly = defaultdict(lambda: {"pnl": 0.0, "n": 0, "wins": 0,
                                   "by_strat": defaultdict(float)})
    eq = START_BALANCE
    month_end_eq = {}

    for t in sorted(trades, key=lambda x: x["ts"]):
        mk = month_key(t["ts"])
        monthly[mk]["pnl"]  += t["pnl"]
        monthly[mk]["n"]    += 1
        if t["won"]:
            monthly[mk]["wins"] += 1
        monthly[mk]["by_strat"][t["strat"]] += t["pnl"]

    months = sorted(monthly.keys())

    # running equity per month
    eq = START_BALANCE
    for mk in months:
        monthly[mk]["eq_start"] = eq
        eq += monthly[mk]["pnl"]
        monthly[mk]["eq_end"] = eq

    # peak / DD
    peak = START_BALANCE
    for mk in months:
        peak = max(peak, monthly[mk]["eq_end"])
        monthly[mk]["peak"] = peak
        dd = (peak - monthly[mk]["eq_end"]) / peak * 100
        monthly[mk]["dd"] = dd

    strat_names = ["BBS_1H_5M", "BBS_1H_15M", "KC_1H_15M", "SwKC_4H_15M"]
    strat_short  = ["BBS_5m",   "BBS_15m",    "KC_15m",    "SwKC"]

    hdr = (f"{'Month':<8}  {'PnL$':>7}  {'PnL%':>6}  {'WR':>5}  "
           f"{'N':>3}  {'Equity':>8}  {'DD%':>5}  "
           + "  ".join(f"{s:>7}" for s in strat_short))
    sep = "=" * len(hdr)

    print(f"\n{'='*30}  Risk {risk_label} per trade  {'='*30}")
    print(sep)
    print(hdr)
    print(sep)

    total_pnl = 0; total_n = 0; total_wins = 0
    for mk in months:
        m = monthly[mk]
        pnl_pct = m["pnl"] / m["eq_start"] * 100 if m["eq_start"] > 0 else 0
        wr = m["wins"] / m["n"] * 100 if m["n"] > 0 else 0
        flag = "v" if m["pnl"] < 0 else " "
        by_s = [m["by_strat"].get(n, 0.0) for n in strat_names]
        cols = "  ".join(f"{v:>+7.1f}" for v in by_s)
        print(f"{flag}{mk:<7}  {m['pnl']:>+7.1f}  {pnl_pct:>+5.1f}%  "
              f"{wr:>4.0f}%  {m['n']:>3}  ${m['eq_end']:>7.0f}  "
              f"{m['dd']:>4.1f}%  {cols}")
        total_pnl  += m["pnl"]
        total_n    += m["n"]
        total_wins += m["wins"]

    print(sep)
    wr_total = total_wins / total_n * 100 if total_n else 0
    pnl_pct_total = total_pnl / START_BALANCE * 100
    print(f"{'TOTAL':<8}  {total_pnl:>+7.1f}  {pnl_pct_total:>+5.1f}%  "
          f"{wr_total:>4.0f}%  {total_n:>3}  ${final_bal:>7.0f}")
    print(f"\n  $1,000 -> ${final_bal:,.0f}  (+{pnl_pct_total:.0f}%)")

    # Worst / best
    sorted_m = sorted(months, key=lambda mk: monthly[mk]["pnl"])
    print(f"\n  Worst months:")
    for mk in sorted_m[:3]:
        m = monthly[mk]
        pct = m["pnl"] / m["eq_start"] * 100
        print(f"    {mk}  {m['pnl']:>+8.1f}$  ({pct:>+5.1f}%)  DD={m['dd']:.1f}%")
    print(f"  Best months:")
    for mk in sorted_m[-3:][::-1]:
        m = monthly[mk]
        pct = m["pnl"] / m["eq_start"] * 100
        print(f"    {mk}  {m['pnl']:>+8.1f}$  ({pct:>+5.1f}%)")

    loss_months = [mk for mk in months if monthly[mk]["pnl"] < 0]
    max_dd = max(monthly[mk]["dd"] for mk in months)
    print(f"\n  Loss months: {len(loss_months)}/{len(months)}"
          f"  |  Max DD from peak: {max_dd:.1f}%")

    # Strategy contribution
    print(f"\n  Strategy contributions:")
    for sn, ss in zip(strat_names, strat_short):
        c = sum(t["pnl"] for t in trades if t["strat"] == sn)
        n = sum(1 for t in trades if t["strat"] == sn)
        pct = c / total_pnl * 100 if total_pnl else 0
        print(f"    {ss:<12}  {c:>+8.1f}$  ({pct:>+5.1f}%)  {n} trades")


def main():
    print("Portfolio Combined Sim -- shared capital, no allocation fractions")
    print("Loading data (365d)...")
    all_data = load_all_tf("365d")
    has_5m = any(d.get("df5") is not None and len(d["df5"]) > 1000 for d in all_data.values())
    print(f"Symbols: {len(all_data)},  5m: {has_5m}")

    if not has_5m:
        # Remove 5m strategy if no data
        STRATEGIES[:] = [s for s in STRATEGIES if s["ltf_key"] != "df5"]

    print("Building signals for all strategies...")
    all_sigs, ltf_cache = build_all_signals(all_data)
    total_sigs = sum(len(v) for v in all_sigs.values())
    print(f"Total signals: {total_sigs}")
    for (name, sym) in list(all_sigs.keys())[:3]:
        print(f"  sample: {name} {sym.split('/')[0]} -> {len(all_sigs[(name,sym)])} sigs")

    # Run with 1% risk
    print("\nRunning 1% risk simulation...")
    trades_1pct, final_1pct = run_combined(all_sigs, ltf_cache, risk_pt=0.010, max_pos=8)
    print_monthly(trades_1pct, final_1pct, "1%")

    # Run with 2% risk
    print("\nRunning 2% risk simulation...")
    trades_2pct, final_2pct = run_combined(all_sigs, ltf_cache, risk_pt=0.020, max_pos=8)
    print_monthly(trades_2pct, final_2pct, "2%")

    # Side by side summary
    pnl_1 = (final_1pct - START_BALANCE) / START_BALANCE * 100
    pnl_2 = (final_2pct - START_BALANCE) / START_BALANCE * 100
    print(f"\n{'='*50}")
    print(f"  Risk 1%:  $1,000 -> ${final_1pct:>7,.0f}  (+{pnl_1:.0f}%)")
    print(f"  Risk 2%:  $1,000 -> ${final_2pct:>7,.0f}  (+{pnl_2:.0f}%)")


if __name__ == "__main__":
    main()
