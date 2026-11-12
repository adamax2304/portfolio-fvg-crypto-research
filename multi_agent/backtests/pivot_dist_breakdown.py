"""
pivot_dist_breakdown.py — Win/Loss breakdown of removed trades by pivot-distance filter

Shows exactly which trades were cut, whether they were winners or losers,
and the resulting performance delta.
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

from multi_agent.backtests.strategy_lab import load, build_regime, atr as _atr
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import load_all_events, build_sentiment_series, simulate_news_aware
from multi_agent.backtests.filter_tests_v7 import simulate_filtered
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter

MAX_DIST = 7.0   # лучший порог из предыдущего теста


def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print(f"  PIVOT-DIST FILTER — Win/Loss breakdown  (threshold = {MAX_DIST}×ATR)")
    print("══════════════════════════════════════════════════════════════════════\n")

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"  Symbols: {len(syms)}  |  Loading signals...")

    sigs_raw, sim_dfs, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    events = load_all_events()
    print(f"  Signals: {len(sigs_raw)}  |  Events: {len(events)}\n")

    # ── Run both simulations ─────────────────────────────────────────────────
    sigs_filt, removed_count, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, MAX_DIST)

    res_base = simulate_filtered(sigs_raw, sim_dfs, regime, events,
                                  base_risk=0.04, enable_l4=True, range_bars=0)
    res_filt = simulate_filtered(sigs_filt, sim_dfs, regime, events,
                                  base_risk=0.04, enable_l4=True, range_bars=0)

    trades_base = res_base["trades"]
    trades_filt = res_filt["trades"]
    mb = res_base["metrics"]
    mf = res_filt["metrics"]

    # ── Find removed trades by matching entry_ts + sym ───────────────────────
    # Key = (sym, entry_ts) — unique per trade
    base_keys = {(t["sym"], t["entry_ts"]): t for t in trades_base}
    filt_keys = {(t["sym"], t["entry_ts"]) for t in trades_filt}
    removed_trades = [t for k, t in base_keys.items() if k not in filt_keys]

    # ── Breakdown ────────────────────────────────────────────────────────────
    wins   = [t for t in removed_trades if t.get("win")]
    losses = [t for t in removed_trades if not t.get("win")]

    total_r_wins   = sum(t["pnl_r"] for t in wins)
    total_r_losses = sum(t["pnl_r"] for t in losses)
    total_r_all    = sum(t["pnl_r"] for t in removed_trades)

    print("──────────────────────────────────────────────────────────────────────")
    print("  Removed trades breakdown")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  Total signals removed : {removed_count}")
    print(f"  Trades removed        : {len(removed_trades)}")
    print(f"  ├─ Winners removed    : {len(wins)}   (sum R = {total_r_wins:+.2f}R)")
    print(f"  └─ Losers removed     : {len(losses)}  (sum R = {total_r_losses:+.2f}R)")
    print(f"  Net R removed         : {total_r_all:+.2f}R")

    # ── Detail table of removed trades ──────────────────────────────────────
    if removed_trades:
        print(f"\n  Detail of removed trades:")
        print(f"  {'Sym':6s} {'Strat':12s} {'Dir':5s} {'Entry date':12s} "
              f"{'Dist ATR':>9s} {'pnl_R':>7s} {'W/L':>5s} {'Reason':>12s}")
        print("  " + "─" * 78)

        # Compute dist_atr for each removed trade so we can show it
        atr_cache = {}
        for t in sorted(removed_trades, key=lambda x: x["entry_ts"]):
            sym, strat = t["sym"], t["strat"]
            key = (sym, strat)
            df  = sim_dfs.get(key)
            dist_str = "n/a"
            if df is not None:
                if key not in atr_cache:
                    atr_cache[key] = _atr(
                        df["high"].to_numpy(float),
                        df["low"].to_numpy(float),
                        df["close"].to_numpy(float), 14
                    )
                a_arr  = atr_cache[key]
                ts_arr = df["timestamp"].to_numpy(int)
                # entry_ts is next_open_ts; signal ts is one bar before
                idx = int(np.searchsorted(ts_arr, t["entry_ts"], side="left")) - 1
                if 0 <= idx < len(a_arr) and not np.isnan(a_arr[idx]) and a_arr[idx] > 0:
                    # dist = SL distance (entry - sl) / atr
                    sl = t.get("sl", None)
                    entry = t.get("entry", None)
                    if sl is not None and entry is not None:
                        dist = abs(entry - sl) / a_arr[idx]
                        dist_str = f"{dist:.2f}×"

            dt = pd.Timestamp(t["entry_ts"], unit="ms", tz="UTC").strftime("%Y-%m-%d")
            wl = "WIN" if t.get("win") else "LOSS"
            reason = t.get("reason", "")[:12]
            print(f"  {sym:6s} {strat:12s} {t.get('direction','?'):5s} {dt:12s} "
                  f"{dist_str:>9s} {t['pnl_r']:>+7.2f}R {wl:>5s} {reason:>12s}")

    # ── Performance comparison ────────────────────────────────────────────────
    print(f"\n──────────────────────────────────────────────────────────────────────")
    print(f"  Performance comparison")
    print(f"──────────────────────────────────────────────────────────────────────")
    print(f"  {'Metric':20s} {'BASELINE':>12s} {'FILTERED':>12s} {'Delta':>10s}")
    print(f"  {'':20s} {'(no filter)':>12s} {f'(>{MAX_DIST}×ATR cut)':>12s}")
    print("  " + "─" * 58)

    metrics = [
        ("Trades (n)",     mb["n"],            mf["n"],            "d"),
        ("Win rate %",     mb["wr"],            mf["wr"],            "f1"),
        ("Profit factor",  mb["pf"],            mf["pf"],            "f2"),
        ("Avg R",          mb["avg_r"],          mf["avg_r"],          "f2"),
        ("Return %",       mb["return_pct"],     mf["return_pct"],     "f1"),
        ("Max DD %",       mb["max_dd_pct"],     mf["max_dd_pct"],     "f1"),
    ]
    for name, bv, fv, fmt in metrics:
        delta = fv - bv
        if fmt == "d":
            print(f"  {name:20s} {bv:>12d} {fv:>12d} {delta:>+10d}")
        elif fmt == "f1":
            print(f"  {name:20s} {bv:>12.1f} {fv:>12.1f} {delta:>+10.1f}")
        else:
            print(f"  {name:20s} {bv:>12.2f} {fv:>12.2f} {delta:>+10.2f}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════════════")
    print(f"  ВЫВОД")
    print(f"══════════════════════════════════════════════════════════════════════")
    print(f"\n  Фильтр >{MAX_DIST}×ATR удалил {len(removed_trades)} трейдов из симуляции:")
    print(f"    • {len(losses)} убыточных  → суммарно {total_r_losses:+.2f}R  (ЭТО МЫ ИЗБЕЖАЛИ)")
    print(f"    • {len(wins)} прибыльных → суммарно {total_r_wins:+.2f}R  (ЭТО МЫ ПОТЕРЯЛИ)")
    print(f"    • Чистый выигрыш от фильтра: {-total_r_all:+.2f}R\n")

    delta_ret = mf["return_pct"] - mb["return_pct"]
    delta_dd  = mf["max_dd_pct"]  - mb["max_dd_pct"]
    delta_wr  = mf["wr"]          - mb["wr"]
    delta_pf  = mf["pf"]          - mb["pf"]
    print(f"  До фильтра  : Ret={mb['return_pct']:+.1f}%  WR={mb['wr']:.1f}%  "
          f"PF={mb['pf']:.2f}  DD={mb['max_dd_pct']:.1f}%")
    print(f"  После фильтра: Ret={mf['return_pct']:+.1f}%  WR={mf['wr']:.1f}%  "
          f"PF={mf['pf']:.2f}  DD={mf['max_dd_pct']:.1f}%")
    print(f"\n  Δ Ret={delta_ret:+.1f}%  ΔWR={delta_wr:+.1f}pp  "
          f"ΔPF={delta_pf:+.2f}  ΔDD={delta_dd:+.1f}pp")
    print()


if __name__ == "__main__":
    main()
