"""
sl_lookahead_audit.py — Honest audit: is SL placed using future data?

For each Feb-Apr 2026 trade, prints:
  - Signal bar timestamp + price (when SuperTrend flipped)
  - Pivot bar timestamp + price (the swing low/high used for SL)
  - "Bars between pivot and signal" — must be ≥ 5 (lookback)
  - "Pivot known at signal time?" — must be YES (no look-ahead)
  - SL location relative to pivot: SL = pivot ± buffer×ATR
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
from datetime import datetime, timezone

from multi_agent.backtests.strategy_lab import load, build_regime, atr as _atr
from multi_agent.backtests.strategy_v3 import (
    find_pivots, latest_pivot_low, latest_pivot_high, supertrend,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import (
    load_all_events, build_sentiment_series, simulate_news_aware,
)


def ts_fmt(ts_ms): return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M")


def audit_trade(trade, sigs_lookup, dfs_lookup):
    """Return audit info dict for one trade."""
    sym, strat = trade["sym"], trade["strat"]
    if strat != "supertrend":
        return None  # only audit supertrend trades

    df = dfs_lookup.get((sym, strat))
    if df is None: return None

    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    cl = df["close"].to_numpy(float)
    op = df["open"].to_numpy(float)
    ts = df["timestamp"].to_numpy(int)
    a  = _atr(hi, lo, cl, 14)

    # Find signal bar (the one that fired the SuperTrend flip)
    j_entry = int(np.searchsorted(ts, trade["entry_ts"], side="left"))
    if j_entry >= len(ts): return None
    i_sig = j_entry - 1   # signal bar is one before entry bar (entry = next_open)

    sig_ts    = ts[i_sig]
    sig_close = cl[i_sig]
    sig_open  = op[i_sig]
    sig_high  = hi[i_sig]
    sig_low   = lo[i_sig]
    sig_atr   = a[i_sig]

    # Compute pivots WITH 5-bar lookback (same as in strategy)
    piv_hi, piv_lo = find_pivots(hi, lo, lookback=5)

    # Look up pivot used at signal bar
    if trade["direction"] == "LONG":
        piv, piv_idx_registered = latest_pivot_low(piv_lo, i_sig, max_lookback=20)
    else:
        piv, piv_idx_registered = latest_pivot_high(piv_hi, i_sig, max_lookback=20)

    if piv is None:
        return None

    # piv_idx_registered = bar where pivot was REGISTERED (i.e. became known)
    # piv_idx_actual    = bar where the pivot HIGH/LOW actually occurred (5 bars earlier)
    piv_idx_actual = piv_idx_registered - 5
    piv_ts_actual    = ts[piv_idx_actual]
    piv_ts_registered = ts[piv_idx_registered]

    # Bars between pivot bar and signal bar
    bars_pivot_to_signal = i_sig - piv_idx_actual

    # SL location math
    buffer_atr = 0.3
    if trade["direction"] == "LONG":
        sl_calc = piv - buffer_atr * sig_atr
    else:
        sl_calc = piv + buffer_atr * sig_atr

    return {
        "sym": sym, "dir": trade["direction"],
        "entry_ts": trade["entry_ts"],
        "entry_price": trade["entry"],
        "sig_bar_ts": sig_ts,
        "sig_close": sig_close,
        "pivot_actual_ts":     piv_ts_actual,      # bar where pivot LOW/HIGH formed
        "pivot_registered_ts": piv_ts_registered,  # bar where strategy "discovered" it
        "pivot_price": piv,
        "bars_pivot_to_signal": bars_pivot_to_signal,
        "lookback_required": 5,
        "pivot_known_at_signal": piv_ts_registered <= sig_ts,
        "sl_price_calc": sl_calc,
        "sig_atr": sig_atr,
        "buffer_atr": buffer_atr,
        "reason": trade["reason"],
        "win": trade["win"],
        "pnl_r": trade["pnl_r"],
    }


def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  SL LOOK-AHEAD AUDIT — Feb-Apr 2026 supertrend trades")
    print("══════════════════════════════════════════════════════════════════════\n")

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    sigs, dfs, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    events    = load_all_events()
    sentiment = build_sentiment_series()

    res = simulate_news_aware(sigs, dfs, regime, events, sentiment,
                               base_risk=0.04, enable_l4=True)

    FEB = int(datetime(2026,2,1,tzinfo=timezone.utc).timestamp()*1000)
    MAY = int(datetime(2026,5,1,tzinfo=timezone.utc).timestamp()*1000)
    period_trades = sorted(
        [t for t in res["trades"] if FEB <= t["entry_ts"] < MAY],
        key=lambda x: x["entry_ts"],
    )

    print(f"  Trades: {len(period_trades)}\n")
    print(f"  {'Sym':5s} {'Dir':4s} {'Entry time':17s} {'PivotForm':17s} "
          f"{'PivotKnown':17s} {'BarsPivSig':>11s} {'OK?':4s} {'PivPrice':>10s} {'SLcalc':>10s} {'Result':>9s}")
    print("  " + "─" * 120)

    cheating = 0
    ok = 0
    for t in period_trades:
        a = audit_trade(t, None, dfs)
        if a is None:
            print(f"  {t['sym']:5s} -- skip (not supertrend or no pivot)"); continue

        bars_diff = a["bars_pivot_to_signal"]
        ok_flag   = "✓" if a["pivot_known_at_signal"] and bars_diff >= 5 else "✗"
        if ok_flag == "✓": ok += 1
        else: cheating += 1

        print(f"  {a['sym']:5s} {a['dir']:4s} {ts_fmt(a['entry_ts'])} "
              f"{ts_fmt(a['pivot_actual_ts'])} "
              f"{ts_fmt(a['pivot_registered_ts'])} "
              f"{bars_diff:>11d} {ok_flag:4s} "
              f"{a['pivot_price']:>10.5g} {a['sl_price_calc']:>10.5g} "
              f"{a['pnl_r']:>+8.2f}R")

    print(f"\n  ══════════════════════════════════════════════════════════════")
    print(f"  Pivots known BEFORE signal: {ok}/{ok+cheating}")
    print(f"  Pivots formed AFTER signal: {cheating}  ← would be look-ahead")
    print(f"  ══════════════════════════════════════════════════════════════\n")

    # Detailed view of suspicious trades (where bars_pivot_to_signal is small)
    print("  Detail view: how far back was the pivot vs signal bar?\n")
    print(f"  {'Sym':5s} {'Bars pivot→sig':>15s} {'Days ago':>10s} {'PivotPrice':>10s} "
          f"{'EntryPrice':>10s} {'%diff':>8s} {'SLprice':>10s} {'SL %below entry':>17s}")
    print("  " + "─" * 110)
    for t in period_trades:
        a = audit_trade(t, None, dfs)
        if a is None: continue
        days_ago = a["bars_pivot_to_signal"] * 4 / 24   # 4h bars → days
        pct_pivot_entry = (a["entry_price"] - a["pivot_price"]) / a["pivot_price"] * 100 \
                          if a["dir"] == "LONG" else \
                          (a["pivot_price"] - a["entry_price"]) / a["pivot_price"] * 100
        pct_sl = (a["entry_price"] - a["sl_price_calc"]) / a["entry_price"] * 100 \
                  if a["dir"] == "LONG" else \
                  (a["sl_price_calc"] - a["entry_price"]) / a["entry_price"] * 100

        print(f"  {a['sym']:5s} {a['bars_pivot_to_signal']:>15d} {days_ago:>9.1f}d "
              f"{a['pivot_price']:>10.5g} {a['entry_price']:>10.5g} "
              f"{pct_pivot_entry:>+7.2f}% {a['sl_price_calc']:>10.5g} "
              f"{pct_sl:>+15.2f}%")


if __name__ == "__main__":
    main()
