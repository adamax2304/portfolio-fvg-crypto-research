"""
Complementary strategy prototype for V7:

Bear Donchian 1D 50-bar short overlay.

Idea:
  - V7 is mostly long/trend-following and its live implementation suppresses shorts
    when BTC is below its 50-day SMA.
  - This overlay is only active in that same BTC<SMA50 regime.
  - It shorts fresh 50-day downside breakouts on the 27-symbol V7 universe.

Rules:
  1. BTC close < BTC SMA50 on the signal day.
  2. Symbol close < prior 50-day Donchian low.
  3. Symbol close < symbol EMA200.
  4. Previous close was not already below the previous lower channel
     (avoid duplicate continuation bars).
  5. Entry = next daily open, SL = 2 ATR, TP = 2R.

Run:
  py -3 multi_agent/backtests/v7_bear_donchian_overlay.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_agent.backtests.strategy_lab import (
    PortfolioSignal,
    _ts_period_ms,
    atr,
    build_regime,
    donchian,
    ema,
    load,
    sma,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, CACHE_DIR, collect_v7_signals
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter
from multi_agent.backtests.news_aware_v7 import load_all_events
from multi_agent.backtests.risk_controls_v7 import build_corr_matrix, simulate_rc


RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _month_summary(res: dict, start_balance: float = 1000.0) -> dict:
    trades = sorted(res.get("trades", []), key=lambda t: t.get("exit_ts") or 0)
    eq = float(start_balance)
    prev_eq = eq
    cur_month = None
    month_start_eq = eq
    vals = []

    for trade in trades:
        ts = trade.get("exit_ts") or trade.get("entry_ts") or trade.get("ts")
        if ts is None:
            continue
        month = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m")
        if cur_month is None:
            cur_month = month
            month_start_eq = eq
        elif month != cur_month:
            vals.append((prev_eq / month_start_eq - 1.0) * 100.0 if month_start_eq else 0.0)
            cur_month = month
            month_start_eq = eq
        eq += float(trade.get("pnl", 0.0))
        prev_eq = eq

    if cur_month is not None:
        vals.append((eq / month_start_eq - 1.0) * 100.0 if month_start_eq else 0.0)

    return {
        "months": len(vals),
        "positive": sum(1 for x in vals if x > 0),
        "negative": sum(1 for x in vals if x < 0),
        "avg": round(float(np.mean(vals)), 2) if vals else None,
        "best": round(float(np.max(vals)), 2) if vals else None,
        "worst": round(float(np.min(vals)), 2) if vals else None,
    }


def build_overlay_signals():
    btc = load("BTC", "1d", "730d")
    if btc is None or btc.empty:
        raise RuntimeError("BTC 1d cache not found")

    btc_ts = btc["timestamp"].to_numpy()
    btc_cl = btc["close"].to_numpy(float)
    btc_s50 = sma(btc_cl, 50)

    def btc_bear(ts_ms: int) -> bool:
        idx = int(np.searchsorted(btc_ts, ts_ms, side="right")) - 1
        if idx < 0 or idx >= len(btc_cl) or np.isnan(btc_s50[idx]):
            return False
        return bool(btc_cl[idx] < btc_s50[idx])

    sigs = []
    sim_dfs = {}
    syms = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    for sym in syms:
        df = load(sym, "1d", "730d")
        if df is None or len(df) < 250:
            continue
        sim_dfs[(sym, "bear_donchian_1d50")] = df
        hi = df["high"].to_numpy(float)
        lo = df["low"].to_numpy(float)
        cl = df["close"].to_numpy(float)
        op = df["open"].to_numpy(float)
        ts = df["timestamp"].to_numpy(int)
        a = atr(hi, lo, cl, 14)
        e200 = ema(cl, 200)
        _upper, lower = donchian(hi, lo, 50)
        period_ms = _ts_period_ms(ts)

        for i in range(210, len(cl) - 1):
            if np.isnan(a[i]) or np.isnan(e200[i]) or np.isnan(lower[i - 1]) or np.isnan(lower[i - 2]):
                continue
            if not btc_bear(int(ts[i])):
                continue
            if cl[i] < lower[i - 1] and cl[i] < e200[i] and cl[i - 1] >= lower[i - 2]:
                entry = op[i + 1]
                sl = entry + 2.0 * a[i]
                tp = entry - 2.0 * (sl - entry)
                sigs.append(
                    PortfolioSignal(
                        ts=int(ts[i]),
                        next_open_ts=int(ts[i]) + period_ms,
                        sym=sym,
                        strat="bear_donchian_1d50",
                        direction="SHORT",
                        entry=float(entry),
                        sl=float(sl),
                        tp=float(tp),
                        dist=float(sl - entry),
                    )
                )

    sigs.sort(key=lambda x: x.next_open_ts)
    return btc, syms, sigs, sim_dfs


def main():
    btc, syms, overlay_sigs, overlay_dfs = build_overlay_signals()
    regime = build_regime(btc)
    events = load_all_events()

    base_sigs, base_dfs, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    base_sigs_pf, _removed, _stats = apply_pivot_dist_filter(base_sigs, base_dfs, 7.0)

    sim_dfs = dict(base_dfs)
    sim_dfs.update(overlay_dfs)
    combined_sigs = sorted(base_sigs_pf + overlay_sigs, key=lambda x: x.next_open_ts)
    corr = build_corr_matrix(sim_dfs)

    combined = simulate_rc(
        combined_sigs,
        sim_dfs,
        regime,
        events,
        base_risk=0.04,
        max_pos=5,
        enable_l4=True,
        max_corr_open=1,
        corr_threshold=0.80,
        corr_matrix=corr,
    )
    overlay_only = simulate_rc(
        overlay_sigs,
        sim_dfs,
        regime,
        events,
        base_risk=0.01,
        max_pos=2,
        enable_l4=True,
        max_corr_open=1,
        corr_threshold=0.80,
        corr_matrix=corr,
    )

    result = {
        "overlay_signal_count": len(overlay_sigs),
        "combined_metrics": combined["metrics"],
        "combined_monthly_summary": _month_summary(combined),
        "overlay_only_1pct_2pos_metrics": overlay_only["metrics"],
        "overlay_only_1pct_2pos_monthly_summary": _month_summary(overlay_only),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
