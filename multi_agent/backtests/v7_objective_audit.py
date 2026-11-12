"""
Objective audit for the live V7 portfolio branch.

Goals:
1. Reproduce the current V7 backtest with the same cached data and portfolio simulator.
2. Quantify live/backtest mismatch around the BTC<SMA50 bear filter used by v7_agent.
3. Measure recent-window behavior (365d / 180d / 90d) instead of relying only on 730d.
4. Evaluate drawdown-reduction controls already supported by risk_controls_v7.py.
5. Evaluate one complementary overlay strategy on the same data / simulation structure.

Run:
  py -3 multi_agent/backtests/v7_objective_audit.py
  py -3 multi_agent/backtests/v7_objective_audit.py --output multi_agent/backtests/results/v7_objective_audit_latest.json
"""
from __future__ import annotations

import argparse
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
DAY_MS = 86_400_000


def _month_stats_from_trades(res: dict, start_balance: float = 1000.0) -> tuple[list[dict], dict]:
    trades = sorted(res.get("trades", []), key=lambda t: t.get("exit_ts") or 0)
    eq = float(start_balance)
    prev_eq = eq
    cur_month = None
    month_start_eq = eq
    monthly = []

    for trade in trades:
        ts = trade.get("exit_ts") or trade.get("entry_ts") or trade.get("ts")
        if ts is None:
            continue
        month = pd.Timestamp(ts, unit="ms", tz="UTC").strftime("%Y-%m")
        if cur_month is None:
            cur_month = month
            month_start_eq = eq
        elif month != cur_month:
            ret_pct = (prev_eq / month_start_eq - 1.0) * 100.0 if month_start_eq else 0.0
            monthly.append(
                {
                    "month": cur_month,
                    "start": round(month_start_eq, 2),
                    "end": round(prev_eq, 2),
                    "ret_pct": round(ret_pct, 2),
                }
            )
            cur_month = month
            month_start_eq = eq

        eq += float(trade.get("pnl", 0.0))
        prev_eq = eq

    if cur_month is not None:
        ret_pct = (eq / month_start_eq - 1.0) * 100.0 if month_start_eq else 0.0
        monthly.append(
            {
                "month": cur_month,
                "start": round(month_start_eq, 2),
                "end": round(eq, 2),
                "ret_pct": round(ret_pct, 2),
            }
        )

    vals = [m["ret_pct"] for m in monthly]
    summary = {
        "months": len(vals),
        "positive": sum(1 for x in vals if x > 0),
        "negative": sum(1 for x in vals if x < 0),
        "avg": round(float(np.mean(vals)), 2) if vals else None,
        "median": round(float(np.median(vals)), 2) if vals else None,
        "best": round(float(np.max(vals)), 2) if vals else None,
        "worst": round(float(np.min(vals)), 2) if vals else None,
        "last12_positive": sum(1 for x in vals[-12:] if x > 0),
        "last12_total": min(12, len(vals)),
    }
    return monthly, summary


def _load_context():
    btc = load("BTC", "1d", "730d")
    if btc is None or btc.empty:
        raise RuntimeError("BTC 1d cache not found")
    regime = build_regime(btc)
    syms = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    events = load_all_events()
    return btc, regime, syms, events


def _btc_bear_checker(btc_df: pd.DataFrame):
    btc_ts = btc_df["timestamp"].to_numpy()
    btc_cl = btc_df["close"].to_numpy(float)
    btc_s50 = sma(btc_cl, 50)

    def _btc_bear(ts_ms: int) -> bool:
        idx = int(np.searchsorted(btc_ts, ts_ms, side="right")) - 1
        if idx < 0 or idx >= len(btc_cl) or np.isnan(btc_s50[idx]):
            return False
        return bool(btc_cl[idx] < btc_s50[idx])

    return _btc_bear


def _run_v7_base(regime, syms, events):
    sigs, sim_dfs, info = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    sigs_pf, removed_count, pf_stats = apply_pivot_dist_filter(sigs, sim_dfs, 7.0)
    corr = build_corr_matrix(sim_dfs)
    res = simulate_rc(
        sigs_pf,
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
    monthly, month_summary = _month_stats_from_trades(res)
    return {
        "signals_raw": len(sigs),
        "signals_after_pivot_filter": len(sigs_pf),
        "signal_info": info,
        "pivot_filter": {"removed_count": removed_count, **pf_stats},
        "metrics": res["metrics"],
        "monthly": monthly,
        "monthly_summary": month_summary,
        "trades": res["trades"],
        "sim_dfs": sim_dfs,
        "sigs_pf": sigs_pf,
        "corr": corr,
    }


def _run_recent_windows(base_pack: dict, regime, events):
    out = {}
    last_ts = max(s.next_open_ts for s in base_pack["sigs_pf"])
    for label, days in {"365d": 365, "180d": 180, "90d": 90}.items():
        cutoff = last_ts - days * DAY_MS
        sigs = [s for s in base_pack["sigs_pf"] if s.next_open_ts >= cutoff]
        res = simulate_rc(
            sigs,
            base_pack["sim_dfs"],
            regime,
            events,
            base_risk=0.04,
            max_pos=5,
            enable_l4=True,
            max_corr_open=1,
            corr_threshold=0.80,
            corr_matrix=base_pack["corr"],
        )
        monthly, summary = _month_stats_from_trades(res)
        out[label] = {
            "metrics": res["metrics"],
            "monthly": monthly,
            "monthly_summary": summary,
        }
    return out


def _run_live_bear_mismatch(btc_df: pd.DataFrame, regime, syms, events):
    btc_bear = _btc_bear_checker(btc_df)
    sigs, sim_dfs, _info = collect_v7_signals(syms, regime, "730d", bear_mode="all")
    sigs = [s for s in sigs if not (btc_bear(s.next_open_ts) and s.direction == "SHORT")]
    sigs_pf, removed_count, pf_stats = apply_pivot_dist_filter(sigs, sim_dfs, 7.0)
    corr = build_corr_matrix(sim_dfs)
    res = simulate_rc(
        sigs_pf,
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
    monthly, summary = _month_stats_from_trades(res)
    return {
        "signals_after_pivot_filter": len(sigs_pf),
        "pivot_filter": {"removed_count": removed_count, **pf_stats},
        "metrics": res["metrics"],
        "monthly": monthly,
        "monthly_summary": summary,
    }


def _run_risk_controls(base_pack: dict, regime, events):
    variants = {
        "dd3": dict(daily_dd_limit=0.03),
        "heat12": dict(max_heat=0.12),
        "daily3_trades2": dict(daily_dd_limit=0.03, max_daily_trades=2),
    }
    out = {}
    for name, extra in variants.items():
        res = simulate_rc(
            base_pack["sigs_pf"],
            base_pack["sim_dfs"],
            regime,
            events,
            base_risk=0.04,
            max_pos=5,
            enable_l4=True,
            max_corr_open=1,
            corr_threshold=0.80,
            corr_matrix=base_pack["corr"],
            **extra,
        )
        monthly, summary = _month_stats_from_trades(res)
        out[name] = {
            "params": extra,
            "metrics": res["metrics"],
            "monthly": monthly,
            "monthly_summary": summary,
        }
    return out


def _build_bear_donchian_overlay(btc_df: pd.DataFrame, syms):
    btc_bear = _btc_bear_checker(btc_df)
    sigs = []
    sim_dfs = {}
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
    return sigs, sim_dfs


def _run_overlay(btc_df: pd.DataFrame, base_pack: dict, regime, events):
    overlay_sigs, overlay_dfs = _build_bear_donchian_overlay(btc_df, BASELINE)
    sim_dfs = dict(base_pack["sim_dfs"])
    sim_dfs.update(overlay_dfs)
    all_sigs = sorted(base_pack["sigs_pf"] + overlay_sigs, key=lambda x: x.next_open_ts)
    corr = build_corr_matrix(sim_dfs)

    combined = simulate_rc(
        all_sigs,
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
    combined_monthly, combined_summary = _month_stats_from_trades(combined)

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
    overlay_only_monthly, overlay_only_summary = _month_stats_from_trades(overlay_only)

    last_ts = max(s.next_open_ts for s in all_sigs)
    recent = {}
    for label, days in {"365d": 365, "180d": 180, "90d": 90}.items():
        cutoff = last_ts - days * DAY_MS
        sigs = [s for s in all_sigs if s.next_open_ts >= cutoff]
        res = simulate_rc(
            sigs,
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
        monthly, summary = _month_stats_from_trades(res)
        recent[label] = {
            "metrics": res["metrics"],
            "monthly": monthly,
            "monthly_summary": summary,
        }

    return {
        "overlay_signal_count": len(overlay_sigs),
        "combined": {
            "metrics": combined["metrics"],
            "monthly": combined_monthly,
            "monthly_summary": combined_summary,
        },
        "overlay_only_1pct_2pos": {
            "metrics": overlay_only["metrics"],
            "monthly": overlay_only_monthly,
            "monthly_summary": overlay_only_summary,
        },
        "recent": recent,
    }


def build_audit():
    btc, regime, syms, events = _load_context()
    base_pack = _run_v7_base(regime, syms, events)
    return {
        "meta": {
            "symbol_count": len(syms),
            "symbols": syms,
            "data_tag": "730d",
            "base_risk": 0.04,
            "max_pos": 5,
            "corr_threshold": 0.80,
        },
        "current_v7": {
            "signals_raw": base_pack["signals_raw"],
            "signals_after_pivot_filter": base_pack["signals_after_pivot_filter"],
            "signal_info": base_pack["signal_info"],
            "pivot_filter": base_pack["pivot_filter"],
            "metrics": base_pack["metrics"],
            "monthly": base_pack["monthly"],
            "monthly_summary": base_pack["monthly_summary"],
        },
        "recent_windows": _run_recent_windows(base_pack, regime, events),
        "live_bear_filter_mismatch": _run_live_bear_mismatch(btc, regime, syms, events),
        "risk_controls": _run_risk_controls(base_pack, regime, events),
        "complementary_overlay": _run_overlay(btc, base_pack, regime, events),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    audit = build_audit()
    text = json.dumps(audit, indent=2)

    if args.output:
        out_path = Path(args.output)
        if not out_path.is_absolute():
            out_path = ROOT / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"Saved audit to {out_path}")
    else:
        print(text)


if __name__ == "__main__":
    main()
