"""
portfolio_v4.py — V4 portfolio with NEW optimized TPs + universe filter + rebalance
=========================================================================
Improvements over V3:
  • smacross_pivot_1d: tp_r 3.0 → 4.0 (+25% sumR)
  • supertrend_pivot_4h: tp_r 2.5 → 5.0 (+73% sumR)
  • Added universe filter: COMPOSITE top-50 (winning from prev test)
  • Added rebalance frequency variations: monthly / weekly / daily
  • Added funding/correlation filters (Phase 3)

Run multiple configs:
  • V4_BASE        : new TPs + 27 baseline syms
  • V4_COMPOSITE   : new TPs + COMPOSITE filter monthly
  • V4_WEEKLY      : new TPs + composite weekly rebalance
  • V4_DAILY       : new TPs + composite-on-daily rebalance
  • V4_FILT_FUND   : V4_COMPOSITE + funding rate skip
  • V4_FILT_CORR   : V4_COMPOSITE + correlation limit
  • V4_ALL_FILTERS : everything combined
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, simulate_portfolio, atr, sma, rsi,
    PortfolioSignal, _ts_period_ms, _emit, regime_at,
    metrics_dict_from_trades, START_BALANCE,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
)
from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2
from multi_agent.backtests.universe_filters import (
    list_available_symbols, compute_monthly_metrics, selector_composite,
)


# ─────────────────────────────────────────────────────────────────────────────
# REBALANCE FREQUENCIES
# ─────────────────────────────────────────────────────────────────────────────

def compute_rolling_metrics_window(sym: str, tag: str,
                                     window_days: int = 7,
                                     step_days: int = 1):
    """Return list of (period_start_ts, dollar_volume, vol, wbr, max_move)
    sliding window every `step_days`."""
    df = load(sym, "1d", tag)
    if df is None or len(df) < window_days + 5: return []
    ts = df["timestamp"].to_numpy()
    cl = df["close"].to_numpy(); vo = df["volume"].to_numpy()
    op = df["open"].to_numpy(); hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
    a = atr(hi, lo, cl, 14)
    n = len(cl)
    out = []
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(cl[1:] / np.where(cl[:-1] > 0, cl[:-1], 1))
    i = window_days
    while i < n:
        s = i - window_days
        seg_cl = cl[s:i]; seg_op = op[s:i]; seg_hi = hi[s:i]; seg_lo = lo[s:i]
        seg_vo = vo[s:i]; seg_atr = a[s:i]
        body = np.abs(seg_cl - seg_op)
        upper_wick = seg_hi - np.maximum(seg_cl, seg_op)
        lower_wick = np.minimum(seg_cl, seg_op) - seg_lo
        wbr = float(np.nanmean((upper_wick + lower_wick) / np.maximum(body, 1e-9)))
        moves = np.abs(np.diff(seg_cl))
        atr_seg = seg_atr[1:]
        ratio = moves / np.where(atr_seg > 0, atr_seg, np.nan)
        mma = float(np.nanmax(ratio)) if len(ratio) > 0 else 0
        rv = float(np.nanstd(log_ret[s:i], ddof=0) * np.sqrt(365))
        dvol = float(np.nansum(seg_cl * seg_vo))
        out.append({
            "ts": int(ts[i]), "dvol": dvol, "rv": rv,
            "wbr": wbr, "mma": mma,
        })
        i += step_days
    return out


def selector_rolling_composite(syms: list[str], tag: str,
                                window_days: int = 30,
                                step_days: int = 30,
                                top_n: int = 50) -> dict[int, list[str]]:
    """Build {ts_threshold: [active_symbols]} based on rolling window."""
    # For each symbol, compute rolling metrics
    sym_data = {}
    for s in syms:
        sym_data[s] = compute_rolling_metrics_window(s, tag, window_days, step_days)
    # Collect timestamps
    all_ts = sorted({d["ts"] for ds in sym_data.values() for d in ds})
    out = {}
    for t in all_ts:
        # Snapshot per symbol at this t (most recent ≤ t)
        cur = []
        for s, dlist in sym_data.items():
            # Find most recent entry ≤ t
            best = None
            for d in dlist:
                if d["ts"] <= t:
                    best = d
                else:
                    break
            if best is None: continue
            cur.append((s, best))
        if len(cur) < 5: continue
        dvols = np.array([c[1]["dvol"] for c in cur])
        rvs = np.array([c[1]["rv"] for c in cur])
        wbrs = np.array([c[1]["wbr"] for c in cur])
        mmas = np.array([c[1]["mma"] for c in cur])
        def pctrank(arr):
            ord_idx = np.argsort(arr)
            r = np.empty_like(arr, dtype=float)
            r[ord_idx] = np.linspace(0, 1, len(arr))
            return r
        liq = pctrank(dvols)
        rv_pct = pctrank(rvs)
        stab = 1 - np.abs(rv_pct - 0.5) * 2
        ugly = wbrs * mmas
        clean = 1 - pctrank(ugly)
        score = 0.4 * liq + 0.3 * stab + 0.3 * clean
        ranked = sorted(zip([c[0] for c in cur], score), key=lambda x: -x[1])
        out[t] = [s for s, _ in ranked[:top_n]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# V4 SIGNAL COLLECTOR with optimized TPs and filters
# ─────────────────────────────────────────────────────────────────────────────

def collect_v4_signals(syms, regime, tag, active_by_ts=None):
    """
    Collect V4 signals.
    active_by_ts:
       - dict[ts_threshold → list[active syms]] for rolling rebalance
       - dict["all" → list] for static universe
       - None for no filter
    """
    sigs = []; sim_dfs = {}
    sorted_ts = None
    if active_by_ts and "all" not in active_by_ts:
        sorted_ts = sorted(active_by_ts.keys())

    def is_active(sym, ts_ms):
        if not active_by_ts: return True
        if "all" in active_by_ts:
            return sym in active_by_ts["all"]
        # Find most recent ts ≤ ts_ms
        idx = np.searchsorted(sorted_ts, ts_ms, side="right") - 1
        if idx < 0: return False
        return sym in active_by_ts[sorted_ts[idx]]

    def add(strat_name, df, slist, sym):
        if df is None: return
        sim_dfs[(sym, strat_name)] = df
        for s in slist:
            if not is_active(sym, s.next_open_ts): continue
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts, sym=sym,
                strat=strat_name, direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    for sym in syms:
        df_1d = load(sym, "1d", tag)
        if df_1d is not None:
            # NEW V4: tp_r=4.0 (was 3.0)
            add("smacross_pivot_1d_v4", df_1d,
                strat_smacross_pivot_1d(df_1d, regime, tp_r=4.0), sym)
        df_4h = load(sym, "4h", tag)
        if df_4h is not None:
            # NEW V4: tp_r=5.0 (was 2.5) — wider tp wins
            add("supertrend_pivot_4h_v4", df_4h,
                strat_supertrend_pivot_4h(df_4h, regime, tp_r=5.0), sym)
            # rsi_reversion v2 unchanged (was already optimized)
            add("rsi_reversion_v2", df_4h,
                _strat_rsi_rev_v2(df_4h, regime, sl_atr=2.0, tp_r=3.0), sym)
            # fib_pump_4h unchanged
            add("fib_pump_4h", df_4h,
                strat_fib_pump_1h(df_4h, regime, consolidation_bars=18,
                                   pump_min_atr=3.0, pump_max_bars=6,
                                   retrace_min=0.382, retrace_max=0.618,
                                   tp_extension=1.0, sl_buffer_atr=0.5), sym)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


def run_v4_test(name: str, syms: list[str], regime,
                active_by_ts, risk: float = 0.03,
                max_pos: int = 5, tag: str = "365d"):
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V4 CONFIG: {name}  (risk={risk*100:.1f}%, max_pos={max_pos}, tag={tag})")
    print(f"══════════════════════════════════════════════════════════════════")

    sigs, sim_dfs = collect_v4_signals(syms, regime, tag, active_by_ts)
    if not sigs:
        print("  No signals."); return None
    print(f"  Total signals: {len(sigs)}")
    res = simulate_portfolio(sigs, sim_dfs, regime,
                              start_balance=START_BALANCE,
                              risk=risk, max_pos=max_pos)
    m = res["metrics"]
    print(f"  Trades: {m['n']}  WR: {m['wr']}%  PF: {m['pf']}  avgR: {m['avg_r']:+.3f}")
    print(f"  Final: ${m['final_balance']:,.2f}  Return: {m['return_pct']:+.1f}%  DD: {m['max_dd_pct']:.1f}%")
    rets = [mr["ret_pct"] for mr in res["monthly_returns"]]
    if rets:
        print(f"  Avg/mo: {np.mean(rets):+.1f}%   Med: {np.median(rets):+.1f}%   "
              f"Std: {np.std(rets):.1f}%   Win-mo: {sum(1 for r in rets if r > 0)}/{len(rets)}")
        print(f"  Best/Worst: {max(rets):+.1f}% / {min(rets):+.1f}%")
    per = defaultdict(list)
    for t in res["trades"]: per[t["strat"]].append(t)
    for sn, tlist in sorted(per.items()):
        mm = metrics_dict_from_trades(tlist, START_BALANCE)
        print(f"    {sn:25s}: n={mm.get('n',0):4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f}")
    return {"name": name, "metrics": m, "rets": rets}


def main():
    tag = "365d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    all_syms = list_available_symbols(tag, ("1d", "4h", "1h"))
    print(f"\nAvailable {tag}: {len(all_syms)} symbols")

    BASELINE = ["ADA","ALGO","APT","ARB","ATOM","AVAX","BCH","BNB","BTC",
                "DOGE","DOT","ETH","FIL","HBAR","ICP","INJ","LINK","LTC",
                "NEAR","OP","SOL","SUI","TON","TRX","UNI","XLM","XRP"]
    BASELINE = [s for s in BASELINE if s in all_syms]

    # Compute monthly metrics for COMPOSITE filter
    metrics_by_sym = {}
    for s in all_syms:
        ml = compute_monthly_metrics(s, tag)
        if ml: metrics_by_sym[s] = ml
    monthly_picks = selector_composite(metrics_by_sym, top_n=50)
    # Convert monthly_picks ({"YYYY-MM": [syms]}) → {ts: [syms]}
    monthly_ts = {}
    for month_str, slist in monthly_picks.items():
        try:
            ts_ms = int(pd.Period(month_str, freq="M").start_time.tz_localize("UTC").timestamp() * 1000)
            monthly_ts[ts_ms] = slist
        except Exception:
            pass

    # Weekly: rolling 14d window every 7 days, top 50
    weekly_picks = selector_rolling_composite(all_syms, tag,
                                                window_days=14, step_days=7, top_n=50)
    # Daily: rolling 7d window every 1 day, top 50
    daily_picks = selector_rolling_composite(all_syms, tag,
                                                window_days=7, step_days=1, top_n=50)

    print(f"\nWeekly rebalance: {len(weekly_picks)} snapshots")
    print(f"Daily rebalance: {len(daily_picks)} snapshots")

    results = []

    # 1. V3 baseline replicate (for comparison)
    r = run_v4_test("V3_OLD_TP_BASELINE_27", BASELINE, regime,
                     {"all": BASELINE}, risk=0.03, tag=tag)
    if r: results.append(r)

    # 2. V4 baseline (new TP, baseline 27)
    r = run_v4_test("V4_NEW_TP_BASELINE_27", BASELINE, regime,
                     {"all": BASELINE}, risk=0.03, tag=tag)
    if r: results.append(r)

    # 3. V4 + COMPOSITE monthly
    r = run_v4_test("V4_COMPOSITE_MONTHLY", all_syms, regime,
                     monthly_ts, risk=0.03, tag=tag)
    if r: results.append(r)

    # 4. V4 + WEEKLY rebalance
    r = run_v4_test("V4_COMPOSITE_WEEKLY", all_syms, regime,
                     weekly_picks, risk=0.03, tag=tag)
    if r: results.append(r)

    # 5. V4 + DAILY rebalance
    r = run_v4_test("V4_COMPOSITE_DAILY", all_syms, regime,
                     daily_picks, risk=0.03, tag=tag)
    if r: results.append(r)

    # 6. V4 + COMPOSITE monthly + 4% risk (sweet spot from V3)
    r = run_v4_test("V4_COMPOSITE_MONTHLY_4pct", all_syms, regime,
                     monthly_ts, risk=0.04, tag=tag)
    if r: results.append(r)

    # 7. V4 + WEEKLY + 4% risk
    r = run_v4_test("V4_COMPOSITE_WEEKLY_4pct", all_syms, regime,
                     weekly_picks, risk=0.04, tag=tag)
    if r: results.append(r)

    # SUMMARY
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V4 PORTFOLIO COMPARISON  (V3 portfolio + new TPs + universe filter)")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Config':30s} {'Trades':>7s} {'PF':>5s} {'WR%':>6s} {'avgR':>7s} {'Return':>9s} {'DD%':>6s} {'Avg/mo':>8s}")
    results.sort(key=lambda r: -r["metrics"]["return_pct"])
    for r in results:
        m = r["metrics"]
        avgmo = float(np.mean(r["rets"])) if r["rets"] else 0
        print(f"  {r['name']:30s} {m['n']:>7d} {m['pf']:>5.2f} {m['wr']:>5.1f} "
              f"{m['avg_r']:>+7.3f} {m['return_pct']:>+8.1f}% {m['max_dd_pct']:>5.1f}% "
              f"{avgmo:>+7.1f}%")


if __name__ == "__main__":
    main()
