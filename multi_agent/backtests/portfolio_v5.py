"""
portfolio_v5.py — FINAL portfolio with all winning improvements
================================================================
Stack of improvements:
  V1 → V2: pivot SL + better TP grid                    (PF 1.32)
  V2 → V3: V3 portfolio with 4 strategies               (PF 1.49)
  V3 → V4: TP fix smacross 3R→4R, supertrend 2.5R→5R   (sumR +25-73%)
  V4 → V5: Add volume-spike entry filter + BTC trend override

Final variants tested:
  • V5_NEW_TP_27syms                  (best baseline)
  • V5_NEW_TP_27syms_volfilter        (entry only on volume>avg)
  • V5_NEW_TP_27syms_btc_filter       (skip when BTC in extreme regime)
  • V5_NEW_TP_27syms_combined         (both filters)
  • V5_COMPOSITE_OLD_TP                (composite + V3 TPs at 4% risk)
  • V5_COMBINED                       (best of everything)
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

from multi_agent.backtests.strategy_lab import (
    load, build_regime, simulate_portfolio, atr, sma, ema, rsi,
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
# SIGNAL FILTERS — apply at entry time
# ─────────────────────────────────────────────────────────────────────────────

def _vol_spike_ok(df, j, threshold=1.0):
    """Entry bar's volume >= threshold × 20-bar avg volume."""
    if j < 21: return True
    vo = df["volume"].to_numpy()
    avg = np.nanmean(vo[max(0, j-20):j])
    return vo[j] >= threshold * avg if avg > 0 else True


def _btc_extreme_regime(regime, ts_ms, max_pct=85):
    """Skip if BTC vol_pct is in extreme zone (>max_pct percentile)."""
    idx = int(np.searchsorted(regime.ts, ts_ms, side="right")) - 1
    if idx < 0: return False
    return regime.vol_pct[idx] > max_pct


def collect_v5_signals(syms, regime, tag,
                        active_by_ts=None,
                        use_vol_filter=False,
                        vol_thresh=1.0,
                        use_btc_filter=False,
                        btc_extreme=85,
                        # TP overrides
                        smacross_tp=4.0,
                        supertrend_tp=5.0,
                        rsi_tp=3.0):
    sigs = []; sim_dfs = {}
    sorted_ts = None
    if active_by_ts and "all" not in active_by_ts:
        sorted_ts = sorted(active_by_ts.keys())

    def is_active(sym, ts_ms):
        if not active_by_ts: return True
        if "all" in active_by_ts:
            return sym in active_by_ts["all"]
        idx = np.searchsorted(sorted_ts, ts_ms, side="right") - 1
        if idx < 0: return False
        return sym in active_by_ts[sorted_ts[idx]]

    def add(strat_name, df, slist, sym):
        if df is None: return
        sim_dfs[(sym, strat_name)] = df
        ts = df["timestamp"].to_numpy()
        for s in slist:
            if not is_active(sym, s.next_open_ts): continue
            # Find entry bar in df
            j = int(np.searchsorted(ts, s.next_open_ts, side="left"))
            if j >= len(ts): continue
            # Volume spike filter
            if use_vol_filter and not _vol_spike_ok(df, j, vol_thresh):
                continue
            # BTC extreme regime filter
            if use_btc_filter and _btc_extreme_regime(regime, s.next_open_ts, btc_extreme):
                continue
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts, sym=sym,
                strat=strat_name, direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    for sym in syms:
        df_1d = load(sym, "1d", tag)
        if df_1d is not None:
            add("smacross_pivot_1d_v5", df_1d,
                strat_smacross_pivot_1d(df_1d, regime, tp_r=smacross_tp), sym)
        df_4h = load(sym, "4h", tag)
        if df_4h is not None:
            add("supertrend_pivot_4h_v5", df_4h,
                strat_supertrend_pivot_4h(df_4h, regime, tp_r=supertrend_tp), sym)
            add("rsi_reversion_v2", df_4h,
                _strat_rsi_rev_v2(df_4h, regime, sl_atr=2.0, tp_r=rsi_tp), sym)
            add("fib_pump_4h", df_4h,
                strat_fib_pump_1h(df_4h, regime, consolidation_bars=18,
                                   pump_min_atr=3.0, pump_max_bars=6,
                                   retrace_min=0.382, retrace_max=0.618,
                                   tp_extension=1.0, sl_buffer_atr=0.5), sym)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


def run_test(name, syms, regime, active_by_ts=None,
              risk=0.03, max_pos=5, tag="365d",
              use_vol=False, vol_thresh=1.0,
              use_btc=False, btc_extreme=85,
              smacross_tp=4.0, supertrend_tp=5.0, rsi_tp=3.0):
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  {name}  (risk={risk*100:.1f}%, max_pos={max_pos}, tag={tag})")
    print(f"══════════════════════════════════════════════════════════════════")
    sigs, sim_dfs = collect_v5_signals(
        syms, regime, tag, active_by_ts,
        use_vol_filter=use_vol, vol_thresh=vol_thresh,
        use_btc_filter=use_btc, btc_extreme=btc_extreme,
        smacross_tp=smacross_tp, supertrend_tp=supertrend_tp, rsi_tp=rsi_tp,
    )
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

    metrics_by_sym = {s: compute_monthly_metrics(s, tag) for s in all_syms}
    metrics_by_sym = {k: v for k, v in metrics_by_sym.items() if v}
    composite = selector_composite(metrics_by_sym, top_n=50)
    composite_ts = {}
    for ms, slist in composite.items():
        try:
            t = int(pd.Period(ms, freq="M").start_time.tz_localize("UTC").timestamp() * 1000)
            composite_ts[t] = slist
        except Exception:
            pass

    results = []

    # Confirmed best from V4: NEW TPs + baseline 27, 3% risk
    r = run_test("V5_BASE_NEW_TPS", BASELINE, regime, {"all": BASELINE},
                  risk=0.03, smacross_tp=4.0, supertrend_tp=5.0)
    if r: results.append(r)

    # + 4% risk
    r = run_test("V5_BASE_NEW_TPS_4pct", BASELINE, regime, {"all": BASELINE},
                  risk=0.04, smacross_tp=4.0, supertrend_tp=5.0)
    if r: results.append(r)

    # + Volume spike filter (1.0×)
    r = run_test("V5_BASE_VOLFILT_1.0", BASELINE, regime, {"all": BASELINE},
                  risk=0.03, smacross_tp=4.0, supertrend_tp=5.0,
                  use_vol=True, vol_thresh=1.0)
    if r: results.append(r)

    # + Volume spike 1.5×
    r = run_test("V5_BASE_VOLFILT_1.5", BASELINE, regime, {"all": BASELINE},
                  risk=0.03, smacross_tp=4.0, supertrend_tp=5.0,
                  use_vol=True, vol_thresh=1.5)
    if r: results.append(r)

    # + BTC extreme regime filter
    r = run_test("V5_BASE_BTC_FILT", BASELINE, regime, {"all": BASELINE},
                  risk=0.03, smacross_tp=4.0, supertrend_tp=5.0,
                  use_btc=True, btc_extreme=85)
    if r: results.append(r)

    # Combined: vol_filter + btc_filter
    r = run_test("V5_BASE_BOTH_FILTERS", BASELINE, regime, {"all": BASELINE},
                  risk=0.03, smacross_tp=4.0, supertrend_tp=5.0,
                  use_vol=True, vol_thresh=1.0,
                  use_btc=True, btc_extreme=85)
    if r: results.append(r)

    # COMPOSITE with V3 TPs (3R/2.5R) — V3 universe was best with these TPs
    r = run_test("V5_COMPOSITE_V3_TPS", all_syms, regime, composite_ts,
                  risk=0.03, smacross_tp=3.0, supertrend_tp=2.5)
    if r: results.append(r)

    # COMPOSITE with V3 TPs at 4% risk
    r = run_test("V5_COMPOSITE_V3_TPS_4pct", all_syms, regime, composite_ts,
                  risk=0.04, smacross_tp=3.0, supertrend_tp=2.5)
    if r: results.append(r)

    # ULTIMATE: BASELINE + new TPs + vol filter + 4% risk
    r = run_test("V5_ULTIMATE", BASELINE, regime, {"all": BASELINE},
                  risk=0.04, smacross_tp=4.0, supertrend_tp=5.0,
                  use_vol=True, vol_thresh=1.0)
    if r: results.append(r)

    # SUMMARY
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V5 FINAL COMPARISON")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Config':28s} {'Trades':>7s} {'PF':>5s} {'WR%':>6s} {'avgR':>7s} {'Return':>9s} {'DD%':>6s} {'Avg/mo':>8s} {'Win-mo':>7s}")
    results.sort(key=lambda r: -r["metrics"]["return_pct"])
    for r in results:
        m = r["metrics"]
        avgmo = float(np.mean(r["rets"])) if r["rets"] else 0
        winmo = sum(1 for x in r["rets"] if x > 0) if r["rets"] else 0
        totmo = len(r["rets"]) if r["rets"] else 0
        print(f"  {r['name']:28s} {m['n']:>7d} {m['pf']:>5.2f} {m['wr']:>5.1f} "
              f"{m['avg_r']:>+7.3f} {m['return_pct']:>+8.1f}% {m['max_dd_pct']:>5.1f}% "
              f"{avgmo:>+7.1f}% {winmo}/{totmo}")


if __name__ == "__main__":
    main()
