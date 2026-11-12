"""
universe_filters.py — Coin selection methodologies
====================================================
Tests 5 universe selectors with V3 portfolio:

  1. baseline27   — original 27 symbols (730d)
  2. all_avail    — all symbols with 365d coverage (44+)
  3. vol_rolling  — top-N by 30d $-volume, monthly rebalance
  4. vol_band     — exclude top-X% volatile (calm coins only)
  5. clean        — exclude top-X% by manipulation proxy (wick/body)
  6. composite    — my methodology: liquidity × stability × cleanness

Metrics computed per symbol per month:
  • dollar_volume_30d   = sum(close × volume) for last 30 days
  • realized_vol_30d    = std(daily log returns) × sqrt(365)
  • wick_body_ratio_30d = mean((upper_wick + lower_wick) / max(body, 1e-9))
  • max_move_atr_30d    = max(|close-prev_close| / ATR14) (manipulation spike)
  • trend_persistence   = % bars closing same direction as previous
  • range_efficiency    = |close[end] - close[start]| / sum(|close[i]-close[i-1]|)

The portfolio sim runs only on symbols passing the filter at each MONTH boundary.
"""
from __future__ import annotations
import sys, json
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
    load, build_regime, simulate_portfolio,
    PortfolioSignal, _ts_period_ms, atr, ema, sma, rsi,
    _emit, regime_at, metrics_dict_from_trades,
    START_BALANCE, RISK_PER_TRADE, MAX_POSITIONS,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
)
from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2

CACHE_DIR = Path(__file__).parent / "data_cache"


def list_available_symbols(tag="365d", required_tfs=("1d", "4h", "1h")) -> list[str]:
    """Return symbols that have ALL required TFs at the given tag."""
    syms = set()
    for tf in required_tfs:
        found = set()
        for p in CACHE_DIR.glob(f"*_{tf}_{tag}.parquet"):
            name = p.name.replace(f"_USDT_USDT_{tf}_{tag}.parquet", "")
            found.add(name)
        if not syms:
            syms = found
        else:
            syms = syms & found
    return sorted(syms)


# ─────────────────────────────────────────────────────────────────────────────
# COIN METRICS (per symbol)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CoinMonthlyMetrics:
    sym: str
    month: str           # "YYYY-MM"
    month_start_ts: int  # ms
    dollar_volume_30d: float
    realized_vol_30d: float    # annualized
    wick_body_ratio: float
    max_move_atr: float
    trend_persistence: float
    range_efficiency: float
    n_bars_30d: int


def compute_monthly_metrics(sym: str, tag="365d") -> list[CoinMonthlyMetrics]:
    """Compute rolling 30d metrics on 1d data, snapshot at each month start."""
    df = load(sym, "1d", tag)
    if df is None or len(df) < 60: return []
    ts = df["timestamp"].to_numpy()
    op = df["open"].to_numpy(); cl = df["close"].to_numpy()
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy()
    vo = df["volume"].to_numpy()
    a = atr(hi, lo, cl, 14)
    n = len(cl)
    out = []
    # daily log returns
    log_ret = np.zeros(n)
    log_ret[1:] = np.log(cl[1:] / np.where(cl[:-1] > 0, cl[:-1], 1))
    direction = np.sign(cl - np.roll(cl, 1))
    direction[0] = 0
    # iterate months
    df_dt = pd.to_datetime(ts, unit="ms", utc=True)
    df_idx = pd.DataFrame({"i": np.arange(n), "dt": df_dt})
    df_idx["month"] = df_idx["dt"].dt.to_period("M")
    months = df_idx.groupby("month")["i"].first()
    for m, i in months.items():
        if i < 30: continue
        s = max(0, i - 30); e = i
        seg_cl = cl[s:e]
        seg_op = op[s:e]; seg_hi = hi[s:e]; seg_lo = lo[s:e]; seg_vo = vo[s:e]
        seg_atr = a[s:e]; seg_dir = direction[s:e]
        if len(seg_cl) < 10: continue
        # Dollar volume
        dvol = float(np.nansum(seg_cl * seg_vo))
        # Realized vol (annualized)
        rv = float(np.nanstd(log_ret[s:e], ddof=0) * np.sqrt(365))
        # Wick/body ratio
        body = np.abs(seg_cl - seg_op)
        upper_wick = seg_hi - np.maximum(seg_cl, seg_op)
        lower_wick = np.minimum(seg_cl, seg_op) - seg_lo
        wbr = float(np.nanmean((upper_wick + lower_wick) / np.maximum(body, 1e-9)))
        # Max move ratio (manipulation spike)
        moves = np.abs(np.diff(seg_cl))
        atr_seg = seg_atr[1:]
        ratio = moves / np.where(atr_seg > 0, atr_seg, np.nan)
        mma = float(np.nanmax(ratio)) if len(ratio) > 0 else 0
        # Trend persistence (% bars closing same direction as prev)
        same_dir = (seg_dir[1:] == seg_dir[:-1]).sum()
        tp = same_dir / max(1, len(seg_dir) - 1)
        # Range efficiency
        net_move = abs(seg_cl[-1] - seg_cl[0])
        path = float(np.nansum(np.abs(np.diff(seg_cl))))
        re = net_move / max(path, 1e-9)
        out.append(CoinMonthlyMetrics(
            sym=sym, month=str(m), month_start_ts=int(ts[i]),
            dollar_volume_30d=dvol, realized_vol_30d=rv,
            wick_body_ratio=wbr, max_move_atr=mma,
            trend_persistence=tp, range_efficiency=re,
            n_bars_30d=len(seg_cl),
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SELECTORS — produce {month: [active_symbols]}
# ─────────────────────────────────────────────────────────────────────────────

def selector_baseline(symbols: list[str]) -> dict[str, list[str]]:
    """Same set every month."""
    return {"all": symbols}


def selector_top_volume(metrics_by_sym: dict[str, list[CoinMonthlyMetrics]],
                        top_n: int = 50) -> dict[str, list[str]]:
    """For each month, select top-N symbols by dollar_volume_30d."""
    by_month = defaultdict(list)
    for sym, mlist in metrics_by_sym.items():
        for m in mlist:
            by_month[m.month].append((sym, m.dollar_volume_30d))
    out = {}
    for month, lst in by_month.items():
        lst.sort(key=lambda x: x[1], reverse=True)
        out[month] = [s for s, _ in lst[:top_n]]
    return out


def selector_vol_band(metrics_by_sym, low_pct=20, high_pct=70) -> dict[str, list[str]]:
    """Per month, keep coins whose realized_vol is in [low_pct, high_pct] percentile.
    Excludes both 'dead' and 'manic' coins."""
    by_month = defaultdict(list)
    for sym, mlist in metrics_by_sym.items():
        for m in mlist:
            by_month[m.month].append((sym, m.realized_vol_30d))
    out = {}
    for month, lst in by_month.items():
        if not lst: continue
        vols = np.array([v for _, v in lst])
        lo_thresh = np.percentile(vols, low_pct)
        hi_thresh = np.percentile(vols, high_pct)
        kept = [s for s, v in lst if lo_thresh <= v <= hi_thresh]
        out[month] = kept
    return out


def selector_clean(metrics_by_sym, max_wick_pct=70, max_spike_pct=70) -> dict[str, list[str]]:
    """Exclude coins in top X% by wick_body_ratio AND max_move_atr.
    Aggressive wicks + huge spikes = manipulation."""
    by_month = defaultdict(list)
    for sym, mlist in metrics_by_sym.items():
        for m in mlist:
            by_month[m.month].append((sym, m.wick_body_ratio, m.max_move_atr))
    out = {}
    for month, lst in by_month.items():
        if not lst: continue
        wbr = np.array([w for _, w, _ in lst])
        mma = np.array([s for _, _, s in lst])
        wbr_thresh = np.percentile(wbr, max_wick_pct)
        mma_thresh = np.percentile(mma, max_spike_pct)
        kept = [s for s, w, sp in lst if w <= wbr_thresh and sp <= mma_thresh]
        out[month] = kept
    return out


def selector_composite(metrics_by_sym, top_n=50) -> dict[str, list[str]]:
    """My methodology: composite score per symbol per month.
       score = 0.4×liq_rank + 0.3×stability_rank + 0.3×cleanness_rank
       liq_rank      = percentile of dollar_volume_30d (higher = better)
       stability_rank = inverse percentile of realized_vol if too volatile;
                        penalty if too low (dead). Triangular peak at P50.
       cleanness_rank = 1 - percentile of (wick_body × max_move_atr)
    """
    by_month = defaultdict(list)
    for sym, mlist in metrics_by_sym.items():
        for m in mlist:
            by_month[m.month].append(m)
    out = {}
    for month, mlist in by_month.items():
        if len(mlist) < 5: continue
        dvol = np.array([m.dollar_volume_30d for m in mlist])
        rv = np.array([m.realized_vol_30d for m in mlist])
        wb = np.array([m.wick_body_ratio for m in mlist])
        mm = np.array([m.max_move_atr for m in mlist])
        # Ranks (0..1)
        def pctrank(arr):
            ord_idx = np.argsort(arr)
            r = np.empty_like(arr, dtype=float)
            r[ord_idx] = np.linspace(0, 1, len(arr))
            return r
        liq = pctrank(dvol)  # higher = better
        # Stability: peak at median vol (avoid both dead and crazy)
        rv_pct = pctrank(rv)
        stability = 1 - np.abs(rv_pct - 0.5) * 2  # triangular peak at median
        # Cleanness: penalize high wick + high spike
        ugly = wb * mm
        clean = 1 - pctrank(ugly)
        score = 0.4 * liq + 0.3 * stability + 0.3 * clean
        ranked = sorted(zip([m.sym for m in mlist], score), key=lambda x: -x[1])
        out[month] = [s for s, _ in ranked[:top_n]]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# COLLECT V3 SIGNALS WITH MONTH-AWARE FILTER
# ─────────────────────────────────────────────────────────────────────────────

def collect_v3_signals_filtered(syms: list[str], regime,
                                  active_by_month: dict[str, list[str]],
                                  tag="365d") -> tuple[list, dict]:
    """Run V3 signals across `syms`, but only emit signals whose timestamp's
    month is in active_by_month and the symbol is in active list for that month."""
    sigs = []; sim_dfs = {}
    # Helper to check active
    def is_active(sym, ts_ms):
        if not active_by_month: return True
        m = pd.Timestamp(ts_ms, unit="ms", tz="UTC").to_period("M")
        m_str = str(m)
        if m_str in active_by_month:
            return sym in active_by_month[m_str]
        # Otherwise default to checking if "all" is the key
        if "all" in active_by_month:
            return sym in active_by_month["all"]
        return False

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
            add("smacross_pivot_1d", df_1d,
                strat_smacross_pivot_1d(df_1d, regime, tp_r=4.0), sym)
        df_4h = load(sym, "4h", tag)
        if df_4h is not None:
            add("supertrend_pivot_4h", df_4h,
                strat_supertrend_pivot_4h(df_4h, regime, tp_r=2.5), sym)
            add("rsi_reversion_v2", df_4h,
                _strat_rsi_rev_v2(df_4h, regime, sl_atr=2.0, tp_r=3.0), sym)
            add("fib_pump_4h", df_4h,
                strat_fib_pump_1h(df_4h, regime, consolidation_bars=18,
                                   pump_min_atr=3.0, pump_max_bars=6,
                                   retrace_min=0.382, retrace_max=0.618,
                                   tp_extension=1.0, sl_buffer_atr=0.5), sym)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


def run_filter(name: str, syms: list[str], regime,
               active_by_month: dict[str, list[str]],
               risk: float = 0.03, max_pos: int = 5, tag="365d"):
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  FILTER: {name}  (risk={risk*100:.1f}%, max_pos={max_pos}, tag={tag})")
    if active_by_month and "all" not in active_by_month:
        sizes = [len(v) for v in active_by_month.values()]
        if sizes:
            print(f"  Active universe per month: avg={np.mean(sizes):.1f}, "
                  f"min={min(sizes)}, max={max(sizes)}, months={len(sizes)}")
    print(f"══════════════════════════════════════════════════════════════════")

    sigs, sim_dfs = collect_v3_signals_filtered(syms, regime, active_by_month, tag)
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
        print(f"  Avg monthly: {np.mean(rets):+.1f}%   Median: {np.median(rets):+.1f}%   Std: {np.std(rets):.1f}%")
        print(f"  Profitable months: {sum(1 for r in rets if r > 0)}/{len(rets)}")
        print(f"  Best: {max(rets):+.1f}%   Worst: {min(rets):+.1f}%")
    per = defaultdict(list)
    for t in res["trades"]: per[t["strat"]].append(t)
    print(f"  Per-strategy:")
    for sn, tlist in sorted(per.items()):
        mm = metrics_dict_from_trades(tlist, START_BALANCE)
        print(f"    {sn:25s}: n={mm.get('n',0):4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f}")
    return {"name": name, "metrics": m, "monthly": rets,
            "n_trades": m["n"], "return_pct": m["return_pct"]}


def main():
    tag = "365d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)

    # Use ALL available 365d symbols
    all_syms = list_available_symbols(tag, required_tfs=("1d", "4h", "1h"))
    print(f"\nAvailable symbols on {tag}: {len(all_syms)}")
    print(f"  {' '.join(all_syms)}")

    # Compute monthly metrics for each
    print(f"\nComputing monthly metrics...")
    metrics_by_sym = {}
    for sym in all_syms:
        ml = compute_monthly_metrics(sym, tag)
        if ml: metrics_by_sym[sym] = ml
    print(f"  Computed for {len(metrics_by_sym)} symbols")

    results = []

    # 1. Baseline: original 27 symbols
    baseline = ["ADA","ALGO","APT","ARB","ATOM","AVAX","BCH","BNB","BTC",
                "DOGE","DOT","ETH","FIL","HBAR","ICP","INJ","LINK","LTC",
                "NEAR","OP","SOL","SUI","TON","TRX","UNI","XLM","XRP"]
    baseline_in_avail = [s for s in baseline if s in all_syms]
    print(f"\nBaseline (27): {len(baseline_in_avail)} in current data")
    r = run_filter("BASELINE_27", baseline_in_avail, regime, {"all": baseline_in_avail},
                   risk=0.03, max_pos=5, tag=tag)
    if r: results.append(r)

    # 2. All available
    r = run_filter(f"ALL_AVAIL_{len(all_syms)}", all_syms, regime,
                   {"all": all_syms}, risk=0.03, max_pos=5, tag=tag)
    if r: results.append(r)

    # 3. Top-N by rolling volume (try 50 and 30)
    for top_n in (30, 50):
        sel = selector_top_volume(metrics_by_sym, top_n=top_n)
        r = run_filter(f"TOP_{top_n}_VOLUME_ROLLING", all_syms, regime, sel,
                       risk=0.03, max_pos=5, tag=tag)
        if r: results.append(r)

    # 4. Volatility band (exclude top 30% volatile and bottom 20% calm)
    for low, high in [(20, 70), (10, 80), (30, 90)]:
        sel = selector_vol_band(metrics_by_sym, low_pct=low, high_pct=high)
        r = run_filter(f"VOL_BAND_p{low}-p{high}", all_syms, regime, sel,
                       risk=0.03, max_pos=5, tag=tag)
        if r: results.append(r)

    # 5. Clean (low manipulation): keep bottom X% by wick × spike
    for max_pct in (50, 60, 70, 80):
        sel = selector_clean(metrics_by_sym, max_wick_pct=max_pct, max_spike_pct=max_pct)
        r = run_filter(f"CLEAN_p<{max_pct}", all_syms, regime, sel,
                       risk=0.03, max_pos=5, tag=tag)
        if r: results.append(r)

    # 6. My composite
    for top_n in (30, 50):
        sel = selector_composite(metrics_by_sym, top_n=top_n)
        r = run_filter(f"COMPOSITE_top{top_n}", all_syms, regime, sel,
                       risk=0.03, max_pos=5, tag=tag)
        if r: results.append(r)

    # FINAL SUMMARY
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  UNIVERSE FILTER COMPARISON  (V3 portfolio, 3% risk, 5 max pos, {tag})")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Filter':28s} {'Trades':>7s} {'PF':>5s} {'WR%':>6s} {'avgR':>7s} {'Return':>9s} {'DD%':>6s} {'Avg/mo':>8s} {'Worst':>7s} {'Best':>7s}")
    results.sort(key=lambda r: -r["metrics"]["return_pct"])
    for r in results:
        m = r["metrics"]; rets = r["monthly"]
        worst = min(rets) if rets else 0
        best = max(rets) if rets else 0
        avgmo = float(np.mean(rets)) if rets else 0
        print(f"  {r['name']:28s} {m['n']:>7d} {m['pf']:>5.2f} {m['wr']:>5.1f} "
              f"{m['avg_r']:>+7.3f} {m['return_pct']:>+8.1f}% {m['max_dd_pct']:>5.1f}% "
              f"{avgmo:>+7.1f}% {worst:>+6.1f}% {best:>+6.1f}%")

    # Save
    out = Path("multi_agent/backtests/results/universe_filters.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([
        {**r, "monthly": r["monthly"], "metrics": r["metrics"]}
        for r in results
    ], indent=2, default=str), encoding="utf-8")
    print(f"\n  Saved: {out}")


if __name__ == "__main__":
    main()
