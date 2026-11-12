"""
full_audit_v7.py — Comprehensive parameter audit for V7 system
==============================================================

Tests every major knob:
  1. Per-strategy TP/SL/pivot grids (isolated)
  2. Portfolio-level risk%, max_pos, L4 on/off
  3. Final best-config walkforward

Baseline: current V7+L4+PivotFilter config
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
    load, build_regime, atr as _atr, PortfolioSignal, _ts_period_ms, _emit, regime_at,
    START_BALANCE,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, CACHE_DIR
from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
    find_pivots, latest_pivot_low, latest_pivot_high,
)
from multi_agent.backtests.news_aware_v7 import load_all_events
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter
from multi_agent.backtests.risk_controls_v7 import simulate_rc, build_corr_matrix

PIVOT_DIST = 7.0
DAY_MS = 86_400_000

# ─────────────────────────────────────────────────────────────────────────────
# Signal collection helpers (parametrised)
# ─────────────────────────────────────────────────────────────────────────────

def collect_single(syms, sim_dfs_out, strat_name, regime, tag,
                   bear_mode="long_only", **kwargs):
    """Collect signals for ONE strategy with custom params. Returns list[PortfolioSignal]."""
    sigs = []
    for sym in syms:
        if strat_name == "supertrend":
            df = load(sym, "4h", tag)
            if df is None: continue
            sim_dfs_out[(sym, strat_name)] = df
            raw = strat_supertrend_pivot_4h(df, regime, **kwargs)
        elif strat_name == "smacross":
            df = load(sym, "1d", tag)
            if df is None: continue
            sim_dfs_out[(sym, strat_name)] = df
            raw = strat_smacross_pivot_1d(df, regime, **kwargs)
        elif strat_name == "rsi_rev":
            df = load(sym, "4h", tag)
            if df is None: continue
            sim_dfs_out[(sym, strat_name)] = df
            raw = _strat_rsi_rev_v2(df, regime, **kwargs)
        elif strat_name == "fib_pump":
            df = load(sym, "4h", tag)
            if df is None: continue
            sim_dfs_out[(sym, strat_name)] = df
            raw = strat_fib_pump_1h(df, regime, **kwargs)
        else:
            continue
        for s in raw:
            trend, _ = regime_at(regime, s.next_open_ts)
            if bear_mode == "long_only" and trend < 0 and s.direction == "SHORT":
                continue
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts,
                sym=sym, strat=strat_name, direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs


def collect_portfolio(syms, regime, tag, bear_mode="long_only",
                      supertrend_kw=None, smacross_kw=None,
                      rsi_rev_kw=None, fib_pump_kw=None):
    """Collect all 4 strategies with custom params each."""
    supertrend_kw = supertrend_kw or dict(tp_r=2.5)
    smacross_kw   = smacross_kw   or dict(tp_r=4.0)
    rsi_rev_kw    = rsi_rev_kw    or dict(sl_atr=2.0, tp_r=3.0)
    fib_pump_kw   = fib_pump_kw   or dict(consolidation_bars=18, pump_min_atr=3.0,
                                           pump_max_bars=6, retrace_min=0.382,
                                           retrace_max=0.618, tp_extension=1.0,
                                           sl_buffer_atr=0.5)
    sim_dfs = {}
    sigs = []
    for strat, kw in [("supertrend", supertrend_kw), ("smacross", smacross_kw),
                      ("rsi_rev", rsi_rev_kw), ("fib_pump", fib_pump_kw)]:
        sigs += collect_single(syms, sim_dfs, strat, regime, tag, bear_mode, **kw)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


# ─────────────────────────────────────────────────────────────────────────────
# Metric helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(name, res, bm=None):
    m = res["metrics"]
    if m["n"] == 0:
        print(f"  {name:52s}  NO TRADES"); return
    dwr  = f"({m['wr']-bm['wr']:+.1f})"        if bm else ""
    dret = f"({m['return_pct']-bm['return_pct']:+.0f})" if bm else ""
    dpf  = f"({m['pf']-bm['pf']:+.2f})"         if bm else ""
    ddd  = f"({m['max_dd_pct']-bm['max_dd_pct']:+.1f})" if bm else ""
    print(f"  {name:52s}  n={m['n']:4d}  "
          f"WR={m['wr']:5.1f}%{dwr:8s}  "
          f"PF={m['pf']:5.2f}{dpf:8s}  "
          f"AvgR={m['avg_r']:+5.2f}  "
          f"Ret={m['return_pct']:+8.1f}%{dret:8s}  "
          f"DD={m['max_dd_pct']:5.1f}%{ddd:7s}")

def hdr():
    print(f"  {'Config':52s}  {'n':4s}  {'WR%':13s}  {'PF':12s}  "
          f"{'AvgR':6s}  {'Ret%':15s}  {'DD%':11s}")

def score(m):
    """Composite score: penalise DD heavily."""
    return m["return_pct"] - 2.5 * m["max_dd_pct"]


# ─────────────────────────────────────────────────────────────────────────────
# Per-strategy isolated tests (no portfolio cap)
# ─────────────────────────────────────────────────────────────────────────────

def test_strategy_isolated(strat_name, syms, regime, events, tag,
                            grid: list[dict]):
    """Run strategy alone (max_pos=999, no pivot filter) to find its own best params."""
    sim_dfs = {}
    print(f"\n{'─'*72}")
    print(f"  ISOLATED: {strat_name}")
    print(f"{'─'*72}")
    hdr()
    best_kw = None; best_sc = -999; best_m = None
    for kw in grid:
        sim_dfs_local = {}
        s = collect_single(syms, sim_dfs_local, strat_name, regime, tag, **kw)
        s_filt, _, _ = apply_pivot_dist_filter(s, sim_dfs_local, PIVOT_DIST)
        res = simulate_rc(s_filt, sim_dfs_local, regime, events,
                          max_pos=999, base_risk=0.04, enable_l4=True)
        m = res["metrics"]
        name_str = "  ".join(f"{k}={v}" for k, v in kw.items())
        fmt(name_str, res)
        sc = score(m)
        if m["n"] >= 5 and sc > best_sc:
            best_sc = sc; best_kw = kw; best_m = m
            sim_dfs.update(sim_dfs_local)
    if best_kw:
        print(f"\n  ★ BEST: " + "  ".join(f"{k}={v}" for k, v in best_kw.items()))
        print(f"    n={best_m['n']}  WR={best_m['wr']:.1f}%  PF={best_m['pf']:.2f}  "
              f"Ret={best_m['return_pct']:+.1f}%  DD={best_m['max_dd_pct']:.1f}%")
    return best_kw


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward helper
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(sigs_base, sigs_new, sim_dfs, regime, events, label, corr_mx=None, **rc_kw):
    all_ts = sorted(s.next_open_ts for s in sigs_new)
    if not all_ts: return
    first = all_ts[0]; last = all_ts[-1]
    cursor = first + 6*30*DAY_MS
    test_ms = 3*30*DAY_MS
    print(f"\n  Walk-forward ({label}):")
    print(f"  {'End':12s}  {'BASE Ret':>10s}  {'NEW Ret':>10s}  {'ΔRet':>8s}  {'ΔWR':>7s}  {'ΔDD':>7s}")
    wins_b = wins_n = folds = 0
    while cursor + test_ms <= last:
        te = cursor + test_ms
        sb = [s for s in sigs_base if cursor <= s.next_open_ts < te]
        sn = [s for s in sigs_new  if cursor <= s.next_open_ts < te]
        rb = simulate_rc(sb, sim_dfs, regime, events,
                         corr_matrix=corr_mx, **rc_kw)
        rn = simulate_rc(sn, sim_dfs, regime, events,
                         corr_matrix=corr_mx, **rc_kw)
        mb = rb["metrics"]; mn = rn["metrics"]
        if mb["return_pct"] > 0: wins_b += 1
        if mn["return_pct"] > 0: wins_n += 1
        folds += 1
        te_s = pd.Timestamp(te, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        print(f"  {te_s:12s}  {mb['return_pct']:>+9.1f}%  {mn['return_pct']:>+9.1f}%  "
              f"{mn['return_pct']-mb['return_pct']:>+7.1f}%  "
              f"{mn['wr']-mb['wr']:>+6.1f}  "
              f"{mn['max_dd_pct']-mb['max_dd_pct']:>+6.1f}")
        cursor += test_ms
    print(f"  Positive folds: BASE {wins_b}/{folds}  NEW {wins_n}/{folds}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "═"*72)
    print("  FULL AUDIT V7 — comprehensive parameter sweep")
    print("═"*72)

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    events = load_all_events()
    print(f"\n  Symbols: {len(syms)}  |  Events: {len(events)}")

    # Precompute sim_dfs for correlation matrix
    print("  Pre-loading data...", flush=True)
    from multi_agent.backtests.portfolio_v7 import collect_v7_signals
    sigs_v7, sim_dfs_all, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    corr_mx = build_corr_matrix(sim_dfs_all)

    # Baseline (current config)
    sigs_base, rem, _ = apply_pivot_dist_filter(sigs_v7, sim_dfs_all, PIVOT_DIST)
    res_base = simulate_rc(sigs_base, sim_dfs_all, regime, events)
    bm = res_base["metrics"]

    print(f"\n  BASELINE  n={bm['n']}  Ret={bm['return_pct']:+.1f}%  "
          f"WR={bm['wr']:.1f}%  PF={bm['pf']:.2f}  DD={bm['max_dd_pct']:.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # PART 1 — Per-strategy parameter grids (isolated)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*72)
    print("  PART 1 — Per-strategy parameter grids (isolated, max_pos=999)")
    print("═"*72)

    # ── supertrend ──
    best_st = test_strategy_isolated("supertrend", syms, regime, events, "730d", [
        dict(tp_r=1.5, buffer_atr=0.3, pivot_lookback=5),
        dict(tp_r=2.0, buffer_atr=0.3, pivot_lookback=5),
        dict(tp_r=2.5, buffer_atr=0.3, pivot_lookback=5),
        dict(tp_r=3.0, buffer_atr=0.3, pivot_lookback=5),
        dict(tp_r=2.0, buffer_atr=0.2, pivot_lookback=5),
        dict(tp_r=2.0, buffer_atr=0.5, pivot_lookback=5),
        dict(tp_r=2.5, buffer_atr=0.2, pivot_lookback=5),
        dict(tp_r=2.5, buffer_atr=0.5, pivot_lookback=5),
        dict(tp_r=2.0, buffer_atr=0.3, pivot_lookback=3),
        dict(tp_r=2.0, buffer_atr=0.3, pivot_lookback=7),
        dict(tp_r=2.5, buffer_atr=0.3, pivot_lookback=3),
        dict(tp_r=2.5, buffer_atr=0.3, pivot_lookback=7),
        dict(tp_r=3.0, buffer_atr=0.2, pivot_lookback=3),
        dict(tp_r=3.0, buffer_atr=0.2, pivot_lookback=5),
    ])

    # ── smacross ──
    best_sc = test_strategy_isolated("smacross", syms, regime, events, "730d", [
        dict(tp_r=2.0, buffer_atr=0.3, pivot_lookback=3),
        dict(tp_r=3.0, buffer_atr=0.3, pivot_lookback=3),
        dict(tp_r=4.0, buffer_atr=0.3, pivot_lookback=3),
        dict(tp_r=5.0, buffer_atr=0.3, pivot_lookback=3),
        dict(tp_r=4.0, buffer_atr=0.2, pivot_lookback=3),
        dict(tp_r=4.0, buffer_atr=0.5, pivot_lookback=3),
        dict(tp_r=4.0, buffer_atr=0.3, pivot_lookback=5),
        dict(tp_r=4.0, buffer_atr=0.3, pivot_lookback=7),
        dict(tp_r=5.0, buffer_atr=0.2, pivot_lookback=3),
        dict(tp_r=5.0, buffer_atr=0.2, pivot_lookback=5),
        dict(tp_r=3.0, buffer_atr=0.2, pivot_lookback=3),
        dict(tp_r=3.0, buffer_atr=0.2, pivot_lookback=5),
    ])

    # ── rsi_rev ──
    best_rsi = test_strategy_isolated("rsi_rev", syms, regime, events, "730d", [
        dict(sl_atr=1.5, tp_r=2.0),
        dict(sl_atr=1.5, tp_r=3.0),
        dict(sl_atr=2.0, tp_r=2.0),
        dict(sl_atr=2.0, tp_r=3.0),
        dict(sl_atr=2.0, tp_r=4.0),
        dict(sl_atr=2.5, tp_r=3.0),
        dict(sl_atr=2.5, tp_r=4.0),
        dict(sl_atr=1.5, tp_r=4.0),
        dict(sl_atr=1.0, tp_r=3.0),
        dict(sl_atr=1.0, tp_r=2.0),
    ])

    # ── fib_pump ──
    best_fib = test_strategy_isolated("fib_pump", syms, regime, events, "730d", [
        dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=6,
             retrace_min=0.382, retrace_max=0.618, tp_extension=1.0, sl_buffer_atr=0.5),
        dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=6,
             retrace_min=0.382, retrace_max=0.618, tp_extension=1.5, sl_buffer_atr=0.5),
        dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=6,
             retrace_min=0.382, retrace_max=0.618, tp_extension=0.5, sl_buffer_atr=0.3),
        dict(consolidation_bars=18, pump_min_atr=2.5, pump_max_bars=6,
             retrace_min=0.382, retrace_max=0.618, tp_extension=1.0, sl_buffer_atr=0.5),
        dict(consolidation_bars=12, pump_min_atr=3.0, pump_max_bars=6,
             retrace_min=0.382, retrace_max=0.618, tp_extension=1.0, sl_buffer_atr=0.5),
        dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=8,
             retrace_min=0.382, retrace_max=0.618, tp_extension=1.0, sl_buffer_atr=0.5),
        dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=6,
             retrace_min=0.30, retrace_max=0.70, tp_extension=1.0, sl_buffer_atr=0.5),
        dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=6,
             retrace_min=0.382, retrace_max=0.618, tp_extension=1.0, sl_buffer_atr=0.3),
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # PART 2 — Portfolio-level risk grid (best per-strat params)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*72)
    print("  PART 2 — Portfolio risk% and max_pos")
    print("═"*72)

    # Build portfolio sigs with best per-strategy params
    st_kw  = best_st  or dict(tp_r=2.5, buffer_atr=0.3, pivot_lookback=5)
    sc_kw  = best_sc  or dict(tp_r=4.0, buffer_atr=0.3, pivot_lookback=3)
    rsi_kw = best_rsi or dict(sl_atr=2.0, tp_r=3.0)
    fib_kw = best_fib or dict(consolidation_bars=18, pump_min_atr=3.0,
                               pump_max_bars=6, retrace_min=0.382,
                               retrace_max=0.618, tp_extension=1.0, sl_buffer_atr=0.5)

    sigs_opt, sim_dfs_opt = collect_portfolio(
        syms, regime, "730d",
        supertrend_kw=st_kw, smacross_kw=sc_kw,
        rsi_rev_kw=rsi_kw, fib_pump_kw=fib_kw,
    )
    sigs_opt_f, _, _ = apply_pivot_dist_filter(sigs_opt, sim_dfs_opt, PIVOT_DIST)

    print(f"\n  Optimised signals: {len(sigs_opt_f)}")
    print(f"\n  ── Risk% grid ──────────────────────────────────────────────────")
    hdr()
    best_risk = 0.04; best_risk_sc = score(bm)
    for risk in [0.02, 0.025, 0.03, 0.035, 0.04, 0.05]:
        res = simulate_rc(sigs_opt_f, sim_dfs_opt, regime, events, base_risk=risk)
        m = res["metrics"]
        fmt(f"risk={risk*100:.1f}%", res, bm)
        sc = score(m)
        if m["n"] >= 60 and sc > best_risk_sc:
            best_risk_sc = sc; best_risk = risk

    print(f"\n  ── max_pos grid ─────────────────────────────────────────────────")
    hdr()
    best_maxpos = 5; best_mp_sc = score(bm)
    for mp in [3, 4, 5, 6]:
        res = simulate_rc(sigs_opt_f, sim_dfs_opt, regime, events,
                          base_risk=best_risk, max_pos=mp)
        m = res["metrics"]
        fmt(f"max_pos={mp}", res, bm)
        sc = score(m)
        if m["n"] >= 60 and sc > best_mp_sc:
            best_mp_sc = sc; best_maxpos = mp

    # ═══════════════════════════════════════════════════════════════════════
    # PART 3 — L4 news filter on/off
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*72)
    print("  PART 3 — News/L4 filter impact")
    print("═"*72)
    hdr()
    res_l4_off = simulate_rc(sigs_opt_f, sim_dfs_opt, regime, events,
                              base_risk=best_risk, max_pos=best_maxpos, enable_l4=False)
    res_l4_on  = simulate_rc(sigs_opt_f, sim_dfs_opt, regime, events,
                              base_risk=best_risk, max_pos=best_maxpos, enable_l4=True)
    fmt("L4 OFF (no macro pause)", res_l4_off, bm)
    fmt("L4 ON  (FOMC pause)",     res_l4_on,  bm)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 4 — Correlation filter (RC5)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*72)
    print("  PART 4 — Correlation filter (RC5)")
    print("═"*72)
    hdr()
    best_l4 = res_l4_on["metrics"]["return_pct"] >= res_l4_off["metrics"]["return_pct"]
    for max_co, thr in [(0, 0.80), (1, 0.80), (1, 0.85), (2, 0.80)]:
        res = simulate_rc(sigs_opt_f, sim_dfs_opt, regime, events,
                          base_risk=best_risk, max_pos=best_maxpos,
                          enable_l4=best_l4,
                          max_corr_open=max_co, corr_threshold=thr,
                          corr_matrix=corr_mx)
        fmt(f"corr: max_open={max_co} thr={thr:.2f}", res, bm)

    # ═══════════════════════════════════════════════════════════════════════
    # PART 5 — FULL BEST CONFIG summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "═"*72)
    print("  PART 5 — FULL BEST CONFIG vs BASELINE")
    print("═"*72)

    res_final = simulate_rc(
        sigs_opt_f, sim_dfs_opt, regime, events,
        base_risk=best_risk, max_pos=best_maxpos,
        enable_l4=True,
        max_corr_open=1, corr_threshold=0.80, corr_matrix=corr_mx,
    )
    mf = res_final["metrics"]

    print(f"\n  Strategy params:")
    print(f"    supertrend : " + "  ".join(f"{k}={v}" for k, v in st_kw.items()))
    print(f"    smacross   : " + "  ".join(f"{k}={v}" for k, v in sc_kw.items()))
    print(f"    rsi_rev    : " + "  ".join(f"{k}={v}" for k, v in rsi_kw.items()))
    print(f"    fib_pump   : " + "  ".join(f"{k}={v}" for k, v in fib_kw.items()))
    print(f"  Portfolio params:")
    print(f"    risk={best_risk*100:.1f}%  max_pos={best_maxpos}  "
          f"L4=ON  corr_max=1@0.80  pivot_dist<{PIVOT_DIST}×ATR")

    print(f"\n  {'':30s}  {'n':>4s}  {'Ret%':>8s}  {'WR%':>6s}  "
          f"{'PF':>5s}  {'AvgR':>6s}  {'DD%':>6s}")
    print(f"  {'─'*68}")
    print(f"  {'BASELINE (original V7 params)':30s}  "
          f"{bm['n']:>4d}  {bm['return_pct']:>+8.1f}%  {bm['wr']:>6.1f}%  "
          f"{bm['pf']:>5.2f}  {bm['avg_r']:>+6.3f}  {bm['max_dd_pct']:>6.1f}%")
    print(f"  {'BEST CONFIG (audited)':30s}  "
          f"{mf['n']:>4d}  {mf['return_pct']:>+8.1f}%  {mf['wr']:>6.1f}%  "
          f"{mf['pf']:>5.2f}  {mf['avg_r']:>+6.3f}  {mf['max_dd_pct']:>6.1f}%")
    print(f"  {'Delta':30s}  "
          f"{'':>4s}  {mf['return_pct']-bm['return_pct']:>+8.1f}%  "
          f"{mf['wr']-bm['wr']:>+6.1f}  "
          f"{mf['pf']-bm['pf']:>+5.2f}  "
          f"{mf['avg_r']-bm['avg_r']:>+6.3f}  "
          f"{mf['max_dd_pct']-bm['max_dd_pct']:>+6.1f}%")

    # ── Walk-forward ─────────────────────────────────────────────────────
    walk_forward(sigs_base, sigs_opt_f, sim_dfs_opt, regime, events,
                 "BEST CONFIG vs BASELINE", corr_mx=corr_mx,
                 max_corr_open=1, corr_threshold=0.80,
                 base_risk=best_risk, max_pos=best_maxpos, enable_l4=True)

    # ── Final parameter summary for live implementation ──────────────────
    print("\n" + "═"*72)
    print("  LIVE IMPLEMENTATION PARAMETERS")
    print("═"*72)
    print(f"""
  Copy these into v7_agent.py / config.py:

  ── Signal parameters ─────────────────────────────────────────────────
  SUPERTREND_TP_R        = {st_kw.get('tp_r', 2.5)}
  SUPERTREND_BUFFER_ATR  = {st_kw.get('buffer_atr', 0.3)}
  SUPERTREND_PIV_LB      = {st_kw.get('pivot_lookback', 5)}

  SMACROSS_TP_R          = {sc_kw.get('tp_r', 4.0)}
  SMACROSS_BUFFER_ATR    = {sc_kw.get('buffer_atr', 0.3)}
  SMACROSS_PIV_LB        = {sc_kw.get('pivot_lookback', 3)}

  RSI_SL_ATR             = {rsi_kw.get('sl_atr', 2.0)}
  RSI_TP_R               = {rsi_kw.get('tp_r', 3.0)}

  FIB_TP_EXT             = {fib_kw.get('tp_extension', 1.0)}
  FIB_SL_BUFFER          = {fib_kw.get('sl_buffer_atr', 0.5)}

  ── Portfolio parameters ────────────────────────────────────────────────
  RISK_PER_TRADE         = {best_risk}    # fraction of equity
  MAX_POSITIONS          = {best_maxpos}
  PIVOT_DIST_MAX_ATR     = {PIVOT_DIST}
  CORR_MAX_OPEN          = 1
  CORR_THRESHOLD         = 0.80
  ENABLE_L4              = True
""")


if __name__ == "__main__":
    main()
