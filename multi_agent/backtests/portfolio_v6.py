"""
portfolio_v6.py — V5 + 5 new high-impact features (toggleable)
================================================================
Each feature can be toggled independently for incremental testing.

  • pyramid_scaling   : at +1R, add 50% qty + move SL to BE
  • per_strat_vol     : different vol thresholds per strategy
  • vol_scaled_sizing : risk = 2-4% based on ATR percentile
  • correlation_cap   : max 1 concurrent pos in correlated cluster
  • funding_skip      : skip if funding > 0.05%/8h in trade direction

Built on top of V5 winning config:
  • Strategies: smacross_pivot_1d (4R), supertrend_pivot_4h (5R),
                rsi_reversion_v2, fib_pump_4h
  • Universe: BASELINE 27 (no composite, no daily rebalance)
  • Volume filter: 1.5× × 30-bar avg (default unless per_strat overrides)
  • Risk: 4% (V5 winner)
  • Max positions: 5
"""
from __future__ import annotations
import sys, json
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass, field

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, sma, _ts_period_ms,
    PortfolioSignal, START_BALANCE, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H,
    MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN, SL_COOLDOWN_MS,
    metrics_dict_from_trades, _monthly_returns, regime_at,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
)
from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2

CACHE_DIR = Path(__file__).parent / "data_cache"
BASELINE = ["ADA","ALGO","APT","ARB","ATOM","AVAX","BCH","BNB","BTC",
            "DOGE","DOT","ETH","FIL","HBAR","ICP","INJ","LINK","LTC",
            "NEAR","OP","SOL","SUI","TON","TRX","UNI","XLM","XRP"]


# ─────────────────────────────────────────────────────────────────────────────
# CORRELATION CLUSTER COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def build_correlation_clusters(syms, tag, threshold=0.85, window_days=60):
    """Compute rolling correlation between symbols, group highly-correlated.
    Returns dict[sym → cluster_id] over time (constant for simplicity).
    threshold=0.85 means symbols with corr ≥ 0.85 are in same cluster."""
    # Load 1d closes for all syms, align on common timestamps
    rets_by_sym = {}
    for s in syms:
        df = load(s, "1d", tag)
        if df is None: continue
        cl = df["close"].to_numpy()
        ts = df["timestamp"].to_numpy()
        if len(cl) < window_days + 5: continue
        log_r = np.zeros(len(cl))
        log_r[1:] = np.log(cl[1:] / np.where(cl[:-1] > 0, cl[:-1], 1))
        rets_by_sym[s] = (ts, log_r)
    # Use the first symbol's timestamps as reference
    if not rets_by_sym: return {}
    ref_ts = rets_by_sym[list(rets_by_sym.keys())[0]][0]
    # Build matrix
    syms_list = list(rets_by_sym.keys())
    n_bars = len(ref_ts)
    mat = np.full((len(syms_list), n_bars), np.nan)
    for i, s in enumerate(syms_list):
        ts, r = rets_by_sym[s]
        # Align by timestamp
        for j in range(min(len(ts), n_bars)):
            idx = int(np.searchsorted(ref_ts, ts[j], side="left"))
            if 0 <= idx < n_bars:
                mat[i, idx] = r[j]
    # Use last `window_days` bars to compute correlation matrix
    last_window = mat[:, -window_days:]
    valid = ~np.isnan(last_window)
    corr_mat = np.zeros((len(syms_list), len(syms_list)))
    for i in range(len(syms_list)):
        for j in range(i+1, len(syms_list)):
            both = valid[i] & valid[j]
            if both.sum() < 20: continue
            ri = last_window[i][both]
            rj = last_window[j][both]
            if ri.std() > 0 and rj.std() > 0:
                c = np.corrcoef(ri, rj)[0, 1]
                corr_mat[i, j] = c
                corr_mat[j, i] = c
    # Greedy clustering: assign each symbol to a cluster
    sym_cluster = {}
    cluster_idx = 0
    for i, s in enumerate(syms_list):
        if s in sym_cluster: continue
        sym_cluster[s] = cluster_idx
        for j, t in enumerate(syms_list):
            if t in sym_cluster: continue
            if corr_mat[i, j] >= threshold:
                sym_cluster[t] = cluster_idx
        cluster_idx += 1
    # Symbols not in any cluster get unique IDs (already set above)
    return sym_cluster


# ─────────────────────────────────────────────────────────────────────────────
# FUNDING RATE LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_funding(sym):
    p = CACHE_DIR / f"{sym}_USDT_USDT_funding_730d.parquet"
    if not p.exists(): return None
    try:
        df = pd.read_parquet(p)
        return df
    except Exception:
        return None


def funding_at(funding_df, ts_ms):
    """Return funding_rate active at ts_ms (or None)."""
    if funding_df is None or funding_df.empty: return None
    arr_ts = funding_df["timestamp"].to_numpy()
    arr_fr = funding_df["funding_rate"].to_numpy()
    idx = np.searchsorted(arr_ts, ts_ms, side="right") - 1
    if idx < 0: return None
    return float(arr_fr[idx])


# ─────────────────────────────────────────────────────────────────────────────
# VOL-SCALED RISK (per-symbol)
# ─────────────────────────────────────────────────────────────────────────────

def get_vol_scaled_risk(df, j, base_risk=0.04, low_vol_risk=0.04, high_vol_risk=0.02,
                        atr_window=14, percentile_window=100):
    """Compute risk based on ATR percentile of last `percentile_window` bars."""
    if j < percentile_window + atr_window: return base_risk
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    a = atr(hi, lo, cl, atr_window)
    cur_atr = a[j]
    if np.isnan(cur_atr): return base_risk
    seg = a[j-percentile_window:j]
    seg = seg[~np.isnan(seg)]
    if len(seg) < 30: return base_risk
    pctile = (seg < cur_atr).sum() / len(seg) * 100
    # Linear interpolation: low pctile = high risk, high pctile = low risk
    if pctile <= 50:
        return low_vol_risk
    elif pctile >= 90:
        return high_vol_risk
    else:
        # 50-90 range: linear from low_vol_risk down to high_vol_risk
        frac = (pctile - 50) / 40
        return low_vol_risk * (1 - frac) + high_vol_risk * frac


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL COLLECTION (V5 base + per-strategy vol filter + funding skip)
# ─────────────────────────────────────────────────────────────────────────────

def _vol_ok(df, j, threshold, window):
    if j < window + 1: return True
    vo = df["volume"].to_numpy()
    avg = np.nanmean(vo[max(0, j-window):j])
    return vo[j] >= threshold * avg if avg > 0 else True


def collect_v6_signals(syms, regime, tag,
                        # TPs (V5 winners)
                        smacross_tp=4.0, supertrend_tp=5.0, rsi_tp=3.0,
                        # Volume thresholds (per-strategy)
                        vol_smacross=1.5, vol_supertrend=1.5,
                        vol_rsi=1.5, vol_fib=1.5, vol_window=30,
                        # Funding filter
                        use_funding=False, funding_thresh=0.0005):
    sigs = []; sim_dfs = {}; funding_data = {}

    if use_funding:
        for sym in syms:
            funding_data[sym] = load_funding(sym)

    def add(strat, df, slist, sym, vol_thresh):
        if df is None: return
        sim_dfs[(sym, strat)] = df
        ts = df["timestamp"].to_numpy()
        fund_df = funding_data.get(sym) if use_funding else None
        for s in slist:
            j = int(np.searchsorted(ts, s.next_open_ts, side="left"))
            if j >= len(ts): continue
            # Volume filter
            if not _vol_ok(df, j, vol_thresh, vol_window): continue
            # Funding filter: skip if funding rate is in our direction & extreme
            if use_funding and fund_df is not None:
                fr = funding_at(fund_df, s.next_open_ts)
                if fr is not None:
                    # If LONG and funding > thresh → crowd is long, expensive to hold → skip
                    if s.direction == "LONG" and fr > funding_thresh: continue
                    # If SHORT and funding < -thresh → crowd is short → skip
                    if s.direction == "SHORT" and fr < -funding_thresh: continue
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts, sym=sym,
                strat=strat, direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    for sym in syms:
        df_1d = load(sym, "1d", tag)
        if df_1d is not None:
            add("smacross", df_1d, strat_smacross_pivot_1d(df_1d, regime, tp_r=smacross_tp),
                sym, vol_smacross)
        df_4h = load(sym, "4h", tag)
        if df_4h is not None:
            add("supertrend", df_4h, strat_supertrend_pivot_4h(df_4h, regime, tp_r=supertrend_tp),
                sym, vol_supertrend)
            add("rsi_rev", df_4h, _strat_rsi_rev_v2(df_4h, regime, sl_atr=2.0, tp_r=rsi_tp),
                sym, vol_rsi)
            add("fib_pump", df_4h, strat_fib_pump_1h(df_4h, regime, consolidation_bars=18,
                                                       pump_min_atr=3.0, pump_max_bars=6,
                                                       retrace_min=0.382, retrace_max=0.618,
                                                       tp_extension=1.0, sl_buffer_atr=0.5),
                sym, vol_fib)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM PORTFOLIO SIMULATOR (with pyramid + correlation cap + vol-scaled risk)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_v6(sigs, sim_dfs, regime,
                start_balance=START_BALANCE,
                base_risk=0.03,
                max_pos=5,
                # Toggles
                use_pyramid=False, pyramid_at_r=2.0, pyramid_add=0.5,
                pyramid_sl_offset_r=0.3,
                use_correlation_cap=False, sym_cluster=None,
                use_vol_scaled=False, low_vol_risk=0.04, high_vol_risk=0.02,
                sl_cooldown_ms=SL_COOLDOWN_MS):
    """Compounding simulator with pyramid + correlation + vol-scaled risk."""
    balance = start_balance
    open_pos = {}      # (sym, strat) → state
    sym_cool = {}      # sym → unlock_ts
    cluster_pos = {}   # cluster_id → count
    trades = []

    def get_cluster(sym):
        if not sym_cluster: return None
        return sym_cluster.get(sym, f"_solo_{sym}")

    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos: continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts: continue

        # Realize closed positions
        to_close = [k for k, p in open_pos.items()
                     if p["exit_ts"] is not None and p["exit_ts"] <= sig.next_open_ts]
        for k in to_close:
            pos = open_pos[k]
            balance = max(0.01, balance + pos["pnl_realized"])
            trades.append(_pos_to_trade(pos))
            if pos["reason"] == "SL":
                sym_cool[pos["sym"]] = pos["exit_ts"] + sl_cooldown_ms
            cl_id = pos.get("cluster_id")
            if cl_id is not None:
                cluster_pos[cl_id] = max(0, cluster_pos.get(cl_id, 1) - 1)
            del open_pos[k]

        # Limit checks
        if len(open_pos) >= max_pos: continue

        cluster_id = get_cluster(sig.sym)
        if use_correlation_cap and cluster_id is not None:
            if cluster_pos.get(cluster_id, 0) >= 1:
                continue

        df = sim_dfs.get(key)
        if df is None: continue
        ts = df["timestamp"].to_numpy()
        op = df["open"].to_numpy(); hi = df["high"].to_numpy()
        lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
        n = len(ts)
        period_ms = _ts_period_ms(ts)
        funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))

        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1: continue

        # Compute risk for this trade
        if use_vol_scaled:
            risk = get_vol_scaled_risk(df, j, base_risk=base_risk,
                                         low_vol_risk=low_vol_risk,
                                         high_vol_risk=high_vol_risk)
        else:
            risk = base_risk

        # Entry
        d = sig.direction
        entry_p = op[j] * (1 + SLIPPAGE) if d == "LONG" else op[j] * (1 - SLIPPAGE)
        sl0 = sig.sl
        if d == "LONG":
            dist = entry_p - sl0
            if dist <= 0: continue
            tp_p = entry_p + (sig.tp - sig.entry) * (dist / sig.dist) if sig.dist > 0 else sig.tp
        else:
            dist = sl0 - entry_p
            if dist <= 0: continue
            tp_p = entry_p - (sig.entry - sig.tp) * (dist / sig.dist) if sig.dist > 0 else sig.tp

        risk_usd = balance * risk
        qty_initial = risk_usd / dist

        # Walk forward — handle pyramid + final exit
        cur_qty = qty_initial
        cur_sl = sl0
        pyramid_done = False
        # Track P&L incrementally
        seg_pnl = 0.0
        seg_costs_entry = entry_p * qty_initial * TAKER_FEE
        seg_costs_exit = 0.0
        seg_funding = 0.0
        bars_held = 0
        exit_bar = None; exit_p = None; reason = "timeout"
        # For pyramid second entry tracking
        pyramid_entry_p = None
        pyramid_qty = 0.0
        pyramid_entry_fee = 0.0

        for k in range(j, min(j + 500, n)):
            h_k = hi[k]; l_k = lo[k]
            bars_held = k - j + 1

            # Pyramid trigger
            if use_pyramid and not pyramid_done:
                if d == "LONG" and h_k >= entry_p + pyramid_at_r * dist:
                    add_qty = qty_initial * pyramid_add
                    pyramid_entry_p = entry_p + pyramid_at_r * dist
                    pyramid_qty = add_qty
                    pyramid_entry_fee = pyramid_entry_p * add_qty * TAKER_FEE
                    cur_qty += add_qty
                    # SL = BE + offset (locks in some profit on initial qty)
                    cur_sl = entry_p + pyramid_sl_offset_r * dist
                    pyramid_done = True
                elif d == "SHORT" and l_k <= entry_p - pyramid_at_r * dist:
                    add_qty = qty_initial * pyramid_add
                    pyramid_entry_p = entry_p - pyramid_at_r * dist
                    pyramid_qty = add_qty
                    pyramid_entry_fee = pyramid_entry_p * add_qty * TAKER_FEE
                    cur_qty += add_qty
                    cur_sl = entry_p - pyramid_sl_offset_r * dist
                    pyramid_done = True

            if d == "LONG":
                if l_k <= cur_sl:
                    exit_p = cur_sl; reason = "SL" if not pyramid_done else "BE_SL"
                    exit_bar = k; break
                if h_k >= tp_p:
                    exit_p = tp_p; reason = "TP"; exit_bar = k; break
            else:
                if h_k >= cur_sl:
                    exit_p = cur_sl; reason = "SL" if not pyramid_done else "BE_SL"
                    exit_bar = k; break
                if l_k <= tp_p:
                    exit_p = tp_p; reason = "TP"; exit_bar = k; break

        if exit_bar is None:
            exit_p = cl[min(j + 499, n - 1)]
            exit_bar = min(j + 499, n - 1)
            reason = "timeout"
            bars_held = exit_bar - j + 1

        # Compute final P&L
        if d == "LONG":
            slip_exit = exit_p * (1 - SLIPPAGE)
            gross_initial = (slip_exit - entry_p) * qty_initial
            seg_pnl += gross_initial
            if pyramid_done:
                gross_pyramid = (slip_exit - pyramid_entry_p) * pyramid_qty
                seg_pnl += gross_pyramid
        else:
            slip_exit = exit_p * (1 + SLIPPAGE)
            gross_initial = (entry_p - slip_exit) * qty_initial
            seg_pnl += gross_initial
            if pyramid_done:
                gross_pyramid = (pyramid_entry_p - slip_exit) * pyramid_qty
                seg_pnl += gross_pyramid

        # Costs
        notional_avg = (entry_p * qty_initial +
                         (pyramid_entry_p * pyramid_qty if pyramid_done else 0))
        seg_costs_exit = slip_exit * cur_qty * TAKER_FEE
        seg_funding = notional_avg * funding_per_bar * bars_held
        total_costs = seg_costs_entry + pyramid_entry_fee + seg_costs_exit + seg_funding
        pnl = seg_pnl - total_costs

        # Bookkeeping
        if cluster_id is not None:
            cluster_pos[cluster_id] = cluster_pos.get(cluster_id, 0) + 1
        open_pos[key] = {
            "sym": sig.sym, "strat": sig.strat, "direction": d,
            "entry_ts": int(ts[j]), "exit_ts": int(ts[exit_bar]),
            "entry": entry_p, "exit": slip_exit, "reason": reason,
            "pnl_realized": pnl, "risk_usd": risk_usd,
            "bars_held": bars_held, "pyramid": pyramid_done,
            "cluster_id": cluster_id,
        }

    # Realize remaining
    for k in list(open_pos.keys()):
        pos = open_pos[k]
        if pos["exit_ts"] is not None:
            balance = max(0.01, balance + pos["pnl_realized"])
            trades.append(_pos_to_trade(pos))

    return _summarize(trades, start_balance)


def _pos_to_trade(pos):
    return {
        "sym": pos["sym"], "strat": pos["strat"], "direction": pos["direction"],
        "entry_ts": pos["entry_ts"], "exit_ts": pos["exit_ts"],
        "entry": pos["entry"], "exit": pos["exit"], "reason": pos["reason"],
        "pnl_usd": pos["pnl_realized"],
        "risk_usd": pos["risk_usd"],
        "pnl_r": pos["pnl_realized"] / pos["risk_usd"] if pos["risk_usd"] > 0 else 0,
        "win": pos["pnl_realized"] > 0,
        "pyramid": pos["pyramid"],
    }


def _summarize(trades, start_balance):
    if not trades:
        return {"trades": [], "metrics": {"n": 0}, "monthly_returns": []}
    trades.sort(key=lambda x: x["exit_ts"])
    eq = start_balance; peak = start_balance; max_dd = 0.0
    eq_curve = [(0, start_balance)]
    for t in trades:
        eq = max(0.01, eq + t["pnl_usd"])
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        eq_curve.append((t["exit_ts"], eq))
    monthly = _monthly_returns(eq_curve, start_balance)
    m = metrics_dict_from_trades(trades, start_balance)
    m["max_dd_pct"] = round(max_dd, 1)
    m["final_balance"] = round(eq, 2)
    m["return_pct"] = round((eq - start_balance) / start_balance * 100, 1)
    return {"trades": trades, "metrics": m, "monthly_returns": monthly}


# ─────────────────────────────────────────────────────────────────────────────
# RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def run_test(name, regime, tag="365d",
              # collect args
              vol_smacross=1.5, vol_supertrend=1.5, vol_rsi=1.5, vol_fib=1.5,
              vol_window=30,
              use_funding=False,
              # simulator args
              base_risk=0.03,
              use_pyramid=False,
              use_correlation_cap=False, sym_cluster=None,
              use_vol_scaled=False):
    syms = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_{tag}.parquet").exists()]
    sigs, sim_dfs = collect_v6_signals(
        syms, regime, tag,
        vol_smacross=vol_smacross, vol_supertrend=vol_supertrend,
        vol_rsi=vol_rsi, vol_fib=vol_fib, vol_window=vol_window,
        use_funding=use_funding,
    )
    if not sigs: return None
    res = simulate_v6(sigs, sim_dfs, regime,
                       start_balance=START_BALANCE,
                       base_risk=base_risk,
                       use_pyramid=use_pyramid,
                       use_correlation_cap=use_correlation_cap,
                       sym_cluster=sym_cluster,
                       use_vol_scaled=use_vol_scaled)
    m = res["metrics"]
    rets = [mr["ret_pct"] for mr in res["monthly_returns"]]
    return {"name": name, "metrics": m, "rets": rets, "trades": res["trades"]}


def print_res(r, label=""):
    if r is None:
        print(f"  {label}: no signals"); return
    m = r["metrics"]
    rets = r["rets"]
    avgmo = float(np.mean(rets)) if rets else 0
    medmo = float(np.median(rets)) if rets else 0
    winmo = sum(1 for x in rets if x > 0) if rets else 0
    totmo = len(rets) if rets else 0
    print(f"  {label or r['name']:30s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
          f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+7.1f}% DD={m['max_dd_pct']:>5.1f}% "
          f"Avg/mo={avgmo:>+5.1f}% Med={medmo:>+5.1f}% Win={winmo}/{totmo}")


def main():
    tag = "365d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    print(f"\nBuilding correlation clusters...")
    sym_cluster = build_correlation_clusters(BASELINE, tag,
                                                threshold=0.85, window_days=60)
    cluster_to_syms = defaultdict(list)
    for s, c in sym_cluster.items():
        cluster_to_syms[c].append(s)
    print(f"Clusters at corr ≥ 0.85:")
    for c, syms in sorted(cluster_to_syms.items()):
        if len(syms) > 1:
            print(f"  cluster {c}: {syms}")

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V6 INCREMENTAL FEATURE TESTING  (BASELINE 27, V5 TPs, vol@1.5×30)")
    print(f"══════════════════════════════════════════════════════════════════")

    all_results = {}
    # 1. V5 baseline (best from prev)
    r = run_test("V5_BASELINE", regime, tag, base_risk=0.03)
    all_results["V5_BASELINE"] = r
    print_res(r, "V5_BASELINE (no new features)")

    # 2. + Pyramid
    r = run_test("V6_PYRAMID", regime, tag, base_risk=0.03, use_pyramid=True)
    all_results["V6_PYRAMID"] = r
    print_res(r, "+ pyramid scaling")

    # 3. + Per-strat vol thresholds (varied)
    for vsm, vst, vrs in [(1.3, 1.7, 1.5), (1.5, 1.7, 1.3), (1.3, 1.5, 1.5)]:
        r = run_test(f"V6_PSV_{vsm}_{vst}_{vrs}", regime, tag, base_risk=0.03,
                       vol_smacross=vsm, vol_supertrend=vst, vol_rsi=vrs)
        all_results[f"V6_PSV_{vsm}_{vst}_{vrs}"] = r
        print_res(r, f"+ per-strat vol  {vsm}/{vst}/{vrs}")

    # 4. + Vol-scaled risk
    r = run_test("V6_VOLSCALED", regime, tag, base_risk=0.03, use_vol_scaled=True)
    all_results["V6_VOLSCALED"] = r
    print_res(r, "+ vol-scaled risk (4-2%)")

    # 5. + Correlation cap
    r = run_test("V6_CORRCAP", regime, tag, base_risk=0.03,
                   use_correlation_cap=True, sym_cluster=sym_cluster)
    all_results["V6_CORRCAP"] = r
    print_res(r, "+ correlation cap (0.85)")

    # 6. + Funding skip
    r = run_test("V6_FUNDING", regime, tag, base_risk=0.03, use_funding=True)
    all_results["V6_FUNDING"] = r
    print_res(r, "+ funding skip (0.05%/8h)")

    # COMBINATIONS — greedy
    # Take winners and combine
    print(f"\n══════ COMBINATIONS (start with V5_BASELINE, add winners) ══════")
    # Combo 1: pyramid + correlation
    r = run_test("V6_PYR+CORR", regime, tag, base_risk=0.03,
                   use_pyramid=True,
                   use_correlation_cap=True, sym_cluster=sym_cluster)
    all_results["V6_PYR+CORR"] = r
    print_res(r, "pyramid + correlation")

    # Combo 2: pyramid + funding
    r = run_test("V6_PYR+FUND", regime, tag, base_risk=0.03,
                   use_pyramid=True, use_funding=True)
    all_results["V6_PYR+FUND"] = r
    print_res(r, "pyramid + funding")

    # Combo 3: pyramid + vol-scaled
    r = run_test("V6_PYR+VS", regime, tag, base_risk=0.03,
                   use_pyramid=True, use_vol_scaled=True)
    all_results["V6_PYR+VS"] = r
    print_res(r, "pyramid + vol-scaled")

    # Combo 4: ALL features
    r = run_test("V6_ALL", regime, tag, base_risk=0.03,
                   use_pyramid=True, use_funding=True,
                   use_correlation_cap=True, sym_cluster=sym_cluster,
                   use_vol_scaled=True,
                   vol_smacross=1.3, vol_supertrend=1.7)
    all_results["V6_ALL"] = r
    print_res(r, "ALL features")

    # Sort and print summary
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V6 RANKED BY RETURN")
    print(f"══════════════════════════════════════════════════════════════════")
    valid = [(k, v) for k, v in all_results.items() if v is not None]
    valid.sort(key=lambda x: -x[1]["metrics"]["return_pct"])
    for k, r in valid:
        print_res(r, k)

    # Save
    out = Path("multi_agent/backtests/results/v6_results.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        [{"name": k, "metrics": v["metrics"], "rets": v["rets"]}
         for k, v in valid],
        indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
