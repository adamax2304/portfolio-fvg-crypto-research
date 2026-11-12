"""
filter_tests_v7.py — Big-candle filter + Range-exit filter on V7+L4
=====================================================================
Two filters from visual analysis of April 2026 trades:

  F1 — Big-candle filter:
       Skip signal if trigger candle body > N×ATR(14).
       Rationale: entering on spike candles = already overextended,
       price hits resistance and reverses to SL.

  F2 — Range-exit filter:
       After M bars in position, if price range < K×ATR AND progress
       toward TP < threshold → close at market (avoid slow bleed to SL).
       Rationale: sideways chop after entry = low continuation probability.

Baseline: V7+L4 @ 4% risk, bear=long_only, FOMC pause
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
    load, build_regime, atr as _atr,
    PortfolioSignal, _ts_period_ms,
    START_BALANCE, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H,
    SL_COOLDOWN_MS, MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN,
    _monthly_returns, metrics_dict_from_trades,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import (
    load_all_events, build_sentiment_series, get_event_context,
    MACRO_CATEGORIES,
)


# ─────────────────────────────────────────────────────────────────────────────
# F1: Big-candle filter — applied to signal list before simulation
# ─────────────────────────────────────────────────────────────────────────────

def apply_bigcandle_filter(sigs, sim_dfs, max_body_atr: float):
    """
    Remove signals where the trigger candle body > max_body_atr × ATR(14).
    Trigger candle = the bar whose close created the signal (sig.ts).
    Also checks candle body% of full range (>92% = near-pure spike bar).
    """
    if max_body_atr <= 0:
        return sigs, 0   # disabled

    kept = []
    removed = 0
    _atr_cache = {}   # (sym, strat) → ATR array, computed once

    for sig in sigs:
        key = (sig.sym, sig.strat)
        df  = sim_dfs.get(key)
        if df is None:
            kept.append(sig); continue

        # Find trigger bar
        ts_arr = df["timestamp"].to_numpy(int)
        idx    = int(np.searchsorted(ts_arr, sig.ts, side="left"))
        if idx >= len(ts_arr) or ts_arr[idx] != sig.ts:
            # Timestamp not found exactly — tolerate and keep
            kept.append(sig); continue

        # Compute ATR (cached per df)
        if key not in _atr_cache:
            _atr_cache[key] = _atr(
                df["high"].to_numpy(float),
                df["low"].to_numpy(float),
                df["close"].to_numpy(float), 14
            )
        a_arr = _atr_cache[key]
        a_i   = float(a_arr[idx]) if idx < len(a_arr) else np.nan
        if np.isnan(a_i) or a_i <= 0:
            kept.append(sig); continue

        o_i   = float(df.iloc[idx]["open"])
        h_i   = float(df.iloc[idx]["high"])
        l_i   = float(df.iloc[idx]["low"])
        c_i   = float(df.iloc[idx]["close"])

        body  = abs(c_i - o_i)
        rng   = h_i - l_i
        body_pct = (body / rng) if rng > 0 else 0

        # Filter: body > max_body_atr × ATR
        is_spike = body > max_body_atr * a_i
        # Extra: body covers >90% of candle range AND body > 1.2×ATR
        is_pure_spike = (body_pct > 0.90) and (body > 1.2 * a_i)

        if is_spike or is_pure_spike:
            removed += 1
        else:
            kept.append(sig)

    return kept, removed


# ─────────────────────────────────────────────────────────────────────────────
# Simulator with both filters wired in
# ─────────────────────────────────────────────────────────────────────────────

def simulate_filtered(
    sigs, sim_dfs, regime, events,
    base_risk     = 0.04,
    max_pos       = 5,
    # L4 macro pause
    enable_l4     = True,
    # F2: Range-exit parameters
    range_bars    = 0,     # 0 = disabled; else: check after this many bars
    range_atr_mult = 2.0,  # range < this × ATR → "stuck"
    range_min_progress = 0.20,  # only exit if <20% progress toward TP
    range_close_mode = "market",  # "market" | "breakeven" (close at entry if profitable)
):
    """
    Extended simulator: V7+L4 + optional F2 range-exit.
    Big-candle filter (F1) is applied BEFORE calling this function.
    """
    balance  = START_BALANCE
    open_pos = {}; sym_cool = {}
    trades   = []
    skipped  = defaultdict(int)

    # Pre-compute ATR arrays for F2
    atr_cache = {}

    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos: continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts: continue

        # Realize closed positions
        for k in list(open_pos.keys()):
            pos = open_pos[k]
            if pos["exit_ts"] is not None and pos["exit_ts"] <= sig.next_open_ts:
                balance = max(0.01, balance + pos["pnl"])
                trades.append(pos)
                if pos["reason"] == "SL":
                    sym_cool[pos["sym"]] = pos["exit_ts"] + SL_COOLDOWN_MS
                del open_pos[k]

        if len(open_pos) >= max_pos: continue

        # L4: macro pause
        if enable_l4:
            ctx = get_event_context(sig.next_open_ts, events)
            if ctx["context"] == "pre_macro" and ctx["action"] == "PAUSE":
                skipped["pre_macro"] += 1; continue
            risk_mult = ctx.get("risk_mult", 1.0) if ctx["context"] == "post_macro" else 1.0
        else:
            risk_mult = 1.0

        risk = base_risk * max(0.2, min(2.0, risk_mult))

        # Load df
        df = sim_dfs.get(key)
        if df is None: continue
        ts  = df["timestamp"].to_numpy(int)
        op  = df["open"].to_numpy(float)
        hi  = df["high"].to_numpy(float)
        lo  = df["low"].to_numpy(float)
        cl  = df["close"].to_numpy(float)
        n   = len(ts)
        period_ms      = _ts_period_ms(ts)
        funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))

        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1: continue

        # Cached ATR for F2
        if key not in atr_cache:
            atr_cache[key] = _atr(hi, lo, cl, 14)
        a_arr = atr_cache[key]

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
        qty      = risk_usd / dist
        notional = entry_p * qty
        max_margin = balance * MAX_MARGIN_FRAC
        if max_margin > 0:
            lev = notional / max_margin
            if lev > MAX_LEVERAGE:
                skipped["over_leverage"] += 1; continue
            if notional / max(lev, 1) < MIN_MARGIN: continue

        # Walk forward
        exit_p = None; exit_bar = None; reason = "timeout"
        a_entry = float(a_arr[j]) if j < len(a_arr) and not np.isnan(a_arr[j]) else dist

        for k in range(j, min(j + 500, n)):
            h_k = hi[k]; l_k = lo[k]

            # SL / TP check
            if d == "LONG":
                if l_k <= sl0: exit_p = sl0;  reason = "SL"; exit_bar = k; break
                if h_k >= tp_p: exit_p = tp_p; reason = "TP"; exit_bar = k; break
            else:
                if h_k >= sl0: exit_p = sl0;  reason = "SL"; exit_bar = k; break
                if l_k <= tp_p: exit_p = tp_p; reason = "TP"; exit_bar = k; break

            # F2: Range-exit check (every bar after range_bars)
            bars_in = k - j
            if range_bars > 0 and bars_in >= range_bars:
                window_hi  = hi[j:k+1].max()
                window_lo  = lo[j:k+1].min()
                price_rng  = window_hi - window_lo
                is_stuck   = price_rng < range_atr_mult * a_entry

                if is_stuck:
                    # Progress toward TP (0..1)
                    if d == "LONG":
                        progress = (cl[k] - entry_p) / (tp_p - entry_p) if tp_p != entry_p else 0
                    else:
                        progress = (entry_p - cl[k]) / (entry_p - tp_p) if entry_p != tp_p else 0

                    if progress < range_min_progress:
                        if range_close_mode == "breakeven" and progress > 0:
                            # Close at entry (breakeven)
                            exit_p = entry_p
                        else:
                            # Close at current market
                            exit_p = cl[k]
                        reason   = "range_exit"
                        exit_bar = k
                        break

        if exit_bar is None:
            exit_p   = cl[min(j + 499, n - 1)]
            exit_bar = min(j + 499, n - 1)

        bars_held = exit_bar - j + 1
        if d == "LONG":
            slip_exit = exit_p * (1 - SLIPPAGE)
            gross     = (slip_exit - entry_p) * qty
        else:
            slip_exit = exit_p * (1 + SLIPPAGE)
            gross     = (entry_p - slip_exit) * qty

        entry_fee = entry_p   * qty * TAKER_FEE
        exit_fee  = slip_exit * qty * TAKER_FEE
        funding   = entry_p   * qty * funding_per_bar * bars_held
        pnl       = gross - entry_fee - exit_fee - funding

        open_pos[key] = {
            "sym": sig.sym, "strat": sig.strat, "direction": d,
            "entry_ts": int(ts[j]), "exit_ts": int(ts[exit_bar]),
            "entry": entry_p, "exit": slip_exit, "reason": reason,
            "pnl": pnl, "risk_usd": risk_usd,
            "pnl_r": pnl / risk_usd if risk_usd > 0 else 0,
            "win": pnl > 0,
        }

    # Realize remaining
    for pos in open_pos.values():
        if pos["exit_ts"] is not None:
            balance = max(0.01, balance + pos["pnl"])
            trades.append(pos)

    trades.sort(key=lambda x: x["exit_ts"])
    eq   = START_BALANCE; peak = START_BALANCE; max_dd = 0
    eq_curve = [(0, START_BALANCE)]
    for t in trades:
        eq   = max(0.01, eq + t["pnl"])
        peak = max(peak, eq)
        dd   = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        eq_curve.append((t["exit_ts"], eq))

    if not trades:
        return {"trades": [], "metrics": {"n": 0, "wr": 0, "pf": 0,
                "avg_r": 0, "return_pct": 0, "max_dd_pct": 0}}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = gw / gl if gl > 0 else 99.0
    wr     = len(wins) / len(trades) * 100
    avg_r  = float(np.mean([t["pnl_r"] for t in trades]))
    ret    = (eq - START_BALANCE) / START_BALANCE * 100

    # Reason breakdown
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["reason"]] += 1

    return {
        "trades": trades,
        "skipped": dict(skipped),
        "reasons": dict(reasons),
        "metrics": {
            "n": len(trades), "wr": round(wr, 1), "pf": round(pf, 2),
            "avg_r": round(avg_r, 3), "return_pct": round(ret, 1),
            "max_dd_pct": round(max_dd, 1), "final_balance": round(eq, 2),
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Printing helpers
# ─────────────────────────────────────────────────────────────────────────────

def fmt(name, res, baseline_m=None):
    m = res["metrics"]
    if m["n"] == 0:
        print(f"  {name:42s} NO TRADES"); return

    d_wr  = f" ({m['wr'] - baseline_m['wr']:+.1f})" if baseline_m else ""
    d_ret = f" ({m['return_pct'] - baseline_m['return_pct']:+.1f})" if baseline_m else ""
    d_pf  = f" ({m['pf'] - baseline_m['pf']:+.2f})" if baseline_m else ""

    rea = res.get("reasons", {})
    rea_str = "  ".join(f"{k}={v}" for k, v in sorted(rea.items()))
    print(
        f"  {name:42s}  n={m['n']:3d}  WR={m['wr']:5.1f}%{d_wr:9s}  "
        f"PF={m['pf']:5.2f}{d_pf:8s}  AvgR={m['avg_r']:+5.2f}  "
        f"Ret={m['return_pct']:+7.1f}%{d_ret:8s}  DD={m['max_dd_pct']:5.1f}%"
        f"   [{rea_str}]"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  FILTER TESTS V7+L4 — Big-candle filter + Range-exit filter")
    print("══════════════════════════════════════════════════════════════════════")

    # Load data
    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n  Symbols: {len(syms)}  |  Loading signals...")

    sigs_raw, sim_dfs, info = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    print(f"  Signals raw: {len(sigs_raw)}  (bear filtered: {info['bear_filtered']})")

    events    = load_all_events()
    print(f"  Events: {len(events)}")

    # ── BASELINE ─────────────────────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  BASELINE (V7+L4, no extra filters)")
    print("──────────────────────────────────────────────────────────────────────")
    base_res = simulate_filtered(sigs_raw, sim_dfs, regime, events,
                                  base_risk=0.04, enable_l4=True,
                                  range_bars=0)
    base_m   = base_res["metrics"]
    fmt("BASELINE", base_res)

    # ── F1: Big-candle filter grid ────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  F1: Big-candle filter  (body > N×ATR → skip signal)")
    print("  Format: Δ vs baseline in parentheses")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  {'Config':42s}  {'n':3s}  {'WR%':14s}  {'PF':13s}  {'AvgR':6s}  "
          f"{'Ret%':15s}  {'DD%':5s}   [TP / SL / range_exit]")

    best_f1 = None; best_f1_ret = base_m["return_pct"]

    for mult in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]:
        sigs_f, removed = apply_bigcandle_filter(sigs_raw, sim_dfs, max_body_atr=mult)
        res = simulate_filtered(sigs_f, sim_dfs, regime, events,
                                 base_risk=0.04, enable_l4=True, range_bars=0)
        label = f"body<{mult:.1f}×ATR  (removed {removed}/{len(sigs_raw)})"
        fmt(label, res, base_m)
        if res["metrics"]["return_pct"] > best_f1_ret and res["metrics"]["n"] >= 80:
            best_f1_ret = res["metrics"]["return_pct"]
            best_f1 = mult

    if best_f1 is None:
        best_f1 = 0.0   # baseline wins on F1
        print(f"\n  → F1 verdict: No big-candle threshold improves on baseline.")
    else:
        print(f"\n  → F1 best: body<{best_f1:.1f}×ATR")

    # ── F2: Range-exit filter grid ────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  F2: Range-exit filter  (stuck M bars in range < K×ATR → close)")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  {'Config':42s}  {'n':3s}  {'WR%':14s}  {'PF':13s}  {'AvgR':6s}  "
          f"{'Ret%':15s}  {'DD%':5s}   [TP / SL / range_exit]")

    best_f2 = None; best_f2_ret = base_m["return_pct"]

    for range_bars in [4, 6, 8, 10, 12]:
        for range_atr in [1.0, 1.5, 2.0, 2.5]:
            for min_prog in [0.10, 0.20, 0.30]:
                res = simulate_filtered(sigs_raw, sim_dfs, regime, events,
                                         base_risk=0.04, enable_l4=True,
                                         range_bars=range_bars,
                                         range_atr_mult=range_atr,
                                         range_min_progress=min_prog,
                                         range_close_mode="market")
                label = f"bars≥{range_bars}  rng<{range_atr:.1f}×ATR  prog<{int(min_prog*100)}%"
                fmt(label, res, base_m)
                if res["metrics"]["return_pct"] > best_f2_ret and res["metrics"]["n"] >= 80:
                    best_f2_ret = res["metrics"]["return_pct"]
                    best_f2 = (range_bars, range_atr, min_prog)

    if best_f2 is None:
        print(f"\n  → F2 verdict: No range-exit configuration improves on baseline.")
    else:
        print(f"\n  → F2 best: bars≥{best_f2[0]} range<{best_f2[1]:.1f}×ATR progress<{int(best_f2[2]*100)}%")

    # ── F2 breakeven mode (alternative: close at entry instead of market) ─────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  F2b: Range-exit = breakeven close (close at entry price if >0)")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  {'Config':42s}  {'n':3s}  {'WR%':14s}  {'PF':13s}  {'AvgR':6s}  "
          f"{'Ret%':15s}  {'DD%':5s}   [TP / SL / range_exit]")

    best_f2b = None; best_f2b_ret = base_m["return_pct"]
    for range_bars in [4, 6, 8]:
        for range_atr in [1.5, 2.0]:
            res = simulate_filtered(sigs_raw, sim_dfs, regime, events,
                                     base_risk=0.04, enable_l4=True,
                                     range_bars=range_bars,
                                     range_atr_mult=range_atr,
                                     range_min_progress=0.15,
                                     range_close_mode="breakeven")
            label = f"breakeven bars≥{range_bars} rng<{range_atr:.1f}×ATR"
            fmt(label, res, base_m)
            if res["metrics"]["return_pct"] > best_f2b_ret and res["metrics"]["n"] >= 80:
                best_f2b_ret = res["metrics"]["return_pct"]
                best_f2b = (range_bars, range_atr)

    # ── Combined: best F1 + best F2 ──────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  COMBINED: Best F1 + Best F2")
    print("──────────────────────────────────────────────────────────────────────")

    # Try a few promising combos
    combos = []
    if best_f1 > 0:
        combos.append(("F1 only",      best_f1, 0,          0,          0))
    if best_f2:
        combos.append(("F2 only",      0.0,     best_f2[0], best_f2[1], best_f2[2]))
    if best_f1 > 0 and best_f2:
        combos.append(("F1+F2",        best_f1, best_f2[0], best_f2[1], best_f2[2]))

    # Also test some hand-picked combinations based on what we saw
    extra_combos = [
        ("F1(1.5)+F2(6b,2.0atr,20%)", 1.5, 6, 2.0, 0.20),
        ("F1(2.0)+F2(8b,1.5atr,20%)", 2.0, 8, 1.5, 0.20),
        ("F1(1.5)+F2(8b,2.0atr,20%)", 1.5, 8, 2.0, 0.20),
        ("F1(2.0)+F2(6b,2.0atr,15%)", 2.0, 6, 2.0, 0.15),
        ("F1(1.2)+F2(8b,2.0atr,20%)", 1.2, 8, 2.0, 0.20),
    ]
    for ec in extra_combos:
        combos.append(ec)

    print(f"  {'Config':42s}  {'n':3s}  {'WR%':14s}  {'PF':13s}  {'AvgR':6s}  "
          f"{'Ret%':15s}  {'DD%':5s}   [TP / SL / range_exit]")
    fmt("BASELINE (reference)", base_res)

    best_combo_ret = base_m["return_pct"]
    best_combo_cfg = None

    for combo in combos:
        name, f1_mult, rb, ra, rp = combo[0], combo[1], combo[2], combo[3], combo[4]
        if f1_mult > 0:
            sigs_f, removed = apply_bigcandle_filter(sigs_raw, sim_dfs, max_body_atr=f1_mult)
        else:
            sigs_f = sigs_raw; removed = 0
        res = simulate_filtered(sigs_f, sim_dfs, regime, events,
                                 base_risk=0.04, enable_l4=True,
                                 range_bars=rb, range_atr_mult=ra,
                                 range_min_progress=rp, range_close_mode="market")
        removed_str = f" rm={removed}" if f1_mult > 0 else ""
        fmt(f"{name}{removed_str}", res, base_m)
        if res["metrics"]["return_pct"] > best_combo_ret and res["metrics"]["n"] >= 80:
            best_combo_ret = res["metrics"]["return_pct"]
            best_combo_cfg = (name, f1_mult, rb, ra, rp)

    # ── FINAL VERDICT ─────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  VERDICT")
    print("══════════════════════════════════════════════════════════════════════")

    if best_combo_cfg:
        name, f1_mult, rb, ra, rp = best_combo_cfg
        if f1_mult > 0:
            sigs_f, removed = apply_bigcandle_filter(sigs_raw, sim_dfs, max_body_atr=f1_mult)
        else:
            sigs_f = sigs_raw; removed = 0
        best_res = simulate_filtered(sigs_f, sim_dfs, regime, events,
                                      base_risk=0.04, enable_l4=True,
                                      range_bars=rb, range_atr_mult=ra,
                                      range_min_progress=rp, range_close_mode="market")
        bm = best_res["metrics"]
        print(f"\n  Best config: {name}")
        if f1_mult > 0:
            print(f"    F1: body<{f1_mult:.1f}×ATR  (removed {removed} signals)")
        if rb > 0:
            print(f"    F2: close if stuck ≥{rb} bars  range<{ra:.1f}×ATR  progress<{int(rp*100)}%")
        print(f"\n  Baseline  →  {base_m['return_pct']:+.1f}%  WR={base_m['wr']:.1f}%  "
              f"PF={base_m['pf']:.2f}  DD={base_m['max_dd_pct']:.1f}%")
        print(f"  Best      →  {bm['return_pct']:+.1f}%  WR={bm['wr']:.1f}%  "
              f"PF={bm['pf']:.2f}  DD={bm['max_dd_pct']:.1f}%")
        print(f"\n  ΔReturn: {bm['return_pct'] - base_m['return_pct']:+.1f}%  "
              f"ΔWR: {bm['wr'] - base_m['wr']:+.1f}pp  "
              f"ΔPF: {bm['pf'] - base_m['pf']:+.2f}  "
              f"ΔDD: {bm['max_dd_pct'] - base_m['max_dd_pct']:+.1f}pp")

        # Reason breakdown
        rea = best_res.get("reasons", {})
        total = sum(rea.values())
        print(f"\n  Exit reasons: " + "  ".join(
            f"{k}={v} ({v/max(1,total)*100:.0f}%)" for k, v in sorted(rea.items())
        ))
    else:
        print(f"\n  → Neither filter improved on baseline.")
        print(f"  Baseline remains optimal: Ret={base_m['return_pct']:+.1f}%  "
              f"WR={base_m['wr']:.1f}%  PF={base_m['pf']:.2f}")
        print(f"  Interpretation: the strategy's 1:2.5R ratio already prices in false breakouts.")
        print(f"  Filtering spike candles removes too many valid momentum entries.")

    print()


if __name__ == "__main__":
    main()
