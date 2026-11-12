"""
risk_controls_v7.py — Risk management controls on V7+L4+PivotFilter baseline
=============================================================================

Seven controls tested individually + best combinations:

  RC1  Daily Drawdown Limit   — if -X% intraday → no new entries today
  RC2  Max Concurrent Pos     — hard cap on open positions at once
  RC3  Daily Trade Limit      — max N entries per calendar day
  RC4  Consecutive Loss Pause — after N straight losses → pause K days
  RC5  Correlation Filter     — skip if ≥M correlated positions already open
  RC6  Weekly Drawdown Cap    — if -X% this week → no new entries until Monday
  RC7  Portfolio Heat Cap     — if Σ(open_risk)/equity > X% → skip

Baseline: V7+L4 + pivot_dist>7.0×ATR filtered signals.
"""
from __future__ import annotations
import sys
from collections import defaultdict, deque
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr as _atr,
    START_BALANCE, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H,
    SL_COOLDOWN_MS, MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN,
    _ts_period_ms,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import (
    load_all_events, get_event_context,
)
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter

PIVOT_DIST_THRESH = 7.0   # apply as baseline pre-filter


# ─────────────────────────────────────────────────────────────────────────────
# Correlation matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_corr_matrix(sim_dfs: dict) -> pd.DataFrame:
    """Static log-return correlation matrix over full 730d dataset."""
    closes = {}
    for (sym, _strat), df in sim_dfs.items():
        if sym not in closes:
            closes[sym] = df.set_index("timestamp")["close"].astype(float)

    if len(closes) < 2:
        return pd.DataFrame()

    price_df = pd.DataFrame(closes).sort_index()
    ret_df   = np.log(price_df / price_df.shift(1)).dropna()
    return ret_df.corr()


# ─────────────────────────────────────────────────────────────────────────────
# Extended simulator with all risk controls
# ─────────────────────────────────────────────────────────────────────────────

def simulate_rc(
    sigs, sim_dfs, regime, events,
    # ── base params ───────────────────────────────────────────────────────────
    base_risk       = 0.04,
    max_pos         = 5,
    enable_l4       = True,
    # ── RC1: Daily Drawdown Limit ─────────────────────────────────────────────
    daily_dd_limit  = 0.0,    # fraction; 0=off. e.g. 0.03 = stop if -3% today
    # ── RC3: Daily Trade Limit ────────────────────────────────────────────────
    max_daily_trades= 0,      # 0=off
    # ── RC4: Consecutive Loss Pause ───────────────────────────────────────────
    consec_loss_n   = 0,      # 0=off; N consecutive losses → pause
    consec_loss_days= 1,      # calendar days to pause
    # ── RC5: Correlation Filter ───────────────────────────────────────────────
    max_corr_open   = 0,      # 0=off; max already-open correlated positions
    corr_threshold  = 0.80,
    corr_matrix     = None,   # precomputed DataFrame
    # ── RC6: Weekly Drawdown Cap ─────────────────────────────────────────────
    weekly_dd_limit = 0.0,    # fraction; 0=off
    # ── RC7: Portfolio Heat Cap ───────────────────────────────────────────────
    max_heat        = 0.0,    # fraction; 0=off. e.g. 0.12 = max 12% equity at risk
):
    balance  = float(START_BALANCE)
    open_pos = {}
    sym_cool = {}
    trades   = []
    skipped  = defaultdict(int)
    atr_cache = {}

    # RC state
    _day_date      = None
    _day_start_eq  = balance
    _day_trades    = 0

    _week_key      = None
    _week_start_eq = balance

    _consec_queue    = deque()          # True=win, False=loss
    _consec_pause_ts = 0               # ms timestamp — don't enter before this

    DAY_MS  = 86_400_000
    WEEK_MS = 7 * DAY_MS

    # ── helpers ───────────────────────────────────────────────────────────────
    def _realize_closed(before_ts: int):
        """Close all positions whose exit_ts ≤ before_ts, update balance."""
        nonlocal balance, _consec_pause_ts
        for k in list(open_pos.keys()):
            pos = open_pos[k]
            if pos["exit_ts"] is not None and pos["exit_ts"] <= before_ts:
                balance = max(0.01, balance + pos["pnl"])
                trades.append(pos)
                if pos["reason"] == "SL":
                    sym_cool[pos["sym"]] = pos["exit_ts"] + SL_COOLDOWN_MS
                # RC4 tracking
                if consec_loss_n > 0:
                    _consec_queue.append(pos["win"])
                    if len(_consec_queue) > consec_loss_n:
                        _consec_queue.popleft()
                    if (len(_consec_queue) == consec_loss_n
                            and all(not w for w in _consec_queue)):
                        _consec_pause_ts = pos["exit_ts"] + consec_loss_days * DAY_MS
                        _consec_queue.clear()
                del open_pos[k]

    # ── main loop ─────────────────────────────────────────────────────────────
    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos:
            continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts:
            continue

        _realize_closed(sig.next_open_ts)

        if len(open_pos) >= max_pos:
            continue

        # ── day/week state update ─────────────────────────────────────────────
        ts_dt = pd.Timestamp(sig.next_open_ts, unit="ms", tz="UTC")
        cur_day  = ts_dt.date()
        cur_week = (ts_dt.year, ts_dt.isocalendar()[1])

        if cur_day != _day_date:
            _day_date     = cur_day
            _day_start_eq = balance
            _day_trades   = 0

        if cur_week != _week_key:
            _week_key      = cur_week
            _week_start_eq = balance

        # ── RC1: Daily DD ─────────────────────────────────────────────────────
        if daily_dd_limit > 0 and _day_start_eq > 0:
            day_dd = (_day_start_eq - balance) / _day_start_eq
            if day_dd >= daily_dd_limit:
                skipped["rc1_daily_dd"] += 1
                continue

        # ── RC3: Daily trade limit ────────────────────────────────────────────
        if max_daily_trades > 0 and _day_trades >= max_daily_trades:
            skipped["rc3_daily_n"] += 1
            continue

        # ── RC4: Consecutive loss pause ───────────────────────────────────────
        if consec_loss_n > 0 and sig.next_open_ts < _consec_pause_ts:
            skipped["rc4_consec"] += 1
            continue

        # ── RC6: Weekly DD cap ────────────────────────────────────────────────
        if weekly_dd_limit > 0 and _week_start_eq > 0:
            week_dd = (_week_start_eq - balance) / _week_start_eq
            if week_dd >= weekly_dd_limit:
                skipped["rc6_weekly_dd"] += 1
                continue

        # ── RC7: Portfolio heat cap ───────────────────────────────────────────
        if max_heat > 0 and balance > 0:
            heat = sum(p["risk_usd"] for p in open_pos.values()) / balance
            if heat >= max_heat:
                skipped["rc7_heat"] += 1
                continue

        # ── L4: macro pause ───────────────────────────────────────────────────
        risk_mult = 1.0
        if enable_l4:
            ctx = get_event_context(sig.next_open_ts, events)
            if ctx["context"] == "pre_macro" and ctx["action"] == "PAUSE":
                skipped["l4_pre_macro"] += 1
                continue
            if ctx["context"] == "post_macro":
                risk_mult = ctx.get("risk_mult", 1.0)

        risk = base_risk * max(0.2, min(2.0, risk_mult))

        # ── RC5: Correlation filter ───────────────────────────────────────────
        if (max_corr_open > 0
                and corr_matrix is not None
                and not corr_matrix.empty
                and sig.sym in corr_matrix.index):
            open_syms = [p["sym"] for p in open_pos.values()]
            n_corr = sum(
                1 for os in open_syms
                if os != sig.sym
                and os in corr_matrix.index
                and abs(corr_matrix.at[os, sig.sym]) >= corr_threshold
            )
            if n_corr >= max_corr_open:
                skipped["rc5_corr"] += 1
                continue

        # ── Load price data ───────────────────────────────────────────────────
        df = sim_dfs.get(key)
        if df is None:
            continue
        ts  = df["timestamp"].to_numpy(int)
        op  = df["open"].to_numpy(float)
        hi  = df["high"].to_numpy(float)
        lo  = df["low"].to_numpy(float)
        cl  = df["close"].to_numpy(float)
        n   = len(ts)
        period_ms      = _ts_period_ms(ts)
        funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))

        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1:
            continue

        if key not in atr_cache:
            atr_cache[key] = _atr(hi, lo, cl, 14)
        a_arr = atr_cache[key]

        d       = sig.direction
        entry_p = op[j] * (1 + SLIPPAGE) if d == "LONG" else op[j] * (1 - SLIPPAGE)
        sl0     = sig.sl
        if d == "LONG":
            dist = entry_p - sl0
            if dist <= 0:
                continue
            tp_p = entry_p + (sig.tp - sig.entry) * (dist / sig.dist) if sig.dist > 0 else sig.tp
        else:
            dist = sl0 - entry_p
            if dist <= 0:
                continue
            tp_p = entry_p - (sig.entry - sig.tp) * (dist / sig.dist) if sig.dist > 0 else sig.tp

        risk_usd = balance * risk
        qty      = risk_usd / dist
        notional = entry_p * qty
        max_margin = balance * MAX_MARGIN_FRAC
        if max_margin > 0:
            lev = notional / max_margin
            if lev > MAX_LEVERAGE:
                skipped["over_leverage"] += 1
                continue
            if notional / max(lev, 1) < MIN_MARGIN:
                continue

        # ── Walk forward (SL/TP) ──────────────────────────────────────────────
        exit_p = None; exit_bar = None; reason = "timeout"
        for k in range(j, min(j + 500, n)):
            h_k = hi[k]; l_k = lo[k]
            if d == "LONG":
                if l_k <= sl0: exit_p = sl0;  reason = "SL"; exit_bar = k; break
                if h_k >= tp_p: exit_p = tp_p; reason = "TP"; exit_bar = k; break
            else:
                if h_k >= sl0: exit_p = sl0;  reason = "SL"; exit_bar = k; break
                if l_k <= tp_p: exit_p = tp_p; reason = "TP"; exit_bar = k; break

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
            "entry": entry_p, "sl": sl0, "exit": slip_exit, "reason": reason,
            "pnl": pnl, "risk_usd": risk_usd,
            "pnl_r": pnl / risk_usd if risk_usd > 0 else 0,
            "win": pnl > 0,
        }
        _day_trades += 1

    # Realize remaining open positions
    _realize_closed(999_999_999_999_999)

    # ── Equity curve + metrics ────────────────────────────────────────────────
    trades.sort(key=lambda x: x["exit_ts"])
    eq = float(START_BALANCE); peak = float(START_BALANCE); max_dd = 0.0
    for t in trades:
        eq   = max(0.01, eq + t["pnl"])
        peak = max(peak, eq)
        dd   = (peak - eq) / peak * 100
        max_dd = max(max_dd, dd)

    if not trades:
        return {"trades": [], "metrics": {
            "n": 0, "wr": 0, "pf": 0, "avg_r": 0, "return_pct": 0,
            "max_dd_pct": 0, "final_balance": float(START_BALANCE)
        }, "skipped": dict(skipped)}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = gw / gl if gl > 0 else 99.0
    wr     = len(wins) / len(trades) * 100
    avg_r  = float(np.mean([t["pnl_r"] for t in trades]))
    ret    = (eq - START_BALANCE) / START_BALANCE * 100

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
# Print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sk(res):
    """Short skipped summary string."""
    sk = res.get("skipped", {})
    rc_keys = [k for k in sk if k.startswith("rc")]
    return "  skip=[" + " ".join(f"{k}:{sk[k]}" for k in rc_keys) + "]" if rc_keys else ""


def fmt(name, res, bm=None):
    m = res["metrics"]
    if m["n"] == 0:
        print(f"  {name:52s}  NO TRADES")
        return
    dwr  = f"({m['wr']-bm['wr']:+.1f})"        if bm else ""
    dret = f"({m['return_pct']-bm['return_pct']:+.0f})" if bm else ""
    dpf  = f"({m['pf']-bm['pf']:+.2f})"         if bm else ""
    ddd  = f"({m['max_dd_pct']-bm['max_dd_pct']:+.1f})" if bm else ""
    sk   = _sk(res)
    print(
        f"  {name:52s}  n={m['n']:3d}  "
        f"WR={m['wr']:5.1f}%{dwr:8s}  "
        f"PF={m['pf']:5.2f}{dpf:8s}  "
        f"AvgR={m['avg_r']:+5.2f}  "
        f"Ret={m['return_pct']:+8.1f}%{dret:8s}  "
        f"DD={m['max_dd_pct']:5.1f}%{ddd:8s}"
        f"{sk}"
    )


def hdr():
    print(f"  {'Config':52s}  {'n':3s}  {'WR%':13s}  {'PF':12s}  "
          f"{'AvgR':6s}  {'Ret%':15s}  {'DD%':12s}")


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward helper
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(sigs_base, sigs_test, sim_dfs, regime, events,
                 corr_mx, label, **rc_kwargs):
    """3-month folds walk-forward. sigs_base = baseline (no RC), sigs_test = RC applied."""
    all_ts = sorted(s.next_open_ts for s in sigs_test)
    if not all_ts:
        return
    first = all_ts[0]; last = all_ts[-1]
    train_ms = 6 * 30 * DAY_MS
    test_ms  = 3 * 30 * DAY_MS
    cursor   = first + train_ms

    print(f"\n  Walk-forward: {label}")
    print(f"  {'End':12s}  {'BASE Ret':>10s}  {'BASE WR':>9s}  "
          f"{'RC Ret':>10s}  {'RC WR':>9s}  {'ΔRet':>8s}  {'ΔWR':>7s}  skip")
    wins_b = wins_r = folds = 0
    while cursor + test_ms <= last:
        te = cursor + test_ms
        sb = [s for s in sigs_base if cursor <= s.next_open_ts < te]
        sr = [s for s in sigs_test  if cursor <= s.next_open_ts < te]
        rb = simulate_rc(sb, sim_dfs, regime, events)
        rr = simulate_rc(sr, sim_dfs, regime, events,
                         corr_matrix=corr_mx, **rc_kwargs)
        mb = rb["metrics"]; mr = rr["metrics"]
        d_ret = mr["return_pct"] - mb["return_pct"]
        d_wr  = mr["wr"] - mb["wr"]
        if mb["return_pct"] > 0: wins_b += 1
        if mr["return_pct"] > 0: wins_r += 1
        folds += 1
        sk    = sum(rr.get("skipped", {}).get(k, 0)
                    for k in rr.get("skipped", {}) if k.startswith("rc"))
        te_s  = pd.Timestamp(te, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        print(f"  {te_s:12s}  {mb['return_pct']:>+9.1f}%  {mb['wr']:>7.1f}%  "
              f"{mr['return_pct']:>+9.1f}%  {mr['wr']:>7.1f}%  "
              f"{d_ret:>+7.1f}%  {d_wr:>+6.1f}  {sk}")
        cursor += test_ms
    print(f"  Positive folds: BASE {wins_b}/{folds}  RC {wins_r}/{folds}")


DAY_MS = 86_400_000


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  RISK CONTROLS — V7+L4+PivotFilter  (730d dataset)")
    print("══════════════════════════════════════════════════════════════════════")

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE
              if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n  Symbols: {len(syms)}  |  Loading signals...")
    sigs_raw, sim_dfs, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    events = load_all_events()

    # Apply pivot-distance filter as baseline
    sigs_base, removed, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, PIVOT_DIST_THRESH)
    print(f"  Raw signals: {len(sigs_raw)} → after pivot filter: {len(sigs_base)}"
          f"  (removed {removed})  |  Events: {len(events)}\n")

    # Precompute correlation matrix
    corr_mx = build_corr_matrix(sim_dfs)
    print(f"  Correlation matrix: {corr_mx.shape[0]} symbols\n")

    # ── BASELINE ─────────────────────────────────────────────────────────────
    print("──────────────────────────────────────────────────────────────────────")
    print("  BASELINE (V7+L4 + PivotFilter, no RC)")
    print("──────────────────────────────────────────────────────────────────────")
    res_base = simulate_rc(sigs_base, sim_dfs, regime, events)
    bm = res_base["metrics"]
    fmt("BASELINE", res_base)

    # ── RC1: Daily Drawdown Limit ─────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC1 — Daily Drawdown Limit (stop trading if -X% today)")
    print("──────────────────────────────────────────────────────────────────────")
    hdr()
    best_rc1 = None; best_rc1_score = -999
    for dd in [0.02, 0.03, 0.04, 0.05, 0.07]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          daily_dd_limit=dd)
        fmt(f"daily_dd ≥{dd*100:.0f}%", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc1_score:
            best_rc1_score = score; best_rc1 = dd

    # ── RC2: Max Concurrent Positions ────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC2 — Max Concurrent Positions")
    print("──────────────────────────────────────────────────────────────────────")
    hdr()
    best_rc2 = None; best_rc2_score = -999
    for mp in [3, 4, 5, 6, 7]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          max_pos=mp)
        fmt(f"max_pos={mp}", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc2_score:
            best_rc2_score = score; best_rc2 = mp

    # ── RC3: Daily Trade Limit ────────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC3 — Daily Trade Limit (max N new entries per day)")
    print("──────────────────────────────────────────────────────────────────────")
    hdr()
    best_rc3 = None; best_rc3_score = -999
    for dt in [2, 3, 4, 5, 6]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          max_daily_trades=dt)
        fmt(f"max_daily={dt}", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc3_score:
            best_rc3_score = score; best_rc3 = dt

    # ── RC4: Consecutive Loss Pause ───────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC4 — Consecutive Loss Pause")
    print("──────────────────────────────────────────────────────────────────────")
    hdr()
    best_rc4 = None; best_rc4_score = -999
    for nl, days in [(2, 1), (2, 2), (3, 1), (3, 2), (4, 1)]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          consec_loss_n=nl, consec_loss_days=days)
        fmt(f"consec_loss={nl}→pause {days}d", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc4_score:
            best_rc4_score = score; best_rc4 = (nl, days)

    # ── RC5: Correlation Filter ───────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC5 — Correlation Filter (skip if ≥N correlated positions open)")
    print("──────────────────────────────────────────────────────────────────────")
    print(f"  (corr matrix sample: BTC↔ETH={corr_mx.at['BTC','ETH']:.2f}"
          f"  BTC↔SOL={corr_mx.at['BTC','SOL']:.2f}"
          f"  BTC↔DOGE={corr_mx.at['BTC','DOGE']:.2f})" if all(
              s in corr_mx.index for s in ['BTC','ETH','SOL','DOGE']) else "")
    hdr()
    best_rc5 = None; best_rc5_score = -999
    for max_co, thr in [(1, 0.80), (2, 0.80), (3, 0.80), (2, 0.85), (1, 0.85)]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          max_corr_open=max_co, corr_threshold=thr,
                          corr_matrix=corr_mx)
        fmt(f"corr: max={max_co} thr={thr:.2f}", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc5_score:
            best_rc5_score = score; best_rc5 = (max_co, thr)

    # ── RC6: Weekly Drawdown Cap ──────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC6 — Weekly Drawdown Cap")
    print("──────────────────────────────────────────────────────────────────────")
    hdr()
    best_rc6 = None; best_rc6_score = -999
    for wd in [0.05, 0.07, 0.10, 0.12, 0.15]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          weekly_dd_limit=wd)
        fmt(f"weekly_dd ≥{wd*100:.0f}%", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc6_score:
            best_rc6_score = score; best_rc6 = wd

    # ── RC7: Portfolio Heat Cap ───────────────────────────────────────────────
    print("\n──────────────────────────────────────────────────────────────────────")
    print("  RC7 — Portfolio Heat Cap (max Σrisk / equity)")
    print("──────────────────────────────────────────────────────────────────────")
    hdr()
    best_rc7 = None; best_rc7_score = -999
    for heat in [0.08, 0.12, 0.16, 0.20, 0.24]:
        res = simulate_rc(sigs_base, sim_dfs, regime, events,
                          max_heat=heat)
        fmt(f"max_heat={heat*100:.0f}%", res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_rc7_score:
            best_rc7_score = score; best_rc7 = heat

    # ── COMBINATIONS ─────────────────────────────────────────────────────────
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  COMBINATIONS — best params from each RC")
    print("══════════════════════════════════════════════════════════════════════")
    hdr()

    combos = []
    if best_rc1: combos.append(("best_RC1 only",   {"daily_dd_limit": best_rc1}))
    if best_rc2: combos.append(("best_RC2 only",   {"max_pos": best_rc2}))
    if best_rc3: combos.append(("best_RC3 only",   {"max_daily_trades": best_rc3}))
    if best_rc4: combos.append(("best_RC4 only",   {"consec_loss_n": best_rc4[0], "consec_loss_days": best_rc4[1]}))
    if best_rc5: combos.append(("best_RC5 only",   {"max_corr_open": best_rc5[0], "corr_threshold": best_rc5[1], "corr_matrix": corr_mx}))
    if best_rc6: combos.append(("best_RC6 only",   {"weekly_dd_limit": best_rc6}))
    if best_rc7: combos.append(("best_RC7 only",   {"max_heat": best_rc7}))

    # Build combined config from best individual controls that improve over baseline
    combined = {}
    if best_rc1: combined["daily_dd_limit"] = best_rc1
    if best_rc4: combined["consec_loss_n"] = best_rc4[0]; combined["consec_loss_days"] = best_rc4[1]
    if best_rc5: combined["max_corr_open"] = best_rc5[0]; combined["corr_threshold"] = best_rc5[1]; combined["corr_matrix"] = corr_mx
    if best_rc7: combined["max_heat"] = best_rc7

    if combined:
        combos.append(("RC1+RC4+RC5+RC7 (auto-combined)", combined))

    # Explicit handcrafted combos
    combos += [
        ("RC1(3%)+RC4(3→1d)+RC7(16%)",
         {"daily_dd_limit": 0.03, "consec_loss_n": 3, "consec_loss_days": 1, "max_heat": 0.16}),
        ("RC1(3%)+RC2(4)+RC6(8%)",
         {"daily_dd_limit": 0.03, "max_pos": 4, "weekly_dd_limit": 0.08}),
        ("RC2(4)+RC5(2@0.80)+RC7(16%)",
         {"max_pos": 4, "max_corr_open": 2, "corr_threshold": 0.80,
          "corr_matrix": corr_mx, "max_heat": 0.16}),
        ("RC1(3%)+RC4(3→1d)+RC5(2@0.80)+RC7(16%)+RC2(4)",
         {"daily_dd_limit": 0.03, "consec_loss_n": 3, "consec_loss_days": 1,
          "max_corr_open": 2, "corr_threshold": 0.80, "corr_matrix": corr_mx,
          "max_heat": 0.16, "max_pos": 4}),
    ]

    best_combo_name = None; best_combo_score = -999; best_combo_kwargs = {}
    for name, kwargs in combos:
        res = simulate_rc(sigs_base, sim_dfs, regime, events, **kwargs)
        fmt(name, res, bm)
        m = res["metrics"]
        score = m["return_pct"] - 2*m["max_dd_pct"]
        if m["n"] >= 80 and score > best_combo_score:
            best_combo_score = score
            best_combo_name  = name
            best_combo_kwargs = kwargs

    # ── Walk-forward on best combo ────────────────────────────────────────────
    if best_combo_kwargs:
        print(f"\n══════════════════════════════════════════════════════════════════════")
        print(f"  WALK-FORWARD  →  best combo: {best_combo_name}")
        print(f"══════════════════════════════════════════════════════════════════════")
        walk_forward(sigs_base, sigs_base, sim_dfs, regime, events,
                     corr_mx, best_combo_name, **best_combo_kwargs)

    # ── VERDICT ───────────────────────────────────────────────────────────────
    print(f"\n══════════════════════════════════════════════════════════════════════")
    print(f"  VERDICT")
    print(f"══════════════════════════════════════════════════════════════════════")

    print(f"\n  Baseline (V7+L4+PivotFilter):")
    print(f"    n={bm['n']}  Ret={bm['return_pct']:+.1f}%  WR={bm['wr']:.1f}%  "
          f"PF={bm['pf']:.2f}  DD={bm['max_dd_pct']:.1f}%")

    if best_combo_kwargs:
        res_best = simulate_rc(sigs_base, sim_dfs, regime, events, **best_combo_kwargs)
        mb2 = res_best["metrics"]
        print(f"\n  Best combo: {best_combo_name}")
        print(f"    n={mb2['n']}  Ret={mb2['return_pct']:+.1f}%  WR={mb2['wr']:.1f}%  "
              f"PF={mb2['pf']:.2f}  DD={mb2['max_dd_pct']:.1f}%")
        print(f"\n  Delta:")
        print(f"    ΔRet  = {mb2['return_pct'] - bm['return_pct']:+.1f}%")
        print(f"    ΔWR   = {mb2['wr'] - bm['wr']:+.1f}pp")
        print(f"    ΔPF   = {mb2['pf'] - bm['pf']:+.2f}")
        print(f"    ΔDD   = {mb2['max_dd_pct'] - bm['max_dd_pct']:+.1f}pp")
        sk = res_best.get("skipped", {})
        print(f"\n  Trades blocked by risk controls: "
              + "  ".join(f"{k}={v}" for k, v in sorted(sk.items()) if k.startswith("rc")))
    print()


if __name__ == "__main__":
    main()
