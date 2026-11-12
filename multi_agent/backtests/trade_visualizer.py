"""
trade_visualizer.py — V7+L4 example trades on candlestick charts
=================================================================
Shows April 2026 trades (+ March trades that extend into April)
with entry/exit/SL/TP levels on OHLC candles.

Usage:
  py -3 multi_agent/backtests/trade_visualizer.py
  py -3 multi_agent/backtests/trade_visualizer.py --month 2026-03  (March)
  py -3 multi_agent/backtests/trade_visualizer.py --all-recent     (Feb-Apr)
"""
from __future__ import annotations
import sys
import argparse
from pathlib import Path
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, sma, _ts_period_ms, SLIPPAGE,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import (
    load_all_events, build_sentiment_series, simulate_news_aware
)

# ─── Colour palette (dark theme) ──────────────────────────────────────────────
BG      = "#0d1117"
BG_AX   = "#161b22"
GREEN   = "#3fb950"
RED     = "#f85149"
CYAN    = "#58a6ff"
ORANGE  = "#ff8c00"
YELLOW  = "#e3b341"
GREY    = "#8b949e"
WHITE   = "#e6edf3"

OUT_DIR = Path(__file__).parent / "charts_april2026"


def ts_fmt(ts_ms: int, fmt="%Y-%m-%d %H:%M") -> str:
    return pd.Timestamp(ts_ms, unit="ms", tz="UTC").strftime(fmt)


def ts_date(ts_ms: int) -> str:
    return ts_fmt(ts_ms, "%m/%d")


# ─── Candlestick drawing ───────────────────────────────────────────────────────

def draw_candles(ax, df, idx_start, idx_end):
    """Draw OHLC candles with volume sub-bar in same axes (offset)."""
    sub = df.iloc[idx_start:idx_end].reset_index(drop=True)
    n   = len(sub)
    if n == 0:
        return np.array([]), np.array([])

    op  = sub["open"].to_numpy(float)
    hi  = sub["high"].to_numpy(float)
    lo  = sub["low"].to_numpy(float)
    cl  = sub["close"].to_numpy(float)
    ts  = sub["timestamp"].to_numpy(int)
    xs  = np.arange(n, dtype=float)

    price_range = hi.max() - lo.min()
    if price_range == 0:
        price_range = 1.0

    for i in range(n):
        color = GREEN if cl[i] >= op[i] else RED
        # wick
        ax.plot([xs[i], xs[i]], [lo[i], hi[i]], color=color, lw=0.8, zorder=2)
        # body
        body_lo = min(op[i], cl[i])
        body_hi = max(op[i], cl[i])
        body_h  = max(body_hi - body_lo, price_range * 0.001)
        ax.bar(xs[i], body_h, bottom=body_lo, width=0.6, color=color,
               linewidth=0, zorder=3)

    # X-axis date labels
    step = max(1, n // 12)
    ax.set_xticks(xs[::step])
    ax.set_xticklabels([ts_date(ts[i]) for i in range(0, n, step)],
                       fontsize=7, color=GREY)
    ax.set_xlim(-1, n)
    ax.tick_params(axis="y", labelsize=7, labelcolor=GREY)
    return xs, ts


# ─── Main per-trade chart ──────────────────────────────────────────────────────

def plot_trade_chart(trade: dict, sig, df: pd.DataFrame, regime,
                     events: list, out_dir: Path) -> Path:
    """
    Draw a single trade chart.
    trade  — dict from simulate_news_aware trades list
    sig    — matching PortfolioSignal (for sl/tp)
    df     — OHLC DataFrame for the relevant timeframe
    """
    ts_arr = df["timestamp"].to_numpy(int)
    n_bars = len(ts_arr)

    # Locate entry/exit bar
    j_entry = int(np.searchsorted(ts_arr, trade["entry_ts"], side="left"))
    j_exit  = int(np.searchsorted(ts_arr, trade["exit_ts"],  side="left"))
    j_entry = min(j_entry, n_bars - 1)
    j_exit  = min(j_exit,  n_bars - 1)

    # Context window: 25 bars before entry, 8 after exit (minimum 40 bars)
    pad_pre  = 25
    pad_post = 8
    idx_s = max(0, j_entry - pad_pre)
    idx_e = min(n_bars, j_exit + pad_post + 1)
    # Ensure minimum window
    if idx_e - idx_s < 40:
        idx_e = min(n_bars, idx_s + 40)

    # ── Recompute exact SL and TP (as simulator does) ──────────────────────
    d       = trade["direction"]
    entry_p = trade["entry"]          # already includes slippage
    sl0     = sig.sl
    if sig.dist > 0:
        if d == "LONG":
            dist  = entry_p - sl0
            tp_p  = entry_p + (sig.tp - sig.entry) * (dist / sig.dist)
        else:
            dist  = sl0 - entry_p
            tp_p  = entry_p - (sig.entry - sig.tp) * (dist / sig.dist)
    else:
        tp_p = sig.tp

    # ── SMA50 for context ──────────────────────────────────────────────────
    cl_full = df["close"].to_numpy(float)
    sma50   = sma(cl_full, 50)

    # ── Figure ────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 7), facecolor=BG)
    ax  = fig.add_subplot(111, facecolor=BG_AX)
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.tick_params(colors=GREY)

    xs, ts_sub = draw_candles(ax, df, idx_s, idx_e)
    if len(xs) == 0:
        plt.close(); return None

    # Map absolute bar indices → relative xs
    j_e_rel = j_entry - idx_s
    j_x_rel = j_exit  - idx_s
    j_e_rel = max(0, min(j_e_rel, len(xs) - 1))
    j_x_rel = max(0, min(j_x_rel, len(xs) - 1))

    # ── SMA50 line ──────────────────────────────────────────────────────────
    sma_sub = sma50[idx_s:idx_e]
    valid   = ~np.isnan(sma_sub)
    if valid.any():
        ax.plot(xs[valid], sma_sub[valid], color="#7d8590", lw=0.8, ls="-",
                alpha=0.6, label="SMA50", zorder=4)

    # ── Hold-zone shading ───────────────────────────────────────────────────
    shade_color = CYAN if d == "LONG" else ORANGE
    ax.axvspan(j_e_rel, j_x_rel + 0.5, alpha=0.07, color=shade_color, zorder=1)

    # ── SL and TP horizontal lines ─────────────────────────────────────────
    x_left  = max(0, j_e_rel - 2)
    x_right = min(len(xs) - 1, j_x_rel + 2)
    ax.hlines(sl0, x_left, x_right, colors=RED, lw=1.5, ls="--",
              alpha=0.9, zorder=5, label=f"SL  {sl0:.4g}")
    ax.hlines(tp_p, x_left, x_right, colors=GREEN, lw=1.5, ls="--",
              alpha=0.9, zorder=5, label=f"TP  {tp_p:.4g}")

    # ── Entry arrow ────────────────────────────────────────────────────────
    entry_marker = "^" if d == "LONG" else "v"
    entry_color  = CYAN if d == "LONG" else ORANGE
    ax.scatter([j_e_rel], [entry_p], color=entry_color, s=200,
               marker=entry_marker, zorder=8, edgecolors=WHITE, linewidths=0.5,
               label=f"Entry  {entry_p:.5g}")

    # Annotate entry
    y_ann = lo = df.iloc[idx_s + j_e_rel]["low"]
    offset = (df.iloc[idx_s + j_e_rel]["high"] - df.iloc[idx_s + j_e_rel]["low"]) * 0.5
    if d == "LONG":
        ax.annotate(f"  ENTRY\n  {entry_p:.4g}", xy=(j_e_rel, entry_p),
                    xytext=(j_e_rel + 0.4, entry_p - offset * 2),
                    fontsize=7, color=entry_color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=entry_color, lw=0.6))
    else:
        ax.annotate(f"  ENTRY\n  {entry_p:.4g}", xy=(j_e_rel, entry_p),
                    xytext=(j_e_rel + 0.4, entry_p + offset * 2),
                    fontsize=7, color=entry_color, fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=entry_color, lw=0.6))

    # ── Exit marker ────────────────────────────────────────────────────────
    exit_color   = GREEN if trade["win"] else RED
    exit_label   = f"{'✓ TP' if trade['reason']=='TP' else ('✗ SL' if trade['reason']=='SL' else '[T] Timeout')}  {trade['exit']:.5g}"
    ax.scatter([j_x_rel], [trade["exit"]], color=exit_color, s=180,
               marker="X", zorder=8, edgecolors=WHITE, linewidths=0.5,
               label=exit_label)

    # ── Vertical lines at entry/exit ────────────────────────────────────────
    ax.axvline(j_e_rel, color=entry_color, lw=0.6, ls=":", alpha=0.5, zorder=2)
    ax.axvline(j_x_rel, color=exit_color,  lw=0.6, ls=":", alpha=0.5, zorder=2)

    # ── News event annotations ──────────────────────────────────────────────
    if len(ts_sub) > 0:
        win_start = ts_sub[0]
        win_end   = ts_sub[-1]
        for ev in events:
            if win_start <= ev["ts"] <= win_end:
                x_ev = int(np.searchsorted(ts_sub, ev["ts"], side="right")) - 1
                if 0 <= x_ev < len(xs):
                    hi_bar = df.iloc[idx_s + x_ev]["high"]
                    price_range = df.iloc[idx_s:idx_e]["high"].max() - df.iloc[idx_s:idx_e]["low"].min()
                    ax.annotate(
                        f"⚡ {ev['name'][:22]}",
                        xy=(x_ev, hi_bar),
                        xytext=(x_ev, hi_bar + price_range * 0.06),
                        fontsize=6, color=YELLOW, ha="center", va="bottom",
                        arrowprops=dict(arrowstyle="-", color=YELLOW, lw=0.6),
                        zorder=9,
                    )
                    ax.axvline(x_ev, color=YELLOW, lw=0.5, ls="--", alpha=0.4, zorder=2)

    # ── Y-axis range with padding ───────────────────────────────────────────
    price_lo = df.iloc[idx_s:idx_e]["low"].min()
    price_hi = df.iloc[idx_s:idx_e]["high"].max()
    prange   = price_hi - price_lo
    ax.set_ylim(min(price_lo, sl0) - prange * 0.08,
                max(price_hi, tp_p) + prange * 0.15)

    # ── Title ──────────────────────────────────────────────────────────────
    r_str    = f"{trade['pnl_r']:+.2f}R  ({trade['pnl']:+.2f} USDT)"
    win_str  = "✓ WIN" if trade["win"] else "✗ LOSS"
    ctx_str  = f" [L4:{trade.get('context','normal')}]" if trade.get("context") != "normal" else ""
    strat_tf = _strat_timeframe(trade["strat"])
    ax.set_title(
        f"{trade['sym']}/USDT  ·  {trade['strat']} ({strat_tf})"
        f"  ·  {d}"
        f"  ·  {win_str}  {r_str}{ctx_str}\n"
        f"Entry: {ts_fmt(trade['entry_ts'])} UTC    Exit: {ts_fmt(trade['exit_ts'])} UTC    "
        f"Reason: {trade['reason']}",
        color=WHITE, fontsize=10, pad=8, loc="left",
    )

    # ── Legend ─────────────────────────────────────────────────────────────
    ax.legend(loc="upper right", fontsize=7.5, framealpha=0.25,
              facecolor=BG, labelcolor=WHITE, edgecolor="#30363d")

    # ── Risk info box ───────────────────────────────────────────────────────
    risk_txt = (f"Risk: {trade['risk_usd']:.2f} USDT"
                f"  |  mult: {trade.get('risk_mult', 1.0):.2f}x"
                f"  |  PnL_R: {trade['pnl_r']:+.2f}")
    ax.text(0.01, 0.02, risk_txt, transform=ax.transAxes,
            fontsize=7, color=GREY, ha="left", va="bottom")

    plt.tight_layout(pad=1.2)

    # ── Save ───────────────────────────────────────────────────────────────
    entry_ts_str = pd.Timestamp(trade["entry_ts"], unit="ms", tz="UTC").strftime("%Y%m%d_%H%M")
    result_tag   = "WIN" if trade["win"] else "LOSS"
    fname = out_dir / f"{entry_ts_str}_{trade['sym']}_{trade['strat']}_{d}_{result_tag}.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight", facecolor=BG)
    plt.close()
    return fname


def _strat_timeframe(strat: str) -> str:
    mapping = {
        "smacross":    "1D",
        "supertrend":  "4H",
        "rsi_rev":     "4H",
        "fib_pump":    "4H",
        "kc_4h":       "4H",
        "donchian50_1d": "1D",
        "1h_pullback": "1H",
    }
    return mapping.get(strat, "?")


def _strat_tf_key(strat: str) -> str:
    mapping = {
        "smacross":    "1d",
        "supertrend":  "4h",
        "rsi_rev":     "4h",
        "fib_pump":    "4h",
        "kc_4h":       "4h",
        "donchian50_1d": "1d",
        "1h_pullback": "1h",
    }
    return mapping.get(strat, "1d")


# ─── Summary chart ─────────────────────────────────────────────────────────────

def plot_summary(trades: list, period_label: str, out_dir: Path) -> Path:
    """Generate a summary equity curve + trade table for the period."""
    if not trades:
        return None

    trades_sorted = sorted(trades, key=lambda t: t["entry_ts"])
    n = len(trades_sorted)

    fig = plt.figure(figsize=(16, 10), facecolor=BG)
    gs  = GridSpec(2, 1, figure=fig, height_ratios=[1, 2], hspace=0.4)

    # ── Top: equity curve (running cumulative R) ────────────────────────────
    ax_eq = fig.add_subplot(gs[0], facecolor=BG_AX)
    for sp in ax_eq.spines.values(): sp.set_color("#30363d")
    ax_eq.tick_params(colors=GREY)

    cum_r = np.cumsum([t["pnl_r"] for t in trades_sorted])
    xs    = np.arange(n)
    ax_eq.plot(xs, cum_r, color=CYAN, lw=1.5, zorder=3)
    ax_eq.fill_between(xs, 0, cum_r, where=cum_r >= 0, alpha=0.15, color=GREEN)
    ax_eq.fill_between(xs, 0, cum_r, where=cum_r < 0,  alpha=0.15, color=RED)
    ax_eq.axhline(0, color=GREY, lw=0.7, ls="--", alpha=0.5)
    ax_eq.set_ylabel("Cumulative R", color=GREY, fontsize=9)
    ax_eq.set_xticks(xs)
    ax_eq.set_xticklabels(
        [f"{t['sym']}\n{ts_fmt(t['entry_ts'], '%m/%d')}" for t in trades_sorted],
        fontsize=6.5, color=GREY, rotation=30
    )
    ax_eq.set_title(f"V7+L4 Trades — {period_label}  "
                    f"({n} trades, WR={sum(1 for t in trades_sorted if t['win'])/n*100:.0f}%)",
                    color=WHITE, fontsize=11, pad=6)

    # Annotate each R value
    for i, t in enumerate(trades_sorted):
        color = GREEN if t["win"] else RED
        ax_eq.annotate(f"{t['pnl_r']:+.1f}R",
                       xy=(i, cum_r[i]), xytext=(i, cum_r[i] + 0.15),
                       fontsize=6, color=color, ha="center", va="bottom")

    # ── Bottom: trade table ─────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1], facecolor=BG)
    ax_tbl.axis("off")

    cols = ["#", "Symbol", "Strategy", "Dir", "Entry Time", "Exit Time",
            "Entry $", "Exit $", "Reason", "PnL (R)", "PnL (USDT)", "Context"]
    rows = []
    for i, t in enumerate(trades_sorted):
        rows.append([
            str(i+1),
            t["sym"],
            t["strat"],
            t["direction"],
            ts_fmt(t["entry_ts"], "%m/%d %H:%M"),
            ts_fmt(t["exit_ts"],  "%m/%d %H:%M"),
            f"{t['entry']:.4g}",
            f"{t['exit']:.4g}",
            t["reason"],
            f"{t['pnl_r']:+.2f}R",
            f"{t['pnl']:+.2f}",
            t.get("context", "normal"),
        ])

    # Cell colours (win=green row, loss=red row)
    cell_colors = []
    for t in trades_sorted:
        base = "#1a2d1a" if t["win"] else "#2d1a1a"
        cell_colors.append([base] * len(cols))

    tbl = ax_tbl.table(cellText=rows, colLabels=cols, loc="center",
                       cellLoc="center", cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.55)

    # Style header
    for j in range(len(cols)):
        tbl[(0, j)].set_facecolor("#1c2128")
        tbl[(0, j)].set_text_props(color=WHITE, fontweight="bold")
    # Style data cells
    for i in range(1, len(rows) + 1):
        for j in range(len(cols)):
            tbl[(i, j)].set_text_props(color=WHITE)
            # Highlight PnL column
            if cols[j] == "PnL (R)":
                is_win = trades_sorted[i-1]["win"]
                tbl[(i, j)].set_text_props(
                    color=GREEN if is_win else RED, fontweight="bold"
                )

    plt.tight_layout()
    fname = out_dir / f"_summary_{period_label.replace(' ','_')}.png"
    plt.savefig(fname, dpi=120, bbox_inches="tight", facecolor=BG)
    plt.close()
    return fname


# ─── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="V7+L4 trade charts")
    parser.add_argument("--month",       default="2026-04",
                        help="Show trades with entry in YYYY-MM (default: 2026-04)")
    parser.add_argument("--all-recent",  action="store_true",
                        help="Show Feb–Apr 2026 all trades")
    parser.add_argument("--active-april", action="store_true",
                        help="Also include trades that were OPEN during April (entry before April)")
    args = parser.parse_args()

    OUT_DIR.mkdir(exist_ok=True)

    # ── Load data & regime ──────────────────────────────────────────────────
    print("Loading BTC regime...")
    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"  {len(syms)} symbols available")

    # ── Generate signals + simulate ─────────────────────────────────────────
    print("Generating V7 signals (bear=long_only)...")
    sigs, sim_dfs, info = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    print(f"  {len(sigs)} signals")

    print("Loading events + sentiment...")
    events    = load_all_events()
    sentiment = build_sentiment_series()

    print("Running simulation (L4=True)...")
    res = simulate_news_aware(
        sigs, sim_dfs, regime, events, sentiment, base_risk=0.04,
        enable_l4=True, enable_l3=False, enable_l3_positive=False, enable_l1=False,
    )
    all_trades = res["trades"]
    print(f"  Total trades: {len(all_trades)}")

    # ── Filter by period ────────────────────────────────────────────────────
    if args.all_recent:
        t_start = int(datetime(2026, 2, 1, tzinfo=timezone.utc).timestamp() * 1000)
        t_end   = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
        period_label = "Feb–Apr 2026"
    else:
        year, mo = int(args.month[:4]), int(args.month[5:7])
        t_start  = int(datetime(year, mo, 1, tzinfo=timezone.utc).timestamp() * 1000)
        if mo == 12:
            t_end = int(datetime(year + 1, 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        else:
            t_end = int(datetime(year, mo + 1, 1, tzinfo=timezone.utc).timestamp() * 1000)
        period_label = args.month

    # Trades with entry in the period
    period_trades = [t for t in all_trades if t_start <= t["entry_ts"] < t_end]

    # Optionally include trades that were ACTIVE (open) during the period even if entry was earlier
    if args.active_april:
        APR_START = int(datetime(2026, 4, 1, tzinfo=timezone.utc).timestamp() * 1000)
        APR_END   = int(datetime(2026, 5, 1, tzinfo=timezone.utc).timestamp() * 1000)
        active = [
            t for t in all_trades
            if t["entry_ts"] < APR_START <= t["exit_ts"]  # open at April start
            and t not in period_trades
        ]
        period_trades = sorted(active + period_trades, key=lambda x: x["entry_ts"])
        period_label  += " (incl. carry-overs)"

    print(f"\n  {period_label}: {len(period_trades)} trades")
    if not period_trades:
        print("  No trades found. Try --all-recent or --active-april")
        return

    # ── Build signal lookup table for SL/TP ─────────────────────────────────
    # key: (sym, strat) → list of signals, sorted by next_open_ts
    from collections import defaultdict
    sig_by_key = defaultdict(list)
    for s in sigs:
        sig_by_key[(s.sym, s.strat)].append(s)

    def find_signal(trade):
        candidates = sig_by_key.get((trade["sym"], trade["strat"]), [])
        best = None
        best_diff = 10**15
        for s in candidates:
            diff = abs(s.next_open_ts - trade["entry_ts"])
            if diff < best_diff:
                best_diff = diff
                best = s
        return best

    # ── Plot summary first ──────────────────────────────────────────────────
    print(f"\nGenerating summary chart...")
    sfname = plot_summary(period_trades, period_label, OUT_DIR)
    if sfname:
        print(f"  → {sfname.name}")

    # ── Plot individual trade charts ─────────────────────────────────────────
    print(f"\nGenerating {len(period_trades)} trade charts...")
    for i, trade in enumerate(sorted(period_trades, key=lambda t: t["entry_ts"])):
        sig = find_signal(trade)
        if sig is None:
            print(f"  [{i+1}/{len(period_trades)}] {trade['sym']} {trade['strat']}: "
                  f"signal not found, skipping")
            continue

        # Load correct timeframe dataframe
        tf = _strat_tf_key(trade["strat"])
        df = load(trade["sym"], tf, "730d")
        if df is None:
            print(f"  [{i+1}/{len(period_trades)}] {trade['sym']} {tf}: data not found")
            continue

        result_tag = "WIN" if trade["win"] else "LOSS"
        entry_s    = ts_fmt(trade["entry_ts"])
        print(f"  [{i+1}/{len(period_trades)}] {trade['sym']:6s} {trade['strat']:15s} "
              f"{trade['direction']:5s} {entry_s}  {result_tag}  {trade['pnl_r']:+.2f}R  "
              f"→ SL={sig.sl:.4g} TP={sig.tp:.4g}",
              end=" ")

        fname = plot_trade_chart(trade, sig, df, regime, events, OUT_DIR)
        if fname:
            print(f"→ {fname.name}")
        else:
            print("→ skipped (no data)")

    print(f"\nDone! Charts saved to: {OUT_DIR}")
    print(f"  Total files: {len(list(OUT_DIR.glob('*.png')))}")

    # Print final summary to console
    wins   = sum(1 for t in period_trades if t["win"])
    losses = len(period_trades) - wins
    sum_r  = sum(t["pnl_r"] for t in period_trades)
    gw     = sum(t["pnl_r"] for t in period_trades if t["win"])
    gl     = abs(sum(t["pnl_r"] for t in period_trades if not t["win"]))
    pf     = gw / gl if gl > 0 else 99

    print(f"\n{'═'*55}")
    print(f"  {period_label} summary")
    print(f"{'═'*55}")
    print(f"  Trades  : {len(period_trades)}")
    print(f"  Wins    : {wins}  Losses: {losses}  WR: {wins/max(1,len(period_trades))*100:.0f}%")
    print(f"  Sum R   : {sum_r:+.2f}R")
    print(f"  PF      : {pf:.2f}")
    for t in sorted(period_trades, key=lambda x: x["entry_ts"]):
        wr  = "✓" if t["win"] else "✗"
        ctx = f"[{t['context']}]" if t.get("context") != "normal" else ""
        print(f"  {wr} {t['sym']:6s} {t['strat']:15s} {t['direction']:5s} "
              f"{ts_fmt(t['entry_ts'])} → {ts_fmt(t['exit_ts'])}  "
              f"{t['pnl_r']:+.2f}R  {t['reason']:7s} {ctx}")


if __name__ == "__main__":
    main()
