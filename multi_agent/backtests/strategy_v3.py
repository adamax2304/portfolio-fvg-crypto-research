"""
strategy_v3.py — New strategies based on user hypotheses
========================================================
2026-05-02

1. fib_pump_retrace_1h / 4h
   - Detect: 24-72h consolidation (low ATR/range)
   - Followed by pump: large green candle(s) breaking out
   - Mark anchor (pump start = first expansion bar) and top (pump high)
   - Wait for retrace to 0.5 of (top - anchor)
   - Enter LONG; SL just below anchor; TP at top + extension
   - SHORT version: red consolidation breakdown + retrace to 0.5

2. pivot_sl_supertrend_4h / smacross_1d
   - SuperTrend signals + SL at nearest swing low/high (5-bar pivot)
   - Hypothesis: structural stops survive manipulation better

3. mtf_4h_1h: 4h trend (EMA50 + supertrend) + 1h pullback to EMA21 entry
4. mtf_4h_15m: 4h trend + 15m breakout entry
"""
from __future__ import annotations
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, sma, ema, rsi, supertrend, donchian,
    _ts_period_ms, _emit, Signal, simulate_signals, metrics, regime_at,
    metrics_by_regime, walk_forward, SYMBOLS_730D,
)


# ────────────────────────────────────────────────────────────────────
# PIVOT DETECTION (swing high/low)
# ────────────────────────────────────────────────────────────────────

def find_pivots(hi, lo, lookback=5):
    """Return arrays of pivot_high_value and pivot_low_value at each bar.
    A pivot is the high/low that's the max/min over [-lookback..+lookback].
    Since we can't see future, we mark pivot on bar i+lookback (delayed)."""
    n = len(hi)
    pivot_hi = np.full(n, np.nan)
    pivot_lo = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        hi_seg = hi[i-lookback:i+lookback+1]
        lo_seg = lo[i-lookback:i+lookback+1]
        if hi[i] == hi_seg.max():
            pivot_hi[i + lookback] = hi[i]  # mark as available `lookback` bars later
        if lo[i] == lo_seg.min():
            pivot_lo[i + lookback] = lo[i]
    return pivot_hi, pivot_lo


def latest_pivot_low(pivot_lo, i, max_lookback=30):
    for k in range(i, max(0, i - max_lookback) - 1, -1):
        if not np.isnan(pivot_lo[k]):
            return pivot_lo[k], k
    return None, None

def latest_pivot_high(pivot_hi, i, max_lookback=30):
    for k in range(i, max(0, i - max_lookback) - 1, -1):
        if not np.isnan(pivot_hi[k]):
            return pivot_hi[k], k
    return None, None


# ────────────────────────────────────────────────────────────────────
# STRATEGY: SuperTrend + pivot-based SL
# ────────────────────────────────────────────────────────────────────

def strat_supertrend_pivot_4h(df, regime, tp_r=2.0, pivot_lookback=5,
                               buffer_atr=0.3, regime_filter=False):
    if df is None or len(df) < 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    st, sdir = supertrend(hi, lo, cl, 10, 3.0)
    piv_hi, piv_lo = find_pivots(hi, lo, lookback=pivot_lookback)
    sigs = []
    for i in range(30, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if sdir[i] == 1 and sdir[i-1] == -1:
            d = "LONG"
            piv, piv_idx = latest_pivot_low(piv_lo, i, max_lookback=20)
            if piv is None: continue
            next_open = op[i+1]
            sl = piv - buffer_atr * a[i]
            if sl >= next_open: continue
            tp = next_open + tp_r * (next_open - sl)
            trend, _ = regime_at(regime, int(ts[i]))
            if regime_filter and trend < 0: continue
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"piv_dist_atr": (next_open - sl) / a[i]})
        elif sdir[i] == -1 and sdir[i-1] == 1:
            d = "SHORT"
            piv, piv_idx = latest_pivot_high(piv_hi, i, max_lookback=20)
            if piv is None: continue
            next_open = op[i+1]
            sl = piv + buffer_atr * a[i]
            if sl <= next_open: continue
            tp = next_open - tp_r * (sl - next_open)
            trend, _ = regime_at(regime, int(ts[i]))
            if regime_filter and trend > 0: continue
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"piv_dist_atr": (sl - next_open) / a[i]})
    return sigs


def strat_smacross_pivot_1d(df, regime, tp_r=3.0, pivot_lookback=3,
                             buffer_atr=0.3):
    if df is None or len(df) < 220: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    s50 = sma(cl, 50); s200 = sma(cl, 200)
    piv_hi, piv_lo = find_pivots(hi, lo, lookback=pivot_lookback)
    sigs = []
    for i in range(201, len(cl) - 1):
        if np.isnan(s50[i]) or np.isnan(s200[i]) or np.isnan(a[i]): continue
        if np.isnan(s50[i-1]) or np.isnan(s200[i-1]): continue
        next_open = op[i+1]
        if s50[i-1] <= s200[i-1] and s50[i] > s200[i]:
            piv, _ = latest_pivot_low(piv_lo, i, max_lookback=15)
            if piv is None: continue
            sl = piv - buffer_atr * a[i]
            if sl >= next_open: continue
            tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif s50[i-1] >= s200[i-1] and s50[i] < s200[i]:
            piv, _ = latest_pivot_high(piv_hi, i, max_lookback=15)
            if piv is None: continue
            sl = piv + buffer_atr * a[i]
            if sl <= next_open: continue
            tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


# ────────────────────────────────────────────────────────────────────
# STRATEGY: Fibonacci pump retracement
# ────────────────────────────────────────────────────────────────────

def strat_fib_pump_1h(df, regime, sl_buffer_atr=0.5, tp_extension=1.0,
                      consolidation_bars=24, consolidation_max_range_atr=2.0,
                      pump_min_atr=2.0, pump_max_bars=8,
                      retrace_min=0.382, retrace_max=0.618,
                      retrace_wait_bars=24, regime_filter=False):
    """
    Pattern:
      1. Consolidation: last `consolidation_bars` had range < `consolidation_max_range_atr` × ATR
      2. Pump: within next `pump_max_bars`, price moves > `pump_min_atr` × ATR above consolidation high
      3. Anchor = consolidation high (LONG) or low (SHORT) [start of pump]
         Top = highest high during pump phase
      4. Wait for retrace to 0.382-0.618 (between retrace_min and retrace_max) of (top - anchor)
      5. Enter; SL just below anchor (or above for SHORT) − sl_buffer; TP at top + extension
    """
    if df is None or len(df) < 100: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    n = len(cl)
    sigs = []
    i = consolidation_bars + 14
    while i < n - retrace_wait_bars - 1:
        if np.isnan(a[i]) or a[i] <= 0:
            i += 1; continue
        # Consolidation check: range over last `consolidation_bars`
        cons_hi = float(np.max(hi[i-consolidation_bars:i]))
        cons_lo = float(np.min(lo[i-consolidation_bars:i]))
        cons_range = cons_hi - cons_lo
        avg_atr_cons = float(np.nanmean(a[i-consolidation_bars:i]))
        if avg_atr_cons <= 0:
            i += 1; continue
        if cons_range > consolidation_max_range_atr * avg_atr_cons:
            i += 1; continue

        # Look for pump in next pump_max_bars
        # LONG pump: high breaks consolidation high by pump_min_atr × ATR
        # SHORT pump: low breaks consolidation low by pump_min_atr × ATR
        long_pump_idx = None; short_pump_idx = None
        for k in range(i, min(i + pump_max_bars, n)):
            if hi[k] >= cons_hi + pump_min_atr * a[i]:
                long_pump_idx = k
                break
            if lo[k] <= cons_lo - pump_min_atr * a[i]:
                short_pump_idx = k
                break
        if long_pump_idx is None and short_pump_idx is None:
            i += 1; continue

        # Process LONG case
        if long_pump_idx is not None:
            # Find pump top: highest high within pump_max_bars from pump start
            pump_end = min(long_pump_idx + pump_max_bars, n - 1)
            top_idx = long_pump_idx + int(np.argmax(hi[long_pump_idx:pump_end+1]))
            top = hi[top_idx]
            anchor = cons_hi  # bottom of measurement = consolidation high
            move = top - anchor
            if move <= 0:
                i = top_idx + 1; continue
            # Wait for retrace
            entered = False
            for k in range(top_idx + 1, min(top_idx + 1 + retrace_wait_bars, n)):
                # Did price retrace into [retrace_min..retrace_max] band?
                fib_hi_band = top - retrace_min * move
                fib_lo_band = top - retrace_max * move
                # Check if low touched band
                if lo[k] <= fib_hi_band and lo[k] >= fib_lo_band:
                    if k + 1 >= n: break
                    next_open = op[k+1]
                    # Strict cons low base (anchor for SL = consolidation high - we use cons_lo for SL)
                    sl_base = cons_hi  # the actual breakout level — broken support if revisited
                    # better: SL = consolidation low (deeper)
                    sl = cons_hi - sl_buffer_atr * a[k]
                    if sl >= next_open: break
                    tp = top + tp_extension * move
                    trend, _ = regime_at(regime, int(ts[k]))
                    if regime_filter and trend < 0: break
                    _emit(sigs, ts[k], period_ms, "LONG", next_open, sl, tp,
                          info={"top": top, "anchor": anchor, "move": move,
                                "retrace": (top - lo[k]) / move if move > 0 else 0})
                    entered = True
                    break
                # Invalidate if went below cons low (broke structure)
                if lo[k] < cons_lo:
                    break
            i = (top_idx + 1) if not entered else (k + 1)
            continue
        # Process SHORT case (symmetric)
        if short_pump_idx is not None:
            pump_end = min(short_pump_idx + pump_max_bars, n - 1)
            bot_idx = short_pump_idx + int(np.argmin(lo[short_pump_idx:pump_end+1]))
            bot = lo[bot_idx]
            anchor = cons_lo
            move = anchor - bot
            if move <= 0:
                i = bot_idx + 1; continue
            entered = False
            for k in range(bot_idx + 1, min(bot_idx + 1 + retrace_wait_bars, n)):
                fib_hi_band = bot + retrace_max * move
                fib_lo_band = bot + retrace_min * move
                if hi[k] >= fib_lo_band and hi[k] <= fib_hi_band:
                    if k + 1 >= n: break
                    next_open = op[k+1]
                    sl = cons_lo + sl_buffer_atr * a[k]
                    if sl <= next_open: break
                    tp = bot - tp_extension * move
                    trend, _ = regime_at(regime, int(ts[k]))
                    if regime_filter and trend > 0: break
                    _emit(sigs, ts[k], period_ms, "SHORT", next_open, sl, tp,
                          info={"bot": bot, "anchor": anchor, "move": move,
                                "retrace": (hi[k] - bot) / move if move > 0 else 0})
                    entered = True
                    break
                if hi[k] > cons_hi:
                    break
            i = (bot_idx + 1) if not entered else (k + 1)
            continue
        i += 1
    return sigs


# ────────────────────────────────────────────────────────────────────
# STRATEGY: MTF — 4h trend + 1h pullback
# ────────────────────────────────────────────────────────────────────

def strat_mtf_4h_1h(df_4h, df_1h, regime, sl_atr=2.0, tp_r=2.5):
    if df_4h is None or df_1h is None: return []
    if len(df_4h) < 50 or len(df_1h) < 50: return []
    cl4 = df_4h["close"].to_numpy(); ts4 = df_4h["timestamp"].to_numpy()
    hi4 = df_4h["high"].to_numpy(); lo4 = df_4h["low"].to_numpy()
    e50_4h = ema(cl4, 50); e200_4h = ema(cl4, 200)
    st_4h, sdir_4h = supertrend(hi4, lo4, cl4, 10, 3.0)
    period4 = _ts_period_ms(ts4)
    # 1h
    cl1 = df_1h["close"].to_numpy(); ts1 = df_1h["timestamp"].to_numpy()
    hi1 = df_1h["high"].to_numpy(); lo1 = df_1h["low"].to_numpy()
    op1 = df_1h["open"].to_numpy()
    a1 = atr(hi1, lo1, cl1, 14)
    e21_1h = ema(cl1, 21); e50_1h = ema(cl1, 50)
    period1 = _ts_period_ms(ts1)
    sigs = []
    for j in range(50, len(cl1) - 1):
        if np.isnan(a1[j]) or a1[j] <= 0: continue
        if np.isnan(e21_1h[j]) or np.isnan(e50_1h[j]): continue
        # find current 4h state
        i4 = int(np.searchsorted(ts4, ts1[j], side="right")) - 1
        if i4 < 50: continue
        if np.isnan(e50_4h[i4]) or np.isnan(e200_4h[i4]): continue
        # Determine 4h trend
        long_trend = (cl4[i4] > e50_4h[i4] > e200_4h[i4]) and sdir_4h[i4] == 1
        short_trend = (cl4[i4] < e50_4h[i4] < e200_4h[i4]) and sdir_4h[i4] == -1
        if not (long_trend or short_trend): continue
        next_open = op1[j+1]
        # Pullback to ema21 on 1h
        if long_trend:
            # Recent pullback: low touched ema21 in last 3 bars, current close > ema21
            recent_pb = any(lo1[k] <= e21_1h[k] for k in range(max(0, j-3), j+1))
            if recent_pb and cl1[j] > e21_1h[j] and cl1[j] > cl1[j-1]:
                sl = next_open - sl_atr * a1[j]
                tp = next_open + tp_r * (next_open - sl)
                _emit(sigs, ts1[j], period1, "LONG", next_open, sl, tp)
        else:
            recent_pb = any(hi1[k] >= e21_1h[k] for k in range(max(0, j-3), j+1))
            if recent_pb and cl1[j] < e21_1h[j] and cl1[j] < cl1[j-1]:
                sl = next_open + sl_atr * a1[j]
                tp = next_open - tp_r * (sl - next_open)
                _emit(sigs, ts1[j], period1, "SHORT", next_open, sl, tp)
    return sigs


# ────────────────────────────────────────────────────────────────────
# RUNNER
# ────────────────────────────────────────────────────────────────────

def run(strat_name, fn, syms, regime, tag, **kwargs):
    all_trades = []
    sym_count = 0; sig_count = 0
    for sym in syms:
        if strat_name == "fib_pump_1h":
            df = load(sym, "1h", tag)
        elif strat_name == "fib_pump_4h":
            df = load(sym, "4h", tag)
        elif strat_name == "supertrend_pivot_4h":
            df = load(sym, "4h", tag)
        elif strat_name == "smacross_pivot_1d":
            df = load(sym, "1d", tag)
        elif strat_name == "mtf_4h_1h":
            df_4h = load(sym, "4h", tag); df_1h = load(sym, "1h", tag)
            if df_4h is None or df_1h is None: continue
            sym_count += 1
            sigs = fn(df_4h, df_1h, regime, **kwargs)
            sig_count += len(sigs)
            if sigs:
                trades = simulate_signals(sym, strat_name, sigs, df_1h, regime)
                all_trades.extend(trades)
            continue
        else:
            df = load(sym, "1h", tag)
        if df is None: continue
        sym_count += 1
        sigs = fn(df, regime, **kwargs)
        sig_count += len(sigs)
        if sigs:
            trades = simulate_signals(sym, strat_name, sigs, df, regime)
            all_trades.extend(trades)
    full = metrics(all_trades)
    by_reg = metrics_by_regime(all_trades)
    wf = walk_forward(all_trades, folds=6)
    return {"strategy": strat_name, "syms": sym_count, "sigs": sig_count,
            "n": len(all_trades), "full": full, "by_regime": by_reg,
            "walk_forward": wf, "trades": all_trades}


def print_res(r):
    s = r["strategy"]
    f = r["full"]
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  {s}  syms={r['syms']}  sigs={r['sigs']}  trades={r['n']}")
    print(f"══════════════════════════════════════════════════════════════════")
    if f.get("n",0) == 0:
        print("  no trades"); return
    print(f"  FULL    n={f['n']:5d}  WR={f['wr']:5.1f}%  PF={f['pf']:5.2f}  avgR={f['avg_r']:+.3f}  sumR={f['sum_r']:+.1f}  ddR={f['max_dd_r']:.1f}")
    for label in ("bull", "sideways", "bear"):
        m = r["by_regime"].get(label, {"n":0})
        if m["n"] == 0:
            print(f"  {label:8s} n={m.get('n',0):5d}  (no trades)")
        else:
            print(f"  {label:8s} n={m['n']:5d}  WR={m['wr']:5.1f}%  PF={m['pf']:5.2f}  avgR={m['avg_r']:+.3f}  sumR={m['sum_r']:+.1f}")
    print(f"  Walk-forward folds:")
    for fd in r["walk_forward"]:
        t0 = pd.Timestamp(fd['from_ts'], unit='ms', tz='UTC').strftime('%Y-%m-%d')
        t1 = pd.Timestamp(fd['to_ts'], unit='ms', tz='UTC').strftime('%Y-%m-%d')
        print(f"    fold {fd['fold']}  {t0}→{t1}  n={fd['n']:4d}  WR={fd['wr']:5.1f}%  PF={fd['pf']:5.2f}  sumR={fd['sum_r']:+.1f}")


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = SYMBOLS_730D

    results = []
    print("\n══════ Phase C: Pivot-based SL ══════")
    results.append(run("supertrend_pivot_4h", strat_supertrend_pivot_4h, syms, regime, "730d", tp_r=2.0))
    print_res(results[-1])
    results.append(run("smacross_pivot_1d", strat_smacross_pivot_1d, syms, regime, "730d", tp_r=3.0))
    print_res(results[-1])

    print("\n══════ Phase D: Fibonacci pump retracement ══════")
    # 1h
    results.append(run("fib_pump_1h", strat_fib_pump_1h, syms, regime, "730d",
                       consolidation_bars=24, pump_min_atr=2.0,
                       pump_max_bars=8, retrace_min=0.382, retrace_max=0.618,
                       sl_buffer_atr=0.5, tp_extension=1.0))
    print_res(results[-1])
    # 4h
    results.append(run("fib_pump_4h", strat_fib_pump_1h, syms, regime, "730d",
                       consolidation_bars=18, pump_min_atr=2.0,
                       pump_max_bars=6, retrace_min=0.382, retrace_max=0.618,
                       sl_buffer_atr=0.5, tp_extension=1.0))
    print_res(results[-1])

    print("\n══════ Phase B: MTF 4h trend + 1h pullback ══════")
    results.append(run("mtf_4h_1h", strat_mtf_4h_1h, syms, regime, "730d",
                       sl_atr=2.0, tp_r=2.5))
    print_res(results[-1])

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V3 SUMMARY")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Strategy':25s} {'n':>5s} {'WR%':>6s} {'PF':>6s} {'avgR':>7s} {'sumR':>8s}")
    for r in results:
        f = r["full"]
        if f.get("n",0) == 0:
            print(f"  {r['strategy']:25s} {0:>5d} {0:>6.1f} {0:>6.2f} {0:+7.3f} {0:+8.1f}")
        else:
            print(f"  {r['strategy']:25s} {f['n']:>5d} {f['wr']:>6.1f} {f['pf']:>6.2f} {f['avg_r']:+7.3f} {f['sum_r']:+8.1f}")


if __name__ == "__main__":
    main()
