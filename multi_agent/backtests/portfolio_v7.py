"""
portfolio_v7.py — Robust improvements over V3 (each tested on 730d only)
=========================================================================
Goal: push beyond V3's +335% / 31% DD without overfitting.

EVERY new feature must IMPROVE 730d return AND not break walk-forward.

Tests:
  1. Bear regime handling (skip / short-only / lower-risk)
  2. Equity-curve risk scaling (anti-martingale / conservative-DD / Kelly)
  3. New uncorrelated strategies (KC 4h / Donchian 1d 50-bar / 1h reversal)

Baseline: V3 ORIGINAL @ 4% risk on 730d = +335%, DD 31%, 80% WF positive
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
    load, build_regime, atr, sma, ema, rsi, supertrend, donchian,
    PortfolioSignal, _ts_period_ms, _emit, Signal, regime_at,
    metrics_dict_from_trades, _monthly_returns,
    START_BALANCE, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H,
    MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN, SL_COOLDOWN_MS,
    simulate_portfolio,
)
from multi_agent.backtests.strategy_v3 import (
    strat_supertrend_pivot_4h, strat_smacross_pivot_1d, strat_fib_pump_1h,
    find_pivots, latest_pivot_low, latest_pivot_high,
)
from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2

CACHE_DIR = Path(__file__).parent / "data_cache"
BASELINE = ["ADA","ALGO","APT","ARB","ATOM","AVAX","BCH","BNB","BTC",
            "DOGE","DOT","ETH","FIL","HBAR","ICP","INJ","LINK","LTC",
            "NEAR","OP","SOL","SUI","TON","TRX","UNI","XLM","XRP"]


# ─────────────────────────────────────────────────────────────────────────────
# NEW STRATEGIES (each tested standalone first)
# ─────────────────────────────────────────────────────────────────────────────

def strat_kc_squeeze_4h(df, regime, sl_atr=1.5, tp_r=2.0, min_sq=10, mom_bars=6):
    """Keltner Channel squeeze on 4h with momentum direction."""
    if df is None or len(df) < 60: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    sma20 = sma(cl, 20)
    std20 = pd.Series(cl).rolling(20, min_periods=20).std(ddof=0).to_numpy()
    bb_u = sma20 + 2*std20; bb_l = sma20 - 2*std20
    kc_atr = atr(hi, lo, cl, 10); kc_mid = ema(cl, 20)
    kc_u = kc_mid + 1.5*kc_atr; kc_l = kc_mid - 1.5*kc_atr
    sq = (bb_l > kc_l) & (bb_u < kc_u)
    sq_len = np.zeros(len(cl), dtype=int)
    for i in range(1, len(cl)):
        sq_len[i] = sq_len[i-1] + 1 if sq[i] else 0
    sigs = []
    for i in range(50, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if sq[i]: continue
        if sq_len[i-1] < min_sq: continue
        # Momentum: linear regression slope of last `mom_bars` closes
        seg = cl[i-mom_bars:i+1]
        if len(seg) < mom_bars or np.any(np.isnan(seg)): continue
        slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
        next_open = op[i+1]
        if slope > 0 and not np.isnan(bb_u[i]) and bb_u[i] > kc_u[i]:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif slope < 0 and not np.isnan(bb_l[i]) and bb_l[i] < kc_l[i]:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


def strat_donchian_50_1d(df, regime, sl_atr=2.5, tp_r=3.0, lookback=50):
    """Long-term Donchian breakout (50-bar) on 1d, with regime filter."""
    if df is None or len(df) < lookback + 30: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    upper, lower = donchian(hi, lo, lookback)
    sigs = []
    for i in range(lookback + 14, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        next_open = op[i+1]
        trend, _ = regime_at(regime, int(ts[i]))
        prev_upper = upper[i-1]; prev_lower = lower[i-1]
        if cl[i] > prev_upper:
            if trend < 0: continue  # only longs in non-bear
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif cl[i] < prev_lower:
            if trend > 0: continue  # only shorts in non-bull
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


def strat_1h_pullback(df, regime, sl_atr=1.5, tp_r=2.0):
    """1h trend continuation: ema21 pullback + ema50 trend filter."""
    if df is None or len(df) < 60: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    e21 = ema(cl, 21); e50 = ema(cl, 50); e200 = ema(cl, 200)
    sigs = []
    for i in range(50, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if np.isnan(e50[i]) or np.isnan(e200[i]) or np.isnan(e21[i]): continue
        next_open = op[i+1]
        # LONG: ema stack up, low touched ema21 in last 2 bars, current close > ema21
        long_stack = cl[i] > e21[i] > e50[i] > e200[i]
        short_stack = cl[i] < e21[i] < e50[i] < e200[i]
        recent_pullback_long = any(lo[k] <= e21[k] for k in range(max(0, i-2), i+1))
        recent_pullback_short = any(hi[k] >= e21[k] for k in range(max(0, i-2), i+1))
        if long_stack and recent_pullback_long and cl[i] > cl[i-1]:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif short_stack and recent_pullback_short and cl[i] < cl[i-1]:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


# ─────────────────────────────────────────────────────────────────────────────
# V7 SIGNAL COLLECTOR with bear-regime handling
# ─────────────────────────────────────────────────────────────────────────────

def collect_v7_signals(syms, regime, tag,
                        # Strategy toggles
                        include_smacross=True, include_supertrend=True,
                        include_rsi_rev=True, include_fib=True,
                        include_kc=False, include_donchian50=False,
                        include_1h_pullback=False,
                        # Bear regime handling
                        bear_mode="all",  # "all" | "skip" | "short_only" | "long_only"
                        # Per-trade size scaling in bear
                        bear_risk_mult=1.0):
    sigs = []; sim_dfs = {}
    bear_signals_count = 0
    bear_filtered_count = 0

    def add(strat, df, slist, sym, tag_meta=None):
        nonlocal bear_signals_count, bear_filtered_count
        if df is None: return
        sim_dfs[(sym, strat)] = df
        for s in slist:
            trend, _ = regime_at(regime, s.next_open_ts)
            if trend < 0:  # bear
                bear_signals_count += 1
                if bear_mode == "skip":
                    bear_filtered_count += 1; continue
                if bear_mode == "long_only" and s.direction == "SHORT":
                    bear_filtered_count += 1; continue
                if bear_mode == "short_only" and s.direction == "LONG":
                    bear_filtered_count += 1; continue
            sigs.append(PortfolioSignal(
                ts=s.ts, next_open_ts=s.next_open_ts, sym=sym,
                strat=strat, direction=s.direction,
                entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
            ))

    for sym in syms:
        df_1d = load(sym, "1d", tag)
        df_4h = load(sym, "4h", tag)
        df_1h = load(sym, "1h", tag)
        if include_smacross and df_1d is not None:
            add("smacross", df_1d, strat_smacross_pivot_1d(df_1d, regime, tp_r=4.0), sym)
        if include_supertrend and df_4h is not None:
            add("supertrend", df_4h, strat_supertrend_pivot_4h(df_4h, regime, tp_r=2.5), sym)
        if include_rsi_rev and df_4h is not None:
            add("rsi_rev", df_4h, _strat_rsi_rev_v2(df_4h, regime, sl_atr=2.0, tp_r=3.0), sym)
        if include_fib and df_4h is not None:
            add("fib_pump", df_4h, strat_fib_pump_1h(df_4h, regime,
                                                      consolidation_bars=18, pump_min_atr=3.0,
                                                      pump_max_bars=6, retrace_min=0.382,
                                                      retrace_max=0.618, tp_extension=1.0,
                                                      sl_buffer_atr=0.5), sym)
        if include_kc and df_4h is not None:
            add("kc_4h", df_4h, strat_kc_squeeze_4h(df_4h, regime), sym)
        if include_donchian50 and df_1d is not None:
            add("donchian50_1d", df_1d, strat_donchian_50_1d(df_1d, regime), sym)
        if include_1h_pullback and df_1h is not None:
            add("1h_pullback", df_1h, strat_1h_pullback(df_1h, regime), sym)
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs, {"bear_signals": bear_signals_count, "bear_filtered": bear_filtered_count}


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM SIMULATOR with equity-curve risk scaling
# ─────────────────────────────────────────────────────────────────────────────

def simulate_v7(sigs, sim_dfs, regime,
                start_balance=START_BALANCE, base_risk=0.04, max_pos=5,
                # Equity-curve scaling
                equity_curve_mode="constant",  # "constant" | "anti_martingale" | "conservative_dd"
                anti_high_risk=0.05, anti_low_risk=0.02,
                # Bear regime risk multiplier
                bear_risk_mult=1.0,
                sl_cooldown_ms=SL_COOLDOWN_MS):
    balance = start_balance
    peak = start_balance
    open_pos = {}
    sym_cool = {}
    trades = []

    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos: continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts: continue
        # Realize closed
        for k in list(open_pos.keys()):
            pos = open_pos[k]
            if pos["exit_ts"] is not None and pos["exit_ts"] <= sig.next_open_ts:
                balance = max(0.01, balance + pos["pnl"])
                peak = max(peak, balance)
                trades.append(pos)
                if pos["reason"] == "SL":
                    sym_cool[pos["sym"]] = pos["exit_ts"] + sl_cooldown_ms
                del open_pos[k]
        if len(open_pos) >= max_pos: continue

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

        # Determine risk for this trade
        risk = base_risk
        if equity_curve_mode == "anti_martingale":
            # If at new high → use higher risk; if in DD → use lower risk
            dd_pct = (peak - balance) / peak if peak > 0 else 0
            if dd_pct < 0.02:  # within 2% of peak
                risk = anti_high_risk
            elif dd_pct > 0.15:  # > 15% DD
                risk = anti_low_risk
            else:
                # interpolate
                frac = (dd_pct - 0.02) / 0.13
                risk = anti_high_risk * (1 - frac) + anti_low_risk * frac
        elif equity_curve_mode == "conservative_dd":
            dd_pct = (peak - balance) / peak if peak > 0 else 0
            if dd_pct > 0.10:
                risk = base_risk * 0.5  # halve risk in DD
            else:
                risk = base_risk
        # Bear regime multiplier
        trend, _ = regime_at(regime, sig.next_open_ts)
        if trend < 0:
            risk = risk * bear_risk_mult

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
        qty = risk_usd / dist
        # Walk forward
        exit_p = None; exit_bar = None; reason = "timeout"
        for k in range(j, min(j + 500, n)):
            h_k = hi[k]; l_k = lo[k]
            if d == "LONG":
                if l_k <= sl0: exit_p = sl0; reason = "SL"; exit_bar = k; break
                if h_k >= tp_p: exit_p = tp_p; reason = "TP"; exit_bar = k; break
            else:
                if h_k >= sl0: exit_p = sl0; reason = "SL"; exit_bar = k; break
                if l_k <= tp_p: exit_p = tp_p; reason = "TP"; exit_bar = k; break
        if exit_bar is None:
            exit_p = cl[min(j + 499, n - 1)]
            exit_bar = min(j + 499, n - 1)
        bars_held = exit_bar - j + 1
        if d == "LONG":
            slip_exit = exit_p * (1 - SLIPPAGE)
            gross = (slip_exit - entry_p) * qty
        else:
            slip_exit = exit_p * (1 + SLIPPAGE)
            gross = (entry_p - slip_exit) * qty
        entry_fee = entry_p * qty * TAKER_FEE
        exit_fee = slip_exit * qty * TAKER_FEE
        funding = entry_p * qty * funding_per_bar * bars_held
        pnl = gross - entry_fee - exit_fee - funding

        open_pos[key] = {
            "sym": sig.sym, "strat": sig.strat, "direction": d,
            "entry_ts": int(ts[j]), "exit_ts": int(ts[exit_bar]),
            "entry": entry_p, "exit": slip_exit, "reason": reason,
            "pnl": pnl, "risk_usd": risk_usd,
            "pnl_r": pnl / risk_usd if risk_usd > 0 else 0,
            "win": pnl > 0, "regime": int(trend),
        }

    # Realize remaining
    for k in list(open_pos.keys()):
        pos = open_pos[k]
        if pos["exit_ts"] is not None:
            balance = max(0.01, balance + pos["pnl"])
            peak = max(peak, balance)
            trades.append(pos)

    # Summarize
    trades.sort(key=lambda x: x["exit_ts"])
    eq = start_balance; peak2 = start_balance; max_dd = 0
    eq_curve = [(0, start_balance)]
    for t in trades:
        eq = max(0.01, eq + t["pnl"])
        peak2 = max(peak2, eq)
        dd = (peak2 - eq) / peak2 * 100 if peak2 > 0 else 0
        max_dd = max(max_dd, dd)
        eq_curve.append((t["exit_ts"], eq))
    monthly = _monthly_returns(eq_curve, start_balance)
    if not trades:
        return {"trades": [], "metrics": {"n": 0}, "monthly": []}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins); gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else 99
    wr = len(wins) / len(trades) * 100
    avg_r = float(np.mean([t["pnl_r"] for t in trades]))
    return_pct = (eq - start_balance) / start_balance * 100
    return {"trades": trades, "monthly": monthly,
            "metrics": {"n": len(trades), "wr": round(wr, 1), "pf": round(pf, 2),
                         "avg_r": round(avg_r, 3), "return_pct": round(return_pct, 1),
                         "max_dd_pct": round(max_dd, 1), "final_balance": round(eq, 2)}}


def print_res(name, res, info=None):
    m = res["metrics"]
    rets = [r["ret_pct"] for r in res["monthly"]]
    avgmo = float(np.mean(rets)) if rets else 0
    medmo = float(np.median(rets)) if rets else 0
    winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
    extra = f" [info: {info}]" if info else ""
    print(f"  {name:35s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
          f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+7.1f}% DD={m['max_dd_pct']:>5.1f}% "
          f"Avg/mo={avgmo:>+5.1f}% Win={winmo}/{totmo}{extra}")


def print_per_strat(res):
    per = defaultdict(list)
    for t in res["trades"]: per[t["strat"]].append(t)
    for sn, tlist in sorted(per.items()):
        wins = [t for t in tlist if t["pnl"] > 0]
        losses = [t for t in tlist if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins); gl = abs(sum(t["pnl"] for t in losses))
        pf = gw / gl if gl > 0 else 99
        wr = len(wins) / max(1, len(tlist)) * 100
        sumr = sum(t["pnl_r"] for t in tlist)
        print(f"      {sn:15s}: n={len(tlist):4d} WR={wr:5.1f}% PF={pf:5.2f} sumR={sumr:+.1f}")


def main():
    tag = "730d"
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    syms = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_{tag}.parquet").exists()]
    print(f"\n730d symbols: {len(syms)}")

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  V7 BASELINE (V3 ORIGINAL @ 4% risk on 730d)")
    print(f"══════════════════════════════════════════════════════════════════")
    sigs, dfs, info = collect_v7_signals(syms, regime, tag)
    res = simulate_v7(sigs, dfs, regime, base_risk=0.04)
    print_res("V3_BASELINE", res)
    print_per_strat(res)

    print(f"\n══════ Phase 1: Bear-regime handling ══════")
    for mode in ["skip", "long_only", "short_only"]:
        sigs, dfs, info = collect_v7_signals(syms, regime, tag, bear_mode=mode)
        res = simulate_v7(sigs, dfs, regime, base_risk=0.04)
        print_res(f"BEAR={mode}", res, info=f"bear filtered {info['bear_filtered']}/{info['bear_signals']}")

    # Bear with reduced risk
    print(f"\n  Bear with reduced risk (1% in bear, 4% otherwise):")
    sigs, dfs, info = collect_v7_signals(syms, regime, tag)
    res = simulate_v7(sigs, dfs, regime, base_risk=0.04, bear_risk_mult=0.25)
    print_res("BEAR_RISK_0.25x", res)

    print(f"\n══════ Phase 2: Equity-curve risk management ══════")
    sigs, dfs, info = collect_v7_signals(syms, regime, tag)
    for mode in ["constant", "anti_martingale", "conservative_dd"]:
        res = simulate_v7(sigs, dfs, regime, base_risk=0.04, equity_curve_mode=mode,
                            anti_high_risk=0.05, anti_low_risk=0.02)
        print_res(f"EC={mode}", res)

    # Anti-martingale variants
    for hi, lo in [(0.06, 0.02), (0.05, 0.025), (0.04, 0.015)]:
        res = simulate_v7(sigs, dfs, regime, base_risk=0.04, equity_curve_mode="anti_martingale",
                            anti_high_risk=hi, anti_low_risk=lo)
        print_res(f"ANTI_MART_{hi*100:.1f}/{lo*100:.1f}", res)

    print(f"\n══════ Phase 3: New strategies STANDALONE on 730d ══════")
    for strat_kw in [{"include_kc": True, "include_smacross": False,
                       "include_supertrend": False, "include_rsi_rev": False,
                       "include_fib": False},
                      {"include_donchian50": True, "include_smacross": False,
                       "include_supertrend": False, "include_rsi_rev": False,
                       "include_fib": False},
                      {"include_1h_pullback": True, "include_smacross": False,
                       "include_supertrend": False, "include_rsi_rev": False,
                       "include_fib": False}]:
        sigs, dfs, info = collect_v7_signals(syms, regime, tag, **strat_kw)
        res = simulate_v7(sigs, dfs, regime, base_risk=0.04)
        name = "STANDALONE_" + ("KC4h" if strat_kw.get("include_kc") else
                                  "DONCHIAN50_1d" if strat_kw.get("include_donchian50") else
                                  "1H_PULLBACK")
        print_res(name, res)

    print(f"\n══════ Phase 4: V3 + each new strategy ══════")
    for add_strat in [{"include_kc": True}, {"include_donchian50": True},
                       {"include_1h_pullback": True}]:
        sigs, dfs, info = collect_v7_signals(syms, regime, tag, **add_strat)
        res = simulate_v7(sigs, dfs, regime, base_risk=0.04)
        name = "V3+" + ("KC4h" if add_strat.get("include_kc") else
                          "DONCH50" if add_strat.get("include_donchian50") else
                          "1H_PB")
        print_res(name, res)
        print_per_strat(res)

    # Best combinations
    print(f"\n══════ Phase 5: Combine best ideas ══════")
    sigs, dfs, info = collect_v7_signals(syms, regime, tag,
                                            include_kc=True,  # if it passed standalone
                                            bear_mode="long_only")
    res = simulate_v7(sigs, dfs, regime, base_risk=0.04, equity_curve_mode="anti_martingale",
                        anti_high_risk=0.05, anti_low_risk=0.02)
    print_res("V7_BEAR_LONG+EC_AM+KC", res)
    print_per_strat(res)


if __name__ == "__main__":
    main()
