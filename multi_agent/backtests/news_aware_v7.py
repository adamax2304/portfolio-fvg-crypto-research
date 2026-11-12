"""
news_aware_v7.py — Layered news/macro/sentiment integration on V7 baseline
==========================================================================
Tests incrementally:
  L4: Macro calendar pause (FOMC pre/post)
  L3: News circuit breaker (hack/war post-event pause)
  L3+: Positive event boost (ETF/listing → LONG bias)
  L1: Sentiment filter (BTC funding + RSI)

NO news trading — only context-aware risk/pause adjustments.

Sources:
  • Events: news_trader/events/dataset.py + dataset_2025.py (65 events)
  • Macro calendar: extracted FOMC dates from events
  • Sentiment: BTC funding (downloaded) + BTC RSI(14) on 1d
"""
from __future__ import annotations
import sys
import ast
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, rsi, regime_at,
    PortfolioSignal, _ts_period_ms,
    START_BALANCE, TAKER_FEE, SLIPPAGE, FUNDING_PER_8H,
    SL_COOLDOWN_MS, MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN,
    _monthly_returns, metrics_dict_from_trades,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals

CACHE_DIR = Path(__file__).parent / "data_cache"
NEWS_TRADER_EVENTS_DIR = Path("C:/Users/adama/Documents/zeph ai projects/news_trader/news_trader/events")


# ─────────────────────────────────────────────────────────────────────────────
# EVENT EXTRACTION (AST parsing, no imports needed)
# ─────────────────────────────────────────────────────────────────────────────

def parse_events_from_file(filepath: Path) -> list[dict]:
    if not filepath.exists():
        return []
    with open(filepath, encoding="utf-8") as f:
        source = f.read()
    tree = ast.parse(source)
    events = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and
                 isinstance(node.func, ast.Name) and node.func.id == "Event"):
            continue
        args = node.args
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        if len(args) < 6: continue
        try:
            name = args[0].value if isinstance(args[0], ast.Constant) else "?"
            symbol = args[1].value if isinstance(args[1], ast.Constant) else "?"
            direction = args[2].attr if isinstance(args[2], ast.Attribute) else "?"
            ts_call = args[3]
            if not (isinstance(ts_call, ast.Call) and isinstance(ts_call.func, ast.Name)
                    and ts_call.func.id == "_utc"):
                continue
            ts_args = []
            for a in ts_call.args:
                if isinstance(a, ast.Constant):
                    ts_args.append(a.value)
                else:
                    ts_args = []; break
            if len(ts_args) < 3: continue
            while len(ts_args) < 5:
                ts_args.append(0)
            ts = int(datetime(*ts_args, tzinfo=timezone.utc).timestamp() * 1000)
            category = args[4].attr if isinstance(args[4], ast.Attribute) else "?"
            tier = "FIXED"
            if len(args) > 5 and isinstance(args[5], ast.Attribute):
                tier = args[5].attr
            # Check for skip_in_backtest
            skip = False
            for kw in node.keywords:
                if kw.arg == "skip_in_backtest" and isinstance(kw.value, ast.Constant):
                    skip = kw.value.value
            events.append({
                "name": name, "symbol": symbol, "direction": direction,
                "ts": ts, "category": category, "tier": tier, "skip": skip
            })
        except Exception:
            continue
    return events


def load_all_events():
    events = []
    events.extend(parse_events_from_file(NEWS_TRADER_EVENTS_DIR / "dataset.py"))
    events.extend(parse_events_from_file(NEWS_TRADER_EVENTS_DIR / "dataset_2025.py"))
    # Filter out skipped events
    events = [e for e in events if not e.get("skip", False)]
    events.sort(key=lambda e: e["ts"])
    return events


# ─────────────────────────────────────────────────────────────────────────────
# CONTEXT CLASSIFIER — categorize each timestamp's context
# ─────────────────────────────────────────────────────────────────────────────

# Negative event categories (post-event pause/long-only mode)
NEGATIVE_CATEGORIES = {"CRYPTO_HACK", "GEO_WAR"}
# Positive event categories (post-event LONG boost)
POSITIVE_CATEGORIES = {"CRYPTO_ETF", "CRYPTO_LISTING", "OPEC_CUT"}
# Macro categories (scheduled pause)
MACRO_CATEGORIES = {"MACRO_FOMC", "MACRO_CPI", "MACRO_NFP"}


def get_event_context(now_ms: int, events: list[dict],
                       neg_window_h=24, pos_window_h=48,
                       pre_macro_h=24, post_macro_h=24):
    """
    Returns dict: {context, action, risk_mult, reason}
    For slow systems (1d/4h), windows must be longer than typical news flash:
      "pre_macro"      — 24h before FOMC → PAUSE
      "post_macro"     — 24h after FOMC → 50% risk (FOMC continuation effect)
      "post_negative"  — 24h after hack/war → PAUSE all (avoid bounce/continuation noise)
      "post_positive"  — 48h after ETF/listing → boost LONG, skip SHORT
    """
    # Check upcoming macro events
    for ev in events:
        if ev["category"] not in MACRO_CATEGORIES: continue
        time_to = ev["ts"] - now_ms
        if 0 < time_to < pre_macro_h * 3_600_000 and ev["tier"] in ("STRONG", "MEDIUM"):
            return {"context": "pre_macro", "action": "PAUSE",
                     "risk_mult": 0.0, "reason": f"pre_{ev['category']}"}
    # Check recent events
    for ev in reversed(events):
        if ev["ts"] > now_ms: continue
        time_since = now_ms - ev["ts"]
        if time_since > 72 * 3_600_000: break
        # Negative: full pause (not just shorts)
        if ev["category"] in NEGATIVE_CATEGORIES and ev["tier"] == "STRONG":
            if time_since < neg_window_h * 3_600_000:
                return {"context": "post_negative", "action": "PAUSE",
                         "risk_mult": 0.0, "reason": f"post_{ev['category']}_{ev['name'][:30]}"}
        # Positive: LONG boost
        if ev["category"] in POSITIVE_CATEGORIES and ev["tier"] == "STRONG":
            if ev.get("direction") == "LONG" and time_since < pos_window_h * 3_600_000:
                return {"context": "post_positive", "action": "LONG_BOOST",
                         "risk_mult": 1.5, "reason": f"post_{ev['category']}_{ev['name'][:30]}"}
        # Macro post-event
        if ev["category"] in MACRO_CATEGORIES and time_since < post_macro_h * 3_600_000:
            return {"context": "post_macro", "action": "REDUCED",
                     "risk_mult": 0.5, "reason": f"post_{ev['category']}"}
    return {"context": "normal", "action": "NORMAL", "risk_mult": 1.0, "reason": "normal"}


# ─────────────────────────────────────────────────────────────────────────────
# SENTIMENT FILTER (L1) — BTC funding + RSI 1d
# ─────────────────────────────────────────────────────────────────────────────

def build_sentiment_series():
    """Build daily sentiment series from BTC RSI(1d) and avg funding (27 syms)."""
    btc_1d = load("BTC", "1d", "730d")
    if btc_1d is None: return None
    cl = btc_1d["close"].to_numpy()
    ts = btc_1d["timestamp"].to_numpy()
    btc_rsi = rsi(cl, 14)

    # Aggregate funding rates from 27 syms (we have these)
    avg_funding_by_ts = {}
    for sym in BASELINE:
        p = CACHE_DIR / f"{sym}_USDT_USDT_funding_730d.parquet"
        if not p.exists(): continue
        df = pd.read_parquet(p)
        for _, row in df.iterrows():
            t = int(row["timestamp"])
            avg_funding_by_ts.setdefault(t, []).append(float(row["funding_rate"]))
    funding_ts = sorted(avg_funding_by_ts.keys())
    funding_avg = np.array([np.mean(avg_funding_by_ts[t]) for t in funding_ts])

    return {"btc_ts": ts, "btc_rsi": btc_rsi,
            "funding_ts": np.array(funding_ts), "funding_avg": funding_avg}


def get_sentiment_signal(sentiment, now_ms):
    """Returns risk_mult based on funding + RSI."""
    score = 0
    # BTC RSI on 1d (use prior day's close)
    idx = int(np.searchsorted(sentiment["btc_ts"], now_ms, side="right")) - 2
    if idx >= 14:
        rsi_v = sentiment["btc_rsi"][idx]
        if not np.isnan(rsi_v):
            if rsi_v > 75:    score -= 1   # extreme overbought → caution
            elif rsi_v < 25:  score += 1   # extreme oversold → contrarian bull
    # Avg funding
    f_idx = int(np.searchsorted(sentiment["funding_ts"], now_ms, side="right")) - 1
    if f_idx >= 0 and f_idx < len(sentiment["funding_avg"]):
        # Smooth: avg of last 3 funding periods (24h)
        lo = max(0, f_idx - 2)
        f_avg = float(np.mean(sentiment["funding_avg"][lo:f_idx+1]))
        if f_avg > 0.0003:    score -= 1   # crowded long → squeeze risk
        elif f_avg < -0.0001: score += 1   # crowd short → squeeze potential
    if score >= 2:    return 1.3, "smart_money_bullish"
    elif score <= -2: return 0.6, "smart_money_bearish"
    return 1.0, "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM SIMULATOR with news + sentiment context
# ─────────────────────────────────────────────────────────────────────────────

def simulate_news_aware(sigs, sim_dfs, regime, events, sentiment,
                          base_risk=0.04, max_pos=5,
                          enable_l4=False, enable_l3=False,
                          enable_l3_positive=False, enable_l1=False):
    """V7 simulator with news/sentiment context filters."""
    balance = START_BALANCE
    open_pos = {}; sym_cool = {}
    trades = []
    skipped = defaultdict(int)
    boosted = 0; reduced = 0

    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos: continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts: continue

        # Realize closed
        for k in list(open_pos.keys()):
            pos = open_pos[k]
            if pos["exit_ts"] is not None and pos["exit_ts"] <= sig.next_open_ts:
                balance = max(0.01, balance + pos["pnl"])
                trades.append(pos)
                if pos["reason"] == "SL":
                    sym_cool[pos["sym"]] = pos["exit_ts"] + SL_COOLDOWN_MS
                del open_pos[k]
        if len(open_pos) >= max_pos: continue

        # ─── Apply news context filter ───
        ctx = get_event_context(sig.next_open_ts, events) if (enable_l3 or enable_l4) else \
              {"context": "normal", "action": "NORMAL", "risk_mult": 1.0}
        risk_mult = 1.0
        if enable_l4 and ctx["context"] in ("pre_macro", "post_macro"):
            if ctx["action"] == "PAUSE":
                skipped["pre_macro"] += 1
                continue
            risk_mult *= ctx["risk_mult"]  # 0.5 for post_macro
        if enable_l3 and ctx["context"] == "post_negative":
            # FULL PAUSE — both directions (avoid bounce noise + continuation)
            skipped["post_negative_full_pause"] += 1
            continue
        if enable_l3_positive and ctx["context"] == "post_positive":
            # LONG boost, SHORTs skipped
            if sig.direction == "SHORT":
                skipped["post_positive_short"] += 1
                continue
            risk_mult *= 1.5
            boosted += 1
        # ─── Apply sentiment filter ───
        if enable_l1 and sentiment is not None:
            s_mult, s_reason = get_sentiment_signal(sentiment, sig.next_open_ts)
            risk_mult *= s_mult

        # Cap risk_mult to safety [0.2, 2.0]
        risk_mult = max(0.2, min(2.0, risk_mult))
        risk = base_risk * risk_mult

        # ─── Open trade (V7 sim with leverage check) ───
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
        notional = entry_p * qty
        max_margin = balance * MAX_MARGIN_FRAC
        if max_margin > 0:
            lev = notional / max_margin
            if lev > MAX_LEVERAGE:
                skipped["over_leverage"] += 1
                continue
            if notional / max(lev, 1) < MIN_MARGIN: continue

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
            "win": pnl > 0,
            "context": ctx["context"], "risk_mult": risk_mult,
        }

    # Realize remaining
    for k in list(open_pos.keys()):
        pos = open_pos[k]
        if pos["exit_ts"] is not None:
            balance = max(0.01, balance + pos["pnl"])
            trades.append(pos)

    trades.sort(key=lambda x: x["exit_ts"])
    eq = START_BALANCE; peak = START_BALANCE; max_dd = 0
    eq_curve = [(0, START_BALANCE)]
    for t in trades:
        eq = max(0.01, eq + t["pnl"])
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        eq_curve.append((t["exit_ts"], eq))
    monthly = _monthly_returns(eq_curve, START_BALANCE)
    if not trades:
        return {"trades": [], "metrics": {"n": 0}, "monthly": [], "skipped": dict(skipped),
                "boosted": 0, "reduced": 0}
    wins = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins); gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else 99
    wr = len(wins) / len(trades) * 100
    avg_r = float(np.mean([t["pnl_r"] for t in trades]))
    return_pct = (eq - START_BALANCE) / START_BALANCE * 100
    return {"trades": trades, "monthly": monthly,
            "skipped": dict(skipped), "boosted": boosted, "reduced": reduced,
            "metrics": {"n": len(trades), "wr": round(wr, 1), "pf": round(pf, 2),
                         "avg_r": round(avg_r, 3), "return_pct": round(return_pct, 1),
                         "max_dd_pct": round(max_dd, 1), "final_balance": round(eq, 2)}}


def print_res(name, res):
    m = res["metrics"]
    rets = [r["ret_pct"] for r in res["monthly"]]
    avgmo = float(np.mean(rets)) if rets else 0
    medmo = float(np.median(rets)) if rets else 0
    winmo = sum(1 for x in rets if x > 0); totmo = len(rets)
    print(f"  {name:30s} n={m['n']:4d} PF={m['pf']:5.2f} WR={m['wr']:5.1f}% "
          f"avgR={m['avg_r']:+5.2f} Ret={m['return_pct']:+8.1f}% DD={m['max_dd_pct']:>5.1f}% "
          f"Avg/mo={avgmo:>+5.1f}% Med={medmo:>+5.1f}% Win={winmo}/{totmo}")
    if res.get("skipped"):
        skips = ", ".join(f"{k}={v}" for k, v in res["skipped"].items() if v > 0)
        if skips: print(f"      → skipped: {skips}")
    if res.get("boosted"): print(f"      → boosted: {res['boosted']}, reduced: {res['reduced']}")


def main():
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = [s for s in BASELINE if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  NEWS-AWARE V7 BACKTEST (730d, {len(syms)} symbols)")
    print(f"══════════════════════════════════════════════════════════════════")

    # Load events
    print(f"\nLoading events from news_trader/events/...")
    events = load_all_events()
    print(f"  Total events parsed: {len(events)}")
    cat_counts = defaultdict(int)
    for e in events: cat_counts[e["category"]] += 1
    print(f"  Categories: {dict(cat_counts)}")
    # Filter events to our 730d window
    btc_ts_min = int(btc["timestamp"].iloc[0])
    btc_ts_max = int(btc["timestamp"].iloc[-1])
    events_in_window = [e for e in events if btc_ts_min <= e["ts"] <= btc_ts_max]
    print(f"  Events in 730d window: {len(events_in_window)}")

    # Build sentiment series
    print(f"\nBuilding sentiment series (BTC RSI + funding)...")
    sentiment = build_sentiment_series()
    print(f"  BTC RSI bars: {len(sentiment['btc_rsi'])}, funding records: {len(sentiment['funding_avg'])}")

    # Generate base V7 signals
    print(f"\nGenerating V7 signals (BEAR=long_only)...")
    sigs, sim_dfs, info = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    print(f"  Total V7 signals: {len(sigs)}")

    print(f"\n══════ INCREMENTAL LAYERS ══════")

    # 0. V7 baseline (no news)
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment, base_risk=0.04)
    print_res("V7 BASELINE (no news)", res)

    # 1. V7 + L4 (macro pause only)
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                base_risk=0.04, enable_l4=True)
    print_res("V7 + L4 (macro pause)", res)

    # 2. V7 + L3 (negative news breaker only)
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                base_risk=0.04, enable_l3=True)
    print_res("V7 + L3 (neg news only)", res)

    # 3. V7 + L4 + L3
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                base_risk=0.04, enable_l4=True, enable_l3=True)
    print_res("V7 + L4 + L3", res)

    # 4. + L3 positive (boost on ETF/listing)
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                base_risk=0.04, enable_l4=True, enable_l3=True,
                                enable_l3_positive=True)
    print_res("V7 + L4 + L3 + positive", res)

    # 5. ALL: + L1 sentiment
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                base_risk=0.04, enable_l4=True, enable_l3=True,
                                enable_l3_positive=True, enable_l1=True)
    print_res("V7 + ALL LAYERS", res)

    # Compare per-strategy in best config
    print(f"\n══════ Per-strategy breakdown (V7 + ALL LAYERS) ══════")
    res_all = res
    per = defaultdict(list)
    for t in res_all["trades"]: per[t["strat"]].append(t)
    for sn, tlist in sorted(per.items()):
        wins = [t for t in tlist if t["pnl"] > 0]
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in tlist if t["pnl"] <= 0))
        pf = gw / gl if gl > 0 else 99
        wr = len(wins) / max(1, len(tlist)) * 100
        sumr = sum(t["pnl_r"] for t in tlist)
        print(f"  {sn:25s}: n={len(tlist):4d} WR={wr:5.1f}% PF={pf:5.2f} sumR={sumr:+.1f}")

    # Try L1 only (without L3/L4)
    print(f"\n══════ L1 (sentiment) ALONE ══════")
    res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                base_risk=0.04, enable_l1=True)
    print_res("V7 + L1 only", res)

    # Pre-FOMC window sensitivity test (KEY for robustness)
    print(f"\n══════ Sensitivity: PRE-MACRO window length (L4 only) ══════")
    import multi_agent.backtests.news_aware_v7 as mod
    orig_fn = mod.get_event_context
    for pre_h in [4, 8, 12, 24, 36, 48, 72]:
        def custom_fn(now_ms, evs, ph=pre_h):
            return orig_fn(now_ms, evs, pre_macro_h=ph)
        mod.get_event_context = custom_fn
        res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                    base_risk=0.04, enable_l4=True)
        print_res(f"L4_pre_FOMC_{pre_h}h", res)
    mod.get_event_context = orig_fn

    # Risk grid for ALL LAYERS
    print(f"\n══════ ALL LAYERS at different risk levels ══════")
    for risk in [0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
        res = simulate_news_aware(sigs, sim_dfs, regime, events_in_window, sentiment,
                                    base_risk=risk, enable_l4=True, enable_l3=True,
                                    enable_l3_positive=True, enable_l1=True)
        print_res(f"ALL_LAYERS_R{int(risk*1000)/10:.1f}pct", res)


if __name__ == "__main__":
    main()
