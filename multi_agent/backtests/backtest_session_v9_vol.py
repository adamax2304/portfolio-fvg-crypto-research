"""
backtest_session_v9_vol.py -- 2026-04-17

Research: VOLATILITY-ADAPTIVE FILTER
-----------------------------------
Hypothesis: one-size-fits-all SL (ATR x 1.5) + fixed R-TP is sub-optimal across
coins with wildly different ATR% regimes. Tier-based SL/TP scaling should:
  - give low-vol coins (TRX, XLM, ALGO) a wider ATR-mult so SL isn't eaten by noise
  - give extreme-vol coins (PEPE, PENGU) tighter ATR-mult + wider R-target

This is PURE RESEARCH. Nothing is applied to production.

What this script does:
  1. Loads 365d OHLCV (1h / 15m / 5m) for ALL symbols in data_cache that pass
     the anomaly filter (vol, ATR, liquidity).
  2. Classifies each symbol into 1 of 4 tiers using ATR% median computed on the
     FIRST 20% of data (pre-sample; no lookahead).
  3. Runs the 3 production MTF strategies (BBS 1h->5m pullback, BBS 1h->15m next,
     KC 1h->15m next) under two regimes:
         (a) BASELINE v10 params (uniform SL=ATR*1.5, TP=3.0R -> like prod)
         (b) TIERED params (tier-specific SL mult / R / floor-cap / slippage)
  4. Shared-deposit portfolio sim: 1% risk/trade, max 4 concurrent positions.
  5. Reports per-tier PF/PnL/DD/trades, per-strategy, overall, and per-symbol.

Honesty protocol:
  - Tier slippage estimates are deliberately pessimistic for Tier 3/4.
  - Fee = 0.055% maker + slippage per tier (Tier1/2=0.05%, Tier3=0.10%, Tier4=0.20%).
  - No look-ahead: tier classification uses only pre-sample period.
  - Report separates TEST period (post-sample 80%) from TRAIN.
  - If a tier has <10 trades we flag it as "insufficient sample".
"""
from __future__ import annotations

import json
import logging
import sys
import time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.backtest_v10 import (
    _load, prepare_tf_arrays, resample_to_30m,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf,
    gen_mtf_signals,
)
from multi_agent.shared.regime_detector import (
    precompute_1h_indicators, precompute_4h_indicators,
)

logger = logging.getLogger("bt_v9_vol")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

# ─── Portfolio / sim constants ──────────────────────────────────────────────
START_BALANCE = 1000.0
TAKER_FEE = 0.00055
RISK_PER_TRADE = 0.010          # 1% risk per trade (user-requested)
CAP_FRAC = 1.00                 # use full balance (production model)
MAX_POS = 4                     # production: 4 concurrent positions
MAX_LEVERAGE = 50

# Pre-sample window for tier classification (no lookahead)
PRESAMPLE_FRAC = 0.20

# ─── Tier definitions ──────────────────────────────────────────────────────
# ATR% = ATR14 / close, computed on 1h bars. Median over pre-sample period.
TIERS = [
    {
        "name": "T1_LOW",       "atr_pct_max": 0.35,
        "sl_atr_mult": 2.5,     "sl_floor_pct": 0.005,  "sl_cap_pct": 0.025,
        "tp_r": 2.0,            "slippage": 0.0005,
        "min_squeeze_add": 0,   "adx_add": 0,
    },
    {
        "name": "T2_MID",       "atr_pct_max": 0.75,
        "sl_atr_mult": 1.5,     "sl_floor_pct": 0.008,  "sl_cap_pct": 0.035,
        "tp_r": 2.5,            "slippage": 0.0005,
        "min_squeeze_add": 0,   "adx_add": 0,
    },
    {
        "name": "T3_HIGH",      "atr_pct_max": 1.50,
        "sl_atr_mult": 1.3,     "sl_floor_pct": 0.010,  "sl_cap_pct": 0.040,
        "tp_r": 3.0,            "slippage": 0.0010,
        "min_squeeze_add": 0,   "adx_add": 0,
    },
    {
        "name": "T4_EXTREME",   "atr_pct_max": 99.0,
        "sl_atr_mult": 1.1,     "sl_floor_pct": 0.015,  "sl_cap_pct": 0.060,
        "tp_r": 3.5,            "slippage": 0.0020,
        "min_squeeze_add": 2,   "adx_add": 3,
    },
]

# Baseline (what prod uses today for all coins, approximately)
BASELINE_TIER = {
    "name": "BASELINE",
    "sl_atr_mult": 1.5, "sl_floor_pct": 0.0, "sl_cap_pct": 0.05,
    "tp_r": 3.0, "slippage": 0.0005,
    "min_squeeze_add": 0, "adx_add": 0,
}

# ─── Anomaly filter thresholds ─────────────────────────────────────────────
ANOM_ATR_PCT_MAX = 3.0          # exclude if median ATR% > 3% (unhinged vol)
ANOM_MIN_BARS_1H = 400          # need ~17 days of 1h data minimum
ANOM_MIN_VOLUME_USD_PER_1H = 100_000   # exclude if avg 1h USD volume < $100k
ANOM_VOL_SKEW_MAX = 5.0         # reject if max/min period ATR% ratio > 5x (regime hell)


# ═══════════════════════════════════════════════════════════════════════════
#  1. LOAD + CLASSIFY UNIVERSE
# ═══════════════════════════════════════════════════════════════════════════

def discover_all_symbols():
    """All symbols that have 1h_365d cached."""
    syms = []
    for p in CACHE_DIR.glob("*_1h_365d.parquet"):
        stem = p.stem.replace("_1h_365d", "")  # e.g. BTC_USDT_USDT
        parts = stem.split("_")
        if len(parts) >= 3:
            ccy = parts[0]
            sym = f"{ccy}/USDT:USDT"
            syms.append(sym)
    return sorted(set(syms))


def load_symbol(sym, days_tag="365d"):
    """Load 1h / 15m / 5m data. Returns dict or None if 1h missing."""
    d = {
        "df1h": _load(sym, "1h", days_tag),
        "df15": _load(sym, "15m", days_tag),
        "df5":  _load(sym, "5m",  days_tag),
    }
    if d["df1h"] is None or len(d["df1h"]) < ANOM_MIN_BARS_1H:
        return None
    return d


def classify_symbol(d, sym, verbose=False):
    """Compute ATR%, volume, skew on pre-sample. Returns tier dict + diagnostics,
    or None if symbol fails anomaly filters."""
    df1h = d["df1h"]
    n = len(df1h)
    pre_end = int(n * PRESAMPLE_FRAC)
    if pre_end < 200:
        return None, {"reason": "presample_too_short"}
    pre = df1h.iloc[:pre_end]
    hi = pre["high"].to_numpy(float)
    lo = pre["low"].to_numpy(float)
    cl = pre["close"].to_numpy(float)
    vo = pre["volume"].to_numpy(float)

    # ATR14
    tr = np.zeros(len(cl))
    tr[0] = hi[0] - lo[0]
    for i in range(1, len(cl)):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    atr = np.full(len(cl), np.nan)
    atr[13] = np.mean(tr[:14])
    for i in range(14, len(cl)):
        atr[i] = (atr[i-1] * 13 + tr[i]) / 14
    atr_pct = atr / cl * 100.0   # percent
    atr_pct_median = float(np.nanmedian(atr_pct[14:]))

    # Volume USD per 1h
    vol_usd = vo * cl
    vol_usd_mean = float(np.nanmean(vol_usd))

    # Period-skew: compare ATR% median of first half vs second half of pre-sample
    h = len(cl) // 2
    am1 = float(np.nanmedian(atr_pct[14:h])) if h > 20 else atr_pct_median
    am2 = float(np.nanmedian(atr_pct[h:]))
    skew = max(am1, am2) / max(0.0001, min(am1, am2))

    diag = {
        "sym": sym, "n_1h": n,
        "atr_pct_median": round(atr_pct_median, 3),
        "vol_usd_mean": round(vol_usd_mean, 0),
        "skew": round(skew, 2),
    }

    # Anomaly filters
    if atr_pct_median > ANOM_ATR_PCT_MAX:
        return None, {**diag, "reason": "atr_too_high"}
    if vol_usd_mean < ANOM_MIN_VOLUME_USD_PER_1H:
        return None, {**diag, "reason": "illiquid"}
    if skew > ANOM_VOL_SKEW_MAX:
        return None, {**diag, "reason": "regime_shift"}

    # Pick tier
    for tier in TIERS:
        if atr_pct_median <= tier["atr_pct_max"]:
            return tier, diag
    return TIERS[-1], diag


def load_universe():
    """Load all symbols, classify, report."""
    all_data = {}
    tier_of = {}
    diagnostics = []
    rejected = []
    print("\n--- Loading universe (365d) ---", flush=True)
    for sym in discover_all_symbols():
        d = load_symbol(sym)
        if d is None:
            rejected.append({"sym": sym, "reason": "no_1h_data"})
            continue
        tier, diag = classify_symbol(d, sym)
        if tier is None:
            rejected.append(diag)
            continue
        all_data[sym] = d
        tier_of[sym] = tier
        diag["tier"] = tier["name"]
        diagnostics.append(diag)

    # Tier counts
    by_tier = defaultdict(list)
    for sym, tier in tier_of.items():
        by_tier[tier["name"]].append(sym)
    print(f"  Total loaded: {len(all_data)}  rejected: {len(rejected)}")
    for t in TIERS:
        nm = t["name"]
        syms = by_tier.get(nm, [])
        print(f"    {nm:<12} n={len(syms):2d}  atr_pct_max={t['atr_pct_max']:<5}  "
              f"symbols: {', '.join(s.split('/')[0] for s in syms[:10])}"
              + (" ..." if len(syms) > 10 else ""))
    if rejected:
        print(f"  Rejected ({len(rejected)}):")
        for r in rejected[:8]:
            print(f"    {r}")
    return all_data, tier_of, diagnostics, rejected


# ═══════════════════════════════════════════════════════════════════════════
#  2. SIM ENGINE (tier-aware; shared deposit; 1% risk; 4 max pos)
# ═══════════════════════════════════════════════════════════════════════════

def _apply_sl_caps(sl_price, entry, direction, floor_pct, cap_pct):
    """Enforce min/max SL distance as % of entry price."""
    sl_dist = abs(entry - sl_price)
    min_d = entry * floor_pct
    max_d = entry * cap_pct
    if sl_dist < min_d:
        sl_dist = min_d
    elif sl_dist > max_d:
        sl_dist = max_d
    if direction == "LONG":
        return entry - sl_dist
    return entry + sl_dist


def run_sim_tiered(signals_by_sym, tf_arrays_by_sym, tier_of,
                   use_baseline=False,
                   start_frac=0.20, end_frac=1.0):
    """
    signals_by_sym: dict sym -> {bar_idx -> {"direction", "entry_price", "default_sl"}}
    tf_arrays_by_sym: dict sym -> prepared LTF arrays
    tier_of: dict sym -> tier dict (or None if symbol dropped)
    use_baseline: if True, use BASELINE_TIER for ALL symbols.
    Starts at PRESAMPLE boundary by default to avoid look-ahead in tier classification.
    """
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    open_pos = {}
    acnt = 0
    completed = []

    max_n = 0
    for sym, arr in tf_arrays_by_sym.items():
        if arr is not None:
            max_n = max(max_n, arr["n"])
    if max_n == 0:
        return _empty()

    i_start = max(300, int(max_n * start_frac))
    i_end = int(max_n * end_frac)

    for i in range(i_start, i_end):
        for sym, arr in tf_arrays_by_sym.items():
            if arr is None or i >= arr["n"]:
                continue
            tier = BASELINE_TIER if use_baseline else tier_of.get(sym)
            if tier is None:
                continue

            cl_i = arr["cl"][i]; hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            # ─── manage open position ───
            if sym in open_pos:
                pos = open_pos[sym]
                d = pos["d"]; entry = pos["entry"]
                sl = pos["sl"]; r = pos["risk"]; q = pos["qty"]
                tp_price = pos["tp"]
                bars = i - pos["bar_i"]

                if d == "LONG":
                    hit_sl = lo_i <= sl
                    hit_tp = hi_i >= tp_price
                else:
                    hit_sl = hi_i >= sl
                    hit_tp = lo_i <= tp_price

                # Conservative: if both hit in same bar, assume SL first
                if hit_sl:
                    ep = sl; won = False
                elif hit_tp:
                    ep = tp_price; won = True
                else:
                    ep = None

                if ep is not None:
                    qo = q
                    gross = ((ep - entry) if d == "LONG" else (entry - ep)) * qo
                    fee = ep * qo * (TAKER_FEE + pos["slip"])
                    net = gross - fee
                    balance += net
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    completed.append({
                        "sym": sym, "d": d, "tier": pos["tier"],
                        "exit_r": round(exit_r, 3), "bars": bars,
                        "won": won, "pnl_usd": round(net, 4),
                        "reason": "tp" if won else "sl",
                    })
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            # ─── open new position ───
            if acnt >= MAX_POS:
                continue
            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig = sym_sigs[i]
            entry = sig["entry_price"]
            d = sig["direction"]

            # SL per tier
            sl_raw = entry - atr_i * tier["sl_atr_mult"] if d == "LONG" \
                else entry + atr_i * tier["sl_atr_mult"]
            sl = _apply_sl_caps(sl_raw, entry, d,
                                tier["sl_floor_pct"], tier["sl_cap_pct"])
            rd = abs(entry - sl)
            if rd <= 0 or rd / entry > 0.08:
                continue

            # TP at tier R
            tp = entry + rd * tier["tp_r"] if d == "LONG" else entry - rd * tier["tp_r"]

            ab = balance * CAP_FRAC
            ru = ab * RISK_PER_TRADE
            qty = ru / rd
            mg = entry * qty / MAX_LEVERAGE
            if mg > ab * 0.40:
                qty = ab * 0.40 * MAX_LEVERAGE / entry
            if qty * entry < 5.0:
                continue

            ef = entry * qty * (TAKER_FEE + tier["slippage"])
            balance -= ef
            open_pos[sym] = {
                "d": d, "entry": entry, "sl": sl, "tp": tp, "risk": rd,
                "qty": qty, "bar_i": i, "tier": tier["name"],
                "slip": tier["slippage"],
            }
            acnt += 1

    return _aggregate(completed, balance)


def _empty():
    return {"trades": 0, "wins": 0, "losses": 0, "wr_pct": 0, "pnl_pct": 0,
            "profit_factor": 0, "max_dd_pct": 0, "final_bal": START_BALANCE,
            "by_tier": {}}


def _aggregate(completed, balance):
    if not completed:
        return _empty()
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    n = len(completed)
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / n * 100
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100

    # DD
    bal = START_BALANCE; peak = START_BALANCE; max_dd = 0.0
    for t in completed:
        bal += t["pnl_usd"]; peak = max(peak, bal)
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    # Per-tier breakdown
    by_tier = defaultdict(list)
    for t in completed:
        by_tier[t["tier"]].append(t)
    tier_stats = {}
    for name, lst in by_tier.items():
        w = sum(1 for x in lst if x["won"])
        gp_t = sum(x["pnl_usd"] for x in lst if x["won"])
        gl_t = abs(sum(x["pnl_usd"] for x in lst if not x["won"]))
        pf_t = gp_t / gl_t if gl_t > 0 else (99.0 if gp_t > 0 else 0.0)
        net_t = sum(x["pnl_usd"] for x in lst)
        tier_stats[name] = {
            "trades": len(lst), "wins": w, "wr_pct": round(w/len(lst)*100, 1),
            "pf": round(min(pf_t, 99), 2), "pnl_usd": round(net_t, 2),
            "pnl_pct": round(net_t / START_BALANCE * 100, 2),
        }

    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 1), "pnl_pct": round(pnl_pct, 2),
        "profit_factor": round(min(pf, 99), 3), "max_dd_pct": round(max_dd, 2),
        "final_bal": round(balance, 2),
        "avg_bars": round(float(np.mean([t["bars"] for t in completed])), 1),
        "by_tier": tier_stats,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  3. SIGNAL BUILDERS (adapted from v10)
# ═══════════════════════════════════════════════════════════════════════════

def build_signals_for_strategy(all_data, tier_of, strategy, tiered_params=False):
    """Build MTF signals for all symbols.
    strategy: 'bbs_1h_5m' | 'bbs_1h_15m' | 'kc_1h_15m'
    tiered_params: if True, bump min_squeeze/adx by tier for T4 (higher bar).
    """
    all_sigs = {}
    ltf_arrays = {}

    if strategy == "bbs_1h_5m":
        htf_key, ltf_key = "df1h", "df5"
        gen_fn = gen_bbs_signals_tf
        base_sig = {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                    "htf_filter": False, "headfake_filter": True}
        entry_p = {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5}
    elif strategy == "bbs_1h_15m":
        htf_key, ltf_key = "df1h", "df15"
        gen_fn = gen_bbs_signals_tf
        base_sig = {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                    "htf_filter": False, "headfake_filter": True}
        entry_p = {"entry_mode": "next_bar", "max_wait_bars": 8, "sl_atr_mult": 2.5}
    elif strategy == "kc_1h_15m":
        htf_key, ltf_key = "df1h", "df15"
        gen_fn = gen_kc_squeeze_signals_tf
        base_sig = {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                    "mom_bars": 6, "sl_atr_mult": 2.0}
        entry_p = {"entry_mode": "next_bar", "sl_atr_mult": 2.5}
    else:
        raise ValueError(strategy)

    for sym, d in all_data.items():
        tier = tier_of.get(sym)
        if tier is None:
            continue
        # Per-symbol sig params (tier can raise bar for T4)
        sig_p = dict(base_sig)
        if tiered_params:
            if "min_squeeze" in sig_p:
                sig_p["min_squeeze"] = base_sig["min_squeeze"] + tier.get("min_squeeze_add", 0)
            if "adx_thresh" in sig_p:
                sig_p["adx_thresh"] = base_sig["adx_thresh"] + tier.get("adx_add", 0)
            if "adx_min" in sig_p:
                sig_p["adx_min"] = base_sig["adx_min"] + tier.get("adx_add", 0)

        htf_arr = prepare_tf_arrays(d.get(htf_key))
        ltf_arr = prepare_tf_arrays(d.get(ltf_key))
        if htf_arr is None or ltf_arr is None:
            continue
        ltf_arrays[sym] = ltf_arr
        # HTF-trend context for headfake/htf_filter: use 4h if available (we skip it here)
        htf_sigs = gen_fn(htf_arr, None, sig_p)
        mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, entry_p)
        all_sigs[sym] = mtf_sigs

    return all_sigs, ltf_arrays


def combine_signals(sigs_a, sigs_b, sigs_c, ltf_a, ltf_b, ltf_c):
    """Merge 3 strategies into a single signal stream per symbol.
    Since 1h->5m uses 5m LTF and 1h->15m uses 15m LTF, we can't trivially
    share one LTF per symbol. We simulate each strategy separately in its own
    timeframe space, then PER-TRADE we treat them as independent strategies
    competing for the same deposit in real time via a sequential merge.

    Simpler approach (used here): run 3 separate sims on a shared balance by
    interleaving trades chronologically. Accomplished by running each sim,
    collecting trade-event list with entry timestamps, then a second pass
    simulates the portfolio bar-by-bar in time."""
    raise NotImplementedError("portfolio merger lives in run_portfolio()")


# ═══════════════════════════════════════════════════════════════════════════
#  4. TRADE-LEVEL PORTFOLIO SIM (time-sorted, shared deposit, all 3 strategies)
# ═══════════════════════════════════════════════════════════════════════════

def collect_trade_events(all_sigs, tf_arrays, tier_of, strategy_name,
                         use_baseline=False):
    """For each signal, produce a trade-event dict with entry/exit timestamps,
    R, SL, TP, tier. Does NOT apply deposit constraint -- just per-signal outcomes.
    Later merged chronologically across strategies + symbols with capital gate.
    Trades use next-bar SL/TP scan, hit-whichever-first (SL wins ties).
    """
    events = []
    for sym, sigs in all_sigs.items():
        arr = tf_arrays.get(sym)
        tier = BASELINE_TIER if use_baseline else tier_of.get(sym)
        if arr is None or tier is None or not sigs:
            continue
        cl = arr["cl"]; hi = arr["hi"]; lo = arr["lo"]
        atr = arr["atr"]; ts = arr["ts"]; n = arr["n"]

        for i, sig in sigs.items():
            if i >= n - 1:
                continue
            atr_i = atr[i] if not np.isnan(atr[i]) else cl[i] * 0.01
            entry = sig["entry_price"]; d = sig["direction"]

            sl_raw = entry - atr_i * tier["sl_atr_mult"] if d == "LONG" \
                else entry + atr_i * tier["sl_atr_mult"]
            sl = _apply_sl_caps(sl_raw, entry, d,
                                tier["sl_floor_pct"], tier["sl_cap_pct"])
            rd = abs(entry - sl)
            if rd <= 0 or rd / entry > 0.08:
                continue
            tp = entry + rd * tier["tp_r"] if d == "LONG" else entry - rd * tier["tp_r"]

            # Scan forward for SL/TP hit
            exit_i = None; exit_price = None; won = None
            max_bars = 200  # cap holding time
            for j in range(i + 1, min(i + 1 + max_bars, n)):
                hj = hi[j]; lj = lo[j]
                if d == "LONG":
                    if lj <= sl:
                        exit_i = j; exit_price = sl; won = False; break
                    if hj >= tp:
                        exit_i = j; exit_price = tp; won = True; break
                else:
                    if hj >= sl:
                        exit_i = j; exit_price = sl; won = False; break
                    if lj <= tp:
                        exit_i = j; exit_price = tp; won = True; break
            if exit_i is None:
                # Force close at last bar
                exit_i = min(i + max_bars, n - 1)
                exit_price = cl[exit_i]
                won = (d == "LONG" and exit_price > entry) or (d == "SHORT" and exit_price < entry)

            events.append({
                "strategy": strategy_name,
                "sym": sym, "tier": tier["name"],
                "entry_ts": int(ts[i]), "exit_ts": int(ts[exit_i]),
                "entry_i": i, "exit_i": exit_i,
                "d": d, "entry": float(entry), "sl": float(sl), "tp": float(tp),
                "exit_price": float(exit_price), "won": bool(won),
                "risk_price_dist": float(rd),
                "slippage": tier["slippage"],
                "risk_pct": rd / entry,
            })
    return events


def run_portfolio(all_events):
    """Time-sorted portfolio sim with shared balance, 1% risk, max 4 open."""
    events = sorted(all_events, key=lambda e: e["entry_ts"])
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    open_pos = []    # list of dicts with exit_ts
    completed = []

    # Two-queue model: open at entry_ts; close at exit_ts
    # To handle in chronological order of events (entries + exits), build a
    # combined event queue.
    pq = []
    for idx, e in enumerate(events):
        pq.append(("entry", e["entry_ts"], idx))
    # We'll build exit events dynamically when opening.

    pq.sort()
    open_by_idx = {}

    # Merge entries and exits chronologically
    # For this we pre-queue exits once we open them.
    timeline = []
    for idx, e in enumerate(events):
        timeline.append((e["entry_ts"], 0, idx, "entry"))
    timeline.sort()

    # We'll process entries in order and when an entry fires, schedule its exit.
    # At each step also process any exits due before/equal to current ts.
    exit_heap = []  # list of (exit_ts, idx)
    import heapq

    result_per_strategy_tier = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    result_per_tier = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})
    result_per_strategy = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0})

    def close_position(idx, now_ts):
        nonlocal balance, peak, max_dd
        pos = open_by_idx.pop(idx)
        e = pos["event"]
        qty = pos["qty"]
        entry = e["entry"]; exit_p = e["exit_price"]; d = e["d"]
        gross = (exit_p - entry) * qty if d == "LONG" else (entry - exit_p) * qty
        fee = exit_p * qty * (TAKER_FEE + e["slippage"])
        net = gross - fee
        balance += net
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
        completed.append({
            "sym": e["sym"], "strategy": e["strategy"], "tier": e["tier"],
            "entry_ts": e["entry_ts"], "exit_ts": e["exit_ts"], "d": d,
            "won": e["won"], "pnl_usd": net,
        })
        key_st = (e["strategy"], e["tier"])
        result_per_strategy_tier[key_st]["pnl"] += net
        result_per_strategy_tier[key_st]["wins"] += 1 if e["won"] else 0
        result_per_strategy_tier[key_st]["losses"] += 0 if e["won"] else 1
        result_per_tier[e["tier"]]["pnl"] += net
        result_per_tier[e["tier"]]["wins"] += 1 if e["won"] else 0
        result_per_tier[e["tier"]]["losses"] += 0 if e["won"] else 1
        result_per_strategy[e["strategy"]]["pnl"] += net
        result_per_strategy[e["strategy"]]["wins"] += 1 if e["won"] else 0
        result_per_strategy[e["strategy"]]["losses"] += 0 if e["won"] else 1

    for ts_now, _typ, idx, kind in timeline:
        # Close any due exits first
        while exit_heap and exit_heap[0][0] <= ts_now:
            exit_ts, eid = heapq.heappop(exit_heap)
            if eid in open_by_idx:
                close_position(eid, exit_ts)

        e = events[idx]
        # Skip same-sym overlap (already open)
        if any(op["event"]["sym"] == e["sym"] for op in open_by_idx.values()):
            continue
        if len(open_by_idx) >= MAX_POS:
            continue

        # Size position
        risk_usd = balance * RISK_PER_TRADE
        qty = risk_usd / e["risk_price_dist"]
        margin = e["entry"] * qty / MAX_LEVERAGE
        if margin > balance * 0.40:
            qty = balance * 0.40 * MAX_LEVERAGE / e["entry"]
        if qty * e["entry"] < 5.0:
            continue
        entry_fee = e["entry"] * qty * (TAKER_FEE + e["slippage"])
        balance -= entry_fee
        open_by_idx[idx] = {"event": e, "qty": qty}
        heapq.heappush(exit_heap, (e["exit_ts"], idx))

    # Drain remaining exits
    while exit_heap:
        exit_ts, eid = heapq.heappop(exit_heap)
        if eid in open_by_idx:
            close_position(eid, exit_ts)

    # Aggregate
    n = len(completed)
    if n == 0:
        return {"trades": 0, "pnl_pct": 0, "profit_factor": 0, "max_dd_pct": 0,
                "wr_pct": 0, "final_bal": START_BALANCE,
                "by_strategy": {}, "by_tier": {}, "by_strategy_tier": {}}
    wins = sum(1 for t in completed if t["won"])
    gp = sum(t["pnl_usd"] for t in completed if t["won"])
    gl = abs(sum(t["pnl_usd"] for t in completed if not t["won"]))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)

    def fmt_group(g):
        return {k: {"trades": v["wins"]+v["losses"],
                    "wr_pct": round(v["wins"]/(v["wins"]+v["losses"])*100, 1) if (v["wins"]+v["losses"]) else 0,
                    "pnl_usd": round(v["pnl"], 2),
                    "pnl_pct": round(v["pnl"]/START_BALANCE*100, 2)}
                for k, v in g.items()}

    return {
        "trades": n,
        "wr_pct": round(wins/n*100, 1),
        "pnl_pct": round((balance-START_BALANCE)/START_BALANCE*100, 2),
        "profit_factor": round(min(pf, 99), 3),
        "max_dd_pct": round(max_dd, 2),
        "final_bal": round(balance, 2),
        "by_strategy": fmt_group(result_per_strategy),
        "by_tier": fmt_group(result_per_tier),
        "by_strategy_tier": {f"{k[0]}|{k[1]}": {"trades": v["wins"]+v["losses"],
            "wr_pct": round(v["wins"]/(v["wins"]+v["losses"])*100, 1) if (v["wins"]+v["losses"]) else 0,
            "pnl_usd": round(v["pnl"], 2),
            "pnl_pct": round(v["pnl"]/START_BALANCE*100, 2)}
            for k, v in result_per_strategy_tier.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════
#  5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"backtest_session_v9_vol.py  (vol-adaptive filter research)", flush=True)
    print(f"Start: {datetime.now().isoformat(timespec='seconds')}", flush=True)
    t0 = _time.time()

    # ── Load universe ──
    all_data, tier_of, diagnostics, rejected = load_universe()
    if not all_data:
        print("No symbols loaded. Aborting.")
        return

    strategies = ["bbs_1h_5m", "bbs_1h_15m", "kc_1h_15m"]

    # ── BASELINE: uniform SL=ATR*1.5, TP=3R ──
    print("\n" + "=" * 72)
    print("  BASELINE (uniform ATR*1.5 SL, 3R TP, slip=0.05% everywhere)")
    print("=" * 72, flush=True)
    all_events_base = []
    for strat in strategies:
        print(f"\n  Generating {strat}...", flush=True)
        sigs, ltf = build_signals_for_strategy(all_data, tier_of, strat, tiered_params=False)
        tot = sum(len(s) for s in sigs.values())
        print(f"    signals={tot}", flush=True)
        events = collect_trade_events(sigs, ltf, tier_of, strat, use_baseline=True)
        print(f"    trade-events={len(events)}", flush=True)
        all_events_base.extend(events)

    # Filter to test period (post-sample 80%)
    # Find cutoff timestamp = 20th percentile of all_events entry_ts? No --
    # better: use actual pre-sample boundary per symbol's 1h data
    # Simpler: cutoff = 20th percentile of all entry timestamps overall
    if all_events_base:
        cutoffs = sorted([e["entry_ts"] for e in all_events_base])
        cutoff_ts = cutoffs[int(len(cutoffs) * PRESAMPLE_FRAC)]
    else:
        cutoff_ts = 0

    events_base_test = [e for e in all_events_base if e["entry_ts"] >= cutoff_ts]
    print(f"\n  Filtering to test period (post {datetime.fromtimestamp(cutoff_ts/1000).date()}): "
          f"{len(events_base_test)} / {len(all_events_base)} events", flush=True)

    print("\n  Running baseline portfolio sim...", flush=True)
    res_base = run_portfolio(events_base_test)
    print_summary("BASELINE (test period, 1% risk, shared deposit, max 4 pos)", res_base)

    # ── TIERED: per-tier SL/TP + slippage ──
    print("\n" + "=" * 72)
    print("  TIERED (per-tier SL mult / R target / slippage / signal-bar)")
    print("=" * 72, flush=True)
    all_events_tier = []
    for strat in strategies:
        print(f"\n  Generating {strat}...", flush=True)
        sigs, ltf = build_signals_for_strategy(all_data, tier_of, strat, tiered_params=True)
        tot = sum(len(s) for s in sigs.values())
        print(f"    signals={tot}", flush=True)
        events = collect_trade_events(sigs, ltf, tier_of, strat, use_baseline=False)
        print(f"    trade-events={len(events)}", flush=True)
        all_events_tier.extend(events)

    events_tier_test = [e for e in all_events_tier if e["entry_ts"] >= cutoff_ts]
    print(f"\n  Filtering to test period: {len(events_tier_test)} / {len(all_events_tier)} events", flush=True)

    print("\n  Running tiered portfolio sim...", flush=True)
    res_tier = run_portfolio(events_tier_test)
    print_summary("TIERED (test period, 1% risk, shared deposit, max 4 pos)", res_tier)

    # ── Delta ──
    print("\n" + "=" * 72)
    print("  BASELINE vs TIERED  (delta)")
    print("=" * 72, flush=True)
    print(f"  Trades     {res_base['trades']:>6}  -> {res_tier['trades']:>6}")
    print(f"  PF         {res_base['profit_factor']:>6.2f}  -> {res_tier['profit_factor']:>6.2f}")
    print(f"  PnL %      {res_base['pnl_pct']:>+6.2f}  -> {res_tier['pnl_pct']:>+6.2f}")
    print(f"  MaxDD %    {res_base['max_dd_pct']:>6.2f}  -> {res_tier['max_dd_pct']:>6.2f}")
    print(f"  WinRate %  {res_base['wr_pct']:>6.1f}  -> {res_tier['wr_pct']:>6.1f}")

    # ── Save results ──
    out = {
        "meta": {
            "ts": ts, "risk_per_trade": RISK_PER_TRADE,
            "max_pos": MAX_POS, "start_balance": START_BALANCE,
            "presample_frac": PRESAMPLE_FRAC,
            "cutoff_ts": cutoff_ts,
            "universe_size": len(all_data),
            "rejected": rejected,
        },
        "diagnostics": diagnostics,
        "tier_of": {s: t["name"] for s, t in tier_of.items()},
        "tier_defs": TIERS,
        "baseline": res_base,
        "tiered": res_tier,
    }
    p = RESULTS_DIR / f"backtest_session_v9_vol_{ts}.json"
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {p}")
    print(f"  Duration: {(_time.time()-t0):.1f}s", flush=True)


def print_summary(title, res):
    print(f"\n  {title}")
    print(f"    trades={res['trades']}  wr={res['wr_pct']}%  "
          f"PF={res['profit_factor']}  PnL={res['pnl_pct']:+.2f}%  "
          f"DD={res['max_dd_pct']}%  final=${res['final_bal']}")
    if res.get("by_tier"):
        print("    by tier:")
        for name, s in sorted(res["by_tier"].items()):
            print(f"      {name:<12} n={s['trades']:>4}  wr={s['wr_pct']:>5}%  "
                  f"PnL=${s['pnl_usd']:>+8.2f} ({s['pnl_pct']:+.2f}%)")
    if res.get("by_strategy"):
        print("    by strategy:")
        for name, s in sorted(res["by_strategy"].items()):
            print(f"      {name:<15} n={s['trades']:>4}  wr={s['wr_pct']:>5}%  "
                  f"PnL=${s['pnl_usd']:>+8.2f} ({s['pnl_pct']:+.2f}%)")


if __name__ == "__main__":
    main()
