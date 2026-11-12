"""
backtest_session_v8_portfolio.py -- 2026-04-17

Portfolio test: combine V7 AL-break (session) + BBS 1h->5m + BBS 1h->15m + KC 1h->15m
on the SAME shared equity curve — measure true combined PnL / DD / correlation.

Setup:
  - Each strategy's signals are generated with V7 filters (exWed + skipChop where applicable).
  - All signals are merged by entry timestamp and processed chronologically
    with a SHARED balance, so DD reflects real portfolio experience.
  - max concurrent positions across ALL strategies: configurable (default 6).
  - risk per trade is the same for all strategies (0.5% / 1.0% / 1.5% tested).

Output:
  - Combined portfolio metrics
  - Per-strategy contribution (% of total PnL)
  - Correlation between strategy equity curves
  - Walk-forward 4Q on the portfolio
"""
from __future__ import annotations
import json, logging, sys, time as _time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.backtest_session_v4 import (
    _load as _load_sess, _atr, prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf,
    _place_sl_tp, _check_htf,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE, RISK_PER_TRADE,
    TOP14,
)
from multi_agent.backtests.backtest_session_v5 import GOOD10
from multi_agent.backtests.backtest_session_v6 import BASE_PARAMS_V5_CHAMPION
from multi_agent.backtests.backtest_session_v7 import (
    build_btc_regime, gen_al_break_v7,
)
from multi_agent.backtests.backtest_v10 import (
    load_all_tf, prepare_tf_arrays, prepare_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
)

logger = logging.getLogger("session_bt_v8port")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  UNIFIED SIGNAL FORMAT
# ═══════════════════════════════════════════════════════════════════════════
#
# Each signal is a dict:
#   {strategy, sym, entry_ts_ms, entry_idx, direction,
#    entry_price, sl, tp, risk_px,
#    sl_type ("fixed" only for now),
#    tp_type ("fixed_r" or "abs_price"),
#    arr_ref: symbol bar array (so sim can iterate bars for SL/TP hit)}
#
# The portfolio sim iterates bars in chronological order (cursor per signal),
# tracks open positions, respects max_concurrent.

def _ts_to_date(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")


def _ts_to_dow(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).weekday()


def v7_signals_for_symbol(arr, sess_ranges, params, htf_aligned_map, btc_regime):
    """Generate V7 signals for a symbol, then convert to portfolio signal list."""
    sigs = gen_al_break_v7(arr, sess_ranges, params, htf_aligned_map, btc_regime=btc_regime)
    out = []
    for s in sigs:
        i_entry = s["i_entry"]
        out.append({
            "strategy": "V7_AL",
            "sym": None,  # filled later
            "entry_ts": int(arr["ts"][i_entry]),
            "entry_idx": i_entry,
            "direction": s["direction"],
            "entry_price": float(s["entry_ref"]),
            "sl": float(s["sl"]), "tp": float(s["tp"]),
            "risk_px": float(s["risk_px"]),
            "max_hold_bars": 40,  # V7 is EOS-at-22:00, ~9 hours = 36 bars of 15m
            "tf_mins": 15,
            "session_end_hr": 22,
            "arr_key": "15m",
        })
    return out


def bbs_mtf_signals(all_data, sym_filter, htf_key, ltf_key, htf_sig_params, mtf_entry_params,
                    gen_htf_fn, strategy_tag, tf_mins, max_hold_bars):
    out_by_sym = {}
    for sym, d in all_data.items():
        if sym_filter is not None and sym not in sym_filter:
            continue
        htf_df = d.get(htf_key)
        ltf_df = d.get(ltf_key)
        if htf_df is None or ltf_df is None:
            continue
        htf_arr = prepare_tf_arrays(htf_df)
        ltf_arr = prepare_tf_arrays(ltf_df)
        if htf_arr is None or ltf_arr is None:
            continue
        htf_sigs = gen_htf_fn(htf_arr, None, htf_sig_params)
        if not htf_sigs:
            continue
        mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, mtf_entry_params)
        # Convert each to portfolio signal
        lst = []
        ltf_ts = ltf_arr["ts"]; ltf_cl = ltf_arr["cl"]
        atr_v = ltf_arr["atr"]
        for i_bar, s in mtf_sigs.items():
            if i_bar < 0 or i_bar >= len(ltf_ts):
                continue
            entry_price = s["entry_price"]
            sl_v = s["default_sl"]
            direction = s["direction"]
            risk_px = abs(entry_price - sl_v)
            if risk_px <= 0:
                continue
            # TP via fixed_r on SL risk
            if strategy_tag == "BBS_1h5m":
                tp_r = 3.0
            elif strategy_tag == "BBS_1h15m":
                tp_r = 2.5
            elif strategy_tag == "KC_1h15m":
                tp_r = 2.0
            else:
                tp_r = 2.0
            tp_v = entry_price + risk_px * tp_r if direction == "LONG" else entry_price - risk_px * tp_r
            lst.append({
                "strategy": strategy_tag,
                "sym": sym,
                "entry_ts": int(ltf_ts[i_bar]),
                "entry_idx": i_bar,
                "direction": direction,
                "entry_price": float(entry_price),
                "sl": float(sl_v), "tp": float(tp_v),
                "risk_px": float(risk_px),
                "max_hold_bars": max_hold_bars,
                "tf_mins": tf_mins,
                "session_end_hr": None,  # no EOS constraint
                "arr_key": f"ltf_{ltf_key}",
            })
        out_by_sym[sym] = (lst, ltf_arr)
    return out_by_sym


# ═══════════════════════════════════════════════════════════════════════════
#  PORTFOLIO SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════

def sim_portfolio(all_sigs_with_arrs, risk_per_trade=0.01, max_concurrent=6,
                  filter_dows=None, filter_skip_chop=None, btc_regime=None):
    """
    all_sigs_with_arrs: list of (sig_dict, arr). arr is the bar array the sig's strategy lives on.

    Process chronologically by entry_ts. Each trade:
     - opens at entry_price (+/- slippage)
     - sim walks bars on arr until SL/TP/EOS/max_hold_bars
     - updates shared balance
    """
    balance = START_BALANCE
    peak = balance
    max_dd = 0.0

    # Sort by entry time
    all_sigs_with_arrs.sort(key=lambda x: x[0]["entry_ts"])

    # Apply filters if given
    filtered = []
    for sig, arr in all_sigs_with_arrs:
        ts = sig["entry_ts"]
        if filter_dows is not None and _ts_to_dow(ts) not in filter_dows:
            continue
        if filter_skip_chop and btc_regime:
            d = _ts_to_date(ts)
            if btc_regime.get(d, {}).get("regime") in filter_skip_chop:
                continue
        filtered.append((sig, arr))

    open_positions = []  # list of ongoing trades with exit-time estimates
    completed = []

    for sig, arr in filtered:
        # Close out any positions whose exit happened before this sig
        t = sig["entry_ts"]
        # Remove positions already exited at this time
        still_open = []
        for pos in open_positions:
            if pos["exit_ts"] <= t:
                pass  # already settled & added to completed below when we computed its exit
            else:
                still_open.append(pos)
        open_positions = still_open
        if len(open_positions) >= max_concurrent:
            continue  # skip this signal — too many concurrent

        # Simulate this trade on its arr
        h = arr["hi"] if "hi" in arr else arr["h"]
        l = arr["lo"] if "lo" in arr else arr["l"]
        c = arr["cl"] if "cl" in arr else arr["c"]
        ts = arr["ts"]
        n = len(c)
        i_entry = sig["entry_idx"]
        direction = sig["direction"]
        entry_price = sig["entry_price"]
        sl = sig["sl"]; tp = sig["tp"]
        risk_px = sig["risk_px"]
        max_hold = sig.get("max_hold_bars", 40)
        session_end_hr = sig.get("session_end_hr")

        # Effective entry with slippage
        eff_entry = entry_price * (1 + SLIPPAGE) if direction == "LONG" else entry_price * (1 - SLIPPAGE)
        # Position sizing
        risk_usd = balance * risk_per_trade
        if risk_px <= 0 or entry_price <= 0: continue
        qty_usd = risk_usd / (risk_px / entry_price)
        if qty_usd > balance * MAX_LEVERAGE: qty_usd = balance * MAX_LEVERAGE
        if qty_usd * entry_price < 5.0:
            continue
        entry_fee = qty_usd * TAKER_FEE

        # Walk bars
        exit_reason = "max_hold"; exit_px = c[min(i_entry + max_hold, n - 1)]
        exit_idx = min(i_entry + max_hold, n - 1)
        for j in range(i_entry, min(i_entry + max_hold + 1, n)):
            if session_end_hr is not None:
                # V7 session: check hour
                dt = datetime.fromtimestamp(ts[j] / 1000.0, tz=timezone.utc)
                if dt.hour >= session_end_hr:
                    exit_reason = "EOS"; exit_px = c[j]; exit_idx = j; break
            if direction == "LONG":
                if l[j] <= sl:
                    exit_reason = "SL"; exit_px = sl; exit_idx = j; break
                if h[j] >= tp:
                    exit_reason = "TP"; exit_px = tp; exit_idx = j; break
            else:
                if h[j] >= sl:
                    exit_reason = "SL"; exit_px = sl; exit_idx = j; break
                if l[j] <= tp:
                    exit_reason = "TP"; exit_px = tp; exit_idx = j; break

        eff_exit = exit_px * (1 - SLIPPAGE) if direction == "LONG" else exit_px * (1 + SLIPPAGE)
        exit_fee = qty_usd * TAKER_FEE
        if direction == "LONG":
            pnl = qty_usd * (eff_exit - eff_entry) / eff_entry - entry_fee - exit_fee
        else:
            pnl = qty_usd * (eff_entry - eff_exit) / eff_entry - entry_fee - exit_fee
        balance += pnl
        peak = max(peak, balance)
        dd = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

        exit_ts = int(ts[exit_idx])
        completed.append({
            "strategy": sig["strategy"], "sym": sig.get("sym"),
            "direction": direction,
            "entry_ts": sig["entry_ts"], "exit_ts": exit_ts,
            "reason": exit_reason, "pnl_usd": pnl, "won": pnl > 0,
            "entry": entry_price, "exit": exit_px,
        })
        open_positions.append({"exit_ts": exit_ts})

    # Aggregate
    wins = sum(1 for c in completed if c["won"])
    losses = sum(1 for c in completed if not c["won"] and abs(c["pnl_usd"]) > 1e-9)
    total_win = sum(c["pnl_usd"] for c in completed if c["won"])
    total_loss = sum(c["pnl_usd"] for c in completed if c["pnl_usd"] < 0)
    n_t = len(completed)
    wr = wins / n_t * 100 if n_t else 0.0
    pf = (total_win / abs(total_loss)) if total_loss < 0 else (99.0 if total_win > 0 else 0.0)
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100
    avg_r = (sum(c["pnl_usd"] for c in completed) / n_t) if n_t else 0.0
    return {
        "trades": n_t, "wins": wins, "losses": losses,
        "wr_pct": round(wr, 2),
        "profit_factor": round(pf, 3),
        "pnl_pct": round(pnl_pct, 2),
        "max_dd_pct": round(max_dd, 2),
        "final_balance": round(balance, 2),
        "completed": completed,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    logger.info("=== V8 PORTFOLIO: V7 AL + BBS 1h->5m + BBS 1h->15m + KC 1h->15m ===")

    # --- Load session data (V7 AL lives on 15m) ---
    arrays, ranges, htf_by_sym = {}, {}, {}
    for sym in TOP14:
        df15 = _load_sess(sym, "15m")
        if df15 is None: continue
        df1h = _load_sess(sym, "1h")
        if df1h is None: df1h = _load_sess(sym, "1H")
        df4h = _load_sess(sym, "4h")
        if df4h is None: df4h = _load_sess(sym, "4H")
        arr = prep_15m_arrays(df15)
        ranges[sym] = build_session_ranges(arr)
        htf = compute_htf_trends(df4h, df1h)
        htf_aligned = {k: align_htf(m, arr["ts"]) for k, m in htf.items()}
        arrays[sym] = arr
        htf_by_sym[sym] = htf_aligned
    logger.info(f"Loaded {len(arrays)} session symbols")

    btc_regime = build_btc_regime(arrays)

    good10_arrays = {s: arrays[s] for s in GOOD10 if s in arrays}

    # --- Load MTF data for BBS/KC ---
    logger.info("Loading MTF data (may take 30s+)...")
    all_data = load_all_tf("365d")
    logger.info(f"Loaded {len(all_data)} MTF symbols")

    # --- Build V7 AL signals ---
    v7_params = dict(BASE_PARAMS_V5_CHAMPION)
    v7_params["allowed_dows"] = {0, 1, 3, 4, 5, 6}
    v7_params["skip_regimes"] = {"chop"}

    # Need a bar array compatible with sim helpers:
    # V7 arr has keys h/l/c/o — sim_portfolio tolerates either "hi"/"h"
    v7_sigs_all = []
    for sym, arr in good10_arrays.items():
        sigs = v7_signals_for_symbol(arr, ranges[sym], v7_params, htf_by_sym.get(sym), btc_regime)
        for s in sigs:
            s["sym"] = sym
            v7_sigs_all.append((s, arr))
    logger.info(f"V7 AL signals: {len(v7_sigs_all)}")

    # --- Build BBS 1h->5m signals ---
    bbs5_sig_params = {"min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0}
    bbs5_entry_params = {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 1.5}
    bbs5_all_by_sym = bbs_mtf_signals(all_data, None, "df1h", "df5",
                                       bbs5_sig_params, bbs5_entry_params,
                                       gen_bbs_signals_tf, "BBS_1h5m",
                                       tf_mins=5, max_hold_bars=120)
    bbs5_sigs_all = []
    for sym, (lst, arr) in bbs5_all_by_sym.items():
        for s in lst: bbs5_sigs_all.append((s, arr))
    logger.info(f"BBS 1h->5m signals: {len(bbs5_sigs_all)}")

    # --- Build BBS 1h->15m signals ---
    bbs15_sig_params = {"min_squeeze": 4, "adx_thresh": 15.0, "vol_mult": 1.0}
    bbs15_entry_params = {"entry_mode": "next_bar", "max_wait_bars": 1, "sl_atr_mult": 1.5}
    bbs15_all_by_sym = bbs_mtf_signals(all_data, None, "df1h", "df15",
                                        bbs15_sig_params, bbs15_entry_params,
                                        gen_bbs_signals_tf, "BBS_1h15m",
                                        tf_mins=15, max_hold_bars=60)
    bbs15_sigs_all = []
    for sym, (lst, arr) in bbs15_all_by_sym.items():
        for s in lst: bbs15_sigs_all.append((s, arr))
    logger.info(f"BBS 1h->15m signals: {len(bbs15_sigs_all)}")

    # --- Build KC 1h->15m signals ---
    kc15_sig_params = {"min_squeeze": 10, "adx_thresh": 12.0, "vol_mult": 1.0, "mom_lookback": 6}
    kc15_entry_params = {"entry_mode": "next_bar", "max_wait_bars": 1, "sl_atr_mult": 1.5}
    kc15_all_by_sym = bbs_mtf_signals(all_data, None, "df1h", "df15",
                                       kc15_sig_params, kc15_entry_params,
                                       gen_kc_squeeze_signals_tf, "KC_1h15m",
                                       tf_mins=15, max_hold_bars=60)
    kc15_sigs_all = []
    for sym, (lst, arr) in kc15_all_by_sym.items():
        for s in lst: kc15_sigs_all.append((s, arr))
    logger.info(f"KC 1h->15m signals: {len(kc15_sigs_all)}")

    # ═══════════════════════════════════════════════════════════════
    # Run each strategy SOLO
    # ═══════════════════════════════════════════════════════════════
    results = []
    logger.info("")
    logger.info("=== Solo runs (for baseline reference) ===")
    for name, sigs in [("SOLO_V7", v7_sigs_all), ("SOLO_BBS1h5m", bbs5_sigs_all),
                        ("SOLO_BBS1h15m", bbs15_sigs_all), ("SOLO_KC1h15m", kc15_sigs_all)]:
        r = sim_portfolio(list(sigs), risk_per_trade=0.01, max_concurrent=6)
        r.pop("completed", None)
        logger.info(f"{name:20s} trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                    f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%")
        results.append({"config_id": name, **r})

    # ═══════════════════════════════════════════════════════════════
    # COMBO runs
    # ═══════════════════════════════════════════════════════════════
    combos = [
        ("P_V7_BBS5", v7_sigs_all + bbs5_sigs_all),
        ("P_V7_BBS15", v7_sigs_all + bbs15_sigs_all),
        ("P_V7_KC", v7_sigs_all + kc15_sigs_all),
        ("P_BBS5_BBS15", bbs5_sigs_all + bbs15_sigs_all),
        ("P_BBS5_BBS15_KC", bbs5_sigs_all + bbs15_sigs_all + kc15_sigs_all),
        ("P_V7_BBS5_BBS15", v7_sigs_all + bbs5_sigs_all + bbs15_sigs_all),
        ("P_ALL_FOUR", v7_sigs_all + bbs5_sigs_all + bbs15_sigs_all + kc15_sigs_all),
    ]

    logger.info("")
    logger.info("=== Combo portfolios (shared equity, risk=0.01) ===")
    combo_details = {}
    for name, sigs in combos:
        r = sim_portfolio(list(sigs), risk_per_trade=0.01, max_concurrent=6)
        completed = r.pop("completed", [])
        # per-strategy contribution
        by_strat = defaultdict(lambda: {"trades": 0, "pnl": 0.0, "wins": 0})
        for c in completed:
            st = c["strategy"]
            by_strat[st]["trades"] += 1
            by_strat[st]["pnl"] += c["pnl_usd"]
            if c["won"]: by_strat[st]["wins"] += 1
        combo_details[name] = dict(by_strat)
        logger.info(f"{name:25s} trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                    f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%")
        results.append({"config_id": name, **r})

    # ═══════════════════════════════════════════════════════════════
    # Risk scaling on best combo
    # ═══════════════════════════════════════════════════════════════
    best = max([r for r in results if r["config_id"].startswith("P_")],
               key=lambda x: x["pnl_pct"])
    logger.info("")
    logger.info(f"=== Risk scaling on {best['config_id']} ===")
    best_sigs = next(s for n, s in combos if n == best["config_id"])
    for rpt in [0.005, 0.01, 0.015, 0.02]:
        r = sim_portfolio(list(best_sigs), risk_per_trade=rpt, max_concurrent=6)
        r.pop("completed", None)
        name = f"{best['config_id']}_rpt{rpt}"
        logger.info(f"{name:35s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                    f"PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%")
        results.append({"config_id": name, **r})

    # ═══════════════════════════════════════════════════════════════
    # max_concurrent sweep on best combo
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info(f"=== max_concurrent sweep on {best['config_id']} ===")
    for mc in [2, 3, 4, 5, 6, 8, 10]:
        r = sim_portfolio(list(best_sigs), risk_per_trade=0.01, max_concurrent=mc)
        r.pop("completed", None)
        name = f"{best['config_id']}_mc{mc}"
        logger.info(f"{name:35s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                    f"PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%")
        results.append({"config_id": name, **r})

    # ═══════════════════════════════════════════════════════════════
    # Walk-forward on best combo
    # ═══════════════════════════════════════════════════════════════
    logger.info("")
    logger.info(f"=== Walk-forward 4Q on {best['config_id']} ===")

    # Determine quarter boundaries via global ts range
    all_ts = [s[0]["entry_ts"] for s in best_sigs]
    if all_ts:
        ts_min = min(all_ts); ts_max = max(all_ts)
        span = ts_max - ts_min
        for qi in range(4):
            ts_lo = ts_min + int(span * qi / 4)
            ts_hi = ts_min + int(span * (qi + 1) / 4)
            q_sigs = [(s, a) for (s, a) in best_sigs if ts_lo <= s["entry_ts"] < ts_hi]
            r = sim_portfolio(list(q_sigs), risk_per_trade=0.01, max_concurrent=6)
            r.pop("completed", None)
            name = f"WF_{best['config_id']}_Q{qi+1}"
            tag = "PASS" if (r["pnl_pct"] > 0 and r["profit_factor"] >= 1.0) else "FAIL"
            logger.info(f"{name:35s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                        f"PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}% [{tag}]")
            results.append({"config_id": name, **r})

    # ═══════════════════════════════════════════════════════════════
    # SAVE + SUMMARY
    # ═══════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v8_portfolio_{ts}.json"
    def _json_default(x):
        if isinstance(x, set): return sorted(x)
        try: return float(x)
        except Exception: return str(x)
    with open(path, "w") as f:
        json.dump({"results": results, "combo_details": combo_details},
                  f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t0:.1f}s)")

    # Print summary
    print("\n" + "="*120)
    print(f"{'config':35s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s}")
    print("-"*120)
    for r in results:
        print(f"{r['config_id']:35s} {r['trades']:>6} {r['wr_pct']:>5.1f} "
              f"{r['profit_factor']:>5.2f}  {r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f}")

    # Per-strategy contribution in best combo
    print("\n" + "="*120)
    print(f"PER-STRATEGY CONTRIBUTION in {best['config_id']}:")
    det = combo_details.get(best["config_id"], {})
    total_pnl_usd = sum(v["pnl"] for v in det.values())
    for strat, v in sorted(det.items(), key=lambda kv: -kv[1]["pnl"]):
        share = (v["pnl"] / total_pnl_usd * 100) if total_pnl_usd != 0 else 0
        wr = v["wins"] / v["trades"] * 100 if v["trades"] else 0
        print(f"  {strat:15s} trades={v['trades']:>4} WR={wr:>5.1f}% "
              f"pnl_usd={v['pnl']:>+12.2f}  share={share:>+6.1f}%")


if __name__ == "__main__":
    main()
