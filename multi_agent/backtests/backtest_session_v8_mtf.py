"""
backtest_session_v8_mtf.py -- 2026-04-17

Goal: validate existing BBS/KC MTF configs (from v10) on CURRENT 365d data,
then overlay V7 filters (DoW ex-Wed, skip BTC chop days) to see if they help.

Scope:
  - BBS 1h->5m pullback_ema21    (v10 champion, expected PF~9.66)
  - BBS 1h->15m next_bar         (v10 workhorse, expected PF~3.92)
  - KC  1h->15m next_bar         (v10 secondary, expected PF~5.76)

For each of the 3 baselines:
  + DoW ex-Wed overlay
  + skip BTC chop-day overlay
  + both overlays

V7 filters are applied ONLY as signal-pruning at entry time — the underlying
v10 signal logic / SL / TP are untouched. Results are directly comparable.

NO PRODUCTION CODE IS CHANGED. This is pure research.
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

from multi_agent.backtests.backtest_v10 import (
    load_all_tf, prepare_tf_arrays, prepare_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf,
    gen_mtf_signals, run_sim_tf,
)

logger = logging.getLogger("bt_v8_mtf")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  V7-style filters (date / dow / regime) — derived from ltf bar timestamps
# ═══════════════════════════════════════════════════════════════════════════

def _ts_to_date_dow(ts_ms):
    """Convert UTC ms timestamp to (date_str, dow) where dow 0=Mon..6=Sun."""
    dt = datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d"), dt.weekday()


def build_btc_daily_regime(all_data):
    """
    Build BTC daily regime by aggregating BTC 1h bars per date.
    Returns: date_str -> {"vol_pct": float, "regime": "chop"|"normal"|"high"|"panic"}
    """
    btc_key = "BTC/USDT:USDT"
    if btc_key not in all_data or all_data[btc_key].get("df1h") is None:
        return {}
    df1h = all_data[btc_key]["df1h"]
    ts = df1h["timestamp"].astype(int).to_numpy()
    hi = df1h["high"].astype(float).to_numpy()
    lo = df1h["low"].astype(float).to_numpy()
    cl = df1h["close"].astype(float).to_numpy()
    by_date = defaultdict(list)
    for i in range(len(ts)):
        d, _ = _ts_to_date_dow(int(ts[i]))
        by_date[d].append(i)
    daily = []
    for d, idxs in by_date.items():
        if not idxs:
            continue
        day_hi = max(hi[i] for i in idxs)
        day_lo = min(lo[i] for i in idxs)
        day_cl = cl[idxs[-1]]
        if day_cl <= 0:
            continue
        vol_pct = (day_hi - day_lo) / day_cl * 100
        daily.append((d, vol_pct))
    if not daily:
        return {}
    vols = np.array([v for _, v in daily])
    p25 = float(np.percentile(vols, 25))
    p75 = float(np.percentile(vols, 75))
    p90 = float(np.percentile(vols, 90))
    logger.info(f"BTC daily vol%: p25={p25:.2f} p75={p75:.2f} p90={p90:.2f}")
    regime = {}
    for d, v in daily:
        if v < p25:    r = "chop"
        elif v > p90:  r = "panic"
        elif v > p75:  r = "high"
        else:          r = "normal"
        regime[d] = {"vol_pct": v, "regime": r}
    return regime


def filter_sigs_v7(sigs_by_sym, ltf_arrays_by_sym, btc_regime=None,
                   allowed_dows=None, skip_regimes=None):
    """
    Prune v10 signal dicts using V7 filters.
    sigs_by_sym:  {sym: {ltf_bar_idx: sig_dict}}
    Returns new dict with offending signals removed.
    """
    if allowed_dows is None and not skip_regimes:
        return sigs_by_sym  # no-op fast path
    out = {}
    for sym, sigs in sigs_by_sym.items():
        arr = ltf_arrays_by_sym.get(sym)
        if arr is None:
            out[sym] = sigs
            continue
        ts = arr["ts"]
        kept = {}
        for i_bar, sig in sigs.items():
            if i_bar < 0 or i_bar >= len(ts):
                continue
            d, dow = _ts_to_date_dow(int(ts[i_bar]))
            if allowed_dows is not None and dow not in allowed_dows:
                continue
            if skip_regimes and btc_regime:
                reg = btc_regime.get(d, {}).get("regime")
                if reg in skip_regimes:
                    continue
            kept[i_bar] = sig
        out[sym] = kept
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  MTF build helpers (local copy — we don't import build_mtf_sigs_and_arrays
#  because we want to customize filter application)
# ═══════════════════════════════════════════════════════════════════════════

def build_mtf_sigs(all_data, htf_key, ltf_key, htf_tf,
                    gen_htf_fn, htf_sig_params, mtf_entry_params,
                    daily_filter=False):
    """Build HTF signals then MTF LTF entries. Returns (sigs_by_sym, ltf_arrays)."""
    sigs_by_sym = {}
    ltf_arrays = {}
    for sym, d in all_data.items():
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
            ltf_arrays[sym] = ltf_arr
            sigs_by_sym[sym] = {}
            continue
        daily = prepare_daily_trend(d.get("df1d")) if daily_filter else None
        mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, mtf_entry_params,
                                   daily_data=daily, use_daily_filter=daily_filter)
        sigs_by_sym[sym] = mtf_sigs
        ltf_arrays[sym] = ltf_arr
    return sigs_by_sym, ltf_arrays


# ═══════════════════════════════════════════════════════════════════════════
#  Runner
# ═══════════════════════════════════════════════════════════════════════════

def _aggregate(res, label, params):
    return {
        "config_id": label,
        "trades":        res.get("trades", 0),
        "wr_pct":        res.get("wr_pct", 0.0),
        "profit_factor": res.get("profit_factor", 0.0),
        "pnl_pct":       res.get("pnl_pct", 0.0),
        "max_dd_pct":    res.get("max_dd_pct", 0.0),
        "avg_r":         res.get("avg_r", 0.0),
        "params":        params,
    }


def log_r(name, r):
    logger.info(f"{name:55s} trades={r['trades']:>4} "
                f"WR={r['wr_pct']:>5.1f}% PF={r['profit_factor']:>5.2f} "
                f"PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%")


def main():
    t0 = _time.time()
    logger.info("=== V8 MTF: BBS/KC validation + V7 overlay ===")

    logger.info("Loading all TF data (this will take a bit)...")
    all_data = load_all_tf("365d")
    logger.info(f"Loaded {len(all_data)} symbols")

    btc_regime = build_btc_daily_regime(all_data)
    by_reg = defaultdict(int)
    for v in btc_regime.values():
        by_reg[v["regime"]] += 1
    logger.info(f"BTC regime counts: {dict(by_reg)}")

    # SL/TP settings from v10 memory
    sl_atr = {"type": "atr", "mult": 1.5}
    tp_bbs_5m = {"type": "fixed_r", "r": 3.0}
    tp_bbs_15 = {"type": "fixed_r", "r": 2.5}
    tp_kc_15 = {"type": "fixed_r", "r": 2.0}

    # v10-optimized signal params
    bbs_5m_htf = {"min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0}
    bbs_5m_entry = {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 1.5}

    bbs_15_htf = {"min_squeeze": 4, "adx_thresh": 15.0, "vol_mult": 1.0}
    bbs_15_entry = {"entry_mode": "next_bar", "max_wait_bars": 1, "sl_atr_mult": 1.5}

    kc_15_htf = {"min_squeeze": 10, "adx_thresh": 12.0, "vol_mult": 1.0,
                 "mom_lookback": 6}
    kc_15_entry = {"entry_mode": "next_bar", "max_wait_bars": 1, "sl_atr_mult": 1.5}

    # Build each strategy's MTF sigs once
    logger.info("Building BBS 1h->5m signals...")
    bbs5_sigs, bbs5_arr = build_mtf_sigs(all_data, "df1h", "df5", "1h",
                                          gen_bbs_signals_tf, bbs_5m_htf, bbs_5m_entry)
    logger.info(f"  BBS 1h->5m total sigs: {sum(len(s) for s in bbs5_sigs.values())}")

    logger.info("Building BBS 1h->15m signals...")
    bbs15_sigs, bbs15_arr = build_mtf_sigs(all_data, "df1h", "df15", "1h",
                                            gen_bbs_signals_tf, bbs_15_htf, bbs_15_entry)
    logger.info(f"  BBS 1h->15m total sigs: {sum(len(s) for s in bbs15_sigs.values())}")

    logger.info("Building KC 1h->15m signals...")
    kc15_sigs, kc15_arr = build_mtf_sigs(all_data, "df1h", "df15", "1h",
                                          gen_kc_squeeze_signals_tf, kc_15_htf, kc_15_entry)
    logger.info(f"  KC 1h->15m total sigs: {sum(len(s) for s in kc15_sigs.values())}")

    results = []

    # Filter variants
    EX_WED = {0, 1, 3, 4, 5, 6}
    SKIP_CHOP = {"chop"}
    SKIP_CHOP_PANIC = {"chop", "panic"}

    strategies = [
        ("BBS1h5m",  bbs5_sigs,  bbs5_arr,  sl_atr, tp_bbs_5m),
        ("BBS1h15m", bbs15_sigs, bbs15_arr, sl_atr, tp_bbs_15),
        ("KC1h15m",  kc15_sigs,  kc15_arr,  sl_atr, tp_kc_15),
    ]

    overlays = [
        ("base",          None,    None),
        ("exWed",         EX_WED,  None),
        ("skipChop",      None,    SKIP_CHOP),
        ("skipChopPan",   None,    SKIP_CHOP_PANIC),
        ("exWed+chop",    EX_WED,  SKIP_CHOP),
        ("exWed+chopPan", EX_WED,  SKIP_CHOP_PANIC),
    ]

    logger.info("")
    logger.info("=== Running all overlay variants ===")
    for strat_name, sigs, arr, sl_cfg, tp_cfg in strategies:
        for ov_name, dows, skip_reg in overlays:
            filt_sigs = filter_sigs_v7(sigs, arr, btc_regime=btc_regime,
                                        allowed_dows=dows, skip_regimes=skip_reg)
            n_sigs = sum(len(s) for s in filt_sigs.values())
            res = run_sim_tf(filt_sigs, arr, sl_cfg, tp_cfg,
                             risk_pt=0.010, cap_frac=0.60, max_pos=4)
            label = f"{strat_name}_{ov_name}"
            r = _aggregate(res, label, {"strat": strat_name, "overlay": ov_name,
                                          "dows": sorted(dows) if dows else None,
                                          "skip_regimes": sorted(skip_reg) if skip_reg else None,
                                          "total_sigs": n_sigs})
            log_r(label, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Walk-forward for each base + best overlay
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Walk-forward 4Q on each strategy's best overlay ===")

    def slice_sigs_arr(sigs_by_sym, arr_by_sym, start_frac, end_frac):
        """Return sigs/arrays restricted to a fractional bar-range."""
        new_sigs = {}
        new_arr = {}
        for sym, arr in arr_by_sym.items():
            n = arr["n"]
            i_start = int(n * start_frac)
            i_end = int(n * end_frac)
            # Slice sigs dict
            raw_sigs = sigs_by_sym.get(sym, {})
            kept = {i: s for i, s in raw_sigs.items() if i_start <= i < i_end}
            new_sigs[sym] = kept
            new_arr[sym] = arr  # arr is unchanged — sim uses i_start/i_end internally
        return new_sigs, new_arr

    # Pick best overlay per strategy
    best_per_strat = {}
    for strat_name, _, _, _, _ in strategies:
        cands = [r for r in results if r["config_id"].startswith(strat_name + "_")
                 and r["trades"] >= 25]
        if cands:
            best_per_strat[strat_name] = max(cands, key=lambda x: x["pnl_pct"])

    for strat_name, sigs, arr, sl_cfg, tp_cfg in strategies:
        b = best_per_strat.get(strat_name)
        if not b:
            continue
        ov_label = b["params"]["overlay"]
        dows = set(b["params"]["dows"]) if b["params"]["dows"] else None
        skip_reg = set(b["params"]["skip_regimes"]) if b["params"]["skip_regimes"] else None
        filt_sigs = filter_sigs_v7(sigs, arr, btc_regime=btc_regime,
                                    allowed_dows=dows, skip_regimes=skip_reg)
        logger.info(f"  WF on {strat_name}_{ov_label}:")
        for qi, (s, e) in enumerate([(0, .25), (.25, .5), (.5, .75), (.75, 1.0)]):
            res = run_sim_tf(filt_sigs, arr, sl_cfg, tp_cfg,
                             risk_pt=0.010, cap_frac=0.60, max_pos=4,
                             start_frac=s, end_frac=e)
            label = f"WF_{strat_name}_{ov_label}_Q{qi+1}"
            r = _aggregate(res, label, b["params"])
            log_r(label, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # SAVE
    # ═════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v8_mtf_{ts}.json"
    def _json_default(x):
        if isinstance(x, set): return sorted(x)
        try: return float(x)
        except Exception: return str(x)
    with open(path, "w") as f:
        json.dump({"results": results, "btc_regime_counts": dict(by_reg)},
                  f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time()-t0:.1f}s)")

    # ═════════════════════════════════════════════════════════════
    # SUMMARY
    # ═════════════════════════════════════════════════════════════
    print("\n" + "="*120)
    print(f"{'config':55s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s}")
    for strat_name, _, _, _, _ in strategies:
        print("-"*120)
        rs = [r for r in results if r["config_id"].startswith(strat_name + "_")]
        rs.sort(key=lambda r: r["pnl_pct"], reverse=True)
        for r in rs:
            print(f"{r['config_id']:55s} {r['trades']:>6} {r['wr_pct']:>5.1f} "
                  f"{r['profit_factor']:>5.2f}  {r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f}")

    # Improvement vs baseline
    print("\n" + "="*120)
    print("V7-FILTER IMPROVEMENT vs BASELINE (per strategy)")
    for strat_name, _, _, _, _ in strategies:
        base = next((r for r in results if r["config_id"] == strat_name + "_base"), None)
        if base is None:
            continue
        print(f"\n  {strat_name}  baseline: PF={base['profit_factor']:.2f} "
              f"PnL={base['pnl_pct']:+.1f}% DD={base['max_dd_pct']:.2f}%")
        for ov_name, _, _ in overlays:
            if ov_name == "base":
                continue
            r = next((x for x in results if x["config_id"] == f"{strat_name}_{ov_name}"), None)
            if r is None:
                continue
            dpf = r["profit_factor"] - base["profit_factor"]
            dpnl = r["pnl_pct"] - base["pnl_pct"]
            ddd = r["max_dd_pct"] - base["max_dd_pct"]
            tag = "+" if (dpf > 0.05 and dpnl > 0) else ("-" if dpf < -0.05 else "=")
            print(f"    [{tag}] {ov_name:16s} dPF={dpf:+.2f} dPnL={dpnl:+.1f}pp "
                  f"dDD={ddd:+.2f}pp  (trades={r['trades']})")

    # WF summary
    wf = [r for r in results if r["config_id"].startswith("WF_")]
    if wf:
        print("\n" + "="*120)
        print("WALK-FORWARD 4Q on best overlay per strategy")
        for r in wf:
            ok = (r["profit_factor"] >= 1.0 and r["pnl_pct"] > 0)
            tag = "PASS" if ok else "FAIL"
            print(f"  {r['config_id']:40s} trades={r['trades']:>4} "
                  f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+7.1f}% "
                  f"DD={r['max_dd_pct']:>5.2f}%  [{tag}]")


if __name__ == "__main__":
    main()
