"""
backtest_v8.py  --  Regime-Allocated Portfolio  (2026-04-13)

v7 showed: BBS dominates TREND, but EMA/Donchian/KC have edge in other regimes.
This test: force each strategy to ONLY trade in its best regimes, then combine.

Goal: Fill the regime gaps that BBS alone can't cover.

Allocations to test:
  BBS     -> TREND, TRANSITION
  KC_SQ   -> SQUEEZE (PF=1.62 in v7)
  DON     -> RANGE, SQUEEZE (PF=1.31/1.07 in v7)
  EMA     -> SQUEEZE, TRANSITION (PF=1.26/1.16 in v7)

Also: deeper BBS optimization with more param combos around the sweet spot.
"""
from __future__ import annotations
import json, logging, sys, time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.shared.regime_detector import SoftRegimeDetector
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
from multi_agent.backtests.backtest_v4 import (
    preload_all, BBS_BLACKLIST_CANDIDATES,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
)
from multi_agent.backtests.backtest_v7 import (
    compute_extra, gen_bbs_signals, gen_ema_mom_signals,
    gen_donchian_signals, gen_kc_squeeze_signals,
    compute_sl, run_sim, _score,
)
import multi_agent.config as cfg

logger = logging.getLogger("v8")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SOFT_CFG = {"confirm_bars": 5, "min_hold_bars": 12, "smooth_alpha": 0.30,
            "min_confidence": 1.0, "transition_margin": 0.5,
            "undefined_threshold": 0.5}


def phase0(all_data):
    extra_ind = {}
    regime_cache = {}
    total = len(all_data)
    for k, (sym, D) in enumerate(all_data.items()):
        print("  [%d/%d] %s ..." % (k+1, total, sym), flush=True)
        extra_ind[sym] = compute_extra(D)
        n = D["n_bars"]
        rarr = ["UNDEFINED"] * n
        det = SoftRegimeDetector(SOFT_CFG)
        for i in range(300, n):
            ts_i = D["ts15"][i]
            htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
            if htf_idx < 210:
                continue
            rs = det.update(sym, D["df1h"], htf_idx, D["df4h"],
                            **(D["ind_1h"]), **(D["ind_4h"]))
            rarr[i] = rs.regime.value
        regime_cache[sym] = rarr
    return extra_ind, regime_cache


def walk_forward(name, signals_by_sym, all_data, extra_ind, sl_cfg, tp_cfg,
                 risk_pt, cap_frac, max_pos, mgt_cfg=None):
    """67/33 walk-forward."""
    train = run_sim(signals_by_sym, all_data, extra_ind,
                    sl_cfg, tp_cfg, mgt_cfg=mgt_cfg,
                    risk_pt=risk_pt, cap_frac=cap_frac, max_pos=max_pos,
                    start_frac=0.0, end_frac=0.67)
    test = run_sim(signals_by_sym, all_data, extra_ind,
                   sl_cfg, tp_cfg, mgt_cfg=mgt_cfg,
                   risk_pt=risk_pt, cap_frac=cap_frac, max_pos=max_pos,
                   start_frac=0.67, end_frac=1.0)
    for r in [train, test]:
        r.pop("completed", None)
    passed = test["profit_factor"] >= 1.0 and test["pnl_pct"] > -5.0
    tag = "PASS" if passed else "FAIL"
    print("  WF %-12s train=%+6.1f%%(PF=%.2f n=%d) test=%+6.1f%%(PF=%.2f n=%d) [%s]"
          % (name, train["pnl_pct"], train["profit_factor"], train["trades"],
             test["pnl_pct"], test["profit_factor"], test["trades"], tag), flush=True)
    return {"name": name, "train": train, "test": test, "passed": passed, "tag": tag}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("backtest_v8.py -- Regime-Allocated Portfolio", flush=True)
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    t0 = _time.time()

    all_data = preload_all("180d", blacklist=set())
    print("Symbols loaded:", len(all_data), flush=True)

    print("\n  Phase 0 -- Pre-computing...", flush=True)
    p0_t = _time.time()
    extra_ind, regime_cache = phase0(all_data)
    print("  Phase 0 done in %d sec" % int(_time.time() - p0_t), flush=True)

    # ================================================================
    #  TEST 1: BBS deeper optimization (sweet spot around sq10, adx20)
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  TEST 1 -- BBS Deep Optimization (20 combos)", flush=True)
    print("=" * 70, flush=True)

    bbs_combos = []
    for sq in [8, 10, 12, 15]:
        for adx in [18, 20, 22, 25]:
            for vol in [1.0, 1.2]:
                bbs_combos.append({
                    "id": "bbs_sq%d_adx%d_vol%.0f" % (sq, adx, vol*10),
                    "params": {"min_squeeze": sq, "adx_thresh": adx, "vol_mult": vol,
                               "htf_filter": True, "headfake_filter": True,
                               "regimes": {"TREND": 1.0, "TRANSITION": 0.85}},
                })

    bbs_results = []
    for c in bbs_combos:
        all_sigs = {}
        total_sigs = 0
        for sym, D in all_data.items():
            s = gen_bbs_signals(D, extra_ind[sym], regime_cache[sym], c["params"])
            all_sigs[sym] = s
            total_sigs += len(s)
        res = run_sim(all_sigs, all_data, extra_ind,
                      {"type": "atr", "mult": 2.5},
                      {"type": "fixed_r", "r": 3.5},
                      risk_pt=0.010, cap_frac=0.75, max_pos=4)
        res.pop("completed", None)
        res["config_id"] = c["id"]
        res["total_sigs"] = total_sigs
        res["score"] = _score(res)
        res["params"] = c["params"]
        bbs_results.append(res)
        print("  %-35s sigs=%4d n=%3d PF=%.2f PnL=%+6.1f%% DD=%.1f%%"
              % (c["id"], total_sigs, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"]), flush=True)

    bbs_results.sort(key=lambda x: x["score"], reverse=True)
    print("\n  BBS Top-5:", flush=True)
    for i, r in enumerate(bbs_results[:5]):
        print("    %d. %-30s PF=%.3f PnL=%+.1f%% n=%d DD=%.1f%%"
              % (i+1, r["config_id"], r["profit_factor"],
                 r["pnl_pct"], r["trades"], r["max_dd_pct"]), flush=True)

    # ================================================================
    #  TEST 2: Regime-restricted strategies
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  TEST 2 -- Regime-Restricted Strategies", flush=True)
    print("=" * 70, flush=True)

    # Best BBS params from grid (use top-1)
    best_bbs = bbs_results[0]
    best_bbs_params = best_bbs["params"]

    # Strategy configs with REGIME RESTRICTIONS
    regime_tests = [
        # KC in SQUEEZE only
        {"name": "KC_SQUEEZE_only", "gen_fn": gen_kc_squeeze_signals,
         "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                    "regimes": {"SQUEEZE": 1.0}, "sl_atr_mult": 2.0},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
         "risk": 0.010, "frac": 0.50, "max_pos": 3},

        # KC in SQUEEZE+TRANSITION
        {"name": "KC_SQ_TRANS", "gen_fn": gen_kc_squeeze_signals,
         "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                    "regimes": {"SQUEEZE": 1.0, "TRANSITION": 0.85}, "sl_atr_mult": 2.0},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
         "risk": 0.010, "frac": 0.50, "max_pos": 3},

        # KC partial in SQUEEZE
        {"name": "KC_partial_SQ", "gen_fn": gen_kc_squeeze_signals,
         "params": {"min_squeeze": 10, "adx_min": 15, "vol_mult": 1.0, "mom_bars": 12,
                    "regimes": {"SQUEEZE": 1.0}, "sl_atr_mult": 2.0},
         "sl": {"type": "atr", "mult": 2.5},
         "tp": {"type": "partial_fixed", "r1": 2.0, "r2": 4.0, "frac": 0.5},
         "risk": 0.010, "frac": 0.50, "max_pos": 3},

        # Donchian in RANGE only
        {"name": "DON_RANGE_only", "gen_fn": gen_donchian_signals,
         "params": {"period": 100, "adx_min": 0, "cooldown": 20, "sl_atr_mult": 2.5,
                    "regimes": {"RANGE": 1.0}},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},

        # Donchian in RANGE+SQUEEZE
        {"name": "DON_RNG_SQ", "gen_fn": gen_donchian_signals,
         "params": {"period": 100, "adx_min": 0, "cooldown": 20, "sl_atr_mult": 2.5,
                    "regimes": {"RANGE": 1.0, "SQUEEZE": 0.8}},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.5},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},

        # Donchian 55 in SQUEEZE+RANGE
        {"name": "DON55_RNG_SQ", "gen_fn": gen_donchian_signals,
         "params": {"period": 55, "adx_min": 0, "cooldown": 10, "sl_atr_mult": 2.0,
                    "regimes": {"RANGE": 1.0, "SQUEEZE": 0.8}},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.0},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},

        # Donchian 55 HTF in SQUEEZE+RANGE
        {"name": "DON55_HTF_SQ", "gen_fn": gen_donchian_signals,
         "params": {"period": 55, "adx_min": 0, "cooldown": 10, "sl_atr_mult": 2.0,
                    "htf_filter": True, "regimes": {"RANGE": 1.0, "SQUEEZE": 0.8}},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.0},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},

        # EMA in SQUEEZE only
        {"name": "EMA_SQUEEZE", "gen_fn": gen_ema_mom_signals,
         "params": {"fast": 8, "slow": 50, "adx_min": 20, "vol_mult": 1.0,
                    "htf_filter": True, "sl_atr_mult": 2.5,
                    "regimes": {"SQUEEZE": 1.0}},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.0},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},

        # EMA in SQUEEZE+TRANSITION
        {"name": "EMA_SQ_TRANS", "gen_fn": gen_ema_mom_signals,
         "params": {"fast": 8, "slow": 50, "adx_min": 20, "vol_mult": 1.0,
                    "htf_filter": True, "sl_atr_mult": 2.5,
                    "regimes": {"SQUEEZE": 1.0, "TRANSITION": 0.85}},
         "sl": {"type": "atr", "mult": 2.5}, "tp": {"type": "fixed_r", "r": 3.0},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},

        # EMA 13/50 in SQUEEZE+TRANSITION
        {"name": "EMA13_50_SQ_TR", "gen_fn": gen_ema_mom_signals,
         "params": {"fast": 13, "slow": 50, "adx_min": 20, "vol_mult": 1.0,
                    "htf_filter": False, "sl_atr_mult": 2.0,
                    "regimes": {"SQUEEZE": 1.0, "TRANSITION": 0.85}},
         "sl": {"type": "atr", "mult": 2.0}, "tp": {"type": "fixed_r", "r": 3.0},
         "risk": 0.010, "frac": 0.40, "max_pos": 3},
    ]

    regime_results = {}
    for cfg_r in regime_tests:
        all_sigs = {}
        total_sigs = 0
        for sym, D in all_data.items():
            s = cfg_r["gen_fn"](D, extra_ind[sym], regime_cache[sym], cfg_r["params"])
            all_sigs[sym] = s
            total_sigs += len(s)
        res = run_sim(all_sigs, all_data, extra_ind,
                      cfg_r["sl"], cfg_r["tp"],
                      risk_pt=cfg_r["risk"], cap_frac=cfg_r["frac"],
                      max_pos=cfg_r["max_pos"])
        res.pop("completed", None)
        res["config_name"] = cfg_r["name"]
        res["total_sigs"] = total_sigs
        res["score"] = _score(res)
        regime_results[cfg_r["name"]] = {"result": res, "cfg": cfg_r, "sigs": all_sigs}

        regime_str = ""
        for rg, v in sorted(res.get("per_regime", {}).items()):
            regime_str += " %s:PF=%.2f(%d)" % (rg, v["pf"], v["n"])
        print("  %-22s sigs=%5d n=%3d PF=%.2f PnL=%+6.1f%% DD=%.1f%% %s"
              % (cfg_r["name"], total_sigs, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"], regime_str), flush=True)

    # ================================================================
    #  TEST 3: Combined portfolios
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  TEST 3 -- Combined Portfolios", flush=True)
    print("=" * 70, flush=True)

    # Generate BBS signals with best params
    bbs_sigs = {}
    for sym, D in all_data.items():
        bbs_sigs[sym] = gen_bbs_signals(D, extra_ind[sym], regime_cache[sym],
                                         best_bbs_params)

    # Try different combinations
    # Find viable regime-restricted strategies (PF > 1.0)
    viable_regime = {k: v for k, v in regime_results.items()
                     if v["result"]["profit_factor"] >= 1.0 and v["result"]["trades"] >= 10}

    print("  Viable regime-restricted: %s" % list(viable_regime.keys()), flush=True)

    portfolio_tests = []

    # Portfolio 1: BBS only (baseline)
    p1_sigs = {sym: dict(bbs_sigs.get(sym, {})) for sym in all_data}
    p1 = run_sim(p1_sigs, all_data, extra_ind,
                 {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5},
                 risk_pt=0.010, cap_frac=0.70, max_pos=5)
    p1.pop("completed", None)
    portfolio_tests.append(("BBS_only", p1, p1_sigs))

    # Portfolio 2+: BBS + each viable regime strategy
    for vname, vdata in viable_regime.items():
        combined = {}
        for sym in all_data:
            combined[sym] = dict(bbs_sigs.get(sym, {}))
            v_sigs = vdata["sigs"].get(sym, {})
            for bar_i, sig in v_sigs.items():
                if bar_i not in combined[sym]:
                    combined[sym][bar_i] = sig
        total_c = sum(len(v) for v in combined.values())
        res_c = run_sim(combined, all_data, extra_ind,
                        {"type": "atr", "mult": 2.5},
                        {"type": "fixed_r", "r": 3.5},
                        risk_pt=0.010, cap_frac=0.65, max_pos=5)
        res_c.pop("completed", None)
        portfolio_tests.append(("BBS+%s" % vname, res_c, combined))

    # Portfolio all viable: BBS + all viable regime strategies
    if len(viable_regime) > 1:
        combined_all = {}
        for sym in all_data:
            combined_all[sym] = dict(bbs_sigs.get(sym, {}))
            for vname, vdata in viable_regime.items():
                v_sigs = vdata["sigs"].get(sym, {})
                for bar_i, sig in v_sigs.items():
                    if bar_i not in combined_all[sym]:
                        combined_all[sym][bar_i] = sig
        res_all = run_sim(combined_all, all_data, extra_ind,
                          {"type": "atr", "mult": 2.5},
                          {"type": "fixed_r", "r": 3.5},
                          risk_pt=0.008, cap_frac=0.60, max_pos=6)
        res_all.pop("completed", None)
        portfolio_tests.append(("BBS+ALL_VIABLE", res_all, combined_all))

    print("\n  Portfolio Results:", flush=True)
    for pname, pres, _ in portfolio_tests:
        regime_str = ""
        for rg, v in sorted(pres.get("per_regime", {}).items()):
            regime_str += " %s:%.2f(%d)" % (rg, v["pf"], v["n"])
        print("  %-30s n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%% WR=%d%%%s"
              % (pname, pres["trades"], pres["profit_factor"],
                 pres["pnl_pct"], pres["max_dd_pct"], pres["wr_pct"],
                 regime_str), flush=True)

    # Sort by PF
    portfolio_tests.sort(key=lambda x: x[1].get("profit_factor", 0), reverse=True)

    # ================================================================
    #  TEST 4: Walk-forward on best portfolios
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  TEST 4 -- Walk-Forward on Top Portfolios", flush=True)
    print("=" * 70, flush=True)

    wf_results = []
    for pname, pres, p_sigs in portfolio_tests[:5]:
        if pres["trades"] < 15:
            continue
        wfr = walk_forward(pname, p_sigs, all_data, extra_ind,
                           {"type": "atr", "mult": 2.5},
                           {"type": "fixed_r", "r": 3.5},
                           risk_pt=0.009, cap_frac=0.65, max_pos=5)
        wf_results.append({"name": pname, **wfr})

    # Also test BBS top-1 standalone walk-forward
    wfr_bbs = walk_forward("BBS_best_solo", bbs_sigs, all_data, extra_ind,
                           {"type": "atr", "mult": 2.5},
                           {"type": "fixed_r", "r": 3.5},
                           risk_pt=0.010, cap_frac=0.75, max_pos=4)
    wf_results.append({"name": "BBS_best_solo", **wfr_bbs})

    # ================================================================
    #  TEST 5: BBS with different SL/TP combos on best params
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  TEST 5 -- BBS Best Params SL/TP Matrix", flush=True)
    print("=" * 70, flush=True)

    sltp_combos = [
        ("atr2.0_3.0r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 3.0}),
        ("atr2.0_3.5r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 3.5}),
        ("atr2.0_4.0r", {"type": "atr", "mult": 2.0}, {"type": "fixed_r", "r": 4.0}),
        ("atr2.5_3.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.0}),
        ("atr2.5_3.5r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 3.5}),
        ("atr2.5_4.0r", {"type": "atr", "mult": 2.5}, {"type": "fixed_r", "r": 4.0}),
        ("atr3.0_3.5r", {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 3.5}),
        ("atr3.0_4.0r", {"type": "atr", "mult": 3.0}, {"type": "fixed_r", "r": 4.0}),
        ("atr2.5_part2_4", {"type": "atr", "mult": 2.5},
         {"type": "partial_fixed", "r1": 2.0, "r2": 4.0, "frac": 0.5}),
        ("atr2.5_part1.5_3.5", {"type": "atr", "mult": 2.5},
         {"type": "partial_fixed", "r1": 1.5, "r2": 3.5, "frac": 0.5}),
        ("def_3.5r", {"type": "default"}, {"type": "fixed_r", "r": 3.5}),
        ("def_4.0r", {"type": "default"}, {"type": "fixed_r", "r": 4.0}),
    ]

    sltp_results = []
    for sname, sl_c, tp_c in sltp_combos:
        res = run_sim(bbs_sigs, all_data, extra_ind,
                      sl_c, tp_c, risk_pt=0.010, cap_frac=0.75, max_pos=4)
        res.pop("completed", None)
        res["sltp_id"] = sname
        sltp_results.append(res)
        print("  %-25s n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%% MFE_cap=%.1f%% GB=%.1f%%"
              % (sname, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"],
                 res.get("mfe_cap", 0), res.get("giveback", 0)), flush=True)

    sltp_results.sort(key=lambda x: _score(x), reverse=True)

    # ================================================================
    #  FINAL REPORT
    # ================================================================
    print("\n" + "=" * 70, flush=True)
    print("  FINAL REPORT -- Regime-Allocated Portfolio v8", flush=True)
    print("=" * 70, flush=True)

    print("\n  == BBS GRID (top 10) ==", flush=True)
    for i, r in enumerate(bbs_results[:10]):
        sq = r["params"]["min_squeeze"]; adx = r["params"]["adx_thresh"]
        vol = r["params"]["vol_mult"]
        print("    %2d. sq=%2d adx=%2d vol=%.1f  n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%%"
              % (i+1, sq, adx, vol, r["trades"], r["profit_factor"],
                 r["pnl_pct"], r["max_dd_pct"]), flush=True)

    print("\n  == REGIME-RESTRICTED (all) ==", flush=True)
    for vname, vdata in sorted(regime_results.items(),
                                key=lambda x: x[1]["result"]["profit_factor"],
                                reverse=True):
        r = vdata["result"]
        print("    %-22s n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%%"
              % (vname, r["trades"], r["profit_factor"],
                 r["pnl_pct"], r["max_dd_pct"]), flush=True)

    print("\n  == COMBINED PORTFOLIOS ==", flush=True)
    for pname, pres, _ in portfolio_tests:
        print("    %-30s n=%3d PF=%.3f PnL=%+6.1f%% DD=%.1f%%"
              % (pname, pres["trades"], pres["profit_factor"],
                 pres["pnl_pct"], pres["max_dd_pct"]), flush=True)

    print("\n  == SL/TP MATRIX (BBS best) ==", flush=True)
    for i, r in enumerate(sltp_results[:5]):
        print("    %d. %-25s PF=%.3f PnL=%+6.1f%% DD=%.1f%% MFE=%.1f%% GB=%.1f%%"
              % (i+1, r["sltp_id"], r["profit_factor"],
                 r["pnl_pct"], r["max_dd_pct"],
                 r.get("mfe_cap", 0), r.get("giveback", 0)), flush=True)

    print("\n  == WALK-FORWARD ==", flush=True)
    for wfr in wf_results:
        train = wfr.get("train", {})
        test_d = wfr.get("test", {})
        print("    %-30s train=%+6.1f%%(PF=%.2f) test=%+6.1f%%(PF=%.2f) [%s]"
              % (wfr["name"],
                 train.get("pnl_pct", 0), train.get("profit_factor", 0),
                 test_d.get("pnl_pct", 0), test_d.get("profit_factor", 0),
                 wfr["tag"]), flush=True)

    passed = [w for w in wf_results if w.get("passed")]
    if passed:
        print("\n  ** WALK-FORWARD PASSED: **", flush=True)
        for w in passed:
            test_d = w.get("test", {})
            print("    %s  test_PF=%.2f test_PnL=%+.1f%% n_test=%d"
                  % (w["name"], test_d.get("profit_factor", 0),
                     test_d.get("pnl_pct", 0), test_d.get("trades", 0)), flush=True)

    # ---- RECOMMENDATION ----
    print("\n  == PRODUCTION RECOMMENDATION ==", flush=True)
    best_bbs_cfg = bbs_results[0]
    print("    BBS: sq=%d, adx=%.0f, vol=%.1f, ATR_SL=2.5, TP=3.5R"
          % (best_bbs_cfg["params"]["min_squeeze"],
             best_bbs_cfg["params"]["adx_thresh"],
             best_bbs_cfg["params"]["vol_mult"]), flush=True)
    print("    Expected: PF=%.2f, ~%d trades/180d, DD<%.0f%%"
          % (best_bbs_cfg["profit_factor"], best_bbs_cfg["trades"],
             best_bbs_cfg["max_dd_pct"]), flush=True)

    best_wf = max(passed, key=lambda w: w.get("test", {}).get("profit_factor", 0)) if passed else None
    if best_wf:
        test_d = best_wf.get("test", {})
        print("    Best WF portfolio: %s (test PF=%.2f)"
              % (best_wf["name"], test_d.get("profit_factor", 0)), flush=True)

    # Save
    out = {
        "bbs_grid": bbs_results[:10],
        "regime_restricted": {k: v["result"] for k, v in regime_results.items()},
        "portfolios": [(n, r) for n, r, _ in portfolio_tests],
        "sltp_matrix": sltp_results[:8],
        "walk_forward": [{k: v for k, v in w.items() if k != "sigs"} for w in wf_results],
        "recommendation": {
            "bbs_params": best_bbs_cfg["params"],
            "bbs_sl": "ATR 2.5",
            "bbs_tp": "fixed 3.5R",
            "expected_pf": best_bbs_cfg["profit_factor"],
            "expected_trades_180d": best_bbs_cfg["trades"],
        },
        "metadata": {"timestamp": ts, "symbols": len(all_data),
                      "runtime_sec": round(_time.time() - t0, 1)},
    }
    out_path = RESULTS_DIR / ("backtest_v8_%s.json" % ts)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print("\n  Results saved:", out_path, flush=True)
    print("\nTotal runtime: %d min %d sec" % divmod(int(_time.time() - t0), 60), flush=True)


if __name__ == "__main__":
    main()
