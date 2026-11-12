"""Quick portfolio simulation: combine 4 strategies with capital allocation."""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_agent.backtests.backtest_v9 import (
    load_all_tf, prepare_tf_arrays,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
    run_sim_tf, START_BALANCE,
)

sl_cfg = {"type": "atr", "mult": 2.5}
tp_cfg = {"type": "fixed_r", "r": 3.5}

portfolio = [
    {
        "name": "MTF_BBS_1h->5m_pb",
        "alloc": 0.40,
        "htf_key": "df1h", "ltf_key": "df5", "htf_tf": "1h",
        "gen_fn": gen_bbs_signals_tf,
        "sig_params": {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
    },
    {
        "name": "MTF_BBS_1h->15m_next",
        "alloc": 0.25,
        "htf_key": "df1h", "ltf_key": "df15", "htf_tf": "1h",
        "gen_fn": gen_bbs_signals_tf,
        "sig_params": {"min_squeeze": 6, "adx_thresh": 18, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
    },
    {
        "name": "MTF_KC_1h->15m_next",
        "alloc": 0.20,
        "htf_key": "df1h", "ltf_key": "df15", "htf_tf": "1h",
        "gen_fn": gen_kc_squeeze_signals_tf,
        "sig_params": {"min_squeeze": 6, "adx_min": 15, "vol_mult": 1.0,
                       "mom_bars": 8, "sl_atr_mult": 2.0},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
    },
    {
        "name": "BBS_15m_single",
        "alloc": 0.15,
        "single_tf": True,
        "tf_key": "df15",
        "gen_fn": gen_bbs_signals_tf,
        "sig_params": {"min_squeeze": 10, "adx_thresh": 22, "vol_mult": 1.0,
                       "htf_filter": True, "headfake_filter": True},
    },
]


def main():
    print("Loading data...", flush=True)
    all_data = load_all_tf("180d")
    print("Symbols: %d" % len(all_data), flush=True)

    results = {}
    for strat in portfolio:
        name = strat["name"]
        print("  %s ..." % name, flush=True)

        if strat.get("single_tf"):
            tf_arrays, htf_arrays = {}, {}
            for sym, d in all_data.items():
                arr = prepare_tf_arrays(d.get(strat["tf_key"]))
                if arr is not None:
                    tf_arrays[sym] = arr
                    htf_arrays[sym] = prepare_tf_arrays(d.get("df1h"))
            sigs = {}
            for sym in tf_arrays:
                sigs[sym] = strat["gen_fn"](tf_arrays[sym], htf_arrays.get(sym), strat["sig_params"])
            res = run_sim_tf(sigs, tf_arrays, sl_cfg, tp_cfg,
                             risk_pt=0.010, cap_frac=0.60, max_pos=4)
        else:
            ltf_arrays, mtf_sigs = {}, {}
            for sym, d in all_data.items():
                htf_arr = prepare_tf_arrays(d.get(strat["htf_key"]))
                ltf_arr = prepare_tf_arrays(d.get(strat["ltf_key"]))
                if htf_arr is None or ltf_arr is None:
                    continue
                ltf_arrays[sym] = ltf_arr
                htf_trend = prepare_tf_arrays(d.get("df4h")) if strat["htf_tf"] == "1h" \
                    else prepare_tf_arrays(d.get("df1h"))
                htf_s = strat["gen_fn"](htf_arr, htf_trend, strat["sig_params"])
                mtf_sigs[sym] = gen_mtf_signals(htf_s, htf_arr, ltf_arr, strat["entry_params"])
            res = run_sim_tf(mtf_sigs, ltf_arrays, sl_cfg, tp_cfg,
                             risk_pt=0.010, cap_frac=0.60, max_pos=4)

        results[name] = res

    # ---- Report ----
    print()
    print("=" * 72)
    print("  INDIVIDUAL RESULTS")
    print("=" * 72)
    total_trades = 0
    for s in portfolio:
        r = results[s["name"]]
        total_trades += r["trades"]
        print("  %-28s alloc=%2d%%  n=%3d  PF=%.2f  PnL=%+7.1f%%  DD=%4.1f%%" % (
            s["name"], int(s["alloc"] * 100), r["trades"],
            r["profit_factor"], r["pnl_pct"], r["max_dd_pct"]))

    # Weighted portfolio
    w_pnl = sum(s["alloc"] * results[s["name"]]["pnl_pct"] for s in portfolio)
    w_dd  = sum(s["alloc"] * results[s["name"]]["max_dd_pct"] for s in portfolio)
    final = START_BALANCE * (1 + w_pnl / 100)

    print()
    print("=" * 72)
    print("  PORTFOLIO (weighted, 180 days)")
    print("=" * 72)
    print("  Starting balance:  $%.0f" % START_BALANCE)
    print("  Final balance:     $%.0f" % final)
    print("  Portfolio PnL:     %+.1f%%" % w_pnl)
    print("  Max DD (upper):    %.1f%% (weighted; actual lower due to decorrelation)" % w_dd)
    print("  Total trades:      %d" % total_trades)
    print("  Annualized PnL:    %+.0f%% (180d -> 365d linear)" % (w_pnl / 180 * 365))

    # Walk-forward test period
    wf = {
        "MTF_BBS_1h->5m_pb":    (17.8, 3.80),
        "MTF_BBS_1h->15m_next": (13.5, 2.45),
        "MTF_KC_1h->15m_next":  (13.8, 2.97),
        "BBS_15m_single":       (0.8,  1.15),
    }
    print()
    print("  --- Walk-Forward test (last 33%%, ~60 days) ---")
    wf_pnl = 0
    for s in portfolio:
        pnl, pf = wf.get(s["name"], (0, 0))
        wf_pnl += s["alloc"] * pnl
        print("  %-28s alloc=%2d%%  WF_PnL=%+5.1f%%  WF_PF=%.2f" % (
            s["name"], int(s["alloc"] * 100), pnl, pf))
    print("  -------")
    print("  Portfolio WF test PnL:  %+.1f%%" % wf_pnl)
    print("  Annualized (test):      %+.0f%% (~60d x6)" % (wf_pnl * 6))


if __name__ == "__main__":
    main()
