"""
sl_audit.py — Stop-Loss quality audit
=======================================
Question: "Was SL hit but price reversed? Does wider SL help?"

For each losing trade (SL exit):
  • Track price for next N bars (12, 24, 48)
  • Did high (LONG) / low (SHORT) reach the original TP?
  • Did high reach 1R, 2R, 3R from entry?
  • If >40% of SLs would have hit TP with wider stop → manipulation hypothesis confirmed

Methodology: re-run signal generation, simulate with original SL,
then for SL exits, peek into future bars.
"""
from __future__ import annotations
import sys
import time as _time
from pathlib import Path
from collections import defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr, sma, ema, rsi, supertrend, _ts_period_ms,
    strat_supertrend_4h, strat_smacross_1d, strat_donchian_1d,
    strat_rsi_reversion_4h, SYMBOLS_730D, TAKER_FEE, SLIPPAGE,
    STRATEGIES, STRATEGIES_MULTI_TF, strat_st_1d_4h,
)


def audit_strategy(strat_name: str, symbols: list[str], regime, tag="730d",
                   peek_bars=(6, 12, 24, 48, 96)):
    """Run strategy, simulate, then audit each SL exit."""
    if strat_name in STRATEGIES:
        fn, tf = STRATEGIES[strat_name]
        is_multi = False
    elif strat_name in STRATEGIES_MULTI_TF:
        is_multi = True
        primary_tf, secondary_tf, fn = STRATEGIES_MULTI_TF[strat_name]
    else:
        print(f"Unknown strategy: {strat_name}")
        return

    audit_records = []
    n_total_sl = 0
    n_audited = 0

    for sym in symbols:
        if is_multi:
            df_p = load(sym, primary_tf, tag)
            if df_p is None: continue
            if secondary_tf:
                df_s = load(sym, secondary_tf, tag)
                sigs = fn(df_p, df_s, regime)
                sim_df = df_s if strat_name == "st_1d_4h" else df_p
            else:
                sigs = fn(df_p, regime)
                sim_df = df_p
        else:
            df = load(sym, tf, tag)
            if df is None: continue
            sigs = fn(df, regime)
            sim_df = df

        if not sigs or sim_df is None: continue
        ts = sim_df["timestamp"].to_numpy()
        op = sim_df["open"].to_numpy()
        hi = sim_df["high"].to_numpy()
        lo = sim_df["low"].to_numpy()
        cl = sim_df["close"].to_numpy()
        n = len(ts)

        for sig in sigs:
            j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
            if j >= n - 1: continue
            entry = op[j] * (1 + SLIPPAGE) if sig.direction == "LONG" else op[j] * (1 - SLIPPAGE)
            sl = sig.sl
            if sig.direction == "LONG":
                dist = entry - sl
                if dist <= 0: continue
                tp = entry + (sig.tp - sig.entry) * (dist / sig.dist) if sig.dist > 0 else sig.tp
            else:
                dist = sl - entry
                if dist <= 0: continue
                tp = entry - (sig.entry - sig.tp) * (dist / sig.dist) if sig.dist > 0 else sig.tp

            # Walk forward to find SL/TP exit
            exit_p = None; exit_bar = None; reason = "timeout"
            for k in range(j, min(j + 500, n)):
                h = hi[k]; l = lo[k]
                if sig.direction == "LONG":
                    if l <= sl:
                        exit_p = sl; reason = "SL"; exit_bar = k; break
                    if h >= tp:
                        exit_p = tp; reason = "TP"; exit_bar = k; break
                else:
                    if h >= sl:
                        exit_p = sl; reason = "SL"; exit_bar = k; break
                    if l <= tp:
                        exit_p = tp; reason = "TP"; exit_bar = k; break

            if reason != "SL" or exit_bar is None:
                continue
            n_total_sl += 1

            # Now PEEK into future from exit_bar
            audit = {"sym": sym, "strat": strat_name, "direction": sig.direction,
                     "entry": entry, "sl": sl, "tp": tp, "dist": dist,
                     "exit_bar": exit_bar, "exit_ts": int(ts[exit_bar])}

            for nbars in peek_bars:
                end = min(exit_bar + nbars, n - 1)
                if sig.direction == "LONG":
                    fut_high = float(np.max(hi[exit_bar+1:end+1])) if end > exit_bar else hi[exit_bar]
                    fut_low  = float(np.min(lo[exit_bar+1:end+1])) if end > exit_bar else lo[exit_bar]
                    # would have hit TP?
                    hit_tp = fut_high >= tp
                    # Max favorable excursion in R-units from entry
                    mfe_r = (fut_high - entry) / dist if dist > 0 else 0
                    mae_r = (entry - fut_low) / dist if dist > 0 else 0
                else:
                    fut_high = float(np.max(hi[exit_bar+1:end+1])) if end > exit_bar else hi[exit_bar]
                    fut_low  = float(np.min(lo[exit_bar+1:end+1])) if end > exit_bar else lo[exit_bar]
                    hit_tp = fut_low <= tp
                    mfe_r = (entry - fut_low) / dist if dist > 0 else 0
                    mae_r = (fut_high - entry) / dist if dist > 0 else 0
                audit[f"hit_tp_{nbars}"] = hit_tp
                audit[f"mfe_r_{nbars}"] = round(mfe_r, 2)
                audit[f"mae_r_{nbars}"] = round(mae_r, 2)

            audit_records.append(audit)
            n_audited += 1

    if not audit_records:
        print(f"  {strat_name}: no SL trades found")
        return

    df = pd.DataFrame(audit_records)
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  SL AUDIT: {strat_name}  (n={len(df)} SL trades)")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  After SL exit, would price have hit ORIGINAL TP within N bars?")
    print(f"  {'Window':>8} {'False stops %':>15} {'Avg MFE (R)':>14} {'Avg MAE (R)':>14} {'Median MFE':>12}")
    for nb in peek_bars:
        false_pct = df[f"hit_tp_{nb}"].mean() * 100
        avg_mfe = df[f"mfe_r_{nb}"].mean()
        avg_mae = df[f"mae_r_{nb}"].mean()
        med_mfe = df[f"mfe_r_{nb}"].median()
        print(f"  {nb:>4d}bars {false_pct:>13.1f}%  {avg_mfe:>+12.2f}R  {avg_mae:>+12.2f}R  {med_mfe:>+10.2f}R")

    # Distribution buckets
    print(f"\n  MFE distribution after 24 bars (LONG would have reached above entry by R):")
    mfe24 = df["mfe_r_24"]
    for thresh in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
        pct = (mfe24 >= thresh).mean() * 100
        print(f"    MFE >= {thresh:.1f}R : {pct:.1f}% of stops")

    print(f"\n  MAE (how far past SL did price go before reversing) after 24 bars:")
    mae24 = df["mae_r_24"]
    for thresh in (0.0, 0.5, 1.0, 1.5, 2.0):
        pct = (mae24 >= thresh).mean() * 100
        print(f"    Went {thresh:.1f}R+ past SL : {pct:.1f}% of stops")

    # Specific buckets: false-stop bands
    pct_recover_to_tp = df["hit_tp_24"].mean() * 100
    pct_recover_to_1r = (df["mfe_r_24"] >= 1.0).mean() * 100
    pct_recover_to_2r = (df["mfe_r_24"] >= 2.0).mean() * 100

    return {
        "strat": strat_name,
        "n_sl": len(df),
        "false_stop_24bar_to_tp": pct_recover_to_tp,
        "false_stop_24bar_to_1r": pct_recover_to_1r,
        "false_stop_24bar_to_2r": pct_recover_to_2r,
        "df": df,
    }


def main():
    btc_1d = load("BTC", "1d", "730d")
    if btc_1d is None:
        print("BTC data missing"); return
    regime = build_regime(btc_1d)
    syms = SYMBOLS_730D

    results = []
    for sname in ["supertrend_4h", "smacross_1d", "donchian_1d", "rsi_reversion_4h", "st_1d_4h"]:
        r = audit_strategy(sname, syms, regime, "730d")
        if r: results.append(r)

    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  SUMMARY: false-stop rates (price reversed within 24 bars after SL)")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Strategy':22s} {'n_SL':>6s} {'TP hit %':>10s} {'>=1R %':>9s} {'>=2R %':>9s}")
    for r in results:
        print(f"  {r['strat']:22s} {r['n_sl']:>6d} {r['false_stop_24bar_to_tp']:>9.1f}% "
              f"{r['false_stop_24bar_to_1r']:>8.1f}% {r['false_stop_24bar_to_2r']:>8.1f}%")


if __name__ == "__main__":
    main()
