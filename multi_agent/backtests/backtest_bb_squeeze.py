"""
multi_agent/backtests/backtest_bb_squeeze.py

BB Squeeze Breakout Strategy Backtester -- 90-day cache, 4 variants.

Variants:
  A) CE_MULT=3.0 (tight Chandelier)
  B) CE_MULT=3.5 (medium)
  C) CE_MULT=4.0 (wide)
  D) Chandelier-only exit (no fixed TP, pure trailing)

Uses cached parquet data from data_cache/ (20 symbols x 2 TFs).

Usage:
    py -3 multi_agent/backtests/backtest_bb_squeeze.py
"""

from __future__ import annotations

import csv
import logging
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import multi_agent.config as cfg

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).parent / "data_cache"

# -- Constants --

BACKTEST_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "POL/USDT:USDT",
    "DOT/USDT:USDT",
    "UNI/USDT:USDT",
    "ATOM/USDT:USDT",
    "NEAR/USDT:USDT",
    "APT/USDT:USDT",
    "ARB/USDT:USDT",
    "OP/USDT:USDT",
    "INJ/USDT:USDT",
    "SUI/USDT:USDT",
    "1000PEPE/USDT:USDT",
]

START_BALANCE  = 1000.0
TAKER_FEE_PCT  = 0.00055
SLIPPAGE_PCT   = 0.00025
RISK_PER_TRADE = 0.02
MAX_LEVERAGE   = 25
MAX_CONCURRENT = 3
MAX_DAILY_LOSS = 4

# -- Variants --

VARIANTS = [
    {
        "name": "A) CE=3.0 (tight)",
        "ce_mult": 3.0,
        "use_fixed_tp": True,     # multi-level TP at 1.5R, 2.5R + trail
        "min_squeeze_bars": 6,
        "adx_threshold": 20.0,
        "vol_mult": 1.2,
    },
    {
        "name": "B) CE=3.5 (medium)",
        "ce_mult": 3.5,
        "use_fixed_tp": True,
        "min_squeeze_bars": 6,
        "adx_threshold": 20.0,
        "vol_mult": 1.2,
    },
    {
        "name": "C) CE=4.0 (wide)",
        "ce_mult": 4.0,
        "use_fixed_tp": True,
        "min_squeeze_bars": 6,
        "adx_threshold": 20.0,
        "vol_mult": 1.2,
    },
    {
        "name": "D) Chandelier-only (no fixed TP)",
        "ce_mult": 3.0,
        "use_fixed_tp": False,    # pure trailing exit only
        "min_squeeze_bars": 6,
        "adx_threshold": 20.0,
        "vol_mult": 1.2,
    },
]


# -- Import indicator functions from agent --

from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators,
    detect_squeeze_signal,
    chandelier_exit_sl,
    check_head_fake,
    _calculate_atr,
    _ema,
)


# -- Data loading --

def _load_cached(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """Load from parquet cache."""
    safe = symbol.replace("/", "_").replace(":", "_")
    fname = f"{safe}_{timeframe}_90d.parquet"
    fpath = CACHE_DIR / fname
    if fpath.exists():
        df = pd.read_parquet(fpath)
        for col in ("open", "high", "low", "close", "volume"):
            if col in df.columns:
                df[col] = df[col].astype(float)
        if "timestamp" in df.columns:
            df["timestamp"] = df["timestamp"].astype(int)
        return df
    logger.warning("Cache miss: %s", fpath)
    return None


# -- Backtest engine --

def run_backtest(variant: dict) -> dict:
    """
    Run full backtest for one variant across all symbols.
    Returns stats dict.
    """
    label     = variant["name"]
    ce_mult   = variant["ce_mult"]
    use_tp    = variant["use_fixed_tp"]
    min_sq    = variant["min_squeeze_bars"]
    adx_thr   = variant["adx_threshold"]
    vol_m     = variant["vol_mult"]

    balance   = START_BALANCE
    peak      = START_BALANCE
    max_dd    = 0.0
    trades    = []
    wins      = 0
    losses    = 0
    open_pos  = {}  # symbol -> pos dict

    bb_len  = cfg.BB_SQUEEZE_BB_LENGTH
    bb_mult = cfg.BB_SQUEEZE_BB_MULT
    kc_len  = cfg.BB_SQUEEZE_KC_LENGTH
    kc_mult = cfg.BB_SQUEEZE_KC_MULT
    mom_len = cfg.BB_SQUEEZE_MOM_LENGTH
    adx_per = cfg.BB_SQUEEZE_ADX_PERIOD
    ce_len  = cfg.BB_SQUEEZE_CE_LENGTH

    for symbol in BACKTEST_SYMBOLS:
        short = symbol.split("/")[0]
        df_15m = _load_cached(symbol, "15m")
        df_1h  = _load_cached(symbol, "1h")
        if df_15m is None or df_1h is None:
            continue
        if len(df_15m) < 250 or len(df_1h) < 210:
            continue

        # Precompute indicators on full 15m data
        ind = compute_squeeze_indicators(
            df_15m, bb_len=bb_len, bb_mult=bb_mult,
            kc_len=kc_len, kc_mult=kc_mult,
            mom_len=mom_len, adx_period=adx_per,
        )

        # Precompute 1h EMA-200 for HTF bias
        closes_1h = df_1h["close"].astype(float).to_numpy()
        ema200_1h = _ema(closes_1h, 200)
        timestamps_1h = df_1h["timestamp"].astype(int).to_numpy()

        timestamps_15m = df_15m["timestamp"].astype(int).to_numpy()
        n = len(df_15m)

        # Walk bar-by-bar
        sym_pos = None  # current position for this symbol

        for i in range(210, n):
            close_i = ind["closes"][i]
            high_i  = ind["highs"][i]
            low_i   = ind["lows"][i]
            ts_i    = int(timestamps_15m[i])

            # Check open position first
            if sym_pos is not None:
                direction = sym_pos["direction"]
                entry     = sym_pos["entry"]
                sl        = sym_pos["sl"]
                risk      = sym_pos["risk"]
                qty       = sym_pos["qty"]

                # Update trailing Chandelier Exit
                new_ce_sl = chandelier_exit_sl(
                    ind["highs"], ind["lows"], ind["ce_atr"],
                    direction, i, ce_length=ce_len, ce_mult=ce_mult,
                )

                if direction == "LONG":
                    # Move trailing SL up only
                    if sym_pos.get("tp1_hit", False):
                        trail = max(sym_pos["trailing_sl"], new_ce_sl)
                        sym_pos["trailing_sl"] = trail

                    pnl_r = (close_i - entry) / risk if risk > 0 else 0

                    # Check TP levels (if using fixed TP)
                    if use_tp and not sym_pos.get("tp1_hit") and pnl_r >= 1.5:
                        sym_pos["tp1_hit"] = True
                        # Close 25%
                        partial_pnl = (close_i - entry) * qty * 0.25
                        fee = close_i * qty * 0.25 * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += partial_pnl - fee
                        sym_pos["qty"] *= 0.75
                        qty = sym_pos["qty"]
                        # Move trailing SL to breakeven
                        sym_pos["trailing_sl"] = max(entry + risk * 0.1, sym_pos["trailing_sl"])

                    if use_tp and not sym_pos.get("tp2_hit") and pnl_r >= 2.5:
                        sym_pos["tp2_hit"] = True
                        # Close 25% of original (33% of remaining)
                        close_frac = 1.0 / 3.0
                        partial_pnl = (close_i - entry) * qty * close_frac
                        fee = close_i * qty * close_frac * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += partial_pnl - fee
                        sym_pos["qty"] *= (1 - close_frac)
                        qty = sym_pos["qty"]

                    # Max TP: 4R
                    if pnl_r >= 4.0:
                        pnl = (close_i - entry) * qty
                        fee = close_i * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += pnl - fee
                        trades.append(pnl - fee)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        sym_pos = None
                        continue

                    # Check SL hit
                    effective_sl = max(sl, sym_pos.get("trailing_sl", sl))
                    if low_i <= effective_sl:
                        exit_price = effective_sl
                        pnl = (exit_price - entry) * qty
                        fee = exit_price * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += pnl - fee
                        trades.append(pnl - fee)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        sym_pos = None
                        continue

                else:  # SHORT
                    if sym_pos.get("tp1_hit", False):
                        trail = min(
                            sym_pos["trailing_sl"] if sym_pos["trailing_sl"] > 0 else float("inf"),
                            new_ce_sl,
                        )
                        sym_pos["trailing_sl"] = trail

                    pnl_r = (entry - close_i) / risk if risk > 0 else 0

                    if use_tp and not sym_pos.get("tp1_hit") and pnl_r >= 1.5:
                        sym_pos["tp1_hit"] = True
                        partial_pnl = (entry - close_i) * qty * 0.25
                        fee = close_i * qty * 0.25 * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += partial_pnl - fee
                        sym_pos["qty"] *= 0.75
                        qty = sym_pos["qty"]
                        sym_pos["trailing_sl"] = min(
                            entry - risk * 0.1,
                            sym_pos["trailing_sl"] if sym_pos["trailing_sl"] > 0 else float("inf"),
                        )

                    if use_tp and not sym_pos.get("tp2_hit") and pnl_r >= 2.5:
                        sym_pos["tp2_hit"] = True
                        close_frac = 1.0 / 3.0
                        partial_pnl = (entry - close_i) * qty * close_frac
                        fee = close_i * qty * close_frac * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += partial_pnl - fee
                        sym_pos["qty"] *= (1 - close_frac)
                        qty = sym_pos["qty"]

                    if pnl_r >= 4.0:
                        pnl = (entry - close_i) * qty
                        fee = close_i * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += pnl - fee
                        trades.append(pnl - fee)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        sym_pos = None
                        continue

                    effective_sl = sym_pos.get("trailing_sl", sl)
                    if effective_sl <= 0:
                        effective_sl = sl
                    if high_i >= effective_sl:
                        exit_price = effective_sl
                        pnl = (entry - exit_price) * qty
                        fee = exit_price * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += pnl - fee
                        trades.append(pnl - fee)
                        if pnl > 0: wins += 1
                        else: losses += 1
                        sym_pos = None
                        continue

                # Update drawdown
                peak = max(peak, balance)
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                continue  # already in position, skip entry logic

            # -- No position: check for entry --
            if len(open_pos) >= MAX_CONCURRENT:
                continue

            # Determine HTF bias from 1h EMA-200
            # Find the last 1h bar before current 15m timestamp
            htf_idx = np.searchsorted(timestamps_1h, ts_i, side="right") - 1
            if htf_idx < 200 or htf_idx >= len(ema200_1h):
                continue
            if np.isnan(ema200_1h[htf_idx]):
                continue
            htf_close = closes_1h[htf_idx]
            htf_ema   = ema200_1h[htf_idx]
            htf_direction = "LONG" if htf_close > htf_ema else "SHORT"

            # Detect squeeze signal
            signal = detect_squeeze_signal(
                ind, bar_idx=i,
                min_squeeze_bars=min_sq,
                adx_threshold=adx_thr,
                vol_mult=vol_m,
            )
            if signal is None:
                continue

            # Must align with HTF
            if signal["direction"] != htf_direction:
                continue

            direction = signal["direction"]
            entry_price = close_i

            # Head-fake check (only in backtest, we can look ahead)
            if check_head_fake(ind, i, check_bars=2):
                continue

            # Chandelier Exit SL
            sl = chandelier_exit_sl(
                ind["highs"], ind["lows"], ind["ce_atr"],
                direction, i, ce_length=ce_len, ce_mult=ce_mult,
            )

            risk = abs(entry_price - sl)
            if risk <= 0 or risk / entry_price > 0.05:
                continue

            # Position size
            risk_usd = balance * RISK_PER_TRADE
            qty = risk_usd / risk
            margin = entry_price * qty / MAX_LEVERAGE
            if margin > balance * 0.25:
                qty = balance * 0.25 * MAX_LEVERAGE / entry_price

            if qty <= 0:
                continue

            # Entry fee
            fee = entry_price * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
            balance -= fee

            sym_pos = {
                "direction": direction,
                "entry": entry_price,
                "sl": sl,
                "risk": risk,
                "qty": qty,
                "trailing_sl": sl,
                "tp1_hit": False,
                "tp2_hit": False,
                "bar_idx": i,
                "symbol": symbol,
            }

            open_pos[symbol] = True

        # Close remaining position at last bar
        if sym_pos is not None:
            last_close = ind["closes"][-1]
            entry = sym_pos["entry"]
            qty = sym_pos["qty"]
            if sym_pos["direction"] == "LONG":
                pnl = (last_close - entry) * qty
            else:
                pnl = (entry - last_close) * qty
            fee = last_close * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
            balance += pnl - fee
            trades.append(pnl - fee)
            if pnl > 0: wins += 1
            else: losses += 1

        open_pos.pop(symbol, None)

    # Compute stats
    total_trades = wins + losses
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100.0
    wr = (wins / total_trades * 100.0) if total_trades > 0 else 0.0

    win_trades = [t for t in trades if t > 0]
    loss_trades = [t for t in trades if t <= 0]
    avg_win = float(np.mean(win_trades)) if win_trades else 0.0
    avg_loss = float(np.mean(loss_trades)) if loss_trades else 0.0
    gross_profit = sum(win_trades)
    gross_loss = abs(sum(loss_trades))
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    peak = max(peak, balance)
    dd = (peak - balance) / peak * 100 if peak > 0 else 0
    max_dd = max(max_dd, dd)

    result = {
        "variant": label,
        "trades": total_trades,
        "wins": wins,
        "losses": losses,
        "wr_pct": round(wr, 1),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "avg_win": round(avg_win, 4),
        "avg_loss": round(avg_loss, 4),
        "final_balance": round(balance, 2),
    }

    logger.info(
        "%-35s | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f",
        label, total_trades, wr, pnl_pct, max_dd, pf,
    )

    return result


# -- Main --

def main():
    logger.info("=" * 80)
    logger.info("BB Squeeze Breakout Backtester -- 4 Variants x 20 Symbols x 90 Days")
    logger.info("=" * 80)

    results = []
    for v in VARIANTS:
        r = run_backtest(v)
        results.append(r)

    # Print comparison table
    print("\n" + "=" * 100)
    print("BB SQUEEZE BACKTEST RESULTS -- COMPARISON TABLE")
    print("=" * 100)
    header = f"{'Variant':<35} {'Trades':>6} {'WR%':>6} {'PnL%':>8} {'MaxDD%':>8} {'PF':>6} {'Balance':>10}"
    print(header)
    print("-" * 100)
    for r in results:
        row = (
            f"{r['variant']:<35} {r['trades']:>6} {r['wr_pct']:>5.1f}% "
            f"{r['pnl_pct']:>+7.2f}% {r['max_dd_pct']:>7.2f}% "
            f"{r['profit_factor']:>5.2f} ${r['final_balance']:>9.2f}"
        )
        print(row)
    print("=" * 100)

    # Save CSV
    safe_label = re.sub(r'[<>:"/\\|?*\s]+', '_', "bb_squeeze_backtest")
    csv_path = RESULTS_DIR / f"{safe_label}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    logger.info("Results saved to %s", csv_path)

    # Find best variant
    best = max(results, key=lambda r: r["pnl_pct"])
    logger.info("BEST VARIANT: %s (PnL=%+.2f%%, WR=%.1f%%, PF=%.2f)",
                best["variant"], best["pnl_pct"], best["wr_pct"], best["profit_factor"])


if __name__ == "__main__":
    main()
