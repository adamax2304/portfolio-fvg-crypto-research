"""
multi_agent/backtests/backtest_portfolio.py

Unified regime-aware backtest engine.

Tests:
  1. FVG standalone (regime-gated to TREND)
  2. BB Squeeze standalone (regime-gated to SQUEEZE)
  3. Density standalone (regime-gated to RANGE)
  4. Mean Reversion standalone (regime-gated to RANGE)
  5. Combined portfolio with regime switching
  6. Ablation: combined WITHOUT regime detector
  7. Ablation: BB Squeeze old (no regime gate) vs restricted

Usage:
    py -3 multi_agent/backtests/backtest_portfolio.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time as _time
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
from multi_agent.shared.regime_detector import (
    RegimeDetector, Regime, RegimeState,
    classify_regime_at_bar, precompute_1h_indicators, precompute_4h_indicators,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake, _calculate_atr, _ema, _adx,
)
from multi_agent.agents.density_agent import (
    compute_volume_profile, compute_rolling_vwap as density_vwap,
    detect_density_signal, _rsi as density_rsi, _atr as density_atr,
    _adx as density_adx,
)
from multi_agent.agents.mean_reversion_agent import (
    compute_mr_indicators, detect_mr_signal,
)

# Reuse FVG backtest logic
from multi_agent.shared.fvg_core import (
    detect_fvgs_v2, _select_active_fvgs, price_in_mid_zone,
    composite_signal_score, evaluate_entry_reaction, compute_ema,
    build_symbol_profile, find_swing_levels, collect_swing_levels,
    build_local_volume_profile, extract_structural_levels,
    _atr_pct_from_arrays, FVG_CORE_CONFIG,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

# ─── Constants ───────────────────────────────────────────────────────────────

BACKTEST_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

START_BALANCE  = 1000.0
TAKER_FEE_PCT  = 0.00055
SLIPPAGE_PCT   = 0.00025
MAX_LEVERAGE   = 25
MAX_CONCURRENT_PER_AGENT = 3

# Per-agent risk
RISK_MAP = {
    "fvg": cfg.PORTFOLIO_FVG_RISK,
    "bbs": cfg.PORTFOLIO_BBS_RISK,
    "density": cfg.PORTFOLIO_DENSITY_RISK,
    "mr": cfg.PORTFOLIO_MR_RISK,
}

CAPITAL_MAP = {
    "fvg": cfg.PORTFOLIO_FVG_FRACTION,
    "bbs": cfg.PORTFOLIO_BBS_FRACTION,
    "density": cfg.PORTFOLIO_DENSITY_FRACTION,
    "mr": cfg.PORTFOLIO_MR_FRACTION,
}


# ─── Data loading ────────────────────────────────────────────────────────────

def _load_cached(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    safe = symbol.replace("/", "_").replace(":", "_")
    # Try CSV first (always available), then parquet
    for ext, reader in [(".csv", pd.read_csv), (".parquet", _safe_read_parquet)]:
        fpath = CACHE_DIR / f"{safe}_{timeframe}_90d{ext}"
        if fpath.exists():
            try:
                df = reader(fpath)
                if df is None or df.empty:
                    continue
                for col in ("open", "high", "low", "close", "volume"):
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                if "timestamp" in df.columns:
                    df["timestamp"] = df["timestamp"].astype(int)
                return df
            except Exception:
                continue
    return None


def _safe_read_parquet(path):
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  FVG Signal Detection (backtest version)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_fvg_signal_at_bar(df_15m: pd.DataFrame, df_1h: pd.DataFrame,
                                bar_idx: int, timestamps_15m: np.ndarray,
                                timestamps_1h: np.ndarray, ema200_1h: np.ndarray,
                                adx_15m: np.ndarray, ema200_4h: np.ndarray = None,
                                timestamps_4h: np.ndarray = None,
                                df_4h: pd.DataFrame = None,
                                # Pre-computed arrays (pass to avoid repeated conversion)
                                _closes_15m: np.ndarray = None,
                                _highs_15m: np.ndarray = None,
                                _lows_15m: np.ndarray = None,
                                _volumes_15m: np.ndarray = None,
                                _closes_1h: np.ndarray = None,
                                _closes_4h: np.ndarray = None,
                                ) -> Optional[dict]:
    """Check for FVG signal at bar_idx of df_15m. Returns signal dict or None."""
    i = bar_idx
    n = len(df_15m)
    if i < 250 or i >= n:
        return None

    # Use pre-computed arrays if provided, else extract (fallback)
    closes_15m = _closes_15m if _closes_15m is not None else df_15m["close"].astype(float).to_numpy()
    highs_15m = _highs_15m if _highs_15m is not None else df_15m["high"].astype(float).to_numpy()
    lows_15m = _lows_15m if _lows_15m is not None else df_15m["low"].astype(float).to_numpy()
    volumes_15m = _volumes_15m if _volumes_15m is not None else df_15m["volume"].astype(float).to_numpy()

    # HTF EMA-200 bias
    ts_i = int(timestamps_15m[i])
    htf_idx = int(np.searchsorted(timestamps_1h, ts_i, side="right")) - 1
    if htf_idx < 200 or htf_idx >= len(ema200_1h):
        return None
    if np.isnan(ema200_1h[htf_idx]):
        return None

    closes_1h = _closes_1h if _closes_1h is not None else df_1h["close"].astype(float).to_numpy()
    htf_close = closes_1h[htf_idx]
    htf_ema = ema200_1h[htf_idx]
    htf_bullish = htf_close > htf_ema
    htf_bearish = htf_close < htf_ema

    # 4h trend confirmation
    if ema200_4h is not None and timestamps_4h is not None:
        j4 = int(np.searchsorted(timestamps_4h, ts_i, side="right")) - 1
        if j4 >= 24 and j4 < len(ema200_4h) and not np.isnan(ema200_4h[j4]):
            closes_4h_arr = _closes_4h if _closes_4h is not None else df_4h["close"].astype(float).to_numpy()
            if htf_bullish and closes_4h_arr[j4] < ema200_4h[j4]:
                return None  # 4h contradicts
            if htf_bearish and closes_4h_arr[j4] > ema200_4h[j4]:
                return None

    # ADX filter on 15m
    if not np.isnan(adx_15m[i]) and adx_15m[i] < 20.0:
        return None

    # Volume filter
    if i >= 10:
        vol_avg = np.mean(volumes_15m[i-10:i])
        if vol_avg > 0 and volumes_15m[i] < 0.8 * vol_avg:
            return None

    # Detect FVGs on the lookback window ending at bar i
    window_start = max(0, i - 384)
    df_window = df_15m.iloc[window_start:i+1].copy().reset_index(drop=True)
    if len(df_window) < 50:
        return None

    all_fvgs = detect_fvgs_v2(
        df_window, lookback=min(384, len(df_window)),
        percentile_threshold=cfg.FVG_PERCENTILE_THRESHOLD,
        percentile_lookback=min(500, len(df_window)),
    )
    active_fvgs = _select_active_fvgs(all_fvgs)

    close_val = closes_15m[i]
    for fvg in active_fvgs:
        if fvg.type == "bullish" and not htf_bullish:
            continue
        if fvg.type == "bearish" and not htf_bearish:
            continue
        if fvg.quality_score < 55.0:
            continue

        # Mid-zone check
        gap_size = max(0.0, fvg.high - fvg.low)
        if gap_size <= 0:
            continue
        entry_band_low = fvg.low + gap_size * 0.35
        entry_band_high = fvg.low + gap_size * 0.65
        if not (entry_band_low <= close_val <= entry_band_high):
            continue

        # Composite score
        reaction_ok, reaction_score = evaluate_entry_reaction(df_window.tail(6), fvg)
        fvg.reaction_score = float(reaction_score)
        comp = composite_signal_score(fvg, reaction_score)
        if comp < 70.0 or comp > 80.0:
            continue

        # Fibonacci confluence (simplified: check if in 38.2-61.8% retracement)
        swing_low = float(np.min(lows_15m[max(0, i-120):i+1]))
        swing_high = float(np.max(highs_15m[max(0, i-120):i+1]))
        fib_range = swing_high - swing_low
        if fib_range > 0:
            fib_382 = swing_high - fib_range * 0.618
            fib_618 = swing_high - fib_range * 0.382
            if not (fib_382 <= close_val <= fib_618):
                continue

        # Build trade plan
        entry = close_val
        if fvg.type == "bullish":
            sl_level = float(np.min(lows_15m[max(0, i-60):i+1]))
            sl = sl_level * (1.0 - 0.0015)
        else:
            sl_level = float(np.max(highs_15m[max(0, i-60):i+1]))
            sl = sl_level * (1.0 + 0.0015)

        risk = abs(entry - sl)
        if risk <= 0 or risk / entry > 0.05:
            continue

        rr_min = 1.5
        direction = "LONG" if fvg.type == "bullish" else "SHORT"

        return {
            "direction": direction,
            "entry_price": float(entry),
            "sl": float(sl),
            "risk": float(risk),
            "composite": float(comp),
            "quality": float(fvg.quality_score),
            "bar_idx": i,
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Unified Backtest Engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_agent_backtest(agent_name: str, use_regime: bool = True) -> dict:
    """
    Run backtest for a single agent across all symbols.
    If use_regime=True, only enter when regime matches the agent's allowed regimes.
    """
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    trades = []
    trade_details = []
    wins = 0
    losses = 0
    regime_rejected = 0
    regime_distribution = defaultdict(int)

    risk_pct = RISK_MAP.get(agent_name, 0.01)
    capital_frac = CAPITAL_MAP.get(agent_name, 0.25)

    for symbol in BACKTEST_SYMBOLS:
        short = symbol.split("/")[0]
        df_15m = _load_cached(symbol, "15m")
        df_1h = _load_cached(symbol, "1h")
        df_4h = _load_cached(symbol, "4h")

        if df_15m is None or df_1h is None:
            continue
        if len(df_15m) < 250 or len(df_1h) < 210:
            continue

        # Pre-compute indicators
        closes_15m = df_15m["close"].astype(float).to_numpy()
        highs_15m = df_15m["high"].astype(float).to_numpy()
        lows_15m = df_15m["low"].astype(float).to_numpy()
        volumes_15m = df_15m["volume"].astype(float).to_numpy()
        timestamps_15m = df_15m["timestamp"].astype(int).to_numpy()
        timestamps_1h = df_1h["timestamp"].astype(int).to_numpy()
        n = len(df_15m)

        # 1h indicators for regime
        closes_1h_arr = df_1h["close"].astype(float).to_numpy()
        regime_ind = precompute_1h_indicators(df_1h)
        ema200_1h = _ema(closes_1h_arr, 200)

        # 4h indicators
        regime_4h = None
        ema200_4h = None
        timestamps_4h = None
        closes_4h_arr = None
        if df_4h is not None and len(df_4h) > 50:
            regime_4h = precompute_4h_indicators(df_4h)
            ema200_4h = regime_4h["ema200_4h"]
            timestamps_4h = df_4h["timestamp"].astype(int).to_numpy()
            closes_4h_arr = df_4h["close"].astype(float).to_numpy()

        # Agent-specific indicators
        if agent_name == "bbs":
            squeeze_ind = compute_squeeze_indicators(df_15m)
        elif agent_name == "density":
            rsi_arr = density_rsi(closes_15m, cfg.DENSITY_RSI_PERIOD)
            adx_arr = density_adx(highs_15m, lows_15m, closes_15m, 14)
            atr_arr = density_atr(highs_15m, lows_15m, closes_15m, 14)
            vwap_arr = density_vwap(highs_15m, lows_15m, closes_15m, volumes_15m, 96)
        elif agent_name == "mr":
            mr_ind = compute_mr_indicators(df_15m)
        elif agent_name == "fvg":
            adx_15m = _adx(highs_15m, lows_15m, closes_15m, 14)

        # Regime detector (per-symbol)
        detector = RegimeDetector()

        sym_pos = None
        mr_consec_losses = {"LONG": 0, "SHORT": 0}

        for i in range(210, n):
            close_i = closes_15m[i]
            high_i = highs_15m[i]
            low_i = lows_15m[i]
            ts_i = int(timestamps_15m[i])

            # Determine regime at this bar
            htf_idx = int(np.searchsorted(timestamps_1h, ts_i, side="right")) - 1
            if htf_idx < 210:
                continue

            regime_state = detector.update(
                symbol, df_1h, htf_idx, df_4h,
                **regime_ind,
                **(regime_4h if regime_4h else {}),
            )
            current_regime = regime_state.regime
            regime_distribution[current_regime.value] += 1

            # ── Check open position ──
            if sym_pos is not None:
                direction = sym_pos["direction"]
                entry = sym_pos["entry"]
                sl = sym_pos["sl"]
                risk = sym_pos["risk"]
                qty = sym_pos["qty"]

                # Update trailing for BBS
                if agent_name == "bbs" and sym_pos.get("trailing_sl") is not None:
                    new_ce = chandelier_exit_sl(
                        highs_15m, lows_15m, squeeze_ind["ce_atr"],
                        direction, i, ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                        ce_mult=cfg.BB_SQUEEZE_CE_MULT,
                    )
                    if direction == "LONG":
                        pnl_r = (close_i - entry) / risk if risk > 0 else 0
                        if pnl_r >= 1.0:
                            sym_pos["trailing_sl"] = max(sym_pos["trailing_sl"], new_ce)
                        effective_sl = max(sl, sym_pos["trailing_sl"])
                        if low_i <= effective_sl:
                            exit_p = effective_sl
                            pnl = (exit_p - entry) * qty
                            fee = exit_p * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            if pnl > 0: wins += 1
                            else: losses += 1
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                        if (close_i - entry) / risk >= 4.0:
                            pnl = (close_i - entry) * qty
                            fee = close_i * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            wins += 1
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                    else:  # SHORT BBS
                        pnl_r = (entry - close_i) / risk if risk > 0 else 0
                        if pnl_r >= 1.0:
                            new_trail = min(
                                sym_pos["trailing_sl"] if sym_pos["trailing_sl"] > 0 else float("inf"),
                                new_ce)
                            sym_pos["trailing_sl"] = new_trail
                        effective_sl = sym_pos["trailing_sl"] if sym_pos["trailing_sl"] > 0 else sl
                        if high_i >= effective_sl:
                            exit_p = effective_sl
                            pnl = (entry - exit_p) * qty
                            fee = exit_p * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            if pnl > 0: wins += 1
                            else: losses += 1
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                        if pnl_r >= 4.0:
                            pnl = (entry - close_i) * qty
                            fee = close_i * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            wins += 1
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                else:
                    # FVG / Density / MR: simple SL/TP
                    if direction == "LONG":
                        if low_i <= sl:
                            pnl = (sl - entry) * qty
                            fee = sl * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            losses += 1
                            if agent_name == "mr":
                                mr_consec_losses["LONG"] = mr_consec_losses.get("LONG", 0) + 1
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                        tp = sym_pos.get("tp", 0)
                        if tp > 0 and high_i >= tp:
                            pnl = (tp - entry) * qty
                            fee = tp * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            wins += 1
                            if agent_name == "mr":
                                mr_consec_losses["LONG"] = 0
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                        # FVG Chandelier trailing
                        if agent_name == "fvg" and sym_pos.get("use_chandelier"):
                            ce_sl = chandelier_exit_sl(
                                highs_15m, lows_15m,
                                _calculate_atr(highs_15m, lows_15m, closes_15m, 22),
                                "LONG", i, ce_length=22, ce_mult=3.0)
                            pnl_r = (close_i - entry) / risk if risk > 0 else 0
                            if pnl_r >= 1.0:
                                sym_pos["trailing_sl"] = max(sym_pos.get("trailing_sl", sl), ce_sl)
                            eff = max(sl, sym_pos.get("trailing_sl", sl))
                            if low_i <= eff:
                                pnl = (eff - entry) * qty
                                fee = eff * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                                balance += pnl - fee
                                trades.append(pnl - fee)
                                if pnl > 0: wins += 1
                                else: losses += 1
                                trade_details.append({"symbol": symbol, "agent": agent_name,
                                    "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                                sym_pos = None
                                continue
                    else:  # SHORT
                        if high_i >= sl:
                            pnl = (entry - sl) * qty
                            fee = sl * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            losses += 1
                            if agent_name == "mr":
                                mr_consec_losses["SHORT"] = mr_consec_losses.get("SHORT", 0) + 1
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                        tp = sym_pos.get("tp", 0)
                        if tp > 0 and low_i <= tp:
                            pnl = (entry - tp) * qty
                            fee = tp * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee
                            trades.append(pnl - fee)
                            wins += 1
                            if agent_name == "mr":
                                mr_consec_losses["SHORT"] = 0
                            trade_details.append({"symbol": symbol, "agent": agent_name,
                                "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                            sym_pos = None
                            continue
                        if agent_name == "fvg" and sym_pos.get("use_chandelier"):
                            ce_sl = chandelier_exit_sl(
                                highs_15m, lows_15m,
                                _calculate_atr(highs_15m, lows_15m, closes_15m, 22),
                                "SHORT", i, ce_length=22, ce_mult=3.0)
                            pnl_r = (entry - close_i) / risk if risk > 0 else 0
                            if pnl_r >= 1.0:
                                prev = sym_pos.get("trailing_sl", sl)
                                sym_pos["trailing_sl"] = min(prev if prev > 0 else float("inf"), ce_sl)
                            eff = sym_pos.get("trailing_sl", sl)
                            if eff <= 0: eff = sl
                            if high_i >= eff:
                                pnl = (entry - eff) * qty
                                fee = eff * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                                balance += pnl - fee
                                trades.append(pnl - fee)
                                if pnl > 0: wins += 1
                                else: losses += 1
                                trade_details.append({"symbol": symbol, "agent": agent_name,
                                    "direction": direction, "pnl": pnl - fee, "bars": i - sym_pos["bar_idx"]})
                                sym_pos = None
                                continue

                peak = max(peak, balance)
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                continue  # in position, skip entry

            # ── No position: check for entry ──
            # Regime gating
            if use_regime:
                allowed = False
                if agent_name == "fvg" and current_regime == Regime.TREND:
                    allowed = True
                elif agent_name == "fvg" and current_regime == Regime.SQUEEZE and regime_state.score_trend > 2.0:
                    allowed = True
                elif agent_name == "bbs" and current_regime == Regime.SQUEEZE:
                    allowed = True
                elif agent_name == "bbs" and current_regime == Regime.TREND and regime_state.kc_squeeze_active:
                    allowed = True
                elif agent_name == "density" and current_regime == Regime.RANGE:
                    allowed = True
                elif agent_name == "mr" and current_regime == Regime.RANGE:
                    allowed = True

                if not allowed:
                    regime_rejected += 1
                    continue

            # Agent-specific signal detection
            signal = None

            if agent_name == "fvg":
                signal = _detect_fvg_signal_at_bar(
                    df_15m, df_1h, i, timestamps_15m, timestamps_1h, ema200_1h,
                    adx_15m, ema200_4h, timestamps_4h, df_4h,
                    _closes_15m=closes_15m, _highs_15m=highs_15m,
                    _lows_15m=lows_15m, _volumes_15m=volumes_15m,
                    _closes_1h=closes_1h_arr, _closes_4h=closes_4h_arr)
                if signal:
                    signal["use_chandelier"] = True
                    signal["trailing_sl"] = signal["sl"]

            elif agent_name == "bbs":
                sig = detect_squeeze_signal(
                    squeeze_ind, bar_idx=i,
                    min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                    adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                    vol_mult=cfg.BB_SQUEEZE_VOL_MULT,
                )
                if sig:
                    # HTF alignment
                    htf_direction = "LONG" if closes_1h_arr[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                    if sig["direction"] == htf_direction:
                        # Head-fake check
                        if not check_head_fake(squeeze_ind, i, check_bars=2):
                            # Late-expansion filter: skip if ADX > 40
                            if sig["adx"] <= 40:
                                sl = chandelier_exit_sl(
                                    highs_15m, lows_15m, squeeze_ind["ce_atr"],
                                    sig["direction"], i,
                                    ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                                    ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                                risk = abs(close_i - sl)
                                if 0 < risk / close_i <= 0.05:
                                    signal = {
                                        "direction": sig["direction"],
                                        "entry_price": float(close_i),
                                        "sl": float(sl),
                                        "risk": float(risk),
                                        "trailing_sl": float(sl),
                                        "bar_idx": i,
                                    }

            elif agent_name == "density":
                # Recompute volume profile for lookback window
                vp_start = max(0, i - cfg.DENSITY_VP_LOOKBACK_BARS)
                vp = compute_volume_profile(
                    highs_15m, lows_15m, closes_15m, volumes_15m,
                    num_bins=cfg.DENSITY_VP_NUM_BINS,
                    start_idx=vp_start, end_idx=i+1)
                sig = detect_density_signal(
                    closes_15m, highs_15m, lows_15m, volumes_15m, i,
                    vp, vwap_arr, rsi_arr, adx_arr, atr_arr)
                if sig:
                    signal = {
                        "direction": sig["direction"],
                        "entry_price": sig["entry_price"],
                        "sl": sig["sl"],
                        "risk": abs(sig["entry_price"] - sig["sl"]),
                        "tp": sig["tp1"],
                        "bar_idx": i,
                    }

            elif agent_name == "mr":
                sig = detect_mr_signal(mr_ind, i, mr_consec_losses)
                if sig:
                    signal = {
                        "direction": sig["direction"],
                        "entry_price": sig["entry_price"],
                        "sl": sig["sl"],
                        "risk": abs(sig["entry_price"] - sig["sl"]),
                        "tp": sig["tp"],
                        "bar_idx": i,
                    }

            if signal is None:
                continue

            # Position sizing
            agent_balance = balance * capital_frac
            risk_usd = agent_balance * risk_pct
            risk_dist = signal["risk"]
            if risk_dist <= 0:
                continue
            qty = risk_usd / risk_dist
            margin = signal["entry_price"] * qty / MAX_LEVERAGE
            if margin > agent_balance * 0.25:
                qty = agent_balance * 0.25 * MAX_LEVERAGE / signal["entry_price"]
            if qty <= 0:
                continue

            # Entry fee
            fee = signal["entry_price"] * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
            balance -= fee

            sym_pos = {
                "direction": signal["direction"],
                "entry": signal["entry_price"],
                "sl": signal["sl"],
                "risk": risk_dist,
                "qty": qty,
                "bar_idx": i,
                "tp": signal.get("tp", 0),
                "trailing_sl": signal.get("trailing_sl", signal["sl"]),
                "use_chandelier": signal.get("use_chandelier", False),
            }

        # Close remaining position
        if sym_pos is not None:
            last_close = closes_15m[-1]
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

    avg_bars = np.mean([t["bars"] for t in trade_details]) if trade_details else 0

    result = {
        "agent": agent_name,
        "regime_gated": use_regime,
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
        "regime_rejected": regime_rejected,
        "regime_distribution": dict(regime_distribution),
        "avg_hold_bars": round(avg_bars, 1),
    }

    return result


def run_combined_portfolio(use_regime: bool = True) -> dict:
    """
    Run all 4 agents simultaneously with cross-agent conflict prevention.
    """
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    all_trades = []
    per_agent_stats = {a: {"wins": 0, "losses": 0, "trades": [], "rejected": 0} for a in ["fvg", "bbs", "density", "mr"]}

    for symbol in BACKTEST_SYMBOLS:
        short = symbol.split("/")[0]
        df_15m = _load_cached(symbol, "15m")
        df_1h = _load_cached(symbol, "1h")
        df_4h = _load_cached(symbol, "4h")
        if df_15m is None or df_1h is None:
            continue
        if len(df_15m) < 250 or len(df_1h) < 210:
            continue

        closes_15m = df_15m["close"].astype(float).to_numpy()
        highs_15m = df_15m["high"].astype(float).to_numpy()
        lows_15m = df_15m["low"].astype(float).to_numpy()
        volumes_15m = df_15m["volume"].astype(float).to_numpy()
        timestamps_15m = df_15m["timestamp"].astype(int).to_numpy()
        timestamps_1h = df_1h["timestamp"].astype(int).to_numpy()
        n = len(df_15m)

        closes_1h_arr = df_1h["close"].astype(float).to_numpy()
        regime_ind = precompute_1h_indicators(df_1h)
        ema200_1h = _ema(closes_1h_arr, 200)

        regime_4h = None
        ema200_4h = None
        timestamps_4h = None
        closes_4h_arr = None
        if df_4h is not None and len(df_4h) > 50:
            regime_4h = precompute_4h_indicators(df_4h)
            ema200_4h = regime_4h["ema200_4h"]
            timestamps_4h = df_4h["timestamp"].astype(int).to_numpy()
            closes_4h_arr = df_4h["close"].astype(float).to_numpy()

        # Pre-compute all agent indicators
        squeeze_ind = compute_squeeze_indicators(df_15m)
        adx_15m = _adx(highs_15m, lows_15m, closes_15m, 14)
        rsi_arr = density_rsi(closes_15m, cfg.DENSITY_RSI_PERIOD)
        adx_dens = density_adx(highs_15m, lows_15m, closes_15m, 14)
        atr_dens = density_atr(highs_15m, lows_15m, closes_15m, 14)
        vwap_arr = density_vwap(highs_15m, lows_15m, closes_15m, volumes_15m, 96)
        mr_ind = compute_mr_indicators(df_15m)

        detector = RegimeDetector()
        sym_pos = None  # only one position per symbol across all agents
        pos_agent = None
        mr_consec = {"LONG": 0, "SHORT": 0}

        for i in range(210, n):
            close_i = closes_15m[i]
            high_i = highs_15m[i]
            low_i = lows_15m[i]
            ts_i = int(timestamps_15m[i])

            htf_idx = int(np.searchsorted(timestamps_1h, ts_i, side="right")) - 1
            if htf_idx < 210:
                continue

            regime_state = detector.update(
                symbol, df_1h, htf_idx, df_4h,
                **regime_ind, **(regime_4h if regime_4h else {}))
            current_regime = regime_state.regime

            # Check open position
            if sym_pos is not None:
                d = sym_pos["direction"]
                entry = sym_pos["entry"]
                sl = sym_pos["sl"]
                risk = sym_pos["risk"]
                qty = sym_pos["qty"]
                agent = pos_agent

                # BBS trailing
                if agent == "bbs":
                    new_ce = chandelier_exit_sl(highs_15m, lows_15m, squeeze_ind["ce_atr"],
                        d, i, ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                    if d == "LONG":
                        pnl_r = (close_i - entry) / risk if risk > 0 else 0
                        if pnl_r >= 1.0:
                            sym_pos["trailing_sl"] = max(sym_pos.get("trailing_sl", sl), new_ce)
                        eff_sl = max(sl, sym_pos.get("trailing_sl", sl))
                        if low_i <= eff_sl or pnl_r >= 4.0:
                            exit_p = eff_sl if low_i <= eff_sl else close_i
                            pnl = (exit_p - entry) * qty
                            fee = exit_p * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee; all_trades.append(pnl - fee)
                            per_agent_stats[agent]["trades"].append(pnl - fee)
                            if pnl > 0: per_agent_stats[agent]["wins"] += 1
                            else: per_agent_stats[agent]["losses"] += 1
                            sym_pos = None; pos_agent = None; continue
                    else:
                        pnl_r = (entry - close_i) / risk if risk > 0 else 0
                        if pnl_r >= 1.0:
                            prev = sym_pos.get("trailing_sl", sl)
                            sym_pos["trailing_sl"] = min(prev if prev > 0 else float("inf"), new_ce)
                        eff_sl = sym_pos.get("trailing_sl", sl)
                        if eff_sl <= 0: eff_sl = sl
                        if high_i >= eff_sl or pnl_r >= 4.0:
                            exit_p = eff_sl if high_i >= eff_sl else close_i
                            pnl = (entry - exit_p) * qty
                            fee = exit_p * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                            balance += pnl - fee; all_trades.append(pnl - fee)
                            per_agent_stats[agent]["trades"].append(pnl - fee)
                            if pnl > 0: per_agent_stats[agent]["wins"] += 1
                            else: per_agent_stats[agent]["losses"] += 1
                            sym_pos = None; pos_agent = None; continue
                else:
                    # Simple SL/TP for FVG, Density, MR (+ FVG chandelier)
                    closed = False
                    if d == "LONG":
                        if low_i <= sl:
                            pnl = (sl - entry) * qty; closed = True; exit_p = sl
                        elif sym_pos.get("tp", 0) > 0 and high_i >= sym_pos["tp"]:
                            pnl = (sym_pos["tp"] - entry) * qty; closed = True; exit_p = sym_pos["tp"]
                        elif agent == "fvg":
                            ce_atr = _calculate_atr(highs_15m, lows_15m, closes_15m, 22)
                            ce_sl = chandelier_exit_sl(highs_15m, lows_15m, ce_atr, "LONG", i, 22, 3.0)
                            pnl_r = (close_i - entry) / risk if risk > 0 else 0
                            if pnl_r >= 1.0:
                                sym_pos["trailing_sl"] = max(sym_pos.get("trailing_sl", sl), ce_sl)
                            eff = max(sl, sym_pos.get("trailing_sl", sl))
                            if low_i <= eff:
                                pnl = (eff - entry) * qty; closed = True; exit_p = eff
                    else:
                        if high_i >= sl:
                            pnl = (entry - sl) * qty; closed = True; exit_p = sl
                        elif sym_pos.get("tp", 0) > 0 and low_i <= sym_pos["tp"]:
                            pnl = (entry - sym_pos["tp"]) * qty; closed = True; exit_p = sym_pos["tp"]
                        elif agent == "fvg":
                            ce_atr = _calculate_atr(highs_15m, lows_15m, closes_15m, 22)
                            ce_sl = chandelier_exit_sl(highs_15m, lows_15m, ce_atr, "SHORT", i, 22, 3.0)
                            pnl_r = (entry - close_i) / risk if risk > 0 else 0
                            if pnl_r >= 1.0:
                                prev = sym_pos.get("trailing_sl", sl)
                                sym_pos["trailing_sl"] = min(prev if prev > 0 else float("inf"), ce_sl)
                            eff = sym_pos.get("trailing_sl", sl)
                            if eff <= 0: eff = sl
                            if high_i >= eff:
                                pnl = (entry - eff) * qty; closed = True; exit_p = eff

                    if closed:
                        fee = exit_p * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                        balance += pnl - fee; all_trades.append(pnl - fee)
                        per_agent_stats[agent]["trades"].append(pnl - fee)
                        if pnl > 0: per_agent_stats[agent]["wins"] += 1
                        else: per_agent_stats[agent]["losses"] += 1
                        if agent == "mr":
                            if pnl < 0: mr_consec[d] = mr_consec.get(d, 0) + 1
                            else: mr_consec[d] = 0
                        sym_pos = None; pos_agent = None; continue

                peak = max(peak, balance)
                dd = (peak - balance) / peak * 100 if peak > 0 else 0
                max_dd = max(max_dd, dd)
                continue

            # ── No position: try each agent in priority order ──
            if not use_regime:
                agents_to_try = ["fvg", "bbs", "density", "mr"]
            elif current_regime == Regime.TREND:
                agents_to_try = ["fvg"]
                if regime_state.kc_squeeze_active:
                    agents_to_try.append("bbs")
            elif current_regime == Regime.RANGE:
                agents_to_try = ["density", "mr"]
            elif current_regime == Regime.SQUEEZE:
                agents_to_try = ["bbs"]
                if regime_state.score_trend > 2.0:
                    agents_to_try.append("fvg")
            else:
                per_agent_stats["fvg"]["rejected"] += 1
                continue

            for agent_name in agents_to_try:
                signal = None

                if agent_name == "fvg":
                    signal = _detect_fvg_signal_at_bar(
                        df_15m, df_1h, i, timestamps_15m, timestamps_1h, ema200_1h,
                        adx_15m, ema200_4h, timestamps_4h, df_4h,
                        _closes_15m=closes_15m, _highs_15m=highs_15m,
                        _lows_15m=lows_15m, _volumes_15m=volumes_15m,
                        _closes_1h=closes_1h_arr, _closes_4h=closes_4h_arr)
                    if signal:
                        signal["use_chandelier"] = True
                        signal["trailing_sl"] = signal["sl"]

                elif agent_name == "bbs":
                    sig = detect_squeeze_signal(squeeze_ind, bar_idx=i,
                        min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                        adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                        vol_mult=cfg.BB_SQUEEZE_VOL_MULT)
                    if sig:
                        htf_dir = "LONG" if closes_1h_arr[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                        if sig["direction"] == htf_dir and not check_head_fake(squeeze_ind, i, 2):
                            if sig["adx"] <= 40:
                                sl = chandelier_exit_sl(highs_15m, lows_15m, squeeze_ind["ce_atr"],
                                    sig["direction"], i, cfg.BB_SQUEEZE_CE_LENGTH, cfg.BB_SQUEEZE_CE_MULT)
                                risk = abs(close_i - sl)
                                if 0 < risk / close_i <= 0.05:
                                    signal = {"direction": sig["direction"], "entry_price": float(close_i),
                                        "sl": float(sl), "risk": float(risk), "trailing_sl": float(sl), "bar_idx": i}

                elif agent_name == "density":
                    vp_start = max(0, i - cfg.DENSITY_VP_LOOKBACK_BARS)
                    vp = compute_volume_profile(highs_15m, lows_15m, closes_15m, volumes_15m,
                        num_bins=cfg.DENSITY_VP_NUM_BINS, start_idx=vp_start, end_idx=i+1)
                    sig = detect_density_signal(closes_15m, highs_15m, lows_15m, volumes_15m, i,
                        vp, vwap_arr, rsi_arr, adx_dens, atr_dens)
                    if sig:
                        signal = {"direction": sig["direction"], "entry_price": sig["entry_price"],
                            "sl": sig["sl"], "risk": abs(sig["entry_price"] - sig["sl"]),
                            "tp": sig["tp1"], "bar_idx": i}

                elif agent_name == "mr":
                    sig = detect_mr_signal(mr_ind, i, mr_consec)
                    if sig:
                        signal = {"direction": sig["direction"], "entry_price": sig["entry_price"],
                            "sl": sig["sl"], "risk": abs(sig["entry_price"] - sig["sl"]),
                            "tp": sig["tp"], "bar_idx": i}

                if signal is None:
                    continue

                # Position sizing
                agent_frac = CAPITAL_MAP.get(agent_name, 0.25)
                agent_risk = RISK_MAP.get(agent_name, 0.01)
                agent_bal = balance * agent_frac
                risk_usd = agent_bal * agent_risk
                risk_dist = signal["risk"]
                if risk_dist <= 0:
                    continue
                qty = risk_usd / risk_dist
                margin = signal["entry_price"] * qty / MAX_LEVERAGE
                if margin > agent_bal * 0.25:
                    qty = agent_bal * 0.25 * MAX_LEVERAGE / signal["entry_price"]
                if qty <= 0:
                    continue

                fee = signal["entry_price"] * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
                balance -= fee

                sym_pos = {
                    "direction": signal["direction"],
                    "entry": signal["entry_price"],
                    "sl": signal["sl"],
                    "risk": risk_dist,
                    "qty": qty,
                    "bar_idx": i,
                    "tp": signal.get("tp", 0),
                    "trailing_sl": signal.get("trailing_sl", signal["sl"]),
                    "use_chandelier": signal.get("use_chandelier", False),
                }
                pos_agent = agent_name
                break  # one entry per bar per symbol

        # Close remaining
        if sym_pos is not None:
            last_close = closes_15m[-1]
            entry = sym_pos["entry"]
            qty = sym_pos["qty"]
            if sym_pos["direction"] == "LONG":
                pnl = (last_close - entry) * qty
            else:
                pnl = (entry - last_close) * qty
            fee = last_close * qty * (TAKER_FEE_PCT + SLIPPAGE_PCT)
            balance += pnl - fee
            all_trades.append(pnl - fee)
            if pos_agent:
                per_agent_stats[pos_agent]["trades"].append(pnl - fee)
                if pnl > 0: per_agent_stats[pos_agent]["wins"] += 1
                else: per_agent_stats[pos_agent]["losses"] += 1

    # Final stats
    total_trades = len(all_trades)
    total_wins = sum(s["wins"] for s in per_agent_stats.values())
    total_losses = sum(s["losses"] for s in per_agent_stats.values())
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100.0
    wr = (total_wins / total_trades * 100.0) if total_trades > 0 else 0.0

    win_t = [t for t in all_trades if t > 0]
    loss_t = [t for t in all_trades if t <= 0]
    gp = sum(win_t); gl = abs(sum(loss_t))
    pf = gp / gl if gl > 0 else float("inf")

    peak = max(peak, balance)
    dd = (peak - balance) / peak * 100 if peak > 0 else 0
    max_dd = max(max_dd, dd)

    result = {
        "mode": "combined" + ("_regime" if use_regime else "_no_regime"),
        "trades": total_trades,
        "wins": total_wins,
        "losses": total_losses,
        "wr_pct": round(wr, 1),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "final_balance": round(balance, 2),
        "per_agent": {},
    }

    for a, s in per_agent_stats.items():
        t = s["wins"] + s["losses"]
        wt = [x for x in s["trades"] if x > 0]
        lt = [x for x in s["trades"] if x <= 0]
        gp_a = sum(wt); gl_a = abs(sum(lt))
        result["per_agent"][a] = {
            "trades": t,
            "wins": s["wins"],
            "losses": s["losses"],
            "wr_pct": round(s["wins"] / t * 100 if t > 0 else 0, 1),
            "pnl_usd": round(sum(s["trades"]), 2),
            "profit_factor": round(gp_a / gl_a if gl_a > 0 else float("inf"), 2),
            "rejected": s["rejected"],
        }

    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  Main — run all backtests and print report
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 90)
    logger.info("REGIME-AWARE PORTFOLIO BACKTESTER — 20 Symbols × 90 Days")
    logger.info("=" * 90)

    results = {}

    # 1. Individual agents with regime gating
    for agent in ["fvg", "bbs", "density", "mr"]:
        logger.info("--- Running %s standalone (regime-gated) ---", agent.upper())
        t0 = _time.time()
        r = run_single_agent_backtest(agent, use_regime=True)
        elapsed = _time.time() - t0
        results[f"{agent}_regime"] = r
        logger.info(
            "%-12s | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
            agent.upper(), r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

    # 2. BB Squeeze WITHOUT regime (ablation)
    logger.info("--- Running BBS standalone (NO regime gate — ablation) ---")
    t0 = _time.time()
    r = run_single_agent_backtest("bbs", use_regime=False)
    elapsed = _time.time() - t0
    results["bbs_no_regime"] = r
    logger.info(
        "BBS_NOREG    | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
        r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

    # 3. FVG WITHOUT regime (ablation)
    logger.info("--- Running FVG standalone (NO regime gate — ablation) ---")
    t0 = _time.time()
    r = run_single_agent_backtest("fvg", use_regime=False)
    elapsed = _time.time() - t0
    results["fvg_no_regime"] = r
    logger.info(
        "FVG_NOREG    | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
        r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

    # 4. Combined portfolio WITH regime
    logger.info("--- Running COMBINED portfolio (regime-aware) ---")
    t0 = _time.time()
    r = run_combined_portfolio(use_regime=True)
    elapsed = _time.time() - t0
    results["combined_regime"] = r
    logger.info(
        "COMBINED_REG | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
        r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

    # 5. Combined portfolio WITHOUT regime (ablation)
    logger.info("--- Running COMBINED portfolio (NO regime — ablation) ---")
    t0 = _time.time()
    r = run_combined_portfolio(use_regime=False)
    elapsed = _time.time() - t0
    results["combined_no_regime"] = r
    logger.info(
        "COMBINED_NOR | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
        r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

    # ── Print report ──
    print("\n" + "=" * 110)
    print("REGIME-AWARE PORTFOLIO BACKTEST — FINAL REPORT")
    print("=" * 110)

    header = f"{'Strategy':<25} {'Trades':>6} {'WR%':>6} {'PnL%':>9} {'MaxDD%':>8} {'PF':>6} {'Balance':>10} {'Rejected':>8}"
    print(header)
    print("-" * 110)

    order = ["fvg_regime", "bbs_regime", "density_regime", "mr_regime",
             "fvg_no_regime", "bbs_no_regime",
             "combined_regime", "combined_no_regime"]

    for key in order:
        r = results.get(key, {})
        if not r:
            continue
        label = key.replace("_", " ").upper()
        rej = r.get("regime_rejected", r.get("per_agent", {}).get("fvg", {}).get("rejected", ""))
        print(f"{label:<25} {r.get('trades', 0):>6} {r.get('wr_pct', 0):>5.1f}% "
              f"{r.get('pnl_pct', 0):>+8.2f}% {r.get('max_dd_pct', 0):>7.2f}% "
              f"{r.get('profit_factor', 0):>5.2f} ${r.get('final_balance', 0):>9.2f} {str(rej):>8}")

    # Per-agent breakdown for combined
    print("\n" + "-" * 80)
    print("PER-AGENT BREAKDOWN (Combined Regime Portfolio)")
    print("-" * 80)
    combined = results.get("combined_regime", {}).get("per_agent", {})
    for agent, stats in combined.items():
        print(f"  {agent.upper():<15} Trades={stats['trades']:>3} WR={stats['wr_pct']:>5.1f}% "
              f"PnL=${stats['pnl_usd']:>+8.2f} PF={stats['profit_factor']:>5.2f}")

    # Save to JSON
    report_path = RESULTS_DIR / f"portfolio_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nFull results saved to: {report_path}")

    return results


if __name__ == "__main__":
    main()
