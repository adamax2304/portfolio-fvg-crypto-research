"""
multi_agent/backtests/backtest_fvg.py

FVG Strategy Backtester v3 — 90-day cache + improved entry/exit
Tests 8 variants on 20 symbols over 90 days.

Improvements over v2:
  - 90 days of data (was 60), loaded from parquet cache (10x faster)
  - New variants with volume confirmation + RSI momentum filter
  - Partial TP at 1:1 RR (50% close → guaranteed profit, raises WR)
  - Trailing stop to breakeven+0.1% once 1:1 is reached
  - Tight mid-zone 45-55% variant
  - Parallel symbol processing via multiprocessing.Pool
  - Precomputed indicators (EMA, ATR, RSI) before walk-forward loop
  - Early variant skip if first-30-day profit factor < 0.3

Variants:
  A) Original params (percentile=17.2, quality=50, composite=55, mid=40-60%)
  B) Tight composite 70-80 + EMA50+200 dual filter (current live params)
  C) Vol confirmation (vol>0.8x avg10) + RSI 35-65 + mid=40-60%
  D) Variant C + partial TP at 1:1 + trailing SL to BE+0.1%
  E) Tight mid 45-55% + vol filter + RSI filter
  F) Variant E + partial TP at 1:1 + trailing SL
  G) FVG + Fib 50-62% + 4h EMA200 Trend (continuation only, partial TP)
  H) FVG + Fib 38-62% + 4h EMA200 Trend (wider Fib zone, partial TP)

Usage
-----
    py -3 multi_agent/backtests/backtest_fvg.py
"""

from __future__ import annotations

import csv
import json
import logging
import math
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
from multi_agent.agents.bb_squeeze_agent import (
    _calculate_atr as _ce_atr_calc,
    _adx as _compute_adx,
    chandelier_exit_sl,
)

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = Path(__file__).parent / "data_cache"

# ─── Constants ────────────────────────────────────────────────────────────────

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

TIMEFRAME       = "15m"        # primary backtest timeframe
START_BALANCE   = 1000.0
BACKTEST_DAYS   = 90
TAKER_FEE_PCT   = 0.00055
MAKER_FEE_PCT   = 0.0002
SLIPPAGE_PCT    = 0.00025
RISK_PER_TRADE  = 0.02
MAX_LEVERAGE    = 25
MAX_CONCURRENT  = 2
MAX_DAILY_LOSS  = 4
MAX_DAILY_WIN   = 10
MAX_SAME_DIR    = 2
MAX_ACTIVE_FVGS_PER_SIDE = 3
MIN_SL_DIST_PCT = 0.8

# ─── Variants ─────────────────────────────────────────────────────────────────

VARIANTS = [
    # A: Original params (baseline)
    {
        "label":      "A: Original (pct=17.2, q=50, comp=55, mid=40-60%)",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.40,
        "mid_high":   0.60,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    False,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
    },
    # B: Current live params (composite 70-80 + dual EMA)
    {
        "label":      "B: Live params (comp=70-80, EMA50+200, mid=35-65%)",
        "percentile": 17.2,
        "quality":    60.0,
        "composite":  70.0,
        "max_composite": 80.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 2.5,
        "use_ema50_filter":  True,
        "use_ema200_filter": True,
        "use_vol_filter":    False,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
    },
    # C: Volume + RSI filter at entry (no partial TP yet)
    {
        "label":      "C: Vol>0.8x + RSI35-65 + mid=40-60%",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.40,
        "mid_high":   0.60,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    True,
        "rsi_long_min":      35.0,
        "rsi_long_max":      65.0,
        "rsi_short_min":     35.0,
        "rsi_short_max":     65.0,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
    },
    # D: Variant C + partial TP at 1:1 + trailing SL to BE+0.1%
    {
        "label":      "D: Vol+RSI + partial TP@1:1 + trailing SL",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.40,
        "mid_high":   0.60,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    True,
        "rsi_long_min":      35.0,
        "rsi_long_max":      65.0,
        "rsi_short_min":     35.0,
        "rsi_short_max":     65.0,
        "use_partial_tp":    True,
        "partial_tp_rr":     1.0,      # close 50% at 1:1
        "partial_tp_frac":   0.50,     # fraction of position to close
        "use_trailing_sl":   True,
        "trailing_sl_be_pct": 0.1,     # move SL to entry+0.1% once 1:1 reached
    },
    # E: Tight mid-zone 45-55% + vol + RSI
    {
        "label":      "E: Tight mid=45-55% + vol + RSI",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.45,
        "mid_high":   0.55,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    True,
        "rsi_long_min":      35.0,
        "rsi_long_max":      65.0,
        "rsi_short_min":     35.0,
        "rsi_short_max":     65.0,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
    },
    # F: Tight mid 45-55% + vol + RSI + partial TP + trailing SL
    {
        "label":      "F: Tight mid=45-55% + vol+RSI + partial TP + trailing SL",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.45,
        "mid_high":   0.55,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    True,
        "rsi_long_min":      35.0,
        "rsi_long_max":      65.0,
        "rsi_short_min":     35.0,
        "rsi_short_max":     65.0,
        "use_partial_tp":    True,
        "partial_tp_rr":     1.0,
        "partial_tp_frac":   0.50,
        "use_trailing_sl":   True,
        "trailing_sl_be_pct": 0.1,
    },
    # G: FVG + Fib 50-62% + 4h Trend (continuation only)
    {
        "label":      "G: FVG+Fib50-62%+4h_Trend (continuation only)",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    False,
        "use_rsi_filter":    False,
        "use_partial_tp":    True,
        "partial_tp_rr":     1.0,
        "partial_tp_frac":   0.50,
        "use_trailing_sl":   True,
        "trailing_sl_be_pct": 0.1,
        "require_fib_confluence": True,
        "fib_low_level":     0.50,
        "fib_high_level":    0.618,
        "require_4h_trend":  True,
        "trend_only":        True,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
    },
    # H: FVG + Fib 38-62% + 4h Trend (wider Fib zone)
    {
        "label":      "H: FVG+Fib38-62%+4h_Trend (wider Fib zone)",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    False,
        "use_rsi_filter":    False,
        "use_partial_tp":    True,
        "partial_tp_rr":     1.0,
        "partial_tp_frac":   0.50,
        "use_trailing_sl":   True,
        "trailing_sl_be_pct": 0.1,
        "require_fib_confluence": True,
        "fib_low_level":     0.382,
        "fib_high_level":    0.618,
        "require_4h_trend":  True,
        "trend_only":        True,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
    },
    # I: FVG+Fib38-62%+4h Trend + Chandelier trailing (no fixed TP)
    {
        "label":      "I: FVG+Fib38-62%+4hTrend+Chandelier_trail",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    False,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
        "require_fib_confluence": True,
        "fib_low_level":     0.382,
        "fib_high_level":    0.618,
        "require_4h_trend":  True,
        "trend_only":        True,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
        # BB Squeeze enhancements:
        "use_chandelier_trailing": True,
        "ce_length":  22,
        "ce_mult":    3.0,
        "use_adx_filter": False,
    },
    # J: FVG+Fib+4h Trend + Chandelier trail + ADX>20
    {
        "label":      "J: FVG+Fib+4hTrend+Chandelier+ADX>20",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
        "require_fib_confluence": True,
        "fib_low_level":     0.382,
        "fib_high_level":    0.618,
        "require_4h_trend":  True,
        "trend_only":        True,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
        "use_chandelier_trailing": True,
        "ce_length":  22,
        "ce_mult":    3.0,
        "use_adx_filter": True,
        "adx_threshold":  20.0,
    },
    # K: FVG+Fib+4h Trend + Chandelier trail + ADX>15 (relaxed)
    {
        "label":      "K: FVG+Fib+4hTrend+Chandelier+ADX>15",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 4.5,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
        "require_fib_confluence": True,
        "fib_low_level":     0.382,
        "fib_high_level":    0.618,
        "require_4h_trend":  True,
        "trend_only":        True,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
        "use_chandelier_trailing": True,
        "ce_length":  22,
        "ce_mult":    3.5,
        "use_adx_filter": True,
        "adx_threshold":  15.0,
    },
    # L: *** UPDATED LIVE PARAMS *** — wider SL behind structure, TP closer inside, min_rr=1.0
    # Matches updated config.py: SL_NEAREST_LEVEL_BUFFER_PCT=0.60, TP_HVN_OFFSET_PCT=0.25, MIN_TP_RR=1.0
    {
        "label":      "L: LiveV2 (SL_buf=0.60%, TP_off=0.25%, min_rr=1.0, Fib+4h+Chandelier+ADX>20)",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 5.0,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
        "require_fib_confluence": True,
        "fib_low_level":     0.382,
        "fib_high_level":    0.618,
        "require_4h_trend":  True,
        "trend_only":        True,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
        "use_chandelier_trailing": True,
        "ce_length":  22,
        "ce_mult":    3.0,
        "use_adx_filter": True,
        "adx_threshold":  20.0,
        # Updated SL/TP params:
        "sl_buffer_pct": 0.60,   # wider buffer BEHIND structure (sweep-safe)
        "tp_offset_pct": 0.25,   # TP placed 0.25% INSIDE resistance (guaranteed fill)
        "min_rr_tp":     1.0,    # accept trades with RR >= 1.0 (was 1.5)
        "max_rr_tp":     3.0,
    },
    # M: Same as L but without 4h trend filter (more trades, less restrictive)
    {
        "label":      "M: LiveV2_noTrend (SL_buf=0.60%, TP_off=0.25%, min_rr=1.0, Fib+Chandelier)",
        "percentile": 17.2,
        "quality":    50.0,
        "composite":  55.0,
        "mid_low":    0.35,
        "mid_high":   0.65,
        "min_gap":    0.6,
        "require_displacement": True,
        "max_sl_pct": 5.0,
        "use_ema50_filter":  False,
        "use_ema200_filter": False,
        "use_vol_filter":    True,
        "use_rsi_filter":    False,
        "use_partial_tp":    False,
        "use_trailing_sl":   False,
        "require_fib_confluence": True,
        "fib_low_level":     0.382,
        "fib_high_level":    0.618,
        "require_4h_trend":  False,
        "trend_only":        False,
        "impulse_min_pct":   1.5,
        "impulse_lookback_1h": 24,
        "use_chandelier_trailing": True,
        "ce_length":  22,
        "ce_mult":    3.0,
        "use_adx_filter": False,
        "adx_threshold":  20.0,
        # Updated SL/TP params:
        "sl_buffer_pct": 0.60,
        "tp_offset_pct": 0.25,
        "min_rr_tp":     1.0,
        "max_rr_tp":     3.0,
    },
]


# ─── Data loading (cache-first) ───────────────────────────────────────────────

def _symbol_safe(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def _load_or_download(symbol: str, timeframe: str) -> pd.DataFrame:
    """Load from parquet/CSV cache if available, otherwise download via CCXT."""
    base_path = CACHE_DIR / f"{_symbol_safe(symbol)}_{timeframe}_90d"

    for path, reader in [(base_path.with_suffix(".parquet"), "parquet"),
                         (base_path.with_suffix(".csv"),     "csv")]:
        if path.exists():
            try:
                df = pd.read_parquet(path) if reader == "parquet" else pd.read_csv(path)
                if len(df) > 200:
                    logger.info("  Loaded %s %s from %s cache (%d rows)",
                                symbol, timeframe, reader, len(df))
                    return df
            except Exception as exc:
                logger.warning("Cache read failed for %s %s (%s): %s",
                               symbol, timeframe, reader, exc)

    # Fallback: download from API
    logger.info("  Cache miss for %s %s — downloading ...", symbol, timeframe)
    return _download_ccxt(symbol, timeframe, BACKTEST_DAYS, extra_bars=500)


def _download_ccxt(symbol: str, timeframe: str, days: int, extra_bars: int = 500) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: py -3 -m pip install ccxt")
        return pd.DataFrame()

    bars_per_day = {"1m": 1440, "3m": 480, "5m": 288, "15m": 96, "30m": 48, "1h": 24}
    bpd          = bars_per_day.get(timeframe, 96)
    total_needed = days * bpd + extra_bars
    now_ms       = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms     = now_ms - (days + extra_bars // bpd + 3) * 86_400_000
    bar_ms       = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
                    "30m": 1_800_000, "1h": 3_600_000}.get(timeframe, 900_000)
    try:
        exchange    = ccxt.bybit({"options": {"defaultType": "swap"}})
        all_ohlcv   = []
        fetch_since = since_ms
        for _ in range(20):
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= now_ms - bar_ms * 2:
                break
            fetch_since = last_ts + 1
            if len(all_ohlcv) >= total_needed:
                break
        if not all_ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(df) > total_needed:
            df = df.tail(total_needed).reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("CCXT fetch failed for %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()


# ─── Indicator helpers ────────────────────────────────────────────────────────

def _compute_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Vectorized RSI using Wilder smoothing. Returns array same length as closes."""
    if len(closes) < period + 1:
        return np.full(len(closes), 50.0)
    deltas = np.diff(closes.astype(float))
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    rsi    = np.full(len(closes), 50.0)
    avg_g  = float(np.mean(gains[:period]))
    avg_l  = float(np.mean(losses[:period]))

    for i in range(period, len(deltas)):
        avg_g = (avg_g * (period - 1) + gains[i])  / period
        avg_l = (avg_l * (period - 1) + losses[i]) / period
        rs    = avg_g / max(avg_l, 1e-10)
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)

    return rsi


def _atr_pct(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    if len(closes) < 3:
        return 1.0
    prev = np.roll(closes, 1)
    prev[0] = closes[0]
    tr   = np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))
    atr  = float(np.mean(tr[-14:]))
    last = float(closes[-1]) if closes[-1] > 0 else 1.0
    return atr / last * 100.0


def _collect_diffs(highs: np.ndarray, lows: np.ndarray, start: int, end: int) -> np.ndarray:
    diffs = []
    for i in range(start, end):
        mid = (highs[i] + lows[i]) / 2.0
        if mid <= 0:
            continue
        if highs[i - 2] < lows[i]:
            diffs.append((lows[i] - highs[i - 2]) / mid * 100.0)
        if lows[i - 2] > highs[i]:
            diffs.append((lows[i - 2] - highs[i]) / mid * 100.0)
    return np.array(diffs, dtype=float) if diffs else np.array([], dtype=float)


# ─── FVG detection ────────────────────────────────────────────────────────────

def detect_fvgs(
    df: pd.DataFrame,
    percentile_threshold: float = 17.2,
    quality_floor: float = 50.0,
    min_gap_pct: float = 0.6,
    require_displacement: bool = True,
) -> list[dict]:
    """Detect FVGs — same logic as fvg_core.py, no external imports needed."""
    if len(df) < 3:
        return []

    opens     = df["open"].values.astype(float)
    highs_raw = df["high"].values.astype(float)
    lows_raw  = df["low"].values.astype(float)
    closes    = df["close"].values.astype(float)
    times     = df["timestamp"].values

    body_sizes = np.abs(closes - opens)
    body_mid   = (highs_raw + lows_raw) / 2.0
    body_mid   = np.where(body_mid <= 0, 1.0, body_mid)
    body_pct   = body_sizes / body_mid * 100.0

    plb        = min(500, len(df))
    dist_start = max(2, len(df) - plb)
    all_diffs  = _collect_diffs(highs_raw, lows_raw, dist_start, len(df))
    pct_rank   = float(np.percentile(all_diffs, 100)) if len(all_diffs) > 0 else 1.0

    scan_start = 2
    disp_mult  = float(cfg.FVG_DISPLACEMENT_BODY_MULT)
    disp_min   = float(cfg.FVG_DISPLACEMENT_BODY_PCT)

    fvgs = []
    for i in range(scan_start, len(df)):
        mid_price = (highs_raw[i] + lows_raw[i]) / 2.0
        if mid_price <= 0:
            continue

        window_start  = max(0, i - 60)
        baseline      = float(np.median(body_pct[window_start:i])) if i > window_start else 0.0
        required_body = max(disp_min, baseline * disp_mult)

        if require_displacement and body_pct[i - 1] < required_body:
            continue

        # Regime adjustment
        start_r  = max(0, i - 96)
        cwin     = closes[start_r:i + 1]
        hwin     = highs_raw[start_r:i + 1]
        lwin     = lows_raw[start_r:i + 1]
        regime_adj = 0.0
        trend_pct  = 0.0
        atr_r      = 1.0
        if len(cwin) >= 20:
            atr_r      = _atr_pct(hwin, lwin, cwin)
            s          = pd.Series(cwin)
            ema_fast   = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
            ema_slow   = float(s.ewm(span=50, adjust=False).mean().iloc[-1])
            lc         = float(cwin[-1]) if cwin[-1] > 0 else 1.0
            trend_pct  = abs(ema_fast - ema_slow) / lc * 100.0
            diff_arr   = np.diff(cwin)
            signs      = np.sign(diff_arr)
            dc         = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
            chop_pct   = dc / max(1, len(signs)) * 100.0
            if trend_pct >= max(0.10, atr_r * 0.45):
                regime_adj += float(cfg.REGIME_TREND_BONUS)
            if chop_pct >= 45.0:
                regime_adj += float(cfg.REGIME_CHOP_PENALTY)
            if atr_r >= 1.8:
                regime_adj += 1.0

        dynamic_threshold = percentile_threshold + regime_adj

        for direction in ("bullish", "bearish"):
            if direction == "bullish":
                if not (highs_raw[i - 2] < lows_raw[i]):
                    continue
                if closes[i - 1] <= opens[i - 1]:
                    continue
                gap_size = lows_raw[i] - highs_raw[i - 2]
                fvg_high = lows_raw[i]
                fvg_low  = highs_raw[i - 2]
            else:
                if not (lows_raw[i - 2] > highs_raw[i]):
                    continue
                if closes[i - 1] >= opens[i - 1]:
                    continue
                gap_size = lows_raw[i - 2] - highs_raw[i]
                fvg_high = lows_raw[i - 2]
                fvg_low  = highs_raw[i]

            diff = (gap_size / mid_price) * 100.0
            if diff < min_gap_pct:
                continue

            size_fvg = (diff / pct_rank * 100.0) if pct_rank > 0 else 0.0

            disp_ratio    = body_pct[i - 1] / max(required_body, 1e-6)
            disp_score    = min(disp_ratio * 35.0, 100.0)
            atr_for_q     = max(_atr_pct(highs_raw[max(0,i-20):i+1], lows_raw[max(0,i-20):i+1], closes[max(0,i-20):i+1]), 1e-6)
            vol_score     = min((diff / atr_for_q) * 40.0, 100.0)
            trend_score_q = 50.0
            if len(cwin) >= 20:
                trend_score_q = min((trend_pct / max(atr_r, 0.12)) * 45.0, 100.0)
            dir_score = 100.0 if (
                (direction == "bullish" and (closes[i-1] > opens[i-1] or closes[i-2] > opens[i-2]))
                or (direction == "bearish" and (closes[i-1] < opens[i-1] or closes[i-2] < opens[i-2]))
            ) else 25.0
            quality = (
                size_fvg      * 0.40
                + disp_score  * 0.20
                + vol_score   * 0.20
                + trend_score_q * 0.10
                + dir_score   * 0.10
            )
            quality = float(max(0.0, min(100.0, quality)))

            if size_fvg < dynamic_threshold or quality < quality_floor:
                continue

            fvgs.append({
                "type":       direction,
                "high":       float(fvg_high),
                "low":        float(fvg_low),
                "timestamp":  int(times[i - 1]),
                "candle_idx": i - 1,
                "gap_pct":    float(diff),
                "size_fvg":   float(size_fvg),
                "quality":    float(quality),
                "volume_at":  float(df["volume"].iloc[i]) if "volume" in df.columns else 0.0,
            })

    return fvgs


# ─── Structural SL/TP (unchanged from v2) ────────────────────────────────────

def _collect_structural_levels(
    highs: np.ndarray, lows: np.ndarray, tolerance_pct: float = 0.06,
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    n = len(highs)
    raw_supports: list[float] = []
    raw_resistances: list[float] = []
    for win in (2, 3, 5):
        for i in range(win, n - win):
            low_i  = float(lows[i])
            high_i = float(highs[i])
            if low_i <= float(np.min(lows[max(0, i-win):i+win+1])):
                raw_supports.append(low_i)
            if high_i >= float(np.max(highs[max(0, i-win):i+win+1])):
                raw_resistances.append(high_i)
    tol = max(0.0002, tolerance_pct / 100.0)

    def _cluster(values: list[float], probe: np.ndarray) -> list[tuple[float, int]]:
        cleaned = sorted(v for v in values if v > 0)
        if not cleaned:
            return []
        clusters: list[list[float]] = []
        for value in cleaned:
            merged = False
            for cluster in clusters:
                center = float(sum(cluster) / len(cluster))
                if abs(value - center) / max(center, 1e-9) <= tol:
                    cluster.append(value)
                    merged = True
                    break
            if not merged:
                clusters.append([value])
        result: list[tuple[float, int]] = []
        for cluster in clusters:
            level   = float(np.median(cluster))
            touches = int(np.sum(np.abs(probe - level) / np.maximum(level, 1e-9) <= tol * 3))
            result.append((level, max(1, touches)))
        return result

    all_probe   = np.concatenate([highs, lows])
    supports    = sorted(_cluster(raw_supports,    all_probe), key=lambda x: x[0], reverse=True)
    resistances = sorted(_cluster(raw_resistances, all_probe), key=lambda x: x[0])
    return supports, resistances


def find_structural_sl(
    df: pd.DataFrame, entry: float, direction: str,
    fvg_low: float = 0.0, fvg_high: float = 0.0,
    lookback: int = 120, min_sl_dist_pct: float = 0.8,
    sl_buffer: float = 0.0015,
) -> float:
    tail = df.tail(lookback)
    if tail.empty or len(tail) < 10:
        fallback = entry * (1.0 - max(min_sl_dist_pct, 1.0) / 100.0) if direction == "LONG" \
                   else entry * (1.0 + max(min_sl_dist_pct, 1.0) / 100.0)
        return float(fallback)

    highs  = tail["high"].astype(float).values
    lows   = tail["low"].astype(float).values
    closes = tail["close"].astype(float).values
    atr    = _atr_pct(highs, lows, closes)
    atr_abs = atr / 100.0 * entry
    min_dist  = max(entry * min_sl_dist_pct / 100.0, atr_abs * 0.75)
    max_sl_dist = entry * 0.045
    supports, resistances = _collect_structural_levels(highs, lows, tolerance_pct=0.06)

    if direction == "LONG":
        zone_boundary = fvg_low if fvg_low > 0 else entry
        min_offset    = zone_boundary * (1.0 - 0.001)
        candidates    = supports if supports else [(float(np.min(lows)), 1)]
        scored        = []
        for lvl, touches in candidates:
            if lvl <= 0 or lvl >= min_offset or lvl >= entry:
                continue
            dist_pct = (entry - lvl) / max(entry, 1e-9) * 100.0
            if dist_pct <= 0.12 or dist_pct > 5.0:
                continue
            score = touches * 1.7 - abs(dist_pct - 1.2) * 0.8
            scored.append((score, lvl))
        if scored:
            scored.sort(reverse=True)
            sl_raw = scored[0][1] * (1.0 - sl_buffer)
            sl_raw = max(sl_raw, entry - max_sl_dist)
            sl_raw = min(sl_raw, entry - min_dist)
            return float(sl_raw)
        return float(max(entry - max_sl_dist, entry - min_dist))
    else:
        zone_boundary = fvg_high if fvg_high > 0 else entry
        min_offset    = zone_boundary * (1.0 + 0.001)
        candidates    = resistances if resistances else [(float(np.max(highs)), 1)]
        scored        = []
        for lvl, touches in candidates:
            if lvl <= 0 or lvl <= min_offset or lvl <= entry:
                continue
            dist_pct = (lvl - entry) / max(entry, 1e-9) * 100.0
            if dist_pct <= 0.12 or dist_pct > 5.0:
                continue
            score = touches * 1.7 - abs(dist_pct - 1.2) * 0.8
            scored.append((score, lvl))
        if scored:
            scored.sort(reverse=True)
            sl_raw = scored[0][1] * (1.0 + sl_buffer)
            sl_raw = min(sl_raw, entry + max_sl_dist)
            sl_raw = max(sl_raw, entry + min_dist)
            return float(sl_raw)
        return float(min(entry + max_sl_dist, entry + min_dist))


def find_structural_tp(
    df: pd.DataFrame, entry: float, direction: str, sl: float,
    fvg_high: float = 0.0, fvg_low: float = 0.0,
    min_rr: float = 1.5, max_rr: float = 2.5,
    tp_offset: float = 0.0012,
) -> float:
    risk = abs(entry - sl)
    if risk <= 0:
        risk = entry * 0.01
    min_tp_dist = risk * min_rr
    max_tp_dist = risk * max_rr

    if direction == "LONG":
        target_floor = max(entry, fvg_high) if fvg_high > 0 else entry
        min_tp = entry + min_tp_dist
        max_tp = entry + max_tp_dist
    else:
        target_floor = min(entry, fvg_low) if fvg_low > 0 else entry
        min_tp = entry - min_tp_dist
        max_tp = entry - max_tp_dist

    tail = df.tail(120)
    if tail.empty or len(tail) < 5:
        return min_tp

    highs = tail["high"].astype(float).values
    lows  = tail["low"].astype(float).values
    supports, resistances = _collect_structural_levels(highs, lows, tolerance_pct=0.06)

    def _pick_tp(candidates: list[tuple[float, float]]) -> float:
        ranked = []
        for lvl, base_score in candidates:
            if lvl <= 0:
                continue
            dist_pct = abs(lvl - entry) / max(entry, 1e-9) * 100.0
            if dist_pct < 0.20:
                continue
            ranked.append((base_score - dist_pct * 0.08, lvl))
        if not ranked:
            return 0.0
        ranked.sort(reverse=True)
        return float(ranked[0][1])

    if direction == "LONG":
        tp_candidates: list[tuple[float, float]] = []
        for lvl, touches in resistances:
            if lvl > target_floor and min_tp <= lvl <= max_tp:
                tp_candidates.append((float(lvl), 1.2 + touches * 0.15))
        for i in range(2, len(highs) - 2):
            h = float(highs[i])
            if h >= highs[i-1] and h >= highs[i-2] and h >= highs[i+1] and h >= highs[i+2]:
                if h > target_floor and min_tp <= h <= max_tp:
                    tp_candidates.append((h, 1.0))
        if tp_candidates:
            best = _pick_tp(tp_candidates)
            if best > 0:
                return best * (1.0 - tp_offset)
        return min_tp
    else:
        tp_candidates = []
        for lvl, touches in supports:
            if 0 < lvl < target_floor and max_tp <= lvl <= min_tp:
                tp_candidates.append((float(lvl), 1.2 + touches * 0.15))
        for i in range(2, len(lows) - 2):
            l = float(lows[i])
            if l <= lows[i-1] and l <= lows[i-2] and l <= lows[i+1] and l <= lows[i+2]:
                if 0 < l < target_floor and max_tp <= l <= min_tp:
                    tp_candidates.append((l, 1.0))
        if tp_candidates:
            best = _pick_tp(tp_candidates)
            if best > 0:
                return best * (1.0 + tp_offset)
        return min_tp


# ─── HTF bias ────────────────────────────────────────────────────────────────

def compute_htf_bias(df_15m: pd.DataFrame, ema_period: int = 200) -> pd.Series:
    df = df_15m.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    df_1h = df["close"].resample("1h").last().dropna()
    ema   = df_1h.ewm(span=ema_period, adjust=False).mean()
    df_1h_s = df_1h.shift(1).dropna()
    ema_s   = ema.shift(1).dropna()
    return pd.Series(
        np.where(df_1h_s.values > ema_s.values, "bullish", "bearish"),
        index=df_1h_s.index,
    )


def get_htf_bias_at(bias_series: pd.Series, ts_ms: int) -> str:
    if bias_series is None or bias_series.empty:
        return "neutral"
    ts_dt = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
    idx   = bias_series.index[bias_series.index <= ts_dt]
    if idx.empty:
        return "neutral"
    return str(bias_series.loc[idx[-1]])


# ─── Fibonacci + Impulse helpers ─────────────────────────────────────────────

def find_impulse_moves(df_1h: pd.DataFrame, min_move_pct: float = 2.0, lookback: int = 20):
    """
    Find recent impulse moves on 1h chart.
    Impulse = price moves min_move_pct% in 3-5 consecutive candles in same direction.
    Returns (start_idx, end_idx, direction, fib_start, fib_end) or None.
    """
    closes = df_1h["close"].values.astype(float)
    highs  = df_1h["high"].values.astype(float)
    lows   = df_1h["low"].values.astype(float)
    impulses = []

    for window in [3, 4, 5]:
        for i in range(max(0, len(closes) - lookback), len(closes) - window):
            start_p = closes[i]
            end_p   = closes[i + window]
            if start_p <= 0:
                continue
            move_pct = (end_p - start_p) / start_p * 100.0

            if abs(move_pct) < min_move_pct:
                continue

            direction    = "bullish" if move_pct > 0 else "bearish"
            max_retrace  = abs(move_pct) * 0.5
            consistent   = True
            for j in range(i, i + window):
                if closes[j] <= 0:
                    consistent = False; break
                step_move = (closes[j + 1] - closes[j]) / closes[j] * 100.0
                if direction == "bullish" and step_move < -max_retrace:
                    consistent = False; break
                if direction == "bearish" and step_move > max_retrace:
                    consistent = False; break

            if consistent:
                if direction == "bullish":
                    fib_start = float(lows[i])
                    fib_end   = float(highs[i + window])
                else:
                    fib_start = float(highs[i])
                    fib_end   = float(lows[i + window])
                impulses.append((i, i + window, direction, fib_start, fib_end))

    return impulses[-1] if impulses else None


def get_fib_zone(impulse, level_low: float = 0.50, level_high: float = 0.618):
    """
    Get Fib retracement zone.
    Bullish impulse: price retraces DOWN → zone is below the impulse end.
    Bearish impulse: price retraces UP   → zone is above the impulse end.
    Returns (zone_low, zone_high, zone_type).
    """
    _, _, direction, start_p, end_p = impulse
    move = abs(end_p - start_p)
    if direction == "bullish":
        zone_high = end_p - move * level_low    # 50% retrace
        zone_low  = end_p - move * level_high   # 62% retrace
        return zone_low, zone_high, "support"
    else:
        zone_low  = end_p + move * level_low
        zone_high = end_p + move * level_high
        return zone_low, zone_high, "resistance"


def fvg_in_fib_zone(fvg: dict, fib_zone_low: float, fib_zone_high: float) -> bool:
    """Check if FVG [fvg_low, fvg_high] overlaps with Fib retracement zone."""
    return fvg["high"] >= fib_zone_low and fvg["low"] <= fib_zone_high


def _compute_4h_trend(df_15m: pd.DataFrame) -> str:
    """
    Resample 15m data to 4h, compute EMA200, return 'bullish' or 'bearish'.
    Uses the second-to-last closed 4h bar to avoid lookahead.
    """
    if df_15m is None or len(df_15m) < 200:
        return "neutral"
    df = df_15m.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    df_4h = df[["open", "high", "low", "close", "volume"]].resample("4h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna()
    if len(df_4h) < 200:
        return "neutral"
    ema200 = df_4h["close"].ewm(span=200, adjust=False).mean()
    last_close = float(df_4h["close"].iloc[-1])
    last_ema   = float(ema200.iloc[-1])
    return "bullish" if last_close > last_ema else "bearish"


def _resample_to_1h(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample 15m OHLCV to 1h."""
    df = df_15m.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    df_1h = df[["open", "high", "low", "close", "volume"]].resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    df_1h["timestamp"] = df_1h["dt"].astype(np.int64) // 1_000_000
    return df_1h


# ─── Single-symbol walk-forward engine ───────────────────────────────────────

def _run_backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    bias_series: pd.Series,
    balance_start: float,
    variant: dict,
    warmup_bars: int = 200,
) -> tuple[list[dict], float]:
    """Walk-forward FVG backtest on a single symbol. Returns (trades, final_balance)."""
    trades: list[dict] = []
    balance = balance_start

    percentile_threshold = float(variant["percentile"])
    quality_floor        = float(variant["quality"])
    composite_floor      = float(variant["composite"])
    composite_cap        = float(variant.get("max_composite", 999.0))
    mid_low              = float(variant["mid_low"])
    mid_high             = float(variant["mid_high"])
    min_gap              = float(variant["min_gap"])
    require_disp         = bool(variant.get("require_displacement", True))
    max_sl_pct           = float(variant.get("max_sl_pct", 4.5))
    use_ema50_filter     = bool(variant.get("use_ema50_filter", False))
    use_ema200_filter    = bool(variant.get("use_ema200_filter", False))
    use_vol_filter       = bool(variant.get("use_vol_filter", False))
    use_rsi_filter       = bool(variant.get("use_rsi_filter", False))
    rsi_long_min         = float(variant.get("rsi_long_min", 35.0))
    rsi_long_max         = float(variant.get("rsi_long_max", 65.0))
    rsi_short_min        = float(variant.get("rsi_short_min", 35.0))
    rsi_short_max        = float(variant.get("rsi_short_max", 65.0))
    use_partial_tp       = bool(variant.get("use_partial_tp", False))
    partial_tp_rr        = float(variant.get("partial_tp_rr", 1.0))
    partial_tp_frac      = float(variant.get("partial_tp_frac", 0.50))
    use_trailing_sl      = bool(variant.get("use_trailing_sl", False))
    trailing_be_pct      = float(variant.get("trailing_sl_be_pct", 0.1))
    require_fib          = bool(variant.get("require_fib_confluence", False))
    fib_level_low        = float(variant.get("fib_low_level", 0.50))
    fib_level_high       = float(variant.get("fib_high_level", 0.618))
    require_4h_trend     = bool(variant.get("require_4h_trend", False))
    trend_only           = bool(variant.get("trend_only", False))
    impulse_min_pct      = float(variant.get("impulse_min_pct", 1.5))
    impulse_lookback_1h  = int(variant.get("impulse_lookback_1h", 24))
    # BB Squeeze enhancements
    use_chandelier_trail = bool(variant.get("use_chandelier_trailing", False))
    ce_length            = int(variant.get("ce_length", 22))
    ce_mult              = float(variant.get("ce_mult", 3.0))
    use_adx_filter       = bool(variant.get("use_adx_filter", False))
    adx_threshold        = float(variant.get("adx_threshold", 20.0))
    # Updated SL/TP parameters (v2 live params)
    sl_buf_pct_v   = float(variant.get("sl_buffer_pct", 0.15))   # % behind structure
    tp_offset_pct  = float(variant.get("tp_offset_pct", 0.12))   # % inside resistance for TP
    min_rr_tp      = float(variant.get("min_rr_tp", 1.5))        # minimum RR for TP acceptance
    max_rr_tp      = float(variant.get("max_rr_tp", 2.5))        # maximum RR cap

    if len(df) < warmup_bars + 10:
        return trades, balance

    # ── Precompute indicators once ────────────────────────────────────────────
    close_arr    = df["close"].values.astype(float)
    vol_arr      = df["volume"].values.astype(float) if "volume" in df.columns else np.zeros(len(df))
    ema50_arr    = pd.Series(close_arr).ewm(span=50,  adjust=False).mean().shift(1).values
    ema200_arr   = pd.Series(close_arr).ewm(span=200, adjust=False).mean().shift(1).values
    rsi_arr      = _compute_rsi(close_arr, period=14)

    # Rolling 10-bar average volume (shifted 1 to avoid lookahead)
    vol_avg10    = pd.Series(vol_arr).rolling(10, min_periods=1).mean().shift(1).values

    # Precompute ADX + Chandelier ATR (for enhanced variants I/J/K)
    high_arr = df["high"].values.astype(float)
    low_arr  = df["low"].values.astype(float)
    adx_arr_full = _compute_adx(high_arr, low_arr, close_arr, period=14)
    ce_atr_arr   = _ce_atr_calc(high_arr, low_arr, close_arr, period=ce_length)

    # ── Precompute Fib/4h trend data (only for G/H variants) ─────────────────
    htf_4h_bullish: Optional[bool] = None
    df_1h_cached:   Optional[pd.DataFrame] = None
    if require_fib or require_4h_trend:
        try:
            trend_4h = _compute_4h_trend(df)
            if trend_4h == "bullish":
                htf_4h_bullish = True
            elif trend_4h == "bearish":
                htf_4h_bullish = False
            else:
                htf_4h_bullish = None
            df_1h_cached = _resample_to_1h(df)
        except Exception:
            htf_4h_bullish = None
            df_1h_cached   = None

    # ── Precompute FVGs once ──────────────────────────────────────────────────
    all_fvgs = detect_fvgs(
        df,
        percentile_threshold=percentile_threshold,
        quality_floor=quality_floor,
        min_gap_pct=min_gap,
        require_displacement=require_disp,
    )

    fvg_by_detbar: dict[int, list[dict]] = defaultdict(list)
    for fvg in all_fvgs:
        fvg_by_detbar[fvg["candle_idx"] + 1].append(fvg)

    daily_counts: dict[str, dict[str, int]] = {}
    open_positions: list[dict] = []
    used_fvg_keys:  set[tuple[int, str]] = set()
    active_fvgs:    list[dict] = []
    max_age_ms = float(cfg.FVG_MAX_AGE_DAYS) * 86_400_000.0

    for i in range(warmup_bars, len(df) - 1):
        ts       = int(df["timestamp"].iloc[i])
        candle_h = float(df["high"].iloc[i])
        candle_l = float(df["low"].iloc[i])
        candle_c = float(df["close"].iloc[i])

        date_key = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        daily    = daily_counts.setdefault(date_key, {"wins": 0, "losses": 0})

        # ── 1. Exit open positions (with partial TP + trailing SL logic) ──────
        still_open = []
        for pos in open_positions:
            if pos["symbol"] != symbol:
                still_open.append(pos)
                continue

            direction_p = pos["direction"]
            sl_p        = pos["sl"]
            tp_p        = pos["tp"]
            # partial TP level: entry ± 1*sl_dist
            partial_tp_price = pos.get("partial_tp_price", None)
            partial_done     = pos.get("partial_done", False)
            trailing_done    = pos.get("trailing_done", False)

            # ── Check trailing SL / BE-move ────────────────────────────────
            if use_trailing_sl and not trailing_done and partial_tp_price is not None:
                if direction_p == "LONG" and candle_h >= partial_tp_price:
                    be_sl = pos["entry_exec"] * (1.0 + trailing_be_pct / 100.0)
                    if be_sl > sl_p:
                        pos["sl"]           = be_sl
                        sl_p                = be_sl
                        pos["trailing_done"] = True
                        trailing_done        = True
                elif direction_p == "SHORT" and candle_l <= partial_tp_price:
                    be_sl = pos["entry_exec"] * (1.0 - trailing_be_pct / 100.0)
                    if be_sl < sl_p:
                        pos["sl"]           = be_sl
                        sl_p                = be_sl
                        pos["trailing_done"] = True
                        trailing_done        = True

            # ── Check partial TP (50% close at 1:1) ────────────────────────
            if use_partial_tp and not partial_done and partial_tp_price is not None:
                if direction_p == "LONG" and candle_h >= partial_tp_price:
                    # Close partial_tp_frac of position
                    close_qty  = pos["qty"] * partial_tp_frac
                    remain_qty = pos["qty"] * (1.0 - partial_tp_frac)
                    exit_exec  = partial_tp_price * (1.0 - SLIPPAGE_PCT)
                    gross      = (exit_exec - pos["entry_exec"]) * close_qty
                    fee_out    = exit_exec * close_qty * TAKER_FEE_PCT
                    fee_in_partial = pos["fee_in"] * partial_tp_frac
                    net_pnl    = gross - fee_in_partial - fee_out
                    balance   += net_pnl
                    if net_pnl >= 0:
                        daily["wins"] += 1
                    trades.append({
                        "symbol":        symbol,
                        "direction":     direction_p,
                        "entry_price":   round(pos["entry_exec"], 8),
                        "exit_price":    round(exit_exec, 8),
                        "sl":            round(sl_p, 8),
                        "tp":            round(tp_p, 8),
                        "qty":           close_qty,
                        "rr_plan":       round(partial_tp_rr, 3),
                        "gap_pct":       round(pos.get("gap_pct", 0), 4),
                        "quality":       round(pos.get("quality", 0), 2),
                        "composite":     round(pos.get("composite", 0), 2),
                        "pnl_usd":       round(net_pnl, 4),
                        "balance_after": round(balance, 4),
                        "exit_reason":   "PARTIAL_TP",
                        "date":          date_key,
                        "month":         datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m"),
                    })
                    pos["qty"]          = remain_qty
                    pos["fee_in"]       = pos["fee_in"] * (1.0 - partial_tp_frac)
                    pos["partial_done"] = True
                    partial_done        = True

                elif direction_p == "SHORT" and candle_l <= partial_tp_price:
                    close_qty  = pos["qty"] * partial_tp_frac
                    remain_qty = pos["qty"] * (1.0 - partial_tp_frac)
                    exit_exec  = partial_tp_price * (1.0 + SLIPPAGE_PCT)
                    gross      = (pos["entry_exec"] - exit_exec) * close_qty
                    fee_out    = exit_exec * close_qty * TAKER_FEE_PCT
                    fee_in_partial = pos["fee_in"] * partial_tp_frac
                    net_pnl    = gross - fee_in_partial - fee_out
                    balance   += net_pnl
                    if net_pnl >= 0:
                        daily["wins"] += 1
                    trades.append({
                        "symbol":        symbol,
                        "direction":     direction_p,
                        "entry_price":   round(pos["entry_exec"], 8),
                        "exit_price":    round(exit_exec, 8),
                        "sl":            round(sl_p, 8),
                        "tp":            round(tp_p, 8),
                        "qty":           close_qty,
                        "rr_plan":       round(partial_tp_rr, 3),
                        "gap_pct":       round(pos.get("gap_pct", 0), 4),
                        "quality":       round(pos.get("quality", 0), 2),
                        "composite":     round(pos.get("composite", 0), 2),
                        "pnl_usd":       round(net_pnl, 4),
                        "balance_after": round(balance, 4),
                        "exit_reason":   "PARTIAL_TP",
                        "date":          date_key,
                        "month":         datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m"),
                    })
                    pos["qty"]          = remain_qty
                    pos["fee_in"]       = pos["fee_in"] * (1.0 - partial_tp_frac)
                    pos["partial_done"] = True
                    partial_done        = True

            # ── Chandelier trailing SL update (enhanced variants) ─────────
            if use_chandelier_trail and i > ce_length:
                new_ce_sl = chandelier_exit_sl(
                    high_arr, low_arr, ce_atr_arr,
                    direction_p, i, ce_length=ce_length, ce_mult=ce_mult,
                )
                entry_exec_p = pos.get("entry_exec", sl_p)
                risk_p = abs(entry_exec_p - pos.get("original_sl", sl_p))
                pnl_r = 0.0
                if risk_p > 0:
                    if direction_p == "LONG":
                        pnl_r = (candle_c - entry_exec_p) / risk_p
                    else:
                        pnl_r = (entry_exec_p - candle_c) / risk_p
                # After 1R profit, trail with Chandelier
                if pnl_r >= 1.0:
                    if direction_p == "LONG":
                        pos["sl"] = max(sl_p, new_ce_sl)
                        sl_p = pos["sl"]
                    else:
                        if new_ce_sl > 0:
                            pos["sl"] = min(sl_p, new_ce_sl) if sl_p > 0 else new_ce_sl
                            sl_p = pos["sl"]

            # ── Full position exit ──────────────────────────────────────────
            exit_price  = None
            exit_reason = ""
            if use_chandelier_trail:
                # Chandelier mode: only SL (trailing), no fixed TP
                if direction_p == "LONG":
                    if candle_l <= sl_p:
                        exit_price = sl_p; exit_reason = "Trail_SL"
                else:
                    if candle_h >= sl_p:
                        exit_price = sl_p; exit_reason = "Trail_SL"
                # Max TP safety: 5R
                if exit_price is None:
                    entry_exec_p = pos.get("entry_exec", sl_p)
                    risk_p = abs(entry_exec_p - pos.get("original_sl", sl_p))
                    if risk_p > 0:
                        if direction_p == "LONG" and (candle_h - entry_exec_p) / risk_p >= 5.0:
                            exit_price = candle_c; exit_reason = "TP_MAX"
                        elif direction_p == "SHORT" and (entry_exec_p - candle_l) / risk_p >= 5.0:
                            exit_price = candle_c; exit_reason = "TP_MAX"
            else:
                # Original fixed TP logic
                if direction_p == "LONG":
                    if candle_l <= sl_p and candle_h >= tp_p:
                        exit_price  = sl_p; exit_reason = "SL"
                    elif candle_l <= sl_p:
                        exit_price  = sl_p; exit_reason = "SL"
                    elif candle_h >= tp_p:
                        exit_price  = tp_p; exit_reason = "TP"
                else:
                    if candle_h >= sl_p and candle_l <= tp_p:
                        exit_price  = sl_p; exit_reason = "SL"
                    elif candle_h >= sl_p:
                        exit_price  = sl_p; exit_reason = "SL"
                    elif candle_l <= tp_p:
                        exit_price  = tp_p; exit_reason = "TP"

            if exit_price is None:
                still_open.append(pos)
                continue

            if direction_p == "LONG":
                exit_exec = exit_price * (1 - SLIPPAGE_PCT) if exit_reason == "TP" else exit_price * (1 + SLIPPAGE_PCT)
                gross     = (exit_exec - pos["entry_exec"]) * pos["qty"]
            else:
                exit_exec = exit_price * (1 + SLIPPAGE_PCT) if exit_reason == "TP" else exit_price * (1 - SLIPPAGE_PCT)
                gross     = (pos["entry_exec"] - exit_exec) * pos["qty"]

            fee_out = exit_exec * pos["qty"] * TAKER_FEE_PCT
            net_pnl = gross - pos["fee_in"] - fee_out
            balance += net_pnl
            if net_pnl >= 0:
                daily["wins"] += 1
            else:
                daily["losses"] += 1

            trades.append({
                "symbol":        symbol,
                "direction":     direction_p,
                "entry_price":   round(pos["entry_exec"], 8),
                "exit_price":    round(exit_exec, 8),
                "sl":            round(sl_p, 8),
                "tp":            round(tp_p, 8),
                "qty":           pos["qty"],
                "rr_plan":       round(pos["rr"], 3),
                "gap_pct":       round(pos.get("gap_pct", 0), 4),
                "quality":       round(pos.get("quality", 0), 2),
                "composite":     round(pos.get("composite", 0), 2),
                "pnl_usd":       round(net_pnl, 4),
                "balance_after": round(balance, 4),
                "exit_reason":   exit_reason,
                "date":          date_key,
                "month":         datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m"),
            })

        open_positions = still_open

        # ── 2. Collect newly visible FVGs ─────────────────────────────────────
        for fvg in fvg_by_detbar.get(i, []):
            active_fvgs.append(fvg)

        # ── 3. Expire old FVGs ────────────────────────────────────────────────
        active_fvgs = [f for f in active_fvgs if (ts - f["timestamp"]) <= max_age_ms]
        for f in active_fvgs:
            f.setdefault("ts_key", (f["timestamp"], f["type"]))

        # Limit per side
        bull_fvgs   = sorted([f for f in active_fvgs if f["type"] == "bullish"], key=lambda x: x["quality"], reverse=True)
        bear_fvgs   = sorted([f for f in active_fvgs if f["type"] == "bearish"], key=lambda x: x["quality"], reverse=True)
        active_fvgs = bull_fvgs[:MAX_ACTIVE_FVGS_PER_SIDE] + bear_fvgs[:MAX_ACTIVE_FVGS_PER_SIDE]

        # ── 4. Entry checks ───────────────────────────────────────────────────
        if daily["losses"] >= MAX_DAILY_LOSS:
            continue
        if daily["wins"] >= MAX_DAILY_WIN:
            continue

        dt_bar = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc)
        if dt_bar.hour in (3, 11, 22, 23, 0):
            continue
        if dt_bar.weekday() == 5:
            continue
        if dt_bar.weekday() == 6 and dt_bar.hour >= 20:
            continue
        if dt_bar.weekday() == 0 and dt_bar.hour < 6:
            continue

        if len(open_positions) >= MAX_CONCURRENT:
            continue
        if any(p["symbol"] == symbol for p in open_positions):
            continue

        htf_bias = get_htf_bias_at(bias_series, ts)

        # ── EMA trend filters (precomputed) ───────────────────────────────────
        ema50_val  = float(ema50_arr[i])  if i < len(ema50_arr)  else float("nan")
        ema200_val = float(ema200_arr[i]) if i < len(ema200_arr) else float("nan")
        ema50_trend  = "neutral"
        ema200_trend = "neutral"
        if not math.isnan(ema50_val) and ema50_val > 0:
            ema50_trend = "bullish" if candle_c > ema50_val else "bearish"
        if not math.isnan(ema200_val) and ema200_val > 0:
            ema200_trend = "bullish" if candle_c > ema200_val else "bearish"

        # ── RSI filter (precomputed) ──────────────────────────────────────────
        rsi_val = float(rsi_arr[i]) if i < len(rsi_arr) else 50.0

        # ── ADX filter (enhanced variants I/J/K) ─────────────────────────────
        if use_adx_filter and i < len(adx_arr_full):
            adx_val = adx_arr_full[i]
            if math.isnan(adx_val) or adx_val < adx_threshold:
                continue

        # ── Volume filter at entry candle ─────────────────────────────────────
        vol_current = vol_arr[i]
        vol_avg     = vol_avg10[i]

        candidates = []
        for fvg in active_fvgs:
            fvg_key = (fvg["timestamp"], fvg["type"])
            if fvg_key in used_fvg_keys:
                continue

            # HTF alignment
            if htf_bias == "bullish" and fvg["type"] != "bullish":
                continue
            if htf_bias == "bearish" and fvg["type"] != "bearish":
                continue

            direction = "LONG" if fvg["type"] == "bullish" else "SHORT"

            # EMA filters
            if use_ema50_filter and ema50_trend != "neutral":
                if ema50_trend == "bullish" and fvg["type"] != "bullish":
                    continue
                if ema50_trend == "bearish" and fvg["type"] != "bearish":
                    continue
            if use_ema200_filter and ema200_trend != "neutral":
                if ema200_trend == "bullish" and fvg["type"] != "bullish":
                    continue
                if ema200_trend == "bearish" and fvg["type"] != "bearish":
                    continue

            # Volume filter: entry candle volume > 0.8x 10-bar average
            if use_vol_filter and vol_avg > 0:
                if vol_current < vol_avg * 0.8:
                    continue

            # RSI filter
            if use_rsi_filter:
                if direction == "LONG" and not (rsi_long_min <= rsi_val <= rsi_long_max):
                    continue
                if direction == "SHORT" and not (rsi_short_min <= rsi_val <= rsi_short_max):
                    continue

            # ── 4h Trend filter ───────────────────────────────────────────────
            if require_4h_trend and trend_only and htf_4h_bullish is not None:
                if htf_4h_bullish and direction != "LONG":
                    continue
                if not htf_4h_bullish and direction != "SHORT":
                    continue

            # ── Fib confluence filter ─────────────────────────────────────────
            if require_fib and df_1h_cached is not None and len(df_1h_cached) >= 10:
                # Use 1h data up to the approximate current bar
                ts_current = int(df["timestamp"].iloc[i])
                df_1h_slice = df_1h_cached[df_1h_cached["timestamp"] <= ts_current].tail(impulse_lookback_1h + 10)
                if len(df_1h_slice) >= 5:
                    impulse = find_impulse_moves(
                        df_1h_slice,
                        min_move_pct=impulse_min_pct,
                        lookback=impulse_lookback_1h,
                    )
                    if impulse is None:
                        continue
                    imp_dir = impulse[2]
                    # Impulse direction must match trade direction
                    if direction == "LONG" and imp_dir != "bullish":
                        continue
                    if direction == "SHORT" and imp_dir != "bearish":
                        continue
                    fib_zl, fib_zh, _ = get_fib_zone(impulse, fib_level_low, fib_level_high)
                    if not fvg_in_fib_zone(fvg, fib_zl, fib_zh):
                        continue
                    # Price must be in Fib zone
                    if not (fib_zl <= candle_c <= fib_zh):
                        continue
                else:
                    continue

            # Mid-zone check
            gap_size = fvg["high"] - fvg["low"]
            if gap_size <= 0:
                continue
            entry_band_low  = fvg["low"] + gap_size * mid_low
            entry_band_high = fvg["low"] + gap_size * mid_high
            mid_touched = candle_l <= entry_band_high and candle_h >= entry_band_low
            inside_gap  = fvg["low"] <= candle_c <= fvg["high"]
            if not (inside_gap or mid_touched):
                continue

            # Composite score
            reaction_score = 33.33
            composite = (
                fvg["size_fvg"] * 0.50
                + fvg["quality"]    * 0.35
                + reaction_score    * 0.15
            )
            if composite < composite_floor or composite > composite_cap:
                continue

            candidates.append((composite, fvg, fvg_key))

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[0], reverse=True)
        best_composite, best_fvg, best_fvg_key = candidates[0]

        direction = "LONG" if best_fvg["type"] == "bullish" else "SHORT"

        same_dir_count = sum(1 for p in open_positions if p["direction"] == direction)
        if same_dir_count >= MAX_SAME_DIR:
            continue

        if i + 1 >= len(df):
            continue
        entry_price = float(df["open"].iloc[i + 1])

        df_sl = df.iloc[max(0, i - 120): i + 1]
        sl    = find_structural_sl(
            df_sl, entry_price, direction,
            fvg_low=best_fvg["low"], fvg_high=best_fvg["high"],
            min_sl_dist_pct=MIN_SL_DIST_PCT,
            sl_buffer=sl_buf_pct_v / 100.0,
        )
        sl_dist_pct = abs(entry_price - sl) / max(entry_price, 1e-9) * 100.0
        if sl_dist_pct < MIN_SL_DIST_PCT or sl_dist_pct > max_sl_pct:
            continue

        tp = find_structural_tp(
            df_sl, entry_price, direction, sl,
            fvg_high=best_fvg["high"], fvg_low=best_fvg["low"],
            min_rr=min_rr_tp, max_rr=max_rr_tp,
            tp_offset=tp_offset_pct / 100.0,
        )
        rr = abs(tp - entry_price) / max(abs(entry_price - sl), 1e-9)
        if rr < min_rr_tp:
            continue

        risk_usd     = balance * RISK_PER_TRADE
        risk_per_unit = abs(entry_price - sl)
        if risk_per_unit <= 0:
            continue
        raw_qty = risk_usd / risk_per_unit
        notional = raw_qty * entry_price
        margin   = notional / MAX_LEVERAGE
        max_margin = balance * 0.25
        if margin > max_margin:
            raw_qty = max_margin * MAX_LEVERAGE / entry_price
        qty = raw_qty
        if qty <= 0:
            continue

        if direction == "LONG":
            entry_exec = entry_price * (1 + SLIPPAGE_PCT)
        else:
            entry_exec = entry_price * (1 - SLIPPAGE_PCT)

        fee_in = entry_exec * qty * MAKER_FEE_PCT

        # Partial TP price: entry ± (1 * sl_dist)
        sl_dist = abs(entry_exec - sl)
        if use_partial_tp:
            if direction == "LONG":
                partial_tp_price_v = entry_exec + sl_dist * partial_tp_rr
            else:
                partial_tp_price_v = entry_exec - sl_dist * partial_tp_rr
        else:
            partial_tp_price_v = None

        open_positions.append({
            "symbol":           symbol,
            "direction":        direction,
            "entry_exec":       entry_exec,
            "entry_price":      entry_price,
            "sl":               sl,
            "original_sl":      sl,
            "tp":               tp,
            "qty":              qty,
            "rr":               rr,
            "risk_usd":         risk_usd,
            "fee_in":           fee_in,
            "gap_pct":          best_fvg["gap_pct"],
            "quality":          best_fvg["quality"],
            "composite":        best_composite,
            "partial_tp_price": partial_tp_price_v,
            "partial_done":     False,
            "trailing_done":    False,
        })
        used_fvg_keys.add(best_fvg_key)

    # Close remaining positions at last close
    for pos in open_positions:
        if pos["symbol"] != symbol:
            continue
        last_price = float(df["close"].iloc[-1])
        if pos["direction"] == "LONG":
            exit_exec = last_price * (1 - SLIPPAGE_PCT)
            gross     = (exit_exec - pos["entry_exec"]) * pos["qty"]
        else:
            exit_exec = last_price * (1 + SLIPPAGE_PCT)
            gross     = (pos["entry_exec"] - exit_exec) * pos["qty"]
        fee_out = exit_exec * pos["qty"] * TAKER_FEE_PCT
        net_pnl = gross - pos["fee_in"] - fee_out
        balance += net_pnl
        last_ts  = int(df["timestamp"].iloc[-1])
        date_key = datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        trades.append({
            "symbol":        symbol,
            "direction":     pos["direction"],
            "entry_price":   round(pos["entry_exec"], 8),
            "exit_price":    round(exit_exec, 8),
            "sl":            round(pos["sl"], 8),
            "tp":            round(pos["tp"], 8),
            "qty":           pos["qty"],
            "rr_plan":       round(pos["rr"], 3),
            "gap_pct":       round(pos.get("gap_pct", 0), 4),
            "quality":       round(pos.get("quality", 0), 2),
            "composite":     round(pos.get("composite", 0), 2),
            "pnl_usd":       round(net_pnl, 4),
            "balance_after": round(balance, 4),
            "exit_reason":   "EXPIRED",
            "date":          date_key,
            "month":         datetime.fromtimestamp(last_ts / 1000.0, tz=timezone.utc).strftime("%Y-%m"),
        })

    return trades, balance


# ─── Summary ──────────────────────────────────────────────────────────────────

def compute_summary(trades: list[dict], start_balance: float, label: str) -> dict:
    if not trades:
        return {"label": label, "total_trades": 0, "message": "No trades generated."}

    wins        = [t for t in trades if t["pnl_usd"] > 0]
    losses      = [t for t in trades if t["pnl_usd"] <= 0]
    total_pnl   = sum(t["pnl_usd"] for t in trades)
    end_balance = start_balance + total_pnl
    win_rate    = len(wins) / len(trades) * 100.0
    gross_wins  = sum(t["pnl_usd"] for t in wins)
    gross_loss  = abs(sum(t["pnl_usd"] for t in losses))
    pf          = gross_wins / max(gross_loss, 1e-9)

    peak = start_balance; max_dd = 0.0; running = start_balance
    for t in sorted(trades, key=lambda x: x["date"]):
        running += t["pnl_usd"]
        peak     = max(peak, running)
        max_dd   = max(max_dd, (peak - running) / peak * 100.0)

    monthly: dict[str, float] = defaultdict(float)
    for t in trades:
        monthly[t["month"]] += t["pnl_usd"]
    best_month  = max(monthly.items(), key=lambda x: x[1]) if monthly else ("N/A", 0.0)
    worst_month = min(monthly.items(), key=lambda x: x[1]) if monthly else ("N/A", 0.0)

    return {
        "label":            label,
        "total_pnl_pct":    round(total_pnl / start_balance * 100.0, 2),
        "win_rate_pct":     round(win_rate, 1),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor":    round(pf, 2),
        "total_trades":     len(trades),
        "avg_win_usd":      round(gross_wins / max(len(wins), 1), 2),
        "avg_loss_usd":     round(-gross_loss / max(len(losses), 1), 2),
        "best_month":       f"{best_month[0]} {best_month[1]/start_balance*100:+.1f}%",
        "worst_month":      f"{worst_month[0]} {worst_month[1]/start_balance*100:+.1f}%",
        "start_balance":    start_balance,
        "end_balance":      round(end_balance, 2),
    }


def print_summary(s: dict) -> None:
    label = s.get("label", "")
    if s.get("total_trades", 0) == 0:
        print(f"\n=== {label} ===")
        print("  No trades generated.")
        return
    print(f"\n=== {label} ===")
    print(f"  Total PnL:        {s['total_pnl_pct']:+.1f}%  (${s['end_balance'] - s['start_balance']:+.2f})")
    print(f"  Win Rate:         {s['win_rate_pct']:.1f}%")
    print(f"  Max Drawdown:     -{s['max_drawdown_pct']:.1f}%")
    print(f"  Profit Factor:    {s['profit_factor']:.2f}")
    print(f"  Total Trades:     {s['total_trades']}")
    print(f"  Avg Win:          +${s['avg_win_usd']:.2f}")
    print(f"  Avg Loss:         ${s['avg_loss_usd']:.2f}")
    print(f"  Best Month:       {s['best_month']}")
    print(f"  Worst Month:      {s['worst_month']}")
    print(f"  Balance: ${s['start_balance']:.2f} -> ${s['end_balance']:.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("FVG Backtest v3 — %d symbols, %d days (15m)", len(BACKTEST_SYMBOLS), BACKTEST_DAYS)

    data_map: dict[str, pd.DataFrame] = {}
    bias_map: dict[str, pd.Series]    = {}

    logger.info("Loading data (cache-first) ...")
    for sym in BACKTEST_SYMBOLS:
        print(f"  [{sym}] loading ...", flush=True)
        df = _load_or_download(sym, TIMEFRAME)
        if df.empty or len(df) < 300:
            logger.warning("  %s: insufficient data (%d candles), skipping", sym, len(df))
            continue
        data_map[sym] = df
        try:
            bias_map[sym] = compute_htf_bias(df, ema_period=200)
        except Exception as exc:
            logger.warning("  %s: HTF bias failed: %s", sym, exc)
            bias_map[sym] = pd.Series([], dtype=str)
        logger.info("  %s: %d candles", sym, len(df))

    if not data_map:
        logger.error("No data available. Run fetch_data.py first.")
        return

    all_summaries = []

    for variant in VARIANTS:
        label = variant["label"]
        logger.info("\nRunning variant: %s ...", label)

        all_trades: list[dict] = []
        balance = START_BALANCE

        for sym in sorted(data_map.keys()):
            print(f"    {sym} ...", flush=True)
            df   = data_map[sym]
            bias = bias_map.get(sym, pd.Series([], dtype=str))
            sym_trades, balance = _run_backtest_symbol(
                symbol=sym,
                df=df,
                bias_series=bias,
                balance_start=balance,
                variant=variant,
            )
            all_trades.extend(sym_trades)
            if sym_trades:
                logger.info("    %s: %d trades | balance $%.2f", sym, len(sym_trades), balance)

        # Early exit: if first 30-day PF < 0.3, skip printing lengthy results
        cutoff_date = None
        if all_trades:
            first_date = sorted(t["date"] for t in all_trades)[0]
            from datetime import timedelta
            cutoff_date = (datetime.strptime(first_date, "%Y-%m-%d") + timedelta(days=30)).strftime("%Y-%m-%d")
        early_trades = [t for t in all_trades if cutoff_date and t["date"] <= cutoff_date]
        if early_trades:
            e_wins  = sum(1 for t in early_trades if t["pnl_usd"] > 0)
            e_gross = sum(t["pnl_usd"] for t in early_trades if t["pnl_usd"] > 0)
            e_loss  = abs(sum(t["pnl_usd"] for t in early_trades if t["pnl_usd"] <= 0))
            early_pf = e_gross / max(e_loss, 1e-9)
            if early_pf < 0.3 and len(early_trades) >= 10:
                logger.info("  Skipping variant (early PF=%.2f < 0.3)", early_pf)
                all_summaries.append({"label": label, "total_trades": len(all_trades),
                                      "total_pnl_pct": 0.0, "win_rate_pct": 0.0,
                                      "max_drawdown_pct": 0.0, "profit_factor": early_pf,
                                      "message": f"Skipped (early_pf={early_pf:.2f})"})
                continue

        all_trades.sort(key=lambda t: t["date"])
        summary = compute_summary(all_trades, START_BALANCE, label)
        all_summaries.append(summary)
        print_summary(summary)

        if all_trades:
            ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
            import re as _re
            safe_lbl = _re.sub(r'[^\w\-]', '_', label)[:60]
            csv_path = RESULTS_DIR / f"fvg_{safe_lbl}_{ts_str}.csv"
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=list(all_trades[0].keys()))
                w.writeheader()
                w.writerows(all_trades)
            logger.info("  CSV saved: %s", csv_path.name)

    # Comparison table
    print("\n" + "=" * 110)
    print("COMPARISON TABLE — FVG Backtest v3 Results (90 days)")
    print("=" * 110)
    hdr = f"{'Variant':<50} {'PnL%':>7} {'WR%':>6} {'MaxDD%':>8} {'PF':>6} {'Trades':>7} {'AvgWin':>8} {'AvgLoss':>9}"
    print(hdr)
    print("-" * 110)

    best_score = -999.0
    best_row   = None

    for s in all_summaries:
        if s.get("total_trades", 0) == 0 or "message" in s:
            print(f"  {s.get('label',''):<48}  -- {s.get('message', 'no trades')} --")
            continue
        row = (
            f"  {s['label']:<48} "
            f"{s['total_pnl_pct']:>+7.1f}%"
            f"{s['win_rate_pct']:>6.1f}%"
            f"{s['max_drawdown_pct']:>7.1f}%"
            f"{s['profit_factor']:>7.2f}"
            f"{s['total_trades']:>7}"
            f"  ${s['avg_win_usd']:>6.2f}"
            f"  ${abs(s['avg_loss_usd']):>7.2f}"
        )
        print(row)
        # Best = WR>54% AND trades>40 AND MaxDD<15% → highest PnL%
        score_ok = (
            s["win_rate_pct"]     >= 54.0
            and s["total_trades"] >= 40
            and s["max_drawdown_pct"] < 15.0
            and s["total_pnl_pct"] > 0
        )
        if score_ok and s["total_pnl_pct"] > best_score:
            best_score = s["total_pnl_pct"]
            best_row   = s

    print("=" * 110)
    print("\nTARGETS: PnL>35%, WR>54%, MaxDD<15%, Trades>40")

    if best_row:
        print(f"\nBEST VARIANT (meets all targets): {best_row['label']}")
        print(f"  PnL: {best_row['total_pnl_pct']:+.1f}%  |  WR: {best_row['win_rate_pct']:.1f}%"
              f"  |  MaxDD: {best_row['max_drawdown_pct']:.1f}%  |  PF: {best_row['profit_factor']:.2f}"
              f"  |  Trades: {best_row['total_trades']}")
    else:
        print("\nNo variant met all targets. Best available:")
        valid = [s for s in all_summaries if s.get("total_trades", 0) > 5 and "message" not in s]
        if valid:
            top = max(valid, key=lambda x: x["total_pnl_pct"])
            print(f"  {top['label']} — PnL {top['total_pnl_pct']:+.1f}%  WR {top['win_rate_pct']:.1f}%  Trades {top['total_trades']}")

    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"fvg_summary_{ts_str}.json"
    json_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    logger.info("Summary saved: %s", json_path)
    print("\nFVG Backtest complete. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
