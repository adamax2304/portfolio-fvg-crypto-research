"""
multi_agent/backtests/backtest_smc.py

SMC Strategy Backtester v4 — 90-day cache + more trade variants
Tests 7 variants on 20 symbols over 90 days.

Improvements over v3:
  - New entry modes to increase trade count to 30-50/90d
  - Mode 1: OB retest without full sweep (relaxed, recent sweep check)
  - Mode 2: Premium/Discount OB (OB in upper/lower 50% of recent range)
  - Mode 3: Double OB confirmation (stacked overlapping OBs)
  - Mode 4: Multi-TF OB (1h OB zone + 15m OB inside for precise entry)

Variants:
  A) Original (baseline)
  B) Tight SL (existing winner: body_50pct)
  C) + Mode 1 (OB retest without full sweep)
  D) + Mode 2 (Premium/Discount OB)
  E) + Mode 3 (Double OB)
  F) + Mode 4 (Multi-TF OB)
  G) Best combo (all modes, tighter params)

Target: 30-50 trades/90d, WR>56%, PnL>20%

Usage
-----
    py -3 multi_agent/backtests/backtest_smc.py
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

START_BALANCE   = 1000.0
BACKTEST_DAYS   = 90
TAKER_FEE_PCT   = 0.00055
MAKER_FEE_PCT   = 0.0002
SLIPPAGE_PCT    = 0.00025
RISK_PER_TRADE  = 0.02
MAX_LEVERAGE    = 25
MAX_CONCURRENT  = 3
MAX_DAILY_LOSS  = 4
MAX_DAILY_WIN   = 10

# ─── Variants ─────────────────────────────────────────────────────────────────

VARIANTS = [
    # A: Original baseline
    {
        "label":                "A: Original (SL=sweep_wick, TP=eq_level, cool=2)",
        "min_rr":               1.5,
        "sl_mode":              "sweep_wick",
        "tp_mode":              "eq_level",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.10,
        "swing_lookback":       20,
        "cooldown_bars":        2,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    False,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
    },
    # B: Tight SL (existing winner)
    {
        "label":                "B: Tight SL body_50pct + swing TP + cool=2",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.10,
        "swing_lookback":       20,
        "cooldown_bars":        2,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    False,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
    },
    # C: + Mode 1 (OB retest without full sweep, recent sweep check in 10 bars)
    {
        "label":                "C: +Mode1 OB_retest_no_sweep (recent_sweep_10bars, RR>=1.8)",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    True,
        "relaxed_choch_pct":    0.75,
        "use_mode1":            True,
        "mode1_min_rr":         1.8,
        "mode1_ob_vol_mult":    1.8,
        "mode1_sweep_lookback": 10,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
    },
    # D: + Mode 2 (Premium/Discount OB)
    {
        "label":                "D: +Mode2 Premium/Discount OB (range50pct filter)",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        True,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    True,
        "relaxed_choch_pct":    0.75,
        "use_mode1":            False,
        "use_mode2":            True,
        "mode2_range_lookback": 50,
        "use_mode3":            False,
        "use_mode4":            False,
    },
    # E: + Mode 3 (Double OB stacked)
    {
        "label":                "E: +Mode3 Double_OB_stacked (tighter SL)",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          1.5,
        "use_relaxed_choch":    True,
        "relaxed_choch_pct":    0.75,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            True,
        "mode3_min_overlap":    0.3,   # at least 30% overlap between two OBs
        "use_mode4":            False,
    },
    # F: + Mode 4 (Multi-TF OB: 1h zone + 15m entry)
    {
        "label":                "F: +Mode4 MultiTF_OB (1h_zone+15m_entry)",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          1.5,
        "use_relaxed_choch":    True,
        "relaxed_choch_pct":    0.75,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            True,
        "mode4_htf_vol_mult":   1.2,  # 1h OB vol threshold
    },
    # G: Best combo (all modes + relaxed params)
    {
        "label":                "G: BestCombo all_modes+relaxed (target 30-50 trades)",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        True,
        "ob_only_min_rr":       1.8,
        "ob_vol_mult":          1.5,
        "use_relaxed_choch":    True,
        "relaxed_choch_pct":    0.75,
        "use_mode1":            True,
        "mode1_min_rr":         1.8,
        "mode1_ob_vol_mult":    1.8,
        "mode1_sweep_lookback": 10,
        "use_mode2":            True,
        "mode2_range_lookback": 50,
        "use_mode3":            True,
        "mode3_min_overlap":    0.3,
        "use_mode4":            True,
        "mode4_htf_vol_mult":   1.2,
    },
    # H: Tight SL + Chandelier trailing (no fixed TP)
    {
        "label":                "H: TightSL+Chandelier_trail (no TP)",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    False,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
        "use_chandelier_trailing": True,
        "ce_length":            22,
        "ce_mult":              3.0,
        "use_adx_filter":       False,
    },
    # I: Tight SL + Chandelier trail + ADX>20
    {
        "label":                "I: TightSL+Chandelier+ADX>20",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    False,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
        "use_chandelier_trailing": True,
        "ce_length":            22,
        "ce_mult":              3.0,
        "use_adx_filter":       True,
        "adx_threshold":        20.0,
    },
    # J: BestCombo + Chandelier trail + ADX>15
    {
        "label":                "J: BestCombo+Chandelier+ADX>15",
        "min_rr":               1.5,
        "sl_mode":              "body_50pct",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.1,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        True,
        "ob_only_min_rr":       1.8,
        "ob_vol_mult":          1.5,
        "use_relaxed_choch":    True,
        "relaxed_choch_pct":    0.75,
        "use_mode1":            True,
        "mode1_min_rr":         1.8,
        "mode1_ob_vol_mult":    1.8,
        "mode1_sweep_lookback": 10,
        "use_mode2":            True,
        "mode2_range_lookback": 50,
        "use_mode3":            True,
        "mode3_min_overlap":    0.3,
        "use_mode4":            True,
        "mode4_htf_vol_mult":   1.2,
        "use_chandelier_trailing": True,
        "ce_length":            22,
        "ce_mult":              3.5,
        "use_adx_filter":       True,
        "adx_threshold":        15.0,
    },
    # K: *** UPDATED LIVE PARAMS *** — full_wick SL (sweep-safe), wider buffer, min_rr=1.0, TP closer inside
    # Matches updated config.py: SL_MODE=full_wick, SL_BUFFER_PCT=0.30, MIN_RR=1.0, TP 0.15% inside swing
    {
        "label":                "K: LiveV2 (full_wick_SL, buf=0.30, min_rr=1.0, TP_inside=0.15%)",
        "min_rr":               1.0,
        "sl_mode":              "full_wick",    # SL behind the full sweep wick
        "tp_mode":              "swing",        # TP at nearest swing structure
        "sl_buffer_pct":        0.30,           # 0.30% buffer beyond wick
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    False,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
        "use_chandelier_trailing": False,
        "tp_inside_pct":        0.15,           # TP placed 0.15% INSIDE resistance
    },
    # L: Updated live params + Chandelier trailing (no fixed TP ceiling)
    {
        "label":                "L: LiveV2+Chandelier (full_wick_SL, buf=0.30, min_rr=1.0, ADX>15)",
        "min_rr":               1.0,
        "sl_mode":              "full_wick",
        "tp_mode":              "swing",
        "sl_buffer_pct":        0.30,
        "eq_tol_pct":           0.15,
        "swing_lookback":       12,
        "cooldown_bars":        1,
        "allow_ob_only":        False,
        "ob_only_min_rr":       2.0,
        "ob_vol_mult":          2.0,
        "use_relaxed_choch":    False,
        "use_mode1":            False,
        "use_mode2":            False,
        "use_mode3":            False,
        "use_mode4":            False,
        "use_chandelier_trailing": True,
        "ce_length":            22,
        "ce_mult":              3.0,
        "use_adx_filter":       True,
        "adx_threshold":        15.0,
        "tp_inside_pct":        0.15,
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
        for _ in range(40):
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


# ─── ATR ──────────────────────────────────────────────────────────────────────

def _calc_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < 2:
        return 0.0
    prev = np.roll(closes, 1)
    prev[0] = closes[0]
    tr  = np.maximum(highs - lows, np.maximum(np.abs(highs - prev), np.abs(lows - prev)))
    rel = tr[-period:] if len(tr) >= period else tr
    return float(np.mean(rel)) if len(rel) > 0 else 0.0


# ─── HTF bias ─────────────────────────────────────────────────────────────────

def compute_htf_bias(df_15m: pd.DataFrame, ema_period: int = 200) -> pd.Series:
    df = df_15m.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    df_1h = df["close"].resample("1h").last().dropna()
    ema   = df_1h.ewm(span=ema_period, adjust=False).mean()
    df_1h_s = df_1h.shift(1).dropna()
    ema_s   = ema.shift(1).dropna()
    return pd.Series(
        np.where(df_1h_s.values > ema_s.values, "LONG", "SHORT"),
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


# ─── SMC signal detection ────────────────────────────────────────────────────

def _detect_equal_levels(
    df: pd.DataFrame, lookback: int = 50, tol_pct: float = 0.10,
) -> tuple[list[float], list[float]]:
    if df is None or len(df) < 5:
        return [], []
    tail  = df.tail(lookback)
    highs = tail["high"].astype(float).values
    lows  = tail["low"].astype(float).values
    tol   = tol_pct / 100.0
    eq_highs: list[float] = []
    eq_lows:  list[float] = []
    n = len(highs)
    for i in range(n):
        for j in range(i + 1, n):
            if highs[i] > 0 and abs(highs[i] - highs[j]) / highs[i] <= tol:
                eq_highs.append((highs[i] + highs[j]) / 2.0)
            if lows[i] > 0 and abs(lows[i] - lows[j]) / lows[i] <= tol:
                eq_lows.append((lows[i] + lows[j]) / 2.0)
    return sorted(set(round(h, 8) for h in eq_highs)), sorted(set(round(l, 8) for l in eq_lows))


def _detect_liquidity_sweep(
    df: pd.DataFrame,
    direction: str,
    eq_highs: list[float],
    eq_lows:  list[float],
    sweep_lookback: int = 5,
) -> Optional[dict]:
    if df is None or len(df) < 3:
        return None
    recent = df.tail(sweep_lookback).reset_index(drop=True)
    for i in range(len(recent) - 1, -1, -1):
        candle  = recent.iloc[i]
        c_high  = float(candle["high"])
        c_low   = float(candle["low"])
        c_open  = float(candle["open"])
        c_close = float(candle["close"])
        if direction == "LONG":
            for eq_low in eq_lows:
                if c_low < eq_low and c_close > eq_low:
                    body_low = min(c_open, c_close)
                    if body_low > eq_low:
                        return {
                            "type": "sweep_low",
                            "sweep_level": eq_low,
                            "candle_high": c_high,
                            "candle_low":  c_low,
                            "candle_open": c_open,
                            "candle_close": c_close,
                            "bar_idx": i,
                        }
        else:
            for eq_high in reversed(eq_highs):
                if c_high > eq_high and c_close < eq_high:
                    body_high = max(c_open, c_close)
                    if body_high < eq_high:
                        return {
                            "type": "sweep_high",
                            "sweep_level": eq_high,
                            "candle_high": c_high,
                            "candle_low":  c_low,
                            "candle_open": c_open,
                            "candle_close": c_close,
                            "bar_idx": i,
                        }
    return None


def _detect_choch(
    df: pd.DataFrame,
    direction: str,
    lookback: int = 20,
    relaxed: bool = False,
    relaxed_pct: float = 0.75,
) -> bool:
    """
    Detect Change of Character (CHoCH).
    relaxed=True: require 75% body close above/below swing point (not full candle close).
    """
    if df is None or len(df) < 5:
        return False
    tail   = df.tail(lookback).reset_index(drop=True)
    highs  = tail["high"].astype(float).values
    lows   = tail["low"].astype(float).values
    closes = tail["close"].astype(float).values
    opens  = tail["open"].astype(float).values
    n      = len(closes)
    if n < 4:
        return False

    last_close = closes[-1]
    last_open  = opens[-1]

    if direction == "LONG":
        mid  = n // 2
        swing_high = float(np.max(highs[:mid])) if mid > 0 else 0.0
        if relaxed:
            # 75% body close: at least 75% of candle body is above swing high
            body_high = max(last_close, last_open)
            body_low  = min(last_close, last_open)
            body_len  = body_high - body_low
            if body_len <= 0:
                return last_close > swing_high
            above_pct = (body_high - swing_high) / max(body_len, 1e-9)
            return above_pct >= relaxed_pct
        return last_close > swing_high
    else:
        mid       = n // 2
        swing_low = float(np.min(lows[:mid])) if mid > 0 else float("inf")
        if relaxed:
            body_high = max(last_close, last_open)
            body_low  = min(last_close, last_open)
            body_len  = body_high - body_low
            if body_len <= 0:
                return last_close < swing_low
            below_pct = (swing_low - body_low) / max(body_len, 1e-9)
            return below_pct >= relaxed_pct
        return last_close < swing_low


def _detect_order_blocks(
    df: pd.DataFrame,
    direction: str,
    vol_mult: float = 1.5,
) -> list[dict]:
    if df is None or len(df) < 5:
        return []
    n       = len(df)
    opens   = df["open"].astype(float).values
    highs   = df["high"].astype(float).values
    lows    = df["low"].astype(float).values
    closes  = df["close"].astype(float).values
    vols    = df["volume"].astype(float).values if "volume" in df.columns else np.ones(n)
    avg_vol = float(np.mean(vols[-20:])) if n >= 20 else float(np.mean(vols))

    obs: list[dict] = []
    lookback = min(n - 2, 80)
    start    = n - lookback
    for i in range(start, n - 1):
        o_i, h_i, l_i, c_i, v_i = opens[i], highs[i], lows[i], closes[i], vols[i]
        c_next = closes[i + 1]

        if avg_vol > 0 and v_i < vol_mult * avg_vol:
            continue

        if direction == "LONG":
            if c_i >= o_i:
                continue
            if c_next <= h_i:
                continue
            mitigated = any(lows[j] <= h_i and highs[j] >= l_i for j in range(i + 2, n))
            if not mitigated:
                obs.append({"direction": "LONG", "ob_high": h_i, "ob_low": l_i, "bar_idx": i, "volume": v_i})
        else:
            if c_i <= o_i:
                continue
            if c_next >= l_i:
                continue
            mitigated = any(highs[j] >= l_i and lows[j] <= h_i for j in range(i + 2, n))
            if not mitigated:
                obs.append({"direction": "SHORT", "ob_high": h_i, "ob_low": l_i, "bar_idx": i, "volume": v_i})
    return list(reversed(obs))


def _build_sl_tp(
    sweep: dict,
    direction: str,
    entry: float,
    atr: float,
    eq_highs: list[float],
    eq_lows:  list[float],
    df: pd.DataFrame,
    sl_mode: str = "sweep_wick",
    tp_mode: str = "eq_level",
    sl_buffer_pct: float = 0.1,
    min_rr: float = 1.5,
    max_rr: float = 3.0,
    tp_inside_pct: float = 0.20,   # how far inside structure to place TP (%)
) -> Optional[tuple[float, float, float]]:
    buf = sl_buffer_pct / 100.0

    if sl_mode in ("sweep_wick", "full_wick"):
        # SL behind the full sweep wick (sweep-safe: cannot be taken out by wick sweeps)
        sl = sweep["candle_low"] * (1.0 - buf) if direction == "LONG" else sweep["candle_high"] * (1.0 + buf)
    elif sl_mode == "body_50pct":
        c_open, c_close = sweep["candle_open"], sweep["candle_close"]
        body_mid = (min(c_open, c_close) + max(c_open, c_close)) / 2.0
        sl = body_mid * (1.0 - buf) if direction == "LONG" else body_mid * (1.0 + buf)
    else:
        sl = sweep["candle_low"] * (1.0 - buf) if direction == "LONG" else sweep["candle_high"] * (1.0 + buf)

    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return None
    min_tp_dist = sl_dist * min_rr
    max_tp_dist = sl_dist * max_rr

    if tp_mode == "eq_level":
        if direction == "LONG":
            min_tp    = entry + min_tp_dist
            candidates = [h for h in eq_highs if h > min_tp]
            tp = min(candidates) * 0.999 if candidates else entry + min_tp_dist
        else:
            min_tp    = entry - min_tp_dist
            candidates = [l for l in eq_lows if l < min_tp]
            tp = max(candidates) * 1.001 if candidates else entry - min_tp_dist
    elif tp_mode == "atr_2x":
        tp = entry + atr * 2.0 if direction == "LONG" else entry - atr * 2.0
    elif tp_mode == "swing":
        tail      = df.tail(60)
        highs_arr = tail["high"].astype(float).values
        lows_arr  = tail["low"].astype(float).values
        swing_highs, swing_lows = [], []
        nk = len(highs_arr)
        for k in range(2, nk - 2):
            if (highs_arr[k] > highs_arr[k-1] and highs_arr[k] > highs_arr[k-2]
                    and highs_arr[k] > highs_arr[k+1] and highs_arr[k] > highs_arr[k+2]):
                swing_highs.append(float(highs_arr[k]))
            if (lows_arr[k] < lows_arr[k-1] and lows_arr[k] < lows_arr[k-2]
                    and lows_arr[k] < lows_arr[k+1] and lows_arr[k] < lows_arr[k+2]):
                swing_lows.append(float(lows_arr[k]))
        _tp_inside = tp_inside_pct / 100.0
        if direction == "LONG":
            min_tp = entry + min_tp_dist
            cands  = [h for h in swing_highs if h >= min_tp]
            tp = min(cands) * (1.0 - _tp_inside) if cands else entry + min_tp_dist
        else:
            min_tp = entry - min_tp_dist
            cands  = [l for l in swing_lows if l <= min_tp]
            tp = max(cands) * (1.0 + _tp_inside) if cands else entry - min_tp_dist
    else:
        tp = entry + min_tp_dist if direction == "LONG" else entry - min_tp_dist

    rr = abs(tp - entry) / max(sl_dist, 1e-9)
    if rr < min_rr:
        return None
    if rr > max_rr:
        tp = entry + sl_dist * max_rr if direction == "LONG" else entry - sl_dist * max_rr
        rr = max_rr

    return sl, tp, rr


def _build_ob_only_sl_tp(
    ob: dict,
    direction: str,
    entry: float,
    atr: float,
    df: pd.DataFrame,
    sl_buffer_pct: float = 0.1,
    min_rr: float = 2.0,
    max_rr: float = 3.0,
    tp_inside_pct: float = 0.20,
) -> Optional[tuple[float, float, float]]:
    """Build SL/TP for OB-only entries (no sweep required)."""
    buf = sl_buffer_pct / 100.0
    if direction == "LONG":
        sl = ob["ob_low"] * (1.0 - buf)
    else:
        sl = ob["ob_high"] * (1.0 + buf)

    sl_dist = abs(entry - sl)
    if sl_dist <= 0:
        return None

    # TP: swing-level based
    tail      = df.tail(60)
    highs_arr = tail["high"].astype(float).values
    lows_arr  = tail["low"].astype(float).values
    swing_highs, swing_lows = [], []
    nk = len(highs_arr)
    for k in range(2, nk - 2):
        if (highs_arr[k] > highs_arr[k-1] and highs_arr[k] > highs_arr[k-2]
                and highs_arr[k] > highs_arr[k+1] and highs_arr[k] > highs_arr[k+2]):
            swing_highs.append(float(highs_arr[k]))
        if (lows_arr[k] < lows_arr[k-1] and lows_arr[k] < lows_arr[k-2]
                and lows_arr[k] < lows_arr[k+1] and lows_arr[k] < lows_arr[k+2]):
            swing_lows.append(float(lows_arr[k]))

    _tp_inside = tp_inside_pct / 100.0
    min_tp_dist = sl_dist * min_rr
    if direction == "LONG":
        min_tp = entry + min_tp_dist
        cands  = [h for h in swing_highs if h >= min_tp]
        tp = min(cands) * (1.0 - _tp_inside) if cands else entry + min_tp_dist
    else:
        min_tp = entry - min_tp_dist
        cands  = [l for l in swing_lows if l <= min_tp]
        tp = max(cands) * (1.0 + _tp_inside) if cands else entry - min_tp_dist

    rr = abs(tp - entry) / max(sl_dist, 1e-9)
    if rr < min_rr:
        return None
    if rr > max_rr:
        tp = entry + sl_dist * max_rr if direction == "LONG" else entry - sl_dist * max_rr
        rr = max_rr
    return sl, tp, rr


# ─── New entry mode helpers ───────────────────────────────────────────────────

def _detect_recent_sweep(
    df: pd.DataFrame,
    direction: str,
    eq_highs: list[float],
    eq_lows: list[float],
    lookback: int = 10,
) -> bool:
    """
    Mode 1: Check if there was a liquidity sweep in the last `lookback` bars.
    Less strict than standard sweep — just checks wick beyond EQ level.
    """
    if df is None or len(df) < 3:
        return False
    recent = df.tail(lookback).reset_index(drop=True)
    for i in range(len(recent)):
        c = recent.iloc[i]
        if direction == "LONG":
            for eq_low in eq_lows:
                if float(c["low"]) < eq_low:
                    return True
        else:
            for eq_high in eq_highs:
                if float(c["high"]) > eq_high:
                    return True
    return False


def _get_range_midpoint(df: pd.DataFrame, lookback: int = 50) -> float:
    """Get midpoint of the price range over last `lookback` bars."""
    tail = df.tail(lookback)
    if tail.empty:
        return 0.0
    h = float(tail["high"].max())
    l = float(tail["low"].min())
    return (h + l) / 2.0


def _detect_discount_premium_ob(
    df: pd.DataFrame,
    direction: str,
    vol_mult: float = 1.5,
    range_lookback: int = 50,
) -> list[dict]:
    """
    Mode 2: Find OBs in premium (upper 50% for SHORT) or discount (lower 50% for LONG) zone.
    Returns list of qualifying OBs.
    """
    mid = _get_range_midpoint(df, lookback=range_lookback)
    if mid <= 0:
        return []
    obs = _detect_order_blocks(df, direction, vol_mult=vol_mult)
    result = []
    for ob in obs:
        ob_mid = (ob["ob_high"] + ob["ob_low"]) / 2.0
        if direction == "LONG" and ob_mid < mid:   # discount zone
            result.append(ob)
        elif direction == "SHORT" and ob_mid > mid: # premium zone
            result.append(ob)
    return result


def _detect_double_ob(
    df: pd.DataFrame,
    direction: str,
    vol_mult: float = 1.5,
    min_overlap_frac: float = 0.3,
) -> Optional[dict]:
    """
    Mode 3: Find two stacked OBs that overlap. Returns merged/combined zone.
    Overlap check: overlap_size / min(ob1_size, ob2_size) >= min_overlap_frac.
    """
    obs = _detect_order_blocks(df, direction, vol_mult=vol_mult)
    if len(obs) < 2:
        return None
    for a in range(len(obs)):
        for b in range(a + 1, len(obs)):
            ob1 = obs[a]
            ob2 = obs[b]
            # Overlap calculation
            overlap_low  = max(ob1["ob_low"],  ob2["ob_low"])
            overlap_high = min(ob1["ob_high"], ob2["ob_high"])
            if overlap_high <= overlap_low:
                continue
            overlap_size = overlap_high - overlap_low
            size1 = ob1["ob_high"] - ob1["ob_low"]
            size2 = ob2["ob_high"] - ob2["ob_low"]
            min_size = min(size1, size2)
            if min_size <= 0:
                continue
            if overlap_size / min_size >= min_overlap_frac:
                # Return the tighter combined zone
                return {
                    "direction": direction,
                    "ob_high": overlap_high,
                    "ob_low":  overlap_low,
                    "bar_idx": ob1["bar_idx"],
                    "volume":  max(ob1["volume"], ob2["volume"]),
                    "double_ob": True,
                }
    return None


def _resample_to_1h_from_15m(df_15m: pd.DataFrame) -> pd.DataFrame:
    """Resample 15m OHLCV DataFrame to 1h."""
    df = df_15m.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("dt")
    df_1h = df[["open", "high", "low", "close", "volume"]].resample("1h").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    ).dropna().reset_index()
    df_1h["timestamp"] = df_1h["dt"].astype(np.int64) // 1_000_000
    return df_1h


def _detect_multitf_ob(
    df_15m: pd.DataFrame,
    direction: str,
    htf_vol_mult: float = 1.2,
    lf_vol_mult: float = 1.5,
) -> Optional[dict]:
    """
    Mode 4: 1h OB defines zone, 15m OB inside it for precise entry.
    Returns 15m OB if it falls within a 1h OB zone.
    """
    try:
        df_1h = _resample_to_1h_from_15m(df_15m)
    except Exception:
        return None
    if len(df_1h) < 10:
        return None

    htf_obs = _detect_order_blocks(df_1h, direction, vol_mult=htf_vol_mult)
    if not htf_obs:
        return None

    # Take the most recent HTF OB
    best_htf = htf_obs[0]
    htf_low  = best_htf["ob_low"]
    htf_high = best_htf["ob_high"]

    # Find 15m OB inside the 1h OB zone
    ltf_obs = _detect_order_blocks(df_15m, direction, vol_mult=lf_vol_mult)
    for ob in ltf_obs:
        # OB must be fully inside the HTF zone
        if ob["ob_low"] >= htf_low and ob["ob_high"] <= htf_high:
            ob["multitf"] = True
            ob["htf_ob_low"]  = htf_low
            ob["htf_ob_high"] = htf_high
            return ob
    return None


# ─── Single-symbol backtest ───────────────────────────────────────────────────

def _run_backtest_symbol(
    symbol: str,
    df: pd.DataFrame,
    bias_series: pd.Series,
    balance_start: float,
    variant: dict,
    warmup_bars: int = 250,
) -> tuple[list[dict], float]:
    """Walk-forward SMC backtest on a single symbol."""
    trades: list[dict] = []
    balance = balance_start

    min_rr          = float(variant["min_rr"])
    sl_mode         = str(variant["sl_mode"])
    tp_mode         = str(variant["tp_mode"])
    sl_buf_pct      = float(variant["sl_buffer_pct"])
    eq_tol_pct      = float(variant["eq_tol_pct"])
    swing_lookback  = int(variant["swing_lookback"])
    cooldown_bars   = int(variant["cooldown_bars"])
    allow_ob_only      = bool(variant.get("allow_ob_only", False))
    ob_only_min_rr     = float(variant.get("ob_only_min_rr", 2.0))
    ob_vol_mult        = float(variant.get("ob_vol_mult", 2.0))
    use_relaxed_choch  = bool(variant.get("use_relaxed_choch", False))
    relaxed_choch_pct  = float(variant.get("relaxed_choch_pct", 0.75))
    # New mode flags
    use_mode1          = bool(variant.get("use_mode1", False))
    mode1_min_rr       = float(variant.get("mode1_min_rr", 1.8))
    mode1_ob_vol_mult  = float(variant.get("mode1_ob_vol_mult", 1.8))
    mode1_sweep_lb     = int(variant.get("mode1_sweep_lookback", 10))
    use_mode2          = bool(variant.get("use_mode2", False))
    mode2_range_lb     = int(variant.get("mode2_range_lookback", 50))
    use_mode3          = bool(variant.get("use_mode3", False))
    mode3_min_overlap  = float(variant.get("mode3_min_overlap", 0.3))
    use_mode4          = bool(variant.get("use_mode4", False))
    mode4_htf_vol_mult = float(variant.get("mode4_htf_vol_mult", 1.2))
    # BB Squeeze enhancements
    use_chandelier_trail = bool(variant.get("use_chandelier_trailing", False))
    ce_length            = int(variant.get("ce_length", 22))
    ce_mult_v            = float(variant.get("ce_mult", 3.0))
    use_adx_filter       = bool(variant.get("use_adx_filter", False))
    adx_threshold        = float(variant.get("adx_threshold", 20.0))
    # Updated SL/TP params (v2 live params)
    tp_inside_pct        = float(variant.get("tp_inside_pct", 0.20))   # % inside resistance for TP

    if len(df) < warmup_bars + 10:
        return trades, balance

    # Precompute ADX + Chandelier ATR arrays (enhanced variants)
    high_arr_full = df["high"].values.astype(float)
    low_arr_full  = df["low"].values.astype(float)
    close_arr_full = df["close"].values.astype(float)
    adx_arr_full = _compute_adx(high_arr_full, low_arr_full, close_arr_full, period=14)
    ce_atr_arr   = _ce_atr_calc(high_arr_full, low_arr_full, close_arr_full, period=ce_length)

    daily_counts:   dict[str, dict[str, int]] = {}
    open_positions: list[dict]                = []
    last_trade_bar: int                       = -cooldown_bars - 1

    for i in range(warmup_bars, len(df) - 1):
        ts       = int(df["timestamp"].iloc[i])
        candle_h = float(df["high"].iloc[i])
        candle_l = float(df["low"].iloc[i])
        candle_c = float(df["close"].iloc[i])

        date_key = datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m-%d")
        daily    = daily_counts.setdefault(date_key, {"wins": 0, "losses": 0})

        # ── 1. Exit open positions ─────────────────────────────────────────────
        still_open = []
        for pos in open_positions:
            if pos["symbol"] != symbol:
                still_open.append(pos)
                continue

            # Chandelier trailing SL update
            if use_chandelier_trail and i > ce_length:
                new_ce_sl = chandelier_exit_sl(
                    high_arr_full, low_arr_full, ce_atr_arr,
                    pos["direction"], i, ce_length=ce_length, ce_mult=ce_mult_v,
                )
                entry_e = pos["entry_exec"]
                risk_p = abs(entry_e - pos.get("original_sl", pos["sl"]))
                pnl_r = 0.0
                if risk_p > 0:
                    if pos["direction"] == "LONG":
                        pnl_r = (candle_c - entry_e) / risk_p
                    else:
                        pnl_r = (entry_e - candle_c) / risk_p
                if pnl_r >= 1.0:
                    if pos["direction"] == "LONG":
                        pos["sl"] = max(pos["sl"], new_ce_sl)
                    else:
                        if new_ce_sl > 0:
                            pos["sl"] = min(pos["sl"], new_ce_sl) if pos["sl"] > 0 else new_ce_sl

            exit_price  = None
            exit_reason = ""
            if use_chandelier_trail:
                # Chandelier mode: only trailing SL, no fixed TP
                if pos["direction"] == "LONG":
                    if candle_l <= pos["sl"]:
                        exit_price = pos["sl"]; exit_reason = "Trail_SL"
                else:
                    if candle_h >= pos["sl"]:
                        exit_price = pos["sl"]; exit_reason = "Trail_SL"
                # Max 5R safety exit
                if exit_price is None:
                    entry_e = pos["entry_exec"]
                    risk_p = abs(entry_e - pos.get("original_sl", pos["sl"]))
                    if risk_p > 0:
                        if pos["direction"] == "LONG" and (candle_h - entry_e) / risk_p >= 5.0:
                            exit_price = candle_c; exit_reason = "TP_MAX"
                        elif pos["direction"] == "SHORT" and (entry_e - candle_l) / risk_p >= 5.0:
                            exit_price = candle_c; exit_reason = "TP_MAX"
            else:
                if pos["direction"] == "LONG":
                    if candle_l <= pos["sl"] and candle_h >= pos["tp"]:
                        exit_price = pos["sl"]; exit_reason = "SL"
                    elif candle_l <= pos["sl"]:
                        exit_price = pos["sl"]; exit_reason = "SL"
                    elif candle_h >= pos["tp"]:
                        exit_price = pos["tp"]; exit_reason = "TP"
                else:
                    if candle_h >= pos["sl"] and candle_l <= pos["tp"]:
                        exit_price = pos["sl"]; exit_reason = "SL"
                    elif candle_h >= pos["sl"]:
                        exit_price = pos["sl"]; exit_reason = "SL"
                    elif candle_l <= pos["tp"]:
                        exit_price = pos["tp"]; exit_reason = "TP"

            if exit_price is None:
                still_open.append(pos)
                continue

            if pos["direction"] == "LONG":
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
            last_trade_bar = i

            trades.append({
                "symbol":        symbol,
                "direction":     pos["direction"],
                "entry_price":   round(pos["entry_exec"], 8),
                "exit_price":    round(exit_exec, 8),
                "sl":            round(pos["sl"], 8),
                "tp":            round(pos["tp"], 8),
                "qty":           pos["qty"],
                "rr":            round(pos["rr"], 3),
                "pnl_usd":       round(net_pnl, 4),
                "balance_after": round(balance, 4),
                "exit_reason":   exit_reason,
                "entry_type":    pos.get("entry_type", "sweep"),
                "date":          date_key,
                "month":         datetime.fromtimestamp(ts / 1000.0, tz=timezone.utc).strftime("%Y-%m"),
            })

        open_positions = still_open

        # ── 2. Entry checks ────────────────────────────────────────────────────
        if (i - last_trade_bar) < cooldown_bars:
            continue
        if daily["losses"] >= MAX_DAILY_LOSS:
            continue
        if daily["wins"] >= MAX_DAILY_WIN:
            continue
        if len(open_positions) >= MAX_CONCURRENT:
            continue
        if any(p["symbol"] == symbol for p in open_positions):
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

        htf_bias = get_htf_bias_at(bias_series, ts)
        if htf_bias == "neutral":
            continue
        direction = htf_bias

        # ADX filter (enhanced variants H/I/J)
        if use_adx_filter and i < len(adx_arr_full):
            adx_val = adx_arr_full[i]
            if np.isnan(adx_val) or adx_val < adx_threshold:
                continue

        df_slice = df.iloc[max(0, i - 100): i + 1]

        # ── ATR precomputed for this slice ─────────────────────────────────────
        df_atr = df_slice.tail(20)
        atr = _calc_atr(
            df_atr["high"].astype(float).values,
            df_atr["low"].astype(float).values,
            df_atr["close"].astype(float).values,
            period=14,
        )
        if atr <= 0:
            continue

        # ── Standard path: sweep + CHoCH + OB ─────────────────────────────────
        eq_highs, eq_lows = _detect_equal_levels(
            df_slice, lookback=50, tol_pct=eq_tol_pct,
        )
        sweep = None
        entry_type = "sweep"

        if eq_highs or eq_lows:
            sweep = _detect_liquidity_sweep(
                df_slice, direction, eq_highs, eq_lows, sweep_lookback=5,
            )

        sl_tp_result = None
        entry_price  = None

        if sweep is not None:
            choch_ok = _detect_choch(
                df_slice, direction, lookback=swing_lookback,
                relaxed=use_relaxed_choch, relaxed_pct=relaxed_choch_pct,
            )
            if choch_ok:
                obs = _detect_order_blocks(
                    df_slice, direction, vol_mult=float(cfg.SMC_OB_VOLUME_MULTIPLIER),
                )
                if obs:
                    best_ob = obs[0]
                    ob_low  = float(best_ob["ob_low"])
                    ob_high = float(best_ob["ob_high"])
                    if (ob_low * 0.999 <= candle_c <= ob_high * 1.001):
                        if i + 1 < len(df):
                            entry_price = float(df["open"].iloc[i + 1])
                            sl_tp_result = _build_sl_tp(
                                sweep=sweep,
                                direction=direction,
                                entry=entry_price,
                                atr=atr,
                                eq_highs=eq_highs,
                                eq_lows=eq_lows,
                                df=df_slice,
                                sl_mode=sl_mode,
                                tp_mode=tp_mode,
                                sl_buffer_pct=sl_buf_pct,
                                min_rr=min_rr,
                                max_rr=3.0,
                                tp_inside_pct=tp_inside_pct,
                            )
                            entry_type = "sweep"

        # ── OB-only fallback ────────────────────────────────────────────────────
        if sl_tp_result is None and allow_ob_only:
            obs_high_vol = _detect_order_blocks(
                df_slice, direction, vol_mult=ob_vol_mult,
            )
            for ob in obs_high_vol:
                ob_low  = float(ob["ob_low"])
                ob_high = float(ob["ob_high"])
                if not (ob_low * 0.999 <= candle_c <= ob_high * 1.001):
                    continue
                if i + 1 >= len(df):
                    continue
                ep = float(df["open"].iloc[i + 1])
                res = _build_ob_only_sl_tp(
                    ob=ob,
                    direction=direction,
                    entry=ep,
                    atr=atr,
                    df=df_slice,
                    sl_buffer_pct=sl_buf_pct,
                    min_rr=ob_only_min_rr,
                    max_rr=3.0,
                    tp_inside_pct=tp_inside_pct,
                )
                if res is not None:
                    entry_price  = ep
                    sl_tp_result = res
                    entry_type   = "ob_only"
                    break

        # ── Mode 1: OB retest without full sweep ─────────────────────────────
        if sl_tp_result is None and use_mode1:
            had_recent_sweep = _detect_recent_sweep(
                df_slice, direction, eq_highs, eq_lows, lookback=mode1_sweep_lb,
            )
            if had_recent_sweep:
                obs_m1 = _detect_order_blocks(
                    df_slice, direction, vol_mult=mode1_ob_vol_mult,
                )
                for ob in obs_m1:
                    ob_low  = float(ob["ob_low"])
                    ob_high = float(ob["ob_high"])
                    if not (ob_low * 0.999 <= candle_c <= ob_high * 1.001):
                        continue
                    if i + 1 >= len(df):
                        continue
                    ep = float(df["open"].iloc[i + 1])
                    res = _build_ob_only_sl_tp(
                        ob=ob,
                        direction=direction,
                        entry=ep,
                        atr=atr,
                        df=df_slice,
                        sl_buffer_pct=sl_buf_pct,
                        min_rr=mode1_min_rr,
                        max_rr=3.0,
                        tp_inside_pct=tp_inside_pct,
                    )
                    if res is not None:
                        entry_price  = ep
                        sl_tp_result = res
                        entry_type   = "mode1_ob_retest"
                        break

        # ── Mode 2: Premium/Discount OB ──────────────────────────────────────
        if sl_tp_result is None and use_mode2:
            obs_pd = _detect_discount_premium_ob(
                df_slice, direction,
                vol_mult=ob_vol_mult,
                range_lookback=mode2_range_lb,
            )
            for ob in obs_pd:
                ob_low  = float(ob["ob_low"])
                ob_high = float(ob["ob_high"])
                if not (ob_low * 0.999 <= candle_c <= ob_high * 1.001):
                    continue
                if i + 1 >= len(df):
                    continue
                ep = float(df["open"].iloc[i + 1])
                res = _build_ob_only_sl_tp(
                    ob=ob,
                    direction=direction,
                    entry=ep,
                    atr=atr,
                    df=df_slice,
                    sl_buffer_pct=sl_buf_pct,
                    min_rr=ob_only_min_rr,
                    max_rr=3.0,
                    tp_inside_pct=tp_inside_pct,
                )
                if res is not None:
                    entry_price  = ep
                    sl_tp_result = res
                    entry_type   = "mode2_pd_ob"
                    break

        # ── Mode 3: Double OB confirmation ───────────────────────────────────
        if sl_tp_result is None and use_mode3:
            double_ob = _detect_double_ob(
                df_slice, direction,
                vol_mult=ob_vol_mult,
                min_overlap_frac=mode3_min_overlap,
            )
            if double_ob is not None:
                ob_low  = float(double_ob["ob_low"])
                ob_high = float(double_ob["ob_high"])
                if ob_low * 0.999 <= candle_c <= ob_high * 1.001 and i + 1 < len(df):
                    ep = float(df["open"].iloc[i + 1])
                    res = _build_ob_only_sl_tp(
                        ob=double_ob,
                        direction=direction,
                        entry=ep,
                        atr=atr,
                        df=df_slice,
                        sl_buffer_pct=sl_buf_pct,
                        min_rr=min_rr,
                        max_rr=3.0,
                        tp_inside_pct=tp_inside_pct,
                    )
                    if res is not None:
                        entry_price  = ep
                        sl_tp_result = res
                        entry_type   = "mode3_double_ob"

        # ── Mode 4: Multi-TF OB (1h zone + 15m entry) ────────────────────────
        if sl_tp_result is None and use_mode4:
            # Pass the full 15m data slice for 1h resampling
            df_m4 = df.iloc[max(0, i - 300): i + 1]
            multitf_ob = _detect_multitf_ob(
                df_m4, direction,
                htf_vol_mult=mode4_htf_vol_mult,
                lf_vol_mult=ob_vol_mult,
            )
            if multitf_ob is not None:
                ob_low  = float(multitf_ob["ob_low"])
                ob_high = float(multitf_ob["ob_high"])
                if ob_low * 0.999 <= candle_c <= ob_high * 1.001 and i + 1 < len(df):
                    ep = float(df["open"].iloc[i + 1])
                    res = _build_ob_only_sl_tp(
                        ob=multitf_ob,
                        direction=direction,
                        entry=ep,
                        atr=atr,
                        df=df_slice,
                        sl_buffer_pct=sl_buf_pct,
                        min_rr=min_rr,
                        max_rr=3.0,
                        tp_inside_pct=tp_inside_pct,
                    )
                    if res is not None:
                        entry_price  = ep
                        sl_tp_result = res
                        entry_type   = "mode4_multitf_ob"

        if sl_tp_result is None or entry_price is None:
            continue

        sl, tp, rr = sl_tp_result

        # Min SL distance 0.5%
        sl_dist_pct = abs(entry_price - sl) / max(entry_price, 1e-9) * 100.0
        if sl_dist_pct < 0.5:
            continue

        # Position sizing
        risk_usd      = balance * RISK_PER_TRADE
        risk_per_unit = abs(entry_price - sl)
        if risk_per_unit <= 0:
            continue
        raw_qty   = risk_usd / risk_per_unit
        notional  = raw_qty * entry_price
        margin    = notional / MAX_LEVERAGE
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

        open_positions.append({
            "symbol":      symbol,
            "direction":   direction,
            "entry_exec":  entry_exec,
            "entry_price": entry_price,
            "sl":          sl,
            "original_sl": sl,
            "tp":          tp,
            "qty":         qty,
            "rr":          rr,
            "risk_usd":    risk_usd,
            "fee_in":      fee_in,
            "entry_type":  entry_type,
        })

    # Close remaining at last close
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
            "rr":            round(pos["rr"], 3),
            "pnl_usd":       round(net_pnl, 4),
            "balance_after": round(balance, 4),
            "exit_reason":   "EXPIRED",
            "entry_type":    pos.get("entry_type", "sweep"),
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

    # Count by entry type
    ob_only    = sum(1 for t in trades if t.get("entry_type") == "ob_only")
    mode1_cnt  = sum(1 for t in trades if t.get("entry_type") == "mode1_ob_retest")
    mode2_cnt  = sum(1 for t in trades if t.get("entry_type") == "mode2_pd_ob")
    mode3_cnt  = sum(1 for t in trades if t.get("entry_type") == "mode3_double_ob")
    mode4_cnt  = sum(1 for t in trades if t.get("entry_type") == "mode4_multitf_ob")
    new_mode_trades = mode1_cnt + mode2_cnt + mode3_cnt + mode4_cnt

    return {
        "label":            label,
        "total_pnl_pct":    round(total_pnl / start_balance * 100.0, 2),
        "win_rate_pct":     round(win_rate, 1),
        "max_drawdown_pct": round(max_dd, 2),
        "profit_factor":    round(pf, 2),
        "total_trades":     len(trades),
        "ob_only_trades":   ob_only,
        "new_mode_trades":  new_mode_trades,
        "mode1_trades":     mode1_cnt,
        "mode2_trades":     mode2_cnt,
        "mode3_trades":     mode3_cnt,
        "mode4_trades":     mode4_cnt,
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
    ob_str   = f" OB-only:{s.get('ob_only_trades',0)}" if s.get("ob_only_trades", 0) > 0 else ""
    mode_str = ""
    if s.get("new_mode_trades", 0) > 0:
        mode_str = (f" | new-modes:{s['new_mode_trades']}"
                    f" [M1:{s.get('mode1_trades',0)}"
                    f" M2:{s.get('mode2_trades',0)}"
                    f" M3:{s.get('mode3_trades',0)}"
                    f" M4:{s.get('mode4_trades',0)}]")
    print(f"\n=== {label} ===")
    print(f"  Total PnL:        {s['total_pnl_pct']:+.1f}%  (${s['end_balance'] - s['start_balance']:+.2f})")
    print(f"  Win Rate:         {s['win_rate_pct']:.1f}%")
    print(f"  Max Drawdown:     -{s['max_drawdown_pct']:.1f}%")
    print(f"  Profit Factor:    {s['profit_factor']:.2f}")
    print(f"  Total Trades:     {s['total_trades']}{ob_str}{mode_str}")
    print(f"  Avg Win:          +${s['avg_win_usd']:.2f}")
    print(f"  Avg Loss:         ${abs(s['avg_loss_usd']):.2f}")
    print(f"  Best Month:       {s['best_month']}")
    print(f"  Worst Month:      {s['worst_month']}")
    print(f"  Balance: ${s['start_balance']:.2f} -> ${s['end_balance']:.2f}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    logger.info("SMC Backtest v4 — %d symbols, %d days", len(BACKTEST_SYMBOLS), BACKTEST_DAYS)

    data_map: dict[str, pd.DataFrame] = {}
    bias_map: dict[str, pd.Series]    = {}

    logger.info("Loading data (cache-first) ...")
    for sym in BACKTEST_SYMBOLS:
        print(f"  [{sym}] loading ...", flush=True)
        df = _load_or_download(sym, "15m")
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

        all_trades.sort(key=lambda t: t["date"])
        summary = compute_summary(all_trades, START_BALANCE, label)
        all_summaries.append(summary)
        print_summary(summary)

        if all_trades:
            ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
            import re as _re
            safe_lbl = _re.sub(r'[^\w\-]', '_', label)[:60]
            csv_path = RESULTS_DIR / f"smc_{safe_lbl}_{ts_str}.csv"
            keys = list(all_trades[0].keys())
            with csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                w.writerows(all_trades)
            logger.info("  CSV saved: %s", csv_path.name)

    # Comparison table
    print("\n" + "=" * 120)
    print("COMPARISON TABLE — SMC Backtest v4 Results (90 days)")
    print("=" * 120)
    hdr = f"{'Variant':<45} {'PnL%':>7} {'WR%':>6} {'MaxDD%':>8} {'PF':>6} {'Trades':>7} {'OBonly':>7} {'NewMode':>8} {'AvgWin':>8}"
    print(hdr)
    print("-" * 120)

    best_score = -999.0
    best_row   = None

    for s in all_summaries:
        if s.get("total_trades", 0) == 0:
            print(f"  {s.get('label',''):<43}  -- no trades --")
            continue
        row = (
            f"  {s['label']:<43} "
            f"{s['total_pnl_pct']:>+7.1f}%"
            f"{s['win_rate_pct']:>6.1f}%"
            f"{s['max_drawdown_pct']:>7.1f}%"
            f"{s['profit_factor']:>7.2f}"
            f"{s['total_trades']:>7}"
            f"{s.get('ob_only_trades', 0):>7}"
            f"{s.get('new_mode_trades', 0):>8}"
            f"  ${s['avg_win_usd']:>6.2f}"
        )
        print(row)
        # Best = WR>56% AND trades>25 AND MaxDD<20% AND PnL>20%
        score_ok = (
            s["win_rate_pct"]     >= 56.0
            and s["total_trades"] >= 25
            and s["max_drawdown_pct"] < 20.0
            and s["total_pnl_pct"]   > 20.0
        )
        if score_ok and s["total_pnl_pct"] > best_score:
            best_score = s["total_pnl_pct"]
            best_row   = s

    print("=" * 120)
    print("\nTARGETS: PnL>20%, WR>56%, Trades>25")

    if best_row:
        print(f"\nBEST VARIANT (meets all targets): {best_row['label']}")
        print(f"  PnL: {best_row['total_pnl_pct']:+.1f}%  |  WR: {best_row['win_rate_pct']:.1f}%"
              f"  |  MaxDD: {best_row['max_drawdown_pct']:.1f}%  |  PF: {best_row['profit_factor']:.2f}"
              f"  |  Trades: {best_row['total_trades']}")
    else:
        print("\nNo variant met all targets. Best available:")
        valid = [s for s in all_summaries if s.get("total_trades", 0) > 3]
        if valid:
            top = max(valid, key=lambda x: x["total_pnl_pct"])
            print(f"  {top['label']} — PnL {top['total_pnl_pct']:+.1f}%  WR {top['win_rate_pct']:.1f}%  Trades {top['total_trades']}")

    ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = RESULTS_DIR / f"smc_summary_{ts_str}.json"
    json_path.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
    logger.info("Summary saved: %s", json_path)
    print("\nSMC Backtest v4 complete. Results in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
