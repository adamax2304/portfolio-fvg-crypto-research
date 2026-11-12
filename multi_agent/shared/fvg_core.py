"""
multi_agent/shared/fvg_core.py

Shared FVG detection core — exact copy of the relevant functions and classes
from fvg_bot.py so that the FVG agent in the multi-agent system uses the same
detection, quality-scoring and entry-confirmation logic as the standalone bot.

Functions/classes copied from fvg_bot.py (v2.6):
  - FVG                        dataclass
  - SignalCandidate             dataclass
  - _collect_all_gap_diffs      helper
  - _atr_pct_from_arrays        helper
  - _market_regime_adjustment   helper
  - _fvg_quality_score          quality scorer
  - update_fvg_lifecycle        lifecycle updater
  - evaluate_entry_reaction     entry-confirmation check
  - composite_signal_score      blended ranking score
  - detect_fvgs_v2              main FVG detector (v2 two-stage)
  - price_in_mid_zone           mid-zone check
  - distance_to_mid_zone_pct    distance to mid-zone
  - compute_ema                 EMA helper
  - build_symbol_profile        adaptive per-symbol profile
  - collect_swing_levels        structural swing levels
  - find_swing_levels           nearest swing low/high
  - build_local_volume_profile  local HVN/POC approximation
  - extract_structural_levels   support/resistance clustering
  - _select_active_fvgs         standalone (was a method on FVGTradingBot)

All CONFIG references use the FVG_CORE_CONFIG dict defined at the top of this
module, which is initialised from multi_agent/config.py so that values stay
in sync with the rest of the multi-agent system.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

log = logging.getLogger(__name__)

# ─── Bridge: map fvg_bot.py CONFIG keys to multi_agent/config.py values ──────
# Only the keys actually used by the copied functions are listed here.
# update_fvg_lifecycle, detect_fvgs_v2, build_symbol_profile etc. all read
# from this dict instead of the original singleton CONFIG.

FVG_CORE_CONFIG: dict[str, Any] = {
    # Gap detection
    "MIN_GAP_PERCENT":               cfg.FVG_MIN_GAP_PERCENT,       # 0.5
    "FVG_MIN_GAP_MULT":              cfg.FVG_MIN_GAP_MULT,           # 1.26
    "FVG_PERCENTILE_THRESHOLD":      cfg.FVG_PERCENTILE_THRESHOLD,   # 17.2
    "FVG_PERCENTILE_LOOKBACK":       cfg.FVG_PERCENTILE_LOOKBACK,    # 500
    "FVG_LOOKBACK":                  cfg.FVG_LOOKBACK,               # 384
    "FVG_MAX_ACTIVE_PER_SIDE":       cfg.FVG_MAX_ACTIVE_PER_SIDE,    # 3
    "FVG_MAX_AGE_DAYS":              cfg.FVG_MAX_AGE_DAYS,           # 4
    "FVG_PRICE_MODE":                cfg.FVG_PRICE_MODE,             # "wick"
    "FVG_REQUIRE_DISPLACEMENT":      cfg.FVG_REQUIRE_DISPLACEMENT,   # True
    "FVG_DISPLACEMENT_BODY_MULT":    cfg.FVG_DISPLACEMENT_BODY_MULT, # 1.2
    "FVG_DISPLACEMENT_BODY_PCT":     cfg.FVG_DISPLACEMENT_BODY_PCT,  # 0.15
    "FVG_REQUIRE_SAME_DIRECTION":    cfg.FVG_REQUIRE_SAME_DIRECTION, # True
    "FVG_QUALITY_MIN_SCORE":         cfg.FVG_QUALITY_MIN_SCORE,      # 47.0
    "FVG_MITIGATION_MODE":           cfg.FVG_MITIGATION_MODE,        # "close_mid"
    "FVG_PARTIAL_FILL_PCT":          cfg.FVG_PARTIAL_FILL_PCT,       # 0.40
    # Entry zone
    "MID_ZONE_LOW":                  cfg.MID_ZONE_LOW,               # 0.40
    "MID_ZONE_HIGH":                 cfg.MID_ZONE_HIGH,              # 0.60
    "PRE_ALERT_DISTANCE_PCT":        cfg.PRE_ALERT_DISTANCE_PCT,     # 0.20
    "ENTRY_CONFIRMATION_MODE":       cfg.ENTRY_CONFIRMATION_MODE,    # "soft"
    "ENTRY_CONFIRM_MIN_BODY_PCT":    cfg.ENTRY_CONFIRM_MIN_BODY_PCT, # 0.05
    "SIGNAL_MIN_COMPOSITE_SCORE":    cfg.SIGNAL_MIN_COMPOSITE_SCORE, # 70.0
    "SIGNAL_MAX_COMPOSITE_SCORE":    getattr(cfg, "SIGNAL_MAX_COMPOSITE_SCORE", 100.0),  # 80.0
    # Regime adaptive
    "REGIME_ADAPTIVE_THRESHOLD":     cfg.REGIME_ADAPTIVE_THRESHOLD,  # True
    "REGIME_TREND_BONUS":            cfg.REGIME_TREND_BONUS,         # -1.5
    "REGIME_CHOP_PENALTY":           cfg.REGIME_CHOP_PENALTY,        # 4.0
    # Adaptive symbol profile
    "ADAPTIVE_SYMBOL_PROFILE":       cfg.ADAPTIVE_SYMBOL_PROFILE,
    "ADAPTIVE_ATR_LOW":              cfg.ADAPTIVE_ATR_LOW,
    "ADAPTIVE_ATR_HIGH":             cfg.ADAPTIVE_ATR_HIGH,
    "ADAPTIVE_HIGH_VOL_GAP_MULT":    cfg.ADAPTIVE_HIGH_VOL_GAP_MULT,
    "ADAPTIVE_HIGH_VOL_QUALITY_BONUS":   cfg.ADAPTIVE_HIGH_VOL_QUALITY_BONUS,
    "ADAPTIVE_HIGH_VOL_COMPOSITE_BONUS": cfg.ADAPTIVE_HIGH_VOL_COMPOSITE_BONUS,
    "ADAPTIVE_HIGH_VOL_STOP_MULT":   cfg.ADAPTIVE_HIGH_VOL_STOP_MULT,
    "ADAPTIVE_HIGH_VOL_TP_RR_CAP":   cfg.ADAPTIVE_HIGH_VOL_TP_RR_CAP,
    "ADAPTIVE_LOW_VOL_QUALITY_BONUS":    cfg.ADAPTIVE_LOW_VOL_QUALITY_BONUS,
    "ADAPTIVE_LOW_VOL_COMPOSITE_BONUS":  cfg.ADAPTIVE_LOW_VOL_COMPOSITE_BONUS,
    "ADAPTIVE_LOW_VOL_STOP_MULT":    cfg.ADAPTIVE_LOW_VOL_STOP_MULT,
    "ADAPTIVE_LOW_VOL_TP_RR_CAP":    cfg.ADAPTIVE_LOW_VOL_TP_RR_CAP,
    # Volume profile / structure
    "SWING_LOOKBACK_BARS":           cfg.SWING_LOOKBACK_BARS,
    "VP_LOOKBACK_BARS":              cfg.VP_LOOKBACK_BARS,
    "VP_NUM_BINS":                   cfg.VP_NUM_BINS,
    "STRUCTURE_TOUCH_TOL_PCT":       cfg.STRUCTURE_TOUCH_TOL_PCT,
    # SL/TP
    "SL_NEAREST_LEVEL_BUFFER_PCT":   cfg.SL_NEAREST_LEVEL_BUFFER_PCT,
    "TP_HVN_OFFSET_PCT":             cfg.TP_HVN_OFFSET_PCT,
    "SL_LEVEL_MIN_OFFSET_PCT":       cfg.SL_LEVEL_MIN_OFFSET_PCT,
    "SL_LEVEL_BUFFER_MIN_PCT":       cfg.SL_LEVEL_BUFFER_MIN_PCT,
    "SL_LEVEL_BUFFER_MAX_PCT":       cfg.SL_LEVEL_BUFFER_MAX_PCT,
    "TP_OFFSET_MIN_PCT":             cfg.TP_OFFSET_MIN_PCT,
    "TP_OFFSET_MAX_PCT":             cfg.TP_OFFSET_MAX_PCT,
    "MIN_TP_RR":                     cfg.MIN_TP_RR,
    # Portfolio
    "PORTFOLIO_CORRELATION_FILTER":  cfg.PORTFOLIO_MAX_CORR > 0,
    "PORTFOLIO_MAX_CORR":            cfg.PORTFOLIO_MAX_CORR,
    "CORRELATION_LOOKBACK_BARS":     cfg.CORRELATION_LOOKBACK_BARS,
    # Max RR (used by build_symbol_profile via BACKTEST_CONFIG reference)
    "MAX_RR":                        2.5,
}

# Alias so functions below can use the same name as in fvg_bot.py
CONFIG = FVG_CORE_CONFIG
# Minimal BACKTEST_CONFIG shim referenced by build_symbol_profile
BACKTEST_CONFIG: dict[str, Any] = {"MAX_RR": 2.5}


# ═══════════════════════════════════════════════════════════════════════════════
#  DATACLASSES
# ═══════════════════════════════════════════════════════════════════════════════

class FVG:
    """Represents a single Fair Value Gap. Exact copy from fvg_bot.py."""

    __slots__ = (
        "type", "high", "low", "timestamp", "candle_index",
        "mitigated", "gap_pct", "size_fvg", "quality_score",
        "status", "fill_ratio", "regime_adjustment", "reaction_score",
    )

    def __init__(
        self,
        fvg_type: str,
        high: float,
        low: float,
        timestamp: int,
        candle_index: int,
        gap_pct: float = 0.0,
        size_fvg: float = 0.0,
        quality_score: float = 0.0,
        regime_adjustment: float = 0.0,
    ) -> None:
        self.type            = fvg_type
        self.high            = high
        self.low             = low
        self.timestamp       = timestamp
        self.candle_index    = candle_index
        self.mitigated       = False
        self.gap_pct         = gap_pct
        self.size_fvg        = size_fvg
        self.quality_score   = quality_score
        self.status          = "active"
        self.fill_ratio      = 0.0
        self.regime_adjustment = regime_adjustment
        self.reaction_score  = 0.0

    def __repr__(self) -> str:
        return (
            f"FVG({self.type}, {self.low:.5f}-{self.high:.5f}, "
            f"gap={self.gap_pct:.3f}%, score={self.size_fvg:.1f}%, "
            f"quality={self.quality_score:.1f}, fill={self.fill_ratio:.2f}, "
            f"status={self.status})"
        )


@dataclass
class SignalCandidate:
    """
    Ready-to-execute signal for one symbol.
    Exact copy from fvg_bot.py.
    """
    symbol:         str
    fvg:            FVG
    entry_price:    float
    df_5m:          Any        # pd.DataFrame
    position_pct:   float
    score:          float
    raw_score:      float
    quality_score:  float
    reaction_score: float


# ═══════════════════════════════════════════════════════════════════════════════
#  LOW-LEVEL HELPERS  (exact copy from fvg_bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_all_gap_diffs(
    highs: np.ndarray,
    lows: np.ndarray,
    start: int,
    end: int,
) -> np.ndarray:
    """
    Collect diff = (gap_size / mid_price) * 100 for ALL gaps in [start, end).
    Mirrors Pine Script ta.percentile_nearest_rank(diff, 1000, 100).
    """
    diffs = []
    for i in range(start, end):
        mid_price = (highs[i] + lows[i]) / 2.0
        if mid_price <= 0:
            continue
        if highs[i - 2] < lows[i]:
            diffs.append((lows[i] - highs[i - 2]) / mid_price * 100.0)
        if lows[i - 2] > highs[i]:
            diffs.append((lows[i - 2] - highs[i]) / mid_price * 100.0)
    return np.array(diffs, dtype=np.float64) if diffs else np.array([], dtype=np.float64)


def _atr_pct_from_arrays(
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
) -> float:
    """Compute ATR as a percentage of the last close."""
    if len(closes) < 3:
        return 0.0
    prev_close = np.roll(closes, 1)
    prev_close[0] = closes[0]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)),
    )
    atr = float(np.mean(tr))
    last_close = float(closes[-1]) if float(closes[-1]) > 0 else 1.0
    return atr / last_close * 100.0


def _market_regime_adjustment(
    highs_raw: np.ndarray,
    lows_raw: np.ndarray,
    closes: np.ndarray,
    index: int,
) -> tuple[float, dict[str, float]]:
    """
    Adaptive threshold tweak:
    - Trending market: allow slightly easier score threshold (bonus)
    - Choppy market: demand stronger score (penalty)
    """
    start = max(0, index - 96)
    close_win = closes[start:index + 1].astype(float)
    high_win = highs_raw[start:index + 1].astype(float)
    low_win = lows_raw[start:index + 1].astype(float)
    if len(close_win) < 20:
        return 0.0, {"atr_pct": 0.0, "trend_pct": 0.0, "chop_pct": 0.0}

    atr_pct = _atr_pct_from_arrays(high_win, low_win, close_win)
    s = pd.Series(close_win)
    ema_fast = s.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_slow = s.ewm(span=50, adjust=False).mean().iloc[-1]
    last_close = float(close_win[-1]) if close_win[-1] > 0 else 1.0
    trend_pct = abs(float(ema_fast) - float(ema_slow)) / last_close * 100.0

    diff = np.diff(close_win)
    signs = np.sign(diff)
    direction_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
    chop_pct = direction_changes / max(1, len(signs)) * 100.0

    adjustment = 0.0
    if bool(CONFIG.get("REGIME_ADAPTIVE_THRESHOLD", True)):
        if trend_pct >= max(0.10, atr_pct * 0.45):
            adjustment += float(CONFIG.get("REGIME_TREND_BONUS", -1.5))
        if chop_pct >= 45.0:
            adjustment += float(CONFIG.get("REGIME_CHOP_PENALTY", 4.0))
        if atr_pct >= 1.8:
            adjustment += 1.0

    return adjustment, {
        "atr_pct": float(atr_pct),
        "trend_pct": float(trend_pct),
        "chop_pct": float(chop_pct),
    }


def _fvg_quality_score(
    diff_pct: float,
    size_fvg: float,
    body_pct_prev: float,
    required_body: float,
    regime_info: dict[str, float],
    direction: str,
    opens: np.ndarray,
    closes: np.ndarray,
    index: int,
) -> float:
    """Composite quality score for a raw FVG candidate (0..100)."""
    displacement_ratio = body_pct_prev / max(required_body, 1e-6)
    displacement_score = min(displacement_ratio * 35.0, 100.0)

    atr_pct = max(regime_info.get("atr_pct", 0.0), 1e-6)
    volatility_score = min((diff_pct / atr_pct) * 40.0, 100.0)

    trend_pct = regime_info.get("trend_pct", 0.0)
    trend_score = min((trend_pct / max(atr_pct, 0.12)) * 45.0, 100.0)

    direction_score = 50.0
    if index >= 2:
        c1_up = closes[index - 1] > opens[index - 1]
        c2_up = closes[index - 2] > opens[index - 2]
        if direction == "bullish":
            direction_score = 100.0 if (c1_up or c2_up) else 25.0
        else:
            direction_score = 100.0 if ((not c1_up) or (not c2_up)) else 25.0

    quality = (
        size_fvg * 0.40
        + displacement_score * 0.20
        + volatility_score * 0.20
        + trend_score * 0.10
        + direction_score * 0.10
    )
    return float(max(0.0, min(100.0, quality)))


def update_fvg_lifecycle(
    fvg: FVG,
    candle_high: float,
    candle_low: float,
    candle_close: float,
) -> None:
    """Update lifecycle state for a zone: active -> partial -> mitigated/invalidated."""
    gap_size = float(fvg.high - fvg.low)
    if gap_size <= 0:
        return

    if fvg.type == "bullish":
        penetration = max(0.0, fvg.high - max(candle_low, fvg.low))
    else:
        penetration = max(0.0, min(candle_high, fvg.high) - fvg.low)

    fill_ratio = max(0.0, min(1.0, penetration / gap_size))
    if fill_ratio > fvg.fill_ratio:
        fvg.fill_ratio = fill_ratio

    partial_threshold = float(CONFIG.get("FVG_PARTIAL_FILL_PCT", 0.40))
    if fvg.status == "active" and fvg.fill_ratio >= partial_threshold:
        fvg.status = "partial"

    mode = str(CONFIG.get("FVG_MITIGATION_MODE", "close_mid")).lower()
    midpoint = (fvg.low + fvg.high) / 2.0

    mitigated = False
    if mode == "close_inside":
        mitigated = fvg.low <= candle_close <= fvg.high
    elif mode == "wick_mid":
        mitigated = fvg.fill_ratio >= 0.5
    elif mode == "close_through":
        if fvg.type == "bullish":
            mitigated = candle_close <= fvg.low
        else:
            mitigated = candle_close >= fvg.high
    else:  # close_mid (default)
        if fvg.type == "bullish":
            mitigated = candle_close <= midpoint
        else:
            mitigated = candle_close >= midpoint

    invalidated = False
    if fvg.type == "bullish" and candle_close < fvg.low * 0.999:
        invalidated = True
    if fvg.type == "bearish" and candle_close > fvg.high * 1.001:
        invalidated = True

    if mitigated or invalidated:
        fvg.mitigated = True
        fvg.status = "invalidated" if invalidated else "mitigated"


def evaluate_entry_reaction(df: pd.DataFrame, fvg: FVG) -> tuple[bool, float]:
    """
    Soft/strict reaction check inside 40-60% zone.
    Exact copy from fvg_bot.py.
    """
    mode = str(CONFIG.get("ENTRY_CONFIRMATION_MODE", "soft")).lower()
    if mode == "off":
        return True, 100.0
    if df is None or df.empty or len(df) < 3:
        return True, 50.0

    candles = df.tail(3).reset_index(drop=True)
    c0 = candles.iloc[-1]
    c1 = candles.iloc[-2]

    close0 = float(c0["close"])
    open0  = float(c0["open"])
    high0  = float(c0["high"])
    low0   = float(c0["low"])
    close1 = float(c1["close"])
    high1  = float(c1["high"])
    low1   = float(c1["low"])
    body_pct = abs(close0 - open0) / max(close0, 1e-6) * 100.0

    gap_size = max(1e-9, float(fvg.high - fvg.low))
    mid_low  = fvg.low + gap_size * float(CONFIG["MID_ZONE_LOW"])
    mid_high = fvg.low + gap_size * float(CONFIG["MID_ZONE_HIGH"])

    if fvg.type == "bullish":
        cond_reject = low0 <= mid_high and close0 > open0 and close0 >= mid_low
        cond_shift  = close0 > max(close1, high1)
        cond_body   = body_pct >= float(CONFIG.get("ENTRY_CONFIRM_MIN_BODY_PCT", 0.05))
    else:
        cond_reject = high0 >= mid_low and close0 < open0 and close0 <= mid_high
        cond_shift  = close0 < min(close1, low1)
        cond_body   = body_pct >= float(CONFIG.get("ENTRY_CONFIRM_MIN_BODY_PCT", 0.05))

    score = float(100.0 * (int(cond_reject) + int(cond_shift) + int(cond_body)) / 3.0)
    passed = int(cond_reject) + int(cond_shift) + int(cond_body)
    if mode == "strict":
        return passed >= 2, score
    return passed >= 1, score


def composite_signal_score(fvg: FVG, reaction_score: float) -> float:
    """Blend percentile score, quality score and reaction score for ranking."""
    return float(
        fvg.size_fvg * 0.50
        + fvg.quality_score * 0.35
        + reaction_score * 0.15
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN FVG DETECTOR  (detect_fvgs_v2) — exact copy from fvg_bot.py
# ═══════════════════════════════════════════════════════════════════════════════

def detect_fvgs_v2(
    df: pd.DataFrame,
    lookback: int = 100,
    percentile_threshold: float = 18.0,
    percentile_lookback: int = 1000,
) -> list[FVG]:
    """Two-stage FVG detection: raw structure -> quality filter -> lifecycle update."""
    fvgs: list[FVG] = []
    if len(df) < 3:
        return fvgs

    opens     = df["open"].values
    highs_raw = df["high"].values
    lows_raw  = df["low"].values
    closes    = df["close"].values
    price_mode = str(CONFIG.get("FVG_PRICE_MODE", "wick")).lower()
    if price_mode == "body":
        highs = np.maximum(opens, closes)
        lows  = np.minimum(opens, closes)
    else:
        highs = highs_raw
        lows  = lows_raw
    times = df["timestamp"].values if "timestamp" in df.columns else df.index.values
    body_sizes = np.abs(closes - opens)
    body_mid   = (highs_raw + lows_raw) / 2.0
    body_mid   = np.where(body_mid <= 0, 1.0, body_mid)
    body_pct   = body_sizes / body_mid * 100.0

    dist_start      = max(2, len(df) - percentile_lookback)
    all_diffs       = _collect_all_gap_diffs(highs, lows, dist_start, len(df))
    percentile_rank = float(np.percentile(all_diffs, 100)) if len(all_diffs) > 0 else 0.0

    scan_start   = max(2, len(df) - lookback)
    min_gap_pct  = CONFIG["MIN_GAP_PERCENT"] * float(CONFIG.get("FVG_MIN_GAP_MULT", 1.0))
    require_disp = bool(CONFIG.get("FVG_REQUIRE_DISPLACEMENT", True))
    disp_mult    = float(CONFIG.get("FVG_DISPLACEMENT_BODY_MULT", 1.2))
    disp_min     = float(CONFIG.get("FVG_DISPLACEMENT_BODY_PCT", 0.15))
    same_dir     = bool(CONFIG.get("FVG_REQUIRE_SAME_DIRECTION", True))
    quality_floor = float(CONFIG.get("FVG_QUALITY_MIN_SCORE", 42.0))

    for i in range(scan_start, len(df)):
        mid_price = (highs[i] + lows[i]) / 2.0
        if mid_price <= 0:
            continue

        window_start  = max(0, i - 60)
        baseline      = float(np.median(body_pct[window_start:i])) if i > window_start else 0.0
        required_body = max(disp_min, baseline * disp_mult)
        if require_disp and body_pct[i - 1] < required_body:
            continue

        regime_adjustment, regime_info = _market_regime_adjustment(highs_raw, lows_raw, closes, i)
        dynamic_threshold = percentile_threshold + regime_adjustment

        # ── Bullish FVG ──────────────────────────────────────────────────────
        if highs[i - 2] < lows[i] and not (same_dir and closes[i - 1] <= opens[i - 1]):
            gap_size = lows[i] - highs[i - 2]
            diff = (gap_size / mid_price) * 100.0
            if diff >= min_gap_pct:
                size_fvg = (diff / percentile_rank * 100.0) if percentile_rank > 0 else 0.0
                quality = _fvg_quality_score(
                    diff_pct=diff,
                    size_fvg=size_fvg,
                    body_pct_prev=float(body_pct[i - 1]),
                    required_body=required_body,
                    regime_info=regime_info,
                    direction="bullish",
                    opens=opens,
                    closes=closes,
                    index=i,
                )
                if size_fvg >= dynamic_threshold and quality >= quality_floor:
                    fvgs.append(FVG(
                        fvg_type="bullish",
                        high=lows[i],
                        low=highs[i - 2],
                        timestamp=int(times[i - 1]),
                        candle_index=i - 1,
                        gap_pct=diff,
                        size_fvg=size_fvg,
                        quality_score=quality,
                        regime_adjustment=regime_adjustment,
                    ))

        # ── Bearish FVG ──────────────────────────────────────────────────────
        if lows[i - 2] > highs[i]:
            if same_dir and closes[i - 1] >= opens[i - 1]:
                continue
            gap_size = lows[i - 2] - highs[i]
            diff = (gap_size / mid_price) * 100.0
            if diff < min_gap_pct:
                continue
            size_fvg = (diff / percentile_rank * 100.0) if percentile_rank > 0 else 0.0
            quality = _fvg_quality_score(
                diff_pct=diff,
                size_fvg=size_fvg,
                body_pct_prev=float(body_pct[i - 1]),
                required_body=required_body,
                regime_info=regime_info,
                direction="bearish",
                opens=opens,
                closes=closes,
                index=i,
            )
            if size_fvg >= dynamic_threshold and quality >= quality_floor:
                fvgs.append(FVG(
                    fvg_type="bearish",
                    high=lows[i - 2],
                    low=highs[i],
                    timestamp=int(times[i - 1]),
                    candle_index=i - 1,
                    gap_pct=diff,
                    size_fvg=size_fvg,
                    quality_score=quality,
                    regime_adjustment=regime_adjustment,
                ))

    # ── Lifecycle update ─────────────────────────────────────────────────────
    for fvg in fvgs:
        for j in range(fvg.candle_index + 2, len(df)):
            update_fvg_lifecycle(
                fvg=fvg,
                candle_high=float(highs_raw[j]),
                candle_low=float(lows_raw[j]),
                candle_close=float(closes[j]),
            )
            if fvg.mitigated:
                break

    max_age_days = float(CONFIG.get("FVG_MAX_AGE_DAYS", 0))
    if max_age_days > 0:
        max_age_ms = max_age_days * 86400000.0
        latest_ts  = int(times[-1]) if len(times) else 0
        fvgs = [fvg for fvg in fvgs if (latest_ts - fvg.timestamp) <= max_age_ms]

    return fvgs


# ═══════════════════════════════════════════════════════════════════════════════
#  MID-ZONE HELPERS  (exact copy from fvg_bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def price_in_mid_zone(price: float, fvg: FVG) -> tuple[bool, float]:
    """
    Check whether *price* is in the 40-60% entry band of *fvg*.
    Returns (in_zone, position_pct).
    """
    gap_size = fvg.high - fvg.low
    if gap_size <= 0:
        return False, 0.0
    position_pct = (price - fvg.low) / gap_size * 100.0
    mid_low  = CONFIG["MID_ZONE_LOW"]  * 100.0
    mid_high = CONFIG["MID_ZONE_HIGH"] * 100.0
    return (mid_low <= position_pct <= mid_high), position_pct


def distance_to_mid_zone_pct(price: float, fvg: FVG) -> float:
    """
    Absolute % distance from *price* to the nearest edge of the 40-60% entry
    band.  Returns 0.0 if already inside.
    """
    gap_size = fvg.high - fvg.low
    if gap_size <= 0 or price <= 0:
        return float("inf")
    entry_band_low  = fvg.low + gap_size * CONFIG["MID_ZONE_LOW"]
    entry_band_high = fvg.low + gap_size * CONFIG["MID_ZONE_HIGH"]
    if entry_band_low <= price <= entry_band_high:
        return 0.0
    target = entry_band_low if price < entry_band_low else entry_band_high
    return abs(target - price) / price * 100.0


# ═══════════════════════════════════════════════════════════════════════════════
#  EMA  (exact copy from fvg_bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


# ═══════════════════════════════════════════════════════════════════════════════
#  SYMBOL PROFILE  (exact copy from fvg_bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def build_symbol_profile(df: pd.DataFrame) -> dict[str, Any]:
    """
    Adaptive per-symbol profile based on 15m ATR/realized volatility.
    Exact copy from fvg_bot.py.
    """
    tail = df.tail(max(96, int(CONFIG.get("SWING_LOOKBACK_BARS", 120)))).copy()
    if tail.empty:
        return {
            "name": "normal",
            "atr_pct": 0.0,
            "realized_vol_pct": 0.0,
            "gap_mult": 1.0,
            "quality_bonus": 0.0,
            "composite_bonus": 0.0,
            "stop_mult": 1.0,
            "tp_rr_cap": float(BACKTEST_CONFIG.get("MAX_RR", 2.5)),
            "vol_metric": 0.0,
        }

    highs  = tail["high"].astype(float).to_numpy()
    lows   = tail["low"].astype(float).to_numpy()
    closes = tail["close"].astype(float).to_numpy()
    atr_pct = _atr_pct_from_arrays(highs, lows, closes)

    if len(closes) >= 10:
        log_returns    = np.diff(np.log(np.clip(closes, 1e-12, None)))
        realized_vol_pct = float(np.std(log_returns) * np.sqrt(96) * 100.0)
    else:
        realized_vol_pct = 0.0

    profile: dict[str, Any] = {
        "name": "normal",
        "atr_pct": float(atr_pct),
        "realized_vol_pct": float(realized_vol_pct),
        "gap_mult": 1.0,
        "quality_bonus": 0.0,
        "composite_bonus": 0.0,
        "stop_mult": 1.0,
        "tp_rr_cap": float(BACKTEST_CONFIG.get("MAX_RR", 2.5)),
        "vol_metric": 0.0,
    }

    if not bool(CONFIG.get("ADAPTIVE_SYMBOL_PROFILE", True)):
        return profile

    low_thr  = float(CONFIG.get("ADAPTIVE_ATR_LOW",  0.32))
    high_thr = float(CONFIG.get("ADAPTIVE_ATR_HIGH", 0.78))

    realized_component = float(realized_vol_pct) * 0.10
    vol_metric = max(float(atr_pct), realized_component)
    profile["vol_metric"] = float(vol_metric)

    if vol_metric >= high_thr:
        profile["name"]           = "high_vol"
        profile["gap_mult"]       = float(CONFIG.get("ADAPTIVE_HIGH_VOL_GAP_MULT", 1.15))
        profile["quality_bonus"]  = float(CONFIG.get("ADAPTIVE_HIGH_VOL_QUALITY_BONUS", 4.0))
        profile["composite_bonus"] = float(CONFIG.get("ADAPTIVE_HIGH_VOL_COMPOSITE_BONUS", 4.0))
        profile["stop_mult"]      = float(CONFIG.get("ADAPTIVE_HIGH_VOL_STOP_MULT", 1.18))
        profile["tp_rr_cap"]      = float(CONFIG.get("ADAPTIVE_HIGH_VOL_TP_RR_CAP", 2.20))
    elif vol_metric <= low_thr:
        profile["name"]           = "low_vol"
        profile["quality_bonus"]  = float(CONFIG.get("ADAPTIVE_LOW_VOL_QUALITY_BONUS", -1.0))
        profile["composite_bonus"] = float(CONFIG.get("ADAPTIVE_LOW_VOL_COMPOSITE_BONUS", -1.0))
        profile["stop_mult"]      = float(CONFIG.get("ADAPTIVE_LOW_VOL_STOP_MULT", 0.94))
        profile["tp_rr_cap"]      = float(CONFIG.get("ADAPTIVE_LOW_VOL_TP_RR_CAP", 2.60))

    return profile


# ═══════════════════════════════════════════════════════════════════════════════
#  SWING / STRUCTURE LEVELS  (exact copy from fvg_bot.py)
# ═══════════════════════════════════════════════════════════════════════════════

def collect_swing_levels(
    df: pd.DataFrame,
    lookback: int = 60,
) -> tuple[list[float], list[float]]:
    """Collect confirmed swing lows and swing highs from the recent 15m structure."""
    if df.empty or len(df) < 5:
        return [], []
    tail  = df.tail(lookback).reset_index(drop=True)
    highs = tail["high"].astype(float).tolist()
    lows  = tail["low"].astype(float).tolist()
    swing_lows:  list[float] = []
    swing_highs: list[float] = []

    for i in range(2, len(tail) - 2):
        if (
            highs[i] > highs[i - 1]
            and highs[i] > highs[i - 2]
            and highs[i] > highs[i + 1]
            and highs[i] > highs[i + 2]
        ):
            swing_highs.append(float(highs[i]))
        if (
            lows[i] < lows[i - 1]
            and lows[i] < lows[i - 2]
            and lows[i] < lows[i + 1]
            and lows[i] < lows[i + 2]
        ):
            swing_lows.append(float(lows[i]))

    return sorted(set(swing_lows)), sorted(set(swing_highs))


def find_swing_levels(df: pd.DataFrame, lookback: int = 30) -> tuple[float, float]:
    """
    Find nearest swing high and swing low on 15m candles.
    Returns (nearest_swing_low, nearest_swing_high).
    """
    if df.empty or len(df) < 5:
        return 0.0, 0.0
    tail = df.tail(lookback)
    highs_arr = tail["high"].values
    lows_arr  = tail["low"].values
    swing_lows, swing_highs = collect_swing_levels(tail, lookback=lookback)
    current_price = float(tail["close"].iloc[-1])

    above      = [h for h in swing_highs if h > current_price]
    nearest_sh = min(above) if above else float(np.max(highs_arr))

    below      = [l for l in swing_lows if l < current_price]
    nearest_sl = max(below) if below else float(np.min(lows_arr))

    return nearest_sl, nearest_sh


def build_local_volume_profile(
    df: pd.DataFrame,
    lookback: int = 96,
    bins: int = 48,
) -> dict[str, Any]:
    """
    Build an approximate fixed-range volume profile from OHLCV candles.
    Exact copy from fvg_bot.py.
    """
    tail = df.tail(max(10, lookback)).copy()
    if tail.empty:
        return {"levels": [], "poc": 0.0}

    price_low  = float(tail["low"].min())
    price_high = float(tail["high"].max())
    if price_high <= price_low:
        level = float(tail["close"].iloc[-1])
        return {"levels": [level], "poc": level}

    bins = max(12, int(bins))
    edges     = np.linspace(price_low, price_high, bins + 1)
    centers   = (edges[:-1] + edges[1:]) / 2.0
    histogram = np.zeros(bins, dtype=float)

    for _, candle in tail.iterrows():
        low    = float(candle["low"])
        high   = float(candle["high"])
        volume = float(candle.get("volume", 0.0) or 0.0)
        if volume <= 0:
            continue
        if high <= low:
            idx = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, bins - 1))
            histogram[idx] += volume
            continue
        overlap = np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
        overlap = np.clip(overlap, 0.0, None)
        overlap_sum = float(overlap.sum())
        if overlap_sum <= 0:
            idx = int(np.clip(
                np.searchsorted(edges, (low + high) / 2.0, side="right") - 1, 0, bins - 1
            ))
            histogram[idx] += volume
            continue
        histogram += volume * (overlap / overlap_sum)

    if not np.any(histogram > 0):
        level = float(tail["close"].iloc[-1])
        return {"levels": [level], "poc": level}

    active_values = histogram[histogram > 0]
    hvn_threshold = float(np.percentile(active_values, 70))
    hvn_indices: list[int] = []
    for idx in range(len(histogram)):
        left  = histogram[idx - 1] if idx > 0 else histogram[idx]
        right = histogram[idx + 1] if idx < len(histogram) - 1 else histogram[idx]
        if histogram[idx] >= hvn_threshold and histogram[idx] >= left and histogram[idx] >= right:
            hvn_indices.append(idx)

    poc_index = int(np.argmax(histogram))
    if poc_index not in hvn_indices:
        hvn_indices.append(poc_index)

    levels = sorted({float(centers[idx]) for idx in hvn_indices})
    return {
        "levels": levels,
        "poc": float(centers[poc_index]),
        "histogram": histogram.tolist(),
        "price_low": price_low,
        "price_high": price_high,
    }


def extract_structural_levels(
    df: pd.DataFrame,
    lookback: int = 160,
    tolerance_pct: float = 0.06,
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    """
    Cluster support/resistance levels and count touches.
    Returns:
      supports     [(level, touches)] sorted high->low
      resistances  [(level, touches)] sorted low->high
    Exact copy from fvg_bot.py.
    """
    if df is None or df.empty or len(df) < 15:
        return [], []

    tail  = df.tail(max(30, lookback)).reset_index(drop=True)
    highs = tail["high"].astype(float).to_numpy()
    lows  = tail["low"].astype(float).to_numpy()
    swing_lows, swing_highs = collect_swing_levels(tail, lookback=len(tail))
    raw_supports    = list(swing_lows)
    raw_resistances = list(swing_highs)

    for win in (3, 5):
        for i in range(win, len(tail) - win):
            low_i  = float(lows[i])
            high_i = float(highs[i])
            if low_i <= float(np.min(lows[i - win:i + win + 1])):
                raw_supports.append(low_i)
            if high_i >= float(np.max(highs[i - win:i + win + 1])):
                raw_resistances.append(high_i)

    tol = max(0.0002, float(tolerance_pct) / 100.0)

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
            level      = float(np.median(cluster))
            touch_band = max(level * tol, level * 0.0002)
            touches    = int(np.sum(np.abs(probe - level) <= touch_band))
            result.append((level, max(1, touches)))
        return result

    supports    = _cluster(raw_supports, lows)
    resistances = _cluster(raw_resistances, highs)
    supports    = sorted(supports, key=lambda item: item[0], reverse=True)
    resistances = sorted(resistances, key=lambda item: item[0])
    return supports, resistances


# ═══════════════════════════════════════════════════════════════════════════════
#  STANDALONE _select_active_fvgs
# ═══════════════════════════════════════════════════════════════════════════════

def _select_active_fvgs(fvgs: list[FVG]) -> list[FVG]:
    """
    Pick top active FVGs per side, prioritising stronger zones while keeping
    recency.  Standalone version of the bot's instance method.
    """
    unmitigated = [fvg for fvg in fvgs if not fvg.mitigated]
    if not unmitigated:
        return []
    unmitigated.sort(
        key=lambda item: (
            item.quality_score,
            item.size_fvg,
            item.reaction_score,
            item.timestamp,
        ),
        reverse=True,
    )
    max_per_side = max(1, int(CONFIG.get("FVG_MAX_ACTIVE_PER_SIDE", 3)))
    bullish = [fvg for fvg in unmitigated if fvg.type == "bullish"][:max_per_side]
    bearish = [fvg for fvg in unmitigated if fvg.type == "bearish"][:max_per_side]
    return bullish + bearish
