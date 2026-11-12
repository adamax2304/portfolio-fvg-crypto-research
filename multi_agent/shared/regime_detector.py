"""
multi_agent/shared/regime_detector.py  — Soft Regime Engine v2

Outputs per symbol/bar:
    regime_label    — TREND / RANGE / SQUEEZE / TRANSITION / UNDEFINED
    confidence      — 0.0 … 1.0
    agent_weights   — {agent: 0.0 … 1.0}   (priority)
    size_mult       — {agent: 0.0 … 1.0}   (position-size multiplier)
    trade_ok        — {agent: bool}         (hard block only in clearly bad conditions)

Hysteresis: 5-bar confirmation, 15-bar minimum hold, EMA-smoothed scores.
"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
#  Enums & data classes
# ═══════════════════════════════════════════════════════════════════════════════

class Regime(str, Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    SQUEEZE = "SQUEEZE"
    TRANSITION = "TRANSITION"
    UNDEFINED = "UNDEFINED"


@dataclass
class SoftRegimeState:
    """Full regime output for one symbol at one bar."""
    regime: Regime = Regime.UNDEFINED
    confidence: float = 0.0          # 0-1 how sure we are
    # Raw scores (before smoothing)
    score_trend: float = 0.0
    score_range: float = 0.0
    score_squeeze: float = 0.0
    # Smoothed scores
    smooth_trend: float = 0.0
    smooth_range: float = 0.0
    smooth_squeeze: float = 0.0
    # Indicator snapshots
    adx_1h: float = 0.0
    ema_slope_1h: float = 0.0
    bbw_percentile_1h: float = 50.0
    atr_percentile_1h: float = 50.0
    kc_squeeze_active: bool = False
    dir_efficiency: float = 0.0      # directional efficiency ratio
    # Soft outputs
    agent_weights: dict = field(default_factory=lambda: {
        "fvg": 0.0, "bbs": 0.0, "density": 0.0, "mr": 0.0})
    size_mult: dict = field(default_factory=lambda: {
        "fvg": 0.0, "bbs": 0.0, "density": 0.0, "mr": 0.0})
    trade_ok: dict = field(default_factory=lambda: {
        "fvg": False, "bbs": False, "density": False, "mr": False})
    reasons: list = field(default_factory=list)


# For backward-compatibility with old code that imports RegimeState
RegimeState = SoftRegimeState


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure indicator helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _ema(data: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    out[period - 1] = np.mean(data[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    cs = np.cumsum(data)
    out[period - 1:] = (cs[period - 1:] - np.concatenate([[0], cs[:-period]])) / period
    return out


def _stdev(data: np.ndarray, period: int) -> np.ndarray:
    out = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        out[i] = np.std(data[i - period + 1:i + 1], ddof=0)
    return out


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    out = np.full(n, np.nan)
    if n >= period:
        out[period - 1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            out[i] = out[i - 1] * (1 - alpha) + tr[i] * alpha
    return out


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period * 2:
        return out
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr_s = np.zeros(n)
    pdm_s = np.zeros(n)
    mdm_s = np.zeros(n)
    atr_s[period] = np.sum(tr[1:period + 1])
    pdm_s[period] = np.sum(plus_dm[1:period + 1])
    mdm_s[period] = np.sum(minus_dm[1:period + 1])
    for i in range(period + 1, n):
        atr_s[i] = atr_s[i - 1] - atr_s[i - 1] / period + tr[i]
        pdm_s[i] = pdm_s[i - 1] - pdm_s[i - 1] / period + plus_dm[i]
        mdm_s[i] = mdm_s[i - 1] - mdm_s[i - 1] / period + minus_dm[i]
    dx = np.zeros(n)
    for i in range(period, n):
        if atr_s[i] == 0:
            continue
        pdi = 100.0 * pdm_s[i] / atr_s[i]
        mdi = 100.0 * mdm_s[i] / atr_s[i]
        s = pdi + mdi
        if s > 0:
            dx[i] = 100.0 * abs(pdi - mdi) / s
    start = period * 2
    if start < n:
        out[start] = np.mean(dx[period + 1:start + 1])
        for i in range(start + 1, n):
            out[i] = (out[i - 1] * (period - 1) + dx[i]) / period
    return out


def _percentile_rank(values: np.ndarray, lookback: int, idx: int) -> float:
    start = max(0, idx - lookback + 1)
    window = values[start:idx + 1]
    window = window[~np.isnan(window)]
    if len(window) < 5:
        return 50.0
    val = values[idx]
    if np.isnan(val):
        return 50.0
    return float(np.sum(window < val) / len(window) * 100.0)


def _directional_efficiency(closes: np.ndarray, lookback: int, idx: int) -> float:
    """Net move / sum of abs bar-to-bar moves. 1.0 = perfect trend, 0 = choppy."""
    start = max(0, idx - lookback + 1)
    seg = closes[start:idx + 1]
    if len(seg) < 3:
        return 0.0
    net = abs(seg[-1] - seg[0])
    total = np.sum(np.abs(np.diff(seg)))
    if total == 0:
        return 0.0
    return float(net / total)


# ═══════════════════════════════════════════════════════════════════════════════
#  Regime scoring at a single bar
# ═══════════════════════════════════════════════════════════════════════════════

# Default soft-regime config (can be overridden by passing a dict)
DEFAULT_SOFT_CFG = dict(
    adx_trend_thresh=22.0,
    adx_range_thresh=17.0,
    ema_slope_trend=0.25,
    ema_slope_flat=0.08,
    bbw_squeeze_pctile=18.0,
    atr_low_pctile=25.0,
    dir_eff_lookback=24,
    dir_eff_trend=0.45,
    # Hysteresis
    confirm_bars=5,
    min_hold_bars=15,
    smooth_alpha=0.35,       # EMA smoothing for scores (0 = no smoothing, 1 = instant)
    # Classification
    min_confidence=1.0,      # lowered from 1.5 — easier to classify
    transition_margin=0.8,   # if best-second gap < this → TRANSITION (was UNDEFINED)
    undefined_threshold=0.5, # if best score < this → truly UNDEFINED
)


def classify_regime_at_bar(
    df_1h: pd.DataFrame,
    bar_idx: int,
    df_4h: pd.DataFrame = None,
    *,
    adx_1h: np.ndarray = None,
    ema200_1h: np.ndarray = None,
    bbw_1h: np.ndarray = None,
    atr_1h: np.ndarray = None,
    bb_upper_1h: np.ndarray = None,
    bb_lower_1h: np.ndarray = None,
    kc_upper_1h: np.ndarray = None,
    kc_lower_1h: np.ndarray = None,
    adx_4h: np.ndarray = None,
    ema200_4h: np.ndarray = None,
    closes_1h_arr: np.ndarray = None,
    highs_1h_arr: np.ndarray = None,
    lows_1h_arr: np.ndarray = None,
    soft_cfg: dict = None,
) -> SoftRegimeState:
    """Score regime at bar_idx of 1h data.  Returns raw (unsmoothed) SoftRegimeState."""
    C = {**DEFAULT_SOFT_CFG, **(soft_cfg or {})}
    state = SoftRegimeState()

    # Extract arrays
    if closes_1h_arr is None:
        closes_1h_arr = df_1h["close"].astype(float).to_numpy()
    if highs_1h_arr is None:
        highs_1h_arr = df_1h["high"].astype(float).to_numpy()
    if lows_1h_arr is None:
        lows_1h_arr = df_1h["low"].astype(float).to_numpy()
    n = len(closes_1h_arr)

    if bar_idx < 0:
        bar_idx = n + bar_idx
    if bar_idx < 210 or bar_idx >= n:
        state.reasons.append("insufficient_data")
        return state
    i = bar_idx

    # ── Compute indicators if not pre-supplied ──
    if adx_1h is None:
        adx_1h = _adx(highs_1h_arr, lows_1h_arr, closes_1h_arr, 14)
    if ema200_1h is None:
        ema200_1h = _ema(closes_1h_arr, 200)
    if atr_1h is None:
        atr_1h = _atr(highs_1h_arr, lows_1h_arr, closes_1h_arr, 14)
    if bbw_1h is None or bb_upper_1h is None:
        bb_basis = _sma(closes_1h_arr, 20)
        bb_std = _stdev(closes_1h_arr, 20)
        bb_upper_1h = bb_basis + 2.0 * bb_std
        bb_lower_1h = bb_basis - 2.0 * bb_std
        bbw_1h = (bb_upper_1h - bb_lower_1h) / np.where(bb_basis > 0, bb_basis, 1.0) * 100.0
    if kc_upper_1h is None:
        kc_basis = _ema(closes_1h_arr, 20)
        kc_atr = _atr(highs_1h_arr, lows_1h_arr, closes_1h_arr, 20)
        kc_upper_1h = kc_basis + 1.5 * kc_atr
        kc_lower_1h = kc_basis - 1.5 * kc_atr

    # ── Values at bar ──
    adx_val = adx_1h[i]
    ema200_val = ema200_1h[i]
    close_val = closes_1h_arr[i]
    bbw_val = bbw_1h[i]
    atr_val = atr_1h[i]

    if np.isnan(adx_val) or np.isnan(ema200_val):
        state.reasons.append("nan_indicators")
        return state

    state.adx_1h = float(adx_val)

    # EMA-200 slope (24-bar)
    slope_lb = 24
    if i >= slope_lb and not np.isnan(ema200_1h[i - slope_lb]):
        ep = ema200_1h[i - slope_lb]
        if ep > 0:
            state.ema_slope_1h = (ema200_val - ep) / ep * 100.0

    state.bbw_percentile_1h = _percentile_rank(bbw_1h, 100, i)
    state.atr_percentile_1h = _percentile_rank(atr_1h, 100, i)

    # KC squeeze
    if not np.isnan(bb_lower_1h[i]) and not np.isnan(kc_lower_1h[i]):
        state.kc_squeeze_active = (
            bb_lower_1h[i] > kc_lower_1h[i] and bb_upper_1h[i] < kc_upper_1h[i])

    # Directional efficiency
    state.dir_efficiency = _directional_efficiency(
        closes_1h_arr, C["dir_eff_lookback"], i)

    abs_slope = abs(state.ema_slope_1h)

    # ════ TREND scoring ════
    if adx_val > C["adx_trend_thresh"]:
        state.score_trend += 2.0 * min((adx_val - C["adx_trend_thresh"]) / 10.0, 1.5)
    elif adx_val > C["adx_range_thresh"]:
        frac = (adx_val - C["adx_range_thresh"]) / max(C["adx_trend_thresh"] - C["adx_range_thresh"], 1)
        state.score_trend += 0.6 * frac

    if abs_slope > C["ema_slope_trend"]:
        state.score_trend += 1.5 * min(abs_slope / 1.0, 1.5)

    # Price on correct side of EMA-200
    if (close_val > ema200_val and state.ema_slope_1h > 0) or \
       (close_val < ema200_val and state.ema_slope_1h < 0):
        state.score_trend += 0.5

    # Directional efficiency supports trend
    if state.dir_efficiency > C["dir_eff_trend"]:
        state.score_trend += 0.8

    # ATR not compressed
    if state.atr_percentile_1h > 40:
        state.score_trend += 0.3

    # ════ RANGE scoring ════
    if adx_val < C["adx_range_thresh"]:
        state.score_range += 2.0 * min((C["adx_range_thresh"] - adx_val) / 8.0, 1.5)

    if abs_slope < C["ema_slope_flat"]:
        state.score_range += 1.5
    elif abs_slope < C["ema_slope_trend"]:
        state.score_range += 0.6

    if 20 < state.bbw_percentile_1h < 70:
        state.score_range += 0.6

    # Low directional efficiency = choppy
    if state.dir_efficiency < 0.25:
        state.score_range += 0.5

    # ════ SQUEEZE scoring ════
    if state.kc_squeeze_active:
        state.score_squeeze += 3.0

    if state.bbw_percentile_1h < C["bbw_squeeze_pctile"]:
        state.score_squeeze += 1.8

    if state.atr_percentile_1h < C["atr_low_pctile"]:
        state.score_squeeze += 0.8

    # ════ 4h confirmation ════
    if df_4h is not None and len(df_4h) > 50:
        closes_4h = df_4h["close"].astype(float).to_numpy()
        highs_4h = df_4h["high"].astype(float).to_numpy()
        lows_4h = df_4h["low"].astype(float).to_numpy()
        ts_1h = df_1h["timestamp"].astype(int).to_numpy()
        ts_4h = df_4h["timestamp"].astype(int).to_numpy()
        j = min(int(np.searchsorted(ts_4h, ts_1h[i], side="right")) - 1, len(closes_4h) - 1)

        if j >= 50:
            if adx_4h is None:
                adx_4h = _adx(highs_4h, lows_4h, closes_4h, 14)
            if ema200_4h is None:
                ema200_4h = _ema(closes_4h, 200)
            adx_4h_val = adx_4h[j] if j < len(adx_4h) and not np.isnan(adx_4h[j]) else 0
            ema200_4h_val = ema200_4h[j] if j < len(ema200_4h) and not np.isnan(ema200_4h[j]) else 0

            # 4h supports trend → small boost
            if adx_4h_val > 20 and state.score_trend > state.score_range:
                state.score_trend += 0.5
            # 4h contradicts 1h trend → penalize mildly
            if state.score_trend > state.score_range and adx_4h_val > 0 and adx_4h_val < 13:
                state.score_trend -= 0.4

            if ema200_4h_val > 0 and j >= 6:
                ep4 = ema200_4h[j - 6] if not np.isnan(ema200_4h[j - 6]) else ema200_4h_val
                slope_4h = (ema200_4h_val - ep4) / max(ep4, 1e-9) * 100.0
                if state.ema_slope_1h > 0 and slope_4h < -0.15:
                    state.score_trend -= 0.3
                elif state.ema_slope_1h < 0 and slope_4h > 0.15:
                    state.score_trend -= 0.3

    # ════ Classification ════
    scores = {
        Regime.TREND: state.score_trend,
        Regime.RANGE: state.score_range,
        Regime.SQUEEZE: state.score_squeeze,
    }
    best_regime = max(scores, key=scores.get)
    best_score = scores[best_regime]
    sorted_scores = sorted(scores.values(), reverse=True)
    second_best = sorted_scores[1] if len(sorted_scores) > 1 else 0
    margin = best_score - second_best

    if best_score < C["undefined_threshold"]:
        state.regime = Regime.UNDEFINED
        state.confidence = 0.0
    elif margin < C["transition_margin"]:
        state.regime = Regime.TRANSITION
        state.confidence = min(best_score / 5.0, 0.6)  # moderate confidence
    else:
        state.regime = best_regime
        state.confidence = min(best_score / 5.0, 1.0)

    # ════ Agent weights & size multipliers ════
    _apply_soft_weights(state, scores)

    return state


def _apply_soft_weights(state: SoftRegimeState, scores: dict):
    """Set agent weights, size multipliers, and trade permissions based on regime + confidence.

    Evidence-based mapping (v3 forensic audit, 2026-04-12):
      FVG: DISABLED — WR 30.5%, -19R total, 50% stops too wide, 45.5% give back +1R.
           WORST in TREND (-22R), BEST in SQUEEZE (+3R). Pending SL/TP rework.
           When re-enabled: flip regime (SQUEEZE only, block TREND).
      BBS: CORE — best in TREND (+19.43R, 49.4% WR, 81 trades),
           worst in SQUEEZE (-9.15R, 27.3% WR). Old mapping was inverted.
      MR:  OVERLAY — best in TRANSITION (+5.17R, 42.9% WR),
           positive in RANGE and marginally in TREND. Old: RANGE-only.
      Density: EXCLUDED — no edge in any regime (MAE 7.55R > MFE 6.65R).
    """
    r = state.regime
    conf = state.confidence

    if r == Regime.TREND:
        # BBS: BEST regime (+19.43R, 49.4% WR on 81 trades) — full weight
        # MR:  marginal (+1.20R on 55 trades) — small overlay
        # FVG: DISABLED (-22R in TREND — worst regime)
        state.agent_weights = {"fvg": 0.0, "bbs": 1.0, "density": 0.0, "mr": 0.3}
        state.size_mult    = {"fvg": 0.0, "bbs": 1.0, "density": 0.0, "mr": 0.3}
        state.trade_ok     = {"fvg": False, "bbs": True, "density": False, "mr": True}

    elif r == Regime.RANGE:
        # MR:  secondary regime — good weight
        # BBS/FVG: no edge in range — blocked
        state.agent_weights = {"fvg": 0.0, "bbs": 0.0, "density": 0.0, "mr": 0.85}
        state.size_mult    = {"fvg": 0.0, "bbs": 0.0, "density": 0.0, "mr": 0.85}
        state.trade_ok     = {"fvg": False, "bbs": False, "density": False, "mr": True}

    elif r == Regime.SQUEEZE:
        # BBS: DISABLED in SQUEEZE — v4 backtest (2026-04-13) confirmed worst regime
        #   regime_no_squeeze: PF=1.58 +29.6% vs regime_base PF=1.44 +21.3% (removes 13 losing trades)
        #   Previously allowed at 0.5 weight, but cost -9.15R over 90d. Now fully blocked.
        # MR: also disabled in SQUEEZE (v4 Phase 2: MR hurts across all regimes)
        state.agent_weights = {"fvg": 0.0, "bbs": 0.0, "density": 0.0, "mr": 0.0}
        state.size_mult    = {"fvg": 0.0, "bbs": 0.0, "density": 0.0, "mr": 0.0}
        state.trade_ok     = {"fvg": False, "bbs": False, "density": False, "mr": False}

    elif r == Regime.TRANSITION:
        # BBS: second-best regime — good weight (0.85)
        # MR:  DISABLED — v4 backtest confirmed MR hurts across all regimes (standalone PF=0.68)
        #   Previously thought best regime for MR (+5.17R), but v4 full concurrent sim shows no edge
        # FVG: DISABLED (0R in TRANSITION — neutral, but overall broken)
        state.agent_weights = {"fvg": 0.0, "bbs": 0.85, "density": 0.0, "mr": 0.0}
        state.size_mult    = {"fvg": 0.0, "bbs": 0.85, "density": 0.0, "mr": 0.0}
        state.trade_ok     = {"fvg": False, "bbs": True, "density": False, "mr": False}

    else:  # UNDEFINED
        state.agent_weights = {"fvg": 0.0, "bbs": 0.1, "density": 0.0, "mr": 0.0}
        state.size_mult    = {"fvg": 0.0, "bbs": 0.15, "density": 0.0, "mr": 0.0}
        state.trade_ok     = {"fvg": False, "bbs": False, "density": False, "mr": False}

    # Modulate by confidence
    for a in state.size_mult:
        state.size_mult[a] = state.size_mult[a] * (0.5 + 0.5 * conf)


# ═══════════════════════════════════════════════════════════════════════════════
#  SoftRegimeDetector — stateful wrapper with hysteresis + score smoothing
# ═══════════════════════════════════════════════════════════════════════════════

class SoftRegimeDetector:
    """
    Stateful regime detector with EMA-smoothed scores and hysteresis.
    Much more stable than the old hard detector.
    """

    def __init__(self, soft_cfg: dict = None):
        self.cfg = {**DEFAULT_SOFT_CFG, **(soft_cfg or {})}
        self._state: dict[str, dict] = {}

    def get_regime(self, symbol: str) -> Regime:
        entry = self._state.get(symbol)
        return entry["regime"] if entry else Regime.UNDEFINED

    def update(
        self,
        symbol: str,
        df_1h: pd.DataFrame,
        bar_idx: int,
        df_4h: pd.DataFrame = None,
        **precomputed,
    ) -> SoftRegimeState:
        raw = classify_regime_at_bar(
            df_1h, bar_idx, df_4h, soft_cfg=self.cfg, **precomputed)

        if symbol not in self._state:
            self._state[symbol] = {
                "regime": raw.regime,
                "hold_count": 0,
                "candidate": raw.regime,
                "candidate_count": 1,
                "smooth_trend": raw.score_trend,
                "smooth_range": raw.score_range,
                "smooth_squeeze": raw.score_squeeze,
            }
            raw.smooth_trend = raw.score_trend
            raw.smooth_range = raw.score_range
            raw.smooth_squeeze = raw.score_squeeze
            return raw

        entry = self._state[symbol]
        entry["hold_count"] += 1

        # ── EMA-smooth scores ──
        a = self.cfg["smooth_alpha"]
        entry["smooth_trend"] = a * raw.score_trend + (1 - a) * entry["smooth_trend"]
        entry["smooth_range"] = a * raw.score_range + (1 - a) * entry["smooth_range"]
        entry["smooth_squeeze"] = a * raw.score_squeeze + (1 - a) * entry["smooth_squeeze"]

        raw.smooth_trend = entry["smooth_trend"]
        raw.smooth_range = entry["smooth_range"]
        raw.smooth_squeeze = entry["smooth_squeeze"]

        # ── Classify from smoothed scores ──
        sm_scores = {
            Regime.TREND: entry["smooth_trend"],
            Regime.RANGE: entry["smooth_range"],
            Regime.SQUEEZE: entry["smooth_squeeze"],
        }
        sm_best = max(sm_scores, key=sm_scores.get)
        sm_best_val = sm_scores[sm_best]
        sm_sorted = sorted(sm_scores.values(), reverse=True)
        sm_margin = sm_best_val - sm_sorted[1]

        if sm_best_val < self.cfg["undefined_threshold"]:
            new_regime = Regime.UNDEFINED
        elif sm_margin < self.cfg["transition_margin"]:
            new_regime = Regime.TRANSITION
        else:
            new_regime = sm_best

        # ── Hysteresis ──
        confirmed = entry["regime"]
        if new_regime == confirmed:
            entry["candidate"] = confirmed
            entry["candidate_count"] = 0
        elif new_regime == entry["candidate"]:
            entry["candidate_count"] += 1
        else:
            entry["candidate"] = new_regime
            entry["candidate_count"] = 1

        min_hold = self.cfg["min_hold_bars"]
        confirm_needed = self.cfg["confirm_bars"]

        if entry["hold_count"] >= min_hold and entry["candidate_count"] >= confirm_needed:
            old = confirmed
            entry["regime"] = new_regime
            entry["hold_count"] = 0
            entry["candidate_count"] = 0
            logger.debug("Regime switch %s: %s -> %s (sm: T=%.2f R=%.2f S=%.2f)",
                         symbol, old.value, new_regime.value,
                         entry["smooth_trend"], entry["smooth_range"], entry["smooth_squeeze"])

        # Apply confirmed regime to output
        raw.regime = entry["regime"]
        # Recompute confidence from smoothed scores for confirmed regime
        conf_score = sm_scores.get(entry["regime"], sm_best_val)
        raw.confidence = min(conf_score / 5.0, 1.0) if entry["regime"] != Regime.UNDEFINED else 0.0
        if entry["regime"] == Regime.TRANSITION:
            raw.confidence = min(sm_best_val / 5.0, 0.6)

        _apply_soft_weights(raw, sm_scores)
        return raw

    def reset(self, symbol: str = None):
        if symbol:
            self._state.pop(symbol, None)
        else:
            self._state.clear()


# Keep old name for backward compatibility
RegimeDetector = SoftRegimeDetector


# ═══════════════════════════════════════════════════════════════════════════════
#  Batch precompute for backtest efficiency
# ═══════════════════════════════════════════════════════════════════════════════

def precompute_1h_indicators(df_1h: pd.DataFrame) -> dict:
    closes = df_1h["close"].astype(float).to_numpy()
    highs = df_1h["high"].astype(float).to_numpy()
    lows = df_1h["low"].astype(float).to_numpy()

    adx_arr = _adx(highs, lows, closes, 14)
    ema200 = _ema(closes, 200)
    atr_arr = _atr(highs, lows, closes, 14)

    bb_basis = _sma(closes, 20)
    bb_std = _stdev(closes, 20)
    bb_upper = bb_basis + 2.0 * bb_std
    bb_lower = bb_basis - 2.0 * bb_std
    bbw = (bb_upper - bb_lower) / np.where(bb_basis > 0, bb_basis, 1.0) * 100.0

    kc_basis = _ema(closes, 20)
    kc_atr = _atr(highs, lows, closes, 20)
    kc_upper = kc_basis + 1.5 * kc_atr
    kc_lower = kc_basis - 1.5 * kc_atr

    return {
        "adx_1h": adx_arr, "ema200_1h": ema200, "atr_1h": atr_arr,
        "bbw_1h": bbw, "bb_upper_1h": bb_upper, "bb_lower_1h": bb_lower,
        "kc_upper_1h": kc_upper, "kc_lower_1h": kc_lower,
        "closes_1h_arr": closes, "highs_1h_arr": highs, "lows_1h_arr": lows,
    }


def precompute_4h_indicators(df_4h: pd.DataFrame) -> dict:
    closes = df_4h["close"].astype(float).to_numpy()
    highs = df_4h["high"].astype(float).to_numpy()
    lows = df_4h["low"].astype(float).to_numpy()
    return {"adx_4h": _adx(highs, lows, closes, 14), "ema200_4h": _ema(closes, 200)}
