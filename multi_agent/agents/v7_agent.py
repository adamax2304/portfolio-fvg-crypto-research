"""
multi_agent/agents/v7_agent.py
===============================
V7 Portfolio Swing Trading Agent — Bybit Live

4 strategies validated on 730d backtest (+1525% ret, 23.8% DD, 168 trades):
  ST   — SuperTrend(10,3) pivot 4h  : flip +1 → LONG, SL at pivot low − 0.3×ATR
  SMA  — SMA50/200 cross 1d         : golden cross LONG, death cross SHORT
  RSI  — RSI reversion 4h           : RSI<20 cross up (LONG), RSI>80 cross down (SHORT)
  FIB  — Fib pump retrace 4h        : consolidation → pump → 38-62% retrace

All entries: post-only limit at next-bar open (= signal bar close ≈ next bar open)
SL/TP: structural (pivot-based) or ATR-based depending on strategy
TP: fixed R multiple via Bybit conditional order

Filters (all validated in full_audit_v7.py):
  • Pivot distance < 7.0×ATR (removes outlier wide-stop entries)
  • Correlation filter: max 1 correlated position per symbol group (corr > 0.80)
  • L4 FOMC macro pause: skip 24h BEFORE scheduled FOMC events
  • SL cooldown: 4h after stop-loss per symbol
  • Max concurrent positions: 5
  • Bear mode (BTC below 50-day SMA): long_only — skip SHORT signals

Backtest-confirmed parameters (DO NOT CHANGE without re-running full_audit_v7.py):
  risk=4%, max_pos=5
  ST:  tp_r=2.5, piv_lookback=5, buffer_atr=0.3
  SMA: tp_r=4.0, piv_lookback=3, buffer_atr=0.3
  RSI: sl_atr=2.0, tp_r=3.0
  FIB: sl_buf=0.5, tp_ext=1.0, cons_bars=18, pump_min=3.0, pump_max_bars=6
"""
from __future__ import annotations

import asyncio
import logging
import math
import time as _time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import multi_agent.config as cfg

if TYPE_CHECKING:
    from multi_agent.shared.bybit_client import BybitClient
    from multi_agent.shared.state_manager import StateManager
    from multi_agent.shared.telegram_notifier import MultiAgentTelegramNotifier
    from multi_agent.shared.risk_manager import RiskManager

# ── Logger ─────────────────────────────────────────────────────────────────────
_LOG_PATH = Path(cfg.BASE_DIR) / "logs" / "v7_agent.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("v7_agent")
if not log.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = RotatingFileHandler(
        str(_LOG_PATH), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(_fmt)
    log.addHandler(_ch)
    log.setLevel(logging.INFO)

AGENT_ID = "v7_agent"

# ── Symbols traded (same BASELINE as backtest) ─────────────────────────────────
BASELINE_SYMBOLS = [
    "ADA", "ALGO", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB", "BTC",
    "DOGE", "DOT", "ETH", "FIL", "HBAR", "ICP", "INJ", "LINK", "LTC",
    "NEAR", "OP", "SOL", "SUI", "TON", "TRX", "UNI", "XLM", "XRP",
]
# Map base ticker → Bybit USDT perpetual symbol
def _bybit_sym(base: str) -> str:
    return f"{base}/USDT:USDT"

SYMBOLS: list[str] = [_bybit_sym(b) for b in BASELINE_SYMBOLS]

# ── Strategy IDs ───────────────────────────────────────────────────────────────
STRAT_SUPERTREND = "V7_ST_PIVOT_4H"
STRAT_SMACROSS   = "V7_SMA_CROSS_1D"
STRAT_RSI_REV    = "V7_RSI_REV_4H"
STRAT_FIB_PUMP   = "V7_FIB_PUMP_4H"

# ── Backtest-validated parameters ─────────────────────────────────────────────
RISK_PCT          = 0.04    # 4% of balance per trade
MAX_POSITIONS     = 5       # max concurrent open positions
MAX_LEVERAGE      = 25      # cap leverage at 25× (backtest MAX_LEVERAGE)
MAX_MARGIN_FRAC   = 0.10    # max 10% of balance as margin per trade
MIN_MARGIN_USD    = 5.0     # Bybit minimum margin floor

PIVOT_DIST_MAX_ATR = 7.0    # reject signal if SL distance > 7×ATR

SL_COOLDOWN_SEC   = 4 * 3600     # 4h cooldown after SL hit
SL_STREAK_LIMIT   = 2            # after N consecutive SLs → extended cooldown
SL_EXTENDED_SEC   = 24 * 3600    # 24h extended cooldown after streak

# SuperTrend strategy params
ST_PERIOD         = 10
ST_MULT           = 3.0
ST_TP_R           = 2.5
ST_PIV_LB         = 5
ST_BUF_ATR        = 0.3

# SMA Cross strategy params
SMA_TP_R          = 4.0
SMA_PIV_LB        = 3
SMA_BUF_ATR       = 0.3

# RSI Reversion strategy params
RSI_OS            = 20      # oversold threshold
RSI_OB            = 80      # overbought threshold
RSI_SL_ATR        = 2.0
RSI_TP_R          = 3.0

# Fib Pump strategy params
FIB_CONS_BARS     = 18
FIB_CONS_RANGE    = 1.5     # consolidation max range in ATR units (portfolio_v7 default)
FIB_PUMP_MIN_ATR  = 3.0
FIB_PUMP_MAX_BARS = 6
FIB_RET_MIN       = 0.382
FIB_RET_MAX       = 0.618
FIB_TP_EXT        = 1.0     # TP extension beyond pump top (× move)
FIB_SL_BUF        = 0.5     # SL buffer in ATR units below consolidation high

# Correlation filter
CORR_MAX_OPEN    = 1        # max 1 position among correlated symbols
CORR_THRESHOLD   = 0.80     # pairwise correlation threshold
CORR_LOOKBACK_D  = 90       # days of 1d closes for correlation matrix

# L4 FOMC blackout (24h before each event)
L4_BLACKOUT_H    = 24
# FOMC 2025-2026 scheduled dates (UTC midnight of meeting day)
# Source: federalreserve.gov calendar — update annually
FOMC_DATES_UTC: list[tuple[int, int, int]] = [
    (2025, 1, 29), (2025, 3, 19), (2025, 5, 7), (2025, 6, 18),
    (2025, 7, 30), (2025, 9, 17), (2025, 10, 29), (2025, 12, 10),
    (2026, 1, 28), (2026, 3, 18), (2026, 4, 29), (2026, 6, 17),
    (2026, 7, 29), (2026, 9, 16), (2026, 10, 28), (2026, 12, 9),
]
# Pre-compute FOMC timestamps in milliseconds
_FOMC_MS: list[int] = [
    int(datetime(y, mo, d, 18, 0, tzinfo=timezone.utc).timestamp() * 1000)
    for y, mo, d in FOMC_DATES_UTC
]  # FOMC typically announces at 18:00 UTC

# Limit order wait: max 3 cycles (= 15 minutes) before expiring
LIMIT_MAX_CYCLES = 3


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR FUNCTIONS  (pure NumPy, no external dependencies)
# ══════════════════════════════════════════════════════════════════════════════

def _atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, n: int) -> np.ndarray:
    """Average True Range (Wilder's smoothing)."""
    sz = len(cl)
    tr = np.full(sz, np.nan)
    for i in range(1, sz):
        tr[i] = max(hi[i] - lo[i],
                    abs(hi[i] - cl[i-1]),
                    abs(lo[i] - cl[i-1]))
    atr_arr = np.full(sz, np.nan)
    if sz < n + 1:
        return atr_arr
    atr_arr[n] = float(np.nanmean(tr[1:n+1]))
    for i in range(n + 1, sz):
        if not np.isnan(atr_arr[i-1]) and not np.isnan(tr[i]):
            atr_arr[i] = (atr_arr[i-1] * (n - 1) + tr[i]) / n
    return atr_arr


def _sma(x: np.ndarray, n: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full(len(x), np.nan)
    for i in range(n - 1, len(x)):
        out[i] = float(np.mean(x[i - n + 1 : i + 1]))
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """Exponential moving average."""
    out = np.full(len(x), np.nan)
    k = 2.0 / (n + 1)
    # Find first non-NaN and seed
    start = n - 1
    while start < len(x) and np.isnan(x[start]):
        start += 1
    if start >= len(x):
        return out
    out[start] = float(np.nanmean(x[max(0, start - n + 1): start + 1]))
    for i in range(start + 1, len(x)):
        if np.isnan(x[i]):
            out[i] = out[i-1]
        else:
            out[i] = x[i] * k + out[i-1] * (1.0 - k)
    return out


def _rsi(x: np.ndarray, n: int) -> np.ndarray:
    """
    Relative Strength Index (Wilder's smoothing).
    Initialization: simple average of first n differences.
    Then Wilder's EMA: avg = (avg*(n-1) + new) / n.
    Output at bar n (using delta[0..n-1]), then n+1, n+2, ...
    """
    out = np.full(len(x), np.nan)
    if len(x) < n + 1:
        return out
    delta  = np.diff(x.astype(float))             # len = len(x)-1
    gains  = np.where(delta > 0, delta, 0.0)
    losses = np.where(delta < 0, -delta, 0.0)
    # Seed with mean of first n differences (covers bars 0..n-1)
    avg_gain = float(np.mean(gains[:n]))
    avg_loss = float(np.mean(losses[:n]))
    if avg_loss == 0:
        out[n] = 100.0
    else:
        out[n] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    # Wilder's smoothing for bars n+1 .. len(x)-1
    # At bar i, the new difference is delta[i-1] = x[i] - x[i-1]
    for i in range(n + 1, len(x)):
        avg_gain = (avg_gain * (n - 1) + gains[i - 1]) / n
        avg_loss = (avg_loss * (n - 1) + losses[i - 1]) / n
        if avg_loss == 0:
            out[i] = 100.0
        else:
            out[i] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
    return out


def _supertrend(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray,
                period: int = 10, mult: float = 3.0):
    """
    SuperTrend indicator.
    Returns: (supertrend_line, direction_array)
      direction: +1 = bullish, -1 = bearish
    """
    a = _atr(hi, lo, cl, period)
    n = len(cl)
    upper_basic = (hi + lo) / 2 + mult * a
    lower_basic = (hi + lo) / 2 - mult * a
    upper = np.copy(upper_basic)
    lower = np.copy(lower_basic)
    direction = np.zeros(n, dtype=int)
    st_line = np.full(n, np.nan)

    for i in range(1, n):
        if np.isnan(a[i]):
            continue
        # Lower band
        if lower_basic[i] > lower[i-1] or cl[i-1] < lower[i-1]:
            lower[i] = lower_basic[i]
        else:
            lower[i] = lower[i-1]
        # Upper band
        if upper_basic[i] < upper[i-1] or cl[i-1] > upper[i-1]:
            upper[i] = upper_basic[i]
        else:
            upper[i] = upper[i-1]
        # Direction
        if np.isnan(st_line[i-1]):
            direction[i] = 1
        elif cl[i] > upper[i-1]:
            direction[i] = 1
        elif cl[i] < lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
        st_line[i] = lower[i] if direction[i] == 1 else upper[i]

    return st_line, direction


def _find_pivots(hi: np.ndarray, lo: np.ndarray,
                 lookback: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect pivot highs/lows.
    A pivot is confirmed `lookback` bars after it forms.
    Returns: (pivot_hi, pivot_lo) — NaN where no pivot.
    """
    n = len(hi)
    piv_hi = np.full(n, np.nan)
    piv_lo = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        seg_hi = hi[i - lookback : i + lookback + 1]
        seg_lo = lo[i - lookback : i + lookback + 1]
        if hi[i] == seg_hi.max():
            piv_hi[i + lookback] = hi[i]  # available `lookback` bars later
        if lo[i] == seg_lo.min():
            piv_lo[i + lookback] = lo[i]
    return piv_hi, piv_lo


def _latest_pivot_low(piv_lo: np.ndarray, i: int,
                       max_lb: int = 30) -> Optional[float]:
    for k in range(i, max(0, i - max_lb) - 1, -1):
        if not np.isnan(piv_lo[k]):
            return float(piv_lo[k])
    return None


def _latest_pivot_high(piv_hi: np.ndarray, i: int,
                        max_lb: int = 30) -> Optional[float]:
    for k in range(i, max(0, i - max_lb) - 1, -1):
        if not np.isnan(piv_hi[k]):
            return float(piv_hi[k])
    return None


# ══════════════════════════════════════════════════════════════════════════════
# STRATEGY SIGNAL DETECTORS  (call on bar[-2] = last confirmed closed bar)
# ══════════════════════════════════════════════════════════════════════════════

def detect_supertrend_signal(
    hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, ts: np.ndarray,
    bear_mode: bool = False,
) -> Optional[dict]:
    """
    SuperTrend pivot 4h signal.
    Signal bar = bar[-2] (confirmed closed). Entry = bar[-1].open ≈ bar[-2].close.
    Returns signal dict or None.
    """
    n = len(cl)
    if n < 50:
        return None
    i = n - 2  # last confirmed bar

    a = _atr(hi, lo, cl, 14)
    if np.isnan(a[i]) or a[i] <= 0:
        return None

    _, sdir = _supertrend(hi, lo, cl, ST_PERIOD, ST_MULT)
    if i < 1:
        return None

    direction: Optional[str] = None
    if sdir[i] == 1 and sdir[i - 1] == -1:
        direction = "LONG"
    elif sdir[i] == -1 and sdir[i - 1] == 1:
        direction = "SHORT"
    else:
        return None

    if bear_mode and direction == "SHORT":
        return None

    piv_hi, piv_lo = _find_pivots(hi, lo, lookback=ST_PIV_LB)
    entry = float(cl[i])  # limit at signal bar close = next bar open

    if direction == "LONG":
        piv = _latest_pivot_low(piv_lo, i, max_lb=20)
        if piv is None:
            return None
        sl = piv - ST_BUF_ATR * a[i]
        if sl >= entry:
            return None
        dist_atr = (entry - sl) / a[i]
        if dist_atr > PIVOT_DIST_MAX_ATR:
            return None
        tp = entry + ST_TP_R * (entry - sl)
    else:  # SHORT
        piv = _latest_pivot_high(piv_hi, i, max_lb=20)
        if piv is None:
            return None
        sl = piv + ST_BUF_ATR * a[i]
        if sl <= entry:
            return None
        dist_atr = (sl - entry) / a[i]
        if dist_atr > PIVOT_DIST_MAX_ATR:
            return None
        tp = entry - ST_TP_R * (sl - entry)

    return {
        "strategy_id": STRAT_SUPERTREND,
        "direction":   direction,
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "dist_atr":    dist_atr,
        "signal_bar_ts": int(ts[i]),
        "atr":         float(a[i]),
    }


def detect_smacross_signal(
    hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, ts: np.ndarray,
    bear_mode: bool = False,
) -> Optional[dict]:
    """
    SMA50/200 golden/death cross on 1d.
    Signal bar = bar[-2]. Entry = bar[-1].open ≈ bar[-2].close.
    """
    n = len(cl)
    if n < 210:
        return None
    i = n - 2

    a = _atr(hi, lo, cl, 14)
    if np.isnan(a[i]) or a[i] <= 0:
        return None

    s50  = _sma(cl, 50)
    s200 = _sma(cl, 200)
    if np.isnan(s50[i]) or np.isnan(s200[i]):
        return None
    if np.isnan(s50[i-1]) or np.isnan(s200[i-1]):
        return None

    direction: Optional[str] = None
    if s50[i-1] <= s200[i-1] and s50[i] > s200[i]:
        direction = "LONG"
    elif s50[i-1] >= s200[i-1] and s50[i] < s200[i]:
        direction = "SHORT"
    else:
        return None

    if bear_mode and direction == "SHORT":
        return None

    piv_hi, piv_lo = _find_pivots(hi, lo, lookback=SMA_PIV_LB)
    entry = float(cl[i])

    if direction == "LONG":
        piv = _latest_pivot_low(piv_lo, i, max_lb=15)
        if piv is None:
            return None
        sl = piv - SMA_BUF_ATR * a[i]
        if sl >= entry:
            return None
        dist_atr = (entry - sl) / a[i]
        if dist_atr > PIVOT_DIST_MAX_ATR:
            return None
        tp = entry + SMA_TP_R * (entry - sl)
    else:
        piv = _latest_pivot_high(piv_hi, i, max_lb=15)
        if piv is None:
            return None
        sl = piv + SMA_BUF_ATR * a[i]
        if sl <= entry:
            return None
        dist_atr = (sl - entry) / a[i]
        if dist_atr > PIVOT_DIST_MAX_ATR:
            return None
        tp = entry - SMA_TP_R * (sl - entry)

    return {
        "strategy_id": STRAT_SMACROSS,
        "direction":   direction,
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "dist_atr":    dist_atr,
        "signal_bar_ts": int(ts[i]),
        "atr":         float(a[i]),
    }


def detect_rsi_rev_signal(
    hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, ts: np.ndarray,
    is_sideways: bool = True,
    bear_mode: bool = False,
) -> Optional[dict]:
    """
    RSI reversion on 4h.
    Only fires in sideways regime (is_sideways=True).
    Signal: RSI crosses back above 20 (LONG) or below 80 (SHORT).
    """
    n = len(cl)
    if n < 30:
        return None
    if not is_sideways:
        return None

    i = n - 2
    a = _atr(hi, lo, cl, 14)
    if np.isnan(a[i]) or a[i] <= 0:
        return None

    r = _rsi(cl, 14)
    if np.isnan(r[i]) or np.isnan(r[i-1]):
        return None

    direction: Optional[str] = None
    if r[i-1] <= RSI_OS and r[i] > RSI_OS:
        direction = "LONG"
    elif r[i-1] >= RSI_OB and r[i] < RSI_OB:
        direction = "SHORT"
    else:
        return None

    if bear_mode and direction == "SHORT":
        return None

    entry = float(cl[i])
    if direction == "LONG":
        sl = entry - RSI_SL_ATR * a[i]
        tp = entry + RSI_TP_R * (entry - sl)
        dist_atr = RSI_SL_ATR
    else:
        sl = entry + RSI_SL_ATR * a[i]
        tp = entry - RSI_TP_R * (sl - entry)
        dist_atr = RSI_SL_ATR

    # Pivot distance check (RSI uses ATR-based SL so dist_atr = RSI_SL_ATR = 2.0)
    if dist_atr > PIVOT_DIST_MAX_ATR:
        return None

    return {
        "strategy_id": STRAT_RSI_REV,
        "direction":   direction,
        "entry":       entry,
        "sl":          sl,
        "tp":          tp,
        "dist_atr":    dist_atr,
        "signal_bar_ts": int(ts[i]),
        "atr":         float(a[i]),
    }


def detect_fib_pump_signal(
    hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, ts: np.ndarray,
    op: np.ndarray,
    bear_mode: bool = False,
) -> Optional[dict]:
    """
    Fibonacci pump retrace on 4h.
    Looks back from bar[-2] for pattern:
      1. Consolidation (18 bars, range < 1.5×ATR)
      2. Pump (min 3.0×ATR break within 6 bars)
      3. Retrace to 38-62% of pump move
    Signal on retrace bar[-2], entry at bar[-2].close.
    """
    n = len(cl)
    if n < FIB_CONS_BARS + FIB_PUMP_MAX_BARS + 20:
        return None

    i = n - 2  # signal bar (confirmed closed)
    a = _atr(hi, lo, cl, 14)
    if np.isnan(a[i]) or a[i] <= 0:
        return None

    # Scan backwards to find a recent pump+retrace ending at bar i
    # Look back up to 30 bars to find the retrace bar
    RETRACE_MAX_BARS = 24
    result = None

    # We search for the pattern ending at bar i (the retrace is bar i)
    # Walk back in time: for each possible pump top position before i
    for top_offset in range(1, RETRACE_MAX_BARS + FIB_PUMP_MAX_BARS):
        top_idx = i - top_offset
        if top_idx < FIB_CONS_BARS + 14:
            break

        # Check if bar i is in retrace zone from top at top_idx
        # First find the pump that led to top_idx
        # Consolidation ends at cons_end = top_idx - pump_bars
        for pump_bars in range(1, FIB_PUMP_MAX_BARS + 1):
            cons_end = top_idx - pump_bars
            if cons_end < FIB_CONS_BARS + 14:
                break

            cons_start = cons_end - FIB_CONS_BARS
            if cons_start < 0:
                break

            cons_hi = float(np.max(hi[cons_start:cons_end]))
            cons_lo = float(np.min(lo[cons_start:cons_end]))
            cons_range = cons_hi - cons_lo
            avg_atr = float(np.nanmean(a[cons_start:cons_end]))
            if avg_atr <= 0:
                continue
            # Consolidation check
            if cons_range > FIB_CONS_RANGE * avg_atr:
                continue

            # LONG pump: high broke cons_hi by pump_min_atr × ATR
            atr_at_cons = float(a[cons_end]) if not np.isnan(a[cons_end]) else avg_atr
            if atr_at_cons <= 0:
                continue

            long_pump = hi[top_idx] >= cons_hi + FIB_PUMP_MIN_ATR * atr_at_cons
            short_pump = lo[top_idx] <= cons_lo - FIB_PUMP_MIN_ATR * atr_at_cons

            if not long_pump and not short_pump:
                continue

            if long_pump:
                # Find pump top: highest high in [top_idx-pump_bars+1 .. top_idx]
                pump_range_start = top_idx - pump_bars + 1
                top_hi = float(np.max(hi[pump_range_start:top_idx + 1]))
                anchor = cons_hi
                move = top_hi - anchor
                if move <= 0:
                    continue
                # Check if bar i is the retrace bar
                fib_hi_band = top_hi - FIB_RET_MIN * move
                fib_lo_band = top_hi - FIB_RET_MAX * move
                if lo[i] <= fib_hi_band and lo[i] >= fib_lo_band:
                    direction = "LONG"
                    if bear_mode:
                        break
                    entry = float(cl[i])
                    sl = cons_hi - FIB_SL_BUF * a[i]
                    if sl >= entry:
                        continue
                    dist = entry - sl
                    if dist <= 0:
                        continue
                    dist_atr = dist / a[i]
                    if dist_atr > PIVOT_DIST_MAX_ATR:
                        continue
                    tp = top_hi + FIB_TP_EXT * move
                    result = {
                        "strategy_id": STRAT_FIB_PUMP,
                        "direction":   "LONG",
                        "entry":       entry,
                        "sl":          sl,
                        "tp":          tp,
                        "dist_atr":    dist_atr,
                        "signal_bar_ts": int(ts[i]),
                        "atr":         float(a[i]),
                    }
                    break
                # Check if bar i is already past the retrace zone (structure broke)
                if lo[i] < cons_lo:
                    break

            elif short_pump:
                pump_range_start = top_idx - pump_bars + 1
                bot_lo = float(np.min(lo[pump_range_start:top_idx + 1]))
                anchor = cons_lo
                move = anchor - bot_lo
                if move <= 0:
                    continue
                fib_hi_band = bot_lo + FIB_RET_MAX * move
                fib_lo_band = bot_lo + FIB_RET_MIN * move
                if hi[i] >= fib_lo_band and hi[i] <= fib_hi_band:
                    if bear_mode:
                        break
                    entry = float(cl[i])
                    sl = cons_lo + FIB_SL_BUF * a[i]
                    if sl <= entry:
                        continue
                    dist = sl - entry
                    if dist <= 0:
                        continue
                    dist_atr = dist / a[i]
                    if dist_atr > PIVOT_DIST_MAX_ATR:
                        continue
                    tp = bot_lo - FIB_TP_EXT * move
                    result = {
                        "strategy_id": STRAT_FIB_PUMP,
                        "direction":   "SHORT",
                        "entry":       entry,
                        "sl":          sl,
                        "tp":          tp,
                        "dist_atr":    dist_atr,
                        "signal_bar_ts": int(ts[i]),
                        "atr":         float(a[i]),
                    }
                    break
                if hi[i] > cons_hi:
                    break

        if result is not None:
            break

    return result


# ══════════════════════════════════════════════════════════════════════════════
# V7 AGENT
# ══════════════════════════════════════════════════════════════════════════════

class V7Agent:
    """
    V7 portfolio swing trading agent.
    Implements 4 validated strategies on Bybit USDT perpetual futures.
    All entries via post-only limit orders. SL/TP via Bybit conditional orders.
    """

    def __init__(
        self,
        agent_id: str,
        state_manager: "StateManager",
        bybit_client: "BybitClient",
        risk_manager: "RiskManager",
        telegram: "MultiAgentTelegramNotifier",
    ) -> None:
        self.agent_id = agent_id
        self.state    = state_manager
        self.client   = bybit_client
        self.risk     = risk_manager
        self.tg       = telegram

        self._trade_lock: asyncio.Lock = asyncio.Lock()

        # Pending signals: (symbol, strategy_id) → signal_dict
        # Signal is stored once per (sym, strat); replaced if same bar re-detected
        self._pending: dict[tuple[str, str], dict] = {}

        # Pending limit orders awaiting fill: (symbol, strategy_id) → order_info
        self._pending_limit: dict[tuple[str, str], dict] = {}

        # Signal deduplication: track last signal bar ts per (sym, strat)
        self._last_signal_bar_ts: dict[tuple[str, str], int] = {}

        # SL cooldown per symbol: {symbol: epoch_time_blocked_until}
        self._sl_cooldown: dict[str, float] = {}
        self._sl_streak:   dict[str, int]   = {}

        # Correlation matrix: {(sym_a, sym_b): correlation}
        # Rebuilt at startup and every CORR_REBUILD_H hours
        self._corr_matrix: dict[tuple[str, str], float] = {}
        self._last_corr_build: float = 0.0
        CORR_REBUILD_H = 24  # hours between matrix rebuilds

        # Bear regime cache: (is_bear, built_at_monotonic)
        self._bear_cache: tuple[bool, float] = (False, 0.0)
        BEAR_CACHE_TTL = 4 * 3600  # refresh every 4h

        # Sideways regime cache: (is_sideways, built_at_monotonic)
        self._sideways_cache: tuple[bool, float] = (True, 0.0)
        SIDEWAYS_CACHE_TTL = 4 * 3600  # refresh every 4h

        # Markets cache for position sizing
        self._markets_cache: dict = {}

        # Daily stats
        self.daily_trades  = 0
        self.daily_wins    = 0
        self.daily_losses  = 0
        self.daily_pnl     = 0.0
        self.last_daily_reset = ""

        # Internal config (for rebuild intervals)
        self._CORR_REBUILD_H = 24
        self._BEAR_CACHE_TTL = 4 * 3600
        self._SIDEWAYS_CACHE_TTL = 4 * 3600

    # ── Heartbeat ─────────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        await self.state.record_heartbeat(self.agent_id)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _short(self, sym: str) -> str:
        return sym.replace("/USDT:USDT", "USDT")

    async def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_daily_reset != today:
            self.daily_trades = 0
            self.daily_wins   = 0
            self.daily_losses = 0
            self.daily_pnl    = 0.0
            self.last_daily_reset = today
            log.info("[V7] Daily stats reset for %s", today)

    # ── SL cooldown ───────────────────────────────────────────────────────────

    def _is_on_cooldown(self, sym: str) -> bool:
        until = self._sl_cooldown.get(sym, 0.0)
        return _time.time() < until

    def _set_sl_cooldown(self, sym: str) -> None:
        streak = self._sl_streak.get(sym, 0) + 1
        self._sl_streak[sym] = streak
        if streak >= SL_STREAK_LIMIT:
            cd = SL_EXTENDED_SEC
            log.warning("[V7] %s SL streak=%d → blocked %.0fh", self._short(sym), streak, cd/3600)
        else:
            cd = SL_COOLDOWN_SEC
            log.info("[V7] %s SL → cooldown %.0fh", self._short(sym), cd/3600)
        self._sl_cooldown[sym] = _time.time() + cd
        # Clear any pending signals/orders for this symbol
        for strat in (STRAT_SUPERTREND, STRAT_SMACROSS, STRAT_RSI_REV, STRAT_FIB_PUMP):
            self._pending.pop((sym, strat), None)
            self._pending_limit.pop((sym, strat), None)

    def _reset_sl_streak(self, sym: str) -> None:
        self._sl_streak.pop(sym, None)

    # ── FOMC blackout ─────────────────────────────────────────────────────────

    def _is_fomc_blackout(self) -> bool:
        """Return True if we are within L4_BLACKOUT_H hours BEFORE a FOMC event."""
        now_ms = int(_time.time() * 1000)
        window = L4_BLACKOUT_H * 3_600_000
        for fomc_ms in _FOMC_MS:
            time_to = fomc_ms - now_ms
            if 0 < time_to < window:
                fomc_dt = datetime.fromtimestamp(fomc_ms / 1000, tz=timezone.utc)
                log.info("[V7] L4 FOMC blackout — %.1fh until %s",
                         time_to / 3_600_000, fomc_dt.strftime("%Y-%m-%d %H:%M"))
                return True
        return False

    # ── Bear regime ───────────────────────────────────────────────────────────

    async def _is_bear_regime(self) -> bool:
        """
        Returns True if BTC is in a bear market (price below 50-day SMA).
        Cached for 4h to avoid repeated fetches.
        """
        is_bear, built_at = self._bear_cache
        if _time.monotonic() - built_at < self._BEAR_CACHE_TTL:
            return is_bear
        try:
            raw = await self.client.fetch_ohlcv("BTC/USDT:USDT", "1d", 60)
            if not raw or len(raw) < 52:
                return is_bear  # reuse old value
            closes = np.array([float(c[4]) for c in raw])
            s50 = _sma(closes, 50)
            bear = closes[-1] < s50[-1]
            self._bear_cache = (bear, _time.monotonic())
            log.info("[V7] Bear regime check: BTC %.0f vs SMA50 %.0f → %s",
                     closes[-1], s50[-1], "BEAR" if bear else "BULL")
            return bear
        except Exception as exc:
            log.warning("[V7] Bear regime fetch failed: %s", exc)
            return is_bear  # reuse cached value on error

    # ── Sideways regime (for RSI strategy) ───────────────────────────────────

    async def _is_sideways_regime(self) -> bool:
        """
        Returns True if BTC is in a sideways (non-trending) regime.
        Simple check: |SMA50 slope over 10d| / price < 3%.
        Cached for 4 hours to avoid redundant API calls.
        """
        is_sideways, cached_at = self._sideways_cache
        if _time.monotonic() - cached_at < self._SIDEWAYS_CACHE_TTL:
            return is_sideways
        try:
            raw = await self.client.fetch_ohlcv("BTC/USDT:USDT", "1d", 60)
            if not raw or len(raw) < 52:
                return True  # assume sideways if no data
            closes = np.array([float(c[4]) for c in raw])
            s50 = _sma(closes, 50)
            # slope over last 10 bars
            slope_pct = abs(s50[-1] - s50[-11]) / s50[-11] * 100 if s50[-11] > 0 else 0
            sideways = slope_pct < 3.0  # <3% trend over 10d = sideways enough
            self._sideways_cache = (sideways, _time.monotonic())
            return sideways
        except Exception:
            return True

    # ── Correlation filter ────────────────────────────────────────────────────

    async def _build_corr_matrix(self) -> None:
        """
        Build pairwise correlation matrix from 90 days of 1d closes.
        Stored as flat dict (sym_a, sym_b) → corr.
        Only rebuilds every CORR_REBUILD_H hours.
        """
        now = _time.monotonic()
        if now - self._last_corr_build < self._CORR_REBUILD_H * 3600:
            return
        log.info("[V7] Building correlation matrix...")
        closes_map: dict[str, np.ndarray] = {}
        for sym in SYMBOLS:
            try:
                raw = await self.client.fetch_ohlcv(sym, "1d", CORR_LOOKBACK_D + 5)
                if raw and len(raw) >= CORR_LOOKBACK_D:
                    closes_map[sym] = np.array([float(c[4]) for c in raw[-CORR_LOOKBACK_D:]])
            except Exception:
                pass
        # Compute log-return correlations
        syms_available = list(closes_map.keys())
        new_matrix: dict[tuple[str, str], float] = {}
        for i, sa in enumerate(syms_available):
            ra = np.diff(np.log(closes_map[sa] + 1e-10))
            for sb in syms_available[i:]:
                rb = np.diff(np.log(closes_map[sb] + 1e-10))
                min_len = min(len(ra), len(rb))
                if min_len < 20:
                    continue
                corr = float(np.corrcoef(ra[-min_len:], rb[-min_len:])[0, 1])
                new_matrix[(sa, sb)] = corr
                new_matrix[(sb, sa)] = corr
        self._corr_matrix = new_matrix
        self._last_corr_build = now
        log.info("[V7] Correlation matrix built: %d symbol pairs", len(new_matrix))

    def _corr_allows_trade(self, new_sym: str, open_positions: dict) -> bool:
        """
        Returns True if trade on new_sym is allowed under correlation filter.
        Blocks if ≥ CORR_MAX_OPEN positions with corr > CORR_THRESHOLD already open.
        """
        corr_open = 0
        for open_sym in open_positions:
            if open_sym == new_sym:
                continue
            key_a = (new_sym, open_sym)
            corr = self._corr_matrix.get(key_a)
            if corr is None:
                continue  # no data → allow
            if abs(corr) > CORR_THRESHOLD:
                corr_open += 1
        return corr_open < CORR_MAX_OPEN

    # ── Position sizing ───────────────────────────────────────────────────────

    async def _calc_position_size(
        self, balance: float, entry: float, sl: float, symbol: str
    ) -> tuple[float, int]:
        """
        Calculate position quantity and leverage.

        Logic (from backtest strategy_lab.py):
          risk_usd  = balance × RISK_PCT
          qty       = risk_usd / |entry - sl|
          notional  = qty × entry
          margin    = notional / leverage
          leverage  = ceil(notional / min(balance × MAX_MARGIN_FRAC, notional))
          → leverage = max(1, ceil(notional / (balance × MAX_MARGIN_FRAC)))
          → capped at MAX_LEVERAGE

        Returns: (qty, leverage) — qty=0 if trade is not viable.
        """
        dist = abs(entry - sl)
        if dist <= 0 or entry <= 0 or balance <= 0:
            return 0.0, 1

        risk_usd = balance * RISK_PCT
        if risk_usd < 0.01:
            return 0.0, 1

        qty = risk_usd / dist
        notional = qty * entry

        # Margin = notional / leverage; we want margin ≤ balance × MAX_MARGIN_FRAC
        max_margin = balance * MAX_MARGIN_FRAC
        if max_margin < MIN_MARGIN_USD:
            return 0.0, 1

        # Required leverage to keep margin within limit
        if notional <= max_margin:
            leverage = 1
        else:
            leverage = math.ceil(notional / max_margin)

        leverage = min(leverage, MAX_LEVERAGE)
        leverage = max(leverage, 1)

        # Verify margin at computed leverage
        margin = notional / leverage
        if margin < MIN_MARGIN_USD:
            # Try to meet minimum by adjusting leverage
            min_notional_at_min_margin = MIN_MARGIN_USD * leverage
            if notional < MIN_MARGIN_USD:
                return 0.0, 1  # too small

        # Apply Bybit contract size rounding (get from market info)
        market_info = self._markets_cache.get(symbol, {})
        precision = market_info.get("precision", {})
        step = precision.get("amount", None)
        if step and step > 0:
            qty = math.floor(qty / step) * step

        if qty <= 0:
            return 0.0, 1

        return round(qty, 8), int(leverage)

    # ── Fetch balance ─────────────────────────────────────────────────────────

    async def _fetch_balance(self) -> float:
        try:
            raw = await self.client.fetch_balance()
            # Standard CCXT format
            bal = float(
                raw.get("USDT", {}).get("total", 0)
                or raw.get("total", {}).get("USDT", 0)
                or 0
            )
            # Bybit UTA accounts: equity lives in info.result.list[].coin[].equity
            if bal < 1.0:
                try:
                    for acct in raw.get("info", {}).get("result", {}).get("list", []):
                        for coin_data in acct.get("coin", []):
                            if coin_data.get("coin") == "USDT":
                                eq = float(
                                    coin_data.get("equity", 0)
                                    or coin_data.get("walletBalance", 0)
                                    or 0
                                )
                                if eq > bal:
                                    bal = eq
                except Exception:
                    pass
            if bal > 0:
                log.debug("[V7] balance=%.4f", bal)
                return bal
        except Exception as exc:
            log.debug("[V7] fetch_balance error: %s", exc)
        return float(await self.state.get_agent_equity(self.agent_id))

    # ── Max positions check ───────────────────────────────────────────────────

    async def _can_open_position(self) -> bool:
        """Returns True if we can open another position (under MAX_POSITIONS)."""
        positions = await self.state.get_agent_positions(self.agent_id)
        # Also count pending limit orders (they will become positions)
        pending_cnt = len(self._pending_limit)
        total = len(positions) + pending_cnt
        return total < MAX_POSITIONS

    # ══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ══════════════════════════════════════════════════════════════════════════

    async def _heartbeat_sleep(self, total_sec: int = 300, interval: int = 60) -> None:
        elapsed = 0
        while elapsed < total_sec:
            step = min(interval, total_sec - elapsed)
            await asyncio.sleep(step)
            elapsed += step
            await self.heartbeat()

    async def run(self) -> None:
        """Main agent loop. Runs every 5 minutes."""
        log.info("[V7] V7 Agent starting. Symbols: %d, Strategies: ST/SMA/RSI/FIB", len(SYMBOLS))
        await self.heartbeat()

        # Build correlation matrix at startup
        try:
            await self._build_corr_matrix()
        except Exception as exc:
            log.warning("[V7] Initial corr matrix build failed: %s", exc)

        # Load markets cache for position sizing
        try:
            markets = self.client._exchange.markets or {}
            if markets:
                self._markets_cache = dict(markets)
        except Exception:
            pass

        while True:
            try:
                await self._reset_daily_if_needed()
                await self._scan_cycle()
                await self._heartbeat_sleep(300, 60)
            except asyncio.CancelledError:
                log.info("[V7] Agent cancelled.")
                break
            except Exception as exc:
                log.exception("[V7] Unexpected error in main loop: %s", exc)
                try:
                    await self.tg.send_error(self.agent_id, str(exc))
                except Exception:
                    pass
                await asyncio.sleep(10)

    async def _scan_cycle(self) -> None:
        """One full cycle: manage positions → rebuild corr → scan signals → check entries."""
        await self.heartbeat()

        # 1. Rebuild correlation matrix if stale
        try:
            await self._build_corr_matrix()
        except Exception as exc:
            log.debug("[V7] Corr rebuild error: %s", exc)

        # 2. Check open positions (detect exchange-closed)
        await self._manage_positions()

        await self.heartbeat()

        # 3. Scan for new signals (if FOMC blackout, skip scanning)
        if self._is_fomc_blackout():
            log.info("[V7] FOMC blackout — skipping signal scan this cycle")
        else:
            await self._scan_for_signals()

        await self.heartbeat()

        # 4. Check pending limit orders
        await self._check_pending_entries()

        # 5. Cycle summary
        positions = await self.state.get_agent_positions(self.agent_id)
        pending_sig = len(self._pending)
        pending_lim = len(self._pending_limit)
        log.info(
            "[V7] Cycle done. pos=%d  pending_sig=%d  pending_lim=%d  trades_today=%d (W:%d L:%d) pnl=$%.2f",
            len(positions), pending_sig, pending_lim,
            self.daily_trades, self.daily_wins, self.daily_losses, self.daily_pnl,
        )

    # ── Position management ───────────────────────────────────────────────────

    async def _manage_positions(self) -> None:
        """Detect positions closed by Bybit (SL/TP fired) and record them."""
        positions = await self.state.get_agent_positions(self.agent_id)
        if not positions:
            return

        bybit_symbols: Optional[set] = None
        if not cfg.SIMULATION_MODE:
            try:
                bybit_pos = await self.client.fetch_all_positions()
                bybit_symbols = {
                    p["symbol"] for p in bybit_pos
                    if float(p.get("contracts", 0) or 0) != 0
                }
            except Exception as exc:
                log.debug("[V7] Bybit position fetch failed: %s", exc)

        for sym, pos in list(positions.items()):
            try:
                ticker = await self.client.fetch_ticker(sym)
                price  = float(ticker.get("last") or 0)
                if price <= 0:
                    continue

                side = str(pos.get("side", "LONG"))
                sl   = float(pos.get("sl", 0))
                tp   = float(pos.get("tp", 0))

                # Path 1: Exchange closed the position
                if bybit_symbols is not None and sym not in bybit_symbols:
                    dist_sl = abs(price - sl) if sl > 0 else float("inf")
                    dist_tp = abs(price - tp) if tp > 0 else float("inf")
                    reason  = "TP" if dist_tp <= dist_sl else "SL"
                    log.info("[V7] %s closed by exchange at %.6g (inferred=%s)",
                             self._short(sym), price, reason)
                    await self._close_position(sym, pos, price, reason,
                                               skip_exchange_order=True)
                    continue

                # Path 2: Price-based detection (simulation or fallback)
                hit = False; reason = ""
                if side == "LONG":
                    if sl > 0 and price <= sl:
                        hit = True; reason = "SL"
                    elif tp > 0 and price >= tp:
                        hit = True; reason = "TP"
                else:
                    if sl > 0 and price >= sl:
                        hit = True; reason = "SL"
                    elif tp > 0 and price <= tp:
                        hit = True; reason = "TP"

                if hit:
                    await self._close_position(sym, pos, price, reason,
                                               skip_exchange_order=False)

            except Exception as exc:
                log.warning("[V7] Position check error %s: %s", self._short(sym), exc)

    async def _close_position(
        self, sym: str, pos: dict, exit_price: float,
        reason: str, skip_exchange_order: bool = False,
        override_pnl: Optional[float] = None,
    ) -> None:
        """Record closed position, update stats, send notification."""
        await self.state.remove_position(self.agent_id, sym)

        side        = str(pos.get("side", "LONG"))
        entry       = float(pos.get("entry", exit_price))
        qty         = float(pos.get("qty", 0))
        strategy_id = pos.get("strategy_id", "V7")

        if override_pnl is not None:
            pnl_usd = override_pnl
        else:
            if side == "LONG":
                pnl_usd = (exit_price - entry) * qty
            else:
                pnl_usd = (entry - exit_price) * qty
            # Deduct taker fees (entry + exit)
            fee = (entry + exit_price) * qty * cfg.TAKER_FEE_PCT
            pnl_usd -= fee

        pnl_pct = pnl_usd / max(entry * qty, 1e-9) * 100.0

        self.daily_pnl    += pnl_usd
        self.daily_trades += 1
        if pnl_usd >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        log.info("[V7] CLOSED [%s] %s %s entry=%.6g exit=%.6g pnl=$%.4f (%.2f%%) reason=%s",
                 strategy_id, self._short(sym), side, entry, exit_price,
                 pnl_usd, pnl_pct, reason)

        if reason == "SL":
            self._set_sl_cooldown(sym)
        elif reason == "TP":
            self._reset_sl_streak(sym)

        # Persist trade
        now_utc = datetime.now(timezone.utc)
        await self.state.record_trade(self.agent_id, {
            "symbol":      sym,
            "strategy_id": strategy_id,
            "side":        side,
            "direction":   side,
            "entry":       entry,
            "exit":        exit_price,
            "sl":          float(pos.get("sl", 0)),
            "tp":          float(pos.get("tp", 0)),
            "qty":         qty,
            "leverage":    pos.get("leverage", 1),
            "pnl_usd":     pnl_usd,
            "pnl_pct":     pnl_pct,
            "reason":      reason,
            "date":        now_utc.strftime("%Y-%m-%d"),
            "opened_at":   pos.get("opened_at", ""),
            "closed_at":   now_utc.isoformat(),
        })

        current_eq = await self.state.get_agent_equity(self.agent_id)
        await self.state.set_agent_equity(self.agent_id, max(0.0, current_eq + pnl_usd))

        if not cfg.SIMULATION_MODE and not skip_exchange_order:
            close_side = "sell" if side == "LONG" else "buy"
            try:
                await self.client.place_market_order(sym, close_side, qty)
            except Exception as exc:
                log.error("[V7] Close order failed %s: %s", self._short(sym), exc)

        try:
            await self.tg.send_position_closed(
                self.agent_id, sym, side, entry, exit_price, pnl_usd, pnl_pct, reason,
            )
        except Exception as exc:
            log.warning("[V7] TG close notification failed: %s", exc)

    async def handle_external_close(self, sym: str, pos: dict) -> None:
        """Called by MasterAgent when position disappears from Bybit."""
        sl    = float(pos.get("sl", 0))
        tp    = float(pos.get("tp", 0))
        entry = float(pos.get("entry", 0))
        side  = str(pos.get("side", "LONG"))

        exit_price = 0.0
        bybit_pnl: Optional[float] = None
        try:
            closed = await self.client.fetch_closed_pnl(hours=1)
            raw_sym = sym.replace("/", "").replace(":USDT", "")
            for rec in closed:
                if rec.get("symbol", "").upper() == raw_sym.upper():
                    ep = float(rec.get("avgExitPrice") or 0)
                    if ep > 0:
                        exit_price = ep
                        bybit_pnl = float(rec.get("closedPnl") or 0)
                        break
        except Exception:
            pass

        if exit_price <= 0:
            try:
                ticker = await self.client.fetch_ticker(sym)
                exit_price = float(ticker.get("last", 0) or 0)
            except Exception:
                pass
        if exit_price <= 0:
            exit_price = entry

        reason = "MANUAL"
        tol = 0.003
        if sl > 0 and tp > 0 and entry > 0:
            if side == "LONG":
                if exit_price <= sl * (1 + tol):
                    reason = "SL"
                elif exit_price >= tp * (1 - tol):
                    reason = "TP"
            else:
                if exit_price >= sl * (1 - tol):
                    reason = "SL"
                elif exit_price <= tp * (1 + tol):
                    reason = "TP"

        await self._close_position(sym, pos, exit_price, reason,
                                   skip_exchange_order=True, override_pnl=bybit_pnl)

    # ── Signal scanning ───────────────────────────────────────────────────────

    async def _scan_for_signals(self) -> None:
        """Scan all BASELINE symbols for new V7 signals."""
        bear = await self._is_bear_regime()
        sideways = await self._is_sideways_regime()

        batch_size = 8
        for i in range(0, len(SYMBOLS), batch_size):
            batch = SYMBOLS[i : i + batch_size]
            tasks = [self._scan_symbol(sym, bear, sideways) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, result in zip(batch, results):
                if isinstance(result, Exception):
                    log.debug("[V7] Scan error %s: %s", self._short(sym), result)
                elif result:
                    for sig in result:
                        key = (sym, sig["strategy_id"])
                        bar_ts = sig["signal_bar_ts"]
                        # Deduplication: skip if same bar already produced this signal
                        if self._last_signal_bar_ts.get(key) == bar_ts:
                            continue
                        self._last_signal_bar_ts[key] = bar_ts
                        self._pending[key] = sig
                        log.info("[V7] SIGNAL [%s] %s %s entry=%.6g sl=%.6g tp=%.6g dist=%.1f×ATR",
                                 sig["strategy_id"], self._short(sym), sig["direction"],
                                 sig["entry"], sig["sl"], sig["tp"], sig["dist_atr"])
            await self.heartbeat()

    async def _scan_symbol(self, sym: str, bear: bool, sideways: bool) -> list[dict]:
        """Scan one symbol for signals from all 4 strategies."""
        # Global gates
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return []
        if self._is_on_cooldown(sym):
            return []
        positions = await self.state.get_agent_positions(self.agent_id)
        if sym in positions:
            return []
        # Check if any pending limit order exists for this symbol (avoid doubling)
        has_pending = any(k[0] == sym for k in self._pending_limit)
        if has_pending:
            return []

        signals: list[dict] = []

        # ── Fetch 4h candles ─────────────────────────────────────────────────
        try:
            raw_4h = await self.client.fetch_ohlcv(sym, "4h", 200)
        except Exception as exc:
            log.debug("[V7] 4h fetch failed %s: %s", self._short(sym), exc)
            raw_4h = None

        if raw_4h and len(raw_4h) >= 60:
            df_4h = pd.DataFrame(raw_4h, columns=["timestamp","open","high","low","close","volume"])
            hi = df_4h["high"].to_numpy(float)
            lo = df_4h["low"].to_numpy(float)
            cl = df_4h["close"].to_numpy(float)
            op = df_4h["open"].to_numpy(float)
            ts = df_4h["timestamp"].to_numpy(int)

            # SuperTrend 4h
            key_st = (sym, STRAT_SUPERTREND)
            if key_st not in self._pending_limit:
                sig = detect_supertrend_signal(hi, lo, cl, ts, bear_mode=bear)
                if sig:
                    signals.append(sig)

            # RSI Reversion 4h
            key_rsi = (sym, STRAT_RSI_REV)
            if key_rsi not in self._pending_limit:
                sig = detect_rsi_rev_signal(hi, lo, cl, ts, is_sideways=sideways, bear_mode=bear)
                if sig:
                    signals.append(sig)

            # Fib Pump 4h
            key_fib = (sym, STRAT_FIB_PUMP)
            if key_fib not in self._pending_limit:
                sig = detect_fib_pump_signal(hi, lo, cl, ts, op, bear_mode=bear)
                if sig:
                    signals.append(sig)

        # ── Fetch 1d candles ─────────────────────────────────────────────────
        key_sma = (sym, STRAT_SMACROSS)
        if key_sma not in self._pending_limit:
            try:
                raw_1d = await self.client.fetch_ohlcv(sym, "1d", 250)
            except Exception as exc:
                log.debug("[V7] 1d fetch failed %s: %s", self._short(sym), exc)
                raw_1d = None

            if raw_1d and len(raw_1d) >= 210:
                df_1d = pd.DataFrame(raw_1d, columns=["timestamp","open","high","low","close","volume"])
                hi1 = df_1d["high"].to_numpy(float)
                lo1 = df_1d["low"].to_numpy(float)
                cl1 = df_1d["close"].to_numpy(float)
                ts1 = df_1d["timestamp"].to_numpy(int)

                sig = detect_smacross_signal(hi1, lo1, cl1, ts1, bear_mode=bear)
                if sig:
                    signals.append(sig)

        return signals

    # ── Entry checking ────────────────────────────────────────────────────────

    async def _check_pending_entries(self) -> None:
        """
        For each pending signal:
          1. If no limit order placed yet → place post-only limit at signal.entry
          2. If limit order in flight → check fill status; expire after LIMIT_MAX_CYCLES
        """
        expired: list[tuple[str, str]] = []

        # --- Check pending signals (no order placed yet) ---
        for key, sig in list(self._pending.items()):
            sym, strat_id = key
            # Already have a limit order for this → skip (handled below)
            if key in self._pending_limit:
                continue

            result = await self._try_place_limit(sym, sig)
            if result == "placed":
                del self._pending[key]
            elif result == "expired":
                expired.append(key)
            # else "blocked" → keep pending, try again next cycle

        for key in expired:
            self._pending.pop(key, None)

        # --- Check pending limit orders (placed, awaiting fill) ---
        expired_limits: list[tuple[str, str]] = []

        for key, order_info in list(self._pending_limit.items()):
            sym, strat_id = key
            result = await self._check_limit_fill(sym, strat_id, order_info)
            if result in ("filled", "expired", "cancelled"):
                expired_limits.append(key)

        for key in expired_limits:
            self._pending_limit.pop(key, None)

    async def _try_place_limit(self, sym: str, sig: dict) -> str:
        """
        Attempt to place a post-only limit order for the signal.
        Returns: "placed" | "blocked" | "expired"
        """
        async with self._trade_lock:
            # Re-validate all gates under lock
            gate = await self._gate_checks(sym, sig)
            if gate != "ok":
                if gate == "expired":
                    return "expired"
                return "blocked"

            direction   = sig["direction"]
            side_str    = "buy" if direction == "LONG" else "sell"
            entry_price = sig["entry"]
            sl          = sig["sl"]
            tp          = sig["tp"]
            strat_id    = sig["strategy_id"]

            balance = await self._fetch_balance()
            if balance <= 0:
                return "blocked"

            qty, leverage = await self._calc_position_size(balance, entry_price, sl, sym)
            if qty <= 0:
                log.warning("[V7] %s %s qty=0 at entry=%.6g sl=%.6g balance=%.2f — skip",
                            self._short(sym), strat_id, entry_price, sl, balance)
                return "blocked"

            margin_used = qty * entry_price / max(leverage, 1)
            risk_usd    = abs(entry_price - sl) * qty

            # Set leverage on Bybit
            if not cfg.SIMULATION_MODE:
                try:
                    await self.client.set_leverage(sym, leverage)
                except Exception as exc:
                    log.warning("[V7] set_leverage failed %s: %s", self._short(sym), exc)

            # Place post-only limit order
            order = await self.client.place_post_only_limit_order(
                sym, side_str, qty, entry_price
            )

            if order is not None and order.get("id") and order["id"] != "SIM":
                # Real order placed — track it
                self._pending_limit[(sym, strat_id)] = {
                    "order_id":    order["id"],
                    "entry_price": entry_price,
                    "sl":          sl,
                    "tp":          tp,
                    "qty":         qty,
                    "leverage":    leverage,
                    "side_str":    side_str,
                    "direction":   direction,
                    "strat_id":    strat_id,
                    "margin_used": margin_used,
                    "risk_usd":    risk_usd,
                    "balance":     balance,
                    "cycles":      0,
                }
                log.info("[V7] LIMIT PLACED [%s] %s %s @ %.6g qty=%.6f lev=%d× margin=$%.2f",
                         strat_id, self._short(sym), direction, entry_price,
                         qty, leverage, margin_used)
                try:
                    await self.tg.send_limit_pending(
                        self.agent_id, sym, direction, entry_price,
                        sl, tp, qty, leverage, margin_used, strat_id,
                    )
                except Exception:
                    pass
                return "placed"

            elif order is not None and order.get("id") == "SIM":
                # Simulation mode — instant fill
                await self._register_opened_position(
                    sym, sig, entry_price, sl, tp, qty, leverage, margin_used, risk_usd, balance
                )
                return "placed"

            else:
                # Post-only rejected → try market order fallback
                log.info("[V7] Post-only rejected %s %s → market fallback",
                         self._short(sym), direction)
                order = await self.client.place_market_order(sym, side_str, qty)
                if order is not None:
                    try:
                        await self.client.place_sl_tp(sym, side_str, qty, sl, tp)
                    except Exception as exc:
                        log.warning("[V7] SL/TP placement failed %s: %s", self._short(sym), exc)
                    await self._register_opened_position(
                        sym, sig, entry_price, sl, tp, qty, leverage, margin_used, risk_usd, balance
                    )
                    return "placed"
                return "blocked"

    async def _check_limit_fill(
        self, sym: str, strat_id: str, order_info: dict
    ) -> str:
        """
        Check fill status of a pending limit order.
        Returns: "filled" | "expired" | "waiting" | "cancelled"
        """
        order_info["cycles"] = order_info.get("cycles", 0) + 1

        # Fast path: MasterAgent sync may have already registered the position
        positions = await self.state.get_agent_positions(self.agent_id)
        if sym in positions:
            log.info("[V7] Limit fill confirmed via sync %s [%s]", self._short(sym), strat_id)
            # Set SL/TP on the sync-registered position
            sl  = order_info.get("sl", 0.0)
            tp  = order_info.get("tp", 0.0)
            qty = order_info.get("qty", 0.0)
            side_str = order_info.get("side_str", "buy")
            if qty > 0 and (sl > 0 or tp > 0) and not cfg.SIMULATION_MODE:
                try:
                    await self.client.place_sl_tp(sym, side_str, qty, sl, tp)
                    # Patch state
                    pos_map = await self.state.get_agent_positions(self.agent_id)
                    ps = pos_map.get(sym)
                    if ps is not None:
                        ps["sl"] = sl
                        ps["tp"] = tp
                except Exception as exc:
                    log.warning("[V7] SL/TP set failed (sync-fill) %s: %s", self._short(sym), exc)
            return "filled"

        # Check timeout
        if order_info["cycles"] > LIMIT_MAX_CYCLES:
            try:
                await self.client.cancel_order(sym, order_info["order_id"])
                log.info("[V7] Limit expired+cancelled %s [%s]", self._short(sym), strat_id)
            except Exception:
                pass
            return "expired"

        # Fetch order status
        if cfg.SIMULATION_MODE:
            return "waiting"

        try:
            order = await self.client.fetch_order(order_info["order_id"], sym)
            status = order.get("status", "open")
            if status == "closed":
                fill_price = float(
                    order.get("average") or order.get("price") or order_info["entry_price"]
                )
                await self._finalize_limit_fill(sym, strat_id, order_info, fill_price)
                return "filled"
            elif status in ("canceled", "rejected", "expired"):
                log.info("[V7] Limit order %s for %s [%s]", status, self._short(sym), strat_id)
                return "cancelled"
            else:
                return "waiting"
        except Exception as exc:
            log.debug("[V7] fetch_order failed %s: %s", self._short(sym), exc)
            return "waiting"

    async def _finalize_limit_fill(
        self, sym: str, strat_id: str, order_info: dict, fill_price: float
    ) -> None:
        """Called when post-only limit confirmed filled."""
        sl          = order_info["sl"]
        tp          = order_info["tp"]
        qty         = order_info["qty"]
        leverage    = order_info["leverage"]
        side_str    = order_info["side_str"]
        direction   = order_info["direction"]
        margin_used = order_info["margin_used"]
        risk_usd    = order_info["risk_usd"]
        balance     = order_info["balance"]

        log.info("[V7] LIMIT FILLED [%s] %s %s fill=%.6g sl=%.6g tp=%.6g qty=%.6f lev=%d×",
                 strat_id, self._short(sym), direction, fill_price, sl, tp, qty, leverage)

        if not cfg.SIMULATION_MODE:
            try:
                await self.client.place_sl_tp(sym, side_str, qty, sl, tp)
            except Exception as exc:
                log.warning("[V7] SL/TP after fill failed %s: %s", self._short(sym), exc)

        # Build a minimal sig dict for register_opened_position
        sig = {"strategy_id": strat_id, "direction": direction}
        await self._register_opened_position(
            sym, sig, fill_price, sl, tp, qty, leverage, margin_used, risk_usd, balance
        )

    async def _register_opened_position(
        self, sym: str, sig: dict,
        entry: float, sl: float, tp: float,
        qty: float, leverage: int, margin_used: float, risk_usd: float, balance: float,
    ) -> None:
        """Register position in state and send TG notification."""
        strat_id  = sig["strategy_id"]
        direction = sig["direction"]

        await self.state.register_position(
            self.agent_id, sym, direction, entry, sl, tp, qty
        )
        try:
            pos = self.state._positions[self.agent_id][sym]
            pos["strategy_id"] = strat_id
            pos["leverage"]    = leverage
            pos["opened_at"]   = datetime.now(timezone.utc).isoformat()
        except Exception:
            pass

        await self.state.set_agent_equity(self.agent_id, balance)

        log.info("[V7] OPENED [%s] %s %s entry=%.6g sl=%.6g tp=%.6g qty=%.6f lev=%d× margin=$%.2f",
                 strat_id, self._short(sym), direction, entry, sl, tp, qty, leverage, margin_used)

        try:
            await self.tg.send_position_opened(
                self.agent_id, sym, direction, entry, sl, tp, qty, leverage, margin_used,
            )
        except Exception as exc:
            log.warning("[V7] TG open notification failed: %s", exc)

    # ── Gate checks ───────────────────────────────────────────────────────────

    async def _gate_checks(self, sym: str, sig: dict) -> str:
        """
        Pre-trade validation. Returns "ok" | "blocked" | "expired".
        All checks re-validated under _trade_lock in _try_place_limit.
        """
        direction = sig["direction"]

        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return "blocked"
        if await self.state.is_symbol_locked(sym):
            return "blocked"
        positions = await self.state.get_agent_positions(self.agent_id)
        if sym in positions:
            return "expired"
        if self._is_on_cooldown(sym):
            return "expired"
        if self._is_fomc_blackout():
            return "blocked"
        if not await self._can_open_position():
            return "blocked"

        # Correlation filter
        if not self._corr_allows_trade(sym, positions):
            log.info("[V7] %s corr filter blocked — correlated position already open",
                     self._short(sym))
            return "blocked"

        # Bear mode re-check (race: bear changed since signal was generated)
        bear = await self._is_bear_regime()
        if bear and direction == "SHORT":
            return "blocked"

        # Daily loss limit
        max_daily_loss_pct = getattr(cfg, "V7_MAX_DAILY_LOSS_PCT", 5.0)
        if not await self.risk.check_daily_loss_limit(
            self.agent_id, self.state, max_daily_loss_pct
        ):
            log.info("[V7] Daily loss limit reached — no new trades today")
            return "blocked"

        return "ok"
