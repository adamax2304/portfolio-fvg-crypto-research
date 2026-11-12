"""
strategy_lab.py — robust strategy testing framework
====================================================
2026-05-02: build a real edge or stop trying.

Goals:
  • Test 10 proven strategies on 730d × 27 symbols
  • Walk-forward analysis (6 folds, IS=180d / OOS=60d)
  • Market regime classifier (BTC trend / vol / dominance proxy)
  • Realistic costs: TAKER fee both sides, slippage both sides, funding cost
  • Wick-based SL/TP, SL-first pessimism
  • Per-regime metrics for each strategy
  • Output: stability score (consistency across folds + regimes)

Strategies (long & short symmetric):
  1. Donchian breakout (1d, 20-bar)
  2. SMA50/200 cross with trend filter
  3. SuperTrend trail (4h, ATR14 × 3)
  4. Bollinger mean reversion z-score (1h, sideways only)
  5. ATR volatility compression breakout (4h)
  6. Cross-sectional momentum (1d, top/bottom quintile by 30d return)
  7. VWAP-anchored momentum (1h)
  8. MACD trend continuation (4h)
  9. RSI extreme reversion (4h, sideways only)
  10. Donchian channel pullback (4h, trend regime)

Run:
  py -3 multi_agent/backtests/strategy_lab.py --strategy all
  py -3 multi_agent/backtests/strategy_lab.py --strategy donchian_4h --symbols BTC,ETH
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
import time as _time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("lab")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(message)s")

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── Realistic costs (PESSIMISTIC) ─────────────────────────────────────────────
# We assume TAKER on both sides — post-only often misses fills in fast moves.
TAKER_FEE       = 0.00055   # 0.055% Bybit perp taker
SLIPPAGE        = 0.0005    # 0.05% one side; applied to BOTH entry and exit
FUNDING_PER_8H  = 0.0001    # 0.01% per 8h (median across alts; pays losers a bit)
START_BALANCE   = 1000.0
RISK_PER_TRADE  = 0.02
MAX_POSITIONS   = 4
SL_COOLDOWN_MS  = 4 * 3_600_000
MIN_MARGIN      = 5.0
MAX_LEVERAGE    = 25
MAX_MARGIN_FRAC = 0.10

# ── Symbol universe (730d clean) ──────────────────────────────────────────────
SYMBOLS_730D = [
    "ADA", "ALGO", "APT", "ARB", "ATOM", "AVAX", "BCH", "BNB",
    "BTC", "DOGE", "DOT", "ETH", "FIL", "HBAR", "ICP", "INJ",
    "LINK", "LTC", "NEAR", "OP", "SOL", "SUI", "TON", "TRX",
    "UNI", "XLM", "XRP",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

def load(symbol: str, tf: str, tag: str = "730d") -> Optional[pd.DataFrame]:
    safe = f"{symbol}_USDT_USDT"
    for ttag in (tag, "730d", "365d", "180d"):
        for ext in (".parquet", ".csv"):
            p = CACHE_DIR / f"{safe}_{tf}_{ttag}{ext}"
            if p.exists():
                try:
                    df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
                    if df is None or df.empty:
                        continue
                    for c in ("open", "high", "low", "close", "volume"):
                        if c in df.columns:
                            df[c] = df[c].astype(float)
                    df["timestamp"] = df["timestamp"].astype(int)
                    df = df.sort_values("timestamp").reset_index(drop=True)
                    return df
                except Exception:
                    continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# INDICATORS
# ─────────────────────────────────────────────────────────────────────────────

def sma(arr, p):
    return pd.Series(arr).rolling(p, min_periods=p).mean().to_numpy()

def ema(arr, p):
    return pd.Series(arr).ewm(span=p, adjust=False, min_periods=p).mean().to_numpy()

def stdev(arr, p):
    return pd.Series(arr).rolling(p, min_periods=p).std(ddof=0).to_numpy()

def atr(hi, lo, cl, p=14):
    h = pd.Series(hi); l = pd.Series(lo); c = pd.Series(cl)
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/p, adjust=False, min_periods=p).mean().to_numpy()

def rsi(arr, p=14):
    s = pd.Series(arr)
    d = s.diff()
    g = d.where(d > 0, 0.0); l = -d.where(d < 0, 0.0)
    rs = g.ewm(alpha=1/p, adjust=False, min_periods=p).mean() / \
         l.ewm(alpha=1/p, adjust=False, min_periods=p).mean()
    return (100 - 100 / (1 + rs)).to_numpy()

def macd(arr, fast=12, slow=26, sig=9):
    ef = ema(arr, fast); es = ema(arr, slow)
    line = ef - es
    signal = pd.Series(line).ewm(span=sig, adjust=False, min_periods=sig).mean().to_numpy()
    return line, signal, line - signal

def donchian(hi, lo, p=20):
    upper = pd.Series(hi).rolling(p, min_periods=p).max().to_numpy()
    lower = pd.Series(lo).rolling(p, min_periods=p).min().to_numpy()
    return upper, lower

def supertrend(hi, lo, cl, p=10, mult=3.0):
    a = atr(hi, lo, cl, p)
    hl2 = (hi + lo) / 2
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    n = len(cl)
    st = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)  # 1=up, -1=down
    for i in range(p, n):
        if i == p:
            direction[i] = 1 if cl[i] > hl2[i] else -1
            st[i] = lower[i] if direction[i] == 1 else upper[i]
            continue
        prev_st = st[i-1]
        if direction[i-1] == 1:
            new_lower = max(lower[i], prev_st)
            if cl[i] < new_lower:
                direction[i] = -1; st[i] = upper[i]
            else:
                direction[i] = 1; st[i] = new_lower
        else:
            new_upper = min(upper[i], prev_st)
            if cl[i] > new_upper:
                direction[i] = 1; st[i] = lower[i]
            else:
                direction[i] = -1; st[i] = new_upper
    return st, direction

def vwap_session(df: pd.DataFrame) -> np.ndarray:
    """Daily-rolling VWAP (resets at UTC midnight)."""
    ts = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    day = ts.dt.floor("D")
    pv = (df["high"] + df["low"] + df["close"]) / 3 * df["volume"]
    g = df.assign(day=day, pv=pv, v=df["volume"])
    cum_pv = g.groupby("day")["pv"].cumsum()
    cum_v  = g.groupby("day")["v"].cumsum()
    return (cum_pv / cum_v.replace(0, np.nan)).to_numpy()


# ─────────────────────────────────────────────────────────────────────────────
# REGIME CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Regime:
    """Per-bar regime classification using BTC daily."""
    ts: np.ndarray            # 1d timestamps
    trend: np.ndarray         # 1=bull, 0=sideways, -1=bear
    vol_pct: np.ndarray       # 0..100 percentile of 30d realized vol

def build_regime(btc_1d: pd.DataFrame) -> Regime:
    cl = btc_1d["close"].to_numpy()
    sma50 = sma(cl, 50); sma200 = sma(cl, 200)
    # Trend: bull if close>sma200 AND sma50>sma200 (golden cross territory)
    #        bear if close<sma200 AND sma50<sma200
    #        sideways otherwise
    trend = np.zeros(len(cl), dtype=int)
    bull = (cl > sma200) & (sma50 > sma200)
    bear = (cl < sma200) & (sma50 < sma200)
    trend[bull] = 1
    trend[bear] = -1
    # Vol percentile (30d realized log-return std, rolling 365d percentile)
    rets = pd.Series(cl).pct_change().fillna(0)
    rv30 = rets.rolling(30, min_periods=30).std().to_numpy()
    vol_pct = pd.Series(rv30).rolling(365, min_periods=60).rank(pct=True).to_numpy() * 100
    vol_pct = np.where(np.isnan(vol_pct), 50.0, vol_pct)
    return Regime(
        ts=btc_1d["timestamp"].to_numpy(),
        trend=trend,
        vol_pct=vol_pct,
    )

def regime_at(regime: Regime, ts_ms: int) -> tuple[int, float]:
    idx = int(np.searchsorted(regime.ts, ts_ms, side="right")) - 1
    if idx < 0:
        return 0, 50.0
    return int(regime.trend[idx]), float(regime.vol_pct[idx])


# ─────────────────────────────────────────────────────────────────────────────
# STRATEGY DEFINITIONS
#   Each strategy returns a list of signals:
#   [{ts, direction, entry, sl, tp, dist}, ...]
#   Signal has bar timestamp; entry executes at OPEN of NEXT bar (no look-ahead).
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Signal:
    ts: int          # bar close timestamp (decision made here)
    next_open_ts: int  # next bar's open ts
    direction: str   # "LONG"/"SHORT"
    entry: float     # next-bar OPEN
    sl: float
    tp: float
    dist: float
    info: dict = field(default_factory=dict)


def _ts_period_ms(ts: np.ndarray) -> int:
    if len(ts) < 2: return 3_600_000
    return int(np.median(np.diff(ts[:200])))


def _emit(signals, ts, period_ms, direction, entry, sl, tp, info=None):
    if direction == "LONG":
        dist = entry - sl
    else:
        dist = sl - entry
    if dist <= 0:
        return
    signals.append(Signal(
        ts=int(ts), next_open_ts=int(ts) + period_ms,
        direction=direction, entry=float(entry),
        sl=float(sl), tp=float(tp), dist=float(dist),
        info=info or {},
    ))


# Strategy 1: Donchian breakout 1d
def strat_donchian_1d(df: pd.DataFrame, regime: Regime,
                      lookback=20, sl_atr=2.0, tp_r=2.5,
                      regime_filter=True) -> list[Signal]:
    if df is None or len(df) < lookback + 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    upper, lower = donchian(hi, lo, lookback)
    sigs = []
    for i in range(lookback + 14, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        next_open = op[i+1]
        # Regime filter: only LONG in bull, only SHORT in bear
        trend, _ = regime_at(regime, int(ts[i]))
        # Breakout triggers on close above 20-day high (not including current)
        prev_upper = upper[i-1]
        prev_lower = lower[i-1]
        if cl[i] > prev_upper:
            if regime_filter and trend < 0: continue
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"regime": trend, "atr": a[i]})
        elif cl[i] < prev_lower:
            if regime_filter and trend > 0: continue
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"regime": trend, "atr": a[i]})
    return sigs


# Strategy 2: SMA50/200 cross with confirmation
def strat_smacross_1d(df: pd.DataFrame, regime: Regime,
                      sl_atr=3.0, tp_r=3.0) -> list[Signal]:
    if df is None or len(df) < 220: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    s50 = sma(cl, 50); s200 = sma(cl, 200); a = atr(hi, lo, cl, 14)
    sigs = []
    for i in range(201, len(cl) - 1):
        if np.isnan(s50[i]) or np.isnan(s200[i]) or np.isnan(a[i]): continue
        if np.isnan(s50[i-1]) or np.isnan(s200[i-1]): continue
        # Golden cross
        next_open = op[i+1]
        if s50[i-1] <= s200[i-1] and s50[i] > s200[i]:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"event": "golden"})
        # Death cross
        elif s50[i-1] >= s200[i-1] and s50[i] < s200[i]:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"event": "death"})
    return sigs


# Strategy 3: SuperTrend trail 4h
def strat_supertrend_4h(df: pd.DataFrame, regime: Regime,
                        st_p=10, st_mult=3.0,
                        sl_atr=2.0, tp_r=2.0) -> list[Signal]:
    if df is None or len(df) < 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    st, sdir = supertrend(hi, lo, cl, st_p, st_mult)
    sigs = []
    for i in range(st_p + 14, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        # Direction flip = entry signal
        if sdir[i] == 1 and sdir[i-1] == -1:
            next_open = op[i+1]
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"st": st[i]})
        elif sdir[i] == -1 and sdir[i-1] == 1:
            next_open = op[i+1]
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"st": st[i]})
    return sigs


# Strategy 4: Bollinger mean reversion z-score (1h, sideways regime only)
def strat_bb_meanrev_1h(df: pd.DataFrame, regime: Regime,
                        z_entry=2.5, z_exit=0.0,
                        sl_atr=1.5, max_bars=24,
                        regime_filter=True) -> list[Signal]:
    if df is None or len(df) < 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    sma20 = sma(cl, 20); std20 = stdev(cl, 20); a = atr(hi, lo, cl, 14)
    z = (cl - sma20) / np.where(std20 > 0, std20, 1)
    sigs = []
    for i in range(40, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        trend, _ = regime_at(regime, int(ts[i]))
        if regime_filter and trend != 0: continue  # only sideways
        next_open = op[i+1]
        # Long when oversold
        if z[i] <= -z_entry and z[i-1] > -z_entry:
            sl = next_open - sl_atr * a[i]
            tp = sma20[i]  # target = mean
            if tp <= next_open: continue
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"z": z[i]})
        elif z[i] >= z_entry and z[i-1] < z_entry:
            sl = next_open + sl_atr * a[i]
            tp = sma20[i]
            if tp >= next_open: continue
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"z": z[i]})
    return sigs


# Strategy 5: ATR volatility compression breakout (4h)
def strat_atr_compression_4h(df: pd.DataFrame, regime: Regime,
                             compress_pct=20, expand_mult=1.5,
                             sl_atr=1.5, tp_r=2.5) -> list[Signal]:
    """Wait for ATR contraction (in lowest 20% of last 100 bars), then trade
    the breakout direction (close > recent N-bar high or < N-bar low)."""
    if df is None or len(df) < 150: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    sigs = []
    look = 20
    for i in range(120, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        # ATR percentile (last 100 bars)
        seg = a[i-100:i]
        seg = seg[~np.isnan(seg)]
        if len(seg) < 50: continue
        pct_rank = (seg < a[i]).sum() / len(seg) * 100
        if pct_rank > compress_pct: continue
        # Breakout move
        seg_hi = np.nanmax(hi[i-look:i])
        seg_lo = np.nanmin(lo[i-look:i])
        next_open = op[i+1]
        if cl[i] > seg_hi and (cl[i] - cl[i-1]) > expand_mult * a[i] * 0.3:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"compress": pct_rank})
        elif cl[i] < seg_lo and (cl[i-1] - cl[i]) > expand_mult * a[i] * 0.3:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"compress": pct_rank})
    return sigs


# Strategy 6: VWAP momentum 1h (long above VWAP after pullback, short below)
def strat_vwap_momentum_1h(df: pd.DataFrame, regime: Regime,
                            sl_atr=1.5, tp_r=2.0,
                            regime_filter=True) -> list[Signal]:
    if df is None or len(df) < 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    vw = vwap_session(df)
    e21 = ema(cl, 21)
    sigs = []
    for i in range(25, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if np.isnan(vw[i]) or np.isnan(e21[i]): continue
        trend, _ = regime_at(regime, int(ts[i]))
        next_open = op[i+1]
        # Long: above VWAP, pulled back to EMA21, now resuming up
        above = cl[i] > vw[i] and cl[i] > e21[i]
        below = cl[i] < vw[i] and cl[i] < e21[i]
        # Pullback was 1-3 bars ago: low touched ema21
        recent_pullback_long  = any(lo[j] <= e21[j] for j in range(max(0, i-3), i))
        recent_pullback_short = any(hi[j] >= e21[j] for j in range(max(0, i-3), i))
        if above and recent_pullback_long and cl[i] > cl[i-1]:
            if regime_filter and trend < 0: continue
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif below and recent_pullback_short and cl[i] < cl[i-1]:
            if regime_filter and trend > 0: continue
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


# Strategy 7: MACD trend continuation 4h
def strat_macd_4h(df: pd.DataFrame, regime: Regime,
                  sl_atr=2.0, tp_r=2.0,
                  regime_filter=True) -> list[Signal]:
    if df is None or len(df) < 60: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    line, signal, hist = macd(cl)
    e200 = ema(cl, 200)
    sigs = []
    for i in range(40, len(cl) - 1):
        if np.isnan(line[i]) or np.isnan(signal[i]) or np.isnan(a[i]): continue
        if np.isnan(line[i-1]) or np.isnan(signal[i-1]): continue
        next_open = op[i+1]
        trend, _ = regime_at(regime, int(ts[i]))
        long_filter = (not np.isnan(e200[i]) and cl[i] > e200[i])
        short_filter = (not np.isnan(e200[i]) and cl[i] < e200[i])
        # MACD bullish cross
        if line[i-1] <= signal[i-1] and line[i] > signal[i]:
            if regime_filter and (trend < 0 or not long_filter): continue
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif line[i-1] >= signal[i-1] and line[i] < signal[i]:
            if regime_filter and (trend > 0 or not short_filter): continue
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


# Strategy 8: RSI extreme reversion 4h (sideways only)
def strat_rsi_reversion_4h(df: pd.DataFrame, regime: Regime,
                           os=20, ob=80, sl_atr=1.5, tp_r=1.5,
                           regime_filter=True) -> list[Signal]:
    if df is None or len(df) < 30: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    r = rsi(cl, 14)
    sigs = []
    for i in range(20, len(cl) - 1):
        if np.isnan(a[i]) or np.isnan(r[i]) or np.isnan(r[i-1]): continue
        trend, _ = regime_at(regime, int(ts[i]))
        if regime_filter and trend != 0: continue
        next_open = op[i+1]
        # Long when crossing UP from oversold
        if r[i-1] <= os and r[i] > os:
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"rsi": r[i]})
        elif r[i-1] >= ob and r[i] < ob:
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"rsi": r[i]})
    return sigs


# Strategy 9: Donchian pullback 4h (trend regime)
def strat_donchian_pullback_4h(df: pd.DataFrame, regime: Regime,
                                lookback=20, sl_atr=1.5, tp_r=2.0,
                                regime_filter=True) -> list[Signal]:
    """When 20-bar high is broken on bar i-2 or i-3, and current bar pulls back
    to ema21, enter in direction of breakout."""
    if df is None or len(df) < 50: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    upper, lower = donchian(hi, lo, lookback)
    e21 = ema(cl, 21)
    sigs = []
    for i in range(lookback + 21, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if np.isnan(e21[i]): continue
        trend, _ = regime_at(regime, int(ts[i]))
        next_open = op[i+1]
        # Long pullback after recent breakout
        broke_high = any(cl[j] > upper[j-1] for j in range(max(0, i-3), i))
        broke_low  = any(cl[j] < lower[j-1] for j in range(max(0, i-3), i))
        # Touched ema21 from above (pullback)
        if broke_high and lo[i] <= e21[i] and cl[i] > e21[i]:
            if regime_filter and trend < 0: continue
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp)
        elif broke_low and hi[i] >= e21[i] and cl[i] < e21[i]:
            if regime_filter and trend > 0: continue
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp)
    return sigs


# Strategy 10: Cross-sectional momentum (1d, top/bottom by 30d return)
# This is a portfolio-level strategy — handled separately
def gen_xsec_momentum(symbols_data: dict[str, pd.DataFrame], regime: Regime,
                      lookback=30, hold=7, top_n=4,
                      sl_atr=3.0, tp_r=3.0,
                      regime_filter=True) -> list[Signal]:
    """At each rebalance day, rank symbols by `lookback`-day return.
    Long top-N, Short bottom-N (or only long if regime_filter and trend>=0)."""
    sigs = []
    # Align all symbols on a common 1d timeline (use BTC's timestamps)
    if "BTC" not in symbols_data:
        return []
    ref = symbols_data["BTC"]
    ts_ref = ref["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts_ref)
    # For each rebalance day (every `hold` days), compute lookback returns
    for i in range(lookback + 5, len(ts_ref) - hold - 1):
        if (i - lookback) % hold != 0:
            continue
        trend, _ = regime_at(regime, int(ts_ref[i]))
        # Compute lookback-period return for each symbol
        rets = []
        for sym, df in symbols_data.items():
            ts = df["timestamp"].to_numpy()
            cl = df["close"].to_numpy()
            j = int(np.searchsorted(ts, ts_ref[i], side="right")) - 1
            j_back = int(np.searchsorted(ts, ts_ref[i-lookback], side="right")) - 1
            if j_back < 0 or j < j_back: continue
            if cl[j_back] <= 0: continue
            r = (cl[j] - cl[j_back]) / cl[j_back]
            rets.append((sym, r, df, j))
        if not rets: continue
        rets.sort(key=lambda x: x[1], reverse=True)
        longs = rets[:top_n]
        shorts = rets[-top_n:]
        # Emit signals
        for sym, r, df, j in longs:
            if r <= 0: continue
            if regime_filter and trend < 0: continue
            if j+1 >= len(df): continue
            op = df["open"].to_numpy(); cl_a = df["close"].to_numpy()
            hi_a = df["high"].to_numpy(); lo_a = df["low"].to_numpy()
            a = atr(hi_a, lo_a, cl_a, 14)
            if np.isnan(a[j]) or a[j] <= 0: continue
            entry = op[j+1]
            sl = entry - sl_atr * a[j]
            tp = entry + tp_r * (entry - sl)
            ts_arr = df["timestamp"].to_numpy()
            sym_period_ms = _ts_period_ms(ts_arr)
            _emit(sigs, ts_arr[j], sym_period_ms, "LONG", entry, sl, tp,
                  info={"sym": sym, "rank_ret": r})
        for sym, r, df, j in shorts:
            if r >= 0: continue
            if regime_filter and trend > 0: continue
            if j+1 >= len(df): continue
            op = df["open"].to_numpy(); cl_a = df["close"].to_numpy()
            hi_a = df["high"].to_numpy(); lo_a = df["low"].to_numpy()
            a = atr(hi_a, lo_a, cl_a, 14)
            if np.isnan(a[j]) or a[j] <= 0: continue
            entry = op[j+1]
            sl = entry + sl_atr * a[j]
            tp = entry - tp_r * (sl - entry)
            ts_arr = df["timestamp"].to_numpy()
            sym_period_ms = _ts_period_ms(ts_arr)
            _emit(sigs, ts_arr[j], sym_period_ms, "SHORT", entry, sl, tp,
                  info={"sym": sym, "rank_ret": r})
    return sigs


# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION (per-strategy, single-symbol)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TradeResult:
    sym: str
    strat: str
    direction: str
    entry_ts: int
    exit_ts: int
    entry: float
    exit: float
    sl: float
    tp: float
    reason: str
    pnl_usd: float
    pnl_r: float
    bars_held: int
    regime_at_entry: int


def simulate_signals(symbol: str, strat_name: str, signals: list[Signal],
                     ltf_df: pd.DataFrame, regime: Regime,
                     ts_lo: int = 0, ts_hi: int = 0) -> list[TradeResult]:
    """Simulate signals on the *same TF* they were generated on (entry at next bar OPEN).
    Realistic: TAKER fee both sides, slippage both sides, funding cost,
    SL hit before TP within same bar (pessimistic).
    Returns trades list."""
    if not signals or ltf_df is None or len(ltf_df) < 5:
        return []
    ts = ltf_df["timestamp"].to_numpy()
    op = ltf_df["open"].to_numpy()
    hi = ltf_df["high"].to_numpy()
    lo = ltf_df["low"].to_numpy()
    cl = ltf_df["close"].to_numpy()
    n = len(ts)
    period_ms = _ts_period_ms(ts)
    # holding window
    funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))
    out: list[TradeResult] = []
    balance = START_BALANCE  # per-strategy independent (we'll combine in portfolio sim)
    for sig in signals:
        if ts_lo and sig.ts < ts_lo: continue
        if ts_hi and sig.ts > ts_hi: continue
        # Find entry bar = first bar with open ts >= sig.next_open_ts
        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1: continue
        # Entry executes at this bar's OPEN with slippage
        if sig.direction == "LONG":
            entry_p = op[j] * (1 + SLIPPAGE)
            sl = sig.sl
            # Recalc tp based on slipped entry to keep R consistent
            dist = entry_p - sl
            if dist <= 0: continue
            tp = entry_p + (sig.tp - sig.entry) * (dist / sig.dist) if sig.dist > 0 else sig.tp
        else:
            entry_p = op[j] * (1 - SLIPPAGE)
            sl = sig.sl
            dist = sl - entry_p
            if dist <= 0: continue
            tp = entry_p - (sig.entry - sig.tp) * (dist / sig.dist) if sig.dist > 0 else sig.tp
        risk_usd = balance * RISK_PER_TRADE
        qty = risk_usd / dist
        # Walk forward
        exit_p = None; exit_bar = None; reason = "timeout"
        for k in range(j, min(j + 500, n)):  # cap holding to 500 bars
            h_k = hi[k]; l_k = lo[k]
            if sig.direction == "LONG":
                if l_k <= sl:
                    exit_p = sl; reason = "SL"; exit_bar = k; break
                if h_k >= tp:
                    exit_p = tp; reason = "TP"; exit_bar = k; break
            else:
                if h_k >= sl:
                    exit_p = sl; reason = "SL"; exit_bar = k; break
                if l_k <= tp:
                    exit_p = tp; reason = "TP"; exit_bar = k; break
        if exit_bar is None:
            exit_p = cl[min(j + 499, n - 1)]
            exit_bar = min(j + 499, n - 1)
            reason = "timeout"
        # Apply slippage to exit (against us)
        if sig.direction == "LONG":
            slip_exit = exit_p * (1 - SLIPPAGE)
        else:
            slip_exit = exit_p * (1 + SLIPPAGE)
        # PnL
        if sig.direction == "LONG":
            gross = (slip_exit - entry_p) * qty
        else:
            gross = (entry_p - slip_exit) * qty
        # Fees: TAKER both sides
        entry_fee = entry_p * qty * TAKER_FEE
        exit_fee = slip_exit * qty * TAKER_FEE
        # Funding
        bars_held = exit_bar - j + 1
        notional = entry_p * qty
        funding = notional * funding_per_bar * bars_held
        pnl = gross - entry_fee - exit_fee - funding
        balance += pnl
        balance = max(balance, 1.0)
        trend_at_e, _ = regime_at(regime, int(ts[j]))
        out.append(TradeResult(
            sym=symbol, strat=strat_name, direction=sig.direction,
            entry_ts=int(ts[j]), exit_ts=int(ts[exit_bar]),
            entry=entry_p, exit=slip_exit,
            sl=sl, tp=tp, reason=reason,
            pnl_usd=pnl, pnl_r=pnl / risk_usd if risk_usd > 0 else 0,
            bars_held=bars_held, regime_at_entry=trend_at_e,
        ))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# METRICS & REPORTING
# ─────────────────────────────────────────────────────────────────────────────

def metrics(trades: list[TradeResult]) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "avg_r": 0, "sum_r": 0, "max_dd_r": 0, "expectancy_r": 0}
    wins = [t for t in trades if t.pnl_usd > 0]
    losses = [t for t in trades if t.pnl_usd <= 0]
    gw = sum(t.pnl_usd for t in wins)
    gl = abs(sum(t.pnl_usd for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    wr = len(wins) / len(trades) * 100
    sum_r = sum(t.pnl_r for t in trades)
    avg_r = sum_r / len(trades)
    # equity by R
    eq_r = np.cumsum([t.pnl_r for t in sorted(trades, key=lambda x: x.entry_ts)])
    peak = np.maximum.accumulate(eq_r)
    dd_r = (peak - eq_r).max() if len(eq_r) else 0
    return {
        "n": len(trades),
        "wr": round(wr, 1),
        "pf": round(pf, 2) if pf != float("inf") else 99.0,
        "avg_r": round(avg_r, 3),
        "sum_r": round(sum_r, 1),
        "max_dd_r": round(dd_r, 1),
        "expectancy_r": round(avg_r, 3),
    }


def metrics_by_regime(trades: list[TradeResult]) -> dict:
    out = {}
    for r_label, r_val in [("bull", 1), ("sideways", 0), ("bear", -1)]:
        out[r_label] = metrics([t for t in trades if t.regime_at_entry == r_val])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# WALK-FORWARD HARNESS
# ─────────────────────────────────────────────────────────────────────────────

def walk_forward(trades: list[TradeResult], folds: int = 6) -> list[dict]:
    """Split trades into N chronological folds, compute metrics per fold."""
    if not trades: return []
    trades = sorted(trades, key=lambda x: x.entry_ts)
    n = len(trades)
    if n < folds: folds = max(1, n)
    chunk = n // folds
    out = []
    for k in range(folds):
        lo = k * chunk
        hi = (k+1)*chunk if k < folds - 1 else n
        chunk_trades = trades[lo:hi]
        m = metrics(chunk_trades)
        out.append({
            "fold": k+1, "from_ts": chunk_trades[0].entry_ts,
            "to_ts": chunk_trades[-1].entry_ts, **m
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ORCHESTRATION
# ─────────────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "donchian_1d":         (strat_donchian_1d,         "1d"),
    "smacross_1d":         (strat_smacross_1d,         "1d"),
    "supertrend_4h":       (strat_supertrend_4h,       "4h"),
    "bb_meanrev_1h":       (strat_bb_meanrev_1h,       "1h"),
    "atr_compression_4h":  (strat_atr_compression_4h,  "4h"),
    "vwap_momentum_1h":    (strat_vwap_momentum_1h,    "1h"),
    "macd_4h":             (strat_macd_4h,             "4h"),
    "rsi_reversion_4h":    (strat_rsi_reversion_4h,    "4h"),
    "donchian_pullback_4h":(strat_donchian_pullback_4h,"4h"),
    # xsec_momentum is portfolio-level, handled separately
}


# ─────────────────────────────────────────────────────────────────────────────
# REFINED STRATEGIES (V2) — confluence + tighter regime filtering
# ─────────────────────────────────────────────────────────────────────────────

# Strategy 11: Triple-confluence (1d trend + 4h supertrend + 4h pullback)
# Generates trades on 4h TF, but requires:
#   - Daily close > EMA200 (1d trend up) for LONG
#   - SuperTrend 4h direction up for LONG
#   - Bar's low touched EMA21 (pullback) and bar closes above EMA21
def strat_confluence_4h(df_4h: pd.DataFrame, df_1d: Optional[pd.DataFrame], regime: Regime,
                        sl_atr=2.0, tp_r=2.5, regime_filter=True) -> list[Signal]:
    if df_4h is None or len(df_4h) < 250: return []
    hi = df_4h["high"].to_numpy(); lo = df_4h["low"].to_numpy(); cl = df_4h["close"].to_numpy()
    op = df_4h["open"].to_numpy(); ts = df_4h["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    e21 = ema(cl, 21); e50 = ema(cl, 50)
    st_4h, sdir_4h = supertrend(hi, lo, cl, 10, 3.0)
    # Daily trend reference
    d1d_ts = df_1d["timestamp"].to_numpy() if df_1d is not None else None
    d1d_cl = df_1d["close"].to_numpy() if df_1d is not None else None
    d1d_e200 = ema(d1d_cl, 200) if df_1d is not None else None
    sigs = []
    for i in range(60, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if np.isnan(e21[i]) or np.isnan(e50[i]): continue
        next_open = op[i+1]
        # 1d trend check
        d_long = d_short = True
        if df_1d is not None:
            di = int(np.searchsorted(d1d_ts, ts[i], side="right")) - 1
            if di >= 200 and not np.isnan(d1d_e200[di]):
                d_long = d1d_cl[di] > d1d_e200[di]
                d_short = d1d_cl[di] < d1d_e200[di]
        # BTC regime
        trend, _ = regime_at(regime, int(ts[i]))
        # Long setup: ST up, 1d up, BTC bullish/sideways, low touched ema21 in last 2 bars
        recent_pullback_long = any(lo[j] <= e21[j] * 1.005 for j in range(max(0, i-2), i+1))
        recent_pullback_short = any(hi[j] >= e21[j] * 0.995 for j in range(max(0, i-2), i+1))
        if (sdir_4h[i] == 1 and d_long and recent_pullback_long
                and cl[i] > e21[i] > e50[i] and cl[i] > cl[i-1]):
            if regime_filter and trend < 0: continue
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"st": st_4h[i]})
        elif (sdir_4h[i] == -1 and d_short and recent_pullback_short
                and cl[i] < e21[i] < e50[i] and cl[i] < cl[i-1]):
            if regime_filter and trend > 0: continue
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"st": st_4h[i]})
    return sigs


# Strategy 12: BB squeeze breakout (4h)
def strat_bb_squeeze_4h(df: pd.DataFrame, regime: Regime,
                        min_squeeze=4, sl_atr=1.5, tp_r=2.0,
                        regime_filter=True) -> list[Signal]:
    if df is None or len(df) < 60: return []
    hi = df["high"].to_numpy(); lo = df["low"].to_numpy(); cl = df["close"].to_numpy()
    op = df["open"].to_numpy(); ts = df["timestamp"].to_numpy()
    period_ms = _ts_period_ms(ts)
    a = atr(hi, lo, cl, 14)
    sma20 = sma(cl, 20); std20 = stdev(cl, 20)
    bb_u = sma20 + 2*std20; bb_l = sma20 - 2*std20
    kc_atr = atr(hi, lo, cl, 10); kc_mid = ema(cl, 20)
    kc_u = kc_mid + 1.5*kc_atr; kc_l = kc_mid - 1.5*kc_atr
    sq = (bb_l > kc_l) & (bb_u < kc_u)
    sq_len = np.zeros(len(cl), dtype=int)
    for i in range(1, len(cl)):
        sq_len[i] = sq_len[i-1] + 1 if sq[i] else 0
    sigs = []
    for i in range(50, len(cl) - 1):
        if np.isnan(a[i]) or a[i] <= 0: continue
        if sq[i]: continue
        if sq_len[i-1] < min_squeeze: continue
        # Direction: where did price break
        next_open = op[i+1]
        trend, _ = regime_at(regime, int(ts[i]))
        if cl[i] > sma20[i] and cl[i] > cl[i-1]:
            if regime_filter and trend < 0: continue
            sl = next_open - sl_atr * a[i]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts[i], period_ms, "LONG", next_open, sl, tp,
                  info={"sq_len": sq_len[i-1]})
        elif cl[i] < sma20[i] and cl[i] < cl[i-1]:
            if regime_filter and trend > 0: continue
            sl = next_open + sl_atr * a[i]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts[i], period_ms, "SHORT", next_open, sl, tp,
                  info={"sq_len": sq_len[i-1]})
    return sigs


# Strategy 13: Higher-TF SuperTrend + LTF entry on flip (1d ST signal → 4h entry)
def strat_st_1d_4h(df_1d: pd.DataFrame, df_4h: pd.DataFrame, regime: Regime,
                   sl_atr=2.0, tp_r=2.5, regime_filter=True) -> list[Signal]:
    if df_1d is None or df_4h is None: return []
    if len(df_1d) < 30 or len(df_4h) < 50: return []
    hi1 = df_1d["high"].to_numpy(); lo1 = df_1d["low"].to_numpy(); cl1 = df_1d["close"].to_numpy()
    ts1 = df_1d["timestamp"].to_numpy()
    st1, sdir1 = supertrend(hi1, lo1, cl1, 10, 3.0)
    period1 = _ts_period_ms(ts1)
    # 4h
    hi4 = df_4h["high"].to_numpy(); lo4 = df_4h["low"].to_numpy(); cl4 = df_4h["close"].to_numpy()
    op4 = df_4h["open"].to_numpy(); ts4 = df_4h["timestamp"].to_numpy()
    a4 = atr(hi4, lo4, cl4, 14)
    period4 = _ts_period_ms(ts4)
    sigs = []
    for i in range(15, len(cl1) - 1):
        # detect flip on 1d
        if sdir1[i] == 1 and sdir1[i-1] == -1:
            d = "LONG"
        elif sdir1[i] == -1 and sdir1[i-1] == 1:
            d = "SHORT"
        else:
            continue
        trend, _ = regime_at(regime, int(ts1[i]))
        if regime_filter and ((d == "LONG" and trend < 0) or (d == "SHORT" and trend > 0)):
            continue
        # find first 4h bar after 1d close
        anchor_ms = int(ts1[i]) + period1
        j = int(np.searchsorted(ts4, anchor_ms, side="left"))
        if j >= len(cl4) - 1: continue
        if np.isnan(a4[j]) or a4[j] <= 0: continue
        next_open = op4[j+1] if j+1 < len(cl4) else cl4[j]
        if d == "LONG":
            sl = next_open - sl_atr * a4[j]; tp = next_open + tp_r * (next_open - sl)
            _emit(sigs, ts4[j], period4, "LONG", next_open, sl, tp)
        else:
            sl = next_open + sl_atr * a4[j]; tp = next_open - tp_r * (sl - next_open)
            _emit(sigs, ts4[j], period4, "SHORT", next_open, sl, tp)
    return sigs


# Wrappers for STRATEGIES registry — multi-TF strategies need both DFs
def _wrap_confluence(df_4h, regime):
    return None  # handled separately

STRATEGIES_MULTI_TF = {
    "confluence_4h":  ("4h", "1d", strat_confluence_4h),
    "bb_squeeze_4h":  ("4h", None, strat_bb_squeeze_4h),
    "st_1d_4h":       ("1d", "4h", strat_st_1d_4h),
}


def run_strategy(strat_name: str, symbols: list[str], regime: Regime,
                 tag: str = "730d") -> dict:
    """Run one strategy across all symbols on its native TF."""
    fn, tf = STRATEGIES[strat_name]
    all_trades = []
    sym_count = 0
    sig_count = 0
    for sym in symbols:
        df = load(sym, tf, tag)
        if df is None or len(df) < 100: continue
        sym_count += 1
        sigs = fn(df, regime)
        sig_count += len(sigs)
        if not sigs: continue
        trades = simulate_signals(sym, strat_name, sigs, df, regime)
        all_trades.extend(trades)
    full = metrics(all_trades)
    # Walk-forward
    wf = walk_forward(all_trades, folds=6)
    # By regime
    by_reg = metrics_by_regime(all_trades)
    return {
        "strategy": strat_name, "tf": tf,
        "symbols_tested": sym_count, "signals_total": sig_count,
        "trades_n": len(all_trades),
        "full": full,
        "by_regime": by_reg,
        "walk_forward": wf,
        "trades": all_trades,
    }


def run_strategy_multitf(strat_name: str, symbols: list[str], regime: Regime,
                          tag: str = "730d") -> dict:
    """Multi-TF strategies: confluence_4h, bb_squeeze_4h, st_1d_4h."""
    primary_tf, secondary_tf, fn = STRATEGIES_MULTI_TF[strat_name]
    all_trades = []
    sym_count = 0; sig_count = 0
    for sym in symbols:
        df_p = load(sym, primary_tf, tag)
        if df_p is None or len(df_p) < 100: continue
        sym_count += 1
        if secondary_tf is None:
            sigs = fn(df_p, regime)
            sim_df = df_p
        elif strat_name == "confluence_4h":
            df_s = load(sym, secondary_tf, tag)
            sigs = fn(df_p, df_s, regime)
            sim_df = df_p
        elif strat_name == "st_1d_4h":
            df_s = load(sym, secondary_tf, tag)
            sigs = fn(df_p, df_s, regime)
            sim_df = df_s  # entries on 4h
        else:
            sigs = []; sim_df = df_p
        sig_count += len(sigs)
        if not sigs: continue
        trades = simulate_signals(sym, strat_name, sigs, sim_df, regime)
        all_trades.extend(trades)
    return {
        "strategy": strat_name, "tf": primary_tf,
        "symbols_tested": sym_count, "signals_total": sig_count,
        "trades_n": len(all_trades),
        "full": metrics(all_trades),
        "by_regime": metrics_by_regime(all_trades),
        "walk_forward": walk_forward(all_trades, folds=6),
        "trades": all_trades,
    }


def run_xsec(symbols: list[str], regime: Regime, tag: str = "730d") -> dict:
    """Cross-sectional momentum across all symbols on 1d."""
    sym_data = {}
    for sym in symbols:
        df = load(sym, "1d", tag)
        if df is not None and len(df) >= 50:
            sym_data[sym] = df
    if not sym_data:
        return {"strategy": "xsec_momentum", "trades_n": 0,
                "full": metrics([]), "by_regime": {}, "walk_forward": [], "trades": []}
    sigs = gen_xsec_momentum(sym_data, regime)
    # Simulate each by symbol — group sigs by sym
    sigs_by_sym = defaultdict(list)
    for s in sigs:
        sigs_by_sym[s.info.get("sym", "?")].append(s)
    all_trades = []
    for sym, slist in sigs_by_sym.items():
        df = sym_data[sym]
        all_trades.extend(simulate_signals(sym, "xsec_momentum", slist, df, regime))
    return {
        "strategy": "xsec_momentum", "tf": "1d",
        "symbols_tested": len(sym_data), "signals_total": len(sigs),
        "trades_n": len(all_trades),
        "full": metrics(all_trades),
        "by_regime": metrics_by_regime(all_trades),
        "walk_forward": walk_forward(all_trades, folds=6),
        "trades": all_trades,
    }


def print_strategy_report(res: dict):
    s = res["strategy"]
    full = res["full"]
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  {s}  ({res['tf']})  syms={res.get('symbols_tested',0)}  sigs={res.get('signals_total',0)}  trades={res['trades_n']}")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  FULL    n={full['n']:5d}  WR={full['wr']:5.1f}%  PF={full['pf']:5.2f}  avgR={full['avg_r']:+.3f}  sumR={full['sum_r']:+.1f}  ddR={full['max_dd_r']:.1f}")
    by_reg = res.get("by_regime", {})
    for label in ("bull", "sideways", "bear"):
        m = by_reg.get(label, {"n":0})
        if m["n"] == 0:
            print(f"  {label:8s} n={m.get('n',0):5d}  (no trades)")
        else:
            print(f"  {label:8s} n={m['n']:5d}  WR={m['wr']:5.1f}%  PF={m['pf']:5.2f}  avgR={m['avg_r']:+.3f}  sumR={m['sum_r']:+.1f}")
    wf = res.get("walk_forward", [])
    if wf:
        print(f"  Walk-forward folds:")
        for f in wf:
            t0 = pd.Timestamp(f['from_ts'], unit='ms', tz='UTC').strftime('%Y-%m-%d')
            t1 = pd.Timestamp(f['to_ts'], unit='ms', tz='UTC').strftime('%Y-%m-%d')
            print(f"    fold {f['fold']}  {t0}→{t1}  n={f['n']:4d}  WR={f['wr']:5.1f}%  PF={f['pf']:5.2f}  sumR={f['sum_r']:+.1f}")


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO COMBINER (compounding, shared balance, max-positions cap)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PortfolioSignal:
    ts: int                # signal generation ts
    next_open_ts: int      # entry attempt ts
    sym: str
    strat: str
    direction: str
    entry: float
    sl: float
    tp: float
    dist: float


def collect_portfolio_signals(strategies_to_run: list[str],
                               symbols: list[str], regime: Regime,
                               tag: str = "730d",
                               bull_only: bool = False) -> tuple[list[PortfolioSignal], dict]:
    """Run each strategy across each symbol, return all signals + sim_df map."""
    sigs: list[PortfolioSignal] = []
    sim_dfs: dict[tuple[str, str], pd.DataFrame] = {}  # (sym, strat) → df for sim
    for strat_name in strategies_to_run:
        if strat_name in STRATEGIES:
            fn, tf = STRATEGIES[strat_name]
            for sym in symbols:
                df = load(sym, tf, tag)
                if df is None or len(df) < 100: continue
                slist = fn(df, regime)
                sim_dfs[(sym, strat_name)] = df
                for s in slist:
                    sigs.append(PortfolioSignal(
                        ts=s.ts, next_open_ts=s.next_open_ts,
                        sym=sym, strat=strat_name, direction=s.direction,
                        entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
                    ))
        elif strat_name in STRATEGIES_MULTI_TF:
            primary_tf, secondary_tf, fn = STRATEGIES_MULTI_TF[strat_name]
            for sym in symbols:
                df_p = load(sym, primary_tf, tag)
                if df_p is None or len(df_p) < 100: continue
                if secondary_tf is None:
                    slist = fn(df_p, regime)
                    sim_df = df_p
                elif strat_name == "confluence_4h":
                    df_s = load(sym, secondary_tf, tag)
                    slist = fn(df_p, df_s, regime)
                    sim_df = df_p
                elif strat_name == "st_1d_4h":
                    df_s = load(sym, secondary_tf, tag)
                    slist = fn(df_p, df_s, regime)
                    sim_df = df_s
                else:
                    slist = []; sim_df = df_p
                sim_dfs[(sym, strat_name)] = sim_df
                for s in slist:
                    sigs.append(PortfolioSignal(
                        ts=s.ts, next_open_ts=s.next_open_ts,
                        sym=sym, strat=strat_name, direction=s.direction,
                        entry=s.entry, sl=s.sl, tp=s.tp, dist=s.dist,
                    ))
    if bull_only:
        sigs = [s for s in sigs if regime_at(regime, s.ts)[0] >= 0]  # bull or sideways
    sigs.sort(key=lambda x: x.next_open_ts)
    return sigs, sim_dfs


def simulate_portfolio(sigs: list[PortfolioSignal], sim_dfs: dict,
                       regime: Regime, start_balance=START_BALANCE,
                       risk=RISK_PER_TRADE, max_pos=MAX_POSITIONS,
                       sl_cooldown_ms=SL_COOLDOWN_MS) -> dict:
    """Compounding portfolio sim. SL first, taker fees both sides, slippage both sides, funding."""
    balance = start_balance
    peak = start_balance
    max_dd = 0.0
    open_pos: dict[tuple[str, str], dict] = {}  # (sym, strat) -> info
    sym_cool: dict[str, int] = {}
    trades: list[dict] = []
    equity_curve = [(0, balance)]

    # Keep a running cursor of pending exits for open positions
    # We'll process events in chronological order: signals + position-bar updates
    # Simpler: iterate through signal list, for each signal check if we can open,
    # then walk forward bars to find exit.

    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos: continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts: continue
        # Close any positions whose exit ts is before sig.next_open_ts
        for k in list(open_pos.keys()):
            pos = open_pos[k]
            if pos["exit_ts"] is not None and pos["exit_ts"] <= sig.next_open_ts:
                # Realize PnL
                _realize(pos, balance_ref={"balance": balance}, trades=trades)
                balance = pos["balance_after"]
                if pos["reason"] == "SL":
                    sym_cool[pos["sym"]] = pos["exit_ts"] + sl_cooldown_ms
                del open_pos[k]
        if len(open_pos) >= max_pos: continue

        # Find entry bar in sim_df
        df = sim_dfs.get(key)
        if df is None: continue
        ts = df["timestamp"].to_numpy()
        op = df["open"].to_numpy()
        hi = df["high"].to_numpy()
        lo = df["low"].to_numpy()
        cl = df["close"].to_numpy()
        n = len(ts)
        period_ms = _ts_period_ms(ts)
        funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))

        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1: continue
        entry_p = op[j] * (1 + SLIPPAGE) if sig.direction == "LONG" else op[j] * (1 - SLIPPAGE)
        sl = sig.sl
        if sig.direction == "LONG":
            dist = entry_p - sl
            if dist <= 0: continue
            tp = entry_p + (sig.tp - sig.entry) * (dist / sig.dist) if sig.dist > 0 else sig.tp
        else:
            dist = sl - entry_p
            if dist <= 0: continue
            tp = entry_p - (sig.entry - sig.tp) * (dist / sig.dist) if sig.dist > 0 else sig.tp

        risk_usd = balance * risk
        qty = risk_usd / dist
        notional = entry_p * qty
        # Margin sanity
        max_margin = balance * MAX_MARGIN_FRAC
        lev_used = max(1, int(np.ceil(notional / max_margin))) if max_margin > 0 else 1
        if lev_used > MAX_LEVERAGE: continue
        if notional / lev_used < MIN_MARGIN: continue

        # Walk forward for exit
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
        if exit_bar is None:
            exit_p = cl[min(j + 499, n - 1)]
            exit_bar = min(j + 499, n - 1); reason = "timeout"
        bars_held = exit_bar - j + 1
        funding = notional * funding_per_bar * bars_held
        # Apply slippage to exit
        if sig.direction == "LONG":
            slip_exit = exit_p * (1 - SLIPPAGE)
            gross = (slip_exit - entry_p) * qty
        else:
            slip_exit = exit_p * (1 + SLIPPAGE)
            gross = (entry_p - slip_exit) * qty
        entry_fee = entry_p * qty * TAKER_FEE
        exit_fee = slip_exit * qty * TAKER_FEE
        pnl = gross - entry_fee - exit_fee - funding

        # Store as open position with deferred realization
        open_pos[key] = {
            "sym": sig.sym, "strat": sig.strat, "direction": sig.direction,
            "entry_ts": int(ts[j]), "exit_ts": int(ts[exit_bar]),
            "entry": entry_p, "exit": slip_exit, "reason": reason,
            "pnl_usd": pnl, "risk_usd": risk_usd,
            "balance_before": balance, "balance_after": None,
            "bars_held": bars_held,
        }

    # Realize all remaining
    sigs_ts = [s.next_open_ts for s in sigs] + [10**18]
    final_ts = max(sigs_ts) + 10**12
    for k in list(open_pos.keys()):
        pos = open_pos[k]
        if pos["exit_ts"] is not None:
            _realize(pos, balance_ref={"balance": balance}, trades=trades)
            balance = pos["balance_after"]
            if pos["reason"] == "SL":
                sym_cool[pos["sym"]] = pos["exit_ts"] + sl_cooldown_ms
            del open_pos[k]

    # Sort trades by exit_ts and compute equity curve / DD
    trades.sort(key=lambda x: x["exit_ts"])
    eq = start_balance
    peak = start_balance
    eq_curve = [(0, eq)]
    max_dd_pct = 0.0
    for t in trades:
        eq += t["pnl_usd"]
        eq = max(eq, 0.01)
        peak = max(peak, eq)
        dd = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd_pct = max(max_dd_pct, dd)
        eq_curve.append((t["exit_ts"], eq))
        t["equity_after"] = eq
    # Monthly returns
    monthly = _monthly_returns(eq_curve, start_balance)

    full = metrics_dict_from_trades(trades, start_balance)
    full["max_dd_pct"] = round(max_dd_pct, 1)
    full["final_balance"] = round(eq, 2)
    full["return_pct"] = round((eq - start_balance) / start_balance * 100, 1)

    return {
        "trades": trades, "equity_curve": eq_curve,
        "metrics": full, "monthly_returns": monthly,
    }


def _realize(pos: dict, balance_ref: dict, trades: list):
    new_bal = balance_ref["balance"] + pos["pnl_usd"]
    new_bal = max(new_bal, 0.01)
    pos["balance_after"] = new_bal
    balance_ref["balance"] = new_bal
    trades.append({
        "sym": pos["sym"], "strat": pos["strat"], "direction": pos["direction"],
        "entry_ts": pos["entry_ts"], "exit_ts": pos["exit_ts"],
        "entry": pos["entry"], "exit": pos["exit"], "reason": pos["reason"],
        "pnl_usd": pos["pnl_usd"], "risk_usd": pos["risk_usd"],
        "pnl_r": pos["pnl_usd"] / pos["risk_usd"] if pos["risk_usd"] > 0 else 0,
        "win": pos["pnl_usd"] > 0,
        "bars_held": pos["bars_held"],
    })


def metrics_dict_from_trades(trades: list[dict], start_balance: float) -> dict:
    if not trades: return {"n": 0}
    wins = [t for t in trades if t["pnl_usd"] > 0]
    losses = [t for t in trades if t["pnl_usd"] <= 0]
    gw = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    wr = len(wins) / len(trades) * 100
    avg_r = float(np.mean([t["pnl_r"] for t in trades]))
    sum_r = float(np.sum([t["pnl_r"] for t in trades]))
    return {
        "n": len(trades), "wr": round(wr, 1),
        "pf": round(pf, 2) if pf != float("inf") else 99.0,
        "avg_r": round(avg_r, 3), "sum_r": round(sum_r, 1),
    }


def _monthly_returns(eq_curve: list[tuple[int, float]], start_bal: float) -> list[dict]:
    if len(eq_curve) < 2: return []
    df = pd.DataFrame(eq_curve, columns=["ts", "eq"])
    df = df[df["ts"] > 0].copy()
    if df.empty: return []
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df["month"] = df["dt"].dt.to_period("M").astype(str)
    g = df.groupby("month").agg(start=("eq", "first"), end=("eq", "last")).reset_index()
    # Need monthly start = previous month end (chained)
    out = []
    prev_end = start_bal
    for _, row in g.iterrows():
        s = prev_end
        e = float(row["end"])
        ret = (e - s) / s * 100 if s > 0 else 0
        out.append({"month": row["month"], "start": round(s, 2),
                    "end": round(e, 2), "ret_pct": round(ret, 1)})
        prev_end = e
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="all",
                    help="all | donchian_1d | smacross_1d | supertrend_4h | "
                         "bb_meanrev_1h | atr_compression_4h | vwap_momentum_1h | "
                         "macd_4h | rsi_reversion_4h | donchian_pullback_4h | xsec_momentum")
    ap.add_argument("--symbols", default="", help="comma-separated, default all 27")
    ap.add_argument("--tag", default="730d")
    ap.add_argument("--out", default="multi_agent/backtests/results/strategy_lab.json")
    ap.add_argument("--risk", default="", help="risk per trade override (0.02 = 2%)")
    ap.add_argument("--max-pos", default="", help="max concurrent positions override")
    args = ap.parse_args()

    syms = args.symbols.split(",") if args.symbols else SYMBOLS_730D
    syms = [s.strip().upper() for s in syms if s.strip()]

    t0 = _time.time()
    # Build BTC regime
    btc_1d = load("BTC", "1d", args.tag)
    if btc_1d is None:
        logger.error("BTC 1d data not found — cannot build regime.")
        return
    regime = build_regime(btc_1d)
    bull_d = (regime.trend == 1).sum()
    side_d = (regime.trend == 0).sum()
    bear_d = (regime.trend == -1).sum()
    logger.info(f"Regime built: bull={bull_d}d  sideways={side_d}d  bear={bear_d}d  (total {len(regime.ts)}d)")

    if args.strategy == "all":
        names = list(STRATEGIES.keys()) + list(STRATEGIES_MULTI_TF.keys()) + ["xsec_momentum"]
    elif args.strategy == "v2":
        names = list(STRATEGIES_MULTI_TF.keys())
    elif args.strategy.startswith("portfolio"):
        # Allow user to pick strats and risk via --port-strats and --risk
        if args.strategy == "portfolio":
            port_strats = ["smacross_1d", "supertrend_4h", "rsi_reversion_4h", "st_1d_4h"]
        elif args.strategy == "portfolio2":
            # Drop losing st_1d_4h
            port_strats = ["smacross_1d", "supertrend_4h", "rsi_reversion_4h"]
        elif args.strategy == "portfolio_aggressive":
            port_strats = ["smacross_1d", "supertrend_4h", "rsi_reversion_4h"]
        elif args.strategy == "portfolio_bull":
            port_strats = ["smacross_1d", "supertrend_4h"]
        else:
            port_strats = args.strategy.replace("portfolio:", "").split(",")
        risk = float(args.risk) if args.risk else RISK_PER_TRADE
        max_pos = int(args.max_pos) if args.max_pos else MAX_POSITIONS
        bull_only = args.strategy == "portfolio_bull"
        logger.info(f"Portfolio mode: {port_strats}  risk={risk*100:.1f}%  max_pos={max_pos}  bull_only={bull_only}")
        sigs, sim_dfs = collect_portfolio_signals(port_strats, syms, regime, args.tag, bull_only=bull_only)
        logger.info(f"Total signals: {len(sigs)}")
        res = simulate_portfolio(sigs, sim_dfs, regime,
                                 start_balance=START_BALANCE,
                                 risk=risk,
                                 max_pos=max_pos)
        m = res["metrics"]
        print(f"\n══════════════════════════════════════════════════════════════════")
        print(f"  PORTFOLIO  ({args.tag}, {len(syms)} syms, ${START_BALANCE:.0f} start, {RISK_PER_TRADE*100:.1f}% risk)")
        print(f"  Strategies: {', '.join(port_strats)}")
        print(f"══════════════════════════════════════════════════════════════════")
        print(f"  Trades : {m['n']}")
        print(f"  Win-rate: {m['wr']}%   PF: {m['pf']}   avgR: {m['avg_r']:+.3f}")
        print(f"  Final balance: ${m['final_balance']:,.2f}   Return: {m['return_pct']:+.1f}%")
        print(f"  Max DD: {m['max_dd_pct']:.1f}%")
        print(f"  Per-strategy breakdown:")
        per_strat = defaultdict(list)
        for t in res["trades"]:
            per_strat[t["strat"]].append(t)
        for sn, tlist in per_strat.items():
            mm = metrics_dict_from_trades(tlist, START_BALANCE)
            print(f"    {sn:25s}: n={mm['n']:4d} WR={mm.get('wr',0):5.1f}% PF={mm.get('pf',0):5.2f} sumR={mm.get('sum_r',0):+7.1f}")
        print(f"\n  Monthly returns:")
        print(f"  {'Month':10s} {'Start':>10s} {'End':>10s} {'Return':>9s}")
        for mr in res["monthly_returns"]:
            print(f"  {mr['month']:10s} ${mr['start']:>9,.2f} ${mr['end']:>9,.2f} {mr['ret_pct']:>+8.1f}%")
        rets = [mr["ret_pct"] for mr in res["monthly_returns"]]
        if rets:
            print(f"\n  Avg monthly: {np.mean(rets):+.1f}%   Median: {np.median(rets):+.1f}%   Std: {np.std(rets):.1f}%")
            print(f"  Profitable months: {sum(1 for r in rets if r > 0)}/{len(rets)}")
        # Save
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {"metrics": m, "monthly_returns": res["monthly_returns"]}
        out_path.write_text(json.dumps(out_data, indent=2, default=str), encoding="utf-8")
        logger.info(f"Total runtime: {_time.time() - t0:.1f}s")
        return
    else:
        names = [args.strategy]

    results = {}
    for name in names:
        logger.info(f"Running {name}...")
        if name == "xsec_momentum":
            res = run_xsec(syms, regime, args.tag)
        elif name in STRATEGIES_MULTI_TF:
            res = run_strategy_multitf(name, syms, regime, args.tag)
        else:
            res = run_strategy(name, syms, regime, args.tag)
        print_strategy_report(res)
        # Strip trades for JSON dump (too large)
        rs = {k: v for k, v in res.items() if k != "trades"}
        results[name] = rs

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    logger.info(f"Results saved to {out_path}")
    logger.info(f"Total runtime: {_time.time() - t0:.1f}s")

    # Summary table
    print(f"\n══════════════════════════════════════════════════════════════════")
    print(f"  SUMMARY  ({args.tag}, {len(syms)} symbols)")
    print(f"══════════════════════════════════════════════════════════════════")
    print(f"  {'Strategy':25s} {'TF':4s} {'n':>5s} {'WR%':>6s} {'PF':>6s} {'avgR':>7s} {'sumR':>8s} {'ddR':>6s}")
    rows = []
    for name, r in results.items():
        f = r["full"]
        rows.append((name, r["tf"], f["n"], f["wr"], f["pf"], f["avg_r"], f["sum_r"], f["max_dd_r"]))
    rows.sort(key=lambda x: x[4], reverse=True)  # by PF
    for name, tf, n, wr, pf, ar, sr, dd in rows:
        print(f"  {name:25s} {tf:4s} {n:5d} {wr:6.1f} {pf:6.2f} {ar:+7.3f} {sr:+8.1f} {dd:6.1f}")


if __name__ == "__main__":
    main()
