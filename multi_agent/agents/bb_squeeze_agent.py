"""
multi_agent/agents/bb_squeeze_agent.py

BB Squeeze Breakout Agent -- TTM Squeeze + Chandelier Exit trailing stop.

Strategy:
  Entry TF: 15m  |  HTF bias: 1h EMA-200
  TTM Squeeze = Bollinger Bands inside Keltner Channels (volatility compression)
  On squeeze release + momentum confirmation -> enter in direction of momentum.

LONG entry (all 8 must be true):
  1. Squeeze was active (BB inside KC) for >= MIN_SQUEEZE_BARS
  2. Squeeze just released (first bar where BB outside KC)
  3. Momentum oscillator (close - linreg(close, 20)) > 0
  4. Momentum increasing (current > previous bar)
  5. ADX(14) > 20 (trending, not ranging)
  6. EMA stack aligned: close > EMA21 > EMA50 > EMA200
  7. Volume > 1.2x 20-bar average
  8. OI delta filter confirms (rising OI for longs)

SHORT: mirror conditions.

SL: Chandelier Exit (highest_high(CE_LENGTH) - ATR(CE_LENGTH) * CE_MULT)
TP: Multi-level: 25% at 1.5R, 25% at 2.5R, 50% trailing via Chandelier.

Head-fake protection: skip if momentum reverses within 2 bars of release.
"""

from __future__ import annotations

import asyncio
import logging
import math
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
    from multi_agent.filters.oi_delta_filter import OIDeltaFilter

# -- Logging with rotation --
_log_path = Path(cfg.BB_SQUEEZE_LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("bb_squeeze_agent")
if not log.handlers:
    _fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    _fh = RotatingFileHandler(
        str(_log_path), maxBytes=10 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(_fmt)
    log.addHandler(_ch)
    log.setLevel(logging.INFO)

AGENT_ID = "bb_squeeze_agent"


# ============================================================================
#  PURE INDICATOR FUNCTIONS  (no I/O, all operate on pd.DataFrame / numpy)
# ============================================================================

def _calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                   period: int = 14) -> np.ndarray:
    """ATR as a full array (NaN for first `period-1` values)."""
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))
    atr = np.full(n, np.nan)
    if n >= period:
        atr[period - 1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def _ema(data: np.ndarray, period: int) -> np.ndarray:
    """Exponential moving average."""
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    out[period - 1] = np.mean(data[:period])
    k = 2.0 / (period + 1)
    for i in range(period, len(data)):
        out[i] = data[i] * k + out[i - 1] * (1 - k)
    return out


def _sma(data: np.ndarray, period: int) -> np.ndarray:
    """Simple moving average."""
    out = np.full(len(data), np.nan)
    if len(data) < period:
        return out
    cumsum = np.cumsum(data)
    out[period - 1:] = (cumsum[period - 1:] - np.concatenate([[0], cumsum[:-period]])) / period
    return out


def _stdev(data: np.ndarray, period: int) -> np.ndarray:
    """Rolling standard deviation."""
    out = np.full(len(data), np.nan)
    for i in range(period - 1, len(data)):
        out[i] = np.std(data[i - period + 1:i + 1], ddof=0)
    return out


def _linreg(data: np.ndarray, period: int) -> np.ndarray:
    """Linear regression value (endpoint of best-fit line over last `period` bars)."""
    out = np.full(len(data), np.nan)
    x = np.arange(period, dtype=float)
    x_mean = x.mean()
    x_var = np.sum((x - x_mean) ** 2)
    for i in range(period - 1, len(data)):
        y = data[i - period + 1:i + 1]
        y_mean = y.mean()
        slope = np.sum((x - x_mean) * (y - y_mean)) / max(x_var, 1e-15)
        out[i] = y_mean + slope * (period - 1 - x_mean)
    return out


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> np.ndarray:
    """Average Directional Index."""
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period * 2:
        return out

    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]
        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0.0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0.0
        tr[i] = max(highs[i] - lows[i],
                     abs(highs[i] - closes[i - 1]),
                     abs(lows[i] - closes[i - 1]))

    # Smoothed TR, +DM, -DM
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

    # Smooth DX -> ADX
    start = period * 2
    if start < n:
        out[start] = np.mean(dx[period + 1:start + 1])
        for i in range(start + 1, n):
            out[i] = (out[i - 1] * (period - 1) + dx[i]) / period

    return out


def compute_squeeze_indicators(df: pd.DataFrame, bb_len: int = 20,
                                bb_mult: float = 2.0, kc_len: int = 20,
                                kc_mult: float = 1.5, mom_len: int = 20,
                                adx_period: int = 14):
    """
    Compute all TTM Squeeze indicators on a DataFrame.
    Returns dict of numpy arrays:
      squeeze_on, squeeze_off, momentum, adx, ema21, ema50, ema200,
      bb_upper, bb_lower, kc_upper, kc_lower, atr
    """
    closes = df["close"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()
    n = len(closes)

    # Bollinger Bands
    bb_basis = _sma(closes, bb_len)
    bb_dev = _stdev(closes, bb_len) * bb_mult
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev

    # Keltner Channels
    kc_basis = _ema(closes, kc_len)
    atr_arr = _calculate_atr(highs, lows, closes, kc_len)
    kc_upper = kc_basis + kc_mult * atr_arr
    kc_lower = kc_basis - kc_mult * atr_arr

    # Squeeze: BB inside KC
    squeeze_on = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(bb_lower[i]) and not np.isnan(kc_lower[i]):
            squeeze_on[i] = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])

    # Momentum: close - linreg(close, mom_len)
    linreg_vals = _linreg(closes, mom_len)
    momentum = closes - linreg_vals

    # ADX
    adx_arr = _adx(highs, lows, closes, adx_period)

    # EMAs
    ema21 = _ema(closes, 21)
    ema50 = _ema(closes, 50)
    ema200 = _ema(closes, 200)

    # Volume average
    volumes = df["volume"].astype(float).to_numpy()
    vol_sma20 = _sma(volumes, 20)

    # ATR for Chandelier Exit
    ce_atr = _calculate_atr(highs, lows, closes, cfg.BB_SQUEEZE_CE_LENGTH)

    return {
        "squeeze_on": squeeze_on,
        "momentum": momentum,
        "adx": adx_arr,
        "ema21": ema21,
        "ema50": ema50,
        "ema200": ema200,
        "bb_upper": bb_upper,
        "bb_lower": bb_lower,
        "kc_upper": kc_upper,
        "kc_lower": kc_lower,
        "atr": atr_arr,
        "ce_atr": ce_atr,
        "volumes": volumes,
        "vol_sma20": vol_sma20,
        "closes": closes,
        "highs": highs,
        "lows": lows,
    }


def detect_squeeze_signal(ind: dict, bar_idx: int = -1,
                           min_squeeze_bars: int = 6,
                           adx_threshold: float = 20.0,
                           vol_mult: float = 1.2) -> Optional[dict]:
    """
    Check all 8 entry conditions at bar_idx.
    Returns {"direction": "LONG"|"SHORT", ...} or None.
    """
    n = len(ind["closes"])
    if bar_idx < 0:
        bar_idx = n + bar_idx
    if bar_idx < 2 or bar_idx >= n:
        return None

    i = bar_idx

    # Check for NaN in key indicators
    for key in ("ema21", "ema50", "ema200", "adx", "momentum"):
        if np.isnan(ind[key][i]):
            return None

    # Condition 1: Squeeze was active for >= min_squeeze_bars before release
    if ind["squeeze_on"][i]:
        return None  # still in squeeze, no release yet

    # Count consecutive squeeze bars before i
    squeeze_count = 0
    for j in range(i - 1, -1, -1):
        if ind["squeeze_on"][j]:
            squeeze_count += 1
        else:
            break

    if squeeze_count < min_squeeze_bars:
        return None

    # Condition 2: This is the first bar after squeeze release
    # Already confirmed: squeeze_on[i] == False and squeeze_on[i-1] == True
    if not ind["squeeze_on"][i - 1]:
        return None  # Not the first release bar

    # Condition 3 & 4: Momentum direction and increasing
    mom = ind["momentum"][i]
    mom_prev = ind["momentum"][i - 1]
    if np.isnan(mom) or np.isnan(mom_prev):
        return None

    # Condition 5: ADX
    if ind["adx"][i] < adx_threshold:
        return None

    # Condition 7: Volume
    vol = ind["volumes"][i]
    vol_avg = ind["vol_sma20"][i]
    if np.isnan(vol_avg) or vol_avg <= 0 or vol < vol_mult * vol_avg:
        return None

    close = ind["closes"][i]
    ema21 = ind["ema21"][i]
    ema50 = ind["ema50"][i]
    ema200 = ind["ema200"][i]

    # LONG setup
    if mom > 0 and mom > mom_prev:  # Conditions 3 & 4
        # Condition 6: EMA stack
        if close > ema21 > ema50 > ema200:
            return {
                "direction": "LONG",
                "entry_price": close,
                "momentum": float(mom),
                "adx": float(ind["adx"][i]),
                "squeeze_bars": squeeze_count,
                "volume_ratio": float(vol / max(vol_avg, 1e-9)),
                "bar_idx": i,
            }

    # SHORT setup
    if mom < 0 and mom < mom_prev:  # Conditions 3 & 4 (mirrored)
        # Condition 6: EMA stack (mirrored)
        if close < ema21 < ema50 < ema200:
            return {
                "direction": "SHORT",
                "entry_price": close,
                "momentum": float(mom),
                "adx": float(ind["adx"][i]),
                "squeeze_bars": squeeze_count,
                "volume_ratio": float(vol / max(vol_avg, 1e-9)),
                "bar_idx": i,
            }

    return None


def chandelier_exit_sl(highs: np.ndarray, lows: np.ndarray, atr: np.ndarray,
                        direction: str, bar_idx: int,
                        ce_length: int = 22, ce_mult: float = 3.0) -> float:
    """
    Chandelier Exit stop-loss.
    LONG:  SL = highest_high(ce_length) - ATR * ce_mult
    SHORT: SL = lowest_low(ce_length)  + ATR * ce_mult
    """
    start = max(0, bar_idx - ce_length + 1)
    end = bar_idx + 1

    if np.isnan(atr[bar_idx]):
        # fallback ATR
        atr_val = np.nanmean(atr[max(0, bar_idx - 14):end])
        if np.isnan(atr_val):
            atr_val = 0.0
    else:
        atr_val = atr[bar_idx]

    if direction == "LONG":
        highest = float(np.max(highs[start:end]))
        sl = highest - atr_val * ce_mult
    else:
        lowest = float(np.min(lows[start:end]))
        sl = lowest + atr_val * ce_mult

    return float(sl)


def check_head_fake(ind: dict, release_bar: int, check_bars: int = 2) -> bool:
    """
    Head-fake detection: if momentum reverses sign within check_bars after
    the squeeze release, it's likely a false breakout.
    """
    n = len(ind["momentum"])
    if release_bar + check_bars >= n:
        return False  # not enough bars to check yet

    mom_at_release = ind["momentum"][release_bar]
    if np.isnan(mom_at_release):
        return True

    for j in range(1, check_bars + 1):
        future_mom = ind["momentum"][release_bar + j]
        if np.isnan(future_mom):
            continue
        # Sign reversal?
        if mom_at_release > 0 and future_mom < 0:
            return True
        if mom_at_release < 0 and future_mom > 0:
            return True
    return False


# ============================================================================
#  BBSqueezeAgent
# ============================================================================

class BBSqueezeAgent:
    """
    BB Squeeze Breakout trading agent for the multi-agent system.

    Uses TTM Squeeze (BB inside KC) detection + Chandelier Exit trailing stop.
    Shares infrastructure with FVG and SMC agents.
    SIMULATION_MODE = True -> orders are logged, never placed.
    """

    def __init__(
        self,
        agent_id: str,
        state_manager: "StateManager",
        bybit_client: "BybitClient",
        risk_manager: "RiskManager",
        telegram: "MultiAgentTelegramNotifier",
        oi_filter: "OIDeltaFilter",
    ) -> None:
        self.agent_id  = agent_id
        self.state     = state_manager
        self.client    = bybit_client
        self.risk      = risk_manager
        self.tg        = telegram
        self.oi_filter = oi_filter

        # Daily tracking
        self.daily_trades     = 0
        self.daily_pnl        = 0.0
        self.daily_wins       = 0
        self.daily_losses     = 0
        self.last_daily_reset = ""
        self.starting_equity  = 0.0

        # Trade execution lock — prevents race condition
        self._trade_lock: asyncio.Lock = asyncio.Lock()

        # Symbol list
        self._symbols: list[str] = []
        self._last_symbols_refresh: float = 0.0
        self._markets_cache: dict = {}

        # Trailing stop state per symbol:
        # {symbol: {"tp1_hit": bool, "tp2_hit": bool, "trailing_sl": float}}
        self._trailing_state: dict[str, dict] = {}

    # -- Helpers --

    async def heartbeat(self) -> None:
        await self.state.record_heartbeat(self.agent_id)

    async def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_daily_reset != today:
            self.daily_trades     = 0
            self.daily_pnl        = 0.0
            self.daily_wins       = 0
            self.daily_losses     = 0
            self.last_daily_reset = today
            equity = await self.state.get_agent_equity(self.agent_id)
            if equity > 0:
                self.starting_equity = equity
            log.info("[BBS] Daily stats reset for %s", today)

    # -- Symbol management --

    async def _refresh_symbols_if_needed(self) -> None:
        import time as _time
        now = _time.monotonic()
        refresh_interval = getattr(cfg, "SYMBOLS_REFRESH_HR", 4) * 3600
        if now - self._last_symbols_refresh < refresh_interval and self._symbols:
            return
        try:
            tickers = await self.client._exchange.fetch_tickers()
            min_vol = float(cfg.MIN_24H_VOLUME_USDT)
            usdt_perp = [
                (sym, t)
                for sym, t in tickers.items()
                if sym.endswith(":USDT")
                and float(t.get("quoteVolume") or 0) >= min_vol
            ]
            usdt_perp.sort(key=lambda x: float(x[1].get("quoteVolume") or 0), reverse=True)
            self._symbols = [sym for sym, _ in usdt_perp[:cfg.TOP_N_COINS]]
            self._last_symbols_refresh = now
            self._markets_cache = dict(self.client._exchange.markets or {})
            log.info("[BBS] %d symbols loaded.", len(self._symbols))
        except Exception as exc:
            log.error("[BBS] Symbol refresh failed: %s", exc)

    # -- Open-position monitor with trailing stop --

    async def _check_open_positions(self) -> None:
        """Check SL/TP for each tracked position. Update trailing Chandelier Exit."""
        positions = await self.state.get_agent_positions(self.agent_id)

        for symbol, pos in list(positions.items()):
            try:
                ticker = await self.client.fetch_ticker(symbol)
                price = float(ticker.get("last") or 0)
                if price <= 0:
                    continue

                side   = str(pos.get("side", "LONG"))
                sl     = float(pos.get("sl", 0))
                entry  = float(pos.get("entry", 0))
                risk   = abs(entry - sl) if sl > 0 and entry > 0 else 0.0
                hit    = False
                reason = ""

                # Get or init trailing state
                ts = self._trailing_state.setdefault(symbol, {
                    "tp1_hit": False, "tp2_hit": False, "trailing_sl": sl,
                })

                # Chandelier-only trailing stop (Variant D winner)
                # Fetch fresh 15m data for Chandelier update
                use_fixed_tp = cfg.BB_SQUEEZE_AGENT_CONFIG.get("USE_FIXED_TP", False)

                if risk > 0:
                    side_str = "buy" if side == "LONG" else "sell"
                    qty = float(pos.get("qty", 0))
                    tp = float(pos.get("tp", 0))

                    if side == "LONG":
                        pnl_r = (price - entry) / risk

                        if use_fixed_tp:
                            if not ts["tp1_hit"] and pnl_r >= 1.5:
                                ts["tp1_hit"] = True
                                log.info("[BBS] TP1 hit (1.5R) for %s LONG", symbol)
                            if not ts["tp2_hit"] and pnl_r >= 2.5:
                                ts["tp2_hit"] = True
                                log.info("[BBS] TP2 hit (2.5R) for %s LONG", symbol)

                        # Trail SL upward: max of initial SL and breakeven after 1R
                        if pnl_r >= 1.0:
                            new_trail = max(entry + risk * 0.1, price - risk * 1.5)
                            if new_trail > ts["trailing_sl"]:
                                ts["trailing_sl"] = new_trail
                                # Push updated SL to Bybit so server-side order is in sync
                                try:
                                    await self.client.place_sl_tp(symbol, side_str, qty, new_trail, tp)
                                    log.info("[BBS] Trailing SL updated on Bybit: %s LONG sl=%.6g", symbol, new_trail)
                                except Exception as _e:
                                    log.warning("[BBS] Failed to update trailing SL on Bybit %s: %s", symbol, _e)

                        effective_sl = max(sl, ts["trailing_sl"])
                        if price <= effective_sl:
                            hit = True
                            reason = "Trailing SL" if ts["trailing_sl"] > sl else "SL"

                    else:  # SHORT
                        pnl_r = (entry - price) / risk

                        if use_fixed_tp:
                            if not ts["tp1_hit"] and pnl_r >= 1.5:
                                ts["tp1_hit"] = True
                                log.info("[BBS] TP1 hit (1.5R) for %s SHORT", symbol)
                            if not ts["tp2_hit"] and pnl_r >= 2.5:
                                ts["tp2_hit"] = True
                                log.info("[BBS] TP2 hit (2.5R) for %s SHORT", symbol)

                        if pnl_r >= 1.0:
                            new_trail = min(entry - risk * 0.1, price + risk * 1.5)
                            prev_trail = ts["trailing_sl"] if ts["trailing_sl"] > 0 else float("inf")
                            if new_trail < prev_trail:
                                ts["trailing_sl"] = new_trail
                                # Push updated SL to Bybit so server-side order is in sync
                                try:
                                    await self.client.place_sl_tp(symbol, side_str, qty, new_trail, tp)
                                    log.info("[BBS] Trailing SL updated on Bybit: %s SHORT sl=%.6g", symbol, new_trail)
                                except Exception as _e:
                                    log.warning("[BBS] Failed to update trailing SL on Bybit %s: %s", symbol, _e)

                        effective_sl = ts["trailing_sl"] if ts["trailing_sl"] > 0 else sl
                        if sl > 0 and price >= effective_sl:
                            hit = True
                            reason = "Trailing SL" if ts["trailing_sl"] < sl else "SL"

                # Fallback: fixed SL/TP
                if not hit and side == "LONG":
                    if sl > 0 and price <= sl:
                        hit = True; reason = "SL"
                elif not hit and side == "SHORT":
                    if sl > 0 and price >= sl:
                        hit = True; reason = "SL"

                # Fixed R exit (v3 evidence: fixed 3R = +49R vs chandelier +32.64R)
                exit_mode = cfg.BB_SQUEEZE_AGENT_CONFIG.get("EXIT_MODE", "chandelier")
                fixed_tp_r = cfg.BB_SQUEEZE_AGENT_CONFIG.get("FIXED_TP_R", 3.0)
                max_r = fixed_tp_r if exit_mode == "fixed_R" else 4.0
                if not hit and risk > 0:
                    if side == "LONG" and (price - entry) / risk >= max_r:
                        hit = True; reason = f"TP_{max_r:.0f}R"
                    elif side == "SHORT" and (entry - price) / risk >= max_r:
                        hit = True; reason = f"TP_{max_r:.0f}R"

                if hit:
                    await self._close_position(symbol, pos, price, reason)
                    self._trailing_state.pop(symbol, None)

            except Exception as exc:
                log.warning("[BBS] Position check error %s: %s", symbol, exc)

    async def _close_position(
        self, symbol: str, pos: dict, exit_price: float, reason: str
    ) -> None:
        await self.state.remove_position(self.agent_id, symbol)
        side  = str(pos.get("side", "LONG"))
        entry = float(pos.get("entry", exit_price))
        qty   = float(pos.get("qty", 0))
        if side == "LONG":
            pnl_usd = (exit_price - entry) * qty
            pnl_pct = (exit_price - entry) / max(entry, 1e-9) * 100.0
        else:
            pnl_usd = (entry - exit_price) * qty
            pnl_pct = (entry - exit_price) / max(entry, 1e-9) * 100.0

        self.daily_pnl    += pnl_usd
        self.daily_trades += 1
        if pnl_usd >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        await self.state.record_trade(self.agent_id, {
            "symbol":    symbol,
            "side":      side,
            "entry":     entry,
            "exit":      exit_price,
            "pnl_usd":   pnl_usd,
            "pnl_pct":   pnl_pct,
            "reason":    reason,
            "date":      datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        current = await self.state.get_agent_equity(self.agent_id)
        await self.state.set_agent_equity(self.agent_id, max(0.0, current + pnl_usd))

        short_sym = symbol.split("/")[0]
        log.info(
            "[BBS] CLOSED %s %s entry=%.6g exit=%.6g pnl=$%.4f reason=%s",
            short_sym, side, entry, exit_price, pnl_usd, reason,
        )
        await self.tg.send_position_closed(
            self.agent_id, symbol, side, entry, exit_price, pnl_usd, pnl_pct, reason
        )

    # -- Full BB Squeeze scan for one symbol --

    async def scan_symbol(self, symbol: str) -> Optional[dict]:
        """
        Full BB Squeeze scan:
        1. Load 1h data, check HTF EMA-200 bias.
        2. Load 15m data.
        3. Compute all TTM Squeeze indicators.
        4. Check 8-condition entry signal.
        5. Apply OI filter.
        6. Calculate Chandelier Exit SL.
        7. Calculate multi-level TP.
        8. Verify RR >= BB_SQUEEZE_MIN_RR.
        9. Return signal dict or None.
        """
        short_name = symbol.replace("/USDT:USDT", "USDT")

        # Gates
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return None

        # Symbol blacklist (v4 backtest: ARB/OP/POL drag BBS performance)
        blacklist = getattr(cfg, "BBS_SYMBOL_BLACKLIST", [])
        if symbol in blacklist:
            return None

        if not await self.risk.check_position_count(
            self.agent_id, self.state, cfg.BB_SQUEEZE_MAX_POSITIONS
        ):
            return None
        if not await self.risk.check_daily_loss_limit(
            self.agent_id, self.state, cfg.BB_SQUEEZE_MAX_DAILY_LOSS_PCT
        ):
            return None
        if await self.state.is_symbol_locked(symbol):
            return None

        # Already have position in this symbol?
        positions = await self.state.get_agent_positions(self.agent_id)
        if symbol in positions:
            return None

        # Fetch candles
        try:
            raw_htf = await self.client.fetch_ohlcv(
                symbol, cfg.BB_SQUEEZE_HTF_TIMEFRAME, cfg.OHLCV_LIMIT_HTF
            )
            raw_entry = await self.client.fetch_ohlcv(
                symbol, cfg.BB_SQUEEZE_ENTRY_TIMEFRAME, cfg.OHLCV_LIMIT_ENTRY
            )
        except Exception as exc:
            log.debug("[BBS] OHLCV fetch failed %s: %s", short_name, exc)
            return None

        if not raw_htf or not raw_entry:
            return None
        if len(raw_htf) < 210 or len(raw_entry) < 210:
            return None

        df_1h = pd.DataFrame(
            raw_htf, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_15m = pd.DataFrame(
            raw_entry, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        # HTF EMA-200 bias
        ema_htf = df_1h["close"].ewm(span=200, adjust=False).mean()
        last_1h = float(df_1h["close"].iloc[-1])
        ema200_val = float(ema_htf.iloc[-1])
        if pd.isna(ema200_val) or ema200_val == 0:
            return None

        htf_direction = "LONG" if last_1h > ema200_val else "SHORT"

        # Compute indicators
        ind = compute_squeeze_indicators(
            df_15m,
            bb_len=cfg.BB_SQUEEZE_BB_LENGTH,
            bb_mult=cfg.BB_SQUEEZE_BB_MULT,
            kc_len=cfg.BB_SQUEEZE_KC_LENGTH,
            kc_mult=cfg.BB_SQUEEZE_KC_MULT,
            mom_len=cfg.BB_SQUEEZE_MOM_LENGTH,
            adx_period=cfg.BB_SQUEEZE_ADX_PERIOD,
        )

        # Detect signal on last bar
        signal = detect_squeeze_signal(
            ind,
            bar_idx=-1,
            min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
            adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
            vol_mult=cfg.BB_SQUEEZE_VOL_MULT,
        )

        if signal is None:
            return None

        # Must align with HTF direction
        if signal["direction"] != htf_direction:
            return None

        direction = signal["direction"]
        entry_price = signal["entry_price"]
        bar_idx = signal["bar_idx"]

        # OI delta filter
        try:
            oi_ok = await self.oi_filter.check(symbol, direction)
            if not oi_ok:
                log.debug("[BBS] OI filter rejected %s %s", short_name, direction)
                return None
        except Exception:
            pass  # OI filter failure -> skip filter, proceed

        # SL: v8 backtest found ATR×2.5 beats Chandelier (PF 1.77 vs 1.47)
        sl_mode = cfg.BB_SQUEEZE_AGENT_CONFIG.get("SL_MODE", "chandelier")
        if sl_mode == "atr":
            atr_mult = cfg.BB_SQUEEZE_AGENT_CONFIG.get("SL_ATR_MULT", 2.5)
            atr_max_pct = cfg.BB_SQUEEZE_AGENT_CONFIG.get("SL_ATR_MAX_PCT", 5.0)
            atr_val = ind["ce_atr"][bar_idx] if bar_idx < len(ind["ce_atr"]) else entry_price * 0.015
            if np.isnan(atr_val) or atr_val <= 0:
                atr_val = entry_price * 0.015
            atr_dist = min(atr_val * atr_mult, entry_price * atr_max_pct / 100)
            atr_dist = max(atr_dist, entry_price * 0.003)  # min 0.3%
            sl = (entry_price - atr_dist) if direction == "LONG" else (entry_price + atr_dist)
        else:
            # Legacy: Chandelier Exit SL
            sl = chandelier_exit_sl(
                ind["highs"], ind["lows"], ind["ce_atr"],
                direction, bar_idx,
                ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                ce_mult=cfg.BB_SQUEEZE_CE_MULT,
            )

        # Validate SL
        risk = abs(entry_price - sl)
        if risk <= 0 or risk / entry_price > 0.05:  # max 5% SL distance
            return None

        # TP: depends on EXIT_MODE (v3 evidence-based)
        exit_mode = cfg.BB_SQUEEZE_AGENT_CONFIG.get("EXIT_MODE", "chandelier")
        if exit_mode == "fixed_R":
            fixed_tp_r = cfg.BB_SQUEEZE_AGENT_CONFIG.get("FIXED_TP_R", 3.0)
            tp_first = entry_price + risk * fixed_tp_r if direction == "LONG" else entry_price - risk * fixed_tp_r
        else:
            # Legacy: first target at 1.5R (for partial close), max at 4R
            tp_first = entry_price + risk * 1.5 if direction == "LONG" else entry_price - risk * 1.5

        # Verify minimum RR
        min_rr = cfg.BB_SQUEEZE_MIN_RR
        if risk <= 0:
            return None
        rr = abs(tp_first - entry_price) / risk
        if rr < min_rr:
            return None

        # Position size — fetch live balance from Bybit, shared deposit model v10 - full balance
        try:
            raw_bal = await self.client.fetch_balance()
            total_bal = float(
                raw_bal.get("USDT", {}).get("total", 0)
                or raw_bal.get("total", {}).get("USDT", 0)
                or 0
            )
            equity = total_bal  # shared deposit model v10 - full balance
        except Exception:
            equity = await self.state.get_agent_equity(self.agent_id)
        if equity <= 0:
            return None

        qty, leverage = self.risk.calculate_position_size(
            balance=equity,
            entry_price=entry_price,
            sl_price=sl,
            symbol=symbol,
        )
        if qty <= 0:
            return None

        log.info(
            "[BBS] SIGNAL %s %s | entry=%.6g sl=%.6g tp1=%.6g rr=%.2f "
            "squeeze=%d bars adx=%.1f vol=%.1fx",
            short_name, direction, entry_price, sl, tp_first, rr,
            signal["squeeze_bars"], signal["adx"], signal["volume_ratio"],
        )

        return {
            "symbol":       symbol,
            "direction":    direction,
            "entry_price":  entry_price,
            "sl":           sl,
            "tp":           tp_first,
            "rr":           rr,
            "qty":          qty,
            "leverage":     leverage,
            "squeeze_bars": signal["squeeze_bars"],
            "adx":          signal["adx"],
            "volume_ratio": signal["volume_ratio"],
            "momentum":     signal["momentum"],
        }

    # -- Execute trade --

    async def _execute_signal(self, sig: dict) -> None:
        async with self._trade_lock:
            await self._execute_signal_locked(sig)

    async def _execute_signal_locked(self, sig: dict) -> None:
        symbol    = sig["symbol"]
        direction = sig["direction"]
        entry     = sig["entry_price"]
        sl        = sig["sl"]
        tp        = sig["tp"]
        qty       = sig["qty"]
        leverage  = sig["leverage"]
        side_str  = "buy" if direction == "LONG" else "sell"

        short_sym = symbol.split("/")[0]

        # Re-check under lock (race condition guard)
        if await self.state.is_symbol_locked(symbol):
            log.debug("[BBS] RACE: %s already locked — skip", short_sym)
            return

        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception as exc:
            log.warning("[BBS] Set leverage failed %s: %s", short_sym, exc)

        try:
            order = await self.client.place_market_order(symbol, side_str, qty)
        except Exception as exc:
            err_str = str(exc)
            if "110007" in err_str or "not enough" in err_str.lower() or "InsufficientFunds" in err_str:
                log.warning("[BBS] Insufficient margin for %s qty=%.6f — skipping trade", short_sym, qty)
            else:
                log.error("[BBS] Market order error %s: %s", short_sym, exc)
            return
        if order is None:
            log.warning("[BBS] Order placement returned None for %s", short_sym)
            return

        # Record position
        await self.state.register_position(
            self.agent_id, symbol, direction, entry, sl, tp, qty
        )

        # Init trailing state
        self._trailing_state[symbol] = {
            "tp1_hit": False, "tp2_hit": False, "trailing_sl": sl,
        }

        try:
            await self.client.place_sl_tp(symbol, side_str, qty, sl, tp)
        except Exception as exc:
            log.warning("[BBS] SL/TP placement failed %s: %s", short_sym, exc)

        risk_usd = abs(entry - sl) * qty
        log.info(
            "[BBS] OPENED %s %s entry=%.6g sl=%.6g tp=%.6g qty=%.6f risk=$%.4f lev=%dx",
            short_sym, direction, entry, sl, tp, qty, risk_usd, leverage,
        )

        margin_used = qty * entry / max(leverage, 1)
        try:
            await self.tg.send_position_opened(
                self.agent_id, symbol, direction, entry, sl, tp, qty, leverage, margin_used,
            )
        except Exception as exc:
            log.warning("[BBS] TG notification failed: %s", exc)

    # -- Main run loop --

    async def run(self) -> None:
        """Main agent loop. Scans all symbols, manages positions."""
        log.info("[BBS] BB Squeeze agent starting.")
        await self.heartbeat()

        while True:
            try:
                await self._reset_daily_if_needed()
                await self._refresh_symbols_if_needed()

                # Check existing positions first
                await self._check_open_positions()
                await self.heartbeat()

                # Scan for new signals
                batch_size = 10
                for i in range(0, len(self._symbols), batch_size):
                    batch = self._symbols[i:i + batch_size]
                    tasks = [self.scan_symbol(sym) for sym in batch]
                    results = await asyncio.gather(*tasks, return_exceptions=True)

                    for sym, result in zip(batch, results):
                        if isinstance(result, Exception):
                            log.debug("[BBS] Scan error %s: %s", sym, result)
                            continue
                        if result is not None:
                            await self._execute_signal(result)

                    await self.heartbeat()

                log.info(
                    "[BBS] Scan complete. %d symbols. "
                    "Trades today: %d (W:%d L:%d) PnL=$%.4f",
                    len(self._symbols),
                    self.daily_trades, self.daily_wins, self.daily_losses,
                    self.daily_pnl,
                )

                await asyncio.sleep(cfg.SCAN_INTERVAL_SEC)

            except asyncio.CancelledError:
                log.info("[BBS] BB Squeeze agent cancelled.")
                break
            except Exception as exc:
                log.exception("[BBS] Unexpected error: %s", exc)
                try:
                    await self.tg.send_error(self.agent_id, str(exc))
                except Exception:
                    pass
                await asyncio.sleep(10)
