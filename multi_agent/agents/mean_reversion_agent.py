"""
multi_agent/agents/mean_reversion_agent.py

Mean Reversion Agent — captures snapback when price overextends from fair value.

Strategy:
  Entry TF: 15m  |  HTF: 1h for regime context
  Only active when RegimeDetector classifies RANGE.

LONG setup:
  - Price below lower BB
  - RSI < 30
  - VWAP deviation sufficient
  - No strong breakdown impulse
  - ADX low (ranging)

SHORT: mirrored above upper BB.

Exits: BB midline / VWAP, with ATR-based hard stop.
Anti-revenge: max 2 consecutive same-direction losses before cooldown.
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

# ─── Logging ─────────────────────────────────────────────────────────────────
_log_path = Path(cfg.MR_LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("mean_reversion_agent")
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _fh = RotatingFileHandler(str(_log_path), maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(_fmt)
    log.addHandler(_ch)
    log.setLevel(logging.INFO)

AGENT_ID = "mean_reversion_agent"


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure indicator functions (no I/O)
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


def _rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period + 1:
        return out
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    if avg_loss == 0:
        out[period] = 100.0
    else:
        out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            out[i + 1] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
    return out


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> np.ndarray:
    n = len(closes)
    tr = np.empty(n)
    tr[0] = highs[0] - lows[0]
    for i in range(1, n):
        tr[i] = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    out = np.full(n, np.nan)
    if n >= period:
        out[period-1] = np.mean(tr[:period])
        alpha = 1.0 / period
        for i in range(period, n):
            out[i] = out[i-1]*(1-alpha) + tr[i]*alpha
    return out


def _adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
         period: int = 14) -> np.ndarray:
    n = len(closes)
    out = np.full(n, np.nan)
    if n < period * 2:
        return out
    plus_dm = np.zeros(n); minus_dm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        up = highs[i] - highs[i-1]
        down = lows[i-1] - lows[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1]))
    atr_s = np.zeros(n); pdm_s = np.zeros(n); mdm_s = np.zeros(n)
    atr_s[period] = np.sum(tr[1:period+1])
    pdm_s[period] = np.sum(plus_dm[1:period+1])
    mdm_s[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        atr_s[i] = atr_s[i-1] - atr_s[i-1]/period + tr[i]
        pdm_s[i] = pdm_s[i-1] - pdm_s[i-1]/period + plus_dm[i]
        mdm_s[i] = mdm_s[i-1] - mdm_s[i-1]/period + minus_dm[i]
    dx = np.zeros(n)
    for i in range(period, n):
        if atr_s[i] == 0: continue
        pdi = 100.0 * pdm_s[i] / atr_s[i]
        mdi = 100.0 * mdm_s[i] / atr_s[i]
        s = pdi + mdi
        if s > 0: dx[i] = 100.0 * abs(pdi - mdi) / s
    start = period * 2
    if start < n:
        out[start] = np.mean(dx[period+1:start+1])
        for i in range(start+1, n): out[i] = (out[i-1]*(period-1) + dx[i]) / period
    return out


def compute_rolling_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                          volumes: np.ndarray, lookback: int = 96) -> np.ndarray:
    n = len(closes)
    typical = (highs + lows + closes) / 3.0
    tp_vol = typical * volumes
    out = np.full(n, np.nan)
    for i in range(lookback - 1, n):
        start = i - lookback + 1
        vol_sum = np.sum(volumes[start:i+1])
        if vol_sum > 0:
            out[i] = np.sum(tp_vol[start:i+1]) / vol_sum
    return out


def compute_mr_indicators(df: pd.DataFrame, bb_len: int = 20, bb_mult: float = 2.0,
                           rsi_period: int = 14, adx_period: int = 14) -> dict:
    """Compute all indicators needed for mean reversion signals."""
    closes = df["close"].astype(float).to_numpy()
    highs = df["high"].astype(float).to_numpy()
    lows = df["low"].astype(float).to_numpy()
    volumes = df["volume"].astype(float).to_numpy()

    bb_basis = _sma(closes, bb_len)
    bb_std = _stdev(closes, bb_len)
    bb_upper = bb_basis + bb_mult * bb_std
    bb_lower = bb_basis - bb_mult * bb_std

    rsi_arr = _rsi(closes, rsi_period)
    adx_arr = _adx(highs, lows, closes, adx_period)
    atr_arr = _atr(highs, lows, closes, 14)
    vwap_arr = compute_rolling_vwap(highs, lows, closes, volumes, 96)

    return {
        "closes": closes, "highs": highs, "lows": lows, "volumes": volumes,
        "bb_upper": bb_upper, "bb_lower": bb_lower, "bb_mid": bb_basis,
        "rsi": rsi_arr, "adx": adx_arr, "atr": atr_arr, "vwap": vwap_arr,
    }


def detect_mr_signal(
    ind: dict, bar_idx: int,
    consecutive_losses: dict,  # {"LONG": int, "SHORT": int}
) -> Optional[dict]:
    """
    Check for mean reversion signal at bar_idx.
    Returns signal dict or None.
    """
    n = len(ind["closes"])
    if bar_idx < 0:
        bar_idx = n + bar_idx
    if bar_idx < 30 or bar_idx >= n:
        return None

    i = bar_idx
    close = ind["closes"][i]
    bb_upper = ind["bb_upper"][i]
    bb_lower = ind["bb_lower"][i]
    bb_mid = ind["bb_mid"][i]
    rsi_val = ind["rsi"][i]
    adx_val = ind["adx"][i]
    atr_val = ind["atr"][i]
    vwap_val = ind["vwap"][i]

    for v in (bb_upper, bb_lower, bb_mid, rsi_val, adx_val, atr_val, vwap_val):
        if np.isnan(v):
            return None

    # ADX must be low (range condition)
    if adx_val > cfg.MR_ADX_MAX:
        return None

    # LONG: price below lower BB, RSI oversold, stretched from VWAP
    if close < bb_lower:
        if rsi_val > cfg.MR_RSI_OVERSOLD:
            return None

        vwap_dev = (vwap_val - close) / vwap_val * 100.0 if vwap_val > 0 else 0
        if vwap_dev < cfg.MR_VWAP_MIN_DEVIATION_PCT:
            return None

        # Anti-revenge
        if consecutive_losses.get("LONG", 0) >= cfg.MR_MAX_CONSECUTIVE_LOSSES_SAME_DIR:
            return None

        # Reject if volume surge suggests real breakdown
        if i >= 5:
            recent_vol = np.mean(ind["volumes"][i-5:i])
            if ind["volumes"][i] > recent_vol * 3.0:
                return None

        # SL: below extension zone
        sl = close - atr_val * cfg.MR_SL_ATR_MULT
        # TP: BB midline
        tp = bb_mid
        entry = close

        risk = entry - sl
        reward = tp - entry
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < cfg.MR_MIN_RR:
            return None

        return {
            "direction": "LONG",
            "entry_price": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "rr": float(rr),
            "rsi": float(rsi_val),
            "adx": float(adx_val),
            "vwap_dev_pct": float(vwap_dev),
            "bb_distance_pct": float((bb_lower - close) / close * 100.0),
            "bar_idx": i,
        }

    # SHORT: price above upper BB, RSI overbought
    if close > bb_upper:
        if rsi_val < cfg.MR_RSI_OVERBOUGHT:
            return None

        vwap_dev = (close - vwap_val) / vwap_val * 100.0 if vwap_val > 0 else 0
        if vwap_dev < cfg.MR_VWAP_MIN_DEVIATION_PCT:
            return None

        if consecutive_losses.get("SHORT", 0) >= cfg.MR_MAX_CONSECUTIVE_LOSSES_SAME_DIR:
            return None

        if i >= 5:
            recent_vol = np.mean(ind["volumes"][i-5:i])
            if ind["volumes"][i] > recent_vol * 3.0:
                return None

        sl = close + atr_val * cfg.MR_SL_ATR_MULT
        tp = bb_mid
        entry = close

        risk = sl - entry
        reward = entry - tp
        if risk <= 0:
            return None
        rr = reward / risk
        if rr < cfg.MR_MIN_RR:
            return None

        return {
            "direction": "SHORT",
            "entry_price": float(entry),
            "sl": float(sl),
            "tp": float(tp),
            "rr": float(rr),
            "rsi": float(rsi_val),
            "adx": float(adx_val),
            "vwap_dev_pct": float(vwap_dev),
            "bb_distance_pct": float((close - bb_upper) / close * 100.0),
            "bar_idx": i,
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MeanReversionAgent (live agent wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class MeanReversionAgent:
    """
    Mean Reversion Agent for the multi-agent system.
    Only active when regime = RANGE.
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

        self.daily_trades = 0
        self.daily_pnl = 0.0
        self.daily_wins = 0
        self.daily_losses = 0
        self.last_daily_reset = ""
        self.starting_equity = 0.0
        self._consecutive_losses = {"LONG": 0, "SHORT": 0}

        self._trade_lock: asyncio.Lock = asyncio.Lock()
        self._symbols: list[str] = []
        self._last_symbols_refresh: float = 0.0
        self._markets_cache: dict = {}

    async def heartbeat(self) -> None:
        await self.state.record_heartbeat(self.agent_id)

    async def _reset_daily_if_needed(self) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.last_daily_reset != today:
            self.daily_trades = 0
            self.daily_pnl = 0.0
            self.daily_wins = 0
            self.daily_losses = 0
            self.last_daily_reset = today
            self._consecutive_losses = {"LONG": 0, "SHORT": 0}
            equity = await self.state.get_agent_equity(self.agent_id)
            if equity > 0:
                self.starting_equity = equity
            log.info("[MR] Daily stats reset for %s", today)

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
                (sym, t) for sym, t in tickers.items()
                if sym.endswith(":USDT") and float(t.get("quoteVolume") or 0) >= min_vol
            ]
            usdt_perp.sort(key=lambda x: float(x[1].get("quoteVolume") or 0), reverse=True)
            self._symbols = [sym for sym, _ in usdt_perp[:cfg.TOP_N_COINS]]
            self._last_symbols_refresh = now
            self._markets_cache = dict(self.client._exchange.markets or {})
            log.info("[MR] %d symbols loaded.", len(self._symbols))
        except Exception as exc:
            log.error("[MR] Symbol refresh failed: %s", exc)

    async def _check_open_positions(self) -> None:
        positions = await self.state.get_agent_positions(self.agent_id)
        for symbol, pos in list(positions.items()):
            try:
                ticker = await self.client.fetch_ticker(symbol)
                price = float(ticker.get("last") or 0)
                if price <= 0:
                    continue
                side = str(pos.get("side", "LONG"))
                sl = float(pos.get("sl", 0))
                tp = float(pos.get("tp", 0))
                hit = False
                reason = ""
                if side == "LONG":
                    if sl > 0 and price <= sl:
                        hit, reason = True, "SL"
                    elif tp > 0 and price >= tp:
                        hit, reason = True, "TP"
                else:
                    if sl > 0 and price >= sl:
                        hit, reason = True, "SL"
                    elif tp > 0 and price <= tp:
                        hit, reason = True, "TP"
                if hit:
                    await self._close_position(symbol, pos, price, reason)
            except Exception as exc:
                log.warning("[MR] Position check error %s: %s", symbol, exc)

    async def _close_position(self, symbol: str, pos: dict, exit_price: float, reason: str) -> None:
        await self.state.remove_position(self.agent_id, symbol)
        side = str(pos.get("side", "LONG"))
        entry = float(pos.get("entry", exit_price))
        qty = float(pos.get("qty", 0))
        if side == "LONG":
            pnl_usd = (exit_price - entry) * qty
            pnl_pct = (exit_price - entry) / max(entry, 1e-9) * 100.0
        else:
            pnl_usd = (entry - exit_price) * qty
            pnl_pct = (entry - exit_price) / max(entry, 1e-9) * 100.0

        # Track consecutive losses for anti-revenge
        if pnl_usd < 0:
            self._consecutive_losses[side] = self._consecutive_losses.get(side, 0) + 1
        else:
            self._consecutive_losses[side] = 0

        self.daily_pnl += pnl_usd
        self.daily_trades += 1
        if pnl_usd >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        await self.state.record_trade(self.agent_id, {
            "symbol": symbol, "side": side, "entry": entry, "exit": exit_price,
            "pnl_usd": pnl_usd, "pnl_pct": pnl_pct, "reason": reason,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "closed_at": datetime.now(timezone.utc).isoformat(),
        })
        current = await self.state.get_agent_equity(self.agent_id)
        await self.state.set_agent_equity(self.agent_id, max(0.0, current + pnl_usd))
        short_sym = symbol.split("/")[0]
        log.info("[MR] CLOSED %s %s entry=%.6g exit=%.6g pnl=$%.4f reason=%s",
                 short_sym, side, entry, exit_price, pnl_usd, reason)

    async def run(self) -> None:
        log.info("[MR] Agent starting.")
        await self.state.set_agent_status(self.agent_id, "running")
        while True:
            try:
                await self.heartbeat()
                await self._reset_daily_if_needed()
                await self._refresh_symbols_if_needed()
                await self._check_open_positions()

                status = await self.state.get_agent_status(self.agent_id)
                if status != "running":
                    await asyncio.sleep(cfg.SCAN_INTERVAL_SEC)
                    continue

                for symbol in self._symbols:
                    try:
                        await self._scan_symbol(symbol)
                    except Exception as exc:
                        log.debug("[MR] Scan error %s: %s", symbol, exc)
                    await asyncio.sleep(0.05)

                await asyncio.sleep(cfg.SCAN_INTERVAL_SEC)
            except asyncio.CancelledError:
                log.info("[MR] Agent cancelled.")
                break
            except Exception as exc:
                log.exception("[MR] Main loop error: %s", exc)
                await asyncio.sleep(10)

    async def _scan_symbol(self, symbol: str) -> None:
        if not await self.risk.check_position_count(self.agent_id, self.state, cfg.MR_MAX_POSITIONS):
            return
        if not await self.risk.check_daily_loss_limit(self.agent_id, self.state, cfg.MR_MAX_DAILY_LOSS_PCT):
            return
        if await self.state.is_symbol_locked(symbol):
            return
        # Regime gating enforced at orchestrator level
        pass
