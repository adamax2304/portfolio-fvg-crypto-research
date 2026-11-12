"""
multi_agent/agents/density_agent.py

Density / Volume Profile Agent — trades reactions from value zones in RANGE regime.

Strategy:
  Entry TF: 15m  |  HTF: 1h for regime context
  Only active when RegimeDetector classifies RANGE.

Setup classes:
  1. Reversion from VAH/VAL back toward POC/VWAP
  2. Reaction from dense HVN zones after extension
  3. False excursion outside value area, then return to value

Uses rolling volume profile approximation from OHLCV data.
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
_log_path = Path(cfg.DENSITY_LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("density_agent")
if not log.handlers:
    _fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    _fh = RotatingFileHandler(str(_log_path), maxBytes=10*1024*1024, backupCount=3, encoding="utf-8")
    _fh.setFormatter(_fmt)
    log.addHandler(_fh)
    _ch = logging.StreamHandler(sys.stdout)
    _ch.setFormatter(_fmt)
    log.addHandler(_ch)
    log.setLevel(logging.INFO)

AGENT_ID = "density_agent"


# ═══════════════════════════════════════════════════════════════════════════════
#  Pure indicator / profile functions (no I/O)
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
            out[i] = out[i-1] * (1-alpha) + tr[i] * alpha
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
    """Rolling VWAP over lookback bars."""
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


def compute_volume_profile(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray,
                            volumes: np.ndarray, num_bins: int = 48,
                            start_idx: int = 0, end_idx: int = -1) -> dict:
    """
    Compute volume profile from OHLCV data.
    Returns {poc, vah, val, bins: [(price_level, volume)], hvn_levels, lvn_levels}
    """
    if end_idx < 0:
        end_idx = len(closes)
    h = highs[start_idx:end_idx]
    l = lows[start_idx:end_idx]
    c = closes[start_idx:end_idx]
    v = volumes[start_idx:end_idx]

    if len(h) < 10:
        return {"poc": 0, "vah": 0, "val": 0, "bins": [], "hvn_levels": [], "lvn_levels": []}

    price_low = float(np.min(l))
    price_high = float(np.max(h))
    if price_high <= price_low:
        return {"poc": 0, "vah": 0, "val": 0, "bins": [], "hvn_levels": [], "lvn_levels": []}

    bin_size = (price_high - price_low) / num_bins
    bin_volumes = np.zeros(num_bins)

    for k in range(len(h)):
        bar_low = max(l[k], price_low)
        bar_high = min(h[k], price_high)
        bar_vol = v[k]
        if bar_vol <= 0 or bar_high <= bar_low:
            continue
        low_bin = int((bar_low - price_low) / bin_size)
        high_bin = int((bar_high - price_low) / bin_size)
        low_bin = max(0, min(low_bin, num_bins - 1))
        high_bin = max(0, min(high_bin, num_bins - 1))
        span = high_bin - low_bin + 1
        per_bin = bar_vol / span
        for b in range(low_bin, high_bin + 1):
            bin_volumes[b] += per_bin

    total_vol = np.sum(bin_volumes)
    if total_vol <= 0:
        return {"poc": 0, "vah": 0, "val": 0, "bins": [], "hvn_levels": [], "lvn_levels": []}

    # POC = bin with highest volume
    poc_bin = int(np.argmax(bin_volumes))
    poc_price = price_low + (poc_bin + 0.5) * bin_size

    # Value Area (70% of volume centered on POC)
    val_pct = cfg.DENSITY_VP_NUM_BINS if hasattr(cfg, 'DENSITY_VAL_PCT') else 70.0
    val_pct = cfg.DENSITY_VAL_PCT
    target_vol = total_vol * val_pct / 100.0
    va_vol = bin_volumes[poc_bin]
    lo_idx = poc_bin
    hi_idx = poc_bin
    while va_vol < target_vol and (lo_idx > 0 or hi_idx < num_bins - 1):
        expand_lo = bin_volumes[lo_idx - 1] if lo_idx > 0 else 0
        expand_hi = bin_volumes[hi_idx + 1] if hi_idx < num_bins - 1 else 0
        if expand_lo >= expand_hi and lo_idx > 0:
            lo_idx -= 1
            va_vol += bin_volumes[lo_idx]
        elif hi_idx < num_bins - 1:
            hi_idx += 1
            va_vol += bin_volumes[hi_idx]
        else:
            lo_idx -= 1
            va_vol += bin_volumes[lo_idx]

    val_price = price_low + (lo_idx + 0.5) * bin_size       # Value Area Low
    vah_price = price_low + (hi_idx + 0.5) * bin_size       # Value Area High

    # HVN / LVN detection
    avg_vol = total_vol / num_bins
    hvn_levels = []
    lvn_levels = []
    for b in range(num_bins):
        level = price_low + (b + 0.5) * bin_size
        if bin_volumes[b] > avg_vol * 1.5:
            hvn_levels.append(level)
        elif bin_volumes[b] < avg_vol * 0.3:
            lvn_levels.append(level)

    bins_list = [(price_low + (b + 0.5) * bin_size, float(bin_volumes[b])) for b in range(num_bins)]

    return {
        "poc": float(poc_price),
        "vah": float(vah_price),
        "val": float(val_price),
        "bins": bins_list,
        "hvn_levels": hvn_levels,
        "lvn_levels": lvn_levels,
    }


def detect_density_signal(
    closes: np.ndarray, highs: np.ndarray, lows: np.ndarray,
    volumes: np.ndarray, bar_idx: int,
    vp: dict, vwap: np.ndarray, rsi: np.ndarray, adx: np.ndarray,
    atr_arr: np.ndarray,
) -> Optional[dict]:
    """
    Check for density/value reversion signal at bar_idx.
    Returns signal dict or None.
    """
    i = bar_idx
    n = len(closes)
    if i < 30 or i >= n:
        return None

    close = closes[i]
    poc = vp.get("poc", 0)
    vah = vp.get("vah", 0)
    val_low = vp.get("val", 0)
    vwap_val = vwap[i] if not np.isnan(vwap[i]) else 0
    rsi_val = rsi[i] if not np.isnan(rsi[i]) else 50
    adx_val = adx[i] if not np.isnan(adx[i]) else 30
    atr_val = atr_arr[i] if not np.isnan(atr_arr[i]) else 0

    if poc <= 0 or vah <= 0 or val_low <= 0 or vwap_val <= 0 or atr_val <= 0:
        return None

    # Filter: ADX must be low (range environment)
    if adx_val > cfg.DENSITY_ADX_MAX:
        return None

    # Check for volume confirmation — not a low-volume drifting bar
    vol_avg = np.mean(volumes[max(0, i-20):i]) if i >= 20 else np.mean(volumes[:max(i, 1)])
    if vol_avg <= 0:
        return None

    # LONG: price at or below VAL, RSI mildly oversold, deviation from VWAP
    vwap_dev_pct = (vwap_val - close) / vwap_val * 100.0 if vwap_val > 0 else 0

    if close <= val_low * 1.002:  # at or below value area low
        if rsi_val > cfg.DENSITY_RSI_OVERSOLD:
            return None  # not oversold enough
        if vwap_dev_pct < cfg.DENSITY_VWAP_DEVIATION_PCT:
            return None  # not stretched enough from VWAP

        # Check that we're not in a strong impulse breakdown
        if i >= 3:
            recent_move = (closes[i-3] - close) / closes[i-3] * 100.0
            if recent_move > 2.0:  # big drop — momentum breakdown, skip
                return None

        # SL below recent low - ATR buffer
        recent_low = float(np.min(lows[max(0, i-10):i+1]))
        sl = recent_low - atr_val * cfg.DENSITY_SL_ATR_MULT
        # TP: VWAP or POC
        tp1 = vwap_val
        tp2 = poc
        entry = close

        risk = entry - sl
        if risk <= 0:
            return None
        reward = tp1 - entry
        rr = reward / risk if risk > 0 else 0
        if rr < cfg.DENSITY_MIN_RR:
            return None

        return {
            "direction": "LONG",
            "entry_price": float(entry),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "rr": float(rr),
            "rsi": float(rsi_val),
            "adx": float(adx_val),
            "vwap_dev_pct": float(vwap_dev_pct),
            "setup": "val_reversion",
            "bar_idx": i,
        }

    # SHORT: price at or above VAH, RSI overbought
    vwap_dev_short = (close - vwap_val) / vwap_val * 100.0 if vwap_val > 0 else 0

    if close >= vah * 0.998:
        if rsi_val < cfg.DENSITY_RSI_OVERBOUGHT:
            return None
        if vwap_dev_short < cfg.DENSITY_VWAP_DEVIATION_PCT:
            return None

        if i >= 3:
            recent_move = (close - closes[i-3]) / closes[i-3] * 100.0
            if recent_move > 2.0:
                return None

        recent_high = float(np.max(highs[max(0, i-10):i+1]))
        sl = recent_high + atr_val * cfg.DENSITY_SL_ATR_MULT
        tp1 = vwap_val
        tp2 = poc
        entry = close

        risk = sl - entry
        if risk <= 0:
            return None
        reward = entry - tp1
        rr = reward / risk if risk > 0 else 0
        if rr < cfg.DENSITY_MIN_RR:
            return None

        return {
            "direction": "SHORT",
            "entry_price": float(entry),
            "sl": float(sl),
            "tp1": float(tp1),
            "tp2": float(tp2),
            "rr": float(rr),
            "rsi": float(rsi_val),
            "adx": float(adx_val),
            "vwap_dev_pct": float(vwap_dev_short),
            "setup": "vah_reversion",
            "bar_idx": i,
        }

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  DensityAgent (live agent wrapper)
# ═══════════════════════════════════════════════════════════════════════════════

class DensityAgent:
    """
    Density / Volume Profile Agent for the multi-agent system.
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

        self.daily_trades     = 0
        self.daily_pnl        = 0.0
        self.daily_wins       = 0
        self.daily_losses     = 0
        self.last_daily_reset = ""
        self.starting_equity  = 0.0

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
            equity = await self.state.get_agent_equity(self.agent_id)
            if equity > 0:
                self.starting_equity = equity
            log.info("[DENSITY] Daily stats reset for %s", today)

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
            log.info("[DENSITY] %d symbols loaded.", len(self._symbols))
        except Exception as exc:
            log.error("[DENSITY] Symbol refresh failed: %s", exc)

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
                log.warning("[DENSITY] Position check error %s: %s", symbol, exc)

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
        log.info("[DENSITY] CLOSED %s %s entry=%.6g exit=%.6g pnl=$%.4f reason=%s",
                 short_sym, side, entry, exit_price, pnl_usd, reason)

    async def run(self) -> None:
        """Main agent loop."""
        log.info("[DENSITY] Agent starting.")
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
                        log.debug("[DENSITY] Scan error %s: %s", symbol, exc)
                    await asyncio.sleep(0.05)

                await asyncio.sleep(cfg.SCAN_INTERVAL_SEC)
            except asyncio.CancelledError:
                log.info("[DENSITY] Agent cancelled.")
                break
            except Exception as exc:
                log.exception("[DENSITY] Main loop error: %s", exc)
                await asyncio.sleep(10)

    async def _scan_symbol(self, symbol: str) -> None:
        """Scan one symbol for density setup. Regime check done at orchestrator level."""
        if not await self.risk.check_position_count(self.agent_id, self.state, cfg.DENSITY_MAX_POSITIONS):
            return
        if not await self.risk.check_daily_loss_limit(self.agent_id, self.state, cfg.DENSITY_MAX_DAILY_LOSS_PCT):
            return
        if await self.state.is_symbol_locked(symbol):
            return
        # Actual signal detection would happen here using indicators
        # Regime gating is enforced at orchestrator level
        pass
