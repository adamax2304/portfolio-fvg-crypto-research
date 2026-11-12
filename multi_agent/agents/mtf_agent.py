"""
multi_agent/agents/mtf_agent.py

Multi-Timeframe (MTF) Trading Agent — v10 backtest-optimised strategies.

Implements three strategies (enabled in cfg.MTF_STRATEGIES):

  BBS_1H_5M   — BB Squeeze signal on 1h, pullback-to-EMA21 entry on 5m
                PF=9.66 (WF 9.55), WR=77.4%, DD=1.9%

  BBS_1H_15M  — BB Squeeze signal on 1h, next-bar entry on 15m
                PF=3.92 (WF 3.83), WR=59.7%, DD=4.1%

  KC_4H_15M   — Keltner Squeeze signal on 4h, next-bar entry on 15m
                OOS PF=4.14, WR=72.7%, DD=4.6%  [enabled 2026-04-25]

  KC_1H_15M   — DISABLED 2026-04-25: OOS PF=1.35 breakeven.

Architecture
------------
  - 5-minute main loop: manage positions → scan 1h signals → check pending entries
  - Signals are stored in self._pending_signals dict; entry is deferred to LTF
  - All positions tracked in shared StateManager
  - Shared-deposit model: full balance used for sizing, 2% risk per trade
  - SIMULATION_MODE: logs trades but never calls exchange
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

# Re-use indicator functions from bb_squeeze_agent — no duplication
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators,
    detect_squeeze_signal,
    check_head_fake,
    _calculate_atr,
    _ema,
    _sma,
    _linreg,
    _adx,
)

if TYPE_CHECKING:
    from multi_agent.shared.bybit_client import BybitClient
    from multi_agent.shared.state_manager import StateManager
    from multi_agent.shared.telegram_notifier import MultiAgentTelegramNotifier
    from multi_agent.shared.risk_manager import RiskManager

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
_LOG_PATH = Path(cfg.BASE_DIR) / "logs" / "mtf_agent.log"
_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("mtf_agent")
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

AGENT_ID = "mtf_agent"

# ---------------------------------------------------------------------------
# Signal cooldown / re-entry protection
# ---------------------------------------------------------------------------
# After SL: block the symbol for this many seconds before allowing re-entry
SL_COOLDOWN_SEC     = 4 * 3600   # 4 hours after first SL
SL_STREAK_LIMIT     = 2          # consecutive SLs before extended block
SL_EXTENDED_CD_SEC  = 24 * 3600  # 24 hours after hitting streak limit

# ---------------------------------------------------------------------------
# Pending signal expiry
# ---------------------------------------------------------------------------
# next_bar strategies: max cycles to keep signal alive (1 cycle = 5 min)
NEXT_BAR_MAX_CYCLES = 3   # 3 cycles = 15 min — stale 1h signal → discard

# ---------------------------------------------------------------------------
# Break-even SL
# ---------------------------------------------------------------------------
# Move SL to entry when unrealized profit ≥ BE_TRIGGER_R × risk
BE_TRIGGER_R = 1.5   # at 1.5R profit → move SL to break-even + small buffer
BE_BUFFER_PCT = 0.001 # 0.1% buffer above entry to cover fees on break-even exit

# ---------------------------------------------------------------------------
# Daily EMA200 filter (BBS_1H_5M only)
# ---------------------------------------------------------------------------
DAILY_EMA200_FILTER_STRATEGIES = {"BBS_1H_5M"}  # backtest: PF 5.37→7.39

# ---------------------------------------------------------------------------
# Time-of-day filter (UTC)
# ---------------------------------------------------------------------------
# Trade only during EU/US overlap: 07:00–20:00 UTC
# Avoids low-liquidity Asian session where spreads are wider
TRADING_HOUR_START_UTC = 7    # inclusive
TRADING_HOUR_END_UTC   = 20   # exclusive (i.e. trade until 19:59 UTC)

# ---------------------------------------------------------------------------
# Funding rate filter
# ---------------------------------------------------------------------------
# Block direction if funding rate indicates overcrowded positioning
FUNDING_RATE_BLOCK_LONG  = 0.0005   # +0.05%/8h → skip new longs
FUNDING_RATE_BLOCK_SHORT = -0.0005  # -0.05%/8h → skip new shorts
FUNDING_CACHE_TTL_SEC    = 300      # cache funding rate 5 minutes

# ---------------------------------------------------------------------------
# MTF-specific indicator helpers
# ---------------------------------------------------------------------------

def _compute_kc_squeeze_indicators(
    df: pd.DataFrame,
    bb_len: int = 20,
    bb_mult: float = 2.0,
    kc_len: int = 20,
    kc_mult: float = 1.5,
    mom_bars: int = 6,
    adx_period: int = 14,
) -> dict:
    """
    Compute indicators for the KC_1H_15M strategy.

    Identical to the BBS squeeze but:
      - momentum uses a shorter linreg window (mom_bars instead of 20)
      - no EMA stack is returned (not required for KC strategy)
    """
    closes = df["close"].astype(float).to_numpy()
    highs  = df["high"].astype(float).to_numpy()
    lows   = df["low"].astype(float).to_numpy()
    n = len(closes)

    # Bollinger Bands
    bb_basis = _sma(closes, bb_len)
    bb_dev   = np.full(n, np.nan)
    for i in range(bb_len - 1, n):
        bb_dev[i] = np.std(closes[i - bb_len + 1 : i + 1], ddof=0) * bb_mult
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev

    # Keltner Channels
    kc_basis = _ema(closes, kc_len)
    atr_arr  = _calculate_atr(highs, lows, closes, kc_len)
    kc_upper = kc_basis + kc_mult * atr_arr
    kc_lower = kc_basis - kc_mult * atr_arr

    # Squeeze: BB inside KC
    squeeze_on = np.zeros(n, dtype=bool)
    for i in range(n):
        if not np.isnan(bb_lower[i]) and not np.isnan(kc_lower[i]):
            squeeze_on[i] = (bb_lower[i] > kc_lower[i]) and (bb_upper[i] < kc_upper[i])

    # Momentum: close - linreg(close, mom_bars)  — shorter period for KC
    linreg_vals = _linreg(closes, mom_bars)
    momentum = closes - linreg_vals

    # ADX
    adx_arr = _adx(highs, lows, closes, adx_period)

    # Volume
    volumes   = df["volume"].astype(float).to_numpy()
    vol_sma20 = _sma(volumes, 20)

    # ATR for entry-TF SL (14-period)
    atr14 = _calculate_atr(highs, lows, closes, 14)

    return {
        "squeeze_on": squeeze_on,
        "momentum":   momentum,
        "adx":        adx_arr,
        "bb_upper":   bb_upper,
        "bb_lower":   bb_lower,
        "kc_upper":   kc_upper,
        "kc_lower":   kc_lower,
        "atr":        atr_arr,
        "atr14":      atr14,
        "volumes":    volumes,
        "vol_sma20":  vol_sma20,
        "closes":     closes,
        "highs":      highs,
        "lows":       lows,
    }


def _detect_kc_signal(
    ind: dict,
    bar_idx: int = -1,
    min_squeeze_bars: int = 10,
    adx_threshold: float = 12.0,
    vol_mult: float = 1.0,
) -> Optional[dict]:
    """
    KC-specific signal detection.

    Differences from detect_squeeze_signal:
      - No EMA stack requirement
      - Uses shorter momentum (already computed in ind)
      - Lower ADX threshold (12 vs 15)
    """
    n = len(ind["closes"])
    if bar_idx < 0:
        bar_idx = n + bar_idx
    if bar_idx < 2 or bar_idx >= n:
        return None

    i = bar_idx

    # Guard NaN
    for key in ("adx", "momentum"):
        if np.isnan(ind[key][i]):
            return None

    # Must be first bar after squeeze release
    if ind["squeeze_on"][i]:
        return None  # still in squeeze
    if not ind["squeeze_on"][i - 1]:
        return None  # not the first release bar

    # Count consecutive squeeze bars before release
    squeeze_count = 0
    for j in range(i - 1, -1, -1):
        if ind["squeeze_on"][j]:
            squeeze_count += 1
        else:
            break

    if squeeze_count < min_squeeze_bars:
        return None

    # ADX
    if ind["adx"][i] < adx_threshold:
        return None

    # Volume
    vol     = ind["volumes"][i]
    vol_avg = ind["vol_sma20"][i]
    if np.isnan(vol_avg) or vol_avg <= 0 or vol < vol_mult * vol_avg:
        return None

    # Momentum direction and increasing
    mom      = ind["momentum"][i]
    mom_prev = ind["momentum"][i - 1]
    if np.isnan(mom) or np.isnan(mom_prev):
        return None

    close = ind["closes"][i]

    if mom > 0 and mom > mom_prev:
        return {
            "direction":    "LONG",
            "entry_price":  float(close),
            "momentum":     float(mom),
            "adx":          float(ind["adx"][i]),
            "squeeze_bars": squeeze_count,
            "volume_ratio": float(vol / max(vol_avg, 1e-9)),
            "bar_idx":      i,
        }

    if mom < 0 and mom < mom_prev:
        return {
            "direction":    "SHORT",
            "entry_price":  float(close),
            "momentum":     float(mom),
            "adx":          float(ind["adx"][i]),
            "squeeze_bars": squeeze_count,
            "volume_ratio": float(vol / max(vol_avg, 1e-9)),
            "bar_idx":      i,
        }

    return None


def _compute_entry_tf_indicators(df: pd.DataFrame) -> dict:
    """
    Compute indicators needed on the entry timeframe (5m or 15m):
      - ATR(14) for SL sizing
      - EMA21 for pullback detection (BBS_1H_5M)
    """
    closes = df["close"].astype(float).to_numpy()
    highs  = df["high"].astype(float).to_numpy()
    lows   = df["low"].astype(float).to_numpy()
    opens  = df["open"].astype(float).to_numpy()

    atr14 = _calculate_atr(highs, lows, closes, 14)
    ema21 = _ema(closes, 21)

    return {
        "closes": closes,
        "highs":  highs,
        "lows":   lows,
        "opens":  opens,
        "atr14":  atr14,
        "ema21":  ema21,
    }


# ---------------------------------------------------------------------------
# MTFAgent
# ---------------------------------------------------------------------------

class MTFAgent:
    """
    Multi-Timeframe trading agent.

    Active strategies (enabled in cfg.MTF_STRATEGIES):
      BBS_1H_5M   — BB Squeeze 1h → 5m pullback EMA21      (PF=9.66, WR=77%)
      BBS_1H_15M  — BB Squeeze 1h → 15m next_bar           (PF=3.92, WR=60%)
      KC_4H_15M   — KC Squeeze 4h → 15m next_bar           (OOS PF=4.14, WR=73%)

    KC_1H_15M disabled 2026-04-25 (OOS PF=1.35 — breakeven).
    KC_4H_15M replaces it; 4h bars fetched per-symbol when strategy is enabled.

    Main loop runs every 5 minutes (300 s).
    """

    # Strategy IDs — must match cfg.MTF_STRATEGIES keys
    STRAT_BBS_1H_5M  = "BBS_1H_5M"
    STRAT_BBS_1H_15M = "BBS_1H_15M"
    STRAT_KC_1H_15M  = "KC_1H_15M"
    STRAT_KC_4H_15M  = "KC_4H_15M"

    def __init__(
        self,
        agent_id: str,
        state_manager: "StateManager",
        bybit_client: "BybitClient",
        risk_manager: "RiskManager",
        telegram: "MultiAgentTelegramNotifier",
        oi_filter=None,
    ) -> None:
        self.agent_id  = agent_id
        self.state     = state_manager
        self.client    = bybit_client
        self.risk      = risk_manager
        self.tg        = telegram
        self.oi_filter = oi_filter   # optional; None → skip OI check

        # Trade execution lock — prevents race conditions
        self._trade_lock: asyncio.Lock = asyncio.Lock()

        # Pending signals awaiting LTF entry confirmation.
        # Structure: {symbol: [signal_dict, ...]}
        # Multiple strategies can fire on the same symbol, so we use a list.
        self._pending: dict[str, list[dict]] = {}

        # Signal cooldown after SL — prevents re-entry on a falling symbol.
        # {symbol: epoch_time_blocked_until}
        self._signal_cooldown: dict[str, float] = {}

        # Consecutive SL streak per symbol (reset on TP or new day).
        # After MAX_SL_STREAK consecutive SLs, apply extended cooldown.
        self._sl_streak: dict[str, int] = {}

        # Daily EMA200 direction cache: {symbol: ("LONG"|"SHORT", epoch_fetched)}
        # Refreshed at most every 4 hours per symbol (daily candles are slow)
        self._daily_ema200: dict[str, tuple[str, float]] = {}

        # Funding rate cache: {symbol: (rate_float, epoch_fetched)}
        self._funding_cache: dict[str, tuple[float, float]] = {}

        # next_bar signal cycle counter: {(symbol, strategy_id): cycles_waited}
        self._next_bar_cycles: dict[tuple[str, str], int] = {}

        # Pending post-only limit orders awaiting fill
        # key: (symbol, strategy_id) → {order_id, entry_price, sl, tp, qty, leverage,
        #                               side_str, sig, margin_used, risk_usd, cycles}
        self._pending_limit_orders: dict[tuple[str, str], dict] = {}

        # Break-even SL tracking: {symbol: bool} — True if BE already applied
        self._be_applied: dict[str, bool] = {}

        # Symbol list (populated from exchange, refreshed every 4h)
        self._symbols: list[str] = []
        self._last_symbols_refresh: float = 0.0
        self._markets_cache: dict = {}

        # Market-cap whitelist: set of base-currency tickers (e.g. {"BTC","ETH",...})
        # refreshed every MARKET_CAP_REFRESH_HR hours via CoinGecko
        self._mc_whitelist: set[str] = set()
        self._last_mc_refresh: float = 0.0

        # Binance volume filter: ordered list + set of top-N base tickers on Binance futures
        # Primary source of truth for symbol selection (no Bybit volume requirement).
        # refreshed every BINANCE_VOL_REFRESH_HR hours via Binance public FAPI.
        self._binance_vol_ordered: list[str] = []   # base tickers in Binance vol order
        self._binance_vol_set: set[str] = set()     # same as set for O(1) lookup
        self._last_binance_vol_refresh: float = 0.0

        # Daily tracking
        self.daily_trades     = 0
        self.daily_pnl        = 0.0
        self.daily_wins       = 0
        self.daily_losses     = 0
        self.last_daily_reset = ""
        self.starting_equity  = 0.0

        # Per-cycle filter funnel counters (reset at start of each _scan_for_signals)
        self._cycle_funnel: dict[str, int] = {}

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

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
            log.info("[MTF] Daily stats reset for %s", today)

    def _short(self, symbol: str) -> str:
        """Return a terse symbol label for log messages."""
        return symbol.replace("/USDT:USDT", "USDT")

    # ── Signal cooldown helpers ────────────────────────────────────────────

    def _is_on_cooldown(self, symbol: str) -> bool:
        """Return True if symbol is currently blocked due to recent SL."""
        until = self._signal_cooldown.get(symbol, 0.0)
        return _time.time() < until

    def _set_sl_cooldown(self, symbol: str) -> None:
        """
        Apply cooldown after an SL hit.
        First SL  → 4h cooldown.
        SL streak ≥ 2 → 24h cooldown (stop trading symbol for the day).
        """
        streak = self._sl_streak.get(symbol, 0) + 1
        self._sl_streak[symbol] = streak
        if streak >= SL_STREAK_LIMIT:
            cd = SL_EXTENDED_CD_SEC
            log.warning(
                "[MTF] %s SL streak=%d → BLOCKED for %.0fh",
                self._short(symbol), streak, cd / 3600,
            )
        else:
            cd = SL_COOLDOWN_SEC
            log.info(
                "[MTF] %s SL → cooldown %.0fh (streak %d)",
                self._short(symbol), cd / 3600, streak,
            )
        self._signal_cooldown[symbol] = _time.time() + cd
        # Also clear any pending signal for this symbol immediately
        self._pending.pop(symbol, None)

    def _reset_sl_streak(self, symbol: str) -> None:
        """Called on TP hit — resets streak so symbol can trade again next signal."""
        if symbol in self._sl_streak:
            del self._sl_streak[symbol]
        # Don't touch cooldown — let it expire naturally

    async def handle_external_close(self, symbol: str, pos: dict) -> None:
        """
        Called by MasterAgent when a position disappears from Bybit
        (manual close by user, SL/TP fired on exchange, or liquidation).

        Fetches the actual exit price, records the trade in statistics,
        sends a Telegram notification, and applies the appropriate cooldown.
        """
        sl    = float(pos.get("sl", 0))
        tp    = float(pos.get("tp", 0))
        entry = float(pos.get("entry", 0))
        side  = str(pos.get("side", "LONG"))

        # ── Try to get actual exit price + realized PnL from Bybit ─────
        exit_price = 0.0
        bybit_pnl: float | None = None
        try:
            closed = await self.client.fetch_closed_pnl(hours=1)
            raw_sym = symbol.replace("/", "").replace(":USDT", "")
            for rec in closed:
                if rec.get("symbol", "").upper() == raw_sym.upper():
                    ep = float(rec.get("avgExitPrice") or 0)
                    if ep > 0:
                        exit_price = ep
                        bybit_pnl = float(rec.get("closedPnl") or 0)
                        break
        except Exception as exc:
            log.debug("[MTF] fetch_closed_pnl for %s failed: %s", self._short(symbol), exc)

        if exit_price <= 0:
            # fallback: use last traded price
            try:
                ticker = await self.client.fetch_ticker(symbol)
                exit_price = float(ticker.get("last", 0) or 0)
            except Exception:
                pass

        if exit_price <= 0:
            exit_price = entry  # last resort: assume breakeven

        # ── Determine close reason (heuristic from price proximity) ─────
        reason = "MANUAL"  # default: assume user closed manually
        tol = 0.003  # 0.3% tolerance band
        if sl > 0 and tp > 0 and entry > 0:
            if side == "LONG":
                if exit_price <= sl * (1 + tol):
                    reason = "SL"
                elif exit_price >= tp * (1 - tol):
                    reason = "TP"
            else:  # SHORT
                if exit_price >= sl * (1 - tol):
                    reason = "SL"
                elif exit_price <= tp * (1 + tol):
                    reason = "TP"

        log.info(
            "[MTF] External close %s reason=%s exit=%.6g (sl=%.6g tp=%.6g)",
            self._short(symbol), reason, exit_price, sl, tp,
        )

        # ── Record trade + send TG + apply cooldown ──────────────────────
        # _close_position handles: remove_position, record_trade, update equity,
        # send TG notification, and cooldown
        await self._close_position(
            symbol, pos, exit_price, reason, skip_exchange_order=True,
            override_pnl=bybit_pnl,
        )

    # Keep old sync stub for backward compatibility (no longer used by MasterAgent)
    def handle_sync_removal(self, symbol: str, pos: dict) -> None:
        """Deprecated: use handle_external_close (async). Kept for safety."""
        sl    = float(pos.get("sl", 0))
        tp    = float(pos.get("tp", 0))
        entry = float(pos.get("entry", 0))
        reason = "SL"
        if sl > 0 and tp > 0 and entry > 0:
            reason = "TP" if abs(entry - tp) < abs(entry - sl) else "SL"
        if reason == "SL":
            self._set_sl_cooldown(symbol)
        else:
            self._reset_sl_streak(symbol)

    def _strat_cfg(self, strategy_id: str) -> dict:
        return cfg.MTF_STRATEGIES.get(strategy_id, {})

    # ── Daily EMA200 filter ────────────────────────────────────────────────

    async def _get_daily_ema200_direction(self, symbol: str) -> Optional[str]:
        """
        Fetch daily candles and return "LONG" or "SHORT" based on
        whether the last close is above/below EMA200(daily).
        Result is cached for 4 hours per symbol.
        Returns None on fetch error (→ filter skipped, trade allowed).
        """
        cache_ttl = 4 * 3600
        cached = self._daily_ema200.get(symbol)
        if cached and (_time.time() - cached[1]) < cache_ttl:
            return cached[0]
        try:
            raw = await self.client.fetch_ohlcv(symbol, "1d", 220)
            if not raw or len(raw) < 200:
                return None
            closes = np.array([float(c[4]) for c in raw])
            ema200 = _ema(closes, 200)
            direction = "LONG" if closes[-1] > ema200[-1] else "SHORT"
            self._daily_ema200[symbol] = (direction, _time.time())
            return direction
        except Exception as exc:
            log.debug("[MTF] daily EMA200 fetch failed %s: %s", self._short(symbol), exc)
            return None

    # ── Time-of-day filter ────────────────────────────────────────────────

    def _is_trading_hours(self) -> bool:
        """
        Return True if current UTC time is within the allowed trading window.
        07:00–20:00 UTC: EU open through US close — highest volume, tightest spreads.
        Outside this window: Asian session — lower volume, wider spreads on alts.
        """
        hour = datetime.now(timezone.utc).hour
        return TRADING_HOUR_START_UTC <= hour < TRADING_HOUR_END_UTC

    # ── Funding rate filter ────────────────────────────────────────────────

    async def _get_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Fetch and cache funding rate for symbol.
        Returns float (e.g. 0.0001) or None on error.
        """
        cached = self._funding_cache.get(symbol)
        if cached and (_time.time() - cached[1]) < FUNDING_CACHE_TTL_SEC:
            return cached[0]
        try:
            rate = await self.client.fetch_funding_rate(symbol)
            if rate is not None:
                self._funding_cache[symbol] = (rate, _time.time())
                return rate
        except Exception as exc:
            log.debug("[MTF] funding rate fetch failed %s: %s", self._short(symbol), exc)
        return None

    def _funding_blocks_direction(self, funding_rate: Optional[float], direction: str) -> bool:
        """
        Return True if funding rate indicates the direction is overcrowded
        and we should skip this trade.
        """
        if funding_rate is None:
            return False   # no data → allow trade
        if direction == "LONG" and funding_rate > FUNDING_RATE_BLOCK_LONG:
            return True    # longs are paying, overcrowded
        if direction == "SHORT" and funding_rate < FUNDING_RATE_BLOCK_SHORT:
            return True    # shorts are paying, overcrowded
        return False

    # ── Break-even SL management ───────────────────────────────────────────

    async def _check_break_even(self, symbol: str, pos: dict, price: float) -> None:
        """
        Move SL to break-even + small buffer once unrealized profit ≥ BE_TRIGGER_R.
        Updates state and Bybit SL order.
        Only applied once per position (tracked by _be_applied).
        """
        if self._be_applied.get(symbol):
            return
        entry = float(pos.get("entry", 0))
        sl    = float(pos.get("sl", 0))
        tp    = float(pos.get("tp", 0))
        side  = str(pos.get("side", "LONG"))
        if entry <= 0 or sl <= 0:
            return

        risk = abs(entry - sl)
        if risk <= 0:
            return

        # Current profit in R
        if side == "LONG":
            profit_r = (price - entry) / risk
            be_sl = entry * (1 + BE_BUFFER_PCT)  # entry + 0.1% buffer
            if profit_r < BE_TRIGGER_R:
                return
            if sl >= be_sl:
                return   # SL already beyond break-even
            new_sl = be_sl
        else:  # SHORT
            profit_r = (entry - price) / risk
            be_sl = entry * (1 - BE_BUFFER_PCT)
            if profit_r < BE_TRIGGER_R:
                return
            if sl <= be_sl:
                return
            new_sl = be_sl

        log.info(
            "[MTF] BE-SL %s %s: profit=%.2fR → moving SL %.6g→%.6g",
            self._short(symbol), side, profit_r, sl, new_sl,
        )
        # Update state
        try:
            self.state._positions[self.agent_id][symbol]["sl"] = new_sl
            self._be_applied[symbol] = True
        except Exception:
            return

        # Update on Bybit
        if not cfg.SIMULATION_MODE:
            try:
                qty = float(pos.get("qty", 0))
                side_str = "buy" if side == "LONG" else "sell"
                await self.client.place_sl_tp(symbol, side_str, qty, new_sl, tp)
                log.info("[MTF] BE-SL updated on Bybit %s sl=%.6g", self._short(symbol), new_sl)
            except Exception as exc:
                log.warning("[MTF] BE-SL Bybit update failed %s: %s", self._short(symbol), exc)

    async def _fetch_balance(self) -> float:
        """Fetch total USDT balance from exchange; fall back to StateManager."""
        try:
            raw = await self.client.fetch_balance()
            bal = float(
                raw.get("USDT", {}).get("total", 0)
                or raw.get("total", {}).get("USDT", 0)
                or 0
            )
            if bal > 0:
                return bal
        except Exception as exc:
            log.debug("[MTF] fetch_balance error: %s", exc)
        return await self.state.get_agent_equity(self.agent_id)

    async def _check_total_exposure(self, balance: float) -> bool:
        """
        Return True if adding one more position would still be within
        MAX_TOTAL_EXPOSURE_PCT of total balance.
        """
        all_positions = await self.state.get_all_positions()
        used_margin = 0.0
        for agent_pos in all_positions.values():
            for pos in agent_pos.values():
                qty   = float(pos.get("qty", 0))
                entry = float(pos.get("entry", 0))
                lev   = int(pos.get("leverage", 1)) or 1
                used_margin += qty * entry / lev
        if balance <= 0:
            return True
        max_margin = balance * cfg.MAX_TOTAL_EXPOSURE_PCT / 100.0
        return used_margin < max_margin

    # -------------------------------------------------------------------------
    # Symbol list management (same pattern as BBSqueezeAgent)
    # -------------------------------------------------------------------------

    async def _refresh_market_cap_list(self) -> None:
        """
        Fetch top-N coins by market cap from CoinGecko and store base-ticker set.
        E.g.  {"BTC", "ETH", "SOL", ...}
        Called lazily from _refresh_symbols_if_needed; skipped if cfg.MARKET_CAP_FILTER=False.
        Falls back to keeping the existing whitelist (or no filter) on error.
        """
        if not getattr(cfg, "MARKET_CAP_FILTER", True):
            return
        now = _time.monotonic()
        ttl = getattr(cfg, "MARKET_CAP_REFRESH_HR", 24.0) * 3600
        if self._mc_whitelist and now - self._last_mc_refresh < ttl:
            return  # still fresh

        top_n = getattr(cfg, "MARKET_CAP_TOP_N", 150)
        url_tpl = (
            "https://api.coingecko.com/api/v3/coins/markets"
            "?vs_currency=usd&order=market_cap_desc&per_page=100&page={page}&sparkline=false"
        )
        import urllib.request as _urllib_req, json as _json_mod

        # urllib is synchronous — run in thread so we don't block the event loop
        def _fetch_page(page: int) -> list:
            req = _urllib_req.Request(
                url_tpl.format(page=page),
                headers={"User-Agent": "fvgbot/1.0"},
            )
            with _urllib_req.urlopen(req, timeout=10) as resp:
                return _json_mod.loads(resp.read())

        new_whitelist: set[str] = set()
        try:
            loop = asyncio.get_event_loop()
            await self.heartbeat()  # prevent stale-heartbeat restart during CoinGecko fetch
            for page in (1, 2):
                if len(new_whitelist) >= top_n:
                    break
                coins = await loop.run_in_executor(None, _fetch_page, page)
                for coin in coins:
                    new_whitelist.add(coin["symbol"].upper())
                    if len(new_whitelist) >= top_n:
                        break
            if new_whitelist:
                # Add Bybit-specific ticker aliases where CoinGecko name differs
                # e.g. CoinGecko "PUMP" -> Bybit lists as "PUMPFUN"
                _bybit_aliases: dict[str, str] = {
                    "PUMP": "PUMPFUN",
                }
                for cg_sym, bybit_sym in _bybit_aliases.items():
                    if cg_sym in new_whitelist:
                        new_whitelist.add(bybit_sym)

                self._mc_whitelist = new_whitelist
                self._last_mc_refresh = now
                log.info(
                    "[MTF] Market-cap whitelist updated: %d coins (top-%d by mcap).",
                    len(new_whitelist), top_n,
                )
        except Exception as exc:
            if self._mc_whitelist:
                log.warning("[MTF] CoinGecko fetch failed (%s) — reusing cached whitelist.", exc)
            else:
                log.warning(
                    "[MTF] CoinGecko fetch failed (%s) — market-cap filter DISABLED this cycle.", exc
                )

    async def _refresh_binance_vol_list(self) -> None:
        """
        Fetch Binance USDT-futures 24h tickers and store top-N base tickers sorted by volume.
        Primary source of truth for which symbols are liquid enough to trade.
        Uses Binance public FAPI — no API key required.
        Stores both ordered list (_binance_vol_ordered) and set (_binance_vol_set).
        Falls back gracefully on error.
        """
        if not getattr(cfg, "BINANCE_VOL_FILTER", True):
            return
        now = _time.monotonic()
        ttl = getattr(cfg, "BINANCE_VOL_REFRESH_HR", 4.0) * 3600
        if self._binance_vol_ordered and now - self._last_binance_vol_refresh < ttl:
            return  # still fresh

        top_n = getattr(cfg, "BINANCE_VOL_TOP_N", 250)
        url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
        import urllib.request as _urllib_req, json as _json_mod

        def _fetch() -> list:
            req = _urllib_req.Request(url, headers={"User-Agent": "fvgbot/1.0"})
            with _urllib_req.urlopen(req, timeout=10) as resp:
                return _json_mod.loads(resp.read())

        def _base_from_binance(symbol: str) -> str:
            """'BTCUSDT' -> 'BTC', '1000PEPEUSDT' -> '1000PEPE'."""
            if symbol.endswith("USDT"):
                return symbol[:-4]
            if symbol.endswith("BUSD"):
                return symbol[:-4]
            return symbol

        try:
            await self.heartbeat()  # prevent stale-heartbeat restart during heavy fetch
            loop = asyncio.get_event_loop()
            tickers = await loop.run_in_executor(None, _fetch)

            usdt = [
                t for t in tickers
                if t.get("symbol", "").endswith("USDT")
                and float(t.get("quoteVolume") or 0) > 0
            ]
            usdt.sort(key=lambda t: float(t.get("quoteVolume") or 0), reverse=True)

            new_ordered: list[str] = []
            new_set: set[str] = set()
            for t in usdt[:top_n]:
                base = _base_from_binance(t["symbol"]).upper()
                if base not in new_set:
                    new_ordered.append(base)
                    new_set.add(base)
                # Add "1000X" alias so Bybit's 1000PEPE matches Binance's PEPE listing
                if not base.startswith("1000"):
                    alias = f"1000{base}"
                    if alias not in new_set:
                        new_ordered.append(alias)
                        new_set.add(alias)

            if new_ordered:
                self._binance_vol_ordered = new_ordered
                self._binance_vol_set = new_set
                self._last_binance_vol_refresh = now
                log.info(
                    "[MTF] Binance vol list refreshed: %d base tickers (top-%d by USDT-futures vol).",
                    len(new_ordered), top_n,
                )
        except Exception as exc:
            if self._binance_vol_ordered:
                log.warning("[MTF] Binance FAPI fetch failed (%s) — reusing cached list.", exc)
            else:
                log.warning(
                    "[MTF] Binance FAPI fetch failed (%s) — Binance vol filter DISABLED this cycle.",
                    exc,
                )

    async def _refresh_symbols_if_needed(self) -> None:
        """
        Build symbol list for scanning.

        New logic (Binance-first):
          1. Fetch Binance USDT-futures top-N by volume  →  ordered base-ticker list
          2. Build map of all active Bybit USDT-perp symbols with 24h volume: base -> sym
             (only symbols with Bybit 24h volume >= BINANCE_VOL_MIN_BYBIT_USD are kept)
          3. For each Binance base ticker (in vol order): if Bybit has that perp → include
          4. Apply CoinGecko market-cap whitelist filter (top-250)

        Bybit volume used only as a dead-listing floor (BINANCE_VOL_MIN_BYBIT_USD=$500k).
        Primary liquidity signal = Binance volume ranking.
        If Binance FAPI is unavailable, fall back to Bybit-only top-N.
        """
        now = _time.monotonic()
        refresh_interval = getattr(cfg, "SYMBOLS_REFRESH_HR", 4) * 3600
        if now - self._last_symbols_refresh < refresh_interval and self._symbols:
            return
        try:
            # ── Step 1: Refresh Binance vol list ──────────────────────────────
            await self._refresh_binance_vol_list()

            # ── Step 2: Build Bybit perp map from cached markets ──────────────
            # markets already loaded at startup; refresh on each symbol-list cycle
            markets = dict(self.client._exchange.markets or {})
            if not markets:
                # If markets not loaded yet, do a lightweight reload
                await self.client._exchange.load_markets()
                markets = dict(self.client._exchange.markets or {})

            def _base(sym: str) -> str:
                return sym.split("/")[0]

            # Fetch Bybit tickers once to get 24h volumes for the floor check
            min_bybit_floor = float(getattr(cfg, "BINANCE_VOL_MIN_BYBIT_USD", 500_000))
            tickers = await self.client._exchange.fetch_tickers()
            bybit_vol: dict[str, float] = {
                sym: float(t.get("quoteVolume") or 0)
                for sym, t in tickers.items()
                if sym.endswith(":USDT")
            }

            # base_ticker -> ccxt symbol, only active USDT-perps above volume floor
            bybit_perp_map: dict[str, str] = {}
            for sym, info in markets.items():
                if sym.endswith(":USDT") and info.get("active", True):
                    if bybit_vol.get(sym, 0) >= min_bybit_floor:
                        bybit_perp_map[_base(sym)] = sym

            # ── Step 3: Build symbol list from Binance vol order ──────────────
            await self._refresh_market_cap_list()
            use_mcap = getattr(cfg, "MARKET_CAP_FILTER", True) and bool(self._mc_whitelist)

            if self._binance_vol_ordered:
                # Primary path: Binance-ordered
                combined: list[str] = []
                seen: set[str] = set()
                for base in self._binance_vol_ordered:
                    bybit_sym = bybit_perp_map.get(base)
                    if bybit_sym is None:
                        continue  # not listed on Bybit — skip
                    if bybit_sym in seen:
                        continue
                    if use_mcap and base not in self._mc_whitelist:
                        continue  # filtered by market cap
                    combined.append(bybit_sym)
                    seen.add(bybit_sym)

                log.info(
                    "[MTF] %d symbols loaded (Binance top-%d -> Bybit >=$%.0fk/day -> mcap-%d). "
                    "Bybit eligible perps: %d.",
                    len(combined),
                    getattr(cfg, "BINANCE_VOL_TOP_N", 250),
                    min_bybit_floor / 1000,
                    getattr(cfg, "MARKET_CAP_TOP_N", 250),
                    len(bybit_perp_map),
                )
            else:
                # Fallback: Bybit volume ranking (old behaviour)
                tickers = await self.client._exchange.fetch_tickers()
                min_vol = float(cfg.MIN_24H_VOLUME_USDT)
                usdt_perp = [
                    (sym, float(t.get("quoteVolume") or 0))
                    for sym, t in tickers.items()
                    if sym.endswith(":USDT") and float(t.get("quoteVolume") or 0) >= min_vol
                ]
                usdt_perp.sort(key=lambda x: x[1], reverse=True)
                top = [sym for sym, _ in usdt_perp[: cfg.TOP_N_COINS]]
                if use_mcap:
                    combined = [s for s in top if _base(s) in self._mc_whitelist]
                else:
                    combined = top
                log.info(
                    "[MTF] %d symbols loaded (Binance unavailable — Bybit fallback, top-%d).",
                    len(combined), cfg.TOP_N_COINS,
                )

            self._symbols = combined
            # ──────────────────────────────────────────────────────────────────

            self._last_symbols_refresh = now
            self._markets_cache = markets
        except Exception as exc:
            log.error("[MTF] Symbol refresh failed: %s", exc)

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------

    async def _heartbeat_sleep(self, total_sec: int = 300, interval: int = 60) -> None:
        """Sleep for total_sec while sending heartbeat every interval seconds.
        Prevents MasterAgent from restarting the agent during long waits."""
        elapsed = 0
        while elapsed < total_sec:
            step = min(interval, total_sec - elapsed)
            await asyncio.sleep(step)
            elapsed += step
            await self.heartbeat()

    async def run(self) -> None:
        """Main agent loop. Runs every 5 minutes."""
        _active = [s for s, c in cfg.MTF_STRATEGIES.items() if c.get("enabled", False)]
        log.info("[MTF] MTF agent starting (strategies: %s).", ", ".join(_active))
        await self.heartbeat()

        # Kick off symbol refresh in background immediately so first cycle
        # uses up-to-date data without blocking the heartbeat.
        self._refresh_task: Optional[asyncio.Task] = asyncio.create_task(
            self._refresh_symbols_if_needed()
        )

        while True:
            try:
                await self._reset_daily_if_needed()

                # Wait for any in-progress symbol refresh to finish before scanning,
                # but don't block the heartbeat if it's still running.
                if self._refresh_task and not self._refresh_task.done():
                    await self._refresh_task

                await self._scan_cycle()
                f = self._cycle_funnel
                log.info(
                    "[MTF] Cycle complete. %d symbols -> scanned:%d "
                    "(cool:%d pos:%d lock:%d pend:%d) -> %d w/signal | "
                    "pending:%d | Trades:%d (W:%d L:%d) PnL=$%.2f",
                    f.get("total", len(self._symbols)),
                    f.get("scanned", 0),
                    f.get("cooldown", 0), f.get("position", 0),
                    f.get("locked", 0), f.get("pending_all", 0),
                    f.get("with_signal", 0),
                    sum(len(v) for v in self._pending.values()),
                    self.daily_trades, self.daily_wins, self.daily_losses,
                    self.daily_pnl,
                )

                # Sleep 5 minutes, sending heartbeat every 60s to avoid
                # MasterAgent's 300s stale-heartbeat restart trigger.
                await self._heartbeat_sleep(300, 60)

                # Kick off next symbol refresh in background during the next cycle.
                self._refresh_task = asyncio.create_task(
                    self._refresh_symbols_if_needed()
                )

            except asyncio.CancelledError:
                log.info("[MTF] Agent cancelled.")
                break
            except Exception as exc:
                log.exception("[MTF] Unexpected error in main loop: %s", exc)
                try:
                    await self.tg.send_error(self.agent_id, str(exc))
                except Exception:
                    pass
                await asyncio.sleep(10)

    async def _scan_cycle(self) -> None:
        """One full cycle: manage open positions → scan 1h → check pending entries."""
        await self.heartbeat()

        # 1. Check existing positions for SL/TP hits
        await self._manage_positions()

        await self.heartbeat()

        # 2. Scan 1h candles for new HTF signals
        await self._scan_for_signals()

        await self.heartbeat()

        # 3. For each pending signal, check LTF entry condition
        await self._check_pending_entries()

    # -------------------------------------------------------------------------
    # Position management
    # -------------------------------------------------------------------------

    async def _manage_positions(self) -> None:
        """
        Check each open position for SL/TP hits.

        Two detection paths:
        1. Bybit already closed it (SL/TP order fired on exchange) → detect via
           fetch_all_positions() and record trade WITHOUT placing a duplicate close.
        2. Price crossed SL/TP in state → bot closes manually (fallback for
           positions opened without exchange-side SL/TP).
        """
        positions = await self.state.get_agent_positions(self.agent_id)
        if not positions:
            return

        # Fetch live Bybit positions once (avoids N ticker + N position calls)
        bybit_symbols: Optional[set] = None
        if not cfg.SIMULATION_MODE:
            try:
                bybit_pos = await self.client.fetch_all_positions()
                bybit_symbols = {
                    p["symbol"] for p in bybit_pos
                    if float(p.get("contracts", 0) or 0) != 0
                }
            except Exception as exc:
                log.debug("[MTF] Bybit position fetch failed (will use price check): %s", exc)

        for symbol, pos in list(positions.items()):
            try:
                ticker = await self.client.fetch_ticker(symbol)
                price  = float(ticker.get("last") or 0)
                if price <= 0:
                    continue

                side  = str(pos.get("side", "LONG"))
                sl    = float(pos.get("sl", 0))
                tp    = float(pos.get("tp", 0))

                # ── Recovery: SL/TP missing (position registered via sync before
                #    _finalize_limit_fill ran). Recalculate from current ATR and set. ──
                if sl == 0 and tp == 0 and not cfg.SIMULATION_MODE:
                    await self._try_recover_sl_tp(symbol, pos, price)
                    sl = float(pos.get("sl", 0))
                    tp = float(pos.get("tp", 0))

                # Break-even SL disabled: backtest v11 shows it reduces PF across
                # all strategies (7.13→6.59 for BBS_1H_5M, 5.17→3.83 for KC).
                # High WR strategies should ride to full TP target.

                # ── Path 1: Exchange closed the position (SL/TP fired on Bybit) ──
                if bybit_symbols is not None and symbol not in bybit_symbols:
                    # Infer whether SL or TP fired based on price proximity
                    dist_sl = abs(price - sl) if sl > 0 else float("inf")
                    dist_tp = abs(price - tp) if tp > 0 else float("inf")
                    reason  = "TP" if dist_tp <= dist_sl else "SL"
                    log.info(
                        "[MTF] %s closed by exchange at %.6g (inferred=%s sl=%.6g tp=%.6g)",
                        self._short(symbol), price, reason, sl, tp,
                    )
                    # skip_exchange_order=True: position already gone on Bybit
                    await self._close_position(symbol, pos, price, reason,
                                               skip_exchange_order=True)
                    continue

                # ── Path 2: Price-based detection (simulation or fallback) ──
                hit    = False
                reason = ""
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
                    await self._close_position(symbol, pos, price, reason,
                                               skip_exchange_order=False)

            except Exception as exc:
                log.warning("[MTF] Position check error %s: %s", self._short(symbol), exc)

    async def _try_recover_sl_tp(self, symbol: str, pos: dict, current_price: float) -> None:
        """
        Recalculate and set SL/TP for a position that was registered without them
        (e.g. Bybit fill detected via background sync before _finalize_limit_fill ran).
        Uses the entry-timeframe (15m default) ATR so levels match the original calc.
        Updates pos dict in-place so _manage_positions immediately sees the new levels.
        """
        strategy_id = pos.get("strategy_id", "")
        scfg        = self._strat_cfg(strategy_id) if strategy_id else {}
        sl_mult     = scfg.get("sl_atr_mult", 1.5)
        tp_r        = scfg.get("tp_r", 2.0)
        entry_tf    = scfg.get("entry_tf", "15m")

        entry    = float(pos.get("entry_price", current_price))
        side     = str(pos.get("side", "LONG"))
        qty      = float(pos.get("qty", 0))
        side_str = "buy" if side == "LONG" else "sell"

        if qty <= 0 or entry <= 0:
            log.warning("[MTF] Recovery: invalid entry/qty for %s (entry=%.6g qty=%.6g) — skip",
                        self._short(symbol), entry, qty)
            return

        try:
            raw_ltf = await self.client.fetch_ohlcv(symbol, entry_tf, 30)
        except Exception as exc:
            log.warning("[MTF] Recovery: %s fetch failed %s: %s", entry_tf, self._short(symbol), exc)
            return

        if not raw_ltf or len(raw_ltf) < 15:
            log.warning("[MTF] Recovery: not enough %s candles for %s", entry_tf, self._short(symbol))
            return

        df_ltf  = pd.DataFrame(raw_ltf, columns=["timestamp", "open", "high", "low", "close", "volume"])
        highs   = df_ltf["high"].astype(float).to_numpy()
        lows    = df_ltf["low"].astype(float).to_numpy()
        closes  = df_ltf["close"].astype(float).to_numpy()
        atr14   = _calculate_atr(highs, lows, closes, 14)
        atr_val = float(atr14[-2])  # bar -2 (closed bar), same convention as entry logic

        if atr_val <= 0 or np.isnan(atr_val):
            log.warning("[MTF] Recovery: ATR invalid for %s (%.6g)", self._short(symbol), atr_val)
            return

        sl_dist = atr_val * sl_mult
        sl_dist = max(sl_dist, entry * 0.003)
        sl_dist = min(sl_dist, entry * 0.05)
        if side == "LONG":
            sl = entry - sl_dist
            tp = entry + sl_dist * tp_r
        else:
            sl = entry + sl_dist
            tp = entry - sl_dist * tp_r

        if sl <= 0 or tp <= 0:
            log.warning("[MTF] Recovery: computed sl/tp invalid %s sl=%.6g tp=%.6g", self._short(symbol), sl, tp)
            return

        log.warning(
            "[MTF] RECOVERY: setting SL/TP on position without stops %s %s "
            "entry=%.6g sl=%.6g tp=%.6g atr=%.6g",
            self._short(symbol), side, entry, sl, tp, atr_val,
        )
        try:
            await self.client.place_sl_tp(symbol, side_str, qty, sl, tp)
            # Update pos dict in-place (same object referenced by state manager)
            pos["sl"] = sl
            pos["tp"] = tp
            log.info("[MTF] RECOVERY OK: SL/TP placed for %s", self._short(symbol))
        except Exception as exc:
            log.warning("[MTF] Recovery: place_sl_tp failed %s: %s", self._short(symbol), exc)

    async def _close_position(
        self,
        symbol: str,
        pos: dict,
        exit_price: float,
        reason: str,
        skip_exchange_order: bool = False,
        override_pnl: float | None = None,
    ) -> None:
        """
        Record a closed position, update stats, and optionally place a market
        close order on the exchange.

        Parameters
        ----------
        skip_exchange_order : set True when Bybit already closed the position
            (SL/TP fired on the exchange).  Prevents a duplicate close order
            that would fail or open an unwanted position.
        override_pnl : if provided (e.g. from Bybit closedPnl API), use this
            as the authoritative realized PnL (already includes exchange fees).
            Otherwise pnl is calculated from prices and taker fees deducted.
        """
        # Remove from state immediately (prevents duplicate processing)
        await self.state.remove_position(self.agent_id, symbol)
        # Clean up per-position tracking
        self._be_applied.pop(symbol, None)
        self._next_bar_cycles.pop((symbol, pos.get("strategy_id", "")), None)

        side        = str(pos.get("side", "LONG"))
        entry       = float(pos.get("entry", exit_price))
        qty         = float(pos.get("qty", 0))
        strategy_id = pos.get("strategy_id", "MTF")
        short_sym   = self._short(symbol)

        if override_pnl is not None:
            # Use Bybit's realized PnL directly (net of exchange fees)
            pnl_usd = override_pnl
            pnl_pct = pnl_usd / max(entry * qty, 1e-9) * 100.0
        else:
            if side == "LONG":
                pnl_usd = (exit_price - entry) * qty
                pnl_pct = (exit_price - entry) / max(entry, 1e-9) * 100.0
            else:
                pnl_usd = (entry - exit_price) * qty
                pnl_pct = (entry - exit_price) / max(entry, 1e-9) * 100.0
            # Deduct taker fees: entry fill + exit fill
            fee_usd = (entry + exit_price) * qty * cfg.TAKER_FEE_PCT
            pnl_usd -= fee_usd
            pnl_pct -= fee_usd / max(entry * qty, 1e-9) * 100.0

        # Update daily counters
        self.daily_pnl    += pnl_usd
        self.daily_trades += 1
        if pnl_usd >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        log.info(
            "[MTF] CLOSED [%s] %s %s entry=%.6g exit=%.6g pnl=$%.4f (%.2f%%) reason=%s%s",
            strategy_id, short_sym, side, entry, exit_price, pnl_usd, pnl_pct, reason,
            " (exchange-closed)" if skip_exchange_order else "",
        )

        # Cooldown / re-entry protection
        if reason == "SL":
            self._set_sl_cooldown(symbol)
        elif reason == "TP":
            self._reset_sl_streak(symbol)

        # Persist trade to state log (used by dashboard /api/v2/trades)
        now_utc = datetime.now(timezone.utc)
        await self.state.record_trade(self.agent_id, {
            "symbol":      symbol,
            "strategy_id": strategy_id,
            "side":        side,
            "direction":   side,       # alias so dashboard 'direction' field works
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

        # Update equity in state
        current = await self.state.get_agent_equity(self.agent_id)
        await self.state.set_agent_equity(self.agent_id, max(0.0, current + pnl_usd))

        # Place close order only if Bybit hasn't already done it
        if not cfg.SIMULATION_MODE and not skip_exchange_order:
            close_side = "sell" if side == "LONG" else "buy"
            try:
                await self.client.place_market_order(symbol, close_side, qty)
            except Exception as exc:
                log.error("[MTF] Close order failed %s: %s", short_sym, exc)

        # Telegram notification
        try:
            await self.tg.send_position_closed(
                self.agent_id, symbol, side, entry, exit_price, pnl_usd, pnl_pct, reason,
            )
        except Exception as exc:
            log.warning("[MTF] TG close notification failed: %s", exc)

    # -------------------------------------------------------------------------
    # 1h signal scanning
    # -------------------------------------------------------------------------

    async def _scan_for_signals(self) -> None:
        """Scan all symbols on 1h; store signals that pass filters in self._pending."""
        self._cycle_funnel = {
            "total": len(self._symbols),
            "cooldown": 0, "position": 0, "locked": 0, "pending_all": 0,
            "scanned": 0, "with_signal": 0,
        }
        batch_size = 10
        for i in range(0, len(self._symbols), batch_size):
            batch = self._symbols[i : i + batch_size]
            tasks = [self._scan_symbol_1h(sym) for sym in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            for sym, result in zip(batch, results):
                if isinstance(result, Exception):
                    log.debug("[MTF] 1h scan error %s: %s", self._short(sym), result)
                elif result:
                    self._cycle_funnel["with_signal"] += 1
                    for sig in result:
                        self._store_pending(sym, sig)

            await self.heartbeat()

    def _store_pending(self, symbol: str, sig: dict) -> None:
        """Add signal to pending dict, avoiding duplicates per (symbol, strategy_id)."""
        if symbol not in self._pending:
            self._pending[symbol] = []
        strategy_id = sig["strategy_id"]
        # Replace existing pending signal for same strategy on same symbol
        self._pending[symbol] = [
            s for s in self._pending[symbol] if s["strategy_id"] != strategy_id
        ]
        self._pending[symbol].append(sig)
        log.info(
            "[MTF] SIGNAL [%s] %s %s | sq=%d adx=%.1f vol=%.1fx",
            strategy_id, self._short(symbol), sig["direction"],
            sig["squeeze_bars"], sig["adx"], sig["volume_ratio"],
        )

    async def _scan_symbol_1h(self, symbol: str) -> list[dict]:
        """
        Fetch 1h candles for symbol; run BBS and KC signal detection.
        If KC_4H_15M is enabled, also fetches 4h candles for that strategy.
        Returns list of signal dicts (0-3 items).
        """
        # Global gates
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return []

        # Blacklist removed: v10 annual backtest ($1000->$2895, PF 9.66/3.92/5.76)
        # used all 20 symbols including ARB/OP/POL. v11 filter comparison was
        # against a different simulation baseline, not reliable enough to exclude.

        if await self.state.is_symbol_locked(symbol):
            self._cycle_funnel["locked"] = self._cycle_funnel.get("locked", 0) + 1
            return []

        # Already have a position in this symbol from this agent?
        positions = await self.state.get_agent_positions(self.agent_id)
        if symbol in positions:
            self._cycle_funnel["position"] = self._cycle_funnel.get("position", 0) + 1
            return []

        # SL cooldown check — block symbol if recent SL on it
        if self._is_on_cooldown(symbol):
            self._cycle_funnel["cooldown"] = self._cycle_funnel.get("cooldown", 0) + 1
            return []

        # Already have pending signal for ALL enabled strategies on this symbol?
        # (No need to re-scan if already pending — avoid overwriting with stale bar)
        pending_strats = {s["strategy_id"] for s in self._pending.get(symbol, [])}
        _enabled_strats = {s for s, c in cfg.MTF_STRATEGIES.items() if c.get("enabled", False)}
        all_strats_pending = pending_strats >= _enabled_strats and bool(_enabled_strats)
        if all_strats_pending:
            self._cycle_funnel["pending_all"] = self._cycle_funnel.get("pending_all", 0) + 1
            return []

        self._cycle_funnel["scanned"] = self._cycle_funnel.get("scanned", 0) + 1
        try:
            raw_1h = await self.client.fetch_ohlcv(symbol, "1h", cfg.OHLCV_LIMIT_HTF)
        except Exception as exc:
            log.debug("[MTF] 1h fetch failed %s: %s", self._short(symbol), exc)
            return []

        if not raw_1h or len(raw_1h) < 250:
            return []

        # ── XAUT (gold) ATH filter ────────────────────────────────────────────
        # Skip all signals when price is within ATH_FILTER_PCT of its 20-day high.
        # Backtest confirmed: Feb-2026 macro cluster produced 4 consecutive losses
        # precisely when XAUT was at ATH — filter eliminates that bad stretch.
        base_ticker = self._short(symbol).replace("USDT", "")
        ath_filter_syms: set = getattr(cfg, "ATH_FILTER_SYMBOLS", {"XAUT"})
        if base_ticker in ath_filter_syms and len(raw_1h) >= 480:
            highs_20d = [c[2] for c in raw_1h[-480:]]
            rolling_high_20d = max(highs_20d)
            current_close = raw_1h[-1][4]
            ath_pct: float = getattr(cfg, "ATH_FILTER_PCT", 0.98)
            if current_close >= rolling_high_20d * ath_pct:
                log.debug(
                    "[MTF] %s ATH filter: close %.4f >= %.0f%% of 20d-high %.4f — skip",
                    self._short(symbol), current_close, ath_pct * 100, rolling_high_20d,
                )
                return []

        # ── Per-symbol strategy exclusions ────────────────────────────────────
        # Strategies with negative edge on specific symbols (e.g. KC on XAUT).
        sym_excl: set = set(
            getattr(cfg, "SYMBOL_STRATEGY_EXCLUSIONS", {}).get(base_ticker, set())
        )

        df_1h = pd.DataFrame(
            raw_1h, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        signals: list[dict] = []

        # ── BBS signals (for BBS_1H_5M and BBS_1H_15M) ──────────────────────
        bbs_strats = [
            s for s in (self.STRAT_BBS_1H_5M, self.STRAT_BBS_1H_15M)
            if s not in pending_strats
            and s not in sym_excl
            and cfg.MTF_STRATEGIES.get(s, {}).get("enabled", False)
        ]
        if bbs_strats:
            bbs_sig = self._detect_bbs_signal_1h(df_1h, symbol)
            if bbs_sig:
                # Daily EMA200 filter — only for strategies in DAILY_EMA200_FILTER_STRATEGIES
                daily_dir: Optional[str] = None
                needs_daily = any(s in DAILY_EMA200_FILTER_STRATEGIES for s in bbs_strats)
                if needs_daily:
                    daily_dir = await self._get_daily_ema200_direction(symbol)

                for strat_id in bbs_strats:
                    scfg = self._strat_cfg(strat_id)
                    sp   = scfg.get("signal_params", {})
                    # Each strategy has its own min_squeeze threshold
                    if bbs_sig["squeeze_bars"] < sp.get("min_squeeze", 4):
                        continue
                    # Daily EMA200 alignment filter (BBS_1H_5M only)
                    if strat_id in DAILY_EMA200_FILTER_STRATEGIES and daily_dir is not None:
                        if bbs_sig["direction"] != daily_dir:
                            log.debug(
                                "[MTF] Daily EMA200 filter rejected [%s] %s %s (daily=%s)",
                                strat_id, self._short(symbol), bbs_sig["direction"], daily_dir,
                            )
                            continue
                    sig = dict(bbs_sig)
                    sig["strategy_id"] = strat_id
                    sig["entry_mode"]  = scfg.get("entry_mode", "next_bar")
                    sig["entry_tf"]    = scfg.get("entry_tf", "15m")
                    sig["sl_atr_mult"] = scfg.get("sl_atr_mult", 1.5)
                    sig["tp_r"]        = scfg.get("tp_r", 2.5)
                    sig["bars_waited"] = 0
                    sig["cycles_waited"] = 0
                    sig["signal_ts"]   = datetime.now(timezone.utc).isoformat()
                    signals.append(sig)

        # ── KC signal (for KC_1H_15M) ────────────────────────────────────────
        if (
            self.STRAT_KC_1H_15M not in pending_strats
            and self.STRAT_KC_1H_15M not in sym_excl
            and cfg.MTF_STRATEGIES.get(self.STRAT_KC_1H_15M, {}).get("enabled", False)
        ):
            kc_sig = self._detect_kc_signal_1h(df_1h, symbol)
            if kc_sig:
                scfg = self._strat_cfg(self.STRAT_KC_1H_15M)
                sig  = dict(kc_sig)
                sig["strategy_id"] = self.STRAT_KC_1H_15M
                sig["entry_mode"]  = scfg.get("entry_mode", "next_bar")
                sig["entry_tf"]    = scfg.get("entry_tf", "15m")
                sig["sl_atr_mult"] = scfg.get("sl_atr_mult", 1.5)
                sig["tp_r"]        = scfg.get("tp_r", 2.0)
                sig["bars_waited"] = 0
                sig["cycles_waited"] = 0
                sig["signal_ts"]   = datetime.now(timezone.utc).isoformat()
                signals.append(sig)

        # ── KC signal (for KC_4H_15M) ────────────────────────────────────────
        # 4h HTF squeeze → 15m next_bar entry. Backtest: OOS PF=4.14, WR=72.7%.
        if (
            self.STRAT_KC_4H_15M not in pending_strats
            and self.STRAT_KC_4H_15M not in sym_excl
            and cfg.MTF_STRATEGIES.get(self.STRAT_KC_4H_15M, {}).get("enabled", False)
        ):
            try:
                raw_4h = await self.client.fetch_ohlcv(symbol, "4h", 200)
            except Exception as exc:
                log.debug("[MTF] 4h fetch failed %s: %s", self._short(symbol), exc)
                raw_4h = None

            if raw_4h and len(raw_4h) >= 50:
                df_4h = pd.DataFrame(
                    raw_4h, columns=["timestamp", "open", "high", "low", "close", "volume"]
                )
                kc4h_sig = self._detect_kc_signal_4h(df_4h, symbol)
                if kc4h_sig:
                    scfg = self._strat_cfg(self.STRAT_KC_4H_15M)
                    sig  = dict(kc4h_sig)
                    sig["strategy_id"] = self.STRAT_KC_4H_15M
                    sig["entry_mode"]  = scfg.get("entry_mode", "next_bar")
                    sig["entry_tf"]    = scfg.get("entry_tf", "15m")
                    sig["sl_atr_mult"] = scfg.get("sl_atr_mult", 1.5)
                    sig["tp_r"]        = scfg.get("tp_r", 2.0)
                    sig["bars_waited"] = 0
                    sig["cycles_waited"] = 0
                    sig["signal_ts"]   = datetime.now(timezone.utc).isoformat()
                    signals.append(sig)

        return signals

    def _detect_bbs_signal_1h(self, df_1h: pd.DataFrame, symbol: str) -> Optional[dict]:
        """
        Run BBS squeeze detection on the 1h dataframe.
        Shared by BBS_1H_5M and BBS_1H_15M (signal criteria are identical;
        the difference is in min_squeeze_bars checked per-strategy above).

        Uses the broadest threshold (adx≥15, vol≥1.0, headfake_filter=True).
        """
        try:
            ind = compute_squeeze_indicators(
                df_1h,
                bb_len=cfg.BB_SQUEEZE_BB_LENGTH,      # 20
                bb_mult=cfg.BB_SQUEEZE_BB_MULT,        # 2.0
                kc_len=cfg.BB_SQUEEZE_KC_LENGTH,       # 20
                kc_mult=cfg.BB_SQUEEZE_KC_MULT,        # 1.5
                mom_len=cfg.BB_SQUEEZE_MOM_LENGTH,     # 20
                adx_period=cfg.BB_SQUEEZE_ADX_PERIOD,  # 14
            )

            sig = detect_squeeze_signal(
                ind,
                bar_idx=-2,              # second-to-last bar (confirmed closed)
                min_squeeze_bars=4,      # minimum; per-strategy check done above
                adx_threshold=15.0,      # BBS_1H_5M + BBS_1H_15M both use 15
                vol_mult=1.0,
            )
            if sig is None:
                return None

            bar_idx   = sig["bar_idx"]
            direction = sig["direction"]

            # EMA stack required for BBS strategies (checked on signal bar)
            close  = float(ind["closes"][bar_idx])
            ema21  = float(ind["ema21"][bar_idx])
            ema50  = float(ind["ema50"][bar_idx])
            ema200 = float(ind["ema200"][bar_idx])

            if direction == "LONG":
                if not (close > ema21 > ema50 > ema200):
                    return None
            else:
                if not (close < ema21 < ema50 < ema200):
                    return None

            # Headfake filter: check if momentum reversed in the 2 bars after release.
            # bar_idx is the release bar (confirmed closed = second-to-last),
            # so bar -1 (last 1h bar) is the first post-release bar.
            if check_head_fake(ind, release_bar=bar_idx, check_bars=1):
                log.debug("[MTF] BBS headfake rejected %s %s", self._short(symbol), direction)
                return None

            # ── Obstacle-ahead filter (global level blocker) ─────────────────
            # Skip if there's a global level IN THE DIRECTION of the trade within
            # OBS_PCT% of entry price.
            #   LONG:  skip if swing HIGH within OBS_PCT% ABOVE close (resistance ahead)
            #   SHORT: skip if swing LOW  within OBS_PCT% BELOW close (support ahead)
            # Backtest (180d, 21 sym): pct=1.0%, bars=200h → OOS PF 2.27→6.26,
            # WR 30.3%→39.3%, DD 18.9%→7.7%. Filters ~69% of BBS_1H_15M signals.
            _OBS_PCT  = 0.010   # 1.0% — distance threshold in price %
            _OBS_BARS = 200     # 200×1H bars lookback (~8 days = global levels)
            try:
                _start = max(0, bar_idx - _OBS_BARS)
                if direction == "LONG":
                    _highs = ind["highs"][_start:bar_idx]
                    _highs_above = _highs[_highs > close]
                    if len(_highs_above) > 0:
                        _nearest = float(np.min(_highs_above))
                        _gap_pct = (_nearest - close) / close
                        if _gap_pct <= _OBS_PCT:
                            log.debug(
                                "[MTF] BBS obstacle-ahead skip %s %s — resistance %.3f%% above (threshold %.1f%%)",
                                self._short(symbol), direction, _gap_pct * 100, _OBS_PCT * 100,
                            )
                            return None
                else:  # SHORT
                    _lows = ind["lows"][_start:bar_idx]
                    _lows_below = _lows[_lows < close]
                    if len(_lows_below) > 0:
                        _nearest = float(np.max(_lows_below))
                        _gap_pct = (close - _nearest) / close
                        if _gap_pct <= _OBS_PCT:
                            log.debug(
                                "[MTF] BBS obstacle-ahead skip %s %s — support %.3f%% below (threshold %.1f%%)",
                                self._short(symbol), direction, _gap_pct * 100, _OBS_PCT * 100,
                            )
                            return None
            except Exception:
                pass  # obstacle check failure → let signal through

            return {
                "symbol":       symbol,
                "direction":    direction,
                "signal_price": float(ind["closes"][bar_idx]),
                "squeeze_bars": sig["squeeze_bars"],
                "adx":          sig["adx"],
                "volume_ratio": sig["volume_ratio"],
                "momentum":     sig["momentum"],
                "signal_bar_idx": bar_idx,
            }

        except Exception as exc:
            log.debug("[MTF] BBS signal error %s: %s", self._short(symbol), exc)
            return None

    def _detect_kc_signal_1h(self, df_1h: pd.DataFrame, symbol: str) -> Optional[dict]:
        """Run KC squeeze detection on the 1h dataframe for KC_1H_15M strategy."""
        try:
            sp = self._strat_cfg(self.STRAT_KC_1H_15M).get("signal_params", {})
            ind = _compute_kc_squeeze_indicators(
                df_1h,
                bb_len=cfg.BB_SQUEEZE_BB_LENGTH,
                bb_mult=cfg.BB_SQUEEZE_BB_MULT,
                kc_len=cfg.BB_SQUEEZE_KC_LENGTH,
                kc_mult=cfg.BB_SQUEEZE_KC_MULT,
                mom_bars=sp.get("mom_bars", 6),
                adx_period=cfg.BB_SQUEEZE_ADX_PERIOD,
            )

            sig = _detect_kc_signal(
                ind,
                bar_idx=-2,  # confirmed closed bar
                min_squeeze_bars=sp.get("min_squeeze", 10),
                adx_threshold=sp.get("adx_min", 12.0),
                vol_mult=sp.get("vol_mult", 1.0),
            )
            if sig is None:
                return None

            return {
                "symbol":         symbol,
                "direction":      sig["direction"],
                "signal_price":   sig["entry_price"],
                "squeeze_bars":   sig["squeeze_bars"],
                "adx":            sig["adx"],
                "volume_ratio":   sig["volume_ratio"],
                "momentum":       sig["momentum"],
                "signal_bar_idx": sig["bar_idx"],
            }

        except Exception as exc:
            log.debug("[MTF] KC signal error %s: %s", self._short(symbol), exc)
            return None

    def _detect_kc_signal_4h(self, df_4h: pd.DataFrame, symbol: str) -> Optional[dict]:
        """
        Run KC squeeze detection on the 4h dataframe for KC_4H_15M strategy.

        Backtest-validated params: min_sq=8, adx_min=15, vol_mult=1.0, mom_bars=6.
        OOS PF=4.14, WR=72.7%, 22 trades (730d dataset, 40 symbols).
        """
        try:
            sp = self._strat_cfg(self.STRAT_KC_4H_15M).get("signal_params", {})
            ind = _compute_kc_squeeze_indicators(
                df_4h,
                bb_len=cfg.BB_SQUEEZE_BB_LENGTH,
                bb_mult=cfg.BB_SQUEEZE_BB_MULT,
                kc_len=cfg.BB_SQUEEZE_KC_LENGTH,
                kc_mult=cfg.BB_SQUEEZE_KC_MULT,
                mom_bars=sp.get("mom_bars", 6),
                adx_period=cfg.BB_SQUEEZE_ADX_PERIOD,
            )

            sig = _detect_kc_signal(
                ind,
                bar_idx=-2,  # confirmed closed bar
                min_squeeze_bars=sp.get("min_squeeze", 8),
                adx_threshold=sp.get("adx_min", 15.0),
                vol_mult=sp.get("vol_mult", 1.0),
            )
            if sig is None:
                return None

            return {
                "symbol":         symbol,
                "direction":      sig["direction"],
                "signal_price":   sig["entry_price"],
                "squeeze_bars":   sig["squeeze_bars"],
                "adx":            sig["adx"],
                "volume_ratio":   sig["volume_ratio"],
                "momentum":       sig["momentum"],
                "signal_bar_idx": sig["bar_idx"],
            }

        except Exception as exc:
            log.debug("[MTF] KC 4h signal error %s: %s", self._short(symbol), exc)
            return None

    # -------------------------------------------------------------------------
    # LTF entry checking
    # -------------------------------------------------------------------------

    async def _check_pending_entries(self) -> None:
        """
        For each pending signal, check the LTF for entry conditions.

        BBS_1H_5M  : wait for price to touch EMA21 on 5m (up to 36 bars)
        BBS_1H_15M : enter at current 15m bar's open (next_bar)
        KC_4H_15M  : enter at current 15m bar's open (next_bar)
        """
        expired: list[tuple[str, str]] = []

        for symbol, sig_list in list(self._pending.items()):
            for sig in list(sig_list):
                strategy_id = sig["strategy_id"]
                entry_mode  = sig.get("entry_mode", "next_bar")

                try:
                    if entry_mode == "pullback_ema21":
                        result = await self._check_pullback_entry(symbol, sig)
                    else:  # next_bar
                        result = await self._check_next_bar_entry(symbol, sig)

                    if result == "entered":
                        expired.append((symbol, strategy_id))
                    elif result == "expired":
                        log.info(
                            "[MTF] Signal expired [%s] %s %s (waited %d bars)",
                            strategy_id, self._short(symbol), sig["direction"],
                            sig.get("bars_waited", 0),
                        )
                        expired.append((symbol, strategy_id))
                    # else: "waiting" → keep in pending

                except Exception as exc:
                    log.warning(
                        "[MTF] Entry check error [%s] %s: %s",
                        strategy_id, self._short(symbol), exc,
                    )

        # Clean up expired / entered signals
        for symbol, strategy_id in expired:
            if symbol in self._pending:
                self._pending[symbol] = [
                    s for s in self._pending[symbol]
                    if s["strategy_id"] != strategy_id
                ]
                if not self._pending[symbol]:
                    del self._pending[symbol]

    async def _check_pullback_entry(self, symbol: str, sig: dict) -> str:
        """
        BBS_1H_5M pullback-to-EMA21 entry logic.

        Fetches 5m candles, checks if price has touched EMA21.
        Returns: "entered" | "expired" | "waiting"
        """
        max_wait = self._strat_cfg(self.STRAT_BBS_1H_5M).get("max_wait_bars", 20)

        sig["bars_waited"] = sig.get("bars_waited", 0) + 1
        if sig["bars_waited"] > max_wait:
            return "expired"

        try:
            raw_5m = await self.client.fetch_ohlcv(symbol, "5m", 100)
        except Exception as exc:
            log.debug("[MTF] 5m fetch failed %s: %s", self._short(symbol), exc)
            return "waiting"

        if not raw_5m or len(raw_5m) < 30:
            return "waiting"

        df_5m = pd.DataFrame(
            raw_5m, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        entry_ind = _compute_entry_tf_indicators(df_5m)

        closes = entry_ind["closes"]
        ema21  = entry_ind["ema21"]
        atr14  = entry_ind["atr14"]
        opens  = entry_ind["opens"]

        # Use last confirmed bar (second-to-last) for pullback detection,
        # then enter on current bar's open.
        i = len(closes) - 2  # last fully confirmed 5m bar
        if i < 1 or np.isnan(ema21[i]) or np.isnan(atr14[i]):
            return "waiting"

        direction = sig["direction"]
        price     = closes[i]
        ema_val   = ema21[i]

        # Pullback with bounce confirmation:
        # LONG:  low touched EMA21 AND close bounced back above EMA21
        #        (prevents entering on breakdowns where price falls through EMA21)
        # SHORT: high touched EMA21 AND close rejected back below EMA21
        touched = False
        close_bar = float(df_5m["close"].iloc[i])
        if direction == "LONG":
            low_bar = float(df_5m["low"].iloc[i])
            touched = (low_bar <= ema_val * 1.002       # low reached EMA21
                       and close_bar >= ema_val * 0.998) # close bounced above EMA21
        else:  # SHORT
            high_bar = float(df_5m["high"].iloc[i])
            touched = (high_bar >= ema_val * 0.998      # high reached EMA21
                       and close_bar <= ema_val * 1.002) # close rejected below EMA21

        if not touched:
            return "waiting"

        # Entry price = open of current (forming) 5m bar
        # Also check current forming bar hasn't already broken through EMA21
        entry_price = float(opens[-1])
        current_ema = ema21[-1] if not np.isnan(ema21[-1]) else ema_val
        if direction == "LONG" and entry_price < current_ema * 0.993:
            return "waiting"   # price already 0.7%+ below EMA21 — skip
        if direction == "SHORT" and entry_price > current_ema * 1.007:
            return "waiting"   # price already 0.7%+ above EMA21 — skip
        if entry_price <= 0:
            entry_price = float(closes[-1])

        # SL = entry ± ATR(14) × sl_atr_mult
        atr_val  = float(atr14[i])
        sl_mult  = sig.get("sl_atr_mult", 1.5)
        sl_dist  = atr_val * sl_mult

        # Clamp SL distance
        sl_dist = max(sl_dist, entry_price * 0.003)   # min 0.3%
        sl_dist = min(sl_dist, entry_price * 0.05)    # max 5%

        if direction == "LONG":
            sl = entry_price - sl_dist
        else:
            sl = entry_price + sl_dist

        tp_r    = sig.get("tp_r", 3.0)
        risk    = abs(entry_price - sl)
        tp      = (entry_price + risk * tp_r) if direction == "LONG" else (entry_price - risk * tp_r)

        executed = await self._execute_entry(symbol, sig, entry_price, sl, tp, entry_ind)
        return "entered" if executed else "waiting"

    async def _check_next_bar_entry(self, symbol: str, sig: dict) -> str:
        """
        BBS_1H_15M / KC_1H_15M next-bar entry logic.

        Enters at current LTF bar's open/close price.
        Signal expires after NEXT_BAR_MAX_CYCLES cycles (~15 min) to prevent
        stale entries on a signal that's no longer valid.

        Note: LTF bar confirmation (bullish bar required for LONG) was removed
        in v11 — backtest showed it reduced PF on both next_bar strategies.

        Returns: "entered" | "expired" | "waiting"
        """
        strategy_id = sig["strategy_id"]
        key = (symbol, strategy_id)

        # ── If a post-only limit is already in flight — check its status ──
        if key in self._pending_limit_orders:
            pending = self._pending_limit_orders[key]

            # Fast-path: MasterAgent sync may have already registered the position
            # (happens when Bybit fills the limit order between our cycles).
            positions = await self.state.get_agent_positions(self.agent_id)
            if symbol in positions:
                log.info(
                    "[MTF] Limit fill confirmed via sync for %s [%s] — cleaning up pending order",
                    self._short(symbol), strategy_id,
                )
                # The sync-registered position has sl=0, tp=0 — place the SL/TP
                # that was computed when the signal was generated.
                _sl      = pending.get("sl", 0.0)
                _tp      = pending.get("tp", 0.0)
                _side    = pending.get("side_str", "buy")
                _qty     = pending.get("qty", 0.0)
                if _qty > 0 and (_sl > 0 or _tp > 0) and not cfg.SIMULATION_MODE:
                    try:
                        await self.client.place_sl_tp(symbol, _side, _qty, _sl, _tp)
                        log.info(
                            "[MTF] SL/TP set on sync-filled position %s sl=%.6g tp=%.6g",
                            self._short(symbol), _sl, _tp,
                        )
                        # Patch state so manage_positions uses correct levels
                        _pos_map = await self.state.get_agent_positions(self.agent_id)
                        _ps = _pos_map.get(symbol)
                        if _ps is not None:
                            _ps["sl"] = _sl
                            _ps["tp"] = _tp
                    except Exception as exc:
                        log.warning(
                            "[MTF] SL/TP set failed (sync-fill) %s: %s",
                            self._short(symbol), exc,
                        )
                # Send TG position-opened notification for sync-filled position
                _sig_data  = pending.get("sig", {})
                _direction = _sig_data.get("direction", "LONG")
                _leverage  = pending.get("leverage", 50)
                _margin    = pending.get("margin_used", 0.0)
                _ep        = pending.get("entry_price", 0.0)
                try:
                    await self.tg.send_position_opened(
                        self.agent_id, symbol, _direction, _ep, _sl, _tp,
                        _qty, _leverage, _margin,
                    )
                except Exception as exc:
                    log.warning(
                        "[MTF] TG open notification failed (sync-fill) %s: %s",
                        self._short(symbol), exc,
                    )
                del self._pending_limit_orders[key]
                return "entered"

            pending["cycles"] = pending.get("cycles", 0) + 1
            sig["cycles_waited"] = pending["cycles"]

            if pending["cycles"] > NEXT_BAR_MAX_CYCLES:
                # Timed out — cancel and give up on this signal
                try:
                    await self.client.cancel_order(symbol, pending["order_id"])
                    log.info("[MTF] Limit order expired+cancelled %s strat=%s", self._short(symbol), strategy_id)
                except Exception:
                    pass
                del self._pending_limit_orders[key]
                return "expired"

            try:
                order = await self.client.fetch_order(pending["order_id"], symbol)
                status = order.get("status", "open")
                if status == "closed":
                    # Filled — finalize position
                    fill_price = float(order.get("average") or order.get("price") or pending["entry_price"])
                    await self._finalize_limit_fill(symbol, pending, fill_price)
                    del self._pending_limit_orders[key]
                    return "entered"
                elif status in ("canceled", "rejected", "expired"):
                    log.info("[MTF] Limit order %s for %s — trying market fallback", status, self._short(symbol))
                    del self._pending_limit_orders[key]
                    # Fall through to place a fresh order below
                else:
                    return "waiting"  # still open
            except Exception as exc:
                log.debug("[MTF] fetch_order %s: %s", self._short(symbol), exc)
                return "waiting"

        # ── No pending limit — check Bybit for an existing open order first ──
        # On restart, the local _pending_limit_orders dict is empty, but a
        # limit order we placed before the crash may still be live on Bybit.
        # Adopt it instead of placing a duplicate.
        if not cfg.SIMULATION_MODE:
            try:
                open_orders = await self.client.fetch_open_orders(symbol)
                buy_sell = "buy" if sig["direction"] == "LONG" else "sell"
                existing = [o for o in open_orders if str(o.get("side", "")).lower() == buy_sell]
                if existing:
                    adopted = existing[0]
                    adopted_price = float(adopted.get("price") or adopted.get("average") or 0)
                    log.info(
                        "[MTF] Adopting existing Bybit order for %s [%s] id=%s price=%.6g",
                        self._short(symbol), strategy_id, adopted["id"], adopted_price,
                    )
                    # We don't have sl/tp yet — they will be set when fill is detected
                    # Use placeholder sl/tp=0; _finalize_limit_fill will compute them
                    # from the actual fill via the ATR at that moment (via pending sig)
                    self._pending_limit_orders[key] = {
                        "order_id":    adopted["id"],
                        "entry_price": adopted_price or sig.get("signal_price", 0),
                        "sl":          0.0,
                        "tp":          0.0,
                        "qty":         float(adopted.get("amount") or adopted.get("remaining") or 0),
                        "leverage":    sig.get("leverage", 50),
                        "side_str":    buy_sell,
                        "sig":         sig,
                        "margin_used": 0.0,
                        "risk_usd":    0.0,
                        "balance":     await self._fetch_balance(),
                        "cycles":      0,
                        "adopted":     True,  # flag: sl/tp unknown, recovery will set them
                    }
                    return "waiting"
            except Exception as exc:
                log.debug("[MTF] fetch_open_orders check failed %s: %s", self._short(symbol), exc)

        # ── No pending limit — calculate entry and place one ──────────────
        sig["cycles_waited"] = sig.get("cycles_waited", 0) + 1
        if sig["cycles_waited"] > NEXT_BAR_MAX_CYCLES:
            return "expired"

        entry_tf = sig.get("entry_tf", "15m")
        try:
            raw_ltf = await self.client.fetch_ohlcv(symbol, entry_tf, 60)
        except Exception as exc:
            log.debug("[MTF] %s fetch failed %s: %s", entry_tf, self._short(symbol), exc)
            return "waiting"

        if not raw_ltf or len(raw_ltf) < 20:
            return "waiting"

        df_ltf    = pd.DataFrame(
            raw_ltf, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        entry_ind = _compute_entry_tf_indicators(df_ltf)

        closes = entry_ind["closes"]
        atr14  = entry_ind["atr14"]
        opens  = entry_ind["opens"]

        i = len(closes) - 2
        if i < 1 or np.isnan(atr14[i]):
            return "waiting"

        direction = sig["direction"]

        entry_price = float(opens[-1])
        if entry_price <= 0:
            entry_price = float(closes[-1])

        atr_val   = float(atr14[i])
        sl_mult   = sig.get("sl_atr_mult", 1.5)
        sl_dist   = atr_val * sl_mult
        sl_dist   = max(sl_dist, entry_price * 0.003)
        sl_dist   = min(sl_dist, entry_price * 0.05)
        sl = (entry_price - sl_dist) if direction == "LONG" else (entry_price + sl_dist)

        tp_r  = sig.get("tp_r", 2.5)
        risk  = abs(entry_price - sl)
        tp    = (entry_price + risk * tp_r) if direction == "LONG" else (entry_price - risk * tp_r)

        # Try post-only limit first; fall back to market if rejected
        placed = await self._execute_limit_entry(symbol, sig, entry_price, sl, tp, entry_ind)
        if placed == "limit_pending":
            return "waiting"
        elif placed == "market_entered":
            return "entered"
        return "waiting"

    # -------------------------------------------------------------------------
    # Order execution
    # -------------------------------------------------------------------------

    async def _execute_limit_entry(
        self,
        symbol: str,
        sig: dict,
        entry_price: float,
        sl: float,
        tp: float,
        entry_ind: dict,
    ) -> str:
        """
        Try to enter via post-only limit order (maker fee = 0.020%).
        Returns: "limit_pending" | "market_entered" | "waiting"
        """
        async with self._trade_lock:
            # Run all gate checks first (same as market order path)
            gate_ok = await self._gate_checks_pass(symbol, sig, entry_price, sl)
            if not gate_ok:
                return "waiting"

            strategy_id = sig["strategy_id"]
            direction   = sig["direction"]
            side_str    = "buy" if direction == "LONG" else "sell"
            short_sym   = self._short(symbol)

            balance = await self._fetch_balance()
            if balance <= 0:
                return "waiting"

            market_info = self._markets_cache.get(symbol)
            qty, leverage = self.risk.calculate_position_size(
                balance=balance,
                entry_price=entry_price,
                sl_price=sl,
                symbol=symbol,
                market_info=market_info,
            )
            if qty <= 0:
                return "waiting"
            risk_usd    = abs(entry_price - sl) * qty
            margin_used = qty * entry_price / max(leverage, 1)

            try:
                await self.client.set_leverage(symbol, leverage)
            except Exception as exc:
                log.warning("[MTF] set_leverage failed %s: %s", short_sym, exc)

            order = await self.client.place_post_only_limit_order(symbol, side_str, qty, entry_price)
            if order is not None and order.get("id") and order["id"] != "SIM":
                # Store pending limit order
                self._pending_limit_orders[(symbol, strategy_id)] = {
                    "order_id":    order["id"],
                    "entry_price": entry_price,
                    "sl":          sl,
                    "tp":          tp,
                    "qty":         qty,
                    "leverage":    leverage,
                    "side_str":    side_str,
                    "sig":         sig,
                    "margin_used": margin_used,
                    "risk_usd":    risk_usd,
                    "balance":     balance,
                    "cycles":      0,
                }
                log.info(
                    "[MTF] LIMIT PLACED [%s] %s %s @ %.6g qty=%.6f lev=%dx (maker fee)",
                    strategy_id, short_sym, direction, entry_price, qty, leverage,
                )
                try:
                    await self.tg.send_limit_pending(
                        self.agent_id, symbol, direction, entry_price,
                        sl, tp, qty, leverage, margin_used, strategy_id,
                    )
                except Exception as exc:
                    log.warning("[MTF] TG pending notification failed %s: %s", short_sym, exc)
                return "limit_pending"
            elif order is not None and order.get("id") == "SIM":
                # Simulation mode — treat as instant fill
                await self._finalize_limit_fill(symbol, {
                    "entry_price": entry_price, "sl": sl, "tp": tp,
                    "qty": qty, "leverage": leverage, "side_str": side_str,
                    "sig": sig, "margin_used": margin_used, "risk_usd": risk_usd,
                    "balance": balance,
                }, entry_price)
                return "market_entered"
            else:
                # Post-only rejected (price would cross spread) — fall back to market
                log.info("[MTF] Post-only rejected for %s %s — falling back to market", short_sym, direction)
                order = await self.client.place_market_order(symbol, side_str, qty)
                if order is None:
                    return "waiting"
                try:
                    await self.client.place_sl_tp(symbol, side_str, qty, sl, tp)
                except Exception as exc:
                    log.warning("[MTF] SL/TP placement failed %s: %s", short_sym, exc)
                await self._register_opened_position(symbol, sig, entry_price, sl, tp, qty, leverage, margin_used, risk_usd, balance)
                return "market_entered"

    async def _finalize_limit_fill(self, symbol: str, pending: dict, fill_price: float) -> None:
        """Called when a pending post-only limit order is confirmed filled."""
        sig         = pending["sig"]
        sl          = pending["sl"]
        tp          = pending["tp"]
        qty         = pending["qty"]
        leverage    = pending["leverage"]
        side_str    = pending["side_str"]
        margin_used = pending["margin_used"]
        risk_usd    = pending["risk_usd"]
        balance     = pending["balance"]
        strategy_id = sig["strategy_id"]
        direction   = sig["direction"]
        short_sym   = self._short(symbol)

        log.info(
            "[MTF] LIMIT FILLED [%s] %s %s fill=%.6g sl=%.6g tp=%.6g qty=%.6f lev=%dx",
            strategy_id, short_sym, direction, fill_price, sl, tp, qty, leverage,
        )

        try:
            await self.client.place_sl_tp(symbol, side_str, qty, sl, tp)
        except Exception as exc:
            log.warning("[MTF] SL/TP after limit fill failed %s: %s", short_sym, exc)

        await self._register_opened_position(symbol, sig, fill_price, sl, tp, qty, leverage, margin_used, risk_usd, balance)

    async def _register_opened_position(
        self, symbol: str, sig: dict,
        entry_price: float, sl: float, tp: float,
        qty: float, leverage: int, margin_used: float, risk_usd: float, balance: float,
    ) -> None:
        """Register position in state and send TG notification."""
        strategy_id = sig["strategy_id"]
        direction   = sig["direction"]
        short_sym   = self._short(symbol)

        await self.state.register_position(
            self.agent_id, symbol, direction, entry_price, sl, tp, qty
        )
        try:
            pos = self.state._positions[self.agent_id][symbol]
            pos["strategy_id"] = strategy_id
            pos["leverage"]    = leverage
        except Exception:
            pass

        await self.state.set_agent_equity(self.agent_id, balance)

        try:
            await self.tg.send_position_opened(
                self.agent_id, symbol, direction, entry_price, sl, tp, qty, leverage, margin_used,
            )
        except Exception as exc:
            log.warning("[MTF] TG open notification failed: %s", exc)

    async def _gate_checks_pass(
        self, symbol: str, sig: dict, entry_price: float, sl: float
    ) -> bool:
        """Run all pre-trade gate checks. Returns True if trade is allowed."""
        strategy_id = sig["strategy_id"]
        direction   = sig["direction"]
        short_sym   = self._short(symbol)

        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return False
        if await self.state.is_symbol_locked(symbol):
            return False
        positions = await self.state.get_agent_positions(self.agent_id)
        if symbol in positions:
            return False
        max_daily_loss = getattr(cfg, "MTF_MAX_DAILY_LOSS_PCT", cfg.BB_SQUEEZE_MAX_DAILY_LOSS_PCT)
        if not await self.risk.check_daily_loss_limit(self.agent_id, self.state, max_daily_loss):
            return False

        # Funding rate check
        funding = await self._get_funding_rate(symbol)
        if self._funding_blocks_direction(funding, direction):
            return False

        # Max positions per strategy
        scfg      = self._strat_cfg(strategy_id)
        max_pos   = scfg.get("max_positions", 4)
        if not await self.risk.check_position_count(self.agent_id, self.state, max_pos):
            return False

        # Directional balance
        same_dir = sum(1 for p in positions.values() if str(p.get("side", "")).upper() == direction)
        if same_dir >= 3:
            return False

        # Total exposure
        if not await self._check_total_exposure(await self._fetch_balance()):
            return False

        return True

    async def _execute_entry(
        self,
        symbol: str,
        sig: dict,
        entry_price: float,
        sl: float,
        tp: float,
        entry_ind: dict,
    ) -> bool:
        """Gate checks + fire market order. Returns True if trade was placed."""
        async with self._trade_lock:
            return await self._execute_entry_locked(symbol, sig, entry_price, sl, tp)

    async def _execute_entry_locked(
        self,
        symbol: str,
        sig: dict,
        entry_price: float,
        sl: float,
        tp: float,
    ) -> bool:
        """
        Executes under _trade_lock.

        Gate checks (re-validated under lock):
          1. Agent running
          2. Symbol not locked by any agent
          3. No existing MTF position in symbol
          4. Daily loss limit OK
          4b. Funding rate not overcrowded (time gate removed — v11 backtest)
          5. Strategy max positions OK
          6. Total portfolio exposure OK
          7. Positive qty from risk manager

        Returns True if trade placed/simulated.
        """
        strategy_id = sig["strategy_id"]
        direction   = sig["direction"]
        side_str    = "buy" if direction == "LONG" else "sell"
        short_sym   = self._short(symbol)

        # 1. Agent status
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return False

        # 2. Symbol lock (cross-agent)
        if await self.state.is_symbol_locked(symbol):
            log.debug("[MTF] RACE: %s already locked — skip", short_sym)
            return False

        # 3. Already have position
        positions = await self.state.get_agent_positions(self.agent_id)
        if symbol in positions:
            return False

        # 4. Daily loss limit
        max_daily_loss = getattr(cfg, "MTF_MAX_DAILY_LOSS_PCT",
                                  cfg.BB_SQUEEZE_MAX_DAILY_LOSS_PCT)
        if not await self.risk.check_daily_loss_limit(self.agent_id, self.state, max_daily_loss):
            log.info("[MTF] Daily loss limit hit — no new trades today.")
            return False

        # 4b. Time-of-day gate removed: backtest v11 shows UTC 07-20 filter
        #     kills 40-46% of trades with zero PF improvement on all 3 strategies.
        #     Funding rate filter retained as it's a risk guard, not a quality filter.

        # 4c. Funding rate filter — skip overcrowded direction
        funding = await self._get_funding_rate(symbol)
        if self._funding_blocks_direction(funding, direction):
            log.info(
                "[MTF] Funding rate %.4f%% blocks %s on %s",
                (funding or 0) * 100, direction, short_sym,
            )
            return False

        # 5. Strategy max positions
        scfg         = self._strat_cfg(strategy_id)
        max_positions = scfg.get("max_positions", 4)
        if not await self.risk.check_position_count(self.agent_id, self.state, max_positions):
            return False

        # 5b. Directional balance: max 3 positions in the same direction
        # Prevents 4× LONG / 4× SHORT exposure — forces at least 1 hedge slot.
        # E.g. with 4 open: 3L+1S or 2L+2S are OK; 4L+0S is blocked.
        MAX_SAME_DIRECTION = 3
        positions_now = await self.state.get_agent_positions(self.agent_id)
        same_dir_count = sum(
            1 for p in positions_now.values()
            if str(p.get("side", "")).upper() == direction
        )
        if same_dir_count >= MAX_SAME_DIRECTION:
            log.info(
                "[MTF] Directional limit: already %d %s positions (max %d) — skipping %s",
                same_dir_count, direction, MAX_SAME_DIRECTION, short_sym,
            )
            return False

        # 6. Total portfolio exposure
        balance = await self._fetch_balance()
        if not await self._check_total_exposure(balance):
            log.info("[MTF] Total exposure limit reached — skipping %s", short_sym)
            return False

        if balance <= 0:
            return False

        # 7. Position size
        market_info = self._markets_cache.get(symbol)
        qty, leverage = self.risk.calculate_position_size(
            balance=balance,
            entry_price=entry_price,
            sl_price=sl,
            symbol=symbol,
            market_info=market_info,
        )
        if qty <= 0:
            return False

        risk_usd    = abs(entry_price - sl) * qty
        margin_used = qty * entry_price / max(leverage, 1)

        if cfg.SIMULATION_MODE:
            log.info(
                "[MTF] SIM OPEN [%s] %s %s entry=%.6g sl=%.6g tp=%.6g "
                "qty=%.6f risk=$%.4f lev=%dx",
                strategy_id, short_sym, direction,
                entry_price, sl, tp, qty, risk_usd, leverage,
            )
        else:
            # Set leverage
            try:
                await self.client.set_leverage(symbol, leverage)
            except Exception as exc:
                log.warning("[MTF] set_leverage failed %s: %s", short_sym, exc)

            # Place market order (MAX_ORDER_RETRIES=1)
            try:
                order = await self.client.place_market_order(symbol, side_str, qty)
            except Exception as exc:
                err = str(exc)
                if "110007" in err or "not enough" in err.lower() or "InsufficientFunds" in err:
                    log.warning("[MTF] Insufficient margin %s qty=%.6f — skipping", short_sym, qty)
                else:
                    log.error("[MTF] Market order error %s: %s", short_sym, exc)
                return False

            if order is None:
                log.warning("[MTF] Order returned None for %s", short_sym)
                return False

            # Place SL/TP on exchange
            try:
                await self.client.place_sl_tp(symbol, side_str, qty, sl, tp)
            except Exception as exc:
                log.warning("[MTF] SL/TP placement failed %s: %s", short_sym, exc)

            log.info(
                "[MTF] OPENED [%s] %s %s entry=%.6g sl=%.6g tp=%.6g "
                "qty=%.6f risk=$%.4f lev=%dx",
                strategy_id, short_sym, direction,
                entry_price, sl, tp, qty, risk_usd, leverage,
            )

        # Register in shared state (includes simulation)
        await self.state.register_position(
            self.agent_id, symbol, direction, entry_price, sl, tp, qty
        )
        # Store strategy_id in position for close logging
        # (register_position doesn't have a strategy_id param, so patch via state directly)
        try:
            pos = self.state._positions[self.agent_id][symbol]
            pos["strategy_id"] = strategy_id
            pos["leverage"]    = leverage
        except Exception:
            pass

        await self.state.set_agent_equity(self.agent_id, balance)

        # Telegram notification
        try:
            await self.tg.send_position_opened(
                self.agent_id, symbol, direction, entry_price, sl, tp, qty, leverage, margin_used,
            )
        except Exception as exc:
            log.warning("[MTF] TG open notification failed: %s", exc)

        return True
