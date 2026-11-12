"""
multi_agent/agents/smc_agent.py

SMC (Smart Money Concepts) trading agent — complete rewrite.

Strategy:
  Entry TF: 15m  |  HTF: 1h
  1. HTF EMA-200 bias (long above, short below)
  2. Detect Equal Highs / Equal Lows (EQH/EQL) as liquidity pools
  3. Identify Liquidity Sweep: last candle wicks above EQH (sweep_high)
     or below EQL (sweep_low) but closes back inside
  4. Change of Character (CHoCH): close beyond previous swing high/low
     confirming structural reversal
  5. Find valid, unmitigated Order Block (OB) in the reversal direction
     - Bearish OB before bullish impulse (LONG setup)
     - Bullish OB before bearish impulse (SHORT setup)
     - OB candle volume > OB_VOLUME_MULTIPLIER × 20-bar average
  6. Entry: price currently inside the OB zone
  7. SL: beyond sweep candle wick + 0.1% buffer, capped at 2 × ATR
  8. TP: nearest equal level in profit direction, offset by 0.1%
  9. Verify RR >= SMC_MIN_RR
  10. OI delta filter applied before entry
  11. SIMULATION_MODE = True: orders logged, never placed

All configuration is read from multi_agent/config.py.
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

# ─── Logging with rotation ────────────────────────────────────────────────────
_log_path = Path(cfg.SMC_LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("smc_agent")
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

AGENT_ID = "smc_agent"


# ═══════════════════════════════════════════════════════════════════════════════
#  PURE SMC ANALYSIS FUNCTIONS  (no I/O, all operate on pd.DataFrame)
# ═══════════════════════════════════════════════════════════════════════════════

def _calculate_atr(df: pd.DataFrame, period: int = 14) -> float:
    """Standard ATR calculation over the last *period* candles."""
    if df is None or len(df) < 2:
        return 0.0
    highs  = df["high"].astype(float).to_numpy()
    lows   = df["low"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    prev_closes = np.roll(closes, 1)
    prev_closes[0] = closes[0]
    tr = np.maximum(
        highs - lows,
        np.maximum(np.abs(highs - prev_closes), np.abs(lows - prev_closes)),
    )
    relevant = tr[-period:] if len(tr) >= period else tr
    return float(np.mean(relevant)) if len(relevant) > 0 else 0.0


def _detect_market_structure(df_1h: pd.DataFrame) -> str:
    """
    Returns "bullish", "bearish", or "neutral" based on the last 20 1h candles.
    Bullish: last 2 swing highs are higher AND last 2 swing lows are higher.
    Bearish: last 2 swing highs are lower AND last 2 swing lows are lower.
    Otherwise: neutral.
    """
    tail = df_1h.tail(20).reset_index(drop=True)
    if len(tail) < 5:
        return "neutral"

    highs = tail["high"].astype(float).tolist()
    lows  = tail["low"].astype(float).tolist()

    swing_highs: list[tuple[int, float]] = []
    swing_lows:  list[tuple[int, float]] = []

    for i in range(2, len(tail) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] \
                and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            swing_highs.append((i, highs[i]))
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] \
                and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            swing_lows.append((i, lows[i]))

    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        hh = swing_highs[-1][1] > swing_highs[-2][1]
        hl = swing_lows[-1][1]  > swing_lows[-2][1]
        lh = swing_highs[-1][1] < swing_highs[-2][1]
        ll = swing_lows[-1][1]  < swing_lows[-2][1]
        if hh and hl:
            return "bullish"
        if lh and ll:
            return "bearish"
    return "neutral"


def _detect_order_blocks(df: pd.DataFrame, direction: str) -> list[dict]:
    """
    Returns list of valid (un-mitigated) Order Blocks.

    For LONG (bullish OB setup):
      - Find last bearish candle (close < open) before a bullish impulse.
      - Impulse: next candle closes above the bearish candle's high.
      - OB volume must be > OB_VOLUME_MULTIPLIER × 20-bar average.
      - OB is invalid if price has since returned to [low, high] (mitigated).

    For SHORT (bearish OB setup):
      - Find last bullish candle before bearish impulse.
      - Same volume and mitigation checks.

    Scans last 100 candles. Returns sorted newest-first.
    """
    if df is None or len(df) < 5:
        return []

    n          = len(df)
    opens      = df["open"].astype(float).to_numpy()
    highs      = df["high"].astype(float).to_numpy()
    lows       = df["low"].astype(float).to_numpy()
    closes     = df["close"].astype(float).to_numpy()
    volumes    = df["volume"].astype(float).to_numpy() if "volume" in df.columns else np.ones(n)
    timestamps = df["timestamp"].to_numpy() if "timestamp" in df.columns else np.arange(n)

    lookback = min(n - 2, 100)
    start    = n - lookback
    avg_vol_window = min(n, 20)
    avg_vol  = float(np.mean(volumes[-avg_vol_window:])) if avg_vol_window > 0 else 1.0
    vol_mult = float(cfg.SMC_OB_VOLUME_MULTIPLIER)

    obs: list[dict] = []

    for i in range(start, n - 1):
        o_i, h_i, l_i, c_i, v_i = opens[i], highs[i], lows[i], closes[i], volumes[i]

        # Volume gate
        if avg_vol > 0 and v_i < vol_mult * avg_vol:
            continue

        # Impulse confirmation
        o_next, h_next, l_next, c_next = opens[i + 1], highs[i + 1], lows[i + 1], closes[i + 1]

        if direction == "LONG":
            # Bearish candle before bullish impulse
            if c_i >= o_i:
                continue
            if c_next <= h_i:
                continue  # not an impulse above OB high
            # Mitigation check: has price returned inside [l_i, h_i] after i+1?
            mitigated = False
            for j in range(i + 2, n):
                if lows[j] <= h_i and highs[j] >= l_i:
                    mitigated = True
                    break
            if mitigated:
                continue
            obs.append({
                "direction":    "LONG",
                "ob_high":      float(h_i),
                "ob_low":       float(l_i),
                "ob_mid":       float((h_i + l_i) / 2),
                "ts":           int(timestamps[i]),
                "bar_idx":      i,
                "volume_ratio": float(v_i / max(avg_vol, 1e-9)),
            })

        elif direction == "SHORT":
            # Bullish candle before bearish impulse
            if c_i <= o_i:
                continue
            if c_next >= l_i:
                continue  # not an impulse below OB low
            mitigated = False
            for j in range(i + 2, n):
                if lows[j] <= h_i and highs[j] >= l_i:
                    mitigated = True
                    break
            if mitigated:
                continue
            obs.append({
                "direction":    "SHORT",
                "ob_high":      float(h_i),
                "ob_low":       float(l_i),
                "ob_mid":       float((h_i + l_i) / 2),
                "ts":           int(timestamps[i]),
                "bar_idx":      i,
                "volume_ratio": float(v_i / max(avg_vol, 1e-9)),
            })

    obs.sort(key=lambda x: x["ts"], reverse=True)
    return obs


def _detect_equal_levels(df: pd.DataFrame) -> tuple[list[float], list[float]]:
    """
    Returns (equal_highs, equal_lows).
    Equal Highs: pairs of swing highs within SMC_EQ_LEVEL_TOLERANCE_PCT of each other.
    Equal Lows:  pairs of swing lows  within the same tolerance.
    Scans last SMC_EQ_LOOKBACK_BARS candles.
    """
    lookback = int(cfg.SMC_EQ_LOOKBACK_BARS)
    tail = df.tail(lookback).reset_index(drop=True)
    if len(tail) < 5:
        return [], []

    tol    = float(cfg.SMC_EQ_LEVEL_TOLERANCE_PCT) / 100.0
    highs  = tail["high"].astype(float).tolist()
    lows   = tail["low"].astype(float).tolist()

    # Collect swing highs / lows
    swing_highs: list[float] = []
    swing_lows:  list[float] = []
    for i in range(2, len(tail) - 2):
        if highs[i] > highs[i - 1] and highs[i] > highs[i - 2] \
                and highs[i] > highs[i + 1] and highs[i] > highs[i + 2]:
            swing_highs.append(float(highs[i]))
        if lows[i] < lows[i - 1] and lows[i] < lows[i - 2] \
                and lows[i] < lows[i + 1] and lows[i] < lows[i + 2]:
            swing_lows.append(float(lows[i]))

    eq_highs: list[float] = []
    for i in range(len(swing_highs)):
        for j in range(i + 1, len(swing_highs)):
            ref = swing_highs[i]
            if ref > 0 and abs(swing_highs[i] - swing_highs[j]) / ref <= tol:
                eq_highs.append((swing_highs[i] + swing_highs[j]) / 2)

    eq_lows: list[float] = []
    for i in range(len(swing_lows)):
        for j in range(i + 1, len(swing_lows)):
            ref = swing_lows[i]
            if ref > 0 and abs(swing_lows[i] - swing_lows[j]) / ref <= tol:
                eq_lows.append((swing_lows[i] + swing_lows[j]) / 2)

    return sorted(set(eq_highs)), sorted(set(eq_lows))


def _detect_liquidity_sweep(
    df: pd.DataFrame,
    equal_highs: list[float],
    equal_lows: list[float],
) -> Optional[dict]:
    """
    Returns sweep info or None.
    Sweep of EQH: last candle high > EQH level BUT close < EQH level (wick above).
    Sweep of EQL: last candle low  < EQL level BUT close > EQL level (wick below).
    Only checks the LAST candle of the DataFrame.
    """
    if df is None or len(df) < 2:
        return None

    last = df.iloc[-1]
    h    = float(last["high"])
    l    = float(last["low"])
    c    = float(last["close"])
    idx  = len(df) - 1

    # Sweep high (price wicked above EQH but closed below)
    for eqh in sorted(equal_highs, reverse=True):
        if h > eqh and c < eqh:
            return {"type": "sweep_high", "level": float(eqh), "sweep_candle_idx": idx}

    # Sweep low (price wicked below EQL but closed above)
    for eql in sorted(equal_lows):
        if l < eql and c > eql:
            return {"type": "sweep_low", "level": float(eql), "sweep_candle_idx": idx}

    return None


def _detect_choch(
    df: pd.DataFrame,
    structure: str,
    after_index: int,
) -> Optional[dict]:
    """
    Detect Change of Character (CHoCH) after *after_index*.

    After a sweep of EQL in a bullish structure:
      CHoCH UP = price closes above previous swing high (confirms bullish reversal).

    After a sweep of EQH in a bearish structure:
      CHoCH DOWN = price closes below previous swing low (confirms bearish reversal).

    Returns {"type": "choch_up"|"choch_down", "level": float, "candle_idx": int} or None.
    """
    if df is None or after_index < 1 or after_index >= len(df):
        return None

    highs  = df["high"].astype(float).to_numpy()
    lows   = df["low"].astype(float).to_numpy()
    closes = df["close"].astype(float).to_numpy()
    n      = len(df)

    # Find most recent swing high/low before the sweep candle
    prev_swing_high = float(np.max(highs[max(0, after_index - 20): after_index]))
    prev_swing_low  = float(np.min(lows[max(0, after_index - 20): after_index]))

    for i in range(after_index + 1, n):
        c_i = closes[i]
        if structure == "bullish" and c_i > prev_swing_high:
            return {"type": "choch_up", "level": float(prev_swing_high), "candle_idx": i}
        if structure == "bearish" and c_i < prev_swing_low:
            return {"type": "choch_down", "level": float(prev_swing_low), "candle_idx": i}

    return None


def _find_swing_tp(df: pd.DataFrame, entry: float, direction: str, sl: float, min_rr: float) -> float:
    """
    SMC principle: TP placed AT the nearest significant swing level AHEAD of entry.
    The TP is placed just INSIDE (0.15% before) the structural level — NOT beyond it.

    - LONG: nearest swing HIGH above entry (resistance) minus 0.15% offset
    - SHORT: nearest swing LOW below entry (support) plus 0.15% offset

    Falls back to min_rr * risk if no structural level found in last 100 bars.
    Minimum RR is enforced only as a filter (trade rejected), not by pushing TP.
    """
    risk = abs(entry - sl)
    tp_offset = 0.0015   # 0.15% inside the structure level

    highs = df["high"].astype(float).to_numpy()
    lows  = df["low"].astype(float).to_numpy()
    n = min(len(df), 100)

    if direction == "LONG":
        # Find ALL swing highs above entry — take the NEAREST one (smallest)
        swing_highs = []
        for i in range(2, n - 2):
            idx = len(df) - n + i
            h = highs[idx]
            if h > highs[idx - 1] and h > highs[idx - 2] and h > highs[idx + 1] and h > highs[idx + 2]:
                if h > entry * 1.003:   # must be at least 0.3% above entry
                    swing_highs.append(h)
        if swing_highs:
            nearest = min(swing_highs)
            return float(nearest * (1.0 - tp_offset))   # just BELOW resistance
        return float(entry + risk * min_rr)   # fallback only
    else:
        # Find ALL swing lows below entry — take the NEAREST one (largest)
        swing_lows = []
        for i in range(2, n - 2):
            idx = len(df) - n + i
            l = lows[idx]
            if l < lows[idx - 1] and l < lows[idx - 2] and l < lows[idx + 1] and l < lows[idx + 2]:
                if l < entry * 0.997:   # must be at least 0.3% below entry
                    swing_lows.append(l)
        if swing_lows:
            nearest = max(swing_lows)
            return float(nearest * (1.0 + tp_offset))   # just ABOVE support
        return float(entry - risk * min_rr)   # fallback only


def _calculate_sl_tp(
    sweep_candle_row: pd.Series,
    direction: str,
    entry: float,
    atr: float,
    eq_levels: list[float],
    df: pd.DataFrame,
) -> Optional[tuple[float, float, float]]:
    """
    Backtest winner (Variant B): SL_MODE=body_50pct, SL_BUFFER=0.10%, MIN_RR=1.5.
    SL = midpoint of sweep candle body + 0.10% buffer.
    TP = nearest swing level (0.15% inside structure), min RR=1.5.
    Maximum SL distance capped at SMC_MAX_SL_ATR_MULT × ATR.
    Returns (sl_price, tp_price, rr) or None if RR < SMC_MIN_RR.
    """
    sl_mode     = cfg.SMC_AGENT_CONFIG.get("SL_MODE", "body_50pct")
    sl_buf      = float(cfg.SMC_AGENT_CONFIG["SL_BUFFER_PCT"]) / 100.0
    max_sl_dist = float(cfg.SMC_AGENT_CONFIG["MAX_SL_ATR_MULT"]) * atr
    min_rr      = float(cfg.SMC_AGENT_CONFIG["MIN_RR"])

    sweep_low   = float(sweep_candle_row.get("low",  entry))
    sweep_high  = float(sweep_candle_row.get("high", entry))
    candle_open  = float(sweep_candle_row.get("open",  entry))
    candle_close = float(sweep_candle_row.get("close", entry))

    if sl_mode == "body_50pct":
        # SL at midpoint of sweep candle body — tighter, more trades (Variant B winner)
        body_mid = (min(candle_open, candle_close) + max(candle_open, candle_close)) / 2.0
        if direction == "LONG":
            sl = body_mid * (1.0 - sl_buf)
            if (entry - sl) > max_sl_dist:
                sl = entry - max_sl_dist
        else:
            sl = body_mid * (1.0 + sl_buf)
            if (sl - entry) > max_sl_dist:
                sl = entry + max_sl_dist
    else:
        # full_wick / sweep_wick: SL beyond the full sweep wick
        if direction == "LONG":
            sl = sweep_low * (1.0 - sl_buf)
            if (entry - sl) > max_sl_dist:
                sl = entry - max_sl_dist
        else:
            sl = sweep_high * (1.0 + sl_buf)
            if (sl - entry) > max_sl_dist:
                sl = entry + max_sl_dist

    risk = abs(entry - sl)
    if risk <= 0:
        return None

    # TP: nearest structural swing level, placed just inside (not beyond) structure
    # SMC principle: TP AT the next liquidity pool (EQH/EQL), not past it
    tp = _find_swing_tp(df, entry, direction, sl, min_rr)

    # Prefer EQ level if closer to entry than swing TP (more precise structure)
    if direction == "LONG":
        eq_above = [lvl * (1.0 - 0.0015) for lvl in eq_levels
                    if lvl > entry * 1.003]
        if eq_above:
            eq_tp = min(eq_above)
            if eq_tp < tp:          # EQ level is closer → use it (TP at nearer structure)
                tp = eq_tp
    else:
        eq_below = [lvl * (1.0 + 0.0015) for lvl in eq_levels
                    if lvl < entry * 0.997]
        if eq_below:
            eq_tp = max(eq_below)
            if eq_tp > tp:          # EQ level is closer → use it
                tp = eq_tp

    reward = abs(tp - entry)
    rr = reward / risk
    if rr < min_rr:                 # filter: structural TP doesn't give enough RR
        return None
    return sl, tp, rr


# ═══════════════════════════════════════════════════════════════════════════════
#  SMCAgent
# ═══════════════════════════════════════════════════════════════════════════════

class SMCAgent:
    """
    SMC trading agent for the multi-agent system.

    Independent from FVGAgent: different strategy, different config namespace
    (SMC_* in config.py), but shares the same infrastructure singletons.
    SIMULATION_MODE = True everywhere.
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

        # Cooldown: {symbol: global_bar_count at last close}
        self._last_close_bar: dict[str, int] = {}
        self._global_bar_count: int          = 0

        # Symbol list
        self._symbols: list[str] = []
        self._last_symbols_refresh: float = 0.0
        self._markets_cache: dict = {}

    # ── Helpers ───────────────────────────────────────────────────────────

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
            log.info("[SMC] Daily stats reset for %s", today)

    def _in_cooldown(self, symbol: str) -> bool:
        last = self._last_close_bar.get(symbol)
        if last is None:
            return False
        return (self._global_bar_count - last) < int(cfg.SMC_COOLDOWN_BARS)

    def _record_close_bar(self, symbol: str) -> None:
        self._last_close_bar[symbol] = self._global_bar_count

    # ── Symbol management ─────────────────────────────────────────────────

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
            self._symbols = [sym for sym, _ in usdt_perp[: cfg.TOP_N_COINS]]
            self._last_symbols_refresh = now
            self._markets_cache = dict(self.client._exchange.markets or {})
            log.info("[SMC] %d symbols loaded.", len(self._symbols))
        except Exception as exc:
            log.error("[SMC] Symbol refresh failed: %s", exc)

    # ── Open-position monitor ─────────────────────────────────────────────

    async def _check_open_positions(self) -> None:
        positions = await self.state.get_agent_positions(self.agent_id)

        for symbol, pos in list(positions.items()):
            try:
                ticker = await self.client.fetch_ticker(symbol)
                price  = float(ticker.get("last") or 0)
                if price <= 0:
                    continue
                side   = str(pos.get("side", "LONG"))
                sl     = float(pos.get("sl", 0))
                tp     = float(pos.get("tp", 0))
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
                    await self._close_position(symbol, pos, price, reason)
            except Exception as exc:
                log.warning("[SMC] Position check error %s: %s", symbol, exc)

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
        self._record_close_bar(symbol)

        short_sym = symbol.split("/")[0]
        log.info(
            "[SMC] CLOSED %s %s entry=%.6g exit=%.6g pnl=$%.4f reason=%s",
            short_sym, side, entry, exit_price, pnl_usd, reason,
        )
        await self.tg.send_position_closed(
            self.agent_id, symbol, side, entry, exit_price, pnl_usd, pnl_pct, reason
        )

    # ── Full SMC scan for one symbol ──────────────────────────────────────

    async def scan_symbol(self, symbol: str) -> Optional[dict]:
        """
        Full SMC scan:
        1. Load 1h data, detect HTF structure.
        2. Load 15m data.
        3. Detect Equal Highs/Lows.
        4. Check for recent Liquidity Sweep.
        5. Check for CHoCH confirmation.
        6. Find valid Order Block in direction.
        7. Check if price is IN the Order Block.
        8. Apply OI filter.
        9. Calculate SL/TP with ATR cap.
        10. Verify RR >= SMC_MIN_RR.
        11. Return signal dict or None.
        """
        short_name = symbol.replace("/USDT:USDT", "USDT")

        # Gates
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return None
        if not await self.risk.check_position_count(
            self.agent_id, self.state, cfg.SMC_MAX_POSITIONS
        ):
            return None
        if not await self.risk.check_daily_loss_limit(
            self.agent_id, self.state, cfg.SMC_MAX_DAILY_LOSS_PCT
        ):
            return None
        if await self.state.is_symbol_locked(symbol):
            return None
        if self._in_cooldown(symbol):
            return None

        # Fetch candles
        try:
            raw_htf   = await self.client.fetch_ohlcv(
                symbol, cfg.SMC_HTF_TIMEFRAME, cfg.OHLCV_LIMIT_HTF
            )
            raw_entry = await self.client.fetch_ohlcv(
                symbol, cfg.SMC_ENTRY_TIMEFRAME, cfg.OHLCV_LIMIT_ENTRY
            )
        except Exception as exc:
            log.debug("[SMC] OHLCV fetch failed %s: %s", short_name, exc)
            return None

        if not raw_htf or not raw_entry:
            return None
        if len(raw_htf) < cfg.SMC_EMA_PERIOD + 10 or len(raw_entry) < 30:
            return None

        df_1h = pd.DataFrame(
            raw_htf, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_15m = pd.DataFrame(
            raw_entry, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_1h["timestamp"]  = df_1h["timestamp"].astype(int)
        df_15m["timestamp"] = df_15m["timestamp"].astype(int)

        # Step 1: HTF EMA-200 bias
        ema_series = df_1h["close"].ewm(span=cfg.SMC_EMA_PERIOD, adjust=False).mean()
        last_1h_close = float(df_1h["close"].iloc[-1])
        ema200        = float(ema_series.iloc[-1])
        if pd.isna(ema200) or ema200 == 0:
            return None
        if last_1h_close > ema200:
            htf_direction = "LONG"
        elif last_1h_close < ema200:
            htf_direction = "SHORT"
        else:
            return None

        # HTF market structure
        structure = _detect_market_structure(df_1h)

        # Step 3: Equal Highs/Lows on 1h
        eq_highs, eq_lows = _detect_equal_levels(df_1h)
        if not eq_highs and not eq_lows:
            return None

        # Step 4: Liquidity Sweep on 1h
        sweep = _detect_liquidity_sweep(df_1h, eq_highs, eq_lows)
        if sweep is None:
            return None

        sweep_type  = sweep["type"]
        sweep_level = sweep["level"]
        sweep_idx   = sweep["sweep_candle_idx"]

        # Directional consistency check
        if htf_direction == "LONG" and sweep_type != "sweep_low":
            return None
        if htf_direction == "SHORT" and sweep_type != "sweep_high":
            return None

        # Step 5: CHoCH confirmation
        choch = _detect_choch(df_1h, structure, after_index=sweep_idx)
        if choch is None:
            # Relaxed: accept sweep alone when structure is already aligned
            if not (
                (htf_direction == "LONG"  and structure == "bullish")
                or (htf_direction == "SHORT" and structure == "bearish")
            ):
                return None

        # Step 6: Order Block in the reversal direction on 15m
        obs = _detect_order_blocks(df_15m, htf_direction)
        if not obs:
            return None
        ob = obs[0]  # newest/highest-volume OB

        # Step 7: Current price inside OB?
        try:
            ticker = await self.client.fetch_ticker(symbol)
            current_price = float(ticker.get("last") or 0)
        except Exception:
            current_price = float(df_15m["close"].iloc[-1])

        if current_price <= 0:
            return None
        if not (ob["ob_low"] <= current_price <= ob["ob_high"]):
            return None

        # Step 8: OI filter
        oi_allowed, oi_reason = await self.oi_filter.check(symbol, htf_direction, current_price)
        if not oi_allowed:
            log.info("[SMC] OI filter BLOCK %s %s: %s", short_name, htf_direction, oi_reason)
            return None

        # Step 9: SL/TP
        atr = _calculate_atr(df_15m, period=14)
        sweep_candle_row = df_1h.iloc[sweep_idx]
        eq_levels_for_tp = eq_highs if htf_direction == "LONG" else eq_lows

        sl_tp = _calculate_sl_tp(
            sweep_candle_row=sweep_candle_row,
            direction=htf_direction,
            entry=current_price,
            atr=atr,
            eq_levels=eq_levels_for_tp,
            df=df_15m,
        )
        if sl_tp is None:
            return None
        sl, tp, rr = sl_tp

        # Step 10: RR range check
        if not (float(cfg.SMC_MIN_RR) <= rr <= float(cfg.SMC_MAX_RR)):
            return None

        score = min(rr / float(cfg.SMC_MAX_RR) * 100.0, 100.0)

        return {
            "symbol":        symbol,
            "direction":     htf_direction,
            "entry_price":   current_price,
            "sl_price":      sl,
            "tp_price":      tp,
            "rr":            rr,
            "score":         score,
            "ob":            ob,
            "sweep":         sweep,
            "structure":     structure,
            "atr":           atr,
        }

    # ── Execute trade ─────────────────────────────────────────────────────

    async def _execute_trade(self, signal: dict) -> None:
        async with self._trade_lock:
            await self._execute_trade_locked(signal)

    async def _execute_trade_locked(self, signal: dict) -> None:
        symbol      = signal["symbol"]
        side        = signal["direction"]
        entry_price = float(signal["entry_price"])
        sl          = float(signal["sl_price"])
        tp          = float(signal["tp_price"])
        rr          = float(signal["rr"])
        score       = float(signal["score"])
        short_name  = symbol.replace("/USDT:USDT", "USDT")
        ccxt_side   = "buy" if side == "LONG" else "sell"

        # Final gates (re-checked under _trade_lock to prevent race condition)
        if await self.state.is_symbol_locked(symbol):
            log.debug("[SMC] RACE: %s already locked — skip", symbol)
            return
        if not await self.risk.check_position_count(
            self.agent_id, self.state, cfg.SMC_MAX_POSITIONS
        ):
            return

        # Fetch live balance from Bybit, apply SMC capital fraction (60% of total)
        try:
            raw_bal = await self.client.fetch_balance()
            total_bal = float(
                raw_bal.get("USDT", {}).get("total", 0)
                or raw_bal.get("total", {}).get("USDT", 0)
                or 0
            )
            equity = total_bal * cfg.SMC_CAPITAL_FRACTION   # 60% of account
        except Exception:
            equity = await self.state.get_agent_equity(self.agent_id)
        if equity <= 0:
            log.error("[SMC] Zero equity — cannot trade")
            return
        if not await self.risk.check_margin_usage(self.agent_id, equity, self.state):
            return

        market = self._markets_cache.get(symbol, {})
        qty, leverage = self.risk.calculate_position_size(
            balance=equity,
            entry_price=entry_price,
            sl_price=sl,
            symbol=symbol,
            market_info=market if market else None,
        )
        if qty <= 0:
            log.warning("[SMC] qty=0 for %s — skipping", short_name)
            return

        notional    = qty * entry_price
        margin_used = notional / max(1.0, float(leverage))

        log.info(
            "[SMC] SIGNAL %s %s entry=%.6g sl=%.6g tp=%.6g rr=%.2f score=%.1f "
            "qty=%.6g lev=%d margin=$%.4f",
            short_name, side, entry_price, sl, tp, rr, score, qty, leverage, margin_used,
        )

        await self.tg.send_signal(
            self.agent_id, symbol, side, entry_price, sl, tp, rr, score, qty, leverage
        )

        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception as exc:
            log.warning("[SMC] set_leverage failed %s: %s", short_name, exc)

        try:
            order = await self.client.place_market_order(symbol, ccxt_side, qty)
        except Exception as exc:
            err_str = str(exc)
            if "110007" in err_str or "not enough" in err_str.lower() or "InsufficientFunds" in err_str:
                log.warning("[SMC] Insufficient margin for %s qty=%.6f — skipping trade", short_name, qty)
            else:
                log.error("[SMC] Market order error %s: %s", short_name, exc)
            return
        if order is None:
            log.error("[SMC] Market order failed %s", short_name)
            await self.tg.send_error(self.agent_id, f"Market order failed {short_name}")
            return

        await self.state.register_position(
            self.agent_id, symbol, side, entry_price, sl, tp, qty
        )

        try:
            await self.client.place_sl_tp(symbol, ccxt_side, qty, sl, tp)
        except Exception as exc:
            log.warning("[SMC] SL/TP placement failed %s: %s", short_name, exc)

        await self.tg.send_position_opened(
            self.agent_id, symbol, side, entry_price, sl, tp, qty, leverage, margin_used
        )
        self.daily_trades += 1
        log.info("[SMC] Position opened: %s %s @ %.6g", short_name, side, entry_price)

    # ── Main scan cycle ───────────────────────────────────────────────────

    async def _run_scan_cycle(self) -> None:
        """Scan all symbols for SMC signals (batch of 5)."""
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return

        self._global_bar_count += 1
        positions = await self.state.get_agent_positions(self.agent_id)
        if len(positions) >= cfg.SMC_MAX_POSITIONS:
            return

        batch_size  = 5
        batch_sleep = float(getattr(cfg, "BATCH_SLEEP_SEC", 2.5))
        symbols     = list(self._symbols)

        signals: list[dict] = []
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i: i + batch_size]
            results = await asyncio.gather(
                *[self.scan_symbol(s) for s in batch],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, dict):
                    signals.append(res)
                elif isinstance(res, Exception):
                    log.error("[SMC] scan_symbol exception: %s", res)
            await self.heartbeat()  # Heartbeat during long scan
            await asyncio.sleep(batch_sleep)

        log.info("[SMC] Scan complete. %d symbols. Signals: %d", len(symbols), len(signals))
        if not signals:
            return

        signals.sort(key=lambda s: s.get("score", 0.0), reverse=True)
        positions = await self.state.get_agent_positions(self.agent_id)
        available = cfg.SMC_MAX_POSITIONS - len(positions)
        log.info(
            "[SMC] %d signal(s) found, %d slot(s) available",
            len(signals), available,
        )
        for sig in signals[:available]:
            pos_now = await self.state.get_agent_positions(self.agent_id)
            if sig["symbol"] in pos_now:
                continue
            await self._execute_trade(sig)

    # ── Main loop ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Main agent loop — same structure as FVGAgent."""
        log.info("[SMC] Agent %s starting.", self.agent_id)
        await self.state.set_agent_status(self.agent_id, "running")
        await self._refresh_symbols_if_needed()

        while True:
            try:
                await self.heartbeat()
                await self._reset_daily_if_needed()
                await self._check_open_positions()
                await self._refresh_symbols_if_needed()
                await self._run_scan_cycle()
                await asyncio.sleep(cfg.SCAN_INTERVAL_SEC)

            except asyncio.CancelledError:
                log.info("[SMC] Agent %s cancelled.", self.agent_id)
                await self.state.set_agent_status(self.agent_id, "stopped")
                break
            except Exception as exc:
                log.error("[SMC] Loop error: %s", exc, exc_info=True)
                await self.tg.send_error(self.agent_id, str(exc))
                await asyncio.sleep(10)

    # Alias kept for master_agent compatibility
    async def run(self) -> None:
        await self.start()
