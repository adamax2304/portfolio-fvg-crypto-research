"""
multi_agent/agents/fvg_agent.py

FVG (Fair Value Gap) trading agent — complete rewrite.

Strategy (identical to fvg_bot.py v2.6):
  - Entry TF: 15m  |  HTF bias: 1h EMA-200
  - LONG above EMA-200, SHORT below
  - Detect unmitigated FVGs via detect_fvgs_v2 (BigBeluga percentile filter)
  - Entry only in the mid-zone (40-60% of gap)
  - Pre-alert when price is within PRE_ALERT_DISTANCE_PCT of the mid-zone
  - SL: anchored to confirmed structural support/resistance + buffer
  - TP: nearest HVN / swing level - offset
  - Composite score filter before entry
  - Portfolio correlation filter (same as original bot)
  - OI delta filter applied before each entry
  - SIMULATION_MODE = True: orders are logged, never placed live
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import multi_agent.config as cfg

from multi_agent.shared.fvg_core import (
    FVG,
    SignalCandidate,
    detect_fvgs_v2,
    _select_active_fvgs,
    price_in_mid_zone,
    distance_to_mid_zone_pct,
    evaluate_entry_reaction,
    composite_signal_score,
    compute_ema,
    build_symbol_profile,
    collect_swing_levels,
    find_swing_levels,
    build_local_volume_profile,
    extract_structural_levels,
    _atr_pct_from_arrays,
    FVG_CORE_CONFIG,
    BACKTEST_CONFIG,
)

if TYPE_CHECKING:
    from multi_agent.shared.bybit_client import BybitClient
    from multi_agent.shared.state_manager import StateManager
    from multi_agent.shared.telegram_notifier import MultiAgentTelegramNotifier
    from multi_agent.shared.risk_manager import RiskManager
    from multi_agent.filters.oi_delta_filter import OIDeltaFilter

# ─── Logging with rotation ────────────────────────────────────────────────────
_log_path = Path(cfg.FVG_LOG_FILE)
_log_path.parent.mkdir(parents=True, exist_ok=True)

log = logging.getLogger("fvg_agent")
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

AGENT_ID = "fvg_agent"

# Convenience aliases so the copied logic below can reference CONFIG exactly
# as in fvg_bot.py without renaming every key access.
CONFIG = FVG_CORE_CONFIG


# ─── Price-rounding helper (mirrors FVGTradingBot.round_price) ────────────────

def _round_price(price: float, market: dict) -> float:
    precision = market.get("precision", {}).get("price", 8)
    if isinstance(precision, (int, float)) and float(precision) >= 1.0:
        return round(price, int(precision))
    elif isinstance(precision, float) and float(precision) < 1.0:
        step = float(precision)
        return round(price / step) * step
    return round(price, 4)


# ─── Build trade plan (standalone version of _build_trade_plan_v2) ────────────

def _build_trade_plan_v2(
    fvg: FVG,
    df_5m: pd.DataFrame,
    market: dict,
    symbol_profile: Optional[dict] = None,
) -> dict[str, float]:
    """
    Exact port of FVGTradingBot._build_trade_plan_v2 from fvg_bot.py.
    Uses FVG_CORE_CONFIG / CONFIG for all parameter lookups.
    """
    swing_lookback = int(CONFIG.get("SWING_LOOKBACK_BARS", 120))
    symbol_profile = symbol_profile or build_symbol_profile(df_5m)
    swing_lows, swing_highs = collect_swing_levels(df_5m, lookback=swing_lookback)
    sw_low, sw_high = find_swing_levels(df_5m, lookback=swing_lookback)
    supports, resistances = extract_structural_levels(
        df_5m,
        lookback=max(120, swing_lookback),
        tolerance_pct=float(CONFIG.get("STRUCTURE_TOUCH_TOL_PCT", 0.06)),
    )
    volume_profile = build_local_volume_profile(
        df_5m,
        lookback=int(CONFIG["VP_LOOKBACK_BARS"]),
        bins=int(CONFIG["VP_NUM_BINS"]),
    )
    volume_levels = list(volume_profile.get("levels", []))
    poc_level = float(volume_profile.get("poc", 0.0) or 0.0)
    if poc_level > 0 and poc_level not in volume_levels:
        volume_levels.append(poc_level)
        volume_levels = sorted(volume_levels)

    entry_price = _round_price((fvg.low + fvg.high) / 2.0, market)
    sl_buffer_pct = max(
        float(CONFIG.get("SL_LEVEL_BUFFER_MIN_PCT", 0.10)),
        min(
            float(CONFIG.get("SL_NEAREST_LEVEL_BUFFER_PCT", 0.15)),
            float(CONFIG.get("SL_LEVEL_BUFFER_MAX_PCT", 0.20)),
        ),
    )
    tp_offset_pct = max(
        float(CONFIG.get("TP_OFFSET_MIN_PCT", 0.10)),
        min(
            float(CONFIG.get("TP_HVN_OFFSET_PCT", 0.12)),
            float(CONFIG.get("TP_OFFSET_MAX_PCT", 0.15)),
        ),
    )
    sl_buffer = sl_buffer_pct / 100.0
    tp_offset = tp_offset_pct / 100.0
    zone_low  = float(fvg.low)
    zone_high = float(fvg.high)

    tail = df_5m.tail(max(40, swing_lookback)).copy()
    highs_tail  = tail["high"].astype(float).to_numpy()
    lows_tail   = tail["low"].astype(float).to_numpy()
    closes_tail = tail["close"].astype(float).to_numpy()
    atr_pct = _atr_pct_from_arrays(highs_tail, lows_tail, closes_tail)
    atr_abs = (atr_pct / 100.0) * max(entry_price, 1e-9)
    stop_mult = max(0.75, float(symbol_profile.get("stop_mult", 1.0)))
    min_stop_distance = max(entry_price * 0.008 * stop_mult, atr_abs * stop_mult)
    min_offset = float(CONFIG.get("SL_LEVEL_MIN_OFFSET_PCT", 0.10)) / 100.0

    def pick_sl_level(
        levels: list[tuple[float, int]], direction: str
    ) -> tuple[float, int]:
        if not levels:
            fallback = float(np.min(lows_tail)) if direction == "LONG" else float(np.max(highs_tail))
            return fallback, 1
        scored: list[tuple[float, float, int]] = []
        for lvl, touches in levels:
            if lvl <= 0:
                continue
            if direction == "LONG":
                if lvl >= zone_low * (1.0 - min_offset):
                    continue
                if lvl >= entry_price:
                    continue
                dist_pct = (entry_price - lvl) / max(entry_price, 1e-9) * 100.0
            else:
                if lvl <= zone_high * (1.0 + min_offset):
                    continue
                if lvl <= entry_price:
                    continue
                dist_pct = (lvl - entry_price) / max(entry_price, 1e-9) * 100.0
            if dist_pct <= 0.12 or dist_pct > 5.0:
                continue
            score = touches * 1.7 - abs(dist_pct - 1.2) * 0.8
            scored.append((score, lvl, touches))
        if not scored:
            fallback = float(np.min(lows_tail)) if direction == "LONG" else float(np.max(highs_tail))
            return fallback, 1
        scored.sort(key=lambda item: (item[0], item[2]), reverse=True)
        return float(scored[0][1]), int(scored[0][2])

    def pick_tp_level(candidates: list[tuple[float, float]]) -> float:
        if not candidates:
            return entry_price
        ranked: list[tuple[float, float]] = []
        for lvl, base_score in candidates:
            if lvl <= 0:
                continue
            dist_pct = abs(lvl - entry_price) / max(entry_price, 1e-9) * 100.0
            if dist_pct < 0.20:
                continue
            score = base_score - dist_pct * 0.08
            ranked.append((score, lvl))
        if not ranked:
            return float(candidates[0][0])
        ranked.sort(key=lambda x: x[0], reverse=True)
        return float(ranked[0][1])

    nearest_level: float = 0.0
    structure_touches: int = 1
    target_level: float = entry_price

    if fvg.type == "bullish":
        support_levels = supports[:]
        if not support_levels:
            support_levels = [(float(x), 1) for x in swing_lows]
        nearest_level, structure_touches = pick_sl_level(support_levels, "LONG")
        sl_raw = nearest_level * (1.0 - sl_buffer)
        if (entry_price - sl_raw) < min_stop_distance:
            sl_raw = entry_price - min_stop_distance

        target_floor = max(entry_price, zone_high)
        target_candidates: list[tuple[float, float]] = []
        for level, touches in resistances:
            if level > target_floor:
                target_candidates.append((float(level), 1.2 + touches * 0.15))
        for level in swing_highs:
            if level > target_floor:
                target_candidates.append((float(level), 1.0))
        for level in volume_levels:
            if level > target_floor:
                target_candidates.append((float(level), 1.35))
        if not target_candidates:
            fallback = (
                sw_high if sw_high > target_floor
                else max(float(df_5m["high"].tail(20).max()), zone_high * 1.01)
            )
            target_candidates = [(float(fallback), 0.8)]
        target_level = pick_tp_level(target_candidates)
        tp_raw = target_level * (1.0 - tp_offset)
        if tp_raw <= entry_price:
            tp_raw = entry_price * (1.0 + max(tp_offset, 0.003))

    else:  # bearish
        resistance_levels = resistances[:]
        if not resistance_levels:
            resistance_levels = [(float(x), 1) for x in swing_highs]
        nearest_level, structure_touches = pick_sl_level(resistance_levels, "SHORT")
        sl_raw = nearest_level * (1.0 + sl_buffer)
        if (sl_raw - entry_price) < min_stop_distance:
            sl_raw = entry_price + min_stop_distance

        target_ceiling = min(entry_price, zone_low)
        target_candidates = []
        for level, touches in supports:
            if 0 < level < target_ceiling:
                target_candidates.append((float(level), 1.2 + touches * 0.15))
        for level in swing_lows:
            if 0 < level < target_ceiling:
                target_candidates.append((float(level), 1.0))
        for level in volume_levels:
            if 0 < level < target_ceiling:
                target_candidates.append((float(level), 1.35))
        if not target_candidates:
            fallback = (
                sw_low if 0 < sw_low < target_ceiling
                else min(float(df_5m["low"].tail(20).min()), zone_low * 0.99)
            )
            target_candidates = [(float(fallback), 0.8)]
        target_level = pick_tp_level(target_candidates)
        tp_raw = target_level * (1.0 + tp_offset)
        if tp_raw >= entry_price:
            tp_raw = entry_price * (1.0 - max(tp_offset, 0.003))

    # SMC principle: TP stays AT the structure level — do NOT push beyond it.
    # If structure-based TP doesn't give enough RR, the trade is filtered out
    # downstream by the RR check (MIN_TP_RR). This prevents TP landing past
    # resistance where price rarely reaches after bouncing.
    _sl_distance = abs(entry_price - sl_raw)
    _min_tp_rr   = float(CONFIG.get("MIN_TP_RR", 1.0))
    _min_tp_dist = _sl_distance * _min_tp_rr
    # Only enforce a hard minimum if no structural level was found at all
    if fvg.type == "bullish":
        if tp_raw <= entry_price:                   # degenerate case only
            tp_raw = entry_price + _min_tp_dist
    else:
        if tp_raw >= entry_price:                   # degenerate case only
            tp_raw = entry_price - _min_tp_dist

    # ── v4 forensic rework: SL mode selection ──────────────────────────────
    # "structural" = old logic (swing-level SL, 50% too wide per forensic)
    # "atr"        = ATR-capped SL (tighter, evidence-based)
    sl_mode = str(CONFIG.get("SL_MODE", "structural"))

    if sl_mode == "atr":
        sl_atr_mult = float(CONFIG.get("SL_ATR_MULT", 2.0))
        sl_atr_max_pct = float(CONFIG.get("SL_ATR_MAX_PCT", 2.5))
        atr_sl_distance = atr_abs * sl_atr_mult
        max_sl_distance = entry_price * sl_atr_max_pct / 100.0
        atr_sl_distance = min(atr_sl_distance, max_sl_distance)
        # Ensure minimum stop distance
        atr_sl_distance = max(atr_sl_distance, min_stop_distance)
        if fvg.type == "bullish":
            sl_raw = entry_price - atr_sl_distance
        else:
            sl_raw = entry_price + atr_sl_distance

    # ── v4 forensic rework: exit mode selection ──────────────────────────
    # "structural" = old logic (TP at nearest resistance/support)
    # "fixed_R"    = fixed multiple of risk (forensic optimum: 3R → +11.96R)
    exit_mode = str(CONFIG.get("EXIT_MODE", "structural"))

    if exit_mode == "fixed_R":
        fixed_tp_r = float(CONFIG.get("FIXED_TP_R", 3.0))
        _risk = abs(entry_price - sl_raw)
        if fvg.type == "bullish":
            tp_raw = entry_price + _risk * fixed_tp_r
        else:
            tp_raw = entry_price - _risk * fixed_tp_r

    sl = _round_price(sl_raw, market)
    tp = _round_price(tp_raw, market)
    risk   = abs(entry_price - sl)
    reward = abs(tp - entry_price)
    rr = reward / risk if risk > 0 else 0.0

    # v4: override MIN_RR if configured
    min_rr_override = float(CONFIG.get("MIN_RR_OVERRIDE", 0))
    effective_min_rr = min_rr_override if min_rr_override > 0 else float(CONFIG.get("MIN_TP_RR", 1.0))

    return {
        "entry_price":       float(entry_price),
        "sl_price":          float(sl),
        "tp_price":          float(tp),
        "risk_reward":       float(rr),
        "swing_low":         float(sw_low),
        "swing_high":        float(sw_high),
        "structure_level":   float(nearest_level),
        "target_level":      float(target_level),
        "vp_poc":            float(poc_level),
        "atr_pct":           float(atr_pct),
        "structure_touches": float(structure_touches),
        "symbol_profile":    str(symbol_profile.get("name", "normal")),
        "profile_gap_mult":  float(symbol_profile.get("gap_mult", 1.0)),
        "profile_tp_rr_cap": float(symbol_profile.get("tp_rr_cap", BACKTEST_CONFIG.get("MAX_RR", 2.5))),
        "sl_mode":           sl_mode,
        "exit_mode":         exit_mode,
        "min_rr":            float(effective_min_rr),
    }


# ─── Portfolio correlation helper ─────────────────────────────────────────────

def _df_correlation(df_a: pd.DataFrame, df_b: pd.DataFrame) -> float:
    lookback = max(40, int(CONFIG.get("CORRELATION_LOOKBACK_BARS", 96)))
    if df_a is None or df_b is None or df_a.empty or df_b.empty:
        return 0.0
    ra = df_a["close"].astype(float).pct_change().dropna().tail(lookback)
    rb = df_b["close"].astype(float).pct_change().dropna().tail(lookback)
    n = min(len(ra), len(rb))
    if n < 20:
        return 0.0
    ra = ra.iloc[-n:].to_numpy()
    rb = rb.iloc[-n:].to_numpy()
    corr = float(np.corrcoef(ra, rb)[0, 1])
    return 0.0 if np.isnan(corr) else corr


def _apply_correlation_filter(
    candidates: list[SignalCandidate],
    available_slots: int,
) -> list[SignalCandidate]:
    if available_slots <= 0:
        return []
    if not bool(CONFIG.get("PORTFOLIO_CORRELATION_FILTER", True)):
        return candidates[:available_slots]
    max_corr = float(CONFIG.get("PORTFOLIO_MAX_CORR", 0.82))
    selected: list[SignalCandidate] = []
    for candidate in candidates:
        if len(selected) >= available_slots:
            break
        blocked = False
        candidate_side = "LONG" if candidate.fvg.type == "bullish" else "SHORT"
        for peer in selected:
            peer_side = "LONG" if peer.fvg.type == "bullish" else "SHORT"
            if candidate_side != peer_side:
                continue
            if _df_correlation(candidate.df_5m, peer.df_5m) >= max_corr:
                blocked = True
                break
        if not blocked:
            selected.append(candidate)
    if len(selected) < available_slots:
        used = {id(item) for item in selected}
        for candidate in candidates:
            if len(selected) >= available_slots:
                break
            if id(candidate) not in used:
                selected.append(candidate)
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
#  FVGAgent
# ═══════════════════════════════════════════════════════════════════════════════

class FVGAgent:
    """
    FVG trading agent for the multi-agent system.

    Uses the same strategy as fvg_bot.py v2.6 but integrates with the shared
    StateManager, RiskManager, BybitClient and TelegramNotifier infrastructure.
    SIMULATION_MODE is always True — orders are logged, never placed.
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
        self.agent_id   = agent_id
        self.state      = state_manager
        self.client     = bybit_client
        self.risk       = risk_manager
        self.tg         = telegram
        self.oi_filter  = oi_filter

        # Deduplication — same logic as fvg_bot.py
        self.last_fvg_trigger: dict[tuple, int]        = {}
        self.last_pre_alert_trigger: dict[tuple, int]  = {}

        # Trade execution lock — prevents race condition when two scan cycles
        # overlap and both pass is_symbol_locked() before either registers
        self._trade_lock: asyncio.Lock = asyncio.Lock()

        # Daily tracking
        self.daily_trades     = 0
        self.daily_pnl        = 0.0
        self.daily_wins       = 0
        self.daily_losses     = 0
        self.last_daily_reset = ""
        self.starting_equity  = 0.0
        self.current_equity   = 0.0

        # Symbol list
        self._symbols: list[str] = []
        self._last_symbols_refresh: float = 0.0

        # Markets cache for price/qty rounding
        self._markets_cache: dict = {}

    # ── Heartbeat ─────────────────────────────────────────────────────────

    async def heartbeat(self) -> None:
        await self.state.record_heartbeat(self.agent_id)

    # ── Daily reset ───────────────────────────────────────────────────────

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
            log.info("[FVG] Daily stats reset for %s", today)

    # ── Symbol management ─────────────────────────────────────────────────

    async def _refresh_symbols_if_needed(self) -> None:
        import time as _time
        now = _time.monotonic()
        refresh_interval = getattr(cfg, "SYMBOLS_REFRESH_HR", 4) * 3600
        if now - self._last_symbols_refresh < refresh_interval and self._symbols:
            return
        try:
            tickers = await self.client._exchange.fetch_tickers()
            excluded = set(cfg.FVG_EXCLUDED_SYMBOLS)
            min_vol  = float(cfg.MIN_24H_VOLUME_USDT)
            usdt_perp = [
                (sym, t)
                for sym, t in tickers.items()
                if sym.endswith(":USDT")
                and sym not in excluded
                and float(t.get("quoteVolume") or 0) >= min_vol
            ]
            usdt_perp.sort(key=lambda x: float(x[1].get("quoteVolume") or 0), reverse=True)
            self._symbols = [sym for sym, _ in usdt_perp[: cfg.TOP_N_COINS]]
            self._last_symbols_refresh = now
            self._markets_cache = dict(self.client._exchange.markets or {})
            log.info("[FVG] %d symbols loaded.", len(self._symbols))
        except Exception as exc:
            log.error("[FVG] Symbol refresh failed: %s", exc)

    # ── Open-position monitor ─────────────────────────────────────────────

    async def _check_open_positions(self) -> None:
        """Check SL/TP for each tracked position against latest ticker price.

        v4 forensic: supports fixed R exit mode — TP is calculated from entry + risk * R.
        """
        positions = await self.state.get_agent_positions(self.agent_id)

        exit_mode = str(cfg.FVG_AGENT_CONFIG.get("EXIT_MODE", "structural"))
        fixed_tp_r = float(cfg.FVG_AGENT_CONFIG.get("FIXED_TP_R", 3.0))

        for symbol, pos in list(positions.items()):
            try:
                ticker = await self.client.fetch_ticker(symbol)
                price = float(ticker.get("last") or 0)
                if price <= 0:
                    continue
                side  = str(pos.get("side", "LONG"))
                entry = float(pos.get("entry", 0))
                sl    = float(pos.get("sl", 0))
                tp    = float(pos.get("tp", 0))

                # v4: recalculate TP based on fixed R if exit mode changed
                if exit_mode == "fixed_R" and entry > 0 and sl > 0:
                    risk = abs(entry - sl)
                    if risk > 0:
                        if side == "LONG":
                            tp = entry + risk * fixed_tp_r
                        else:
                            tp = entry - risk * fixed_tp_r

                hit  = False
                reason = ""
                if side == "LONG":
                    if sl > 0 and price <= sl:
                        hit = True
                        reason = "SL"
                    elif tp > 0 and price >= tp:
                        hit = True
                        reason = f"TP_{fixed_tp_r:.0f}R" if exit_mode == "fixed_R" else "TP"
                else:
                    if sl > 0 and price >= sl:
                        hit = True
                        reason = "SL"
                    elif tp > 0 and price <= tp:
                        hit = True
                        reason = f"TP_{fixed_tp_r:.0f}R" if exit_mode == "fixed_R" else "TP"

                if hit:
                    await self._close_position(symbol, pos, price, reason)
            except Exception as exc:
                log.warning("[FVG] Position check error %s: %s", symbol, exc)

    async def _close_position(
        self, symbol: str, pos: dict, exit_price: float, reason: str
    ) -> None:
        """Close a simulated position and record PnL."""
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

        self.daily_pnl  += pnl_usd
        self.daily_trades += 1
        if pnl_usd >= 0:
            self.daily_wins += 1
        else:
            self.daily_losses += 1

        await self.state.record_trade(self.agent_id, {
            "symbol":     symbol,
            "side":       side,
            "entry":      entry,
            "exit":       exit_price,
            "pnl_usd":    pnl_usd,
            "pnl_pct":    pnl_pct,
            "reason":     reason,
            "date":       datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "closed_at":  datetime.now(timezone.utc).isoformat(),
        })

        # Update equity
        current = await self.state.get_agent_equity(self.agent_id)
        await self.state.set_agent_equity(self.agent_id, max(0.0, current + pnl_usd))

        short_sym = symbol.split("/")[0]
        log.info(
            "[FVG] CLOSED %s %s entry=%.6g exit=%.6g pnl=$%.4f reason=%s",
            short_sym, side, entry, exit_price, pnl_usd, reason,
        )
        await self.tg.send_position_closed(
            self.agent_id, symbol, side, entry, exit_price, pnl_usd, pnl_pct, reason
        )

    # ── Scan one symbol ───────────────────────────────────────────────────

    async def scan_symbol(self, symbol: str) -> Optional[SignalCandidate]:
        """
        Check one symbol for an FVG signal.
        Mirrors fvg_bot.py scan_symbol() exactly including pre-alerts.
        """
        short_name = symbol.replace("/USDT:USDT", "USDT")

        # Gates
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return None
        if not await self.risk.check_position_count(self.agent_id, self.state, cfg.FVG_MAX_POSITIONS):
            return None
        if not await self.risk.check_daily_loss_limit(
            self.agent_id, self.state, cfg.FVG_MAX_DAILY_LOSS_PCT
        ):
            return None
        if await self.state.is_symbol_locked(symbol):
            return None

        # Fetch candles
        try:
            raw_entry = await self.client.fetch_ohlcv(
                symbol, cfg.FVG_ENTRY_TIMEFRAME, cfg.OHLCV_LIMIT_ENTRY
            )
            raw_htf = await self.client.fetch_ohlcv(
                symbol, cfg.FVG_HTF_TIMEFRAME, cfg.OHLCV_LIMIT_HTF
            )
        except Exception as exc:
            log.debug("[FVG] OHLCV fetch failed %s: %s", short_name, exc)
            return None

        if not raw_entry or not raw_htf:
            return None
        if len(raw_entry) < 20 or len(raw_htf) < cfg.FVG_EMA_PERIOD + 5:
            return None

        df_entry = pd.DataFrame(
            raw_entry, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_1h = pd.DataFrame(
            raw_htf, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df_entry["timestamp"] = df_entry["timestamp"].astype(int)
        df_1h["timestamp"]    = df_1h["timestamp"].astype(int)

        symbol_profile = build_symbol_profile(df_entry)

        # HTF EMA-200 bias
        df_1h["ema200"] = compute_ema(df_1h["close"], cfg.FVG_EMA_PERIOD)
        last_1h_close = float(df_1h["close"].iloc[-1])
        last_ema200   = float(df_1h["ema200"].iloc[-1])
        if pd.isna(last_ema200) or last_ema200 == 0:
            return None

        htf_bullish = last_1h_close > last_ema200
        htf_bearish = last_1h_close < last_ema200

        # Variant F: EMA50 on entry TF (15m) — dual trend filter
        # LONG only when price is above BOTH EMA200 (1h) AND EMA50 (15m)
        # SHORT only when price is below BOTH
        df_entry["ema50"] = compute_ema(df_entry["close"], cfg.FVG_EMA50_PERIOD)
        ema50_current = float(df_entry["ema50"].iloc[-1])
        if pd.isna(ema50_current) or ema50_current == 0:
            ema50_current = float(df_entry["close"].iloc[-1])  # fallback: no filter applied

        # Detect FVGs
        all_fvgs = detect_fvgs_v2(
            df_entry,
            lookback=cfg.FVG_LOOKBACK,
            percentile_threshold=cfg.FVG_PERCENTILE_THRESHOLD,
            percentile_lookback=cfg.FVG_PERCENTILE_LOOKBACK,
        )
        active_fvgs = _select_active_fvgs(all_fvgs)

        # Live price
        last_closed_price = float(df_entry["close"].iloc[-2])
        live_price = last_closed_price
        try:
            ticker = await self.client.fetch_ticker(symbol)
            live_price = float(ticker.get("last") or last_closed_price)
        except Exception:
            pass

        live_candle = df_entry.iloc[-1]
        live_candle_low  = float(live_candle["low"])
        live_candle_high = float(live_candle["high"])

        # Variant F dual filter: EMA200 (1h HTF) AND EMA50 (15m entry TF)
        # LONG: price above EMA50 on entry TF AND above EMA200 on 1h
        # SHORT: price below EMA50 on entry TF AND below EMA200 on 1h
        entry_bullish = htf_bullish and (live_price > ema50_current)
        entry_bearish = htf_bearish and (live_price < ema50_current)

        eligible_fvgs: list[FVG] = [
            fvg for fvg in active_fvgs
            if (fvg.type == "bullish" and entry_bullish)
            or (fvg.type == "bearish" and entry_bearish)
        ]
        eligible_fvgs.sort(
            key=lambda item: (item.quality_score, item.size_fvg, item.timestamp),
            reverse=True,
        )

        market = self._markets_cache.get(symbol, {})
        ready_candidates: list[SignalCandidate] = []

        for fvg in eligible_fvgs:
            effective_gap_floor = (
                float(CONFIG["MIN_GAP_PERCENT"])
                * float(symbol_profile.get("gap_mult", 1.0))
            )
            effective_quality_floor = (
                float(CONFIG.get("FVG_QUALITY_MIN_SCORE", 0.0))
                + float(symbol_profile.get("quality_bonus", 0.0))
            )
            effective_composite_floor = (
                float(CONFIG.get("SIGNAL_MIN_COMPOSITE_SCORE", 0.0))
                + float(symbol_profile.get("composite_bonus", 0.0))
            )
            if float(fvg.gap_pct) < effective_gap_floor:
                continue
            if float(fvg.quality_score) < effective_quality_floor:
                continue

            gap_size        = max(0.0, float(fvg.high - fvg.low))
            entry_band_low  = float(fvg.low + gap_size * CONFIG["MID_ZONE_LOW"])
            entry_band_high = float(fvg.low + gap_size * CONFIG["MID_ZONE_HIGH"])
            inside_gap      = fvg.low <= live_price <= fvg.high
            mid_zone_touched = live_candle_low <= entry_band_high and live_candle_high >= entry_band_low
            in_zone, position_pct = price_in_mid_zone(live_price, fvg)
            distance_to_mid       = distance_to_mid_zone_pct(live_price, fvg)
            reaction_ok, reaction_score = evaluate_entry_reaction(df_entry.tail(6), fvg)
            fvg.reaction_score = float(reaction_score)
            composite_score    = composite_signal_score(fvg, reaction_score)
            trigger_key = (symbol, fvg.type)

            if composite_score < effective_composite_floor:
                continue
            # Variant F: upper bound filter — skip if score > 80 (fake-outs, reaction_score=33)
            effective_composite_ceiling = float(
                CONFIG.get("SIGNAL_MAX_COMPOSITE_SCORE", 100.0)
            )
            if composite_score > effective_composite_ceiling:
                log.debug(
                    "[FVG] SKIP %s %s score=%.1f > ceiling %.1f",
                    short_name, fvg.type, composite_score, effective_composite_ceiling,
                )
                continue

            # Pre-alert
            if inside_gap and not in_zone and distance_to_mid <= CONFIG["PRE_ALERT_DISTANCE_PCT"]:
                plan = _build_trade_plan_v2(fvg, df_entry, market, symbol_profile=symbol_profile)
                if plan.get("risk_reward", 0.0) >= 1.0:
                    if self.last_pre_alert_trigger.get(trigger_key) != fvg.timestamp:
                        signal_side = "LONG" if fvg.type == "bullish" else "SHORT"
                        short_sym   = symbol.split("/")[0]
                        await self.tg._send(
                            f"👀 <b>Potential {signal_side} on {short_sym}</b>\n"
                            f"Gap: <b>{fvg.gap_pct:.2f}%</b> | Score: <b>{composite_score:.1f}</b>\n"
                            f"Entry zone: <b>40-60%</b> (price is <b>{distance_to_mid:.2f}%</b> away)\n"
                            f"SL: <code>{plan['sl_price']:.5f}</code> | "
                            f"TP: <code>{plan['tp_price']:.5f}</code> | "
                            f"RR: <b>{plan['risk_reward']:.2f}</b>\n"
                            f"Waiting for mid-zone entry"
                        )
                        self.last_pre_alert_trigger[trigger_key] = fvg.timestamp
                        log.info(
                            "[FVG] PRE-ALERT %s [%s] gap=%.3f%% score=%.1f dist=%.3f%%",
                            short_name, fvg.type.upper(), fvg.gap_pct, composite_score, distance_to_mid,
                        )
                        # Register in shared state for /pending TG command
                        try:
                            await self.state.register_pending_signal(
                                self.agent_id, symbol, signal_side,
                                float(live_price), plan["sl_price"], plan["tp_price"],
                                composite_score, fvg.gap_pct, distance_to_mid,
                            )
                        except Exception:
                            pass

            if not inside_gap:
                continue
            if not (in_zone or mid_zone_touched):
                continue
            if not in_zone:
                continue
            if not reaction_ok:
                continue
            if self.last_fvg_trigger.get(trigger_key) == fvg.timestamp:
                log.debug("[FVG] DEDUP %s %s ts=%d", short_name, fvg.type, fvg.timestamp)
                continue

            ready_candidates.append(SignalCandidate(
                symbol=symbol,
                fvg=fvg,
                entry_price=float(live_price),
                df_5m=df_entry,
                position_pct=position_pct,
                score=composite_score,
                raw_score=float(fvg.size_fvg),
                quality_score=float(fvg.quality_score),
                reaction_score=float(reaction_score),
            ))

        if ready_candidates:
            ready_candidates.sort(key=lambda c: c.score, reverse=True)
            best       = ready_candidates[0]
            trade_plan = _build_trade_plan_v2(
                best.fvg, df_entry, market, symbol_profile=symbol_profile
            )
            min_rr = float(trade_plan.get("min_rr", 1.0))
            if trade_plan.get("risk_reward", 0.0) < min_rr:
                log.info(
                    "[FVG] RR %.2f < %.1f — skip %s %s",
                    trade_plan["risk_reward"], min_rr, short_name, best.fvg.type,
                )
                return None
            log.info(
                "[FVG] CANDIDATE %s [%s] gap=%.3f%% score=%.1f pos=%.1f%% entry=%.5f",
                short_name, best.fvg.type.upper(),
                best.fvg.gap_pct, best.score, best.position_pct, best.entry_price,
            )
            return best

        return None

    # ── Execute trade ─────────────────────────────────────────────────────

    async def _execute_trade(self, candidate: SignalCandidate) -> None:
        """Build plan, size position, place order (simulated), register state."""
        async with self._trade_lock:
            await self._execute_trade_locked(candidate)

    async def _execute_trade_locked(self, candidate: SignalCandidate) -> None:
        """Actual trade execution — called only under _trade_lock."""
        symbol     = candidate.symbol
        fvg        = candidate.fvg
        entry_price = candidate.entry_price
        df_5m      = candidate.df_5m
        short_name = symbol.replace("/USDT:USDT", "USDT")

        # Final lock check (re-checked under _trade_lock to prevent race condition)
        if await self.state.is_symbol_locked(symbol):
            log.debug("[FVG] RACE: %s already locked — skip", short_name)
            return
        if not await self.risk.check_position_count(
            self.agent_id, self.state, cfg.FVG_MAX_POSITIONS
        ):
            return

        market = self._markets_cache.get(symbol, {})
        symbol_profile = build_symbol_profile(df_5m)
        trade_plan = _build_trade_plan_v2(fvg, df_5m, market, symbol_profile=symbol_profile)
        sl = float(trade_plan["sl_price"])
        tp = float(trade_plan["tp_price"])
        min_rr = float(trade_plan.get("min_rr", 1.0))
        if trade_plan.get("risk_reward", 0.0) < min_rr:
            log.info(
                "[FVG] RR %.2f < %.1f — skip execution %s %s",
                trade_plan["risk_reward"], min_rr, short_name, fvg.type,
            )
            return

        # Fetch live balance from Bybit, apply FVG capital fraction (60% of total)
        try:
            raw_bal = await self.client.fetch_balance()
            total_bal = float(
                raw_bal.get("USDT", {}).get("total", 0)
                or raw_bal.get("total", {}).get("USDT", 0)
                or 0
            )
            equity = total_bal * cfg.FVG_CAPITAL_FRACTION   # 60% of account
        except Exception:
            equity = await self.state.get_agent_equity(self.agent_id)
        if equity <= 0:
            log.error("[FVG] Zero equity — cannot trade")
            return

        if not await self.risk.check_margin_usage(self.agent_id, equity, self.state):
            return

        qty, leverage = self.risk.calculate_position_size(
            balance=equity,
            entry_price=entry_price,
            sl_price=sl,
            symbol=symbol,
            market_info=market if market else None,
        )
        if qty <= 0:
            log.warning("[FVG] qty=0 for %s — skipping", short_name)
            return

        notional    = qty * entry_price
        margin_used = notional / max(1.0, float(leverage))
        signal_side = "LONG" if fvg.type == "bullish" else "SHORT"
        ccxt_side   = "buy" if fvg.type == "bullish" else "sell"

        # Mark dedup
        self.last_fvg_trigger[(symbol, fvg.type)] = fvg.timestamp

        log.info(
            "[FVG] SIGNAL %s %s entry=%.6g sl=%.6g tp=%.6g rr=%.2f score=%.1f "
            "qty=%.6g lev=%d margin=$%.4f",
            short_name, signal_side, entry_price, sl, tp,
            trade_plan["risk_reward"], candidate.score,
            qty, leverage, margin_used,
        )

        await self.tg.send_signal(
            self.agent_id, symbol, signal_side, entry_price, sl, tp,
            trade_plan["risk_reward"], candidate.score, qty, leverage,
        )

        # Place order (simulation-aware)
        try:
            await self.client.set_leverage(symbol, leverage)
        except Exception as exc:
            log.warning("[FVG] set_leverage failed %s: %s", short_name, exc)

        try:
            order = await self.client.place_market_order(symbol, ccxt_side, qty)
        except Exception as exc:
            err_str = str(exc)
            if "110007" in err_str or "not enough" in err_str.lower() or "InsufficientFunds" in err_str:
                log.warning("[FVG] Insufficient margin for %s qty=%.6f — skipping trade", short_name, qty)
            else:
                log.error("[FVG] Market order error %s: %s", short_name, exc)
            return
        if order is None:
            log.error("[FVG] Market order failed %s", short_name)
            await self.tg.send_error(self.agent_id, f"Market order failed {short_name}")
            return

        await self.state.register_position(
            self.agent_id, symbol, signal_side, entry_price, sl, tp, qty
        )

        try:
            await self.client.place_sl_tp(symbol, ccxt_side, qty, sl, tp)
        except Exception as exc:
            log.warning("[FVG] SL/TP placement failed %s: %s", short_name, exc)

        await self.tg.send_position_opened(
            self.agent_id, symbol, signal_side, entry_price, sl, tp, qty, leverage, margin_used
        )
        self.daily_trades += 1
        log.info("[FVG] Position opened: %s %s @ %.6g", short_name, signal_side, entry_price)

    # ── Main scan cycle ───────────────────────────────────────────────────

    async def _run_scan_cycle(self) -> None:
        """Scan all symbols for FVG signals (batch of 5, mirrors fvg_bot.py)."""
        status = await self.state.get_agent_status(self.agent_id)
        if status != "running":
            return

        positions = await self.state.get_agent_positions(self.agent_id)
        available_slots = cfg.FVG_MAX_POSITIONS - len(positions)

        all_candidates: list[SignalCandidate] = []
        batch_size = 5
        batch_sleep = float(getattr(cfg, "BATCH_SLEEP_SEC", 2.5))
        symbols = list(self._symbols)

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i: i + batch_size]
            results = await asyncio.gather(
                *[self.scan_symbol(s) for s in batch],
                return_exceptions=True,
            )
            for res in results:
                if isinstance(res, SignalCandidate):
                    all_candidates.append(res)
                elif isinstance(res, Exception):
                    log.error("[FVG] scan_symbol exception: %s", res)
            # Heartbeat during long scan so master_agent doesn't restart us
            await self.heartbeat()
            await asyncio.sleep(batch_sleep)

        log.info(
            "[FVG] Scan complete. %d symbols. Candidates: %d",
            len(symbols), len(all_candidates),
        )
        if not all_candidates:
            return

        all_candidates.sort(key=lambda c: c.score, reverse=True)

        if available_slots <= 0:
            log.info(
                "[FVG] All %d positions occupied — no new trades (candidates=%d)",
                cfg.FVG_MAX_POSITIONS, len(all_candidates),
            )
            return

        selected = _apply_correlation_filter(all_candidates, available_slots)
        log.info(
            "[FVG] Candidates: %d | Selected top-%d: %s",
            len(all_candidates),
            len(selected),
            [(c.symbol.replace("/USDT:USDT", ""), f"{c.score:.1f}") for c in selected],
        )

        for candidate in selected:
            positions = await self.state.get_agent_positions(self.agent_id)
            if candidate.symbol in positions:
                continue
            await self._execute_trade(candidate)

    # ── Main loop ─────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Main agent loop."""
        log.info("[FVG] Agent %s starting.", self.agent_id)
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
                log.info("[FVG] Agent %s cancelled.", self.agent_id)
                await self.state.set_agent_status(self.agent_id, "stopped")
                break
            except Exception as exc:
                log.error("[FVG] Loop error: %s", exc, exc_info=True)
                await self.tg.send_error(self.agent_id, str(exc))
                await asyncio.sleep(10)

    # Alias kept for master_agent compatibility
    async def run(self) -> None:
        await self.start()
