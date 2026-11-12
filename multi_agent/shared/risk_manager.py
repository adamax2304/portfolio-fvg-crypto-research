"""
multi_agent/shared/risk_manager.py

Position-sizing and risk-gate logic shared by all agents.

Key formula
-----------
    risk_usd = balance * RISK_PER_TRADE
    distance = abs(entry_price - sl_price)
    raw_qty  = risk_usd / distance

Then a series of hard caps are applied:
    - MAX_MARGIN_PER_TRADE_USD        (absolute USD margin cap)
    - MAX_MARGIN_PER_TRADE_BALANCE_FRACTION (fraction of balance)
    - MAX_LEVERAGE                    (leverage cap)
    - Market min/max qty and precision from CCXT market_info

Returns (0.0, 1) if the trade cannot meet minimum requirements.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

if TYPE_CHECKING:
    from shared.state_manager import StateManager

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Stateless risk and position-sizing calculator.

    All methods are synchronous (no I/O). Pass a ``StateManager`` instance
    where required for drawdown/equity checks.
    """

    # ── Position sizing ────────────────────────────────────────────────────

    def calculate_position_size(
        self,
        balance: float,
        entry_price: float,
        sl_price: float,
        symbol: str,
        market_info: Optional[dict] = None,
    ) -> tuple[float, int]:
        """
        Calculate position quantity and leverage for a trade.

        Parameters
        ----------
        balance     : agent's available equity in USDT
        entry_price : planned entry price
        sl_price    : stop-loss price
        symbol      : trading symbol (used only for logging)
        market_info : CCXT market dict (for precision / min-qty checks);
                      if None, precision rules are skipped

        Returns
        -------
        (qty, leverage)
            qty      – contract quantity in base currency (0.0 if invalid)
            leverage – recommended leverage (1 if qty == 0)
        """
        if balance <= 0 or entry_price <= 0:
            logger.debug("Invalid balance or entry for %s", symbol)
            return 0.0, 1

        distance = abs(entry_price - sl_price)
        if distance <= 0:
            logger.warning("SL price equals entry for %s – skipping.", symbol)
            return 0.0, 1

        risk_usd = balance * cfg.RISK_PER_TRADE
        if risk_usd < cfg.MIN_RISK_USD:
            logger.debug(
                "Risk $%.4f below MIN_RISK_USD $%.4f for %s.", risk_usd, cfg.MIN_RISK_USD, symbol
            )
            return 0.0, 1

        raw_qty = risk_usd / distance

        # ── Cap: margin per trade ─────────────────────────────────────────
        # margin = (qty * entry_price) / leverage
        # We want: MIN_MARGIN <= margin <= MAX_MARGIN

        raw_margin = raw_qty * entry_price  # margin at leverage=1

        # Upper cap: min of absolute USD cap and fraction of balance
        max_margin_fraction = balance * cfg.MAX_MARGIN_PER_TRADE_BALANCE_FRACTION
        max_margin = min(cfg.MAX_MARGIN_PER_TRADE_USD, max_margin_fraction)

        # Lower floor: ensure margin >= MIN_MARGIN so orders actually execute
        min_margin = getattr(cfg, "MIN_MARGIN_PER_TRADE_USD", 0.0)

        # Determine leverage needed so that margin stays within cap
        if max_margin > 0 and raw_margin > max_margin:
            required_lev = math.ceil(raw_margin / max_margin)
        else:
            required_lev = 1

        leverage = min(required_lev, cfg.MAX_LEVERAGE)

        # Adjust qty so that margin = qty * entry / leverage <= max_margin
        effective_margin = raw_margin / leverage
        if effective_margin > max_margin and max_margin > 0:
            raw_qty = max_margin * leverage / entry_price

        # Enforce minimum margin: if margin too small, REJECT the trade.
        # Do NOT bump qty — that would risk more than allocated and cause
        # infinite retries (retCode:110007) when balance is insufficient.
        actual_margin = raw_qty * entry_price / leverage
        if min_margin > 0 and actual_margin < min_margin:
            logger.warning(
                "Margin $%.4f < MIN $%.2f for %s — skipping trade.",
                actual_margin, min_margin, symbol,
            )
            return 0.0, 1

        qty = raw_qty

        # ── Market precision / min qty ────────────────────────────────────
        if market_info:
            precision = market_info.get("precision", {})
            amount_precision = precision.get("amount", None)
            if amount_precision is not None:
                # amount_precision is the tick size (e.g. 0.001 → 3 dp)
                tick = float(amount_precision)
                if tick > 0:
                    qty = math.floor(qty / tick) * tick

            limits = market_info.get("limits", {})
            min_qty = limits.get("amount", {}).get("min", 0.0)
            if min_qty and qty < float(min_qty):
                logger.debug("qty %.6f < min_qty %.6f for %s – skipping.", qty, min_qty, symbol)
                return 0.0, 1

        if qty <= 0:
            return 0.0, 1

        logger.debug(
            "Position size: %s qty=%.6f lev=%d margin≈$%.4f risk=$%.4f",
            symbol,
            qty,
            leverage,
            qty * entry_price / leverage,
            risk_usd,
        )
        return qty, leverage

    # ── Daily loss limit ───────────────────────────────────────────────────

    async def check_daily_loss_limit(
        self,
        agent_id: str,
        state_manager: "StateManager",
        max_daily_loss_pct: float,
    ) -> bool:
        """
        Return True if the agent may open new trades today.

        Checks whether today's realised PnL is worse than *max_daily_loss_pct*
        (expressed as a percentage of starting equity).
        """
        trades = await state_manager.get_trades(agent_id)
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        today_pnl = sum(
            t.get("pnl_usd", 0.0)
            for t in trades
            if t.get("date", "") == today
        )
        start_equity = await state_manager.get_agent_equity(agent_id)
        if start_equity <= 0:
            return True
        loss_pct = -today_pnl / start_equity * 100.0
        if loss_pct >= max_daily_loss_pct:
            logger.warning(
                "Agent %s daily loss limit hit: %.2f%% (max %.2f%%)",
                agent_id,
                loss_pct,
                max_daily_loss_pct,
            )
            return False
        return True

    # ── Position count ─────────────────────────────────────────────────────

    async def check_position_count(
        self,
        agent_id: str,
        state_manager: "StateManager",
        max_positions: int,
    ) -> bool:
        """
        Return True if the agent has fewer than *max_positions* open trades.
        """
        positions = await state_manager.get_agent_positions(agent_id)
        if len(positions) >= max_positions:
            logger.debug(
                "Agent %s at max positions (%d/%d).", agent_id, len(positions), max_positions
            )
            return False
        return True

    # ── Margin usage ───────────────────────────────────────────────────────

    async def check_margin_usage(
        self,
        agent_id: str,
        balance: float,
        state_manager: "StateManager",
    ) -> bool:
        """
        Return True if total margin currently in use is below MAX_MARGIN_IN_USE_PCT.

        We approximate used margin as sum of (qty * entry_price / leverage) for
        all open positions. If leverage is not stored, we use 1.
        """
        positions = await state_manager.get_agent_positions(agent_id)
        used_margin = 0.0
        for pos in positions.values():
            qty = pos.get("qty", 0.0)
            entry = pos.get("entry", 0.0)
            lev = pos.get("leverage", 1)
            used_margin += (qty * entry) / max(lev, 1)

        if balance <= 0:
            return True
        usage_pct = used_margin / balance
        if usage_pct >= cfg.MAX_MARGIN_IN_USE_PCT:
            logger.debug(
                "Agent %s margin in use %.1f%% >= limit %.1f%%.",
                agent_id,
                usage_pct * 100,
                cfg.MAX_MARGIN_IN_USE_PCT * 100,
            )
            return False
        return True

    # ── Maximum risk amount ────────────────────────────────────────────────

    def get_max_risk_for_agent(self, agent_id: str, balance: float) -> float:
        """
        Return the maximum risk amount in USD for a single trade.

        This is simply balance * RISK_PER_TRADE, clamped to MIN_RISK_USD.
        """
        risk = balance * cfg.RISK_PER_TRADE
        return max(risk, 0.0)


if __name__ == "__main__":
    rm = RiskManager()
    qty, lev = rm.calculate_position_size(
        balance=100.0,
        entry_price=60000.0,
        sl_price=59400.0,
        symbol="BTC/USDT:USDT",
    )
    print(f"qty={qty:.6f}  leverage={lev}")
    print(f"margin≈${qty * 60000 / lev:.4f}")
