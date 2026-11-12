"""
multi_agent/filters/oi_delta_filter.py

Open-Interest delta filter.

Logic
-----
We compare current OI to OI *OI_COMPARE_PERIOD* intervals ago (default 3 × 5 min = 15 min).

LONG trade:
    OI rising + price falling  → new shorts entering → BLOCK

SHORT trade:
    OI rising + price rising   → new longs entering  → BLOCK

All other combinations → ALLOW.

The filter is *optional* — if OI data cannot be fetched or is insufficient,
``check()`` returns (True, "insufficient_data") so the trade is never blocked
due to a data issue.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional, TYPE_CHECKING

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

if TYPE_CHECKING:
    from shared.bybit_client import BybitClient

logger = logging.getLogger(__name__)


class OIDeltaFilter:
    """
    Open-Interest delta filter for directional trade validation.

    Parameters
    ----------
    bybit_client : a ``BybitClient`` instance (already initialised)
    """

    def __init__(self, bybit_client: "BybitClient") -> None:
        self._client = bybit_client
        # {symbol: (fetched_at_epoch, [oi_float, ...])}
        self._cache: dict[str, tuple[float, list[float]]] = {}
        self._cache_ttl: int = cfg.OI_CACHE_TTL_SEC

    # ── OI data fetching ───────────────────────────────────────────────────

    async def get_oi_data(self, symbol: str) -> list[float]:
        """
        Return a list of open-interest values (most recent last).

        Data is cached for *OI_CACHE_TTL_SEC* (default 300 s = 5 min).

        Fetches from Bybit /v5/market/open-interest using intervalTime='5min',
        limit=OI_LOOKBACK_PERIODS.

        Returns an empty list on any failure (the caller treats this as
        insufficient data and allows the trade).
        """
        now = time.monotonic()
        if symbol in self._cache:
            fetched_at, values = self._cache[symbol]
            if now - fetched_at < self._cache_ttl:
                return values

        try:
            raw = await self._client.fetch_open_interest(
                symbol,
                interval=cfg.OI_INTERVAL,
                limit=cfg.OI_LOOKBACK_PERIODS,
            )
        except Exception as exc:
            logger.warning("OI fetch failed for %s: %s – allowing trade.", symbol, exc)
            return []

        if not raw:
            return []

        values: list[float] = []
        for entry in raw:
            # CCXT or Bybit v5 dict may use different keys
            oi_val = (
                entry.get("openInterestValue")
                or entry.get("openInterest")
                or entry.get("oi")
                or 0.0
            )
            try:
                values.append(float(oi_val))
            except (TypeError, ValueError):
                pass

        # Sort ascending by time (assume API returns newest first on Bybit)
        # We reverse so index 0 = oldest, -1 = newest
        if values:
            values = list(reversed(values))

        self._cache[symbol] = (now, values)
        logger.debug("OI data cached for %s: %d points", symbol, len(values))
        return values

    # ── Price trend helper ─────────────────────────────────────────────────

    async def _get_price_change(self, symbol: str, live_price: float) -> Optional[float]:
        """
        Estimate price change over the last OI_COMPARE_PERIOD intervals.

        Uses OHLCV (entry timeframe) as a proxy.  Returns positive if price
        rose, negative if it fell, None on failure.
        """
        try:
            candles = await self._client.fetch_ohlcv(
                symbol,
                cfg.FVG_ENTRY_TIMEFRAME,
                limit=cfg.OI_COMPARE_PERIOD + 2,
            )
            if len(candles) < cfg.OI_COMPARE_PERIOD + 1:
                return None
            # candles[-1] is the most recent; index [-(compare+1)] is compare periods ago
            past_close = candles[-(cfg.OI_COMPARE_PERIOD + 1)][4]  # close
            return live_price - past_close
        except Exception as exc:
            logger.debug("Price-change fetch failed for %s: %s", symbol, exc)
            return None

    # ── Main filter ────────────────────────────────────────────────────────

    async def check(
        self,
        symbol: str,
        direction: str,
        live_price: float,
    ) -> tuple[bool, str]:
        """
        Evaluate whether a trade in *direction* is allowed given current OI dynamics.

        Parameters
        ----------
        symbol    : e.g. "BTC/USDT:USDT"
        direction : "LONG" or "SHORT"
        live_price: current market price (used for price-trend comparison)

        Returns
        -------
        (allowed: bool, reason: str)
            allowed = True  → trade is allowed
            allowed = False → trade is blocked
        """
        try:
            oi_data = await self.get_oi_data(symbol)
        except Exception as exc:
            logger.warning("OI check error for %s: %s – allowing trade.", symbol, exc)
            return True, "oi_fetch_error"

        if len(oi_data) < 4:
            return True, "insufficient_data"

        oi_current = oi_data[-1]
        oi_prev = oi_data[-(cfg.OI_COMPARE_PERIOD + 1)]

        if oi_prev <= 0:
            return True, "oi_prev_zero"

        oi_delta_pct = (oi_current - oi_prev) / oi_prev * 100.0

        # Determine price direction
        price_change = await self._get_price_change(symbol, live_price)
        if price_change is None:
            # Cannot determine price direction; be permissive
            return True, f"price_trend_unknown_oi_delta={oi_delta_pct:.2f}%"

        price_falling = price_change < 0
        price_rising = price_change > 0

        if direction == "LONG":
            if oi_delta_pct > cfg.OI_DELTA_THRESHOLD_PCT and price_falling:
                # OI rising + price falling = new shorts adding = bad for longs
                reason = f"oi_rising_{oi_delta_pct:.2f}%_price_falling"
                logger.info("OI filter BLOCK %s LONG: %s", symbol, reason)
                return False, reason
            return True, f"oi_ok_{oi_delta_pct:.2f}%"

        if direction == "SHORT":
            if oi_delta_pct > cfg.OI_DELTA_THRESHOLD_PCT and price_rising:
                # OI rising + price rising = new longs adding = bad for shorts
                reason = f"oi_rising_{oi_delta_pct:.2f}%_price_rising"
                logger.info("OI filter BLOCK %s SHORT: %s", symbol, reason)
                return False, reason
            return True, f"oi_ok_{oi_delta_pct:.2f}%"

        # Unknown direction – allow
        return True, "unknown_direction"


if __name__ == "__main__":
    import asyncio
    import sys
    from pathlib import Path

    # Add project root so we can import shared modules
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from multi_agent.shared.bybit_client import BybitClient

    async def _demo() -> None:
        client = await BybitClient.get_instance()
        ticker = await client.fetch_ticker("BTC/USDT:USDT")
        price = ticker["last"]
        oi_filter = OIDeltaFilter(client)
        allowed, reason = await oi_filter.check("BTC/USDT:USDT", "LONG", price)
        print(f"LONG allowed={allowed}  reason={reason}")
        allowed, reason = await oi_filter.check("BTC/USDT:USDT", "SHORT", price)
        print(f"SHORT allowed={allowed}  reason={reason}")
        await client.close()

    asyncio.run(_demo())
