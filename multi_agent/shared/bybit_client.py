"""
multi_agent/shared/bybit_client.py

Shared async Bybit client using CCXT.
Implements the Singleton pattern so all agents share one exchange connection.
Includes:
  - In-memory LRU cache with TTL for OHLCV data (30 s default)
  - Exponential-backoff retry for every API call (max 5 attempts)
  - asyncio.Semaphore rate-limiting (max 10 concurrent requests)
  - SIMULATION_MODE flag (orders are logged but never sent)
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import OrderedDict
from typing import Any, Optional

import ccxt.async_support as ccxt

# Local imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

logger = logging.getLogger(__name__)

# ─── Simple TTL-LRU cache ────────────────────────────────────────────────────

class _TTLCache:
    """In-memory LRU cache with per-entry time-to-live."""

    def __init__(self, maxsize: int = 512, ttl: float = 30.0) -> None:
        self._maxsize = maxsize
        self._ttl = ttl
        self._store: OrderedDict[str, tuple[float, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Any]:
        if key not in self._store:
            return None
        ts, value = self._store[key]
        if time.monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        # Move to end (most-recently used)
        self._store.move_to_end(key)
        return value

    def set(self, key: str, value: Any) -> None:
        if key in self._store:
            self._store.move_to_end(key)
        self._store[key] = (time.monotonic(), value)
        while len(self._store) > self._maxsize:
            self._store.popitem(last=False)

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)


# ─── Bybit client (Singleton) ─────────────────────────────────────────────────

class BybitClient:
    """
    Async Bybit client wrapping CCXT.

    Usage
    -----
    client = await BybitClient.get_instance()
    ohlcv  = await client.fetch_ohlcv("BTC/USDT:USDT", "15m", 500)
    """

    _instance: Optional["BybitClient"] = None
    _lock: asyncio.Lock = asyncio.Lock()

    def __init__(self) -> None:
        self._exchange: Optional[ccxt.bybit] = None
        self._ohlcv_cache: _TTLCache = _TTLCache(maxsize=512, ttl=30.0)
        self._ticker_cache: _TTLCache = _TTLCache(maxsize=256, ttl=5.0)
        self._semaphore: asyncio.Semaphore = asyncio.Semaphore(5)   # max 5 concurrent API calls
        self._initialized: bool = False

    # ── Singleton ──────────────────────────────────────────────────────────

    @classmethod
    async def get_instance(cls) -> "BybitClient":
        """Return the shared singleton, initialising it on first call."""
        async with cls._lock:
            if cls._instance is None or not cls._instance._initialized:
                cls._instance = cls()
                try:
                    await cls._instance.init()
                except Exception:
                    cls._instance = None  # reset so next call retries
                    raise
            return cls._instance

    # ── Initialisation ─────────────────────────────────────────────────────

    async def init(self) -> None:
        """Initialise the CCXT exchange connection."""
        if self._initialized:
            return
        self._exchange = ccxt.bybit(
            {
                "apiKey": cfg.BYBIT_API_KEY,
                "secret": cfg.BYBIT_API_SECRET,
                "timeout": 15000,  # 15s request timeout — prevents infinite hang on network issues
                "options": {
                    "defaultType": "linear",
                    "adjustForTimeDifference": True,
                    "recvWindow": 60000,
                },
                "enableRateLimit": True,
            }
        )
        if cfg.BYBIT_TESTNET:
            self._exchange.set_sandbox_mode(True)
        try:
            # Sync clock before loading markets
            for attempt in range(3):
                try:
                    await self._exchange.load_time_difference()
                    break
                except Exception:
                    if attempt < 2:
                        await asyncio.sleep(2)
            # load_markets retry — network can be flaky at startup / after PC sleep
            for attempt in range(6):
                try:
                    await self._exchange.load_markets()
                    break
                except Exception as e:
                    if attempt < 5:
                        wait = 5 * (attempt + 1)  # 5s, 10s, 15s, 20s, 25s
                        logger.warning(
                            "load_markets failed (attempt %d/6): %s — retrying in %ds…",
                            attempt + 1, e, wait,
                        )
                        await asyncio.sleep(wait)
                    else:
                        raise
            self._initialized = True
            # Start background clock-sync task (resync every 2 min)
            asyncio.create_task(self._clock_sync_loop(120.0), name="bybit_clock_sync")
            logger.info(
                "BybitClient initialised (testnet=%s, simulation=%s)",
                cfg.BYBIT_TESTNET,
                cfg.SIMULATION_MODE,
            )
        except Exception as exc:
            logger.error("BybitClient init failed: %s", exc)
            raise

    async def close(self) -> None:
        """Close the underlying exchange session."""
        if self._exchange:
            await self._exchange.close()
            self._exchange = None
            self._initialized = False
            logger.info("BybitClient closed.")

    async def _clock_sync_loop(self, interval: float = 120.0) -> None:
        """Background task: resync exchange clock every *interval* seconds."""
        while True:
            await asyncio.sleep(interval)
            if self._exchange and self._initialized:
                try:
                    await self._exchange.load_time_difference()
                    logger.debug("BybitClient: clock resynced.")
                except Exception as exc:
                    logger.warning("BybitClient: clock resync failed: %s", exc)

    # ── Internal retry helper ──────────────────────────────────────────────

    async def _call_with_retry(self, coro_fn, *args, max_attempts: int = 5, **kwargs) -> Any:
        """
        Execute *coro_fn(*args, **kwargs)* with exponential-backoff retry.

        Parameters
        ----------
        coro_fn   : callable that returns an awaitable
        max_attempts : maximum number of attempts before re-raising
        """
        delay = 1.0
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with self._semaphore:
                    return await coro_fn(*args, **kwargs)
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as exc:
                last_exc = exc
                if attempt == max_attempts:
                    break
                wait = delay * (2 ** (attempt - 1))
                logger.warning(
                    "API call %s attempt %d/%d failed (%s). Retrying in %.1fs …",
                    getattr(coro_fn, "__name__", str(coro_fn)),
                    attempt,
                    max_attempts,
                    exc,
                    wait,
                )
                await asyncio.sleep(wait)
            except ccxt.RateLimitExceeded as exc:
                last_exc = exc
                # On rate limit, wait longer and count against max_attempts
                wait = 2.0 * (2 ** attempt)   # 4s, 8s, 16s, 32s …
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Sleeping %.1fs …",
                    attempt, max_attempts, wait,
                )
                await asyncio.sleep(wait)
                if attempt == max_attempts:
                    break
            except ccxt.BaseError as exc:
                exc_str = str(exc)
                # retCode 10002 = timestamp out of sync — resync and retry
                if "10002" in exc_str and attempt < max_attempts:
                    logger.warning(
                        "Timestamp sync error (attempt %d/%d) — resyncing clock …",
                        attempt, max_attempts,
                    )
                    try:
                        await self._exchange.load_time_difference()
                    except Exception:
                        pass
                    await asyncio.sleep(0.5)
                    continue
                logger.error("Non-retryable CCXT error: %s", exc)
                raise
        raise last_exc  # type: ignore[misc]

    # ── Public API methods ─────────────────────────────────────────────────

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 500,
    ) -> list[list]:
        """
        Return OHLCV candles for *symbol* on *timeframe*.

        Results are cached in memory for 30 seconds (LRU, max 512 entries).

        Returns
        -------
        list of [timestamp_ms, open, high, low, close, volume]
        """
        cache_key = f"ohlcv:{symbol}:{timeframe}:{limit}"
        cached = self._ohlcv_cache.get(cache_key)
        if cached is not None:
            return cached
        data = await self._call_with_retry(
            self._exchange.fetch_ohlcv, symbol, timeframe, limit=limit
        )
        self._ohlcv_cache.set(cache_key, data)
        return data

    async def fetch_ticker(self, symbol: str) -> dict:
        """
        Return the current ticker for *symbol*.

        Cached for 5 seconds.
        """
        cached = self._ticker_cache.get(symbol)
        if cached is not None:
            return cached
        ticker = await self._call_with_retry(self._exchange.fetch_ticker, symbol)
        self._ticker_cache.set(symbol, ticker)
        return ticker

    async def fetch_balance(self) -> dict:
        """
        Return account balance information.

        Returns the raw CCXT balance dict which includes 'total', 'free', 'used'
        keys per currency plus 'info' for the raw exchange response.
        """
        return await self._call_with_retry(
            self._exchange.fetch_balance, {"type": "linear"}
        )

    async def fetch_all_positions(self) -> list[dict]:
        """
        Return all open positions on the account (contracts != 0).
        """
        try:
            positions = await self._call_with_retry(
                self._exchange.fetch_positions, []
            )
            return [
                p for p in positions
                if float(p.get("contracts", 0) or p.get("info", {}).get("size", 0) or 0) != 0
            ]
        except Exception as exc:
            logger.warning("fetch_all_positions failed: %s", exc)
            return []

    async def get_position(self, symbol: str) -> Optional[dict]:
        """
        Return the open position for *symbol*, or None if flat.

        Only returns a position with non-zero contracts.
        """
        positions = await self._call_with_retry(
            self._exchange.fetch_positions, [symbol]
        )
        for pos in positions:
            if pos.get("contracts", 0) and float(pos["contracts"]) != 0:
                return pos
        return None

    async def set_leverage(self, symbol: str, leverage: int) -> dict:
        """Set isolated-margin leverage for *symbol*."""
        params = {"marginMode": "isolated"}
        return await self._call_with_retry(
            self._exchange.set_leverage, leverage, symbol, params
        )

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Optional[dict]:
        """
        Place a market order.

        When SIMULATION_MODE is True the order is logged but not submitted to
        the exchange.

        Parameters
        ----------
        symbol : e.g. "BTC/USDT:USDT"
        side   : "buy" or "sell"
        qty    : contract quantity (base currency units)
        """
        if cfg.SIMULATION_MODE:
            logger.info(
                "[SIM] MARKET ORDER | %s %s qty=%.6f",
                side.upper(),
                symbol,
                qty,
            )
            return {"id": "SIM", "symbol": symbol, "side": side, "amount": qty, "type": "market"}
        order = await self._call_with_retry(
            self._exchange.create_order, symbol, "market", side, qty
        )
        logger.info("Market order placed: %s", order)
        return order

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> Optional[dict]:
        """
        Place a limit order.

        When SIMULATION_MODE is True the order is logged but not submitted.
        """
        if cfg.SIMULATION_MODE:
            logger.info(
                "[SIM] LIMIT ORDER  | %s %s qty=%.6f @ %.8f",
                side.upper(),
                symbol,
                qty,
                price,
            )
            return {
                "id": "SIM",
                "symbol": symbol,
                "side": side,
                "amount": qty,
                "price": price,
                "type": "limit",
            }
        order = await self._call_with_retry(
            self._exchange.create_order, symbol, "limit", side, qty, price
        )
        logger.info("Limit order placed: %s", order)
        return order

    async def fetch_order(self, order_id: str, symbol: str) -> dict:
        """Fetch a single order's current status."""
        if cfg.SIMULATION_MODE:
            return {"id": order_id, "status": "closed", "average": 0.0, "filled": 0.0}
        result = await self._call_with_retry(
            self._exchange.fetch_order, order_id, symbol,
            params={"acknowledged": True},
        )
        return result or {}

    async def place_post_only_limit_order(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
    ) -> Optional[dict]:
        """Place a post-only limit order (maker fee guaranteed). Returns None if rejected."""
        if cfg.SIMULATION_MODE:
            logger.info("[SIM] POST-ONLY LIMIT | %s %s qty=%.6f @ %.8f", side.upper(), symbol, qty, price)
            return {"id": "SIM", "symbol": symbol, "side": side, "amount": qty, "price": price, "type": "limit"}
        try:
            order = await self._call_with_retry(
                self._exchange.create_order,
                symbol, "limit", side, qty, price,
                {"timeInForce": "PostOnly"},
            )
            logger.info("Post-only limit order placed: %s", order)
            return order
        except Exception as exc:
            logger.warning("Post-only limit rejected (%s %s @ %.8f): %s", side, symbol, price, exc)
            return None

    async def fetch_open_orders(self, symbol: str) -> list[dict]:
        """Return list of open/pending orders for symbol from Bybit."""
        if cfg.SIMULATION_MODE:
            return []
        try:
            orders = await self._call_with_retry(
                self._exchange.fetch_open_orders, symbol
            )
            return orders or []
        except Exception as exc:
            logger.debug("fetch_open_orders %s: %s", symbol, exc)
            return []

    async def cancel_order(self, symbol: str, order_id: str) -> dict:
        """Cancel an open order by ID. Returns silently if order already gone."""
        if cfg.SIMULATION_MODE:
            logger.info("[SIM] CANCEL ORDER | %s id=%s", symbol, order_id)
            return {"id": order_id, "symbol": symbol, "status": "canceled"}
        try:
            result = await self._call_with_retry(
                self._exchange.cancel_order, order_id, symbol
            )
            logger.info("Order cancelled: %s", result)
            return result or {}
        except Exception as exc:
            err = str(exc)
            # 110001 = order not exists or too late to cancel — already filled/cancelled
            if "110001" in err or "not exists" in err.lower() or "Order does not exist" in err:
                logger.debug("cancel_order %s: already gone (%s)", order_id, exc)
                return {"id": order_id, "status": "canceled"}
            raise

    async def place_sl_tp(
        self,
        symbol: str,
        side: str,
        qty: float,
        sl_price: float,
        tp_price: float,
    ) -> dict:
        """
        Place conditional stop-loss and take-profit orders via Bybit v5 private API.

        Parameters
        ----------
        side      : "buy" (for a long position, SL is a sell stop; TP is a sell limit)
                    "sell" (for a short position, vice-versa)
        qty       : position quantity to close
        sl_price  : stop-loss trigger price
        tp_price  : take-profit trigger price
        """
        opposite = "sell" if side == "buy" else "buy"
        sl_order: Optional[dict] = None
        tp_order: Optional[dict] = None

        if cfg.SIMULATION_MODE:
            logger.info(
                "[SIM] SL/TP | %s %s sl=%.8f tp=%.8f qty=%.6f",
                opposite.upper(),
                symbol,
                sl_price,
                tp_price,
                qty,
            )
            return {"sl": sl_price, "tp": tp_price, "simulated": True}

        # Use Bybit v5 /position/trading-stop to set SL/TP directly on the position
        # This is the correct method — creating separate conditional orders fails with v5 API
        ccxt_symbol = symbol.replace("/", "").replace(":USDT", "")  # "ICP/USDT:USDT" → "ICPUSDT"
        params = {
            "category": "linear",
            "symbol": ccxt_symbol,
            "tpslMode": "Full",
            "slTriggerBy": "LastPrice",
            "tpTriggerBy": "LastPrice",
        }
        if sl_price and sl_price > 0:
            params["stopLoss"] = str(sl_price)
        if tp_price and tp_price > 0:
            params["takeProfit"] = str(tp_price)

        try:
            result = await self._call_with_retry(
                self._exchange.privatePostV5PositionTradingStop,
                params,
            )
            logger.info("SL/TP set for %s: sl=%.6g tp=%.6g → %s", symbol, sl_price, tp_price, result)
            return {"sl": sl_price, "tp": tp_price, "result": result}
        except Exception as exc:
            logger.error("SL/TP set failed for %s: %s", symbol, exc)
            return {"sl": sl_price, "tp": tp_price, "error": str(exc)}

    async def fetch_open_interest(
        self,
        symbol: str,
        interval: str = "5min",
        limit: int = 6,
    ) -> list[dict]:
        """
        Fetch open-interest history from Bybit v5 /market/open-interest.

        Parameters
        ----------
        symbol   : e.g. "BTCUSDT"
        interval : Bybit interval string: 5min, 15min, 30min, 1h, 4h, 1d
        limit    : number of periods to return

        Returns
        -------
        List of dicts with keys: symbol, openInterest, timestamp
        """
        # Derive the raw Bybit symbol (strip slash and :USDT suffix)
        raw_symbol = symbol.replace("/", "").replace(":USDT", "")

        try:
            result = await self._call_with_retry(
                self._exchange.fetch_open_interest_history,
                symbol,
                interval,
                limit=limit,
            )
            return result
        except Exception:
            # Fall back to private v5 REST endpoint
            try:
                params = {
                    "category": "linear",
                    "symbol": raw_symbol,
                    "intervalTime": interval,
                    "limit": limit,
                }
                resp = await self._call_with_retry(
                    self._exchange.public_get_v5_market_open_interest,
                    params,
                )
                return resp.get("result", {}).get("list", [])
            except Exception as exc:
                logger.warning("fetch_open_interest failed for %s: %s", symbol, exc)
                return []


    async def fetch_closed_pnl(self, hours: int = 48) -> list[dict]:
        """
        Fetch closed P&L records from Bybit for the last N hours.
        Used at startup to recover missed trade notifications.
        Returns list of raw Bybit position/closed-pnl entries.
        """
        import time as _t
        start_ms = str(int((_t.time() - hours * 3600) * 1000))
        try:
            resp = await self._call_with_retry(
                self._exchange.privateGetV5PositionClosedPnl,
                {"category": "linear", "limit": 100, "startTime": start_ms},
            )
            return resp.get("result", {}).get("list", [])
        except Exception as exc:
            logger.warning("fetch_closed_pnl failed: %s", exc)
            return []

    async def fetch_funding_rate(self, symbol: str) -> Optional[float]:
        """
        Fetch the current funding rate for a symbol.

        Returns funding rate as a float (e.g. 0.0001 = 0.01%/8h),
        or None on error.
        """
        raw_symbol = symbol.replace("/", "").replace(":USDT", "")
        try:
            # ccxt standard method
            result = await self._call_with_retry(
                self._exchange.fetch_funding_rate,
                symbol,
            )
            rate = result.get("fundingRate")
            if rate is not None:
                return float(rate)
        except Exception:
            pass
        # Fallback: Bybit v5 REST
        try:
            params = {"category": "linear", "symbol": raw_symbol}
            resp = await self._call_with_retry(
                self._exchange.public_get_v5_market_tickers,
                params,
            )
            items = resp.get("result", {}).get("list", [])
            if items:
                rate = items[0].get("fundingRate")
                if rate is not None:
                    return float(rate)
        except Exception as exc:
            logger.debug("fetch_funding_rate failed for %s: %s", symbol, exc)
        return None


if __name__ == "__main__":
    import asyncio

    async def _smoke_test() -> None:
        client = await BybitClient.get_instance()
        ticker = await client.fetch_ticker("BTC/USDT:USDT")
        print("BTC last price:", ticker.get("last"))
        await client.close()

    asyncio.run(_smoke_test())
