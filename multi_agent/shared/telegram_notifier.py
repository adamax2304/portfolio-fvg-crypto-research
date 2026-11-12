"""
multi_agent/shared/telegram_notifier.py

Telegram notification layer for the multi-agent trading system.

Features
--------
- All sends are async with 3-attempt retry on network failure.
- Rate limiting: at most 1 message per second (asyncio.Semaphore + last-send
  timestamp enforcement).
- Messages use HTML parse_mode for clean formatting.
- Single ``aiohttp.ClientSession`` reused across calls.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Optional

import aiohttp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

logger = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org/bot{token}/sendMessage"


class MultiAgentTelegramNotifier:
    """
    Async Telegram notifier for the multi-agent system.

    All public methods are coroutines.  Create one instance and reuse it.

    Example
    -------
    notifier = MultiAgentTelegramNotifier()
    await notifier.send_signal("fvg", "BTC/USDT:USDT", "LONG", 60000, 59000, 62000, 2.0, 85, 0.01, 10)
    """

    def __init__(
        self,
        bot_token: str = cfg.TG_BOT_TOKEN,
        chat_id: str = cfg.TG_CHAT_ID,
    ) -> None:
        self._token = bot_token
        self._chat_id = chat_id
        self._url = _TELEGRAM_API.format(token=self._token)
        self._session: Optional[aiohttp.ClientSession] = None
        self._rate_lock = asyncio.Semaphore(1)
        self._last_sent_at: float = 0.0          # epoch seconds
        self._min_interval: float = 1.0          # max 1 msg / second

    # ── Session lifecycle ──────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(ssl=False)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def close(self) -> None:
        """Close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Core send ──────────────────────────────────────────────────────────

    async def _send(self, text: str, max_retries: int = 3) -> bool:
        """
        Send *text* to the configured Telegram chat with rate-limiting and retry.

        Returns True on success, False on permanent failure.
        """
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_sent_at
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)

            session = await self._get_session()
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_web_page_preview": True,
            }
            delay = 1.0
            for attempt in range(1, max_retries + 1):
                try:
                    async with session.post(self._url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                        if resp.status == 200:
                            self._last_sent_at = time.monotonic()
                            return True
                        body = await resp.text()
                        logger.warning(
                            "Telegram returned HTTP %d (attempt %d/%d): %s",
                            resp.status,
                            attempt,
                            max_retries,
                            body[:120],
                        )
                except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                    logger.warning(
                        "Telegram send failed (attempt %d/%d): %s", attempt, max_retries, exc
                    )
                if attempt < max_retries:
                    await asyncio.sleep(delay * attempt)

            logger.error("Telegram message permanently failed after %d attempts.", max_retries)
            return False

    # ── Notification methods ───────────────────────────────────────────────

    async def send_startup(self, agents_info: dict[str, Any]) -> None:
        """Send system-started notification with agent details + menu keyboard."""
        import json as _json
        lines = ["<b>Multi-Agent System Started</b>"]
        for agent_id, info in agents_info.items():
            equity = info.get("equity", 0.0)
            strategy = info.get("strategy", agent_id)
            lines.append(f"  <b>{agent_id}</b> ({strategy}) — <code>${equity:.2f}</code>")
        mode = "SIMULATION" if cfg.SIMULATION_MODE else "LIVE"
        lines.append(f"\nMode: <b>{mode}</b>")

        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Статус",    "callback_data": "status"},
                    {"text": "Позиции",   "callback_data": "positions"},
                    {"text": "Ожидание",  "callback_data": "pending"},
                ],
                [
                    {"text": "Управление агентами", "callback_data": "agents"},
                ],
            ]
        }
        # Use sendMessage directly with keyboard
        session = await self._get_session()
        payload = {
            "chat_id": self._chat_id,
            "text": "\n".join(lines),
            "parse_mode": "HTML",
            "reply_markup": _json.dumps(keyboard),
        }
        try:
            async with session.post(self._url, json=payload, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                pass
        except Exception as exc:
            logger.warning("send_startup with keyboard failed: %s", exc)
            await self._send("\n".join(lines))

    async def send_signal(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        rr: float,
        score: float,
        qty: float,
        lev: int,
    ) -> None:
        """Send a trade signal notification (pre-entry)."""
        direction_emoji = "🟢" if side == "LONG" else "🔴"
        short_sym = symbol.split("/")[0]
        text = (
            f"{direction_emoji} <b>SIGNAL [{agent_id.upper()}]</b> — {short_sym}\n"
            f"Direction: <b>{side}</b>\n"
            f"Entry:  <code>{entry:.6g}</code>\n"
            f"SL:     <code>{sl:.6g}</code>\n"
            f"TP:     <code>{tp:.6g}</code>\n"
            f"RR:     <code>{rr:.2f}</code>  |  Score: <code>{score:.1f}</code>\n"
            f"Qty:    <code>{qty:.6g}</code>  @  <b>{lev}×</b>"
        )
        await self._send(text)

    async def send_position_opened(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        qty: float,
        lev: int,
        margin: float,
    ) -> None:
        """Send execution-confirmed notification."""
        direction_emoji = "🟢" if side == "LONG" else "🔴"
        short_sym = symbol.split("/")[0]
        sim_tag = " [SIM]" if cfg.SIMULATION_MODE else ""
        text = (
            f"{direction_emoji} <b>OPENED{sim_tag} [{agent_id.upper()}]</b> — {short_sym}\n"
            f"Side:   <b>{side}</b> @ <code>{entry:.6g}</code>\n"
            f"SL:     <code>{sl:.6g}</code>  |  TP: <code>{tp:.6g}</code>\n"
            f"Qty:    <code>{qty:.6g}</code>  ×{lev}  (margin ≈ <code>${margin:.4f}</code>)"
        )
        await self._send(text)

    async def send_limit_pending(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        qty: float,
        lev: int,
        margin: float,
        strategy_id: str = "",
    ) -> None:
        """Send pending limit order placed notification."""
        direction_emoji = "🟡" if side == "LONG" else "🟠"
        short_sym = symbol.split("/")[0]
        sim_tag = " [SIM]" if cfg.SIMULATION_MODE else ""
        strat_tag = f" [{strategy_id}]" if strategy_id else ""
        text = (
            f"{direction_emoji} <b>PENDING{sim_tag} [{agent_id.upper()}]{strat_tag}</b> — {short_sym}\n"
            f"Side:   <b>{side}</b> @ <code>{entry:.6g}</code>\n"
            f"SL:     <code>{sl:.6g}</code>  |  TP: <code>{tp:.6g}</code>\n"
            f"Qty:    <code>{qty:.6g}</code>  ×{lev}  (margin ≈ <code>${margin:.4f}</code>)"
        )
        await self._send(text)

    async def send_position_closed(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        entry: float,
        exit_price: float,
        pnl_usd: float,
        pnl_pct: float,
        reason: str,
    ) -> None:
        """Send position-closed notification."""
        short_sym = symbol.split("/")[0]
        pnl_emoji = "✅" if pnl_usd >= 0 else "❌"
        direction = "📈" if side == "LONG" else "📉"
        text = (
            f"{pnl_emoji} <b>CLOSED [{agent_id.upper()}]</b> — {short_sym} {direction}\n"
            f"Entry → Exit: <code>{entry:.6g}</code> → <code>{exit_price:.6g}</code>\n"
            f"PnL: <code>${pnl_usd:+.4f}</code> (<code>{pnl_pct:+.2f}%</code>)\n"
            f"Reason: <b>{reason}</b>"
        )
        await self._send(text)

    async def send_agent_paused(self, agent_id: str, reason: str) -> None:
        """Send agent-paused notification."""
        text = (
            f"⏸ <b>AGENT PAUSED</b> — {agent_id.upper()}\n"
            f"Reason: {reason}"
        )
        await self._send(text)

    async def send_daily_summary(self, stats: dict[str, Any]) -> None:
        """Send end-of-day PnL summary."""
        lines = ["📊 <b>Daily Summary</b>"]
        total_pnl = stats.get("total_pnl_usd", 0.0)
        total_trades = stats.get("total_trades", 0)
        winrate = stats.get("combined_winrate_pct", 0.0)
        lines.append(
            f"Total PnL: <code>${total_pnl:+.4f}</code>  |  "
            f"Trades: <code>{total_trades}</code>  |  "
            f"WR: <code>{winrate:.1f}%</code>"
        )
        for agent_id, info in stats.get("agents", {}).items():
            agent_pnl = info.get("pnl_usd", 0.0)
            agent_trades = info.get("trades", 0)
            agent_wr = info.get("winrate_pct", 0.0)
            equity = info.get("equity", 0.0)
            dd = info.get("drawdown_pct", 0.0)
            lines.append(
                f"\n<b>{agent_id}</b>: PnL <code>${agent_pnl:+.4f}</code>  "
                f"T={agent_trades}  WR={agent_wr:.0f}%  "
                f"Eq=<code>${equity:.2f}</code>  DD={dd:.1f}%"
            )
        await self._send("\n".join(lines))

    async def send_error(self, agent_id: str, error_msg: str) -> None:
        """Send error notification."""
        text = (
            f"🚨 <b>ERROR [{agent_id.upper()}]</b>\n"
            f"<code>{error_msg[:400]}</code>"
        )
        await self._send(text)


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        notifier = MultiAgentTelegramNotifier()
        await notifier.send_signal(
            agent_id="fvg",
            symbol="BTC/USDT:USDT",
            side="LONG",
            entry=60000,
            sl=59400,
            tp=62000,
            rr=2.0,
            score=82.5,
            qty=0.01,
            lev=10,
        )
        await notifier.close()
        print("Done.")

    asyncio.run(_demo())
