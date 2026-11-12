"""
multi_agent/modules/tg_command_handler.py

Telegram command + inline-keyboard handler for the V7 Portfolio Bot.

Команды:
    /menu | /start  — главное меню
    /status         — баланс, режим рынка, DD, сделки
    /positions      — открытые позиции
    /signals        — ожидающие сигналы + pending limit ордера
    /history        — последние 10 закрытых сделок

Кнопки меню:
    📊 Статус  |  📌 Позиции  |  👀 Сигналы  |  📜 История
    ⚙️ Управление
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import aiohttp

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import multi_agent.config as cfg
from multi_agent.shared.state_manager import StateManager

logger = logging.getLogger(__name__)

_TG_API = "https://api.telegram.org/bot{token}"

# ── V7 Agent registration ──────────────────────────────────────────────────────
_AGENT_ID    = "v7_agent"
_AGENT_LABEL = "V7 Portfolio (ST/SMA/RSI/FIB)"
_ALL_AGENTS  = (_AGENT_ID,)

# Стратегии V7 — короткие метки для дашборда
_STRAT_LABELS = {
    "V7_ST_PIVOT_4H":  "ST",
    "V7_SMA_CROSS_1D": "SMA",
    "V7_RSI_REV_4H":   "RSI",
    "V7_FIB_PUMP_4H":  "FIB",
}


# ── Inline keyboards ───────────────────────────────────────────────────────────

def _main_menu_keyboard() -> dict:
    return {
        "inline_keyboard": [
            [
                {"text": "📊 Статус",    "callback_data": "status"},
                {"text": "📌 Позиции",  "callback_data": "positions"},
            ],
            [
                {"text": "👀 Сигналы",  "callback_data": "signals"},
                {"text": "📜 История",  "callback_data": "history"},
            ],
            [
                {"text": "⚙️ Управление", "callback_data": "control"},
            ],
        ]
    }


async def _control_keyboard(state: StateManager) -> dict:
    status = await state.get_agent_status(_AGENT_ID)
    if status == "running":
        btn = {"text": f"🟢 {_AGENT_LABEL} — остановить", "callback_data": "disable:v7"}
    else:
        btn = {"text": f"⏸ {_AGENT_LABEL} — запустить", "callback_data": "enable:v7"}
    return {"inline_keyboard": [
        [btn],
        [{"text": "◀️ Назад", "callback_data": "menu"}],
    ]}


# ── Handler ────────────────────────────────────────────────────────────────────

class TelegramCommandHandler:
    def __init__(
        self,
        state_manager: StateManager,
        master_agent: Any,
        bot_token: str = cfg.TG_BOT_TOKEN,
        chat_id: str = cfg.TG_CHAT_ID,
        poll_timeout: int = 30,
    ) -> None:
        self._state       = state_manager
        self._master      = master_agent
        self._token       = bot_token
        self._chat_id     = str(chat_id)
        self._poll_timeout = poll_timeout
        self._base_url    = _TG_API.format(token=bot_token)
        self._session: Optional[aiohttp.ClientSession] = None
        self._offset: int  = 0
        self._running: bool = False
        self._rate_lock    = asyncio.Semaphore(1)
        self._last_sent_at = 0.0

    # ── Session ────────────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(ssl=False)
            )
        return self._session

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()

    # ── Low-level API ──────────────────────────────────────────────────────────

    async def _api(self, method: str, payload: dict) -> Optional[dict]:
        async with self._rate_lock:
            elapsed = time.monotonic() - self._last_sent_at
            if elapsed < 0.5:
                await asyncio.sleep(0.5 - elapsed)
            session = await self._get_session()
            try:
                async with session.post(
                    f"{self._base_url}/{method}",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    self._last_sent_at = time.monotonic()
                    data = await resp.json()
                    if resp.status != 200:
                        logger.warning("TG %s → %d: %s", method, resp.status, str(data)[:200])
                        return None
                    return data
            except Exception as exc:
                logger.warning("TG API error (%s): %s", method, exc)
                return None

    async def _send(self, text: str, keyboard: Optional[dict] = None) -> Optional[dict]:
        payload: dict = {
            "chat_id": self._chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = json.dumps(keyboard)
        return await self._api("sendMessage", payload)

    async def _edit(self, message_id: int, text: str,
                    keyboard: Optional[dict] = None) -> None:
        payload: dict = {
            "chat_id":    self._chat_id,
            "message_id": message_id,
            "text":       text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        if keyboard:
            payload["reply_markup"] = json.dumps(keyboard)
        await self._api("editMessageText", payload)

    async def _ack(self, cb_id: str, text: str = "") -> None:
        await self._api("answerCallbackQuery", {
            "callback_query_id": cb_id,
            "text": text,
        })

    # ── Poll loop ──────────────────────────────────────────────────────────────

    async def _drain_pending(self) -> None:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self._base_url}/getUpdates",
                params={"offset": -1, "timeout": 0},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    results = (await resp.json()).get("result", [])
                    if results:
                        self._offset = results[-1]["update_id"] + 1
                        logger.info("TG: drained %d updates, offset=%d",
                                    len(results), self._offset)
        except Exception as exc:
            logger.debug("TG drain error: %s", exc)

    async def run(self) -> None:
        self._running = True
        await self._drain_pending()
        logger.info("TelegramCommandHandler started.")
        while self._running:
            try:
                for upd in await self._poll():
                    await self._dispatch(upd)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.warning("TG poll error: %s", exc)
                await asyncio.sleep(5)

    async def stop(self) -> None:
        self._running = False
        await self.close()

    async def _poll(self) -> list[dict]:
        session = await self._get_session()
        try:
            async with session.get(
                f"{self._base_url}/getUpdates",
                params={"offset": self._offset, "timeout": self._poll_timeout,
                        "allowed_updates": ["message", "callback_query"]},
                timeout=aiohttp.ClientTimeout(total=self._poll_timeout + 10),
            ) as resp:
                if resp.status != 200:
                    await asyncio.sleep(3); return []
                results = (await resp.json()).get("result", [])
                if results:
                    self._offset = results[-1]["update_id"] + 1
                return results
        except Exception as exc:
            logger.debug("getUpdates error: %s", exc)
            await asyncio.sleep(3)
            return []

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    async def _dispatch(self, upd: dict) -> None:
        if "callback_query" in upd:
            cb = upd["callback_query"]
            if str(cb.get("message", {}).get("chat", {}).get("id", "")) != self._chat_id:
                return
            await self._handle_callback(cb)
            return
        msg = upd.get("message", {})
        if not msg:
            return
        if str(msg.get("chat", {}).get("id", "")) != self._chat_id:
            return
        text: str = (msg.get("text") or "").strip()
        if not text.startswith("/"):
            return
        cmd = text.split()[0].lower().split("@")[0]
        logger.info("TG cmd: %s", cmd)
        if cmd in ("/start", "/menu", "/help"):
            await self._show_menu()
        elif cmd == "/status":
            await self._send(await self._build_status(), _main_menu_keyboard())
        elif cmd == "/positions":
            await self._send(await self._build_positions(), _main_menu_keyboard())
        elif cmd in ("/signals", "/pending"):
            await self._send(await self._build_signals(), _main_menu_keyboard())
        elif cmd == "/history":
            await self._send(await self._build_history(), _main_menu_keyboard())
        else:
            await self._send("Используйте /menu", _main_menu_keyboard())

    async def _handle_callback(self, cb: dict) -> None:
        cb_id  = cb["id"]
        data   = cb.get("data", "")
        msg_id = cb.get("message", {}).get("message_id")
        await self._ack(cb_id)

        if data == "menu":
            await self._edit(msg_id, "Выберите действие:", _main_menu_keyboard())
        elif data == "status":
            await self._edit(msg_id, await self._build_status(), _main_menu_keyboard())
        elif data == "positions":
            await self._edit(msg_id, await self._build_positions(), _main_menu_keyboard())
        elif data == "signals":
            await self._edit(msg_id, await self._build_signals(), _main_menu_keyboard())
        elif data == "history":
            await self._edit(msg_id, await self._build_history(), _main_menu_keyboard())
        elif data == "control":
            kb = await _control_keyboard(self._state)
            await self._edit(msg_id, "⚙️ Управление агентом:", kb)
        elif data.startswith("enable:"):
            result = await self._do_enable(data.split(":")[1])
            await self._edit(msg_id, result, await _control_keyboard(self._state))
        elif data.startswith("disable:"):
            result = await self._do_disable(data.split(":")[1])
            await self._edit(msg_id, result, await _control_keyboard(self._state))

    async def _show_menu(self) -> None:
        await self._send("V7 Portfolio Bot — выберите:", _main_menu_keyboard())

    # ── Content builders ───────────────────────────────────────────────────────

    def _load_v7_trades(self) -> list[dict]:
        hist_path = Path(cfg.STATE_FILE).parent / "trades_history.jsonl"
        trades = []
        if not hist_path.exists():
            return trades
        try:
            with open(hist_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        t = json.loads(line)
                        if t.get("agent_id") == _AGENT_ID:
                            trades.append(t)
        except Exception:
            pass
        return trades

    def _fomc_status(self) -> str:
        """Return next FOMC date or 'blackout active' message."""
        try:
            from multi_agent.agents.v7_agent import _FOMC_MS, L4_BLACKOUT_H
            now_ms = int(time.time() * 1000)
            blackout_ms = L4_BLACKOUT_H * 3_600_000
            for fomc_ms in sorted(_FOMC_MS):
                if fomc_ms >= now_ms - blackout_ms:
                    dt = datetime.fromtimestamp(fomc_ms / 1000, tz=timezone.utc)
                    if now_ms >= fomc_ms - blackout_ms and now_ms <= fomc_ms + blackout_ms:
                        return f"🚫 FOMC BLACKOUT до {dt.strftime('%d %b %H:%M')} UTC"
                    return f"📅 Следующий FOMC: {dt.strftime('%d %b %Y')}"
        except Exception:
            pass
        return ""

    def _regime_status(self) -> str:
        """Get bear/bull regime from v7_agent cache."""
        try:
            v7 = getattr(self._master, "v7_agent", None)
            if v7 is None:
                return ""
            bear = getattr(v7, "_bear_cache", None)
            if bear is None:
                return ""
            return "🐻 Режим: BEAR (только LONG)" if bear else "🐂 Режим: BULL"
        except Exception:
            return ""

    async def _build_status(self) -> str:
        stats       = await self._state.get_combined_stats()
        agent_info  = stats.get("agents", {}).get(_AGENT_ID, {})
        trades      = self._load_v7_trades()
        now_iso     = datetime.now(timezone.utc).strftime("%d %b %H:%M UTC")

        equity   = agent_info.get("equity",        0.0)
        start_eq = agent_info.get("starting_equity", equity)
        dd       = agent_info.get("drawdown_pct",  0.0)
        status   = agent_info.get("status",        "stopped")
        n_pos    = agent_info.get("open_positions", 0)

        wins   = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        total  = len(trades)
        pnl    = sum(t.get("pnl_usd", 0) for t in trades)
        wr     = wins / total * 100 if total > 0 else 0.0

        # Calc equity growth
        growth = (equity - start_eq) / start_eq * 100 if start_eq > 0 else 0.0

        status_emoji = {"running": "🟢", "paused": "⏸", "stopped": "🔴"}.get(status, "⚪")
        mode = "LIVE 🔴" if not cfg.SIMULATION_MODE else "SIMULATION 🟡"

        lines = [
            f"<b>V7 Portfolio Bot</b>  <i>{now_iso}</i>",
            f"Режим: <b>{mode}</b>   {status_emoji} <b>{status.upper()}</b>",
            "",
            f"Equity:  <code>${equity:.2f}</code>  ({growth:+.1f}% от старта)",
            f"PnL:     <code>${pnl:+.2f}</code>   DD: <code>{dd:.1f}%</code>",
            f"Позиций: <b>{n_pos}</b>   Сделок: <b>{total}</b>  WR: <b>{wr:.0f}%</b>  (W:{wins}/L:{total-wins})",
        ]

        # Режим рынка и FOMC
        regime = self._regime_status()
        if regime:
            lines.append(regime)
        fomc = self._fomc_status()
        if fomc:
            lines.append(fomc)

        # Per-strategy breakdown
        if trades:
            per: dict[str, list] = {}
            for t in trades:
                sid = t.get("strategy_id", "?")
                per.setdefault(sid, []).append(t)
            lines.append("\n<b>По стратегиям:</b>")
            for sid, tl in sorted(per.items()):
                label = _STRAT_LABELS.get(sid, sid)
                w = sum(1 for t in tl if t.get("pnl_usd", 0) > 0)
                p = sum(t.get("pnl_usd", 0) for t in tl)
                wr_s = w / len(tl) * 100
                lines.append(
                    f"  <b>{label}</b>: {len(tl)} сделок  "
                    f"WR {wr_s:.0f}%  PnL <code>${p:+.2f}</code>"
                )

        return "\n".join(lines)

    async def _build_positions(self) -> str:
        positions = (await self._state.get_all_positions()).get(_AGENT_ID, {})

        # Also show pending limit orders
        v7 = getattr(self._master, "v7_agent", None)
        pending_limit: dict = getattr(v7, "_pending_limit", {}) if v7 else {}

        if not positions and not pending_limit:
            return "<b>Открытых позиций нет</b>\n\nБот ждёт сигнала 👀"

        lines = [f"<b>Позиции V7</b>  ({len(positions)} открытых)\n"]

        for sym, pos in positions.items():
            side   = pos.get("side", "?")
            entry  = pos.get("entry", 0.0)
            sl     = pos.get("sl",    0.0)
            tp     = pos.get("tp",    0.0)
            qty    = pos.get("qty",   0.0)
            strat  = _STRAT_LABELS.get(pos.get("strategy_id", ""), pos.get("strategy_id", "?"))
            opened = pos.get("opened_at", "")[:16].replace("T", " ")
            emoji  = "🟢" if side == "LONG" else "🔴"
            short  = sym.split("/")[0]
            # Risk-to-reward
            if side == "LONG" and entry > 0 and sl > 0 and tp > 0:
                dist_sl = entry - sl
                dist_tp = tp - entry
                rr = dist_tp / dist_sl if dist_sl > 0 else 0
                rr_str = f"  RR: {rr:.1f}R"
            elif side == "SHORT" and entry > 0 and sl > 0 and tp > 0:
                dist_sl = sl - entry
                dist_tp = entry - tp
                rr = dist_tp / dist_sl if dist_sl > 0 else 0
                rr_str = f"  RR: {rr:.1f}R"
            else:
                rr_str = ""
            lines.append(
                f"{emoji} <b>{short}</b> {side}  [{strat}]{rr_str}\n"
                f"  Entry: <code>{entry:.6g}</code>  Qty: <code>{qty:.6g}</code>\n"
                f"  SL: <code>{sl:.6g}</code>  TP: <code>{tp:.6g}</code>\n"
                f"  <i>{opened}</i>"
            )

        if pending_limit:
            lines.append(f"\n<b>Лимитные ордера ({len(pending_limit)} ожидают fill):</b>")
            for (sym, strat_id), lim in pending_limit.items():
                direction = lim.get("direction", "?")
                ep = lim.get("entry_price", 0.0)
                sl = lim.get("sl", 0.0)
                tp = lim.get("tp", 0.0)
                short = sym.split("/")[0]
                label = _STRAT_LABELS.get(strat_id, strat_id)
                emoji = "🟡" if direction == "LONG" else "🟠"
                lines.append(
                    f"{emoji} <b>{short}</b> {direction} [{label}]\n"
                    f"  @ <code>{ep:.6g}</code>  SL: <code>{sl:.6g}</code>  TP: <code>{tp:.6g}</code>"
                )

        return "\n".join(lines)

    async def _build_signals(self) -> str:
        v7 = getattr(self._master, "v7_agent", None)
        pending: dict = getattr(v7, "_pending", {}) if v7 else {}
        pending_limit: dict = getattr(v7, "_pending_limit", {}) if v7 else {}

        if not pending and not pending_limit:
            return "<b>Активных сигналов нет</b>\n\nV7 сканирует 27 монет каждые 5 мин."

        lines = []

        if pending:
            lines.append(f"<b>Новые сигналы ({len(pending)}):</b>")
            for (sym, strat_id), sig in pending.items():
                direction = sig.get("direction", "?")
                entry     = sig.get("entry",     0.0)
                sl        = sig.get("sl",         0.0)
                tp        = sig.get("tp",         0.0)
                dist_atr  = sig.get("dist_atr",   0.0)
                label     = _STRAT_LABELS.get(strat_id, strat_id)
                short     = sym.split("/")[0]
                emoji     = "🟢" if direction == "LONG" else "🔴"
                lines.append(
                    f"{emoji} <b>{short}</b> {direction}  [{label}]  dist={dist_atr:.1f}×ATR\n"
                    f"  Entry: <code>{entry:.6g}</code>  SL: <code>{sl:.6g}</code>  TP: <code>{tp:.6g}</code>"
                )

        if pending_limit:
            lines.append(f"\n<b>Ждут fill ({len(pending_limit)}):</b>")
            for (sym, strat_id), lim in pending_limit.items():
                direction = lim.get("direction", "?")
                ep  = lim.get("entry_price", 0.0)
                sl  = lim.get("sl",           0.0)
                tp  = lim.get("tp",           0.0)
                cyc = lim.get("cycles",        0)
                label = _STRAT_LABELS.get(strat_id, strat_id)
                short = sym.split("/")[0]
                emoji = "🟡" if direction == "LONG" else "🟠"
                lines.append(
                    f"{emoji} <b>{short}</b> {direction} [{label}]  цикл {cyc}/3\n"
                    f"  @ <code>{ep:.6g}</code>  SL: <code>{sl:.6g}</code>  TP: <code>{tp:.6g}</code>"
                )

        return "\n".join(lines)

    async def _build_history(self) -> str:
        trades = self._load_v7_trades()
        if not trades:
            return "<b>История сделок пуста</b>"

        last10 = trades[-10:]
        total  = len(trades)
        wins   = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        pnl    = sum(t.get("pnl_usd", 0) for t in trades)
        wr     = wins / total * 100 if total > 0 else 0.0

        lines = [
            f"<b>История сделок</b>  (последние {len(last10)} из {total})",
            f"Всего PnL: <code>${pnl:+.2f}</code>  WR: <b>{wr:.0f}%</b>  (W:{wins}/L:{total-wins})\n",
        ]

        for t in reversed(last10):
            sym    = t.get("symbol", "?").split("/")[0]
            side   = t.get("side",   "?")
            p      = t.get("pnl_usd", 0.0)
            reason = t.get("reason", "?")
            strat  = _STRAT_LABELS.get(t.get("strategy_id", ""), t.get("strategy_id", "?"))
            date   = t.get("closed_at", "")[:10]
            emoji  = "✅" if p > 0 else "❌"
            lines.append(
                f"{emoji} <b>{sym}</b> {side}  [{strat}]  {date}\n"
                f"   PnL: <code>${p:+.4f}</code>  причина: {reason}"
            )

        return "\n".join(lines)

    # ── Agent control ──────────────────────────────────────────────────────────

    async def _do_enable(self, alias: str) -> str:
        if alias != "v7":
            return "Неизвестный агент"
        current = await self._state.get_agent_status(_AGENT_ID)
        if current == "running":
            return f"<b>{_AGENT_LABEL}</b> уже запущен."
        await self._state.set_agent_status(_AGENT_ID, "running")
        try:
            await self._master._ensure_agent_running(_AGENT_ID)
        except Exception as exc:
            logger.warning("Could not restart v7: %s", exc)
        logger.info("v7_agent enabled via TG.")
        return f"✅ <b>{_AGENT_LABEL}</b> включён."

    async def _do_disable(self, alias: str) -> str:
        if alias != "v7":
            return "Неизвестный агент"
        current = await self._state.get_agent_status(_AGENT_ID)
        if current == "paused":
            return f"<b>{_AGENT_LABEL}</b> уже остановлен."
        await self._state.set_agent_status(_AGENT_ID, "paused")
        try:
            task = self._master._agent_tasks.get(_AGENT_ID)
            if task and not task.done():
                task.cancel()
        except Exception as exc:
            logger.warning("Could not cancel v7 task: %s", exc)
        logger.info("v7_agent disabled via TG.")
        return f"⏸ <b>{_AGENT_LABEL}</b> остановлен."
