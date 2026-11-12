"""
multi_agent/master_agent.py

MasterAgent: V7 portfolio swing trading orchestrator (2026-05-03).

Architecture:
  - V7Agent runs 4 backtest-validated swing strategies (730d, +1525%, DD=23.8%)
  - Shared-deposit model: full balance used for sizing at 4% risk per trade
  - Max 5 concurrent positions, correlation filter, FOMC pause, bear-mode protection

Active:
  V7Agent — ST_PIVOT_4H, SMA_CROSS_1D, RSI_REV_4H, FIB_PUMP_4H
    Backtest (730d, 27 symbols, full_audit_v7.py):
      Ret=+1525%, DD=23.8%, WR=45.2%, PF=1.60, n=168 trades
    With RC5 corr filter: Ret=+1606%, DD=24.7%, WF 7/10 positive

Disabled (preserved for reference — do NOT delete):
  MTFAgent (BBS_1H_5M, BBS_1H_15M, KC_4H_15M) — all enabled=False in MTF_STRATEGIES
  FVGAgent, BBSqueezeAgent, DensityAgent, MeanReversionAgent — not started
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import multi_agent.config as cfg
from multi_agent.agents.mtf_agent import MTFAgent
from multi_agent.agents.v7_agent import V7Agent
from multi_agent.shared.state_manager import StateManager
from multi_agent.shared.bybit_client import BybitClient
from multi_agent.shared.risk_manager import RiskManager
from multi_agent.shared.telegram_notifier import MultiAgentTelegramNotifier
from multi_agent.filters.oi_delta_filter import OIDeltaFilter
from multi_agent.modules.tg_command_handler import TelegramCommandHandler

# ── Logging setup ──────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    if logging.getLogger().handlers:
        return

    log_path = Path(cfg.MASTER_LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    from logging.handlers import RotatingFileHandler
    fh = RotatingFileHandler(
        log_path, maxBytes=50*1024*1024, backupCount=2,
        encoding="utf-8", delay=True
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    root.addHandler(fh)

    import io
    ch = logging.StreamHandler(io.TextIOWrapper(
        open(1, "wb", closefd=False), encoding="utf-8", errors="replace"))
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    root.addHandler(ch)


_setup_logging()
logger = logging.getLogger(__name__)

# Active agent IDs — V7 portfolio swing agent
AGENT_IDS = ("v7_agent",)
# MTFAgent kept but NOT started (strategies disabled in MTF_STRATEGIES)
_MTF_AGENT_DISABLED = True


class MasterAgent:
    """
    Regime-aware orchestrator that manages agent lifecycle, capital allocation,
    regime detection, and cross-agent conflict prevention.
    """

    def __init__(self) -> None:
        # ── Shared infrastructure ──
        self.state: StateManager = StateManager()
        self._client_instance: Optional[BybitClient] = None
        self.risk: RiskManager = RiskManager()
        self.tg: MultiAgentTelegramNotifier = MultiAgentTelegramNotifier()
        self._oi_filter: Optional[OIDeltaFilter] = None

        # ── V7 Agent (active) ──
        self.v7_agent: Optional[V7Agent] = None

        # ── MTF Agent (disabled — BBS/KC strategies paused) ──
        self.mtf_agent: Optional[MTFAgent] = None

        # ── Telegram command handler ──
        self._tg_cmd_handler: Optional[TelegramCommandHandler] = None
        self._tg_cmd_task: Optional[asyncio.Task] = None

        # ── Task tracking ──
        self._agent_tasks: dict[str, Optional[asyncio.Task]] = {
            aid: None for aid in AGENT_IDS
        }
        self._monitor_task: Optional[asyncio.Task] = None
        self._persist_task: Optional[asyncio.Task] = None
        self._running: bool = False

    # ── Startup ────────────────────���───────────────────────────────���─────────

    async def start(self) -> None:
        logger.info("MasterAgent starting up (regime-aware orchestrator v2).")

        self._client_instance = await BybitClient.get_instance()
        self._oi_filter = OIDeltaFilter(self._client_instance)

        # Build V7 Agent (active — 4 swing portfolio strategies)
        self.v7_agent = V7Agent(
            agent_id="v7_agent",
            state_manager=self.state,
            bybit_client=self._client_instance,
            risk_manager=self.risk,
            telegram=self.tg,
        )

        # MTFAgent kept but not started (BBS/KC disabled in config)
        self.mtf_agent = MTFAgent(
            agent_id="mtf_agent",
            state_manager=self.state,
            bybit_client=self._client_instance,
            risk_manager=self.risk,
            telegram=self.tg,
            oi_filter=self._oi_filter,
        )

        # Start state persistence loop
        self._persist_task = asyncio.create_task(
            self.state.start_persistence_loop(), name="state_persistence",
        )

        # Allocate capital
        allocation = await self._allocate_capital()

        # Telegram startup notification
        agents_info = {
            "v7_agent": {
                "equity": allocation.get("v7_agent", 0.0),
                "strategy": "V7 Swing Portfolio (ST_4H + SMA_1D + RSI_4H + FIB_4H)",
            },
        }
        try:
            await self.tg.send_startup(agents_info)
        except Exception as exc:
            logger.warning("Startup Telegram notification failed: %s", exc)

        # ── Pre-startup position sync ─────────────────────────────────────
        # IMPORTANT: sync with Bybit BEFORE starting agent scan loops.
        # This ensures that positions already open on the exchange are loaded
        # into state before the agent's first scan cycle, preventing duplicate
        # orders when the bot is restarted while positions are open.
        logger.info("Pre-startup: syncing positions from Bybit...")
        try:
            await self._sync_positions_with_bybit()
            logger.info("Pre-startup sync complete.")
        except Exception as exc:
            logger.warning("Pre-startup sync failed (non-fatal): %s", exc)

        # ── Recover missed trades from offline period ─────────────────────
        try:
            await self._recover_missed_trades()
        except Exception as exc:
            logger.warning("Missed trade recovery failed (non-fatal): %s", exc)

        # Start agent tasks (staggered 40s apart)
        self._running = True
        _stagger_sec = 40
        for i, aid in enumerate(AGENT_IDS):
            if i > 0:
                logger.info("Staggering agent %s start by %ds …", aid, i * _stagger_sec)
                await asyncio.sleep(_stagger_sec)
            await self._ensure_agent_running(aid)

        # Telegram command handler
        self._tg_cmd_handler = TelegramCommandHandler(
            state_manager=self.state, master_agent=self,
        )
        self._tg_cmd_task = asyncio.create_task(
            self._tg_cmd_handler.run(), name="tg_cmd_handler"
        )

        # Monitor loop
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(), name="master_monitor"
        )

        logger.info("MasterAgent fully started. Monitoring %d agents.", len(AGENT_IDS))
        await asyncio.gather(self._monitor_task, return_exceptions=True)

    # ── Capital allocation ──────────���───────────────────────────��───────────

    async def _allocate_capital(self) -> dict[str, float]:
        total = 0.0
        try:
            balance = await self._client_instance.fetch_balance()
            # Standard CCXT format
            total = float(
                balance.get("USDT", {}).get("total", 0.0)
                or balance.get("total", {}).get("USDT", 0.0)
                or 0.0
            )
            # Bybit UTA: equity in info.result.list[].coin[].equity
            if total < 1.0:
                try:
                    for acct in balance.get("info", {}).get("result", {}).get("list", []):
                        for coin_data in acct.get("coin", []):
                            if coin_data.get("coin") == "USDT":
                                eq = float(
                                    coin_data.get("equity", 0)
                                    or coin_data.get("walletBalance", 0)
                                    or 0
                                )
                                if eq > total:
                                    total = eq
                except Exception:
                    pass
        except Exception as exc:
            logger.error("Could not fetch balance: %s", exc)

        # V7 shared-deposit model: V7Agent uses the full balance
        allocation: dict[str, float] = {"v7_agent": total}
        try:
            await self.state.set_agent_equity("v7_agent", total)
        except Exception as exc:
            logger.warning("Could not set equity for v7_agent: %s", exc)
        logger.info("V7Agent allocated full balance: $%.4f", total)

        return allocation

    # ── Agent lifecycle ─────────────────���──────────────────────��─────────────

    def _get_agent(self, aid: str):
        return {"v7_agent": self.v7_agent, "mtf_agent": self.mtf_agent}.get(aid)

    async def _ensure_agent_running(self, aid: str) -> None:
        if not self._running:
            return
        task = self._agent_tasks.get(aid)
        if task is not None and not task.done():
            return
        try:
            status = await self.state.get_agent_status(aid)
            if status == "paused":
                return
        except Exception:
            pass
        agent = self._get_agent(aid)
        if agent is None:
            logger.error("No agent object for id=%s", aid)
            return
        logger.info("(Re)starting agent task: %s", aid)
        self._agent_tasks[aid] = asyncio.create_task(agent.run(), name=f"agent_{aid}")
        try:
            await self.state.set_agent_status(aid, "running")
        except Exception as exc:
            logger.warning("Could not set status for %s: %s", aid, exc)

    async def _restart_agent(self, aid: str) -> None:
        task = self._agent_tasks.get(aid)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._agent_tasks[aid] = None
        await self._ensure_agent_running(aid)

    # ── Monitor loop ──────────────────────────────────────────────────────────

    async def _monitor_loop(self) -> None:
        tick = 0
        ticks_per_day = max(1, int(86_400 / cfg.HEARTBEAT_INTERVAL_SEC))

        while self._running:
            try:
                await asyncio.sleep(cfg.HEARTBEAT_INTERVAL_SEC)
                tick += 1

                await self._check_heartbeats()
                await self._check_drawdown()
                await self._sync_positions_with_bybit()
                await self._check_portfolio_conflicts()

                for aid in AGENT_IDS:
                    await self._ensure_agent_running(aid)

                if tick % ticks_per_day == 0:
                    await self._send_daily_summary()

            except asyncio.CancelledError:
                logger.info("MasterAgent monitor loop cancelled.")
                break
            except Exception as exc:
                logger.exception("Monitor loop error: %s", exc)
                try:
                    await self.tg.send_error("master", str(exc))
                except Exception:
                    pass
                await asyncio.sleep(5)

    # ── Portfolio conflict prevention ─────────────────��────────────────────────

    async def _check_portfolio_conflicts(self) -> None:
        """
        Check for cross-agent conflicts:
        - Same symbol with opposing directions across agents
        - Total open positions exceeding portfolio limit
        """
        all_positions = await self.state.get_all_positions()
        symbol_positions: dict[str, list[tuple[str, str]]] = {}  # symbol -> [(agent_id, side)]

        total_open = 0
        for aid, positions in all_positions.items():
            for sym, pos in positions.items():
                total_open += 1
                side = pos.get("side", "LONG")
                symbol_positions.setdefault(sym, []).append((aid, side))

        # Check per-symbol conflicts
        for sym, entries in symbol_positions.items():
            if len(entries) > cfg.PORTFOLIO_MAX_PER_SYMBOL:
                logger.warning(
                    "Portfolio conflict: %s has %d positions (max %d): %s",
                    sym, len(entries), cfg.PORTFOLIO_MAX_PER_SYMBOL,
                    [(a, s) for a, s in entries],
                )

            # Check opposing directions
            sides = set(s for _, s in entries)
            if len(sides) > 1:
                logger.warning(
                    "Portfolio conflict: %s has opposing positions: %s",
                    sym, [(a, s) for a, s in entries],
                )

        if total_open > cfg.PORTFOLIO_MAX_OPEN_POSITIONS:
            logger.warning(
                "Portfolio over limit: %d open positions (max %d)",
                total_open, cfg.PORTFOLIO_MAX_OPEN_POSITIONS,
            )

    # ── Missed-trade recovery ─────────────────────────────────────────────────

    async def _recover_missed_trades(self) -> None:
        """
        Called once at startup. Fetches recent closed P&L from Bybit (last 48h),
        compares against trades_history.jsonl, and for any trade closed while the
        bot was offline: records it to JSONL and sends a TG notification.
        """
        if cfg.SIMULATION_MODE:
            return

        import json as _json
        from datetime import datetime as _dt, timezone as _tz
        from pathlib import Path as _Path

        # ── Load already-known trades ─────────────────────────────────────
        # Two dedup keys per trade:
        #   1. (sym, minute_bucket)  — exact timestamp match (±1 min)
        #   2. (sym, date, pnl_bin)  — same symbol + same day + PnL within $0.20
        # Bybit createdTime vs bot-detected close can differ by many minutes,
        # so key #2 catches duplicates that key #1 misses.
        hist_path = _Path(cfg.STATE_FILE).parent / "trades_history.jsonl"
        known_min:  set[tuple[str, int]]              = set()
        known_day:  set[tuple[str, str, int]]         = set()
        if hist_path.exists():
            try:
                for line in hist_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    t = _json.loads(line)
                    sym_raw   = t.get("symbol", "").replace("/", "").replace(":USDT", "")
                    closed_at = t.get("closed_at", "")
                    pnl_usd   = float(t.get("pnl_usd", 0) or 0)
                    if sym_raw and closed_at:
                        ts  = _dt.fromisoformat(closed_at).timestamp()
                        day = closed_at[:10]
                        known_min.add((sym_raw, round(ts / 60)))
                        known_day.add((sym_raw, day, round(pnl_usd / 0.20)))  # $0.20 bins
            except Exception as exc:
                logger.warning("recover_missed: could not read jsonl: %s", exc)

        # ── Fetch Bybit closed P&L ────────────────────────────────────────
        closed = await self._client_instance.fetch_closed_pnl(hours=48)
        if not closed:
            logger.info("recover_missed: no recent closed trades on Bybit.")
            return

        recovered = 0
        for entry in closed:
            sym_raw  = entry.get("symbol", "")
            close_ms = int(entry.get("createdTime", 0) or 0)
            if not sym_raw or not close_ms:
                continue
            close_ts  = close_ms / 1000.0
            close_min = round(close_ts / 60)

            close_day = _dt.fromtimestamp(close_ts, _tz.utc).strftime("%Y-%m-%d")
            pnl_usd_bybit = float(entry.get("closedPnl", 0) or 0)
            day_key = (sym_raw, close_day, round(pnl_usd_bybit / 0.20))

            if (sym_raw, close_min) in known_min or day_key in known_day:
                continue  # already recorded

            # Convert "LTCUSDT" → "LTC/USDT:USDT"
            base    = sym_raw[:-4] if sym_raw.endswith("USDT") else sym_raw
            ccxt_sym = f"{base}/USDT:USDT"

            # "Sell" = closing a LONG; "Buy" = closing a SHORT
            pos_side  = "LONG" if entry.get("side", "Sell") == "Sell" else "SHORT"
            avg_entry = float(entry.get("avgEntryPrice", 0) or 0)
            avg_exit  = float(entry.get("avgExitPrice", 0) or 0)
            qty       = float(entry.get("qty", 0) or 0)
            pnl_usd   = float(entry.get("closedPnl", 0) or 0)
            pnl_pct   = (avg_exit - avg_entry) / max(avg_entry, 1e-9) * 100.0
            if pos_side == "SHORT":
                pnl_pct = -pnl_pct
            reason    = "TP" if pnl_usd > 0 else "SL"
            close_iso = _dt.fromtimestamp(close_ts, _tz.utc).isoformat()

            logger.info(
                "recover_missed: %s %s %s pnl=$%.4f (%s) closed=%s",
                ccxt_sym, pos_side, reason, pnl_usd, f"{pnl_pct:+.2f}%", close_iso,
            )

            try:
                await self.state.record_trade("mtf_agent", {
                    "symbol":      ccxt_sym,
                    "strategy_id": "MTF",
                    "side":        pos_side,
                    "direction":   pos_side,
                    "entry":       avg_entry,
                    "exit":        avg_exit,
                    "sl":          0.0,
                    "tp":          0.0,
                    "qty":         qty,
                    "leverage":    int(entry.get("leverage", 1) or 1),
                    "pnl_usd":     pnl_usd,
                    "pnl_pct":     pnl_pct,
                    "reason":      reason,
                    "date":        _dt.fromtimestamp(close_ts, _tz.utc).strftime("%Y-%m-%d"),
                    "opened_at":   "",
                    "closed_at":   close_iso,
                    "recovered":   True,
                })
            except Exception as exc:
                logger.warning("recover_missed: record_trade failed: %s", exc)
                continue

            try:
                await self.tg.send_position_closed(
                    "mtf_agent", ccxt_sym, pos_side,
                    avg_entry, avg_exit, pnl_usd, pnl_pct,
                    f"{reason} [recovered]",
                )
            except Exception as exc:
                logger.warning("recover_missed: TG send failed: %s", exc)

            recovered += 1

        if recovered:
            logger.info("recover_missed: %d missed trade(s) recovered and sent to TG.", recovered)
        else:
            logger.info("recover_missed: all recent Bybit trades already recorded.")

    # ── Bybit position sync ──────────────────────────────────────────��───────

    async def _sync_positions_with_bybit(self) -> None:
        if cfg.SIMULATION_MODE:
            return
        try:
            real_positions = await self._client_instance.fetch_all_positions()
        except Exception as exc:
            logger.warning("Could not fetch Bybit positions: %s", exc)
            return

        real_map: dict[str, dict] = {}
        for pos in real_positions:
            sym = pos.get("symbol", "")
            size = float(pos.get("contracts", 0) or pos.get("info", {}).get("size", 0) or 0)
            if sym and size != 0:
                real_map[sym] = pos

        all_state = await self.state.get_all_positions()
        for aid in AGENT_IDS:
            state_positions = all_state.get(aid, {})
            for sym in list(state_positions.keys()):
                if sym not in real_map:
                    pos_data = state_positions[sym]
                    logger.info(
                        "Sync: %s position %s not on Bybit — recording close "
                        "(manual/SL/TP on exchange)",
                        aid, sym,
                    )
                    agent = self._get_agent(aid)
                    if agent is not None and hasattr(agent, "handle_external_close"):
                        # handle_external_close fetches exit price, records trade,
                        # sends TG notification, removes from state, applies cooldown
                        try:
                            await agent.handle_external_close(sym, pos_data)
                        except Exception as exc:
                            logger.warning("handle_external_close %s: %s", sym, exc)
                            # Fallback: at least remove from state
                            await self.state.remove_position(aid, sym)
                    else:
                        await self.state.remove_position(aid, sym)

        state_all_symbols: set[str] = set()
        refreshed = await self.state.get_all_positions()
        for pos_dict in refreshed.values():
            state_all_symbols.update(pos_dict.keys())

        for sym, pos in real_map.items():
            side_raw = pos.get("side", "long")
            side = "LONG" if str(side_raw).lower() in ("long", "buy") else "SHORT"
            entry = float(pos.get("entryPrice", 0) or 0)
            qty   = float(pos.get("contracts", 0) or 0)
            info  = pos.get("info", {}) or {}
            sl    = float(info.get("stopLoss", 0) or pos.get("stopLossPrice", 0) or 0)
            tp    = float(info.get("takeProfit", 0) or pos.get("takeProfitPrice", 0) or 0)

            if sym not in state_all_symbols:
                # New position on exchange, not yet in state → register
                logger.info(
                    "Sync: %s %s found on Bybit — registering (sl=%.6g tp=%.6g)",
                    sym, side, sl, tp,
                )
                await self.state.register_position("mtf_agent", sym, side, entry, sl, tp, qty)
            else:
                # Position exists in state — check if qty matches (catches restart duplicates)
                state_pos = refreshed.get("mtf_agent", {}).get(sym, {})
                state_qty = float(state_pos.get("qty", 0))
                if state_qty > 0 and abs(qty - state_qty) / state_qty > 0.05:
                    logger.warning(
                        "Sync qty mismatch %s: state=%.4f Bybit=%.4f — updating state qty",
                        sym, state_qty, qty,
                    )
                    # Update qty in state to match exchange reality
                    async with self.state._lock:
                        if sym in self.state._positions.get("mtf_agent", {}):
                            self.state._positions["mtf_agent"][sym]["qty"] = qty

    # ── Heartbeat check ────────────────────────��─────────────────────���───────

    async def _check_heartbeats(self) -> None:
        now = time.time()
        # MTF agent has a 5-minute cycle + potential refresh time.
        # Use 20 minutes as threshold to avoid false restarts.
        threshold = max(cfg.HEARTBEAT_INTERVAL_SEC * 10, 1200)

        for aid in AGENT_IDS:
            try:
                last = await self.state.get_last_heartbeat(aid)
            except Exception:
                continue
            if last is None:
                continue
            age = now - last
            if age > threshold:
                logger.warning("Agent %s heartbeat stale (%.0fs ago). Restarting.", aid, age)
                try:
                    await self.tg.send_error(aid, f"Heartbeat stale ({age:.0f}s). Restarting.")
                except Exception:
                    pass
                await self._restart_agent(aid)

    # ── Drawdown watchdog ───────────────────────────────���────────────────────

    async def _check_drawdown(self) -> None:
        try:
            stats = await self.state.get_combined_stats()
        except Exception:
            return

        agents_info = stats.get("agents", {})
        total_start = sum(info.get("starting_equity", 0.0) for info in agents_info.values())
        total_current = stats.get("total_equity", 0.0)

        if total_start <= 0:
            return

        combined_dd = (total_start - total_current) / total_start * 100.0
        if combined_dd >= cfg.MAX_TOTAL_DRAWDOWN_PCT:
            logger.warning("Combined DD %.2f%% >= %.2f%%. Pausing all agents.",
                           combined_dd, cfg.MAX_TOTAL_DRAWDOWN_PCT)
            for aid in AGENT_IDS:
                try:
                    current_status = await self.state.get_agent_status(aid)
                    if current_status == "running":
                        await self.state.set_agent_status(aid, "paused")
                        await self.tg.send_agent_paused(
                            aid, f"Combined DD {combined_dd:.2f}% exceeded {cfg.MAX_TOTAL_DRAWDOWN_PCT}%")
                except Exception as exc:
                    logger.warning("Could not pause %s: %s", aid, exc)

    # ── Daily summary ───────────────────────────────────────────────────��────

    async def _send_daily_summary(self) -> None:
        try:
            stats = await self.state.get_combined_stats()
            await self.tg.send_daily_summary(stats)
            logger.info("Daily summary sent.")
        except Exception as exc:
            logger.warning("Daily summary failed: %s", exc)

    # ── Graceful shutdown ────────────────────────────────────────────────────

    async def stop(self) -> None:
        logger.info("MasterAgent stopping.")
        self._running = False

        if self._tg_cmd_handler:
            await self._tg_cmd_handler.stop()
        if self._tg_cmd_task and not self._tg_cmd_task.done():
            self._tg_cmd_task.cancel()

        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()

        tasks_to_cancel = [t for t in self._agent_tasks.values() if t and not t.done()]
        for t in tasks_to_cancel:
            t.cancel()

        all_tasks = [
            self._monitor_task, self._tg_cmd_task,
        ] + list(self._agent_tasks.values())
        all_tasks = [t for t in all_tasks if t is not None]
        if all_tasks:
            await asyncio.gather(*all_tasks, return_exceptions=True)

        if self._persist_task and not self._persist_task.done():
            self._persist_task.cancel()
            await asyncio.gather(self._persist_task, return_exceptions=True)

        if self._client_instance is not None:
            try:
                await self._client_instance.close()
            except Exception as exc:
                logger.warning("Error closing BybitClient: %s", exc)

        try:
            await self.tg.close()
        except Exception as exc:
            logger.warning("Error closing Telegram: %s", exc)

        logger.info("MasterAgent stopped.")


# ── Entry point ──────────────────────────────��─────────────────────────���─────

async def _main() -> None:
    master = MasterAgent()
    try:
        await master.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        pass
    finally:
        await master.stop()


if __name__ == "__main__":
    asyncio.run(_main())
