"""
multi_agent/shared/state_manager.py

Thread-safe (asyncio-based) shared state synchronisation between agents.

Every agent registers its positions, equity, heartbeats and status here.
State is persisted to state.json every STATE_PERSIST_INTERVAL_SEC seconds so
that it survives a restart.

Typical usage
-------------
    sm = StateManager()
    asyncio.create_task(sm.start_persistence_loop())

    await sm.register_position("fvg", "BTC/USDT:USDT", "LONG", 60000, 59000, 62000, 0.01)
    locked = await sm.is_symbol_locked("BTC/USDT:USDT")
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config as cfg

logger = logging.getLogger(__name__)


class StateManager:
    """
    Central in-memory state store for the multi-agent system.

    All mutating operations are protected by a single asyncio.Lock so that
    concurrent agents cannot corrupt shared data.
    """

    def __init__(self) -> None:
        self._lock: asyncio.Lock = asyncio.Lock()
        # {agent_id: {symbol: position_dict}}
        self._positions: dict[str, dict[str, dict]] = {}
        # {agent_id: {symbol: pending_signal_dict}}  — pre-alerts / approach signals
        self._pending_signals: dict[str, dict[str, dict]] = {}
        # {agent_id: float}  (current equity allocated to agent)
        self._equities: dict[str, float] = {}
        # {agent_id: float}  (equity at agent start, for drawdown calc)
        self._starting_equities: dict[str, float] = {}
        # {agent_id: str}  "running" | "paused" | "stopped"
        self._statuses: dict[str, str] = {}
        # {agent_id: float}  (epoch time of last heartbeat)
        self._heartbeats: dict[str, float] = {}
        # {agent_id: [trade_dict, ...]}
        self._trade_log: dict[str, list[dict]] = {}
        self._state_file = Path(cfg.STATE_FILE)
        self._persist_interval = cfg.STATE_PERSIST_INTERVAL_SEC
        self._running = False

    # ── Persistence ────────────────────────────────────────────────────────

    async def start_persistence_loop(self) -> None:
        """Coroutine: persist state to disk every STATE_PERSIST_INTERVAL_SEC."""
        self._running = True
        logger.info("StateManager persistence loop started (interval=%ds).", self._persist_interval)
        while self._running:
            await asyncio.sleep(self._persist_interval)
            await self._save_state()

    async def stop(self) -> None:
        """Stop the persistence loop and do a final save."""
        self._running = False
        await self._save_state()

    async def _save_state(self) -> None:
        """Serialise current state to JSON on disk."""
        snapshot = await self._snapshot()
        try:
            self._state_file.parent.mkdir(parents=True, exist_ok=True)
            self._state_file.write_text(
                json.dumps(snapshot, indent=2, default=str), encoding="utf-8"
            )
            logger.debug("State persisted to %s", self._state_file)
        except Exception as exc:
            logger.error("State persistence failed: %s", exc)

    async def load_state(self) -> None:
        """Load persisted state from disk (call once at startup)."""
        if not self._state_file.exists():
            return
        try:
            raw = json.loads(self._state_file.read_text(encoding="utf-8"))
            async with self._lock:
                self._positions = raw.get("positions", {})
                self._pending_signals = raw.get("pending_signals", {})
                self._equities = raw.get("equities", {})
                self._starting_equities = raw.get("starting_equities", {})
                self._statuses = raw.get("statuses", {})
                self._heartbeats = raw.get("heartbeats", {})
                self._trade_log = raw.get("trade_log", {})
            logger.info("State loaded from %s.", self._state_file)
        except Exception as exc:
            logger.warning("Could not load state: %s", exc)

    async def _snapshot(self) -> dict:
        async with self._lock:
            return {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "positions": self._positions,
                "pending_signals": self._pending_signals,
                "equities": self._equities,
                "starting_equities": self._starting_equities,
                "statuses": self._statuses,
                "heartbeats": self._heartbeats,
                "trade_log": self._trade_log,
            }

    # ── Positions ──────────────────────────────────────────────────────────

    async def register_position(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        qty: float,
    ) -> None:
        """
        Register a newly opened position.

        Parameters
        ----------
        agent_id : identifier of the agent owning the position
        symbol   : e.g. "BTC/USDT:USDT"
        side     : "LONG" or "SHORT"
        entry    : entry price
        sl       : stop-loss price
        tp       : take-profit price
        qty      : position size in base units
        """
        async with self._lock:
            if agent_id not in self._positions:
                self._positions[agent_id] = {}
            self._positions[agent_id][symbol] = {
                "agent_id": agent_id,
                "symbol": symbol,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "qty": qty,
                "opened_at": datetime.now(timezone.utc).isoformat(),
            }
        logger.info("Position registered: agent=%s %s %s entry=%.6f", agent_id, symbol, side, entry)

    async def remove_position(self, agent_id: str, symbol: str) -> None:
        """Remove a closed position."""
        async with self._lock:
            agent_positions = self._positions.get(agent_id, {})
            agent_positions.pop(symbol, None)
        logger.info("Position removed: agent=%s %s", agent_id, symbol)

    async def is_symbol_locked(self, symbol: str) -> bool:
        """Return True if any agent currently holds a position in *symbol*."""
        async with self._lock:
            for agent_positions in self._positions.values():
                if symbol in agent_positions:
                    return True
        return False

    async def get_all_positions(self) -> dict[str, dict[str, dict]]:
        """Return a copy of all open positions keyed by agent_id -> symbol."""
        async with self._lock:
            return {aid: dict(pos) for aid, pos in self._positions.items()}

    async def get_agent_positions(self, agent_id: str) -> dict[str, dict]:
        """Return open positions for a single agent."""
        async with self._lock:
            return dict(self._positions.get(agent_id, {}))

    # ── Pending signals (pre-alerts) ───────────────────────────────────────

    async def register_pending_signal(
        self,
        agent_id: str,
        symbol: str,
        side: str,
        entry: float,
        sl: float,
        tp: float,
        score: float,
        gap_pct: float = 0.0,
        distance_pct: float = 0.0,
    ) -> None:
        """Register a pre-alert/pending signal (price approaching entry zone)."""
        async with self._lock:
            if agent_id not in self._pending_signals:
                self._pending_signals[agent_id] = {}
            self._pending_signals[agent_id][symbol] = {
                "agent_id": agent_id,
                "symbol": symbol,
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "score": score,
                "gap_pct": gap_pct,
                "distance_pct": distance_pct,
                "alerted_at": datetime.now(timezone.utc).isoformat(),
            }

    async def clear_pending_signal(self, agent_id: str, symbol: str) -> None:
        """Remove a pending signal (fired or invalidated)."""
        async with self._lock:
            self._pending_signals.get(agent_id, {}).pop(symbol, None)

    async def get_all_pending_signals(self) -> dict[str, dict[str, dict]]:
        """Return all pending signals keyed by agent_id -> symbol."""
        async with self._lock:
            return {aid: dict(sigs) for aid, sigs in self._pending_signals.items()}

    async def get_agent_pending_signals(self, agent_id: str) -> dict[str, dict]:
        """Return pending signals for a single agent."""
        async with self._lock:
            return dict(self._pending_signals.get(agent_id, {}))

    # ── Equity ─────────────────────────────────────────────────────────────

    async def get_total_equity(self) -> float:
        """Return the sum of all agent equities."""
        async with self._lock:
            return sum(self._equities.values())

    async def get_agent_equity(self, agent_id: str) -> float:
        """Return the current equity allocated to *agent_id*."""
        async with self._lock:
            return self._equities.get(agent_id, 0.0)

    async def set_agent_equity(self, agent_id: str, equity: float) -> None:
        """Update the equity for *agent_id*. Also sets the starting equity if not set."""
        async with self._lock:
            self._equities[agent_id] = equity
            if agent_id not in self._starting_equities:
                self._starting_equities[agent_id] = equity
                logger.info("Starting equity set for agent=%s: %.4f", agent_id, equity)

    # ── Status ─────────────────────────────────────────────────────────────

    async def get_agent_status(self, agent_id: str) -> str:
        """Return the status string for *agent_id* (default 'stopped')."""
        async with self._lock:
            return self._statuses.get(agent_id, "stopped")

    async def set_agent_status(self, agent_id: str, status: str) -> None:
        """
        Set the status of *agent_id*.

        Valid values: "running" | "paused" | "stopped"
        """
        async with self._lock:
            self._statuses[agent_id] = status
        logger.info("Agent %s status -> %s", agent_id, status)

    # ── Heartbeats ─────────────────────────────────────────────────────────

    async def record_heartbeat(self, agent_id: str) -> None:
        """Update the last-seen timestamp for *agent_id*."""
        async with self._lock:
            self._heartbeats[agent_id] = time.time()

    async def get_last_heartbeat(self, agent_id: str) -> Optional[float]:
        """Return the epoch timestamp of the last heartbeat, or None."""
        async with self._lock:
            return self._heartbeats.get(agent_id)

    # ── Drawdown ───────────────────────────────────────────────────────────

    async def get_drawdown(self, agent_id: str) -> float:
        """
        Calculate the current drawdown percentage for *agent_id* from its
        starting equity.

        Returns 0.0 if no starting equity has been recorded.
        """
        async with self._lock:
            start = self._starting_equities.get(agent_id, 0.0)
            current = self._equities.get(agent_id, 0.0)
        if start <= 0:
            return 0.0
        dd = (start - current) / start * 100.0
        return max(dd, 0.0)

    # ── Trade log ──────────────────────────────────────────────────────────

    async def record_trade(self, agent_id: str, trade_dict: dict) -> None:
        """Append *trade_dict* to the shared trade log for *agent_id*.

        Also writes immediately to trades_history.jsonl (append-only) so that
        trade history is never lost on abrupt restart.
        """
        async with self._lock:
            if agent_id not in self._trade_log:
                self._trade_log[agent_id] = []
            self._trade_log[agent_id].append(trade_dict)
            # ── Immediate append to persistent JSONL file ─────────────────
            try:
                hist_path = self._state_file.parent / "trades_history.jsonl"
                record = {"agent_id": agent_id, **trade_dict}
                with open(hist_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
            except Exception as exc:
                logger.warning("Could not write to trades_history.jsonl: %s", exc)

    async def get_trades(self, agent_id: str) -> list[dict]:
        """Return the trade log for *agent_id*."""
        async with self._lock:
            return list(self._trade_log.get(agent_id, []))

    # ── Aggregate statistics ───────────────────────────────────────────────

    async def get_combined_stats(self) -> dict[str, Any]:
        """
        Return aggregate statistics across all agents.

        Returns
        -------
        dict with keys:
            total_equity, agents (per-agent breakdown),
            total_trades, total_wins, total_losses,
            total_pnl_usd, combined_winrate_pct
        """
        async with self._lock:
            stats: dict[str, Any] = {
                "total_equity": sum(self._equities.values()),
                "agents": {},
                "total_trades": 0,
                "total_wins": 0,
                "total_losses": 0,
                "total_pnl_usd": 0.0,
            }
            # Collect all known agent IDs from any source
            all_known = set(self._equities) | set(self._statuses) | set(self._trade_log)
            for agent_id in all_known:
                trades = self._trade_log.get(agent_id, [])
                wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
                losses = sum(1 for t in trades if t.get("pnl_usd", 0) <= 0)
                pnl = sum(t.get("pnl_usd", 0.0) for t in trades)
                start = self._starting_equities.get(agent_id, 0.0)
                current = self._equities.get(agent_id, 0.0)
                dd = (start - current) / start * 100.0 if start > 0 else 0.0
                stats["agents"][agent_id] = {
                    "equity": current,
                    "starting_equity": start,
                    "drawdown_pct": max(dd, 0.0),
                    "status": self._statuses.get(agent_id, "stopped"),
                    "trades": len(trades),
                    "wins": wins,
                    "losses": losses,
                    "pnl_usd": pnl,
                    "winrate_pct": wins / len(trades) * 100 if trades else 0.0,
                    "open_positions": len(self._positions.get(agent_id, {})),
                }
                stats["total_trades"] += len(trades)
                stats["total_wins"] += wins
                stats["total_losses"] += losses
                stats["total_pnl_usd"] += pnl

        total = stats["total_trades"]
        stats["combined_winrate_pct"] = (
            stats["total_wins"] / total * 100.0 if total > 0 else 0.0
        )
        return stats


if __name__ == "__main__":
    import asyncio

    async def _demo() -> None:
        sm = StateManager()
        await sm.set_agent_equity("fvg", 500.0)
        await sm.set_agent_equity("smc", 500.0)
        await sm.register_position("fvg", "BTC/USDT:USDT", "LONG", 60000, 59000, 62000, 0.01)
        print("BTC locked:", await sm.is_symbol_locked("BTC/USDT:USDT"))
        print("ETH locked:", await sm.is_symbol_locked("ETH/USDT:USDT"))
        stats = await sm.get_combined_stats()
        print("Stats:", json.dumps(stats, indent=2))

    asyncio.run(_demo())
