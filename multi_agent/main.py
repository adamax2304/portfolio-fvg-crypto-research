"""
multi_agent/main.py

Entry point for the multi-agent trading system.

Run with:
    py -3 multi_agent/main.py

What this does
--------------
1. Configures logging for all modules.
2. Initialises the shared BybitClient singleton.
3. Creates the StateManager and starts its persistence loop.
4. Creates the Telegram notifier.
5. Initialises and starts the MasterAgent, which in turn
   spawns and supervises FvgAgent and SmcAgent.
6. Waits for a SIGINT / SIGTERM (Ctrl-C) and performs a clean shutdown.
7. Auto-restart watchdog: if the bot crashes unexpectedly, restarts it
   with exponential back-off (max 5 min between attempts).
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure the parent directory (fvgbot/) is on sys.path so that
#   "from multi_agent.xxx import yyy" works when running directly.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ─── Single-instance guard (pid-file) ─────────────────────────────────────────
_PID_FILE = ROOT / "multi_agent" / "fvgbot.pid"
_CRASH_LOG = ROOT / "multi_agent" / "logs" / "crash.log"


def _check_single_instance() -> None:
    """Exit immediately if another instance is already running."""
    if _PID_FILE.exists():
        try:
            existing_pid = int(_PID_FILE.read_text().strip())
            # Check whether the pid is actually alive (Windows-compatible)
            try:
                os.kill(existing_pid, 0)
                # Process exists — refuse to start
                print(
                    f"[FATAL] Another instance is already running (PID {existing_pid}). "
                    f"Delete {_PID_FILE} manually if that process is dead.",
                    file=sys.stderr,
                )
                sys.exit(1)
            except (ProcessLookupError, PermissionError):
                # PID is stale — safe to overwrite
                pass
        except (ValueError, OSError):
            pass
    _PID_FILE.write_text(str(os.getpid()))


def _remove_pid_file() -> None:
    try:
        _PID_FILE.unlink(missing_ok=True)
    except OSError:
        pass


_check_single_instance()

import multi_agent.config as cfg
from multi_agent.master_agent import MasterAgent


# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging() -> None:
    # master_agent._setup_logging() handles all logging config — skip here
    pass


logger = logging.getLogger(__name__)


# ─── Main coroutine ───────────────────────────────────────────────────────────

async def main() -> None:
    _setup_logging()
    logger.info("=" * 70)
    logger.info("Multi-Agent Trading System starting ...")
    logger.info("Simulation mode: %s", cfg.SIMULATION_MODE)
    logger.info("=" * 70)

    # Reset BybitClient singleton — each asyncio.run() creates a new event loop,
    # so any aiohttp sessions from the previous run are invalid and must be discarded.
    from multi_agent.shared.bybit_client import BybitClient as _BC
    _BC._instance = None
    _BC._lock = asyncio.Lock()

    # ── Master agent (self-contained; owns all infrastructure) ────────────
    master = MasterAgent()

    # ── Graceful shutdown handling ─────────────────────────────────────────
    stop_event = asyncio.Event()

    def _handle_signal() -> None:
        logger.info("Shutdown signal received.")
        stop_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, _handle_signal)
        except NotImplementedError:
            # Windows does not support add_signal_handler for all signals.
            pass

    # ── Start master (it internally starts all agents + persistence) ──────
    master_task = asyncio.create_task(
        master.start(),
        name="master_agent",
    )

    logger.info("Master agent started. Press Ctrl-C to stop.")

    # ── Wait for stop signal OR unexpected master_task exit ────────────────
    # If master_task exits without a stop signal being sent, it crashed —
    # propagate the exception so the watchdog can restart.
    stop_task = asyncio.create_task(stop_event.wait(), name="stop_event_waiter")

    try:
        done, pending = await asyncio.wait(
            {master_task, stop_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        if stop_task in done:
            # Clean shutdown path — user sent Ctrl-C or SIGTERM
            logger.info("Stop signal received, initiating shutdown ...")
        else:
            # master_task finished first — unexpected
            if not master_task.cancelled():
                exc = master_task.exception()
                if exc is not None:
                    logger.error(
                        "MasterAgent crashed: %s: %s",
                        type(exc).__name__, exc,
                    )
                    raise exc
            logger.error("MasterAgent exited unexpectedly (no exception). Restarting ...")
            raise RuntimeError("MasterAgent exited unexpectedly")
    finally:
        stop_task.cancel()
        try:
            await stop_task
        except (asyncio.CancelledError, Exception):
            pass

    # ── Shutdown ───────────────────────────────────────────────────────────
    logger.info("Shutting down ...")
    master_task.cancel()
    await asyncio.gather(master_task, return_exceptions=True)
    logger.info("Shutdown complete.")


# ─── Watchdog restart loop ────────────────────────────────────────────────────

_CLEAN_EXIT_EXCEPTIONS = (KeyboardInterrupt, SystemExit)
_MAX_RESTARTS = 100
_BASE_WAIT_SEC = 15   # first crash: 15s, second: 30s ... cap at 300s


def _log_crash(restart_count: int, exc: BaseException, wait_s: int) -> None:
    msg = (
        f"[WATCHDOG] CRASH #{restart_count} at "
        f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}: "
        f"{type(exc).__name__}: {exc}  (restarting in {wait_s}s)"
    )
    print(msg, file=sys.stderr)
    try:
        _CRASH_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _CRASH_LOG.open("a", encoding="utf-8") as f:
            f.write("\n" + "=" * 70 + "\n")
            f.write(msg + "\n")
            traceback.print_exc(file=f)
    except Exception:
        pass


if __name__ == "__main__":
    restart_count = 0

    while True:
        # Refresh PID file at each restart attempt so it stays current.
        try:
            _PID_FILE.write_text(str(os.getpid()))
        except OSError:
            pass

        try:
            asyncio.run(main())
            # asyncio.run returned normally — clean shutdown (Ctrl-C / SIGTERM).
            print("[WATCHDOG] Clean shutdown. Goodbye.")
            break

        except _CLEAN_EXIT_EXCEPTIONS:
            print("[WATCHDOG] Keyboard interrupt / system exit. Goodbye.")
            break

        except Exception as exc:
            restart_count += 1
            wait_s = min(300, _BASE_WAIT_SEC * restart_count)

            _log_crash(restart_count, exc, wait_s)

            if restart_count >= _MAX_RESTARTS:
                print(
                    f"[WATCHDOG] Reached {_MAX_RESTARTS} restarts. Giving up.",
                    file=sys.stderr,
                )
                break

            print(f"[WATCHDOG] Restarting in {wait_s}s ...", file=sys.stderr)
            time.sleep(wait_s)

        finally:
            _remove_pid_file()
