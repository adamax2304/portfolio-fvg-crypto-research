"""
# v2.6 — bot runtime control service
#
# Provides safe start/stop/status wrappers for fvg_bot.py process.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional


ROOT_DIR = Path(__file__).resolve().parents[2]
BOT_FILE = ROOT_DIR / "fvg_bot.py"
BOT_STDOUT = ROOT_DIR / "bot_live.out.log"
BOT_STDERR = ROOT_DIR / "bot_live.err.log"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BotRuntimeController:
    def __init__(self, root_dir: Path = ROOT_DIR) -> None:
        self.root_dir = root_dir
        self._lock = Lock()
        self._process: Optional[subprocess.Popen[str]] = None
        self._started_at: Optional[str] = None

    def _is_running(self, process: Optional[subprocess.Popen[str]]) -> bool:
        return bool(process and process.poll() is None)

    def status(self) -> dict[str, Any]:
        with self._lock:
            running = self._is_running(self._process)
            pid = self._process.pid if running and self._process else None
            return {
                "managed_running": running,
                "managed_pid": pid,
                "managed_started_at": self._started_at,
                "checked_at": utc_now_iso(),
            }

    def start(self, args: Optional[list[str]] = None) -> dict[str, Any]:
        with self._lock:
            if self._is_running(self._process):
                return {
                    "ok": True,
                    "message": "Bot is already running",
                    "status": self.status(),
                }

            if not BOT_FILE.exists():
                return {"ok": False, "message": f"Cannot find {BOT_FILE}"}

            run_args = [sys.executable, str(BOT_FILE)]
            if args:
                run_args.extend(args)

            stdout_file = BOT_STDOUT.open("a", encoding="utf-8")
            stderr_file = BOT_STDERR.open("a", encoding="utf-8")

            process = subprocess.Popen(
                run_args,
                cwd=str(self.root_dir),
                stdout=stdout_file,
                stderr=stderr_file,
                text=True,
            )
            self._process = process
            self._started_at = utc_now_iso()
            return {
                "ok": True,
                "message": "Bot process started",
                "status": self.status(),
            }

    def stop(self, timeout_sec: float = 8.0) -> dict[str, Any]:
        with self._lock:
            if not self._is_running(self._process):
                self._process = None
                return {
                    "ok": True,
                    "message": "Bot is not running",
                    "status": self.status(),
                }

            assert self._process is not None
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout_sec)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=3.0)

            self._process = None
            return {
                "ok": True,
                "message": "Bot process stopped",
                "status": self.status(),
            }

    def restart(self) -> dict[str, Any]:
        self.stop()
        return self.start()


bot_runtime = BotRuntimeController()

