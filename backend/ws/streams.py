"""
# v2.6 — WebSocket streams for live terminal updates
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.services import bot_runtime, data_store


router = APIRouter(tags=["ws"])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@router.websocket("/ws/price")
async def ws_price(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            state = data_store.load_state()
            symbols_payload = []
            for symbol, snapshot in state.get("symbols", {}).items():
                symbols_payload.append(
                    {
                        "symbol": symbol,
                        "short_symbol": snapshot.get("short_symbol", symbol.replace("/USDT:USDT", "USDT")),
                        "last_price": snapshot.get("last_price"),
                        "timeframe": snapshot.get("timeframe", "15m"),
                        "htf_bias": snapshot.get("htf_bias", "neutral"),
                        "zones_count": len(snapshot.get("zones", []) or []),
                    }
                )

            await websocket.send_json(
                {
                    "type": "price",
                    "timestamp": utc_now_iso(),
                    "cycle": state.get("cycle", 0),
                    "updated_at": state.get("updated_at"),
                    "symbols": symbols_payload,
                }
            )
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()


@router.websocket("/ws/trades")
async def ws_trades(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            positions = data_store.get_active_positions()
            pending = data_store.get_pending_signals(limit=100)
            await websocket.send_json(
                {
                    "type": "trades",
                    "timestamp": utc_now_iso(),
                    "positions": positions,
                    "pending_signals": pending,
                    "bot_runtime": bot_runtime.status(),
                }
            )
            await asyncio.sleep(2.5)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()


@router.websocket("/ws/pnl")
async def ws_pnl(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            overview = data_store.get_overview()
            analytics = data_store.get_analytics()
            await websocket.send_json(
                {
                    "type": "pnl",
                    "timestamp": utc_now_iso(),
                    "overview": overview,
                    "analytics": {
                        "win_rate": analytics.get("win_rate"),
                        "profit_factor": analytics.get("profit_factor"),
                        "avg_rr": analytics.get("avg_rr"),
                        "max_drawdown_pct": analytics.get("max_drawdown_pct"),
                        "trades_count": analytics.get("trades_count"),
                    },
                }
            )
            await asyncio.sleep(3.0)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()


@router.websocket("/ws/logs")
async def ws_logs(websocket: WebSocket) -> None:
    await websocket.accept()
    log_file: Path = data_store.log_file
    cursor = 0
    if log_file.exists():
        try:
            cursor = log_file.stat().st_size
        except OSError:
            cursor = 0

    try:
        await websocket.send_json(
            {
                "type": "logs",
                "timestamp": utc_now_iso(),
                "lines": data_store.tail_logs(lines=120),
            }
        )
        while True:
            if not log_file.exists():
                await asyncio.sleep(1.2)
                continue

            try:
                file_size = log_file.stat().st_size
            except OSError:
                await asyncio.sleep(1.2)
                continue

            if file_size < cursor:
                cursor = 0

            if file_size > cursor:
                with log_file.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(cursor)
                    chunk = handle.read()
                    cursor = handle.tell()
                lines = [line for line in chunk.splitlines() if line.strip()]
                if lines:
                    await websocket.send_json(
                        {
                            "type": "logs",
                            "timestamp": utc_now_iso(),
                            "lines": lines,
                        }
                    )
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        await websocket.close()

