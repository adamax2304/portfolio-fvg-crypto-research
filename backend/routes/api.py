"""
# v2.6 — REST API for premium terminal dashboard
# v2 endpoints added for MTF trading dashboard
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from backend.services import bot_runtime, data_store

# Ensure multi_agent package is importable
_project_root = str(Path(__file__).resolve().parents[2])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

router = APIRouter(prefix="/api", tags=["api"])


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class SettingsUpdateRequest(BaseModel):
    updates: dict[str, Any] = Field(default_factory=dict)
    apply_restart_scan: bool = False


@router.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "timestamp": utc_now_iso(),
        "bot_runtime": bot_runtime.status(),
    }


@router.get("/state")
def state() -> dict[str, Any]:
    return data_store.load_state()


@router.get("/overview")
def overview() -> dict[str, Any]:
    payload = data_store.get_overview()
    payload["bot_runtime"] = bot_runtime.status()
    return payload


@router.get("/analytics")
def analytics() -> dict[str, Any]:
    return data_store.get_analytics()


@router.get("/market/symbols")
def market_symbols(
    q: str = Query(default=""),
    min_score: float = Query(default=0.0),
    min_gap: float = Query(default=0.0),
    include_mitigated: bool = Query(default=False),
) -> dict[str, Any]:
    rows = data_store.get_symbols(
        query=q,
        min_score=min_score,
        min_gap=min_gap,
        include_mitigated=include_mitigated,
    )
    return {
        "items": rows,
        "count": len(rows),
        "filters": {
            "query": q,
            "min_score": min_score,
            "min_gap": min_gap,
            "include_mitigated": include_mitigated,
        },
    }


@router.get("/market/chart-data")
def market_chart_data(
    symbol: str,
    timeframe: str = Query(default="15m"),
    limit: int = Query(default=600, ge=120, le=10000),
    min_score: float = Query(default=0.0),
    min_gap: float = Query(default=0.0),
    min_rr: float = Query(default=1.0),
    include_mitigated: bool = Query(default=False),
) -> dict[str, Any]:
    try:
        candles = data_store.fetch_candles(symbol=symbol, timeframe=timeframe, limit=limit)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to fetch candles for {symbol}: {exc}") from exc

    zones = data_store.get_zones(
        symbol=symbol,
        include_mitigated=include_mitigated,
        min_score=min_score,
        min_gap=min_gap,
        min_rr=min_rr,
    )
    state = data_store.load_state()
    snapshot = state.get("symbols", {}).get(symbol, {})
    return {
        "symbol": symbol,
        "short_symbol": snapshot.get("short_symbol", symbol.replace("/USDT:USDT", "USDT")),
        "timeframe": timeframe,
        "candles": candles,
        "zones": zones,
        "last_price": snapshot.get("last_price"),
        "htf_bias": snapshot.get("htf_bias"),
        "updated_at": state.get("updated_at"),
    }


@router.get("/pending-signals")
def pending_signals(limit: int = Query(default=120, ge=1, le=500)) -> dict[str, Any]:
    items = data_store.get_pending_signals(limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/active-positions")
def active_positions() -> dict[str, Any]:
    items = data_store.get_active_positions()
    return {"items": items, "count": len(items)}


@router.get("/trade-history")
def trade_history(limit: int = Query(default=300, ge=1, le=5000)) -> dict[str, Any]:
    items = data_store.get_trade_history(limit=limit)
    return {"items": items, "count": len(items)}


@router.get("/logs")
def logs(lines: int = Query(default=250, ge=50, le=2000)) -> dict[str, Any]:
    return {"lines": data_store.tail_logs(lines=lines)}


@router.get("/settings")
def get_settings() -> dict[str, Any]:
    return data_store.load_config()


@router.put("/settings/bot")
def update_bot_settings(request: SettingsUpdateRequest) -> dict[str, Any]:
    if not isinstance(request.updates, dict) or not request.updates:
        raise HTTPException(status_code=400, detail="`updates` must be a non-empty object")
    config = data_store.save_config(bot_updates=request.updates)
    runtime_result: Optional[dict[str, Any]] = None
    if request.apply_restart_scan:
        runtime_result = bot_runtime.restart()
    return {
        "ok": True,
        "updated_at": config.get("updated_at"),
        "bot": config.get("bot", {}),
        "runtime": runtime_result,
    }


@router.put("/settings/visual")
def update_visual_settings(request: SettingsUpdateRequest) -> dict[str, Any]:
    if not isinstance(request.updates, dict) or not request.updates:
        raise HTTPException(status_code=400, detail="`updates` must be a non-empty object")
    config = data_store.save_config(visual_updates=request.updates)
    return {
        "ok": True,
        "updated_at": config.get("updated_at"),
        "visual": config.get("dashboard", {}).get("visual", {}),
    }


@router.get("/export/zones.csv")
def export_csv(
    include_mitigated: bool = Query(default=False),
    min_score: float = Query(default=0.0),
    min_gap: float = Query(default=0.0),
) -> PlainTextResponse:
    csv_text = data_store.export_zones_csv(
        include_mitigated=include_mitigated,
        min_score=min_score,
        min_gap=min_gap,
    )
    return PlainTextResponse(
        csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="fvg_zones_export.csv"'},
    )


@router.get("/export/zones.pine")
def export_pine(
    include_mitigated: bool = Query(default=False),
    min_score: float = Query(default=0.0),
    min_gap: float = Query(default=0.0),
) -> PlainTextResponse:
    pine = data_store.export_zones_pine(
        include_mitigated=include_mitigated,
        min_score=min_score,
        min_gap=min_gap,
    )
    return PlainTextResponse(
        pine,
        media_type="text/plain",
        headers={"Content-Disposition": 'attachment; filename="fvg_zones_export.pine"'},
    )


@router.get("/bot/status")
def bot_status() -> dict[str, Any]:
    state = data_store.load_state()
    return {
        "runtime": bot_runtime.status(),
        "state_bot_running": bool(state.get("bot_running", False)),
        "state_cycle": state.get("cycle", 0),
        "state_updated_at": state.get("updated_at"),
    }


@router.post("/bot/start")
def bot_start() -> dict[str, Any]:
    return bot_runtime.start()


@router.post("/bot/stop")
def bot_stop() -> dict[str, Any]:
    return bot_runtime.stop()


@router.post("/bot/restart")
def bot_restart() -> dict[str, Any]:
    return bot_runtime.restart()


@router.post("/charts/refresh")
def charts_refresh() -> dict[str, Any]:
    state = data_store.load_state()
    return {
        "ok": True,
        "message": "Refresh requested",
        "state_updated_at": state.get("updated_at"),
    }


@router.post("/charts/clear")
def charts_clear() -> dict[str, Any]:
    return {"ok": True, "message": "Charts cleared on client side"}


# ═════════════════════════════════════════════════════════════════════════════
# v2 — MTF Dashboard Endpoints
# ═════════════════════════════════════════════════════════════════════════════

def _load_mtf_strategies() -> dict[str, Any]:
    """Safely import MTF_STRATEGIES from multi_agent.config, returning {} if not defined."""
    try:
        from multi_agent.config import MTF_STRATEGIES  # type: ignore[import-untyped]
        return MTF_STRATEGIES
    except (ImportError, AttributeError):
        return {}


@router.get("/v2/dashboard")
def get_dashboard() -> dict[str, Any]:
    """Combined dashboard overview for the new MTF UI."""
    from multi_agent.config import SIMULATION_MODE, MAX_TOTAL_EXPOSURE_PCT  # type: ignore[import-untyped]

    # --- Primary source: multi_agent/logs/state.json ---
    balance = data_store.get_mtf_balance()
    starting_balance = data_store.get_mtf_starting_balance()
    positions = data_store.get_mtf_positions()

    # Fallback to fvg_zones.json if MTF state not available
    if balance <= 0:
        state = data_store.load_state()
        account = state.get("account", {})
        balance = account.get("equity", 1000.0)

    # Trade history — MTF-only, no old FVG fallback
    trades = data_store.get_mtf_trade_history(limit=500)

    # Stats from closed trades
    total_pnl_trades = sum(t.get("pnl_usd", 0) for t in trades)
    wins = [t for t in trades if t.get("pnl_usd", 0) > 0]
    losses = [t for t in trades if t.get("pnl_usd", 0) <= 0]
    win_rate = len(wins) / max(len(trades), 1) * 100

    # Total PnL = closed trades + unrealized drift vs starting balance
    # Use equity delta as ground truth if no trades yet
    total_pnl = total_pnl_trades if trades else round(balance - starting_balance, 4)

    # Today's trades
    today = date.today().isoformat()
    today_trades = [t for t in trades if (t.get("exit_time") or "").startswith(today)]
    daily_pnl = sum(t.get("pnl_usd", 0) for t in today_trades)

    return {
        "balance": balance,
        "starting_balance": starting_balance,
        "total_pnl": total_pnl,
        "daily_pnl": daily_pnl,
        "win_rate": round(win_rate, 1),
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "active_positions": len(positions),
        "positions": positions,
        "simulation_mode": SIMULATION_MODE,
        "max_exposure_pct": MAX_TOTAL_EXPOSURE_PCT,
    }


@router.get("/v2/strategies")
def get_strategies() -> list[dict[str, Any]]:
    """Returns list of MTF strategies with their current status."""
    mtf_strategies = _load_mtf_strategies()
    state = data_store.load_state()
    result = []
    for sid, cfg in mtf_strategies.items():
        # Get per-strategy stats from state if available
        strat_stats = state.get("strategy_stats", {}).get(sid, {})
        result.append({
            "id": sid,
            "name": cfg.get("name", sid),
            "description": cfg.get("description", ""),
            "enabled": cfg.get("enabled", False),
            "signal_tf": cfg.get("signal_tf", ""),
            "entry_tf": cfg.get("entry_tf", ""),
            "signal_type": cfg.get("signal_type", ""),
            "entry_mode": cfg.get("entry_mode", ""),
            "sl_atr_mult": cfg.get("sl_atr_mult", 0),
            "tp_r": cfg.get("tp_r", 0),
            "max_positions": cfg.get("max_positions", 0),
            "backtest": cfg.get("backtest", {}),
            "trades_today": strat_stats.get("trades_today", 0),
            "pnl_today": strat_stats.get("pnl_today", 0),
            "active_positions": strat_stats.get("active_positions", 0),
        })
    return result


@router.put("/v2/strategies/{strategy_id}/toggle")
def toggle_strategy(strategy_id: str) -> dict[str, Any]:
    """Enable or disable a strategy at runtime."""
    try:
        from multi_agent import config  # type: ignore[import-untyped]
        mtf = getattr(config, "MTF_STRATEGIES", None)
        if mtf is None or strategy_id not in mtf:
            raise HTTPException(status_code=404, detail="Strategy not found")
        current = mtf[strategy_id]["enabled"]
        mtf[strategy_id]["enabled"] = not current
        return {"id": strategy_id, "enabled": not current}
    except ImportError:
        raise HTTPException(status_code=404, detail="Strategy configuration not available")


@router.get("/v2/signals")
def get_signals() -> list[dict[str, Any]]:
    """Get pending signals / potential entries from all strategies."""
    # Primary source: multi_agent/logs/state.json pending_signals
    mtf_state = data_store.load_mtf_state()
    signals = mtf_state.get("pending_signals", {})
    result: list[dict[str, Any]] = []
    # Flatten: agent_id -> symbol -> signal_dict
    if isinstance(signals, dict):
        for agent_id, agent_sigs in signals.items():
            if isinstance(agent_sigs, dict):
                for sym, sig in agent_sigs.items():
                    if sig:
                        sig_copy = dict(sig)
                        sig_copy["strategy"] = agent_id
                        result.append(sig_copy)
            elif isinstance(agent_sigs, list):
                for sig in agent_sigs:
                    sig_copy = dict(sig)
                    sig_copy["strategy"] = agent_id
                    result.append(sig_copy)
    elif isinstance(signals, list):
        result = list(signals)
    return result


@router.get("/v2/trades")
def get_trades(
    limit: int = Query(default=200, ge=1, le=5000),
    strategy: Optional[str] = Query(default=None),
) -> list[dict[str, Any]]:
    """Trade history from MTF multi_agent/logs/state.json (no legacy fallback)."""
    return data_store.get_mtf_trade_history(limit=limit, strategy=strategy)


@router.get("/v2/chart/{symbol:path}")
def get_chart_data(
    symbol: str,
    timeframe: str = Query(default="15m"),
    limit: int = Query(default=200, ge=50, le=10000),
) -> dict[str, Any]:
    """Get OHLCV candles for charting."""
    try:
        candles = data_store.fetch_candles(symbol, timeframe, limit)
        return {"symbol": symbol, "timeframe": timeframe, "candles": candles}
    except Exception as e:
        return {"symbol": symbol, "timeframe": timeframe, "candles": [], "error": str(e)}


@router.put("/v2/settings")
def update_settings(body: dict[str, Any]) -> dict[str, Any]:
    """Update bot config settings at runtime."""
    try:
        from multi_agent import config  # type: ignore[import-untyped]
    except ImportError:
        raise HTTPException(status_code=500, detail="Config module not available")
    updates = body.get("updates", {})
    applied: list[str] = []
    for key, val in updates.items():
        if hasattr(config, key):
            setattr(config, key, val)
            applied.append(key)
    return {"status": "ok", "updated": applied}


@router.get("/v2/equity-curve")
def get_equity_curve() -> list[dict[str, Any]]:
    """Equity curve data computed from MTF trade history."""
    trades = data_store.get_mtf_trade_history(limit=1000)
    starting = data_store.get_mtf_starting_balance()
    curve: list[dict[str, Any]] = []
    balance = starting if starting > 0 else 1000.0
    for t in sorted(trades, key=lambda x: x.get("exit_time") or ""):
        balance += t.get("pnl_usd", 0)
        curve.append({
            "time":    t.get("exit_time", ""),
            "balance": round(balance, 2),
            "pnl":     t.get("pnl_usd", 0),
            "strategy": t.get("strategy", ""),
            "symbol":  t.get("short_symbol", t.get("symbol", "")),
        })
    return curve
