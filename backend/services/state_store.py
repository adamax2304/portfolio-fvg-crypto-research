"""
# v2.6 — premium terminal backend state service
#
# Reads bot runtime artifacts (fvg_zones.json, config.json, logs, backtest reports),
# applies safe caching, and exposes normalized data for REST + WebSocket layers.
"""

from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Optional

import ccxt


ROOT_DIR = Path(__file__).resolve().parents[2]
STATE_FILE = ROOT_DIR / "fvg_zones.json"
CONFIG_FILE = ROOT_DIR / "config.json"
LOG_FILE = ROOT_DIR / "fvg_bot.log"
BACKTEST_REPORTS_DIR = ROOT_DIR / "backtest_reports"
LIVE_TRADES_FILE = ROOT_DIR / "live_trades.json"
MTF_STATE_FILE = ROOT_DIR / "multi_agent" / "logs" / "state.json"

OHLCV_CACHE_TTL_SEC = 20
TIMEFRAME_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def short_symbol(symbol: str) -> str:
    return symbol.replace("/USDT:USDT", "USDT")


def parse_any_timestamp(value: Any) -> Optional[datetime]:
    if value is None:
        return None

    # Milliseconds since epoch
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 10_000_000_000:  # looks like ms
            seconds /= 1000.0
        try:
            return datetime.fromtimestamp(seconds, tz=timezone.utc)
        except (OverflowError, OSError, ValueError):
            return None

    # Numeric string (ms or sec)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        if stripped.isdigit():
            return parse_any_timestamp(int(stripped))
        try:
            return datetime.fromisoformat(stripped.replace("Z", "+00:00"))
        except ValueError:
            return None

    return None


def round_price(price: float) -> float:
    price_abs = abs(price)
    if price_abs >= 1000:
        return round(price, 2)
    if price_abs >= 100:
        return round(price, 3)
    if price_abs >= 1:
        return round(price, 4)
    if price_abs >= 0.01:
        return round(price, 6)
    return round(price, 8)


def default_config_payload() -> dict[str, Any]:
    return {
        "version": "2.6-final-react",
        "updated_at": utc_now_iso(),
        "bot": {},
        "backtest": {},
        "dashboard": {
            "visual": {
                "threshold": 17.2,
                "min_gap_percent": 0.6,
                "risk_percent": 2.0,
                "bullish_color": "#22c55e",
                "bearish_color": "#ef4444",
                "entry_color": "#facc15",
                "sl_color": "#fb7185",
                "tp_color": "#4ade80",
                "show_zones": True,
                "show_levels": True,
                "show_labels": True,
                "show_mitigated": False,
                "glow_intensity": "high",
            }
        },
    }


def default_state_payload() -> dict[str, Any]:
    return {
        "schema_version": "2.6.0",
        "updated_at": None,
        "bot_running": False,
        "cycle": 0,
        "last_scan_completed_at": None,
        "active_symbols": [],
        "stats": {},
        "settings": {},
        "pending_signals": [],
        "active_positions": [],
        "symbols": {},
        "account": {},
        "_warning": "Waiting for fvg_zones.json from bot",
    }


@dataclass
class CachedOHLCV:
    loaded_at: float
    rows: list[dict[str, Any]]


class DataStore:
    """
    Shared backend data service with lightweight caching and graceful recovery.
    """

    def __init__(
        self,
        state_file: Path = STATE_FILE,
        config_file: Path = CONFIG_FILE,
        log_file: Path = LOG_FILE,
        reports_dir: Path = BACKTEST_REPORTS_DIR,
    ) -> None:
        self.state_file = state_file
        self.config_file = config_file
        self.log_file = log_file
        self.reports_dir = reports_dir
        self._lock = Lock()
        self._state_mtime: Optional[float] = None
        self._state_cache: dict[str, Any] = default_state_payload()
        self._config_mtime: Optional[float] = None
        self._config_cache: dict[str, Any] = default_config_payload()
        self._mtf_state_mtime: Optional[float] = None
        self._mtf_state_cache: dict[str, Any] = {}
        self._ohlcv_cache: dict[tuple[str, str, int], CachedOHLCV] = {}
        self._exchange: Optional[ccxt.Exchange] = None

    def _get_exchange(self) -> ccxt.Exchange:
        with self._lock:
            if self._exchange is None:
                exchange = ccxt.bybit(
                    {
                        "enableRateLimit": True,
                        "timeout": 10_000,
                        "options": {"defaultType": "swap"},
                    }
                )
                exchange.load_markets()
                self._exchange = exchange
            return self._exchange

    def _read_json_cached(
        self,
        path: Path,
        cache_attr: str,
        mtime_attr: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        if not path.exists():
            return json.loads(json.dumps(fallback))

        try:
            mtime = path.stat().st_mtime
        except OSError:
            return json.loads(json.dumps(fallback))

        with self._lock:
            cached_mtime = getattr(self, mtime_attr)
            if cached_mtime == mtime:
                return json.loads(json.dumps(getattr(self, cache_attr)))

            try:
                with path.open("r", encoding="utf-8-sig") as handle:
                    payload = json.load(handle)
            except Exception:
                return json.loads(json.dumps(getattr(self, cache_attr)))

            setattr(self, cache_attr, payload)
            setattr(self, mtime_attr, mtime)
            return json.loads(json.dumps(payload))

    def load_state(self) -> dict[str, Any]:
        payload = self._read_json_cached(
            path=self.state_file,
            cache_attr="_state_cache",
            mtime_attr="_state_mtime",
            fallback=default_state_payload(),
        )
        state = default_state_payload()
        if isinstance(payload, dict):
            state.update(payload)
        if "symbols" not in state or not isinstance(state["symbols"], dict):
            state["symbols"] = {}
        if "pending_signals" not in state or not isinstance(state["pending_signals"], list):
            state["pending_signals"] = []
        if "active_positions" not in state or not isinstance(state["active_positions"], list):
            state["active_positions"] = []
        if "active_symbols" not in state or not isinstance(state["active_symbols"], list):
            state["active_symbols"] = []
        state["_warning"] = payload.get("_warning") if isinstance(payload, dict) else None
        return state

    def load_config(self) -> dict[str, Any]:
        config = self._read_json_cached(
            path=self.config_file,
            cache_attr="_config_cache",
            mtime_attr="_config_mtime",
            fallback=default_config_payload(),
        )
        default = default_config_payload()
        merged = default | config
        merged["bot"] = dict(default["bot"]) | dict(merged.get("bot", {}))
        merged["backtest"] = dict(default["backtest"]) | dict(merged.get("backtest", {}))
        dashboard_section = dict(default["dashboard"]) | dict(merged.get("dashboard", {}))
        dashboard_section["visual"] = dict(default["dashboard"]["visual"]) | dict(
            dashboard_section.get("visual", {})
        )
        merged["dashboard"] = dashboard_section
        return merged

    def save_config(
        self,
        bot_updates: Optional[dict[str, Any]] = None,
        visual_updates: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        config = self.load_config()
        if bot_updates:
            config["bot"] = dict(config.get("bot", {})) | dict(bot_updates)
        if visual_updates:
            dashboard_section = dict(config.get("dashboard", {}))
            visual = dict(dashboard_section.get("visual", {})) | dict(visual_updates)
            dashboard_section["visual"] = visual
            config["dashboard"] = dashboard_section

        config["updated_at"] = utc_now_iso()
        tmp_file = self.config_file.with_suffix(".json.tmp")
        with tmp_file.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, ensure_ascii=False, indent=2)
        os.replace(tmp_file, self.config_file)

        with self._lock:
            self._config_cache = config
            try:
                self._config_mtime = self.config_file.stat().st_mtime
            except OSError:
                self._config_mtime = None
        return config

    def get_symbols(
        self,
        query: str = "",
        min_score: float = 0.0,
        min_gap: float = 0.0,
        include_mitigated: bool = False,
    ) -> list[dict[str, Any]]:
        state = self.load_state()
        q = query.strip().lower()
        rows: list[dict[str, Any]] = []
        symbols_map: dict[str, dict[str, Any]] = state.get("symbols", {}) or {}
        active_symbols = list(state.get("active_symbols", []) or [])
        universe = list(dict.fromkeys(active_symbols + list(symbols_map.keys())))

        for symbol in universe:
            snapshot = symbols_map.get(symbol, {})
            zones = snapshot.get("zones", []) or []
            filtered = []
            for zone in zones:
                if not include_mitigated and zone.get("mitigated"):
                    continue
                score = safe_float(zone.get("score"))
                gap = safe_float(zone.get("gap_pct"))
                if score < min_score or gap < min_gap:
                    continue
                filtered.append(zone)

            row = {
                "symbol": symbol,
                "short_symbol": snapshot.get("short_symbol", short_symbol(symbol)),
                "timeframe": snapshot.get("timeframe", state.get("settings", {}).get("timeframe", "15m")),
                "last_price": safe_float(snapshot.get("last_price"), 0.0),
                "htf_bias": snapshot.get("htf_bias", "neutral"),
                "zones_count": len(filtered),
                "top_score": max((safe_float(zone.get("score")) for zone in filtered), default=0.0),
                "top_gap_pct": max((safe_float(zone.get("gap_pct")) for zone in filtered), default=0.0),
                "updated_at": state.get("updated_at"),
            }
            if q and (q not in symbol.lower()) and (q not in row["short_symbol"].lower()):
                continue
            rows.append(row)

        rows.sort(key=lambda item: (item["top_score"], item["zones_count"]), reverse=True)
        return rows

    def get_zones(
        self,
        symbol: str,
        include_mitigated: bool = False,
        min_score: float = 0.0,
        min_gap: float = 0.0,
        min_rr: float = 0.0,
    ) -> list[dict[str, Any]]:
        state = self.load_state()
        snapshot = state.get("symbols", {}).get(symbol, {})
        zones = snapshot.get("zones", []) or []
        output = []
        for zone in zones:
            if not include_mitigated and zone.get("mitigated"):
                continue
            score = safe_float(zone.get("score"))
            gap_pct = safe_float(zone.get("gap_pct"))
            rr = safe_float(zone.get("risk_reward"))
            if score < min_score or gap_pct < min_gap:
                continue
            if min_rr > 0 and rr < min_rr:
                continue
            output.append(zone)
        output.sort(key=lambda z: (safe_float(z.get("score")), safe_float(z.get("gap_pct"))), reverse=True)
        return output

    def get_pending_signals(self, limit: int = 120) -> list[dict[str, Any]]:
        state = self.load_state()
        pending = list(state.get("pending_signals", []))
        if not pending:
            # v2.6 fallback: expose potential setups from active zones when bot has no explicit pending list.
            generated: list[dict[str, Any]] = []
            for symbol, snapshot in (state.get("symbols", {}) or {}).items():
                last_price = safe_float(snapshot.get("last_price"))
                if last_price <= 0:
                    continue
                for zone in snapshot.get("zones", []) or []:
                    if zone.get("mitigated"):
                        continue
                    low = safe_float(zone.get("low"))
                    high = safe_float(zone.get("high"))
                    mid = safe_float(zone.get("entry_price"), safe_float(zone.get("mid_price"), (low + high) / 2.0))
                    if low <= 0 or high <= 0 or mid <= 0:
                        continue
                    distance_pct = abs(last_price - mid) / max(last_price, 1e-9) * 100.0
                    generated.append(
                        {
                            "rank": 0,
                            "symbol": symbol,
                            "short_symbol": snapshot.get("short_symbol", short_symbol(symbol)),
                            "direction": zone.get("direction", ""),
                            "signal_side": "LONG" if zone.get("direction") == "bullish" else "SHORT",
                            "entry_price": mid,
                            "sl_price": safe_float(zone.get("sl_price")),
                            "tp_price": safe_float(zone.get("tp_price")),
                            "risk_reward": safe_float(zone.get("risk_reward")),
                            "position_pct": safe_float(zone.get("position_pct")),
                            "score": safe_float(zone.get("score")),
                            "quality_score": safe_float(zone.get("quality_score")),
                            "reaction_score": safe_float(zone.get("reaction_score")),
                            "gap_pct": safe_float(zone.get("gap_pct")),
                            "zone_low": low,
                            "zone_high": high,
                            "timestamp": zone.get("timestamp"),
                            "formed_at": zone.get("formed_at"),
                            "status": "pending",
                            "entry_status": f"watching-mid-zone ({distance_pct:.2f}% away)",
                            "distance_to_entry_pct": distance_pct,
                        }
                    )
            generated.sort(
                key=lambda item: (
                    safe_float(item.get("distance_to_entry_pct"), 999.0),
                    -safe_float(item.get("score")),
                )
            )
            pending = generated

        pending.sort(
            key=lambda item: (
                safe_float(item.get("score")),
                safe_float(item.get("quality_score")),
                safe_float(item.get("gap_pct")),
            ),
            reverse=True,
        )
        return pending[: max(1, limit)]

    def get_active_positions(self) -> list[dict[str, Any]]:
        state = self.load_state()
        positions = list(state.get("active_positions", []))
        for position in positions:
            entry = safe_float(position.get("entry"))
            current = safe_float(position.get("mark_price", position.get("last_price", entry)))
            side = str(position.get("side", "")).upper()
            qty = safe_float(position.get("qty"))
            if entry > 0 and current > 0 and qty > 0:
                if side == "SHORT":
                    pnl = (entry - current) * qty
                    pnl_pct = ((entry - current) / entry) * 100.0
                else:
                    pnl = (current - entry) * qty
                    pnl_pct = ((current - entry) / entry) * 100.0
            else:
                pnl = 0.0
                pnl_pct = 0.0
            position["unrealized_pnl"] = pnl
            position["unrealized_pnl_pct"] = pnl_pct
        return positions

    def fetch_candles(self, symbol: str, timeframe: str = "15m", limit: int = 220) -> list[dict[str, Any]]:
        key = (symbol, timeframe, limit)
        now = time.time()
        with self._lock:
            cached = self._ohlcv_cache.get(key)
            if cached and (now - cached.loaded_at) <= OHLCV_CACHE_TTL_SEC:
                return list(cached.rows)

        exchange = self._get_exchange()
        tf_ms = TIMEFRAME_MS.get(timeframe, 900_000)
        target = max(120, min(int(limit), 10000))
        per_request_cap = 200
        end_ms = int(time.time() * 1000)
        since = max(0, end_ms - target * tf_ms)

        raw: list[list[float]] = []
        remaining = target
        cursor = since
        max_rounds = max(10, int(target / per_request_cap) + 6)
        for _ in range(max_rounds):
            if remaining <= 0:
                break
            request_limit = min(per_request_cap, remaining)
            chunk = exchange.fetch_ohlcv(symbol, timeframe, since=cursor, limit=request_limit)
            if not chunk:
                break
            raw.extend(chunk)
            last_ts = int(chunk[-1][0])
            cursor = last_ts + tf_ms
            if len(chunk) < request_limit:
                break
            remaining -= len(chunk)

        if not raw:
            raw = exchange.fetch_ohlcv(symbol, timeframe, limit=min(target, per_request_cap))

        # Deduplicate and keep only newest requested bars.
        dedup: dict[int, list[float]] = {}
        for row in raw:
            if len(row) < 6:
                continue
            dedup[int(row[0])] = row
        ordered = [dedup[ts] for ts in sorted(dedup.keys())]
        if len(ordered) > target:
            ordered = ordered[-target:]

        rows: list[dict[str, Any]] = []
        for ts, op, hi, lo, cl, vol in ordered:
            rows.append(
                {
                    "timestamp": int(ts),
                    "open": float(op),
                    "high": float(hi),
                    "low": float(lo),
                    "close": float(cl),
                    "volume": float(vol),
                }
            )
        with self._lock:
            self._ohlcv_cache[key] = CachedOHLCV(loaded_at=now, rows=rows)
        return rows

    def get_latest_backtest_dir(self) -> Optional[Path]:
        if not self.reports_dir.exists():
            return None
        dirs = [
            path
            for path in self.reports_dir.iterdir()
            if path.is_dir() and path.name.startswith("backtest_")
        ]
        if not dirs:
            return None
        dirs.sort(key=lambda item: item.stat().st_mtime, reverse=True)
        return dirs[0]

    def _read_trades_from_latest_report(self) -> list[dict[str, Any]]:
        latest = self.get_latest_backtest_dir()
        if latest is None:
            return []
        trades_csv = latest / "trades.csv"
        if not trades_csv.exists():
            return []

        trades: list[dict[str, Any]] = []
        with trades_csv.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                exit_time = parse_any_timestamp(row.get("exit_time"))
                entry_time = parse_any_timestamp(row.get("entry_time"))
                duration_min = 0.0
                if entry_time and exit_time:
                    duration_min = max((exit_time - entry_time).total_seconds() / 60.0, 0.0)
                trades.append(
                    {
                        "trade_id": int(safe_float(row.get("trade_id"), 0)),
                        "symbol": row.get("symbol", ""),
                        "short_symbol": short_symbol(row.get("symbol", "")),
                        "direction": row.get("direction", ""),
                        "entry_time": entry_time.isoformat() if entry_time else None,
                        "exit_time": exit_time.isoformat() if exit_time else None,
                        "entry_price": safe_float(row.get("entry_price")),
                        "exit_price": safe_float(row.get("exit_price")),
                        "sl": safe_float(row.get("sl")),
                        "tp": safe_float(row.get("tp")),
                        "qty": safe_float(row.get("qty")),
                        "rr": safe_float(row.get("rr")),
                        "gap_pct": safe_float(row.get("gap_pct")),
                        "score": safe_float(row.get("score")),
                        "quality_score": safe_float(row.get("quality_score")),
                        "reaction_score": safe_float(row.get("reaction_score")),
                        "composite_score": safe_float(row.get("composite_score")),
                        "pnl_usd": safe_float(row.get("pnl_usd")),
                        "balance_after": safe_float(row.get("balance_after")),
                        "fee_entry": safe_float(row.get("fee_entry")),
                        "fee_exit": safe_float(row.get("fee_exit")),
                        "funding_cost": safe_float(row.get("funding_cost")),
                        "exit_reason": row.get("exit_reason", ""),
                        "duration_min": duration_min,
                        "date": row.get("date", ""),
                    }
                )
        trades.sort(key=lambda item: item.get("trade_id", 0))
        return trades

    def _read_live_trades(self) -> list[dict[str, Any]]:
        """Read live trades from live_trades.json written by fvg_bot.py."""
        if not LIVE_TRADES_FILE.exists():
            return []
        try:
            with LIVE_TRADES_FILE.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if not isinstance(data, list):
                return []
            # Normalize to same schema as backtest trades
            trades = []
            for i, t in enumerate(data):
                exit_time = parse_any_timestamp(t.get("exit_time"))
                entry_time = parse_any_timestamp(t.get("entry_time"))
                duration_min = 0.0
                if entry_time and exit_time:
                    duration_min = max((exit_time - entry_time).total_seconds() / 60.0, 0.0)
                trades.append({
                    "trade_id": int(safe_float(t.get("trade_id", i + 1))),
                    "symbol": t.get("symbol", ""),
                    "short_symbol": t.get("short_symbol", short_symbol(t.get("symbol", ""))),
                    "direction": t.get("direction", ""),
                    "entry_time": entry_time.isoformat() if entry_time else None,
                    "exit_time": exit_time.isoformat() if exit_time else None,
                    "entry_price": safe_float(t.get("entry_price")),
                    "exit_price": safe_float(t.get("exit_price")),
                    "sl": safe_float(t.get("sl")),
                    "tp": safe_float(t.get("tp")),
                    "qty": safe_float(t.get("qty")),
                    "rr": safe_float(t.get("rr")),
                    "gap_pct": safe_float(t.get("gap_pct")),
                    "score": safe_float(t.get("score")),
                    "quality_score": safe_float(t.get("quality_score")),
                    "reaction_score": safe_float(t.get("reaction_score")),
                    "composite_score": safe_float(t.get("composite_score")),
                    "pnl_usd": safe_float(t.get("pnl_usd")),
                    "balance_after": safe_float(t.get("balance_after")) if t.get("balance_after") else None,
                    "fee_entry": safe_float(t.get("fee_entry")),
                    "fee_exit": safe_float(t.get("fee_exit")),
                    "funding_cost": safe_float(t.get("funding_cost")),
                    "exit_reason": t.get("exit_reason", ""),
                    "duration_min": duration_min,
                    "date": t.get("date", ""),
                })
            trades.sort(key=lambda item: item.get("trade_id", 0))
            return trades
        except Exception:
            return []

    def get_trade_history(self, limit: int = 300) -> list[dict[str, Any]]:
        # MTF multi-agent trades take priority (state.json trade_log)
        mtf_trades = self.get_mtf_trade_history(limit=limit)
        if mtf_trades:
            return mtf_trades
        # Fallback: live_trades.json (old FVG bot)
        live = self._read_live_trades()
        if live:
            return live[-max(1, limit):]
        # Fallback: backtest CSV report
        trades = self._read_trades_from_latest_report()
        if not trades:
            return []
        return trades[-max(1, limit):]

    def get_pnl_series(self) -> list[dict[str, Any]]:
        latest = self.get_latest_backtest_dir()
        if latest is None:
            return []
        curve_file = latest / "equity_curve.json"
        if not curve_file.exists():
            return []
        try:
            with curve_file.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return payload
            return []
        except Exception:
            return []

    def _calc_drawdown_from_curve(self, pnl_series: list[dict[str, Any]]) -> float:
        peak = None
        max_dd = 0.0
        for point in pnl_series:
            equity = safe_float(point.get("balance", point.get("equity")))
            if equity <= 0:
                continue
            if peak is None or equity > peak:
                peak = equity
            if peak and peak > 0:
                dd = (peak - equity) / peak * 100.0
                if dd > max_dd:
                    max_dd = dd
        return max_dd

    def get_overview(self) -> dict[str, Any]:
        state = self.load_state()
        stats = state.get("stats", {}) or {}
        account = state.get("account", {}) or {}
        live_balance = account.get("equity")
        live_available = account.get("available")
        wins = int(safe_float(stats.get("daily_wins"), 0))
        losses = int(safe_float(stats.get("daily_losses"), 0))
        total = wins + losses
        win_rate = (wins / total * 100.0) if total > 0 else 0.0

        # Use only live state values for headline metrics to avoid misleading backtest numbers.
        day_pnl = safe_float(stats.get("daily_pnl"), 0.0)
        week_pnl = None
        month_pnl = None
        start_balance = account.get("session_start_equity")
        total_pnl = None
        if live_balance is not None and start_balance is not None:
            total_pnl = safe_float(live_balance) - safe_float(start_balance)

        return {
            "as_of": utc_now_iso(),
            "balance": live_balance,
            "available_balance": live_available,
            "start_balance": start_balance,
            "total_pnl": total_pnl,
            "balance_source": "live_state",
            "daily_pnl": day_pnl,
            "weekly_pnl": 0.0 if week_pnl is None else week_pnl,
            "monthly_pnl": 0.0 if month_pnl is None else month_pnl,
            "win_rate": win_rate,
            "wins": wins,
            "losses": losses,
            "active_trades": len(state.get("active_positions", [])),
            "pending_signals": len(self.get_pending_signals(limit=5000)),
            "bot_running": bool(state.get("bot_running", False)),
            "tracked_symbols": int(safe_float(state.get("stats", {}).get("tracked_symbols"), 0)),
            "active_zones": int(safe_float(state.get("stats", {}).get("active_zones"), 0)),
            "max_drawdown_pct": None,
        }

    def get_analytics(self) -> dict[str, Any]:
        trades = self.get_trade_history(limit=5000)
        pnl_curve = self.get_pnl_series()

        total_pnl = sum(safe_float(trade.get("pnl_usd")) for trade in trades)
        wins = [trade for trade in trades if safe_float(trade.get("pnl_usd")) >= 0]
        losses = [trade for trade in trades if safe_float(trade.get("pnl_usd")) < 0]
        gross_profit = sum(safe_float(trade.get("pnl_usd")) for trade in wins)
        gross_loss = abs(sum(safe_float(trade.get("pnl_usd")) for trade in losses))
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else 0.0
        avg_rr = (
            sum(safe_float(trade.get("rr")) for trade in trades) / len(trades)
            if trades
            else 0.0
        )
        avg_duration = (
            sum(safe_float(trade.get("duration_min")) for trade in trades) / len(trades)
            if trades
            else 0.0
        )
        max_drawdown = self._calc_drawdown_from_curve(pnl_curve)

        symbol_pnl: dict[str, float] = {}
        heatmap: dict[str, float] = {}
        for trade in trades:
            symbol = trade.get("short_symbol", trade.get("symbol", ""))
            pnl = safe_float(trade.get("pnl_usd"))
            symbol_pnl[symbol] = symbol_pnl.get(symbol, 0.0) + pnl
            date_key = str(trade.get("date") or "")[:10]
            if date_key:
                heatmap[date_key] = heatmap.get(date_key, 0.0) + pnl

        per_symbol = [
            {"symbol": key, "pnl_usd": value}
            for key, value in symbol_pnl.items()
        ]
        per_symbol.sort(key=lambda item: item["pnl_usd"], reverse=True)

        heatmap_rows = [{"date": key, "pnl_usd": value} for key, value in sorted(heatmap.items())]

        return {
            "trades_count": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades) * 100.0) if trades else 0.0,
            "profit_factor": profit_factor,
            "avg_rr": avg_rr,
            "avg_duration_min": avg_duration,
            "total_pnl": total_pnl,
            "max_drawdown_pct": max_drawdown,
            "equity_curve": pnl_curve,
            "symbol_pnl": per_symbol[:30],
            "heatmap": heatmap_rows,
        }

    # ── Multi-agent (MTF) state helpers ────────────────────────────────────

    def load_mtf_state(self) -> dict[str, Any]:
        """Load multi-agent runtime state from multi_agent/logs/state.json."""
        return self._read_json_cached(
            path=MTF_STATE_FILE,
            cache_attr="_mtf_state_cache",
            mtime_attr="_mtf_state_mtime",
            fallback={},
        )

    def get_mtf_trade_history(
        self, limit: int = 300, strategy: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Extract and normalise closed trades from multi_agent/logs/state.json
        trade_log section.  Returns an empty list if no trades exist yet.
        """
        mtf = self.load_mtf_state()
        trade_log = mtf.get("trade_log", {})
        all_trades: list[dict[str, Any]] = []
        idx = 0
        for agent_id, trades in trade_log.items():
            if not isinstance(trades, list):
                continue
            for t in trades:
                idx += 1
                raw_entry_time = t.get("opened_at") or t.get("entry_time")
                raw_exit_time  = t.get("closed_at") or t.get("exit_time")
                entry_dt = parse_any_timestamp(raw_entry_time)
                exit_dt  = parse_any_timestamp(raw_exit_time)
                duration = 0.0
                if entry_dt and exit_dt:
                    duration = max((exit_dt - entry_dt).total_seconds() / 60.0, 0.0)
                all_trades.append({
                    "trade_id":    idx,
                    "symbol":      t.get("symbol", ""),
                    "short_symbol": short_symbol(t.get("symbol", "")),
                    "strategy":    t.get("strategy_id", agent_id),
                    "strategy_id": t.get("strategy_id", agent_id),
                    "direction":   t.get("direction", t.get("side", "")),
                    "side":        t.get("side", ""),
                    "entry_time":  entry_dt.isoformat() if entry_dt else raw_entry_time,
                    "exit_time":   exit_dt.isoformat() if exit_dt else raw_exit_time,
                    "entry_price": safe_float(t.get("entry")),
                    "exit_price":  safe_float(t.get("exit")),
                    "sl":          safe_float(t.get("sl")),
                    "tp":          safe_float(t.get("tp")),
                    "qty":         safe_float(t.get("qty")),
                    "pnl_usd":     safe_float(t.get("pnl_usd")),
                    "pnl_pct":     safe_float(t.get("pnl_pct")),
                    "exit_reason": t.get("reason", ""),
                    "reason":      t.get("reason", ""),
                    "date":        t.get("date", ""),
                    "rr":          safe_float(t.get("rr")),
                    "duration_min": duration,
                })
        if strategy:
            all_trades = [t for t in all_trades if t.get("strategy", "") == strategy]
        all_trades.sort(key=lambda x: x.get("exit_time") or "")
        return all_trades[-max(1, limit):]

    def get_mtf_positions(self) -> list[dict[str, Any]]:
        """Return currently open positions from multi_agent/logs/state.json."""
        mtf = self.load_mtf_state()
        positions_by_agent = mtf.get("positions", {})
        result: list[dict[str, Any]] = []
        for agent_id, agent_pos in positions_by_agent.items():
            if not isinstance(agent_pos, dict):
                continue
            for symbol, pos in agent_pos.items():
                if not isinstance(pos, dict):
                    continue
                entry = safe_float(pos.get("entry"))
                result.append({
                    "symbol":      symbol,
                    "short_symbol": short_symbol(symbol),
                    "side":        pos.get("side", "LONG"),
                    "direction":   pos.get("side", "LONG"),
                    "strategy":    pos.get("strategy_id", agent_id),
                    "strategy_id": pos.get("strategy_id", "MTF"),
                    "entry":       entry,
                    "entry_price": entry,
                    "sl":          safe_float(pos.get("sl")),
                    "tp":          safe_float(pos.get("tp")),
                    "qty":         safe_float(pos.get("qty")),
                    "leverage":    pos.get("leverage", 1),
                    "opened_at":   pos.get("opened_at"),
                    "agent_id":    agent_id,
                })
        return result

    def get_mtf_balance(self) -> float:
        """Return current total equity from multi_agent/logs/state.json."""
        mtf = self.load_mtf_state()
        equities = mtf.get("equities", {})
        return sum(safe_float(v) for v in equities.values()) if equities else 0.0

    def get_mtf_starting_balance(self) -> float:
        """Return starting equity from multi_agent/logs/state.json."""
        mtf = self.load_mtf_state()
        starting = mtf.get("starting_equities", {})
        return sum(safe_float(v) for v in starting.values()) if starting else 1000.0

    def tail_logs(self, lines: int = 250) -> list[str]:
        if not self.log_file.exists():
            return ["[INFO] log file is not available yet."]
        buffer: list[str] = []
        try:
            with self.log_file.open("r", encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    buffer.append(line.rstrip("\n"))
            return buffer[-max(1, lines) :]
        except Exception as exc:
            return [f"[ERROR] cannot read log file: {exc}"]

    def export_zones_csv(
        self,
        include_mitigated: bool = False,
        min_score: float = 0.0,
        min_gap: float = 0.0,
    ) -> str:
        state = self.load_state()
        columns = [
            "symbol",
            "direction",
            "status",
            "gap_pct",
            "score",
            "quality_score",
            "reaction_score",
            "low",
            "high",
            "mid_price",
            "entry_price",
            "sl_price",
            "tp_price",
            "risk_reward",
            "formed_at",
        ]
        rows: list[dict[str, Any]] = []
        for symbol, snapshot in state.get("symbols", {}).items():
            for zone in snapshot.get("zones", []) or []:
                if not include_mitigated and zone.get("mitigated"):
                    continue
                if safe_float(zone.get("score")) < min_score:
                    continue
                if safe_float(zone.get("gap_pct")) < min_gap:
                    continue
                rows.append(
                    {
                        "symbol": symbol,
                        "direction": zone.get("direction", ""),
                        "status": zone.get("status", ""),
                        "gap_pct": round(safe_float(zone.get("gap_pct")), 4),
                        "score": round(safe_float(zone.get("score")), 4),
                        "quality_score": round(safe_float(zone.get("quality_score")), 4),
                        "reaction_score": round(safe_float(zone.get("reaction_score")), 4),
                        "low": round_price(safe_float(zone.get("low"))),
                        "high": round_price(safe_float(zone.get("high"))),
                        "mid_price": round_price(safe_float(zone.get("mid_price"))),
                        "entry_price": round_price(safe_float(zone.get("entry_price"))),
                        "sl_price": round_price(safe_float(zone.get("sl_price"))),
                        "tp_price": round_price(safe_float(zone.get("tp_price"))),
                        "risk_reward": round(safe_float(zone.get("risk_reward")), 4),
                        "formed_at": zone.get("formed_at", ""),
                    }
                )

        out = []
        out.append(",".join(columns))
        for row in rows:
            out.append(",".join(str(row.get(col, "")) for col in columns))
        return "\n".join(out) + "\n"

    def export_zones_pine(
        self,
        include_mitigated: bool = False,
        min_score: float = 0.0,
        min_gap: float = 0.0,
    ) -> str:
        state = self.load_state()
        lines = [
            "//@version=5",
            'indicator("FVG Zones Export v2.6", overlay=true, max_boxes_count=500, max_lines_count=500)',
            "",
        ]
        idx = 0
        for symbol, snapshot in state.get("symbols", {}).items():
            for zone in snapshot.get("zones", []) or []:
                if not include_mitigated and zone.get("mitigated"):
                    continue
                if safe_float(zone.get("score")) < min_score:
                    continue
                if safe_float(zone.get("gap_pct")) < min_gap:
                    continue
                low = round_price(safe_float(zone.get("low")))
                high = round_price(safe_float(zone.get("high")))
                entry = round_price(safe_float(zone.get("entry_price", zone.get("mid_price"))))
                color_name = "color.new(color.green, 82)" if zone.get("direction") == "bullish" else "color.new(color.red, 82)"
                lines.append(f"var box z{idx} = na")
                lines.append(
                    f'if barstate.islast\n    z{idx} := box.new(left=bar_index-50, top={high}, right=bar_index, bottom={low}, bgcolor={color_name}, border_color=color.new(color.white, 85))'
                )
                lines.append(
                    f'plot({entry}, title="{short_symbol(symbol)}_{idx}_entry", color=color.new(color.yellow, 0), linewidth=1, style=plot.style_linebr)'
                )
                lines.append("")
                idx += 1

        if idx == 0:
            lines.append("// no zones passed selected filters")
        return "\n".join(lines) + "\n"


data_store = DataStore()
