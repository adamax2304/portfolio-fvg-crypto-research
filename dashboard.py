#!/usr/bin/env python3
"""
Standalone professional dashboard for FVG Trading Bot v2.6.

Run with:
    python dashboard.py
"""

from __future__ import annotations

import csv
import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from io import StringIO
from threading import Lock
from typing import Any, Optional

import pandas as pd
import numpy as np

try:
    import ccxt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency 'ccxt'. Install it with: pip install ccxt"
    ) from exc

try:
    import dash_bootstrap_components as dbc
    import plotly.graph_objects as go
    from dash import ALL, Dash, Input, Output, State, ctx, dcc, html, no_update
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dashboard dependencies. Install them with: "
        "pip install dash dash-bootstrap-components plotly"
    ) from exc


APP_TITLE = "FVG Trading Bot v2.6 Dashboard"
# v2.6 — correct SL under real swing low + conservative TP + pre-alert 0.2% away
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATE_FILE = os.path.join(BASE_DIR, "fvg_zones.json")
BOT_LOG_FILE = os.path.join(BASE_DIR, "fvg_bot.log")
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")
POLL_INTERVAL_MS = 30_000
OHLCV_CACHE_TTL_SEC = 25
MAX_VISIBLE_CHARTS = 6
TIMEFRAME_TO_MS = {"5m": 5 * 60 * 1000, "15m": 15 * 60 * 1000}

DEFAULT_SETTINGS = {
    "threshold": 16.5,
    "min_gap_percent": 0.22,
    "risk_percent": 20.0,
    "mid_zone_low": 40.0,
    "mid_zone_high": 60.0,
    "bullish_color": "#22c55e",
    "bearish_color": "#ef4444",
    "entry_color": "#f4d35e",
    "sl_color": "#ff4d4f",
    "tp_color": "#22c55e",
    "show_zones": True,
    "show_levels": True,
    "show_labels": True,
    "show_mitigated": False,
}

DEFAULT_BOT_SETTINGS = {
    "FVG_PERCENTILE_THRESHOLD": 17.2,
    "MIN_GAP_PERCENT": 0.22,
    "FVG_MIN_GAP_MULT": 1.26,
    "FVG_QUALITY_MIN_SCORE": 47.0,
    "FVG_LOOKBACK": 384,
    "FVG_MAX_AGE_DAYS": 4,
    "FVG_PRICE_MODE": "wick",
    "FVG_REQUIRE_DISPLACEMENT": True,
    "FVG_DISPLACEMENT_BODY_MULT": 1.2,
    "FVG_DISPLACEMENT_BODY_PCT": 0.15,
    "FVG_REQUIRE_SAME_DIRECTION": True,
    "FVG_MAX_ACTIVE_PER_SIDE": 3,
    "SWING_LOOKBACK_BARS": 120,
    "RISK_PER_TRADE": 0.20,
    "MAX_CONCURRENT_POSITIONS": 2,
    "TOP_N_COINS": 70,
    "POLL_INTERVAL_SEC": 60,
    "SL_NEAREST_LEVEL_BUFFER_PCT": 0.15,
    "TP_HVN_OFFSET_PCT": 0.12,
    "SL_LEVEL_MIN_OFFSET_PCT": 0.10,
    "SL_LEVEL_BUFFER_MIN_PCT": 0.10,
    "SL_LEVEL_BUFFER_MAX_PCT": 0.20,
    "TP_OFFSET_MIN_PCT": 0.10,
    "TP_OFFSET_MAX_PCT": 0.15,
    "STRUCTURE_TOUCH_TOL_PCT": 0.06,
    "VP_LOOKBACK_BARS": 96,
    "VP_NUM_BINS": 48,
    "MID_ZONE_LOW": 0.40,
    "MID_ZONE_HIGH": 0.60,
    "PRE_ALERT_DISTANCE_PCT": 0.20,
    "ENTRY_CONFIRMATION_MODE": "soft",
    "ENTRY_CONFIRM_MIN_BODY_PCT": 0.05,
    "SIGNAL_MIN_COMPOSITE_SCORE": 50.0,
}

FALLBACK_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "XRP/USDT:USDT",
    "BNB/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "TRX/USDT:USDT", "DOT/USDT:USDT", "LTC/USDT:USDT",
    "SUI/USDT:USDT", "APT/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT",
]


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def age_label(value: Optional[str]) -> str:
    dt_value = parse_iso_datetime(value)
    if dt_value is None:
        return "unknown"
    delta = datetime.now(timezone.utc) - dt_value
    seconds = max(int(delta.total_seconds()), 0)
    if seconds < 60:
        return f"{seconds}s ago"
    if seconds < 3600:
        return f"{seconds // 60}m ago"
    return f"{seconds // 3600}h ago"


def state_is_stale(value: Optional[str], stale_after_sec: int = 150) -> bool:
    dt_value = parse_iso_datetime(value)
    if dt_value is None:
        return True
    return (datetime.now(timezone.utc) - dt_value).total_seconds() > stale_after_sec


def short_symbol(symbol: str) -> str:
    return symbol.replace("/USDT:USDT", "USDT")


def round_price_generic(price: float) -> float:
    abs_price = abs(price)
    if abs_price >= 1000:
        return round(price, 2)
    if abs_price >= 100:
        return round(price, 3)
    if abs_price >= 1:
        return round(price, 4)
    if abs_price >= 0.01:
        return round(price, 6)
    return round(price, 8)


def current_bot_settings() -> dict[str, Any]:
    config = provider.load_config()
    return dict(DEFAULT_BOT_SETTINGS) | dict(config.get("bot", {}))


def dashboard_symbol_pool(state: Optional[dict[str, Any]]) -> list[str]:
    if not isinstance(state, dict):
        return list(FALLBACK_SYMBOLS)
    symbols = list(state.get("active_symbols") or list(state.get("symbols", {}).keys()) or [])
    if symbols:
        return symbols
    top_n = max(1, int(safe_float(current_bot_settings().get("TOP_N_COINS"), len(FALLBACK_SYMBOLS))))
    return list(FALLBACK_SYMBOLS[: min(len(FALLBACK_SYMBOLS), top_n)])


def empty_state(message: str = "") -> dict[str, Any]:
    return {
        "schema_version": "2.6.0",
        "updated_at": None,
        "bot_running": False,
        "cycle": 0,
        "last_scan_completed_at": None,
        "active_symbols": [],
        "settings": {
            "threshold": DEFAULT_SETTINGS["threshold"],
            "min_gap_percent": DEFAULT_SETTINGS["min_gap_percent"],
            "risk_percent": DEFAULT_SETTINGS["risk_percent"],
            "mid_zone_low": DEFAULT_SETTINGS["mid_zone_low"],
            "mid_zone_high": DEFAULT_SETTINGS["mid_zone_high"],
            "timeframe": "15m",
            "colors": {
                "bullish": "rgba(34, 197, 94, 0.22)",
                "bearish": "rgba(239, 68, 68, 0.22)",
                "entry": DEFAULT_SETTINGS["entry_color"],
                "sl": DEFAULT_SETTINGS["sl_color"],
                "tp": DEFAULT_SETTINGS["tp_color"],
            },
        },
        "stats": {
            "tracked_symbols": 0,
            "zones_total": 0,
            "active_zones": 0,
            "mitigated_zones": 0,
            "average_active_score": 0.0,
            "pending_signals": 0,
            "open_positions": 0,
            "daily_trades": 0,
            "daily_wins": 0,
            "daily_losses": 0,
            "daily_pnl": 0.0,
        },
        "pending_signals": [],
        "active_positions": [],
        "symbols": {},
        "_warning": message,
    }


def normalize_state(raw_state: Optional[dict[str, Any]]) -> dict[str, Any]:
    state = empty_state()
    if not isinstance(raw_state, dict):
        return state

    state.update({
        "schema_version": raw_state.get("schema_version", state["schema_version"]),
        "updated_at": raw_state.get("updated_at"),
        "bot_running": bool(raw_state.get("bot_running", False)),
        "cycle": int(raw_state.get("cycle", 0) or 0),
        "last_scan_completed_at": raw_state.get("last_scan_completed_at"),
        "active_symbols": list(raw_state.get("active_symbols", [])),
        "pending_signals": list(raw_state.get("pending_signals", [])),
        "active_positions": list(raw_state.get("active_positions", [])),
        "symbols": dict(raw_state.get("symbols", {})),
    })

    settings = dict(state["settings"])
    settings.update(raw_state.get("settings", {}))
    settings["colors"] = dict(state["settings"]["colors"]) | dict(settings.get("colors", {}))
    state["settings"] = settings

    stats = dict(state["stats"])
    stats.update(raw_state.get("stats", {}))
    state["stats"] = stats
    return state


def default_config_payload() -> dict[str, Any]:
    return {
        "version": "2.6.0",
        "updated_at": None,
        "bot": dict(DEFAULT_BOT_SETTINGS),
        "dashboard": {
            "visual": {
                "threshold": DEFAULT_SETTINGS["threshold"],
                "min_gap_percent": DEFAULT_SETTINGS["min_gap_percent"],
                "risk_percent": DEFAULT_SETTINGS["risk_percent"],
                "bullish_color": DEFAULT_SETTINGS["bullish_color"],
                "bearish_color": DEFAULT_SETTINGS["bearish_color"],
                "show_zones": DEFAULT_SETTINGS["show_zones"],
                "show_levels": DEFAULT_SETTINGS["show_levels"],
                "show_labels": DEFAULT_SETTINGS["show_labels"],
                "show_mitigated": DEFAULT_SETTINGS["show_mitigated"],
            }
        },
    }


class DashboardDataProvider:
    """Caches shared JSON state and live OHLCV to keep the dashboard responsive."""

    def __init__(self, state_file: str, log_file: str, config_file: str):
        self.state_file = state_file
        self.log_file = log_file
        self.config_file = config_file
        self._lock = Lock()
        self._state_mtime: Optional[float] = None
        self._state_cache: dict[str, Any] = empty_state()
        self._config_mtime: Optional[float] = None
        self._config_cache: dict[str, Any] = default_config_payload()
        self._exchange: Optional[ccxt.Exchange] = None
        self._ohlcv_cache: dict[tuple[str, str, int], dict[str, Any]] = {}

    def clear_ohlcv_cache(self) -> None:
        with self._lock:
            self._ohlcv_cache.clear()

    def get_exchange(self) -> ccxt.Exchange:
        with self._lock:
            if self._exchange is None:
                exchange = ccxt.bybit({
                    "enableRateLimit": True,
                    "timeout": 10_000,
                    "options": {"defaultType": "swap"},
                })
                exchange.load_markets()
                self._exchange = exchange
            return self._exchange

    def load_state(self) -> tuple[dict[str, Any], str, Optional[str]]:
        if not os.path.exists(self.state_file):
            warning = "fvg_zones.json not found yet. The dashboard is waiting for the bot state file."
            return empty_state(warning), "Waiting for shared state", warning

        try:
            mtime = os.path.getmtime(self.state_file)
        except OSError as exc:
            warning = f"Unable to read state file metadata: {exc}"
            return normalize_state(self._state_cache), "State metadata error", warning

        with self._lock:
            if self._state_mtime == mtime:
                state = normalize_state(self._state_cache)
            else:
                try:
                    with open(self.state_file, "r", encoding="utf-8") as handle:
                        loaded = json.load(handle)
                    state = normalize_state(loaded)
                    self._state_cache = state
                    self._state_mtime = mtime
                except json.JSONDecodeError as exc:
                    cached = normalize_state(self._state_cache)
                    warning = (
                        f"State file is temporarily corrupted: {exc}. "
                        "Showing the last valid snapshot instead."
                    )
                    cached["_warning"] = warning
                    return cached, "Using cached state", warning
                except Exception as exc:
                    cached = normalize_state(self._state_cache)
                    warning = f"Failed to load state file: {exc}"
                    cached["_warning"] = warning
                    return cached, "State load error", warning

        warning = state.get("_warning")
        status = "Live state loaded"
        if state_is_stale(state.get("updated_at")):
            stale_message = (
                f"Bot state looks stale ({age_label(state.get('updated_at'))}). "
                "Charts still use the last saved zones."
            )
            warning = stale_message if not warning else f"{warning} {stale_message}"
            status = "Showing last saved snapshot"
        return state, status, warning

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 220,
        retries: int = 2,
    ) -> tuple[pd.DataFrame, Optional[str]]:
        cache_key = (symbol, timeframe, limit)
        now = time.time()

        with self._lock:
            cached = self._ohlcv_cache.get(cache_key)
            if cached and (now - cached["loaded_at"]) <= OHLCV_CACHE_TTL_SEC:
                return cached["data"].copy(), None

        last_error: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                exchange = self.get_exchange()
                ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                if df.empty:
                    raise ValueError(f"No OHLCV data for {symbol} {timeframe}")
                df["timestamp"] = df["timestamp"].astype(int)
                with self._lock:
                    self._ohlcv_cache[cache_key] = {"loaded_at": now, "data": df.copy()}
                return df, None
            except Exception as exc:  # pragma: no cover
                last_error = exc
                time.sleep(0.6 * (attempt + 1))

        with self._lock:
            cached = self._ohlcv_cache.get(cache_key)
            if cached:
                warning = (
                    f"Using cached candles for {short_symbol(symbol)} {timeframe}: {last_error}"
                )
                return cached["data"].copy(), warning

        warning = f"Unable to fetch candles for {short_symbol(symbol)} {timeframe}: {last_error}"
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"]), warning

    def tail_log(self, line_count: int = 140) -> str:
        if not os.path.exists(self.log_file):
            return "Log file not found yet."
        try:
            with open(self.log_file, "r", encoding="utf-8", errors="replace") as handle:
                return "".join(deque(handle, maxlen=line_count))
        except Exception as exc:
            return f"Unable to read log file: {exc}"

    def load_config(self) -> dict[str, Any]:
        default_payload = default_config_payload()
        if not os.path.exists(self.config_file):
            self.save_config(default_payload)
            return default_payload

        try:
            mtime = os.path.getmtime(self.config_file)
        except OSError:
            return default_payload

        with self._lock:
            if self._config_mtime == mtime:
                return json.loads(json.dumps(self._config_cache))

            try:
                with open(self.config_file, "r", encoding="utf-8-sig") as handle:
                    loaded = json.load(handle)
            except Exception:
                return json.loads(json.dumps(self._config_cache))

            config = default_payload | dict(loaded)
            config["bot"] = dict(default_payload["bot"]) | dict(config.get("bot", {}))
            dashboard_section = dict(default_payload["dashboard"]) | dict(config.get("dashboard", {}))
            dashboard_section["visual"] = dict(default_payload["dashboard"]["visual"]) | dict(dashboard_section.get("visual", {}))
            config["dashboard"] = dashboard_section
            self._config_cache = config
            self._config_mtime = mtime
            return json.loads(json.dumps(config))

    def save_config(self, config: dict[str, Any]) -> None:
        payload = default_config_payload()
        payload["updated_at"] = datetime.now(timezone.utc).isoformat()
        payload["bot"] = dict(payload["bot"]) | dict(config.get("bot", {}))
        dashboard_section = dict(payload["dashboard"]) | dict(config.get("dashboard", {}))
        dashboard_section["visual"] = dict(payload["dashboard"]["visual"]) | dict(dashboard_section.get("visual", {}))
        payload["dashboard"] = dashboard_section

        temp_file = f"{self.config_file}.tmp"
        with open(temp_file, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
        os.replace(temp_file, self.config_file)

        with self._lock:
            self._config_cache = payload
            try:
                self._config_mtime = os.path.getmtime(self.config_file)
            except OSError:
                self._config_mtime = None


def compute_gap_diffs(highs: list[float], lows: list[float], start: int, end: int) -> list[float]:
    diffs: list[float] = []
    for index in range(start, end):
        mid_price = (highs[index] + lows[index]) / 2.0
        if mid_price <= 0:
            continue
        if highs[index - 2] < lows[index]:
            diffs.append((lows[index] - highs[index - 2]) / mid_price * 100.0)
        if lows[index - 2] > highs[index]:
            diffs.append((lows[index - 2] - highs[index]) / mid_price * 100.0)
    return diffs


def atr_pct_from_arrays(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray) -> float:
    if len(highs) < 2 or len(lows) < 2 or len(closes) < 2:
        return 0.0
    prev_close = closes[:-1]
    highs = highs[1:]
    lows = lows[1:]
    tr = np.maximum(highs - lows, np.maximum(np.abs(highs - prev_close), np.abs(lows - prev_close)))
    atr = float(np.mean(tr))
    last_close = float(closes[-1]) if float(closes[-1]) > 0 else 1.0
    return atr / last_close * 100.0


def dashboard_market_regime(
    highs_raw: np.ndarray,
    lows_raw: np.ndarray,
    closes: np.ndarray,
    index: int,
    bot_settings: dict[str, Any],
) -> tuple[float, dict[str, float]]:
    start = max(0, index - 96)
    close_win = closes[start:index + 1].astype(float)
    high_win = highs_raw[start:index + 1].astype(float)
    low_win = lows_raw[start:index + 1].astype(float)
    if len(close_win) < 20:
        return 0.0, {"atr_pct": 0.0, "trend_pct": 0.0, "chop_pct": 0.0}

    atr_pct = atr_pct_from_arrays(high_win, low_win, close_win)
    close_series = pd.Series(close_win)
    ema_fast = close_series.ewm(span=20, adjust=False).mean().iloc[-1]
    ema_slow = close_series.ewm(span=50, adjust=False).mean().iloc[-1]
    last_close = float(close_win[-1]) if close_win[-1] > 0 else 1.0
    trend_pct = abs(float(ema_fast) - float(ema_slow)) / last_close * 100.0

    diff = np.diff(close_win)
    signs = np.sign(diff)
    direction_changes = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
    chop_pct = direction_changes / max(1, len(signs)) * 100.0

    adjustment = 0.0
    if bool(bot_settings.get("REGIME_ADAPTIVE_THRESHOLD", True)):
        if trend_pct >= max(0.10, atr_pct * 0.45):
            adjustment += safe_float(bot_settings.get("REGIME_TREND_BONUS"), -1.5)
        if chop_pct >= 45.0:
            adjustment += safe_float(bot_settings.get("REGIME_CHOP_PENALTY"), 4.0)
        if atr_pct >= 1.8:
            adjustment += 1.0

    return adjustment, {
        "atr_pct": float(atr_pct),
        "trend_pct": float(trend_pct),
        "chop_pct": float(chop_pct),
    }


def dashboard_fvg_quality_score(
    diff_pct: float,
    size_fvg: float,
    body_pct_prev: float,
    required_body: float,
    regime_info: dict[str, float],
    direction: str,
    opens: np.ndarray,
    closes: np.ndarray,
    index: int,
) -> float:
    displacement_ratio = body_pct_prev / max(required_body, 1e-6)
    displacement_score = min(displacement_ratio * 35.0, 100.0)
    atr_pct = max(regime_info.get("atr_pct", 0.0), 1e-6)
    volatility_score = min((diff_pct / atr_pct) * 40.0, 100.0)
    trend_pct = regime_info.get("trend_pct", 0.0)
    trend_score = min((trend_pct / max(atr_pct, 0.12)) * 45.0, 100.0)

    direction_score = 50.0
    if index >= 2:
        c1_up = closes[index - 1] > opens[index - 1]
        c2_up = closes[index - 2] > opens[index - 2]
        if direction == "bullish":
            direction_score = 100.0 if (c1_up or c2_up) else 25.0
        else:
            direction_score = 100.0 if ((not c1_up) or (not c2_up)) else 25.0

    quality = (
        size_fvg * 0.40 +
        displacement_score * 0.20 +
        volatility_score * 0.20 +
        trend_score * 0.10 +
        direction_score * 0.10
    )
    return float(max(0.0, min(100.0, quality)))


def collect_swing_levels(df: pd.DataFrame, lookback: int = 60) -> tuple[list[float], list[float]]:
    if df.empty or len(df) < 5:
        return [], []

    tail = df.tail(lookback).reset_index(drop=True)
    highs = tail["high"].astype(float).tolist()
    lows = tail["low"].astype(float).tolist()
    swing_lows: list[float] = []
    swing_highs: list[float] = []

    for index in range(2, len(tail) - 2):
        if (
            highs[index] > highs[index - 1]
            and highs[index] > highs[index - 2]
            and highs[index] > highs[index + 1]
            and highs[index] > highs[index + 2]
        ):
            swing_highs.append(float(highs[index]))
        if (
            lows[index] < lows[index - 1]
            and lows[index] < lows[index - 2]
            and lows[index] < lows[index + 1]
            and lows[index] < lows[index + 2]
        ):
            swing_lows.append(float(lows[index]))

    return sorted(set(swing_lows)), sorted(set(swing_highs))


def build_local_volume_profile(df: pd.DataFrame, lookback: int = 96, bins: int = 48) -> dict[str, Any]:
    tail = df.tail(max(10, lookback)).copy()
    if tail.empty:
        return {"levels": [], "poc": 0.0}

    price_low = float(tail["low"].min())
    price_high = float(tail["high"].max())
    if price_high <= price_low:
        level = float(tail["close"].iloc[-1])
        return {"levels": [level], "poc": level}

    bins = max(12, int(bins))
    edges = np.linspace(price_low, price_high, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    histogram = np.zeros(bins, dtype=float)

    for _, candle in tail.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        volume = float(candle.get("volume", 0.0) or 0.0)
        if volume <= 0:
            continue
        if high <= low:
            idx = int(np.clip(np.searchsorted(edges, low, side="right") - 1, 0, bins - 1))
            histogram[idx] += volume
            continue
        overlap = np.minimum(edges[1:], high) - np.maximum(edges[:-1], low)
        overlap = np.clip(overlap, 0.0, None)
        overlap_sum = float(overlap.sum())
        if overlap_sum <= 0:
            idx = int(np.clip(np.searchsorted(edges, (low + high) / 2.0, side="right") - 1, 0, bins - 1))
            histogram[idx] += volume
            continue
        histogram += volume * (overlap / overlap_sum)

    if not np.any(histogram > 0):
        level = float(tail["close"].iloc[-1])
        return {"levels": [level], "poc": level}

    active_values = histogram[histogram > 0]
    hvn_threshold = float(np.percentile(active_values, 70))
    hvn_indices: list[int] = []
    for idx in range(len(histogram)):
        left = histogram[idx - 1] if idx > 0 else histogram[idx]
        right = histogram[idx + 1] if idx < len(histogram) - 1 else histogram[idx]
        if histogram[idx] >= hvn_threshold and histogram[idx] >= left and histogram[idx] >= right:
            hvn_indices.append(idx)

    poc_index = int(np.argmax(histogram))
    if poc_index not in hvn_indices:
        hvn_indices.append(poc_index)

    levels = sorted({float(centers[idx]) for idx in hvn_indices})
    return {
        "levels": levels,
        "poc": float(centers[poc_index]),
        "price_low": price_low,
        "price_high": price_high,
    }


def find_swing_levels(df: pd.DataFrame, lookback: int = 30) -> tuple[float, float]:
    if df.empty or len(df) < 5:
        return 0.0, 0.0

    tail = df.tail(lookback).reset_index(drop=True)
    highs = tail["high"].tolist()
    lows = tail["low"].tolist()
    swing_highs: list[float] = []
    swing_lows: list[float] = []

    for index in range(2, len(tail) - 2):
        if (
            highs[index] > highs[index - 1]
            and highs[index] > highs[index - 2]
            and highs[index] > highs[index + 1]
            and highs[index] > highs[index + 2]
        ):
            swing_highs.append(highs[index])
        if (
            lows[index] < lows[index - 1]
            and lows[index] < lows[index - 2]
            and lows[index] < lows[index + 1]
            and lows[index] < lows[index + 2]
        ):
            swing_lows.append(lows[index])

    current_price = safe_float(tail["close"].iloc[-1])
    above = [value for value in swing_highs if value > current_price]
    below = [value for value in swing_lows if value < current_price]
    return (max(below) if below else min(lows), min(above) if above else max(highs))


def extract_structural_levels(
    df: pd.DataFrame,
    lookback: int = 160,
    tolerance_pct: float = 0.06,
) -> tuple[list[tuple[float, int]], list[tuple[float, int]]]:
    if df is None or df.empty or len(df) < 15:
        return [], []

    tail = df.tail(max(30, lookback)).reset_index(drop=True)
    highs = tail["high"].astype(float).to_numpy()
    lows = tail["low"].astype(float).to_numpy()
    swing_lows, swing_highs = collect_swing_levels(tail, lookback=len(tail))
    raw_supports = list(swing_lows)
    raw_resistances = list(swing_highs)

    for win in (3, 5):
        for i in range(win, len(tail) - win):
            low_i = float(lows[i])
            high_i = float(highs[i])
            if low_i <= float(np.min(lows[i - win:i + win + 1])):
                raw_supports.append(low_i)
            if high_i >= float(np.max(highs[i - win:i + win + 1])):
                raw_resistances.append(high_i)

    tol = max(0.0002, float(tolerance_pct) / 100.0)

    def _cluster(values: list[float], probe: np.ndarray) -> list[tuple[float, int]]:
        cleaned = sorted(v for v in values if v > 0)
        if not cleaned:
            return []
        clusters: list[list[float]] = []
        for value in cleaned:
            merged = False
            for cluster in clusters:
                center = float(sum(cluster) / len(cluster))
                if abs(value - center) / max(center, 1e-9) <= tol:
                    cluster.append(value)
                    merged = True
                    break
            if not merged:
                clusters.append([value])

        out: list[tuple[float, int]] = []
        for cluster in clusters:
            level = float(np.median(cluster))
            band = max(level * tol, level * 0.0002)
            touches = int(np.sum(np.abs(probe - level) <= band))
            out.append((level, max(1, touches)))
        return out

    supports = sorted(_cluster(raw_supports, lows), key=lambda item: item[0], reverse=True)
    resistances = sorted(_cluster(raw_resistances, highs), key=lambda item: item[0])
    return supports, resistances


def build_trade_plan_from_zone(zone: dict[str, Any], df: pd.DataFrame) -> dict[str, float]:
    swing_lookback = int(DEFAULT_BOT_SETTINGS.get("SWING_LOOKBACK_BARS", 120))
    swing_lows, swing_highs = collect_swing_levels(df, lookback=swing_lookback)
    sw_low, sw_high = find_swing_levels(df, lookback=swing_lookback)
    supports, resistances = extract_structural_levels(
        df,
        lookback=max(120, swing_lookback),
        tolerance_pct=float(DEFAULT_BOT_SETTINGS.get("STRUCTURE_TOUCH_TOL_PCT", 0.06)),
    )
    volume_profile = build_local_volume_profile(
        df,
        lookback=int(DEFAULT_BOT_SETTINGS.get("VP_LOOKBACK_BARS", 96)),
        bins=int(DEFAULT_BOT_SETTINGS.get("VP_NUM_BINS", 48)),
    )
    volume_levels = list(volume_profile.get("levels", []))
    poc_level = float(volume_profile.get("poc", 0.0) or 0.0)
    if poc_level > 0 and poc_level not in volume_levels:
        volume_levels.append(poc_level)
        volume_levels = sorted(volume_levels)

    entry = safe_float(
        zone.get("mid_price"),
        (safe_float(zone.get("low")) + safe_float(zone.get("high"))) / 2.0,
    )
    low = safe_float(zone.get("low"))
    high = safe_float(zone.get("high"))
    direction = zone.get("direction", "bullish")

    sl_buffer_pct = max(
        float(DEFAULT_BOT_SETTINGS.get("SL_LEVEL_BUFFER_MIN_PCT", 0.10)),
        min(
            float(DEFAULT_BOT_SETTINGS.get("SL_NEAREST_LEVEL_BUFFER_PCT", 0.15)),
            float(DEFAULT_BOT_SETTINGS.get("SL_LEVEL_BUFFER_MAX_PCT", 0.20)),
        ),
    )
    tp_offset_pct = max(
        float(DEFAULT_BOT_SETTINGS.get("TP_OFFSET_MIN_PCT", 0.10)),
        min(
            float(DEFAULT_BOT_SETTINGS.get("TP_HVN_OFFSET_PCT", 0.12)),
            float(DEFAULT_BOT_SETTINGS.get("TP_OFFSET_MAX_PCT", 0.15)),
        ),
    )
    sl_buffer = sl_buffer_pct / 100.0
    tp_offset = tp_offset_pct / 100.0
    min_offset = float(DEFAULT_BOT_SETTINGS.get("SL_LEVEL_MIN_OFFSET_PCT", 0.10)) / 100.0

    highs_tail = df["high"].tail(max(40, swing_lookback)).astype(float).to_numpy()
    lows_tail = df["low"].tail(max(40, swing_lookback)).astype(float).to_numpy()
    closes_tail = df["close"].tail(max(40, swing_lookback)).astype(float).to_numpy()
    if len(closes_tail) >= 2:
        atr_like = np.mean(np.abs(np.diff(closes_tail)))
    else:
        atr_like = entry * 0.0015
    min_stop_distance = max(entry * 0.0022, atr_like * 0.40)

    def pick_sl_level(levels: list[tuple[float, int]], side: str) -> tuple[float, int]:
        if not levels:
            fallback = float(np.min(lows_tail)) if side == "LONG" else float(np.max(highs_tail))
            return fallback, 1
        scored: list[tuple[float, float, int]] = []
        for lvl, touches in levels:
            if lvl <= 0:
                continue
            if side == "LONG":
                if lvl >= low * (1.0 - min_offset) or lvl >= entry:
                    continue
                dist_pct = (entry - lvl) / max(entry, 1e-9) * 100.0
            else:
                if lvl <= high * (1.0 + min_offset) or lvl <= entry:
                    continue
                dist_pct = (lvl - entry) / max(entry, 1e-9) * 100.0
            if dist_pct <= 0.12 or dist_pct > 5.0:
                continue
            score = touches * 1.7 - abs(dist_pct - 0.9) * 0.8
            scored.append((score, lvl, touches))
        if not scored:
            fallback = float(np.min(lows_tail)) if side == "LONG" else float(np.max(highs_tail))
            return fallback, 1
        scored.sort(key=lambda item: (item[0], item[2]), reverse=True)
        return float(scored[0][1]), int(scored[0][2])

    def pick_tp_level(candidates: list[tuple[float, float]]) -> float:
        if not candidates:
            return entry
        scored: list[tuple[float, float]] = []
        for lvl, base_score in candidates:
            dist_pct = abs(lvl - entry) / max(entry, 1e-9) * 100.0
            if dist_pct < 0.20:
                continue
            scored.append((base_score - dist_pct * 0.08, lvl))
        if not scored:
            return float(candidates[0][0])
        scored.sort(key=lambda item: item[0], reverse=True)
        return float(scored[0][1])

    if direction == "bullish":
        support_levels = supports[:] if supports else [(float(x), 1) for x in swing_lows]
        structure_level, structure_touches = pick_sl_level(support_levels, side="LONG")
        sl_raw = structure_level * (1.0 - sl_buffer)
        if (entry - sl_raw) < min_stop_distance:
            sl_raw = entry - min_stop_distance

        target_floor = max(entry, high)
        target_candidates: list[tuple[float, float]] = []
        for level, touches in resistances:
            if level > target_floor:
                target_candidates.append((float(level), 1.2 + touches * 0.15))
        for level in swing_highs:
            if level > target_floor:
                target_candidates.append((float(level), 1.0))
        for level in volume_levels:
            if level > target_floor:
                target_candidates.append((float(level), 1.35))
        if not target_candidates:
            fallback = sw_high if sw_high > target_floor else max(float(df["high"].tail(20).max()), high * 1.01)
            target_candidates = [(float(fallback), 0.8)]
        target_level = pick_tp_level(target_candidates)
        tp_raw = target_level * (1.0 - tp_offset)
        if tp_raw <= entry:
            tp_raw = entry * (1.0 + max(tp_offset, 0.003))
    else:
        resistance_levels = resistances[:] if resistances else [(float(x), 1) for x in swing_highs]
        structure_level, structure_touches = pick_sl_level(resistance_levels, side="SHORT")
        sl_raw = structure_level * (1.0 + sl_buffer)
        if (sl_raw - entry) < min_stop_distance:
            sl_raw = entry + min_stop_distance

        target_ceiling = min(entry, low)
        target_candidates = []
        for level, touches in supports:
            if 0 < level < target_ceiling:
                target_candidates.append((float(level), 1.2 + touches * 0.15))
        for level in swing_lows:
            if 0 < level < target_ceiling:
                target_candidates.append((float(level), 1.0))
        for level in volume_levels:
            if 0 < level < target_ceiling:
                target_candidates.append((float(level), 1.35))
        if not target_candidates:
            fallback = sw_low if 0 < sw_low < target_ceiling else min(float(df["low"].tail(20).min()), low * 0.99)
            target_candidates = [(float(fallback), 0.8)]
        target_level = pick_tp_level(target_candidates)
        tp_raw = target_level * (1.0 + tp_offset)
        if tp_raw >= entry:
            tp_raw = entry * (1.0 - max(tp_offset, 0.003))

    sl = round_price_generic(sl_raw)
    tp = round_price_generic(tp_raw)
    risk = abs(entry - sl)
    reward = abs(tp - entry)

    return {
        "entry_price": round_price_generic(entry),
        "sl_price": sl,
        "tp_price": tp,
        "risk_reward": reward / risk if risk > 0 else 0.0,
        "swing_low": round_price_generic(sw_low),
        "swing_high": round_price_generic(sw_high),
        "structure_level": round_price_generic(structure_level),
        "target_level": round_price_generic(target_level),
        "vp_poc": round_price_generic(poc_level),
        "structure_touches": float(structure_touches),
    }


def detect_local_fvgs(
    df: pd.DataFrame,
    symbol: str,
    timeframe: str,
    threshold: float,
    min_gap_percent: float,
    last_price: float,
    bot_settings: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    if df.empty or len(df) < 10:
        return []

    bot_settings = dict(DEFAULT_BOT_SETTINGS) | dict(bot_settings or {})
    opens = df["open"].to_numpy(dtype=float)
    highs_raw = df["high"].to_numpy(dtype=float)
    lows_raw = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    timestamps = df["timestamp"].tolist()
    price_mode = str(bot_settings.get("FVG_PRICE_MODE", "wick")).lower()
    if price_mode == "body":
        highs = np.maximum(opens, closes)
        lows = np.minimum(opens, closes)
    else:
        highs = highs_raw
        lows = lows_raw
    body_sizes = np.abs(closes - opens)
    body_mid = (highs_raw + lows_raw) / 2.0
    body_mid = np.where(body_mid <= 0, 1.0, body_mid)
    body_pct = body_sizes / body_mid * 100.0
    percentile_lookback = int(safe_float(bot_settings.get("FVG_PERCENTILE_LOOKBACK"), 500))
    percentile_rank = max(compute_gap_diffs(highs.tolist(), lows.tolist(), max(2, len(df) - percentile_lookback), len(df)) or [0.0])
    scan_start = max(2, len(df) - 100)
    zones: list[dict[str, Any]] = []
    min_gap_pct = safe_float(min_gap_percent, safe_float(bot_settings.get("MIN_GAP_PERCENT"), 0.22))
    min_gap_pct *= safe_float(bot_settings.get("FVG_MIN_GAP_MULT"), 1.0)
    require_disp = bool(bot_settings.get("FVG_REQUIRE_DISPLACEMENT", True))
    require_same_dir = bool(bot_settings.get("FVG_REQUIRE_SAME_DIRECTION", True))
    disp_mult = safe_float(bot_settings.get("FVG_DISPLACEMENT_BODY_MULT"), 1.2)
    disp_min = safe_float(bot_settings.get("FVG_DISPLACEMENT_BODY_PCT"), 0.15)
    quality_floor = safe_float(bot_settings.get("FVG_QUALITY_MIN_SCORE"), 0.0)

    for index in range(scan_start, len(df)):
        mid_price = (highs[index] + lows[index]) / 2.0
        if mid_price <= 0:
            continue

        window_start = max(0, index - 60)
        baseline = float(np.median(body_pct[window_start:index])) if index > window_start else 0.0
        required_body = max(disp_min, baseline * disp_mult)
        if require_disp and body_pct[index - 1] < required_body:
            continue
        regime_adjustment, regime_info = dashboard_market_regime(highs_raw, lows_raw, closes, index, bot_settings)
        dynamic_threshold = threshold + regime_adjustment

        candidates = []
        if highs[index - 2] < lows[index]:
            if require_same_dir and closes[index - 1] <= opens[index - 1]:
                pass
            else:
                candidates.append(("bullish", highs[index - 2], lows[index]))
        if lows[index - 2] > highs[index]:
            if require_same_dir and closes[index - 1] >= opens[index - 1]:
                pass
            else:
                candidates.append(("bearish", highs[index], lows[index - 2]))

        for direction, low, high in candidates:
            gap_size = high - low
            gap_pct = gap_size / mid_price * 100.0
            if gap_pct < min_gap_pct:
                continue
            score = (gap_pct / percentile_rank * 100.0) if percentile_rank > 0 else 0.0
            quality_score = dashboard_fvg_quality_score(
                diff_pct=gap_pct,
                size_fvg=score,
                body_pct_prev=float(body_pct[index - 1]),
                required_body=required_body,
                regime_info=regime_info,
                direction=direction,
                opens=opens,
                closes=closes,
                index=index,
            )
            if score < dynamic_threshold or quality_score < quality_floor:
                continue
            zones.append({
                "symbol": symbol,
                "short_symbol": short_symbol(symbol),
                "timeframe": timeframe,
                "direction": direction,
                "low": low,
                "high": high,
                "mid_price": (low + high) / 2.0,
                "gap_size": gap_size,
                "gap_pct": gap_pct,
                "score": score,
                "quality_score": quality_score,
                "regime_adjustment": regime_adjustment,
                "timestamp": int(timestamps[index - 1]),
                "formed_at": datetime.fromtimestamp(timestamps[index - 1] / 1000.0, tz=timezone.utc).isoformat(),
                "candle_index": index - 1,
                "source": "dashboard-local",
            })

    for zone in zones:
        zone_low = safe_float(zone["low"])
        zone_high = safe_float(zone["high"])
        gap_size = zone_high - zone_low
        direction = str(zone.get("direction", "")).lower()

        mitigated = False
        max_fill_ratio = 0.0
        for forward_index in range(zone["candle_index"] + 2, len(df)):
            f_low = safe_float(lows[forward_index])
            f_high = safe_float(highs[forward_index])
            f_close = safe_float(closes[forward_index])

            if gap_size > 0:
                if direction == "bullish":
                    penetration = max(0.0, min(zone_high, zone_high) - max(f_low, zone_low))
                    fill_ratio = min(1.0, max(0.0, penetration / gap_size))
                    max_fill_ratio = max(max_fill_ratio, fill_ratio)
                    if f_close <= zone_low:
                        mitigated = True
                        break
                else:
                    penetration = max(0.0, min(f_high, zone_high) - zone_low)
                    fill_ratio = min(1.0, max(0.0, penetration / gap_size))
                    max_fill_ratio = max(max_fill_ratio, fill_ratio)
                    if f_close >= zone_high:
                        mitigated = True
                        break

        zone["fill_ratio"] = max_fill_ratio
        zone["mitigated"] = mitigated
        if mitigated:
            zone["status"] = "mitigated"
        elif max_fill_ratio >= 0.40:
            zone["status"] = "partial"
        else:
            zone["status"] = "active"

        zone["entry_band_low"] = zone_low + gap_size * 0.40
        zone["entry_band_high"] = zone_low + gap_size * 0.60
        zone["position_pct"] = ((last_price - zone_low) / gap_size * 100.0) if gap_size > 0 else None
        zone["last_price"] = last_price
        zone.update(build_trade_plan_from_zone(zone, df))

    zones.sort(key=lambda item: item["timestamp"], reverse=True)
    return zones


def filter_zones(
    zones: list[dict[str, Any]],
    min_score: float,
    min_gap: float,
    status_filter: str,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    for zone in zones:
        if safe_float(zone.get("score")) < min_score:
            continue
        if safe_float(zone.get("gap_pct")) < min_gap:
            continue
        zone_status = str(zone.get("status", "active"))
        if status_filter == "active":
            if zone_status not in {"active", "partial"}:
                continue
        elif status_filter != "all" and zone_status != status_filter:
            continue
        filtered.append(zone)
    return filtered


def build_zone_hover(zone: dict[str, Any]) -> str:
    position_pct = zone.get("position_pct")
    position_text = f"{position_pct:.1f}%" if isinstance(position_pct, (int, float)) else "n/a"
    return (
        f"<b>{zone.get('short_symbol')}</b><br>"
        f"Direction: {zone.get('direction', '').title()}<br>"
        f"Status: {zone.get('status', 'n/a')}<br>"
        f"Gap: {safe_float(zone.get('gap_pct')):.3f}%<br>"
        f"Score: {safe_float(zone.get('score')):.1f}<br>"
        f"Quality: {safe_float(zone.get('quality_score')):.1f}<br>"
        f"Position in zone: {position_text}<br>"
        f"RR: {safe_float(zone.get('risk_reward')):.2f}<br>"
        f"Formed: {zone.get('formed_at', 'n/a')}<extra></extra>"
    )


def build_chart_figure(
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
    zones: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    settings: dict[str, Any],
    show_zones: bool,
    show_levels: bool,
    show_labels: bool,
) -> go.Figure:
    figure = go.Figure()

    if df.empty:
        figure.update_layout(
            template=None,
            paper_bgcolor="#0b1220",
            plot_bgcolor="#0b1220",
            font={"color": "#dbe4ff"},
            annotations=[
                {
                    "text": f"No market data for {short_symbol(symbol)} {timeframe}",
                    "showarrow": False,
                    "xref": "paper",
                    "yref": "paper",
                    "x": 0.5,
                    "y": 0.5,
                    "font": {"size": 16},
                }
            ],
        )
        return figure

    chart_df = df.copy()
    chart_df["datetime"] = pd.to_datetime(chart_df["timestamp"], unit="ms", utc=True)

    figure.add_trace(
        go.Candlestick(
            x=chart_df["datetime"],
            open=chart_df["open"],
            high=chart_df["high"],
            low=chart_df["low"],
            close=chart_df["close"],
            increasing_line_color="#16c784",
            increasing_fillcolor="#16c784",
            decreasing_line_color="#ea3943",
            decreasing_fillcolor="#ea3943",
            name=short_symbol(symbol),
            showlegend=False,
        )
    )

    chart_end = chart_df["datetime"].iloc[-1] + pd.to_timedelta(TIMEFRAME_TO_MS[timeframe] * 2, unit="ms")

    for index, zone in enumerate(zones):
        zone_start = pd.to_datetime(int(zone["timestamp"]), unit="ms", utc=True)
        direction = zone.get("direction")
        active = zone.get("status") == "active"
        base_color = settings["bullish_color"] if direction == "bullish" else settings["bearish_color"]
        fill = "rgba(34, 197, 94, 0.24)" if direction == "bullish" else "rgba(239, 68, 68, 0.24)"
        if not active:
            fill = fill.replace("0.24", "0.10")
        hovertemplate = build_zone_hover(zone)

        if show_zones:
            figure.add_shape(
                type="rect",
                x0=zone_start,
                x1=chart_end,
                y0=safe_float(zone.get("low")),
                y1=safe_float(zone.get("high")),
                fillcolor=fill,
                line={
                    "color": base_color,
                    "width": 1.2 if active else 0.8,
                    "dash": "solid" if active else "dot",
                },
                layer="below",
            )

        if show_levels:
            for price_key, color_key, dash_style in (
                ("entry_price", "entry_color", "dash"),
                ("sl_price", "sl_color", "solid"),
                ("tp_price", "tp_color", "solid"),
            ):
                price = safe_float(zone.get(price_key))
                figure.add_trace(
                    go.Scatter(
                        x=[zone_start, chart_end],
                        y=[price, price],
                        mode="lines",
                        line={"color": settings[color_key], "dash": dash_style, "width": 1.0},
                        showlegend=False,
                        hovertemplate=hovertemplate,
                        opacity=0.95 if active else 0.45,
                    )
                )
                if show_labels:
                    figure.add_annotation(
                        x=chart_end,
                        y=price,
                        text=price_key.replace("_price", "").upper(),
                        xanchor="left",
                        yanchor="middle",
                        showarrow=False,
                        font={"size": 9, "color": settings[color_key]},
                        bgcolor="rgba(11,18,32,0.65)",
                        bordercolor="rgba(148,163,184,0.08)",
                        borderpad=2,
                    )

        midpoint = zone_start + (chart_end - zone_start) / 2
        figure.add_trace(
            go.Scatter(
                x=[midpoint],
                y=[safe_float(zone.get("mid_price"))],
                mode="markers",
                marker={"size": 9, "color": base_color, "opacity": 0.01},
                hovertemplate=hovertemplate,
                showlegend=False,
            )
        )

    for position in [item for item in positions if item.get("symbol") == symbol]:
        figure.add_hline(
            y=safe_float(position.get("entry")),
            line_color="#4da3ff",
            line_dash="dash",
            line_width=1.4,
            annotation_text=f"{position.get('side', 'LIVE')} entry",
            annotation_font_color="#9ec5ff",
            annotation_position="top left",
        )

    figure.update_layout(
        template=None,
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font={"family": "Segoe UI, Arial, sans-serif", "color": "#dbe4ff"},
        margin={"l": 18, "r": 18, "t": 12, "b": 18},
        xaxis={
            "showgrid": True,
            "gridcolor": "rgba(148, 163, 184, 0.08)",
            "rangeslider": {"visible": False},
            "showline": False,
            "zeroline": False,
            "fixedrange": False,
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "rgba(148, 163, 184, 0.08)",
            "side": "right",
            "showline": False,
            "zeroline": False,
            "fixedrange": False,
        },
        hovermode="closest",
        dragmode="pan",
        uirevision=f"{symbol}-{timeframe}",
    )
    return figure


def build_stat_card(label: str, value: str, accent: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(label, className="small text-uppercase text-secondary mb-2"),
                html.Div(value, style={"fontSize": "1.35rem", "fontWeight": 700, "color": accent}),
            ]
        ),
        className="shadow-sm border-0 h-100",
        style={"background": "#121a2b"},
    )


def build_placeholder_card(title: str, message: str) -> dbc.Card:
    return dbc.Card(
        dbc.CardBody(
            [
                html.H6(title, className="mb-2", style={"color": "#f8fafc"}),
                html.Div(message, className="text-secondary small"),
            ]
        ),
        className="border-0 shadow-sm",
        style={"background": "#121a2b"},
    )


def build_signal_card(signal: dict[str, Any]) -> dbc.Card:
    badge_color = "success" if signal.get("signal_side") == "LONG" else "danger"
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Strong(signal.get("short_symbol", ""), style={"color": "#f8fafc"}),
                        dbc.Badge(signal.get("signal_side", "PENDING"), color=badge_color, className="ms-2"),
                        dbc.Badge(f"#{signal.get('rank', '-')}", color="secondary", className="ms-2"),
                    ],
                    className="mb-2",
                ),
                html.Div(f"Score {safe_float(signal.get('score')):.1f} | Gap {safe_float(signal.get('gap_pct')):.3f}%", className="small text-secondary"),
                html.Div(f"Entry {safe_float(signal.get('entry_price')):.6f}", className="small text-light"),
                html.Div(f"SL {safe_float(signal.get('sl_price')):.6f} | TP {safe_float(signal.get('tp_price')):.6f}", className="small text-light"),
                html.Div(f"RR {safe_float(signal.get('risk_reward')):.2f} | Zone {safe_float(signal.get('position_pct')):.1f}%", className="small text-secondary"),
                html.Div(f"Dist to mid {safe_float(signal.get('distance_to_mid')):.2f}%", className="small text-secondary"),
            ]
        ),
        className="border-0 mb-2",
        style={"background": "#101827"},
    )


def build_position_card(position: dict[str, Any]) -> dbc.Card:
    badge_color = "success" if position.get("side") == "LONG" else "danger"
    return dbc.Card(
        dbc.CardBody(
            [
                html.Div(
                    [
                        html.Strong(position.get("short_symbol", ""), style={"color": "#f8fafc"}),
                        dbc.Badge(position.get("side", "LIVE"), color=badge_color, className="ms-2"),
                    ],
                    className="mb-2",
                ),
                html.Div(f"Entry {safe_float(position.get('entry')):.6f}", className="small text-light"),
                html.Div(f"SL {safe_float(position.get('sl')):.6f} | TP {safe_float(position.get('tp')):.6f}", className="small text-light"),
                html.Div(f"Qty {safe_float(position.get('qty')):.4f} | Risk ${safe_float(position.get('risk_usd')):.2f}", className="small text-secondary"),
                html.Div(f"Opened {position.get('opened_at', 'n/a')}", className="small text-secondary"),
            ]
        ),
        className="border-0 mb-2",
        style={"background": "#101827"},
    )


provider = DashboardDataProvider(STATE_FILE, BOT_LOG_FILE, CONFIG_FILE)

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    suppress_callback_exceptions=True,
    title=APP_TITLE,
)
app.index_string = """
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            body {
                background: radial-gradient(circle at top left, #16233e 0%, #0b1220 38%, #07101c 100%);
                min-height: 100vh;
            }
            .dashboard-shell { color: #dbe4ff; }
            .panel-card {
                background: rgba(12, 18, 32, 0.92);
                border: 1px solid rgba(148, 163, 184, 0.12);
                border-radius: 18px;
                box-shadow: 0 18px 60px rgba(2, 6, 23, 0.35);
                backdrop-filter: blur(10px);
            }
            .scroll-panel {
                max-height: calc(100vh - 255px);
                overflow-y: auto;
                padding-right: 6px;
            }
            .chart-panel { min-height: 430px; }
            .dash-graph { border-radius: 14px; overflow: hidden; }
            .symbol-button {
                width: 100%;
                text-align: left;
                border-radius: 14px;
                margin-bottom: 8px;
                border: 1px solid rgba(148, 163, 184, 0.12);
                background: rgba(15, 23, 42, 0.9);
                color: #dbe4ff;
            }
            .symbol-button.selected {
                border-color: rgba(59, 130, 246, 0.65);
                background: rgba(30, 41, 59, 0.96);
                box-shadow: 0 0 0 1px rgba(59, 130, 246, 0.25);
            }
            .symbol-button:hover {
                background: rgba(30, 41, 59, 0.96);
                color: #f8fafc;
            }
            .mono-log {
                background: #09111f;
                color: #cbd5e1;
                font-family: Consolas, Monaco, monospace;
                font-size: 0.78rem;
                padding: 14px;
                border-radius: 14px;
                white-space: pre-wrap;
                min-height: 380px;
            }
            .compact-input .form-control,
            .compact-input .form-select {
                background: rgba(15, 23, 42, 0.92);
                border-color: rgba(148, 163, 184, 0.16);
                color: #f8fafc;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

app.layout = dbc.Container(
    fluid=True,
    className="dashboard-shell py-3",
    children=[
        dcc.Interval(id="poll-interval", interval=POLL_INTERVAL_MS, n_intervals=0),
        dcc.Store(id="state-store"),
        dcc.Store(id="selected-symbols-store", data=[]),
        dcc.Store(id="sidebar-collapsed-store", data=False),
        dcc.Download(id="export-download"),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            [
                                                html.Div("FVG Trading Bot v2.6", className="text-uppercase small text-secondary"),
                                                html.H2("Professional FVG Dashboard", className="mb-1", style={"fontWeight": 800}),
                                                html.Div(
                                                    "BigBeluga-style zones, live entries, SL, TP, pending signals and bot monitoring in one view.",
                                                    className="text-secondary",
                                                ),
                                            ],
                                            md=8,
                                        ),
                                        dbc.Col(
                                            [
                                                html.Div(id="status-line", className="text-end small text-secondary mb-2"),
                                                dbc.Stack(
                                                    [
                                                        dbc.Button("Sidebar", id="toggle-sidebar", color="secondary", className="me-2"),
                                                        dbc.Button("Clear All Charts", id="clear-all", color="secondary", className="me-2"),
                                                        dbc.Button("Refresh All", id="refresh-all", color="primary", className="me-2"),
                                                        dbc.RadioItems(
                                                            id="timeframe-selector",
                                                            options=[{"label": "5m", "value": "5m"}, {"label": "15m", "value": "15m"}],
                                                            value="5m",
                                                            inline=True,
                                                            inputClassName="me-1",
                                                            labelClassName="me-3",
                                                        ),
                                                    ],
                                                    direction="horizontal",
                                                    className="justify-content-end",
                                                ),
                                            ],
                                            md=4,
                                        ),
                                    ],
                                    className="g-3 align-items-center",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(
                                            dbc.Checklist(
                                                id="chart-toggle-checklist",
                                                options=[
                                                    {"label": "Show Zones", "value": "zones"},
                                                    {"label": "Show SL-TP Lines", "value": "levels"},
                                                    {"label": "Show Labels", "value": "labels"},
                                                    {"label": "Show Mitigated", "value": "mitigated"},
                                                ],
                                                value=["zones", "levels", "labels"],
                                                inline=True,
                                                switch=True,
                                            ),
                                            width=12,
                                        )
                                    ],
                                    className="mt-2",
                                ),
                                html.Div(id="warning-banner", className="mt-3"),
                            ]
                        ),
                        className="panel-card border-0",
                    ),
                    width=12,
                )
            ],
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.Div("Market Universe", className="text-uppercase small text-secondary mb-2"),
                                dbc.Input(id="symbol-search", placeholder="Search BTC, ETH, SOL...", className="mb-2 compact-input"),
                                dbc.Select(
                                    id="zone-status-filter",
                                    options=[
                                        {"label": "All zones", "value": "all"},
                                        {"label": "Active only", "value": "active"},
                                        {"label": "Mitigated only", "value": "mitigated"},
                                    ],
                                    value="all",
                                    className="mb-2 compact-input",
                                ),
                                dbc.Row(
                                    [
                                        dbc.Col(dbc.Input(id="min-score-filter", type="number", value=0, min=0, step=0.5, placeholder="Min score", className="compact-input"), md=6),
                                        dbc.Col(dbc.Input(id="min-gap-filter", type="number", value=0, min=0, step=0.01, placeholder="Min gap %", className="compact-input"), md=6),
                                    ],
                                    className="g-2 mb-3",
                                ),
                                html.Div("Select up to 6 symbols for multi-chart mode.", className="small text-secondary mb-2"),
                                html.Div(id="symbol-list", className="scroll-panel"),
                            ]
                        ),
                        className="panel-card border-0 h-100",
                    ),
                    id="left-sidebar-col",
                    xl=2,
                    lg=2,
                    md=4,
                    sm=12,
                    className="mb-3",
                ),
                dbc.Col(
                    [dbc.Row(id="stats-row", className="g-3 mb-3"), html.Div(id="charts-container")],
                    id="main-chart-col",
                    xl=8,
                    lg=8,
                    md=8,
                    sm=12,
                    className="mb-3",
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Visual Settings", tab_id="visual-settings-tab"),
                                            dbc.Tab(label="Bot Settings", tab_id="bot-settings-tab"),
                                        ],
                                        id="settings-tabs",
                                        active_tab="visual-settings-tab",
                                    )
                                ),
                                dbc.CardBody(
                                    [
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Input(id="threshold-input", type="number", value=DEFAULT_SETTINGS["threshold"], step=0.1, min=0, className="compact-input"), md=6),
                                                        dbc.Col(dbc.Input(id="settings-min-gap-input", type="number", value=DEFAULT_SETTINGS["min_gap_percent"], step=0.01, min=0, className="compact-input"), md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col(dbc.Input(id="risk-input", type="number", value=DEFAULT_SETTINGS["risk_percent"], step=0.5, min=0, className="compact-input"), md=6),
                                                        dbc.Col(html.Div("v2.6 - wider chart, cleaner labels, dashboard-only filters.", className="small text-secondary"), md=6),
                                                    ],
                                                    className="g-2 mb-3",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Bullish", className="small text-secondary mb-1"), dbc.Input(id="bullish-color-input", type="color", value=DEFAULT_SETTINGS["bullish_color"], className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Bearish", className="small text-secondary mb-1"), dbc.Input(id="bearish-color-input", type="color", value=DEFAULT_SETTINGS["bearish_color"], className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-3",
                                                ),
                                                dbc.Stack(
                                                    [
                                                        dbc.Button("Export Zones CSV", id="export-csv", color="secondary"),
                                                        dbc.Button("Export Pine Script", id="export-pine", color="secondary"),
                                                    ],
                                                    gap=2,
                                                ),
                                            ],
                                            id="visual-settings-pane",
                                        ),
                                        html.Div(
                                            [
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Threshold", className="small text-secondary mb-1"), dbc.Input(id="bot-threshold-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_PERCENTILE_THRESHOLD"], step=0.1, min=0, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Min gap %", className="small text-secondary mb-1"), dbc.Input(id="bot-min-gap-input", type="number", value=DEFAULT_BOT_SETTINGS["MIN_GAP_PERCENT"], step=0.01, min=0, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("FVG lookback", className="small text-secondary mb-1"), dbc.Input(id="bot-fvg-lookback-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_LOOKBACK"], step=8, min=50, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Max age days", className="small text-secondary mb-1"), dbc.Input(id="bot-fvg-age-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_MAX_AGE_DAYS"], step=1, min=0, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Min gap mult", className="small text-secondary mb-1"), dbc.Input(id="bot-fvg-min-mult-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_MIN_GAP_MULT"], step=0.05, min=1.0, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Max active/side", className="small text-secondary mb-1"), dbc.Input(id="bot-fvg-max-active-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_MAX_ACTIVE_PER_SIDE"], step=1, min=1, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([
                                                            html.Div("FVG price mode", className="small text-secondary mb-1"),
                                                            dbc.Select(
                                                                id="bot-fvg-price-mode-input",
                                                                options=[
                                                                    {"label": "Wick (ICT)", "value": "wick"},
                                                                    {"label": "Body (aggressive)", "value": "body"},
                                                                ],
                                                                value=DEFAULT_BOT_SETTINGS["FVG_PRICE_MODE"],
                                                                className="compact-input",
                                                            ),
                                                        ], md=6),
                                                        dbc.Col([html.Div("Swing lookback", className="small text-secondary mb-1"), dbc.Input(id="bot-swing-lookback-input", type="number", value=DEFAULT_BOT_SETTINGS["SWING_LOOKBACK_BARS"], step=10, min=20, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([
                                                            dbc.Checklist(
                                                                id="bot-disp-toggle",
                                                                options=[{"label": "Require displacement", "value": "on"}],
                                                                value=["on"] if DEFAULT_BOT_SETTINGS["FVG_REQUIRE_DISPLACEMENT"] else [],
                                                                switch=True,
                                                                className="small",
                                                            )
                                                        ], md=6),
                                                        dbc.Col([
                                                            dbc.Checklist(
                                                                id="bot-same-dir-toggle",
                                                                options=[{"label": "Require same direction", "value": "on"}],
                                                                value=["on"] if DEFAULT_BOT_SETTINGS["FVG_REQUIRE_SAME_DIRECTION"] else [],
                                                                switch=True,
                                                                className="small",
                                                            )
                                                        ], md=6),
                                                    ],
                                                    className="g-2 mb-1",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Disp body mult", className="small text-secondary mb-1"), dbc.Input(id="bot-disp-mult-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_DISPLACEMENT_BODY_MULT"], step=0.05, min=0, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Disp body %", className="small text-secondary mb-1"), dbc.Input(id="bot-disp-pct-input", type="number", value=DEFAULT_BOT_SETTINGS["FVG_DISPLACEMENT_BODY_PCT"], step=0.01, min=0, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Risk %", className="small text-secondary mb-1"), dbc.Input(id="bot-risk-input", type="number", value=DEFAULT_BOT_SETTINGS["RISK_PER_TRADE"] * 100.0, step=0.5, min=0, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Max positions", className="small text-secondary mb-1"), dbc.Input(id="bot-max-positions-input", type="number", value=DEFAULT_BOT_SETTINGS["MAX_CONCURRENT_POSITIONS"], step=1, min=1, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Top coins", className="small text-secondary mb-1"), dbc.Input(id="bot-top-coins-input", type="number", value=DEFAULT_BOT_SETTINGS["TOP_N_COINS"], step=1, min=1, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Poll sec", className="small text-secondary mb-1"), dbc.Input(id="bot-poll-input", type="number", value=DEFAULT_BOT_SETTINGS["POLL_INTERVAL_SEC"], step=5, min=10, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("SL below level %", className="small text-secondary mb-1"), dbc.Input(id="bot-sl-buffer-input", type="number", value=DEFAULT_BOT_SETTINGS["SL_NEAREST_LEVEL_BUFFER_PCT"], step=0.01, min=0, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("TP before HVN %", className="small text-secondary mb-1"), dbc.Input(id="bot-tp-buffer-input", type="number", value=DEFAULT_BOT_SETTINGS["TP_HVN_OFFSET_PCT"], step=0.01, min=0, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("VP lookback", className="small text-secondary mb-1"), dbc.Input(id="bot-vp-lookback-input", type="number", value=DEFAULT_BOT_SETTINGS["VP_LOOKBACK_BARS"], step=4, min=20, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("VP bins", className="small text-secondary mb-1"), dbc.Input(id="bot-vp-bins-input", type="number", value=DEFAULT_BOT_SETTINGS["VP_NUM_BINS"], step=2, min=12, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Mid zone low", className="small text-secondary mb-1"), dbc.Input(id="bot-mid-low-input", type="number", value=DEFAULT_BOT_SETTINGS["MID_ZONE_LOW"] * 100.0, step=1, min=0, max=100, className="compact-input")], md=6),
                                                        dbc.Col([html.Div("Mid zone high", className="small text-secondary mb-1"), dbc.Input(id="bot-mid-high-input", type="number", value=DEFAULT_BOT_SETTINGS["MID_ZONE_HIGH"] * 100.0, step=1, min=0, max=100, className="compact-input")], md=6),
                                                    ],
                                                    className="g-2 mb-2",
                                                ),
                                                dbc.Row(
                                                    [
                                                        dbc.Col([html.Div("Pre-alert dist %", className="small text-secondary mb-1"), dbc.Input(id="bot-prealert-distance-input", type="number", value=DEFAULT_BOT_SETTINGS["PRE_ALERT_DISTANCE_PCT"], step=0.01, min=0, className="compact-input")], md=6),
                                                        dbc.Col(html.Div("v2.6 final - swing-aware SL, conservative TP, active FVG only.", className="small text-secondary pt-4"), md=6),
                                                    ],
                                                    className="g-2 mb-3",
                                                ),
                                                dbc.Button("Apply & Restart Scan", id="apply-bot-settings", color="warning", className="w-100"),
                                                html.Div(id="bot-settings-status", className="small text-secondary mt-2"),
                                            ],
                                            id="bot-settings-pane",
                                            style={"display": "none"},
                                        ),
                                    ]
                                ),
                            ],
                            className="panel-card border-0 mb-3",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader(
                                    dbc.Tabs(
                                        [
                                            dbc.Tab(label="Pending Signals", tab_id="pending-tab"),
                                            dbc.Tab(label="Active Positions", tab_id="positions-tab"),
                                            dbc.Tab(label="Bot Log", tab_id="log-tab"),
                                        ],
                                        id="right-tabs",
                                        active_tab="pending-tab",
                                    )
                                ),
                                dbc.CardBody(html.Div(id="right-tab-content", className="scroll-panel")),
                            ],
                            className="panel-card border-0",
                        ),
                    ],
                    id="right-panel-col",
                    xl=2,
                    lg=2,
                    md=12,
                    sm=12,
                ),
            ],
            className="g-3",
        ),
    ],
)


@app.callback(
    Output("state-store", "data"),
    Output("status-line", "children"),
    Output("warning-banner", "children"),
    Input("poll-interval", "n_intervals"),
    Input("refresh-all", "n_clicks"),
)
def refresh_state(_: int, __: Optional[int]) -> tuple[dict[str, Any], str, Any]:
    if ctx.triggered_id == "refresh-all":
        provider.clear_ohlcv_cache()

    provider.load_config()
    state, status_text, warning = provider.load_state()
    updated_at = state.get("updated_at")
    state_label = "Bot online" if state.get("bot_running") and not state_is_stale(updated_at) else "Snapshot mode"
    status_line = f"{status_text} | {state_label} | Updated {age_label(updated_at)}"
    banner = dbc.Alert(warning, color="warning", className="mb-0") if warning else ""
    return state, status_line, banner


@app.callback(
    Output("selected-symbols-store", "data"),
    Input("state-store", "data"),
    Input("clear-all", "n_clicks"),
    Input({"type": "symbol-btn", "symbol": ALL}, "n_clicks"),
    State({"type": "symbol-btn", "symbol": ALL}, "id"),
    State("selected-symbols-store", "data"),
)
def update_selected_symbols(
    state: dict[str, Any],
    __: Optional[int],
    _: list[Optional[int]],
    ___: list[dict[str, str]],
    current_selection: list[str],
) -> list[str]:
    current_selection = list(current_selection or [])
    triggered = ctx.triggered_id

    if triggered == "clear-all":
        return []

    if isinstance(triggered, dict) and triggered.get("type") == "symbol-btn":
        symbol = triggered.get("symbol")
        if symbol in current_selection:
            return [item for item in current_selection if item != symbol]
        return (current_selection + [symbol])[-MAX_VISIBLE_CHARTS:]

    available_symbols = dashboard_symbol_pool(state)
    valid = [symbol for symbol in current_selection if symbol in available_symbols]
    if valid:
        return valid[-MAX_VISIBLE_CHARTS:]

    preferred = [item.get("symbol") for item in state.get("pending_signals", []) if item.get("symbol")]
    preferred += [item.get("symbol") for item in state.get("active_positions", []) if item.get("symbol")]
    preferred += available_symbols

    selected: list[str] = []
    for symbol in preferred:
        if symbol and symbol not in selected:
            selected.append(symbol)
        if len(selected) >= min(MAX_VISIBLE_CHARTS, max(1, len(available_symbols) or 1)):
            break
    return selected[:MAX_VISIBLE_CHARTS]


@app.callback(
    Output("symbol-list", "children"),
    Input("state-store", "data"),
    Input("symbol-search", "value"),
    Input("zone-status-filter", "value"),
    Input("min-score-filter", "value"),
    Input("min-gap-filter", "value"),
    Input("selected-symbols-store", "data"),
)
def render_symbol_list(
    state: dict[str, Any],
    search_value: Optional[str],
    status_filter: str,
    min_score: Optional[float],
    min_gap: Optional[float],
    selected_symbols: list[str],
) -> list[Any]:
    selected_symbols = selected_symbols or []
    search_text = (search_value or "").strip().lower()
    score_floor = safe_float(min_score, 0.0)
    gap_floor = safe_float(min_gap, 0.0)
    symbol_order = dashboard_symbol_pool(state)
    cards: list[Any] = []

    for symbol in symbol_order:
        snapshot = state.get("symbols", {}).get(symbol, {})
        zones = filter_zones(snapshot.get("zones", []), score_floor, gap_floor, status_filter)

        if search_text and search_text not in symbol.lower() and search_text not in short_symbol(symbol).lower():
            continue
        if (score_floor > 0 or gap_floor > 0 or status_filter != "all") and not zones:
            continue

        top_zone = max(zones, key=lambda item: safe_float(item.get("score")), default=None)
        active_count = len([zone for zone in snapshot.get("zones", []) if zone.get("status") == "active"])
        score_text = f"{safe_float(top_zone.get('score')):.1f}" if top_zone else "0.0"
        gap_text = f"{safe_float(top_zone.get('gap_pct')):.3f}%" if top_zone else "-"
        last_price = safe_float(snapshot.get("last_price"), 0.0)

        cards.append(
            html.Button(
                [
                    html.Div(short_symbol(symbol), style={"fontWeight": 700, "fontSize": "1rem"}),
                    html.Div(
                        [
                            dbc.Badge(f"{active_count} active", color="secondary", className="me-1"),
                            dbc.Badge(f"Score {score_text}", color="info", className="me-1"),
                        ],
                        className="my-1",
                    ),
                    html.Div(f"Gap {gap_text}", className="small text-secondary"),
                    html.Div(f"Last {last_price:.6f}" if last_price else "Last price unavailable", className="small text-secondary"),
                ],
                id={"type": "symbol-btn", "symbol": symbol},
                n_clicks=0,
                className=f"symbol-button{' selected' if symbol in selected_symbols else ''}",
            )
        )

    if not cards:
        return [html.Div("No symbols match the current filters.", className="text-secondary small")]
    return cards


@app.callback(
    Output("stats-row", "children"),
    Input("state-store", "data"),
)
def render_stats(state: dict[str, Any]) -> list[Any]:
    stats = state.get("stats", {})
    return [
        dbc.Col(build_stat_card("Active Zones", str(int(stats.get("active_zones", 0))), "#22c55e"), xl=3, md=6, sm=12),
        dbc.Col(build_stat_card("Mitigated Zones", str(int(stats.get("mitigated_zones", 0))), "#f59e0b"), xl=3, md=6, sm=12),
        dbc.Col(build_stat_card("Avg Active Score", f"{safe_float(stats.get('average_active_score')):.1f}", "#60a5fa"), xl=3, md=6, sm=12),
        dbc.Col(build_stat_card("Open Positions", str(int(stats.get("open_positions", 0))), "#f87171"), xl=3, md=6, sm=12),
    ]


@app.callback(
    Output("sidebar-collapsed-store", "data"),
    Input("toggle-sidebar", "n_clicks"),
    State("sidebar-collapsed-store", "data"),
    prevent_initial_call=True,
)
def toggle_sidebar(_: Optional[int], collapsed: bool) -> bool:
    return not bool(collapsed)


@app.callback(
    Output("left-sidebar-col", "style"),
    Output("main-chart-col", "style"),
    Input("sidebar-collapsed-store", "data"),
)
def resize_layout(collapsed: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    if collapsed:
        return {"display": "none"}, {"flex": "0 0 83.333333%", "maxWidth": "83.333333%"}
    return {}, {}


@app.callback(
    Output("visual-settings-pane", "style"),
    Output("bot-settings-pane", "style"),
    Input("settings-tabs", "active_tab"),
)
def switch_settings_tab(active_tab: str) -> tuple[dict[str, Any], dict[str, Any]]:
    if active_tab == "bot-settings-tab":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@app.callback(
    Output("bot-threshold-input", "value"),
    Output("bot-min-gap-input", "value"),
    Output("bot-fvg-lookback-input", "value"),
    Output("bot-fvg-age-input", "value"),
    Output("bot-fvg-min-mult-input", "value"),
    Output("bot-fvg-max-active-input", "value"),
    Output("bot-fvg-price-mode-input", "value"),
    Output("bot-swing-lookback-input", "value"),
    Output("bot-disp-toggle", "value"),
    Output("bot-same-dir-toggle", "value"),
    Output("bot-disp-mult-input", "value"),
    Output("bot-disp-pct-input", "value"),
    Output("bot-risk-input", "value"),
    Output("bot-max-positions-input", "value"),
    Output("bot-top-coins-input", "value"),
    Output("bot-poll-input", "value"),
    Output("bot-sl-buffer-input", "value"),
    Output("bot-tp-buffer-input", "value"),
    Output("bot-vp-lookback-input", "value"),
    Output("bot-vp-bins-input", "value"),
    Output("bot-mid-low-input", "value"),
    Output("bot-mid-high-input", "value"),
    Output("bot-prealert-distance-input", "value"),
    Input("state-store", "data"),
)
def populate_bot_settings(
    _: dict[str, Any]
) -> tuple[
    float, float, int, int, float, int, str, int, list[str], list[str], float, float,
    float, int, int, int, float, float, float, float, float, float, float
]:
    config = provider.load_config()
    bot_config = dict(DEFAULT_BOT_SETTINGS) | dict(config.get("bot", {}))
    return (
        safe_float(bot_config.get("FVG_PERCENTILE_THRESHOLD"), DEFAULT_BOT_SETTINGS["FVG_PERCENTILE_THRESHOLD"]),
        safe_float(bot_config.get("MIN_GAP_PERCENT"), DEFAULT_BOT_SETTINGS["MIN_GAP_PERCENT"]),
        int(safe_float(bot_config.get("FVG_LOOKBACK"), DEFAULT_BOT_SETTINGS["FVG_LOOKBACK"])),
        int(safe_float(bot_config.get("FVG_MAX_AGE_DAYS"), DEFAULT_BOT_SETTINGS["FVG_MAX_AGE_DAYS"])),
        safe_float(bot_config.get("FVG_MIN_GAP_MULT"), DEFAULT_BOT_SETTINGS["FVG_MIN_GAP_MULT"]),
        int(safe_float(bot_config.get("FVG_MAX_ACTIVE_PER_SIDE"), DEFAULT_BOT_SETTINGS["FVG_MAX_ACTIVE_PER_SIDE"])),
        str(bot_config.get("FVG_PRICE_MODE", DEFAULT_BOT_SETTINGS["FVG_PRICE_MODE"])),
        int(safe_float(bot_config.get("SWING_LOOKBACK_BARS"), DEFAULT_BOT_SETTINGS["SWING_LOOKBACK_BARS"])),
        ["on"] if bool(bot_config.get("FVG_REQUIRE_DISPLACEMENT", DEFAULT_BOT_SETTINGS["FVG_REQUIRE_DISPLACEMENT"])) else [],
        ["on"] if bool(bot_config.get("FVG_REQUIRE_SAME_DIRECTION", DEFAULT_BOT_SETTINGS["FVG_REQUIRE_SAME_DIRECTION"])) else [],
        safe_float(bot_config.get("FVG_DISPLACEMENT_BODY_MULT"), DEFAULT_BOT_SETTINGS["FVG_DISPLACEMENT_BODY_MULT"]),
        safe_float(bot_config.get("FVG_DISPLACEMENT_BODY_PCT"), DEFAULT_BOT_SETTINGS["FVG_DISPLACEMENT_BODY_PCT"]),
        safe_float(bot_config.get("RISK_PER_TRADE"), DEFAULT_BOT_SETTINGS["RISK_PER_TRADE"]) * 100.0,
        int(safe_float(bot_config.get("MAX_CONCURRENT_POSITIONS"), DEFAULT_BOT_SETTINGS["MAX_CONCURRENT_POSITIONS"])),
        int(safe_float(bot_config.get("TOP_N_COINS"), DEFAULT_BOT_SETTINGS["TOP_N_COINS"])),
        int(safe_float(bot_config.get("POLL_INTERVAL_SEC"), DEFAULT_BOT_SETTINGS["POLL_INTERVAL_SEC"])),
        safe_float(bot_config.get("SL_NEAREST_LEVEL_BUFFER_PCT"), DEFAULT_BOT_SETTINGS["SL_NEAREST_LEVEL_BUFFER_PCT"]),
        safe_float(bot_config.get("TP_HVN_OFFSET_PCT"), DEFAULT_BOT_SETTINGS["TP_HVN_OFFSET_PCT"]),
        int(safe_float(bot_config.get("VP_LOOKBACK_BARS"), DEFAULT_BOT_SETTINGS["VP_LOOKBACK_BARS"])),
        int(safe_float(bot_config.get("VP_NUM_BINS"), DEFAULT_BOT_SETTINGS["VP_NUM_BINS"])),
        safe_float(bot_config.get("MID_ZONE_LOW"), DEFAULT_BOT_SETTINGS["MID_ZONE_LOW"]) * 100.0,
        safe_float(bot_config.get("MID_ZONE_HIGH"), DEFAULT_BOT_SETTINGS["MID_ZONE_HIGH"]) * 100.0,
        safe_float(bot_config.get("PRE_ALERT_DISTANCE_PCT"), DEFAULT_BOT_SETTINGS["PRE_ALERT_DISTANCE_PCT"]),
    )


@app.callback(
    Output("bot-settings-status", "children"),
    Input("apply-bot-settings", "n_clicks"),
    State("bot-threshold-input", "value"),
    State("bot-min-gap-input", "value"),
    State("bot-fvg-lookback-input", "value"),
    State("bot-fvg-age-input", "value"),
    State("bot-fvg-min-mult-input", "value"),
    State("bot-fvg-max-active-input", "value"),
    State("bot-fvg-price-mode-input", "value"),
    State("bot-swing-lookback-input", "value"),
    State("bot-disp-toggle", "value"),
    State("bot-same-dir-toggle", "value"),
    State("bot-disp-mult-input", "value"),
    State("bot-disp-pct-input", "value"),
    State("bot-risk-input", "value"),
    State("bot-max-positions-input", "value"),
    State("bot-top-coins-input", "value"),
    State("bot-poll-input", "value"),
    State("bot-sl-buffer-input", "value"),
    State("bot-tp-buffer-input", "value"),
    State("bot-vp-lookback-input", "value"),
    State("bot-vp-bins-input", "value"),
    State("bot-mid-low-input", "value"),
    State("bot-mid-high-input", "value"),
    State("bot-prealert-distance-input", "value"),
    prevent_initial_call=True,
)
def apply_bot_settings(
    _: Optional[int],
    threshold: Optional[float],
    min_gap: Optional[float],
    fvg_lookback: Optional[float],
    fvg_max_age: Optional[float],
    fvg_min_mult: Optional[float],
    fvg_max_active: Optional[float],
    fvg_price_mode: Optional[str],
    swing_lookback: Optional[float],
    disp_toggle: Optional[list[str]],
    same_dir_toggle: Optional[list[str]],
    disp_mult: Optional[float],
    disp_pct: Optional[float],
    risk_percent: Optional[float],
    max_positions: Optional[float],
    top_n: Optional[float],
    poll_interval: Optional[float],
    sl_buffer: Optional[float],
    tp_buffer: Optional[float],
    vp_lookback: Optional[float],
    vp_bins: Optional[float],
    mid_low: Optional[float],
    mid_high: Optional[float],
    prealert_distance: Optional[float],
) -> str:
    config = provider.load_config()
    config["bot"] = dict(config.get("bot", {})) | {
        "FVG_PERCENTILE_THRESHOLD": safe_float(threshold, DEFAULT_BOT_SETTINGS["FVG_PERCENTILE_THRESHOLD"]),
        "MIN_GAP_PERCENT": safe_float(min_gap, DEFAULT_BOT_SETTINGS["MIN_GAP_PERCENT"]),
        "FVG_LOOKBACK": int(safe_float(fvg_lookback, DEFAULT_BOT_SETTINGS["FVG_LOOKBACK"])),
        "FVG_MAX_AGE_DAYS": int(safe_float(fvg_max_age, DEFAULT_BOT_SETTINGS["FVG_MAX_AGE_DAYS"])),
        "FVG_MIN_GAP_MULT": safe_float(fvg_min_mult, DEFAULT_BOT_SETTINGS["FVG_MIN_GAP_MULT"]),
        "FVG_MAX_ACTIVE_PER_SIDE": int(safe_float(fvg_max_active, DEFAULT_BOT_SETTINGS["FVG_MAX_ACTIVE_PER_SIDE"])),
        "FVG_PRICE_MODE": str(fvg_price_mode or DEFAULT_BOT_SETTINGS["FVG_PRICE_MODE"]).lower(),
        "SWING_LOOKBACK_BARS": int(safe_float(swing_lookback, DEFAULT_BOT_SETTINGS["SWING_LOOKBACK_BARS"])),
        "FVG_REQUIRE_DISPLACEMENT": bool(disp_toggle),
        "FVG_REQUIRE_SAME_DIRECTION": bool(same_dir_toggle),
        "FVG_DISPLACEMENT_BODY_MULT": safe_float(disp_mult, DEFAULT_BOT_SETTINGS["FVG_DISPLACEMENT_BODY_MULT"]),
        "FVG_DISPLACEMENT_BODY_PCT": safe_float(disp_pct, DEFAULT_BOT_SETTINGS["FVG_DISPLACEMENT_BODY_PCT"]),
        "RISK_PER_TRADE": safe_float(risk_percent, DEFAULT_BOT_SETTINGS["RISK_PER_TRADE"] * 100.0) / 100.0,
        "MAX_CONCURRENT_POSITIONS": int(safe_float(max_positions, DEFAULT_BOT_SETTINGS["MAX_CONCURRENT_POSITIONS"])),
        "TOP_N_COINS": max(40, int(safe_float(top_n, DEFAULT_BOT_SETTINGS["TOP_N_COINS"]))),
        "POLL_INTERVAL_SEC": int(safe_float(poll_interval, DEFAULT_BOT_SETTINGS["POLL_INTERVAL_SEC"])),
        "SL_NEAREST_LEVEL_BUFFER_PCT": safe_float(sl_buffer, DEFAULT_BOT_SETTINGS["SL_NEAREST_LEVEL_BUFFER_PCT"]),
        "TP_HVN_OFFSET_PCT": safe_float(tp_buffer, DEFAULT_BOT_SETTINGS["TP_HVN_OFFSET_PCT"]),
        "VP_LOOKBACK_BARS": int(safe_float(vp_lookback, DEFAULT_BOT_SETTINGS["VP_LOOKBACK_BARS"])),
        "VP_NUM_BINS": int(safe_float(vp_bins, DEFAULT_BOT_SETTINGS["VP_NUM_BINS"])),
        "MID_ZONE_LOW": safe_float(mid_low, DEFAULT_BOT_SETTINGS["MID_ZONE_LOW"] * 100.0) / 100.0,
        "MID_ZONE_HIGH": safe_float(mid_high, DEFAULT_BOT_SETTINGS["MID_ZONE_HIGH"] * 100.0) / 100.0,
        "PRE_ALERT_DISTANCE_PCT": safe_float(prealert_distance, DEFAULT_BOT_SETTINGS["PRE_ALERT_DISTANCE_PCT"]),
    }
    provider.save_config(config)
    return (
        f"Bot settings saved at {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC. "
        "The running bot will reload them on the next cycle and restart scanning with the new values."
    )


@app.callback(
    Output("charts-container", "children"),
    Input("state-store", "data"),
    Input("selected-symbols-store", "data"),
    Input("timeframe-selector", "value"),
    Input("chart-toggle-checklist", "value"),
    Input("threshold-input", "value"),
    Input("settings-min-gap-input", "value"),
    Input("risk-input", "value"),
    Input("bullish-color-input", "value"),
    Input("bearish-color-input", "value"),
    Input("zone-status-filter", "value"),
    Input("min-score-filter", "value"),
    Input("min-gap-filter", "value"),
)
def render_charts(
    state: dict[str, Any],
    selected_symbols: list[str],
    timeframe: str,
    chart_toggles: list[str],
    threshold: Optional[float],
    settings_min_gap: Optional[float],
    risk_percent: Optional[float],
    bullish_color: str,
    bearish_color: str,
    status_filter: str,
    min_score_filter: Optional[float],
    min_gap_filter: Optional[float],
) -> Any:
    selected_symbols = (selected_symbols or [])[:MAX_VISIBLE_CHARTS]
    if not selected_symbols:
        return build_placeholder_card("No Charts Selected", "Choose one or more symbols from the left sidebar.")

    chart_toggles = chart_toggles or []
    show_zones = "zones" in chart_toggles
    show_levels = "levels" in chart_toggles
    show_labels = "labels" in chart_toggles
    show_mitigated = "mitigated" in chart_toggles
    bot_timeframe = str(state.get("settings", {}).get("timeframe", "15m"))
    bot_settings = current_bot_settings()

    chart_settings = {
        "threshold": safe_float(threshold, DEFAULT_SETTINGS["threshold"]),
        "min_gap_percent": safe_float(settings_min_gap, DEFAULT_SETTINGS["min_gap_percent"]),
        "risk_percent": safe_float(risk_percent, DEFAULT_SETTINGS["risk_percent"]),
        "bullish_color": bullish_color or DEFAULT_SETTINGS["bullish_color"],
        "bearish_color": bearish_color or DEFAULT_SETTINGS["bearish_color"],
        "entry_color": DEFAULT_SETTINGS["entry_color"],
        "sl_color": DEFAULT_SETTINGS["sl_color"],
        "tp_color": DEFAULT_SETTINGS["tp_color"],
    }
    score_floor = safe_float(min_score_filter, 0.0)
    gap_floor = safe_float(min_gap_filter, 0.0)
    positions = state.get("active_positions", [])
    columns: list[Any] = []

    for symbol in selected_symbols:
        df, warning = provider.fetch_ohlcv(symbol, timeframe, limit=220)
        snapshot = state.get("symbols", {}).get(symbol, {})
        zones = [dict(zone) for zone in snapshot.get("zones", [])]
        if not df.empty:
            last_price = safe_float(df["close"].iloc[-1], safe_float(snapshot.get("last_price")))
        else:
            last_price = safe_float(snapshot.get("last_price"))
        for zone in zones:
            zone.setdefault("source", "bot")
            zone["last_price"] = last_price
            gap_size = safe_float(zone.get("high")) - safe_float(zone.get("low"))
            if gap_size > 0:
                zone["position_pct"] = ((last_price - safe_float(zone.get("low"))) / gap_size) * 100.0
                zone.setdefault("entry_band_low", safe_float(zone.get("low")) + gap_size * 0.40)
                zone.setdefault("entry_band_high", safe_float(zone.get("low")) + gap_size * 0.60)
            if not zone.get("entry_price") and not df.empty:
                zone.update(build_trade_plan_from_zone(zone, df))

        # Fallback mode: when bot snapshot has no zones, detect local FVG zones
        # directly from chart candles so the dashboard still shows potential setups.
        if not zones and not df.empty:
            zones = detect_local_fvgs(
                df=df,
                symbol=symbol,
                timeframe=timeframe,
                threshold=chart_settings["threshold"],
                min_gap_percent=chart_settings["min_gap_percent"],
                last_price=last_price,
                bot_settings=bot_settings,
            )

        zones = filter_zones(zones, score_floor, gap_floor, status_filter)
        if not show_mitigated:
            zones = [zone for zone in zones if zone.get("status") != "mitigated"]

        figure = build_chart_figure(
            symbol,
            timeframe,
            df,
            zones,
            positions,
            chart_settings,
            show_zones=show_zones,
            show_levels=show_levels,
            show_labels=show_labels,
        )

        badges = [
            dbc.Badge(short_symbol(symbol), color="primary", className="me-2"),
            dbc.Badge(timeframe, color="secondary", className="me-2"),
            dbc.Badge(f"{len(zones)} zones", color="dark", className="me-2"),
        ]
        source_set = {str(zone.get("source", "bot")) for zone in zones}
        if zones and source_set == {"bot"}:
            badges.append(
                dbc.Badge(
                    f"{bot_timeframe} bot zones",
                    color="success",
                    className="me-2",
                )
            )
        elif zones and "dashboard-local" in source_set:
            badges.append(
                dbc.Badge(
                    f"{timeframe} local detection",
                    color="warning",
                    text_color="dark",
                    className="me-2",
                )
            )
        if warning:
            badges.append(dbc.Badge("cached candles", color="warning", text_color="dark"))

        columns.append(
            dbc.Col(
                dbc.Card(
                    [
                        dbc.CardHeader(badges, className="border-0", style={"background": "transparent"}),
                        dbc.CardBody(
                            dcc.Graph(
                                figure=figure,
                                config={
                                    "displaylogo": False,
                                    "responsive": True,
                                    "displayModeBar": True,
                                    "scrollZoom": True,
                                    "doubleClick": "reset+autosize",
                                },
                                className="chart-panel",
                            ),
                            className="pt-0",
                        ),
                    ],
                    className="panel-card border-0 mb-3",
                ),
                xl=12 if len(selected_symbols) == 1 else 6 if len(selected_symbols) == 2 else 4,
                md=12 if len(selected_symbols) == 1 else 6,
                sm=12,
            )
        )

    return dbc.Row(columns, className="g-3")


def build_local_pending_signals(
    state: dict[str, Any],
    selected_symbols: list[str],
    timeframe: str,
    threshold: float,
    min_gap_percent: float,
    max_symbols: int = 18,
    max_items: int = 20,
) -> list[dict[str, Any]]:
    """
    Dashboard-side fallback scanner.
    If bot snapshot has no pending setups, compute potential setups directly
    from current candles so user still sees near-entry opportunities.
    """
    # Dashboard fallback is intentionally less strict than bot pre-alerts,
    # so traders can see a watchlist before price gets very close.

    primary_symbols = selected_symbols or []
    secondary_symbols = dashboard_symbol_pool(state)
    ordered_symbols = [symbol for symbol in (primary_symbols + secondary_symbols) if symbol]
    dedup_symbols: list[str] = []
    for symbol in ordered_symbols:
        if symbol not in dedup_symbols:
            dedup_symbols.append(symbol)
    symbols = dedup_symbols[:max_symbols]
    bot_settings = current_bot_settings()

    pending: list[dict[str, Any]] = []

    for symbol in symbols:
        df, _warning = provider.fetch_ohlcv(symbol, timeframe, limit=220)
        if df.empty:
            continue
        last_price = safe_float(df["close"].iloc[-1])
        zones = detect_local_fvgs(
            df=df,
            symbol=symbol,
            timeframe=timeframe,
            threshold=threshold,
            min_gap_percent=min_gap_percent,
            last_price=last_price,
            bot_settings=bot_settings,
        )

        for zone in zones:
            if str(zone.get("status", "active")) not in {"active", "partial"}:
                continue
            quality_score = safe_float(zone.get("quality_score"))
            if quality_score < safe_float(bot_settings.get("FVG_QUALITY_MIN_SCORE"), 0.0):
                continue
            rr = safe_float(zone.get("risk_reward"))
            if rr < 1.0:
                continue

            pos_pct = safe_float(zone.get("position_pct"), -999.0)
            in_mid_zone = 40.0 <= pos_pct <= 60.0
            distance_to_mid = 0.0
            if not in_mid_zone:
                distance_to_mid = min(abs(pos_pct - 40.0), abs(pos_pct - 60.0))
            direction = str(zone.get("direction", "")).lower()
            signal_side = "LONG" if direction == "bullish" else "SHORT"
            composite_score = (
                safe_float(zone.get("score")) * 0.50 +
                quality_score * 0.35 +
                (100.0 if in_mid_zone else max(0.0, 100.0 - distance_to_mid)) * 0.15
            )
            if composite_score < safe_float(bot_settings.get("SIGNAL_MIN_COMPOSITE_SCORE"), 0.0):
                continue
            pending.append({
                "symbol": symbol,
                "short_symbol": short_symbol(symbol),
                "signal_side": signal_side,
                "direction": direction,
                "score": composite_score,
                "raw_score": safe_float(zone.get("score")),
                "quality_score": quality_score,
                "gap_pct": safe_float(zone.get("gap_pct")),
                "entry_price": safe_float(zone.get("entry_price")),
                "sl_price": safe_float(zone.get("sl_price")),
                "tp_price": safe_float(zone.get("tp_price")),
                "risk_reward": rr,
                "position_pct": pos_pct,
                "formed_at": zone.get("formed_at"),
                "timestamp": int(zone.get("timestamp", 0) or 0),
                "distance_to_mid": distance_to_mid,
                "entry_status": "ready-mid-zone" if in_mid_zone else "watching-mid-zone",
                "status": "ready" if in_mid_zone else "pending",
                "source": "dashboard-local",
            })

    pending.sort(
        key=lambda item: (
            1 if item.get("status") == "ready" else 0,
            safe_float(item.get("score")),
            -safe_float(item.get("distance_to_mid")),
        ),
        reverse=True,
    )
    for rank, item in enumerate(pending[:max_items], start=1):
        item["rank"] = rank
    return pending[:max_items]


@app.callback(
    Output("right-tab-content", "children"),
    Input("right-tabs", "active_tab"),
    Input("state-store", "data"),
    Input("selected-symbols-store", "data"),
    Input("timeframe-selector", "value"),
    Input("threshold-input", "value"),
    Input("settings-min-gap-input", "value"),
)
def render_right_tab(
    active_tab: str,
    state: dict[str, Any],
    selected_symbols: list[str],
    timeframe: str,
    threshold: Optional[float],
    settings_min_gap: Optional[float],
) -> Any:
    if active_tab == "positions-tab":
        positions = state.get("active_positions", [])
        if not positions:
            return html.Div("No active positions from the bot right now.", className="text-secondary small")
        return [build_position_card(position) for position in positions]

    if active_tab == "log-tab":
        return html.Pre(provider.tail_log(), className="mono-log")

    pending = sorted(state.get("pending_signals", []), key=lambda item: safe_float(item.get("score")), reverse=True)
    if not pending:
        fallback = build_local_pending_signals(
            state=state,
            selected_symbols=selected_symbols or [],
            timeframe=str(timeframe or state.get("settings", {}).get("timeframe", "15m")),
            threshold=safe_float(threshold, DEFAULT_SETTINGS["threshold"]),
            min_gap_percent=safe_float(settings_min_gap, DEFAULT_SETTINGS["min_gap_percent"]),
        )
        if not fallback:
            return html.Div("No pending signals right now (bot + local fallback).", className="text-secondary small")
        return [
            html.Div(
                "Fallback mode: setups detected locally by dashboard (bot snapshot is empty).",
                className="small text-warning mb-2",
            ),
            *[build_signal_card(signal) for signal in fallback],
        ]
    return [build_signal_card(signal) for signal in pending[:20]]


def collect_export_rows(
    state: dict[str, Any],
    selected_symbols: list[str],
    timeframe: str,
    threshold: float,
    min_gap_percent: float,
    min_score_filter: float,
    min_gap_filter: float,
    status_filter: str,
) -> list[dict[str, Any]]:
    symbols = selected_symbols or dashboard_symbol_pool(state)
    rows: list[dict[str, Any]] = []

    for symbol in symbols:
        snapshot = state.get("symbols", {}).get(symbol, {})
        zones = [dict(zone) for zone in snapshot.get("zones", [])]

        rows.extend(filter_zones(zones, min_score_filter, min_gap_filter, status_filter))

    return rows


def build_csv_export(rows: list[dict[str, Any]]) -> str:
    buffer = StringIO()
    fieldnames = [
        "symbol",
        "short_symbol",
        "timeframe",
        "direction",
        "status",
        "formed_at",
        "low",
        "high",
        "mid_price",
        "entry_price",
        "sl_price",
        "tp_price",
        "gap_pct",
        "score",
        "position_pct",
        "risk_reward",
        "source",
    ]
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({name: row.get(name) for name in fieldnames})
    return buffer.getvalue()


def build_pine_export(rows: list[dict[str, Any]]) -> str:
    lines = [
        "//@version=5",
        'indicator("FVG Zones Export", overlay=true, max_boxes_count=500, max_lines_count=500)',
        "if barstate.islast",
    ]

    for row in rows:
        color_name = "color.lime" if row.get("direction") == "bullish" else "color.red"
        transparency = 82 if row.get("status") == "active" else 90
        left = int(row.get("timestamp", 0) or 0)
        low = safe_float(row.get("low"))
        high = safe_float(row.get("high"))
        entry = safe_float(row.get("entry_price"))
        sl = safe_float(row.get("sl_price"))
        tp = safe_float(row.get("tp_price"))
        label_text = f"{row.get('short_symbol')} {row.get('direction', '').upper()} {safe_float(row.get('score')):.1f}"
        lines.extend(
            [
                f"    box.new(left={left}, top={high}, right=timenow, bottom={low}, xloc=xloc.bar_time, bgcolor=color.new({color_name}, {transparency}), border_color={color_name})",
                f"    line.new(x1={left}, y1={entry}, x2=timenow, y2={entry}, xloc=xloc.bar_time, color=color.yellow, style=line.style_dashed)",
                f"    line.new(x1={left}, y1={sl}, x2=timenow, y2={sl}, xloc=xloc.bar_time, color=color.red)",
                f"    line.new(x1={left}, y1={tp}, x2=timenow, y2={tp}, xloc=xloc.bar_time, color=color.lime)",
                f'    label.new(x={left}, y={high}, text="{label_text}", xloc=xloc.bar_time, style=label.style_label_down, textcolor=color.white, color=color.new({color_name}, 65))',
            ]
        )

    if len(lines) == 3:
        lines.append('    label.new(x=timenow, y=close, text="No zones to export", xloc=xloc.bar_time)')
    return "\n".join(lines)


@app.callback(
    Output("export-download", "data"),
    Input("export-csv", "n_clicks"),
    Input("export-pine", "n_clicks"),
    State("state-store", "data"),
    State("selected-symbols-store", "data"),
    State("timeframe-selector", "value"),
    State("threshold-input", "value"),
    State("settings-min-gap-input", "value"),
    State("min-score-filter", "value"),
    State("min-gap-filter", "value"),
    State("zone-status-filter", "value"),
    prevent_initial_call=True,
)
def export_current_view(
    _: Optional[int],
    __: Optional[int],
    state: dict[str, Any],
    selected_symbols: list[str],
    timeframe: str,
    threshold: Optional[float],
    settings_min_gap: Optional[float],
    min_score_filter: Optional[float],
    min_gap_filter: Optional[float],
    status_filter: str,
):
    rows = collect_export_rows(
        state=state,
        selected_symbols=selected_symbols or [],
        timeframe=timeframe,
        threshold=safe_float(threshold, DEFAULT_SETTINGS["threshold"]),
        min_gap_percent=safe_float(settings_min_gap, DEFAULT_SETTINGS["min_gap_percent"]),
        min_score_filter=safe_float(min_score_filter, 0.0),
        min_gap_filter=safe_float(min_gap_filter, 0.0),
        status_filter=status_filter,
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    if ctx.triggered_id == "export-pine":
        return dcc.send_string(build_pine_export(rows), filename=f"fvg_zones_{timeframe}_{timestamp}.pine")
    return dcc.send_string(build_csv_export(rows), filename=f"fvg_zones_{timeframe}_{timestamp}.csv")


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8050)
