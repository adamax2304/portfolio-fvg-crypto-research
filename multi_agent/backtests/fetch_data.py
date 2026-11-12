"""
multi_agent/backtests/fetch_data.py

Download 90-day OHLCV cache for all 20 backtest symbols.
- Downloads 15m and 1h data (90 days + 500-bar warmup)
- Saves as Parquet files in data_cache/
- Skips symbols whose cache file is < 24h old (incremental updates only)
- Falls back gracefully if a symbol fails

Usage
-----
    py -3 multi_agent/backtests/fetch_data.py
"""

from __future__ import annotations

import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)

# ─── Config ───────────────────────────────────────────────────────────────────

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BACKTEST_DAYS  = 90       # download 90 days of data
EXTRA_BARS     = 500      # warmup bars on top
CACHE_TTL_SECS = 86_400   # re-download if cache older than 24 h

BACKTEST_SYMBOLS = [
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "DOGE/USDT:USDT",
    "ADA/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "POL/USDT:USDT",
    "DOT/USDT:USDT",
    "UNI/USDT:USDT",
    "ATOM/USDT:USDT",
    "NEAR/USDT:USDT",
    "APT/USDT:USDT",
    "ARB/USDT:USDT",
    "OP/USDT:USDT",
    "INJ/USDT:USDT",
    "SUI/USDT:USDT",
    "1000PEPE/USDT:USDT",
]

TIMEFRAMES = ["15m", "1h", "4h"]

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _symbol_safe(symbol: str) -> str:
    """Convert symbol to filesystem-safe name, e.g. BTC/USDT:USDT -> BTC_USDT_USDT"""
    return symbol.replace("/", "_").replace(":", "_")


def _cache_path(symbol: str, timeframe: str) -> Path:
    return CACHE_DIR / f"{_symbol_safe(symbol)}_{timeframe}_90d.parquet"


def _is_cache_fresh(path: Path) -> bool:
    """Check parquet or CSV fallback."""
    for p in [path, path.with_suffix(".csv")]:
        if p.exists():
            age = time.time() - p.stat().st_mtime
            if age < CACHE_TTL_SECS:
                return True
    return False


def _download_ccxt(symbol: str, timeframe: str, days: int, extra_bars: int = 500) -> pd.DataFrame:
    """
    Download OHLCV from Bybit via CCXT (sync).
    Paginates until full history is fetched.
    """
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed. Run: py -3 -m pip install ccxt")
        return pd.DataFrame()

    bars_per_day = {
        "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
        "30m": 48, "1h": 24, "4h": 6, "1d": 1,
    }
    bpd          = bars_per_day.get(timeframe, 96)
    total_needed = days * bpd + extra_bars
    bar_ms       = {
        "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
        "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
    }.get(timeframe, 900_000)

    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = now_ms - (days + extra_bars // bpd + 3) * 86_400_000

    try:
        exchange    = ccxt.bybit({"options": {"defaultType": "swap"}})
        all_ohlcv   = []
        fetch_since = since_ms

        for _ in range(50):   # up to 50 pages (50 000 candles)
            batch = exchange.fetch_ohlcv(symbol, timeframe, since=fetch_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= now_ms - bar_ms * 2:
                break
            fetch_since = last_ts + 1
            if len(all_ohlcv) >= total_needed:
                break
            time.sleep(0.05)   # polite rate-limiting

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(df) > total_needed:
            df = df.tail(total_needed).reset_index(drop=True)
        return df

    except Exception as exc:
        logger.warning("CCXT fetch failed for %s %s: %s", symbol, timeframe, exc)
        return pd.DataFrame()


# ─── Main ─────────────────────────────────────────────────────────────────────

def fetch_all(force: bool = False) -> dict[str, dict[str, pd.DataFrame]]:
    """
    Download all symbols × timeframes.
    Returns nested dict: result[symbol][timeframe] = DataFrame

    If cache is fresh (< 24 h old) AND force=False, load from disk instead.
    """
    result: dict[str, dict[str, pd.DataFrame]] = {}

    total = len(BACKTEST_SYMBOLS) * len(TIMEFRAMES)
    done  = 0

    for symbol in BACKTEST_SYMBOLS:
        result[symbol] = {}
        for tf in TIMEFRAMES:
            done += 1
            path = _cache_path(symbol, tf)

            if not force and _is_cache_fresh(path):
                loaded = False
                for cache_file, reader in [(path, pd.read_parquet), (path.with_suffix(".csv"), pd.read_csv)]:
                    if cache_file.exists():
                        try:
                            df = reader(cache_file)
                            logger.info("[%d/%d] %s %s — loaded from cache (%d rows)",
                                        done, total, symbol, tf, len(df))
                            result[symbol][tf] = df
                            loaded = True
                            break
                        except Exception as exc:
                            logger.warning("Cache read failed for %s %s from %s: %s",
                                           symbol, tf, cache_file.suffix, exc)
                if loaded:
                    continue

            logger.info("[%d/%d] %s %s — downloading ...", done, total, symbol, tf)
            df = _download_ccxt(symbol, tf, BACKTEST_DAYS, EXTRA_BARS)

            if df.empty or len(df) < 200:
                logger.warning("  -> FAILED or too short (%d rows), skipping", len(df))
                result[symbol][tf] = pd.DataFrame()
                continue

            saved = False
            try:
                df.to_parquet(path, index=False)
                logger.info("  -> %d rows saved to %s", len(df), path.name)
                saved = True
            except Exception as exc:
                logger.warning("  -> Parquet save failed (%s), trying CSV ...", exc)

            if not saved:
                csv_path = path.with_suffix(".csv")
                try:
                    df.to_csv(csv_path, index=False)
                    logger.info("  -> %d rows saved to %s (CSV fallback)", len(df), csv_path.name)
                except Exception as exc2:
                    logger.warning("  -> CSV save also failed: %s", exc2)

            result[symbol][tf] = df

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download 90-day OHLCV cache")
    parser.add_argument("--force", action="store_true",
                        help="Force re-download even if cache is fresh")
    args = parser.parse_args()

    logger.info("=== fetch_data.py — 90-day cache builder ===")
    logger.info("Cache dir: %s", CACHE_DIR)
    logger.info("Symbols: %d  |  Timeframes: %s  |  Days: %d",
                len(BACKTEST_SYMBOLS), TIMEFRAMES, BACKTEST_DAYS)
    if args.force:
        logger.info("--force flag set: re-downloading all data")

    data = fetch_all(force=args.force)

    ok   = sum(1 for s in data.values() for df in s.values() if not df.empty)
    fail = sum(1 for s in data.values() for df in s.values() if df.empty)
    logger.info("Done. %d/%d datasets ready, %d failed.", ok, ok + fail, fail)
    print(f"\nCache ready: {ok}/{ok+fail} datasets in {CACHE_DIR}")
