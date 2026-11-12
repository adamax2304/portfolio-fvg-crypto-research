"""
fetch_data_2y.py  --  Download 2-year OHLCV cache for TOP_50+ symbols
Saves as  data_cache/{sym}_{tf}_730d.parquet

Usage:
    py -3 multi_agent/backtests/fetch_data_2y.py
    py -3 multi_agent/backtests/fetch_data_2y.py --force

Notes:
    - 5m needs roughly 211 fetches of 1000 bars to cover 730d (+ warmup).
    - Older versions capped the loop at 100 fetches, which silently truncated
      low timeframes to ~347 days while still saving them as *_730d.parquet.
"""
from __future__ import annotations
import logging, sys, time, argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

CACHE_DIR    = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DAYS         = 730          # 2 years
EXTRA_BARS   = 600          # warmup
CACHE_AGE    = 3 * 86_400   # re-download if > 3 days old

# ── TOP_50 candidates ─────────────────────────────────────────────────────────
# Only established coins that have been in top-50 mcap at some point in 2024-2026
# AND should have 2y of Bybit data.
# Newer/meme coins (< 2023 listing) are intentionally excluded to reduce survivorship bias.
TOP_50_SYMBOLS = [
    # Core top-10 (always top-10 by mcap)
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "BNB/USDT:USDT",  "SOL/USDT:USDT",
    "XRP/USDT:USDT",  "DOGE/USDT:USDT", "ADA/USDT:USDT",  "TRX/USDT:USDT",
    "AVAX/USDT:USDT", "TON/USDT:USDT",
    # 11-20 (frequently top-20)
    "LINK/USDT:USDT", "DOT/USDT:USDT",  "NEAR/USDT:USDT", "ATOM/USDT:USDT",
    "UNI/USDT:USDT",  "APT/USDT:USDT",  "SUI/USDT:USDT",  "ARB/USDT:USDT",
    "OP/USDT:USDT",   "INJ/USDT:USDT",
    # 21-35 (LIVE_34 additions)
    "BCH/USDT:USDT",  "LTC/USDT:USDT",  "ICP/USDT:USDT",  "HBAR/USDT:USDT",
    "XLM/USDT:USDT",  "FIL/USDT:USDT",  "ALGO/USDT:USDT", "POL/USDT:USDT",
    "SEI/USDT:USDT",  "XMR/USDT:USDT",  "CRV/USDT:USDT",  "MNT/USDT:USDT",
    "XAUT/USDT:USDT", "ONDO/USDT:USDT", "1000PEPE/USDT:USDT",
    # 36-50 (additional established coins, top-50 in 2024-2025)
    "AAVE/USDT:USDT",    # DeFi blue chip, top 30-45
    "RENDER/USDT:USDT",  # AI/GPU, top 25-40 in 2025
    "LDO/USDT:USDT",     # Lido staking, top 30-45
    "STX/USDT:USDT",     # Bitcoin L2, top 30-40 in 2024
    "DYDX/USDT:USDT",    # DeFi perps, top 40-55
    "IMX/USDT:USDT",     # Gaming L2, top 35-50
    "GRT/USDT:USDT",     # The Graph, top 40-60
    "SAND/USDT:USDT",    # Metaverse, top 30-50 in 2024
    "GALA/USDT:USDT",    # Gaming, top 35-55 in 2024
    "FTM/USDT:USDT",     # Fantom (now Sonic), top 30-45
    "EGLD/USDT:USDT",    # MultiversX, top 35-55
    "WLD/USDT:USDT",     # Worldcoin, top 20-35 in 2025
    "PENDLE/USDT:USDT",  # Yield trading, top 40-60 in 2024-25
    "TAO/USDT:USDT",     # Bittensor AI, top 15-30 in 2025
    "1000BONK/USDT:USDT",# BONK meme, top 25-40 in 2024
]

TIMEFRAMES = ["1h", "5m", "15m", "4h", "1d"]

BAR_MS = {
    "1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
    "30m": 1_800_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
}
BARS_PER_DAY = {
    "1m": 1440, "3m": 480, "5m": 288, "15m": 96,
    "30m": 48, "1h": 24, "4h": 6, "1d": 1,
}


def _safe(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "_")


def _cache_path(sym: str, tf: str) -> Path:
    return CACHE_DIR / f"{_safe(sym)}_{tf}_730d.parquet"


def _is_fresh(path: Path) -> bool:
    for p in [path, path.with_suffix(".csv")]:
        if p.exists() and (time.time() - p.stat().st_mtime) < CACHE_AGE:
            return True
    return False


def _download(sym: str, tf: str) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed: py -3 -m pip install ccxt")
        return pd.DataFrame()

    bpd   = BARS_PER_DAY.get(tf, 96)
    bms   = BAR_MS.get(tf, 900_000)
    total = DAYS * bpd + EXTRA_BARS

    now_ms    = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms  = now_ms - (DAYS + EXTRA_BARS // bpd + 5) * 86_400_000

    try:
        ex = ccxt.bybit({"options": {"defaultType": "swap"}})
        all_ohlcv: list = []
        fetch_since = since_ms

        # Bybit/ccxt returns up to 1000 candles per call. Compute the number of
        # iterations from the requested total bars instead of hard-coding 100,
        # otherwise 5m data gets truncated to ~100k bars (~347 days).
        max_iters = max(8, (total + 999) // 1000 + 8)
        for _ in range(max_iters):
            batch = ex.fetch_ohlcv(sym, tf, since=fetch_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= now_ms - bms * 2:
                break
            fetch_since = last_ts + 1
            if len(all_ohlcv) >= total:
                break
            time.sleep(0.06)

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(df) > total:
            df = df.tail(total).reset_index(drop=True)
        return df

    except Exception as exc:
        logger.warning("  FAILED %s %s: %s", sym, tf, exc)
        return pd.DataFrame()


def fetch_all(symbols: list[str], timeframes: list[str], force: bool = False) -> None:
    total_tasks = len(symbols) * len(timeframes)
    done = 0
    ok = 0; fail = 0; skipped = 0

    for sym in symbols:
        for tf in timeframes:
            done += 1
            path = _cache_path(sym, tf)

            if not force and _is_fresh(path):
                logger.info("[%d/%d] SKIP (fresh)  %s %s", done, total_tasks, sym, tf)
                skipped += 1
                continue

            logger.info("[%d/%d] Downloading  %s %s ...", done, total_tasks, sym, tf)
            df = _download(sym, tf)

            min_rows = {"1d": 200, "1h": 300, "5m": 1000, "15m": 500}.get(tf, 200)
            if df.empty or len(df) < min_rows:
                logger.warning("  -> FAILED/short (%d rows)", len(df) if not df.empty else 0)
                fail += 1
                continue

            # Save parquet, fallback CSV
            try:
                df.to_parquet(path, index=False)
                logger.info("  -> OK  %d rows -> %s", len(df), path.name)
                ok += 1
            except Exception as e:
                csv_p = path.with_suffix(".csv")
                try:
                    df.to_csv(csv_p, index=False)
                    logger.info("  -> OK (csv)  %d rows -> %s", len(df), csv_p.name)
                    ok += 1
                except Exception as e2:
                    logger.error("  -> Save failed: %s | %s", e, e2)
                    fail += 1

    print(f"\nDone: {ok} saved, {skipped} fresh (skipped), {fail} failed")
    print(f"Cache dir: {CACHE_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download 2-year OHLCV cache for TOP_50")
    parser.add_argument("--force", action="store_true", help="Re-download even if fresh")
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=TIMEFRAMES,
        help=f"Subset of timeframes to fetch. Default: {' '.join(TIMEFRAMES)}",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=TOP_50_SYMBOLS,
        help="Optional subset of symbols, e.g. BTC/USDT:USDT ETH/USDT:USDT",
    )
    args = parser.parse_args()

    selected_tfs = []
    for tf in args.timeframes:
        if tf not in TIMEFRAMES:
            raise SystemExit(f"Unsupported timeframe: {tf}. Allowed: {', '.join(TIMEFRAMES)}")
        selected_tfs.append(tf)

    logger.info("=== fetch_data_2y.py  |  %d symbols x %d TFs x 730d ===",
                len(args.symbols), len(selected_tfs))
    logger.info("Cache: %s", CACHE_DIR)
    if args.force:
        logger.info("--force: re-downloading all")

    t0 = time.time()
    fetch_all(args.symbols, selected_tfs, force=args.force)
    logger.info("Total time: %.1f min", (time.time() - t0) / 60)
