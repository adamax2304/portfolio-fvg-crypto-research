"""
Fetch 5m OHLCV data for 180 days, all symbols.
Uses same infrastructure as fetch_data.py.
"""
import sys, time, logging
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

CACHE_DIR = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

def download_5m(symbol, days=180):
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt not installed")
        return pd.DataFrame()

    bpd = 288  # 5m bars per day
    total_needed = days * bpd + 500
    bar_ms = 300_000  # 5 min in ms
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (days + 3) * 86_400_000

    exchange = ccxt.bybit({"options": {"defaultType": "swap"}})
    all_ohlcv = []
    fetch_since = since_ms

    for page in range(120):  # 5m needs more pages
        try:
            batch = exchange.fetch_ohlcv(symbol, "5m", since=fetch_since, limit=1000)
        except Exception as e:
            logger.warning("  page %d failed: %s", page, e)
            time.sleep(1)
            continue
        if not batch:
            break
        all_ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= now_ms - bar_ms * 2:
            break
        fetch_since = last_ts + 1
        if len(all_ohlcv) >= total_needed:
            break
        time.sleep(0.05)

    if not all_ohlcv:
        return pd.DataFrame()

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    if len(df) > total_needed:
        df = df.tail(total_needed).reset_index(drop=True)
    return df

def main():
    t0 = time.time()
    for i, sym in enumerate(SYMBOLS):
        safe = sym.replace("/", "_").replace(":", "_")
        path = CACHE_DIR / f"{safe}_5m_180d.parquet"
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600
            if age_h < 24:
                logger.info("[%d/%d] %s 5m — cached (%.0fh old)", i+1, len(SYMBOLS), sym, age_h)
                continue
        logger.info("[%d/%d] %s 5m — downloading ...", i+1, len(SYMBOLS), sym)
        df = download_5m(sym, 180)
        if df.empty or len(df) < 1000:
            logger.warning("  FAILED or short (%d rows)", len(df))
            continue
        df.to_parquet(path, index=False)
        logger.info("  -> %d rows saved", len(df))
    logger.info("Done in %.0f sec", time.time() - t0)

if __name__ == "__main__":
    main()
