"""Download Bybit funding rate history for baseline 27 symbols."""
from __future__ import annotations
import argparse
import logging, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data_cache"
DAYS = 730  # try 2 years

SYMBOLS = ["ADA","ALGO","APT","ARB","ATOM","AVAX","BCH","BNB","BTC",
           "DOGE","DOT","ETH","FIL","HBAR","ICP","INJ","LINK","LTC",
           "NEAR","OP","SOL","SUI","TON","TRX","UNI","XLM","XRP"]


def fetch(sym: str, force: bool = False):
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt missing"); return pd.DataFrame()
    out_path = CACHE_DIR / f"{sym}_USDT_USDT_funding_730d.parquet"
    if (not force) and out_path.exists() and (time.time() - out_path.stat().st_mtime) < 3 * 86400:
        return None  # already fresh
    full_sym = f"{sym}/USDT:USDT"
    try:
        ex = ccxt.bybit({"options": {"defaultType": "swap"}})
        now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
        since = now_ms - DAYS * 86_400_000
        all_records = []
        cursor_since = since
        for _ in range(60):  # up to 60 batches × 200 = 12000 funding records
            batch = ex.fetch_funding_rate_history(full_sym, since=cursor_since, limit=200)
            if not batch: break
            all_records.extend(batch)
            last_ts = batch[-1]["timestamp"]
            if last_ts >= now_ms - 8 * 3_600_000: break
            cursor_since = last_ts + 1
            time.sleep(0.1)
        if not all_records:
            return pd.DataFrame()
        rows = [{"timestamp": r["timestamp"],
                  "funding_rate": float(r["fundingRate"])} for r in all_records]
        df = pd.DataFrame(rows).drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        df.to_parquet(out_path, index=False)
        logger.info(f"  saved {out_path.name} ({len(df)} records)")
        return df
    except Exception as e:
        logger.warning(f"  FAIL {sym}: {e}")
        return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description="Download Bybit funding history")
    parser.add_argument("--force", action="store_true", help="Re-download even if file is fresh")
    parser.add_argument("--symbols", nargs="+", default=SYMBOLS, help="Subset of base symbols, e.g. BTC ETH SOL")
    args = parser.parse_args()

    ok = 0; fail = 0; skip = 0
    for sym in args.symbols:
        r = fetch(sym, force=args.force)
        if r is None: skip += 1
        elif r.empty: fail += 1
        else: ok += 1
    logger.info(f"DONE  ok={ok} fail={fail} skip={skip}")


if __name__ == "__main__":
    main()
