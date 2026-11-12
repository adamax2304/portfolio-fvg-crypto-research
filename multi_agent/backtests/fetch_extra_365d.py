"""Download missing symbols for 365d to expand universe to 50+."""
from __future__ import annotations
import logging, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).parent / "data_cache"
DAYS = 365
EXTRA = 600

# Targets: missing or partial symbols. Pick coins with known 1y+ history on Bybit.
EXTRA_SYMBOLS = [
    "AAVE/USDT:USDT", "LDO/USDT:USDT", "STX/USDT:USDT", "DYDX/USDT:USDT",
    "IMX/USDT:USDT", "GRT/USDT:USDT", "SAND/USDT:USDT", "GALA/USDT:USDT",
    "FTM/USDT:USDT", "EGLD/USDT:USDT", "WLD/USDT:USDT", "PENDLE/USDT:USDT",
    "TAO/USDT:USDT", "1000BONK/USDT:USDT", "AXS/USDT:USDT", "MANA/USDT:USDT",
    "ENS/USDT:USDT", "JUP/USDT:USDT", "STRK/USDT:USDT", "FET/USDT:USDT",
    "RUNE/USDT:USDT", "ENJ/USDT:USDT", "SNX/USDT:USDT", "FLOW/USDT:USDT",
    "JTO/USDT:USDT", "PYTH/USDT:USDT", "TIA/USDT:USDT", "ETC/USDT:USDT",
    "FLOKI/USDT:USDT", "ORDI/USDT:USDT",
]

TFS = ["1d", "4h", "1h"]
BARS_PER_DAY = {"1d": 1, "4h": 6, "1h": 24, "15m": 96, "5m": 288}
BAR_MS = {"1d": 86400000, "4h": 14400000, "1h": 3600000, "15m": 900000, "5m": 300000}


def safe(s): return s.replace("/", "_").replace(":", "_")
def cpath(s, tf): return CACHE_DIR / f"{safe(s)}_{tf}_365d.parquet"


def download(sym, tf):
    try:
        import ccxt
    except ImportError:
        logger.error("ccxt missing"); return pd.DataFrame()
    bpd = BARS_PER_DAY[tf]; bms = BAR_MS[tf]
    total = DAYS * bpd + EXTRA
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    since = now_ms - (DAYS + EXTRA // bpd + 5) * 86_400_000
    try:
        ex = ccxt.bybit({"options": {"defaultType": "swap"}})
        all_o = []; fetch_since = since
        for _ in range(150):
            batch = ex.fetch_ohlcv(sym, tf, since=fetch_since, limit=1000)
            if not batch: break
            all_o.extend(batch)
            last = batch[-1][0]
            if last >= now_ms - bms * 2: break
            fetch_since = last + 1
            if len(all_o) >= total: break
            time.sleep(0.05)
        if not all_o: return pd.DataFrame()
        df = pd.DataFrame(all_o, columns=["timestamp","open","high","low","close","volume"])
        df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
        if len(df) > total: df = df.tail(total).reset_index(drop=True)
        return df
    except Exception as e:
        logger.warning(f"  FAIL {sym} {tf}: {e}")
        return pd.DataFrame()


def main():
    ok = 0; fail = 0; skip = 0
    for sym in EXTRA_SYMBOLS:
        for tf in TFS:
            p = cpath(sym, tf)
            if p.exists():
                skip += 1; continue
            logger.info(f"  fetch {sym} {tf}")
            df = download(sym, tf)
            if df.empty or len(df) < 100:
                fail += 1; continue
            df.to_parquet(p, index=False)
            logger.info(f"  saved {p.name} ({len(df)} bars)")
            ok += 1
    logger.info(f"DONE ok={ok}  fail={fail}  skip={skip}")


if __name__ == "__main__":
    main()
