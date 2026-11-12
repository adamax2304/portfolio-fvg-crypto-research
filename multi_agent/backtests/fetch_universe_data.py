"""
fetch_universe_data.py — Скачивает 730d данных для ~50 символов
Только 4h + 1d TF (V7 не нужны 1h/5m/15m)

Usage:
    py -3 multi_agent/backtests/fetch_universe_data.py
    py -3 multi_agent/backtests/fetch_universe_data.py --force
"""
from __future__ import annotations
import logging, sys, time, argparse
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)
log = logging.getLogger(__name__)

CACHE_DIR  = Path(__file__).parent / "data_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DAYS       = 730
EXTRA_BARS = 600          # тёплый старт
CACHE_AGE  = 3 * 86_400   # не перекачивать если свежее 3 дней

# V7 нужны только 4h и 1d
TIMEFRAMES = ["4h", "1d"]

BAR_MS = {"4h": 14_400_000, "1d": 86_400_000}
BARS_PER_DAY = {"4h": 6, "1d": 1}

# ── Universe ──────────────────────────────────────────────────────────────────
# Монеты которые бывали в топ-50 по объёму на Bybit в 2023-2026
# Включаем только монеты листингованные ДО 2023-07 (минимум 730d истории)
UNIVERSE = [
    # Топ-10 (всегда топ-10)
    "BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "TRX/USDT:USDT",
    "AVAX/USDT:USDT", "TON/USDT:USDT",
    # 11-20
    "LINK/USDT:USDT", "DOT/USDT:USDT", "NEAR/USDT:USDT", "ATOM/USDT:USDT",
    "UNI/USDT:USDT", "APT/USDT:USDT", "SUI/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT",
    # 21-35
    "BCH/USDT:USDT", "LTC/USDT:USDT", "ICP/USDT:USDT", "HBAR/USDT:USDT",
    "XLM/USDT:USDT", "FIL/USDT:USDT", "ALGO/USDT:USDT",
    "CRV/USDT:USDT", "ONDO/USDT:USDT",
    "1000PEPE/USDT:USDT",
    # 36-55 — дополнительные, бывшие в топ-50 объёма
    "AAVE/USDT:USDT",    # DeFi blue chip
    "RENDER/USDT:USDT",  # AI/GPU, 2023+
    "LDO/USDT:USDT",     # Lido staking
    "STX/USDT:USDT",     # Bitcoin L2
    "IMX/USDT:USDT",     # Gaming L2
    "GRT/USDT:USDT",     # The Graph
    "SAND/USDT:USDT",    # Metaverse
    "GALA/USDT:USDT",    # Gaming
    "FTM/USDT:USDT",     # Fantom
    "EGLD/USDT:USDT",    # MultiversX
    "WLD/USDT:USDT",     # Worldcoin (листинг Jul-2023)
    "PENDLE/USDT:USDT",  # Yield trading
    "TAO/USDT:USDT",     # Bittensor AI
    "1000BONK/USDT:USDT",# BONK meme
    "SEI/USDT:USDT",     # Sei Network
    "JTO/USDT:USDT",     # Jito (листинг Dec-2023 — пропустим если мало данных)
    "PYTH/USDT:USDT",    # Pyth Network
    "TIA/USDT:USDT",     # Celestia
    "MANTA/USDT:USDT",   # Manta Network
    "JUP/USDT:USDT",     # Jupiter (Jan-2024 — мало истории, но пробуем)
]

# ─────────────────────────────────────────────────────────────────────────────

def _safe(sym: str) -> str:
    return sym.replace("/", "_").replace(":", "_")


def _cache_path(sym: str, tf: str) -> Path:
    return CACHE_DIR / f"{_safe(sym)}_{tf}_730d.parquet"


def _is_fresh(path: Path) -> bool:
    if path.exists() and (time.time() - path.stat().st_mtime) < CACHE_AGE:
        return True
    return False


def _download(sym: str, tf: str) -> pd.DataFrame:
    try:
        import ccxt
    except ImportError:
        log.error("ccxt not installed — py -3 -m pip install ccxt")
        return pd.DataFrame()

    bpd   = BARS_PER_DAY.get(tf, 6)
    bms   = BAR_MS.get(tf, 14_400_000)
    total = DAYS * bpd + EXTRA_BARS

    now_ms   = int(datetime.now(timezone.utc).timestamp() * 1000)
    since_ms = now_ms - (DAYS + EXTRA_BARS // bpd + 10) * 86_400_000

    try:
        ex = ccxt.bybit({"options": {"defaultType": "swap"}})
        all_ohlcv: list = []
        fetch_since = since_ms

        for _ in range(60):   # max 60k баров
            batch = ex.fetch_ohlcv(sym, tf, since=fetch_since, limit=1000)
            if not batch:
                break
            all_ohlcv.extend(batch)
            last_ts = batch[-1][0]
            if last_ts >= now_ms - bms * 2:
                break
            fetch_since = last_ts + 1
            time.sleep(0.07)

        if not all_ohlcv:
            return pd.DataFrame()

        df = pd.DataFrame(all_ohlcv,
                          columns=["timestamp", "open", "high", "low", "close", "volume"])
        df = (df.drop_duplicates("timestamp")
                .sort_values("timestamp")
                .reset_index(drop=True))
        if len(df) > total:
            df = df.tail(total).reset_index(drop=True)
        return df

    except Exception as exc:
        log.warning("  FAILED %s %s: %s", sym, tf, exc)
        return pd.DataFrame()


# Минимум строк для считать данные годными
MIN_ROWS = {"4h": 1200, "1d": 400}


def fetch_all(symbols: list[str], force: bool = False) -> dict[str, bool]:
    """Скачать данные, вернуть {sym: ok}."""
    total   = len(symbols) * len(TIMEFRAMES)
    done    = 0; ok = 0; fail = 0; skipped = 0

    sym_ok: dict[str, bool] = {}

    for sym in symbols:
        sym_good = True
        for tf in TIMEFRAMES:
            done += 1
            path = _cache_path(sym, tf)

            if not force and _is_fresh(path):
                log.info("[%d/%d] SKIP (fresh)  %-30s %s", done, total, sym, tf)
                skipped += 1
                continue

            log.info("[%d/%d] Downloading  %-30s %s ...", done, total, sym, tf)
            df = _download(sym, tf)

            min_r = MIN_ROWS.get(tf, 200)
            if df.empty or len(df) < min_r:
                log.warning("  -> SKIP (%d rows, min=%d)", len(df) if not df.empty else 0, min_r)
                fail += 1
                sym_good = False
                continue

            try:
                df.to_parquet(path, index=False)
                log.info("  -> OK  %d rows  %s", len(df), path.name)
                ok += 1
            except Exception as e:
                log.error("  -> Save error: %s", e)
                fail += 1
                sym_good = False

        sym_ok[sym] = sym_good

    log.info("=" * 60)
    log.info("Сохранено: %d  |  Пропущено (свежие): %d  |  Ошибок: %d",
             ok, skipped, fail)
    log.info("Cache: %s", CACHE_DIR)

    good = [s for s, v in sym_ok.items() if v]
    log.info("Символов с полными данными: %d / %d", len(good), len(symbols))
    return sym_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Перекачать даже свежие")
    args = parser.parse_args()

    log.info("=== Universe data fetch (%d символов × %s) ===",
             len(UNIVERSE), "/".join(TIMEFRAMES))
    t0 = time.time()
    fetch_all(UNIVERSE, force=args.force)
    log.info("Готово за %.1f мин", (time.time() - t0) / 60)
