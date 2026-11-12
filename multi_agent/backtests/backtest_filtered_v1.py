"""
backtest_filtered_v1.py
=======================
6-month backtest on the FILTERED universe:
  1. CoinGecko top-150 by market cap  (no low-cap manipulation)
  2. Bybit USDT-perp with >= $20M 24h volume  (liquid only)
  3. Top-100 by Bybit volume, then mcap cross-filter

Strategies tested (v10 params, unchanged):
  - BBS_1H_5M   : BBS signal on 1h, pullback entry on 5m, SL=ATR1.5, TP=3R
  - BBS_1H_15M  : BBS signal on 1h, next-bar on 15m, SL=ATR1.5, TP=2.5R
  - KC_1H_15M   : KC signal on 1h, next-bar on 15m, SL=ATR1.5, TP=2.0R

Output:
  - Per-strategy summary (PF, PnL, DD, WR, trades)
  - Per-symbol contribution table
  - Monthly PnL breakdown
  - Universe composition (which coins passed filters)
  - Comparison vs v10 20-symbol baseline
"""
from __future__ import annotations
import sys, json, time, urllib.request
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import ccxt

from multi_agent.backtests.backtest_v10 import (
    CACHE_DIR, START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
    prepare_tf_arrays, prepare_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
    compute_sl,
)

DAYS_TAG   = "180d"
MIN_VOLUME = 20_000_000    # $20M minimum 24h volume on Bybit
TOP_N_VOL  = 100           # take top 100 by Bybit volume before mcap filter
MCAP_TOP_N = 150           # CoinGecko top-N by market cap

# Original v10 20-symbol baseline (for comparison)
V10_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

# Stablecoins + wrapped tokens to skip
_SKIP_TICKERS = {
    "USDT","USDC","BUSD","DAI","TUSD","USDP","USDD","FDUSD","USDS","USDE",
    "PYUSD","FRAX","GUSD","LUSD","MIM","SUSD","USD1","USDF","USDG","USDTB",
    "USDY","RLUSD","BFUSD","CRVUSD","GHO","USDAI","USYC","BUIDL","PAXG","XAUT",
    "WBTC","WETH","WBNB","WSTETH","STETH","CBBTC","TBTC","RBTC",
    "LEO","WBT","OKB","BGB","KCS","GT","HTX",   # exchange tokens (skip)
    # NOTE: HYPE (Hyperliquid) removed from skip list — it IS listed on Bybit perps
    # and is rank #13 by mcap ($10B+). Was incorrectly excluded before.
}

# Bybit ticker -> CoinGecko symbol mapping
# Bybit uses "1000X" contracts for low-price tokens, and sometimes different names
_BYBIT_TICKER_MAP = {
    "PEPE":     "1000PEPE",   # CoinGecko: PEPE  -> Bybit: 1000PEPE
    "SHIB":     "1000SHIB",
    "FLOKI":    "1000FLOKI",
    "BONK":     "1000BONK",
    "NEIRO":    "1000NEIRO",
    "PUMP":     "PUMPFUN",    # CoinGecko: PUMP  -> Bybit: PUMPFUN (ticker mismatch bug fix)
}

# ===========================================================================
#  STEP 1: Build filtered symbol universe
# ===========================================================================

def fetch_coingecko_mcap(top_n=150) -> set[str]:
    """Return set of uppercase base tickers in top-N by market cap."""
    coins = []
    for page in (1, 2):
        if len(coins) >= top_n:
            break
        url = (
            "https://api.coingecko.com/api/v3/coins/markets"
            f"?vs_currency=usd&order=market_cap_desc&per_page=100&page={page}&sparkline=false"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "fvgbot-backtest/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=12) as r:
                coins += json.loads(r.read())
        except Exception as e:
            print(f"  CoinGecko page {page} error: {e}")
            break

    result = set()
    for c in coins[:top_n]:
        sym = c["symbol"].upper()
        if sym not in _SKIP_TICKERS:
            result.add(sym)
    return result


def fetch_bybit_universe(min_volume=20_000_000, top_n=100) -> list[str]:
    """Return list of Bybit USDT-perp symbols with volume >= min_volume, sorted by volume desc."""
    exchange = ccxt.bybit({"options": {"defaultType": "swap"}})
    print("  Fetching Bybit tickers (public)...")
    tickers = exchange.fetch_tickers()

    usdt_perp = []
    for sym, t in tickers.items():
        if not sym.endswith(":USDT"):
            continue
        vol = float(t.get("quoteVolume") or 0)
        if vol >= min_volume:
            usdt_perp.append((sym, vol))

    usdt_perp.sort(key=lambda x: x[1], reverse=True)
    return [sym for sym, _ in usdt_perp[:top_n]], {sym: vol for sym, vol in usdt_perp[:top_n]}


def build_filtered_universe() -> tuple[list[str], dict]:
    """Apply 3-level filter and return (symbol_list, info_dict)."""
    print("\n[1/4] Building filtered symbol universe...")

    print("  Fetching CoinGecko top-150 by market cap...")
    mcap_set = fetch_coingecko_mcap(MCAP_TOP_N)
    print(f"  CoinGecko: {len(mcap_set)} eligible tickers")

    bybit_syms, vol_map = fetch_bybit_universe(MIN_VOLUME, TOP_N_VOL)
    print(f"  Bybit: {len(bybit_syms)} symbols with >=${MIN_VOLUME/1e6:.0f}M volume")

    # Apply mcap cross-filter
    filtered = []
    skipped  = []
    for sym in bybit_syms:
        base = sym.split("/")[0]  # "BTC/USDT:USDT" -> "BTC"
        # Handle 1000X contracts: "1000PEPE" -> check "PEPE" in mcap
        cg_ticker = base
        for real, bybit_name in _BYBIT_TICKER_MAP.items():
            if base == bybit_name:
                cg_ticker = real
                break
        if cg_ticker in mcap_set or base in mcap_set:
            filtered.append(sym)
        else:
            skipped.append(base)

    print(f"\n  Filtered universe: {len(filtered)} symbols (removed {len(skipped)} low-mcap)")
    if skipped:
        print(f"  Excluded: {sorted(skipped)}")

    info = {s: {"volume_24h": vol_map.get(s, 0)} for s in filtered}
    return filtered, info


# ===========================================================================
#  STEP 2: Download OHLCV data for symbols not in cache
# ===========================================================================

_TF_MAP = {"5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
_BARS_NEEDED = {"5m": 52_560, "15m": 17_520, "1h": 4_380, "4h": 1_095, "1d": 182}
# 180 days worth of bars for each TF


def _cache_path(symbol: str, tf: str) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}_{tf}_{DAYS_TAG}.parquet"


def _needs_download(symbol: str, tfs: list[str]) -> list[str]:
    missing = []
    for tf in tfs:
        p = _cache_path(symbol, tf)
        if not p.exists():
            missing.append(tf)
    return missing


def download_ohlcv(symbol: str, tf: str, days: int = 182) -> pd.DataFrame | None:
    """Download OHLCV from Bybit via ccxt, return DataFrame."""
    exchange = ccxt.bybit({
        "options": {"defaultType": "swap"},
        "enableRateLimit": True,
    })
    limit = min(1000, _BARS_NEEDED.get(tf, 1000))
    since_ms = int((time.time() - days * 86400) * 1000)

    all_bars = []
    fetch_since = since_ms
    max_iters = 60

    for _ in range(max_iters):
        try:
            bars = exchange.fetch_ohlcv(symbol, tf, since=fetch_since, limit=limit)
        except Exception as e:
            print(f"    fetch_ohlcv error {symbol} {tf}: {e}")
            time.sleep(2)
            break
        if not bars:
            break
        all_bars.extend(bars)
        if len(bars) < limit:
            break
        fetch_since = bars[-1][0] + 1
        time.sleep(0.3)  # rate limit

    if not all_bars:
        return None

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    # Keep only last 180 days
    cutoff = int((time.time() - 182 * 86400) * 1000)
    df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
    return df if len(df) >= 100 else None


def ensure_data(symbols: list[str]) -> None:
    """Download missing OHLCV data for symbols not in cache."""
    tfs_needed = ["5m", "15m", "1h", "4h", "1d"]
    to_download = {}
    for sym in symbols:
        missing = _needs_download(sym, tfs_needed)
        if missing:
            to_download[sym] = missing

    if not to_download:
        print("[2/4] All data already cached.")
        return

    total = sum(len(v) for v in to_download.values())
    print(f"\n[2/4] Downloading data for {len(to_download)} new symbols ({total} TF files)...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    done = 0
    for sym, tfs in to_download.items():
        base = sym.split("/")[0]
        print(f"  Downloading {base} ({len(tfs)} TFs)...", end="", flush=True)
        for tf in tfs:
            df = download_ohlcv(sym, tf, days=185)
            if df is not None and len(df) > 50:
                p = _cache_path(sym, tf)
                df.to_parquet(p, index=False)
                done += 1
                print(f" {tf}({len(df)})", end="", flush=True)
            else:
                print(f" {tf}(FAIL)", end="", flush=True)
        print()
    print(f"  Done. {done}/{total} files downloaded.")


# ===========================================================================
#  STEP 3: Load data for all symbols
# ===========================================================================

def load_symbol_data(symbol: str) -> dict:
    """Load all TF data for a symbol from cache."""
    def _load(tf):
        p = _cache_path(symbol, tf)
        if not p.exists():
            # Try 365d fallback
            safe = symbol.replace("/", "_").replace(":", "_")
            for tag in ("365d", "90d"):
                p2 = CACHE_DIR / f"{safe}_{tf}_{tag}.parquet"
                if p2.exists():
                    try:
                        df = pd.read_parquet(p2)
                        # Trim to last 180 days
                        cutoff = int((time.time() - 182 * 86400) * 1000)
                        df = df[df["timestamp"] >= cutoff].reset_index(drop=True)
                        return df if len(df) > 50 else None
                    except Exception:
                        continue
            return None
        try:
            return pd.read_parquet(p)
        except Exception:
            return None

    d = {}
    d["df5"]  = _load("5m")
    d["df15"] = _load("15m")
    d["df1h"] = _load("1h")
    d["df4h"] = _load("4h")
    d["df1d"] = _load("1d")
    # Resample 30m from 15m if needed
    if d["df15"] is not None and len(d["df15"]) > 4:
        df = d["df15"].copy()
        ms_30m = 1_800_000
        df["ts_30m"] = (df["timestamp"] // ms_30m) * ms_30m
        agg = df.groupby("ts_30m").agg({
            "timestamp": "first", "open": "first", "high": "max",
            "low": "min", "close": "last", "volume": "sum",
        }).reset_index(drop=True)
        d["df30"] = agg
    return d


# ===========================================================================
#  STEP 4: Run strategies
# ===========================================================================

STRATEGIES = [
    {
        "id": "BBS_1H_5M",
        "name": "BBS 1h->5m Pullback",
        "htf_key": "df1h", "ltf_key": "df5", "gen_fn": "bbs",
        "sig_params": {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 3.0},
        "daily_filter": True,
        "max_pos": 4,
        "v10_ref": {"pf": 9.66, "pnl": 216.1, "dd": 1.9, "wr": 77.4, "n": 93},
    },
    {
        "id": "BBS_1H_15M",
        "name": "BBS 1h->15m Next Bar",
        "htf_key": "df1h", "ltf_key": "df15", "gen_fn": "bbs",
        "sig_params": {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
        "max_pos": 4,
        "v10_ref": {"pf": 3.92, "pnl": 592.5, "dd": 4.1, "wr": 59.7, "n": 288},
    },
    {
        "id": "KC_1H_15M",
        "name": "KC 1h->15m Next Bar",
        "htf_key": "df1h", "ltf_key": "df15", "gen_fn": "kc",
        "sig_params": {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                       "mom_bars": 6, "sl_atr_mult": 1.5},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.0},
        "daily_filter": False,
        "max_pos": 3,
        "v10_ref": {"pf": 5.76, "pnl": 93.0, "dd": 2.0, "wr": 68.7, "n": 83},
    },
]


def run_strategy_on_universe(strat: dict, all_data: dict[str, dict]) -> dict:
    """Run a single strategy across all symbols. Returns full result dict."""
    gen_fn = gen_bbs_signals_tf if strat["gen_fn"] == "bbs" else gen_kc_squeeze_signals_tf

    signals_by_sym = {}
    ltf_arrays     = {}

    for sym, d in all_data.items():
        htf_df = d.get(strat["htf_key"])
        ltf_df = d.get(strat["ltf_key"])
        if htf_df is None or ltf_df is None:
            continue

        htf_arr = prepare_tf_arrays(htf_df)
        ltf_arr = prepare_tf_arrays(ltf_df)
        if htf_arr is None or ltf_arr is None or htf_arr["n"] < 100 or ltf_arr["n"] < 100:
            continue

        ltf_arrays[sym] = ltf_arr

        # For BBS on 1h: use 4h as context / for KC 1h: use 4h as context
        ctx_df  = d.get("df4h")
        # EMA200 requires >= 200 bars; skip context if too short
        ctx_arr = None
        if ctx_df is not None and len(ctx_df) >= 200:
            ctx_arr = prepare_tf_arrays(ctx_df)

        htf_sigs = gen_fn(htf_arr, ctx_arr, strat["sig_params"])

        daily_d = prepare_daily_trend(d.get("df1d")) if strat["daily_filter"] else None
        mtf_sigs = gen_mtf_signals(
            htf_sigs, htf_arr, ltf_arr, strat["entry_params"],
            daily_data=daily_d, use_daily_filter=strat["daily_filter"]
        )
        signals_by_sym[sym] = mtf_sigs

    if not signals_by_sym:
        return {"error": "no signals"}

    # ── Inline simulation (captures completed trades for per-symbol/monthly breakdown) ──
    risk_pt  = 0.010        # 1% risk per trade
    cap_frac = 0.50         # position sizing fraction
    max_pos  = strat["max_pos"]
    sl_cfg   = strat["sl_cfg"]
    tp_cfg   = strat["tp_cfg"]

    balance  = START_BALANCE
    peak     = START_BALANCE
    max_dd   = 0.0
    open_pos = {}
    acnt     = 0
    completed = []

    max_n = max((a["n"] for a in ltf_arrays.values() if a), default=0)
    if max_n == 0:
        return {"error": "no ltf data"}

    for i in range(max(300, 0), max_n):
        for sym, arr in ltf_arrays.items():
            if arr is None or i >= arr["n"]:
                continue
            cl_i  = arr["cl"][i]
            hi_i  = arr["hi"][i]
            lo_i  = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            if sym in open_pos:
                pos    = open_pos[sym]
                d      = pos["d"]
                entry  = pos["entry"]
                sl_p   = pos["sl"]
                r      = pos["risk"]

                hit_sl = (d == "LONG"  and lo_i <= sl_p) or \
                         (d == "SHORT" and hi_i >= sl_p)
                hit_tp = False
                tp_px  = 0.0

                pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)
                if not hit_sl and tp_cfg["type"] == "fixed_r" and pnl_r >= tp_cfg["r"]:
                    hit_tp = True
                    tp_px  = cl_i

                if hit_sl or hit_tp:
                    ep     = tp_px if hit_tp else sl_p
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    fee_r  = (TAKER_FEE + SLIPPAGE) * 2 * entry / r
                    net_r  = exit_r - fee_r
                    pnl_usd = balance * risk_pt * cap_frac * net_r
                    balance += pnl_usd
                    if balance > peak:
                        peak = balance
                    dd = (peak - balance) / peak * 100
                    if dd > max_dd:
                        max_dd = dd
                    completed.append({
                        "sym":     sym,
                        "d":       d,
                        "won":     net_r > 0,
                        "pnl":     round(net_r, 4),     # R-multiple
                        "pnl_usd": round(pnl_usd, 4),
                        "exit_ts": int(arr["ts"][i]),
                        "reason":  "tp" if hit_tp else "sl",
                    })
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            if acnt >= max_pos:
                continue
            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig    = sym_sigs[i]
            new_sl = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                atr_i, arr["hi"], arr["lo"], i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue
            open_pos[sym] = {"d": sig["direction"], "entry": sig["entry_price"],
                             "sl": new_sl, "risk": rd}
            acnt += 1

    # Aggregate
    if not completed:
        return {"error": "no trades"}

    n      = len(completed)
    wins   = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    gp     = sum(t["pnl_usd"] for t in wins)
    gl     = abs(sum(t["pnl_usd"] for t in losses))
    pf     = gp / gl if gl > 0.001 else (99.0 if gp > 0 else 0.0)
    wr     = len(wins) / n * 100
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100

    return {
        "n_trades":  n,
        "pf":        round(pf, 3),
        "pnl_pct":   round(pnl_pct, 2),
        "max_dd":    round(max_dd, 2),
        "wr_pct":    round(wr, 1),
        "completed": completed,
    }


# ===========================================================================
#  STEP 5: Per-symbol breakdown
# ===========================================================================

def per_symbol_stats(completed: list[dict]) -> dict:
    """Return {symbol: {n, wins, gross_win, gross_loss, pf, wr}} sorted by contribution."""
    by_sym = defaultdict(lambda: {"n": 0, "wins": 0, "gw": 0.0, "gl": 0.0})
    for t in completed:
        sym = t.get("sym", "?").split("/")[0]
        by_sym[sym]["n"] += 1
        if t["pnl"] > 0:
            by_sym[sym]["wins"] += 1
            by_sym[sym]["gw"]   += t["pnl"]
        else:
            by_sym[sym]["gl"]   += abs(t["pnl"])

    stats = {}
    for sym, s in by_sym.items():
        pf = s["gw"] / s["gl"] if s["gl"] > 0.001 else float("inf")
        wr = s["wins"] / s["n"] * 100 if s["n"] > 0 else 0
        stats[sym] = {"n": s["n"], "wr": round(wr, 1), "pf": round(pf, 2),
                      "gw": round(s["gw"], 2), "gl": round(s["gl"], 2),
                      "net": round(s["gw"] - s["gl"], 2)}
    return dict(sorted(stats.items(), key=lambda x: x[1]["net"], reverse=True))


# ===========================================================================
#  STEP 6: Monthly breakdown
# ===========================================================================

def monthly_breakdown(completed: list[dict]) -> dict:
    """Return {YYYY-MM: {n, wins, pnl}} per month."""
    months = defaultdict(lambda: {"n": 0, "wins": 0, "pnl": 0.0})
    for t in completed:
        ts_ms = t.get("exit_ts") or t.get("ts", 0)
        dt    = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        key   = dt.strftime("%Y-%m")
        months[key]["n"]    += 1
        months[key]["pnl"]  += t["pnl"]
        if t["pnl"] > 0:
            months[key]["wins"] += 1
    return dict(sorted(months.items()))


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    print("=" * 70)
    print("Backtest Filtered Universe v1 -- 6 months (180d)")
    print(f"Filters: CoinGecko top-{MCAP_TOP_N} mcap + Bybit >=${MIN_VOLUME/1e6:.0f}M/day volume")
    print("=" * 70)

    # ── 1. Build universe ──────────────────────────────────────────────────
    filtered_syms, sym_info = build_filtered_universe()

    # ── 2. Download missing data ───────────────────────────────────────────
    ensure_data(filtered_syms)

    # ── 3. Load all data ───────────────────────────────────────────────────
    print("\n[3/4] Loading OHLCV data...")
    all_data = {}
    for sym in filtered_syms:
        d = load_symbol_data(sym)
        has_15m = d.get("df15") is not None and len(d["df15"]) > 200
        has_1h  = d.get("df1h") is not None and len(d["df1h"]) > 100
        if has_15m and has_1h:
            all_data[sym] = d
        else:
            base = sym.split("/")[0]
            print(f"  Skipping {base}: insufficient data (15m={d.get('df15') is not None}, 1h={d.get('df1h') is not None})")

    print(f"  Loaded data for {len(all_data)} symbols")

    # Also keep track of v10 overlap
    v10_bases = {s.split("/")[0] for s in V10_SYMBOLS}
    new_syms  = {s.split("/")[0] for s in all_data} - v10_bases
    kept_v10  = {s.split("/")[0] for s in all_data} & v10_bases
    print(f"  v10 symbols retained: {len(kept_v10)}/{len(V10_SYMBOLS)}")
    print(f"  New symbols added:    {len(new_syms)} -> {sorted(new_syms)}")

    # ── 4. Run strategies ──────────────────────────────────────────────────
    print("\n[4/4] Running strategies...")
    results = {}
    for strat in STRATEGIES:
        print(f"  {strat['id']}...", end="", flush=True)
        r = run_strategy_on_universe(strat, all_data)
        results[strat["id"]] = r
        if "error" not in r:
            print(f" {r.get('n_trades', 0)} trades, PF={r.get('pf', 0):.2f}, "
                  f"PnL={r.get('pnl_pct', 0):+.1f}%")
        else:
            print(f" ERROR: {r['error']}")

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════════════
    sep = "=" * 70

    print(f"\n{sep}")
    print("UNIVERSE COMPOSITION")
    print(sep)
    syms_sorted = sorted(all_data.keys(), key=lambda s: sym_info.get(s, {}).get("volume_24h", 0), reverse=True)
    print(f"{'Symbol':<15} {'24h Vol ($M)':>12}  {'In v10':>6}")
    for sym in syms_sorted:
        base = sym.split("/")[0]
        vol  = sym_info.get(sym, {}).get("volume_24h", 0) / 1e6
        flag = "YES" if base in v10_bases else "NEW"
        print(f"  {base:<13} {vol:>10.0f}M  {flag:>6}")

    print(f"\n{sep}")
    print("STRATEGY RESULTS (6 months, 1% risk per trade, no compounding base)")
    print(sep)
    print(f"{'Strategy':<18}  {'N':>5}  {'WR%':>5}  {'PF':>6}  {'PnL%':>7}  "
          f"{'DD%':>5}  {'v10 PF':>7}  {'v10 N/6m':>9}")

    for strat in STRATEGIES:
        sid = strat["id"]
        r   = results.get(sid, {})
        ref = strat["v10_ref"]
        # v10 was 365d, so 6m approx half
        v10_n_6m = ref["n"] // 2
        if "error" not in r and r.get("n_trades", 0) > 0:
            print(f"  {sid:<16}  {r['n_trades']:>5}  {r['wr_pct']:>4.1f}%  "
                  f"{r['pf']:>6.2f}  {r['pnl_pct']:>+6.1f}%  {r['max_dd']:>4.1f}%  "
                  f"{ref['pf']:>6.2f}  ~{v10_n_6m:>3}")
        else:
            print(f"  {sid:<16}  {'N/A':>5}")

    # Per strategy detail
    for strat in STRATEGIES:
        sid = strat["id"]
        r   = results.get(sid, {})
        if "error" in r or not r.get("completed"):
            continue

        completed = r["completed"]
        print(f"\n{sep}")
        print(f"STRATEGY: {strat['name']} -- Per-Symbol Breakdown")
        print(sep)
        sym_stats = per_symbol_stats(completed)
        print(f"  {'Symbol':<12}  {'N':>4}  {'WR%':>5}  {'PF':>6}  {'Net PnL':>10}")
        for sym, s in sym_stats.items():
            pf_str = f"{s['pf']:.2f}" if s['pf'] != float('inf') else "  inf"
            print(f"  {sym:<12}  {s['n']:>4}  {s['wr']:>4.1f}%  {pf_str:>6}  {s['net']:>+9.2f}")

        print(f"\n  --- Monthly PnL ({strat['id']}) ---")
        mon = monthly_breakdown(completed)
        total_pnl = 0.0
        for month, ms in mon.items():
            wr = ms["wins"] / ms["n"] * 100 if ms["n"] > 0 else 0
            sign = "+" if ms["pnl"] >= 0 else ""
            total_pnl += ms["pnl"]
            print(f"  {month}  n={ms['n']:>3}  WR={wr:4.1f}%  PnL={sign}{ms['pnl']:7.2f}")
        print(f"  Total PnL (units): {total_pnl:+.2f}")

    # ── Universe vs v10 comparison ─────────────────────────────────────────
    print(f"\n{sep}")
    print("SUMMARY: Filtered Universe vs v10 Baseline")
    print(sep)
    print(f"  Universe size:   v10={len(V10_SYMBOLS)}  filtered={len(all_data)}")
    print(f"  New symbols:     {sorted(new_syms) if new_syms else 'none'}")
    v10_removed = v10_bases - {s.split('/')[0] for s in all_data}
    print(f"  v10 removed:     {sorted(v10_removed) if v10_removed else 'none'}")
    print()
    for strat in STRATEGIES:
        sid = strat["id"]
        r   = results.get(sid, {})
        ref = strat["v10_ref"]
        if "error" not in r and r.get("n_trades", 0) > 0:
            # v10 6m estimate
            est_pf_6m = ref["pf"]  # WF test showed minimal degradation
            delta_pf  = r["pf"] - est_pf_6m
            delta_n   = r["n_trades"] - ref["n"] // 2
            print(f"  {sid}: PF={r['pf']:.2f} (v10={est_pf_6m:.2f}, d={delta_pf:+.2f}), "
                  f"trades={r['n_trades']} (v10~{ref['n']//2}, d={delta_n:+d})")

    # Save results
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts,
        "days": 180,
        "universe": [s.split("/")[0] for s in all_data],
        "n_symbols": len(all_data),
        "new_vs_v10": sorted(new_syms),
        "filters": {
            "min_volume_usd": MIN_VOLUME,
            "top_n_vol": TOP_N_VOL,
            "mcap_top_n": MCAP_TOP_N,
        },
        "strategies": {},
    }
    for strat in STRATEGIES:
        sid = strat["id"]
        r   = results.get(sid, {})
        if "error" not in r:
            out["strategies"][sid] = {
                "n_trades": r.get("n_trades", 0),
                "pf":       round(r.get("pf", 0), 3),
                "pnl_pct":  round(r.get("pnl_pct", 0), 2),
                "max_dd":   round(r.get("max_dd", 0), 2),
                "wr_pct":   round(r.get("wr_pct", 0), 1),
            }

    out_path = Path(__file__).parent / "results" / f"backtest_filtered_v1_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved: {out_path.name}")
    print("Done.")


if __name__ == "__main__":
    main()
