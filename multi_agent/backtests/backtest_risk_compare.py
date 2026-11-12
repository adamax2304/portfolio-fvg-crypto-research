"""
backtest_risk_compare.py
========================
Annual (365d) backtest comparing risk allocation schemes on the FULL
filtered universe (~50-60 symbols: top-150 mcap + $5M Bybit volume).

Three schemes tested:
  A) Equal 2%    -- current live config (all 3 strategies, 2% per trade)
  B) Weighted    -- BBS_5m 0.5%, BBS_15m 2.0%, KC 0%  (optimizer recommendation)
  C) BBS15m only -- BBS_15m 2.0% only (pure dominant strategy)

All schemes use compounding (each trade risks % of current equity).
Bot is NOT modified. This is analysis only.
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
    CACHE_DIR, START_BALANCE, TAKER_FEE, SLIPPAGE,
    prepare_tf_arrays, prepare_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
    compute_sl,
)

DAYS_TAG   = "365d"
MIN_VOLUME = 5_000_000
TOP_N_VOL  = 100
MCAP_TOP_N = 150

_SKIP_TICKERS = {
    "USDT","USDC","BUSD","DAI","TUSD","USDP","USDD","FDUSD","USDS","USDE",
    "PYUSD","FRAX","GUSD","LUSD","MIM","SUSD","USD1","USDF","USDG","USDTB",
    "USDY","RLUSD","BFUSD","CRVUSD","GHO","USDAI","USYC","BUIDL","PAXG","XAUT",
    "WBTC","WETH","WBNB","WSTETH","STETH","CBBTC","TBTC","RBTC",
    "LEO","WBT","OKB","BGB","KCS","GT","HTX",
}
_BYBIT_TICKER_MAP = {
    "PEPE":  "1000PEPE", "SHIB": "1000SHIB", "FLOKI": "1000FLOKI",
    "BONK":  "1000BONK", "NEIRO":"1000NEIRO", "PUMP": "PUMPFUN",
}

# Strategy definitions (v10 optimized params, unchanged)
STRATEGIES = [
    {
        "id": "BBS_1H_5M", "name": "BBS 1h->5m Pullback",
        "htf_key": "df1h", "ltf_key": "df5", "gen_fn": "bbs",
        "sig_params": {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5}, "tp_cfg": {"type": "fixed_r", "r": 3.0},
        "daily_filter": True, "max_pos": 4,
    },
    {
        "id": "BBS_1H_15M", "name": "BBS 1h->15m Next Bar",
        "htf_key": "df1h", "ltf_key": "df15", "gen_fn": "bbs",
        "sig_params": {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                       "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5}, "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False, "max_pos": 4,
    },
    {
        "id": "KC_1H_15M", "name": "KC 1h->15m Next Bar",
        "htf_key": "df1h", "ltf_key": "df15", "gen_fn": "kc",
        "sig_params": {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                       "mom_bars": 6, "sl_atr_mult": 1.5},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5}, "tp_cfg": {"type": "fixed_r", "r": 2.0},
        "daily_filter": False, "max_pos": 3,
    },
]

# Risk allocation schemes to compare
SCHEMES = {
    "A_equal_2pct": {
        "desc": "Current live: all 3 strategies, 2% per trade",
        "allocs": {"BBS_1H_5M": 1.0, "BBS_1H_15M": 1.0, "KC_1H_15M": 1.0},
        "risk_pt": 0.02,
    },
    "B_weighted": {
        "desc": "Optimizer: BBS_5m 0.5% / BBS_15m 2.0% / KC off",
        "allocs": {"BBS_1H_5M": 0.25, "BBS_1H_15M": 1.0, "KC_1H_15M": 0.0},
        "risk_pt": 0.02,
    },
    "C_bbs15m_only": {
        "desc": "Pure BBS_15m 2.0% only",
        "allocs": {"BBS_1H_5M": 0.0, "BBS_1H_15M": 1.0, "KC_1H_15M": 0.0},
        "risk_pt": 0.02,
    },
    "D_equal_1pct": {
        "desc": "Conservative: all 3 strategies, 1% per trade",
        "allocs": {"BBS_1H_5M": 1.0, "BBS_1H_15M": 1.0, "KC_1H_15M": 1.0},
        "risk_pt": 0.01,
    },
    "E_weighted_1pct": {
        "desc": "Optimizer conservative: weighted, 1% base risk",
        "allocs": {"BBS_1H_5M": 0.25, "BBS_1H_15M": 1.0, "KC_1H_15M": 0.0},
        "risk_pt": 0.01,
    },
}

# ===========================================================================
# 1. Build filtered universe
# ===========================================================================

def fetch_coingecko_mcap(top_n=150) -> set[str]:
    coins = []
    for page in (1, 2):
        if len(coins) >= top_n:
            break
        url = (f"https://api.coingecko.com/api/v3/coins/markets"
               f"?vs_currency=usd&order=market_cap_desc&per_page=100&page={page}&sparkline=false")
        req = urllib.request.Request(url, headers={"User-Agent": "fvgbot-bt/1.0"})
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
    # Add Bybit aliases
    for cg, bybit in _BYBIT_TICKER_MAP.items():
        if cg in result:
            result.add(bybit)
    return result


def build_universe() -> list[str]:
    print("  Fetching CoinGecko top-150 mcap...")
    mcap_set = fetch_coingecko_mcap(MCAP_TOP_N)
    print(f"  CoinGecko: {len(mcap_set)} eligible tickers")

    print("  Fetching Bybit tickers...")
    exchange = ccxt.bybit({"options": {"defaultType": "swap"}})
    tickers = exchange.fetch_tickers()
    usdt_perp = [
        (s, float(t.get("quoteVolume") or 0))
        for s, t in tickers.items()
        if s.endswith(":USDT") and float(t.get("quoteVolume") or 0) >= MIN_VOLUME
    ]
    usdt_perp.sort(key=lambda x: x[1], reverse=True)
    top_vol = [s for s, _ in usdt_perp[:TOP_N_VOL]]
    vol_map = {s: v for s, v in usdt_perp[:TOP_N_VOL]}

    filtered, skipped = [], []
    for sym in top_vol:
        base = sym.split("/")[0]
        if base in mcap_set:
            filtered.append(sym)
        else:
            skipped.append(base)

    print(f"  Universe: {len(filtered)} symbols (skipped {len(skipped)} low-mcap)")
    return filtered, vol_map


# ===========================================================================
# 2. Data download
# ===========================================================================

def _cache_path(symbol: str, tf: str, tag: str = DAYS_TAG) -> Path:
    safe = symbol.replace("/", "_").replace(":", "_")
    return CACHE_DIR / f"{safe}_{tf}_{tag}.parquet"


def _has_data(symbol: str, tfs: list[str], tag: str = DAYS_TAG) -> bool:
    # Accept 365d OR 180d (fallback)
    for tf in tfs:
        found = False
        for t in (tag, "180d"):
            if _cache_path(symbol, tf, t).exists():
                found = True
                break
        if not found:
            return False
    return True


def download_symbol(symbol: str, tfs: list[str], days: int = 370) -> dict[str, bool]:
    exchange = ccxt.bybit({"options": {"defaultType": "swap"}, "enableRateLimit": True})
    since_ms = int((time.time() - days * 86400) * 1000)
    results = {}
    for tf in tfs:
        if _cache_path(symbol, tf).exists():
            results[tf] = True
            continue
        all_bars = []
        fetch_since = since_ms
        for _ in range(60):
            try:
                bars = exchange.fetch_ohlcv(symbol, tf, since=fetch_since, limit=1000)
            except Exception as e:
                print(f"    {tf} error: {e}")
                time.sleep(2)
                break
            if not bars:
                break
            all_bars.extend(bars)
            if len(bars) < 1000:
                break
            fetch_since = bars[-1][0] + 1
            time.sleep(0.25)

        if all_bars:
            df = pd.DataFrame(all_bars, columns=["timestamp","open","high","low","close","volume"])
            df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
            df.to_parquet(_cache_path(symbol, tf), index=False)
            results[tf] = True
        else:
            results[tf] = False
    return results


def ensure_data(symbols: list[str]) -> None:
    req_tfs = ["5m", "15m", "1h", "4h", "1d"]
    need = [s for s in symbols if not _has_data(s, req_tfs)]
    if not need:
        print("[2/4] All data cached.")
        return
    print(f"\n[2/4] Downloading 365d data for {len(need)} new symbols...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    for sym in need:
        base = sym.split("/")[0]
        miss = [tf for tf in req_tfs if not _cache_path(sym, tf).exists()
                and not _cache_path(sym, tf, "180d").exists()]
        if not miss:
            continue
        print(f"  {base} ({len(miss)} TFs)...", end="", flush=True)
        r = download_symbol(sym, miss)
        for tf, ok in r.items():
            print(f" {tf}({'ok' if ok else 'FAIL'})", end="", flush=True)
        print()


# ===========================================================================
# 3. Load data
# ===========================================================================

def load_symbol(symbol: str) -> dict:
    def _load(tf):
        for tag in (DAYS_TAG, "180d", "90d"):
            p = _cache_path(symbol, tf, tag)
            if p.exists():
                try:
                    df = pd.read_parquet(p)
                    if df is not None and len(df) > 100:
                        return df
                except Exception:
                    continue
        return None

    d = {}
    d["df5"]  = _load("5m")
    d["df15"] = _load("15m")
    d["df1h"] = _load("1h")
    d["df4h"] = _load("4h")
    d["df1d"] = _load("1d")
    return d


# ===========================================================================
# 4. Generate trade logs per strategy
# ===========================================================================

def build_trade_logs(all_data: dict) -> dict[str, list]:
    """For each strategy, simulate and return list of {ts, r, sym, won} trades."""
    logs = {}
    for strat in STRATEGIES:
        gen_fn = gen_bbs_signals_tf if strat["gen_fn"] == "bbs" else gen_kc_squeeze_signals_tf
        signals_by_sym, ltf_arrays = {}, {}

        for sym, d in all_data.items():
            htf_df = d.get(strat["htf_key"])
            ltf_df = d.get(strat["ltf_key"])
            if htf_df is None or ltf_df is None:
                continue
            htf_arr = prepare_tf_arrays(htf_df)
            ltf_arr = prepare_tf_arrays(ltf_df)
            if htf_arr is None or ltf_arr is None or htf_arr["n"] < 200 or ltf_arr["n"] < 200:
                continue
            ltf_arrays[sym] = ltf_arr

            ctx_df  = d.get("df4h")
            ctx_arr = prepare_tf_arrays(ctx_df) if ctx_df is not None and len(ctx_df) >= 200 else None
            htf_sigs = gen_fn(htf_arr, ctx_arr, strat["sig_params"])
            daily_d  = prepare_daily_trend(d.get("df1d")) if strat["daily_filter"] else None
            mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, strat["entry_params"],
                                       daily_data=daily_d, use_daily_filter=strat["daily_filter"])
            signals_by_sym[sym] = mtf_sigs

        trades = _sim_trades(signals_by_sym, ltf_arrays, strat["sl_cfg"],
                             strat["tp_cfg"], strat["max_pos"])
        logs[strat["id"]] = trades
    return logs


def _sim_trades(signals_by_sym, ltf_arrays, sl_cfg, tp_cfg, max_pos) -> list:
    """Simulate and return list of closed trades with R-multiple."""
    open_pos, acnt, results = {}, 0, []
    max_n = max((a["n"] for a in ltf_arrays.values() if a), default=0)
    if max_n == 0:
        return results

    for i in range(max(300, 0), max_n):
        for sym, arr in ltf_arrays.items():
            if arr is None or i >= arr["n"]:
                continue
            cl_i = arr["cl"][i]; hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            if sym in open_pos:
                pos = open_pos[sym]
                d, entry, sl_p, r = pos["d"], pos["entry"], pos["sl"], pos["risk"]
                hit_sl = (d == "LONG" and lo_i <= sl_p) or (d == "SHORT" and hi_i >= sl_p)
                hit_tp, tp_px = False, 0.0
                pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)
                if not hit_sl and tp_cfg["type"] == "fixed_r" and pnl_r >= tp_cfg["r"]:
                    hit_tp, tp_px = True, cl_i
                if hit_sl or hit_tp:
                    ep = tp_px if hit_tp else sl_p
                    exit_r  = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    fee_r   = (TAKER_FEE + SLIPPAGE) * 2 * entry / r
                    net_r   = exit_r - fee_r
                    results.append({"ts": int(arr["ts"][i]), "r": net_r,
                                    "sym": sym, "won": net_r > 0})
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            if acnt >= max_pos:
                continue
            if i not in signals_by_sym.get(sym, {}):
                continue
            sig    = signals_by_sym[sym][i]
            new_sl = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                atr_i, arr["hi"], arr["lo"], i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue
            open_pos[sym] = {"d": sig["direction"], "entry": sig["entry_price"],
                             "sl": new_sl, "risk": rd}
            acnt += 1
    return results


# ===========================================================================
# 5. Portfolio simulation with compounding
# ===========================================================================

def sim_portfolio(trade_logs: dict, allocs: dict, risk_pt: float) -> dict:
    """Simulate portfolio with Kelly compounding. Returns stats + monthly breakdown."""
    all_trades = []
    for sid, trades in trade_logs.items():
        alloc = allocs.get(sid, 0.0)
        if alloc <= 0:
            continue
        for t in trades:
            all_trades.append({**t, "strat": sid, "alloc": alloc})
    all_trades.sort(key=lambda x: x["ts"])

    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    monthly = defaultdict(lambda: {"n": 0, "wins": 0, "pnl_usd": 0.0, "by_strat": defaultdict(float)})

    for t in all_trades:
        pnl = balance * risk_pt * t["alloc"] * t["r"]
        balance += pnl
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100
        max_dd   = max(max_dd, dd)

        dt  = datetime.fromtimestamp(t["ts"] / 1000, tz=timezone.utc)
        mon = dt.strftime("%Y-%m")
        monthly[mon]["n"]       += 1
        monthly[mon]["pnl_usd"] += pnl
        monthly[mon]["by_strat"][t["strat"]] += pnl
        if t["r"] > 0:
            monthly[mon]["wins"] += 1

    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100
    calmar  = pnl_pct / max_dd if max_dd > 0.01 else pnl_pct
    n_total = len(all_trades)
    wins    = sum(1 for t in all_trades if t["r"] > 0)

    return {
        "final":    round(balance, 2),
        "pnl_pct":  round(pnl_pct, 1),
        "max_dd":   round(max_dd, 2),
        "calmar":   round(calmar, 1),
        "n_trades": n_total,
        "wr_pct":   round(wins / n_total * 100, 1) if n_total > 0 else 0,
        "monthly":  dict(sorted(monthly.items())),
    }


# ===========================================================================
# 6. Per-strategy standalone stats
# ===========================================================================

def strat_stats(trades: list) -> dict:
    if not trades:
        return {"n": 0, "wr": 0, "pf": 0, "avg_r": 0, "pnl_1pct": 0}
    wins   = [t for t in trades if t["won"]]
    losses = [t for t in trades if not t["won"]]
    gw = sum(t["r"] for t in wins)
    gl = abs(sum(t["r"] for t in losses))
    pf = gw / gl if gl > 0.001 else 99.0

    # Simulate 1% risk compounded to get standalone PnL
    bal = START_BALANCE
    for t in trades:
        bal += bal * 0.01 * t["r"]
    pnl_1pct = (bal - START_BALANCE) / START_BALANCE * 100

    return {
        "n": len(trades), "wr": round(len(wins)/len(trades)*100, 1),
        "pf": round(pf, 2), "avg_r": round(sum(t["r"] for t in trades)/len(trades), 3),
        "pnl_1pct": round(pnl_1pct, 1),
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    SEP = "=" * 72
    print(SEP)
    print("Risk Allocation Comparison -- 365d annual backtest")
    print(f"Universe: top-{MCAP_TOP_N} mcap + ${MIN_VOLUME/1e6:.0f}M Bybit volume")
    print(SEP)

    # ── 1. Universe ────────────────────────────────────────────────────────
    print("\n[1/4] Building universe...")
    filtered_syms, vol_map = build_universe()
    print(f"  Final universe: {len(filtered_syms)} symbols")

    # ── 2. Download missing data ───────────────────────────────────────────
    ensure_data(filtered_syms)

    # ── 3. Load ────────────────────────────────────────────────────────────
    print("\n[3/4] Loading data...")
    all_data = {}
    for sym in filtered_syms:
        d = load_symbol(sym)
        if d.get("df15") is not None and d.get("df1h") is not None:
            if len(d["df15"]) > 200 and len(d["df1h"]) > 100:
                all_data[sym] = d
    print(f"  Loaded: {len(all_data)} symbols")

    # How many have full 365d vs 180d
    full_365 = sum(1 for sym in all_data
                   if _cache_path(sym, "1h", DAYS_TAG).exists())
    print(f"  Full 365d data: {full_365}  |  180d fallback: {len(all_data)-full_365}")

    # ── 4. Generate trade logs ─────────────────────────────────────────────
    print("\n[4/4] Running strategy simulations...")
    trade_logs = {}
    for strat in STRATEGIES:
        sid = strat["id"]
        print(f"  {sid}...", end="", flush=True)
        logs = build_trade_logs({sym: all_data[sym] for sym in all_data})
        trade_logs = logs  # build_trade_logs returns all strategies at once
        break  # call once, get all
    trade_logs = build_trade_logs(all_data)
    for sid, tl in trade_logs.items():
        s = strat_stats(tl)
        print(f"  {sid:<16}: {s['n']:>4} trades  WR={s['wr']:.1f}%  PF={s['pf']:.2f}"
              f"  avgR={s['avg_r']:+.3f}  standalone+1%risk={s['pnl_1pct']:+.0f}%")

    # ── 5. Portfolio simulations ───────────────────────────────────────────
    results = {}
    for scheme_id, scheme in SCHEMES.items():
        r = sim_portfolio(trade_logs, scheme["allocs"], scheme["risk_pt"])
        results[scheme_id] = r

    # ══════════════════════════════════════════════════════════════════════
    #  RESULTS
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{SEP}")
    print("INDIVIDUAL STRATEGY STATS (365d, standalone)")
    print(SEP)
    print(f"  {'Strategy':<18}  {'Trades':>6}  {'WR%':>5}  {'PF':>6}  {'avgR':>7}  "
          f"{'PnL% @1%risk':>13}")
    for strat in STRATEGIES:
        sid = strat["id"]
        s   = strat_stats(trade_logs.get(sid, []))
        print(f"  {sid:<18}  {s['n']:>6}  {s['wr']:>4.1f}%  {s['pf']:>6.2f}  "
              f"{s['avg_r']:>+6.3f}  {s['pnl_1pct']:>+12.1f}%")

    print(f"\n{SEP}")
    print("PORTFOLIO COMPARISON -- All risk schemes (365d, compounding)")
    print(SEP)
    print(f"  {'Scheme':<22}  {'Risk/trade':>10}  {'Trades':>6}  {'WR%':>5}  "
          f"{'PF':>5}  {'PnL%':>8}  {'DD%':>5}  {'Calmar':>7}  {'$1k->':>8}")
    print(f"  {'-'*22}  {'-'*10}  {'-'*6}  {'-'*5}  {'-'*5}  {'-'*8}  {'-'*5}  {'-'*7}  {'-'*8}")

    for scheme_id, scheme in SCHEMES.items():
        r   = results[scheme_id]
        tl  = [t for sid, al in scheme["allocs"].items()
               for t in trade_logs.get(sid, []) if al > 0]
        wins = sum(1 for t in tl if t["won"])
        n    = len(tl)
        wr   = wins / n * 100 if n > 0 else 0
        gw   = sum(t["r"] for t in tl if t["won"])
        gl   = abs(sum(t["r"] for t in tl if not t["won"]))
        pf   = round(gw / gl, 2) if gl > 0.001 else 99.0
        alloc_str = "+".join(
            f"{int(al*100)}%{sid.replace('BBS_1H_','').replace('KC_1H_','KC').replace('5M','5m').replace('15M','15m')}"
            for sid, al in scheme["allocs"].items() if al > 0
        )
        risk_str = f"{scheme['risk_pt']*100:.0f}% x alloc"
        print(f"  {scheme_id:<22}  {risk_str:>10}  {n:>6}  {wr:>4.1f}%  "
              f"{pf:>5.2f}  {r['pnl_pct']:>+7.1f}%  {r['max_dd']:>4.1f}%  "
              f"{r['calmar']:>7.1f}  ${r['final']:>7,.0f}")

    print(f"\n  Desc:")
    for scheme_id, scheme in SCHEMES.items():
        print(f"    {scheme_id}: {scheme['desc']}")

    # ── Monthly breakdown for key schemes ─────────────────────────────────
    for scheme_id in ("A_equal_2pct", "B_weighted", "C_bbs15m_only"):
        r = results[scheme_id]
        print(f"\n{SEP}")
        print(f"MONTHLY BREAKDOWN -- {scheme_id}: {SCHEMES[scheme_id]['desc']}")
        print(SEP)
        print(f"  {'Month':<8}  {'N':>4}  {'WR%':>5}  {'PnL $':>10}  {'PnL %':>7}  "
              f"{'Balance':>10}  {'Loss?':>6}")
        bal = START_BALANCE
        for mon, ms in r["monthly"].items():
            wr  = ms["wins"] / ms["n"] * 100 if ms["n"] > 0 else 0
            bal += ms["pnl_usd"]
            pct = ms["pnl_usd"] / (bal - ms["pnl_usd"]) * 100
            flag = "LOSS" if ms["pnl_usd"] < 0 else ""
            print(f"  {mon:<8}  {ms['n']:>4}  {wr:>4.1f}%  "
                  f"{ms['pnl_usd']:>+9.2f}  {pct:>+6.1f}%  ${bal:>9,.2f}  {flag}")

    # ── Conclusion ─────────────────────────────────────────────────────────
    best_pnl    = max(results, key=lambda k: results[k]["pnl_pct"])
    best_calmar = max(results, key=lambda k: results[k]["calmar"])

    print(f"\n{SEP}")
    print("CONCLUSION")
    print(SEP)
    print(f"  Best absolute PnL:     {best_pnl} -> "
          f"+{results[best_pnl]['pnl_pct']:.1f}%  ${results[best_pnl]['final']:,.0f}")
    print(f"  Best risk-adjusted:    {best_calmar} -> "
          f"Calmar={results[best_calmar]['calmar']:.1f}  "
          f"DD={results[best_calmar]['max_dd']:.1f}%")
    print()
    a = results["A_equal_2pct"]
    b = results["B_weighted"]
    c = results["C_bbs15m_only"]
    print(f"  Current live (A) vs Weighted (B): "
          f"PnL {a['pnl_pct']:+.1f}% vs {b['pnl_pct']:+.1f}%  "
          f"| Calmar {a['calmar']:.0f} vs {b['calmar']:.0f}")
    print(f"  Current live (A) vs BBS15m only (C): "
          f"PnL {a['pnl_pct']:+.1f}% vs {c['pnl_pct']:+.1f}%  "
          f"| DD {a['max_dd']:.1f}% vs {c['max_dd']:.1f}%")

    winner = best_calmar
    print(f"\n  >> RECOMMENDED: {winner} -- {SCHEMES[winner]['desc']}")
    print(f"     PnL={results[winner]['pnl_pct']:+.1f}%  "
          f"DD={results[winner]['max_dd']:.1f}%  "
          f"Calmar={results[winner]['calmar']:.1f}  "
          f"$1k->${results[winner]['final']:,.0f}")

    # Save
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = {
        "timestamp": ts, "universe_size": len(all_data),
        "symbols": [s.split("/")[0] for s in all_data],
        "strategy_stats": {sid: strat_stats(tl) for sid, tl in trade_logs.items()},
        "schemes": {sid: {**results[sid], "monthly": {
            m: {"n": v["n"], "wins": v["wins"], "pnl_usd": round(v["pnl_usd"], 2)}
            for m, v in results[sid]["monthly"].items()
        }} for sid in results},
    }
    out_path = Path(__file__).parent / "results" / f"risk_compare_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path.name}")


if __name__ == "__main__":
    main()
