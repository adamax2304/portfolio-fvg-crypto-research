"""
backtest_range_lgc.py  --  Range Liquidity Grab + CHoCH + FVG Strategy (2026-04-14)

Strategy Logic:
  1. RANGE DETECTION: Price oscillates between support and resistance
     - At least 3 touches on each boundary (high and low)
     - Boundaries within ATR * range_atr_factor of each other
     - Range height must be within [min_range_pct, max_range_pct] of price
     - Minimum range_lookback bars to qualify

  2. LIQUIDITY GRAB (LG): False breakout beyond range boundary
     - Price breaks below support or above resistance (closes beyond by < lg_close_pct)
     - OR wicks beyond by at least lg_wick_min_pct
     - Closes BACK inside the range on the same bar (or within lg_close_bars)

  3. CHANGE OF CHARACTER (CHoCH): Micro structure shift after LG
     - After bullish LG (below support): next swing high breaks last swing high inside range
     - After bearish LG (above resistance): next swing low breaks last swing low inside range
     - Confirmed within choch_bars bars after LG bar

  4. FVG ENTRY: Enter on retest of Fair Value Gap
     - FVG = 3-bar pattern: gap between bar[i-2].high and bar[i].low (for bullish FVG)
     - Entry when price retraces back into FVG zone
     - FVG must form AFTER CHoCH confirmation
     - Wait up to fvg_wait_bars bars for retest

  5. SL: Behind the liquidity grab extremum (low of LG candle for LONG)
     + sl_atr_buffer * ATR buffer

  6. TP: Opposite range boundary (or tp_r * risk if boundary too far)

  7. FILTERS:
     - Session: London (07-12 UTC) or NY (13-17 UTC) only
     - Min ADX: trend must NOT be too strong (we want range, not trend)
     - Max ADX: range confirmation
     - Range must be "recent" (max range_age_bars since last touch)

Decision: Add to live system only if PF >= 2.0 and trades >= 30/year
"""
from __future__ import annotations
import json, logging, sys, time as _time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("lgc")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
    "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]  # 17 symbols (blacklist applied)

START_BALANCE = 1000.0
TAKER_FEE  = 0.00055
SLIPPAGE   = 0.0005
MAX_LEVERAGE = 50

# ===========================================================================
#  DATA LOADING
# ===========================================================================

def _load(symbol, tf, days_tag="365d"):
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in (days_tag, "180d", "90d"):
        for ext in (".parquet", ".csv"):
            fpath = CACHE_DIR / f"{safe}_{tf}_{tag}{ext}"
            if fpath.exists():
                try:
                    df = pd.read_parquet(fpath) if ext == ".parquet" else pd.read_csv(fpath)
                    if df is not None and not df.empty:
                        for c in ("open", "high", "low", "close", "volume"):
                            if c in df.columns:
                                df[c] = df[c].astype(float)
                        if "timestamp" in df.columns:
                            df["timestamp"] = df["timestamp"].astype(int)
                        return df
                except Exception:
                    continue
    return None


def load_data(days_tag="365d"):
    all_data = {}
    for sym in SYMBOLS:
        d = {}
        d["df15"] = _load(sym, "15m", days_tag)
        d["df1h"] = _load(sym, "1h", days_tag)
        if d["df15"] is None or len(d["df15"]) < 500:
            continue
        all_data[sym] = d
    return all_data


# ===========================================================================
#  INDICATORS
# ===========================================================================

def _atr(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i-1]*(period-1)+tr[i])/period
    return out

def _adx(hi, lo, cl, period=14):
    n = len(cl)
    adx = np.full(n, 20.0)
    pdm = np.zeros(n); ndm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up = hi[i]-hi[i-1]; down = lo[i-1]-lo[i]
        pdm[i] = up if (up > down and up > 0) else 0
        ndm[i] = down if (down > up and down > 0) else 0
    if n < period * 3:
        return adx
    atr_s = np.mean(tr[1:period+1])
    pdm_s = np.mean(pdm[1:period+1])
    ndm_s = np.mean(ndm[1:period+1])
    dx_arr = []
    for i in range(period, n):
        atr_s = atr_s - atr_s/period + tr[i]
        pdm_s = pdm_s - pdm_s/period + pdm[i]
        ndm_s = ndm_s - ndm_s/period + ndm[i]
        pdi = 100*pdm_s/atr_s if atr_s > 0 else 0
        ndi = 100*ndm_s/atr_s if atr_s > 0 else 0
        s = pdi + ndi
        dx = abs(pdi-ndi)/s*100 if s > 0 else 0
        dx_arr.append(dx)
        if len(dx_arr) == period:
            adx[i] = np.mean(dx_arr)
        elif len(dx_arr) > period:
            adx[i] = (adx[i-1]*(period-1)+dx)/period
    return adx

def _sma(arr, period):
    out = np.full_like(arr, np.nan, dtype=float)
    cs = np.cumsum(arr)
    out[period-1:] = (cs[period-1:] - np.concatenate(([0], cs[:-period])))/period
    return out

def _swing_highs(hi, lo, n_bars=2):
    """Returns array of swing high indices (local maxima over n_bars each side)."""
    out = []
    for i in range(n_bars, len(hi) - n_bars):
        if all(hi[i] > hi[i-j] for j in range(1, n_bars+1)) and \
           all(hi[i] > hi[i+j] for j in range(1, n_bars+1)):
            out.append(i)
    return out

def _swing_lows(hi, lo, n_bars=2):
    """Returns array of swing low indices (local minima over n_bars each side)."""
    out = []
    for i in range(n_bars, len(lo) - n_bars):
        if all(lo[i] < lo[i-j] for j in range(1, n_bars+1)) and \
           all(lo[i] < lo[i+j] for j in range(1, n_bars+1)):
            out.append(i)
    return out


# ===========================================================================
#  RANGE DETECTION
# ===========================================================================

def detect_ranges(cl, hi, lo, atr, params, warmup=200):
    """
    Scan for range formations.
    Returns dict: bar_i -> {resistance, support, range_start, range_height}
    """
    n = len(cl)
    lookback = params["range_lookback"]       # bars to look back for range
    min_touches = params["min_touches"]        # min touches per boundary
    max_range_pct = params["max_range_pct"]    # max range height as % of price
    min_range_pct = params["min_range_pct"]    # min range height as % of price
    max_adx = params.get("max_adx", 25.0)      # ADX must be below this (range-bound)
    adx = _adx(hi, lo, cl, 14)
    touch_tol = params.get("touch_tol_atr", 0.3)  # touch tolerance in ATR

    ranges = {}  # bar_i -> range info

    for i in range(warmup + lookback, n):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue
        # ADX filter: must be in range-bound condition
        if not np.isnan(adx[i]) and adx[i] > max_adx:
            continue

        seg_hi = hi[i-lookback:i+1]
        seg_lo = lo[i-lookback:i+1]
        seg_cl = cl[i-lookback:i+1]

        resistance = float(np.percentile(seg_hi, 95))  # upper boundary
        support    = float(np.percentile(seg_lo, 5))   # lower boundary
        range_h    = resistance - support
        mid        = (resistance + support) / 2.0

        # Range height as % of price must be in valid band
        h_pct = range_h / mid
        if h_pct > max_range_pct or h_pct < min_range_pct:
            continue

        # Count touches on each boundary
        tol = atr[i] * touch_tol
        hi_touches = int(np.sum(seg_hi >= resistance - tol))
        lo_touches = int(np.sum(seg_lo <= support + tol))

        if hi_touches < min_touches or lo_touches < min_touches:
            continue

        # Range confirmed — store for this bar
        ranges[i] = {
            "resistance": resistance,
            "support": support,
            "range_height": range_h,
            "range_pct": round(h_pct * 100, 2),
            "adx": round(float(adx[i]), 1),
            "range_start": i - lookback,
        }

    return ranges


# ===========================================================================
#  LIQUIDITY GRAB DETECTION
# ===========================================================================

def detect_liquidity_grabs(cl, hi, lo, ranges, params):
    """
    After a range is confirmed at bar i, scan forward for liquidity grabs.
    LG = price breaks beyond boundary and closes back inside.

    Returns list of: {bar: bar_of_LG, direction: LONG/SHORT, range: {...}, lg_low/lg_high}
    """
    n = len(cl)
    lg_close_pct = params.get("lg_close_pct", 0.002)   # max close beyond boundary
    atr = _atr(hi, lo, cl, 14)
    lg_signals = []
    used_bars = set()

    # Sort ranges by bar index
    sorted_ranges = sorted(ranges.items())

    for r_bar, r_info in sorted_ranges:
        resistance = r_info["resistance"]
        support    = r_info["support"]
        range_h    = r_info["range_height"]

        # Look forward for LG (within next lg_scan_bars)
        lg_scan = params.get("lg_scan_bars", 20)

        for j in range(r_bar + 1, min(r_bar + lg_scan, n)):
            if j in used_bars:
                continue

            # Bullish LG: wick below support, close back above support
            wick_below = lo[j] < support
            close_inside_bull = cl[j] >= support * (1 - lg_close_pct)
            if wick_below and close_inside_bull:
                lg_signals.append({
                    "bar": j,
                    "direction": "LONG",
                    "range": r_info,
                    "lg_extreme": float(lo[j]),    # lowest point of LG
                    "entry_ref": float(support),   # reference (support level)
                    "range_bar": r_bar,
                })
                used_bars.add(j)
                break

            # Bearish LG: wick above resistance, close back below resistance
            wick_above = hi[j] > resistance
            close_inside_bear = cl[j] <= resistance * (1 + lg_close_pct)
            if wick_above and close_inside_bear:
                lg_signals.append({
                    "bar": j,
                    "direction": "SHORT",
                    "range": r_info,
                    "lg_extreme": float(hi[j]),    # highest point of LG
                    "entry_ref": float(resistance),
                    "range_bar": r_bar,
                })
                used_bars.add(j)
                break

    return lg_signals


# ===========================================================================
#  CHoCH CONFIRMATION
# ===========================================================================

def confirm_choch(lg_sig, cl, hi, lo, atr, params):
    """
    After bullish LG: look for CHoCH (price breaks last swing high inside range)
    After bearish LG: look for CHoCH (price breaks last swing low inside range)

    Returns bar index of CHoCH confirmation or None.
    """
    bar_lg = lg_sig["bar"]
    direction = lg_sig["direction"]
    r_info = lg_sig["range"]
    choch_bars = params.get("choch_bars", 8)
    n = len(cl)

    # Find the last swing high/low INSIDE range before LG (within range_lookback bars)
    r_start = r_info["range_start"]
    resistance = r_info["resistance"]
    support    = r_info["support"]

    seg_hi = hi[r_start:bar_lg+1]
    seg_lo = lo[r_start:bar_lg+1]
    seg_cl = cl[r_start:bar_lg+1]

    if direction == "LONG":
        # For bullish CHoCH: find the most recent swing high INSIDE range before LG
        target_high = None
        for k in range(len(seg_hi)-1, 1, -1):
            # simple local high: higher than neighbors
            if (seg_hi[k] > seg_hi[k-1] and
                    (k+1 >= len(seg_hi) or seg_hi[k] > seg_hi[k+1])):
                # Must be inside range
                if seg_hi[k] < resistance:
                    target_high = float(seg_hi[k])
                    break
        if target_high is None:
            target_high = float(np.max(seg_hi)) * 0.98

        # Look forward for price to break above target_high (CHoCH)
        for j in range(bar_lg + 1, min(bar_lg + choch_bars + 1, n)):
            if hi[j] > target_high:
                return j, target_high
        return None, None

    else:  # SHORT
        # For bearish CHoCH: find the most recent swing low INSIDE range before LG
        target_low = None
        for k in range(len(seg_lo)-1, 1, -1):
            if (seg_lo[k] < seg_lo[k-1] and
                    (k+1 >= len(seg_lo) or seg_lo[k] < seg_lo[k+1])):
                if seg_lo[k] > support:
                    target_low = float(seg_lo[k])
                    break
        if target_low is None:
            target_low = float(np.min(seg_lo)) * 1.02

        for j in range(bar_lg + 1, min(bar_lg + choch_bars + 1, n)):
            if lo[j] < target_low:
                return j, target_low
        return None, None


# ===========================================================================
#  FVG DETECTION AND ENTRY
# ===========================================================================

def find_fvg_entry(choch_bar, direction, cl, hi, lo, atr, params):
    """
    After CHoCH, look for a Fair Value Gap (FVG) pattern and wait for retest.

    Bullish FVG: bar[i-2].high < bar[i].low (gap between them)
    Bearish FVG: bar[i-2].low > bar[i].high

    Entry: when price retraces into the FVG zone.
    Returns entry bar or None.
    """
    n = len(cl)
    fvg_wait = params.get("fvg_wait_bars", 15)
    fvg_min_size_atr = params.get("fvg_min_size_atr", 0.05)  # min FVG size in ATR units
    sl_atr_mult = params.get("sl_atr_mult", 1.5)

    # Scan for FVGs formed AFTER CHoCH
    fvgs = []
    for i in range(choch_bar + 2, min(choch_bar + fvg_wait, n)):
        atr_v = atr[i] if not np.isnan(atr[i]) and atr[i] > 0 else cl[i] * 0.01
        if direction == "LONG":
            # Bullish FVG: gap between bar[i-2] high and bar[i] low
            gap_bottom = hi[i-2]
            gap_top    = lo[i]
            if gap_top > gap_bottom and (gap_top - gap_bottom) >= fvg_min_size_atr * atr_v:
                fvgs.append({"bar": i, "gap_top": float(gap_top),
                              "gap_bottom": float(gap_bottom), "direction": "LONG"})
        else:
            # Bearish FVG: gap between bar[i] high and bar[i-2] low
            gap_top    = lo[i-2]
            gap_bottom = hi[i]
            if gap_top > gap_bottom and (gap_top - gap_bottom) >= fvg_min_size_atr * atr_v:
                fvgs.append({"bar": i, "gap_top": float(gap_top),
                              "gap_bottom": float(gap_bottom), "direction": "SHORT"})

    if not fvgs:
        return None

    # Take the most recent FVG
    fvg = fvgs[-1]
    fvg_bar = fvg["bar"]

    # Wait for retest (price enters FVG zone)
    for j in range(fvg_bar + 1, min(fvg_bar + fvg_wait, n)):
        atr_v = atr[j] if not np.isnan(atr[j]) and atr[j] > 0 else cl[j] * 0.01
        if direction == "LONG":
            # Retest: price dips into FVG zone but closes above gap_bottom
            if lo[j] <= fvg["gap_top"] and cl[j] >= fvg["gap_bottom"]:
                entry_price = float(cl[j])
                # SL below the FVG gap
                sl_price = fvg["gap_bottom"] - atr_v * sl_atr_mult
                return {
                    "bar": j, "direction": "LONG",
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "fvg": fvg,
                }
        else:
            # Retest: price rises into FVG zone but closes below gap_top
            if hi[j] >= fvg["gap_bottom"] and cl[j] <= fvg["gap_top"]:
                entry_price = float(cl[j])
                sl_price = fvg["gap_top"] + atr_v * sl_atr_mult
                return {
                    "bar": j, "direction": "SHORT",
                    "entry_price": entry_price,
                    "sl_price": sl_price,
                    "fvg": fvg,
                }

    return None


# ===========================================================================
#  SESSION FILTER
# ===========================================================================

def is_session(ts_ms, mode="london_ny"):
    """UTC session filter."""
    hour = (int(ts_ms) // 3_600_000) % 24
    if mode == "london_ny":
        return (7 <= hour < 17)  # London + NY overlap
    elif mode == "london":
        return (7 <= hour < 12)
    elif mode == "ny":
        return (13 <= hour < 17)
    elif mode == "eu_us":
        return (7 <= hour < 20)
    return True  # no filter


# ===========================================================================
#  FULL STRATEGY SIGNAL GENERATOR
# ===========================================================================

def gen_lgc_signals(df, params):
    """
    Generate all LGC (Liquidity Grab + CHoCH + FVG) signals for a single symbol.
    Returns dict: entry_bar -> signal_dict
    """
    if df is None or len(df) < 500:
        return {}

    cl = df["close"].astype(float).to_numpy()
    hi = df["high"].astype(float).to_numpy()
    lo = df["low"].astype(float).to_numpy()
    op = df["open"].astype(float).to_numpy()
    ts = df["timestamp"].astype(int).to_numpy()
    n = len(cl)
    atr = _atr(hi, lo, cl, 14)

    # Detect ranges
    ranges = detect_ranges(cl, hi, lo, atr, params)
    if not ranges:
        return {}

    # Detect liquidity grabs
    lg_signals = detect_liquidity_grabs(cl, hi, lo, ranges, params)
    if not lg_signals:
        return {}

    # For each LG: check CHoCH -> FVG -> entry
    signals = {}
    used_entry_bars = set()

    for lg_sig in lg_signals:
        bar_lg = lg_sig["bar"]
        direction = lg_sig["direction"]

        # Session filter on LG bar
        session_mode = params.get("session", "london_ny")
        if not is_session(ts[bar_lg], session_mode):
            continue

        # CHoCH confirmation
        choch_bar, choch_level = confirm_choch(lg_sig, cl, hi, lo, atr, params)
        if choch_bar is None:
            continue

        # FVG entry
        entry_sig = find_fvg_entry(choch_bar, direction, cl, hi, lo, atr, params)
        if entry_sig is None:
            continue

        entry_bar = entry_sig["bar"]
        if entry_bar in used_entry_bars:
            continue

        # Session filter on entry bar
        if not is_session(ts[entry_bar], session_mode):
            continue

        # TP = opposite range boundary
        r_info = lg_sig["range"]
        if direction == "LONG":
            tp_target = r_info["resistance"]
        else:
            tp_target = r_info["support"]

        signals[entry_bar] = {
            "direction": direction,
            "entry_price": entry_sig["entry_price"],
            "sl_price": entry_sig["sl_price"],
            "tp_target": tp_target,
            "range_info": r_info,
            "lg_bar": bar_lg,
            "choch_bar": choch_bar,
            "fvg": entry_sig["fvg"],
        }
        used_entry_bars.add(entry_bar)

    return signals


# ===========================================================================
#  SIMULATION ENGINE
# ===========================================================================

def run_sim_lgc(signals_by_sym, df_by_sym, params,
                risk_pt=0.010, cap_frac=0.30, max_pos=4,
                start_frac=0.0, end_frac=1.0,
                use_break_even=True, be_r=1.5, be_buffer=0.001):
    balance = START_BALANCE
    peak = START_BALANCE
    open_pos: dict = {}
    acnt = 0
    completed = []

    # Find max bar count
    arr_cache = {}
    for sym, df in df_by_sym.items():
        if df is not None and len(df) >= 100:
            cl = df["close"].astype(float).to_numpy()
            hi = df["high"].astype(float).to_numpy()
            lo = df["low"].astype(float).to_numpy()
            op = df["open"].astype(float).to_numpy()
            ts = df["timestamp"].astype(int).to_numpy()
            atr = _atr(hi, lo, cl, 14)
            arr_cache[sym] = {"cl": cl, "hi": hi, "lo": lo, "op": op, "ts": ts,
                               "atr": atr, "n": len(cl)}

    max_n = max((a["n"] for a in arr_cache.values()), default=0)
    if max_n == 0:
        return _empty_result()

    i_start = max(400, int(max_n * start_frac))
    i_end = int(max_n * end_frac)
    tp_r_fallback = params.get("tp_r_fallback", 3.0)  # use R if range TP too far

    for i in range(i_start, i_end):
        for sym, arr in arr_cache.items():
            if i >= arr["n"]:
                continue
            cl_i = arr["cl"][i]; hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            # ---- Position management ----
            if sym in open_pos:
                pos = open_pos[sym]; d = pos["d"]; entry = pos["entry"]
                sl = pos["sl"]; r = pos["risk"]; q = pos["qty"]
                bars = i - pos["bar_i"]
                tp = pos["tp"]

                if d == "LONG":
                    pos["mfe"] = max(pos["mfe"], (hi_i - entry) / r)
                    pos["mae"] = max(pos["mae"], (entry - lo_i) / r)
                else:
                    pos["mfe"] = max(pos["mfe"], (entry - lo_i) / r)
                    pos["mae"] = max(pos["mae"], (hi_i - entry) / r)
                pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)

                # Break-even
                if use_break_even and not pos.get("be_applied", False):
                    if pnl_r >= be_r:
                        if d == "LONG":
                            new_be = entry * (1.0 + be_buffer)
                            if new_be > pos["sl"]:
                                pos["sl"] = new_be; sl = new_be
                        else:
                            new_be = entry * (1.0 - be_buffer)
                            if new_be < pos["sl"]:
                                pos["sl"] = new_be; sl = new_be
                        pos["be_applied"] = True

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = False; tp_price = 0.0; exit_rsn = ""

                if not hit_sl:
                    # Check TP
                    if d == "LONG" and hi_i >= tp:
                        hit_tp = True; tp_price = tp; exit_rsn = "tp_range"
                    elif d == "SHORT" and lo_i <= tp:
                        hit_tp = True; tp_price = tp; exit_rsn = "tp_range"
                    # Max bars exit (trend got away without tp)
                    elif bars >= params.get("max_trade_bars", 60):
                        hit_tp = True; tp_price = cl_i; exit_rsn = "timeout"
                else:
                    exit_rsn = "sl_hit"

                if hit_sl or hit_tp:
                    ep = tp_price if hit_tp else sl
                    gross = ((ep - entry) if d == "LONG" else (entry - ep)) * q
                    fee = ep * q * (TAKER_FEE + SLIPPAGE)
                    net = gross - fee
                    balance += net
                    peak = max(peak, balance)
                    exit_r = ((ep - entry) / r) if d == "LONG" else ((entry - ep) / r)
                    completed.append({"sym": sym, "d": d, "exit_r": round(exit_r, 3),
                        "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                        "bars": bars, "won": net > 0, "pnl_usd": round(net, 4),
                        "reason": exit_rsn})
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            # ---- Entry logic ----
            if acnt >= max_pos:
                continue
            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig = sym_sigs[i]
            direction = sig["direction"]
            entry_p  = sig["entry_price"]
            sl_p     = sig["sl_price"]
            tp_tgt   = sig["tp_target"]

            rd = abs(entry_p - sl_p)
            if rd <= 0 or rd / entry_p > 0.08 or rd / entry_p < 0.001:
                continue

            # TP: use range boundary, but cap at tp_r_fallback * R if too far
            if direction == "LONG":
                tp_dist = tp_tgt - entry_p
            else:
                tp_dist = entry_p - tp_tgt

            if tp_dist <= 0 or tp_dist / rd > tp_r_fallback:
                # Range target too far or wrong side — use fixed R
                if direction == "LONG":
                    tp_price_final = entry_p + rd * tp_r_fallback
                else:
                    tp_price_final = entry_p - rd * tp_r_fallback
            else:
                tp_price_final = tp_tgt

            ab = balance * cap_frac
            ru = ab * risk_pt
            qty = ru / rd
            mg = entry_p * qty / MAX_LEVERAGE
            if mg > ab * 0.4:
                qty = ab * 0.4 * MAX_LEVERAGE / entry_p
            if qty * entry_p < 5.0:
                continue
            ef = entry_p * qty * (TAKER_FEE + SLIPPAGE)
            balance -= ef
            open_pos[sym] = {"d": direction, "entry": entry_p, "sl": sl_p,
                "tp": tp_price_final, "risk": rd, "qty": qty, "bar_i": i,
                "mfe": 0.0, "mae": 0.0, "be_applied": False}
            acnt += 1

    return _aggregate(completed, balance)


def _empty_result():
    return {"trades": 0, "wr_pct": 0, "pnl_pct": 0, "profit_factor": 0,
            "max_dd_pct": 0, "avg_exit_r": 0, "avg_mfe": 0, "avg_mae": 0,
            "exit_reasons": {}}


def _aggregate(completed, balance):
    if not completed:
        return _empty_result()
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    n = len(completed)
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / n * 100
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100
    reasons = defaultdict(int)
    for t in completed:
        reasons[t["reason"]] += 1
    bal = START_BALANCE; peak = START_BALANCE; max_dd = 0.0
    for t in completed:
        bal += t["pnl_usd"]; peak = max(peak, bal)
        dd = (peak - bal) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 1), "pnl_pct": round(pnl_pct, 2),
        "profit_factor": round(min(pf, 99), 3), "max_dd_pct": round(max_dd, 2),
        "avg_exit_r": round(float(np.mean([t["exit_r"] for t in completed])), 3),
        "avg_mfe": round(float(np.mean([t["mfe"] for t in completed])), 3),
        "avg_mae": round(float(np.mean([t["mae"] for t in completed])), 3),
        "avg_bars": round(float(np.mean([t["bars"] for t in completed])), 1),
        "exit_reasons": dict(reasons), "final_bal": round(balance, 2),
    }


# ===========================================================================
#  PARAMETER GRIDS
# ===========================================================================

BASE_PARAMS = {
    "range_lookback": 48,      # bars to define range (48 x 15m = 12h)
    "min_touches": 3,          # min touches per boundary
    "max_range_pct": 0.08,     # range height max 8% of price
    "min_range_pct": 0.01,     # range height min 1% of price
    "max_adx": 22.0,           # ADX must be below (range-bound)
    "touch_tol_atr": 0.3,      # touch tolerance = 0.3 * ATR
    "lg_close_pct": 0.003,     # LG close within 0.3% of boundary
    "lg_scan_bars": 20,        # bars to scan for LG after range confirmed
    "choch_bars": 8,            # bars to confirm CHoCH after LG
    "fvg_wait_bars": 15,       # bars to wait for FVG entry
    "fvg_min_size_atr": 0.05,  # min FVG size (ATR fraction)
    "sl_atr_mult": 1.5,        # SL = fvg_gap_bottom - ATR * mult
    "tp_r_fallback": 3.0,      # TP = range boundary or 3R max
    "session": "london_ny",    # session filter
    "max_trade_bars": 60,      # max bars in trade (15h on 15m)
}

PARAM_GRID = []

# Grid 1: Vary range lookback
for lb in (24, 48, 72, 96):
    p = dict(BASE_PARAMS); p["range_lookback"] = lb
    p["id"] = "lgc_lb%d" % lb
    PARAM_GRID.append(p)

# Grid 2: Vary min touches
for mt in (2, 3, 4):
    p = dict(BASE_PARAMS); p["min_touches"] = mt
    p["id"] = "lgc_mt%d" % mt
    PARAM_GRID.append(p)

# Grid 3: Vary max ADX (range-bound filter)
for adx in (18, 22, 25, 30):
    p = dict(BASE_PARAMS); p["max_adx"] = adx
    p["id"] = "lgc_adx%d" % adx
    PARAM_GRID.append(p)

# Grid 4: Vary SL mult
for sl_m in (1.0, 1.5, 2.0):
    p = dict(BASE_PARAMS); p["sl_atr_mult"] = sl_m
    p["id"] = "lgc_sl%.1f" % sl_m
    PARAM_GRID.append(p)

# Grid 5: Vary TP fallback R
for tp_r in (2.0, 2.5, 3.0, 4.0):
    p = dict(BASE_PARAMS); p["tp_r_fallback"] = tp_r
    p["id"] = "lgc_tp%.1fR" % tp_r
    PARAM_GRID.append(p)

# Grid 6: Session filter variants
for sess in ("london_ny", "london", "ny", "eu_us"):
    p = dict(BASE_PARAMS); p["session"] = sess
    p["id"] = "lgc_sess_%s" % sess
    PARAM_GRID.append(p)

# Grid 7: CHoCH bars
for cb in (4, 6, 8, 12):
    p = dict(BASE_PARAMS); p["choch_bars"] = cb
    p["id"] = "lgc_choch%d" % cb
    PARAM_GRID.append(p)

# Deduplicate
seen_ids = set()
PARAM_GRID_DEDUP = []
for p in PARAM_GRID:
    if p["id"] not in seen_ids:
        seen_ids.add(p["id"])
        PARAM_GRID_DEDUP.append(p)


# ===========================================================================
#  MAIN
# ===========================================================================

def main():
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("backtest_range_lgc.py -- Range Liquidity Grab + CHoCH + FVG", flush=True)
    print("Started:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), flush=True)
    t0 = _time.time()

    # Load data
    print("\n  Loading 15m data (365d, 17 symbols)...", flush=True)
    all_data = load_data("365d")
    print("  Symbols loaded: %d" % len(all_data), flush=True)

    # Pre-extract DataFrames for sim engine
    df_by_sym = {sym: d["df15"] for sym, d in all_data.items()}

    all_results = []

    # =====================================================================
    # Phase 1: Parameter grid search
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 1 -- Parameter Grid Search (%d configs)" % len(PARAM_GRID_DEDUP), flush=True)
    print("=" * 72, flush=True)

    for params in PARAM_GRID_DEDUP:
        pid = params["id"]

        # Generate signals for all symbols
        all_sigs = {}
        total_sigs = 0
        for sym, d in all_data.items():
            sigs = gen_lgc_signals(d["df15"], params)
            all_sigs[sym] = sigs
            total_sigs += len(sigs)

        res = run_sim_lgc(all_sigs, df_by_sym, params,
                          risk_pt=0.010, cap_frac=0.30, max_pos=4)
        viable = res["profit_factor"] >= 1.5 and res["trades"] >= 10
        tag = "VIABLE" if viable else ("LOW" if res["trades"] >= 5 else "DEAD")
        print("  %-30s sigs=%4d n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%% [%s]"
              % (pid, total_sigs, res["trades"], res["profit_factor"],
                 res["pnl_pct"], res["max_dd_pct"], tag), flush=True)
        all_results.append({"id": pid, "params": {k: v for k, v in params.items() if k != "id"},
                             **{k: v for k, v in res.items() if k != "exit_reasons"}})

    all_results.sort(key=lambda x: (x["profit_factor"] if x["trades"] >= 15 else -1), reverse=True)

    # =====================================================================
    # Phase 2: Deep test on top 5 configs
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 2 -- Deep test on top configs", flush=True)
    print("=" * 72, flush=True)

    top5 = [r for r in all_results if r["trades"] >= 15][:5]
    if not top5:
        top5 = all_results[:5]

    wf_results = []
    for r in top5:
        pid = r["id"]
        params = next((p for p in PARAM_GRID_DEDUP if p["id"] == pid), None)
        if params is None:
            continue

        # Regenerate signals
        all_sigs = {}
        for sym, d in all_data.items():
            all_sigs[sym] = gen_lgc_signals(d["df15"], params)

        # Walk-forward 67/33
        train = run_sim_lgc(all_sigs, df_by_sym, params,
                             risk_pt=0.010, cap_frac=0.30, max_pos=4,
                             start_frac=0.0, end_frac=0.67)
        test = run_sim_lgc(all_sigs, df_by_sym, params,
                            risk_pt=0.010, cap_frac=0.30, max_pos=4,
                            start_frac=0.67, end_frac=1.0)
        passed = test["profit_factor"] >= 1.0 and test["pnl_pct"] > -5.0
        tag = "PASS" if passed else "FAIL"
        print("  WF %-25s train=%+6.1f%%(PF=%.2f n=%d) test=%+6.1f%%(PF=%.2f n=%d) [%s]"
              % (pid, train["pnl_pct"], train["profit_factor"], train["trades"],
                 test["pnl_pct"], test["profit_factor"], test["trades"], tag), flush=True)
        wf_results.append({"id": pid, "train_pf": train["profit_factor"],
            "test_pf": test["profit_factor"], "train_n": train["trades"],
            "test_n": test["trades"], "passed": passed, "tag": tag})

    # =====================================================================
    # Phase 3: Final decision
    # =====================================================================
    print("\n" + "=" * 72, flush=True)
    print("  PHASE 3 -- FINAL DECISION", flush=True)
    print("=" * 72, flush=True)

    if all_results:
        best = all_results[0]
        print("  Best config: %s" % best["id"], flush=True)
        print("    PF=%.2f  trades=%d  WR=%.1f%%  DD=%.1f%%  PnL=%+.1f%%"
              % (best["profit_factor"], best["trades"], best["wr_pct"],
                 best["max_dd_pct"], best["pnl_pct"]), flush=True)

        viable_wf = [w for w in wf_results if w["passed"]]
        add_to_live = (best["profit_factor"] >= 2.0 and
                       best["trades"] >= 30 and
                       len(viable_wf) > 0)
        verdict = "ADD TO LIVE SYSTEM" if add_to_live else "DO NOT ADD (insufficient edge)"
        print("\n  Min requirements: PF>=2.0, trades>=30/year, WF PASS", flush=True)
        print("  Verdict: %s" % verdict, flush=True)

        # Top 3 for reference
        print("\n  Top 5 configs:", flush=True)
        for r in all_results[:5]:
            wf_tag = next((w["tag"] for w in wf_results if w["id"] == r["id"]), "N/A")
            print("    %-25s n=%3d PF=%.2f PnL=%+7.1f%% DD=%4.1f%% WF=%s"
                  % (r["id"], r["trades"], r["profit_factor"],
                     r["pnl_pct"], r["max_dd_pct"], wf_tag), flush=True)

    # Save results
    out_path = RESULTS_DIR / ("backtest_lgc_%s.json" % ts_str)
    save_data = {
        "metadata": {
            "timestamp": ts_str,
            "symbols": len(all_data),
            "runtime_sec": round(_time.time() - t0, 1),
        },
        "p1_grid": [{k: v for k, v in r.items() if k != "params"} for r in all_results],
        "p2_walkforward": wf_results,
        "best_params": BASE_PARAMS,
    }
    with open(out_path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print("\n  Results saved:", out_path, flush=True)
    print("Total runtime: %d min %d sec" % divmod(int(_time.time() - t0), 60), flush=True)


if __name__ == "__main__":
    main()
