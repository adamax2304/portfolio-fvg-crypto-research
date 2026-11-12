"""
backtest_session_v4.py -- Session V4 (2026-04-16)

V3 showed naive Asian breakout doesn't work with realistic execution.
But user is right: US session DOES show strong directional moves.
What's wrong? We were using wrong signal type. Need smarter entry logic.

NEW HYPOTHESES:
  H1. NY Opening Range Breakout (first 1h of NY defines range, trade its break)
  H2. Judas Swing (NY first sweeps Asian extreme, THEN reverses → trade reversal)
  H3. Compression + Expansion (Asian + London must be tight range → NY expansion)
  H4. Trend-pullback in NY (strong 1h trend + pullback to VWAP/EMA21 → entry)
  H5. First decisive 15m candle (wait for big NY candle body, trade continuation)
  H6. Combined Asian+London range breakout (ref = high/low of 00-13 UTC)
  H7. Session VWAP bounce (price near session VWAP + trend = continuation)

Realistic execution kept from V3:
  - Entry at next-bar open (market) OR limit retest
  - Slippage 0.05% + taker fee 0.055%
  - Session end forces exit
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

logger = logging.getLogger("session_bt_v4")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

START_BALANCE = 1000.0
TAKER_FEE = 0.00055
SLIPPAGE  = 0.0005
MAX_LEVERAGE = 50
RISK_PER_TRADE = 0.01

TOP14 = ["BTC/USDT:USDT", "ETH/USDT:USDT", "BNB/USDT:USDT", "SOL/USDT:USDT",
         "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
         "LINK/USDT:USDT", "DOT/USDT:USDT", "NEAR/USDT:USDT", "UNI/USDT:USDT",
         "APT/USDT:USDT", "ATOM/USDT:USDT"]


def _load(symbol, tf, days_tag="365d"):
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in (days_tag, "180d", "90d"):
        for ext in (".parquet", ".csv"):
            fp = CACHE_DIR / f"{safe}_{tf}_{tag}{ext}"
            if fp.exists():
                try:
                    df = pd.read_parquet(fp) if ext == ".parquet" else pd.read_csv(fp)
                    for c in ("open", "high", "low", "close", "volume"):
                        if c in df.columns:
                            df[c] = df[c].astype(float)
                    if "timestamp" in df.columns:
                        df["timestamp"] = df["timestamp"].astype(np.int64)
                    return df
                except Exception:
                    continue
    return None


def _atr(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def _ema(x, period):
    x = np.asarray(x, dtype=float)
    n = len(x)
    out = np.zeros(n)
    if n == 0: return out
    k = 2 / (period + 1)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out


def _adx(hi, lo, cl, period=14):
    n = len(cl)
    tr = np.zeros(n); plus_dm = np.zeros(n); minus_dm = np.zeros(n)
    for i in range(1, n):
        up = hi[i] - hi[i-1]; dn = lo[i-1] - lo[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr_w = np.zeros(n); plus_s = np.zeros(n); minus_s = np.zeros(n)
    if n <= period: return np.zeros(n), np.zeros(n), np.zeros(n)
    atr_w[period] = np.sum(tr[1:period+1])
    plus_s[period] = np.sum(plus_dm[1:period+1])
    minus_s[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        atr_w[i] = atr_w[i-1] - atr_w[i-1]/period + tr[i]
        plus_s[i] = plus_s[i-1] - plus_s[i-1]/period + plus_dm[i]
        minus_s[i] = minus_s[i-1] - minus_s[i-1]/period + minus_dm[i]
    plus_di = np.zeros(n); minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_w[i] > 0:
            plus_di[i] = 100 * plus_s[i] / atr_w[i]
            minus_di[i] = 100 * minus_s[i] / atr_w[i]
    dx = np.zeros(n)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        if s > 0: dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s
    adx = np.zeros(n)
    if n >= 2*period:
        adx[2*period-1] = np.mean(dx[period:2*period])
        for i in range(2*period, n):
            adx[i] = (adx[i-1] * (period-1) + dx[i]) / period
    return adx, plus_di, minus_di


def _supertrend(hi, lo, cl, period=10, mult=3.0):
    atr = _atr(hi, lo, cl, period)
    n = len(cl)
    med = (hi + lo) / 2.0
    upper = med + mult * atr; lower = med - mult * atr
    fub = np.copy(upper); flb = np.copy(lower)
    for i in range(1, n):
        if upper[i] < fub[i-1] or cl[i-1] > fub[i-1]: fub[i] = upper[i]
        else: fub[i] = fub[i-1]
        if lower[i] > flb[i-1] or cl[i-1] < flb[i-1]: flb[i] = lower[i]
        else: flb[i] = flb[i-1]
    st = np.zeros(n, dtype=np.int32); st[0] = 1
    for i in range(1, n):
        if st[i-1] == 1 and cl[i] < flb[i]: st[i] = -1
        elif st[i-1] == -1 and cl[i] > fub[i]: st[i] = 1
        else: st[i] = st[i-1]
    return st


def prep_15m_arrays(df):
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    ts = df["timestamp"].values.astype(np.int64)
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    hours = dt.hour.values.astype(np.int32)
    mins = dt.minute.values.astype(np.int32)
    dates = np.array(dt.strftime("%Y-%m-%d"))
    dow = dt.dayofweek.values.astype(np.int32)
    atr = _atr(h, l, c, 14)
    volma = pd.Series(v).rolling(20, min_periods=1).mean().values
    ema21 = _ema(c, 21)
    # Session-rooted VWAP per day (reset each 00:00 UTC)
    vwap = np.zeros_like(c)
    pv_cum = 0.0; v_cum = 0.0; cur_date = ""
    for i in range(len(c)):
        if dates[i] != cur_date:
            cur_date = dates[i]; pv_cum = 0.0; v_cum = 0.0
        typical = (h[i] + l[i] + c[i]) / 3
        pv_cum += typical * v[i]; v_cum += v[i]
        vwap[i] = pv_cum / v_cum if v_cum > 0 else typical
    # Session NY VWAP (reset at 13:00 UTC)
    ny_vwap = np.zeros_like(c)
    pv_cum = 0.0; v_cum = 0.0; cur_ny_date = ""
    for i in range(len(c)):
        ny_key = f"{dates[i]}_NY"
        if hours[i] == 13 and mins[i] < 15:
            if cur_ny_date != ny_key:
                cur_ny_date = ny_key; pv_cum = 0.0; v_cum = 0.0
        if 13 <= hours[i] < 22 and cur_ny_date == ny_key:
            typical = (h[i] + l[i] + c[i]) / 3
            pv_cum += typical * v[i]; v_cum += v[i]
            ny_vwap[i] = pv_cum / v_cum if v_cum > 0 else typical
        else:
            ny_vwap[i] = c[i]
    return {
        "o": o, "h": h, "l": l, "c": c, "v": v, "ts": ts,
        "hour": hours, "min": mins, "date": dates, "dow": dow,
        "atr": atr, "volma": volma, "ema21": ema21,
        "vwap": vwap, "ny_vwap": ny_vwap, "n": len(c),
    }


def build_session_ranges(arr):
    n = arr["n"]
    h, l, c = arr["h"], arr["l"], arr["c"]
    date = arr["date"]; hour = arr["hour"]
    by_date = defaultdict(list)
    for i in range(n):
        by_date[date[i]].append(i)
    ranges = {}
    dates_sorted = sorted(by_date.keys())
    prev_hi = None; prev_lo = None; prev_close = None
    for d in dates_sorted:
        idxs = by_date[d]
        a_i = [i for i in idxs if 0 <= hour[i] < 7]    # Asian 00-07
        l_i = [i for i in idxs if 7 <= hour[i] < 13]   # London 07-13
        al_i = [i for i in idxs if 0 <= hour[i] < 13]  # Asian+London combined
        n_i = [i for i in idxs if 13 <= hour[i] < 22]  # NY 13-22
        ny1_i = [i for i in idxs if hour[i] == 13]     # NY first hour 13-14
        ahi = max((h[i] for i in a_i), default=np.nan)
        alo = min((l[i] for i in a_i), default=np.nan)
        lhi = max((h[i] for i in l_i), default=np.nan)
        llo = min((l[i] for i in l_i), default=np.nan)
        alhi = max((h[i] for i in al_i), default=np.nan)
        allo = min((l[i] for i in al_i), default=np.nan)
        nhi = max((h[i] for i in n_i), default=np.nan)
        nlo = min((l[i] for i in n_i), default=np.nan)
        ny1hi = max((h[i] for i in ny1_i), default=np.nan)
        ny1lo = min((l[i] for i in ny1_i), default=np.nan)
        ranges[d] = {
            "asian_hi": ahi, "asian_lo": alo,
            "london_hi": lhi, "london_lo": llo,
            "al_hi": alhi, "al_lo": allo,       # Asian+London combined
            "ny_hi": nhi, "ny_lo": nlo,
            "ny1_hi": ny1hi, "ny1_lo": ny1lo,   # first hour of NY (Opening Range)
            "pd_hi": prev_hi, "pd_lo": prev_lo, "pd_close": prev_close,
        }
        day_hi = max((h[i] for i in idxs), default=prev_hi)
        day_lo = min((l[i] for i in idxs), default=prev_lo)
        day_cl = (c[idxs[-1]] if idxs else prev_close)
        prev_hi = day_hi; prev_lo = day_lo; prev_close = day_cl
    return ranges


def compute_htf_trends(df_4h, df_1h):
    out = {}
    if df_1h is not None and len(df_1h) > 60:
        cl = df_1h["close"].values.astype(float)
        hi = df_1h["high"].values.astype(float)
        lo = df_1h["low"].values.astype(float)
        ts = df_1h["timestamp"].values.astype(np.int64)
        ema50 = _ema(cl, 50)
        st = _supertrend(hi, lo, cl, 10, 3.0)
        adx, pdi, mdi = _adx(hi, lo, cl, 14)
        # EMA50 slope over 5 bars
        slope = np.zeros(len(cl))
        for i in range(5, len(cl)):
            slope[i] = (ema50[i] - ema50[i-5]) / (ema50[i-5] + 1e-9)
        out["ema50_1h"] = {int(ts[i]): (1 if cl[i] > ema50[i] else -1) for i in range(50, len(cl))}
        out["st_1h"] = {int(ts[i]): int(st[i]) for i in range(len(cl))}
        def adir(i):
            if adx[i] < 20: return 0
            return 1 if pdi[i] > mdi[i] else -1
        def strong_adir(i):
            if adx[i] < 25: return 0
            return 1 if pdi[i] > mdi[i] else -1
        out["adx_1h"] = {int(ts[i]): adir(i) for i in range(30, len(cl))}
        out["adx25_1h"] = {int(ts[i]): strong_adir(i) for i in range(30, len(cl))}
        # Combined strong trend: ADX>25 AND EMA50 slope aligned
        def combo(i):
            a = strong_adir(i)
            if a == 0: return 0
            if a == 1 and slope[i] > 0.0005: return 1
            if a == -1 and slope[i] < -0.0005: return -1
            return 0
        out["trend_combo_1h"] = {int(ts[i]): combo(i) for i in range(50, len(cl))}
    if df_4h is not None and len(df_4h) > 60:
        cl = df_4h["close"].values.astype(float)
        ts = df_4h["timestamp"].values.astype(np.int64)
        ema50 = _ema(cl, 50)
        out["ema50_4h"] = {int(ts[i]): (1 if cl[i] > ema50[i] else -1) for i in range(50, len(cl))}
    return out


def align_htf(htf_map, ts_15m):
    if not htf_map:
        return np.zeros(len(ts_15m), dtype=np.int32)
    ks = np.array(sorted(htf_map.keys()), dtype=np.int64)
    vs = np.array([htf_map[int(k)] for k in ks], dtype=np.int32)
    idx = np.searchsorted(ks, ts_15m, side="right") - 1
    idx = np.clip(idx, 0, len(vs)-1)
    aligned = vs[idx]
    aligned[ts_15m < ks[0]] = 0
    return aligned


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGIES
# ═══════════════════════════════════════════════════════════════════════════

def _place_sl_tp(entry_ref, direction, params, atr_i, a_range):
    sl_mode = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 2.5)
    sl_pct = params.get("sl_pct", 0.015)
    tp_mode = params.get("tp_mode", "fixed_r")
    tp_r = params.get("tp_r", 2.0)
    tp_range_mult = params.get("tp_range_mult", 2.0)
    if sl_mode == "atr":
        sl = entry_ref - atr_i * sl_atr_mult if direction == "LONG" else entry_ref + atr_i * sl_atr_mult
    elif sl_mode == "pct":
        sl = entry_ref * (1 - sl_pct) if direction == "LONG" else entry_ref * (1 + sl_pct)
    else:
        sl = entry_ref - atr_i * 2.0 if direction == "LONG" else entry_ref + atr_i * 2.0
    risk_px = abs(entry_ref - sl)
    if tp_mode == "fixed_r":
        tp = entry_ref + risk_px * tp_r if direction == "LONG" else entry_ref - risk_px * tp_r
    elif tp_mode == "range_mult":
        tp = entry_ref + a_range * tp_range_mult if direction == "LONG" else entry_ref - a_range * tp_range_mult
    else:
        tp = entry_ref + risk_px * 2.0 if direction == "LONG" else entry_ref - risk_px * 2.0
    return sl, tp, risk_px


def _check_htf(params, htf_aligned_map, i, direction):
    f = params.get("htf_filter")
    if f is None or htf_aligned_map is None: return True
    arr = htf_aligned_map.get(f)
    if arr is None: return True
    flag = int(arr[i])
    if direction == "LONG" and flag != 1: return False
    if direction == "SHORT" and flag != -1: return False
    return True


def gen_ny_orb(arr, sess_ranges, params, htf_aligned_map=None):
    """
    H1: NY Opening Range Breakout.
    OR = 13:00-14:00 UTC. Entry on break after 14:00. Window till 17:00.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, mn, date = arr["hour"], arr["min"], arr["date"]
    atr = arr["atr"]; volma = arr["volma"]; v = arr["v"]
    n = arr["n"]
    entry_start_hr = 14
    entry_end_hr = params.get("entry_end_hr", 17)
    min_or_atr = params.get("min_or_atr", 0.3)
    max_or_atr = params.get("max_or_atr", 4.0)
    vol_mult = params.get("vol_mult", 0.0)
    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        orh, orl = r["ny1_hi"], r["ny1_lo"]
        if np.isnan(orh) or np.isnan(orl): continue
        if hour[i] < entry_start_hr or hour[i] >= entry_end_hr: continue
        if last_sig_date == d: continue
        or_range = orh - orl
        if or_range <= 0 or atr[i] <= 0: continue
        rR = or_range / atr[i]
        if rR < min_or_atr or rR > max_or_atr: continue

        direction = None
        if h[i] > orh and c[i] > orh:
            direction = "LONG"
        elif l[i] < orl and c[i] < orl:
            direction = "SHORT"
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue
        if vol_mult > 0 and volma[i] > 0 and v[i] < volma[i] * vol_mult: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], or_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(or_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_judas_swing(arr, sess_ranges, params, htf_aligned_map=None):
    """
    H2: Judas Swing (sweep + reverse).
    In NY first 2 hours (13-15): price sweeps Asian hi/lo (fake breakout) then closes back inside.
    Enter in reversal direction.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    sweep_window_h = params.get("sweep_window_h", 2)  # after NY open
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo): continue
        if hour[i] < 13 or hour[i] >= 13 + sweep_window_h: continue
        if last_sig_date == d: continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0: continue

        direction = None
        # SHORT: swept above asian_hi, closes back below
        if h[i] > ahi and c[i] < ahi:
            direction = "SHORT"
            trigger = ahi
        # LONG: swept below asian_lo, closes back above
        elif l[i] < alo and c[i] > alo:
            direction = "LONG"
            trigger = alo
        if direction is None: continue
        # Require the sweep bar to have reasonable size (body < 70% of total, wick > 30%)
        body = abs(c[i] - o[i])
        total = h[i] - l[i]
        if total <= 0: continue
        if body / total > 0.7: continue  # must have wick/rejection
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], a_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_compression(arr, sess_ranges, params, htf_aligned_map=None):
    """
    H3: Asian+London compression then NY breakout.
    Filter: al_range/atr < compress_thr (tight range).
    Entry: NY breakout of al_hi/al_lo.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    compress_thr = params.get("compress_thr", 2.0)
    max_entry_hr = params.get("max_entry_hr", 4)
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        alhi, allo = r["al_hi"], r["al_lo"]
        if np.isnan(alhi) or np.isnan(allo): continue
        if hour[i] < 13 or hour[i] >= 13 + max_entry_hr: continue
        if last_sig_date == d: continue
        al_range = alhi - allo
        if al_range <= 0 or atr[i] <= 0: continue
        if al_range / atr[i] > compress_thr: continue  # not compressed enough

        direction = None
        if h[i] > alhi and c[i] > alhi:
            direction = "LONG"
        elif l[i] < allo and c[i] < allo:
            direction = "SHORT"
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], al_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(al_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_trend_pullback(arr, sess_ranges, params, htf_aligned_map=None):
    """
    H4: Strong 1h trend + pullback to EMA21 or NY VWAP during NY session → entry.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]; ema21 = arr["ema21"]; ny_vwap = arr["ny_vwap"]
    n = arr["n"]
    ref_line = params.get("ref_line", "ema21")  # "ema21" or "ny_vwap"
    pullback_atr_min = params.get("pullback_atr_min", 0.2)  # how deep pullback
    entry_end_hr = params.get("entry_end_hr", 19)
    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        if hour[i] < 13 or hour[i] >= entry_end_hr: continue
        if last_sig_date == d: continue
        if atr[i] <= 0: continue
        trend = 0
        if htf_aligned_map is not None:
            f = params.get("htf_filter", "trend_combo_1h")
            if htf_aligned_map.get(f) is not None:
                trend = int(htf_aligned_map[f][i])
        if trend == 0: continue

        ref = ema21[i] if ref_line == "ema21" else ny_vwap[i]
        if ref <= 0: continue
        direction = "LONG" if trend > 0 else "SHORT"
        # Pullback condition: price touched ref recently and bounces
        # For LONG: low[i] <= ref AND close[i] > ref AND close[i] > close[i-1]
        # For SHORT: high[i] >= ref AND close[i] < ref AND close[i] < close[i-1]
        touched = False
        if direction == "LONG":
            if l[i] <= ref and c[i] > ref and c[i] > c[i-1]:
                touched = True
                # Also require pullback depth ≥ min atr
                if (h[i-1] - l[i]) < pullback_atr_min * atr[i]: touched = False
        else:
            if h[i] >= ref and c[i] < ref and c[i] < c[i-1]:
                touched = True
                if (h[i] - l[i-1]) < pullback_atr_min * atr[i]: touched = False
        if not touched: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        a_range = atr[i] * 3  # fake range for TP
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], a_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_decisive_candle(arr, sess_ranges, params, htf_aligned_map=None):
    """
    H5: Wait for first decisive 15m candle in NY session.
    Decisive = body ≥ X * ATR AND body/total_range ≥ Y%. Enter in direction.
    """
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    body_atr_min = params.get("body_atr_min", 0.5)
    body_pct_min = params.get("body_pct_min", 0.6)
    entry_end_hr = params.get("entry_end_hr", 17)
    signals = []
    last_sig_date = None
    for i in range(50, n-1):
        d = date[i]
        if hour[i] < 13 or hour[i] >= entry_end_hr: continue
        if last_sig_date == d: continue
        if atr[i] <= 0: continue
        body = abs(c[i] - o[i])
        total = h[i] - l[i]
        if total <= 0: continue
        if body < body_atr_min * atr[i]: continue
        if body / total < body_pct_min: continue

        direction = "LONG" if c[i] > o[i] else "SHORT"
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        a_range = atr[i] * 3
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], a_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range), "date": d,
        })
        last_sig_date = d
    return signals


def gen_al_break(arr, sess_ranges, params, htf_aligned_map=None):
    """H6: Asian+London combined range breakout at NY."""
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr = arr["atr"]
    n = arr["n"]
    signals = []
    last_sig_date = None
    max_entry_hr = params.get("max_entry_hr", 4)
    min_range_atr = params.get("min_range_atr", 0.5)
    max_range_atr = params.get("max_range_atr", 5.0)
    for i in range(50, n-1):
        d = date[i]
        if d not in sess_ranges: continue
        r = sess_ranges[d]
        alhi, allo = r["al_hi"], r["al_lo"]
        if np.isnan(alhi) or np.isnan(allo): continue
        if hour[i] < 13 or hour[i] >= 13 + max_entry_hr: continue
        if last_sig_date == d: continue
        al_range = alhi - allo
        if al_range <= 0 or atr[i] <= 0: continue
        rR = al_range / atr[i]
        if rR < min_range_atr or rR > max_range_atr: continue

        direction = None
        if h[i] > alhi and c[i] > alhi:
            direction = "LONG"
        elif l[i] < allo and c[i] < allo:
            direction = "SHORT"
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], al_range)
        if risk_px <= 0: continue
        signals.append({
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(al_range), "date": d,
        })
        last_sig_date = d
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATE
# ═══════════════════════════════════════════════════════════════════════════

def simulate(sigs_by_sym, arrays_by_sym, session_end_hr=22, trail_cfg=None):
    balance = START_BALANCE
    completed = []
    peak = balance
    max_dd = 0.0

    for sym, sigs in sigs_by_sym.items():
        arr = arrays_by_sym[sym]
        h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
        hour, date = arr["hour"], arr["date"]
        atr = arr["atr"]
        n = arr["n"]
        for sig in sigs:
            i_entry = sig["i_entry"]
            entry_ref = sig["entry_ref"]
            sl_init = sig["sl"]
            tp = sig["tp"]
            direction = sig["direction"]
            risk_px = sig["risk_px"]
            entry_date = date[i_entry]
            risk_usd = balance * RISK_PER_TRADE
            if risk_px <= 0 or entry_ref <= 0: continue
            qty_usd = risk_usd / (risk_px / entry_ref)
            if qty_usd > balance * MAX_LEVERAGE: qty_usd = balance * MAX_LEVERAGE
            eff_entry = entry_ref * (1 + SLIPPAGE) if direction == "LONG" else entry_ref * (1 - SLIPPAGE)
            entry_fee = qty_usd * TAKER_FEE

            sl_cur = sl_init
            trail_active = False
            mfe = 0.0; mae = 0.0
            exit_reason = "EOS"
            exit_px = c[n-1]; exit_bar = n - 1

            for j in range(i_entry, n):
                if date[j] != entry_date:
                    exit_reason = "EOS"
                    exit_px = c[j-1]; exit_bar = j - 1
                    break
                if hour[j] >= session_end_hr:
                    exit_reason = "EOS"
                    exit_px = c[j]; exit_bar = j
                    break
                bar_h = h[j]; bar_l = l[j]

                if direction == "LONG":
                    mfe = max(mfe, (bar_h - eff_entry) / risk_px)
                    mae = max(mae, (eff_entry - bar_l) / risk_px)
                else:
                    mfe = max(mfe, (eff_entry - bar_l) / risk_px)
                    mae = max(mae, (bar_h - eff_entry) / risk_px)

                if trail_cfg is not None:
                    act_r = trail_cfg.get("activate_at_r", 1.0)
                    if not trail_active and mfe >= act_r: trail_active = True
                    if trail_active:
                        mode = trail_cfg.get("trail_mode", "swing")
                        if mode == "breakeven":
                            if direction == "LONG": sl_cur = max(sl_cur, eff_entry)
                            else: sl_cur = min(sl_cur, eff_entry)
                        elif mode == "atr":
                            mlt = trail_cfg.get("atr_mult", 2.0)
                            if direction == "LONG": sl_cur = max(sl_cur, c[j] - atr[j] * mlt)
                            else: sl_cur = min(sl_cur, c[j] + atr[j] * mlt)
                        elif mode == "swing":
                            k = trail_cfg.get("swing_bars", 5)
                            buf = trail_cfg.get("buffer_atr", 0.2)
                            kk = min(k, j - i_entry)
                            if kk >= 1:
                                if direction == "LONG":
                                    new_sl = float(np.min(l[j-kk:j+1])) - atr[j] * buf
                                    sl_cur = max(sl_cur, new_sl)
                                else:
                                    new_sl = float(np.max(h[j-kk:j+1])) + atr[j] * buf
                                    sl_cur = min(sl_cur, new_sl)

                # Conservative: SL first
                if direction == "LONG":
                    if bar_l <= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_h >= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break
                else:
                    if bar_h >= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur; exit_bar = j; break
                    if bar_l <= tp:
                        exit_reason = "TP"; exit_px = tp; exit_bar = j; break

            eff_exit = exit_px * (1 - SLIPPAGE) if direction == "LONG" else exit_px * (1 + SLIPPAGE)
            exit_fee = qty_usd * TAKER_FEE
            if direction == "LONG":
                pnl_usd = qty_usd * (eff_exit - eff_entry) / eff_entry - entry_fee - exit_fee
            else:
                pnl_usd = qty_usd * (eff_entry - eff_exit) / eff_entry - entry_fee - exit_fee
            balance += pnl_usd
            peak = max(peak, balance)
            dd = (peak - balance) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
            exit_r = ((eff_exit - eff_entry) if direction == "LONG" else (eff_entry - eff_exit)) / risk_px
            completed.append({
                "sym": sym, "direction": direction,
                "i_entry": i_entry, "exit_bar": exit_bar,
                "bars": exit_bar - i_entry,
                "reason": exit_reason, "pnl_usd": pnl_usd, "won": pnl_usd > 0,
                "exit_r": exit_r, "mfe": mfe, "mae": mae,
                "entry": entry_ref, "exit": exit_px, "sl": sl_init, "tp": tp, "date": entry_date,
            })
    return _aggregate(completed, balance, max_dd)


def _aggregate(completed, balance, max_dd):
    if not completed:
        return {"trades": 0, "wins": 0, "losses": 0, "wr_pct": 0.0,
                "pnl_usd": 0.0, "pnl_pct": 0.0, "profit_factor": 0.0,
                "max_dd_pct": 0.0, "avg_r": 0.0, "avg_mfe": 0.0,
                "avg_bars": 0.0, "final_bal": balance, "exit_reasons": {}}
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / len(completed) * 100
    reasons = defaultdict(int)
    for t in completed: reasons[t["reason"]] += 1
    return {
        "trades": len(completed), "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "pnl_pct": round((balance - START_BALANCE) / START_BALANCE * 100, 2),
        "profit_factor": round(pf, 3),
        "max_dd_pct": round(max_dd, 2),
        "avg_r": round(float(np.mean([t["exit_r"] for t in completed])), 3),
        "avg_mfe": round(float(np.mean([t["mfe"] for t in completed])), 3),
        "avg_bars": round(float(np.mean([t["bars"] for t in completed])), 1),
        "final_bal": round(balance, 2),
        "exit_reasons": dict(reasons),
    }


def run_strategy(name, gen_fn, arrays, ranges, htf_aligned_bysym, params,
                  trail_cfg=None, session_end_hr=22, compute_per_sym=False):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_fn(arr, ranges[sym], params, htf_aligned_bysym.get(sym))
    res = simulate(sigs, arrays, session_end_hr=session_end_hr, trail_cfg=trail_cfg)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    if compute_per_sym:
        per_sym = {}
        for sym in sigs:
            per_sym[sym] = simulate({sym: sigs[sym]}, {sym: arrays[sym]},
                                     session_end_hr=session_end_hr, trail_cfg=trail_cfg)
        res["per_sym"] = per_sym
    return res


def log_r(name, r):
    logger.info("%-36s trades=%4d WR=%5.1f%% PF=%5.2f PnL=%+8.1f%% DD=%5.2f%% avgR=%+.2f avgMFE=%+.2f",
                 name, r["trades"], r["wr_pct"], r["profit_factor"],
                 r["pnl_pct"], r["max_dd_pct"], r["avg_r"], r["avg_mfe"])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    logger.info("=== Session V4 (NEW hypotheses: ORB/Judas/Compression/Pullback/Decisive/AL-break) ===")
    arrays = {}; ranges = {}; htf_aligned_bysym = {}
    loaded = []
    for sym in TOP14:
        df15 = _load(sym, "15m", "365d")
        if df15 is None or len(df15) < 5000: continue
        df1h = _load(sym, "1h", "365d")
        df4h = _load(sym, "4h", "365d")
        arr = prep_15m_arrays(df15)
        rng = build_session_ranges(arr)
        htfs = compute_htf_trends(df4h, df1h)
        aligned = {nm: align_htf(m, arr["ts"]) for nm, m in htfs.items()}
        arrays[sym] = arr; ranges[sym] = rng; htf_aligned_bysym[sym] = aligned
        loaded.append(sym)
    logger.info("Loaded %d symbols", len(loaded))
    be = {s: arrays[s] for s in ["BTC/USDT:USDT", "ETH/USDT:USDT"] if s in arrays}
    be_r = {s: ranges[s] for s in be}
    be_htf = {s: htf_aligned_bysym[s] for s in be}

    results = []
    trail_sw5 = {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}

    # ═══ TEST 1: NY Opening Range Breakout (H1) ═══
    logger.info("--- TEST 1: NY Opening Range Breakout (13-14 OR, trade break after 14) ---")
    orb_configs = [
        ("ORB_none_R2",    {"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.0}, None),
        ("ORB_adx_R2",     {"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"adx_1h"}, None),
        ("ORB_st_R2",      {"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"st_1h"}, None),
        ("ORB_combo_R2",   {"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"trend_combo_1h"}, None),
        ("ORB_combo_R3_sw5", {"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
        ("ORB_combo_rng2_sw5", {"sl_mode":"atr","sl_atr_mult":2.5,"tp_mode":"range_mult","tp_range_mult":2.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
        ("ORB_adx_vol1.2", {"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.5,"htf_filter":"adx_1h","vol_mult":1.2}, None),
    ]
    for nm, p, tr in orb_configs:
        r = run_strategy(nm, gen_ny_orb, arrays, ranges, htf_aligned_bysym, p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r); log_r(nm, r)

    # ═══ TEST 2: Judas swing (H2) ═══
    logger.info("--- TEST 2: Judas swing (fake sweep + reverse) ---")
    judas_configs = [
        ("Judas_none_R2",   {"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0}, None),
        ("Judas_adx_R2",    {"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"adx_1h"}, None),
        ("Judas_st_R2",     {"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"st_1h"}, None),
        ("Judas_combo_R2",  {"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"trend_combo_1h"}, None),
        ("Judas_adx_R3_sw5",{"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"adx_1h"}, trail_sw5),
        ("Judas_no_filter_R3_sw5",{"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0}, trail_sw5),
    ]
    for nm, p, tr in judas_configs:
        r = run_strategy(nm, gen_judas_swing, arrays, ranges, htf_aligned_bysym, p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r); log_r(nm, r)

    # ═══ TEST 3: Compression + breakout (H3) ═══
    logger.info("--- TEST 3: Asian+London compression + NY breakout ---")
    comp_configs = [
        ("Comp2.0_adx_R2",  {"compress_thr":2.0,"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"adx_1h"}, None),
        ("Comp1.5_adx_R2",  {"compress_thr":1.5,"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"adx_1h"}, None),
        ("Comp1.5_combo_R25",{"compress_thr":1.5,"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":2.5,"htf_filter":"trend_combo_1h"}, None),
        ("Comp1.5_combo_R3_sw5",{"compress_thr":1.5,"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
        ("Comp2.0_combo_R3_sw5",{"compress_thr":2.0,"sl_mode":"atr","sl_atr_mult":2.0,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
    ]
    for nm, p, tr in comp_configs:
        r = run_strategy(nm, gen_compression, arrays, ranges, htf_aligned_bysym, p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r); log_r(nm, r)

    # ═══ TEST 4: Trend pullback (H4) ═══
    logger.info("--- TEST 4: Trend pullback to EMA21 / NY VWAP ---")
    pullback_configs = [
        ("Pb_ema21_adx_R2",     {"ref_line":"ema21","htf_filter":"adx_1h","sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0}, None),
        ("Pb_ema21_combo_R2",   {"ref_line":"ema21","htf_filter":"trend_combo_1h","sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0}, None),
        ("Pb_ema21_combo_R3_sw5",{"ref_line":"ema21","htf_filter":"trend_combo_1h","sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0}, trail_sw5),
        ("Pb_vwap_combo_R2",    {"ref_line":"ny_vwap","htf_filter":"trend_combo_1h","sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0}, None),
        ("Pb_vwap_combo_R3_sw5",{"ref_line":"ny_vwap","htf_filter":"trend_combo_1h","sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0}, trail_sw5),
        ("Pb_vwap_st_R3_sw5",   {"ref_line":"ny_vwap","htf_filter":"st_1h","sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0}, trail_sw5),
    ]
    for nm, p, tr in pullback_configs:
        r = run_strategy(nm, gen_trend_pullback, arrays, ranges, htf_aligned_bysym, p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r); log_r(nm, r)

    # ═══ TEST 5: Decisive candle (H5) ═══
    logger.info("--- TEST 5: First decisive 15m candle in NY ---")
    dc_configs = [
        ("DC_0.5_0.6_none",     {"body_atr_min":0.5,"body_pct_min":0.6,"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0}, None),
        ("DC_0.7_0.6_adx",      {"body_atr_min":0.7,"body_pct_min":0.6,"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":2.0,"htf_filter":"adx_1h"}, None),
        ("DC_0.8_0.65_combo_R3_sw5",{"body_atr_min":0.8,"body_pct_min":0.65,"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
        ("DC_1.0_0.7_combo_R3_sw5",{"body_atr_min":1.0,"body_pct_min":0.7,"sl_mode":"atr","sl_atr_mult":1.5,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
    ]
    for nm, p, tr in dc_configs:
        r = run_strategy(nm, gen_decisive_candle, arrays, ranges, htf_aligned_bysym, p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r); log_r(nm, r)

    # ═══ TEST 6: Asian+London combined breakout (H6) ═══
    logger.info("--- TEST 6: Asian+London combined range breakout ---")
    al_configs = [
        ("AL_adx_R2",        {"sl_mode":"atr","sl_atr_mult":2.5,"tp_mode":"range_mult","tp_range_mult":2.0,"htf_filter":"adx_1h"}, None),
        ("AL_combo_R2",      {"sl_mode":"atr","sl_atr_mult":2.5,"tp_mode":"range_mult","tp_range_mult":2.0,"htf_filter":"trend_combo_1h"}, None),
        ("AL_combo_R3_sw5",  {"sl_mode":"atr","sl_atr_mult":2.5,"tp_mode":"fixed_r","tp_r":3.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
        ("AL_combo_rngm2_sw5",{"sl_mode":"atr","sl_atr_mult":2.5,"tp_mode":"range_mult","tp_range_mult":2.0,"htf_filter":"trend_combo_1h"}, trail_sw5),
    ]
    for nm, p, tr in al_configs:
        r = run_strategy(nm, gen_al_break, arrays, ranges, htf_aligned_bysym, p, trail_cfg=tr, compute_per_sym=True)
        r["config_id"] = nm; r["params"] = p; r["trail_cfg"] = tr
        results.append(r); log_r(nm, r)

    # ═══ REPORT ═══
    out_path = RESULTS_DIR / f"backtest_session_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(out_path, "w") as f:
            json.dump({"symbols": loaded, "results": results}, f, indent=2, default=str)
        logger.info("Saved %s (%.1fs)", out_path.name, _time.time()-t0)
    except Exception as e:
        logger.warning("Save failed: %s", e)

    def _fmt(r):
        return (f"{r['config_id'][:36]:36s} {r['trades']:>5} {r['wr_pct']:>5.1f} "
                f"{r['profit_factor']:>5.2f} {r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f} {r['avg_r']:>+6.2f}")

    print("\n" + "="*110)
    print("OVERALL TOP-20 by PF (min 40 trades)")
    print(f"{'config':36s} {'trades':>5} {'WR%':>5} {'PF':>5} {'PnL%':>8} {'DD%':>6} {'avgR':>6}")
    flt = [r for r in results if r.get("trades", 0) >= 40]
    for r in sorted(flt, key=lambda x: x["profit_factor"], reverse=True)[:20]:
        print(_fmt(r))

    print("\nOVERALL TOP-10 by PnL (min 40 trades)")
    for r in sorted(flt, key=lambda x: x["pnl_pct"], reverse=True)[:10]:
        print(_fmt(r))

    # Per-family comparison (best in each test)
    print("\n" + "="*110)
    print("BEST per hypothesis family (min 40 trades, ranked by PnL)")
    families = {
        "H1 NY ORB":      "ORB_",
        "H2 Judas":       "Judas_",
        "H3 Compression": "Comp",
        "H4 Pullback":    "Pb_",
        "H5 Decisive":    "DC_",
        "H6 AL-break":    "AL_",
    }
    for fam_name, pref in families.items():
        candidates = [r for r in flt if r["config_id"].startswith(pref)]
        if not candidates:
            print(f"{fam_name:20s} — no qualifying configs")
            continue
        best = max(candidates, key=lambda r: r["pnl_pct"])
        print(f"{fam_name:20s} {_fmt(best)}")

    # Show per-symbol for top-3 PnL configs
    print("\n" + "="*110)
    print("Per-symbol breakdown for TOP-3 by PnL")
    for r in sorted(flt, key=lambda x: x["pnl_pct"], reverse=True)[:3]:
        print(f"\n>>> {r['config_id']} (overall PF={r['profit_factor']} PnL={r['pnl_pct']}%)")
        per = r.get("per_sym", {})
        if per:
            sorted_sym = sorted(per.items(), key=lambda kv: kv[1]["pnl_pct"], reverse=True)
            for sym, sr in sorted_sym:
                s = sym.replace("/USDT:USDT", "")
                profit_marker = "✓" if sr["pnl_pct"] > 0 else "✗"
                print(f"  {profit_marker} {s:6s} trades={sr['trades']:>4} WR={sr['wr_pct']:>5.1f}% PF={sr['profit_factor']:>5.2f} PnL={sr['pnl_pct']:>+7.1f}%  avgR={sr['avg_r']:>+.2f}")


if __name__ == "__main__":
    main()
