"""
backtest_session_v2.py -- Session strategy V2 (2026-04-16)

User feedback from Phase 1-5 (v1):
  - ATR*1.5 SL too tight for volatile crypto sessions (wiped out by noise)
  - Need WIDER structural SL (behind swing high/low with ATR buffer)
  - Need TP at nearest level (PDH/PDL, prior session extreme) not fixed R
  - Need dynamic trail to capture full session move
  - Compare HTF trend indicators (not just daily EMA200): EMA50 4h, Supertrend 1h/4h, ADX, MACD
  - Optional grid entry (scale-in)

Phases:
  A. Session statistics (avg range%, excursion%, net move, bull day %)
  B. SL mode comparison (ATR / Pct / Structural)
  C. HTF trend filter comparison
  D. Dynamic trailing variants
  E. Level-based TP + trail combinations

Symbols: BTC, ETH.  Data: 365d 15m + 1h/4h/1d.
Run: py -3 multi_agent/backtests/backtest_session_v2.py
"""
from __future__ import annotations
import json, logging, sys, time as _time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("session_bt_v2")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

SYMBOLS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]

START_BALANCE = 1000.0
TAKER_FEE = 0.00055
SLIPPAGE  = 0.0005
MAX_LEVERAGE = 50
RISK_PER_TRADE = 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════════
#  INDICATORS
# ═══════════════════════════════════════════════════════════════════════════

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
    if n == 0:
        return out
    k = 2 / (period + 1)
    out[0] = x[0]
    for i in range(1, n):
        out[i] = x[i] * k + out[i-1] * (1 - k)
    return out


def _macd(cl, fast=12, slow=26, sig=9):
    ef = _ema(cl, fast)
    es = _ema(cl, slow)
    macd = ef - es
    sig_line = _ema(macd, sig)
    hist = macd - sig_line
    return macd, sig_line, hist


def _adx(hi, lo, cl, period=14):
    """Wilder-smoothed ADX + DI. Returns (adx, +DI, -DI)."""
    n = len(cl)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = hi[i] - hi[i-1]
        dn = lo[i-1] - lo[i]
        plus_dm[i] = up if (up > dn and up > 0) else 0.0
        minus_dm[i] = dn if (dn > up and dn > 0) else 0.0
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    atr_w = np.zeros(n)
    plus_s = np.zeros(n)
    minus_s = np.zeros(n)
    if n <= period:
        return np.zeros(n), np.zeros(n), np.zeros(n)
    atr_w[period] = np.sum(tr[1:period+1])
    plus_s[period] = np.sum(plus_dm[1:period+1])
    minus_s[period] = np.sum(minus_dm[1:period+1])
    for i in range(period+1, n):
        atr_w[i] = atr_w[i-1] - atr_w[i-1]/period + tr[i]
        plus_s[i] = plus_s[i-1] - plus_s[i-1]/period + plus_dm[i]
        minus_s[i] = minus_s[i-1] - minus_s[i-1]/period + minus_dm[i]
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)
    for i in range(period, n):
        if atr_w[i] > 0:
            plus_di[i] = 100 * plus_s[i] / atr_w[i]
            minus_di[i] = 100 * minus_s[i] / atr_w[i]
    dx = np.zeros(n)
    for i in range(period, n):
        s = plus_di[i] + minus_di[i]
        if s > 0:
            dx[i] = 100 * abs(plus_di[i] - minus_di[i]) / s
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
    upper = med + mult * atr
    lower = med - mult * atr
    fub = np.copy(upper)
    flb = np.copy(lower)
    for i in range(1, n):
        if upper[i] < fub[i-1] or cl[i-1] > fub[i-1]:
            fub[i] = upper[i]
        else:
            fub[i] = fub[i-1]
        if lower[i] > flb[i-1] or cl[i-1] < flb[i-1]:
            flb[i] = lower[i]
        else:
            flb[i] = flb[i-1]
    st = np.zeros(n, dtype=np.int32)
    st[0] = 1
    for i in range(1, n):
        if st[i-1] == 1 and cl[i] < flb[i]:
            st[i] = -1
        elif st[i-1] == -1 and cl[i] > fub[i]:
            st[i] = 1
        else:
            st[i] = st[i-1]
    return st


# ═══════════════════════════════════════════════════════════════════════════
#  PREP & SESSION RANGES
# ═══════════════════════════════════════════════════════════════════════════

def prep_15m_arrays(df):
    o = df["open"].values.astype(float)
    h = df["high"].values.astype(float)
    l = df["low"].values.astype(float)
    c = df["close"].values.astype(float)
    v = df["volume"].values.astype(float)
    ts = df["timestamp"].values.astype(np.int64)
    dt = pd.to_datetime(ts, unit="ms", utc=True)
    hours = dt.hour.values.astype(np.int32)
    dates = np.array(dt.strftime("%Y-%m-%d"))
    dow = dt.dayofweek.values.astype(np.int32)
    atr = _atr(h, l, c, 14)
    volma = pd.Series(v).rolling(20, min_periods=1).mean().values
    return {
        "o": o, "h": h, "l": l, "c": c, "v": v, "ts": ts,
        "hour": hours, "date": dates, "dow": dow,
        "atr": atr, "volma": volma, "n": len(c),
    }


def build_session_ranges(arr):
    """Per-date dict with Asian (00-07), London (07-13), NY (13-22) hi/lo, plus PDH/PDL."""
    n = arr["n"]
    h, l = arr["h"], arr["l"]
    date = arr["date"]
    hour = arr["hour"]
    by_date = defaultdict(list)
    for i in range(n):
        by_date[date[i]].append(i)
    ranges = {}
    dates_sorted = sorted(by_date.keys())
    prev_hi = None
    prev_lo = None
    for d in dates_sorted:
        idxs = by_date[d]
        ai = [i for i in idxs if 0 <= hour[i] < 7]
        li = [i for i in idxs if 7 <= hour[i] < 13]
        ni = [i for i in idxs if 13 <= hour[i] < 22]
        ahi = max((h[i] for i in ai), default=np.nan)
        alo = min((l[i] for i in ai), default=np.nan)
        lhi = max((h[i] for i in li), default=np.nan)
        llo = min((l[i] for i in li), default=np.nan)
        nhi = max((h[i] for i in ni), default=np.nan)
        nlo = min((l[i] for i in ni), default=np.nan)
        ranges[d] = {
            "asian_hi": ahi, "asian_lo": alo,
            "london_hi": lhi, "london_lo": llo,
            "ny_hi": nhi, "ny_lo": nlo,
            "pd_hi": prev_hi, "pd_lo": prev_lo,
        }
        day_hi = max((h[i] for i in idxs), default=prev_hi)
        day_lo = min((l[i] for i in idxs), default=prev_lo)
        prev_hi = day_hi
        prev_lo = day_lo
    return ranges


def compute_htf_trends(df_4h, df_1h):
    """Returns dict of HTF trend maps: {ts_ms_int: +1/-1/0}."""
    out = {}
    if df_4h is not None and len(df_4h) > 60:
        cl = df_4h["close"].values.astype(float)
        hi = df_4h["high"].values.astype(float)
        lo = df_4h["low"].values.astype(float)
        ts = df_4h["timestamp"].values.astype(np.int64)
        ema50 = _ema(cl, 50)
        st = _supertrend(hi, lo, cl, 10, 3.0)
        out["ema50_4h"] = {int(ts[i]): (1 if cl[i] > ema50[i] else -1)
                          for i in range(50, len(cl))}
        out["st_4h"] = {int(ts[i]): int(st[i]) for i in range(len(cl))}
    if df_1h is not None and len(df_1h) > 60:
        cl = df_1h["close"].values.astype(float)
        hi = df_1h["high"].values.astype(float)
        lo = df_1h["low"].values.astype(float)
        ts = df_1h["timestamp"].values.astype(np.int64)
        ema50 = _ema(cl, 50)
        st = _supertrend(hi, lo, cl, 10, 3.0)
        _, _, hist = _macd(cl)
        adx, pdi, mdi = _adx(hi, lo, cl, 14)
        out["ema50_1h"] = {int(ts[i]): (1 if cl[i] > ema50[i] else -1)
                          for i in range(50, len(cl))}
        out["st_1h"] = {int(ts[i]): int(st[i]) for i in range(len(cl))}
        out["macd_hist_1h"] = {int(ts[i]): (1 if hist[i] > 0 else -1)
                               for i in range(35, len(cl))}
        def adir(i):
            if adx[i] < 20: return 0
            return 1 if pdi[i] > mdi[i] else -1
        out["adx_1h"] = {int(ts[i]): adir(i) for i in range(30, len(cl))}
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
#  SESSION STATISTICS
# ═══════════════════════════════════════════════════════════════════════════

def session_stats(arr):
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    n = arr["n"]
    by_date = defaultdict(list)
    for i in range(n):
        by_date[date[i]].append(i)
    stats = {"asian": [], "london": [], "ny": []}
    for d, idxs in by_date.items():
        for name, lo_h, hi_h in [("asian", 0, 7), ("london", 7, 13), ("ny", 13, 22)]:
            si = [i for i in idxs if lo_h <= hour[i] < hi_h]
            if len(si) < 4:
                continue
            s_open = o[si[0]]
            s_high = max(h[i] for i in si)
            s_low  = min(l[i] for i in si)
            s_close = c[si[-1]]
            if s_open <= 0:
                continue
            stats[name].append({
                "range_pct": (s_high - s_low) / s_open * 100,
                "up_pct": (s_high - s_open) / s_open * 100,
                "dn_pct": (s_open - s_low) / s_open * 100,
                "net_pct": (s_close - s_open) / s_open * 100,
            })
    summary = {}
    for name, recs in stats.items():
        if not recs:
            summary[name] = None
            continue
        rng = np.array([r["range_pct"] for r in recs])
        up = np.array([r["up_pct"] for r in recs])
        dn = np.array([r["dn_pct"] for r in recs])
        net = np.array([r["net_pct"] for r in recs])
        summary[name] = {
            "n": len(recs),
            "range_mean": float(np.mean(rng)),
            "range_med": float(np.median(rng)),
            "range_p75": float(np.percentile(rng, 75)),
            "range_p90": float(np.percentile(rng, 90)),
            "up_mean": float(np.mean(up)),
            "up_p75": float(np.percentile(up, 75)),
            "dn_mean": float(np.mean(dn)),
            "dn_p75": float(np.percentile(dn, 75)),
            "net_mean": float(np.mean(net)),
            "net_std": float(np.std(net)),
            "bull_pct": float(np.mean(net > 0) * 100),
        }
    return summary


# ═══════════════════════════════════════════════════════════════════════════
#  SIGNAL GEN — ARB V2
# ═══════════════════════════════════════════════════════════════════════════

def gen_arb_v2(arr, sess_ranges, params, htf_aligned_map=None):
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    atr, volma, v = arr["atr"], arr["volma"], arr["v"]
    n = arr["n"]

    open_hr = params.get("open_hr", 13)
    close_hr = params.get("close_hr", 22)
    sl_mode = params.get("sl_mode", "structural")
    sl_atr_mult = params.get("sl_atr_mult", 2.5)
    sl_pct = params.get("sl_pct", 0.02)
    sl_swing_bars = params.get("sl_swing_bars", 16)
    sl_buffer_atr = params.get("sl_buffer_atr", 0.3)
    tp_mode = params.get("tp_mode", "fixed_r")
    tp_r = params.get("tp_r", 2.0)
    tp_range_mult = params.get("tp_range_mult", 1.0)
    htf_filter = params.get("htf_filter", None)
    max_entry_hr = params.get("max_entry_hr", None)
    vol_mult = params.get("vol_mult", 0.0)
    min_range_atr = params.get("min_range_atr", 0.3)
    max_range_atr = params.get("max_range_atr", 5.0)

    signals = []
    last_sig_date = None

    for i in range(50, n):
        d = date[i]
        if d not in sess_ranges:
            continue
        r = sess_ranges[d]
        ahi, alo = r["asian_hi"], r["asian_lo"]
        if np.isnan(ahi) or np.isnan(alo):
            continue
        if hour[i] < open_hr or hour[i] >= close_hr:
            continue
        if max_entry_hr is not None and hour[i] >= open_hr + max_entry_hr:
            continue
        if last_sig_date == d:
            continue
        a_range = ahi - alo
        if a_range <= 0 or atr[i] <= 0:
            continue
        a_range_atr = a_range / atr[i]
        if a_range_atr < min_range_atr or a_range_atr > max_range_atr:
            continue

        direction = None
        entry = None
        if h[i] > ahi and c[i] > ahi:
            direction = "LONG"
            entry = ahi
        elif l[i] < alo and c[i] < alo:
            direction = "SHORT"
            entry = alo
        if direction is None:
            continue

        if htf_filter is not None and htf_aligned_map is not None:
            f = htf_aligned_map.get(htf_filter)
            if f is not None:
                flag = int(f[i])
                if direction == "LONG" and flag != 1: continue
                if direction == "SHORT" and flag != -1: continue

        if vol_mult > 0:
            ma = volma[i]
            if ma > 0 and v[i] < ma * vol_mult:
                continue

        # Stop loss
        if sl_mode == "atr":
            sl = entry - atr[i] * sl_atr_mult if direction == "LONG" else entry + atr[i] * sl_atr_mult
        elif sl_mode == "pct":
            sl = entry * (1 - sl_pct) if direction == "LONG" else entry * (1 + sl_pct)
        elif sl_mode == "structural":
            k = min(sl_swing_bars, i)
            if direction == "LONG":
                swing_lo = float(np.min(l[i-k:i+1]))
                sl = swing_lo - atr[i] * sl_buffer_atr
            else:
                swing_hi = float(np.max(h[i-k:i+1]))
                sl = swing_hi + atr[i] * sl_buffer_atr
        else:
            sl = entry - atr[i] * 2.0 if direction == "LONG" else entry + atr[i] * 2.0

        risk_px = abs(entry - sl)
        if risk_px <= 0:
            continue

        # Take profit
        if tp_mode == "fixed_r":
            tp = entry + risk_px * tp_r if direction == "LONG" else entry - risk_px * tp_r
        elif tp_mode == "range_mult":
            tp = entry + a_range * tp_range_mult if direction == "LONG" else entry - a_range * tp_range_mult
        elif tp_mode == "nearest_level":
            candidates = []
            if direction == "LONG":
                if r["pd_hi"] is not None and not np.isnan(r["pd_hi"]) and r["pd_hi"] > entry + risk_px:
                    candidates.append(r["pd_hi"])
                if not np.isnan(r["london_hi"]) and hour[i] >= 13 and r["london_hi"] > entry + risk_px:
                    candidates.append(r["london_hi"])
                proj = ahi + a_range  # 100% extension of Asian range
                if proj > entry + risk_px:
                    candidates.append(proj)
                tp = min(candidates) if candidates else entry + risk_px * 2.0
            else:
                if r["pd_lo"] is not None and not np.isnan(r["pd_lo"]) and r["pd_lo"] < entry - risk_px:
                    candidates.append(r["pd_lo"])
                if not np.isnan(r["london_lo"]) and hour[i] >= 13 and r["london_lo"] < entry - risk_px:
                    candidates.append(r["london_lo"])
                proj = alo - a_range
                if proj < entry - risk_px:
                    candidates.append(proj)
                tp = max(candidates) if candidates else entry - risk_px * 2.0
        else:
            tp = entry + risk_px * 2.0 if direction == "LONG" else entry - risk_px * 2.0

        signals.append({
            "i": i, "entry": float(entry), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(a_range),
        })
        last_sig_date = d

    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATE WITH DYNAMIC TRAIL
# ═══════════════════════════════════════════════════════════════════════════

def simulate(sigs_by_sym, arrays_by_sym, session_end_hr=22, trail_cfg=None):
    balance = START_BALANCE
    completed = []
    peak = balance
    max_dd = 0.0

    for sym, sigs in sigs_by_sym.items():
        arr = arrays_by_sym[sym]
        h, l, c = arr["h"], arr["l"], arr["c"]
        hour, date = arr["hour"], arr["date"]
        atr = arr["atr"]
        n = arr["n"]

        for sig in sigs:
            i = sig["i"]
            entry = sig["entry"]
            sl_init = sig["sl"]
            tp = sig["tp"]
            direction = sig["direction"]
            risk_px = sig["risk_px"]
            entry_date = date[i]

            risk_usd = balance * RISK_PER_TRADE
            if risk_px <= 0 or entry <= 0:
                continue
            qty_usd = risk_usd / (risk_px / entry)
            if qty_usd <= 0:
                continue
            # cap notional by leverage
            if qty_usd > balance * MAX_LEVERAGE:
                qty_usd = balance * MAX_LEVERAGE

            eff_entry = entry * (1 + SLIPPAGE) if direction == "LONG" else entry * (1 - SLIPPAGE)
            entry_fee = qty_usd * TAKER_FEE

            sl_cur = sl_init
            trail_active = False
            mfe = 0.0
            mae = 0.0
            exit_reason = "EOD"
            exit_px = c[n-1]
            exit_bar = n - 1

            for j in range(i, n):
                # exit on next day OR session close hour
                if date[j] != entry_date:
                    exit_reason = "EOS"
                    exit_px = c[j]
                    exit_bar = j
                    break
                if hour[j] >= session_end_hr:
                    exit_reason = "EOS"
                    exit_px = c[j]
                    exit_bar = j
                    break

                bar_h, bar_l = h[j], l[j]
                if direction == "LONG":
                    mfe = max(mfe, (bar_h - eff_entry) / risk_px)
                    mae = max(mae, (eff_entry - bar_l) / risk_px)
                else:
                    mfe = max(mfe, (eff_entry - bar_l) / risk_px)
                    mae = max(mae, (bar_h - eff_entry) / risk_px)

                # trail
                if trail_cfg is not None:
                    act_r = trail_cfg.get("activate_at_r", 1.0)
                    if not trail_active and mfe >= act_r:
                        trail_active = True
                    if trail_active:
                        mode = trail_cfg.get("trail_mode", "swing")
                        if mode == "breakeven":
                            if direction == "LONG":
                                sl_cur = max(sl_cur, eff_entry)
                            else:
                                sl_cur = min(sl_cur, eff_entry)
                        elif mode == "atr":
                            mlt = trail_cfg.get("atr_mult", 2.0)
                            if direction == "LONG":
                                new_sl = c[j] - atr[j] * mlt
                                sl_cur = max(sl_cur, new_sl)
                            else:
                                new_sl = c[j] + atr[j] * mlt
                                sl_cur = min(sl_cur, new_sl)
                        elif mode == "swing":
                            k = trail_cfg.get("swing_bars", 5)
                            buf = trail_cfg.get("buffer_atr", 0.2)
                            kk = min(k, j - i)
                            if kk >= 1:
                                if direction == "LONG":
                                    new_sl = float(np.min(l[j-kk:j+1])) - atr[j] * buf
                                    sl_cur = max(sl_cur, new_sl)
                                else:
                                    new_sl = float(np.max(h[j-kk:j+1])) + atr[j] * buf
                                    sl_cur = min(sl_cur, new_sl)

                # check SL/TP
                if direction == "LONG":
                    if bar_l <= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur
                        exit_bar = j
                        break
                    if bar_h >= tp:
                        exit_reason = "TP"
                        exit_px = tp
                        exit_bar = j
                        break
                else:
                    if bar_h >= sl_cur:
                        exit_reason = "TRAIL" if trail_active else "SL"
                        exit_px = sl_cur
                        exit_bar = j
                        break
                    if bar_l <= tp:
                        exit_reason = "TP"
                        exit_px = tp
                        exit_bar = j
                        break

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
            exit_r = (eff_exit - eff_entry) / risk_px if direction == "LONG" else (eff_entry - eff_exit) / risk_px
            completed.append({
                "sym": sym, "direction": direction,
                "entry_bar": i, "exit_bar": exit_bar,
                "bars": exit_bar - i,
                "reason": exit_reason,
                "pnl_usd": pnl_usd,
                "won": pnl_usd > 0,
                "exit_r": exit_r,
                "mfe": mfe, "mae": mae,
                "entry": entry, "exit": exit_px, "sl": sl_init, "tp": tp,
                "date": entry_date,
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
        "trades": len(completed),
        "wins": len(wins), "losses": len(losses),
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


# ═══════════════════════════════════════════════════════════════════════════
#  RUN
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy(name, arrays, ranges, htf_aligned_bysym, params, trail_cfg=None, session_end_hr=22):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_arb_v2(arr, ranges[sym], params, htf_aligned_bysym.get(sym))
    res = simulate(sigs, arrays, session_end_hr=session_end_hr, trail_cfg=trail_cfg)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    per_sym = {}
    for sym in sigs:
        per_sym[sym] = simulate({sym: sigs[sym]}, {sym: arrays[sym]},
                                session_end_hr=session_end_hr, trail_cfg=trail_cfg)
    res["per_sym"] = per_sym
    return res


def log_r(name, r):
    logger.info("%-28s trades=%3d WR=%5.1f%% PF=%5.2f PnL=%+7.1f%% DD=%5.2f%% avgR=%+.2f avgMFE=%+.2f",
                name, r["trades"], r["wr_pct"], r["profit_factor"],
                r["pnl_pct"], r["max_dd_pct"], r["avg_r"], r["avg_mfe"])


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    logger.info("=== Session V2 backtest ===")
    arrays = {}
    ranges = {}
    htf_aligned_bysym = {}

    for sym in SYMBOLS:
        df15 = _load(sym, "15m", "365d")
        df1h = _load(sym, "1h", "365d")
        df4h = _load(sym, "4h", "365d")
        if df15 is None:
            logger.warning("No 15m data for %s", sym)
            continue
        logger.info("%s: 15m=%d 1h=%s 4h=%s", sym.replace("/USDT:USDT", ""),
                     len(df15), len(df1h) if df1h is not None else "-",
                     len(df4h) if df4h is not None else "-")
        arr = prep_15m_arrays(df15)
        rng = build_session_ranges(arr)
        htfs = compute_htf_trends(df4h, df1h)
        aligned = {nm: align_htf(m, arr["ts"]) for nm, m in htfs.items()}
        arrays[sym] = arr
        ranges[sym] = rng
        htf_aligned_bysym[sym] = aligned

    # ═══ Phase A: Session stats ═══
    logger.info("--- Phase A: Session statistics ---")
    all_stats = {}
    for sym, arr in arrays.items():
        st = session_stats(arr)
        all_stats[sym] = st
        short = sym.replace("/USDT:USDT", "")
        logger.info("%s session ranges & excursions:", short)
        for name, s in st.items():
            if s is None: continue
            logger.info("  %-7s n=%3d | range mean=%.2f%% med=%.2f%% p75=%.2f%% p90=%.2f%% | up_p75=%.2f%% dn_p75=%.2f%% | net=%+.2f%%+-%.2f%% bull=%.0f%%",
                         name, s["n"], s["range_mean"], s["range_med"], s["range_p75"], s["range_p90"],
                         s["up_p75"], s["dn_p75"], s["net_mean"], s["net_std"], s["bull_pct"])

    results = []

    # ═══ Phase B: SL mode comparison ═══
    logger.info("--- Phase B: SL mode comparison ---")
    base_params = {
        "open_hr": 13, "close_hr": 22,
        "min_range_atr": 0.3, "max_range_atr": 5.0,
        "tp_mode": "fixed_r", "tp_r": 2.0,
        "max_entry_hr": 3,
    }
    sl_variants = [
        ("SL_atr1.5",   {"sl_mode": "atr", "sl_atr_mult": 1.5}),
        ("SL_atr2.5",   {"sl_mode": "atr", "sl_atr_mult": 2.5}),
        ("SL_atr3.5",   {"sl_mode": "atr", "sl_atr_mult": 3.5}),
        ("SL_pct1.0",   {"sl_mode": "pct", "sl_pct": 0.010}),
        ("SL_pct1.5",   {"sl_mode": "pct", "sl_pct": 0.015}),
        ("SL_pct2.0",   {"sl_mode": "pct", "sl_pct": 0.020}),
        ("SL_pct3.0",   {"sl_mode": "pct", "sl_pct": 0.030}),
        ("SL_struct8",  {"sl_mode": "structural", "sl_swing_bars": 8,  "sl_buffer_atr": 0.3}),
        ("SL_struct16", {"sl_mode": "structural", "sl_swing_bars": 16, "sl_buffer_atr": 0.3}),
        ("SL_struct32", {"sl_mode": "structural", "sl_swing_bars": 32, "sl_buffer_atr": 0.3}),
    ]
    for name, extra in sl_variants:
        p = dict(base_params); p.update(extra)
        r = run_strategy(name, arrays, ranges, htf_aligned_bysym, p)
        r["config_id"] = name; r["params"] = p; results.append(r)
        log_r(name, r)

    # ═══ Phase C: HTF filter comparison (on ATR SL) ═══
    logger.info("--- Phase C: HTF filter comparison (ATR2.5 SL) ---")
    base_atr = dict(base_params); base_atr.update({"sl_mode": "atr", "sl_atr_mult": 2.5})
    htf_variants = [
        ("HTFa_none",     None),
        ("HTFa_ema50_4h", "ema50_4h"),
        ("HTFa_st_4h",    "st_4h"),
        ("HTFa_ema50_1h", "ema50_1h"),
        ("HTFa_st_1h",    "st_1h"),
        ("HTFa_macd_1h",  "macd_hist_1h"),
        ("HTFa_adx_1h",   "adx_1h"),
    ]
    for name, flt in htf_variants:
        p = dict(base_atr); p["htf_filter"] = flt
        r = run_strategy(name, arrays, ranges, htf_aligned_bysym, p)
        r["config_id"] = name; r["params"] = p; results.append(r)
        log_r(name, r)

    # ═══ Phase C2: HTF filter with structural SL ═══
    logger.info("--- Phase C2: HTF filter comparison (structural16 SL) ---")
    base_s = dict(base_params); base_s.update({"sl_mode": "structural", "sl_swing_bars": 16, "sl_buffer_atr": 0.3})
    for name, flt in htf_variants:
        nm = name.replace("HTFa_", "HTFs_")
        p = dict(base_s); p["htf_filter"] = flt
        r = run_strategy(nm, arrays, ranges, htf_aligned_bysym, p)
        r["config_id"] = nm; r["params"] = p; results.append(r)
        log_r(nm, r)

    # ═══ Phase D: Dynamic trail ═══
    logger.info("--- Phase D: Dynamic trail with wider SL + TP=3R ---")
    base_d = dict(base_params)
    base_d.update({"sl_mode": "structural", "sl_swing_bars": 16, "sl_buffer_atr": 0.3,
                   "htf_filter": "ema50_4h", "tp_mode": "fixed_r", "tp_r": 3.0})
    trails = [
        ("Trail_BE@1R",       {"activate_at_r": 1.0, "trail_mode": "breakeven"}),
        ("Trail_BE@1.5R",     {"activate_at_r": 1.5, "trail_mode": "breakeven"}),
        ("Trail_ATR2@1R",     {"activate_at_r": 1.0, "trail_mode": "atr", "atr_mult": 2.0}),
        ("Trail_ATR3@1R",     {"activate_at_r": 1.0, "trail_mode": "atr", "atr_mult": 3.0}),
        ("Trail_sw3@1R",      {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 3, "buffer_atr": 0.2}),
        ("Trail_sw5@1R",      {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("Trail_sw8@1R",      {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 8, "buffer_atr": 0.2}),
        ("Trail_sw5@1.5R",    {"activate_at_r": 1.5, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("Trail_sw8@1.5R",    {"activate_at_r": 1.5, "trail_mode": "swing", "swing_bars": 8, "buffer_atr": 0.2}),
    ]
    for name, trail_cfg in trails:
        r = run_strategy(name, arrays, ranges, htf_aligned_bysym, base_d, trail_cfg=trail_cfg)
        r["config_id"] = name; r["params"] = base_d; r["trail_cfg"] = trail_cfg
        results.append(r)
        log_r(name, r)

    # ═══ Phase E: Level-based TP + trail ═══
    logger.info("--- Phase E: Level-based TP & range multiple ---")
    base_e = dict(base_params); base_e.update({"sl_mode": "atr", "sl_atr_mult": 2.5, "htf_filter": "ema50_4h"})
    tp_variants = [
        ("LvlTP_noTrail",   {"tp_mode": "nearest_level"}, None),
        ("LvlTP_BE1R",      {"tp_mode": "nearest_level"}, {"activate_at_r": 1.0, "trail_mode": "breakeven"}),
        ("LvlTP_sw5@1R",    {"tp_mode": "nearest_level"}, {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("RngM1.5_noTrail", {"tp_mode": "range_mult", "tp_range_mult": 1.5}, None),
        ("RngM2.0_noTrail", {"tp_mode": "range_mult", "tp_range_mult": 2.0}, None),
        ("RngM2.0_sw5@1R",  {"tp_mode": "range_mult", "tp_range_mult": 2.0}, {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
    ]
    for name, extra, trail_cfg in tp_variants:
        p = dict(base_e); p.update(extra)
        r = run_strategy(name, arrays, ranges, htf_aligned_bysym, p, trail_cfg=trail_cfg)
        r["config_id"] = name; r["params"] = p; r["trail_cfg"] = trail_cfg
        results.append(r)
        log_r(name, r)

    # ═══ Phase F: Combine best elements ═══
    logger.info("--- Phase F: Combine best (HTF + TP + trail) ---")
    combos = [
        # ADX filter + different TPs
        ("Combo_ADX_RngM2_sw5",  {"htf_filter": "adx_1h", "tp_mode": "range_mult", "tp_range_mult": 2.0},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("Combo_ADX_RngM2_noTr", {"htf_filter": "adx_1h", "tp_mode": "range_mult", "tp_range_mult": 2.0}, None),
        ("Combo_ADX_LvlTP_sw5",  {"htf_filter": "adx_1h", "tp_mode": "nearest_level"},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("Combo_ADX_LvlTP_noTr", {"htf_filter": "adx_1h", "tp_mode": "nearest_level"}, None),
        ("Combo_ADX_R3_sw5",     {"htf_filter": "adx_1h", "tp_mode": "fixed_r", "tp_r": 3.0},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        # Supertrend filter
        ("Combo_ST_RngM2_sw5",   {"htf_filter": "st_1h", "tp_mode": "range_mult", "tp_range_mult": 2.0},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("Combo_ST_LvlTP_sw5",   {"htf_filter": "st_1h", "tp_mode": "nearest_level"},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        ("Combo_ST_R3_sw5",      {"htf_filter": "st_1h", "tp_mode": "fixed_r", "tp_r": 3.0},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
        # Supertrend + wider stop + runner
        ("Combo_ST_wSL3_R4_sw8", {"htf_filter": "st_1h", "sl_atr_mult": 3.0, "tp_mode": "fixed_r", "tp_r": 4.0},
                                  {"activate_at_r": 1.5, "trail_mode": "swing", "swing_bars": 8, "buffer_atr": 0.2}),
        # ADX + conservative (low DD)
        ("Combo_ADX_Pct2_LvlTP", {"htf_filter": "adx_1h", "sl_mode": "pct", "sl_pct": 0.02, "tp_mode": "nearest_level"}, None),
        # Volume confirmation
        ("Combo_ADX_vol1.2_R3",  {"htf_filter": "adx_1h", "vol_mult": 1.2, "tp_mode": "fixed_r", "tp_r": 3.0},
                                  {"activate_at_r": 1.0, "trail_mode": "swing", "swing_bars": 5, "buffer_atr": 0.2}),
    ]
    base_f = dict(base_params); base_f.update({"sl_mode": "atr", "sl_atr_mult": 2.5})
    for name, extra, trail_cfg in combos:
        p = dict(base_f); p.update(extra)
        r = run_strategy(name, arrays, ranges, htf_aligned_bysym, p, trail_cfg=trail_cfg)
        r["config_id"] = name; r["params"] = p; r["trail_cfg"] = trail_cfg
        results.append(r)
        log_r(name, r)

    # ═══ Phase G: Walk-forward on top-3 ═══
    logger.info("--- Phase G: Walk-forward (split 365d -> 4 quarters) ---")
    # Pick best 3 by PF so far
    top3 = sorted([r for r in results if r.get("trades", 0) >= 50],
                  key=lambda x: x["profit_factor"], reverse=True)[:3]
    logger.info("Walk-forward candidates: %s", [r["config_id"] for r in top3])

    # Split each symbol's data into 4 quarters
    wf_report = {}
    for best in top3:
        cfg_id = best["config_id"]
        params_cp = dict(best.get("params", {}))
        trail_cp = best.get("trail_cfg", None)
        quarters_res = []
        for q_idx in range(4):
            q_arrays = {}
            q_ranges = {}
            q_htf = {}
            for sym, arr in arrays.items():
                n = arr["n"]
                seg = n // 4
                lo = q_idx * seg
                hi = (q_idx + 1) * seg if q_idx < 3 else n
                sub = {k: (v[lo:hi] if isinstance(v, np.ndarray) else v) for k, v in arr.items()}
                sub["n"] = hi - lo
                q_arrays[sym] = sub
                # Filter ranges by dates in this slice
                slice_dates = set(sub["date"])
                q_ranges[sym] = {d: r for d, r in ranges[sym].items() if d in slice_dates}
                q_htf[sym] = {nm: v[lo:hi] for nm, v in htf_aligned_bysym[sym].items()}
            r = run_strategy(f"{cfg_id}_Q{q_idx+1}", q_arrays, q_ranges, q_htf,
                              params_cp, trail_cfg=trail_cp)
            quarters_res.append({
                "q": q_idx+1, "trades": r["trades"], "WR": r["wr_pct"],
                "PF": r["profit_factor"], "PnL": r["pnl_pct"], "DD": r["max_dd_pct"],
            })
            logger.info("  %s Q%d trades=%3d PF=%.2f PnL=%+7.1f%% WR=%.1f%% DD=%.1f%%",
                         cfg_id, q_idx+1, r["trades"], r["profit_factor"],
                         r["pnl_pct"], r["wr_pct"], r["max_dd_pct"])
        wf_report[cfg_id] = quarters_res

    # ═══ Save & report ═══
    out_path = RESULTS_DIR / f"backtest_session_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(out_path, "w") as f:
            json.dump({"session_stats": all_stats, "results": results,
                        "walk_forward": wf_report}, f, indent=2, default=str)
        logger.info("Saved %s (%.1fs)", out_path.name, _time.time()-t0)
    except Exception as e:
        logger.warning("Save failed: %s", e)

    print("\n=== Session stats (compact) ===")
    print(f"{'sym':4s} {'sess':7s} {'rng med%':>8s} {'rng p75%':>8s} {'up p75%':>8s} {'dn p75%':>8s} {'net%':>7s} {'bull%':>6s}")
    for sym, st in all_stats.items():
        s = sym.replace("/USDT:USDT", "")
        for name, v in st.items():
            if v is None: continue
            print(f"{s:4s} {name:7s} {v['range_med']:>8.2f} {v['range_p75']:>8.2f} {v['up_p75']:>8.2f} {v['dn_p75']:>8.2f} {v['net_mean']:>+7.2f} {v['bull_pct']:>6.1f}")

    print("\n=== Top 20 by PF (min 20 trades) ===")
    print(f"{'name':28s} {'trades':>6} {'WR%':>5} {'PF':>5} {'PnL%':>7} {'DD%':>6} {'avgR':>6}")
    filtered = [r for r in results if r.get("trades", 0) >= 20]
    for r in sorted(filtered, key=lambda x: x.get("profit_factor", 0), reverse=True)[:20]:
        print(f"{r['config_id'][:28]:28s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f} {r['pnl_pct']:>+7.1f} {r['max_dd_pct']:>6.2f} {r['avg_r']:>+6.2f}")

    print("\n=== Top 10 by PnL (min 20 trades) ===")
    print(f"{'name':28s} {'trades':>6} {'WR%':>5} {'PF':>5} {'PnL%':>7} {'DD%':>6} {'avgR':>6}")
    for r in sorted(filtered, key=lambda x: x.get("pnl_pct", 0), reverse=True)[:10]:
        print(f"{r['config_id'][:28]:28s} {r['trades']:>6} {r['wr_pct']:>5.1f} {r['profit_factor']:>5.2f} {r['pnl_pct']:>+7.1f} {r['max_dd_pct']:>6.2f} {r['avg_r']:>+6.2f}")

    print("\n=== Per-symbol of top-8 by PnL ===")
    print(f"{'name':28s} {'sym':4s} {'trades':>6} {'WR%':>5} {'PF':>5} {'PnL%':>7} {'avgR':>6}")
    for r in sorted(filtered, key=lambda x: x.get("pnl_pct", 0), reverse=True)[:8]:
        for sym, sr in r.get("per_sym", {}).items():
            s = sym.replace("/USDT:USDT", "")
            print(f"{r['config_id'][:28]:28s} {s:4s} {sr['trades']:>6} {sr['wr_pct']:>5.1f} {sr['profit_factor']:>5.2f} {sr['pnl_pct']:>+7.1f} {sr['avg_r']:>+6.2f}")


if __name__ == "__main__":
    main()
