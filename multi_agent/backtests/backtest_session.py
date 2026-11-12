"""
backtest_session.py  --  Session-based trading strategies (2026-04-16)

Goal: develop a new strategy that trades session-driven moves (Tokyo / London / NY).
Hypotheses:
  H1  ARB-LO    Asian Range Breakout at London open  (07:00 UTC)
  H2  ARB-NY    Asian Range Breakout at NY open      (13:00 UTC)
  H3  ORB-NY    Opening Range Breakout — first 30/60m of NY session
  H4  PDH/PDL   Previous-day high/low breakout at NY open
  H5  Sweep     Session sweeps prior-session extreme then reverses

Symbols: BTC, ETH.  Data: 365d 15m.

Session times (UTC):
  Tokyo       00:00 - 09:00
  London      07:00 - 16:00
  NY          13:00 - 22:00

Produces a comparative report: PF / WR / PnL% / DD% / trades per hypothesis.
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

logger = logging.getLogger("session_bt")
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
RISK_PER_TRADE = 0.01   # 1% risk per trade (standard for optimization)

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
    out = np.full(n, np.nan)
    if n <= period:
        return out
    out[period-1] = np.mean(tr[:period])
    for i in range(period, n):
        out[i] = (out[i-1] * (period-1) + tr[i]) / period
    return out


def _ema(arr, period):
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) < period:
        return out
    k = 2.0 / (period + 1)
    out[period-1] = np.mean(arr[:period])
    for i in range(period, len(arr)):
        out[i] = arr[i] * k + out[i-1] * (1 - k)
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def ms_to_utc_hm(ts_ms: int) -> tuple[int, int, str]:
    """Returns (hour, minute, date_str 'YYYY-MM-DD') for a ms timestamp in UTC."""
    dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
    return dt.hour, dt.minute, dt.strftime("%Y-%m-%d")


def prep_15m_arrays(df):
    """Prepare numpy arrays for fast iteration + session tags."""
    ts = df["timestamp"].to_numpy(dtype=np.int64)
    op = df["open"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    cl = df["close"].to_numpy(dtype=float)
    vo = df["volume"].to_numpy(dtype=float)
    n = len(ts)

    # Derive UTC hour/date arrays
    hours = np.empty(n, dtype=np.int8)
    mins = np.empty(n, dtype=np.int8)
    dates = np.empty(n, dtype="U10")
    for i in range(n):
        dt = datetime.fromtimestamp(ts[i] / 1000, tz=timezone.utc)
        hours[i] = dt.hour
        mins[i] = dt.minute
        dates[i] = dt.strftime("%Y-%m-%d")

    atr14 = _atr(hi, lo, cl, 14)
    # Daily trend via EMA200 on 1h equivalent — we'll compute EMA200 directly on 15m close × 4
    # (actually use 1d EMA200 loaded separately; skip here, apply as optional filter)
    return {"ts": ts, "op": op, "hi": hi, "lo": lo, "cl": cl, "vo": vo,
            "hours": hours, "mins": mins, "dates": dates, "n": n, "atr": atr14}


def compute_daily_trend(df_daily):
    """Daily EMA200 trend filter. Returns dict with ts and trend array (1=up, -1=down, 0=na)."""
    if df_daily is None or len(df_daily) < 210:
        return None
    ts = df_daily["timestamp"].to_numpy(dtype=np.int64)
    cl = df_daily["close"].to_numpy(dtype=float)
    ema200 = _ema(cl, 200)
    return {"ts": ts, "cl": cl, "ema200": ema200, "n": len(ts)}


def daily_trend_at(dtrend, bar_ts) -> int:
    """Returns 1 (up), -1 (down), or 0 (na) for daily trend at given timestamp."""
    if dtrend is None:
        return 0
    idx = int(np.searchsorted(dtrend["ts"], bar_ts, side="right")) - 1
    if idx < 200 or np.isnan(dtrend["ema200"][idx]):
        return 0
    return 1 if dtrend["cl"][idx] > dtrend["ema200"][idx] else -1


# ═══════════════════════════════════════════════════════════════════════════
#  SESSION RANGES (pre-compute per-day)
# ═══════════════════════════════════════════════════════════════════════════

def build_session_ranges(arr):
    """Per date, compute:
      asian_hi, asian_lo     (00:00-07:00 UTC)
      london_hi, london_lo   (07:00-13:00 UTC)
      ny_hi, ny_lo           (13:00-22:00 UTC)
      pd_hi, pd_lo           (previous day 00:00-24:00 UTC)
    Returns dict date -> {fields}.
    """
    n = arr["n"]
    result = {}
    # First pass: accumulate per-date min/max for each session
    daily = defaultdict(lambda: {"asian_hi": -np.inf, "asian_lo": np.inf,
                                  "london_hi": -np.inf, "london_lo": np.inf,
                                  "ny_hi": -np.inf, "ny_lo": np.inf,
                                  "day_hi": -np.inf, "day_lo": np.inf})
    for i in range(n):
        h = arr["hours"][i]; d = arr["dates"][i]
        hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
        rec = daily[d]
        rec["day_hi"] = max(rec["day_hi"], hi_i)
        rec["day_lo"] = min(rec["day_lo"], lo_i)
        if 0 <= h < 7:
            rec["asian_hi"] = max(rec["asian_hi"], hi_i)
            rec["asian_lo"] = min(rec["asian_lo"], lo_i)
        if 7 <= h < 13:
            rec["london_hi"] = max(rec["london_hi"], hi_i)
            rec["london_lo"] = min(rec["london_lo"], lo_i)
        if 13 <= h < 22:
            rec["ny_hi"] = max(rec["ny_hi"], hi_i)
            rec["ny_lo"] = min(rec["ny_lo"], lo_i)

    # Second pass: attach previous-day range
    sorted_dates = sorted(daily.keys())
    for idx, d in enumerate(sorted_dates):
        rec = daily[d]
        if idx > 0:
            prev = daily[sorted_dates[idx - 1]]
            rec["pd_hi"] = prev["day_hi"]
            rec["pd_lo"] = prev["day_lo"]
        else:
            rec["pd_hi"] = np.nan
            rec["pd_lo"] = np.nan
        result[d] = rec
    return result


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 1 — Asian Range Breakout (ARB)
# ═══════════════════════════════════════════════════════════════════════════

def gen_arb_signals(arr, sess_ranges, params):
    """Asian Range Breakout (strict).
    Entry: actual breakout candle — previous bar inside Asian range, current closes outside.
    Only triggered during target session.  One trade per day.
    """
    open_hr       = params["open_hr"]          # 7 (London) or 13 (NY)
    close_hr      = params["close_hr"]
    min_range_atr = params.get("min_range_atr", 0.5)
    max_range_atr = params.get("max_range_atr", 4.0)
    sl_mode       = params.get("sl_mode", "opposite")
    sl_atr_mult   = params.get("sl_atr_mult", 1.0)
    tp_mode       = params.get("tp_mode", "range_mult")
    tp_range_mult = params.get("tp_range_mult", 1.0)
    tp_r          = params.get("tp_r", 2.0)
    dtrend_filter = params.get("dtrend_filter", None)
    dtrend_arr    = params.get("_dtrend")
    strict_break  = params.get("strict_break", True)
    require_inside_at_open = params.get("require_inside_at_open", True)
    max_entry_hr  = params.get("max_entry_hr", None)  # restrict entries to first N hours
    vol_mult      = params.get("vol_mult", 0.0)       # breakout volume vs MA

    n = arr["n"]
    hi = arr["hi"]; lo = arr["lo"]; cl = arr["cl"]; op = arr["op"]; vo = arr["vo"]
    atr = arr["atr"]; hours = arr["hours"]; dates = arr["dates"]; mins = arr["mins"]; ts = arr["ts"]

    # Volume MA(20) for vol filter
    vol_ma = np.zeros(n)
    if vol_mult > 0:
        for i in range(20, n):
            vol_ma[i] = np.mean(vo[i-20:i])

    sigs = {}
    taken_today = defaultdict(lambda: False)
    intact_at_open = defaultdict(lambda: None)

    for i in range(200, n):
        if np.isnan(atr[i]):
            continue
        h = hours[i]; d = dates[i]; m = mins[i]
        if d not in sess_ranges: continue
        sr = sess_ranges[d]
        ah = sr["asian_hi"]; al = sr["asian_lo"]
        if ah == -np.inf or al == np.inf: continue
        rng = ah - al
        if rng <= 0: continue

        if h == open_hr and m == 0 and intact_at_open[d] is None:
            intact_at_open[d] = (cl[i] <= ah and cl[i] >= al)

        if h < open_hr or h >= close_hr: continue
        if max_entry_hr is not None and h >= open_hr + max_entry_hr: continue
        if taken_today[d]: continue
        if rng / atr[i] < min_range_atr or rng / atr[i] > max_range_atr: continue
        if require_inside_at_open and intact_at_open.get(d) is not True:
            continue
        if vol_mult > 0 and vol_ma[i] > 0 and vo[i] / vol_ma[i] < vol_mult:
            continue

        dtr = 0
        if dtrend_filter is not None and dtrend_arr is not None:
            dtr = daily_trend_at(dtrend_arr, ts[i])

        close_i = cl[i]; prev_close = cl[i-1] if i > 0 else close_i

        # Strict breakout: prev was inside, current closes outside
        long_break  = close_i > ah and (not strict_break or prev_close <= ah)
        short_break = close_i < al and (not strict_break or prev_close >= al)

        if long_break:
            if dtrend_filter == "with" and dtr != 1: continue
            if dtrend_filter == "against" and dtr != -1: continue
            entry = close_i
            if sl_mode == "opposite":  sl = al
            elif sl_mode == "half":    sl = ah - rng * 0.5
            elif sl_mode == "atr":     sl = entry - atr[i] * sl_atr_mult
            else:                       sl = al
            if sl >= entry: continue
            risk = entry - sl
            tp = entry + (rng * tp_range_mult if tp_mode == "range_mult" else risk * tp_r)
            sigs[i] = {"direction": "LONG", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp), "risk": float(risk),
                       "date": d, "hr": int(h)}
            taken_today[d] = True

        elif short_break:
            if dtrend_filter == "with" and dtr != -1: continue
            if dtrend_filter == "against" and dtr != 1: continue
            entry = close_i
            if sl_mode == "opposite":  sl = ah
            elif sl_mode == "half":    sl = al + rng * 0.5
            elif sl_mode == "atr":     sl = entry + atr[i] * sl_atr_mult
            else:                       sl = ah
            if sl <= entry: continue
            risk = sl - entry
            tp = entry - (rng * tp_range_mult if tp_mode == "range_mult" else risk * tp_r)
            sigs[i] = {"direction": "SHORT", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp), "risk": float(risk),
                       "date": d, "hr": int(h)}
            taken_today[d] = True
    return sigs


def gen_mr_signals(arr, sess_ranges, params):
    """Session Mean Reversion (boundary bounce) — matches user's described idea.
    At NY open, check price position within reference range (Asian or Asian+London).
    If near bottom → LONG target top; if near top → SHORT target bottom.
    Uses daily trend filter: LONG bounces in up-trend, SHORT bounces in down-trend.
    """
    open_hr     = params.get("open_hr", 13)
    close_hr    = params.get("close_hr", 22)
    ref_source  = params.get("ref_source", "asian")  # "asian" | "combined"
    zone_pct    = params.get("zone_pct", 0.25)       # how close to boundary
    sl_buf_atr  = params.get("sl_buf_atr", 0.3)
    tp_mode     = params.get("tp_mode", "opposite")  # "opposite" | "fixed_r" | "mid"
    tp_r        = params.get("tp_r", 1.5)
    min_range_atr = params.get("min_range_atr", 0.8)
    max_range_atr = params.get("max_range_atr", 5.0)
    dtrend_filter = params.get("dtrend_filter", False)
    dtrend_arr    = params.get("_dtrend")
    entry_window_h = params.get("entry_window_h", 3)

    n = arr["n"]
    hi = arr["hi"]; lo = arr["lo"]; cl = arr["cl"]
    atr = arr["atr"]; hours = arr["hours"]; dates = arr["dates"]; mins = arr["mins"]; ts = arr["ts"]
    sigs = {}
    taken_today = defaultdict(lambda: False)
    session_open_idx = {}  # date -> first session bar index

    for i in range(200, n):
        if np.isnan(atr[i]): continue
        h = hours[i]; d = dates[i]; m = mins[i]
        if d not in sess_ranges: continue
        sr = sess_ranges[d]
        if ref_source == "asian":
            rh = sr["asian_hi"]; rl = sr["asian_lo"]
        else:
            rh = max(sr["asian_hi"], sr["london_hi"] if sr["london_hi"] != -np.inf else sr["asian_hi"])
            rl = min(sr["asian_lo"], sr["london_lo"] if sr["london_lo"] != np.inf else sr["asian_lo"])
        if rh == -np.inf or rl == np.inf or rh <= rl: continue
        rng = rh - rl
        if rng / atr[i] < min_range_atr or rng / atr[i] > max_range_atr: continue

        if h == open_hr and m == 0 and d not in session_open_idx:
            session_open_idx[d] = i
        if d not in session_open_idx: continue
        if h >= close_hr: continue
        if taken_today[d]: continue

        bars_since_open = i - session_open_idx[d]
        if bars_since_open > entry_window_h * 4: continue

        close_i = cl[i]
        pos_in_range = (close_i - rl) / rng  # 0 = at bottom, 1 = at top

        dtr = 0
        if dtrend_filter and dtrend_arr is not None:
            dtr = daily_trend_at(dtrend_arr, ts[i])

        # LONG bounce — price at bottom, target top
        if pos_in_range <= zone_pct and close_i >= rl:
            if dtrend_filter and dtr != 1:
                continue
            entry = close_i
            sl = rl - atr[i] * sl_buf_atr
            if sl >= entry: continue
            risk = entry - sl
            if tp_mode == "opposite":  tp = rh
            elif tp_mode == "mid":     tp = rl + rng * 0.5
            else:                       tp = entry + risk * tp_r
            if tp <= entry: continue
            sigs[i] = {"direction": "LONG", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp), "risk": float(risk),
                       "date": d, "hr": int(h)}
            taken_today[d] = True

        # SHORT bounce — price at top, target bottom
        elif pos_in_range >= 1 - zone_pct and close_i <= rh:
            if dtrend_filter and dtr != -1:
                continue
            entry = close_i
            sl = rh + atr[i] * sl_buf_atr
            if sl <= entry: continue
            risk = sl - entry
            if tp_mode == "opposite":  tp = rl
            elif tp_mode == "mid":     tp = rh - rng * 0.5
            else:                       tp = entry - risk * tp_r
            if tp >= entry: continue
            sigs[i] = {"direction": "SHORT", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp), "risk": float(risk),
                       "date": d, "hr": int(h)}
            taken_today[d] = True
    return sigs


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 2 — Opening Range Breakout (ORB)
# ═══════════════════════════════════════════════════════════════════════════

def gen_orb_signals(arr, params):
    """Opening Range Breakout for a session.
    Define first N bars of session as the opening range, then take breakout.
    """
    open_hr   = params["open_hr"]                # 13 for NY, 7 for London
    or_bars   = params.get("or_bars", 2)         # 2x 15m = 30min opening range
    window_h  = params.get("window_h", 4)        # try entries within first 4h of session
    min_width_atr = params.get("min_width_atr", 0.3)
    sl_mode   = params.get("sl_mode", "opposite")
    sl_atr_mult = params.get("sl_atr_mult", 1.0)
    tp_mode   = params.get("tp_mode", "range_mult")
    tp_range_mult = params.get("tp_range_mult", 1.5)
    tp_r      = params.get("tp_r", 2.0)

    n = arr["n"]
    hi = arr["hi"]; lo = arr["lo"]; cl = arr["cl"]
    atr = arr["atr"]; hours = arr["hours"]; dates = arr["dates"]; mins = arr["mins"]

    sigs = {}

    # Find session start bars per date (first bar where hour == open_hr and min == 0)
    or_state = {}  # date -> {"start_i": int, "or_hi": float, "or_lo": float, "done": bool, "taken": bool}
    for i in range(n):
        if np.isnan(atr[i]):
            continue
        h = hours[i]; d = dates[i]; m = mins[i]
        if d not in or_state and h == open_hr and m == 0:
            or_state[d] = {"start_i": i, "or_hi": hi[i], "or_lo": lo[i],
                            "done": False, "taken": False}
            continue
        st = or_state.get(d)
        if st is None:
            continue
        elapsed = i - st["start_i"]
        # Building OR
        if elapsed < or_bars:
            st["or_hi"] = max(st["or_hi"], hi[i])
            st["or_lo"] = min(st["or_lo"], lo[i])
            continue
        if not st["done"]:
            st["done"] = True
        if st["taken"]:
            continue
        if elapsed > (window_h * 4):  # 4 bars/hour @ 15m
            continue
        rng = st["or_hi"] - st["or_lo"]
        if rng <= 0 or rng / atr[i] < min_width_atr:
            continue

        # Breakout
        if cl[i] > st["or_hi"]:
            entry = cl[i]
            if sl_mode == "opposite":   sl = st["or_lo"]
            elif sl_mode == "mid":       sl = st["or_lo"] + rng * 0.5
            elif sl_mode == "atr":       sl = entry - atr[i] * sl_atr_mult
            else: sl = st["or_lo"]
            if sl >= entry:
                continue
            risk = entry - sl
            tp = entry + (rng * tp_range_mult if tp_mode == "range_mult" else risk * tp_r)
            sigs[i] = {"direction": "LONG", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp),
                       "risk": float(risk), "date": d, "hr": int(h)}
            st["taken"] = True

        elif cl[i] < st["or_lo"]:
            entry = cl[i]
            if sl_mode == "opposite":   sl = st["or_hi"]
            elif sl_mode == "mid":       sl = st["or_hi"] - rng * 0.5
            elif sl_mode == "atr":       sl = entry + atr[i] * sl_atr_mult
            else: sl = st["or_hi"]
            if sl <= entry:
                continue
            risk = sl - entry
            tp = entry - (rng * tp_range_mult if tp_mode == "range_mult" else risk * tp_r)
            sigs[i] = {"direction": "SHORT", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp),
                       "risk": float(risk), "date": d, "hr": int(h)}
            st["taken"] = True

    return sigs


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 3 — Previous Day High/Low Breakout (PDH/PDL)
# ═══════════════════════════════════════════════════════════════════════════

def gen_pd_signals(arr, sess_ranges, params):
    """Previous Day High/Low breakout during NY session."""
    open_hr     = params.get("open_hr", 13)
    close_hr    = params.get("close_hr", 22)
    min_range_atr = params.get("min_range_atr", 0.5)
    sl_mode     = params.get("sl_mode", "atr")
    sl_atr_mult = params.get("sl_atr_mult", 1.0)
    tp_mode     = params.get("tp_mode", "fixed_r")
    tp_r        = params.get("tp_r", 1.5)
    tp_range_mult = params.get("tp_range_mult", 0.5)

    n = arr["n"]
    hi = arr["hi"]; lo = arr["lo"]; cl = arr["cl"]
    atr = arr["atr"]; hours = arr["hours"]; dates = arr["dates"]; ts = arr["ts"]
    sigs = {}
    taken_today = defaultdict(lambda: False)

    for i in range(200, n):
        if np.isnan(atr[i]):
            continue
        h = hours[i]; d = dates[i]
        if h < open_hr or h >= close_hr:
            continue
        if d not in sess_ranges: continue
        sr = sess_ranges[d]
        pdh = sr["pd_hi"]; pdl = sr["pd_lo"]
        if np.isnan(pdh) or np.isnan(pdl) or pdh <= pdl:
            continue
        pd_rng = pdh - pdl
        if pd_rng / atr[i] < min_range_atr:
            continue
        if taken_today[d]:
            continue
        close_i = cl[i]

        if close_i > pdh:
            entry = close_i
            if sl_mode == "atr":   sl = entry - atr[i] * sl_atr_mult
            else:                   sl = pdl
            if sl >= entry: continue
            risk = entry - sl
            tp = entry + (risk * tp_r if tp_mode == "fixed_r" else pd_rng * tp_range_mult)
            sigs[i] = {"direction": "LONG", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp),
                       "risk": float(risk), "date": d, "hr": int(h)}
            taken_today[d] = True

        elif close_i < pdl:
            entry = close_i
            if sl_mode == "atr":   sl = entry + atr[i] * sl_atr_mult
            else:                   sl = pdh
            if sl <= entry: continue
            risk = sl - entry
            tp = entry - (risk * tp_r if tp_mode == "fixed_r" else pd_rng * tp_range_mult)
            sigs[i] = {"direction": "SHORT", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp),
                       "risk": float(risk), "date": d, "hr": int(h)}
            taken_today[d] = True
    return sigs


# ═══════════════════════════════════════════════════════════════════════════
#  STRATEGY 4 — Sweep & Reverse (ICT-style)
# ═══════════════════════════════════════════════════════════════════════════

def gen_sweep_signals(arr, sess_ranges, params):
    """At NY open: check if price has swept Asian or London extreme (made new HH/LL then closed back inside).
    Take reversal trade.
    """
    open_hr     = params.get("open_hr", 13)
    close_hr    = params.get("close_hr", 20)
    sweep_ref   = params.get("sweep_ref", "asian")   # "asian" | "london" | "both"
    sl_buf_atr  = params.get("sl_buf_atr", 0.25)
    tp_mode     = params.get("tp_mode", "fixed_r")
    tp_r        = params.get("tp_r", 2.0)
    need_confirm = params.get("need_confirm", True)  # need close back inside range

    n = arr["n"]
    hi = arr["hi"]; lo = arr["lo"]; cl = arr["cl"]
    atr = arr["atr"]; hours = arr["hours"]; dates = arr["dates"]
    sigs = {}
    taken_today = defaultdict(lambda: False)

    for i in range(200, n):
        if np.isnan(atr[i]):
            continue
        h = hours[i]; d = dates[i]
        if h < open_hr or h >= close_hr:
            continue
        if d not in sess_ranges: continue
        if taken_today[d]: continue
        sr = sess_ranges[d]

        # Pick reference range
        if sweep_ref == "asian":
            rh = sr["asian_hi"]; rl = sr["asian_lo"]
        elif sweep_ref == "london":
            rh = sr["london_hi"]; rl = sr["london_lo"]
        else:
            rh = max(sr["asian_hi"], sr["london_hi"] if sr["london_hi"] != -np.inf else sr["asian_hi"])
            rl = min(sr["asian_lo"], sr["london_lo"] if sr["london_lo"] != np.inf else sr["asian_lo"])
        if rh == -np.inf or rl == np.inf or rh <= rl:
            continue

        # Sweep high → reversal SHORT
        if hi[i] > rh:
            if need_confirm and cl[i] >= rh:
                continue  # didn't close back inside
            entry = cl[i]
            sl = hi[i] + atr[i] * sl_buf_atr
            if sl <= entry: continue
            risk = sl - entry
            tp = entry - risk * tp_r if tp_mode == "fixed_r" else rl
            if tp >= entry: continue
            sigs[i] = {"direction": "SHORT", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp),
                       "risk": float(risk), "date": d, "hr": int(h)}
            taken_today[d] = True

        # Sweep low → reversal LONG
        elif lo[i] < rl:
            if need_confirm and cl[i] <= rl:
                continue
            entry = cl[i]
            sl = lo[i] - atr[i] * sl_buf_atr
            if sl >= entry: continue
            risk = entry - sl
            tp = entry + risk * tp_r if tp_mode == "fixed_r" else rh
            if tp <= entry: continue
            sigs[i] = {"direction": "LONG", "entry_price": float(entry),
                       "sl": float(sl), "tp": float(tp),
                       "risk": float(risk), "date": d, "hr": int(h)}
            taken_today[d] = True
    return sigs


# ═══════════════════════════════════════════════════════════════════════════
#  SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def simulate(signals_by_sym, arrays_by_sym, session_end_hr=22, trail_cfg=None,
             risk_pct=RISK_PER_TRADE, cap_frac=1.0):
    """Trade simulator: execute each signal sequentially with per-bar SL/TP/EOD checks.
    trail_cfg: None or {"start_r": 1.0, "atr_mult": 1.5}
    session_end_hr: force exit at this UTC hour if still open (session-bound strategy)
    """
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    completed = []
    open_pos = {}  # sym -> pos dict

    all_syms = list(signals_by_sym.keys())
    max_n = max(arrays_by_sym[s]["n"] for s in all_syms)

    for i in range(max_n):
        for sym in all_syms:
            arr = arrays_by_sym[sym]
            if i >= arr["n"]:
                continue
            hi_i = arr["hi"][i]; lo_i = arr["lo"][i]; cl_i = arr["cl"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            # --- manage open position ---
            if sym in open_pos:
                pos = open_pos[sym]
                d = pos["d"]; entry = pos["entry"]; sl = pos["sl"]; tp = pos["tp"]
                risk = pos["risk"]; qty = pos["qty"]
                bars = i - pos["bar_i"]

                # MFE/MAE
                if d == "LONG":
                    pos["mfe"] = max(pos["mfe"], (hi_i - entry) / risk)
                    pos["mae"] = max(pos["mae"], (entry - lo_i) / risk)
                else:
                    pos["mfe"] = max(pos["mfe"], (entry - lo_i) / risk)
                    pos["mae"] = max(pos["mae"], (hi_i - entry) / risk)

                # Trail SL (activated after price moves R in our favor)
                if trail_cfg is not None:
                    start_r = trail_cfg.get("start_r", 1.0)
                    atr_mult = trail_cfg.get("atr_mult", 1.5)
                    if d == "LONG":
                        move_r = (hi_i - entry) / risk
                        if move_r >= start_r:
                            pos["best"] = max(pos.get("best", entry), hi_i)
                            ns = pos["best"] - atr_i * atr_mult
                            if ns > sl:
                                sl = ns
                                pos["sl"] = sl
                    else:
                        move_r = (entry - lo_i) / risk
                        if move_r >= start_r:
                            pos["best"] = min(pos.get("best", entry), lo_i)
                            ns = pos["best"] + atr_i * atr_mult
                            if ns < sl:
                                sl = ns
                                pos["sl"] = sl

                # Check exits in the proper order: first SL (worst), then TP
                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = (d == "LONG" and hi_i >= tp) or (d == "SHORT" and lo_i <= tp)

                # If both hit in same bar, assume SL first (conservative)
                exit_price = None; reason = None
                if hit_sl:
                    exit_price = sl; reason = "sl"
                elif hit_tp:
                    exit_price = tp; reason = "tp"

                # EOD exit at session_end_hr
                if exit_price is None and session_end_hr is not None:
                    if arr["hours"][i] >= session_end_hr or arr["dates"][i] != pos["date"]:
                        exit_price = cl_i; reason = "eod"

                if exit_price is not None:
                    gross = ((exit_price - entry) if d == "LONG" else (entry - exit_price)) * qty
                    fee = exit_price * qty * (TAKER_FEE + SLIPPAGE)
                    net = gross - fee
                    balance += net
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    exit_r = ((exit_price - entry) / risk) if d == "LONG" else ((entry - exit_price) / risk)
                    completed.append({
                        "sym": sym, "d": d, "entry": round(entry, 6),
                        "exit": round(exit_price, 6), "sl": round(pos["sl0"], 6), "tp": round(tp, 6),
                        "bars": bars, "exit_r": round(exit_r, 3),
                        "mfe": round(pos["mfe"], 3), "mae": round(pos["mae"], 3),
                        "won": net > 0, "pnl_usd": round(net, 4), "reason": reason,
                        "date": pos["date"], "hr": pos["hr"],
                    })
                    del open_pos[sym]
                    continue

            # --- open new position ---
            if sym in open_pos:
                continue
            sigs_sym = signals_by_sym.get(sym, {})
            if i not in sigs_sym:
                continue
            sig = sigs_sym[i]
            entry = sig["entry_price"]; sl = sig["sl"]; tp = sig["tp"]
            risk = abs(entry - sl)
            if risk / entry > 0.1 or risk / entry < 0.0005:
                continue
            alloc = balance * cap_frac
            risk_usd = alloc * risk_pct
            qty = risk_usd / risk
            notional = qty * entry
            if notional / MAX_LEVERAGE > alloc * 0.9:
                qty = alloc * 0.9 * MAX_LEVERAGE / entry
            if qty * entry < 5.0:
                continue
            fee_open = entry * qty * (TAKER_FEE + SLIPPAGE)
            balance -= fee_open
            open_pos[sym] = {
                "d": sig["direction"], "entry": entry, "sl": sl, "sl0": sl,
                "tp": tp, "risk": risk, "qty": qty, "bar_i": i,
                "mfe": 0.0, "mae": 0.0, "date": sig["date"], "hr": sig["hr"],
            }

    return _aggregate(completed, balance, max_dd)


def _aggregate(completed, balance, max_dd):
    if not completed:
        return {"trades": 0, "wins": 0, "losses": 0, "wr_pct": 0.0,
                "pnl_usd": 0.0, "pnl_pct": 0.0, "profit_factor": 0.0,
                "max_dd_pct": 0.0, "avg_r": 0.0, "avg_bars": 0.0,
                "final_bal": balance, "exit_reasons": {}}
    wins = [t for t in completed if t["won"]]
    losses = [t for t in completed if not t["won"]]
    n = len(completed)
    gp = sum(t["pnl_usd"] for t in wins)
    gl = abs(sum(t["pnl_usd"] for t in losses))
    pf = gp / gl if gl > 0 else (999.0 if gp > 0 else 0.0)
    wr = len(wins) / n * 100
    reasons = defaultdict(int)
    for t in completed: reasons[t["reason"]] += 1
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "wr_pct": round(wr, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "pnl_pct": round((balance - START_BALANCE) / START_BALANCE * 100, 2),
        "profit_factor": round(pf, 3),
        "max_dd_pct": round(max_dd, 2),
        "avg_r": round(np.mean([t["exit_r"] for t in completed]), 3),
        "avg_mfe": round(np.mean([t["mfe"] for t in completed]), 3),
        "avg_mae": round(np.mean([t["mae"] for t in completed]), 3),
        "avg_bars": round(np.mean([t["bars"] for t in completed]), 1),
        "final_bal": round(balance, 2),
        "exit_reasons": dict(reasons),
    }


# ═══════════════════════════════════════════════════════════════════════════
#  RUN TEST
# ═══════════════════════════════════════════════════════════════════════════

def run_strategy(strat_name, gen_fn, arrays_by_sym, ranges_by_sym, params,
                 trail_cfg=None, session_end_hr=22, _daily_trends=None):
    """Run one configuration across all symbols."""
    sigs_by_sym = {}
    # Decide whether gen function needs ranges by inspecting its name
    fname = gen_fn.__name__
    needs_ranges = fname in ("gen_arb_signals", "gen_pd_signals",
                             "gen_sweep_signals", "gen_mr_signals")
    for sym, arr in arrays_by_sym.items():
        p = dict(params)
        if _daily_trends is not None and sym in _daily_trends:
            p["_dtrend"] = _daily_trends[sym]
        if needs_ranges:
            sigs = gen_fn(arr, ranges_by_sym[sym], p)
        else:
            sigs = gen_fn(arr, p)
        sigs_by_sym[sym] = sigs
    res = simulate(sigs_by_sym, arrays_by_sym, session_end_hr=session_end_hr,
                    trail_cfg=trail_cfg)
    total_sigs = sum(len(s) for s in sigs_by_sym.values())
    res["total_sigs"] = total_sigs
    return res


def main():
    t0 = _time.time()
    logger.info("=== Session-based backtest ===")
    logger.info("Loading data for %s...", SYMBOLS)

    arrays_by_sym = {}
    ranges_by_sym = {}
    daily_trends = {}
    for sym in SYMBOLS:
        df15 = _load(sym, "15m", "365d")
        df1d = _load(sym, "1d", "365d")
        if df15 is None or len(df15) < 300:
            logger.warning("No 15m data for %s — skipping", sym)
            continue
        logger.info("%s: %d 15m bars loaded", sym, len(df15))
        arr = prep_15m_arrays(df15)
        ranges = build_session_ranges(arr)
        arrays_by_sym[sym] = arr
        ranges_by_sym[sym] = ranges
        daily_trends[sym] = compute_daily_trend(df1d)

    results = []

    # ─── Phase 2: strict entries + trend filters + MR strategy ───────────
    logger.info("--- Phase 2: Strict + filtered ---")

    configs = [
        # ===== Asian Range Breakout =====
        ("ARB-NY_strict_break", gen_arb_signals, {
            "open_hr": 13, "close_hr": 22, "min_range_atr": 0.5, "max_range_atr": 4.0,
            "sl_mode": "opposite", "tp_mode": "range_mult", "tp_range_mult": 1.0,
            "strict_break": True, "require_inside_at_open": False,
        }, 22),
        ("ARB-NY_strict_trendwith", gen_arb_signals, {
            "open_hr": 13, "close_hr": 22, "min_range_atr": 0.5, "max_range_atr": 4.0,
            "sl_mode": "opposite", "tp_mode": "range_mult", "tp_range_mult": 1.0,
            "strict_break": True, "require_inside_at_open": False,
            "dtrend_filter": "with",
        }, 22),
        ("ARB-NY_atrSL_R20", gen_arb_signals, {
            "open_hr": 13, "close_hr": 22, "min_range_atr": 0.5, "max_range_atr": 4.0,
            "sl_mode": "atr", "sl_atr_mult": 1.5,
            "tp_mode": "fixed_r", "tp_r": 2.0,
            "strict_break": True, "require_inside_at_open": False,
            "dtrend_filter": "with",
        }, 22),
        ("ARB-NY_intactopen", gen_arb_signals, {
            "open_hr": 13, "close_hr": 22, "min_range_atr": 0.5, "max_range_atr": 4.0,
            "sl_mode": "opposite", "tp_mode": "range_mult", "tp_range_mult": 1.0,
            "strict_break": True, "require_inside_at_open": True,
        }, 22),
        ("ARB-LO_strict", gen_arb_signals, {
            "open_hr": 7, "close_hr": 13, "min_range_atr": 0.5, "max_range_atr": 4.0,
            "sl_mode": "opposite", "tp_mode": "range_mult", "tp_range_mult": 1.0,
            "strict_break": True, "require_inside_at_open": False,
        }, 13),
        ("ARB-LO_strict_trendwith", gen_arb_signals, {
            "open_hr": 7, "close_hr": 13, "min_range_atr": 0.5, "max_range_atr": 4.0,
            "sl_mode": "opposite", "tp_mode": "range_mult", "tp_range_mult": 1.0,
            "strict_break": True, "require_inside_at_open": False,
            "dtrend_filter": "with",
        }, 13),

        # ===== ORB =====
        ("ORB-NY_30m_base", gen_orb_signals, {
            "open_hr": 13, "or_bars": 2, "window_h": 3, "min_width_atr": 0.5,
            "sl_mode": "opposite", "tp_mode": "fixed_r", "tp_r": 2.0,
        }, 22),
        ("ORB-NY_60m_base", gen_orb_signals, {
            "open_hr": 13, "or_bars": 4, "window_h": 3, "min_width_atr": 0.5,
            "sl_mode": "opposite", "tp_mode": "fixed_r", "tp_r": 2.0,
        }, 22),

        # ===== Mean Reversion (user's described idea) =====
        ("MR-NY_asian_opposite", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "asian",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "opposite",
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 22),
        ("MR-NY_combined_opposite", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "combined",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "opposite",
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 22),
        ("MR-NY_combined_mid", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "combined",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "mid",
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 22),
        ("MR-NY_combined_R15", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "combined",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "fixed_r", "tp_r": 1.5,
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 22),
        ("MR-NY_combined_R20", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "combined",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "fixed_r", "tp_r": 2.0,
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 22),
        ("MR-NY_combined_trendwith_opp", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "combined",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "opposite",
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
            "dtrend_filter": True,
        }, 22),
        ("MR-NY_tight_zone15", gen_mr_signals, {
            "open_hr": 13, "close_hr": 22, "ref_source": "combined",
            "zone_pct": 0.15, "sl_buf_atr": 0.3, "tp_mode": "opposite",
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 22),
        ("MR-LO_asian_opposite", gen_mr_signals, {
            "open_hr": 7, "close_hr": 13, "ref_source": "asian",
            "zone_pct": 0.25, "sl_buf_atr": 0.3, "tp_mode": "opposite",
            "min_range_atr": 0.8, "max_range_atr": 5.0, "entry_window_h": 3,
        }, 13),

        # ===== Sweep & reverse =====
        ("Sweep-NY_combined_R20", gen_sweep_signals, {
            "open_hr": 13, "close_hr": 20, "sweep_ref": "both",
            "sl_buf_atr": 0.3, "tp_mode": "fixed_r", "tp_r": 2.0, "need_confirm": True,
        }, 22),
    ]

    for name, fn, p, sess_end in configs:
        r = run_strategy(name, fn, arrays_by_sym, ranges_by_sym, p,
                          session_end_hr=sess_end, _daily_trends=daily_trends)
        r["config_id"] = name; r["params"] = {k: v for k, v in p.items() if not k.startswith("_")}
        results.append(r)
        logger.info("%-28s trades=%3d  WR=%5.1f%%  PF=%5.2f  PnL=%+7.1f%%  DD=%5.2f%%",
                    name, r["trades"], r["wr_pct"], r["profit_factor"],
                    r["pnl_pct"], r["max_dd_pct"])

    # ─── Phase 3: Grid optimize ARB-NY (the only positive baseline) ─────
    logger.info("--- Phase 3: Grid optimize ARB-NY ---")
    grid = []
    for sl_atr in [1.0, 1.5, 2.0]:
        for tp_r in [1.5, 2.0, 2.5, 3.0]:
            for min_rng in [0.3, 0.5, 0.8]:
                for max_rng in [3.0, 5.0]:
                    for dtf in ["with", None]:
                        for strict in [True, False]:
                            grid.append({
                                "open_hr": 13, "close_hr": 22,
                                "min_range_atr": min_rng, "max_range_atr": max_rng,
                                "sl_mode": "atr", "sl_atr_mult": sl_atr,
                                "tp_mode": "fixed_r", "tp_r": tp_r,
                                "strict_break": strict, "require_inside_at_open": False,
                                "dtrend_filter": dtf,
                            })
    logger.info("Phase 3 grid size: %d configs", len(grid))
    grid_results = []
    for idx, p in enumerate(grid):
        name = f"ARB_slA{p['sl_atr_mult']}_r{p['tp_r']}_R{p['min_range_atr']}-{p['max_range_atr']}_dt{p['dtrend_filter'] or 'none'}_{'strict' if p['strict_break'] else 'free'}"
        r = run_strategy(name, gen_arb_signals, arrays_by_sym, ranges_by_sym, p,
                          session_end_hr=22, _daily_trends=daily_trends)
        r["config_id"] = name; r["params"] = {k: v for k, v in p.items() if not k.startswith("_")}
        grid_results.append(r)
        if idx % 20 == 0:
            logger.info("  grid %3d/%d", idx+1, len(grid))

    # Show top 15
    grid_results.sort(key=lambda r: (r["profit_factor"], r["pnl_pct"]), reverse=True)
    print("\n=== Phase 3 TOP-15 ARB-NY grid ===")
    print(f"{'name':<58s} {'trades':>6s} {'WR%':>6s} {'PF':>6s} {'PnL%':>7s} {'DD%':>6s} {'avgR':>6s}")
    for r in grid_results[:15]:
        print(f"{r['config_id']:<58s} {r['trades']:>6d} {r['wr_pct']:>6.1f} "
              f"{r['profit_factor']:>6.2f} {r['pnl_pct']:>+7.1f} "
              f"{r['max_dd_pct']:>6.2f} {r['avg_r']:>+6.2f}")

    results.extend(grid_results[:20])
    best_phase3 = grid_results[0]

    # ─── Phase 4: time-window + volume filters ──────────────────────────
    logger.info("--- Phase 4: Time-window + volume filters ---")

    base_p = dict(best_phase3["params"])
    phase4_configs = []

    # 4a: time-window restrictions (only N hours after session open)
    for mh in [2, 3, 4, 5]:
        p = dict(base_p); p["max_entry_hr"] = mh
        phase4_configs.append((f"P4_window_{mh}h", p, None))

    # 4b: volume filter
    for vm in [1.0, 1.2, 1.5, 2.0]:
        p = dict(base_p); p["vol_mult"] = vm; p["max_entry_hr"] = 4
        phase4_configs.append((f"P4_vol{vm}_w4h", p, None))

    # 4c: combo — window + vol + stricter trend
    p = dict(base_p); p["max_entry_hr"] = 3; p["vol_mult"] = 1.2
    phase4_configs.append(("P4_w3h_vol1.2", p, None))

    p = dict(base_p); p["max_entry_hr"] = 2; p["vol_mult"] = 1.2
    phase4_configs.append(("P4_w2h_vol1.2", p, None))

    # 4d: add strict breakout with window
    p = dict(base_p); p["strict_break"] = True; p["max_entry_hr"] = 3
    phase4_configs.append(("P4_strict_w3h", p, None))

    p = dict(base_p); p["strict_break"] = True; p["max_entry_hr"] = 3; p["vol_mult"] = 1.2
    phase4_configs.append(("P4_strict_w3h_vol1.2", p, None))

    # 4e: try TP range_mult (adaptive to range size)
    p = dict(base_p); p["tp_mode"] = "range_mult"; p["tp_range_mult"] = 1.5; p["max_entry_hr"] = 3
    phase4_configs.append(("P4_rangeTP1.5_w3h", p, None))

    # 4f: higher r (trails not helpful earlier; try fixed r=2.5/3 with window)
    for tp_r in [2.5, 3.0]:
        p = dict(base_p); p["tp_r"] = tp_r; p["max_entry_hr"] = 3
        phase4_configs.append((f"P4_r{tp_r}_w3h", p, None))

    phase4_results = []
    for name, p, trail_cfg in phase4_configs:
        r = run_strategy(name, gen_arb_signals, arrays_by_sym, ranges_by_sym, p,
                          session_end_hr=22, trail_cfg=trail_cfg,
                          _daily_trends=daily_trends)
        r["config_id"] = name
        r["params"] = {k: v for k, v in p.items() if not k.startswith("_")}
        phase4_results.append(r)
        logger.info("%-30s trades=%3d  WR=%5.1f%%  PF=%5.2f  PnL=%+7.1f%%  DD=%5.2f%%  avgR=%+.2f",
                    name, r["trades"], r["wr_pct"], r["profit_factor"],
                    r["pnl_pct"], r["max_dd_pct"], r["avg_r"])

    phase4_results.sort(key=lambda r: (r["pnl_pct"], r["profit_factor"]), reverse=True)
    print("\n=== Phase 4 (time/vol filters) ranking (by PnL) ===")
    print(f"{'name':<32s} {'trades':>6s} {'WR%':>6s} {'PF':>6s} {'PnL%':>7s} {'DD%':>6s} {'avgR':>6s}")
    for r in phase4_results:
        print(f"{r['config_id']:<32s} {r['trades']:>6d} {r['wr_pct']:>6.1f} "
              f"{r['profit_factor']:>6.2f} {r['pnl_pct']:>+7.1f} "
              f"{r['max_dd_pct']:>6.2f} {r['avg_r']:>+6.2f}")
    results.extend(phase4_results)

    # ─── Phase 5: per-symbol breakdown of top configs ────────────────────
    logger.info("--- Phase 5: Per-symbol analysis of top ---")
    top_configs = [("BEST_grid", dict(best_phase3["params"]), None)]
    if phase4_results:
        top_configs.append(("BEST_phase4", phase4_results[0]["params"], None))

    print("\n=== Per-symbol breakdown ===")
    print(f"{'config':<14s} {'symbol':<16s} {'trades':>6s} {'WR%':>6s} {'PF':>6s} {'PnL%':>7s} {'avgR':>6s}")
    for cname, cp, ctrail in top_configs:
        for sym in SYMBOLS:
            if sym not in arrays_by_sym: continue
            tmp_arr = {sym: arrays_by_sym[sym]}
            tmp_rng = {sym: ranges_by_sym[sym]}
            tmp_dt = {sym: daily_trends[sym]}
            r = run_strategy(cname+sym, gen_arb_signals, tmp_arr, tmp_rng, cp,
                              session_end_hr=22, trail_cfg=ctrail,
                              _daily_trends=tmp_dt)
            print(f"{cname:<14s} {sym.replace('/USDT:USDT',''):<16s} "
                  f"{r['trades']:>6d} {r['wr_pct']:>6.1f} {r['profit_factor']:>6.2f} "
                  f"{r['pnl_pct']:>+7.1f} {r['avg_r']:>+6.2f}")

    # ─── Save ───────────────────────────────────────────────────────────
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"backtest_session_phase4_{stamp}.json"
    out = {"generated": stamp, "symbols": SYMBOLS, "phase": "phase2",
            "duration_sec": round(_time.time() - t0, 1),
            "results": results}
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    logger.info("Phase 2 saved to %s (%.1fs)", out_path, _time.time() - t0)

    # Sort and display
    results.sort(key=lambda r: (r["profit_factor"], r["pnl_pct"]), reverse=True)
    print("\n=== Phase 2 ranking (by PF) ===")
    print(f"{'name':<28s} {'trades':>6s} {'WR%':>6s} {'PF':>6s} {'PnL%':>7s} {'DD%':>6s} {'avgR':>6s}")
    for r in results:
        print(f"{r['config_id']:<28s} {r['trades']:>6d} {r['wr_pct']:>6.1f} "
              f"{r['profit_factor']:>6.2f} {r['pnl_pct']:>+7.1f} "
              f"{r['max_dd_pct']:>6.2f} {r['avg_r']:>+6.2f}")

    return out_path


if __name__ == "__main__":
    main()
