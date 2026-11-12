"""
backtest_new_strategies.py  --  New strategy research & TP/SL grid search
==========================================================================

Стратегии для исследования:
  KC_4H_15M      : 4h KC squeeze → 15m next_bar  (v10 validated: PF=5.68, исторически)
  BBS_4H_5M      : 4h BB squeeze → 5m pullback_ema21  (cross-TF, консервативно)
  BBS_1H_15M_PB  : 1h BB squeeze → 15m pullback_ema21 (vs текущий next_bar BBS_1H_15M)

TP/SL grid per strategy:
  SL_MULTS = [1.0, 1.5, 2.0]  × ATR
  TP_RS    = [1.5, 2.0, 2.5, 3.0, 4.0]  × risk (R)

Методология:
  - IS = первые 80% данных (по времени), OOS = последние 20%
  - Лучший конфиг по IS → валидация на OOS
  - Отбор: PF ≥ 2.0 AND WR ≥ 60% AND DD ≤ 35% (на IS)
  - Финал: combo-портфель из лучших per-стратегии конфигов

ПРЕДУПРЕЖДЕНИЕ:
  - 4h данных может не быть если fetch_data_2y.py ещё не закончил
  - Символы берём из TOP_50 (survivorship bias — только текущий топ-50 mcap)
  - При старте $100 + 2% риска большинство сделок на дешёвых монетах заблокировано
    min_margin=$5 → используем $500 старт для более реалистичных результатов

Запуск:
  py -3 multi_agent/backtests/backtest_new_strategies.py
  py -3 multi_agent/backtests/backtest_new_strategies.py > new_strats_out.txt
"""
from __future__ import annotations
import sys, logging, time as _time
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timezone
from itertools import product

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

logger = logging.getLogger("new_strats")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── Параметры портфеля ───────────────────────────────────────────────────────
START_BALANCE   = 500.0     # $500 — реалистично, не блокирует min_margin
RISK_PER_TRADE  = 0.01      # 1% от депозита
TAKER_FEE       = 0.00055   # 0.055% taker
MAKER_FEE       = 0.00020   # 0.020% maker (для next_bar стратегий)
SLIPPAGE        = 0.0005    # 0.05%
MIN_MARGIN      = 5.0       # $5 минимум Bybit
MAX_LEVERAGE    = 50
MAX_MARGIN_FRAC = 0.05      # 5% депозита макс маржа
MAX_POSITIONS   = 4
SL_COOLDOWN_MS  = 4 * 3_600_000   # 4 часа кулдаун после стопа

# ── TP/SL Grid ───────────────────────────────────────────────────────────────
SL_MULTS = [1.0, 1.5, 2.0]
TP_RS    = [1.5, 2.0, 2.5, 3.0, 4.0]

# ── Символы (TOP_50, те что должны быть в кеше) ──────────────────────────────
SYMBOLS = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "BNB/USDT:USDT",  "SOL/USDT:USDT",
    "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT",  "LINK/USDT:USDT", "UNI/USDT:USDT",  "ATOM/USDT:USDT",
    "NEAR/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",  "OP/USDT:USDT",
    "INJ/USDT:USDT",  "SUI/USDT:USDT",  "TRX/USDT:USDT",  "TON/USDT:USDT",
    "BCH/USDT:USDT",  "LTC/USDT:USDT",  "ICP/USDT:USDT",  "HBAR/USDT:USDT",
    "XLM/USDT:USDT",  "FIL/USDT:USDT",  "ALGO/USDT:USDT", "POL/USDT:USDT",
    "SEI/USDT:USDT",  "XMR/USDT:USDT",  "CRV/USDT:USDT",  "MNT/USDT:USDT",
    "XAUT/USDT:USDT", "ONDO/USDT:USDT", "1000PEPE/USDT:USDT",
    "AAVE/USDT:USDT", "RENDER/USDT:USDT","LDO/USDT:USDT",  "STX/USDT:USDT",
    "DYDX/USDT:USDT", "IMX/USDT:USDT",  "GRT/USDT:USDT",  "SAND/USDT:USDT",
    "GALA/USDT:USDT", "FTM/USDT:USDT",  "EGLD/USDT:USDT", "WLD/USDT:USDT",
    "PENDLE/USDT:USDT","TAO/USDT:USDT", "1000BONK/USDT:USDT",
]

# ── Data loading ─────────────────────────────────────────────────────────────

def _load_raw(symbol: str, tf: str) -> pd.DataFrame | None:
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in ("730d", "365d", "180d"):
        for ext in (".parquet", ".csv"):
            p = CACHE_DIR / f"{safe}_{tf}_{tag}{ext}"
            if p.exists():
                try:
                    df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
                    if df is not None and not df.empty:
                        for c in ("open","high","low","close","volume"):
                            if c in df.columns:
                                df[c] = df[c].astype(float)
                        if "timestamp" in df.columns:
                            df["timestamp"] = df["timestamp"].astype(int)
                        return df
                except Exception:
                    continue
    return None


# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    """EMA via pandas ewm — ~20× faster than Python loop on large arrays."""
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    # pandas ewm(span=p, adjust=False) is equivalent to k=2/(p+1)
    s = pd.Series(arr)
    result = s.ewm(span=p, adjust=False).mean().to_numpy()
    # Blank out warmup period to match our convention
    out[p - 1:] = result[p - 1:]
    return out

def _sma(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs  = np.nancumsum(arr)
    out[p - 1:] = (cs[p - 1:] - np.concatenate(([0.0], cs[:-p]))) / p
    return out

def _stdev(arr: np.ndarray, p: int) -> np.ndarray:
    """Vectorized rolling std using cumsum trick — O(n) not O(n*p)."""
    n = len(arr)
    out = np.full(n, np.nan)
    if n < p:
        return out
    a = np.asarray(arr, dtype=float)
    cs  = np.cumsum(a)
    cs2 = np.cumsum(a ** 2)
    roll_sum  = np.empty(n - p + 1)
    roll_sum2 = np.empty(n - p + 1)
    roll_sum[0]  = cs[p - 1]
    roll_sum2[0] = cs2[p - 1]
    if n > p:
        roll_sum[1:]  = cs[p:] - cs[:n - p]
        roll_sum2[1:] = cs2[p:] - cs2[:n - p]
    var = (roll_sum2 - roll_sum ** 2 / p) / p
    out[p - 1:] = np.sqrt(np.maximum(var, 0.0))
    return out

def _atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, p: int = 14) -> np.ndarray:
    """Vectorized ATR using Wilder smoothing via EWM."""
    n = len(cl)
    if n < 2:
        return np.full(n, np.nan)
    # True Range — vectorized
    hl  = hi - lo
    hcp = np.abs(hi[1:] - cl[:-1])
    lcp = np.abs(lo[1:] - cl[:-1])
    tr  = np.concatenate([[hl[0]], np.maximum(hl[1:], np.maximum(hcp, lcp))])
    # Wilder smoothing = EWM with alpha=1/p (equiv. com=p-1)
    atr = pd.Series(tr).ewm(com=p - 1, adjust=False).mean().to_numpy()
    out = np.full(n, np.nan)
    out[p - 1:] = atr[p - 1:]
    return out

def _rolling_slope(arr: np.ndarray, p: int) -> np.ndarray:
    """
    Vectorized rolling linear regression slope over window p.
    Returns slope for each point (nan for first p-1 points).
    Uses exact OLS formula via convolution.
    """
    n = len(arr)
    out = np.full(n, np.nan)
    if n < p:
        return out
    xi = np.arange(p, dtype=float)
    xi_c = xi - xi.mean()             # centered x-values
    denom = float(np.sum(xi_c ** 2))  # constant denominator
    if denom == 0:
        return out
    weights = xi_c / denom            # slope = sum(weights * y_window)
    # Convolve to get rolling slope — weights reversed for convolution
    # mode='full' output size = n + p - 1; valid region starts at p-1, length n-p+1
    conv = np.convolve(arr, weights[::-1], mode="full")
    out[p - 1:] = conv[p - 1: n]   # length n-p+1, matching out[p-1:]
    return out


def _adx(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, p: int = 14) -> np.ndarray:
    """Vectorized ADX using numpy + pandas EWM for Wilder smoothing."""
    n = len(cl)
    out = np.full(n, 20.0)
    if n < p * 3:
        return out

    # +DM, -DM, TR vectorized
    up = np.diff(hi, prepend=hi[0])
    dn = np.diff(lo[::-1], prepend=lo[-1])[::-1] * -1  # lo[i-1] - lo[i]
    dn_vec = np.concatenate([[0], lo[:-1] - lo[1:]])

    pdm = np.where((up > dn_vec) & (up > 0), up, 0.0)
    ndm = np.where((dn_vec > up) & (dn_vec > 0), dn_vec, 0.0)

    hl  = hi - lo
    hcp = np.abs(np.concatenate([[0], hi[1:] - cl[:-1]]))
    lcp = np.abs(np.concatenate([[0], lo[1:] - cl[:-1]]))
    tr  = np.maximum(hl, np.maximum(hcp, lcp))

    # Wilder smooth (alpha = 1/p)
    tr_s  = pd.Series(tr).ewm(com=p - 1, adjust=False).mean().to_numpy()
    pdm_s = pd.Series(pdm).ewm(com=p - 1, adjust=False).mean().to_numpy()
    ndm_s = pd.Series(ndm).ewm(com=p - 1, adjust=False).mean().to_numpy()

    with np.errstate(divide="ignore", invalid="ignore"):
        pdi = np.where(tr_s > 0, 100 * pdm_s / tr_s, 0.0)
        ndi = np.where(tr_s > 0, 100 * ndm_s / tr_s, 0.0)
        denom = pdi + ndi
        dx  = np.where(denom > 0, np.abs(pdi - ndi) / denom * 100, 0.0)

    # ADX = EWM of DX
    adx_arr = pd.Series(dx).ewm(com=p - 1, adjust=False).mean().to_numpy()
    out[p * 2:] = adx_arr[p * 2:]
    return out

def _prepare(df: pd.DataFrame, min_bars: int = 60) -> dict | None:
    if df is None or len(df) < min_bars:
        return None
    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    vo = df["volume"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=int)
    sma20 = _sma(cl, 20); std20 = _stdev(cl, 20)
    vol_ma = _sma(vo, 20)
    kc_atr = _atr(hi, lo, cl, 10)
    kc_mid = _ema(cl, 20)
    return {
        "cl": cl, "hi": hi, "lo": lo, "ts": ts, "n": len(cl),
        "atr": _atr(hi, lo, cl, 14),
        "adx": _adx(hi, lo, cl, 14),
        "ema21": _ema(cl, 21), "ema50": _ema(cl, 50), "ema200": _ema(cl, 200),
        "bb_upper": sma20 + 2*std20, "bb_lower": sma20 - 2*std20, "bb_mid": sma20,
        "kc_mid": kc_mid, "kc_atr": kc_atr,
        "vol_ratio": np.where(vol_ma > 0, vo / vol_ma, 1.0),
    }


# ── Signal detection ──────────────────────────────────────────────────────────

def _detect_kc_signals(dhtf: dict, min_sq: int, adx_min: float,
                       vol_mult: float, mom_bars: int) -> dict:
    """
    KC Squeeze на HTF (1h или 4h):
    BB внутри KC ≥ min_sq баров → выход → направление по rolling slope(mom_bars).
    Returns {bar_idx: {direction, atr_v}}
    """
    cl = dhtf["cl"]; n = dhtf["n"]
    adx = dhtf["adx"]; vol_r = dhtf["vol_ratio"]
    bb_u = dhtf["bb_upper"]; bb_l = dhtf["bb_lower"]
    kc_mid = dhtf["kc_mid"]; kc_atr = dhtf["kc_atr"]
    atr_arr = dhtf["atr"]

    kc_u = kc_mid + 1.5 * kc_atr
    kc_l = kc_mid - 1.5 * kc_atr

    # Vectorized squeeze length
    in_sq = np.zeros(n, dtype=bool)
    valid = ~(np.isnan(bb_u) | np.isnan(kc_u))
    in_sq[valid] = (bb_l[valid] > kc_l[valid]) & (bb_u[valid] < kc_u[valid])
    sq_len = np.zeros(n, dtype=int)
    for i in range(1, n):
        sq_len[i] = sq_len[i-1] + 1 if in_sq[i] else 0

    # Pre-compute rolling slope for all bars at once
    slope_arr = _rolling_slope(cl, mom_bars)

    sigs = {}
    warmup = max(250, mom_bars + 20)
    for i in range(warmup, n - 1):
        if in_sq[i]:
            continue
        if sq_len[i-1] < min_sq:
            continue
        if np.isnan(adx[i]) or adx[i] < adx_min:
            continue
        if vol_r[i] < vol_mult:
            continue
        slope = slope_arr[i]
        if np.isnan(slope):
            continue
        atr_v = atr_arr[i] if not np.isnan(atr_arr[i]) else cl[i]*0.01
        if slope > 0:
            if np.isnan(bb_u[i]) or bb_u[i] <= kc_u[i]:
                continue
            d = "LONG"
        else:
            if np.isnan(bb_l[i]) or bb_l[i] >= kc_l[i]:
                continue
            d = "SHORT"
        sigs[i] = {"direction": d, "atr_v": atr_v}
    return sigs


def _detect_bbs_signals(dhtf: dict, min_sq: int, adx_thresh: float,
                        vol_mult: float, use_ema_stack: bool = True,
                        headfake: bool = True) -> dict:
    """
    BB Squeeze на HTF (1h или 4h): vectorized manual implementation.
    100x faster than going through compute_squeeze_indicators.
    """
    return _detect_bbs_manual(dhtf, min_sq, adx_thresh, vol_mult,
                               use_ema_stack, headfake)


def _reconstruct_df(d: dict) -> pd.DataFrame:
    """Reconstruct minimal DataFrame from prepared dict (for bb_squeeze_agent)."""
    return pd.DataFrame({
        "timestamp": d["ts"],
        "open":  d["cl"],   # open not stored → use close as approximation
        "high":  d["hi"],
        "low":   d["lo"],
        "close": d["cl"],
        "volume": np.ones(d["n"]),
    })


def _detect_bbs_manual(dhtf: dict, min_sq: int, adx_thresh: float,
                        vol_mult: float, use_ema_stack: bool,
                        headfake: bool) -> dict:
    """
    Manual BBS detection (fallback когда bb_squeeze_agent недоступен).
    Uses same logic as backtest_v12 detect_bbs_signals_1h.
    """
    cl = dhtf["cl"]; hi = dhtf["hi"]; lo = dhtf["lo"]
    n  = dhtf["n"]
    adx = dhtf["adx"]; vol_r = dhtf["vol_ratio"]
    ema21 = dhtf["ema21"]; ema50 = dhtf["ema50"]; ema200 = dhtf["ema200"]
    bb_u = dhtf["bb_upper"]; bb_l = dhtf["bb_lower"]
    kc_mid = dhtf["kc_mid"]; kc_atr = dhtf["kc_atr"]

    kc_u = kc_mid + 1.5 * kc_atr
    kc_l = kc_mid - 1.5 * kc_atr

    # Squeeze: BB inside KC (vectorized)
    valid = ~(np.isnan(bb_u) | np.isnan(kc_u))
    sq = np.zeros(n, dtype=bool)
    sq[valid] = (bb_l[valid] > kc_l[valid]) & (bb_u[valid] < kc_u[valid])
    sq_len = np.zeros(n, dtype=int)
    for i in range(1, n):
        sq_len[i] = sq_len[i-1] + 1 if sq[i] else 0

    # Momentum = close - LinReg(close, 20)
    # LinReg value at end of window = EMA20 of fitted line at last point
    # Approximate: use rolling_slope * (p-1)/2 from SMA20 (efficient)
    p_mom = 20
    sma20_arr = _sma(cl, p_mom)
    slope20 = _rolling_slope(cl, p_mom)
    # LinReg end value = SMA20 + slope * (p-1)/2
    linreg_end = sma20_arr + slope20 * ((p_mom - 1) / 2.0)
    mom = cl - linreg_end

    sigs = {}
    warmup = max(250, int(n * 0.04))
    for i in range(warmup, n - 1):
        if sq[i]:
            continue
        if sq_len[i-1] < min_sq:
            continue
        if np.isnan(adx[i]) or adx[i] < adx_thresh:
            continue
        if vol_r[i] < vol_mult:
            continue
        if np.isnan(mom[i]):
            continue
        # Head-fake: momentum reversed next bar
        if headfake and i + 1 < n and not np.isnan(mom[i+1]):
            if mom[i] > 0 and mom[i+1] < 0:
                continue
            if mom[i] < 0 and mom[i+1] > 0:
                continue
        d = "LONG" if mom[i] > 0 else "SHORT"
        if use_ema_stack:
            if d == "LONG" and not (cl[i] > ema21[i] > ema50[i] > ema200[i]):
                continue
            if d == "SHORT" and not (cl[i] < ema21[i] < ema50[i] < ema200[i]):
                continue
        atr_v = dhtf["atr"][i] if not np.isnan(dhtf["atr"][i]) else cl[i]*0.01
        # Confirm: BB must be breaking out of KC in direction
        if d == "LONG" and (np.isnan(bb_u[i]) or bb_u[i] <= kc_u[i]):
            continue
        if d == "SHORT" and (np.isnan(bb_l[i]) or bb_l[i] >= kc_l[i]):
            continue
        sigs[i] = {"direction": d, "atr_v": atr_v}
    return sigs


# ── LTF entry matching ────────────────────────────────────────────────────────

def _match_ltf(htf_sigs: dict, dhtf: dict, dltf: dict,
               entry_mode: str, max_wait: int,
               sl_atr: float, tp_r: float) -> dict:
    """
    Match HTF signals to LTF entry bars.
    Returns {ltf_bar_idx: trade_dict}
    """
    if dltf is None or not htf_sigs:
        return {}

    ts_htf = dhtf["ts"]
    ts_ltf = dltf["ts"]
    cl_ltf = dltf["cl"]; hi_ltf = dltf["hi"]; lo_ltf = dltf["lo"]
    atr_ltf = dltf["atr"]; ema21_ltf = dltf["ema21"]
    n_ltf = dltf["n"]

    entries = {}
    for htf_bar, sig in htf_sigs.items():
        htf_ts    = ts_htf[htf_bar]
        direction = sig["direction"]

        # Find first LTF bar after HTF signal
        ltf_start = int(np.searchsorted(ts_ltf, htf_ts, side="left"))
        if ltf_start >= n_ltf:
            continue

        for j in range(ltf_start, min(ltf_start + max_wait, n_ltf)):
            if j in entries:
                continue
            atr_v   = atr_ltf[j] if not np.isnan(atr_ltf[j]) else cl_ltf[j]*0.01
            entry_p = cl_ltf[j]

            if entry_mode == "next_bar":
                sl   = entry_p - sl_atr*atr_v if direction=="LONG" else entry_p + sl_atr*atr_v
                dist = abs(entry_p - sl)
                if dist <= 0:
                    continue
                tp = entry_p + tp_r*dist if direction=="LONG" else entry_p - tp_r*dist
                entries[j] = {"direction": direction, "entry": entry_p,
                               "sl": sl, "tp": tp, "dist": dist}
                break

            elif entry_mode == "pullback_ema21":
                if np.isnan(ema21_ltf[j]):
                    continue
                ema = ema21_ltf[j]
                touched = False
                if direction == "LONG"  and lo_ltf[j] <= ema*1.002 and cl_ltf[j] > ema:
                    touched = True
                elif direction == "SHORT" and hi_ltf[j] >= ema*0.998 and cl_ltf[j] < ema:
                    touched = True
                if not touched:
                    continue
                sl   = entry_p - sl_atr*atr_v if direction=="LONG" else entry_p + sl_atr*atr_v
                dist = abs(entry_p - sl)
                if dist <= 0:
                    continue
                tp = entry_p + tp_r*dist if direction=="LONG" else entry_p - tp_r*dist
                entries[j] = {"direction": direction, "entry": entry_p,
                               "sl": sl, "tp": tp, "dist": dist}
                break

    return entries


# ── Portfolio simulation ──────────────────────────────────────────────────────

def _simulate(all_entries: dict[str, dict],
              all_ltf:     dict[str, dict],
              start_frac: float = 0.0,
              end_frac:   float = 1.0,
              use_maker:  bool  = False) -> dict:
    """
    all_entries: {sym: {ltf_bar: trade_dict}}  (merged across strategies)
    all_ltf:     {sym: dltf}
    """
    fee = MAKER_FEE if use_maker else TAKER_FEE

    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    trades  = []
    open_pos:    dict[str, bool] = {}     # sym -> True
    sym_cooldown: dict[str, int] = {}

    timeline = []
    for sym, entries in all_entries.items():
        dltf = all_ltf.get(sym)
        if dltf is None:
            continue
        ts_arr = dltf["ts"]
        n_ltf  = dltf["n"]
        i_start = int(n_ltf * start_frac)
        i_end   = int(n_ltf * end_frac)
        for bar, td in entries.items():
            if bar < i_start or bar >= i_end:
                continue
            timeline.append((int(ts_arr[bar]), sym, bar, td))

    timeline.sort(key=lambda x: x[0])

    for ts_entry, sym, bar, td in timeline:
        dltf = all_ltf[sym]
        cl = dltf["cl"]; hi = dltf["hi"]; lo = dltf["lo"]
        n  = dltf["n"]

        if sym in open_pos:
            continue
        if sym_cooldown.get(sym, 0) > ts_entry:
            continue
        if len(open_pos) >= MAX_POSITIONS:
            continue

        d      = td["direction"]
        entry  = td["entry"]
        sl     = td["sl"]
        tp     = td["tp"]
        dist   = td["dist"]

        risk_usd   = balance * RISK_PER_TRADE
        if dist <= 0 or risk_usd < 0.01:
            continue
        raw_qty    = risk_usd / dist
        raw_margin = raw_qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        if max_margin <= 0:
            continue
        lev = max(1, min(int(np.ceil(raw_margin / max_margin)), MAX_LEVERAGE))
        eff_margin = raw_margin / lev
        if eff_margin < MIN_MARGIN:
            continue

        open_pos[sym] = True

        exit_bar = None; exit_price = None; reason = "timeout"
        for j in range(bar + 1, n):
            if d == "LONG":
                if lo[j] <= sl: exit_price = sl; reason = "SL"; exit_bar = j; break
                if hi[j] >= tp: exit_price = tp; reason = "TP"; exit_bar = j; break
            else:
                if hi[j] >= sl: exit_price = sl; reason = "SL"; exit_bar = j; break
                if lo[j] <= tp: exit_price = tp; reason = "TP"; exit_bar = j; break

        del open_pos[sym]
        if exit_bar is None:
            continue

        gross = (exit_price - entry)*raw_qty if d=="LONG" else (entry - exit_price)*raw_qty
        fees  = (entry + exit_price) * raw_qty * (fee + SLIPPAGE)
        pnl   = gross - fees

        balance += pnl
        balance  = max(balance, 0.0)
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100.0 if peak > 0 else 0.0
        max_dd   = max(max_dd, dd)

        trades.append({
            "sym": sym, "direction": d, "entry": entry, "exit": exit_price,
            "reason": reason, "pnl": pnl, "win": pnl > 0,
        })

        if reason == "SL":
            sym_cooldown[sym] = ts_entry + SL_COOLDOWN_MS

    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "max_dd": 0.0,
                "final_balance": balance, "score": -999}

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gw  = sum(t["pnl"] for t in wins)
    gl  = abs(sum(t["pnl"] for t in losses))
    pf  = gw / gl if gl > 0 else float("inf")
    wr  = len(wins) / len(trades) * 100

    # Score: PF × WR/100 − DD/100  (balanced metric for grid search)
    score = pf * (wr / 100.0) - (max_dd / 100.0)

    return {
        "trades":        len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "pf":            round(pf, 2),
        "wr":            round(wr, 1),
        "max_dd":        round(max_dd, 1),
        "final_balance": round(balance, 2),
        "score":         round(score, 3),
    }


# ── Load all data ─────────────────────────────────────────────────────────────

def _load_all(symbols: list[str]) -> dict:
    """Returns {sym: {"1h": d, "4h": d|None, "5m": d, "15m": d}}"""
    out = {}
    loaded = 0
    for sym in symbols:
        d1h  = _prepare(_load_raw(sym, "1h"),  300)
        d4h  = _prepare(_load_raw(sym, "4h"),  100)
        d5m  = _prepare(_load_raw(sym, "5m"),  1000)
        d15m = _prepare(_load_raw(sym, "15m"), 500)
        if d1h is None:
            continue
        out[sym] = {"1h": d1h, "4h": d4h, "5m": d5m, "15m": d15m}
        loaded += 1
    logger.info("Loaded %d/%d symbols", loaded, len(symbols))
    return out


# ── Strategy runners ──────────────────────────────────────────────────────────

class StrategyRunner:
    """
    Encapsulates a single strategy config + TP/SL combo.
    Generates per-symbol entries and runs simulation.
    """

    def __init__(self, name: str, htf_tf: str, ltf_tf: str,
                 sig_fn: str, entry_mode: str, max_wait: int,
                 sig_params: dict, sl_atr: float, tp_r: float):
        self.name       = name
        self.htf_tf     = htf_tf
        self.ltf_tf     = ltf_tf
        self.sig_fn     = sig_fn         # "kc" or "bbs"
        self.entry_mode = entry_mode
        self.max_wait   = max_wait
        self.sig_params = sig_params
        self.sl_atr     = sl_atr
        self.tp_r       = tp_r

    def label(self) -> str:
        return f"{self.name}  SL={self.sl_atr}×ATR  TP={self.tp_r}R"

    def detect_all_signals(self, all_data: dict) -> tuple[dict, dict]:
        """
        Pre-compute HTF signals and collect LTF dicts.
        Returns ({sym: htf_sigs}, {sym: dltf})
        Call once per strategy group; SL/TP do not affect signal detection.
        """
        all_sigs: dict[str, dict] = {}
        all_ltf:  dict[str, dict] = {}
        for sym, data in all_data.items():
            dhtf = data.get(self.htf_tf)
            dltf = data.get(self.ltf_tf)
            if dhtf is None or dltf is None:
                continue
            if self.sig_fn == "kc":
                sigs = _detect_kc_signals(
                    dhtf,
                    min_sq   = self.sig_params["min_sq"],
                    adx_min  = self.sig_params["adx_min"],
                    vol_mult = self.sig_params.get("vol_mult", 1.0),
                    mom_bars = self.sig_params.get("mom_bars", 6),
                )
            else:
                sigs = _detect_bbs_signals(
                    dhtf,
                    min_sq        = self.sig_params["min_sq"],
                    adx_thresh    = self.sig_params["adx_thresh"],
                    vol_mult      = self.sig_params.get("vol_mult", 1.0),
                    use_ema_stack = self.sig_params.get("ema_stack", True),
                    headfake      = self.sig_params.get("headfake", True),
                )
            if sigs:
                all_sigs[sym] = sigs
                all_ltf[sym]  = dltf
        return all_sigs, all_ltf

    def build_entries_from_signals(self, all_sigs: dict, all_ltf: dict,
                                   all_data: dict) -> tuple[dict, dict]:
        """Build entries using pre-computed signals. Fast (only _match_ltf)."""
        all_entries: dict[str, dict] = {}
        for sym, htf_sigs in all_sigs.items():
            dhtf = all_data[sym].get(self.htf_tf)
            dltf = all_ltf.get(sym)
            if dhtf is None or dltf is None:
                continue
            entries = _match_ltf(
                htf_sigs, dhtf, dltf,
                entry_mode = self.entry_mode,
                max_wait   = self.max_wait,
                sl_atr     = self.sl_atr,
                tp_r       = self.tp_r,
            )
            if entries:
                all_entries[sym] = entries
        return all_entries, {sym: all_ltf[sym] for sym in all_entries}

    def build_entries(self, all_data: dict) -> tuple[dict, dict]:
        """Returns (all_entries, all_ltf) for this strategy config."""
        all_entries: dict[str, dict] = {}
        all_ltf:     dict[str, dict] = {}

        for sym, data in all_data.items():
            dhtf = data.get(self.htf_tf)
            dltf = data.get(self.ltf_tf)
            if dhtf is None or dltf is None:
                continue

            if self.sig_fn == "kc":
                htf_sigs = _detect_kc_signals(
                    dhtf,
                    min_sq    = self.sig_params["min_sq"],
                    adx_min   = self.sig_params["adx_min"],
                    vol_mult  = self.sig_params.get("vol_mult", 1.0),
                    mom_bars  = self.sig_params.get("mom_bars", 6),
                )
            else:
                htf_sigs = _detect_bbs_signals(
                    dhtf,
                    min_sq        = self.sig_params["min_sq"],
                    adx_thresh    = self.sig_params["adx_thresh"],
                    vol_mult      = self.sig_params.get("vol_mult", 1.0),
                    use_ema_stack = self.sig_params.get("ema_stack", True),
                    headfake      = self.sig_params.get("headfake", True),
                )

            entries = _match_ltf(
                htf_sigs, dhtf, dltf,
                entry_mode = self.entry_mode,
                max_wait   = self.max_wait,
                sl_atr     = self.sl_atr,
                tp_r       = self.tp_r,
            )
            if entries:
                all_entries[sym] = entries
                all_ltf[sym]     = dltf

        return all_entries, all_ltf


# ── Strategy definitions ──────────────────────────────────────────────────────

def _make_strategies() -> list[StrategyRunner]:
    """
    Base configs for each strategy. TP/SL grid applied separately.
    """
    strategies = []

    # ── 1. KC_4H_15M: 4h Keltner squeeze → 15m next_bar ─────────────────────
    # v10 "Swing KC 4h→15m" showed PF=5.68, WF PF=6.44 on 365d
    # Expectation: fewer trades (~2-4/month) but high quality
    for sl, tp in product(SL_MULTS, TP_RS):
        strategies.append(StrategyRunner(
            name       = "KC_4H_15M",
            htf_tf     = "4h",
            ltf_tf     = "15m",
            sig_fn     = "kc",
            entry_mode = "next_bar",
            max_wait   = 1,
            sig_params = {
                "min_sq":   8,       # 8 bars × 4h = 32h squeeze
                "adx_min":  15.0,    # was 12 in v10; 15 is safer
                "vol_mult": 1.0,
                "mom_bars": 6,       # 6 bars = 24h momentum slope
            },
            sl_atr = sl, tp_r = tp,
        ))

    # ── 2. BBS_4H_5M: 4h BB squeeze → 5m pullback_ema21 ─────────────────────
    # Conservative cross-TF: big picture signal, precise 5m entry
    # Expectation: very selective (~1-3/month), high WR
    for sl, tp in product(SL_MULTS, TP_RS):
        strategies.append(StrategyRunner(
            name       = "BBS_4H_5M",
            htf_tf     = "4h",
            ltf_tf     = "5m",
            sig_fn     = "bbs",
            entry_mode = "pullback_ema21",
            max_wait   = 72,          # 72 × 5m = 6h — longer wait for 4h signal
            sig_params = {
                "min_sq":    6,       # 6 bars × 4h = 24h squeeze (was 8 in 1h)
                "adx_thresh":15.0,
                "vol_mult":  1.0,
                "ema_stack": True,    # strict: price must be above/below all EMAs
                "headfake":  True,
            },
            sl_atr = sl, tp_r = tp,
        ))

    # ── 3. BBS_1H_15M_PB: 1h BB squeeze → 15m pullback_ema21 ────────────────
    # vs current BBS_1H_15M which enters on next_bar (worse entry, no pullback)
    # Expectation: same signal quality as BBS_1H_15M but ~30-50% fewer fills
    # (only enters on pullback) → higher WR and PF
    for sl, tp in product(SL_MULTS, TP_RS):
        strategies.append(StrategyRunner(
            name       = "BBS_1H_15M_PB",
            htf_tf     = "1h",
            ltf_tf     = "15m",
            sig_fn     = "bbs",
            entry_mode = "pullback_ema21",
            max_wait   = 24,          # 24 × 15m = 6h wait for pullback
            sig_params = {
                "min_sq":    4,       # same as live BBS_1H_15M
                "adx_thresh":15.0,
                "vol_mult":  1.0,
                "ema_stack": True,
                "headfake":  True,
            },
            sl_atr = sl, tp_r = tp,
        ))

    return strategies


# ── Grid search ───────────────────────────────────────────────────────────────

def _run_grid(strategies: list[StrategyRunner], all_data: dict) -> None:
    """
    For each strategy:
    1. Run all SL/TP combos on IS (80%)
    2. Show top-5 configs by score
    3. Validate best on OOS (20%)
    """

    # Check how many symbols have 4h data
    n_4h = sum(1 for d in all_data.values() if d.get("4h") is not None)
    n_1h = sum(1 for d in all_data.values() if d.get("1h") is not None)
    n_5m = sum(1 for d in all_data.values() if d.get("5m") is not None)
    n_15m= sum(1 for d in all_data.values() if d.get("15m") is not None)

    # Group strategies by name
    by_name: dict[str, list[StrategyRunner]] = defaultdict(list)
    for s in strategies:
        by_name[s.name].append(s)

    best_overall: list[tuple] = []  # (name, best_runner, is_res, oos_res)

    for strat_name, runners in by_name.items():
        print(f"\n{'='*80}")
        print(f"  STRATEGY: {strat_name}  ({len(runners)} SL/TP combos)")

        htf_tf = runners[0].htf_tf
        ltf_tf = runners[0].ltf_tf
        n_htf  = n_4h if htf_tf == "4h" else n_1h
        n_ltf  = n_5m if ltf_tf == "5m" else n_15m

        print(f"  HTF={htf_tf} ({n_htf} syms with data)  "
              f"LTF={ltf_tf} ({n_ltf} syms with data)")

        if n_htf == 0:
            print(f"  WARNING: NO {htf_tf} DATA — run fetch_data_2y.py first!")
            print(f"{'='*80}")
            continue

        # ── Pre-compute signals once for all configs ─────────────────────────
        t_sig = _time.time()
        base_runner = runners[0]  # all same sig_fn + sig_params
        cached_sigs, cached_ltf = base_runner.detect_all_signals(all_data)
        n_sigs = sum(len(s) for s in cached_sigs.values())
        print(f"  Signals detected: {n_sigs} across {len(cached_sigs)} symbols "
              f"in {_time.time()-t_sig:.1f}s")

        if n_sigs == 0:
            print(f"  No signals detected — skip")
            continue

        # ── IS scan (only _match_ltf varies per SL/TP) ─────────────────────
        is_results = []
        for runner in runners:
            entries, ltf = runner.build_entries_from_signals(
                cached_sigs, cached_ltf, all_data)
            if not entries:
                continue
            res = _simulate(entries, ltf, 0.0, 0.80,
                            use_maker=(runner.entry_mode == "next_bar"))
            if res["trades"] < 10:   # not enough trades to trust
                continue
            is_results.append((runner, res, entries, ltf))

        if not is_results:
            print(f"  No valid configs (all < 10 trades on IS)")
            continue

        # Sort by score, show top-5
        is_results.sort(key=lambda x: x[1]["score"], reverse=True)

        print(f"\n  IS results (80% data)  —  top configs by score:")
        print(f"  {'Config':<40}  {'Tr':>4}  {'WR%':>5}  {'PF':>5}  {'DD%':>5}  "
              f"{'Scr':>5}  {'Verdict'}")
        print(f"  {'-'*80}")

        top5_pass = []
        for runner, res, entries, ltf in is_results[:10]:
            sl_tp = f"SL={runner.sl_atr}×ATR  TP={runner.tp_r}R"
            v = _verdict_is(res)
            flag = "✓" if v == "OK" else "✗"
            print(f"  {strat_name} {sl_tp:<26}  {res['trades']:>4}  "
                  f"{res['wr']:>4.1f}%  {res['pf']:>5.2f}  {res['max_dd']:>4.1f}%  "
                  f"{res['score']:>5.3f}  {flag} {v}")
            if v == "OK":
                top5_pass.append((runner, res, entries, ltf))

        # ── OOS validation of best passing config ───────────────────────────
        if not top5_pass:
            print(f"\n  ✗ No config passed IS criteria (PF≥2.0, WR≥60%, DD≤35%)")
            # Still validate the best config by score
            best_runner, best_is, best_entries, best_ltf = is_results[0]
        else:
            best_runner, best_is, best_entries, best_ltf = top5_pass[0]

        oos = _simulate(best_entries, best_ltf, 0.80, 1.0,
                        use_maker=(best_runner.entry_mode == "next_bar"))

        print(f"\n  Best config: SL={best_runner.sl_atr}×ATR  TP={best_runner.tp_r}R")
        print(f"  IS  (80%): trades={best_is['trades']}  WR={best_is['wr']}%  "
              f"PF={best_is['pf']}  DD={best_is['max_dd']}%  score={best_is['score']}")
        print(f"  OOS (20%): trades={oos['trades']}  WR={oos['wr']}%  "
              f"PF={oos['pf']}  DD={oos['max_dd']}%")

        oos_verdict = _verdict_oos(best_is, oos)
        print(f"  OOS verdict: {oos_verdict}")

        best_overall.append((strat_name, best_runner, best_is, oos))

    # ── Combined portfolio ───────────────────────────────────────────────────
    _run_combo(best_overall, all_data)


def _verdict_is(r: dict) -> str:
    if r["trades"] < 10:
        return "too few trades"
    if r["pf"] < 2.0:
        return f"PF↓{r['pf']:.2f}"
    if r["wr"] < 60.0:
        return f"WR↓{r['wr']:.1f}%"
    if r["max_dd"] > 35.0:
        return f"DD↑{r['max_dd']:.1f}%"
    return "OK"


def _verdict_oos(is_r: dict, oos_r: dict) -> str:
    if oos_r["trades"] < 5:
        return "⚠️  too few OOS trades"
    degradation_pf = is_r["pf"] - oos_r["pf"]
    if oos_r["pf"] < 1.3:
        return f"✗ OOS PF too low ({oos_r['pf']:.2f}) — overfit likely"
    if oos_r["pf"] < 1.8:
        return f"~ OOS PF marginal ({oos_r['pf']:.2f}) — borderline"
    if degradation_pf > 1.0:
        return f"~ PF degraded IS→OOS by {degradation_pf:.2f} — possible overfit"
    return f"✓ OOS PF={oos_r['pf']:.2f} — solid"


def _run_combo(best_overall: list, all_data: dict) -> None:
    """
    Run combined portfolio of all strategies with their best configs.
    Each strategy contributes its entries; portfolio simulation handles
    position limits and cooldowns.
    """
    if not best_overall:
        return

    print(f"\n{'='*80}")
    print(f"  COMBINED PORTFOLIO  ({len(best_overall)} strategies, best configs)")
    print(f"  Balance: ${START_BALANCE}  Risk: {RISK_PER_TRADE*100:.0f}%/trade  "
          f"MaxPos: {MAX_POSITIONS}")
    print(f"{'='*80}")

    # Merge all entries — sym -> merged list; since different strategies
    # may use different LTF, we need to handle conflicts.
    # Strategy: each strategy has its own namespace (sym_strat key), but
    # the portfolio sim uses sym-level cooldowns.
    # We'll merge by giving each (sym, strat_name) a unique key, but
    # flattening to sym for cooldown purposes.

    # Actually, we'll run a multi-strategy combined simulation manually.
    _combo_simulate(best_overall, all_data)


def _combo_simulate(best_overall: list, all_data: dict) -> None:
    """Multi-strategy portfolio simulation."""
    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    trades  = []
    open_pos:    set = set()      # (sym, strat_name)
    sym_cooldown: dict[str, int] = {}

    # Build timeline: (ts, sym, strat_name, bar, td, dltf)
    timeline = []
    for strat_name, runner, is_res, oos_res in best_overall:
        entries, ltf = runner.build_entries(all_data)
        for sym, sym_entries in entries.items():
            dltf = ltf.get(sym)
            if dltf is None:
                continue
            ts_arr = dltf["ts"]
            for bar, td in sym_entries.items():
                timeline.append((int(ts_arr[bar]), sym, strat_name, bar, td, dltf))

    timeline.sort(key=lambda x: x[0])

    strat_pnl: dict[str, float]   = defaultdict(float)
    strat_wins: dict[str, int]    = defaultdict(int)
    strat_losses: dict[str, int]  = defaultdict(int)

    for ts_entry, sym, strat_name, bar, td, dltf in timeline:
        key = (sym, strat_name)
        if key in open_pos:
            continue
        if sym_cooldown.get(sym, 0) > ts_entry:
            continue
        if len(open_pos) >= MAX_POSITIONS:
            continue

        cl = dltf["cl"]; hi = dltf["hi"]; lo = dltf["lo"]
        n  = dltf["n"]

        d     = td["direction"]
        entry = td["entry"]; sl = td["sl"]; tp = td["tp"]; dist = td["dist"]

        risk_usd   = balance * RISK_PER_TRADE
        if dist <= 0 or risk_usd < 0.01:
            continue
        raw_qty    = risk_usd / dist
        raw_margin = raw_qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        if max_margin <= 0:
            continue
        lev = max(1, min(int(np.ceil(raw_margin / max_margin)), MAX_LEVERAGE))
        eff_margin = raw_margin / lev
        if eff_margin < MIN_MARGIN:
            continue

        open_pos.add(key)

        exit_bar = None; exit_price = None; reason = "timeout"
        for j in range(bar + 1, n):
            if d == "LONG":
                if lo[j] <= sl: exit_price = sl; reason = "SL"; exit_bar = j; break
                if hi[j] >= tp: exit_price = tp; reason = "TP"; exit_bar = j; break
            else:
                if hi[j] >= sl: exit_price = sl; reason = "SL"; exit_bar = j; break
                if lo[j] <= tp: exit_price = tp; reason = "TP"; exit_bar = j; break

        open_pos.discard(key)
        if exit_bar is None:
            continue

        fee_rate = MAKER_FEE + SLIPPAGE  # next_bar = maker; pullback = taker+slip
        gross = (exit_price-entry)*raw_qty if d=="LONG" else (entry-exit_price)*raw_qty
        fees  = (entry + exit_price) * raw_qty * fee_rate
        pnl   = gross - fees

        balance += pnl
        balance  = max(balance, 0.0)
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100.0 if peak > 0 else 0.0
        max_dd   = max(max_dd, dd)

        trades.append({
            "sym": sym, "strat": strat_name, "direction": d,
            "reason": reason, "pnl": pnl, "win": pnl > 0,
        })

        if pnl > 0:
            strat_wins[strat_name] += 1
        else:
            strat_losses[strat_name] += 1
        strat_pnl[strat_name] += pnl

        if reason == "SL":
            sym_cooldown[sym] = ts_entry + SL_COOLDOWN_MS

    if not trades:
        print("  No trades in combined portfolio.")
        return

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gw  = sum(t["pnl"] for t in wins)
    gl  = abs(sum(t["pnl"] for t in losses))
    pf  = gw / gl if gl > 0 else float("inf")
    wr  = len(wins) / len(trades) * 100

    print(f"\n  FULL period (all data):")
    print(f"  Trades={len(trades)}  WR={wr:.1f}%  PF={pf:.2f}  "
          f"MaxDD={max_dd:.1f}%  Balance: ${START_BALANCE}→${balance:.0f}")

    print(f"\n  Per-strategy breakdown:")
    print(f"  {'Strategy':<20}  {'Trades':>6}  {'WR%':>5}  {'PF':>5}")
    print(f"  {'-'*48}")
    for strat_name, _, _, _ in best_overall:
        n_w  = strat_wins.get(strat_name, 0)
        n_l  = strat_losses.get(strat_name, 0)
        n_t  = n_w + n_l
        if n_t == 0:
            continue
        wr_s = n_w / n_t * 100
        gw_s = sum(t["pnl"] for t in trades if t["strat"]==strat_name and t["win"])
        gl_s = abs(sum(t["pnl"] for t in trades if t["strat"]==strat_name and not t["win"]))
        pf_s = gw_s / gl_s if gl_s > 0 else float("inf")
        print(f"  {strat_name:<20}  {n_t:>6}  {wr_s:>4.1f}%  {pf_s:>5.2f}")

    # IS/OOS comparison
    is_res = _combo_simulate_range(best_overall, all_data, 0.0, 0.80)
    oos_res= _combo_simulate_range(best_overall, all_data, 0.80, 1.0)
    if is_res and oos_res:
        print(f"\n  Walk-forward check:")
        print(f"  IS  (80%): trades={is_res['trades']}  WR={is_res['wr']}%  "
              f"PF={is_res['pf']}  DD={is_res['max_dd']}%")
        print(f"  OOS (20%): trades={oos_res['trades']}  WR={oos_res['wr']}%  "
              f"PF={oos_res['pf']}  DD={oos_res['max_dd']}%")

        if oos_res["pf"] >= 1.8 and oos_res["trades"] >= 10:
            print(f"  → ✓ OOS PF={oos_res['pf']:.2f} — portfolio validates")
        elif oos_res["pf"] >= 1.3:
            print(f"  → ~ OOS PF={oos_res['pf']:.2f} — marginal, monitor closely")
        else:
            print(f"  → ✗ OOS PF={oos_res['pf']:.2f} — does not validate, likely overfit")


def _combo_simulate_range(best_overall: list, all_data: dict,
                           sf: float, ef: float) -> dict | None:
    """Helper: run combo portfolio on a specific time slice."""
    all_e: dict[str, dict] = {}
    all_l: dict[str, dict] = {}

    for strat_name, runner, _, _ in best_overall:
        entries, ltf = runner.build_entries(all_data)
        for sym, sym_entries in entries.items():
            # Merge entries; later strategy overwrites same bar (rare)
            if sym not in all_e:
                all_e[sym] = {}
                all_l[sym] = ltf.get(sym)
            all_e[sym].update(sym_entries)

    if not all_e:
        return None
    return _simulate(all_e, all_l, sf, ef)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = _time.time()

    print("\n" + "="*80)
    print(f"  NEW STRATEGIES RESEARCH  |  TP/SL grid  |  {len(SYMBOLS)} symbols")
    print(f"  Balance: ${START_BALANCE}  Risk: {RISK_PER_TRADE*100:.0f}%/trade  "
          f"Fee: taker={TAKER_FEE*100:.3f}%  maker={MAKER_FEE*100:.3f}%")
    print(f"  IS=80% / OOS=20%  |  Min IS trades for inclusion: 10")
    print(f"")
    print(f"  Strategies:")
    print(f"  1. KC_4H_15M      — 4h Keltner squeeze → 15m next_bar  (v10 validated PF≈5.7)")
    print(f"  2. BBS_4H_5M      — 4h BB squeeze → 5m pullback EMA21  (conservative cross-TF)")
    print(f"  3. BBS_1H_15M_PB  — 1h BB squeeze → 15m pullback EMA21 (vs current next_bar)")
    print(f"")
    print(f"  TP/SL grid: SL={SL_MULTS}×ATR  TP={TP_RS}R  = {len(SL_MULTS)*len(TP_RS)} combos each")
    print(f"")
    print(f"  WARNINGS:")
    print(f"  - 4h data may be missing if fetch_data_2y.py hasn't finished yet")
    print(f"  - Survivorship bias: TOP_50 = current top, not historical composition")
    print(f"  - $500/1% selected: $100/2% blocks too many trades via min_margin=$5")
    print("="*80)

    logger.info("Loading data for %d symbols...", len(SYMBOLS))
    t_load = _time.time()
    all_data = _load_all(SYMBOLS)
    logger.info("Data loaded in %.1fs", _time.time() - t_load)

    if not all_data:
        print("  ERROR: No data found. Run fetch_data_2y.py first.")
        return

    n_4h = sum(1 for d in all_data.values() if d.get("4h") is not None)
    if n_4h == 0:
        print(f"\n  ⚠️  WARNING: 0 symbols have 4h data!")
        print(f"  KC_4H_15M and BBS_4H_5M will be skipped.")
        print(f"  Run: py -3 multi_agent/backtests/fetch_data_2y.py")
        print(f"  Then re-run this script.\n")
    else:
        print(f"\n  Data: {len(all_data)} symbols loaded  |  4h data: {n_4h} symbols")

    strategies = _make_strategies()
    n_per = len(SL_MULTS) * len(TP_RS)
    n_strats = 3
    print(f"  Running {n_strats} strategies × {n_per} configs = {n_strats*n_per} total IS runs")

    _run_grid(strategies, all_data)

    print(f"\n{'='*80}")
    print(f"  Total runtime: {_time.time()-t0:.1f}s")
    print("="*80)


if __name__ == "__main__":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    main()
