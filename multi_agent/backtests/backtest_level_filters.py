"""
backtest_level_filters.py  --  Level-proximity filter research
===============================================================

Исследует два фильтра для BBS_1H_15M стратегии:

  Variant A (proximity filter):
    Пропускает сигнал если цена в пределах N×ATR от недавнего swing high/low.
    "Первый тест уровня" → опасно → skip.

  Variant B (second-test):
    Когда Variant A отфильтровал сигнал → ждём отскок (rejection ≥ R×ATR)
    затем возврат к уровню → входим (второй тест уровня).

Режимы:
  baseline  — без фильтров (текущая система)
  varA      — только proximity filter
  varB      — только second-test (без pre-filter, все сигналы + watcher)
  varAB     — A фильтрует, B ловит отфильтрованные на ретесте

Grid search:
  Variant A: proximity_mult ∈ [1.0, 1.5, 2.0, 2.5, 3.0]
             lookback ∈ [20, 30, 50]  (1H баров назад)
  Variant B: reject_dist ∈ [0.75, 1.0, 1.5]  (×ATR — минимальный отскок)
             return_zone ∈ [0.3, 0.5]         (×ATR — зона возврата к уровню)
             max_wait    ∈ [24, 48]            (1H баров ожидания)

Реалистичные условия:
  - Maker fee 0.020% вход (limit/next_bar)
  - Taker fee 0.055% выход (SL/TP market)
  - Slippage 0.05% вход + 0.05% выход
  - Break-even при +1.5R (SL в entry)
  - 4h cooldown после SL
  - Max 4 одновременных позиций
  - Min margin $5 (как на Bybit)

Данные: 180d parquet из data_cache/
"""
from __future__ import annotations
import logging
import sys
import time as _time
from pathlib import Path
from itertools import product

# Fix Windows console/file encoding for Cyrillic output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators,
    detect_squeeze_signal,
    check_head_fake,
)

logger = logging.getLogger("lvl_filter")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-7s | %(message)s")
logger.setLevel(logging.INFO)

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── Портфельные параметры ─────────────────────────────────────────────────────
START_BALANCE   = 500.0      # $500 — не блокирует min_margin
RISK_PER_TRADE  = 0.02       # 2% (как в живом боте)
MAKER_FEE       = 0.00020    # 0.020% — вход limit/post-only
TAKER_FEE       = 0.00055    # 0.055% — выход по SL/TP
SLIPPAGE_IN     = 0.0005     # 0.05% slippage при входе
SLIPPAGE_OUT    = 0.0005     # 0.05% slippage при выходе (пессимистично)
MIN_MARGIN      = 5.0        # Bybit минимум
MAX_LEVERAGE    = 50
MAX_MARGIN_FRAC = 0.05       # 5% баланса макс маржа
MAX_POSITIONS   = 4
SL_COOLDOWN_MS  = 4 * 3_600_000   # 4ч кулдаун
BE_TRIGGER_R    = 1.5        # break-even при +1.5R

# BBS_1H_15M параметры (live config)
BBS_PARAMS = {
    "min_squeeze": 4, "adx_thresh": 15.0, "vol_mult": 1.0,
    "headfake": True, "sl_atr": 1.5, "tp_r": 2.5,
    "max_wait_15m": 3,   # максимум 3 бара 15m ждём fill (45 мин)
}

# ── Символы (те у кого есть 180d данные) ────────────────────────────────────
SYMBOLS = [
    "BTC/USDT:USDT",  "ETH/USDT:USDT",  "BNB/USDT:USDT",  "SOL/USDT:USDT",
    "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT",  "LINK/USDT:USDT", "UNI/USDT:USDT",  "ATOM/USDT:USDT",
    "NEAR/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",  "OP/USDT:USDT",
    "INJ/USDT:USDT",  "SUI/USDT:USDT",  "1000PEPE/USDT:USDT",
    "HBAR/USDT:USDT", "POL/USDT:USDT",
]

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _load(symbol: str, tf: str) -> pd.DataFrame | None:
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in ("730d", "365d", "180d", "90d"):
        for ext in (".parquet", ".csv"):
            p = CACHE_DIR / f"{safe}_{tf}_{tag}{ext}"
            if p.exists():
                try:
                    df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
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


# ─────────────────────────────────────────────────────────────────────────────
# INDICATOR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    s = pd.Series(arr)
    result = s.ewm(span=p, adjust=False).mean().to_numpy()
    out[p - 1:] = result[p - 1:]
    return out

def _sma(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.nancumsum(arr)
    out[p - 1:] = (cs[p - 1:] - np.concatenate(([0.0], cs[:-p]))) / p
    return out

def _stdev(arr: np.ndarray, p: int) -> np.ndarray:
    s = pd.Series(arr)
    return s.rolling(p).std(ddof=0).to_numpy()

def _atr(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, p: int = 14) -> np.ndarray:
    n = len(cl)
    tr = np.zeros(n)
    tr[0] = hi[0] - lo[0]
    for i in range(1, n):
        tr[i] = max(hi[i] - lo[i], abs(hi[i] - cl[i-1]), abs(lo[i] - cl[i-1]))
    out = np.full(n, np.nan)
    if n < p:
        return out
    out[p - 1] = np.mean(tr[:p])
    for i in range(p, n):
        out[i] = out[i-1] * (p-1)/p + tr[i]/p
    return out

def _adx(hi: np.ndarray, lo: np.ndarray, cl: np.ndarray, p: int = 14) -> np.ndarray:
    n = len(cl)
    adx = np.full(n, 20.0)
    pdm = np.zeros(n); ndm = np.zeros(n); tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(hi[i]-lo[i], abs(hi[i]-cl[i-1]), abs(lo[i]-cl[i-1]))
        up = hi[i]-hi[i-1]; dn = lo[i-1]-lo[i]
        pdm[i] = up if up > dn and up > 0 else 0
        ndm[i] = dn if dn > up and dn > 0 else 0
    if n < p * 3:
        return adx
    atr_s = np.mean(tr[1:p+1])
    pdm_s = np.mean(pdm[1:p+1])
    ndm_s = np.mean(ndm[1:p+1])
    dx_buf = []
    for i in range(p, n):
        atr_s = atr_s - atr_s/p + tr[i]
        pdm_s = pdm_s - pdm_s/p + pdm[i]
        ndm_s = ndm_s - ndm_s/p + ndm[i]
        pdi = 100*pdm_s/atr_s if atr_s > 0 else 0
        ndi = 100*ndm_s/atr_s if atr_s > 0 else 0
        s = pdi + ndi
        dx = abs(pdi-ndi)/s*100 if s > 0 else 0
        dx_buf.append(dx)
        if len(dx_buf) == p:
            adx[i] = np.mean(dx_buf)
        elif len(dx_buf) > p:
            adx[i] = (adx[i-1]*(p-1)+dx)/p
    return adx


def prepare(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 60:
        return None
    cl = df["close"].to_numpy(float)
    hi = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    vo = df["volume"].to_numpy(float)
    ts = df["timestamp"].to_numpy(int)
    sma20 = _sma(cl, 20); std20 = _stdev(cl, 20)
    vol_ma = _sma(vo, 20)
    kc_atr = _atr(hi, lo, cl, 10)
    return {
        "cl": cl, "hi": hi, "lo": lo, "ts": ts, "n": len(cl),
        "atr": _atr(hi, lo, cl, 14),
        "adx": _adx(hi, lo, cl, 14),
        "ema21": _ema(cl, 21), "ema50": _ema(cl, 50), "ema200": _ema(cl, 200),
        "bb_upper": sma20 + 2*std20, "bb_lower": sma20 - 2*std20,
        "kc_mid": _ema(cl, 20),
        "kc_atr": kc_atr,
        "vol_ratio": np.where(vol_ma > 0, vo / vol_ma, 1.0),
        "df": df,
    }


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT A: PROXIMITY FILTER
# ─────────────────────────────────────────────────────────────────────────────

def is_near_swing_level(d1h: dict, bar_idx: int, direction: str,
                        proximity_mult: float, lookback: int) -> tuple[bool, float]:
    """
    Возвращает (True, dist_atr) если сигнал слишком близко к swing уровню.
    dist_atr = расстояние до уровня в единицах ATR (меньше = опаснее).
    """
    hi = d1h["hi"]; lo = d1h["lo"]; atr = d1h["atr"]
    cl = d1h["cl"]

    atr_val = atr[bar_idx] if not np.isnan(atr[bar_idx]) else cl[bar_idx] * 0.01
    if atr_val <= 0:
        return False, 999.0

    start = max(0, bar_idx - lookback)
    window_hi = hi[start:bar_idx]   # НЕ включаем текущий бар (no look-ahead)
    window_lo = lo[start:bar_idx]

    if len(window_hi) < 5:
        return False, 999.0

    signal_price = cl[bar_idx]

    if direction == "SHORT":
        recent_high = float(np.max(window_hi))
        dist = recent_high - signal_price         # >0 = цена ниже уровня
        dist_atr = dist / atr_val
        too_close = dist < proximity_mult * atr_val
        return too_close, dist_atr
    else:  # LONG
        recent_low = float(np.min(window_lo))
        dist = signal_price - recent_low          # >0 = цена выше уровня
        dist_atr = dist / atr_val
        too_close = dist < proximity_mult * atr_val
        return too_close, dist_atr


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT C: OBSTACLE-AHEAD FILTER (percentage-based, global levels)
# ─────────────────────────────────────────────────────────────────────────────

def is_obstacle_ahead(d1h: dict, bar_idx: int, direction: str,
                      level_pct: float, level_bars: int) -> bool:
    """
    Возвращает True если по ходу сделки есть "препятствие" — глобальный уровень,
    который остановит движение цены.

    LONG:  skip если ближайший swing high ВЫШЕ текущей цены ≤ level_pct%
           (resistance ahead — цена уткнётся в потолок)
    SHORT: skip если ближайший swing low НИЖЕ текущей цены ≤ level_pct%
           (support ahead — цена упрётся в пол)

    level_bars — глобальный lookback (50-500 1H баров = 2-21 дней).
    level_pct  — порог в долях (0.003 = 0.3%).
    """
    cl    = d1h["cl"]
    hi    = d1h["hi"]
    lo    = d1h["lo"]
    close = cl[bar_idx]
    if close <= 0:
        return False

    start = max(0, bar_idx - level_bars)

    if direction == "LONG":
        # Resistance выше текущей цены (блокирует рост)
        highs = hi[start:bar_idx]
        highs_above = highs[highs > close]
        if len(highs_above) == 0:
            return False  # нет resistance выше — путь свободен
        nearest_resistance = float(np.min(highs_above))
        gap_pct = (nearest_resistance - close) / close
        return gap_pct <= level_pct
    else:  # SHORT
        # Support ниже текущей цены (блокирует падение)
        lows = lo[start:bar_idx]
        lows_below = lows[lows < close]
        if len(lows_below) == 0:
            return False  # нет support ниже — путь свободен
        nearest_support = float(np.max(lows_below))
        gap_pct = (close - nearest_support) / close
        return gap_pct <= level_pct


# ─────────────────────────────────────────────────────────────────────────────
# VARIANT B: SECOND-TEST TRACKER
# ─────────────────────────────────────────────────────────────────────────────

def find_second_test_entry(d1h: dict, signal_bar: int, direction: str,
                           level_price: float,
                           reject_dist: float, return_zone: float,
                           max_wait: int) -> int | None:
    """
    После фильтрованного сигнала на баре signal_bar:
    1. Ищет rejection: цена уходит от level_price на ≥ reject_dist × ATR
    2. Затем возврат: цена возвращается в пределах return_zone × ATR от level_price
    Возвращает индекс 1H бара (second-test entry signal), или None.
    """
    cl = d1h["cl"]; hi = d1h["hi"]; lo = d1h["lo"]; atr = d1h["atr"]
    n  = d1h["n"]

    atr_val = atr[signal_bar] if not np.isnan(atr[signal_bar]) else level_price * 0.01
    if atr_val <= 0:
        return None

    reject_threshold  = reject_dist * atr_val
    return_threshold  = return_zone * atr_val
    end_bar = min(signal_bar + max_wait, n - 1)

    # Phase 1: rejection (цена уходит от уровня в благоприятную сторону)
    rejected = False
    reject_bar = signal_bar
    for j in range(signal_bar + 1, end_bar + 1):
        if direction == "SHORT":
            # Для SHORT: rejection = цена ушла ВНИЗ от уровня resistance
            if cl[j] < level_price - reject_threshold:
                rejected = True; reject_bar = j; break
        else:  # LONG
            # Для LONG: rejection = цена ушла ВВЕРХ от уровня support
            if cl[j] > level_price + reject_threshold:
                rejected = True; reject_bar = j; break

    if not rejected:
        return None

    # Phase 2: return to level (возврат в зону уровня)
    end_bar2 = min(signal_bar + max_wait * 2, n - 1)
    for j in range(reject_bar + 1, end_bar2 + 1):
        if direction == "SHORT":
            # HIGH бара достигает зоны resistance
            if hi[j] >= level_price - return_threshold:
                return j
        else:  # LONG
            # LOW бара достигает зоны support
            if lo[j] <= level_price + return_threshold:
                return j

    return None


# ─────────────────────────────────────────────────────────────────────────────
# SIGNAL DETECTION (BBS 1H)
# ─────────────────────────────────────────────────────────────────────────────

def detect_bbs_signals(d1h: dict, params: dict,
                       mode: str = "baseline",
                       prox_mult: float = 2.0,
                       prox_lookback: int = 30,
                       reject_dist: float = 1.0,
                       return_zone: float = 0.4,
                       max_wait_b: int = 36,
                       use_ema_stack: bool = True,
                       obs_pct: float = 0.005,
                       obs_bars: int = 100) -> dict:
    """
    Возвращает {bar_idx: {direction, entry_price, sl_price, source}}
    source: "normal" | "second_test"

    mode:
      "baseline" — без фильтров
      "varA"     — proximity filter (skip near levels)
      "varB"     — second-test (все сигналы + watcher for level signals)
      "varAB"    — A filters → B catches on retest
    """
    sq_ind = None
    try:
        sq_ind = compute_squeeze_indicators(d1h["df"])
    except Exception as e:
        return {}
    if sq_ind is None:
        return {}

    cl = d1h["cl"]; hi = d1h["hi"]; lo = d1h["lo"]
    n  = d1h["n"]
    adx = d1h["adx"]; vol_r = d1h["vol_ratio"]
    ema21 = d1h["ema21"]; ema50 = d1h["ema50"]; ema200 = d1h["ema200"]

    min_sq = params["min_squeeze"]
    adx_th = params["adx_thresh"]
    vol_m  = params["vol_mult"]
    use_hf = params["headfake"]

    signals = {}
    # Watchers for Variant B: {bar_idx: {direction, level, atr_val}}
    watchers = {}

    warmup = max(250, prox_lookback + 10, obs_bars + 10)

    for i in range(warmup, n - 1):
        # ── Базовая проверка сигнала ───────────────────────────────────────
        if np.isnan(adx[i]) or adx[i] < adx_th:
            continue
        if vol_r[i] < vol_m:
            continue

        sig = detect_squeeze_signal(sq_ind, bar_idx=i,
                                    min_squeeze_bars=min_sq,
                                    adx_threshold=adx_th,
                                    vol_mult=vol_m)
        if sig is None:
            continue
        if use_hf and check_head_fake(sq_ind, i, 1):
            continue

        d = sig["direction"]

        # EMA stack filter (как в live BBS, опционально)
        if use_ema_stack:
            if d == "LONG":
                if not (cl[i] > ema21[i] > ema50[i] > ema200[i]):
                    continue
            else:
                if not (cl[i] < ema21[i] < ema50[i] < ema200[i]):
                    continue

        atr_v = d1h["atr"][i]
        if np.isnan(atr_v) or atr_v <= 0:
            continue

        sl = (cl[i] - 1.5*atr_v) if d == "LONG" else (cl[i] + 1.5*atr_v)

        # ── Proximity check (Variant A / AB) ──────────────────────────────
        near_level = False
        level_price = cl[i]

        if mode in ("varA", "varAB"):
            near_level, dist_atr = is_near_swing_level(d1h, i, d, prox_mult, prox_lookback)
            if near_level:
                # Сохранить уровень для Variant B
                level_hi = d1h["hi"][max(0, i - prox_lookback):i]
                level_lo = d1h["lo"][max(0, i - prox_lookback):i]
                if len(level_hi) > 0:
                    level_price = float(np.max(level_hi)) if d == "SHORT" else float(np.min(level_lo))
                if mode == "varAB":
                    watchers[i] = {"direction": d, "level": level_price, "atr_val": atr_v}
                continue   # Пропускаем сигнал (Variant A работает)

        elif mode == "varC":
            # Obstacle-ahead filter: препятствие по ходу сделки (процентный, глобальные уровни)
            if is_obstacle_ahead(d1h, i, d, obs_pct, obs_bars):
                continue   # Глобальный уровень блокирует движение — пропускаем

        elif mode == "varB":
            # Для varB: все нормальные сигналы входят, но для near-level добавляем watcher
            near_level, dist_atr = is_near_swing_level(d1h, i, d, prox_mult, prox_lookback)
            if near_level:
                level_hi = d1h["hi"][max(0, i - prox_lookback):i]
                level_lo = d1h["lo"][max(0, i - prox_lookback):i]
                if len(level_hi) > 0:
                    level_price = float(np.max(level_hi)) if d == "SHORT" else float(np.min(level_lo))
                watchers[i] = {"direction": d, "level": level_price, "atr_val": atr_v}
                # НЕ пропускаем — обычный сигнал тоже добавляем

        # Добавляем нормальный сигнал
        if i not in signals:
            signals[i] = {
                "direction": d, "entry_price": cl[i], "sl_price": sl, "source": "normal"
            }

    # ── Variant B: обработка watchers ─────────────────────────────────────
    if mode in ("varB", "varAB") and watchers:
        for wbar, w in watchers.items():
            st_bar = find_second_test_entry(
                d1h, wbar, w["direction"], w["level"],
                reject_dist=reject_dist,
                return_zone=return_zone,
                max_wait=max_wait_b,
            )
            if st_bar is None:
                continue
            if st_bar >= n - 1:
                continue
            # Не перезаписываем существующий нормальный сигнал на том же баре
            if st_bar in signals:
                continue
            atr_v2 = d1h["atr"][st_bar]
            if np.isnan(atr_v2) or atr_v2 <= 0:
                continue
            d2 = w["direction"]
            sl2 = (cl[st_bar] - 1.5*atr_v2) if d2=="LONG" else (cl[st_bar] + 1.5*atr_v2)
            signals[st_bar] = {
                "direction": d2, "entry_price": cl[st_bar],
                "sl_price": sl2, "source": "second_test"
            }

    return signals


# ─────────────────────────────────────────────────────────────────────────────
# LTF ENTRY MATCHING (1H signal → 15M next_bar)
# ─────────────────────────────────────────────────────────────────────────────

def match_15m_entries(sigs_1h: dict, d1h: dict, d15m: dict,
                      sl_atr: float = 1.5, tp_r: float = 2.5,
                      max_wait: int = 3) -> dict:
    """
    Возвращает {ltf_bar_idx: trade_dict} с реальными entry/sl/tp на 15m.
    Входим на open следующего 15m бара после 1H сигнала.
    """
    if d15m is None or not sigs_1h:
        return {}

    ts_1h  = d1h["ts"]
    ts_15m = d15m["ts"]
    cl_15m = d15m["cl"]; hi_15m = d15m["hi"]; lo_15m = d15m["lo"]
    atr_15m = d15m["atr"]
    n_15m = d15m["n"]

    entries = {}
    for htf_bar, sig in sigs_1h.items():
        htf_ts  = ts_1h[htf_bar]
        direction = sig["direction"]

        # Найти первый 15m бар ПОСЛЕ HTF сигнала
        ltf_start = int(np.searchsorted(ts_15m, htf_ts, side="right"))
        if ltf_start >= n_15m:
            continue

        # Входим на OPEN следующего бара (next_bar semantics)
        # max_wait = 3 → ждём максимум 3 бара (45 мин)
        for j in range(ltf_start, min(ltf_start + max_wait, n_15m)):
            if j in entries:
                continue
            # Вход по open бара + slippage (pessimistic)
            open_j = d15m["df"]["open"].iloc[j] if hasattr(d15m, "df") else cl_15m[j]
            # Fallback: используем close как приближение open
            entry_p = cl_15m[j]

            atr_v = atr_15m[j] if not np.isnan(atr_15m[j]) else entry_p * 0.01

            # Slippage на вход (пессимистично: хуже цена)
            if direction == "LONG":
                entry_p_slip = entry_p * (1 + SLIPPAGE_IN)
                sl  = entry_p_slip - sl_atr * atr_v
                dist = abs(entry_p_slip - sl)
                if dist <= 0:
                    continue
                tp  = entry_p_slip + tp_r * dist
            else:
                entry_p_slip = entry_p * (1 - SLIPPAGE_IN)
                sl  = entry_p_slip + sl_atr * atr_v
                dist = abs(entry_p_slip - sl)
                if dist <= 0:
                    continue
                tp  = entry_p_slip - tp_r * dist

            entries[j] = {
                "direction": direction,
                "entry":     entry_p_slip,
                "sl":        sl, "tp": tp, "dist": dist,
                "source":    sig.get("source", "normal"),
            }
            break

    return entries


# ─────────────────────────────────────────────────────────────────────────────
# PORTFOLIO SIMULATION
# ─────────────────────────────────────────────────────────────────────────────

def run_sim(all_entries: dict[str, dict],
            all_d15m:   dict[str, dict],
            start_frac: float = 0.0,
            end_frac:   float = 1.0) -> dict:
    """
    all_entries: {sym: {ltf_bar: trade_dict}}
    all_d15m:    {sym: d15m_dict}

    Возвращает статистику: trades, PF, WR, DD, PnL, balance.
    """
    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    trades  = []
    open_pos  = {}                 # sym -> pos_dict
    sym_cool  = {}                 # sym -> cooldown_until_ms

    # Строим единую временную шкалу
    timeline = []
    for sym, entries in all_entries.items():
        d15m = all_d15m.get(sym)
        if d15m is None:
            continue
        ts_arr = d15m["ts"]; n = d15m["n"]
        i_start = int(n * start_frac)
        i_end   = int(n * end_frac)
        for bar, td in entries.items():
            if bar < i_start or bar >= i_end:
                continue
            timeline.append((int(ts_arr[bar]), sym, bar, td))

    timeline.sort(key=lambda x: x[0])

    for ts_entry, sym, bar, td in timeline:
        d15m = all_d15m[sym]
        cl   = d15m["cl"]; hi = d15m["hi"]; lo = d15m["lo"]
        n    = d15m["n"]

        # Пропускаем если символ уже в позиции
        if sym in open_pos:
            continue
        # Кулдаун после SL
        if sym_cool.get(sym, 0) > ts_entry:
            continue
        # Лимит позиций
        if len(open_pos) >= MAX_POSITIONS:
            continue

        d     = td["direction"]
        entry = td["entry"]
        sl    = td["sl"]
        tp    = td["tp"]
        dist  = td["dist"]

        # Position sizing
        risk_usd  = balance * RISK_PER_TRADE
        if dist <= 0 or risk_usd < 0.01:
            continue
        qty       = risk_usd / dist
        raw_margin = qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        lev = 1
        if raw_margin > max_margin and max_margin > 0:
            lev = min(int(np.ceil(raw_margin / max_margin)), MAX_LEVERAGE)
        eff_margin = raw_margin / max(lev, 1)
        if eff_margin < MIN_MARGIN:
            continue

        open_pos[sym] = {
            "bar": bar, "d": d, "qty": qty,
            "entry": entry, "sl": sl, "tp": tp, "dist": dist,
            "be_triggered": False, "source": td.get("source", "normal"),
        }

        # Симуляция бар за баром
        pos        = open_pos[sym]
        exit_bar   = None
        exit_price = None
        reason     = "timeout"

        for j in range(bar + 1, n):
            h_j = hi[j]; l_j = lo[j]; c_j = cl[j]
            cur_sl = pos["sl"]
            cur_tp = pos["tp"]

            if d == "LONG":
                # Break-even check (до проверки SL/TP)
                favorable_move = h_j - entry
                if not pos["be_triggered"] and favorable_move >= BE_TRIGGER_R * dist:
                    pos["sl"] = entry * 1.001  # SL в entry+0.1%
                    pos["be_triggered"] = True
                    cur_sl = pos["sl"]

                # Пессимистично: SL проверяем первым
                if l_j <= cur_sl:
                    # SL пробит — выход по SL с slippage
                    exit_price = cur_sl * (1 - SLIPPAGE_OUT)
                    reason = "SL"; exit_bar = j; break
                if h_j >= cur_tp:
                    exit_price = cur_tp * (1 - SLIPPAGE_OUT * 0.5)  # меньше slippage на TP
                    reason = "TP"; exit_bar = j; break

            else:  # SHORT
                favorable_move = entry - l_j
                if not pos["be_triggered"] and favorable_move >= BE_TRIGGER_R * dist:
                    pos["sl"] = entry * 0.999
                    pos["be_triggered"] = True
                    cur_sl = pos["sl"]

                if h_j >= cur_sl:
                    exit_price = cur_sl * (1 + SLIPPAGE_OUT)
                    reason = "SL"; exit_bar = j; break
                if l_j <= cur_tp:
                    exit_price = cur_tp * (1 + SLIPPAGE_OUT * 0.5)
                    reason = "TP"; exit_bar = j; break

        if exit_bar is None:
            open_pos.pop(sym, None)
            continue

        # PnL с реалистичными комиссиями
        gross = (exit_price - entry) * qty if d == "LONG" else (entry - exit_price) * qty
        entry_fee = entry * qty * MAKER_FEE          # лимитный вход
        exit_fee  = exit_price * qty * TAKER_FEE     # рыночный выход
        pnl = gross - entry_fee - exit_fee

        balance += pnl
        balance  = max(balance, 0.01)
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100.0 if peak > 0 else 0.0
        max_dd   = max(max_dd, dd)

        win = pnl > 0
        trades.append({
            "sym": sym, "direction": d,
            "entry": entry, "exit": exit_price, "reason": reason,
            "pnl_usd": round(pnl, 4),
            "pnl_r": round(pnl / (risk_usd if risk_usd > 0 else 1), 2),
            "win": win, "source": td.get("source", "normal"),
        })

        if reason == "SL":
            sym_cool[sym] = int(d15m["ts"][exit_bar]) + SL_COOLDOWN_MS

        open_pos.pop(sym, None)

    # Статистика
    n_t   = len(trades)
    wins  = [t for t in trades if t["win"]]
    loses = [t for t in trades if not t["win"]]
    gross_w = sum(t["pnl_usd"] for t in wins)
    gross_l = abs(sum(t["pnl_usd"] for t in loses))
    pf    = gross_w / gross_l if gross_l > 0 else float("inf")
    wr    = len(wins) / n_t * 100 if n_t > 0 else 0.0
    total = sum(t["pnl_usd"] for t in trades)
    avg_r = np.mean([t["pnl_r"] for t in trades]) if trades else 0.0

    n_normal = sum(1 for t in trades if t.get("source") == "normal")
    n_second = sum(1 for t in trades if t.get("source") == "second_test")

    return {
        "trades": n_t, "wins": len(wins), "losses": len(loses),
        "wr": round(wr, 1), "pf": round(pf, 2),
        "total_pnl": round(total, 2), "avg_r": round(avg_r, 2),
        "max_dd": round(max_dd, 1),
        "final_bal": round(balance, 2),
        "n_normal": n_normal, "n_second": n_second,
        "trade_list": trades,
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────────────────────

def load_all_data() -> dict[str, dict]:
    """Загружает 1H и 15M данные для всех символов."""
    data = {}
    loaded = 0
    for sym in SYMBOLS:
        df1h  = _load(sym, "1h")
        df15m = _load(sym, "15m")
        if df1h is None or df15m is None:
            logger.debug("Skip %s — no data", sym)
            continue
        d1h  = prepare(df1h)
        d15m = prepare(df15m)
        if d1h is None or d15m is None:
            continue
        data[sym] = {"d1h": d1h, "d15m": d15m}
        loaded += 1
    logger.info("Loaded %d symbols", loaded)
    return data


def run_config(data: dict, mode: str,
               prox_mult: float = 2.0, prox_lookback: int = 30,
               reject_dist: float = 1.0, return_zone: float = 0.4,
               max_wait_b: int = 36,
               is_frac: float = 0.8,
               use_ema_stack: bool = True,
               obs_pct: float = 0.005,
               obs_bars: int = 100) -> tuple[dict, dict]:
    """
    Запускает один конфиг и возвращает (IS_stats, OOS_stats).
    """
    all_entries_15m: dict[str, dict] = {}
    all_d15m: dict[str, dict] = {}

    for sym, d in data.items():
        d1h  = d["d1h"]
        d15m = d["d15m"]

        sigs_1h = detect_bbs_signals(
            d1h, BBS_PARAMS, mode=mode,
            prox_mult=prox_mult, prox_lookback=prox_lookback,
            reject_dist=reject_dist, return_zone=return_zone,
            max_wait_b=max_wait_b,
            use_ema_stack=use_ema_stack,
            obs_pct=obs_pct, obs_bars=obs_bars,
        )
        if not sigs_1h:
            continue

        entries_15m = match_15m_entries(
            sigs_1h, d1h, d15m,
            sl_atr=BBS_PARAMS["sl_atr"], tp_r=BBS_PARAMS["tp_r"],
            max_wait=BBS_PARAMS["max_wait_15m"],
        )
        if not entries_15m:
            continue

        all_entries_15m[sym] = entries_15m
        all_d15m[sym] = d15m

    is_res  = run_sim(all_entries_15m, all_d15m, 0.0, is_frac)
    oos_res = run_sim(all_entries_15m, all_d15m, is_frac, 1.0)
    return is_res, oos_res


def count_filtered(data: dict, prox_mults: list, prox_lookback: int = 20) -> None:
    """Диагностика: сколько сигналов фильтрует каждый порог prox_mult."""
    # Считаем базовые сигналы и их dist_atr для каждого символа
    all_dist_atrs: list[float] = []
    total_base = 0
    for sym, d in data.items():
        d1h = d["d1h"]
        cl = d1h["cl"]; hi = d1h["hi"]; lo = d1h["lo"]
        atr = d1h["atr"]; n = d1h["n"]
        ema21 = d1h["ema21"]; ema50 = d1h["ema50"]; ema200 = d1h["ema200"]
        vol_r = d1h["vol_ratio"]; adx = d1h["adx"]
        sq_ind = None
        try:
            sq_ind = compute_squeeze_indicators(d1h["df"])
        except Exception:
            continue
        if sq_ind is None:
            continue
        warmup = max(250, prox_lookback + 10)
        for i in range(warmup, n - 1):
            if np.isnan(adx[i]) or adx[i] < BBS_PARAMS["adx_thresh"]:
                continue
            if vol_r[i] < BBS_PARAMS["vol_mult"]:
                continue
            sig = detect_squeeze_signal(sq_ind, bar_idx=i,
                                        min_squeeze_bars=BBS_PARAMS["min_squeeze"],
                                        adx_threshold=BBS_PARAMS["adx_thresh"],
                                        vol_mult=BBS_PARAMS["vol_mult"])
            if sig is None:
                continue
            if BBS_PARAMS["headfake"] and check_head_fake(sq_ind, i, 1):
                continue
            d_dir = sig["direction"]
            if d_dir == "LONG":
                if not (cl[i] > ema21[i] > ema50[i] > ema200[i]):
                    continue
            else:
                if not (cl[i] < ema21[i] < ema50[i] < ema200[i]):
                    continue
            atr_v = atr[i]
            if np.isnan(atr_v) or atr_v <= 0:
                continue
            total_base += 1
            _, dist_atr = is_near_swing_level(d1h, i, d_dir, 999.0, prox_lookback)
            all_dist_atrs.append(dist_atr)

    if not all_dist_atrs:
        return
    arr = np.array(all_dist_atrs)
    print(f"\n  ДИАГНОСТИКА proximity (lookback={prox_lookback}h, базовых сигналов={total_base}):")
    print(f"  dist_atr: min={arr.min():.2f}  p10={np.percentile(arr,10):.2f}  "
          f"p25={np.percentile(arr,25):.2f}  median={np.median(arr):.2f}  "
          f"p75={np.percentile(arr,75):.2f}  p90={np.percentile(arr,90):.2f}  max={arr.max():.2f}")
    print(f"  prox_mult | filtered | filtered%")
    for pm in prox_mults:
        n_filt = int(np.sum(arr < pm))
        print(f"    {pm:>5.1f}× |  {n_filt:>6}  |  {n_filt/total_base*100:>5.1f}%")


def fmt_row(label: str, is_r: dict, oos_r: dict) -> str:
    return (f"  {label:<28} "
            f"IS:  T={is_r['trades']:>3}  WR={is_r['wr']:>5.1f}%  PF={is_r['pf']:>5.2f}  "
            f"DD={is_r['max_dd']:>4.1f}%  PnL=${is_r['total_pnl']:>7.2f}  "
            f"avgR={is_r['avg_r']:>+5.2f} | "
            f"OOS: T={oos_r['trades']:>3}  WR={oos_r['wr']:>5.1f}%  PF={oos_r['pf']:>5.2f}  "
            f"DD={oos_r['max_dd']:>4.1f}%  PnL=${oos_r['total_pnl']:>7.2f}  "
            f"avgR={oos_r['avg_r']:>+5.2f}")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--varc-only", action="store_true",
                    help="Skip varA/B/AB sections, run only baseline + varC")
    args, _ = ap.parse_known_args()
    varc_only = args.varc_only

    t0 = _time.time()
    logger.info("=== Backtest Level Filters v2 ===")
    logger.info("Loading data...")
    data = load_all_data()
    if not data:
        logger.error("No data found in %s", CACHE_DIR)
        return

    SEP = "=" * 150
    print(f"\n{SEP}")
    print(f"  BACKTEST LEVEL FILTERS v2 --- BBS_1H_15M")
    print(f"  Data: {len(data)} symbols | Balance: ${START_BALANCE} | Risk: {RISK_PER_TRADE*100:.0f}%/trade")
    print(f"  Fees: maker {MAKER_FEE*100:.3f}% in / taker {TAKER_FEE*100:.3f}% out + slip {SLIPPAGE_IN*100:.2f}%/{SLIPPAGE_OUT*100:.2f}%")
    print(f"  Break-even @ {BE_TRIGGER_R}R | Cooldown 4h | Max {MAX_POSITIONS} positions")
    print(f"  IS=first 80% / OOS=last 20%")
    print(f"{SEP}\n")

    # ── 0. BASELINE сравнение: с EMA-стеком vs без (объясняет разницу с v12) ──
    print("== 0. BASELINE сравнение (EMA-stack ON vs OFF)")
    is_base, oos_base = run_config(data, "baseline", use_ema_stack=True)
    print(fmt_row("baseline (EMA-stack ON)", is_base, oos_base))
    is_base_ne, oos_base_ne = run_config(data, "baseline", use_ema_stack=False)
    print(fmt_row("baseline (EMA-stack OFF)", is_base_ne, oos_base_ne))
    print()

    best_a = None; best_ab = None; best_b = None

    if not varc_only:
        # ── 1. ДИАГНОСТИКА proximity ──────────────────────────────────────────
        all_prox_mults = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        count_filtered(data, all_prox_mults, prox_lookback=20)
        count_filtered(data, all_prox_mults, prox_lookback=30)

        # ── 2. VARIANT A — расширенный грид ──────────────────────────────────
        print(f"\n== 1. VARIANT A (proximity filter) --- расширенный грид")
        prox_mults_a = [3.0, 4.0, 5.0, 6.0, 8.0]
        prox_lookbacks_a = [15, 20, 30]

        best_a_oos_pf = 0.0
        for pm, lb in product(prox_mults_a, prox_lookbacks_a):
            label = f"A prox={pm:.1f}x look={lb}h"
            is_r, oos_r = run_config(data, "varA", prox_mult=pm, prox_lookback=lb, use_ema_stack=True)
            n_filt = is_base["trades"] - is_r["trades"]
            filt_pct = n_filt / is_base["trades"] * 100 if is_base["trades"] > 0 else 0
            marker = " [BEST]" if oos_r["pf"] > best_a_oos_pf and oos_r["trades"] >= 8 else ""
            if oos_r["pf"] > best_a_oos_pf and oos_r["trades"] >= 8:
                best_a_oos_pf = oos_r["pf"]
                best_a = {"prox_mult": pm, "prox_lookback": lb, "is": is_r, "oos": oos_r,
                          "filt_pct": filt_pct}
            print(fmt_row(label + f" (filt {filt_pct:.0f}%)", is_r, oos_r) + marker)

        if best_a:
            print(f"\n  Best-A: prox_mult={best_a['prox_mult']:.1f}x look={best_a['prox_lookback']}h "
                  f"-> OOS PF={best_a['oos']['pf']:.2f} (фильтрует {best_a['filt_pct']:.0f}% IS-сделок)")

        # ── 3. VARIANT B — second-test ────────────────────────────────────────
        print(f"\n== 2. VARIANT B (second-test) --- grid search")
        best_pm = best_a["prox_mult"] if best_a else 4.0
        best_lb = best_a["prox_lookback"] if best_a else 20

        reject_dists  = [0.5, 0.75, 1.0, 1.5]
        return_zones  = [0.25, 0.4, 0.6]
        max_waits_b   = [12, 24, 48]

        best_b_oos_pf = 0.0
        for rd, rz, mw in product(reject_dists, return_zones, max_waits_b):
            label = f"B rej={rd:.2f}x ret={rz:.2f}x wait={mw}h"
            is_r, oos_r = run_config(data, "varB",
                                     prox_mult=best_pm, prox_lookback=best_lb,
                                     reject_dist=rd, return_zone=rz, max_wait_b=mw,
                                     use_ema_stack=True)
            n_2nd = is_r.get("n_second", 0)
            marker = " [BEST]" if oos_r["pf"] > best_b_oos_pf and oos_r["trades"] >= 8 else ""
            if oos_r["pf"] > best_b_oos_pf and oos_r["trades"] >= 8:
                best_b_oos_pf = oos_r["pf"]
                best_b = {"reject_dist": rd, "return_zone": rz, "max_wait_b": mw,
                          "is": is_r, "oos": oos_r}
            print(fmt_row(label + f" 2nd={n_2nd}", is_r, oos_r) + marker)

        # ── 4. VARIANT A+B ────────────────────────────────────────────────────
        print(f"\n== 3. VARIANT A+B (A фильтрует -> B ловит второй тест)")
        best_rd = best_b["reject_dist"] if best_b else 0.75
        best_rz = best_b["return_zone"]  if best_b else 0.4
        best_mw = best_b["max_wait_b"]   if best_b else 24

        best_ab_oos_pf = 0.0; best_ab_dd = 999.0
        for pm in prox_mults_a:
            label = f"AB prox={pm:.1f}x rej={best_rd:.2f}x ret={best_rz:.2f}"
            is_r, oos_r = run_config(data, "varAB",
                                     prox_mult=pm, prox_lookback=best_lb,
                                     reject_dist=best_rd, return_zone=best_rz, max_wait_b=best_mw,
                                     use_ema_stack=True)
            n_2nd = is_r.get("n_second", 0)
            score = oos_r["pf"] - oos_r["max_dd"] * 0.05
            marker = " [BEST]" if score > (best_ab_oos_pf - best_ab_dd * 0.05) and oos_r["trades"] >= 8 else ""
            if score > (best_ab_oos_pf - best_ab_dd * 0.05) and oos_r["trades"] >= 8:
                best_ab_oos_pf = oos_r["pf"]
                best_ab_dd = oos_r["max_dd"]
                best_ab = {"prox_mult": pm, "prox_lookback": best_lb,
                           "reject_dist": best_rd, "return_zone": best_rz, "max_wait_b": best_mw,
                           "is": is_r, "oos": oos_r}
            print(fmt_row(label + f" 2nd={n_2nd}", is_r, oos_r) + marker)

    # ── 4. VARIANT C (obstacle-ahead filter, percentage-based, глобальные уровни) ──
    print(f"\n== 4. VARIANT C (obstacle-ahead — % proximity к глобальному уровню)")
    print(f"  LONG: skip если swing high ≤ X% выше entry | SHORT: skip если swing low ≤ X% ниже")
    obs_pcts  = [0.002, 0.003, 0.005, 0.010, 0.015, 0.020]
    obs_barss = [50, 100, 200, 500]

    best_c = None; best_c_oos_pf = 0.0

    for op, ob in product(obs_pcts, obs_barss):
        label = f"C pct={op*100:.1f}% bars={ob}h"
        is_r, oos_r = run_config(data, "varC", obs_pct=op, obs_bars=ob, use_ema_stack=True)
        filt_is  = is_base["trades"] - is_r["trades"]
        filt_pct = filt_is / is_base["trades"] * 100 if is_base["trades"] > 0 else 0
        marker = " [BEST]" if oos_r["pf"] > best_c_oos_pf and oos_r["trades"] >= 8 else ""
        if oos_r["pf"] > best_c_oos_pf and oos_r["trades"] >= 8:
            best_c_oos_pf = oos_r["pf"]
            best_c = {"obs_pct": op, "obs_bars": ob, "is": is_r, "oos": oos_r,
                      "filt_pct": filt_pct}
        print(fmt_row(label + f" (filt {filt_pct:.0f}%)", is_r, oos_r) + marker)

    if best_c:
        print(f"\n  Best-C: pct={best_c['obs_pct']*100:.1f}% bars={best_c['obs_bars']}h "
              f"-> OOS PF={best_c['oos']['pf']:.2f} WR={best_c['oos']['wr']:.1f}% "
              f"DD={best_c['oos']['max_dd']:.1f}% (фильтрует {best_c['filt_pct']:.0f}% IS-сделок)")

    # ── 5. ИТОГОВОЕ СРАВНЕНИЕ ─────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  ИТОГОВОЕ СРАВНЕНИЕ (лучшие конфиги)")
    print(f"{SEP}")

    rows = [
        ("BASELINE (EMA ON)", is_base, oos_base),
        ("BASELINE (EMA OFF, v12-style)", is_base_ne, oos_base_ne),
    ]
    if best_a:
        rows.append((f"Var-A (prox={best_a['prox_mult']:.1f}x look={best_a['prox_lookback']}h)",
                     best_a["is"], best_a["oos"]))
    if best_b:
        rows.append((f"Var-B (rej={best_b['reject_dist']:.2f}x ret={best_b['return_zone']:.2f}x)",
                     best_b["is"], best_b["oos"]))
    if best_ab:
        rows.append((f"Var-AB (prox={best_ab['prox_mult']:.1f}x rej={best_ab['reject_dist']:.2f}x)",
                     best_ab["is"], best_ab["oos"]))
    if best_c:
        rows.append((f"Var-C (pct={best_c['obs_pct']*100:.1f}% bars={best_c['obs_bars']}h)",
                     best_c["is"], best_c["oos"]))

    for label, isr, oosr in rows:
        print(fmt_row(label, isr, oosr))

    # IS vs OOS деградация
    print(f"\n  {'Config':<38}  IS->OOS PF          IS->OOS WR          IS->OOS DD")
    for label, isr, oosr in rows:
        pf_deg = oosr["pf"] / isr["pf"] * 100 if isr["pf"] > 0 else 0
        wr_deg = oosr["wr"] / isr["wr"] * 100 if isr["wr"] > 0 else 0
        print(f"  {label:<38}  {isr['pf']:.2f}->{oosr['pf']:.2f} ({pf_deg:.0f}%)  "
              f"{isr['wr']:.1f}%->{oosr['wr']:.1f}% ({wr_deg:.0f}%)  "
              f"DD {isr['max_dd']:.1f}%->{oosr['max_dd']:.1f}%")

    elapsed = _time.time() - t0
    print(f"\n  Время выполнения: {elapsed:.1f}s")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()
