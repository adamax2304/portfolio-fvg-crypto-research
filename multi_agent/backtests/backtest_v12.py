"""
backtest_v12.py  --  Live Config Validation (2026-04-20)

Цель: проверить текущий live конфиг на 44 символах, $100 стартовый баланс.
      NO optimization — просто запуск текущих параметров как есть.

Стратегии (точно как в config.py):
  BBS_1H_5M:  min_sq=8,  adx=15, vol=1.0, pullback_ema21, wait=20, sl=1.5×ATR, tp=3.0R
  BBS_1H_15M: min_sq=4,  adx=15, vol=1.0, next_bar,       sl=1.5×ATR, tp=2.5R, daily_ema200=OFF
  KC_1H_15M:  min_sq=10, adx=12, vol=1.0, mom=6,  next_bar, sl=1.5×ATR, tp=2.0R

Risk: 2% депо за трейд, fee=0.055%, slippage=0.05%
Walk-forward: 80% in-sample / 20% out-of-sample (последние ~73 дня)
"""
from __future__ import annotations
import logging, sys, time as _time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    check_head_fake,
)

logger = logging.getLogger("v12")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

CACHE_DIR = Path(__file__).parent / "data_cache"

# ── Параметры ────────────────────────────────────────────────────────────────
START_BALANCE = 100.0
RISK_PER_TRADE = 0.02        # 2% депо
TAKER_FEE     = 0.00055      # 0.055%
SLIPPAGE      = 0.0005       # 0.05%
MIN_MARGIN    = 5.0          # $5 мин маржа (как на Bybit)
MAX_LEVERAGE  = 50
MAX_MARGIN_FRAC = 0.05       # 5% баланса макс маржа на трейд
MAX_POSITIONS = 4            # макс одновременных позиций
SL_COOLDOWN_BARS_1H = 4      # 4 часа кулдаун после SL

STRATEGIES = {
    "BBS_1H_5M": {
        "type": "bbs", "entry_tf": "5m", "entry_mode": "pullback_ema21",
        "min_squeeze": 8, "adx_thresh": 15.0, "vol_mult": 1.0,
        "headfake": True, "daily_ema200": True,
        "max_wait": 20, "sl_atr": 1.5, "tp_r": 3.0,
    },
    "BBS_1H_15M": {
        "type": "bbs", "entry_tf": "15m", "entry_mode": "next_bar",
        "min_squeeze": 4, "adx_thresh": 15.0, "vol_mult": 1.0,
        "headfake": True, "daily_ema200": False,
        "max_wait": 1, "sl_atr": 1.5, "tp_r": 2.5,
    },
    "KC_1H_15M": {
        "type": "kc", "entry_tf": "15m", "entry_mode": "next_bar",
        "min_squeeze": 10, "adx_min": 12.0, "vol_mult": 1.0,
        "mom_bars": 6, "daily_ema200": False,
        "max_wait": 1, "sl_atr": 1.5, "tp_r": 2.0,
    },
}

# Символы — все у кого есть 1h+5m+15m за 365д
SYMBOLS_365D = [
    "1000PEPE/USDT:USDT", "ADA/USDT:USDT",  "ALGO/USDT:USDT", "APT/USDT:USDT",
    "ARB/USDT:USDT",       "ATOM/USDT:USDT", "AVAX/USDT:USDT", "BCH/USDT:USDT",
    "BNB/USDT:USDT",       "BTC/USDT:USDT",  "CRV/USDT:USDT",  "DASH/USDT:USDT",
    "DEXE/USDT:USDT",      "DOGE/USDT:USDT", "DOT/USDT:USDT",  "ETH/USDT:USDT",
    "FIL/USDT:USDT",       "HBAR/USDT:USDT", "ICP/USDT:USDT",  "INJ/USDT:USDT",
    "LINK/USDT:USDT",      "NEAR/USDT:USDT", "OP/USDT:USDT",   "POL/USDT:USDT",
    "SEI/USDT:USDT",       "SOL/USDT:USDT",  "SUI/USDT:USDT",  "TON/USDT:USDT",
    "TRX/USDT:USDT",       "UNI/USDT:USDT",  "XAUT/USDT:USDT", "XLM/USDT:USDT",
    "XMR/USDT:USDT",       "XRP/USDT:USDT",  "ONDO/USDT:USDT", "MNT/USDT:USDT",
]

# ── Data loading ─────────────────────────────────────────────────────────────

def _load(symbol: str, tf: str) -> pd.DataFrame | None:
    safe = symbol.replace("/", "_").replace(":", "_")
    for tag in ("730d", "365d", "180d"):
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

# ── Indicator helpers ─────────────────────────────────────────────────────────

def _ema(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    if len(arr) < p:
        return out
    k = 2.0 / (p + 1)
    out[p - 1] = np.nanmean(arr[:p])
    for i in range(p, len(arr)):
        if not np.isnan(out[i - 1]):
            out[i] = arr[i] * k + out[i - 1] * (1 - k)
    return out

def _sma(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    cs = np.nancumsum(arr)
    out[p - 1:] = (cs[p - 1:] - np.concatenate(([0.0], cs[:-p]))) / p
    return out

def _stdev(arr: np.ndarray, p: int) -> np.ndarray:
    out = np.full(len(arr), np.nan)
    for i in range(p - 1, len(arr)):
        out[i] = np.std(arr[i - p + 1:i + 1])
    return out

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
    atr_s = np.mean(tr[1:p+1]); pdm_s = np.mean(pdm[1:p+1]); ndm_s = np.mean(ndm[1:p+1])
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
        if len(dx_buf) >= p:
            adx[i] = np.mean(dx_buf[-p:]) if len(dx_buf) == p else (adx[i-1]*(p-1)+dx)/p
    return adx

def prepare(df: pd.DataFrame) -> dict | None:
    if df is None or len(df) < 60:
        return None
    cl = df["close"].to_numpy(dtype=float)
    hi = df["high"].to_numpy(dtype=float)
    lo = df["low"].to_numpy(dtype=float)
    vo = df["volume"].to_numpy(dtype=float)
    ts = df["timestamp"].to_numpy(dtype=int)
    sma20 = _sma(cl, 20); std20 = _stdev(cl, 20)
    vol_ma = _sma(vo, 20)
    return {
        "cl": cl, "hi": hi, "lo": lo, "ts": ts, "n": len(cl),
        "atr": _atr(hi, lo, cl, 14),
        "adx": _adx(hi, lo, cl, 14),
        "ema21": _ema(cl, 21),
        "ema50": _ema(cl, 50),
        "ema200": _ema(cl, 200),
        "bb_upper": sma20 + 2*std20, "bb_lower": sma20 - 2*std20, "bb_mid": sma20,
        "kc_mid": _ema(cl, 20),
        "kc_atr": _atr(hi, lo, cl, 10),
        "vol_ratio": np.where(vol_ma > 0, vo / vol_ma, 1.0),
        "df": df,
    }

# ── Signal detection ──────────────────────────────────────────────────────────

def detect_bbs_signals_1h(d1h: dict, params: dict) -> dict:
    """Returns {bar_idx: {direction, sl_price}}"""
    min_sq  = params["min_squeeze"]
    adx_th  = params["adx_thresh"]
    vol_m   = params["vol_mult"]
    use_hf  = params["headfake"]

    try:
        sq_ind = compute_squeeze_indicators(d1h["df"])
    except Exception:
        return {}
    if sq_ind is None:
        return {}

    cl = d1h["cl"]; hi = d1h["hi"]; lo = d1h["lo"]
    n = d1h["n"]
    adx = d1h["adx"]; vol_r = d1h["vol_ratio"]
    ema21 = d1h["ema21"]; ema50 = d1h["ema50"]; ema200 = d1h["ema200"]

    sigs = {}
    warmup = max(250, int(n * 0.04))
    for i in range(warmup, n - 1):  # -1: confirmed closed bar = i, last bar = i+1
        if np.isnan(adx[i]) or adx[i] < adx_th:
            continue
        if vol_r[i] < vol_m:
            continue
        sig = detect_squeeze_signal(sq_ind, bar_idx=i, min_squeeze_bars=min_sq,
                                    adx_threshold=adx_th, vol_mult=vol_m)
        if sig is None:
            continue
        if use_hf and check_head_fake(sq_ind, i, 1):
            continue
        d = sig["direction"]
        # EMA stack filter (как в live)
        if d == "LONG":
            if not (cl[i] > ema21[i] > ema50[i] > ema200[i]):
                continue
        else:
            if not (cl[i] < ema21[i] < ema50[i] < ema200[i]):
                continue
        atr_v = d1h["atr"][i] if not np.isnan(d1h["atr"][i]) else cl[i]*0.01
        sl = (cl[i] - 1.5*atr_v) if d == "LONG" else (cl[i] + 1.5*atr_v)
        sigs[i] = {"direction": d, "entry_price": cl[i], "sl_price": sl}
    return sigs


def detect_kc_signals_1h(d1h: dict, params: dict) -> dict:
    """Returns {bar_idx: {direction, sl_price}}"""
    min_sq   = params["min_squeeze"]
    adx_min  = params["adx_min"]
    vol_m    = params["vol_mult"]
    mom_bars = params["mom_bars"]

    cl = d1h["cl"]; hi = d1h["hi"]; lo = d1h["lo"]
    n = d1h["n"]
    adx = d1h["adx"]; vol_r = d1h["vol_ratio"]
    bb_u = d1h["bb_upper"]; bb_l = d1h["bb_lower"]
    kc_mid = d1h["kc_mid"]; kc_atr = d1h["kc_atr"]

    # KC bands
    kc_u = kc_mid + 1.5 * kc_atr
    kc_l = kc_mid - 1.5 * kc_atr

    # Squeeze: BB inside KC
    sq = np.zeros(n, dtype=bool)
    sq_len = np.zeros(n, dtype=int)
    for i in range(20, n):
        if not np.isnan(bb_u[i]) and not np.isnan(kc_u[i]):
            sq[i] = (bb_l[i] > kc_l[i]) and (bb_u[i] < kc_u[i])
    for i in range(1, n):
        sq_len[i] = sq_len[i-1] + 1 if sq[i] else 0

    sigs = {}
    warmup = max(250, mom_bars + 10)
    for i in range(warmup, n - 1):
        if sq[i]:              # still in squeeze
            continue
        if sq_len[i-1] < min_sq:
            continue
        if np.isnan(adx[i]) or adx[i] < adx_min:
            continue
        if vol_r[i] < vol_m:
            continue
        seg = cl[i-mom_bars:i+1]
        if len(seg) < mom_bars or np.any(np.isnan(seg)):
            continue
        slope = np.polyfit(np.arange(len(seg)), seg, 1)[0]
        atr_v = d1h["atr"][i] if not np.isnan(d1h["atr"][i]) else cl[i]*0.01
        if slope > 0:
            if np.isnan(bb_u[i]) or bb_u[i] <= kc_u[i]:
                continue
            d = "LONG"; sl = cl[i] - 1.5*atr_v
        else:
            if np.isnan(bb_l[i]) or bb_l[i] >= kc_l[i]:
                continue
            d = "SHORT"; sl = cl[i] + 1.5*atr_v
        sigs[i] = {"direction": d, "entry_price": cl[i], "sl_price": sl}
    return sigs


def get_daily_dir(d1d: dict | None, ts_1h: int) -> str | None:
    if d1d is None:
        return None
    idx = int(np.searchsorted(d1d["ts"], ts_1h, side="right")) - 1
    if idx < 200 or np.isnan(d1d["ema200"][idx]):
        return None
    return "LONG" if d1d["cl"][idx] > d1d["ema200"][idx] else "SHORT"


# ── MTF entry matching ────────────────────────────────────────────────────────

def match_ltf_entries(htf_sigs: dict, d1h: dict, dltf: dict,
                      d1d: dict | None, params: dict) -> dict:
    """Convert 1h signals → LTF entry bars. Returns {ltf_bar_idx: trade_dict}"""
    if dltf is None or not htf_sigs:
        return {}

    entry_mode = params["entry_mode"]
    max_wait   = params["max_wait"]
    sl_atr     = params["sl_atr"]
    tp_r       = params["tp_r"]
    use_daily  = params["daily_ema200"]

    ts_1h  = d1h["ts"]
    ts_ltf = dltf["ts"]
    cl_ltf = dltf["cl"]; hi_ltf = dltf["hi"]; lo_ltf = dltf["lo"]
    atr_ltf = dltf["atr"]; ema21_ltf = dltf["ema21"]
    n_ltf = dltf["n"]

    entries = {}
    for htf_bar, sig in htf_sigs.items():
        htf_ts = ts_1h[htf_bar]
        direction = sig["direction"]

        # Daily EMA200 filter
        if use_daily and d1d is not None:
            dd = get_daily_dir(d1d, htf_ts)
            if dd is not None and dd != direction:
                continue

        # Find first LTF bar after HTF signal
        ltf_start = int(np.searchsorted(ts_ltf, htf_ts, side="left"))
        if ltf_start >= n_ltf:
            continue

        for j in range(ltf_start, min(ltf_start + max_wait, n_ltf)):
            if j in entries:
                continue
            atr_v = atr_ltf[j] if not np.isnan(atr_ltf[j]) else cl_ltf[j]*0.01
            entry_p = cl_ltf[j]

            if entry_mode == "next_bar":
                sl = entry_p - sl_atr*atr_v if direction=="LONG" else entry_p + sl_atr*atr_v
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
                if direction == "LONG" and lo_ltf[j] <= ema*1.002 and cl_ltf[j] > ema:
                    sl = entry_p - sl_atr*atr_v
                    dist = abs(entry_p - sl)
                    if dist <= 0:
                        continue
                    tp = entry_p + tp_r*dist
                    entries[j] = {"direction": direction, "entry": entry_p,
                                  "sl": sl, "tp": tp, "dist": dist}
                    break
                elif direction == "SHORT" and hi_ltf[j] >= ema*0.998 and cl_ltf[j] < ema:
                    sl = entry_p + sl_atr*atr_v
                    dist = abs(entry_p - sl)
                    if dist <= 0:
                        continue
                    tp = entry_p - tp_r*dist
                    entries[j] = {"direction": direction, "entry": entry_p,
                                  "sl": sl, "tp": tp, "dist": dist}
                    break
    return entries


# ── Portfolio simulation ──────────────────────────────────────────────────────

def run_portfolio(all_entries: dict[str, dict[str, dict]],
                  all_ltf: dict[str, dict[str, dict]],
                  start_frac: float = 0.0,
                  end_frac: float = 1.0) -> dict:
    """
    all_entries: {sym: {strat_id: {ltf_bar: trade_dict}}}
    all_ltf:     {sym: {strat_id: dltf}}
    """
    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0
    trades  = []
    open_pos: dict[tuple, dict] = {}   # (sym, strat) -> pos
    sym_cooldown: dict[str, int] = {}  # sym -> cooldown_until_bar (in 1h bars ~)

    # Build flat timeline: (ltf_ts, sym, strat, ltf_bar, trade_dict)
    timeline = []
    for sym, strat_map in all_entries.items():
        for strat_id, entries in strat_map.items():
            dltf = all_ltf.get(sym, {}).get(strat_id)
            if dltf is None:
                continue
            ts_arr = dltf["ts"]
            n_ltf  = dltf["n"]
            i_start = int(n_ltf * start_frac)
            i_end   = int(n_ltf * end_frac)
            for bar, td in entries.items():
                if bar < i_start or bar >= i_end:
                    continue
                timeline.append((int(ts_arr[bar]), sym, strat_id, bar, td))

    timeline.sort(key=lambda x: x[0])

    for ts_entry, sym, strat_id, bar, td in timeline:
        dltf = all_ltf[sym][strat_id]
        cl   = dltf["cl"]; hi = dltf["hi"]; lo = dltf["lo"]
        n    = dltf["n"]

        key = (sym, strat_id)

        # Skip if already in position for this sym+strat
        if key in open_pos:
            continue
        # Skip if symbol on cooldown
        if sym_cooldown.get(sym, 0) > ts_entry:
            continue
        # Skip if max positions reached
        if len(open_pos) >= MAX_POSITIONS:
            continue

        d      = td["direction"]
        entry  = td["entry"]
        sl     = td["sl"]
        tp     = td["tp"]
        dist   = td["dist"]

        # Position sizing
        risk_usd  = balance * RISK_PER_TRADE
        if dist <= 0 or risk_usd < 0.01:
            continue
        raw_qty   = risk_usd / dist
        raw_margin = raw_qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        lev = 1
        if raw_margin > max_margin and max_margin > 0:
            lev = min(int(np.ceil(raw_margin / max_margin)), MAX_LEVERAGE)
        eff_margin = raw_margin / max(lev, 1)
        if eff_margin < MIN_MARGIN:
            continue  # would be rejected on Bybit

        open_pos[key] = {
            "entry": entry, "sl": sl, "tp": tp, "dist": dist,
            "d": d, "qty": raw_qty, "bar": bar, "sym": sym, "strat": strat_id,
            "lev": lev,
        }

        # Simulate bar-by-bar until SL/TP
        exit_bar = None; exit_price = None; reason = "timeout"
        for j in range(bar + 1, n):
            hi_j = hi[j]; lo_j = lo[j]; cl_j = cl[j]
            if d == "LONG":
                if lo_j <= sl:
                    exit_price = sl; reason = "SL"; exit_bar = j; break
                if hi_j >= tp:
                    exit_price = tp; reason = "TP"; exit_bar = j; break
            else:
                if hi_j >= sl:
                    exit_price = sl; reason = "SL"; exit_bar = j; break
                if lo_j <= tp:
                    exit_price = tp; reason = "TP"; exit_bar = j; break

        if exit_bar is None:
            open_pos.pop(key, None)
            continue

        # PnL
        qty = raw_qty
        gross = (exit_price - entry) * qty if d == "LONG" else (entry - exit_price) * qty
        fees  = (entry * qty + exit_price * qty) * (TAKER_FEE + SLIPPAGE)
        pnl   = gross - fees

        balance += pnl
        balance  = max(balance, 0.0)
        peak     = max(peak, balance)
        dd       = (peak - balance) / peak * 100.0 if peak > 0 else 0.0
        max_dd   = max(max_dd, dd)

        trades.append({
            "sym": sym, "strat": strat_id, "direction": d,
            "entry": entry, "exit": exit_price, "reason": reason,
            "pnl_usd": round(pnl, 4), "pnl_r": round(pnl / (risk_usd if risk_usd > 0 else 1), 2),
            "win": pnl > 0,
        })

        # Cooldown after SL (4h = ~48 × 5m bars or ~16 × 15m bars — use ts offset)
        if reason == "SL":
            cooldown_ms = 4 * 3_600_000
            sym_cooldown[sym] = int(dltf["ts"][exit_bar]) + cooldown_ms

        open_pos.pop(key, None)

    if not trades:
        return {"trades": 0, "pf": 0.0, "wr": 0.0, "pnl_usd": 0.0,
                "pnl_pct": 0.0, "max_dd": 0.0, "final_balance": balance}

    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gross_win  = sum(t["pnl_usd"] for t in wins)
    gross_loss = abs(sum(t["pnl_usd"] for t in losses))
    pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
    wr = len(wins) / len(trades) * 100

    return {
        "trades":        len(trades),
        "wins":          len(wins),
        "losses":        len(losses),
        "pf":            round(pf, 2),
        "wr":            round(wr, 1),
        "pnl_usd":       round(balance - START_BALANCE, 2),
        "pnl_pct":       round((balance - START_BALANCE) / START_BALANCE * 100, 1),
        "max_dd":        round(max_dd, 1),
        "final_balance": round(balance, 2),
        "per_strat":     _per_strat(trades),
    }


def _per_strat(trades: list) -> dict:
    from collections import defaultdict
    by_s: dict[str, list] = defaultdict(list)
    for t in trades:
        by_s[t["strat"]].append(t)
    out = {}
    for s, ts in by_s.items():
        wins = [t for t in ts if t["win"]]
        losses = [t for t in ts if not t["win"]]
        gw = sum(t["pnl_usd"] for t in wins)
        gl = abs(sum(t["pnl_usd"] for t in losses))
        out[s] = {
            "trades": len(ts),
            "wr": round(len(wins)/len(ts)*100, 1),
            "pf": round(gw/gl, 2) if gl > 0 else float("inf"),
            "pnl_usd": round(sum(t["pnl_usd"] for t in ts), 2),
        }
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

# Symbol sets for comparison
SYMBOL_SETS = {
    "CORE_20": [                                   # тот самый топ из бэктеста v10
        "BTC/USDT:USDT",  "ETH/USDT:USDT",  "BNB/USDT:USDT",  "SOL/USDT:USDT",
        "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
        "DOT/USDT:USDT",  "LINK/USDT:USDT", "UNI/USDT:USDT",  "ATOM/USDT:USDT",
        "NEAR/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",  "OP/USDT:USDT",
        "INJ/USDT:USDT",  "SUI/USDT:USDT",  "TRX/USDT:USDT",  "TON/USDT:USDT",
    ],
    "LIVE_34": [                                   # текущий live (top-100 mcap + $5M vol)
        "BTC/USDT:USDT",  "ETH/USDT:USDT",  "BNB/USDT:USDT",  "SOL/USDT:USDT",
        "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
        "DOT/USDT:USDT",  "LINK/USDT:USDT", "UNI/USDT:USDT",  "ATOM/USDT:USDT",
        "NEAR/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",  "OP/USDT:USDT",
        "INJ/USDT:USDT",  "SUI/USDT:USDT",  "TRX/USDT:USDT",  "TON/USDT:USDT",
        "BCH/USDT:USDT",  "ICP/USDT:USDT",  "FIL/USDT:USDT",  "HBAR/USDT:USDT",
        "XLM/USDT:USDT",  "ALGO/USDT:USDT", "ONDO/USDT:USDT", "1000PEPE/USDT:USDT",
        "SEI/USDT:USDT",  "XMR/USDT:USDT",  "CRV/USDT:USDT",  "POL/USDT:USDT",
        "MNT/USDT:USDT",  "XAUT/USDT:USDT",
    ],
    "FULL_36": SYMBOLS_365D,                       # все 36 с кешем 365d/730d
    "TOP_50": [                                    # топ-50 по mcap (устоявшиеся монеты)
        # CORE_20
        "BTC/USDT:USDT",  "ETH/USDT:USDT",  "BNB/USDT:USDT",  "SOL/USDT:USDT",
        "XRP/USDT:USDT",  "ADA/USDT:USDT",  "AVAX/USDT:USDT", "DOGE/USDT:USDT",
        "DOT/USDT:USDT",  "LINK/USDT:USDT", "UNI/USDT:USDT",  "ATOM/USDT:USDT",
        "NEAR/USDT:USDT", "APT/USDT:USDT",  "ARB/USDT:USDT",  "OP/USDT:USDT",
        "INJ/USDT:USDT",  "SUI/USDT:USDT",  "TRX/USDT:USDT",  "TON/USDT:USDT",
        # LIVE_34 additions
        "BCH/USDT:USDT",  "LTC/USDT:USDT",  "ICP/USDT:USDT",  "HBAR/USDT:USDT",
        "XLM/USDT:USDT",  "FIL/USDT:USDT",  "ALGO/USDT:USDT", "POL/USDT:USDT",
        "SEI/USDT:USDT",  "XMR/USDT:USDT",  "CRV/USDT:USDT",  "MNT/USDT:USDT",
        "XAUT/USDT:USDT", "ONDO/USDT:USDT", "1000PEPE/USDT:USDT",
        # Top-50 extensions (топ-50 mcap в 2024-2025, листинг на Bybit > 2022)
        "AAVE/USDT:USDT",       # DeFi blue-chip,     top 30-45
        "RENDER/USDT:USDT",     # AI/GPU rendering,   top 25-40
        "LDO/USDT:USDT",        # Lido staking,       top 30-45
        "STX/USDT:USDT",        # Bitcoin L2,         top 30-40 (2024 peak)
        "DYDX/USDT:USDT",       # DeFi perps,         top 40-55
        "IMX/USDT:USDT",        # Gaming L2,          top 35-50
        "GRT/USDT:USDT",        # The Graph,          top 40-60
        "SAND/USDT:USDT",       # Metaverse,          top 30-50
        "GALA/USDT:USDT",       # Gaming,             top 35-55
        "FTM/USDT:USDT",        # Fantom/Sonic,       top 30-45
        "EGLD/USDT:USDT",       # MultiversX,         top 35-55
        "WLD/USDT:USDT",        # Worldcoin,          top 20-35 (2025)
        "PENDLE/USDT:USDT",     # Yield trading,      top 40-60
        "TAO/USDT:USDT",        # Bittensor AI,       top 15-30 (2025)
        "1000BONK/USDT:USDT",   # BONK,              top 25-40 (2024)
    ],
}


def _load_all_data(symbols: list[str], t0: float) -> tuple[dict, int]:
    all_data: dict[str, dict] = {}
    loaded = 0
    for sym in symbols:
        d1h     = prepare(_load(sym, "1h"))
        d5m     = prepare(_load(sym, "5m"))
        d15m    = prepare(_load(sym, "15m"))
        d1d_raw = _load(sym, "1d")
        d1d = None
        if d1d_raw is not None and len(d1d_raw) >= 210:
            cl1d = d1d_raw["close"].to_numpy(dtype=float)
            ts1d = d1d_raw["timestamp"].to_numpy(dtype=int)
            d1d = {"cl": cl1d, "ts": ts1d, "ema200": _ema(cl1d, 200)}
        if d1h is None or len(d1h["cl"]) < 300:
            continue
        all_data[sym] = {"1h": d1h, "5m": d5m, "15m": d15m, "1d": d1d}
        loaded += 1
    logger.info("Loaded %d/%d symbols in %.1fs", loaded, len(symbols), _time.time()-t0)
    return all_data, loaded


def _build_entries(all_data: dict, t0: float) -> tuple[dict, dict, int]:
    all_entries: dict[str, dict[str, dict]] = {}
    all_ltf:     dict[str, dict[str, dict]] = {}
    total_sigs = 0
    for sym, data in all_data.items():
        d1h = data["1h"]; d1d = data["1d"]
        all_entries[sym] = {}; all_ltf[sym] = {}
        for strat_id, params in STRATEGIES.items():
            htf_sigs = detect_bbs_signals_1h(d1h, params) if params["type"] == "bbs" \
                       else detect_kc_signals_1h(d1h, params)
            total_sigs += len(htf_sigs)
            tf_key = params["entry_tf"]
            dltf = data.get(tf_key)
            if dltf is None:
                continue
            entries = match_ltf_entries(htf_sigs, d1h, dltf, d1d, params)
            all_entries[sym][strat_id] = entries
            all_ltf[sym][strat_id]     = dltf
    logger.info("1h signals: %d, entries matched in %.1fs", total_sigs, _time.time()-t0)
    return all_entries, all_ltf, total_sigs


def _monthly_breakdown(all_entries: dict, all_ltf: dict) -> None:
    """
    Month-by-month portfolio simulation.
    Shows ACTUAL balance growth % per calendar month (start→end balance).
    Also computes CAGR and annotates market context.
    """
    from collections import defaultdict
    from datetime import datetime, timezone

    # ── Build flat timeline ──────────────────────────────────────────────────
    timeline = []
    for sym, strat_map in all_entries.items():
        for strat_id, entries in strat_map.items():
            dltf = all_ltf.get(sym, {}).get(strat_id)
            if dltf is None:
                continue
            ts_arr = dltf["ts"]
            for bar, td in entries.items():
                timeline.append((int(ts_arr[bar]), sym, strat_id, bar, td))
    timeline.sort(key=lambda x: x[0])

    # ── Full simulation with monthly balance tracking ────────────────────────
    balance = START_BALANCE
    open_pos: dict[tuple, dict] = {}
    sym_cooldown: dict[str, int] = {}

    month_start_bal: dict[str, float] = {}   # first trade of the month → balance before
    month_end_bal:   dict[str, float] = {}   # last trade of the month → balance after
    month_trades: dict[str, list]     = defaultdict(list)

    for ts_entry, sym, strat_id, bar, td in timeline:
        dltf = all_ltf[sym][strat_id]
        cl = dltf["cl"]; hi = dltf["hi"]; lo = dltf["lo"]
        n  = dltf["n"]
        key = (sym, strat_id)

        if key in open_pos:
            continue
        if sym_cooldown.get(sym, 0) > ts_entry:
            continue
        if len(open_pos) >= MAX_POSITIONS:
            continue

        d = td["direction"]; entry = td["entry"]; sl = td["sl"]
        tp = td["tp"]; dist = td["dist"]

        risk_usd  = balance * RISK_PER_TRADE
        if dist <= 0 or risk_usd < 0.01:
            continue
        raw_qty    = risk_usd / dist
        raw_margin = raw_qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        lev = max(1, min(int(np.ceil(raw_margin / max_margin)) if max_margin > 0 else 1,
                         MAX_LEVERAGE))
        eff_margin = raw_margin / lev
        if eff_margin < MIN_MARGIN:
            continue

        open_pos[key] = True

        exit_price = None; reason = "timeout"
        for j in range(bar + 1, n):
            if d == "LONG":
                if lo[j] <= sl: exit_price = sl; reason = "sl"; break
                if hi[j] >= tp: exit_price = tp; reason = "tp"; break
            else:
                if hi[j] >= sl: exit_price = sl; reason = "sl"; break
                if lo[j] <= tp: exit_price = tp; reason = "tp"; break
        del open_pos[key]
        if exit_price is None:
            continue

        pnl   = (exit_price - entry)*raw_qty if d=="LONG" else (entry - exit_price)*raw_qty
        fees  = (entry + exit_price) * raw_qty * (TAKER_FEE + SLIPPAGE)
        pnl_net = pnl - fees

        month_key = datetime.fromtimestamp(ts_entry/1000, tz=timezone.utc).strftime("%Y-%m")

        # Record balance BEFORE this trade as month start (first trade of month)
        if month_key not in month_start_bal:
            month_start_bal[month_key] = balance

        balance += pnl_net
        balance  = max(balance, 0.0)

        # Update month end balance after every trade
        month_end_bal[month_key] = balance
        month_trades[month_key].append(pnl_net > 0)

        if reason == "sl":
            sym_cooldown[sym] = ts_entry + SL_COOLDOWN_BARS_1H * 3_600_000

    # ── Build display ────────────────────────────────────────────────────────
    months = sorted(month_start_bal.keys())
    if not months:
        return

    # Identify first/last date for context
    first_month = months[0]; last_month = months[-1]

    # Known crypto market regimes (for context annotation)
    # bear = sustained downtrend, sideways = choppy, bull = strong uptrend
    REGIME: dict[str, str] = {
        "2024-01": "sidew", "2024-02": "sidew", "2024-03": "bull ",
        "2024-04": "sidew", "2024-05": "sidew", "2024-06": "sidew",
        "2024-07": "sidew", "2024-08": "sidew", "2024-09": "sidew",
        "2024-10": "bull ", "2024-11": "bull ", "2024-12": "bull ",
        "2025-01": "bull ", "2025-02": "bull ", "2025-03": "sidew",
        "2025-04": "sidew", "2025-05": "bull ", "2025-06": "bull ",
        "2025-07": "bull ", "2025-08": "sidew", "2025-09": "bull ",
        "2025-10": "bull ", "2025-11": "bull ", "2025-12": "bull ",
        "2026-01": "bull ", "2026-02": "sidew", "2026-03": "bull ",
        "2026-04": "sidew",
    }

    print(f"\n  Monthly actual balance growth  (compounding 2%/trade, ${START_BALANCE} start):")
    print(f"  NOTE: % = actual (end_bal - start_bal) / start_bal for that calendar month")
    print(f"  {'Month':<10}  {'Regime':<6}  {'Trades':>6}  {'WR%':>5}  "
          f"{'Start$':>8}  {'End$':>8}  {'Month%':>8}  Bar")

    n_pos = 0; n_months = len(months)
    monthly_returns = []

    for m in months:
        s_bal = month_start_bal[m]
        e_bal = month_end_bal.get(m, s_bal)
        pct   = (e_bal - s_bal) / s_bal * 100 if s_bal > 0 else 0.0
        trades = month_trades[m]
        wr     = sum(trades)/len(trades)*100 if trades else 0
        regime = REGIME.get(m, "     ")
        bar_len = min(int(abs(pct) / 5), 24)
        sign   = "+" if pct >= 0 else "-"
        print(f"  {m:<10}  {regime:<6}  {len(trades):>6}  {wr:>4.0f}%  "
              f"${s_bal:>7.0f}  ${e_bal:>7.0f}  {pct:>+7.1f}%  {sign}{'#'*bar_len}")
        if pct > 0:
            n_pos += 1
        monthly_returns.append(pct / 100)

    # ── CAGR ────────────────────────────────────────────────────────────────
    if n_months > 0 and balance > START_BALANCE:
        # Geometric mean monthly return
        cagr_m = (balance / START_BALANCE) ** (1.0 / n_months) - 1
        cagr_a = (1 + cagr_m) ** 12 - 1
        # Arithmetic mean and std for context
        import statistics
        avg_m   = statistics.mean(monthly_returns) * 100
        stdev_m = statistics.stdev(monthly_returns) * 100 if n_months > 1 else 0

        print(f"\n  Profitable months : {n_pos}/{n_months}")
        print(f"  CAGR              : {cagr_m*100:+.1f}%/month  ({cagr_a*100:+.0f}%/year)")
        print(f"  Avg monthly       : {avg_m:+.1f}% +/- {stdev_m:.1f}% (arith. mean +/- std)")
        print(f"  Final balance     : ${balance:.2f}  (started ${START_BALANCE})")
        print(f"  Period            : {first_month} to {last_month}  ({n_months} months)")
    else:
        print(f"  Profitable months: {n_pos}/{n_months}")


def main():
    t0 = _time.time()

    print("\n" + "="*70)
    print(f"  v12 LIVE CONFIG VALIDATION  |  start=${START_BALANCE}  risk=2%  fee=0.055%")
    print(f"  Strategies: {', '.join(STRATEGIES.keys())}")
    print(f"  Data: up to 730d (2yr) if cached, fallback 365d")
    print(f"  Walk-forward: IS=80% / OOS=20% (most recent period)")
    print(f"")
    print(f"  NOTES ON REALISM:")
    print(f"  - Survivorship bias: symbols chosen from CURRENT top-N (not historical)")
    print(f"  - Regime context shown per month (bull/sideways/bear)")
    print(f"  - PnL$ per strategy NOT shown (misleading with compounding from $100)")
    print(f"  - Monthly% = actual balance growth (start->end of calendar month)")
    print(f"  - CAGR = geometric mean monthly return (more conservative than simple avg)")
    print("="*70)

    for set_name, symbols in SYMBOL_SETS.items():
        print(f"\n{'='*70}")
        print(f"  SYMBOL SET: {set_name}  ({len(symbols)} symbols)")
        print(f"{'='*70}")

        all_data, loaded = _load_all_data(symbols, t0)
        if loaded == 0:
            print("  No data loaded — skip")
            continue

        all_entries, all_ltf, _ = _build_entries(all_data, t0)

        # Detect actual data period from first loaded symbol
        data_label = "?"
        for _sym_d in all_data.values():
            _d1h = _sym_d.get("1h")
            if _d1h is not None and _d1h["n"] > 0:
                ts0 = _d1h["ts"][0]; ts1 = _d1h["ts"][-1]
                days = (ts1 - ts0) // (86_400_000)
                data_label = f"{days}d"
                break

        res_full = run_portfolio(all_entries, all_ltf, 0.0, 1.0)
        res_is   = run_portfolio(all_entries, all_ltf, 0.0, 0.80)
        res_oos  = run_portfolio(all_entries, all_ltf, 0.80, 1.0)

        _print_result(f"FULL {data_label}  [{set_name}]", res_full)
        _print_result(f"IS  80%    [{set_name}]", res_is)
        _print_result(f"OOS 20%    [{set_name}]", res_oos)

        # Monthly breakdown only for FULL period
        _monthly_breakdown(all_entries, all_ltf)

    print(f"\n  Total runtime: {_time.time()-t0:.1f}s")
    print("="*70)


def _print_result(label: str, r: dict):
    if r["trades"] == 0:
        print(f"\n  [{label}]  No trades.")
        return
    print(f"\n  [{label}]")
    print(f"  Trades: {r['trades']}  WR: {r['wr']}%  PF: {r['pf']}  MaxDD: {r['max_dd']}%")
    ps = r.get("per_strat", {})
    for s, v in sorted(ps.items(), key=lambda x: -x[1]["trades"]):
        # Show WR/PF per strategy — NOT absolute PnL (misleading due to compounding)
        print(f"    {s:15s}: {v['trades']:3d} tr  WR={v['wr']}%  PF={v['pf']}")


if __name__ == "__main__":
    main()
