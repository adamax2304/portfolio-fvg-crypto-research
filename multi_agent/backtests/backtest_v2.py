"""
multi_agent/backtests/backtest_v2.py  — Soft-Regime Portfolio Backtest v2

Redesigned from evidence:
  - FVG: loosened filters (composite 55-90, fib optional, ADX 15, wider zone)
  - BBS: soft gating (trades in SQUEEZE/TRANSITION/TREND), partial TP at 2R
  - Density: minor exit refinement (kept — already profitable)
  - MR: redesigned (VWAP target, tighter stretch, anti-cluster)
  - Portfolio: soft regime sizing (size_mult × confidence)

Usage:
    py -3 multi_agent/backtests/backtest_v2.py
"""
from __future__ import annotations
import json, logging, time as _time, itertools
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.shared.regime_detector import (
    SoftRegimeDetector, SoftRegimeState, Regime,
    precompute_1h_indicators, precompute_4h_indicators,
    _ema, _sma, _stdev, _atr, _adx, _percentile_rank,
    DEFAULT_SOFT_CFG,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
from multi_agent.agents.density_agent import (
    compute_volume_profile, compute_rolling_vwap as density_vwap,
    detect_density_signal, _rsi as density_rsi, _atr as density_atr,
    _adx as density_adx,
)
from multi_agent.shared.fvg_core import (
    detect_fvgs_v2, _select_active_fvgs,
    composite_signal_score, evaluate_entry_reaction,
)
import multi_agent.config as cfg

logger = logging.getLogger("bt_v2")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

BACKTEST_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

START_BALANCE = 1000.0
TAKER_FEE = 0.00055
SLIPPAGE = 0.00025
MAX_LEVERAGE = 25


# ═══════════════════════════════════════════════════════════════════════════════
#  Tunable parameters — baseline config
# ═══════════════════════════════════════════════════════════════════════════════

def baseline_params():
    """Returns the baseline parameter dict for the full portfolio backtest."""
    return dict(
        # ── Portfolio allocation & risk ──
        fvg_frac=0.35, bbs_frac=0.25, density_frac=0.25, mr_frac=0.15,
        fvg_risk=0.010, bbs_risk=0.008, density_risk=0.009, mr_risk=0.005,
        max_positions=8, max_per_symbol=1,

        # ── Soft regime config ──
        regime_confirm_bars=5, regime_min_hold=15, regime_smooth_alpha=0.35,
        regime_min_confidence=1.0, regime_transition_margin=0.8,
        regime_undefined_threshold=0.5,

        # ── FVG params ──
        fvg_composite_lo=55.0, fvg_composite_hi=90.0,
        fvg_quality_min=40.0,
        fvg_adx_min=15.0,
        fvg_volume_mult=0.6,
        fvg_entry_lo=0.20, fvg_entry_hi=0.80,  # wider FVG zone
        fvg_fib_mandatory=False,                 # fib as boost not gate
        fvg_fib_bonus=5.0,                       # composite bonus for fib confluence

        # ── BBS params ──
        bbs_partial_tp_r=2.0,    # take 50% at 2R, trail rest
        bbs_partial_frac=0.5,
        bbs_max_adx=45.0,       # late-expansion ceiling (was 40)

        # ── Density params ──
        density_vp_lookback=96,
        density_rsi_os=35.0, density_rsi_ob=65.0,
        density_adx_max=22.0,
        density_vwap_dev=0.8,
        density_sl_atr=1.8,

        # ── MR params ──
        mr_bb_len=20, mr_bb_mult=2.0,
        mr_rsi_os=28.0, mr_rsi_ob=72.0,
        mr_adx_max=22.0,          # loosened from 20
        mr_vwap_dev=0.5,          # loosened from 0.6
        mr_sl_atr=1.8,            # wider SL
        mr_tp_target="vwap",      # VWAP instead of BB midline
        mr_max_consec=2,
        mr_cluster_cooldown=6,    # bars to wait after 2 same-dir losses
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load(symbol: str, tf: str) -> Optional[pd.DataFrame]:
    safe = symbol.replace("/", "_").replace(":", "_")
    for ext in (".csv", ".parquet"):
        fpath = CACHE_DIR / f"{safe}_{tf}_90d{ext}"
        if fpath.exists():
            try:
                df = pd.read_csv(fpath) if ext == ".csv" else pd.read_parquet(fpath)
                if df is None or df.empty:
                    continue
                for c in ("open", "high", "low", "close", "volume"):
                    if c in df.columns:
                        df[c] = df[c].astype(float)
                if "timestamp" in df.columns:
                    df["timestamp"] = df["timestamp"].astype(int)
                return df
            except Exception:
                continue
    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  FVG signal detection — LOOSENED
# ═══════════════════════════════════════════════════════════════════════════════

def _fvg_signal(df_15m, i, closes, highs, lows, volumes, timestamps_15m,
                closes_1h, timestamps_1h, ema200_1h,
                closes_4h, timestamps_4h, ema200_4h,
                adx_15m, P):
    """Loosened FVG signal detection at bar i. Returns signal dict or None."""
    if i < 250:
        return None

    ts_i = int(timestamps_15m[i])
    htf_idx = int(np.searchsorted(timestamps_1h, ts_i, side="right")) - 1
    if htf_idx < 200 or htf_idx >= len(ema200_1h) or np.isnan(ema200_1h[htf_idx]):
        return None

    htf_close = closes_1h[htf_idx]
    htf_ema = ema200_1h[htf_idx]
    htf_bullish = htf_close > htf_ema
    htf_bearish = htf_close < htf_ema

    # 4h confirmation (soft: only reject clear contradiction)
    if ema200_4h is not None and timestamps_4h is not None:
        j4 = int(np.searchsorted(timestamps_4h, ts_i, side="right")) - 1
        if 24 <= j4 < len(ema200_4h) and not np.isnan(ema200_4h[j4]):
            if htf_bullish and closes_4h[j4] < ema200_4h[j4] * 0.995:
                return None
            if htf_bearish and closes_4h[j4] > ema200_4h[j4] * 1.005:
                return None

    # ADX filter (loosened from 20 to param)
    adx_val = adx_15m[i]
    if not np.isnan(adx_val) and adx_val < P["fvg_adx_min"]:
        return None

    # Volume filter (loosened)
    if i >= 10:
        vol_avg = np.mean(volumes[i - 10:i])
        if vol_avg > 0 and volumes[i] < P["fvg_volume_mult"] * vol_avg:
            return None

    # Detect FVGs
    ws = max(0, i - 384)
    df_w = df_15m.iloc[ws:i + 1].copy().reset_index(drop=True)
    if len(df_w) < 50:
        return None

    all_fvgs = detect_fvgs_v2(df_w, lookback=min(384, len(df_w)),
                               percentile_threshold=cfg.FVG_PERCENTILE_THRESHOLD,
                               percentile_lookback=min(500, len(df_w)))
    active = _select_active_fvgs(all_fvgs)

    cv = closes[i]
    for fvg in active:
        if fvg.type == "bullish" and not htf_bullish:
            continue
        if fvg.type == "bearish" and not htf_bearish:
            continue
        if fvg.quality_score < P["fvg_quality_min"]:
            continue

        # Wider entry zone
        gap = max(0.0, fvg.high - fvg.low)
        if gap <= 0:
            continue
        lo_band = fvg.low + gap * P["fvg_entry_lo"]
        hi_band = fvg.low + gap * P["fvg_entry_hi"]
        if not (lo_band <= cv <= hi_band):
            continue

        # Composite score (loosened range)
        reaction_ok, reaction_score = evaluate_entry_reaction(df_w.tail(6), fvg)
        fvg.reaction_score = float(reaction_score)
        comp = composite_signal_score(fvg, reaction_score)

        # Fib confluence as optional bonus
        fib_bonus = 0.0
        if P["fvg_fib_mandatory"] or True:  # always check, just bonus vs gate
            swing_lo = float(np.min(lows[max(0, i - 120):i + 1]))
            swing_hi = float(np.max(highs[max(0, i - 120):i + 1]))
            fib_range = swing_hi - swing_lo
            if fib_range > 0:
                f382 = swing_hi - fib_range * 0.618
                f618 = swing_hi - fib_range * 0.382
                if f382 <= cv <= f618:
                    fib_bonus = P["fvg_fib_bonus"]

        effective_comp = comp + fib_bonus

        if effective_comp < P["fvg_composite_lo"] or effective_comp > P["fvg_composite_hi"]:
            if P["fvg_fib_mandatory"]:
                continue
            # Without fib mandatory, still need base composite in wider band
            if comp < P["fvg_composite_lo"] - 5:
                continue

        # Build trade
        direction = "LONG" if fvg.type == "bullish" else "SHORT"
        if direction == "LONG":
            sl = float(np.min(lows[max(0, i - 60):i + 1])) * 0.9985
        else:
            sl = float(np.max(highs[max(0, i - 60):i + 1])) * 1.0015
        risk = abs(cv - sl)
        if risk <= 0 or risk / cv > 0.05:
            continue

        return {"direction": direction, "entry_price": float(cv), "sl": float(sl),
                "risk": float(risk), "composite": float(effective_comp), "bar_idx": i}

    return None


# ═══════════════════════════════════════════════════════════════════════════════
#  MR signal detection — REDESIGNED
# ═══════════════════════════════════════════════════════════════════════════════

def _mr_signal_v2(closes, highs, lows, volumes, i, vwap_arr, P,
                  consec_losses, cluster_cooldown):
    """Redesigned MR: BB stretch + VWAP distance + RSI + anti-cluster."""
    if i < 50:
        return None

    # BB
    bb_len = P["mr_bb_len"]
    if i < bb_len + 5:
        return None
    seg = closes[i - bb_len:i + 1]
    mid = np.mean(seg)
    std = np.std(seg, ddof=0)
    if std == 0:
        return None
    upper = mid + P["mr_bb_mult"] * std
    lower = mid - P["mr_bb_mult"] * std

    # Z-score stretch
    zscore = (closes[i] - mid) / std

    # RSI
    deltas = np.diff(closes[max(0, i - 14):i + 1])
    if len(deltas) < 14:
        return None
    gains = np.where(deltas > 0, deltas, 0)
    losses_arr = np.where(deltas < 0, -deltas, 0)
    avg_g = np.mean(gains[-14:])
    avg_l = np.mean(losses_arr[-14:])
    rsi = 100.0 - 100.0 / (1 + avg_g / max(avg_l, 1e-10))

    # ADX
    adx_seg = _adx(highs[max(0, i - 40):i + 1], lows[max(0, i - 40):i + 1],
                    closes[max(0, i - 40):i + 1], 14)
    adx_val = adx_seg[-1] if len(adx_seg) > 0 and not np.isnan(adx_seg[-1]) else 50

    if adx_val > P["mr_adx_max"]:
        return None

    # VWAP distance
    vwap_val = vwap_arr[i] if i < len(vwap_arr) and not np.isnan(vwap_arr[i]) else mid
    vwap_dev = abs(closes[i] - vwap_val) / max(vwap_val, 1e-10) * 100.0

    # ATR for stop
    atr_seg = _atr(highs[max(0, i - 20):i + 1], lows[max(0, i - 20):i + 1],
                    closes[max(0, i - 20):i + 1], 14)
    atr_val = atr_seg[-1] if len(atr_seg) > 0 and not np.isnan(atr_seg[-1]) else 0
    if atr_val <= 0:
        return None

    direction = None

    # LONG: price below lower BB, RSI oversold-ish, stretched from VWAP
    if closes[i] < lower and rsi < P["mr_rsi_os"] and zscore < -1.8:
        if vwap_dev >= P["mr_vwap_dev"]:
            direction = "LONG"

    # SHORT: price above upper BB, RSI overbought-ish
    elif closes[i] > upper and rsi > P["mr_rsi_ob"] and zscore > 1.8:
        if vwap_dev >= P["mr_vwap_dev"]:
            direction = "SHORT"

    if direction is None:
        return None

    # Anti-cluster: check consecutive losses
    if consec_losses.get(direction, 0) >= P["mr_max_consec"]:
        if cluster_cooldown.get(direction, 0) > 0:
            return None

    # Volume: reject if volume surge (impulse continuation)
    if i >= 10:
        v_avg = np.mean(volumes[i - 10:i])
        if v_avg > 0 and volumes[i] > 2.5 * v_avg:
            return None

    entry = closes[i]
    if direction == "LONG":
        sl = entry - atr_val * P["mr_sl_atr"]
        tp = vwap_val if P["mr_tp_target"] == "vwap" else mid
    else:
        sl = entry + atr_val * P["mr_sl_atr"]
        tp = vwap_val if P["mr_tp_target"] == "vwap" else mid

    risk = abs(entry - sl)
    if risk <= 0 or risk / entry > 0.05:
        return None

    return {"direction": direction, "entry_price": float(entry), "sl": float(sl),
            "tp": float(tp), "risk": float(risk), "bar_idx": i}


# ═══════════════════════════════════════════════════════════════════════════════
#  Core backtest engine
# ═══════════════════════════════════════════════════════════════════════════════

def run_portfolio_backtest(params: dict, use_regime: bool = True,
                           agent_filter: str = None,
                           start_frac: float = 0.0, end_frac: float = 1.0,
                           skip_fvg: bool = False) -> dict:
    """
    Run the full portfolio backtest.

    agent_filter: if set, only run this agent (e.g. "fvg", "bbs")
    start_frac/end_frac: slice of data for walk-forward (0.0 = start, 1.0 = end)
    skip_fvg: if True, skip FVG agent entirely (for fast optimization sweeps)
    """
    P = params
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    all_trades = []
    regime_dist = defaultdict(int)
    regime_switches = 0

    per_agent = {a: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": [], "rejected": 0}
                 for a in ["fvg", "bbs", "density", "mr"]}

    agents_to_run = ["fvg", "bbs", "density", "mr"]
    if agent_filter:
        agents_to_run = [agent_filter]
    if skip_fvg and "fvg" in agents_to_run:
        agents_to_run = [a for a in agents_to_run if a != "fvg"]

    soft_cfg = {
        "confirm_bars": P["regime_confirm_bars"],
        "min_hold_bars": P["regime_min_hold"],
        "smooth_alpha": P["regime_smooth_alpha"],
        "min_confidence": P["regime_min_confidence"],
        "transition_margin": P["regime_transition_margin"],
        "undefined_threshold": P["regime_undefined_threshold"],
    }

    risk_map = {"fvg": P["fvg_risk"], "bbs": P["bbs_risk"],
                "density": P["density_risk"], "mr": P["mr_risk"]}
    frac_map = {"fvg": P["fvg_frac"], "bbs": P["bbs_frac"],
                "density": P["density_frac"], "mr": P["mr_frac"]}

    for symbol in BACKTEST_SYMBOLS:
        df_15m = _load(symbol, "15m")
        df_1h = _load(symbol, "1h")
        df_4h = _load(symbol, "4h")
        if df_15m is None or df_1h is None:
            continue
        if len(df_15m) < 250 or len(df_1h) < 210:
            continue

        # Slice data for walk-forward
        n_total = len(df_15m)
        i_start = max(210, int(n_total * start_frac))
        i_end = int(n_total * end_frac)

        # Pre-extract arrays
        cl15 = df_15m["close"].astype(float).to_numpy()
        hi15 = df_15m["high"].astype(float).to_numpy()
        lo15 = df_15m["low"].astype(float).to_numpy()
        vo15 = df_15m["volume"].astype(float).to_numpy()
        ts15 = df_15m["timestamp"].astype(int).to_numpy()
        ts1h = df_1h["timestamp"].astype(int).to_numpy()

        # 1h indicators
        ind_1h = precompute_1h_indicators(df_1h)
        ema200_1h = ind_1h["ema200_1h"]
        cl1h = ind_1h["closes_1h_arr"]

        # 4h
        ind_4h = None
        ema200_4h = None
        ts4h = None
        cl4h = None
        if df_4h is not None and len(df_4h) > 50:
            ind_4h = precompute_4h_indicators(df_4h)
            ema200_4h = ind_4h["ema200_4h"]
            ts4h = df_4h["timestamp"].astype(int).to_numpy()
            cl4h = df_4h["close"].astype(float).to_numpy()

        # Agent-specific pre-compute
        squeeze_ind = compute_squeeze_indicators(df_15m) if "bbs" in agents_to_run else None
        adx_15m = _adx(hi15, lo15, cl15, 14) if "fvg" in agents_to_run else None
        if "density" in agents_to_run:
            d_rsi = density_rsi(cl15, cfg.DENSITY_RSI_PERIOD)
            d_adx = density_adx(hi15, lo15, cl15, 14)
            d_atr = density_atr(hi15, lo15, cl15, 14)
            d_vwap = density_vwap(hi15, lo15, cl15, vo15, P["density_vp_lookback"])
        if "mr" in agents_to_run:
            mr_vwap = density_vwap(hi15, lo15, cl15, vo15, 96)

        detector = SoftRegimeDetector(soft_cfg)
        prev_regime = Regime.UNDEFINED

        # Per-symbol state
        position = None  # {agent, direction, entry, sl, tp, risk, qty, bar_idx, trailing_sl, partial_taken}
        mr_consec = {"LONG": 0, "SHORT": 0}
        mr_cooldown = {"LONG": 0, "SHORT": 0}

        for i in range(i_start, i_end):
            close_i = cl15[i]
            high_i = hi15[i]
            low_i = lo15[i]

            # Regime
            htf_idx = int(np.searchsorted(ts1h, int(ts15[i]), side="right")) - 1
            if htf_idx < 210:
                continue

            rs = detector.update(symbol, df_1h, htf_idx, df_4h,
                                 **(ind_1h), **(ind_4h or {}))
            regime_dist[rs.regime.value] += 1
            if rs.regime != prev_regime:
                regime_switches += 1
                prev_regime = rs.regime

            # Tick MR cooldowns
            for d in ("LONG", "SHORT"):
                if mr_cooldown[d] > 0:
                    mr_cooldown[d] -= 1
                    if mr_cooldown[d] == 0:
                        mr_consec[d] = 0

            # ── Manage open position ──
            if position is not None:
                pos = position
                d = pos["direction"]
                entry = pos["entry"]
                sl = pos["sl"]
                r = pos["risk"]
                q = pos["qty"]
                agent = pos["agent"]

                hit_sl = (d == "LONG" and low_i <= sl) or (d == "SHORT" and high_i >= sl)
                hit_tp = False
                tp = pos.get("tp", 0)

                # BBS: trailing + partial TP
                if agent == "bbs" and pos.get("trailing_sl") is not None:
                    new_ce = chandelier_exit_sl(
                        hi15, lo15, squeeze_ind["ce_atr"], d, i,
                        ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                    pnl_r = ((close_i - entry) / r) if d == "LONG" else ((entry - close_i) / r)
                    pnl_r = pnl_r if r > 0 else 0

                    # Partial TP at specified R
                    if not pos.get("partial_taken") and pnl_r >= P["bbs_partial_tp_r"]:
                        frac = P["bbs_partial_frac"]
                        pq = q * frac
                        pnl_partial = ((close_i - entry) if d == "LONG" else (entry - close_i)) * pq
                        fee = close_i * pq * (TAKER_FEE + SLIPPAGE)
                        balance += pnl_partial - fee
                        pos["qty"] -= pq
                        pos["partial_taken"] = True
                        q = pos["qty"]

                    # Update trailing
                    if pnl_r >= 1.0:
                        if d == "LONG":
                            pos["trailing_sl"] = max(pos["trailing_sl"], new_ce)
                        else:
                            pos["trailing_sl"] = min(pos["trailing_sl"], new_ce) if pos["trailing_sl"] > 0 else new_ce

                    eff_sl = pos["trailing_sl"] if pos["trailing_sl"] > 0 else sl
                    if d == "LONG":
                        eff_sl = max(sl, eff_sl)
                        hit_sl = low_i <= eff_sl
                        if hit_sl:
                            sl = eff_sl
                    else:
                        if eff_sl > 0:
                            eff_sl = min(sl, eff_sl)
                        hit_sl = high_i >= eff_sl if eff_sl > 0 else high_i >= sl
                        if hit_sl:
                            sl = eff_sl if eff_sl > 0 else sl

                    # 4R max cap
                    if pnl_r >= 4.0:
                        hit_tp = True
                        tp = close_i

                # FVG: trailing chandelier
                elif agent == "fvg" and pos.get("trailing_sl") is not None:
                    ce_atr = _atr(hi15[max(0, i - 30):i + 1], lo15[max(0, i - 30):i + 1],
                                  cl15[max(0, i - 30):i + 1], 14)
                    ce_atr_val = ce_atr[-1] if len(ce_atr) > 0 and not np.isnan(ce_atr[-1]) else r
                    if d == "LONG":
                        trail = float(np.max(hi15[max(0, i - 22):i + 1])) - 3.0 * ce_atr_val
                        pnl_r = (close_i - entry) / r if r > 0 else 0
                        if pnl_r >= 1.0:
                            pos["trailing_sl"] = max(pos["trailing_sl"], trail)
                        eff_sl = max(sl, pos["trailing_sl"])
                        hit_sl = low_i <= eff_sl
                        if hit_sl:
                            sl = eff_sl
                    else:
                        trail = float(np.min(lo15[max(0, i - 22):i + 1])) + 3.0 * ce_atr_val
                        pnl_r = (entry - close_i) / r if r > 0 else 0
                        if pnl_r >= 1.0:
                            pos["trailing_sl"] = min(pos["trailing_sl"], trail) if pos["trailing_sl"] > 0 else trail
                        eff_sl = pos["trailing_sl"] if pos["trailing_sl"] > 0 else sl
                        if eff_sl > 0:
                            eff_sl = min(sl, eff_sl)
                        hit_sl = high_i >= eff_sl if eff_sl > 0 else high_i >= sl
                        if hit_sl:
                            sl = eff_sl if eff_sl > 0 else sl

                else:
                    # Density / MR: simple SL/TP
                    if tp > 0:
                        hit_tp = (d == "LONG" and high_i >= tp) or (d == "SHORT" and low_i <= tp)

                # Exit
                if hit_sl or hit_tp:
                    exit_p = tp if hit_tp else sl
                    if d == "LONG":
                        pnl = (exit_p - entry) * q
                    else:
                        pnl = (entry - exit_p) * q
                    fee = exit_p * q * (TAKER_FEE + SLIPPAGE)
                    net = pnl - fee
                    balance += net
                    all_trades.append(net)
                    pa = per_agent[agent]
                    pa["trades"].append(net)
                    pa["pnl"] += net
                    if net > 0:
                        pa["wins"] += 1
                    else:
                        pa["losses"] += 1
                        if agent == "mr":
                            mr_consec[d] = mr_consec.get(d, 0) + 1
                            if mr_consec[d] >= P["mr_max_consec"]:
                                mr_cooldown[d] = P["mr_cluster_cooldown"]
                    if agent == "mr" and net > 0:
                        mr_consec[d] = 0  # reset on win

                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    position = None
                    continue

                continue  # still in position

            # ── No position: try agents ──
            # Portfolio limits
            # (simplified: 1 position per symbol tracked above)

            for agent in agents_to_run:
                if not use_regime:
                    size_m = 1.0
                else:
                    if not rs.trade_ok.get(agent, False):
                        per_agent[agent]["rejected"] += 1
                        continue
                    size_m = rs.size_mult.get(agent, 0.0)
                    if size_m < 0.05:
                        per_agent[agent]["rejected"] += 1
                        continue

                signal = None

                if agent == "fvg":
                    signal = _fvg_signal(
                        df_15m, i, cl15, hi15, lo15, vo15, ts15,
                        cl1h, ts1h, ema200_1h, cl4h, ts4h, ema200_4h,
                        adx_15m, P)
                    if signal:
                        signal["trailing_sl"] = signal["sl"]
                        signal["partial_taken"] = False

                elif agent == "bbs" and squeeze_ind is not None:
                    sig = detect_squeeze_signal(
                        squeeze_ind, bar_idx=i,
                        min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                        adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                        vol_mult=cfg.BB_SQUEEZE_VOL_MULT)
                    if sig:
                        htf_dir = "LONG" if cl1h[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                        if sig["direction"] == htf_dir and not check_head_fake(squeeze_ind, i, 2):
                            if sig["adx"] <= P["bbs_max_adx"]:
                                ce_sl = chandelier_exit_sl(
                                    hi15, lo15, squeeze_ind["ce_atr"], sig["direction"], i,
                                    ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                                risk = abs(close_i - ce_sl)
                                if 0 < risk / close_i <= 0.05:
                                    signal = {
                                        "direction": sig["direction"],
                                        "entry_price": float(close_i),
                                        "sl": float(ce_sl), "risk": float(risk),
                                        "trailing_sl": float(ce_sl),
                                        "partial_taken": False, "bar_idx": i,
                                    }

                elif agent == "density":
                    vp_start = max(0, i - P["density_vp_lookback"])
                    vp = compute_volume_profile(hi15, lo15, cl15, vo15,
                                                 num_bins=cfg.DENSITY_VP_NUM_BINS,
                                                 start_idx=vp_start, end_idx=i + 1)
                    sig = detect_density_signal(cl15, hi15, lo15, vo15, i,
                                                vp, d_vwap, d_rsi, d_adx, d_atr)
                    if sig:
                        signal = {
                            "direction": sig["direction"],
                            "entry_price": sig["entry_price"],
                            "sl": sig["sl"],
                            "risk": abs(sig["entry_price"] - sig["sl"]),
                            "tp": sig["tp1"], "bar_idx": i,
                        }

                elif agent == "mr":
                    sig = _mr_signal_v2(cl15, hi15, lo15, vo15, i, mr_vwap, P,
                                        mr_consec, mr_cooldown)
                    if sig:
                        signal = sig

                if signal is None:
                    continue

                # Position sizing with soft multiplier
                agent_bal = balance * frac_map[agent]
                base_risk = risk_map[agent]
                effective_risk = base_risk * size_m
                risk_usd = agent_bal * effective_risk
                risk_dist = signal["risk"]
                if risk_dist <= 0:
                    continue
                qty = risk_usd / risk_dist
                margin = signal["entry_price"] * qty / MAX_LEVERAGE
                if margin > agent_bal * 0.3:
                    qty = agent_bal * 0.3 * MAX_LEVERAGE / signal["entry_price"]
                if qty * signal["entry_price"] < 5.0:
                    continue

                fee = signal["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
                balance -= fee

                position = {
                    "agent": agent,
                    "direction": signal["direction"],
                    "entry": signal["entry_price"],
                    "sl": signal["sl"],
                    "tp": signal.get("tp", 0),
                    "risk": signal["risk"],
                    "qty": qty,
                    "bar_idx": i,
                    "trailing_sl": signal.get("trailing_sl"),
                    "partial_taken": signal.get("partial_taken", False),
                }
                break  # one trade at a time per symbol

    # ── Stats ──
    total_trades = sum(pa["wins"] + pa["losses"] for pa in per_agent.values())
    total_wins = sum(pa["wins"] for pa in per_agent.values())
    total_losses = sum(pa["losses"] for pa in per_agent.values())
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100.0
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0

    win_t = [t for t in all_trades if t > 0]
    loss_t = [t for t in all_trades if t <= 0]
    gp = sum(win_t)
    gl = abs(sum(loss_t))
    pf = gp / gl if gl > 0 else float("inf") if gp > 0 else 0

    per_agent_out = {}
    for a, pa in per_agent.items():
        t = pa["wins"] + pa["losses"]
        w_t = [x for x in pa["trades"] if x > 0]
        l_t = [x for x in pa["trades"] if x <= 0]
        gp_a = sum(w_t)
        gl_a = abs(sum(l_t))
        per_agent_out[a] = {
            "trades": t,
            "wins": pa["wins"],
            "losses": pa["losses"],
            "wr_pct": round(pa["wins"] / t * 100, 1) if t > 0 else 0,
            "pnl_usd": round(pa["pnl"], 2),
            "profit_factor": round(gp_a / gl_a, 2) if gl_a > 0 else (float("inf") if gp_a > 0 else 0),
            "rejected": pa["rejected"],
        }

    return {
        "trades": total_trades, "wins": total_wins, "losses": total_losses,
        "wr_pct": round(wr, 1),
        "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "final_balance": round(balance, 2),
        "per_agent": per_agent_out,
        "regime_distribution": dict(regime_dist),
        "regime_switches": regime_switches,
        "params": {k: v for k, v in P.items() if not callable(v)},
    }


def _fmt(r: dict, label: str = "") -> str:
    return (f"{label:<30} Trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
            f"PnL={r['pnl_pct']:>+7.2f}% MaxDD={r['max_dd_pct']:>6.2f}% "
            f"PF={r['profit_factor']:>5.2f} Bal=${r['final_balance']:>.2f}")


# ═══════════════════════════════════════════════════════════════════════════════
#  Parameter optimization
# ═══════════════════════════════════════════════════════════════════════════════

def run_optimization(base_params: dict) -> list:
    """
    Staged parameter sweep. Returns leaderboard of (params, result) tuples.
    FVG is SKIPPED in sweep stages (too slow ~15min/run) and only included
    in baseline + final combo validation.
    """
    import math
    leaderboard = []

    def _score(r):
        return r["pnl_pct"] - 0.5 * r["max_dd_pct"] + 0.3 * math.log(r["trades"] + 1)

    # Stage 0: baseline with and without regime (skip FVG for speed)
    logger.info("=== STAGE 0: Baseline (skip_fvg for speed) ===")
    r_base = run_portfolio_backtest(base_params, use_regime=True, skip_fvg=True)
    logger.info(_fmt(r_base, "BASELINE_noFVG"))
    leaderboard.append(("baseline_nofvg", base_params.copy(), r_base))

    r_noreg = run_portfolio_backtest(base_params, use_regime=False, skip_fvg=True)
    logger.info(_fmt(r_noreg, "NO_REGIME_noFVG"))
    leaderboard.append(("no_regime_nofvg", base_params.copy(), r_noreg))

    # Stage 1: Regime tuning
    logger.info("=== STAGE 1: Regime tuning ===")
    regime_variants = [
        {"regime_confirm_bars": 3, "regime_min_hold": 10, "regime_smooth_alpha": 0.40, "regime_transition_margin": 0.6},
        {"regime_confirm_bars": 5, "regime_min_hold": 15, "regime_smooth_alpha": 0.35, "regime_transition_margin": 0.8},
        {"regime_confirm_bars": 7, "regime_min_hold": 20, "regime_smooth_alpha": 0.25, "regime_transition_margin": 1.0},
        {"regime_confirm_bars": 5, "regime_min_hold": 12, "regime_smooth_alpha": 0.30, "regime_transition_margin": 0.5},
        {"regime_confirm_bars": 4, "regime_min_hold": 18, "regime_smooth_alpha": 0.20, "regime_transition_margin": 0.7},
    ]
    for vi, rv in enumerate(regime_variants):
        p = {**base_params, **rv}
        r = run_portfolio_backtest(p, use_regime=True, skip_fvg=True)
        label = f"regime_v{vi}"
        logger.info(_fmt(r, label))
        leaderboard.append((label, p, r))

    # Stage 2: BBS tuning
    logger.info("=== STAGE 2: BBS tuning ===")
    bbs_variants = [
        {"bbs_partial_tp_r": 1.5, "bbs_partial_frac": 0.5, "bbs_max_adx": 40.0, "bbs_risk": 0.007},
        {"bbs_partial_tp_r": 2.0, "bbs_partial_frac": 0.5, "bbs_max_adx": 45.0, "bbs_risk": 0.009},
        {"bbs_partial_tp_r": 2.5, "bbs_partial_frac": 0.4, "bbs_max_adx": 50.0, "bbs_risk": 0.008},
        {"bbs_partial_tp_r": 0, "bbs_partial_frac": 0.0, "bbs_max_adx": 50.0, "bbs_risk": 0.008},  # pure trailing
    ]
    for vi, bv in enumerate(bbs_variants):
        p = {**base_params, **bv}
        r = run_portfolio_backtest(p, use_regime=True, skip_fvg=True)
        label = f"bbs_v{vi}"
        logger.info(_fmt(r, label))
        leaderboard.append((label, p, r))

    # Stage 3: MR tuning
    logger.info("=== STAGE 3: MR tuning ===")
    mr_variants = [
        {"mr_rsi_os": 25.0, "mr_rsi_ob": 75.0, "mr_adx_max": 20.0, "mr_sl_atr": 2.0, "mr_vwap_dev": 0.4},
        {"mr_rsi_os": 30.0, "mr_rsi_ob": 70.0, "mr_adx_max": 25.0, "mr_sl_atr": 1.5, "mr_vwap_dev": 0.6},
        {"mr_rsi_os": 28.0, "mr_rsi_ob": 72.0, "mr_adx_max": 22.0, "mr_sl_atr": 2.0, "mr_vwap_dev": 0.5},
        {"mr_risk": 0.003, "mr_frac": 0.10},  # reduce MR weight
        {"mr_risk": 0.006, "mr_frac": 0.15, "mr_adx_max": 24.0, "mr_sl_atr": 1.8},
    ]
    for vi, mv in enumerate(mr_variants):
        p = {**base_params, **mv}
        r = run_portfolio_backtest(p, use_regime=True, skip_fvg=True)
        label = f"mr_v{vi}"
        logger.info(_fmt(r, label))
        leaderboard.append((label, p, r))

    # Stage 4: Risk/allocation tuning
    logger.info("=== STAGE 4: Risk / allocation ===")
    alloc_variants = [
        {"fvg_frac": 0.30, "bbs_frac": 0.30, "density_frac": 0.25, "mr_frac": 0.15},
        {"fvg_frac": 0.35, "bbs_frac": 0.25, "density_frac": 0.30, "mr_frac": 0.10},
        {"fvg_frac": 0.25, "bbs_frac": 0.30, "density_frac": 0.30, "mr_frac": 0.15},
        {"density_risk": 0.010, "bbs_risk": 0.009, "mr_risk": 0.004},
        {"density_risk": 0.007, "bbs_risk": 0.007, "mr_risk": 0.006},
    ]
    for vi, av in enumerate(alloc_variants):
        p = {**base_params, **av}
        r = run_portfolio_backtest(p, use_regime=True, skip_fvg=True)
        label = f"alloc_v{vi}"
        logger.info(_fmt(r, label))
        leaderboard.append((label, p, r))

    # Stage 5: Density tuning
    logger.info("=== STAGE 5: Density tuning ===")
    density_variants = [
        {"density_rsi_os": 30.0, "density_rsi_ob": 70.0, "density_adx_max": 25.0, "density_vwap_dev": 0.6},
        {"density_rsi_os": 38.0, "density_rsi_ob": 62.0, "density_adx_max": 20.0, "density_vwap_dev": 1.0},
        {"density_sl_atr": 2.0, "density_vwap_dev": 0.7, "density_adx_max": 24.0},
    ]
    for vi, dv in enumerate(density_variants):
        p = {**base_params, **dv}
        r = run_portfolio_backtest(p, use_regime=True, skip_fvg=True)
        label = f"density_v{vi}"
        logger.info(_fmt(r, label))
        leaderboard.append((label, p, r))

    # Stage 6: Combine best from each stage
    logger.info("=== STAGE 6: Best combination (with FVG — full run) ===")
    best_by_stage = {}
    for prefix in ("regime_", "bbs_", "mr_", "alloc_", "density_"):
        candidates = [x for x in leaderboard if x[0].startswith(prefix)]
        if candidates:
            best_by_stage[prefix] = max(candidates, key=lambda x: _score(x[2]))

    combo = base_params.copy()
    for prefix, (label, p, r) in best_by_stage.items():
        for k, v in p.items():
            if k in combo:
                combo[k] = v
        logger.info(f"  Best {prefix}: {label} score={_score(r):.2f}")

    # Full run WITH FVG
    r_combo = run_portfolio_backtest(combo, use_regime=True)
    logger.info(_fmt(r_combo, "BEST_COMBO (with FVG)"))
    leaderboard.append(("best_combo", combo, r_combo))

    # Also run baseline with FVG for comparison
    r_base_full = run_portfolio_backtest(base_params, use_regime=True)
    logger.info(_fmt(r_base_full, "BASELINE (with FVG)"))
    leaderboard.append(("baseline_full", base_params.copy(), r_base_full))

    # Stage 7: Walk-forward validation
    logger.info("=== STAGE 7: Walk-forward validation ===")
    r_train = run_portfolio_backtest(combo, use_regime=True, start_frac=0.0, end_frac=0.67, skip_fvg=True)
    r_test = run_portfolio_backtest(combo, use_regime=True, start_frac=0.67, end_frac=1.0, skip_fvg=True)
    logger.info(_fmt(r_train, "COMBO train(60d)"))
    logger.info(_fmt(r_test, "COMBO test(30d)"))
    leaderboard.append(("combo_train", combo, r_train))
    leaderboard.append(("combo_test", combo, r_test))

    r_base_train = run_portfolio_backtest(base_params, use_regime=True, start_frac=0.0, end_frac=0.67, skip_fvg=True)
    r_base_test = run_portfolio_backtest(base_params, use_regime=True, start_frac=0.67, end_frac=1.0, skip_fvg=True)
    logger.info(_fmt(r_base_train, "BASE train(60d)"))
    logger.info(_fmt(r_base_test, "BASE test(30d)"))
    leaderboard.append(("base_train", base_params.copy(), r_base_train))
    leaderboard.append(("base_test", base_params.copy(), r_base_test))

    # Final leaderboard
    full_entries = [x for x in leaderboard if not x[0].endswith("_train") and not x[0].endswith("_test")]
    full_entries.sort(key=lambda x: _score(x[2]), reverse=True)

    logger.info("\n=== LEADERBOARD (top 10) ===")
    for rank, (label, p, r) in enumerate(full_entries[:10], 1):
        logger.info(f"#{rank} {_fmt(r, label)} score={_score(r):.2f}")

    return leaderboard


# ═══════════════════════════════════════════════════════════════════════════════
#  Standalone agent tests
# ═══════════════════════════════════════════════════════════════════════════════

def run_standalone_tests(params: dict, include_fvg: bool = False) -> dict:
    """Run each agent standalone (regime-gated and no-regime).
    FVG is skipped by default due to ~15min runtime."""
    results = {}
    agents = ["fvg", "bbs", "density", "mr"] if include_fvg else ["bbs", "density", "mr"]
    for agent in agents:
        logger.info(f"--- {agent.upper()} standalone (soft regime) ---")
        t0 = _time.time()
        r = run_portfolio_backtest(params, use_regime=True, agent_filter=agent)
        logger.info(_fmt(r, f"{agent.upper()}_REGIME") + f" ({_time.time()-t0:.0f}s)")
        results[f"{agent}_regime"] = r

        logger.info(f"--- {agent.upper()} standalone (no regime) ---")
        t0 = _time.time()
        r = run_portfolio_backtest(params, use_regime=False, agent_filter=agent)
        logger.info(_fmt(r, f"{agent.upper()}_NOREG") + f" ({_time.time()-t0:.0f}s)")
        results[f"{agent}_noreg"] = r

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 100)
    logger.info("SOFT-REGIME PORTFOLIO BACKTEST v2 — 20 Symbols — 90 Days")
    logger.info("=" * 100)

    P = baseline_params()

    # 1. Standalone agent tests (skip FVG — too slow, run separately)
    logger.info("\n### PHASE 1: Standalone Agent Tests (no FVG) ###")
    standalone = run_standalone_tests(P, include_fvg=False)

    # 2. Optimization
    logger.info("\n### PHASE 2: Portfolio Optimization ###")
    leaderboard = run_optimization(P)

    # 3. Print final report
    print("\n" + "=" * 110)
    print("SOFT-REGIME PORTFOLIO BACKTEST v2 — FINAL REPORT")
    print("=" * 110)

    print("\n--- Standalone Agent Results ---")
    header = f"{'Config':<30} {'Trades':>6} {'WR%':>6} {'PnL%':>9} {'MaxDD%':>8} {'PF':>6}"
    print(header)
    print("-" * 80)
    for key, r in standalone.items():
        label = key.replace("_", " ").upper()
        print(f"{label:<30} {r['trades']:>6} {r['wr_pct']:>5.1f}% {r['pnl_pct']:>+8.2f}% "
              f"{r['max_dd_pct']:>7.2f}% {r['profit_factor']:>5.2f}")

    print("\n--- Optimization Leaderboard (top 10) ---")
    print(header)
    print("-" * 80)

    def _score(r):
        import math
        return r["pnl_pct"] - 0.5 * r["max_dd_pct"] + 0.3 * math.log(r["trades"] + 1)

    full = [x for x in leaderboard if not x[0].endswith("_train") and not x[0].endswith("_test")]
    full.sort(key=lambda x: _score(x[2]), reverse=True)
    for label, p, r in full[:10]:
        print(f"{label:<30} {r['trades']:>6} {r['wr_pct']:>5.1f}% {r['pnl_pct']:>+8.2f}% "
              f"{r['max_dd_pct']:>7.2f}% {r['profit_factor']:>5.2f}")

    # Per-agent breakdown of best
    if full:
        best_label, best_params, best_result = full[0]
        print(f"\n--- Best Configuration: {best_label} ---")
        print(f"  Trades={best_result['trades']} WR={best_result['wr_pct']}% "
              f"PnL={best_result['pnl_pct']:+.2f}% MaxDD={best_result['max_dd_pct']:.2f}% "
              f"PF={best_result['profit_factor']:.2f}")
        print(f"  Regime switches: {best_result.get('regime_switches', '?')}")
        print(f"  Regime distribution: {best_result.get('regime_distribution', {})}")
        print("  Per-agent:")
        for a, s in best_result.get("per_agent", {}).items():
            print(f"    {a.upper():<12} T={s['trades']:>3} WR={s['wr_pct']:>5.1f}% "
                  f"PnL=${s['pnl_usd']:>+8.2f} PF={s['profit_factor']:>5.2f} rej={s['rejected']}")

        # Walk-forward
        wf = [x for x in leaderboard if x[0] in ("combo_train", "combo_test")]
        if wf:
            print("\n--- Walk-Forward Validation (best combo) ---")
            for label, _, r in wf:
                period = "Train (60d)" if "train" in label else "Test (30d)"
                print(f"  {period:<15} T={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                      f"PnL={r['pnl_pct']:>+7.2f}% DD={r['max_dd_pct']:>6.2f}% PF={r['profit_factor']:>5.2f}")

        # Save best params
        print(f"\n--- Best Params ---")
        key_params = {k: v for k, v in best_params.items()
                      if k in ("fvg_composite_lo", "fvg_composite_hi", "fvg_quality_min",
                               "fvg_adx_min", "fvg_entry_lo", "fvg_entry_hi",
                               "bbs_partial_tp_r", "bbs_max_adx",
                               "mr_rsi_os", "mr_adx_max", "mr_sl_atr", "mr_vwap_dev",
                               "regime_confirm_bars", "regime_min_hold", "regime_smooth_alpha",
                               "regime_transition_margin",
                               "fvg_frac", "bbs_frac", "density_frac", "mr_frac",
                               "fvg_risk", "bbs_risk", "density_risk", "mr_risk")}
        for k, v in sorted(key_params.items()):
            print(f"  {k}: {v}")

    # Save everything
    save_data = {
        "standalone": standalone,
        "leaderboard": [(l, {k: v for k, v in p.items() if not callable(v)}, r)
                        for l, p, r in leaderboard],
        "best": full[0] if full else None,
    }
    path = RESULTS_DIR / f"v2_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")

    return save_data


if __name__ == "__main__":
    main()
