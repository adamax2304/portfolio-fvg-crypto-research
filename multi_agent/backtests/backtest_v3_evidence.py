"""
multi_agent/backtests/backtest_v3_evidence.py — Evidence-Based Portfolio Rebuild

FORENSIC FINDINGS (Phase 1 evidence):

BBS:
  - BEST in TREND (81 trades, 49.4% WR, +19.43R), WORST in SQUEEZE (22t, 27.3%, -9.15R)
  - Regime gating HURTS: +21.57R without regime vs +11.47R with regime
  - 90.6% of trades reach +1R but 45.3% give it back (exit problem, not entry)
  - Fixed 3R exit: +49R (best). Chandelier trail: +32.64R. Partial 1.5R+trail: +26.36R
  - Stop: 47.9% too wide. Only 1.7% too tight. Stops OK but trailing gives back too much.

MR:
  - POSITIVE without regime (+3.14R), NEGATIVE with regime (-5.77R)
  - Regime rejects 86% of signals — over-filtering destroys the strategy
  - BEST in TRANSITION (14t, 42.9%, +5.17R) and even TREND (55t, 34.5%, +1.20R)
  - 67.6% reach +1R but end as losers — massive monetization failure
  - BB midline exit: +15.05R (best). Fixed 1R: +14R. Chandelier: +17.03R
  - Current BB midline TP is actually correct, problem is stops too wide (60.5% MAE>2R)

DENSITY:
  - EXCLUDED. No edge. Negative in every regime, every exit scheme. MAE > MFE.

PORTFOLIO REDESIGN:
  1. BBS regime: flip — TREND=1.0, TRANSITION=0.85, SQUEEZE=0.5, RANGE=0
  2. BBS exit: test fixed-R and partial+trail vs pure chandelier
  3. MR regime: widen — TRANSITION=1.0, RANGE=0.85, TREND=0.3, SQUEEZE=0
  4. MR exit: keep BB midline TP, tighten stops
  5. No density.
"""
from __future__ import annotations
import json, logging, time as _time, math
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
    _ema, _sma, _stdev, _atr, _adx,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
from multi_agent.agents.mean_reversion_agent import (
    compute_mr_indicators, detect_mr_signal,
)
import multi_agent.config as cfg

logger = logging.getLogger("bt_v3")
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


def _load(symbol, tf):
    safe = symbol.replace("/", "_").replace(":", "_")
    for ext in (".parquet", ".csv"):
        fpath = CACHE_DIR / f"{safe}_{tf}_90d{ext}"
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath) if ext == ".parquet" else pd.read_csv(fpath)
                if df is None or df.empty: continue
                for c in ("open", "high", "low", "close", "volume"):
                    if c in df.columns: df[c] = df[c].astype(float)
                if "timestamp" in df.columns:
                    df["timestamp"] = df["timestamp"].astype(int)
                return df
            except Exception: continue
    return None


def run_v3(params: dict, agents_enabled: list = None,
           start_frac: float = 0.0, end_frac: float = 1.0) -> dict:
    """
    Evidence-based portfolio backtest v3.
    Key changes from v2:
      - BBS regime flipped (best in TREND, not SQUEEZE)
      - BBS exit: configurable (fixed R, partial+trail, or chandelier)
      - MR regime widened (TRANSITION + RANGE, not just RANGE)
      - MR exit: BB midline TP (proven best by forensic)
      - No density
    """
    P = params
    balance = START_BALANCE
    peak = START_BALANCE
    max_dd = 0.0
    all_trades = []

    if agents_enabled is None:
        agents_enabled = ["bbs", "mr"]

    per_agent = {a: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []}
                 for a in agents_enabled}

    soft_cfg = {
        "confirm_bars": P.get("regime_confirm_bars", 5),
        "min_hold_bars": P.get("regime_min_hold", 12),
        "smooth_alpha": P.get("regime_smooth_alpha", 0.30),
        "min_confidence": P.get("regime_min_confidence", 1.0),
        "transition_margin": P.get("regime_transition_margin", 0.5),
        "undefined_threshold": P.get("regime_undefined_threshold", 0.5),
    }

    # EVIDENCE-BASED regime multipliers (flipped for BBS!)
    bbs_mult = {
        "TREND": P.get("bbs_mult_trend", 1.0),          # was 0.3 — forensic shows +19.43R here
        "TRANSITION": P.get("bbs_mult_transition", 0.85),
        "SQUEEZE": P.get("bbs_mult_squeeze", 0.5),       # was 1.0 — forensic shows -9.15R here!
        "RANGE": P.get("bbs_mult_range", 0.0),
        "UNDEFINED": 0.0,
    }
    mr_mult = {
        "TRANSITION": P.get("mr_mult_transition", 1.0),  # forensic: +5.17R
        "RANGE": P.get("mr_mult_range", 0.85),
        "TREND": P.get("mr_mult_trend", 0.3),            # forensic: +1.20R
        "SQUEEZE": 0.0,
        "UNDEFINED": 0.0,
    }

    risk_map = {"bbs": P.get("bbs_risk", 0.009), "mr": P.get("mr_risk", 0.005)}
    frac_map = {"bbs": P.get("bbs_frac", 0.70), "mr": P.get("mr_frac", 0.30)}

    # BBS exit mode: "chandelier", "fixed_NR", "partial_NR_trail"
    bbs_exit_mode = P.get("bbs_exit_mode", "partial_1.5R_trail")
    bbs_fixed_r = P.get("bbs_fixed_r", 3.0)
    bbs_partial_r = P.get("bbs_partial_r", 1.5)
    bbs_partial_frac = P.get("bbs_partial_frac", 0.5)

    for symbol in BACKTEST_SYMBOLS:
        df_15m = _load(symbol, "15m")
        df_1h = _load(symbol, "1h")
        df_4h = _load(symbol, "4h")
        if df_15m is None or df_1h is None: continue
        if len(df_15m) < 250 or len(df_1h) < 210: continue

        n_total = len(df_15m)
        i_start = max(250, int(n_total * start_frac))
        i_end = int(n_total * end_frac)

        cl15 = df_15m["close"].astype(float).to_numpy()
        hi15 = df_15m["high"].astype(float).to_numpy()
        lo15 = df_15m["low"].astype(float).to_numpy()
        vo15 = df_15m["volume"].astype(float).to_numpy()
        ts15 = df_15m["timestamp"].astype(int).to_numpy()
        ts1h = df_1h["timestamp"].astype(int).to_numpy()

        ind_1h = precompute_1h_indicators(df_1h)
        ema200_1h = ind_1h["ema200_1h"]
        cl1h = ind_1h["closes_1h_arr"]
        ind_4h = precompute_4h_indicators(df_4h) if df_4h is not None and len(df_4h) > 50 else None

        squeeze_ind = compute_squeeze_indicators(df_15m) if "bbs" in agents_enabled else None
        mr_ind = compute_mr_indicators(df_15m) if "mr" in agents_enabled else None
        atr_15m = _atr(hi15, lo15, cl15, 14)

        detector = SoftRegimeDetector(soft_cfg)
        position = None
        mr_consec = {"LONG": 0, "SHORT": 0}

        for i in range(i_start, i_end):
            close_i = cl15[i]
            high_i = hi15[i]
            low_i = lo15[i]

            htf_idx = int(np.searchsorted(ts1h, int(ts15[i]), side="right")) - 1
            if htf_idx < 210: continue

            rs = detector.update(symbol, df_1h, htf_idx, df_4h,
                                 **(ind_1h), **(ind_4h or {}))
            regime_str = rs.regime.value

            # ── Position management ──
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
                exit_p = sl

                if agent == "bbs":
                    pnl_r_now = ((close_i - entry) / r) if d == "LONG" else ((entry - close_i) / r)

                    if bbs_exit_mode.startswith("fixed_"):
                        # Fixed R target
                        if d == "LONG":
                            tp = entry + bbs_fixed_r * r
                            hit_tp = high_i >= tp
                        else:
                            tp = entry - bbs_fixed_r * r
                            hit_tp = low_i <= tp
                        if hit_tp: exit_p = tp

                    elif bbs_exit_mode.startswith("partial_"):
                        # Partial TP at partial_r, trail remainder
                        if not pos.get("partial_taken") and pnl_r_now >= bbs_partial_r:
                            pq = q * bbs_partial_frac
                            pnl_partial = ((close_i - entry) if d == "LONG" else (entry - close_i)) * pq
                            fee = close_i * pq * (TAKER_FEE + SLIPPAGE)
                            balance += pnl_partial - fee
                            pos["qty"] -= pq
                            pos["partial_taken"] = True
                            q = pos["qty"]

                        # Chandelier trailing for remainder
                        if pnl_r_now >= 1.0 and squeeze_ind is not None:
                            new_ce = chandelier_exit_sl(
                                hi15, lo15, squeeze_ind["ce_atr"], d, i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            if d == "LONG":
                                pos["trailing_sl"] = max(pos.get("trailing_sl", sl), new_ce)
                            else:
                                ts_val = pos.get("trailing_sl", sl)
                                pos["trailing_sl"] = min(ts_val, new_ce) if ts_val > 0 else new_ce

                        eff_sl = pos.get("trailing_sl", sl)
                        if d == "LONG":
                            eff_sl = max(sl, eff_sl)
                            hit_sl = low_i <= eff_sl
                            if hit_sl: exit_p = eff_sl
                        else:
                            if eff_sl > 0: eff_sl = min(sl, eff_sl)
                            hit_sl = high_i >= eff_sl if eff_sl > 0 else high_i >= sl
                            if hit_sl: exit_p = eff_sl if eff_sl > 0 else sl

                        # Cap at 4R
                        if pnl_r_now >= 4.0:
                            hit_tp = True
                            exit_p = close_i

                    else:  # chandelier
                        if pnl_r_now >= 1.0 and squeeze_ind is not None:
                            new_ce = chandelier_exit_sl(
                                hi15, lo15, squeeze_ind["ce_atr"], d, i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            if d == "LONG":
                                pos["trailing_sl"] = max(pos.get("trailing_sl", sl), new_ce)
                            else:
                                ts_val = pos.get("trailing_sl", sl)
                                pos["trailing_sl"] = min(ts_val, new_ce) if ts_val > 0 else new_ce

                        eff_sl = pos.get("trailing_sl", sl)
                        if d == "LONG":
                            eff_sl = max(sl, eff_sl)
                            hit_sl = low_i <= eff_sl
                            if hit_sl: exit_p = eff_sl
                        else:
                            if eff_sl > 0: eff_sl = min(sl, eff_sl)
                            hit_sl = high_i >= eff_sl if eff_sl > 0 else high_i >= sl
                            if hit_sl: exit_p = eff_sl if eff_sl > 0 else sl

                        if pnl_r_now >= 4.0:
                            hit_tp = True
                            exit_p = close_i

                elif agent == "mr":
                    # MR: BB midline TP (forensic-proven best)
                    tp = pos.get("tp", 0)
                    if tp > 0:
                        if d == "LONG": hit_tp = high_i >= tp
                        else: hit_tp = low_i <= tp
                        if hit_tp: exit_p = tp

                if hit_sl or hit_tp:
                    if not hit_tp: exit_p = sl if exit_p == 0 else exit_p
                    pnl = ((exit_p - entry) if d == "LONG" else (entry - exit_p)) * q
                    fee = exit_p * q * (TAKER_FEE + SLIPPAGE)
                    net = pnl - fee
                    balance += net
                    all_trades.append(net)
                    pa = per_agent[agent]
                    pa["trades"].append(net)
                    pa["pnl"] += net
                    if net > 0: pa["wins"] += 1
                    else: pa["losses"] += 1
                    if agent == "mr":
                        if net <= 0: mr_consec[d] = mr_consec.get(d, 0) + 1
                        else: mr_consec[d] = 0
                    peak = max(peak, balance)
                    dd = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    position = None
                    continue
                continue

            # ── Signal search ──
            for agent in agents_enabled:
                if agent == "bbs":
                    size_m = bbs_mult.get(regime_str, 0.0)
                elif agent == "mr":
                    size_m = mr_mult.get(regime_str, 0.0)
                else:
                    continue

                if size_m < 0.05: continue

                signal = None

                if agent == "bbs" and squeeze_ind is not None:
                    sig = detect_squeeze_signal(
                        squeeze_ind, bar_idx=i,
                        min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                        adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                        vol_mult=cfg.BB_SQUEEZE_VOL_MULT)
                    if sig:
                        htf_dir = "LONG" if cl1h[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                        if sig["direction"] == htf_dir and not check_head_fake(squeeze_ind, i, 2):
                            ce_sl = chandelier_exit_sl(
                                hi15, lo15, squeeze_ind["ce_atr"], sig["direction"], i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH, ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            risk_v = abs(close_i - ce_sl)
                            if 0 < risk_v / close_i <= 0.05:
                                signal = {"direction": sig["direction"],
                                          "entry_price": float(close_i),
                                          "sl": float(ce_sl), "risk": float(risk_v),
                                          "tp": 0}

                elif agent == "mr" and mr_ind is not None:
                    sig = detect_mr_signal(mr_ind, i, mr_consec)
                    if sig:
                        signal = sig

                if signal is None: continue

                # Sizing
                agent_bal = balance * frac_map.get(agent, 0.30)
                base_risk = risk_map.get(agent, 0.008)
                effective_risk = base_risk * size_m
                risk_usd = agent_bal * effective_risk
                risk_dist = signal.get("risk", abs(signal["entry_price"] - signal.get("sl", 0)))
                if risk_dist <= 0: continue
                qty = risk_usd / risk_dist
                margin = signal["entry_price"] * qty / MAX_LEVERAGE
                if margin > agent_bal * 0.3:
                    qty = agent_bal * 0.3 * MAX_LEVERAGE / signal["entry_price"]
                if qty * signal["entry_price"] < 5.0: continue

                fee = signal["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
                balance -= fee

                position = {
                    "agent": agent, "direction": signal["direction"],
                    "entry": signal["entry_price"], "sl": signal.get("sl", 0),
                    "tp": signal.get("tp", 0), "risk": risk_dist,
                    "qty": qty, "bar_idx": i,
                    "trailing_sl": signal.get("sl", 0), "partial_taken": False,
                }
                break

    # Stats
    total_trades = sum(pa["wins"] + pa["losses"] for pa in per_agent.values())
    total_wins = sum(pa["wins"] for pa in per_agent.values())
    pnl_pct = (balance - START_BALANCE) / START_BALANCE * 100.0
    wr = total_wins / total_trades * 100 if total_trades > 0 else 0
    win_t = [t for t in all_trades if t > 0]
    loss_t = [t for t in all_trades if t <= 0]
    gp = sum(win_t); gl = abs(sum(loss_t))
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0)

    per_agent_out = {}
    for a, pa in per_agent.items():
        t = pa["wins"] + pa["losses"]
        w_t = [x for x in pa["trades"] if x > 0]
        l_t = [x for x in pa["trades"] if x <= 0]
        gp_a = sum(w_t); gl_a = abs(sum(l_t))
        per_agent_out[a] = {
            "trades": t, "wins": pa["wins"], "losses": pa["losses"],
            "wr_pct": round(pa["wins"] / t * 100, 1) if t > 0 else 0,
            "pnl_usd": round(pa["pnl"], 2),
            "pf": round(gp_a / gl_a, 2) if gl_a > 0 else 0,
        }

    return {
        "trades": total_trades, "wins": total_wins,
        "losses": total_trades - total_wins,
        "wr_pct": round(wr, 1), "pnl_pct": round(pnl_pct, 2),
        "pnl_usd": round(balance - START_BALANCE, 2),
        "max_dd_pct": round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "final_balance": round(balance, 2),
        "per_agent": per_agent_out,
    }


def _fmt(r, label=""):
    return (f"{label:<30} T={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
            f"PnL={r['pnl_pct']:>+7.2f}% DD={r['max_dd_pct']:>6.2f}% "
            f"PF={r['profit_factor']:>5.2f} Bal=${r['final_balance']:>.2f}")


def _score(r):
    return r["pnl_pct"] - 0.5 * r["max_dd_pct"] + 0.3 * math.log(r["trades"] + 1)


def main():
    logger.info("=" * 100)
    logger.info("EVIDENCE-BASED PORTFOLIO v3 -- BBS(flipped regime) + MR(widened regime)")
    logger.info("=" * 100)

    results = {}

    # ═══ Test matrix ═══
    test_configs = {
        # Baseline: old regime mapping
        "old_regime_bbs_only": {
            "bbs_mult_trend": 0.3, "bbs_mult_transition": 0.6, "bbs_mult_squeeze": 1.0,
            "bbs_exit_mode": "chandelier", "bbs_frac": 1.0, "mr_frac": 0.0,
        },
        "old_regime_bbs_mr": {
            "bbs_mult_trend": 0.3, "bbs_mult_transition": 0.6, "bbs_mult_squeeze": 1.0,
            "mr_mult_transition": 0.3, "mr_mult_range": 1.0, "mr_mult_trend": 0.0,
            "bbs_exit_mode": "chandelier", "bbs_frac": 0.70, "mr_frac": 0.30,
        },

        # Evidence-based: flipped BBS regime + various exit modes
        "flipped_bbs_chandelier": {
            "bbs_exit_mode": "chandelier",
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },
        "flipped_bbs_fixed3R": {
            "bbs_exit_mode": "fixed_3R", "bbs_fixed_r": 3.0,
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },
        "flipped_bbs_fixed2R": {
            "bbs_exit_mode": "fixed_2R", "bbs_fixed_r": 2.0,
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },
        "flipped_bbs_partial1.5R": {
            "bbs_exit_mode": "partial_1.5R_trail",
            "bbs_partial_r": 1.5, "bbs_partial_frac": 0.5,
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },
        "flipped_bbs_partial1R": {
            "bbs_exit_mode": "partial_1R_trail",
            "bbs_partial_r": 1.0, "bbs_partial_frac": 0.5,
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },

        # No regime at all
        "no_regime_bbs_fixed3R": {
            "bbs_exit_mode": "fixed_3R", "bbs_fixed_r": 3.0,
            "bbs_mult_trend": 1.0, "bbs_mult_transition": 1.0,
            "bbs_mult_squeeze": 1.0, "bbs_mult_range": 1.0,
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },
        "no_regime_bbs_chandelier": {
            "bbs_exit_mode": "chandelier",
            "bbs_mult_trend": 1.0, "bbs_mult_transition": 1.0,
            "bbs_mult_squeeze": 1.0, "bbs_mult_range": 1.0,
            "bbs_frac": 1.0, "mr_frac": 0.0,
        },

        # MR standalone tests
        "widened_mr_only": {
            "bbs_frac": 0.0, "mr_frac": 1.0,
            "mr_mult_transition": 1.0, "mr_mult_range": 0.85,
            "mr_mult_trend": 0.3,
        },
        "old_mr_only": {
            "bbs_frac": 0.0, "mr_frac": 1.0,
            "mr_mult_transition": 0.3, "mr_mult_range": 1.0, "mr_mult_trend": 0.0,
        },
        "no_regime_mr_only": {
            "bbs_frac": 0.0, "mr_frac": 1.0,
            "mr_mult_transition": 1.0, "mr_mult_range": 1.0,
            "mr_mult_trend": 1.0, "mr_mult_squeeze": 1.0,
        },

        # Combined portfolios
        "v3_flipped_bbs_p1.5_mr_wide": {
            "bbs_exit_mode": "partial_1.5R_trail",
            "bbs_partial_r": 1.5, "bbs_partial_frac": 0.5,
            "bbs_frac": 0.70, "mr_frac": 0.30,
            "bbs_risk": 0.009, "mr_risk": 0.005,
        },
        "v3_flipped_bbs_f3r_mr_wide": {
            "bbs_exit_mode": "fixed_3R", "bbs_fixed_r": 3.0,
            "bbs_frac": 0.70, "mr_frac": 0.30,
            "bbs_risk": 0.009, "mr_risk": 0.005,
        },
        "v3_flipped_bbs_chand_mr_wide": {
            "bbs_exit_mode": "chandelier",
            "bbs_frac": 0.70, "mr_frac": 0.30,
            "bbs_risk": 0.009, "mr_risk": 0.005,
        },

        # Higher risk variants
        "v3_aggressive_p1.5": {
            "bbs_exit_mode": "partial_1.5R_trail",
            "bbs_partial_r": 1.5, "bbs_partial_frac": 0.5,
            "bbs_frac": 0.75, "mr_frac": 0.25,
            "bbs_risk": 0.011, "mr_risk": 0.006,
        },
        "v3_aggressive_f3r": {
            "bbs_exit_mode": "fixed_3R", "bbs_fixed_r": 3.0,
            "bbs_frac": 0.75, "mr_frac": 0.25,
            "bbs_risk": 0.011, "mr_risk": 0.006,
        },

        # BBS squeeze penalty (punish SQUEEZE more since forensic shows -9.15R there)
        "v3_squeeze_penalized": {
            "bbs_exit_mode": "partial_1.5R_trail",
            "bbs_partial_r": 1.5, "bbs_partial_frac": 0.5,
            "bbs_mult_squeeze": 0.2,  # heavily reduced
            "bbs_frac": 0.75, "mr_frac": 0.25,
            "bbs_risk": 0.010, "mr_risk": 0.005,
        },
    }

    base_params = {
        "regime_confirm_bars": 5, "regime_min_hold": 12,
        "regime_smooth_alpha": 0.30, "regime_transition_margin": 0.5,
        "regime_undefined_threshold": 0.5, "regime_min_confidence": 1.0,
        "bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85,
        "bbs_mult_squeeze": 0.5, "bbs_mult_range": 0.0,
        "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.3,
        "bbs_risk": 0.009, "mr_risk": 0.005,
        "bbs_frac": 0.70, "mr_frac": 0.30,
        "bbs_exit_mode": "partial_1.5R_trail",
        "bbs_partial_r": 1.5, "bbs_partial_frac": 0.5,
        "bbs_fixed_r": 3.0,
    }

    for name, overrides in test_configs.items():
        p = {**base_params, **overrides}
        agents = []
        if p.get("bbs_frac", 0) > 0: agents.append("bbs")
        if p.get("mr_frac", 0) > 0: agents.append("mr")
        if not agents: continue

        t0 = _time.time()
        r = run_v3(p, agents_enabled=agents)
        elapsed = _time.time() - t0
        results[name] = r
        logger.info(f"{_fmt(r, name)} ({elapsed:.0f}s)")

    # Walk-forward for top 3
    sorted_results = sorted(results.items(), key=lambda x: _score(x[1]), reverse=True)
    top3 = sorted_results[:3]

    logger.info("\n=== WALK-FORWARD VALIDATION (top 3) ===")
    wf_results = {}
    for name, _ in top3:
        overrides = test_configs[name]
        p = {**base_params, **overrides}
        agents = []
        if p.get("bbs_frac", 0) > 0: agents.append("bbs")
        if p.get("mr_frac", 0) > 0: agents.append("mr")

        r_train = run_v3(p, agents_enabled=agents, start_frac=0.0, end_frac=0.67)
        r_test = run_v3(p, agents_enabled=agents, start_frac=0.67, end_frac=1.0)
        wf_results[name] = {"train": r_train, "test": r_test}
        logger.info(f"  {name}")
        logger.info(f"    Train: T={r_train['trades']} PnL={r_train['pnl_pct']:+.2f}% DD={r_train['max_dd_pct']:.2f}%")
        logger.info(f"    Test:  T={r_test['trades']} PnL={r_test['pnl_pct']:+.2f}% DD={r_test['max_dd_pct']:.2f}%")

    # ═══ FINAL REPORT ═══
    print("\n" + "=" * 110)
    print("EVIDENCE-BASED PORTFOLIO v3 -- FULL COMPARISON")
    print("=" * 110)

    header = f"{'Config':<35} {'T':>4} {'WR%':>6} {'PnL%':>8} {'DD%':>7} {'PF':>6} {'Score':>7}"
    print(header)
    print("-" * 85)

    for name, r in sorted_results:
        sc = _score(r)
        print(f"{name:<35} {r['trades']:>4} {r['wr_pct']:>5.1f}% {r['pnl_pct']:>+7.2f}% "
              f"{r['max_dd_pct']:>6.2f}% {r['profit_factor']:>5.2f} {sc:>+6.2f}")
        for a, s in r["per_agent"].items():
            if s["trades"] > 0:
                print(f"   {a.upper():<10} T={s['trades']:>3} WR={s['wr_pct']:>5.1f}% "
                      f"PnL=${s['pnl_usd']:>+8.2f} PF={s['pf']:>5.2f}")

    print("\n--- Walk-Forward (top 3) ---")
    for name, wf in wf_results.items():
        t = wf["train"]; te = wf["test"]
        print(f"  {name}")
        print(f"    Train(60d): T={t['trades']:>3} WR={t['wr_pct']:>5.1f}% PnL={t['pnl_pct']:>+7.2f}% DD={t['max_dd_pct']:>6.2f}%")
        print(f"    Test(30d):  T={te['trades']:>3} WR={te['wr_pct']:>5.1f}% PnL={te['pnl_pct']:>+7.2f}% DD={te['max_dd_pct']:>6.2f}%")

    # Save results
    save_data = {
        "results": results,
        "walk_forward": {k: {"train": v["train"], "test": v["test"]} for k, v in wf_results.items()},
        "leaderboard": [(n, r) for n, r in sorted_results],
        "best": sorted_results[0][0] if sorted_results else None,
    }
    path = RESULTS_DIR / f"v3_evidence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(path, "w") as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to: {path}")

    return save_data


if __name__ == "__main__":
    main()
