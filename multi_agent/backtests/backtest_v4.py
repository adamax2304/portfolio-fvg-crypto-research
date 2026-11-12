"""
backtest_v4.py — BBS / MR / FVG optimization engine  (2026-04-13)

KEY FIXES vs backtest_forensic.py (v3):
  BUG-1  Entry fees now attributed to per-agent PnL
         (v3: pa["pnl"] only counted exit-net, entry fee disappeared into balance)
  BUG-2  Concurrent multi-position (one position per SYMBOL, not one globally)
         (v3: sequential per-symbol → BTC 90d, then ETH 90d, etc.)
  BUG-3  Explicit BBS exit-mode enum
         (v3: always chandelier+fixed-cap mixed; no pure-fixed or pure-chandelier)
  BUG-4  pnl_usd now matches balance delta (sum(pa.pnl) == balance - START_BALANCE)

PHASES:
  0  Accounting verifier — confirm bug-fix is consistent
  1  BBS core sprint — exit × regime × risk grid
  2  MR overlay tests
  3  FVG v4 research track (ATR SL + fixed-R, standalone only)
  4  Portfolio matrix: BBS / BBS+MR / best combos
  5  Walk-forward on top-3 finalists
"""
from __future__ import annotations
import json, logging, math, itertools
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
    SoftRegimeDetector, Regime,
    precompute_1h_indicators, precompute_4h_indicators,
    _ema, _atr, _adx,
)
from multi_agent.agents.bb_squeeze_agent import (
    compute_squeeze_indicators, detect_squeeze_signal,
    chandelier_exit_sl, check_head_fake,
)
from multi_agent.agents.mean_reversion_agent import (
    compute_mr_indicators, detect_mr_signal,
)
from multi_agent.shared.fvg_core import (
    detect_fvgs_v2, _select_active_fvgs,
    composite_signal_score, evaluate_entry_reaction,
)
import multi_agent.config as cfg

logger = logging.getLogger("v4")
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = Path(__file__).parent / "data_cache"

SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
    "XRP/USDT:USDT", "DOGE/USDT:USDT", "ADA/USDT:USDT", "AVAX/USDT:USDT",
    "LINK/USDT:USDT", "POL/USDT:USDT", "DOT/USDT:USDT", "UNI/USDT:USDT",
    "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT", "ARB/USDT:USDT",
    "OP/USDT:USDT", "INJ/USDT:USDT", "SUI/USDT:USDT", "1000PEPE/USDT:USDT",
]

# Symbols with forensic evidence of consistent losses for BBS — blacklist candidates
BBS_BLACKLIST_CANDIDATES = {"ARB/USDT:USDT", "OP/USDT:USDT", "POL/USDT:USDT"}

START_BALANCE = 1000.0
TAKER_FEE     = 0.00055
SLIPPAGE      = 0.00025
ROUND_TRIP    = (TAKER_FEE + SLIPPAGE) * 2   # both entry and exit
MAX_LEVERAGE  = 25


# ═══════════════════════════════════════════════════════════════════════════════
#  Data loading
# ═══════════════════════════════════════════════════════════════════════════════

def _load(symbol: str, tf: str, days_tag: str = "90d") -> Optional[pd.DataFrame]:
    safe = symbol.replace("/", "_").replace(":", "_")
    for ext in (".parquet", ".csv"):
        fpath = CACHE_DIR / f"{safe}_{tf}_{days_tag}{ext}"
        if fpath.exists():
            try:
                df = pd.read_parquet(fpath) if ext == ".parquet" else pd.read_csv(fpath)
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


def _check_180d_available() -> bool:
    """Return True if 180d data exists for at least one symbol."""
    safe = SYMBOLS[0].replace("/", "_").replace(":", "_")
    for ext in (".parquet", ".csv"):
        if (CACHE_DIR / f"{safe}_15m_180d{ext}").exists():
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
#  Pre-load all symbol data into memory
# ═══════════════════════════════════════════════════════════════════════════════

def preload_all(days_tag: str = "90d", blacklist: set = None) -> dict:
    """Load and pre-compute indicators for all symbols."""
    blacklist = blacklist or set()
    all_data = {}
    for sym in SYMBOLS:
        if sym in blacklist:
            continue
        df15 = _load(sym, "15m", days_tag)
        df1h = _load(sym, "1h", days_tag)
        df4h = _load(sym, "4h", days_tag)
        if df15 is None or df1h is None or len(df15) < 300 or len(df1h) < 210:
            logger.warning("Skipping %s — insufficient data", sym)
            continue

        cl15 = df15["close"].astype(float).to_numpy()
        hi15 = df15["high"].astype(float).to_numpy()
        lo15 = df15["low"].astype(float).to_numpy()
        vo15 = df15["volume"].astype(float).to_numpy()
        ts15 = df15["timestamp"].astype(int).to_numpy()
        ts1h = df1h["timestamp"].astype(int).to_numpy()

        ind_1h = precompute_1h_indicators(df1h)
        ind_4h = precompute_4h_indicators(df4h) if df4h is not None and len(df4h) > 50 else {}
        ts4h = df4h["timestamp"].astype(int).to_numpy() if df4h is not None else None
        cl4h = df4h["close"].astype(float).to_numpy() if df4h is not None else None

        squeeze_ind = compute_squeeze_indicators(df15)
        mr_ind      = compute_mr_indicators(df15)
        adx_15m     = _adx(hi15, lo15, cl15, 14)
        atr_15m     = _atr(hi15, lo15, cl15, 14)

        all_data[sym] = {
            "df15": df15, "df1h": df1h, "df4h": df4h,
            "cl15": cl15, "hi15": hi15, "lo15": lo15,
            "vo15": vo15, "ts15": ts15, "ts1h": ts1h,
            "ts4h": ts4h, "cl4h": cl4h,
            "ind_1h": ind_1h, "ind_4h": ind_4h,
            "squeeze": squeeze_ind, "mr_ind": mr_ind,
            "adx15": adx_15m, "atr15": atr_15m,
            "n_bars": len(df15),
        }
    logger.info("Preloaded %d symbols (%s)", len(all_data), days_tag)
    return all_data


# ═══════════════════════════════════════════════════════════════════════════════
#  Core portfolio engine — FIXED accounting + concurrent multi-position
# ═══════════════════════════════════════════════════════════════════════════════

def run_portfolio(P: dict, agents: list, all_data: dict,
                  start_frac: float = 0.0, end_frac: float = 1.0) -> dict:
    """
    Multi-agent backtest with FIXED accounting.

    Position model:
      - One open position per symbol at a time.
      - No global "one position" limit — multiple symbols can be open.
      - Per-agent max concurrent: bbs_max_pos, mr_max_pos.
      - Balance is shared and compounding.

    BBS exit modes (P["bbs_exit_mode"]):
      "fixed"          — exit at fixed_tp_r × risk (pure fixed, no trailing)
      "chandelier"     — pure chandelier trailing stop (no fixed cap)
      "partial_trail"  — partial at partial_tp_r, remainder with chandelier
      "partial_fixed"  — partial at partial_tp_r, remainder at fixed_tp_r

    Fee accounting (BUG-1 fix):
      Entry fee is deducted from balance AND attributed to per-agent PnL.
      This means sum(pa["pnl"]) == balance - START_BALANCE.
    """
    balance = START_BALANCE
    peak    = START_BALANCE
    max_dd  = 0.0

    per_agent = {a: {"wins": 0, "losses": 0, "pnl": 0.0, "trades": []}
                 for a in agents}
    sym_pnl = defaultdict(float)  # per-symbol PnL for contribution table

    # Open positions: {symbol: pos_dict}
    open_pos = {}
    # Concurrent count per agent
    agent_open = {a: 0 for a in agents}

    soft_cfg = {
        "confirm_bars":         P.get("regime_confirm_bars", 5),
        "min_hold_bars":        P.get("regime_min_hold", 12),
        "smooth_alpha":         P.get("regime_smooth_alpha", 0.30),
        "min_confidence":       P.get("regime_min_confidence", 1.0),
        "transition_margin":    P.get("regime_transition_margin", 0.5),
        "undefined_threshold":  P.get("regime_undefined_threshold", 0.5),
    }

    bbs_regime_mult = {
        "TREND":      P.get("bbs_mult_trend",      1.0),
        "TRANSITION": P.get("bbs_mult_transition",  0.85),
        "SQUEEZE":    P.get("bbs_mult_squeeze",     0.5),
        "RANGE":      P.get("bbs_mult_range",       0.0),
        "UNDEFINED":  0.0,
    }
    mr_regime_mult = {
        "TRANSITION": P.get("mr_mult_transition",  1.0),
        "RANGE":      P.get("mr_mult_range",        0.85),
        "TREND":      P.get("mr_mult_trend",        0.0),
        "SQUEEZE":    0.0,
        "UNDEFINED":  0.0,
    }
    fvg_regime_mult = {
        "SQUEEZE":    P.get("fvg_mult_squeeze",    1.0),
        "TRANSITION": P.get("fvg_mult_transition", 0.5),
        "TREND":      0.0,
        "RANGE":      0.0,
        "UNDEFINED":  0.0,
    }

    risk_map  = {"bbs": P.get("bbs_risk",  0.010),
                 "mr":  P.get("mr_risk",   0.005),
                 "fvg": P.get("fvg_risk",  0.010)}
    frac_map  = {"bbs": P.get("bbs_frac",  0.75),
                 "mr":  P.get("mr_frac",   0.15),
                 "fvg": P.get("fvg_frac",  0.50)}
    max_pos   = {"bbs": P.get("bbs_max_pos", 4),
                 "mr":  P.get("mr_max_pos",  3),
                 "fvg": P.get("fvg_max_pos", 2)}

    bbs_exit_mode   = P.get("bbs_exit_mode",   "fixed")  # fixed | chandelier | partial_trail | partial_fixed
    bbs_fixed_tp_r  = P.get("bbs_fixed_tp_r",  3.0)
    bbs_partial_r   = P.get("bbs_partial_tp_r", 0.0)
    bbs_partial_frac= P.get("bbs_partial_frac", 0.5)
    fvg_exit_mode   = P.get("fvg_exit_mode",    "fixed")
    fvg_fixed_tp_r  = P.get("fvg_fixed_tp_r",   3.0)

    # Mr anti-revenge state per agent-direction
    mr_consec = {sym: {"LONG": 0, "SHORT": 0} for sym in all_data}

    # Per-symbol regime detectors (stateful)
    detectors = {sym: SoftRegimeDetector(soft_cfg) for sym in all_data}

    # Max bar count across symbols — process bar by bar
    max_n = max(d["n_bars"] for d in all_data.values())
    i_start = max(300, int(max_n * start_frac))
    i_end   = int(max_n * end_frac)

    for i in range(i_start, i_end):
        for sym, D in all_data.items():
            n = D["n_bars"]
            if i >= n:
                continue

            cl_i = D["cl15"][i]
            hi_i = D["hi15"][i]
            lo_i = D["lo15"][i]
            ts_i = D["ts15"][i]

            # HTF regime index
            htf_idx = int(np.searchsorted(D["ts1h"], ts_i, side="right")) - 1
            if htf_idx < 210:
                continue

            rs = detectors[sym].update(
                sym, D["df1h"], htf_idx, D["df4h"],
                **(D["ind_1h"]), **(D["ind_4h"]))
            regime_str = rs.regime.value

            # ─── Manage open position ─────────────────────────────────────
            if sym in open_pos:
                pos    = open_pos[sym]
                agent  = pos["agent"]
                d      = pos["direction"]
                entry  = pos["entry"]
                sl     = pos["sl"]
                r      = pos["risk"]
                q      = pos["qty"]

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = False
                tp_price = 0.0

                if agent == "bbs":
                    pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)

                    # Partial TP check
                    if bbs_partial_r > 0 and not pos.get("partial_done") and pnl_r >= bbs_partial_r:
                        pq   = q * bbs_partial_frac
                        exit_p_partial = cl_i
                        gross_p = ((exit_p_partial - entry) if d == "LONG" else (entry - exit_p_partial)) * pq
                        fee_p   = exit_p_partial * pq * (TAKER_FEE + SLIPPAGE)
                        net_p   = gross_p - fee_p
                        balance += net_p
                        per_agent[agent]["pnl"] += net_p
                        sym_pnl[sym] += net_p
                        pos["qty"] -= pq
                        pos["partial_done"] = True
                        q = pos["qty"]

                    # Chandelier trailing (if applicable)
                    if bbs_exit_mode in ("chandelier", "partial_trail"):
                        if pnl_r >= 1.0 and D["squeeze"] is not None:
                            new_ce = chandelier_exit_sl(
                                D["hi15"], D["lo15"], D["squeeze"]["ce_atr"], d, i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                                ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            if d == "LONG":
                                pos["trailing_sl"] = max(pos.get("trailing_sl", sl), new_ce)
                            else:
                                cur = pos.get("trailing_sl", 0)
                                pos["trailing_sl"] = min(cur, new_ce) if cur > 0 else new_ce
                        eff_sl = pos.get("trailing_sl", sl)
                        if d == "LONG":
                            eff_sl = max(sl, eff_sl)
                            hit_sl = lo_i <= eff_sl
                            if hit_sl: sl = eff_sl
                        else:
                            eff_sl = min(sl, eff_sl) if eff_sl > 0 else sl
                            hit_sl = hi_i >= eff_sl
                            if hit_sl: sl = eff_sl

                    # Fixed TP check
                    if bbs_exit_mode in ("fixed", "partial_fixed"):
                        if pnl_r >= bbs_fixed_tp_r:
                            hit_tp = True
                            tp_price = cl_i

                elif agent == "mr":
                    tp_stored = pos.get("tp", 0)
                    if tp_stored > 0:
                        hit_tp = (d == "LONG" and hi_i >= tp_stored) or \
                                 (d == "SHORT" and lo_i <= tp_stored)
                        if hit_tp: tp_price = tp_stored

                elif agent == "fvg":
                    tp_stored = pos.get("tp", 0)
                    if tp_stored > 0:
                        hit_tp = (d == "LONG" and hi_i >= tp_stored) or \
                                 (d == "SHORT" and lo_i <= tp_stored)
                        if hit_tp: tp_price = tp_stored
                    # FVG also supports fixed_R mode via running check
                    if fvg_exit_mode == "fixed" and not hit_tp:
                        pnl_r_fvg = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)
                        if pnl_r_fvg >= fvg_fixed_tp_r:
                            hit_tp = True
                            tp_price = cl_i

                if hit_sl or hit_tp:
                    exit_p = tp_price if hit_tp else sl
                    gross  = ((exit_p - entry) if d == "LONG" else (entry - exit_p)) * q
                    fee    = exit_p * q * (TAKER_FEE + SLIPPAGE)
                    net    = gross - fee
                    balance += net
                    per_agent[agent]["pnl"] += net
                    per_agent[agent]["trades"].append(net)
                    sym_pnl[sym] += net
                    if net > 0: per_agent[agent]["wins"] += 1
                    else:       per_agent[agent]["losses"] += 1
                    if agent == "mr":
                        if net <= 0: mr_consec[sym][d] = mr_consec[sym].get(d, 0) + 1
                        else:        mr_consec[sym][d] = 0
                    agent_open[agent] = max(0, agent_open[agent] - 1)
                    peak  = max(peak, balance)
                    dd    = (peak - balance) / peak * 100 if peak > 0 else 0
                    max_dd = max(max_dd, dd)
                    del open_pos[sym]

                continue  # skip signal scan if position was open (even if just closed)

            # ─── Signal scan ──────────────────────────────────────────────
            for agent in agents:
                if agent_open[agent] >= max_pos[agent]:
                    continue

                # Regime gate
                if agent == "bbs":
                    size_m = bbs_regime_mult.get(regime_str, 0.0)
                elif agent == "mr":
                    size_m = mr_regime_mult.get(regime_str, 0.0)
                else:  # fvg
                    size_m = fvg_regime_mult.get(regime_str, 0.0)

                if size_m < 0.05:
                    continue

                signal = None

                if agent == "bbs" and D["squeeze"] is not None:
                    bbs_s = detect_squeeze_signal(
                        D["squeeze"], bar_idx=i,
                        min_squeeze_bars=cfg.BB_SQUEEZE_MIN_SQUEEZE_BARS,
                        adx_threshold=cfg.BB_SQUEEZE_ADX_THRESHOLD,
                        vol_mult=cfg.BB_SQUEEZE_VOL_MULT)
                    if bbs_s:
                        ema200_1h = D["ind_1h"]["ema200_1h"]
                        cl1h      = D["ind_1h"]["closes_1h_arr"]
                        htf_dir   = "LONG" if cl1h[htf_idx] > ema200_1h[htf_idx] else "SHORT"
                        if bbs_s["direction"] == htf_dir and not check_head_fake(D["squeeze"], i, 2):
                            ce_sl  = chandelier_exit_sl(
                                D["hi15"], D["lo15"], D["squeeze"]["ce_atr"],
                                bbs_s["direction"], i,
                                ce_length=cfg.BB_SQUEEZE_CE_LENGTH,
                                ce_mult=cfg.BB_SQUEEZE_CE_MULT)
                            risk_v = abs(cl_i - ce_sl)
                            if 0 < risk_v / cl_i <= 0.05:
                                signal = {"direction": bbs_s["direction"],
                                          "entry_price": float(cl_i),
                                          "sl": float(ce_sl), "risk": float(risk_v), "tp": 0}

                elif agent == "mr" and D["mr_ind"] is not None:
                    mr_s = detect_mr_signal(D["mr_ind"], i, mr_consec[sym])
                    if mr_s:
                        signal = mr_s

                elif agent == "fvg":
                    signal = _fvg_signal_v4(D, i, P)

                if signal is None:
                    continue

                # Position sizing
                agent_bal  = balance * frac_map.get(agent, 0.25)
                base_risk  = risk_map.get(agent, 0.008)
                eff_risk   = base_risk * size_m
                risk_usd   = agent_bal * eff_risk
                risk_dist  = signal.get("risk", abs(signal["entry_price"] - signal.get("sl", 0)))
                if risk_dist <= 0:
                    continue
                qty        = risk_usd / risk_dist
                margin     = signal["entry_price"] * qty / MAX_LEVERAGE
                if margin > agent_bal * 0.4:
                    qty = agent_bal * 0.4 * MAX_LEVERAGE / signal["entry_price"]
                if qty * signal["entry_price"] < 5.0:
                    continue

                # Entry fee — BUG-1 FIX: charged to balance AND per_agent["pnl"]
                entry_fee = signal["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
                balance   -= entry_fee
                per_agent[agent]["pnl"] -= entry_fee  # ← was missing in v3

                # Register position
                open_pos[sym] = {
                    "agent":       agent,
                    "direction":   signal["direction"],
                    "entry":       signal["entry_price"],
                    "sl":          signal.get("sl", 0),
                    "tp":          signal.get("tp", 0),
                    "risk":        risk_dist,
                    "qty":         qty,
                    "bar_idx":     i,
                    "trailing_sl": signal.get("sl", 0),
                    "partial_done": False,
                }
                agent_open[agent] += 1
                break  # one signal per symbol per bar

    # ─── Stats ────────────────────────────────────────────────────────────────
    total_trades = sum(pa["wins"] + pa["losses"] for pa in per_agent.values())
    total_wins   = sum(pa["wins"]               for pa in per_agent.values())
    pnl_usd      = balance - START_BALANCE
    pnl_pct      = pnl_usd / START_BALANCE * 100.0
    wr            = total_wins / total_trades * 100.0 if total_trades > 0 else 0

    all_trades = []
    for pa in per_agent.values():
        all_trades.extend(pa["trades"])
    gp = sum(t for t in all_trades if t > 0)
    gl = abs(sum(t for t in all_trades if t <= 0))
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else 0)

    per_agent_out = {}
    for a, pa in per_agent.items():
        t      = pa["wins"] + pa["losses"]
        w_t    = [x for x in pa["trades"] if x > 0]
        l_t    = [x for x in pa["trades"] if x <= 0]
        gp_a   = sum(w_t)
        gl_a   = abs(sum(l_t))
        per_agent_out[a] = {
            "trades": t, "wins": pa["wins"], "losses": pa["losses"],
            "wr_pct":  round(pa["wins"] / t * 100, 1) if t > 0 else 0,
            "pnl_usd": round(pa["pnl"], 2),
            "pf":      round(gp_a / gl_a, 2) if gl_a > 0 else 0,
        }

    # Symbol contribution table
    sym_contrib = sorted(sym_pnl.items(), key=lambda x: x[1], reverse=True)

    return {
        "trades":       total_trades,
        "wins":         total_wins,
        "losses":       total_trades - total_wins,
        "wr_pct":       round(wr, 1),
        "pnl_pct":      round(pnl_pct, 2),
        "pnl_usd":      round(pnl_usd, 2),
        "max_dd_pct":   round(max_dd, 2),
        "profit_factor": round(pf, 2),
        "final_balance": round(balance, 2),
        "per_agent":    per_agent_out,
        "sym_top5":     sym_contrib[:5],
        "sym_bot5":     sym_contrib[-5:],
        "open_at_end":  len(open_pos),
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  FVG v4 signal (ATR SL + fixed-R exit variant)
# ═══════════════════════════════════════════════════════════════════════════════

def _fvg_signal_v4(D: dict, i: int, P: dict) -> Optional[dict]:
    """FVG v4: ATR-based SL, no structural swing-low."""
    if i < 300:
        return None
    try:
        from multi_agent.backtests.backtest_v2 import _fvg_signal as _fvg_v2
        cl15 = D["cl15"]; hi15 = D["hi15"]; lo15 = D["lo15"]
        vo15 = D["vo15"]; ts15 = D["ts15"]; ts1h = D["ts1h"]
        ind1h = D["ind_1h"]
        cl1h  = ind1h["closes_1h_arr"]
        ema200_1h = ind1h["ema200_1h"]
        ts4h  = D.get("ts4h")
        cl4h  = D.get("cl4h")
        ema200_4h = D["ind_4h"].get("ema200_4h") if D["ind_4h"] else None

        fvg_params = {
            "fvg_adx_min":       P.get("fvg_adx_min",       15.0),
            "fvg_volume_mult":   P.get("fvg_volume_mult",    0.6),
            "fvg_quality_min":   P.get("fvg_quality_min",    40.0),
            "fvg_entry_lo":      P.get("fvg_entry_lo",       0.20),
            "fvg_entry_hi":      P.get("fvg_entry_hi",       0.80),
            "fvg_composite_lo":  P.get("fvg_composite_lo",   55.0),
            "fvg_composite_hi":  P.get("fvg_composite_hi",   90.0),
            "fvg_fib_mandatory": P.get("fvg_fib_mandatory",  False),
            "fvg_fib_bonus":     P.get("fvg_fib_bonus",      5.0),
        }
        sig = _fvg_v2(D["df15"], i, cl15, hi15, lo15, vo15, ts15,
                      cl1h, ts1h, ema200_1h, cl4h, ts4h, ema200_4h,
                      D["adx15"], fvg_params)
        if sig is None:
            return None

        # Override SL to ATR-based (BUG FIX: structural swing-low was too wide)
        sl_atr_mult    = P.get("fvg_sl_atr_mult",    2.0)
        sl_atr_max_pct = P.get("fvg_sl_atr_max_pct", 2.5)
        atr_abs = D["atr15"][i] if i < len(D["atr15"]) else cl15[i] * 0.015
        atr_dist = min(atr_abs * sl_atr_mult, cl15[i] * sl_atr_max_pct / 100)
        atr_dist = max(atr_dist, cl15[i] * 0.003)  # min 0.3%
        dir = sig["direction"]
        new_sl = (cl15[i] - atr_dist) if dir == "LONG" else (cl15[i] + atr_dist)
        sig["sl"]   = new_sl
        sig["risk"] = atr_dist

        # TP at fixed R if mode="fixed"
        fvg_exit = P.get("fvg_exit_mode", "fixed")
        if fvg_exit == "fixed":
            tp_r  = P.get("fvg_fixed_tp_r", 3.0)
            sig["tp"] = (cl15[i] + atr_dist * tp_r) if dir == "LONG" else (cl15[i] - atr_dist * tp_r)
        return sig

    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 0: Accounting verifier
# ═══════════════════════════════════════════════════════════════════════════════

def phase0_verify(all_data: dict) -> dict:
    """
    Verify BUG-1 fix: run identical BBS-only params and check that
    per_agent["bbs"]["pnl_usd"] ≈ portfolio pnl_usd.
    (Old v3: they differed by ~entry_fees total)
    """
    base_p = {
        "bbs_exit_mode": "fixed", "bbs_fixed_tp_r": 3.0,
        "bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85,
        "bbs_mult_squeeze": 0.5, "bbs_mult_range": 0.0,
        "bbs_risk": 0.010, "bbs_frac": 0.75, "bbs_max_pos": 4,
    }
    r = run_portfolio(base_p, ["bbs"], all_data)
    portfolio_pnl = r["pnl_usd"]
    agent_pnl     = r["per_agent"]["bbs"]["pnl_usd"]
    discrepancy   = abs(portfolio_pnl - agent_pnl)

    verified = discrepancy < 0.01  # should be < 1 cent now

    logger.info("=== PHASE 0: ACCOUNTING VERIFY ===")
    logger.info("  portfolio_pnl_usd = $%.2f", portfolio_pnl)
    logger.info("  agent_pnl_usd     = $%.2f", agent_pnl)
    logger.info("  discrepancy       = $%.4f  →  %s",
                discrepancy, "PASS ✓" if verified else "FAIL ✗ (BUG STILL PRESENT)")
    return {"portfolio_pnl": portfolio_pnl, "agent_pnl": agent_pnl,
            "discrepancy": round(discrepancy, 4), "verified": verified}


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 1: BBS core sprint
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_bbs(all_data: dict, results: dict):
    logger.info("=== PHASE 1: BBS CORE OPTIMIZATION ===")

    base = {
        "bbs_risk": 0.010, "bbs_frac": 0.75, "bbs_max_pos": 4,
        "bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85,
        "bbs_mult_squeeze": 0.5, "bbs_mult_range": 0.0,
        "bbs_partial_tp_r": 0, "bbs_partial_frac": 0,
    }

    # ── 1A: Exit mode search (base regime, risk=0.010) ─────────────────────
    exit_configs = [
        ("pure_fixed_2.5R",      {"bbs_exit_mode": "fixed",         "bbs_fixed_tp_r": 2.5}),
        ("pure_fixed_3.0R",      {"bbs_exit_mode": "fixed",         "bbs_fixed_tp_r": 3.0}),
        ("pure_fixed_3.5R",      {"bbs_exit_mode": "fixed",         "bbs_fixed_tp_r": 3.5}),
        ("pure_chandelier",      {"bbs_exit_mode": "chandelier"}),
        ("partial1.5R_trail",    {"bbs_exit_mode": "partial_trail",  "bbs_partial_tp_r": 1.5, "bbs_partial_frac": 0.5}),
        ("partial2.0R_trail",    {"bbs_exit_mode": "partial_trail",  "bbs_partial_tp_r": 2.0, "bbs_partial_frac": 0.5}),
        ("partial1.5R_fixed4R",  {"bbs_exit_mode": "partial_fixed",  "bbs_partial_tp_r": 1.5, "bbs_partial_frac": 0.5, "bbs_fixed_tp_r": 4.0}),
        ("partial2.0R_fixed4R",  {"bbs_exit_mode": "partial_fixed",  "bbs_partial_tp_r": 2.0, "bbs_partial_frac": 0.4, "bbs_fixed_tp_r": 4.0}),
    ]
    logger.info("--- 1A: Exit search ---")
    exit_results = {}
    for name, overrides in exit_configs:
        p = {**base, **overrides}
        r = run_portfolio(p, ["bbs"], all_data)
        exit_results[name] = r
        logger.info("  %-25s T=%3d WR=%5.1f%% PnL=%+7.2f%% DD=%5.2f%% PF=%.2f  $%+.2f",
                    name, r["trades"], r["wr_pct"], r["pnl_pct"],
                    r["max_dd_pct"], r["profit_factor"], r["pnl_usd"])
    results["phase1_exit"] = exit_results

    # Pick best exit by score (PnL% - 0.4×DD% + 0.2×log(trades+1))
    def score(r):
        return r["pnl_pct"] - 0.4 * r["max_dd_pct"] + 0.2 * math.log(r["trades"] + 1)
    best_exit_name = max(exit_results, key=lambda k: score(exit_results[k]))
    best_exit_cfg  = dict(exit_configs[[x[0] for x in exit_configs].index(best_exit_name)][1])
    logger.info("  >>> Best exit: %s (score=%.3f)", best_exit_name, score(exit_results[best_exit_name]))

    # ── 1B: Regime search (best exit, risk=0.010) ─────────────────────────
    regime_configs = [
        ("regime_base",         {"bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85, "bbs_mult_squeeze": 0.5,  "bbs_mult_range": 0.0}),
        ("regime_no_squeeze",   {"bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85, "bbs_mult_squeeze": 0.0,  "bbs_mult_range": 0.0}),
        ("regime_trend_trans",  {"bbs_mult_trend": 1.0, "bbs_mult_transition": 1.0,  "bbs_mult_squeeze": 0.0,  "bbs_mult_range": 0.0}),
        ("regime_conservative", {"bbs_mult_trend": 0.8, "bbs_mult_transition": 0.8,  "bbs_mult_squeeze": 0.0,  "bbs_mult_range": 0.0}),
        ("regime_broad",        {"bbs_mult_trend": 1.0, "bbs_mult_transition": 1.0,  "bbs_mult_squeeze": 0.3,  "bbs_mult_range": 0.2}),
    ]
    logger.info("--- 1B: Regime search ---")
    regime_results = {}
    for name, overrides in regime_configs:
        p = {**base, **best_exit_cfg, **overrides}
        r = run_portfolio(p, ["bbs"], all_data)
        regime_results[name] = r
        logger.info("  %-22s T=%3d WR=%5.1f%% PnL=%+7.2f%% DD=%5.2f%% PF=%.2f  $%+.2f",
                    name, r["trades"], r["wr_pct"], r["pnl_pct"],
                    r["max_dd_pct"], r["profit_factor"], r["pnl_usd"])
    results["phase1_regime"] = regime_results

    best_regime_name = max(regime_results, key=lambda k: score(regime_results[k]))
    best_regime_cfg  = dict(regime_configs[[x[0] for x in regime_configs].index(best_regime_name)][1])
    logger.info("  >>> Best regime: %s", best_regime_name)

    # ── 1C: Risk/size search (best exit + regime) ─────────────────────────
    risk_configs = [
        ("risk_0.008", {"bbs_risk": 0.008}),
        ("risk_0.010", {"bbs_risk": 0.010}),
        ("risk_0.012", {"bbs_risk": 0.012}),
        ("risk_0.015", {"bbs_risk": 0.015}),
    ]
    logger.info("--- 1C: Risk search ---")
    risk_results = {}
    for name, overrides in risk_configs:
        p = {**base, **best_exit_cfg, **best_regime_cfg, **overrides}
        r = run_portfolio(p, ["bbs"], all_data)
        risk_results[name] = r
        logger.info("  %-15s T=%3d WR=%5.1f%% PnL=%+7.2f%% DD=%5.2f%% PF=%.2f  $%+.2f",
                    name, r["trades"], r["wr_pct"], r["pnl_pct"],
                    r["max_dd_pct"], r["profit_factor"], r["pnl_usd"])
    results["phase1_risk"] = risk_results

    best_risk_name = max(risk_results, key=lambda k: score(risk_results[k]))
    best_risk_cfg  = dict(risk_configs[[x[0] for x in risk_configs].index(best_risk_name)][1])
    logger.info("  >>> Best risk: %s", best_risk_name)

    # ── 1D: Symbol blacklist test ─────────────────────────────────────────
    logger.info("--- 1D: Blacklist test ---")
    bl_configs = [
        ("no_blacklist", all_data),
        ("blacklist_3",  {k: v for k, v in all_data.items() if k not in BBS_BLACKLIST_CANDIDATES}),
    ]
    bl_results = {}
    best_combined = {**base, **best_exit_cfg, **best_regime_cfg, **best_risk_cfg}
    for name, data_subset in bl_configs:
        r = run_portfolio(best_combined, ["bbs"], data_subset)
        bl_results[name] = r
        logger.info("  %-15s T=%3d PnL=%+7.2f%% DD=%5.2f%% PF=%.2f  $%+.2f",
                    name, r["trades"], r["pnl_pct"], r["max_dd_pct"],
                    r["profit_factor"], r["pnl_usd"])
    results["phase1_blacklist"] = bl_results

    # ── Final BBS-optimal config ──────────────────────────────────────────
    bbs_optimal = {**base, **best_exit_cfg, **best_regime_cfg, **best_risk_cfg}
    results["bbs_optimal_params"]  = bbs_optimal
    results["bbs_optimal_exit"]    = best_exit_name
    results["bbs_optimal_regime"]  = best_regime_name
    results["bbs_optimal_risk"]    = best_risk_name
    return bbs_optimal


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 2: MR overlay
# ═══════════════════════════════════════════════════════════════════════════════

def phase2_mr(bbs_optimal: dict, all_data: dict, results: dict):
    logger.info("=== PHASE 2: MR OVERLAY ===")

    mr_configs = [
        ("mr_10pct_r0.3", {"mr_frac": 0.10, "mr_risk": 0.003, "mr_max_pos": 3,
                           "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.0}),
        ("mr_15pct_r0.4", {"mr_frac": 0.15, "mr_risk": 0.004, "mr_max_pos": 3,
                           "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.0}),
        ("mr_20pct_r0.5", {"mr_frac": 0.20, "mr_risk": 0.005, "mr_max_pos": 3,
                           "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.0}),
        ("mr_20pct_r0.6", {"mr_frac": 0.20, "mr_risk": 0.006, "mr_max_pos": 3,
                           "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.0}),
        ("mr_15pct_wide", {"mr_frac": 0.15, "mr_risk": 0.004, "mr_max_pos": 3,
                           "mr_mult_transition": 1.0, "mr_mult_range": 0.85, "mr_mult_trend": 0.3}),
    ]

    mr_results = {}
    bbs_baseline = run_portfolio(bbs_optimal, ["bbs"], all_data)
    results["bbs_only_optimal"] = bbs_baseline
    logger.info("  BBS baseline: T=%d WR=%.1f%% PnL=%+.2f%% DD=%.2f%% PF=%.2f $%+.2f",
                bbs_baseline["trades"], bbs_baseline["wr_pct"], bbs_baseline["pnl_pct"],
                bbs_baseline["max_dd_pct"], bbs_baseline["profit_factor"], bbs_baseline["pnl_usd"])

    best_combined = None
    best_combined_score = -999

    def score(r):
        return r["pnl_pct"] - 0.4 * r["max_dd_pct"] + 0.2 * math.log(r["trades"] + 1)

    bbs_score = score(bbs_baseline)

    for name, mr_p in mr_configs:
        p = {**bbs_optimal, **mr_p}
        r = run_portfolio(p, ["bbs", "mr"], all_data)
        mr_results[name] = r
        sc = score(r)
        delta = sc - bbs_score
        logger.info("  %-20s T=%3d WR=%5.1f%% PnL=%+7.2f%% DD=%5.2f%% PF=%.2f $%+.2f  Δscore=%+.3f",
                    name, r["trades"], r["wr_pct"], r["pnl_pct"],
                    r["max_dd_pct"], r["profit_factor"], r["pnl_usd"], delta)
        if sc > best_combined_score:
            best_combined_score = sc
            best_combined = {**p}

    results["phase2_mr"] = mr_results
    results["best_combined_params"] = best_combined

    # MR verdict
    best_mr_name = max(mr_results, key=lambda k: score(mr_results[k]))
    best_mr_r    = mr_results[best_mr_name]
    mr_improves  = score(best_mr_r) > bbs_score

    logger.info("  MR VERDICT: %s (best=%s, bbs_score=%.3f, mr_score=%.3f)",
                "HELPS" if mr_improves else "HURTS or NEUTRAL",
                best_mr_name, bbs_score, score(best_mr_r))
    results["mr_verdict"] = {
        "adds_value": mr_improves,
        "best_mr_config": best_mr_name,
        "bbs_score": round(bbs_score, 3),
        "best_mr_score": round(score(best_mr_r), 3),
    }
    return best_combined if mr_improves else bbs_optimal


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 3: FVG v4 research track (standalone only)
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_fvg_v4(all_data: dict, results: dict):
    logger.info("=== PHASE 3: FVG v4 RESEARCH TRACK ===")

    fvg_configs = [
        ("fvg_v4_atr2_3R_squeeze",    {"fvg_sl_atr_mult": 2.0, "fvg_sl_atr_max_pct": 2.5,
                                        "fvg_exit_mode": "fixed",  "fvg_fixed_tp_r": 3.0,
                                        "fvg_mult_squeeze": 1.0, "fvg_mult_transition": 0.0,
                                        "fvg_risk": 0.010, "fvg_frac": 0.75, "fvg_max_pos": 3}),
        ("fvg_v4_atr2_3R_sq_trans",   {"fvg_sl_atr_mult": 2.0, "fvg_sl_atr_max_pct": 2.5,
                                        "fvg_exit_mode": "fixed",  "fvg_fixed_tp_r": 3.0,
                                        "fvg_mult_squeeze": 1.0, "fvg_mult_transition": 0.5,
                                        "fvg_risk": 0.010, "fvg_frac": 0.75, "fvg_max_pos": 3}),
        ("fvg_v4_atr1.5_2R_squeeze",  {"fvg_sl_atr_mult": 1.5, "fvg_sl_atr_max_pct": 2.0,
                                        "fvg_exit_mode": "fixed",  "fvg_fixed_tp_r": 2.0,
                                        "fvg_mult_squeeze": 1.0, "fvg_mult_transition": 0.0,
                                        "fvg_risk": 0.010, "fvg_frac": 0.75, "fvg_max_pos": 3}),
        ("fvg_v4_atr2.5_4R_broad",    {"fvg_sl_atr_mult": 2.5, "fvg_sl_atr_max_pct": 3.0,
                                        "fvg_exit_mode": "fixed",  "fvg_fixed_tp_r": 4.0,
                                        "fvg_mult_squeeze": 1.0, "fvg_mult_transition": 0.5,
                                        "fvg_risk": 0.008, "fvg_frac": 0.75, "fvg_max_pos": 2}),
    ]

    fvg_results = {}
    for name, p in fvg_configs:
        r = run_portfolio(p, ["fvg"], all_data)
        fvg_results[name] = r
        logger.info("  %-30s T=%3d WR=%5.1f%% PnL=%+7.2f%% DD=%5.2f%% PF=%.2f $%+.2f",
                    name, r["trades"], r["wr_pct"], r["pnl_pct"],
                    r["max_dd_pct"], r["profit_factor"], r["pnl_usd"])

    results["phase3_fvg_v4"] = fvg_results

    best_fvg = max(fvg_results, key=lambda k: fvg_results[k]["pnl_usd"])
    best_fvg_r = fvg_results[best_fvg]
    fvg_viable = best_fvg_r["profit_factor"] > 1.0 and best_fvg_r["pnl_usd"] > 0
    logger.info("  FVG v4 VERDICT: %s (best=%s PF=%.2f PnL=$%+.2f)",
                "SHOWS PROMISE" if fvg_viable else "STILL BROKEN",
                best_fvg, best_fvg_r["profit_factor"], best_fvg_r["pnl_usd"])
    results["fvg_v4_verdict"] = {
        "viable": fvg_viable,
        "best_config": best_fvg,
        "best_pf": best_fvg_r["profit_factor"],
        "best_pnl": best_fvg_r["pnl_usd"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 4: Final portfolio matrix
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_portfolio(bbs_opt: dict, combined_opt: dict, all_data: dict, results: dict):
    logger.info("=== PHASE 4: PORTFOLIO MATRIX ===")

    # BBS-only baseline (default params)
    baseline_p = {
        "bbs_exit_mode": "fixed", "bbs_fixed_tp_r": 3.0,
        "bbs_mult_trend": 1.0, "bbs_mult_transition": 0.85,
        "bbs_mult_squeeze": 0.5, "bbs_mult_range": 0.0,
        "bbs_risk": 0.010, "bbs_frac": 0.75, "bbs_max_pos": 4,
    }

    tests = [
        ("01_bbs_baseline",     ["bbs"],       baseline_p),
        ("02_bbs_optimal",      ["bbs"],       bbs_opt),
        ("03_mr_only",          ["mr"],        {**baseline_p, "mr_frac": 0.20, "mr_risk": 0.005,
                                                "mr_max_pos": 3, "mr_mult_transition": 1.0,
                                                "mr_mult_range": 0.85, "mr_mult_trend": 0.0}),
        ("04_bbs_mr_baseline",  ["bbs", "mr"], {**baseline_p, "mr_frac": 0.15, "mr_risk": 0.004,
                                                "mr_max_pos": 3, "mr_mult_transition": 1.0,
                                                "mr_mult_range": 0.85, "mr_mult_trend": 0.0}),
        ("05_bbs_mr_optimal",   ["bbs", "mr"], combined_opt),
        ("06_bbs_v3_baseline",  ["bbs"],       {**baseline_p,   # match v3 forensic
                                                "bbs_exit_mode": "fixed", "bbs_fixed_tp_r": 3.0,
                                                "bbs_mult_trend": 1.0, "bbs_mult_squeeze": 0.5,
                                                "bbs_frac": 0.35, "bbs_risk": 0.010}),
    ]

    # Add FVG v4 if it showed positive PnL
    fvg_v4_best = results.get("phase3_fvg_v4", {})
    best_fvg_name = max(fvg_v4_best, key=lambda k: fvg_v4_best[k]["pnl_usd"]) if fvg_v4_best else None
    if best_fvg_name and fvg_v4_best[best_fvg_name]["pnl_usd"] > 0:
        fvg_p = results.get("phase3_fvg_v4", {}).get(best_fvg_name, {})
        # Add FVG-only to matrix
        fvg_config = {
            "fvg_sl_atr_mult": 2.0, "fvg_sl_atr_max_pct": 2.5,
            "fvg_exit_mode": "fixed", "fvg_fixed_tp_r": 3.0,
            "fvg_mult_squeeze": 1.0, "fvg_mult_transition": 0.5,
            "fvg_risk": 0.010, "fvg_frac": 0.75, "fvg_max_pos": 3,
        }
        tests.append(("07_fvg_v4_standalone", ["fvg"], fvg_config))
        tests.append(("08_bbs_fvg_v4",        ["bbs", "fvg"], {**bbs_opt, **fvg_config}))

    matrix = {}
    for name, agents, p in tests:
        r = run_portfolio(p, agents, all_data)
        matrix[name] = r
        per_str = " | ".join(
            f"{a.upper()}: WR={s['wr_pct']:.0f}% ${s['pnl_usd']:+.0f}"
            for a, s in r["per_agent"].items() if s["trades"] > 0
        )
        logger.info("  %-25s T=%3d PnL=%+8.2f%% DD=%5.2f%% PF=%.2f  [%s]",
                    name, r["trades"], r["pnl_pct"],
                    r["max_dd_pct"], r["profit_factor"], per_str)

    results["phase4_matrix"] = matrix
    return matrix


# ═══════════════════════════════════════════════════════════════════════════════
#  Phase 5: Walk-forward on top-3 finalists
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_walkforward(matrix: dict, bbs_opt: dict, combined_opt: dict,
                        all_data: dict, results: dict):
    logger.info("=== PHASE 5: WALK-FORWARD VALIDATION ===")

    def score(r):
        return r["pnl_pct"] - 0.4 * r["max_dd_pct"] + 0.2 * math.log(r["trades"] + 1)

    # Pick top-3 by score
    top3 = sorted(matrix.items(), key=lambda x: score(x[1]), reverse=True)[:3]
    wf_results = {}

    agent_map = {
        "01_bbs_baseline":    ["bbs"],
        "02_bbs_optimal":     ["bbs"],
        "03_mr_only":         ["mr"],
        "04_bbs_mr_baseline": ["bbs", "mr"],
        "05_bbs_mr_optimal":  ["bbs", "mr"],
        "06_bbs_v3_baseline": ["bbs"],
        "07_fvg_v4_standalone": ["fvg"],
        "08_bbs_fvg_v4":      ["bbs", "fvg"],
    }

    param_map = {
        "01_bbs_baseline":    {**bbs_opt, "bbs_exit_mode": "fixed", "bbs_fixed_tp_r": 3.0},
        "02_bbs_optimal":     bbs_opt,
        "04_bbs_mr_baseline": combined_opt,
        "05_bbs_mr_optimal":  combined_opt,
    }

    for name, r in top3:
        agents = agent_map.get(name, ["bbs"])
        # Use the stored params from the bbs_opt / combined_opt as fallback
        p = param_map.get(name, bbs_opt if "bbs" in agents else combined_opt)

        r_train = run_portfolio(p, agents, all_data, start_frac=0.0,  end_frac=0.67)
        r_test  = run_portfolio(p, agents, all_data, start_frac=0.67, end_frac=1.0)
        wf_results[name] = {"train": r_train, "test": r_test}

        train_score = score(r_train)
        test_score  = score(r_test)
        degradation = (train_score - test_score) / abs(train_score + 1e-9) * 100

        logger.info("  %-25s  TRAIN: PnL=%+.2f%% DD=%.2f%%   TEST: PnL=%+.2f%% DD=%.2f%%   Δ=%.1f%%",
                    name,
                    r_train["pnl_pct"], r_train["max_dd_pct"],
                    r_test["pnl_pct"],  r_test["max_dd_pct"],
                    degradation)

    results["phase5_walkforward"] = wf_results


# ═══════════════════════════════════════════════════════════════════════════════
#  180d test (if data available)
# ═══════════════════════════════════════════════════════════════════════════════

def test_180d(bbs_opt: dict, combined_opt: dict, results: dict):
    if not _check_180d_available():
        logger.info("=== 180d data not available — skipping ===")
        results["180d"] = "data_not_available"
        return

    logger.info("=== 180d TEST ===")
    all_data_180 = preload_all("180d")
    if not all_data_180:
        results["180d"] = "load_failed"
        return

    r_bbs  = run_portfolio(bbs_opt,      ["bbs"],        all_data_180)
    r_both = run_portfolio(combined_opt, ["bbs", "mr"],  all_data_180)
    logger.info("  BBS 180d:     T=%d PnL=%+.2f%% DD=%.2f%% PF=%.2f $%+.2f",
                r_bbs["trades"],  r_bbs["pnl_pct"],  r_bbs["max_dd_pct"],  r_bbs["profit_factor"],  r_bbs["pnl_usd"])
    logger.info("  BBS+MR 180d:  T=%d PnL=%+.2f%% DD=%.2f%% PF=%.2f $%+.2f",
                r_both["trades"], r_both["pnl_pct"], r_both["max_dd_pct"], r_both["profit_factor"], r_both["pnl_usd"])
    results["180d"] = {"bbs_only": r_bbs, "bbs_mr": r_both}


# ═══════════════════════════════════════════════════════════════════════════════
#  Final report
# ═══════════════════════════════════════════════════════════════════════════════

def print_final_report(results: dict):
    print("\n" + "=" * 110)
    print("BACKTEST v4 — FINAL REPORT")
    print("=" * 110)

    # Phase 0
    v = results.get("phase0_verify", {})
    print(f"\n[PHASE 0] Accounting verify: portfolio_pnl=${v.get('portfolio_pnl',0):.2f}  "
          f"agent_pnl=${v.get('agent_pnl',0):.2f}  "
          f"discrepancy=${v.get('discrepancy',0):.4f}  "
          f"{'PASS [OK]' if v.get('verified') else 'FAIL [!!]'}")

    # Phase 1
    print(f"\n[PHASE 1] BBS Optimal:")
    print(f"  Best exit:   {results.get('bbs_optimal_exit','?')}")
    print(f"  Best regime: {results.get('bbs_optimal_regime','?')}")
    print(f"  Best risk:   {results.get('bbs_optimal_risk','?')}")
    if "bbs_only_optimal" in results:
        r = results["bbs_only_optimal"]
        print(f"  Performance: T={r['trades']} WR={r['wr_pct']}% PnL={r['pnl_pct']:+.2f}% "
              f"DD={r['max_dd_pct']:.2f}% PF={r['profit_factor']:.2f} ${r['pnl_usd']:+.2f}")
        if r.get("sym_top5"):
            top3 = ["{}={:+.1f}".format(s, v) for s, v in r["sym_top5"][:3]]
            print(f"  Top symbols:  {top3}")
        if r.get("sym_bot5"):
            bot3 = ["{}={:+.1f}".format(s, v) for s, v in r["sym_bot5"][:3]]
            print(f"  Worst symbols:{bot3}")

    # Phase 2
    mv = results.get("mr_verdict", {})
    print(f"\n[PHASE 2] MR Verdict: {'ADDS VALUE' if mv.get('adds_value') else 'NO BENEFIT'} "
          f"(best={mv.get('best_mr_config','?')} BBS_score={mv.get('bbs_score',0):.3f} "
          f"MR_score={mv.get('best_mr_score',0):.3f})")

    # Phase 3
    fvg_v = results.get("fvg_v4_verdict", {})
    print(f"\n[PHASE 3] FVG v4 Verdict: {'SHOWS PROMISE' if fvg_v.get('viable') else 'STILL BROKEN'} "
          f"(best={fvg_v.get('best_config','?')} PF={fvg_v.get('best_pf',0):.2f} "
          f"${fvg_v.get('best_pnl',0):+.2f})")

    # Phase 4 matrix
    print(f"\n[PHASE 4] Portfolio Matrix:")
    def score(r):
        return r["pnl_pct"] - 0.4 * r["max_dd_pct"] + 0.2 * math.log(r["trades"] + 1)
    matrix = results.get("phase4_matrix", {})
    sorted_m = sorted(matrix.items(), key=lambda x: score(x[1]), reverse=True)
    print(f"  {'Config':<27} {'T':>4} {'WR%':>6} {'PnL%':>8} {'MaxDD%':>7} {'PF':>5} {'Score':>7}")
    print("  " + "-" * 65)
    for name, r in sorted_m:
        sc = score(r)
        print(f"  {name:<27} {r['trades']:>4} {r['wr_pct']:>5.1f}% {r['pnl_pct']:>+7.2f}% "
              f"{r['max_dd_pct']:>6.2f}% {r['profit_factor']:>4.2f} {sc:>+6.2f}")

    # Phase 5 walk-forward
    print(f"\n[PHASE 5] Walk-Forward (top-3):")
    for name, wf in results.get("phase5_walkforward", {}).items():
        tr = wf["train"]; te = wf["test"]
        print(f"  {name:<27}  Train: {tr['pnl_pct']:>+6.2f}% DD={tr['max_dd_pct']:.2f}%  "
              f"Test: {te['pnl_pct']:>+6.2f}% DD={te['max_dd_pct']:.2f}%")

    # Final recommendation
    print(f"\n{'='*110}")
    if sorted_m:
        top_name, top_r = sorted_m[0]
        top_agents = "BBS+MR" if "mr" in top_name else "BBS only"
        print(f"FINAL RECOMMENDATION: {top_name} ({top_agents})")
        print(f"  PnL: {top_r['pnl_pct']:+.2f}%  MaxDD: {top_r['max_dd_pct']:.2f}%  PF: {top_r['profit_factor']:.2f}  ${top_r['pnl_usd']:+.2f}")

    print("=" * 110)


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = __import__("time").time()
    logger.info("=" * 100)
    logger.info("BACKTEST v4 — BBS/MR/FVG Optimization Engine")
    logger.info("=" * 100)

    results = {}

    # Load data
    all_data = preload_all("90d")
    if not all_data:
        logger.error("No data loaded — run fetch_data.py first")
        return

    # Phase 0: accounting verification
    results["phase0_verify"] = phase0_verify(all_data)

    # Phase 1: BBS optimization
    bbs_opt = phase1_bbs(all_data, results)

    # Phase 2: MR overlay
    combined_opt = phase2_mr(bbs_opt, all_data, results)

    # Phase 3: FVG v4 research
    phase3_fvg_v4(all_data, results)

    # Phase 4: portfolio matrix
    matrix = phase4_portfolio(bbs_opt, combined_opt, all_data, results)

    # Phase 5: walk-forward
    phase5_walkforward(matrix, bbs_opt, combined_opt, all_data, results)

    # 180d test
    test_180d(bbs_opt, combined_opt, results)

    # Final report
    print_final_report(results)

    # Save
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_v4_{ts}.json"
    with open(path, "w") as f:
        # Convert numpy types for JSON
        def _default(o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray): return o.tolist()
            return str(o)
        json.dump(results, f, indent=2, default=_default)

    elapsed = __import__("time").time() - t0
    logger.info("Done in %.1f min. Results saved to: %s", elapsed / 60, path)
    print(f"\nResults saved to: {path}")
    print(f"Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
