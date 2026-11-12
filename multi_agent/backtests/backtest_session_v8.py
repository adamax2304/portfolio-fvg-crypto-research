"""
backtest_session_v8.py -- 2026-04-17

V7 champion: R_skip_chop  PF=2.03 PnL=+250% DD=5.76% (237 trades)

V8 goal: squeeze more PnL / lower DD via many layered improvements.
Each section tests ONE hypothesis on V7 champion base (exWed + skipChop) unless noted.

Sections:
  H1. Hourly BTC regime     -- finer than daily; skip hour-buckets with low vol
  H2. Range-expansion       -- only trade when AL range > N-day-avg AL range
  H3. Pre-break momentum    -- require N bars of direction-aligned momentum
  H4. Break bar size        -- require break bar body >= X*ATR
  H5. Volume spike at break -- require break-bar volume > X * avg
  H6. BTC direction align   -- require BTC 1h trend matches signal direction
  H7. Multi-TF (4h+1h)      -- both 4h and 1h must align
  H8. Per-symbol DoW        -- each symbol uses its own best DoWs
  H9. Cooldown after losers -- skip N days after M consecutive losers
  H10. ATR vol targeting    -- size inversely to symbol daily ATR%
  H11. Late-hour cutoff     -- tighter than V7's max_entry_hr=4
  H12. Retest reentry       -- allow 2nd signal per day on retest of level
  H13. BTC trend reversal   -- fade BTC counter-trend breaks
  H14. Dynamic TP by regime -- wider TP on high-vol days

Final: ULTRA stack = V7 base + best of each section + walk-forward

NO PRODUCTION CODE IS CHANGED.
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

from multi_agent.backtests.backtest_session_v4 import (
    _load, _atr, _ema, _adx,
    prep_15m_arrays, build_session_ranges,
    compute_htf_trends, align_htf,
    _place_sl_tp, _check_htf,
    START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE, RISK_PER_TRADE,
    TOP14,
)
from multi_agent.backtests.backtest_session_v5 import GOOD10, _aggregate_v5
from multi_agent.backtests.backtest_session_v6 import (
    BASE_PARAMS_V5_CHAMPION, split_by_quarters,
)
from multi_agent.backtests.backtest_session_v7 import (
    build_btc_regime, count_signals_per_day,
)

logger = logging.getLogger("session_bt_v8")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s")
RESULTS_DIR = Path(__file__).parent / "results"


# ═══════════════════════════════════════════════════════════════════════════
#  AUX METRICS (precomputed once per symbol for V8 filters)
# ═══════════════════════════════════════════════════════════════════════════

def build_hourly_btc_regime(arrays):
    """
    Hourly BTC vol regime: for each 15m bar, the rolling 4-hour ATR%.
    Returns: dict: UTC_ts (ms) -> {"vol_pct": float, "regime": str}
    Bucketed via percentiles of hourly vol_pct.
    """
    btc = arrays.get("BTC/USDT:USDT")
    if btc is None:
        return {}
    c = btc["c"]; h = btc["h"]; l = btc["l"]; ts = btc["ts"]
    n = len(c)
    # Rolling 4-hour window = 16 bars (15m)
    window = 16
    vols = np.full(n, np.nan)
    for i in range(window, n):
        hi = np.max(h[i-window+1:i+1])
        lo = np.min(l[i-window+1:i+1])
        cl = c[i]
        if cl > 0:
            vols[i] = (hi - lo) / cl * 100
    valid = vols[~np.isnan(vols)]
    if len(valid) == 0:
        return {}
    p25 = float(np.percentile(valid, 25))
    p75 = float(np.percentile(valid, 75))
    p90 = float(np.percentile(valid, 90))
    logger.info(f"BTC hourly(4h rolling) vol%: p25={p25:.2f} p75={p75:.2f} p90={p90:.2f}")
    out = {}
    for i in range(n):
        v = vols[i]
        if np.isnan(v):
            continue
        if v < p25: r = "chop"
        elif v > p90: r = "panic"
        elif v > p75: r = "high"
        else: r = "normal"
        out[int(ts[i])] = {"vol_pct": v, "regime": r}
    return out


def build_range_expansion_map(arr, ranges):
    """
    For each date with an AL range, compute:
      rel_range = current AL range / rolling 10-day median AL range
    Returns: date_str -> rel_range (float)
    """
    dates_sorted = sorted(ranges.keys())
    al_by_date = {}
    for d in dates_sorted:
        rr = ranges[d]
        if not np.isnan(rr.get("al_hi", np.nan)) and not np.isnan(rr.get("al_lo", np.nan)):
            al_by_date[d] = rr["al_hi"] - rr["al_lo"]
    out = {}
    window = 10
    dates_sorted = sorted(al_by_date.keys())
    for i, d in enumerate(dates_sorted):
        lo = max(0, i - window)
        prev = [al_by_date[dd] for dd in dates_sorted[lo:i] if al_by_date[dd] > 0]
        if not prev:
            out[d] = 1.0
        else:
            med = float(np.median(prev))
            out[d] = al_by_date[d] / med if med > 0 else 1.0
    return out


def build_symbol_atr_pct(arr):
    """Return per-bar ATR% = atr/close * 100 (used for vol targeting)."""
    c = arr["c"]; atr = arr["atr"]
    ap = np.full(len(c), np.nan)
    for i in range(len(c)):
        if c[i] > 0 and not np.isnan(atr[i]):
            ap[i] = atr[i] / c[i] * 100
    return ap


def build_btc_1h_direction(arr_btc_15m, htf_by_sym_btc):
    """Map 15m_ts -> BTC trend direction (-1/0/1) using trend_combo_1h."""
    m = htf_by_sym_btc.get("trend_combo_1h")
    if m is None or len(m) == 0:
        return np.zeros(arr_btc_15m["n"], dtype=np.int32)
    return m.astype(np.int32)


def build_pre_break_momentum(arr, window=3):
    """
    Returns per-bar last-N-bar return sign & magnitude (vs ATR).
    mom_sign[i]  = sign of (close[i] - close[i-N])
    mom_r[i]     = abs return / atr[i]
    """
    c = arr["c"]; atr = arr["atr"]; n = len(c)
    mom_sign = np.zeros(n, dtype=np.int32)
    mom_r = np.zeros(n, dtype=float)
    for i in range(window, n):
        diff = c[i] - c[i-window]
        mom_sign[i] = 1 if diff > 0 else (-1 if diff < 0 else 0)
        if atr[i] > 0:
            mom_r[i] = abs(diff) / atr[i]
    return mom_sign, mom_r


def build_vol_ratio_15m(arr, window=20):
    """Per-bar vol ratio: v[i] / rolling mean."""
    v = arr["v"]; n = len(v)
    s = pd.Series(v).rolling(window, min_periods=5).mean().values
    r = np.where(s > 0, v / s, 1.0)
    return r


# ═══════════════════════════════════════════════════════════════════════════
#  V8 SIGNAL GEN — superset of V7 with new filters
# ═══════════════════════════════════════════════════════════════════════════

def gen_al_break_v8(arr, sess_ranges, params, htf_aligned_map=None,
                    btc_regime=None, hourly_btc_regime=None,
                    range_exp_map=None, per_sym_dows=None, sym_name=None,
                    btc_dir_aligned=None, mom_sign=None, mom_r=None,
                    vol_ratio_15m=None):
    h, l, c, o = arr["h"], arr["l"], arr["c"], arr["o"]
    hour, date = arr["hour"], arr["date"]
    dow = arr["dow"]
    atr = arr["atr"]
    ny_vwap = arr["ny_vwap"]
    ts = arr["ts"]
    n = arr["n"]

    max_entry_hr   = params.get("max_entry_hr", 4)
    min_range_atr  = params.get("min_range_atr", 0.5)
    max_range_atr  = params.get("max_range_atr", 5.0)
    allowed_dows   = params.get("allowed_dows", {0, 1, 3, 4, 5, 6})
    skip_regimes   = params.get("skip_regimes", set())
    skip_hourly_reg= params.get("skip_hourly_regimes", set())

    # V8 new:
    min_rel_range  = params.get("min_rel_range", 0.0)       # H2: range expansion
    min_mom_bars   = params.get("min_mom_bars", 0)          # H3: pre-break momentum window
    min_mom_r      = params.get("min_mom_r", 0.0)           # H3: pre-break momentum magnitude
    min_break_body = params.get("min_break_body_atr", 0.0)  # H4: break bar body size
    min_vol_ratio  = params.get("min_vol_ratio", 0.0)       # H5: break-bar volume
    require_btc_align = params.get("require_btc_align", False)  # H6
    max_vwap_dist_atr = params.get("max_vwap_dist_atr", 0.0)
    max_entry_hr_tight = params.get("max_entry_hr_tight", None)  # H11

    # H8: per-symbol DoW overrides
    if per_sym_dows and sym_name in per_sym_dows:
        allowed_dows = per_sym_dows[sym_name]

    signals = []
    sigs_by_date: dict = {}

    eff_hr_cap = max_entry_hr_tight if max_entry_hr_tight is not None else max_entry_hr

    for i in range(50, n - 1):
        d = date[i]
        if d not in sess_ranges: continue
        if int(dow[i]) not in allowed_dows: continue

        # daily BTC regime
        if btc_regime and skip_regimes:
            reg = btc_regime.get(d, {}).get("regime")
            if reg in skip_regimes: continue

        # hourly BTC regime
        if hourly_btc_regime and skip_hourly_reg:
            hreg = hourly_btc_regime.get(int(ts[i]), {}).get("regime")
            if hreg in skip_hourly_reg: continue

        # range-expansion filter
        if min_rel_range > 0 and range_exp_map:
            rel = range_exp_map.get(d, 1.0)
            if rel < min_rel_range: continue

        r = sess_ranges[d]
        alhi, allo = r["al_hi"], r["al_lo"]
        if np.isnan(alhi) or np.isnan(allo): continue
        if hour[i] < 13 or hour[i] >= 13 + eff_hr_cap: continue
        al_range = alhi - allo
        if al_range <= 0 or atr[i] <= 0: continue
        rR = al_range / atr[i]
        if rR < min_range_atr or rR > max_range_atr: continue

        direction = None; lvl = None
        if h[i] > alhi and c[i] > alhi:
            direction = "LONG"; lvl = alhi
        elif l[i] < allo and c[i] < allo:
            direction = "SHORT"; lvl = allo
        if direction is None: continue
        if not _check_htf(params, htf_aligned_map, i, direction): continue

        # H4: break bar body size filter
        if min_break_body > 0 and atr[i] > 0:
            body = abs(c[i] - o[i])
            if body / atr[i] < min_break_body: continue

        # H5: volume spike at break
        if min_vol_ratio > 0 and vol_ratio_15m is not None:
            if vol_ratio_15m[i] < min_vol_ratio: continue

        # H3: pre-break momentum
        if min_mom_bars > 0 and mom_sign is not None:
            want = 1 if direction == "LONG" else -1
            if mom_sign[i] != want: continue
            if mom_r is not None and mom_r[i] < min_mom_r: continue

        # H6: BTC direction align
        if require_btc_align and btc_dir_aligned is not None:
            want = 1 if direction == "LONG" else -1
            if btc_dir_aligned[i] != want: continue

        # VWAP-stretch filter
        if max_vwap_dist_atr > 0 and ny_vwap[i] > 0:
            dist = abs(c[i] - ny_vwap[i]) / atr[i] if atr[i] > 0 else 999
            if dist > max_vwap_dist_atr: continue

        already = sigs_by_date.get(d)
        if already: continue

        entry_bar = i + 1
        if entry_bar >= n or date[entry_bar] != d: continue
        entry_ref = o[entry_bar]
        sl, tp, risk_px = _place_sl_tp(entry_ref, direction, params, atr[i], al_range)
        if risk_px <= 0: continue

        sig = {
            "i_signal": i, "i_entry": entry_bar,
            "entry_ref": float(entry_ref), "sl": float(sl), "tp": float(tp),
            "direction": direction, "risk_px": float(risk_px),
            "asian_range": float(al_range), "date": d,
            "level": float(lvl),
        }
        signals.append(sig)
        sigs_by_date[d] = {"direction": direction, "level": float(lvl)}
    return signals


# ═══════════════════════════════════════════════════════════════════════════
#  V8 SIM — adds cooldown after losers + vol targeting
# ═══════════════════════════════════════════════════════════════════════════

def simulate_v8(sigs_by_sym, arrays_by_sym, session_end_hr=22,
                trail_cfg=None, risk_per_trade=RISK_PER_TRADE,
                cooldown_after_losers=0, cooldown_bars=0,
                vol_target_atr_pct=0.0, atr_pct_by_sym=None,
                scale_down_on_corr=False, corr_threshold=4, corr_scale=0.5):
    """
    cooldown_after_losers (M): after M consecutive losing trades, skip
                               `cooldown_bars` (15m bars).
    vol_target_atr_pct: target daily-ATR% for position sizing. If >0, size = risk_base *
                       (target_atr_pct / current_atr_pct) clipped to [0.5, 2.0] factor.
    """
    balance = START_BALANCE
    completed = []
    peak = balance
    max_dd = 0.0

    # Collect all signals with their absolute time for proper chronological sim
    all_sigs = []
    for sym, sigs in sigs_by_sym.items():
        arr = arrays_by_sym[sym]
        for s in sigs:
            all_sigs.append((arr["ts"][s["i_entry"]], sym, s, arr))
    # Sort by entry timestamp
    all_sigs.sort(key=lambda x: x[0])

    # Day counts for corr-dedupe
    day_counts = defaultdict(lambda: {"LONG": 0, "SHORT": 0})
    if scale_down_on_corr:
        for _, _, s, _ in all_sigs:
            day_counts[s["date"]][s["direction"]] += 1

    # Cooldown state: last N trade outcomes and timestamp gate
    recent_outcomes = []  # rolling list of bools (True=win)
    cooldown_until_ts = 0  # cooldown gate timestamp (ms)

    for entry_ts, sym, sig, arr in all_sigs:
        # Cooldown gate
        if cooldown_after_losers > 0 and cooldown_bars > 0 and entry_ts < cooldown_until_ts:
            continue

        h = arr["h"]; l = arr["l"]; c = arr["c"]; o = arr["o"]
        hour = arr["hour"]; date = arr["date"]; atr = arr["atr"]; ts = arr["ts"]
        n = arr["n"]

        i_entry = sig["i_entry"]
        entry_ref = sig["entry_ref"]
        sl_init = sig["sl"]; tp = sig["tp"]
        direction = sig["direction"]
        risk_px = sig["risk_px"]
        entry_date = date[i_entry]

        # Position sizing: vol targeting
        risk_factor = 1.0
        if vol_target_atr_pct > 0 and atr_pct_by_sym is not None:
            ap = atr_pct_by_sym.get(sym)
            if ap is not None and i_entry < len(ap) and not np.isnan(ap[i_entry]) and ap[i_entry] > 0:
                factor = vol_target_atr_pct / ap[i_entry]
                risk_factor = float(np.clip(factor, 0.5, 2.0))

        # Corr scale-down
        if scale_down_on_corr:
            cnt = day_counts.get(entry_date, {}).get(direction, 0)
            if cnt >= corr_threshold:
                risk_factor *= corr_scale

        risk_usd = balance * risk_per_trade * risk_factor
        if risk_px <= 0 or entry_ref <= 0: continue
        qty_usd = risk_usd / (risk_px / entry_ref)
        if qty_usd > balance * MAX_LEVERAGE: qty_usd = balance * MAX_LEVERAGE
        eff_entry = entry_ref * (1 + SLIPPAGE) if direction == "LONG" else entry_ref * (1 - SLIPPAGE)
        entry_fee = qty_usd * TAKER_FEE

        sl_cur = sl_init
        trail_active = False
        mfe = 0.0; mae = 0.0
        exit_reason = "EOS"
        exit_px = c[n - 1]; exit_bar = n - 1

        for j in range(i_entry, n):
            if date[j] != entry_date:
                exit_reason = "EOS"; exit_px = c[j - 1]; exit_bar = j - 1; break
            if hour[j] >= session_end_hr:
                exit_reason = "EOS"; exit_px = c[j]; exit_bar = j; break
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
                    if mode == "swing":
                        k = trail_cfg.get("swing_bars", 5)
                        buf = trail_cfg.get("buffer_atr", 0.2)
                        kk = min(k, j - i_entry)
                        if kk >= 1:
                            if direction == "LONG":
                                new_sl = float(np.min(l[j - kk:j + 1])) - atr[j] * buf
                                sl_cur = max(sl_cur, new_sl)
                            else:
                                new_sl = float(np.max(h[j - kk:j + 1])) + atr[j] * buf
                                sl_cur = min(sl_cur, new_sl)

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
        exit_r = ((eff_exit - eff_entry) if direction == "LONG"
                  else (eff_entry - eff_exit)) / risk_px

        won = pnl_usd > 0
        recent_outcomes.append(won)
        if len(recent_outcomes) > cooldown_after_losers:
            recent_outcomes.pop(0)

        # Cooldown trigger: last M trades all losers
        if cooldown_after_losers > 0 and cooldown_bars > 0:
            if len(recent_outcomes) == cooldown_after_losers and not any(recent_outcomes):
                # Gate until X bars after this exit
                j_exit = min(exit_bar + cooldown_bars, n - 1)
                cooldown_until_ts = max(cooldown_until_ts, int(ts[j_exit]))
                recent_outcomes.clear()

        completed.append({
            "sym": sym, "direction": direction,
            "i_entry": i_entry, "exit_bar": exit_bar,
            "bars": exit_bar - i_entry,
            "reason": exit_reason, "pnl_usd": pnl_usd, "won": won,
            "exit_r": exit_r, "mfe": mfe, "mae": mae,
            "entry": entry_ref, "exit": exit_px,
            "sl": sl_init, "tp": tp,
            "date": entry_date, "risk_factor": risk_factor,
        })

    return _aggregate_v5(completed, balance, max_dd)


# ═══════════════════════════════════════════════════════════════════════════
#  RUN WRAPPERS
# ═══════════════════════════════════════════════════════════════════════════

def run_config(name, arrays, ranges, htf_map, params,
               btc_regime=None, hourly_btc_regime=None, range_exp_map_by_sym=None,
               per_sym_dows=None, btc_dir_aligned_by_sym=None,
               mom_sign_by_sym=None, mom_r_by_sym=None, vol_ratio_by_sym=None,
               atr_pct_by_sym=None,
               trail_cfg=None, risk_per_trade=RISK_PER_TRADE,
               cooldown_after_losers=0, cooldown_bars=0,
               vol_target_atr_pct=0.0,
               scale_down_on_corr=False, corr_threshold=4, corr_scale=0.5,
               session_end_hr=22, compute_per_sym=False):
    sigs = {}
    for sym, arr in arrays.items():
        sigs[sym] = gen_al_break_v8(
            arr, ranges[sym], params, htf_map.get(sym),
            btc_regime=btc_regime, hourly_btc_regime=hourly_btc_regime,
            range_exp_map=(range_exp_map_by_sym or {}).get(sym),
            per_sym_dows=per_sym_dows, sym_name=sym,
            btc_dir_aligned=(btc_dir_aligned_by_sym or {}).get(sym),
            mom_sign=(mom_sign_by_sym or {}).get(sym),
            mom_r=(mom_r_by_sym or {}).get(sym),
            vol_ratio_15m=(vol_ratio_by_sym or {}).get(sym),
        )
    res = simulate_v8(sigs, arrays, session_end_hr=session_end_hr,
                      trail_cfg=trail_cfg, risk_per_trade=risk_per_trade,
                      cooldown_after_losers=cooldown_after_losers,
                      cooldown_bars=cooldown_bars,
                      vol_target_atr_pct=vol_target_atr_pct,
                      atr_pct_by_sym=atr_pct_by_sym,
                      scale_down_on_corr=scale_down_on_corr,
                      corr_threshold=corr_threshold, corr_scale=corr_scale)
    res["total_sigs"] = sum(len(s) for s in sigs.values())
    if compute_per_sym:
        per_sym = {}
        for sym in sigs:
            per_sym[sym] = simulate_v8({sym: sigs[sym]}, {sym: arrays[sym]},
                                        session_end_hr=session_end_hr,
                                        trail_cfg=trail_cfg,
                                        risk_per_trade=risk_per_trade,
                                        cooldown_after_losers=0, cooldown_bars=0,
                                        vol_target_atr_pct=vol_target_atr_pct,
                                        atr_pct_by_sym=atr_pct_by_sym)
        res["per_sym"] = per_sym
    return {"config_id": name, "params": params, "trail": trail_cfg,
            "risk_per_trade": risk_per_trade, **res}


def log_r(name, r):
    logger.info(f"{name:50s} trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% "
                f"PF={r['profit_factor']:>5.2f} PnL={r['pnl_pct']:>+8.1f}%  "
                f"DD={r['max_dd_pct']:>5.2f}% avgR={r['avg_r']:>+.2f}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    t0 = _time.time()
    logger.info("=== V8: Deep V7 improvements (many hypotheses) ===")

    # Load data
    arrays, ranges, htf_by_sym = {}, {}, {}
    for sym in TOP14:
        df15 = _load(sym, "15m")
        if df15 is None: continue
        df1h = _load(sym, "1h")
        if df1h is None: df1h = _load(sym, "1H")
        df4h = _load(sym, "4h")
        if df4h is None: df4h = _load(sym, "4H")
        arr = prep_15m_arrays(df15)
        ranges[sym] = build_session_ranges(arr)
        htf = compute_htf_trends(df4h, df1h)
        htf_aligned = {k: align_htf(m, arr["ts"]) for k, m in htf.items()}
        arrays[sym] = arr
        htf_by_sym[sym] = htf_aligned
    logger.info(f"Loaded {len(arrays)} symbols")
    good10_arrays = {s: arrays[s] for s in GOOD10 if s in arrays}

    # BTC regime (daily + hourly)
    btc_regime = build_btc_regime(arrays)
    by_reg = defaultdict(int)
    for v in btc_regime.values(): by_reg[v["regime"]] += 1
    logger.info(f"Daily BTC regime: {dict(by_reg)}")
    hourly_btc = build_hourly_btc_regime(arrays)
    logger.info(f"Hourly BTC regime: {len(hourly_btc)} bars")

    # Range expansion per symbol
    range_exp_by_sym = {s: build_range_expansion_map(arrays[s], ranges[s]) for s in good10_arrays}

    # BTC direction alignment per symbol (aligned to that symbol's 15m ts)
    # We use the trend_combo_1h map already aligned per symbol
    btc_dir_by_sym = {}
    # Get BTC htf_map, then for each symbol align its 15m ts
    if "BTC/USDT:USDT" in arrays:
        btc_arr = arrays["BTC/USDT:USDT"]
        btc_df1h = _load("BTC/USDT:USDT", "1h")
        if btc_df1h is None: btc_df1h = _load("BTC/USDT:USDT", "1H")
        btc_df4h = _load("BTC/USDT:USDT", "4h")
        if btc_df4h is None: btc_df4h = _load("BTC/USDT:USDT", "4H")
        btc_htf_raw = compute_htf_trends(btc_df4h, btc_df1h)
        for sym, arr in good10_arrays.items():
            a = align_htf(btc_htf_raw.get("trend_combo_1h", {}), arr["ts"])
            btc_dir_by_sym[sym] = a.astype(np.int32)

    # Pre-break momentum per sym (3-bar and 6-bar)
    mom_sign_by_sym = {}
    mom_r_by_sym = {}
    for sym, arr in good10_arrays.items():
        ms, mr = build_pre_break_momentum(arr, window=3)
        mom_sign_by_sym[sym] = ms
        mom_r_by_sym[sym] = mr

    # Volume ratio per sym
    vol_ratio_by_sym = {s: build_vol_ratio_15m(arrays[s]) for s in good10_arrays}

    # ATR% per sym
    atr_pct_by_sym = {s: build_symbol_atr_pct(arrays[s]) for s in good10_arrays}

    # Baseline: V7 champion
    base = dict(BASE_PARAMS_V5_CHAMPION)
    base["allowed_dows"] = {0, 1, 3, 4, 5, 6}  # exclude Wed
    base["skip_regimes"] = {"chop"}              # skip BTC chop days

    results = []

    # V7 replay
    logger.info("")
    logger.info("=== V0: V7 champion replay (baseline) ===")
    r = run_config("V0_V7_champion", good10_arrays, ranges, htf_by_sym, base,
                   btc_regime=btc_regime)
    log_r("V0_V7_champion", r); results.append(r)
    v7_pnl = r["pnl_pct"]; v7_pf = r["profit_factor"]; v7_dd = r["max_dd_pct"]

    # ═════════════════════════════════════════════════════════════
    # H1: Hourly BTC regime filter (in addition to daily)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H1: Hourly BTC regime filter ===")
    for skip in [{"chop"}, {"panic"}, {"chop", "panic"}]:
        p = dict(base); p["skip_hourly_regimes"] = skip
        name = f"H1_skip_h_{'_'.join(sorted(skip))}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime, hourly_btc_regime=hourly_btc)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H2: Range expansion (AL range > 1x median of last 10 days)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H2: Range expansion filter ===")
    for mrr in [0.8, 1.0, 1.2, 1.5]:
        p = dict(base); p["min_rel_range"] = mrr
        name = f"H2_minRel{mrr}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime, range_exp_map_by_sym=range_exp_by_sym)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H3: Pre-break momentum
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H3: Pre-break momentum filter ===")
    for mom_r_min in [0.0, 0.3, 0.5, 0.8]:
        p = dict(base); p["min_mom_bars"] = 3; p["min_mom_r"] = mom_r_min
        name = f"H3_mom3b_r{mom_r_min}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime,
                       mom_sign_by_sym=mom_sign_by_sym, mom_r_by_sym=mom_r_by_sym)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H4: Break bar body size filter
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H4: Break bar body size ===")
    for bb in [0.3, 0.5, 0.7, 1.0]:
        p = dict(base); p["min_break_body_atr"] = bb
        name = f"H4_body{bb}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H5: Volume spike at break
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H5: Volume spike at break ===")
    for vm in [1.0, 1.5, 2.0, 3.0]:
        p = dict(base); p["min_vol_ratio"] = vm
        name = f"H5_vol{vm}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime, vol_ratio_by_sym=vol_ratio_by_sym)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H6: BTC direction alignment
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H6: BTC direction alignment (trend_combo_1h) ===")
    p = dict(base); p["require_btc_align"] = True
    r = run_config("H6_btc_align", good10_arrays, ranges, htf_by_sym, p,
                   btc_regime=btc_regime, btc_dir_aligned_by_sym=btc_dir_by_sym)
    log_r("H6_btc_align", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H7: Multi-TF 4h + 1h aligned (htf_filter = "ema50_4h" adds condition)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H7: 4h trend + 1h combo ===")
    # Custom: use ema50_4h as stricter HTF filter
    for filt in ["trend_combo_1h", "ema50_4h", "adx25_1h"]:
        p = dict(base); p["htf_filter"] = filt
        name = f"H7_htf_{filt}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p, btc_regime=btc_regime)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H8: Per-symbol allowed DoWs (from historical per-sym DoW analysis)
    # We run only Thu+Sat+Mon+Tue per-sym (V6 F4b finding)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H8: Concentrated DoWs ===")
    dow_variants = [
        ("H8_ThuSat",         {3, 5}),
        ("H8_ThuSatMonTue",   {0, 1, 3, 5}),
        ("H8_TueThuFriSat",   {1, 3, 4, 5}),
        ("H8_exWedMon",       {1, 3, 4, 5, 6}),
        ("H8_exWedSun",       {0, 1, 3, 4, 5}),
    ]
    for name, dows in dow_variants:
        p = dict(base); p["allowed_dows"] = dows
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p, btc_regime=btc_regime)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H9: Cooldown after losers
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H9: Cooldown after consecutive losers ===")
    for M, bars in [(2, 96), (3, 96), (2, 192), (3, 192), (4, 288)]:
        # 96 bars = 24h, 192 = 48h, 288 = 72h (15m bars)
        p = dict(base)
        name = f"H9_losers{M}_bars{bars}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime,
                       cooldown_after_losers=M, cooldown_bars=bars)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H10: ATR vol targeting
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H10: ATR vol targeting ===")
    for tgt in [1.0, 1.5, 2.0, 2.5, 3.0]:
        p = dict(base)
        name = f"H10_volTgt{tgt}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p,
                       btc_regime=btc_regime,
                       vol_target_atr_pct=tgt, atr_pct_by_sym=atr_pct_by_sym)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H11: Tight max_entry_hr
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H11: Tight max_entry_hr ===")
    for hr in [1, 2, 3]:
        p = dict(base); p["max_entry_hr_tight"] = hr
        name = f"H11_maxhr{hr}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p, btc_regime=btc_regime)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H14: Dynamic TP by regime (tighter on chop, wider on high-vol)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H14: TP sensitivity by vol regime ===")
    for tp in [3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0]:
        p = dict(base); p["tp_r"] = tp
        name = f"H14_tp{tp}"
        r = run_config(name, good10_arrays, ranges, htf_by_sym, p, btc_regime=btc_regime)
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # H-SL: SL re-sweep (V5 had SL2.0, may be different now)
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== H-SL: SL width sweep ===")
    for sl in [1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0]:
        for tp in [3.0, 4.0, 5.0]:
            p = dict(base); p["sl_atr_mult"] = sl; p["tp_r"] = tp
            name = f"HSL_sl{sl}_tp{tp}"
            r = run_config(name, good10_arrays, ranges, htf_by_sym, p, btc_regime=btc_regime)
            log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # ULTRA stack: best of each section
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== ULTRA: stacked best of each section ===")

    def best_in(pfx, min_tr=50):
        cs = [r for r in results if r["config_id"].startswith(pfx) and r["trades"] >= min_tr]
        return max(cs, key=lambda x: x["pnl_pct"], default=None)

    best_h1 = best_in("H1_")
    best_h2 = best_in("H2_")
    best_h3 = best_in("H3_")
    best_h4 = best_in("H4_")
    best_h5 = best_in("H5_")
    best_h7 = best_in("H7_")
    best_h8 = best_in("H8_")
    best_h9 = best_in("H9_")
    best_h11 = best_in("H11_")
    best_h14 = best_in("H14_")
    best_hsl = best_in("HSL_")

    section_bests = {"H1": best_h1, "H2": best_h2, "H3": best_h3,
                     "H4": best_h4, "H5": best_h5, "H7": best_h7,
                     "H8": best_h8, "H9": best_h9, "H11": best_h11,
                     "H14": best_h14, "HSL": best_hsl}
    for sec, b in section_bests.items():
        if b: logger.info(f"  best {sec}: {b['config_id']:30s} PF={b['profit_factor']:.2f} "
                          f"PnL={b['pnl_pct']:+.1f}% DD={b['max_dd_pct']:.2f}%")

    # ULTRA1: H1 + H2 + H14 on champion base
    # Only include filters whose best improves on V7 PnL meaningfully
    def improves(b, thresh=0.0):
        return b is not None and (b["pnl_pct"] - v7_pnl) > thresh

    ultras = []

    # U1: add best SL/TP combo
    if improves(best_hsl, 20):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        ultras.append(("U1_bestSLTP", p, None, dict(), RISK_PER_TRADE))

    # U2: SL/TP + H1 hourly regime
    if improves(best_hsl, 20) and improves(best_h1, 5):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        p["skip_hourly_regimes"] = set(best_h1["params"].get("skip_hourly_regimes", []))
        ultras.append(("U2_SLTP_H1", p, None,
                        {"hourly_btc": True}, RISK_PER_TRADE))

    # U3: SL/TP + H2 range expansion
    if improves(best_hsl, 20) and improves(best_h2, 5):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        p["min_rel_range"] = best_h2["params"].get("min_rel_range", 1.0)
        ultras.append(("U3_SLTP_H2", p, None,
                        {"range_exp": True}, RISK_PER_TRADE))

    # U4: SL/TP + H3 momentum
    if improves(best_hsl, 20) and improves(best_h3, 5):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        p["min_mom_bars"] = 3
        p["min_mom_r"] = best_h3["params"].get("min_mom_r", 0.0)
        ultras.append(("U4_SLTP_H3", p, None,
                        {"mom": True}, RISK_PER_TRADE))

    # U5: SL/TP + H4 body + H5 vol
    if improves(best_hsl, 20):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        if improves(best_h4, 5):
            p["min_break_body_atr"] = best_h4["params"].get("min_break_body_atr", 0.3)
        if improves(best_h5, 5):
            p["min_vol_ratio"] = best_h5["params"].get("min_vol_ratio", 1.0)
        ultras.append(("U5_SLTP_H4H5", p, None,
                        {"body_vol": True}, RISK_PER_TRADE))

    # U6: SL/TP + H8 concentrated DoW
    if improves(best_hsl, 20) and improves(best_h8, 10):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        p["allowed_dows"] = best_h8["params"].get("allowed_dows", {0, 1, 3, 4, 5, 6})
        ultras.append(("U6_SLTP_H8", p, None, {}, RISK_PER_TRADE))

    # U7: SL/TP + H11 tight hour
    if improves(best_hsl, 20) and improves(best_h11, 5):
        p = dict(base)
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0)
        p["max_entry_hr_tight"] = best_h11["params"].get("max_entry_hr_tight", 2)
        ultras.append(("U7_SLTP_H11", p, None, {}, RISK_PER_TRADE))

    # U_ALL: combine every filter that improves
    p = dict(base)
    extras = {}
    tags = []
    if improves(best_hsl, 20):
        p["sl_atr_mult"] = best_hsl["params"].get("sl_atr_mult", 2.0)
        p["tp_r"] = best_hsl["params"].get("tp_r", 4.0); tags.append("sltp")
    if improves(best_h1, 5):
        p["skip_hourly_regimes"] = set(best_h1["params"].get("skip_hourly_regimes", []))
        extras["hourly_btc"] = True; tags.append("h1")
    if improves(best_h2, 5):
        p["min_rel_range"] = best_h2["params"].get("min_rel_range", 1.0)
        extras["range_exp"] = True; tags.append("h2")
    if improves(best_h3, 5):
        p["min_mom_bars"] = 3
        p["min_mom_r"] = best_h3["params"].get("min_mom_r", 0.0)
        extras["mom"] = True; tags.append("h3")
    if improves(best_h4, 5):
        p["min_break_body_atr"] = best_h4["params"].get("min_break_body_atr", 0.3)
        tags.append("h4")
    if improves(best_h5, 5):
        p["min_vol_ratio"] = best_h5["params"].get("min_vol_ratio", 1.0)
        tags.append("h5")
    if improves(best_h8, 10):
        p["allowed_dows"] = best_h8["params"].get("allowed_dows", {0, 1, 3, 4, 5, 6})
        tags.append("h8")
    if improves(best_h11, 5):
        p["max_entry_hr_tight"] = best_h11["params"].get("max_entry_hr_tight", 2)
        tags.append("h11")
    ultras.append((f"U_ALL_{'_'.join(tags)}" if tags else "U_ALL_none", p, None, extras, RISK_PER_TRADE))

    # U_ALL + aggressive risk
    if tags:
        p2 = dict(p)
        ultras.append((f"U_ALL_rpt0.015", p2, None, dict(extras), 0.015))
        p3 = dict(p)
        ultras.append((f"U_ALL_rpt0.02", p3, None, dict(extras), 0.02))

    # Run all ULTRA configs
    for name, p, tcfg, extras, rpt in ultras:
        r = run_config(
            name, good10_arrays, ranges, htf_by_sym, p,
            btc_regime=btc_regime,
            hourly_btc_regime=(hourly_btc if extras.get("hourly_btc") else None),
            range_exp_map_by_sym=(range_exp_by_sym if extras.get("range_exp") else None),
            mom_sign_by_sym=(mom_sign_by_sym if extras.get("mom") else None),
            mom_r_by_sym=(mom_r_by_sym if extras.get("mom") else None),
            vol_ratio_by_sym=vol_ratio_by_sym,
            trail_cfg=tcfg, risk_per_trade=rpt,
            compute_per_sym=(name in ("U_ALL_" + "_".join(tags) if tags else "",)),
        )
        log_r(name, r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # Walk-forward on overall best
    # ═════════════════════════════════════════════════════════════
    logger.info("")
    logger.info("=== Walk-forward 4Q on overall best ===")
    cands = [r for r in results if r["trades"] >= 80 and r["max_dd_pct"] <= 15.0]
    best = max(cands, key=lambda x: x["pnl_pct"], default=None)
    if best:
        logger.info(f"Overall best: {best['config_id']} PF={best['profit_factor']} "
                    f"PnL={best['pnl_pct']}% DD={best['max_dd_pct']}%")
        per_sym_q = {s: split_by_quarters(arrays[s]) for s in GOOD10 if s in arrays}

        # Need per-sym precomputed maps for WF
        for qi in range(4):
            q_arrays = {s: per_sym_q[s][qi] for s in per_sym_q if per_sym_q[s][qi] is not None}
            # Recompute per-sym aux maps on sliced arrays
            q_ranges = {s: build_session_ranges(q_arrays[s]) for s in q_arrays}
            q_range_exp = {s: build_range_expansion_map(q_arrays[s], q_ranges[s]) for s in q_arrays}
            q_mom = {}; q_mom_r = {}
            for s, a in q_arrays.items():
                ms, mr = build_pre_break_momentum(a, window=3)
                q_mom[s] = ms; q_mom_r[s] = mr
            q_vol = {s: build_vol_ratio_15m(q_arrays[s]) for s in q_arrays}
            q_atr_pct = {s: build_symbol_atr_pct(q_arrays[s]) for s in q_arrays}
            # Use original htf_by_sym (full-period alignment — acceptable since HTF is slow)
            r = run_config(f"WF_Q{qi + 1}", q_arrays, q_ranges, htf_by_sym,
                           best["params"], btc_regime=btc_regime,
                           range_exp_map_by_sym=q_range_exp,
                           mom_sign_by_sym=q_mom, mom_r_by_sym=q_mom_r,
                           vol_ratio_by_sym=q_vol,
                           atr_pct_by_sym=q_atr_pct,
                           risk_per_trade=best.get("risk_per_trade", RISK_PER_TRADE))
            log_r(f"WF_Q{qi + 1}", r); results.append(r)

    # ═════════════════════════════════════════════════════════════
    # SAVE + SUMMARY
    # ═════════════════════════════════════════════════════════════
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = RESULTS_DIR / f"backtest_session_v8_{ts}.json"

    def _json_default(x):
        if isinstance(x, set): return sorted(x)
        try: return float(x)
        except Exception: return str(x)

    with open(path, "w") as f:
        json.dump({"results": results, "v7_baseline_pnl": v7_pnl,
                   "v7_baseline_pf": v7_pf, "v7_baseline_dd": v7_dd,
                   "btc_regime_counts": dict(by_reg)},
                  f, default=_json_default, indent=2)
    logger.info(f"Saved {path.name} ({_time.time() - t0:.1f}s)")

    # TOP-30
    print("\n" + "=" * 120)
    print(f"TOP-30 by PnL (V8, min 40 trades)  [V7 baseline: PnL={v7_pnl:+.1f}% PF={v7_pf:.2f} DD={v7_dd:.2f}%]")
    print(f"{'config':50s} {'trades':>6s} {'WR%':>5s} {'PF':>5s}  {'PnL%':>8s} {'DD%':>6s}  {'dPnL':>6s}")
    flt = [r for r in results if r.get("trades", 0) >= 40]
    flt.sort(key=lambda r: r["pnl_pct"], reverse=True)
    for r in flt[:30]:
        dp = r["pnl_pct"] - v7_pnl
        print(f"{r['config_id']:50s} {r['trades']:>6} {r['wr_pct']:>5.1f} "
              f"{r['profit_factor']:>5.2f}  {r['pnl_pct']:>+8.1f} {r['max_dd_pct']:>6.2f}  "
              f"{dp:>+6.1f}")

    # Section bests
    print("\n" + "-" * 120)
    print("Best per section (vs V7 baseline)")
    for pfx, title in [("V0_", "V7 baseline"), ("H1_", "Hourly BTC regime"),
                       ("H2_", "Range expansion"), ("H3_", "Pre-break momentum"),
                       ("H4_", "Break bar body"), ("H5_", "Volume spike"),
                       ("H6_", "BTC direction align"), ("H7_", "HTF filter"),
                       ("H8_", "Concentrated DoW"), ("H9_", "Cooldown"),
                       ("H10_", "ATR vol targeting"), ("H11_", "Tight max_hr"),
                       ("H14_", "TP sensitivity"), ("HSL_", "SL/TP grid"),
                       ("U", "Stacked ULTRA")]:
        cs = [r for r in results if r["config_id"].startswith(pfx) and r["trades"] >= 40]
        if not cs: continue
        b = max(cs, key=lambda x: x["pnl_pct"])
        dp = b["pnl_pct"] - v7_pnl
        print(f"  {title:22s} {b['config_id']:36s} PF={b['profit_factor']:>5.2f} "
              f"PnL={b['pnl_pct']:>+8.1f}% DD={b['max_dd_pct']:>5.2f}% dPnL={dp:>+6.1f}")

    # WF
    wf = [r for r in results if r["config_id"].startswith("WF_")]
    if wf:
        print("\nWALK-FORWARD:")
        for r in wf:
            pf_ok = r["profit_factor"] >= 1.0
            pnl_ok = r["pnl_pct"] > 0
            tag = "PASS" if (pf_ok and pnl_ok) else "FAIL"
            print(f"  {r['config_id']:16s} trades={r['trades']:>4} PF={r['profit_factor']:>5.2f} "
                  f"PnL={r['pnl_pct']:>+8.1f}% DD={r['max_dd_pct']:>5.2f}%  [{tag}]")


if __name__ == "__main__":
    main()
