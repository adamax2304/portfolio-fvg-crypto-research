"""
universe_v7_test.py — V7 на разных universe в FAIR и STRICT режимах
======================================================================================
FAIR  = SLIPPAGE 0.05%, SL fill по цене SL (как simulate_v7 → дал +1525%/23.8% DD)
STRICT = SLIPPAGE 0.10%, SL fill по wick lo/hi (максимально пессимистично)

Universe selectors:
  • BASELINE_27        — оригинальный пул из portfolio_v7.py
  • QUALITY_STATIC     — ~40 ликвидных не-мем монет (мой ручной + composite фильтр)
  • TOP_30_VOL_ROLLING — топ-30 по объёму, ребалансировка ежемесячно
  • COMPOSITE_30       — composite score, ребалансировка ежемесячно

Risk stack (одинаков везде):
  pivot_dist<7×ATR, corr max=1@0.80, L4 FOMC pause, leverage check,
  SL cooldown 4h, max_pos=5, risk=4%, bear=long_only

Запуск:
    py -3 multi_agent/backtests/universe_v7_test.py
"""
from __future__ import annotations
import sys
import json
from collections import defaultdict, deque
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import (
    load, build_regime, atr as _atr, PortfolioSignal, _ts_period_ms,
    START_BALANCE, TAKER_FEE, FUNDING_PER_8H, SLIPPAGE as STD_SLIPPAGE,
    SL_COOLDOWN_MS, MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN,
)
from multi_agent.backtests.portfolio_v7 import BASELINE, CACHE_DIR
from multi_agent.backtests.full_audit_v7 import collect_portfolio
from multi_agent.backtests.news_aware_v7 import load_all_events, get_event_context
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter
from multi_agent.backtests.risk_controls_v7 import build_corr_matrix
from multi_agent.backtests.universe_filters import (
    list_available_symbols,
    compute_monthly_metrics,
    selector_top_volume,
    selector_composite,
)

# ── Параметры ──────────────────────────────────────────────────────────────────
PIVOT_DIST = 7.0
TAG        = "730d"
RISK       = 0.04
MAX_POS    = 5

# V7 best-config params (из full_audit_v7.py)
ST_KW  = dict(tp_r=2.5,  buffer_atr=0.3, pivot_lookback=5)
SC_KW  = dict(tp_r=4.0,  buffer_atr=0.3, pivot_lookback=3)
RSI_KW = dict(sl_atr=2.0, tp_r=3.0)
FIB_KW = dict(consolidation_bars=18, pump_min_atr=3.0, pump_max_bars=6,
              retrace_min=0.382, retrace_max=0.618,
              tp_extension=1.0, sl_buffer_atr=0.5)

# ── Параметры симуляции (FAIR vs STRICT) ───────────────────────────────────────
STRICT_SLIPPAGE = 0.0010   # 0.10% за сторону
FAIR_SLIPPAGE   = 0.0005   # 0.05% — как в simulate_v7
DAY_MS          = 86_400_000

# ── QUALITY_STATIC blacklist ────────────────────────────────────────────────────
# Монеты которые мы исключаем как явно не-trending / манипулируемые / мемы / новые.
# Список построен на основе:
#   - фактически замеренных wick_body_ratio + max_move_atr,
#   - даты листинга < 18 мес → мало истории,
#   - meme-status / низкая ликвидность.
QUALITY_BLACKLIST = {
    # === МЕМЫ (выбивают стопы тенями) ===
    "1000PEPE", "1000BONK",
    # === Новые/малая история (листинг 2024+ → мало 730d) ===
    "MANTA",   # Jan-2024
    "JUP",     # Jan-2024
    "JTO",     # Dec-2023
    "ONDO",    # Jan-2024
    "PYTH",    # Nov-2023
    # === Высокая манипулируемость / низкий float ===
    "WLD",     # Worldcoin — низкий float, pump'n'dump паттерны
    "TAO",     # Bittensor — спайки, низкая ликвидность
    "SEI",     # Резкие импульсы
    "TIA",     # Celestia — мало истории + спайки
    # === Слабый тренд / range-bound ===
    "GALA",    # Gaming, постоянные пилы
    "SAND",    # Metaverse, низкий мочкоhи объём
    "GRT",     # The Graph — болтается в range
    "ALGO",    # ALGO — практически dead коин
    "EGLD",    # MultiversX — low vol, шум
}

# Дополнительно: hard-keep "core" ликвидные монеты (top market cap),
# которые точно должны попасть в QUALITY если есть данные.
QUALITY_HARDKEEP = {
    "BTC", "ETH", "BNB", "SOL", "XRP", "DOGE", "ADA", "TRX", "AVAX", "TON",
    "LINK", "DOT", "BCH", "LTC", "NEAR", "ATOM", "UNI", "APT", "SUI", "ARB",
    "OP", "INJ", "AAVE", "FIL", "ICP", "HBAR", "XLM", "RENDER", "LDO", "CRV",
}


def selector_quality_static(metrics_by_sym: dict, all_syms: list[str]) -> dict[str, list[str]]:
    """
    Quality static universe ~40 монет:
      1. Hard exclude blacklist (memes, manipulated, new).
      2. Hard keep core liquid majors (top-30 market cap).
      3. Из остальных доступных — добавляем по composite score (liq×stability×cleanness)
         пока не дойдём до 40 (или сколько есть).
    Static (без monthly rebalance).
    """
    # Step 1: filter by blacklist
    avail = [s for s in all_syms if s not in QUALITY_BLACKLIST]

    # Step 2: hard keep (intersect with available)
    hard = [s for s in avail if s in QUALITY_HARDKEEP]

    # Step 3: rank остальных по среднему composite score
    rest = [s for s in avail if s not in QUALITY_HARDKEEP]

    def avg_score(sym: str) -> float:
        ml = metrics_by_sym.get(sym, [])
        if not ml:
            return -999
        # liq * cleanness — favors high vol + clean candles
        scores = []
        for m in ml:
            liq   = np.log1p(m.dollar_volume_30d)            # log of vol
            clean = 1.0 / (1.0 + m.wick_body_ratio * m.max_move_atr)  # inverse manipulation
            scores.append(liq * clean)
        return float(np.mean(scores)) if scores else -999

    rest_ranked = sorted(rest, key=avg_score, reverse=True)
    target = 40
    fill = rest_ranked[: max(0, target - len(hard))]
    quality = sorted(hard) + sorted(fill)

    return {"all": quality}


# ─────────────────────────────────────────────────────────────────────────────
# Monthly returns из trades
# ─────────────────────────────────────────────────────────────────────────────

def _monthly_rets_from_trades(trades: list,
                               start_balance: float = START_BALANCE) -> list[float]:
    """Return % per month based on closed trades sorted by exit_ts."""
    if not trades:
        return []
    by_month: dict[str, float] = defaultdict(float)
    for t in sorted(trades, key=lambda x: x["exit_ts"]):
        m = pd.Timestamp(t["exit_ts"], unit="ms", tz="UTC").strftime("%Y-%m")
        by_month[m] += t["pnl"]
    rets = []
    bal = start_balance
    for m in sorted(by_month):
        pnl = by_month[m]
        rets.append(pnl / bal * 100)
        bal = max(0.01, bal + pnl)
    return rets


# ─────────────────────────────────────────────────────────────────────────────
# СТРОГАЯ симуляция (wick SL fills + higher slippage)
# Базируется на simulate_rc но с двумя ключевыми отличиями:
#   1. SLIPPAGE = STRICT_SLIPPAGE (0.10%)
#   2. SL-fill: LONG → lo[k], SHORT → hi[k]  (худший wick цена)
# ─────────────────────────────────────────────────────────────────────────────

def simulate_strict(
    sigs, sim_dfs, regime, events,
    base_risk      = RISK,
    max_pos        = MAX_POS,
    enable_l4      = True,
    max_corr_open  = 1,
    corr_threshold = 0.80,
    corr_matrix    = None,
    mode           = "strict",   # "fair" (slip 0.05%, SL@sl0) or "strict" (slip 0.10%, SL@wick)
):
    """
    Симуляция в одном из двух режимов:
      mode="fair"   — SLIPPAGE 0.05%, SL fill = sl0 (как simulate_v7 → +1525%/23.8% baseline)
      mode="strict" — SLIPPAGE 0.10%, SL fill = wick lo/hi (максимально пессимистично)
    """
    slip       = STRICT_SLIPPAGE if mode == "strict" else FAIR_SLIPPAGE
    wick_fill  = (mode == "strict")

    balance  = float(START_BALANCE)
    open_pos = {}
    sym_cool = {}
    trades   = []
    skipped  = defaultdict(int)
    atr_cache = {}

    _day_date      = None
    _day_start_eq  = balance
    _week_key      = None
    _week_start_eq = balance

    def _realize_closed(before_ts: int):
        nonlocal balance
        for k in list(open_pos.keys()):
            pos = open_pos[k]
            if pos["exit_ts"] is not None and pos["exit_ts"] <= before_ts:
                balance = max(0.01, balance + pos["pnl"])
                trades.append(pos)
                if pos["reason"] == "SL":
                    sym_cool[pos["sym"]] = pos["exit_ts"] + SL_COOLDOWN_MS
                del open_pos[k]

    for sig in sigs:
        key = (sig.sym, sig.strat)
        if key in open_pos:
            continue
        if sym_cool.get(sig.sym, 0) > sig.next_open_ts:
            continue

        _realize_closed(sig.next_open_ts)

        if len(open_pos) >= max_pos:
            continue

        ts_dt    = pd.Timestamp(sig.next_open_ts, unit="ms", tz="UTC")
        cur_day  = ts_dt.date()
        cur_week = (ts_dt.year, ts_dt.isocalendar()[1])

        if cur_day != _day_date:
            _day_date    = cur_day
            _day_start_eq = balance
        if cur_week != _week_key:
            _week_key     = cur_week
            _week_start_eq = balance

        # L4 macro pause
        risk_mult = 1.0
        if enable_l4:
            ctx = get_event_context(sig.next_open_ts, events)
            if ctx["context"] == "pre_macro" and ctx["action"] == "PAUSE":
                skipped["l4_pre_macro"] += 1
                continue
            if ctx["context"] == "post_macro":
                risk_mult = ctx.get("risk_mult", 1.0)

        risk = base_risk * max(0.2, min(2.0, risk_mult))

        # Correlation filter
        if (max_corr_open > 0
                and corr_matrix is not None
                and not corr_matrix.empty
                and sig.sym in corr_matrix.index):
            open_syms = [p["sym"] for p in open_pos.values()]
            n_corr = sum(
                1 for os in open_syms
                if os != sig.sym
                and os in corr_matrix.index
                and abs(corr_matrix.at[os, sig.sym]) >= corr_threshold
            )
            if n_corr >= max_corr_open:
                skipped["rc5_corr"] += 1
                continue

        df = sim_dfs.get(key)
        if df is None:
            continue

        ts  = df["timestamp"].to_numpy(int)
        op  = df["open"].to_numpy(float)
        hi  = df["high"].to_numpy(float)
        lo  = df["low"].to_numpy(float)
        cl  = df["close"].to_numpy(float)
        n   = len(ts)
        period_ms       = _ts_period_ms(ts)
        funding_per_bar = FUNDING_PER_8H * (period_ms / (8 * 3_600_000))

        j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
        if j >= n - 1:
            continue

        if key not in atr_cache:
            atr_cache[key] = _atr(hi, lo, cl, 14)

        d       = sig.direction
        # Entry с повышенным слиппажем
        entry_p = op[j] * (1 + slip) if d == "LONG" else op[j] * (1 - slip)
        sl0     = sig.sl

        if d == "LONG":
            dist = entry_p - sl0
            if dist <= 0:
                continue
            tp_p = (entry_p + (sig.tp - sig.entry) * (dist / sig.dist)
                    if sig.dist > 0 else sig.tp)
        else:
            dist = sl0 - entry_p
            if dist <= 0:
                continue
            tp_p = (entry_p - (sig.entry - sig.tp) * (dist / sig.dist)
                    if sig.dist > 0 else sig.tp)

        risk_usd = balance * risk
        qty      = risk_usd / dist
        notional = entry_p * qty
        max_margin = balance * MAX_MARGIN_FRAC
        if max_margin > 0:
            lev = notional / max_margin
            if lev > MAX_LEVERAGE:
                skipped["over_leverage"] += 1
                continue
            if notional / max(lev, 1) < MIN_MARGIN:
                continue

        # Walk forward
        exit_p = None; exit_bar = None; reason = "timeout"
        for k in range(j, min(j + 500, n)):
            h_k = hi[k]; l_k = lo[k]
            if d == "LONG":
                if l_k <= sl0:
                    # FAIR: fill по цене SL; STRICT: fill по wick low (хуже)
                    exit_p = l_k if wick_fill else sl0
                    reason = "SL"; exit_bar = k; break
                if h_k >= tp_p:
                    exit_p = tp_p
                    reason = "TP"; exit_bar = k; break
            else:
                if h_k >= sl0:
                    exit_p = h_k if wick_fill else sl0
                    reason = "SL"; exit_bar = k; break
                if l_k <= tp_p:
                    exit_p = tp_p
                    reason = "TP"; exit_bar = k; break

        if exit_bar is None:
            exit_p   = cl[min(j + 499, n - 1)]
            exit_bar = min(j + 499, n - 1)

        bars_held = exit_bar - j + 1

        if d == "LONG":
            slip_exit = exit_p * (1 - slip)
            gross     = (slip_exit - entry_p) * qty
        else:
            slip_exit = exit_p * (1 + slip)
            gross     = (entry_p - slip_exit) * qty

        entry_fee = entry_p   * qty * TAKER_FEE
        exit_fee  = slip_exit * qty * TAKER_FEE
        funding   = entry_p   * qty * funding_per_bar * bars_held
        pnl       = gross - entry_fee - exit_fee - funding

        open_pos[key] = {
            "sym":      sig.sym,
            "strat":    sig.strat,
            "direction": d,
            "entry_ts": int(ts[j]),
            "exit_ts":  int(ts[exit_bar]),
            "entry":    entry_p,
            "exit":     slip_exit,
            "sl":       sl0,
            "reason":   reason,
            "pnl":      pnl,
            "risk_usd": risk_usd,
            "pnl_r":    pnl / risk_usd if risk_usd > 0 else 0,
            "win":      pnl > 0,
        }

    _realize_closed(999_999_999_999_999)

    trades.sort(key=lambda x: x["exit_ts"])
    eq = float(START_BALANCE); peak = float(START_BALANCE); max_dd = 0.0
    for t in trades:
        eq   = max(0.01, eq + t["pnl"])
        peak = max(peak, eq)
        dd   = (peak - eq) / peak * 100 if peak > 0 else 0
        max_dd = max(max_dd, dd)

    if not trades:
        return {"trades": [], "metrics": {
            "n": 0, "wr": 0, "pf": 0, "avg_r": 0,
            "return_pct": 0, "max_dd_pct": 0, "final_balance": float(START_BALANCE),
        }, "skipped": dict(skipped)}

    wins   = [t for t in trades if t["pnl"] > 0]
    losses = [t for t in trades if t["pnl"] <= 0]
    gw     = sum(t["pnl"] for t in wins)
    gl     = abs(sum(t["pnl"] for t in losses))
    pf     = gw / gl if gl > 0 else 99.0
    wr     = len(wins) / len(trades) * 100
    avg_r  = float(np.mean([t["pnl_r"] for t in trades]))
    ret    = (eq - START_BALANCE) / START_BALANCE * 100

    reasons = defaultdict(int)
    for t in trades:
        reasons[t["reason"]] += 1

    return {
        "trades":  trades,
        "skipped": dict(skipped),
        "reasons": dict(reasons),
        "metrics": {
            "n":             len(trades),
            "wr":            round(wr, 1),
            "pf":            round(pf, 2),
            "avg_r":         round(avg_r, 3),
            "return_pct":    round(ret, 1),
            "max_dd_pct":    round(max_dd, 1),
            "final_balance": round(eq, 2),
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Month-aware signal filter
# ─────────────────────────────────────────────────────────────────────────────

def _filter_by_month(sigs: list, active_by_month: dict) -> list:
    """
    Фильтр по активному universe в каждом месяце.
      active_by_month = {"all": [...]}        → статичный пул
      active_by_month = {"2024-01": [...], …} → dynamic monthly
    """
    if not active_by_month:
        return sigs

    def is_active(sym: str, ts_ms: int) -> bool:
        m_str = str(pd.Timestamp(ts_ms, unit="ms", tz="UTC").to_period("M"))
        if m_str in active_by_month:
            return sym in active_by_month[m_str]
        if "all" in active_by_month:
            return sym in active_by_month["all"]
        return False

    return [s for s in sigs if is_active(s.sym, s.next_open_ts)]


# ─────────────────────────────────────────────────────────────────────────────
# Walk-forward
# ─────────────────────────────────────────────────────────────────────────────

def _walk_forward(sigs, sim_dfs, regime, events, corr_mx, label, mode="strict",
                  train_months=6, test_months=3):
    all_ts = sorted(s.next_open_ts for s in sigs)
    if not all_ts:
        return []
    first = all_ts[0]; last = all_ts[-1]
    train_ms = train_months * 30 * DAY_MS
    test_ms  = test_months  * 30 * DAY_MS
    cursor   = first + train_ms
    folds = []

    while cursor + test_ms <= last:
        te = cursor + test_ms
        tsigs = [s for s in sigs if cursor <= s.next_open_ts < te]
        if tsigs:
            res = simulate_strict(tsigs, sim_dfs, regime, events,
                                  corr_matrix=corr_mx, mode=mode)
            m = res["metrics"]
            folds.append({
                "start": pd.Timestamp(cursor, unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                "end":   pd.Timestamp(te,     unit="ms", tz="UTC").strftime("%Y-%m-%d"),
                **m,
            })
        cursor += test_ms

    pos_folds = sum(1 for f in folds if f.get("return_pct", 0) > 0)
    print(f"\n  Walk-forward ({label})  train={train_months}mo  test={test_months}mo  "
          f"→  {pos_folds}/{len(folds)} позитивных:")
    print(f"  {'':2s} {'Start':10s}  {'End':10s}  {'n':>4s}  {'PF':>5s}  "
          f"{'WR%':>5s}  {'Ret%':>8s}  {'DD%':>5s}")
    for f in folds:
        tick = "✓" if f.get("return_pct", 0) > 0 else "✗"
        print(f"  {tick} {f['start']:10s}  {f['end']:10s}  {f['n']:>4d}  "
              f"{f['pf']:>5.2f}  {f['wr']:>5.1f}  "
              f"{f['return_pct']:>+7.1f}%  {f['max_dd_pct']:>5.1f}%")
    return folds


# ─────────────────────────────────────────────────────────────────────────────
# Run one selector
# ─────────────────────────────────────────────────────────────────────────────

def run_selector(name: str,
                 active_by_month: dict,
                 regime, events,
                 sim_dfs_all: dict,
                 sigs_all: list,
                 mode: str = "strict",
                 run_wf: bool = True) -> dict:

    print(f"\n{'═'*72}")
    print(f"  SELECTOR: {name}")

    if "all" in active_by_month:
        print(f"  Пул: СТАТИЧНЫЙ — {len(active_by_month['all'])} монет")
    else:
        sizes = [len(v) for v in active_by_month.values()]
        if sizes:
            print(f"  Пул: ДИНАМИЧЕСКИЙ — avg={np.mean(sizes):.1f}/mo  "
                  f"min={min(sizes)}  max={max(sizes)}  months={len(sizes)}")

    # Month filter
    sigs_m = _filter_by_month(sigs_all, active_by_month)
    print(f"  Сигналов после month-filter: {len(sigs_m)} (из {len(sigs_all)})")

    if not sigs_m:
        print("  Нет сигналов — пропуск")
        return {"name": name, "metrics": {"n": 0, "return_pct": -999, "max_dd_pct": 0,
                                           "wr": 0, "pf": 0, "avg_r": 0}, "monthly": []}

    # Pivot distance filter
    sigs_f, removed, _ = apply_pivot_dist_filter(sigs_m, sim_dfs_all, PIVOT_DIST)
    print(f"  После pivot_dist<{PIVOT_DIST}×ATR: {len(sigs_f)} ({removed} убрано)")

    if not sigs_f:
        return {"name": name, "metrics": {"n": 0, "return_pct": -999, "max_dd_pct": 0,
                                           "wr": 0, "pf": 0, "avg_r": 0}, "monthly": []}

    # Corr matrix — только по активным символам
    active_syms = {s.sym for s in sigs_f}
    sub_dfs = {k: v for k, v in sim_dfs_all.items() if k[0] in active_syms}
    corr_mx = build_corr_matrix(sub_dfs)

    # Симуляция
    res = simulate_strict(sigs_f, sim_dfs_all, regime, events,
                          corr_matrix=corr_mx, mode=mode)
    m   = res["metrics"]

    # Per-strategy breakdown
    per = defaultdict(list)
    for t in res["trades"]:
        per[t["strat"]].append(t)

    # Monthly returns
    rets   = _monthly_rets_from_trades(res["trades"])
    avg_mo = float(np.mean(rets)) if rets else 0
    pos_mo = sum(1 for r in rets if r > 0)

    # Exit reasons
    rea = res.get("reasons", {})
    n_tot = sum(rea.values())
    rea_str = "  ".join(f"{k}={v}({v/max(1,n_tot)*100:.0f}%)" for k, v in sorted(rea.items()))

    print(f"\n  Trades={m['n']}  WR={m['wr']:.1f}%  PF={m['pf']:.2f}  AvgR={m['avg_r']:+.3f}")
    print(f"  Return={m['return_pct']:+.1f}%  DD={m['max_dd_pct']:.1f}%  "
          f"Final=${m['final_balance']:,.0f}")
    if rets:
        print(f"  Avg/mo={avg_mo:+.1f}%  Прибыльных мес: {pos_mo}/{len(rets)}  "
              f"Best={max(rets):+.1f}%  Worst={min(rets):+.1f}%")
    print(f"  Exits: {rea_str}")

    print(f"  По стратегиям:")
    for sn, tlist in sorted(per.items()):
        wins = sum(1 for t in tlist if t["pnl"] > 0)
        gw   = sum(t["pnl"] for t in tlist if t["pnl"] > 0)
        gl   = abs(sum(t["pnl"] for t in tlist if t["pnl"] <= 0))
        pf_s = gw / gl if gl > 0 else 99.0
        wr_s = wins / len(tlist) * 100
        # Средняя SL потеря у стратегии
        sl_trades = [t for t in tlist if t["reason"] == "SL"]
        avg_sl_r  = float(np.mean([t["pnl_r"] for t in sl_trades])) if sl_trades else 0
        print(f"    {sn:20s}: n={len(tlist):4d}  WR={wr_s:5.1f}%  PF={pf_s:5.2f}"
              f"  SL_avgR={avg_sl_r:+.2f}  SL_n={len(sl_trades)}")

    # Walk-forward
    wf_folds = None
    if run_wf:
        wf_folds = _walk_forward(sigs_f, sim_dfs_all, regime, events, corr_mx, name, mode=mode)

    return {
        "name":     name,
        "mode":     mode,
        "metrics":  m,
        "monthly":  rets,
        "wf_folds": wf_folds,
        "per_strat": {
            sn: {
                "n":  len(tl),
                "wr": sum(1 for t in tl if t["pnl"] > 0) / max(1, len(tl)) * 100,
                "pf": (sum(t["pnl"] for t in tl if t["pnl"] > 0) /
                       max(1e-9, abs(sum(t["pnl"] for t in tl if t["pnl"] <= 0)))),
            }
            for sn, tl in per.items()
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def _print_summary(results: list, label: str, baseline_name: str = "BASELINE_27"):
    print(f"\n\n{'═'*78}")
    print(f"  ИТОГ: {label}  ({TAG}  risk={RISK*100:.0f}%  max_pos={MAX_POS}  L4=ON  corr=1@0.80)")
    print(f"{'═'*78}")
    print(f"  {'Selector':28s} {'n':>4s}  {'WR%':>6s}  {'PF':>5s}  "
          f"{'Return':>9s}  {'DD%':>6s}  {'Avg/mo':>8s}  {'Med/mo':>7s}  {'WF':>5s}")
    print(f"  {'─'*82}")
    results_sorted = sorted(results, key=lambda r: -r["metrics"].get("return_pct", -9999))
    baseline_m = next((r["metrics"] for r in results if r["name"] == baseline_name), None)
    for r in results_sorted:
        m    = r["metrics"]
        rets = r.get("monthly", [])
        folds = r.get("wf_folds") or []
        avg_mo = float(np.mean(rets)) if rets else 0
        med_mo = float(np.median(rets)) if rets else 0
        pos_folds = sum(1 for f in folds if f.get("return_pct", 0) > 0) if folds else 0
        wf = f"{pos_folds}/{len(folds)}" if folds else "—"
        marker = " ← BASE" if r["name"] == baseline_name else ""
        print(f"  {r['name']:28s} {m['n']:>4d}  {m['wr']:>5.1f}%  "
              f"{m['pf']:>5.2f}  "
              f"{m['return_pct']:>+8.1f}%  {m['max_dd_pct']:>5.1f}%  "
              f"{avg_mo:>+7.1f}%  {med_mo:>+6.1f}%  {wf:>5s}{marker}")
    if baseline_m:
        print(f"\n  Δ vs {baseline_name}:")
        for r in results_sorted:
            if r["name"] == baseline_name:
                continue
            m = r["metrics"]
            d_ret = m["return_pct"] - baseline_m["return_pct"]
            d_dd  = m["max_dd_pct"] - baseline_m["max_dd_pct"]
            d_pf  = m["pf"]         - baseline_m["pf"]
            sign  = "▲" if d_ret > 0 else "▼"
            print(f"    {r['name']:28s}  {sign} ΔRet={d_ret:+.1f}%  "
                  f"ΔDD={d_dd:+.1f}%  ΔPF={d_pf:+.2f}")


def main():
    print("\n" + "═"*78)
    print("  UNIVERSE V7 — FAIR vs STRICT comparison")
    print(f"  FAIR:   slippage={FAIR_SLIPPAGE*100:.2f}%   SL fill = sl0 price (= simulate_v7 baseline)")
    print(f"  STRICT: slippage={STRICT_SLIPPAGE*100:.2f}%   SL fill = wick lo/hi (worst-case)")
    print("═"*78)

    # Загрузка
    btc    = load("BTC", "1d", TAG)
    regime = build_regime(btc)
    events = load_all_events()

    # Все символы с 730d данными
    all_syms = list_available_symbols(TAG, required_tfs=("1d", "4h"))
    baseline_avail = [s for s in BASELINE if s in all_syms]
    extra_syms     = [s for s in all_syms if s not in BASELINE]

    print(f"\n  Доступных символов: {len(all_syms)}")
    print(f"  BASELINE_27 в наличии: {len(baseline_avail)}/27")

    # Monthly metrics
    print(f"\n  Вычисляю monthly metrics...", flush=True)
    metrics_by_sym = {}
    for sym in all_syms:
        ml = compute_monthly_metrics(sym, TAG)
        if ml:
            metrics_by_sym[sym] = ml
    print(f"  Computed: {len(metrics_by_sym)} символов")

    # ── QUALITY_STATIC ────────────────────────────────────────────────────────
    sel_quality = selector_quality_static(metrics_by_sym, all_syms)
    quality_syms = sel_quality["all"]
    print(f"\n  ── QUALITY_STATIC ({len(quality_syms)} монет) ──")
    print(f"    {'  '.join(quality_syms)}")
    excluded = [s for s in all_syms if s not in quality_syms]
    print(f"  Исключено ({len(excluded)}):")
    print(f"    {'  '.join(sorted(excluded))}")

    # Selectors
    sel_baseline = {"all": baseline_avail}
    sel_top30    = selector_top_volume(metrics_by_sym, top_n=30)

    # Один раз собираем сигналы для ВСЕХ символов
    print(f"\n  Собираю V7 сигналы ({len(all_syms)} символов)...", flush=True)
    sigs_all, sim_dfs_all = collect_portfolio(
        all_syms, regime, TAG, bear_mode="long_only",
        supertrend_kw=ST_KW, smacross_kw=SC_KW,
        rsi_rev_kw=RSI_KW, fib_pump_kw=FIB_KW,
    )
    print(f"  Итого сигналов: {len(sigs_all)}")

    # ── FAIR режим (для сравнения с +1525% baseline) ──────────────────────────
    print(f"\n\n{'#'*78}\n#  FAIR MODE (slip {FAIR_SLIPPAGE*100:.2f}%, SL@sl0)\n{'#'*78}")
    fair_results = [
        run_selector("BASELINE_27",       sel_baseline, regime, events, sim_dfs_all, sigs_all, mode="fair"),
        run_selector("QUALITY_STATIC",    sel_quality,  regime, events, sim_dfs_all, sigs_all, mode="fair"),
        run_selector("TOP_30_VOL_ROLLING", sel_top30,   regime, events, sim_dfs_all, sigs_all, mode="fair"),
    ]

    # ── STRICT режим (пессимистичный) ─────────────────────────────────────────
    print(f"\n\n{'#'*78}\n#  STRICT MODE (slip {STRICT_SLIPPAGE*100:.2f}%, SL@wick)\n{'#'*78}")
    strict_results = [
        run_selector("BASELINE_27",       sel_baseline, regime, events, sim_dfs_all, sigs_all, mode="strict"),
        run_selector("QUALITY_STATIC",    sel_quality,  regime, events, sim_dfs_all, sigs_all, mode="strict"),
        run_selector("TOP_30_VOL_ROLLING", sel_top30,   regime, events, sim_dfs_all, sigs_all, mode="strict"),
    ]

    _print_summary(fair_results,   "FAIR   (matches +1525% baseline assumptions)")
    _print_summary(strict_results, "STRICT (worst-case wick fills)")

    # Сохранение
    out_path = Path("multi_agent/backtests/results/universe_v7_test.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps({
            "fair":   [{"name": r["name"], "mode": r["mode"], "metrics": r["metrics"],
                        "monthly": r.get("monthly", []), "per_strat": r.get("per_strat", {}),
                        "wf_positive": sum(1 for f in (r.get("wf_folds") or [])
                                           if f.get("return_pct", 0) > 0),
                        "wf_total":    len(r.get("wf_folds") or [])}
                       for r in fair_results],
            "strict": [{"name": r["name"], "mode": r["mode"], "metrics": r["metrics"],
                        "monthly": r.get("monthly", []), "per_strat": r.get("per_strat", {}),
                        "wf_positive": sum(1 for f in (r.get("wf_folds") or [])
                                           if f.get("return_pct", 0) > 0),
                        "wf_total":    len(r.get("wf_folds") or [])}
                       for r in strict_results],
            "quality_universe": quality_syms,
        }, indent=2, default=str),
        encoding="utf-8",
    )
    print(f"\n  Сохранено: {out_path}")


if __name__ == "__main__":
    main()
