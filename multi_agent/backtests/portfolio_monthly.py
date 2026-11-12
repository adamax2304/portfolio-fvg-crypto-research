"""
portfolio_monthly.py -- Месячная разбивка портфеля v10 (365d)

Аллокации из итогового отчёта v10:
  BBS 1h->5m   pullback  35%  SL=ATR1.5  TP=3.0R  daily_ema200=ON
  BBS 1h->15m  next_bar  30%  SL=ATR1.5  TP=2.5R
  KC  1h->15m  next_bar  20%  SL=ATR1.5  TP=2.0R
  Swing KC 4h->15m       15%  SL=ATR2.0  TP=2.5R

Метод: каждая стратегия симулируется независимо на своей доле capital,
затем ежемесячные PnL складываются.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ─── переиспользуем все вспомогательные функции из backtest_v10 ─────────────
from multi_agent.backtests.backtest_v10 import (
    load_all_tf, prepare_tf_arrays, prepare_daily_trend, get_daily_trend,
    gen_bbs_signals_tf, gen_kc_squeeze_signals_tf, gen_mtf_signals,
    compute_sl, _aggregate, _empty_result,
    SYMBOLS, START_BALANCE, TAKER_FEE, SLIPPAGE, MAX_LEVERAGE,
)

RESULTS_DIR = Path(__file__).parent / "results"

# ─── Конфиги портфеля ────────────────────────────────────────────────────────
PORTFOLIO = [
    {
        "name":   "BBS 1h->5m",
        "alloc":  0.35,
        "htf_key": "df1h", "ltf_key": "df5",
        "gen_fn": "bbs",
        "sig_params":   {"min_squeeze": 8, "adx_thresh": 15, "vol_mult": 1.0,
                         "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "pullback_ema21", "max_wait_bars": 20, "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 3.0},
        "daily_filter": True,
        "max_pos": 4,
    },
    {
        "name":   "BBS 1h->15m",
        "alloc":  0.30,
        "htf_key": "df1h", "ltf_key": "df15",
        "gen_fn": "bbs",
        "sig_params":   {"min_squeeze": 4, "adx_thresh": 15, "vol_mult": 1.0,
                         "htf_filter": False, "headfake_filter": True},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
        "max_pos": 4,
    },
    {
        "name":   "KC 1h->15m",
        "alloc":  0.20,
        "htf_key": "df1h", "ltf_key": "df15",
        "gen_fn": "kc",
        "sig_params":   {"min_squeeze": 10, "adx_min": 12, "vol_mult": 1.0,
                         "mom_bars": 6, "sl_atr_mult": 1.5},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 1.5},
        "tp_cfg": {"type": "fixed_r", "r": 2.0},
        "daily_filter": False,
        "max_pos": 4,
    },
    {
        "name":   "Swing KC 4h->15m",
        "alloc":  0.15,
        "htf_key": "df4h", "ltf_key": "df15",
        "gen_fn": "kc",
        "sig_params":   {"min_squeeze": 8, "adx_min": 15, "vol_mult": 1.0,
                         "mom_bars": 6, "sl_atr_mult": 2.0},
        "entry_params": {"entry_mode": "next_bar", "sl_atr_mult": 2.5},
        "sl_cfg": {"type": "atr", "mult": 2.0},
        "tp_cfg": {"type": "fixed_r", "r": 2.5},
        "daily_filter": False,
        "max_pos": 2,
    },
]


# ─── Симуляция с трек-рекордом по сделкам ────────────────────────────────────

def run_sim_with_trades(signals_by_sym, tf_arrays_by_sym, sl_cfg, tp_cfg,
                        risk_pt=0.010, cap_frac=0.35, max_pos=4,
                        capital_share=1.0):
    """
    Возвращает список закрытых trades с timestamp выхода.
    capital_share: доля общего capital ($1000) для этой стратегии.
    """
    balance = START_BALANCE * capital_share
    peak    = balance
    open_pos = {}
    acnt = 0
    trades = []

    max_n = max((a["n"] for a in tf_arrays_by_sym.values() if a), default=0)
    if max_n == 0:
        return trades

    for i in range(max(300, 0), max_n):
        for sym, arr in tf_arrays_by_sym.items():
            if arr is None or i >= arr["n"]:
                continue
            cl_i = arr["cl"][i]; hi_i = arr["hi"][i]; lo_i = arr["lo"][i]
            atr_i = arr["atr"][i]
            if np.isnan(atr_i) or atr_i <= 0:
                atr_i = cl_i * 0.01

            if sym in open_pos:
                pos = open_pos[sym]; d = pos["d"]; entry = pos["entry"]
                sl = pos["sl"]; r = pos["risk"]; q = pos["qty"]

                hit_sl = (d == "LONG" and lo_i <= sl) or (d == "SHORT" and hi_i >= sl)
                hit_tp = False; tp_price = 0.0; exit_rsn = "sl"

                if not hit_sl:
                    tp_type = tp_cfg["type"]
                    pnl_r = ((cl_i - entry) / r) if d == "LONG" else ((entry - cl_i) / r)
                    if tp_type == "fixed_r" and pnl_r >= tp_cfg["r"]:
                        hit_tp = True; tp_price = cl_i; exit_rsn = "tp"

                if hit_sl or hit_tp:
                    ep  = tp_price if hit_tp else sl
                    gross = ((ep - entry) if d == "LONG" else (entry - ep)) * q
                    fee   = ep * q * (TAKER_FEE + SLIPPAGE)
                    net   = gross - fee
                    balance += net
                    peak = max(peak, balance)
                    # timestamp milliseconds → секунды
                    exit_ts = int(arr["ts"][i]) // 1000
                    trades.append({
                        "ts":     exit_ts,
                        "sym":    sym,
                        "d":      d,
                        "pnl":    round(net, 4),
                        "reason": exit_rsn,
                        "won":    net > 0,
                    })
                    acnt = max(0, acnt - 1)
                    del open_pos[sym]
                continue

            if acnt >= max_pos:
                continue
            sym_sigs = signals_by_sym.get(sym, {})
            if i not in sym_sigs:
                continue
            sig = sym_sigs[i]
            new_sl = compute_sl(sl_cfg, sig["direction"], sig["entry_price"],
                                atr_i, arr["hi"], arr["lo"], i, sig["default_sl"])
            rd = abs(sig["entry_price"] - new_sl)
            if rd <= 0 or rd / sig["entry_price"] > 0.05 or rd / sig["entry_price"] < 0.001:
                continue
            ab = balance * cap_frac
            ru = ab * risk_pt
            qty = ru / rd
            mg  = sig["entry_price"] * qty / MAX_LEVERAGE
            if mg > ab * 0.4:
                qty = ab * 0.4 * MAX_LEVERAGE / sig["entry_price"]
            if qty * sig["entry_price"] < 5.0:
                continue
            ef = sig["entry_price"] * qty * (TAKER_FEE + SLIPPAGE)
            balance -= ef
            open_pos[sym] = {
                "d": sig["direction"], "entry": sig["entry_price"],
                "sl": new_sl, "risk": rd, "qty": qty, "bar_i": i,
            }
            acnt += 1

    return trades


def build_signals(cfg_entry, all_data):
    gen_fn = gen_bbs_signals_tf if cfg_entry["gen_fn"] == "bbs" else gen_kc_squeeze_signals_tf
    htf_key = cfg_entry["htf_key"]
    ltf_key = cfg_entry["ltf_key"]

    all_sigs  = {}
    ltf_arrays = {}

    for sym, d in all_data.items():
        htf_arr = prepare_tf_arrays(d.get(htf_key))
        ltf_arr = prepare_tf_arrays(d.get(ltf_key))
        if htf_arr is None or ltf_arr is None:
            continue
        ltf_arrays[sym] = ltf_arr

        # HTF trend filter для gen_fn
        htf_trend = prepare_tf_arrays(d.get("df4h")) if htf_key == "df1h" else prepare_tf_arrays(d.get("df1h"))
        htf_sigs = gen_fn(htf_arr, htf_trend, cfg_entry["sig_params"])

        daily_d = prepare_daily_trend(d.get("df1d")) if cfg_entry["daily_filter"] else None
        mtf_sigs = gen_mtf_signals(htf_sigs, htf_arr, ltf_arr, cfg_entry["entry_params"],
                                   daily_data=daily_d, use_daily_filter=cfg_entry["daily_filter"])
        all_sigs[sym] = mtf_sigs

    return all_sigs, ltf_arrays


def month_key(ts_sec):
    dt = datetime.fromtimestamp(ts_sec, tz=timezone.utc)
    return dt.strftime("%Y-%m")


def main():
    print("Portfolio Monthly Breakdown -- v10 configs (365d)", flush=True)
    print("Loading data...", flush=True)
    all_data = load_all_tf("365d")
    has_5m = any(d.get("df5") is not None and len(d["df5"]) > 1000 for d in all_data.values())
    print(f"Symbols: {len(all_data)},  5m data: {has_5m}", flush=True)

    # Собираем все сделки по всем стратегиям
    all_trades = []   # список: {ts, strat, pnl_abs, pnl_pct_of_total}

    for strat in PORTFOLIO:
        name  = strat["name"]
        alloc = strat["alloc"]

        if "5m" in strat["ltf_key"] and not has_5m:
            print(f"  SKIP {name} (no 5m data)", flush=True)
            continue

        print(f"  Building signals for {name}...", flush=True)
        sigs, arrs = build_signals(strat, all_data)
        total_sigs = sum(len(s) for s in sigs.values())

        trades = run_sim_with_trades(
            sigs, arrs,
            sl_cfg=strat["sl_cfg"],
            tp_cfg=strat["tp_cfg"],
            risk_pt=0.010,
            cap_frac=0.60,       # same as v10 portfolio sim
            max_pos=strat["max_pos"],
            capital_share=alloc,  # strategy starts with its allocated fraction of $1000
        )

        print(f"    {name}: {total_sigs} sigs, {len(trades)} trades", flush=True)

        for t in trades:
            all_trades.append({
                "ts":    t["ts"],
                "strat": name,
                "alloc": alloc,
                "pnl":   t["pnl"],   # абсолютный PnL в $ (от доли capital)
                "won":   t["won"],
            })

    if not all_trades:
        print("Нет trades — data not loaded?")
        return

    all_trades.sort(key=lambda x: x["ts"])

    # ── Месячная агрегация ───────────────────────────────────────────────────
    monthly: dict[str, dict] = defaultdict(lambda: {
        "pnl": 0.0, "trades": 0, "wins": 0,
        "by_strat": defaultdict(float),
    })

    equity = START_BALANCE
    monthly_equity: dict[str, float] = {}

    # Стартовый equity по месяцам
    first_month = month_key(all_trades[0]["ts"])
    monthly_equity[first_month] = START_BALANCE

    for t in all_trades:
        mk = month_key(t["ts"])
        monthly[mk]["pnl"]    += t["pnl"]
        monthly[mk]["trades"] += 1
        if t["won"]:
            monthly[mk]["wins"] += 1
        monthly[mk]["by_strat"][t["strat"]] += t["pnl"]
        equity += t["pnl"]
        monthly_equity[mk] = equity

    # Вычисляем пиковый equity до каждого месяца (для DD)
    all_months = sorted(monthly.keys())
    peak = START_BALANCE
    month_start_eq: dict[str, float] = {}
    cur_eq = START_BALANCE

    # Восстанавливаем equity-кривую помесячно
    month_start_eq[all_months[0]] = START_BALANCE
    running_eq = START_BALANCE
    for mk in all_months:
        m_pnl = monthly[mk]["pnl"]
        month_end = running_eq + m_pnl
        monthly[mk]["eq_start"] = running_eq
        monthly[mk]["eq_end"]   = month_end
        running_eq = month_end

    # Drawdown от пика
    peak = START_BALANCE
    for mk in all_months:
        peak = max(peak, monthly[mk]["eq_end"])
        monthly[mk]["peak"] = peak

    # ── Вывод таблицы ────────────────────────────────────────────────────────
    STRAT_LABELS = {
        "BBS 1h->5m":        "BBS_5m",
        "BBS 1h->15m":       "BBS_15m",
        "KC 1h->15m":        "KC_15m",
        "Swing KC 4h->15m":  "SwKC",
    }

    header = f"{'Month':<8}  {'PnL$':>7}  {'PnL%':>6}  {'WR':>5}  {'N':>3}  {'Equity':>8}  {'DD%':>5}  {'BBS_5m':>7}  {'BBS_15m':>8}  {'KC_15m':>7}  {'SwKC':>6}"
    print()
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    total_pnl = 0.0
    total_wins = 0
    total_n = 0

    for mk in all_months:
        m = monthly[mk]
        pnl_pct = m["pnl"] / m["eq_start"] * 100 if m["eq_start"] > 0 else 0
        wr = m["wins"] / m["trades"] * 100 if m["trades"] > 0 else 0
        eq_end = m["eq_end"]
        dd_from_peak = (m["peak"] - eq_end) / m["peak"] * 100 if m["peak"] > 0 else 0

        by_s = m["by_strat"]
        bbs5  = by_s.get("BBS 1h->5m", 0)
        bbs15 = by_s.get("BBS 1h->15m", 0)
        kc15  = by_s.get("KC 1h->15m", 0)
        swkc  = by_s.get("Swing KC 4h->15m", 0)

        # Помечаем убыточные месяцы
        sign = "v" if m["pnl"] < 0 else " "

        print(f"{sign}{mk:<7}  {m['pnl']:>+7.1f}  {pnl_pct:>+5.1f}%  {wr:>4.0f}%  {m['trades']:>3}  ${eq_end:>7.0f}  {dd_from_peak:>4.1f}%  {bbs5:>+7.1f}  {bbs15:>+8.1f}  {kc15:>+7.1f}  {swkc:>+6.1f}")

        total_pnl  += m["pnl"]
        total_wins += m["wins"]
        total_n    += m["trades"]

    print("=" * len(header))
    final_eq = START_BALANCE + total_pnl
    total_pnl_pct = total_pnl / START_BALANCE * 100
    total_wr = total_wins / total_n * 100 if total_n > 0 else 0
    print(f"{'TOTAL':<8}  {total_pnl:>+7.1f}  {total_pnl_pct:>+5.1f}%  {total_wr:>4.0f}%  {total_n:>3}  ${final_eq:>7.0f}")
    print()
    print(f"$1,000 -> ${final_eq:.0f}  (+{total_pnl_pct:.0f}%)")
    print()

    # ── Лучший / худший месяц ────────────────────────────────────────────────
    sorted_by_pnl = sorted(all_months, key=lambda mk: monthly[mk]["pnl"])
    worst3 = sorted_by_pnl[:3]
    best3  = sorted_by_pnl[-3:][::-1]

    print("Top-3 WORST months:")
    for mk in worst3:
        m = monthly[mk]
        pct = m["pnl"] / m["eq_start"] * 100
        print(f"  {mk}  {m['pnl']:>+7.1f}$  ({pct:>+5.1f}%)  {m['trades']} trades")

    print()
    print("Top-3 BEST months:")
    for mk in best3:
        m = monthly[mk]
        pct = m["pnl"] / m["eq_start"] * 100
        print(f"  {mk}  {m['pnl']:>+7.1f}$  ({pct:>+5.1f}%)  {m['trades']} trades")

    # ── Убыточные месяцы подряд ──────────────────────────────────────────────
    loss_months = [mk for mk in all_months if monthly[mk]["pnl"] < 0]
    print(f"\nLoss months: {len(loss_months)} of {len(all_months)}")
    print(f"Loss: {', '.join(loss_months) if loss_months else 'none'}")

    # ── Аллокация ────────────────────────────────────────────────────────────
    print()
    print("Portfolio allocation:")
    for s in PORTFOLIO:
        contrib = sum(t["pnl"] for t in all_trades if t["strat"] == s["name"])
        pct_of_total = contrib / total_pnl * 100 if total_pnl else 0
        print(f"  {s['name']:<22} {int(s['alloc']*100):>3}% capital   "
              f"contrib: {contrib:>+7.1f}$ ({pct_of_total:>+5.1f}%)")


if __name__ == "__main__":
    main()
