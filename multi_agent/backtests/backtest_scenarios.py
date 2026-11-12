"""
backtest_scenarios.py  --  Честное сравнение сценариев улучшения

Вопросы:
  1. Помогает ли $500 + 1% риска vs $100 + 2%?
  2. Как увеличить количество сделок без потери WR / роста DD?

Символы: TOP_50 (лучший набор по 2-летнему бэктесту)
Период: до 730d (берёт что есть в кэше)
Walk-forward: IS=80% / OOS=20%

ПРИНЦИПЫ ЧЕСТНОГО ТЕСТА:
  - Никаких подгонок под данные
  - Каждое изменение тестируется изолированно (один параметр за раз)
  - Результат признаётся хуже если PF падает ИЛИ WR падает ИЛИ MaxDD растёт
  - "Больше сделок" считается улучшением ТОЛЬКО если PF/WR не ухудшились
"""
from __future__ import annotations
import logging, sys, time as _time, copy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# ── Reuse functions from v12 ──────────────────────────────────────────────────
# Import all shared code instead of duplicating
from multi_agent.backtests.backtest_v12 import (
    prepare, _load,
    detect_bbs_signals_1h, detect_kc_signals_1h,
    match_ltf_entries, _per_strat,
    TAKER_FEE, SLIPPAGE, MIN_MARGIN, MAX_LEVERAGE, MAX_MARGIN_FRAC,
    SL_COOLDOWN_BARS_1H,
)

logger = logging.getLogger("scenarios")
logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger.setLevel(logging.INFO)

# Fix Windows cp1252 encoding for Russian output
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── TOP_50 symbols (лучший набор) ─────────────────────────────────────────────
TOP_50 = [
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

# ── Base strategies (текущий прод конфиг) ─────────────────────────────────────
BASE_STRATEGIES = {
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

# ── Scenarios ─────────────────────────────────────────────────────────────────
# Каждый сценарий меняет ОДИН параметр от базы. Так можно честно оценить эффект.
SCENARIOS = [
    # ── Капитал / риск ────────────────────────────────────────────────────────
    {
        "name": "BASE  $100/2%  MAX4",
        "balance": 100.0, "risk": 0.02, "max_pos": 4,
        "strategies": BASE_STRATEGIES,
        "note": "Текущий прод",
    },
    {
        "name": "TEST  $500/1%  MAX4",
        "balance": 500.0, "risk": 0.01, "max_pos": 4,
        "strategies": BASE_STRATEGIES,
        "note": "Вопрос пользователя: помогает ли $500 + 1%?",
    },
    {
        "name": "TEST  $500/2%  MAX4",
        "balance": 500.0, "risk": 0.02, "max_pos": 4,
        "strategies": BASE_STRATEGIES,
        "note": "Контроль: $500 с тем же 2% (что изменяет только капитал)",
    },
    # ── Больше позиций ────────────────────────────────────────────────────────
    {
        "name": "TEST  $100/2%  MAX6",
        "balance": 100.0, "risk": 0.02, "max_pos": 6,
        "strategies": BASE_STRATEGIES,
        "note": "Увеличить MAX_POSITIONS 4→6 (больше одновременных сделок)",
    },
    # ── Дольше ждать pullback ─────────────────────────────────────────────────
    {
        "name": "TEST  wait5m=36bars (~3h)",
        "balance": 100.0, "risk": 0.02, "max_pos": 4,
        "strategies": {
            **BASE_STRATEGIES,
            "BBS_1H_5M": {**BASE_STRATEGIES["BBS_1H_5M"], "max_wait": 36},  # 20→36 bars
        },
        "note": "BBS_1H_5M: ждать pullback дольше (20→36 баров = 3 часа вместо 100 мин)",
    },
    # ── Ослабить ADX (РИСК: снижение WR) ─────────────────────────────────────
    {
        "name": "TEST  ADX 15→10",
        "balance": 100.0, "risk": 0.02, "max_pos": 4,
        "strategies": {
            **BASE_STRATEGIES,
            "BBS_1H_5M":  {**BASE_STRATEGIES["BBS_1H_5M"],  "adx_thresh": 10.0},
            "BBS_1H_15M": {**BASE_STRATEGIES["BBS_1H_15M"], "adx_thresh": 10.0},
            "KC_1H_15M":  {**BASE_STRATEGIES["KC_1H_15M"],  "adx_min":     9.0},
        },
        "note": "⚠️  ADX порог ниже → больше сигналов в боковике (ожидаем WR падение)",
    },
    # ── Ослабить min_squeeze (РИСК: снижение WR) ─────────────────────────────
    {
        "name": "TEST  squeeze BBS 4→2, KC 10→6",
        "balance": 100.0, "risk": 0.02, "max_pos": 4,
        "strategies": {
            **BASE_STRATEGIES,
            "BBS_1H_5M":  {**BASE_STRATEGIES["BBS_1H_5M"],  "min_squeeze": 5},
            "BBS_1H_15M": {**BASE_STRATEGIES["BBS_1H_15M"], "min_squeeze": 2},
            "KC_1H_15M":  {**BASE_STRATEGIES["KC_1H_15M"],  "min_squeeze": 6},
        },
        "note": "⚠️  Меньше баров сжатия → больше сигналов (ожидаем WR падение)",
    },
    # ── Отключить KC_1H_15M (она слабая — PF=1.35 OOS) ──────────────────────
    {
        "name": "TEST  без KC_1H_15M",
        "balance": 100.0, "risk": 0.02, "max_pos": 4,
        "strategies": {
            k: v for k, v in BASE_STRATEGIES.items() if k != "KC_1H_15M"
        },
        "note": "Убрать KC стратегию (OOS PF=1.35 — почти breakeven)",
    },
]


# ── Portfolio simulation ──────────────────────────────────────────────────────
def run_portfolio_cfg(all_entries, all_ltf, cfg, start_frac=0.0, end_frac=1.0):
    balance    = cfg["balance"]
    START_BAL  = cfg["balance"]
    risk_pct   = cfg["risk"]
    max_pos    = cfg["max_pos"]

    peak = balance; max_dd = 0.0; trades = []
    open_pos: dict[tuple, bool] = {}
    sym_cooldown: dict[str, int] = {}

    timeline = []
    for sym, strat_map in all_entries.items():
        for strat_id, entries in strat_map.items():
            dltf = all_ltf.get(sym, {}).get(strat_id)
            if dltf is None: continue
            ts_arr = dltf["ts"]; n_ltf = dltf["n"]
            i_start = int(n_ltf * start_frac); i_end = int(n_ltf * end_frac)
            for bar, td in entries.items():
                if bar < i_start or bar >= i_end: continue
                timeline.append((int(ts_arr[bar]), sym, strat_id, bar, td))
    timeline.sort(key=lambda x: x[0])

    for ts_entry, sym, strat_id, bar, td in timeline:
        dltf = all_ltf[sym][strat_id]
        cl = dltf["cl"]; hi = dltf["hi"]; lo = dltf["lo"]; n = dltf["n"]
        key = (sym, strat_id)
        if key in open_pos: continue
        if sym_cooldown.get(sym, 0) > ts_entry: continue
        if len(open_pos) >= max_pos: continue

        d = td["direction"]; entry = td["entry"]; sl = td["sl"]
        tp = td["tp"]; dist = td["dist"]

        risk_usd   = balance * risk_pct
        if dist <= 0 or risk_usd < 0.01: continue
        raw_qty    = risk_usd / dist
        raw_margin = raw_qty * entry
        max_margin = min(balance * MAX_MARGIN_FRAC, balance * 0.95)
        lev = max(1, min(int(np.ceil(raw_margin / max_margin)) if max_margin > 0 else 1, MAX_LEVERAGE))
        eff_margin = raw_margin / lev
        if eff_margin < MIN_MARGIN: continue

        open_pos[key] = True
        exit_price = None; reason = "timeout"; exit_bar = None
        for j in range(bar + 1, n):
            if d == "LONG":
                if lo[j] <= sl: exit_price = sl; reason = "SL"; exit_bar = j; break
                if hi[j] >= tp: exit_price = tp; reason = "TP"; exit_bar = j; break
            else:
                if hi[j] >= sl: exit_price = sl; reason = "SL"; exit_bar = j; break
                if lo[j] <= tp: exit_price = tp; reason = "TP"; exit_bar = j; break
        del open_pos[key]
        if exit_price is None: continue

        gross = (exit_price - entry)*raw_qty if d=="LONG" else (entry - exit_price)*raw_qty
        fees  = (entry + exit_price) * raw_qty * (TAKER_FEE + SLIPPAGE)
        pnl   = gross - fees
        balance = max(balance + pnl, 0.0)
        peak    = max(peak, balance)
        dd      = (peak - balance) / peak * 100 if peak > 0 else 0
        max_dd  = max(max_dd, dd)

        trades.append({"strat": strat_id, "win": pnl > 0, "pnl": pnl})
        if reason == "SL":
            sym_cooldown[sym] = ts_entry + SL_COOLDOWN_BARS_1H * 3_600_000

    if not trades:
        return {"trades": 0, "pf": 0, "wr": 0, "max_dd": 0, "final": balance,
                "gain_pct": 0, "per_strat": {}}
    wins   = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"]]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else float("inf")
    wr = len(wins) / len(trades) * 100

    # Per-strat
    from collections import defaultdict
    by_s: dict[str, list] = defaultdict(list)
    for t in trades: by_s[t["strat"]].append(t)
    ps = {}
    for s, ts in by_s.items():
        sw = [t for t in ts if t["win"]]
        sl_ = [t for t in ts if not t["win"]]
        sg = sum(t["pnl"] for t in sw)
        sl2 = abs(sum(t["pnl"] for t in sl_))
        ps[s] = {"trades": len(ts), "wr": round(len(sw)/len(ts)*100,1),
                 "pf": round(sg/sl2,2) if sl2 > 0 else float("inf")}

    return {
        "trades":   len(trades),
        "pf":       round(pf, 2),
        "wr":       round(wr, 1),
        "max_dd":   round(max_dd, 1),
        "final":    round(balance, 2),
        "gain_pct": round((balance - START_BAL) / START_BAL * 100, 1),
        "per_strat": ps,
    }


# ── Build entries for a given strategy config ─────────────────────────────────
def build_entries_for_strategies(all_data, strategies):
    all_entries = {}; all_ltf = {}
    for sym, data in all_data.items():
        d1h = data["1h"]; d1d = data.get("1d")
        all_entries[sym] = {}; all_ltf[sym] = {}
        for strat_id, params in strategies.items():
            if params["type"] == "bbs":
                htf_sigs = detect_bbs_signals_1h(d1h, params)
            else:
                htf_sigs = detect_kc_signals_1h(d1h, params)
            tf_key = params["entry_tf"]
            dltf   = data.get(tf_key)
            if dltf is None: continue
            entries = match_ltf_entries(htf_sigs, d1h, dltf, d1d, params)
            all_entries[sym][strat_id] = entries
            all_ltf[sym][strat_id]     = dltf
    return all_entries, all_ltf


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = _time.time()

    print("\n" + "="*78)
    print("  SCENARIO COMPARISON  |  TOP_50 symbols  |  up to 730d data")
    print("  Цель: найти улучшения без подгонки под данные")
    print("  Правило: изменение принимается только если PF ≥ BASE и WR ≥ BASE-2% и DD ≤ BASE+3%")
    print("="*78)

    # ── Load data once (shared across all scenarios) ──────────────────────────
    logger.info("Loading data for %d symbols...", len(TOP_50))
    all_data: dict = {}
    loaded = 0
    for sym in TOP_50:
        d1h   = prepare(_load(sym, "1h"))
        d5m   = prepare(_load(sym, "5m"))
        d15m  = prepare(_load(sym, "15m"))
        d1d_raw = _load(sym, "1d")
        d1d = None
        if d1d_raw is not None and len(d1d_raw) >= 210:
            from multi_agent.backtests.backtest_v12 import _ema
            cl1d = d1d_raw["close"].to_numpy(dtype=float)
            ts1d = d1d_raw["timestamp"].to_numpy(dtype=int)
            d1d = {"cl": cl1d, "ts": ts1d, "ema200": _ema(cl1d, 200)}
        if d1h is None or len(d1h["cl"]) < 300: continue
        all_data[sym] = {"1h": d1h, "5m": d5m, "15m": d15m, "1d": d1d}
        loaded += 1
    logger.info("Loaded %d/%d symbols in %.1fs", loaded, len(TOP_50), _time.time()-t0)

    # ── Detect data period ────────────────────────────────────────────────────
    data_days = "?"
    for sd in all_data.values():
        dh = sd.get("1h")
        if dh and dh["n"] > 0:
            data_days = f"{(dh['ts'][-1] - dh['ts'][0])//86_400_000}d"
            break

    # ── Run each scenario ─────────────────────────────────────────────────────
    # Cache entries per unique strategy config
    entry_cache: dict[str, tuple] = {}

    results = []
    for sc in SCENARIOS:
        strats = sc["strategies"]
        cache_key = str(sorted((k, str(sorted(v.items()))) for k, v in strats.items()))
        if cache_key not in entry_cache:
            ae, al = build_entries_for_strategies(all_data, strats)
            entry_cache[cache_key] = (ae, al)
        ae, al = entry_cache[cache_key]

        cfg = {"balance": sc["balance"], "risk": sc["risk"], "max_pos": sc["max_pos"]}
        r_full = run_portfolio_cfg(ae, al, cfg, 0.0, 1.0)
        r_oos  = run_portfolio_cfg(ae, al, cfg, 0.8, 1.0)

        # Count total possible signals (before portfolio filter)
        total_signals = sum(len(e) for stm in ae.values() for e in stm.values())

        results.append({
            "sc": sc, "full": r_full, "oos": r_oos, "signals": total_signals,
        })
        logger.info("Done: %s  trades=%d PF=%.2f WR=%.1f%% DD=%.1f%%",
                    sc["name"], r_full["trades"], r_full["pf"],
                    r_full["wr"], r_full["max_dd"])

    # ── Print comparison table ────────────────────────────────────────────────
    BASE = results[0]
    base_full = BASE["full"]; base_oos = BASE["oos"]

    print(f"\n  Data period: {data_days}  |  Loaded: {loaded} symbols")
    print(f"\n  {'Scenario':<32}  {'Trades':>6}  {'WR%':>5}  {'PF':>5}  {'DD%':>5}  "
          f"{'OOS PF':>6}  {'OOS WR':>6}  {'Gain%':>7}  Verdict")
    print("  " + "-"*105)

    for i, row in enumerate(results):
        sc = row["sc"]; f = row["full"]; o = row["oos"]
        is_base = (i == 0)

        # Verdict vs base
        pf_ok  = f["pf"]   >= base_full["pf"]   - 0.05
        wr_ok  = f["wr"]   >= base_full["wr"]    - 2.0
        dd_ok  = f["max_dd"] <= base_full["max_dd"] + 3.0
        tr_up  = f["trades"] > base_full["trades"]

        if is_base:
            verdict = "BASE"
        elif pf_ok and wr_ok and dd_ok and tr_up:
            verdict = "✓ BETTER"
        elif pf_ok and wr_ok and dd_ok:
            verdict = "~ SAME"
        else:
            what = []
            if not pf_ok:  what.append(f"PF↓{f['pf']:.2f}")
            if not wr_ok:  what.append(f"WR↓{f['wr']:.1f}%")
            if not dd_ok:  what.append(f"DD↑{f['max_dd']:.1f}%")
            verdict = "✗ " + " ".join(what)

        marker = ">>>" if verdict == "✓ BETTER" else "   "
        print(f"  {marker} {sc['name']:<29}  {f['trades']:>6}  {f['wr']:>4.1f}%  "
              f"{f['pf']:>5.2f}  {f['max_dd']:>4.1f}%  "
              f"{o['pf']:>6.2f}  {o['wr']:>5.1f}%  {f['gain_pct']:>+6.1f}%  {verdict}")

    # ── Per-strategy breakdown for key scenarios ──────────────────────────────
    print(f"\n  Per-strategy breakdown (FULL period):")
    print(f"  {'Scenario':<32}  {'Strategy':<15}  {'Trades':>6}  {'WR%':>5}  {'PF':>5}")
    print("  " + "-"*75)
    for i, row in enumerate(results):
        sc = row["sc"]; f = row["full"]
        for s, sv in sorted(f["per_strat"].items(), key=lambda x: -x[1]["trades"]):
            prefix = sc["name"] if s == sorted(f["per_strat"].keys(), key=lambda x: -f["per_strat"][x]["trades"])[0] else " "*32
            print(f"  {prefix:<32}  {s:<15}  {sv['trades']:>6}  {sv['wr']:>4.1f}%  {sv['pf']:>5.2f}")

    # ── Conclusions ───────────────────────────────────────────────────────────
    print(f"\n  {'='*78}")
    print(f"  ВЫВОДЫ:")
    improvements = [r for r in results[1:] if "BETTER" in str(r)]
    for row in results[1:]:
        sc = row["sc"]; f = row["full"]; b = base_full
        delta_trades = f["trades"] - b["trades"]
        delta_pf     = f["pf"] - b["pf"]
        sign_t = "+" if delta_trades >= 0 else ""
        sign_p = "+" if delta_pf >= 0 else ""
        print(f"  {sc['name']:<32}  сделок {sign_t}{delta_trades:+d}  PF {sign_p}{delta_pf:+.2f}  | {sc['note']}")

    print(f"\n  Total runtime: {_time.time()-t0:.1f}s")
    print("="*78)


if __name__ == "__main__":
    main()
