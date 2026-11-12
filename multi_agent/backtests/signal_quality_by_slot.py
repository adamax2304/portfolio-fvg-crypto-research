"""
signal_quality_by_slot.py — Does signal quality drop as portfolio fills?

Key question: are trades taken when portfolio is already "full" (3-4 open)
weaker than trades taken when few slots are occupied?

If WR/PF drops as portfolio fills → signals not selective enough.
If WR/PF stable → portfolio staying full is fine, system is genuinely busy.
"""
from __future__ import annotations
import sys
from collections import defaultdict
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from multi_agent.backtests.strategy_lab import load, build_regime, START_BALANCE
from multi_agent.backtests.portfolio_v7 import BASELINE, collect_v7_signals, CACHE_DIR
from multi_agent.backtests.news_aware_v7 import load_all_events
from multi_agent.backtests.pivot_dist_filter import apply_pivot_dist_filter
from multi_agent.backtests.risk_controls_v7 import simulate_rc


def analyse_by_slot(trades):
    """For each trade record how many positions were open AT ENTRY TIME."""
    # Build sorted timeline of (ts, event, trade)
    events = []
    for t in trades:
        events.append((t["entry_ts"], 0, "open",  t))   # 0 = open first
        events.append((t["exit_ts"],  1, "close", t))   # 1 = close second
    events.sort(key=lambda x: (x[0], x[1]))

    open_set  = set()
    slot_at_entry = {}   # trade id → n_open BEFORE this trade opened

    for ts, order, kind, t in events:
        tid = id(t)
        if kind == "open":
            slot_at_entry[tid] = len(open_set)   # how many already open
            open_set.add(tid)
        else:
            open_set.discard(tid)

    # Attach slot info to trades
    for t in trades:
        t["_slot"] = slot_at_entry.get(id(t), 0)

    return trades


def print_by_slot(trades, label):
    by_slot = defaultdict(list)
    for t in trades:
        by_slot[t["_slot"]].append(t)

    print(f"\n{'━'*72}")
    print(f"  {label}")
    print(f"{'━'*72}")
    print(f"\n  Качество сигнала в зависимости от заполненности портфеля при входе:\n")
    print(f"  {'Слот':>6s}  {'n':>4s}  {'WR%':>6s}  {'PF':>5s}  {'AvgR':>6s}  "
          f"{'SumR':>7s}  {'Wins':>5s}  {'Loss':>5s}  Интерпретация")
    print(f"  {'':>6s}  {'':>4s}  {'':>6s}  {'':>5s}  {'':>6s}  "
          f"{'':>7s}  {'(% n)':>5s}  {'(% n)':>5s}")
    print("  " + "─" * 70)

    all_pnl_r  = []
    prev_wr    = None
    degrades   = []

    for slot in sorted(by_slot.keys()):
        ts = by_slot[slot]
        wins   = [t for t in ts if t["pnl"] > 0]
        losses = [t for t in ts if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in wins)
        gl = abs(sum(t["pnl"] for t in losses))
        pf = gw / gl if gl > 0 else 99.0
        wr = len(wins) / len(ts) * 100
        avg_r = np.mean([t["pnl_r"] for t in ts])
        sum_r = sum(t["pnl_r"] for t in ts)
        all_pnl_r.extend([t["pnl_r"] for t in ts])

        label_slot = {
            0: "первый  (пустой портфель)",
            1: "второй  (1 открыта)",
            2: "третий  (2 открыто)",
            3: "четвёртый (3 открыто)",
            4: "пятый   (4 открыто)",
        }.get(slot, f"слот {slot+1} ({slot} открыто)")

        degrade = ""
        if prev_wr is not None and wr < prev_wr - 5:
            degrade = " ← хуже!"
            degrades.append(slot)
        prev_wr = wr

        print(f"  {slot+1:>6d}  {len(ts):>4d}  {wr:>6.1f}%  {pf:>5.2f}  "
              f"{avg_r:>+6.2f}  {sum_r:>+7.1f}  "
              f"{len(wins)/len(ts)*100:>4.0f}%  {len(losses)/len(ts)*100:>4.0f}%"
              f"  {label_slot}{degrade}")

    print("  " + "─" * 70)
    # Overall
    all_trades = [t for ts in by_slot.values() for t in ts]
    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    gw = sum(t["pnl"] for t in wins)
    gl = abs(sum(t["pnl"] for t in losses))
    pf = gw / gl if gl > 0 else 99.0
    wr = len(wins) / len(all_trades) * 100
    avg_r = np.mean([t["pnl_r"] for t in all_trades])
    print(f"  {'TOTAL':>6s}  {len(all_trades):>4d}  {wr:>6.1f}%  {pf:>5.2f}  "
          f"{avg_r:>+6.2f}  {sum(t['pnl_r'] for t in all_trades):>+7.1f}")

    print(f"\n  Вывод: ", end="")
    if not degrades:
        print("качество СТАБИЛЬНО по всем слотам — портфель заполнен обоснованно.")
    else:
        print(f"деградация на слотах {[s+1 for s in degrades]} — "
              f"система берёт слабые сигналы чтобы заполнить слот.")

    # By strategy per slot
    print(f"\n  WR% по стратегии + слоту:\n")
    strats = sorted({t["strat"] for t in all_trades})
    header = f"  {'':>12s}" + "".join(f"  {'Слот'+str(s+1):>7s}" for s in sorted(by_slot.keys()))
    print(header)
    print("  " + "─" * (14 + 9 * len(by_slot)))
    for strat in strats:
        row = f"  {strat:>12s}"
        for slot in sorted(by_slot.keys()):
            ts = [t for t in by_slot[slot] if t["strat"] == strat]
            if ts:
                wr_s = sum(1 for t in ts if t["pnl"] > 0) / len(ts) * 100
                row += f"  {wr_s:>5.0f}% " + f"({len(ts)})"[:3]
            else:
                row += f"  {'—':>7s}"
        print(row)

    # Signals blocked by max_pos — how many and what quality
    return degrades


def print_blocked_signals(sigs_all, trades, sim_dfs, label=""):
    """Estimate quality of signals that were BLOCKED (never became trades)."""
    # Trades that actually happened
    taken_keys = {(t["sym"], t["entry_ts"]) for t in trades}
    # All signals as potential trades
    from multi_agent.backtests.strategy_lab import (
        atr as _atr, _ts_period_ms,
        TAKER_FEE, SLIPPAGE, FUNDING_PER_8H,
        MAX_LEVERAGE, MAX_MARGIN_FRAC, MIN_MARGIN,
    )
    import numpy as np

    # We can't know which signals were "blocked" without re-running with max_pos=999
    # Instead: run with max_pos=999 (uncapped) to see all possible trades
    pass


def main():
    print("\n══════════════════════════════════════════════════════════════════════")
    print("  SIGNAL QUALITY BY PORTFOLIO SLOT")
    print("══════════════════════════════════════════════════════════════════════")

    btc    = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms   = [s for s in BASELINE
              if (CACHE_DIR / f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"\n  Symbols: {len(syms)}  |  Loading...")
    sigs_raw, sim_dfs, _ = collect_v7_signals(syms, regime, "730d", bear_mode="long_only")
    events = load_all_events()
    sigs, _, _ = apply_pivot_dist_filter(sigs_raw, sim_dfs, 7.0)
    print(f"  Signals: {len(sigs)}\n")

    # ── Baseline (max_pos=5) ──────────────────────────────────────────────────
    res5 = simulate_rc(sigs, sim_dfs, regime, events, max_pos=5)
    trades5 = analyse_by_slot(res5["trades"])
    print_by_slot(trades5, "BASELINE max_pos=5")

    # ── Uncapped (max_pos=999) — all signals that could trade ────────────────
    res_all = simulate_rc(sigs, sim_dfs, regime, events, max_pos=999)
    trades_all = analyse_by_slot(res_all["trades"])
    print_by_slot(trades_all, "UNCAPPED max_pos=999 (все возможные сигналы)")

    # ── Compare: what do the BLOCKED trades look like? ───────────────────────
    taken_keys = {(t["sym"], t["entry_ts"]) for t in res5["trades"]}
    blocked    = [t for t in trades_all if (t["sym"], t["entry_ts"]) not in taken_keys]

    if blocked:
        bw = [t for t in blocked if t["pnl"] > 0]
        bl = [t for t in blocked if t["pnl"] <= 0]
        gw = sum(t["pnl"] for t in bw)
        gl = abs(sum(t["pnl"] for t in bl))
        pf_b = gw / gl if gl > 0 else 99.0
        wr_b = len(bw) / len(blocked) * 100 if blocked else 0
        avg_r_b = np.mean([t["pnl_r"] for t in blocked]) if blocked else 0

        taken = res5["trades"]
        tw = [t for t in taken if t["pnl"] > 0]
        tl = [t for t in taken if t["pnl"] <= 0]
        gw2 = sum(t["pnl"] for t in tw)
        gl2 = abs(sum(t["pnl"] for t in tl))
        pf_t = gw2 / gl2 if gl2 > 0 else 99.0
        wr_t = len(tw) / len(taken) * 100 if taken else 0
        avg_r_t = np.mean([t["pnl_r"] for t in taken]) if taken else 0

        m_all = res_all["metrics"]

        print(f"\n{'━'*72}")
        print(f"  ЗАБЛОКИРОВАННЫЕ vs ВЗЯТЫЕ трейды (max_pos=5)")
        print(f"{'━'*72}")
        print(f"\n  {'':20s}  {'n':>5s}  {'WR%':>6s}  {'PF':>5s}  {'AvgR':>6s}")
        print(f"  {'─'*50}")
        print(f"  {'Взятые (max_pos=5)':20s}  {len(taken):>5d}  {wr_t:>6.1f}%  "
              f"{pf_t:>5.2f}  {avg_r_t:>+6.2f}")
        print(f"  {'Заблокированные':20s}  {len(blocked):>5d}  {wr_b:>6.1f}%  "
              f"{pf_b:>5.2f}  {avg_r_b:>+6.2f}")

        print(f"\n  ВЫВОД по заблокированным:")
        if wr_b > wr_t - 3 and pf_b > pf_t - 0.1:
            print(f"  → Заблокированные НЕ хуже взятых. Снятие лимита даст ещё +{m_all['return_pct'] - res5['metrics']['return_pct']:+.0f}%")
            print(f"    но DD вырастет: {res5['metrics']['max_dd_pct']:.1f}% → {m_all['max_dd_pct']:.1f}%")
        elif wr_b < wr_t - 5:
            print(f"  → Заблокированные ХУЖЕ взятых (WR {wr_b:.1f}% vs {wr_t:.1f}%)")
            print(f"    → max_pos=5 правильно отсекает слабые сигналы (FIFO работает).")
        else:
            print(f"  → Заблокированные примерно одинаковы с взятыми.")
            print(f"    → Очерёдность (FIFO) не создаёт явного перекоса.")

        # By strategy breakdown for blocked
        print(f"\n  Заблокированные по стратегии:")
        print(f"  {'Стратегия':>14s}  {'Взято':>6s}  {'Заблок':>7s}  "
              f"{'WR взято':>9s}  {'WR заблок':>10s}  {'PF взято':>9s}  {'PF заблок':>10s}")
        print("  " + "─" * 72)
        for strat in sorted({t["strat"] for t in trades_all}):
            tk = [t for t in taken    if t["strat"] == strat]
            bk = [t for t in blocked  if t["strat"] == strat]
            if not tk and not bk: continue
            wr_tk = sum(1 for t in tk if t["pnl"]>0)/len(tk)*100 if tk else 0
            wr_bk = sum(1 for t in bk if t["pnl"]>0)/len(bk)*100 if bk else 0
            gw_t = sum(t["pnl"] for t in tk if t["pnl"]>0)
            gl_t = abs(sum(t["pnl"] for t in tk if t["pnl"]<=0))
            gw_b = sum(t["pnl"] for t in bk if t["pnl"]>0)
            gl_b = abs(sum(t["pnl"] for t in bk if t["pnl"]<=0))
            pf_tk = gw_t/gl_t if gl_t>0 else 99.0
            pf_bk = gw_b/gl_b if gl_b>0 else 99.0
            diff = " ← ХУЖЕ" if wr_bk < wr_tk - 5 else (" ← ЛУЧШЕ" if wr_bk > wr_tk + 5 else "")
            print(f"  {strat:>14s}  {len(tk):>6d}  {len(bk):>7d}  "
                  f"{wr_tk:>7.1f}%  {wr_bk:>8.1f}%  "
                  f"{pf_tk:>7.2f}  {pf_bk:>8.2f}{diff}")

    print(f"\n{'━'*72}\n")


if __name__ == "__main__":
    main()
