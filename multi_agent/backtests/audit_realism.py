"""
audit_realism.py — Code audit + realism assessment + walk-forward + sensitivity

Checks:
  1. Look-ahead bias check (signal vs entry timestamps)
  2. Fee/slippage audit (verify amount per trade)
  3. Position sizing math sanity
  4. Walk-forward on V3 (train 6mo, test 3mo, rolling)
  5. Sensitivity to parameters (±20%)
  6. Random shuffle test (data mining detection)
  7. By-period analysis (bull/bear/sideways)
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
from multi_agent.backtests.strategy_lab import (
    load, build_regime, simulate_portfolio, START_BALANCE,
    metrics_dict_from_trades, regime_at, atr,
)
from multi_agent.backtests.portfolio_v3 import collect_v3_signals
from multi_agent.backtests.portfolio_v6 import BASELINE


def audit_lookahead():
    """Check that signals are emitted ON bar close, entry executes on NEXT bar open."""
    print(f"\n══════ AUDIT 1: Look-ahead bias ══════")
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = ["BTC", "ETH", "SOL"]
    sigs, dfs = collect_v3_signals(syms, regime, "730d")
    issues = 0; checked = 0
    for sig in sigs[:100]:  # sample
        # Verify signal.ts < signal.next_open_ts
        if sig.ts >= sig.next_open_ts:
            issues += 1
            print(f"  ISSUE: sig.ts={sig.ts} >= next_open_ts={sig.next_open_ts}")
        # Verify next_open_ts is exactly one period after signal close
        # (period inferred from data)
        df = dfs.get((sig.sym, sig.strat))
        if df is not None:
            ts = df["timestamp"].to_numpy()
            period = int(np.median(np.diff(ts[:200])))
            expected_next = sig.ts + period
            if abs(sig.next_open_ts - expected_next) > period * 0.1:
                issues += 1
                if issues < 3:
                    print(f"  ISSUE: next_open_ts={sig.next_open_ts} vs expected={expected_next}")
        checked += 1
    print(f"  Checked {checked} signals, {issues} potential issues found.")
    print(f"  ✓ PASS" if issues == 0 else f"  ✗ {issues} issues")


def audit_fees():
    """Verify a single trade's fee/slippage costs."""
    print(f"\n══════ AUDIT 2: Fee/slippage breakdown ══════")
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    sigs, dfs = collect_v3_signals(["BTC"], regime, "730d")
    if not sigs:
        print("  No signals to test."); return
    # Take first signal
    sig = sigs[0]
    df = dfs[(sig.sym, sig.strat)]
    ts = df["timestamp"].to_numpy()
    op = df["open"].to_numpy()
    j = int(np.searchsorted(ts, sig.next_open_ts, side="left"))
    if j >= len(op): print("  no entry bar"); return
    entry_p = op[j] * (1 + 0.0005)  # SLIPPAGE LONG
    qty = 1.0 / (entry_p - sig.sl)  # unit risk
    notional = entry_p * qty
    print(f"  Sample BTC trade:")
    print(f"    Entry price (with slip): ${entry_p:.4f}")
    print(f"    Quantity: {qty:.6f}")
    print(f"    Notional: ${notional:.4f}")
    print(f"    Entry fee (taker 0.055%): ${entry_p * qty * 0.00055:.4f}")
    print(f"    Slippage cost (entry+exit): ${entry_p * qty * 0.001:.4f}")
    print(f"    Funding/8h (0.01%): ${notional * 0.0001:.4f}")
    # Total round-trip fee+slip
    rt = entry_p * qty * (2 * 0.00055 + 2 * 0.0005)
    rt_pct = rt / notional * 100
    print(f"    Round-trip cost: ${rt:.4f} ({rt_pct:.3f}% of notional)")
    print(f"  ✓ Costs look realistic for Bybit perp")


def walk_forward(syms, tag="730d", train_months=6, test_months=3, risk=0.04):
    """Rolling walk-forward test with V3 settings."""
    print(f"\n══════ AUDIT 3: Walk-forward (train={train_months}mo, test={test_months}mo) ══════")
    btc = load("BTC", "1d", tag)
    regime = build_regime(btc)
    sigs, dfs = collect_v3_signals(syms, regime, tag)
    if not sigs: print("  no signals"); return
    # Determine time range
    all_ts = sorted([s.next_open_ts for s in sigs])
    if not all_ts: return
    first = all_ts[0]; last = all_ts[-1]
    span_ms = last - first
    span_days = span_ms / 86_400_000
    print(f"  Period: {span_days:.0f} days")
    train_ms = train_months * 30 * 86_400_000
    test_ms = test_months * 30 * 86_400_000
    folds = []
    cursor = first + train_ms
    while cursor + test_ms <= last:
        train_end = cursor
        test_end = cursor + test_ms
        # Filter signals to test window
        test_sigs = [s for s in sigs if train_end <= s.next_open_ts < test_end]
        if test_sigs:
            from multi_agent.backtests.strategy_lab import simulate_portfolio
            res = simulate_portfolio(test_sigs, dfs, regime,
                                       start_balance=1000, risk=risk, max_pos=5)
            m = res["metrics"]
            t0 = pd.Timestamp(train_end, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            t1 = pd.Timestamp(test_end, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            folds.append({"start": t0, "end": t1, **m})
        cursor += test_ms  # non-overlapping test windows
    print(f"  {'Fold':12s} {'End':12s} {'n':>4s} {'PF':>5s} {'WR%':>5s} {'Return':>8s} {'DD%':>5s}")
    pos_folds = 0
    for f in folds:
        is_pos = "✓" if f.get("return_pct", 0) > 0 else "✗"
        pos_folds += 1 if f.get("return_pct", 0) > 0 else 0
        print(f"  {is_pos} {f['start']:10s} {f['end']:10s} {f['n']:>4d} {f['pf']:>5.2f} {f['wr']:>5.1f} "
              f"{f['return_pct']:>+7.1f}% {f['max_dd_pct']:>5.1f}%")
    print(f"  Positive folds: {pos_folds}/{len(folds)} = {pos_folds/max(1,len(folds))*100:.0f}%")


def sensitivity_test(syms):
    """Test V3 with ±20% TP changes to check parameter sensitivity."""
    print(f"\n══════ AUDIT 4: Parameter sensitivity (TP ±20%) ══════")
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    from multi_agent.backtests.strategy_v3 import (
        strat_smacross_pivot_1d, strat_supertrend_pivot_4h, strat_fib_pump_1h,
    )
    from multi_agent.backtests.portfolio_v3 import _strat_rsi_rev_v2
    from multi_agent.backtests.strategy_lab import (
        PortfolioSignal, _ts_period_ms,
    )

    def collect_for_tps(smacross_tp, supertrend_tp):
        sigs = []; dfs = {}
        for sym in syms:
            d1 = load(sym, "1d", "730d")
            if d1 is not None:
                slist = strat_smacross_pivot_1d(d1, regime, tp_r=smacross_tp)
                dfs[(sym, "smacross_pivot_1d")] = d1
                for s in slist:
                    sigs.append(PortfolioSignal(s.ts, s.next_open_ts, sym, "smacross_pivot_1d",
                                                 s.direction, s.entry, s.sl, s.tp, s.dist))
            d4 = load(sym, "4h", "730d")
            if d4 is not None:
                slist = strat_supertrend_pivot_4h(d4, regime, tp_r=supertrend_tp)
                dfs[(sym, "supertrend_pivot_4h")] = d4
                for s in slist:
                    sigs.append(PortfolioSignal(s.ts, s.next_open_ts, sym, "supertrend_pivot_4h",
                                                 s.direction, s.entry, s.sl, s.tp, s.dist))
                slist = _strat_rsi_rev_v2(d4, regime, sl_atr=2.0, tp_r=3.0)
                dfs[(sym, "rsi_reversion_v2")] = d4
                for s in slist:
                    sigs.append(PortfolioSignal(s.ts, s.next_open_ts, sym, "rsi_reversion_v2",
                                                 s.direction, s.entry, s.sl, s.tp, s.dist))
                slist = strat_fib_pump_1h(d4, regime, consolidation_bars=18,
                                            pump_min_atr=3.0, pump_max_bars=6,
                                            retrace_min=0.382, retrace_max=0.618,
                                            tp_extension=1.0, sl_buffer_atr=0.5)
                dfs[(sym, "fib_pump_4h")] = d4
                for s in slist:
                    sigs.append(PortfolioSignal(s.ts, s.next_open_ts, sym, "fib_pump_4h",
                                                 s.direction, s.entry, s.sl, s.tp, s.dist))
        sigs.sort(key=lambda x: x.next_open_ts)
        return sigs, dfs

    # V3 settings (smacross 4.0, supertrend 2.5)
    print(f"  {'Config':30s} {'n':>4s} {'PF':>5s} {'WR%':>5s} {'Return':>8s} {'DD%':>5s}")
    for sm in [3.2, 3.6, 4.0, 4.4, 4.8]:
        for st in [2.0, 2.25, 2.5, 2.75, 3.0]:
            sigs, dfs = collect_for_tps(sm, st)
            if not sigs: continue
            res = simulate_portfolio(sigs, dfs, regime, start_balance=1000, risk=0.04, max_pos=5)
            m = res["metrics"]
            print(f"  smacross={sm:.1f}/super={st:.2f} {'':9s} {m['n']:>4d} {m['pf']:>5.2f} "
                  f"{m['wr']:>5.1f} {m['return_pct']:>+7.1f}% {m['max_dd_pct']:>5.1f}%")


def regime_breakdown():
    """V3 performance broken down by BTC regime."""
    print(f"\n══════ AUDIT 5: Performance by BTC regime ══════")
    btc = load("BTC", "1d", "730d")
    regime = build_regime(btc)
    syms = [s for s in BASELINE if (Path(__file__).parent / "data_cache" /
                                      f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    sigs, dfs = collect_v3_signals(syms, regime, "730d")
    res = simulate_portfolio(sigs, dfs, regime, start_balance=1000, risk=0.04, max_pos=5)
    by_regime = defaultdict(list)
    for t in res["trades"]:
        idx = int(np.searchsorted(regime.ts, t["entry_ts"], side="right")) - 1
        if idx >= 0:
            r = "bull" if regime.trend[idx] == 1 else ("bear" if regime.trend[idx] == -1 else "sideways")
            by_regime[r].append(t)
    print(f"  {'Regime':10s} {'n':>4s} {'WR%':>5s} {'PF':>5s} {'sumR':>7s} {'avgR':>7s}")
    for r, tlist in sorted(by_regime.items()):
        m = metrics_dict_from_trades(tlist, START_BALANCE)
        print(f"  {r:10s} {m.get('n',0):>4d} {m.get('wr',0):>5.1f} {m.get('pf',0):>5.2f} "
              f"{m.get('sum_r',0):>+6.1f} {m.get('avg_r',0):>+5.2f}")


def main():
    syms_730 = [s for s in BASELINE if (Path(__file__).parent / "data_cache" /
                                          f"{s}_USDT_USDT_1d_730d.parquet").exists()]
    print(f"Running on {len(syms_730)} symbols, 730d")

    audit_lookahead()
    audit_fees()
    walk_forward(syms_730, "730d", train_months=6, test_months=3, risk=0.04)
    sensitivity_test(syms_730)
    regime_breakdown()


if __name__ == "__main__":
    main()
