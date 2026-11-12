"""Run only the 3 remaining backtest configurations."""
import sys, os, time, json, logging
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from multi_agent.backtests.backtest_portfolio import (
    run_single_agent_backtest, run_combined_portfolio, RESULTS_DIR
)

logging.basicConfig(level=logging.WARNING, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("remaining")
logger.setLevel(logging.INFO)

results = {}

# Test 6: FVG no-regime ablation
logger.info("--- Running FVG standalone (NO regime gate — ablation) ---")
t0 = time.time()
r = run_single_agent_backtest("fvg", use_regime=False)
elapsed = time.time() - t0
results["fvg_no_regime"] = r
logger.info("FVG_NOREG    | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
    r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

# Test 7: Combined portfolio WITH regime
logger.info("--- Running COMBINED portfolio (regime-aware) ---")
t0 = time.time()
r = run_combined_portfolio(use_regime=True)
elapsed = time.time() - t0
results["combined_regime"] = r
logger.info("COMBINED_REG | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
    r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

# Test 8: Combined portfolio WITHOUT regime
logger.info("--- Running COMBINED portfolio (NO regime — ablation) ---")
t0 = time.time()
r = run_combined_portfolio(use_regime=False)
elapsed = time.time() - t0
results["combined_no_regime"] = r
logger.info("COMBINED_NOR | Trades=%3d | WR=%.1f%% | PnL=%+.2f%% | MaxDD=%.2f%% | PF=%.2f | %.1fs",
    r["trades"], r["wr_pct"], r["pnl_pct"], r["max_dd_pct"], r["profit_factor"], elapsed)

# Print summary
print("\n" + "=" * 90)
print("REMAINING TESTS — RESULTS")
print("=" * 90)
for key, r in results.items():
    label = key.replace("_", " ").upper()
    print(f"{label:<25} Trades={r['trades']:>4} WR={r['wr_pct']:>5.1f}% PnL={r['pnl_pct']:>+7.2f}% "
          f"MaxDD={r['max_dd_pct']:>6.2f}% PF={r['profit_factor']:>5.2f} Bal=${r['final_balance']:>.2f}")

# Per-agent breakdown for combined
for key in ["combined_regime", "combined_no_regime"]:
    combo = results.get(key, {}).get("per_agent", {})
    if combo:
        print(f"\n  {key.upper()} per-agent:")
        for agent, stats in combo.items():
            print(f"    {agent.upper():<12} Trades={stats['trades']:>3} WR={stats['wr_pct']:>5.1f}% "
                  f"PnL=${stats['pnl_usd']:>+8.2f} PF={stats['profit_factor']:>5.2f}")

# Save
from datetime import datetime
path = RESULTS_DIR / f"remaining_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(path, "w") as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nSaved to: {path}")
