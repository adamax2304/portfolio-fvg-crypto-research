"""Quick: show what symbols the COMPOSITE filter picks for each month."""
from __future__ import annotations
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from multi_agent.backtests.universe_filters import (
    list_available_symbols, compute_monthly_metrics, selector_composite,
)

all_syms = list_available_symbols("365d", required_tfs=("1d", "4h", "1h"))
print(f"Available 365d: {len(all_syms)}")
metrics = {}
for s in all_syms:
    m = compute_monthly_metrics(s, "365d")
    if m: metrics[s] = m
sel = selector_composite(metrics, top_n=50)
print(f"\nComposite picks per month (top 50):")
all_picks_count = {}
for month in sorted(sel.keys()):
    syms = sel[month]
    print(f"\n  {month} ({len(syms)} syms): {' '.join(syms[:25])}...")
    for s in syms:
        all_picks_count[s] = all_picks_count.get(s, 0) + 1

print(f"\n══ Symbol pick frequency (months selected / total months) ══")
sorted_freq = sorted(all_picks_count.items(), key=lambda x: -x[1])
total_months = len(sel)
print(f"  Total months: {total_months}")
print(f"  {'Symbol':10s} {'Picks':>6s} {'%':>6s}")
for s, cnt in sorted_freq:
    pct = cnt / total_months * 100
    print(f"  {s:10s} {cnt:>6d} {pct:>5.0f}%")
