# Market-Neutral Phase 2 (2026-05-04)

## What changed

- Refreshed full `1h` OHLCV cache for the 27-symbol research universe using the fixed
  [fetch_data_2y.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/fetch_data_2y.py:1).
- Added force/subset support to
  [fetch_funding.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/fetch_funding.py:1)
  and refreshed funding history for the same 27 symbols.
- Built a paper-trading helper:
  [xsec_v3_paper_runner.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/xsec_v3_paper_runner.py:1)
  which exports both historical validation and the latest target basket.

## Important finding

Refreshing the cache materially changed the results.

The previously favored `xsec v3` broad config (`corr_cap=0.70`, `retain_extra=3`)
no longer holds up on the refreshed sample:

- old view on stale cache: `+52.65%`, `PF 1.70`, `MaxDD 12.72%`
- refreshed sample: `+3.30%`, `PF 1.127`, `MaxDD 22.19%`

This is exactly why the refresh step mattered: the earlier picture was too optimistic.

## New best refreshed config

Fresh full-grid report:
[xsec_momentum_v3_20260504_010415.json](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/xsec_momentum_v3_20260504_010415.json:1)

Current best broad candidate:

- `lookback_h=24`
- `hold_h=24`
- `top_n=2`
- `min_abs_ret=0.02`
- `liquid_n=15`
- `funding_tilt=0`
- `corr_cap=0.55`
- `retain_extra=1`

Metrics on refreshed cache:

- `+29.12%`
- `PF 1.616`
- `MaxDD 12.66%`
- `72` rebalances
- `11 / 26` profitable months

This is still not a “monthly income machine”, but it is the best surviving
market-neutral candidate in the current local research set.

## Current paper signal

Latest paper snapshot:
[xsec_v3_paper_runner_20260504_010611.json](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/xsec_v3_paper_runner_20260504_010611.json:1)

Historical rebalance ledger:
[xsec_v3_paper_rebalances_20260504_010611.csv](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/xsec_v3_paper_rebalances_20260504_010611.csv:1)

Current state is `flat_no_signal`, not because of a bug, but because the latest
rebalance window had only `2` qualifying longs and `0` qualifying shorts in the
liquid universe. In other words, the long/short filter is currently doing its job
and refusing to force a basket.

Latest diagnostics:

- rebalance timestamp: `2026-05-02 22:00 UTC`
- latest cache bar: `2026-05-03 21:00 UTC`
- next rebalance in roughly `1h`
- top longs: `ALGO`, `DOT`
- no qualifying shorts

Previous active basket before going flat:

- longs: `TRX`, `INJ`
- shorts: `ARB`, `OP`

## Practical recommendation

- Do **not** promote the stale-cache `corr_cap=0.70 / retain_extra=3` version.
- If we continue this track, paper-track only the refreshed winner
  `corr_cap=0.55 / retain_extra=1`.
- Treat `flat` periods as a feature, not a defect. This model should be allowed
  to hold no basket when cross-sectional dispersion is one-sided.
- Before any live wrapper, the next useful step is a small daily monitor around
  [xsec_v3_paper_runner.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/xsec_v3_paper_runner.py:1):
  run it on schedule, archive the JSON/CSV outputs, and compare signal stability
  over several weeks.
