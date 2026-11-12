# Market-Neutral Phase 1 — 2026-05-04

## What changed

1. Fixed the 2-year low-timeframe fetch bug in:

- [fetch_data_2y.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/fetch_data_2y.py:1)

The old script capped downloads at `100` batches of `1000` candles, which silently truncated `5m` history to about `347` days. After the fix, a targeted BTC refetch produced `210,840` rows and now covers roughly `2024-04-30` to `2026-05-02`.

2. Added dedicated research runners:

- [backtest_xsec_momentum_ls.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/backtest_xsec_momentum_ls.py:1)
- [backtest_xsec_momentum_v2.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/backtest_xsec_momentum_v2.py:1)
- [backtest_funding_carry_approx.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/backtest_funding_carry_approx.py:1)

3. Saved fresh reports:

- [xsec_momentum_ls_20260504_000212.json](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/xsec_momentum_ls_20260504_000212.json:1)
- [xsec_momentum_v2_20260504_000754.json](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/xsec_momentum_v2_20260504_000754.json:1)
- [funding_carry_approx_20260504_000103.json](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/funding_carry_approx_20260504_000103.json:1)

## Best findings

### 1. Cross-sectional long/short momentum

Best narrow regime:

- lookback: `24h`
- hold: `8h`
- `top_n = 4`
- minimum absolute momentum: `2%`
- result: `+12.6%`
- `PF 2.083`
- `MaxDD 4.87%`
- `31` basket trades

Interpretation:

- this is the first non-trivial fast strategy here that actually cleared `PF > 1` with low DD
- but it is sparse and regime-specific, not a broad “every month” machine

### 1b. Cross-sectional momentum v2

Enhancements:

- inverse-volatility weights inside each side of the basket
- funding-aware score tilt

Best v2 result:

- lookback: `24h`
- hold: `8h`
- `top_n = 4`
- minimum absolute momentum: `2%`
- funding tilt: `150`
- result: `+18.07%`
- `PF 2.59`
- `MaxDD 5.05%`
- `31` basket trades

Interpretation:

- the edge improved materially after portfolio-aware sizing
- this is now clearly the strongest candidate in the whole new research branch
- it is still sparse, so the next task is improving continuity without killing PF

More stable broad version:

- lookback: `24h`
- hold: `24h`
- `top_n = 2`
- minimum absolute momentum: `2%`
- result: `+6.9%`
- `PF 1.067`
- `MaxDD 18.96%`
- profitable months: `12 / 24`

Interpretation:

- lower edge, but more continuous
- this is the most realistic candidate for the next serious development cycle

### 2. Funding carry approximation

Best approximation:

- entry threshold: `0.012% / 8h`
- exit threshold: `0.0%`
- max positions: `3`
- roundtrip fee assumption: `0.10%`
- result: `+3.52%`
- `MaxDD 0.0719%`

Interpretation:

- the funding stream is real
- it is extremely smooth
- but it is too slow and too small to be the main recovery engine

## Objective conclusion

The next strategy track should be:

- primary: `cross-sectional long/short momentum v2`
- secondary sleeve: `funding carry`

I would **not** spend the next cycle on another fast directional single-name bot.

## Recommended next action

1. Rebuild the full `5m` cache for the symbols we care about most first:

`BTC ETH SOL XRP DOGE ADA AVAX LINK TON NEAR INJ`

2. Keep developing `xsec momentum` as the main candidate:

- add turnover controls
- add volatility scaling
- add symbol clustering so the basket is less correlated
- add funding-aware weighting instead of pure equal weights

3. Keep `funding carry` as a small low-DD sleeve, not the core strategy.

4. Only after that start a true carry-arb upgrade:

- fetch historical spot OHLCV
- fetch spot/perp basis
- estimate borrow and maker/taker execution properly

## What I would do next in code

If we continue immediately, the highest-value next implementation is:

- a `v3` portfolio strategy that takes `xsec momentum v2` and adds
  turnover controls, correlation caps, and a live-paper execution wrapper

That is the cleanest path from “research edge” to something you could actually consider for paper/live.
