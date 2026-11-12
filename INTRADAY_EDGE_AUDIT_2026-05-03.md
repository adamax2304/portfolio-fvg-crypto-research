# Intraday Edge Audit — 2026-05-03

## Scope

Goal: find low-timeframe crypto edges that could plausibly beat the current swing system on drawdown and monthly smoothness.

I tested three directional intraday families on the local 2y cache with one consistent execution model:

- next-bar entries only
- taker fees on entry and exit
- slippage on entry and exit
- actual cached funding history per symbol
- portfolio-level concurrency cap

Research runner:

- [backtest_intraday_edges.py](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/backtest_intraday_edges.py:1)
- [intraday_edges_20260503_231624.json](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/results/intraday_edges_20260503_231624.json:1)

## Critical data finding

The local `5m_730d` cache is mislabeled.

Example:

- [BTC_USDT_USDT_5m_730d.parquet](/C:/Users/adama/Documents/zeph%20ai%20projects/fvgbot/multi_agent/backtests/data_cache/BTC_USDT_USDT_5m_730d.parquet) covers `2024-04-17 23:05 UTC` to `2025-03-30 20:05 UTC`, which is about `347` days, not `730`.
- `15m`, `1h`, `1d`, and `funding_730d` caches do extend close to the expected 2-year window.

Implication:

- any old "2-year" results that depended on `5m_730d` files are not actually 2-year tests
- fast 5m strategy conclusions must be treated as partial-period only unless the missing year is refetched

## Directional intraday results

### 1. `VWAP_PULLBACK_15M`

- trades: `12,199`
- WR: `40.69%`
- PF: `0.955`
- return: `-99.71%`
- max DD: `99.74%`
- profitable months: `6 / 25`

Verdict:

- no edge after realistic fees
- the raw idea overtrades and gets fee-dragged to death

### 2. `FUNDING_FADE_15M`

- trades: `769`
- WR: `38.62%`
- PF: `0.731`
- return: `-75.37%`
- max DD: `76.46%`
- profitable months: `4 / 24`

Verdict:

- funding extremes alone did not translate into a tradable short-horizon fade on this cache
- interesting as research, not good enough for live

### 3. `BBS_1H_5M_PULLBACK`

- trades: `174`
- WR: `28.74%`
- PF: `0.762`
- return: `-24.69%`
- max DD: `33.35%`
- profitable months: `5 / 25`

Important caveat:

- this uses the mislabeled `5m_730d` cache, so its effective test window ends in `2025-03`

Verdict:

- not live-worthy from the honest run

## Best near-flat directional follow-up

I separately re-checked a cleaner `1h signal -> 15m next-bar` squeeze version with full 15m coverage.

Follow-up result:

- strategy: `BBS_1H_15M_NEXTBAR`
- trades: `703`
- WR: `34.71%`
- PF: `0.963`
- return: `-18.98%`
- max DD: `48.03%`
- profitable months: `13 / 25`

What this means:

- among the fast directional ideas, this was the least bad
- but it still does not clear the line for a new live deployment
- I also ran a small parameter grid around squeeze length, SL ATR, TP R, daily filter, and funding filter; nothing crossed `PF > 1`

## Market-neutral follow-ups

### 4. `XSEC_MOMENTUM_LS_1H`

Prototype:

- rebalance every `24h`
- long top `3` liquid symbols by `24h` return
- short bottom `3`
- equal-dollar long/short basket
- actual funding included

Best simple variant I found:

- lookback: `24h`
- hold: `24h`
- threshold: `2%`
- trades: `38`
- WR: `39.47%`
- PF: `0.99`
- return: `-0.33%`
- max DD: `9.97%`

Verdict:

- this is much healthier than the directional intraday bots
- still not proven profitable
- but it is one of the few fast ideas here that stayed near flat without blowing up

### 5. `POSITIVE_FUNDING_CARRY_ARB`

Prototype:

- short perp / long spot
- only when funding is strongly positive
- equal-cap allocation across top-funded symbols
- market-neutral by construction

Best simple variant I found:

- entry threshold: `0.012% / 8h`
- exit threshold: `0.0%`
- max positions: `3`
- roundtrip fee assumption: `0.10%`
- capital usage per pair: `1.2x` notional
- return: `+3.52%` over the full sample
- max DD: `0.04%`
- profitable months: `6 / 25`
- avg month: about `+0.14%`

Verdict:

- this is the only clearly structural edge I found
- but it is too slow and too small to quickly recover directional losses
- it is infrastructure-heavy and sensitive to fee tier, maker quality, and spot leg availability

## Objective conclusion

If the target is:

- low drawdown
- many profitable months
- fast crypto execution
- suitability for a small deposit

then the current local evidence does **not** support launching a new fast directional bot yet.

What survived the audit:

- `funding carry` is structurally real, but too small for your stated goal unless you have proper spot/perp infra and cheap execution
- `cross-sectional long/short momentum` is the most promising research path among the fast strategies, because it was almost flat with much lower DD than the directional bots

What failed:

- raw intraday VWAP continuation
- raw funding-extreme fade
- honest squeeze-based directional intraday execution

## What I would do next

1. Do **not** replace the current live system with a new fast directional bot on this evidence.
2. Build the next research batch around market-neutral or semi-neutral ideas, not pure direction.
3. Collect missing data before touching orderbook-style strategies:
   - full 2-year `5m`
   - spot history for the same symbols
   - spot-perp basis snapshots
   - L2 orderbook history
   - liquidations and open interest history
4. After that, retest only these three families:
   - spot/perp funding carry with real basis and fee tiers
   - cross-sectional long/short momentum with volume weighting
   - orderbook imbalance / liquidity-taker or maker model on real L2 history

## External research used

- [Bybit funding history API](https://bybit-exchange.github.io/docs/v5/market/history-fund-rate)
- [Bybit orderbook WebSocket docs](https://bybit-exchange.github.io/docs/v5/websocket/public/orderbook)
- [Bybit post-only order docs](https://www.bybit.com/en/help-center/article/Post-Only-Order)
- [Fundamentals of Perpetual Futures](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4301150)
- [Predictability of Funding Rates](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5576424)
- [Cryptocurrency Volume-Weighted Time Series Momentum](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4825389)
- [Explainable Patterns in Cryptocurrency Microstructure](https://arxiv.org/abs/2602.00776)
