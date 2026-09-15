# Free-source D003/D004 acquisition research — 2026-09-15

Scope: additive DATA work after DATA PR #54 and MODEL PR #53 were merged. No model promotion or paid API.

## Current gap

The existing daily collector attempts Binance funding/OI/basis/taker data and Deribit funding/snapshots, but several Binance requests have failed by region/network and short-retention endpoints do not provide a durable multi-year history. FRED is currently read through fredgraph CSV as today's revised history, which is not sufficient for point-in-time macro backtests.

## Free source decisions

### Bybit public V5
- Funding history: `/v5/market/funding/history`, max 200 rows per request.
- Open interest: `/v5/market/open-interest`, cursor pagination; Bybit documents query availability back to the symbol launch time. For linear contracts OI unit is the base coin, so ETHUSDT OI is recorded as ETH, not silently converted to USD.
- Premium index: `/v5/market/premium-index-price-kline`, up to 1000 candles per request. Used as a free perpetual-premium/basis proxy, not labeled as futures calendar basis.

The collector stores raw daily aggregates and deterministic past-only features. Completed day D becomes eligible at D+1 00:00 UTC for conservative hourly joining.

Official docs:
- https://bybit-exchange.github.io/docs/v5/market/history-fund-rate
- https://bybit-exchange.github.io/docs/v5/market/open-interest
- https://bybit-exchange.github.io/docs/v5/market/premium-index-kline

### FRED / ALFRED
When `FRED_API_KEY` is configured, use `fred/series/observations` with `output_type=4`, which FRED documents as "Observations, Initial Release Only". Keep observation date separate from `realtime_start`. Because only a date is retained here, initial release date D is made eligible at D+1 00:00 UTC rather than assuming an intraday release time.

Initial series: M2SL, CPIAUCSL, UNRATE, WALCL, WTREGEN, RRPONTSYD, SOFR, DFF, T10Y2Y, BAMLH0A0HYM2.

Official docs:
- https://fred.stlouisfed.org/docs/api/fred/series_observations.html
- https://fred.stlouisfed.org/docs/api/fred/realtime_period.html

### DeFiLlama
Reuse the repository's existing historical Ethereum stablecoin, chain TVL and DEX-volume collectors but force an initial 2017 start during free-source backfill. The existing daily collector then owns incremental refresh.

## Persistence

`daily_forecast.yml` already restores/persists all allowlisted CSVs under `lake/raw/vendor`, commits them to `data/daily-forecast`, and the daily Drive snapshot archives those paths. Therefore the new source collector writes to the existing vendor cache rather than introducing a second storage system.

## Safety

- No synthetic OI history.
- No reverse fill or future interpolation.
- No USD conversion for ETH-denominated OI unless an explicit price/time conversion is added later.
- Source/API availability is not evidence of predictive value.
- Historical API backfill remains reconstructed research unless original receipt/vintage evidence exists.
- Source terms for commercial/public redistribution remain unreviewed; public site exposure is not authorized by this work.
