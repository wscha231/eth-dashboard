# Historical input reinforcement — 2026-09-07

The main branch's short derivative history is not evidence that longer history
does not exist. This change supplies bounded research collectors and OHLCV
proxies, with completeness and receipt-time checks. It does not select a new
forecast model or certify any improvement in predictive performance.

## Findings

- A live Deribit request for 2020-01-01 through 2020-04-01 returned only 744
  hourly records, from 2020-03-01 01:00 to 2020-04-01 00:00 UTC. A successful
  HTTP response is not a successful full backfill. The new funding collector
  requests intervals of at most 28 days and checks every expected hour.
- The completed bounded collection recovered 62,254 of 62,256 requested hourly
  funding intervals from 2019-08-01 01:00 through 2026-09-07 00:00 UTC (99.9968%
  coverage). The missing timestamps are 2020-08-27 06:00 and 07:00 UTC. A narrow
  re-query also returned no records. They remain missing; the strict report is
  `complete=false`. Daily aggregation yields 2,593 complete days and excludes
  the incomplete day. Reindex on the UTC grid before building hourly features.
- ETH DVOL was retrieved for 2021-04-01 through 2026-09-06: 1,985 complete daily
  candles with no missing day. Daily closing values become eligible only after
  the candle ends. They are implied volatility, not a full options surface.
- Existing PR #10, commit `cd031be8b61cf866523c406d88c8bb16d9cb50b3`, already
  contains Ethereum TVL (3,260 days from 2017-09-27), stablecoin data (3,197 days
  from 2017-11-29), and DEX volume (2,859 days from 2018-11-02), through
  2026-08-30. The stored SHA256 values and daily continuity were rechecked.
  These histories are outside the current event-model input path. The earlier
  operating-data audit did not account for this unmerged research branch.
- The existing daily ETH price dataset supports 3,198 complete rows of five
  exploratory proxies after a 24-day warmup. These are daily features, not
  hourly reconstructions. The price-impact calculation on that dataset assumes
  quote-currency volume and requires source-unit confirmation before adoption.
- No missing OI or liquidation observations have been reconstructed as truth.

## Reproduce

Install the repository's `requirements-events.txt` and pytest. Keep downloaded
data outside tracked/public files unless redistribution permission is known.

```sh
python scripts/backfill_event_funding.py --root /path/to/private/funding --start 2019-08-01 --end 2026-09-07 --budget-seconds 600 --max-requests 120
python -m pytest tests/test_funding_history.py tests/test_history_proxies.py -q
```

The end is exclusive for daily source dates and inclusive for hourly funding
interval ends. The first example includes funding intervals through the end of
2026-09-06. Run the same command to resume: verified cached windows retain their
original receipt times. Partial responses are not certified or cached as
complete. HTTP 401/403/451/429 stops further queued network requests; already
running requests are bounded. No alternate host is selected.

For daily implied volatility:

```python
from research_pipeline.volatility_history import backfill_dvol
frame, report = backfill_dvol(
    '/path/to/private/dvol', start='2021-04-01', end='2026-09-07')
assert report['complete']
```

For optional observed-price features:

```python
from research_pipeline.price_proxies import build_price_proxies
# One instrument, complete OHLCV, DatetimeIndex at candle close.
# Coinbase ETH-USD volume is native ETH; declare units for every source.
features = build_price_proxies(bars, volume_unit='native', frequency='h', window=24)
```

The features are downside variance share, volume-weighted closing location,
price-impact proxy, trend efficiency, and realized volatility. None measures
OI, liquidation flow, aggressive buy/sell volume or implied volatility.
Missing observations invalidate affected windows; no forward/backward filling
is performed. Dataset-wide normalization is not performed.

## Source strategy and adoption conditions

| Missing input | First choice | Alternative and limitation |
|---|---|---|
| Funding | Restore ETH-PERPETUAL hourly history using bounded API windows | Other venues are separate features: settlement intervals, quote/settlement currency and market composition differ |
| Expected volatility | ETH DVOL daily history, with its known candle end | Realized downside variance is a risk proxy, not reconstructed options expectations |
| OI / liquidations | A permitted full archive or prospective collection | Price-volume stress features do not recover historical position amounts |
| Stablecoins / DeFi | Reuse the already verified historical source where use is permitted | Preserve provider revisions and actual receipt time; aggregate TVL is not pure capital inflow |
| ETF flow | Data after the product's existence and actual release | Earlier rows are structurally unavailable, not zero flow; use the long-history base model then |
| Macro | Release-time/vintage data | Apply publication delays and vintage revisions; current revised data are not historical observations |

Compare a long-history base model with an optional recent-data model on the
same available dates, same origins and same event definition. A third candidate
can add price proxies, but changing the evaluation period must not count as
feature improvement. Gate model combinations by preceding validation data;
missing optional sources fall back to the base model. Additional hourly rows
do not create independent 30-day events. Synthetic training data never count
as observed evaluation outcomes.

Calibration/model-refit consistency, identified in the preceding audit, remains
a required separate fix before evaluating alert improvements. Data collection
alone does not fix the existing alert-threshold problem.

## Provenance and commercial use

Raw responses are hashed, saved with actual receipt times, and referenced in
per-run manifests. Backfilled data are explicitly historical reconstructions,
not proof of what the service received years ago. Do not join on event time
alone and label the result a prospective performance record.

Coin Metrics' community archive uses CC BY-NC 4.0. DefiLlama's current terms
cover its public APIs and require written consent for commercial exploitation
and redistribution. Public availability is not permission for incorporation
into a paid product. Deribit production use also needs the applicable data-use
terms checked. This PR contains code and audit statistics, not provider raw
datasets or a new production data contract.

The 14 focused tests passed locally, including truncated responses, cached
receipt-time preservation, corruption detection, access denial, daily funding
aggregation, future perturbations, missing bars, volume units, and DVOL paging.

Sources checked on 2026-09-07:

- [Deribit hourly funding API](https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history)
- [Deribit DVOL API and continuation](https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data)
- [Previously collected sources, PR #10](https://github.com/wscha231/eth-dashboard/pull/10)
- [Coin Metrics community archive license](https://github.com/coinmetrics/data/blob/master/README.md)
- [DefiLlama terms](https://defillama.com/terms)
- [OKX historical download scope](https://www.okx.com/en-us/historical-data)

The official OKX page lists funding archives from March 2022, but a complete
download and continuity audit was not performed here. Binance API access
restrictions were not bypassed. The GitHub artifact connector returned an
artifact reference, but its file download returned HTTP 403 in this session;
the full Coinbase hourly research artifact was therefore not used for a new
hourly proxy-performance experiment.
