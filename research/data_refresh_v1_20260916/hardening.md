# Continuous DATA refresh hardening — first live-run findings

Date: 2026-09-16

## Evidence from the first merged live run

PR #60 successfully enabled the new workflows and the first four-hour derivative collector completed end-to-end. It wrote receipt-timestamped Bitget and Hyperliquid context at 2026-09-16T00:00:00Z and pushed commit `20ce344` to `data/daily-forecast`.

The simultaneous push-triggered first runs exposed a repository-writer race:

- `Free ETH research source backfill` completed collection successfully through 2026-09-16 but its final push was rejected because the fast snapshot workflow had already advanced `data/daily-forecast`.
- `DATA refresh freshness guard` calculated a valid report (`status=ok`, no critical hard failures) but its final push was rejected for the same reason.
- This is a persistence-concurrency defect, not a source-collection failure.

The first freshness audit also proved the unregistered-file detector works. It found nine stored CSVs not yet represented explicitly in the registry:

1. coingecko_bnb_daily.csv
2. coingecko_btc_daily.csv
3. coingecko_defi_daily.csv
4. coingecko_global_daily.csv
5. coingecko_sol_daily.csv
6. deribit_eth_historical_volatility.csv
7. fred_current_vintage_long.csv
8. fred_daily.csv
9. fred_initial_release_long.csv

## Hardening decision

1. Put the three new workflows that write `data/daily-forecast` behind one repository-wide concurrency group (`data-daily-forecast-writer`) with `cancel-in-progress: false`. Scheduled execution remains staggered, so ordinary runs should rarely queue.
2. Register every currently detected source CSV so the registry is the explicit current inventory. Retain automatic `unregistered_files` discovery for future additions.
3. Re-run the failed long refresh and freshness audit after this change. Confirm both persist successfully after the existing fast-context commit.
4. Confirm Drive archive completion after the successful writer runs. The existing six-hour sweep remains fallback protection.

## Scope boundary

This hardening only fixes collection/persistence coordination and registry coverage. It does not promote any new feature to the forecasting model, change predictions, or infer missing historical OI/liquidation observations.
