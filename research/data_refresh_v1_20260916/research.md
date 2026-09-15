# EtherForecast continuous DATA refresh research

Date: 2026-09-16

## Goal

Replace one-off backfills with a durable collection program whose cadence matches the information speed of each source. Preserve point-in-time semantics, accumulate sources that have no free historical backfill, keep research data separate from production promotion, and archive every durable checkpoint to the existing Google Drive store.

## Current verified state

- Hourly ETH event forecasting already runs every hour from `event_hourly.yml`; this remains the fastest operational market path.
- `daily_forecast.yml` currently runs twice daily and refreshes the daily master plus vendor caches.
- `free_source_backfill.yml` currently runs once daily at 01:10 UTC and stores Bitget, Deribit DVOL, Hyperliquid, and FRED/ALFRED research caches on `data/daily-forecast`.
- `gdrive_archive.yml` sweeps every six hours and archives `data/daily-forecast`, but it does not currently trigger directly from `Free ETH research source backfill` and it does not classify its artifact as a dedicated research-source stream.
- Fast-changing OI/current-context snapshots are currently only daily. That is too sparse for 6h/1d research and wastes the prospective-history opportunity because historical OI is not freely backfillable from these sources.
- Current long-history backfills are deliberately research-only. Revised FRED history must not be treated as historical PIT data; ALFRED/as-of state remains the preferred macro research input when available.

## Source-speed classes

### Class A: hourly operational market data
ETH/BTC closed bars and the active event ledger. Keep the existing hourly job. Do not duplicate it in the DATA refresh jobs.

### Class B: fast derivatives/current context
Bitget and Hyperliquid OI/current funding/premium/mark context. These change intraday and have weak/no free historical OI backfill. Accumulate a receipt-timestamped snapshot every 4 hours. Preserve venue, contract and native units. Never synthesize the missing past.

### Class C: daily historical research sources
Bitget basis/funding, Deribit DVOL, Hyperliquid historical funding/premium, DefiLlama TVL/DEX/stablecoins, CoinGecko/Fear&Greed and other daily caches. Refresh daily with overlap; periodically reconcile longer history.

### Class D: macro/release-driven data
FRED current-vintage and ALFRED/PIT series. Poll twice daily so US releases are captured without forcing a minute-level collector. Store observed/release/ingest semantics separately. A release calendar can later tighten this further.

### Class E: snapshot-only options/on-chain candidates
Deribit options/futures, exchange flows, staking/validator/blob/fee data and liquidation streams where historical free access is insufficient. Accumulate prospectively on the fastest affordable cadence that matches their role; do not backfill with estimates.

## Cadence policy

- Hourly market/event inputs: every 1 hour (existing system).
- Fast derivative snapshots: every 4 hours.
- Full free research refresh: twice daily, 01:10 and 13:10 UTC.
- Daily gold/master refresh: existing 00:37 and 02:37 UTC jobs remain; they consume validated daily sources.
- Freshness guard: every 6 hours. It produces machine-readable status and fails only when a critical collected source exceeds its hard SLA.
- Drive archive: after research-source workflows and on the existing six-hour sweep.
- Weekly reconciliation: Sunday full gap/coverage audit over recent history and source-status review.
- Monthly deep reconciliation: rerun long-history coverage, hashes, PIT/revision checks, licensing metadata and storage-restore drill. This is research maintenance, not automatic model promotion.

## Freshness/SLA principles

A source has three independent states: collection health, learning readiness and production eligibility. Fresh data does not imply predictive value. A source may be healthy but research-only, or stale but non-critical. Each registry entry records cadence, soft and hard staleness limits, timing status, collection state and intended role.

For daily aggregates, allow roughly 2x cadence before warning and 3x before hard stale. Fast derivative snapshots warn after 8h and hard-fail after 16h. Macro PIT data is allowed wider gaps because many series are weekly/monthly; the collector itself must still run on schedule and report per-series freshness separately.

## Drive/storage

Use `data/daily-forecast` as the hot research cache. Add a dedicated `research-source-data` Drive stream for source artifacts while retaining the existing `daily-data` branch archive. Every new snapshot is content-addressed and immutable in Drive; no manual deletion of shared `ef1-blob-*` objects.

## Risks

1. More frequent snapshots can increase Git history. Keep fast files compact and rotate only via immutable Drive checkpoints if they become large.
2. API limits/geo restrictions can break one provider. Keep per-source errors isolated so another source can still update.
3. A source timestamp is not automatically its availability timestamp. Backfills remain reconstructed research unless receipt/vintage evidence exists.
4. Continuous collection creates more data but not necessarily better forecasts. MODEL must run matched-origin ablation before promotion.
