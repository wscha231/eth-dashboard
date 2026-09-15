# EtherForecast continuous DATA refresh implementation plan

Date: 2026-09-16
Status: approved by user instruction to proceed; implementation follows on this branch.

## 1. Registry

Create `config/data_source_registry.json` as the machine-readable source contract. Each source records:

- source id and family
- current collection state (`active`, `planned`, `blocked`)
- hot-cache file path and timestamp column
- collection cadence and workflow
- soft/hard staleness SLA
- point-in-time status (`receipt`, `reconstructed`, `revised_history`, `partial_pit`)
- research role and production eligibility
- durable storage stream

The registry is operational metadata only; it does not assert predictive value.

## 2. Fast derivative snapshots

Create `scripts/collect_fast_derivative_snapshots.py` and `.github/workflows/fast_derivative_snapshots.yml`.

Every four hours:
1. restore `data/daily-forecast` vendor caches;
2. fetch current Bitget ETH perpetual context;
3. fetch current Hyperliquid ETH perpetual context;
4. re-index the observation to the actual UTC receipt hour instead of the calendar day;
5. append to venue-specific 4h snapshot CSVs without synthesizing old history;
6. persist to `data/daily-forecast`;
7. upload a short-lived GitHub artifact that the Drive archive workflow consumes.

This creates useful prospective OI/current-context history for the 6h/1d research horizons without repeatedly redownloading multi-year backfills.

## 3. Full research refresh

Change `free_source_backfill.yml` from once daily to twice daily (01:10 and 13:10 UTC). It remains responsible for long/release-driven sources: Bitget basis/funding, Deribit DVOL, Hyperliquid funding history, FRED/ALFRED and related daily research caches.

Do not run this multi-year backfill every four hours.

## 4. Freshness guard

Create `scripts/data_refresh_status.py`, tests, and `.github/workflows/data_refresh_guard.yml`.

Every six hours the guard:
- restores current data branch caches;
- reads `config/data_source_registry.json`;
- finds the latest real observation/receipt for each active source;
- classifies `ok`, `warning`, `stale`, `missing`, or `planned`;
- writes `lake/reports/data_refresh_status.json` and a small CSV summary;
- persists the report on `data/daily-forecast`;
- fails the workflow only when a source marked `critical=true` exceeds its hard SLA.

Warnings remain visible without stopping all research collection.

## 5. Drive durability

Extend the existing Drive archive:

- add `research-source-data` to allowed streams;
- archive `free-source-fallback-state` and `fast-derivative-state` artifacts to that stream;
- trigger Drive archival directly after `Free ETH research source backfill` and `Fast ETH derivative snapshots` complete;
- keep the six-hour branch sweep as a second line of defense.

No existing Drive snapshots are deleted or overwritten.

## 6. Reconciliation layers

The cadence registry defines:

- daily refresh: ordinary incremental collection;
- weekly reconciliation: overlap/gap audit, stale-source review and recent-history refetch;
- monthly deep audit: complete coverage/manifest/hash/PIT/licensing review and restore drill.

The first implementation creates the schedule metadata and freshness reporting. Existing full-history collectors already provide reconciliation capability; follow-up sources can be added to the same registry without changing the contract.

## 7. Tests and promotion boundaries

Tests cover:

- registry schema and unique source ids;
- timestamp parsing and staleness classification;
- planned sources not causing failures;
- missing critical sources causing a hard failure;
- fast snapshots preserving receipt-hour timestamps;
- Drive stream allowlist and artifact mapping.

No forecast model, model champion, public prediction, trading behavior or site output is promoted by this change.
