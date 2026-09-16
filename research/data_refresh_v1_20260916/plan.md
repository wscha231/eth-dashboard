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
7. upload a short-lived GitHub artifact and trigger the existing Drive archive workflow.

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

Reuse the already tested `daily-data` Drive stream rather than introducing a second overlapping source-data stream.

- `data/daily-forecast` remains the hot source-data branch and is already mapped to `daily-data` by `archive_to_drive.py`.
- Add workflow-run triggers after `Free ETH research source backfill`, `Fast ETH derivative snapshots`, and the freshness guard so the updated branch is archived promptly.
- Keep the independent six-hour Drive sweep as a second line of defense.
- Keep short-lived workflow artifacts as execution receipts; durable recovery remains the content-addressed branch snapshot in Drive.

No existing Drive snapshots are deleted or overwritten.

## 6. Reconciliation layers

The cadence registry defines:

- ordinary incremental/full refresh throughout each day;
- weekly Sunday review of overlap/gaps, stale-source status and acquisition priorities;
- monthly deep review of coverage, manifests/hashes, PIT/revision semantics, licensing and restore integrity.

The automated collectors and six-hour freshness report provide the evidence for these maintenance reviews. Follow-up sources are added to the same registry rather than getting ad-hoc schedules or storage layouts.

## 7. Tests and promotion boundaries

Tests cover:

- registry schema and unique source ids;
- timestamp parsing and staleness classification;
- planned sources not causing failures;
- missing critical sources causing a hard failure;
- fast snapshots preserving receipt-hour timestamps;
- the existing Drive storage self-tests after archive-trigger changes.

No forecast model, model champion, public prediction, trading behavior or site output is promoted by this change.
