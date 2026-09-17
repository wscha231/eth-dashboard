# R3 P0-1 — ETH ETF flow source contract v1

Status: **research only; not model-facing, not site-facing, no public redistribution**  
Frozen: 2026-09-17

## Objective

Create a durable point-in-time research history for U.S. spot Ethereum ETF daily net flows before any price-alpha hypothesis is tested.

A successful collection run proves only that the data pipeline works. It is not evidence that ETF flows predict ETH returns.

## Primary research source

- Provider: Farside Investors.
- Public page: `https://farside.co.uk/ethereum-etf-flow-all-data/`.
- Native unit: US$ millions.
- Grain: U.S. trading date, per fund plus reported total.
- Fund tickers are discovered from the source table instead of freezing a hard-coded fund list.
- The provider describes the table as automatically generated and updated in real time.
- Rights status: `unreviewed_no_public_redistribution`. The page states all rights are reserved. Until a rights review establishes permitted downstream use, raw bytes and normalized rows remain only in access-controlled research storage and short-lived Actions artifacts.

## Point-in-time contract

Every normalized row carries:

`date, event_time, published_at, publication_time_status, available_at, ingested_at, revision, kind, vintage_mode, license_status, availability_mode, source_id, source_url, unit, raw_sha256`

Flow columns are dynamic `flow_<ticker>_usd_m`, plus `total_usd_m`.

### Historical backfill

The first collection is explicitly `vintage_mode=reconstructed`.

Farside does not provide the original publication timestamp for each historical row. We do not fabricate one:

- `published_at` stays empty;
- `publication_time_status=not_provided_by_source`;
- research backtests may use a conservative source-availability proxy of **trade date + 1 day 12:00 UTC**;
- reconstructed history remains blocked from prospective/live certification even if that proxy precedes the actual September 2026 ingestion time.

### Prospective receipts

After collector start is frozen:

- a newly observed trading-date row at/after the collector start date is `vintage_mode=observed_vintages`;
- `available_at = ingested_at = actual collector receipt time`;
- no earlier availability is inferred.

### Revisions

When a previously stored row changes:

- the row revision increments;
- the new value becomes available only at that revision receipt time;
- the prior version remains in the append-only receipt journal;
- unchanged rows retain their original receipt/hash metadata.

This prevents later edits from leaking backward in time.

## Evidence preservation

Each run retains:

- normalized latest table;
- append-only receipt/revision journal;
- collector state including first start time;
- raw source response for that run;
- SHA256 of the raw response;
- machine-readable coverage/PIT/rights report.

Research state is carried forward through the previous successful Actions artifact. It must not be copied to the public Git data branch while rights remain unreviewed. After a successful main-branch run, the artifact can be preserved in the private EtherForecast Google Drive archive.

## Initial quality gates

Collection fails closed if:

- the Ethereum ETF ticker table cannot be identified;
- duplicate ticker headers occur;
- a dated row is structurally incomplete;
- no dated rows are found;
- the reported total column has no numeric observations.

The report records first/last date, row count, non-null total count, new rows, revisions, unchanged rows and raw SHA256.

## Promotion boundary

The source remains blocked from MODEL and Website use until all of the following are complete:

1. rights/intended-use review;
2. coverage and continuity report;
3. point-in-time audit of prospective receipts;
4. duplicate/revision behavior review;
5. official issuer and/or SEC cross-check snapshots where practical;
6. preregistered hypothesis and frozen MODEL gate.

No paid API is authorized by this contract.
