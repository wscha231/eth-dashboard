# R3 P0-2 — ETH supply / issuance / burn source contract v1

Status: **research only; not model-facing, not site-facing, no public redistribution**  
Frozen: 2026-09-17

## Objective

Start a durable point-in-time receipt history for ETH supply dynamics before any price-alpha hypothesis is tested.

This P0-2 gate captures only values that EtherForecast actually receives prospectively. It does **not** create a retrospective history for backtesting.

## Protocol definition

Ethereum supply changes are modeled as:

`next supply = previous supply + issuance - execution fee burn - consensus penalties - other proven destruction`

Transfers, deposits and withdrawals move ETH but do not themselves create or destroy total ETH.

Primary protocol references:

- `https://ethereum.org/eth/supply`
- `https://ethereum.org/roadmap/merge/issuance/`
- `https://geth.ethereum.org/docs/developers/evm-tracing/live-tracing`

Geth's supply tracer plus canonical consensus history remains the required long-run self-derived backfill path. Any retrospective alpha claim requires that separate gate and explicit reorg handling.

## First prospective research source

Provider: `ethsupply.fyi`.

Methodology: `https://ethsupply.fyi/methodology/`.

Operational P0-2 uses two public contracts only:

- `GET https://ethsupply.fyi/api/tracked` — retained cumulative issuance/burn/net accounting through the live tail;
- `GET https://ethsupply.fyi/api/live` — current total/finalized supply and finalized-epoch accounting.

The provider documents exact base-10 wei strings and successful-response headers `x-supply-sha256`, `x-supply-generated-at`, and `x-supply-revision`.

## Why historical chart endpoints are excluded from P0-2

Two live gates were intentionally allowed to fail rather than relaxing the contract:

1. `history?range=retained` did not expose detailed epoch observations to the collector;
2. switching to `history?range=30d` still produced no usable epoch observations at collection time, despite the published schema allowing them.

EtherForecast therefore does not infer, synthesize, or substitute a historical interval series. P0-2 is narrowed to prospective `/tracked` and `/live` receipts. Historical supply/issuance/burn becomes eligible only after the self-derived Geth/consensus replay is implemented and audited.

This decision was made before any supply-based model experiment.

## Rights boundary

A public API is not automatically a commercial-data license. No sufficiently explicit commercial redistribution right was identified when this contract was frozen.

Therefore:

- `license_status=rights_unreviewed_public_api`;
- raw bytes and normalized rows remain in access-controlled research artifacts / private Drive;
- no public Git data-branch copy;
- no website/API redistribution;
- no production MODEL feature;
- no paid-product dependency.

Coin Metrics Community is deliberately **not** persisted as the primary source because its free Community data is documented for non-commercial use.

## Point-in-time contract

There is no inferred historical `available_at` in this P0-2 collector.

Every successful receipt stores:

- `received_at = actual EtherForecast collector time`;
- `available_at = received_at`;
- source revision;
- source generation time;
- exact response SHA-256;
- normalized exact wei fields and human-readable ETH values.

`collector_started_at` is immutable once created.

### `/api/tracked` receipts

Each receipt preserves:

- retained from/to slot and UTC timestamps;
- cumulative issuance;
- cumulative burn and published burn components;
- cumulative net issuance;
- warnings/finalized-tail metadata;
- native accounting identity result.

Successive cumulative receipts can later be differenced using only information that was genuinely available at each receipt time.

### `/api/live` receipts

Each receipt preserves:

- live head and finalized position;
- total/finalized supply;
- supply `asOf` position;
- finalized-epoch issuance;
- burn;
- net issuance;
- native availability/kind/as-of metadata;
- native accounting identity result.

## Accounting gates

For `/tracked` and `/live`, where all native components are available:

`netWei == issuanceWei - burnWei`

The collector records `identity_error_wei`. A non-zero value is preserved as evidence and fails the workflow gate; it is never silently corrected.

## Evidence preservation

Each run retains:

- append-only tracked receipts (`jsonl` plus inspection CSV);
- append-only live receipts (`jsonl` plus inspection CSV);
- raw tracked response;
- raw live response;
- collector state;
- machine-readable rights/PIT/accounting report.

The collector intentionally uses only Python's standard library so the 6-hour scheduled job does not install model-training dependencies.

GitHub Actions carries state forward between runs. The existing content-addressed Google Drive archive is extended with dedicated `eth-supply-research` and `eth-etf-research` streams so successful main-branch receipts survive disposable Actions retention.

## Promotion boundary

MODEL/site use remains blocked until all of the following are complete:

1. commercial/intended-use rights review or replacement with a self-derived chain source;
2. sufficient prospective receipt continuity;
3. zero unexplained tracked/live accounting-identity failures;
4. independent chain-position cross-checks;
5. self-derived historical PIT backfill for any retrospective model claim;
6. preregistered supply hypothesis with a frozen MODEL gate.

No paid API is authorized by this contract.
