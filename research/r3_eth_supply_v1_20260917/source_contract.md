# R3 P0-2 — ETH supply / issuance / burn source contract v1.1

Status: **research only; not model-facing, not site-facing, no public redistribution**  
Frozen: 2026-09-17  
Accounting semantics correction: 2026-09-17, before any supply-based model experiment

## Objective

Start a durable point-in-time receipt history for ETH supply dynamics before any price-alpha hypothesis is tested.

This P0-2 gate captures only values that EtherForecast actually receives prospectively. It does **not** create a retrospective history for backtesting.

## Protocol definition

The provider methodology defines total ETH supply changes as:

`next supply = previous supply + issuance - execution fee burn - consensus penalties - other proven destruction`

Transfers, deposits and withdrawals move ETH but do not themselves create or destroy total ETH.

Primary references:

- `https://ethsupply.fyi/methodology/`
- `https://ethereum.org/eth/supply`
- `https://ethereum.org/roadmap/merge/issuance/`
- `https://geth.ethereum.org/docs/developers/evm-tracing/live-tracing`

Geth's supply tracer plus canonical consensus history remains the required long-run self-derived backfill path. Any retrospective alpha claim requires that separate gate and explicit reorg handling.

## First prospective research source

Provider: `ethsupply.fyi`.

Operational P0-2 uses two public contracts only:

- `GET https://ethsupply.fyi/api/tracked` — retained cumulative accounting through the live tail;
- `GET https://ethsupply.fyi/api/live` — current total/finalized supply and finalized-epoch accounting.

The provider documents exact base-10 wei strings and successful-response headers `x-supply-sha256`, `x-supply-generated-at`, and `x-supply-revision`.

## Accounting semantics — fail closed

The first live receipt exposed a semantic error in EtherForecast's original verification equation. The raw provider values were preserved and the workflow correctly failed; no values were altered to make the gate pass.

### `/api/tracked`

`HistoricalSummary.burnWei` is execution fee burn. Consensus penalties and other execution destruction are published separately.

The verified equation is therefore:

`netWei = issuanceWei - burnWei - consensusPenaltiesWei - otherExecutionBurnWei`

A second component gate requires:

`burnWei = baseFeeBurnWei + blobBaseFeeBurnWei`

The first failed receipt itself demonstrated why the original `net = issuance - burnWei` equation was invalid: the difference was exactly the separately published consensus penalties plus other execution destruction.

### `/api/live`

`accounting.burn.totalWei` is execution-side destruction represented by its execution components. `accounting.burn.consensus.totalWei` is separate.

The verified equation is:

`accounting.netWei = accounting.issuance.totalWei - accounting.burn.totalWei - accounting.burn.consensus.totalWei`

A second component gate requires:

`burn.totalWei = burn.baseFeeWei + burn.blobBaseFeeWei + burn.otherExecutionWei`

The first failed live receipt differed from the original simple equation by exactly `burn.consensus.totalWei`, confirming the published component separation.

These are semantic corrections to the source contract, not relaxed tolerances. All comparisons remain exact integer-wei identities.

## Why historical chart endpoints are excluded from P0-2

Two live gates were intentionally allowed to fail rather than relaxing the history contract:

1. `history?range=retained` did not expose detailed epoch observations to the collector;
2. switching to `history?range=30d` still produced no usable epoch observations at collection time, despite the published schema allowing them.

EtherForecast therefore does not infer, synthesize, or substitute a historical interval series. P0-2 is narrowed to prospective `/tracked` and `/live` receipts. Historical supply/issuance/burn becomes eligible only after the self-derived Geth/consensus replay is implemented and audited.

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
- normalized exact wei fields and human-readable ETH values;
- the exact accounting equation used for that source contract version;
- exact integer component and identity errors.

`collector_started_at` is immutable once created.

### `/api/tracked` receipts

Each receipt preserves retained from/to slot and UTC timestamps, cumulative issuance, execution burn and components, consensus penalties, other execution destruction, cumulative net issuance, warnings/finalized-tail metadata, component reconciliation, and protocol accounting identity.

Successive cumulative receipts can later be differenced using only information genuinely available at each receipt time.

### `/api/live` receipts

Each receipt preserves live head and finalized position, total/finalized supply, supply `asOf` position, finalized-epoch issuance, execution burn, consensus penalties, net issuance, native availability/kind/as-of metadata, component reconciliation, and protocol accounting identity.

## Evidence preservation

Each run retains:

- append-only tracked receipts (`jsonl` plus inspection CSV);
- append-only live receipts (`jsonl` plus inspection CSV);
- raw tracked response;
- raw live response;
- collector state;
- machine-readable rights/PIT/accounting report.

The collector intentionally uses only Python's standard library so the 6-hour scheduled job does not install model-training dependencies.

GitHub Actions carries state forward between successful runs. The existing content-addressed Google Drive archive preserves successful main-branch receipts so they survive disposable Actions retention.

## Promotion boundary

MODEL/site use remains blocked until all of the following are complete:

1. commercial/intended-use rights review or replacement with a self-derived chain source;
2. sufficient prospective receipt continuity;
3. zero unexplained accounting-component or protocol-identity failures;
4. independent chain-position cross-checks;
5. self-derived historical PIT backfill for any retrospective model claim;
6. preregistered supply hypothesis with a frozen MODEL gate.

No paid API is authorized by this contract.
