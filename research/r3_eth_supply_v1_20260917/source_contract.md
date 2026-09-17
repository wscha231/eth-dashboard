# R3 P0-2 — ETH supply / issuance / burn source contract v1

Status: **research only; not model-facing, not site-facing, no public redistribution**  
Frozen: 2026-09-17

## Objective

Build a durable point-in-time receipt history for ETH supply dynamics before testing any price-alpha hypothesis.

The data family contains:

- total ETH supply;
- consensus issuance;
- total burn and its base-fee/blob/penalty components when published;
- net issuance;
- exact source revision/hash metadata.

Collection success is not evidence that supply dynamics predict ETH returns.

## Protocol definition

Ethereum supply changes are modeled as:

`next supply = previous supply + issuance - execution fee burn - consensus penalties - other proven destruction`

Transfers, execution/consensus deposits and withdrawals do not themselves create or destroy ETH.

Primary protocol references:

- `https://ethereum.org/eth/supply`
- `https://ethereum.org/roadmap/merge/issuance/`
- `https://geth.ethereum.org/docs/developers/evm-tracing/live-tracing`

Geth's supply tracer is the intended long-run self-derived source: it emits block-bound issuance and burn components and can be independently replayed. A historical self-derived backfill is a separate gate because it requires a trusted canonical execution/consensus history and explicit reorg handling.

## First prospective research source

- Provider: `ethsupply.fyi`.
- Methodology: `https://ethsupply.fyi/methodology/`.
- Current public contracts:
  - `GET https://ethsupply.fyi/api/live`
  - `GET https://ethsupply.fyi/api/history?range=retained`
- API responses expose source revision, generation time and SHA-256 headers and publish exact wei-denominated accounting fields.
- The methodology documents an independent execution-state / protocol-flow / consensus reconciliation and an exact supply equation.

### Rights boundary

A public API is not automatically a commercial-data license. No sufficiently explicit commercial redistribution right was identified when this contract was frozen.

Therefore:

- `license_status=rights_unreviewed_public_api`;
- raw bytes and normalized rows remain in access-controlled research artifacts / private Drive;
- no public Git data branch copy;
- no website/API redistribution;
- no production MODEL feature;
- no paid-product dependency.

Coin Metrics Community is deliberately **not** used as the primary stored source because its official API documentation says Community data is free for **non-commercial use**. It may be consulted manually for methodology or a future licensed cross-check, but is not silently incorporated into EtherForecast's planned commercial product.

## Point-in-time contract

### Historical retained document

The first retained history fetched by EtherForecast is `vintage_mode=reconstructed`.

We do not infer that the values were available to EtherForecast at their historical timestamps. Every historical row receives:

- `available_at = actual first EtherForecast receipt time`;
- `ingested_at = actual receipt time`;
- `revision = 0`;
- raw response SHA-256.

Consequently this first backfill is **not eligible for retrospective price-alpha promotion**.

### Prospective observations

After `collector_started_at` is frozen:

- newly observed source intervals at/after collector start are `vintage_mode=observed_vintages`;
- `available_at = actual EtherForecast receipt time`;
- later source edits increment row revision and become available only at the revision receipt;
- the prior version remains in the append-only receipt journal.

### Live snapshot receipts

Every collection also stores the source live revision and:

- total/finalized supply;
- accounting epoch/interval;
- issuance;
- burn;
- net issuance;
- source `asOf` metadata;
- exact raw SHA-256.

These are receipt snapshots, not a license to interpolate missing history.

## Accounting identity gate

Where all components are available:

`netWei == issuanceWei - burnWei`

The collector records `identity_error_wei`. Non-zero identities block the source gate until reconciled rather than being silently corrected.

## Evidence preservation

Each run retains:

- latest interval table;
- append-only interval revision journal;
- append-only live receipt table;
- raw retained-history response;
- raw live response;
- collector state;
- machine-readable rights/PIT/coverage/accounting report.

GitHub Actions carries state forward between runs. The existing content-addressed Google Drive archive is extended with dedicated `eth-supply-research` and `eth-etf-research` streams so successful main-branch receipts survive disposable Actions retention.

## Promotion boundary

MODEL/site use remains blocked until all of the following are complete:

1. commercial/intended-use rights review or replacement with a self-derived chain source;
2. sufficient prospective receipt continuity;
3. zero unexplained accounting-identity failures;
4. independent chain-position cross-checks;
5. self-derived historical PIT backfill for any retrospective model claim;
6. preregistered supply hypothesis with a frozen MODEL gate.

No paid API is authorized by this contract.
