# R3 P0-2 — ETH supply / issuance / burn source contract v1

Status: **research only; not model-facing, not site-facing, no public redistribution**  
Frozen: 2026-09-17

## Objective

Build a durable point-in-time receipt history for ETH supply dynamics before testing any price-alpha hypothesis.

The data family contains total ETH supply, consensus issuance, execution/consensus burn, derived interval net issuance, retained cumulative accounting, and exact source revision/hash metadata.

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

Provider: `ethsupply.fyi`.

Methodology: `https://ethsupply.fyi/methodology/`.

The collector deliberately separates three public contracts:

- `GET https://ethsupply.fyi/api/history?range=30d` — rolling detailed chart history with epoch observations;
- `GET https://ethsupply.fyi/api/tracked` — retained cumulative accounting summary;
- `GET https://ethsupply.fyi/api/live` — current exact supply and finalized-epoch accounting.

The provider documents exact base-10 wei strings and successful-response headers `x-supply-sha256`, `x-supply-generated-at`, and `x-supply-revision`.

### Why detailed history uses 30d instead of retained

The first live gate demonstrated that `history?range=retained` did not expose detailed epoch rows suitable for the interval parser. The provider separately defines `/api/tracked` as the retained cumulative-accounting contract. We therefore keep the two semantics separate rather than weakening the parser:

- rolling `30d` supplies detailed observations;
- `/api/tracked` supplies the retained cumulative identity check;
- EtherForecast's append-only receipts accumulate a longer prospective detailed history over time.

This change was made after a failed live-source gate and before any model experiment.

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

### Initial rolling history

The first `30d` document fetched by EtherForecast is `vintage_mode=reconstructed`.

We do not infer that its values were available to EtherForecast at their historical timestamps. Every first-load interval receives:

- `available_at = actual first EtherForecast receipt time`;
- `ingested_at = actual receipt time`;
- `revision = 0`;
- raw response SHA-256.

Consequently this first 30-day backfill is **not eligible for retrospective price-alpha promotion**.

### Prospective detailed history

After `collector_started_at` is frozen:

- newly observed intervals at/after collector start are `vintage_mode=observed_vintages`;
- `available_at = actual EtherForecast receipt time`;
- later source edits increment row revision and become available only at the revision receipt;
- the prior version remains in the append-only receipt journal;
- as the rolling 30-day window advances, EtherForecast's retained append-only table becomes a genuinely prospective long history.

### Retained summary receipts

Every collection appends `/api/tracked` as an immutable receipt with:

- source revision and generation time;
- retained from/to slots and timestamps;
- cumulative issuance, burn and net issuance;
- exact raw SHA-256.

It is an accounting summary, not a substitute for historical interval vintages.

### Live snapshot receipts

Every collection also appends `/api/live` with:

- total/finalized supply;
- accounting epoch/interval;
- issuance;
- burn;
- net issuance;
- source `asOf` metadata;
- exact raw SHA-256.

These are receipt snapshots, not a license to interpolate missing history.

## Accounting gates

For retained and live documents where all native components are available:

`netWei == issuanceWei - burnWei`

The collector records `identity_error_wei`. Non-zero identities block the source gate until reconciled rather than being silently corrected.

The detailed HistoryPoint contract does not publish a native `netWei` field, so EtherForecast derives interval net as `issuanceWei - burnWei` and labels it derived. It is not counted as an independent identity test.

## Evidence preservation

Each run retains:

- accumulated detailed interval table;
- append-only interval revision journal;
- append-only retained-summary receipts;
- append-only live receipts;
- raw 30-day history response;
- raw tracked response;
- raw live response;
- collector state;
- machine-readable rights/PIT/coverage/accounting report.

GitHub Actions carries state forward between runs. The existing content-addressed Google Drive archive is extended with dedicated `eth-supply-research` and `eth-etf-research` streams so successful main-branch receipts survive disposable Actions retention.

## Promotion boundary

MODEL/site use remains blocked until all of the following are complete:

1. commercial/intended-use rights review or replacement with a self-derived chain source;
2. sufficient prospective receipt continuity;
3. zero unexplained retained/live accounting-identity failures;
4. independent chain-position cross-checks;
5. self-derived historical PIT backfill for any retrospective model claim;
6. preregistered supply hypothesis with a frozen MODEL gate.

No paid API is authorized by this contract.
