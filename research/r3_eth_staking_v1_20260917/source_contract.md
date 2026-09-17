# R3 P0-3 — ETH staking / validator / queue source contract v1.0

Frozen: 2026-09-17  
Status: **prospective research only; not model-facing, not site-facing, no public redistribution**

## Objective

Start an immutable point-in-time history of Ethereum staking state and validator queues before any staking-based price/event hypothesis is tested.

This contract deliberately reuses the exact `/api/live` response already collected by R3 P0-2 supply research. It does **not** make a second HTTP request and does not infer historical availability.

## Transport / PIT identity

Shared source receipt: `https://ethsupply.fyi/api/live`.

For every staking row:

- `received_at` is copied from the matching supply live receipt;
- `available_at = received_at`;
- `raw_sha256` must exactly equal the supply receipt SHA-256;
- source `revision` must exactly equal the supply receipt revision;
- each source-native staking metric retains its own `status`, `kind`, `asOf`, and `sources` metadata.

If SHA/revision identity does not match, collection fails closed.

## Source-native fields

No ratios, deltas or investment features are created in the collector. The receipt preserves only the source-native state:

- active balance;
- effective balance;
- active validator count;
- pending deposit validators / balance / genesis-slot-zero balance;
- scheduled activation validators / balance;
- scheduled exit validators / balance;
- pending partial-withdrawal requests / balance;
- pending consolidation requests / balance;
- activation/exit churn limit;
- consolidation churn limit.

Gwei balances are preserved exactly as integer strings and additionally rendered to ETH for inspection. Counts remain exact integer strings.

## Sanity gates

These are schema/integrity gates, not predictive assumptions:

1. all declared metric nodes must exist;
2. values must be non-negative base-10 integers;
3. every metric must expose `exact` or `approximate` status;
4. every metric must expose `observed` or `derived` kind;
5. every metric must expose non-empty source provenance;
6. effective balance may not exceed active balance;
7. active validator count must be positive.

No economic threshold is tuned in this source collector.

## Rights boundary

The source is a public API, but EtherForecast has not established a sufficiently explicit commercial redistribution license.

Therefore:

- `license_status=rights_unreviewed_public_api`;
- public redistribution is blocked;
- website/API use is blocked;
- production MODEL use is blocked;
- receipts remain in research artifacts and private Google Drive;
- later use requires rights review, prospective continuity and a separately preregistered ablation/hypothesis gate.

## Historical boundary

This P0-3 collector is **prospective only**. No historical validator/queue series is synthesized from the current receipt.

A later historical source may be admitted only with reproducible observation-time semantics and its own source/rights contract. Current prospective records must never be relabeled as historical PIT data.

## Evidence reuse

Because staking and supply come from the same `/api/live` raw object, both ledgers carry the same receipt timestamp, revision and SHA. This prevents duplicate network calls and prevents supply/staking state from being compared across mismatched snapshots.

The existing `eth-supply-research-state` artifact and its private Drive archive automatically preserve the staking ledger once this contract is merged.

## Promotion boundary

This data contract alone authorizes no forecast change. Before any staking-derived predictor is considered, require:

1. sufficient prospective continuity;
2. rights review or a self-derived replacement source;
3. a frozen hypothesis specifying horizon and task;
4. matched-origin ablation against the appropriate V2 head baseline;
5. the relevant #56 historical and prospective promotion gates.
