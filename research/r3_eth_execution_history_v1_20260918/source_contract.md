# R3 P1 — self-derived Ethereum execution history Wave 1

Frozen: 2026-09-18  
Status: **data infrastructure / research only; not model-facing, not site-facing**

## Objective

Create a vendor-independent historical Ethereum execution dataset that EtherForecast can reproduce from an EtherForecast-controlled Geth node. The immediate purpose is to remove commercial-data rights as a bottleneck for future R4 hypotheses, not to claim price alpha.

Wave 1 deliberately limits scope to deterministic execution-layer quantities. Total ETH supply, consensus issuance and validator economics are deferred because they require a separate consensus-layer accounting contract.

## Primary source

Upstream go-ethereum / Geth.

Geth's built-in live `supply` tracer can emit append-only per-block JSONL during a full sync. EtherForecast Wave 1 consumes the current structured tracer fields:

- `blockNumber`;
- `hash`;
- `parentHash`;
- `burn.1559`;
- `burn.blob`;
- `burn.misc`.

Issuance-side tracer fields may be preserved as raw evidence but **are not interpreted as total-network ETH issuance in Wave 1**.

Official source documentation: `https://geth.ethereum.org/docs/developers/evm-tracing/live-tracing`.

A companion block-header JSONL export must come from the same EtherForecast-controlled execution node using Ethereum JSON-RPC (`eth_getBlockByNumber`) and provide, where the hard-fork schema supports them:

- block number / hash / parent hash;
- block timestamp;
- gas used / gas limit;
- base fee per gas;
- transaction list or transaction count;
- blob gas used / excess blob gas.

Official JSON-RPC reference: `https://ethereum.org/developers/docs/apis/json-rpc/`.

No third-party RPC response is silently relabeled as self-derived data. A third-party endpoint may be used only as an explicitly identified cross-check with its own rights contract.

## Exact integer policy

All wei, gas, block-number and timestamp quantities are parsed as arbitrary-precision integers. Hex quantities remain exactly representable. No floating-point value participates in identity or burn reconciliation.

Human-readable ETH values are presentation-only derivatives from exact wei integers.

## Reorg and canonical-chain contract

Raw rows are append-only evidence. A repeated block height with a different hash is not a duplicate to delete; it is potential reorg evidence.

Canonicalization rules for a bounded chunk:

1. identify a unique highest-height tip within the chunk;
2. walk `parentHash` backward through available rows;
3. require each in-chunk parent to be exactly one block lower;
4. mark the walked chain `canonical_within_chunk`;
5. preserve every other row as `orphaned_within_chunk`;
6. if multiple competing highest tips exist, fail closed unless an external canonical anchor is supplied by a later reviewed contract;
7. preserve the first parent outside the chunk as the boundary anchor for cross-chunk reconciliation.

A later full-history assembler must reconcile chunk boundaries before declaring global canonical coverage.

## Cross-source identity

Supply-tracer and block-header rows are joined only when `blockNumber`, `hash` and `parentHash` all agree. A height-only join is forbidden.

For post-EIP-1559 blocks where `baseFeePerGas` is present, Wave 1 verifies:

`base_fee_burn_wei = baseFeePerGas * gasUsed = tracer burn.1559`

A mismatch fails the canonical normalized row. Blob burn is preserved from the Geth supply tracer; Wave 1 does not independently recompute the blob-base-fee function.

## History / PIT evidence grade

This backfill is labeled:

`history_mode=reconstructed_canonical_chain`

It is **not** prospective receipt evidence. We do not fabricate historical `received_at` or `available_at` values from the date the reconstruction job runs.

Before any R4 backtest may consume a derived feature, a separate preregistered availability contract must define how canonical/finality information available at prediction time is represented without using future reorg resolution.

Until that gate exists:

- historical alpha eligibility: blocked;
- prospective alpha eligibility: blocked;
- MODEL use: blocked;
- website/API use: blocked;
- paid-product use: blocked.

## Wave-1 normalized fields

Per canonical block, minimum fields are:

- `block_number`;
- `block_hash`;
- `parent_hash`;
- `block_timestamp`;
- `canonical_status`;
- `burn_1559_wei`;
- `burn_blob_wei`;
- `burn_misc_wei`;
- `burn_total_wei`;
- `gas_used`;
- `gas_limit`;
- `base_fee_per_gas_wei` when applicable;
- `tx_count`;
- `blob_gas_used` / `excess_blob_gas` when available;
- raw source hashes and schema identifiers.

Hourly/daily aggregation is deterministic and may include sums of burn/gas/transactions/block count and explicitly named utilization summaries. Aggregation does not change the evidence grade.

## Storage

Raw full-history files do not belong in normal Git history.

Expected durable layout:

- `lake/raw/ethereum_execution/geth_supply/*.jsonl`;
- `lake/raw/ethereum_execution/block_headers/*.jsonl`;
- immutable content hashes / manifests;
- canonical normalized chunks;
- hourly/daily research aggregates;
- coverage/reorg/reconciliation reports;
- Google Drive or another durable object store for large artifacts.

GitHub stores code, contracts, small fixtures and manifests only.

## Infrastructure boundary

GitHub Actions validates parsers and bounded chunks. It must not be treated as the Ethereum-node host.

The actual historical source requires persistent compute/storage capable of a Geth full sync with the supply tracer enabled. The node/export layer is operational infrastructure outside disposable Actions runners.

## Promotion boundary

Wave 1 authorizes no forecast change. MODEL eligibility requires, in order:

1. exact parser and identity tests;
2. reorg/canonicalization tests and cross-chunk reconciliation;
3. real-chain fixture validation;
4. historical coverage report;
5. frozen availability/finality semantics;
6. units/hard-fork schema registry;
7. one preregistered R4 hypothesis and matched-origin ablation;
8. relevant #56 historical gate;
9. separate prospective evidence.
