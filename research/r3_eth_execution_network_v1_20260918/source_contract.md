# R3 P1 Ethereum execution-network source contract — 2026-09-18

## Objective

Collect Ethereum execution-layer network demand directly from finalized canonical blocks, without relying on a market-data vendor feature feed. Initial fields are intended to cover the P1 `blob/fees/network use` family:

- execution gas utilization;
- execution base fee and execution base-fee burn;
- blob gas utilization;
- blob base fee and blob base-fee burn;
- finalized block and transaction counts;
- priority-fee percentiles where supported by `eth_feeHistory`.

This source is **research only**. It does not authorize MODEL, website, API, public redistribution, or paid-product use.

## Protocol source

The semantic source of truth is the Ethereum Execution JSON-RPC specification, not a third-party analytics dashboard.

Required calls:

- `eth_chainId` — collector must reject any chain other than Ethereum mainnet (`0x1`).
- `eth_getBlockByNumber("finalized", false)` — establishes the finalized head.
- `eth_getBlockByNumber(blockNumber, false)` — canonical finalized block headers and transaction-hash counts.
- `eth_feeHistory(blockCount, newestBlock, [10,50,90])` — block execution base fees, gas-used ratios, blob base fees / blob-gas-used ratios when supported, and priority-fee percentiles.

References:

- https://ethereum.github.io/execution-apis/
- https://ethereum.github.io/execution-apis/api/methods/eth_feeHistory/
- https://ethereum.org/developers/docs/apis/json-rpc/

Do not hard-code an old EIP-4844 target/max blob schedule to infer current blob utilization. Ethereum can change fork parameters. Prefer the active client's standardized `eth_feeHistory` output and preserve raw header fields (`blobGasUsed`, `excessBlobGas`) when present.

## Transport rights gate

Blockchain facts and the RPC transport are separate concerns. A public RPC endpoint being unauthenticated or free does **not** prove that automated commercial collection/republication is permitted.

The collector therefore has **no default public RPC endpoint**. `ETHEREUM_RPC_URL` must point to one of:

1. an EtherForecast-controlled/self-hosted Ethereum execution node; or
2. a third-party RPC for which the intended automated/commercial use has been explicitly reviewed and allowed.

PublicNode was reviewed as an example and is deliberately **not** adopted as the default: its current terms restrict copying, scraping, redistribution/derivative use of the Service/Service Content and commercial solicitation/use beyond authorization. Other free endpoints must not be substituted without their own terms review.

Until an authorized transport is configured:

- `collection_state = blocked_missing_authorized_rpc`
- public redistribution: blocked
- MODEL/site/paid-product use: blocked
- R4 historical/prospective alpha eligibility: blocked

## PIT / finality policy

1. Only the execution client's `finalized` chain is accepted. `latest` and `safe` are not substitutes for this collector.
2. `received_at` and `available_at` are the actual EtherForecast UTC receipt timestamp for the collection run.
3. On the first successful run, a bounded window ending at the finalized head is retained as `bootstrap_reconstructed_finalized_window`. Those rows are **not historical PIT observations**; they became available to EtherForecast only at receipt time.
4. On subsequent runs, blocks strictly after the last stored finalized block are `prospective_finalized_receipt` rows.
5. The parent hash of the first new block must match the prior stored finalized hash. Every consecutive block must match both block number and parent hash. Gaps or conflicting finalized hashes fail the run.
6. If more blocks were missed than the configured bounded catch-up limit, the collector fails closed instead of silently truncating the gap.
7. No block timestamp may be relabeled as historical `available_at`.

## Derived-field equations

All integer fee arithmetic is performed in wei without floating-point conversion.

- `execution_base_fee_burn_wei = base_fee_per_gas_wei * gas_used`
- `blob_base_fee_burn_wei = base_fee_per_blob_gas_wei * blob_gas_used` when both standardized inputs are available.
- `gas_used_ratio` and `blob_gas_used_ratio` are taken from `eth_feeHistory`; header arithmetic is retained as an independent consistency check for execution gas.

A ratio mismatch beyond a small serialization tolerance fails validation. Blob ratio is not re-derived from hard-coded fork constants.

## Storage contract

State is append-only research evidence:

- `finalized_blocks.jsonl` — normalized finalized-block receipts;
- `collection_runs.jsonl` — one immutable receipt per collection attempt/run;
- `state.json` — collector start time and latest accepted finalized block/hash;
- `report.json` — current gate/coverage summary;
- `finalized_blocks.csv` — convenience projection only; JSONL remains source of truth.

The Actions artifact is `eth-execution-network-research-state`. Once a real authorized-RPC collection exists, it is eligible for the private immutable Drive stream `eth-execution-network-research`. It must never be persisted to the public `data/daily-forecast` branch by default.

## Promotion gates

The source remains blocked until all applicable gates pass:

1. authorized/self-hosted RPC transport documented;
2. mainnet/finality/continuity checks pass;
3. at least the preregistered prospective observation minimum is accumulated;
4. no public-vendor-storage rights conflict exists;
5. an R4 hypothesis names the exact fields and causal availability rule before evaluation;
6. same-origin ablation beats the frozen baseline under predefined metrics;
7. prospective shadow evidence passes before production promotion.

Collector availability alone is not evidence of predictive alpha.
