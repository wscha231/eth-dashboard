# EtherForecast self-derived Ethereum execution-history node runbook

Status: operational preparation only. No MODEL eligibility is granted by this document.

## 1. Why a persistent node is required

Geth's built-in `supply` live tracer emits per-block supply/burn data only while blocks are being processed. For a history from genesis, upstream Geth documentation requires a **full block-by-block sync** with the tracer enabled.

Ethereum proof-of-stake also requires Geth to sync together with a consensus client. GitHub Actions is therefore the wrong execution environment: the node needs persistent SSD, RAM, bandwidth and a long-lived datadir.

Official references:

- Geth live tracing: `https://geth.ethereum.org/docs/developers/evm-tracing/live-tracing`
- Geth sync modes: `https://geth.ethereum.org/docs/fundamentals/sync-modes`
- Ethereum JSON-RPC: `https://ethereum.org/developers/docs/apis/json-rpc/`

## 2. Archive mode is not required for Wave 1

Wave 1 needs the tracer output generated during execution plus historical block headers/bodies. It does **not** require retaining every historical Ethereum state.

Do not enable archive mode merely for this project. Archive mode is a separate storage-heavy requirement for fast historical-state queries. A normal full sync from genesis can prune old state while the append-only tracer output is preserved externally.

## 3. Geth tracer command

Upstream documentation currently shows:

```bash
geth \
  --syncmode full \
  --vmtrace supply \
  --vmtrace.jsonconfig '{"path":"/data/etherforecast/geth-supply","maxSize":100}' \
  [OTHER_GETH_FLAGS_REQUIRED_BY_YOUR_EXECUTION+CONSENSUS_SETUP]
```

The exact Engine API / JWT / consensus-client flags depend on the node deployment and are intentionally not invented in this repo. The supply path must live on persistent storage and should be backed up independently of the Geth datadir.

The expected current structured tracer schema contains:

- `blockNumber`, `hash`, `parentHash`;
- nested `issuance`;
- nested `burn` with `1559`, `blob`, `misc`.

The Wave-1 parser fails closed if a different schema is observed. If a deployed Geth version emits a legacy/changed schema, add a separately versioned parser and fixture rather than silently coercing it.

## 4. Same-node block headers

Expose Geth JSON-RPC on loopback only for the exporter unless the host networking layer supplies its own authenticated tunnel.

Example bounded export after the relevant blocks exist locally:

```bash
python scripts/export_ethereum_block_headers.py \
  --rpc-url http://127.0.0.1:8545 \
  --start 19000000 \
  --end 19099999 \
  --output /data/etherforecast/block-headers/19000000-19099999.jsonl
```

The repo exporter intentionally rejects non-loopback URLs so a third-party RPC cannot accidentally be labeled self-derived.

## 5. Process an immutable chunk

Pair tracer rows and headers covering the same bounded block range, then run:

```bash
python scripts/import_ethereum_execution_history.py \
  --supply /data/etherforecast/geth-supply/<chunk>.jsonl \
  --headers /data/etherforecast/block-headers/<chunk>.jsonl \
  --output /data/etherforecast/normalized/<immutable-chunk-id>
```

The importer:

- preserves reorg/orphan rows;
- canonicalizes the unique in-chunk parentHash path;
- fails on ambiguous equal-height tips;
- joins only on number + hash + parentHash;
- verifies EIP-1559 burn exactly against `baseFeePerGas * gasUsed`;
- emits canonical block, hourly, daily and manifest outputs;
- never invents historical `available_at`;
- keeps MODEL/site/paid use blocked.

## 6. Chunk boundaries

A chunk's first canonical block points to a `boundary_parent_hash` outside that chunk. Before claiming global historical coverage, a later assembler must prove that:

- adjacent chunk heights are contiguous;
- prior tip hash equals next boundary parent hash;
- there are no gaps or overlapping conflicting canonical blocks;
- all source-file hashes are retained in manifests.

Until cross-chunk reconciliation is implemented, bounded chunks are useful parser evidence but not a globally certified history.

## 7. Durable storage

Large raw tracer and header histories must not be committed to normal Git history.

Target architecture:

1. persistent node disk produces raw data;
2. immutable chunks are normalized;
3. raw + normalized + manifests are copied to EtherForecast's durable Drive/object-storage layer;
4. GitHub stores only code, contracts, small fixtures, manifests and audit summaries;
5. MODEL consumes nothing until the finality/PIT availability gate is separately frozen and passed.

## 8. Resource planning

Use current Geth documentation and actual node telemetry when sizing. Official hardware guidance identifies disk I/O/capacity as the main bottleneck and recommends SSD-class storage and substantial RAM; published absolute storage figures are time-sensitive and should not be treated as a current mainnet measurement.

A supply-tracer full sync also adds synchronous tracing overhead, so node sync throughput should be measured before selecting the final machine size.

## 9. Completion criteria for the external node layer

The node layer is ready for historical import only when:

- Geth and a compatible consensus client are fully synced;
- `eth_syncing` reports the execution client caught up;
- tracer JSONL is continuously produced on persistent storage;
- a real small block range has matching same-node header exports;
- Wave-1 importer passes on that real fixture;
- reorg/identity/burn checks produce zero unexplained failures;
- raw and normalized chunks are durably archived.

Only after this should the project freeze a historical finality/availability rule for R4 backtests.
