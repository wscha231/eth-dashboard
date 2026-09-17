"""Self-derived Ethereum execution history Wave 1.

Parses Geth supply-tracer JSONL and same-node block-header JSONL, preserves exact
integer quantities, canonicalizes bounded chunks by parentHash continuity and
joins only on full block identity. Research infrastructure only: this module does
not assign historical MODEL availability or promote any feature.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Callable, Iterable

SCHEMA_VERSION = "eth_execution_history_wave1_v1"
HISTORY_MODE = "reconstructed_canonical_chain"
CANONICAL = "canonical_within_chunk"
ORPHANED = "orphaned_within_chunk"


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def file_sha256(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def parse_quantity(value, field: str) -> int:
    if isinstance(value, bool) or value is None:
        raise ValueError(f"{field} must be a non-negative integer quantity")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        text = value.strip().lower()
        if not text:
            raise ValueError(f"{field} is empty")
        try:
            parsed = int(text, 16) if text.startswith("0x") else int(text, 10)
        except ValueError as exc:
            raise ValueError(f"{field} is not an integer quantity") from exc
    else:
        raise ValueError(f"{field} must not be floating-point")
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def normalize_hash(value, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a hex hash")
    text = value.strip().lower()
    if not text.startswith("0x") or len(text) != 66:
        raise ValueError(f"{field} must be a 32-byte 0x-prefixed hash")
    try:
        int(text[2:], 16)
    except ValueError as exc:
        raise ValueError(f"{field} contains non-hex characters") from exc
    return text


def parse_supply_row(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Geth supply row must be an object")
    burn = raw.get("burn")
    if not isinstance(burn, dict):
        raise ValueError("Wave 1 requires the structured Geth burn object")
    block_number = parse_quantity(raw.get("blockNumber"), "blockNumber")
    block_hash = normalize_hash(raw.get("hash"), "hash")
    parent_hash = normalize_hash(raw.get("parentHash"), "parentHash")
    burn_1559 = parse_quantity(burn.get("1559", "0x0"), "burn.1559")
    burn_blob = parse_quantity(burn.get("blob", "0x0"), "burn.blob")
    burn_misc = parse_quantity(burn.get("misc", "0x0"), "burn.misc")
    return {
        "schema": SCHEMA_VERSION,
        "source_kind": "geth_supply_tracer",
        "history_mode": HISTORY_MODE,
        "block_number": block_number,
        "block_hash": block_hash,
        "parent_hash": parent_hash,
        "burn_1559_wei": burn_1559,
        "burn_blob_wei": burn_blob,
        "burn_misc_wei": burn_misc,
        "burn_total_wei": burn_1559 + burn_blob + burn_misc,
        "raw_sha256": digest(raw),
    }


def _unwrap_jsonrpc(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("block header row must be an object")
    if "result" in raw:
        result = raw.get("result")
        if not isinstance(result, dict):
            raise ValueError("JSON-RPC block result must be an object")
        return result
    return raw


def parse_header_row(raw: dict) -> dict:
    block = _unwrap_jsonrpc(raw)
    transactions = block.get("transactions")
    if not isinstance(transactions, list):
        raise ValueError("block header export must retain the transactions array")
    result = {
        "schema": SCHEMA_VERSION,
        "source_kind": "same_node_eth_getBlockByNumber",
        "history_mode": HISTORY_MODE,
        "block_number": parse_quantity(block.get("number"), "number"),
        "block_hash": normalize_hash(block.get("hash"), "hash"),
        "parent_hash": normalize_hash(block.get("parentHash"), "parentHash"),
        "block_timestamp": parse_quantity(block.get("timestamp"), "timestamp"),
        "gas_used": parse_quantity(block.get("gasUsed"), "gasUsed"),
        "gas_limit": parse_quantity(block.get("gasLimit"), "gasLimit"),
        "base_fee_per_gas_wei": (
            None if block.get("baseFeePerGas") is None else parse_quantity(block["baseFeePerGas"], "baseFeePerGas")
        ),
        "blob_gas_used": (
            None if block.get("blobGasUsed") is None else parse_quantity(block["blobGasUsed"], "blobGasUsed")
        ),
        "excess_blob_gas": (
            None if block.get("excessBlobGas") is None else parse_quantity(block["excessBlobGas"], "excessBlobGas")
        ),
        "tx_count": len(transactions),
        "raw_sha256": digest(raw),
    }
    if result["gas_limit"] <= 0:
        raise ValueError("gasLimit must be positive")
    return result


def read_jsonl(path: Path, parser: Callable[[dict], dict]) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for lineno, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
                rows.append(parser(raw))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                raise ValueError(f"{path}:{lineno}: {exc}") from exc
    if not rows:
        raise ValueError(f"{path} contains no parseable rows")
    return rows


def _dedupe_hash_identity(rows: Iterable[dict]) -> dict[str, dict]:
    by_hash: dict[str, dict] = {}
    for row in rows:
        block_hash = row["block_hash"]
        prior = by_hash.get(block_hash)
        if prior is None:
            by_hash[block_hash] = dict(row)
            continue
        identity = ("block_number", "block_hash", "parent_hash")
        if any(prior[key] != row[key] for key in identity):
            raise ValueError(f"conflicting identity for block hash {block_hash}")
        # Exact repeated observations are harmless for canonicalization; source
        # append-only evidence remains in the raw JSONL itself.
        comparison = {k: v for k, v in prior.items() if k != "raw_sha256"}
        candidate = {k: v for k, v in row.items() if k != "raw_sha256"}
        if comparison != candidate:
            raise ValueError(f"conflicting content for repeated block hash {block_hash}")
    return by_hash


def canonicalize_chunk(rows: Iterable[dict]) -> dict:
    by_hash = _dedupe_hash_identity(rows)
    if not by_hash:
        raise ValueError("cannot canonicalize an empty chunk")
    max_height = max(row["block_number"] for row in by_hash.values())
    tips = [row for row in by_hash.values() if row["block_number"] == max_height]
    if len(tips) != 1:
        raise ValueError("ambiguous highest-height tips; external canonical anchor required")

    walked: set[str] = set()
    current = tips[0]
    boundary_parent_hash = None
    while True:
        if current["block_hash"] in walked:
            raise ValueError("cycle detected in parentHash chain")
        walked.add(current["block_hash"])
        parent = by_hash.get(current["parent_hash"])
        if parent is None:
            boundary_parent_hash = current["parent_hash"]
            break
        if parent["block_number"] != current["block_number"] - 1:
            raise ValueError("parentHash points to a non-adjacent in-chunk block height")
        current = parent

    normalized = []
    for row in sorted(by_hash.values(), key=lambda value: (value["block_number"], value["block_hash"])):
        value = dict(row)
        value["canonical_status"] = CANONICAL if row["block_hash"] in walked else ORPHANED
        normalized.append(value)
    return {
        "rows": normalized,
        "tip_block_number": max_height,
        "tip_hash": tips[0]["block_hash"],
        "boundary_parent_hash": boundary_parent_hash,
        "canonical_rows": sum(row["canonical_status"] == CANONICAL for row in normalized),
        "orphaned_rows": sum(row["canonical_status"] == ORPHANED for row in normalized),
    }


def join_execution_rows(supply_rows: Iterable[dict], header_rows: Iterable[dict]) -> list[dict]:
    # Materialize once because node exporters may stream generators. Both inputs
    # are inspected more than once for identity/integrity checks.
    supply_rows = list(supply_rows)
    header_rows = list(header_rows)
    supply_chain = canonicalize_chunk(supply_rows)
    deduped_headers = _dedupe_hash_identity(header_rows)
    header_by_identity = {
        (row["block_number"], row["block_hash"], row["parent_hash"]): row
        for row in deduped_headers.values()
    }
    if len(header_by_identity) != len(deduped_headers):
        raise ValueError("duplicate header identities are ambiguous")

    joined = []
    for supply in supply_chain["rows"]:
        identity = (supply["block_number"], supply["block_hash"], supply["parent_hash"])
        header = header_by_identity.get(identity)
        if header is None:
            if supply["canonical_status"] == CANONICAL:
                raise ValueError(f"missing same-node block header for canonical block {supply['block_number']}")
            continue
        if supply["canonical_status"] != CANONICAL:
            continue
        base_fee = header["base_fee_per_gas_wei"]
        expected_1559 = None if base_fee is None else int(base_fee) * int(header["gas_used"])
        if expected_1559 is not None and expected_1559 != supply["burn_1559_wei"]:
            raise ValueError(
                f"EIP-1559 burn mismatch at block {supply['block_number']}: "
                f"tracer={supply['burn_1559_wei']} header={expected_1559}"
            )
        joined.append({
            "schema": SCHEMA_VERSION,
            "history_mode": HISTORY_MODE,
            "block_number": supply["block_number"],
            "block_hash": supply["block_hash"],
            "parent_hash": supply["parent_hash"],
            "block_timestamp": header["block_timestamp"],
            "canonical_status": CANONICAL,
            "burn_1559_wei": supply["burn_1559_wei"],
            "burn_blob_wei": supply["burn_blob_wei"],
            "burn_misc_wei": supply["burn_misc_wei"],
            "burn_total_wei": supply["burn_total_wei"],
            "gas_used": header["gas_used"],
            "gas_limit": header["gas_limit"],
            "base_fee_per_gas_wei": base_fee,
            "tx_count": header["tx_count"],
            "blob_gas_used": header["blob_gas_used"],
            "excess_blob_gas": header["excess_blob_gas"],
            "geth_supply_raw_sha256": supply["raw_sha256"],
            "block_header_raw_sha256": header["raw_sha256"],
            "burn_1559_header_reconciliation_wei": (
                None if expected_1559 is None else supply["burn_1559_wei"] - expected_1559
            ),
            "available_at": None,
            "model_eligible": False,
        })
    if not joined:
        raise ValueError("no canonical supply/header rows joined")
    joined.sort(key=lambda row: row["block_number"])
    return joined


def _bucket_timestamp(timestamp: int, frequency: str) -> str:
    dt = datetime.fromtimestamp(int(timestamp), tz=timezone.utc)
    if frequency == "hour":
        bucket = dt.replace(minute=0, second=0, microsecond=0)
    elif frequency == "day":
        bucket = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    else:
        raise ValueError("frequency must be 'hour' or 'day'")
    return bucket.isoformat()


def aggregate_execution(rows: Iterable[dict], frequency: str) -> list[dict]:
    buckets: dict[str, dict] = defaultdict(lambda: {
        "block_count": 0,
        "burn_1559_wei": 0,
        "burn_blob_wei": 0,
        "burn_misc_wei": 0,
        "burn_total_wei": 0,
        "gas_used": 0,
        "gas_limit": 0,
        "tx_count": 0,
        "blob_gas_used": 0,
        "blob_gas_observed_blocks": 0,
    })
    for row in rows:
        if row.get("canonical_status") != CANONICAL:
            continue
        key = _bucket_timestamp(row["block_timestamp"], frequency)
        bucket = buckets[key]
        bucket["block_count"] += 1
        for field in ("burn_1559_wei", "burn_blob_wei", "burn_misc_wei", "burn_total_wei", "gas_used", "gas_limit", "tx_count"):
            bucket[field] += int(row[field])
        if row.get("blob_gas_used") is not None:
            bucket["blob_gas_used"] += int(row["blob_gas_used"])
            bucket["blob_gas_observed_blocks"] += 1
    output = []
    for timestamp in sorted(buckets):
        value = {"timestamp": timestamp, "frequency": frequency, "history_mode": HISTORY_MODE, **buckets[timestamp]}
        value["gas_utilization_ratio"] = (
            float(value["gas_used"] / value["gas_limit"]) if value["gas_limit"] > 0 else None
        )
        value["available_at"] = None
        value["model_eligible"] = False
        output.append(value)
    return output


def build_manifest(supply_path: Path, header_path: Path, canonical: dict, joined: list[dict]) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "history_mode": HISTORY_MODE,
        "supply_file": str(Path(supply_path).name),
        "supply_sha256": file_sha256(supply_path),
        "header_file": str(Path(header_path).name),
        "header_sha256": file_sha256(header_path),
        "tip_block_number": canonical["tip_block_number"],
        "tip_hash": canonical["tip_hash"],
        "boundary_parent_hash": canonical["boundary_parent_hash"],
        "canonical_rows": canonical["canonical_rows"],
        "orphaned_rows": canonical["orphaned_rows"],
        "joined_canonical_rows": len(joined),
        "first_joined_block": min(row["block_number"] for row in joined),
        "last_joined_block": max(row["block_number"] for row in joined),
        "historical_model_eligibility": "blocked_pending_finality_availability_contract",
        "public_site_use": "blocked",
        "paid_product_use": "blocked",
    }
