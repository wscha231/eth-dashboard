from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import time
from typing import Any, Callable, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

SOURCE_ID = "ethereum_execution_finalized_rpc"
SOURCE_NAME = "Ethereum execution layer / finalized JSON-RPC"
CHAIN_ID = 1
RIGHTS_STATUS = "authorized_or_self_hosted_rpc_required"
PUBLIC_REDISTRIBUTION = "blocked"
MODEL_USE = "blocked_pending_authorized_rpc_prospective_continuity_and_preregistered_gate"
SITE_USE = "blocked"
PAID_PRODUCT_USE = "blocked"
HISTORY_MODE = "bootstrap_reconstructed_then_prospective_finalized_receipts"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_BLOCKS = 360
PRIORITY_PERCENTILES = (10, 50, 90)
FETCH_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError)


class RpcError(RuntimeError):
    pass


class ContinuityError(RuntimeError):
    pass


def receipt_iso(value: Any | None = None) -> str:
    if value is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    else:
        raise TypeError("receipt time must be datetime, ISO string, or None")
    return dt.isoformat()


def hex_int(value: Any, *, field: str = "value", allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        raise ValueError(f"{field} is missing")
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{field} must be non-negative")
        return value
    text = str(value)
    if not text.startswith("0x") or len(text) < 3:
        raise ValueError(f"{field} is not a JSON-RPC quantity")
    try:
        parsed = int(text, 16)
    except ValueError as exc:
        raise ValueError(f"{field} is not a valid hex quantity") from exc
    if parsed < 0:
        raise ValueError(f"{field} must be non-negative")
    return parsed


def unix_iso(value: int) -> str:
    return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()


def validate_rpc_url(url: str) -> None:
    parsed = urlparse(url)
    if not parsed.hostname:
        raise ValueError("RPC URL must include a host")
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}:
        return
    raise ValueError("remote Ethereum RPC transport must use HTTPS")


def _post_json(
    url: str,
    payload: Any,
    *,
    timeout: int = 40,
    retries: int = 3,
    backoff_seconds: float = 1.0,
) -> Any:
    validate_rpc_url(url)
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "EtherForecast research collector/1.0 (+https://etherforecast.live)",
    }
    last: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            request = Request(url, data=body, headers=headers, method="POST")
            with urlopen(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("oversized Ethereum RPC response")
            return json.loads(raw)
        except FETCH_ERRORS as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= retries - 1:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    assert last is not None
    raise last


def rpc_call(url: str, method: str, params: list[Any]) -> Any:
    payload = {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
    response = _post_json(url, payload)
    if not isinstance(response, dict):
        raise RpcError(f"{method} returned a non-object response")
    if response.get("error") is not None:
        raise RpcError(f"{method} returned JSON-RPC error")
    if "result" not in response:
        raise RpcError(f"{method} result is missing")
    return response["result"]


def rpc_blocks(url: str, block_numbers: Iterable[int], *, chunk_size: int = 100) -> list[dict[str, Any]]:
    numbers = [int(value) for value in block_numbers]
    output: list[dict[str, Any]] = []
    for offset in range(0, len(numbers), max(1, int(chunk_size))):
        chunk = numbers[offset : offset + max(1, int(chunk_size))]
        payload = [
            {
                "jsonrpc": "2.0",
                "id": index + 1,
                "method": "eth_getBlockByNumber",
                "params": [hex(number), False],
            }
            for index, number in enumerate(chunk)
        ]
        response = _post_json(url, payload)
        if not isinstance(response, list):
            raise RpcError("batch eth_getBlockByNumber returned a non-list response")
        by_id: dict[int, dict[str, Any]] = {}
        for item in response:
            if not isinstance(item, dict) or item.get("error") is not None or "result" not in item:
                raise RpcError("batch eth_getBlockByNumber contains an invalid item")
            identifier = int(item.get("id"))
            result = item["result"]
            if not isinstance(result, dict):
                raise RpcError("eth_getBlockByNumber block result is missing")
            by_id[identifier] = result
        if set(by_id) != set(range(1, len(chunk) + 1)):
            raise RpcError("batch eth_getBlockByNumber response ids are incomplete")
        output.extend(by_id[index] for index in range(1, len(chunk) + 1))
    return output


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _fee_array(history: dict[str, Any], key: str, count: int, *, plus_one: bool = False) -> list[Any]:
    values = history.get(key)
    if values is None:
        return [None] * (count + (1 if plus_one else 0))
    if not isinstance(values, list):
        raise ValueError(f"eth_feeHistory {key} is not an array")
    expected = count + (1 if plus_one else 0)
    if len(values) != expected:
        raise ValueError(f"eth_feeHistory {key} expected {expected} rows, got {len(values)}")
    return values


def _priority_row(value: Any) -> tuple[int | None, int | None, int | None]:
    if value is None:
        return None, None, None
    if not isinstance(value, list) or len(value) != len(PRIORITY_PERCENTILES):
        raise ValueError("eth_feeHistory reward row has unexpected shape")
    parsed = tuple(hex_int(item, field="priority fee") for item in value)
    return parsed  # type: ignore[return-value]


def collect_finalized_window(
    rpc_url: str,
    *,
    prior_block_number: int | None = None,
    prior_block_hash: str | None = None,
    max_blocks: int = DEFAULT_MAX_BLOCKS,
    receipt_time: Any | None = None,
    call_fn: Callable[[str, str, list[Any]], Any] = rpc_call,
    blocks_fn: Callable[[str, Iterable[int]], list[dict[str, Any]]] = rpc_blocks,
) -> dict[str, Any]:
    """Collect one bounded finalized window.

    A first run bootstraps only the last ``max_blocks`` finalized blocks. A
    continuation run starts strictly after ``prior_block_number`` and fails if
    the gap exceeds ``max_blocks``. This prevents silent loss of prospective
    observations.
    """
    if max_blocks < 1:
        raise ValueError("max_blocks must be positive")
    received_at = receipt_iso(receipt_time)

    chain_id = hex_int(call_fn(rpc_url, "eth_chainId", []), field="chain id")
    if chain_id != CHAIN_ID:
        raise ValueError(f"expected Ethereum mainnet chain id {CHAIN_ID}, got {chain_id}")

    finalized = call_fn(rpc_url, "eth_getBlockByNumber", ["finalized", False])
    if not isinstance(finalized, dict):
        raise RpcError("finalized block result is missing")
    finalized_number = hex_int(finalized.get("number"), field="finalized block number")
    assert finalized_number is not None

    if prior_block_number is None:
        start_number = max(0, finalized_number - max_blocks + 1)
        receipt_kind = "bootstrap_reconstructed_finalized_window"
    else:
        if prior_block_hash is None:
            raise ValueError("prior_block_hash is required with prior_block_number")
        if finalized_number < prior_block_number:
            raise ContinuityError("finalized head moved behind stored finalized state")
        start_number = prior_block_number + 1
        receipt_kind = "prospective_finalized_receipt"
        missing = finalized_number - prior_block_number
        if missing > max_blocks:
            raise ContinuityError(
                f"prospective finalized gap {missing} exceeds catch-up limit {max_blocks}"
            )
        if missing == 0:
            return {
                "source_id": SOURCE_ID,
                "chain_id": CHAIN_ID,
                "received_at": received_at,
                "available_at": received_at,
                "receipt_kind": receipt_kind,
                "finalized_block": finalized_number,
                "finalized_hash": str(finalized.get("hash") or ""),
                "start_block": None,
                "end_block": None,
                "rows": [],
                "raw_sha256": _canonical_sha256({"chain_id": chain_id, "finalized": finalized}),
            }

    count = finalized_number - start_number + 1
    if count < 1 or count > max_blocks:
        raise ContinuityError("computed finalized window is outside the bounded catch-up limit")

    history = call_fn(
        rpc_url,
        "eth_feeHistory",
        [hex(count), hex(finalized_number), list(PRIORITY_PERCENTILES)],
    )
    if not isinstance(history, dict):
        raise RpcError("eth_feeHistory result is missing")
    oldest = hex_int(history.get("oldestBlock"), field="feeHistory oldestBlock")
    if oldest != start_number:
        raise ContinuityError(
            f"eth_feeHistory returned oldest block {oldest}, expected {start_number}"
        )

    base_fees = _fee_array(history, "baseFeePerGas", count, plus_one=True)
    gas_ratios = _fee_array(history, "gasUsedRatio", count)
    blob_base_fees = _fee_array(history, "baseFeePerBlobGas", count, plus_one=True)
    blob_ratios = _fee_array(history, "blobGasUsedRatio", count)
    rewards = _fee_array(history, "reward", count)

    numbers = list(range(start_number, finalized_number + 1))
    blocks = blocks_fn(rpc_url, numbers)
    if len(blocks) != count:
        raise ContinuityError(f"expected {count} finalized block headers, got {len(blocks)}")

    rows: list[dict[str, Any]] = []
    previous_hash = prior_block_hash if prior_block_number is not None else None
    for index, (expected_number, block) in enumerate(zip(numbers, blocks, strict=True)):
        number = hex_int(block.get("number"), field="block number")
        if number != expected_number:
            raise ContinuityError(f"block number mismatch: expected {expected_number}, got {number}")
        block_hash = str(block.get("hash") or "")
        parent_hash = str(block.get("parentHash") or "")
        if not block_hash.startswith("0x") or len(block_hash) != 66:
            raise ValueError("invalid block hash")
        if not parent_hash.startswith("0x") or len(parent_hash) != 66:
            raise ValueError("invalid parent hash")
        if previous_hash is not None and parent_hash.lower() != previous_hash.lower():
            raise ContinuityError(f"parent hash mismatch at finalized block {number}")

        gas_used = hex_int(block.get("gasUsed"), field="gasUsed")
        gas_limit = hex_int(block.get("gasLimit"), field="gasLimit")
        timestamp = hex_int(block.get("timestamp"), field="timestamp")
        base_fee_header = hex_int(block.get("baseFeePerGas"), field="baseFeePerGas")
        assert gas_used is not None and gas_limit is not None and timestamp is not None
        assert base_fee_header is not None
        if gas_limit <= 0:
            raise ValueError("gasLimit must be positive")

        base_fee_history = hex_int(base_fees[index], field="feeHistory baseFeePerGas")
        if base_fee_header != base_fee_history:
            raise ContinuityError(f"base fee mismatch at finalized block {number}")
        gas_ratio = float(gas_ratios[index])
        header_ratio = gas_used / gas_limit
        if not math.isfinite(gas_ratio) or abs(gas_ratio - header_ratio) > 1e-9:
            raise ContinuityError(f"gas-used ratio mismatch at finalized block {number}")

        blob_gas_used = hex_int(block.get("blobGasUsed"), field="blobGasUsed", allow_none=True)
        excess_blob_gas = hex_int(
            block.get("excessBlobGas"), field="excessBlobGas", allow_none=True
        )
        blob_base_fee = hex_int(
            blob_base_fees[index], field="baseFeePerBlobGas", allow_none=True
        )
        blob_ratio_value = blob_ratios[index]
        blob_ratio = None if blob_ratio_value is None else float(blob_ratio_value)
        if blob_ratio is not None and (not math.isfinite(blob_ratio) or blob_ratio < 0 or blob_ratio > 1):
            raise ValueError("blobGasUsedRatio must be between zero and one")

        p10, p50, p90 = _priority_row(rewards[index])
        transactions = block.get("transactions")
        if not isinstance(transactions, list):
            raise ValueError("block transactions must be an array")

        row = {
            "source_id": SOURCE_ID,
            "chain_id": CHAIN_ID,
            "receipt_kind": receipt_kind,
            "received_at": received_at,
            "available_at": received_at,
            "block_number": number,
            "block_hash": block_hash,
            "parent_hash": parent_hash,
            "block_timestamp": timestamp,
            "block_time_utc": unix_iso(timestamp),
            "gas_limit": gas_limit,
            "gas_used": gas_used,
            "gas_used_ratio": gas_ratio,
            "base_fee_per_gas_wei": base_fee_header,
            "execution_base_fee_burn_wei": base_fee_header * gas_used,
            "blob_gas_used": blob_gas_used,
            "excess_blob_gas": excess_blob_gas,
            "blob_gas_used_ratio": blob_ratio,
            "base_fee_per_blob_gas_wei": blob_base_fee,
            "blob_base_fee_burn_wei": (
                None if blob_base_fee is None or blob_gas_used is None else blob_base_fee * blob_gas_used
            ),
            "priority_fee_p10_wei": p10,
            "priority_fee_p50_wei": p50,
            "priority_fee_p90_wei": p90,
            "transaction_count": len(transactions),
            "historical_pit_eligible": False,
            "prospective_candidate": receipt_kind == "prospective_finalized_receipt",
        }
        rows.append(row)
        previous_hash = block_hash

    finalized_hash = str(finalized.get("hash") or "")
    if rows[-1]["block_hash"].lower() != finalized_hash.lower():
        raise ContinuityError("fetched end block does not match finalized head hash")

    evidence = {
        "chain_id": chain_id,
        "finalized": finalized,
        "fee_history": history,
        "blocks": blocks,
        "received_at": received_at,
    }
    digest = _canonical_sha256(evidence)
    for row in rows:
        row["raw_sha256"] = digest

    return {
        "source_id": SOURCE_ID,
        "chain_id": CHAIN_ID,
        "received_at": received_at,
        "available_at": received_at,
        "receipt_kind": receipt_kind,
        "finalized_block": finalized_number,
        "finalized_hash": finalized_hash,
        "start_block": start_number,
        "end_block": finalized_number,
        "rows": rows,
        "raw_sha256": digest,
    }
