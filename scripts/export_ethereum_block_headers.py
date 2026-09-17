#!/usr/bin/env python3
"""Export bounded Ethereum block headers from an EtherForecast-controlled local Geth RPC.

The exporter intentionally accepts loopback HTTP(S) endpoints only. This prevents a
third-party RPC URL from being silently relabeled as self-derived history. Each
JSON-RPC response is preserved as one JSONL line for the Wave-1 importer.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen


def _require_local_rpc(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("RPC URL must be an HTTP(S) URL")
    host = parsed.hostname.lower()
    local = host == "localhost"
    if not local:
        try:
            local = ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
    if not local:
        raise ValueError("self-derived header export requires a loopback/local Geth RPC endpoint")
    return url


def fetch_block(rpc_url: str, block_number: int, *, timeout: int = 30) -> dict:
    payload = {
        "jsonrpc": "2.0",
        "method": "eth_getBlockByNumber",
        "params": [hex(block_number), False],
        "id": block_number,
    }
    request = Request(
        _require_local_rpc(rpc_url),
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        raw = response.read()
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON-RPC response for block {block_number}") from exc
    if not isinstance(result, dict) or result.get("error"):
        raise ValueError(f"JSON-RPC error for block {block_number}: {result.get('error') if isinstance(result, dict) else result}")
    block = result.get("result")
    if not isinstance(block, dict):
        raise ValueError(f"missing block result for {block_number}")
    returned = block.get("number")
    if not isinstance(returned, str) or int(returned, 16) != block_number:
        raise ValueError(f"block-number mismatch: requested {block_number}, received {returned}")
    if not isinstance(block.get("transactions"), list):
        raise ValueError(f"block {block_number} is missing transactions array")
    return result


def export_headers(rpc_url: str, start: int, end: int, output: Path) -> dict:
    if start < 0 or end < start:
        raise ValueError("require 0 <= start <= end")
    output = Path(output)
    if output.exists():
        raise FileExistsError("header export is immutable; choose a new output file")
    output.parent.mkdir(parents=True, exist_ok=True)
    _require_local_rpc(rpc_url)
    temporary = output.with_suffix(output.suffix + ".tmp")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for number in range(start, end + 1):
                row = fetch_block(rpc_url, number)
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                count += 1
            stream.flush()
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return {"start_block": start, "end_block": end, "rows": count, "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rpc-url", default="http://127.0.0.1:8545")
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(export_headers(args.rpc_url, args.start, args.end, args.output), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
