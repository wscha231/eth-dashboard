#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_lab.eth_execution_network import (
    DEFAULT_MAX_BLOCKS,
    HISTORY_MODE,
    MODEL_USE,
    PAID_PRODUCT_USE,
    PUBLIC_REDISTRIBUTION,
    RIGHTS_STATUS,
    SITE_USE,
    SOURCE_ID,
    collect_finalized_window,
    receipt_iso,
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists() or path.stat().st_size == 0:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "on"}


def blocked_report(*, reason: str, now: str, existing_rows: int) -> dict[str, Any]:
    return {
        "schema": 1,
        "generated_at_utc": now,
        "source_id": SOURCE_ID,
        "status": reason,
        "collection_state": reason,
        "rights_status": RIGHTS_STATUS,
        "public_redistribution": PUBLIC_REDISTRIBUTION,
        "model_use": MODEL_USE,
        "site_use": SITE_USE,
        "paid_product_use": PAID_PRODUCT_USE,
        "history_mode": HISTORY_MODE,
        "historical_alpha_eligibility": "blocked_no_historical_pit_receipts",
        "prospective_alpha_eligibility": "blocked_until_authorized_transport_continuity_and_preregistered_gate",
        "existing_receipt_rows": existing_rows,
        "new_receipt_rows": 0,
        "production_effect": "none",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="lake/research/eth_execution_network")
    parser.add_argument("--rpc-url", default=os.getenv("ETHEREUM_RPC_URL", ""))
    parser.add_argument("--max-blocks", type=int, default=DEFAULT_MAX_BLOCKS)
    parser.add_argument(
        "--transport-authorized",
        action="store_true",
        default=truthy(os.getenv("ETHEREUM_RPC_AUTHORIZED")),
        help="Attest that the configured RPC transport permits this automated research use.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Write a blocked report and exit zero when RPC/authorization is missing.",
    )
    args = parser.parse_args()

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    rows_path = root / "finalized_blocks.jsonl"
    runs_path = root / "collection_runs.jsonl"
    state_path = root / "state.json"
    report_path = root / "report.json"
    csv_path = root / "finalized_blocks.csv"

    existing = read_jsonl(rows_path)
    state = read_json(state_path)
    now = receipt_iso()

    rpc_url = str(args.rpc_url or "").strip()
    if not rpc_url:
        report = blocked_report(
            reason="blocked_missing_authorized_rpc", now=now, existing_rows=len(existing)
        )
        write_json(report_path, report)
        write_csv(csv_path, existing)
        print(json.dumps(report, sort_keys=True))
        return 0 if args.allow_blocked else 2
    if not args.transport_authorized:
        report = blocked_report(
            reason="blocked_rpc_rights_not_attested", now=now, existing_rows=len(existing)
        )
        write_json(report_path, report)
        write_csv(csv_path, existing)
        print(json.dumps(report, sort_keys=True))
        return 0 if args.allow_blocked else 2

    prior_number = state.get("last_finalized_block")
    prior_hash = state.get("last_finalized_hash")
    if (prior_number is None) != (prior_hash is None):
        raise RuntimeError("stored execution-network state has incomplete finalized identity")

    result = collect_finalized_window(
        rpc_url,
        prior_block_number=None if prior_number is None else int(prior_number),
        prior_block_hash=None if prior_hash is None else str(prior_hash),
        max_blocks=int(args.max_blocks),
        receipt_time=now,
    )
    new_rows = list(result["rows"])

    by_number = {int(row["block_number"]): row for row in existing}
    for row in new_rows:
        number = int(row["block_number"])
        old = by_number.get(number)
        if old is not None:
            if str(old.get("block_hash", "")).lower() != str(row["block_hash"]).lower():
                raise RuntimeError(f"conflicting finalized hash for stored block {number}")
            raise RuntimeError(f"collector attempted to append duplicate finalized block {number}")
        by_number[number] = row

    append_jsonl(rows_path, new_rows)
    all_rows = read_jsonl(rows_path)
    if all_rows:
        numbers = [int(row["block_number"]) for row in all_rows]
        if numbers != sorted(numbers) or len(numbers) != len(set(numbers)):
            raise RuntimeError("stored finalized block receipts are not strictly unique and ordered")
        for left, right in zip(all_rows, all_rows[1:]):
            if int(right["block_number"]) != int(left["block_number"]) + 1:
                raise RuntimeError("stored finalized block receipts contain a gap")
            if str(right["parent_hash"]).lower() != str(left["block_hash"]).lower():
                raise RuntimeError("stored finalized block receipts fail parent-hash continuity")

    collector_started_at = str(state.get("collector_started_at") or now)
    latest_block = int(result["finalized_block"])
    latest_hash = str(result["finalized_hash"])
    if all_rows:
        latest_block = int(all_rows[-1]["block_number"])
        latest_hash = str(all_rows[-1]["block_hash"])
    new_state = {
        "schema": 1,
        "source_id": SOURCE_ID,
        "collector_started_at": collector_started_at,
        "updated_at": now,
        "last_finalized_block": latest_block,
        "last_finalized_hash": latest_hash,
        "history_mode": HISTORY_MODE,
        "transport_authorization_attested": True,
    }
    write_json(state_path, new_state)

    run_row = {
        "source_id": SOURCE_ID,
        "received_at": result["received_at"],
        "available_at": result["available_at"],
        "receipt_kind": result["receipt_kind"],
        "start_block": result["start_block"],
        "end_block": result["end_block"],
        "finalized_block": result["finalized_block"],
        "finalized_hash": result["finalized_hash"],
        "new_rows": len(new_rows),
        "raw_sha256": result["raw_sha256"],
        "transport_authorization_attested": True,
    }
    append_jsonl(runs_path, [run_row])
    write_csv(csv_path, all_rows)

    bootstrap_rows = sum(
        row.get("receipt_kind") == "bootstrap_reconstructed_finalized_window" for row in all_rows
    )
    prospective_rows = sum(
        row.get("receipt_kind") == "prospective_finalized_receipt" for row in all_rows
    )
    latest = all_rows[-1] if all_rows else {}
    report = {
        "schema": 1,
        "generated_at_utc": now,
        "source_id": SOURCE_ID,
        "status": "research_only_authorized_transport_model_blocked",
        "collection_state": "research_active_authorized_transport",
        "rights_status": RIGHTS_STATUS,
        "transport_authorization_attested": True,
        "public_redistribution": PUBLIC_REDISTRIBUTION,
        "model_use": MODEL_USE,
        "site_use": SITE_USE,
        "paid_product_use": PAID_PRODUCT_USE,
        "history_mode": HISTORY_MODE,
        "historical_alpha_eligibility": "blocked_bootstrap_not_historical_pit",
        "prospective_alpha_eligibility": "blocked_until_continuity_and_preregistered_gate",
        "collector_started_at": collector_started_at,
        "new_receipt_rows": len(new_rows),
        "total_receipt_rows": len(all_rows),
        "bootstrap_rows": bootstrap_rows,
        "prospective_rows": prospective_rows,
        "run_receipt_rows": len(read_jsonl(runs_path)),
        "run_raw_sha256": result["raw_sha256"],
        "last_finalized_block": latest_block,
        "last_finalized_hash": latest_hash,
        "latest_block_time_utc": latest.get("block_time_utc"),
        "latest_gas_used_ratio": latest.get("gas_used_ratio"),
        "latest_base_fee_per_gas_wei": latest.get("base_fee_per_gas_wei"),
        "latest_blob_gas_used_ratio": latest.get("blob_gas_used_ratio"),
        "latest_base_fee_per_blob_gas_wei": latest.get("base_fee_per_blob_gas_wei"),
        "production_effect": "none",
    }
    write_json(report_path, report)
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
