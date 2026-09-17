#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_lab.eth_staking import append_live_receipt


def last_jsonl(path: Path) -> dict:
    if not path.exists():
        raise ValueError("ETH supply live receipt ledger is missing")
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise ValueError("ETH supply live receipt ledger is empty")
    return json.loads(lines[-1])


def collect(root: str | Path) -> dict:
    base = Path(root)
    raw_path = base / "raw" / "ethsupply_live_latest.json"
    supply_ledger = base / "eth_supply_live_receipts.jsonl"
    if not raw_path.exists() or not raw_path.is_file():
        raise ValueError("shared ethsupply live raw receipt is missing")
    raw = raw_path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    source = last_jsonl(supply_ledger)
    if digest != source.get("raw_sha256"):
        raise ValueError("staking raw receipt does not match supply receipt SHA256")
    payload = json.loads(raw)
    if int(payload.get("revision", -1)) != int(source.get("source_revision", -2)):
        raise ValueError("staking raw receipt revision does not match supply receipt")
    return append_live_receipt(
        base,
        payload,
        received_at=str(source["received_at"]),
        raw_sha256=digest,
        header_revision=str(source.get("header_revision", "")),
        header_generated_at=str(source.get("header_generated_at", "")),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="lake/research/eth_supply")
    args = parser.parse_args()
    print(json.dumps(collect(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
