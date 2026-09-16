#!/usr/bin/env python3
"""Collect versioned ETH ETF flow research data without making it model-facing.

The numerical dataset is third-party and rights-unreviewed. It is stored for
private/internal research recovery via the existing daily-data Drive snapshot,
not for public site/API redistribution.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_lab.eth_etf_flows import (
    CONTRACT_VERSION,
    FARSIDE_URL,
    PARSER_VERSION,
    PRODUCTION_USE_APPROVED,
    RIGHTS_STATUS,
    SOURCE,
    append_fetch_receipt,
    append_versions,
    archive_raw,
    availability_sidecar,
    fetch_html,
    latest_snapshot,
    load_versions,
    parse_farside_html,
    sha256,
    validation_report,
)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def save_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("lake/raw/vendor"))
    parser.add_argument("--report", type=Path, default=Path("lake/reports/eth_etf_flows.json"))
    parser.add_argument("--url", default=FARSIDE_URL)
    parser.add_argument("--fixture", type=Path, default=None, help="offline HTML fixture; receipt time is still explicit")
    parser.add_argument("--received-at", default="", help="test/replay-only explicit receipt timestamp")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    versions_path = out / "eth_etf_farside_versions.csv"
    latest_path = out / "eth_etf_farside_daily.csv"
    availability_path = out / "eth_etf_farside_daily.availability.csv"
    receipts_path = out / "eth_etf_farside_fetch_receipts.csv"

    if args.fixture:
        raw = args.fixture.read_bytes()
        received_at = pd.Timestamp(args.received_at or datetime.now(timezone.utc))
        if received_at.tzinfo is None:
            received_at = received_at.tz_localize("UTC")
        else:
            received_at = received_at.tz_convert("UTC")
    else:
        if args.received_at:
            raise SystemExit("--received-at is allowed only with --fixture")
        raw, received_at = fetch_html(args.url)

    raw_hash = sha256(raw)
    raw_path = archive_raw(raw, out)
    parsed = parse_farside_html(raw)
    prior = load_versions(versions_path)
    versions, changes = append_versions(
        prior,
        parsed,
        received_at=received_at,
        raw_sha256=raw_hash,
        source_url=args.url,
    )
    snapshot = latest_snapshot(versions)
    if snapshot.empty:
        raise SystemExit("ETF snapshot is empty after version append")

    save_csv(versions_path, versions)
    save_csv(latest_path, snapshot)
    save_csv(availability_path, availability_sidecar(snapshot))
    fetch_receipts = append_fetch_receipt(
        receipts_path,
        received_at=received_at,
        raw_sha256=raw_hash,
        source_url=args.url,
    )
    validation = validation_report(snapshot)

    report = {
        "schema": 1,
        "source_contract": CONTRACT_VERSION,
        "parser_version": PARSER_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "received_at": received_at.isoformat(),
        "source": SOURCE,
        "source_url": args.url,
        "rights_status": RIGHTS_STATUS,
        "production_use_approved": PRODUCTION_USE_APPROVED,
        "public_redistribution_approved": False,
        "model_consumer_enabled": False,
        "pit_status": "reconstructed_d_plus_1_12utc_then_actual_receipt_versions",
        "historical_availability_note": "D+1 12:00 UTC is a conservative research convention, not an asserted publication timestamp",
        "raw_sha256": raw_hash,
        "raw_archive": raw_path.relative_to(out.parent.parent.parent).as_posix() if len(raw_path.parents) >= 3 else raw_path.as_posix(),
        "parsed_rows": int(len(parsed)),
        "version_rows": int(len(versions)),
        "fetch_receipts": int(len(fetch_receipts)),
        "changes": changes,
        "validation": validation,
        "storage": {
            "normalized_latest": latest_path.as_posix(),
            "version_ledger": versions_path.as_posix(),
            "availability_sidecar": availability_path.as_posix(),
            "receipt_log": receipts_path.as_posix(),
            "raw_receipts": (out / "receipts" / "eth_etf_farside").as_posix(),
            "durable_stream": "daily-data",
            "public_git_persistence": False,
        },
        "claim_boundary": "research data plumbing only; no predictive edge or production eligibility established",
    }
    atomic_text(args.report, json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    print(json.dumps(report, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
