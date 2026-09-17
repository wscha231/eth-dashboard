#!/usr/bin/env python3
"""Restore or commit the private ETH ETF research state in Google Drive."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.gdrive_store import Drive, Store, report

STREAM = "etf-research"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("restore", "backup"))
    parser.add_argument("--target", type=Path, default=Path("lake/etf-previous"))
    parser.add_argument("--root", type=Path, default=Path("lake/etf-research"))
    parser.add_argument("--optional", action="store_true")
    args = parser.parse_args()

    store = Store(Drive())
    if args.action == "restore":
        result = store.restore(STREAM, args.target, required=not args.optional)
        report(result)
        return

    report_path = args.root / "report.json"
    if not report_path.is_file():
        raise SystemExit("ETF report.json is required before Drive backup")
    metadata = json.loads(report_path.read_text(encoding="utf-8"))
    raw_sha = metadata["source"]["raw_sha256"]
    key = f"collection:{metadata['generated_at']}:{raw_sha}"
    result = store.backup(
        args.root,
        STREAM,
        key,
        metadata["generated_at"],
        {
            "workflow": "eth_etf_flows.yml",
            "contract_id": metadata["contract_id"],
            "rights_status": metadata["rights_status"],
            "role": "private_research_only",
            "raw_sha256": raw_sha,
        },
    )
    report(result)


if __name__ == "__main__":
    main()
