#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_lab.eth_etf_flows import collect


def main():
    parser = argparse.ArgumentParser(description="Collect PIT-labelled Ethereum ETF flow research data")
    parser.add_argument("--output-dir", type=Path, default=Path("lake/etf-research"))
    parser.add_argument("--previous-dir", type=Path)
    parser.add_argument("--received-at", default="")
    args = parser.parse_args()
    report = collect(
        args.output_dir,
        previous_dir=args.previous_dir if args.previous_dir and args.previous_dir.exists() else None,
        received_at=args.received_at or None,
    )
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
