#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_lab.eth_supply import write_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect research-only ETH supply/issuance/burn receipts")
    parser.add_argument("--root", default="lake/research/eth_supply")
    args = parser.parse_args()
    report = write_state(args.root, receipt_time=datetime.now(timezone.utc))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
