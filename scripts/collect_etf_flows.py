#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_lab.etf_flows import fetch_farside_html, write_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect research-only U.S. spot ETH ETF flow receipts")
    parser.add_argument("--root", default="lake/research/eth_etf_flows")
    parser.add_argument("--source-file", default="", help="Optional local HTML fixture; never used by scheduled collection")
    args = parser.parse_args()

    raw = Path(args.source_file).read_bytes() if args.source_file else fetch_farside_html()
    report = write_state(args.root, raw, receipt_time=datetime.now(timezone.utc))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
