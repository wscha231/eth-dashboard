#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_lab.liquidation_cascade import collect_bybit_liquidations


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect prospective Bybit ETHUSDT liquidation receipts")
    parser.add_argument("--root", type=Path, default=Path("lake/research/eth_liquidations"))
    parser.add_argument("--duration-seconds", type=int, default=60)
    parser.add_argument("--websocket-url", default=None)
    args = parser.parse_args()
    kwargs = {"duration_seconds": args.duration_seconds}
    if args.websocket_url:
        kwargs["websocket_url"] = args.websocket_url
    report = collect_bybit_liquidations(args.root, **kwargs)
    print(json.dumps(report, sort_keys=True))
    if not report.get("subscription_acknowledged") or not report.get("last_session_complete"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
