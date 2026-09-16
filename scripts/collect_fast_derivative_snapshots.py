#!/usr/bin/env python3
"""Collect receipt-timestamped ETH derivatives context for short-horizon research.

This collector intentionally records only what the public endpoints expose at run
time. It never manufactures historical OI. Long-history funding/basis/DVOL are
handled by the slower free-source backfill workflow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_lab.free_source_fallbacks import collect_bitget_ticker_snapshot
from data_lab.market_research_sources import collect_hyperliquid_eth_snapshot
from scripts.collect_free_research_sources import save_history


def _coverage(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty:
        return {"rows": 0, "columns": 0, "start": None, "end": None}
    index = pd.DatetimeIndex(frame.index)
    return {
        "rows": int(len(frame)),
        "columns": int(frame.shape[1]),
        "start": index.min().isoformat(),
        "end": index.max().isoformat(),
    }


def _receipt_hour(value: datetime | pd.Timestamp | None = None) -> pd.Timestamp:
    stamp = pd.Timestamp(value or datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp.floor("h")


def _stamp_snapshot(frame: pd.DataFrame, receipt: pd.Timestamp) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if len(frame) != 1:
        raise ValueError("fast derivative snapshot must contain exactly one row")
    out = frame.copy()
    out.index = pd.DatetimeIndex([receipt], name="date")
    return out


def collect(output_dir: Path, receipt: pd.Timestamp) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema": 1,
        "receipt_at_utc": receipt.isoformat(),
        "status": "research_only_prospective_receipts",
        "sources": {},
    }

    jobs = [
        (
            "bitget_context",
            output_dir / "bitget_eth_context_4h.csv",
            lambda: collect_bitget_ticker_snapshot(receipt),
        ),
        (
            "hyperliquid_context",
            output_dir / "hyperliquid_eth_context_4h.csv",
            lambda: collect_hyperliquid_eth_snapshot(receipt),
        ),
    ]

    for source_id, path, fn in jobs:
        try:
            snapshot = _stamp_snapshot(fn(), receipt)
            merged = save_history(path, snapshot)
            report["sources"][source_id] = {
                "status": "ok" if not snapshot.empty else "empty",
                "path": str(path),
                "note": "receipt-timestamped public snapshot; old OI is never synthesized",
                **_coverage(merged),
            }
        except Exception as exc:
            report["sources"][source_id] = {
                "status": "error",
                "path": str(path),
                "error": type(exc).__name__,
            }

    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="lake/raw/vendor")
    parser.add_argument("--receipt-time", default="")
    args = parser.parse_args()

    receipt = _receipt_hour(pd.Timestamp(args.receipt_time) if args.receipt_time else None)
    output_dir = Path(args.output_dir)
    report = collect(output_dir, receipt)
    report_path = output_dir.parent.parent / "reports" / "fast_derivative_snapshot.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
