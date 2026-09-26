#!/usr/bin/env python3
"""Restore and verify all private derivative research streams from Drive."""
from __future__ import annotations

from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import archive_r3_research_to_drive as r3
from scripts.gdrive_store import Drive, Store

EXPECTED = {
    "derivative-history-research": {
        "lake/raw/vendor/bitget_eth_free_features.csv",
        "lake/raw/vendor/deribit_eth_dvol_daily.csv",
        "lake/raw/vendor/hyperliquid_eth_free_features.csv",
    },
    "derivative-fast-research": {
        "lake/raw/vendor/bitget_eth_context_4h.csv",
        "lake/raw/vendor/hyperliquid_eth_context_4h.csv",
    },
    "derivative-public-seed": {
        "lake/raw/vendor/bitget_eth_context_4h.csv",
        "lake/raw/vendor/bitget_eth_free_features.csv",
        "lake/raw/vendor/hyperliquid_eth_context_4h.csv",
        "lake/raw/vendor/hyperliquid_eth_free_features.csv",
        "lake/raw/vendor/deribit_eth_dvol_daily.csv",
        "lake/raw/vendor/deribit_eth_funding_daily.csv",
        "lake/raw/vendor/deribit_eth_future_snapshot_daily.csv",
        "lake/raw/vendor/deribit_eth_historical_volatility.csv",
        "lake/raw/vendor/deribit_eth_option_snapshot_daily.csv",
    },
}


def verify(output: Path) -> dict:
    r3.register()
    store = Store(Drive())
    results = {}
    with tempfile.TemporaryDirectory(prefix="r3-derivative-recovery-") as tmp:
        root = Path(tmp)
        for stream, expected in EXPECTED.items():
            target = root / stream
            restored = store.restore(stream, target)
            present = {
                p.relative_to(target).as_posix()
                for p in target.rglob("*")
                if p.is_file()
            }
            missing = sorted(expected - present)
            if missing:
                raise ValueError(f"{stream} restore missing required files: {missing}")
            hashes = {
                name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                for name in sorted(expected)
            }
            results[stream] = {
                **restored,
                "required_files": sorted(expected),
                "required_sha256": hashes,
            }
    document = {
        "schema": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "streams": results,
        "cleanup_gate": "private_restore_verified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(document, sort_keys=True))
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("drive-archive-status/r3_derivative_recovery_drills.json"),
    )
    args = parser.parse_args()
    verify(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
