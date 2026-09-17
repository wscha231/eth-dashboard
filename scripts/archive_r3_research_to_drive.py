#!/usr/bin/env python3
"""Register R3 research artifacts with the existing immutable Drive archive."""
from __future__ import annotations

from pathlib import Path
import sys

# When this file is invoked as `python scripts/archive_r3_research_to_drive.py`,
# Python places `scripts/` rather than the repository root on sys.path. Add the
# checkout root explicitly before importing the package-style `scripts.*` modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gdrive_store as storage

R3_STREAMS = {
    "eth-etf-research",
    "eth-supply-research",
    "eth-liquidation-research",
    "eth-execution-network-research",
    # Keep reconstructed/long derivative history and fast prospective context
    # in separate streams. A stream has one immutable watermark; sharing it
    # across independent workflows could make a newer run hide another source.
    "derivative-history-research",
    "derivative-fast-research",
}
R3_ARTIFACTS = {
    "eth-etf-flow-state": ("eth-etf-research", "eth_etf_flows.yml"),
    "eth-supply-research-state": ("eth-supply-research", "eth_supply_research.yml"),
    "eth-liquidation-research-state": ("eth-liquidation-research", "eth_liquidation_research.yml"),
    "eth-execution-network-research-state": (
        "eth-execution-network-research",
        "eth_execution_network_research.yml",
    ),
    # This artifact intentionally excludes FRED and other mixed fallback data.
    "derivative-history-research-state": (
        "derivative-history-research",
        "free_source_backfill.yml",
    ),
    "fast-derivative-state": (
        "derivative-fast-research",
        "fast_derivative_snapshots.yml",
    ),
}


def register() -> None:
    # Mutate the shared set before archive_to_drive validates stream names.
    storage.STREAMS.update(R3_STREAMS)
    from scripts import archive_to_drive as archive
    archive.STREAMS.update(R3_STREAMS)
    archive.ARTIFACTS.update(R3_ARTIFACTS)


def main() -> int:
    register()
    from scripts import archive_to_drive as archive
    return archive.main()


if __name__ == "__main__":
    raise SystemExit(main())
