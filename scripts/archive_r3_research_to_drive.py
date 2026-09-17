#!/usr/bin/env python3
"""Register R3 research artifacts with the existing immutable Drive archive."""
from __future__ import annotations

from scripts import gdrive_store as storage

R3_STREAMS = {"eth-etf-research", "eth-supply-research"}
R3_ARTIFACTS = {
    "eth-etf-flow-state": ("eth-etf-research", "eth_etf_flows.yml"),
    "eth-supply-research-state": ("eth-supply-research", "eth_supply_research.yml"),
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
