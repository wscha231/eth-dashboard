#!/usr/bin/env python3
"""Import one bounded self-derived Ethereum execution-history chunk.

Input files are expected to come from an EtherForecast-controlled Geth node:
structured supply-tracer JSONL and same-node eth_getBlockByNumber JSONL.
The output directory is immutable and research-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_lab.ethereum_execution_history import (
    CANONICAL,
    aggregate_execution,
    build_manifest,
    canonical_json,
    canonicalize_chunk,
    join_execution_rows,
    parse_header_row,
    parse_supply_row,
    read_jsonl,
)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("wb") as stream:
        for row in rows:
            stream.write(canonical_json(row) + b"\n")


def render_report(manifest: dict, hourly_rows: int, daily_rows: int) -> str:
    return "\n".join([
        "# Self-derived Ethereum execution history — bounded chunk",
        "",
        f"- Canonical rows: **{manifest['canonical_rows']}**",
        f"- Orphaned/reorg rows preserved: **{manifest['orphaned_rows']}**",
        f"- Joined canonical rows: **{manifest['joined_canonical_rows']}**",
        f"- Joined block range: **{manifest['first_joined_block']}–{manifest['last_joined_block']}**",
        f"- Hourly aggregate rows: **{hourly_rows}**",
        f"- Daily aggregate rows: **{daily_rows}**",
        f"- Boundary parent hash: `{manifest['boundary_parent_hash']}`",
        "",
        "Evidence grade: `reconstructed_canonical_chain`.",
        "",
        "Historical MODEL availability is intentionally **blocked** until a separately frozen finality/observation-time contract exists. This importer does not invent historical `available_at` values.",
        "",
    ])


def import_chunk(supply_path: Path, header_path: Path, output: Path) -> dict:
    supply_path = Path(supply_path)
    header_path = Path(header_path)
    output = Path(output)
    if output.exists():
        raise FileExistsError("execution-history output is immutable; choose a new output directory")
    output.mkdir(parents=True)

    supply_rows = read_jsonl(supply_path, parse_supply_row)
    header_rows = read_jsonl(header_path, parse_header_row)
    canonical = canonicalize_chunk(supply_rows)
    joined = join_execution_rows(supply_rows, header_rows)
    hourly = aggregate_execution(joined, "hour")
    daily = aggregate_execution(joined, "day")
    manifest = build_manifest(supply_path, header_path, canonical, joined)
    manifest.update({
        "hourly_rows": len(hourly),
        "daily_rows": len(daily),
        "model_use": "blocked",
        "site_use": "blocked",
        "paid_product_use": "blocked",
        "availability_contract": "not_yet_frozen",
    })

    write_jsonl(output / "supply_canonicalized.jsonl", canonical["rows"])
    write_jsonl(output / "execution_canonical.jsonl", joined)
    write_jsonl(output / "execution_hourly.jsonl", hourly)
    write_jsonl(output / "execution_daily.jsonl", daily)
    (output / "manifest.json").write_bytes(canonical_json(manifest))
    (output / "REPORT.md").write_text(render_report(manifest, len(hourly), len(daily)), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--supply", type=Path, required=True, help="Geth supply-tracer JSONL")
    parser.add_argument("--headers", type=Path, required=True, help="same-node block-header JSONL")
    parser.add_argument("--output", type=Path, required=True, help="new immutable output directory")
    args = parser.parse_args()
    manifest = import_chunk(args.supply, args.headers, args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
