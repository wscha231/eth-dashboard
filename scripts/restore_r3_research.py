#!/usr/bin/env python3
"""Restore one registered R3 research stream from the immutable private Drive store."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import archive_r3_research_to_drive as r3
from scripts.gdrive_store import CheckError, Drive, Store


def restore(stream: str, target: Path, *, optional: bool = False) -> dict:
    r3.register()
    if stream not in r3.R3_STREAMS:
        raise CheckError("Unknown R3 research stream")
    result = Store(Drive()).restore(stream, target, required=not optional)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stream", choices=sorted(r3.R3_STREAMS), required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--optional", action="store_true")
    args = parser.parse_args()
    try:
        restore(args.stream, args.target, optional=args.optional)
    except Exception as exc:
        print("R3 restore failed: " + (str(exc) if isinstance(exc, CheckError) else type(exc).__name__), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
