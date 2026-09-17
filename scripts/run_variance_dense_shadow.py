#!/usr/bin/env python3
"""Issue and settle the isolated 24h/72h dense HAR-RV variance shadow."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_pipeline.variance_dense_shadow import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({
        "generated_at": result["generated_at"],
        "source_as_of": result["source_as_of"],
        "issuance_audit": result["issuance_audit"],
        "issued_this_run": result["issued_this_run"],
        "duplicate_this_run": result["duplicate_this_run"],
        "settled_this_run": result["settled_this_run"],
        "errors": result["errors"],
        "prospective": result["prospective"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
