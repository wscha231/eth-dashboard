#!/usr/bin/env python3
"""Issue and settle the isolated HAR-RV prospective variance shadow."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_pipeline.variance_shadow import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps({
        "generated_at": result["generated_at"],
        "source_as_of": result["source_as_of"],
        "issued_this_run": result["issued_this_run"],
        "settled_this_run": result["settled_this_run"],
        "errors": result["errors"],
        "prospective": result["prospective"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
