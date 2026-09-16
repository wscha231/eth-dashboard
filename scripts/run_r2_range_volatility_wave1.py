#!/usr/bin/env python3
"""Run the frozen R2 Wave 1 evaluator with repository-root import and pandas Index compatibility.

This runner changes no R2 policy, candidate, target, fold, metric, or gate. It exists because
executing a file below research/model directly does not put the repository root on sys.path,
and pandas Index exposes comparison via == rather than Series.eq.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from research.model import r2_range_volatility_wave1 as r2  # noqa: E402


def compatible_indices(features, targets, start, end, *, require_variance=True):
    start, end = r2.utc(start), r2.utc(end)
    ready = features[["eth_vol_24", "eth_vol_168", "eth_vol_720"]].notna().all(axis=1)
    ready &= targets["return"].notna()
    if require_variance:
        ready &= targets["realized_variance_total"].gt(0)
    ready &= features.index.hour == 0
    ready &= features.index >= start
    ready &= targets["target_end"] < end - pd.Timedelta(hours=r2.POLICY["embargo_hours"])
    return features.index[ready]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Exact semantic equivalent of the preregistered helper, with Index == scalar syntax.
    r2._indices = compatible_indices
    result = r2.evaluate(args.root, args.output)
    print(r2.json.dumps({
        "experiment_id": r2.POLICY["experiment_id"],
        "data_as_of": result["data_as_of"],
        "r2_wave1_pass": result["r2_wave1_pass"],
        "horizons": {
            h: {
                "variance_gate": v["variance"]["gate"]["passed"],
                "distribution_gate": v["distribution"]["gate"]["passed"],
            }
            for h, v in result["horizons"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
