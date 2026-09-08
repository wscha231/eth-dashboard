"""Report each complete price window separately; never bridge missing hours.

Window boundaries depend only on source availability, not strategy performance.
Returns from different windows are not compounded into a fictitious full record.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from research_pipeline.event_calibration import long_cash_diagnostic
from signal_pipeline.data import utc


def complete_windows(bars, start, end):
    start, end = utc(start), utc(end)
    eth = bars.loc[bars["product"].eq("ETH-USD")].set_index("open_time")
    grid = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    valid = eth.reindex(grid)[["open", "close"]].notna().all(axis=1)
    windows = []; first = None
    for t, ok in valid.items():
        if ok and first is None:
            first = t
        if not ok and first is not None:
            windows.append((first, t)); first = None
    if first is not None:
        windows.append((first, end))
    return windows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--study-output", type=Path, required=True)
    args = parser.parse_args()
    results = json.loads((args.study_output / "results.json").read_text())
    bars = pd.read_parquet(args.hourly)
    windows = complete_windows(bars, results["start"], results["data_cutoff"])
    report = {"reason": "full-calendar returns unavailable because source prices have gaps",
              "selection": "all contiguous complete source windows; no performance selection",
              "limits": "separate flat starts; no compounding across missing periods; hourly-close drawdown",
              "windows": [[a.isoformat(), b.isoformat()] for a, b in windows], "horizons": {}}
    for horizon in results["horizons"]:
        rows = pd.read_csv(args.study_output / f"h{horizon}_predictions.csv.gz")
        report["horizons"][horizon] = {}
        for variant, group in rows.groupby("variant"):
            report["horizons"][horizon][variant] = [
                long_cash_diagnostic(group.to_dict("records"), bars, start, end, cost)
                for start, end in windows for cost in (0, 10, 25)]
    path = args.study_output / "trading_windows.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
