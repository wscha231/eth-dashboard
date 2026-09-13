"""Recover research funding history without training or publishing forecasts."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research_pipeline.funding_history import backfill, daily_funding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--budget-seconds", type=float, default=600)
    parser.add_argument("--max-requests", type=int, default=120)
    args = parser.parse_args()
    hourly, report = backfill(args.root, start=args.start, end=args.end,
                             budget_seconds=args.budget_seconds, max_requests=args.max_requests)
    daily = daily_funding(hourly)
    path = Path(args.root)/report["snapshot"].replace("hourly", "daily")
    daily.to_parquet(path, index=False)
    print(json.dumps({k:report[k] for k in ("complete", "rows", "expected_hours", "missing_hours",
          "cached_windows", "downloaded_windows", "snapshot")}, indent=2))
    print(json.dumps({"complete_days": len(daily), "daily_snapshot": path.name}))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
