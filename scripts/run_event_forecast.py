#!/usr/bin/env python3
"""Incremental hourly collection, bounded research replay and immutable inference."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_pipeline.data import collect, utc
from signal_pipeline.engine import atomic_json, daily, replay
from signal_pipeline.ledger import backup, connect, history
from signal_pipeline.evaluate import prospective_report
from signal_pipeline.protocol import DEFAULT_HORIZONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="lake/signals")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--backfill", action="store_true")
    parser.add_argument("--replay", action="store_true")
    parser.add_argument("--daily", action="store_true")
    parser.add_argument("--review", action="store_true")
    parser.add_argument("--backup")
    parser.add_argument("--horizons", nargs="+", type=int, default=list(DEFAULT_HORIZONS))
    parser.add_argument("--budget-seconds", type=int, default=1200)
    parser.add_argument("--start")
    args = parser.parse_args()
    started = utc(); root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    status = "success"; detail = {}
    try:
        if args.collect or args.backfill:
            start = "2017-01-01" if args.backfill else utc().floor("D")-__import__("pandas").Timedelta(days=35)
            _, detail["collection"] = collect(root, start=start, budget_seconds=min(args.budget_seconds, 600 if args.backfill else 90), max_requests=1000 if args.backfill else 12)
        if args.replay:
            detail["replay"] = replay(root, horizons=args.horizons, budget_seconds=args.budget_seconds, start=args.start)["runtime"]
        if args.daily:
            result = daily(root, horizons=args.horizons)
            detail["forecast"] = {k: result[k] for k in ("status", "errors", "runtime_seconds", "release_id")}
            status = result["status"]
        if args.review:
            records = history(root); now = utc()
            review = {"generated_at": now.isoformat(), "all": prospective_report(records),
                      "windows": {str(days): prospective_report([r for r in records if utc(r["issued_at"]) >= now-__import__("pandas").Timedelta(days=days)]) for days in (28,84,365)},
                      "promotion": "research only; no automatic paid-service promotion",
                      "retrain_policy": "monthly checkpoint; weekly replay reuses compatible checkpoints"}
            atomic_json(root/"weekly_review.json", review)
        if args.backup:
            backup(root, args.backup)
    except Exception as exc:
        status = "failed"; detail["error"] = {"type": type(exc).__name__, "message": str(exc)[:300]}
        raise
    finally:
        with connect(root) as con:
            con.execute("INSERT INTO runs VALUES (?,?,?,?)", (started.isoformat(), started.isoformat(), status, json.dumps(detail)))
        print(json.dumps({"status":status, **detail}, indent=2), flush=True)


if __name__ == "__main__":
    main()
