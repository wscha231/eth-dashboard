"""Honor the recorded provider cooldown while preserving Git data updates."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_BRANCH = "data/daily-forecast"


def timestamp(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("deployment policy requires timezone-aware timestamps")
    return result.astimezone(timezone.utc)


def cooldown(source=ROOT, now=None):
    policy = json.loads((Path(source) / "ops/deployment_cooldown.json").read_text())
    start, end = timestamp(policy["detected_at"]), timestamp(policy["not_before"])
    if end <= start:
        raise ValueError("invalid deployment cooldown interval")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        raise ValueError("deployment clock requires a timezone")
    return policy if start <= now < end else None


def configure(target, source=ROOT, now=None, stage=False):
    target, source = Path(target), Path(source)
    policy = cooldown(source, now)
    config = json.loads((source / "forecast_site/vercel.json").read_text())
    config.setdefault("git", {})["deploymentEnabled"] = {
        "**": False, PRODUCTION_BRANCH: policy is None,
    }
    path = target / "forecast_site/vercel.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2) + "\n")
    if stage:
        subprocess.run(["git", "-C", str(target), "add", "-f", "--", "forecast_site/vercel.json"], check=True)
    return policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path)
    parser.add_argument("--stage", action="store_true")
    parser.add_argument("--require-enabled", action="store_true")
    args = parser.parse_args()
    policy = configure(args.target, stage=args.stage) if args.target else cooldown()
    if policy:
        print(f"::warning::Hosting cooldown until {policy['not_before']}: {policy['reason']} "
              "Git data retained; external publication has not been verified.")
        if args.require_enabled:
            raise SystemExit(75)
    else:
        print("Deployment eligible on data/daily-forecast only; external verification is still required.")


if __name__ == "__main__":
    main()
