#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_pipeline.v2_replay import run_v2_replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Export same-origin ForecastBundleV2 R1 replay metrics")
    parser.add_argument("--root", default="lake/signals")
    parser.add_argument("--output", default="lake/signals/v2_replay.json")
    args = parser.parse_args()
    result = run_v2_replay(args.root, args.output)
    print(json.dumps({
        "policy": result["policy"],
        "target_spec_id": result["target_spec_id"],
        "horizons": {h: {"origins": v["origins"], "head_status": v["head_status"]} for h, v in result["horizons"].items()},
        "promotion": result["promotion"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
