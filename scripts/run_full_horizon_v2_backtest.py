#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.model.full_horizon_v2_backtest_replay import evaluate


def main() -> None:
    parser = argparse.ArgumentParser(description="Run frozen six-horizon EtherForecast V2 retrospective audit")
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root, args.output)
    compact = {
        "data_as_of": result["data_as_of"],
        "promotion": result["promotion"],
        "horizons": {
            h: {
                "origins": row["origins"],
                "center": row["center"]["passed"],
                "volatility": row["volatility"]["passed"],
                "distribution": row["distribution"]["passed"],
                "event": row["event"]["passed"],
            }
            for h, row in result["horizons"].items()
        },
        "site_recommendation": result["site_recommendation"],
    }
    print(json.dumps(compact, sort_keys=True))


if __name__ == "__main__":
    main()
