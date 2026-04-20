"""Run the full daily pipeline locally (match what .github/workflows/daily_forecast.yml
runs in CI). Useful for:

- First-time backfill before the GitHub Action takes over.
- Rerunning a day after fixing a data-collector bug.
- Integration testing the whole chain on your workstation.

Steps:
    1. eth_data_collector.py        -- refresh master dataset CSV
    2. eth_price_forecast.py        -- run full forecast suite -> CSVs
    3. forecast_site.persist_forecast -- CSV -> DB
    4. forecast_site.backfill_actuals -- resolve past forecasts + accuracy

Each step is opt-out via a CLI flag so you can rerun just the DB side without
re-downloading data.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER_CSV = PROJECT_ROOT / "lake" / "gold" / "eth_master_daily.csv"
DEFAULT_SUMMARY_CSV = PROJECT_ROOT / "eth_forecast_outputs" / "latest_forecast_summary.csv"
DEFAULT_DB = PROJECT_ROOT / "forecast_site" / "predictions.db"
DEFAULT_CODE = PROJECT_ROOT / "eth_price_forecast.py"


def _run(cmd: list[str]) -> None:
    print(f"\n$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(f"Step failed: {' '.join(str(c) for c in cmd)} "
                         f"(exit {result.returncode})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-phase", default="phase4b_production",
                        help="Label written into forecast_runs.model_phase")
    parser.add_argument("--fast-mode", action="store_true",
                        help="Pass --fast-mode to eth_price_forecast")
    parser.add_argument("--skip-collector", action="store_true",
                        help="Skip eth_data_collector (use existing master CSV)")
    parser.add_argument("--skip-forecast", action="store_true",
                        help="Skip eth_price_forecast (use existing latest_forecast_summary.csv)")
    parser.add_argument("--skip-persist", action="store_true",
                        help="Skip persist_forecast step")
    parser.add_argument("--skip-backfill", action="store_true",
                        help="Skip backfill_actuals step")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip export_json step")
    parser.add_argument("--skip-notify", action="store_true",
                        help="Skip Telegram notification step")
    parser.add_argument("--master-data-csv", default=str(DEFAULT_MASTER_CSV))
    parser.add_argument("--summary-csv", default=str(DEFAULT_SUMMARY_CSV))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    args = parser.parse_args()

    python = sys.executable

    if not args.skip_collector:
        _run([python, "eth_data_collector.py"])

    if not args.skip_forecast:
        forecast_cmd = [
            python, "eth_price_forecast.py",
            "--master-data-csv", args.master_data_csv,
        ]
        if args.fast_mode:
            forecast_cmd.append("--fast-mode")
        _run(forecast_cmd)

    if not args.skip_persist:
        persist_cmd = [
            python, "-m", "forecast_site.persist_forecast",
            "--summary-csv", args.summary_csv,
            "--model-phase", args.model_phase,
            "--code-source", str(DEFAULT_CODE),
            "--db", args.db,
        ]
        if args.fast_mode:
            persist_cmd.append("--fast-mode")
        _run(persist_cmd)

    if not args.skip_backfill:
        _run([
            python, "-m", "forecast_site.backfill_actuals",
            "--master-data-csv", args.master_data_csv,
            "--db", args.db,
        ])

    # Export JSON blobs the frontend consumes — run AFTER backfill so the
    # accuracy snapshot reflects the freshest resolved actuals.
    if not args.skip_export:
        _run([
            python, "-m", "forecast_site.export_json",
            "--db", args.db,
        ])

    # Telegram alert. Best-effort; never aborts the pipeline.
    if not args.skip_notify:
        _run([python, "-m", "forecast_site.notify_telegram"])

    print("\n[run_daily] Complete.")


if __name__ == "__main__":
    main()
