"""Freeze the current pipeline's walk-forward metrics as the Phase 0 baseline.

Run this ONCE before making any Phase 1 changes. The resulting pickle is the
reference against which all later changes are compared. Never overwrite it
without explicit approval.

Usage (from H:/codex):
    python -m tests.phase0.freeze_baseline --master-data-csv lake/gold/eth_master_daily.csv

If --master-data-csv is omitted, the script invokes the full pipeline via
eth_price_forecast.main() with --json so the collector chain runs end-to-end.

Output:
    tests/phase0/baseline_metrics.pkl   — pickled dict of leaderboards per horizon
    tests/phase0/baseline_metrics.json  — same metrics in human-readable form
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp

OUTPUT_PKL = Path(__file__).with_name("baseline_metrics.pkl")
OUTPUT_JSON = Path(__file__).with_name("baseline_metrics.json")


def _leaderboard_to_records(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard is None or leaderboard.empty:
        return []
    return leaderboard.to_dict(orient="records")


def run_inline(master_data_csv: Path) -> dict[str, Any]:
    market_data = efp.load_market_data_csv(master_data_csv)
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {master_data_csv}")

    horizons = [7, 30]
    per_horizon: dict[int, dict[str, Any]] = {}
    for horizon in horizons:
        feature_frame, feature_columns = efp.build_features(market_data, horizon=horizon)
        full_mask = (
            feature_frame["target_return"].notna()
            & feature_frame["target_close"].notna()
            & feature_frame["eth_close"].notna()
        )
        full_dataset = feature_frame.loc[
            full_mask,
            feature_columns + ["eth_close", "target_return", "target_close",
                               "target_regime", "target_reversal_state",
                               "target_bottom_reversal", "target_top_reversal"],
        ].copy()
        trimmed = efp.trim_training_window(full_dataset, interval="1d", horizon=horizon)
        training_mask = feature_frame.index.isin(trimmed.index) & full_mask
        selected_features, _ = efp.select_usable_feature_columns(
            feature_frame=feature_frame,
            feature_columns=feature_columns,
            training_mask=training_mask,
            min_feature_coverage=0.03,
            horizon=horizon,
        )
        training_dataset = feature_frame.loc[
            training_mask,
            selected_features + ["eth_close", "target_return", "target_close",
                                 "target_regime", "target_reversal_state",
                                 "target_bottom_reversal", "target_top_reversal"],
        ].copy()
        sample_weights = efp.build_time_decay_sample_weights(training_dataset.index, interval="1d", horizon=horizon)

        regression_leaderboard, _ = efp.walk_forward_leaderboard(
            dataset=training_dataset, feature_columns=selected_features,
            n_splits=3, test_size=20, gap=horizon,
            sample_weight=sample_weights, verbose=False,
        )
        classification_leaderboard, _ = efp.walk_forward_classification(
            dataset=training_dataset, feature_columns=selected_features,
            n_splits=3, test_size=20, gap=horizon,
            sample_weight=sample_weights, verbose=False,
        )
        per_horizon[horizon] = {
            "feature_count": len(selected_features),
            "training_rows": int(len(training_dataset)),
            "regression_leaderboard": _leaderboard_to_records(regression_leaderboard),
            "classification_leaderboard": _leaderboard_to_records(classification_leaderboard),
        }

    return {
        "frozen_at": datetime.utcnow().isoformat() + "Z",
        "master_data_csv": str(master_data_csv),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "horizons": per_horizon,
    }


def run_via_cli(drive_root: Path) -> dict[str, Any]:
    cmd = [sys.executable, str(PROJECT_ROOT / "eth_price_forecast.py"), "--drive-root", str(drive_root), "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return {
        "frozen_at": datetime.utcnow().isoformat() + "Z",
        "cli_invocation": " ".join(cmd),
        "cli_stdout_head": result.stdout[:2000],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", default="", help="Prebuilt master dataset CSV. If omitted, runs the full CLI.")
    parser.add_argument("--drive-root", default=str(PROJECT_ROOT), help="Drive root forwarded to eth_price_forecast.py when running via CLI.")
    args = parser.parse_args(argv)

    if args.master_data_csv:
        baseline = run_inline(Path(args.master_data_csv))
    else:
        baseline = run_via_cli(Path(args.drive_root))

    OUTPUT_PKL.write_bytes(pickle.dumps(baseline))
    OUTPUT_JSON.write_text(json.dumps(baseline, indent=2, default=str), encoding="utf-8")
    print(f"[frozen] {OUTPUT_PKL}")
    print(f"[frozen] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
