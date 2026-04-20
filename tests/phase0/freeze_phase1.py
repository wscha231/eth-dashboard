"""Re-freeze walk-forward metrics using the Phase 1.1 fold-safe feature selection.

This mirrors ``freeze_baseline.run_inline`` but:
- builds ``training_dataset`` with the FULL raw candidate feature list, and
- calls the walk-forward functions with ``fold_feature_selection=True`` so
  rank/prune runs per fold over train rows only.

The output goes to ``phase1_metrics.{pkl,json}``; the Phase 0 baseline pkl is
never overwritten, so the two files can be diffed side-by-side.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp

OUTPUT_PKL = Path(__file__).with_name("phase1_metrics.pkl")
OUTPUT_JSON = Path(__file__).with_name("phase1_metrics.json")


def _records(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard is None or leaderboard.empty:
        return []
    return leaderboard.to_dict(orient="records")


def run_inline(master_data_csv: Path) -> dict[str, Any]:
    market_data = efp.load_market_data_csv(master_data_csv)
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {master_data_csv}")

    state_cols = ["target_regime", "target_reversal_state", "target_bottom_reversal", "target_top_reversal"]
    horizons = [7, 30]
    cv_test_size = 60
    per_horizon: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        embargo = max(1, horizon // 2)
        feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
        full_mask = (
            feature_frame["target_return"].notna()
            & feature_frame["target_close"].notna()
            & feature_frame["eth_close"].notna()
        )
        full_dataset = feature_frame.loc[
            full_mask,
            raw_candidates + ["eth_close", "target_return", "target_close", *state_cols],
        ].copy()
        trimmed = efp.trim_training_window(full_dataset, interval="1d", horizon=horizon)
        training_mask = feature_frame.index.isin(trimmed.index) & full_mask

        training_dataset = feature_frame.loc[
            training_mask,
            raw_candidates + ["eth_close", "target_return", "target_close", *state_cols],
        ].copy()
        sample_weights = efp.build_time_decay_sample_weights(training_dataset.index, interval="1d", horizon=horizon)

        regression_leaderboard, _ = efp.walk_forward_leaderboard(
            dataset=training_dataset,
            feature_columns=raw_candidates,
            n_splits=3,
            test_size=cv_test_size,
            gap=horizon,
            sample_weight=sample_weights,
            verbose=False,
            fold_feature_selection=True,
            min_feature_coverage=0.03,
            embargo=embargo,
        )
        classification_leaderboard, _ = efp.walk_forward_classification(
            dataset=training_dataset,
            feature_columns=raw_candidates,
            n_splits=3,
            test_size=cv_test_size,
            gap=horizon,
            sample_weight=sample_weights,
            verbose=False,
            fold_feature_selection=True,
            min_feature_coverage=0.03,
            embargo=embargo,
        )

        per_horizon[horizon] = {
            "candidate_feature_count": len(raw_candidates),
            "training_rows": int(len(training_dataset)),
            "cv_test_size": cv_test_size,
            "embargo": embargo,
            "regression_leaderboard": _records(regression_leaderboard),
            "classification_leaderboard": _records(classification_leaderboard),
        }

    return {
        "frozen_at": datetime.utcnow().isoformat() + "Z",
        "master_data_csv": str(master_data_csv),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "mode": "phase1_fold_feature_selection+purged_embargo",
        "cv_test_size": cv_test_size,
        "horizons": per_horizon,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True, help="Prebuilt master dataset CSV.")
    args = parser.parse_args(argv)

    metrics = run_inline(Path(args.master_data_csv))
    OUTPUT_PKL.write_bytes(pickle.dumps(metrics))
    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"[frozen] {OUTPUT_PKL}")
    print(f"[frozen] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
