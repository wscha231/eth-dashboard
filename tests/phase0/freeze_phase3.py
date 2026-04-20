"""Re-freeze walk-forward metrics under Phase 3 derivatives-regime features.

Phase 3 adds five domain-specific derived features from existing Deribit
columns (funding z-score, option IV z-score, PCR z-score, futures basis,
OI change). Everything else — purged+embargo CV, tight member selection,
best-single guard — is inherited from Phase 2.5. Output lands in
``phase3_metrics.{pkl,json}``.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp

OUTPUT_PKL = Path(__file__).with_name("phase3_metrics.pkl")
OUTPUT_JSON = Path(__file__).with_name("phase3_metrics.json")

EQUAL_WEIGHT_REGRESSION_MODEL = "trimmed_regression_ensemble_equal"
EQUAL_WEIGHT_CLASSIFICATION_MODEL = "trimmed_classification_ensemble_equal"

PHASE3_FEATURE_NAMES = [
    "deribit_funding_8h_z_90",
    "deribit_option_iv_z_90",
    "deribit_pcr_z_90",
    "deribit_futures_basis_ratio",
    "deribit_futures_basis_z_30",
    "deribit_future_oi_change_7",
]


def _records(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard is None or leaderboard.empty:
        return []
    return leaderboard.to_dict(orient="records")


def _append_equal_weight_regression_row(
    leaderboard: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    component_models: list[str],
) -> pd.DataFrame:
    available = [m for m in component_models if f"{m}_pred_return" in oof_predictions.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof_predictions[[f"{m}_pred_return" for m in available]]
    )
    valid = blended.dropna()
    if valid.empty:
        return leaderboard
    matched = dataset.loc[valid.index]
    summary = efp.evaluate_predictions(
        current_close=matched["eth_close"].to_numpy(),
        actual_return=matched["target_return"].to_numpy(),
        predicted_return=valid.to_numpy(),
    )
    summary["model"] = EQUAL_WEIGHT_REGRESSION_MODEL
    summary["folds"] = float(min(len(available), 4))
    summary["component_models"] = "|".join(available)
    return pd.concat([leaderboard, pd.DataFrame([summary])], ignore_index=True)


def _append_equal_weight_classification_row(
    leaderboard: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    horizon: int,
    component_models: list[str],
) -> pd.DataFrame:
    available = [m for m in component_models if f"{m}_prob_up" in oof_predictions.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof_predictions[[f"{m}_prob_up" for m in available]]
    ).clip(lower=0.0, upper=1.0)
    valid = blended.dropna()
    if valid.empty:
        return leaderboard
    actual = (dataset.loc[valid.index, "target_return"] > 0).astype(int)
    threshold, metrics = efp.choose_classification_evaluation_threshold(
        actual_label=actual, probability_up=valid, horizon=horizon,
    )
    metrics["model"] = EQUAL_WEIGHT_CLASSIFICATION_MODEL
    metrics["folds"] = float(min(len(available), 4))
    metrics["signal_threshold"] = float(threshold)
    metrics["component_models"] = "|".join(available)
    return pd.concat([leaderboard, pd.DataFrame([metrics])], ignore_index=True)


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
        phase3_present = [c for c in PHASE3_FEATURE_NAMES if c in feature_frame.columns]
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

        regression_leaderboard, regression_oof = efp.walk_forward_leaderboard(
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
        classification_leaderboard, classification_oof = efp.walk_forward_classification(
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

        regression_members = efp.select_trimmed_regression_ensemble_members(
            regression_leaderboard, horizon=horizon,
        )
        classification_members = efp.select_trimmed_classification_ensemble_members(
            classification_leaderboard, horizon=horizon,
        )

        regression_leaderboard, regression_oof = efp.append_trimmed_regression_ensemble_candidate(
            regression_leaderboard, regression_oof, training_dataset,
            horizon=horizon, component_models=regression_members,
        )
        classification_leaderboard, classification_oof = efp.append_trimmed_classification_ensemble_candidate(
            classification_leaderboard, classification_oof, training_dataset,
            horizon=horizon, component_models=classification_members,
        )

        regression_leaderboard = _append_equal_weight_regression_row(
            regression_leaderboard, regression_oof, training_dataset, regression_members,
        )
        classification_leaderboard = _append_equal_weight_classification_row(
            classification_leaderboard, classification_oof, training_dataset,
            horizon=horizon, component_models=classification_members,
        )

        per_horizon[horizon] = {
            "candidate_feature_count": len(raw_candidates),
            "phase3_features_present": phase3_present,
            "training_rows": int(len(training_dataset)),
            "cv_test_size": cv_test_size,
            "embargo": embargo,
            "regression_ensemble_members": regression_members,
            "classification_ensemble_members": classification_members,
            "regression_leaderboard": _records(regression_leaderboard),
            "classification_leaderboard": _records(classification_leaderboard),
        }

    return {
        "frozen_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "master_data_csv": str(master_data_csv),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "mode": "phase3_derivatives_regime_features",
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
