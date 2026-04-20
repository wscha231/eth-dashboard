"""Phase 5 production freeze: Phase 4B (synergy-aware selection + all Phase 3B
composites removed) PLUS any Phase 3B composites the Leave-One-In ablation
marked SAFE.

The allowlist below starts EMPTY — fill it only with features whose LOO variant
kept h=30 cls brier / AUC within tolerance of baseline (see
analyze_phase5_loo.py verdicts). Running this before the LOO results are
analysed will behave identically to Phase 4B production.

Output: phase5_production_metrics.{pkl,json}.
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

# Reuse the Phase 5 LOO inline feature computation.
sys.path.insert(0, str(Path(__file__).parent))
from freeze_phase5_loo import _compute_phase3b_feature, PHASE3B_FEATURES  # noqa: E402

OUTPUT_PKL = Path(__file__).with_name("phase5_production_metrics.pkl")
OUTPUT_JSON = Path(__file__).with_name("phase5_production_metrics.json")

EQUAL_WEIGHT_REGRESSION_MODEL = "trimmed_regression_ensemble_equal"
EQUAL_WEIGHT_CLASSIFICATION_MODEL = "trimmed_classification_ensemble_equal"

# Populated after running analyze_phase5_loo.py on the full LOO output
# (2026-04-20). Verdict summary:
#
#   SAFE    : fred_real_yield_10y, fred_credit_stress
#   CULPRIT : fred_yield_curve_change_20, fred_real_yield_10y_z_90,
#             fred_credit_stress_change_20, macro_asymmetry_20
#
# Pattern: every culprit is a derivative / z-score / asymmetry form of a slow
# FRED series, while every safe feature is a raw level. Derivatives on
# monthly-cadence macro data amplify noise — Phase 5 keeps only the levels.
PHASE5_ALLOWLIST: set[str] = {
    "fred_real_yield_10y",
    "fred_credit_stress",
}


def _records(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard is None or leaderboard.empty:
        return []
    return leaderboard.to_dict(orient="records")


def _append_equal_weight_regression_row(leaderboard, oof, dataset, members):
    available = [m for m in members if f"{m}_pred_return" in oof.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(oof[[f"{m}_pred_return" for m in available]])
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


def _append_equal_weight_classification_row(leaderboard, oof, dataset, horizon, members):
    available = [m for m in members if f"{m}_prob_up" in oof.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof[[f"{m}_prob_up" for m in available]]
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

    unknown = PHASE5_ALLOWLIST - set(PHASE3B_FEATURES)
    if unknown:
        raise SystemExit(f"PHASE5_ALLOWLIST contains unknown features: {unknown}")

    state_cols = ["target_regime", "target_reversal_state", "target_bottom_reversal", "target_top_reversal"]
    horizons = [7, 30]
    cv_test_size = 60
    per_horizon: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        embargo = max(1, horizon // 2)
        feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
        feature_frame = feature_frame.copy()
        candidates = list(raw_candidates)

        added: list[str] = []
        for feature in sorted(PHASE5_ALLOWLIST):
            series = _compute_phase3b_feature(feature_frame, feature)
            if series is None:
                raise RuntimeError(f"Base columns missing for {feature}")
            feature_frame[feature] = series.reindex(feature_frame.index)
            candidates.append(feature)
            added.append(feature)

        full_mask = (
            feature_frame["target_return"].notna()
            & feature_frame["target_close"].notna()
            & feature_frame["eth_close"].notna()
        )
        full_dataset = feature_frame.loc[
            full_mask,
            candidates + ["eth_close", "target_return", "target_close", *state_cols],
        ].copy()
        trimmed = efp.trim_training_window(full_dataset, interval="1d", horizon=horizon)
        training_mask = feature_frame.index.isin(trimmed.index) & full_mask
        training_dataset = feature_frame.loc[
            training_mask,
            candidates + ["eth_close", "target_return", "target_close", *state_cols],
        ].copy()
        sample_weights = efp.build_time_decay_sample_weights(
            training_dataset.index, interval="1d", horizon=horizon,
        )

        reg_lb, reg_oof = efp.walk_forward_leaderboard(
            dataset=training_dataset, feature_columns=candidates,
            n_splits=3, test_size=cv_test_size, gap=horizon,
            sample_weight=sample_weights, verbose=False,
            fold_feature_selection=True, min_feature_coverage=0.03, embargo=embargo,
        )
        cls_lb, cls_oof = efp.walk_forward_classification(
            dataset=training_dataset, feature_columns=candidates,
            n_splits=3, test_size=cv_test_size, gap=horizon,
            sample_weight=sample_weights, verbose=False,
            fold_feature_selection=True, min_feature_coverage=0.03, embargo=embargo,
        )

        reg_members = efp.select_trimmed_regression_ensemble_members(
            reg_lb, horizon=horizon, oof_predictions=reg_oof,
        )
        cls_members = efp.select_trimmed_classification_ensemble_members(
            cls_lb, horizon=horizon, oof_predictions=cls_oof,
        )

        reg_lb, reg_oof = efp.append_trimmed_regression_ensemble_candidate(
            reg_lb, reg_oof, training_dataset,
            horizon=horizon, component_models=reg_members,
        )
        cls_lb, cls_oof = efp.append_trimmed_classification_ensemble_candidate(
            cls_lb, cls_oof, training_dataset,
            horizon=horizon, component_models=cls_members,
        )

        reg_lb = _append_equal_weight_regression_row(reg_lb, reg_oof, training_dataset, reg_members)
        cls_lb = _append_equal_weight_classification_row(
            cls_lb, cls_oof, training_dataset, horizon, cls_members,
        )

        per_horizon[horizon] = {
            "candidate_feature_count": len(candidates),
            "phase5_allowlist_features": added,
            "training_rows": int(len(training_dataset)),
            "cv_test_size": cv_test_size,
            "embargo": embargo,
            "regression_ensemble_members": reg_members,
            "classification_ensemble_members": cls_members,
            "regression_leaderboard": _records(reg_lb),
            "classification_leaderboard": _records(cls_lb),
        }

    return {
        "frozen_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "master_data_csv": str(master_data_csv),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "mode": "phase5_production_phase4b_plus_allowlisted_phase3b",
        "phase5_allowlist": sorted(PHASE5_ALLOWLIST),
        "cv_test_size": cv_test_size,
        "horizons": per_horizon,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True)
    args = parser.parse_args(argv)
    metrics = run_inline(Path(args.master_data_csv))
    OUTPUT_PKL.write_bytes(pickle.dumps(metrics))
    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"[frozen] {OUTPUT_PKL}")
    print(f"[frozen] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
