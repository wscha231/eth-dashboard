"""Phase 5 Leave-One-In: identify which Phase 3B macro composite caused the
h=30 classification regression.

Phase 4B production dropped all 6 Phase 3B composites together (brier regressed
from 0.173 -> 0.266 when all six were present). This script adds them back ONE
AT A TIME over the Phase 4B baseline to isolate the culprit(s).

Baseline (no composites added): expected h=30 cls brier ~0.173, h=30 reg rmse ~967
matching freeze_phase4b_production_metrics.json.

Interpretation of each variant vs baseline:
  - "safe" features: metrics stay near baseline -> candidate for Phase 5 re-inclusion.
  - "culprit" features: brier drifts toward 0.266 or AUC drifts toward 0.585.

Output: phase5_loo_metrics.{pkl,json}.
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

OUTPUT_PKL = Path(__file__).with_name("phase5_loo_metrics.pkl")
OUTPUT_JSON = Path(__file__).with_name("phase5_loo_metrics.json")

EQUAL_WEIGHT_REGRESSION_MODEL = "trimmed_regression_ensemble_equal"
EQUAL_WEIGHT_CLASSIFICATION_MODEL = "trimmed_classification_ensemble_equal"

# Order matches freeze_phase4b_production.PHASE3B_FEATURES_TO_DROP.
PHASE3B_FEATURES: list[str] = [
    "fred_yield_curve_change_20",
    "fred_real_yield_10y",
    "fred_real_yield_10y_z_90",
    "fred_credit_stress",
    "fred_credit_stress_change_20",
    "macro_asymmetry_20",
]


def _compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def _compute_phase3b_feature(feature_frame: pd.DataFrame, name: str) -> pd.Series | None:
    """Reconstruct one Phase 3B composite from base columns present in
    feature_frame. Returns None if the required base columns are missing.

    Formulas match the code that lived in build_features before the Phase 3B
    ablation removal (see the comment block in eth_price_forecast.build_features).
    """
    cols = feature_frame.columns
    if name == "fred_yield_curve_change_20":
        if "fred_yield_curve_10y_2y" in cols:
            curve = pd.to_numeric(feature_frame["fred_yield_curve_10y_2y"], errors="coerce")
            return curve.diff(20)
        return None
    if name in {"fred_real_yield_10y", "fred_real_yield_10y_z_90"}:
        if {"fred_treasury_10y", "fred_breakeven_10y"}.issubset(cols):
            real_yield = (
                pd.to_numeric(feature_frame["fred_treasury_10y"], errors="coerce")
                - pd.to_numeric(feature_frame["fred_breakeven_10y"], errors="coerce")
            )
            if name == "fred_real_yield_10y":
                return real_yield
            return _compute_rolling_zscore(real_yield, 90)
        return None
    if name in {"fred_credit_stress", "fred_credit_stress_change_20"}:
        if {"fred_hy_oas", "fred_bbb_oas"}.issubset(cols):
            credit_stress = 0.5 * (
                pd.to_numeric(feature_frame["fred_hy_oas"], errors="coerce")
                + pd.to_numeric(feature_frame["fred_bbb_oas"], errors="coerce")
            )
            if name == "fred_credit_stress":
                return credit_stress
            return credit_stress.diff(20)
        return None
    if name == "macro_asymmetry_20":
        if {"dxy_return_20", "tnx_return_20"}.issubset(cols):
            dxy = pd.to_numeric(feature_frame["dxy_return_20"], errors="coerce")
            tnx = pd.to_numeric(feature_frame["tnx_return_20"], errors="coerce")
            return dxy - tnx
        return None
    raise ValueError(f"Unknown Phase 3B feature: {name}")


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


def _run_variant(
    market_data: pd.DataFrame,
    feature_to_add: str | None,
    horizons: list[int],
    cv_test_size: int,
    state_cols: list[str],
) -> dict[int, dict[str, Any]]:
    """Run the walk-forward pipeline either at pure Phase 4B baseline (if
    feature_to_add is None) or with exactly one Phase 3B composite added back.
    """
    per_horizon: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        embargo = max(1, horizon // 2)
        feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
        feature_frame = feature_frame.copy()
        candidates = list(raw_candidates)

        added_feature_nonnull: int | None = None
        if feature_to_add is not None:
            series = _compute_phase3b_feature(feature_frame, feature_to_add)
            if series is None:
                raise RuntimeError(
                    f"Cannot compute {feature_to_add}: base columns missing"
                )
            feature_frame[feature_to_add] = series.reindex(feature_frame.index)
            candidates.append(feature_to_add)
            added_feature_nonnull = int(feature_frame[feature_to_add].notna().sum())

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
            "added_feature": feature_to_add,
            "added_feature_nonnull": added_feature_nonnull,
            "training_rows": int(len(training_dataset)),
            "cv_test_size": cv_test_size,
            "embargo": embargo,
            "regression_ensemble_members": reg_members,
            "classification_ensemble_members": cls_members,
            "regression_leaderboard": _records(reg_lb),
            "classification_leaderboard": _records(cls_lb),
        }

    return per_horizon


def run_inline(
    master_data_csv: Path,
    *,
    features: list[str] | None = None,
    skip_baseline: bool = False,
) -> dict[str, Any]:
    market_data = efp.load_market_data_csv(master_data_csv)
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {master_data_csv}")

    state_cols = ["target_regime", "target_reversal_state", "target_bottom_reversal", "target_top_reversal"]
    horizons = [7, 30]
    cv_test_size = 60

    variants: dict[str, dict[int, dict[str, Any]]] = {}

    if not skip_baseline:
        print("[phase5_loo] variant=baseline (Phase 4B, no Phase 3B composites)", flush=True)
        variants["baseline"] = _run_variant(
            market_data, feature_to_add=None, horizons=horizons,
            cv_test_size=cv_test_size, state_cols=state_cols,
        )

    features_to_run = features if features is not None else PHASE3B_FEATURES
    for feature in features_to_run:
        if feature not in PHASE3B_FEATURES:
            raise SystemExit(f"Unknown feature {feature!r}; must be one of {PHASE3B_FEATURES}")
        print(f"[phase5_loo] variant=add_{feature}", flush=True)
        variants[f"add_{feature}"] = _run_variant(
            market_data, feature_to_add=feature, horizons=horizons,
            cv_test_size=cv_test_size, state_cols=state_cols,
        )

    return {
        "frozen_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "master_data_csv": str(master_data_csv),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "mode": "phase5_leave_one_in_phase3b_ablation",
        "cv_test_size": cv_test_size,
        "phase3b_features_tested": features_to_run,
        "baseline_included": not skip_baseline,
        "variants": variants,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True)
    parser.add_argument(
        "--features", nargs="*", default=None,
        help="Subset of Phase 3B features to test (default: all 6).",
    )
    parser.add_argument(
        "--skip-baseline", action="store_true",
        help="Skip the Phase 4B baseline run (useful when extending an earlier run).",
    )
    parser.add_argument(
        "--output-suffix", default="",
        help="Optional suffix for output files, e.g. --output-suffix=_smoke",
    )
    args = parser.parse_args(argv)
    metrics = run_inline(
        Path(args.master_data_csv),
        features=args.features,
        skip_baseline=args.skip_baseline,
    )

    out_pkl = OUTPUT_PKL.with_name(f"phase5_loo_metrics{args.output_suffix}.pkl")
    out_json = OUTPUT_JSON.with_name(f"phase5_loo_metrics{args.output_suffix}.json")
    out_pkl.write_bytes(pickle.dumps(metrics))
    out_json.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"[frozen] {out_pkl}")
    print(f"[frozen] {out_json}")


if __name__ == "__main__":
    main()
