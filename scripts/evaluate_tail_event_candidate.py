"""Evaluate a three-day upside-tail head without touching daily forecasts.

Gate A uses six recent 30-day outer blocks and is infrastructure-only.  Gate B
uses expanding calendar-year blocks from 2019 onward and is the historical
promotion authority.  Every outer test prediction uses a model fitted only on
earlier rows, a three-day purge gap, prior-only sigmoid calibration, and a
prior-only alert threshold constrained by episode-level false alerts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as forecast
from forecasting.tail_events import (
    CORE_TAIL_FEATURES,
    DYNAMIC_LABEL_COLUMN,
    DYNAMIC_THRESHOLD_COLUMN,
    FUTURE_RETURN_COLUMN,
    MAX_CORE_FEATURES,
    MAX_FALSE_ALERT_EPISODES_PER_90_DAYS,
    PRIMARY_LABEL_COLUMN,
    TAIL_HORIZON_DAYS,
    build_episode_sample_weights,
    build_tail_targets,
    fit_sigmoid_calibrator,
    json_safe,
    moving_block_bootstrap_improvement,
    probability_metrics,
    select_alert_threshold,
)

CLIMATOLOGY_MODEL = "expanding_climatology"
HEURISTIC_MODEL = "tail_pressure_logistic"
LOGISTIC_MODEL = "episode_logistic"
CATBOOST_MODEL = "catboost_tail"
LIGHTGBM_MODEL = "lightgbm_tail"
BASE_MODELS = (CLIMATOLOGY_MODEL, HEURISTIC_MODEL, LOGISTIC_MODEL)
LABEL_COLUMNS = (PRIMARY_LABEL_COLUMN, DYNAMIC_LABEL_COLUMN)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class OuterFold:
    fold_id: str
    train_positions: np.ndarray
    test_positions: np.ndarray


@dataclass(frozen=True)
class NestedPartitions:
    inner_train_positions: np.ndarray
    calibration_positions: np.ndarray
    threshold_positions: np.ndarray


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def load_market_data(
    *,
    data_path: Path | None,
    git_ref: str | None,
    git_data_path: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if bool(data_path) == bool(git_ref):
        raise ValueError("Specify exactly one of --data-path or --data-git-ref")
    if git_ref:
        command = ["git", "show", f"{git_ref}:{git_data_path}"]
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
        raw = result.stdout
        source = f"git:{git_ref}:{git_data_path}"
    else:
        assert data_path is not None
        raw = data_path.read_bytes()
        source = str(data_path.resolve())

    frame = pd.read_csv(io.BytesIO(raw), index_col=0, parse_dates=True)
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index)).tz_localize(None)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Market data contains duplicate dates")
    if "eth_close" not in frame.columns:
        raise ValueError("Market data is missing eth_close")
    eth_close = pd.to_numeric(frame["eth_close"], errors="coerce")
    if eth_close.isna().any():
        missing = [
            timestamp.date().isoformat()
            for timestamp in frame.index[eth_close.isna()][:5]
        ]
        raise ValueError(f"Market data has missing eth_close rows: {missing}")
    expected = pd.date_range(frame.index.min(), frame.index.max(), freq="D")
    missing_dates = expected.difference(frame.index)
    if len(missing_dates):
        sample = [timestamp.date().isoformat() for timestamp in missing_dates[:5]]
        raise ValueError(f"Market data has date gaps: {sample}")
    return frame, {
        "source": source,
        "sha256": hashlib.sha256(raw).hexdigest(),
        "rows": len(frame),
        "start_date": frame.index.min().date().isoformat(),
        "end_date": frame.index.max().date().isoformat(),
    }


def make_outer_folds(
    index: pd.DatetimeIndex,
    *,
    profile: str,
    gap_days: int = TAIL_HORIZON_DAYS,
    smoke_fold_count: int = 6,
    smoke_test_days: int = 30,
    full_start_year: int = 2019,
) -> list[OuterFold]:
    timestamps = pd.DatetimeIndex(index)
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Evaluation index must be unique and increasing")
    folds: list[OuterFold] = []
    if profile == "smoke":
        required_test_rows = int(smoke_fold_count) * int(smoke_test_days)
        first_test = len(timestamps) - required_test_rows
        if first_test <= gap_days + 180:
            raise ValueError("Not enough history for six smoke folds")
        for fold_number in range(int(smoke_fold_count)):
            test_start = first_test + fold_number * int(smoke_test_days)
            test_stop = test_start + int(smoke_test_days)
            train_stop = test_start - int(gap_days)
            folds.append(
                OuterFold(
                    fold_id=f"smoke_{fold_number + 1:02d}",
                    train_positions=np.arange(0, train_stop, dtype=int),
                    test_positions=np.arange(test_start, test_stop, dtype=int),
                )
            )
    elif profile == "full":
        last_year = int(timestamps.max().year)
        for year in range(int(full_start_year), last_year + 1):
            test_positions = np.flatnonzero(timestamps.year == year).astype(int)
            if not len(test_positions):
                continue
            train_stop = int(test_positions[0]) - int(gap_days)
            if train_stop < 180:
                continue
            folds.append(
                OuterFold(
                    fold_id=str(year),
                    train_positions=np.arange(0, train_stop, dtype=int),
                    test_positions=test_positions,
                )
            )
    else:
        raise ValueError(f"Unknown profile: {profile}")

    if not folds:
        raise ValueError("No outer folds were generated")
    for fold in folds:
        if not len(fold.train_positions) or not len(fold.test_positions):
            raise ValueError(f"Fold {fold.fold_id} is empty")
        if int(fold.train_positions[-1]) + TAIL_HORIZON_DAYS >= int(
            fold.test_positions[0]
        ):
            raise ValueError(f"Fold {fold.fold_id} violates the target purge gap")
    return folds


def make_nested_partitions(
    train_positions: np.ndarray,
    labels: pd.Series,
    *,
    profile: str,
    gap_days: int = TAIL_HORIZON_DAYS,
    minimum_model_rows: int = 180,
) -> NestedPartitions:
    """Find deterministic prior-only calibration and threshold windows."""
    positions = np.asarray(train_positions, dtype=int)
    if len(positions) < minimum_model_rows + 2 * gap_days + 180:
        minimum_model_rows = max(120, len(positions) // 3)
    preferred_threshold = 180 if profile == "smoke" else 365
    minimum_tune_rows = 90
    max_threshold = (
        len(positions) - minimum_model_rows - 2 * gap_days - minimum_tune_rows
    )
    if max_threshold < minimum_tune_rows:
        raise ValueError("Outer training history is too short for nested tuning")
    threshold_sizes = []
    for value in (
        min(preferred_threshold, max_threshold),
        min(540, max_threshold),
        min(365, max_threshold),
        min(270, max_threshold),
        min(180, max_threshold),
        min(120, max_threshold),
        min(90, max_threshold),
    ):
        if value >= minimum_tune_rows and value not in threshold_sizes:
            threshold_sizes.append(int(value))
    calibration_preferences = (180, 270, 365, 540, 120, 90)
    y = pd.to_numeric(labels, errors="coerce")

    for threshold_size in threshold_sizes:
        threshold_start_offset = len(positions) - threshold_size
        threshold_positions = positions[threshold_start_offset:]
        calibration_end_offset = threshold_start_offset - int(gap_days)
        max_calibration = calibration_end_offset - int(gap_days) - minimum_model_rows
        if max_calibration < minimum_tune_rows:
            continue
        calibration_sizes: list[int] = []
        for value in (*calibration_preferences, max_calibration):
            clipped = min(int(value), int(max_calibration))
            if clipped >= minimum_tune_rows and clipped not in calibration_sizes:
                calibration_sizes.append(clipped)
        for calibration_size in calibration_sizes:
            calibration_start_offset = calibration_end_offset - calibration_size
            inner_train_stop = calibration_start_offset - int(gap_days)
            if inner_train_stop < minimum_model_rows:
                continue
            inner_train = positions[:inner_train_stop]
            calibration = positions[calibration_start_offset:calibration_end_offset]
            for name, partition in (
                ("inner training", inner_train),
                ("calibration", calibration),
                ("threshold", threshold_positions),
            ):
                partition_labels = y.iloc[partition].dropna()
                if partition_labels.nunique() != 2:
                    break
                if name == "threshold" and int(partition_labels.sum()) < 1:
                    break
            else:
                if int(inner_train[-1]) + TAIL_HORIZON_DAYS >= int(calibration[0]):
                    continue
                if int(calibration[-1]) + TAIL_HORIZON_DAYS >= int(
                    threshold_positions[0]
                ):
                    continue
                return NestedPartitions(
                    inner_train_positions=inner_train,
                    calibration_positions=calibration,
                    threshold_positions=threshold_positions,
                )
    raise ValueError("No valid prior-only calibration/threshold path with both classes")


def select_core_features(
    dataset: pd.DataFrame,
    fit_positions: np.ndarray,
    *,
    minimum_coverage: float = 0.80,
) -> tuple[list[str], list[dict[str, Any]]]:
    selected: list[str] = []
    rows: list[dict[str, Any]] = []
    fit = dataset.iloc[np.asarray(fit_positions, dtype=int)]
    for feature in CORE_TAIL_FEATURES:
        if feature not in fit.columns:
            rows.append(
                {"feature": feature, "coverage": 0.0, "unique": 0, "kept": False}
            )
            continue
        values = pd.to_numeric(fit[feature], errors="coerce")
        coverage = float(values.notna().mean())
        unique = int(values.nunique(dropna=True))
        # The predeclared heuristic itself contains a 90-day component.  The
        # first 2019 expanding fold has only about 180 inner-fit rows, so its
        # warm-up is expected and explicit.  Availability is still decided on
        # inner history only, and the model imputer handles the early rows.
        required_coverage = (
            min(float(minimum_coverage), 0.40)
            if feature == "eth_tail_event_pressure"
            else float(minimum_coverage)
        )
        kept = bool(coverage >= required_coverage and unique >= 2)
        rows.append(
            {
                "feature": feature,
                "coverage": coverage,
                "required_coverage": required_coverage,
                "unique": unique,
                "kept": kept,
            }
        )
        if kept:
            selected.append(feature)
    if not 8 <= len(selected) <= MAX_CORE_FEATURES:
        raise ValueError(
            f"Usable core feature count is {len(selected)}, expected 8..64"
        )
    return selected, rows


def candidate_models(nonlinear: str) -> tuple[str, ...]:
    if nonlinear == "none":
        return BASE_MODELS
    if nonlinear == "catboost":
        return (*BASE_MODELS, CATBOOST_MODEL)
    if nonlinear == "lightgbm":
        return (*BASE_MODELS, LIGHTGBM_MODEL)
    raise ValueError(f"Unsupported nonlinear model: {nonlinear}")


def build_estimator(model_name: str) -> Any:
    if model_name in (HEURISTIC_MODEL, LOGISTIC_MODEL):
        return Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.35 if model_name == LOGISTIC_MODEL else 0.75,
                        solver="lbfgs",
                        max_iter=1500,
                        random_state=42,
                    ),
                ),
            ]
        )
    if model_name == CATBOOST_MODEL:
        try:
            from catboost import CatBoostClassifier
        except ImportError as exc:  # pragma: no cover - exercised in minimal CI images
            raise RuntimeError(
                "catboost_tail requested but catboost is not installed"
            ) from exc
        return CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="Logloss",
            iterations=260,
            depth=5,
            learning_rate=0.035,
            l2_leaf_reg=8.0,
            random_strength=1.5,
            boosting_type="Ordered",
            has_time=True,
            random_seed=42,
            verbose=False,
            allow_writing_files=False,
            thread_count=2,
        )
    if model_name == LIGHTGBM_MODEL:
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - exercised in minimal CI images
            raise RuntimeError(
                "lightgbm_tail requested but lightgbm is not installed"
            ) from exc
        return LGBMClassifier(
            objective="binary",
            n_estimators=320,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=5,
            min_child_samples=30,
            subsample=0.85,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=0.10,
            reg_lambda=2.0,
            random_state=42,
            n_jobs=2,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
        )
    raise ValueError(f"No estimator for {model_name}")


def fit_estimator(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series,
) -> Any:
    if isinstance(model, Pipeline):
        model.fit(
            X,
            y.astype(int),
            model__sample_weight=sample_weight.to_numpy(dtype=float),
        )
    else:
        model.fit(
            X,
            y.astype(int),
            sample_weight=sample_weight.to_numpy(dtype=float),
        )
    return model


def positive_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Candidate did not produce two-class probabilities")
    classes = np.asarray(model.classes_)
    positive_columns = np.flatnonzero(classes == 1)
    if len(positive_columns) != 1:
        raise ValueError("Candidate classes do not contain one positive class")
    result = probabilities[:, int(positive_columns[0])]
    if not np.isfinite(result).all() or bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError("Candidate produced invalid probabilities")
    return result


def _feature_names_for_model(model_name: str, selected: list[str]) -> list[str]:
    if model_name == HEURISTIC_MODEL:
        if "eth_tail_event_pressure" not in selected:
            raise ValueError("Tail-pressure heuristic feature is unavailable")
        return ["eth_tail_event_pressure"]
    return selected


def evaluate_fold_label(
    dataset: pd.DataFrame,
    fold: OuterFold,
    label_column: str,
    *,
    profile: str,
    models: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = pd.to_numeric(dataset[label_column], errors="coerce")
    nested = make_nested_partitions(
        fold.train_positions,
        labels,
        profile=profile,
    )
    selected_features, coverage = select_core_features(
        dataset,
        nested.inner_train_positions,
    )
    y_inner = labels.iloc[nested.inner_train_positions]
    y_calibration = labels.iloc[nested.calibration_positions]
    y_threshold = labels.iloc[nested.threshold_positions]
    y_outer = labels.iloc[fold.train_positions]
    y_test = labels.iloc[fold.test_positions]
    if any(
        values.isna().any()
        for values in (y_inner, y_calibration, y_threshold, y_outer, y_test)
    ):
        raise ValueError(
            f"Fold {fold.fold_id}/{label_column} contains unavailable labels"
        )

    rows: list[dict[str, Any]] = []
    model_metadata: dict[str, Any] = {}
    for model_name in models:
        started = time.monotonic()
        if model_name == CLIMATOLOGY_MODEL:
            inner_probability = float(y_inner.mean())
            outer_probability = float(y_outer.mean())
            raw_test = np.full(len(fold.test_positions), outer_probability, dtype=float)
            calibrated_test = raw_test.copy()
            alert_threshold = float(np.nextafter(1.0, np.inf))
            threshold_report = {
                "threshold": alert_threshold,
                "max_false_alert_episodes_per_90_days": MAX_FALSE_ALERT_EPISODES_PER_90_DAYS,
                "note": "constant climatology has no alert operating point",
                "inner_probability": inner_probability,
            }
            calibrator_metadata = {"kind": "identity"}
            model_features: list[str] = []
        else:
            model_features = _feature_names_for_model(model_name, selected_features)
            X_inner = dataset.iloc[nested.inner_train_positions][model_features]
            X_calibration = dataset.iloc[nested.calibration_positions][model_features]
            X_threshold = dataset.iloc[nested.threshold_positions][model_features]
            X_outer = dataset.iloc[fold.train_positions][model_features]
            X_test = dataset.iloc[fold.test_positions][model_features]

            inner_model = build_estimator(model_name)
            inner_weight = build_episode_sample_weights(y_inner)
            fit_estimator(inner_model, X_inner, y_inner, inner_weight)
            raw_calibration = positive_probability(inner_model, X_calibration)
            raw_threshold = positive_probability(inner_model, X_threshold)
            calibrator = fit_sigmoid_calibrator(
                pd.Series(raw_calibration, index=y_calibration.index),
                y_calibration,
            )
            calibrated_threshold = calibrator.predict(
                pd.Series(raw_threshold, index=y_threshold.index)
            )
            if float(np.nanstd(calibrated_threshold)) <= 1e-10:
                raise ValueError(
                    f"Degenerate calibrated threshold probabilities: "
                    f"{fold.fold_id}/{label_column}/{model_name}"
                )
            threshold_report = select_alert_threshold(
                y_threshold,
                pd.Series(calibrated_threshold, index=y_threshold.index),
            )
            alert_threshold = float(threshold_report["threshold"])

            final_model = build_estimator(model_name)
            outer_weight = build_episode_sample_weights(y_outer)
            fit_estimator(final_model, X_outer, y_outer, outer_weight)
            raw_test = positive_probability(final_model, X_test)
            calibrated_test = calibrator.predict(
                pd.Series(raw_test, index=y_test.index)
            )
            calibrator_metadata = {
                "kind": "prior_only_sigmoid",
                "method": calibrator.method,
                "coefficient": calibrator.coefficient,
                "intercept": calibrator.intercept,
            }

        test_index = dataset.index[fold.test_positions]
        alerts = calibrated_test >= alert_threshold
        for offset, timestamp in enumerate(test_index):
            dataset_position = int(fold.test_positions[offset])
            rows.append(
                {
                    "label": label_column,
                    "fold_id": fold.fold_id,
                    "model": model_name,
                    "prediction_date": pd.Timestamp(timestamp).date().isoformat(),
                    "actual_label": int(y_test.iloc[offset]),
                    "future_return_3d": float(
                        dataset.iloc[dataset_position][FUTURE_RETURN_COLUMN]
                    ),
                    "dynamic_threshold": (
                        float(dataset.iloc[dataset_position][DYNAMIC_THRESHOLD_COLUMN])
                        if pd.notna(
                            dataset.iloc[dataset_position][DYNAMIC_THRESHOLD_COLUMN]
                        )
                        else None
                    ),
                    "raw_probability": float(raw_test[offset]),
                    "calibrated_probability": float(calibrated_test[offset]),
                    "alert_threshold": alert_threshold,
                    "alert": bool(alerts[offset]),
                }
            )
        model_metadata[model_name] = {
            "features": model_features,
            "feature_count": len(model_features),
            "calibrator": calibrator_metadata,
            "threshold_selection": threshold_report,
            "runtime_seconds": float(time.monotonic() - started),
        }

    metadata = {
        "fold_id": fold.fold_id,
        "label": label_column,
        "train_start": dataset.index[fold.train_positions[0]].date().isoformat(),
        "train_end": dataset.index[fold.train_positions[-1]].date().isoformat(),
        "test_start": dataset.index[fold.test_positions[0]].date().isoformat(),
        "test_end": dataset.index[fold.test_positions[-1]].date().isoformat(),
        "train_rows": len(fold.train_positions),
        "test_rows": len(fold.test_positions),
        "inner_train_rows": len(nested.inner_train_positions),
        "calibration_rows": len(nested.calibration_positions),
        "threshold_rows": len(nested.threshold_positions),
        "selected_core_features": selected_features,
        "feature_coverage": coverage,
        "models": model_metadata,
    }
    return rows, metadata


def _model_prediction_frame(
    predictions: pd.DataFrame,
    label_column: str,
    model_name: str,
) -> pd.DataFrame:
    frame = predictions.loc[
        (predictions["label"] == label_column) & (predictions["model"] == model_name)
    ].copy()
    frame["prediction_date"] = pd.to_datetime(frame["prediction_date"])
    frame = frame.sort_values("prediction_date").set_index("prediction_date")
    if frame.index.has_duplicates:
        raise ValueError(f"Duplicate predictions for {label_column}/{model_name}")
    return frame


def summarize_predictions(
    prediction_rows: list[dict[str, Any]],
    *,
    models: tuple[str, ...],
    bootstrap_samples: int,
) -> tuple[dict[str, Any], dict[str, dict[str, pd.DataFrame]]]:
    predictions = pd.DataFrame(prediction_rows)
    summaries: dict[str, Any] = {}
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for label_column in LABEL_COLUMNS:
        baseline = _model_prediction_frame(predictions, label_column, CLIMATOLOGY_MODEL)
        frames[label_column] = {CLIMATOLOGY_MODEL: baseline}
        label_summary: dict[str, Any] = {}
        for model_name in models:
            model_frame = _model_prediction_frame(predictions, label_column, model_name)
            if not model_frame.index.equals(baseline.index):
                raise ValueError(f"Unmatched OOF dates for {label_column}/{model_name}")
            frames[label_column][model_name] = model_frame
            metrics = probability_metrics(
                model_frame["actual_label"],
                model_frame["calibrated_probability"],
                baseline_probabilities=baseline["calibrated_probability"],
                alerts=model_frame["alert"].astype(float),
            )
            metrics["bootstrap_vs_climatology"] = moving_block_bootstrap_improvement(
                model_frame["actual_label"],
                model_frame["calibrated_probability"],
                baseline["calibrated_probability"],
                block_length=7,
                samples=int(bootstrap_samples),
            )
            block_metrics: dict[str, Any] = {}
            for fold_id, fold_frame in model_frame.groupby("fold_id", sort=False):
                baseline_fold = baseline.loc[fold_frame.index]
                block_metrics[str(fold_id)] = probability_metrics(
                    fold_frame["actual_label"],
                    fold_frame["calibrated_probability"],
                    baseline_probabilities=baseline_fold["calibrated_probability"],
                    alerts=fold_frame["alert"].astype(float),
                )
            metrics["blocks"] = block_metrics
            label_summary[model_name] = metrics
        summaries[label_column] = label_summary
    return summaries, frames


def build_gate_report(
    *,
    profile: str,
    models: tuple[str, ...],
    metrics: dict[str, Any],
    frames: dict[str, dict[str, pd.DataFrame]],
    expected_fold_count: int,
    runtime_seconds: float,
    max_runtime_seconds: float,
) -> dict[str, Any]:
    failures: list[str] = []
    warnings: list[str] = []
    for label_column in LABEL_COLUMNS:
        baseline_n = metrics[label_column][CLIMATOLOGY_MODEL]["n"]
        for model_name in models:
            model_metrics = metrics[label_column][model_name]
            model_frame = frames[label_column][model_name]
            if model_metrics["n"] != baseline_n:
                failures.append(f"unmatched rows: {label_column}/{model_name}")
            if model_frame["fold_id"].nunique() != expected_fold_count:
                failures.append(f"incomplete folds: {label_column}/{model_name}")
            probabilities = model_frame["calibrated_probability"].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all() or bool(
                ((probabilities < 0.0) | (probabilities > 1.0)).any()
            ):
                failures.append(f"invalid probabilities: {label_column}/{model_name}")
            if model_name != CLIMATOLOGY_MODEL:
                if (
                    model_metrics["average_precision"]
                    <= metrics[label_column][CLIMATOLOGY_MODEL]["average_precision"]
                ):
                    warnings.append(
                        f"{label_column}/{model_name} AP did not beat climatology"
                    )
                brier_skill = model_metrics["brier_skill"]
                if brier_skill is None or brier_skill <= 0.0:
                    warnings.append(
                        f"{label_column}/{model_name} has non-positive Brier skill"
                    )
    if runtime_seconds > max_runtime_seconds:
        failures.append(
            f"runtime {runtime_seconds:.1f}s exceeded budget {max_runtime_seconds:.1f}s"
        )

    infrastructure_status = "PASS" if not failures else "FAIL"
    report: dict[str, Any] = {
        "profile": profile,
        "gate_status": infrastructure_status,
        "infrastructure_status": infrastructure_status,
        "promotion_status": "NOT_EVALUATED" if profile == "smoke" else "FAIL",
        "failures": failures,
        "warnings": sorted(set(warnings)),
        "runtime_seconds": runtime_seconds,
        "runtime_budget_seconds": max_runtime_seconds,
    }
    if profile == "smoke" or failures:
        return report

    primary = metrics[PRIMARY_LABEL_COLUMN]
    non_climatology = [model for model in models if model != CLIMATOLOGY_MODEL]
    winner = max(
        non_climatology,
        key=lambda model: (
            primary[model]["average_precision"],
            -primary[model]["brier_score"],
        ),
    )
    baseline_candidates = [
        model for model in BASE_MODELS if model in models and model != winner
    ]
    best_baseline = min(
        baseline_candidates,
        key=lambda model: primary[model]["brier_score"],
    )
    winner_frame = frames[PRIMARY_LABEL_COLUMN][winner]
    baseline_frame = frames[PRIMARY_LABEL_COLUMN][best_baseline]
    paired_bootstrap = moving_block_bootstrap_improvement(
        winner_frame["actual_label"],
        winner_frame["calibrated_probability"],
        baseline_frame["calibrated_probability"],
        block_length=7,
        samples=2000,
    )
    winner_primary = primary[winner]
    winner_dynamic = metrics[DYNAMIC_LABEL_COLUMN][winner]
    climatology_ap = primary[CLIMATOLOGY_MODEL]["average_precision"]
    reference_ap = max(
        primary[model]["average_precision"] for model in BASE_MODELS if model in models
    )
    primary_recall = winner_primary["events"]["episode_recall"]
    dynamic_recall = winner_dynamic["events"]["episode_recall"]
    gross_failures = []
    for fold_id, block in winner_primary["blocks"].items():
        candidate_brier = block["brier_score"]
        baseline_brier = block["baseline_brier_score"]
        if (
            baseline_brier is not None
            and candidate_brier > baseline_brier * 1.5
            and candidate_brier - baseline_brier > 0.02
        ):
            gross_failures.append(fold_id)

    checks = {
        "average_precision_20pct_above_climatology": bool(
            winner_primary["average_precision"] >= 1.20 * climatology_ap
        ),
        "average_precision_not_below_reference": bool(
            winner_primary["average_precision"] + 1e-12 >= reference_ap
        ),
        "positive_brier_skill": bool(
            winner_primary["brier_skill"] is not None
            and winner_primary["brier_skill"] > 0.0
        ),
        "episode_recall_at_least_35pct": bool(
            primary_recall is not None and primary_recall >= 0.35
        ),
        "false_alert_budget": bool(
            winner_primary["events"]["false_alert_episodes_per_90_days"]
            <= MAX_FALSE_ALERT_EPISODES_PER_90_DAYS
        ),
        "bootstrap_probability_at_least_90pct": bool(
            paired_bootstrap["brier_probability_improvement"] >= 0.90
        ),
        "no_probability_inversion": bool(
            winner_primary["roc_auc"] is not None and winner_primary["roc_auc"] >= 0.50
        ),
        "no_gross_calendar_calibration_failure": not gross_failures,
        "dynamic_nonnegative_brier_skill": bool(
            winner_dynamic["brier_skill"] is not None
            and winner_dynamic["brier_skill"] >= 0.0
        ),
        "dynamic_recall_drop_within_10pp": bool(
            primary_recall is not None
            and dynamic_recall is not None
            and dynamic_recall >= primary_recall - 0.10
        ),
    }
    promotion_status = "PASS" if all(checks.values()) else "FAIL"
    report.update(
        {
            "gate_status": promotion_status,
            "promotion_status": promotion_status,
            "winner": winner,
            "best_paired_baseline": best_baseline,
            "checks": checks,
            "paired_bootstrap": paired_bootstrap,
            "gross_calendar_failure_folds": gross_failures,
        }
    )
    return report


def render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Three-day upside-tail evaluation",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Data: `{payload['data']['start_date']}` to `{payload['data']['end_date']}`",
        f"- Infrastructure: **{gate['infrastructure_status']}**",
        f"- Promotion: **{gate['promotion_status']}**",
        f"- Runtime: {payload['runtime_seconds']:.1f}s",
        "",
    ]
    for label_column in LABEL_COLUMNS:
        lines.extend(
            [
                f"## {label_column}",
                "",
                "| Model | AP | Brier skill | Episode recall | False alerts / 90d |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for model_name, metrics in payload["metrics"][label_column].items():
            events = metrics["events"]
            brier_skill = metrics["brier_skill"]
            recall = events["episode_recall"]
            false_rate = events["false_alert_episodes_per_90_days"]
            lines.append(
                f"| {model_name} | {metrics['average_precision']:.4f} | "
                f"{brier_skill if brier_skill is not None else float('nan'):.4f} | "
                f"{recall if recall is not None else float('nan'):.4f} | "
                f"{false_rate if false_rate is not None else float('nan'):.2f} |"
            )
        lines.append("")
    if gate["failures"]:
        lines.extend(
            ["## Failures", "", *[f"- {item}" for item in gate["failures"]], ""]
        )
    if gate["warnings"]:
        lines.extend(
            ["## Warnings", "", *[f"- {item}" for item in gate["warnings"]], ""]
        )
    return "\n".join(lines)


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    data, data_metadata = load_market_data(
        data_path=args.data_path,
        git_ref=args.data_git_ref,
        git_data_path=args.git_data_path,
    )
    feature_frame, _ = forecast.build_features(data, TAIL_HORIZON_DAYS)
    targets = build_tail_targets(data["eth_close"])
    for column in targets.columns:
        feature_frame[column] = targets[column]
    # Use one matched calendar for both label definitions.  The shifted
    # volatility label needs a short warm-up, and neither label may silently
    # coerce those unavailable origins to negatives.
    valid_labels = feature_frame[list(LABEL_COLUMNS)].notna().all(axis=1)
    dataset = feature_frame.loc[valid_labels].copy()
    folds = make_outer_folds(dataset.index, profile=args.profile)
    models = candidate_models(args.nonlinear)
    prediction_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for fold_number, fold in enumerate(folds, start=1):
        print(
            f"[{args.profile}] fold {fold_number}/{len(folds)} {fold.fold_id}",
            flush=True,
        )
        for label_column in LABEL_COLUMNS:
            rows, metadata = evaluate_fold_label(
                dataset,
                fold,
                label_column,
                profile=args.profile,
                models=models,
            )
            prediction_rows.extend(rows)
            fold_metadata.append(metadata)
    metrics, frames = summarize_predictions(
        prediction_rows,
        models=models,
        bootstrap_samples=args.bootstrap_samples,
    )
    runtime_seconds = float(time.monotonic() - started)
    max_runtime_seconds = float(
        args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else (600.0 if args.profile == "smoke" else 2700.0)
    )
    gate = build_gate_report(
        profile=args.profile,
        models=models,
        metrics=metrics,
        frames=frames,
        expected_fold_count=len(folds),
        runtime_seconds=runtime_seconds,
        max_runtime_seconds=max_runtime_seconds,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "mode": "offline_tail_event_candidate",
        "profile": args.profile,
        "data": data_metadata,
        "label_manifest": {
            "horizon_days": TAIL_HORIZON_DAYS,
            "primary": {"column": PRIMARY_LABEL_COLUMN, "return_threshold": 0.12},
            "dynamic": {
                "column": DYNAMIC_LABEL_COLUMN,
                "formula": "clip(2 * shifted_trailing_30d_daily_vol * sqrt(3), 0.10, 0.25)",
            },
        },
        "split_manifest": {
            "outer_profile": args.profile,
            "outer_fold_count": len(folds),
            "gap_days": TAIL_HORIZON_DAYS,
            "inner_calibration": "prior-only sigmoid",
            "inner_threshold": "prior-only, max 3 false alert episodes per 90 days",
        },
        "candidate_manifest": {
            "models": list(models),
            "nonlinear": args.nonlinear,
            "core_feature_cap": MAX_CORE_FEATURES,
            "core_feature_allowlist": list(CORE_TAIL_FEATURES),
        },
        "folds": fold_metadata,
        "metrics": metrics,
        "gate": gate,
        "predictions": prediction_rows,
        "runtime_seconds": runtime_seconds,
        "peak_rss_mb": float(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        ),
    }
    return json_safe(payload)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--data-path", type=Path)
    source.add_argument("--data-git-ref")
    parser.add_argument(
        "--git-data-path",
        default="lake/gold/eth_master_daily.csv",
    )
    parser.add_argument(
        "--nonlinear",
        choices=("none", "catboost", "lightgbm"),
        default="catboost",
        help="Evaluate at most one nonlinear candidate per run.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=500)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_evaluation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if payload["gate"]["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
