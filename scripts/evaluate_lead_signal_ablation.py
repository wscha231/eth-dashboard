"""Evaluate matched-date lead-signal ablations without touching daily forecasts.

Gate A uses six recent 30-day outer blocks and is infrastructure-only.  Gate B
uses expanding calendar-year blocks from 2022 onward and is the historical
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as forecast
from forecasting.tail_evaluation import (
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

PRIMARY_LARGE_MOVE_COLUMN = "large_move_primary"
DYNAMIC_LARGE_MOVE_COLUMN = "large_move_dynamic"
DIRECTION_UP_COLUMN = "direction_up"
PRIMARY_CLASS_COLUMN = "tail_class_primary"
DYNAMIC_CLASS_COLUMN = "tail_class_dynamic"

FEATURE_SET_GROUPS: dict[str, tuple[str, ...]] = {
    "core": (),
    "order_flow": ("order_flow",),
    "liquidity": ("ethereum_liquidity",),
    "market_leads": (
        "order_flow",
        "leverage_basis",
        "intraday_risk",
        "cross_asset_leadership",
    ),
    "all_leads": (
        "order_flow",
        "leverage_basis",
        "intraday_risk",
        "cross_asset_leadership",
        "ethereum_liquidity",
    ),
}


@dataclass(frozen=True)
class CandidateSpec:
    name: str
    estimator_family: str
    target_family: str
    feature_set: str

    @property
    def source_augmented(self) -> bool:
        return self.feature_set != "core"

    @property
    def baseline_name(self) -> str:
        return (
            f"{self.target_family}_{self.estimator_family}_core"
            if self.name != CLIMATOLOGY_MODEL
            else CLIMATOLOGY_MODEL
        )


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


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_lead_signal_features(
    *,
    feature_path: Path,
    manifest_path: Path,
    readiness_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]], dict[str, Any]]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    feature_hash = _sha256_path(feature_path)
    manifest_hash = _sha256_path(manifest_path)
    if manifest.get("scope") != "offline_lead_signal_daily_features":
        raise ValueError("Unexpected lead-signal feature manifest scope")
    if manifest["feature_table"]["sha256"] != feature_hash:
        raise ValueError("Lead-signal feature table SHA-256 mismatch")
    if readiness.get("feature_manifest_sha256") != manifest_hash:
        raise ValueError("Lead-signal readiness and manifest SHA-256 disagree")
    if readiness.get("decision") != "pass_for_pr3_offline_evaluation":
        raise RuntimeError("PR 2 readiness does not approve PR 3 offline evaluation")
    if readiness["gate"].get("production_use_approved") is not False:
        raise ValueError(
            "Offline lead-signal evidence unexpectedly approves production"
        )

    frame = pd.read_csv(feature_path, parse_dates=["date"]).set_index("date")
    frame.index = pd.DatetimeIndex(frame.index).tz_localize(None)
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("Lead-signal feature dates must be unique")
    if frame.index.max() != pd.Timestamp(manifest["as_of_date"]):
        raise ValueError("Lead-signal feature cutoff does not match its manifest")
    available = pd.to_datetime(frame["feature_available_at_utc"], utc=True)
    expected_available = frame.index.tz_localize("UTC") + pd.Timedelta(days=1)
    if not pd.DatetimeIndex(available).equals(expected_available):
        raise ValueError("Lead-signal availability timestamps violate the UTC cutoff")

    groups = {
        name: list(columns)
        for name, columns in manifest["feature_table"]["feature_groups"].items()
    }
    if set(groups) != set(FEATURE_SET_GROUPS["all_leads"]):
        raise ValueError("Lead-signal manifest feature groups are incomplete")
    flattened = [column for columns in groups.values() for column in columns]
    if len(flattened) != len(set(flattened)):
        raise ValueError("Lead-signal feature groups must be disjoint")
    missing = sorted(set(flattened).difference(frame.columns))
    if missing:
        raise ValueError(
            f"Lead-signal table is missing declared columns: {missing[:5]}"
        )
    return (
        frame,
        groups,
        {
            "feature_path": str(feature_path),
            "feature_sha256": feature_hash,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "readiness_path": str(readiness_path),
            "as_of_date": manifest["as_of_date"],
            "common_hourly_coverage": readiness["common_hourly_coverage"],
            "group_column_counts": readiness["feature_table"]["group_column_counts"],
        },
    )


def add_target_variants(targets: pd.DataFrame) -> pd.DataFrame:
    output = targets.copy()
    returns = pd.to_numeric(output[FUTURE_RETURN_COLUMN], errors="coerce")
    dynamic_threshold = pd.to_numeric(output[DYNAMIC_THRESHOLD_COLUMN], errors="coerce")
    primary_valid = returns.notna()
    dynamic_valid = primary_valid & dynamic_threshold.notna()

    def nullable(values: np.ndarray, valid: pd.Series) -> pd.Series:
        result = pd.Series(np.nan, index=output.index, dtype=float)
        result.loc[valid] = values[valid.to_numpy()].astype(float)
        return result

    direction = (returns > 0.0).to_numpy(dtype=bool)
    output[DIRECTION_UP_COLUMN] = nullable(direction, primary_valid)
    output[PRIMARY_LARGE_MOVE_COLUMN] = nullable(
        (returns.abs() >= 0.12).to_numpy(dtype=bool),
        primary_valid,
    )
    output[DYNAMIC_LARGE_MOVE_COLUMN] = nullable(
        (returns.abs() >= dynamic_threshold).to_numpy(dtype=bool),
        dynamic_valid,
    )
    primary_class = np.select(
        [returns.to_numpy() <= -0.12, returns.to_numpy() >= 0.12],
        [-1, 1],
        default=0,
    )
    dynamic_class = np.select(
        [
            returns.to_numpy() <= -dynamic_threshold.to_numpy(),
            returns.to_numpy() >= dynamic_threshold.to_numpy(),
        ],
        [-1, 1],
        default=0,
    )
    output[PRIMARY_CLASS_COLUMN] = nullable(primary_class, primary_valid)
    output[DYNAMIC_CLASS_COLUMN] = nullable(dynamic_class, dynamic_valid)
    return output


def make_outer_folds(
    index: pd.DatetimeIndex,
    *,
    profile: str,
    gap_days: int = TAIL_HORIZON_DAYS,
    smoke_fold_count: int = 6,
    smoke_test_days: int = 30,
    full_start_year: int = 2022,
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


def select_lead_features(
    dataset: pd.DataFrame,
    fit_positions: np.ndarray,
    declared_groups: dict[str, list[str]],
    *,
    minimum_coverage: float = 0.70,
) -> tuple[dict[str, list[str]], list[dict[str, Any]]]:
    fit = dataset.iloc[np.asarray(fit_positions, dtype=int)]
    selected: dict[str, list[str]] = {}
    audit: list[dict[str, Any]] = []
    for group_name in FEATURE_SET_GROUPS["all_leads"]:
        group_selected: list[str] = []
        for feature in declared_groups[group_name]:
            values = pd.to_numeric(fit[feature], errors="coerce")
            coverage = float(values.notna().mean())
            unique = int(values.nunique(dropna=True))
            kept = bool(coverage >= float(minimum_coverage) and unique >= 2)
            audit.append(
                {
                    "group": group_name,
                    "feature": feature,
                    "coverage": coverage,
                    "required_coverage": float(minimum_coverage),
                    "unique": unique,
                    "kept": kept,
                }
            )
            if kept:
                group_selected.append(feature)
        if not group_selected:
            raise ValueError(
                f"No fold-eligible lead features remain for group {group_name}"
            )
        selected[group_name] = group_selected
    return selected, audit


def lead_candidate_specs(nonlinear: str) -> tuple[CandidateSpec, ...]:
    specs = [CandidateSpec(CLIMATOLOGY_MODEL, "none", "climatology", "core")]
    specs.extend(
        CandidateSpec(
            f"direct_logistic_{feature_set}",
            "logistic",
            "direct",
            feature_set,
        )
        for feature_set in FEATURE_SET_GROUPS
    )
    specs.extend(
        CandidateSpec(
            f"factorized_logistic_{feature_set}",
            "logistic",
            "factorized",
            feature_set,
        )
        for feature_set in ("core", "all_leads")
    )
    specs.extend(
        CandidateSpec(
            f"multiclass_logistic_{feature_set}",
            "logistic",
            "multiclass",
            feature_set,
        )
        for feature_set in ("core", "all_leads")
    )
    if nonlinear != "none":
        if nonlinear not in {"histgradient", "lightgbm"}:
            raise ValueError(f"Unsupported nonlinear model: {nonlinear}")
        specs.extend(
            CandidateSpec(
                f"direct_{nonlinear}_{feature_set}",
                nonlinear,
                "direct",
                feature_set,
            )
            for feature_set in FEATURE_SET_GROUPS
        )
    return tuple(specs)


def feature_names_for_spec(
    spec: CandidateSpec,
    core_features: list[str],
    lead_features: dict[str, list[str]],
) -> list[str]:
    names = list(core_features)
    for group_name in FEATURE_SET_GROUPS[spec.feature_set]:
        names.extend(lead_features[group_name])
    if len(names) != len(set(names)):
        raise ValueError(f"Candidate {spec.name} contains duplicate features")
    return names


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


def build_family_estimator(family: str, *, multiclass: bool = False) -> Any:
    if family == "logistic":
        classifier: Any = LogisticRegression(
            C=0.35,
            solver="lbfgs" if multiclass else "liblinear",
            max_iter=1000,
            random_state=42,
        )
    elif family == "histgradient":
        classifier = HistGradientBoostingClassifier(
            learning_rate=0.04,
            max_iter=180,
            max_leaf_nodes=15,
            max_depth=5,
            min_samples_leaf=30,
            l2_regularization=2.0,
            random_state=42,
        )
    elif family == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError as exc:  # pragma: no cover - optional local dependency
            raise RuntimeError("lightgbm candidates require lightgbm") from exc
        classifier = LGBMClassifier(
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
    else:
        raise ValueError(f"Unsupported estimator family: {family}")
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", classifier),
        ]
    )


def class_probability(model: Any, X: pd.DataFrame, class_value: int) -> np.ndarray:
    probabilities = np.asarray(model.predict_proba(X), dtype=float)
    classes = np.asarray(model.classes_)
    columns = np.flatnonzero(classes == int(class_value))
    if probabilities.ndim != 2 or len(columns) != 1:
        raise ValueError(f"Candidate is missing probability class {class_value}")
    result = probabilities[:, int(columns[0])]
    if not np.isfinite(result).all() or bool(((result < 0.0) | (result > 1.0)).any()):
        raise ValueError("Candidate produced invalid probabilities")
    return result


def class_balanced_weights(labels: pd.Series) -> pd.Series:
    values = pd.to_numeric(labels, errors="coerce")
    if values.isna().any():
        raise ValueError("Class weights require complete labels")
    counts = values.value_counts()
    if len(counts) < 2:
        raise ValueError("Class weights require at least two classes")
    mapping = {
        value: len(values) / (len(counts) * count) for value, count in counts.items()
    }
    return values.map(mapping).astype(float)


def _variant_columns(label_column: str) -> tuple[str, str]:
    if label_column == PRIMARY_LABEL_COLUMN:
        return PRIMARY_LARGE_MOVE_COLUMN, PRIMARY_CLASS_COLUMN
    if label_column == DYNAMIC_LABEL_COLUMN:
        return DYNAMIC_LARGE_MOVE_COLUMN, DYNAMIC_CLASS_COLUMN
    raise ValueError(f"Unexpected label column: {label_column}")


def fit_candidate_probabilities(
    spec: CandidateSpec,
    dataset: pd.DataFrame,
    *,
    train_positions: np.ndarray,
    predict_positions: np.ndarray,
    features: list[str],
    label_column: str,
) -> np.ndarray:
    train = dataset.iloc[np.asarray(train_positions, dtype=int)]
    predict = dataset.iloc[np.asarray(predict_positions, dtype=int)]
    X_train = train[features]
    X_predict = predict[features]

    if spec.target_family == "direct":
        labels = pd.to_numeric(train[label_column], errors="coerce")
        model = build_family_estimator(spec.estimator_family)
        fit_estimator(model, X_train, labels, build_episode_sample_weights(labels))
        return class_probability(model, X_predict, 1)

    large_move_column, class_column = _variant_columns(label_column)
    if spec.target_family == "factorized":
        magnitude = pd.to_numeric(train[large_move_column], errors="coerce")
        direction = pd.to_numeric(train[DIRECTION_UP_COLUMN], errors="coerce")
        magnitude_model = build_family_estimator(spec.estimator_family)
        direction_model = build_family_estimator(spec.estimator_family)
        fit_estimator(
            magnitude_model,
            X_train,
            magnitude,
            build_episode_sample_weights(magnitude),
        )
        fit_estimator(
            direction_model,
            X_train,
            direction,
            class_balanced_weights(direction),
        )
        return class_probability(magnitude_model, X_predict, 1) * class_probability(
            direction_model,
            X_predict,
            1,
        )

    if spec.target_family == "multiclass":
        labels = pd.to_numeric(train[class_column], errors="coerce")
        model = build_family_estimator(spec.estimator_family, multiclass=True)
        fit_estimator(model, X_train, labels, class_balanced_weights(labels))
        return class_probability(model, X_predict, 1)
    raise ValueError(f"Unsupported target family: {spec.target_family}")


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


def evaluate_lead_fold_label(
    dataset: pd.DataFrame,
    fold: OuterFold,
    label_column: str,
    *,
    profile: str,
    specs: tuple[CandidateSpec, ...],
    declared_groups: dict[str, list[str]],
    candidate_rejections: dict[str, list[str]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = pd.to_numeric(dataset[label_column], errors="coerce")
    nested = make_nested_partitions(
        fold.train_positions,
        labels,
        profile=profile,
    )
    selected_core, core_audit = select_core_features(
        dataset,
        nested.inner_train_positions,
    )
    selected_leads, lead_audit = select_lead_features(
        dataset,
        nested.inner_train_positions,
        declared_groups,
    )
    feature_map = {
        spec.name: feature_names_for_spec(spec, selected_core, selected_leads)
        for spec in specs
        if spec.name != CLIMATOLOGY_MODEL
    }

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
    candidate_metadata: dict[str, Any] = {}
    tuning_positions = np.concatenate(
        [nested.calibration_positions, nested.threshold_positions]
    )
    calibration_count = len(nested.calibration_positions)
    for spec in specs:
        started = time.monotonic()
        if spec.name in candidate_rejections:
            candidate_metadata[spec.name] = {
                "status": "excluded_after_prior_rejection",
                "reasons": candidate_rejections[spec.name],
            }
            continue
        if spec.name == CLIMATOLOGY_MODEL:
            raw_test = np.full(len(fold.test_positions), float(y_outer.mean()))
            calibrated_test = raw_test.copy()
            alert_threshold = float(np.nextafter(1.0, np.inf))
            threshold_report = {
                "threshold": alert_threshold,
                "max_false_alert_episodes_per_90_days": (
                    MAX_FALSE_ALERT_EPISODES_PER_90_DAYS
                ),
                "note": "constant climatology has no alert operating point",
            }
            calibrator_metadata = {"kind": "identity"}
            model_features: list[str] = []
        else:
            model_features = feature_map[spec.name]
            raw_tuning = fit_candidate_probabilities(
                spec,
                dataset,
                train_positions=nested.inner_train_positions,
                predict_positions=tuning_positions,
                features=model_features,
                label_column=label_column,
            )
            raw_calibration = raw_tuning[:calibration_count]
            raw_threshold = raw_tuning[calibration_count:]
            calibrator = fit_sigmoid_calibrator(
                pd.Series(raw_calibration, index=y_calibration.index),
                y_calibration,
            )
            calibrated_threshold = calibrator.predict(
                pd.Series(raw_threshold, index=y_threshold.index)
            )
            if float(np.nanstd(calibrated_threshold)) <= 1e-10:
                reason = (
                    "degenerate prior-threshold probabilities: "
                    f"{fold.fold_id}/{label_column}"
                )
                candidate_rejections.setdefault(spec.name, []).append(reason)
                candidate_metadata[spec.name] = {
                    "status": "rejected",
                    "reason": reason,
                    "features": model_features,
                    "feature_count": len(model_features),
                    "raw_threshold_std": float(np.nanstd(raw_threshold)),
                    "calibrated_threshold_std": float(np.nanstd(calibrated_threshold)),
                    "runtime_seconds": float(time.monotonic() - started),
                }
                continue
            threshold_report = select_alert_threshold(
                y_threshold,
                pd.Series(calibrated_threshold, index=y_threshold.index),
            )
            alert_threshold = float(threshold_report["threshold"])
            raw_test = fit_candidate_probabilities(
                spec,
                dataset,
                train_positions=fold.train_positions,
                predict_positions=fold.test_positions,
                features=model_features,
                label_column=label_column,
            )
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
            position = int(fold.test_positions[offset])
            rows.append(
                {
                    "label": label_column,
                    "fold_id": fold.fold_id,
                    "model": spec.name,
                    "prediction_date": pd.Timestamp(timestamp).date().isoformat(),
                    "actual_label": int(y_test.iloc[offset]),
                    "future_return_3d": float(
                        dataset.iloc[position][FUTURE_RETURN_COLUMN]
                    ),
                    "dynamic_threshold": (
                        float(dataset.iloc[position][DYNAMIC_THRESHOLD_COLUMN])
                        if pd.notna(dataset.iloc[position][DYNAMIC_THRESHOLD_COLUMN])
                        else None
                    ),
                    "raw_probability": float(raw_test[offset]),
                    "calibrated_probability": float(calibrated_test[offset]),
                    "alert_threshold": alert_threshold,
                    "alert": bool(alerts[offset]),
                }
            )
        candidate_metadata[spec.name] = {
            "status": "evaluated",
            "estimator_family": spec.estimator_family,
            "target_family": spec.target_family,
            "feature_set": spec.feature_set,
            "baseline_name": spec.baseline_name,
            "features": model_features,
            "feature_count": len(model_features),
            "calibrator": calibrator_metadata,
            "threshold_selection": threshold_report,
            "runtime_seconds": float(time.monotonic() - started),
        }

    return rows, {
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
        "selected_core_features": selected_core,
        "selected_lead_feature_counts": {
            name: len(columns) for name, columns in selected_leads.items()
        },
        "core_feature_audit": core_audit,
        "lead_feature_audit": lead_audit,
        "candidates": candidate_metadata,
    }


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


def build_source_gate_report(
    *,
    profile: str,
    specs: tuple[CandidateSpec, ...],
    metrics: dict[str, Any],
    frames: dict[str, dict[str, pd.DataFrame]],
    expected_fold_count: int,
    runtime_seconds: float,
    peak_rss_mb: float,
    max_runtime_seconds: float,
    max_peak_rss_mb: float = 1024.0,
    bootstrap_samples: int = 2000,
    candidate_rejections: dict[str, list[str]] | None = None,
) -> dict[str, Any]:
    model_names = tuple(spec.name for spec in specs)
    failures: list[str] = []
    warnings = [
        f"candidate rejected: {name}: {'; '.join(reasons)}"
        for name, reasons in sorted((candidate_rejections or {}).items())
    ]
    for label_column in LABEL_COLUMNS:
        baseline = frames[label_column][CLIMATOLOGY_MODEL]
        for model_name in model_names:
            frame = frames[label_column][model_name]
            if not frame.index.equals(baseline.index):
                failures.append(f"unmatched OOF dates: {label_column}/{model_name}")
            if frame["fold_id"].nunique() != expected_fold_count:
                failures.append(f"incomplete folds: {label_column}/{model_name}")
            probabilities = frame["calibrated_probability"].to_numpy(dtype=float)
            if not np.isfinite(probabilities).all() or bool(
                ((probabilities < 0.0) | (probabilities > 1.0)).any()
            ):
                failures.append(f"invalid probabilities: {label_column}/{model_name}")
    if runtime_seconds > max_runtime_seconds:
        failures.append(
            f"runtime {runtime_seconds:.1f}s exceeded budget {max_runtime_seconds:.1f}s"
        )
    if peak_rss_mb > max_peak_rss_mb:
        failures.append(
            f"peak RSS {peak_rss_mb:.1f}MB exceeded budget {max_peak_rss_mb:.1f}MB"
        )

    infrastructure_status = "PASS" if not failures else "FAIL"
    report: dict[str, Any] = {
        "profile": profile,
        "gate_status": infrastructure_status,
        "infrastructure_status": infrastructure_status,
        "promotion_status": "NOT_EVALUATED" if profile == "smoke" else "FAIL",
        "failures": failures,
        "warnings": warnings,
        "runtime_seconds": runtime_seconds,
        "runtime_budget_seconds": max_runtime_seconds,
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_budget_mb": max_peak_rss_mb,
    }
    if profile == "smoke" or failures:
        return report

    spec_by_name = {spec.name: spec for spec in specs}
    source_candidates = [spec for spec in specs if spec.source_augmented]
    comparisons: dict[str, Any] = {}
    for spec in source_candidates:
        baseline_name = spec.baseline_name
        if baseline_name not in spec_by_name:
            failures.append(f"missing matched core baseline: {spec.name}")
            continue
        primary_frame = frames[PRIMARY_LABEL_COLUMN][spec.name]
        primary_baseline = frames[PRIMARY_LABEL_COLUMN][baseline_name]
        dynamic_frame = frames[DYNAMIC_LABEL_COLUMN][spec.name]
        dynamic_baseline = frames[DYNAMIC_LABEL_COLUMN][baseline_name]
        primary = probability_metrics(
            primary_frame["actual_label"],
            primary_frame["calibrated_probability"],
            baseline_probabilities=primary_baseline["calibrated_probability"],
            alerts=primary_frame["alert"].astype(float),
        )
        dynamic = probability_metrics(
            dynamic_frame["actual_label"],
            dynamic_frame["calibrated_probability"],
            baseline_probabilities=dynamic_baseline["calibrated_probability"],
            alerts=dynamic_frame["alert"].astype(float),
        )
        bootstrap = moving_block_bootstrap_improvement(
            primary_frame["actual_label"],
            primary_frame["calibrated_probability"],
            primary_baseline["calibrated_probability"],
            block_length=7,
            samples=int(bootstrap_samples),
        )
        block_contributions: dict[str, float] = {}
        for fold_id, candidate_block in primary_frame.groupby("fold_id", sort=False):
            baseline_block = primary_baseline.loc[candidate_block.index]
            labels = candidate_block["actual_label"].to_numpy(dtype=float)
            candidate_probability = candidate_block["calibrated_probability"].to_numpy(
                dtype=float
            )
            baseline_probability = baseline_block["calibrated_probability"].to_numpy(
                dtype=float
            )
            block_contributions[str(fold_id)] = float(
                np.sum((labels - baseline_probability) ** 2)
                - np.sum((labels - candidate_probability) ** 2)
            )
        aggregate_gain = float(sum(block_contributions.values()))
        positive_contributions = [
            value for value in block_contributions.values() if value > 0.0
        ]
        largest_share = (
            float(max(positive_contributions) / aggregate_gain)
            if aggregate_gain > 0.0 and positive_contributions
            else None
        )
        comparisons[spec.name] = {
            "baseline": baseline_name,
            "primary": primary,
            "dynamic": dynamic,
            "bootstrap": bootstrap,
            "block_brier_contributions": block_contributions,
            "improved_calendar_blocks": int(
                sum(value > 0.0 for value in block_contributions.values())
            ),
            "aggregate_brier_gain": aggregate_gain,
            "largest_positive_block_share": largest_share,
        }

    if failures or not comparisons:
        report["failures"] = failures or ["no source-augmented comparison available"]
        report["gate_status"] = "FAIL"
        report["infrastructure_status"] = "FAIL"
        return report

    winner = max(
        comparisons,
        key=lambda name: (
            comparisons[name]["primary"]["average_precision"],
            -comparisons[name]["primary"]["brier_score"],
        ),
    )
    comparison = comparisons[winner]
    primary = comparison["primary"]
    dynamic = comparison["dynamic"]
    baseline_ap = metrics[PRIMARY_LABEL_COLUMN][comparison["baseline"]][
        "average_precision"
    ]
    events = primary["events"]
    largest_share = comparison["largest_positive_block_share"]
    checks = {
        "average_precision_20pct_above_matched_core": bool(
            primary["average_precision"] >= 1.20 * baseline_ap
        ),
        "positive_brier_skill_vs_matched_core": bool(
            primary["brier_skill"] is not None and primary["brier_skill"] > 0.0
        ),
        "episode_recall_at_least_35pct": bool(
            events["episode_recall"] is not None and events["episode_recall"] >= 0.35
        ),
        "false_alert_budget": bool(
            events["false_alert_episodes_per_90_days"]
            <= MAX_FALSE_ALERT_EPISODES_PER_90_DAYS
        ),
        "bootstrap_probability_at_least_90pct": bool(
            comparison["bootstrap"]["brier_probability_improvement"] >= 0.90
        ),
        "improvement_in_at_least_four_calendar_blocks": bool(
            comparison["improved_calendar_blocks"] >= 4
        ),
        "largest_block_contribution_at_most_50pct": bool(
            largest_share is not None and largest_share <= 0.50
        ),
        "dynamic_nonnegative_brier_skill": bool(
            dynamic["brier_skill"] is not None and dynamic["brier_skill"] >= 0.0
        ),
    }
    promotion_status = "PASS" if all(checks.values()) else "FAIL"
    report.update(
        {
            "gate_status": promotion_status,
            "promotion_status": promotion_status,
            "winner": winner,
            "matched_core_baseline": comparison["baseline"],
            "checks": checks,
            "winner_comparison": comparison,
            "comparisons": comparisons,
        }
    )
    return report


def render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Three-day lead-signal source ablation",
        "",
        f"- Profile: `{payload['profile']}`",
        (
            f"- Data: `{payload['data']['matched']['start_date']}` to "
            f"`{payload['data']['matched']['end_date']}`"
        ),
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
    lead_frame, declared_groups, lead_metadata = load_lead_signal_features(
        feature_path=args.lead_feature_path,
        manifest_path=args.lead_manifest_path,
        readiness_path=args.lead_readiness_path,
    )
    feature_frame, _ = forecast.build_features(data, TAIL_HORIZON_DAYS)
    targets = add_target_variants(build_tail_targets(data["eth_close"]))
    for column in targets.columns:
        feature_frame[column] = targets[column]

    common = lead_metadata["common_hourly_coverage"]
    common_start = pd.Timestamp(common["first_date"])
    common_end = pd.Timestamp(common["last_date"])
    dataset = feature_frame.join(lead_frame, how="inner", rsuffix="_lead")
    dataset = dataset.loc[common_start:common_end].copy()
    valid_labels = dataset[list(LABEL_COLUMNS)].notna().all(axis=1)
    valid_market = (
        pd.to_numeric(dataset["market_data_excluded"], errors="coerce")
        .fillna(1.0)
        .eq(0.0)
    )
    dataset = dataset.loc[valid_labels & valid_market].copy()
    if dataset.index.min() < common_start or dataset.index.max() > common_end:
        raise ValueError("Matched dataset escaped the declared common-date window")
    if len(dataset) < 730:
        raise ValueError("Matched dataset lacks the required two-year source history")

    folds = make_outer_folds(dataset.index, profile=args.profile)
    specs = lead_candidate_specs(args.nonlinear)
    prediction_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    candidate_rejections: dict[str, list[str]] = {}
    for fold_number, fold in enumerate(folds, start=1):
        print(
            f"[{args.profile}] fold {fold_number}/{len(folds)} {fold.fold_id}",
            flush=True,
        )
        for label_column in LABEL_COLUMNS:
            rows, metadata = evaluate_lead_fold_label(
                dataset,
                fold,
                label_column,
                profile=args.profile,
                specs=specs,
                declared_groups=declared_groups,
                candidate_rejections=candidate_rejections,
            )
            prediction_rows.extend(rows)
            fold_metadata.append(metadata)
    active_specs = tuple(
        spec for spec in specs if spec.name not in candidate_rejections
    )
    active_names = {spec.name for spec in active_specs}
    prediction_rows = [row for row in prediction_rows if row["model"] in active_names]
    models = tuple(spec.name for spec in active_specs)
    metrics, frames = summarize_predictions(
        prediction_rows,
        models=models,
        bootstrap_samples=args.bootstrap_samples,
    )
    runtime_seconds = float(time.monotonic() - started)
    max_runtime_seconds = float(
        args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else (600.0 if args.profile == "smoke" else 1800.0)
    )
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    gate = build_source_gate_report(
        profile=args.profile,
        specs=active_specs,
        metrics=metrics,
        frames=frames,
        expected_fold_count=len(folds),
        runtime_seconds=runtime_seconds,
        peak_rss_mb=peak_rss_mb,
        max_runtime_seconds=max_runtime_seconds,
        bootstrap_samples=(args.bootstrap_samples if args.profile == "full" else 100),
        candidate_rejections=candidate_rejections,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "mode": "offline_lead_signal_source_ablation",
        "profile": args.profile,
        "data": {
            "market": data_metadata,
            "lead_signals": lead_metadata,
            "matched": {
                "rows": len(dataset),
                "start_date": dataset.index.min().date().isoformat(),
                "end_date": dataset.index.max().date().isoformat(),
                "excluded_market_days": int((~valid_market).sum()),
            },
        },
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
            "candidates": [
                {
                    "name": spec.name,
                    "estimator_family": spec.estimator_family,
                    "target_family": spec.target_family,
                    "feature_set": spec.feature_set,
                    "baseline_name": spec.baseline_name,
                    "status": (
                        "rejected" if spec.name in candidate_rejections else "evaluated"
                    ),
                }
                for spec in specs
            ],
            "candidate_rejections": candidate_rejections,
            "nonlinear": args.nonlinear,
            "feature_sets": {
                name: list(groups) for name, groups in FEATURE_SET_GROUPS.items()
            },
            "core_feature_cap": MAX_CORE_FEATURES,
            "core_feature_allowlist": list(CORE_TAIL_FEATURES),
        },
        "folds": fold_metadata,
        "metrics": metrics,
        "gate": gate,
        "predictions": prediction_rows,
        "runtime_seconds": runtime_seconds,
        "peak_rss_mb": peak_rss_mb,
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
        choices=("none", "histgradient", "lightgbm"),
        default="histgradient",
        help="Evaluate at most one nonlinear candidate per run.",
    )
    parser.add_argument(
        "--lead-feature-path",
        type=Path,
        default=PROJECT_ROOT / "lake" / "gold" / "lead_signal_daily.csv.gz",
    )
    parser.add_argument(
        "--lead-manifest-path",
        type=Path,
        default=PROJECT_ROOT / "lake" / "manifests" / "lead_signal_features.json",
    )
    parser.add_argument(
        "--lead-readiness-path",
        type=Path,
        default=PROJECT_ROOT
        / "lake"
        / "reports"
        / "lead_signal_feature_readiness.json",
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
    gate = payload["gate"]
    console_summary = {
        "profile": gate["profile"],
        "gate_status": gate["gate_status"],
        "infrastructure_status": gate["infrastructure_status"],
        "promotion_status": gate["promotion_status"],
        "runtime_seconds": gate["runtime_seconds"],
        "peak_rss_mb": gate["peak_rss_mb"],
        "failures": gate["failures"],
        "warnings": gate["warnings"],
    }
    for optional_key in ("winner", "matched_core_baseline", "checks"):
        if optional_key in gate:
            console_summary[optional_key] = gate[optional_key]
    print(json.dumps(console_summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if payload["gate"]["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
