"""Evaluate three-day adaptive asymmetric return intervals offline.

The runner compares four fixed 90% interval methods on identical purged OOF
dates.  Gate A is engineering-only.  Gate B is the historical shadow-entry
authority, but even a passing result cannot modify public forecasts without a
later integration plan.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as forecast
from forecasting.adaptive_intervals import (
    BASELINE_METHOD,
    CQR_METHOD,
    INTERVAL_ALPHA,
    INTERVAL_METHODS,
    MIN_REGIME_SCORES,
    ONLINE_WINDOW,
    REGIME_ACI_METHOD,
    VOLATILITY_ACI_METHOD,
    conformalized_quantile_interval,
    moving_block_bootstrap_interval_improvement,
    pointwise_interval_losses,
    residual_conformal_interval,
    scaled_aci_interval,
    summarize_interval_predictions,
    update_adaptive_alpha,
    volatility_scale,
)
from forecasting.tail_evaluation import TAIL_HORIZON_DAYS, json_safe
from scripts.evaluate_lead_signal_ablation import (
    OuterFold,
    load_market_data,
    make_outer_folds,
    select_core_features,
)

SCHEMA_VERSION = 1
TARGET_COLUMN = "future_return_3d"
VOLATILITY_COLUMN = "eth_vol_30"
REGIME_COLUMN = "eth_vol_30_180_ratio"
POINT_MODEL_ITERATIONS = 140
QUANTILE_MODEL_ITERATIONS = 160
MIN_INNER_ROWS = 360


@dataclass(frozen=True)
class IntervalPartitions:
    inner_train_positions: np.ndarray
    calibration_positions: np.ndarray


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def make_interval_partitions(
    train_positions: np.ndarray,
    *,
    profile: str,
    gap_days: int = TAIL_HORIZON_DAYS,
) -> IntervalPartitions:
    positions = np.asarray(train_positions, dtype=int)
    preferred_calibration = 180 if profile == "smoke" else 365
    max_calibration = len(positions) - MIN_INNER_ROWS - int(gap_days)
    calibration_size = min(preferred_calibration, max_calibration)
    if calibration_size < 120:
        raise ValueError("Outer history is too short for interval calibration")
    calibration_start = len(positions) - calibration_size
    inner_stop = calibration_start - int(gap_days)
    inner = positions[:inner_stop]
    calibration = positions[calibration_start:]
    if len(inner) < MIN_INNER_ROWS:
        raise ValueError("Interval inner-training window is too short")
    if int(inner[-1]) + int(gap_days) >= int(calibration[0]):
        raise ValueError("Interval calibration boundary violates the purge gap")
    return IntervalPartitions(inner, calibration)


def build_regressor(*, quantile: float | None = None) -> Pipeline:
    parameters: dict[str, Any] = {
        "learning_rate": 0.05,
        "max_depth": 3,
        "max_iter": (
            QUANTILE_MODEL_ITERATIONS
            if quantile is not None
            else POINT_MODEL_ITERATIONS
        ),
        "min_samples_leaf": 20,
        "l2_regularization": 0.5,
        "early_stopping": False,
        "random_state": 42,
    }
    if quantile is None:
        parameters["loss"] = "squared_error"
    else:
        parameters["loss"] = "quantile"
        parameters["quantile"] = float(quantile)
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingRegressor(**parameters)),
        ]
    )


def fit_interval_models(
    dataset: pd.DataFrame,
    *,
    train_positions: np.ndarray,
    predict_positions: np.ndarray,
    features: list[str],
) -> dict[str, np.ndarray]:
    train = dataset.iloc[np.asarray(train_positions, dtype=int)]
    predict = dataset.iloc[np.asarray(predict_positions, dtype=int)]
    y = pd.to_numeric(train[TARGET_COLUMN], errors="coerce")
    if y.isna().any():
        raise ValueError("Interval training target contains missing values")
    predictions: dict[str, np.ndarray] = {}
    for name, quantile in (
        ("point", None),
        ("q05", 0.05),
        ("q50", 0.50),
        ("q95", 0.95),
    ):
        model = build_regressor(quantile=quantile)
        model.fit(train[features], y)
        predictions[name] = np.asarray(model.predict(predict[features]), dtype=float)
    quantiles = np.sort(
        np.column_stack([predictions["q05"], predictions["q50"], predictions["q95"]]),
        axis=1,
    )
    predictions["q05"] = quantiles[:, 0]
    predictions["q50"] = quantiles[:, 1]
    predictions["q95"] = quantiles[:, 2]
    return predictions


def _regime_state(dataset: pd.DataFrame) -> pd.Series:
    ratio = pd.to_numeric(dataset[REGIME_COLUMN], errors="coerce").fillna(0.0)
    return (ratio > 0.0).astype(int)


def _regime_age(state: pd.Series) -> pd.Series:
    values = pd.to_numeric(state, errors="coerce").fillna(0).astype(int).to_numpy()
    ages = np.zeros(len(values), dtype=int)
    for position in range(1, len(values)):
        ages[position] = (
            0 if values[position] != values[position - 1] else ages[position - 1] + 1
        )
    return pd.Series(ages, index=state.index, dtype=int)


def _within_interval(actual: float, interval: tuple[float, float, float]) -> bool:
    return bool(float(interval[0]) <= float(actual) <= float(interval[2]))


def build_online_fold_predictions(
    dataset: pd.DataFrame,
    fold: OuterFold,
    *,
    calibration_predictions: dict[str, np.ndarray],
    test_predictions: dict[str, np.ndarray],
    calibration_positions: np.ndarray,
) -> list[dict[str, Any]]:
    calibration = dataset.iloc[np.asarray(calibration_positions, dtype=int)]
    test = dataset.iloc[np.asarray(fold.test_positions, dtype=int)]
    calibration_actual = calibration[TARGET_COLUMN].to_numpy(float)
    test_actual = test[TARGET_COLUMN].to_numpy(float)
    calibration_scale = volatility_scale(
        calibration[VOLATILITY_COLUMN].to_numpy(float),
        horizon_days=TAIL_HORIZON_DAYS,
    )
    test_scale = volatility_scale(
        test[VOLATILITY_COLUMN].to_numpy(float),
        horizon_days=TAIL_HORIZON_DAYS,
    )
    all_state = _regime_state(dataset)
    all_age = _regime_age(all_state)
    calibration_state = all_state.iloc[calibration_positions].to_numpy(int)
    test_state = all_state.iloc[fold.test_positions].to_numpy(int)
    test_age = all_age.iloc[fold.test_positions].to_numpy(int)

    residual_history = list(calibration_actual - calibration_predictions["point"])
    cqr_history = list(
        np.maximum(
            calibration_predictions["q05"] - calibration_actual,
            calibration_actual - calibration_predictions["q95"],
        )
    )
    normalized_history = list(
        (calibration_actual - calibration_predictions["point"]) / calibration_scale
    )
    regime_history = list(zip(normalized_history, calibration_state, strict=True))

    alpha_global = float(INTERVAL_ALPHA)
    alpha_by_regime = {0: float(INTERVAL_ALPHA), 1: float(INTERVAL_ALPHA)}
    stored_intervals: dict[str, list[tuple[float, float, float]]] = {
        method: [] for method in INTERVAL_METHODS
    }
    test_dates = pd.DatetimeIndex(test.index)
    maturity_cursor = 0
    rows: list[dict[str, Any]] = []

    for offset, timestamp in enumerate(test_dates):
        while (
            maturity_cursor < offset
            and test_dates[maturity_cursor] + pd.Timedelta(days=TAIL_HORIZON_DAYS)
            <= timestamp
        ):
            actual = float(test_actual[maturity_cursor])
            state = int(test_state[maturity_cursor])
            residual = actual - float(test_predictions["point"][maturity_cursor])
            residual_history.append(residual)
            cqr_history.append(
                max(
                    float(test_predictions["q05"][maturity_cursor]) - actual,
                    actual - float(test_predictions["q95"][maturity_cursor]),
                )
            )
            normalized_history.append(residual / float(test_scale[maturity_cursor]))
            regime_history.append((normalized_history[-1], state))
            alpha_global = update_adaptive_alpha(
                alpha_global,
                missed=not _within_interval(
                    actual,
                    stored_intervals[VOLATILITY_ACI_METHOD][maturity_cursor],
                ),
                step_size=0.01,
            )
            alpha_by_regime[state] = update_adaptive_alpha(
                alpha_by_regime[state],
                missed=not _within_interval(
                    actual,
                    stored_intervals[REGIME_ACI_METHOD][maturity_cursor],
                ),
                step_size=(0.02 if int(test_age[maturity_cursor]) == 0 else 0.01),
            )
            maturity_cursor += 1

        baseline_history = residual_history[-ONLINE_WINDOW:]
        baseline_interval = residual_conformal_interval(
            float(test_predictions["point"][offset]),
            baseline_history,
        )
        cqr_scores = cqr_history[-ONLINE_WINDOW:]
        cqr_interval = conformalized_quantile_interval(
            float(test_predictions["q05"][offset]),
            float(test_predictions["q50"][offset]),
            float(test_predictions["q95"][offset]),
            cqr_scores,
        )
        normalized_scores = normalized_history[-ONLINE_WINDOW:]
        volatility_interval = scaled_aci_interval(
            float(test_predictions["point"][offset]),
            float(test_scale[offset]),
            normalized_scores,
            adaptive_alpha=alpha_global,
        )
        state = int(test_state[offset])
        recent_regime = regime_history[-ONLINE_WINDOW:]
        same_regime = [value for value, regime in recent_regime if int(regime) == state]
        regime_scores = (
            same_regime
            if len(same_regime) >= MIN_REGIME_SCORES
            else [value for value, _ in recent_regime]
        )
        regime_alpha = alpha_by_regime[state]
        if int(test_age[offset]) == 0:
            regime_alpha = max(0.02, regime_alpha - 0.02)
        regime_interval = scaled_aci_interval(
            float(test_predictions["point"][offset]),
            float(test_scale[offset]),
            regime_scores,
            adaptive_alpha=regime_alpha,
        )
        intervals = {
            BASELINE_METHOD: baseline_interval,
            CQR_METHOD: cqr_interval,
            VOLATILITY_ACI_METHOD: volatility_interval,
            REGIME_ACI_METHOD: regime_interval,
        }
        for method, interval in intervals.items():
            stored_intervals[method].append(interval)
            rows.append(
                {
                    "fold_id": fold.fold_id,
                    "method": method,
                    "prediction_date": pd.Timestamp(timestamp).date().isoformat(),
                    "actual_return": float(test_actual[offset]),
                    "lower": float(interval[0]),
                    "median": float(interval[1]),
                    "upper": float(interval[2]),
                    "width": float(interval[2] - interval[0]),
                    "volatility_scale": float(test_scale[offset]),
                    "high_volatility_regime": bool(state),
                    "regime_age_days": int(test_age[offset]),
                    "adaptive_alpha": (
                        float(alpha_global)
                        if method == VOLATILITY_ACI_METHOD
                        else (
                            float(regime_alpha) if method == REGIME_ACI_METHOD else None
                        )
                    ),
                    "history_rows": (
                        len(regime_scores)
                        if method == REGIME_ACI_METHOD
                        else int(
                            len(cqr_scores)
                            if method == CQR_METHOD
                            else len(baseline_history)
                        )
                    ),
                }
            )
    return rows


def evaluate_fold(
    dataset: pd.DataFrame,
    fold: OuterFold,
    *,
    profile: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    partitions = make_interval_partitions(fold.train_positions, profile=profile)
    selected_features, feature_audit = select_core_features(
        dataset,
        partitions.inner_train_positions,
    )
    calibration_predictions = fit_interval_models(
        dataset,
        train_positions=partitions.inner_train_positions,
        predict_positions=partitions.calibration_positions,
        features=selected_features,
    )
    test_predictions = fit_interval_models(
        dataset,
        train_positions=fold.train_positions,
        predict_positions=fold.test_positions,
        features=selected_features,
    )
    rows = build_online_fold_predictions(
        dataset,
        fold,
        calibration_predictions=calibration_predictions,
        test_predictions=test_predictions,
        calibration_positions=partitions.calibration_positions,
    )
    metadata = {
        "fold_id": fold.fold_id,
        "train_start": dataset.index[fold.train_positions[0]].date().isoformat(),
        "train_end": dataset.index[fold.train_positions[-1]].date().isoformat(),
        "calibration_start": dataset.index[partitions.calibration_positions[0]]
        .date()
        .isoformat(),
        "calibration_end": dataset.index[partitions.calibration_positions[-1]]
        .date()
        .isoformat(),
        "test_start": dataset.index[fold.test_positions[0]].date().isoformat(),
        "test_end": dataset.index[fold.test_positions[-1]].date().isoformat(),
        "train_rows": len(fold.train_positions),
        "inner_train_rows": len(partitions.inner_train_positions),
        "calibration_rows": len(partitions.calibration_positions),
        "test_rows": len(fold.test_positions),
        "selected_core_features": selected_features,
        "feature_audit": feature_audit,
    }
    return rows, metadata


def _method_frames(predictions: list[dict[str, Any]]) -> dict[str, pd.DataFrame]:
    frame = pd.DataFrame(predictions)
    frames: dict[str, pd.DataFrame] = {}
    for method in INTERVAL_METHODS:
        subset = frame.loc[frame["method"] == method].copy()
        subset = subset.sort_values("prediction_date").reset_index(drop=True)
        frames[method] = subset
    return frames


def _breakdowns(frame: pd.DataFrame) -> dict[str, Any]:
    calendar = {
        str(fold): summarize_interval_predictions(group)
        for fold, group in frame.groupby("fold_id", sort=True)
    }
    volatility = {
        ("high" if bool(regime) else "normal"): summarize_interval_predictions(group)
        for regime, group in frame.groupby("high_volatility_regime", sort=True)
    }
    after_change: dict[str, Any] = {}
    for days in (1, 3, 7):
        subset = frame.loc[pd.to_numeric(frame["regime_age_days"]) < days]
        after_change[str(days)] = (
            summarize_interval_predictions(subset) if not subset.empty else None
        )
    return {
        "calendar": calendar,
        "volatility_regime": volatility,
        "after_regime_change_days": after_change,
    }


def _calendar_comparison(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
) -> dict[str, Any]:
    candidate_scored = pointwise_interval_losses(candidate)
    baseline_scored = pointwise_interval_losses(baseline)
    contributions: dict[str, float] = {}
    upper_coverage: dict[str, float] = {}
    for fold_id in sorted(candidate_scored["fold_id"].astype(str).unique()):
        candidate_fold = candidate_scored.loc[
            candidate_scored["fold_id"].astype(str) == fold_id
        ]
        baseline_fold = baseline_scored.loc[
            baseline_scored["fold_id"].astype(str) == fold_id
        ]
        contributions[fold_id] = float(
            baseline_fold["wis"].sum() - candidate_fold["wis"].sum()
        )
        upper_coverage[fold_id] = float(candidate_fold["upper_covered"].mean())
    return {
        "wis_contribution_by_block": contributions,
        "improved_calendar_blocks": int(
            sum(value > 0.0 for value in contributions.values())
        ),
        "minimum_block_upper_coverage": float(min(upper_coverage.values())),
        "upper_coverage_by_block": upper_coverage,
    }


def build_interval_gate(
    *,
    profile: str,
    frames: dict[str, pd.DataFrame],
    metrics: dict[str, dict[str, Any]],
    runtime_seconds: float,
    peak_rss_mb: float,
    expected_fold_count: int,
    max_runtime_seconds: float,
    bootstrap_samples: int,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_rows = len(frames[BASELINE_METHOD])
    baseline_dates = frames[BASELINE_METHOD]["prediction_date"].tolist()
    for method in INTERVAL_METHODS:
        frame = frames[method]
        if len(frame) != expected_rows:
            failures.append(
                f"{method}: expected {expected_rows} matched rows, got {len(frame)}"
            )
        if frame["prediction_date"].tolist() != baseline_dates:
            failures.append(f"{method}: prediction dates do not match baseline")
        if frame["fold_id"].nunique() != int(expected_fold_count):
            failures.append(
                f"{method}: expected {expected_fold_count} folds, got {frame['fold_id'].nunique()}"
            )
        try:
            pointwise_interval_losses(frame)
        except ValueError as exc:
            failures.append(f"{method}: {exc}")
    if runtime_seconds > float(max_runtime_seconds):
        failures.append(
            f"runtime {runtime_seconds:.1f}s exceeds {float(max_runtime_seconds):.1f}s"
        )
    if peak_rss_mb > 1024.0:
        failures.append(f"peak RSS {peak_rss_mb:.1f}MB exceeds 1024MB")
    infrastructure_status = "PASS" if not failures else "FAIL"
    report: dict[str, Any] = {
        "profile": profile,
        "gate_status": infrastructure_status,
        "infrastructure_status": infrastructure_status,
        "promotion_status": "NOT_EVALUATED" if profile == "smoke" else "FAIL",
        "failures": failures,
        "warnings": [],
        "runtime_seconds": float(runtime_seconds),
        "runtime_budget_seconds": float(max_runtime_seconds),
        "peak_rss_mb": float(peak_rss_mb),
        "peak_rss_budget_mb": 1024.0,
    }
    if profile == "smoke" or failures:
        return report

    baseline = frames[BASELINE_METHOD]
    comparisons: dict[str, Any] = {}
    for method in INTERVAL_METHODS:
        if method == BASELINE_METHOD:
            continue
        bootstrap = moving_block_bootstrap_interval_improvement(
            frames[method],
            baseline,
            samples=int(bootstrap_samples),
        )
        calendar = _calendar_comparison(frames[method], baseline)
        comparisons[method] = {
            "metrics": metrics[method],
            "baseline_metrics": metrics[BASELINE_METHOD],
            "bootstrap": bootstrap,
            **calendar,
        }

    winner = min(
        comparisons,
        key=lambda name: float(metrics[name]["weighted_interval_score"]),
    )
    winner_metrics = metrics[winner]
    baseline_metrics = metrics[BASELINE_METHOD]
    comparison = comparisons[winner]
    bootstrap = comparison["bootstrap"]
    wis_improved = bool(
        winner_metrics["weighted_interval_score"]
        < baseline_metrics["weighted_interval_score"]
    )
    q95_improved = bool(winner_metrics["pinball_q95"] < baseline_metrics["pinball_q95"])
    winner_tail = winner_metrics["up_tail"]
    baseline_tail = baseline_metrics["up_tail"]
    tail_improved = bool(
        winner_tail["exceedance_rate"] <= baseline_tail["exceedance_rate"]
        and winner_tail["mean_exceedance"] < baseline_tail["mean_exceedance"]
    )
    width_ratio = float(
        winner_metrics["median_width"] / baseline_metrics["median_width"]
    )
    large_tail_reduction = bool(
        baseline_tail["mean_exceedance"] > 0.0
        and winner_tail["mean_exceedance"] <= 0.5 * baseline_tail["mean_exceedance"]
        and comparison["improved_calendar_blocks"] >= 4
    )
    winner_change = _breakdowns(frames[winner])["after_regime_change_days"]["7"]
    baseline_change = _breakdowns(baseline)["after_regime_change_days"]["7"]
    change_coverage_ok = bool(
        winner_change is not None
        and baseline_change is not None
        and winner_change["upper_coverage"] >= baseline_change["upper_coverage"] - 0.03
    )
    bootstrap_ok = bool(
        (wis_improved and bootstrap["wis_probability_improvement"] >= 0.90)
        or (q95_improved and bootstrap["q95_probability_improvement"] >= 0.90)
    )
    checks = {
        "wis_or_q95_pinball_improved": bool(wis_improved or q95_improved),
        "upper_coverage_within_92_to_98pct": bool(
            0.92 <= winner_metrics["upper_coverage"] <= 0.98
        ),
        "up_tail_exceedance_improved": tail_improved,
        "median_width_guardrail": bool(width_ratio <= 1.25 or large_tail_reduction),
        "paired_bootstrap_probability_at_least_90pct": bootstrap_ok,
        "improvement_in_at_least_four_calendar_blocks": bool(
            comparison["improved_calendar_blocks"] >= 4
        ),
        "minimum_calendar_upper_coverage_at_least_88pct": bool(
            comparison["minimum_block_upper_coverage"] >= 0.88
        ),
        "post_change_upper_coverage_not_worse_by_over_3pct": change_coverage_ok,
    }
    promotion_status = "PASS" if all(checks.values()) else "FAIL"
    report.update(
        {
            "gate_status": promotion_status,
            "promotion_status": promotion_status,
            "winner": winner,
            "baseline": BASELINE_METHOD,
            "checks": checks,
            "winner_comparison": {
                **comparison,
                "median_width_ratio": width_ratio,
                "wis_improved": wis_improved,
                "q95_improved": q95_improved,
                "up_tail_improved": tail_improved,
                "post_change_upper_coverage_ok": change_coverage_ok,
            },
            "comparisons": comparisons,
        }
    )
    return report


def render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Three-day adaptive asymmetric interval evaluation",
        "",
        f"- Profile: `{payload['profile']}`",
        (
            f"- Data: `{payload['data']['matched']['start_date']}` to "
            f"`{payload['data']['matched']['end_date']}`"
        ),
        f"- Infrastructure: **{gate['infrastructure_status']}**",
        f"- Promotion: **{gate['promotion_status']}**",
        f"- Runtime: {gate['runtime_seconds']:.1f}s",
        "",
        "| Method | WIS | q95 pinball | Upper coverage | Tail coverage | Median width |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in INTERVAL_METHODS:
        metrics = payload["metrics"][method]
        lines.append(
            f"| {method} | {metrics['weighted_interval_score']:.5f} | "
            f"{metrics['pinball_q95']:.5f} | {metrics['upper_coverage']:.3f} | "
            f"{metrics['up_tail']['coverage']:.3f} | {metrics['median_width']:.3f} |"
        )
    if "checks" in gate:
        lines.extend(["", "## Gate checks", ""])
        lines.extend(
            f"- [{'x' if passed else ' '}] {name}"
            for name, passed in gate["checks"].items()
        )
    return "\n".join(lines) + "\n"


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    data, data_metadata = load_market_data(
        data_path=args.data_path,
        git_ref=args.data_git_ref,
        git_data_path=args.git_data_path,
    )
    feature_frame, _ = forecast.build_features(data, TAIL_HORIZON_DAYS)
    dataset = feature_frame.copy()
    dataset[TARGET_COLUMN] = (
        pd.to_numeric(data["eth_close"], errors="coerce").shift(-TAIL_HORIZON_DAYS)
        / pd.to_numeric(data["eth_close"], errors="coerce")
        - 1.0
    )
    dataset = dataset.loc[dataset[TARGET_COLUMN].notna()].copy()
    if VOLATILITY_COLUMN not in dataset or REGIME_COLUMN not in dataset:
        raise ValueError("Required prior-only volatility features are unavailable")

    folds = make_outer_folds(
        dataset.index,
        profile=args.profile,
        full_start_year=2020,
    )
    predictions: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []
    for number, fold in enumerate(folds, start=1):
        print(
            f"[{args.profile}] interval fold {number}/{len(folds)} {fold.fold_id}",
            flush=True,
        )
        rows, metadata = evaluate_fold(dataset, fold, profile=args.profile)
        predictions.extend(rows)
        fold_metadata.append(metadata)

    frames = _method_frames(predictions)
    metrics = {
        method: summarize_interval_predictions(frame)
        for method, frame in frames.items()
    }
    breakdowns = {method: _breakdowns(frame) for method, frame in frames.items()}
    runtime_seconds = float(time.monotonic() - started)
    max_runtime_seconds = float(
        args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else (600.0 if args.profile == "smoke" else 1800.0)
    )
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    gate = build_interval_gate(
        profile=args.profile,
        frames=frames,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        peak_rss_mb=peak_rss_mb,
        expected_fold_count=len(folds),
        max_runtime_seconds=max_runtime_seconds,
        bootstrap_samples=(args.bootstrap_samples if args.profile == "full" else 100),
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "mode": "offline_adaptive_asymmetric_intervals",
        "profile": args.profile,
        "data": {
            "market": data_metadata,
            "matched": {
                "rows": len(dataset),
                "start_date": dataset.index.min().date().isoformat(),
                "end_date": dataset.index.max().date().isoformat(),
            },
        },
        "interval_contract": {
            "horizon_days": TAIL_HORIZON_DAYS,
            "nominal_coverage": 1.0 - INTERVAL_ALPHA,
            "lower_quantile": 0.05,
            "median_quantile": 0.50,
            "upper_quantile": 0.95,
            "online_window": ONLINE_WINDOW,
            "maturity_delay_days": TAIL_HORIZON_DAYS,
            "methods": list(INTERVAL_METHODS),
            "production_use": False,
        },
        "split_manifest": {
            "outer_profile": args.profile,
            "outer_fold_count": len(folds),
            "gap_days": TAIL_HORIZON_DAYS,
            "calibration": "prior-only split conformal; test residuals enter only after three-day maturity",
            "feature_selection": "inner-training rows only",
        },
        "folds": fold_metadata,
        "metrics": metrics,
        "breakdowns": breakdowns,
        "gate": gate,
        "predictions": predictions,
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
    console = {
        key: gate[key]
        for key in (
            "profile",
            "gate_status",
            "infrastructure_status",
            "promotion_status",
            "runtime_seconds",
            "peak_rss_mb",
            "failures",
            "warnings",
            "winner",
            "baseline",
            "checks",
        )
        if key in gate
    }
    print(json.dumps(console, indent=2, ensure_ascii=False, allow_nan=False))
    return 0 if gate["gate_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
