from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.tail_evaluation import (
    CORE_TAIL_FEATURES,
    DYNAMIC_LABEL_COLUMN,
    PRIMARY_LABEL_COLUMN,
)
from scripts.evaluate_lead_signal_ablation import (
    BASE_MODELS,
    CATBOOST_MODEL,
    CLIMATOLOGY_MODEL,
    FEATURE_SET_GROUPS,
    HEURISTIC_MODEL,
    LOGISTIC_MODEL,
    CandidateSpec,
    add_target_variants,
    build_gate_report,
    build_source_gate_report,
    candidate_models,
    lead_candidate_specs,
    load_lead_signal_features,
    make_nested_partitions,
    make_outer_folds,
    select_core_features,
    select_lead_features,
    summarize_predictions,
)


def test_smoke_outer_folds_are_six_matched_thirty_day_blocks_with_purge() -> None:
    index = pd.date_range("2020-01-01", periods=1000, freq="D")

    folds = make_outer_folds(index, profile="smoke")

    assert len(folds) == 6
    assert all(len(fold.test_positions) == 30 for fold in folds)
    assert folds[0].test_positions[0] == 820
    assert folds[-1].test_positions[-1] == 999
    for fold in folds:
        assert fold.train_positions[-1] + 3 < fold.test_positions[0]


def test_full_outer_folds_expand_by_calendar_year_without_target_overlap() -> None:
    index = pd.date_range("2017-11-09", "2026-08-26", freq="D")

    folds = make_outer_folds(index, profile="full")

    assert [fold.fold_id for fold in folds] == [str(year) for year in range(2022, 2027)]
    for fold in folds:
        test_dates = index[fold.test_positions]
        assert test_dates.year.nunique() == 1
        assert fold.train_positions[-1] + 3 < fold.test_positions[0]


def test_nested_partitions_are_prior_only_and_have_purged_boundaries() -> None:
    index = pd.date_range("2020-01-01", periods=1200, freq="D")
    labels = pd.Series(0.0, index=index)
    labels.iloc[25::45] = 1.0
    positions = np.arange(0, 1100, dtype=int)

    partitions = make_nested_partitions(positions, labels, profile="smoke")

    assert (
        partitions.inner_train_positions[-1] + 3 < partitions.calibration_positions[0]
    )
    assert partitions.calibration_positions[-1] + 3 < partitions.threshold_positions[0]
    assert partitions.threshold_positions[-1] == positions[-1]
    for partition in (
        partitions.inner_train_positions,
        partitions.calibration_positions,
        partitions.threshold_positions,
    ):
        assert labels.iloc[partition].nunique() == 2


def test_core_feature_availability_is_fitted_only_on_inner_history() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2020-01-01", periods=300, freq="D")
    dataset = pd.DataFrame(
        {feature: rng.normal(size=len(index)) for feature in CORE_TAIL_FEATURES},
        index=index,
    )
    fit_positions = np.arange(0, 200, dtype=int)
    selected_before, _ = select_core_features(dataset, fit_positions)
    dataset.loc[index[200:], list(CORE_TAIL_FEATURES)[:20]] = np.nan
    selected_after, _ = select_core_features(dataset, fit_positions)

    assert selected_before == selected_after
    assert len(selected_before) == len(CORE_TAIL_FEATURES)


def test_tail_pressure_allows_declared_rolling_warmup_in_first_fold() -> None:
    rng = np.random.default_rng(7)
    index = pd.date_range("2017-11-09", periods=180, freq="D")
    dataset = pd.DataFrame(
        {feature: rng.normal(size=len(index)) for feature in CORE_TAIL_FEATURES},
        index=index,
    )
    dataset.loc[index[:100], "eth_tail_event_pressure"] = np.nan

    selected, coverage = select_core_features(dataset, np.arange(len(index)))

    assert "eth_tail_event_pressure" in selected
    row = next(
        item for item in coverage if item["feature"] == "eth_tail_event_pressure"
    )
    assert row["coverage"] < 0.80
    assert row["required_coverage"] == 0.40


def _synthetic_prediction_rows(
    *,
    fold_count: int,
    rows_per_fold: int,
    models: tuple[str, ...],
) -> list[dict]:
    dates = pd.date_range("2019-01-01", periods=fold_count * rows_per_fold, freq="D")
    rows: list[dict] = []
    for label_column in (PRIMARY_LABEL_COLUMN, DYNAMIC_LABEL_COLUMN):
        for position, timestamp in enumerate(dates):
            actual = int(position % 50 in (0, 1, 2))
            fold_id = f"fold_{position // rows_per_fold + 1:02d}"
            probabilities = {
                CLIMATOLOGY_MODEL: 0.06,
                HEURISTIC_MODEL: 0.52 if actual else 0.14,
                LOGISTIC_MODEL: 0.72 if actual else 0.08,
                CATBOOST_MODEL: 0.95 if actual else 0.02,
            }
            thresholds = {
                CLIMATOLOGY_MODEL: 1.0000001,
                HEURISTIC_MODEL: 0.40,
                LOGISTIC_MODEL: 0.40,
                CATBOOST_MODEL: 0.40,
            }
            for model_name in models:
                probability = probabilities[model_name]
                threshold = thresholds[model_name]
                rows.append(
                    {
                        "label": label_column,
                        "fold_id": fold_id,
                        "model": model_name,
                        "prediction_date": timestamp.date().isoformat(),
                        "actual_label": actual,
                        "future_return_3d": 0.15 if actual else 0.0,
                        "dynamic_threshold": 0.10,
                        "raw_probability": probability,
                        "calibrated_probability": probability,
                        "alert_threshold": threshold,
                        "alert": probability >= threshold,
                    }
                )
    return rows


def test_smoke_gate_passes_infrastructure_without_claiming_promotion() -> None:
    models = candidate_models("catboost")
    rows = _synthetic_prediction_rows(fold_count=6, rows_per_fold=30, models=models)
    metrics, frames = summarize_predictions(rows, models=models, bootstrap_samples=50)

    gate = build_gate_report(
        profile="smoke",
        models=models,
        metrics=metrics,
        frames=frames,
        expected_fold_count=6,
        runtime_seconds=10.0,
        max_runtime_seconds=600.0,
    )

    assert gate["gate_status"] == "PASS"
    assert gate["promotion_status"] == "NOT_EVALUATED"
    assert gate["failures"] == []


def test_full_gate_accepts_predeclared_strong_synthetic_candidate() -> None:
    models = candidate_models("catboost")
    rows = _synthetic_prediction_rows(fold_count=8, rows_per_fold=100, models=models)
    metrics, frames = summarize_predictions(rows, models=models, bootstrap_samples=100)

    gate = build_gate_report(
        profile="full",
        models=models,
        metrics=metrics,
        frames=frames,
        expected_fold_count=8,
        runtime_seconds=120.0,
        max_runtime_seconds=2700.0,
    )

    assert gate["gate_status"] == "PASS"
    assert gate["promotion_status"] == "PASS"
    assert gate["winner"] == CATBOOST_MODEL
    assert all(gate["checks"].values())


def test_candidate_registry_never_runs_two_nonlinear_models_together() -> None:
    assert candidate_models("none") == BASE_MODELS
    assert CATBOOST_MODEL in candidate_models("catboost")
    assert CATBOOST_MODEL not in candidate_models("lightgbm")
    assert len(candidate_models("catboost")) == len(BASE_MODELS) + 1


def test_target_variants_keep_primary_and_dynamic_definitions_separate() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    targets = pd.DataFrame(
        {
            "tail_future_return_3d": [0.13, -0.14, 0.11, np.nan],
            PRIMARY_LABEL_COLUMN: [1.0, 0.0, 0.0, np.nan],
            DYNAMIC_LABEL_COLUMN: [1.0, 0.0, 1.0, np.nan],
            "tail_dynamic_threshold": [0.10, 0.15, 0.10, np.nan],
        },
        index=index,
    )

    result = add_target_variants(targets)

    assert result["large_move_primary"].tolist()[:3] == [1.0, 1.0, 0.0]
    assert result["large_move_dynamic"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert result["tail_class_primary"].tolist()[:3] == [1.0, -1.0, 0.0]
    assert result["tail_class_dynamic"].tolist()[:3] == [1.0, 0.0, 1.0]
    assert pd.isna(result["tail_class_primary"].iloc[-1])


def test_lead_feature_eligibility_uses_only_inner_training_rows() -> None:
    rng = np.random.default_rng(11)
    index = pd.date_range("2020-01-01", periods=200, freq="D")
    declared = {group: [f"{group}_signal"] for group in FEATURE_SET_GROUPS["all_leads"]}
    dataset = pd.DataFrame(
        {columns[0]: rng.normal(size=len(index)) for columns in declared.values()},
        index=index,
    )
    fit_positions = np.arange(0, 120, dtype=int)

    selected_before, _ = select_lead_features(dataset, fit_positions, declared)
    dataset.loc[index[120:], :] = np.nan
    selected_after, _ = select_lead_features(dataset, fit_positions, declared)

    assert selected_before == selected_after


def test_candidate_specs_pair_every_augmented_model_with_same_family_core() -> None:
    specs = lead_candidate_specs("histgradient")
    names = {spec.name for spec in specs}

    assert sum(spec.estimator_family == "histgradient" for spec in specs) == len(
        FEATURE_SET_GROUPS
    )
    assert not any(spec.estimator_family == "lightgbm" for spec in specs)
    for spec in specs:
        if spec.source_augmented:
            assert spec.baseline_name in names


def test_ci_candidate_scope_is_bounded_to_one_matched_logistic_pair() -> None:
    specs = lead_candidate_specs("none", candidate_scope="ci_matched_pair")

    assert tuple(spec.name for spec in specs) == (
        CLIMATOLOGY_MODEL,
        "direct_logistic_core",
        "direct_logistic_all_leads",
    )
    assert specs[-1].baseline_name == specs[-2].name


def test_ci_candidate_scope_rejects_nonlinear_search() -> None:
    with np.testing.assert_raises_regex(ValueError, "logistic-only"):
        lead_candidate_specs("histgradient", candidate_scope="ci_matched_pair")


def test_committed_lead_artifact_is_approved_only_for_offline_evaluation() -> None:
    frame, groups, metadata = load_lead_signal_features(
        feature_path=Path("lake/gold/lead_signal_daily.csv.gz"),
        manifest_path=Path("lake/manifests/lead_signal_features.json"),
        readiness_path=Path("lake/reports/lead_signal_feature_readiness.json"),
    )

    assert len(frame) == 3269
    assert set(groups) == set(FEATURE_SET_GROUPS["all_leads"])
    assert metadata["common_hourly_coverage"]["row_count"] == 2388


def test_source_gate_accepts_strong_matched_candidate_across_five_blocks() -> None:
    specs = (
        CandidateSpec(CLIMATOLOGY_MODEL, "none", "climatology", "core"),
        CandidateSpec("direct_logistic_core", "logistic", "direct", "core"),
        CandidateSpec("direct_logistic_all_leads", "logistic", "direct", "all_leads"),
    )
    dates = pd.date_range("2022-01-01", periods=500, freq="D")
    rows: list[dict] = []
    for label in (PRIMARY_LABEL_COLUMN, DYNAMIC_LABEL_COLUMN):
        for position, date in enumerate(dates):
            actual = int(position % 50 in (0, 1, 2))
            fold = str(2022 + position // 100)
            for name, probability, threshold in (
                (CLIMATOLOGY_MODEL, 0.06, 1.1),
                ("direct_logistic_core", 0.06, 1.1),
                (
                    "direct_logistic_all_leads",
                    0.95 if actual else 0.02,
                    0.40,
                ),
            ):
                rows.append(
                    {
                        "label": label,
                        "fold_id": fold,
                        "model": name,
                        "prediction_date": date.date().isoformat(),
                        "actual_label": actual,
                        "future_return_3d": 0.15 if actual else 0.0,
                        "dynamic_threshold": 0.10,
                        "raw_probability": probability,
                        "calibrated_probability": probability,
                        "alert_threshold": threshold,
                        "alert": probability >= threshold,
                    }
                )
    models = tuple(spec.name for spec in specs)
    metrics, frames = summarize_predictions(rows, models=models, bootstrap_samples=50)

    gate = build_source_gate_report(
        profile="full",
        specs=specs,
        metrics=metrics,
        frames=frames,
        expected_fold_count=5,
        runtime_seconds=10.0,
        peak_rss_mb=200.0,
        max_runtime_seconds=1800.0,
        bootstrap_samples=100,
    )

    assert gate["promotion_status"] == "PASS"
    assert gate["winner"] == "direct_logistic_all_leads"
    assert all(gate["checks"].values())


def test_frozen_source_ablation_evidence_blocks_production_promotion() -> None:
    evidence = json.loads(
        Path("tests/phase0/lead_signal_source_ablation_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["decision"] == "retain_offline_only"
    assert evidence["production_use"] is False
    assert evidence["daily_forecast_wiring"] is False
    assert evidence["gate_a"]["infrastructure_status"] == "PASS"
    assert evidence["gate_b"]["infrastructure_status"] == "PASS"
    assert evidence["gate_b"]["promotion_status"] == "FAIL"
    assert not all(evidence["gate_b"]["checks"].values())
