from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from forecasting.tail_events import (
    CORE_TAIL_FEATURES,
    DYNAMIC_LABEL_COLUMN,
    DYNAMIC_THRESHOLD_COLUMN,
    FUTURE_RETURN_COLUMN,
    MAX_CORE_FEATURES,
    PRIMARY_LABEL_COLUMN,
    alert_event_metrics,
    build_episode_base_weights,
    build_episode_sample_weights,
    build_tail_targets,
    fit_sigmoid_calibrator,
    group_signal_episodes,
    moving_block_bootstrap_improvement,
    probability_metrics,
    select_alert_threshold,
)


def dated_series(values: list[float], start: str = "2024-01-01") -> pd.Series:
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="D"))


def test_fixed_tail_target_uses_exact_three_day_return_and_keeps_unmatured_nan() -> (
    None
):
    close = dated_series([100.0] * 8)
    close.iloc[3] = 113.0

    targets = build_tail_targets(close)

    assert math.isclose(targets.iloc[0][FUTURE_RETURN_COLUMN], 0.13)
    assert targets.iloc[0][PRIMARY_LABEL_COLUMN] == 1.0
    assert targets.iloc[1][PRIMARY_LABEL_COLUMN] == 0.0
    assert targets.iloc[-3:][PRIMARY_LABEL_COLUMN].isna().all()


def test_tail_target_and_shifted_threshold_ignore_rows_after_target_horizon() -> None:
    rng = np.random.default_rng(42)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.025, 80))),
        index=pd.date_range("2024-01-01", periods=80, freq="D"),
    )
    origin = close.index[50]
    original = build_tail_targets(close)
    mutated = close.copy()
    mutated.loc[close.index[54:]] *= 10.0
    changed = build_tail_targets(mutated)

    assert (
        original.loc[origin, PRIMARY_LABEL_COLUMN]
        == changed.loc[origin, PRIMARY_LABEL_COLUMN]
    )
    assert (
        original.loc[origin, DYNAMIC_LABEL_COLUMN]
        == changed.loc[origin, DYNAMIC_LABEL_COLUMN]
    )
    assert (
        original.loc[origin, DYNAMIC_THRESHOLD_COLUMN]
        == changed.loc[origin, DYNAMIC_THRESHOLD_COLUMN]
    )


def test_dynamic_threshold_is_shifted_and_does_not_use_origin_return() -> None:
    rng = np.random.default_rng(7)
    close = pd.Series(
        100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.02, 70))),
        index=pd.date_range("2024-01-01", periods=70, freq="D"),
    )
    origin = close.index[45]
    original_threshold = build_tail_targets(close).loc[origin, DYNAMIC_THRESHOLD_COLUMN]
    close.loc[origin] *= 1.5
    mutated_threshold = build_tail_targets(close).loc[origin, DYNAMIC_THRESHOLD_COLUMN]

    assert original_threshold == mutated_threshold


def test_adjacent_and_overlapping_positive_origins_form_one_episode() -> None:
    labels = dated_series([1, 1, 0, 1, 0, 0, 0, 1])

    identifiers = group_signal_episodes(labels, max_gap_days=3)

    assert identifiers.iloc[[0, 1, 3]].nunique() == 1
    assert identifiers.iloc[7] != identifiers.iloc[0]
    assert identifiers.iloc[2] is pd.NA


def test_positive_episode_base_weight_has_unit_mass_per_episode() -> None:
    labels = dated_series([1, 1, 0, 1, 0, 0, 0, 1, 1])
    identifiers = group_signal_episodes(labels, max_gap_days=3)
    weights = build_episode_base_weights(labels, max_gap_days=3)

    for episode_id in identifiers.dropna().unique():
        assert math.isclose(float(weights.loc[identifiers == episode_id].sum()), 1.0)


def test_sample_weights_do_not_depend_on_labels_after_training_cutoff() -> None:
    labels = dated_series(([0] * 20) + [1, 1] + ([0] * 8))
    training = labels.iloc[:24]
    original = build_episode_sample_weights(training)
    labels.iloc[25:] = 1
    changed = build_episode_sample_weights(labels.iloc[:24])

    pd.testing.assert_series_equal(original, changed)
    assert math.isclose(float(original.mean()), 1.0)


def test_event_metrics_collapse_duplicate_alert_rows_and_count_false_episodes() -> None:
    labels = dated_series([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
    alerts = dated_series([0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0])

    metrics = alert_event_metrics(labels, alerts, event_gap_days=2, alert_gap_days=2)

    assert metrics["event_episode_count"] == 2
    assert metrics["alert_episode_count"] == 3
    assert metrics["detected_event_episode_count"] == 2
    assert metrics["false_alert_episode_count"] == 1
    assert metrics["episode_recall"] == 1.0
    assert math.isclose(metrics["episode_precision"], 2.0 / 3.0)


def test_continuous_alert_is_renewed_once_per_forecast_horizon() -> None:
    labels = dated_series([0.0] * 12)
    alerts = dated_series([1.0] * 12)

    metrics = alert_event_metrics(labels, alerts)

    assert metrics["alert_episode_count"] == 4
    assert metrics["false_alert_episode_count"] == 4
    assert metrics["false_alert_episodes_per_90_days"] == 30.0


def test_renewed_alerts_during_one_long_true_event_are_not_false() -> None:
    labels = dated_series(([0.0] * 2) + ([1.0] * 8) + ([0.0] * 2))
    alerts = dated_series(([0.0] * 2) + ([1.0] * 8) + ([0.0] * 2))

    metrics = alert_event_metrics(labels, alerts)

    assert metrics["event_episode_count"] == 1
    assert metrics["alert_episode_count"] == 3
    assert metrics["detected_event_episode_count"] == 1
    assert metrics["false_alert_episode_count"] == 0


def test_alert_threshold_obeys_episode_false_alert_budget() -> None:
    labels = dated_series(([0] * 20) + [1, 1] + ([0] * 20) + [1, 1] + ([0] * 20))
    probabilities = pd.Series(0.05, index=labels.index)
    probabilities.loc[labels > 0] = 0.85
    probabilities.iloc[[5, 6, 30, 31]] = 0.70

    selected = select_alert_threshold(
        labels,
        probabilities,
        max_false_alerts_per_90_days=3.0,
    )

    assert selected["false_alert_episodes_per_90_days"] <= 3.0
    assert selected["episode_recall"] == 1.0
    assert selected["threshold"] > 0.70


def test_calibrator_rejects_single_class_calibration_window() -> None:
    raw = dated_series([0.1] * 30)
    labels = dated_series([0.0] * 30)

    with pytest.raises(ValueError, match="both event and non-event"):
        fit_sigmoid_calibrator(raw, labels)


def test_sigmoid_calibrator_never_inverts_probability_ranking() -> None:
    raw = dated_series(list(np.linspace(0.01, 0.99, 40)))
    labels = dated_series(([1.0] * 20) + ([0.0] * 20))

    calibrator = fit_sigmoid_calibrator(raw, labels)
    calibrated = calibrator.predict(raw)

    assert calibrator.coefficient >= 0.0
    assert calibrator.method == "monotone_prior_intercept_fallback"
    assert np.all(np.diff(calibrated) >= -1e-12)


def test_probability_metrics_and_bootstrap_reward_known_better_candidate() -> None:
    labels = dated_series(([0] * 90) + ([1] * 10))
    baseline = pd.Series(0.10, index=labels.index)
    candidate = pd.Series(0.02, index=labels.index)
    candidate.loc[labels > 0] = 0.80
    alerts = (candidate >= 0.5).astype(float)

    metrics = probability_metrics(
        labels,
        candidate,
        baseline_probabilities=baseline,
        alerts=alerts,
    )
    bootstrap = moving_block_bootstrap_improvement(
        labels,
        candidate,
        baseline,
        block_length=5,
        samples=200,
    )

    assert metrics["average_precision"] == 1.0
    assert metrics["brier_skill"] > 0.0
    assert metrics["events"]["episode_recall"] == 1.0
    assert bootstrap["brier_probability_improvement"] == 1.0


def test_core_feature_manifest_is_unique_and_capped() -> None:
    assert len(CORE_TAIL_FEATURES) <= MAX_CORE_FEATURES
    assert len(CORE_TAIL_FEATURES) == len(set(CORE_TAIL_FEATURES))
    assert all("target" not in feature for feature in CORE_TAIL_FEATURES)
    assert all("history_" not in feature for feature in CORE_TAIL_FEATURES)
