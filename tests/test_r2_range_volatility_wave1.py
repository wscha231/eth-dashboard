import numpy as np
import pandas as pd

from research.model.r2_range_volatility_wave1 import (
    POLICY,
    conformal_symmetric_quantiles,
    distribution_gate,
    distribution_metrics,
    paired_probability,
    qlike,
    variance_gate,
)


def test_policy_is_frozen_to_short_horizons_and_primary_har():
    assert POLICY["horizons_hours"] == [6, 24, 72]
    assert POLICY["variance_primary"] == "har_rv"
    assert POLICY["selection_rule"].startswith("no post-hoc")
    assert len(POLICY["folds"]) == 3


def test_qlike_zero_for_perfect_variance_forecast():
    actual = np.array([0.01, 0.02, 0.04])
    assert qlike(actual, actual) == 0.0


def test_symmetric_conformal_quantiles_are_ordered_and_zero_centered():
    q = conformal_symmetric_quantiles(np.array([0.01, 0.04]), 24, 1.25)
    assert np.all(q[:, 0] < 0)
    assert np.all(q[:, 1] == 0)
    assert np.all(q[:, 2] > 0)
    assert np.all(np.diff(q, axis=1) >= 0)


def test_distribution_metrics_reward_tighter_correct_interval():
    actual = np.array([0.0, 0.01, -0.01, 0.02, -0.02])
    good = np.column_stack((np.full(5, -0.03), np.zeros(5), np.full(5, 0.03)))
    broad = np.column_stack((np.full(5, -0.20), np.zeros(5), np.full(5, 0.20)))
    assert distribution_metrics(actual, good)["wis80"] < distribution_metrics(actual, broad)["wis80"]


def test_variance_gate_requires_two_blocks_and_five_percent_aggregate_skill():
    blocks = [
        {"har_rv_qlike": 0.80, "persistence_qlike": 1.00},
        {"har_rv_qlike": 0.90, "persistence_qlike": 1.00},
        {"har_rv_qlike": 1.05, "persistence_qlike": 1.00},
    ]
    passed = variance_gate(blocks, aggregate_candidate=0.90, aggregate_persistence=1.00)
    assert passed["passed"] is True
    failed = variance_gate(blocks, aggregate_candidate=0.97, aggregate_persistence=1.00)
    assert failed["passed"] is False


def test_distribution_gate_requires_both_baselines_and_pairing():
    candidate = {"wis80": 0.90, "coverage80": 0.80, "mean_width": 1.00}
    simple = {"wis80": 1.00, "coverage80": 0.80, "mean_width": 1.00}
    incumbent = {"wis80": 0.98, "coverage80": 0.80, "mean_width": 1.00}
    paired = {"probability_candidate_better": 0.95}
    assert distribution_gate(candidate, simple, incumbent, paired, paired)["passed"] is True
    weak_incumbent = {"wis80": 0.91, "coverage80": 0.80, "mean_width": 1.00}
    assert distribution_gate(candidate, simple, weak_incumbent, paired, paired)["passed"] is False


def test_paired_probability_is_matched_and_deterministic():
    slots = pd.date_range("2025-01-01", periods=140, freq="D", tz="UTC")
    candidate = np.full(140, 0.8)
    baseline = np.full(140, 1.0)
    first = paired_probability(candidate, baseline, slots, 24, samples=100)
    second = paired_probability(candidate, baseline, slots, 24, samples=100)
    assert first == second
    assert first["probability_candidate_better"] == 1.0
