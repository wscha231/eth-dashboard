import numpy as np
import pandas as pd

from research.model.har_rv_clock_robustness import (
    HORIZONS,
    ORIGIN_HOURS,
    POLICY,
    fit_indices,
    gate_clock,
    prospective_timing,
    test_indices as evaluation_indices,
)


def _frames():
    index = pd.date_range("2026-01-01T00:00:00Z", periods=72, freq="h")
    features = pd.DataFrame(
        {
            "eth_vol_24": 0.02,
            "eth_vol_168": 0.018,
            "eth_vol_720": 0.015,
        },
        index=index,
    )
    targets = pd.DataFrame(
        {
            "return": 0.01,
            "realized_variance_total": 0.002,
            "target_end": index + pd.Timedelta(hours=7),
        },
        index=index,
    )
    return features, targets


def test_policy_predeclares_all_hours_without_posthoc_selection():
    assert HORIZONS == (6, 24, 72)
    assert ORIGIN_HOURS == tuple(range(24))
    assert POLICY["tested_origin_hours_utc"] == list(range(24))
    assert "all 24" in POLICY["expansion_rule"]
    assert "no post-hoc" in POLICY["expansion_rule"]
    assert POLICY["promotion"] == "disabled"


def test_training_stays_midnight_but_test_covers_all_clock_hours():
    features, targets = _frames()
    train = fit_indices(
        features,
        targets,
        "2026-01-01T00:00:00Z",
        "2026-01-03T00:00:00Z",
    )
    test = evaluation_indices(
        features,
        targets,
        "2026-01-01T00:00:00Z",
        "2026-01-03T00:00:00Z",
    )
    assert len(train) > 0
    assert set(train.hour) == {0}
    assert len(set(test.hour)) == 24


def test_clock_gate_reuses_frozen_variance_thresholds():
    blocks = [
        {"har_rv_qlike": 0.80, "persistence_qlike": 1.00},
        {"har_rv_qlike": 0.85, "persistence_qlike": 1.00},
        {"har_rv_qlike": 0.90, "persistence_qlike": 1.00},
    ]
    gate = gate_clock(blocks, aggregate_candidate=0.85, aggregate_persistence=1.00)
    assert gate["passed"]
    assert np.isclose(gate["qlike_improvement_vs_persistence"], 0.15)
    assert gate["improving_blocks"] == 3


def test_clock_gate_fails_large_single_block_degradation():
    blocks = [
        {"har_rv_qlike": 0.70, "persistence_qlike": 1.00},
        {"har_rv_qlike": 0.70, "persistence_qlike": 1.00},
        {"har_rv_qlike": 1.20, "persistence_qlike": 1.00},
    ]
    gate = gate_clock(blocks, aggregate_candidate=0.86, aggregate_persistence=1.00)
    assert not gate["passed"]
    assert gate["maximum_block_degradation"] > 0.10


def test_prospective_timing_quantifies_current_bottleneck_without_lowering_minima():
    six = prospective_timing(6)
    day = prospective_timing(24)
    three_day = prospective_timing(72)
    assert six["minimum_nonoverlap"] == 120
    assert day["minimum_nonoverlap"] == 60
    assert three_day["minimum_nonoverlap"] == 40
    assert six["current_00utc_daily_cohort_days_approx"] == 120
    assert six["hourly_cohort_theoretical_floor_days"] == 35
    assert day["current_00utc_daily_cohort_days_approx"] == 120
    assert day["hourly_cohort_theoretical_floor_days"] == 63
    assert three_day["current_00utc_daily_cohort_days_approx"] == 160
    assert three_day["hourly_cohort_theoretical_floor_days"] == 122
