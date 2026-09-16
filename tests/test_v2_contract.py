import json

import pytest

from signal_pipeline.v2_contract import (
    CenterOutput,
    DistributionOutput,
    EventOutput,
    ForecastBundleV2,
    HEAD_OBJECTIVES,
    VolatilityOutput,
    select_head,
)


def decision(head, baseline=1.0, candidate=0.9, required=0.05):
    return select_head(
        head=head,
        baseline_name=f"{head}_baseline",
        candidate_name=f"{head}_candidate",
        baseline_loss=baseline,
        candidate_loss=candidate,
        required_relative_improvement=required,
    )


def valid_bundle(**overrides):
    payload = dict(
        horizon_hours=24,
        issue_time="2026-09-16T10:05:00+00:00",
        input_cutoff="2026-09-16T10:00:00+00:00",
        available_at="2026-09-16T09:59:00+00:00",
        target_end="2026-09-17T11:00:00+00:00",
        reference_price=4000.0,
        input_snapshot="sha256:abc",
        feature_version="features-v2-test",
        center=CenterOutput(log_return_median=0.0, decision=decision("center")),
        volatility=VolatilityOutput(realized_variance=0.01, decision=decision("volatility")),
        distribution=DistributionOutput(
            quantiles={0.1: -0.12, 0.5: 0.0, 0.9: 0.15},
            decision=decision("distribution"),
        ),
        event=EventOutput(
            terminal_down_flat_up=(0.2, 0.5, 0.3),
            hit_up=0.25,
            hit_down=0.20,
            decision=decision("event"),
        ),
    )
    payload.update(overrides)
    return ForecastBundleV2(**payload)


def test_head_objectives_are_distinct_and_explicit():
    assert set(HEAD_OBJECTIVES) == {"center", "volatility", "distribution", "event"}
    assert HEAD_OBJECTIVES["center"] != HEAD_OBJECTIVES["volatility"]
    assert HEAD_OBJECTIVES["distribution"] != HEAD_OBJECTIVES["event"]


def test_baseline_fallback_is_head_local():
    center = decision("center", baseline=1.0, candidate=0.99, required=0.02)
    volatility = decision("volatility", baseline=1.0, candidate=0.90, required=0.05)
    assert center.selected_name == "center_baseline"
    assert center.passed_gate is False
    assert volatility.selected_name == "volatility_candidate"
    assert volatility.passed_gate is True


def test_distribution_rejects_crossing_quantiles():
    with pytest.raises(ValueError, match="non-decreasing"):
        DistributionOutput(
            quantiles={0.1: -0.1, 0.5: 0.2, 0.9: 0.1},
            decision=decision("distribution"),
        )


def test_event_terminal_probabilities_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to one"):
        EventOutput(
            terminal_down_flat_up=(0.2, 0.5, 0.4),
            hit_up=0.3,
            hit_down=0.2,
            decision=decision("event"),
        )


def test_bundle_rejects_future_availability():
    with pytest.raises(ValueError, match="unavailable"):
        valid_bundle(available_at="2026-09-16T10:06:00+00:00")


def test_bundle_preserves_frozen_h_plus_one_endpoint():
    with pytest.raises(ValueError, match="h\+1"):
        valid_bundle(target_end="2026-09-17T10:00:00+00:00")


def test_bundle_canonical_json_is_deterministic_and_complete():
    bundle = valid_bundle()
    first = bundle.canonical_json()
    second = bundle.canonical_json()
    assert first == second
    payload = json.loads(first)
    assert payload["schema_version"] == 2
    assert payload["horizon_hours"] == 24
    assert payload["center"]["decision"]["objective"] == "MAE/median_pinball"
    assert payload["volatility"]["decision"]["objective"] == "QLIKE"
    assert payload["distribution"]["decision"]["objective"] == "WIS/pinball/coverage_width"
    assert payload["event"]["decision"]["objective"] == "Brier/logloss/ECE"
    assert list(payload["distribution"]["quantiles"]) == ["0.100000", "0.500000", "0.900000"]
