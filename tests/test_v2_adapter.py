import numpy as np
import pandas as pd
import pytest

from signal_pipeline.v2_adapter import adapt_shadow_prediction, legacy_head_decisions
from signal_pipeline.v2_evaluate import (
    center_metrics,
    distribution_metrics,
    event_metrics,
    qlike,
    volatility_metrics,
)


def legacy_bundle(point_choice="365:catboost"):
    return {
        "choices": {
            "point": point_choice,
            "interval": "365:catboost",
            "probability": "365:catboost",
        },
        "no_change_selection_mae": 0.10,
        "selection_scores": {
            "365:climatology": {"point": 0.10, "interval": 0.50, "probability": 0.30},
            "365:catboost": {"point": 0.08, "interval": 0.40, "probability": 0.24},
        },
    }


def prediction():
    return {
        "quantiles": np.array([[-0.10, 0.02, 0.15]]),
        "terminal": np.array([[0.20, 0.50, 0.30]]),
        "path": np.array([[0.25, 0.20]]),
    }


def feature_row():
    return pd.Series(
        {
            "reference_price": 4000.0,
            "sigma": 0.012,
            "available_at": pd.Timestamp("2026-09-16T09:59:00Z"),
        }
    )


def test_legacy_decisions_map_heads_without_inventing_volatility_candidate():
    decisions = legacy_head_decisions(legacy_bundle())
    assert decisions["center"].passed_gate is True
    assert decisions["distribution"].passed_gate is True
    assert decisions["event"].passed_gate is True
    assert decisions["volatility"].passed_gate is False
    assert decisions["volatility"].selected_name == "variance_persistence"


def test_no_change_legacy_choice_remains_center_baseline():
    decisions = legacy_head_decisions(legacy_bundle(point_choice="no_change"))
    assert decisions["center"].passed_gate is False
    assert decisions["center"].selected_name == "no_change"


def test_adapter_preserves_head_separation_and_h_plus_one_semantics():
    result = adapt_shadow_prediction(
        bundle=legacy_bundle(),
        prediction=prediction(),
        feature_row=feature_row(),
        horizon_hours=24,
        issue_time="2026-09-16T10:05:00Z",
        input_cutoff="2026-09-16T10:00:00Z",
        input_snapshot="sha256:test",
        feature_version="shadow-separate-heads-v1",
    )
    assert result.center.log_return_median == pytest.approx(0.02)
    assert result.distribution.quantiles[0.1] == pytest.approx(-0.10)
    assert result.event.terminal_down_flat_up == pytest.approx((0.20, 0.50, 0.30))
    assert result.volatility.decision.selected_name == "variance_persistence"
    assert result.volatility.realized_variance == pytest.approx(0.012**2 * 24)
    assert result.target_end == "2026-09-17T11:00:00+00:00"


def test_adapter_rejects_non_single_origin_prediction():
    bad = prediction()
    bad["quantiles"] = np.vstack([bad["quantiles"], bad["quantiles"]])
    with pytest.raises(ValueError, match="single-origin"):
        adapt_shadow_prediction(
            bundle=legacy_bundle(),
            prediction=bad,
            feature_row=feature_row(),
            horizon_hours=24,
            issue_time="2026-09-16T10:05:00Z",
            input_cutoff="2026-09-16T10:00:00Z",
            input_snapshot="sha256:test",
            feature_version="shadow-separate-heads-v1",
        )


def test_center_metrics_use_simple_return_space():
    result = center_metrics([0.0, np.log1p(0.10)], [0.0, 0.0])
    assert result["mae"] == pytest.approx(0.05)
    assert result["rmse"] == pytest.approx(np.sqrt(0.01 / 2))


def test_qlike_is_zero_for_perfect_positive_variance_forecast():
    assert qlike([0.01, 0.04], [0.01, 0.04]) == pytest.approx(0.0)
    result = volatility_metrics([0.01, 0.04], [0.01, 0.04])
    assert result["qlike"] == pytest.approx(0.0)


def test_distribution_metrics_report_coverage_and_nonnegative_wis():
    result = distribution_metrics(
        [-0.05, 0.00, 0.05],
        [-0.10, -0.10, -0.10],
        [0.00, 0.00, 0.00],
        [0.10, 0.10, 0.10],
    )
    assert result["coverage80"] == pytest.approx(1.0)
    assert result["wis80"] >= 0


def test_event_metrics_are_zero_for_perfect_probabilities():
    terminal = [0, 1, 2]
    probs = np.eye(3)
    result = event_metrics(
        terminal,
        probs,
        up=[0, 0, 1],
        hit_up=[0.0, 0.0, 1.0],
        down=[1, 0, 0],
        hit_down=[1.0, 0.0, 0.0],
    )
    assert result["terminal_brier"] == pytest.approx(0.0)
    assert result["path_brier"] == pytest.approx(0.0)
    assert result["terminal_logloss"] == pytest.approx(0.0)
    assert result["ece"] == pytest.approx(0.0)
