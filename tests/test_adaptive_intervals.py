from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.adaptive_intervals import (
    BASELINE_METHOD,
    CQR_METHOD,
    INTERVAL_METHODS,
    VOLATILITY_ACI_METHOD,
    conformal_quantile,
    pointwise_interval_losses,
    repair_interval,
    summarize_interval_predictions,
    update_adaptive_alpha,
)
from scripts.evaluate_adaptive_intervals import (
    IntervalPartitions,
    build_interval_gate,
    build_online_fold_predictions,
    make_interval_partitions,
)
from scripts.evaluate_lead_signal_ablation import OuterFold


def test_conformal_quantile_uses_finite_sample_higher_rank() -> None:
    values = np.arange(1.0, 21.0)

    result = conformal_quantile(values, 0.90)

    assert result == 19.0


def test_interval_repair_orders_crossing_quantiles() -> None:
    assert repair_interval(0.2, 0.0, -0.1) == (-0.1, 0.0, 0.2)


def test_adaptive_alpha_widens_after_a_miss_and_tightens_after_coverage() -> None:
    after_miss = update_adaptive_alpha(0.10, missed=True, step_size=0.01)
    after_hit = update_adaptive_alpha(0.10, missed=False, step_size=0.01)

    assert after_miss < 0.10
    assert after_hit > 0.10


def test_interval_metrics_measure_upper_tail_exceedance() -> None:
    frame = pd.DataFrame(
        {
            "actual_return": [0.0, 0.15, -0.14],
            "lower": [-0.10, -0.05, -0.10],
            "median": [0.0, 0.05, 0.0],
            "upper": [0.10, 0.12, 0.10],
        }
    )

    metrics = summarize_interval_predictions(frame)

    assert metrics["rows"] == 3
    assert metrics["up_tail"]["rows"] == 1
    assert metrics["up_tail"]["coverage"] == 0.0
    assert np.isclose(metrics["up_tail"]["mean_exceedance"], 0.03)
    assert metrics["down_tail"]["coverage"] == 0.0


def test_interval_partitions_purge_inner_training_from_calibration() -> None:
    partitions = make_interval_partitions(np.arange(1000), profile="full")

    assert isinstance(partitions, IntervalPartitions)
    assert len(partitions.calibration_positions) == 365
    assert (
        partitions.inner_train_positions[-1] + 3 < partitions.calibration_positions[0]
    )


def _online_fixture() -> tuple[
    pd.DataFrame,
    OuterFold,
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    index = pd.date_range("2024-01-01", periods=520, freq="D")
    target = 0.02 * np.sin(np.arange(len(index)) / 9.0)
    dataset = pd.DataFrame(
        {
            "future_return_3d": target,
            "eth_vol_30": np.full(len(index), 0.025),
            "eth_vol_30_180_ratio": np.where(np.arange(len(index)) < 470, -0.2, 0.2),
        },
        index=index,
    )
    calibration_positions = np.arange(300, 450, dtype=int)
    test_positions = np.arange(453, 503, dtype=int)
    fold = OuterFold("fixture", np.arange(0, 450, dtype=int), test_positions)

    def predictions(positions: np.ndarray) -> dict[str, np.ndarray]:
        size = len(positions)
        return {
            "point": np.zeros(size),
            "q05": np.full(size, -0.08),
            "q50": np.zeros(size),
            "q95": np.full(size, 0.08),
        }

    return (
        dataset,
        fold,
        calibration_positions,
        predictions(calibration_positions),
        predictions(test_positions),
    )


def test_online_intervals_do_not_use_test_outcome_before_three_day_maturity() -> None:
    dataset, fold, calibration, calibration_predictions, test_predictions = (
        _online_fixture()
    )
    original = build_online_fold_predictions(
        dataset,
        fold,
        calibration_predictions=calibration_predictions,
        test_predictions=test_predictions,
        calibration_positions=calibration,
    )
    mutation_offset = 20
    mutated_dataset = dataset.copy()
    mutated_dataset.iloc[
        fold.test_positions[mutation_offset],
        mutated_dataset.columns.get_loc("future_return_3d"),
    ] = 0.80
    mutated = build_online_fold_predictions(
        mutated_dataset,
        fold,
        calibration_predictions=calibration_predictions,
        test_predictions=test_predictions,
        calibration_positions=calibration,
    )

    original_frame = pd.DataFrame(original)
    mutated_frame = pd.DataFrame(mutated)
    cutoff_date = (
        dataset.index[fold.test_positions[mutation_offset] + 2].date().isoformat()
    )
    columns = ["method", "prediction_date", "lower", "median", "upper"]
    pd.testing.assert_frame_equal(
        original_frame.loc[
            original_frame["prediction_date"] <= cutoff_date, columns
        ].reset_index(drop=True),
        mutated_frame.loc[
            mutated_frame["prediction_date"] <= cutoff_date, columns
        ].reset_index(drop=True),
    )


def _synthetic_gate_frames() -> dict[str, pd.DataFrame]:
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    actual = 0.12 * np.sin(np.arange(len(dates)) / 8.0)
    actual[np.arange(len(dates)) % 50 == 0] = 0.18
    frames: dict[str, pd.DataFrame] = {}
    for method in INTERVAL_METHODS:
        rows = []
        for position, (date, value) in enumerate(zip(dates, actual, strict=True)):
            if method == BASELINE_METHOD:
                lower, median, upper = -0.05, 0.0, 0.05
            elif method == CQR_METHOD:
                lower, median, upper = value - 0.10, value, value + 0.10
                if position % 20 == 1:
                    median, upper = value - 0.02, value - 0.01
            elif method == VOLATILITY_ACI_METHOD:
                lower, median, upper = -0.08, 0.0, 0.08
            else:
                lower, median, upper = -0.07, 0.0, 0.07
            rows.append(
                {
                    "fold_id": str(2020 + position // 100),
                    "method": method,
                    "prediction_date": date.date().isoformat(),
                    "actual_return": float(value),
                    "lower": float(lower),
                    "median": float(median),
                    "upper": float(upper),
                    "high_volatility_regime": bool(position % 2),
                    "regime_age_days": int(position % 9),
                }
            )
        frames[method] = pd.DataFrame(rows)
    return frames


def test_full_interval_gate_accepts_stable_tail_coverage_improvement() -> None:
    frames = _synthetic_gate_frames()
    metrics = {
        method: summarize_interval_predictions(frame)
        for method, frame in frames.items()
    }

    gate = build_interval_gate(
        profile="full",
        frames=frames,
        metrics=metrics,
        runtime_seconds=10.0,
        peak_rss_mb=200.0,
        expected_fold_count=5,
        max_runtime_seconds=1800.0,
        bootstrap_samples=100,
    )

    assert gate["winner"] == CQR_METHOD
    assert gate["promotion_status"] == "PASS"
    assert all(gate["checks"].values())


def test_pointwise_losses_reject_crossed_intervals() -> None:
    frame = pd.DataFrame(
        {
            "actual_return": [0.0],
            "lower": [0.1],
            "median": [0.0],
            "upper": [-0.1],
        }
    )

    try:
        pointwise_interval_losses(frame)
    except ValueError as exc:
        assert "ordered" in str(exc)
    else:
        raise AssertionError("crossed interval was accepted")


def test_frozen_interval_evidence_cannot_change_the_public_range() -> None:
    evidence = json.loads(
        Path("tests/phase0/adaptive_interval_gate_metrics.json").read_text(
            encoding="utf-8"
        )
    )

    assert evidence["decision"] == "retain_existing_interval_only"
    assert evidence["production_use"] is False
    assert evidence["daily_forecast_wiring"] is False
    assert evidence["public_range_change"] is False
    assert evidence["gate_a"]["infrastructure_status"] == "PASS"
    assert evidence["gate_b"]["infrastructure_status"] == "PASS"
    assert evidence["gate_b"]["promotion_status"] == "FAIL"
    assert not all(evidence["gate_b"]["checks"].values())
    assert not any(
        row["all_methods_covered"] for row in evidence["august_2026_up_tail_audit"]
    )
