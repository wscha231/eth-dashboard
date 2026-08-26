"""Phase 0 sanity checks on build_features — lightweight, no live data required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

import eth_price_forecast as efp
from tests.phase0.longrun_oof_common import (
    backfill_classification_prediction_rows,
    prediction_fold_indices,
    summarize_selected_features_by_fold,
)


class _RecordingClassifier(BaseEstimator, ClassifierMixin):
    """Cloneable test estimator that records every temporal fit window."""

    fit_windows: list[pd.Index] = []

    def fit(self, X, y, sample_weight=None):
        del y, sample_weight
        type(self).fit_windows.append(X.index.copy())
        self.classes_ = np.array([0, 1])
        self.center_ = float(pd.to_numeric(X.iloc[:, 0], errors="coerce").median())
        return self

    def predict_proba(self, X):
        values = pd.to_numeric(X.iloc[:, 0], errors="coerce").to_numpy(dtype=float)
        probability_up = 1.0 / (1.0 + np.exp(-(values - self.center_)))
        return np.column_stack([1.0 - probability_up, probability_up])


class _DefaultAttributes:
    """Small forecast-artifact stub for summary contract tests."""

    def __init__(self, **values):
        self.__dict__.update(values)

    def __getattr__(self, name):
        del name
        return 0.0


def test_build_features_returns_nonempty_columns(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    frame, columns = efp.build_features(synthetic_ohlcv_with_companions, horizon=7)
    assert len(columns) > 50, f"Expected >50 feature columns, got {len(columns)}"
    assert "target_return" in frame.columns
    assert "target_close" in frame.columns
    for column in efp.DIRECTION_TARGET_COLUMNS:
        assert column in frame.columns


def test_build_features_no_infinities(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    frame, columns = efp.build_features(synthetic_ohlcv_with_companions, horizon=7)
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce")
    inf_count = int(np.isinf(numeric.to_numpy()).sum())
    assert inf_count == 0, f"Found {inf_count} infinite values in feature frame"


def test_target_return_last_horizon_rows_are_nan(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    horizon = 7
    frame, _ = efp.build_features(synthetic_ohlcv_with_companions, horizon=horizon)
    tail = frame["target_return"].tail(horizon)
    assert tail.isna().all(), (
        f"Last {horizon} target_return rows must be NaN (no future), got {tail.to_list()}"
    )


def test_regime_targets_are_in_expected_set(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    frame, _ = efp.build_features(synthetic_ohlcv_with_companions, horizon=7)
    regime = frame["target_regime"].dropna().unique()
    assert set(regime.tolist()).issubset({0.0, 1.0, 2.0}), f"Unexpected regime labels: {regime}"


def test_direction_target_excludes_range_noise(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    frame, _ = efp.build_features(synthetic_ohlcv_with_companions, horizon=30)
    target = efp.get_direction_classification_target(frame, 30)
    clean = target.dropna()

    assert not clean.empty
    assert set(clean.unique().tolist()).issubset({0.0, 1.0})
    assert len(clean) < frame["target_return"].notna().sum()
    assert 0.0 < efp.direction_target_actionable_rate(frame, 30) < 1.0


def test_time_series_splitter_respects_gap() -> None:
    splitter = efp.safe_time_series_split(n_samples=500, n_splits=3, test_size=30, gap=7)
    X = pd.DataFrame(np.zeros((500, 3)))
    for train_idx, test_idx in splitter.split(X):
        assert test_idx.min() - train_idx.max() > 7, (
            f"Gap violated: train_max={train_idx.max()} test_min={test_idx.min()}"
        )


def test_oof_return_calibration_shrinks_overconfident_regression() -> None:
    index = pd.date_range("2020-01-01", periods=320, freq="D")
    raw_oof = pd.Series(
        np.linspace(-0.18, 0.18, len(index)) + 0.025 * np.sin(np.arange(len(index)) / 9.0),
        index=index,
    )
    actual = raw_oof * 0.45 + 0.006
    metadata = efp.fit_oof_return_calibration(raw_oof, actual, horizon=30)

    assert metadata["applied"] is True
    assert 0.0 < metadata["slope"] < 0.75

    training = pd.DataFrame({"target_return": actual}, index=index)
    adjusted, calibrated_oof, applied_metadata = efp.apply_oof_return_calibration(
        predicted_return=0.18,
        residual_oof_prediction=raw_oof,
        training_dataset=training,
        horizon=30,
    )
    assert applied_metadata["applied"] is True
    assert calibrated_oof is not None
    assert abs(adjusted) < 0.18


def test_prediction_history_loader_allows_empty_file(tmp_path) -> None:
    history_path = tmp_path / "prediction_history.csv"
    history_path.write_text("", encoding="utf-8")

    history = efp.load_prediction_history_csv(history_path)

    assert history.empty


def test_feature_selection_stability_is_fold_auditable() -> None:
    report = summarize_selected_features_by_fold(
        {
            0: ["always", "twice"],
            1: ["always", "once"],
            2: ["always", "twice"],
        },
        candidate_feature_count=4,
        target_column="target_return",
    )

    assert report["folds_analyzed"] == 3
    assert report["selected_feature_count_median"] == 2.0
    assert report["stable_feature_count_50pct"] == 2
    assert report["stable_feature_count_80pct"] == 1
    assert report["selection_counts"] == {"always": 3, "once": 1, "twice": 2}
    assert report["selected_features_by_fold"]["1"] == ["always", "once"]
    assert report["top_features"][0]["feature"] == "always"


def test_empirical_probability_percentiles_are_monotonic() -> None:
    reference = np.array([0.10, 0.20, 0.40, 0.80])
    values = np.array([0.05, 0.10, 0.30, 0.80, 0.95])

    mapped = efp.empirical_probability_percentiles(values, reference)

    assert np.all(np.diff(mapped) >= 0.0)
    assert np.all((mapped > 0.0) & (mapped < 1.0))
    assert mapped[0] < 0.5 < mapped[-1]


def test_classifier_uses_recent_train_probability_reference() -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="D")
    x = np.linspace(-3.0, 3.0, len(index))
    X = pd.DataFrame(
        {
            "trend": x,
            "cycle": np.sin(np.arange(len(index)) / 9.0),
        },
        index=index,
    )
    y = pd.Series((X["cycle"].to_numpy() + 0.15 * x > 0.0).astype(int), index=index)

    fitted = efp.fit_calibrated_classifier(
        efp.make_classification_models(horizon=7)["logistic"],
        X,
        y,
        min_calibration_rows=80,
        horizon=7,
    )

    assert fitted.calibration_method == "isotonic_empirical_cdf_holdout_shrunk"
    assert fitted.probability_mapping == "empirical_cdf"
    assert fitted.calibrator is not None
    assert len(np.asarray(fitted.probability_reference)) == 80
    probability = fitted.predict_proba(X.tail(5))[:, 1]
    raw_probability = fitted.base_estimator.predict_proba(X.tail(5))[:, 1]
    percentile = efp.empirical_probability_percentiles(
        raw_probability,
        fitted.probability_reference,
    )
    direction_score = fitted.predict_direction_score(X.tail(5))
    assert np.all((probability > 0.0) & (probability < 1.0))
    assert np.all(np.diff(probability[np.argsort(raw_probability)]) >= 0.0)
    assert not np.allclose(probability, percentile)
    assert np.allclose(direction_score, percentile)


@pytest.mark.parametrize(
    ("horizon", "expected_shadow_rows"),
    [(7, 153), (30, 130)],
)
def test_classifier_calibration_boundary_is_purged(
    horizon: int,
    expected_shadow_rows: int,
) -> None:
    index = pd.date_range("2024-01-01", periods=240, freq="D")
    X = pd.DataFrame({"signal": np.linspace(-3.0, 3.0, len(index))}, index=index)
    y = pd.Series(np.arange(len(index)) % 2, index=index)
    _RecordingClassifier.fit_windows.clear()

    efp.fit_calibrated_classifier(
        _RecordingClassifier(),
        X,
        y,
        min_calibration_rows=80,
        horizon=horizon,
    )

    assert len(_RecordingClassifier.fit_windows) == 2
    shadow_window, final_window = _RecordingClassifier.fit_windows
    calibration_start = index[-80]
    assert len(shadow_window) == expected_shadow_rows
    assert shadow_window[-1] + pd.Timedelta(days=horizon) < calibration_start
    assert final_window.equals(index)


def test_oof_direction_scores_fall_back_per_legacy_row() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    oof = pd.DataFrame(
        {
            "model_prob_up": [0.41, 0.42, 0.43, 0.44],
            "model_direction_score_up": [np.nan, np.nan, 0.71, 0.72],
        },
        index=index,
    )

    scores = efp.classification_oof_direction_scores(oof, "model")

    assert np.allclose(scores.to_numpy(), [0.41, 0.42, 0.71, 0.72])


def test_rank_calibration_does_not_expose_weak_score_tails_as_certainty() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2023-01-01", periods=360, freq="D")
    X = pd.DataFrame(
        rng.normal(size=(len(index), 3)),
        columns=["a", "b", "c"],
        index=index,
    )
    y = pd.Series(rng.integers(0, 2, size=len(index)), index=index)

    fitted = efp.fit_calibrated_classifier(
        efp.make_classification_models(horizon=7)["logistic"],
        X,
        y,
        min_calibration_rows=100,
        horizon=7,
    )
    raw_train = fitted.base_estimator.predict_proba(X)[:, 1]
    extreme_rows = X.iloc[[int(np.argmin(raw_train)), int(np.argmax(raw_train))]]
    raw_extremes = fitted.base_estimator.predict_proba(extreme_rows)[:, 1]
    percentiles = efp.empirical_probability_percentiles(
        raw_extremes,
        fitted.probability_reference,
    )
    calibrated = fitted.predict_proba(extreme_rows)[:, 1]
    direction_scores = fitted.predict_direction_score(extreme_rows)

    assert percentiles[0] < 0.05 and percentiles[1] > 0.95
    assert np.allclose(direction_scores, percentiles)
    assert np.max(np.abs(calibrated - 0.5)) < np.max(np.abs(percentiles - 0.5))


def test_rank_calibration_respects_time_decay_effective_sample_size() -> None:
    scores = np.linspace(0.05, 0.95, 100)
    labels = np.tile([0, 1], 50)
    sample_weight = np.full(100, 0.5)

    calibrator, method = efp.fit_rank_probability_calibrator(
        scores,
        labels,
        sample_weight=sample_weight,
    )

    assert method == "isotonic_empirical_cdf_holdout_shrunk"
    assert isinstance(calibrator, efp.ShrunkIsotonicProbabilityCalibrator)
    assert np.isclose(calibrator.learned_weight, 50.0 / 250.0)


def test_threshold_selection_uses_direction_score_but_brier_uses_probability() -> None:
    index = pd.date_range("2024-01-01", periods=120, freq="D")
    actual = pd.Series(([0, 1] * 60), index=index)
    direction_score = pd.Series(np.where(actual.eq(1), 0.70, 0.30), index=index)
    calibrated_probability = pd.Series(np.where(actual.eq(1), 0.56, 0.44), index=index)

    threshold, metrics = efp.choose_classification_evaluation_threshold(
        actual_label=actual,
        probability_up=calibrated_probability,
        direction_score_up=direction_score,
        horizon=7,
    )

    assert 0.50 <= threshold <= 0.70
    assert np.isclose(metrics["balanced_accuracy"], 1.0)
    assert np.isclose(metrics["roc_auc"], 1.0)
    assert np.isclose(metrics["brier_score"], (0.44**2))


@pytest.mark.parametrize(
    ("predicted_direction", "probability_up", "direction_score_up"),
    [("UP", 0.40, 0.95), ("DOWN", 0.60, 0.05)],
)
def test_direction_confidence_is_capped_by_selected_class_probability(
    predicted_direction: str,
    probability_up: float,
    direction_score_up: float,
) -> None:
    leaderboard = pd.DataFrame([{
        "model": "model",
        "balanced_accuracy": 0.60,
        "roc_auc": 0.70,
        "f1": 0.60,
    }])
    backtest = pd.DataFrame([{"model": "model", "total_return": 0.10, "sharpe": 1.0}])
    holdout = pd.DataFrame([{
        "task": "classification",
        "model": "model",
        "total_return": 0.10,
        "sharpe": 1.0,
    }])

    confidence = efp.calibrate_direction_confidence(
        model_name="model",
        selection_basis="validated",
        predicted_direction=predicted_direction,
        probability_up=probability_up,
        direction_score_up=direction_score_up,
        lower_threshold=0.40,
        upper_threshold=0.60,
        classification_leaderboard=leaderboard,
        classification_backtest=backtest,
        recent_holdout_report=holdout,
        horizon=7,
    )

    selected_probability = (
        probability_up if predicted_direction == "UP" else 1.0 - probability_up
    )
    assert confidence == pytest.approx(selected_probability)


def test_latest_summary_carries_live_direction_score() -> None:
    regression = _DefaultAttributes(
        model_name="regression",
        selection_basis="test",
        prediction_timestamp="2026-09-01 00:00:00",
        last_close=2000.0,
        reference_price_source="test",
        reference_price_timestamp="2026-08-25 00:00:00",
        model_input_close=2000.0,
    )
    classification = _DefaultAttributes(
        model_name="classification",
        selection_basis="test",
        predicted_direction="UP",
        signal_threshold=0.60,
        direction_score_up=0.73,
        probability_up=0.56,
        probability_down=0.44,
        confidence=0.56,
    )
    horizon_artifacts = _DefaultAttributes(
        data_window_summary=pd.DataFrame([{
            "latest_prediction_input_timestamp": "2026-08-25 00:00:00",
        }]),
        regression_backtest=pd.DataFrame(),
        classification_backtest=pd.DataFrame(),
        regime_backtest=pd.DataFrame(),
        reversal_backtest=pd.DataFrame(),
        regression_forecast=regression,
        classification_forecast=classification,
        regime_forecast=_DefaultAttributes(model_name="regime", selection_basis="test"),
        reversal_forecast=_DefaultAttributes(model_name="reversal", selection_basis="test"),
        hybrid_forecast=_DefaultAttributes(model_name="hybrid", selection_basis="test"),
    )
    artifacts = _DefaultAttributes(horizons={7: horizon_artifacts})

    summary = efp.build_latest_forecast_summary(artifacts)

    assert summary.loc[0, "classification_direction_score_up"] == pytest.approx(0.73)


def test_live_gap_adjustment_is_signed_and_bounded() -> None:
    index = pd.date_range("2026-08-20", periods=1, freq="D")
    frame = pd.DataFrame(index=index)
    base = pd.Series([0.50], index=index)

    up = efp.apply_direction_live_gap_adjustment(base, frame, live_gap=0.08)
    down = efp.apply_direction_live_gap_adjustment(base, frame, live_gap=-0.08)
    capped = efp.apply_direction_live_gap_adjustment(base, frame, live_gap=0.50)

    assert np.isclose(up.iloc[0], 0.515)
    assert np.isclose(down.iloc[0], 0.485)
    assert np.isclose(capped.iloc[0], 0.515)


def test_chunk_fold_progress_counts_rows_instead_of_last_index() -> None:
    rows = [
        {"horizon_days": 7, "fold_index": 33},
        {"horizon_days": 7, "fold_index": 33},
        {"horizon_days": 7, "fold_index": 34},
        {"horizon_days": 7, "fold_index": 35},
        {"horizon_days": 30, "fold_index": 35},
        {"horizon_days": 7, "fold_index": -1},
        {"horizon_days": 7, "fold_index": None},
    ]

    assert prediction_fold_indices(rows, horizon=7) == {33, 34, 35}


def test_legacy_checkpoint_row_is_backfilled_with_direction_score() -> None:
    rows = [{
        "head": "classification",
        "model": "legacy_model",
        "probability_up": 0.72,
        "predicted_label": None,
    }]

    backfill_classification_prediction_rows(rows, {"legacy_model": 0.60})

    assert rows[0]["direction_score_up"] == pytest.approx(0.72)
    assert rows[0]["predicted_label"] == 1
