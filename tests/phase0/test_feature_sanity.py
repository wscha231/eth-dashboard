"""Phase 0 sanity checks on build_features — lightweight, no live data required."""
from __future__ import annotations

import numpy as np
import pandas as pd

import eth_price_forecast as efp


def test_build_features_returns_nonempty_columns(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    frame, columns = efp.build_features(synthetic_ohlcv_with_companions, horizon=7)
    assert len(columns) > 50, f"Expected >50 feature columns, got {len(columns)}"
    assert "target_return" in frame.columns
    assert "target_close" in frame.columns


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


def test_time_series_splitter_respects_gap() -> None:
    splitter = efp.safe_time_series_split(n_samples=500, n_splits=3, test_size=30, gap=7)
    X = pd.DataFrame(np.zeros((500, 3)))
    for train_idx, test_idx in splitter.split(X):
        assert test_idx.min() - train_idx.max() > 7, (
            f"Gap violated: train_max={train_idx.max()} test_min={test_idx.min()}"
        )
