from __future__ import annotations

import pandas as pd
import pytest

from forecasting.tail_events import (
    assign_event_episodes,
    build_barrier_diagnostics,
    build_tail_targets,
)


def test_factorized_and_multiclass_targets_include_exact_boundaries() -> None:
    index = pd.date_range("2020-01-01", periods=8, freq="D")
    close = pd.Series(
        [100.0, 100.0, 100.0, 112.0, 88.0, 101.0, 110.0, 90.0], index=index
    )
    targets = build_tail_targets(close, horizon_days=3, threshold=0.12)

    assert targets.loc[index[0], "forward_return_3d"] == pytest.approx(0.12)
    assert targets.loc[index[0], "tail_up_primary"] == 1
    assert targets.loc[index[0], "large_move_primary"] == 1
    assert targets.loc[index[0], "direction_up"] == 1
    assert targets.loc[index[0], "tail_class"] == "UP_TAIL"

    assert targets.loc[index[1], "forward_return_3d"] == pytest.approx(-0.12)
    assert targets.loc[index[1], "tail_down_primary"] == 1
    assert targets.loc[index[1], "large_move_primary"] == 1
    assert targets.loc[index[1], "direction_up"] == 0
    assert targets.loc[index[1], "tail_class"] == "DOWN_TAIL"

    assert targets.loc[index[2], "large_move_primary"] == 0
    assert targets.loc[index[2], "tail_class"] == "NORMAL"
    assert targets.iloc[-3:]["tail_up_primary"].isna().all()
    assert targets.iloc[-3:]["tail_class"].isna().all()


def test_event_episode_ids_join_only_overlapping_origins() -> None:
    index = pd.date_range("2020-01-01", periods=9, freq="D")
    event = pd.Series([1, 0, 1, 0, 0, 0, 1, 0, 1], index=index)
    episodes = assign_event_episodes(event, maximum_origin_gap_days=3)

    assert episodes.loc[index[0]] == 1
    assert episodes.loc[index[2]] == 1
    assert episodes.loc[index[6]] == 2
    assert episodes.loc[index[8]] == 2
    assert pd.isna(episodes.loc[index[1]])


def test_barrier_labels_are_diagnostic_and_horizon_safe() -> None:
    index = pd.date_range("2020-01-01", periods=6, freq="D")
    close = pd.Series([100.0] * 6, index=index)
    high = pd.Series([100.0, 113.0, 100.0, 100.0, 100.0, 100.0], index=index)
    low = pd.Series([100.0, 100.0, 87.0, 100.0, 100.0, 100.0], index=index)
    labels = build_barrier_diagnostics(
        close=close,
        high=high,
        low=low,
        horizon_days=3,
        threshold=0.12,
    )

    assert labels.loc[index[0], "up_barrier_hit"] == 1
    assert labels.loc[index[0], "down_barrier_hit"] == 1
    assert labels.loc[index[0], "both_barriers_hit"] == 1
    assert labels.loc[index[0], "barrier_order_ambiguous"] == 1
    assert labels.iloc[-3:]["either_barrier_hit"].isna().all()
