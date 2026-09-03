from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_HORIZON_DAYS = 3
DEFAULT_TAIL_THRESHOLD = 0.12
TAIL_CLASSES = ("DOWN_TAIL", "NORMAL", "UP_TAIL")


def _numeric_series(values: pd.Series, name: str) -> pd.Series:
    output = pd.to_numeric(values, errors="coerce").astype(float)
    output.name = name
    if not output.index.is_monotonic_increasing:
        output = output.sort_index()
    if output.index.duplicated().any():
        raise ValueError(f"{name} contains duplicate dates")
    return output


def _nullable_binary(condition: pd.Series, valid: pd.Series) -> pd.Series:
    output = pd.Series(pd.NA, index=condition.index, dtype="Int8")
    output.loc[valid] = condition.loc[valid].astype("int8")
    return output


def assign_event_episodes(
    event: pd.Series,
    *,
    maximum_origin_gap_days: int = DEFAULT_HORIZON_DAYS,
) -> pd.Series:
    if maximum_origin_gap_days < 1:
        raise ValueError("maximum_origin_gap_days must be positive")
    event_values = event.fillna(0).astype(bool)
    dates = pd.DatetimeIndex(pd.to_datetime(event_values.index))
    output = pd.Series(pd.NA, index=event.index, dtype="Int64")
    episode = 0
    previous_positive: pd.Timestamp | None = None
    for position in np.flatnonzero(event_values.to_numpy()):
        current = pd.Timestamp(dates[position])
        if (
            previous_positive is None
            or (current - previous_positive).days > maximum_origin_gap_days
        ):
            episode += 1
        output.iloc[position] = episode
        previous_positive = current
    return output


def build_tail_targets(
    close: pd.Series,
    *,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    threshold: float = DEFAULT_TAIL_THRESHOLD,
) -> pd.DataFrame:
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    if not 0 < threshold < 1:
        raise ValueError("threshold must be in (0, 1)")
    close_values = _numeric_series(close, "close")
    forward_return = close_values.shift(-horizon_days) / close_values - 1.0
    valid = close_values.gt(0) & forward_return.notna() & np.isfinite(forward_return)

    output = pd.DataFrame(index=close_values.index)
    output[f"forward_return_{horizon_days}d"] = forward_return.where(valid)
    output["tail_up_primary"] = _nullable_binary(forward_return.ge(threshold), valid)
    output["tail_down_primary"] = _nullable_binary(forward_return.le(-threshold), valid)
    output["large_move_primary"] = _nullable_binary(
        forward_return.abs().ge(threshold), valid
    )
    output["direction_up"] = _nullable_binary(forward_return.gt(0), valid)

    tail_class = pd.Series(pd.NA, index=close_values.index, dtype="string")
    tail_class.loc[valid] = "NORMAL"
    tail_class.loc[valid & forward_return.ge(threshold)] = "UP_TAIL"
    tail_class.loc[valid & forward_return.le(-threshold)] = "DOWN_TAIL"
    output["tail_class"] = tail_class
    output["tail_up_episode_id"] = assign_event_episodes(
        output["tail_up_primary"], maximum_origin_gap_days=horizon_days
    )
    output["tail_down_episode_id"] = assign_event_episodes(
        output["tail_down_primary"], maximum_origin_gap_days=horizon_days
    )
    output["large_move_episode_id"] = assign_event_episodes(
        output["large_move_primary"], maximum_origin_gap_days=horizon_days
    )
    return output


def build_barrier_diagnostics(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    horizon_days: int = DEFAULT_HORIZON_DAYS,
    threshold: float = DEFAULT_TAIL_THRESHOLD,
) -> pd.DataFrame:
    if horizon_days < 1:
        raise ValueError("horizon_days must be positive")
    close_values = _numeric_series(close, "close")
    high_values = _numeric_series(high, "high").reindex(close_values.index)
    low_values = _numeric_series(low, "low").reindex(close_values.index)
    future_highs = pd.concat(
        [high_values.shift(-offset) for offset in range(1, horizon_days + 1)],
        axis=1,
    )
    future_lows = pd.concat(
        [low_values.shift(-offset) for offset in range(1, horizon_days + 1)],
        axis=1,
    )
    valid = (
        close_values.gt(0)
        & future_highs.notna().all(axis=1)
        & future_lows.notna().all(axis=1)
    )
    up = future_highs.max(axis=1).ge(close_values * (1.0 + threshold))
    down = future_lows.min(axis=1).le(close_values * (1.0 - threshold))

    output = pd.DataFrame(index=close_values.index)
    output["up_barrier_hit"] = _nullable_binary(up, valid)
    output["down_barrier_hit"] = _nullable_binary(down, valid)
    output["either_barrier_hit"] = _nullable_binary(up | down, valid)
    output["both_barriers_hit"] = _nullable_binary(up & down, valid)
    output["barrier_order_ambiguous"] = output["both_barriers_hit"].copy()
    return output
