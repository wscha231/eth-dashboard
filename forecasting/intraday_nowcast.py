"""Causal intraday breakout nowcast primitives for offline evaluation.

The rules in this module are deliberately separate from the daily forecast.
They use only fully closed hourly bars and rolling thresholds whose windows end
before the current decision bar.  A successful offline gate may authorize a
shadow experiment, never an automatic production promotion.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

NOWCAST_HORIZON_HOURS = 48
NOWCAST_RETURN_THRESHOLD = 0.08
THRESHOLD_LOOKBACK_HOURS = 24 * 180
THRESHOLD_MIN_OBSERVATIONS = 24 * 90
PAST_COMPLETENESS_HOURS = 24
ALERT_REFRESH_HOURS = 24
EVENT_MERGE_GAP_HOURS = 24
MAX_FALSE_ALERTS_PER_90_DAYS = 3.0

PRICE_IMPULSE = "price_impulse"
VOLUME_CONFIRMED = "volume_confirmed"
CROSS_MARKET_CONFIRMED = "cross_market_confirmed"
CANDIDATES = (PRICE_IMPULSE, VOLUME_CONFIRMED, CROSS_MARKET_CONFIRMED)

REQUIRED_STREAMS = (
    "binance_spot_ethusdt_1h",
    "binance_um_ethusdt_1h",
    "binance_spot_btcusdt_1h",
    "binance_um_btcusdt_1h",
)
REQUIRED_COLUMNS = (
    "open_time",
    "close_time",
    "close",
    "quote_volume",
    "taker_buy_quote_volume",
)


@dataclass(frozen=True)
class PositiveEvent:
    event_id: int
    start: pd.Timestamp
    end: pd.Timestamp


def _normalize_stream(
    frame: pd.DataFrame,
    prefix: str,
    excluded_dates: set[object],
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"{prefix} is missing hourly columns: {missing}")
    if frame.empty:
        raise ValueError(f"{prefix} is empty")

    output = frame.loc[:, REQUIRED_COLUMNS].copy()
    output["open_time"] = pd.to_datetime(
        output["open_time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    output["close_time"] = pd.to_datetime(
        output["close_time"], utc=True, errors="coerce"
    ).dt.tz_convert(None)
    for column in ("close", "quote_volume", "taker_buy_quote_volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if output.isna().any().any():
        raise ValueError(f"{prefix} contains invalid hourly values")
    if bool((output["close"] <= 0.0).any()):
        raise ValueError(f"{prefix} contains non-positive close values")
    invalid_volume = (
        (output["quote_volume"] < 0.0)
        | (output["taker_buy_quote_volume"] < 0.0)
        | (output["taker_buy_quote_volume"] > output["quote_volume"])
    )
    if bool(invalid_volume.any()):
        raise ValueError(f"{prefix} contains invalid quote-volume values")
    if output["open_time"].duplicated().any():
        raise ValueError(f"{prefix} contains duplicate open times")

    if excluded_dates:
        output = output.loc[~output["open_time"].dt.date.isin(excluded_dates)].copy()
        if output.empty:
            raise ValueError(f"{prefix} is empty after date quarantine")

    output = output.sort_values("open_time")
    durations = output["close_time"] - output["open_time"]
    valid_duration = (
        durations > pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=2)
    ) & (durations <= pd.Timedelta(hours=1))
    aligned = output["open_time"].eq(output["open_time"].dt.floor("h"))
    if not bool((valid_duration & aligned).all()):
        raise ValueError(f"{prefix} contains irregular hourly bars")

    output["decision_time"] = output["open_time"] + pd.Timedelta(hours=1)
    output = output.set_index("decision_time")
    return output.rename(
        columns={
            "close": f"{prefix}_close",
            "quote_volume": f"{prefix}_quote_volume",
            "taker_buy_quote_volume": f"{prefix}_taker_buy_quote_volume",
        }
    ).drop(columns=["open_time", "close_time"])


def _prior_quantile(series: pd.Series, quantile: float) -> pd.Series:
    return (
        series.shift(1)
        .rolling(
            f"{THRESHOLD_LOOKBACK_HOURS}h",
            min_periods=THRESHOLD_MIN_OBSERVATIONS,
        )
        .quantile(quantile)
    )


def _forward_targets(close: pd.Series) -> pd.DataFrame:
    values = close.to_numpy(dtype=float)
    horizon = NOWCAST_HORIZON_HOURS
    maximum = np.full(len(values), np.nan, dtype=float)
    hours_to_target = np.full(len(values), np.nan, dtype=float)
    if len(values) <= horizon:
        return pd.DataFrame(
            {
                "future_max_return_48h": maximum,
                "hours_to_target": hours_to_target,
                "target_up_8pct_48h": np.full(len(values), np.nan),
            },
            index=close.index,
        )

    windows = np.lib.stride_tricks.sliding_window_view(values, horizon + 1)
    complete = np.isfinite(windows).all(axis=1)
    origins = windows[:, [0]]
    forward_returns = windows[:, 1:] / origins - 1.0
    valid_maximum = np.max(forward_returns, axis=1)
    maximum[: len(windows)] = np.where(complete, valid_maximum, np.nan)
    hits = forward_returns >= NOWCAST_RETURN_THRESHOLD
    any_hit = hits.any(axis=1) & complete
    first_hit = np.argmax(hits, axis=1) + 1
    hours_to_target[: len(windows)] = np.where(any_hit, first_hit, np.nan)
    labels = np.where(
        np.isfinite(maximum),
        (maximum >= NOWCAST_RETURN_THRESHOLD).astype(float),
        np.nan,
    )
    return pd.DataFrame(
        {
            "future_max_return_48h": maximum,
            "hours_to_target": hours_to_target,
            "target_up_8pct_48h": labels,
        },
        index=close.index,
    )


def build_nowcast_table(
    streams: dict[str, pd.DataFrame],
    *,
    excluded_dates: tuple[pd.Timestamp, ...] = (),
) -> pd.DataFrame:
    """Build matched hourly features, causal thresholds, targets, and alerts."""
    if set(streams) != set(REQUIRED_STREAMS):
        raise ValueError("Nowcast requires exactly four matched Binance streams")

    excluded_day_values = {pd.Timestamp(value).date() for value in excluded_dates}
    normalized = {
        "eth_spot": _normalize_stream(
            streams["binance_spot_ethusdt_1h"], "eth_spot", excluded_day_values
        ),
        "eth_perp": _normalize_stream(
            streams["binance_um_ethusdt_1h"], "eth_perp", excluded_day_values
        ),
        "btc_spot": _normalize_stream(
            streams["binance_spot_btcusdt_1h"], "btc_spot", excluded_day_values
        ),
        "btc_perp": _normalize_stream(
            streams["binance_um_btcusdt_1h"], "btc_perp", excluded_day_values
        ),
    }
    start = max(frame.index.min() for frame in normalized.values())
    end = min(frame.index.max() for frame in normalized.values())
    if start >= end:
        raise ValueError("Hourly streams have no common window")
    grid = pd.date_range(start.floor("h"), end.floor("h"), freq="h")
    joined = pd.concat(
        [frame.reindex(grid) for frame in normalized.values()], axis=1, join="inner"
    )
    joined.index.name = "decision_time"

    if excluded_day_values:
        bad = pd.Series(
            [value.date() in excluded_day_values for value in joined.index], index=grid
        )
        joined.loc[bad, :] = np.nan

    available = joined.notna().all(axis=1)
    past_complete = (
        available.rolling(PAST_COMPLETENESS_HOURS, min_periods=PAST_COMPLETENESS_HOURS)
        .sum()
        .eq(PAST_COMPLETENESS_HOURS)
    )

    eth_close = joined["eth_spot_close"]
    btc_close = joined["btc_spot_close"]
    features = pd.DataFrame(index=grid)
    features["eth_return_1h"] = eth_close.pct_change(1, fill_method=None)
    features["eth_return_4h"] = eth_close.pct_change(4, fill_method=None)
    features["eth_return_12h"] = eth_close.pct_change(12, fill_method=None)
    features["eth_return_24h"] = eth_close.pct_change(24, fill_method=None)
    features["btc_return_4h"] = btc_close.pct_change(4, fill_method=None)
    features["eth_btc_relative_return_4h"] = (
        features["eth_return_4h"] - features["btc_return_4h"]
    )

    quote_volume = joined["eth_spot_quote_volume"]
    taker_buy = joined["eth_spot_taker_buy_quote_volume"]
    features["eth_quote_volume_4h"] = quote_volume.rolling(4, min_periods=4).sum()
    features["eth_taker_buy_share_4h"] = taker_buy.rolling(
        4, min_periods=4
    ).sum() / features["eth_quote_volume_4h"].where(
        features["eth_quote_volume_4h"].abs() > 1e-15
    )
    features["eth_perp_basis"] = (
        joined["eth_perp_close"] / joined["eth_spot_close"] - 1.0
    )
    features["btc_perp_basis"] = (
        joined["btc_perp_close"] / joined["btc_spot_close"] - 1.0
    )
    features["eth_btc_basis_spread"] = (
        features["eth_perp_basis"] - features["btc_perp_basis"]
    )
    features["eth_btc_basis_spread_change_4h"] = features[
        "eth_btc_basis_spread"
    ] - features["eth_btc_basis_spread"].shift(4)

    features["prior_q98_eth_return_4h"] = _prior_quantile(
        features["eth_return_4h"], 0.98
    )
    features["prior_q99_eth_return_4h"] = _prior_quantile(
        features["eth_return_4h"], 0.99
    )
    features["prior_q90_eth_quote_volume_4h"] = _prior_quantile(
        features["eth_quote_volume_4h"], 0.90
    )
    features["prior_q75_eth_btc_relative_return_4h"] = _prior_quantile(
        features["eth_btc_relative_return_4h"], 0.75
    )
    features["prior_q60_eth_btc_basis_spread_change_4h"] = _prior_quantile(
        features["eth_btc_basis_spread_change_4h"], 0.60
    )

    targets = _forward_targets(eth_close)
    output = pd.concat([joined, features, targets], axis=1)
    threshold_columns = [column for column in features if column.startswith("prior_q")]
    eligible = (
        past_complete
        & output[threshold_columns].notna().all(axis=1)
        & output["target_up_8pct_48h"].notna()
    )
    output = output.loc[eligible].copy()
    output[PRICE_IMPULSE] = output["eth_return_4h"] >= output["prior_q99_eth_return_4h"]
    output[VOLUME_CONFIRMED] = (
        (output["eth_return_4h"] >= output["prior_q98_eth_return_4h"])
        & (output["eth_quote_volume_4h"] >= output["prior_q90_eth_quote_volume_4h"])
        & (output["eth_taker_buy_share_4h"] >= 0.52)
    )
    output[CROSS_MARKET_CONFIRMED] = (
        (output["eth_return_4h"] >= output["prior_q98_eth_return_4h"])
        & (
            output["eth_btc_relative_return_4h"]
            >= output["prior_q75_eth_btc_relative_return_4h"]
        )
        & (
            output["eth_btc_basis_spread_change_4h"]
            >= output["prior_q60_eth_btc_basis_spread_change_4h"]
        )
        & (output["eth_taker_buy_share_4h"] >= 0.50)
    )
    return output


def select_alert_origins(
    alerts: pd.Series,
    *,
    refresh_hours: int = ALERT_REFRESH_HOURS,
) -> pd.DatetimeIndex:
    """Count a sustained alarm again every refresh window."""
    selected: list[pd.Timestamp] = []
    last: pd.Timestamp | None = None
    for timestamp in alerts.index[alerts.astype(bool)]:
        current = pd.Timestamp(timestamp)
        if last is None or current - last >= pd.Timedelta(hours=refresh_hours):
            selected.append(current)
            last = current
    return pd.DatetimeIndex(selected)


def positive_events(
    labels: pd.Series,
    *,
    merge_gap_hours: int = EVENT_MERGE_GAP_HOURS,
) -> tuple[PositiveEvent, ...]:
    positives = pd.DatetimeIndex(labels.index[labels.astype(bool)])
    if positives.empty:
        return ()
    events: list[PositiveEvent] = []
    start = positives[0]
    previous = positives[0]
    for timestamp in positives[1:]:
        if timestamp - previous > pd.Timedelta(hours=merge_gap_hours):
            events.append(PositiveEvent(len(events) + 1, start, previous))
            start = timestamp
        previous = timestamp
    events.append(PositiveEvent(len(events) + 1, start, previous))
    return tuple(events)


def score_candidate(table: pd.DataFrame, candidate: str) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown nowcast candidate: {candidate}")
    if table.empty:
        raise ValueError("Cannot score an empty nowcast table")
    labels = table["target_up_8pct_48h"].astype(bool)
    events = positive_events(labels)
    selected = select_alert_origins(table[candidate].astype(bool))
    selected_frame = table.loc[selected]
    true_alerts = selected_frame.loc[selected_frame["target_up_8pct_48h"].astype(bool)]
    false_alerts = selected_frame.loc[
        ~selected_frame["target_up_8pct_48h"].astype(bool)
    ]

    detected_events: set[int] = set()
    first_alerts: list[dict[str, Any]] = []
    for event in events:
        matched = true_alerts.loc[event.start : event.end]
        if matched.empty:
            continue
        first = matched.iloc[0]
        timestamp = pd.Timestamp(matched.index[0])
        detected_events.add(event.event_id)
        first_alerts.append(
            {
                "event_id": event.event_id,
                "event_start": event.start.isoformat(),
                "event_end": event.end.isoformat(),
                "first_alert": timestamp.isoformat(),
                "latency_from_label_start_hours": float(
                    (timestamp - event.start) / pd.Timedelta(hours=1)
                ),
                "hours_to_target": float(first["hours_to_target"]),
                "future_max_return_48h": float(first["future_max_return_48h"]),
            }
        )

    duration_hours = max(
        float((table.index.max() - table.index.min()) / pd.Timedelta(hours=1)),
        1.0,
    )
    alert_count = len(selected_frame)
    true_count = len(true_alerts)
    return {
        "candidate": candidate,
        "rows": len(table),
        "start": table.index.min().isoformat(),
        "end": table.index.max().isoformat(),
        "event_count": len(events),
        "detected_event_count": len(detected_events),
        "event_recall": len(detected_events) / len(events) if events else None,
        "alert_episode_count": alert_count,
        "true_alert_episode_count": true_count,
        "false_alert_episode_count": len(false_alerts),
        "alert_precision": true_count / alert_count if alert_count else None,
        "false_alerts_per_90_days": len(false_alerts)
        / (duration_hours / (24.0 * 90.0)),
        "median_hours_to_target": (
            float(true_alerts["hours_to_target"].median())
            if not true_alerts.empty
            else None
        ),
        "median_future_max_return_48h": (
            float(true_alerts["future_max_return_48h"].median())
            if not true_alerts.empty
            else None
        ),
        "detected_calendar_blocks": len(
            {pd.Timestamp(item["first_alert"]).year for item in first_alerts}
        ),
        "event_first_alerts": first_alerts,
        "false_alert_samples": [value.isoformat() for value in false_alerts.index[:20]],
    }


def evaluate_candidates(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {candidate: score_candidate(table, candidate) for candidate in CANDIDATES}
