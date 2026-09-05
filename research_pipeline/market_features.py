"""UTC hourly aggregation reused from PR #11 (5e645555); market-only subset."""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

STREAM_PREFIXES = {
    "binance_spot_ethusdt_1h": "eth_spot",
    "binance_um_ethusdt_1h": "eth_perp",
    "binance_spot_btcusdt_1h": "btc_spot",
    "binance_um_btcusdt_1h": "btc_perp",
}
REQUIRED_STREAM_IDS = tuple(STREAM_PREFIXES)
CHANGE_WINDOWS = (1, 3, 7)
FINAL_WINDOWS = (4, 8, 12)
UTC_DAY = pd.Timedelta(days=1)

HOURLY_NUMERIC_COLUMNS = (
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "trade_count",
    "taker_buy_quote_volume",
)


def _naive_utc(value: Any) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = pd.to_numeric(denominator, errors="coerce")
    output = pd.to_numeric(numerator, errors="coerce") / denominator.where(
        denominator.abs() > 1e-15
    )
    return output.replace([np.inf, -np.inf], np.nan)


def _normalize_hourly_frame(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    excluded_dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    required = {"open_time", "close_time", *HOURLY_NUMERIC_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing hourly columns: {missing}")
    if frame.empty:
        raise ValueError("Cannot aggregate an empty hourly frame")

    output = frame.copy()
    output["open_time"] = pd.to_datetime(output["open_time"], utc=True, errors="coerce")
    output["close_time"] = pd.to_datetime(
        output["close_time"], utc=True, errors="coerce"
    )
    output["open_time"] = output["open_time"].dt.tz_convert(None)
    output["close_time"] = output["close_time"].dt.tz_convert(None)
    if output[["open_time", "close_time"]].isna().any().any():
        raise ValueError("Hourly frame contains invalid timestamps")

    for column in HOURLY_NUMERIC_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    numeric = output.loc[:, HOURLY_NUMERIC_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Hourly frame contains non-finite numeric values")
    if output["open_time"].duplicated().any():
        raise ValueError("Hourly frame contains duplicate open times")

    output = output.sort_values("open_time").reset_index(drop=True)
    if not output["open_time"].is_monotonic_increasing:
        raise ValueError("Hourly frame is not monotonic after sorting")
    output = output.loc[output["close_time"] <= cutoff].copy()
    if output.empty:
        raise ValueError("No closed bars remain at the declared cutoff")
    output["date"] = output["open_time"].dt.floor("D")
    duration = output["close_time"] - output["open_time"]
    invalid_duration = ~(
        (duration > pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=2))
        & (duration <= pd.Timedelta(hours=1))
    )
    invalid_dates = pd.DatetimeIndex(output.loc[invalid_duration, "date"].unique())
    aligned = output["open_time"].dt.floor("h").eq(output["open_time"])
    unaligned_dates = pd.DatetimeIndex(output.loc[~aligned, "date"].unique())
    undeclared = invalid_dates.union(unaligned_dates).difference(excluded_dates)
    if len(undeclared):
        sample = ", ".join(value.date().isoformat() for value in undeclared[:5])
        raise ValueError(f"Undeclared irregular hourly sessions on UTC days: {sample}")
    return output


def _complete_utc_days(
    frame: pd.DataFrame,
    cutoff: pd.Timestamp,
    excluded_dates: pd.DatetimeIndex,
) -> pd.DatetimeIndex:
    counts = frame.groupby("date", sort=True)["open_time"].count()
    first_open = frame.groupby("date", sort=True)["open_time"].min()
    last_open = frame.groupby("date", sort=True)["open_time"].max()
    complete = (
        counts.eq(24)
        & first_open.eq(first_open.index)
        & last_open.eq(last_open.index + pd.Timedelta(hours=23))
    )
    available = complete.index + UTC_DAY <= cutoff
    complete = complete & available

    eligible_days = counts.index[counts.index + UTC_DAY <= cutoff]
    if len(eligible_days):
        expected = pd.date_range(eligible_days.min(), eligible_days.max(), freq="D")
        observed_complete = pd.DatetimeIndex(complete.index[complete])
        incomplete = expected.difference(observed_complete)
        first_observed = pd.Timestamp(counts.index.min())
        incomplete = incomplete[incomplete != first_observed]
        incomplete = incomplete.difference(excluded_dates)
        if len(incomplete):
            sample = ", ".join(value.date().isoformat() for value in incomplete[:5])
            raise ValueError(f"Incomplete interior UTC days: {sample}")
    return pd.DatetimeIndex(complete.index[complete])


def _daily_intraday_metrics(group: pd.DataFrame, prefix: str) -> dict[str, float]:
    returns = group["hourly_log_return"].to_numpy(dtype=float)
    squared = np.square(returns)
    realized_variance = float(np.nansum(squared))
    upside_semivariance = float(np.nansum(squared[returns > 0]))
    downside_semivariance = float(np.nansum(squared[returns < 0]))

    absolute = np.abs(returns)
    bipower = float((math.pi / 2.0) * np.nansum(absolute[1:] * absolute[:-1]))
    jump_ratio = (
        max(realized_variance - bipower, 0.0) / realized_variance
        if realized_variance > 1e-18
        else 0.0
    )
    quote_volume = float(group["quote_volume"].sum())
    taker_buy_quote = float(group["taker_buy_quote_volume"].sum())
    signed_flow = 2.0 * taker_buy_quote - quote_volume

    metrics: dict[str, float] = {
        f"{prefix}_close": float(group["close"].iloc[-1]),
        f"{prefix}_quote_volume": quote_volume,
        f"{prefix}_trade_count": float(group["trade_count"].sum()),
        f"{prefix}_taker_buy_quote_share": (
            taker_buy_quote / quote_volume if quote_volume > 0 else np.nan
        ),
        f"{prefix}_signed_taker_quote_flow": signed_flow,
        f"{prefix}_signed_taker_flow_ratio": (
            signed_flow / quote_volume if quote_volume > 0 else np.nan
        ),
        f"{prefix}_realized_volatility": math.sqrt(realized_variance),
        f"{prefix}_upside_semivariance": upside_semivariance,
        f"{prefix}_downside_semivariance": downside_semivariance,
        f"{prefix}_jump_variation_ratio": jump_ratio,
        f"{prefix}_bar_count": float(len(group)),
    }
    for hours in FINAL_WINDOWS:
        tail = group.iloc[-hours:]
        metrics[f"{prefix}_last_{hours}h_return"] = float(
            tail["close"].iloc[-1] / tail["open"].iloc[0] - 1.0
        )
        metrics[f"{prefix}_last_{hours}h_quote_volume_share"] = (
            float(tail["quote_volume"].sum()) / quote_volume
            if quote_volume > 0
            else np.nan
        )
    return metrics


def aggregate_hourly_stream(
    frame: pd.DataFrame,
    *,
    prefix: str,
    cutoff: Any,
    excluded_dates: tuple[Any, ...] = (),
) -> pd.DataFrame:
    """Aggregate only complete UTC days whose final bar closed by ``cutoff``."""

    cutoff_timestamp = _naive_utc(cutoff)
    excluded = pd.DatetimeIndex(
        [_naive_utc(value).floor("D") for value in excluded_dates]
    )
    hourly = _normalize_hourly_frame(frame, cutoff_timestamp, excluded)
    complete_days = _complete_utc_days(hourly, cutoff_timestamp, excluded)
    if complete_days.empty:
        raise ValueError("No complete UTC days are available at the declared cutoff")

    previous_close = hourly["close"].shift(1)
    gap = hourly["open_time"].diff().ne(pd.Timedelta(hours=1))
    previous_close.loc[gap] = hourly.loc[gap, "open"]
    hourly["hourly_log_return"] = np.log(hourly["close"] / previous_close)
    hourly = hourly.loc[hourly["date"].isin(complete_days)].copy()

    rows: list[dict[str, Any]] = []
    for date, group in hourly.groupby("date", sort=True):
        row: dict[str, Any] = {"date": pd.Timestamp(date)}
        row.update(_daily_intraday_metrics(group, prefix))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date").sort_index()


def _add_change_features(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        if column not in output:
            continue
        values = pd.to_numeric(output[column], errors="coerce")
        for window in CHANGE_WINDOWS:
            output[f"{column}_delta_{window}d"] = values.diff(window)
    return output


def build_market_daily_features(
    streams: dict[str, pd.DataFrame],
    *,
    cutoff: Any,
    excluded_dates: tuple[Any, ...] = (),
) -> pd.DataFrame:
    missing = sorted(set(REQUIRED_STREAM_IDS) - set(streams))
    if missing:
        raise ValueError(f"Missing required hourly streams: {missing}")
    cutoff_timestamp = _naive_utc(cutoff)

    daily_frames = []
    for source_id in REQUIRED_STREAM_IDS:
        daily_frames.append(
            aggregate_hourly_stream(
                streams[source_id],
                prefix=STREAM_PREFIXES[source_id],
                cutoff=cutoff_timestamp,
                excluded_dates=excluded_dates,
            )
        )
    output = pd.concat(daily_frames, axis=1, join="outer").sort_index()
    output = output.reindex(pd.date_range(output.index.min(), output.index.max(), freq="D"))
    output["market_data_excluded"] = 0.0
    excluded = pd.DatetimeIndex(
        [_naive_utc(value).floor("D") for value in excluded_dates]
    )
    present_exclusions = output.index.intersection(excluded)
    if len(present_exclusions):
        numeric = output.select_dtypes(include=[np.number]).columns
        output.loc[present_exclusions, numeric] = np.nan
        output.loc[present_exclusions, "market_data_excluded"] = 1.0

    for asset in ("eth", "btc"):
        output[f"{asset}_spot_perp_flow_divergence"] = (
            output[f"{asset}_perp_signed_taker_flow_ratio"]
            - output[f"{asset}_spot_signed_taker_flow_ratio"]
        )
        output[f"{asset}_perp_basis"] = (
            output[f"{asset}_perp_close"] / output[f"{asset}_spot_close"] - 1.0
        )
        output[f"{asset}_futures_spot_quote_volume_ratio"] = _safe_divide(
            output[f"{asset}_perp_quote_volume"],
            output[f"{asset}_spot_quote_volume"],
        )
        output[f"{asset}_futures_spot_trade_count_ratio"] = _safe_divide(
            output[f"{asset}_perp_trade_count"],
            output[f"{asset}_spot_trade_count"],
        )
        for window in (3, 7):
            output[f"{asset}_perp_basis_positive_share_{window}d"] = (
                output[f"{asset}_perp_basis"]
                .gt(0)
                .where(output[f"{asset}_perp_basis"].notna())
                .rolling(window, min_periods=window)
                .mean()
            )

    output["eth_btc_spot_flow_spread"] = (
        output["eth_spot_signed_taker_flow_ratio"]
        - output["btc_spot_signed_taker_flow_ratio"]
    )
    output["eth_btc_perp_flow_spread"] = (
        output["eth_perp_signed_taker_flow_ratio"]
        - output["btc_perp_signed_taker_flow_ratio"]
    )
    output["eth_btc_basis_spread"] = output["eth_perp_basis"] - output["btc_perp_basis"]
    for market in ("spot", "perp"):
        output[f"eth_btc_{market}_volatility_spread"] = (
            output[f"eth_{market}_realized_volatility"]
            - output[f"btc_{market}_realized_volatility"]
        )
        for hours in FINAL_WINDOWS:
            output[f"eth_btc_{market}_last_{hours}h_return_spread"] = (
                output[f"eth_{market}_last_{hours}h_return"]
                - output[f"btc_{market}_last_{hours}h_return"]
            )

    change_columns = [
        column
        for column in output.columns
        if any(
            token in column
            for token in (
                "taker_buy_quote_share",
                "signed_taker_flow_ratio",
                "flow_divergence",
                "perp_basis",
                "quote_volume_ratio",
                "trade_count_ratio",
                "volatility_spread",
                "flow_spread",
                "basis_spread",
            )
        )
        and "_delta_" not in column
        and "positive_share" not in column
    ]
    output = _add_change_features(output, change_columns)
    output["feature_available_at_utc"] = (output.index + UTC_DAY).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return output
