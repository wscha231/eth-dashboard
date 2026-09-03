from __future__ import annotations

import math
from dataclasses import asdict, dataclass
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
    undeclared = invalid_dates.difference(excluded_dates)
    if len(undeclared):
        sample = ", ".join(value.date().isoformat() for value in undeclared[:5])
        raise ValueError(f"Undeclared partial hourly sessions on UTC days: {sample}")
    return output


def _complete_utc_days(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DatetimeIndex:
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
    complete_days = _complete_utc_days(hourly, cutoff_timestamp)
    if complete_days.empty:
        raise ValueError("No complete UTC days are available at the declared cutoff")

    previous_close = hourly["close"].shift(1)
    previous_close.iloc[0] = hourly["open"].iloc[0]
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


def _normalize_daily_source(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], utc=True, errors="coerce")
        output["date"] = output["date"].dt.tz_convert(None).dt.floor("D")
        output = output.set_index("date")
    else:
        output.index = pd.to_datetime(output.index, utc=True, errors="coerce")
        output.index = output.index.tz_convert(None).floor("D")
        output.index.name = "date"
    if output.index.isna().any():
        raise ValueError("Daily source contains invalid dates")
    if output.index.duplicated().any():
        raise ValueError("Daily source contains duplicate dates")
    return output.sort_index()


def _causal_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    minimum = max(10, window // 3)
    mean = series.rolling(window, min_periods=minimum).mean()
    scale = series.rolling(window, min_periods=minimum).std(ddof=0)
    return ((series - mean) / scale.where(scale > 1e-15)).replace(
        [np.inf, -np.inf], np.nan
    )


def build_ethereum_liquidity_features(
    *,
    stablecoins: pd.DataFrame,
    tvl: pd.DataFrame,
    dex_volume: pd.DataFrame,
    as_of_date: Any,
    market_daily: pd.DataFrame | None = None,
    reporting_lag_days: int = 1,
) -> pd.DataFrame:
    if reporting_lag_days < 1:
        raise ValueError("DefiLlama reporting_lag_days must be at least one")
    as_of = _naive_utc(as_of_date).floor("D")

    sources = [
        _normalize_daily_source(stablecoins),
        _normalize_daily_source(tvl),
        _normalize_daily_source(dex_volume),
    ]
    shifted: list[pd.DataFrame] = []
    for source in sources:
        item = source.copy()
        item.index = item.index + pd.Timedelta(days=reporting_lag_days)
        shifted.append(item)
    output = pd.concat(shifted, axis=1, join="outer").sort_index()
    output = output.loc[output.index <= as_of]
    output["defillama_source_date"] = (
        output.index - pd.Timedelta(days=reporting_lag_days)
    ).strftime("%Y-%m-%d")

    stable_column = "defillama_ethereum_stablecoin_total_usd"
    if stable_column not in output:
        stable_column = "defillama_ethereum_stablecoin_pegged_usd"
    tvl_column = "defillama_ethereum_chain_tvl_usd"
    dex_column = "defillama_ethereum_dex_volume_usd"
    required = {stable_column, tvl_column, dex_column}
    missing = sorted(required - set(output.columns))
    if missing:
        raise ValueError(f"Missing DefiLlama daily columns: {missing}")

    stable_values = pd.to_numeric(output[stable_column], errors="coerce")
    tvl_values = pd.to_numeric(output[tvl_column], errors="coerce")
    dex_values = pd.to_numeric(output[dex_column], errors="coerce")
    for window in (7, 30, 90):
        output[f"eth_stablecoin_supply_growth_{window}d"] = stable_values.pct_change(
            window, fill_method=None
        )
        output[f"eth_tvl_growth_{window}d"] = tvl_values.pct_change(
            window, fill_method=None
        )
    for window in (7, 30):
        growth = output[f"eth_stablecoin_supply_growth_{window}d"]
        output[f"eth_stablecoin_supply_acceleration_{window}d"] = growth.diff(window)
    output["eth_dex_volume_to_tvl"] = _safe_divide(dex_values, tvl_values)
    log_dex = np.log(dex_values.where(dex_values > 0))
    output["eth_dex_volume_zscore_30d"] = _causal_rolling_zscore(log_dex, 30)
    output["eth_dex_volume_zscore_90d"] = _causal_rolling_zscore(log_dex, 90)

    if market_daily is not None:
        market = _normalize_daily_source(market_daily)
        market = market.loc[market.index <= as_of]
        for column in (
            "cg_eth_market_cap_usd",
            "cg_global_total_market_cap_usd",
        ):
            if column in market:
                output[column] = pd.to_numeric(market[column], errors="coerce")
        if "cg_eth_market_cap_usd" in output:
            output["eth_tvl_to_market_cap"] = _safe_divide(
                tvl_values, output["cg_eth_market_cap_usd"]
            )
            output["eth_stablecoin_to_market_cap"] = _safe_divide(
                stable_values, output["cg_eth_market_cap_usd"]
            )
        if "cg_global_total_market_cap_usd" in output:
            output["eth_stablecoin_to_global_market_cap"] = _safe_divide(
                stable_values, output["cg_global_total_market_cap_usd"]
            )

    numeric_columns = output.select_dtypes(include=[np.number]).columns
    output.loc[:, numeric_columns] = output.loc[:, numeric_columns].replace(
        [np.inf, -np.inf], np.nan
    )
    output["feature_available_at_utc"] = (output.index + UTC_DAY).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return output


def build_lead_signal_daily_features(
    *,
    streams: dict[str, pd.DataFrame],
    stablecoins: pd.DataFrame,
    tvl: pd.DataFrame,
    dex_volume: pd.DataFrame,
    as_of_date: Any,
    market_daily: pd.DataFrame | None = None,
    reporting_lag_days: int = 1,
    excluded_market_dates: tuple[Any, ...] = (),
) -> pd.DataFrame:
    as_of = _naive_utc(as_of_date).floor("D")
    market_features = build_market_daily_features(
        streams,
        cutoff=as_of + UTC_DAY,
        excluded_dates=excluded_market_dates,
    ).drop(columns="feature_available_at_utc")
    liquidity = build_ethereum_liquidity_features(
        stablecoins=stablecoins,
        tvl=tvl,
        dex_volume=dex_volume,
        as_of_date=as_of,
        market_daily=market_daily,
        reporting_lag_days=reporting_lag_days,
    ).drop(columns="feature_available_at_utc")
    output = market_features.join(liquidity, how="outer").sort_index()
    output = output.loc[output.index <= as_of]
    output["feature_available_at_utc"] = (output.index + UTC_DAY).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    output.index.name = "date"
    return output


def feature_group_columns(frame: pd.DataFrame) -> dict[str, list[str]]:
    numeric = set(frame.select_dtypes(include=[np.number]).columns)
    diagnostics = {
        column
        for column in numeric
        if column.endswith("_bar_count")
        or column
        in {
            "cg_eth_market_cap_usd",
            "cg_global_total_market_cap_usd",
            "market_data_excluded",
        }
    }
    usable = numeric - diagnostics

    liquidity = {column for column in usable if column.startswith("defillama_")}
    liquidity |= {column for column in usable if column.startswith("eth_stablecoin_")}
    liquidity |= {column for column in usable if column.startswith("eth_tvl_")}
    liquidity |= {column for column in usable if column.startswith("eth_dex_")}

    cross_asset = {
        column
        for column in usable - liquidity
        if column.startswith(("btc_", "eth_btc_"))
    }
    leverage = {
        column
        for column in usable - liquidity - cross_asset
        if any(
            token in column
            for token in (
                "perp_basis",
                "futures_spot_quote_volume_ratio",
                "futures_spot_trade_count_ratio",
            )
        )
    }
    intraday = {
        column
        for column in usable - liquidity - cross_asset - leverage
        if any(
            token in column
            for token in (
                "realized_volatility",
                "semivariance",
                "jump_variation",
                "last_4h_",
                "last_8h_",
                "last_12h_",
            )
        )
    }
    order_flow = usable - liquidity - cross_asset - leverage - intraday
    return {
        "order_flow": sorted(order_flow),
        "leverage_basis": sorted(leverage),
        "intraday_risk": sorted(intraday),
        "cross_asset_leadership": sorted(cross_asset),
        "ethereum_liquidity": sorted(liquidity),
    }


@dataclass
class FoldLocalStandardizer:
    min_coverage: float = 0.8
    minimum_rows: int = 30
    selected_columns_: tuple[str, ...] = ()
    medians_: dict[str, float] | None = None
    means_: dict[str, float] | None = None
    scales_: dict[str, float] | None = None
    training_start_: str | None = None
    training_end_: str | None = None

    def fit(self, training: pd.DataFrame) -> FoldLocalStandardizer:
        if len(training) < self.minimum_rows:
            raise ValueError(
                f"Fold training requires at least {self.minimum_rows} rows"
            )
        if not 0 < self.min_coverage <= 1:
            raise ValueError("min_coverage must be in (0, 1]")
        numeric = training.select_dtypes(include=[np.number]).replace(
            [np.inf, -np.inf], np.nan
        )
        coverage = numeric.notna().mean()
        eligible = sorted(coverage.index[coverage >= self.min_coverage])
        medians: dict[str, float] = {}
        means: dict[str, float] = {}
        scales: dict[str, float] = {}
        selected: list[str] = []
        for column in eligible:
            values = numeric[column]
            median = float(values.median())
            filled = values.fillna(median)
            mean = float(filled.mean())
            scale = float(filled.std(ddof=0))
            if not all(math.isfinite(value) for value in (median, mean, scale)):
                continue
            if scale <= 1e-12:
                continue
            selected.append(column)
            medians[column] = median
            means[column] = mean
            scales[column] = scale
        if not selected:
            raise ValueError("No non-constant fold-local features passed coverage")

        index = pd.DatetimeIndex(pd.to_datetime(training.index))
        self.selected_columns_ = tuple(selected)
        self.medians_ = medians
        self.means_ = means
        self.scales_ = scales
        self.training_start_ = pd.Timestamp(index.min()).date().isoformat()
        self.training_end_ = pd.Timestamp(index.max()).date().isoformat()
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        if not self.selected_columns_ or self.medians_ is None:
            raise RuntimeError("FoldLocalStandardizer must be fitted before transform")
        missing = sorted(set(self.selected_columns_) - set(frame.columns))
        if missing:
            raise ValueError(f"Transform frame is missing fitted columns: {missing}")
        output = pd.DataFrame(index=frame.index)
        for column in self.selected_columns_:
            values = pd.to_numeric(frame[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            filled = values.fillna(self.medians_[column])
            output[column] = (filled - self.means_[column]) / self.scales_[column]
        return output

    def fit_transform(self, training: pd.DataFrame) -> pd.DataFrame:
        return self.fit(training).transform(training)

    def state(self) -> dict[str, Any]:
        if not self.selected_columns_ or self.medians_ is None:
            raise RuntimeError("FoldLocalStandardizer has not been fitted")
        return asdict(self)
