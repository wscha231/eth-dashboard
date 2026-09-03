from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from forecasting.lead_signal_data import strict_json_dumps
from forecasting.lead_signals import (
    REQUIRED_STREAM_IDS,
    FoldLocalStandardizer,
    aggregate_hourly_stream,
    build_ethereum_liquidity_features,
    build_lead_signal_daily_features,
    build_market_daily_features,
    feature_group_columns,
)


def _hourly_stream(
    *,
    start: str = "2020-01-01",
    days: int = 12,
    base_price: float = 100.0,
    price_multiplier: float = 1.0,
    taker_share: float = 0.6,
    quote_volume: float = 1_000.0,
) -> pd.DataFrame:
    opened = pd.date_range(start, periods=days * 24, freq="h")
    path = base_price * np.power(1.0002, np.arange(len(opened) + 1))
    opening = path[:-1] * price_multiplier
    closing = path[1:] * price_multiplier
    return pd.DataFrame(
        {
            "open_time": opened,
            "close_time": opened + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1),
            "open": opening,
            "high": np.maximum(opening, closing) * 1.001,
            "low": np.minimum(opening, closing) * 0.999,
            "close": closing,
            "quote_volume": quote_volume,
            "trade_count": 50,
            "taker_buy_quote_volume": quote_volume * taker_share,
        }
    )


def _streams(days: int = 12) -> dict[str, pd.DataFrame]:
    return {
        "binance_spot_ethusdt_1h": _hourly_stream(days=days, taker_share=0.60),
        "binance_um_ethusdt_1h": _hourly_stream(
            days=days, price_multiplier=1.01, taker_share=0.55
        ),
        "binance_spot_btcusdt_1h": _hourly_stream(
            days=days, base_price=1_000.0, taker_share=0.52
        ),
        "binance_um_btcusdt_1h": _hourly_stream(
            days=days,
            base_price=1_000.0,
            price_multiplier=1.005,
            taker_share=0.51,
        ),
    }


def _defillama_sources(days: int = 40) -> tuple[pd.DataFrame, ...]:
    index = pd.date_range("2019-12-01", periods=days, freq="D")
    stable = pd.DataFrame(
        {"defillama_ethereum_stablecoin_total_usd": np.arange(days) + 100.0},
        index=index,
    )
    tvl = pd.DataFrame(
        {"defillama_ethereum_chain_tvl_usd": np.arange(days) + 1_000.0},
        index=index,
    )
    dex = pd.DataFrame(
        {"defillama_ethereum_dex_volume_usd": np.arange(days) + 200.0},
        index=index,
    )
    return stable, tvl, dex


def test_aggregate_hourly_stream_excludes_partial_and_future_days() -> None:
    base = _hourly_stream(days=4)
    cutoff = pd.Timestamp("2020-01-03 12:00:00", tz="UTC")
    observed = aggregate_hourly_stream(base, prefix="eth_spot", cutoff=cutoff)

    mutated = base.copy()
    future = mutated["close_time"] > pd.Timestamp("2020-01-03")
    mutated.loc[future, "quote_volume"] = 1e15
    changed = aggregate_hourly_stream(mutated, prefix="eth_spot", cutoff=cutoff)

    assert observed.index.tolist() == list(pd.date_range("2020-01-01", periods=2))
    pd.testing.assert_frame_equal(observed, changed)


def test_aggregate_hourly_stream_rejects_interior_missing_day() -> None:
    frame = _hourly_stream(days=4)
    frame = frame.loc[frame["open_time"] != pd.Timestamp("2020-01-02 04:00:00")]
    with pytest.raises(ValueError, match="Incomplete interior UTC days"):
        aggregate_hourly_stream(
            frame,
            prefix="eth_spot",
            cutoff=pd.Timestamp("2020-01-05"),
        )


def test_partial_session_requires_declaration_and_nulls_entire_day() -> None:
    streams = _streams(days=4)
    bad_date = pd.Timestamp("2020-01-02")
    for frame in streams.values():
        mask = frame["open_time"] == pd.Timestamp("2020-01-02 12:00:00")
        frame.loc[mask, "close_time"] = pd.Timestamp("2020-01-02 12:30:00")

    with pytest.raises(ValueError, match="Undeclared partial hourly sessions"):
        build_market_daily_features(streams, cutoff=pd.Timestamp("2020-01-05"))

    output = build_market_daily_features(
        streams,
        cutoff=pd.Timestamp("2020-01-05"),
        excluded_dates=(bad_date,),
    )
    assert output.loc[bad_date, "market_data_excluded"] == 1
    assert pd.isna(output.loc[bad_date, "eth_spot_close"])
    assert pd.isna(output.loc[bad_date, "btc_perp_basis"])
    assert output.loc[pd.Timestamp("2020-01-01"), "market_data_excluded"] == 0


def test_market_features_match_flow_basis_and_cross_asset_contract() -> None:
    output = build_market_daily_features(_streams(), cutoff=pd.Timestamp("2020-01-13"))
    row = output.loc[pd.Timestamp("2020-01-12")]

    assert row["eth_spot_taker_buy_quote_share"] == pytest.approx(0.60)
    assert row["eth_spot_signed_taker_flow_ratio"] == pytest.approx(0.20)
    assert row["eth_spot_perp_flow_divergence"] == pytest.approx(-0.10)
    assert row["eth_perp_basis"] == pytest.approx(0.01)
    assert row["btc_perp_basis"] == pytest.approx(0.005)
    assert row["eth_btc_basis_spread"] == pytest.approx(0.005)
    assert row["eth_spot_bar_count"] == 24
    assert row["feature_available_at_utc"] == "2020-01-13T00:00:00Z"
    assert "eth_perp_basis_delta_7d" in output


def test_defillama_values_are_lagged_and_future_mutation_is_causal() -> None:
    stable, tvl, dex = _defillama_sources()
    baseline = build_ethereum_liquidity_features(
        stablecoins=stable,
        tvl=tvl,
        dex_volume=dex,
        as_of_date="2020-01-09",
        reporting_lag_days=1,
    )

    changed_stable = stable.copy()
    changed_stable.loc[changed_stable.index >= "2020-01-05"] *= 1e6
    changed = build_ethereum_liquidity_features(
        stablecoins=changed_stable,
        tvl=tvl,
        dex_volume=dex,
        as_of_date="2020-01-09",
        reporting_lag_days=1,
    )

    assert baseline.loc["2020-01-04", "defillama_source_date"] == "2020-01-03"
    assert (
        baseline.loc["2020-01-04", "defillama_ethereum_stablecoin_total_usd"]
        == stable.loc["2020-01-03"].iloc[0]
    )
    pd.testing.assert_frame_equal(
        baseline.loc[:"2020-01-05"], changed.loc[:"2020-01-05"]
    )


def test_combined_feature_groups_are_disjoint_and_complete() -> None:
    stable, tvl, dex = _defillama_sources()
    output = build_lead_signal_daily_features(
        streams=_streams(),
        stablecoins=stable,
        tvl=tvl,
        dex_volume=dex,
        as_of_date="2020-01-12",
    )
    groups = feature_group_columns(output)
    flattened = [column for columns in groups.values() for column in columns]

    assert set(groups) == {
        "order_flow",
        "leverage_basis",
        "intraday_risk",
        "cross_asset_leadership",
        "ethereum_liquidity",
    }
    assert len(flattened) == len(set(flattened))
    assert "defillama_ethereum_chain_tvl_usd" in groups["ethereum_liquidity"]
    assert "btc_spot_signed_taker_flow_ratio" in groups["cross_asset_leadership"]
    assert "eth_spot_signed_taker_flow_ratio" in groups["order_flow"]
    assert output.index.max() == pd.Timestamp("2020-01-12")
    assert set(REQUIRED_STREAM_IDS) == set(_streams())


def test_fold_local_standardizer_never_fits_on_test_rows() -> None:
    index = pd.date_range("2020-01-01", periods=45, freq="D")
    frame = pd.DataFrame(
        {
            "a": np.arange(45, dtype=float),
            "b": np.arange(45, dtype=float) * 2,
            "constant": 1.0,
            "text": "x",
        },
        index=index,
    )
    training = frame.iloc[:40]
    test = frame.iloc[40:]
    scaler = FoldLocalStandardizer(minimum_rows=30).fit(training)
    original_state = copy.deepcopy(scaler.state())
    original_training = scaler.transform(training)

    mutated_test = test.copy()
    mutated_test.loc[:, ["a", "b"]] = 1e12
    transformed_test = scaler.transform(mutated_test)

    assert scaler.state() == original_state
    pd.testing.assert_frame_equal(original_training, scaler.transform(training))
    assert scaler.selected_columns_ == ("a", "b")
    assert transformed_test.min().min() > 1e6
    assert "NaN" not in strict_json_dumps(scaler.state())
