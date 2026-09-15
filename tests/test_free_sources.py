import numpy as np
import pandas as pd
import pytest

from data_lab.free_sources import (
    availability_sidecar,
    build_bybit_feature_frame,
    build_fred_initial_asof_daily,
    collect_bybit_funding_history,
    collect_bybit_open_interest_history,
    collect_bybit_premium_index_history,
    collect_fred_initial_release_long,
)


def ms(text):
    return str(int(pd.Timestamp(text, tz="UTC").timestamp() * 1000))


def test_bybit_funding_paginates_backwards_and_aggregates():
    calls = []
    pages = [
        {
            "retCode": 0,
            "result": {
                "list": [
                    {"fundingRateTimestamp": ms("2026-01-03 16:00"), "fundingRate": "0.0003"},
                    {"fundingRateTimestamp": ms("2026-01-03 08:00"), "fundingRate": "0.0001"},
                    {"fundingRateTimestamp": ms("2026-01-03 00:00"), "fundingRate": "0.0002"},
                ]
            },
        }
    ]

    def fake(url, params=None):
        calls.append((url, params))
        return pages.pop(0)

    out = collect_bybit_funding_history("2026-01-01", "2026-01-04", get_json=fake)
    assert len(calls) == 1
    assert out.loc[pd.Timestamp("2026-01-03", tz="UTC"), "bybit_eth_funding_rate_mean"] == pytest.approx(0.0002)
    assert out.loc[pd.Timestamp("2026-01-03", tz="UTC"), "bybit_eth_funding_settlements"] == 3


def test_bybit_open_interest_follows_cursor_and_preserves_eth_units():
    responses = [
        {
            "retCode": 0,
            "result": {
                "list": [
                    {"timestamp": ms("2026-01-02"), "openInterest": "100", "singleOpenInterest": "50"},
                ],
                "nextPageCursor": "abc",
            },
        },
        {
            "retCode": 0,
            "result": {
                "list": [
                    {"timestamp": ms("2026-01-01"), "openInterest": "90", "singleOpenInterest": "45"},
                ],
                "nextPageCursor": "",
            },
        },
    ]

    def fake(url, params=None):
        return responses.pop(0)

    out = collect_bybit_open_interest_history(
        "2026-01-01", "2026-01-02", get_json=fake, chunk_days=180
    )
    assert out.loc[pd.Timestamp("2026-01-01", tz="UTC"), "bybit_eth_open_interest_eth_last"] == 90
    assert out.loc[pd.Timestamp("2026-01-02", tz="UTC"), "bybit_eth_single_open_interest_eth_last"] == 50


def test_bybit_oi_cursor_loop_fails_closed():
    def fake(url, params=None):
        return {
            "retCode": 0,
            "result": {
                "list": [{"timestamp": ms("2026-01-01"), "openInterest": "90"}],
                "nextPageCursor": "same",
            },
        }

    with pytest.raises(ValueError, match="cursor loop"):
        collect_bybit_open_interest_history("2026-01-01", "2026-01-02", get_json=fake)


def test_premium_index_kline_and_feature_frame_are_past_only():
    def fake(url, params=None):
        return {
            "retCode": 0,
            "result": {
                "list": [
                    [ms("2026-01-02"), "0.001", "0.003", "0.000", "0.002"],
                    [ms("2026-01-01"), "0.000", "0.002", "-0.001", "0.001"],
                ]
            },
        }

    premium = collect_bybit_premium_index_history("2026-01-01", "2026-01-02", get_json=fake)
    assert premium.loc[pd.Timestamp("2026-01-02", tz="UTC"), "bybit_eth_premium_index_close"] == pytest.approx(0.002)
    combined = build_bybit_feature_frame(pd.DataFrame(), pd.DataFrame(), premium)
    assert combined.index.tz is not None
    # rolling features need history and therefore stay NaN rather than using future rows.
    assert np.isnan(combined.iloc[0]["bybit_eth_premium_index_mean_7d"])


def test_availability_is_next_utc_day():
    idx = pd.DatetimeIndex(["2026-01-01"], tz="UTC")
    frame = pd.DataFrame({"x": [1.0]}, index=idx)
    sidecar = availability_sidecar(frame, "src")
    assert sidecar.loc[0, "available_at"] == "2026-01-02T00:00:00+00:00"


def test_fred_initial_release_uses_output_type_four_and_preserves_release_date():
    seen = []

    def fake(url, params=None):
        seen.append(params)
        return {
            "observations": [
                {
                    "date": "2025-12-01",
                    "realtime_start": "2026-01-15",
                    "realtime_end": "9999-12-31",
                    "value": "100.0",
                }
            ]
        }

    long = collect_fred_initial_release_long(
        "x" * 32, series={"M2SL": "alfred_m2_initial"}, get_json=fake
    )
    assert seen[0]["output_type"] == 4
    assert long.loc[0, "initial_release_date"] == "2026-01-15"
    assert long.loc[0, "observation_date"] == "2025-12-01"


def test_fred_asof_waits_until_day_after_release():
    long = pd.DataFrame(
        [
            {
                "series_id": "M2SL",
                "alias": "alfred_m2_initial",
                "observation_date": "2025-12-01",
                "initial_release_date": "2026-01-15",
                "value": 100.0,
            },
            {
                "series_id": "M2SL",
                "alias": "alfred_m2_initial",
                "observation_date": "2026-01-01",
                "initial_release_date": "2026-02-12",
                "value": 102.0,
            },
        ]
    )
    out = build_fred_initial_asof_daily(long, "2026-01-14", "2026-02-14")
    assert pd.isna(out.loc[pd.Timestamp("2026-01-15", tz="UTC"), "alfred_m2_initial"])
    assert out.loc[pd.Timestamp("2026-01-16", tz="UTC"), "alfred_m2_initial"] == 100.0
    assert out.loc[pd.Timestamp("2026-02-13", tz="UTC"), "alfred_m2_initial"] == 102.0


def test_nonzero_bybit_retcode_rejected():
    def fake(url, params=None):
        return {"retCode": 10001, "retMsg": "bad", "result": {}}

    with pytest.raises(ValueError, match="retCode"):
        collect_bybit_funding_history("2026-01-01", "2026-01-02", get_json=fake)
