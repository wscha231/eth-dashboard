from __future__ import annotations

import numpy as np
import pandas as pd

from eth_data_collector import merge_history_frame, seed_market_cache_from_master
import eth_price_forecast as efp
from eth_price_forecast import load_market_data_csv
from forecast_site.smoke_test import select_resolvable_input_timestamp


def test_merge_history_fills_old_holes_but_only_overwrites_refresh_window() -> None:
    index = pd.date_range("2026-01-01", periods=5, freq="D")
    existing = pd.DataFrame(
        {
            "eth_close": [10.0, np.nan, np.nan, 40.0, 50.0],
            "vendor_signal": [1.0, 2.0, np.nan, 4.0, 5.0],
        },
        index=index,
    )
    incoming = pd.DataFrame(
        {
            "eth_close": [11.0, 20.0, 30.0, 44.0, 55.0],
            "vendor_signal": [10.0, 20.0, 30.0, 40.0, 50.0],
        },
        index=index,
    )

    merged = merge_history_frame(
        existing,
        incoming,
        overwrite_start=pd.Timestamp("2026-01-04"),
    )

    assert merged["eth_close"].tolist() == [10.0, 20.0, 30.0, 44.0, 55.0]
    assert merged["vendor_signal"].tolist() == [1.0, 2.0, 30.0, 40.0, 50.0]


def test_merge_history_keeps_new_rows_before_refresh_window() -> None:
    existing = pd.DataFrame(
        {"eth_close": [10.0]},
        index=pd.to_datetime(["2026-01-01"]),
    )
    incoming = pd.DataFrame(
        {"eth_close": [11.0, 20.0, 30.0, 40.0]},
        index=pd.date_range("2026-01-01", periods=4, freq="D"),
    )

    merged = merge_history_frame(
        existing,
        incoming,
        overwrite_start=pd.Timestamp("2026-01-04"),
    )

    assert merged["eth_close"].tolist() == [10.0, 20.0, 30.0, 40.0]


def test_seed_market_cache_uses_non_null_master_market_history(tmp_path) -> None:
    index = pd.date_range("2026-01-01", periods=3, freq="D")
    master = pd.DataFrame(
        {
            "eth_close": [10.0, np.nan, 12.0],
            "eth_open": [9.0, np.nan, 11.0],
            "btc_close": [100.0, np.nan, 120.0],
            "fred_sofr": [4.0, 4.0, 4.0],
        },
        index=index,
    )
    cache = tmp_path / "market.csv"

    assert seed_market_cache_from_master(cache, master) is True
    seeded = load_market_data_csv(cache)
    assert seeded.index.tolist() == [index[0], index[2]]
    assert set(seeded.columns) == {"eth_close", "eth_open", "btc_close"}
    assert seed_market_cache_from_master(cache, master) is False


def test_market_cache_refresh_recovers_the_oldest_eth_gap(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "market.csv"
    cached_index = pd.to_datetime(
        ["2026-01-01", "2026-01-02", "2026-01-05", "2026-01-06"]
    )
    cached = pd.DataFrame(
        {
            "eth_open": [10.0, 11.0, 14.0, 15.0],
            "eth_high": [11.0, 12.0, 15.0, 16.0],
            "eth_low": [9.0, 10.0, 13.0, 14.0],
            "eth_close": [10.5, 11.5, 14.5, 15.5],
            "eth_volume": [100.0, 110.0, 140.0, 150.0],
        },
        index=cached_index,
    )
    efp.save_market_data_csv(cached, cache)
    starts: list[pd.Timestamp] = []

    def fake_download_symbol_history(
        symbol: str,
        alias: str,
        interval: str,
        period: str | None = None,
        start: str | pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        del symbol, interval, period
        starts.append(pd.Timestamp(start))
        index = pd.date_range("2026-01-02", "2026-01-07", freq="D")
        values = pd.Series(range(len(index)), index=index, dtype=float) + 20.0
        return pd.DataFrame(
            {
                f"{alias}_open": values,
                f"{alias}_high": values + 1.0,
                f"{alias}_low": values - 1.0,
                f"{alias}_close": values + 0.5,
                f"{alias}_volume": 200.0,
            },
            index=index,
        )

    monkeypatch.setattr(efp, "download_symbol_history", fake_download_symbol_history)

    refreshed, metadata = efp.update_market_data_cache(
        {"eth": "ETH-USD"},
        dataset_csv=cache,
        period="10y",
        interval="1d",
        lookback_rows=1,
        verbose=False,
        return_metadata=True,
    )

    assert starts == [pd.Timestamp("2026-01-02")]
    assert metadata["gap_refresh_start"] == "2026-01-02 00:00:00"
    assert metadata["refresh_start"] == "2026-01-02 00:00:00"
    assert refreshed.index.equals(pd.date_range("2026-01-01", "2026-01-07", freq="D"))
    assert refreshed.loc["2026-01-03":"2026-01-04", "eth_close"].notna().all()
    assert efp.find_missing_daily_eth_dates(refreshed).empty


def test_complete_market_cache_keeps_the_bounded_recent_refresh(monkeypatch, tmp_path) -> None:
    cache = tmp_path / "market.csv"
    index = pd.date_range("2026-01-01", "2026-01-06", freq="D")
    values = pd.Series(range(len(index)), index=index, dtype=float) + 10.0
    cached = pd.DataFrame(
        {
            "eth_open": values,
            "eth_high": values + 1.0,
            "eth_low": values - 1.0,
            "eth_close": values + 0.5,
            "eth_volume": 100.0,
        },
        index=index,
    )
    efp.save_market_data_csv(cached, cache)
    starts: list[pd.Timestamp] = []

    def fake_download_symbol_history(*args, start=None, **kwargs) -> pd.DataFrame:
        del args, kwargs
        starts.append(pd.Timestamp(start))
        return cached.loc[cached.index >= pd.Timestamp(start)]

    monkeypatch.setattr(efp, "download_symbol_history", fake_download_symbol_history)

    _, metadata = efp.update_market_data_cache(
        {"eth": "ETH-USD"},
        dataset_csv=cache,
        period="10y",
        interval="1d",
        lookback_rows=1,
        verbose=False,
        return_metadata=True,
    )

    assert starts == [pd.Timestamp("2026-01-05")]
    assert metadata["gap_refresh_start"] == ""
    assert metadata["refresh_start"] == "2026-01-05 00:00:00"


def test_smoke_selects_latest_input_with_both_resolved_targets() -> None:
    dates = pd.to_datetime(
        [
            "2026-07-27",
            "2026-07-28",
            "2026-08-03",
            "2026-08-26",
        ],
        utc=True,
    )
    close_lookup = pd.Series([100.0, 101.0, 102.0, 103.0], index=dates)

    selected = select_resolvable_input_timestamp(
        close_lookup,
        pd.Timestamp("2026-08-27", tz="UTC"),
    )

    assert selected == pd.Timestamp("2026-07-27", tz="UTC")
