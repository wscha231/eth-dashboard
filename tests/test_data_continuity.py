from __future__ import annotations

import numpy as np
import pandas as pd

from eth_data_collector import merge_history_frame, seed_market_cache_from_master
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
