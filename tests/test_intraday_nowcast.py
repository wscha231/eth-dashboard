from __future__ import annotations

import numpy as np
import pandas as pd

from forecasting.intraday_nowcast import (
    CANDIDATES,
    PRICE_IMPULSE,
    build_nowcast_table,
    positive_events,
    score_candidate,
    select_alert_origins,
)


def _hourly_streams(periods: int = 3000) -> dict[str, pd.DataFrame]:
    open_time = pd.date_range("2025-01-01", periods=periods, freq="h")
    position = np.arange(periods, dtype=float)
    eth_close = 2000.0 * np.exp(np.cumsum(0.00005 + 0.0003 * np.sin(position / 17.0)))
    btc_close = 40000.0 * np.exp(np.cumsum(0.00004 + 0.0002 * np.sin(position / 19.0)))

    def frame(close: np.ndarray, *, basis: float = 0.0) -> pd.DataFrame:
        adjusted = close * (1.0 + basis + 0.0001 * np.sin(position / 13.0))
        quote = 1_000_000.0 * (1.0 + 0.2 * np.cos(position / 11.0))
        share = 0.50 + 0.04 * np.sin(position / 7.0)
        return pd.DataFrame(
            {
                "open_time": open_time,
                "close_time": open_time
                + pd.Timedelta(hours=1)
                - pd.Timedelta(milliseconds=1),
                "close": adjusted,
                "quote_volume": quote,
                "taker_buy_quote_volume": quote * share,
            }
        )

    return {
        "binance_spot_ethusdt_1h": frame(eth_close),
        "binance_um_ethusdt_1h": frame(eth_close, basis=0.001),
        "binance_spot_btcusdt_1h": frame(btc_close),
        "binance_um_btcusdt_1h": frame(btc_close, basis=0.0008),
    }


def test_nowcast_table_has_matched_causal_candidates() -> None:
    table = build_nowcast_table(_hourly_streams())

    assert len(table) > 700
    assert table.index.is_unique
    assert table.index.is_monotonic_increasing
    assert table["target_up_8pct_48h"].notna().all()
    for candidate in CANDIDATES:
        assert table[candidate].dtype == bool


def test_current_impulse_cannot_raise_its_own_prior_threshold() -> None:
    streams = _hourly_streams()
    baseline = build_nowcast_table(streams)
    decision_time = baseline.index[100]
    open_time = decision_time - pd.Timedelta(hours=1)
    row = streams["binance_spot_ethusdt_1h"]["open_time"].eq(open_time)
    streams["binance_spot_ethusdt_1h"].loc[row, "close"] *= 1.20

    changed = build_nowcast_table(streams)

    assert (
        changed.loc[decision_time, "prior_q99_eth_return_4h"]
        == baseline.loc[decision_time, "prior_q99_eth_return_4h"]
    )
    assert bool(changed.loc[decision_time, PRICE_IMPULSE])


def test_future_market_changes_do_not_change_current_features_or_thresholds() -> None:
    streams = _hourly_streams()
    baseline = build_nowcast_table(streams)
    decision_time = baseline.index[200]
    future_open = decision_time + pd.Timedelta(hours=60)
    for frame in streams.values():
        mask = frame["open_time"] >= future_open
        frame.loc[mask, "close"] *= 1.5
        frame.loc[mask, "quote_volume"] *= 3.0
        frame.loc[mask, "taker_buy_quote_volume"] *= 3.0

    changed = build_nowcast_table(streams)
    causal_columns = [
        "eth_return_4h",
        "eth_quote_volume_4h",
        "eth_taker_buy_share_4h",
        "eth_btc_basis_spread_change_4h",
        "prior_q98_eth_return_4h",
        "prior_q99_eth_return_4h",
        "prior_q90_eth_quote_volume_4h",
        "prior_q75_eth_btc_relative_return_4h",
        "prior_q60_eth_btc_basis_spread_change_4h",
        *CANDIDATES,
    ]

    pd.testing.assert_series_equal(
        changed.loc[decision_time, causal_columns],
        baseline.loc[decision_time, causal_columns],
    )


def test_sustained_alert_is_counted_again_every_24_hours() -> None:
    index = pd.date_range("2026-01-01", periods=72, freq="h")
    alerts = pd.Series(True, index=index)

    selected = select_alert_origins(alerts)

    assert selected.tolist() == [index[0], index[24], index[48]]


def test_positive_events_only_merge_gaps_up_to_24_hours() -> None:
    index = pd.date_range("2026-01-01", periods=80, freq="h")
    labels = pd.Series(False, index=index)
    labels.loc[[index[0], index[1], index[20], index[50]]] = True

    events = positive_events(labels)

    assert len(events) == 2
    assert events[0].start == index[0]
    assert events[0].end == index[20]
    assert events[1].start == index[50]


def test_candidate_scoring_separates_true_and_false_refreshed_alerts() -> None:
    index = pd.date_range("2026-01-01", periods=24 * 100, freq="h")
    table = pd.DataFrame(index=index)
    table["target_up_8pct_48h"] = False
    table["hours_to_target"] = np.nan
    table["future_max_return_48h"] = 0.0
    table.loc[index[100:121], "target_up_8pct_48h"] = True
    table.loc[index[100:121], "hours_to_target"] = 12.0
    table.loc[index[100:121], "future_max_return_48h"] = 0.10
    for candidate in CANDIDATES:
        table[candidate] = False
    table.loc[[index[105], index[1000]], PRICE_IMPULSE] = True

    result = score_candidate(table, PRICE_IMPULSE)

    assert result["event_count"] == 1
    assert result["detected_event_count"] == 1
    assert result["true_alert_episode_count"] == 1
    assert result["false_alert_episode_count"] == 1
