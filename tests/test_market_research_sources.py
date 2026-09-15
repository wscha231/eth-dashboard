import pandas as pd

from data_lab.market_research_sources import (
    build_hyperliquid_features,
    collect_deribit_eth_dvol,
    collect_hyperliquid_eth_funding,
    collect_hyperliquid_eth_snapshot,
)


def test_deribit_dvol_parses_daily_candle():
    def get_json(url, params=None, **kwargs):
        return {"result": {"data": [[1704067200000, 50, 55, 45, 52]], "continuation": None}}

    frame = collect_deribit_eth_dvol("2024-01-01", "2024-01-02", get_json=get_json)
    assert len(frame) == 1
    assert frame.iloc[0]["deribit_eth_dvol_close"] == 52


def test_hyperliquid_funding_paginates_and_aggregates():
    calls = []

    def post_json(url, payload, **kwargs):
        calls.append(payload["startTime"])
        return [
            {"coin": "ETH", "fundingRate": "0.001", "premium": "0.002", "time": 1704067200000},
            {"coin": "ETH", "fundingRate": "-0.001", "premium": "0.001", "time": 1704070800000},
        ]

    frame = collect_hyperliquid_eth_funding(
        "2024-01-01", "2024-01-02", post_json=post_json, max_pages=1
    )
    assert len(frame) == 1
    assert abs(frame.iloc[0]["hyperliquid_eth_funding_rate_mean"]) < 1e-12
    assert frame.iloc[0]["hyperliquid_eth_funding_observations"] == 2


def test_hyperliquid_snapshot_maps_eth_context_by_universe():
    def post_json(url, payload, **kwargs):
        return [
            {"universe": [{"name": "BTC"}, {"name": "ETH"}]},
            [
                {"openInterest": "1", "funding": "0", "premium": "0", "markPx": "100", "oraclePx": "100", "midPx": "100", "dayNtlVlm": "1000"},
                {"openInterest": "12.5", "funding": "0.0001", "premium": "0.0002", "markPx": "2000", "oraclePx": "1998", "midPx": "1999", "dayNtlVlm": "5000000"},
            ],
        ]

    frame = collect_hyperliquid_eth_snapshot("2024-01-01", post_json=post_json)
    assert frame.iloc[0]["hyperliquid_eth_open_interest_native_snapshot"] == 12.5
    assert frame.iloc[0]["hyperliquid_eth_mark_price"] == 2000


def test_hyperliquid_feature_builder_keeps_historical_funding_and_snapshot():
    idx = pd.DatetimeIndex(["2024-01-01", "2024-01-02"], tz="UTC")
    funding = pd.DataFrame({
        "hyperliquid_eth_funding_rate_mean": [0.001, 0.002],
        "hyperliquid_eth_premium_mean": [0.001, 0.002],
    }, index=idx)
    snapshot = pd.DataFrame({
        "hyperliquid_eth_open_interest_native_snapshot": [10.0]
    }, index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"))
    out = build_hyperliquid_features(funding, snapshot)
    assert len(out) == 2
    assert out.loc[pd.Timestamp("2024-01-02", tz="UTC"), "hyperliquid_eth_open_interest_native_snapshot"] == 10.0
