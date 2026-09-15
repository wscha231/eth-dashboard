import pandas as pd
from data_lab.free_source_fallbacks import (
    collect_bitget_basis,
    collect_bitget_funding,
    collect_bitget_ticker_snapshot,
    collect_fred_initial_partial,
)


def test_bitget_funding_parses_history():
    def get_json(url, params=None, **kwargs):
        return {"code": "00000", "data": [
            {"fundingRate": "0.001", "fundingTime": "1704153600000"},
            {"fundingRate": "-0.001", "fundingTime": "1704124800000"},
        ]}
    frame = collect_bitget_funding("2024-01-01", "2024-01-03", get_json, max_pages=1)
    assert not frame.empty
    assert "bitget_eth_funding_rate_mean" in frame


def test_bitget_basis_uses_mark_over_index():
    def get_json(url, params=None, **kwargs):
        price = 100 if "index" in url else 101
        return {"code": "00000", "data": [["1704067200000", str(price), str(price), str(price), str(price), "0", "0"]]}
    frame = collect_bitget_basis("2024-01-01", "2024-01-01", get_json)
    assert abs(frame.iloc[0]["bitget_eth_mark_index_basis"] - 0.01) < 1e-12


def test_bitget_ticker_snapshot_preserves_oi_unit():
    def get_json(url, params=None, **kwargs):
        return {"code": "00000", "data": [{
            "symbol": "ETHUSDT", "openInterest": "12", "fundingRate": "0.0001",
            "markPrice": "2000", "indexPrice": "1990", "volume24h": "3", "turnover24h": "6000",
        }]}
    frame = collect_bitget_ticker_snapshot("2024-01-01", get_json)
    assert frame.iloc[0]["bitget_eth_open_interest_eth_snapshot"] == 12


def test_fred_initial_failure_is_per_series_not_global():
    class ExampleError(Exception):
        code = 400
    def get_json(url, params=None, **kwargs):
        if params["series_id"] == "M2SL":
            return {"observations": [{"date": "2024-01-01", "realtime_start": "2024-02-01", "value": "10"}]}
        raise ExampleError()
    frame, errors = collect_fred_initial_partial("key", {"M2SL": "m2", "BAD": "bad"}, get_json)
    assert len(frame) == 1
    assert errors["BAD"] == 400
