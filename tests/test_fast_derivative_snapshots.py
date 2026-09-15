from pathlib import Path
from unittest.mock import patch

import pandas as pd

from scripts.collect_fast_derivative_snapshots import _receipt_hour, _stamp_snapshot, collect


def test_receipt_hour_is_utc_hour_floor():
    stamp = _receipt_hour(pd.Timestamp("2026-09-16T12:34:56+09:00"))
    assert stamp == pd.Timestamp("2026-09-16T03:00:00Z")


def test_stamp_snapshot_replaces_calendar_day_with_receipt_hour():
    original = pd.DataFrame({"oi": [10]}, index=pd.DatetimeIndex([pd.Timestamp("2026-09-16T00:00:00Z")], name="date"))
    receipt = pd.Timestamp("2026-09-16T04:00:00Z")
    stamped = _stamp_snapshot(original, receipt)
    assert stamped.index[0] == receipt
    assert stamped.iloc[0]["oi"] == 10


def test_collect_accumulates_prospective_context(tmp_path):
    receipt = pd.Timestamp("2026-09-16T04:00:00Z")
    bitget = pd.DataFrame({"bitget_eth_open_interest_eth_snapshot": [12.0]}, index=pd.DatetimeIndex([receipt.floor("D")], name="date"))
    hyper = pd.DataFrame({"hyperliquid_eth_open_interest_native_snapshot": [34.0]}, index=pd.DatetimeIndex([receipt.floor("D")], name="date"))
    with patch("scripts.collect_fast_derivative_snapshots.collect_bitget_ticker_snapshot", return_value=bitget), patch(
        "scripts.collect_fast_derivative_snapshots.collect_hyperliquid_eth_snapshot", return_value=hyper
    ):
        report = collect(tmp_path, receipt)
    assert report["sources"]["bitget_context"]["status"] == "ok"
    assert report["sources"]["hyperliquid_context"]["status"] == "ok"
    saved = pd.read_csv(tmp_path / "bitget_eth_context_4h.csv")
    assert saved.iloc[0]["date"].startswith("2026-09-16 04:00:00") or saved.iloc[0]["date"].startswith("2026-09-16T04:00:00")
