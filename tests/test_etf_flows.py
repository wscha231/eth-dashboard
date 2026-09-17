import math

import pandas as pd

from data_lab.etf_flows import (
    LICENSE_STATUS,
    parse_farside_eth_table,
    reconcile_snapshot,
    write_state,
)


SAMPLE = b"""
<html><body>
<table><tr><td>other</td></tr></table>
<table>
<tr><th></th><th>Blackrock</th><th>Fidelity</th><th>Grayscale</th><th>Bitwise</th><th>VanEck</th></tr>
<tr><th></th><th>ETHA</th><th>FETH</th><th>ETHE</th><th>ETHW</th><th>ETHV</th></tr>
<tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>2.5%</td><td>0.2%</td><td>0.2%</td></tr>
<tr><td>23 Jul 2024</td><td>266.5</td><td>71.3</td><td>(484.1)</td><td>204.0</td><td>-</td><td>57.7</td></tr>
<tr><td>24 Jul 2024</td><td>17.4</td><td>74.5</td><td>(326.9)</td><td>29.6</td><td>19.8</td><td>(185.6)</td></tr>
<tr><td>Total</td><td>283.9</td><td>145.8</td><td>(811.0)</td><td>233.6</td><td>19.8</td><td>(127.9)</td></tr>
</table>
</body></html>
"""


def test_farside_table_parses_dynamic_tickers_negative_and_missing_cells():
    frame = parse_farside_eth_table(SAMPLE)
    assert list(frame.index.strftime("%Y-%m-%d")) == ["2024-07-23", "2024-07-24"]
    assert frame.iloc[0]["flow_etha_usd_m"] == 266.5
    assert frame.iloc[0]["flow_ethe_usd_m"] == -484.1
    assert math.isnan(frame.iloc[0]["flow_ethv_usd_m"])
    assert frame.iloc[0]["total_usd_m"] == 57.7
    assert frame.iloc[0]["fund_count"] == 5
    assert frame.iloc[0]["reported_fund_count"] == 4


def test_initial_backfill_is_reconstructed_and_future_row_uses_actual_receipt():
    snapshot = parse_farside_eth_table(SAMPLE)
    latest, _, state, stats = reconcile_snapshot(
        snapshot, receipt_time="2026-09-17T03:00:00Z", raw_hash="a" * 64
    )
    assert stats["new_rows"] == 2
    assert set(latest["vintage_mode"]) == {"reconstructed"}
    first = latest.iloc[0]
    assert first["available_at"] == "2024-07-24T12:00:00+00:00"
    assert first["license_status"] == LICENSE_STATUS
    assert first["published_at"] == ""

    future = snapshot.iloc[[1]].copy()
    future.index = pd.DatetimeIndex(["2026-09-17T00:00:00Z"])
    future.iloc[0, future.columns.get_loc("total_usd_m")] = 12.3
    latest2, _, _, _ = reconcile_snapshot(
        future,
        existing=latest,
        state=state,
        receipt_time="2026-09-18T03:00:00Z",
        raw_hash="b" * 64,
    )
    added = latest2[latest2["date"] == "2026-09-17"].iloc[0]
    assert added["vintage_mode"] == "observed_vintages"
    assert added["availability_mode"] == "prospective_receipt"
    assert added["available_at"] == "2026-09-18T03:00:00+00:00"


def test_revision_becomes_available_only_at_revision_receipt_and_unchanged_row_is_preserved():
    snapshot = parse_farside_eth_table(SAMPLE)
    latest, _, state, _ = reconcile_snapshot(
        snapshot, receipt_time="2026-09-17T03:00:00Z", raw_hash="a" * 64
    )
    changed = snapshot.copy()
    changed.iloc[0, changed.columns.get_loc("total_usd_m")] = 58.8
    latest2, events, _, stats = reconcile_snapshot(
        changed,
        existing=latest,
        state=state,
        receipt_time="2026-09-18T04:00:00Z",
        raw_hash="b" * 64,
    )
    assert stats == {"new_rows": 0, "revised_rows": 1, "unchanged_rows": 1}
    assert len(events) == 1
    revised = latest2[latest2["date"] == "2024-07-23"].iloc[0]
    unchanged = latest2[latest2["date"] == "2024-07-24"].iloc[0]
    assert int(revised["revision"]) == 1
    assert revised["available_at"] == "2026-09-18T04:00:00+00:00"
    assert revised["raw_sha256"] == "b" * 64
    assert int(unchanged["revision"]) == 0
    assert unchanged["raw_sha256"] == "a" * 64


def test_write_state_keeps_append_only_receipt_journal(tmp_path):
    report = write_state(tmp_path, SAMPLE, receipt_time="2026-09-17T03:00:00Z")
    assert report["rows"] == 2
    assert report["new_rows"] == 2
    assert (tmp_path / "eth_etf_flows_daily.csv").exists()
    assert (tmp_path / "eth_etf_flows_receipts.csv").exists()
    assert (tmp_path / "raw" / "farside_latest.txt").exists()

    report2 = write_state(tmp_path, SAMPLE, receipt_time="2026-09-18T03:00:00Z")
    assert report2["new_rows"] == 0
    assert report2["unchanged_rows"] == 2
    receipts = pd.read_csv(tmp_path / "eth_etf_flows_receipts.csv")
    assert len(receipts) == 2
