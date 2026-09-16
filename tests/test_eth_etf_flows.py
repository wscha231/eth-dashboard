import gzip

import numpy as np
import pandas as pd
import pytest

from data_lab.eth_etf_flows import (
    CONTRACT_VERSION,
    NUMERIC_COLUMNS,
    RIGHTS_STATUS,
    append_versions,
    archive_raw,
    availability_sidecar,
    empty_versions,
    latest_snapshot,
    parse_farside_html,
    parse_flow_cell,
    row_digest,
    sha256,
    validation_report,
)
from scripts.daily_source_state import allowed as public_daily_cache_allowed


HTML = b"""<!doctype html><html><body>
<table><tr><th>noise</th></tr><tr><td>nothing</td></tr></table>
<table>
<tr><th></th><th>Blackrock</th><th>Blackrock</th><th>Fidelity</th><th>Bitwise</th><th>21 Shares</th><th>VanEck</th><th>Invesco</th><th>Franklin</th><th>Grayscale</th><th>Grayscale</th><th>Total</th></tr>
<tr><th></th><th>ETHA</th><th>ETHB</th><th>FETH</th><th>ETHW</th><th>TETH</th><th>ETHV</th><th>QETH</th><th>EZET</th><th>ETHE</th><th>ETH</th><th></th></tr>
<tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>0.25%</td><td>0.20%</td><td>0.21%</td><td>0.20%</td><td>0.25%</td><td>0.19%</td><td>2.50%</td><td>0.15%</td><td></td></tr>
<tr><td>Seed</td><td>10.6</td><td>104.7</td><td>4.4</td><td>2.5</td><td>2.3</td><td>10.2</td><td>1.1</td><td>2.7</td><td>9,199.3*</td><td>1,022.5*</td><td>10,360</td></tr>
<tr><td>23 Jul 2024</td><td>266.5</td><td>-</td><td>71.3</td><td>204.0</td><td>7.5</td><td>7.6</td><td>5.5</td><td>13.2</td><td>(484.1)</td><td>15.1</td><td>106.6</td></tr>
<tr><td>24 Jul 2024</td><td>17.4</td><td>-</td><td>74.5</td><td>29.6</td><td>0.0</td><td>19.8</td><td>2.5</td><td>3.9</td><td>(326.9)</td><td>45.9</td><td>(133.3)</td></tr>
<tr><td>Total</td><td>283.9</td><td>-</td><td>145.8</td><td>233.6</td><td>7.5</td><td>27.4</td><td>8.0</td><td>17.1</td><td>(811.0)</td><td>61.0</td><td>(26.7)</td></tr>
</table></body></html>"""


def parsed():
    return parse_farside_html(HTML)


def test_parse_flow_cell_preserves_missing_and_parentheses_negative():
    assert np.isnan(parse_flow_cell("-"))
    assert parse_flow_cell("(484.1)") == -484.1
    assert parse_flow_cell("9,199.3*") == 9199.3
    with pytest.raises(ValueError, match="unrecognized"):
        parse_flow_cell("Fee")


def test_parser_keeps_only_dated_rows_and_expected_schema():
    frame = parsed()
    assert len(frame) == 2
    assert frame.date.iloc[0] == pd.Timestamp("2024-07-23T00:00:00Z")
    assert np.isnan(frame.ethb_usd_m.iloc[0])
    assert frame.ethe_usd_m.iloc[0] == -484.1
    assert frame.total_usd_m.iloc[1] == -133.3
    assert list(frame.columns) == ["date", *NUMERIC_COLUMNS]


def test_initial_backfill_is_reconstructed_d_plus_one_noon():
    versions, changes = append_versions(
        empty_versions(), parsed(), received_at="2026-09-16T11:00:00Z", raw_sha256="a" * 64
    )
    assert changes["reconstructed"] == 2
    assert set(versions.availability_mode) == {"reconstructed_conservative_d_plus_1_12utc"}
    assert versions.available_at.iloc[0] == pd.Timestamp("2024-07-24T12:00:00Z")
    assert versions.received_at.iloc[0] == pd.Timestamp("2026-09-16T11:00:00Z")
    assert versions.contract_version.iloc[0] == CONTRACT_VERSION


def test_new_date_and_correction_use_actual_receipt_not_historical_availability():
    prior, _ = append_versions(
        empty_versions(), parsed(), received_at="2026-09-16T11:00:00Z", raw_sha256="a" * 64
    )
    current = parsed()
    current.loc[current.date.eq(pd.Timestamp("2024-07-24T00:00:00Z")), "etha_usd_m"] = 18.0
    extra = current.iloc[[-1]].copy()
    extra["date"] = pd.Timestamp("2026-09-15T00:00:00Z")
    extra["etha_usd_m"] = 12.3
    current = pd.concat([current, extra], ignore_index=True)
    receipt = pd.Timestamp("2026-09-16T11:30:00Z")
    versions, changes = append_versions(prior, current, received_at=receipt, raw_sha256="b" * 64)
    assert changes == {"new_dates": 1, "corrections": 1, "unchanged": 1, "reconstructed": 0}
    latest = latest_snapshot(versions)
    corrected = latest.loc[latest.date.eq(pd.Timestamp("2024-07-24T00:00:00Z"))].iloc[0]
    new = latest.loc[latest.date.eq(pd.Timestamp("2026-09-15T00:00:00Z"))].iloc[0]
    assert corrected.available_at == receipt
    assert corrected.availability_mode == "observed_correction_receipt"
    assert new.available_at == receipt
    assert new.availability_mode == "observed_new_date_receipt"


def test_unchanged_refetch_does_not_create_new_row_versions():
    first, _ = append_versions(empty_versions(), parsed(), received_at="2026-09-16T11:00:00Z", raw_sha256="a" * 64)
    second, changes = append_versions(first, parsed(), received_at="2026-09-16T12:00:00Z", raw_sha256="a" * 64)
    assert len(second) == len(first)
    assert changes["unchanged"] == len(first)


def test_availability_sidecar_matches_latest_row_version():
    versions, _ = append_versions(empty_versions(), parsed(), received_at="2026-09-16T11:00:00Z", raw_sha256="a" * 64)
    sidecar = availability_sidecar(latest_snapshot(versions))
    # ETHB is missing in both fixture rows, so it must not manufacture a sidecar value.
    assert "ethb_usd_m" not in set(sidecar.column)
    assert (pd.to_datetime(sidecar.available_at, utc=True) >= pd.to_datetime(sidecar.date, utc=True)).all()


def test_raw_receipt_archive_is_content_addressed_and_deterministic(tmp_path):
    first = archive_raw(HTML, tmp_path)
    second = archive_raw(HTML, tmp_path)
    assert first == second
    assert first.name == sha256(HTML) + ".html.gz"
    assert gzip.decompress(first.read_bytes()) == HTML


def test_validation_reports_total_disagreement_without_rewriting_source_total():
    versions, _ = append_versions(empty_versions(), parsed(), received_at="2026-09-16T11:00:00Z", raw_sha256="a" * 64)
    snapshot = latest_snapshot(versions)
    original = snapshot.total_usd_m.copy()
    report = validation_report(snapshot)
    assert snapshot.total_usd_m.equals(original)
    # Fixture has missing ETHB, therefore no complete-fund total check is invented.
    assert report["total_check_rows"] == 0
    assert RIGHTS_STATUS == "unreviewed_no_public_redistribution"


def test_row_digest_changes_on_numeric_correction():
    row = parsed().iloc[0].copy()
    before = row_digest(row)
    row["etha_usd_m"] += 0.1
    assert row_digest(row) != before


def test_rights_unreviewed_etf_rows_are_drive_only_not_public_git_cache():
    for name in (
        "lake/raw/vendor/eth_etf_farside_daily.csv",
        "lake/raw/vendor/eth_etf_farside_versions.csv",
        "lake/raw/vendor/eth_etf_farside_fetch_receipts.csv",
        "lake/raw/vendor/eth_etf_farside_daily.availability.csv",
    ):
        assert public_daily_cache_allowed(name) is False
    assert public_daily_cache_allowed("lake/raw/vendor/bitget_eth_free_features.csv") is True
