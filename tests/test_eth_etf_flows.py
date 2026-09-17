import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data_lab.eth_etf_flows import (
    FLOW_COLUMNS,
    TICKERS,
    assumed_available_at,
    collect,
    latest_view,
    merge_revisions,
    parse_farside_html,
)


HTML = b"""<!doctype html><html><body><table>
<tr><th></th><th>Blackrock</th><th>Blackrock</th><th>Fidelity</th><th>Bitwise</th><th>21 Shares</th><th>VanEck</th><th>Invesco</th><th>Franklin</th><th>Grayscale</th><th>Grayscale</th><th>Total</th></tr>
<tr><td></td><td>ETHA</td><td>ETHB</td><td>FETH</td><td>ETHW</td><td>TETH</td><td>ETHV</td><td>QETH</td><td>EZET</td><td>ETHE</td><td>ETH</td><td></td></tr>
<tr><td>Fee</td><td>0.25%</td><td>0.25%</td><td>0.25%</td><td>0.20%</td><td>0.21%</td><td>0.20%</td><td>0.25%</td><td>0.19%</td><td>2.50%</td><td>0.15%</td><td></td></tr>
<tr><td>23 Jul 2024</td><td>266.5</td><td>-</td><td>71.3</td><td>204.0</td><td>7.5</td><td>7.6</td><td>5.5</td><td>13.2</td><td>(484.1)</td><td>15.1</td><td>106.6</td></tr>
<tr><td>24 Jul 2024</td><td>17.4</td><td>-</td><td>74.5</td><td>29.6</td><td>0.0</td><td>19.8</td><td>2.5</td><td>3.9</td><td>(326.9)</td><td>45.9</td><td>(133.3)</td></tr>
<tr><td>02 Sep 2024</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>0.0</td></tr>
</table></body></html>"""


def test_parse_preserves_missing_zero_and_parenthesized_negative():
    frame = parse_farside_html(HTML)
    assert list(frame.source_date.dt.strftime("%Y-%m-%d")) == ["2024-07-23", "2024-07-24", "2024-09-02"]
    first = frame.iloc[0]
    assert first.etha_usd_m == 266.5
    assert np.isnan(first.ethb_usd_m)
    assert first.ethe_usd_m == -484.1
    assert first.total_usd_m == 106.6
    second = frame.iloc[1]
    assert second.teth_usd_m == 0.0
    holiday = frame.iloc[2]
    assert holiday[list(FLOW_COLUMNS)].isna().all()
    assert holiday.total_usd_m == 0.0


def test_parser_requires_full_expected_ticker_header():
    broken = HTML.replace(b"ETHB", b"BAD")
    with pytest.raises(ValueError, match="ticker header"):
        parse_farside_html(broken)


def test_first_collection_is_reconstructed_with_d_plus_one_assumption():
    current = parse_farside_html(HTML)
    revisions, stats = merge_revisions(current, pd.DataFrame(), "2026-09-17T01:00:00Z")
    assert stats["reconstructed_rows"] == 3
    assert set(revisions.availability_mode) == {"reconstructed_backfill"}
    assert revisions.iloc[0].available_at == pd.Timestamp("2024-07-24T00:00:00Z")
    assert assumed_available_at("2024-07-23T00:00:00Z") == pd.Timestamp("2024-07-24T00:00:00Z")


def test_unchanged_rows_do_not_create_fake_prospective_revisions():
    current = parse_farside_html(HTML)
    previous, _ = merge_revisions(current, pd.DataFrame(), "2026-09-17T01:00:00Z")
    merged, stats = merge_revisions(current, previous, "2026-09-18T01:00:00Z")
    assert len(merged) == len(previous)
    assert stats == {"new_revisions": 0, "new_dates": 0, "corrections": 0, "reconstructed_rows": 0}


def test_new_date_and_historical_correction_get_actual_receipt_boundary():
    initial = parse_farside_html(HTML)
    previous, _ = merge_revisions(initial, pd.DataFrame(), "2026-09-17T01:00:00Z")
    corrected = HTML.replace(b"266.5", b"267.0").replace(
        b"</table>",
        b"<tr><td>17 Sep 2026</td><td>10.0</td><td>1.0</td><td>2.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>0.0</td><td>13.0</td></tr></table>"
    )
    now = pd.Timestamp("2026-09-18T00:15:00Z")
    merged, stats = merge_revisions(parse_farside_html(corrected), previous, now)
    assert stats["new_dates"] == 1
    assert stats["corrections"] == 1
    latest = latest_view(merged)
    corrected_row = latest.loc[latest.source_date.eq(pd.Timestamp("2024-07-23T00:00:00Z"))].iloc[0]
    assert corrected_row.etha_usd_m == 267.0
    assert corrected_row.availability_mode == "prospective_correction"
    assert corrected_row.available_at == now
    new_row = latest.loc[latest.source_date.eq(pd.Timestamp("2026-09-17T00:00:00Z"))].iloc[0]
    assert new_row.availability_mode == "prospective_receipt"
    assert new_row.available_at == now


def test_collect_writes_private_research_state_and_report(tmp_path):
    out = tmp_path / "etf"
    report = collect(out, received_at="2026-09-17T01:02:03Z", raw=HTML)
    assert report["status"] == "research_only_rights_unreviewed"
    assert report["public_redistribution_approved"] is False
    assert report["model_use_approved"] is False
    assert (out / "eth_etf_flow_revisions.csv.gz").is_file()
    assert (out / "eth_etf_flows_daily.csv").is_file()
    assert (out / "report.json").is_file()
    raws = list((out / "raw").glob("*.html.gz"))
    assert len(raws) == 1
    with gzip.open(raws[0], "rb") as stream:
        assert stream.read() == HTML
    parsed_report = json.loads((out / "report.json").read_text())
    assert parsed_report["rows"] == 3
    assert parsed_report["availability_modes"] == {"reconstructed_backfill": 3}
