from __future__ import annotations

import hashlib
import io
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forecasting.lead_signal_data import (
    BINANCE_KLINE_COLUMNS,
    BinanceArchiveSpec,
    daily_equivalent_funding_rate,
    infer_epoch_unit,
    parse_binance_s3_listing,
    parse_checksum_text,
    parse_defillama_chain_tvl,
    parse_defillama_dex_volume,
    parse_defillama_stablecoins,
    parse_deribit_funding_history,
    parse_okx_funding_history,
    read_binance_kline_zip,
    strict_json_dumps,
    summarize_binance_listing,
    summarize_funding_intervals,
    validate_daily_history,
    validate_hourly_klines,
    verify_checksum,
    write_daily_csv,
    write_immutable_json,
)


def _epoch(timestamp: pd.Timestamp, unit: str) -> int:
    nanoseconds = int(timestamp.value)
    if unit == "ms":
        return nanoseconds // 1_000_000
    if unit == "us":
        return nanoseconds // 1_000
    raise ValueError(unit)


def _kline_rows(
    start: str = "2025-01-01",
    *,
    periods: int = 3,
    unit: str = "us",
) -> list[list[object]]:
    rows: list[list[object]] = []
    for offset, opened_at in enumerate(pd.date_range(start, periods=periods, freq="h")):
        closed_at = opened_at + pd.Timedelta(hours=1) - pd.Timedelta(microseconds=1)
        if unit == "ms":
            closed_at = opened_at + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
        price = 100.0 + offset
        rows.append(
            [
                _epoch(opened_at, unit),
                price,
                price + 2.0,
                price - 2.0,
                price + 1.0,
                10.0,
                _epoch(closed_at, unit),
                1000.0,
                50,
                4.0,
                400.0,
                0,
            ]
        )
    return rows


def _write_kline_zip(
    path: Path,
    rows: list[list[object]],
    *,
    header: bool = False,
) -> None:
    output = io.StringIO()
    if header:
        output.write(",".join(BINANCE_KLINE_COLUMNS) + "\n")
    for row in rows:
        output.write(",".join(str(value) for value in row) + "\n")
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(path.with_suffix(".csv").name, output.getvalue())


def test_checksum_requires_exact_digest_and_filename(tmp_path: Path) -> None:
    archive = tmp_path / "ETHUSDT-1h-2025-01.zip"
    archive.write_bytes(b"archive")
    digest = hashlib.sha256(b"archive").hexdigest()
    checksum = f"{digest}  {archive.name}\n"

    assert parse_checksum_text(checksum, archive.name) == digest
    assert verify_checksum(archive, checksum) == (digest, digest)

    with pytest.raises(ValueError, match="filename mismatch"):
        parse_checksum_text(f"{digest}  other.zip\n", archive.name)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_checksum(archive, f"{'0' * 64}  {archive.name}\n")


def test_timestamp_unit_detection_covers_binance_2025_transition() -> None:
    milliseconds = [_epoch(pd.Timestamp("2024-12-31"), "ms")]
    microseconds = [_epoch(pd.Timestamp("2025-01-01"), "us")]

    assert infer_epoch_unit(milliseconds) == "ms"
    assert infer_epoch_unit(microseconds) == "us"
    with pytest.raises(ValueError, match="Unsupported epoch"):
        infer_epoch_unit([123])


def test_s3_listing_records_missing_month_and_checksum() -> None:
    spec = BinanceArchiveSpec("spot_eth", "spot", "ETHUSDT")
    xml = f"""<?xml version="1.0" encoding="UTF-8"?>
    <ListBucketResult xmlns="http://s3.amazonaws.com/doc/2006-03-01/">
      <Prefix>{spec.prefix}</Prefix><IsTruncated>false</IsTruncated>
      <Contents><Key>{spec.archive_key("2024-12")}</Key><LastModified>a</LastModified><ETag>e1</ETag><Size>10</Size></Contents>
      <Contents><Key>{spec.checksum_key("2024-12")}</Key><LastModified>a</LastModified><ETag>e2</ETag><Size>88</Size></Contents>
      <Contents><Key>{spec.archive_key("2025-02")}</Key><LastModified>b</LastModified><ETag>e3</ETag><Size>11</Size></Contents>
    </ListBucketResult>"""

    items = parse_binance_s3_listing(xml, spec.prefix)
    summary = summarize_binance_listing(spec, items)

    assert summary["first_month"] == "2024-12"
    assert summary["last_month"] == "2025-02"
    assert summary["missing_months"] == ["2025-01"]
    assert summary["missing_checksum_keys"] == [spec.archive_key("2025-02")]


@pytest.mark.parametrize(
    ("month", "unit"),
    [("2024-12", "ms"), ("2025-01", "us")],
)
def test_kline_zip_reads_ms_and_us_archives(
    tmp_path: Path,
    month: str,
    unit: str,
) -> None:
    path = tmp_path / f"ETHUSDT-1h-{month}.zip"
    rows = _kline_rows(f"{month}-01", unit=unit)
    _write_kline_zip(path, rows, header=month == "2025-01")

    frame = read_binance_kline_zip(path)
    validation = validate_hourly_klines(
        frame,
        expected_start=pd.Timestamp(f"{month}-01"),
        expected_end_exclusive=pd.Timestamp(f"{month}-01") + pd.Timedelta(hours=3),
    )

    assert frame.attrs["open_time_unit"] == unit
    assert validation["status"] == "pass"
    assert validation["row_count"] == 3


def test_kline_validation_rejects_gaps_duplicates_order_and_bad_values(
    tmp_path: Path,
) -> None:
    rows = _kline_rows(periods=4)
    rows.pop(1)
    rows.append(rows[-1].copy())
    rows[-1][1] = 200.0
    rows[-1][2] = 150.0
    rows[-1][5] = -1.0
    rows[-1][9] = 20.0
    rows[0], rows[1] = rows[1], rows[0]
    path = tmp_path / "bad.zip"
    _write_kline_zip(path, rows)

    frame = read_binance_kline_zip(path)
    validation = validate_hourly_klines(
        frame,
        expected_start=pd.Timestamp("2025-01-01"),
        expected_end_exclusive=pd.Timestamp("2025-01-01 04:00:00"),
    )

    assert validation["status"] == "fail"
    assert validation["counts"]["missing_bar_count"] == 1
    assert validation["counts"]["duplicate_open_time_count"] == 1
    assert validation["counts"]["non_monotonic_input"] is True
    assert validation["counts"]["ohlc_violation_count"] >= 1
    assert validation["counts"]["volume_violation_count"] >= 1
    assert validation["counts"]["taker_volume_violation_count"] >= 1


def test_cutoff_rejects_a_partial_or_future_bar(tmp_path: Path) -> None:
    path = tmp_path / "future.zip"
    _write_kline_zip(path, _kline_rows(periods=2))
    frame = read_binance_kline_zip(path)

    validation = validate_hourly_klines(
        frame,
        expected_start=pd.Timestamp("2025-01-01"),
        expected_end_exclusive=pd.Timestamp("2025-01-01 02:00:00"),
        cutoff=pd.Timestamp("2025-01-01 01:30:00"),
    )

    assert validation["status"] == "fail"
    assert validation["counts"]["bar_after_cutoff_count"] == 1


def test_defillama_parsers_filter_as_of_and_report_calendar_gaps() -> None:
    day_1 = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp())
    day_3 = int(pd.Timestamp("2024-01-03", tz="UTC").timestamp())
    after = int(pd.Timestamp("2024-01-04", tz="UTC").timestamp())
    as_of = pd.Timestamp("2024-01-03")

    stablecoins = parse_defillama_stablecoins(
        [
            {"date": day_1, "totalCirculatingUSD": {"peggedUSD": 10.0}},
            {"date": day_3, "totalCirculatingUSD": {"peggedUSD": 12.0}},
            {"date": after, "totalCirculatingUSD": {"peggedUSD": 13.0}},
        ],
        chain_slug="ethereum",
        as_of_date=as_of,
    )
    tvl = parse_defillama_chain_tvl(
        [{"date": day_1, "tvl": 100.0}, {"date": day_3, "tvl": 120.0}],
        chain_slug="ethereum",
        as_of_date=as_of,
    )
    dex = parse_defillama_dex_volume(
        {"totalDataChart": [[day_1, 50.0], [day_3, 60.0]]},
        chain_slug="ethereum",
        as_of_date=as_of,
    )

    assert stablecoins.index.max() == as_of
    assert len(tvl) == len(dex) == 2
    validation = validate_daily_history(stablecoins)
    assert validation["status"] == "fail"
    assert validation["missing_day_count"] == 1


def test_funding_parsers_preserve_interval_metadata() -> None:
    okx = parse_okx_funding_history(
        {
            "code": "0",
            "data": [
                {"fundingTime": "1704096000000", "fundingRate": "0.001"},
                {"fundingTime": "1704067200000", "fundingRate": "0.002"},
            ],
        }
    )
    intervals = summarize_funding_intervals(okx)
    assert intervals["observed_intervals_hours"] == [8.0]
    assert daily_equivalent_funding_rate(0.001, 8.0) == pytest.approx(0.003)

    deribit = parse_deribit_funding_history(
        {
            "result": [
                {"timestamp": 1704067200000, "interest_1h": "0.0001"},
                {"timestamp": 1704070800000, "interest_1h": "0.0002"},
            ]
        }
    )
    assert len(deribit) == 2
    assert summarize_funding_intervals(deribit)["observed_intervals_hours"] == [1.0]


def test_strict_json_and_immutable_manifest(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    payload = {"b": np.int64(2), "a": pd.Timestamp("2025-01-01")}

    first_hash = write_immutable_json(path, payload)
    assert write_immutable_json(path, payload) == first_hash
    assert json.loads(path.read_text(encoding="utf-8"))["b"] == 2
    assert strict_json_dumps(payload).endswith("\n")

    with pytest.raises(FileExistsError, match="immutable manifest"):
        write_immutable_json(path, {"b": 3})
    with pytest.raises(ValueError, match="Non-finite"):
        strict_json_dumps({"bad": np.nan})


def test_daily_source_table_gzip_is_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {"value": [1.0, 2.0]},
        index=pd.to_datetime(["2025-01-01", "2025-01-02"]),
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"

    assert write_daily_csv(frame, first) == write_daily_csv(frame, second)
    assert first.read_bytes() == second.read_bytes()
    restored = pd.read_csv(first, index_col="date")
    assert restored["value"].tolist() == [1.0, 2.0]


def test_generated_readiness_is_strict_and_offline_only() -> None:
    manifest_path = Path("lake/manifests/lead_signal_sources.json")
    report_path = Path("lake/reports/lead_signal_source_readiness.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["decision"] == "pass_for_pr2_offline"
    assert report["source_status"]["binance_hourly"] == "ready"
    assert report["source_status"]["defillama_history"] == "ready"
    assert report["gate"]["offline_feature_work_approved"] is True
    assert report["gate"]["production_use_approved"] is False
    assert report["gate"]["model_training_performed"] is False
    assert report["gate"]["daily_forecast_modified"] is False
    assert (
        report["manifest_sha256"]
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert manifest["scope"] == "offline_lead_signal_sources"
    for source in manifest["sources"]["defillama"]["sources"].values():
        source_path = Path(source["local_path"])
        assert source_path.is_file()
        assert (
            hashlib.sha256(source_path.read_bytes()).hexdigest()
            == source["local_sha256"]
        )
