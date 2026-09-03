from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import time
import zipfile
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from xml.etree import ElementTree

import numpy as np
import pandas as pd

BINANCE_ARCHIVE_BASE_URL = "https://data.binance.vision"
BINANCE_ARCHIVE_FALLBACK_URL = (
    "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
)
BINANCE_S3_LIST_URL = BINANCE_ARCHIVE_FALLBACK_URL
BINANCE_KLINE_COLUMNS = (
    "open_time_raw",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time_raw",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
)
BINANCE_SPOT_MICROSECOND_START = pd.Timestamp("2025-01-01")
HOUR = pd.Timedelta(hours=1)
HTTP_EXCEPTIONS = (HTTPError, URLError, TimeoutError, OSError)
FETCH_EXCEPTIONS = HTTP_EXCEPTIONS + (ValueError,)


@dataclass(frozen=True)
class BinanceArchiveSpec:
    source_id: str
    market: str
    symbol: str
    interval: str = "1h"

    @property
    def prefix(self) -> str:
        if self.market == "spot":
            root = "data/spot/monthly/klines"
        elif self.market == "um_futures":
            root = "data/futures/um/monthly/klines"
        else:
            raise ValueError(f"Unsupported Binance market: {self.market}")
        return f"{root}/{self.symbol}/{self.interval}/"

    def filename(self, month: str) -> str:
        validate_month(month)
        return f"{self.symbol}-{self.interval}-{month}.zip"

    def archive_key(self, month: str) -> str:
        return f"{self.prefix}{self.filename(month)}"

    def checksum_key(self, month: str) -> str:
        return f"{self.archive_key(month)}.CHECKSUM"


BINANCE_ARCHIVE_SPECS = (
    BinanceArchiveSpec("binance_spot_ethusdt_1h", "spot", "ETHUSDT"),
    BinanceArchiveSpec("binance_um_ethusdt_1h", "um_futures", "ETHUSDT"),
    BinanceArchiveSpec("binance_spot_btcusdt_1h", "spot", "BTCUSDT"),
    BinanceArchiveSpec("binance_um_btcusdt_1h", "um_futures", "BTCUSDT"),
)


def validate_month(month: str) -> None:
    if not re.fullmatch(r"\d{4}-\d{2}", month):
        raise ValueError(f"Invalid month: {month!r}")
    period = pd.Period(month, freq="M")
    if period.strftime("%Y-%m") != month:
        raise ValueError(f"Invalid month: {month!r}")


def month_bounds(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    validate_month(month)
    period = pd.Period(month, freq="M")
    return period.start_time, (period + 1).start_time


def calendar_months(start_month: str, end_month: str) -> list[str]:
    validate_month(start_month)
    validate_month(end_month)
    start = pd.Period(start_month, freq="M")
    end = pd.Period(end_month, freq="M")
    if end < start:
        raise ValueError("end_month must not precede start_month")
    return [
        period.strftime("%Y-%m") for period in pd.period_range(start, end, freq="M")
    ]


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_text(text: str, expected_filename: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("Checksum payload must contain exactly one non-empty line")
    parts = lines[0].split()
    if len(parts) != 2:
        raise ValueError("Checksum payload must contain a digest and filename")
    digest, filename = parts
    filename = filename.lstrip("*")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
        raise ValueError("Checksum payload does not contain a SHA-256 digest")
    if Path(filename).name != expected_filename or filename != Path(filename).name:
        raise ValueError(
            f"Checksum filename mismatch: expected {expected_filename!r}, got {filename!r}"
        )
    return digest.lower()


def verify_checksum(path: Path, checksum_text: str) -> tuple[str, str]:
    expected = parse_checksum_text(checksum_text, path.name)
    observed = sha256_file(path)
    if observed != expected:
        raise ValueError(
            f"SHA-256 mismatch for {path.name}: expected {expected}, observed {observed}"
        )
    return expected, observed


def fetch_url_bytes(
    url: str,
    *,
    max_bytes: int,
    timeout_seconds: float = 30.0,
    retries: int = 3,
) -> bytes:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    request = Request(
        url,
        headers={
            "Accept": "*/*",
            "User-Agent": "eth-lead-signal-preflight/1.0",
        },
    )
    last_error: Exception | None = None
    for attempt in range(max(retries, 1)):
        try:
            with urlopen(request, timeout=timeout_seconds) as response:
                declared = response.headers.get("Content-Length")
                if declared is not None and int(declared) > max_bytes:
                    raise ValueError(
                        f"Response exceeds byte limit: {declared} > {max_bytes} for {url}"
                    )
                chunks: list[bytes] = []
                total = 0
                while True:
                    chunk = response.read(min(1024 * 1024, max_bytes + 1 - total))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(
                            f"Response exceeded byte limit while streaming: {url}"
                        )
                    chunks.append(chunk)
                return b"".join(chunks)
        except FETCH_EXCEPTIONS as exc:
            last_error = exc
            if attempt + 1 < max(retries, 1):
                time.sleep(0.5 * (2**attempt))
    assert last_error is not None
    raise last_error


def fetch_json(
    url: str,
    *,
    max_bytes: int = 50 * 1024 * 1024,
    timeout_seconds: float = 30.0,
    retries: int = 3,
) -> Any:
    payload = fetch_url_bytes(
        url,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        retries=retries,
    )
    return json.loads(payload.decode("utf-8"))


def parse_binance_s3_listing(
    xml_payload: bytes | str, prefix: str
) -> list[dict[str, Any]]:
    if isinstance(xml_payload, str):
        xml_payload = xml_payload.encode("utf-8")
    root = ElementTree.fromstring(xml_payload)
    observed_prefix = root.findtext("{*}Prefix") or ""
    if observed_prefix != prefix:
        raise ValueError(
            f"S3 prefix mismatch: expected {prefix!r}, observed {observed_prefix!r}"
        )
    if (root.findtext("{*}IsTruncated") or "false").lower() == "true":
        raise ValueError("Truncated S3 listings are not accepted by the preflight")

    items: list[dict[str, Any]] = []
    for node in root.findall("{*}Contents"):
        key = node.findtext("{*}Key") or ""
        if not key.startswith(prefix):
            raise ValueError(f"S3 listing escaped requested prefix: {key!r}")
        items.append(
            {
                "key": key,
                "last_modified": node.findtext("{*}LastModified") or "",
                "etag": (node.findtext("{*}ETag") or "").strip('"'),
                "size_bytes": int(node.findtext("{*}Size") or 0),
            }
        )
    return sorted(items, key=lambda item: item["key"])


def summarize_binance_listing(
    spec: BinanceArchiveSpec,
    items: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    filename_pattern = re.compile(
        rf"^{re.escape(spec.prefix + spec.symbol + '-' + spec.interval + '-')}"
        r"(\d{4}-\d{2})\.zip$"
    )
    months: list[str] = []
    zip_items: list[dict[str, Any]] = []
    checksum_keys = {item["key"] for item in items if item["key"].endswith(".CHECKSUM")}
    for item in items:
        match = filename_pattern.fullmatch(item["key"])
        if not match:
            continue
        month = match.group(1)
        validate_month(month)
        months.append(month)
        zip_items.append(item)
    if not months:
        raise ValueError(f"No monthly archives listed for {spec.source_id}")
    if len(months) != len(set(months)):
        raise ValueError(f"Duplicate monthly archives listed for {spec.source_id}")

    months = sorted(months)
    expected_months = calendar_months(months[0], months[-1])
    missing = sorted(set(expected_months) - set(months))
    missing_checksums = [
        item["key"]
        for item in zip_items
        if f"{item['key']}.CHECKSUM" not in checksum_keys
    ]
    canonical_listing = json.dumps(
        zip_items,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return {
        "source_id": spec.source_id,
        "first_month": months[0],
        "last_month": months[-1],
        "archive_count": len(months),
        "missing_months": missing,
        "missing_checksum_keys": missing_checksums,
        "listing_sha256": sha256_bytes(canonical_listing),
        "months": months,
    }


def select_binance_probe_months(
    spec: BinanceArchiveSpec,
    listed_months: Sequence[str],
) -> list[str]:
    months = sorted(set(listed_months))
    if not months:
        return []
    selected = {months[0], months[-1]}
    if spec.market == "spot":
        for boundary_month in ("2024-12", "2025-01"):
            if boundary_month in months:
                selected.add(boundary_month)
    return sorted(selected)


def infer_epoch_unit(values: Iterable[Any]) -> str:
    numeric = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    numeric = numeric[numeric > 0]
    if numeric.empty:
        raise ValueError("Cannot infer timestamp unit from empty values")
    magnitude = float(numeric.median())
    if 1e11 <= magnitude < 1e14:
        return "ms"
    if 1e14 <= magnitude < 1e17:
        return "us"
    raise ValueError(f"Unsupported epoch timestamp magnitude: {magnitude}")


def _safe_zip_member(
    archive: zipfile.ZipFile, max_uncompressed_bytes: int
) -> zipfile.ZipInfo:
    members = [member for member in archive.infolist() if not member.is_dir()]
    csv_members = [
        member for member in members if member.filename.lower().endswith(".csv")
    ]
    if len(csv_members) != 1 or len(members) != 1:
        raise ValueError("Binance archive must contain exactly one CSV member")
    member = csv_members[0]
    member_path = Path(member.filename)
    if member_path.is_absolute() or ".." in member_path.parts:
        raise ValueError("Unsafe path in Binance archive")
    if member.file_size > max_uncompressed_bytes:
        raise ValueError(
            f"Uncompressed CSV exceeds limit: {member.file_size} > {max_uncompressed_bytes}"
        )
    if member.compress_size and member.file_size / member.compress_size > 100:
        raise ValueError("Suspicious compression ratio in Binance archive")
    return member


def read_binance_kline_zip(
    path: Path,
    *,
    max_uncompressed_bytes: int = 25 * 1024 * 1024,
) -> pd.DataFrame:
    with zipfile.ZipFile(path) as archive:
        member = _safe_zip_member(archive, max_uncompressed_bytes)
        csv_payload = archive.read(member)

    raw = pd.read_csv(io.BytesIO(csv_payload), header=None, dtype=str)
    if raw.empty:
        raise ValueError(f"Empty Binance kline archive: {path.name}")
    first_token = str(raw.iloc[0, 0]).strip().lower().replace(" ", "_")
    if first_token in {"open_time", "open_time_raw", "opentime"}:
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] != len(BINANCE_KLINE_COLUMNS):
        raise ValueError(
            f"Expected {len(BINANCE_KLINE_COLUMNS)} kline columns, got {raw.shape[1]}"
        )
    raw.columns = BINANCE_KLINE_COLUMNS

    numeric_columns = [column for column in BINANCE_KLINE_COLUMNS if column != "ignore"]
    for column in numeric_columns:
        raw[column] = pd.to_numeric(raw[column], errors="coerce")

    open_unit = infer_epoch_unit(raw["open_time_raw"])
    close_unit = infer_epoch_unit(raw["close_time_raw"])
    if open_unit != close_unit:
        raise ValueError(
            f"Open/close timestamp unit mismatch: {open_unit} versus {close_unit}"
        )
    frame = raw.copy()
    frame["open_time"] = pd.to_datetime(
        frame["open_time_raw"], unit=open_unit, utc=True, errors="coerce"
    ).dt.tz_convert(None)
    frame["close_time"] = pd.to_datetime(
        frame["close_time_raw"], unit=close_unit, utc=True, errors="coerce"
    ).dt.tz_convert(None)
    frame.attrs["open_time_unit"] = open_unit
    frame.attrs["close_time_unit"] = close_unit
    frame.attrs["archive_member"] = member.filename
    return frame


def _iso_samples(values: Iterable[pd.Timestamp], limit: int = 10) -> list[str]:
    samples: list[str] = []
    for value in values:
        timestamp = pd.Timestamp(value)
        samples.append(timestamp.isoformat())
        if len(samples) >= limit:
            break
    return samples


def validate_hourly_klines(
    frame: pd.DataFrame,
    *,
    expected_start: pd.Timestamp | None = None,
    expected_end_exclusive: pd.Timestamp | None = None,
    allowed_missing_open_times: Sequence[pd.Timestamp] = (),
    cutoff: pd.Timestamp | None = None,
) -> dict[str, Any]:
    required = {
        "open_time",
        "close_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    }
    missing_columns = sorted(required - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing kline columns: {missing_columns}")
    if frame.empty:
        raise ValueError("Cannot validate an empty kline frame")

    open_times = pd.DatetimeIndex(frame["open_time"])
    close_times = pd.DatetimeIndex(frame["close_time"])
    invalid_timestamp_count = int(open_times.isna().sum() + close_times.isna().sum())
    valid_open_times = open_times.dropna()
    duplicate_mask = pd.Series(open_times).duplicated(keep=False).to_numpy()
    duplicate_times = (
        pd.DatetimeIndex(open_times[duplicate_mask]).dropna().unique().sort_values()
    )
    non_monotonic = not pd.Series(open_times).is_monotonic_increasing

    unique_sorted = valid_open_times.unique().sort_values()
    effective_start = (
        pd.Timestamp(expected_start)
        if expected_start is not None
        else unique_sorted.min()
    )
    effective_end = (
        pd.Timestamp(expected_end_exclusive)
        if expected_end_exclusive is not None
        else unique_sorted.max() + HOUR
    )
    if effective_end <= effective_start:
        raise ValueError("Expected kline range must be positive")
    expected = pd.date_range(
        effective_start, effective_end, freq=HOUR, inclusive="left"
    )
    allowed = pd.DatetimeIndex(pd.to_datetime(list(allowed_missing_open_times)))
    missing_times = expected.difference(unique_sorted).difference(allowed)
    extra_times = unique_sorted[
        (unique_sorted < effective_start) | (unique_sorted >= effective_end)
    ]

    numeric_required = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_volume",
        "trade_count",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    numeric = frame[numeric_required].apply(pd.to_numeric, errors="coerce")
    non_finite_count = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    price_positive = (numeric[["open", "high", "low", "close"]] > 0).all(axis=1)
    ohlc_valid = (
        (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1))
        & (numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1))
        & price_positive
    )
    volume_columns = [
        "volume",
        "quote_volume",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    ]
    volume_nonnegative = (numeric[volume_columns] >= 0).all(axis=1)
    tolerance_base = np.maximum(numeric["volume"].abs() * 1e-9, 1e-9)
    tolerance_quote = np.maximum(numeric["quote_volume"].abs() * 1e-9, 1e-9)
    taker_valid = (
        numeric["taker_buy_base_volume"] <= numeric["volume"] + tolerance_base
    ) & (numeric["taker_buy_quote_volume"] <= numeric["quote_volume"] + tolerance_quote)
    trade_count_valid = (numeric["trade_count"] >= 0) & np.isclose(
        numeric["trade_count"], numeric["trade_count"].round()
    )

    duration = pd.Series(close_times - open_times)
    duration_valid = (duration > HOUR - pd.Timedelta(milliseconds=2)) & (
        duration <= HOUR
    )
    aligned = pd.Series(open_times).dt.floor("h") == pd.Series(open_times)
    after_cutoff_count = 0
    if cutoff is not None:
        after_cutoff_count = int((close_times > pd.Timestamp(cutoff)).sum())

    counts = {
        "invalid_timestamp_count": invalid_timestamp_count,
        "duplicate_open_time_count": len(duplicate_times),
        "non_monotonic_input": bool(non_monotonic),
        "missing_bar_count": len(missing_times),
        "extra_bar_count": len(extra_times),
        "non_finite_numeric_count": non_finite_count,
        "ohlc_violation_count": int((~ohlc_valid).sum()),
        "volume_violation_count": int((~volume_nonnegative).sum()),
        "taker_volume_violation_count": int((~taker_valid).sum()),
        "trade_count_violation_count": int((~trade_count_valid).sum()),
        "close_duration_violation_count": int((~duration_valid).sum()),
        "hour_alignment_violation_count": int((~aligned).sum()),
        "bar_after_cutoff_count": after_cutoff_count,
    }
    passed = all(
        value is False if isinstance(value, bool) else value == 0
        for value in counts.values()
    )
    return {
        "status": "pass" if passed else "fail",
        "row_count": len(frame),
        "open_time_unit": frame.attrs.get("open_time_unit", "unknown"),
        "first_open_time": unique_sorted.min().isoformat(),
        "last_open_time": unique_sorted.max().isoformat(),
        "expected_start": effective_start.isoformat(),
        "expected_end_exclusive": effective_end.isoformat(),
        "counts": counts,
        "samples": {
            "duplicate_open_times": _iso_samples(duplicate_times),
            "missing_open_times": _iso_samples(missing_times),
            "extra_open_times": _iso_samples(extra_times),
        },
    }


def _to_utc_day(raw_value: Any, unit: str = "s") -> pd.Timestamp | None:
    try:
        value = pd.to_datetime(int(raw_value), unit=unit, utc=True)
    except (TypeError, ValueError, OverflowError):
        try:
            value = pd.to_datetime(raw_value, utc=True)
        except (TypeError, ValueError, OverflowError):
            return None
    return value.tz_convert(None).floor("D")


def _finalize_daily_frame(
    rows: list[dict[str, Any]],
    *,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame = frame.dropna(subset=["date"])
    frame = frame.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    frame = frame.loc[frame["date"] <= pd.Timestamp(as_of_date).floor("D")]
    return frame.set_index("date")


def parse_defillama_stablecoins(
    payload: Any,
    *,
    chain_slug: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise TypeError("DefiLlama stablecoin payload must be a list")
    rows: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        date = _to_utc_day(item.get("date"), unit="s")
        circulating = item.get("totalCirculatingUSD")
        if date is None or not isinstance(circulating, dict):
            continue
        values = [
            float(value)
            for value in circulating.values()
            if pd.notna(pd.to_numeric(value, errors="coerce"))
        ]
        pegged = pd.to_numeric(circulating.get("peggedUSD"), errors="coerce")
        rows.append(
            {
                "date": date,
                f"defillama_{chain_slug}_stablecoin_total_usd": (
                    float(sum(values)) if values else np.nan
                ),
                f"defillama_{chain_slug}_stablecoin_pegged_usd": pegged,
            }
        )
    return _finalize_daily_frame(rows, as_of_date=as_of_date)


def parse_defillama_chain_tvl(
    payload: Any,
    *,
    chain_slug: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(payload, list):
        raise TypeError("DefiLlama TVL payload must be a list")
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        date = _to_utc_day(item.get("date"), unit="s")
        if date is None:
            continue
        rows.append(
            {
                "date": date,
                f"defillama_{chain_slug}_chain_tvl_usd": pd.to_numeric(
                    item.get("tvl"), errors="coerce"
                ),
            }
        )
    return _finalize_daily_frame(rows, as_of_date=as_of_date)


def parse_defillama_dex_volume(
    payload: Any,
    *,
    chain_slug: str,
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise TypeError("DefiLlama DEX payload must be an object")
    chart = payload.get("totalDataChart")
    if not isinstance(chart, list):
        raise TypeError("DefiLlama DEX payload is missing totalDataChart")
    rows = []
    for item in chart:
        if not isinstance(item, (list, tuple)) or len(item) < 2:
            continue
        date = _to_utc_day(item[0], unit="s")
        if date is None:
            continue
        rows.append(
            {
                "date": date,
                f"defillama_{chain_slug}_dex_volume_usd": pd.to_numeric(
                    item[1], errors="coerce"
                ),
            }
        )
    return _finalize_daily_frame(rows, as_of_date=as_of_date)


def validate_daily_history(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {
            "status": "fail",
            "row_count": 0,
            "first_date": None,
            "last_date": None,
            "missing_day_count": 0,
            "duplicate_date_count": 0,
            "non_finite_value_count": 0,
            "negative_value_count": 0,
            "missing_dates": [],
        }
    index = pd.DatetimeIndex(frame.index)
    duplicate_count = int(index.duplicated(keep=False).sum())
    sorted_unique = index.unique().sort_values()
    expected = pd.date_range(sorted_unique.min(), sorted_unique.max(), freq="D")
    missing = expected.difference(sorted_unique)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    non_finite = int((~np.isfinite(numeric.to_numpy(dtype=float))).sum())
    negative = int((numeric < 0).sum().sum())
    passed = (
        duplicate_count == 0 and len(missing) == 0 and non_finite == 0 and negative == 0
    )
    return {
        "status": "pass" if passed else "fail",
        "row_count": len(frame),
        "first_date": sorted_unique.min().date().isoformat(),
        "last_date": sorted_unique.max().date().isoformat(),
        "missing_day_count": len(missing),
        "duplicate_date_count": duplicate_count,
        "non_finite_value_count": non_finite,
        "negative_value_count": negative,
        "missing_dates": _iso_samples(missing),
    }


def write_daily_csv(
    frame: pd.DataFrame,
    path: Path,
    *,
    float_format: str | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    output = frame.copy()
    output.index = pd.DatetimeIndex(output.index).strftime("%Y-%m-%d")
    output.index.name = "date"
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    csv_payload = output.to_csv(
        lineterminator="\n",
        float_format=float_format,
    ).encode("utf-8")
    if path.suffix == ".gz":
        with (
            temporary.open("wb") as raw_handle,
            gzip.GzipFile(
                filename="",
                mode="wb",
                compresslevel=9,
                fileobj=raw_handle,
                mtime=0,
            ) as compressed,
        ):
            compressed.write(csv_payload)
    else:
        temporary.write_bytes(csv_payload)
    os.replace(temporary, path)
    return sha256_file(path)


def parse_okx_funding_history(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict) or str(payload.get("code")) != "0":
        raise ValueError("OKX funding payload did not return code 0")
    data = payload.get("data")
    if not isinstance(data, list):
        raise TypeError("OKX funding payload is missing data")
    rows: list[dict[str, Any]] = []
    for item in data:
        if not isinstance(item, dict):
            continue
        timestamp = pd.to_datetime(
            pd.to_numeric(item.get("fundingTime"), errors="coerce"),
            unit="ms",
            utc=True,
            errors="coerce",
        )
        if pd.isna(timestamp):
            continue
        rows.append(
            {
                "timestamp": timestamp.tz_convert(None),
                "funding_rate": pd.to_numeric(item.get("fundingRate"), errors="coerce"),
                "realized_rate": pd.to_numeric(
                    item.get("realizedRate"), errors="coerce"
                ),
                "formula_type": str(item.get("formulaType") or ""),
                "method": str(item.get("method") or ""),
            }
        )
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def summarize_funding_intervals(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty or len(frame) < 2:
        return {"observed_intervals_hours": [], "interval_counts": {}}
    deltas = pd.Series(frame.index).diff().dropna().dt.total_seconds() / 3600.0
    rounded = deltas.round(6)
    counts = rounded.value_counts().sort_index()
    return {
        "observed_intervals_hours": [float(value) for value in counts.index],
        "interval_counts": {
            str(float(key)): int(value) for key, value in counts.items()
        },
    }


def daily_equivalent_funding_rate(
    rate: float, settlement_interval_hours: float
) -> float:
    if not math.isfinite(rate):
        raise ValueError("Funding rate must be finite")
    if settlement_interval_hours <= 0 or settlement_interval_hours > 24:
        raise ValueError("Settlement interval must be in (0, 24]")
    return float(rate * (24.0 / settlement_interval_hours))


def parse_deribit_funding_history(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, dict):
        raise TypeError("Deribit payload must be an object")
    data = payload.get("result")
    if not isinstance(data, list):
        raise TypeError("Deribit payload is missing result")
    if not data:
        return pd.DataFrame()
    frame = pd.DataFrame(data)
    if "timestamp" not in frame.columns:
        raise ValueError("Deribit funding payload is missing timestamp")
    frame["timestamp"] = pd.to_datetime(
        pd.to_numeric(frame["timestamp"], errors="coerce"),
        unit="ms",
        utc=True,
        errors="coerce",
    ).dt.tz_convert(None)
    frame = frame.dropna(subset=["timestamp"])
    for column in ("index_price", "interest_8h", "interest_1h", "prev_index_price"):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return (
        frame.drop_duplicates(subset=["timestamp"], keep="last")
        .sort_values("timestamp")
        .set_index("timestamp")
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("Non-finite value cannot be serialized to strict JSON")
        return number
    if isinstance(value, np.bool_):
        return bool(value)
    if hasattr(value, "__dataclass_fields__"):
        return json_safe(asdict(value))
    return value


def strict_json_dumps(payload: Any) -> str:
    return (
        json.dumps(
            json_safe(payload),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )


def write_immutable_json(path: Path, payload: Any, *, replace: bool = False) -> str:
    rendered = strict_json_dumps(payload)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing == rendered:
            return sha256_bytes(rendered.encode("utf-8"))
        if not replace:
            raise FileExistsError(
                f"Refusing to replace immutable manifest {path}; pass replace=True explicitly"
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(rendered, encoding="utf-8")
    os.replace(temporary, path)
    return sha256_bytes(rendered.encode("utf-8"))
