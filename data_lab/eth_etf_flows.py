"""ETH spot-ETF flow research data with explicit PIT and rights boundaries.

Historical Farside rows are reconstructed research, not historical receipts.
The initial backfill uses a conservative D+1 12:00 UTC eligibility convention.
After collection begins, newly observed dates and changed historical rows become
eligible only at their actual HTTP receipt timestamp.

This module does not authorize public redistribution, production model use,
or any price-alpha claim.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

FARSIDE_URL = "https://farside.co.uk/ethereum-etf-flow-all-data/"
SOURCE = "farside_investors"
CONTRACT_VERSION = "eth_etf_flow_source_contract_v1_1"
PARSER_VERSION = "farside_eth_table_v1"
TICKERS = ("ETHA", "ETHB", "FETH", "ETHW", "TETH", "ETHV", "QETH", "EZET", "ETHE", "ETH")
FLOW_COLUMNS = tuple(t.lower() + "_usd_m" for t in TICKERS)
NUMERIC_COLUMNS = FLOW_COLUMNS + ("total_usd_m",)
RIGHTS_STATUS = "unreviewed_no_public_redistribution"
PRODUCTION_USE_APPROVED = False
HISTORICAL_AVAILABLE_HOUR_UTC = 12
MAX_RESPONSE_BYTES = 5_000_000
FETCH_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError)


def utc(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("finite timestamp required")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fetch_html(
    url: str = FARSIDE_URL,
    *,
    timeout: int = 30,
    retries: int = 3,
    opener: Callable[..., Any] = urlopen,
) -> tuple[bytes, pd.Timestamp]:
    """Fetch one bounded public HTML response and stamp the actual receipt time."""
    headers = {
        "User-Agent": "EtherForecast-research/1.0 (+https://etherforecast.live)",
        "Accept": "text/html,application/xhtml+xml",
    }
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            request = Request(url, headers=headers)
            with opener(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                final_url = getattr(response, "url", url)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("ETF source response exceeded size limit")
            if not raw:
                raise ValueError("ETF source returned an empty response")
            if not str(final_url).startswith("https://farside.co.uk/"):
                raise ValueError("unexpected ETF source redirect")
            return raw, utc(datetime.now(timezone.utc))
        except FETCH_ERRORS as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= retries - 1:
                break
            time.sleep(1.5 * (attempt + 1))
    if last is not None:
        raise last
    raise RuntimeError("ETF source fetch failed without an exception")


def parse_flow_cell(value: Any) -> float:
    text = str(value).replace("\xa0", " ").strip()
    if text in {"", "-", "—", "–", "N/A", "n/a"}:
        return np.nan
    text = text.replace("*", "").replace(",", "").replace("$", "").strip()
    negative = text.startswith("(") and text.endswith(")")
    if negative:
        text = text[1:-1].strip()
    if not re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)", text):
        raise ValueError(f"unrecognized ETF flow cell: {value!r}")
    number = float(text)
    return -abs(number) if negative else number


def _find_flow_table(soup: BeautifulSoup):
    candidates = []
    required = {"ETHA", "FETH", "ETHE", "EZET"}
    for table in soup.find_all("table"):
        text = table.get_text(" ", strip=True)
        if required.issubset(set(re.findall(r"\b[A-Z]{3,5}\b", text))):
            candidates.append(table)
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one Ethereum ETF flow table, found {len(candidates)}")
    return candidates[0]


def parse_farside_html(raw: bytes) -> pd.DataFrame:
    """Parse only dated flow rows; fee/seed/summary rows never enter the dataset."""
    if not raw:
        raise ValueError("empty ETF HTML")
    soup = BeautifulSoup(raw, "html.parser")
    table = _find_flow_table(soup)
    dated: list[dict[str, Any]] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True) for cell in tr.find_all(["th", "td"])]
        if not cells:
            continue
        try:
            date = pd.to_datetime(cells[0], format="%d %b %Y", utc=True, errors="raise").floor("D")
        except (ValueError, TypeError):
            continue
        expected = len(TICKERS) + 2
        if len(cells) != expected:
            raise ValueError(f"dated ETF row has {len(cells)} cells; expected {expected}")
        row: dict[str, Any] = {"date": date}
        for ticker, cell in zip(TICKERS, cells[1 : 1 + len(TICKERS)], strict=True):
            row[ticker.lower() + "_usd_m"] = parse_flow_cell(cell)
        row["total_usd_m"] = parse_flow_cell(cells[-1])
        dated.append(row)
    if not dated:
        raise ValueError("no dated Ethereum ETF rows found")
    frame = pd.DataFrame(dated).sort_values("date").reset_index(drop=True)
    if frame["date"].duplicated().any():
        raise ValueError("duplicate Ethereum ETF event dates")
    if frame.iloc[0]["date"] != pd.Timestamp("2024-07-23", tz="UTC"):
        raise ValueError("unexpected first Ethereum ETF flow date")
    if not frame["date"].is_monotonic_increasing:
        raise ValueError("Ethereum ETF dates are not monotonic")
    for column in NUMERIC_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame[list(NUMERIC_COLUMNS)].isna().all(axis=1).any():
        raise ValueError("ETF row contains no numeric evidence")
    return frame


def _float_or_none(value: Any) -> float | None:
    return None if pd.isna(value) else float(value)


def row_digest(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    payload = {
        "date": utc(get("date")).strftime("%Y-%m-%d"),
        **{column: _float_or_none(get(column)) for column in NUMERIC_COLUMNS},
    }
    return sha256(canonical(payload))


def historical_available_at(event_date: Any) -> pd.Timestamp:
    return utc(event_date).floor("D") + pd.Timedelta(days=1, hours=HISTORICAL_AVAILABLE_HOUR_UTC)


def empty_versions() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", *NUMERIC_COLUMNS, "available_at", "availability_mode", "received_at",
        "source", "source_url", "raw_sha256", "row_sha256", "contract_version", "parser_version",
    ])


def load_versions(path: Path) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        return empty_versions()
    frame = pd.read_csv(path)
    required = set(empty_versions().columns)
    if set(frame.columns) != required:
        raise ValueError("ETF version ledger schema mismatch")
    for column in ("date", "available_at", "received_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if frame.duplicated(["date", "row_sha256"]).any():
        raise ValueError("ETF version ledger contains duplicate row versions")
    return frame.sort_values(["date", "received_at", "row_sha256"]).reset_index(drop=True)


def append_versions(
    prior: pd.DataFrame,
    current: pd.DataFrame,
    *,
    received_at: Any,
    raw_sha256: str,
    source_url: str = FARSIDE_URL,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Append new row versions without backdating newly observed corrections."""
    receipt = utc(received_at)
    if not re.fullmatch(r"[0-9a-f]{64}", raw_sha256):
        raise ValueError("raw_sha256 must be lowercase SHA256")
    prior = prior.copy() if not prior.empty else empty_versions()
    latest_hash: dict[pd.Timestamp, str] = {}
    if not prior.empty:
        ordered = prior.sort_values(["date", "received_at"])
        latest_hash = {
            utc(row.date).floor("D"): str(row.row_sha256)
            for row in ordered.groupby("date", sort=False).tail(1).itertuples(index=False)
        }
    first_collection = prior.empty
    appended: list[dict[str, Any]] = []
    counts = {"new_dates": 0, "corrections": 0, "unchanged": 0, "reconstructed": 0}
    for row in current.to_dict("records"):
        date = utc(row["date"]).floor("D")
        digest = row_digest(row)
        old = latest_hash.get(date)
        if old == digest:
            counts["unchanged"] += 1
            continue
        if first_collection:
            available = historical_available_at(date)
            mode = "reconstructed_conservative_d_plus_1_12utc"
            counts["reconstructed"] += 1
        elif old is None:
            available = receipt
            mode = "observed_new_date_receipt"
            counts["new_dates"] += 1
        else:
            available = receipt
            mode = "observed_correction_receipt"
            counts["corrections"] += 1
        if available < date:
            raise ValueError("ETF available_at precedes event date")
        payload = {
            "date": date,
            **{column: _float_or_none(row.get(column)) for column in NUMERIC_COLUMNS},
            "available_at": available,
            "availability_mode": mode,
            "received_at": receipt,
            "source": SOURCE,
            "source_url": source_url,
            "raw_sha256": raw_sha256,
            "row_sha256": digest,
            "contract_version": CONTRACT_VERSION,
            "parser_version": PARSER_VERSION,
        }
        appended.append(payload)
    if not appended:
        return prior, counts
    merged = pd.concat([prior, pd.DataFrame(appended)], ignore_index=True)
    for column in ("date", "available_at", "received_at"):
        merged[column] = pd.to_datetime(merged[column], utc=True)
    if merged.duplicated(["date", "row_sha256"]).any():
        raise ValueError("attempted to append duplicate ETF row version")
    return merged.sort_values(["date", "received_at", "row_sha256"]).reset_index(drop=True), counts


def latest_snapshot(versions: pd.DataFrame) -> pd.DataFrame:
    if versions.empty:
        return pd.DataFrame(columns=empty_versions().columns)
    frame = versions.sort_values(["date", "received_at", "row_sha256"]).groupby("date", as_index=False).tail(1)
    return frame.sort_values("date").reset_index(drop=True)


def availability_sidecar(snapshot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in snapshot.itertuples(index=False):
        for column in NUMERIC_COLUMNS:
            value = getattr(row, column)
            if pd.isna(value):
                continue
            rows.append({
                "date": utc(row.date).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
                "column": column,
                "source": SOURCE,
                "available_at": utc(row.available_at).isoformat(),
                "availability_mode": row.availability_mode,
                "row_sha256": row.row_sha256,
            })
    return pd.DataFrame(rows)


def validation_report(snapshot: pd.DataFrame) -> dict[str, Any]:
    if snapshot.empty:
        raise ValueError("ETF snapshot is empty")
    complete = snapshot[list(FLOW_COLUMNS)].notna().all(axis=1)
    computed = snapshot.loc[complete, list(FLOW_COLUMNS)].sum(axis=1)
    source_total = snapshot.loc[complete, "total_usd_m"]
    differences = (computed - source_total).abs()
    mismatch = differences > 0.25  # source rows are rounded to 0.1 US$m
    return {
        "rows": int(len(snapshot)),
        "first_date": utc(snapshot["date"].min()).strftime("%Y-%m-%d"),
        "last_date": utc(snapshot["date"].max()).strftime("%Y-%m-%d"),
        "complete_fund_rows": int(complete.sum()),
        "total_check_rows": int(len(differences)),
        "total_mismatch_rows": int(mismatch.sum()),
        "max_abs_total_difference_usd_m": float(differences.max()) if len(differences) else None,
        "missing_by_column": {column: int(snapshot[column].isna().sum()) for column in NUMERIC_COLUMNS},
    }


def archive_raw(raw: bytes, output_dir: Path) -> Path:
    digest = sha256(raw)
    directory = Path(output_dir) / "receipts" / "eth_etf_farside"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{digest}.html.gz"
    compressed = gzip.compress(raw, mtime=0)
    if path.exists() and path.read_bytes() != compressed:
        raise ValueError("raw ETF receipt hash path has conflicting bytes")
    if not path.exists():
        path.write_bytes(compressed)
    return path


def append_fetch_receipt(path: Path, *, received_at: Any, raw_sha256: str, source_url: str = FARSIDE_URL) -> pd.DataFrame:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame(
        columns=["received_at", "source", "source_url", "raw_sha256", "contract_version", "parser_version"]
    )
    row = pd.DataFrame([{
        "received_at": utc(received_at).isoformat(),
        "source": SOURCE,
        "source_url": source_url,
        "raw_sha256": raw_sha256,
        "contract_version": CONTRACT_VERSION,
        "parser_version": PARSER_VERSION,
    }])
    merged = pd.concat([old, row], ignore_index=True)
    merged = merged.drop_duplicates(["received_at", "raw_sha256"], keep="last").sort_values("received_at")
    merged.to_csv(path, index=False)
    return merged
