"""Collect and version US spot Ethereum ETF daily flows for research.

Historical third-party data is reconstructed research evidence, not a historical receipt
vintage. Raw/history files from this module are intentionally kept out of public Git data
until rights review; the workflow stores them in the private durable Drive stream.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
from io import StringIO
import json
from pathlib import Path
import re
from typing import Iterable

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import requests

SOURCE_URL = "https://farside.co.uk/ethereum-etf-flow-all-data/"
CONTRACT_ID = "eth_etf_flows_v1_20260917"
TICKERS = ("ETHA", "ETHB", "FETH", "ETHW", "TETH", "ETHV", "QETH", "EZET", "ETHE", "ETH")
FLOW_COLUMNS = tuple(f"{ticker.lower()}_usd_m" for ticker in TICKERS)
REVISION_FIELDS = ("source_date", *FLOW_COLUMNS, "total_usd_m")
USER_AGENT = "EtherForecast research collector/1.0 (+https://etherforecast.live/)"


def utc(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    else:
        stamp = stamp.tz_convert("UTC")
    return stamp


def _amount(text: str) -> float:
    value = str(text).strip().replace("\xa0", " ")
    if value in {"", "-", "–", "—", "N/A", "n/a"}:
        return np.nan
    negative = value.startswith("(") and value.endswith(")")
    if negative:
        value = value[1:-1]
    value = value.replace(",", "").replace("*", "").strip()
    number = float(value)
    return -number if negative else number


def _date(text: str) -> pd.Timestamp | None:
    value = " ".join(str(text).replace("\xa0", " ").split())
    if not re.fullmatch(r"\d{1,2} [A-Za-z]{3} \d{4}", value):
        return None
    stamp = pd.to_datetime(value, format="%d %b %Y", utc=True, errors="raise")
    return stamp.normalize()


def parse_farside_html(raw: bytes | str) -> pd.DataFrame:
    """Parse only the daily Ethereum ETF table; dashes remain missing, never zero."""
    if isinstance(raw, bytes):
        text = raw.decode("utf-8", errors="strict")
    else:
        text = raw
    soup = BeautifulSoup(text, "html.parser")
    chosen = None
    ticker_row = None
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = [" ".join(cell.get_text(" ", strip=True).split()) for cell in tr.find_all(["th", "td"])]
            if cells:
                rows.append(cells)
        for i, cells in enumerate(rows):
            if set(TICKERS).issubset(set(cells)):
                chosen, ticker_row = rows, i
                break
        if chosen is not None:
            break
    if chosen is None:
        raise ValueError("Ethereum ETF ticker header not found")

    header = chosen[ticker_row]
    positions = {}
    for ticker in TICKERS:
        matches = [i for i, value in enumerate(header) if value == ticker]
        if len(matches) != 1:
            raise ValueError(f"ambiguous ETF ticker column: {ticker}")
        positions[ticker] = matches[0]
    total_index = max(positions.values()) + 1

    records = []
    for cells in chosen[ticker_row + 1:]:
        if not cells:
            continue
        source_date = _date(cells[0])
        if source_date is None:
            continue
        if len(cells) <= total_index:
            raise ValueError(f"short ETF data row for {source_date.date()}")
        row = {"source_date": source_date}
        for ticker, index in positions.items():
            row[f"{ticker.lower()}_usd_m"] = _amount(cells[index])
        row["total_usd_m"] = _amount(cells[total_index])
        records.append(row)
    if not records:
        raise ValueError("no daily Ethereum ETF flow rows found")
    frame = pd.DataFrame(records).sort_values("source_date").reset_index(drop=True)
    if frame.source_date.duplicated().any():
        raise ValueError("duplicate source dates in ETF table")
    if frame.source_date.min() != pd.Timestamp("2024-07-23", tz="UTC"):
        raise ValueError("unexpected Ethereum ETF history start")
    if frame.total_usd_m.isna().any():
        raise ValueError("reported Total is required for every daily row")
    return frame


def row_hash(row) -> str:
    payload = {}
    for key in REVISION_FIELDS:
        value = row[key]
        if key == "source_date":
            payload[key] = utc(value).strftime("%Y-%m-%d")
        elif pd.isna(value):
            payload[key] = None
        else:
            payload[key] = round(float(value), 8)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def assumed_available_at(source_date) -> pd.Timestamp:
    return utc(source_date).normalize() + pd.Timedelta(days=1)


def _read_revisions(path: Path | None) -> pd.DataFrame:
    if not path or not Path(path).is_file():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["source_date", "available_at", "received_at"])
    for col in ("source_date", "available_at", "received_at"):
        frame[col] = pd.to_datetime(frame[col], utc=True)
    return frame


def merge_revisions(current: pd.DataFrame, previous: pd.DataFrame, received_at) -> tuple[pd.DataFrame, dict]:
    received_at = utc(received_at)
    history = previous.copy()
    known = set()
    known_dates = set()
    if not history.empty:
        known = set(zip(history.source_date.dt.strftime("%Y-%m-%d"), history.row_hash.astype(str)))
        known_dates = set(history.source_date.dt.strftime("%Y-%m-%d"))
    additions = []
    reconstructed = previous.empty
    for row in current.to_dict("records"):
        date_key = utc(row["source_date"]).strftime("%Y-%m-%d")
        hash_value = row_hash(row)
        if (date_key, hash_value) in known:
            continue
        if reconstructed:
            mode = "reconstructed_backfill"
            available = assumed_available_at(row["source_date"])
        elif date_key not in known_dates:
            mode = "prospective_receipt"
            available = max(assumed_available_at(row["source_date"]), received_at)
        else:
            mode = "prospective_correction"
            available = max(assumed_available_at(row["source_date"]), received_at)
        additions.append({
            **row,
            "row_hash": hash_value,
            "availability_mode": mode,
            "available_at": available,
            "received_at": received_at,
        })
    if additions:
        history = pd.concat([history, pd.DataFrame(additions)], ignore_index=True)
    if history.empty:
        raise ValueError("empty ETF revision history")
    history["source_date"] = pd.to_datetime(history.source_date, utc=True)
    history["available_at"] = pd.to_datetime(history.available_at, utc=True)
    history["received_at"] = pd.to_datetime(history.received_at, utc=True)
    history = history.sort_values(["source_date", "received_at", "row_hash"]).reset_index(drop=True)
    if history.duplicated(["source_date", "row_hash"]).any():
        raise ValueError("duplicate ETF source revision")
    stats = {
        "new_revisions": len(additions),
        "new_dates": sum(a["availability_mode"] == "prospective_receipt" for a in additions),
        "corrections": sum(a["availability_mode"] == "prospective_correction" for a in additions),
        "reconstructed_rows": sum(a["availability_mode"] == "reconstructed_backfill" for a in additions),
    }
    return history, stats


def latest_view(revisions: pd.DataFrame) -> pd.DataFrame:
    ordered = revisions.sort_values(["source_date", "received_at", "row_hash"])
    latest = ordered.groupby("source_date", as_index=False).tail(1).copy()
    first_seen = ordered.groupby("source_date")["received_at"].min()
    latest["first_received_at"] = latest.source_date.map(first_seen)
    latest["component_sum_usd_m"] = latest[list(FLOW_COLUMNS)].sum(axis=1, min_count=len(FLOW_COLUMNS))
    latest["component_gap_usd_m"] = latest.component_sum_usd_m - latest.total_usd_m
    columns = ["source_date", *FLOW_COLUMNS, "total_usd_m", "component_sum_usd_m", "component_gap_usd_m",
               "row_hash", "availability_mode", "available_at", "first_received_at", "received_at"]
    return latest[columns].sort_values("source_date").reset_index(drop=True)


def quality_report(latest: pd.DataFrame, revisions: pd.DataFrame, *, received_at, raw_sha256: str, stats: dict) -> dict:
    complete_components = latest[list(FLOW_COLUMNS)].notna().all(axis=1)
    gaps = latest.loc[complete_components, "component_gap_usd_m"].abs()
    modes = revisions.availability_mode.value_counts().to_dict()
    coverage = {ticker: float(latest[f"{ticker.lower()}_usd_m"].notna().mean()) for ticker in TICKERS}
    return {
        "schema": 1,
        "contract_id": CONTRACT_ID,
        "generated_at": utc(received_at).isoformat(),
        "status": "research_only_rights_unreviewed",
        "source": {"name": "Farside Investors", "url": SOURCE_URL, "raw_sha256": raw_sha256},
        "rights_status": "unreviewed_no_public_redistribution",
        "pit_status": "reconstructed_history_plus_prospective_receipts",
        "rows": int(len(latest)),
        "revisions": int(len(revisions)),
        "start": latest.source_date.min().isoformat(),
        "end": latest.source_date.max().isoformat(),
        "fund_nonmissing_fraction": coverage,
        "fully_reported_component_rows": int(complete_components.sum()),
        "max_abs_component_gap_usd_m_when_fully_reported": float(gaps.max()) if len(gaps) else None,
        "availability_modes": {str(k): int(v) for k, v in modes.items()},
        "changes_this_run": stats,
        "model_use_approved": False,
        "public_redistribution_approved": False,
        "notes": [
            "Dash cells remain missing/not-reported and are never converted to zero.",
            "Parenthesized values are negative flows.",
            "Reported Total is preserved from source rather than reconstructed from incomplete components.",
            "Historical D+1 availability is a conservative research assumption, not an observed historical publication timestamp."
        ],
    }


def fetch_source(url: str = SOURCE_URL, *, timeout: int = 45) -> bytes:
    response = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}, timeout=timeout)
    response.raise_for_status()
    raw = response.content
    if len(raw) < 10_000 or b"Ethereum" not in raw or b"ETHA" not in raw:
        raise ValueError("unexpected ETF source response")
    return raw


def collect(output_dir: Path, *, previous_dir: Path | None = None, received_at=None,
            raw: bytes | None = None) -> dict:
    received_at = utc(received_at)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    raw = raw if raw is not None else fetch_source()
    raw_sha = hashlib.sha256(raw).hexdigest()
    raw_name = f"farside_eth_{received_at.strftime('%Y%m%dT%H%M%SZ')}_{raw_sha[:12]}.html.gz"
    with gzip.open(raw_dir / raw_name, "wb", mtime=0) as stream:
        stream.write(raw)

    parsed = parse_farside_html(raw)
    prior_path = Path(previous_dir) / "eth_etf_flow_revisions.csv.gz" if previous_dir else None
    previous = _read_revisions(prior_path)
    revisions, stats = merge_revisions(parsed, previous, received_at)
    latest = latest_view(revisions)
    revisions.to_csv(output_dir / "eth_etf_flow_revisions.csv.gz", index=False,
                     compression={"method": "gzip", "mtime": 0})
    latest.to_csv(output_dir / "eth_etf_flows_daily.csv", index=False)
    report = quality_report(latest, revisions, received_at=received_at, raw_sha256=raw_sha, stats=stats)
    (output_dir / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
