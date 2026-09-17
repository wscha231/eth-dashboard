from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

FARSIDE_ETH_ALL_DATA_URL = "https://farside.co.uk/ethereum-etf-flow-all-data/"
SOURCE_ID = "farside_eth_etf_all_data"
SOURCE_NAME = "Farside Investors"
UNIT = "USD_millions"
LICENSE_STATUS = "unreviewed_no_public_redistribution"
HISTORICAL_AVAILABILITY_MODE = "reconstructed_conservative_t_plus_1_noon_utc"
PROSPECTIVE_AVAILABILITY_MODE = "prospective_receipt"
REVISION_AVAILABILITY_MODE = "prospective_revision_receipt"
PUBLICATION_TIME_STATUS = "not_provided_by_source"
SOURCE_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError)


class _Tables(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: list[list[list[str]]] = []
        self.depth = 0
        self.rows: list[list[str]] | None = None
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if tag == "table":
            self.depth += 1
            if self.depth == 1:
                self.rows = []
        elif self.depth == 1 and tag == "tr":
            self.row = []
        elif self.depth == 1 and tag in {"td", "th"} and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if self.depth == 1 and tag in {"td", "th"} and self.cell is not None:
            assert self.row is not None
            self.row.append(" ".join("".join(self.cell).split()))
            self.cell = None
        elif self.depth == 1 and tag == "tr" and self.row is not None:
            if any(self.row):
                assert self.rows is not None
                self.rows.append(self.row)
            self.row = self.cell = None
        elif tag == "table" and self.depth:
            if self.depth == 1:
                if self.rows:
                    self.tables.append(self.rows)
                self.rows = None
            self.depth -= 1


def utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def fetch_farside_html(
    *, url: str = FARSIDE_ETH_ALL_DATA_URL, timeout: int = 30, retries: int = 3,
    backoff_seconds: float = 1.5, max_bytes: int = 5_000_000,
    opener: Callable[..., Any] = urlopen,
) -> bytes:
    headers = {
        "User-Agent": "EtherForecast research collector/1.0 (+https://etherforecast.live)",
        "Accept": "text/html,application/xhtml+xml",
    }
    last: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            with opener(Request(url, headers=headers), timeout=timeout) as response:
                raw = response.read(max_bytes + 1)
            if len(raw) > max_bytes:
                raise ValueError("oversized Farside response")
            if b"<table" not in raw.lower():
                raise ValueError("Farside response does not contain an HTML table")
            return raw
        except SOURCE_ERRORS as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt == retries - 1:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    assert last is not None
    raise last


def _number(value: str) -> float:
    text = str(value).replace("\xa0", " ").replace(",", "").replace("$", "").replace("*", "").strip()
    if text.lower() in {"", "-", "–", "—", "n/a", "na", "null"}:
        return float("nan")
    negative = text.startswith("(") and text.endswith(")")
    text = text[1:-1].strip() if negative else text
    number = float(text)
    return -number if negative else number


def _ticker(value: str) -> str | None:
    token = re.sub(r"\s+", "", str(value).upper())
    return token if re.fullmatch(r"[A-Z][A-Z0-9.]{1,7}", token) else None


def parse_farside_eth_table(html: bytes | str) -> pd.DataFrame:
    parser = _Tables()
    parser.feed(html.decode("utf-8", errors="replace") if isinstance(html, bytes) else str(html))
    table = None
    ticker_row = None
    for candidate in parser.tables:
        for i, row in enumerate(candidate):
            if {"ETHA", "FETH", "ETHE"}.issubset({cell.strip().upper() for cell in row}):
                table, ticker_row = candidate, i
                break
        if table is not None:
            break
    if table is None or ticker_row is None:
        raise ValueError("Ethereum ETF ticker table not found")

    tickers = [t for cell in table[ticker_row] if (t := _ticker(cell)) and t != "TOTAL"]
    if len(tickers) < 5 or len(tickers) != len(set(tickers)):
        raise ValueError("invalid or duplicate ETF ticker header")

    rows: list[dict[str, Any]] = []
    for raw in table[ticker_row + 1:]:
        try:
            day = pd.to_datetime(raw[0], format="%d %b %Y", utc=True)
        except (ValueError, TypeError, IndexError):
            continue
        if len(raw) < len(tickers) + 2:
            raise ValueError(f"incomplete ETF flow row for {day.date().isoformat()}")
        record: dict[str, Any] = {"date": day}
        for ticker, cell in zip(tickers, raw[1:1 + len(tickers)]):
            record[f"flow_{ticker.lower()}_usd_m"] = _number(cell)
        record["total_usd_m"] = _number(raw[-1])
        values = [record[f"flow_{ticker.lower()}_usd_m"] for ticker in tickers]
        record.update(
            reported_fund_count=int(sum(np.isfinite(v) for v in values)),
            fund_count=len(tickers),
            known_fund_sum_usd_m=float(np.nansum(values)) if any(np.isfinite(v) for v in values) else float("nan"),
            all_funds_reported=bool(all(np.isfinite(v) for v in values)),
        )
        rows.append(record)
    if not rows:
        raise ValueError("no dated Ethereum ETF flow rows found")
    frame = pd.DataFrame(rows).set_index("date").sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    if frame["total_usd_m"].notna().sum() == 0:
        raise ValueError("ETF total flow column is empty")
    frame.index.name = "date"
    return frame


def raw_sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _same(a: Any, b: Any) -> bool:
    if _missing(a) and _missing(b):
        return True
    if _missing(a) or _missing(b):
        return False
    try:
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12)
    except (TypeError, ValueError):
        return str(a) == str(b)


def _iso(value: Any) -> str:
    return utc(value).isoformat()


def _metadata(day: pd.Timestamp, receipt: pd.Timestamp, started: pd.Timestamp, sha: str, *, initial: bool) -> dict[str, Any]:
    prospective = not initial and day.floor("D") >= started.floor("D")
    return {
        "event_time": _iso(day.floor("D")),
        "published_at": "",
        "publication_time_status": PUBLICATION_TIME_STATUS,
        "available_at": _iso(receipt if prospective else day.floor("D") + pd.Timedelta(days=1, hours=12)),
        "ingested_at": _iso(receipt),
        "revision": 0,
        "kind": "observed",
        "vintage_mode": "observed_vintages" if prospective else "reconstructed",
        "license_status": LICENSE_STATUS,
        "availability_mode": PROSPECTIVE_AVAILABILITY_MODE if prospective else HISTORICAL_AVAILABILITY_MODE,
        "source_id": SOURCE_ID, "source_url": FARSIDE_ETH_ALL_DATA_URL, "unit": UNIT, "raw_sha256": sha,
    }


def _revision_metadata(old: dict[str, Any], day: pd.Timestamp, receipt: pd.Timestamp, sha: str) -> dict[str, Any]:
    parsed = pd.to_numeric(old.get("revision", 0), errors="coerce")
    return {
        "event_time": old.get("event_time") or _iso(day.floor("D")),
        "published_at": "", "publication_time_status": PUBLICATION_TIME_STATUS,
        "available_at": _iso(receipt), "ingested_at": _iso(receipt),
        "revision": (int(parsed) if pd.notna(parsed) else 0) + 1,
        "kind": "observed", "vintage_mode": "observed_vintages",
        "license_status": LICENSE_STATUS, "availability_mode": REVISION_AVAILABILITY_MODE,
        "source_id": SOURCE_ID, "source_url": FARSIDE_ETH_ALL_DATA_URL, "unit": UNIT, "raw_sha256": sha,
    }


def reconcile_snapshot(
    snapshot: pd.DataFrame, *, existing: pd.DataFrame | None = None, state: dict[str, Any] | None = None,
    receipt_time: Any, raw_hash: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    if snapshot is None or snapshot.empty:
        raise ValueError("snapshot is empty")
    receipt = utc(receipt_time)
    snapshot = snapshot.copy()
    snapshot.index = pd.to_datetime(snapshot.index, utc=True)
    snapshot = snapshot[~snapshot.index.duplicated(keep="last")].sort_index()

    initial = not state or not state.get("collector_started_at")
    state = dict(state or {})
    if initial:
        state.update(schema=1, source_id=SOURCE_ID, source_url=FARSIDE_ETH_ALL_DATA_URL,
                     collector_started_at=_iso(receipt), license_status=LICENSE_STATUS)
    started = utc(state["collector_started_at"])

    oldmap: dict[pd.Timestamp, dict[str, Any]] = {}
    if existing is not None and not existing.empty:
        old = existing.copy()
        if "date" in old.columns:
            old.index = pd.to_datetime(old.pop("date"), utc=True)
        else:
            old.index = pd.to_datetime(old.index, utc=True)
        oldmap = {utc(i).floor("D"): r.to_dict() for i, r in old.iterrows()}

    out = {k: dict(v) for k, v in oldmap.items()}
    events, stats = [], {"new_rows": 0, "revised_rows": 0, "unchanged_rows": 0}
    flow_columns = sorted(c for c in snapshot.columns if c.startswith("flow_") and c.endswith("_usd_m"))

    for index, row in snapshot.iterrows():
        day, incoming = utc(index).floor("D"), row.to_dict()
        prior = out.get(day)
        if prior is None:
            merged = {**incoming, **_metadata(day, receipt, started, raw_hash, initial=initial)}
            out[day] = merged
            events.append({"date": day.date().isoformat(), "change_type": "new", **merged})
            stats["new_rows"] += 1
            continue
        compare = sorted(set(flow_columns) | {c for c in prior if c.startswith("flow_") and c.endswith("_usd_m")})
        changed = any(not _same(prior.get(c), incoming.get(c)) for c in compare)
        changed = changed or not _same(prior.get("total_usd_m"), incoming.get("total_usd_m"))
        if not changed:
            stats["unchanged_rows"] += 1
            continue
        merged = {**prior, **incoming, **_revision_metadata(prior, day, receipt, raw_hash)}
        out[day] = merged
        events.append({"date": day.date().isoformat(), "change_type": "revision", **merged})
        stats["revised_rows"] += 1

    preferred = [
        "total_usd_m", "reported_fund_count", "fund_count", "known_fund_sum_usd_m", "all_funds_reported",
        "event_time", "published_at", "publication_time_status", "available_at", "ingested_at", "revision",
        "kind", "vintage_mode", "license_status", "availability_mode", "source_id", "source_url", "unit", "raw_sha256",
    ]
    all_columns = {c for row in out.values() for c in row}
    flows = sorted(c for c in all_columns if c.startswith("flow_") and c.endswith("_usd_m"))
    columns = flows + [c for c in preferred if c in all_columns] + sorted(all_columns - set(flows) - set(preferred))
    latest = pd.DataFrame([{"date": d.date().isoformat(), **out[d]} for d in sorted(out)])
    latest = latest[["date", *columns]]
    event_frame = pd.DataFrame(events)

    state.update(last_received_at=_iso(receipt), last_raw_sha256=raw_hash,
                 latest_source_date=snapshot.index.max().date().isoformat(), rows=int(len(latest)))
    return latest, event_frame, state, stats


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() and path.stat().st_size else {}


def write_state(root: str | Path, raw: bytes, *, receipt_time: Any | None = None) -> dict[str, Any]:
    base = Path(root)
    (base / "raw").mkdir(parents=True, exist_ok=True)
    receipt = utc(receipt_time or datetime.now(timezone.utc))
    sha = raw_sha256(raw)
    snapshot = parse_farside_eth_table(raw)
    latest_path, receipts_path = base / "eth_etf_flows_daily.csv", base / "eth_etf_flows_receipts.csv"
    state_path, report_path = base / "state.json", base / "report.json"

    latest, events, state, stats = reconcile_snapshot(
        snapshot, existing=_read_csv(latest_path), state=_read_json(state_path), receipt_time=receipt, raw_hash=sha
    )
    latest.to_csv(latest_path, index=False)
    receipts = pd.concat([_read_csv(receipts_path), events], ignore_index=True, sort=False)
    receipts.to_csv(receipts_path, index=False)
    (base / "raw" / "farside_latest.txt").write_bytes(raw)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    totals = pd.to_numeric(latest["total_usd_m"], errors="coerce")
    report = {
        "schema": 1, "generated_at_utc": _iso(receipt), "status": "research_only_rights_unreviewed",
        "source_id": SOURCE_ID, "source_name": SOURCE_NAME, "source_url": FARSIDE_ETH_ALL_DATA_URL,
        "source_unit": UNIT, "license_status": LICENSE_STATUS,
        "public_redistribution": "blocked_pending_rights_review",
        "published_at_status": PUBLICATION_TIME_STATUS, "historical_vintage_mode": "reconstructed",
        "historical_available_at_policy": HISTORICAL_AVAILABILITY_MODE,
        "prospective_available_at_policy": "actual collector receipt time", "raw_sha256": sha,
        "rows": int(len(latest)), "receipt_rows": int(len(receipts)),
        "first_date": str(latest["date"].min()), "last_date": str(latest["date"].max()),
        "non_null_total_rows": int(totals.notna().sum()), **stats,
        "model_use": "blocked_until_rights_coverage_pit_and_source_cross_checks_pass", "site_use": "blocked",
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
