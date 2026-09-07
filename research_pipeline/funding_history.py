"""Bounded, resumable Deribit funding history; no production/model mutation.

Funding timestamps denote hourly interval ends. A large API request can return
only its final month, so each request covers <=28 days and completeness is
verified against the requested UTC grid. Retrieved vintages are never dated
back to their event times.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import hashlib
import json
from pathlib import Path
from threading import Event
import time
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

HOST = "https://www.deribit.com/api/v2/public/get_funding_rate_history"
VALUE_COLUMNS = ["interest_1h", "interest_8h", "index_price"]


def utc(value):
    stamp = pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def windows(start, end):
    start, end = utc(start), utc(end)
    if start >= end or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("start < end and whole UTC hours required")
    while start < end:
        upper = min(start + pd.Timedelta(days=28), end)
        yield start, upper
        start = upper


def request_url(start, end):
    return HOST + "?" + urlencode({"instrument_name": "ETH-PERPETUAL",
        "start_timestamp": int(utc(start).timestamp()*1000),
        "end_timestamp": int(utc(end).timestamp()*1000)})


def fetch_bytes(url, timeout):
    with urlopen(Request(url, headers={"User-Agent": "eth-dashboard-history/1.0"}),
                 timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("oversized funding response")
    return raw


def parse_funding(raw, start, end, received_at):
    payload = json.loads(raw)
    if not isinstance(payload, dict) or "error" in payload or not isinstance(payload.get("result"), list):
        raise ValueError("invalid funding response or JSON-RPC error")
    rows = payload["result"]
    columns = ["event_time", *VALUE_COLUMNS, "received_at"]
    if not rows:
        return pd.DataFrame(columns=columns)
    frame = pd.DataFrame(rows)
    required = ["timestamp", *VALUE_COLUMNS]
    if not set(required).issubset(frame):
        raise ValueError("missing funding fields")
    numeric = frame[required].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(numeric.to_numpy(float)).all():
        raise ValueError("non-finite funding data")
    if (numeric.timestamp % 3_600_000 != 0).any() or (numeric.index_price <= 0).any():
        raise ValueError("invalid hourly timestamp or index price")
    frame = numeric[VALUE_COLUMNS].copy()
    frame["event_time"] = pd.to_datetime(numeric.timestamp, unit="ms", utc=True)
    if frame.event_time.duplicated().any():
        raise ValueError("duplicate funding timestamp")
    if (frame.event_time > utc(received_at)).any():
        raise ValueError("future funding observation")
    # Boundary conventions may include the start; keep (start, end] only.
    frame = frame.loc[(frame.event_time > utc(start)) & (frame.event_time <= utc(end))]
    frame["received_at"] = utc(received_at)
    return frame[columns].sort_values("event_time").reset_index(drop=True)


def backfill(root, *, start, end, budget_seconds=600, max_requests=120,
             workers=3, fetcher=fetch_bytes):
    if budget_seconds <= 0 or max_requests < 1 or not 1 <= workers <= 3:
        raise ValueError("positive budgets and 1..3 workers required")
    start, end = utc(start), utc(end)
    if end > pd.Timestamp.now(tz="UTC").floor("h"):
        raise ValueError("end must be a closed hour")
    chunks = list(windows(start, end))
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    for name in ("raw", "cache", "manifests"):
        (root/name).mkdir(exist_ok=True)
    deadline = time.monotonic() + budget_seconds
    stopped = Event(); frames = []; records = []; errors = []; tasks = []
    for lower, upper in chunks:
        url = request_url(lower, upper)
        key = hashlib.sha256(url.encode()).hexdigest()
        cache = root/"cache"/(key + ".json")
        if cache.exists():
            record = json.loads(cache.read_text())
            raw = gzip.decompress((root/"raw"/(record["sha256"]+".json.gz")).read_bytes())
            if hashlib.sha256(raw).hexdigest() != record["sha256"] or record["url"] != url:
                raise ValueError("cached evidence checksum mismatch")
            frame = parse_funding(raw, lower, upper, record["received_at"])
            frames.append(frame); records.append({**record, "cached": True})
        elif len(tasks) < max_requests:
            tasks.append((lower, upper, url, cache))
        else:
            errors.append({"start": lower.isoformat(), "end": upper.isoformat(), "error": "request_budget"})

    def download(task):
        lower, upper, url, cache = task
        if stopped.is_set() or time.monotonic() >= deadline:
            return None
        try:
            raw = fetcher(url, min(20, deadline-time.monotonic()))
            received = pd.Timestamp.now(tz="UTC").isoformat()
            frame = parse_funding(raw, lower, upper, received)
            sha = hashlib.sha256(raw).hexdigest()
            record = {"url": url, "sha256": sha, "received_at": received,
                      "start": lower.isoformat(), "end": upper.isoformat(), "rows": len(frame)}
            target = root/"raw"/(sha+".json.gz")
            if not target.exists():
                target.write_bytes(gzip.compress(raw, mtime=0))
            # Partial responses remain evidence, but must be retried on a later run.
            expected = int((upper-lower)/pd.Timedelta(hours=1))
            if len(frame) == expected:
                cache.write_text(json.dumps(record, indent=2))
            return frame, {**record, "cached": False}
        except HTTPError as exc:
            if exc.code in (401, 403, 451, 429):
                stopped.set()  # no alternate host or retry storm
            raise

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download, task): task for task in tasks}
        for future in as_completed(pending):
            lower, upper, _, _ = pending[future]
            try:
                result = future.result()
                if result is None:
                    errors.append({"start": lower.isoformat(), "end": upper.isoformat(), "error": "stopped_or_time_budget"})
                else:
                    frame, record = result
                    frames.append(frame); records.append(record)
            except Exception as exc:
                errors.append({"start": lower.isoformat(), "end": upper.isoformat(),
                               "error": type(exc).__name__, "detail": str(exc)[:180]})
    columns = ["event_time", *VALUE_COLUMNS, "received_at"]
    hourly = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=columns)
    if len(hourly):
        hourly = hourly.sort_values("event_time").reset_index(drop=True)
        if hourly.event_time.duplicated().any():
            raise ValueError("overlapping funding windows")
    expected = pd.date_range(start+pd.Timedelta(hours=1), end, freq="h")
    actual = pd.DatetimeIndex(pd.to_datetime(hourly.event_time, utc=True))
    missing = expected.difference(actual)
    generated = pd.Timestamp.now(tz="UTC")
    report = {"generated_at": generated.isoformat(), "source": "deribit_eth_perpetual_funding",
              "requested_start_exclusive": start.isoformat(), "requested_end_inclusive": end.isoformat(),
              "rows": len(hourly), "expected_hours": len(expected), "missing_hours": len(missing),
              "missing_sample": [x.isoformat() for x in missing[:30]],
              "complete": len(missing) == 0 and not errors, "errors": errors,
              "cached_windows": sum(r["cached"] for r in records),
              "downloaded_windows": sum(not r["cached"] for r in records),
              "records": sorted(records, key=lambda r:r["start"]),
              "history_kind": "historical_reconstruction_observed_now",
              "live_eligible": False, "commercial_use": "separate_terms_review_required"}
    # Preserve every run. These files are research source snapshots, not issuance records.
    run_id = generated.strftime("%Y%m%dT%H%M%S%fZ")
    snapshot = root/("funding_hourly_"+run_id+".parquet")
    hourly.to_parquet(snapshot, index=False)
    report["snapshot"] = snapshot.name
    report["snapshot_sha256"] = hashlib.sha256(snapshot.read_bytes()).hexdigest()
    (root/"manifests"/(run_id+".json")).write_text(json.dumps(report, indent=2))
    (root/"latest_report.json").write_text(json.dumps(report, indent=2))
    return hourly, report


def daily_funding(hourly):
    """Sum disjoint 1h funding intervals; never sum overlapping 8h rates."""
    frame = hourly.copy()
    frame["event_time"] = pd.to_datetime(frame.event_time, utc=True)
    if frame.event_time.duplicated().any():
        raise ValueError("duplicate funding timestamp")
    frame["day"] = (frame.event_time-pd.Timedelta(hours=1)).dt.floor("D")
    rows = []
    for day, group in frame.groupby("day", sort=True):
        expected = pd.date_range(day+pd.Timedelta(hours=1), periods=24, freq="h")
        if not pd.DatetimeIndex(group.event_time.sort_values()).equals(expected):
            continue
        if not np.isfinite(group[VALUE_COLUMNS].to_numpy(float)).all():
            continue
        rows.append({"date": day, "funding_sum_1h": group.interest_1h.sum(),
                     "event_end": day+pd.Timedelta(days=1),
                     "received_at": pd.to_datetime(group.received_at, utc=True).max()})
    return pd.DataFrame(rows, columns=["date", "funding_sum_1h", "event_end", "received_at"])
