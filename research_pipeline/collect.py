"""Incremental public archives with checksums, complete UTC days and provenance.

Only the documented Binance archive host is used. A denied or missing archive
is recorded; no alternate-host routing and no fabricated observations.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import io
import json
from pathlib import Path
import time
from threading import Event
from urllib.error import HTTPError
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd

from research_pipeline.market_features import REQUIRED_STREAM_IDS, build_market_daily_features
from research_pipeline.protocol import FLOW_COLUMNS

BASE = "https://data.binance.vision/data"
KLINE_COLUMNS = ["open_raw", "open", "high", "low", "close", "volume", "close_raw",
                 "quote_volume", "trade_count", "taker_base", "taker_buy_quote_volume", "ignore"]


def fetch_bytes(url, max_bytes=2_000_000, timeout=12):
    request = Request(url, headers={"User-Agent": "eth-dashboard-forward-research/1.0"})
    with urlopen(request, timeout=timeout) as response:
        payload = response.read(max_bytes + 1)
    if len(payload) > max_bytes:
        raise ValueError("Archive response exceeds size limit")
    return payload


def archive_url(stream, day):
    symbol = "ETHUSDT" if "ethusdt" in stream else "BTCUSDT"
    market = "spot" if "spot" in stream else "futures/um"
    return f"{BASE}/{market}/daily/klines/{symbol}/1h/{symbol}-1h-{day}.zip"


def parse_archive(payload, checksum, filename, day, stream):
    tokens = checksum.strip().split()
    if len(tokens) != 2 or tokens[1].lstrip("*") != filename:
        raise ValueError("Unexpected archive checksum filename")
    if hashlib.sha256(payload).hexdigest() != tokens[0].lower():
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        members = archive.infolist()
        if len(members) != 1 or members[0].file_size > 2_000_000:
            raise ValueError("Unexpected ZIP member count or expanded size")
        raw = pd.read_csv(archive.open(members[0]), header=None, dtype=str)
    if raw.shape[1] != 12:
        raise ValueError("Expected 12 kline fields")
    if str(raw.iloc[0, 0]).lower() in {"open_time", "open time"}:
        raw = raw.iloc[1:].copy()
    raw.columns = KLINE_COLUMNS
    for column in KLINE_COLUMNS:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    unit = "us" if "spot" in stream and pd.Timestamp(day) >= pd.Timestamp("2025-01-01") else "ms"
    raw["open_time"] = pd.to_datetime(raw.open_raw, unit=unit, utc=True).dt.tz_localize(None)
    raw["close_time"] = pd.to_datetime(raw.close_raw, unit=unit, utc=True).dt.tz_localize(None)
    raw = raw.sort_values("open_time").reset_index(drop=True)
    expected = pd.date_range(day, periods=24, freq="h")
    if len(raw) != 24 or not np.array_equal(raw.open_time.to_numpy(), expected.to_numpy()):
        raise ValueError("Archive must contain exactly 24 distinct aligned UTC hours")
    durations = raw.close_time - raw.open_time
    if not ((durations > pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=2)) &
            (durations <= pd.Timedelta(hours=1))).all():
        raise ValueError("Unexpected kline close-time convention")
    numeric = raw[KLINE_COLUMNS].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Non-finite kline")
    if (raw[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive OHLC")
    if ((raw.high < raw[["open", "close", "low"]].max(axis=1)) |
        (raw.low > raw[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC bounds")
    if ((raw.quote_volume <= 0) | (raw.taker_buy_quote_volume < 0) |
        (raw.taker_buy_quote_volume > raw.quote_volume) | (raw.volume < 0) |
        (raw.trade_count < 0)).any():
        raise ValueError("Invalid kline volume")
    raw["stream"] = stream
    return raw.drop(columns=["open_raw", "close_raw", "ignore", "taker_base"])


def _download(stream, day, fetch):
    url = archive_url(stream, day)
    checksum = fetch(url + ".CHECKSUM", max_bytes=4096).decode()
    payload = fetch(url)
    frame = parse_archive(payload, checksum, url.rsplit("/", 1)[-1], day, stream)
    observed = pd.Timestamp.now(tz="UTC").isoformat()
    return frame, {"stream": stream, "source_day": day, "url": url,
                   "sha256": hashlib.sha256(payload).hexdigest(), "observed_at_utc": observed,
                   "rows": len(frame), "status": "verified"}


def read_flow(path):
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if frame.index.has_duplicates:
        raise ValueError("Duplicate flow source days")
    for column, default in (("observed_at_utc", None), ("provenance_kind", "historical_reconstruction")):
        if column not in frame:
            frame[column] = default
    return frame[[*FLOW_COLUMNS, "market_data_excluded", "feature_available_at_utc", "observed_at_utc", "provenance_kind"]].sort_index()


def collect_incremental(root, *, bootstrap, now=None, max_days=45, budget_seconds=480, fetch=fetch_bytes):
    started = time.monotonic()
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    current = pd.Timestamp.now(tz="UTC") if now is None else pd.Timestamp(now)
    current = current.tz_localize("UTC") if current.tzinfo is None else current.tz_convert("UTC")
    last_day = current.tz_localize(None).floor("D") - pd.Timedelta(days=1)
    path = root / "flow_daily.csv"
    old = read_flow(path if path.exists() else bootstrap)
    initial_rows = len(old)
    # A bounded rolling buffer supplies exact seven-calendar-day changes.
    hourly_path = root / "recent_hourly.csv.gz"
    cached = pd.read_csv(hourly_path, parse_dates=["open_time", "close_time"]) if hourly_path.exists() else pd.DataFrame()
    targets = pd.date_range(max(old.index.min(), last_day-pd.Timedelta(days=max_days-8)), last_day)
    valid_old = old.index[np.isfinite(old[FLOW_COLUMNS].to_numpy(float)).all(axis=1)]
    needed = targets.difference(valid_old)
    start = needed.min()-pd.Timedelta(days=7) if len(needed) else last_day+pd.Timedelta(days=1)
    days = pd.date_range(start, last_day, freq="D")
    tasks = []
    for day in days:
        for stream in REQUIRED_STREAM_IDS:
            present = cached.loc[(cached.stream == stream) & (cached.open_time.dt.floor("D") == day)] if not cached.empty else cached
            if len(present) != 24:
                tasks.append((stream, day.date().isoformat()))
    fresh, records, errors = [], [], []
    stopped = Event()
    def bounded_download(stream, day):
        if stopped.is_set() or time.monotonic()-started > budget_seconds:
            stopped.set()
            return None
        try:
            return _download(stream, day, fetch)
        except HTTPError as exc:
            if exc.code in (401, 403, 451):
                stopped.set()  # Do not keep requesting a denied source.
            raise
    with ThreadPoolExecutor(max_workers=4) as pool:
        pending = {pool.submit(bounded_download, stream, day): (stream, day) for stream, day in tasks}
        for future in as_completed(pending):
            stream, day = pending[future]
            try:
                downloaded = future.result()
                if downloaded is None:
                    continue
                frame, record = downloaded
                fresh.append(frame); records.append(record)
            except Exception as exc:
                errors.append({"stream": stream, "source_day": day, "error": type(exc).__name__,
                               "status": "unavailable_or_invalid", "detail": str(exc)[:180]})
    all_frames = ([cached] if not cached.empty else []) + fresh
    latest = old.copy()
    if all_frames:
        hourly = pd.concat(all_frames, ignore_index=True).drop_duplicates(["stream", "open_time"], keep="first")
        streams = {s: hourly.loc[hourly.stream == s].copy() for s in REQUIRED_STREAM_IDS}
        if all(not f.empty for f in streams.values()):
            available = [set(f.open_time.dt.floor("D")) for f in streams.values()]
            complete = set.intersection(*available)
            missing = set(pd.date_range(hourly.open_time.min().floor("D"), last_day)) - complete
            if complete:
                result = build_market_daily_features(streams, cutoff=current, excluded_dates=tuple(sorted(missing)))
                result = result.reindex(columns=[*FLOW_COLUMNS, "market_data_excluded", "feature_available_at_utc"])
                # Reconstructed history is not assigned a fabricated historical retrieval time.
                result["observed_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
                result["provenance_kind"] = "verified_archive_observed_now"
                result = result.loc[result.index.isin(needed) & result["market_data_excluded"].eq(0)
                                    & np.isfinite(result[FLOW_COLUMNS].to_numpy(float)).all(axis=1)]
                old = old.loc[~old.index.isin(result.index)]
                latest = pd.concat([old, result]).sort_index()
                latest = latest.loc[~latest.index.duplicated(keep="first")]
        tail_start = last_day - pd.Timedelta(days=8)
        hourly.loc[hourly.open_time >= tail_start].to_csv(hourly_path, index=False, compression={"method":"gzip", "mtime":0})
    latest.index.name = "date"
    latest.to_csv(path, float_format="%.12g")
    if records:
        with (root / "archive_manifest.jsonl").open("a") as handle:
            for record in sorted(records, key=lambda r:(r["source_day"],r["stream"])):
                handle.write(json.dumps(record, sort_keys=True)+"\n")
    expected = pd.date_range(old.index.min(), last_day)
    missing_dates = expected.difference(latest.index)
    ready = last_day in latest.index and bool(np.isfinite(latest.loc[last_day, FLOW_COLUMNS].to_numpy(float)).all())
    report = {"generated_at": pd.Timestamp.now(tz="UTC").isoformat(), "latest_source_day": str(latest.index.max().date()),
              "expected_source_day": str(last_day.date()), "new_days": len(latest)-initial_rows,
              "stored_days": len(latest), "downloaded_archives": len(records), "request_tasks": len(tasks),
              "missing_day_count": len(missing_dates), "recent_missing_days": [str(d.date()) for d in missing_dates[-10:]],
              "errors": errors[-30:], "error_count": len(errors), "collection_stopped": stopped.is_set(),
              "ready_for_current_origin": ready,
              "runtime_seconds": time.monotonic()-started,
              "publication_note": "bar end and observed download time are separate; archives can arrive late"}
    (root / "source_status.json").write_text(json.dumps(report, indent=2))
    return latest, report
