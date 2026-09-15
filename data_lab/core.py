"""Fail-closed DATA contracts; preserve timestamps, nulls and source revisions."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
import hashlib
import json
import math
import re
import sqlite3
import stat
import zipfile

import numpy as np
import pandas as pd


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def check_hash(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected lowercase SHA256")


def utc(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("finite explicit timezone required")
    return t.tz_convert("UTC").as_unit("ns")


def relative(value: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("path must be a string")
    p = PurePosixPath(value)
    if (not value or value == "." or "\\" in value or ":" in value or "\x00" in value
            or p.is_absolute() or ".." in p.parts or str(p) != value):
        raise ValueError("unsafe or noncanonical relative path")
    return p


def safe_file(root: Path, name: str) -> Path:
    base = Path(root).resolve()
    p = base / relative(name)
    if not p.resolve().is_relative_to(base):
        raise ValueError("path escaped root")
    current = base
    for part in relative(name).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("symlinks are not release partitions")
    if not p.is_file():
        raise ValueError("missing regular partition")
    return p


def read_zip(path: Path, expected_hash: str, *, max_bytes: int = 512_000_000,
             max_files: int = 2000) -> dict[str, bytes]:
    """Bounded ZIP read without extraction, model deserialization or path execution."""
    check_hash(expected_hash)
    if sha256(path) != expected_hash:
        raise ValueError("archive checksum mismatch")
    output: dict[str, bytes] = {}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
        if len(infos) > max_files or sum(i.file_size for i in infos) > max_bytes:
            raise ValueError("archive exceeds size/file budget")
        for i in infos:
            if i.is_dir():
                relative(i.filename.rstrip("/"))
                continue
            relative(i.filename)
            mode = i.external_attr >> 16
            if i.flag_bits & 1 or stat.S_ISLNK(mode):
                raise ValueError("encrypted or symlink archive member")
            if i.filename in output:
                raise ValueError("duplicate archive member")
            raw = z.read(i)
            if len(raw) != i.file_size:
                raise ValueError("archive member length mismatch")
            output[i.filename] = raw
    if sha256(path) != expected_hash:
        raise ValueError("archive changed during read")
    return output


def validate_handoff(path: Path, expected_hash: str) -> tuple[dict, dict[str, bytes]]:
    files = read_zip(path, expected_hash)
    if "HANDOFF_MANIFEST.json" not in files:
        raise ValueError("original handoff manifest required")
    manifest = json.loads(files["HANDOFF_MANIFEST.json"])
    entries = manifest["files"]
    if set(files) != set(entries) | {"HANDOFF_MANIFEST.json"}:
        raise ValueError("handoff member set differs from original manifest")
    for name, spec in entries.items():
        if (len(files[name]) != spec["bytes"] or
                hashlib.sha256(files[name]).hexdigest() != spec["sha256"]):
            raise ValueError("handoff content differs from original manifest")
    registry = json.loads(files["research_20260915/feature_registry.json"])
    if registry["count"] != len(registry["features"]):
        raise ValueError("research registry count mismatch")
    ids = [f["id"] for f in registry["features"]]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate research feature id")
    return manifest, files


def missing_runs(index: pd.DatetimeIndex, expected: pd.DatetimeIndex) -> list[dict]:
    """Exact grid gaps, not a claim about provider history or absence of trades."""
    missing = ~expected.isin(index)
    changes = np.diff(np.r_[False, missing, False].astype(int))
    return [{"start": expected[a].isoformat(), "end": expected[b-1].isoformat(),
             "rows": int(b-a), "reason": "not_in_pinned_snapshot_source_reason_unknown"}
            for a, b in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1))]


def read_bars(path: Path, expected_hash: str, *, cutoff: Any, generated: Any) -> tuple[pd.DataFrame, dict]:
    """Validate standalone Coinbase snapshot; never mutate original SQLite state."""
    check_hash(expected_hash)
    path = Path(path)
    if path.is_symlink() or sha256(path) != expected_hash:
        raise ValueError("source database hash/symlink rejection")
    for suffix in ("-wal", "-journal"):
        sidecar = Path(str(path) + suffix)
        if sidecar.exists() and sidecar.stat().st_size:
            raise ValueError("standalone SQLite backup required")
    cutoff, generated = utc(cutoff), utc(generated)
    if cutoff > generated:
        raise ValueError("cutoff after generation")
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as con:
        con.execute("PRAGMA trusted_schema=OFF")
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity failure")
        all_rows = pd.read_sql_query("SELECT * FROM bars", con)
    if all_rows.empty:
        raise ValueError("empty source database")
    for col in ("open_time", "observed_at"):
        all_rows[col] = pd.DatetimeIndex([utc(x) for x in all_rows[col]])
    all_rows["close_time"] = all_rows.open_time + pd.Timedelta(hours=1)
    rev = pd.to_numeric(all_rows.revision, errors="raise")
    if ((rev < 1).any() or not np.isfinite(rev).all() or (rev % 1 != 0).any()
            or all_rows.duplicated(["product", "open_time", "revision"]).any()):
        raise ValueError("invalid source revision")
    numeric = all_rows[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any()
            or (numeric[:, 4] < 0).any()
            or (all_rows.high < all_rows[["open", "close", "low"]].max(axis=1)).any()
            or (all_rows.low > all_rows[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("invalid source OHLCV")
    if (not set(all_rows["product"]) == {"ETH-USD", "BTC-USD"}
            or not all_rows.open_time.eq(all_rows.open_time.dt.floor("h")).all()
            or (all_rows.observed_at < all_rows.close_time).any()
            or (all_rows.close_time > cutoff).any()
            or (all_rows.observed_at > generated).any()):
        raise ValueError("source time/instrument boundary violation")
    bars = (all_rows.sort_values("revision").drop_duplicates(["product", "open_time"], keep="last")
            .sort_values(["close_time", "product"]).reset_index(drop=True))
    grid = pd.date_range(bars.open_time.min(), bars.open_time.max(), freq="h")
    report = {"sqlite_integrity": "ok", "all_revision_rows": len(all_rows),
              "unique_rows": len(bars), "products": {},
              "vintage_mode": "reconstructed", "latency_interpretation":
              "Receipt minus close includes historical backfill; not live API latency or SLA."}
    for product, g in bars.groupby("product"):
        delays = (g.observed_at - g.close_time).dt.total_seconds()
        gaps = missing_runs(pd.DatetimeIndex(g.open_time), grid)
        report["products"][product] = {
            "rows": len(g), "expected_grid_rows": len(grid),
            "first_open": g.open_time.min().isoformat(), "last_open": g.open_time.max().isoformat(),
            "first_receipt": g.observed_at.min().isoformat(), "last_receipt": g.observed_at.max().isoformat(),
            "missing_rows": sum(x["rows"] for x in gaps),
            "longest_gap_hours": max((x["rows"] for x in gaps), default=0), "gaps": gaps,
            "receipt_delay_seconds": {str(q): float(delays.quantile(q)) for q in (0, .5, .95, 1)},
            "receipts_within_one_hour_of_close": int((delays <= 3600).sum()),
            "live_latency_sla": "not_established", "source_unit": "USD prices; base-asset volume"}
    if sha256(path) != expected_hash:
        raise ValueError("source changed during audit")
    return bars, report


@dataclass(frozen=True)
class MetricSpec:
    series_id: str
    provider: str
    venue: str
    instrument: str
    unit: str
    max_age_seconds: int

    def __post_init__(self) -> None:
        if any(not isinstance(v, str) or not v.strip() for v in
               (self.series_id, self.provider, self.venue, self.instrument, self.unit)):
            raise ValueError("explicit series identity and unit required")
        if isinstance(self.max_age_seconds, bool) or not isinstance(self.max_age_seconds, int) or self.max_age_seconds <= 0:
            raise ValueError("positive integer max age required")


REQUIRED_OBSERVATION = {"series_id", "provider", "venue", "instrument", "unit", "event_time",
    "published_at", "available_at", "ingested_at", "revision", "value", "missing_reason", "kind",
    "vintage_mode", "license_status", "raw_sha256"}


def normalize_observation(record: dict, spec: MetricSpec) -> dict:
    """Use actual receipts. Missing publication time stays unknown, never invented."""
    if not REQUIRED_OBSERVATION <= record.keys():
        raise ValueError("missing observation contract fields")
    r = {key: record[key] for key in REQUIRED_OBSERVATION}
    for key in ("series_id", "provider", "venue", "instrument", "unit"):
        if r[key] != getattr(spec, key):
            raise ValueError("series identity/unit mismatch")
    times = {k: utc(r[k]) for k in ("event_time", "available_at", "ingested_at")}
    pub = None if r["published_at"] is None else utc(r["published_at"])
    if (times["event_time"] > times["available_at"] or times["available_at"] > times["ingested_at"]
            or (pub is not None and (pub < times["event_time"] or pub > times["available_at"]))):
        raise ValueError("invalid observation timestamp order")
    if (isinstance(r["revision"], bool) or not isinstance(r["revision"], int) or r["revision"] < 1):
        raise ValueError("positive integer revision required")
    if r["kind"] not in {"observed", "derived", "estimated", "synthetic"}:
        raise ValueError("unknown data kind")
    if r["vintage_mode"] not in {"observed_vintages", "reconstructed"} or not r["license_status"]:
        raise ValueError("vintage and rights status required")
    if r["value"] is None:
        if not isinstance(r["missing_reason"], str) or not r["missing_reason"]:
            raise ValueError("missing value requires reason")
    elif (isinstance(r["value"], bool) or not isinstance(r["value"], (int, float))
          or not math.isfinite(r["value"]) or r["missing_reason"] is not None):
        raise ValueError("finite value or explicit null required")
    check_hash(r["raw_sha256"])
    for key, value in times.items():
        r[key] = value.isoformat()
    r["published_at"] = pub.isoformat() if pub is not None else None
    r["eligible_at"] = times["ingested_at"].isoformat()
    r["availability_basis"] = "publication_and_receipt" if pub is not None else "receipt_only_publication_unknown"
    return r


def point_in_time_join(records: Iterable[dict], decisions: Iterable[Any], spec: MetricSpec,
                       *, require_approved_rights: bool = True) -> pd.DataFrame:
    """One source series, latest EVENT then latest eligible REVISION at each cutoff.

    An old observation revised today must not replace a newer observation. Future
    revisions are filtered BEFORE selection. A null latest value is not backfilled.
    This strict reader never consumes synthetic/estimated or reconstructed records.
    """
    rows = [normalize_observation(r, spec) for r in records]
    seen = set()
    for r in rows:
        key = (r["event_time"], r["revision"])
        if key in seen:
            raise ValueError("duplicate series event/revision")
        seen.add(key)
    decision_times = [utc(x) for x in decisions]
    # Sweep in receipt order. O((N+M)log(N+M)), not an N*M scan of long histories.
    arrivals = sorted(rows, key=lambda r: utc(r["eligible_at"]))
    result = []; cursor = 0; r = None; selected_key = None
    for position in sorted(range(len(decision_times)), key=lambda i: decision_times[i]):
        t = decision_times[position]
        while cursor < len(arrivals) and utc(arrivals[cursor]["eligible_at"]) <= t:
            candidate = arrivals[cursor]
            key = (utc(candidate["event_time"]), candidate["revision"])
            if selected_key is None or key > selected_key:
                selected_key = key; r = candidate
            cursor += 1
        reason = "not_yet_available"; value = None; age = None
        if r is not None:
            age = float((t - utc(r["event_time"])).total_seconds())
            if r["kind"] not in {"observed", "derived"}:
                reason = "non_observed_kind_blocked"
            elif r["vintage_mode"] != "observed_vintages":
                reason = "historical_reconstruction_blocked"
            elif require_approved_rights and r["license_status"] != "approved_for_intended_use":
                reason = "rights_unapproved"
            elif age > spec.max_age_seconds:
                reason = "stale_observation"
            elif r["value"] is None:
                reason = r["missing_reason"]
            else:
                value = r["value"]; reason = None
        result.append({"_position": position, "decision_at": t, "value": value, "missing_reason": reason,
                       "age_seconds": age, "source_event_time": r["event_time"] if r else None,
                       "source_revision": r["revision"] if r else None,
                       "eligible_at": r["eligible_at"] if r else None})
    result.sort(key=lambda x: x.pop("_position"))
    return pd.DataFrame(result, columns=["decision_at", "value", "missing_reason", "age_seconds",
                                         "source_event_time", "source_revision", "eligible_at"])


class ObservationStore:
    """Append-only normalized input journal, not a second Drive storage engine."""
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as con:
            con.execute("CREATE TABLE IF NOT EXISTS observations (series_id TEXT, event_time TEXT, "
                        "revision INTEGER, payload TEXT NOT NULL, PRIMARY KEY(series_id,event_time,revision))")

    def append(self, records: Iterable[dict], spec: MetricSpec) -> dict:
        rows = [normalize_observation(r, spec) for r in records]
        added = reused = 0
        with sqlite3.connect(self.path) as con:
            con.execute("BEGIN IMMEDIATE")
            for r in rows:
                key = (r["series_id"], r["event_time"], r["revision"])
                payload = canonical(r).decode()
                prior = con.execute("SELECT payload FROM observations WHERE series_id=? AND event_time=? AND revision=?", key).fetchone()
                if prior:
                    if prior[0] != payload:
                        raise ValueError("conflicting revision; original remains immutable")
                    reused += 1
                else:
                    con.execute("INSERT INTO observations VALUES (?,?,?,?)", (*key, payload))
                    added += 1
        return {"added": added, "reused": reused}

    def read(self, spec: MetricSpec) -> list[dict]:
        with sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True) as con:
            return [json.loads(r[0]) for r in con.execute(
                "SELECT payload FROM observations WHERE series_id=? ORDER BY event_time,revision", (spec.series_id,))]
