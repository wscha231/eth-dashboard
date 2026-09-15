"""Pinned DATA consumption, compatible targets and fail-closed research boundaries.

No collection, external requests, model deserialization, publication or source writes.
The existing DATA feature builder is imported lazily; its definition is not copied.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

HORIZONS = (6, 24, 72, 168, 336, 720)
FLOORS = dict(zip(HORIZONS, (.015, .03, .05, .07, .10, .15)))
TARGET_SPEC = {
    "target_spec_id": "model_lab_hourly_endpoint_v1",
    "compatible_event_version": "hourly_events_v1",
    "source": "coinbase_exchange", "instrument": "ETH-USD", "quote": "USD",
    "bar_timestamp": "open_time; close_time=open_time+1h",
    "reference": "ETH close at input_cutoff=UTC hourly slot",
    "window_start_offset_hours": 1,
    "window_end_offset_hours": "horizon_hours+1",
    "terminal_return": "log(close_at_window_end/reference_price)",
    "direction": "positive / exactly_zero / negative terminal log return",
    "legacy_terminal": "down / inside_barriers / up; not a direction label",
    "barrier": "max(log1p(floor[h]), trailing720h_logreturn_std*sqrt(h))",
    "path": "h bars opening at slot+1h through slot+h; independent high/low hits",
    "variance": "sum of h squared close-to-close log returns inside path window",
    "endpoint_vs_path": "endpoint return spans h+1h; path variance spans h hours",
    "missing": "missing truth stays NaN, including intermediate path bars",
}
RELEASE_FIELDS = {
    "release_id", "schema_version", "dataset_manifest_id", "dataset_manifest_hash",
    "feature_set_id", "feature_set_version", "code_sha", "data_cutoff_utc",
    "generated_at_utc", "source_versions", "coverage_by_feature", "quality_status",
    "vintage_mode", "license_status", "partition_locations", "checksums",
    "exclusions", "breaking_changes",
}


def digest_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def utc(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if pd.isna(t) or t.tzinfo is None:
        raise ValueError("an explicit timezone and finite timestamp are required")
    return t.tz_convert("UTC")


def safe_relative(value: str) -> PurePosixPath:
    p = PurePosixPath(value)
    if (not value or "\\" in value or p.is_absolute() or ".." in p.parts
            or ":" in value or str(p) != value or value == "."):
        raise ValueError("unsafe or noncanonical relative path")
    return p


def check_sha(value: Any) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("expected a lowercase SHA256")


@dataclass(frozen=True)
class PinnedRelease:
    """Canonical bytes, not a mutable dict or a moving latest pointer."""
    raw: bytes
    sha256: str

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> "PinnedRelease":
        check_sha(expected_sha256)
        raw = Path(path).read_bytes()
        if digest_bytes(raw) != expected_sha256:
            raise ValueError("release checksum mismatch")
        d = json.loads(raw)
        missing = RELEASE_FIELDS - d.keys()
        if missing:
            raise ValueError(f"missing release fields: {sorted(missing)}")
        if d["schema_version"] != 1 or d["quality_status"] not in {"pass", "partial"}:
            raise ValueError("unsupported schema or rejected data quality")
        if d["vintage_mode"] not in {"reconstructed", "observed_vintages"}:
            raise ValueError("explicit vintage_mode required")
        for name in ("release_id", "feature_set_id", "feature_set_version", "dataset_manifest_id"):
            if not isinstance(d[name], str) or not d[name].strip() or d[name] == "latest":
                raise ValueError(f"immutable {name} required")
        if not re.fullmatch(r"[0-9a-f]{40}", d["code_sha"]):
            raise ValueError("code_sha must be a full Git commit")
        check_sha(d["dataset_manifest_hash"])
        cutoff, generated = utc(d["data_cutoff_utc"]), utc(d["generated_at_utc"])
        if cutoff > generated:
            raise ValueError("data cutoff later than release generation")
        if not isinstance(d["source_versions"], dict) or not d["source_versions"]:
            raise ValueError("source versions required")
        if not isinstance(d["coverage_by_feature"], dict) or not d["coverage_by_feature"]:
            raise ValueError("coverage report required")
        if not isinstance(d["license_status"], str) or not d["license_status"]:
            raise ValueError("license status must be explicit, including unreviewed")
        if not isinstance(d["exclusions"], list) or not isinstance(d["breaking_changes"], list):
            raise ValueError("explicit exclusions and changes lists required")
        if not d["partition_locations"] or set(d["partition_locations"]) != set(d["checksums"]):
            raise ValueError("every logical partition requires a checksum")
        for key, path_value in d["partition_locations"].items():
            safe_relative(key); safe_relative(path_value); check_sha(d["checksums"][key])
        return cls(canonical(d), expected_sha256)

    @property
    def metadata(self) -> dict:
        return json.loads(self.raw)

    def path(self, root: Path, key: str) -> Path:
        d = self.metadata
        base = Path(root).resolve()
        p = base / safe_relative(d["partition_locations"][key])
        if not p.resolve().is_relative_to(base) or p.is_symlink():
            raise ValueError("partition escaped the restored input directory")
        if not p.is_file() or file_hash(p) != d["checksums"][key]:
            raise ValueError("partition checksum mismatch")
        return p

    def authorize(self, mode: str) -> None:
        if mode not in {"retrospective", "prospective"}:
            raise ValueError("unknown execution mode")
        if mode == "prospective":
            d = self.metadata
            if (d["vintage_mode"] != "observed_vintages" or d["quality_status"] != "pass"
                    or d["license_status"] != "approved_for_intended_use"):
                raise ValueError("reconstruction, partial quality or unapproved rights cannot certify live use")


def load_bars(release: PinnedRelease, root: Path, *, mode: str = "retrospective",
              issued_at: Any = None) -> pd.DataFrame:
    """Consume a restored standalone SQLite snapshot without changing journal state."""
    release.authorize(mode)
    p = release.path(root, "observations.db")
    if any(Path(str(p) + suffix).exists() and Path(str(p) + suffix).stat().st_size
           for suffix in ("-wal", "-journal")):
        raise ValueError("restore a standalone SQLite backup, not a live WAL database")
    with sqlite3.connect(p.resolve().as_uri() + "?mode=ro&immutable=1", uri=True) as con:
        con.execute("PRAGMA trusted_schema=OFF")
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("SQLite integrity failure")
        where, params = "", ()
        if mode == "prospective":
            if issued_at is None:
                raise ValueError("actual issuance time required")
            # Filter BEFORE ranking revisions. A future correction must not hide a past vintage.
            where, params = "WHERE julianday(observed_at)<=julianday(?)", (utc(issued_at).isoformat(),)
        if mode == "prospective":
            # SQLite julian-day arithmetic may round sub-millisecond timestamps.
            # Keep all candidate revisions; apply exact pandas timestamps BEFORE ranking.
            bars = pd.read_sql_query(f"SELECT * FROM bars {where}", con, params=params)
        else:
            bars = pd.read_sql_query("""SELECT * FROM (
                SELECT *, ROW_NUMBER() OVER(PARTITION BY product,open_time ORDER BY revision DESC) rn
                FROM bars) WHERE rn=1 ORDER BY open_time,product""", con).drop(columns="rn")
    if file_hash(p) != release.metadata["checksums"]["observations.db"]:
        raise ValueError("input changed during consumption")
    if bars.empty:
        raise ValueError("no usable source observations")
    for col in ("open_time", "observed_at"):
        if not bars[col].map(lambda x: bool(re.search(r"(Z|[+-]\d{2}:\d{2})$", str(x)))).all():
            raise ValueError("source times must carry explicit timezone offsets")
        bars[col] = pd.to_datetime(bars[col], utc=True, format="mixed").dt.as_unit("ns")
    if mode == "prospective":
        bars = bars.loc[bars.observed_at <= utc(issued_at)].sort_values("revision", ascending=False)
        bars = bars.drop_duplicates(["product", "open_time"]).sort_values(["open_time", "product"])
        if bars.empty:
            raise ValueError("no observations available at actual issuance time")
    bars["close_time"] = bars.open_time + pd.Timedelta(hours=1)
    numeric = bars[["open", "high", "low", "close", "volume"]].to_numpy(float)
    if (not np.isfinite(numeric).all() or (numeric[:, :4] <= 0).any()
            or (numeric[:, 4] < 0).any()):
        raise ValueError("nonfinite, negative-volume or nonpositive-price input")
    if ((bars.high < bars[["open", "close", "low"]].max(axis=1)).any()
            or (bars.low > bars[["open", "close", "high"]].min(axis=1)).any()):
        raise ValueError("invalid OHLC bounds")
    if (not bars["product"].isin(["ETH-USD", "BTC-USD"]).all()
            or bars.duplicated(["product", "close_time"]).any()
            or not bars.open_time.eq(bars.open_time.dt.floor("h")).all()
            or (bars.observed_at < bars.close_time).any()
            or (bars.observed_at > utc(release.metadata["generated_at_utc"])).any()
            or (bars.close_time > utc(release.metadata["data_cutoff_utc"])).any()):
        raise ValueError("wrong instrument, duplicate, misaligned or unavailable bar")
    return bars


def consume_features(bars: pd.DataFrame, release: PinnedRelease) -> pd.DataFrame:
    """Reuse DATA's current builder; register formula changes with DATA, not here."""
    import inspect
    from signal_pipeline.data import build_features
    expected = release.metadata["source_versions"].get("feature_builder_sha256")
    check_sha(expected)
    if digest_bytes(inspect.getsource(build_features).encode()) != expected:
        raise ValueError("DATA feature formula differs from the pinned release")
    features = build_features(bars)
    if features.empty:
        raise ValueError("required DATA feature inputs missing")
    return features


def target_frame(bars: pd.DataFrame, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    if horizon not in HORIZONS:
        raise ValueError("unsupported horizon")
    idx = features.index
    if (not isinstance(idx, pd.DatetimeIndex) or idx.tz is None or idx.has_duplicates
            or not idx.is_monotonic_increasing
            or not idx.equals(pd.date_range(idx.min(), idx.max(), freq="h", tz=idx.tz))):
        raise ValueError("targets require a complete ordered timezone-aware hourly grid")
    eth = bars.loc[bars["product"].eq("ETH-USD")].set_index("close_time").reindex(idx)
    ref = features.reference_price
    barrier = np.maximum(np.log1p(FLOORS[horizon]), features.sigma * np.sqrt(horizon))
    end_close = eth.close.shift(-(horizon + 1))
    high = eth.high.rolling(horizon, min_periods=horizon).max().shift(-(horizon + 1))
    low = eth.low.rolling(horizon, min_periods=horizon).min().shift(-(horizon + 1))
    complete = eth[["open", "high", "low", "close"]].notna().all(axis=1).astype(int)
    complete = complete.rolling(horizon, min_periods=horizon).sum().shift(-(horizon + 1)).eq(horizon)
    ret = np.log(end_close / ref)
    usable = complete & ret.notna() & np.isfinite(barrier) & ref.gt(0)
    out = pd.DataFrame(index=idx)
    out["window_start"] = idx + pd.Timedelta(hours=1)
    out["target_end"] = idx + pd.Timedelta(hours=horizon + 1)
    out["return"] = ret.where(usable)
    out["barrier"] = barrier
    out["up"] = (np.log(high / ref) >= barrier).astype(float).where(usable)
    out["down"] = (np.log(low / ref) <= -barrier).astype(float).where(usable)
    out["terminal"] = pd.Series(np.where(ret > barrier, 2., np.where(ret < -barrier, 0., 1.)), index=idx).where(usable)
    out["positive"] = ret.gt(0).astype(float).where(usable)
    out["negative"] = ret.lt(0).astype(float).where(usable)
    rv = np.log(eth.close).diff().pow(2).rolling(horizon, min_periods=horizon).sum().shift(-(horizon + 1))
    out["realized_variance_total"] = rv.where(usable)
    return out


def purged_indices(features: pd.DataFrame, targets: pd.DataFrame, start: Any,
                   boundary: Any, *, embargo_hours: int = 1, stride_hours: int = 24) -> pd.DatetimeIndex:
    if embargo_hours < 0 or stride_hours < 1 or 24 % stride_hours:
        raise ValueError("invalid purge/stride configuration")
    a, b = utc(start), utc(boundary)
    if a >= b:
        raise ValueError("non-increasing training bounds")
    cols = [c for c in features if c not in {"reference_price", "sigma", "available_at"}]
    ready = features[cols].notna().all(axis=1) & targets["return"].notna()
    ready &= (targets.target_end < b - pd.Timedelta(hours=embargo_hours))
    ready &= (features.index >= a) & (features.index.hour % stride_hours == 0)
    return features.index[ready]


def validate_prediction(record: dict) -> None:
    """Contract check only, not issuance or a promotion certificate."""
    for key in ("release_id", "feature_set_version", "model_version"):
        if not isinstance(record.get(key), str) or not record[key] or record[key] == "latest":
            raise ValueError("immutable output provenance required")
    if record.get("target_spec_id") != TARGET_SPEC["target_spec_id"]:
        raise ValueError("unexpected target specification")
    h = record["horizon_hours"]
    if h not in HORIZONS:
        raise ValueError("unsupported horizon")
    t = utc(record["input_cutoff"]); issue = utc(record["issued_at"])
    start, end = utc(record["window_start"]), utc(record["target_end"])
    if (t != t.floor("h") or not t <= issue < t + pd.Timedelta(minutes=55)
            or start != t + pd.Timedelta(hours=1) or end != start + pd.Timedelta(hours=h)
            or utc(record["available_at"]) > issue or utc(record["training_target_end"]) >= t
            or utc(record["validation_target_end"]) >= t):
        raise ValueError("invalid temporal ordering, stale slot or unmatured label")
    ref = float(record["reference_price"])
    q = np.asarray(record["price_quantiles"], dtype=float)
    direction = np.asarray(record["direction_positive_zero_negative"], dtype=float)
    path = np.asarray(record["path_hit_up_down"], dtype=float)
    if (q.shape != (3,) or direction.shape != (3,) or path.shape != (2,)
            or not np.isfinite(np.r_[ref, q, direction, path]).all()
            or ref <= 0 or np.any(q <= 0) or np.any(np.diff(q) < 0)
            or np.any(direction < 0) or np.any(direction > 1)
            or not np.isclose(direction.sum(), 1., atol=1e-8, rtol=0)
            or np.any(path < 0) or np.any(path > 1)):
        raise ValueError("invalid prices/probabilities/quantiles")
    # Both path events can happen; deliberately do NOT constrain path.sum().
    ppos, pzero, pneg = direction
    if ((q[1] > ref and ppos < .5 - 1e-8)
            or (q[1] < ref and pneg < .5 - 1e-8)):
        raise ValueError("median and endpoint direction distribution are inconsistent")
    if "median_simple_return" in record and not np.isclose(record["median_simple_return"], q[1] / ref - 1, rtol=0, atol=1e-10):
        raise ValueError("price/return conversion mismatch")
    if "variance_total" in record and (not np.isfinite(record["variance_total"]) or record["variance_total"] < 0):
        raise ValueError("invalid total variance")


def nonoverlap_count(origins: pd.DatetimeIndex, ends: pd.Series) -> int:
    selected, last_end = 0, None
    for t in origins.sort_values():
        if last_end is None or t >= last_end:
            selected += 1; last_end = ends.loc[t]
    return selected
