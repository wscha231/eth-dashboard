"""Prospective HAR-RV variance shadow for R2-passed 6h/24h/72h heads.

This module is deliberately isolated from public forecast issuance. It fits only the
pre-registered R2 HAR-RV specification, issues variance-only shadow records, settles the
same frozen h-bar realized-variance target, and never promotes a model automatically.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_features, read_bars, utc

VERSION = "har_rv_variance_shadow_v1"
HORIZONS = (6, 24, 72)
VOL_COLUMNS = ("eth_vol_24", "eth_vol_168", "eth_vol_720")
POLICY = {
    "version": VERSION,
    "historical_gate": "r2_range_volatility_wave1_20260916",
    "horizons_hours": list(HORIZONS),
    "target_spec": "model_lab_hourly_endpoint_v1:path_realized_variance_h_bars",
    "train_days": 730,
    "calibration_days": 180,
    "embargo_hours": 1,
    "train_origin": "00:00 UTC daily",
    "issue_origin": "00:00 UTC daily; exact match to retrospective origin clock",
    "refit_days": 30,
    "model": "StandardScaler + Ridge(alpha=10) on log(realized_variance_total/h)",
    "calibration": "prior calibration mean actual/predicted variance ratio",
    "baseline": "h * trailing_720h hourly sample variance",
    "prospective_min_nonoverlap": {"6": 120, "24": 60, "72": 40},
    "promotion": "shadow_only; manual review after frozen prospective minima",
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value))
    temporary.replace(path)


def connect(state: Path):
    state = Path(state)
    state.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(state / "issued.db", timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    con.executescript("""
      CREATE TABLE IF NOT EXISTS forecasts (
        forecast_id TEXT PRIMARY KEY, slot TEXT, horizon_hours INTEGER,
        model_version TEXT, issued_at TEXT, target_end TEXT, payload TEXT,
        UNIQUE(slot,horizon_hours,model_version));
      CREATE TABLE IF NOT EXISTS outcomes (
        forecast_id TEXT REFERENCES forecasts(forecast_id), revision INTEGER,
        evaluated_at TEXT, truth_hash TEXT, payload TEXT,
        PRIMARY KEY(forecast_id,revision), UNIQUE(forecast_id,truth_hash));
    """)
    return con


def _target_frame(bars: pd.DataFrame, features: pd.DataFrame, horizon: int) -> pd.DataFrame:
    # Reuse the exact target implementation that R2 Wave 1 gated.
    from model_lab.contracts import target_frame
    return target_frame(bars, features, horizon)


def _fit_indices(features: pd.DataFrame, targets: pd.DataFrame, start, boundary) -> pd.DatetimeIndex:
    start, boundary = utc(start), utc(boundary)
    ready = features[list(VOL_COLUMNS)].notna().all(axis=1)
    ready &= targets["realized_variance_total"].gt(0)
    ready &= features.index.hour == 0
    ready &= features.index >= start
    ready &= targets["target_end"] < boundary - pd.Timedelta(hours=POLICY["embargo_hours"])
    return features.index[ready]


def _source_hash(bars: pd.DataFrame, cutoff=None) -> str:
    frame = bars if cutoff is None else bars.loc[bars["close_time"] <= utc(cutoff)]
    frame = frame.sort_values(["product", "open_time"])
    h = hashlib.sha256()
    for row in frame[["product", "open_time", "content_hash"]].itertuples(index=False):
        h.update(str(row.product).encode())
        h.update(b"|")
        h.update(pd.Timestamp(row.open_time).isoformat().encode())
        h.update(b"|")
        h.update(str(row.content_hash).encode())
        h.update(b"\n")
    return h.hexdigest()


def fit_checkpoint(root: Path, horizon: int, *, now=None) -> dict:
    if horizon not in HORIZONS:
        raise ValueError("unsupported variance-shadow horizon")
    fit_time = utc(now)
    boundary = fit_time.floor("h")
    bars = read_bars(root, as_of=fit_time)
    latest = bars.groupby("product").close_time.max().min()
    if pd.isna(latest) or latest < boundary:
        raise ValueError("source snapshot is stale for variance refit")
    features = build_features(bars)
    targets = _target_frame(bars, features, horizon)
    calibration_start = boundary - pd.Timedelta(days=POLICY["calibration_days"])
    training_start = calibration_start - pd.Timedelta(days=POLICY["train_days"])
    train = _fit_indices(features, targets, training_start, calibration_start)
    calibration = _fit_indices(features, targets, calibration_start, boundary)
    if len(train) < 400 or len(calibration) < 45:
        raise ValueError("insufficient purged HAR-RV train/calibration history")

    vx = np.log(features[list(VOL_COLUMNS)].pow(2).clip(lower=1e-16))
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        vx.loc[train], np.log(targets.loc[train, "realized_variance_total"].to_numpy() / horizon)
    )
    raw_calibration = np.exp(model.predict(vx.loc[calibration]))
    actual_per_hour = targets.loc[calibration, "realized_variance_total"].to_numpy() / horizon
    multiplier = float(np.mean(actual_per_hour / raw_calibration))
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("invalid HAR-RV calibration multiplier")

    checkpoint = {
        "schema": 1,
        "policy": POLICY,
        "horizon_hours": horizon,
        "fit_time": fit_time.isoformat(),
        "fit_boundary": boundary.isoformat(),
        "training_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "training_target_end": targets.loc[train, "target_end"].max().isoformat(),
        "calibration_target_end": targets.loc[calibration, "target_end"].max().isoformat(),
        "source_snapshot": _source_hash(bars, boundary),
        "columns": list(VOL_COLUMNS),
        "scale_mean": model[0].mean_.tolist(),
        "scale_std": model[0].scale_.tolist(),
        "coef": model[-1].coef_.tolist(),
        "intercept": float(model[-1].intercept_),
        "multiplier": multiplier,
    }
    checkpoint["model_version"] = digest(checkpoint)
    return checkpoint


def validate_checkpoint(checkpoint: dict, horizon: int) -> dict:
    if checkpoint.get("schema") != 1 or checkpoint.get("policy") != POLICY:
        raise ValueError("variance checkpoint policy mismatch")
    if checkpoint.get("horizon_hours") != horizon or checkpoint.get("columns") != list(VOL_COLUMNS):
        raise ValueError("variance checkpoint horizon/column mismatch")
    if checkpoint.get("model_version") != digest({k: v for k, v in checkpoint.items() if k != "model_version"}):
        raise ValueError("variance checkpoint identity mismatch")
    fit_boundary = utc(checkpoint["fit_boundary"])
    if utc(checkpoint["training_target_end"]) >= fit_boundary or utc(checkpoint["calibration_target_end"]) >= fit_boundary:
        raise ValueError("variance checkpoint contains unmatured target")
    arrays = np.r_[checkpoint["scale_mean"], checkpoint["scale_std"], checkpoint["coef"],
                   checkpoint["intercept"], checkpoint["multiplier"]]
    if not np.isfinite(arrays).all() or (np.asarray(checkpoint["scale_std"]) <= 0).any() or checkpoint["multiplier"] <= 0:
        raise ValueError("invalid variance checkpoint parameters")
    return checkpoint


def _active_path(state: Path) -> Path:
    return Path(state) / "active.json"


def obtain_checkpoint(root: Path, state: Path, horizon: int, *, now=None) -> dict:
    now = utc(now)
    active_path = _active_path(state)
    active = json.loads(active_path.read_text()) if active_path.exists() else {"schema": 1, "models": {}}
    entry = active.get("models", {}).get(str(horizon))
    checkpoint = None
    if entry:
        path = Path(state) / "models" / entry["file"]
        if path.is_file() and file_hash(path) == entry.get("sha256"):
            checkpoint = validate_checkpoint(json.loads(path.read_text()), horizon)
    if checkpoint and now < utc(checkpoint["fit_time"]) + pd.Timedelta(days=POLICY["refit_days"]):
        return checkpoint

    checkpoint = fit_checkpoint(root, horizon, now=now)
    models = Path(state) / "models"
    models.mkdir(parents=True, exist_ok=True)
    filename = f"h{horizon}_{utc(checkpoint['fit_time']).strftime('%Y%m%dT%H%M%SZ')}_{checkpoint['model_version'][:16]}.json"
    path = models / filename
    if not path.exists():
        atomic_json(path, checkpoint)
    active.setdefault("models", {})[str(horizon)] = {
        "file": filename,
        "sha256": file_hash(path),
        "model_version": checkpoint["model_version"],
        "fit_time": checkpoint["fit_time"],
    }
    active.update(schema=1, policy_version=VERSION, updated_at=now.isoformat())
    atomic_json(active_path, active)
    return checkpoint


def predict_variance(checkpoint: dict, feature_row, horizon: int) -> dict:
    checkpoint = validate_checkpoint(checkpoint, horizon)
    x = np.log(np.asarray([float(feature_row[c]) ** 2 for c in VOL_COLUMNS]).clip(min=1e-16))
    mean = np.asarray(checkpoint["scale_mean"], float)
    scale = np.asarray(checkpoint["scale_std"], float)
    coef = np.asarray(checkpoint["coef"], float)
    log_per_hour = float(checkpoint["intercept"] + np.dot((x - mean) / scale, coef))
    candidate = float(horizon * np.exp(log_per_hour) * checkpoint["multiplier"])
    persistence = float(horizon * float(feature_row["eth_vol_720"]) ** 2)
    if not np.isfinite([candidate, persistence]).all() or candidate <= 0 or persistence <= 0:
        raise ValueError("invalid variance shadow prediction")
    return {
        "predicted_variance_total": candidate,
        "persistence_variance_total": persistence,
        "predicted_endpoint_sigma": float(np.sqrt(candidate * (horizon + 1.0) / horizon)),
        "persistence_endpoint_sigma": float(np.sqrt(persistence * (horizon + 1.0) / horizon)),
    }


def validate_record(record: dict, *, now=None) -> None:
    now = utc(now)
    slot = utc(record["slot"])
    horizon = int(record["horizon_hours"])
    if horizon not in HORIZONS or slot != now.floor("h") or slot.hour != 0:
        raise ValueError("variance shadow issues only the current 00:00 UTC origin")
    if now >= slot + pd.Timedelta(minutes=55):
        raise ValueError("variance shadow issuance deadline passed")
    if utc(record["input_cutoff"]) != slot or utc(record["available_at"]) > now:
        raise ValueError("variance shadow input unavailable at issue time")
    if utc(record["window_start"]) != slot + pd.Timedelta(hours=1):
        raise ValueError("variance path must start one hour after origin")
    if utc(record["target_end"]) != slot + pd.Timedelta(hours=horizon + 1):
        raise ValueError("variance target must preserve h+1 endpoint/path boundary")
    checkpoint = validate_checkpoint(record["checkpoint"], horizon)
    if utc(checkpoint["fit_time"]) > now or utc(checkpoint["calibration_target_end"]) >= utc(checkpoint["fit_boundary"]):
        raise ValueError("variance model was unavailable or used unmatured calibration labels")
    values = [record["predicted_variance_total"], record["persistence_variance_total"],
              record["predicted_endpoint_sigma"], record["persistence_endpoint_sigma"]]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("invalid variance forecast values")


def issue(state: Path, record: dict, *, now=None) -> dict:
    now = utc(now)
    validate_record(record, now=now)
    identity = {"policy": VERSION, "slot": record["slot"], "horizon_hours": record["horizon_hours"],
                "model_version": record["model_version"]}
    forecast_id = digest(identity)
    payload = {**record, "forecast_id": forecast_id, "issued_at": now.isoformat(),
               "role": "variance_shadow_research", "promotion": "none"}
    with connect(state) as con:
        con.execute("BEGIN IMMEDIATE")
        prior = con.execute("SELECT payload FROM forecasts WHERE forecast_id=?", (forecast_id,)).fetchone()
        if prior:
            return json.loads(prior[0])
        con.execute("INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)",
                    (forecast_id, record["slot"], record["horizon_hours"], record["model_version"],
                     now.isoformat(), record["target_end"], json.dumps(payload, sort_keys=True, allow_nan=False)))
    return payload


def _truth_hash(bars: pd.DataFrame, slot, target_end) -> tuple[str, str] | tuple[None, None]:
    slot, target_end = utc(slot), utc(target_end)
    eth = bars.loc[bars["product"].eq("ETH-USD") & (bars["close_time"] > slot) & (bars["close_time"] <= target_end)]
    if len(eth) == 0 or eth["content_hash"].isna().any():
        return None, None
    return digest(eth.sort_values("close_time")["content_hash"].tolist()), eth.observed_at.max().isoformat()


def settle(state: Path, bars: pd.DataFrame, *, now=None) -> int:
    now = utc(now)
    features = build_features(bars)
    targets = {h: _target_frame(bars, features, h) for h in HORIZONS}
    settled = 0
    with connect(state) as con:
        rows = con.execute("SELECT forecast_id,payload FROM forecasts WHERE target_end<=?", (now.isoformat(),)).fetchall()
        for forecast_id, raw in rows:
            forecast = json.loads(raw)
            horizon = int(forecast["horizon_hours"])
            slot = utc(forecast["slot"])
            if slot not in targets[horizon].index:
                continue
            actual = targets[horizon].loc[slot, "realized_variance_total"]
            if pd.isna(actual) or not np.isfinite(actual) or actual <= 0:
                continue
            truth_hash, truth_available_at = _truth_hash(bars, slot, forecast["target_end"])
            if not truth_hash or utc(truth_available_at) > now:
                continue
            prior = con.execute("SELECT 1 FROM outcomes WHERE forecast_id=? AND truth_hash=?", (forecast_id, truth_hash)).fetchone()
            if prior:
                continue
            candidate = float(forecast["predicted_variance_total"])
            baseline = float(forecast["persistence_variance_total"])
            candidate_ratio = float(actual / candidate)
            baseline_ratio = float(actual / baseline)
            outcome = {
                "realized_variance_total": float(actual),
                "candidate_qlike": float(candidate_ratio - np.log(candidate_ratio) - 1.0),
                "persistence_qlike": float(baseline_ratio - np.log(baseline_ratio) - 1.0),
                "truth_available_at": truth_available_at,
            }
            revision = con.execute("SELECT COALESCE(MAX(revision),0)+1 FROM outcomes WHERE forecast_id=?", (forecast_id,)).fetchone()[0]
            con.execute("INSERT INTO outcomes VALUES (?,?,?,?,?)",
                        (forecast_id, revision, now.isoformat(), truth_hash,
                         json.dumps(outcome, sort_keys=True, allow_nan=False)))
            settled += 1
    return settled


def history(state: Path) -> list[dict]:
    with connect(state) as con:
        rows = con.execute("""SELECT f.payload,o.payload,o.revision,o.evaluated_at
            FROM forecasts f LEFT JOIN outcomes o
            ON f.forecast_id=o.forecast_id AND o.revision=(SELECT MAX(revision) FROM outcomes WHERE forecast_id=f.forecast_id)
            ORDER BY f.issued_at,f.horizon_hours""").fetchall()
    return [{**json.loads(f), "outcome": json.loads(o) if o else None,
             "truth_revision": revision, "outcome_evaluated_at": evaluated}
            for f, o, revision, evaluated in rows]


def prospective_report(records: list[dict]) -> dict:
    report = {}
    for horizon in HORIZONS:
        issued = sorted([r for r in records if r["horizon_hours"] == horizon], key=lambda r: utc(r["slot"]))
        resolved = [r for r in issued if r.get("outcome")]
        independent = []
        previous_end = None
        for row in resolved:
            if previous_end is None or utc(row["slot"]) >= previous_end:
                independent.append(row)
                previous_end = utc(row["target_end"])

        def score(rows):
            if not rows:
                return {"rows": 0, "candidate_qlike": None, "persistence_qlike": None, "qlike_improvement": None}
            candidate = float(np.mean([r["outcome"]["candidate_qlike"] for r in rows]))
            baseline = float(np.mean([r["outcome"]["persistence_qlike"] for r in rows]))
            return {"rows": len(rows), "candidate_qlike": candidate, "persistence_qlike": baseline,
                    "qlike_improvement": float(1.0 - candidate / baseline) if baseline > 0 else None}

        minimum = POLICY["prospective_min_nonoverlap"][str(horizon)]
        report[str(horizon)] = {
            "issued": len(issued),
            "resolved": len(resolved),
            "pending": len(issued) - len(resolved),
            "all_resolved": score(resolved),
            "nonoverlap": score(independent),
            "minimum_nonoverlap_for_review": minimum,
            "performance_watch": "manual_review_required" if len(independent) >= minimum else "insufficient_evidence",
            "promotion": "shadow_only",
        }
    return report


def run(root: Path, *, now=None) -> dict:
    root = Path(root)
    state = root / "variance_shadow"
    current = utc(now)
    slot = current.floor("h")
    bars = read_bars(root, as_of=current)
    settled = settle(state, bars, now=current)
    features = build_features(bars)
    issued = []
    errors = []

    ready = (not features.empty and slot in features.index
             and features.loc[slot, list(VOL_COLUMNS)].notna().all()
             and features.loc[slot, "available_at"] <= current)
    for horizon in HORIZONS:
        try:
            checkpoint = obtain_checkpoint(root, state, horizon, now=current)
            if not ready or slot.hour != 0:
                continue
            values = predict_variance(checkpoint, features.loc[slot], horizon)
            record = {
                "schema": 1,
                "policy": POLICY,
                "slot": slot.isoformat(),
                "input_cutoff": slot.isoformat(),
                "available_at": features.loc[slot, "available_at"].isoformat(),
                "window_start": (slot + pd.Timedelta(hours=1)).isoformat(),
                "target_end": (slot + pd.Timedelta(hours=horizon + 1)).isoformat(),
                "horizon_hours": horizon,
                "instrument": "ETH-USD",
                "reference_price": float(features.loc[slot, "reference_price"]),
                "model_version": checkpoint["model_version"],
                "checkpoint": checkpoint,
                "input_snapshot": _source_hash(bars, slot),
                **values,
            }
            issued.append(issue(state, record, now=current))
        except (ValueError, KeyError, OSError) as exc:
            errors.append({"horizon_hours": horizon, "reason": str(exc)[:300]})

    records = history(state)
    report = {
        "schema": 1,
        "policy": POLICY,
        "generated_at": current.isoformat(),
        "source_as_of": bars.groupby("product").close_time.max().min().isoformat(),
        "settled_this_run": settled,
        "issued_this_run": [{k: r[k] for k in ("forecast_id", "slot", "horizon_hours", "model_version", "target_end")}
                            for r in issued],
        "errors": errors,
        "prospective": prospective_report(records),
        "claims": "prospective variance shadow only; historical gate passed but production promotion remains disabled",
    }
    atomic_json(state / "report.json", report)
    return report
