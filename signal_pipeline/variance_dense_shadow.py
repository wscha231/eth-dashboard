"""Dense prospective HAR-RV variance shadow for clock-robust 24h/72h horizons.

This cohort is intentionally separate from the legacy 00:00 UTC variance ledger.
The frozen all-clock audit (run 35242529360) authorized denser prospective shadow
issuance only for 24h and 72h. 6h failed the all-clock gate and is excluded.

The dense cohort never refits a model. It reuses an already-valid checkpoint from
the original HAR-RV variance shadow and fails closed when that checkpoint is absent
or stale. No late/reconstructed issue is allowed and no production promotion is
automatic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import build_features, read_bars, utc
from . import variance_shadow as base

VERSION = "har_rv_dense_shadow_24h72h_v1"
HORIZONS = (24, 72)
AUDIT_RUN = 35242529360
AUDIT_ARTIFACT = 10505882629
AUDIT_SHA256 = "cca0180479036aeaa3a7a398d134401a0ecd40808e29855ded21020a6e6d5cf3"
POLICY = {
    "version": VERSION,
    "authorization": "har_rv_clock_robustness_20260918",
    "authorization_run": AUDIT_RUN,
    "authorization_artifact": AUDIT_ARTIFACT,
    "authorization_sha256": AUDIT_SHA256,
    "horizons_hours": list(HORIZONS),
    "excluded_horizons": {"6": "all-clock gate failed; UTC 08-16 failed"},
    "checkpoint_source": "legacy 00UTC HAR-RV variance shadow; reuse only, no dense refit",
    "target_spec": base.POLICY["target_spec"],
    "issue_origin": "every current closed UTC hour with source-ready feature row",
    "issue_deadline_minutes_after_origin": 55,
    "late_or_reconstructed_issue_allowed": False,
    "prospective_min_nonoverlap": {"24": 60, "72": 40},
    "review_cohort": "dense ledger only; never combine with legacy 00UTC ledger without a new preregistration",
    "promotion": "shadow_only; manual review after frozen prospective minima",
    "public_claim": "research volatility scale only; not a validated target price or predictive interval",
}


def atomic_json(path: Path, value) -> None:
    base.atomic_json(path, value)


def _state(root: Path) -> Path:
    return Path(root) / "variance_shadow_dense"


def _base_state(root: Path) -> Path:
    return Path(root) / "variance_shadow"


def load_base_checkpoint(root: Path, horizon: int, *, now=None) -> dict:
    if horizon not in HORIZONS:
        raise ValueError("dense variance shadow supports only clock-authorized 24h/72h horizons")
    now = utc(now)
    state = _base_state(root)
    active_path = state / "active.json"
    if not active_path.is_file():
        raise ValueError("legacy HAR-RV active checkpoint manifest is unavailable")
    active = json.loads(active_path.read_text())
    entry = active.get("models", {}).get(str(horizon))
    if not entry:
        raise ValueError(f"legacy HAR-RV checkpoint missing for {horizon}h")
    path = state / "models" / entry["file"]
    if not path.is_file() or base.file_hash(path) != entry.get("sha256"):
        raise ValueError(f"legacy HAR-RV checkpoint integrity failure for {horizon}h")
    checkpoint = base.validate_checkpoint(json.loads(path.read_text()), horizon)
    if utc(checkpoint["fit_time"]) > now:
        raise ValueError("legacy HAR-RV checkpoint was not yet available")
    if now >= utc(checkpoint["fit_time"]) + pd.Timedelta(days=base.POLICY["refit_days"]):
        raise ValueError("legacy HAR-RV checkpoint is stale; wait for the 00UTC refit path")
    return checkpoint


def validate_record(record: dict, *, now=None) -> None:
    now = utc(now)
    slot = utc(record["slot"])
    horizon = int(record["horizon_hours"])
    if horizon not in HORIZONS:
        raise ValueError("dense variance horizon was not authorized by the all-clock audit")
    if slot != now.floor("h"):
        raise ValueError("dense variance shadow can issue only the current UTC hourly origin")
    if now >= slot + pd.Timedelta(minutes=POLICY["issue_deadline_minutes_after_origin"]):
        raise ValueError("dense variance issuance deadline passed; late backfill is forbidden")
    if utc(record["input_cutoff"]) != slot:
        raise ValueError("dense variance input cutoff must equal the current origin")
    if utc(record["available_at"]) > now:
        raise ValueError("dense variance input was unavailable at issue time")
    if utc(record["window_start"]) != slot + pd.Timedelta(hours=1):
        raise ValueError("dense variance path must start one hour after origin")
    if utc(record["target_end"]) != slot + pd.Timedelta(hours=horizon + 1):
        raise ValueError("dense variance target must preserve the frozen h+1 endpoint/path boundary")
    checkpoint = base.validate_checkpoint(record["checkpoint"], horizon)
    if utc(checkpoint["fit_time"]) > now:
        raise ValueError("dense variance checkpoint was unavailable at issue time")
    values = [
        record["predicted_variance_total"],
        record["persistence_variance_total"],
        record["predicted_endpoint_sigma"],
        record["persistence_endpoint_sigma"],
    ]
    if not np.isfinite(values).all() or min(values) <= 0:
        raise ValueError("invalid dense variance forecast values")


def issue(state: Path, record: dict, *, now=None) -> tuple[dict, bool]:
    now = utc(now)
    validate_record(record, now=now)
    identity = {
        "policy": VERSION,
        "slot": record["slot"],
        "horizon_hours": record["horizon_hours"],
        "model_version": record["model_version"],
    }
    forecast_id = base.digest(identity)
    payload = {
        **record,
        "forecast_id": forecast_id,
        "issued_at": now.isoformat(),
        "role": "variance_shadow_dense_research",
        "cohort": VERSION,
        "promotion": "none",
    }
    with base.connect(state) as con:
        con.execute("BEGIN IMMEDIATE")
        prior = con.execute(
            "SELECT payload FROM forecasts WHERE slot=? AND horizon_hours=? ORDER BY issued_at LIMIT 1",
            (record["slot"], record["horizon_hours"]),
        ).fetchone()
        if prior:
            return json.loads(prior[0]), False
        con.execute(
            "INSERT INTO forecasts VALUES (?,?,?,?,?,?,?)",
            (
                forecast_id,
                record["slot"],
                record["horizon_hours"],
                record["model_version"],
                now.isoformat(),
                record["target_end"],
                json.dumps(payload, sort_keys=True, allow_nan=False),
            ),
        )
    return payload, True


def settle(state: Path, bars: pd.DataFrame, *, now=None) -> int:
    now = utc(now)
    features = build_features(bars)
    targets = {h: base._target_frame(bars, features, h) for h in HORIZONS}
    settled = 0
    with base.connect(state) as con:
        rows = con.execute(
            "SELECT forecast_id,payload FROM forecasts WHERE target_end<=?", (now.isoformat(),)
        ).fetchall()
        for forecast_id, raw in rows:
            forecast = json.loads(raw)
            horizon = int(forecast["horizon_hours"])
            if horizon not in targets:
                continue
            slot = utc(forecast["slot"])
            if slot not in targets[horizon].index:
                continue
            actual = targets[horizon].loc[slot, "realized_variance_total"]
            if pd.isna(actual) or not np.isfinite(actual) or actual <= 0:
                continue
            truth_hash, truth_available_at = base._truth_hash(bars, slot, forecast["target_end"])
            if not truth_hash or utc(truth_available_at) > now:
                continue
            prior = con.execute(
                "SELECT 1 FROM outcomes WHERE forecast_id=? AND truth_hash=?", (forecast_id, truth_hash)
            ).fetchone()
            if prior:
                continue
            candidate = float(forecast["predicted_variance_total"])
            persistence = float(forecast["persistence_variance_total"])
            candidate_ratio = float(actual / candidate)
            persistence_ratio = float(actual / persistence)
            outcome = {
                "realized_variance_total": float(actual),
                "candidate_qlike": float(candidate_ratio - np.log(candidate_ratio) - 1.0),
                "persistence_qlike": float(persistence_ratio - np.log(persistence_ratio) - 1.0),
                "truth_available_at": truth_available_at,
            }
            revision = con.execute(
                "SELECT COALESCE(MAX(revision),0)+1 FROM outcomes WHERE forecast_id=?", (forecast_id,)
            ).fetchone()[0]
            con.execute(
                "INSERT INTO outcomes VALUES (?,?,?,?,?)",
                (
                    forecast_id,
                    revision,
                    now.isoformat(),
                    truth_hash,
                    json.dumps(outcome, sort_keys=True, allow_nan=False),
                ),
            )
            settled += 1
    return settled


def history(state: Path) -> list[dict]:
    with base.connect(state) as con:
        rows = con.execute(
            """SELECT f.payload,o.payload,o.revision,o.evaluated_at
            FROM forecasts f LEFT JOIN outcomes o
            ON f.forecast_id=o.forecast_id
            AND o.revision=(SELECT MAX(revision) FROM outcomes WHERE forecast_id=f.forecast_id)
            ORDER BY f.issued_at,f.horizon_hours"""
        ).fetchall()
    return [
        {
            **json.loads(forecast),
            "outcome": json.loads(outcome) if outcome else None,
            "truth_revision": revision,
            "outcome_evaluated_at": evaluated_at,
        }
        for forecast, outcome, revision, evaluated_at in rows
    ]


def _score(rows: list[dict]) -> dict:
    if not rows:
        return {"rows": 0, "candidate_qlike": None, "persistence_qlike": None, "qlike_improvement": None}
    candidate = float(np.mean([row["outcome"]["candidate_qlike"] for row in rows]))
    persistence = float(np.mean([row["outcome"]["persistence_qlike"] for row in rows]))
    return {
        "rows": len(rows),
        "candidate_qlike": candidate,
        "persistence_qlike": persistence,
        "qlike_improvement": float(1.0 - candidate / persistence) if persistence > 0 else None,
    }


def prospective_report(records: list[dict]) -> dict:
    report = {}
    for horizon in HORIZONS:
        issued = sorted(
            [row for row in records if int(row["horizon_hours"]) == horizon],
            key=lambda row: (utc(row["slot"]), utc(row["issued_at"])),
        )
        unique = {}
        for row in issued:
            unique.setdefault(row["slot"], row)
        issued = list(unique.values())
        resolved = [row for row in issued if row.get("outcome")]
        independent = []
        previous_end = None
        for row in resolved:
            if previous_end is None or utc(row["slot"]) >= previous_end:
                independent.append(row)
                previous_end = utc(row["target_end"])
        minimum = POLICY["prospective_min_nonoverlap"][str(horizon)]
        report[str(horizon)] = {
            "issued": len(issued),
            "resolved": len(resolved),
            "pending": len(issued) - len(resolved),
            "all_resolved": _score(resolved),
            "nonoverlap": _score(independent),
            "minimum_nonoverlap_for_review": minimum,
            "performance_watch": "manual_review_required" if len(independent) >= minimum else "insufficient_evidence",
            "promotion": "shadow_only",
        }
    return report


def public_payload(records: list[dict], report: dict) -> dict:
    horizons = {}
    for horizon in HORIZONS:
        rows = sorted(
            [row for row in records if int(row["horizon_hours"]) == horizon],
            key=lambda row: utc(row["slot"]),
        )
        latest = rows[-1] if rows else None
        row = {
            "horizon_hours": horizon,
            "clock_robustness_gate": "pass_all_24_clocks",
            "authorization_run": AUDIT_RUN,
            "prospective": report["prospective"][str(horizon)],
            "status": "dense_shadow_only",
        }
        if latest:
            row.update(
                forecast_id=latest["forecast_id"],
                slot=latest["slot"],
                issued_at=latest["issued_at"],
                target_end=latest["target_end"],
                reference_price=float(latest["reference_price"]),
                predicted_endpoint_sigma=float(latest["predicted_endpoint_sigma"]),
                persistence_endpoint_sigma=float(latest["persistence_endpoint_sigma"]),
                model_version=latest["model_version"],
            )
        horizons[str(horizon)] = row
    return {
        "schema": 1,
        "version": VERSION,
        "generated_at": report["generated_at"],
        "source_as_of": report["source_as_of"],
        "status": "research_dense_shadow",
        "claim": POLICY["public_claim"],
        "horizons": horizons,
    }


def run(root: Path, *, now=None) -> dict:
    root = Path(root)
    state = _state(root)
    current = utc(now)
    slot = current.floor("h")
    bars = read_bars(root, as_of=current)
    settled = settle(state, bars, now=current)
    features = build_features(bars)
    source_as_of = bars.groupby("product").close_time.max().min()

    deadline_ok = current < slot + pd.Timedelta(minutes=POLICY["issue_deadline_minutes_after_origin"])
    feature_ready = (
        not features.empty
        and slot in features.index
        and features.loc[slot, list(base.VOL_COLUMNS)].notna().all()
        and features.loc[slot, "available_at"] <= current
    )
    ready = bool(deadline_ok and feature_ready)
    issued = []
    duplicate = []
    errors = []

    if ready:
        for horizon in HORIZONS:
            try:
                checkpoint = load_base_checkpoint(root, horizon, now=current)
                values = base.predict_variance(checkpoint, features.loc[slot], horizon)
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
                    "input_snapshot": base._source_hash(bars, slot),
                    **values,
                }
                payload, created = issue(state, record, now=current)
                summary = {key: payload[key] for key in ("forecast_id", "slot", "horizon_hours", "model_version", "target_end")}
                (issued if created else duplicate).append(summary)
            except (ValueError, KeyError, OSError) as exc:
                errors.append({"horizon_hours": horizon, "reason": str(exc)[:300]})

    records = history(state)
    report = {
        "schema": 1,
        "policy": POLICY,
        "generated_at": current.isoformat(),
        "source_as_of": source_as_of.isoformat(),
        "issuance_audit": {
            "slot": slot.isoformat(),
            "deadline_ok": bool(deadline_ok),
            "feature_ready": bool(feature_ready),
            "eligible": ready,
            "late_or_reconstructed_issue_allowed": False,
            "base_checkpoint_refit_allowed": False,
        },
        "settled_this_run": settled,
        "issued_this_run": issued,
        "duplicate_this_run": duplicate,
        "errors": errors,
        "prospective": prospective_report(records),
        "claims": "24h/72h all-clock prospective variance shadow only; no production promotion",
    }
    atomic_json(state / "report.json", report)
    atomic_json(state / "public.json", public_payload(records, report))
    return report
