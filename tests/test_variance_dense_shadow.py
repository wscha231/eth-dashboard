import json

import pandas as pd
import pytest

from signal_pipeline import variance_shadow as base
from signal_pipeline.variance_dense_shadow import (
    AUDIT_RUN,
    HORIZONS,
    POLICY,
    VERSION,
    issue,
    load_base_checkpoint,
    prospective_report,
    validate_record,
)


def checkpoint(horizon=24, fit_time="2026-09-15T00:17:00+00:00"):
    value = {
        "schema": 1,
        "policy": base.POLICY,
        "horizon_hours": horizon,
        "fit_time": fit_time,
        "fit_boundary": "2026-09-15T00:00:00+00:00",
        "training_rows": 700,
        "calibration_rows": 180,
        "training_target_end": "2026-03-18T23:00:00+00:00",
        "calibration_target_end": "2026-09-14T22:00:00+00:00",
        "source_snapshot": "a" * 64,
        "columns": list(base.VOL_COLUMNS),
        "scale_mean": [-8.0, -8.0, -8.0],
        "scale_std": [1.0, 1.0, 1.0],
        "coef": [0.2, 0.3, 0.5],
        "intercept": -8.0,
        "multiplier": 1.2,
    }
    value["model_version"] = base.digest(value)
    return value


def record(horizon=24, slot="2026-09-18T05:00:00+00:00"):
    cp = checkpoint(horizon)
    origin = pd.Timestamp(slot)
    return {
        "schema": 1,
        "policy": POLICY,
        "slot": origin.isoformat(),
        "input_cutoff": origin.isoformat(),
        "available_at": (origin + pd.Timedelta(minutes=6)).isoformat(),
        "window_start": (origin + pd.Timedelta(hours=1)).isoformat(),
        "target_end": (origin + pd.Timedelta(hours=horizon + 1)).isoformat(),
        "horizon_hours": horizon,
        "instrument": "ETH-USD",
        "reference_price": 4000.0,
        "model_version": cp["model_version"],
        "checkpoint": cp,
        "input_snapshot": "b" * 64,
        "predicted_variance_total": 0.02,
        "persistence_variance_total": 0.025,
        "predicted_endpoint_sigma": 0.145,
        "persistence_endpoint_sigma": 0.160,
    }


def resolved(horizon, hour, *, candidate=0.8, baseline=1.0):
    slot = pd.Timestamp("2026-09-18T00:00:00Z") + pd.Timedelta(hours=hour)
    row = record(horizon, slot.isoformat())
    row.update(
        forecast_id=f"f-{horizon}-{hour}",
        issued_at=(slot + pd.Timedelta(minutes=8)).isoformat(),
        outcome={"candidate_qlike": candidate, "persistence_qlike": baseline},
    )
    return row


def install_checkpoint(tmp_path, horizon=24, *, fit_time="2026-09-15T00:17:00+00:00"):
    state = tmp_path / "variance_shadow"
    models = state / "models"
    models.mkdir(parents=True)
    cp = checkpoint(horizon, fit_time=fit_time)
    filename = f"h{horizon}.json"
    path = models / filename
    path.write_text(json.dumps(cp))
    active = {
        "schema": 1,
        "models": {
            str(horizon): {
                "file": filename,
                "sha256": base.file_hash(path),
                "model_version": cp["model_version"],
                "fit_time": cp["fit_time"],
            }
        },
    }
    (state / "active.json").write_text(json.dumps(active))
    return cp


def test_dense_policy_is_only_for_authorized_horizons():
    assert HORIZONS == (24, 72)
    assert POLICY["horizons_hours"] == [24, 72]
    assert POLICY["excluded_horizons"]["6"].startswith("all-clock gate failed")
    assert POLICY["authorization_run"] == AUDIT_RUN == 35242529360
    assert POLICY["late_or_reconstructed_issue_allowed"] is False
    assert POLICY["promotion"].startswith("shadow_only")
    assert "never combine" in POLICY["review_cohort"]


def test_dense_record_accepts_non_midnight_current_origin_and_preserves_h_plus_one():
    value = record(24, "2026-09-18T05:00:00+00:00")
    validate_record(value, now="2026-09-18T05:08:00+00:00")
    broken = dict(value)
    broken["target_end"] = "2026-09-19T05:00:00+00:00"
    with pytest.raises(ValueError, match=r"h\+1"):
        validate_record(broken, now="2026-09-18T05:08:00+00:00")


def test_dense_record_rejects_6h_and_late_backfill():
    six = record(24)
    six["horizon_hours"] = 6
    with pytest.raises(ValueError, match="not authorized"):
        validate_record(six, now="2026-09-18T05:08:00+00:00")
    with pytest.raises(ValueError, match="late backfill"):
        validate_record(record(), now="2026-09-18T05:56:00+00:00")


def test_dense_issue_is_idempotent_in_separate_ledger(tmp_path):
    state = tmp_path / "dense"
    first, created = issue(state, record(), now="2026-09-18T05:08:00+00:00")
    second, created_again = issue(state, record(), now="2026-09-18T05:09:00+00:00")
    assert created is True
    assert created_again is False
    assert first == second
    with base.connect(state) as con:
        assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1


def test_dense_reuses_valid_legacy_checkpoint_and_never_refits(tmp_path):
    cp = install_checkpoint(tmp_path, 24)
    loaded = load_base_checkpoint(tmp_path, 24, now="2026-09-18T05:08:00+00:00")
    assert loaded == cp
    with pytest.raises(ValueError, match="only clock-authorized"):
        load_base_checkpoint(tmp_path, 6, now="2026-09-18T05:08:00+00:00")


def test_dense_rejects_stale_legacy_checkpoint(tmp_path):
    install_checkpoint(tmp_path, 24, fit_time="2026-08-01T00:17:00+00:00")
    with pytest.raises(ValueError, match="stale"):
        load_base_checkpoint(tmp_path, 24, now="2026-09-18T05:08:00+00:00")


def test_dense_nonoverlap_is_computed_inside_dense_cohort_only():
    rows = [resolved(24, hour) for hour in range(80)]
    report = prospective_report(rows)["24"]
    assert report["issued"] == 80
    assert report["resolved"] == 80
    assert report["nonoverlap"]["rows"] == 4
    assert report["nonoverlap"]["qlike_improvement"] == pytest.approx(0.2)
    assert report["minimum_nonoverlap_for_review"] == 60
    assert report["performance_watch"] == "insufficient_evidence"
