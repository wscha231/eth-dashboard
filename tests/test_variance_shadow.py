import numpy as np
import pandas as pd
import pytest

from signal_pipeline.variance_shadow import (
    HORIZONS,
    POLICY,
    connect,
    digest,
    issue,
    predict_variance,
    prospective_report,
    validate_checkpoint,
    validate_record,
)


def checkpoint(horizon=24):
    value = {
        "schema": 1,
        "policy": POLICY,
        "horizon_hours": horizon,
        "fit_time": "2026-09-15T00:10:00+00:00",
        "fit_boundary": "2026-09-15T00:00:00+00:00",
        "training_rows": 700,
        "calibration_rows": 180,
        "training_target_end": "2026-03-18T23:00:00+00:00",
        "calibration_target_end": "2026-09-14T22:00:00+00:00",
        "source_snapshot": "a" * 64,
        "columns": ["eth_vol_24", "eth_vol_168", "eth_vol_720"],
        "scale_mean": [-8.0, -8.0, -8.0],
        "scale_std": [1.0, 1.0, 1.0],
        "coef": [0.2, 0.3, 0.5],
        "intercept": -8.0,
        "multiplier": 1.2,
    }
    value["model_version"] = digest(value)
    return value


def record(horizon=24):
    cp = checkpoint(horizon)
    return {
        "schema": 1,
        "policy": POLICY,
        "slot": "2026-09-16T00:00:00+00:00",
        "input_cutoff": "2026-09-16T00:00:00+00:00",
        "available_at": "2026-09-16T00:08:00+00:00",
        "window_start": "2026-09-16T01:00:00+00:00",
        "target_end": (pd.Timestamp("2026-09-16T00:00:00Z") + pd.Timedelta(hours=horizon + 1)).isoformat(),
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


def test_policy_matches_only_r2_passed_short_horizons():
    assert HORIZONS == (6, 24, 72)
    assert POLICY["historical_gate"] == "r2_range_volatility_wave1_20260916"
    assert POLICY["issue_origin"].startswith("00:00 UTC daily")
    assert POLICY["promotion"].startswith("shadow_only")


def test_checkpoint_identity_and_maturity_are_enforced():
    assert validate_checkpoint(checkpoint(), 24)["horizon_hours"] == 24
    broken = checkpoint()
    broken["coef"][0] += 1
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(broken, 24)


def test_prediction_uses_har_and_persistence_independently():
    row = {"eth_vol_24": 0.02, "eth_vol_168": 0.018, "eth_vol_720": 0.015}
    result = predict_variance(checkpoint(), row, 24)
    assert result["predicted_variance_total"] > 0
    assert result["persistence_variance_total"] == pytest.approx(24 * 0.015**2)
    assert result["predicted_endpoint_sigma"] > 0


def test_issue_contract_rejects_non_midnight_origin():
    value = record()
    value["slot"] = "2026-09-16T01:00:00+00:00"
    value["input_cutoff"] = value["slot"]
    value["window_start"] = "2026-09-16T02:00:00+00:00"
    value["target_end"] = "2026-09-17T02:00:00+00:00"
    with pytest.raises(ValueError, match="00:00 UTC"):
        validate_record(value, now="2026-09-16T01:17:00+00:00")


def test_issue_is_idempotent_and_append_only(tmp_path):
    state = tmp_path / "variance"
    first = issue(state, record(), now="2026-09-16T00:17:00+00:00")
    second = issue(state, record(), now="2026-09-16T00:18:00+00:00")
    assert first == second
    with connect(state) as con:
        assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1


def test_prospective_report_counts_nonoverlap_separately():
    rows = []
    for day in range(4):
        slot = pd.Timestamp("2026-09-01T00:00:00Z") + pd.Timedelta(days=day)
        rows.append({
            "slot": slot.isoformat(),
            "target_end": (slot + pd.Timedelta(hours=25)).isoformat(),
            "horizon_hours": 24,
            "outcome": {"candidate_qlike": 0.8, "persistence_qlike": 1.0},
        })
    report = prospective_report(rows)["24"]
    assert report["resolved"] == 4
    assert report["nonoverlap"]["rows"] == 2
    assert report["nonoverlap"]["qlike_improvement"] == pytest.approx(0.2)
    assert report["performance_watch"] == "insufficient_evidence"


def test_validate_record_preserves_h_plus_one_boundary():
    value = record(6)
    value["target_end"] = "2026-09-16T06:00:00+00:00"
    with pytest.raises(ValueError, match="h\+1"):
        validate_record(value, now="2026-09-16T00:17:00+00:00")
