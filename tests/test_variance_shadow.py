import json

import numpy as np
import pandas as pd
import pytest

from signal_pipeline.variance_shadow import (
    HISTORICAL_QLIKE_SKILL,
    HORIZONS,
    POLICY,
    VERSION,
    connect,
    digest,
    issue,
    predict_variance,
    prospective_report,
    public_payload,
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


def record(horizon=24, slot="2026-09-16T00:00:00+00:00"):
    cp = checkpoint(horizon)
    origin = pd.Timestamp(slot)
    return {
        "schema": 1,
        "policy": POLICY,
        "slot": origin.isoformat(),
        "input_cutoff": origin.isoformat(),
        "available_at": (origin + pd.Timedelta(minutes=8)).isoformat(),
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


def resolved(horizon, day, *, candidate=0.8, baseline=1.0):
    slot = pd.Timestamp("2026-09-01T00:00:00Z") + pd.Timedelta(days=day)
    row = record(horizon, slot.isoformat())
    row.update(
        forecast_id=f"f-{horizon}-{day}",
        issued_at=(slot + pd.Timedelta(minutes=17)).isoformat(),
        outcome={"candidate_qlike": candidate, "persistence_qlike": baseline},
    )
    return row


def test_policy_matches_all_six_retrospective_variance_passes():
    assert HORIZONS == (6, 24, 72, 168, 336, 720)
    assert VERSION == "har_rv_variance_shadow_v2_all_horizons"
    assert POLICY["historical_gate"] == "full_horizon_v2_backtest_20260917_run_35188058829"
    assert POLICY["prospective_min_nonoverlap"] == {
        "6": 120, "24": 60, "72": 40, "168": 26, "336": 18, "720": 12
    }
    assert set(HISTORICAL_QLIKE_SKILL) == {str(h) for h in HORIZONS}
    assert all(HISTORICAL_QLIKE_SKILL[str(h)] >= 0.05 for h in HORIZONS)
    assert "not a validated target price" in POLICY["public_claim"]


def test_checkpoint_identity_and_maturity_are_enforced():
    assert validate_checkpoint(checkpoint(), 24)["horizon_hours"] == 24
    broken = checkpoint()
    broken["coef"][0] += 1
    with pytest.raises(ValueError, match="identity"):
        validate_checkpoint(broken, 24)


def test_prediction_uses_har_and_persistence_independently_for_long_horizon():
    row = {"eth_vol_24": 0.02, "eth_vol_168": 0.018, "eth_vol_720": 0.015}
    result = predict_variance(checkpoint(720), row, 720)
    assert result["predicted_variance_total"] > 0
    assert result["persistence_variance_total"] == pytest.approx(720 * 0.015**2)
    assert result["predicted_endpoint_sigma"] > 0


def test_issue_contract_rejects_non_midnight_origin():
    value = record()
    value["slot"] = "2026-09-16T01:00:00+00:00"
    value["input_cutoff"] = value["slot"]
    value["window_start"] = "2026-09-16T02:00:00+00:00"
    value["target_end"] = "2026-09-17T02:00:00+00:00"
    with pytest.raises(ValueError, match="00:00 UTC"):
        validate_record(value, now="2026-09-16T01:17:00+00:00")


def test_issue_is_idempotent_across_policy_transition_for_same_origin(tmp_path):
    state = tmp_path / "variance"
    first = issue(state, record(), now="2026-09-16T00:17:00+00:00")
    # Simulate a legacy row with the same origin/horizon but a different model identity.
    second_input = record()
    second_input["model_version"] = "c" * 64
    second_input["checkpoint"] = checkpoint()
    second = issue(state, second_input, now="2026-09-16T00:18:00+00:00")
    assert first == second
    with connect(state) as con:
        assert con.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0] == 1


def test_prospective_report_counts_nonoverlap_for_long_horizon():
    rows = [resolved(336, day) for day in range(30)]
    report = prospective_report(rows)["336"]
    assert report["resolved"] == 30
    assert report["nonoverlap"]["rows"] >= 2
    assert report["nonoverlap"]["qlike_improvement"] == pytest.approx(0.2)
    assert report["minimum_nonoverlap_for_review"] == 18
    assert report["performance_watch"] == "insufficient_evidence"


def test_validate_record_preserves_h_plus_one_boundary():
    value = record(6)
    value["target_end"] = "2026-09-16T06:00:00+00:00"
    with pytest.raises(ValueError, match=r"h\+1"):
        validate_record(value, now="2026-09-16T00:17:00+00:00")


def test_public_payload_exposes_scale_not_price_range():
    records = []
    for h in HORIZONS:
        row = record(h)
        row.update(forecast_id=f"f-{h}", issued_at="2026-09-16T00:17:00+00:00")
        records.append(row)
    report = {
        "generated_at": "2026-09-16T00:17:00+00:00",
        "source_as_of": "2026-09-16T00:00:00+00:00",
        "prospective": prospective_report(records),
    }
    payload = public_payload(records, report)
    assert payload["status"] == "research_shadow"
    assert set(payload["horizons"]) == {str(h) for h in HORIZONS}
    row = payload["horizons"]["720"]
    assert row["one_sigma_move_pct"] == pytest.approx(0.145)
    assert row["one_sigma_move_usd"] == pytest.approx(580.0)
    assert row["historical_gate"] == "pass"
    encoded = json.dumps(payload)
    assert "price_quantiles" not in encoded
    assert "q10" not in encoded and "q90" not in encoded
