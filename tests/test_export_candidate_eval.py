from __future__ import annotations

import json

from forecast_site import export_candidate_eval


def _candidate() -> dict:
    rows = [
        {
            "horizon_days": 7,
            "head": "regression",
            "model": "trimmed_regression_ensemble_equal",
            "prediction_date": "2026-01-01",
            "target_date": "2026-01-08",
            "reference_close": 100.0,
            "actual_close": 110.0,
            "actual_return": 0.10,
            "predicted_close": 80.0,
            "predicted_return": -0.20,
        },
        {
            "horizon_days": 7,
            "head": "regression",
            "model": "other_model",
            "prediction_date": "2026-01-01",
            "target_date": "2026-01-08",
            "reference_close": 100.0,
            "actual_close": 110.0,
            "predicted_close": 109.0,
        },
    ]
    return {
        "mode": "longrun_oof_phase6_production_36x30",
        "frozen_at": "2026-01-08T02:00:00+00:00",
        "last_checkpoint_utc": "2026-01-08T03:00:00+00:00",
        "horizons": {"7": {"predictions": rows}},
    }


def test_candidate_history_publishes_failure_and_uses_anchor_when_raw_loses() -> None:
    report = {
        "gate_status": "FAIL",
        "evaluated_at_utc": "2026-01-08T04:00:00+00:00",
        "evaluated_through_by_horizon": {"7": "2026-01-08"},
        "failures": ["h7 failed"],
        "warnings": [],
    }

    payload = export_candidate_eval.build_candidate_history(
        _candidate(),
        report,
        generated_at="2026-01-08T05:00:00+00:00",
    )

    assert payload["evaluation_status"] == "FAIL"
    phase = next(iter(payload["phases"].values()))
    assert phase["chart_model_by_horizon"]["7"]["chart_model"] == "no_change_anchor"
    point = phase["points"]["7"][0]
    assert point["predicted_close"] == 100.0
    assert point["model_predicted_close"] == 80.0


def test_candidate_export_main_writes_compact_json(tmp_path) -> None:
    candidate_path = tmp_path / "candidate.json"
    report_path = tmp_path / "report.json"
    output_path = tmp_path / "candidate_history.json"
    candidate_path.write_text(json.dumps(_candidate()), encoding="utf-8")
    report_path.write_text(json.dumps({"gate_status": "PASS", "failures": []}), encoding="utf-8")

    status = export_candidate_eval.main([
        "--candidate", str(candidate_path),
        "--report", str(report_path),
        "--output", str(output_path),
    ])

    assert status == 0
    assert json.loads(output_path.read_text(encoding="utf-8"))["evaluation_status"] == "PASS"
