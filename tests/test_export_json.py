from __future__ import annotations

import datetime as dt
import json
import sys

import pandas as pd
import pytest

from forecast_site import export_json
from forecast_site.db import connect
from forecast_site.persist_forecast import persist_forecast


def _seed_forecast(conn, *, input_ts: str = "2026-05-20 00:00:00") -> int:
    run_id = conn.execute(
        """
        INSERT INTO forecast_runs (
            run_timestamp_utc, input_timestamp_utc, reference_price,
            reference_price_source, reference_price_timestamp_utc,
            model_phase, code_version, fast_mode
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "2026-05-20T01:00:00+00:00",
            input_ts,
            2000.0,
            "test",
            input_ts,
            "phase_test",
            "sha256:test",
            0,
        ),
    ).lastrowid
    conn.execute(
        """
        INSERT INTO forecasts (
            run_id, horizon_days, forecast_target_timestamp_utc,
            regression_model, regression_predicted_return, regression_predicted_close,
            classification_model, classification_predicted_direction,
            classification_direction_score_up, classification_probability_up,
            classification_signal_threshold, active_predicted_direction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            7,
            "2026-05-27 00:00:00",
            "test_reg",
            0.01,
            2020.0,
            "test_cls",
            "UP",
            0.72,
            0.60,
            0.60,
            "UP",
        ),
    )
    conn.commit()
    return int(run_id)


def test_export_health_reports_bootstrap_when_live_history_is_empty(temp_db) -> None:
    _seed_forecast(temp_db)

    health = export_json.export_health(
        temp_db,
        now=dt.datetime(2026, 5, 21, tzinfo=dt.timezone.utc),
    )

    assert health["status"] in {"bootstrap", "degraded"}
    assert health["schema_version"] == 2
    assert health["components"]["live_forecast"]["status"] == "ok"
    assert health["components"]["resolved_history"]["status"] == "bootstrap"
    assert health["db_counts"]["forecast_runs"] == 1
    assert health["db_counts"]["forecasts"] == 1
    assert health["db_counts"]["actuals"] == 0
    assert "7" in health["latest_forecasts_by_horizon"]


def test_export_health_reports_resolved_history(temp_db) -> None:
    _seed_forecast(temp_db)
    forecast_id = temp_db.execute("SELECT forecast_id FROM forecasts").fetchone()["forecast_id"]
    temp_db.execute(
        """
        INSERT INTO actuals (
            forecast_id, resolved_at_utc, actual_close, actual_return,
            direction_actual, direction_correct, active_direction_correct,
            return_absolute_error, price_absolute_error, brier_contribution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            forecast_id,
            "2026-05-28T00:00:00+00:00",
            2040.0,
            0.02,
            "UP",
            1,
            1,
            0.01,
            20.0,
            0.16,
        ),
    )
    temp_db.commit()

    health = export_json.export_health(
        temp_db,
        now=dt.datetime(2026, 5, 28, tzinfo=dt.timezone.utc),
    )

    assert health["db_counts"]["actuals"] == 1
    assert health["latest_resolved_by_horizon"]["7"]["resolved_count"] == 1
    assert health["latest_resolved_by_horizon"]["7"]["latest_resolved_target_utc"] == "2026-05-27 00:00:00"


def test_export_history_carries_live_chart_selection_fields(temp_db) -> None:
    _seed_forecast(temp_db)
    forecast_id = temp_db.execute("SELECT forecast_id FROM forecasts").fetchone()["forecast_id"]
    temp_db.execute(
        """
        UPDATE forecasts
        SET active_predicted_close = ?, active_predicted_direction = ?,
            forecast_decision_mode = ?, forecast_actionability = ?,
            forecast_point_price_reliable = ?
        WHERE forecast_id = ?
        """,
        (2030.0, "UP", "uncertainty_range_only", "range_only", 0, forecast_id),
    )
    temp_db.execute(
        """
        INSERT INTO actuals (
            forecast_id, resolved_at_utc, actual_close, actual_return,
            direction_actual, direction_correct, active_direction_correct,
            return_absolute_error, price_absolute_error, brier_contribution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (forecast_id, "2026-05-28T00:00:00+00:00", 2040.0, 0.02, "UP", 1, 1, 0.01, 20.0, 0.16),
    )
    temp_db.commit()

    row = export_json.export_history(temp_db)[0]

    assert row["regression_model"] == "test_reg"
    assert row["active_predicted_close"] == 2030.0
    assert row["classification_direction_score_up"] == pytest.approx(0.72)
    assert row["classification_signal_threshold"] == pytest.approx(0.60)
    assert row["forecast_actionability"] == "range_only"
    assert row["forecast_point_price_reliable"] == 0


def test_export_health_separates_stale_oof_from_fresh_live_run(temp_db) -> None:
    _seed_forecast(temp_db, input_ts="2026-05-20T00:00:00+00:00")
    report = {
        "gate_status": "FAIL",
        "evaluated_through_by_horizon": {"7": "2026-05-01", "30": "2026-05-01"},
    }

    health = export_json.export_health(
        temp_db,
        model_eval_report=report,
        now=dt.datetime(2026, 5, 20, 12, tzinfo=dt.timezone.utc),
    )

    assert health["components"]["live_forecast"]["status"] == "ok"
    assert health["components"]["oof_evaluation"]["status"] == "stale"
    assert health["status"] == "stale"


def test_export_json_main_writes_health_file(temp_db_path, tmp_path, monkeypatch) -> None:
    from forecast_site.db import connect

    conn = connect(temp_db_path)
    try:
        _seed_forecast(conn)
    finally:
        conn.close()

    out_dir = tmp_path / "public"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_json",
            "--db",
            str(temp_db_path),
            "--output-dir",
            str(out_dir),
        ],
    )
    export_json.main()

    health_path = out_dir / "health.json"
    assert health_path.exists()
    payload = json.loads(health_path.read_text(encoding="utf-8"))
    assert payload["db_counts"]["forecast_runs"] == 1


def test_persist_and_export_live_direction_score(temp_db_path, tmp_path) -> None:
    summary_path = tmp_path / "latest_forecast_summary.csv"
    pd.DataFrame([{
        "horizon_steps": 7,
        "forecast_input_timestamp": "2026-08-25 00:00:00",
        "forecast_target_timestamp": "2026-09-01 00:00:00",
        "reference_price": 2000.0,
        "classification_predicted_direction": "UP",
        "classification_direction_score_up": 0.73,
        "classification_probability_up": 0.56,
        "classification_probability_down": 0.44,
        "classification_signal_threshold": 0.60,
        "classification_confidence": 0.56,
    }]).to_csv(summary_path, index=False)

    run_id = persist_forecast(
        summary_path,
        model_phase="direction_score_test",
        db_path=temp_db_path,
    )
    conn = connect(temp_db_path)
    try:
        stored = conn.execute(
            """
            SELECT classification_direction_score_up, classification_signal_threshold
            FROM forecasts WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        latest = export_json.export_latest(conn)["horizons"]["7"]
    finally:
        conn.close()

    assert stored["classification_direction_score_up"] == pytest.approx(0.73)
    assert stored["classification_signal_threshold"] == pytest.approx(0.60)
    assert latest["classification_direction_score_up"] == pytest.approx(0.73)
    assert latest["classification_signal_threshold"] == pytest.approx(0.60)
