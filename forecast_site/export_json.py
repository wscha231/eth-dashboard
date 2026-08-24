"""Query the DB and export website-ready JSON.

The web frontend (Next.js, static HTML, whatever) should consume a handful of
small JSON blobs rather than running SQL itself. This script snapshots those
blobs into forecast_site/public/ on every cron tick:

    public/latest.json
        {"generated_at": ..., "reference_price": ..., "horizons": {"7": {...}, "30": {...}}}
        -- The "hero card" on the homepage.

    public/accuracy.json
        {"7": {"30": {...}, "90": {...}, "180": {...}, "all": {...}}, "30": {...}}
        -- Rolling accuracy badges.

    public/history.json
        [{"run_timestamp_utc": ..., "horizon": 7, "predicted_close": ...,
          "actual_close": ..., "direction_correct": ...}, ...]
        -- Past predictions for the "track record" chart.

    public/health.json
        Component-level freshness for live input, resolved history, the newest
        OOF candidate, and collector data quality.

Run:
    python -m forecast_site.export_json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
from forecast_site.db import DEFAULT_DB_PATH, connect  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "public"


def _row_to_dict(row) -> dict:
    return {k: row[k] for k in row.keys()}


def export_latest(conn) -> dict:
    run = conn.execute(
        """
        SELECT run_id, run_timestamp_utc, input_timestamp_utc, reference_price,
               reference_price_source, reference_price_timestamp_utc,
               model_phase, code_version, fast_mode
        FROM forecast_runs
        ORDER BY run_timestamp_utc DESC
        LIMIT 1
        """
    ).fetchone()
    if run is None:
        return {"generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
                "run": None, "horizons": {}}
    forecasts = conn.execute(
        """
        SELECT horizon_days, forecast_target_timestamp_utc,
               regression_model, regression_predicted_return, regression_predicted_close,
               regression_lower_close_10, regression_upper_close_90,
               classification_model, classification_predicted_direction,
               classification_probability_up, classification_confidence,
               hybrid_predicted_direction, hybrid_signal_tier, hybrid_confidence,
               hybrid_volatility_scale, hybrid_scenario_spread,
               forecast_decision_mode, forecast_actionability,
               forecast_point_price_reliable, forecast_center_return,
               forecast_uncertainty_return, forecast_lower_return, forecast_upper_return,
               active_predicted_return, active_predicted_close,
               active_bear_predicted_close, active_bull_predicted_close,
               regime_predicted_state, reversal_predicted_signal,
               macro_risk_regime, macro_directional_bias, macro_readiness_status
        FROM forecasts
        WHERE run_id = ?
        ORDER BY horizon_days
        """,
        (run["run_id"],),
    ).fetchall()

    horizons = {}
    for row in forecasts:
        h = str(row["horizon_days"])
        horizons[h] = _row_to_dict(row)
    return {
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "run": _row_to_dict(run),
        "horizons": horizons,
    }


def export_accuracy(conn) -> dict:
    """Latest accuracy snapshot per (horizon, window)."""
    rows = conn.execute(
        """
        SELECT horizon_days, window_days, snapshot_utc, resolved_count,
               direction_accuracy, brier_score, price_mape_percent, price_rmse, return_mae
        FROM accuracy_snapshot a
        WHERE snapshot_utc = (
            SELECT MAX(snapshot_utc) FROM accuracy_snapshot
            WHERE horizon_days = a.horizon_days AND window_days = a.window_days
        )
        ORDER BY horizon_days, window_days
        """
    ).fetchall()
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        h = str(row["horizon_days"])
        w = "all" if row["window_days"] >= 9999 else str(row["window_days"])
        out.setdefault(h, {})[w] = _row_to_dict(row)
    return {
        "generated_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "horizons": out,
    }


def export_history(conn, limit: int = 200) -> list[dict]:
    """Newest resolved forecasts; the frontend sorts this window for charts."""
    rows = conn.execute(
        """
        SELECT r.run_timestamp_utc, r.input_timestamp_utc, r.reference_price,
               f.horizon_days, f.forecast_target_timestamp_utc,
               f.regression_model, f.regression_predicted_close,
               f.active_predicted_close, f.active_predicted_direction,
               f.forecast_decision_mode, f.forecast_actionability,
               f.forecast_point_price_reliable,
               f.classification_probability_up,
               f.classification_predicted_direction, f.hybrid_signal_tier,
               a.actual_close, a.actual_return, a.direction_actual,
               a.direction_correct, a.price_absolute_error, a.brier_contribution
        FROM actuals a
        JOIN forecasts f ON f.forecast_id = a.forecast_id
        JOIN forecast_runs r ON r.run_id = f.run_id
        ORDER BY f.forecast_target_timestamp_utc DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


def _scalar(conn, sql: str, params: tuple = ()) -> int:
    row = conn.execute(sql, params).fetchone()
    if row is None:
        return 0
    value = row[0]
    return int(value or 0)


def _parse_utc(value: Any) -> _dt.datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = _dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_dt.timezone.utc)
    return parsed.astimezone(_dt.timezone.utc)


def _age_hours(value: Any, now: _dt.datetime) -> float | None:
    parsed = _parse_utc(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds() / 3600.0)


def _evaluation_age_hours(report: dict[str, Any] | None, now: _dt.datetime) -> float | None:
    if not isinstance(report, dict):
        return None
    through = report.get("evaluated_through_by_horizon") or {}
    ages = [
        age
        for age in (_age_hours(value, now) for value in through.values())
        if age is not None
    ]
    if ages:
        # Every advertised horizon must meet the SLO, so use the oldest one.
        return max(ages)
    return _age_hours(
        report.get("candidate_checkpoint_utc") or report.get("candidate_frozen_at"),
        now,
    )


def _data_quality_component(data_quality_report: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data_quality_report, dict):
        return {
            "status": "unknown",
            "tail_event_readiness": None,
            "reason": "data-quality report is unavailable",
        }
    readiness = (
        data_quality_report.get("tail_event_readiness")
        or (data_quality_report.get("summary") or {}).get("tail_event_readiness")
    )
    reported_status = str(data_quality_report.get("status") or "").strip().lower()
    weak_values = {"weak", "degraded", "poor", "fail", "failed", "error"}
    status = "degraded" if str(readiness).lower() in weak_values or reported_status in weak_values else "ok"
    return {
        "status": status,
        "tail_event_readiness": readiness,
        "reported_status": reported_status or None,
        "reason": "tail-event feature coverage is weak" if status == "degraded" else None,
    }


def export_health(
    conn,
    *,
    model_eval_report: dict[str, Any] | None = None,
    data_quality_report: dict[str, Any] | None = None,
    now: _dt.datetime | None = None,
    live_slo_hours: float = 27.0,
    oof_slo_hours: float = 8.0 * 24.0,
) -> dict:
    """Operational freshness report split by independently updated component."""
    now = now or _dt.datetime.now(tz=_dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=_dt.timezone.utc)
    now = now.astimezone(_dt.timezone.utc)
    generated_at = now.isoformat()
    latest_run = conn.execute(
        """
        SELECT run_id, run_timestamp_utc, input_timestamp_utc, model_phase, code_version
        FROM forecast_runs
        ORDER BY run_timestamp_utc DESC
        LIMIT 1
        """
    ).fetchone()
    latest_by_horizon = conn.execute(
        """
        SELECT f.horizon_days, f.forecast_target_timestamp_utc,
               r.input_timestamp_utc, r.run_timestamp_utc
        FROM forecasts f
        JOIN forecast_runs r ON r.run_id = f.run_id
        WHERE r.run_timestamp_utc = (SELECT MAX(run_timestamp_utc) FROM forecast_runs)
        ORDER BY f.horizon_days
        """
    ).fetchall()
    resolved_by_horizon = conn.execute(
        """
        SELECT f.horizon_days, COUNT(*) AS resolved_count,
               MAX(f.forecast_target_timestamp_utc) AS latest_resolved_target_utc
        FROM actuals a
        JOIN forecasts f ON f.forecast_id = a.forecast_id
        GROUP BY f.horizon_days
        ORDER BY f.horizon_days
        """
    ).fetchall()
    due_by_horizon = conn.execute(
        """
        SELECT f.horizon_days, COUNT(*) AS due_count
        FROM forecasts f
        LEFT JOIN actuals a ON a.forecast_id = f.forecast_id
        WHERE a.forecast_id IS NULL
          AND DATE(f.forecast_target_timestamp_utc) < DATE('now')
        GROUP BY f.horizon_days
        ORDER BY f.horizon_days
        """
    ).fetchall()

    counts = {
        "forecast_runs": _scalar(conn, "SELECT COUNT(*) FROM forecast_runs"),
        "forecasts": _scalar(conn, "SELECT COUNT(*) FROM forecasts"),
        "actuals": _scalar(conn, "SELECT COUNT(*) FROM actuals"),
        "accuracy_snapshot": _scalar(conn, "SELECT COUNT(*) FROM accuracy_snapshot"),
        "backtest_runs": _scalar(conn, "SELECT COUNT(*) FROM backtest_runs"),
        "backtest_predictions": _scalar(conn, "SELECT COUNT(*) FROM backtest_predictions"),
    }
    latest = {
        str(row["horizon_days"]): _row_to_dict(row)
        for row in latest_by_horizon
    }
    resolved = {
        str(row["horizon_days"]): _row_to_dict(row)
        for row in resolved_by_horizon
    }
    due = {
        str(row["horizon_days"]): int(row["due_count"])
        for row in due_by_horizon
    }
    live_age = _age_hours(latest_run["input_timestamp_utc"], now) if latest_run is not None else None
    if latest_run is None or live_age is None:
        live_status = "stale"
        live_reason = "no usable forecast run in DB"
    elif live_age > live_slo_hours:
        live_status = "stale"
        live_reason = f"latest input is older than {live_slo_hours:g} hours"
    else:
        live_status = "ok"
        live_reason = None

    if counts["actuals"] == 0:
        resolved_status = "bootstrap"
        resolved_reason = "live forecast history has not accumulated yet"
    elif any(count > 0 for count in due.values()):
        resolved_status = "degraded"
        resolved_reason = "one or more due forecasts are still unresolved"
    else:
        resolved_status = "ok"
        resolved_reason = None

    evaluation_age = _evaluation_age_hours(model_eval_report, now)
    gate_status = str((model_eval_report or {}).get("gate_status") or "").upper() or None
    if evaluation_age is None:
        evaluation_status = "unknown"
        evaluation_reason = "latest candidate evaluation metadata is unavailable"
    elif evaluation_age > oof_slo_hours:
        evaluation_status = "stale"
        evaluation_reason = f"latest OOF target is older than {oof_slo_hours / 24:g} days"
    elif gate_status == "FAIL":
        evaluation_status = "degraded"
        evaluation_reason = "latest candidate failed its promotion gate"
    else:
        evaluation_status = "ok"
        evaluation_reason = None

    components = {
        "live_forecast": {
            "status": live_status,
            "input_timestamp_utc": latest_run["input_timestamp_utc"] if latest_run is not None else None,
            "age_hours": round(live_age, 2) if live_age is not None else None,
            "slo_hours": live_slo_hours,
            "reason": live_reason,
        },
        "resolved_history": {
            "status": resolved_status,
            "latest_by_horizon": resolved,
            "unresolved_due_count_by_horizon": due,
            "reason": resolved_reason,
        },
        "oof_evaluation": {
            "status": evaluation_status,
            "gate_status": gate_status,
            "evaluated_through_by_horizon": (model_eval_report or {}).get("evaluated_through_by_horizon") or {},
            "age_hours": round(evaluation_age, 2) if evaluation_age is not None else None,
            "slo_hours": oof_slo_hours,
            "reason": evaluation_reason,
        },
        "data_quality": _data_quality_component(data_quality_report),
    }
    component_statuses = [component["status"] for component in components.values()]
    if "stale" in component_statuses:
        status = "stale"
    elif "degraded" in component_statuses or "unknown" in component_statuses:
        status = "degraded"
    elif "bootstrap" in component_statuses:
        status = "bootstrap"
    else:
        status = "ok"
    notes = [
        component["reason"]
        for component in components.values()
        if component.get("reason")
    ]

    return {
        "schema_version": 2,
        "generated_at": generated_at,
        "status": status,
        "notes": notes,
        "components": components,
        "slos": {
            "live_forecast_max_age_hours": live_slo_hours,
            "oof_evaluation_max_age_hours": oof_slo_hours,
        },
        "latest_run": _row_to_dict(latest_run) if latest_run is not None else None,
        "latest_forecasts_by_horizon": latest,
        "latest_resolved_by_horizon": resolved,
        "unresolved_due_count_by_horizon": due,
        "db_counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--history-limit", type=int, default=200)
    parser.add_argument(
        "--model-eval-report",
        default=str(DEFAULT_OUTPUT_DIR / "model_eval_latest.json"),
    )
    parser.add_argument(
        "--data-quality-report",
        default=str(PROJECT_ROOT / "lake" / "reports" / "collector_data_quality_audit.json"),
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def load_optional_json(path_value: str) -> dict[str, Any] | None:
        path = Path(path_value)
        if not path.exists():
            return None
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    model_eval_report = load_optional_json(args.model_eval_report)
    if model_eval_report is None:
        model_eval_report = load_optional_json(str(DEFAULT_OUTPUT_DIR / "model_eval_report.json"))
    data_quality_report = load_optional_json(args.data_quality_report)

    conn = connect(args.db)
    try:
        latest = export_latest(conn)
        accuracy = export_accuracy(conn)
        history = export_history(conn, limit=args.history_limit)
        health = export_health(
            conn,
            model_eval_report=model_eval_report,
            data_quality_report=data_quality_report,
        )
    finally:
        conn.close()

    (out_dir / "latest.json").write_text(
        json.dumps(latest, indent=2, default=str), encoding="utf-8")
    (out_dir / "accuracy.json").write_text(
        json.dumps(accuracy, indent=2, default=str), encoding="utf-8")
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, default=str), encoding="utf-8")
    (out_dir / "health.json").write_text(
        json.dumps(health, indent=2, default=str), encoding="utf-8")

    print(f"[export_json] wrote:")
    print(f"  {out_dir / 'latest.json'}    horizons={list(latest['horizons'].keys())}")
    print(f"  {out_dir / 'accuracy.json'}  horizons={list(accuracy['horizons'].keys())}")
    print(f"  {out_dir / 'history.json'}   rows={len(history)}")
    print(f"  {out_dir / 'health.json'}    status={health['status']}")


if __name__ == "__main__":
    main()
