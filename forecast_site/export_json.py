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

Run:
    python -m forecast_site.export_json
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
from pathlib import Path

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
    """Chronological list of resolved forecasts for chart rendering."""
    rows = conn.execute(
        """
        SELECT r.run_timestamp_utc, r.input_timestamp_utc, r.reference_price,
               f.horizon_days, f.forecast_target_timestamp_utc,
               f.regression_predicted_close, f.classification_probability_up,
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--history-limit", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    try:
        latest = export_latest(conn)
        accuracy = export_accuracy(conn)
        history = export_history(conn, limit=args.history_limit)
    finally:
        conn.close()

    (out_dir / "latest.json").write_text(
        json.dumps(latest, indent=2, default=str), encoding="utf-8")
    (out_dir / "accuracy.json").write_text(
        json.dumps(accuracy, indent=2, default=str), encoding="utf-8")
    (out_dir / "history.json").write_text(
        json.dumps(history, indent=2, default=str), encoding="utf-8")

    print(f"[export_json] wrote:")
    print(f"  {out_dir / 'latest.json'}    horizons={list(latest['horizons'].keys())}")
    print(f"  {out_dir / 'accuracy.json'}  horizons={list(accuracy['horizons'].keys())}")
    print(f"  {out_dir / 'history.json'}   rows={len(history)}")


if __name__ == "__main__":
    main()
