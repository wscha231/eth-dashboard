"""Emit `public/backtest.json` summarizing every frozen model version.

Two frontend views:

1. Version comparison table: for each model_phase, the best regression model
   (by price_rmse) and best classification model (by brier_score) at each
   horizon. This is the "which version performs best on the same 3 years"
   chart.

2. Full metric matrix: every (phase, horizon, head, model) row flat-packed
   so the site can build custom charts (e.g. brier-over-phases for h=7
   catboost_classifier).

Also emits `public/backtest_versions.json` — just the phase metadata list so
the site sidebar can show a chronological version picker.

Output:
    public/backtest.json
    public/backtest_versions.json

Run:
    python -m forecast_site.export_backtest_json --db forecast_site/predictions.db
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from forecast_site.db import DEFAULT_DB_PATH, connect  # noqa: E402

DEFAULT_OUTPUT_DIR = Path(__file__).parent / "public"


def _row_to_dict(row) -> dict:
    return {k: row[k] for k in row.keys()}


def _drop_none(d: dict) -> dict:
    """Shrink frontend payload: drop keys whose value is None."""
    return {k: v for k, v in d.items() if v is not None}


def export_versions(conn) -> list[dict]:
    """Chronological list of every frozen version with run-level metadata.

    Feeds the site's version picker / timeline.
    """
    rows = conn.execute(
        """
        SELECT backtest_run_id, model_phase, mode, frozen_utc,
               cv_test_size, code_bytes, master_data_csv, freeze_json_path
        FROM backtest_runs
        ORDER BY frozen_utc ASC, backtest_run_id ASC
        """
    ).fetchall()
    return [_drop_none(_row_to_dict(r)) for r in rows]


def export_headline_table(conn) -> list[dict]:
    """One row per (model_phase, horizon) with the best metric on each head.

    "Best" = lowest price_rmse for regression, lowest brier_score for
    classification. This is what the headline comparison chart reads.
    """
    rows = conn.execute(
        """
        WITH ranked_reg AS (
            SELECT
                br.backtest_run_id,
                br.model_phase,
                br.frozen_utc,
                bm.horizon_days,
                bm.model               AS reg_model,
                bm.price_rmse          AS reg_price_rmse,
                bm.price_mape_percent  AS reg_price_mape,
                bm.directional_accuracy AS reg_directional_accuracy,
                ROW_NUMBER() OVER (
                    PARTITION BY br.backtest_run_id, bm.horizon_days
                    ORDER BY bm.price_rmse ASC NULLS LAST
                ) AS rnk
            FROM backtest_metrics bm
            JOIN backtest_runs   br USING (backtest_run_id)
            WHERE bm.head = 'regression'
              AND bm.price_rmse IS NOT NULL
        ),
        ranked_cls AS (
            SELECT
                br.backtest_run_id,
                br.model_phase,
                bm.horizon_days,
                bm.model          AS cls_model,
                bm.brier_score    AS cls_brier,
                bm.roc_auc        AS cls_roc_auc,
                bm.accuracy       AS cls_accuracy,
                bm.signal_threshold AS cls_threshold,
                ROW_NUMBER() OVER (
                    PARTITION BY br.backtest_run_id, bm.horizon_days
                    ORDER BY bm.brier_score ASC NULLS LAST
                ) AS rnk
            FROM backtest_metrics bm
            JOIN backtest_runs   br USING (backtest_run_id)
            WHERE bm.head = 'classification'
              AND bm.brier_score IS NOT NULL
        )
        SELECT
            r.backtest_run_id, r.model_phase, r.frozen_utc, r.horizon_days,
            r.reg_model, r.reg_price_rmse, r.reg_price_mape, r.reg_directional_accuracy,
            c.cls_model, c.cls_brier, c.cls_roc_auc, c.cls_accuracy, c.cls_threshold
        FROM ranked_reg r
        LEFT JOIN ranked_cls c
               ON c.backtest_run_id = r.backtest_run_id
              AND c.horizon_days    = r.horizon_days
              AND c.rnk = 1
        WHERE r.rnk = 1
        ORDER BY r.frozen_utc ASC, r.horizon_days ASC
        """
    ).fetchall()
    return [_drop_none(_row_to_dict(r)) for r in rows]


def export_full_matrix(conn) -> list[dict]:
    """Every (phase, horizon, head, model) metric row flat-packed."""
    rows = conn.execute(
        """
        SELECT br.model_phase, br.frozen_utc,
               bm.horizon_days, bm.head, bm.model,
               bm.return_mae, bm.price_mae, bm.price_rmse, bm.price_mape_percent,
               bm.directional_accuracy,
               bm.accuracy, bm.balanced_accuracy,
               bm.precision_score, bm.recall_score, bm.f1,
               bm.roc_auc, bm.brier_score, bm.signal_threshold,
               bm.folds, bm.component_models
        FROM backtest_metrics bm
        JOIN backtest_runs   br USING (backtest_run_id)
        ORDER BY br.frozen_utc ASC, bm.horizon_days ASC, bm.head ASC, bm.model ASC
        """
    ).fetchall()
    return [_drop_none(_row_to_dict(r)) for r in rows]


LONGRUN_CHART_MODEL = "trimmed_regression_ensemble_equal"


def _finite_float(value) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rmse(errors: list[float]) -> float | None:
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None


def export_longrun_history(conn, model: str = LONGRUN_CHART_MODEL) -> dict:
    """Compact chart-ready payload: per-phase, per-horizon predicted-vs-actual
    for the 3-year OOF backtest.

    The full ``backtest_predictions_{phase}.json`` is ~10 MB because it holds
    all 16 models × 2 horizons × 1,080 days. For the hero line chart on the
    homepage we only need ONE model (the production ensemble). Pre-aggregating
    here brings the frontend payload down to ~130 KB.
    """
    rows = conn.execute(
        """
        SELECT br.model_phase, br.frozen_utc, br.mode,
               bp.horizon_days, bp.target_date, bp.prediction_date,
               bp.reference_close, bp.actual_close, bp.actual_return,
               bp.predicted_return, bp.predicted_close
        FROM backtest_predictions bp
        JOIN backtest_runs        br USING (backtest_run_id)
        WHERE br.mode LIKE 'longrun_oof_%'
          AND bp.head = 'regression'
          AND bp.model = ?
        ORDER BY br.model_phase ASC, bp.horizon_days ASC, bp.target_date ASC
        """,
        (model,),
    ).fetchall()

    chart_stats: dict[tuple[str, str], dict] = {}
    grouped_rows: dict[tuple[str, str], list] = {}
    for r in rows:
        grouped_rows.setdefault((r["model_phase"], str(r["horizon_days"])), []).append(r)

    for key, group in grouped_rows.items():
        model_errors: list[float] = []
        anchor_errors: list[float] = []
        for r in group:
            predicted_close = _finite_float(r["predicted_close"])
            reference_close = _finite_float(r["reference_close"])
            actual_close = _finite_float(r["actual_close"])
            if actual_close is None:
                continue
            if predicted_close is not None:
                model_errors.append(predicted_close - actual_close)
            if reference_close is not None:
                anchor_errors.append(reference_close - actual_close)
        model_rmse = _rmse(model_errors)
        no_change_rmse = _rmse(anchor_errors)
        use_anchor = bool(
            no_change_rmse is not None
            and (model_rmse is None or no_change_rmse <= model_rmse)
        )
        edge_pct = (
            100.0 * (no_change_rmse - model_rmse) / no_change_rmse
            if model_rmse is not None and no_change_rmse
            else None
        )
        chart_stats[key] = {
            "raw_model": model,
            "chart_model": "no_change_anchor" if use_anchor else model,
            "model_price_rmse": model_rmse,
            "no_change_price_rmse": no_change_rmse,
            "point_forecast_beats_no_change": bool(
                model_rmse is not None
                and no_change_rmse is not None
                and model_rmse < no_change_rmse
            ),
            "point_edge_vs_no_change_pct": edge_pct,
        }

    phases: dict[str, dict] = {}
    for r in rows:
        phase = r["model_phase"]
        bundle = phases.setdefault(phase, {
            "model_phase": phase,
            "frozen_utc":  r["frozen_utc"],
            "mode":        r["mode"],
            "model":       model,
            "raw_model":   model,
            "chart_selection": "fallback_to_no_change_when_point_rmse_has_no_edge",
            "chart_model_by_horizon": {},
            "points":      {},
        })
        horizon_key = str(r["horizon_days"])
        stats = chart_stats.get((phase, horizon_key), {})
        chart_model = stats.get("chart_model") or model
        bundle["chart_model_by_horizon"][horizon_key] = stats
        bucket = bundle["points"].setdefault(horizon_key, [])
        predicted_close = r["predicted_close"]
        predicted_return = r["predicted_return"]
        raw_predicted_close = r["predicted_close"]
        raw_predicted_return = r["predicted_return"]
        # The selected line follows the RMSE decision. The payload retains the
        # raw model separately so the frontend can show it as a secondary,
        # explicitly unselected candidate instead of silently overriding the
        # no-change decision.
        if chart_model == "no_change_anchor":
            predicted_close = r["reference_close"]
            predicted_return = 0.0
        row = {
            "display_date":      r["target_date"],
            "target_date":      r["target_date"],
            "prediction_date":  r["prediction_date"],
            "reference_close":  r["reference_close"],
            "predicted_close":  predicted_close,
            "actual_close":     r["actual_close"],
            "predicted_return": predicted_return,
            "actual_return":    r["actual_return"],
            "chart_model":      chart_model,
            "model_predicted_close":  raw_predicted_close,
            "model_predicted_return": raw_predicted_return,
        }
        if chart_model == "no_change_anchor":
            row["raw_predicted_close"] = raw_predicted_close
            row["raw_predicted_return"] = raw_predicted_return
            row["benchmark_predicted_close"] = predicted_close
            row["benchmark_predicted_return"] = predicted_return
        bucket.append(row)
    return phases


def export_predictions_by_phase(conn) -> dict[str, dict]:
    """Per-phase bundles of OOF prediction rows.

    Returns a dict keyed by ``model_phase`` so the caller can write one JSON
    file per phase (keeps payloads tractable — a single 36×30 longrun has
    ~2000 rows per horizon per model). Phases with zero prediction rows are
    omitted entirely (standard 3-fold freezes don't persist per-date OOF).
    """
    rows = conn.execute(
        """
        SELECT br.model_phase, br.frozen_utc, br.mode,
               bp.horizon_days, bp.head, bp.model,
               bp.prediction_date, bp.target_date, bp.fold_index,
               bp.reference_close, bp.actual_close, bp.actual_return, bp.actual_label,
               bp.predicted_return, bp.predicted_close,
               bp.probability_up, bp.direction_score_up, bp.predicted_label
        FROM backtest_predictions bp
        JOIN backtest_runs        br USING (backtest_run_id)
        ORDER BY br.model_phase ASC, bp.horizon_days ASC, bp.head ASC,
                 bp.model ASC, bp.prediction_date ASC
        """
    ).fetchall()

    bundles: dict[str, dict] = {}
    for r in rows:
        phase = r["model_phase"]
        bundle = bundles.setdefault(phase, {
            "model_phase": phase,
            "frozen_utc":  r["frozen_utc"],
            "mode":        r["mode"],
            "predictions": [],
        })
        bundle["predictions"].append(_drop_none({
            "horizon_days":     r["horizon_days"],
            "head":             r["head"],
            "model":            r["model"],
            "prediction_date":  r["prediction_date"],
            "target_date":      r["target_date"],
            "fold_index":       r["fold_index"],
            "reference_close":  r["reference_close"],
            "actual_close":     r["actual_close"],
            "actual_return":    r["actual_return"],
            "actual_label":     r["actual_label"],
            "predicted_return": r["predicted_return"],
            "predicted_close":  r["predicted_close"],
            "probability_up":   r["probability_up"],
            "direction_score_up": r["direction_score_up"],
            "predicted_label":  r["predicted_label"],
        }))
    return bundles


def longrun_evaluation_metadata(phases: dict[str, dict]) -> dict[str, object]:
    """Describe the newest stored OOF phase without confusing export time for eval time."""
    if not phases:
        return {
            "evaluation_generated_at": None,
            "evaluated_through_by_horizon": {},
        }
    newest = max(
        phases.values(),
        key=lambda bundle: str(bundle.get("frozen_utc") or ""),
    )
    evaluated_through: dict[str, str] = {}
    for horizon, points in (newest.get("points") or {}).items():
        target_dates = [
            str(point["target_date"])
            for point in points
            if isinstance(point, dict) and point.get("target_date")
        ]
        if target_dates:
            evaluated_through[str(horizon)] = max(target_dates)
    return {
        "evaluation_generated_at": newest.get("frozen_utc"),
        "evaluated_through_by_horizon": evaluated_through,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = connect(args.db)
    try:
        versions        = export_versions(conn)
        headline        = export_headline_table(conn)
        matrix          = export_full_matrix(conn)
        predictions     = export_predictions_by_phase(conn)
        longrun_history = export_longrun_history(conn)
    finally:
        conn.close()

    now_iso = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    payload_versions = {
        "generated_at": now_iso,
        "count": len(versions),
        "versions": versions,
    }
    # Include a lightweight manifest of which phases have prediction JSONs
    # available, so the frontend can lazy-load only the ones it needs.
    prediction_manifest = [
        {
            "model_phase": phase,
            "frozen_utc": bundle.get("frozen_utc"),
            "mode":       bundle.get("mode"),
            "row_count":  len(bundle.get("predictions") or []),
            "file":       f"backtest_predictions_{phase}.json",
        }
        for phase, bundle in sorted(predictions.items())
    ]
    payload_backtest = {
        "generated_at": now_iso,
        "headline_count": len(headline),
        "matrix_count":   len(matrix),
        "headline": headline,   # best reg+cls per (phase, horizon)
        "matrix":   matrix,     # every metric row
        "prediction_phases": prediction_manifest,
    }

    payload_longrun = {
        "schema_version": 2,
        "generated_at": now_iso,
        "export_generated_at": now_iso,
        **longrun_evaluation_metadata(longrun_history),
        "chart_model":  LONGRUN_CHART_MODEL,
        "raw_chart_model":  LONGRUN_CHART_MODEL,
        "chart_selection": "fallback_to_no_change_when_point_rmse_has_no_edge",
        "phases":       longrun_history,
    }

    (out_dir / "backtest_versions.json").write_text(
        json.dumps(payload_versions, indent=2, default=str), encoding="utf-8")
    (out_dir / "backtest.json").write_text(
        json.dumps(payload_backtest, indent=2, default=str), encoding="utf-8")
    # Minified — this file is consumed by the chart in index.html, never read
    # by humans. indent=2 would triple the payload for no benefit.
    (out_dir / "backtest_longrun_history.json").write_text(
        json.dumps(payload_longrun, separators=(",", ":"), default=str), encoding="utf-8")

    # Per-phase prediction payloads. Splitting by phase avoids a single
    # multi-MB JSON — longrun OOF freezes can have thousands of rows.
    prediction_files: list[Path] = []
    for phase, bundle in sorted(predictions.items()):
        file_path = out_dir / f"backtest_predictions_{phase}.json"
        payload = {
            "generated_at": now_iso,
            **bundle,
            "row_count": len(bundle.get("predictions") or []),
        }
        file_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        prediction_files.append(file_path)

    print(f"[export_backtest_json] wrote:")
    print(f"  {out_dir / 'backtest_versions.json'}          versions={len(versions)}")
    print(f"  {out_dir / 'backtest.json'}                   headline_rows={len(headline)} matrix_rows={len(matrix)}")
    total_longrun_points = sum(
        len(pts) for bundle in longrun_history.values() for pts in bundle["points"].values()
    )
    print(f"  {out_dir / 'backtest_longrun_history.json'}   phases={len(longrun_history)} points={total_longrun_points}")
    for fp in prediction_files:
        rows = len(predictions[fp.stem.replace('backtest_predictions_', '')].get('predictions') or [])
        print(f"  {fp}  prediction_rows={rows}")


if __name__ == "__main__":
    main()
