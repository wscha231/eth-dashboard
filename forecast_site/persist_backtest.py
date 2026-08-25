"""Ingest freeze_phase*.py metric JSONs into the backtest_* DB tables.

A freeze JSON looks like:
    {
        "frozen_at":   "2026-04-20T17:12:08+00:00",
        "master_data_csv":   "lake/gold/eth_master_daily.csv",
        "eth_price_forecast_bytes": 412345,
        "mode":        "phase5_production_phase4b_plus_allowlisted_phase3b",
        "cv_test_size": 60,
        "horizons": {
            "7":  {"regression_leaderboard": [...], "classification_leaderboard": [...], ...},
            "30": {...}
        },
        ...
    }

We insert one `backtest_runs` row per JSON and one `backtest_metrics` row per
leaderboard entry. Idempotent: the (model_phase, freeze_json_sha1) unique
index means re-ingesting the same JSON is a no-op, so running on the whole
`tests/phase0/*_metrics.json` glob every day is safe.

Usage:
    python -m forecast_site.persist_backtest \\
        --freeze-glob "tests/phase0/*_metrics.json" \\
        --db forecast_site/predictions.db
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from forecast_site.db import DEFAULT_DB_PATH, connect  # noqa: E402


# Regression leaderboard metric keys -> DB column names.
REGRESSION_METRIC_COLUMNS = {
    "return_mae":           "return_mae",
    "price_mae":            "price_mae",
    "price_rmse":           "price_rmse",
    "price_mape_percent":   "price_mape_percent",
    "directional_accuracy": "directional_accuracy",
}
# Classification leaderboard metric keys -> DB column names.
CLASSIFICATION_METRIC_COLUMNS = {
    "accuracy":             "accuracy",
    "balanced_accuracy":    "balanced_accuracy",
    "precision":            "precision_score",
    "recall":               "recall_score",
    "f1":                   "f1",
    "roc_auc":              "roc_auc",
    "brier_score":          "brier_score",
    "signal_threshold":     "signal_threshold",
    "threshold_validation_scheme": "threshold_validation_scheme",
    "validation_leakage_safe":     "validation_leakage_safe",
}


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    h.update(path.read_bytes())
    return h.hexdigest()


def _read_payload_and_sha1(path: Path) -> tuple[dict[str, Any], str]:
    """Read a freeze JSON once, returning ``(payload, sha1)`` derived from
    the same byte snapshot.

    Longrun checkpoints get atomically replaced mid-run via ``os.replace``;
    two independent reads could see different file versions (payload from
    v1, sha1 from v2). Using a single read closes that window."""
    raw = path.read_bytes()
    sha1 = hashlib.sha1(raw).hexdigest()
    payload = json.loads(raw.decode("utf-8"))
    return payload, sha1


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    # Swap NaN -> NULL so SQL comparisons behave (and the "nan" that shows up
    # in component_models fields when an ensemble row has no members is
    # excluded here — it's a string column handled separately).
    if math.isnan(v):
        return None
    return v


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1 if value else 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return str(value)


def _phase_from_path(path: Path) -> str:
    """Derive a stable model_phase label from the freeze JSON filename.

    Examples:
      phase5_production_metrics.json -> phase5_production
      phase4b_production_metrics.json -> phase4b_production
      baseline_metrics.json           -> baseline
      phase3b_ablation_metrics.json   -> phase3b_ablation
    """
    stem = path.stem
    return re.sub(r"_metrics$", "", stem)


def _extract_model_phase(path: Path, payload: dict[str, Any]) -> str:
    """Model-phase slug comes from the filename — one slug per freeze script,
    not per-run mode string. Keeps labels short + predictable for the UI
    ("baseline", "phase1", "phase5_production", ...). The verbose `mode`
    field is preserved separately in its own column.
    """
    del payload  # unused; slug is filename-derived for stability
    return _phase_from_path(path)


def _iter_leaderboard_rows(
    horizon: int, head: str, leaderboard: Iterable[dict[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in leaderboard or []:
        if not isinstance(row, dict):
            continue
        model = _coerce_str(row.get("model"))
        if not model:
            continue
        base = {
            "horizon_days":     horizon,
            "head":             head,
            "model":            model,
            "folds":            _coerce_float(row.get("folds")),
            "component_models": _coerce_str(row.get("component_models")),
        }
        if head == "regression":
            for src_key, db_col in REGRESSION_METRIC_COLUMNS.items():
                base[db_col] = _coerce_float(row.get(src_key))
        elif head == "classification":
            for src_key, db_col in CLASSIFICATION_METRIC_COLUMNS.items():
                raw = row.get(src_key)
                if db_col in ("threshold_validation_scheme",):
                    base[db_col] = _coerce_str(raw)
                elif db_col == "validation_leakage_safe":
                    base[db_col] = _coerce_int(raw)
                else:
                    base[db_col] = _coerce_float(raw)
            # Regression-side directional accuracy column is still useful as a
            # secondary signal for a classification row if provided.
            base["directional_accuracy"] = _coerce_float(row.get("directional_accuracy"))
        yield base


def _upsert_run(
    conn: sqlite3.Connection,
    *,
    path: Path,
    payload: dict[str, Any],
    sha1: str,
) -> int | None:
    model_phase = _extract_model_phase(path, payload)
    existing = conn.execute(
        "SELECT backtest_run_id FROM backtest_runs WHERE model_phase = ? AND freeze_json_sha1 = ?",
        (model_phase, sha1),
    ).fetchone()
    if existing is not None:
        return None  # already ingested, nothing to do

    cursor = conn.execute(
        """
        INSERT INTO backtest_runs (
            frozen_utc, model_phase, mode, code_bytes, master_data_csv,
            cv_test_size, freeze_json_path, freeze_json_sha1, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _coerce_str(payload.get("frozen_at")),
            model_phase,
            _coerce_str(payload.get("mode")),
            _coerce_int(payload.get("eth_price_forecast_bytes")),
            _coerce_str(payload.get("master_data_csv")),
            _coerce_int(payload.get("cv_test_size")),
            str(path.relative_to(PROJECT_ROOT) if path.is_absolute() and str(path).startswith(str(PROJECT_ROOT)) else path),
            sha1,
            None,
        ),
    )
    return int(cursor.lastrowid)


def _insert_metrics(
    conn: sqlite3.Connection,
    backtest_run_id: int,
    horizons: dict[str, Any],
) -> int:
    count = 0
    for horizon_key, horizon_payload in (horizons or {}).items():
        if not isinstance(horizon_payload, dict):
            continue
        try:
            horizon = int(horizon_key)
        except (ValueError, TypeError):
            continue
        for head, key in (("regression", "regression_leaderboard"),
                          ("classification", "classification_leaderboard")):
            leaderboard = horizon_payload.get(key)
            for row in _iter_leaderboard_rows(horizon, head, leaderboard):
                row_full = {"backtest_run_id": backtest_run_id, **row}
                columns = list(row_full.keys())
                placeholders = ",".join("?" for _ in columns)
                conn.execute(
                    f"INSERT OR REPLACE INTO backtest_metrics ({','.join(columns)}) VALUES ({placeholders})",
                    tuple(row_full[col] for col in columns),
                )
                count += 1
    return count


# Longrun OOF predictions carry per-date rows. Stored in backtest_predictions
# so the frontend can draw true "predicted vs actual" time-series charts.
_PREDICTION_COLUMNS = (
    "backtest_run_id", "horizon_days", "head", "model",
    "prediction_date", "target_date", "fold_index",
    "reference_close", "actual_close", "actual_return", "actual_label",
    "predicted_return", "predicted_close",
    "probability_up", "direction_score_up", "predicted_label",
)


def _insert_predictions(
    conn: sqlite3.Connection,
    backtest_run_id: int,
    horizons: dict[str, Any],
) -> int:
    """Insert per-date OOF prediction rows (longrun freezes only)."""
    count = 0
    placeholders = ",".join("?" for _ in _PREDICTION_COLUMNS)
    sql = (
        f"INSERT OR REPLACE INTO backtest_predictions "
        f"({','.join(_PREDICTION_COLUMNS)}) VALUES ({placeholders})"
    )
    for horizon_key, horizon_payload in (horizons or {}).items():
        if not isinstance(horizon_payload, dict):
            continue
        try:
            horizon = int(horizon_key)
        except (ValueError, TypeError):
            continue
        for row in horizon_payload.get("predictions") or []:
            if not isinstance(row, dict):
                continue
            model = _coerce_str(row.get("model"))
            head = _coerce_str(row.get("head"))
            prediction_date = _coerce_str(row.get("prediction_date"))
            if not (model and head and prediction_date):
                continue
            values = (
                backtest_run_id,
                horizon,
                head,
                model,
                prediction_date,
                _coerce_str(row.get("target_date")),
                _coerce_int(row.get("fold_index")),
                _coerce_float(row.get("reference_close")),
                _coerce_float(row.get("actual_close")),
                _coerce_float(row.get("actual_return")),
                _coerce_int(row.get("actual_label")),
                _coerce_float(row.get("predicted_return")),
                _coerce_float(row.get("predicted_close")),
                _coerce_float(row.get("probability_up")),
                _coerce_float(row.get("direction_score_up")),
                _coerce_int(row.get("predicted_label")),
            )
            conn.execute(sql, values)
            count += 1
    return count


def _is_longrun_file(path: Path, payload: dict[str, Any]) -> bool:
    mode = (payload.get("mode") or "").lower()
    return mode.startswith("longrun_") or "longrun_oof" in path.stem


def _refresh_longrun_run(
    conn: sqlite3.Connection,
    *, path: Path, payload: dict[str, Any], sha1: str,
) -> int:
    """For longrun freezes, always refresh: drop prior run rows for this
    phase (CASCADE nukes metrics + predictions) and insert fresh.

    Reason: each checkpoint rewrites the JSON, so sha1 changes across runs.
    Uniqueness by (phase, sha1) would pile up one row per checkpoint which
    is noise. Uniqueness by phase alone gives a clean "current longrun
    state" per phase."""
    model_phase = _extract_model_phase(path, payload)
    conn.execute("DELETE FROM backtest_runs WHERE model_phase = ?", (model_phase,))
    cursor = conn.execute(
        """
        INSERT INTO backtest_runs (
            frozen_utc, model_phase, mode, code_bytes, master_data_csv,
            cv_test_size, freeze_json_path, freeze_json_sha1, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            _coerce_str(payload.get("frozen_at")),
            model_phase,
            _coerce_str(payload.get("mode")),
            _coerce_int(payload.get("eth_price_forecast_bytes")),
            _coerce_str(payload.get("master_data_csv")),
            _coerce_int(payload.get("cv_test_size")),
            str(path.relative_to(PROJECT_ROOT) if path.is_absolute() and str(path).startswith(str(PROJECT_ROOT)) else path),
            sha1,
            f"longrun partial={payload.get('partial', True)} "
            f"folds_completed={payload.get('folds_completed', {})}",
        ),
    )
    return int(cursor.lastrowid)


def ingest_file(conn: sqlite3.Connection, path: Path) -> tuple[str, int | None, int, int]:
    """Ingest one freeze JSON.

    Returns ``(model_phase, run_id_or_None, metrics_count, predictions_count)``.
    ``run_id_or_None`` is ``None`` when the standard (non-longrun) freeze was
    already ingested under the same ``(phase, sha1)`` — a no-op by design."""
    payload, sha1 = _read_payload_and_sha1(path)
    model_phase = _extract_model_phase(path, payload)

    if _is_longrun_file(path, payload):
        # Atomic refresh: a concurrent reader never observes the
        # DELETE -> INSERT window. Essential because the daily cron runs
        # persist_backtest + export_backtest_json on the same DB; the export
        # step should never see "run deleted, no predictions yet".
        conn.execute("BEGIN IMMEDIATE")
        try:
            run_id = _refresh_longrun_run(conn, path=path, payload=payload, sha1=sha1)
            metrics = _insert_metrics(conn, run_id, payload.get("horizons", {}))
            predictions = _insert_predictions(conn, run_id, payload.get("horizons", {}))
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        return model_phase, run_id, metrics, predictions

    run_id = _upsert_run(conn, path=path, payload=payload, sha1=sha1)
    if run_id is None:
        return model_phase, None, 0, 0
    metrics = _insert_metrics(conn, run_id, payload.get("horizons", {}))
    return model_phase, run_id, metrics, 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-glob",
        default=str(PROJECT_ROOT / "tests" / "phase0" / "*_metrics.json"),
        help="Glob pattern for freeze JSON files to ingest.",
    )
    parser.add_argument(
        "--freeze-json",
        nargs="*",
        default=[],
        help="Explicit freeze JSON paths (overrides --freeze-glob when supplied).",
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH))
    args = parser.parse_args(argv)

    if args.freeze_json:
        candidates = [Path(p) for p in args.freeze_json]
    else:
        candidates = sorted(Path(p) for p in glob.glob(args.freeze_glob))

    if not candidates:
        print(f"[persist_backtest] no freeze JSONs matched {args.freeze_glob!r}")
        return

    conn = connect(args.db)
    total_new = 0
    total_metrics = 0
    total_predictions = 0
    try:
        for path in candidates:
            try:
                model_phase, run_id, metrics_count, predictions_count = ingest_file(conn, path)
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[persist_backtest]   SKIP {path.name}: {exc}")
                continue
            if run_id is None:
                print(f"[persist_backtest]   SKIP {path.name}: already ingested (phase={model_phase})")
                continue
            total_new += 1
            total_metrics += metrics_count
            total_predictions += predictions_count
            pred_suffix = f"  predictions={predictions_count}" if predictions_count else ""
            print(
                f"[persist_backtest]   INGEST {path.name}  phase={model_phase}  "
                f"run_id={run_id}  metrics={metrics_count}{pred_suffix}"
            )
    finally:
        conn.close()

    print(
        f"[persist_backtest] summary: {total_new} new runs, "
        f"{total_metrics} metric rows, {total_predictions} prediction rows, "
        f"db={args.db}"
    )


if __name__ == "__main__":
    main()
