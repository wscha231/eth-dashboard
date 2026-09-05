"""Persist a completed forecast run into the SQLite DB.

Input: eth_forecast_outputs/latest_forecast_summary.csv (produced by
eth_price_forecast.save_artifacts). One row per horizon with every UI-relevant
field. This script inserts one forecast_runs row + N forecasts rows (one per
horizon) and is idempotent — if the same (input_timestamp, model_phase) tuple
already exists we skip rather than duplicate.

CLI:
    python -m forecast_site.persist_forecast \\
        --summary-csv eth_forecast_outputs/latest_forecast_summary.csv \\
        --model-phase phase4b_production \\
        --db forecast_site/predictions.db

The `--model-phase` argument names the algorithm version that produced the
file; use it to distinguish phase4b vs future phase5 rows coexisting in one DB.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import math
import sqlite3
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast_site.db import DEFAULT_DB_PATH, connect  # noqa: E402
from forecasting.daily_data import TIME_CONTRACT, iso_utc, utc_timestamp

# Columns we pull straight from the CSV into the forecasts table. The CSV has
# the same column names — this map documents the 1:1 projection.
FORECAST_CSV_COLUMNS: list[str] = [
    "time_contract", "classification_event_threshold", "classification_probability_event",
    "regression_model_name",
    "regression_selection_basis",
    "regression_predicted_return",
    "regression_predicted_close",
    "regression_lower_return_10",
    "regression_lower_close_10",
    "regression_upper_return_90",
    "regression_upper_close_90",
    "classification_model_name",
    "classification_selection_basis",
    "classification_predicted_direction",
    "classification_direction_score_up",
    "classification_probability_up",
    "classification_probability_down",
    "classification_confidence",
    "classification_signal_threshold",
    "hybrid_model_name",
    "hybrid_predicted_direction",
    "hybrid_bias_direction",
    "hybrid_predicted_market_state",
    "hybrid_predicted_reversal_signal",
    "hybrid_score",
    "hybrid_confidence",
    "hybrid_signal_tier",
    "hybrid_high_confidence_signal",
    "hybrid_signal_reason",
    "hybrid_volatility_scale",
    "hybrid_scenario_spread",
    "forecast_decision_mode",
    "forecast_actionability",
    "forecast_point_price_reliable",
    "forecast_center_return",
    "forecast_uncertainty_return",
    "forecast_lower_return",
    "forecast_upper_return",
    "active_predicted_direction",
    "active_confidence",
    "active_predicted_return",
    "active_predicted_close",
    "active_bear_predicted_return",
    "active_bear_predicted_close",
    "active_bull_predicted_return",
    "active_bull_predicted_close",
    "regime_model_name",
    "regime_predicted_state",
    "regime_probability_uptrend",
    "regime_probability_sideways",
    "regime_probability_downtrend",
    "reversal_model_name",
    "reversal_predicted_signal",
    "reversal_probability_bottom",
    "reversal_probability_top",
    "macro_readiness_status",
    "macro_risk_regime",
    "macro_directional_bias",
    "macro_selection_tag",
    "macro_risk_on_score",
    "macro_risk_off_score",
    "macro_trend_bias_score",
    "macro_volatility_stress",
    "macro_used_feature_count",
    "macro_missing_feature_count",
]

# CSV column -> forecasts column (when names differ). We only rename the
# *_model_name columns to *_model for table hygiene.
CSV_TO_DB_RENAME: dict[str, str] = {
    "regression_model_name": "regression_model",
    "classification_model_name": "classification_model",
    "hybrid_model_name": "hybrid_model",
    "regime_model_name": "regime_model",
    "reversal_model_name": "reversal_model",
}


def _nan_to_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and (value != value):  # NaN check
        return None
    # pandas may give us numpy scalars — coerce to native Python.
    if hasattr(value, "item"):
        try:
            value = value.item()
        except (AttributeError, ValueError):
            pass
    if isinstance(value, bool):
        return int(value)
    return value


def _compute_code_version(path: Path | None) -> str | None:
    if path is None or not path.exists():
        return None
    h = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    return f"sha256:{h}"


def persist_forecast(
    summary_csv: Path,
    *,
    model_phase: str,
    db_path: Path | None = None,
    code_source: Path | None = None,
    fast_mode: bool = False,
    notes: str | None = None,
    required_horizons: tuple[int, ...] | None = None,
) -> int:
    """Insert the run + its per-horizon forecasts, return the new run_id.

    Raises SystemExit with a non-zero code if the CSV is missing or malformed.
    If a run with the same (input_timestamp_utc, model_phase) already exists,
    no new rows are written and the existing run_id is returned.
    """
    if not summary_csv.exists():
        raise SystemExit(f"Summary CSV not found: {summary_csv}")

    df = pd.read_csv(summary_csv)
    if df.empty:
        raise SystemExit(f"Summary CSV is empty: {summary_csv}")
    horizons = pd.to_numeric(df["horizon_steps"], errors="raise").astype(int)
    if horizons.duplicated().any() or (required_horizons is not None and set(horizons) != set(required_horizons)):
        raise SystemExit("Summary must contain exactly one forecast for each required horizon")
    if required_horizons is not None:
        for column in ("reference_price", "regression_predicted_close"):
            values = pd.to_numeric(df[column], errors="coerce") if column in df else []
            if len(values) != len(df) or not all(pd.notna(v) and math.isfinite(v) and v > 0 for v in values):
                raise SystemExit(f"Summary requires finite positive {column} for every horizon")
        if df["reference_price"].nunique() != 1:
            raise SystemExit("Horizons have different reference prices")
        for column in ("time_contract", "model_version", "source_bar_date", "data_hash", "training_cutoff_utc"):
            if column in df and df[column].nunique(dropna=False) != 1:
                raise SystemExit(f"Horizons have different {column}")
    for _, item in df.iterrows():
        if pd.isna(item.get("forecast_target_timestamp")):
            raise SystemExit("Missing forecast target timestamp")
        if str(item.get("forecast_input_timestamp")) != str(df.iloc[0].get("forecast_input_timestamp")):
            raise SystemExit("Horizons have different input timestamps")
        if item.get("time_contract") == TIME_CONTRACT:
            origin = utc_timestamp(item["forecast_input_timestamp"])
            if utc_timestamp(item["forecast_target_timestamp"]) != origin + pd.Timedelta(days=int(item["horizon_steps"])):
                raise SystemExit("Forecast target violates the UTC bar-end contract")

    # Use row 0 to extract shared run metadata (identical across horizons).
    head = df.iloc[0]
    input_ts = str(head.get("forecast_input_timestamp", "") or "")
    if not input_ts:
        raise SystemExit("forecast_input_timestamp missing from summary CSV")
    input_ts = iso_utc(input_ts)
    ref_price = _nan_to_none(head.get("reference_price"))
    ref_source = _nan_to_none(head.get("reference_price_source"))
    ref_ts = _nan_to_none(head.get("reference_price_timestamp"))
    if ref_price is None:
        raise SystemExit("reference_price missing from summary CSV")

    run_timestamp = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    code_version = _compute_code_version(code_source)
    code_bytes = int(code_source.stat().st_size) if code_source and code_source.exists() else None

    conn = connect(db_path)
    try:
        # Idempotency check via the UNIQUE (input_timestamp_utc, model_phase) index.
        existing = conn.execute(
            "SELECT run_id FROM forecast_runs "
            "WHERE julianday(input_timestamp_utc) = julianday(?) AND model_phase = ?",
            (input_ts, model_phase),
        ).fetchone()
        if existing is not None:
            found = {int(r[0]) for r in conn.execute("SELECT horizon_days FROM forecasts WHERE run_id=?", (existing["run_id"],))}
            if found != set(horizons):
                raise SystemExit("Existing run is incomplete; refusing to silently publish it")
            contracts = {r[0] for r in conn.execute("SELECT time_contract FROM forecasts WHERE run_id=?", (existing["run_id"],))}
            if contracts != {_nan_to_none(row.get("time_contract")) for _, row in df.iterrows()}:
                raise SystemExit("Existing run uses a different time contract; use a new model phase")
            print(f"[persist] run already exists: run_id={existing['run_id']} "
                  f"input_ts={input_ts} phase={model_phase} -- skipping")
            return int(existing["run_id"])

        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            """
            INSERT INTO forecast_runs (
                run_timestamp_utc, input_timestamp_utc,
                reference_price, reference_price_source, reference_price_timestamp_utc,
                model_phase, code_version, eth_price_forecast_bytes, fast_mode, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_timestamp, input_ts, ref_price, ref_source, ref_ts,
             model_phase, code_version, code_bytes, int(fast_mode), notes),
        )
        run_id = int(cur.lastrowid)
        conn.execute("UPDATE forecast_runs SET model_version=?, source_bar_date=?, data_hash=?, training_cutoff_utc=? WHERE run_id=?",
                     tuple(_nan_to_none(head.get(c)) for c in ("model_version", "source_bar_date", "data_hash", "training_cutoff_utc")) + (run_id,))

        for _, row in df.iterrows():
            horizon = int(row["horizon_steps"])
            target_ts = iso_utc(row.get("forecast_target_timestamp"))
            if not target_ts:
                raise SystemExit(f"forecast_target_timestamp missing for horizon {horizon}")

            values: dict[str, Any] = {
                "run_id": run_id,
                "horizon_days": horizon,
                "forecast_target_timestamp_utc": target_ts,
            }
            for csv_col in FORECAST_CSV_COLUMNS:
                db_col = CSV_TO_DB_RENAME.get(csv_col, csv_col)
                values[db_col] = _nan_to_none(row.get(csv_col))

            columns = list(values.keys())
            placeholders = ",".join(["?"] * len(columns))
            col_list = ",".join(columns)
            conn.execute(
                f"INSERT INTO forecasts ({col_list}) VALUES ({placeholders})",
                [values[c] for c in columns],
            )

        conn.commit()
        print(f"[persist] inserted run_id={run_id} input_ts={input_ts} "
              f"phase={model_phase} horizons={len(df)}")
        return run_id
    finally:
        if conn.in_transaction:
            conn.rollback()
        conn.close()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", required=True, type=Path,
                        help="Path to latest_forecast_summary.csv")
    parser.add_argument("--model-phase", required=True,
                        help="Algorithm version label, e.g. phase4b_production")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--code-source", type=Path, default=None,
                        help="Path to eth_price_forecast.py for version hashing")
    parser.add_argument("--fast-mode", action="store_true",
                        help="Mark the run as fast-mode (degraded accuracy)")
    parser.add_argument("--notes", default=None)
    args = parser.parse_args(argv)

    persist_forecast(
        args.summary_csv,
        model_phase=args.model_phase,
        db_path=args.db,
        code_source=args.code_source,
        fast_mode=args.fast_mode,
        notes=args.notes,
        required_horizons=(7, 30),
    )


if __name__ == "__main__":
    main()
