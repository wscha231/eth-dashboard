"""Resolve past forecasts against realized ETH closes + refresh accuracy snapshots.

Run this after persist_forecast on each cron tick. It:

1. Finds forecasts whose forecast_target_timestamp_utc has passed AND are not
   yet in `actuals`.
2. Looks up the realized eth_close from the master market data CSV.
3. Writes one actuals row per resolved forecast (realized close, return,
   direction_correct, errors, brier contribution).
4. Recomputes accuracy_snapshot for each (horizon, window_days) pair.

CLI:
    python -m forecast_site.backfill_actuals \\
        --master-data-csv lake/gold/eth_master_daily.csv \\
        --db forecast_site/predictions.db

The script is idempotent: reruns with no new target dates are zero-cost.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import math
import sqlite3
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402
from forecast_site.db import DEFAULT_DB_PATH, connect  # noqa: E402

DEFAULT_MASTER_CSV = PROJECT_ROOT / "lake" / "gold" / "eth_master_daily.csv"

# Windows over which the website reports rolling accuracy. `None` means "all
# time", and is stored in the DB as window_days = 9999.
ACCURACY_WINDOWS: list[int | None] = [30, 90, 180, None]


def _parse_utc(ts_raw: str | None) -> _dt.datetime | None:
    if not ts_raw:
        return None
    try:
        parsed = pd.to_datetime(ts_raw, utc=True, errors="coerce")
    except (ValueError, TypeError):
        return None
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime()


def _load_eth_close_lookup(master_csv: Path) -> pd.Series:
    """Return a Series indexed by date (UTC midnight) of eth_close values."""
    market = efp.load_market_data_csv(master_csv)
    if market.empty or "eth_close" not in market.columns:
        raise SystemExit(f"Master dataset missing eth_close: {master_csv}")
    # Normalize index to date (some rows may be intraday but master is daily).
    close = pd.to_numeric(market["eth_close"], errors="coerce").dropna()
    close.index = pd.to_datetime(close.index, utc=True).floor("D")
    close = close[~close.index.duplicated(keep="last")]
    close = close.sort_index()
    return close


def _resolve_actual_close(close_lookup: pd.Series, target_ts: _dt.datetime) -> float | None:
    """Return realized eth_close at target_ts (rounded down to date) or None
    if the target is still in the future or the CSV has a gap.
    """
    target_date = pd.Timestamp(target_ts).tz_convert("UTC").floor("D")
    latest = close_lookup.index.max()
    if target_date > latest:
        return None
    if target_date in close_lookup.index:
        return float(close_lookup.loc[target_date])
    # Fall back to the latest close at or before target_date (covers holidays
    # in source data — rare for crypto but defensive).
    earlier = close_lookup.loc[close_lookup.index <= target_date]
    if earlier.empty:
        return None
    return float(earlier.iloc[-1])


def _compute_direction_metrics(
    predicted_direction: str | None,
    actual_return: float,
    horizon_days: int,
) -> tuple[str, int | None]:
    """Return (direction_actual, direction_correct).

    direction_correct is 1 if the string prediction matches direction_actual,
    0 if they differ AND the prediction is non-abstain (UP/DOWN), or None when
    the prediction is FLAT (treated as abstention — not counted toward the
    rolling accuracy so the model isn't penalised for correctly flagging
    "no edge").
    """
    direction_actual = efp.infer_direction_from_return_value(actual_return, horizon_days)
    if not direction_actual:
        direction_actual = "FLAT"

    pred = (predicted_direction or "").strip().upper()
    if pred in {"UP", "DOWN"}:
        return direction_actual, int(pred == direction_actual)
    # FLAT or empty -> abstention, don't count.
    return direction_actual, None


def _fetch_unresolved(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT
            f.forecast_id,
            f.horizon_days,
            f.forecast_target_timestamp_utc,
            f.regression_predicted_return,
            f.regression_predicted_close,
            f.classification_predicted_direction,
            f.classification_probability_up,
            f.active_predicted_direction,
            r.reference_price,
            r.input_timestamp_utc
        FROM forecasts f
        JOIN forecast_runs r ON f.run_id = r.run_id
        LEFT JOIN actuals a ON a.forecast_id = f.forecast_id
        WHERE a.forecast_id IS NULL
        ORDER BY f.forecast_target_timestamp_utc
        """
    ).fetchall()


def _insert_actual(
    conn: sqlite3.Connection,
    forecast_id: int,
    resolved_at: str,
    actual_close: float,
    actual_return: float,
    direction_actual: str,
    direction_correct: int | None,
    active_direction_correct: int | None,
    return_ae: float | None,
    price_ae: float | None,
    brier_contribution: float | None,
) -> None:
    conn.execute(
        """
        INSERT INTO actuals (
            forecast_id, resolved_at_utc, actual_close, actual_return,
            direction_actual, direction_correct, active_direction_correct,
            return_absolute_error, price_absolute_error, brier_contribution
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (forecast_id, resolved_at, actual_close, actual_return,
         direction_actual, direction_correct, active_direction_correct,
         return_ae, price_ae, brier_contribution),
    )


def resolve_pending_forecasts(
    conn: sqlite3.Connection,
    close_lookup: pd.Series,
) -> int:
    resolved_at = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    rows = _fetch_unresolved(conn)
    if not rows:
        return 0

    inserted = 0
    for row in rows:
        target = _parse_utc(row["forecast_target_timestamp_utc"])
        if target is None:
            continue
        actual_close = _resolve_actual_close(close_lookup, target)
        if actual_close is None:
            continue  # target still in the future

        ref_price = float(row["reference_price"])
        actual_return = (actual_close - ref_price) / ref_price if ref_price else float("nan")
        horizon = int(row["horizon_days"])

        direction_actual, dir_correct = _compute_direction_metrics(
            row["classification_predicted_direction"], actual_return, horizon,
        )
        _, active_correct = _compute_direction_metrics(
            row["active_predicted_direction"], actual_return, horizon,
        )

        pred_return = row["regression_predicted_return"]
        return_ae = (
            abs(float(pred_return) - actual_return)
            if pred_return is not None and not _is_nan(pred_return) else None
        )
        pred_close = row["regression_predicted_close"]
        price_ae = (
            abs(float(pred_close) - actual_close)
            if pred_close is not None and not _is_nan(pred_close) else None
        )

        prob_up = row["classification_probability_up"]
        brier = None
        if prob_up is not None and not _is_nan(prob_up):
            actual_up = 1 if actual_return > 0 else 0
            brier = (float(prob_up) - actual_up) ** 2

        _insert_actual(
            conn,
            forecast_id=int(row["forecast_id"]),
            resolved_at=resolved_at,
            actual_close=actual_close,
            actual_return=actual_return,
            direction_actual=direction_actual,
            direction_correct=dir_correct,
            active_direction_correct=active_correct,
            return_ae=return_ae,
            price_ae=price_ae,
            brier_contribution=brier,
        )
        inserted += 1
    conn.commit()
    return inserted


def _is_nan(value) -> bool:
    try:
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def refresh_accuracy_snapshots(conn: sqlite3.Connection) -> int:
    """Recompute rolling accuracy per (horizon, window_days) and insert a new
    snapshot row. Returns the number of snapshots inserted."""
    snapshot_utc = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    horizons = [int(r["horizon_days"]) for r in conn.execute(
        "SELECT DISTINCT horizon_days FROM forecasts ORDER BY horizon_days"
    ).fetchall()]
    inserted = 0
    for horizon in horizons:
        for window in ACCURACY_WINDOWS:
            window_days = window if window is not None else 9999
            if window is not None:
                cutoff = (_dt.datetime.now(tz=_dt.timezone.utc)
                          - _dt.timedelta(days=window)).isoformat()
                filter_clause = "AND a.resolved_at_utc >= ?"
                params: tuple = (horizon, cutoff)
            else:
                filter_clause = ""
                params = (horizon,)
            row = conn.execute(
                f"""
                SELECT
                    COUNT(*) AS n,
                    AVG(CASE WHEN a.direction_correct IS NOT NULL
                             THEN a.direction_correct END) AS dir_acc,
                    AVG(a.brier_contribution) AS brier,
                    AVG(a.price_absolute_error / NULLIF(a.actual_close, 0)) * 100 AS mape,
                    AVG(a.price_absolute_error * a.price_absolute_error) AS mse,
                    AVG(a.return_absolute_error) AS return_mae
                FROM actuals a
                JOIN forecasts f ON f.forecast_id = a.forecast_id
                WHERE f.horizon_days = ? {filter_clause}
                """,
                params,
            ).fetchone()
            if row["n"] == 0:
                continue
            rmse = math.sqrt(row["mse"]) if row["mse"] is not None else None
            conn.execute(
                """
                INSERT INTO accuracy_snapshot (
                    snapshot_utc, horizon_days, window_days, resolved_count,
                    direction_accuracy, brier_score, price_mape_percent,
                    price_rmse, return_mae
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (snapshot_utc, horizon, window_days, int(row["n"]),
                 row["dir_acc"], row["brier"], row["mape"], rmse, row["return_mae"]),
            )
            inserted += 1
    conn.commit()
    return inserted


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", type=Path, default=DEFAULT_MASTER_CSV,
                        help=f"ETH master daily CSV (default: {DEFAULT_MASTER_CSV})")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH,
                        help=f"SQLite DB path (default: {DEFAULT_DB_PATH})")
    parser.add_argument("--skip-snapshots", action="store_true",
                        help="Only backfill actuals, skip accuracy_snapshot refresh")
    args = parser.parse_args(argv)

    close_lookup = _load_eth_close_lookup(args.master_data_csv)
    conn = connect(args.db)
    try:
        resolved = resolve_pending_forecasts(conn, close_lookup)
        print(f"[backfill] newly resolved forecasts: {resolved}")
        if not args.skip_snapshots:
            snapshots = refresh_accuracy_snapshots(conn)
            print(f"[backfill] inserted accuracy snapshots: {snapshots}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
