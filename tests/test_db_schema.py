"""Tests for forecast_site.db — schema + connection bootstrapping.

The DDL is idempotent (CREATE TABLE IF NOT EXISTS + INSERT OR IGNORE),
so these tests are light: just pin the contract that every table the
pipeline expects exists after ``connect()``, and that re-opening is
safe (the daily cron + the webserver both connect to the same file).
"""
from __future__ import annotations

import pytest

from forecast_site import db as dbmod


EXPECTED_TABLES = {
    # v1 — live forecast pipeline
    "forecast_runs",
    "forecasts",
    "actuals",
    "accuracy_snapshot",
    # v2 — backtest tables
    "backtest_runs",
    "backtest_metrics",
    # v3 — per-date OOF predictions
    "backtest_predictions",
    # metadata
    "schema_version",
}


def test_connect_creates_all_expected_tables(temp_db_path):
    conn = dbmod.connect(temp_db_path)
    try:
        present = set(dbmod.table_names(conn))
    finally:
        conn.close()
    missing = EXPECTED_TABLES - present
    assert not missing, f"schema.sql did not create: {missing}"


def test_schema_version_advances_to_latest(temp_db_path):
    """schema_version table should record v4 as the highest applied."""
    conn = dbmod.connect(temp_db_path)
    try:
        assert dbmod.schema_version(conn) == 4
    finally:
        conn.close()


def test_reconnect_is_idempotent(temp_db_path):
    """Opening the same DB twice (cron + webserver case) must not fail
    and must not duplicate schema_version rows."""
    conn_a = dbmod.connect(temp_db_path)
    try:
        first_count = conn_a.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    finally:
        conn_a.close()

    # Second connect replays schema.sql — INSERT OR IGNORE should keep
    # the count stable.
    conn_b = dbmod.connect(temp_db_path)
    try:
        second_count = conn_b.execute("SELECT COUNT(*) FROM schema_version").fetchone()[0]
    finally:
        conn_b.close()

    assert first_count == second_count == 4


def test_direction_score_columns_exist(temp_db_path):
    conn = dbmod.connect(temp_db_path)
    try:
        backtest_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(backtest_predictions)").fetchall()
        }
        forecast_columns = {
            str(row["name"])
            for row in conn.execute("PRAGMA table_info(forecasts)").fetchall()
        }
    finally:
        conn.close()

    assert "direction_score_up" in backtest_columns
    assert "classification_direction_score_up" in forecast_columns


def test_reconnect_migrates_legacy_direction_score_columns(temp_db_path):
    legacy = dbmod.connect(temp_db_path)
    try:
        legacy.execute("ALTER TABLE backtest_predictions DROP COLUMN direction_score_up")
        legacy.execute("ALTER TABLE forecasts DROP COLUMN classification_direction_score_up")
    finally:
        legacy.close()

    migrated = dbmod.connect(temp_db_path)
    try:
        backtest_columns = {
            str(row["name"])
            for row in migrated.execute("PRAGMA table_info(backtest_predictions)").fetchall()
        }
        forecast_columns = {
            str(row["name"])
            for row in migrated.execute("PRAGMA table_info(forecasts)").fetchall()
        }
    finally:
        migrated.close()

    assert "direction_score_up" in backtest_columns
    assert "classification_direction_score_up" in forecast_columns


def test_foreign_key_cascade_predictions_follow_runs(temp_db_path):
    """Per-phase ON DELETE CASCADE: dropping a backtest_runs row must
    take the dependent backtest_metrics + backtest_predictions with it.

    Critical for longrun refresh semantics — ``_refresh_longrun_run``
    relies on CASCADE to clean prior metrics/predictions when it
    deletes the run."""
    conn = dbmod.connect(temp_db_path)
    try:
        cur = conn.execute("""
            INSERT INTO backtest_runs (frozen_utc, model_phase, mode, freeze_json_sha1)
            VALUES ('2026-01-01T00:00:00+00:00', 'phase_fk', 'test_mode', 'sha_xyz')
        """)
        run_id = cur.lastrowid
        conn.execute("""
            INSERT INTO backtest_metrics (backtest_run_id, horizon_days, head, model, price_rmse)
            VALUES (?, 7, 'regression', 'foo', 100.0)
        """, (run_id,))
        conn.execute("""
            INSERT INTO backtest_predictions
                (backtest_run_id, horizon_days, head, model,
                 prediction_date, target_date, fold_index,
                 reference_close, predicted_close)
            VALUES (?, 7, 'regression', 'foo', '2024-01-01', '2024-01-08', 0,
                    2000.0, 2050.0)
        """, (run_id,))

        # Sanity: rows exist.
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_metrics WHERE backtest_run_id = ?", (run_id,),
        ).fetchone()[0] == 1
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_predictions WHERE backtest_run_id = ?", (run_id,),
        ).fetchone()[0] == 1

        # Delete the parent run — CASCADE should sweep the children.
        conn.execute("DELETE FROM backtest_runs WHERE backtest_run_id = ?", (run_id,))

        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_metrics WHERE backtest_run_id = ?", (run_id,),
        ).fetchone()[0] == 0
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_predictions WHERE backtest_run_id = ?", (run_id,),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_unique_index_phase_sha1_enforced(temp_db_path):
    """Non-longrun idempotency relies on (model_phase, freeze_json_sha1)
    being unique. Inserting a duplicate should raise IntegrityError."""
    import sqlite3
    conn = dbmod.connect(temp_db_path)
    try:
        conn.execute("""
            INSERT INTO backtest_runs (frozen_utc, model_phase, freeze_json_sha1)
            VALUES ('2026-01-01T00:00:00+00:00', 'phase_u', 'sha_abc')
        """)
        # Duplicate (phase, sha) → must fail.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("""
                INSERT INTO backtest_runs (frozen_utc, model_phase, freeze_json_sha1)
                VALUES ('2026-01-01T00:00:00+00:00', 'phase_u', 'sha_abc')
            """)
        # Different sha → OK.
        conn.execute("""
            INSERT INTO backtest_runs (frozen_utc, model_phase, freeze_json_sha1)
            VALUES ('2026-01-02T00:00:00+00:00', 'phase_u', 'sha_def')
        """)
        assert conn.execute(
            "SELECT COUNT(*) FROM backtest_runs WHERE model_phase = 'phase_u'"
        ).fetchone()[0] == 2
    finally:
        conn.close()
