"""SQLite connection helper + schema initialiser.

The schema lives in schema.sql alongside this module — always loaded from disk
so we don't duplicate the DDL in Python strings. The single public entry point
is `connect`, which opens the DB, enables foreign keys + WAL, and runs
schema.sql idempotently.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_PATH = Path(__file__).with_name("schema.sql")
DEFAULT_DB_PATH = Path(__file__).parent / "predictions.db"
FORECAST_OPTIONAL_COLUMNS: dict[str, str] = {
    "forecast_decision_mode": "TEXT",
    "forecast_actionability": "TEXT",
    "forecast_point_price_reliable": "INTEGER",
    "forecast_center_return": "REAL",
    "forecast_uncertainty_return": "REAL",
    "forecast_lower_return": "REAL",
    "forecast_upper_return": "REAL",
}


def connect(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Return a sqlite3.Connection with schema applied and sensible pragmas.

    - foreign_keys=ON so ON DELETE CASCADE works.
    - journal_mode=WAL so concurrent reads (e.g. website) don't block writes
      (cron job).
    - Row factory returns sqlite3.Row for dict-style access.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, isolation_level=None, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA journal_mode = WAL;")
    _init_schema(conn)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(ddl)
    _ensure_optional_columns(conn)


def _ensure_optional_columns(conn: sqlite3.Connection) -> None:
    """Add new nullable columns when an existing committed DB predates schema.sql."""
    existing = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(forecasts)").fetchall()
    }
    for column, column_type in FORECAST_OPTIONAL_COLUMNS.items():
        if column not in existing:
            conn.execute(f"ALTER TABLE forecasts ADD COLUMN {column} {column_type}")


def table_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    return [r["name"] for r in rows]


def schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT version FROM schema_version ORDER BY version DESC LIMIT 1").fetchone()
    return int(row["version"]) if row else 0
