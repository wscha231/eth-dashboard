"""SQLite store for crypto/macro news headlines.

Design goals
------------
- **Idempotent ingest**: same headline fetched twice is a no-op. We hash
  ``url`` (or ``guid`` when the source provides one) and use that as the
  UNIQUE key, not title+timestamp, because titles and publish times drift
  between mirrors of the same article.
- **Date-gating is cheap**: ``published_at`` is indexed. The backtest
  harness wants "all headlines <= date X" in sub-millisecond time.
- **No ORM**: sqlite3 stdlib only. The whole schema is 2 tables + 3 indexes
  and changes rarely. An ORM here would be ceremony for its own sake.
- **Source-agnostic**: columns are generic. Per-source quirks (RSS item
  namespaces, content:encoded vs description) are normalised by the
  fetcher before insert, so the store doesn't know about CoinTelegraph
  vs Decrypt.

Schema
------
headlines
  id                INTEGER PRIMARY KEY AUTOINCREMENT
  url_sha1          TEXT    UNIQUE NOT NULL   -- sha1 of canonical URL
  source            TEXT    NOT NULL          -- "cointelegraph" | "decrypt" | ...
  title             TEXT    NOT NULL
  summary           TEXT                      -- first paragraph / RSS description
  url               TEXT    NOT NULL
  published_at      TEXT    NOT NULL          -- ISO-8601 UTC
  fetched_at        TEXT    NOT NULL
  categories        TEXT                      -- comma-joined tags from the feed

ingest_log
  id                INTEGER PRIMARY KEY AUTOINCREMENT
  source            TEXT    NOT NULL
  feed_url          TEXT    NOT NULL
  fetched_at        TEXT    NOT NULL
  items_found       INTEGER NOT NULL
  items_inserted    INTEGER NOT NULL
  http_status       INTEGER
  error             TEXT                      -- NULL on success

Run this module directly to create / upgrade the DB:
    python -m llm_analyst.news_store --db llm_analyst/news.db --init
"""
from __future__ import annotations

import argparse
import hashlib
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

DEFAULT_DB_PATH = Path("llm_analyst/news.db")

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS headlines (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    url_sha1      TEXT    UNIQUE NOT NULL,
    source        TEXT    NOT NULL,
    title         TEXT    NOT NULL,
    summary       TEXT,
    url           TEXT    NOT NULL,
    published_at  TEXT    NOT NULL,
    fetched_at    TEXT    NOT NULL,
    categories    TEXT
);

CREATE INDEX IF NOT EXISTS ix_headlines_published_at
    ON headlines(published_at);

CREATE INDEX IF NOT EXISTS ix_headlines_source_published
    ON headlines(source, published_at);

CREATE TABLE IF NOT EXISTS ingest_log (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source         TEXT    NOT NULL,
    feed_url       TEXT    NOT NULL,
    fetched_at     TEXT    NOT NULL,
    items_found    INTEGER NOT NULL,
    items_inserted INTEGER NOT NULL,
    http_status    INTEGER,
    error          TEXT
);

CREATE INDEX IF NOT EXISTS ix_ingest_log_fetched_at
    ON ingest_log(fetched_at);
"""


def canonical_url_sha1(url: str) -> str:
    """Hash the URL with light canonicalisation.

    We strip trailing slash and UTM-style query params so the same article
    fetched via different share links doesn't duplicate. We deliberately
    keep the case and path intact — news CMSes sometimes use case-sensitive
    slugs.
    """
    stripped = url.strip()
    if "?" in stripped:
        base, _, query = stripped.partition("?")
        keep = []
        for pair in query.split("&"):
            key = pair.split("=", 1)[0].lower()
            if key.startswith("utm_") or key in {"ref", "source", "ref_src"}:
                continue
            keep.append(pair)
        stripped = base + ("?" + "&".join(keep) if keep else "")
    stripped = stripped.rstrip("/")
    return hashlib.sha1(stripped.encode("utf-8")).hexdigest()


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    """Yield a connection with sensible defaults (foreign keys on, row factory)."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    """Create tables + indexes. Safe to call repeatedly."""
    with connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)


def insert_headlines(
    conn: sqlite3.Connection,
    items: Iterable[dict],
) -> int:
    """Insert headlines, skipping duplicates by url_sha1. Returns insert count."""
    inserted = 0
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    for item in items:
        if not item.get("url") or not item.get("title") or not item.get("published_at"):
            continue
        url_sha1 = canonical_url_sha1(item["url"])
        try:
            conn.execute(
                """
                INSERT INTO headlines
                    (url_sha1, source, title, summary, url, published_at,
                     fetched_at, categories)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    url_sha1,
                    item.get("source", "unknown"),
                    item["title"],
                    item.get("summary"),
                    item["url"],
                    item["published_at"],
                    item.get("fetched_at", now),
                    item.get("categories"),
                ),
            )
            inserted += 1
        except sqlite3.IntegrityError:
            # Duplicate url_sha1: not an error, just already have it.
            continue
    return inserted


def record_ingest(
    conn: sqlite3.Connection,
    source: str,
    feed_url: str,
    items_found: int,
    items_inserted: int,
    http_status: int | None = None,
    error: str | None = None,
) -> None:
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    conn.execute(
        """
        INSERT INTO ingest_log
            (source, feed_url, fetched_at, items_found, items_inserted,
             http_status, error)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (source, feed_url, now, items_found, items_inserted, http_status, error),
    )


def fetch_headlines_before(
    conn: sqlite3.Connection,
    cutoff_iso: str,
    limit: int = 30,
    sources: list[str] | None = None,
) -> list[dict]:
    """Return most recent ``limit`` headlines with ``published_at <= cutoff``.

    The backtest harness calls this to reconstruct "what did the analyst
    know on date X" without looking into the future. ``cutoff_iso`` should
    be a UTC ISO timestamp; compare is string-lex which is correct for
    ISO-8601 in UTC.
    """
    args: list = [cutoff_iso]
    where = ["published_at <= ?"]
    if sources:
        placeholders = ",".join("?" * len(sources))
        where.append(f"source IN ({placeholders})")
        args.extend(sources)
    args.append(limit)
    sql = f"""
        SELECT source, title, summary, url, published_at, categories
        FROM headlines
        WHERE {' AND '.join(where)}
        ORDER BY published_at DESC
        LIMIT ?
    """
    return [dict(row) for row in conn.execute(sql, args).fetchall()]


def stats(conn: sqlite3.Connection) -> dict:
    row = conn.execute(
        "SELECT COUNT(*) AS n, MIN(published_at) AS oldest, MAX(published_at) AS newest FROM headlines"
    ).fetchone()
    per_source = conn.execute(
        "SELECT source, COUNT(*) AS n FROM headlines GROUP BY source ORDER BY n DESC"
    ).fetchall()
    return {
        "total": row["n"],
        "oldest": row["oldest"],
        "newest": row["newest"],
        "per_source": {r["source"]: r["n"] for r in per_source},
    }


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Initialise / inspect the news store.")
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="SQLite DB path.")
    parser.add_argument("--init", action="store_true", help="Create the schema.")
    parser.add_argument("--stats", action="store_true", help="Print row counts.")
    args = parser.parse_args()
    if args.init:
        init_db(args.db)
        print(f"# initialised {args.db}")
    if args.stats:
        with connect(args.db) as conn:
            print(stats(conn))


if __name__ == "__main__":
    _cli()
