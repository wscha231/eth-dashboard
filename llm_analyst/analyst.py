"""LLM-based market analyst: snapshot + headlines -> structured view.

Architecture
------------
1. Caller passes ``as_of_date``. We build the numeric snapshot
   (llm_analyst.snapshot) and pull headlines up to that date
   (llm_analyst.news_store).
2. We assemble a prompt that:
    - Defines the analyst role and the output contract
    - Gives the snapshot in Markdown
    - Lists N most recent headlines
    - Demands strict JSON on one tightly-specified schema
3. We call Anthropic's Messages API via plain HTTPS (no SDK dependency).
4. We parse the JSON, validate every required field, and persist the
   result to a SQLite table so we can accumulate a track record that
   mirrors the backtest archive for the ML models.

API choice
----------
Anthropic messages API: v1. The model is parametric — default is
``claude-sonnet-4-5-20250929`` but any Claude 3.5+ equivalent works.
We never set temperature > 0: for a research-signal product we want
the same inputs to produce the same view every time (reproducibility
beats creativity here).

Prompt discipline
-----------------
- **No free-form text in output.** Strict JSON only, or we reject and
  retry once. LLMs will chat if you let them.
- **Confidence calibration guidance.** We tell the model that 30-day
  crypto moves are near-random; low confidence + "flat" is an
  acceptable answer and preferable to a guessed direction.
- **Explicit "what would change my view"** forces the model to name
  the killing conditions upfront — useful for auditing, and for us
  humans to learn what signals it's watching.
- **Hard cap on expected-range magnitudes.** Without this, the model
  sometimes writes ±50% ranges which are technically correct but
  useless for a chart.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import requests

from llm_analyst import news_store, snapshot

DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_API_VERSION = "2023-06-01"
DEFAULT_DB_PATH = Path("llm_analyst/analyst_views.db")
# More headlines = more context but more tokens (and more noise). 20 keeps
# cost low and focuses the model on what's fresh. For historical backtest
# we may want to bump to 30-40.
DEFAULT_HEADLINE_LIMIT = 20
DEFAULT_MAX_OUTPUT_TOKENS = 1500

VIEW_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS analyst_views (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    as_of_date           TEXT    NOT NULL,       -- YYYY-MM-DD
    horizon_days         INTEGER NOT NULL,
    model                TEXT    NOT NULL,
    created_at           TEXT    NOT NULL,
    direction            TEXT    NOT NULL CHECK (direction IN ('up','down','flat')),
    confidence           REAL    NOT NULL CHECK (confidence BETWEEN 0.0 AND 1.0),
    expected_return_low  REAL,                    -- fraction, e.g. -0.15 = -15%
    expected_return_high REAL,
    thesis               TEXT    NOT NULL,
    key_risks            TEXT    NOT NULL,        -- JSON array
    would_change_view_if TEXT    NOT NULL,        -- JSON array
    raw_response_json    TEXT    NOT NULL,        -- full LLM output, for audit
    prompt_tokens        INTEGER,
    completion_tokens    INTEGER,
    eth_close_at_as_of   REAL,                    -- so backtest can score without re-loading the CSV
    headlines_count      INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_analyst_views_as_of
    ON analyst_views(as_of_date, horizon_days);

CREATE INDEX IF NOT EXISTS ix_analyst_views_model
    ON analyst_views(model, as_of_date);
"""


# -----------------------------------------------------------------------------
# Persistence
# -----------------------------------------------------------------------------


@contextmanager
def connect(db_path: Path | str = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path | str = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as conn:
        conn.executescript(VIEW_SCHEMA_SQL)


# -----------------------------------------------------------------------------
# Prompt assembly
# -----------------------------------------------------------------------------


SYSTEM_PROMPT = """You are a disciplined crypto market analyst. Your job is to \
form a directional view on ETH over a fixed forward horizon, given a market \
snapshot and recent news headlines.

Output rules:
- Reply with a **single JSON object**, no prose, no code fences.
- All required fields MUST be present.
- Be honest about uncertainty: if signals are mixed or the market looks \
random at this horizon, set direction to "flat" and confidence low. \
"I don't know" is a correct answer. Overconfidence hurts the track record.
- Keep `thesis` short (2-4 sentences), name the dominant drivers.
- `key_risks` and `would_change_view_if` must be concrete, observable events \
or thresholds — not vague ("market sentiment shifts").
- `expected_return_low` / `expected_return_high` are ETH 30d return fractions. \
Cap: [-0.50, +0.50]. They should bracket a rough 60–70% confidence range, \
NOT worst-case.
"""


OUTPUT_CONTRACT = """Required JSON schema:
{
  "direction": "up" | "down" | "flat",
  "confidence": number in [0.0, 1.0],
  "thesis": string,
  "key_risks": [string, ...]           // 2-4 items
  "would_change_view_if": [string, ...] // 2-4 items
  "expected_return_low":  number in [-0.50, 0.50],
  "expected_return_high": number in [-0.50, 0.50]
}
Nothing else. No markdown. No trailing commentary.
"""


def format_headlines_for_prompt(items: list[dict], max_len_chars: int = 180) -> str:
    """Render headlines as a terse bullet list the LLM can scan."""
    if not items:
        return "_(no recent headlines available)_"
    lines = []
    for item in items:
        published = item.get("published_at", "")[:10]  # YYYY-MM-DD only
        title = (item.get("title") or "").strip()[:max_len_chars]
        source = item.get("source", "?")
        lines.append(f"- [{published} · {source}] {title}")
    return "\n".join(lines)


def build_user_prompt(
    snap: snapshot.MarketSnapshot,
    headlines: list[dict],
    horizon_days: int,
) -> str:
    return (
        f"Form a **{horizon_days}-day forward view on ETH/USD** as of "
        f"{snap.as_of_date}. Use ONLY the information below; do not assume "
        f"anything about events after {snap.as_of_date}.\n\n"
        f"## Market snapshot\n\n"
        f"{snapshot.snapshot_to_markdown(snap)}\n\n"
        f"## Recent headlines (most recent first, up to {len(headlines)})\n\n"
        f"{format_headlines_for_prompt(headlines)}\n\n"
        f"## Output\n\n"
        f"{OUTPUT_CONTRACT}"
    )


# -----------------------------------------------------------------------------
# LLM call
# -----------------------------------------------------------------------------


@dataclass
class AnalystView:
    as_of_date: str
    horizon_days: int
    model: str
    direction: str
    confidence: float
    thesis: str
    key_risks: list[str]
    would_change_view_if: list[str]
    expected_return_low: float
    expected_return_high: float
    raw_response: dict = field(default_factory=dict)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    eth_close_at_as_of: float | None = None
    headlines_count: int = 0

    def to_persistable(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "horizon_days": self.horizon_days,
            "model": self.model,
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "direction": self.direction,
            "confidence": self.confidence,
            "expected_return_low": self.expected_return_low,
            "expected_return_high": self.expected_return_high,
            "thesis": self.thesis,
            "key_risks": json.dumps(self.key_risks, ensure_ascii=False),
            "would_change_view_if": json.dumps(self.would_change_view_if, ensure_ascii=False),
            "raw_response_json": json.dumps(self.raw_response, ensure_ascii=False),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "eth_close_at_as_of": self.eth_close_at_as_of,
            "headlines_count": self.headlines_count,
        }


class AnalystError(Exception):
    """Raised when the LLM output fails schema validation after retries."""


def call_anthropic(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    api_key: str,
    api_url: str = DEFAULT_API_URL,
    api_version: str = DEFAULT_API_VERSION,
    max_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    timeout: float = 60.0,
) -> dict:
    """Call Anthropic messages API and return the parsed JSON body."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": api_version,
        "content-type": "application/json",
    }
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
    if response.status_code >= 400:
        raise AnalystError(
            f"Anthropic API {response.status_code}: {response.text[:500]}"
        )
    return response.json()


def _extract_text(body: dict) -> str:
    """Pull the text from Anthropic's Messages API response.

    Structure: ``body["content"]`` is a list of blocks; we concatenate all
    ``type == "text"`` blocks. Non-text blocks (tool use, etc.) are ignored.
    """
    blocks = body.get("content") or []
    chunks: list[str] = []
    for block in blocks:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if text:
                chunks.append(text)
    return "".join(chunks).strip()


def _parse_json_strict(raw: str) -> dict:
    """Parse a JSON object from the LLM response.

    We tolerate one common failure: the model wrapping JSON in ``` fences.
    Anything else fails loudly — no silent truncation, no "just take the
    first {" heuristics. If the LLM can't follow the contract, we retry or
    escalate. Silent parsing = silent data corruption.
    """
    text = raw.strip()
    if text.startswith("```"):
        # Strip leading/trailing fences (```json ... ``` pattern).
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip("`").strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AnalystError(f"LLM did not return valid JSON: {exc}\nRaw: {raw[:400]}") from exc
    if not isinstance(parsed, dict):
        raise AnalystError(f"LLM returned non-object JSON: {type(parsed).__name__}")
    return parsed


def _validate_view_payload(payload: dict) -> None:
    required = {
        "direction", "confidence", "thesis",
        "key_risks", "would_change_view_if",
        "expected_return_low", "expected_return_high",
    }
    missing = required - set(payload.keys())
    if missing:
        raise AnalystError(f"View JSON missing fields: {sorted(missing)}")
    if payload["direction"] not in {"up", "down", "flat"}:
        raise AnalystError(f"Invalid direction: {payload['direction']!r}")
    try:
        conf = float(payload["confidence"])
    except (TypeError, ValueError) as exc:
        raise AnalystError(f"Confidence must be number: {payload['confidence']!r}") from exc
    if not 0.0 <= conf <= 1.0:
        raise AnalystError(f"Confidence out of range [0,1]: {conf}")
    for field_name in ("expected_return_low", "expected_return_high"):
        try:
            value = float(payload[field_name])
        except (TypeError, ValueError) as exc:
            raise AnalystError(f"{field_name} not numeric: {payload[field_name]!r}") from exc
        if not -0.60 <= value <= 0.60:
            raise AnalystError(f"{field_name} out of sanity range: {value}")
    if float(payload["expected_return_low"]) > float(payload["expected_return_high"]):
        raise AnalystError("expected_return_low > expected_return_high")
    for list_field in ("key_risks", "would_change_view_if"):
        value = payload[list_field]
        if not isinstance(value, list) or not all(isinstance(x, str) for x in value):
            raise AnalystError(f"{list_field} must be list[str]")


# -----------------------------------------------------------------------------
# Orchestration
# -----------------------------------------------------------------------------


def request_view(
    as_of_date: str,
    *,
    horizon_days: int = 30,
    master_csv: Path | str = snapshot.DEFAULT_MASTER_CSV,
    news_db_path: Path | str = news_store.DEFAULT_DB_PATH,
    headline_limit: int = DEFAULT_HEADLINE_LIMIT,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    api_url: str = DEFAULT_API_URL,
    dry_run: bool = False,
) -> AnalystView | dict:
    """Produce an ``AnalystView`` for ``as_of_date``.

    ``dry_run=True`` returns the assembled prompt dict instead of calling
    the API — useful for prompt iteration without burning tokens.
    """
    snap = snapshot.build_snapshot(as_of_date, master_csv=master_csv)
    with news_store.connect(news_db_path) as conn:
        # News uses UTC timestamps; gate at end-of-day of as_of_date so a
        # headline published 2024-08-01 22:00 UTC is visible for a
        # 2024-08-01 snapshot but a headline published 2024-08-02 01:00
        # UTC is not.
        cutoff_iso = f"{snap.as_of_date}T23:59:59+00:00"
        headlines = news_store.fetch_headlines_before(
            conn, cutoff_iso, limit=headline_limit
        )

    user_prompt = build_user_prompt(snap, headlines, horizon_days=horizon_days)
    eth_close_at_as_of = snap.price.get("eth_close")

    if dry_run:
        return {
            "as_of_date": snap.as_of_date,
            "horizon_days": horizon_days,
            "model": model,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "headline_count": len(headlines),
            "eth_close_at_as_of": eth_close_at_as_of,
        }

    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise AnalystError(
            "No Anthropic API key: set ANTHROPIC_API_KEY or pass api_key=..."
        )

    body = call_anthropic(
        SYSTEM_PROMPT,
        user_prompt,
        model=model,
        api_key=api_key,
        api_url=api_url,
    )
    raw_text = _extract_text(body)
    parsed = _parse_json_strict(raw_text)
    _validate_view_payload(parsed)

    usage = body.get("usage") or {}
    view = AnalystView(
        as_of_date=snap.as_of_date,
        horizon_days=horizon_days,
        model=model,
        direction=parsed["direction"],
        confidence=float(parsed["confidence"]),
        thesis=parsed["thesis"],
        key_risks=list(parsed["key_risks"]),
        would_change_view_if=list(parsed["would_change_view_if"]),
        expected_return_low=float(parsed["expected_return_low"]),
        expected_return_high=float(parsed["expected_return_high"]),
        raw_response=body,
        prompt_tokens=usage.get("input_tokens"),
        completion_tokens=usage.get("output_tokens"),
        eth_close_at_as_of=eth_close_at_as_of,
        headlines_count=len(headlines),
    )
    return view


def persist_view(view: AnalystView, db_path: Path | str = DEFAULT_DB_PATH) -> int:
    """Insert the view into the local track-record DB and return its id."""
    init_db(db_path)
    payload = view.to_persistable()
    with connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO analyst_views (
                as_of_date, horizon_days, model, created_at,
                direction, confidence,
                expected_return_low, expected_return_high,
                thesis, key_risks, would_change_view_if, raw_response_json,
                prompt_tokens, completion_tokens, eth_close_at_as_of, headlines_count
            ) VALUES (
                :as_of_date, :horizon_days, :model, :created_at,
                :direction, :confidence,
                :expected_return_low, :expected_return_high,
                :thesis, :key_risks, :would_change_view_if, :raw_response_json,
                :prompt_tokens, :completion_tokens, :eth_close_at_as_of, :headlines_count
            )
            """,
            payload,
        )
        return int(cursor.lastrowid)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _print_view(view: AnalystView) -> None:
    print(f"# AnalystView as_of={view.as_of_date}  horizon={view.horizon_days}d  model={view.model}")
    print(f"  direction: {view.direction}  confidence: {view.confidence:.2f}")
    print(f"  expected 30d return band: "
          f"[{view.expected_return_low*100:+.1f}%, {view.expected_return_high*100:+.1f}%]")
    print(f"  thesis   : {view.thesis}")
    if view.key_risks:
        print(f"  risks    :")
        for r in view.key_risks:
            print(f"    - {r}")
    if view.would_change_view_if:
        print(f"  would change view if:")
        for r in view.would_change_view_if:
            print(f"    - {r}")
    if view.prompt_tokens is not None:
        print(f"  tokens   : prompt={view.prompt_tokens} completion={view.completion_tokens}")


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Ask the LLM analyst for an ETH view.")
    parser.add_argument("--as-of", required=True, help="Snapshot date YYYY-MM-DD.")
    parser.add_argument("--horizon", type=int, default=30, help="Forward horizon in days.")
    parser.add_argument("--csv", default=str(snapshot.DEFAULT_MASTER_CSV))
    parser.add_argument("--news-db", default=str(news_store.DEFAULT_DB_PATH))
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH),
                        help="Where to persist the analyst view.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--headlines", type=int, default=DEFAULT_HEADLINE_LIMIT)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the assembled prompt and exit (no API call).")
    parser.add_argument("--no-persist", action="store_true",
                        help="Run the API call but don't write to the view DB.")
    args = parser.parse_args()

    result = request_view(
        args.as_of,
        horizon_days=args.horizon,
        master_csv=args.csv,
        news_db_path=args.news_db,
        headline_limit=args.headlines,
        model=args.model,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        assert isinstance(result, dict)
        print(f"# dry-run prompt preview  as_of={result['as_of_date']}  "
              f"headlines={result['headline_count']}")
        print("\n--- system ---")
        print(result["system_prompt"])
        print("\n--- user ---")
        print(result["user_prompt"])
        return

    assert isinstance(result, AnalystView)
    _print_view(result)
    if not args.no_persist:
        row_id = persist_view(result, db_path=args.db)
        print(f"  persisted: {args.db} id={row_id}")


if __name__ == "__main__":
    _cli()
