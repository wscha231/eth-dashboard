from __future__ import annotations

import datetime as dt

from scripts.check_site_freshness import validate_health


NOW = dt.datetime(2026, 8, 23, 4, 30, tzinfo=dt.timezone.utc)


def _health(*, input_timestamp: str, generated_at: str, runs: int = 87) -> dict:
    return {
        "generated_at": generated_at,
        "latest_run": {"run_id": runs, "input_timestamp_utc": input_timestamp},
        "db_counts": {"forecast_runs": runs},
    }


def test_validate_health_accepts_fresh_deployment() -> None:
    payload = _health(
        input_timestamp="2026-08-23T00:00:00+00:00",
        generated_at="2026-08-23T03:30:00+00:00",
    )

    assert validate_health(payload, now=NOW) == []


def test_validate_health_rejects_stale_input_even_if_health_was_regenerated() -> None:
    payload = _health(
        input_timestamp="2026-08-21T00:00:00+00:00",
        generated_at="2026-08-23T04:00:00+00:00",
    )

    errors = validate_health(payload, now=NOW)

    assert any("latest input" in error for error in errors)


def test_validate_health_rejects_cached_old_health_payload() -> None:
    payload = _health(
        input_timestamp="2026-08-23T00:00:00+00:00",
        generated_at="2026-08-21T00:00:00+00:00",
    )

    errors = validate_health(payload, now=NOW)

    assert any("health payload" in error for error in errors)
