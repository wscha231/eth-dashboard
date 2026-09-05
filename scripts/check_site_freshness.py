"""Verify that the deployed ETH forecast site exposes a fresh daily run."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
import urllib.error
import urllib.request
from typing import Any


def _parse_utc(value: Any) -> dt.datetime | None:
    if value in (None, ""):
        return None
    try:
        parsed = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    return parsed.astimezone(dt.timezone.utc)


def _age_hours(value: Any, now: dt.datetime) -> float | None:
    parsed = _parse_utc(value)
    return None if parsed is None else max(0.0, (now - parsed).total_seconds() / 3600.0)


def validate_health(
    payload: dict[str, Any],
    *,
    now: dt.datetime | None = None,
    max_live_age_hours: float = 27.0,
    max_health_age_hours: float = 30.0,
    expected_run_id: int | None = None,
    expected_model_version: str | None = None,
) -> list[str]:
    now = now or dt.datetime.now(tz=dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    now = now.astimezone(dt.timezone.utc)
    errors: list[str] = []

    latest_run = payload.get("latest_run") or {}
    horizons = payload.get("latest_forecasts_by_horizon") or {}
    if set(horizons) != {"7", "30"}:
        errors.append("latest run must include both required horizons: 7 and 30")
    if expected_run_id is not None and latest_run.get("run_id") != expected_run_id:
        errors.append("deployed run_id differs from the expected run")
    if expected_model_version is not None and latest_run.get("model_version") != expected_model_version:
        errors.append("deployed model_version differs from the expected model")
    for horizon, forecast in horizons.items():
        if forecast.get("input_timestamp_utc") != latest_run.get("input_timestamp_utc"):
            errors.append(f"horizon {horizon} is from a different input timestamp")
    input_timestamp = latest_run.get("input_timestamp_utc")
    live_age = _age_hours(input_timestamp, now)
    if live_age is None:
        errors.append("health payload has no usable latest input timestamp")
    elif live_age > max_live_age_hours:
        errors.append(
            f"latest input is {live_age:.2f}h old; limit is {max_live_age_hours:.2f}h"
        )

    health_age = _age_hours(payload.get("generated_at"), now)
    if health_age is None:
        errors.append("health payload has no usable generated_at timestamp")
    elif health_age > max_health_age_hours:
        errors.append(
            f"health payload is {health_age:.2f}h old; limit is {max_health_age_hours:.2f}h"
        )

    if int((payload.get("db_counts") or {}).get("forecast_runs") or 0) < 1:
        errors.append("health payload reports zero forecast runs")
    return errors


def fetch_json(url: str, *, timeout_seconds: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "Cache-Control": "no-cache"},
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("health endpoint did not return a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://etherforecast.live/health.json")
    parser.add_argument("--max-live-age-hours", type=float, default=27.0)
    parser.add_argument("--max-health-age-hours", type=float, default=30.0)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--retry-seconds", type=float, default=60.0)
    parser.add_argument("--expected-run-id", type=int)
    parser.add_argument("--expected-model-version")
    args = parser.parse_args(argv)

    attempts = max(1, args.retries)
    last_error = "unknown freshness error"
    for attempt in range(1, attempts + 1):
        try:
            payload = fetch_json(args.url, timeout_seconds=args.timeout_seconds)
            errors = validate_health(
                payload,
                max_live_age_hours=args.max_live_age_hours,
                max_health_age_hours=args.max_health_age_hours,
                expected_run_id=args.expected_run_id,
                expected_model_version=args.expected_model_version,
            )
            if not errors:
                latest = payload.get("latest_run") or {}
                print(
                    "freshness check passed: "
                    f"run_id={latest.get('run_id')} input={latest.get('input_timestamp_utc')} "
                    f"generated_at={payload.get('generated_at')}"
                )
                return 0
            last_error = "; ".join(errors)
        except (OSError, ValueError, json.JSONDecodeError, urllib.error.URLError) as exc:
            last_error = str(exc)
        print(f"freshness attempt {attempt}/{attempts} failed: {last_error}")
        if attempt < attempts:
            time.sleep(max(0.0, min(args.retry_seconds, 60.0)))

    print(f"freshness check failed: {last_error}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
