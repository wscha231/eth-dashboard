"""Bounded exact-release preflight. This is not a publication/skill certificate."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen

SITE_URL = "https://etherforecast.live/signals.json"
UPSTREAM_URL = "https://raw.githubusercontent.com/wscha231/eth-dashboard/data/event-feed/forecast_site/public/signals.json"
MAX_WAIT = 420.0
MAX_RESPONSE = 16 * 1024 * 1024


def utc(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if not isinstance(dt, datetime) or dt.tzinfo is None:
        raise ValueError("timezone-aware publication clock required")
    return dt.astimezone(timezone.utc)


def fetch_json(url, timeout):
    request = Request(url, headers={"Accept": "application/json", "Cache-Control": "no-cache", "Pragma": "no-cache"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(MAX_RESPONSE + 1)
        headers = {name.lower(): response.headers.get(name) for name in
                   ("Date", "Age", "Cache-Control", "CDN-Cache-Control", "X-Vercel-Cache", "X-Cache")}
    if len(raw) > MAX_RESPONSE:
        raise ValueError("oversized public forecast payload")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("public forecast payload must be an object")
    return payload, headers


def wait_for_release(expected, *, fetch=fetch_json, clock=None, monotonic=time.monotonic,
                     sleep=time.sleep, max_wait=MAX_WAIT, interval=8.0):
    if not 0 <= max_wait <= MAX_WAIT or not math.isfinite(max_wait) or not 0 < interval <= 30:
        raise ValueError("invalid bounded publication retry budget")
    release = expected.get("release_id")
    if not isinstance(release, str) or not release or len(release) > 128:
        raise ValueError("expected release ID required")
    clock = clock or (lambda: datetime.now(timezone.utc))
    started, tick = utc(clock()), monotonic()
    starts = [utc(row["window_start"]) for row in expected.get("current", [])]
    cutoff = min(starts) if starts else None
    # Reserve time for the existing, stronger interface/archive verifier.
    def remaining():
        budget = max_wait - (monotonic() - tick)
        if cutoff is not None:
            budget = min(budget, (cutoff - utc(clock())).total_seconds() - 15)
        return budget
    report = {"schema_version": 1, "expected_release_id": release, "started_at": started.isoformat(),
              "maximum_wait_seconds": max_wait, "forecast_window_start": cutoff.isoformat() if cutoff else None,
              "status": "waiting", "attempts": [], "publication_verified": False,
              "interpretation": "Preflight only; existing exact payload, interface and archive checks still required."}
    upstream_expected = None
    for attempt in range(1, 65):
        now = utc(clock())
        budget = remaining()
        if (cutoff is not None and (cutoff - now).total_seconds() <= 15) or (attempt > 1 and budget <= 0):
            report["status"] = "forecast_deadline_reached" if cutoff is not None and (cutoff - now).total_seconds() <= 15 else "propagation_timeout"
            break
        query = "?" + urlencode({"release_probe": release, "attempt": attempt, "clock": now.timestamp()})
        observation = {"attempt": attempt, "observed_at": now.isoformat()}
        try:
            actual, headers = fetch(SITE_URL + query, timeout=max(0.1, min(12, budget if budget > 0 else 0.1)))
            observation.update(served_release_id=actual.get("release_id"), response_headers=headers,
                               exact_payload_match=actual == expected)
            if actual == expected:
                # A slow HTTP response must not push a successful probe past the deadline.
                if cutoff is not None and utc(clock()) >= cutoff:
                    report["status"] = "forecast_deadline_reached"
                elif monotonic() - tick > max_wait and max_wait != 0:
                    report["status"] = "propagation_timeout"
                else:
                    report["status"] = "matched_requires_full_verification"
                report["attempts"].append(observation)
                break
        except Exception as exc:
            observation["error"] = type(exc).__name__ + ": " + str(exc)[:300]
        # Upstream reads are diagnostic only and can NEVER substitute for the public site.
        budget = remaining()
        if (attempt == 1 or attempt % 5 == 0) and budget > 0:
            try:
                upstream, headers = fetch(UPSTREAM_URL + query, timeout=max(0.1, min(12, budget)))
                upstream_expected = upstream == expected
                observation.update(upstream_release_id=upstream.get("release_id"), upstream_headers=headers,
                                   upstream_exact_match=upstream_expected)
            except Exception as exc:
                observation["upstream_error"] = type(exc).__name__ + ": " + str(exc)[:300]
        report["attempts"].append(observation)
        if remaining() <= 0:
            report["status"] = "propagation_timeout"
            break
        sleep(min(interval, remaining()))
    if report["status"] == "waiting":
        report["status"] = "attempt_limit_reached"
    report["finished_at"] = utc(clock()).isoformat()
    report["elapsed_seconds"] = max(0.0, monotonic() - tick)
    report["diagnosis"] = ("expected_upstream_observed_but_site_did_not_match" if
                            upstream_expected and report["status"] != "matched_requires_full_verification" else
                            "site_and_upstream_observations_recorded")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--max-wait-seconds", type=float, default=MAX_WAIT)
    args = parser.parse_args()
    expected = json.loads(args.expected.read_text())
    report = wait_for_release(expected, max_wait=args.max_wait_seconds)
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({key: report[key] for key in ("status", "expected_release_id", "elapsed_seconds", "diagnosis")}), flush=True)
    if report["status"] != "matched_requires_full_verification":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
