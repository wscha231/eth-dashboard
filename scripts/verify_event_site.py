"""Verify the externally served release, exact issued IDs and hourly freshness."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.request import urlopen

HORIZONS_SECONDS = {h * 3600 for h in (6, 24, 72, 168, 336, 720)}
MAX_AGE = timedelta(minutes=100)
CLOCK_TOLERANCE = timedelta(minutes=2)


def timestamp(value):
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        raise ValueError("event timestamp lacks timezone")
    return result.astimezone(timezone.utc)


def fresh(value, now, label):
    result = timestamp(value)
    if result - now > CLOCK_TOLERANCE:
        raise ValueError(f"future {label}")
    if now - result > MAX_AGE:
        raise ValueError(f"stale {label}")
    return result


def verify(actual, expected=None, now=None, require_ready=False):
    now=now or datetime.now(timezone.utc)
    if actual.get("schema_version")!=1:raise ValueError("event schema mismatch")
    if expected and actual.get("release_id")!=expected.get("release_id"):raise ValueError("release mismatch")
    if expected and [r["forecast_id"] for r in actual.get("current",[])]!=[r["forecast_id"] for r in expected.get("current",[])]:
        raise ValueError("issued records mismatch")
    generated = fresh(actual["generated_at"], now, "hourly worker heartbeat")
    slot = fresh(actual["expected_slot"], now, "hourly slot")
    if slot != slot.replace(minute=0, second=0, microsecond=0):
        raise ValueError("hourly slot is not aligned")
    if require_ready and actual.get("status")!="ready":raise ValueError("hourly forecast delayed")
    if actual.get("status") == "ready":
        records = actual.get("current", [])
        if len(records) != len(HORIZONS_SECONDS) or {r["horizon_seconds"] for r in records} != HORIZONS_SECONDS:
            raise ValueError("incomplete hourly horizons")
        ids = [r.get("forecast_id") for r in records]
        if not all(ids) or len(set(ids)) != len(ids):
            raise ValueError("missing or duplicate issued IDs")
    if require_ready:
        # Normal trigger is :08; allow seven minutes for queue/build/CDN propagation.
        due_slot = now.replace(minute=0, second=0, microsecond=0)
        if now.minute < 15:
            due_slot -= timedelta(hours=1)
        if slot < due_slot:
            raise ValueError("latest hourly slot missing after publication grace period")
    for record in actual.get("current",[]):
        cutoff = fresh(record["input_cutoff"], now, "hourly input")
        if cutoff != slot:raise ValueError("mixed hourly slots")
        issued = timestamp(record["issued_at"])
        if issued > generated + CLOCK_TOLERANCE:raise ValueError("issuance after publication timestamp")
        if issued >= timestamp(record["window_start"]):raise ValueError("event began before issuance")
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument("--expected");p.add_argument("--require-ready",action="store_true")
    p.add_argument("--expected-replay");args=p.parse_args()
    expected=json.loads(Path(args.expected).read_text()) if args.expected else None
    expected_replay=json.loads(Path(args.expected_replay).read_text()) if args.expected_replay else None
    for attempt in range(15 if expected else 1):
        try:
            suffix=f"?verify={int(time.time())}"
            with urlopen('https://etherforecast.live/signals.json'+suffix,timeout=12) as response:actual=json.load(response)
            with urlopen('https://etherforecast.live/'+suffix,timeout=12) as response:html=response.read().decode()
            with urlopen('https://etherforecast.live/events.js'+suffix,timeout=12) as response:js=response.read().decode()
            if 'id="event-system"' not in html or 'loadEventForecasts' not in js:raise ValueError("new event charts missing")
            if expected is not None:
                for name, served in (("index.html", html), ("events.js", js)):
                    source = Path("forecast_site/public", name).read_text()
                    if source != served:raise ValueError("published interface differs from source: " + name)
                print('Event interface verified: english-outlook-v1',
                      'html_sha256='+hashlib.sha256(html.encode()).hexdigest(),
                      'js_sha256='+hashlib.sha256(js.encode()).hexdigest())
            verify(actual,expected,require_ready=args.require_ready)
            if expected_replay is not None:
                with urlopen('https://etherforecast.live/signals_replay.json'+suffix,timeout=30) as response:actual_replay=json.load(response)
                if actual_replay != expected_replay:raise ValueError("published replay differs from evaluated result")
                for control in ('event-market-filter','event-price-unit','event-period-comparison'):
                    if f'id="{control}"' not in html or control not in js:raise ValueError("period comparison controls missing")
                with urlopen('https://etherforecast.live/signals_segments.csv'+suffix,timeout=15) as response:actual_csv=response.read()
                from tempfile import TemporaryDirectory
                from export_event_segments import export
                with TemporaryDirectory() as temp:
                    csv_path=Path(temp)/'segments.csv'
                    row_count=export(expected_replay,csv_path)
                    if actual_csv != csv_path.read_bytes():raise ValueError("published segment CSV differs from evaluated result")
                print('Event replay verified:',actual_replay['generated_at'],
                      'segmented_horizons='+str(sum(bool(r.get('segments')) for r in actual_replay['horizons'].values())),
                      'csv_rows='+str(row_count))
            print('Event site verified:',actual['release_id'],actual['status'],
                  'slot='+actual['expected_slot'], 'generated='+actual['generated_at'],
                  'horizons='+str(len(actual.get('current',[]))));return
        except Exception as exc:
            print('verification pending:',str(exc),flush=True)
            if expected and attempt<14:time.sleep(8)
    raise SystemExit("Event site did not pass publication verification")


if __name__=='__main__':main()
