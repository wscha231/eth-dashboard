"""Verify the externally served release, exact issued IDs and hourly freshness."""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from urllib.request import urlopen

HORIZONS_SECONDS = {h * 3600 for h in (6, 24, 72, 168, 336, 720)}
MAX_AGE = timedelta(minutes=100)
CLOCK_TOLERANCE = timedelta(minutes=2)


def timestamp(value):
    try:
        result = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("invalid event timestamp") from exc
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
    if not isinstance(actual.get("release_id"), str) or not actual["release_id"]:
        raise ValueError("missing release ID")
    if expected and actual.get("release_id")!=expected.get("release_id"):raise ValueError("release mismatch")
    if expected and [r["forecast_id"] for r in actual.get("current",[])]!=[r["forecast_id"] for r in expected.get("current",[])]:
        raise ValueError("issued records mismatch")
    if expected is not None and actual != expected:
        raise ValueError("published payload differs from expected release")
    records = actual.get("current")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("invalid current forecast list")
    generated = fresh(actual["generated_at"], now, "hourly worker heartbeat")
    slot = fresh(actual["expected_slot"], now, "hourly slot")
    if slot != slot.replace(minute=0, second=0, microsecond=0):
        raise ValueError("hourly slot is not aligned")
    if require_ready and actual.get("status")!="ready":raise ValueError("hourly forecast delayed")
    if actual.get("status") == "ready":
        if len(records) != len(HORIZONS_SECONDS) or {r["horizon_seconds"] for r in records} != HORIZONS_SECONDS:
            raise ValueError("incomplete hourly horizons")
    ids = [r.get("forecast_id") for r in records]
    if any(not isinstance(fid, str) or not fid for fid in ids) or len(set(ids)) != len(ids):
        raise ValueError("missing or duplicate issued IDs")
    if require_ready:
        # Normal trigger is :08; allow seven minutes for queue/build/CDN propagation.
        due_slot = now.replace(minute=0, second=0, microsecond=0)
        if now.minute < 15:
            due_slot -= timedelta(hours=1)
        if slot < due_slot:
            raise ValueError("latest hourly slot missing after publication grace period")
    for record in records:
        cutoff = fresh(record["input_cutoff"], now, "hourly input")
        if cutoff != slot:raise ValueError("mixed hourly slots")
        issued = timestamp(record["issued_at"])
        if issued > generated + CLOCK_TOLERANCE:raise ValueError("issuance after publication timestamp")
        start = timestamp(record["window_start"])
        if issued >= start:raise ValueError("event began before issuance")
        if timestamp(record.get("slot")) != slot or issued < slot or issued >= slot+timedelta(minutes=55):
            raise ValueError("invalid hourly issuance time")
        available = timestamp(record.get("available_at"))
        if available < cutoff or available > issued:
            raise ValueError("invalid source receipt time")
        if (record.get("horizon_seconds") not in HORIZONS_SECONDS or start != slot+timedelta(hours=1)
                or timestamp(record.get("target_end"))-start != timedelta(seconds=record["horizon_seconds"])):
            raise ValueError("inconsistent forecast window")
        def finite(value):
            return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        p, q = record.get("terminal_down_flat_up"), record.get("price_quantiles")
        if (not isinstance(p, list) or len(p) != 3 or not all(finite(v) and 0 <= v <= 1 for v in p)
                or not math.isclose(sum(p), 1., abs_tol=1e-6)
                or not all(finite(record.get(k)) and 0 <= record[k] <= 1 for k in ("hit_up", "hit_down"))):
            raise ValueError("invalid forecast probabilities")
        if (not isinstance(q, list) or len(q) != 3 or not all(finite(v) and v > 0 for v in q)
                or q != sorted(q)
                or not all(finite(record.get(k)) and record[k] > 0 for k in ("reference_price", "lower_barrier_price", "upper_barrier_price"))
                or not record["lower_barrier_price"] < record["reference_price"] < record["upper_barrier_price"]):
            raise ValueError("invalid forecast prices")
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument("--expected");p.add_argument("--require-ready",action="store_true")
    p.add_argument("--expected-replay");args=p.parse_args()
    expected=json.loads(Path(args.expected).read_text()) if args.expected else None
    expected_replay=json.loads(Path(args.expected_replay).read_text()) if args.expected_replay else None
    # Publication retries wait for the CDN; readiness must fail promptly on a
    # delivered outage payload, after the publisher has preserved its receipts.
    attempts = 15 if expected and not args.require_ready else 1
    for attempt in range(attempts):
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
            print('Event site verified:' if args.require_ready else 'Event payload verified:',actual['release_id'],actual['status'],
                  'slot='+actual['expected_slot'], 'generated='+actual['generated_at'],
                  'horizons='+str(len(actual.get('current',[]))));return
        except Exception as exc:
            print('verification pending:',str(exc),flush=True)
            if attempt+1 < attempts:time.sleep(8)
    raise SystemExit("Event site did not pass publication verification")


if __name__=='__main__':main()
