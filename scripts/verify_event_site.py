"""Verify the externally served release, exact issued IDs and hourly freshness."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.request import urlopen


def verify(actual, expected=None, now=None, require_ready=False):
    now=now or datetime.now(timezone.utc)
    if actual.get("schema_version")!=1:raise ValueError("event schema mismatch")
    if expected and actual.get("release_id")!=expected.get("release_id"):raise ValueError("release mismatch")
    if expected and [r["forecast_id"] for r in actual.get("current",[])]!=[r["forecast_id"] for r in expected.get("current",[])]:
        raise ValueError("issued records mismatch")
    if (now-datetime.fromisoformat(actual["generated_at"])).total_seconds()>100*60:raise ValueError("hourly worker heartbeat is stale")
    if require_ready and actual.get("status")!="ready":raise ValueError("hourly forecast delayed")
    for record in actual.get("current",[]):
        if (now-datetime.fromisoformat(record["input_cutoff"])).total_seconds()>100*60:raise ValueError("stale hourly input")
        if datetime.fromisoformat(record["issued_at"]) >= datetime.fromisoformat(record["window_start"]):raise ValueError("event began before issuance")
    return True


def main():
    p=argparse.ArgumentParser();p.add_argument("--expected");p.add_argument("--require-ready",action="store_true");args=p.parse_args()
    expected=json.loads(Path(args.expected).read_text()) if args.expected else None
    for attempt in range(15 if expected else 1):
        try:
            suffix=f"?verify={int(time.time())}"
            with urlopen('https://etherforecast.live/signals.json'+suffix,timeout=12) as response:actual=json.load(response)
            with urlopen('https://etherforecast.live/'+suffix,timeout=12) as response:html=response.read().decode()
            with urlopen('https://etherforecast.live/events.js'+suffix,timeout=12) as response:js=response.read().decode()
            if 'id="event-system"' not in html or 'loadEventForecasts' not in js:raise ValueError("new event charts missing")
            verify(actual,expected,require_ready=args.require_ready)
            print('Event site verified:',actual['release_id'],actual['status']);return
        except Exception as exc:
            print('verification pending:',str(exc),flush=True)
            if expected and attempt<14:time.sleep(8)
    raise SystemExit("Event site did not pass publication verification")


if __name__=='__main__':main()
