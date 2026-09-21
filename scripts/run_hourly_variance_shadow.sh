#!/usr/bin/env bash
set -euo pipefail

status_file="lake/signals/variance_daily_status.json"
phase="window_check"
final_status="failed"
message=""
started_at="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

write_status() {
  local code="$1"
  STATUS_FILE="$status_file" PHASE="$phase" FINAL_STATUS="$final_status" MESSAGE="$message"   STARTED_AT="$started_at" EXIT_CODE="$code" python - <<'PY'
import json, os
from datetime import datetime, timezone
from pathlib import Path

path=Path(os.environ["STATUS_FILE"])
payload={
    "schema":1,
    "stage":"all_horizon_variance_00utc",
    "run_id":os.environ.get("GITHUB_RUN_ID"),
    "run_attempt":os.environ.get("GITHUB_RUN_ATTEMPT"),
    "started_at":os.environ["STARTED_AT"],
    "finished_at":datetime.now(timezone.utc).isoformat(),
    "status":os.environ["FINAL_STATUS"],
    "phase":os.environ["PHASE"],
    "exit_code":int(os.environ["EXIT_CODE"]),
    "message":os.environ["MESSAGE"],
    "promotion":"none",
    "public_delivery":False,
}
report=path.parent/"variance_shadow/report.json"
if report.is_file():
    try:
        current=json.loads(report.read_text())
        payload["generated_at"]=current.get("generated_at")
        payload["source_as_of"]=current.get("source_as_of")
        payload["issuance_audit"]=current.get("issuance_audit")
        payload["issued_horizons"]=sorted({int(r["horizon_hours"]) for r in current.get("issued_this_run",[])})
        payload["errors"]=current.get("errors",[])
    except Exception as exc:
        payload["report_error"]=type(exc).__name__+": "+str(exc)[:240]
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_suffix(".tmp")
tmp.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
tmp.replace(path)
PY
}

finish() {
  code=$?
  trap - EXIT
  if [ "$code" -eq 0 ] && [ "$final_status" = "failed" ]; then
    final_status="success"
  fi
  write_status "$code" || true
  exit "$code"
}
trap finish EXIT

hour="$(date -u +%H)"
minute="$(date -u +%M)"
if [ "$hour" != "00" ] || [ "$minute" -ge 55 ]; then
  final_status="not_due"
  message="Outside the frozen 00:00-00:54 UTC issuance window; no variance issuance attempted."
  exit 0
fi

phase="restore"
git fetch origin data/event-ledger:refs/remotes/origin/data/event-ledger
restore="$RUNNER_TEMP/variance-current"
rm -rf "$restore"
mkdir -p "$restore"
if ! git archive refs/remotes/origin/data/event-ledger -- lake/event-ledger/variance-shadow | tar -x -C "$restore"; then
  message="Trusted all-horizon variance state could not be restored."
  exit 1
fi
test -s "$restore/lake/event-ledger/variance-shadow/issued.db"
test -s "$restore/lake/event-ledger/variance-shadow/active.json"
rm -rf lake/signals/variance_shadow
mkdir -p lake/signals/variance_shadow
cp -a "$restore/lake/event-ledger/variance-shadow/." lake/signals/variance_shadow/

phase="issue"
python scripts/run_variance_shadow.py --root lake/signals --no-refit --require-current-issue

phase="persist"
bash scripts/persist_variance_ledger.sh

phase="complete"
final_status="success"
message="All six frozen 00UTC HAR-RV variance horizons were issued/idempotently restored and persisted."
