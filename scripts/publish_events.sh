#!/usr/bin/env bash
set -euo pipefail
# Caller holds the shared daily-forecast group. State must be persisted before publishing.
# Always retain a gap/delivery audit; a failed public release must stay failed.
audit_publication_exit() {
  original_status=$?
  trap - EXIT
  audit_status=0
  python scripts/audit_forecast_continuity.py --root lake/signals || audit_status=$?
  if [ -n "${GITHUB_STEP_SUMMARY:-}" ] && [ -s lake/signals/continuity_audit.md ]; then
    cat lake/signals/continuity_audit.md >> "$GITHUB_STEP_SUMMARY" || true
  fi
  if [ "$original_status" -eq 0 ] && [ "$audit_status" -ne 0 ]; then
    original_status=$audit_status
  fi
  exit "$original_status"
}
trap audit_publication_exit EXIT
python scripts/publish_event_feed.py
# First wait for the exact public payload; never replace the served release with
# the Git source or download every historical archive on a known stale release.
python scripts/wait_event_release.py --expected lake/signals/signals.json --receipt lake/signals/publication_probe.json
python scripts/verify_event_site.py --expected lake/signals/signals.json --expected-replay lake/signals/replay.json
python - <<'PY'
import json
from signal_pipeline.ledger import mark_verified
d=json.load(open('lake/signals/signals.json'))
mark_verified('lake/signals',[r['forecast_id'] for r in d['current']],d['release_id'])
mark_verified('lake/signals/shadow',[r['forecast_id'] for r in d.get('shadow_current',[])],d['release_id'])
mark_verified('lake/signals/range_shadow',[r['forecast_id'] for r in d.get('range_candidate',{}).get('current',[])],d['release_id'])
PY
bash scripts/persist_event_ledger.sh
# A delivered payload can describe an outage. Persist its receipts before failing
# the run when a complete, fresh forecast has not actually recovered.
python scripts/verify_event_site.py --expected lake/signals/signals.json --require-ready
