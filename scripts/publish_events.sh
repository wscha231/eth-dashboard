#!/usr/bin/env bash
set -euo pipefail
# Caller holds the shared daily-forecast group. State must be persisted before publishing.
python scripts/publish_event_feed.py
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
