#!/usr/bin/env bash
set -euo pipefail
# Caller holds the shared daily-forecast group. State must be persisted before publishing.
git config user.name eth-forecast-bot
git config user.email eth-forecast-bot@users.noreply.github.com
git fetch origin data/daily-forecast:refs/remotes/origin/data/daily-forecast
git worktree add --detach "$RUNNER_TEMP/event-site" origin/data/daily-forecast
cp forecast_site/public/index.html forecast_site/public/events.js "$RUNNER_TEMP/event-site/forecast_site/public/"
cp forecast_site/vercel.json "$RUNNER_TEMP/event-site/forecast_site/vercel.json"
cp lake/signals/signals.json "$RUNNER_TEMP/event-site/forecast_site/public/signals.json"
cp lake/signals/replay.json "$RUNNER_TEMP/event-site/forecast_site/public/signals_replay.json"
git -C "$RUNNER_TEMP/event-site" add -f forecast_site/public/index.html forecast_site/public/events.js forecast_site/public/signals.json forecast_site/public/signals_replay.json forecast_site/vercel.json
if ! git -C "$RUNNER_TEMP/event-site" diff --cached --quiet; then
  git -C "$RUNNER_TEMP/event-site" commit -m "chore(site): publish immutable hourly ETH event forecasts"
  git -C "$RUNNER_TEMP/event-site" push origin HEAD:refs/heads/data/daily-forecast
fi
python scripts/verify_event_site.py --expected lake/signals/signals.json
python - <<'PY'
import json
from signal_pipeline.ledger import mark_verified
d=json.load(open('lake/signals/signals.json'))
mark_verified('lake/signals',[r['forecast_id'] for r in d['current']],d['release_id'])
PY
bash scripts/persist_event_ledger.sh
