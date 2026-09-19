#!/usr/bin/env bash
set -euo pipefail
# Called under the existing shared writer lock, AFTER core publication.
export AVAILABILITY_PHASE=restore
finish_availability() {
  result=$?
  trap - EXIT
  export AVAILABILITY_EXIT="$result"
  status_result=0
  python - <<'PY' || status_result=$?
import json, os
from pathlib import Path
from datetime import datetime, timezone
path=Path('lake/signals/availability_stage_status.json')
try:
    prior=json.loads(path.read_text())
    if prior.get('run_id') != os.environ.get('GITHUB_RUN_ID'): prior={}
except (OSError, ValueError): prior={}
prior.update(run_id=os.environ.get('GITHUB_RUN_ID'),
             finished_at=datetime.now(timezone.utc).isoformat(),
             status='success' if os.environ['AVAILABILITY_EXIT']=='0' else 'failed',
             last_phase=os.environ['AVAILABILITY_PHASE'],
             exit_code=int(os.environ['AVAILABILITY_EXIT']),
             public_delivery=False, promotion='none')
if prior['status']=='failed' and not prior.get('error'):
    prior['error']='availability stage failed at '+prior['last_phase']
path.parent.mkdir(parents=True,exist_ok=True)
tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(prior,sort_keys=True)+'\n');tmp.replace(path)
PY
  if [ "$result" -eq 0 ] && [ "$status_result" -ne 0 ]; then result="$status_result"; fi
  exit "$result"
}
trap finish_availability EXIT
# Explicitly refresh the trusted ref; errors are not reinterpreted as a new cohort.
git fetch origin data/event-ledger:refs/remotes/origin/data/event-ledger
python scripts/availability_shadow.py restore --root lake/signals
export AVAILABILITY_PHASE=run
python scripts/availability_shadow.py run --root lake/signals
export AVAILABILITY_PHASE=persist
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
worktree="$RUNNER_TEMP/event-ledger"
if [ ! -d "$worktree" ]; then
  git worktree add --detach "$worktree" refs/remotes/origin/data/event-ledger
fi
# Never commit unrelated staged changes left by a failed core writer.
git -C "$worktree" diff --cached --quiet
python scripts/availability_shadow.py snapshot --root lake/signals --target "$worktree/lake/event-ledger/availability-shadow"
git -C "$worktree" config user.name eth-forecast-bot
git -C "$worktree" config user.email eth-forecast-bot@users.noreply.github.com
git -C "$worktree" add -f lake/event-ledger/availability-shadow/
if ! git -C "$worktree" diff --cached --quiet; then
  git -C "$worktree" commit -m "chore(audit): preserve received-input CENTER availability shadow"
  git -C "$worktree" push origin HEAD:refs/heads/data/event-ledger
fi
export AVAILABILITY_PHASE=complete
