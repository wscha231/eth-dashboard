#!/usr/bin/env bash
set -euo pipefail
# Research price/model blobs stay in bounded artifacts; the issuance audit has durable git history.
git config user.name eth-forecast-bot
git config user.email eth-forecast-bot@users.noreply.github.com
if [ ! -d "$RUNNER_TEMP/event-ledger" ]; then
  if git ls-remote --exit-code --heads origin data/event-ledger > /dev/null; then
    git fetch origin data/event-ledger:refs/remotes/origin/data/event-ledger
    ledger_base=origin/data/event-ledger
  else
    ledger_base=HEAD
  fi
  git worktree add --detach "$RUNNER_TEMP/event-ledger" "$ledger_base"
fi
mkdir -p "$RUNNER_TEMP/event-ledger/lake/event-ledger"
python - <<'PY'
import os,sqlite3,pathlib
target=pathlib.Path(os.environ['RUNNER_TEMP'])/'event-ledger/lake/event-ledger/issued.db'
with sqlite3.connect('lake/signals/issued.db') as source,sqlite3.connect(target) as destination:
    source.backup(destination)
    assert destination.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
PY
git -C "$RUNNER_TEMP/event-ledger" add -f lake/event-ledger/issued.db
if ! git -C "$RUNNER_TEMP/event-ledger" diff --cached --quiet; then
  git -C "$RUNNER_TEMP/event-ledger" commit -m "chore(audit): preserve immutable hourly issuance and publication receipts"
  git -C "$RUNNER_TEMP/event-ledger" push origin HEAD:refs/heads/data/event-ledger
fi
