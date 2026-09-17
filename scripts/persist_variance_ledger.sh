#!/usr/bin/env bash
set -euo pipefail

# Preserve the isolated HAR-RV variance shadow on the same durable audit branch as
# other event ledgers. Only public.json is intended for the public website; model
# checkpoints and the issued SQLite ledger remain audit/recovery data.
git config user.name eth-forecast-bot
git config user.email eth-forecast-bot@users.noreply.github.com

if [ ! -d "$RUNNER_TEMP/variance-ledger" ]; then
  if git ls-remote --exit-code --heads origin data/event-ledger > /dev/null; then
    git fetch origin data/event-ledger:refs/remotes/origin/data/event-ledger
    ledger_base=origin/data/event-ledger
  else
    ledger_base=HEAD
  fi
  git worktree add --detach "$RUNNER_TEMP/variance-ledger" "$ledger_base"
fi

target="$RUNNER_TEMP/variance-ledger/lake/event-ledger/variance-shadow"
mkdir -p "$target/models"

python - <<'PY'
import pathlib
import shutil
import sqlite3
import os

source = pathlib.Path('lake/signals/variance_shadow')
target = pathlib.Path(os.environ['RUNNER_TEMP']) / 'variance-ledger/lake/event-ledger/variance-shadow'
target.mkdir(parents=True, exist_ok=True)

ledger = source / 'issued.db'
if not ledger.exists():
    raise SystemExit('variance shadow ledger is missing')
with sqlite3.connect(ledger) as src, sqlite3.connect(target / 'issued.db') as dst:
    src.backup(dst)
    assert dst.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'

for name in ('active.json', 'report.json', 'public.json'):
    path = source / name
    if path.exists():
        shutil.copy2(path, target / name)

public = target / 'public.json'
if not public.is_file() or public.stat().st_size == 0:
    raise SystemExit('variance shadow public summary is missing')

models = source / 'models'
if models.exists():
    shutil.copytree(models, target / 'models', dirs_exist_ok=True)
PY

git -C "$RUNNER_TEMP/variance-ledger" add -f lake/event-ledger/variance-shadow/
if ! git -C "$RUNNER_TEMP/variance-ledger" diff --cached --quiet; then
  git -C "$RUNNER_TEMP/variance-ledger" commit -m "chore(audit): preserve all-horizon HAR-RV variance shadow"
  git -C "$RUNNER_TEMP/variance-ledger" push origin HEAD:refs/heads/data/event-ledger
fi
