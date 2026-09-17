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
import os,sqlite3,pathlib,shutil
target=pathlib.Path(os.environ['RUNNER_TEMP'])/'event-ledger/lake/event-ledger/issued.db'
with sqlite3.connect('lake/signals/issued.db') as source,sqlite3.connect(target) as destination:
    source.backup(destination)
    assert destination.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
shadow=pathlib.Path('lake/signals/shadow/issued.db')
if shadow.exists():
    with sqlite3.connect(shadow) as source,sqlite3.connect(target.with_name('shadow_issued.db')) as destination:
        source.backup(destination)
        assert destination.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
range_ledger=pathlib.Path('lake/signals/range_shadow/issued.db')
if range_ledger.exists():
    with sqlite3.connect(range_ledger) as source,sqlite3.connect(target.with_name('range_issued.db')) as destination:
        source.backup(destination)
        assert destination.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
seed_source=pathlib.Path('lake/signals/range_seeds')
if seed_source.exists():
    shutil.copytree(seed_source,target.parent/'range_seeds',dirs_exist_ok=True)

dense_source=pathlib.Path('lake/signals/variance_shadow_dense')
if dense_source.exists():
    dense_target=target.parent/'variance-shadow-dense'
    dense_target.mkdir(parents=True,exist_ok=True)
    dense_ledger=dense_source/'issued.db'
    if dense_ledger.exists():
        with sqlite3.connect(dense_ledger) as source,sqlite3.connect(dense_target/'issued.db') as destination:
            source.backup(destination)
            assert destination.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
    for name in ('report.json','public.json'):
        path=dense_source/name
        if path.exists():shutil.copy2(path,dense_target/name)
PY
python scripts/archive_event_inputs.py lake/signals "$RUNNER_TEMP/event-ledger/lake/event-inputs"
python scripts/deployment_policy.py --target "$RUNNER_TEMP/event-ledger" --stage
git -C "$RUNNER_TEMP/event-ledger" add -f lake/event-ledger/
git -C "$RUNNER_TEMP/event-ledger" add -f lake/event-inputs/
if ! git -C "$RUNNER_TEMP/event-ledger" diff --cached --quiet; then
  git -C "$RUNNER_TEMP/event-ledger" commit -m "chore(audit): preserve immutable hourly issuance and publication receipts"
  git -C "$RUNNER_TEMP/event-ledger" push origin HEAD:refs/heads/data/event-ledger
fi