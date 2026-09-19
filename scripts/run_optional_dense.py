"""Stage optional dense shadow work after core delivery; commit only valid output."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[1]
FILES = ('issued.db', 'report.json', 'public.json')
REF = 'refs/remotes/origin/data/event-ledger'
BASE_PATH = 'lake/event-ledger/variance-shadow'
DENSE_PATH = 'lake/event-ledger/variance-shadow-dense'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def snapshot_database(source, target):
    with sqlite3.connect(Path(source).resolve().as_uri() + '?mode=ro', uri=True) as src:
        with sqlite3.connect(target) as dst:
            src.backup(dst)
            if dst.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('SQLite integrity failure')


def git_bytes(path):
    return subprocess.check_output(['git', 'show', f'{REF}:{path}'], cwd=REPO, timeout=20)


def restore_git(stage):
    """Only trusted audit-branch files; missing existing ledger fails closed."""
    base = stage / 'variance_shadow'
    (base / 'models').mkdir(parents=True)
    raw = git_bytes(BASE_PATH + '/active.json')
    manifest = json.loads(raw)
    (base / 'active.json').write_bytes(raw)
    for horizon in ('24', '72'):
        entry = manifest['models'][horizon]
        name = entry['file']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9_.-]+\.json', name):
            raise ValueError('unsafe checkpoint filename')
        content = git_bytes(BASE_PATH + '/models/' + name)
        if hashlib.sha256(content).hexdigest() != entry['sha256']:
            raise ValueError('checkpoint SHA256 mismatch')
        (base / 'models' / name).write_bytes(content)
    dense = stage / 'variance_shadow_dense'
    dense.mkdir()
    # This cohort already exists. Never replace a failed restore with an empty DB.
    (dense / 'issued.db').write_bytes(git_bytes(DENSE_PATH + '/issued.db'))
    snapshot_database(dense / 'issued.db', stage / 'prior_dense.db')


def execute_dense(stage):
    result = subprocess.run([sys.executable, str(REPO / 'scripts/run_variance_dense_shadow.py'),
                             '--root', str(stage)], cwd=REPO, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
    if result.returncode:
        raise RuntimeError(f'dense process failed ({result.returncode}): {result.stderr[-400:]}')


def validate_output(stage):
    dense = stage / 'variance_shadow_dense'
    for name in FILES:
        if not (dense / name).is_file() or not (dense / name).stat().st_size:
            raise ValueError('missing dense output: ' + name)
    report = json.loads((dense / 'report.json').read_text())
    public = json.loads((dense / 'public.json').read_text())
    if report.get('errors') != []:
        raise ValueError('dense report contains errors or lacks an explicit empty errors list')
    if set(public.get('horizons', {})) != {'24', '72'}:
        raise ValueError('unexpected dense public horizons')
    if public.get('generated_at') != report.get('generated_at'):
        raise ValueError('mixed dense output generations')
    before, after = stage / 'prior_dense.db', dense / 'issued.db'
    with sqlite3.connect(after.resolve().as_uri() + '?mode=ro', uri=True) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('invalid dense output DB')
        db.execute('ATTACH DATABASE ? AS prior', (before.resolve().as_uri() + '?mode=ro',))
        for table in ('forecasts', 'outcomes'):
            # Existing records, including outcomes, must be preserved byte-for-byte.
            removed = db.execute(f'SELECT * FROM prior.{table} EXCEPT SELECT * FROM main.{table} LIMIT 1').fetchone()
            if removed is not None:
                raise ValueError('dense append-only violation in ' + table)
        if db.execute('SELECT 1 FROM forecasts WHERE horizon_hours NOT IN (24,72) LIMIT 1').fetchone():
            raise ValueError('unauthorized dense horizon')
    return report


def run(root, *, restore=restore_git, execute=execute_dense):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt = {'schema': 1, 'stage': 'dense_variance', 'started_at': datetime.now(timezone.utc).isoformat(),
               'status': 'failed', 'core_delivery_independent': True, 'committed_output': False,
               'run_id': __import__('os').environ.get('GITHUB_RUN_ID'), 'errors': []}
    try:
        with tempfile.TemporaryDirectory(prefix='.dense-stage-', dir=root.parent) as tmp:
            stage = Path(tmp)
            snapshot_database(root / 'observations.db', stage / 'observations.db')
            restore(stage)
            execute(stage)
            report = validate_output(stage)
            candidate = stage / 'variance_shadow_dense'
            destination = root / 'variance_shadow_dense'
            old = stage / 'previous_local_dense'
            if destination.exists():
                destination.rename(old)
            try:
                candidate.rename(destination)
            except Exception:
                if old.exists():
                    old.rename(destination)
                raise
            receipt.update(status='success', committed_output=True,
                           output_sha256={n: sha(destination / n) for n in FILES},
                           source_as_of=report.get('source_as_of'),
                           issuance_audit=report.get('issuance_audit'))
    except Exception as exc:
        receipt['errors'].append(type(exc).__name__ + ': ' + str(exc)[:500])
    receipt['finished_at'] = datetime.now(timezone.utc).isoformat()
    write_json(root / 'dense_stage_status.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('lake/signals'))
    args = parser.parse_args()
    receipt = run(args.root)
    print(json.dumps(receipt, sort_keys=True), flush=True)
    if receipt['status'] != 'success':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
