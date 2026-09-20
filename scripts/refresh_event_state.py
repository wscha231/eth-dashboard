"""Validate optional input updates on a copy, then atomically exchange Linux directories.

The hourly workflow is the sole writer under its existing daily-forecast lock.
Only artifacts selected from successful same-repository main workflows are inputs.
No training, publication, issuance or promotion takes place here.
"""
from __future__ import annotations

import argparse
from contextlib import closing
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_pipeline.data import utc
from signal_pipeline.engine import obtain_bundle, check_replay_cutoff
from signal_pipeline.protocol import HORIZONS, PROTOCOL_HASH
from scripts.copy_event_evidence import copy_evidence
from scripts.event_state import merge_research
from scripts.import_event_study import import_study

MARKERS = {'research': 'research_run.txt', 'historical': 'historical_run.txt',
           'range': 'range_seed_run.txt'}


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def plain_tree(root):
    root = Path(root)
    if root.is_symlink() or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('symlinks are not allowed in an input generation')


def validate_core(root, *, now=None):
    """Check the same current-month saved bundles used by inference; never fit."""
    root = Path(root)
    plain_tree(root)
    replay = json.loads((root / 'replay.json').read_text())
    if replay.get('protocol_hash') != PROTOCOL_HASH or set(replay.get('horizons', {})) != set(map(str, HORIZONS)):
        raise ValueError('compatible complete six-horizon replay required')
    current = utc(now)
    generated = utc(replay['generated_at'])
    if generated > current or __import__('pandas').isna(generated):
        raise ValueError('replay timestamp is invalid or in the future')
    active = json.loads((root / 'active.json').read_text())
    if set(active) != set(map(str, HORIZONS)):
        raise ValueError('all six active checkpoints required')
    cutoff = current.floor('h').replace(day=1, hour=0)
    for h in HORIZONS:
        bundle, _ = obtain_bundle(root, None, None, None, cutoff, h, allow_fit=False)
        check_replay_cutoff(bundle, cutoff)
    path = root / 'observations.db'
    with closing(sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True)) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('observation database integrity failure')
        db.execute('SELECT product, open_time, observed_at, revision, content_hash FROM bars LIMIT 1')
        db.execute('SELECT * FROM downloads LIMIT 1')
    return {'checkpoints': {h: v['sha256'] for h, v in active.items()},
            'replay_sha256': file_hash(root / 'replay.json')}


def protected_files(root, kind):
    """Research may append observations, but may not replace any live ledger/feed."""
    protected = {}
    for p in Path(root).rglob('*'):
        if not p.is_file():
            continue
        name = p.relative_to(root).as_posix()
        if (p.suffix == '.db' and (kind != 'research' or name != 'observations.db')
                or name == 'signals.json'
                or kind != 'research' and (name == 'active.json' or name.startswith('models/'))):
            protected[name] = file_hash(p)
    return protected


def assert_observations_preserved(root, stage):
    original = Path(root) / 'observations.db'
    if not original.exists():
        return
    # Even SQLite mode=ro can write WAL shared-memory sidecars. Compare against
    # a private copy so a rejected refresh leaves every live byte untouched.
    with tempfile.TemporaryDirectory(prefix='previous-', dir=Path(stage).parent) as temporary:
        previous = Path(temporary) / 'observations.db'
        for suffix in ('', '-wal', '-shm'):
            path = Path(str(original) + suffix)
            if path.exists():shutil.copy2(path, str(previous) + suffix)
        with closing(sqlite3.connect((Path(stage) / 'observations.db').resolve().as_uri() + '?mode=ro', uri=True)) as db:
            db.execute('ATTACH DATABASE ? AS previous', (previous.as_uri() + '?mode=ro',))
            for table in ('bars', 'downloads'):
                if db.execute(f'SELECT * FROM previous.{table} EXCEPT SELECT * FROM main.{table} LIMIT 1').fetchone():
                    raise ValueError('existing observation history was modified: ' + table)


def exchange(stage, root):
    """One Linux syscall: termination leaves either the old or the complete new tree.

    Do not replace this with two renames: termination between them loses the root.
    Unsupported filesystems fail before changing the existing generation.
    """
    stage, root = Path(stage), Path(root)
    if not root.exists():
        os.rename(stage, root)
        return
    rename = getattr(ctypes.CDLL(None, use_errno=True), 'renameat2', None)
    if rename is None:
        raise OSError('Linux renameat2 is required for atomic generation exchange')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(root), 2) != 0:
        raise OSError(ctypes.get_errno(), 'atomic generation exchange failed')


def refresh(source, root, *, kind, source_run, now=None):
    source, root = Path(source).absolute(), Path(root).absolute()
    if kind not in MARKERS or not str(source_run).isdigit():
        raise ValueError('known input kind and numeric source run required')
    if source == root or source.is_relative_to(root) or root.is_relative_to(source):
        raise ValueError('source and live generations must be separate')
    plain_tree(source)
    plain_tree(root)
    before = protected_files(root, kind)
    root.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.event-refresh-', dir=root.parent, ignore_cleanup_errors=True) as scratch:
        stage = Path(scratch) / 'candidate'
        if root.exists():
            shutil.copytree(root, stage)
        else:
            stage.mkdir()
        if kind == 'research':
            incoming = json.loads((source / 'replay.json').read_text())
            if (root / 'replay.json').exists():
                previous = json.loads((root / 'replay.json').read_text())
                if utc(incoming['generated_at']) < utc(previous['generated_at']):
                    raise ValueError('research would move the generation backwards')
            merge_research(source, stage)
            assert_observations_preserved(root, stage)
        elif kind == 'historical':
            import_study(source, stage)
        else:
            from signal_pipeline.range_candidate import install_seed
            install_seed(source / 'range_seed.json', stage, now=now)
        validated = validate_core(stage, now=now)
        # Validate every new AND already-published archive reference before commit.
        copy_evidence(stage, stage)
        if protected_files(stage, kind) != before:
            raise ValueError('refresh changed protected live records, feed or checkpoints')
        (stage / MARKERS[kind]).write_text(str(source_run))
        # Flush the accepted generation before exposing it to later workflow steps.
        for p in stage.rglob('*'):
            if p.is_file():
                with p.open('rb') as handle:
                    os.fsync(handle.fileno())
        exchange(stage, root)
        return {'status': 'committed', 'kind': kind, 'source_run': str(source_run),
                'protected_files': before, **validated}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['check', *MARKERS])
    p.add_argument('--root', type=Path, default=Path('lake/signals'))
    p.add_argument('--source', type=Path)
    p.add_argument('--source-run')
    p.add_argument('--report', type=Path)
    args = p.parse_args()
    if args.action == 'check':
        try:
            validate_core(args.root)
            ready = True
        except Exception as exc:
            ready = False
            print('Restored checkpoint unavailable:', type(exc).__name__, str(exc)[:250])
        with open(os.environ['GITHUB_OUTPUT'], 'a') as handle:
            handle.write(f'ready={str(ready).lower()}\n')
        return
    report = {'run_id': os.environ.get('GITHUB_RUN_ID'), 'attempt': os.environ.get('GITHUB_RUN_ATTEMPT'),
              'started_at': utc().isoformat(), 'kind': args.action, 'source_run': args.source_run}
    try:
        report.update(refresh(args.source, args.root, kind=args.action, source_run=args.source_run))
    except Exception as exc:
        report.update(status='rejected', error=type(exc).__name__ + ': ' + str(exc)[:400])
        raise
    finally:
        report['finished_at'] = utc().isoformat()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
