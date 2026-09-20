"""Recovery evidence must survive the real bounded hourly snapshot function."""
import sqlite3
import json

import pytest

from scripts import event_state
from signal_pipeline import data, ledger
from signal_pipeline.diagnostics import export_archive, validate_object


@pytest.mark.parametrize('reports_present', [False, True])
def test_hourly_snapshot_preserves_optional_recovery_reports(tmp_path, monkeypatch, reports_present):
    source, target = tmp_path/'source', tmp_path/'target'
    source.mkdir()
    (source/'active.json').write_text('{}')
    names = ('continuity_audit.json', 'continuity_audit.md', 'publication_probe.json')
    if reports_present:
        for name in names:
            (source/name).write_text('immutable test receipt: '+name+'\n')

    def backup(_root, destination):
        with sqlite3.connect(destination/'observations.db') as db:
            db.executescript('CREATE TABLE bars(open_time TEXT, raw_hash TEXT); CREATE TABLE downloads(raw_hash TEXT);')
        with sqlite3.connect(destination/'issued.db') as db:
            db.execute('CREATE TABLE sentinel(value TEXT)')
            db.execute("INSERT INTO sentinel VALUES ('preserved')")

    monkeypatch.setattr(event_state, 'backup', backup)
    monkeypatch.setattr(event_state, 'copy_evidence', lambda *_args, **_kwargs: None)
    event_state.snapshot(source, target)
    for name in names:
        assert (target/name).exists() == reports_present
        if reports_present:
            assert (target/name).read_bytes() == (source/name).read_bytes()
    with sqlite3.connect(target/'issued.db') as db:
        assert db.execute('SELECT value FROM sentinel').fetchone() == ('preserved',)


@pytest.mark.parametrize('snapshot_name', ['hourly-state', 'final-hourly-state'])
def test_real_snapshot_resolves_precopied_reports_from_source_archives(tmp_path, snapshot_name):
    """Regression from run35483661981: report copied before target archives exist."""
    source, target=tmp_path/'source', tmp_path/snapshot_name
    source.mkdir()
    db=data.connect(source);db.close()
    db=ledger.connect(source);db.close()
    (source/'active.json').write_text('{}')
    refs={kind:export_archive(source, {str(h):[] for h in (6,24,72,168,336,720)},
                             kind=kind,as_of='2026-09-20T02:00:00Z')
          for kind in ('published','historical','candidate_history')}
    (source/'replay.json').write_text(json.dumps({'archive':refs['historical']}))
    (source/'historical_study.json').write_text(json.dumps({'archive':refs['candidate_history']}))
    (source/'signals.json').write_text(json.dumps({'evidence_archives':refs}))
    # Use real backup, trimming and archive copying; no mocked copy_evidence.
    event_state.snapshot(source,target)
    for kind,ref in refs.items():
        assert validate_object(target,ref)['kind']==kind
        assert (target/ref['path']).read_bytes()==(source/ref['path']).read_bytes()
    assert (target/'signals.json').read_bytes()==(source/'signals.json').read_bytes()
    with sqlite3.connect(target/'issued.db') as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0]=='ok'
