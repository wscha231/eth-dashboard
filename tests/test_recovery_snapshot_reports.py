"""Recovery evidence must survive the real bounded hourly snapshot function."""
import sqlite3

import pytest

from scripts import event_state


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
