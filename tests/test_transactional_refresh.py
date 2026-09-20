from pathlib import Path
import hashlib
import json
import os
import sqlite3
import subprocess
import sys

import joblib
import pytest

from scripts import refresh_event_state as refresh
from signal_pipeline.data import connect, ingest
from signal_pipeline.diagnostics import export_archive, validate_object
from signal_pipeline.protocol import HORIZONS, PROTOCOL_HASH, training_hash

NOW = '2026-09-20T02:00:00Z'


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def generation(root, *, value=105, generated='2026-09-19T01:05:00Z'):
    root.mkdir(parents=True)
    with connect(root) as con:
        raw = json.dumps([[1789776000,90,110,100,value,3]]).encode()
        ingest(con, root, 'ETH-USD', raw, '2026-09-19', '2026-09-20', generated)
    con.close()
    archive = export_archive(root, {str(h): [] for h in HORIZONS}, kind='historical', as_of=generated)
    replay = {'protocol_hash':PROTOCOL_HASH, 'generated_at':generated,
              'horizons':{str(h): {} for h in HORIZONS}, 'archive':archive}
    (root/'replay.json').write_text(json.dumps(replay))
    (root/'source_status.json').write_text('{}')
    (root/'models').mkdir()
    active = {}
    for h in HORIZONS:
        bundle = {'protocol_hash': PROTOCOL_HASH, 'training_hash': training_hash(),
                  'fit_cutoff': '2026-09-01T00:00:00+00:00',
                  'training_target_end': '2026-08-31T20:00:00+00:00',
                  'validation_target_end': '2026-08-31T20:00:00+00:00', 'fixture_value':value}
        name = f'h{h}_2026-09_{value}.joblib'
        joblib.dump(bundle, root/'models'/name)
        active[str(h)] = {'file':name, 'sha256':refresh.file_hash(root/'models'/name)}
    (root/'active.json').write_text(json.dumps(active))
    return root


@pytest.fixture
def states(tmp_path):
    live = generation(tmp_path/'live')
    incoming = generation(tmp_path/'incoming', value=106, generated='2026-09-19T02:00:00Z')
    # These are byte sentinels: research must never open, rewrite or import them.
    for root in (live, incoming):
        (root/'issued.db').write_bytes(b'actual issued' if root == live else b'forbidden research ledger')
    (live/'shadow').mkdir(); (live/'shadow/issued.db').write_bytes(b'existing shadow ledger')
    (live/'research_run.txt').write_text('1')
    published = export_archive(live, {str(h): [] for h in HORIZONS}, kind='published', as_of=NOW)
    (live/'signals.json').write_text(json.dumps({'evidence_archives':{'published':published}}))
    return live, incoming


def test_complete_refresh_preserves_live_records_and_public_archive(states):
    live, incoming = states
    protected = refresh.protected_files(live, 'research')
    report = refresh.refresh(incoming, live, kind='research', source_run='2', now=NOW)
    assert report['status'] == 'committed'
    assert refresh.protected_files(live, 'research') == protected
    assert (live/'research_run.txt').read_text() == '2'
    ref = json.loads((live/'signals.json').read_text())['evidence_archives']['published']
    assert validate_object(live, ref)['kind'] == 'published'
    assert json.loads((live/'active.json').read_text())['6']['file'].endswith('_106.joblib')


@pytest.mark.parametrize('fault', ['checkpoint', 'missing_horizon', 'wrong_month', 'wrong_code',
                                  'purge', 'archive', 'source_status', 'future', 'backwards'])
def test_invalid_update_keeps_entire_original_generation(states, fault):
    live, incoming = states
    before = hashes(live)
    if fault == 'checkpoint':
        (incoming/'models/h6_2026-09_106.joblib').write_bytes(b'corrupt')
    elif fault == 'missing_horizon':
        (incoming/'models/h6_2026-09_106.joblib').unlink()
    elif fault in ('wrong_month', 'wrong_code', 'purge'):
        path = incoming/'models/h6_2026-09_106.joblib'; bundle=joblib.load(path)
        bundle[{'wrong_month':'fit_cutoff','wrong_code':'training_hash','purge':'training_target_end'}[fault]] = {
            'wrong_month':'2026-08-01T00:00:00+00:00', 'wrong_code':'wrong',
            'purge':'2026-09-01T00:00:00+00:00'}[fault]
        joblib.dump(bundle,path)
    elif fault == 'archive':
        (incoming/json.loads((incoming/'replay.json').read_text())['archive']['path']).write_text('{}')
    elif fault == 'source_status':
        (incoming/'source_status.json').unlink()
    else:
        path=incoming/'replay.json'; d=json.loads(path.read_text())
        d['generated_at']='2027-01-01T00:00:00Z' if fault=='future' else '2026-09-01T00:00:00Z'
        path.write_text(json.dumps(d))
    with pytest.raises(Exception):
        refresh.refresh(incoming, live, kind='research', source_run='2', now=NOW)
    assert hashes(live) == before


def test_disk_failure_after_observation_merge_is_isolated(states, monkeypatch):
    live, incoming = states; before=hashes(live)
    original=refresh.merge_research
    def fail(source, candidate):
        original(source,candidate)
        raise OSError('injected disk failure after merge')
    monkeypatch.setattr(refresh,'merge_research',fail)
    with pytest.raises(OSError):refresh.refresh(incoming,live,kind='research',source_run='2',now=NOW)
    assert hashes(live)==before


def test_commit_syscall_failure_keeps_original(states, monkeypatch):
    live, incoming = states;before=hashes(live)
    def fail(*args):raise OSError('injected atomic exchange error')
    monkeypatch.setattr(refresh,'exchange',fail)
    with pytest.raises(OSError):refresh.refresh(incoming,live,kind='research',source_run='2',now=NOW)
    assert hashes(live)==before


def test_bad_merger_cannot_modify_existing_observations(states, monkeypatch):
    live, incoming = states;before=hashes(live);original=refresh.merge_research
    def bad(source,stage):
        original(source,stage)
        with sqlite3.connect(stage/'observations.db') as db: db.execute('DELETE FROM bars')
    monkeypatch.setattr(refresh,'merge_research',bad)
    with pytest.raises(ValueError,match='observation history'):refresh.refresh(incoming,live,kind='research',source_run='2',now=NOW)
    assert hashes(live)==before


def test_bad_merger_cannot_replace_issued_records(states, monkeypatch):
    live, incoming=states;before=hashes(live);original=refresh.merge_research
    def bad(source,stage):
        original(source,stage); (stage/'issued.db').write_bytes(b'fake ledger')
    monkeypatch.setattr(refresh,'merge_research',bad)
    with pytest.raises(ValueError,match='protected'):refresh.refresh(incoming,live,kind='research',source_run='2',now=NOW)
    assert hashes(live)==before


@pytest.mark.parametrize('when', ['before', 'after'])
def test_process_death_never_exposes_partial_generation(tmp_path, when):
    root=tmp_path/'root';stage=tmp_path/'stage';root.mkdir();stage.mkdir()
    (root/'old').write_text('old');(stage/'new').write_text('new')
    code='''import os,sys
from scripts.refresh_event_state import exchange
if sys.argv[3]=='before': os._exit(71)
exchange(sys.argv[1],sys.argv[2])
os._exit(72)
'''
    run=subprocess.run([sys.executable,'-c',code,str(stage),str(root),when])
    assert run.returncode == (71 if when=='before' else 72)
    assert {p.name for p in root.iterdir()} == ({'old'} if when=='before' else {'new'})


def test_symlink_source_and_nested_input_rejected(states):
    live,incoming=states;before=hashes(live)
    (incoming/'link').symlink_to(live/'issued.db')
    with pytest.raises(ValueError,match='symlinks'):refresh.refresh(incoming,live,kind='research',source_run='2',now=NOW)
    with pytest.raises(ValueError,match='separate'):refresh.refresh(live,live,kind='research',source_run='2',now=NOW)
    assert hashes(live)==before


def test_bootstrap_does_not_import_research_issued_ledger(states, tmp_path):
    _,incoming=states;target=tmp_path/'empty'
    refresh.refresh(incoming,target,kind='research',source_run='2',now=NOW)
    assert not (target/'issued.db').exists()
    assert set(json.loads((target/'active.json').read_text()))==set(map(str,HORIZONS))


def test_workflow_orders_refresh_after_publication_and_retains_failure():
    import re
    text=Path('.github/workflows/event_hourly.yml').read_text()
    blocks=text.split('\n      - ')[1:]
    indexed={m.group(1):(i,block) for i,block in enumerate(blocks)
             if (m:=re.search(r'\n        id: (\w+)\n',block))}
    for name in ('research','historical','range_seed'):
        assert indexed[name][0] > indexed['publication'][0]
    for name in ('research_install','historical_install','range_install','bootstrap_install'):
        assert 'continue-on-error: true' in indexed[name][1]
        assert 'refresh_event_state.py' in indexed[name][1]
    assert "if: steps.restored.outputs.ready != 'true'" in indexed['bootstrap_research'][1]
    assert indexed['operation_audit'][0] < indexed['final_snapshot'][0]
    assert 'group: daily-forecast' in text and 'cron: "8 * * * *"' in text
    assert "steps.operation_audit.outputs.refresh_outcome == 'failure'" in blocks[-1]
