"""Exercise actual recovery code and externally observable publication invariants."""
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import re
import subprocess

import pytest

from scripts.verify_event_site import verify


ROOT = Path(__file__).resolve().parents[1]


def publication():
    return {
        'schema_version': 1, 'release_id': 'release', 'status': 'ready',
        'generated_at': '2026-09-07T06:08:00Z', 'expected_slot': '2026-09-07T06:00:00Z',
        'current': [dict(forecast_id=f'fixed-{h}', horizon_seconds=h*3600,
                         input_cutoff='2026-09-07T06:00:00Z', issued_at='2026-09-07T06:08:00Z',
                         window_start='2026-09-07T07:00:00Z') for h in (6, 24, 72, 168, 336, 720)],
    }


@pytest.mark.parametrize('now,passes', [('06:38', True), ('07:14', True), ('07:15', False)])
def test_watchdog_detects_a_missed_slot_after_the_publication_grace(now, passes):
    when = datetime.fromisoformat(f'2026-09-07T{now}:00+00:00')
    if passes:
        assert verify(publication(), now=when, require_ready=True)
    else:
        with pytest.raises(ValueError, match='latest hourly slot missing'):
            verify(publication(), now=when, require_ready=True)
        # Exact-release publication at a boundary remains auditable, even if late.
        assert verify(publication(), publication(), now=when)


@pytest.mark.parametrize('corruption,message', [
    ('empty', 'incomplete hourly horizons'), ('missing', 'incomplete hourly horizons'),
    ('duplicate_horizon', 'incomplete hourly horizons'), ('duplicate_id', 'duplicate issued IDs'),
    ('future', 'future hourly worker'), ('timezone', 'lacks timezone'),
    ('mixed_slot', 'mixed hourly slots'), ('issued_after_publication', 'issuance after publication'),
])
def test_ready_status_cannot_hide_invalid_or_incomplete_data(corruption, message):
    data = deepcopy(publication())
    if corruption == 'empty': data['current'] = []
    if corruption == 'missing': data['current'].pop()
    if corruption == 'duplicate_horizon': data['current'][1]['horizon_seconds'] = 6*3600
    if corruption == 'duplicate_id': data['current'][1]['forecast_id'] = 'fixed-6'
    if corruption == 'future': data['generated_at'] = '2026-09-07T08:08:00Z'
    if corruption == 'timezone': data['generated_at'] = '2026-09-07T06:08:00'
    if corruption == 'mixed_slot': data['current'][1]['input_cutoff'] = '2026-09-07T05:00:00Z'
    if corruption == 'issued_after_publication': data['current'][1]['issued_at'] = '2026-09-07T06:30:00Z'
    with pytest.raises(ValueError, match=message):
        verify(data, now=datetime.fromisoformat('2026-09-07T06:38:00+00:00'), require_ready=True)


def recovery(*, active=None, recent=None, now='2026-09-07T06:38:00Z'):
    harness = r'''
const recover=require('./scripts/event_recovery.cjs'),input=JSON.parse(process.argv[1]);
const outputs={},dispatches=[],notices=[],failures=[];
const github={paginate:async(fn,args)=>(input.active||[]).filter(r=>r.status===args.status),rest:{actions:{
  listWorkflowRunsForRepo(){},listWorkflowRuns:async()=>({data:{workflow_runs:input.recent||[]}}),
  createWorkflowDispatch:async args=>{dispatches.push(args);}
}}};
const summary={addHeading(){return this},addRaw(){return this},async write(){}};
const core={setOutput:(k,v)=>{outputs[k]=v},notice:v=>notices.push(v),setFailed:v=>failures.push(v),summary};
(async()=>{await recover({github,context:{repo:{owner:'wscha231',repo:'eth-dashboard'}},core,now:new Date(input.now)});
 console.log(JSON.stringify({outputs,dispatches,notices,failures}));})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run(['node', '-e', harness, json.dumps(dict(active=active, recent=recent, now=now))],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


@pytest.mark.parametrize('status', ['queued', 'in_progress', 'requested', 'waiting', 'pending'])
def test_recovery_does_not_replace_active_shared_publication_workers(status):
    result = recovery(active=[dict(id=12, name='source refresh', status=status,
                                   path='.github/workflows/forward_research.yml')])
    assert result['outputs']['recovery'] == 'active_worker'
    assert not result['dispatches']
    assert result['failures']  # A deferral must not claim the stale site is healthy.


def test_first_failure_dispatches_once_but_a_recent_recovery_attempt_cools_down():
    regular = dict(id=10, event='schedule', updated_at='2026-09-07T06:37:00Z')
    result = recovery(recent=[regular])
    assert result['outputs']['recovery'] == 'dispatched'
    assert result['dispatches'] == [dict(owner='wscha231', repo='eth-dashboard',
                                        workflow_id='event_hourly.yml', ref='main')]
    assert result['failures']  # Dispatch alone is never a verified successful recovery.
    retry = dict(id=11, event='workflow_dispatch', created_at='2026-09-06T00:00:00Z',
                 run_started_at='2026-09-07T06:25:00Z')
    result = recovery(recent=[retry])
    assert result['outputs']['recovery'] == 'cooldown'
    assert not result['dispatches']


def test_old_recovery_expires_but_new_issuance_deadline_is_respected():
    old = dict(id=11, event='workflow_dispatch', updated_at='2026-09-07T06:10:00Z')
    assert recovery(recent=[old])['outputs']['recovery'] == 'dispatched'
    result = recovery(recent=[old], now='2026-09-07T06:55:00Z')
    assert result['outputs']['recovery'] == 'issuance_deadline'
    assert not result['dispatches']


def test_recovery_guard_covers_all_workflows_sharing_the_publisher_lock():
    guard = (ROOT/'scripts/event_recovery.cjs').read_text()
    names = set(re.findall(r"'([a-z_]+\.yml)'", guard))
    for workflow in (ROOT/'.github/workflows').glob('*.yml'):
        if 'group: daily-forecast' in workflow.read_text():
            assert workflow.name in names
