"""Exercise actual recovery code and externally observable publication invariants."""
from copy import deepcopy
from datetime import datetime, timedelta
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
                         slot='2026-09-07T06:00:00Z', available_at='2026-09-07T06:02:00Z',
                         window_start='2026-09-07T07:00:00Z',
                         target_end=(datetime.fromisoformat('2026-09-07T07:00:00+00:00')+timedelta(hours=h)).isoformat(),
                         reference_price=2000., lower_barrier_price=1800., upper_barrier_price=2200.,
                         price_quantiles=[1800., 2100., 2400.], terminal_down_flat_up=[.2, .5, .3],
                         hit_up=.6, hit_down=.4) for h in (6, 24, 72, 168, 336, 720)],
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


def recovery(*, active=None, recent=None, now='2026-09-07T06:38:00Z', dispatch_time=None):
    harness = r'''
const recover=require('./scripts/event_recovery.cjs'),input=JSON.parse(process.argv[1]);
const outputs={},dispatches=[],notices=[],failures=[];
const github={paginate:async(fn,args)=>(input.active||[]).filter(r=>r.status===args.status),rest:{actions:{
  listWorkflowRunsForRepo(){},listWorkflowRuns:async()=>({data:{workflow_runs:input.recent||[]}}),
  createWorkflowDispatch:async args=>{dispatches.push(args);}
}}};
const summary={addHeading(){return this},addRaw(){return this},async write(){}};
const core={setOutput:(k,v)=>{outputs[k]=v},notice:v=>notices.push(v),setFailed:v=>failures.push(v),summary};
(async()=>{await recover({github,context:{repo:{owner:'wscha231',repo:'eth-dashboard'}},core,now:new Date(input.now),
 clock:input.dispatch_time?()=>new Date(input.dispatch_time):undefined});
 console.log(JSON.stringify({outputs,dispatches,notices,failures}));})().catch(e=>{console.error(e);process.exit(1)});
'''
    result = subprocess.run(['node', '-e', harness, json.dumps(dict(active=active, recent=recent, now=now, dispatch_time=dispatch_time))],
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


@pytest.mark.parametrize('at,decision', [('06:49:59', 'dispatched'), ('06:50:00', 'issuance_deadline'), ('06:54:45', 'issuance_deadline')])
def test_recovery_reserves_setup_time_before_issuance_deadline(at, decision):
    result = recovery(now=f'2026-09-07T{at}Z')
    assert result['outputs']['recovery'] == decision
    assert len(result['dispatches']) == int(decision == 'dispatched')


def test_recovery_checks_the_clock_after_api_calls():
    result = recovery(now='2026-09-07T06:49:59Z', dispatch_time='2026-09-07T06:50:01Z')
    assert result['outputs']['recovery'] == 'issuance_deadline'
    assert not result['dispatches']


@pytest.mark.parametrize('change,message', [
    ({'price_quantiles': [1800, float('nan'), 2400]}, 'prices'),
    ({'price_quantiles': [2400, 2100, 1800]}, 'prices'),
    ({'reference_price': None}, 'prices'),
    ({'terminal_down_flat_up': [.5, .5, .5]}, 'probabilities'),
    ({'hit_down': 1.1}, 'probabilities'),
    ({'target_end': '2026-09-07T13:01:00Z'}, 'window'),
    ({'available_at': '2026-09-07T06:09:00Z'}, 'receipt'),
    ({'slot': '2026-09-07T05:00:00Z'}, 'issuance'),
])
def test_ready_forecasts_validate_values_and_time_contract(change, message):
    data = publication(); data['current'][0].update(change)
    with pytest.raises(ValueError, match=message):
        verify(data, now=datetime.fromisoformat('2026-09-07T06:38:00+00:00'), require_ready=True)


def test_exact_publication_rejects_changed_values_even_with_matching_ids():
    expected = publication(); actual = deepcopy(expected)
    actual['current'][0]['price_quantiles'][1] += 1
    with pytest.raises(ValueError, match='payload differs'):
        verify(actual, expected, now=datetime.fromisoformat('2026-09-07T06:38:00+00:00'))


def test_previous_records_do_not_turn_an_empty_or_partial_release_ready():
    when = datetime.fromisoformat('2026-09-07T06:38:00+00:00')
    for count in (0, 5):
        data = publication(); data['recent_issued'] = deepcopy(data['current'])
        data['current'] = data['current'][:count]; data['status'] = 'delayed'
        assert verify(data, data, now=when)
        with pytest.raises(ValueError, match='delayed'):
            verify(data, data, now=when, require_ready=True)


@pytest.mark.parametrize('status,require_ready', [('delayed', False), ('delayed', True), ('ready', True)])
def test_cli_distinguishes_payload_delivery_from_forecast_readiness(tmp_path, monkeypatch, capsys, status, require_ready):
    import io
    import sys
    import scripts.verify_event_site as module
    data = publication(); data['status'] = status
    if status == 'delayed': data['current'] = []
    expected = tmp_path/'expected.json'; expected.write_text(json.dumps(data))
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None): return datetime.fromisoformat('2026-09-07T06:38:00+00:00')
    monkeypatch.setattr(module, 'datetime', Clock)
    def fetch(url, **kwargs):
        resource = url.split('?')[0].rsplit('/', 1)[-1]
        body = json.dumps(data) if resource == 'signals.json' else (ROOT/'forecast_site/public'/(resource or 'index.html')).read_text()
        return io.BytesIO(body.encode())
    monkeypatch.setattr(module, 'urlopen', fetch)
    monkeypatch.setattr(module, 'verify_archives', lambda *a: None)
    monkeypatch.setattr(module.time, 'sleep', lambda _: pytest.fail('readiness must fail without polling'))
    monkeypatch.setattr(sys, 'argv', ['verify_event_site.py', '--expected', str(expected)] + (['--require-ready'] if require_ready else []))
    if require_ready and status == 'delayed':
        with pytest.raises(SystemExit): module.main()
        assert 'Event site verified:' not in capsys.readouterr().out
    else:
        module.main(); output = capsys.readouterr().out
        assert ('Event site verified:' if require_ready else 'Event payload verified:') in output


def test_publisher_preserves_receipts_before_failing_readiness(tmp_path):
    import os
    import sys
    binaries = tmp_path/'bin'; binaries.mkdir()
    trace = tmp_path/'trace'; runner = tmp_path/'runner'
    (runner/'event-site/forecast_site/public').mkdir(parents=True)
    for name in ('forecast_site/public/index.html', 'forecast_site/public/events.js', 'forecast_site/public/event_diagnostics.js', 'forecast_site/vercel.json',
                 'forecast_site/public/sitemap.xml', 'forecast_site/public/robots.txt',
                 'lake/signals/signals.json', 'lake/signals/replay.json'):
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('{}')
    (binaries/'git').write_text('#!/bin/sh\nexit 0\n')
    (binaries/'python').write_text(f'#!{sys.executable}\n' + '''import os,sys,pathlib
args=sys.argv[1:]
kind='search' if 'publish_search_assets.py' in args[0] else 'archive' if 'copy_event_evidence.py' in args[0] else 'receipts' if args==['-'] else 'readiness' if '--require-ready' in args else 'payload' if 'verify_event_site.py' in args[0] else 'export'
with open(os.environ['OUTLOOK_TEST_TRACE'],'a') as f:f.write(kind+'\\n')
if kind=='receipts':sys.stdin.read()
if kind=='export':pathlib.Path(args[-1]).write_text('')
if kind=='readiness':sys.exit(1)
''')
    for path in binaries.iterdir(): path.chmod(0o755)
    (tmp_path/'scripts').mkdir()
    (tmp_path/'scripts/persist_event_ledger.sh').write_text('printf "persisted\\n" >> "$OUTLOOK_TEST_TRACE"\n')
    result = subprocess.run(['bash', str(ROOT/'scripts/publish_events.sh')], cwd=tmp_path, capture_output=True, text=True,
                            env={**os.environ, 'PATH': str(binaries)+os.pathsep+os.environ['PATH'],
                                 'RUNNER_TEMP': str(runner), 'OUTLOOK_TEST_TRACE': str(trace)})
    assert result.returncode != 0
    assert trace.read_text().splitlines() == ['export', 'archive', 'search', 'payload', 'receipts', 'persisted', 'readiness']
