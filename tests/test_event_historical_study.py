"""Historical evaluation must stay causal, resumable and separate from live evidence."""
import copy
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
import pytest

from signal_pipeline import historical_study as study
from signal_pipeline.data import utc
from signal_pipeline.diagnostics import export_archive, validate_object
from signal_pipeline.protocol import HORIZONS, PROTOCOL_HASH, training_hash
from tests.test_event_diagnostics import row


def test_thirty_day_targets_purge_every_candidate_boundary():
    cutoff = utc('2025-01-01'); span = pd.Timedelta(days=365)
    bundle = {'fit_cutoff': cutoff.isoformat(), 'models': {}, 'fitted_identity': study.model_identity({}), 'threshold_rows': 60}
    for key, end in [('training_target_end', cutoff-span), ('validation_target_end', cutoff-span),
                     ('calibration_target_end', cutoff-span/2), ('threshold_target_end', cutoff)]:
        bundle[key] = (end-pd.Timedelta(hours=2)).isoformat()
    study.audit_candidate(bundle, cutoff, 720)
    for key, end in [('training_target_end', cutoff-span), ('calibration_target_end', cutoff-span/2), ('threshold_target_end', cutoff)]:
        with pytest.raises(ValueError, match='crosses partition'):
            study.audit_candidate({**bundle, key: (end-pd.Timedelta(hours=1)).isoformat()}, cutoff, 720)


def test_monthly_origins_need_complete_thirty_day_future_window():
    idx = pd.date_range('2025-01-01', '2025-03-01', freq='h', tz='UTC')
    f = pd.DataFrame({'eth_ret_24': 1., 'reference_price': 100.}, index=idx)
    y = pd.DataFrame({'return': .1, 'target_end': idx+pd.Timedelta(hours=721)}, index=idx)
    result = study.eligible_origins(f, y, utc('2025-01-01'), utc('2025-02-15'))
    assert len(result) == 15 and (y.loc[result, 'target_end'] <= utc('2025-02-15')).all()
    assert (result.hour == 0).all()
    y.loc[result[0], 'return'] = np.nan
    assert len(study.eligible_origins(f, y, utc('2025-01-01'), utc('2025-02-15'))) == 14


def test_loss_intervals_match_dates_and_account_for_long_overlapping_targets():
    values = [{**row(), 'slot': s.isoformat(), 'target_end': (s+pd.Timedelta(hours=721)).isoformat()}
              for s in pd.date_range('2020-01-01', periods=900, tz='UTC')]
    result = study.paired_losses(values, values, 720)
    assert all(v['difference'] == v['lower95'] == v['upper95'] == 0 for v in result.values())
    assert all(v['block_days'] == 31 for v in result.values())
    assert len(study.nonoverlap(values)) == 30
    altered = copy.deepcopy(values); altered[0]['return'] += .1
    with pytest.raises(ValueError, match='truth mismatch'): study.paired_losses(values, altered, 720)
    with pytest.raises(ValueError, match='identical ordered'): study.paired_losses(values, values[::-1], 720)


def study_report(root):
    groups = {str(h): [{**row(), 'horizon_hours': h, 'target_end': (utc(row()['slot'])+pd.Timedelta(hours=h+1)).isoformat()}] for h in HORIZONS}
    archive = export_archive(root, groups, kind='candidate_history', as_of='2026-03-01T00:00Z')
    data = {'status': 'complete', 'protocol_hash': PROTOCOL_HASH, 'training_hash': training_hash(),
            'horizons': {str(h): {'origins': 1} for h in HORIZONS}, 'archive': archive}
    (root/'historical_study.json').write_text(json.dumps(data))
    return data


def test_import_cannot_overwrite_active_models_or_issued_ledger(tmp_path):
    from scripts.import_event_study import import_study
    source = tmp_path/'source'; source.mkdir(); target = tmp_path/'target'; target.mkdir()
    data = study_report(source)
    for name in ('issued.db', 'active.json', 'shadow_active.json'):
        (source/name).write_text('poison'); (target/name).write_text('original')
    import_study(source, target)
    for name in ('issued.db', 'active.json', 'shadow_active.json'): assert (target/name).read_text() == 'original'
    before = (target/'historical_study.json').read_bytes()
    manifest = validate_object(source, data['archive']); shard = manifest['horizons']['720']['shards'][0]
    (source/shard['path']).write_text('{}')
    with pytest.raises(ValueError, match='integrity'): import_study(source, target)
    assert (target/'historical_study.json').read_bytes() == before


def test_incumbent_merge_does_not_prune_separate_candidate_history(tmp_path):
    from scripts.copy_event_evidence import copy_evidence
    target = tmp_path/'target'; target.mkdir(); source = tmp_path/'source'; source.mkdir()
    data = study_report(target)
    copy_evidence(source, target)
    manifest = validate_object(target, data['archive'])
    assert len(validate_object(target, manifest['horizons']['720']['shards'][0])['points']) == 1


def test_incomplete_budget_retains_last_complete_report(tmp_path, monkeypatch):
    features = pd.DataFrame({'eth_ret_24': [1.]}, index=[utc('2025-01-01')])
    bars = pd.DataFrame({'product': ['ETH-USD', 'BTC-USD'], 'close_time': [utc('2025-01-01')]*2})
    monkeypatch.setattr(study, 'build_features', lambda _: features)
    monkeypatch.setattr(study, 'origin_market_states', lambda _: pd.DataFrame(index=features.index))
    monkeypatch.setattr(study, 'labels', lambda *a: None)
    (tmp_path/'historical_study.json').write_text('last complete report')
    with pytest.raises(TimeoutError): study.run_study(tmp_path, start='2025-01-01', bars=bars, budget_seconds=0)
    assert (tmp_path/'historical_study.json').read_text() == 'last complete report'
    assert json.loads((tmp_path/'historical_progress.json').read_text())['status'] == 'incomplete'


def test_artifact_selector_rejects_forks_pr_inputs_and_wrong_workflows():
    harness=r'''
const select=require('./scripts/event_historical_artifacts.cjs'),outputs={};
const candidates=[{id:1,head_repository_id:999,head_branch:'main'},{id:2,head_repository_id:123,head_branch:'main'},
 {id:3,head_repository_id:123,head_branch:'main'},{id:4,head_repository_id:123,head_branch:'main'}];
const github={rest:{repos:{listPullRequestsAssociatedWithCommit:async()=>({data:[]})},actions:{
 listArtifactsForRepo:async({name})=>({data:{artifacts:candidates.map(workflow_run=>({workflow_run}))}}),
 getWorkflowRun:async({run_id})=>({data:{status:'completed',conclusion:'success',event:run_id===2?'pull_request':'push',
 path:run_id===3?'.github/workflows/untrusted.yml':'.github/workflows/'+(outputs.source?'event_historical_study.yml':'event_research.yml')}})
}}};
(async()=>{await select({github,context:{eventName:'workflow_dispatch',repo:{owner:'a',repo:'b'},payload:{repository:{id:123}}},core:{setOutput:(k,v)=>outputs[k]=v}});console.log(JSON.stringify(outputs));})();
'''
    result = subprocess.check_output(['node', '-e', harness], text=True)
    assert json.loads(result) == {'source': '4', 'cache': '4'}
