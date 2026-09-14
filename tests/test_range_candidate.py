from copy import deepcopy
import json
import numpy as np
import pandas as pd
import pytest
from signal_pipeline import range_candidate as rc
from signal_pipeline.data import utc
from signal_pipeline.protocol import digest,PROTOCOL_HASH,training_hash


def seed(now='2026-09-14T06:00:00Z'):
    now=utc(now)
    value={'policy':rc.POLICY,'protocol_hash':PROTOCOL_HASH,'training_hash':training_hash(),
           'available_at':now.isoformat(),'data_as_of':(now-pd.Timedelta(days=4)).isoformat(),'horizons':{}}
    for h in rc.HORIZONS:
        value['horizons'][str(h)]=[{'slot':t.isoformat(),'target_end':(t+pd.Timedelta(hours=h+1)).isoformat(),
                                   'losses':[.3,.4,.1]} for t in pd.date_range(now.floor('D')-pd.Timedelta(days=200),periods=100)]
    value['seed_id']=digest(value)
    return value


def test_seed_integrity_availability_and_atomic_install(tmp_path):
    value=seed();source=tmp_path/'seed.json';source.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='unavailable'):rc.install_seed(source,tmp_path/'state',now='2026-09-13')
    assert not (tmp_path/'state/range_seed.json').exists()
    rc.install_seed(source,tmp_path/'state',now='2026-09-14T07:00Z')
    saved=(tmp_path/'state/range_seed.json').read_bytes()
    value['horizons']['6'][0]['losses'][0]=999;source.write_text(json.dumps(value))
    with pytest.raises(ValueError,match='integrity'):rc.install_seed(source,tmp_path/'state',now='2026-09-14T07:00Z')
    assert (tmp_path/'state/range_seed.json').read_bytes()==saved


def test_only_mature_available_feedback_changes_weights():
    value=seed();slot=utc('2026-09-14T07:00Z');rows=value['horizons']['6']
    weights,audit=rc.weights_for(rows,[],slot,6)
    assert weights[2]>weights[0]>weights[1] and np.isclose(sum(weights),1)
    r={'slot':'2026-09-12T00:00:00+00:00','issued_at':'2026-09-12T00:10:00Z','target_end':'2026-09-12T07:00:00Z',
       'horizon_seconds':21600,'reference_price':100,'range_experts':[[90,100,110],[85,100,115],[99,100,101]],
       'outcome':{'actual_price':150,'truth_available_at':'2026-09-14T06:30:00Z'},'outcome_evaluated_at':'2026-09-14T06:30:00Z'}
    delayed,_=rc.weights_for(rows,[r],slot,6)
    assert np.array_equal(weights,delayed)
    r['outcome']['truth_available_at']='2026-09-13T00:00Z';r['outcome_evaluated_at']='2026-09-13T00:00Z'
    changed,counts=rc.weights_for(rows,[r],slot,6)
    assert not np.array_equal(weights,changed) and counts['live_rows']==1
    future=deepcopy(r);future['slot']='2026-09-14T00:00Z';future['target_end']='2026-09-14T07:00Z'
    unchanged,_=rc.weights_for(rows,[future],slot,6)
    assert np.array_equal(weights,unchanged)


def test_range_mixture_preserves_center_and_checks_inputs():
    result=rc.mix_bounds([[90,100,110],[80,90,120],[95,105,115]],np.array([.2,.3,.5]))
    assert result==[89.5,100.,115.5]
    with pytest.raises(ValueError):rc.mix_bounds([[100,90,110]]*3,np.ones(3)/3)


def test_real_ledger_issue_is_idempotent_and_does_not_change_probabilities(tmp_path,monkeypatch):
    from tests.test_event_recovery import publication
    value=seed();(tmp_path/'range_seed.json').write_text(json.dumps(value))
    monkeypatch.setattr(rc,'settle',lambda *a,**k:0)
    a=publication()['current'][0]
    origin=utc('2026-09-14T07:00Z')
    for key in ('slot','input_cutoff'):a[key]=origin.isoformat()
    a.update(available_at=(origin+pd.Timedelta(minutes=1)).isoformat(),issued_at=(origin+pd.Timedelta(minutes=2)).isoformat(),
             window_start=(origin+pd.Timedelta(hours=1)).isoformat(),target_end=(origin+pd.Timedelta(hours=7)).isoformat(),
             training_target_end='2026-09-01T00:00Z',validation_target_end='2026-09-01T00:00Z',model_version='active',
             baseline={'price_quantiles':[1700.,2000.,2500.]})
    b=deepcopy(a);b['model_version']='candidate';b['price_quantiles']=[1850.,2050.,2300.]
    original=deepcopy(a)
    out,_,errors=rc.infer(tmp_path,[a],[b],None,now=origin+pd.Timedelta(minutes=8))
    assert not errors and a==original
    assert out[0]['price_quantiles'][1]==a['price_quantiles'][1]
    assert out[0]['terminal_down_flat_up']==a['terminal_down_flat_up']
    assert out[0]['hit_up']==a['hit_up'] and out[0]['hit_down']==a['hit_down']
    b['price_quantiles']=[1000.,2000.,3000.]
    again,_,_=rc.infer(tmp_path,[a],[b],None,now=origin+pd.Timedelta(minutes=20))
    assert again==out
    assert not (tmp_path/'issued.db').exists()


def test_missing_candidate_does_not_silently_change_mixture(tmp_path,monkeypatch):
    (tmp_path/'range_seed.json').write_text(json.dumps(seed()))
    monkeypatch.setattr(rc,'settle',lambda *a,**k:0)
    a={'slot':'2026-09-14T07:00Z','horizon_seconds':21600}
    out,_,errors=rc.infer(tmp_path,[a],[],None,now='2026-09-14T07:08Z')
    assert not out and errors[0]['reason']=='current paired candidate unavailable'
