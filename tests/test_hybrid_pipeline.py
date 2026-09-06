"""Causality, real attention fits, immutable issuance and complete cohort gates."""
import copy
import io
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest
import torch

from hybrid_pipeline import data,engine,evaluate,models
from hybrid_pipeline.protocol import BASE_MODELS,PROTOCOL
from hybrid_pipeline.ledger import publish_issued
from research_pipeline.protocol import FLOW_COLUMNS


def fixture(n=950):
    rng=np.random.default_rng(717);idx=pd.date_range('2023-01-01',periods=n)
    master=pd.DataFrame(index=idx)
    for asset in ('eth','btc'):
        close=100*np.exp(np.cumsum(rng.normal(.0002,.02,n)))
        master[f'{asset}_close']=close;master[f'{asset}_open']=close*.998
        master[f'{asset}_high']=close*1.02;master[f'{asset}_low']=close*.98
        master[f'{asset}_volume']=rng.uniform(100,300,n)
    flow=pd.DataFrame(rng.normal(0,.05,(n,len(FLOW_COLUMNS))),index=idx,columns=FLOW_COLUMNS)
    flow['market_data_excluded']=0
    now=idx[-1]+pd.Timedelta(days=1,hours=6)
    return master,flow,now


def base_fixture():
    master,flow,now=fixture(1100);raw,x=data.features(master,flow,now=now)
    rows=[]
    for h in (7,30):
        for origin in raw.index[600:]:
            rows.append({'origin':origin,'horizon':h,**{name:.1 if name.startswith('cat') else -.05 for name in BASE_MODELS}})
    return master,raw,engine.settle_table(pd.DataFrame(rows),raw),now


def test_future_and_unverifiable_macro_values_cannot_change_past_features():
    master,flow,now=fixture();raw,a=data.features(master,flow,now=now)
    origin=raw.index[750];changed=master.copy();changed.loc[changed.index>=origin]*=3
    changed['fred_unreleased_future']=np.arange(len(changed))*1e9
    future_flow=flow.copy();future_flow.loc[future_flow.index>=origin]*=4
    _,b=data.features(changed,future_flow,now=now)
    pd.testing.assert_frame_equal(a.loc[:origin],b.loc[:origin])
    col=FLOW_COLUMNS[0]
    assert a.loc[origin,col]==flow.loc[origin-pd.Timedelta(days=2),col]


def test_raw_price_gaps_and_incomplete_ohlcv_stop_the_replay():
    m,f,now=fixture()
    with pytest.raises(ValueError,match='gap'):data.features(m.drop(m.index[300]),f,now=now)
    m.iloc[-1,m.columns.get_loc('btc_high')]=np.nan
    with pytest.raises(ValueError,match='OHLCV'):data.features(m,f,now=now)


def test_purge_applies_to_outer_and_inner_targets():
    m,f,now=fixture();raw,x=data.features(m,f,now=now);month=x.index[-1].replace(day=1)
    for h in (7,30):
        y,_=data.targets(raw,h);p=data.training_positions(x.index,y,month,h,'long');train,val=data.inner_split(x.index,p,h)
        gap=pd.Timedelta(days=PROTOCOL['embargo_days'][str(h)])
        assert x.index[p[-1]]+pd.Timedelta(days=h)<month-gap
        assert x.index[train[-1]]+pd.Timedelta(days=h)<x.index[val[0]]-gap


def test_scaler_uses_training_values_and_keeps_unseen_features_inactive():
    values=np.column_stack([np.arange(100),np.full(100,np.nan)])
    scaler=data.PastScaler().fit(values[:60]);changed=values.copy();changed[60:,0]=1e8;changed[60:,1]=100
    assert np.all(scaler.transform(changed)[60:,1]==0)
    np.testing.assert_array_equal(scaler.transform(values)[:60],scaler.transform(changed)[:60])
    np.testing.assert_array_equal(scaler.transform(changed),data.PastScaler.from_state(scaler.state()).transform(changed))


def test_transformer_sequences_end_at_the_prediction_origin():
    values=np.arange(200,dtype=np.float32).reshape(100,2)
    seq=models.sequences(values,[31,50],32).numpy()
    np.testing.assert_array_equal(seq[0,-1],values[31]);np.testing.assert_array_equal(seq[1,0],values[19])
    with pytest.raises(ValueError,match='start'):models.sequences(values,[10],32)


@pytest.mark.parametrize('name',['cat_short','transformer_short'])
def test_real_models_roundtrip_and_ignore_future_observations(name):
    m,f,now=fixture(850);raw,x=data.features(m,f,now=now);y,_=data.targets(raw,7);month=x.index[-1].replace(day=1)
    p=data.training_positions(x.index,y,month,7,'short');past=x.index<month
    state,meta=models.fit_model(name,x.loc[past],y.loc[past],p,7)
    buffer=io.BytesIO();torch.save(state,buffer);buffer.seek(0);restored=torch.load(buffer,weights_only=True)
    origin=month+pd.Timedelta(days=1);pos=x.index.get_loc(origin)
    first=models.predict_model(restored,x,[pos]);changed=x.copy();changed.loc[changed.index>origin]=1e6
    second=models.predict_model(restored,changed,[pos]);np.testing.assert_allclose(first,second,rtol=0,atol=0)
    assert meta['steps']>=1
    if name.startswith('transformer'):
        assert any('self_attn' in key for key in state['torch_state'])


def test_future_outcomes_cannot_change_prior_blend_or_uncertainty():
    _,raw,base,_=base_fixture();first,decisions=evaluate.make_paths(base,raw)
    cutoff=pd.Timestamp('2025-06-01');changed=base.copy();changed.loc[changed.target>=cutoff,'actual_return']=5
    second,changed_decisions=evaluate.make_paths(changed,raw)
    cols=['origin','horizon','model','predicted_return','p_up','lower_return','upper_return']
    # Outcomes during June are unavailable to the June monthly choice.
    pd.testing.assert_frame_equal(first.loc[first.origin<'2025-07-01',cols].reset_index(drop=True),second.loc[second.origin<'2025-07-01',cols].reset_index(drop=True))
    for d in decisions:
        assert d['selection_latest_target'] is None or d['selection_latest_target']<d['month']


def test_point_guard_limits_extrapolation_without_using_future_prices():
    _,raw,base,_=base_fixture();base.loc[:,'transformer_long']=1000.
    first,bounds=evaluate.guard_points(base,raw)
    assert first.transformer_long_guarded.all()
    assert (first.transformer_long*first.sigma < 3).all()
    cutoff=pd.Timestamp('2025-06-01');changed=raw.copy();changed.loc[changed.index>=cutoff,'eth_close']*=100
    second,later_bounds=evaluate.guard_points(base,changed)
    before=base.origin<'2025-07-01'
    pd.testing.assert_frame_equal(first.loc[before],second.loc[before])
    for (h,month),windows in bounds.items():
        for value in windows.values():
            assert pd.Timestamp(value['last_training_target'])<pd.Timestamp(month)-pd.Timedelta(days=PROTOCOL['embargo_days'][str(h)])
        if month<='2025-06-01':assert windows==later_bounds[(h,month)]


def test_missing_calendar_cohorts_cannot_be_published(monkeypatch):
    _,raw,base,now=base_fixture();monkeypatch.setattr(evaluate,'block_interval',lambda *args:[-.1,.1])
    bad=base.drop(base[(base.horizon==7)].index[50])
    with pytest.raises(ValueError,match='Missing eligible'):evaluate.report(bad,raw,[],{},now=now)


def test_current_bundle_is_reused_when_a_new_day_changes_only_test_inputs(tmp_path,monkeypatch):
    m,f,now=fixture(850);calls=[]
    def fake_fit(name,x,y,positions,horizon):
        calls.append(name);return {'value':.1},{'seconds':0,'steps':1}
    monkeypatch.setattr(engine,'fit_model',fake_fit)
    monkeypatch.setattr(engine,'predict_model',lambda state,x,positions:np.full(len(positions),state['value']))
    a,meta=engine.replay(m,f,7,tmp_path,part='current',now=now)
    assert len(calls)==4
    updated=m.copy();updated.iloc[-1,updated.columns.get_loc('eth_close')]*=1.01
    b,meta=engine.replay(updated,f,7,tmp_path,part='current',now=now)
    assert len(calls)==4 and meta['refitted_months']==0 and meta['reused_current_bundles']==1
    np.testing.assert_array_equal(a[BASE_MODELS],b[BASE_MODELS])


def test_cache_uses_source_observations_and_invalidates_changed_flow(tmp_path,monkeypatch):
    m,f,now=fixture(850);calls=[];original_features=engine.features
    def fake_fit(name,x,y,positions,horizon):
        calls.append(name);return {'value':.1},{'seconds':0,'steps':1}
    monkeypatch.setattr(engine,'fit_model',fake_fit)
    monkeypatch.setattr(engine,'predict_model',lambda state,x,positions:np.full(len(positions),state['value']))
    first,_=engine.replay(m,f,7,tmp_path,part='current',now=now)
    def rounded_features(*args,**kwargs):
        raw,x=original_features(*args,**kwargs)
        x['eth_btc_correlation_30']+=1e-15
        return raw,x
    monkeypatch.setattr(engine,'features',rounded_features)
    again,meta=engine.replay(m,f,7,tmp_path,part='current',now=now)
    assert len(calls)==4 and meta['cached_months']==1 and meta['refitted_months']==0
    pd.testing.assert_frame_equal(first,again)
    changed=f.copy();changed.loc[changed.index[-100],FLOW_COLUMNS[0]]+=.01
    _,meta=engine.replay(m,changed,7,tmp_path,part='current',now=now)
    assert len(calls)==8 and meta['refitted_months']==1


def test_new_live_record_is_immutable_and_never_backdates_replay(tmp_path,monkeypatch):
    master,raw,base,now=base_fixture();monkeypatch.setattr(evaluate,'block_interval',lambda *args:[-.1,.1])
    payload,_=evaluate.report(base,raw,[],{'master':'fixture','flow':'fixture'},now=now)
    first=publish_issued(copy.deepcopy(payload),master,tmp_path/'issued.db',now=now)
    changed=copy.deepcopy(payload);changed['horizons']['30']['current']['predicted_price']=99999
    again=publish_issued(changed,master,tmp_path/'issued.db',now=now)
    assert first['prospective']['issued']==2 and again['prospective']['new_issues']==0
    assert first['horizons']['30']['current']==again['horizons']['30']['current']
    retired=publish_issued(copy.deepcopy(payload),master,tmp_path/'issued.db',now=now+pd.Timedelta(days=1),issue_new=False)
    assert retired['prospective']['issued']==2 and retired['prospective']['new_issues']==0
    with pytest.raises(ValueError,match='backdate'):publish_issued(copy.deepcopy(payload),master,tmp_path/'late.db',now=now+pd.Timedelta(days=1))
    conn=sqlite3.connect(tmp_path/'issued.db')
    with pytest.raises(sqlite3.IntegrityError):conn.execute('DELETE FROM issued')
    conn.close()


def test_report_probabilities_and_point_identity(monkeypatch):
    _,raw,base,now=base_fixture();monkeypatch.setattr(evaluate,'block_interval',lambda *args:[-.1,.1])
    payload,rows=evaluate.report(base,raw,[],{},now=now)
    for h,d in payload['horizons'].items():
        assert len(d['leaderboard'])==8
        point=d['current'];assert point['lower_price']<=point['upper_price']
        assert sum(point['probability_down_flat_up'])==pytest.approx(1)
        assert point['predicted_price']==pytest.approx(point['reference_price']*(1+point['predicted_return']))
        assert d['default_chart']=='optimized_hybrid'
    json.dumps(payload,allow_nan=False)
