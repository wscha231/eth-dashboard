"""Test causal selection, checkpoint identity and complete-cohort publication."""
import json

import numpy as np
import pandas as pd
import pytest

import backtest_pipeline.replay as replay
from backtest_pipeline.protocol import CANDIDATES, PROTOCOL
from research_pipeline.protocol import FLOW_COLUMNS


def replay_table():
    rows=[]
    for horizon in (7,30):
        for origin in pd.date_range('2024-01-01',periods=250):
            actual=.10 if origin < pd.Timestamp('2024-05-01') else -.10
            for model in CANDIDATES:
                pred=.09 if model=='extra_trees' else -.09 if model=='random_forest' else 0.
                rows.append({'origin':origin,'target':origin+pd.Timedelta(days=horizon),'horizon':horizon,
                             'model':model,'predicted_return':pred,'actual_return':actual,
                             'reference_price':100.,'actual_price':100*(1+actual),
                             'event_threshold':.05,'fit_cutoff':origin.replace(day=1).isoformat()})
    return pd.DataFrame(rows)


def test_selector_never_uses_unmatured_or_future_test_outcomes():
    data=replay_table(); baseline=replay.past_only_selector(data)
    assert baseline.loc[baseline.origin<'2024-04-01','selected_model'].eq('no_change_anchor').all()
    for row in baseline.itertuples():
        if row.selection_latest_target:
            assert pd.Timestamp(row.selection_latest_target) < row.origin.replace(day=1)
    changed=data.copy()
    # May outcomes are unavailable to the May model-choice decision.
    changed.loc[changed.target>='2024-05-01','actual_return']=1000
    updated=replay.past_only_selector(changed)
    assert baseline.loc[baseline.origin<'2024-06-01','selected_model'].tolist()==updated.loc[updated.origin<'2024-06-01','selected_model'].tolist()


def test_training_mask_purges_overlapping_labels_and_embargo():
    index=pd.date_range('2016-01-01','2026-08-01');month=pd.Timestamp('2026-08-01')
    for horizon in (7,30):
        kept=index[replay.train_mask(index,month,horizon)]
        assert kept.min()>=month-pd.DateOffset(years=PROTOCOL['train_years'][str(horizon)])
        assert kept.max()+pd.Timedelta(days=horizon)<month-pd.Timedelta(days=PROTOCOL['embargo_days'][str(horizon)])


def test_vendor_cadence_is_frozen_before_test_month(synthetic_ohlcv_with_companions):
    market=synthetic_ohlcv_with_companions.copy()
    market['fred_test']=np.nan
    market.loc[market.index[::30],'fred_test']=np.arange(len(market.index[::30]))
    month=market.index[650].replace(day=1);origin=month+pd.Timedelta(days=10)
    a,cols=replay.asof_wide_features(market,7,month)
    altered=market.copy();altered.loc[altered.index>origin]=altered.loc[altered.index>origin]*7
    altered.loc[altered.index>origin,'fred_test']=np.arange((altered.index>origin).sum())
    b,_=replay.asof_wide_features(altered,7,month)
    pd.testing.assert_series_equal(a.loc[origin,cols],b.loc[origin,cols],check_names=False)


def test_report_rejects_partial_or_mixed_date_cohorts():
    data=replay_table()
    with pytest.raises(ValueError,match='identical origins'):
        replay.build_report(data.iloc[1:],[],source_hashes={})
    data.loc[0,'target']+=pd.Timedelta(days=1)
    with pytest.raises(ValueError,match='Target time'):
        replay.build_report(data,[],source_hashes={})


def test_best_fixed_and_causal_selector_are_separate_reported_paths(monkeypatch):
    monkeypatch.setattr(replay,'block_interval',lambda *args:[-.01,.01])
    result,rows=replay.build_report(replay_table(),[],source_hashes={'test':'fixture'})
    for h,d in result['horizons'].items():
        assert d['matched_origins']==250
        assert len(d['leaderboard'])==len(CANDIDATES)+1
        assert len(d['yearly'])==len(CANDIDATES)+1
        assert 'past_only_selector' in d['points'][0]['returns']
        assert d['points'][0]['selected_model']=='no_change_anchor'
        score=next(r for r in d['leaderboard'] if r['model']=='no_change_anchor')
        assert score['mae_skill_vs_no_change']==pytest.approx(0)
        assert sum(score['actual_state_counts'].values())==250
    assert len(rows)==250*2*(len(CANDIDATES)+1)
    json.dumps(result,allow_nan=False)


def test_month_cache_reuses_predictions_but_refreshes_exact_actuals(tmp_path,monkeypatch,synthetic_ohlcv_with_companions):
    master=synthetic_ohlcv_with_companions.iloc[:550].copy()
    flow=pd.DataFrame(.1,index=master.index,columns=FLOW_COLUMNS)
    flow['market_data_excluded']=0
    called=[]
    def fake_fit(market,compact,price_columns,horizon,month,test_index):
        called.append((horizon,month))
        rows=[{'origin':d.isoformat(),'horizon':horizon,'model':model,'predicted_return':.02 if model!='no_change_anchor' else 0.,'fit_cutoff':month.isoformat()} for model in CANDIDATES for d in test_index]
        return {'rows':rows,'training_rows':500,'last_training_target':(month-pd.Timedelta(days=4)).isoformat(),'feature_count':1,'selected_features':['test'],'prepare_seconds':0,'fit_seconds':{},'total_seconds':0}
    monkeypatch.setattr(replay,'fit_month',fake_fit)
    now=master.index[-1]+pd.Timedelta(days=1,hours=6)
    first,meta=replay.replay_horizon(master,flow,7,tmp_path,now=now)
    assert len(called)>0
    count=len(called)
    again,meta2=replay.replay_horizon(master,flow,7,tmp_path,now=now)
    assert len(called)==count and meta2['cached_months']==meta2['months']
    pd.testing.assert_frame_equal(first,again)
    # The final source row is a realized target, after the last cached origin.
    revised=master.copy();revised.iloc[-1,revised.columns.get_loc('eth_close')]*=1.2
    corrected,meta3=replay.replay_horizon(revised,flow,7,tmp_path,now=now)
    assert len(called)==count
    np.testing.assert_allclose(first.predicted_return,corrected.predicted_return)
    assert corrected.actual_price.iloc[-1]!=first.actual_price.iloc[-1]
    altered=master.copy();altered.iloc[520,altered.columns.get_loc('eth_close')]*=1.1
    replay.replay_horizon(altered,flow,7,tmp_path,now=now)
    assert len(called)>count


def test_real_month_predicts_all_candidates_with_correct_cutoff(synthetic_ohlcv_with_companions):
    master=synthetic_ohlcv_with_companions.iloc[:730].copy()
    now=master.index[-1]+pd.Timedelta(days=1,hours=6)
    market=replay.model_daily_rows(master,now=now)
    flow=pd.DataFrame(.1,index=master.index,columns=FLOW_COLUMNS);flow['market_data_excluded']=0
    compact,price_columns,_,_=replay.compact_features(master,flow,now=now)
    month=market.index[-1].replace(day=1)
    result=replay.fit_month(market,compact,price_columns,7,month,market.index[-4:])
    assert len(result['rows'])==4*len(CANDIDATES)
    assert pd.Timestamp(result['last_training_target'])<month-pd.Timedelta(days=3)
    assert all(np.isfinite(r['predicted_return']) for r in result['rows'])
