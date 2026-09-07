"""Matched origin slices, maturity boundaries, purge audit and exported evidence."""
import copy
import csv

import numpy as np
import pandas as pd
import pytest

from scripts.export_event_segments import export
from signal_pipeline.engine import check_replay_cutoff
from signal_pipeline.evaluate import report_replay
from signal_pipeline.segments import origin_market_states


def rows(horizon=24):
    result=[]
    for i, slot in enumerate(pd.date_range('2025-11-01', '2026-09-07', tz='UTC')):
        common=dict(slot=slot.isoformat(), target_end=(slot+pd.Timedelta(hours=horizon+1)).isoformat(),
                    horizon_hours=horizon, selected_model='catboost', reference_price=2000, actual_price=2020,
                    **{'return': .01, 'up': i%2, 'down': (i+1)%2, 'terminal': i%3},
                    p_down=.2, p_flat=.5, p_up=.3, threshold_up=.8, threshold_down=.8,
                    q10=-.1, q50=.01, q90=.1, trend_state=('up','down','range')[i%3],
                    volatility_state='high' if i%4==0 else 'normal')
        result += [dict(common, model=name, hit_up=prob, hit_down=prob) for name, prob in
                   [('selected', .6), ('climatology', .5), ('catboost', .6)]]
    return result


def test_all_period_market_intersections_use_identical_origins_and_unchanged_model_choice():
    data=rows()
    # Remove different origins from different candidates: every comparison must match.
    data=[r for r in data if not (r['model']=='climatology' and r['slot'].startswith('2026-02-01'))
          and not (r['model']=='catboost' and r['slot'].startswith('2026-02-02'))]
    report=report_replay(data,24,as_of='2026-09-07T12:00:00Z')
    assert not any(p['slot'].startswith(('2026-02-01','2026-02-02')) for p in report['points'])
    for result in report['segments']['results'].values():
        assert all(m['rows']==result['common_origins'] for m in result['models'])
        assert set(result['selected_model_counts']) <= {'catboost'}
    selected=report['segments']['results']['recent_90|trend_up']
    expected=[p for p in report['points'] if p['slot'] >= '2026-06-10' and p['trend_state']=='up']
    assert selected['common_origins']==len(expected)
    assert selected['nonoverlapping_selected']['rows'] <= selected['common_origins']


def test_recent_windows_share_data_asof_and_do_not_slide_back_to_matured_long_horizon():
    short=report_replay(rows(6),6,as_of='2026-09-07T12:00:00Z')['segments']
    long=report_replay(rows(720),720,as_of='2026-09-07T12:00:00Z')['segments']
    assert short['periods']==long['periods']
    assert short['results']['recent_30|all']['common_origins']==30
    assert long['results']['recent_30|all']['common_origins']==0
    assert long['results']['recent_90|all']['common_origins']==60
    assert long['results']['recent_90|all']['paired_event_brier'] is None
    assert long['results']['recent_90|all']['nonoverlapping_selected']['rows']==2


def test_future_outcomes_cannot_choose_segment_membership():
    original=rows(); changed=copy.deepcopy(original)
    for r in changed: r['up']=1-r['up']; r['return']=-r['return']
    a=report_replay(original,24,as_of='2026-09-07T12:00:00Z')['segments']['results']
    b=report_replay(changed,24,as_of='2026-09-07T12:00:00Z')['segments']['results']
    assert {k:v['common_origins'] for k,v in a.items()}=={k:v['common_origins'] for k,v in b.items()}


@pytest.mark.parametrize('bad', ['duplicate','truth','missing_model_after_maturity'])
def test_inconsistent_replay_rows_are_rejected_or_unscored(bad):
    data=rows()[:6]
    if bad=='duplicate':data.append(data[0].copy())
    if bad=='truth':data[1]['up']=1-data[1]['up']
    if bad=='missing_model_after_maturity':
        for r in data:
            if r['model']=='climatology':r['target_end']='2027-01-01T00:00:00Z'
        assert report_replay(data,24,as_of='2026-09-07T12:00:00Z')['status']=='insufficient_history'
    else:
        with pytest.raises(ValueError):report_replay(data,24)


@pytest.mark.parametrize('key,value', [('training_target_end','2026-08-31T23:00:00Z'),
    ('validation_target_end','2026-09-01T00:00:00Z'),('validation_target_end','NaT'),
    ('fit_cutoff','2026-10-01T00:00:00Z')])
def test_every_restored_monthly_checkpoint_must_pass_the_purge_boundary(key,value):
    bundle=dict(fit_cutoff='2026-09-01T00:00:00Z',training_target_end='2026-08-31T22:00:00Z',
                validation_target_end='2026-08-31T20:00:00Z')
    cutoff=pd.Timestamp('2026-09-01T00:00:00Z');check_replay_cutoff(bundle,cutoff)
    with pytest.raises(ValueError):check_replay_cutoff({**bundle,key:value},cutoff)


def test_state_thresholds_are_fixed_and_unavailable_history_is_not_a_quiet_market():
    sigma=.01
    features=pd.DataFrame({'eth_vol_720':[sigma]*4,'eth_ret_24':[2*sigma*np.sqrt(24),-2*sigma*np.sqrt(24),0,np.nan],
                           'eth_vol_24':[.02,.01,.005,np.nan]})
    states=origin_market_states(features)
    assert states.trend_state.tolist()==['up','down','range','unknown']
    assert states.volatility_state.tolist()==['high','normal','normal','unknown']


def test_download_reports_zero_mature_origins_without_invented_scores(tmp_path):
    report=report_replay(rows(720),720,as_of='2026-09-07T12:00:00Z')
    output=tmp_path/'segments.csv'
    assert export({'horizons':{'720':report}},output)>0
    with output.open(encoding='utf-8-sig') as handle: result=list(csv.DictReader(handle))
    empty=[r for r in result if r['period']=='recent_30' and r['market_state']=='all']
    assert len(empty)==1 and empty[0]['common_origins']=='0' and empty[0]['event_brier']==''
    baseline=next(r for r in result if r['model']=='climatology')
    assert float(baseline['event_skill_vs_frequency'])==0
    assert baseline['selected_brier_difference_low95']==''
