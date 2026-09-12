import json
import numpy as np
import pandas as pd
import pytest

from signal_pipeline import adaptive_research as study
from signal_pipeline.data import utc
from signal_pipeline.protocol import HORIZONS, PROTOCOL_HASH, training_hash
from tests.test_event_diagnostics import row


def source(n=480, h=720):
    rows = []
    for i, slot in enumerate(pd.date_range('2020-01-01', periods=n, tz='UTC')):
        for name in study.EXPERTS:
            rows.append({**row(), 'slot': slot.isoformat(), 'target_end': (slot+pd.Timedelta(hours=h+1)).isoformat(),
                         'model': name, 'model_version': 'fixture', 'return': float(.15*np.sin(i/11)),
                         'terminal': i % 3, 'up': int(i % 5 == 0), 'down': int(i % 7 == 0),
                         'q50': 0. if name == 'no_change' else .02, 'trend_state': 'range', 'volatility_state': 'normal'})
    return pd.DataFrame(rows)


def test_thirty_day_feedback_and_calibration_are_fully_mature_before_prediction():
    data = source(); before = data.copy(deep=True)
    rows, audit = study.replay(data, np.full(480, .2), 720)
    assert before.equals(data)
    first = next(r for r in audit if r['scored'])
    assert first['feedback_rows'] == 365
    assert utc(first['slot']) == utc('2021-01-30')
    for r in rows:
        if r['feedback_target_end']:
            assert utc(r['feedback_target_end']) < utc(r['slot'])-pd.Timedelta(hours=1)
        if r['calibration_cutoff']:
            assert utc(r['calibration_target_end']) < utc(r['calibration_cutoff'])-pd.Timedelta(hours=1)
            assert utc(r['calibration_cutoff']) <= utc(r['slot'])
        assert r['q10'] <= r['q50'] <= r['q90']
        assert np.isclose(sum(r[k] for k in study.P[:3]), 1)
        assert all(0 <= r[k] <= 1 for k in study.P)


def test_future_and_not_yet_matured_outcomes_cannot_change_earlier_forecasts():
    data = source(); scale = np.full(480, .2); cutoff = '2021-03-01'
    original, _ = study.replay(data, scale, 720)
    revised = data.copy(); future_truth = pd.to_datetime(revised.target_end, utc=True) >= utc(cutoff)
    revised.loc[future_truth, 'return'] += .3
    revised.loc[future_truth, ['up', 'down']] = 1-revised.loc[future_truth, ['up', 'down']]
    revised.loc[future_truth, 'terminal'] = (revised.loc[future_truth, 'terminal']+1) % 3
    future_input = pd.to_datetime(revised.slot, utc=True) >= utc(cutoff)
    revised.loc[future_input, study.Q] += .1
    changed_scale = scale.copy(); changed_scale[np.arange(480) >= (utc(cutoff)-utc('2020-01-01')).days] *= 3
    changed, _ = study.replay(revised, changed_scale, 720)
    keys = study.P+study.Q+['threshold_up', 'threshold_down', 'model_version', 'feedback_target_end', 'calibration_cutoff']
    a = [{k:r[k] for k in keys} for r in original if utc(r['slot']) < utc(cutoff)]
    b = [{k:r[k] for k in keys} for r in changed if utc(r['slot']) < utc(cutoff)]
    assert a == b


def test_unpaired_truth_and_invalid_volatility_fail():
    data = source(10, 6)
    data.loc[0, 'return'] += .1
    with pytest.raises(ValueError, match='mismatch'): study.replay(data, np.ones(10), 6)
    with pytest.raises(ValueError, match='volatility'): study.replay(source(10, 6), np.zeros(10), 6)


def test_wrong_source_snapshot_preserves_previous_complete_report(tmp_path, monkeypatch):
    (tmp_path/'historical_study.json').write_text(json.dumps({'status':'complete',
        'horizons':{str(h):{} for h in HORIZONS}, 'protocol_hash':PROTOCOL_HASH,
        'training_hash':training_hash(), 'data_as_of':'2026-01-01', 'source_snapshot':'expected'}))
    (tmp_path/'adaptive_study.json').write_text('previous complete report')
    monkeypatch.setattr(study,'read_bars',lambda _:None)
    monkeypatch.setattr(study,'source_hash',lambda *a:'different')
    with pytest.raises(ValueError, match='source snapshot mismatch'): study.run_review(tmp_path)
    assert (tmp_path/'adaptive_study.json').read_text() == 'previous complete report'


def test_strict_delay_also_applies_to_short_horizon_and_budget():
    data = source(110, 6)
    with pytest.raises(TimeoutError): study.replay(data, np.ones(110), 6, deadline=0)
    _, audit = study.replay(data, np.ones(110), 6)
    assert next(i for i,r in enumerate(audit) if r['scored']) == 90


def test_soft_weights_reduce_a_worse_expert_and_residual_quantiles_are_ordered():
    weights = study.mix_weights(np.array([.1,.2]), np.array([.5,.5]))
    assert weights[0] > weights[1] and np.isclose(weights.sum(),1)
    lo, hi = study.weighted_quantile(np.array([2.,-2.,0.]), np.ones(3), [.1,.9])
    assert lo < 0 < hi
