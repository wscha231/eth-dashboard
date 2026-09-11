"""Bounded shadow candidates: separate heads and calibration tied to frozen fits.

This module deliberately does not change the incumbent training hash or active.json.
Historical diagnostics alone cannot authorize prospective promotion.
"""
import hashlib
import io
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .data import feature_columns, build_features, read_bars, utc
from .models import fit_models, predict_models, brier, labels, train_bundle
from .evaluate import metrics, paired_block_interval
from .protocol import digest, training_hash, PROTOCOL_HASH

POLICY = 'separate_heads_frozen_calibration_v1'


def model_identity(models):
    # Joblib normalizes fitted array views on first serialization. Hash the portable
    # representation, so saving/loading a legitimate fit cannot invalidate calibration.
    stream = io.BytesIO()
    joblib.dump(models, stream)
    stream.seek(0)
    return joblib.hash(joblib.load(stream))


def interval_loss(q, returns):
    y = np.expm1(np.asarray(returns)); lo, hi = np.expm1(q[:, 0]), np.expm1(q[:, 2])
    return float(np.mean(hi-lo+10*np.maximum(lo-y, 0)+10*np.maximum(y-hi, 0)))


def calibration_indices(features, outcomes, start, end):
    ready = features[feature_columns(features)].notna().all(axis=1) & outcomes['return'].notna()
    return features.index[ready & (features.index.hour % 6 == 0) & (features.index >= start) &
                          (outcomes.target_end < end-pd.Timedelta(hours=1))]


def train_candidate(features, outcomes, cutoff, horizon, *, deadline=float('inf')):
    cutoff = utc(cutoff)
    long = horizon >= 336
    cal_start = cutoff-pd.Timedelta(days=365 if long else 90)
    selection_start = cal_start-pd.Timedelta(days=365 if long else 90)
    cal_middle = cal_start+(cutoff-cal_start)/2
    val = calibration_indices(features, outcomes, selection_start, cal_start)
    cal = calibration_indices(features, outcomes, cal_start, cal_middle)
    threshold = calibration_indices(features, outcomes, cal_middle, cutoff)
    if min(len(val), len(cal)) < 60:
        raise ValueError('insufficient purged selection/calibration/threshold history')
    windows = (730, 1095) if long else (365, 730)
    results = {}; choices = {}
    for days in windows:
        if time.monotonic() >= deadline: raise TimeoutError('candidate budget exhausted')
        train = calibration_indices(features, outcomes, selection_start-pd.Timedelta(days=days), selection_start)
        if len(train) < 700: continue
        model = fit_models(features, outcomes, train)
        for name, pred in predict_models(model, features.loc[val]).items():
            key = f'{days}:{name}'
            results[key] = {'probability': brier(pred, outcomes.loc[val]),
                            'point': float(np.mean(np.abs(np.expm1(pred['quantiles'][:,1])-np.expm1(outcomes.loc[val,'return'])))),
                            'interval': interval_loss(pred['quantiles'], outcomes.loc[val,'return'])}
    if not results: raise ValueError('insufficient purged candidate training history')
    for head in ('probability', 'point', 'interval'):
        choices[head] = min(results, key=lambda k: (results[k][head], k))
    no_change = float(np.mean(np.abs(np.expm1(outcomes.loc[val,'return']))))
    if no_change <= results[choices['point']]['point']: choices['point'] = 'no_change'
    # Refit before calibration, then freeze exactly these objects for all later steps.
    models = {}; training_ends = []
    for days in sorted({int(k.split(':')[0]) for k in choices.values() if k != 'no_change'}):
        if time.monotonic() >= deadline: raise TimeoutError('candidate budget exhausted')
        train = calibration_indices(features, outcomes, cal_start-pd.Timedelta(days=days), cal_start)
        models[days] = fit_models(features, outcomes, train)
        training_ends.append(outcomes.loc[train,'target_end'].max())
    bundle = {'models': models, 'choices': choices, 'policy': POLICY, 'fit_cutoff': cutoff.isoformat(),
              'training_target_end': max(training_ends).isoformat(), 'validation_target_end': outcomes.loc[val,'target_end'].max().isoformat(),
              'calibration_target_end': outcomes.loc[cal,'target_end'].max().isoformat(),
              'threshold_target_end': outcomes.loc[threshold,'target_end'].max().isoformat() if len(threshold) else None,
              'selection_rows': len(val), 'calibration_rows': len(cal), 'threshold_rows': len(threshold),
              'selection_scores': results, 'no_change_selection_mae': no_change,
              'fitted_identity': model_identity(models), 'temperature': 1., 'path_calibrators': [None, None],
              'quantile_offsets': np.zeros(3), 'alert_thresholds': {'up': 1., 'down': 1.}}
    pred = predict_candidate(bundle, features.loc[cal]); truth = outcomes.loc[cal]
    # Fixed small temperature grid; calibration labels never reach fitting or selection.
    losses = {}
    for temperature in (.5, .75, 1., 1.5, 2.):
        logits = np.log(np.clip(pred['terminal'],1e-8,1))/temperature
        probabilities = np.exp(logits-logits.max(axis=1,keepdims=True)); probabilities /= probabilities.sum(axis=1,keepdims=True)
        losses[temperature] = -np.log(probabilities[np.arange(len(truth)),truth.terminal.astype(int)]).mean()
    bundle['temperature'] = min(losses, key=lambda t: (losses[t],abs(t-1)))
    for j,event in enumerate(('up','down')):
        if truth[event].nunique() > 1:
            x = np.log(np.clip(pred['path'][:,j],1e-6,1-1e-6)/(1-np.clip(pred['path'][:,j],1e-6,1-1e-6)))[:,None]
            bundle['path_calibrators'][j] = LogisticRegression(C=.1,max_iter=200).fit(x,truth[event].astype(int))
    for j,a in enumerate((.1,.5,.9)):
        # Only bounds are adjusted; the independently selected point head is retained.
        if j != 1: bundle['quantile_offsets'][j] = np.quantile(truth['return'].to_numpy()-pred['quantiles'][:,j],a)
    calibrated = predict_candidate(bundle, features.loc[threshold]) if len(threshold) else None
    for j,event in enumerate(('up','down')):
        negatives = calibrated['path'][outcomes.loc[threshold,event].eq(0).to_numpy(),j] if calibrated is not None else []
        bundle['alert_thresholds'][event] = float(np.quantile(negatives,.95,method='higher')) if len(negatives)>=40 else 1.
    if model_identity(models) != bundle['fitted_identity']: raise ValueError('model changed during calibration')
    bundle['model_version'] = digest({'policy':POLICY,'model':bundle['fitted_identity'],'calibration':joblib.hash(bundle),'cutoff':cutoff.isoformat(),'horizon':horizon})
    return bundle


def predict_candidate(bundle, features):
    if model_identity(bundle['models']) != bundle['fitted_identity']:
        raise ValueError('calibrator/model identity mismatch')
    predictions = {days: predict_models(m, features) for days,m in bundle['models'].items()}
    def head(name):
        days, family = bundle['choices'][name].split(':'); return predictions[int(days)][family]
    prob = head('probability')
    logits = np.log(np.clip(prob['terminal'],1e-8,1))/bundle['temperature']
    terminal = np.exp(logits-logits.max(axis=1,keepdims=True)); terminal /= terminal.sum(axis=1,keepdims=True)
    path = prob['path'].copy()
    for j,clf in enumerate(bundle['path_calibrators']):
        if clf is not None:
            p = np.clip(path[:,j],1e-6,1-1e-6)
            path[:,j] = clf.predict_proba(np.log(p/(1-p))[:,None])[:,1]
    q = head('interval')['quantiles'].copy()+bundle['quantile_offsets']
    q[:,1] = 0. if bundle['choices']['point']=='no_change' else head('point')['quantiles'][:,1]
    q[:,0] = np.minimum(q[:,0],q[:,1]); q[:,2] = np.maximum(q[:,2],q[:,1])
    return {'terminal':terminal,'path':path,'quantiles':q}


def prediction_rows(pred, features, outcomes, indices, thresholds):
    rows = []
    for j,slot in enumerate(indices):
        y = outcomes.loc[slot];q=pred['quantiles'][j];p=pred['terminal'][j]
        rows.append({'slot':slot.isoformat(),'target_end':y.target_end.isoformat(),'return':float(y['return']),
                     'reference_price':float(features.loc[slot,'reference_price']), 'terminal':int(y.terminal),
                     'up':int(y.up),'down':int(y.down),'hit_up':float(pred['path'][j,0]),'hit_down':float(pred['path'][j,1]),
                     'p_down':float(p[0]),'p_flat':float(p[1]),'p_up':float(p[2]),
                     'q10':float(q[0]),'q50':float(q[1]),'q90':float(q[2]),
                     'threshold_up':thresholds['up'],'threshold_down':thresholds['down']})
    return rows


def run_review(root, *, budget_seconds=1500, bars=None):
    from .engine import atomic_json
    root=Path(root); directory=root/'shadow_models'; directory.mkdir(parents=True,exist_ok=True)
    started=time.monotonic(); deadline=started+budget_seconds
    bars=read_bars(root) if bars is None else bars
    features=build_features(bars); as_of=bars.close_time.max(); reports={}; active={}
    code=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    # Content-sensitive cache: revised bars cannot reuse a differently trained candidate.
    source=hashlib.sha256(pd.util.hash_pandas_object(bars.sort_values(['product','close_time']),index=False).values.tobytes()).hexdigest()
    for h in (6,24,72,168,336,720):
        if time.monotonic()>=deadline: raise TimeoutError('candidate review budget exhausted; prior complete review retained')
        outcomes=labels(bars,features,h)
        cutoff=(as_of-pd.Timedelta(hours=h+1)-pd.Timedelta(days=90)).floor('D')
        key=digest({'policy':POLICY,'code':code,'legacy':training_hash(),'source':source,'cutoff':str(cutoff),'h':h})
        report_path=directory/f'review_{h}_{key}.json'
        if report_path.exists(): reports[str(h)]=json.loads(report_path.read_text())
        else:
            candidate=train_candidate(features,outcomes,cutoff,h,deadline=deadline)
            incumbent=train_bundle(features,outcomes,cutoff,h)
            indices=calibration_indices(features,outcomes,cutoff,as_of)[::4]
            candidate_pred=predict_candidate(candidate,features.loc[indices])
            existing=predict_models(incumbent['model'],features.loc[indices])
            selected=prediction_rows(candidate_pred,features,outcomes,indices,candidate['alert_thresholds'])
            reference=prediction_rows(existing[incumbent['choice']],features,outcomes,indices,incumbent['alert_thresholds'])
            baseline=prediction_rows(existing['climatology'],features,outcomes,indices,incumbent['alert_thresholds_by_model']['climatology'])
            reports[str(h)]={'horizon_hours':h,'cutoff':cutoff.isoformat(),'test_last_target':max((r['target_end'] for r in selected),default=None),
                'heads':candidate['choices'],'candidate':metrics(selected),'incumbent':metrics(reference),'baseline':metrics(baseline),
                'paired_vs_incumbent':paired_block_interval(selected,reference,h),'paired_vs_baseline':paired_block_interval(selected,baseline,h),
                'promotion':'held_for_prospective_evidence','audit':{k:candidate[k] for k in ('fitted_identity','training_target_end','validation_target_end','calibration_target_end','threshold_target_end','selection_rows','calibration_rows','threshold_rows')},
                'rows':selected,'incumbent_rows':reference,'baseline_rows':baseline}
            atomic_json(report_path,reports[str(h)])
        # Short windows refresh weekly; longer windows only on the first available monthly run.
        live_cutoff=as_of.floor('D') if h<=168 else as_of.floor('D').replace(day=1)
        live_source=hashlib.sha256(pd.util.hash_pandas_object(bars[bars.close_time < live_cutoff].sort_values(['product','close_time']),index=False).values.tobytes()).hexdigest()
        live_key=digest({'code':code,'training':training_hash(),'source':live_source,'cutoff':str(live_cutoff),'h':h})
        path=directory/f'h{h}_{live_key}.joblib'
        if not path.exists():
            live=train_candidate(features,outcomes,live_cutoff,h,deadline=deadline)
            temp=path.with_suffix('.tmp'); joblib.dump(live,temp);temp.replace(path)
        active[str(h)]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    if time.monotonic()>=deadline: raise TimeoutError('candidate review budget exhausted; prior complete review retained')
    public={h:{k:v for k,v in r.items() if k not in ('rows','incumbent_rows','baseline_rows')} for h,r in reports.items()}
    result={'schema_version':1,'policy':POLICY,'generated_at':utc().isoformat(),'data_as_of':as_of.isoformat(),
            'status':'shadow_only','horizons':public,'runtime_seconds':time.monotonic()-started,
            'claims':'Separate price, probability and interval heads are evaluated on a later 90-day test. Historical data is development evidence; no prospective superiority or automatic model promotion is claimed.',
            'schedule':'Weekly short-window review; monthly all-window refit; hourly saved-model inference only.',
            'promotion_gate':'Retain incumbent until same-origin prospective comparisons, at least 20 non-overlapping settled windows, and calendar-block uncertainty support improvement without material price/probability/coverage regression.'}
    atomic_json(root/'optimization.json',result);atomic_json(root/'shadow_active.json',active)
    return result


def infer_shadow(root, features, slot, bars, *, now=None):
    """Separate immutable ledger; candidates never substitute for a public incumbent."""
    from .engine import forecast_record, baseline_record
    from .ledger import issue, settle, history
    from .segments import origin_market_states
    root=Path(root); shadow=root/'shadow'; current=utc(now)
    path=root/'shadow_active.json'
    records=[]; errors=[]
    if not path.exists(): return records,history(shadow),['shadow checkpoint unavailable']
    settle(shadow,bars,now=current)
    manifest=json.loads(path.read_text())
    for horizon,entry in manifest.items():
        try:
            h=int(horizon); name=entry['file']
            if Path(name).name!=name: raise ValueError('invalid shadow checkpoint path')
            checkpoint=root/'shadow_models'/name
            if hashlib.sha256(checkpoint.read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('shadow checkpoint integrity mismatch')
            bundle=joblib.load(checkpoint)
            max_age=pd.Timedelta(days=40 if h>=336 else 10)
            if not utc(bundle['fit_cutoff'])<=slot<=utc(bundle['fit_cutoff'])+max_age:raise ValueError('shadow checkpoint expired')
            pred=predict_candidate(bundle,features.loc[[slot]])
            # Reuse the same window construction and immutable issuance validation.
            days,family=bundle['choices']['probability'].split(':')
            wrapper={'model':bundle['models'][int(days)],'choice':family,'alert_thresholds':bundle['alert_thresholds'],
                     'model_version':bundle['model_version'],'training_target_end':bundle['training_target_end'],
                     'validation_target_end':bundle['validation_target_end'],'source_snapshot':entry['sha256'],
                     'runtime_hash':POLICY,'alert_thresholds_by_model':{'climatology':{'up':1.,'down':1.}}}
            record=forecast_record(features,slot,h,wrapper)
            record.update({'terminal_down_flat_up':pred['terminal'][0].tolist(),'hit_up':float(pred['path'][0,0]),
                           'hit_down':float(pred['path'][0,1]),'price_quantiles':(record['reference_price']*np.exp(pred['quantiles'][0])).tolist(),
                           'selected_model':POLICY,'head_choices':bundle['choices'],'calibration_identity':bundle['fitted_identity'],
                           'origin_market':origin_market_states(features.loc[[slot]]).loc[slot].to_dict(),
                           'baseline':baseline_record(features,slot,wrapper)})
            records.append(issue(shadow,record,now=utc() if now is None else current))
        except (ValueError,OSError,KeyError) as exc: errors.append({'horizon':int(horizon),'reason':str(exc)})
    return records,history(shadow),errors
