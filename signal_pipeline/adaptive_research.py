"""Fixed delayed-feedback candidates evaluated on previously reconstructed forecasts."""
from pathlib import Path
import gzip
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_features, read_bars, utc
from .engine import atomic_json, source_hash
from .evaluate import metrics
from .historical_study import nonoverlap, paired_losses, losses
from .protocol import HORIZONS, PROTOCOL_HASH, training_hash, digest

VERSION = 'delayed_output_adaptation_v1'
EXPERTS = ('incumbent', 'frequency', 'candidate', 'no_change')
METHODS = ('conservative', 'adaptive_mix', 'adaptive_risk')
P = ['p_down', 'p_flat', 'p_up', 'hit_up', 'hit_down']
Q = ['q10', 'q50', 'q90']
POLICY = {'version': VERSION, 'short_window_days': 365, 'long_window_days': 1095,
          'short_half_life_days': 90, 'long_half_life_days': 365,
          'short_min_feedback': 90, 'long_min_feedback': 365, 'sensitivity': 8.,
          'point_prior': [.25, .125, .125, .5], 'probability_prior': [.5, .4, .1],
          'interval_prior': [1/3]*3, 'calibration_C': .1, 'calibration_mix': .5,
          'embargo_hours': 1, 'alert_negative_minimum': 40}


def weighted_quantile(values, weights, quantiles):
    order = np.argsort(values, kind='stable'); values = np.asarray(values)[order]
    weights = np.asarray(weights)[order]
    if not len(values) or not np.isfinite(values).all() or (weights <= 0).any():
        raise ValueError('invalid residual quantile input')
    return np.interp(quantiles, (np.cumsum(weights)-.5*weights)/weights.sum(), values)


def mix_weights(score, prior, scale=1.):
    logits = np.log(prior)-POLICY['sensitivity']*score/max(float(scale), 1e-8)
    weights = np.exp(logits-logits.max()); return weights/weights.sum()


def probability_features(p):
    p = np.clip(p, 1e-6, 1-1e-6)
    return np.column_stack([np.log(p[:, :3]), np.log(p[:, 3:]/(1-p[:, 3:]))])


def fit_calibration(prediction, truth, weights):
    x = probability_features(prediction); result = []
    for columns, y, classes in [(slice(0, 3), truth[:, 0].astype(int), 3),
                                (slice(3, 4), truth[:, 1].astype(int), 2),
                                (slice(4, 5), truth[:, 2].astype(int), 2)]:
        counts = np.bincount(y, weights=weights, minlength=classes)+1
        if len(np.unique(y)) < 2:
            result.append((columns, counts/counts.sum())); continue
        model = make_pipeline(StandardScaler(), LogisticRegression(C=POLICY['calibration_C'],
                              max_iter=300, random_state=1729))
        model.fit(x[:, columns], y, logisticregression__sample_weight=weights*len(weights)/weights.sum())
        result.append((columns, model))
    return result


def calibrated_probability(calibration, p):
    x = probability_features(p[None, :]); outputs = []
    for columns, model in calibration:
        classes = 3 if columns.start == 0 else 2
        if isinstance(model, np.ndarray): value = model
        else:
            value = np.full(classes, 1e-8)
            value[model[-1].classes_] = model.predict_proba(x[:, columns])[0]
            value /= value.sum()
        outputs.extend(value if classes == 3 else [value[1]])
    return np.asarray(outputs)


def paired_inputs(frame):
    frames = {name: frame[frame.model.eq(name)].sort_values('slot').reset_index(drop=True) for name in EXPERTS}
    base = frames['incumbent']; truth = ['slot', 'target_end', 'return', 'terminal', 'up', 'down', 'reference_price']
    if base.empty: raise ValueError('no paired historical rows')
    for name, f in frames.items():
        if f.slot.duplicated().any() or not f[truth].equals(base[truth]):
            raise ValueError('adaptive input truth/origin mismatch: '+name)
        if not np.isfinite(f[P+Q+['return', 'reference_price']].to_numpy(float)).all():
            raise ValueError('non-finite adaptive input')
    return frames


def replay(frame, scale, horizon, *, deadline=float('inf')):
    frames = paired_inputs(frame); base = frames['incumbent']; n = len(base)
    dates = pd.to_datetime(base.slot, utc=True).astype('int64').to_numpy()
    targets = pd.to_datetime(base.target_end, utc=True).astype('int64').to_numpy()
    hour, day = 3600*10**9, 86400*10**9
    if not np.all(targets-dates == (horizon+1)*hour): raise ValueError('incorrect adaptive target span')
    scale = np.asarray(scale, float)
    if scale.shape != (n,) or not np.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('invalid origin-time volatility scale')
    long = horizon >= 336
    window = (POLICY['long_window_days'] if long else POLICY['short_window_days'])*day
    half = (POLICY['long_half_life_days'] if long else POLICY['short_half_life_days'])*day
    minimum = POLICY['long_min_feedback'] if long else POLICY['short_min_feedback']
    probability = np.stack([frames[k][P].to_numpy() for k in EXPERTS[:3]], axis=1)
    quantiles = np.stack([frames[k][Q].to_numpy() for k in EXPERTS], axis=1)
    point_loss = np.stack([losses(frames[k], 'point') for k in EXPERTS], axis=1)
    probability_loss = np.stack([losses(frames[k], 'probability') for k in EXPERTS[:3]], axis=1)
    interval_loss = np.stack([losses(frames[k], 'interval') for k in EXPERTS[:3]], axis=1)
    truth = base[['terminal', 'up', 'down']].to_numpy(); returns = base['return'].to_numpy()
    pred_p = np.zeros((2, n, 5)); pred_q = np.zeros((2, n, 3)); thresholds = np.ones((2, n, 2))
    records = []; audit = []; calibration = None; calibration_cutoff = None; cal_target = None; last_month = None
    for i in range(n):
        if time.monotonic() >= deadline: raise TimeoutError('adaptive review budget exhausted')
        end = np.searchsorted(targets, dates[i]-hour, side='left')
        start = np.searchsorted(dates, dates[i]-window, side='left')
        past = np.arange(start, end); ready = len(past) >= minimum
        if end > i: raise ValueError('feedback includes current prediction')
        weights = np.exp2(-(dates[i]-targets[past])/half)
        if ready:
            averages = lambda values: np.average(values[past], axis=0, weights=weights)
            pl, pr, il = averages(point_loss), averages(probability_loss), averages(interval_loss)
            wp = mix_weights(pl, POLICY['point_prior'], pl[1])
            wb = mix_weights(pr, POLICY['probability_prior'])
            wi = mix_weights(il, POLICY['interval_prior'], il[1])
        else:
            wp, wb, wi = map(np.asarray, (POLICY['point_prior'], POLICY['probability_prior'], POLICY['interval_prior']))
        pred_q[0, i] = np.log1p(np.average(np.expm1(quantiles[i, :3]), axis=0, weights=wi))
        pred_q[0, i, 1] = np.log1p(np.dot(wp, np.expm1(quantiles[i, :, 1])))
        pred_q[0, i, 0] = min(pred_q[0, i, 0], pred_q[0, i, 1]); pred_q[0, i, 2] = max(pred_q[0, i, 2], pred_q[0, i, 1])
        pred_p[0, i] = np.average(probability[i], axis=0, weights=wb)
        pred_q[1, i], pred_p[1, i] = pred_q[0, i], pred_p[0, i]
        month = base.slot.iloc[i][:7]
        if month != last_month:
            last_month = month; cutoff = utc(month+'-01').value
            ci = np.arange(np.searchsorted(dates, cutoff-window, side='left'),
                           np.searchsorted(targets, cutoff-hour, side='left'))
            calibration = None; calibration_cutoff = None; cal_target = None
            if len(ci) >= minimum:
                calibration = fit_calibration(pred_p[0, ci], truth[ci], np.exp2(-(cutoff-targets[ci])/half))
                calibration_cutoff = utc(month+'-01').isoformat(); cal_target = base.target_end.iloc[ci[-1]]
        if ready:
            residual = (returns[past]-pred_q[0, past, 1])/scale[past]
            bounds = pred_q[0, i, 1]+scale[i]*weighted_quantile(residual, weights, [.1, .9])
            pred_q[1, i, 0], pred_q[1, i, 2] = min(bounds[0], pred_q[0, i, 1]), max(bounds[1], pred_q[0, i, 1])
            if calibration is not None:
                pred_p[1, i] = .5*pred_p[0, i]+.5*calibrated_probability(calibration, pred_p[0, i])
            for method in range(2):
                for event in range(2):
                    negative = past[truth[past, event+1] == 0]
                    if len(negative) >= POLICY['alert_negative_minimum']:
                        thresholds[method, i, event] = np.quantile(pred_p[method, negative, event+3], .95, method='higher')
        meta = {'feedback_rows': len(past), 'feedback_target_end': base.target_end.iloc[past[-1]] if len(past) else None,
                'calibration_cutoff': calibration_cutoff, 'calibration_target_end': cal_target,
                'origin_scale': float(scale[i]), 'scored': ready}
        for j, method in enumerate(METHODS[1:]):
            records.append({**base.iloc[i].to_dict(), **meta, 'model': method, 'selected_model': method,
                            'model_version': digest({'policy': POLICY, 'method': method, 'slot': base.slot.iloc[i],
                                'probabilities': pred_p[j, i].tolist(), 'quantiles': pred_q[j, i].tolist(), **meta}),
                            'calibration_cutoff': calibration_cutoff if j else None,
                            'calibration_target_end': cal_target if j else None,
                            **dict(zip(P, map(float, pred_p[j, i]))), **dict(zip(Q, map(float, pred_q[j, i]))),
                            'threshold_up': float(thresholds[j, i, 0]), 'threshold_down': float(thresholds[j, i, 1])})
        audit.append({'slot': base.slot.iloc[i], **meta})
    return records, audit


def summarize(frame, horizon):
    groups = {k: frame[frame.model.eq(k)].sort_values('slot').to_dict('records') for k in (*EXPERTS, *METHODS)}
    origins = groups['incumbent']; independent = nonoverlap(origins); slots = {r['slot'] for r in independent}
    comparisons = {m: {b: paired_losses(groups[m], groups[b], horizon) for b in ('incumbent', 'candidate', 'frequency', 'no_change')} for m in METHODS}
    evidence = {}
    for method in METHODS:
        evidence[method] = {}
        for head, baseline in [('point', 'no_change'), ('probability', 'frequency'), ('interval', 'frequency')]:
            values = [comparisons[method][b][head]['upper95'] for b in ('incumbent', baseline)]
            evidence[method][head] = all(x is not None and x < 0 for x in values)
    years = sorted({r['slot'][:4] for r in origins})
    return {'horizon_hours': horizon, 'origins': len(origins), 'nonoverlap': len(independent),
            'first_origin': origins[0]['slot'], 'last_origin': origins[-1]['slot'],
            'metrics': {k: metrics(v) for k, v in groups.items()}, 'paired_losses': comparisons, 'head_evidence': evidence,
            'nonoverlapping_metrics': {k: metrics([r for r in v if r['slot'] in slots]) for k, v in groups.items()},
            'yearly': {y: {k: metrics([r for r in v if r['slot'].startswith(y)]) for k, v in groups.items()} for y in years},
            'origin_states': {s: {k: metrics([r for r in v if r[column] == value]) for k, v in groups.items()}
                              for s, column, value in [('trend_up', 'trend_state', 'up'), ('trend_down', 'trend_state', 'down'),
                                  ('trend_range', 'trend_state', 'range'), ('vol_high', 'volatility_state', 'high'), ('vol_normal', 'volatility_state', 'normal')]}}


def run_review(root, *, budget_seconds=300):
    root = Path(root); started = time.monotonic(); deadline = started+budget_seconds
    original = json.loads((root/'historical_study.json').read_text())
    if original.get('status') != 'complete' or set(original['horizons']) != set(map(str, HORIZONS)):
        raise ValueError('complete six-horizon source study required')
    if original['protocol_hash'] != PROTOCOL_HASH or original['training_hash'] != training_hash():
        raise ValueError('incompatible original study')
    bars = read_bars(root); as_of = utc(original['data_as_of'])
    if source_hash(bars, as_of+pd.Timedelta(hours=1)) != original['source_snapshot']:
        raise ValueError('adaptive source snapshot mismatch')
    features = build_features(bars); directory = root/'adaptive_study'; directory.mkdir(exist_ok=True)
    reports = {}; files = {}; audits = {}
    for h in HORIZONS:
        path = root/'historical_study'/f'h{h}_rows.csv.gz'; frame = pd.read_csv(path)
        groups = paired_inputs(frame); base = groups['incumbent']; scale = features.loc[pd.to_datetime(base.slot, utc=True), 'sigma'].to_numpy()*np.sqrt(h)
        if len(base) != original['horizons'][str(h)]['origins']: raise ValueError('source row count mismatch')
        for name, values in groups.items():
            measured = metrics(values.to_dict('records')); expected = original['horizons'][str(h)][name]
            for key in ('return_mae_pp', 'event_brier', 'coverage80'):
                if not np.isclose(measured[key], expected[key], rtol=1e-10, atol=1e-10):
                    raise ValueError('source metric mismatch: '+name+'/'+key)
        rows, temporal = replay(frame, scale, h, deadline=deadline)
        scored = {a['slot'] for a in temporal if a['scored']}
        if not scored: raise ValueError('no adaptive scored origins')
        rows = [r for r in rows if r['slot'] in scored]
        references = frame[frame.slot.isin(scored)].to_dict('records')
        conservative = [{**r, 'model': 'conservative'} for r in references if r['model'] == 'no_change']
        evaluated = pd.DataFrame([*references, *conservative, *rows])
        encoded = gzip.compress(evaluated.to_csv(index=False).encode(), mtime=0)
        rows_hash = hashlib.sha256(encoded).hexdigest(); output = directory/f'h{h}_{rows_hash}.csv.gz'
        if not output.exists():
            temporary = output.with_suffix('.tmp'); temporary.write_bytes(encoded); temporary.replace(output)
        reports[str(h)] = summarize(evaluated, h)
        audits[str(h)] = {'warmup_origins': len(base)-len(scored), 'generated_origins': len(base),
                         'scored_origins': len(scored), 'max_feedback_target_end': temporal[-1]['feedback_target_end']}
        files[str(h)] = {'source_rows_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                         'scored_rows_path': str(output.relative_to(root)), 'scored_rows_sha256': rows_hash}
        print(f'adaptive {h}h: {len(scored)} scored; {reports[str(h)]["nonoverlap"]} nonoverlapping', flush=True)
    report = {'status': 'complete', 'policy': POLICY, 'policy_hash': digest(POLICY),
              'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'data_as_of': original['data_as_of'],
              'original_study_sha256': hashlib.sha256((root/'historical_study.json').read_bytes()).hexdigest(),
              'generated_at': utc().isoformat(), 'runtime_seconds': time.monotonic()-started,
              'horizons': reports, 'temporal_audits': audits, 'files': files,
              'claims': 'Fixed delayed-feedback historical development retest. Same inspected history, not an untouched holdout. Empirical interval adaptation has no per-forecast coverage guarantee. No production adoption or retroactive issuance.'}
    atomic_json(root/'adaptive_study.json', report)
    return report
