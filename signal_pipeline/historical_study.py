"""Resumable multi-year monthly evaluation of the registered shadow policy."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .data import build_features, feature_columns, read_bars, utc
from .diagnostics import export_archive
from .engine import atomic_json, check_replay_cutoff, obtain_bundle, source_hash
from .evaluate import metrics
from .models import labels, predict_models
from .optimization import POLICY, model_identity, prediction_rows, predict_candidate, train_candidate
from .protocol import HORIZONS, PROTOCOL_HASH, digest, training_hash
from .segments import origin_market_states

STUDY_VERSION = 'monthly_separate_heads_history_v1'
DEFAULT_START = '2020-01-01'
FAMILIES = ('candidate', 'incumbent', 'frequency', 'no_change')


def audit_candidate(bundle, cutoff, horizon):
    cutoff = utc(cutoff)
    span = pd.Timedelta(days=365 if horizon >= 336 else 90)
    boundaries = {'training_target_end': cutoff-span, 'validation_target_end': cutoff-span,
                  'calibration_target_end': cutoff-span/2, 'threshold_target_end': cutoff}
    if utc(bundle['fit_cutoff']) != cutoff:
        raise ValueError('candidate cutoff mismatch')
    for key, boundary in boundaries.items():
        value = bundle.get(key)
        if value is None and key == 'threshold_target_end' and bundle['threshold_rows'] == 0:
            continue
        if value is None or pd.isna(utc(value)) or utc(value) >= boundary-pd.Timedelta(hours=1):
            raise ValueError('candidate target crosses partition: '+key)
    if model_identity(bundle['models']) != bundle['fitted_identity']:
        raise ValueError('candidate calibrated model identity mismatch')


def eligible_origins(features, outcomes, cutoff, as_of):
    ready = features[feature_columns(features)].notna().all(axis=1) & outcomes['return'].notna()
    return features.index[ready & (features.index.hour == 0) & (features.index >= cutoff) &
                          (features.index < cutoff+pd.offsets.MonthBegin(1)) &
                          (outcomes.target_end <= as_of)]


def nonoverlap(rows):
    selected = []; end = None
    for row in sorted(rows, key=lambda r: r['slot']):
        if end is None or utc(row['slot']) >= end:
            selected.append(row); end = utc(row['target_end'])
    return selected


def losses(frame, kind):
    if kind == 'probability':
        return ((frame.hit_up-frame.up)**2+(frame.hit_down-frame.down)**2).to_numpy()/2
    y = np.expm1(frame['return'].to_numpy())
    if kind == 'point':
        return np.abs(np.expm1(frame.q50.to_numpy())-y)
    lo, hi = np.expm1(frame.q10.to_numpy()), np.expm1(frame.q90.to_numpy())
    return hi-lo+10*np.maximum(lo-y, 0)+10*np.maximum(y-hi, 0)


def paired_losses(candidate, reference, horizon, samples=500):
    a, b = pd.DataFrame(candidate), pd.DataFrame(reference)
    if a.empty or len(a) != len(b) or a.slot.tolist() != b.slot.tolist():
        raise ValueError('paired study requires identical ordered origins')
    for key in ('target_end', 'return', 'terminal', 'up', 'down'):
        if not a[key].equals(b[key]):
            raise ValueError('paired study truth mismatch')
    width = max(7, int(np.ceil((horizon+1)/24)))
    groups = (pd.to_datetime(a.slot, utc=True).astype('int64')//(86400*10**9*width)).to_numpy()
    ids = np.unique(groups); result = {}
    for kind in ('point', 'probability', 'interval'):
        delta = losses(a, kind)-losses(b, kind)
        item = {'difference': float(delta.mean()), 'lower95': None, 'upper95': None,
                'blocks': len(ids), 'block_days': width,
                'interpretation': 'negative favors candidate; historical development evidence'}
        if len(ids) >= 10:
            sums = np.array([delta[groups == g].sum() for g in ids])
            counts = np.array([(groups == g).sum() for g in ids])
            indices = np.random.default_rng(1729).integers(len(ids), size=(samples, len(ids)))
            estimates = sums[indices].sum(axis=1)/counts[indices].sum(axis=1)
            item.update(lower95=float(np.quantile(estimates, .025)), upper95=float(np.quantile(estimates, .975)))
        result[kind] = item
    return result


def summarize(rows, horizon):
    groups = {name: sorted([r for r in rows if r['model'] == name], key=lambda r: r['slot']) for name in FAMILIES}
    candidate = groups['candidate']
    if not candidate:
        return {'horizon_hours': horizon, 'status': 'insufficient_history', 'origins': 0}
    for name, values in groups.items():
        if len({r['slot'] for r in values}) != len(values):
            raise ValueError('duplicate historical origin: '+name)
        if [r['slot'] for r in values] != [r['slot'] for r in candidate]:
            raise ValueError('unmatched historical origins')
    independent = nonoverlap(candidate); slots = {r['slot'] for r in independent}
    comparisons = {name: paired_losses(candidate, groups[name], horizon) for name in FAMILIES[1:]}
    scores = {name: metrics(values) for name, values in groups.items()}
    # These are predeclared research screens, not automatic production promotion.
    def better(name, loss):
        upper = comparisons[name][loss]['upper95']
        return upper is not None and upper < 0
    head_evidence = {
        'point': better('incumbent', 'point') and better('no_change', 'point'),
        'probability': better('incumbent', 'probability') and better('frequency', 'probability'),
        'interval': better('incumbent', 'interval') and better('frequency', 'interval'),
    }
    years = sorted({r['slot'][:4] for r in candidate})
    regimes = {'trend_'+state: lambda r, s=state: r['trend_state'] == s for state in ('up', 'down', 'range')}
    regimes.update({'vol_'+state: lambda r, s=state: r['volatility_state'] == s for state in ('high', 'normal')})
    return {'horizon_hours': horizon, 'status': 'historical_development', 'origins': len(candidate),
            'first_origin': candidate[0]['slot'], 'last_origin': candidate[-1]['slot'],
            'nonoverlap': len(independent),
            'calendar_coverage': len(candidate)/((utc(candidate[-1]['slot'])-utc(candidate[0]['slot'])).days+1),
            **scores, 'paired_losses': comparisons,
            'nonoverlapping_metrics': {name: metrics([r for r in values if r['slot'] in slots]) for name, values in groups.items()},
            'yearly': {year: {name: metrics([r for r in values if r['slot'].startswith(year)]) for name, values in groups.items()} for year in years},
            'origin_states': {state: {name: metrics([r for r in values if predicate(r)]) for name, values in groups.items()} for state, predicate in regimes.items()},
            'head_evidence': head_evidence,
            'decision': 'review_candidate_outputs' if any(head_evidence.values()) else 'retain_incumbent_no_broad_historical_edge',
            'promotion': 'research evidence only; original active models and issued records unchanged'}


def evaluate_month(root, bars, features, outcomes, states, cutoff, horizon, directory, deadline):
    """Training and result caches have separate, past-bounded source dependencies."""
    optimizer_code = hashlib.sha256(Path(__file__).with_name('optimization.py').read_bytes()).hexdigest()
    model_key = digest({'policy': POLICY, 'optimizer': optimizer_code, 'training': training_hash(),
                        'protocol': PROTOCOL_HASH, 'cutoff': cutoff.isoformat(), 'horizon': horizon,
                        'source': source_hash(bars, cutoff)})
    model_path = directory/f'h{horizon}_{cutoff:%Y-%m}_{model_key}.joblib'
    as_of = bars.groupby('product').close_time.max().min()
    indices = eligible_origins(features, outcomes, cutoff, as_of)
    if not len(indices):
        return {'cutoff': cutoff.isoformat(), 'horizon_hours': horizon, 'status': 'excluded', 'reason': 'no complete matured target origins', 'rows': []}
    outcome_boundary = outcomes.loc[indices, 'target_end'].max()+pd.Timedelta(hours=1)
    evaluation_key = digest({'version': STUDY_VERSION, 'code': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'model': model_key, 'origins': [t.isoformat() for t in indices],
                             'source_through_last_target': source_hash(bars, outcome_boundary)})
    report_path = directory/f'h{horizon}_{cutoff:%Y-%m}_{evaluation_key}.json'
    if report_path.exists():
        result = json.loads(report_path.read_text())
        if result.get('evaluation_key') != evaluation_key:
            raise ValueError('historical cache identity mismatch')
        return result
    try:
        if model_path.exists():
            candidate = joblib.load(model_path)
        else:
            candidate = train_candidate(features, outcomes, cutoff, horizon, deadline=deadline)
            audit_candidate(candidate, cutoff, horizon)
            temporary = model_path.with_suffix('.tmp'); joblib.dump(candidate, temporary, compress=3); temporary.replace(model_path)
        audit_candidate(candidate, cutoff, horizon)
        incumbent, _ = obtain_bundle(root, bars, features, outcomes, cutoff, horizon, allow_fit=True)
        check_replay_cutoff(incumbent, cutoff)
    except ValueError as exc:
        if 'insufficient purged' not in str(exc):
            raise
        result = {'evaluation_key': evaluation_key, 'cutoff': cutoff.isoformat(), 'horizon_hours': horizon,
                  'status': 'excluded', 'reason': str(exc), 'rows': []}
        atomic_json(report_path, result)
        return result
    pred = predict_candidate(candidate, features.loc[indices])
    original = predict_models(incumbent['model'], features.loc[indices])
    predictions = {'candidate': (pred, candidate['alert_thresholds']),
                   'incumbent': (original[incumbent['choice']], incumbent['alert_thresholds']),
                   'frequency': (original['climatology'], incumbent['alert_thresholds_by_model']['climatology'])}
    rows = []
    for name, (values, thresholds) in predictions.items():
        values = prediction_rows(values, features, outcomes, indices, thresholds)
        for row in values:
            row.update(model=name, horizon_hours=horizon, fit_cutoff=cutoff.isoformat(),
                       model_version=candidate['model_version'] if name == 'candidate' else incumbent['model_version'],
                       selected_model=POLICY if name == 'candidate' else incumbent['choice'] if name == 'incumbent' else 'climatology',
                       actual_price=float(row['reference_price']*np.exp(row['return'])),
                       **states.loc[utc(row['slot'])].to_dict())
        rows.extend(values)
    rows.extend([{**r, 'model': 'no_change', 'selected_model': 'no_change_point_with_frequency_risk', 'q50': 0.,
                  'q10': min(r['q10'], 0.), 'q90': max(r['q90'], 0.)} for r in rows if r['model'] == 'frequency'])
    result = {'evaluation_key': evaluation_key, 'cutoff': cutoff.isoformat(), 'horizon_hours': horizon,
              'status': 'evaluated', 'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
              'heads': candidate['choices'], 'audit': {k: candidate[k] for k in (
                  'fitted_identity', 'training_target_end', 'validation_target_end', 'calibration_target_end', 'threshold_target_end',
                  'selection_rows', 'calibration_rows', 'threshold_rows')}, 'rows': rows}
    atomic_json(report_path, result)
    return result


def run_study(root, *, start=DEFAULT_START, end=None, horizons=tuple(HORIZONS), budget_seconds=1500, bars=None):
    root = Path(root); directory = root/'historical_study'; directory.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); deadline = started+budget_seconds
    bars = read_bars(root) if bars is None else bars
    features = build_features(bars)
    if features.empty:
        raise ValueError('both historical ETH and BTC inputs are required')
    as_of = bars.groupby('product').close_time.max().min()
    first = utc(start).floor('D').replace(day=1)
    last = min(utc(end), as_of) if end is not None else as_of
    months = pd.date_range(first, last, freq='MS')
    if not len(months) or not horizons or any(h not in HORIZONS for h in horizons):
        raise ValueError('empty or invalid historical study scope')
    states = origin_market_states(features); reports = {}; archives = {}; audits = []; exclusions = []
    progress = {'version': STUDY_VERSION, 'status': 'running', 'started_at': utc().isoformat(),
                'data_as_of': as_of.isoformat(), 'requested_start': first.isoformat(),
                'requested_months': len(months)*len(horizons), 'completed_months': 0}
    atomic_json(root/'historical_progress.json', progress)
    try:
        for h in horizons:
            outcomes = labels(bars, features, h); rows = []
            for cutoff in months:
                if time.monotonic() >= deadline:
                    raise TimeoutError('historical study budget exhausted; completed months retained for resume')
                result = evaluate_month(root, bars, features, outcomes, states, cutoff, h, directory, deadline)
                rows.extend(result['rows'])
                metadata = {k: v for k, v in result.items() if k != 'rows'}
                (audits if result['status'] == 'evaluated' else exclusions).append(metadata)
                progress['completed_months'] += 1
                atomic_json(root/'historical_progress.json', progress)
                print(f"historical h={h} month={cutoff:%Y-%m} status={result['status']} origins={len(result['rows'])//4}", flush=True)
            reports[str(h)] = summarize(rows, h)
            pd.DataFrame(rows).to_csv(directory/f'h{h}_rows.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
            archives[str(h)] = [r for r in rows if r['model'] == 'candidate']
        result = {'schema_version': 1, 'version': STUDY_VERSION, 'candidate_policy': POLICY, 'status': 'complete',
                  'generated_at': utc().isoformat(), 'data_as_of': as_of.isoformat(),
                  'requested_start': first.isoformat(), 'requested_last_month': months[-1].isoformat(),
                  'protocol_hash': PROTOCOL_HASH, 'training_hash': training_hash(),
                  'source_snapshot': source_hash(bars, as_of+pd.Timedelta(hours=1)),
                  'horizons': reports, 'monthly_audits': audits, 'excluded_months': exclusions,
                  'runtime_seconds': time.monotonic()-started,
                  'claims': 'Monthly chronological reconstruction across multiple years. Each fit uses earlier matured labels only. Original historical receipt vintages are unavailable; this is development evidence, not actual past publication or a newly untouched holdout.',
                  'archive': export_archive(root, archives, kind='candidate_history', as_of=as_of.isoformat())}
        atomic_json(root/'historical_study.json', result)
        progress.update(status='complete', completed_at=utc().isoformat())
        atomic_json(root/'historical_progress.json', progress)
        return result
    except Exception as exc:
        progress.update(status='incomplete', error=str(exc)[:300], updated_at=utc().isoformat())
        atomic_json(root/'historical_progress.json', progress)
        raise
