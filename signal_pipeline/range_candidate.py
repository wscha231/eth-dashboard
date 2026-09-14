"""Range-only prospective candidate; never changes the incumbent or its probabilities."""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .data import utc
from .ledger import history, issue, settle
from .protocol import PROTOCOL_HASH, digest, training_hash

VERSION = 'short_range_mixture_v1'
HORIZONS = (6, 24, 72)
EXPERTS = ('incumbent', 'frequency', 'candidate')
POLICY = {'version': VERSION, 'window_days': 365, 'half_life_days': 90,
          'minimum_daily_origins': 90, 'sensitivity': 8., 'embargo_hours': 1,
          'feedback_stride': 'one first-issued paired forecast per UTC day',
          'point_and_probability': 'unchanged incumbent', 'promotion': 'shadow_only'}
BOOTSTRAP = Path(__file__).resolve().parents[1]/'research/range_shadow_20260914/seed.json'


def interval_scores(quantiles, truth):
    q = np.asarray(quantiles, float)
    if q.shape[-1] != 3 or not np.isfinite(q).all() or not np.isfinite(truth):
        raise ValueError('non-finite range feedback')
    if (np.diff(q, axis=-1) < 0).any():
        raise ValueError('unordered expert range')
    lo, hi = q[..., 0], q[..., 2]
    return hi-lo+10*np.maximum(lo-truth, 0)+10*np.maximum(truth-hi, 0)


def validate_seed(seed, now):
    unsigned = {k:v for k,v in seed.items() if k != 'seed_id'}
    if (seed.get('seed_id') != digest(unsigned) or seed.get('policy') != POLICY
            or seed.get('protocol_hash') != PROTOCOL_HASH or seed.get('training_hash') != training_hash()
            or set(seed.get('horizons', {})) != set(map(str, HORIZONS))):
        raise ValueError('range seed integrity or policy mismatch')
    if utc(seed['available_at']) > utc(now):
        raise ValueError('range seed was unavailable at issuance')
    for h, rows in seed['horizons'].items():
        seen = set()
        for row in rows:
            origin, end = utc(row['slot']), utc(row['target_end'])
            if (origin != origin.floor('D') or origin.isoformat() in seen
                    or end-origin != pd.Timedelta(hours=int(h)+1)
                    or end > utc(seed['data_as_of']) or end >= utc(seed['available_at'])):
                raise ValueError('invalid historical range feedback dates')
            seen.add(origin.isoformat())
            losses = np.asarray(row['losses'], float)
            if losses.shape != (3,) or not np.isfinite(losses).all() or (losses < 0).any():
                raise ValueError('invalid historical range losses')
    return seed


def weights_for(seed_rows, live_records, slot, h):
    """Feedback is mature AND actually available; future revisions cannot enter."""
    slot = utc(slot); cutoff = slot-pd.Timedelta(hours=1)
    start = slot-pd.Timedelta(days=POLICY['window_days'])
    paired = {r['slot']:r for r in seed_rows if start <= utc(r['slot']) and utc(r['target_end']) < cutoff}
    # One first issued pair per day prevents hourly feedback overwhelming daily research.
    days = set()
    for r in sorted(live_records, key=lambda x:x['issued_at']):
        if r['horizon_seconds'] != h*3600:
            continue
        origin = utc(r['slot']); day = origin.floor('D').isoformat()
        if day in days:
            continue
        days.add(day)
        y = r.get('outcome')
        if (origin < start or utc(r['target_end']) >= cutoff or not y
                or utc(y['truth_available_at']) >= cutoff or utc(r['outcome_evaluated_at']) >= cutoff):
            continue
        losses = interval_scores(np.asarray(r['range_experts'])/r['reference_price']-1,
                                 y['actual_price']/r['reference_price']-1)
        paired.pop(day, None)
        paired[origin.isoformat()] = {'slot':origin.isoformat(), 'target_end':r['target_end'],
                                      'losses':losses.tolist(), 'live':True}
    rows = sorted(paired.values(), key=lambda x:x['slot'])
    if len(rows) < POLICY['minimum_daily_origins']:
        raise ValueError('insufficient mature daily range feedback')
    decay = np.array([2**(-(slot-utc(r['target_end'])).total_seconds()/86400/POLICY['half_life_days']) for r in rows])
    losses = np.average([r['losses'] for r in rows], axis=0, weights=decay)
    logits = -POLICY['sensitivity']*losses/max(float(losses[1]), 1e-8)
    weights = np.exp(logits-logits.max()); weights /= weights.sum()
    return weights, {'rows':len(rows), 'live_rows':sum(r.get('live',False) for r in rows),
                     'target_end':max(r['target_end'] for r in rows), 'cutoff':cutoff.isoformat()}


def mix_bounds(experts, weights):
    q = np.asarray(experts, float)
    if q.shape != (3,3) or not np.isfinite(q).all() or (q <= 0).any() or (np.diff(q) < 0).any():
        raise ValueError('invalid current range experts')
    mixed = np.average(q, axis=0, weights=weights)
    mixed[1] = q[0,1]  # Range-only: no alteration to the incumbent center.
    mixed[0] = min(mixed[0], mixed[1]); mixed[2] = max(mixed[2], mixed[1])
    return mixed.tolist()


def install_seed(source, root, *, now=None):
    source,root=Path(source),Path(root)
    seed=validate_seed(json.loads(source.read_text()),utc(now))
    target=root/'range_seed.json'
    if target.exists():
        old=validate_seed(json.loads(target.read_text()),utc(now))
        if utc(seed['data_as_of']) < utc(old['data_as_of']) or utc(seed['available_at']) < utc(old['available_at']):
            raise ValueError('range seed would move backwards')
    root.mkdir(parents=True,exist_ok=True)
    temporary=root/'range_seed.tmp';temporary.write_bytes(source.read_bytes());temporary.replace(target)
    return seed['seed_id']


def infer(root, incumbent, candidates, bars, *, now=None, clock=None):
    root = Path(root); state = root/'range_shadow'
    current = utc(now); clock = clock or (utc if now is None else lambda:current)
    settle(state, bars, now=current)
    past = history(state); output = []; errors = []
    path = root/'range_seed.json'
    if not path.exists(): path = BOOTSTRAP
    try:
        seed = validate_seed(json.loads(path.read_text()), current)
    except (ValueError, OSError, KeyError) as exc:
        return [], past, [{'reason':str(exc)}]
    # Keep every seed revision required to reconstruct future weight decisions.
    saved = root/'range_seeds'/f"{seed['seed_id']}.json"
    saved.parent.mkdir(parents=True, exist_ok=True)
    if not saved.exists(): saved.write_bytes(path.read_bytes())
    peers = {(r['slot'],r['horizon_seconds']):r for r in candidates}
    for active in incumbent:
        h = active['horizon_seconds']//3600
        if h not in HORIZONS: continue
        existing = next((r for r in past if r['slot']==active['slot'] and r['horizon_seconds']==h*3600), None)
        if existing:
            output.append({k:v for k,v in existing.items() if k not in ('outcome','truth_revision','published_at','outcome_evaluated_at')})
            continue
        try:
            peer = peers.get((active['slot'],h*3600))
            if not peer or any(active[k] != peer[k] for k in ('reference_price','input_cutoff','target_end','window_start')):
                raise ValueError('current paired candidate unavailable')
            weights, audit = weights_for(seed['horizons'][str(h)], past, active['slot'], h)
            experts = [active['price_quantiles'],active['baseline']['price_quantiles'],peer['price_quantiles']]
            record = copy.deepcopy(active)
            for key in ('forecast_id','issued_at','role','target_definition'):
                record.pop(key, None)
            record.update(price_quantiles=mix_bounds(experts,weights), selected_model=VERSION,
                          model_version=digest({'family':VERSION,'seed':seed['seed_id'],
                                                'incumbent':active['model_version'],'candidate':peer['model_version']}),
                          range_weights=dict(zip(EXPERTS,weights.tolist())), range_experts=experts,
                          range_feedback=audit, range_seed=seed['seed_id'],
                          incumbent_forecast_id=active['forecast_id'], paired_candidate_id=peer['forecast_id'],
                          candidate_scope='range_only_shadow; incumbent center and probabilities retained')
            output.append(issue(state,record,now=clock()))
        except (ValueError, KeyError) as exc:
            errors.append({'horizon':h,'reason':str(exc)})
    return output, history(state), errors
