"""Versioned public evidence, derived from immutable records; no forecast rewriting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluate import metrics

VERSION = 1


def write_object(root, value, prefix):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    sha = hashlib.sha256(raw).hexdigest()
    name = f'{prefix}_{sha}.json'
    path = Path(root) / 'archive' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        temp = path.with_suffix('.tmp'); temp.write_bytes(raw); temp.replace(path)
    return {'path': 'archive/' + name, 'sha256': sha, 'bytes': len(raw)}


def live_point(record):
    """Keep the original payload plus an explicitly eligible, normalized evaluation row."""
    r = record
    timely = bool(r.get('published_at') and pd.Timestamp(r['published_at']) < pd.Timestamp(r['window_start']))
    q = np.log(np.asarray(r['price_quantiles']) / r['reference_price'])
    point = {**r, 'eligible': timely, 'settled': r.get('outcome') is not None,
             'p_down': r['terminal_down_flat_up'][0], 'p_flat': r['terminal_down_flat_up'][1],
             'p_up': r['terminal_down_flat_up'][2], 'q10': float(q[0]), 'q50': float(q[1]), 'q90': float(q[2]),
             'threshold_up': r['alert_thresholds']['up'], 'threshold_down': r['alert_thresholds']['down'],
             'trend_state': r.get('origin_market', {}).get('trend_state', 'unknown'),
             'volatility_state': r.get('origin_market', {}).get('volatility_state', 'unknown')}
    if r.get('outcome'):
        point.update(r['outcome'])
    return point


def export_archive(root, groups, *, kind, as_of):
    horizons = {}
    for horizon, points in groups.items():
        points = sorted(points, key=lambda r: (r['slot'], r.get('forecast_id', '')))
        partitions = {}
        for point in points:
            partitions.setdefault(point['slot'][:4], []).append(point)
        shards = []
        for month, rows in sorted(partitions.items()):
            shards.append({**write_object(root, {'schema_version': VERSION, 'kind': kind, 'horizon_hours': int(horizon), 'points': rows}, f'{kind}_{horizon}_{month}'),
                           'rows': len(rows), 'first_origin': rows[0]['slot'], 'last_origin': rows[-1]['slot']})
        eligible = [p for p in points if p.get('eligible', True) and p.get('settled', True)]
        independent = []; end = None
        for p in eligible:
            if end is None or pd.Timestamp(p['slot']) >= end:
                independent.append(p); end = pd.Timestamp(p['target_end'])
        horizons[str(horizon)] = {'rows': len(points), 'scored': len(eligible),
                                 'pending': sum(not p.get('settled', True) for p in points),
                                 'excluded_publication': sum(not p.get('eligible', True) for p in points),
                                 'nonoverlap': len(independent), 'metrics': metrics(eligible), 'shards': shards}
    manifest = {'schema_version': VERSION, 'kind': kind, 'as_of': as_of, 'horizons': horizons,
                'interpretation': 'Historical reconstruction is development evidence. Published results require on-time verification and settled outcomes. Overlapping rows are not independent trials.'}
    return write_object(root, manifest, f'{kind}_manifest')


def export_live(root, records, as_of):
    groups = {str(h): [] for h in (6, 24, 72, 168, 336, 720)}
    for r in records:
        groups.setdefault(str(r['horizon_seconds'] // 3600), []).append(live_point(r))
    return export_archive(root, groups, kind='published', as_of=as_of)


def validate_object(root, entry):
    """Used by publisher and tests. Manifest paths may never escape the archive."""
    path = Path(entry['path'])
    if path.parent != Path('archive') or path.name != str(path).split('/')[-1] or path.suffix != '.json':
        raise ValueError('unsafe archive path')
    raw = (Path(root) / path).read_bytes()
    if len(raw) != entry['bytes'] or hashlib.sha256(raw).hexdigest() != entry['sha256']:
        raise ValueError('archive integrity mismatch')
    return json.loads(raw)


def compare_published(incumbent, candidate):
    """Pair the first timely publication per horizon/origin, never cherry-pick outcomes."""
    from .evaluate import paired_block_interval
    def indexed(records):
        result={}
        for r in sorted(records,key=lambda r:r.get('published_at') or 'z'):
            p=live_point(r)
            if p['eligible'] and p['settled']:
                result.setdefault((r['horizon_seconds']//3600,r['slot']),p)
        return result
    a,b=indexed(incumbent),indexed(candidate);report={}
    for h in (6,24,72,168,336,720):
        keys=sorted(k for k in a.keys() & b.keys() if k[0]==h)
        # Truth revisions must agree; unaligned settlements are held out until repaired.
        keys=[k for k in keys if all(a[k][field]==b[k][field] for field in ('target_end','return','terminal','up','down'))]
        original=[a[k] for k in keys];shadow=[b[k] for k in keys]
        independent=0;end=None
        for r in shadow:
            if end is None or pd.Timestamp(r['slot'])>=end:
                independent+=1;end=pd.Timestamp(r['target_end'])
        report[str(h)]={'paired_rows':len(keys),'nonoverlap':independent,'candidate':metrics(shadow),
                        'incumbent':metrics(original),'paired_event_brier':paired_block_interval(shadow,original,h),
                        'candidate_versions':sorted({r['model_version'] for r in shadow}),
                        'status':'insufficient_prospective_evidence' if independent<20 else 'requires_multi_metric_review',
                        'promotion':'incumbent_retained'}
    return report
