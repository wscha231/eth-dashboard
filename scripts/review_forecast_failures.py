"""Read-only failure review and a frozen OFFLINE matured-error diagnostic.

Historical reconstructions never enter the prospective ledger. No promotion or
threshold/weight changes are performed by this module.
"""
from __future__ import annotations
import argparse
from collections import Counter, deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import statistics

HORIZONS = (6, 24, 72, 168, 336, 720)
POLICY = {'version': 'matured_median_feedback_diagnostic_v1', 'max_prior_origins': 90,
          'min_prior_origins': 60, 'shrinkage_pseudocount': 90, 'truth_maturity_lag_hours': 1,
          'bootstrap_calendar_block_days': 90, 'bootstrap_draws': 1000, 'seed': 20260919,
          'role': 'offline_diagnostic_only', 'promotion_allowed': False}


def utc(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(result, datetime) or result.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    return result.astimezone(timezone.utc)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise ValueError('finite number required')
    return float(value)


def verified_json(root, ref):
    relative = ref['path']
    if not re.fullmatch(r'archive/[A-Za-z0-9_.-]+\.json', relative):
        raise ValueError('unsafe historical archive path')
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('archive escapes root')
    raw = path.read_bytes()
    if len(raw) != ref['bytes'] or hashlib.sha256(raw).hexdigest() != ref['sha256']:
        raise ValueError('historical archive integrity mismatch')
    return json.loads(raw)


def historical_rows(root):
    root = Path(root)
    replay = json.loads((root / 'replay.json').read_text())
    manifest = verified_json(root, replay['archive'])
    if manifest['kind'] != 'historical' or set(manifest['horizons']) != {str(h) for h in HORIZONS}:
        raise ValueError('historical six-horizon manifest required')
    result, calendars = {}, {}
    for h in HORIZONS:
        info = manifest['horizons'][str(h)]
        points = []
        for ref in info['shards']:
            shard = verified_json(root, ref)
            if shard['kind'] != 'historical' or shard['horizon_hours'] != h or len(shard['points']) != ref['rows']:
                raise ValueError('historical shard identity/count mismatch')
            points.extend(shard['points'])
        if len(points) != info['rows']:
            raise ValueError('historical total row count mismatch')
        points.sort(key=lambda p: utc(p['slot']))
        seen, valid, invalid = set(), [], []
        for point in points:
            try:
                slot, end = utc(point['slot']), utc(point['target_end'])
                if slot in seen:
                    raise ValueError('duplicate historical origin')
                seen.add(slot)
                if point.get('horizon_hours') != h or end != slot + timedelta(hours=h+1):
                    raise ValueError('historical target mismatch')
                if (utc(point['training_target_end']) >= slot or utc(point['validation_target_end']) >= slot
                        or utc(point['fit_cutoff']) > slot):
                    raise ValueError('historical label/fit cutoff violation')
                for key in ('q50', 'return', 'hit_up', 'hit_down', 'threshold_up', 'threshold_down'):
                    number(point[key])
                for key in ('hit_up', 'hit_down', 'threshold_up', 'threshold_down'):
                    if not 0 <= point[key] <= 1:
                        raise ValueError('probability/threshold outside [0,1]')
                if point['up'] not in (0, 1) or point['down'] not in (0, 1):
                    raise ValueError('invalid historical binary outcome')
                valid.append(point)
            except (KeyError, ValueError, TypeError) as exc:
                invalid.append({'slot': point.get('slot'), 'reason': str(exc)})
        result[h] = valid
        slots = sorted(seen)
        stride = int(replay['protocol']['replay_stride_hours'])
        if stride <= 0:
            raise ValueError('invalid replay cadence')
        missing = []
        if slots:
            cursor = slots[0]
            while cursor <= slots[-1]:
                if cursor not in seen:
                    missing.append(cursor.isoformat())
                cursor += timedelta(hours=stride)
        calendars[str(h)] = {'cadence_hours': stride, 'stored_rows': len(points), 'valid_rows': len(valid),
                             'first_origin': slots[0].isoformat() if slots else None,
                             'last_origin': slots[-1].isoformat() if slots else None,
                             'not_in_archive_slots': missing, 'invalid_rows': invalid,
                             'missing_reason': 'not_in_pinned_archive; input-versus-checkpoint cause not yet proven',
                             'published_source_eligible_calendar_coverage': replay['horizons'][str(h)].get('eligible_calendar_coverage')}
    return result, calendars


def actual_rows(root, as_of):
    path = Path(root) / 'issued.db'
    before = digest(path)
    rows, exclusions, runs = [], Counter(), Counter()
    with sqlite3.connect(path.resolve().as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        forecasts = [dict(r) for r in db.execute('SELECT * FROM forecasts ORDER BY issued_at,forecast_id')]
        deliveries, outcomes = {}, {}
        for row in db.execute('SELECT * FROM deliveries'):
            if row['release_id'] and utc(row['verified_at']) <= as_of:
                deliveries.setdefault(row['forecast_id'], []).append(utc(row['verified_at']))
        for row in db.execute('SELECT * FROM outcomes ORDER BY revision'):
            if utc(row['evaluated_at']) <= as_of:
                outcomes.setdefault(row['forecast_id'], []).append(dict(row))
        for row in db.execute('SELECT * FROM runs'):
            if utc(row['started_at']) <= as_of:
                runs[row['status']] += 1
    seen = set()
    for row in forecasts:
        try:
            point = json.loads(row['payload'])
            h = int(row['horizon_seconds']) // 3600
            slot, issued, start, end = map(utc, [row['slot'], row['issued_at'], point['window_start'], row['target_end']])
            if issued > as_of or slot > as_of:
                continue
            if (h not in HORIZONS or row['horizon_seconds'] != h*3600 or point['forecast_id'] != row['forecast_id']
                    or start != slot+timedelta(hours=1) or end != start+timedelta(hours=h)
                    or utc(point['slot']) != slot or utc(point['issued_at']) != issued or utc(point['target_end']) != end
                    or point['horizon_seconds'] != row['horizon_seconds']
                    or utc(point['input_cutoff']) != slot or not slot <= utc(point['available_at']) <= issued < start
                    or issued >= slot+timedelta(minutes=55)):
                raise ValueError('invalid issuance contract')
            key = (slot, h, row['target_definition'])
            if key in seen:
                exclusions['later_model_same_origin'] += 1
                continue
            seen.add(key)
            if end > as_of:
                exclusions['not_yet_mature'] += 1
                continue
            receipt = next((t for t in sorted(deliveries.get(row['forecast_id'], [])) if issued <= t < start), None)
            if receipt is None:
                exclusions['no_proven_timely_delivery'] += 1
                continue
            outcome = None
            for candidate in reversed(outcomes.get(row['forecast_id'], [])):
                truth = json.loads(candidate['payload'])
                if end <= utc(truth['truth_available_at']) <= utc(candidate['evaluated_at']) <= as_of:
                    outcome = truth
                    break
            if outcome is None:
                exclusions['matured_without_valid_truth'] += 1
                continue
            ref = number(point['reference_price'])
            q50 = math.log(number(point['price_quantiles'][1]) / ref)
            rows.append({'slot': slot.isoformat(), 'target_end': end.isoformat(), 'horizon_hours': h,
                         'forecast_id': row['forecast_id'], 'model_version': row['model_version'],
                         'target_definition': row['target_definition'], 'q50': q50,
                         'return': number(outcome['return']), 'up': int(outcome['up']), 'down': int(outcome['down']),
                         'hit_up': number(point['hit_up']), 'hit_down': number(point['hit_down']),
                         'threshold_up': number(point['alert_thresholds']['up']),
                         'threshold_down': number(point['alert_thresholds']['down']),
                         'verified_at': receipt.isoformat(), 'selected_model': point.get('selected_model')})
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            exclusions['invalid_record:' + str(exc)] += 1
    if before != digest(path):
        raise RuntimeError('ledger changed during read-only review')
    return rows, {'ledger_sha256': before, 'excluded': dict(exclusions),
                  'recorded_run_status': dict(runs),
                  'run_status_is_not_delivery_status': True}


def score(rows, field='q50'):
    if not rows:
        return {'rows': 0, 'mae': None, 'no_change_mae': None, 'skill_vs_no_change': None, 'rmse': None}
    mae = statistics.fmean(abs(p['return']-p[field]) for p in rows)
    no_change = statistics.fmean(abs(p['return']) for p in rows)
    result = {'rows': len(rows), 'mae': mae, 'no_change_mae': no_change,
              'skill_vs_no_change': 1-mae/no_change if no_change else None,
              'rmse': math.sqrt(statistics.fmean((p['return']-p[field])**2 for p in rows))}
    for side in ('up', 'down'):
        events = sum(p[side] for p in rows)
        alerts = [p for p in rows if p['hit_'+side] >= p['threshold_'+side]]
        hits = sum(p[side] for p in alerts)
        result[side] = {'events': events, 'alerts': len(alerts), 'hits': hits,
                        'missed': events-hits, 'false_alerts': len(alerts)-hits,
                        'recall': hits/events if events else None,
                        'precision': hits/len(alerts) if alerts else None,
                        'brier': statistics.fmean((p['hit_'+side]-p[side])**2 for p in rows)}
    return result


def feedback_predictions(points):
    """Only labels matured strictly before each origin; frozen original-q50 residuals."""
    points = sorted(points, key=lambda p: utc(p['slot']))
    if len({p['slot'] for p in points}) != len(points):
        raise ValueError('feedback requires unique same-horizon origins')
    if len({p['horizon_hours'] for p in points}) > 1:
        raise ValueError('feedback cannot pool horizons')
    prior, cursor, result = deque(maxlen=90), 0, []
    for i, point in enumerate(points):
        origin = utc(point['slot'])
        while cursor < i and utc(points[cursor]['target_end'])+timedelta(hours=1) < origin:
            previous = points[cursor]
            prior.append(previous)
            cursor += 1
        n = len(prior)
        correction = (n/(n+90))*statistics.median(p['return']-p['q50'] for p in prior) if n >= 60 else 0.0
        result.append({**point, 'feedback_q50': point['q50']+correction, 'feedback_correction': correction,
                       'feedback_labels': n,
                       'latest_feedback_maturity': (utc(prior[-1]['target_end'])+timedelta(hours=1)).isoformat() if prior else None})
    return result


def feedback_report(points):
    rows = feedback_predictions(points)
    base, new = score(rows), score(rows, 'feedback_q50')
    blocks = {}
    for p in rows:
        key = utc(p['slot']).date().toordinal() // 90
        blocks.setdefault(key, []).append(abs(p['return']-p['q50'])-abs(p['return']-p['feedback_q50']))
    sums = [(sum(v), len(v)) for v in blocks.values()]
    rng = random.Random(POLICY['seed'])
    positives = 0
    if sums:
        for _ in range(1000):
            sample = [rng.choice(sums) for _ in sums]
            positives += sum(a for a, n in sample) > 0
    years = sorted({p['slot'][:4] for p in rows})
    regimes = sorted({p.get('trend_state', 'unknown') for p in rows})
    return {'policy': POLICY, 'evaluation_role': 'reused_history_diagnostic_not_untouched_holdout',
            'historical_truth_availability': 'target_end_plus_1h_proxy_not_original_receipt',
            'incumbent': base, 'feedback': new,
            'skill_vs_incumbent': 1-new['mae']/base['mae'] if base['mae'] else None,
            'calendar_blocks': len(sums), 'paired_block_bootstrap_positive_fraction': positives/1000 if sums else None,
            'corrected_rows': sum(p['feedback_labels'] >= 60 for p in rows),
            'by_year': {y: {'incumbent': score([p for p in rows if p['slot'].startswith(y)]),
                            'feedback': score([p for p in rows if p['slot'].startswith(y)], 'feedback_q50')} for y in years},
            'by_regime': {r: {'incumbent': score([p for p in rows if p.get('trend_state','unknown') == r]),
                              'feedback': score([p for p in rows if p.get('trend_state','unknown') == r], 'feedback_q50')} for r in regimes},
            'promotion_allowed': False}


def nonoverlap(points):
    result, end = [], None
    for point in sorted(points, key=lambda p: utc(p['slot'])):
        if end is None or utc(point['slot']) >= end:
            result.append(point)
            end = utc(point['target_end'])
    return result


def review(root, *, as_of=None, feedback=False):
    root = Path(root)
    now = utc(as_of or datetime.now(timezone.utc))
    points, provenance = actual_rows(root, now)
    historical, calendars = historical_rows(root)
    report = {'schema': 1, 'as_of': now.isoformat(), 'read_only': True,
              'loss_units': 'natural_log_return; do not substitute for other runs using simple-return loss',
              'actual_provenance': provenance, 'historical_calendar': calendars,
              'actual_performance': {}, 'historical_performance': {},
              'interpretation': 'Service gaps, forecast error and new-model skill are separate. No past forecast was changed.',
              'omission_cause': 'no recorded issue alone does not identify scheduler, source, checkpoint or lock failure',
              'promotion': 'none'}
    for h in HORIZONS:
        rows = [p for p in points if p['horizon_hours'] == h]
        failures = [p for p in rows if any(p[s] and p['hit_'+s] < p['threshold_'+s] for s in ('up','down'))]
        worst = sorted(failures, key=lambda p: abs(p['return']-p['q50']), reverse=True)[:5]
        cohorts = sorted({(p['model_version'],p['target_definition']) for p in rows})
        report['actual_performance'][str(h)] = {
            'timely_resolved': score(rows), 'nonoverlap_descriptive': score(nonoverlap(rows)),
            'model_target_cohorts': [{'model_version': v, 'target_definition': t,
                'metrics': score([p for p in rows if (p['model_version'],p['target_definition']) == (v,t)])} for v,t in cohorts],
            'missed_barrier_examples': worst,
            'not_independent_events': True, 'model_promoted': False}
        report['historical_performance'][str(h)] = score(historical[h])
    if feedback:
        report['feedback_diagnostic'] = {str(h): feedback_report(historical[h]) for h in HORIZONS}
    return report


def markdown(report):
    lines = ['# Forecast failure review', '', 'As of: '+report['as_of'], '',
             '| Hours | Timely resolved | MAE skill vs no-change | Up hits/events | Down hits/events |',
             '|---|---:|---:|---:|---:|']
    for h, row in report['actual_performance'].items():
        s = row['timely_resolved']; skill = s['skill_vs_no_change']
        up, down = s.get('up',{}), s.get('down',{})
        lines.append(f"| {h} | {s['rows']} | {f'{100*skill:.3f}%' if skill is not None else 'pending'} | {up.get('hits',0)}/{up.get('events',0)} | {down.get('hits',0)}/{down.get('events',0)} |")
    lines += ['', 'Overlapping windows are NOT independent events. This is descriptive, not promotion evidence.',
              'Historical rows are integrity-checked reconstructions, never genuine historical publication.',
              'Calendar gaps have not been assigned a causal explanation without run/input evidence.']
    if 'feedback_diagnostic' in report:
        lines += ['', '## Frozen matured-error feedback (offline diagnostic only)',
                  '| Hours | Origins | Incumbent MAE skill | Feedback MAE skill | Improvement vs incumbent |',
                  '|---|---:|---:|---:|---:|']
        for h, row in report['feedback_diagnostic'].items():
            a,b = row['incumbent'],row['feedback']
            lines.append(f"| {h} | {a['rows']} | {100*a['skill_vs_no_change']:.3f}% | {100*b['skill_vs_no_change']:.3f}% | {100*row['skill_vs_incumbent']:.3f}% |")
        lines += ['', 'Parameters were frozen before this diagnostic. Reused history cannot authorize production promotion.']
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('lake/signals'))
    p.add_argument('--as-of')
    p.add_argument('--feedback', action='store_true')
    args=p.parse_args()
    # Do not leave an older successful report visible when the current review fails.
    args.root.mkdir(parents=True, exist_ok=True)
    for name in ('failure_review.json','failure_review.md'):
        (args.root/name).unlink(missing_ok=True)
    try:
        report=review(args.root, as_of=args.as_of, feedback=args.feedback)
    except Exception as exc:
        error={'status':'failed','run_id':__import__('os').environ.get('GITHUB_RUN_ID'),
               'generated_at':datetime.now(timezone.utc).isoformat(),'error':str(exc)[:500]}
        (args.root/'failure_review.json').write_text(json.dumps(error,indent=2)+'\n')
        raise
    report['run_id']=__import__('os').environ.get('GITHUB_RUN_ID')
    (args.root/'failure_review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    (args.root/'failure_review.md').write_text(markdown(report))
    print(markdown(report))


if __name__ == '__main__': main()
