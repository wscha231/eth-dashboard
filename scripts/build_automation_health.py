"""Small read-only automation receipt; no training, publishing or source downloads."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

HORIZONS = (6, 24, 72, 168, 336, 720)
STAGES = ('inference', 'publication', 'variance_daily', 'dense_variance', 'dense_persist', 'failure_review',
          'availability_shadow', 'final_snapshot', 'availability_snapshot', 'input_refresh', 'operation_audit')
ALLOWED = {'success', 'failure', 'skipped', 'cancelled', ''}
MAX_REPORT_BYTES = 4 * 1024 * 1024


def utc(value):
    value = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    return value.astimezone(timezone.utc)


def finite_json(raw):
    def reject(value):
        raise ValueError('non-finite JSON number: ' + value)
    value = json.loads(raw, parse_constant=reject)
    if not isinstance(value, dict):
        raise ValueError('report must be an object')
    return value


def build(root, *, run_id, attempt, commit, outcomes, now=None):
    """Aggregate an identified final snapshot, never infer success from old files."""
    root = Path(root)
    now = utc(now or datetime.now(timezone.utc))
    if not str(run_id).isdigit() or not str(attempt).isdigit() or int(attempt) < 1:
        raise ValueError('real numeric run/attempt identity required')
    if set(outcomes) != set(STAGES) or any(v not in ALLOWED for v in outcomes.values()):
        raise ValueError('explicit outcome for each supported stage required')
    attention, evidence = [], {}
    failed = [k for k, v in outcomes.items() if v in ('failure', 'cancelled')]
    attention.extend('stage_' + k + '_' + outcomes[k] for k in failed)
    snapshot_ready = outcomes['final_snapshot'] == 'success' and root.is_dir()

    def read(name, stage=None, identified=False):
        if not snapshot_ready:
            attention.append(name + ':no_completed_snapshot')
            return None
        if stage and outcomes[stage] != 'success':
            attention.append(name + ':stage_not_success')
            return None
        try:
            p = root / name
            if not p.resolve().is_relative_to(root.resolve()):
                raise ValueError('report outside snapshot')
            with p.open('rb') as handle:
                raw = handle.read(MAX_REPORT_BYTES + 1)
            if len(raw) > MAX_REPORT_BYTES:
                raise ValueError('report exceeds bounded read')
            data = finite_json(raw)
            evidence[name] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
            if identified and str(data.get('run_id')) != str(run_id):
                raise ValueError('wrong or missing source run_id')
            if identified and int(attempt) > 1 and str(data.get('attempt', data.get('run_attempt', ''))) != str(attempt):
                raise ValueError('legacy receipt cannot bind this retry attempt')
            if data.get('status') == 'failed':
                raise ValueError('source report records a failure')
            return data
        except (OSError, TypeError, ValueError) as exc:
            attention.append(name + ':' + type(exc).__name__ + ':' + str(exc)[:180])
            return None

    result = {'schema': 1, 'repository': 'wscha231/eth-dashboard', 'run_id': str(run_id),
              'attempt': str(attempt), 'commit': commit, 'generated_at': now.isoformat(),
              'stage_outcomes': {k: v or 'not_reached' for k, v in outcomes.items()},
              'core_publication_step_succeeded': outcomes['publication'] == 'success',
              'optional_degraded': any(outcomes[k] in ('failure', 'cancelled') for k in STAGES[2:]),
              'snapshot_complete': snapshot_ready and outcomes['availability_snapshot'] == 'success',
              'public_site_rechecked_by_this_report': False,
              'workflow_final_conclusion': 'check_Actions_after_completion',
              'drive_archive_result': 'check_separate_archive_receipt',
              'model_promotion': 'unchanged_not_evaluated',
              'availability': {'status': 'unknown'}, 'continuity': {'status': 'unknown'},
              'limitations': ['A task success is not source eligibility, predictive skill or a new public receipt.',
                              'Historical cumulative gaps and recent-window continuity are separate.',
                              'Legacy receipts lacking an attempt identity are not trusted on retries.']}
    recovery = read('recovery_health.json', identified=True)
    if recovery is not None:
        if recovery.get('core_publication') != outcomes['publication']:
            attention.append('recovery_health:publication_outcome_disagrees')
        result['recovery_summary_schema'] = recovery.get('schema')
    operation = read('operation_health.json', stage='operation_audit', identified=True)
    if operation is not None:
        result['operation'] = {k: operation.get(k) for k in
                               ('input_refresh_degraded', 'failed_refresh_steps', 'attribution')}
        if operation.get('input_refresh_degraded'):
            attention.append('input_refresh:rejected_or_incomplete_optional_update')
    variance_daily = read('variance_daily_status.json', stage='variance_daily', identified=True)
    if variance_daily is not None:
        result['variance_daily'] = {k: variance_daily.get(k) for k in
                                    ('status', 'phase', 'generated_at', 'source_as_of',
                                     'issued_horizons', 'issuance_audit', 'errors')}
        if variance_daily.get('status') == 'not_due':
            result['variance_daily']['status'] = 'not_due'
        elif variance_daily.get('status') != 'success':
            attention.append('variance_daily:invalid_status')

    availability = read('availability_stage_status.json', stage='availability_shadow', identified=True)
    if availability is not None:
        decision = availability.get('decision') or {}
        if not isinstance(decision, dict) or availability.get('status') != 'success':
            attention.append('availability:invalid_decision_or_status')
        else:
            result['availability'] = {k: decision.get(k) for k in ('status', 'reason', 'origin',
                                       'new_forecasts', 'new_endpoint_outcomes', 'strict_missing_counts')}
            result['availability'].update(public_delivery=False, promotion='none',
                                           totals=availability.get('totals'))
            if decision.get('status') == 'blocked':
                attention.append('availability:input_or_deadline_blocked')
            elif decision.get('status') not in ('not_needed', 'eligible'):
                attention.append('availability:unknown_decision')
    for name, stage in (('dense_stage_status.json', 'dense_variance'), ('failure_review.json', 'failure_review')):
        data = read(name, stage=stage, identified=True)
        if data is not None and name == 'failure_review.json':
            result['failure_review_as_of'] = data.get('as_of')
            result['failure_review_horizons'] = sorted(data.get('actual_performance', {}))
    continuity = read('continuity_audit.json', stage='publication')
    if continuity is not None:
        try:
            age = (now - utc(continuity['as_of'])).total_seconds()
            rows = continuity['horizons']
            if age < 0 or age > 7200 or {r['horizon_hours'] for r in rows} != set(HORIZONS) or len(rows) != 6:
                raise ValueError('stale/future/incomplete continuity snapshot')
            fields = ('horizon_hours', 'expected_slots', 'issued_slots', 'published_before_window_slots',
                      'missing_issue_slots', 'overdue_settlement_records', 'recent_24h')
            result['continuity'] = {'status': 'pinned_snapshot_only', 'as_of': continuity['as_of'],
                                    'from_slot': continuity.get('from_slot'), 'through_slot': continuity.get('through_slot'),
                                    'horizons': [{k: r.get(k) for k in fields} for r in rows]}
            for r in rows:
                recent = r.get('recent_24h', {})
                if recent.get('published_before_window_slots', 0) < recent.get('expected_slots', 0):
                    attention.append('recent_24h:unproven_delivery_h' + str(r['horizon_hours']))
                if r.get('overdue_settlement_records', 0):
                    attention.append('overdue_settlement_h' + str(r['horizon_hours']))
        except (KeyError, TypeError, ValueError) as exc:
            attention.append('continuity:' + str(exc)[:180])
    result['evidence'] = evidence
    result['attention_reasons'] = sorted(set(attention))
    result['status'] = 'attention' if attention or outcomes['publication'] != 'success' else 'ready'
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('final-hourly-state'))
    p.add_argument('--output', type=Path, default=Path('automation-health/health.json'))
    a = p.parse_args()
    outcomes = {k: os.environ.get('HEALTH_' + k.upper(), '') for k in STAGES}
    report = build(a.root, run_id=os.environ.get('GITHUB_RUN_ID', ''),
                   attempt=os.environ.get('GITHUB_RUN_ATTEMPT', ''),
                   commit=os.environ.get('GITHUB_SHA', ''), outcomes=outcomes)
    raw = (json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(raw)
    if outcomes['final_snapshot'] == 'success' and a.root.is_dir():
        (a.root / 'automation_health.json').write_bytes(raw)
    print(json.dumps({'status': report['status'], 'run_id': report['run_id'],
                      'attention_reasons': report['attention_reasons']}), flush=True)


if __name__ == '__main__':
    main()
