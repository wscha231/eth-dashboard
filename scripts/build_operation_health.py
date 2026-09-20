"""Attribute the current run using Actions times and this run's actual feed errors."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path

INFERENCE = 'Collect closed bars, settle matured outcomes and infer from saved models'
REFRESH_STEPS = ('bootstrap_research', 'bootstrap_download', 'bootstrap_install',
                 'research', 'research_download', 'research_install',
                 'historical', 'historical_download', 'historical_install',
                 'range_seed', 'range_download', 'range_install')


def stamp(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
    if dt.tzinfo is None:
        raise ValueError('timezone-aware timestamp required')
    return dt.astimezone(timezone.utc)


def attribute(metadata, signal, *, run_id, attempt, now=None):
    now = now or datetime.now(timezone.utc)
    if (str(metadata.get('id')) != str(run_id) or str(metadata.get('run_attempt')) != str(attempt)):
        raise ValueError('Actions run/attempt does not match current execution')
    created = stamp(metadata['created_at'])
    jobs = [j for j in metadata['jobs'] if j['name'] == 'forecast' and str(j.get('run_attempt')) == str(attempt)]
    if len(jobs) != 1:
        raise ValueError('exact current forecast job required')
    job = jobs[0]
    started = stamp(job['started_at'])
    steps = [s for s in job['steps'] if s['name'] == INFERENCE]
    step = steps[0] if len(steps) == 1 else {}
    result = {'run_created_at': created.isoformat(), 'job_started_at': started.isoformat(),
              'dispatch_to_job_seconds': (started-created).total_seconds() if int(attempt) == 1 else None,
              'event': metadata['event'], 'run_sha': metadata['head_sha'],
              'inference_step': step, 'causes': [], 'slot': None,
              'shared_lock_cause': 'unproven; dispatch-to-job includes more than concurrency waiting',
              'missing_schedule_cause': 'unknown; absence of a run is not proof of scheduler failure'}
    if step.get('conclusion') not in ('success', 'failure'):
        result['causes'].append('inference_not_reached_or_cancelled')
        return result
    begin, end = stamp(step['started_at']), stamp(step['completed_at'])
    result['job_to_inference_seconds'] = (begin-started).total_seconds()
    try:
        generated = stamp(signal['generated_at'])
    except (KeyError, TypeError, ValueError, AttributeError):
        generated = None
    if generated is None or not begin <= generated <= end + timedelta(seconds=1) or generated > now:
        result['causes'].append('no_current_inference_report')
        return result
    slot = stamp(signal['expected_slot'])
    if slot.minute or slot.second or slot.microsecond or not slot <= begin < slot+timedelta(hours=1):
        result['causes'].append('inference_slot_binding_unproven')
        return result
    deadline = slot + timedelta(minutes=55)
    result.update(slot=slot.isoformat(), issuance_deadline=deadline.isoformat(),
                  forecast_status=signal.get('status'), errors=signal.get('errors', []),
                  release_id=signal.get('release_id'))
    if created >= deadline:
        result['causes'].append('run_created_after_issuance_deadline')
    elif started >= deadline:
        result['causes'].append('job_started_after_issuance_deadline')
    elif begin >= deadline:
        result['causes'].append('preinference_steps_crossed_deadline')
    for error in signal.get('errors', []):
        reason = error.get('reason', '')
        if 'issuance_deadline' in reason or 'current hourly slot' in reason:
            result['causes'].append('issuance_clock_guard')
        elif 'input window unavailable' in reason:
            result['causes'].append('input_window_unavailable')
        elif 'checkpoint' in reason or 'research release' in reason:
            result['causes'].append('checkpoint_unavailable')
        else:
            result['causes'].append('other_inference_error')
    result['causes'] = sorted(set(result['causes']))
    return result


def build(metadata, signal, outcomes, *, run_id, attempt, now=None):
    if set(outcomes) != set(REFRESH_STEPS) or any(v not in ('success', 'failure', 'cancelled', 'skipped', '') for v in outcomes.values()):
        raise ValueError('explicit original refresh step outcomes required')
    failed = [k for k, v in outcomes.items() if v in ('failure', 'cancelled')]
    return {'schema': 1, 'run_id': str(run_id), 'attempt': str(attempt),
            'generated_at': (now or datetime.now(timezone.utc)).isoformat(),
            'input_refresh_degraded': bool(failed), 'failed_refresh_steps': failed,
            'refresh_step_outcomes': outcomes,
            'attribution': attribute(metadata, signal, run_id=run_id, attempt=attempt, now=now),
            'historical_gaps_repaired': False, 'promotion': 'none'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata', type=Path, default=Path('operation-health/actions.json'))
    p.add_argument('--root', type=Path, default=Path('lake/signals'))
    p.add_argument('--output', type=Path, default=Path('operation-health/health.json'))
    args = p.parse_args()
    raw = None
    try:
        raw = (args.root/'signals.json').read_bytes()
        signal = json.loads(raw)
    except (OSError, ValueError):
        signal = None
    outcomes = {k: os.environ.get('REFRESH_' + k.upper(), '') for k in REFRESH_STEPS}
    report = build(json.loads(args.metadata.read_text()), signal, outcomes,
                   run_id=os.environ['GITHUB_RUN_ID'], attempt=os.environ['GITHUB_RUN_ATTEMPT'])
    report['signals_sha256'] = hashlib.sha256(raw).hexdigest() if raw is not None else None
    report['refresh_receipts'] = {}
    for name in ('bootstrap', 'research', 'historical', 'range'):
        receipt = args.output.parent / (name + '.json')
        if receipt.exists():
            data = json.loads(receipt.read_text())
            if data.get('run_id') != report['run_id'] or data.get('attempt') != report['attempt']:
                raise ValueError('wrong refresh receipt run/attempt')
            report['refresh_receipts'][name] = data
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(report, indent=2, allow_nan=False)+'\n'
    args.output.write_text(encoded)
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / 'operation_health.json').write_text(encoded)
    with open(os.environ['GITHUB_OUTPUT'], 'a') as handle:
        handle.write('refresh_outcome=' + ('failure' if report['input_refresh_degraded'] else 'success') + '\n')
    print(encoded, flush=True)


if __name__ == '__main__':
    main()
