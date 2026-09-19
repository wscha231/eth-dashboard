"""EV1-A availability shadow: received ETH only, CENTER only, never public alpha."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
from urllib.parse import urlparse

HOUR = timedelta(hours=1)
HORIZONS = (6, 24, 72, 168, 336, 720)
PRODUCTS = ('ETH-USD', 'BTC-USD')
VERSION = 'eth169_nochange_availability_shadow_v1'
POLICY = {'version': VERSION, 'strict_bars': 721, 'fallback_eth_bars': 169,
          'deadline_minute': 55, 'horizons': list(HORIZONS), 'head': 'center',
          'role': 'availability_shadow_only', 'imputation': False,
          'public_delivery': False, 'promotion': 'none'}
STATE = 'availability_shadow'
FILES = ('issued.db', 'manifest.json', 'report.json')
REF = 'refs/remotes/origin/data/event-ledger'
GIT_PATH = 'lake/event-ledger/availability-shadow'
MAX_RAW_BYTES = 2_000_000


def utc(value):
    dt = datetime.fromisoformat(value.replace('Z', '+00:00')) if isinstance(value, str) else value
    if not isinstance(dt, datetime) or dt.tzinfo is None:
        raise ValueError('timezone-aware datetime required')
    return dt.astimezone(timezone.utc)


def clock_now():
    return datetime.now(timezone.utc)


def encode(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(encode(value).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(encode(value) + '\n', encoding='utf-8')
    tmp.replace(path)


def readonly(path):
    con = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True, timeout=10)
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA query_only=ON')
    return con


def source_view(root, as_of, origin):
    """Choose revisions only after filtering actual receipt time; never latest-vintage backfill."""
    selected, downloads = {}, {}
    with closing(readonly(Path(root) / 'observations.db')) as con:
        con.execute('BEGIN')
        # Wider than the input window so endpoint settlements from older issues can be read.
        lower = origin - timedelta(days=41)
        rows = con.execute('SELECT * FROM bars WHERE product IN (?,?) AND julianday(open_time)>=julianday(?)',
                           (*PRODUCTS, lower.isoformat())).fetchall()
        for raw in rows:
            row = dict(raw)
            opened, received = utc(row['open_time']), utc(row['observed_at'])
            if received > as_of or opened + HOUR > origin:
                continue
            if opened != opened.replace(minute=0, second=0, microsecond=0) or received < opened + HOUR:
                raise ValueError('unaligned or pre-close received candle')
            rev = row['revision']
            if not isinstance(rev, int) or rev < 1:
                raise ValueError('invalid candle revision')
            key = (row['product'], opened)
            previous = selected.get(key)
            if previous is None or rev > previous['revision']:
                selected[key] = row
        for raw in con.execute('SELECT * FROM downloads'):
            row = dict(raw)
            if utc(row['observed_at']) <= as_of:
                downloads.setdefault(row['raw_hash'], []).append(row)
    return selected, downloads


def validate_bar(row):
    values = [row[k] for k in ('open', 'high', 'low', 'close', 'volume')]
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError('non-finite or nonnumeric candle')
    o, h, l, c, v = map(float, values)
    if min(o, h, l, c) <= 0 or v < 0 or h < max(o, l, c) or l > min(o, h, c):
        raise ValueError('invalid OHLCV')
    # Exact normalization used by signal_pipeline.data.ingest.
    calculated = hashlib.sha256(json.dumps([o, h, l, c, v]).encode()).hexdigest()
    if row['content_hash'] != calculated:
        raise ValueError('normalized candle hash mismatch')
    if not isinstance(row['raw_hash'], str) or not re.fullmatch(r'[0-9a-f]{64}', row['raw_hash']):
        raise ValueError('raw SHA256 missing')


def verify_receipts(root, rows, downloads):
    """Match the actual response bytes and source receipt to every ETH input/endpoint."""
    cache = {}
    for row in rows:
        validate_bar(row)
        sha = row['raw_hash']
        matches = []
        for download in downloads.get(sha, []):
            url = urlparse(download['url'])
            if (url.scheme == 'https' and url.netloc == 'api.exchange.coinbase.com'
                    and url.path == '/products/' + row['product'] + '/candles'
                    and utc(download['observed_at']) <= utc(row['observed_at'])):
                matches.append(download)
        if not matches:
            raise ValueError('missing matching Coinbase receipt')
        if sha not in cache:
            with gzip.open(Path(root) / 'raw' / (sha + '.json.gz'), 'rb') as handle:
                raw = handle.read(MAX_RAW_BYTES + 1)
            if len(raw) > MAX_RAW_BYTES or hashlib.sha256(raw).hexdigest() != sha:
                raise ValueError('raw response integrity mismatch')
            payload = json.loads(raw)
            if not isinstance(payload, list):
                raise ValueError('invalid raw response')
            cache[sha] = payload
        expected = [int(utc(row['open_time']).timestamp()), row['low'], row['high'],
                    row['open'], row['close'], row['volume']]
        if expected not in cache[sha]:
            raise ValueError('normalized candle not in raw receipt')


def assess(root, now, view, downloads):
    origin = now.replace(minute=0, second=0, microsecond=0)
    windows, missing = {}, {}
    for product in PRODUCTS:
        times = [origin - n * HOUR for n in range(721, 0, -1)]
        windows[product] = [view[(product, t)] for t in times if (product, t) in view]
        missing[product] = [t.isoformat() for t in times if (product, t) not in view]
    result = {'policy': POLICY, 'origin': origin.isoformat(), 'assessed_at': now.isoformat(),
              'strict_missing_counts': {p: len(v) for p, v in missing.items()},
              'strict_missing_open_times': missing, 'status': 'blocked', 'reason': None}
    # Invalid data is NOT turned into an apparent missing-data fallback opportunity.
    for rows in windows.values():
        for row in rows:
            validate_bar(row)
    if now >= origin + timedelta(minutes=55):
        result['reason'] = 'issuance_deadline'
        return result
    if not any(missing.values()):
        result.update(status='not_needed', reason='strict_input_condition_satisfied')
        return result
    times = [origin - n * HOUR for n in range(169, 0, -1)]
    eth = [view[('ETH-USD', t)] for t in times if ('ETH-USD', t) in view]
    if len(eth) != 169:
        result['reason'] = 'incomplete_recent_eth'
        return result
    verify_receipts(root, eth, downloads)
    available = max(utc(r['observed_at']) for r in eth)
    fingerprint = [{k: row[k] for k in ('product', 'open_time', 'observed_at', 'revision', 'content_hash', 'raw_hash')}
                   for row in eth]
    result.update(status='eligible', reason='strict_input_gap_eth169_received',
                  reference_price=float(eth[-1]['close']), input_sha256=digest(fingerprint),
                  input_available_at=available.isoformat(), input_rows=169,
                  trigger_sha256=digest(missing))
    return result


def validate_state(folder, *, manifest=True):
    folder = Path(folder)
    with closing(readonly(folder / 'issued.db')) as con:
        if con.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('availability ledger integrity failure')
        row = con.execute('SELECT payload FROM metadata WHERE name=?', ('policy',)).fetchone()
        if row is None or json.loads(row[0]) != POLICY:
            raise ValueError('unexpected availability policy')
        for table in ('decisions', 'forecasts', 'outcomes'):
            con.execute('SELECT * FROM ' + table + ' LIMIT 1')
    if manifest:
        saved = json.loads((folder / 'manifest.json').read_text())
        if saved['policy_sha256'] != digest(POLICY) or saved['database_sha256'] != file_sha(folder / 'issued.db'):
            raise ValueError('availability manifest mismatch')


def initialize(folder):
    folder = Path(folder)
    existing = (folder / 'issued.db').exists() or (folder / 'manifest.json').exists()
    if existing:
        validate_state(folder)
        return
    if folder.exists() and any(folder.iterdir()):
        raise ValueError('partial availability state cannot initialize a new cohort')
    folder.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(folder / 'issued.db')) as con:
        con.executescript('''
        CREATE TABLE metadata(name TEXT PRIMARY KEY,payload TEXT NOT NULL);
        CREATE TABLE decisions(id TEXT PRIMARY KEY,origin TEXT NOT NULL,payload TEXT NOT NULL);
        CREATE TABLE forecasts(id TEXT PRIMARY KEY,origin TEXT NOT NULL,horizon INTEGER NOT NULL,
          target_end TEXT NOT NULL,payload TEXT NOT NULL,UNIQUE(origin,horizon));
        CREATE TABLE outcomes(forecast_id TEXT NOT NULL,truth_hash TEXT NOT NULL,evaluated_at TEXT NOT NULL,
          payload TEXT NOT NULL,PRIMARY KEY(forecast_id,truth_hash));
        ''')
        con.execute('INSERT INTO metadata VALUES (?,?)', ('policy', encode(POLICY)))
        for table in ('metadata', 'decisions', 'forecasts', 'outcomes'):
            for action in ('UPDATE', 'DELETE'):
                con.execute(f"CREATE TRIGGER immutable_{table}_{action} BEFORE {action} ON {table} BEGIN SELECT RAISE(ABORT,'append-only availability ledger'); END")
        con.commit()
    write_manifest(folder)


def write_manifest(folder):
    atomic_json(Path(folder) / 'manifest.json', {'schema': 1, 'policy_sha256': digest(POLICY),
                'database_sha256': file_sha(Path(folder) / 'issued.db')})


def row_sets(folder):
    validate_state(folder)
    with closing(readonly(Path(folder) / 'issued.db')) as con:
        return {t: {tuple(r) for r in con.execute('SELECT * FROM ' + t)}
                for t in ('metadata', 'decisions', 'forecasts', 'outcomes')}


def snapshot(source_folder, target_folder):
    """Export a verified consistent database, without touching a core ledger."""
    source, target = Path(source_folder), Path(target_folder)
    if source.resolve() == target.resolve():
        raise ValueError('snapshot source and target must differ')
    if not source.exists():
        return False
    validate_state(source)
    target.mkdir(parents=True, exist_ok=True)
    # Do not overwrite a later/conflicting durable state with a stale restored copy.
    if (target / 'issued.db').exists():
        old, new = row_sets(target), row_sets(source)
        if any(not old[t] <= new[t] for t in old):
            raise ValueError('availability snapshot would regress durable records')
    tmp = target / 'issued.tmp.db'
    tmp.unlink(missing_ok=True)
    with closing(readonly(source / 'issued.db')) as src, closing(sqlite3.connect(tmp)) as dst:
        src.backup(dst)
    tmp.replace(target / 'issued.db')
    write_manifest(target)
    if (source / 'report.json').exists():
        shutil.copyfile(source / 'report.json', target / 'report.json')
    validate_state(target)
    return True


def restore(root, *, git=subprocess.check_output):
    """Restore only fixed audit paths; never replace a newer local final snapshot."""
    local = Path(root) / STATE
    git(['git', 'rev-parse', '--verify', REF], timeout=15)
    listing = git(['git', 'ls-tree', '-r', '--name-only', REF, '--', GIT_PATH], timeout=15).decode().splitlines()
    if not listing:
        if local.exists():
            validate_state(local)
            # Valid final artifact may hold an accepted run not yet committed to the audit branch.
            return 'local_committed_snapshot_retained'
        return 'first_cohort_initialization'
    required = {GIT_PATH + '/' + name for name in ('issued.db', 'manifest.json')}
    if not required <= set(listing):
        raise ValueError('established audit state incomplete; refuse empty replacement')
    with tempfile.TemporaryDirectory(prefix='availability-restore-') as tmp:
        saved = Path(tmp)
        for name in FILES:
            if GIT_PATH + '/' + name in listing:
                raw = git(['git', 'show', REF + ':' + GIT_PATH + '/' + name], timeout=20)
                (saved / name).write_bytes(raw)
        remote_rows = row_sets(saved)
        if local.exists():
            local_rows = row_sets(local)
            if all(remote_rows[t] <= local_rows[t] for t in local_rows):
                return 'local_snapshot_at_least_as_new_as_audit'
            if any(not local_rows[t] <= remote_rows[t] for t in local_rows):
                raise ValueError('divergent availability histories; manual reconciliation required')
        snapshot(saved, local)
    return 'audit_state_restored'


def run(root, *, clock=clock_now):
    root = Path(root)
    started = utc(clock())
    origin = started.replace(minute=0, second=0, microsecond=0)
    view, downloads = source_view(root, started, origin)
    decision = assess(root, started, view, downloads)
    # A slow read/validation must not produce a backdated origin or late issue.
    issued = utc(clock())
    if issued < started:
        raise ValueError('clock moved backwards')
    if issued >= origin + timedelta(minutes=55):
        decision.update(status='blocked', reason='issuance_deadline_after_validation')
    decision.update(completed_at=issued.isoformat(), run_id=os.environ.get('GITHUB_RUN_ID'),
                    core_publication_outcome=os.environ.get('CORE_PUBLICATION_OUTCOME', 'not_supplied'),
                    public_delivery=False)
    folder = root / STATE
    initialize(folder)
    created, resolved = 0, 0
    settlement_pending = {'immature': 0, 'missing_endpoint': 0, 'invalid_receipt': 0}
    with closing(sqlite3.connect(folder / 'issued.db')) as con:
        con.execute('BEGIN IMMEDIATE')
        if decision['status'] == 'eligible':
            for h in HORIZONS:
                fid = digest({'policy': VERSION, 'origin': origin.isoformat(), 'horizon': h})
                payload = {'forecast_id': fid, 'policy': POLICY, 'origin': origin.isoformat(),
                           'input_cutoff': origin.isoformat(), 'issued_at': issued.isoformat(),
                           'input_available_at': decision['input_available_at'],
                           'input_sha256': decision['input_sha256'], 'trigger_sha256': decision['trigger_sha256'],
                           'window_start': (origin+HOUR).isoformat(), 'target_end': (origin+(h+1)*HOUR).isoformat(),
                           'horizon_hours': h, 'reference_price': decision['reference_price'],
                           'center_price': decision['reference_price'], 'center_log_return': 0.0,
                           'role': 'availability_shadow_only', 'event': None, 'distribution': None,
                           'variance': None, 'public_delivery': False, 'promotion': 'none'}
                changed = con.execute('INSERT OR IGNORE INTO forecasts VALUES (?,?,?,?,?)',
                                      (fid, origin.isoformat(), h, payload['target_end'], encode(payload))).rowcount
                created += changed
        # Center-only truth needs the exact endpoint, not an invented complete path.
        for fid, end, raw in con.execute('SELECT id,target_end,payload FROM forecasts').fetchall():
            end = utc(end)
            if end > started:
                settlement_pending['immature'] += 1
                continue
            truth = view.get(('ETH-USD', end-HOUR))
            if truth is None:
                settlement_pending['missing_endpoint'] += 1
                continue
            try:
                verify_receipts(root, [truth], downloads)
            except (ValueError, OSError):
                settlement_pending['invalid_receipt'] += 1
                continue  # Remains unresolved, never substituted with a nearby or filled candle.
            prediction = json.loads(raw)
            actual, reference = float(truth['close']), prediction['reference_price']
            value = {'actual_price': actual, 'simple_return': actual/reference-1,
                     'absolute_error_usd': abs(actual-reference), 'truth_available_at': truth['observed_at'],
                     'truth_raw_sha256': truth['raw_hash'], 'source_revision': truth['revision'],
                     'evaluated_at': issued.isoformat(), 'head': 'center_only'}
            truth_hash = digest({k: truth[k] for k in ('open_time','observed_at','revision','content_hash','raw_hash')})
            resolved += con.execute('INSERT OR IGNORE INTO outcomes VALUES (?,?,?,?)',
                                    (fid, truth_hash, issued.isoformat(), encode(value))).rowcount
        decision['new_forecasts'] = created
        decision['new_endpoint_outcomes'] = resolved
        decision['settlement_pending'] = settlement_pending
        con.execute('INSERT OR IGNORE INTO decisions VALUES (?,?,?)', (digest(decision), origin.isoformat(), encode(decision)))
        con.commit()
        totals = {t: con.execute('SELECT count(*) FROM ' + t).fetchone()[0] for t in ('decisions','forecasts','outcomes')}
    write_manifest(folder)
    report = {'schema': 1, 'policy': POLICY, 'decision': decision, 'totals': totals,
              'generated_at': issued.isoformat(), 'public_coverage_gain': 'not_measured_shadow_only',
              'predictive_skill': 'not_claimed', 'promotion': 'none'}
    atomic_json(folder / 'report.json', report)
    validate_state(folder)
    return report


def export_attempt(root, target, outcome):
    """Preserve core snapshot even when this optional stage has corrupt/missing state."""
    if outcome not in ('success', 'failure', 'cancelled', 'skipped', ''):
        raise ValueError('invalid workflow outcome')
    root, target = Path(root), Path(target)
    if not target.is_dir():
        raise FileNotFoundError('core final snapshot must exist first')
    run_id = os.environ.get('GITHUB_RUN_ID')
    try:
        status = json.loads((root / 'availability_stage_status.json').read_text())
        if not isinstance(status, dict) or status.get('run_id') != run_id:
            raise ValueError('previous-run report')
    except (OSError, ValueError):
        status = {'status': 'failed' if outcome in ('failure','success') else 'not_run',
                  'error': 'no_current_attempt_report', 'run_id': run_id}
    status.update(step_outcome=outcome or 'not_reached', public_delivery=False, promotion='none')
    error = None
    try:
        status['snapshot_retained'] = snapshot(root / STATE, target / STATE)
        if outcome == 'success' and (status.get('status') != 'success' or not status['snapshot_retained']):
            raise ValueError('successful stage lacks verified current state/status')
    except Exception as exc:
        status['snapshot_error'] = type(exc).__name__ + ': ' + str(exc)[:300]
        error = exc
    finally:
        atomic_json(target / 'availability_stage_status.json', status)
    if error is not None:
        raise error
    return status



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('run','restore','snapshot','export'))
    parser.add_argument('--root', type=Path, default=Path('lake/signals'))
    parser.add_argument('--target', type=Path)
    parser.add_argument('--outcome', default='')
    args = parser.parse_args()
    if args.action == 'restore':
        print(restore(args.root))
    elif args.action == 'export':
        if args.target is None:
            parser.error('--target required for export')
        print(encode(export_attempt(args.root, args.target, args.outcome)))
    elif args.action == 'snapshot':
        if args.target is None:
            parser.error('--target required for snapshot')
        print(snapshot(args.root / STATE, args.target))
    else:
        status = {'run_id': os.environ.get('GITHUB_RUN_ID'), 'status': 'failed', 'promotion': 'none'}
        try:
            result = run(args.root)
            status.update(status='success', decision=result['decision'], totals=result['totals'])
            print(encode(status))
        except Exception as exc:
            status['error'] = type(exc).__name__ + ': ' + str(exc)[:400]
            raise
        finally:
            args.root.mkdir(parents=True, exist_ok=True)
            atomic_json(args.root / 'availability_stage_status.json', status)


if __name__ == '__main__':
    main()
