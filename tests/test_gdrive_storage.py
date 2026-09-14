import copy
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import stat
import tempfile
import threading
import unittest
from unittest.mock import patch
import zipfile

from scripts import gdrive_store as storage
from scripts.archive_to_drive import extract_zip, trusted, archive_artifact, automation_status
from scripts.prepare_daily_drive_snapshot import prepare


class MemoryDrive:
    def __init__(self):
        self.index, self.data, self.writes = {}, {}, 0
        self.fail_after = None

    def put(self, name, data, properties):
        if name in self.data:
            if self.data[name] != data:
                raise storage.CheckError('immutable conflict')
            return False
        if self.fail_after is not None and self.writes >= self.fail_after:
            raise storage.CheckError('interrupted upload')
        self.data[name] = data
        self.index[name] = {'id': 'fake-' + str(self.writes), 'name': name, 'md5Checksum': hashlib.md5(data).hexdigest(),
                            'size': str(len(data)), 'appProperties': properties}
        self.writes += 1
        return True

    def get(self, name, **kwargs):
        if name not in self.data:
            raise storage.CheckError('missing object')
        return self.data[name]


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.drive = MemoryDrive()
        self.store = storage.Store(self.drive)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, key='run1', at='2026-09-14T01:00:00Z', stream='event-hourly'):
        return self.store.backup(self.source, stream, key, at, {'run_id': 123})

    def test_roundtrip_deduplication_and_immutable_history(self):
        path = self.source / 'signals.json'
        path.write_text('{"forecast":1}')
        first = self.save()
        self.assertEqual(first['uploaded_chunks'], 1)
        second = self.save('run2', '2026-09-14T02:00:00Z')
        self.assertEqual(second['uploaded_chunks'], 0)
        self.assertEqual(second['reused_chunks'], 1)
        self.assertTrue(self.save('run2')['already_saved'])
        path.write_text('{"forecast":2}')
        self.save('run3', '2026-09-14T03:00:00Z')
        latest, prior = self.root / 'latest', self.root / 'prior'
        self.store.restore('event-hourly', latest)
        self.store.restore('event-hourly', prior, as_of='2026-09-14T01:30:00Z')
        self.assertEqual((latest / 'signals.json').read_text(), '{"forecast":2}')
        self.assertEqual((prior / 'signals.json').read_text(), '{"forecast":1}')

    def test_range_ledger_and_seed_versions_survive_drive_restore(self):
        root = self.root / 'operational'; root.mkdir()
        (root / 'active.json').write_text('{}')
        (root / 'range_shadow').mkdir()
        with sqlite3.connect(root / 'range_shadow/issued.db') as con:
            con.execute('CREATE TABLE audit (forecast_id TEXT)')
            con.execute("INSERT INTO audit VALUES ('immutable-range-id')")
        (root / 'range_seeds').mkdir()
        (root / 'range_seeds/abc.json').write_text('{"seed_id":"abc"}')
        (root / 'range_seed.json').write_text('{"seed_id":"abc"}')
        self.source = root
        self.save()
        target = self.root / 'restored-range'
        self.store.restore('event-hourly', target)
        with sqlite3.connect(target / 'range_shadow/issued.db') as con:
            self.assertEqual(con.execute('SELECT forecast_id FROM audit').fetchone()[0], 'immutable-range-id')
            self.assertEqual(con.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertEqual((target / 'range_seeds/abc.json').read_bytes(), (root / 'range_seed.json').read_bytes())

    def test_automation_receipt_identifies_exact_status_and_separate_streams(self):
        (self.source / 'data.json').write_text('{"raw_private_payload": 1}')
        self.save(stream='event-hourly')
        self.save(stream='event-research')
        output = self.root / 'status'
        with patch.dict('os.environ', {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '2', 'SOURCE_RUN': '100'}):
            receipt = automation_status(self.store, {'status': 'failure', 'errors': [{'source': 'partial'}]},
                                        {'restored': True, 'stream': 'event-hourly'}, output)
        data = (output / 'status.json').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), receipt['sha256'])
        self.assertEqual(self.drive.get(receipt['name']), data)
        status = json.loads(data)
        self.assertEqual(status['status'], 'failure')
        self.assertEqual(status['source_run'], '100')
        self.assertEqual(status['attempt'], '2')
        self.assertTrue(status['recovery_drill']['restored'])
        self.assertEqual(set(status['snapshots']), {'event-hourly', 'event-research'})
        self.assertNotIn(b'raw_private_payload', data)

    def test_automation_failure_can_be_read_without_drive(self):
        output = self.root / 'status'
        with patch.dict('os.environ', {'GITHUB_RUN_ID': '123', 'GITHUB_RUN_ATTEMPT': '1'}):
            receipt = automation_status(None, {'status': 'failure', 'errors': [{'source': 'archive'}]}, None, output)
        self.assertIsNone(receipt)
        self.assertFalse((output / 'receipt.json').exists())
        self.assertEqual(json.loads((output / 'status.json').read_text())['status'], 'failure')

    def test_partial_upload_has_no_manifest_and_can_resume(self):
        (self.source / 'a.json').write_text('a')
        (self.source / 'b.json').write_text('b')
        self.drive.fail_after = 1
        with self.assertRaises(storage.CheckError):
            self.save()
        self.assertIsNone(self.store.latest('event-hourly'))
        self.drive.fail_after = None
        result = self.save()
        self.assertEqual(result['reused_chunks'], 1)
        self.store.restore('event-hourly', self.root / 'restored')

    def test_parallel_uploads_share_identical_chunks_and_propagate_failure(self):
        class ConcurrentDrive(MemoryDrive):
            def __init__(self, fail=False):
                super().__init__()
                self.lock, self.ready = threading.Lock(), threading.Event()
                self.active, self.peak, self.calls = 0, 0, {}
                self.fail = fail

            def put(self, name, data, properties):
                if not name.startswith('ef1-blob-'):
                    return super().put(name, data, properties)
                with self.lock:
                    self.active += 1
                    self.peak = max(self.peak, self.active)
                    self.calls[name] = self.calls.get(name, 0) + 1
                    if self.active == 4:
                        self.ready.set()
                try:
                    if not self.ready.wait(3):
                        raise AssertionError('Uploads did not overlap')
                    with self.lock:
                        if self.fail:
                            self.fail = False
                            raise storage.CheckError('worker upload failed')
                        return super().put(name, data, properties)
                finally:
                    with self.lock:
                        self.active -= 1

        for i in range(12):
            (self.source / f'{i:02d}.json').write_text(str(i // 2))
        for fail in (False, True):
            with self.subTest(fail=fail):
                drive = ConcurrentDrive(fail)
                store = storage.Store(drive)
                if fail:
                    with self.assertRaisesRegex(storage.CheckError, 'worker upload failed'):
                        store.backup(self.source, 'event-hourly', 'parallel', '2026-09-14T01:00:00Z', {})
                    self.assertIsNone(store.latest('event-hourly'))
                else:
                    result = store.backup(self.source, 'event-hourly', 'parallel', '2026-09-14T01:00:00Z', {})
                    self.assertEqual(result['uploaded_chunks'], 6)
                    self.assertEqual(result['reused_chunks'], 6)
                    store.restore('event-hourly', self.root / 'parallel')
                    for path in self.source.iterdir():
                        self.assertEqual(path.read_bytes(), (self.root / 'parallel' / path.name).read_bytes())
                self.assertEqual(drive.peak, 4)
                self.assertTrue(all(n == 1 for n in drive.calls.values()))

    def test_sqlite_wal_is_preserved(self):
        path = self.source / 'issued.db'
        with sqlite3.connect(path) as con:
            con.execute('PRAGMA journal_mode=WAL')
            con.execute('CREATE TABLE forecasts(id INTEGER PRIMARY KEY, payload TEXT)')
            con.execute("INSERT INTO forecasts VALUES (1,'original')")
            con.commit()
            self.assertTrue(path.with_name('issued.db-wal').exists())
            self.save()
        target = self.root / 'restored'
        self.store.restore('event-hourly', target)
        self.assertFalse((target / 'issued.db-wal').exists())
        with sqlite3.connect(target / 'issued.db') as con:
            self.assertEqual(con.execute('SELECT payload FROM forecasts').fetchone()[0], 'original')

    def test_chunks_are_reused_after_append(self):
        with patch.object(storage, 'CHUNK', 64):
            path = self.source / 'rows.csv'
            path.write_bytes(b'a' * 64 + b'b' * 64)
            self.save()
            path.write_bytes(path.read_bytes() + b'c' * 20)
            second = self.save('new')
            self.assertEqual(second['uploaded_chunks'], 1)
            self.assertEqual(second['reused_chunks'], 2)
            self.store.restore('event-hourly', self.root / 'restored')
            self.assertEqual((self.root / 'restored/rows.csv').read_bytes(), path.read_bytes())

    def test_corruption_does_not_install_partial_restore(self):
        (self.source / 'data.json').write_text('preserve')
        self.save()
        name = next(n for n in self.drive.data if n.startswith('ef1-blob-'))
        self.drive.data[name] = b'corrupt'
        target = self.root / 'restored'
        with self.assertRaises(storage.CheckError):
            self.store.restore('event-hourly', target)
        self.assertFalse(target.exists())

    def test_existing_data_is_never_overwritten(self):
        (self.source / 'data.json').write_text('new')
        self.save()
        target = self.root / 'existing'
        target.mkdir()
        (target / 'issued.db').write_text('original')
        with self.assertRaises(storage.CheckError):
            self.store.restore('event-hourly', target)
        self.assertEqual((target / 'issued.db').read_text(), 'original')

    def test_manifest_path_traversal_and_stream_substitution_rejected(self):
        (self.source / 'data.json').write_text('safe')
        result = self.save()
        original = json.loads(self.drive.data[result['snapshot']])
        for key in ('../outside.json', '/tmp/outside.json', 'scripts/attack.py'):
            modified = copy.deepcopy(original)
            modified['files'][key] = modified['files'].pop('data.json')
            self.drive.data[result['snapshot']] = storage.canonical(modified)
            with self.assertRaises(storage.CheckError):
                self.store.restore('event-hourly', self.root / 'restore')
        modified = copy.deepcopy(original)
        modified['stream'] = 'event-research'
        self.drive.data[result['snapshot']] = storage.canonical(modified)
        with self.assertRaises(storage.CheckError):
            self.store.restore('event-hourly', self.root / 'restore')

    def test_secret_files_and_executable_files_are_excluded(self):
        (self.source / 'model.joblib').write_bytes(b'model')
        for name in ('token.json', 'client_secret.txt', 'rclone.conf', 'run.py'):
            (self.source / name).write_text('fake-sensitive')
        result = self.save()
        self.assertEqual(result['files'], 1)
        raw = str(self.drive.data)
        self.assertNotIn('fake-sensitive', raw)

    def test_symlink_source_and_destination_rejected(self):
        (self.source / 'outside.json').symlink_to(self.root / 'missing')
        with self.assertRaises(storage.CheckError):
            self.save()
        (self.source / 'outside.json').unlink()
        (self.source / 'data.json').write_text('x')
        self.save()
        target = self.root / 'link'
        target.symlink_to(self.root / 'elsewhere')
        with self.assertRaises(storage.CheckError):
            self.store.restore('event-hourly', target)

    def test_retrospective_snapshots_never_selected_for_live_stream(self):
        (self.source / 'issued.db.json').write_text('research')
        self.save(stream='event-research')
        self.assertIsNone(self.store.latest('event-hourly'))
        result = self.store.restore('event-hourly', self.root / 'none', required=False)
        self.assertFalse(result['restored'])

    def test_size_and_decompression_limits(self):
        (self.source / 'data.json').write_text('x' * 100)
        result = self.save()
        manifest = json.loads(self.drive.data[result['snapshot']])
        manifest['files']['data.json']['chunks'][0]['raw_bytes'] = 1
        self.drive.data[result['snapshot']] = storage.canonical(manifest)
        with self.assertRaises(storage.CheckError):
            self.store.restore('event-hourly', self.root / 'bad')

    def test_daily_preparation_preserves_db_and_excludes_website_code(self):
        (self.source / 'lake/gold').mkdir(parents=True)
        (self.source / 'lake/gold/eth_master_daily.csv').write_text('date,price\n2026-09-14,1\n')
        (self.source / 'forecast_site/public').mkdir(parents=True)
        (self.source / 'forecast_site/public/index.html').write_text('code')
        (self.source / 'forecast_site/public/history.json').write_text('[]')
        with sqlite3.connect(self.source / 'forecast_site/predictions.db') as con:
            con.execute('CREATE TABLE predictions(id TEXT)')
        target = self.root / 'daily'
        prepare(self.source, target)
        self.assertTrue((target / 'forecast_site/predictions.db').exists())
        self.assertFalse((target / 'forecast_site/public/index.html').exists())


class SourceTests(unittest.TestCase):
    def test_untrusted_runs_rejected(self):
        valid = {'head_repository': {'id': 1}, 'repository': {'id': 1}, 'head_branch': 'main',
                 'status': 'completed', 'event': 'schedule', 'path': '.github/workflows/event_hourly.yml'}
        self.assertTrue(trusted(valid, 'event_hourly.yml', 1))
        for update in ({'event': 'pull_request'}, {'head_branch': 'feature'}, {'status': 'in_progress'},
                       {'head_repository': {'id': 2}}, {'path': '.github/workflows/unknown.yml'}):
            self.assertFalse(trusted(dict(valid, **update), 'event_hourly.yml', 1))

    def test_archive_extraction_never_writes_outside_staging(self):
        for name in ('../bad.json', '/tmp/bad.json', 'x/../../bad.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                zip_path = Path(temporary) / 'bad.zip'
                with zipfile.ZipFile(zip_path, 'w') as z:
                    z.writestr(name, 'unsafe')
                with self.assertRaises(storage.CheckError):
                    extract_zip(zip_path, Path(temporary) / 'extract')

    def test_zip_symlink_and_duplicate_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'link.zip'
            with zipfile.ZipFile(path, 'w') as z:
                info = zipfile.ZipInfo('link.json')
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, '../outside')
            with self.assertRaises(storage.CheckError):
                extract_zip(path, Path(temporary) / 'out')


if __name__ == '__main__':
    unittest.main()
