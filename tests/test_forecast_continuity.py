import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from scripts.audit_forecast_continuity import audit, due_slot

BASE = datetime(2026, 9, 1, tzinfo=timezone.utc)


class ContinuityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        with sqlite3.connect(self.root / 'issued.db') as db:
            db.executescript('''CREATE TABLE forecasts (forecast_id TEXT, slot TEXT, horizon_seconds INTEGER, model_version TEXT, target_definition TEXT, issued_at TEXT, target_end TEXT, payload TEXT);
                CREATE TABLE deliveries (forecast_id TEXT, verified_at TEXT, release_id TEXT);
                CREATE TABLE outcomes (forecast_id TEXT, evaluated_at TEXT, payload TEXT);''')

    def add(self, hour, h=6, version='v1', delivery_minutes=12, issue_minutes=10, settle=False, early_truth=False):
        slot = BASE + timedelta(hours=hour)
        start, end = slot + timedelta(hours=1), slot + timedelta(hours=h + 1)
        issued = slot + timedelta(minutes=issue_minutes)
        fid = f'{hour}-{h}-{version}'
        record = {'forecast_id': fid, 'slot': slot.isoformat(), 'input_cutoff': slot.isoformat(),
                  'available_at': (slot + timedelta(minutes=9)).isoformat(), 'issued_at': issued.isoformat(),
                  'window_start': start.isoformat(), 'target_end': end.isoformat(), 'horizon_seconds': h*3600,
                  'selected_model': 'climatology', 'price_quantiles': [90, 101, 110], 'reference_price': 100}
        with sqlite3.connect(self.root / 'issued.db') as db:
            db.execute('INSERT INTO forecasts VALUES (?,?,?,?,?,?,?,?)', (fid, slot.isoformat(), h*3600, version, 'target-v1', issued.isoformat(), end.isoformat(), json.dumps(record)))
            if delivery_minutes is not None:
                db.execute('INSERT INTO deliveries VALUES (?,?,?)', (fid, (slot+timedelta(minutes=delivery_minutes)).isoformat(), 'release'))
            if settle:
                truth_at = end + timedelta(minutes=-1 if early_truth else 5)
                db.execute('INSERT INTO outcomes VALUES (?,?,?)', (fid, (end+timedelta(minutes=6)).isoformat(), json.dumps({'truth_available_at': truth_at.isoformat()})))

    def run_audit(self, hour=12, through=2):
        return audit(self.root, as_of=BASE+timedelta(hours=hour, minutes=30), from_slot=BASE, through_slot=BASE+timedelta(hours=through))

    def test_gaps_late_delivery_and_read_only(self):
        self.add(0, settle=True)
        self.add(2, delivery_minutes=65)
        before = hashlib.sha256((self.root/'issued.db').read_bytes()).hexdigest()
        result = self.run_audit()
        h = result['horizons'][0]
        self.assertEqual((h['expected_slots'], h['issued_slots'], h['missing_issue_slots']), (3, 2, 1))
        self.assertEqual(h['published_before_window_slots'], 1)
        self.assertEqual(h['late_or_unverified_closed_slots'], 1)
        self.assertEqual(h['overdue_settlement_records'], 1)
        self.assertEqual(before, hashlib.sha256((self.root/'issued.db').read_bytes()).hexdigest())
        self.assertEqual(result['historical']['status'], 'missing')

    def test_multiple_versions_do_not_manufacture_service_coverage(self):
        self.add(0); self.add(0, version='v2')
        h = self.run_audit(through=0)['horizons'][0]
        self.assertEqual(h['stored_records'], 2)
        self.assertEqual(h['issued_slots'], 1)
        self.assertEqual(h['multiple_version_slots'], 1)
        self.assertEqual(h['published_before_window_slots'], 1)

    def test_future_forecasts_and_receipts_excluded_from_asof(self):
        self.add(0, delivery_minutes=65); self.add(2)
        h = self.run_audit(hour=0, through=0)['horizons'][0]
        self.assertEqual(h['stored_records'], 1)
        self.assertEqual(h['published_before_window_slots'], 0)
        self.assertEqual(h['delivery_still_pending_slots'], 1)

    def test_early_truth_not_counted_as_valid_settlement(self):
        self.add(0, settle=True, early_truth=True)
        result = self.run_audit(through=0)
        self.assertEqual(result['horizons'][0]['overdue_settlement_records'], 1)
        self.assertTrue(result['errors'])

    def test_long_horizon_pending_is_not_settlement_failure(self):
        self.add(0, h=720)
        h = self.run_audit(through=0)['horizons'][-1]
        self.assertEqual(h['matured_records'], 0)
        self.assertEqual(h['overdue_settlement_records'], 0)

    def test_invalid_issue_time_not_counted_as_timely(self):
        self.add(0, issue_minutes=56, delivery_minutes=58)
        result = self.run_audit(through=0)
        self.assertEqual(result['horizons'][0]['timing_valid_slots'], 0)
        self.assertEqual(result['horizons'][0]['published_before_window_slots'], 0)
        self.assertTrue(result['errors'])

    def test_constant_baseline_is_reported_without_claiming_alpha(self):
        self.add(0); self.add(1)
        result = self.run_audit(through=1)
        self.assertAlmostEqual(result['horizons'][0]['constant_median_return'], .01)
        self.assertEqual(result['promotion'], 'not_evaluated_no_model_changes')

    def test_no_empty_ledger_is_created(self):
        path = self.root/'missing'
        with self.assertRaises(FileNotFoundError): audit(path, as_of=BASE)
        self.assertFalse(path.exists())

    def test_naive_clock_and_not_due_slot_rejected(self):
        self.add(0)
        with self.assertRaises(ValueError): audit(self.root, as_of=datetime(2026,9,1))
        with self.assertRaises(ValueError): audit(self.root, as_of=BASE+timedelta(minutes=10), from_slot=BASE, through_slot=BASE)
        self.assertEqual(due_slot(BASE+timedelta(minutes=10)), BASE-timedelta(hours=1))

    def test_replay_remains_separate_and_missing_horizons_not_hidden(self):
        self.add(0)
        (self.root/'replay.json').write_text(json.dumps({'generated_at': BASE.isoformat(), 'horizons': {'720': {'common_origins': 2000}}}))
        result = self.run_audit(through=0)
        self.assertEqual(result['historical']['horizons']['720']['common_origins'], 2000)
        self.assertEqual(result['horizons'][-1]['issued_slots'], 0)
        self.assertEqual(result['horizons'][-1]['missing_issue_slots'], 1)


if __name__ == '__main__': unittest.main()
