import json
from pathlib import Path
import tempfile
import unittest
from scripts.recovery_health import record,retain


class Wave2IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.text=Path('.github/workflows/event_hourly.yml').read_text()
    def test_publication_precedes_optional_work(self):
        self.assertLess(self.text.index('id: publication'),self.text.index('id: dense_variance'))
        self.assertLess(self.text.index('id: dense_variance'),self.text.index('id: final_snapshot'))
    def test_optional_failure_does_not_stop_core_or_final_snapshot(self):
        dense=self.text.split('id: dense_variance',1)[1].split('      - name:',1)[0]
        self.assertIn('continue-on-error: true',dense)
        self.assertIn("always() && steps.inference.outcome == 'success'",dense)
        self.assertIn('timeout-minutes: 3',dense)
    def test_only_successful_dense_state_is_persisted(self):
        step=self.text.split('id: dense_persist',1)[1].split('      - name:',1)[0]
        self.assertIn("steps.dense_variance.outcome == 'success'",step)
    def test_failure_is_reported_after_evidence_upload(self):
        self.assertGreater(self.text.index('Fail optional health only'),self.text.index('name: event-final-state'))
        tail=self.text.split('Fail optional health only',1)[1]
        for field in ('dense_variance','dense_persist','failure_review'):
            self.assertIn("steps."+field+".outcome == 'failure'",tail)
        self.assertIn('exit 1',tail)
    def test_shared_lock_and_core_six_horizon_contract_preserved(self):
        self.assertIn('group: daily-forecast',self.text)
        self.assertIn('cancel-in-progress: false',self.text)
        self.assertIn('--horizons 6 24 72 168 336 720',self.text)
        self.assertNotIn('--feedback',self.text)
    def test_new_modules_trigger_integration_and_reports_survive(self):
        for name in ('run_optional_dense.py','review_forecast_failures.py','recovery_health.py'):
            self.assertIn('- scripts/'+name,self.text)
        self.assertIn('python scripts/recovery_health.py --retain final-hourly-state',self.text)
    def test_failed_optional_does_not_revoke_real_delivery(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=record(tmp,'success','failure','success')
            self.assertTrue(r['core_delivery_succeeded']);self.assertTrue(r['optional_degraded'])
    def test_failed_dense_persistence_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=record(tmp,'success','success','success','failure')
            self.assertTrue(r['optional_degraded'])
    def test_old_reports_removed_for_skipped_stage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for name in ('dense_stage_status.json','failure_review.json','failure_review.md'):(root/name).write_text('old')
            record(root,'failure','skipped','skipped')
            self.assertFalse((root/'failure_review.json').exists())
            self.assertFalse((root/'dense_stage_status.json').exists())
    def test_snapshot_copies_reports_not_core_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);src=root/'src';dst=root/'dst';src.mkdir();dst.mkdir()
            record(src,'success','success','success','success')
            (src/'issued.db').write_text('do not copy');(dst/'issued.db').write_text('original snapshot')
            retain(src,dst)
            self.assertEqual((src/'recovery_health.json').read_bytes(),(dst/'recovery_health.json').read_bytes())
            self.assertEqual((dst/'issued.db').read_text(),'original snapshot')
    def test_snapshot_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):retain(tmp,Path(tmp)/'missing')


if __name__=='__main__':unittest.main()
