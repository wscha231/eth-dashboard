from copy import deepcopy
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from scripts import build_automation_health as health
from scripts import recovery_health

NOW = datetime(2026, 9, 20, 1, 5, tzinfo=timezone.utc)


class HealthTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.outcomes = dict.fromkeys(health.STAGES, 'success')
        self.data = {
            'operation_health.json': {'run_id':'123','attempt':'1','input_refresh_degraded':False,
                                     'failed_refresh_steps':[],'attribution':{}},
            'recovery_health.json': {'run_id':'123', 'schema':2, 'core_publication':'success'},
            'availability_stage_status.json': {'run_id':'123','status':'success',
                'decision':{'status':'not_needed','reason':'strict_input_condition_satisfied','new_forecasts':0},
                'totals':{'forecasts':0}},
            'dense_stage_status.json': {'run_id':'123','status':'success'},
            'failure_review.json': {'run_id':'123','actual_performance':{str(h):{} for h in health.HORIZONS}},
            'continuity_audit.json': {'as_of':NOW.isoformat(), 'horizons':[
                {'horizon_hours':h,'expected_slots':100,'issued_slots':90,'published_before_window_slots':85,
                 'missing_issue_slots':10,'overdue_settlement_records':0,
                 'recent_24h':{'expected_slots':24,'issued_slots':24,'published_before_window_slots':24}}
                for h in health.HORIZONS]}}
        self.write()
    def write(self):
        for name, d in self.data.items(): (self.root/name).write_text(json.dumps(d))
    def build(self, **kwargs):
        return health.build(self.root,run_id='123',attempt=kwargs.pop('attempt','1'),commit='pinned',
                            outcomes=self.outcomes,now=NOW,**kwargs)
    def test_clean_zero_fallback_is_normal(self):
        r=self.build(); self.assertEqual(r['status'],'ready')
        self.assertEqual(r['availability']['status'],'not_needed')
        self.assertFalse(r['availability']['public_delivery'])
        self.assertFalse(r['public_site_rechecked_by_this_report'])
    def test_availability_failure_cannot_be_hidden_by_legacy_success(self):
        self.outcomes['availability_shadow']='failure'
        r=self.build(); self.assertTrue(r['optional_degraded'])
        self.assertTrue(r['core_publication_step_succeeded'])
        self.assertEqual(r['availability']['status'],'unknown')
        self.assertEqual(r['status'],'attention')
    def test_snapshot_failure_keeps_report_but_not_old_evidence(self):
        self.outcomes['final_snapshot']='failure'
        r=self.build();self.assertFalse(r['snapshot_complete']);self.assertEqual(r['evidence'],{})
    def test_preinference_failure_not_invented_success(self):
        self.outcomes=dict.fromkeys(health.STAGES,'')
        r=self.build();self.assertFalse(r['core_publication_step_succeeded'])
        self.assertEqual(r['status'],'attention');self.assertEqual(r['availability']['status'],'unknown')
    def test_cancelled_optional_is_degraded(self):
        self.outcomes['dense_variance']='cancelled';self.assertTrue(self.build()['optional_degraded'])
    def test_wrong_run_source_not_used(self):
        self.data['availability_stage_status.json']['run_id']='999';self.write()
        self.assertEqual(self.build()['availability']['status'],'unknown')
    def test_retry_does_not_trust_unbound_legacy_receipts(self):
        r=self.build(attempt='2');self.assertEqual(r['availability']['status'],'unknown')
        self.assertTrue(any('retry attempt' in s for s in r['attention_reasons']))
    def test_retry_with_bound_receipts(self):
        for d in self.data.values(): d['attempt']='2'
        self.write();self.assertEqual(self.build(attempt='2')['status'],'ready')
    def test_missing_receipt_is_attention(self):
        (self.root/'dense_stage_status.json').unlink();self.assertEqual(self.build()['status'],'attention')
    def test_explicit_failed_report_beats_green_stage(self):
        self.data['dense_stage_status.json']['status']='failed';self.write()
        self.assertEqual(self.build()['status'],'attention')
    def test_oversized_report_is_not_read_unbounded(self):
        with patch.object(health,'MAX_REPORT_BYTES',4):
            self.assertEqual(self.build()['status'],'attention')
    def test_nonfinite_or_nonobject_json_rejected(self):
        for raw in ('{"x":NaN}', '[]'):
            with self.subTest(raw=raw):
                (self.root/'dense_stage_status.json').write_text(raw)
                self.assertEqual(self.build()['status'],'attention')
    def test_recent_missing_delivery_not_hidden_by_current_success(self):
        self.data['continuity_audit.json']['horizons'][0]['recent_24h']['published_before_window_slots']=16
        self.write();r=self.build();self.assertTrue(r['core_publication_step_succeeded'])
        self.assertIn('recent_24h:unproven_delivery_h6',r['attention_reasons'])
    def test_cumulative_historical_gaps_not_a_new_job_failure(self):
        r=self.build();self.assertEqual(r['status'],'ready');self.assertFalse(r['optional_degraded'])
        self.assertEqual(r['continuity']['horizons'][0]['missing_issue_slots'],10)
    def test_stale_future_and_missing_horizon_block_continuity(self):
        base=deepcopy(self.data)
        for mode in ('stale','future','missing','duplicate'):
            self.data=deepcopy(base)
            if mode in ('stale','future'):
                self.data['continuity_audit.json']['as_of']=(NOW+timedelta(hours=-3 if mode=='stale' else 1)).isoformat()
            elif mode=='missing': self.data['continuity_audit.json']['horizons'].pop()
            else:self.data['continuity_audit.json']['horizons'][0]['horizon_hours']=24
            self.write();self.assertEqual(self.build()['continuity']['status'],'unknown')
    def test_legacy_recovery_publication_disagreement_visible(self):
        self.data['recovery_health.json']['core_publication']='failure';self.write()
        self.assertIn('recovery_health:publication_outcome_disagrees',self.build()['attention_reasons'])
    def test_guard_blocked_is_not_new_skill(self):
        self.data['availability_stage_status.json']['decision']['status']='blocked';self.write()
        r=self.build();self.assertEqual(r['status'],'attention');self.assertEqual(r['model_promotion'],'unchanged_not_evaluated')
    def test_bytes_unchanged_and_no_ledger_creation(self):
        before={p.name:p.read_bytes() for p in self.root.iterdir()};self.build()
        self.assertEqual(before,{p.name:p.read_bytes() for p in self.root.iterdir()})
    def test_outside_symlink_rejected(self):
        with tempfile.TemporaryDirectory() as other:
            p=Path(other)/'report.json';p.write_text('{}')
            q=self.root/'dense_stage_status.json';q.unlink();q.symlink_to(p)
            self.assertEqual(self.build()['status'],'attention')
    def test_invalid_stage_value_rejected(self):
        self.outcomes['publication']='passed'
        with self.assertRaises(ValueError):self.build()
    def test_unknown_or_missing_stage_rejected(self):
        self.outcomes.pop('publication')
        with self.assertRaises(ValueError):self.build()
    def test_legacy_health_now_includes_availability(self):
        with patch.dict(os.environ,{'GITHUB_RUN_ID':'123','GITHUB_RUN_ATTEMPT':'1'}):
            r=recovery_health.record(self.root,'success','success','success','success','failure')
            self.assertTrue(r['optional_degraded']);self.assertEqual(r['optional_availability'],'failure')
            self.assertTrue(r['core_delivery_succeeded']);self.assertEqual(r['schema'],2)
    def test_old_health_call_signature_still_works(self):
        r=recovery_health.record(self.root,'success','success','success')
        self.assertFalse(r['optional_degraded']);self.assertEqual(r['optional_availability'],'not_reached')


class WiringTests(unittest.TestCase):
    def test_health_after_export_before_artifact_and_final_failure_gate(self):
        text=Path('.github/workflows/event_hourly.yml').read_text()
        for left,right in (('id: publication','id: availability_shadow'),
                           ('id: availability_snapshot','id: automation_health'),
                           ('id: automation_health','name: event-final-state'),
                           ('name: event-final-state','Fail optional health only')):
            self.assertLess(text.index(left),text.index(right))
        self.assertIn('--availability "$AVAILABILITY_OUTCOME"',text)
        self.assertIn("steps.automation_health.outcome == 'failure'",text)
        self.assertIn("steps.automation_health_upload.outcome == 'failure'",text)
        self.assertIn('group: daily-forecast',text)
        self.assertIn('cron: "8 * * * *"',text)
    def test_duplicate_archive_completion_subscription_removed_not_collector(self):
        core=Path('.github/workflows/gdrive_archive.yml').read_text()
        r3=Path('.github/workflows/r3_research_drive_archive.yml').read_text()
        for name in ('Free ETH research source backfill','Fast ETH derivative snapshots'):
            self.assertNotIn('"'+name+'"',core)
            self.assertIn('"'+name+'"',r3)
        self.assertIn('19 */6 * * *',core)
        self.assertIn('23,53 * * * *',r3)
        self.assertIn('Hourly ETH event forecast',core)
        self.assertIn('gdrive-archive-${{',core);self.assertIn('gdrive-archive-${{',r3)

if __name__=='__main__':unittest.main()
