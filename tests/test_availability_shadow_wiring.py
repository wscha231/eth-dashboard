"""Exercise optional-stage failures and snapshot contracts without touching a live account."""
from pathlib import Path
from contextlib import closing
import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import availability_shadow as a

ROOT = Path(__file__).resolve().parents[1]


class AvailabilityWiringTests(unittest.TestCase):
    def test_after_publication_same_schedule_and_lock(self):
        text=(ROOT/'.github/workflows/event_hourly.yml').read_text()
        self.assertLess(text.index('id: publication'),text.index('id: availability_shadow'))
        self.assertEqual(text.count('schedule:'),1)
        self.assertIn('cron: "8 * * * *"',text)
        self.assertIn('group: daily-forecast',text)
        self.assertIn('cancel-in-progress: false',text)
        self.assertIn('--horizons 6 24 72 168 336 720',text)
    def test_optional_failure_still_runs_on_core_failure_but_cannot_mask_it(self):
        text=(ROOT/'.github/workflows/event_hourly.yml').read_text()
        step=text.split('id: availability_shadow',1)[1].split('      # Optional',1)[0]
        self.assertIn("steps.inference.outcome == 'failure'",step)
        self.assertIn('continue-on-error: true',step)
        self.assertIn('timeout-minutes: 2',step)
        gate=text.split('Fail optional health only',1)[1]
        self.assertIn("steps.availability_shadow.outcome == 'failure'",gate)
        self.assertIn("steps.availability_snapshot.outcome == 'failure'",gate)
        self.assertGreater(text.index('Fail optional health only'),text.index('name: event-final-state'))
    def test_export_failure_cannot_block_core_artifact(self):
        text=(ROOT/'.github/workflows/event_hourly.yml').read_text()
        step=text.split('id: availability_snapshot',1)[1].split('      - uses:',1)[0]
        self.assertIn('continue-on-error: true',step)
        self.assertIn('export --root lake/signals --target final-hourly-state',step)
        upload=text.split('name: event-final-state',1)[0].rsplit('      - uses:',1)[1]
        self.assertNotIn('availability_snapshot.outcome',upload)
    def test_only_own_subtree_staged_and_no_force(self):
        text=(ROOT/'scripts/run_availability_shadow.sh').read_text()
        self.assertIn('add -f lake/event-ledger/availability-shadow/',text)
        self.assertNotIn('push --force',text)
        self.assertNotIn('publish_events.sh',text)
        self.assertNotIn('signals.json',text)
        self.assertIn('diff --cached --quiet',text)
    def test_shell_exit_preserves_failure_and_records_phase(self):
        for fail,code,phase in [('restore',7,'restore'),('run',8,'run'),('snapshot',9,'persist'),('',0,'complete')]:
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);(root/'bin').mkdir();(root/'runner/event-ledger').mkdir(parents=True)
                (root/'lake/signals').mkdir(parents=True)
                (root/'lake/signals/availability_stage_status.json').write_text(json.dumps({'run_id':'old','status':'success'}))
                fake=root/'bin/python'
                fake.write_text(f'#!{sys.executable}\n'+'''import os,sys,json,pathlib
if sys.argv[1]=='-':os.execv(sys.executable,[sys.executable,*sys.argv[1:]])
action=sys.argv[2]
if action==os.environ['INJECT_FAIL']:sys.exit(int(os.environ['INJECT_CODE']))
if action=='run':
 p=pathlib.Path('lake/signals/availability_stage_status.json')
 p.write_text(json.dumps({'run_id':os.environ['GITHUB_RUN_ID'],'status':'success','totals':{'forecasts':0}}))
''')
                fake.chmod(0o755)
                git=root/'bin/git';git.write_text('#!/bin/sh\nexit 0\n');git.chmod(0o755)
                result=subprocess.run(['bash',str(ROOT/'scripts/run_availability_shadow.sh')],cwd=root,
                    env={**os.environ,'PATH':str(root/'bin')+os.pathsep+os.environ['PATH'],
                         'RUNNER_TEMP':str(root/'runner'),'GITHUB_RUN_ID':'new','INJECT_FAIL':fail,'INJECT_CODE':str(code)},
                    capture_output=True,text=True,timeout=10)
                self.assertEqual(result.returncode,code,result.stderr)
                status=json.loads((root/'lake/signals/availability_stage_status.json').read_text())
                self.assertEqual(status['run_id'],'new');self.assertEqual(status['last_phase'],phase)
                self.assertEqual(status['status'],'success' if code==0 else 'failed')
                self.assertFalse(status['public_delivery'])
    def test_killed_stage_cannot_reuse_stale_success(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GITHUB_RUN_ID':'new'}):
            root=Path(tmp);src=root/'source';dst=root/'final';src.mkdir();dst.mkdir()
            (src/'availability_stage_status.json').write_text(json.dumps({'run_id':'old','status':'success'}))
            result=a.export_attempt(src,dst,'failure')
            self.assertEqual(result['run_id'],'new');self.assertEqual(result['status'],'failed')
            self.assertFalse(result['snapshot_retained'])
    def test_corrupt_optional_state_leaves_core_snapshot_and_error_evidence(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GITHUB_RUN_ID':'new'}):
            root=Path(tmp);src=root/'source';dst=root/'final';(src/a.STATE).mkdir(parents=True);dst.mkdir()
            (src/a.STATE/'issued.db').write_bytes(b'corrupt')
            (dst/'issued.db').write_bytes(b'core immutable snapshot')
            with self.assertRaises(sqlite3.DatabaseError):a.export_attempt(src,dst,'failure')
            self.assertEqual((dst/'issued.db').read_bytes(),b'core immutable snapshot')
            self.assertIn('snapshot_error',json.loads((dst/'availability_stage_status.json').read_text()))
    def test_export_success_requires_state_and_current_report(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GITHUB_RUN_ID':'new'}):
            root=Path(tmp);src=root/'source';dst=root/'final';src.mkdir();dst.mkdir()
            with self.assertRaises(ValueError):a.export_attempt(src,dst,'success')
    def test_export_valid_state_preserved_separately(self):
        with tempfile.TemporaryDirectory() as tmp,patch.dict(os.environ,{'GITHUB_RUN_ID':'new'}):
            root=Path(tmp);src=root/'source';dst=root/'final';src.mkdir();dst.mkdir()
            a.initialize(src/a.STATE)
            (src/'availability_stage_status.json').write_text(json.dumps({'run_id':'new','status':'success'}))
            result=a.export_attempt(src,dst,'success')
            self.assertTrue(result['snapshot_retained']);a.validate_state(dst/a.STATE)
            self.assertFalse((dst/'issued.db').exists())
    def test_no_historical_clock_argument_on_operational_cli(self):
        result=subprocess.run([sys.executable,str(ROOT/'scripts/availability_shadow.py'),'run','--as-of','2020-01-01'],
                              capture_output=True,text=True,timeout=5)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('unrecognized arguments',result.stderr)


if __name__=='__main__':unittest.main()
