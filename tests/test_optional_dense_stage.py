import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import unittest
from scripts.run_optional_dense import run, snapshot_database, sha, restore_git
from unittest.mock import patch


def database(path, dense=False):
    with sqlite3.connect(path) as db:
        if dense:
            db.executescript('CREATE TABLE forecasts(id TEXT,horizon_hours INTEGER,payload TEXT); CREATE TABLE outcomes(id TEXT,payload TEXT);')
            db.execute("INSERT INTO forecasts VALUES ('old',24,'immutable')")
            db.execute("INSERT INTO outcomes VALUES ('old','truth')")
        else: db.execute('CREATE TABLE bars(value TEXT)')


class OptionalDenseTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'live'; self.root.mkdir()
        database(self.root/'observations.db')
        self.core=self.root/'issued.db'; self.core.write_bytes(b'core ledger must not change')
        self.core_sha=sha(self.core)
        (self.root/'variance_shadow_dense').mkdir()
        (self.root/'variance_shadow_dense/prior_local').write_text('preserve on failure')
    def restore(self, stage):
        (stage/'variance_shadow_dense').mkdir()
        database(stage/'variance_shadow_dense/issued.db',dense=True)
        snapshot_database(stage/'variance_shadow_dense/issued.db',stage/'prior_dense.db')
    def execute(self,stage):
        d=stage/'variance_shadow_dense'
        with sqlite3.connect(d/'issued.db') as db: db.execute("INSERT INTO forecasts VALUES ('new',72,'valid')")
        (d/'report.json').write_text(json.dumps({'errors':[],'generated_at':'2026-09-19T05:00:00Z'}))
        (d/'public.json').write_text(json.dumps({'horizons':{'24':{},'72':{}},'generated_at':'2026-09-19T05:00:00Z'}))
    def check_failed(self,restore=None,execute=None):
        r=run(self.root,restore=restore or self.restore,execute=execute or self.execute)
        self.assertEqual(r['status'],'failed'); self.assertFalse(r['committed_output'])
        self.assertTrue((self.root/'variance_shadow_dense/prior_local').exists())
        self.assertEqual(sha(self.core),self.core_sha)
        self.assertTrue((self.root/'dense_stage_status.json').exists())
        self.assertTrue(r['errors'])
    def test_success_commits_complete_output_and_keeps_core(self):
        r=run(self.root,restore=self.restore,execute=self.execute)
        self.assertEqual(r['status'],'success'); self.assertTrue(r['committed_output'])
        self.assertEqual(sha(self.core),self.core_sha)
        self.assertEqual(set(r['output_sha256']),{'issued.db','report.json','public.json'})
    def test_missing_checkpoint_restore_does_not_create_fresh_cohort(self):
        def fail(stage): raise FileNotFoundError('checkpoint not available')
        self.check_failed(restore=fail)
    def test_timeout_isolated(self):
        def fail(stage): raise subprocess.TimeoutExpired('dense',120)
        self.check_failed(execute=fail)
    def test_partial_write_is_never_committed(self):
        def fail(stage):
            (stage/'variance_shadow_dense/issued.db').write_bytes(b'bad partial db')
            raise RuntimeError('injected failure')
        self.check_failed(execute=fail)
    def test_zero_exit_with_per_horizon_errors_is_failure(self):
        def invalid(stage):
            self.execute(stage)
            (stage/'variance_shadow_dense/report.json').write_text(json.dumps({'errors':[{'h':24,'reason':'stale'}]}))
        self.check_failed(execute=invalid)
    def test_prior_record_deletion_rejected(self):
        def invalid(stage):
            self.execute(stage)
            with sqlite3.connect(stage/'variance_shadow_dense/issued.db') as db: db.execute("DELETE FROM forecasts WHERE id='old'")
        self.check_failed(execute=invalid)
    def test_prior_outcome_mutation_rejected(self):
        def invalid(stage):
            self.execute(stage)
            with sqlite3.connect(stage/'variance_shadow_dense/issued.db') as db: db.execute("UPDATE outcomes SET payload='changed'")
        self.check_failed(execute=invalid)
    def test_unauthorized_six_hour_cohort_rejected(self):
        def invalid(stage):
            self.execute(stage)
            with sqlite3.connect(stage/'variance_shadow_dense/issued.db') as db: db.execute("INSERT INTO forecasts VALUES ('bad',6,'x')")
        self.check_failed(execute=invalid)
    def test_mixed_public_generation_rejected(self):
        def invalid(stage):
            self.execute(stage)
            (stage/'variance_shadow_dense/public.json').write_text(json.dumps({'horizons':{'24':{},'72':{}},'generated_at':'old'}))
        self.check_failed(execute=invalid)
    def test_missing_input_fails_without_empty_input_creation(self):
        (self.root/'observations.db').unlink()
        self.check_failed()
        self.assertFalse((self.root/'observations.db').exists())
    def test_unsafe_checkpoint_filename_rejected(self):
        raw=json.dumps({'models':{'24':{'file':'../escape.json'}}}).encode()
        with tempfile.TemporaryDirectory() as tmp, patch('scripts.run_optional_dense.git_bytes',return_value=raw):
            with self.assertRaisesRegex(ValueError,'unsafe'): restore_git(Path(tmp))


if __name__=='__main__':unittest.main()
