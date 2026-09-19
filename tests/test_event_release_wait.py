from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import subprocess
import os
import unittest
from scripts.wait_event_release import wait_for_release, SITE_URL


class Clock:
    def __init__(self): self.seconds = 0
    def now(self): return datetime(2026,9,19,0,10,tzinfo=timezone.utc)+timedelta(seconds=self.seconds)
    def tick(self): return self.seconds
    def sleep(self, seconds): self.seconds += seconds


class ReleaseWaitTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.expected = {'release_id': 'new', 'current': [{'window_start': '2026-09-19T01:00:00+00:00'}]}
    def wait(self, fetch, **kwargs):
        return wait_for_release(self.expected, fetch=fetch, clock=self.clock.now,
                                monotonic=self.clock.tick, sleep=self.clock.sleep, **kwargs)
    def test_exact_match_requires_later_full_verification(self):
        r = self.wait(lambda *a, **k: (deepcopy(self.expected), {}))
        self.assertEqual(r['status'], 'matched_requires_full_verification')
        self.assertFalse(r['publication_verified'])
        self.assertEqual(len(r['attempts']), 1)
    def test_upstream_match_never_substitutes_for_stale_site(self):
        def fetch(url, timeout):
            return ({'release_id': 'old'} if url.startswith(SITE_URL) else deepcopy(self.expected)), {'age': '300'}
        r = self.wait(fetch, max_wait=16)
        self.assertEqual(r['status'], 'propagation_timeout')
        self.assertEqual(r['diagnosis'], 'expected_upstream_observed_but_site_did_not_match')
        self.assertLessEqual(r['elapsed_seconds'], 16)
    def test_retries_stop_after_matching_public_payload(self):
        def fetch(url, timeout):
            return (deepcopy(self.expected) if self.clock.seconds >= 8 else {'release_id': 'old'}), {}
        r = self.wait(fetch, max_wait=20)
        self.assertEqual(r['status'], 'matched_requires_full_verification')
        self.assertEqual(r['elapsed_seconds'], 8)
    def test_same_id_different_payload_fails(self):
        r = self.wait(lambda *a, **k: ({'release_id': 'new', 'current': []}, {}), max_wait=8)
        self.assertNotEqual(r['status'], 'matched_requires_full_verification')
    def test_deadline_reached_does_not_fetch(self):
        self.clock.seconds = 3000
        def never(*a, **k): self.fail('must not fetch after target start')
        r = self.wait(never)
        self.assertEqual(r['status'], 'forecast_deadline_reached')
    def test_slow_match_after_target_start_rejected(self):
        self.clock.seconds = 2970
        def slow(*a, **k):
            self.clock.seconds += 40
            return deepcopy(self.expected), {}
        r = self.wait(slow)
        self.assertEqual(r['status'], 'forecast_deadline_reached')
    def test_network_failures_are_bounded_and_logged(self):
        def fail(*a, **k): raise TimeoutError('test timeout')
        r = self.wait(fail, max_wait=8)
        self.assertEqual(r['status'], 'propagation_timeout')
        self.assertIn('TimeoutError', r['attempts'][0]['error'])
    def test_unbounded_or_invalid_budget_rejected(self):
        for budget in (-1, 421, float('nan')):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.wait(lambda *a, **k: (self.expected, {}), max_wait=budget)
    def test_shell_audit_runs_on_failure_and_preserves_original_exit_code(self):
        source = Path('scripts/publish_events.sh').read_text()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root/'bin').mkdir(); (root/'lake/signals').mkdir(parents=True)
            fake = root/'bin/python'
            fake.write_text('#!/bin/sh\necho "$*" >> "$TRACE"\ncase "$1" in *publish_event_feed.py) exit 7;; *audit_forecast_continuity.py) exit 0;; esac\nexit 0\n')
            fake.chmod(0o755); script = root/'publish.sh'; script.write_text(source)
            trace = root/'trace'
            result = subprocess.run(['bash', str(script)], cwd=root, env={**os.environ, 'PATH': str(root/'bin')+':'+os.environ['PATH'], 'TRACE': str(trace)}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 7)
            self.assertIn('audit_forecast_continuity.py', trace.read_text())
            self.assertNotIn('verify_event_site.py', trace.read_text())


if __name__ == '__main__': unittest.main()
