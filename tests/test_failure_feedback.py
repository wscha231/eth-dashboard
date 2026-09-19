from copy import deepcopy
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from scripts.review_forecast_failures import feedback_predictions,feedback_report,verified_json,score,nonoverlap,utc


def points(n=140,h=24):
    start=datetime(2023,1,1,tzinfo=timezone.utc)
    return [dict(slot=(start+timedelta(days=i)).isoformat(),target_end=(start+timedelta(days=i,hours=h+1)).isoformat(),
                 horizon_hours=h,q50=.02,**{'return':.01},up=0,down=0,hit_up=.2,hit_down=.2,threshold_up=.5,threshold_down=.5) for i in range(n)]


class FeedbackTests(unittest.TestCase):
    def test_warmup_does_not_fabricate_correction(self):
        out=feedback_predictions(points(30))
        self.assertTrue(all(p['feedback_q50']==p['q50'] for p in out))
    def test_matured_errors_shrink_the_correction(self):
        out=feedback_predictions(points())
        self.assertEqual(out[-1]['feedback_labels'],90)
        self.assertAlmostEqual(out[-1]['feedback_correction'],-.005)
    def test_no_future_label_can_change_earlier_prediction(self):
        original=points(); modified=deepcopy(original)
        for row in modified[100:]: row['return']=-1000
        a,b=feedback_predictions(original),feedback_predictions(modified)
        self.assertEqual([p['feedback_q50'] for p in a[:101]],[p['feedback_q50'] for p in b[:101]])
    def test_exact_maturity_boundary_excluded(self):
        p=points(140,h=720); out=feedback_predictions(p)
        for row in out:
            if row['latest_feedback_maturity']:
                self.assertLess(utc(row['latest_feedback_maturity']),utc(row['slot']))
        # h+1h endpoint plus 1h lag: daily row 30 has no matured label.
        self.assertEqual(out[30]['feedback_labels'],0)
    def test_input_not_mutated_and_no_promotion(self):
        p=points(); original=deepcopy(p); r=feedback_report(p)
        self.assertEqual(p,original); self.assertFalse(r['promotion_allowed'])
        self.assertFalse(r['policy']['promotion_allowed'])
    def test_duplicate_origins_and_mixed_horizons_rejected(self):
        p=points(5)
        with self.assertRaises(ValueError):feedback_predictions(p+[p[0]])
        p[1]['horizon_hours']=6
        with self.assertRaises(ValueError):feedback_predictions(p)
    def test_deterministic_bootstrap(self):
        self.assertEqual(feedback_report(points()),feedback_report(points()))
    def test_score_all_events_not_only_alert_days(self):
        p=points(2);p[0]['up']=1
        r=score(p)
        self.assertEqual(r['up']['events'],1);self.assertEqual(r['up']['missed'],1)
        self.assertEqual(r['up']['recall'],0)
    def test_nonoverlap_not_raw_row_count(self):
        self.assertLess(len(nonoverlap(points(140,h=720))),10)
    def test_missing_history_is_not_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):verified_json(Path(tmp),{'path':'archive/missing.json','bytes':0,'sha256':''})
    def test_tampered_history_blocked(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'archive').mkdir();p=root/'archive/x.json';p.write_text('{}')
            ref={'path':'archive/x.json','bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}
            self.assertEqual(verified_json(root,ref),{})
            p.write_text('[]')
            with self.assertRaisesRegex(ValueError,'integrity'):verified_json(root,ref)
    def test_unsafe_history_paths_rejected(self):
        for path in ('../x.json','archive/../x.json','/tmp/x.json'):
            with self.subTest(path=path),self.assertRaises(ValueError):verified_json(Path('.'),{'path':path})
    def test_naive_timestamp_rejected(self):
        with self.assertRaises(ValueError):utc('2026-09-19T00:00:00')


if __name__=='__main__':unittest.main()
