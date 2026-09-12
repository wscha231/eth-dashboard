"""Independently reconcile completed study metrics and temporal audit evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from signal_pipeline.data import utc
from signal_pipeline.diagnostics import validate_object
from signal_pipeline.historical_study import FAMILIES, nonoverlap


def audit(root):
    root = Path(root); data = json.loads((root/'historical_study.json').read_text())
    if data['status'] != 'complete': raise ValueError('study is incomplete')
    manifest = validate_object(root, data['archive']); checked = {}; months = 0
    for item in data['monthly_audits']:
        cutoff = utc(item['cutoff']); span = pd.Timedelta(days=365 if item['horizon_hours'] >= 336 else 90)
        for key, boundary in [('training_target_end', cutoff-span), ('validation_target_end', cutoff-span),
                              ('calibration_target_end', cutoff-span/2), ('threshold_target_end', cutoff)]:
            target = item['audit'][key]
            if target is not None and not utc(target) < boundary-pd.Timedelta(hours=1):
                raise ValueError('temporal audit failed: '+key)
        months += 1
    for h, expected in data['horizons'].items():
        frame = pd.read_csv(root/'historical_study'/f'h{h}_rows.csv.gz')
        candidate = frame[frame.model.eq('candidate')].sort_values('slot')
        if frame.duplicated(['model','slot']).any(): raise ValueError('duplicate scored origin')
        if not (pd.to_datetime(frame.target_end, utc=True)-pd.to_datetime(frame.slot, utc=True)).eq(pd.Timedelta(hours=int(h)+1)).all():
            raise ValueError('incorrect target window length')
        if not (pd.to_datetime(frame.target_end, utc=True) <= utc(data['data_as_of'])).all(): raise ValueError('immature target scored')
        for family in FAMILIES:
            values = frame[frame.model.eq(family)].sort_values('slot')
            if values.slot.tolist() != candidate.slot.tolist(): raise ValueError('different paired dates')
            if not np.array_equal(values[['return','up','down','terminal','target_end']].to_numpy(),candidate[['return','up','down','terminal','target_end']].to_numpy()):
                raise ValueError('different paired outcomes')
            errors = np.expm1(values.q50.to_numpy())-np.expm1(values['return'].to_numpy())
            mae = float(np.abs(errors).mean()*100)
            brier = float((((values.hit_up-values.up)**2+(values.hit_down-values.down)**2)/2).mean())
            coverage = float(((values['return'] >= values.q10)&(values['return'] <= values.q90)).mean())
            for key, measured in [('return_mae_pp', mae), ('event_brier', brier), ('coverage80', coverage)]:
                if not np.isclose(measured, expected[family][key], rtol=1e-10, atol=1e-10): raise ValueError('metric reconciliation failed: '+key)
        archive_rows = []
        for entry in manifest['horizons'][h]['shards']:
            archive_rows.extend(validate_object(root, entry)['points'])
        if len(archive_rows) != len(candidate) or len(nonoverlap(archive_rows)) != expected['nonoverlap']:
            raise ValueError('archive denominator mismatch')
        checked[h] = {'origins': len(candidate), 'nonoverlap': expected['nonoverlap'], 'first': expected['first_origin'], 'last': expected['last_origin']}
    return {'status': 'passed', 'data_as_of': data['data_as_of'], 'monthly_audits': months, 'horizons': checked,
            'report_sha256': hashlib.sha256((root/'historical_study.json').read_bytes()).hexdigest(),
            'checks': 'exact target lengths and maturity; partition purge; identical paired truth; independently recomputed MAE/Brier/coverage; archive hashes and counts'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root', default='lake/signals'); args = p.parse_args()
    print(json.dumps(audit(args.root), indent=2))
