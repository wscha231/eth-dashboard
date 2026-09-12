"""Audit dependency bounds of an existing actual-data monthly cache, without refits."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_pipeline import historical_study as study
from signal_pipeline.data import read_bars, build_features, utc
from signal_pipeline.models import labels
from signal_pipeline.segments import origin_market_states


def check(root, cutoff, horizon):
    root = Path(root); cutoff = utc(cutoff)
    bars = read_bars(root); features = build_features(bars)
    outcomes = labels(bars, features, horizon); states = origin_market_states(features)
    indices = study.eligible_origins(features, outcomes, cutoff, bars.groupby('product').close_time.max().min())
    last_target = outcomes.loc[indices, 'target_end'].max()
    original_train, original_predict = study.train_candidate, study.predict_candidate
    class Probe(Exception): pass
    def training(*args, **kwargs): raise Probe('training cache miss')
    def prediction(*args, **kwargs): raise Probe('outcome cache miss')
    def evaluate(source):
        return study.evaluate_month(root, source, features, outcomes, states, cutoff, horizon,
                                    root/'historical_study', time.monotonic()+60)
    def revised(mask):
        values = bars.copy(); eligible = values.index[mask]
        if not len(eligible): raise ValueError('cache probe has no source row')
        values.loc[eligible[0], 'content_hash'] = 'revision-probe'
        return values
    try:
        study.train_candidate, study.predict_candidate = training, prediction
        cached = evaluate(bars)
        if cached.get('status') != 'evaluated': raise ValueError('completed evaluated cache required')
        future = evaluate(revised(bars.close_time > last_target))
        if future != cached: raise ValueError('irrelevant future source changed cached result')
        probes = [
            ('within_evaluated_window', (bars.close_time > cutoff) & (bars.close_time <= last_target), 'outcome cache miss'),
            ('before_fit_cutoff', bars.close_time < cutoff, 'training cache miss'),
        ]
        results = {}
        for name, mask, expected in probes:
            try: evaluate(revised(mask))
            except Probe as exc:
                if str(exc) != expected: raise
                results[name] = str(exc)
            else: raise ValueError('source revision reused stale cache: '+name)
        return {'status': 'passed', 'horizon_hours': horizon, 'cutoff': cutoff.isoformat(),
                'last_evaluated_target': last_target.isoformat(), 'evaluation_key': cached['evaluation_key'],
                'identical_source': 'completed result reused without fitting or prediction',
                'after_last_target': 'exact cached result unchanged', **results,
                'scope': 'Source content-hash dependency probes. No source data, fitted models or issued records were modified.'}
    finally:
        study.train_candidate, study.predict_candidate = original_train, original_predict


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='lake/signals'); p.add_argument('--cutoff', default='2024-01-01')
    p.add_argument('--horizon', type=int, default=6); p.add_argument('--output', required=True)
    args = p.parse_args(); result = check(args.root, args.cutoff, args.horizon)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))
