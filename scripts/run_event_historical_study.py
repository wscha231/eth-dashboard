#!/usr/bin/env python3
"""Evaluate all six horizons from earlier years, resuming saved monthly work."""
import argparse
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_pipeline.historical_study import run_study
from signal_pipeline.protocol import HORIZONS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='lake/signals')
    p.add_argument('--start', default='2020-01-01')
    p.add_argument('--end')
    p.add_argument('--horizons', nargs='+', type=int, default=list(HORIZONS))
    p.add_argument('--budget-seconds', type=int, default=1500)
    p.add_argument('--resume-budget', action='store_true', help='Treat only budget exhaustion as resumable work')
    args = p.parse_args()
    try:
        result = run_study(args.root, start=args.start, end=args.end, horizons=tuple(args.horizons), budget_seconds=args.budget_seconds)
    except TimeoutError:
        if not args.resume_budget:
            raise
        if os.environ.get('GITHUB_OUTPUT'):
            with open(os.environ['GITHUB_OUTPUT'], 'a') as f: f.write('complete=false\n')
        print('Historical study needs resume; completed months retained. No complete report published.')
        return
    if os.environ.get('GITHUB_OUTPUT'):
        with open(os.environ['GITHUB_OUTPUT'], 'a') as f: f.write('complete=true\n')
    print(json.dumps({'status': result['status'], 'data_as_of': result['data_as_of'], 'horizons': {
        h: {k: v.get(k) for k in ('origins', 'nonoverlap', 'first_origin', 'last_origin', 'head_evidence', 'decision')}
        for h, v in result['horizons'].items()}}, indent=2))


if __name__ == '__main__':
    main()
