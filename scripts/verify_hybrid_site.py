"""Verify actual live hybrid metadata, model origin and the replacement chart."""
import argparse
from datetime import datetime,timezone
import json
import time
from urllib.request import urlopen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--expected')
    parser.add_argument('--max-age-hours',type=float,default=33)
    args=parser.parse_args()
    expected=json.load(open(args.expected)) if args.expected else None
    for attempt in range(18 if expected else 1):
        try:
            with urlopen('https://etherforecast.live/hybrid_forecast.json',timeout=10) as response: actual=json.load(response)
            with urlopen('https://etherforecast.live/',timeout=10) as response: html=response.read().decode()
            if expected and actual.get('generated_at')!=expected['generated_at']:raise ValueError('Deployment timestamp mismatch')
            if expected and actual.get('runtime_hash')!=expected['runtime_hash']:raise ValueError('Deployment code fingerprint mismatch')
            if 'id="hybrid-system"' not in html:raise ValueError('Replacement chart not deployed')
            for h in (() if actual.get('retired_from_issuance') else ('7','30')):
                origin=datetime.fromisoformat(actual['horizons'][h]['current']['origin']).replace(tzinfo=timezone.utc)
                if (datetime.now(timezone.utc)-origin).total_seconds()>3600*args.max_age_hours:raise ValueError('Stale hybrid origin')
            print('hybrid deployment verified:',actual['generated_at'],{h:d['matched_origins'] for h,d in actual['horizons'].items()})
            return
        except Exception as exc:
            print('verification pending:',type(exc).__name__,str(exc))
            if expected:time.sleep(8)
    raise SystemExit('Hybrid website did not pass freshness and deployment verification')


if __name__=='__main__':main()
