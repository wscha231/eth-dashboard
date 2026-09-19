"""Record separate core/optional outcomes and retain only current-run diagnostics."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil

NAMES = ('dense_stage_status.json', 'failure_review.json', 'failure_review.md', 'recovery_health.json')


def retain(root, target):
    root, target = Path(root), Path(target)
    if not target.is_dir():
        raise FileNotFoundError('final hourly snapshot required before retaining diagnostics')
    for name in NAMES:
        if (root / name).is_file():
            shutil.copyfile(root / name, target / name)


def record(root, publication, dense, review, dense_persist=""):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    allowed = {'success','failure','skipped','cancelled',''}
    if any(v not in allowed for v in (publication,dense,review,dense_persist)):
        raise ValueError('invalid workflow step outcome')
    # Clear older retained evidence only for a stage which did not run at all.
    if dense in ('skipped','cancelled',''):
        (root / 'dense_stage_status.json').unlink(missing_ok=True)
    if review in ('skipped','cancelled',''):
        for name in ('failure_review.json','failure_review.md'):
            (root / name).unlink(missing_ok=True)
    report = {'schema':1,'run_id':os.environ.get('GITHUB_RUN_ID'),
              'generated_at':datetime.now(timezone.utc).isoformat(),
              'core_publication':publication or 'not_reached',
              'optional_dense':dense or 'not_reached','failure_review':review or 'not_reached',
              'optional_dense_persistence':dense_persist or 'not_reached',
              'core_delivery_succeeded':publication=='success',
              'optional_degraded':'failure' in (dense,review,dense_persist),
              'interpretation':'A failed optional stage does not revoke real core delivery; it still fails the final health gate.',
              'model_promotion':'none'}
    (root/'recovery_health.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('lake/signals'))
    p.add_argument('--publication',default='')
    p.add_argument('--dense',default='')
    p.add_argument('--review',default='')
    p.add_argument('--dense-persist',default='')
    p.add_argument('--retain',type=Path)
    a=p.parse_args()
    if a.retain is not None: retain(a.root,a.retain)
    else: print(json.dumps(record(a.root,a.publication,a.dense,a.review,a.dense_persist)),flush=True)


if __name__=='__main__':main()
