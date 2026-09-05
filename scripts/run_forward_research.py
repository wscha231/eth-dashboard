"""Refresh archive features, issue frozen candidates, settle and publish evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from research_pipeline.collect import collect_incremental, read_flow
from research_pipeline.forward import connect, issue_candidates, settle, report
from research_pipeline.protocol import PROTOCOL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', default='lake/gold/eth_master_daily.csv')
    parser.add_argument('--root', default='lake/forward')
    parser.add_argument('--bootstrap', default='research/forward/bootstrap_flow.csv.gz')
    parser.add_argument('--skip-collection', action='store_true', help='Offline verification only; use the saved source status')
    parser.add_argument('--retired', action='store_true', help='Collect sources and settle existing records without issuing retired candidates')
    args = parser.parse_args()
    started = time.monotonic(); root = Path(args.root); root.mkdir(parents=True, exist_ok=True)
    if args.skip_collection:
        flow = read_flow(root/'flow_daily.csv')
        source = json.loads((root/'source_status.json').read_text())
    else:
        print('Collecting bounded UTC archives with checksums', flush=True)
        flow, source = collect_incremental(root, bootstrap=args.bootstrap)
    print(json.dumps({k:v for k,v in source.items() if k != 'errors'}), flush=True)
    master = pd.read_csv(args.master, index_col=0, parse_dates=True)
    hashes = {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.master, str(root/'flow_daily.csv'))}
    conn = connect(root/'forward.db')
    revised = settle(conn, master, source_hash=hashes[args.master])
    try:
        attempt = {'new_issues':0,'skips':[],'status':'retired_source_collection_only'} if args.retired else issue_candidates(conn, master, flow, source, source_hashes=hashes)
    except ValueError as exc:
        attempt = {'new_issues': 0, 'skips': [], 'error': str(exc)}
    attempt['actual_revisions'] = revised
    attempt['total_seconds'] = time.monotonic()-started
    result = report(conn, source, attempt)
    conn.close()
    (root/'forward.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    (root/'protocol.json').write_text(json.dumps(PROTOCOL, indent=2)+'\n')
    print(json.dumps({'attempt':attempt, 'issued':result['issued_count'], 'resolved':result['resolved_count']}), flush=True)


if __name__ == '__main__':
    main()
