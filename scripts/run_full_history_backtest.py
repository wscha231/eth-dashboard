"""Run one resumable horizon or combine both into the historical chart payload."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pandas as pd
from backtest_pipeline.protocol import PROTOCOL, PROTOCOL_HASH
from backtest_pipeline.replay import atomic_json, build_report, replay_horizon, runtime_hash
from research_pipeline.collect import read_flow


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master', default='lake/gold/eth_master_daily.csv')
    parser.add_argument('--flow', default='lake/forward/flow_daily.csv')
    parser.add_argument('--output', default='lake/full_backtest')
    parser.add_argument('--horizon', type=int, choices=[7,30])
    parser.add_argument('--combine', action='store_true')
    parser.add_argument('--budget-seconds', type=float, default=1500)
    args = parser.parse_args()
    root = Path(args.output); root.mkdir(parents=True, exist_ok=True)
    hashes = {Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (args.master,args.flow)}
    if args.combine:
        tables, runtimes = [], []
        for h in (7,30):
            meta = json.loads((root/f'runtime_h{h}.json').read_text())
            if meta['source_hashes'] != hashes or meta['runtime_hash'] != runtime_hash() or meta['protocol_hash'] != PROTOCOL_HASH:
                raise ValueError('Refusing to combine different input or code snapshots')
            table = pd.read_csv(root/f'predictions_h{h}.csv.gz',parse_dates=['origin','target'])
            if set(table.horizon) != {h}:
                raise ValueError('Horizon artifact mismatch')
            tables.append(table); runtimes.append(meta)
        started = time.monotonic()
        result, rows = build_report(pd.concat(tables, ignore_index=True), runtimes, source_hashes=hashes)
        result['aggregation_seconds'] = time.monotonic()-started
        atomic_json(root/'full_backtest.json', result)
        rows.to_csv(root/'full_backtest_predictions.csv.gz', index=False, compression={'method':'gzip','mtime':0})
        print(json.dumps({'aggregation_seconds':result['aggregation_seconds'],'horizons':{h:{'best':d['best_fixed_model'],'rows':d['matched_origins'],'first_origin':d['first_origin'],'last_target':d['last_target'],'leaderboard':d['leaderboard']} for h,d in result['horizons'].items()}}),flush=True)
    else:
        if args.horizon is None:
            parser.error('--horizon or --combine is required')
        atomic_json(root/'protocol.json', PROTOCOL)
        master = pd.read_csv(args.master, index_col=0, parse_dates=True)
        flow = read_flow(args.flow)
        table, runtime = replay_horizon(master,flow,args.horizon,root/'cache',budget_seconds=args.budget_seconds)
        runtime['source_hashes'] = hashes
        table.to_csv(root/f'predictions_h{args.horizon}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        atomic_json(root/f'runtime_h{args.horizon}.json',runtime)
        print(json.dumps({k:v for k,v in runtime.items() if k != 'folds'}),flush=True)


if __name__ == '__main__':
    main()
