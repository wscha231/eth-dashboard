"""Fit one replay segment, combine complete history, or infer from saved models."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from hybrid_pipeline.data import features,runtime_hash
from hybrid_pipeline.engine import replay,settle_table,atomic_json,input_hashes
from hybrid_pipeline.evaluate import report
from hybrid_pipeline.ledger import publish_issued
from hybrid_pipeline.protocol import PROTOCOL,PROTOCOL_HASH
from research_pipeline.collect import read_flow


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--master',default='lake/gold/eth_master_daily.csv')
    parser.add_argument('--flow',default='lake/forward/flow_daily.csv')
    parser.add_argument('--root',default='lake/hybrid')
    parser.add_argument('--horizon',type=int,choices=[7,30])
    parser.add_argument('--part',choices=['early','late'],default='late')
    parser.add_argument('--combine',action='store_true')
    parser.add_argument('--daily',action='store_true')
    parser.add_argument('--budget-seconds',type=float,default=1500)
    args=parser.parse_args();started=time.monotonic();root=Path(args.root);root.mkdir(parents=True,exist_ok=True)
    master=pd.read_csv(args.master,index_col=0,parse_dates=True);flow=read_flow(args.flow)
    hashes=input_hashes(args.master,args.flow)
    if not args.combine and not args.daily:
        if args.horizon is None:parser.error('--horizon is required for a replay part')
        table,meta=replay(master,flow,args.horizon,root,part=args.part,budget_seconds=args.budget_seconds)
        meta['source_hashes']=hashes
        table.to_csv(root/f'base_h{args.horizon}_{args.part}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
        atomic_json(root/f'runtime_h{args.horizon}_{args.part}.json',meta)
        print(json.dumps({k:v for k,v in meta.items() if k!='folds'}),flush=True)
        return
    raw,_=features(master,flow);tables=[];runtimes=[]
    if args.daily:
        previous=json.loads((root/'hybrid_forecast.json').read_text())
        if previous['runtime_hash']!=runtime_hash() or previous['protocol_hash']!=PROTOCOL_HASH:
            raise ValueError('Run a full replay before publishing a new protocol/code version')
        old=pd.read_csv(root/'base_predictions.csv.gz',parse_dates=['origin','target'])
        for h in (7,30):
            current,meta=replay(master,flow,h,root,part='current',budget_seconds=args.budget_seconds)
            meta['source_hashes']=hashes;runtimes.append(meta)
            tables.append(pd.concat([old[(old.horizon==h)&(old.origin<current.origin.min())],current],ignore_index=True))
    else:
        for h in (7,30):
            for part in ('early','late'):
                meta=json.loads((root/f'runtime_h{h}_{part}.json').read_text())
                if meta['source_hashes']!=hashes or meta['runtime_hash']!=runtime_hash() or meta['protocol_hash']!=PROTOCOL_HASH:
                    raise ValueError('All replay parts must share the exact source and code snapshot')
                table=pd.read_csv(root/f'base_h{h}_{part}.csv.gz',parse_dates=['origin','target'])
                if set(table.horizon)!={h}:raise ValueError('Horizon artifact mismatch')
                tables.append(table);runtimes.append(meta)
    base=settle_table(pd.concat(tables,ignore_index=True),raw)
    legacy_path=Path('lake/full_backtest/full_backtest_predictions.csv.gz')
    legacy=pd.read_csv(legacy_path,parse_dates=['origin','target']) if legacy_path.exists() else None
    aggregation=time.monotonic();payload,paths=report(base,raw,runtimes,hashes,legacy=legacy)
    payload['aggregation_seconds']=time.monotonic()-aggregation
    payload=publish_issued(payload,master,root/'issued.db')
    payload['runtime_seconds']=time.monotonic()-started
    if args.daily:
        # Keep the original full replay's timings/date separate from daily settlement.
        payload['last_full_replay']=previous.get('last_full_replay',{'generated_at':previous['generated_at'],'runtimes':previous['runtimes']})
    else:
        payload['last_full_replay']={'generated_at':payload['generated_at'],'runtimes':runtimes}
    atomic_json(root/'hybrid_forecast.json',payload)
    atomic_json(root/'protocol.json',PROTOCOL)
    base.to_csv(root/'base_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    paths.to_csv(root/'hybrid_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    print(json.dumps({'generated_at':payload['generated_at'],'runtime_seconds':payload['runtime_seconds'],'aggregation_seconds':payload['aggregation_seconds'],
                      'horizons':{h:{'origins':d['matched_origins'],'first':d['first_origin'],'last_target':d['last_target'],
                                     'best_retrospective':d['best_fixed_retrospective'],'leaderboard':d['leaderboard'],'current':d['current']} for h,d in payload['horizons'].items()},
                      'prospective':{k:v for k,v in payload['prospective'].items() if k!='recent_issued'}}),flush=True)


if __name__=='__main__':main()
