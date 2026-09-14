"""Export past losses from a complete trusted paired study for future-only reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from signal_pipeline.adaptive_research import paired_inputs
from signal_pipeline.data import utc
from signal_pipeline.engine import atomic_json
from signal_pipeline.evaluate import metrics
from signal_pipeline.protocol import PROTOCOL_HASH, training_hash, digest
from signal_pipeline.range_candidate import POLICY, HORIZONS, EXPERTS, interval_scores, validate_seed


def build(root, *, now=None):
    root=Path(root); current=utc(now); source=root/'historical_study.json'
    data=json.loads(source.read_text())
    if data['status']!='complete' or data['protocol_hash']!=PROTOCOL_HASH or data['training_hash']!=training_hash():
        raise ValueError('complete compatible historical study required')
    if utc(data['data_as_of']) >= current:
        raise ValueError('historical source must precede seed availability')
    seed={'policy':POLICY,'protocol_hash':PROTOCOL_HASH,'training_hash':training_hash(),
          'available_at':current.isoformat(),'data_as_of':data['data_as_of'],
          'source_report_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
          'source_snapshot':data['source_snapshot'],'horizons':{},'row_hashes':{},
          'role':'historical_development_feedback_for_future_shadow_only'}
    for h in HORIZONS:
        path=root/'historical_study'/f'h{h}_rows.csv.gz'
        frames=paired_inputs(pd.read_csv(path)); base=frames['incumbent']
        if len(base)!=data['horizons'][str(h)]['origins']:
            raise ValueError('source count mismatch')
        for name in EXPERTS:
            actual=metrics(frames[name].to_dict('records'))
            for metric in ('return_mae','coverage80','event_brier'):
                if not np.isclose(actual[metric],data['horizons'][str(h)][name][metric],rtol=1e-9,atol=1e-10):
                    raise ValueError('source score mismatch')
        rows=[]
        for i,r in base.iterrows():
            if utc(r.slot)<current-pd.Timedelta(days=365) or utc(r.target_end)>=current-pd.Timedelta(hours=1):continue
            q=np.expm1(np.array([frames[name].loc[i,['q10','q50','q90']].to_numpy(float) for name in EXPERTS]))
            rows.append({'slot':r.slot,'target_end':r.target_end,
                         'losses':interval_scores(q,float(np.expm1(r['return']))).tolist()})
        seed['horizons'][str(h)]=rows
        seed['row_hashes'][str(h)]=hashlib.sha256(path.read_bytes()).hexdigest()
    seed['seed_id']=digest(seed)
    return validate_seed(seed,current)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='lake/signals');p.add_argument('--output',default='lake/signals/range_seed.json')
    a=p.parse_args();seed=build(a.root);atomic_json(a.output,seed)
    print(json.dumps({'seed_id':seed['seed_id'],'available_at':seed['available_at'],
                      'origins':{h:len(r) for h,r in seed['horizons'].items()}}))
