"""Perturb real future truths/base forecasts and verify earlier adaptive outputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from signal_pipeline.adaptive_research import replay, paired_inputs, P, Q
from signal_pipeline.data import read_bars, build_features, utc


def check(root, cutoff):
    root=Path(root);cutoff=utc(cutoff);features=build_features(read_bars(root));checked={}
    for h in (6,720):
        path=root/'historical_study'/f'h{h}_rows.csv.gz';before=hashlib.sha256(path.read_bytes()).hexdigest()
        frame=pd.read_csv(path);base=paired_inputs(frame)['incumbent']
        dates=pd.to_datetime(base.slot,utc=True);scale=features.loc[dates,'sigma'].to_numpy()*np.sqrt(h)
        original,_=replay(frame,scale,h)
        changed=frame.copy();future_truth=pd.to_datetime(changed.target_end,utc=True)>=cutoff
        changed.loc[future_truth,'return']+=.4
        changed.loc[future_truth,['up','down']]=1-changed.loc[future_truth,['up','down']]
        changed.loc[future_truth,'terminal']=(changed.loc[future_truth,'terminal']+1)%3
        future_input=pd.to_datetime(changed.slot,utc=True)>=cutoff
        changed.loc[future_input,Q]+=.2
        changed_scale=scale.copy();changed_scale[dates>=cutoff]*=3
        revised,_=replay(changed,changed_scale,h)
        keys=P+Q+['model','model_version','threshold_up','threshold_down','feedback_target_end','feedback_rows','calibration_cutoff','calibration_target_end']
        a=[{k:r[k] for k in keys} for r in original if utc(r['slot'])<cutoff]
        b=[{k:r[k] for k in keys} for r in revised if utc(r['slot'])<cutoff]
        if a!=b:raise ValueError('future information changed earlier adaptive predictions')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=before:raise ValueError('source was modified')
        checked[str(h)]={'earlier_adaptive_outputs_compared':len(a),'numeric_forecasts_thresholds_and_versions_identical':True,
                         'source_unchanged':True,'source_rows_sha256':before}
    return {'status':'passed','cutoff':cutoff.isoformat(),'horizons':checked,
            'perturbation':'Returns +0.4 log units and event/class labels altered only when target_end >= cutoff; base quantiles +0.2 and volatility x3 only at origins >= cutoff. Pending earlier-origin 30d labels are included in the perturbation.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='lake/signals')
    p.add_argument('--cutoff',default='2024-01-01');p.add_argument('--output',required=True);a=p.parse_args()
    result=check(a.root,a.cutoff);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
