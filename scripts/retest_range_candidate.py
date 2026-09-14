"""Recheck the range-only change on identical past origins; no full model fitting."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from signal_pipeline.adaptive_research import paired_inputs
from signal_pipeline.engine import atomic_json
from signal_pipeline.evaluate import metrics
from signal_pipeline.historical_study import losses,nonoverlap,paired_losses
from signal_pipeline.range_candidate import POLICY,HORIZONS,EXPERTS,mix_bounds


def replay(frame,h):
    frames=paired_inputs(frame);base=frames['incumbent']
    dates=pd.to_datetime(base.slot,utc=True).astype('int64').to_numpy()
    ends=pd.to_datetime(base.target_end,utc=True).astype('int64').to_numpy()
    hour=3600*10**9;day=24*hour
    if not np.all(ends-dates==(h+1)*hour):raise ValueError('bad target span')
    score=np.stack([losses(frames[k],'interval') for k in EXPERTS],axis=1)
    quantiles=np.exp(np.stack([frames[k][['q10','q50','q90']].to_numpy() for k in EXPERTS],axis=1))
    rows=[];indices=[]
    for i,origin in enumerate(dates):
        start=np.searchsorted(dates,origin-POLICY['window_days']*day)
        stop=np.searchsorted(ends,origin-hour,side='left')
        if stop-start<POLICY['minimum_daily_origins']:continue
        decay=np.exp2(-(origin-ends[start:stop])/(POLICY['half_life_days']*day))
        loss=np.average(score[start:stop],axis=0,weights=decay)
        logits=-POLICY['sensitivity']*loss/max(float(loss[1]),1e-8)
        w=np.exp(logits-logits.max());w/=w.sum()
        q=np.log(mix_bounds(quantiles[i],w))
        rows.append({**base.iloc[i].to_dict(),'model':POLICY['version'],'q10':q[0],'q50':q[1],'q90':q[2],
                     'feedback_target_end':base.target_end.iloc[stop-1]})
        indices.append(i)
    return rows,{k:frames[k].iloc[indices].to_dict('records') for k in EXPERTS}


def main(root,output):
    started=time.monotonic();root=Path(root)
    report={'policy':POLICY,'status':'historical_development_retest','horizons':{},
            'limits':'Previously inspected development data; not untouched holdout or prospective evidence.'}
    for h in HORIZONS:
        path=root/'historical_study'/f'h{h}_rows.csv.gz';f=pd.read_csv(path)
        candidate,refs=replay(f,h)
        def score(rows):
            return {**metrics(rows),'interval_score80_pp':float(losses(pd.DataFrame(rows),'interval').mean()*100)}
        result={'origins':len(candidate),'nonoverlap':len(nonoverlap(candidate)),
                'first_origin':candidate[0]['slot'],'last_origin':candidate[-1]['slot'],
                'candidate':score(candidate),'incumbent':score(refs['incumbent']),'frequency':score(refs['frequency']),
                'paired_vs_incumbent':paired_losses(candidate,refs['incumbent'],h),
                'paired_vs_frequency':paired_losses(candidate,refs['frequency'],h),
                'by_year':{},'rows_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        for year in sorted({r['slot'][:4] for r in candidate}):
            result['by_year'][year]={k:score([r for r in rows if r['slot'].startswith(year)])
                                     for k,rows in [('candidate',candidate),('incumbent',refs['incumbent'])]}
        # Perturb all future/immature truth. Earlier bounds must remain unchanged.
        cutoff=pd.Timestamp('2024-01-01',tz='UTC');altered=f.copy()
        altered.loc[pd.to_datetime(altered.target_end,utc=True)>=cutoff,'return']+=.4
        altered.loc[pd.to_datetime(altered.slot,utc=True)>=cutoff,['q10','q50','q90']]+=.2
        changed,_=replay(altered,h)
        keys=['slot','q10','q50','q90','feedback_target_end']
        before=[{k:r[k] for k in keys} for r in candidate if pd.Timestamp(r['slot'])<cutoff]
        after=[{k:r[k] for k in keys} for r in changed if pd.Timestamp(r['slot'])<cutoff]
        if before!=after:raise ValueError('future mutation changed earlier range')
        result['future_perturbation']={'unchanged':len(before),'cutoff':cutoff.isoformat()}
        # Price and probability remain exactly those from the same incumbent rows.
        for a,b in zip(candidate,refs['incumbent']):
            for k in ('p_down','p_flat','p_up','hit_up','hit_down'):
                if a[k]!=b[k]:raise ValueError('non-range output changed')
            if not np.isclose(a['q50'],b['q50'],atol=1e-15,rtol=0):raise ValueError('center changed')
        report['horizons'][str(h)]=result
    report['seconds']=time.monotonic()-started;atomic_json(output,report)
    print(json.dumps({h:{'origins':r['origins'],'coverage':r['candidate']['coverage80'],
                        'interval_improvement_pct':100*(1-r['candidate']['interval_score80_pp']/r['incumbent']['interval_score80_pp']),
                        'unchanged_future_test':r['future_perturbation']['unchanged']} for h,r in report['horizons'].items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.root,a.output)
