"""Frozen official checkpoints; endpoint distributions, NOT fabricated path probabilities.

Run only in a bounded research job. No candidate here can auto-promote or change
actual issued forecasts. Historical pretraining overlap is unknown.
"""
import argparse
import hashlib
import json
from pathlib import Path
import resource
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
import torch
from huggingface_hub import HfApi, snapshot_download

from signal_pipeline.data import build_features, feature_columns, read_bars, utc
from signal_pipeline.engine import atomic_json
from signal_pipeline.models import labels

CANDIDATES={"chronos2":("amazon/chronos-2","apache-2.0"),"tirex2":("NX-AI/TiRex-2","apache-2.0")}
REVISIONS={"chronos2":"29ec3766d36d6f73f0696f85560a422f50e8498c", "tirex2":"05e5b26db52bfb256f1ae1bdf785589850482de3"}


def load_candidate(name):
    repo,license_name=CANDIDATES[name]
    info=HfApi().model_info(repo,revision=REVISIONS[name],timeout=15,token=False)
    if info.card_data.get('license')!=license_name:
        raise ValueError('checkpoint license requires review')
    checkpoint=snapshot_download(repo,revision=info.sha,token=False,
                                 allow_patterns=['*.json','*.yaml','*.yml','*.safetensors','*.pt','*.pth','*.ckpt','*.bin'])
    if name=='chronos2':
        from chronos import Chronos2Pipeline
        model=Chronos2Pipeline.from_pretrained(checkpoint,device_map='cpu',torch_dtype=torch.float32)
    else:
        from tirex2 import load_model
        model=load_model(checkpoint,device='cpu')
    return model,{'repository':repo,'revision':info.sha,'license':license_name,
                  'pretraining_overlap':'unknown; historical results are diagnostic only'}


def predict_quantiles(model,name,context,horizon):
    # The only inputs are timestamped past ETH/BTC log prices. No future market covariates.
    if name=='chronos2':
        frame=pd.DataFrame({'id':'ETH-USD','timestamp':context.index.tz_localize(None),
                            'target':context.eth.to_numpy(),'btc':context.btc.to_numpy()})
        result=model.predict_df(frame,id_column='id',timestamp_column='timestamp',target='target',
                                prediction_length=horizon+1,quantile_levels=[.1,.5,.9],
                                batch_size=2,cross_learning=False,freq='h')
        values=result[['0.1','0.5','0.9']].iloc[-1].to_numpy(float)
    else:
        from tirex2 import TimeseriesType
        tensor=torch.tensor(context[['eth','btc']].to_numpy().T,dtype=torch.float32)
        series=TimeseriesType(target=tensor,past_covariates=None,future_covariates=None)
        result=model.forecast([series],prediction_length=horizon+1,output_type='numpy')[0]
        values=np.asarray(result)[0,[0,4,8],-1]
    if not np.isfinite(values).all():raise ValueError('non-finite foundation forecast')
    return np.sort(values)-context.eth.iloc[-1]


def price_ensemble_report(rows,root):
    """Convex endpoint-quantile blends; weights use earlier matured paired outcomes.

    These are price-only diagnostic ensembles. They do not turn marginal price
    quantiles into joint up/down barrier-hit probabilities.
    """
    reports={}
    for h in sorted({r['horizon_hours'] for r in rows}):
        path=Path(root)/'replay'/f'h{h}.csv.gz'
        if not path.exists():continue
        prior=pd.read_csv(path);prior=prior[prior.model.eq('selected')].set_index('slot')
        common=sorted([r for r in rows if r['horizon_hours']==h and r['slot'] in prior.index],key=lambda r:r['slot'])
        history=[];evaluated=[]
        for r in common:
            base=prior.loc[r['slot']];b=base[['q10','q50','q90']].to_numpy(float)
            neural=np.array([r['q10'],r['q50'],r['q90']])
            known=[x for x in history if pd.Timestamp(x['target_end'])<pd.Timestamp(r['slot'])]
            weight=0.
            if len(known)>=20:
                weight=min((0.,.5,1.),key=lambda w:np.mean([abs(np.expm1((1-w)*x['base'][1]+w*x['neural'][1])-np.expm1(x['actual'])) for x in known]))
            evaluated.append({'base':b,'foundation':neural,'fixed_half':.5*(b+neural),
                              'past_selected_blend':(1-weight)*b+weight*neural,
                              'actual':r['return'],'neural_weight':weight})
            history.append({'target_end':base.target_end,'base':b,'neural':neural,'actual':r['return']})
        if not evaluated:continue
        y=np.array([r['actual'] for r in evaluated]);naive=float(np.abs(np.expm1(y)).mean())
        scores={}
        for model in ('base','foundation','fixed_half','past_selected_blend'):
            q=np.stack([r[model] for r in evaluated]);mae=float(np.abs(np.expm1(q[:,1])-np.expm1(y)).mean())
            scores[model]={'return_mae':mae,'mae_skill_vs_no_change':1-mae/naive,
                           'coverage80':float(((y>=q[:,0])&(y<=q[:,2])).mean())}
        reports[str(h)]={'common_origins':len(evaluated),'models':scores,'no_change_mae':naive,
                         'selected_neural_weights':{str(w):sum(r['neural_weight']==w for r in evaluated) for w in (0.,.5,1.)},
                         'status':'diagnostic; unknown foundation pretraining overlap; not a deployed winner'}
    return reports


def run(name,root,*,horizons=(24,72,168),max_origins=100,budget_seconds=600):
    started=time.monotonic();root=Path(root);torch.set_num_threads(2)
    bars=read_bars(root);features=build_features(bars)
    eligible=features.index[features[feature_columns(features)].notna().all(axis=1) & (features.index.hour==0)]
    # Fixed evenly spaced origins, not origins selected for attractive model performance.
    eligible=eligible[eligible+pd.Timedelta(hours=max(horizons)+1)<=features.index.max()]
    chosen=eligible[np.unique(np.linspace(0,len(eligible)-1,min(max_origins,len(eligible)),dtype=int))] if len(eligible) else []
    output={'candidate':name,'generated_at':utc().isoformat(),'purpose':'bounded runtime and historical diagnostic',
            'auto_promotion':False,'path_probability_status':'not inferred from marginal endpoint quantiles',
            'status':'started','rows':[],'errors':[]}
    target=root/f'foundation_{name}.json'
    try:
        if not len(chosen):raise ValueError('no eligible diagnostic origins')
        model,metadata=load_candidate(name);output['checkpoint']=metadata
        context=pd.concat([np.log(bars.loc[bars['product']==p].set_index('close_time').close).rename(n)
                           for p,n in [('ETH-USD','eth'),('BTC-USD','btc')]],axis=1)
        outcomes={h:labels(bars,features,h) for h in horizons}
        for slot in chosen:
            past=context.loc[:slot].tail(512)
            if len(past)!=512 or past.isna().any().any():continue
            for h in horizons:
                if time.monotonic()-started>budget_seconds:
                    output['status']='budget_exhausted';return output
                before=time.monotonic();q=predict_quantiles(model,name,past,h)
                y=outcomes[h].loc[slot]
                if not np.isfinite(y['return']):continue
                output['rows'].append({'slot':slot.isoformat(),'horizon_hours':h,'q10':float(q[0]),'q50':float(q[1]),
                                       'q90':float(q[2]),'return':float(y['return']),'inference_seconds':time.monotonic()-before})
                print(name,slot.isoformat(),h,output['rows'][-1]['inference_seconds'],flush=True)
        output['status']='completed'
    except Exception as exc:
        output['status']='unavailable_or_failed';output['errors'].append({'type':type(exc).__name__,'detail':str(exc)[:300]})
    finally:
        output['runtime_seconds']=time.monotonic()-started
        output['peak_rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
        output['metrics']={}
        output['price_ensembles']=price_ensemble_report(output['rows'],root)
        for h in horizons:
            f=pd.DataFrame([r for r in output['rows'] if r['horizon_hours']==h])
            if f.empty:continue
            mae=float(np.abs(np.expm1(f.q50)-np.expm1(f['return'])).mean());base=float(np.abs(np.expm1(f['return'])).mean())
            output['metrics'][str(h)]={'origins':len(f),'return_mae':mae,'no_change_mae':base,
                'mae_skill':1-mae/base,'coverage80':float(((f['return']>=f.q10)&(f['return']<=f.q90)).mean()),
                'p95_seconds':float(f.inference_seconds.quantile(.95))}
        atomic_json(target,output)
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate',choices=CANDIDATES,required=True);p.add_argument('--root',default='lake/signals')
    p.add_argument('--max-origins',type=int,default=100);p.add_argument('--budget-seconds',type=int,default=600)
    a=p.parse_args();r=run(a.candidate,a.root,max_origins=a.max_origins,budget_seconds=a.budget_seconds)
    print(json.dumps({k:v for k,v in r.items() if k!='rows'},indent=2))
