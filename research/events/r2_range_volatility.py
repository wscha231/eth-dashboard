"""R2 preregistered short-horizon range/volatility evaluation.

Research only. No production registry, issuance, or site writes.
The policy is intentionally small: persistence, EWMA, and HAR-RV on 6/24/72h.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signal_pipeline.data import build_features, read_bars

HORIZONS = (6, 24, 72)
FOLDS = (2024, 2025, 2026)
POLICY = {
    "experiment_id": "r2_short_range_volatility_v1",
    "horizons": list(HORIZONS),
    "fold_years": list(FOLDS),
    "models": ["persistence_720", "ewma_168", "har_rv"],
    "target": "sum of h squared ETH close-to-close log returns from slot+1h through slot+h",
    "range": "zero-center Gaussian q10/q50/q90 with endpoint variance=(h+1)/h*path variance",
    "train_days": 730,
    "calibration_days": 180,
    "embargo_hours": 1,
    "stride_hours": 24,
    "variance_gate": {"qlike_skill_vs_persistence": 0.05, "blocks_improve_min": 2, "max_block_degradation": 0.10},
    "distribution_gate": {"wis_skill_vs_better_baseline": 0.02, "coverage80_min": 0.76, "coverage80_max": 0.84,
                          "width_ratio_max": 1.10, "paired_improvement_probability_min": 0.90},
    "promotion": "disabled_research_only",
}


def qlike(actual, predicted):
    a=np.asarray(actual,float); p=np.asarray(predicted,float)
    if (a<=0).any() or (p<=0).any() or not np.isfinite(np.r_[a,p]).all():
        raise ValueError("QLIKE requires finite positive variance")
    r=a/p
    return float(np.mean(r-np.log(r)-1))


def pinball(y,q,a):
    d=np.asarray(y)-np.asarray(q)
    return float(np.mean(np.maximum(a*d,(a-1)*d)))


def wis80(y,q10,q50,q90):
    y=np.asarray(y); lo=np.asarray(q10); mid=np.asarray(q50); hi=np.asarray(q90)
    return float(np.mean((hi-lo)+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)+2*np.abs(mid-y))/4)


def _targets(bars, features, h):
    eth=bars.loc[bars.product.eq("ETH-USD")].set_index("close_time").reindex(features.index)
    lr=np.log(eth.close).diff()
    rv=lr.pow(2).rolling(h,min_periods=h).sum().shift(-(h+1))
    endpoint=np.log(eth.close.shift(-(h+1))/features.reference_price)
    out=pd.DataFrame(index=features.index)
    out["rv"]=rv
    out["endpoint_return"]=endpoint
    out["target_end"]=features.index+pd.Timedelta(hours=h+1)
    return out


def _ewma_variance(bars, features, span=168):
    eth=bars.loc[bars.product.eq("ETH-USD")].set_index("close_time").reindex(features.index)
    r=np.log(eth.close).diff()
    return r.pow(2).ewm(span=span,adjust=False,min_periods=span).mean()


def _indices(features, target, start, end, boundary=None):
    idx=features.index
    ready=(features[["eth_vol_24","eth_vol_168","eth_vol_720"]].notna().all(axis=1)&target.rv.gt(0)&target.endpoint_return.notna())
    ready &= (idx.hour == 0)
    ready &= (idx>=start)&(idx<end)
    if boundary is not None:
        ready &= target.target_end < boundary-pd.Timedelta(hours=1)
    return idx[ready]


def _fit_predict_har(features,target,train,cal,test,h):
    cols=["eth_vol_24","eth_vol_168","eth_vol_720"]
    x=np.log(features[cols].pow(2).clip(lower=1e-16))
    y=np.log((target.rv/h).clip(lower=1e-16))
    m=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(x.loc[train],y.loc[train])
    raw=np.exp(m.predict(x.loc[test]))
    cal_raw=np.exp(m.predict(x.loc[cal]))
    mult=float(np.mean((target.loc[cal,"rv"].to_numpy()/h)/cal_raw))
    return h*raw*mult


def _distribution_metrics(actual_return, variance_path, h):
    z=1.2815515655446004
    sd=np.sqrt(np.asarray(variance_path)*(h+1)/h)
    q10=-z*sd; q50=np.zeros(len(sd)); q90=z*sd
    y=np.asarray(actual_return,float)
    return {"wis80":wis80(y,q10,q50,q90),"coverage80":float(((y>=q10)&(y<=q90)).mean()),
            "mean_width":float(np.mean(q90-q10)),"pinball10":pinball(y,q10,.1),
            "pinball50":pinball(y,q50,.5),"pinball90":pinball(y,q90,.9)}


def _paired_probability(a,b,block=7,samples=1000):
    a=np.asarray(a,float); b=np.asarray(b,float); d=a-b
    blocks=[d[i:i+block] for i in range(0,len(d),block) if len(d[i:i+block])]
    if len(blocks)<8:return None
    rng=np.random.default_rng(1729); vals=[]
    for _ in range(samples):
        pick=rng.integers(len(blocks),size=len(blocks)); vals.append(np.concatenate([blocks[i] for i in pick]).mean())
    return float(np.mean(np.asarray(vals)<0))


def run(root="lake/signals", output=None):
    root=Path(root); bars=read_bars(root); features=build_features(bars)
    if features.empty: raise ValueError("hourly ETH/BTC history required")
    ewma=_ewma_variance(bars,features)
    result={"schema_version":1,"policy":POLICY,"horizons":{},"promotion":"none"}
    for h in HORIZONS:
        target=_targets(bars,features,h); folds=[]
        for year in FOLDS:
            start=pd.Timestamp(f"{year}-01-01",tz="UTC"); end=min(pd.Timestamp(f"{year+1}-01-01",tz="UTC"),features.index.max()+pd.Timedelta(hours=1))
            if start>=end: continue
            cal_start=start-pd.Timedelta(days=POLICY["calibration_days"]); train_start=cal_start-pd.Timedelta(days=POLICY["train_days"])
            train=_indices(features,target,train_start,cal_start,boundary=cal_start)
            cal=_indices(features,target,cal_start,start,boundary=start)
            test=_indices(features,target,start,end)
            if len(train)<400 or len(cal)<45 or len(test)<30: raise ValueError(f"insufficient fold {h}/{year}")
            actual=target.loc[test,"rv"].to_numpy(); ret=target.loc[test,"endpoint_return"].to_numpy()
            pred={
                "persistence_720":h*features.loc[test,"eth_vol_720"].to_numpy()**2,
                "ewma_168":h*ewma.loc[test].to_numpy(),
                "har_rv":_fit_predict_har(features,target,train,cal,test,h),
            }
            fold={"year":year,"rows":len(test),"models":{}}
            for name,p in pred.items():
                vm={"qlike":qlike(actual,p),"rmse_variance":float(np.sqrt(np.mean((p-actual)**2)))}
                dm=_distribution_metrics(ret,p,h)
                fold["models"][name]={"variance":vm,"distribution":dm}
            folds.append(fold)
        base=[f["models"]["persistence_720"]["variance"]["qlike"] for f in folds]
        summary={}
        for name in POLICY["models"]:
            q=[f["models"][name]["variance"]["qlike"] for f in folds]
            skills=[1-x/b for x,b in zip(q,base)]
            ws=[f["models"][name]["distribution"]["wis80"] for f in folds]
            cov=[f["models"][name]["distribution"]["coverage80"] for f in folds]
            wid=[f["models"][name]["distribution"]["mean_width"] for f in folds]
            summary[name]={"mean_qlike":float(np.mean(q)),"mean_qlike_skill_vs_persistence":float(np.mean(skills)),
                           "blocks_improved":int(sum(s>0 for s in skills)),"worst_block_skill":float(min(skills)),
                           "mean_wis80":float(np.mean(ws)),"mean_coverage80":float(np.mean(cov)),"mean_width":float(np.mean(wid))}
        eligible=[]
        for name,s in summary.items():
            if name=="persistence_720": continue
            vg=(s["mean_qlike_skill_vs_persistence"]>=.05 and s["blocks_improved"]>=2 and s["worst_block_skill"]>=-.10)
            eligible.append((name,vg,s["mean_qlike"]))
        winner=min((x for x in eligible if x[1]),key=lambda x:x[2],default=None)
        result["horizons"][str(h)]={"folds":folds,"summary":summary,"variance_gate_winner":winner[0] if winner else None,
                                      "status":"variance_gate_pass" if winner else "keep_persistence"}
    out=Path(output) if output else root/"r2_short_range_volatility.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default="lake/signals");p.add_argument("--output")
    a=p.parse_args();print(json.dumps(run(a.root,a.output),allow_nan=False))
