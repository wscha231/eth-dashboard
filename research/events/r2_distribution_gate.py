"""R2-2 frozen short-horizon distribution/range gate.

Uses the R2 HAR-RV variance head and adds one prior-only normalized-conformal
central-80 interval. Research only: no production registry, issuance, or site writes.
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
Z80 = 1.2815515655446004
POLICY = {
    "experiment_id": "r2_short_distribution_v1",
    "horizons": list(HORIZONS),
    "fold_years": list(FOLDS),
    "candidate": "har_norm_conformal80",
    "diagnostic_only": "har_gaussian_raw",
    "baselines": ["persistence_gaussian", "incumbent_selected"],
    "center": "zero log-return; no-change center fixed independently of variance head",
    "variance": "HAR-RV log variance on prior 24/168/720h sample variances with prior calibration mean-ratio correction",
    "conformal": "prior-only q80 of abs(endpoint_return)/sqrt(HAR endpoint variance); method=higher",
    "train_days": 730,
    "calibration_days": 180,
    "embargo_hours": 1,
    "test_stride_hours": 24,
    "gate": {
        "wis_skill_vs_each_baseline": 0.02,
        "coverage80_min": 0.76,
        "coverage80_max": 0.84,
        "width_ratio_vs_better_wis_baseline_max": 1.10,
        "paired_improvement_probability_vs_each_baseline_min": 0.90,
    },
    "bootstrap_samples": 2000,
    "promotion": "disabled_research_only; historical development evidence cannot satisfy prospective gate",
}


def wis80_rows(y, q10, q50, q90):
    y=np.asarray(y,float); lo=np.asarray(q10,float); mid=np.asarray(q50,float); hi=np.asarray(q90,float)
    return ((hi-lo)+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)+2*np.abs(mid-y))/4


def metrics(y, q10, q50, q90):
    losses=wis80_rows(y,q10,q50,q90)
    y=np.asarray(y,float); lo=np.asarray(q10,float); hi=np.asarray(q90,float)
    return {"wis80":float(losses.mean()),"coverage80":float(((y>=lo)&(y<=hi)).mean()),
            "mean_width":float((hi-lo).mean()),"rows":int(len(y))}


def _targets(bars, features, h):
    eth=bars.loc[bars["product"].eq("ETH-USD")].set_index("close_time").reindex(features.index)
    lr=np.log(eth.close).diff()
    out=pd.DataFrame(index=features.index)
    out["rv"]=lr.pow(2).rolling(h,min_periods=h).sum().shift(-(h+1))
    out["return"]=np.log(eth.close.shift(-(h+1))/features.reference_price)
    out["target_end"]=features.index+pd.Timedelta(hours=h+1)
    return out


def _indices(features, target, start, end, boundary=None):
    idx=features.index
    ready=(features[["eth_vol_24","eth_vol_168","eth_vol_720"]].notna().all(axis=1)
           & target.rv.gt(0) & target["return"].notna() & (idx.hour==0)
           & (idx>=start) & (idx<end))
    if boundary is not None:
        ready &= target.target_end < boundary-pd.Timedelta(hours=1)
    return idx[ready]


def _har_predictions(features,target,train,cal,test,h):
    cols=["eth_vol_24","eth_vol_168","eth_vol_720"]
    x=np.log(features[cols].pow(2).clip(lower=1e-16))
    y=np.log((target.rv/h).clip(lower=1e-16))
    model=make_pipeline(StandardScaler(),Ridge(alpha=10.)).fit(x.loc[train],y.loc[train])
    raw_cal=np.exp(model.predict(x.loc[cal])); raw_test=np.exp(model.predict(x.loc[test]))
    multiplier=float(np.mean((target.loc[cal,"rv"].to_numpy()/h)/raw_cal))
    return h*raw_cal*multiplier, h*raw_test*multiplier


def _candidate_quantiles(cal_return,cal_path_var,test_path_var,h):
    cal_endpoint_var=np.asarray(cal_path_var)*(h+1)/h
    test_endpoint_var=np.asarray(test_path_var)*(h+1)/h
    standardized=np.abs(np.asarray(cal_return,float))/np.sqrt(np.clip(cal_endpoint_var,1e-16,None))
    factor=float(np.quantile(standardized,.80,method="higher"))
    half=factor*np.sqrt(np.clip(test_endpoint_var,1e-16,None))
    return -half,np.zeros(len(half)),half,factor


def _persistence_quantiles(features,test,h):
    path=h*features.loc[test,"eth_vol_720"].to_numpy()**2
    half=Z80*np.sqrt(path*(h+1)/h)
    return -half,np.zeros(len(half)),half


def _load_incumbent(root,h):
    p=Path(root)/"replay"/f"h{h}.csv.gz"
    if not p.exists(): raise ValueError(f"incumbent replay missing for {h}h")
    f=pd.read_csv(p)
    f=f[f["model"].eq("selected")].copy()
    if f.empty or f["slot"].duplicated().any(): raise ValueError(f"invalid incumbent replay for {h}h")
    f["slot_ts"]=pd.to_datetime(f["slot"],utc=True)
    return f.set_index("slot_ts")


def _paired_probability(candidate_loss,baseline_loss,slots,h,samples=2000):
    delta=np.asarray(candidate_loss)-np.asarray(baseline_loss)
    slots=pd.DatetimeIndex(slots)
    block_days=max(7,int(np.ceil((h+1)/24)))
    groups=(slots.astype("int64")//(86400*10**9*block_days)).to_numpy()
    blocks=[delta[groups==g] for g in np.unique(groups)]
    if len(blocks)<10:return None
    rng=np.random.default_rng(1729); values=[]
    for _ in range(samples):
        pick=rng.integers(len(blocks),size=len(blocks))
        values.append(np.concatenate([blocks[i] for i in pick]).mean())
    return {"probability_candidate_improves":float(np.mean(np.asarray(values)<0)),
            "mean_loss_difference":float(delta.mean()),"blocks":len(blocks),"block_days":block_days}


def run(root="lake/signals",output=None):
    root=Path(root); bars=read_bars(root); features=build_features(bars)
    if features.empty: raise ValueError("hourly ETH/BTC history required")
    result={"schema_version":1,"policy":POLICY,"horizons":{},"promotion":"none"}
    for h in HORIZONS:
        target=_targets(bars,features,h); incumbent=_load_incumbent(root,h); fold_outputs=[]
        all_rows=[]
        for year in FOLDS:
            start=pd.Timestamp(f"{year}-01-01",tz="UTC")
            end=min(pd.Timestamp(f"{year+1}-01-01",tz="UTC"),features.index.max()+pd.Timedelta(hours=1))
            if start>=end:continue
            cal_start=start-pd.Timedelta(days=POLICY["calibration_days"])
            train_start=cal_start-pd.Timedelta(days=POLICY["train_days"])
            train=_indices(features,target,train_start,cal_start,boundary=cal_start)
            cal=_indices(features,target,cal_start,start,boundary=start)
            test=_indices(features,target,start,end)
            test=test.intersection(incumbent.index)
            if len(train)<400 or len(cal)<45 or len(test)<30: raise ValueError(f"insufficient matched fold {h}/{year}")
            cal_var,test_var=_har_predictions(features,target,train,cal,test,h)
            cq10,cq50,cq90,factor=_candidate_quantiles(target.loc[cal,"return"],cal_var,test_var,h)
            rq10=np.full(len(test),np.nan); rq50=np.zeros(len(test)); rq90=np.full(len(test),np.nan)
            half_raw=Z80*np.sqrt(test_var*(h+1)/h);rq10=-half_raw;rq90=half_raw
            pq10,pq50,pq90=_persistence_quantiles(features,test,h)
            inc=incumbent.loc[test]
            truth=target.loc[test,"return"].to_numpy()
            if not np.allclose(truth,inc["return"].to_numpy(float),rtol=0,atol=1e-12,equal_nan=False):
                raise ValueError(f"incumbent truth mismatch for {h}/{year}")
            rows=pd.DataFrame({"slot":test,"truth":truth,
                "candidate_q10":cq10,"candidate_q50":cq50,"candidate_q90":cq90,
                "raw_q10":rq10,"raw_q50":rq50,"raw_q90":rq90,
                "persistence_q10":pq10,"persistence_q50":pq50,"persistence_q90":pq90,
                "incumbent_q10":inc.q10.to_numpy(float),"incumbent_q50":inc.q50.to_numpy(float),"incumbent_q90":inc.q90.to_numpy(float)})
            rows["year"]=year;all_rows.append(rows)
            fold_outputs.append({"year":year,"rows":len(test),"conformal_factor":factor,
                "candidate":metrics(truth,cq10,cq50,cq90),"har_gaussian_raw":metrics(truth,rq10,rq50,rq90),
                "persistence_gaussian":metrics(truth,pq10,pq50,pq90),
                "incumbent_selected":metrics(truth,inc.q10,inc.q50,inc.q90)})
        f=pd.concat(all_rows,ignore_index=True).sort_values("slot")
        y=f.truth.to_numpy()
        names={
            "candidate":("candidate_q10","candidate_q50","candidate_q90"),
            "har_gaussian_raw":("raw_q10","raw_q50","raw_q90"),
            "persistence_gaussian":("persistence_q10","persistence_q50","persistence_q90"),
            "incumbent_selected":("incumbent_q10","incumbent_q50","incumbent_q90"),
        }
        overall={name:metrics(y,f[a],f[b],f[c]) for name,(a,b,c) in names.items()}
        losses={name:wis80_rows(y,f[a],f[b],f[c]) for name,(a,b,c) in names.items()}
        candidate=overall["candidate"]
        base_names=["persistence_gaussian","incumbent_selected"]
        skills={b:1-candidate["wis80"]/overall[b]["wis80"] for b in base_names}
        paired={b:_paired_probability(losses["candidate"],losses[b],f.slot,h,POLICY["bootstrap_samples"]) for b in base_names}
        better=min(base_names,key=lambda b:overall[b]["wis80"])
        width_ratio=candidate["mean_width"]/overall[better]["mean_width"]
        g=POLICY["gate"]
        passed=(all(skills[b]>=g["wis_skill_vs_each_baseline"] for b in base_names)
                and g["coverage80_min"]<=candidate["coverage80"]<=g["coverage80_max"]
                and width_ratio<=g["width_ratio_vs_better_wis_baseline_max"]
                and all(paired[b] is not None and paired[b]["probability_candidate_improves"]>=g["paired_improvement_probability_vs_each_baseline_min"] for b in base_names))
        result["horizons"][str(h)]={"folds":fold_outputs,"overall":overall,"candidate_wis_skill_vs":skills,
            "paired_vs":paired,"better_wis_baseline":better,"candidate_width_ratio_vs_better_baseline":float(width_ratio),
            "distribution_gate_pass":bool(passed),"status":"distribution_gate_pass" if passed else "keep_existing_distribution"}
    out=Path(output) if output else root/"r2_short_distribution.json"
    out.write_text(json.dumps(result,indent=2,allow_nan=False))
    return result

if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser();p.add_argument("--root",default="lake/signals");p.add_argument("--output")
    a=p.parse_args();print(json.dumps(run(a.root,a.output),allow_nan=False))
