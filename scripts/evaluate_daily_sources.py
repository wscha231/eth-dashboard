"""Bounded, matched-origin screening of compact models and historical flow data.

No automatic promotion. Hyperparameters and period are fixed before execution;
neutral outcomes remain in multiclass scoring. All fitting is monthly and purged.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

PROTOCOL = {
    "version": "matched_daily_screen_v1", "seed": 42,
    "test_start": "2025-07-01", "last_target_end": "2026-07-31",
    "train_years": 3, "refit": "monthly", "purge": "label_end < first_test_origin",
    "horizons": [7, 30], "candidates": ["ridge_price", "hist_price", "hist_price_flow"],
    "hgb": {"max_iter": 100, "max_leaf_nodes": 15, "min_samples_leaf": 30, "learning_rate": .05, "l2_regularization": 1.0},
    "ridge_alpha": 100.0, "classification": "UP/FLAT/DOWN on every test origin",
    "event_threshold": {"formula": "clip(std(return_1d,30)*sqrt(horizon)*multiplier,floor,cap)",
                        "7": {"multiplier": .65, "floor": .055, "cap": .16},
                        "30": {"multiplier": .70, "floor": .08, "cap": .24}},
    "bootstrap": {"replicates": 1000, "block_rows": "max(horizon,30)"},
    "promotion": "screen only; independent future shadow evidence required",
    "data_limit": "historical reconstructed bars; no historical macro/DeFi publication vintages",
}


def features(master, lead):
    master.index = pd.to_datetime(master.index, utc=True).tz_localize(None) + pd.Timedelta(days=1)
    close = master.eth_close.astype(float)
    x = pd.DataFrame(index=master.index)
    ret = close.pct_change(fill_method=None)
    for n in [1, 3, 7, 14, 30, 60, 90]:
        x[f"eth_momentum_{n}"] = close.pct_change(n, fill_method=None)
    for n in [7, 30, 90]:
        x[f"volatility_{n}"] = ret.rolling(n).std()
    for n in [20, 60]:
        x[f"distance_ma_{n}"] = close / close.rolling(n).mean() - 1
    if "eth_volume" in master:
        volume = master.eth_volume.astype(float)
        x["volume_ratio_30"] = volume / volume.rolling(30).mean()
    for n in [7, 30]:
        x[f"btc_momentum_{n}"] = master.btc_close.pct_change(n, fill_method=None)
        x[f"relative_eth_btc_{n}"] = x[f"eth_momentum_{n}"] - x[f"btc_momentum_{n}"]
    price_columns = list(x)
    available = pd.to_datetime(lead.pop("feature_available_at_utc"), utc=True).dt.tz_localize(None)
    lead.index = pd.DatetimeIndex(available)
    if lead.index.has_duplicates:
        raise ValueError("Duplicate source availability times")
    names = ["eth_spot_taker_buy_quote_share", "eth_spot_signed_taker_flow_ratio", "eth_spot_jump_variation_ratio",
             "eth_spot_last_4h_return", "eth_spot_last_8h_return", "eth_spot_last_12h_return",
             "eth_spot_last_4h_quote_volume_share", "eth_perp_basis", "eth_spot_perp_flow_divergence",
             "eth_futures_spot_quote_volume_ratio", "eth_btc_spot_flow_spread", "eth_btc_perp_flow_spread",
             "eth_btc_basis_spread", "btc_spot_last_4h_return", "eth_perp_taker_buy_quote_share",
             "eth_perp_basis_delta_7d", "eth_spot_signed_taker_flow_ratio_delta_7d"]
    good = pd.to_numeric(lead["market_data_excluded"], errors="coerce").fillna(1).eq(0)
    source = lead.loc[good, names].apply(pd.to_numeric, errors="coerce")
    x = x.join(source, how="inner").replace([np.inf,-np.inf], np.nan)
    return x, price_columns, close, ret


def interval(values, horizon):
    values = np.asarray(values, dtype=float)
    block = max(horizon, 30)
    rng = np.random.default_rng(42)
    means = []
    for _ in range(1000):
        starts = rng.integers(0, len(values), size=int(np.ceil(len(values)/block)))
        ix = np.concatenate([(start+np.arange(block)) % len(values) for start in starts])[:len(values)]
        means.append(float(values[ix].mean()))
    return [float(v) for v in np.quantile(means,[.025,.975])]


def evaluate(master, lead, budget_seconds=180):
    started = time.perf_counter()
    x, price_columns, close, ret = features(master, lead)
    rows=[]
    with threadpool_limits(limits=2):
        for h in [7,30]:
            y=(close.shift(-h)/close-1).reindex(x.index)
            multiplier,floor,cap = (.65,.055,.16) if h<=7 else (.70,.08,.24)
            threshold=(ret.rolling(30).std()*np.sqrt(h)*multiplier).clip(floor,cap).reindex(x.index)
            target=pd.Series(np.where(y>=threshold,2,np.where(y<=-threshold,0,1)),index=x.index)
            valid=y.notna() & threshold.notna()
            for month in pd.date_range(PROTOCOL["test_start"], PROTOCOL["last_target_end"], freq="MS"):
                train=(x.index+pd.Timedelta(days=h)<month)&(x.index>=month-pd.DateOffset(years=3))&valid
                test=(x.index>=month)&(x.index<month+pd.offsets.MonthBegin(1))&(x.index+pd.Timedelta(days=h)<=pd.Timestamp(PROTOCOL["last_target_end"]))&valid
                if not test.any():
                    continue
                if train.sum()<500:
                    raise ValueError("Insufficient matched training history")
                prior=np.bincount(target.loc[train],minlength=3)/train.sum()
                for name in PROTOCOL["candidates"]:
                    if time.perf_counter()-started>budget_seconds:
                        raise TimeoutError("Predeclared screening budget exceeded")
                    cols=price_columns if name!="hist_price_flow" else list(x)
                    cols=[c for c in cols if x.loc[train,c].notna().mean()>=.6]
                    if name=="ridge_price":
                        model=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),Ridge(alpha=100.0))
                        classifier=make_pipeline(SimpleImputer(keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=500,random_state=42))
                    else:
                        model=HistGradientBoostingRegressor(random_state=42,**PROTOCOL["hgb"])
                        classifier=HistGradientBoostingClassifier(random_state=42,**PROTOCOL["hgb"])
                    model.fit(x.loc[train,cols],y.loc[train])
                    pred=model.predict(x.loc[test,cols])
                    classifier.fit(x.loc[train,cols],target.loc[train])
                    probabilities=classifier.predict_proba(x.loc[test,cols])
                    aligned=np.zeros((test.sum(),3)); aligned[:,classifier.classes_]=probabilities
                    for i,date in enumerate(x.index[test]):
                        label=int(target.loc[date]); onehot=np.eye(3)[label]
                        rows.append({"origin":date.isoformat(),"target_end":(date+pd.Timedelta(days=h)).isoformat(),
                            "horizon":h,"candidate":name,"actual_return":float(y.loc[date]),"predicted_return":float(pred[i]),
                            "class_actual":label,"class_predicted":int(np.argmax(aligned[i])),
                            "brier":float(np.square(aligned[i]-onehot).sum()),"prior_brier":float(np.square(prior-onehot).sum()),
                            "feature_count":len(cols),"training_rows":int(train.sum())})
                print(f"screen h={h} month={month.date()} elapsed={time.perf_counter()-started:.1f}s",flush=True)
    table=pd.DataFrame(rows)
    reports=[]
    for (h,name),group in table.groupby(["horizon","candidate"]):
        actual=group.actual_return.to_numpy();pred=group.predicted_return.to_numpy()
        loss=np.abs(pred-actual); base=np.abs(actual)
        rmse=np.sqrt(np.mean((pred-actual)**2)); base_rmse=np.sqrt(np.mean(actual**2))
        ci=interval(base-loss,h)
        reports.append({"horizon":int(h),"candidate":name,"rows":len(group),"return_mae":float(loss.mean()),
            "return_rmse":float(rmse),"mae_skill_vs_no_change":float(1-loss.mean()/base.mean()),
            "rmse_skill_vs_no_change":float(1-rmse/base_rmse),"mean_mae_improvement_95ci":ci,
            "multiclass_brier":float(group.brier.mean()),"brier_skill_vs_train_prior":float(1-group.brier.mean()/group.prior_brier.mean()),
            "all_origin_accuracy":float((group.class_actual==group.class_predicted).mean()),
            "actual_class_counts":{str(k):int(v) for k,v in group.class_actual.value_counts().items()},
            "screen_price_pass":bool(ci[0]>0 and rmse<base_rmse),"promotion":"not_authorized_by_screen"})
    ablations=[]
    for h in [7,30]:
        base=table[(table.horizon==h)&(table.candidate=="hist_price")].set_index("origin")
        flow=table[(table.horizon==h)&(table.candidate=="hist_price_flow")].set_index("origin")
        if not base.index.equals(flow.index):
            raise AssertionError("Source ablation origin mismatch")
        gain=(base.predicted_return-base.actual_return).abs()-(flow.predicted_return-flow.actual_return).abs()
        ablations.append({"horizon":h,"matched_origins":len(base),"flow_mae_improvement":float(gain.mean()),"95ci":interval(gain,h)})
    return {"protocol":PROTOCOL,"runtime_seconds":time.perf_counter()-started,"results":reports,"source_ablation":ablations},table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv",required=True)
    parser.add_argument("--lead-data-csv",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--budget-seconds",type=float,default=180)
    args=parser.parse_args()
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/"protocol.json").write_text(json.dumps(PROTOCOL,indent=2))
    master=pd.read_csv(args.master_data_csv,index_col=0,parse_dates=True)
    lead=pd.read_csv(args.lead_data_csv)
    result,rows=evaluate(master,lead,args.budget_seconds)
    result["source_hashes"]={Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [args.master_data_csv,args.lead_data_csv]}
    (out/"results.json").write_text(json.dumps(result,indent=2))
    rows.to_csv(out/"predictions.csv.gz",index=False)
    print(json.dumps(result["results"],indent=2))


if __name__=="__main__":
    main()
