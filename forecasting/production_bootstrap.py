"""Bootstrap saved estimators from issued model choices and existing OOF evidence.

Does not repeat model selection. Historical OOF evidence is calibration context,
not new proof that the deployed composite policy beats its baseline.
"""
from __future__ import annotations

from dataclasses import fields
import json
import re
import sqlite3

import pandas as pd

import eth_price_forecast as efp
from forecasting.daily_data import file_sha256, iso_utc
from forecasting.model_bundle import final_model_cache
from forecasting.runtime_metrics import stage


def bootstrap_issued_policy(market, db_path, evidence_path):
    conn=sqlite3.connect(f"file:{db_path}?mode=ro",uri=True)
    conn.row_factory=sqlite3.Row
    try:
        issued=[dict(r) for r in conn.execute("SELECT * FROM forecasts WHERE run_id=(SELECT MAX(run_id) FROM forecast_runs)")]
    finally:
        conn.close()
    if {r["horizon_days"] for r in issued}!={7,30}:
        raise ValueError("Bootstrap requires a complete previously issued 7/30 policy")
    evidence=json.loads(evidence_path.read_text())
    completed = evidence.get("folds_completed", 0)
    complete = all(int(completed.get(str(h), 0)) >= 36 for h in (7, 30)) if isinstance(completed, dict) else int(completed) >= 36
    if evidence.get("partial") or not complete:
        raise ValueError("Bootstrap evidence must be a completed historical evaluation")
    horizons={}
    for row in sorted(issued,key=lambda r:r["horizon_days"]):
        h=int(row["horizon_days"])
        print(f"Bootstrap issued choices h={h}: {row['regression_model']} / {row['classification_model']}",flush=True)
        with stage(f"h{h}_bootstrap_from_issued_policy"):
            feature_frame,candidates=efp.build_features(market,horizon=h)
            target_columns=["eth_close","target_return","target_close",*efp.DIRECTION_TARGET_COLUMNS,
                            "target_regime","target_reversal_state","target_bottom_reversal","target_top_reversal"]
            valid=feature_frame.target_return.notna() & feature_frame.target_close.notna()
            dataset=efp.trim_training_window(feature_frame.loc[valid,list(dict.fromkeys(candidates+target_columns))],"1d",h)
            features,coverage=efp.select_usable_feature_columns(feature_frame,candidates,feature_frame.index.isin(dataset.index),efp.DEFAULT_FEATURE_MIN_COVERAGE,horizon=h)
            prediction=feature_frame.loc[feature_frame.eth_close.notna(),list(dict.fromkeys(features+["eth_close"]))]
            saved=evidence["horizons"][str(h)]
            reg_lb=pd.DataFrame(saved["regression_leaderboard"])
            cls_lb=pd.DataFrame(saved["classification_leaderboard"])
            components=re.search(r"skill_weighted\[([^]]+)\]",row["regression_selection_basis"] or "")
            if components is not None:
                members=efp.parse_component_models(components.group(1))
                if not set(members).issubset(efp.make_models(h)):
                    raise ValueError("Issued ensemble contains an unavailable production estimator")
                reg_lb=reg_lb.loc[reg_lb.model.isin(members)].copy()
                # An issued composite can be absent from the weekly candidate
                # table. Its membership is known; its composite OOS score is
                # not. Leave that score missing instead of substituting a
                # different ensemble or inventing performance evidence.
                reg_lb=pd.concat([reg_lb,pd.DataFrame([{"model":row["regression_model"],"component_models":components.group(1)}])],ignore_index=True)
            interval_match=re.search(r"independent_conformal_interval\[([^]]+)\]",row["regression_selection_basis"] or "")
            interval_model=interval_match.group(1) if interval_match else None
            if components is None:
                reg_lb=reg_lb.loc[reg_lb.model.isin([row["regression_model"],interval_model])].copy()
            predictions=pd.DataFrame(saved["predictions"])
            reg=predictions.loc[predictions["head"].eq("regression")].copy()
            # Old OOF dates label source bar starts; the new model index labels ends.
            reg["origin"]=pd.to_datetime(reg.prediction_date,utc=True).dt.tz_localize(None)+pd.Timedelta(days=1)
            reg=reg.loc[reg.origin+pd.Timedelta(days=h)<=market.index[-1]]
            oof=reg.pivot(index="origin",columns="model",values="predicted_return")
            oof.columns=[f"{c}_pred_return" for c in oof.columns]
            if oof.empty:
                raise ValueError("No matured OOF observations available for bootstrap")
            empty=pd.DataFrame()
            weight=efp.build_time_decay_sample_weights(dataset.index,"1d",h)
            policy={"training_dataset":dataset,"feature_columns":features,"interval":"1d","horizon":h,
                "sample_weights":weight,"regression_oof_predictions":oof,"regression_leaderboard":reg_lb,
                "regression_backtest":empty,"recent_holdout_report":empty,"prediction_feedback_summary":None,
                "regression_interval_model":interval_model,"regression_interval_selection_basis":"frozen_issued_interval_choice",
                "best_regression_model":row["regression_model"],"regression_selection_basis":row["regression_selection_basis"],
                "best_classification_model":row["classification_model"],"classification_selection_basis":"frozen_issued_choice+historical_oof_context",
                "classification_threshold":float(row["classification_signal_threshold"]),"classification_leaderboard":cls_lb,
                "classification_backtest":empty,
                "best_regime_model":None if str(row["regime_model"]).startswith("heuristic") else row["regime_model"],
                "regime_selection_basis":"frozen_issued_choice",
                "best_reversal_model":None if str(row["reversal_model"]).startswith("heuristic") else row["reversal_model"],
                "reversal_selection_basis":"frozen_issued_choice"}
            models={}
            with final_model_cache("record",models):
                forecasts=efp.predict_frozen_policy(policy,prediction,efp.build_shared_price_reference(market,prefer_realtime=False))
            values={f.name:pd.DataFrame() for f in fields(efp.HorizonArtifacts)}
            values.update(horizon_steps=h,**dict(zip(("regression_forecast","classification_forecast","regime_forecast","reversal_forecast","hybrid_forecast"),forecasts)))
            values.update(regression_leaderboard=reg_lb,classification_leaderboard=cls_lb,regression_oof_predictions=oof,
                feature_coverage_summary=coverage,feature_snapshot=feature_frame.tail(1),
                data_window_summary=pd.DataFrame([{"latest_prediction_input_timestamp":iso_utc(market.index[-1]),
                    "training_rows":len(dataset),"used_feature_count":len(features)}]),
                regression_explainability=efp.empty_explainability_artifacts(row["regression_model"],"Daily inference excludes explainability"),
                classification_explainability=efp.empty_explainability_artifacts(row["classification_model"],"Daily inference excludes explainability"))
            horizons[h]={"policy":policy,"models":models,"candidate_feature_columns":candidates,"artifacts":efp.HorizonArtifacts(**values),
                         "evidence_cutoff_utc":iso_utc(oof.index.max()+pd.Timedelta(days=h)),
                         "evidence_sha256":file_sha256(evidence_path)}
            print(f"Bootstrap h={h} ready: {len(models)} fitted estimators, {len(features)} features",flush=True)
    return horizons
