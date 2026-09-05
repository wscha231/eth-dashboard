"""Closed-bar daily inference; training is an explicit, separate operation."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import eth_price_forecast as efp
from forecasting.daily_data import TIME_CONTRACT, file_sha256, iso_utc, model_daily_rows, utc_timestamp
from forecasting.feature_store import fold_feature_cache
from forecasting.model_bundle import final_model_cache, frame_hash
from forecasting.runtime_metrics import stage

FORECAST_FIELDS = ("regression_forecast", "classification_forecast", "regime_forecast", "reversal_forecast", "hybrid_forecast")


def read_market(path, *, now=None, require_fresh=True):
    with stage("read_closed_bars"):
        return model_daily_rows(efp.load_market_data_csv(path), now=now, require_fresh=require_fresh)


def latest_frame(market, policy):
    features, _ = efp.build_features(market, horizon=policy["horizon"])
    columns = policy["feature_columns"]
    missing = set(columns) - set(features)
    if missing:
        raise ValueError(f"Bundle features missing from input: {sorted(missing)[:5]}")
    return features, features.loc[features["eth_close"].notna(), list(dict.fromkeys(columns + ["eth_close"]))]


def infer_horizon(item, market):
    policy = item["policy"]
    features, prediction_frame = latest_frame(market, policy)
    reference = efp.build_shared_price_reference(market, prefer_realtime=False)
    with stage(f"h{policy['horizon']}_inference"), final_model_cache("predict", item["models"]):
        forecasts = efp.predict_frozen_policy(policy, prediction_frame, reference)
    artifact = item["artifacts"]
    window = artifact.data_window_summary.copy()
    window["latest_prediction_input_timestamp"] = iso_utc(market.index[-1])
    window["prediction_target_timestamp"] = iso_utc(market.index[-1] + pd.Timedelta(days=policy["horizon"]))
    return replace(artifact, **dict(zip(FORECAST_FIELDS, forecasts)),
                   data_window_summary=window, feature_snapshot=features.tail(1))


def predict_bundle(bundle, market, *, now=None):
    meta = bundle["metadata"]
    if set(bundle["horizons"]) != {7, 30}:
        raise ValueError("Bundle must contain both horizons")
    training_cutoff = utc_timestamp(meta["training_cutoff_utc"])
    if utc_timestamp(market.index[-1]) < training_cutoff:
        raise ValueError("Inference input predates the fitted bundle")
    if utc_timestamp(now) - training_cutoff > pd.Timedelta(days=8):
        raise ValueError("Bundle training cutoff is older than 8 days")
    results = {h: infer_horizon(bundle["horizons"][h], market) for h in (7, 30)}
    return efp.PipelineArtifacts(raw_data=market, horizons=results)


def summary_for_bundle(bundle, artifacts, source_path):
    summary = efp.build_latest_forecast_summary(artifacts)
    summary["time_contract"] = TIME_CONTRACT
    summary["model_version"] = bundle["metadata"]["model_version"]
    summary["training_cutoff_utc"] = bundle["metadata"]["training_cutoff_utc"]
    summary["data_hash"] = file_sha256(source_path)
    summary["source_bar_date"] = str((artifacts.raw_data.index[-1] - pd.Timedelta(days=1)).date())
    summary["classification_probability_event"] = "up_conditional_on_large_move"
    daily_vol = artifacts.raw_data["eth_close"].pct_change(fill_method=None).tail(30).std()
    for idx, row in summary.iterrows():
        h = int(row["horizon_steps"])
        multiplier, floor, cap = efp.classification_direction_threshold_bounds(h)
        summary.loc[idx, "classification_event_threshold"] = float(np.clip(daily_vol * np.sqrt(h) * multiplier, floor, cap))
    for column in ("forecast_input_timestamp", "forecast_target_timestamp", "reference_price_timestamp"):
        summary[column] = summary[column].map(iso_utc)
    return summary


def train_bundle(market, source_path, *, previous=None, cv_splits=3, cv_test_size=30,
                 policy_db=None, evidence_path=None, full_search=False):
    """Bootstrap once, then refit the frozen policy without CV/model search.

    Refit retains the versioned OOF policy evidence; this is not new OOS proof.
    Classifier calibration is fitted together with each new base classifier.
    """
    horizons = {}
    if previous is None and not full_search:
        from forecasting.production_bootstrap import bootstrap_issued_policy
        if policy_db is None or evidence_path is None:
            raise ValueError("Bootstrap needs issued policy DB and frozen OOF evidence")
        horizons = bootstrap_issued_policy(market, str(Path(policy_db).resolve()), Path(evidence_path))
    elif previous is None:
        with fold_feature_cache():
            for h in (7, 30):
                with stage(f"h{h}_bootstrap_selection"):
                    artifact = efp.run_horizon_pipeline(market_data=market, interval="1d", horizon=h,
                        cv_splits=cv_splits, cv_test_size=cv_test_size, use_realtime_reference_price=False,
                        build_explainability=False, bundle_record=horizons)
                    horizons[h]["artifacts"] = artifact
    else:
        # Refit the chosen settings only. No hidden model search or changes to
        # feature order, thresholds, blend weights, or evaluation evidence.
        for h in (7, 30):
            item = previous["horizons"][h]
            policy = dict(item["policy"])
            features, prediction_frame = latest_frame(market, policy)
            old_columns = list(policy["training_dataset"].columns)
            dataset = features.loc[features["target_return"].notna() & features["target_close"].notna(), old_columns]
            dataset = efp.trim_training_window(dataset, interval="1d", horizon=h)
            policy["training_dataset"] = dataset
            policy["sample_weights"] = efp.build_time_decay_sample_weights(dataset.index, "1d", h)
            models = {}
            with stage(f"h{h}_selected_refit"), final_model_cache("record", models):
                forecasts = efp.predict_frozen_policy(policy, prediction_frame, efp.build_shared_price_reference(market, prefer_realtime=False))
            artifact = replace(item["artifacts"], **dict(zip(FORECAST_FIELDS, forecasts)))
            horizons[h] = {**item, "policy": policy, "models": models, "artifacts": artifact}
    return {"horizons": horizons, "metadata": {
        "training_cutoff_utc": iso_utc(market.index[-1]),
        "last_label_available_at_utc": iso_utc(market.index[-1]),
        "data_manifest_hash": file_sha256(source_path), "time_contract": TIME_CONTRACT,
        "feature_schema_hash": frame_hash(pd.DataFrame({str(h): pd.Series(horizons[h]["policy"]["feature_columns"]) for h in (7, 30)})),
        "policy_evidence_cutoff_utc": previous["metadata"]["policy_evidence_cutoff_utc"] if previous else max(
            horizons[h].get("evidence_cutoff_utc", iso_utc(market.index[-1])) for h in (7, 30)),
        "policy_evidence_sha256": previous["metadata"].get("policy_evidence_sha256") if previous else
            horizons[7].get("evidence_sha256"),
        "training_mode": "selected_refit" if previous else ("bootstrap_selection" if full_search else "issued_choices_existing_evidence"),
        "promotion": "existing_method_only; new_predictive_edge_unverified",
    }}
