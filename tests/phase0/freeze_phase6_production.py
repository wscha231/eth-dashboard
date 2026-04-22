"""Phase 6 production freeze: Phase 5 model (no algorithmic changes) measured
against a data-enriched master dataset.

What changed in Phase 6:
- `eth_data_collector.py` added three DefiLlama collectors (chain TVL,
  stablecoin supply total + pegged, DEX daily volume) producing ~3,000 days
  of crypto-native on-chain history.
- Glassnode scaffolding: activates automatically when GLASSNODE_API_KEY is
  set. Without a key the freeze still reflects the DefiLlama-only delta.
- Binance futures endpoints restored (User-Agent fix for fapi.binance.com
  + takerlongshortRatio rename + 25-day retention window) so OI / basis /
  funding / taker long-short rows start persisting daily.
- `eth_price_forecast.py` added `defillama_` and `glassnode_` to
  `GENERIC_VENDOR_PREFIXES`. Every new raw vendor column automatically
  generates ~11 derived features (__asof, __is_missing, __age_days,
  diff_{1,7}, pct_{1,7,30}, z_{30,90}, ma_30_ratio). No named regime
  composite is introduced — that's deferred behind a separate LOO gate.

The model architecture, selection logic, and Phase 5 allowlist are
UNCHANGED. This freeze isolates the pure data-enrichment delta so we can
decide whether Phase 6 moves on to named crypto-native regime composites
(stablecoin_supply_z, chain_tvl_momentum, etc.) guarded by a second LOO.

Compare against:
  - tests/phase0/phase5_production_metrics.json (same model, pre-Phase 6 master)

Output: tests/phase0/phase6_production_metrics.{pkl,json}.

Runtime note: full walk-forward CV over ~680 features at h=7 + h=30 takes
~50-70 min on a 2-core free-tier runner. Invoke locally on a machine with
full CPU before any GitHub-Actions promotion.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp

OUTPUT_PKL = Path(__file__).with_name("phase6_production_metrics.pkl")
OUTPUT_JSON = Path(__file__).with_name("phase6_production_metrics.json")

EQUAL_WEIGHT_REGRESSION_MODEL = "trimmed_regression_ensemble_equal"
EQUAL_WEIGHT_CLASSIFICATION_MODEL = "trimmed_classification_ensemble_equal"

# Phase 6 introduces no manual feature additions here — everything flows
# through `build_features` via the extended GENERIC_VENDOR_PREFIXES tuple.
# The Phase 5 macro allowlist (fred_real_yield_10y, fred_credit_stress) is
# already baked into `build_features`'s regime_feature_map and does not need
# to be re-applied here.


def _records(leaderboard: pd.DataFrame) -> list[dict[str, Any]]:
    if leaderboard is None or leaderboard.empty:
        return []
    return leaderboard.to_dict(orient="records")


def _append_equal_weight_regression_row(leaderboard, oof, dataset, members):
    available = [m for m in members if f"{m}_pred_return" in oof.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof[[f"{m}_pred_return" for m in available]]
    )
    valid = blended.dropna()
    if valid.empty:
        return leaderboard
    matched = dataset.loc[valid.index]
    summary = efp.evaluate_predictions(
        current_close=matched["eth_close"].to_numpy(),
        actual_return=matched["target_return"].to_numpy(),
        predicted_return=valid.to_numpy(),
    )
    summary["model"] = EQUAL_WEIGHT_REGRESSION_MODEL
    summary["folds"] = float(min(len(available), 4))
    summary["component_models"] = "|".join(available)
    return pd.concat([leaderboard, pd.DataFrame([summary])], ignore_index=True)


def _append_equal_weight_classification_row(leaderboard, oof, dataset, horizon, members):
    available = [m for m in members if f"{m}_prob_up" in oof.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof[[f"{m}_prob_up" for m in available]]
    ).clip(lower=0.0, upper=1.0)
    valid = blended.dropna()
    if valid.empty:
        return leaderboard
    actual = (dataset.loc[valid.index, "target_return"] > 0).astype(int)
    threshold, metrics = efp.choose_classification_evaluation_threshold(
        actual_label=actual, probability_up=valid, horizon=horizon,
    )
    metrics["model"] = EQUAL_WEIGHT_CLASSIFICATION_MODEL
    metrics["folds"] = float(min(len(available), 4))
    metrics["signal_threshold"] = float(threshold)
    metrics["component_models"] = "|".join(available)
    return pd.concat([leaderboard, pd.DataFrame([metrics])], ignore_index=True)


def _phase6_vendor_coverage(feature_frame: pd.DataFrame) -> dict[str, Any]:
    """Report how many Phase 6-added vendor columns and derived features
    cleared build_features and what their non-null coverage looks like.
    Written into the freeze metadata so we can pinpoint whether a metric
    change came from new-data signal or from model variance.
    """
    info: dict[str, Any] = {"raw_columns": {}, "derived_feature_count": 0}
    derived = 0
    for column in feature_frame.columns:
        if column.startswith(("defillama_", "glassnode_")):
            if any(column.endswith(suffix) for suffix in (
                "__asof", "__is_missing", "__age_days",
                "_diff_1", "_diff_7", "_pct_1", "_pct_7", "_pct_30",
                "_z_30", "_z_90", "_ma_30_ratio",
            )):
                derived += 1
            else:
                series = pd.to_numeric(feature_frame[column], errors="coerce")
                non_null = int(series.notna().sum())
                info["raw_columns"][column] = {
                    "non_null": non_null,
                    "coverage_ratio": float(non_null / len(feature_frame)) if len(feature_frame) else 0.0,
                    "start": str(series.dropna().index.min().date()) if non_null else "",
                    "end":   str(series.dropna().index.max().date()) if non_null else "",
                }
    info["derived_feature_count"] = derived
    return info


def run_inline(master_data_csv: Path) -> dict[str, Any]:
    market_data = efp.load_market_data_csv(master_data_csv)
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {master_data_csv}")

    state_cols = ["target_regime", "target_reversal_state", "target_bottom_reversal", "target_top_reversal"]
    horizons = [7, 30]
    cv_test_size = 60
    per_horizon: dict[int, dict[str, Any]] = {}

    for horizon in horizons:
        embargo = max(1, horizon // 2)
        feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
        feature_frame = feature_frame.copy()
        candidates = list(raw_candidates)

        # Record Phase 6 coverage once per horizon; useful for forensics.
        phase6_vendor_info = _phase6_vendor_coverage(feature_frame)

        full_mask = (
            feature_frame["target_return"].notna()
            & feature_frame["target_close"].notna()
            & feature_frame["eth_close"].notna()
        )
        full_dataset = feature_frame.loc[
            full_mask,
            candidates + ["eth_close", "target_return", "target_close", *state_cols],
        ].copy()
        trimmed = efp.trim_training_window(full_dataset, interval="1d", horizon=horizon)
        training_mask = feature_frame.index.isin(trimmed.index) & full_mask
        training_dataset = feature_frame.loc[
            training_mask,
            candidates + ["eth_close", "target_return", "target_close", *state_cols],
        ].copy()
        sample_weights = efp.build_time_decay_sample_weights(
            training_dataset.index, interval="1d", horizon=horizon,
        )

        reg_lb, reg_oof = efp.walk_forward_leaderboard(
            dataset=training_dataset, feature_columns=candidates,
            n_splits=3, test_size=cv_test_size, gap=horizon,
            sample_weight=sample_weights, verbose=False,
            fold_feature_selection=True, min_feature_coverage=0.03, embargo=embargo,
        )
        cls_lb, cls_oof = efp.walk_forward_classification(
            dataset=training_dataset, feature_columns=candidates,
            n_splits=3, test_size=cv_test_size, gap=horizon,
            sample_weight=sample_weights, verbose=False,
            fold_feature_selection=True, min_feature_coverage=0.03, embargo=embargo,
        )

        reg_members = efp.select_trimmed_regression_ensemble_members(
            reg_lb, horizon=horizon, oof_predictions=reg_oof,
        )
        cls_members = efp.select_trimmed_classification_ensemble_members(
            cls_lb, horizon=horizon, oof_predictions=cls_oof,
        )

        reg_lb, reg_oof = efp.append_trimmed_regression_ensemble_candidate(
            reg_lb, reg_oof, training_dataset,
            horizon=horizon, component_models=reg_members,
        )
        cls_lb, cls_oof = efp.append_trimmed_classification_ensemble_candidate(
            cls_lb, cls_oof, training_dataset,
            horizon=horizon, component_models=cls_members,
        )

        reg_lb = _append_equal_weight_regression_row(reg_lb, reg_oof, training_dataset, reg_members)
        cls_lb = _append_equal_weight_classification_row(
            cls_lb, cls_oof, training_dataset, horizon, cls_members,
        )

        per_horizon[horizon] = {
            "candidate_feature_count": len(candidates),
            "training_rows": int(len(training_dataset)),
            "cv_test_size": cv_test_size,
            "embargo": embargo,
            "regression_ensemble_members": reg_members,
            "classification_ensemble_members": cls_members,
            "regression_leaderboard": _records(reg_lb),
            "classification_leaderboard": _records(cls_lb),
            "phase6_vendor_coverage": phase6_vendor_info,
        }

    return {
        "frozen_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "master_data_csv": str(master_data_csv),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "mode": "phase6_production_phase5_model_plus_crypto_native_data",
        "cv_test_size": cv_test_size,
        "horizons": per_horizon,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True)
    args = parser.parse_args(argv)
    metrics = run_inline(Path(args.master_data_csv))
    OUTPUT_PKL.write_bytes(pickle.dumps(metrics))
    OUTPUT_JSON.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    print(f"[frozen] {OUTPUT_PKL}")
    print(f"[frozen] {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
