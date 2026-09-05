"""Train a versioned production-method bundle; daily prediction never trains."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadpoolctl import threadpool_limits
from forecasting.daily_pipeline import read_market, train_bundle, predict_bundle, summary_for_bundle
from forecasting.model_bundle import load_bundle, save_bundle
from forecasting.runtime_metrics import write_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", default="lake/gold/eth_master_daily.csv")
    parser.add_argument("--bundle", default="models/champion.joblib")
    parser.add_argument("--bootstrap", action="store_true", help="Rebuild from issued choices and frozen OOF evidence; never run on inference failure")
    parser.add_argument("--policy-db", default="forecast_site/predictions.db")
    parser.add_argument("--evidence", default="tests/phase0/phase6_production_longrun_oof_metrics.json")
    parser.add_argument("--full-search", action="store_true", help="Explicit expensive research search; bootstrap normally reuses issued choices")
    parser.add_argument("--metrics", default="eth_forecast_outputs/training_runtime.json")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            market = read_market(args.master_data_csv)
            previous = None if args.bootstrap else load_bundle(args.bundle)
            bundle = train_bundle(market, args.master_data_csv, previous=previous,
                                  policy_db=args.policy_db, evidence_path=args.evidence, full_search=args.full_search)
            # Verify both horizons before replacing a known-good bundle.
            bundle["metadata"]["model_version"] = "validation"
            artifacts = predict_bundle(bundle, market)
            summary = summary_for_bundle(bundle, artifacts, args.master_data_csv)
            if summary["regression_predicted_close"].isna().any() or (summary["regression_predicted_close"] <= 0).any():
                raise ValueError("Invalid point forecast in trained bundle")
            manifest = save_bundle(bundle, args.bundle)
            print(f"Bundle ready: {manifest['model_version']}; training mode={manifest['training_mode']}", flush=True)
    finally:
        write_metrics(args.metrics)


if __name__ == "__main__":
    main()
