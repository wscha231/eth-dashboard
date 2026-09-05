"""Predict both horizons with a trusted saved bundle; no fit or CV fallback."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from threadpoolctl import threadpool_limits
from forecasting.daily_pipeline import read_market, predict_bundle, summary_for_bundle
from forecasting.model_bundle import load_bundle
from forecasting.runtime_metrics import write_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", default="lake/gold/eth_master_daily.csv")
    parser.add_argument("--bundle", default="models/champion.joblib")
    parser.add_argument("--summary-csv", default="eth_forecast_outputs/latest_forecast_summary.csv")
    parser.add_argument("--metrics", default="eth_forecast_outputs/inference_runtime.json")
    args = parser.parse_args()
    try:
        with threadpool_limits(limits=2):
            market = read_market(args.master_data_csv)
            bundle = load_bundle(args.bundle)
            artifacts = predict_bundle(bundle, market)
            summary = summary_for_bundle(bundle, artifacts, args.master_data_csv)
            path = Path(args.summary_csv)
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            summary.to_csv(temp, index=False)
            temp.replace(path)
            print(summary[["horizon_steps", "forecast_input_timestamp", "forecast_target_timestamp", "regression_predicted_return", "classification_probability_up", "model_version"]].to_string(index=False))
    finally:
        write_metrics(args.metrics)


if __name__ == "__main__":
    main()
