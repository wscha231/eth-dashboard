from __future__ import annotations

import argparse
import json
from pathlib import Path

from eth_price_forecast import (
    CACHE_UPDATE_LOOKBACK_ROWS,
    DEFAULT_FEATURE_MIN_COVERAGE,
    DEFAULT_MASTER_DATA_CSV,
    run_pipeline,
)


HORIZON = 7


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the dedicated 7-day ETH forecast from the master dataset.")
    parser.add_argument("--period", default="max")
    parser.add_argument("--interval", default="1d")
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--cv-test-size", type=int, default=20)
    parser.add_argument("--dataset-csv", default="market_data_cache.csv")
    parser.add_argument("--master-data-csv", default=DEFAULT_MASTER_DATA_CSV)
    parser.add_argument("--prediction-history-csv", default="")
    parser.add_argument("--drive-root", default="")
    parser.add_argument("--external-features-csv", nargs="*", default=[])
    parser.add_argument("--cache-lookback-rows", type=int, default=CACHE_UPDATE_LOOKBACK_ROWS)
    parser.add_argument("--min-feature-coverage", type=float, default=DEFAULT_FEATURE_MIN_COVERAGE)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--no-dataset-cache", action="store_true")
    parser.add_argument("--no-prediction-history-features", action="store_true")
    parser.add_argument("--full-output", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args, _ = parser.parse_known_args(argv)
    return args


def resolve_output_dir(output_dir: str, drive_root: str) -> str:
    if output_dir:
        return output_dir
    if drive_root:
        return str(Path(drive_root) / "outputs" / "7d")
    return "eth_forecast_outputs_7d"


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = run_pipeline(
        period=args.period,
        interval=args.interval,
        horizon=HORIZON,
        cv_splits=args.cv_splits,
        cv_test_size=args.cv_test_size,
        output_dir=resolve_output_dir(args.output_dir, args.drive_root),
        dataset_csv=args.dataset_csv,
        master_data_csv=(args.master_data_csv or None),
        prediction_history_csv=(args.prediction_history_csv or None),
        external_feature_csvs=(args.external_features_csv or None),
        drive_root=(args.drive_root or None),
        use_dataset_cache=not args.no_dataset_cache,
        use_prediction_history_features=not args.no_prediction_history_features,
        cache_lookback_rows=args.cache_lookback_rows,
        min_feature_coverage=args.min_feature_coverage,
        compact_output=not args.full_output,
        save_outputs=not args.no_save,
        verbose=(not args.quiet) and (not args.json),
    )

    payload = {
        "horizon_steps": HORIZON,
        "regression_model": artifact.regression_forecast.model_name,
        "regression_selection_basis": artifact.regression_forecast.selection_basis,
        "predicted_close": artifact.regression_forecast.predicted_close,
        "predicted_return": artifact.regression_forecast.predicted_return,
        "direction_model": artifact.classification_forecast.model_name,
        "direction_selection_basis": artifact.classification_forecast.selection_basis,
        "predicted_direction": artifact.classification_forecast.predicted_direction,
        "signal_threshold": artifact.classification_forecast.signal_threshold,
        "probability_up": artifact.classification_forecast.probability_up,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("ETH 7-day forecast")
    print(
        f"- Regression: model={payload['regression_model']}, "
        f"selection_basis={payload['regression_selection_basis']}, "
        f"predicted_close={payload['predicted_close']}, "
        f"predicted_return={payload['predicted_return']}"
    )
    print(
        f"- Direction: model={payload['direction_model']}, "
        f"selection_basis={payload['direction_selection_basis']}, "
        f"prediction={payload['predicted_direction']}, "
        f"signal_threshold={payload['signal_threshold']}, "
        f"prob_up={payload['probability_up']}"
    )


if __name__ == "__main__":
    main()
