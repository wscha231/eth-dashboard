"""Merge horizon/fold-split long-run OOF metric JSON files.

GitHub-hosted runners cannot reliably finish the 7d and 30d 36-fold freezes
in one job. The full evaluation workflow therefore runs each horizon/fold
chunk in a separate job and merges the resulting checkpoint JSONs before the
threshold sweep and candidate gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402
from longrun_oof_common import (  # noqa: E402
    EQUAL_WEIGHT_CLASSIFICATION_MODEL,
    EQUAL_WEIGHT_REGRESSION_MODEL,
    FoldRunner,
    finalize_run,
)
from longrun_oof_phase6_production import build_horizon_payload  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON: {path}")
    return payload


def prediction_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("horizon_days"),
        row.get("head"),
        row.get("model"),
        row.get("prediction_date"),
        row.get("fold_index"),
    )


def is_base_prediction_row(row: dict[str, Any]) -> bool:
    model = str(row.get("model") or "")
    return model not in {EQUAL_WEIGHT_REGRESSION_MODEL, EQUAL_WEIGHT_CLASSIFICATION_MODEL}


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def filter_rows_to_dataset(rows: list[dict[str, Any]], index) -> list[dict[str, Any]]:
    valid_index = set(index)
    filtered: list[dict[str, Any]] = []
    for row in rows:
        try:
            prediction_date = pd_timestamp(row.get("prediction_date"))
        except (TypeError, ValueError):
            continue
        if prediction_date in valid_index:
            filtered.append(row)
    return filtered


def pd_timestamp(value: Any):
    import pandas as pd

    return pd.Timestamp(value)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--finalize-master-data-csv",
        type=Path,
        default=None,
        help="Rebuild final leaderboards/ensembles from merged fold chunks using this master CSV.",
    )
    args = parser.parse_args(argv)

    merged: dict[str, Any] | None = None
    rows_by_horizon: dict[str, dict[tuple[Any, ...], dict[str, Any]]] = defaultdict(dict)
    extras_by_horizon: dict[str, dict[str, Any]] = {}
    folds_target_by_horizon: dict[str, int] = {}
    sources_by_horizon: dict[str, list[str]] = defaultdict(list)

    for path in args.inputs:
        state = load_json(path)
        horizons = state.get("horizons") or {}
        if not horizons:
            raise SystemExit(f"No horizons found in {path}")

        if merged is None:
            merged = {
                key: value
                for key, value in state.items()
                if key not in {"horizons", "folds_completed", "folds_target"}
            }
            merged["mode"] = state.get("mode", "longrun_oof_phase6_production_36x30")
            merged["partial"] = False
            merged["folds_completed"] = {}
            merged["folds_target"] = {}
            merged["horizons"] = {}
            merged["merged_from"] = []

        merged["partial"] = bool(merged.get("partial")) or bool(state.get("partial"))
        merged["cv_test_size"] = max(int(merged.get("cv_test_size") or 0), int(state.get("cv_test_size") or 0))
        merged["n_splits"] = max(int(merged.get("n_splits") or 0), int(state.get("n_splits") or 0))
        merged["merged_from"].append(str(path))

        for horizon, payload in horizons.items():
            horizon_key = str(horizon)
            sources_by_horizon[horizon_key].append(str(path))
            extras_by_horizon.setdefault(
                horizon_key,
                {
                    key: value
                    for key, value in payload.items()
                    if key not in {
                        "predictions",
                        "regression_leaderboard",
                        "classification_leaderboard",
                        "regression_models",
                        "classification_models",
                    }
                },
            )
            for row in payload.get("predictions") or []:
                if not isinstance(row, dict) or not is_base_prediction_row(row):
                    continue
                rows_by_horizon[horizon_key][prediction_key(row)] = row

        folds_completed = state.get("folds_completed") or {}
        folds_target = state.get("folds_target") or {}
        for horizon in horizons:
            horizon_key = str(horizon)
            folds_target_by_horizon[horizon_key] = max(
                folds_target_by_horizon.get(horizon_key, 0),
                safe_int(folds_target.get(horizon_key, folds_target.get(int(horizon_key), 0))),
            )

    if merged is None:
        raise SystemExit("No inputs provided")

    seen_horizons = set(rows_by_horizon)
    if seen_horizons != {"7", "30"}:
        raise SystemExit(f"Expected horizons 7 and 30, got {sorted(seen_horizons)}")

    if args.finalize_master_data_csv is None:
        for horizon_key in sorted(rows_by_horizon, key=int):
            rows = list(rows_by_horizon[horizon_key].values())
            fold_indices = {
                safe_int(row.get("fold_index"), -1)
                for row in rows
                if safe_int(row.get("fold_index"), -1) >= 0
            }
            merged["folds_completed"][horizon_key] = len(fold_indices)
            merged["folds_target"][horizon_key] = folds_target_by_horizon.get(horizon_key, len(fold_indices))
            merged["horizons"][horizon_key] = {
                **extras_by_horizon.get(horizon_key, {}),
                "merged_from": sources_by_horizon[horizon_key],
                "predictions": rows,
            }
    else:
        market_data = efp.load_market_data_csv(args.finalize_master_data_csv)
        if market_data.empty:
            raise SystemExit(f"Master dataset is empty at {args.finalize_master_data_csv}")
        merged["horizons"] = {}
        for horizon_key in sorted(rows_by_horizon, key=int):
            horizon = int(horizon_key)
            rows = list(rows_by_horizon[horizon_key].values())
            horizon_payload = build_horizon_payload(market_data, horizon)
            rows = filter_rows_to_dataset(rows, horizon_payload["dataset"].index)
            runner = FoldRunner(
                dataset=horizon_payload["dataset"],
                feature_columns=horizon_payload["feature_columns"],
                sample_weights=horizon_payload.get("sample_weights"),
                horizon=horizon,
                n_splits=horizon_payload["n_splits"],
                test_size=horizon_payload["test_size"],
                gap=horizon_payload["gap"],
                embargo=horizon_payload["embargo"],
                min_feature_coverage=horizon_payload.get("min_feature_coverage", 0.03),
            )
            runner.restore_oof_from_rows(rows)
            prediction_map = {horizon: rows}
            summary = finalize_run(runner, prediction_map)
            finalized_rows = prediction_map[horizon]
            fold_indices = {
                safe_int(row.get("fold_index"), -1)
                for row in finalized_rows
                if safe_int(row.get("fold_index"), -1) >= 0 and is_base_prediction_row(row)
            }
            target_folds = folds_target_by_horizon.get(horizon_key, runner.n_splits)
            merged["folds_completed"][horizon_key] = len(fold_indices)
            merged["folds_target"][horizon_key] = target_folds
            merged["horizons"][horizon_key] = {
                **horizon_payload.get("extras", {}),
                **summary,
                "merged_from": sources_by_horizon[horizon_key],
                "predictions": finalized_rows,
            }
        merged["partial"] = any(
            safe_int(merged["folds_completed"].get(str(h)), 0) < safe_int(merged["folds_target"].get(str(h)), 0)
            for h in (7, 30)
        )

    merged["last_checkpoint_utc"] = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    merged["horizons_requested"] = "7,30"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"Wrote merged longrun metrics: {args.output}")


if __name__ == "__main__":
    main()
