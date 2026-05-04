"""Merge horizon-split long-run OOF metric JSON files.

GitHub-hosted runners cannot reliably finish the 7d and 30d 36-fold freezes
sequentially in one job. The full evaluation workflow therefore runs each
horizon in a separate job and merges the resulting checkpoint JSONs before the
threshold sweep and candidate gate.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Expected object JSON: {path}")
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    merged: dict[str, Any] | None = None
    seen_horizons: set[str] = set()

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
            if horizon_key in seen_horizons:
                raise SystemExit(f"Duplicate horizon {horizon_key} in {path}")
            seen_horizons.add(horizon_key)
            merged["horizons"][horizon_key] = payload

        folds_completed = state.get("folds_completed") or {}
        folds_target = state.get("folds_target") or {}
        for horizon in horizons:
            horizon_key = str(horizon)
            merged["folds_completed"][horizon_key] = folds_completed.get(horizon_key, folds_completed.get(int(horizon_key), 0))
            merged["folds_target"][horizon_key] = folds_target.get(horizon_key, folds_target.get(int(horizon_key), 0))

    if merged is None:
        raise SystemExit("No inputs provided")

    if seen_horizons != {"7", "30"}:
        raise SystemExit(f"Expected horizons 7 and 30, got {sorted(seen_horizons)}")

    merged["last_checkpoint_utc"] = dt.datetime.now(tz=dt.timezone.utc).isoformat()
    merged["horizons_requested"] = "7,30"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"Wrote merged longrun metrics: {args.output}")


if __name__ == "__main__":
    main()
