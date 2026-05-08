"""Long-run OOF freeze of Phase 6 production — Phase 5 model + crypto-native
data enrichment, 36 folds × 30-day test chunks.

Identical to ``longrun_oof_phase5_production.py`` except the feature frame
comes from the Phase-6-enriched master CSV (DefiLlama stablecoin + TVL + DEX
volume, Glassnode when API key available). No model-level changes vs Phase 5;
this run isolates the pure data-enrichment delta over a ~3-year OOF window.

Compare against:
  - phase5_production_longrun_oof_metrics.json (same model, pre-Phase 6 master)

Output: tests/phase0/phase6_production_longrun_oof_metrics.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from longrun_oof_common import STATE_COLS, run_longrun  # noqa: E402

OUTPUT_JSON = Path(__file__).with_name("phase6_production_longrun_oof_metrics.json")

N_SPLITS = 36
TEST_SIZE = 30


def parse_horizons(raw: str) -> tuple[int, ...]:
    horizons: list[int] = []
    for part in str(raw).split(","):
        value = part.strip()
        if not value:
            continue
        try:
            horizon = int(value)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid horizon '{value}'") from exc
        if horizon not in {7, 30}:
            raise argparse.ArgumentTypeError("supported horizons are 7 and 30")
        if horizon not in horizons:
            horizons.append(horizon)
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return tuple(horizons)


def build_horizon_payload(market_data, horizon: int) -> dict:
    embargo = max(1, horizon // 2)
    feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
    feature_frame = feature_frame.copy()
    candidates = list(raw_candidates)

    # Phase 6 introduces no manual feature additions — GENERIC_VENDOR_PREFIXES
    # picks up the new defillama_/glassnode_ columns via build_features.
    full_mask = (
        feature_frame["target_return"].notna()
        & feature_frame["target_close"].notna()
        & feature_frame["eth_close"].notna()
    )
    training_dataset = feature_frame.loc[
        full_mask,
        candidates + ["eth_close", "target_return", "target_close", *efp.DIRECTION_TARGET_COLUMNS, *STATE_COLS],
    ].copy()
    sample_weights = efp.build_time_decay_sample_weights(
        training_dataset.index, interval="1d", horizon=horizon,
    )

    # Count Phase 6 vendor columns that cleared build_features for forensics.
    phase6_vendor_columns = sum(
        1 for c in candidates
        if c.startswith(("defillama_", "glassnode_"))
    )

    return {
        "dataset":              training_dataset,
        "feature_columns":      candidates,
        "sample_weights":       sample_weights,
        "n_splits":             N_SPLITS,
        "test_size":            TEST_SIZE,
        "gap":                  horizon,
        "embargo":              embargo,
        "min_feature_coverage": 0.03,
        "extras": {
            "candidate_feature_count":      len(candidates),
            "phase6_vendor_feature_count":  phase6_vendor_columns,
            "training_rows":                int(len(training_dataset)),
            "direction_actionable_rate":     efp.direction_target_actionable_rate(training_dataset, horizon),
            "direction_actionable_rows":     int(
                efp.get_direction_classification_target(training_dataset, horizon).notna().sum()
            ),
            "embargo":                      embargo,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    parser.add_argument(
        "--fold-start",
        type=int,
        default=0,
        help="Zero-based first fold to run. Used by GitHub Actions chunked full evaluation.",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons("7,30"),
        help="Comma-separated horizons to run. Supported: 7,30.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=OUTPUT_JSON,
        help="Checkpoint/output JSON path.",
    )
    args = parser.parse_args(argv)

    market_data = efp.load_market_data_csv(Path(args.master_data_csv))
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {args.master_data_csv}")

    horizon_payloads = {
        horizon: build_horizon_payload(market_data, horizon)
        for horizon in args.horizons
    }

    state = run_longrun(
        checkpoint_path=args.output_json,
        horizon_payloads=horizon_payloads,
        run_metadata={
            "mode":            "longrun_oof_phase6_production_36x30",
            "master_data_csv": str(args.master_data_csv),
            "horizons_requested": ",".join(str(h) for h in args.horizons),
            "fold_start":      int(max(args.fold_start, 0)),
            "max_folds":       args.max_folds,
        },
        resume=args.resume,
        flush_every=args.flush_every,
        max_folds=args.max_folds,
        fold_start=args.fold_start,
    )

    done = state.get("folds_completed", {})
    target = state.get("folds_target", {})
    print(f"[longrun_oof_phase6_production] wrote {args.output_json}")
    for h in sorted(target.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) else 0):
        print(f"  h={h}: folds {done.get(h, 0)}/{target.get(h, '?')} "
              f"({'partial' if state.get('partial') else 'complete'})")


if __name__ == "__main__":
    main()
