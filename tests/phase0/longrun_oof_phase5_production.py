"""Long-run OOF freeze of Phase 5 production — 36 folds × 30-day test chunks.

Unlike ``freeze_phase5_production.py`` (3-fold CV, summary metrics only), this
script records **per-date out-of-fold predictions across ~3 years** so the
frontend can show a real day-by-day track-record chart. Resumable: rerun with
``--resume`` and it picks up from the last flushed fold.

Key differences vs the 3-fold freeze:

  - ``n_splits=36, test_size=30`` (~3 years of OOF per horizon).
  - ``trim_training_window`` intentionally skipped — h=7's 3-year default
    window leaves too little training data for 36 folds. We use the full
    feature frame (~3,087 days) for both horizons so fold 0's training set
    still contains ~5 years of history.
  - Per-date predictions persisted to JSON between folds (atomic write).
  - Post-run leaderboard + ensemble rows computed from the accumulated OOF,
    matching the 3-fold freeze JSON shape so ``persist_backtest.py`` can
    ingest both formats with one code path.

Leakage guarantees are inherited from the fold runner — see
``longrun_oof_common.py`` module docstring.

Output: tests/phase0/phase5_production_longrun_oof_metrics.{json}
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402

# Reuse the Phase 5 LOO inline feature computation.
sys.path.insert(0, str(Path(__file__).parent))
from freeze_phase5_loo import _compute_phase3b_feature, PHASE3B_FEATURES  # noqa: E402
from longrun_oof_common import STATE_COLS, run_longrun  # noqa: E402

OUTPUT_JSON = Path(__file__).with_name("phase5_production_longrun_oof_metrics.json")

# Same allowlist as freeze_phase5_production (real_yield_10y, credit_stress).
PHASE5_ALLOWLIST: set[str] = {
    "fred_real_yield_10y",
    "fred_credit_stress",
}

N_SPLITS = 36
TEST_SIZE = 30


def build_horizon_payload(market_data, horizon: int) -> dict:
    embargo = max(1, horizon // 2)
    feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
    feature_frame = feature_frame.copy()
    candidates = list(raw_candidates)

    # Phase 5 macro allowlist — raw-level FRED series only.
    added: list[str] = []
    unknown = PHASE5_ALLOWLIST - set(PHASE3B_FEATURES)
    if unknown:
        raise SystemExit(f"PHASE5_ALLOWLIST contains unknown features: {unknown}")
    existing = set(candidates)
    for feature in sorted(PHASE5_ALLOWLIST):
        series = _compute_phase3b_feature(feature_frame, feature)
        if series is None:
            raise RuntimeError(f"Base columns missing for {feature}")
        feature_frame[feature] = series.reindex(feature_frame.index)
        if feature not in existing:
            candidates.append(feature)
            existing.add(feature)
        added.append(feature)

    # Keep ALL valid training rows (skip trim_training_window) so 36×30 folds fit.
    full_mask = (
        feature_frame["target_return"].notna()
        & feature_frame["target_close"].notna()
        & feature_frame["eth_close"].notna()
    )
    training_dataset = feature_frame.loc[
        full_mask,
        candidates + ["eth_close", "target_return", "target_close", *STATE_COLS],
    ].copy()
    sample_weights = efp.build_time_decay_sample_weights(
        training_dataset.index, interval="1d", horizon=horizon,
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
            "candidate_feature_count":  len(candidates),
            "phase5_allowlist_features": added,
            "training_rows":            int(len(training_dataset)),
            "embargo":                  embargo,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True)
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the last flushed fold if the checkpoint exists.")
    parser.add_argument("--flush-every", type=int, default=1,
                        help="Checkpoint every N folds (default 1 = flush after each).")
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Process at most N additional folds then exit cleanly "
                             "(useful for time-boxing a session; next run resumes).")
    args = parser.parse_args(argv)

    market_data = efp.load_market_data_csv(Path(args.master_data_csv))
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {args.master_data_csv}")

    horizon_payloads = {
        horizon: build_horizon_payload(market_data, horizon)
        for horizon in (7, 30)
    }

    state = run_longrun(
        checkpoint_path=OUTPUT_JSON,
        horizon_payloads=horizon_payloads,
        run_metadata={
            "mode":            "longrun_oof_phase5_production_36x30",
            "master_data_csv": str(args.master_data_csv),
            "phase5_allowlist": sorted(PHASE5_ALLOWLIST),
        },
        resume=args.resume,
        flush_every=args.flush_every,
        max_folds=args.max_folds,
    )

    done = state.get("folds_completed", {})
    target = state.get("folds_target", {})
    print(f"[longrun_oof_phase5_production] wrote {OUTPUT_JSON}")
    for h in sorted(target.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) else 0):
        print(f"  h={h}: folds {done.get(h, 0)}/{target.get(h, '?')} "
              f"({'partial' if state.get('partial') else 'complete'})")


if __name__ == "__main__":
    main()
