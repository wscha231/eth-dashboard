"""Long-run OOF freeze of Phase 5 **without** Phase 6 data enrichment —
36 folds × 30-day test chunks, identical to ``longrun_oof_phase5_production``
except we strip every ``defillama_*`` / ``glassnode_*`` column from the
candidate set.

Rationale / why this script exists
----------------------------------
The original 3-fold Phase 5 freeze (``phase5_production_metrics.json``,
frozen 2026-04-20) was run against a **pre-enrichment** master CSV with
**633 features**. The 3-fold Phase 6 freeze (frozen 2026-04-21) ran against
the post-enrichment master with **677 features** (+44 defillama_-derived
columns). That delta is what motivated Phase 6 in the first place.

When ``longrun_oof_phase5_production.py`` was re-run today on the *current*
master, ``GENERIC_VENDOR_PREFIXES`` auto-picked up all 4 defillama_ raw
columns → Phase 5 ended up at 677 features, byte-identical to Phase 6's
longrun. The intended A/B collapses to a determinism self-check. This
script restores the definition ("Phase 5 = the model without crypto-native
enrichment") by an explicit strip, so we get a real enrichment delta
measured over ~3 years of OOF.

Compare against:
  - phase6_production_longrun_oof_metrics.json (same model, WITH defillama_)

Output: tests/phase0/phase5_nodefi_longrun_oof_metrics.json
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
from freeze_phase5_loo import _compute_phase3b_feature, PHASE3B_FEATURES  # noqa: E402
from longrun_oof_common import STATE_COLS, run_longrun  # noqa: E402

OUTPUT_JSON = Path(__file__).with_name("phase5_nodefi_longrun_oof_metrics.json")

# Same allowlist as freeze_phase5_production (real_yield_10y, credit_stress).
PHASE5_ALLOWLIST: set[str] = {
    "fred_real_yield_10y",
    "fred_credit_stress",
}

# Feature prefixes that Phase 6 introduced. Stripping these from the
# candidate set reconstructs the original "Phase 5 data scope."
PHASE6_ENRICHMENT_PREFIXES: tuple[str, ...] = ("defillama_", "glassnode_")

N_SPLITS = 36
TEST_SIZE = 30


def build_horizon_payload(market_data, horizon: int) -> dict:
    embargo = max(1, horizon // 2)
    feature_frame, raw_candidates = efp.build_features(market_data, horizon=horizon)
    feature_frame = feature_frame.copy()

    # Strip Phase 6 enrichment columns. We leave the columns in the
    # feature_frame itself (harmless — they're just never referenced) but
    # remove them from the candidate list so select_fold_features never
    # considers them. Any defillama_-derived feature (e.g. z-scores, deltas)
    # that build_features emitted also starts with the same prefix.
    candidates = [
        c for c in raw_candidates
        if not c.startswith(PHASE6_ENRICHMENT_PREFIXES)
    ]
    stripped_count = len(raw_candidates) - len(candidates)

    # Phase 5 macro allowlist — raw-level FRED series only. Dedupe guard
    # mirrors longrun_oof_phase5_production.py (these may already be present
    # in raw_candidates via GENERIC_VENDOR_PREFIXES).
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
            "candidate_feature_count":      len(candidates),
            "phase5_allowlist_features":    added,
            "phase6_columns_stripped":      stripped_count,
            "training_rows":                int(len(training_dataset)),
            "embargo":                      embargo,
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-data-csv", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=1)
    parser.add_argument("--max-folds", type=int, default=None)
    args = parser.parse_args(argv)

    market_data = efp.load_market_data_csv(Path(args.master_data_csv))
    if market_data.empty:
        raise SystemExit(f"Master dataset is empty at {args.master_data_csv}")

    horizon_payloads = {
        horizon: build_horizon_payload(market_data, horizon)
        for horizon in (7, 30)
    }

    # Sanity print: if candidates unchanged, the strip had no effect (e.g.,
    # running against a master CSV that never had DefiLlama). We want to see
    # a clear reduction (~40 columns) before committing to the long wait.
    for horizon, payload in horizon_payloads.items():
        extras = payload["extras"]
        print(
            f"[longrun_oof_phase5_nodefi] h={horizon}: "
            f"{extras['candidate_feature_count']} candidates "
            f"(stripped {extras['phase6_columns_stripped']} phase-6 cols)"
        )

    state = run_longrun(
        checkpoint_path=OUTPUT_JSON,
        horizon_payloads=horizon_payloads,
        run_metadata={
            "mode":            "longrun_oof_phase5_nodefi_36x30",
            "master_data_csv": str(args.master_data_csv),
            "phase5_allowlist": sorted(PHASE5_ALLOWLIST),
            "phase6_columns_stripped": True,
        },
        resume=args.resume,
        flush_every=args.flush_every,
        max_folds=args.max_folds,
    )

    done = state.get("folds_completed", {})
    target = state.get("folds_target", {})
    print(f"[longrun_oof_phase5_nodefi] wrote {OUTPUT_JSON}")
    for h in sorted(target.keys(), key=lambda x: int(x) if isinstance(x, (int, str)) else 0):
        print(f"  h={h}: folds {done.get(h, 0)}/{target.get(h, '?')} "
              f"({'partial' if state.get('partial') else 'complete'})")


if __name__ == "__main__":
    main()
