"""Analyze phase5_loo_metrics.json and produce a culprit report.

For each of the 6 Phase 3B composites, compare the variant's h=7 and h=30
metrics against baseline (Phase 4B) and classify as:
  - SAFE     : delta within tolerance -> candidate for Phase 5 re-inclusion
  - WATCH    : mild regression on one metric -> marginal case
  - CULPRIT  : multiple metrics regress meaningfully

Usage:
  python analyze_phase5_loo.py
  python analyze_phase5_loo.py --input phase5_loo_metrics.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_INPUT = Path(__file__).with_name("phase5_loo_metrics.json")

# Regression tolerance — anything worse than these deltas from baseline triggers
# a WATCH/CULPRIT flag on that metric.
TOL_BRIER_ABS = 0.005       # +0.005 on brier_score (small but meaningful)
TOL_AUC_ABS = 0.02          # -0.02 on AUC
TOL_RMSE_PCT = 0.04         # +4% on price_rmse


def _best_single(leaderboard: list[dict], metric: str, lower_is_better: bool = True) -> dict | None:
    singles = [
        r for r in leaderboard
        if "ensemble" not in r["model"]
        and "trimmed_equal_weight" not in r["model"]
    ]
    if not singles:
        return None
    singles.sort(key=lambda r: r.get(metric, 1e9 if lower_is_better else -1e9),
                 reverse=not lower_is_better)
    return singles[0]


def _find(leaderboard: list[dict], model_name: str) -> dict | None:
    for r in leaderboard:
        if r["model"] == model_name:
            return r
    return None


def _summarize_variant(variant: dict, horizon: str) -> dict:
    data = variant[horizon]
    cls_lb = data["classification_leaderboard"]
    reg_lb = data["regression_leaderboard"]

    cls_single = _best_single(cls_lb, "brier_score", lower_is_better=True)
    reg_single = _best_single(reg_lb, "price_rmse", lower_is_better=True)

    cls_ens = _find(cls_lb, "trimmed_classification_ensemble")
    reg_ens = _find(reg_lb, "trimmed_regression_ensemble")

    return {
        "horizon": horizon,
        "cls_members": data.get("classification_ensemble_members") or [],
        "reg_members": data.get("regression_ensemble_members") or [],
        "cls_best_single_model": cls_single["model"] if cls_single else None,
        "cls_best_single_brier": cls_single["brier_score"] if cls_single else None,
        "cls_best_single_auc": cls_single.get("roc_auc") if cls_single else None,
        "cls_ens_brier": cls_ens["brier_score"] if cls_ens else None,
        "cls_ens_auc": cls_ens.get("roc_auc") if cls_ens else None,
        "reg_best_single_model": reg_single["model"] if reg_single else None,
        "reg_best_single_rmse": reg_single["price_rmse"] if reg_single else None,
    }


def _classify(delta: dict) -> str:
    """Given per-horizon deltas dict (brier_delta, auc_delta, rmse_pct_delta),
    return SAFE / WATCH / CULPRIT.
    """
    strikes = 0
    severe = 0

    for key, tol, is_higher_bad in [
        ("brier_delta", TOL_BRIER_ABS, True),
        ("auc_delta", -TOL_AUC_ABS, False),   # AUC: lower is bad
        ("rmse_pct_delta", TOL_RMSE_PCT, True),
    ]:
        value = delta.get(key)
        if value is None:
            continue
        if is_higher_bad:
            if value > tol * 2:
                severe += 1
            elif value > tol:
                strikes += 1
        else:
            if value < tol * 2:
                severe += 1
            elif value < tol:
                strikes += 1

    if severe >= 1 or strikes >= 2:
        return "CULPRIT"
    if strikes == 1:
        return "WATCH"
    return "SAFE"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        metrics = json.load(f)

    variants = metrics["variants"]
    if "baseline" not in variants:
        raise SystemExit("No baseline variant in metrics; re-run without --skip-baseline")

    baseline = variants["baseline"]
    baseline_summary = {h: _summarize_variant(baseline, h) for h in ("7", "30")}

    print("=" * 78)
    print(" Phase 5 Leave-One-In ablation report")
    print("=" * 78)
    print(f"Source: {args.input}")
    print(f"Frozen at: {metrics.get('frozen_at')}")
    print()

    # Baseline header
    for h in ("7", "30"):
        b = baseline_summary[h]
        print(f"[baseline h={h}] cls_members={b['cls_members']}")
        print(f"               cls best_single={b['cls_best_single_model']} "
              f"brier={b['cls_best_single_brier']:.4f} auc={b['cls_best_single_auc']:.4f}")
        print(f"               reg best_single={b['reg_best_single_model']} rmse={b['reg_best_single_rmse']:.1f}")
    print()

    # Per-variant analysis
    print(f"{'feature':40s} {'h':>2s} {'verdict':>8s}  {'Δbrier':>8s}  {'Δauc':>7s}  {'Δrmse%':>7s}  cls_members_change")
    print("-" * 115)

    verdicts: dict[str, dict[str, str]] = {}
    for variant_name, variant in variants.items():
        if variant_name == "baseline":
            continue
        feature = variant_name.removeprefix("add_")
        verdicts[feature] = {}
        for h in ("7", "30"):
            s = _summarize_variant(variant, h)
            b = baseline_summary[h]
            delta = {
                "brier_delta": (s["cls_best_single_brier"] - b["cls_best_single_brier"])
                    if s["cls_best_single_brier"] and b["cls_best_single_brier"] else None,
                "auc_delta": (s["cls_best_single_auc"] - b["cls_best_single_auc"])
                    if s["cls_best_single_auc"] and b["cls_best_single_auc"] else None,
                "rmse_pct_delta": ((s["reg_best_single_rmse"] - b["reg_best_single_rmse"]) / b["reg_best_single_rmse"])
                    if s["reg_best_single_rmse"] and b["reg_best_single_rmse"] else None,
            }
            verdict = _classify(delta)
            verdicts[feature][h] = verdict

            members_changed = "same" if s["cls_members"] == b["cls_members"] else f"→ {s['cls_members']}"

            print(f"{feature:40s} {h:>2s} {verdict:>8s}  "
                  f"{delta['brier_delta']:+.4f}  "
                  f"{delta['auc_delta']:+.4f}  "
                  f"{delta['rmse_pct_delta']*100:+6.2f}%  {members_changed}")
        print()

    # Summary: pool features by verdict (using worst across horizons)
    print("=" * 78)
    print(" Classification by worst-horizon verdict:")
    print("=" * 78)
    order = {"SAFE": 0, "WATCH": 1, "CULPRIT": 2}
    worst_overall = {}
    for feature, h_verdicts in verdicts.items():
        worst = max(h_verdicts.values(), key=lambda v: order[v])
        worst_overall[feature] = worst

    for tier in ("SAFE", "WATCH", "CULPRIT"):
        features = [f for f, v in worst_overall.items() if v == tier]
        print(f"  {tier:8s}: {features}")

    print()
    print("Recommendation: re-include SAFE features, revisit WATCH with additional")
    print("data, permanently exclude CULPRIT features. Combined re-inclusion")
    print("(all SAFE together) must be verified in a follow-up freeze to catch")
    print("interaction effects.")


if __name__ == "__main__":
    main()
