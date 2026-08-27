"""Threshold sweep over Phase 6 longrun OOF classification predictions.

The standard Phase 6 freeze picks a single signal threshold per model
(typically 0.60-0.65). At that threshold we get ~55% accuracy on the
signal-emitted subset. The user wants to know what threshold (if any) gets
us to 70% accuracy, and what fraction of days actually emit a signal at that
tighter threshold.

This script answers that empirically without retraining anything — it just
re-evaluates the same OOF probabilities at a grid of thresholds.

For each (horizon, model, threshold), we compute:
    n_signal_days       — days where prob >= thr OR prob <= (1 - thr)
                           ("two-sided" — confident UP or confident DOWN)
    n_total             — total OOF predictions
    signal_pct_days     — n_signal_days / n_total
    accuracy_on_signals — fraction correct on signal subset
    coverage_per_year   — extrapolated signal days per 365 days

Output: prints leaderboard + writes JSON to threshold_sweep_results.json

Usage:
    python tests/phase0/threshold_sweep.py
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

LONGRUN_JSON = Path(__file__).with_name("phase6_production_longrun_oof_metrics.json")
OUTPUT_JSON = Path(__file__).with_name("threshold_sweep_results.json")

THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95]


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float | None:
    """95% Wilson lower bound as a percentage, for robust threshold ranking."""
    if total <= 0:
        return None
    p = successes / total
    denom = 1.0 + (z * z / total)
    centre = p + (z * z / (2.0 * total))
    margin = z * ((p * (1.0 - p) / total + z * z / (4.0 * total * total)) ** 0.5)
    return round(100 * ((centre - margin) / denom), 2)


def sweep_one_model(predictions: list[dict], horizon: int, model: str) -> list[dict]:
    """Return one row per threshold for (horizon, model)."""
    rows = [
        p for p in predictions
        if p.get("horizon_days") == horizon
        and p.get("head") == "classification"
        and p.get("model") == model
        and p.get("probability_up") is not None
        and p.get("actual_label") is not None
    ]
    n_total = len(rows)
    if n_total == 0:
        return []

    # Base rates over the full OOF. New artifacts keep the directional rank
    # separate from the label-calibrated event probability; old freezes fall
    # back to probability_up for backward compatibility.
    base_up_rate = sum(1 for r in rows if r["actual_label"] == 1) / n_total

    def decision_score_up(row: dict) -> float:
        try:
            score = float(row.get("direction_score_up"))
        except (TypeError, ValueError):
            score = float("nan")
        return score if math.isfinite(score) else float(row["probability_up"])

    out = []
    for thr in THRESHOLDS:
        # Two-sided signal: confident UP (score >= thr) or DOWN (score <= 1-thr).
        signals_up   = [r for r in rows if decision_score_up(r) >= thr]
        signals_down = [r for r in rows if decision_score_up(r) <= (1 - thr)]
        n_up         = len(signals_up)
        n_down       = len(signals_down)
        n_signal     = n_up + n_down

        if n_signal == 0:
            acc = float("nan")
            up_acc = float("nan")
            down_acc = float("nan")
        else:
            up_correct   = sum(1 for r in signals_up   if r["actual_label"] == 1)
            down_correct = sum(1 for r in signals_down if r["actual_label"] == 0)
            correct      = up_correct + down_correct
            acc          = correct / n_signal
            up_acc       = up_correct   / n_up   if n_up   else float("nan")
            down_acc     = down_correct / n_down if n_down else float("nan")

        out.append({
            "horizon": horizon,
            "model":   model,
            "threshold": thr,
            "n_total": n_total,
            "n_signal": n_signal,
            "n_up_signals": n_up,
            "n_down_signals": n_down,
            "signal_pct_days": round(100 * n_signal / n_total, 2) if n_total else 0.0,
            "accuracy_on_signals": round(100 * acc, 2) if n_signal else None,
            "accuracy_wilson_lower_95": wilson_lower_bound(correct, n_signal) if n_signal else None,
            "up_signal_accuracy":   round(100 * up_acc, 2)   if n_up   else None,
            "down_signal_accuracy": round(100 * down_acc, 2) if n_down else None,
            "base_up_rate_full_oof": round(100 * base_up_rate, 2),
            "signals_per_year_extrapolated": round(365 * n_signal / n_total, 1) if n_total else 0.0,
        })
    return out


def main() -> None:
    if not LONGRUN_JSON.exists():
        raise SystemExit(f"Missing longrun JSON: {LONGRUN_JSON}")
    state = json.loads(LONGRUN_JSON.read_text(encoding="utf-8"))
    horizons = state.get("horizons", {})

    results = []
    for h_str, h_payload in horizons.items():
        horizon = int(h_str)
        predictions = h_payload.get("predictions") or []
        if not predictions:
            print(f"  h={horizon}: no predictions")
            continue

        # Identify classification models present.
        cls_models = sorted({
            p["model"] for p in predictions
            if p.get("head") == "classification" and p.get("model")
        })
        # Sweep the priority models first, then every remaining classifier.
        # The best selective signal can move between model families by regime,
        # so restricting the sweep silently hides useful gates.
        priority = [
            "trimmed_classification_ensemble_equal",
            "trimmed_equal_weight_direction",
            "logistic",
            "knn_clf",
            "catboost_clf",
            "random_forest_clf",
            "extra_trees_clf",
            "hist_gbc",
            "mlp_clf",
        ]
        models_to_sweep = [m for m in priority if m in cls_models]
        models_to_sweep.extend(m for m in cls_models if m not in models_to_sweep)

        print(f"\n=== h={horizon} ===")
        for model in models_to_sweep:
            rows = sweep_one_model(predictions, horizon, model)
            results.extend(rows)
            if not rows:
                continue
            print(f"\n  Model: {model}")
            print(f"  {'thr':>5s}  {'sig%':>6s}  {'acc':>6s}  {'n_sig':>6s}  {'up_acc':>7s}  {'dn_acc':>7s}  {'sig/yr':>7s}")
            for r in rows:
                acc = r["accuracy_on_signals"]
                up = r["up_signal_accuracy"]
                dn = r["down_signal_accuracy"]
                acc_s = f"{acc:5.1f}%" if acc is not None else "  n/a"
                up_s  = f"{up:5.1f}%"  if up is not None  else "   n/a"
                dn_s  = f"{dn:5.1f}%"  if dn is not None  else "   n/a"
                print(f"  {r['threshold']:>5.2f}  {r['signal_pct_days']:>5.1f}%  {acc_s:>6s}  {r['n_signal']:>6d}  {up_s:>7s}  {dn_s:>7s}  {r['signals_per_year_extrapolated']:>7.1f}")

    OUTPUT_JSON.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\n\nWrote: {OUTPUT_JSON}")
    print(f"Total rows: {len(results)}")


if __name__ == "__main__":
    main()
