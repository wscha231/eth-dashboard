"""Compare two longrun OOF freezes and surface where the new run wins/loses
vs the old one — at per-model, per-regime, and per-date granularity.

This is the "improvement loop" companion to the longrun scripts. The longrun
JSONs give us **leakage-safe per-date predictions** across ~3 years; this
script aligns them and computes deltas so a Go/No-Go decision for promoting
a new phase can be made from data rather than aggregate leaderboard alone.

Designed to work on PARTIAL checkpoints too — if both runs have completed N
folds, we diff only the overlapping date range and clearly label the coverage.

Usage:
  python tests/phase0/diff_longrun_oof.py \\
      --old tests/phase0/phase5_production_longrun_oof_metrics.json \\
      --new tests/phase0/phase6_production_longrun_oof_metrics.json \\
      [--output-json forecast_site/public/backtest_diff_phase5_vs_phase6.json]

What it prints:

  1. Per-(horizon, head, model) aggregate delta table — RMSE/MAE/MAPE/dir_acc
     for regression, Brier/AUC/Accuracy for classification.
  2. Per-regime (bull / bear / chop) breakdown — catches "new wins only in bear".
  3. Paired Wilcoxon signed-rank p-value on abs_err for regression — asks
     "is the improvement statistically non-zero?" (non-parametric, autocorr-
     agnostic proxy for Diebold-Mariano).
  4. Worst N regressions — top dates where the new run made predictions that
     are both (a) far from old's prediction and (b) worse than actual. Useful
     for anecdote-driven debugging.

What the JSON contains (when --output-json is given): the same info in a
machine-readable shape so a future frontend comparison page can consume it.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

# --- Regime tagging ---------------------------------------------------------
# Classify a target's realised return into a market regime. Absolute thresholds
# are easier to reason about than percentile-based ones and give stable bucket
# definitions across runs even as the OOF window grows.
REGIME_BULL_THRESHOLD = 0.02    # target_return >= +2% -> bull
REGIME_BEAR_THRESHOLD = -0.02   # target_return <= -2% -> bear
# Anything in between -> chop


def _regime(actual_return: float | None) -> str:
    if actual_return is None or not math.isfinite(actual_return):
        return "unknown"
    if actual_return >= REGIME_BULL_THRESHOLD:
        return "bull"
    if actual_return <= REGIME_BEAR_THRESHOLD:
        return "bear"
    return "chop"


# --- Metrics ---------------------------------------------------------------
def _rmse(errors: list[float]) -> float | None:
    if not errors:
        return None
    return math.sqrt(sum(e * e for e in errors) / len(errors))


def _mae(errors: list[float]) -> float | None:
    if not errors:
        return None
    return sum(abs(e) for e in errors) / len(errors)


def _mape(actuals: list[float], preds: list[float]) -> float | None:
    # Mean absolute percentage error on close prices. Skips zero/tiny actuals.
    pairs = [(a, p) for a, p in zip(actuals, preds)
             if a is not None and p is not None and abs(a) > 1e-9]
    if not pairs:
        return None
    return 100.0 * sum(abs((a - p) / a) for a, p in pairs) / len(pairs)


def _directional_accuracy(actuals: list[float], preds: list[float]) -> float | None:
    pairs = [(a, p) for a, p in zip(actuals, preds) if a is not None and p is not None]
    if not pairs:
        return None
    hits = sum(1 for a, p in pairs if (a > 0) == (p > 0))
    return hits / len(pairs)


def _brier(labels: list[int], probs: list[float]) -> float | None:
    pairs = [(y, p) for y, p in zip(labels, probs) if y is not None and p is not None]
    if not pairs:
        return None
    return sum((p - y) ** 2 for y, p in pairs) / len(pairs)


def _accuracy(labels: list[int], pred_labels: list[int]) -> float | None:
    pairs = [(y, p) for y, p in zip(labels, pred_labels) if y is not None and p is not None]
    if not pairs:
        return None
    return sum(1 for y, p in pairs if y == p) / len(pairs)


def _auc(labels: list[int], probs: list[float]) -> float | None:
    """Mann-Whitney-U-style AUC on binary labels (avoids sklearn import)."""
    pos = [p for y, p in zip(labels, probs) if y == 1 and p is not None]
    neg = [p for y, p in zip(labels, probs) if y == 0 and p is not None]
    if not pos or not neg:
        return None
    # For each (pos, neg) pair, count +1 if pos > neg, +0.5 on tie, 0 if neg > pos.
    # O(n*m) which is fine for ~2k positives × ~2k negatives at full longrun.
    wins = 0.0
    for p_pos in pos:
        for p_neg in neg:
            if p_pos > p_neg:
                wins += 1
            elif p_pos == p_neg:
                wins += 0.5
    return wins / (len(pos) * len(neg))


def _wilcoxon_signed_rank_p(diffs: list[float]) -> float | None:
    """Two-sided Wilcoxon signed-rank test p-value approximation.

    Null hypothesis: median of paired differences = 0.
    Non-parametric; robust to outliers; no autocorrelation correction (so it's
    a directional proxy rather than a publication-grade stat). For strict
    time-series significance use Diebold-Mariano; for a first cut this is fine.
    """
    non_zero = [d for d in diffs if d != 0 and math.isfinite(d)]
    n = len(non_zero)
    if n < 10:
        return None
    # Rank |d|, then sign-sum the ranks. Ties get average rank.
    abs_sorted = sorted(enumerate(non_zero), key=lambda kv: abs(kv[1]))
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # find tie block on |d|
        while j + 1 < n and abs(abs_sorted[j + 1][1]) == abs(abs_sorted[i][1]):
            j += 1
        avg_rank = (i + j + 2) / 2.0  # ranks are 1-indexed
        for k in range(i, j + 1):
            orig_idx, _ = abs_sorted[k]
            ranks[orig_idx] = avg_rank
        i = j + 1
    # Signed sum.
    W = sum(r if d > 0 else -r for d, r in zip(non_zero, ranks))
    # Normal approximation: mu=0, sigma = sqrt(n(n+1)(2n+1)/6).
    sigma = math.sqrt(n * (n + 1) * (2 * n + 1) / 6.0)
    if sigma == 0:
        return None
    z = W / sigma
    # Two-sided p from standard normal.
    return math.erfc(abs(z) / math.sqrt(2))


# --- Prediction indexing ----------------------------------------------------
def _index_predictions(
    payload: dict[str, Any],
) -> dict[tuple[int, str, str, str], dict[str, Any]]:
    """(horizon, head, model, prediction_date) -> row"""
    index: dict[tuple[int, str, str, str], dict[str, Any]] = {}
    for h_key, h_payload in (payload.get("horizons") or {}).items():
        try:
            horizon = int(h_key)
        except (TypeError, ValueError):
            continue
        for row in h_payload.get("predictions") or []:
            model = row.get("model")
            head = row.get("head")
            pdate = row.get("prediction_date")
            if not (model and head and pdate):
                continue
            index[(horizon, head, model, pdate)] = row
    return index


# --- Main comparison --------------------------------------------------------
def compare_runs(old: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Align predictions and compute aggregate + regime + paired metrics."""
    old_idx = _index_predictions(old)
    new_idx = _index_predictions(new)

    # Intersection: only keys present in both runs.
    keys_common = set(old_idx) & set(new_idx)
    keys_old_only = set(old_idx) - keys_common
    keys_new_only = set(new_idx) - keys_common

    # Group the common keys by (horizon, head, model).
    groups: dict[tuple[int, str, str], list[tuple[str, dict, dict]]] = defaultdict(list)
    for key in keys_common:
        horizon, head, model, pdate = key
        groups[(horizon, head, model)].append((pdate, old_idx[key], new_idx[key]))
    # Sort each group by date so time-ordered metrics are consistent.
    for g in groups.values():
        g.sort(key=lambda x: x[0])

    model_comparisons: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    worst_regressions: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)

    for (horizon, head, model), triples in sorted(groups.items()):
        if head == "regression":
            actual_ret = [t[1].get("actual_return") for t in triples]
            pred_old   = [t[1].get("predicted_return") for t in triples]
            pred_new   = [t[2].get("predicted_return") for t in triples]
            act_close  = [t[1].get("actual_close") for t in triples]
            pclose_old = [t[1].get("predicted_close") for t in triples]
            pclose_new = [t[2].get("predicted_close") for t in triples]
            ref_close  = [t[1].get("reference_close") for t in triples]

            # Skip dates where any side is missing a prediction.
            mask = [
                a is not None and po is not None and pn is not None
                for a, po, pn in zip(actual_ret, pred_old, pred_new)
            ]
            if not any(mask):
                continue

            def _f(lst): return [x for x, m in zip(lst, mask) if m]

            actual_ret = _f(actual_ret); pred_old = _f(pred_old); pred_new = _f(pred_new)
            act_close  = _f(act_close);  pclose_old = _f(pclose_old); pclose_new = _f(pclose_new)
            ref_close  = _f(ref_close)
            dates      = _f([t[0] for t in triples])

            err_old_ret = [po - a for a, po in zip(actual_ret, pred_old)]
            err_new_ret = [pn - a for a, pn in zip(actual_ret, pred_new)]
            err_old_close = [po - a for a, po in zip(act_close, pclose_old) if a is not None and po is not None]
            err_new_close = [pn - a for a, pn in zip(act_close, pclose_new) if a is not None and pn is not None]

            # Per-regime breakdown.
            per_regime: dict[str, dict[str, Any]] = {}
            for regime in ("bull", "bear", "chop"):
                idx = [i for i, a in enumerate(actual_ret) if _regime(a) == regime]
                if not idx:
                    continue
                e_old = [err_old_ret[i] for i in idx]
                e_new = [err_new_ret[i] for i in idx]
                per_regime[regime] = {
                    "n":           len(idx),
                    "rmse_old":    _rmse(e_old),
                    "rmse_new":    _rmse(e_new),
                    "mae_old":     _mae(e_old),
                    "mae_new":     _mae(e_new),
                    "dir_acc_old": _directional_accuracy(
                        [actual_ret[i] for i in idx], [pred_old[i] for i in idx]),
                    "dir_acc_new": _directional_accuracy(
                        [actual_ret[i] for i in idx], [pred_new[i] for i in idx]),
                }
                regime_rows.append({
                    "horizon": horizon, "head": head, "model": model,
                    "regime": regime, **per_regime[regime],
                })

            # Paired diff of absolute errors: positive means new WORSE.
            paired_abs_diff = [abs(en) - abs(eo) for en, eo in zip(err_new_ret, err_old_ret)]
            p_value = _wilcoxon_signed_rank_p(paired_abs_diff)
            mean_abs_diff = statistics.fmean(paired_abs_diff) if paired_abs_diff else None

            # Anecdote mining: dates where (a) new differs notably from old AND
            # (b) new is worse than old.
            anecdotes: list[dict[str, Any]] = []
            for dte, a, po, pn, eo, en in zip(
                dates, actual_ret, pred_old, pred_new, err_old_ret, err_new_ret,
            ):
                divergence = abs(pn - po)
                if divergence < 0.005:  # <0.5 pp - not interesting
                    continue
                if abs(en) <= abs(eo):
                    continue  # new beat old on this date
                anecdotes.append({
                    "date": dte,
                    "actual_return": a,
                    "predicted_return_old": po,
                    "predicted_return_new": pn,
                    "abs_error_old": abs(eo),
                    "abs_error_new": abs(en),
                    "abs_err_delta": abs(en) - abs(eo),
                    "regime": _regime(a),
                })
            anecdotes.sort(key=lambda r: r["abs_err_delta"], reverse=True)
            worst_regressions[(horizon, model)] = anecdotes[:10]

            model_comparisons.append({
                "horizon": horizon, "head": head, "model": model,
                "n_dates": len(dates),
                "date_range": [dates[0], dates[-1]] if dates else None,
                # Return-space metrics.
                "rmse_ret_old":  _rmse(err_old_ret),
                "rmse_ret_new":  _rmse(err_new_ret),
                "mae_ret_old":   _mae(err_old_ret),
                "mae_ret_new":   _mae(err_new_ret),
                # Price-space metrics.
                "rmse_close_old": _rmse(err_old_close) if err_old_close else None,
                "rmse_close_new": _rmse(err_new_close) if err_new_close else None,
                "mape_close_old": _mape(act_close, pclose_old),
                "mape_close_new": _mape(act_close, pclose_new),
                "dir_acc_old":    _directional_accuracy(actual_ret, pred_old),
                "dir_acc_new":    _directional_accuracy(actual_ret, pred_new),
                # Significance.
                "paired_abs_err_mean_diff": mean_abs_diff,
                "wilcoxon_p_value":         p_value,
                "per_regime": per_regime,
            })

        elif head == "classification":
            labels   = [t[1].get("actual_label") for t in triples]
            prob_old = [t[1].get("probability_up") for t in triples]
            prob_new = [t[2].get("probability_up") for t in triples]
            plab_old = [t[1].get("predicted_label") for t in triples]
            plab_new = [t[2].get("predicted_label") for t in triples]

            mask = [
                y is not None and po is not None and pn is not None
                for y, po, pn in zip(labels, prob_old, prob_new)
            ]
            if not any(mask):
                continue

            def _fc(lst): return [x for x, m in zip(lst, mask) if m]
            labels   = _fc(labels); prob_old = _fc(prob_old); prob_new = _fc(prob_new)
            plab_old = _fc(plab_old); plab_new = _fc(plab_new)
            dates    = _fc([t[0] for t in triples])
            actual_ret_for_regime = _fc([t[1].get("actual_return") for t in triples])

            per_regime: dict[str, dict[str, Any]] = {}
            for regime in ("bull", "bear", "chop"):
                idx = [i for i, a in enumerate(actual_ret_for_regime) if _regime(a) == regime]
                if not idx:
                    continue
                per_regime[regime] = {
                    "n":          len(idx),
                    "brier_old":  _brier([labels[i] for i in idx], [prob_old[i] for i in idx]),
                    "brier_new":  _brier([labels[i] for i in idx], [prob_new[i] for i in idx]),
                    "accuracy_old": _accuracy([labels[i] for i in idx], [plab_old[i] for i in idx]),
                    "accuracy_new": _accuracy([labels[i] for i in idx], [plab_new[i] for i in idx]),
                }
                regime_rows.append({
                    "horizon": horizon, "head": head, "model": model,
                    "regime": regime, **per_regime[regime],
                })

            # Paired diff of Brier components (per-row): positive means new WORSE.
            paired_brier_diff = [
                (pn - y) ** 2 - (po - y) ** 2
                for y, po, pn in zip(labels, prob_old, prob_new)
            ]
            p_value = _wilcoxon_signed_rank_p(paired_brier_diff)

            model_comparisons.append({
                "horizon": horizon, "head": head, "model": model,
                "n_dates": len(dates),
                "date_range": [dates[0], dates[-1]] if dates else None,
                "brier_old":   _brier(labels, prob_old),
                "brier_new":   _brier(labels, prob_new),
                "accuracy_old": _accuracy(labels, plab_old),
                "accuracy_new": _accuracy(labels, plab_new),
                "auc_old":     _auc(labels, prob_old),
                "auc_new":     _auc(labels, prob_new),
                "paired_brier_mean_diff":  statistics.fmean(paired_brier_diff)
                                           if paired_brier_diff else None,
                "wilcoxon_p_value":        p_value,
                "per_regime": per_regime,
            })

    return {
        "coverage": {
            "common_keys":   len(keys_common),
            "old_only_keys": len(keys_old_only),
            "new_only_keys": len(keys_new_only),
            "old_folds_completed": old.get("folds_completed", {}),
            "new_folds_completed": new.get("folds_completed", {}),
            "old_partial": bool(old.get("partial", True)),
            "new_partial": bool(new.get("partial", True)),
        },
        "per_model":    model_comparisons,
        "per_regime":   regime_rows,
        "worst_cases":  {
            f"{h}:{m}": rows
            for (h, m), rows in sorted(worst_regressions.items())
        },
    }


# --- Presentation -----------------------------------------------------------
def _fmt(v: float | None, fmt: str = ".4f") -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "       —"
    return format(v, fmt)


def _delta(new: float | None, old: float | None, lower_is_better: bool = True) -> str:
    if new is None or old is None:
        return "  —"
    diff = new - old
    marker = (
        "+" if (diff > 0 and not lower_is_better) or (diff < 0 and lower_is_better) else
        "-" if (diff < 0 and not lower_is_better) or (diff > 0 and lower_is_better) else
        " "
    )
    return f"{marker}{abs(diff):.4f}"


def print_report(result: dict[str, Any], old_name: str, new_name: str) -> None:
    cov = result["coverage"]
    print("=" * 88)
    print(f"LONGRUN OOF DIFF — {old_name}  vs  {new_name}")
    print("=" * 88)
    print(f"Coverage: {cov['common_keys']} common rows "
          f"(old-only {cov['old_only_keys']}, new-only {cov['new_only_keys']})")
    print(f"Folds:    old={cov['old_folds_completed']} (partial={cov['old_partial']})  "
          f"new={cov['new_folds_completed']} (partial={cov['new_partial']})")
    print()

    # --- Per-model regression ---
    reg = [r for r in result["per_model"] if r["head"] == "regression"]
    if reg:
        print("REGRESSION — per-(horizon, model) aggregate")
        print(f"{'h':>3}  {'model':<30}  {'n':>4}  "
              f"{'rmse_ret_old':>12}  {'rmse_ret_new':>12}  {'Δ(↓good)':>9}  "
              f"{'dir_acc_old':>11}  {'dir_acc_new':>11}  {'Δ(↑good)':>9}  "
              f"{'p':>8}")
        print("-" * 120)
        for r in sorted(reg, key=lambda x: (x["horizon"], x["model"])):
            print(f"{r['horizon']:>3}  {r['model']:<30}  {r['n_dates']:>4}  "
                  f"{_fmt(r['rmse_ret_old']):>12}  {_fmt(r['rmse_ret_new']):>12}  "
                  f"{_delta(r['rmse_ret_new'], r['rmse_ret_old'], True):>9}  "
                  f"{_fmt(r['dir_acc_old']):>11}  {_fmt(r['dir_acc_new']):>11}  "
                  f"{_delta(r['dir_acc_new'], r['dir_acc_old'], False):>9}  "
                  f"{_fmt(r['wilcoxon_p_value'], '.3f'):>8}")
        print()

    # --- Per-model classification ---
    cls = [r for r in result["per_model"] if r["head"] == "classification"]
    if cls:
        print("CLASSIFICATION — per-(horizon, model) aggregate")
        print(f"{'h':>3}  {'model':<30}  {'n':>4}  "
              f"{'brier_old':>9}  {'brier_new':>9}  {'Δ(↓good)':>9}  "
              f"{'auc_old':>8}  {'auc_new':>8}  {'Δ(↑good)':>9}  "
              f"{'p':>8}")
        print("-" * 120)
        for r in sorted(cls, key=lambda x: (x["horizon"], x["model"])):
            print(f"{r['horizon']:>3}  {r['model']:<30}  {r['n_dates']:>4}  "
                  f"{_fmt(r['brier_old']):>9}  {_fmt(r['brier_new']):>9}  "
                  f"{_delta(r['brier_new'], r['brier_old'], True):>9}  "
                  f"{_fmt(r['auc_old']):>8}  {_fmt(r['auc_new']):>8}  "
                  f"{_delta(r['auc_new'], r['auc_old'], False):>9}  "
                  f"{_fmt(r['wilcoxon_p_value'], '.3f'):>8}")
        print()

    # --- Per-regime (regression only, most actionable) ---
    regime_reg = [r for r in result["per_regime"] if r["head"] == "regression"]
    if regime_reg:
        print("REGRESSION per-regime breakdown (regression)")
        print(f"{'h':>3}  {'model':<30}  {'regime':<6}  {'n':>4}  "
              f"{'rmse_old':>9}  {'rmse_new':>9}  {'Δ(↓good)':>9}  "
              f"{'dir_acc_old':>11}  {'dir_acc_new':>11}")
        print("-" * 108)
        for r in sorted(regime_reg, key=lambda x: (x["horizon"], x["model"], x["regime"])):
            print(f"{r['horizon']:>3}  {r['model']:<30}  {r['regime']:<6}  {r['n']:>4}  "
                  f"{_fmt(r['rmse_old']):>9}  {_fmt(r['rmse_new']):>9}  "
                  f"{_delta(r['rmse_new'], r['rmse_old'], True):>9}  "
                  f"{_fmt(r['dir_acc_old']):>11}  {_fmt(r['dir_acc_new']):>11}")
        print()

    # --- Worst anecdotes (first 3 models only, top-5 dates each) ---
    worst = result["worst_cases"]
    if worst:
        print("WORST NEW-SIDE REGRESSIONS (per model, top 5 dates where new >> old error)")
        shown = 0
        for key, rows in list(worst.items())[:6]:
            if not rows:
                continue
            horizon, model = key.split(":")
            print(f"  h={horizon}  {model}:")
            for row in rows[:5]:
                print(f"    {row['date']}  regime={row['regime']:<5}  "
                      f"actual_ret={row['actual_return']:+.4f}  "
                      f"old={row['predicted_return_old']:+.4f} "
                      f"new={row['predicted_return_new']:+.4f}  "
                      f"Δ|err|={row['abs_err_delta']:+.4f}")
            shown += 1
            if shown >= 6:
                break
        print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--new", required=True, type=Path)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args(argv)

    old_payload = json.loads(args.old.read_text(encoding="utf-8"))
    new_payload = json.loads(args.new.read_text(encoding="utf-8"))

    result = compare_runs(old_payload, new_payload)
    result["meta"] = {
        "old_file":  str(args.old),
        "new_file":  str(args.new),
        "old_mode":  old_payload.get("mode"),
        "new_mode":  new_payload.get("mode"),
    }

    print_report(result, args.old.stem, args.new.stem)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"[diff_longrun_oof] wrote {args.output_json}")


if __name__ == "__main__":
    main()
