"""Gate ETH forecast candidates against a stored production baseline.

This script is intentionally small and conservative. It reads a long-run OOF
metrics JSON, optionally reads a threshold sweep JSON, and exits non-zero when
the candidate fails the profile configured in model_baseline_manifest.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


def _json_constant(value: str) -> float:
    if value == "NaN":
        return float("nan")
    if value == "Infinity":
        return float("inf")
    if value == "-Infinity":
        return float("-inf")
    raise ValueError(value)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=_json_constant)


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isfinite(result):
        return result
    return None


def json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def rows_for_horizon(payload: dict[str, Any], horizon: str, key: str) -> list[dict[str, Any]]:
    horizon_payload = (payload.get("horizons") or {}).get(str(horizon)) or {}
    rows = horizon_payload.get(key) or []
    return [row for row in rows if isinstance(row, dict)]


def best_min(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in (finite_float(row.get(key)) for row in rows) if value is not None]
    return min(values) if values else None


def best_max(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [value for value in (finite_float(row.get(key)) for row in rows) if value is not None]
    return max(values) if values else None


def all_leakage_safe(rows: list[dict[str, Any]]) -> bool:
    checked = [row for row in rows if "validation_leakage_safe" in row]
    return bool(checked) and all(bool(row.get("validation_leakage_safe")) for row in checked)


def sign_label(value: float | None, *, deadband: float = 0.0) -> str:
    if value is None:
        return "UNKNOWN"
    if value > deadband:
        return "UP"
    if value < -deadband:
        return "DOWN"
    return "FLAT"


def percentile(values: list[float], q: float) -> float | None:
    clean = sorted(value for value in values if math.isfinite(value))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    position = (len(clean) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return clean[lower]
    weight = position - lower
    return clean[lower] * (1.0 - weight) + clean[upper] * weight


def regime_breakdown(candidate: dict[str, Any], horizon: str) -> list[dict[str, Any]]:
    predictions = rows_for_horizon(candidate, horizon, "predictions")
    if not predictions:
        return []
    actual_abs_returns = [
        abs(value)
        for value in (finite_float(row.get("actual_return")) for row in predictions)
        if value is not None
    ]
    high_vol_cutoff = percentile(actual_abs_returns, 0.75)
    low_vol_cutoff = percentile(actual_abs_returns, 0.25)

    enriched: list[dict[str, Any]] = []
    for row in predictions:
        actual_return = finite_float(row.get("actual_return"))
        if actual_return is None:
            continue
        abs_return = abs(actual_return)
        if high_vol_cutoff is not None and abs_return >= high_vol_cutoff:
            volatility_bucket = "HIGH_VOL"
        elif low_vol_cutoff is not None and abs_return <= low_vol_cutoff:
            volatility_bucket = "LOW_VOL"
        else:
            volatility_bucket = "MID_VOL"
        enriched.append(
            {
                **row,
                "_actual_return": actual_return,
                "_actual_direction": sign_label(actual_return),
                "_volatility_bucket": volatility_bucket,
            }
        )

    breakdown_rows: list[dict[str, Any]] = []
    for bucket_name, bucket_getter in (
        ("realized_direction", lambda item: item["_actual_direction"]),
        ("realized_volatility", lambda item: item["_volatility_bucket"]),
    ):
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
        for row in enriched:
            key = (
                str(row.get("head") or ""),
                str(row.get("model") or ""),
                str(bucket_getter(row)),
            )
            groups.setdefault(key, []).append(row)
        for (head, model, bucket), group in groups.items():
            actual = [item["_actual_return"] for item in group]
            predicted_return = [finite_float(item.get("predicted_return")) for item in group]
            predicted_return_clean = [value for value in predicted_return if value is not None]
            probability_up = [finite_float(item.get("probability_up")) for item in group]
            probability_up_clean = [value for value in probability_up if value is not None]
            predicted_label = [finite_float(item.get("predicted_label")) for item in group]
            predicted_label_clean = [int(value) for value in predicted_label if value is not None]
            actual_labels = [
                int(label)
                if (label := finite_float(item.get("actual_label"))) is not None
                else None
                for item in group
            ]

            row_out: dict[str, Any] = {
                "bucket_type": bucket_name,
                "bucket": bucket,
                "head": head,
                "model": model,
                "n": len(group),
                "mean_actual_return": sum(actual) / len(actual) if actual else None,
            }
            if predicted_return_clean and len(predicted_return_clean) == len(actual):
                errors = [pred - act for pred, act in zip(predicted_return_clean, actual)]
                row_out.update(
                    {
                        "return_mae": sum(abs(error) for error in errors) / len(errors),
                        "return_rmse": math.sqrt(sum(error * error for error in errors) / len(errors)),
                        "directional_accuracy": sum(
                            1
                            for pred, act in zip(predicted_return_clean, actual)
                            if sign_label(pred) == sign_label(act)
                        )
                        / len(actual),
                    }
                )
            probability_label_pairs = [
                (probability, actual_label)
                for probability, actual_label in zip(probability_up, actual_labels)
                if probability is not None and actual_label is not None
            ]
            if probability_label_pairs:
                brier = [
                    (probability - actual_label) ** 2
                    for probability, actual_label in probability_label_pairs
                ]
                row_out["brier_score"] = sum(brier) / len(brier)
            predicted_label_pairs = [
                (pred, actual_label)
                for pred, actual_label in zip(predicted_label, actual_labels)
                if pred is not None and actual_label is not None
            ]
            if predicted_label_pairs:
                row_out["classification_accuracy"] = sum(
                    1
                    for pred, actual_label in predicted_label_pairs
                    if int(pred) == int(actual_label)
                ) / len(predicted_label_pairs)
            breakdown_rows.append(row_out)

    return sorted(
        breakdown_rows,
        key=lambda row: (
            str(row.get("bucket_type")),
            str(row.get("bucket")),
            str(row.get("head")),
            finite_float(row.get("return_mae")) if finite_float(row.get("return_mae")) is not None else 999.0,
            finite_float(row.get("brier_score")) if finite_float(row.get("brier_score")) is not None else 999.0,
            str(row.get("model")),
        ),
    )


def completed_folds(payload: dict[str, Any], horizon: str) -> int:
    raw = (payload.get("folds_completed") or {}).get(str(horizon))
    value = finite_float(raw)
    if value is not None:
        return int(value)
    horizon_payload = (payload.get("horizons") or {}).get(str(horizon)) or {}
    fold_values: list[float] = []
    for key in ("regression_leaderboard", "classification_leaderboard"):
        for row in horizon_payload.get(key) or []:
            fold_value = finite_float(row.get("folds"))
            if fold_value is not None:
                fold_values.append(fold_value)
    return int(max(fold_values)) if fold_values else 0


def threshold_best(
    threshold_rows: list[dict[str, Any]],
    horizon: str,
    *,
    min_signal_pct_days: float,
    min_n_signal: int,
) -> dict[str, Any] | None:
    eligible: list[dict[str, Any]] = []
    for row in threshold_rows:
        if str(row.get("horizon")) != str(horizon):
            continue
        signal_pct = finite_float(row.get("signal_pct_days"))
        n_signal = finite_float(row.get("n_signal"))
        accuracy = finite_float(row.get("accuracy_on_signals"))
        if signal_pct is None or n_signal is None or accuracy is None:
            continue
        if signal_pct >= min_signal_pct_days and int(n_signal) >= min_n_signal:
            eligible.append(row)
    if not eligible:
        return None

    def accuracy_lower_bound(row: dict[str, Any]) -> float:
        existing = finite_float(row.get("accuracy_wilson_lower_95"))
        if existing is not None:
            return existing
        accuracy = finite_float(row.get("accuracy_on_signals"))
        n_signal = finite_float(row.get("n_signal"))
        if accuracy is None or n_signal is None or n_signal <= 0:
            return float("-inf")
        p = max(0.0, min(1.0, accuracy / 100.0))
        z = 1.96
        denom = 1.0 + (z * z / n_signal)
        centre = p + (z * z / (2.0 * n_signal))
        margin = z * ((p * (1.0 - p) / n_signal + z * z / (4.0 * n_signal * n_signal)) ** 0.5)
        return 100.0 * ((centre - margin) / denom)

    return max(
        eligible,
        key=lambda row: (
            accuracy_lower_bound(row),
            finite_float(row.get("accuracy_on_signals")) or float("-inf"),
            finite_float(row.get("signal_pct_days")) or float("-inf"),
            finite_float(row.get("n_signal")) or float("-inf"),
        ),
    )


def selective_signal_override_state(
    threshold_row: dict[str, Any],
    cls_spec: dict[str, Any],
) -> dict[str, Any]:
    """Return whether a selective threshold gate can offset weak global ROC AUC.

    AUC evaluates global probability ranking over every day. The live strategy is
    intentionally selective: most days should be FLAT/range-only, and only high
    threshold days become directional calls. This override is therefore allowed
    only when the threshold sweep clears explicit hit-rate and coverage floors.
    """
    enabled = bool(cls_spec.get("allow_selective_signal_override", False))
    accuracy = finite_float(threshold_row.get("accuracy_on_signals"))
    signal_pct = finite_float(threshold_row.get("signal_pct_days"))
    n_signal = finite_float(threshold_row.get("n_signal"))
    min_accuracy = finite_float(cls_spec.get("selective_min_accuracy_on_signals"))
    min_signal_pct = finite_float(cls_spec.get("selective_min_signal_pct_days"))
    min_n_signal = finite_float(cls_spec.get("selective_min_n_signal"))

    passed = bool(
        enabled
        and accuracy is not None
        and signal_pct is not None
        and n_signal is not None
        and (min_accuracy is None or accuracy >= min_accuracy)
        and (min_signal_pct is None or signal_pct >= min_signal_pct)
        and (min_n_signal is None or int(n_signal) >= int(min_n_signal))
    )
    return {
        "enabled": enabled,
        "passed": passed,
        "applied": False,
        "accuracy_on_signals": accuracy,
        "signal_pct_days": signal_pct,
        "n_signal": int(n_signal) if n_signal is not None else None,
        "min_accuracy_on_signals": min_accuracy,
        "min_signal_pct_days": min_signal_pct,
        "min_n_signal": int(min_n_signal) if min_n_signal is not None else None,
    }


def check_max(
    failures: list[str],
    warnings: list[str],
    *,
    label: str,
    value: float | None,
    limit: float | None,
    severity: str,
) -> None:
    if limit is None:
        return
    if value is None:
        message = f"{label}: missing value"
    elif value <= limit:
        return
    else:
        message = f"{label}: {value:.6g} > max {limit:.6g}"
    (warnings if severity == "warn" else failures).append(message)


def check_min(
    failures: list[str],
    warnings: list[str],
    *,
    label: str,
    value: float | None,
    limit: float | None,
    severity: str,
) -> None:
    if limit is None:
        return
    if value is None:
        message = f"{label}: missing value"
    elif value >= limit:
        return
    else:
        message = f"{label}: {value:.6g} < min {limit:.6g}"
    (warnings if severity == "warn" else failures).append(message)


def evaluate(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold_rows: list[dict[str, Any]],
    profile_name: str,
) -> tuple[dict[str, Any], list[str], list[str]]:
    profiles = baseline.get("profiles") or {}
    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        raise SystemExit(f"Unknown profile '{profile_name}'. Available profiles: {sorted(profiles)}")

    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, Any] = {
        "profile": profile_name,
        "candidate_mode": candidate.get("mode"),
        "partial": bool(candidate.get("partial")),
        "horizons": {},
    }

    require_complete = bool(profile.get("require_complete", False))
    require_leakage_safe = bool(profile.get("require_leakage_safe", True))
    min_folds = {str(key): int(value) for key, value in (profile.get("min_folds_completed") or {}).items()}
    horizon_specs = profile.get("horizons") or {}

    for horizon, spec in horizon_specs.items():
        reg_rows = rows_for_horizon(candidate, horizon, "regression_leaderboard")
        cls_rows = rows_for_horizon(candidate, horizon, "classification_leaderboard")
        folds = completed_folds(candidate, horizon)
        min_fold_count = int(min_folds.get(str(horizon), 0))

        horizon_report = {
            "folds_completed": folds,
            "regression": {
                "best_price_rmse": best_min(reg_rows, "price_rmse"),
                "best_price_mae": best_min(reg_rows, "price_mae"),
                "best_return_mae": best_min(reg_rows, "return_mae"),
                "best_directional_accuracy": best_max(reg_rows, "directional_accuracy"),
            },
            "classification": {
                "best_balanced_accuracy": best_max(cls_rows, "balanced_accuracy"),
                "best_brier_score": best_min(cls_rows, "brier_score"),
                "best_roc_auc": best_max(cls_rows, "roc_auc"),
                "leakage_safe": all_leakage_safe(cls_rows),
            },
            "threshold_sweep": {},
            "regime_breakdown": regime_breakdown(candidate, horizon),
        }
        report["horizons"][str(horizon)] = horizon_report

        if folds < min_fold_count:
            failures.append(f"h{horizon}: folds_completed {folds} < required {min_fold_count}")
        if require_complete and bool(candidate.get("partial")):
            failures.append(f"h{horizon}: candidate run is partial but profile requires complete")
        if require_leakage_safe and not horizon_report["classification"]["leakage_safe"]:
            failures.append(f"h{horizon}: threshold validation is not leakage-safe for all classification rows")

        reg_spec = spec.get("regression") or {}
        reg_severity = str(reg_spec.get("severity", "fail"))
        check_max(
            failures,
            warnings,
            label=f"h{horizon} regression best_price_rmse",
            value=horizon_report["regression"]["best_price_rmse"],
            limit=finite_float(reg_spec.get("max_best_price_rmse")),
            severity=reg_severity,
        )
        check_max(
            failures,
            warnings,
            label=f"h{horizon} regression best_return_mae",
            value=horizon_report["regression"]["best_return_mae"],
            limit=finite_float(reg_spec.get("max_best_return_mae")),
            severity=reg_severity,
        )
        check_min(
            failures,
            warnings,
            label=f"h{horizon} regression best_directional_accuracy",
            value=horizon_report["regression"]["best_directional_accuracy"],
            limit=finite_float(reg_spec.get("min_best_directional_accuracy")),
            severity=reg_severity,
        )

        sweep_spec = spec.get("threshold_sweep") or {}
        if sweep_spec and threshold_rows:
            best = threshold_best(
                threshold_rows,
                horizon,
                min_signal_pct_days=float(sweep_spec.get("min_signal_pct_days", 0.0)),
                min_n_signal=int(sweep_spec.get("min_n_signal", 0)),
            )
            horizon_report["threshold_sweep"] = best or {}
            sweep_accuracy = finite_float((best or {}).get("accuracy_on_signals"))
            check_min(
                failures,
                warnings,
                label=f"h{horizon} threshold_sweep accuracy_on_signals",
                value=sweep_accuracy,
                limit=finite_float(sweep_spec.get("min_accuracy_on_signals")),
                severity=str(sweep_spec.get("severity", "warn")),
            )
        elif sweep_spec and str(sweep_spec.get("severity", "warn")) == "fail":
            failures.append(f"h{horizon}: threshold sweep rows are missing")

        cls_spec = spec.get("classification") or {}
        cls_severity = str(cls_spec.get("severity", "fail"))
        selective_state = selective_signal_override_state(
            horizon_report.get("threshold_sweep") or {},
            cls_spec,
        )
        horizon_report["classification"]["selective_signal_override"] = selective_state
        check_min(
            failures,
            warnings,
            label=f"h{horizon} classification best_balanced_accuracy",
            value=horizon_report["classification"]["best_balanced_accuracy"],
            limit=finite_float(cls_spec.get("min_best_balanced_accuracy")),
            severity=cls_severity,
        )
        check_max(
            failures,
            warnings,
            label=f"h{horizon} classification best_brier_score",
            value=horizon_report["classification"]["best_brier_score"],
            limit=finite_float(cls_spec.get("max_best_brier_score")),
            severity=cls_severity,
        )
        roc_value = horizon_report["classification"]["best_roc_auc"]
        roc_limit = finite_float(cls_spec.get("min_best_roc_auc"))
        if roc_limit is not None:
            if roc_value is None:
                message = f"h{horizon} classification best_roc_auc: missing value"
            elif roc_value >= roc_limit:
                message = ""
            else:
                message = f"h{horizon} classification best_roc_auc: {roc_value:.6g} < min {roc_limit:.6g}"
            if message:
                if selective_state.get("passed"):
                    selective_state["applied"] = True
                    warnings.append(
                        f"{message}; selective signal gate passed "
                        f"({selective_state.get('accuracy_on_signals')}% hit, "
                        f"{selective_state.get('signal_pct_days')}% days)"
                    )
                else:
                    (warnings if cls_severity == "warn" else failures).append(message)

    return report, failures, warnings


def write_markdown(path: Path, report: dict[str, Any], failures: list[str], warnings: list[str]) -> None:
    status = "FAIL" if failures else "PASS"
    lines = [
        f"# ETH model evaluation: {status}",
        "",
        f"- Profile: `{report.get('profile')}`",
        f"- Candidate mode: `{report.get('candidate_mode')}`",
        f"- Partial: `{report.get('partial')}`",
        "",
    ]
    for horizon, horizon_report in sorted((report.get("horizons") or {}).items(), key=lambda item: int(item[0])):
        reg = horizon_report.get("regression") or {}
        cls = horizon_report.get("classification") or {}
        sweep = horizon_report.get("threshold_sweep") or {}
        lines.extend(
            [
                f"## Horizon {horizon}d",
                "",
                f"- Folds completed: `{horizon_report.get('folds_completed')}`",
                f"- Regression best RMSE: `{reg.get('best_price_rmse')}`",
                f"- Regression best return MAE: `{reg.get('best_return_mae')}`",
                f"- Regression best directional accuracy: `{reg.get('best_directional_accuracy')}`",
                f"- Classification best balanced accuracy: `{cls.get('best_balanced_accuracy')}`",
                f"- Classification best Brier: `{cls.get('best_brier_score')}`",
                f"- Classification best ROC AUC: `{cls.get('best_roc_auc')}`",
                f"- Classification leakage safe: `{cls.get('leakage_safe')}`",
            ]
        )
        if sweep:
            lines.append(
                f"- Threshold sweep best: `{sweep.get('model')}` threshold `{sweep.get('threshold')}` "
                f"accuracy `{sweep.get('accuracy_on_signals')}` signal% `{sweep.get('signal_pct_days')}`"
            )
        override = (cls.get("selective_signal_override") or {})
        if override.get("enabled"):
            lines.append(
                f"- Selective signal override: passed `{override.get('passed')}` "
                f"applied `{override.get('applied')}`"
            )
        regime_rows = horizon_report.get("regime_breakdown") or []
        if regime_rows:
            worst_regression = [
                row for row in regime_rows
                if row.get("head") == "regression" and finite_float(row.get("return_mae")) is not None
            ]
            worst_classification = [
                row for row in regime_rows
                if row.get("head") == "classification" and finite_float(row.get("brier_score")) is not None
            ]
            worst_regression = sorted(
                worst_regression,
                key=lambda row: finite_float(row.get("return_mae")) or float("-inf"),
                reverse=True,
            )[:3]
            worst_classification = sorted(
                worst_classification,
                key=lambda row: finite_float(row.get("brier_score")) or float("-inf"),
                reverse=True,
            )[:3]
            lines.append("- Regime breakdown: included in JSON for realized direction and realized volatility buckets")
            for row in worst_regression:
                lines.append(
                    f"  - Regression weak bucket `{row.get('bucket_type')}={row.get('bucket')}` "
                    f"model `{row.get('model')}` return MAE `{row.get('return_mae')}` n `{row.get('n')}`"
                )
            for row in worst_classification:
                lines.append(
                    f"  - Classification weak bucket `{row.get('bucket_type')}={row.get('bucket')}` "
                    f"model `{row.get('model')}` Brier `{row.get('brier_score')}` n `{row.get('n')}`"
                )
        lines.append("")
    if failures:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
        lines.append("")
    if warnings:
        lines.extend(["## Warnings", ""])
        lines.extend(f"- {warning}" for warning in warnings)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--threshold-sweep", type=Path, default=None)
    parser.add_argument("--profile", choices=["smoke", "full"], default="smoke")
    parser.add_argument("--output-json", type=Path, default=Path("tests/phase0/model_eval_report.json"))
    parser.add_argument("--output-md", type=Path, default=Path("tests/phase0/model_eval_report.md"))
    args = parser.parse_args(argv)

    baseline = load_json(args.baseline)
    candidate = load_json(args.candidate)
    threshold_rows: list[dict[str, Any]] = []
    if args.threshold_sweep and args.threshold_sweep.exists():
        loaded = load_json(args.threshold_sweep)
        threshold_rows = loaded if isinstance(loaded, list) else []

    report, failures, warnings = evaluate(
        baseline=baseline,
        candidate=candidate,
        threshold_rows=threshold_rows,
        profile_name=args.profile,
    )
    report["failures"] = failures
    report["warnings"] = warnings

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(args.output_md, report, failures, warnings)

    print(args.output_md.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
