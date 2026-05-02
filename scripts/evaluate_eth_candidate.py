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
    return max(eligible, key=lambda row: finite_float(row.get("accuracy_on_signals")) or float("-inf"))


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

        cls_spec = spec.get("classification") or {}
        cls_severity = str(cls_spec.get("severity", "fail"))
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
        check_min(
            failures,
            warnings,
            label=f"h{horizon} classification best_roc_auc",
            value=horizon_report["classification"]["best_roc_auc"],
            limit=finite_float(cls_spec.get("min_best_roc_auc")),
            severity=cls_severity,
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
