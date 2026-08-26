"""Gate the 30-day compact CatBoost regressor against matched OOF anchors.

The focused runner deliberately evaluates only the incumbent CatBoost model,
the 192-feature compact challenger, and the zero-cost no-change anchor.  This
keeps candidate validation cheap while preserving same-date, leakage-safe
comparisons.  A full gate additionally uses a deterministic 30-day moving
block bootstrap so overlapping 30-day targets are not treated as independent.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HORIZON = "30"
CHALLENGER_MODEL = "catboost_compact_h30_regressor"
INCUMBENT_MODEL = "catboost_regressor"
ANCHOR_MODEL = "no_change_anchor"
EXPECTED_FEATURE_COUNT = 192


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def finite_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def model_metrics(payload: dict[str, Any], model_name: str) -> dict[str, Any] | None:
    horizon = (payload.get("horizons") or {}).get(HORIZON) or {}
    for row in horizon.get("regression_leaderboard") or []:
        if isinstance(row, dict) and row.get("model") == model_name:
            return row
    return None


def completed_folds(payload: dict[str, Any]) -> int:
    raw = (payload.get("folds_completed") or {}).get(HORIZON)
    if raw is None:
        raw = (payload.get("folds_completed") or {}).get(int(HORIZON))
    value = finite_float(raw)
    return int(value) if value is not None else 0


def compact_feature_budget(payload: dict[str, Any]) -> int | None:
    horizon_registry = (payload.get("model_registry") or {}).get(HORIZON) or {}
    model_registry = (horizon_registry.get("regression") or {}).get(CHALLENGER_MODEL) or {}
    params = model_registry.get("params") or {}
    value = finite_float(params.get("feature_budget__max_features"))
    return int(value) if value is not None else None


def improvement_pct(challenger_value: Any, baseline_value: Any) -> float | None:
    challenger = finite_float(challenger_value)
    baseline = finite_float(baseline_value)
    if challenger is None or baseline is None or baseline <= 0.0:
        return None
    return 100.0 * (baseline - challenger) / baseline


def _prediction_rows_by_key(
    payload: dict[str, Any],
    model_name: str,
) -> dict[tuple[str, str], dict[str, Any]]:
    horizon = (payload.get("horizons") or {}).get(HORIZON) or {}
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for row in horizon.get("predictions") or []:
        if not isinstance(row, dict):
            continue
        if row.get("head") != "regression" or row.get("model") != model_name:
            continue
        key = (str(row.get("prediction_date") or ""), str(row.get("target_date") or ""))
        rows[key] = row
    return rows


def moving_block_rmse_improvement(
    payload: dict[str, Any],
    baseline_model: str,
    *,
    block_length: int = 30,
    samples: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    challenger_rows = _prediction_rows_by_key(payload, CHALLENGER_MODEL)
    baseline_rows = _prediction_rows_by_key(payload, baseline_model)
    squared_errors: list[tuple[float, float]] = []
    for key in sorted(set(challenger_rows).intersection(baseline_rows)):
        challenger_row = challenger_rows[key]
        baseline_row = baseline_rows[key]
        actual = finite_float(challenger_row.get("actual_close"))
        challenger_prediction = finite_float(challenger_row.get("predicted_close"))
        baseline_prediction = finite_float(baseline_row.get("predicted_close"))
        if actual is None or challenger_prediction is None or baseline_prediction is None:
            continue
        squared_errors.append(
            (
                (challenger_prediction - actual) ** 2,
                (baseline_prediction - actual) ** 2,
            )
        )

    if not squared_errors:
        return {
            "baseline_model": baseline_model,
            "n": 0,
            "block_length": int(block_length),
            "samples": int(samples),
            "rmse_improvement_pct": None,
            "bootstrap_probability_improvement": None,
            "bootstrap_improvement_pct_p05": None,
            "bootstrap_improvement_pct_p95": None,
        }

    values = np.asarray(squared_errors, dtype=float)
    challenger_loss = values[:, 0]
    baseline_loss = values[:, 1]
    challenger_rmse = float(np.sqrt(np.mean(challenger_loss)))
    baseline_rmse = float(np.sqrt(np.mean(baseline_loss)))
    point_improvement = improvement_pct(challenger_rmse, baseline_rmse)

    n = len(values)
    block = max(1, min(int(block_length), n))
    block_count = int(math.ceil(n / block))
    rng = np.random.default_rng(seed)
    bootstrap_improvements: list[float] = []
    offsets = np.arange(block, dtype=int)
    for _ in range(int(samples)):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        challenger_sample_rmse = float(np.sqrt(np.mean(challenger_loss[indices])))
        baseline_sample_rmse = float(np.sqrt(np.mean(baseline_loss[indices])))
        sample_improvement = improvement_pct(challenger_sample_rmse, baseline_sample_rmse)
        if sample_improvement is not None:
            bootstrap_improvements.append(sample_improvement)

    bootstrap = np.asarray(bootstrap_improvements, dtype=float)
    return {
        "baseline_model": baseline_model,
        "n": int(n),
        "block_length": int(block),
        "samples": int(len(bootstrap)),
        "rmse_improvement_pct": point_improvement,
        "bootstrap_probability_improvement": (
            float(np.mean(bootstrap > 0.0)) if len(bootstrap) else None
        ),
        "bootstrap_improvement_pct_p05": (
            float(np.quantile(bootstrap, 0.05)) if len(bootstrap) else None
        ),
        "bootstrap_improvement_pct_p95": (
            float(np.quantile(bootstrap, 0.95)) if len(bootstrap) else None
        ),
    }


def evaluate(payload: dict[str, Any], profile: str) -> tuple[dict[str, Any], list[str], list[str]]:
    failures: list[str] = []
    warnings: list[str] = []
    folds = completed_folds(payload)
    required_folds = {"smoke": 3, "intermediate": 12, "full": 36}[profile]
    if folds < required_folds:
        failures.append(f"completed folds {folds} < required {required_folds}")
    if profile == "full" and bool(payload.get("partial")):
        failures.append("full candidate is marked partial")

    feature_budget = compact_feature_budget(payload)
    if feature_budget != EXPECTED_FEATURE_COUNT:
        failures.append(
            f"compact feature budget {feature_budget!r} != expected {EXPECTED_FEATURE_COUNT}"
        )

    models = {
        CHALLENGER_MODEL: model_metrics(payload, CHALLENGER_MODEL),
        INCUMBENT_MODEL: model_metrics(payload, INCUMBENT_MODEL),
        ANCHOR_MODEL: model_metrics(payload, ANCHOR_MODEL),
    }
    for name, metrics in models.items():
        if not metrics:
            failures.append(f"missing regression leaderboard row: {name}")
        elif finite_float(metrics.get("price_rmse")) is None:
            failures.append(f"missing finite price_rmse: {name}")

    challenger = models[CHALLENGER_MODEL] or {}
    incumbent = models[INCUMBENT_MODEL] or {}
    anchor = models[ANCHOR_MODEL] or {}
    improvement_vs_incumbent = improvement_pct(
        challenger.get("price_rmse"), incumbent.get("price_rmse")
    )
    improvement_vs_anchor = improvement_pct(
        challenger.get("price_rmse"), anchor.get("price_rmse")
    )
    return_mae_ratio = None
    challenger_return_mae = finite_float(challenger.get("return_mae"))
    incumbent_return_mae = finite_float(incumbent.get("return_mae"))
    if challenger_return_mae is not None and incumbent_return_mae:
        return_mae_ratio = challenger_return_mae / incumbent_return_mae
    directional_delta = None
    challenger_direction = finite_float(challenger.get("directional_accuracy"))
    incumbent_direction = finite_float(incumbent.get("directional_accuracy"))
    if challenger_direction is not None and incumbent_direction is not None:
        directional_delta = challenger_direction - incumbent_direction

    bootstrap = {
        baseline: moving_block_rmse_improvement(payload, baseline)
        for baseline in (INCUMBENT_MODEL, ANCHOR_MODEL)
    }
    expected_matched_rows = folds * 30
    for baseline, evidence in bootstrap.items():
        matched_rows = int(evidence.get("n") or 0)
        if matched_rows < expected_matched_rows:
            failures.append(
                f"matched bootstrap rows vs {baseline} are {matched_rows} "
                f"< expected {expected_matched_rows}"
            )

    gate_checks = [
        (
            improvement_vs_incumbent is not None and improvement_vs_incumbent >= 1.0,
            f"RMSE improvement vs incumbent is {improvement_vs_incumbent!r}% (need >= 1.0%)",
        ),
        (
            improvement_vs_anchor is not None and improvement_vs_anchor >= 0.5,
            f"RMSE improvement vs no-change is {improvement_vs_anchor!r}% (need >= 0.5%)",
        ),
        (
            return_mae_ratio is not None and return_mae_ratio <= 1.005,
            f"return MAE ratio vs incumbent is {return_mae_ratio!r} (need <= 1.005)",
        ),
        (
            directional_delta is not None and directional_delta >= -0.02,
            f"directional accuracy delta is {directional_delta!r} (need >= -0.02)",
        ),
    ]
    for baseline in (INCUMBENT_MODEL, ANCHOR_MODEL):
        probability = finite_float(
            bootstrap[baseline].get("bootstrap_probability_improvement")
        )
        gate_checks.append(
            (
                probability is not None
                and probability >= (0.80 if profile == "full" else 0.75),
                f"block-bootstrap P(improvement) vs {baseline} is {probability!r} "
                f"(need >= {0.80 if profile == 'full' else 0.75})",
            )
        )

    for passed, message in gate_checks:
        if not passed:
            (failures if profile != "smoke" else warnings).append(message)

    report = {
        "schema_version": 1,
        "evaluated_at_utc": dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "profile": profile,
        "gate_status": "FAIL" if failures else "PASS",
        "partial": bool(payload.get("partial")),
        "folds_completed": folds,
        "feature_budget": feature_budget,
        "challenger_model": CHALLENGER_MODEL,
        "incumbent_model": INCUMBENT_MODEL,
        "models": models,
        "rmse_improvement_vs_incumbent_pct": improvement_vs_incumbent,
        "rmse_improvement_vs_anchor_pct": improvement_vs_anchor,
        "return_mae_ratio_vs_incumbent": return_mae_ratio,
        "directional_accuracy_delta_vs_incumbent": directional_delta,
        "moving_block_bootstrap": bootstrap,
        "expected_matched_rows": expected_matched_rows,
        "failures": failures,
        "warnings": warnings,
    }
    return report, failures, warnings


def write_markdown(path: Path, report: dict[str, Any]) -> None:
    lines = [
        f"# Compact h30 evaluation: {report['gate_status']}",
        "",
        f"- Profile: `{report['profile']}`",
        f"- Folds: `{report['folds_completed']}`",
        f"- Feature budget: `{report['feature_budget']}`",
        f"- RMSE improvement vs incumbent: `{report['rmse_improvement_vs_incumbent_pct']}`%",
        f"- RMSE improvement vs no-change: `{report['rmse_improvement_vs_anchor_pct']}`%",
        f"- Return-MAE ratio vs incumbent: `{report['return_mae_ratio_vs_incumbent']}`",
        f"- Directional-accuracy delta: `{report['directional_accuracy_delta_vs_incumbent']}`",
        "",
    ]
    for baseline, evidence in report["moving_block_bootstrap"].items():
        lines.append(
            f"- Bootstrap vs `{baseline}`: P(improvement) "
            f"`{evidence.get('bootstrap_probability_improvement')}`, 90% interval "
            f"`[{evidence.get('bootstrap_improvement_pct_p05')}, "
            f"{evidence.get('bootstrap_improvement_pct_p95')}]`%"
        )
    if report["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {item}" for item in report["failures"])
    if report["warnings"]:
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {item}" for item in report["warnings"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument(
        "--profile",
        choices=("smoke", "intermediate", "full"),
        default="smoke",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("tests/phase0/h30_compact_eval_report.json"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("tests/phase0/h30_compact_eval_report.md"),
    )
    args = parser.parse_args(argv)

    payload = load_json(args.candidate)
    report, failures, _ = evaluate(payload, args.profile)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )
    write_markdown(args.output_md, report)
    print(args.output_md.read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
