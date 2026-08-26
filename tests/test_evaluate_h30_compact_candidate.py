from __future__ import annotations

import math

from scripts.evaluate_h30_compact_candidate import (
    ANCHOR_MODEL,
    CHALLENGER_MODEL,
    INCUMBENT_MODEL,
    evaluate,
    json_safe,
)


def candidate_payload(*, folds: int, partial: bool, challenger_error: float) -> dict:
    predictions = []
    model_errors = {
        CHALLENGER_MODEL: challenger_error,
        INCUMBENT_MODEL: 2.0,
        ANCHOR_MODEL: 3.0,
    }
    row_count = folds * 30
    for row_index in range(row_count):
        actual_close = 1000.0 + 0.25 * row_index
        for model, error in model_errors.items():
            predictions.append(
                {
                    "horizon_days": 30,
                    "head": "regression",
                    "model": model,
                    "prediction_date": f"2024-01-{(row_index % 28) + 1:02d}-{row_index:04d}",
                    "target_date": f"2024-02-{(row_index % 28) + 1:02d}-{row_index:04d}",
                    "actual_close": actual_close,
                    "predicted_close": actual_close + error,
                }
            )

    def metrics(model: str, error: float) -> dict:
        return {
            "model": model,
            "price_rmse": abs(error),
            "price_mae": abs(error),
            "return_mae": abs(error) / 1000.0,
            "directional_accuracy": 0.55 if model == CHALLENGER_MODEL else 0.54,
            "folds": float(folds),
        }

    return {
        "partial": partial,
        "folds_completed": {"30": folds},
        "model_registry": {
            "30": {
                "regression": {
                    CHALLENGER_MODEL: {
                        "params": {"feature_budget__max_features": 192}
                    }
                },
                "classification": {},
            }
        },
        "horizons": {
            "30": {
                "regression_leaderboard": [
                    metrics(model, error) for model, error in model_errors.items()
                ],
                "classification_leaderboard": [],
                "predictions": predictions,
            }
        },
    }


def test_full_compact_gate_passes_strong_matched_improvement() -> None:
    report, failures, warnings = evaluate(
        candidate_payload(folds=36, partial=False, challenger_error=1.0),
        "full",
    )

    assert failures == []
    assert warnings == []
    assert report["gate_status"] == "PASS"
    assert math.isclose(report["rmse_improvement_vs_incumbent_pct"], 50.0)
    assert report["moving_block_bootstrap"][INCUMBENT_MODEL][
        "bootstrap_probability_improvement"
    ] == 1.0


def test_smoke_gate_warns_on_weak_candidate_without_blocking_infrastructure() -> None:
    report, failures, warnings = evaluate(
        candidate_payload(folds=3, partial=True, challenger_error=4.0),
        "smoke",
    )

    assert failures == []
    assert report["gate_status"] == "PASS"
    assert warnings


def test_intermediate_gate_requires_twelve_folds() -> None:
    report, failures, _ = evaluate(
        candidate_payload(folds=11, partial=True, challenger_error=1.0),
        "intermediate",
    )

    assert report["gate_status"] == "FAIL"
    assert any("required 12" in failure for failure in failures)


def test_full_compact_gate_rejects_partial_or_wrong_budget() -> None:
    payload = candidate_payload(folds=3, partial=True, challenger_error=1.0)
    payload["model_registry"]["30"]["regression"][CHALLENGER_MODEL]["params"][
        "feature_budget__max_features"
    ] = 128

    report, failures, _ = evaluate(payload, "full")

    assert report["gate_status"] == "FAIL"
    assert any("completed folds" in failure for failure in failures)
    assert any("marked partial" in failure for failure in failures)
    assert any("feature budget" in failure for failure in failures)


def test_report_json_sanitizer_replaces_non_finite_metrics() -> None:
    assert json_safe({"nan": float("nan"), "inf": float("inf")}) == {
        "nan": None,
        "inf": None,
    }
