"""Offline metrics and conformal helpers for adaptive return intervals.

The functions in this module are intentionally independent from the promoted
7/30-day forecast path.  They operate on already-generated out-of-fold
predictions and never write public forecast artifacts.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

INTERVAL_ALPHA = 0.10
LOWER_QUANTILE = 0.05
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.95
ONLINE_WINDOW = 360
MIN_REGIME_SCORES = 90
VOLATILITY_SCALE_FLOOR = 0.01

BASELINE_METHOD = "residual_conformal"
CQR_METHOD = "cqr_histgradient"
VOLATILITY_ACI_METHOD = "volatility_scaled_aci"
REGIME_ACI_METHOD = "regime_reset_aci"
INTERVAL_METHODS = (
    BASELINE_METHOD,
    CQR_METHOD,
    VOLATILITY_ACI_METHOD,
    REGIME_ACI_METHOD,
)


def finite_values(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def conformal_quantile(values: Any, coverage: float) -> float:
    """Return the finite-sample corrected empirical conformal quantile."""
    finite = finite_values(values)
    if not 0.0 < float(coverage) <= 1.0:
        raise ValueError("coverage must be in (0, 1]")
    if finite.size < 20:
        raise ValueError("At least 20 finite calibration scores are required")
    rank = math.ceil((finite.size + 1) * float(coverage))
    rank = min(max(rank, 1), finite.size)
    return float(np.partition(finite, rank - 1)[rank - 1])


def empirical_quantile(values: Any, quantile: float) -> float:
    finite = finite_values(values)
    if not 0.0 <= float(quantile) <= 1.0:
        raise ValueError("quantile must be in [0, 1]")
    if finite.size < 20:
        raise ValueError("At least 20 finite values are required")
    return float(np.quantile(finite, float(quantile)))


def repair_interval(
    lower: float,
    median: float,
    upper: float,
) -> tuple[float, float, float]:
    values = np.asarray([lower, median, upper], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Interval values must be finite")
    ordered = np.sort(values)
    return float(ordered[0]), float(ordered[1]), float(ordered[2])


def residual_conformal_interval(
    point_prediction: float,
    residual_history: Any,
    *,
    alpha: float = INTERVAL_ALPHA,
) -> tuple[float, float, float]:
    residuals = finite_values(residual_history)
    center = float(np.median(residuals))
    radius = conformal_quantile(np.abs(residuals - center), 1.0 - float(alpha))
    signed_lower = empirical_quantile(residuals, float(alpha) / 2.0)
    signed_upper = empirical_quantile(residuals, 1.0 - float(alpha) / 2.0)
    lower_offset = min(signed_lower, center - radius)
    upper_offset = max(signed_upper, center + radius)
    return repair_interval(
        float(point_prediction) + lower_offset,
        float(point_prediction) + center,
        float(point_prediction) + upper_offset,
    )


def conformalized_quantile_interval(
    lower_prediction: float,
    median_prediction: float,
    upper_prediction: float,
    score_history: Any,
    *,
    alpha: float = INTERVAL_ALPHA,
) -> tuple[float, float, float]:
    correction = conformal_quantile(score_history, 1.0 - float(alpha))
    return repair_interval(
        float(lower_prediction) - correction,
        float(median_prediction),
        float(upper_prediction) + correction,
    )


def scaled_aci_interval(
    point_prediction: float,
    scale: float,
    normalized_residual_history: Any,
    *,
    adaptive_alpha: float,
) -> tuple[float, float, float]:
    if not np.isfinite(scale) or float(scale) <= 0.0:
        raise ValueError("scale must be positive and finite")
    alpha = float(np.clip(adaptive_alpha, 0.02, 0.25))
    residuals = finite_values(normalized_residual_history)
    center = float(np.median(residuals))
    radius = conformal_quantile(np.abs(residuals - center), 1.0 - alpha)
    return repair_interval(
        float(point_prediction) + (center - radius) * float(scale),
        float(point_prediction) + center * float(scale),
        float(point_prediction) + (center + radius) * float(scale),
    )


def update_adaptive_alpha(
    current_alpha: float,
    *,
    missed: bool,
    target_alpha: float = INTERVAL_ALPHA,
    step_size: float = 0.01,
) -> float:
    error = 1.0 if missed else 0.0
    updated = float(current_alpha) + float(step_size) * (float(target_alpha) - error)
    return float(np.clip(updated, 0.02, 0.25))


def volatility_scale(
    daily_volatility: pd.Series | np.ndarray | list[float],
    *,
    horizon_days: int,
    floor: float = VOLATILITY_SCALE_FLOOR,
) -> np.ndarray:
    values = np.asarray(daily_volatility, dtype=float)
    scaled = np.abs(values) * math.sqrt(float(horizon_days))
    scaled[~np.isfinite(scaled)] = float(floor)
    return np.maximum(scaled, float(floor))


def pinball_loss(actual: Any, prediction: Any, quantile: float) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    qhat = np.asarray(prediction, dtype=float)
    residual = y - qhat
    return np.maximum(float(quantile) * residual, (float(quantile) - 1.0) * residual)


def interval_score(
    actual: Any,
    lower: Any,
    upper: Any,
    *,
    alpha: float = INTERVAL_ALPHA,
) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    width = hi - lo
    below = np.maximum(lo - y, 0.0)
    above = np.maximum(y - hi, 0.0)
    return width + (2.0 / float(alpha)) * (below + above)


def weighted_interval_score(
    actual: Any,
    lower: Any,
    median: Any,
    upper: Any,
    *,
    alpha: float = INTERVAL_ALPHA,
) -> np.ndarray:
    y = np.asarray(actual, dtype=float)
    med = np.asarray(median, dtype=float)
    score = interval_score(actual, lower, upper, alpha=alpha)
    return (0.5 * np.abs(y - med) + (float(alpha) / 2.0) * score) / 1.5


def pointwise_interval_losses(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"actual_return", "lower", "median", "upper"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Interval frame is missing columns: {missing}")
    output = frame.copy()
    y = pd.to_numeric(output["actual_return"], errors="coerce").to_numpy(float)
    lower = pd.to_numeric(output["lower"], errors="coerce").to_numpy(float)
    median = pd.to_numeric(output["median"], errors="coerce").to_numpy(float)
    upper = pd.to_numeric(output["upper"], errors="coerce").to_numpy(float)
    if not np.isfinite(np.column_stack([y, lower, median, upper])).all():
        raise ValueError("Interval predictions must be finite")
    if np.any(lower > median) or np.any(median > upper):
        raise ValueError("Interval predictions must be ordered")
    output["pinball_q05"] = pinball_loss(y, lower, LOWER_QUANTILE)
    output["pinball_q50"] = pinball_loss(y, median, MEDIAN_QUANTILE)
    output["pinball_q95"] = pinball_loss(y, upper, UPPER_QUANTILE)
    output["interval_score"] = interval_score(y, lower, upper)
    output["wis"] = weighted_interval_score(y, lower, median, upper)
    output["covered"] = (y >= lower) & (y <= upper)
    output["lower_covered"] = y >= lower
    output["upper_covered"] = y <= upper
    output["width"] = upper - lower
    output["upside_exceedance"] = np.maximum(y - upper, 0.0)
    output["downside_exceedance"] = np.maximum(lower - y, 0.0)
    return output


def summarize_interval_predictions(frame: pd.DataFrame) -> dict[str, Any]:
    scored = pointwise_interval_losses(frame)
    y = scored["actual_return"].to_numpy(float)
    up_tail = y >= 0.12
    down_tail = y <= -0.12

    def tail_summary(mask: np.ndarray, exceedance_column: str) -> dict[str, Any]:
        subset = scored.loc[mask]
        if subset.empty:
            return {
                "rows": 0,
                "coverage": None,
                "exceedance_rate": None,
                "mean_exceedance": None,
                "max_exceedance": None,
            }
        coverage_column = (
            "upper_covered"
            if exceedance_column == "upside_exceedance"
            else "lower_covered"
        )
        return {
            "rows": len(subset),
            "coverage": float(subset[coverage_column].mean()),
            "exceedance_rate": float((subset[exceedance_column] > 0.0).mean()),
            "mean_exceedance": float(subset[exceedance_column].mean()),
            "max_exceedance": float(subset[exceedance_column].max()),
        }

    return {
        "rows": len(scored),
        "pinball_q05": float(scored["pinball_q05"].mean()),
        "pinball_q50": float(scored["pinball_q50"].mean()),
        "pinball_q95": float(scored["pinball_q95"].mean()),
        "weighted_interval_score": float(scored["wis"].mean()),
        "interval_score": float(scored["interval_score"].mean()),
        "interval_coverage": float(scored["covered"].mean()),
        "lower_coverage": float(scored["lower_covered"].mean()),
        "upper_coverage": float(scored["upper_covered"].mean()),
        "average_width": float(scored["width"].mean()),
        "median_width": float(scored["width"].median()),
        "median_absolute_error": float(
            np.median(np.abs(y - scored["median"].to_numpy(float)))
        ),
        "up_tail": tail_summary(up_tail, "upside_exceedance"),
        "down_tail": tail_summary(down_tail, "downside_exceedance"),
    }


def moving_block_bootstrap_interval_improvement(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    samples: int = 2000,
    block_length: int = 7,
    seed: int = 42,
) -> dict[str, Any]:
    candidate_scored = pointwise_interval_losses(candidate).set_index("prediction_date")
    baseline_scored = pointwise_interval_losses(baseline).set_index("prediction_date")
    common = candidate_scored.index.intersection(baseline_scored.index)
    if len(common) < 60:
        raise ValueError("At least 60 matched interval predictions are required")
    candidate_scored = candidate_scored.loc[common]
    baseline_scored = baseline_scored.loc[common]
    wis_diff = baseline_scored["wis"].to_numpy(float) - candidate_scored[
        "wis"
    ].to_numpy(float)
    q95_diff = baseline_scored["pinball_q95"].to_numpy(float) - candidate_scored[
        "pinball_q95"
    ].to_numpy(float)
    n = len(common)
    block = max(1, min(int(block_length), n))
    rng = np.random.default_rng(int(seed))
    starts = np.arange(0, n - block + 1, dtype=int)
    draw_count = math.ceil(n / block)
    wis_draws = np.empty(int(samples), dtype=float)
    q95_draws = np.empty(int(samples), dtype=float)
    for draw in range(int(samples)):
        chosen = rng.choice(starts, size=draw_count, replace=True)
        positions = np.concatenate(
            [np.arange(start, start + block, dtype=int) for start in chosen]
        )[:n]
        wis_draws[draw] = float(np.mean(wis_diff[positions]))
        q95_draws[draw] = float(np.mean(q95_diff[positions]))
    return {
        "rows": int(n),
        "samples": int(samples),
        "block_length": int(block),
        "wis_improvement": float(np.mean(wis_diff)),
        "wis_probability_improvement": float(np.mean(wis_draws > 0.0)),
        "wis_improvement_p05": float(np.quantile(wis_draws, 0.05)),
        "wis_improvement_p95": float(np.quantile(wis_draws, 0.95)),
        "q95_improvement": float(np.mean(q95_diff)),
        "q95_probability_improvement": float(np.mean(q95_draws > 0.0)),
        "q95_improvement_p05": float(np.quantile(q95_draws, 0.05)),
        "q95_improvement_p95": float(np.quantile(q95_draws, 0.95)),
    }
