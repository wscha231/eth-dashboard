"""Head-specific loss functions for ForecastBundleV2 research.

These functions are deliberately small and model-agnostic.  They define the
measurement boundary for R1; model selection/promotion remains governed by the
predeclared rules in issue #56.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

EPS = 1e-12


def _arrays(actual: Iterable[float], predicted: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if y.shape != p.shape or y.ndim != 1 or not len(y):
        raise ValueError("matched non-empty one-dimensional arrays required")
    if not np.isfinite(y).all() or not np.isfinite(p).all():
        raise ValueError("finite values required")
    return y, p


def center_metrics(actual_log_return, predicted_log_return) -> dict:
    """Evaluate central return in simple-return space, as frozen in #56."""
    y, p = _arrays(actual_log_return, predicted_log_return)
    y_simple, p_simple = np.expm1(y), np.expm1(p)
    errors = p_simple - y_simple
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "median_pinball": float(np.mean(0.5 * np.abs(errors))),
        "rows": int(len(y)),
    }


def qlike(actual_variance, predicted_variance) -> float:
    y, p = _arrays(actual_variance, predicted_variance)
    if np.any(y < 0) or np.any(p <= 0):
        raise ValueError("variance must be non-negative actual and positive prediction")
    y = np.maximum(y, EPS)
    p = np.maximum(p, EPS)
    ratio = y / p
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def volatility_metrics(actual_variance, predicted_variance) -> dict:
    y, p = _arrays(actual_variance, predicted_variance)
    return {
        "qlike": qlike(y, p),
        "rmse_variance": float(np.sqrt(np.mean((p - y) ** 2))),
        "rows": int(len(y)),
    }


def pinball(actual: np.ndarray, predicted: np.ndarray, level: float) -> float:
    if not 0 < level < 1:
        raise ValueError("quantile level must be in (0,1)")
    error = actual - predicted
    return float(np.mean(np.maximum(level * error, (level - 1.0) * error)))


def distribution_metrics(actual_log_return, q10, q50, q90) -> dict:
    y, lo = _arrays(actual_log_return, q10)
    _, med = _arrays(actual_log_return, q50)
    _, hi = _arrays(actual_log_return, q90)
    if np.any(lo > med) or np.any(med > hi):
        raise ValueError("crossing quantiles")
    alpha = 0.20
    interval_score = (hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0)
    # One-central-interval WIS: w0=1/2 for median, w1=alpha/2 for IS_alpha,
    # normalized by K+1/2 (=1.5 for K=1).
    wis = (0.5 * np.abs(y - med) + (alpha / 2) * interval_score) / 1.5
    return {
        "wis80": float(np.mean(wis)),
        "pinball10": pinball(y, lo, 0.10),
        "pinball50": pinball(y, med, 0.50),
        "pinball90": pinball(y, hi, 0.90),
        "coverage80": float(np.mean((y >= lo) & (y <= hi))),
        "mean_width_logreturn": float(np.mean(hi - lo)),
        "rows": int(len(y)),
    }


def _ece_binary(truth: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    total = len(truth)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (probability >= lo) & (probability <= hi if i == bins - 1 else probability < hi)
        if not mask.any():
            continue
        result += float(mask.mean()) * abs(float(probability[mask].mean()) - float(truth[mask].mean()))
    return float(result) if total else float("nan")


def event_metrics(terminal, terminal_probability, up, hit_up, down, hit_down) -> dict:
    terminal = np.asarray(list(terminal), dtype=int)
    probs = np.asarray(terminal_probability, dtype=float)
    up = np.asarray(list(up), dtype=int)
    down = np.asarray(list(down), dtype=int)
    hit_up = np.asarray(list(hit_up), dtype=float)
    hit_down = np.asarray(list(hit_down), dtype=float)
    n = len(terminal)
    if not n or probs.shape != (n, 3) or any(len(x) != n for x in (up, down, hit_up, hit_down)):
        raise ValueError("matched terminal/path arrays required")
    if np.any((probs < 0) | (probs > 1)) or not np.allclose(probs.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("invalid terminal probabilities")
    if np.any((hit_up < 0) | (hit_up > 1)) or np.any((hit_down < 0) | (hit_down > 1)):
        raise ValueError("invalid path probabilities")
    truth = np.eye(3)[terminal]
    terminal_brier = float(np.mean(np.sum((probs - truth) ** 2, axis=1) / 2.0))
    path_brier = float(np.mean(((hit_up - up) ** 2 + (hit_down - down) ** 2) / 2.0))
    logloss = float(-np.mean(np.log(np.clip(probs[np.arange(n), terminal], EPS, 1.0))))
    terminal_ece = max(_ece_binary((terminal == j).astype(int), probs[:, j]) for j in range(3))
    path_ece = max(_ece_binary(up, hit_up), _ece_binary(down, hit_down))
    if not all(math.isfinite(v) for v in (terminal_brier, path_brier, logloss, terminal_ece, path_ece)):
        raise ValueError("non-finite event metric")
    return {
        "terminal_brier": terminal_brier,
        "path_brier": path_brier,
        "combined_brier": float((terminal_brier + path_brier) / 2.0),
        "terminal_logloss": logloss,
        "ece": float(max(terminal_ece, path_ece)),
        "rows": int(n),
    }
