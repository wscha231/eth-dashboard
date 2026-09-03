"""Leakage-safe metrics and calibration for offline tail evaluation.

This module is deliberately independent from the live 7/30-day forecast
selection path.  It defines labels, event episodes, sample weights,
calibration, alert operating points, and rare-event metrics used by the
offline tail candidate runner.  Nothing here changes a production forecast.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    roc_auc_score,
)

TAIL_HORIZON_DAYS = 3
PRIMARY_RETURN_THRESHOLD = 0.12
DYNAMIC_VOL_MULTIPLIER = 2.0
DYNAMIC_THRESHOLD_FLOOR = 0.10
DYNAMIC_THRESHOLD_CAP = 0.25
DYNAMIC_VOL_WINDOW = 30
DYNAMIC_VOL_MIN_PERIODS = 20
MAX_FALSE_ALERT_EPISODES_PER_90_DAYS = 3.0
MAX_CORE_FEATURES = 64

PRIMARY_LABEL_COLUMN = "tail_up_primary"
DYNAMIC_LABEL_COLUMN = "tail_up_dynamic"
FUTURE_RETURN_COLUMN = "tail_future_return_3d"
DYNAMIC_THRESHOLD_COLUMN = "tail_dynamic_threshold"


# Fixed before candidate evaluation.  The allowlist contains only mature
# price/volume features generated from ETH and BTC daily data.  Sparse vendor,
# derivatives, on-chain, prediction-history, and target columns are excluded.
CORE_TAIL_FEATURES: tuple[str, ...] = (
    "eth_return_1",
    "eth_return_2",
    "eth_return_3",
    "eth_return_5",
    "eth_return_7",
    "eth_return_14",
    "eth_return_21",
    "eth_return_30",
    "eth_return_60",
    "eth_return_90",
    "eth_range_pct",
    "eth_close_open_pct",
    "eth_up_streak",
    "eth_down_streak",
    "eth_up_day_ratio_7",
    "eth_up_day_ratio_14",
    "eth_rebound_from_20d_low",
    "eth_rebound_from_60d_low",
    "eth_rebound_from_180d_low",
    "eth_distance_to_20d_high",
    "eth_distance_to_60d_high",
    "eth_distance_to_180d_high",
    "eth_ma_7_ratio",
    "eth_ma_14_ratio",
    "eth_ma_30_ratio",
    "eth_ma_90_ratio",
    "eth_ma_180_ratio",
    "eth_vol_7",
    "eth_vol_14",
    "eth_vol_30",
    "eth_vol_60",
    "eth_vol_90",
    "eth_vol_180",
    "eth_vol_30_180_ratio",
    "eth_downside_vol_14",
    "eth_downside_vol_30",
    "eth_vol_regime_7_30",
    "eth_vol_squeeze_20_90",
    "eth_range_expansion_z_30",
    "eth_volume_impulse_3_30",
    "eth_tail_event_pressure",
    "eth_volume_change_1",
    "eth_volume_change_7",
    "eth_volume_z_30",
    "eth_rsi_14",
    "eth_rsi_28",
    "eth_macd_hist",
    "eth_macd_hist_change_3",
    "eth_atr_14_ratio",
    "eth_drawdown_30",
    "eth_drawdown_90",
    "eth_drawdown_180",
    "eth_close_z_20",
    "eth_bollinger_width_20",
    "eth_bollinger_z_20",
    "eth_adx_14",
    "eth_adx_trend_bias_14",
    "btc_return_1",
    "btc_return_3",
    "btc_return_7",
    "btc_return_30",
    "btc_vol_14",
    "eth_btc_corr_30",
    "eth_vs_btc_strength_14",
)


def _numeric_series(
    values: Iterable[Any] | pd.Series, index: pd.Index | None = None
) -> pd.Series:
    if isinstance(values, pd.Series):
        result = pd.to_numeric(values, errors="coerce").astype(float)
        if index is not None:
            result = result.reindex(index)
        return result
    return pd.Series(values, index=index, dtype=float)


def _validated_time_index(index: pd.Index) -> pd.DatetimeIndex:
    timestamps = pd.DatetimeIndex(pd.to_datetime(index, errors="raise"))
    if timestamps.has_duplicates:
        raise ValueError("Tail-event timestamps must be unique")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Tail-event timestamps must be increasing")
    return timestamps


def build_tail_targets(
    close: pd.Series,
    *,
    horizon: int = TAIL_HORIZON_DAYS,
    primary_threshold: float = PRIMARY_RETURN_THRESHOLD,
    volatility_window: int = DYNAMIC_VOL_WINDOW,
    volatility_min_periods: int = DYNAMIC_VOL_MIN_PERIODS,
    volatility_multiplier: float = DYNAMIC_VOL_MULTIPLIER,
    dynamic_floor: float = DYNAMIC_THRESHOLD_FLOOR,
    dynamic_cap: float = DYNAMIC_THRESHOLD_CAP,
) -> pd.DataFrame:
    """Build the fixed and volatility-normalized upside labels.

    The volatility estimate is shifted by one row.  Therefore the dynamic
    threshold at origin ``t`` only uses returns ending at ``t-1``.  The final
    ``horizon`` rows remain unavailable instead of being coerced to negatives.
    """
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if not 0.0 < primary_threshold < 1.0:
        raise ValueError("primary_threshold must be between zero and one")
    if not 0.0 < dynamic_floor <= dynamic_cap < 1.0:
        raise ValueError("dynamic threshold bounds are invalid")

    close_series = pd.to_numeric(close, errors="coerce").astype(float)
    future_return = close_series.shift(-horizon) / close_series - 1.0
    daily_return = close_series.pct_change(fill_method=None)
    trailing_volatility = (
        daily_return.rolling(
            int(volatility_window),
            min_periods=int(volatility_min_periods),
        )
        .std()
        .shift(1)
    )
    dynamic_threshold = (
        trailing_volatility * float(volatility_multiplier) * math.sqrt(float(horizon))
    ).clip(lower=float(dynamic_floor), upper=float(dynamic_cap))

    primary = pd.Series(np.nan, index=close_series.index, dtype=float)
    primary_valid = future_return.notna()
    primary.loc[primary_valid] = (
        future_return.loc[primary_valid] >= float(primary_threshold)
    ).astype(float)

    dynamic = pd.Series(np.nan, index=close_series.index, dtype=float)
    dynamic_valid = future_return.notna() & dynamic_threshold.notna()
    dynamic.loc[dynamic_valid] = (
        future_return.loc[dynamic_valid] >= dynamic_threshold.loc[dynamic_valid]
    ).astype(float)

    return pd.DataFrame(
        {
            FUTURE_RETURN_COLUMN: future_return,
            PRIMARY_LABEL_COLUMN: primary,
            DYNAMIC_THRESHOLD_COLUMN: dynamic_threshold.where(future_return.notna()),
            DYNAMIC_LABEL_COLUMN: dynamic,
        },
        index=close_series.index,
    )


def group_signal_episodes(
    signal: pd.Series | Iterable[Any],
    *,
    max_gap_days: int = TAIL_HORIZON_DAYS,
    max_span_days: int | None = None,
) -> pd.Series:
    """Assign an integer ID to active origins that belong to one episode."""
    if isinstance(signal, pd.Series):
        series = signal.copy()
    else:
        series = pd.Series(signal)
    timestamps = _validated_time_index(series.index)
    active = pd.to_numeric(series, errors="coerce").fillna(0.0).to_numpy() > 0.0
    identifiers = pd.Series(pd.NA, index=series.index, dtype="Int64")
    previous_timestamp: pd.Timestamp | None = None
    episode_start: pd.Timestamp | None = None
    episode_id = -1
    for position in np.flatnonzero(active):
        timestamp = pd.Timestamp(timestamps[position])
        if (
            previous_timestamp is None
            or (timestamp - previous_timestamp).days > int(max_gap_days)
            or (
                max_span_days is not None
                and episode_start is not None
                and (timestamp - episode_start).days >= int(max_span_days)
            )
        ):
            episode_id += 1
            episode_start = timestamp
        identifiers.iloc[position] = episode_id
        previous_timestamp = timestamp
    return identifiers


def summarize_signal_episodes(
    signal: pd.Series | Iterable[Any],
    *,
    max_gap_days: int = TAIL_HORIZON_DAYS,
    max_span_days: int | None = None,
) -> pd.DataFrame:
    if not isinstance(signal, pd.Series):
        signal = pd.Series(signal)
    identifiers = group_signal_episodes(
        signal,
        max_gap_days=max_gap_days,
        max_span_days=max_span_days,
    )
    rows: list[dict[str, Any]] = []
    for episode_id in identifiers.dropna().astype(int).unique():
        mask = identifiers == int(episode_id)
        episode_index = pd.DatetimeIndex(pd.to_datetime(identifiers.index[mask]))
        rows.append(
            {
                "episode_id": int(episode_id),
                "start": pd.Timestamp(episode_index.min()),
                "end": pd.Timestamp(episode_index.max()),
                "origin_count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows, columns=["episode_id", "start", "end", "origin_count"])


def build_episode_base_weights(
    labels: pd.Series,
    *,
    max_gap_days: int = TAIL_HORIZON_DAYS,
) -> pd.Series:
    """Give every positive episode unit mass before recency/class balancing."""
    numeric = _numeric_series(labels)
    weights = pd.Series(np.nan, index=numeric.index, dtype=float)
    valid = numeric.notna()
    weights.loc[valid & (numeric <= 0.0)] = 1.0
    episode_ids = group_signal_episodes(
        (numeric > 0.0).where(valid, 0.0),
        max_gap_days=max_gap_days,
    )
    for episode_id in episode_ids.dropna().astype(int).unique():
        mask = episode_ids == int(episode_id)
        weights.loc[mask] = 1.0 / float(mask.sum())
    return weights


def build_episode_sample_weights(
    labels: pd.Series,
    *,
    half_life_years: float = 2.0,
    minimum_recency_weight: float = 0.10,
    balance_classes: bool = True,
    max_gap_days: int = TAIL_HORIZON_DAYS,
) -> pd.Series:
    """Combine episode normalization, time decay, and event-class balancing."""
    numeric = _numeric_series(labels)
    timestamps = _validated_time_index(numeric.index)
    weights = build_episode_base_weights(numeric, max_gap_days=max_gap_days)
    valid = numeric.notna()
    if not bool(valid.any()):
        return weights

    latest = pd.Timestamp(timestamps[valid.to_numpy()].max())
    age_years = np.asarray(
        (latest - timestamps) / pd.Timedelta(days=365.25),
        dtype=float,
    )
    if half_life_years <= 0.0:
        recency = np.ones(len(numeric), dtype=float)
    else:
        recency = np.power(0.5, age_years / float(half_life_years))
        recency = np.clip(recency, float(minimum_recency_weight), 1.0)
    weights.loc[valid] = weights.loc[valid] * recency[valid.to_numpy()]

    positive = valid & (numeric > 0.0)
    negative = valid & (numeric <= 0.0)
    if balance_classes and bool(positive.any()) and bool(negative.any()):
        positive_mass = float(weights.loc[positive].sum())
        negative_mass = float(weights.loc[negative].sum())
        if positive_mass > 0.0:
            weights.loc[positive] *= negative_mass / positive_mass

    mean_weight = float(weights.loc[valid].mean())
    if mean_weight > 0.0:
        weights.loc[valid] /= mean_weight
    return weights


def validate_probabilities(probabilities: pd.Series | Iterable[Any]) -> pd.Series:
    numeric = _numeric_series(probabilities)
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain NaN or infinity")
    if bool(((numeric < 0.0) | (numeric > 1.0)).any()):
        raise ValueError("Probabilities must be between zero and one")
    return numeric


@dataclass(frozen=True)
class SigmoidCalibrator:
    """Serializable coefficients for a one-dimensional Platt mapping."""

    coefficient: float
    intercept: float
    method: str = "platt"

    def predict(self, raw_probabilities: pd.Series | Iterable[Any]) -> np.ndarray:
        raw = validate_probabilities(raw_probabilities).to_numpy(dtype=float)
        logits = np.log(np.clip(raw, 1e-6, 1.0 - 1e-6) / np.clip(1.0 - raw, 1e-6, 1.0))
        linear = np.clip(self.intercept + self.coefficient * logits, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-linear))


def fit_sigmoid_calibrator(
    raw_probabilities: pd.Series | Iterable[Any],
    labels: pd.Series | Iterable[Any],
) -> SigmoidCalibrator:
    raw = validate_probabilities(raw_probabilities)
    y = _numeric_series(labels, raw.index)
    valid = y.notna()
    raw = raw.loc[valid]
    y = y.loc[valid].astype(int)
    if len(y) < 20:
        raise ValueError("Calibration requires at least 20 prior observations")
    if y.nunique() != 2:
        raise ValueError("Calibration requires both event and non-event labels")
    logits = np.log(
        np.clip(raw.to_numpy(dtype=float), 1e-6, 1.0 - 1e-6)
        / np.clip(1.0 - raw.to_numpy(dtype=float), 1e-6, 1.0)
    ).reshape(-1, 1)
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    model.fit(logits, y.to_numpy(dtype=int))
    coefficient = float(model.coef_[0, 0])
    intercept = float(model.intercept_[0])
    method = "platt"
    # Platt calibration must never reverse a candidate's ranking.  When the
    # unconstrained slope is negative, retain the model's ordering (slope=1)
    # and solve only a prior intercept.  This avoids both inversion and the
    # degenerate constant mapping that a zero-slope boundary would create.
    if coefficient < 0.0:
        coefficient = 1.0
        prevalence = float(y.mean())
        lower, upper = -35.0, 35.0
        flat_logits = logits.reshape(-1)
        for _ in range(100):
            midpoint = 0.5 * (lower + upper)
            mapped = 1.0 / (1.0 + np.exp(-np.clip(midpoint + flat_logits, -35.0, 35.0)))
            if float(mapped.mean()) < prevalence:
                lower = midpoint
            else:
                upper = midpoint
        intercept = float(0.5 * (lower + upper))
        method = "monotone_prior_intercept_fallback"
    return SigmoidCalibrator(
        coefficient=coefficient,
        intercept=intercept,
        method=method,
    )


def alert_event_metrics(
    labels: pd.Series,
    alerts: pd.Series,
    *,
    event_gap_days: int = TAIL_HORIZON_DAYS,
    alert_gap_days: int = TAIL_HORIZON_DAYS,
    lead_window_days: int = TAIL_HORIZON_DAYS,
    alert_renewal_days: int = TAIL_HORIZON_DAYS,
) -> dict[str, Any]:
    """Score event and alert episodes, avoiding row-level duplicate credit."""
    y = _numeric_series(labels)
    alert_values = _numeric_series(alerts, y.index)
    valid = y.notna() & alert_values.notna()
    y = y.loc[valid]
    alert_values = alert_values.loc[valid]
    _validated_time_index(y.index)

    true_episodes = summarize_signal_episodes(y > 0.0, max_gap_days=event_gap_days)
    # A continuously active three-day alert is renewed every three days.  This
    # prevents an always-on signal from counting as one harmless episode for
    # an entire year while still collapsing duplicate alerts inside a horizon.
    alert_episodes = summarize_signal_episodes(
        alert_values > 0.0,
        max_gap_days=alert_gap_days,
        max_span_days=alert_renewal_days,
    )
    matched_alerts: set[int] = set()
    lead_times: list[float] = []
    detected = 0

    for event in true_episodes.itertuples(index=False):
        eligible: list[tuple[pd.Timestamp, int]] = []
        event_window_start = pd.Timestamp(event.start) - pd.Timedelta(
            days=int(lead_window_days)
        )
        for alert in alert_episodes.itertuples(index=False):
            alert_id = int(alert.episode_id)
            overlaps = pd.Timestamp(alert.end) >= event_window_start and pd.Timestamp(
                alert.start
            ) <= pd.Timestamp(event.end)
            if overlaps:
                eligible.append((pd.Timestamp(alert.start), alert_id))
        if not eligible:
            continue
        alert_start, _ = min(eligible)
        matched_alerts.update(alert_id for _, alert_id in eligible)
        detected += 1
        lead_times.append(
            max(0.0, float((pd.Timestamp(event.start) - alert_start).days))
        )

    true_count = len(true_episodes)
    alert_count = len(alert_episodes)
    false_alerts = max(0, alert_count - len(matched_alerts))
    evaluated_days = len(y)
    recall = float(detected / true_count) if true_count else None
    precision = float(len(matched_alerts) / alert_count) if alert_count else None
    return {
        "evaluated_days": evaluated_days,
        "event_episode_count": true_count,
        "alert_episode_count": alert_count,
        "detected_event_episode_count": int(detected),
        "matched_alert_episode_count": len(matched_alerts),
        "false_alert_episode_count": int(false_alerts),
        "false_alert_episodes_per_90_days": (
            float(false_alerts * 90.0 / evaluated_days) if evaluated_days else None
        ),
        "episode_recall": recall,
        "episode_precision": precision,
        "median_lead_days": float(np.median(lead_times)) if lead_times else None,
    }


def select_alert_threshold(
    labels: pd.Series,
    probabilities: pd.Series,
    *,
    max_false_alerts_per_90_days: float = MAX_FALSE_ALERT_EPISODES_PER_90_DAYS,
    event_gap_days: int = TAIL_HORIZON_DAYS,
    lead_window_days: int = TAIL_HORIZON_DAYS,
) -> dict[str, Any]:
    """Choose the most sensitive prior-only threshold within the alert budget."""
    y = _numeric_series(labels)
    p = validate_probabilities(_numeric_series(probabilities, y.index))
    valid = y.notna() & p.notna()
    y = y.loc[valid]
    p = p.loc[valid]
    if y.nunique() != 2:
        raise ValueError("Threshold selection requires both event and non-event labels")

    quantiles = np.quantile(p.to_numpy(dtype=float), np.linspace(0.0, 1.0, 101))
    grid = np.unique(
        np.concatenate(
            [
                np.linspace(0.01, 0.99, 99),
                quantiles,
                np.asarray([np.nextafter(float(p.max()), math.inf)]),
            ]
        )
    )
    candidates: list[
        tuple[tuple[float, float, float, float], float, dict[str, Any]]
    ] = []
    for threshold in grid:
        alerts = (p >= float(threshold)).astype(float)
        metrics = alert_event_metrics(
            y,
            alerts,
            event_gap_days=event_gap_days,
            alert_gap_days=event_gap_days,
            lead_window_days=lead_window_days,
        )
        false_rate = metrics["false_alert_episodes_per_90_days"]
        if (
            false_rate is None
            or false_rate > float(max_false_alerts_per_90_days) + 1e-12
        ):
            continue
        recall = metrics["episode_recall"]
        precision = metrics["episode_precision"]
        key = (
            float(recall) if recall is not None else -1.0,
            float(precision) if precision is not None else -1.0,
            -float(false_rate),
            -float(threshold),
        )
        candidates.append((key, float(threshold), metrics))
    if not candidates:
        raise ValueError("No alert threshold satisfies the false-alert budget")
    _, threshold, metrics = max(candidates, key=lambda item: item[0])
    return {
        "threshold": float(threshold),
        "max_false_alert_episodes_per_90_days": float(max_false_alerts_per_90_days),
        **metrics,
    }


def probability_metrics(
    labels: pd.Series,
    probabilities: pd.Series,
    *,
    baseline_probabilities: pd.Series | None = None,
    alerts: pd.Series | None = None,
) -> dict[str, Any]:
    y = _numeric_series(labels)
    p = validate_probabilities(_numeric_series(probabilities, y.index))
    valid = y.notna() & p.notna()
    y = y.loc[valid].astype(int)
    p = p.loc[valid]
    if y.empty:
        raise ValueError("No valid probability observations")
    prevalence = float(y.mean())
    average_precision = float(average_precision_score(y, p)) if y.sum() else 0.0
    brier = float(brier_score_loss(y, p))
    roc_auc = float(roc_auc_score(y, p)) if y.nunique() == 2 else None

    baseline_brier: float | None = None
    brier_skill: float | None = None
    if baseline_probabilities is not None:
        baseline = validate_probabilities(
            _numeric_series(baseline_probabilities, y.index)
        )
        baseline_brier = float(brier_score_loss(y, baseline))
        if baseline_brier > 0.0:
            brier_skill = float(1.0 - brier / baseline_brier)

    balanced_accuracy: float | None = None
    event_metrics: dict[str, Any] | None = None
    if alerts is not None:
        aligned_alerts = _numeric_series(alerts, y.index)
        if aligned_alerts.isna().any():
            raise ValueError("Alerts contain unavailable values")
        balanced_accuracy = (
            float(balanced_accuracy_score(y, aligned_alerts.astype(int)))
            if y.nunique() == 2
            else None
        )
        event_metrics = alert_event_metrics(y, aligned_alerts)

    return {
        "n": len(y),
        "positive_rows": int(y.sum()),
        "prevalence": prevalence,
        "average_precision": average_precision,
        "brier_score": brier,
        "baseline_brier_score": baseline_brier,
        "brier_skill": brier_skill,
        "roc_auc": roc_auc,
        "balanced_accuracy": balanced_accuracy,
        "events": event_metrics,
    }


def moving_block_bootstrap_improvement(
    labels: pd.Series,
    candidate_probabilities: pd.Series,
    baseline_probabilities: pd.Series,
    *,
    block_length: int = 7,
    samples: int = 1000,
    seed: int = 42,
) -> dict[str, Any]:
    """Bootstrap Brier/AP improvements using circular contiguous blocks."""
    y = _numeric_series(labels)
    candidate = validate_probabilities(
        _numeric_series(candidate_probabilities, y.index)
    )
    baseline = validate_probabilities(_numeric_series(baseline_probabilities, y.index))
    valid = y.notna() & candidate.notna() & baseline.notna()
    y_values = y.loc[valid].astype(int).to_numpy()
    candidate_values = candidate.loc[valid].to_numpy(dtype=float)
    baseline_values = baseline.loc[valid].to_numpy(dtype=float)
    n = len(y_values)
    if n < 2:
        raise ValueError("Bootstrap requires at least two matched observations")
    block = max(1, min(int(block_length), n))
    block_count = math.ceil(n / block)
    offsets = np.arange(block, dtype=int)
    rng = np.random.default_rng(int(seed))
    brier_deltas: list[float] = []
    ap_deltas: list[float] = []
    for _ in range(int(samples)):
        starts = rng.integers(0, n, size=block_count)
        indices = ((starts[:, None] + offsets[None, :]) % n).reshape(-1)[:n]
        sampled_y = y_values[indices]
        sampled_candidate = candidate_values[indices]
        sampled_baseline = baseline_values[indices]
        brier_deltas.append(
            float(
                brier_score_loss(sampled_y, sampled_baseline)
                - brier_score_loss(sampled_y, sampled_candidate)
            )
        )
        if np.unique(sampled_y).size == 2:
            ap_deltas.append(
                float(
                    average_precision_score(sampled_y, sampled_candidate)
                    - average_precision_score(sampled_y, sampled_baseline)
                )
            )
    brier_array = np.asarray(brier_deltas, dtype=float)
    ap_array = np.asarray(ap_deltas, dtype=float)
    return {
        "n": n,
        "block_length": block,
        "samples": len(brier_array),
        "brier_improvement": float(
            brier_score_loss(y_values, baseline_values)
            - brier_score_loss(y_values, candidate_values)
        ),
        "brier_probability_improvement": float(np.mean(brier_array > 0.0)),
        "brier_improvement_p05": float(np.quantile(brier_array, 0.05)),
        "brier_improvement_p95": float(np.quantile(brier_array, 0.95)),
        "average_precision_improvement": float(
            average_precision_score(y_values, candidate_values)
            - average_precision_score(y_values, baseline_values)
        ),
        "average_precision_probability_improvement": (
            float(np.mean(ap_array > 0.0)) if len(ap_array) else None
        ),
    }


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if value is pd.NA:
        return None
    return value


if len(CORE_TAIL_FEATURES) > MAX_CORE_FEATURES:
    raise RuntimeError("Core tail feature allowlist exceeds the fixed 64-column budget")
