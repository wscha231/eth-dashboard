"""Compatibility adapter from existing separate-head shadow research to V2.

R1 only.  This module does not change the legacy candidate fitter or live
issuance.  It makes the existing point/interval/probability separation explicit
under ForecastBundleV2 and leaves Volatility baseline-only until R2 supplies a
real variance challenger.
"""
from __future__ import annotations

from typing import Mapping

import numpy as np
import pandas as pd

from .v2_contract import (
    CenterOutput,
    DistributionOutput,
    EventOutput,
    ForecastBundleV2,
    HeadDecision,
    VolatilityOutput,
    select_head,
)

CENTER_REQUIRED_IMPROVEMENT = 0.02
VOLATILITY_REQUIRED_IMPROVEMENT = 0.05
DISTRIBUTION_REQUIRED_IMPROVEMENT = 0.02
EVENT_REQUIRED_IMPROVEMENT = 0.02


def _legacy_climatology_loss(bundle: Mapping, metric: str) -> float:
    scores = bundle.get("selection_scores", {})
    values = [
        float(row[metric])
        for key, row in scores.items()
        if str(key).endswith(":climatology") and metric in row
    ]
    if not values:
        raise ValueError(f"legacy climatology {metric} score unavailable")
    return min(values)


def _legacy_choice_loss(bundle: Mapping, choice_key: str, metric: str) -> tuple[str, float]:
    choice = str(bundle.get("choices", {}).get(choice_key, ""))
    if not choice:
        raise ValueError(f"legacy {choice_key} choice unavailable")
    scores = bundle.get("selection_scores", {})
    if choice not in scores or metric not in scores[choice]:
        raise ValueError(f"legacy score unavailable for {choice_key} choice")
    return choice, float(scores[choice][metric])


def legacy_head_decisions(bundle: Mapping) -> dict[str, HeadDecision]:
    """Translate existing selection evidence without inventing a volatility win.

    These decisions expose compatibility semantics only.  They are not a #56
    promotion certificate because the old selection blocks and baselines are
    not the complete frozen V2 gate.
    """
    no_change = float(bundle.get("no_change_selection_mae", np.nan))
    point_choice = str(bundle.get("choices", {}).get("point", ""))
    if not np.isfinite(no_change) or no_change <= 0:
        raise ValueError("legacy no-change selection MAE unavailable")
    if point_choice == "no_change":
        center_candidate_name = "legacy_center_candidate_unavailable"
        center_candidate_loss = no_change
    else:
        center_candidate_name, center_candidate_loss = _legacy_choice_loss(bundle, "point", "point")
    center = select_head(
        head="center",
        baseline_name="no_change",
        candidate_name=center_candidate_name,
        baseline_loss=no_change,
        candidate_loss=center_candidate_loss,
        required_relative_improvement=CENTER_REQUIRED_IMPROVEMENT,
    )

    interval_name, interval_loss = _legacy_choice_loss(bundle, "interval", "interval")
    distribution = select_head(
        head="distribution",
        baseline_name="legacy_climatology_interval",
        candidate_name=interval_name,
        baseline_loss=_legacy_climatology_loss(bundle, "interval"),
        candidate_loss=interval_loss,
        required_relative_improvement=DISTRIBUTION_REQUIRED_IMPROVEMENT,
    )

    event_name, event_loss = _legacy_choice_loss(bundle, "probability", "probability")
    event = select_head(
        head="event",
        baseline_name="legacy_climatology_probability",
        candidate_name=event_name,
        baseline_loss=_legacy_climatology_loss(bundle, "probability"),
        candidate_loss=event_loss,
        required_relative_improvement=EVENT_REQUIRED_IMPROVEMENT,
    )

    # R1 deliberately has no independently trained variance challenger.  A
    # neutral synthetic loss pair forces explicit fallback to persistence.
    volatility = select_head(
        head="volatility",
        baseline_name="variance_persistence",
        candidate_name="volatility_candidate_unavailable_until_R2",
        baseline_loss=1.0,
        candidate_loss=1.0,
        required_relative_improvement=VOLATILITY_REQUIRED_IMPROVEMENT,
    )
    return {
        "center": center,
        "volatility": volatility,
        "distribution": distribution,
        "event": event,
    }


def adapt_shadow_prediction(
    *,
    bundle: Mapping,
    prediction: Mapping[str, np.ndarray],
    feature_row: pd.Series,
    horizon_hours: int,
    issue_time,
    input_cutoff,
    input_snapshot: str,
    feature_version: str,
) -> ForecastBundleV2:
    """Build one V2 research record from a frozen legacy shadow prediction."""
    decisions = legacy_head_decisions(bundle)
    q = np.asarray(prediction["quantiles"], dtype=float)
    terminal = np.asarray(prediction["terminal"], dtype=float)
    path = np.asarray(prediction["path"], dtype=float)
    if q.shape != (1, 3) or terminal.shape != (1, 3) or path.shape != (1, 2):
        raise ValueError("single-origin legacy prediction required")

    cutoff = pd.Timestamp(input_cutoff)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    else:
        cutoff = cutoff.tz_convert("UTC")
    issue = pd.Timestamp(issue_time)
    if issue.tzinfo is None:
        issue = issue.tz_localize("UTC")
    else:
        issue = issue.tz_convert("UTC")
    available = pd.Timestamp(feature_row["available_at"])
    if available.tzinfo is None:
        available = available.tz_localize("UTC")
    else:
        available = available.tz_convert("UTC")

    reference = float(feature_row["reference_price"])
    sigma = float(feature_row["sigma"])
    if not np.isfinite(reference) or reference <= 0 or not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("positive finite reference_price and sigma required")
    # Persistence-scale placeholder only: trailing hourly sigma projected over h.
    variance_persistence = max(sigma * sigma * int(horizon_hours), 1e-12)

    return ForecastBundleV2(
        horizon_hours=int(horizon_hours),
        issue_time=issue.isoformat(),
        input_cutoff=cutoff.isoformat(),
        available_at=available.isoformat(),
        target_end=(cutoff + pd.Timedelta(hours=int(horizon_hours) + 1)).isoformat(),
        reference_price=reference,
        input_snapshot=str(input_snapshot),
        feature_version=str(feature_version),
        center=CenterOutput(log_return_median=float(q[0, 1]), decision=decisions["center"]),
        volatility=VolatilityOutput(
            realized_variance=variance_persistence,
            decision=decisions["volatility"],
        ),
        distribution=DistributionOutput(
            quantiles={0.1: float(q[0, 0]), 0.5: float(q[0, 1]), 0.9: float(q[0, 2])},
            decision=decisions["distribution"],
        ),
        event=EventOutput(
            terminal_down_flat_up=tuple(float(v) for v in terminal[0]),
            hit_up=float(path[0, 0]),
            hit_down=float(path[0, 1]),
            decision=decisions["event"],
        ),
    )
