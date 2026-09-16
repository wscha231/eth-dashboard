"""ForecastBundleV2 contract for independent forecast heads.

Offline/research-only in R1.  Nothing in the incumbent inference path imports
this module yet.  The purpose is to freeze semantics before model experiments:
Center, Volatility, Distribution and Event heads have different objectives and
must be selected independently.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import math
from typing import Any, Mapping

from .protocol import HORIZONS

SCHEMA_VERSION = 2
CONTRACT_VERSION = "forecast_bundle_v2_contract_1"
HEAD_OBJECTIVES = {
    "center": "MAE/median_pinball",
    "volatility": "QLIKE",
    "distribution": "WIS/pinball/coverage_width",
    "event": "Brier/logloss/ECE",
}
REQUIRED_HEADS = tuple(HEAD_OBJECTIVES)


def _finite(value: float, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _utc(value: str, name: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{name} must be ISO-8601") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{name} must be timezone-aware")
    return parsed


@dataclass(frozen=True)
class HeadDecision:
    head: str
    objective: str
    baseline_name: str
    candidate_name: str
    baseline_loss: float
    candidate_loss: float
    required_relative_improvement: float
    selected_name: str
    relative_improvement: float
    passed_gate: bool
    reason: str

    def __post_init__(self) -> None:
        if self.head not in HEAD_OBJECTIVES:
            raise ValueError(f"unknown head: {self.head}")
        if self.objective != HEAD_OBJECTIVES[self.head]:
            raise ValueError("head objective mismatch")
        _finite(self.baseline_loss, "baseline_loss")
        _finite(self.candidate_loss, "candidate_loss")
        if self.baseline_loss <= 0:
            raise ValueError("baseline_loss must be positive")
        if self.required_relative_improvement < 0:
            raise ValueError("required improvement must be non-negative")
        if self.selected_name not in {self.baseline_name, self.candidate_name}:
            raise ValueError("selected_name must be baseline or candidate")


def select_head(
    *,
    head: str,
    baseline_name: str,
    candidate_name: str,
    baseline_loss: float,
    candidate_loss: float,
    required_relative_improvement: float,
) -> HeadDecision:
    """Select one head only by that head's frozen objective.

    Lower loss is better.  A candidate that does not meet the required
    relative improvement falls back to the mandatory baseline.
    """
    if head not in HEAD_OBJECTIVES:
        raise ValueError(f"unknown head: {head}")
    baseline_loss = _finite(baseline_loss, "baseline_loss")
    candidate_loss = _finite(candidate_loss, "candidate_loss")
    if baseline_loss <= 0:
        raise ValueError("baseline_loss must be positive")
    required = float(required_relative_improvement)
    if not 0 <= required < 1:
        raise ValueError("required improvement must be in [0,1)")
    improvement = 1.0 - candidate_loss / baseline_loss
    passed = improvement >= required
    selected = candidate_name if passed else baseline_name
    reason = "candidate_passed_head_gate" if passed else "baseline_fallback"
    return HeadDecision(
        head=head,
        objective=HEAD_OBJECTIVES[head],
        baseline_name=str(baseline_name),
        candidate_name=str(candidate_name),
        baseline_loss=baseline_loss,
        candidate_loss=candidate_loss,
        required_relative_improvement=required,
        selected_name=selected,
        relative_improvement=improvement,
        passed_gate=passed,
        reason=reason,
    )


@dataclass(frozen=True)
class CenterOutput:
    log_return_median: float
    decision: HeadDecision

    def __post_init__(self) -> None:
        _finite(self.log_return_median, "log_return_median")
        if self.decision.head != "center":
            raise ValueError("center output requires center decision")


@dataclass(frozen=True)
class VolatilityOutput:
    realized_variance: float
    decision: HeadDecision

    def __post_init__(self) -> None:
        value = _finite(self.realized_variance, "realized_variance")
        if value <= 0:
            raise ValueError("realized_variance must be positive")
        if self.decision.head != "volatility":
            raise ValueError("volatility output requires volatility decision")


@dataclass(frozen=True)
class DistributionOutput:
    quantiles: Mapping[float, float]
    decision: HeadDecision

    def __post_init__(self) -> None:
        if self.decision.head != "distribution":
            raise ValueError("distribution output requires distribution decision")
        if len(self.quantiles) < 3:
            raise ValueError("distribution requires at least three quantiles")
        levels = [float(k) for k in self.quantiles]
        if levels != sorted(levels) or len(levels) != len(set(levels)):
            raise ValueError("quantile levels must be unique and increasing")
        if levels[0] <= 0 or levels[-1] >= 1:
            raise ValueError("quantile levels must be strictly between 0 and 1")
        values = [_finite(self.quantiles[k], f"quantile_{k}") for k in self.quantiles]
        if values != sorted(values):
            raise ValueError("quantile values must be non-decreasing")


@dataclass(frozen=True)
class EventOutput:
    terminal_down_flat_up: tuple[float, float, float]
    hit_up: float
    hit_down: float
    decision: HeadDecision

    def __post_init__(self) -> None:
        if self.decision.head != "event":
            raise ValueError("event output requires event decision")
        probs = tuple(_finite(v, "event_probability") for v in self.terminal_down_flat_up)
        if any(v < 0 or v > 1 for v in probs):
            raise ValueError("terminal probabilities must be in [0,1]")
        if not math.isclose(sum(probs), 1.0, rel_tol=0, abs_tol=1e-8):
            raise ValueError("terminal probabilities must sum to one")
        for name, value in (("hit_up", self.hit_up), ("hit_down", self.hit_down)):
            probability = _finite(value, name)
            if not 0 <= probability <= 1:
                raise ValueError(f"{name} must be in [0,1]")


@dataclass(frozen=True)
class ForecastBundleV2:
    horizon_hours: int
    issue_time: str
    input_cutoff: str
    available_at: str
    target_end: str
    reference_price: float
    input_snapshot: str
    feature_version: str
    center: CenterOutput
    volatility: VolatilityOutput
    distribution: DistributionOutput
    event: EventOutput
    schema_version: int = SCHEMA_VERSION
    contract_version: str = CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError("schema_version mismatch")
        if self.contract_version != CONTRACT_VERSION:
            raise ValueError("contract_version mismatch")
        if int(self.horizon_hours) not in HORIZONS:
            raise ValueError("unsupported horizon")
        reference = _finite(self.reference_price, "reference_price")
        if reference <= 0:
            raise ValueError("reference_price must be positive")
        issue = _utc(self.issue_time, "issue_time")
        cutoff = _utc(self.input_cutoff, "input_cutoff")
        available = _utc(self.available_at, "available_at")
        target_end = _utc(self.target_end, "target_end")
        if cutoff > issue:
            raise ValueError("input_cutoff cannot be after issue_time")
        if available > issue:
            raise ValueError("source was unavailable at issue_time")
        expected_seconds = (int(self.horizon_hours) + 1) * 3600
        if abs((target_end - cutoff).total_seconds() - expected_seconds) > 1e-6:
            raise ValueError("target_end must preserve frozen h+1 endpoint semantics")
        if not str(self.input_snapshot).strip():
            raise ValueError("input_snapshot required")
        if not str(self.feature_version).strip():
            raise ValueError("feature_version required")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        # JSON keys cannot retain float key semantics reliably across runtimes.
        payload["distribution"]["quantiles"] = {
            f"{float(k):.6f}": float(v) for k, v in self.distribution.quantiles.items()
        }
        return payload

    def canonical_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)


def validate_payload(payload: Mapping[str, Any]) -> None:
    """Validate an already-serialized V2 bundle without constructing models."""
    if int(payload.get("schema_version", -1)) != SCHEMA_VERSION:
        raise ValueError("schema_version mismatch")
    if payload.get("contract_version") != CONTRACT_VERSION:
        raise ValueError("contract_version mismatch")
    heads = payload.get("heads")
    if heads is not None and tuple(heads) != REQUIRED_HEADS:
        raise ValueError("head order/identity mismatch")
