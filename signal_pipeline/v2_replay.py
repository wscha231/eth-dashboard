"""Same-origin V2 scoring for existing separate-head shadow reviews.

R1 exit plumbing only.  The module consumes immutable review outputs produced by
`optimization.run_review`; it does not fit, calibrate, select or issue models.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from model_lab.contracts import TARGET_SPEC, target_frame
from .data import build_features, read_bars
from .protocol import HORIZONS
from .v2_evaluate import center_metrics, distribution_metrics, event_metrics, volatility_metrics

POLICY = "forecast_bundle_v2_same_origin_replay_r1"


def _hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _public_review(review: Mapping) -> dict:
    return {k: v for k, v in review.items() if k not in {"rows", "incumbent_rows", "baseline_rows"}}


def discover_current_review(root: Path, horizon: int, public: Mapping) -> Path:
    candidates = sorted((root / "shadow_models").glob(f"review_{horizon}_*.json"))
    matches = []
    for path in candidates:
        try:
            review = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if _public_review(review) == dict(public):
            matches.append(path)
    if len(matches) != 1:
        raise ValueError(f"expected exactly one immutable current review for h={horizon}, found {len(matches)}")
    return matches[0]


def _sort_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda row: pd.Timestamp(row["slot"]))


def _assert_matched(*groups: list[dict]) -> list[pd.Timestamp]:
    if not groups or any(not group for group in groups):
        raise ValueError("candidate, incumbent and baseline rows are required")
    ordered = [_sort_rows(group) for group in groups]
    slots = [[row["slot"] for row in group] for group in ordered]
    if any(items != slots[0] for items in slots[1:]):
        raise ValueError("V2 replay requires identical origins")
    truth = ("target_end", "return", "terminal", "up", "down")
    for i in range(len(slots[0])):
        base = tuple(ordered[0][i][key] for key in truth)
        if any(tuple(group[i][key] for key in truth) != base for group in ordered[1:]):
            raise ValueError("matched origins contain inconsistent truth")
    return pd.DatetimeIndex(pd.to_datetime(slots[0], utc=True))


def _score_rows(rows: list[dict]) -> dict:
    frame = pd.DataFrame(_sort_rows(rows))
    return {
        "center": center_metrics(frame["return"], frame["q50"]),
        "distribution": distribution_metrics(frame["return"], frame["q10"], frame["q50"], frame["q90"]),
        "event": event_metrics(
            frame["terminal"],
            frame[["p_down", "p_flat", "p_up"]].to_numpy(float),
            frame["up"], frame["hit_up"], frame["down"], frame["hit_down"],
        ),
    }


def _skill(candidate: float, baseline: float) -> float | None:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline <= 0:
        return None
    return float(1.0 - candidate / baseline)


def score_shadow_review(
    review: Mapping,
    features: pd.DataFrame,
    targets: pd.DataFrame,
    horizon: int,
) -> dict:
    """Score all R1 heads on the exact review origins.

    Volatility has no R1 challenger: the persistence forecast is scored and
    explicitly reported baseline-only.  R2 will add candidate QLIKE comparisons.
    """
    if horizon not in HORIZONS or int(review.get("horizon_hours", -1)) != horizon:
        raise ValueError("horizon mismatch")
    candidate = list(review.get("rows", []))
    incumbent = list(review.get("incumbent_rows", []))
    climatology = list(review.get("baseline_rows", []))
    slots = _assert_matched(candidate, incumbent, climatology)
    if not slots.isin(features.index).all() or not slots.isin(targets.index).all():
        raise ValueError("review origin absent from features/targets")
    realized = targets.loc[slots, "realized_variance_total"].to_numpy(float)
    persistence = horizon * np.square(features.loc[slots, "sigma"].to_numpy(float))
    if not np.isfinite(realized).all() or not np.isfinite(persistence).all() or np.any(realized <= 0) or np.any(persistence <= 0):
        raise ValueError("positive finite variance target/persistence required")

    c = _score_rows(candidate)
    i = _score_rows(incumbent)
    b = _score_rows(climatology)
    no_change = center_metrics(pd.DataFrame(_sort_rows(candidate))["return"], np.zeros(len(candidate)))
    volatility = volatility_metrics(realized, persistence)

    return {
        "horizon_hours": horizon,
        "origins": len(slots),
        "first_origin": slots.min().isoformat(),
        "last_origin": slots.max().isoformat(),
        "target_spec_id": TARGET_SPEC["target_spec_id"],
        "variance_target": TARGET_SPEC["variance"],
        "candidate": c,
        "incumbent": i,
        "climatology": b,
        "mandatory_baselines": {
            "center_no_change": no_change,
            "volatility_persistence": volatility,
        },
        "relative_skill": {
            "center_vs_no_change_mae": _skill(c["center"]["mae"], no_change["mae"]),
            "center_vs_incumbent_mae": _skill(c["center"]["mae"], i["center"]["mae"]),
            "distribution_vs_climatology_wis80": _skill(c["distribution"]["wis80"], b["distribution"]["wis80"]),
            "distribution_vs_incumbent_wis80": _skill(c["distribution"]["wis80"], i["distribution"]["wis80"]),
            "event_vs_climatology_combined_brier": _skill(c["event"]["combined_brier"], b["event"]["combined_brier"]),
            "event_vs_incumbent_combined_brier": _skill(c["event"]["combined_brier"], i["event"]["combined_brier"]),
            "volatility_candidate_vs_persistence_qlike": None,
        },
        "head_status": {
            "center": "retrospective_compatibility_only",
            "volatility": "baseline_only_until_R2",
            "distribution": "retrospective_compatibility_only",
            "event": "retrospective_compatibility_only",
        },
        "legacy_head_choices": dict(review.get("heads", {})),
        "promotion": "none",
    }


def markdown_report(result: Mapping) -> str:
    lines = [
        "# ForecastBundleV2 R1 same-origin replay",
        "",
        "**Status: retrospective compatibility evidence only; no production promotion.**",
        "",
        f"Policy: `{result['policy']}`",
        f"Target spec: `{result['target_spec_id']}`",
        "",
        "| Horizon | Origins | Center vs no-change | Distribution vs incumbent | Event vs incumbent | Volatility |",
        "|---:|---:|---:|---:|---:|:---|",
    ]
    for h in sorted(result["horizons"], key=int):
        row = result["horizons"][h]
        skills = row["relative_skill"]
        def pct(value):
            return "NA" if value is None else f"{100*value:+.2f}%"
        lines.append(
            f"| {h}h | {row['origins']} | {pct(skills['center_vs_no_change_mae'])} | "
            f"{pct(skills['distribution_vs_incumbent_wis80'])} | "
            f"{pct(skills['event_vs_incumbent_combined_brier'])} | baseline-only |"
        )
    lines.extend([
        "",
        "Volatility is intentionally not assigned a candidate skill in R1. The table only confirms that the frozen realized-variance target and persistence QLIKE can be replayed on the same origins. R2 must preregister and test actual variance challengers.",
        "",
        "No value in this report changes the live registry, shadow checkpoints, public site, issuance ledger or issue #56 promotion gates.",
    ])
    return "\n".join(lines) + "\n"


def run_v2_replay(root: str | Path, output: str | Path) -> dict:
    root, output = Path(root), Path(output)
    summary_path = root / "optimization.json"
    if not summary_path.is_file():
        raise ValueError("optimization.json unavailable; run frozen shadow review first")
    optimization = json.loads(summary_path.read_text())
    bars = read_bars(root)
    features = build_features(bars)
    horizons = {}
    inputs = {"optimization.json": _hash(summary_path)}
    for horizon in HORIZONS:
        public = optimization.get("horizons", {}).get(str(horizon))
        if public is None:
            raise ValueError(f"optimization summary missing h={horizon}")
        review_path = discover_current_review(root, horizon, public)
        review = json.loads(review_path.read_text())
        targets = target_frame(bars, features, horizon)
        horizons[str(horizon)] = score_shadow_review(review, features, targets, horizon)
        inputs[review_path.relative_to(root).as_posix()] = _hash(review_path)
    result = {
        "schema_version": 1,
        "policy": POLICY,
        "target_spec_id": TARGET_SPEC["target_spec_id"],
        "variance_target": TARGET_SPEC["variance"],
        "source_data_as_of": optimization.get("data_as_of"),
        "inputs": inputs,
        "horizons": horizons,
        "promotion": "disabled",
        "writes": "offline report only; no model/site/ledger mutation",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")
    output.with_suffix(".md").write_text(markdown_report(result))
    return result
