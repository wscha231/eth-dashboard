"""Full historical Center/Event layer for the frozen six-horizon V2 audit.

The first audit attempt correctly refused to apply 2024/2025/2026 block gates to
a ~90-day shadow-review window.  This module keeps the frozen gates unchanged and
instead uses the repository's immutable monthly-refit historical replay, which
covers 2018-2026 depending on horizon.  R2 variance/distribution evaluation is
reused unchanged from the preregistered audit module.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model_lab.contracts import target_frame
from research.model import full_horizon_v2_backtest as base
from signal_pipeline.data import build_features, read_bars

POLICY = {
    **base.POLICY,
    "center_event_evidence": "immutable monthly-refit historical replay points; 2024/2025/2026 frozen calendar blocks",
    "shadow_review_role": "supplemental recent compatibility evidence only; not used for multi-year Center/Event gate",
}


def _frames_from_replay(horizon_payload: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    points = horizon_payload.get("points") or []
    if not points:
        raise ValueError("historical replay points are required")
    candidate_rows: list[dict] = []
    baseline_rows: list[dict] = []
    for point in points:
        baseline = point.get("baseline")
        if not isinstance(baseline, dict):
            raise ValueError("historical replay point is missing its climatology baseline")
        truth = {
            "slot": point["slot"],
            "target_end": point["target_end"],
            "return": point["return"],
            "terminal": point["terminal"],
            "up": point["up"],
            "down": point["down"],
        }
        candidate_rows.append({
            **truth,
            "q50": point["q50"],
            "p_down": point["p_down"],
            "p_flat": point["p_flat"],
            "p_up": point["p_up"],
            "hit_up": point["hit_up"],
            "hit_down": point["hit_down"],
        })
        baseline_rows.append({
            **truth,
            "q50": baseline["q50"],
            "p_down": baseline["p_down"],
            "p_flat": baseline["p_flat"],
            "p_up": baseline["p_up"],
            "hit_up": baseline["hit_up"],
            "hit_down": baseline["hit_down"],
        })
    candidate = base._sorted_frame(candidate_rows)
    climatology = base._sorted_frame(baseline_rows)
    # For this historical production-replay audit the explicit incumbent comparator
    # is the frozen climatology stream embedded at each origin.  No new Center/Event
    # challenger was introduced by R3 data work, so passing still requires the
    # selected historical stream to beat both no-change and climatology.
    incumbent = climatology.copy()
    base._assert_matched(candidate, incumbent, climatology)
    return candidate, incumbent, climatology


def evaluate(root: str | Path, output: str | Path) -> dict:
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("full-horizon audit output is immutable; choose a new output path")
    output.mkdir(parents=True)
    (output / "policy.json").write_bytes(base.canonical(POLICY))

    replay_path = root / "replay.json"
    if not replay_path.is_file():
        raise ValueError("immutable historical replay.json is required")
    replay = json.loads(replay_path.read_text())
    if set(replay.get("horizons", {})) != {str(h) for h in base.ALL_HORIZONS}:
        raise ValueError("historical replay must contain exactly the six operational horizons")

    # Re-run the original frozen R2 short-horizon gate and the preregistered long
    # extension against the same immutable hourly input.
    short = base.run_r2(root, output / "r2_short", long=False)
    long = base.run_r2(root, output / "r2_long", long=True)

    bars = read_bars(root)
    features = build_features(bars)
    result = {
        "schema_version": 2,
        "policy": POLICY,
        "data_as_of": replay.get("data_as_of"),
        "historical_replay_generated_at": replay.get("generated_at"),
        "protocol_hash": replay.get("protocol_hash"),
        "horizons": {},
        "promotion": "disabled_pending_prospective_evidence",
    }

    for horizon in base.ALL_HORIZONS:
        historical = replay["horizons"][str(horizon)]
        candidate, incumbent, climatology = _frames_from_replay(historical)
        targets = target_frame(bars, features, horizon)
        slots = pd.DatetimeIndex(candidate["slot"])
        if not slots.isin(targets.index).all():
            raise ValueError(f"historical replay origin missing from current target frame h={horizon}")

        r2_result = short if horizon in base.SHORT_HORIZONS else long
        r2_h = r2_result["horizons"][str(horizon)]
        result["horizons"][str(horizon)] = {
            "origins": int(len(candidate)),
            "first_origin": candidate["slot"].min().isoformat(),
            "last_origin": candidate["slot"].max().isoformat(),
            "center": base.center_gate(candidate, incumbent, climatology, horizon),
            "volatility": r2_h["variance"]["gate"],
            "distribution": r2_h["distribution"]["gate"],
            "event": base.event_gate(candidate, incumbent, climatology, horizon),
            "r2_variance_metrics": r2_h["variance"],
            "r2_distribution_metrics": r2_h["distribution"],
            "historical_replay_paired_event_brier": historical.get("paired_event_brier"),
            "historical_replay_yearly_selected": historical.get("yearly_selected"),
        }

    result["site_recommendation"] = base.site_recommendation(result["horizons"])
    (output / "result.json").write_bytes(base.canonical(result))
    (output / "REPORT.md").write_text(base.render_report(result), encoding="utf-8")
    return result
