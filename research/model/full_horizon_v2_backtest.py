"""Frozen six-horizon EtherForecast V2 retrospective audit.

This evaluator does not promote production models. It replays all six operational
horizons on one immutable event-research-state, applies the frozen #56 head gates,
and emits a site-surface recommendation that can only hide weak outputs or label
retrospective/shadow evidence. Prospective promotion remains a separate gate.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from model_lab.contracts import target_frame
from research.model import r2_range_volatility_wave1 as r2
from signal_pipeline.data import build_features, read_bars
from signal_pipeline.protocol import HORIZONS
from signal_pipeline.v2_evaluate import center_metrics, event_metrics
from signal_pipeline.v2_replay import discover_current_review, run_v2_replay

ALL_HORIZONS = tuple(HORIZONS)
SHORT_HORIZONS = (6, 24, 72)
LONG_HORIZONS = (168, 336, 720)
CALENDAR_BLOCKS = (2024, 2025, 2026)

POLICY = {
    "experiment_id": "full_horizon_v2_backtest_20260917",
    "frozen_before_results": True,
    "horizons_hours": list(ALL_HORIZONS),
    "calendar_blocks": list(CALENDAR_BLOCKS),
    "center_gate": {
        "mae_skill_vs_no_change_min": 0.02,
        "mae_improvement_vs_incumbent_min": 0.01,
        "rmse_max_multiple_of_better_baseline": 1.02,
        "minimum_improving_calendar_blocks": 2,
        "paired_improvement_probability_min": 0.90,
    },
    "event_gate": {
        "brier_skill_vs_prior_min": 0.02,
        "logloss_nonworse": True,
        "ece_max": 0.03,
        "paired_improvement_probability_min": 0.90,
    },
    "variance_distribution_short_policy": "reuse frozen R2 Wave1 unchanged",
    "variance_distribution_long_policy": {
        "horizons_hours": list(LONG_HORIZONS),
        "models": ["persistence_720", "ewma_168", "har_rv"],
        "variance_primary": "har_rv",
        "har_features": ["eth_vol_24", "eth_vol_168", "eth_vol_720"],
        "har_model": "StandardScaler + Ridge(alpha=10) on log variance; prior calibration mean-ratio correction",
        "distribution_primary": "har_rv_conformal80_zero_center",
        "distribution_simple": "persistence_conformal80_zero_center",
        "distribution_incumbent": "legacy selected quantiles from train_bundle at the same fold cutoff",
        "train_days": 730,
        "calibration_days": 180,
        "test_stride_hours": 24,
        "folds": [2024, 2025, 2026],
        "selection_rule": "fixed before long-horizon results; no post-hoc model selection or threshold tuning",
    },
    "prospective_promotion": "disabled; retrospective pass authorizes at most shadow/prospective evaluation",
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _skill(candidate: float, baseline: float) -> float:
    if not np.isfinite(candidate) or not np.isfinite(baseline) or baseline <= 0:
        raise ValueError("skill requires finite positive baseline")
    return float(1.0 - candidate / baseline)


def _simple_return(log_return: Iterable[float]) -> np.ndarray:
    return np.expm1(np.asarray(list(log_return), dtype=float))


def _sorted_frame(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        raise ValueError("review rows are required")
    frame = pd.DataFrame(sorted(rows, key=lambda row: pd.Timestamp(row["slot"])))
    frame["slot"] = pd.to_datetime(frame["slot"], utc=True)
    return frame


def _assert_matched(candidate: pd.DataFrame, incumbent: pd.DataFrame, baseline: pd.DataFrame) -> None:
    if not candidate["slot"].equals(incumbent["slot"]) or not candidate["slot"].equals(baseline["slot"]):
        raise ValueError("candidate/incumbent/baseline origins are not identical")
    truth = ["target_end", "return", "terminal", "up", "down"]
    for col in truth:
        if candidate[col].astype(str).tolist() != incumbent[col].astype(str).tolist():
            raise ValueError(f"candidate/incumbent truth mismatch: {col}")
        if candidate[col].astype(str).tolist() != baseline[col].astype(str).tolist():
            raise ValueError(f"candidate/baseline truth mismatch: {col}")


def paired_probability(candidate_loss: np.ndarray, baseline_loss: np.ndarray, slots: pd.DatetimeIndex, horizon: int, samples: int = 2000) -> dict:
    candidate_loss = np.asarray(candidate_loss, float)
    baseline_loss = np.asarray(baseline_loss, float)
    if len(candidate_loss) != len(baseline_loss) or len(candidate_loss) != len(slots) or len(slots) < 60:
        raise ValueError("paired bootstrap requires >=60 matched origins")
    delta = candidate_loss - baseline_loss
    block_days = max(7, int(np.ceil((horizon + 1) / 24)))
    groups = (slots.astype("int64") // (86400 * 10**9 * block_days)).to_numpy()
    blocks = [delta[groups == g] for g in np.unique(groups)]
    if len(blocks) < 10:
        raise ValueError("paired bootstrap has too few calendar blocks")
    rng = np.random.default_rng(20260917)
    estimates = np.empty(samples, dtype=float)
    for i in range(samples):
        choice = rng.integers(len(blocks), size=len(blocks))
        estimates[i] = np.concatenate([blocks[j] for j in choice]).mean()
    return {
        "probability_candidate_better": float(np.mean(estimates < 0)),
        "mean_loss_difference": float(delta.mean()),
        "lower95": float(np.quantile(estimates, 0.025)),
        "upper95": float(np.quantile(estimates, 0.975)),
        "block_days": block_days,
        "calendar_blocks": len(blocks),
    }


def center_gate(candidate: pd.DataFrame, incumbent: pd.DataFrame, baseline: pd.DataFrame, horizon: int) -> dict:
    actual = candidate["return"].to_numpy(float)
    c = center_metrics(actual, candidate["q50"].to_numpy(float))
    i = center_metrics(actual, incumbent["q50"].to_numpy(float))
    n = center_metrics(actual, np.zeros(len(candidate)))
    mae_skill = _skill(c["mae"], n["mae"])
    incumbent_skill = _skill(c["mae"], i["mae"])
    better_rmse = min(n["rmse"], i["rmse"])
    rmse_multiple = c["rmse"] / better_rmse

    years = candidate["slot"].dt.year.to_numpy()
    block_rows = []
    improving = 0
    for year in CALENDAR_BLOCKS:
        mask = years == year
        if not mask.any():
            raise ValueError(f"missing center calendar block {year} for h={horizon}")
        cm = center_metrics(actual[mask], candidate.loc[mask, "q50"].to_numpy(float))
        im = center_metrics(actual[mask], incumbent.loc[mask, "q50"].to_numpy(float))
        nm = center_metrics(actual[mask], np.zeros(int(mask.sum())))
        better = min(im["mae"], nm["mae"])
        did_improve = cm["mae"] < better
        improving += int(did_improve)
        block_rows.append({"year": year, "rows": int(mask.sum()), "candidate_mae": cm["mae"], "better_baseline_mae": better, "improved": did_improve})

    actual_simple = _simple_return(actual)
    candidate_simple = _simple_return(candidate["q50"])
    incumbent_simple = _simple_return(incumbent["q50"])
    no_change_simple = np.zeros(len(actual_simple))
    candidate_loss = np.abs(candidate_simple - actual_simple)
    incumbent_loss = np.abs(incumbent_simple - actual_simple)
    no_change_loss = np.abs(no_change_simple - actual_simple)
    better_loss = np.minimum(incumbent_loss, no_change_loss)
    paired = paired_probability(candidate_loss, better_loss, pd.DatetimeIndex(candidate["slot"]), horizon)

    gate = POLICY["center_gate"]
    passed = (
        mae_skill >= gate["mae_skill_vs_no_change_min"]
        and incumbent_skill >= gate["mae_improvement_vs_incumbent_min"]
        and rmse_multiple <= gate["rmse_max_multiple_of_better_baseline"]
        and improving >= gate["minimum_improving_calendar_blocks"]
        and paired["probability_candidate_better"] >= gate["paired_improvement_probability_min"]
    )
    return {
        "passed": bool(passed),
        "candidate": c,
        "incumbent": i,
        "no_change": n,
        "mae_skill_vs_no_change": mae_skill,
        "mae_improvement_vs_incumbent": incumbent_skill,
        "rmse_multiple_of_better_baseline": float(rmse_multiple),
        "improving_calendar_blocks": improving,
        "blocks": block_rows,
        "paired_vs_rowwise_better_baseline": paired,
    }


def _event_row_loss(frame: pd.DataFrame) -> np.ndarray:
    terminal = frame["terminal"].to_numpy(int)
    probs = frame[["p_down", "p_flat", "p_up"]].to_numpy(float)
    truth = np.eye(3)[terminal]
    terminal_brier = np.sum((probs - truth) ** 2, axis=1) / 2.0
    up = frame["up"].to_numpy(float)
    down = frame["down"].to_numpy(float)
    path_brier = ((frame["hit_up"].to_numpy(float) - up) ** 2 + (frame["hit_down"].to_numpy(float) - down) ** 2) / 2.0
    return (terminal_brier + path_brier) / 2.0


def _event_metrics(frame: pd.DataFrame) -> dict:
    return event_metrics(
        frame["terminal"],
        frame[["p_down", "p_flat", "p_up"]].to_numpy(float),
        frame["up"], frame["hit_up"], frame["down"], frame["hit_down"],
    )


def event_gate(candidate: pd.DataFrame, incumbent: pd.DataFrame, baseline: pd.DataFrame, horizon: int) -> dict:
    c = _event_metrics(candidate)
    i = _event_metrics(incumbent)
    b = _event_metrics(baseline)
    brier_skill = _skill(c["combined_brier"], b["combined_brier"])
    incumbent_skill = _skill(c["combined_brier"], i["combined_brier"])
    paired = paired_probability(_event_row_loss(candidate), _event_row_loss(baseline), pd.DatetimeIndex(candidate["slot"]), horizon)
    gate = POLICY["event_gate"]
    passed = (
        brier_skill >= gate["brier_skill_vs_prior_min"]
        and c["terminal_logloss"] <= b["terminal_logloss"]
        and c["ece"] <= gate["ece_max"]
        and paired["probability_candidate_better"] >= gate["paired_improvement_probability_min"]
    )
    return {
        "passed": bool(passed),
        "candidate": c,
        "incumbent": i,
        "prior_frequency": b,
        "brier_skill_vs_prior": brier_skill,
        "brier_skill_vs_incumbent": incumbent_skill,
        "logloss_nonworse_vs_prior": bool(c["terminal_logloss"] <= b["terminal_logloss"]),
        "ece": c["ece"],
        "paired_vs_prior": paired,
    }


def _compatible_indices(features, targets, start, end, *, require_variance=True):
    start, end = r2.utc(start), r2.utc(end)
    ready = features[["eth_vol_24", "eth_vol_168", "eth_vol_720"]].notna().all(axis=1)
    ready &= targets["return"].notna()
    if require_variance:
        ready &= targets["realized_variance_total"].gt(0)
    ready &= features.index.hour == 0
    ready &= features.index >= start
    ready &= targets["target_end"] < end - pd.Timedelta(hours=r2.POLICY["embargo_hours"])
    return features.index[ready]


def _long_policy() -> dict:
    policy = deepcopy(r2.POLICY)
    policy["experiment_id"] = "r2_range_volatility_wave2_long_20260917"
    policy["roadmap_stage"] = "R2_wave2_long_horizons"
    policy["horizons_hours"] = list(LONG_HORIZONS)
    policy["selection_rule"] = "same frozen HAR-RV/conformal specification extended to 7d/14d/30d before results; no post-hoc selection"
    policy["promotion"] = "disabled; retrospective long-horizon pass authorizes only shadow/prospective work"
    return policy


def run_r2(root: Path, output: Path, *, long: bool) -> dict:
    original_policy = r2.POLICY
    original_indices = r2._indices
    try:
        if long:
            r2.POLICY = _long_policy()
        r2._indices = _compatible_indices
        result = r2.evaluate(root, output)
    finally:
        r2.POLICY = original_policy
        r2._indices = original_indices
    if long:
        result["r2_wave2_pass"] = result.pop("r2_wave1_pass")
        (output / "result.json").write_bytes(canonical(result))
        report = r2.render_report(result).replace("R2 Wave 1", "R2 Wave 2 — Long Horizons")
        (output / "REPORT.md").write_text(report, encoding="utf-8")
    return result


def site_recommendation(heads: dict) -> dict:
    center_pass = any(v["center"]["passed"] for v in heads.values())
    event_pass = any(v["event"]["passed"] for v in heads.values())
    distribution_pass = any(v["distribution"]["passed"] for v in heads.values())
    variance_pass = any(v["volatility"]["passed"] for v in heads.values())
    return {
        "remove_legacy_exact_target_price_from_default_surface": not center_pass,
        "remove_unvalidated_direction_probabilities_from_default_surface": not event_pass,
        "show_reference_price_as_center_when_center_fails": True,
        "show_predictive_interval_only_where_distribution_gate_passes": True,
        "show_volatility_shadow_where_variance_gate_passes": variance_pass,
        "label_retrospective_pass_as_shadow_not_production": True,
        "preserve_old_forecasts_in_history_audit": True,
        "any_center_pass": center_pass,
        "any_event_pass": event_pass,
        "any_distribution_pass": distribution_pass,
        "any_variance_pass": variance_pass,
    }


def render_report(result: dict) -> str:
    lines = [
        "# EtherForecast V2 full-horizon backtest audit",
        "",
        f"Data as-of: `{result['data_as_of']}`",
        "",
        "Retrospective PIT/reconstructed evidence only. No result here bypasses prospective promotion gates.",
        "",
        "| Horizon | Center MAE skill vs no-change | Center | HAR-RV QLIKE skill | Volatility | Distribution | Event Brier skill vs prior | Event |",
        "|---:|---:|:---:|---:|:---:|:---:|---:|:---:|",
    ]
    for h in ALL_HORIZONS:
        row = result["horizons"][str(h)]
        c = row["center"]
        v = row["volatility"]
        d = row["distribution"]
        e = row["event"]
        lines.append(
            f"| {h}h | {c['mae_skill_vs_no_change']:+.2%} | {'PASS' if c['passed'] else 'FAIL'} | "
            f"{v['qlike_improvement_vs_persistence']:+.2%} | {'PASS' if v['passed'] else 'FAIL'} | "
            f"{'PASS' if d['passed'] else 'FAIL'} | {e['brier_skill_vs_prior']:+.2%} | {'PASS' if e['passed'] else 'FAIL'} |"
        )
    rec = result["site_recommendation"]
    lines.extend([
        "",
        "## Site decision",
        "",
        f"- Legacy exact target-price cards: **{'remove from default surface' if rec['remove_legacy_exact_target_price_from_default_surface'] else 'eligible only for passed horizons'}**.",
        f"- Unvalidated direction/event probabilities: **{'remove from default surface' if rec['remove_unvalidated_direction_probabilities_from_default_surface'] else 'eligible only for passed horizons'}**.",
        "- Failed outputs stay in immutable historical/audit records; they are not deleted from evidence.",
        "- A retrospective variance pass is shown only as research/shadow until prospective minimums mature.",
        "",
    ])
    return "\n".join(lines)


def evaluate(root: str | Path, output: str | Path) -> dict:
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("full-horizon audit output is immutable; choose a new output path")
    output.mkdir(parents=True)
    (output / "policy.json").write_bytes(canonical(POLICY))

    v2_path = output / "v2_replay.json"
    v2 = run_v2_replay(root, v2_path)
    short = run_r2(root, output / "r2_short", long=False)
    long = run_r2(root, output / "r2_long", long=True)

    optimization = json.loads((root / "optimization.json").read_text())
    bars = read_bars(root)
    features = build_features(bars)
    result = {
        "schema_version": 1,
        "policy": POLICY,
        "data_as_of": v2.get("source_data_as_of"),
        "horizons": {},
        "promotion": "disabled_pending_prospective_evidence",
    }

    for horizon in ALL_HORIZONS:
        public = optimization["horizons"][str(horizon)]
        review_path = discover_current_review(root, horizon, public)
        review = json.loads(review_path.read_text())
        candidate = _sorted_frame(review["rows"])
        incumbent = _sorted_frame(review["incumbent_rows"])
        baseline = _sorted_frame(review["baseline_rows"])
        _assert_matched(candidate, incumbent, baseline)
        # Confirm the review origins remain valid under the current frozen target spec.
        targets = target_frame(bars, features, horizon)
        if not pd.DatetimeIndex(candidate["slot"]).isin(targets.index).all():
            raise ValueError(f"review origin missing from current target frame h={horizon}")

        r2_result = short if horizon in SHORT_HORIZONS else long
        r2_h = r2_result["horizons"][str(horizon)]
        result["horizons"][str(horizon)] = {
            "origins": int(len(candidate)),
            "first_origin": candidate["slot"].min().isoformat(),
            "last_origin": candidate["slot"].max().isoformat(),
            "center": center_gate(candidate, incumbent, baseline, horizon),
            "volatility": r2_h["variance"]["gate"],
            "distribution": r2_h["distribution"]["gate"],
            "event": event_gate(candidate, incumbent, baseline, horizon),
            "r2_variance_metrics": r2_h["variance"],
            "r2_distribution_metrics": r2_h["distribution"],
        }

    result["site_recommendation"] = site_recommendation(result["horizons"])
    (output / "result.json").write_bytes(canonical(result))
    (output / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    return result
