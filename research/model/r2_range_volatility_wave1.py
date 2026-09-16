"""R2 Wave 1: preregistered 6h/24h/72h variance and range evaluation.

Research only. This module does not issue forecasts, modify active checkpoints, or promote
models. It reuses the frozen hourly endpoint/path target semantics and compares one primary
variance challenger (HAR-RV) plus one diagnostic EWMA model against persistence. The
Distribution head uses a zero/no-change center and past-only symmetric conformal calibration.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from model_lab.contracts import target_frame
from signal_pipeline.data import build_features
from signal_pipeline.models import labels, predict_models, train_bundle

POLICY = {
    "experiment_id": "r2_range_volatility_wave1_20260916",
    "roadmap_stage": "R2_wave1",
    "evidence_grade": "retrospective_reconstructed_development_not_prospective",
    "horizons_hours": [6, 24, 72],
    "folds": [
        {"id": "2024", "start": "2024-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        {"id": "2025", "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
        {"id": "2026", "start": "2026-01-01T00:00:00Z", "end": "2027-01-01T00:00:00Z"},
    ],
    "train_days": 730,
    "calibration_days": 180,
    "embargo_hours": 1,
    "test_stride_hours": 24,
    "variance_primary": "har_rv",
    "variance_diagnostic": "ewma72_calibrated",
    "variance_target": "sum of h squared close-to-close log returns inside the h-bar path window",
    "har_features": ["eth_vol_24", "eth_vol_168", "eth_vol_720"],
    "har_model": "StandardScaler + Ridge(alpha=10) on log variance; prior calibration mean-ratio correction",
    "distribution_primary": "har_rv_conformal80_zero_center",
    "distribution_simple": "persistence_conformal80_zero_center",
    "distribution_incumbent": "legacy selected quantiles from train_bundle at the same fold cutoff",
    "distribution_calibration": "past-only 80th percentile of abs(log endpoint return)/predicted endpoint sigma",
    "center": "fixed zero log-return/no-change; R2 does not claim Center-head alpha",
    "variance_gate": {
        "qlike_improvement_vs_persistence_min": 0.05,
        "minimum_improving_calendar_blocks": 2,
        "calendar_blocks": 3,
        "maximum_single_block_degradation": 0.10,
    },
    "distribution_gate": {
        "wis_improvement_vs_simple_min": 0.02,
        "wis_improvement_vs_incumbent_min": 0.02,
        "coverage80_min": 0.76,
        "coverage80_max": 0.84,
        "mean_width_max_multiple_of_better_baseline": 1.10,
        "paired_improvement_probability_min": 0.90,
    },
    "selection_rule": "no post-hoc model selection; HAR-RV is the fixed primary challenger",
    "promotion": "disabled; passing retrospective R2 authorizes shadow/prospective work only",
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def utc(value) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    actual = np.asarray(actual, float)
    predicted = np.asarray(predicted, float)
    if (not np.isfinite(np.r_[actual, predicted]).all() or (actual <= 0).any() or (predicted <= 0).any()):
        raise ValueError("QLIKE needs finite strictly positive realized/predicted variance")
    ratio = actual / predicted
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def pinball(y: np.ndarray, q: np.ndarray, alpha: float) -> float:
    error = np.asarray(y, float) - np.asarray(q, float)
    return float(np.mean(np.maximum(alpha * error, (alpha - 1.0) * error)))


def per_row_wis80(actual_log: np.ndarray, qlog: np.ndarray) -> np.ndarray:
    """One-central-interval WIS in simple-return space: median + central 80%."""
    actual = np.expm1(np.asarray(actual_log, float))
    q = np.expm1(np.asarray(qlog, float))
    lo, med, hi = q[:, 0], q[:, 1], q[:, 2]
    interval_score = (hi - lo) + 10.0 * np.maximum(lo - actual, 0.0) + 10.0 * np.maximum(actual - hi, 0.0)
    return (0.5 * np.abs(actual - med) + 0.1 * interval_score) / 1.5


def distribution_metrics(actual_log: np.ndarray, qlog: np.ndarray) -> dict:
    actual_log = np.asarray(actual_log, float)
    qlog = np.asarray(qlog, float)
    if qlog.ndim != 2 or qlog.shape[1] != 3 or (np.diff(qlog, axis=1) < 0).any():
        raise ValueError("distribution quantiles must be finite non-decreasing q10/q50/q90")
    if not np.isfinite(np.r_[actual_log, qlog.ravel()]).all():
        raise ValueError("distribution inputs must be finite")
    actual = np.expm1(actual_log)
    q = np.expm1(qlog)
    row_wis = per_row_wis80(actual_log, qlog)
    return {
        "rows": int(len(actual_log)),
        "wis80": float(row_wis.mean()),
        "pinball_q10": pinball(actual, q[:, 0], 0.10),
        "pinball_q50": pinball(actual, q[:, 1], 0.50),
        "pinball_q90": pinball(actual, q[:, 2], 0.90),
        "coverage80": float(((actual_log >= qlog[:, 0]) & (actual_log <= qlog[:, 2])).mean()),
        "mean_width": float(np.mean(q[:, 2] - q[:, 0])),
    }


def conformal_symmetric_quantiles(variance_total: np.ndarray, horizon: int, scale: float) -> np.ndarray:
    variance_total = np.asarray(variance_total, float)
    if (variance_total <= 0).any() or not np.isfinite(variance_total).all() or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid variance/scale for conformal interval")
    endpoint_sigma = np.sqrt(variance_total * (horizon + 1.0) / horizon)
    width = scale * endpoint_sigma
    return np.column_stack((-width, np.zeros(len(width)), width))


def calibrate_scale(actual_log: np.ndarray, variance_total: np.ndarray, horizon: int) -> float:
    endpoint_sigma = np.sqrt(np.asarray(variance_total, float) * (horizon + 1.0) / horizon)
    standardized = np.abs(np.asarray(actual_log, float)) / np.maximum(endpoint_sigma, 1e-12)
    standardized = standardized[np.isfinite(standardized)]
    if len(standardized) < 45:
        raise ValueError("insufficient conformal calibration rows")
    return float(np.quantile(standardized, 0.80, method="higher"))


def paired_probability(candidate_losses: np.ndarray, baseline_losses: np.ndarray, slots: pd.DatetimeIndex,
                       horizon: int, samples: int = 1000) -> dict:
    a = np.asarray(candidate_losses, float)
    b = np.asarray(baseline_losses, float)
    if len(a) != len(b) or len(a) != len(slots) or len(a) < 60:
        raise ValueError("paired comparison requires matched origins")
    delta = a - b
    block_days = max(7, int(np.ceil((horizon + 1) / 24)))
    groups = (slots.astype("int64") // (86400 * 10**9 * block_days)).to_numpy()
    unique = np.unique(groups)
    blocks = [delta[groups == g] for g in unique]
    if len(blocks) < 10:
        raise ValueError("too few calendar blocks for paired comparison")
    rng = np.random.default_rng(1729)
    estimates = []
    for _ in range(samples):
        choice = rng.integers(len(blocks), size=len(blocks))
        estimates.append(np.concatenate([blocks[i] for i in choice]).mean())
    estimates = np.asarray(estimates)
    return {
        "difference_candidate_minus_baseline": float(delta.mean()),
        "probability_candidate_better": float(np.mean(estimates < 0)),
        "lower95": float(np.quantile(estimates, 0.025)),
        "upper95": float(np.quantile(estimates, 0.975)),
        "calendar_blocks": int(len(blocks)),
        "block_days": int(block_days),
    }


def variance_gate(blocks: list[dict], aggregate_candidate: float, aggregate_persistence: float) -> dict:
    gate = POLICY["variance_gate"]
    improvement = 1.0 - aggregate_candidate / aggregate_persistence
    improving = sum(b["har_rv_qlike"] < b["persistence_qlike"] for b in blocks)
    degradations = [b["har_rv_qlike"] / b["persistence_qlike"] - 1.0 for b in blocks]
    passed = (
        improvement >= gate["qlike_improvement_vs_persistence_min"]
        and improving >= gate["minimum_improving_calendar_blocks"]
        and max(degradations) <= gate["maximum_single_block_degradation"]
    )
    return {
        "passed": bool(passed),
        "qlike_improvement_vs_persistence": float(improvement),
        "improving_blocks": int(improving),
        "maximum_block_degradation": float(max(degradations)),
    }


def distribution_gate(candidate: dict, simple: dict, incumbent: dict, paired_simple: dict, paired_incumbent: dict) -> dict:
    gate = POLICY["distribution_gate"]
    improvement_simple = 1.0 - candidate["wis80"] / simple["wis80"]
    improvement_incumbent = 1.0 - candidate["wis80"] / incumbent["wis80"]
    better = simple if simple["wis80"] <= incumbent["wis80"] else incumbent
    width_multiple = candidate["mean_width"] / better["mean_width"]
    passed = (
        improvement_simple >= gate["wis_improvement_vs_simple_min"]
        and improvement_incumbent >= gate["wis_improvement_vs_incumbent_min"]
        and gate["coverage80_min"] <= candidate["coverage80"] <= gate["coverage80_max"]
        and width_multiple <= gate["mean_width_max_multiple_of_better_baseline"]
        and paired_simple["probability_candidate_better"] >= gate["paired_improvement_probability_min"]
        and paired_incumbent["probability_candidate_better"] >= gate["paired_improvement_probability_min"]
    )
    return {
        "passed": bool(passed),
        "wis_improvement_vs_simple": float(improvement_simple),
        "wis_improvement_vs_incumbent": float(improvement_incumbent),
        "coverage80": float(candidate["coverage80"]),
        "width_multiple_of_better_baseline": float(width_multiple),
        "paired_probability_vs_simple": float(paired_simple["probability_candidate_better"]),
        "paired_probability_vs_incumbent": float(paired_incumbent["probability_candidate_better"]),
    }


def _indices(features: pd.DataFrame, targets: pd.DataFrame, start, end, *, require_variance=True) -> pd.DatetimeIndex:
    start, end = utc(start), utc(end)
    ready = features[["eth_vol_24", "eth_vol_168", "eth_vol_720"]].notna().all(axis=1)
    ready &= targets["return"].notna()
    if require_variance:
        ready &= targets["realized_variance_total"].gt(0)
    ready &= features.index.hour.eq(0)
    ready &= features.index >= start
    ready &= targets["target_end"] < end - pd.Timedelta(hours=POLICY["embargo_hours"])
    return features.index[ready]


def _ewma_variance_per_hour(bars: pd.DataFrame, index: pd.DatetimeIndex) -> pd.Series:
    eth = bars.loc[bars["product"].eq("ETH-USD")].set_index("close_time").reindex(index)
    sq = np.log(eth.close).diff().pow(2)
    return sq.ewm(halflife=72, adjust=False, min_periods=720).mean()


def evaluate(root: Path, output: Path) -> dict:
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("R2 output is immutable; choose a new output path")
    output.mkdir(parents=True)
    (output / "policy.json").write_bytes(canonical(POLICY))
    started = time.monotonic()

    source = root / "hourly.parquet"
    if not source.exists():
        raise FileNotFoundError("event research hourly.parquet is required")
    bars = pd.read_parquet(source)
    for col in ("open_time", "close_time", "observed_at"):
        bars[col] = pd.to_datetime(bars[col], utc=True)
    bars = bars.sort_values(["product", "close_time"]).reset_index(drop=True)
    features = build_features(bars)
    as_of = bars.groupby("product").close_time.max().min()
    ewma_per_hour = _ewma_variance_per_hour(bars, features.index)

    result = {
        "policy": POLICY,
        "data_as_of": as_of.isoformat(),
        "horizons": {},
        "promotion": "disabled",
        "claims": "retrospective reconstructed development evidence; no prospective superiority established",
    }

    for horizon in POLICY["horizons_hours"]:
        targets = target_frame(bars, features, horizon)
        legacy_outcomes = labels(bars, features, horizon)
        all_actual_var = []
        all_persistence_var = []
        all_har_var = []
        all_ewma_var = []
        all_returns = []
        all_har_q = []
        all_simple_q = []
        all_incumbent_q = []
        all_slots = []
        block_reports = []

        for fold in POLICY["folds"]:
            test_start = utc(fold["start"])
            test_end = min(utc(fold["end"]), as_of + pd.Timedelta(hours=1))
            cal_start = test_start - pd.Timedelta(days=POLICY["calibration_days"])
            train_start = cal_start - pd.Timedelta(days=POLICY["train_days"])

            train = _indices(features, targets, train_start, cal_start)
            cal = _indices(features, targets, cal_start, test_start)
            test = _indices(features, targets, test_start, test_end)
            if len(train) < 400 or len(cal) < 45 or len(test) < 60:
                raise ValueError(f"insufficient rows for {horizon}h/{fold['id']}: {len(train)}/{len(cal)}/{len(test)}")

            vcols = POLICY["har_features"]
            vx = np.log(features[vcols].pow(2).clip(lower=1e-16))
            har = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
                vx.loc[train], np.log(targets.loc[train, "realized_variance_total"].to_numpy() / horizon)
            )
            cal_raw_per_hour = np.exp(har.predict(vx.loc[cal]))
            cal_actual_per_hour = targets.loc[cal, "realized_variance_total"].to_numpy() / horizon
            multiplier = float(np.mean(cal_actual_per_hour / cal_raw_per_hour))
            har_cal = horizon * cal_raw_per_hour * multiplier
            har_test = horizon * np.exp(har.predict(vx.loc[test])) * multiplier

            persistence_cal = horizon * features.loc[cal, "eth_vol_720"].to_numpy() ** 2
            persistence_test = horizon * features.loc[test, "eth_vol_720"].to_numpy() ** 2

            ewma_cal_raw = horizon * ewma_per_hour.loc[cal].to_numpy()
            ewma_test_raw = horizon * ewma_per_hour.loc[test].to_numpy()
            if not np.isfinite(np.r_[ewma_cal_raw, ewma_test_raw]).all() or (ewma_cal_raw <= 0).any() or (ewma_test_raw <= 0).any():
                raise ValueError("EWMA variance unavailable on declared fold")
            ewma_multiplier = float(np.mean(targets.loc[cal, "realized_variance_total"].to_numpy() / ewma_cal_raw))
            ewma_test = ewma_test_raw * ewma_multiplier

            actual_var = targets.loc[test, "realized_variance_total"].to_numpy()
            returns = targets.loc[test, "return"].to_numpy()

            har_scale = calibrate_scale(targets.loc[cal, "return"].to_numpy(), har_cal, horizon)
            persistence_scale = calibrate_scale(targets.loc[cal, "return"].to_numpy(), persistence_cal, horizon)
            har_q = conformal_symmetric_quantiles(har_test, horizon, har_scale)
            simple_q = conformal_symmetric_quantiles(persistence_test, horizon, persistence_scale)

            incumbent_bundle = train_bundle(features, legacy_outcomes, test_start, horizon)
            incumbent_predictions = predict_models(incumbent_bundle["model"], features.loc[test])
            incumbent_q = incumbent_predictions[incumbent_bundle["choice"]]["quantiles"]

            block_reports.append({
                "fold": fold["id"],
                "train_rows": int(len(train)),
                "calibration_rows": int(len(cal)),
                "test_rows": int(len(test)),
                "har_multiplier": multiplier,
                "har_conformal_scale": har_scale,
                "persistence_conformal_scale": persistence_scale,
                "persistence_qlike": qlike(actual_var, persistence_test),
                "har_rv_qlike": qlike(actual_var, har_test),
                "ewma72_calibrated_qlike": qlike(actual_var, ewma_test),
                "har_distribution": distribution_metrics(returns, har_q),
                "simple_distribution": distribution_metrics(returns, simple_q),
                "incumbent_distribution": distribution_metrics(returns, incumbent_q),
                "incumbent_family": incumbent_bundle["choice"],
            })

            all_actual_var.append(actual_var)
            all_persistence_var.append(persistence_test)
            all_har_var.append(har_test)
            all_ewma_var.append(ewma_test)
            all_returns.append(returns)
            all_har_q.append(har_q)
            all_simple_q.append(simple_q)
            all_incumbent_q.append(incumbent_q)
            all_slots.extend(test)

        actual_var = np.concatenate(all_actual_var)
        persistence_var = np.concatenate(all_persistence_var)
        har_var = np.concatenate(all_har_var)
        ewma_var = np.concatenate(all_ewma_var)
        returns = np.concatenate(all_returns)
        har_q = np.vstack(all_har_q)
        simple_q = np.vstack(all_simple_q)
        incumbent_q = np.vstack(all_incumbent_q)
        slots = pd.DatetimeIndex(all_slots)

        variance_summary = {
            "persistence_qlike": qlike(actual_var, persistence_var),
            "har_rv_qlike": qlike(actual_var, har_var),
            "ewma72_calibrated_qlike": qlike(actual_var, ewma_var),
        }
        variance_summary["gate"] = variance_gate(
            block_reports, variance_summary["har_rv_qlike"], variance_summary["persistence_qlike"]
        )

        candidate_dist = distribution_metrics(returns, har_q)
        simple_dist = distribution_metrics(returns, simple_q)
        incumbent_dist = distribution_metrics(returns, incumbent_q)
        candidate_loss = per_row_wis80(returns, har_q)
        simple_loss = per_row_wis80(returns, simple_q)
        incumbent_loss = per_row_wis80(returns, incumbent_q)
        paired_simple = paired_probability(candidate_loss, simple_loss, slots, horizon)
        paired_incumbent = paired_probability(candidate_loss, incumbent_loss, slots, horizon)
        dist_gate = distribution_gate(candidate_dist, simple_dist, incumbent_dist, paired_simple, paired_incumbent)

        result["horizons"][str(horizon)] = {
            "blocks": block_reports,
            "variance": variance_summary,
            "distribution": {
                "candidate": candidate_dist,
                "simple": simple_dist,
                "incumbent": incumbent_dist,
                "paired_vs_simple": paired_simple,
                "paired_vs_incumbent": paired_incumbent,
                "gate": dist_gate,
            },
            "center_status": "fixed_no_change_not_evaluated_for_alpha",
            "event_status": "unchanged_not_part_of_R2_wave1",
        }

    result["runtime_seconds"] = time.monotonic() - started
    result["r2_wave1_pass"] = bool(any(
        h["variance"]["gate"]["passed"] or h["distribution"]["gate"]["passed"]
        for h in result["horizons"].values()
    ))
    (output / "result.json").write_bytes(canonical(result))
    (output / "REPORT.md").write_text(render_report(result))
    return result


def render_report(result: dict) -> str:
    lines = [
        "# EtherForecast R2 Wave 1 — Range / Volatility",
        "",
        f"Data as-of: `{result['data_as_of']}`",
        "",
        "Retrospective reconstructed development evidence only. Production promotion is disabled.",
        "",
        "| Horizon | HAR QLIKE skill | Variance gate | WIS vs simple | WIS vs incumbent | Coverage80 | Distribution gate |",
        "|---:|---:|:---:|---:|---:|---:|:---:|",
    ]
    for h, value in result["horizons"].items():
        vg = value["variance"]["gate"]
        dg = value["distribution"]["gate"]
        lines.append(
            f"| {h}h | {vg['qlike_improvement_vs_persistence']:.2%} | {'PASS' if vg['passed'] else 'FAIL'} "
            f"| {dg['wis_improvement_vs_simple']:.2%} | {dg['wis_improvement_vs_incumbent']:.2%} "
            f"| {dg['coverage80']:.2%} | {'PASS' if dg['passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "Passing a retrospective gate does not authorize live promotion. A passing head may proceed only to shadow/prospective evidence.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.root, args.output)
    print(json.dumps({
        "experiment_id": POLICY["experiment_id"],
        "data_as_of": result["data_as_of"],
        "r2_wave1_pass": result["r2_wave1_pass"],
        "horizons": {
            h: {
                "variance_gate": v["variance"]["gate"]["passed"],
                "distribution_gate": v["distribution"]["gate"]["passed"],
            }
            for h, v in result["horizons"].items()
        },
    }, sort_keys=True))


if __name__ == "__main__":
    main()
