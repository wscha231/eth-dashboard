"""Preregistered HAR-RV origin-clock robustness audit.

Research only. This study asks whether the already-passing HAR-RV variance model is
stable when *issued* at UTC hours other than the original 00:00 cohort. Training
and calibration remain exactly on 00:00 UTC daily origins, matching the frozen R2
specification. No prospective record, production checkpoint, website output or
promotion state is modified here.

The expansion rule is intentionally strict and frozen before results: for a given
horizon, every one of the 24 UTC origin-hour cohorts must independently pass the
existing variance gate. We never select a subset of hours after inspecting the
result. A pass authorizes only a separate denser prospective cohort; it does not
promote HAR-RV to production.
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
from research.model.r2_range_volatility_wave1 import qlike, utc
from signal_pipeline.data import build_features

HORIZONS = (6, 24, 72)
VOL_COLUMNS = ("eth_vol_24", "eth_vol_168", "eth_vol_720")
ORIGIN_HOURS = tuple(range(24))

POLICY = {
    "experiment_id": "har_rv_clock_robustness_20260918",
    "roadmap_stage": "R4_R5_bottleneck_clock_robustness",
    "frozen_before_results": True,
    "evidence_grade": "retrospective_reconstructed_clock_robustness_not_prospective",
    "horizons_hours": list(HORIZONS),
    "tested_origin_hours_utc": list(ORIGIN_HOURS),
    "folds": [
        {"id": "2024", "start": "2024-01-01T00:00:00Z", "end": "2025-01-01T00:00:00Z"},
        {"id": "2025", "start": "2025-01-01T00:00:00Z", "end": "2026-01-01T00:00:00Z"},
        {"id": "2026", "start": "2026-01-01T00:00:00Z", "end": "2027-01-01T00:00:00Z"},
    ],
    "train_days": 730,
    "calibration_days": 180,
    "embargo_hours": 1,
    "training_origin": "00:00 UTC daily only; unchanged from frozen R2",
    "test_origin": "all 24 UTC clock hours, scored as 24 predeclared cohorts",
    "variance_target": "sum of h squared close-to-close log returns inside the h-bar path window",
    "har_features": list(VOL_COLUMNS),
    "har_model": "StandardScaler + Ridge(alpha=10) on log(realized_variance_total/h)",
    "calibration": "prior 00UTC calibration mean actual/predicted variance ratio",
    "baseline": "h * trailing_720h hourly sample variance",
    "clock_gate": {
        "qlike_improvement_vs_persistence_min": 0.05,
        "minimum_improving_calendar_blocks": 2,
        "calendar_blocks": 3,
        "maximum_single_block_degradation": 0.10,
    },
    "expansion_rule": "all 24 predeclared clock cohorts must pass independently for that horizon; no post-hoc hour selection",
    "prospective_effect_if_pass": "authorize only a separate denser shadow cohort; existing 00UTC ledger remains immutable",
    "promotion": "disabled",
    "prospective_min_nonoverlap": {"6": 120, "24": 60, "72": 40},
}


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _ready(features: pd.DataFrame, targets: pd.DataFrame) -> pd.Series:
    mask = features[list(VOL_COLUMNS)].notna().all(axis=1)
    mask &= targets["realized_variance_total"].gt(0)
    mask &= targets["return"].notna()
    return mask


def fit_indices(features: pd.DataFrame, targets: pd.DataFrame, start, boundary) -> pd.DatetimeIndex:
    start, boundary = utc(start), utc(boundary)
    mask = _ready(features, targets)
    mask &= features.index.hour == 0
    mask &= features.index >= start
    mask &= targets["target_end"] < boundary - pd.Timedelta(hours=POLICY["embargo_hours"])
    return features.index[mask]


def test_indices(features: pd.DataFrame, targets: pd.DataFrame, start, end) -> pd.DatetimeIndex:
    start, end = utc(start), utc(end)
    mask = _ready(features, targets)
    mask &= features.index >= start
    mask &= targets["target_end"] < end - pd.Timedelta(hours=POLICY["embargo_hours"])
    return features.index[mask]


def gate_clock(blocks: list[dict], aggregate_candidate: float, aggregate_persistence: float) -> dict:
    if len(blocks) != POLICY["clock_gate"]["calendar_blocks"]:
        raise ValueError("clock gate requires all declared calendar blocks")
    gate = POLICY["clock_gate"]
    improvement = 1.0 - aggregate_candidate / aggregate_persistence
    improving = sum(row["har_rv_qlike"] < row["persistence_qlike"] for row in blocks)
    degradations = [row["har_rv_qlike"] / row["persistence_qlike"] - 1.0 for row in blocks]
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


def prospective_timing(horizon: int) -> dict:
    minimum = int(POLICY["prospective_min_nonoverlap"][str(horizon)])
    current_spacing_days = int(np.ceil((horizon + 1) / 24.0))
    current_days = minimum * current_spacing_days
    theoretical_hourly_days = int(np.ceil(minimum * (horizon + 1) / 24.0))
    return {
        "minimum_nonoverlap": minimum,
        "current_00utc_daily_cohort_days_approx": current_days,
        "hourly_cohort_theoretical_floor_days": theoretical_hourly_days,
        "best_case_calendar_acceleration": float(current_days / theoretical_hourly_days),
    }


def evaluate(root: Path, output: Path) -> dict:
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError("clock-robustness output is immutable; choose a new path")
    output.mkdir(parents=True)
    (output / "policy.json").write_bytes(canonical(POLICY))
    source = root / "hourly.parquet"
    if not source.is_file():
        raise FileNotFoundError("event research hourly.parquet is required")

    started = time.monotonic()
    bars = pd.read_parquet(source)
    for column in ("open_time", "close_time", "observed_at"):
        bars[column] = pd.to_datetime(bars[column], utc=True)
    bars = bars.sort_values(["product", "close_time"]).reset_index(drop=True)
    features = build_features(bars)
    as_of = bars.groupby("product").close_time.max().min()

    result = {
        "policy": POLICY,
        "data_as_of": as_of.isoformat(),
        "horizons": {},
        "promotion": "disabled",
        "claims": "clock robustness only; no prospective superiority or price-alpha claim",
    }

    for horizon in HORIZONS:
        targets = target_frame(bars, features, horizon)
        clock_candidate: dict[int, list[np.ndarray]] = {hour: [] for hour in ORIGIN_HOURS}
        clock_persistence: dict[int, list[np.ndarray]] = {hour: [] for hour in ORIGIN_HOURS}
        clock_actual: dict[int, list[np.ndarray]] = {hour: [] for hour in ORIGIN_HOURS}
        clock_blocks: dict[int, list[dict]] = {hour: [] for hour in ORIGIN_HOURS}
        fold_reports = []

        for fold in POLICY["folds"]:
            test_start = utc(fold["start"])
            test_end = min(utc(fold["end"]), as_of + pd.Timedelta(hours=1))
            calibration_start = test_start - pd.Timedelta(days=POLICY["calibration_days"])
            training_start = calibration_start - pd.Timedelta(days=POLICY["train_days"])
            train = fit_indices(features, targets, training_start, calibration_start)
            calibration = fit_indices(features, targets, calibration_start, test_start)
            test = test_indices(features, targets, test_start, test_end)
            if len(train) < 400 or len(calibration) < 45 or len(test) < 24 * 60:
                raise ValueError(
                    f"insufficient rows for {horizon}h/{fold['id']}: {len(train)}/{len(calibration)}/{len(test)}"
                )

            vx = np.log(features[list(VOL_COLUMNS)].pow(2).clip(lower=1e-16))
            model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
                vx.loc[train],
                np.log(targets.loc[train, "realized_variance_total"].to_numpy() / horizon),
            )
            calibration_raw = np.exp(model.predict(vx.loc[calibration]))
            calibration_actual = targets.loc[calibration, "realized_variance_total"].to_numpy() / horizon
            multiplier = float(np.mean(calibration_actual / calibration_raw))
            if not np.isfinite(multiplier) or multiplier <= 0:
                raise ValueError("invalid HAR-RV calibration multiplier")

            candidate = horizon * np.exp(model.predict(vx.loc[test])) * multiplier
            persistence = horizon * features.loc[test, "eth_vol_720"].to_numpy() ** 2
            actual = targets.loc[test, "realized_variance_total"].to_numpy()
            fold_clock_rows = []

            for hour in ORIGIN_HOURS:
                mask = test.hour == hour
                slots = test[mask]
                if len(slots) < 60:
                    raise ValueError(f"missing declared clock cohort h={horizon} fold={fold['id']} hour={hour}")
                a = actual[mask]
                c = candidate[mask]
                p = persistence[mask]
                c_loss = qlike(a, c)
                p_loss = qlike(a, p)
                block = {
                    "fold": fold["id"],
                    "hour_utc": hour,
                    "rows": int(len(slots)),
                    "har_rv_qlike": c_loss,
                    "persistence_qlike": p_loss,
                    "qlike_improvement": float(1.0 - c_loss / p_loss),
                }
                clock_blocks[hour].append(block)
                clock_actual[hour].append(a)
                clock_candidate[hour].append(c)
                clock_persistence[hour].append(p)
                fold_clock_rows.append(block)

            fold_reports.append({
                "fold": fold["id"],
                "train_rows_00utc": int(len(train)),
                "calibration_rows_00utc": int(len(calibration)),
                "test_rows_all_clocks": int(len(test)),
                "har_multiplier": multiplier,
                "clock_rows": fold_clock_rows,
            })

        clock_reports = {}
        for hour in ORIGIN_HOURS:
            actual = np.concatenate(clock_actual[hour])
            candidate = np.concatenate(clock_candidate[hour])
            persistence = np.concatenate(clock_persistence[hour])
            c_loss = qlike(actual, candidate)
            p_loss = qlike(actual, persistence)
            clock_reports[str(hour)] = {
                "rows": int(len(actual)),
                "har_rv_qlike": c_loss,
                "persistence_qlike": p_loss,
                "gate": gate_clock(clock_blocks[hour], c_loss, p_loss),
                "blocks": clock_blocks[hour],
            }

        failed = [hour for hour in ORIGIN_HOURS if not clock_reports[str(hour)]["gate"]["passed"]]
        improvements = [clock_reports[str(hour)]["gate"]["qlike_improvement_vs_persistence"] for hour in ORIGIN_HOURS]
        result["horizons"][str(horizon)] = {
            "folds": fold_reports,
            "clocks": clock_reports,
            "all_clock_expansion_gate": {
                "passed": not failed,
                "passed_clocks": 24 - len(failed),
                "required_clocks": 24,
                "failed_clocks_utc": failed,
                "worst_clock_qlike_improvement": float(min(improvements)),
                "best_clock_qlike_improvement": float(max(improvements)),
                "rule": POLICY["expansion_rule"],
            },
            "prospective_timing": prospective_timing(horizon),
        }

    result["runtime_seconds"] = time.monotonic() - started
    result["any_dense_cohort_authorized"] = bool(
        any(value["all_clock_expansion_gate"]["passed"] for value in result["horizons"].values())
    )
    (output / "result.json").write_bytes(canonical(result))
    (output / "REPORT.md").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: dict) -> str:
    lines = [
        "# HAR-RV origin-clock robustness audit",
        "",
        f"Data as-of: `{result['data_as_of']}`",
        "",
        "This is a preregistered retrospective clock-robustness test. It does not modify the existing 00:00 UTC prospective ledger and cannot promote a production model.",
        "",
        "| Horizon | Passing clocks | Worst clock QLIKE skill | Dense prospective cohort | Current approx days | Theoretical hourly floor |",
        "|---:|---:|---:|:---:|---:|---:|",
    ]
    for horizon in HORIZONS:
        value = result["horizons"][str(horizon)]
        gate = value["all_clock_expansion_gate"]
        timing = value["prospective_timing"]
        lines.append(
            f"| {horizon}h | {gate['passed_clocks']}/24 | {gate['worst_clock_qlike_improvement']:.2%} | "
            f"{'AUTHORIZED FOR SEPARATE SHADOW ONLY' if gate['passed'] else 'BLOCKED'} | "
            f"{timing['current_00utc_daily_cohort_days_approx']} | {timing['hourly_cohort_theoretical_floor_days']} |"
        )
    lines.extend([
        "",
        "Expansion is all-or-nothing across the 24 predeclared UTC clock cohorts for each horizon. Failed clocks are not discarded after inspection.",
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
        "any_dense_cohort_authorized": result["any_dense_cohort_authorized"],
        "horizons": {
            h: {
                "passed": v["all_clock_expansion_gate"]["passed"],
                "passed_clocks": v["all_clock_expansion_gate"]["passed_clocks"],
                "failed_clocks_utc": v["all_clock_expansion_gate"]["failed_clocks_utc"],
                "worst_clock_qlike_improvement": v["all_clock_expansion_gate"]["worst_clock_qlike_improvement"],
            }
            for h, v in result["horizons"].items()
        },
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
