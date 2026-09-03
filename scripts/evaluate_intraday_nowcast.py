"""Evaluate a causal four-hour breakout nowcast on official hourly archives."""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.intraday_nowcast import (
    ALERT_REFRESH_HOURS,
    CANDIDATES,
    EVENT_MERGE_GAP_HOURS,
    MAX_FALSE_ALERTS_PER_90_DAYS,
    NOWCAST_HORIZON_HOURS,
    NOWCAST_RETURN_THRESHOLD,
    PRICE_IMPULSE,
    THRESHOLD_LOOKBACK_HOURS,
    THRESHOLD_MIN_OBSERVATIONS,
    build_nowcast_table,
    evaluate_candidates,
    positive_events,
)
from forecasting.lead_signal_data import strict_json_dumps
from scripts.build_lead_signal_features import (
    archive_evidence_records,
    download_all_archives,
    load_hourly_streams,
)

SCHEMA_VERSION = 1
SMOKE_MONTH_COUNT = 13
SMOKE_TEST_DAYS = 180
FULL_START = pd.Timestamp("2020-01-01T00:00:00")


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected an object in {path}")
    return payload


def restrict_source_manifest(
    source_manifest: dict[str, Any], profile: str
) -> dict[str, Any]:
    """Bound automatic smoke downloads without changing source contracts."""
    output = copy.deepcopy(source_manifest)
    if profile == "full":
        return output
    if profile != "smoke":
        raise ValueError(f"Unknown profile: {profile}")

    coverage = output["sources"]["binance"]["coverage"]
    common_last = min(pd.Period(item["last_month"]) for item in coverage.values())
    first = common_last - (SMOKE_MONTH_COUNT - 1)
    for item in coverage.values():
        item["months"] = [
            month
            for month in item["months"]
            if first <= pd.Period(month) <= common_last
        ]
        if len(item["months"]) != SMOKE_MONTH_COUNT:
            raise ValueError("Smoke source restriction did not retain 13 months")
    return output


def select_profile_table(table: pd.DataFrame, profile: str) -> pd.DataFrame:
    if table.empty:
        raise ValueError("Nowcast table is empty")
    if profile == "smoke":
        cutoff = table.index.max()
        start = cutoff - pd.Timedelta(days=SMOKE_TEST_DAYS) + pd.Timedelta(hours=1)
        selected = table.loc[table.index >= start].copy()
        block_start = selected.index.min().floor("D")
        block_number = ((selected.index - block_start) / pd.Timedelta(days=30)).astype(
            int
        )
        selected["fold_id"] = [f"smoke_{value + 1:02d}" for value in block_number]
    elif profile == "full":
        selected = table.loc[table.index >= FULL_START].copy()
        selected["fold_id"] = selected.index.year.astype(str)
    else:
        raise ValueError(f"Unknown profile: {profile}")
    if selected.empty:
        raise ValueError(f"No eligible {profile} rows")
    return selected


def _source_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    evidence = archive_evidence_records(records)
    evidence_sha = hashlib.sha256(
        strict_json_dumps(evidence).encode("utf-8")
    ).hexdigest()
    sources: dict[str, Any] = {}
    for source_id in sorted({item["source_id"] for item in records}):
        items = [item for item in records if item["source_id"] == source_id]
        sources[source_id] = {
            "archive_count": len(items),
            "first_month": min(item["month"] for item in items),
            "last_month": max(item["month"] for item in items),
            "first_open_time": min(item["first_open_time"] for item in items),
            "last_open_time": max(item["last_open_time"] for item in items),
            "validation_statuses": sorted(
                {item["validation_status"] for item in items}
            ),
        }
    return {
        "archive_count": len(records),
        "download_bytes": sum(int(item["size_bytes"]) for item in records),
        "archive_evidence_sha256": evidence_sha,
        "sources": sources,
    }


def _calendar_breakdown(table: pd.DataFrame) -> dict[str, Any]:
    breakdown: dict[str, Any] = {}
    for fold_id, frame in table.groupby("fold_id", sort=True):
        metrics = evaluate_candidates(frame)
        breakdown[str(fold_id)] = {
            candidate: {
                key: value
                for key, value in result.items()
                if key not in {"event_first_alerts", "false_alert_samples"}
            }
            for candidate, result in metrics.items()
        }
    return breakdown


def _event_rows(table: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in positive_events(table["target_up_8pct_48h"]):
        frame = table.loc[event.start : event.end]
        rows.append(
            {
                "event_id": event.event_id,
                "start": event.start.isoformat(),
                "end": event.end.isoformat(),
                "origin_count": len(
                    frame.loc[frame["target_up_8pct_48h"].astype(bool)]
                ),
                "maximum_future_return_48h": float(
                    frame["future_max_return_48h"].max()
                ),
            }
        )
    return rows


def build_gate(
    *,
    profile: str,
    table: pd.DataFrame,
    metrics: dict[str, dict[str, Any]],
    runtime_seconds: float,
    peak_rss_mb: float,
    max_runtime_seconds: float,
    max_peak_rss_mb: float = 1024.0,
) -> dict[str, Any]:
    failures: list[str] = []
    if not table.index.is_unique or not table.index.is_monotonic_increasing:
        failures.append("eligible timestamps are not unique and monotonic")
    minimum_rows = 24 * (150 if profile == "smoke" else 365 * 4)
    if len(table) < minimum_rows:
        failures.append(f"only {len(table)} eligible rows; require {minimum_rows}")
    for candidate in CANDIDATES:
        if metrics[candidate]["rows"] != len(table):
            failures.append(f"unmatched rows for {candidate}")
        for key in ("event_recall", "false_alerts_per_90_days"):
            value = metrics[candidate][key]
            if value is not None and not np.isfinite(float(value)):
                failures.append(f"non-finite {key} for {candidate}")
    if runtime_seconds > max_runtime_seconds:
        failures.append(
            f"runtime {runtime_seconds:.1f}s exceeded {max_runtime_seconds:.1f}s"
        )
    if peak_rss_mb > max_peak_rss_mb:
        failures.append(
            f"peak RSS {peak_rss_mb:.1f}MB exceeded {max_peak_rss_mb:.1f}MB"
        )

    infrastructure_status = "PASS" if not failures else "FAIL"
    report: dict[str, Any] = {
        "profile": profile,
        "gate_status": infrastructure_status,
        "infrastructure_status": infrastructure_status,
        "promotion_status": "NOT_EVALUATED" if profile == "smoke" else "FAIL",
        "failures": failures,
        "runtime_seconds": runtime_seconds,
        "runtime_budget_seconds": max_runtime_seconds,
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_budget_mb": max_peak_rss_mb,
    }
    if profile == "smoke" or failures:
        return report

    within_budget = [
        candidate
        for candidate in CANDIDATES
        if metrics[candidate]["false_alerts_per_90_days"]
        <= MAX_FALSE_ALERTS_PER_90_DAYS
    ]
    eligible_winners = within_budget or list(CANDIDATES)

    def ranking(candidate: str) -> tuple[float, float, float]:
        result = metrics[candidate]
        recall = result["event_recall"] or 0.0
        precision = result["alert_precision"] or 0.0
        hours = result["median_hours_to_target"]
        return recall, precision, -(hours if hours is not None else float("inf"))

    winner = max(eligible_winners, key=ranking)
    result = metrics[winner]
    price = metrics[PRICE_IMPULSE]
    confirmed_guard = True
    if winner != PRICE_IMPULSE:
        confirmed_guard = bool(
            result["false_alerts_per_90_days"] <= price["false_alerts_per_90_days"]
            and (result["event_recall"] or 0.0) >= (price["event_recall"] or 0.0) - 0.05
        )
    checks = {
        "at_least_20_independent_events": result["event_count"] >= 20,
        "event_recall_at_least_35pct": bool(
            result["event_recall"] is not None and result["event_recall"] >= 0.35
        ),
        "alert_precision_at_least_20pct": bool(
            result["alert_precision"] is not None and result["alert_precision"] >= 0.20
        ),
        "false_alert_budget": bool(
            result["false_alerts_per_90_days"] <= MAX_FALSE_ALERTS_PER_90_DAYS
        ),
        "median_target_arrival_within_36h": bool(
            result["median_hours_to_target"] is not None
            and result["median_hours_to_target"] <= 36.0
        ),
        "detections_in_at_least_four_calendar_blocks": (
            result["detected_calendar_blocks"] >= 4
        ),
        "confirmation_does_not_trade_recall_for_false_budget": confirmed_guard,
    }
    promotion_status = "PASS" if all(checks.values()) else "FAIL"
    report.update(
        {
            "gate_status": promotion_status,
            "promotion_status": promotion_status,
            "winner": winner,
            "checks": checks,
            "within_false_alert_budget": within_budget,
        }
    )
    return report


def render_markdown(payload: dict[str, Any]) -> str:
    gate = payload["gate"]
    lines = [
        "# Intraday Breakout Nowcast Evaluation",
        "",
        f"- Profile: `{payload['profile']}`",
        f"- Infrastructure: **{gate['infrastructure_status']}**",
        f"- Shadow promotion: **{gate['promotion_status']}**",
        f"- Eligible hours: {payload['data']['eligible_rows']:,}",
        f"- Range: {payload['data']['start']} to {payload['data']['end']}",
        f"- Runtime: {payload['runtime_seconds']:.1f}s",
        "",
        "| Candidate | Event recall | Alert precision | False / 90d | Median hours |",
        "|---|---:|---:|---:|---:|",
    ]
    for candidate, result in payload["metrics"].items():
        recall = result["event_recall"]
        precision = result["alert_precision"]
        hours = result["median_hours_to_target"]
        lines.append(
            f"| {candidate} | {recall if recall is not None else float('nan'):.3f} | "
            f"{precision if precision is not None else float('nan'):.3f} | "
            f"{result['false_alerts_per_90_days']:.2f} | "
            f"{hours if hours is not None else float('nan'):.1f} |"
        )
    if gate["failures"]:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in gate["failures"])
    return "\n".join(lines) + "\n"


def run_evaluation(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    source_manifest = _read_json(args.source_manifest)
    readiness = _read_json(args.source_readiness)
    if readiness.get("decision") != "pass_for_pr2_offline":
        raise RuntimeError("Lead-signal source gate does not approve offline use")
    manifest_sha = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    if readiness.get("manifest_sha256") != manifest_sha:
        raise ValueError("Source readiness and manifest SHA-256 disagree")

    bounded_manifest = restrict_source_manifest(source_manifest, args.profile)
    records = download_all_archives(
        source_manifest=bounded_manifest,
        raw_dir=args.raw_dir,
        workers=args.workers,
        max_archive_bytes=args.max_archive_bytes,
        timeout_seconds=args.timeout_seconds,
    )
    streams = load_hourly_streams(records)
    excluded_dates = tuple(
        sorted(
            {
                pd.Timestamp(timestamp).floor("D")
                for item in records
                for timestamp in item["declared_market_gap_open_times"]
            }
        )
    )
    full_table = build_nowcast_table(streams, excluded_dates=excluded_dates)
    table = select_profile_table(full_table, args.profile)
    metrics = evaluate_candidates(table)
    runtime_seconds = float(time.monotonic() - started)
    peak_rss_mb = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
    max_runtime_seconds = float(
        args.max_runtime_seconds
        if args.max_runtime_seconds is not None
        else (600.0 if args.profile == "smoke" else 1800.0)
    )
    gate = build_gate(
        profile=args.profile,
        table=table,
        metrics=metrics,
        runtime_seconds=runtime_seconds,
        peak_rss_mb=peak_rss_mb,
        max_runtime_seconds=max_runtime_seconds,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": utc_now(),
        "mode": "offline_intraday_breakout_nowcast",
        "profile": args.profile,
        "decision_contract": {
            "bar_availability": "only after the full hourly bar closes",
            "horizon_hours": NOWCAST_HORIZON_HOURS,
            "target": "maximum forward ETH spot return reaches +8% within 48 hours",
            "return_threshold": NOWCAST_RETURN_THRESHOLD,
            "threshold_lookback_hours": THRESHOLD_LOOKBACK_HOURS,
            "threshold_min_observations": THRESHOLD_MIN_OBSERVATIONS,
            "current_bar_excluded_from_thresholds": True,
            "alert_refresh_hours": ALERT_REFRESH_HOURS,
            "event_merge_gap_hours": EVENT_MERGE_GAP_HOURS,
            "maximum_false_alerts_per_90_days": MAX_FALSE_ALERTS_PER_90_DAYS,
            "production_use": False,
            "daily_forecast_wiring": False,
        },
        "source": {
            "manifest_sha256": manifest_sha,
            "profile_archive_scope": (
                "last_13_complete_months" if args.profile == "smoke" else "full"
            ),
            **_source_summary(records),
        },
        "data": {
            "eligible_rows": len(table),
            "start": table.index.min().isoformat(),
            "end": table.index.max().isoformat(),
            "calendar_blocks": sorted(table["fold_id"].unique().tolist()),
            "quarantined_utc_dates": [
                value.date().isoformat() for value in excluded_dates
            ],
        },
        "candidates": list(CANDIDATES),
        "metrics": metrics,
        "calendar_breakdown": _calendar_breakdown(table),
        "events": _event_rows(table),
        "gate": gate,
        "runtime_seconds": runtime_seconds,
        "peak_rss_mb": peak_rss_mb,
    }
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "full"), required=True)
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=PROJECT_ROOT / "lake" / "manifests" / "lead_signal_sources.json",
    )
    parser.add_argument(
        "--source-readiness",
        type=Path,
        default=PROJECT_ROOT / "lake" / "reports" / "lead_signal_source_readiness.json",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "lake" / "raw" / "intraday_nowcast",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-archive-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--max-runtime-seconds", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between one and eight")
    payload = run_evaluation(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    gate = payload["gate"]
    print(
        json.dumps(
            {
                "profile": payload["profile"],
                "infrastructure_status": gate["infrastructure_status"],
                "promotion_status": gate["promotion_status"],
                "winner": gate.get("winner"),
                "runtime_seconds": payload["runtime_seconds"],
                "peak_rss_mb": payload["peak_rss_mb"],
                "failures": gate["failures"],
            },
            indent=2,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0 if gate["infrastructure_status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
