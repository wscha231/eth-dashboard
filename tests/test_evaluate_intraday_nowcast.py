from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from forecasting.intraday_nowcast import CANDIDATES, PRICE_IMPULSE
from scripts.evaluate_intraday_nowcast import (
    build_gate,
    restrict_source_manifest,
    select_profile_table,
)


def test_smoke_source_scope_retains_exactly_13_complete_months() -> None:
    manifest = json.loads(
        Path("lake/manifests/lead_signal_sources.json").read_text(encoding="utf-8")
    )

    bounded = restrict_source_manifest(manifest, "smoke")
    coverage = bounded["sources"]["binance"]["coverage"]

    assert {len(item["months"]) for item in coverage.values()} == {13}
    assert {item["months"][-1] for item in coverage.values()} == {"2026-07"}
    assert all(item["months"][0] == "2025-07" for item in coverage.values())


def test_smoke_profile_keeps_six_30_day_calendar_blocks() -> None:
    index = pd.date_range("2025-01-01", periods=24 * 400, freq="h")
    table = pd.DataFrame({"value": range(len(index))}, index=index)

    selected = select_profile_table(table, "smoke")

    assert selected.index.max() - selected.index.min() <= pd.Timedelta(days=180)
    assert selected["fold_id"].nunique() == 6


def _strong_metrics() -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for candidate in CANDIDATES:
        metrics[candidate] = {
            "rows": 24 * 365 * 4,
            "event_count": 30,
            "event_recall": 0.40,
            "alert_precision": 0.25,
            "false_alerts_per_90_days": 2.5,
            "median_hours_to_target": 24.0,
            "detected_calendar_blocks": 5,
        }
    metrics[PRICE_IMPULSE]["event_recall"] = 0.38
    metrics[PRICE_IMPULSE]["false_alerts_per_90_days"] = 2.8
    metrics["cross_market_confirmed"]["event_recall"] = 0.45
    metrics["cross_market_confirmed"]["alert_precision"] = 0.35
    metrics["cross_market_confirmed"]["false_alerts_per_90_days"] = 2.0
    return metrics


def test_full_gate_only_authorizes_shadow_when_all_checks_pass() -> None:
    index = pd.date_range("2021-01-01", periods=24 * 365 * 4, freq="h")
    table = pd.DataFrame(index=index)

    gate = build_gate(
        profile="full",
        table=table,
        metrics=_strong_metrics(),
        runtime_seconds=60.0,
        peak_rss_mb=200.0,
        max_runtime_seconds=1800.0,
    )

    assert gate["infrastructure_status"] == "PASS"
    assert gate["promotion_status"] == "PASS"
    assert gate["winner"] == "cross_market_confirmed"
    assert all(gate["checks"].values())


def test_smoke_gate_never_claims_shadow_promotion() -> None:
    index = pd.date_range("2026-01-01", periods=24 * 180, freq="h")
    table = pd.DataFrame(index=index)
    metrics = _strong_metrics()
    for result in metrics.values():
        result["rows"] = len(table)

    gate = build_gate(
        profile="smoke",
        table=table,
        metrics=metrics,
        runtime_seconds=60.0,
        peak_rss_mb=200.0,
        max_runtime_seconds=600.0,
    )

    assert gate["infrastructure_status"] == "PASS"
    assert gate["promotion_status"] == "NOT_EVALUATED"
