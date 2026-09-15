import json
from pathlib import Path

import pandas as pd

from scripts.data_refresh_status import evaluate, load_registry


def _registry(sources):
    return {"schema": 1, "policy": {}, "sources": sources}


def test_repository_registry_has_unique_ids_and_valid_slas():
    registry = load_registry(Path(__file__).resolve().parents[1] / "config/data_source_registry.json")
    ids = [row["id"] for row in registry["sources"]]
    assert len(ids) == len(set(ids))
    for row in registry["sources"]:
        assert row["state"] in {"active", "planned", "blocked"}
        assert row["cadence_minutes"] > 0
        assert row["soft_stale_hours"] > 0
        assert row["hard_stale_hours"] >= row["soft_stale_hours"]


def test_refresh_guard_classifies_ok_warning_stale_and_planned(tmp_path):
    now = pd.Timestamp("2026-09-16T12:00:00Z")
    for name, stamp in (("ok.csv", "2026-09-16T08:00:00Z"), ("warn.csv", "2026-09-16T02:00:00Z"), ("stale.csv", "2026-09-15T00:00:00Z")):
        pd.DataFrame({"date": [stamp], "value": [1]}).to_csv(tmp_path / name, index=False)
    registry = _registry([
        {"id": "ok", "family": "x", "state": "active", "critical": False, "path": "ok.csv", "timestamp_column": "date", "cadence_minutes": 240, "soft_stale_hours": 8, "hard_stale_hours": 16},
        {"id": "warn", "family": "x", "state": "active", "critical": False, "path": "warn.csv", "timestamp_column": "date", "cadence_minutes": 240, "soft_stale_hours": 8, "hard_stale_hours": 16},
        {"id": "stale", "family": "x", "state": "active", "critical": False, "path": "stale.csv", "timestamp_column": "date", "cadence_minutes": 240, "soft_stale_hours": 8, "hard_stale_hours": 16},
        {"id": "future", "family": "x", "state": "planned", "critical": False, "path": None, "timestamp_column": None, "cadence_minutes": 60, "soft_stale_hours": 4, "hard_stale_hours": 12},
    ])
    report = evaluate(registry, tmp_path, now)
    statuses = {row["id"]: row["status"] for row in report["sources"]}
    assert statuses == {"ok": "ok", "warn": "warning", "stale": "stale", "future": "planned"}
    assert report["status"] == "ok"


def test_missing_or_stale_critical_source_causes_failure(tmp_path):
    now = pd.Timestamp("2026-09-16T12:00:00Z")
    registry = _registry([
        {"id": "critical", "family": "market", "state": "active", "critical": True, "path": "missing.csv", "timestamp_column": "date", "cadence_minutes": 60, "soft_stale_hours": 2, "hard_stale_hours": 4}
    ])
    report = evaluate(registry, tmp_path, now)
    assert report["status"] == "failure"
    assert report["hard_failures"] == ["critical"]


def test_timestamp_falls_back_to_first_parseable_column(tmp_path):
    pd.DataFrame({"when": ["2026-09-16T04:00:00Z"], "x": [1]}).to_csv(tmp_path / "data.csv", index=False)
    registry = _registry([
        {"id": "source", "family": "x", "state": "active", "critical": False, "path": "data.csv", "timestamp_column": "not_there", "cadence_minutes": 240, "soft_stale_hours": 8, "hard_stale_hours": 16}
    ])
    report = evaluate(registry, tmp_path, pd.Timestamp("2026-09-16T08:00:00Z"))
    assert report["sources"][0]["status"] == "ok"
