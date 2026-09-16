#!/usr/bin/env python3
"""Evaluate freshness of EtherForecast research data against the source registry."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd


SUPPLEMENT_NAME = "data_source_registry_existing_supplement.json"


def _utc(value) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        return stamp.tz_localize("UTC")
    return stamp.tz_convert("UTC")


def load_registry(path: str | Path) -> dict:
    registry_path = Path(path)
    data = json.loads(registry_path.read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("sources"), list):
        raise ValueError("invalid data source registry")

    supplement_path = registry_path.with_name(SUPPLEMENT_NAME)
    if supplement_path.exists():
        supplement = json.loads(supplement_path.read_text(encoding="utf-8"))
        if supplement.get("schema") != 1 or not isinstance(supplement.get("sources"), list):
            raise ValueError("invalid data source registry supplement")
        data["sources"] = [*data["sources"], *supplement["sources"]]
        data["supplement"] = supplement_path.name

    ids = [str(row.get("id", "")) for row in data["sources"]]
    if any(not source_id for source_id in ids) or len(ids) != len(set(ids)):
        raise ValueError("registry source ids must be non-empty and unique")
    paths = [str(row["path"]) for row in data["sources"] if row.get("path")]
    if len(paths) != len(set(paths)):
        raise ValueError("registry file paths must be unique")
    return data


def latest_timestamp(path: Path, configured_column: str | None) -> pd.Timestamp | None:
    if not path.exists() or not path.is_file() or path.stat().st_size == 0:
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None

    candidates = []
    if configured_column and configured_column in frame.columns:
        candidates.append(configured_column)
    for name in ("date", "timestamp", "time", "open_time", "observation_date", "initial_release_date"):
        if name in frame.columns and name not in candidates:
            candidates.append(name)
    if not candidates:
        candidates.append(frame.columns[0])

    for column in candidates:
        parsed = pd.to_datetime(frame[column], utc=True, errors="coerce")
        parsed = parsed.dropna()
        if not parsed.empty:
            return _utc(parsed.max())
    return None


def discover_unregistered_files(root: Path, registry: dict) -> list[str]:
    registered = {str(row["path"]) for row in registry["sources"] if row.get("path")}
    discovered: set[str] = set()
    for relative_dir in ("lake/raw/market", "lake/raw/vendor"):
        folder = root / relative_dir
        if not folder.exists():
            continue
        for path in folder.glob("*.csv"):
            if path.is_file() and not path.is_symlink():
                relative = path.relative_to(root).as_posix()
                if relative not in registered:
                    discovered.add(relative)
    return sorted(discovered)


def evaluate_source(root: Path, source: dict, now: pd.Timestamp) -> dict:
    state = str(source.get("state", "active"))
    result = {
        "id": source.get("id"),
        "family": source.get("family"),
        "state": state,
        "critical": bool(source.get("critical", False)),
        "path": source.get("path"),
        "workflow": source.get("workflow"),
        "cadence_minutes": source.get("cadence_minutes"),
        "soft_stale_hours": source.get("soft_stale_hours"),
        "hard_stale_hours": source.get("hard_stale_hours"),
        "pit_status": source.get("pit_status"),
        "storage_stream": source.get("storage_stream"),
        "role": source.get("role"),
    }

    if state != "active":
        result.update({"status": state, "latest": None, "age_hours": None, "hard_failure": False})
        return result

    relative = source.get("path")
    latest = latest_timestamp(root / relative, source.get("timestamp_column")) if relative else None
    if latest is None:
        result.update({
            "status": "missing",
            "latest": None,
            "age_hours": None,
            "hard_failure": bool(source.get("critical", False)),
        })
        return result

    age_hours = max(0.0, (now - latest).total_seconds() / 3600.0)
    soft = float(source.get("soft_stale_hours", 48))
    hard = float(source.get("hard_stale_hours", max(soft * 2, soft + 1)))
    if age_hours > hard:
        status = "stale"
    elif age_hours > soft:
        status = "warning"
    else:
        status = "ok"
    result.update({
        "status": status,
        "latest": latest.isoformat(),
        "age_hours": round(age_hours, 3),
        "hard_failure": bool(source.get("critical", False) and status == "stale"),
    })
    return result


def evaluate(registry: dict, root: Path, now: pd.Timestamp) -> dict:
    rows = [evaluate_source(root, source, now) for source in registry["sources"]]
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    hard_failures = [row["id"] for row in rows if row.get("hard_failure")]
    unregistered = discover_unregistered_files(root, registry)
    return {
        "schema": 1,
        "generated_at_utc": now.isoformat(),
        "status": "failure" if hard_failures else "ok",
        "hard_failures": hard_failures,
        "counts": counts,
        "unregistered_files": unregistered,
        "policy": registry.get("policy", {}),
        "registry_supplement": registry.get("supplement"),
        "sources": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", default="config/data_source_registry.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--json-output", default="lake/reports/data_refresh_status.json")
    parser.add_argument("--csv-output", default="lake/reports/data_refresh_status.csv")
    parser.add_argument("--now", default="")
    parser.add_argument("--fail-on-critical-stale", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    registry = load_registry(root / args.registry)
    now = _utc(args.now or datetime.now(timezone.utc))
    report = evaluate(registry, root, now)

    json_path = root / args.json_output
    csv_path = root / args.csv_output
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(report["sources"]).to_csv(csv_path, index=False)
    print(json.dumps({
        "status": report["status"],
        "counts": report["counts"],
        "hard_failures": report["hard_failures"],
        "unregistered_files": report["unregistered_files"],
    }, sort_keys=True))

    if args.fail_on_critical_stale and report["hard_failures"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
