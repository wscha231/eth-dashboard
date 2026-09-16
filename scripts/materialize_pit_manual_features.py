#!/usr/bin/env python3
"""Materialize daily manual features only at their declared availability time.

Raw vendor/event-date files stay unchanged.  This adapter consumes the DATA
availability sidecar and writes a separate model-facing CSV whose row index is
the first UTC daily decision boundary at or after ``available_at``.

The transform is deliberately fail-closed: every non-null feature cell must
have exactly one matching (event date, column) sidecar row.  Historical
backfills remain reconstructed research; this script does not invent receipt
or publication times.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

REQUIRED_SIDECAR_COLUMNS = {"date", "column", "available_at"}


def _utc_index(values: Any) -> pd.DatetimeIndex:
    index = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors="raise"))
    if index.hasnans:
        raise ValueError("timestamps must be finite")
    return index


def load_feature_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"feature CSV missing or empty: {source}")
    frame = pd.read_csv(source)
    if frame.empty:
        raise ValueError(f"feature CSV has no rows: {source}")
    date_col = next(
        (c for c in ("date", "timestamp", "time") if c in frame.columns),
        frame.columns[0],
    )
    index = _utc_index(frame.pop(date_col))
    if index.has_duplicates:
        raise ValueError("feature dates must be unique")
    frame.index = index
    frame.index.name = "date"
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.sort_index()


def load_sidecar(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.is_file() or source.stat().st_size == 0:
        raise ValueError(f"availability sidecar missing or empty: {source}")
    sidecar = pd.read_csv(source)
    missing = REQUIRED_SIDECAR_COLUMNS - set(sidecar.columns)
    if missing:
        raise ValueError(f"availability sidecar missing columns: {sorted(missing)}")
    sidecar = sidecar.copy()
    sidecar["date"] = _utc_index(sidecar["date"])
    sidecar["available_at"] = _utc_index(sidecar["available_at"])
    sidecar["column"] = sidecar["column"].astype(str)
    if sidecar[["date", "column"]].duplicated().any():
        raise ValueError("duplicate availability entry for event date and column")
    return sidecar


def first_daily_decision_at_or_after(value: pd.Timestamp) -> pd.Timestamp:
    """Return the first UTC midnight that cannot precede ``value``."""
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("available_at must be timezone-aware")
    timestamp = timestamp.tz_convert("UTC")
    return timestamp.ceil("D")


def materialize_pit_frame(features: pd.DataFrame, sidecar: pd.DataFrame) -> pd.DataFrame:
    """Move each observed feature cell to its first eligible daily decision row.

    A single daily feature row can contain columns with different availability
    times, so the transform is cell-wise rather than a blanket one-day shift.
    If several event rows for one column become eligible on the same decision
    date, the latest event row wins deterministically.
    """
    if features.empty:
        result = features.copy()
        result.index.name = "date"
        return result

    frame = features.copy()
    frame.index = _utc_index(frame.index)
    if frame.index.has_duplicates:
        raise ValueError("feature dates must be unique")

    required = REQUIRED_SIDECAR_COLUMNS - set(sidecar.columns)
    if required:
        raise ValueError(f"availability sidecar missing columns: {sorted(required)}")
    availability = sidecar.copy()
    availability["date"] = _utc_index(availability["date"])
    availability["available_at"] = _utc_index(availability["available_at"])
    availability["column"] = availability["column"].astype(str)
    if availability[["date", "column"]].duplicated().any():
        raise ValueError("duplicate availability entry for event date and column")

    lookup = {
        (row.date, row.column): row.available_at
        for row in availability[["date", "column", "available_at"]].itertuples(index=False)
    }

    chosen: dict[tuple[pd.Timestamp, str], tuple[pd.Timestamp, float]] = {}
    for event_at, row in frame.sort_index().iterrows():
        for column, raw_value in row.items():
            if pd.isna(raw_value):
                continue
            key = (event_at, str(column))
            if key not in lookup:
                raise ValueError(
                    "missing availability for non-null feature: "
                    f"{event_at.isoformat()} {column}"
                )
            available_at = lookup[key]
            if available_at < event_at:
                raise ValueError(
                    "available_at precedes event date: "
                    f"{event_at.isoformat()} {column} -> {available_at.isoformat()}"
                )
            decision_at = first_daily_decision_at_or_after(available_at)
            output_key = (decision_at, str(column))
            previous = chosen.get(output_key)
            candidate = (event_at, float(raw_value))
            if previous is None or event_at > previous[0]:
                chosen[output_key] = candidate

    if not chosen:
        output = pd.DataFrame(columns=frame.columns, dtype=float)
        output.index = pd.DatetimeIndex([], tz="UTC", name="date")
        return output

    decision_index = pd.DatetimeIndex(sorted({key[0] for key in chosen}), name="date")
    output = pd.DataFrame(index=decision_index, columns=frame.columns, dtype=float)
    for (decision_at, column), (_, value) in chosen.items():
        output.loc[decision_at, column] = value
    return output.sort_index()


def materialize(input_path: str | Path, sidecar_path: str | Path, output_path: str | Path) -> dict:
    features = load_feature_csv(input_path)
    sidecar = load_sidecar(sidecar_path)
    output = materialize_pit_frame(features, sidecar)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index_label="date")
    return {
        "input": str(input_path),
        "availability": str(sidecar_path),
        "output": str(destination),
        "input_rows": int(len(features)),
        "output_rows": int(len(output)),
        "columns": int(len(features.columns)),
        "first_output": output.index.min().isoformat() if len(output) else None,
        "last_output": output.index.max().isoformat() if len(output) else None,
        "policy": "first_utc_midnight_at_or_after_available_at",
        "vintage_note": "availability is enforced; reconstructed history is not converted into original receipts",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--availability", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = materialize(args.input, args.availability, args.output)
    if args.receipt:
        receipt = Path(args.receipt)
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
