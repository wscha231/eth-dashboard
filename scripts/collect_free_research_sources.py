#!/usr/bin/env python3
"""Backfill free ETH research sources into durable daily vendor caches.

Outputs are ordinary data files only; no model is loaded or promoted.
Historical backfills are labeled reconstructed research. Data files can be
merged by the existing daily collector via --manual-features-csv.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_lab.free_sources import (
    DEFAULT_FRED_INITIAL_SERIES,
    availability_sidecar,
    build_bybit_feature_frame,
    build_fred_initial_asof_daily,
    collect_bybit_funding_history,
    collect_bybit_open_interest_history,
    collect_bybit_premium_index_history,
    collect_fred_initial_release_long,
    coverage,
    utc,
)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    if frame.empty:
        return frame
    date_col = next((c for c in ("date", "timestamp", "time") if c in frame.columns), frame.columns[0])
    idx = pd.to_datetime(frame.pop(date_col), utc=True, errors="coerce")
    frame.index = idx
    frame = frame.loc[frame.index.notna()].sort_index()
    for c in frame:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame


def save_history(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = load_csv(path)
    frames = [f for f in (old, new) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames).sort_index()
    merged = merged[~merged.index.duplicated(keep="last")]
    merged.index = pd.to_datetime(merged.index, utc=True)
    merged.index.name = "date"
    merged.to_csv(path, index_label="date")
    return merged


def append_long(path: Path, new: pd.DataFrame) -> pd.DataFrame:
    path.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(path) if path.exists() and path.stat().st_size else pd.DataFrame()
    frames = [f for f in (old, new) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True)
    key = [c for c in ("series_id", "observation_date", "initial_release_date") if c in merged]
    if key:
        merged = merged.drop_duplicates(key, keep="last")
    order = [c for c in ("series_id", "initial_release_date", "observation_date") if c in merged]
    if order:
        merged = merged.sort_values(order)
    merged.to_csv(path, index=False)
    return merged


def choose_start(path: Path, requested: pd.Timestamp, end: pd.Timestamp, overlap_days: int) -> pd.Timestamp:
    existing = load_csv(path)
    if existing.empty:
        return requested
    # If the cache does not reach the requested historical start, finish the
    # backfill. Otherwise refresh only an overlapping tail.
    if existing.index.min() > requested + pd.Timedelta(days=7):
        return requested
    return max(requested, existing.index.max() - pd.Timedelta(days=overlap_days))


def collect_defillama_full(output_dir: Path, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:
    """Reuse the existing tested collectors but force a historical start."""
    from eth_data_collector import (
        DEFAULT_DEFILLAMA_CHAIN,
        collect_defillama_chain_tvl,
        collect_defillama_dex_volume,
        collect_defillama_stablecoins,
    )
    configs = [
        ("defillama_ethereum_stablecoins_daily.csv", collect_defillama_stablecoins),
        ("defillama_ethereum_chain_tvl_daily.csv", collect_defillama_chain_tvl),
        ("defillama_ethereum_dex_volume_daily.csv", collect_defillama_dex_volume),
    ]
    result: dict[str, Any] = {}
    for filename, fn in configs:
        path = output_dir / filename
        try:
            fetched = fn(DEFAULT_DEFILLAMA_CHAIN, start.tz_localize(None), end.tz_localize(None))
            if fetched is not None and not fetched.empty:
                if fetched.index.tz is None:
                    fetched.index = fetched.index.tz_localize("UTC")
                else:
                    fetched.index = fetched.index.tz_convert("UTC")
            merged = save_history(path, fetched)
            result[filename] = {"status": "ok", **asdict(coverage(merged))}
        except Exception as exc:
            result[filename] = {"status": "error", "error": type(exc).__name__}
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="lake/raw/vendor")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--skip-bybit", action="store_true")
    parser.add_argument("--skip-fred-vintages", action="store_true")
    parser.add_argument("--skip-defillama-backfill", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = utc(args.start_date).floor("D")
    end = utc(args.end_date or datetime.now(timezone.utc)).floor("D")
    if start > end:
        raise SystemExit("start date is later than end date")

    report: dict[str, Any] = {
        "schema": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "research_status": "reconstructed_backfill_unless_receipt_timestamp_is_recorded",
        "sources": {},
    }

    manual_files: list[str] = []

    if not args.skip_bybit:
        feature_path = out / "bybit_eth_free_features.csv"
        incremental_start = choose_start(feature_path, start, end, overlap_days=10)
        try:
            funding = collect_bybit_funding_history(incremental_start, end + pd.Timedelta(days=1))
            oi = collect_bybit_open_interest_history(incremental_start, end + pd.Timedelta(days=1))
            premium = collect_bybit_premium_index_history(incremental_start, end + pd.Timedelta(days=1))
            features = build_bybit_feature_frame(funding, oi, premium)
            merged = save_history(feature_path, features)
            manual_files.append(str(feature_path))
            sidecar = availability_sidecar(merged, "bybit_public_v5")
            sidecar.to_csv(out / "bybit_eth_free_features.availability.csv", index=False)
            report["sources"]["bybit"] = {
                "status": "ok",
                "symbol": "ETHUSDT",
                "units": {
                    "open_interest": "ETH for linear ETHUSDT",
                    "funding_rate": "rate",
                    "premium_index": "ratio/index level",
                },
                "available_at_policy": "daily aggregate becomes eligible next UTC day",
                "history_start_requested": incremental_start.isoformat(),
                **asdict(coverage(merged)),
            }
        except Exception as exc:
            cached = load_csv(feature_path)
            if not cached.empty:
                manual_files.append(str(feature_path))
                report["sources"]["bybit"] = {
                    "status": "warning_cached",
                    "error": type(exc).__name__,
                    **asdict(coverage(cached)),
                }
            else:
                report["sources"]["bybit"] = {"status": "error", "error": type(exc).__name__}

    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if not args.skip_fred_vintages and fred_key:
        long_path = out / "fred_initial_release_long.csv"
        asof_path = out / "fred_initial_asof_daily.csv"
        try:
            initial = collect_fred_initial_release_long(fred_key)
            merged_long = append_long(long_path, initial)
            asof = build_fred_initial_asof_daily(merged_long, start, end)
            asof.index.name = "date"
            asof.to_csv(asof_path, index_label="date")
            manual_files.append(str(asof_path))
            report["sources"]["fred_initial_release"] = {
                "status": "ok",
                "series": list(DEFAULT_FRED_INITIAL_SERIES),
                "long_rows": int(len(merged_long)),
                "asof_rows": int(len(asof)),
                "available_at_policy": "date-only initial release becomes eligible next UTC day",
                "rights": "FRED/ALFRED source terms still require intended-use review",
            }
        except Exception as exc:
            report["sources"]["fred_initial_release"] = {
                "status": "error",
                "error": type(exc).__name__,
            }
    elif not args.skip_fred_vintages:
        report["sources"]["fred_initial_release"] = {
            "status": "skipped",
            "reason": "FRED_API_KEY not configured",
        }

    if not args.skip_defillama_backfill:
        report["sources"]["defillama_backfill"] = collect_defillama_full(out, start, end)

    report["manual_feature_files"] = manual_files
    report_path = out.parent.parent / "reports" / "free_source_backfill.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
