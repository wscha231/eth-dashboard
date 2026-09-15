#!/usr/bin/env python3
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_lab.free_sources import build_fred_initial_asof_daily, coverage, utc
from data_lab.free_source_fallbacks import (
    build_bitget_features,
    collect_bitget_basis,
    collect_bitget_funding,
    collect_bitget_ticker_snapshot,
    collect_fred_current_long,
    collect_fred_initial_partial,
)
from scripts.collect_free_research_sources import append_long, load_csv, save_history


def error_name(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    return f"HTTP {code}" if code is not None else type(exc).__name__


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="lake/raw/vendor")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = utc(args.start_date).floor("D")
    end = utc(args.end_date or datetime.now(timezone.utc)).floor("D")
    report: dict = {
        "schema": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "status": "research_only_reconstructed_history",
        "sources": {},
    }

    bitget_parts = {}
    for name, fn in (
        ("funding", lambda: collect_bitget_funding(start, end + pd.Timedelta(days=1))),
        ("basis", lambda: collect_bitget_basis(start, end)),
        ("ticker_snapshot", lambda: collect_bitget_ticker_snapshot(end)),
    ):
        try:
            frame = fn()
            bitget_parts[name] = frame
            report["sources"][f"bitget_{name}"] = {"status": "ok" if not frame.empty else "empty", **coverage(frame).__dict__}
        except Exception as exc:
            bitget_parts[name] = pd.DataFrame()
            report["sources"][f"bitget_{name}"] = {"status": "error", "error": error_name(exc)}

    features = build_bitget_features(
        bitget_parts.get("funding", pd.DataFrame()),
        bitget_parts.get("basis", pd.DataFrame()),
        bitget_parts.get("ticker_snapshot", pd.DataFrame()),
    )
    feature_path = out / "bitget_eth_free_features.csv"
    merged_features = save_history(feature_path, features)
    report["sources"]["bitget_features"] = {
        "status": "ok" if not merged_features.empty else "empty",
        "note": "OI is a daily snapshot accumulated prospectively; historical OI is not synthesized",
        **coverage(merged_features).__dict__,
    }

    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key:
        current, current_errors = collect_fred_current_long(fred_key, start, end)
        current_path = out / "fred_current_vintage_long.csv"
        current_merged = append_long(current_path, current)
        report["sources"]["fred_current_vintage"] = {
            "status": "ok" if not current_merged.empty else "empty",
            "rows": int(len(current_merged)),
            "series_errors": current_errors,
            "warning": "latest revised history; research only for old dates unless release lag is modeled",
        }

        initial, initial_errors = collect_fred_initial_partial(fred_key)
        initial_path = out / "fred_initial_release_long.csv"
        initial_merged = append_long(initial_path, initial)
        asof_path = out / "fred_initial_asof_daily.csv"
        if not initial_merged.empty:
            asof = build_fred_initial_asof_daily(initial_merged, start, end)
            asof.to_csv(asof_path, index_label="date")
        else:
            asof = load_csv(asof_path)
        report["sources"]["fred_initial_release"] = {
            "status": "partial" if initial_errors and not initial_merged.empty else ("ok" if not initial_merged.empty else "error"),
            "long_rows": int(len(initial_merged)),
            "asof_rows": int(len(asof)),
            "series_errors": initial_errors,
            "available_at_policy": "date-only initial release becomes eligible next UTC day",
        }
    else:
        report["sources"]["fred_current_vintage"] = {"status": "skipped", "reason": "FRED_API_KEY missing"}
        report["sources"]["fred_initial_release"] = {"status": "skipped", "reason": "FRED_API_KEY missing"}

    report_path = out.parent.parent / "reports" / "free_source_fallbacks.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
