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
from data_lab.market_research_sources import (
    build_hyperliquid_features,
    collect_deribit_eth_dvol,
    collect_hyperliquid_eth_funding,
    collect_hyperliquid_eth_snapshot,
)
from scripts.collect_free_research_sources import append_long, load_csv, save_history

ROLLING_WARMUP_DAYS = 30
DERIVATIVE_OVERLAP_DAYS = 60
FRED_OVERLAP_DAYS = 180


def error_name(exc: Exception) -> str:
    code = getattr(exc, "code", None)
    return f"HTTP {code}" if code is not None else type(exc).__name__


def cached_tail_start(
    path: Path,
    requested_start: pd.Timestamp,
    *,
    overlap_days: int,
    full_reconcile: bool,
) -> pd.Timestamp:
    if full_reconcile:
        return requested_start
    existing = load_csv(path)
    if existing.empty:
        return requested_start
    latest = pd.DatetimeIndex(existing.index).max()
    if pd.isna(latest):
        return requested_start
    latest = utc(latest)
    return max(requested_start, latest - pd.Timedelta(days=int(overlap_days)))


def long_tail_start(
    path: Path,
    date_column: str,
    requested_start: pd.Timestamp,
    *,
    overlap_days: int,
    full_reconcile: bool,
) -> pd.Timestamp:
    if full_reconcile or not path.exists() or path.stat().st_size == 0:
        return requested_start
    try:
        frame = pd.read_csv(path, usecols=[date_column])
    except (ValueError, OSError, pd.errors.EmptyDataError):
        return requested_start
    if frame.empty:
        return requested_start
    parsed = pd.to_datetime(frame[date_column], utc=True, errors="coerce").dropna()
    if parsed.empty:
        return requested_start
    return max(requested_start, utc(parsed.max()) - pd.Timedelta(days=int(overlap_days)))


def trim_incremental_warmup(
    frame: pd.DataFrame,
    fetch_start: pd.Timestamp,
    requested_start: pd.Timestamp,
    *,
    full_reconcile: bool,
    warmup_days: int = ROLLING_WARMUP_DAYS,
) -> pd.DataFrame:
    if frame is None or frame.empty or full_reconcile or fetch_start <= requested_start:
        return frame
    cutoff = fetch_start + pd.Timedelta(days=int(warmup_days))
    return frame.loc[frame.index >= cutoff]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="lake/raw/vendor")
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="")
    parser.add_argument("--full-reconcile", action="store_true")
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    start = utc(args.start_date).floor("D")
    end = utc(args.end_date or datetime.now(timezone.utc)).floor("D")
    refresh_mode = "full_reconcile" if args.full_reconcile else "incremental"
    report: dict = {
        "schema": 3,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "requested_start": start.isoformat(),
        "requested_end": end.isoformat(),
        "refresh_mode": refresh_mode,
        "status": "research_only_reconstructed_history",
        "sources": {},
    }

    bitget_path = out / "bitget_eth_free_features.csv"
    bitget_start = cached_tail_start(
        bitget_path,
        start,
        overlap_days=DERIVATIVE_OVERLAP_DAYS,
        full_reconcile=args.full_reconcile,
    )
    bitget_parts = {}
    for name, fn in (
        ("funding", lambda: collect_bitget_funding(bitget_start, end + pd.Timedelta(days=1))),
        ("basis", lambda: collect_bitget_basis(bitget_start, end)),
        ("ticker_snapshot", lambda: collect_bitget_ticker_snapshot(end)),
    ):
        try:
            frame = fn()
            bitget_parts[name] = frame
            report["sources"][f"bitget_{name}"] = {
                "status": "ok" if not frame.empty else "empty",
                "fetch_start": bitget_start.isoformat() if name != "ticker_snapshot" else end.isoformat(),
                **coverage(frame).__dict__,
            }
        except Exception as exc:
            bitget_parts[name] = pd.DataFrame()
            report["sources"][f"bitget_{name}"] = {
                "status": "error",
                "fetch_start": bitget_start.isoformat() if name != "ticker_snapshot" else end.isoformat(),
                "error": error_name(exc),
            }

    bitget_features = build_bitget_features(
        bitget_parts.get("funding", pd.DataFrame()),
        bitget_parts.get("basis", pd.DataFrame()),
        bitget_parts.get("ticker_snapshot", pd.DataFrame()),
    )
    bitget_update = trim_incremental_warmup(
        bitget_features, bitget_start, start, full_reconcile=args.full_reconcile
    )
    merged_features = save_history(bitget_path, bitget_update)
    report["sources"]["bitget_features"] = {
        "status": "ok" if not merged_features.empty else "empty",
        "fetch_start": bitget_start.isoformat(),
        "note": "OI is a daily snapshot accumulated prospectively; historical OI is not synthesized",
        **coverage(merged_features).__dict__,
    }

    dvol_path = out / "deribit_eth_dvol_daily.csv"
    dvol_start = cached_tail_start(
        dvol_path,
        start,
        overlap_days=DERIVATIVE_OVERLAP_DAYS,
        full_reconcile=args.full_reconcile,
    )
    try:
        dvol = collect_deribit_eth_dvol(dvol_start, end + pd.Timedelta(days=1))
        dvol_update = trim_incremental_warmup(
            dvol, dvol_start, start, full_reconcile=args.full_reconcile
        )
        dvol_merged = save_history(dvol_path, dvol_update)
        report["sources"]["deribit_eth_dvol"] = {
            "status": "ok" if not dvol_merged.empty else "empty",
            "fetch_start": dvol_start.isoformat(),
            "note": "Deribit volatility-index history in native DVOL units; reconstructed research backfill",
            **coverage(dvol_merged).__dict__,
        }
    except Exception as exc:
        cached = load_csv(dvol_path)
        report["sources"]["deribit_eth_dvol"] = {
            "status": "warning_cached" if not cached.empty else "error",
            "fetch_start": dvol_start.isoformat(),
            "error": error_name(exc),
            **coverage(cached).__dict__,
        }

    hyper_path = out / "hyperliquid_eth_free_features.csv"
    hyper_start = cached_tail_start(
        hyper_path,
        start,
        overlap_days=DERIVATIVE_OVERLAP_DAYS,
        full_reconcile=args.full_reconcile,
    )
    hyper_parts = {}
    for name, fn in (
        ("funding", lambda: collect_hyperliquid_eth_funding(hyper_start, end + pd.Timedelta(days=1))),
        ("snapshot", lambda: collect_hyperliquid_eth_snapshot(end)),
    ):
        try:
            frame = fn()
            hyper_parts[name] = frame
            report["sources"][f"hyperliquid_{name}"] = {
                "status": "ok" if not frame.empty else "empty",
                "fetch_start": hyper_start.isoformat() if name == "funding" else end.isoformat(),
                **coverage(frame).__dict__,
            }
        except Exception as exc:
            hyper_parts[name] = pd.DataFrame()
            report["sources"][f"hyperliquid_{name}"] = {
                "status": "error",
                "fetch_start": hyper_start.isoformat() if name == "funding" else end.isoformat(),
                "error": error_name(exc),
            }

    hyper_features = build_hyperliquid_features(
        hyper_parts.get("funding", pd.DataFrame()),
        hyper_parts.get("snapshot", pd.DataFrame()),
    )
    hyper_update = trim_incremental_warmup(
        hyper_features, hyper_start, start, full_reconcile=args.full_reconcile
    )
    hyper_merged = save_history(hyper_path, hyper_update)
    report["sources"]["hyperliquid_features"] = {
        "status": "ok" if not hyper_merged.empty else "empty",
        "fetch_start": hyper_start.isoformat(),
        "note": "Funding is historical; OI/premium/mark context snapshots accumulate prospectively and are never backfilled synthetically",
        **coverage(hyper_merged).__dict__,
    }

    fred_key = os.getenv("FRED_API_KEY", "").strip()
    if fred_key:
        current_path = out / "fred_current_vintage_long.csv"
        current_start = long_tail_start(
            current_path,
            "observation_date",
            start,
            overlap_days=FRED_OVERLAP_DAYS,
            full_reconcile=args.full_reconcile,
        )
        current, current_errors = collect_fred_current_long(fred_key, current_start, end)
        current_merged = append_long(current_path, current)
        report["sources"]["fred_current_vintage"] = {
            "status": "ok" if not current_merged.empty else "empty",
            "fetch_start": current_start.isoformat(),
            "rows": int(len(current_merged)),
            "series_errors": current_errors,
            "warning": "latest revised history; research only for old dates unless release lag is modeled",
        }

        initial_path = out / "fred_initial_release_long.csv"
        initial_start = long_tail_start(
            initial_path,
            "initial_release_date",
            start,
            overlap_days=FRED_OVERLAP_DAYS,
            full_reconcile=args.full_reconcile,
        )
        initial, initial_errors = collect_fred_initial_partial(
            fred_key,
            realtime_start=None if args.full_reconcile else initial_start,
        )
        initial_merged = append_long(initial_path, initial)
        asof_path = out / "fred_initial_asof_daily.csv"
        if not initial_merged.empty:
            asof = build_fred_initial_asof_daily(initial_merged, start, end)
            asof.to_csv(asof_path, index_label="date")
        else:
            asof = load_csv(asof_path)
        report["sources"]["fred_initial_release"] = {
            "status": "partial" if initial_errors and not initial_merged.empty else ("ok" if not initial_merged.empty else "error"),
            "fetch_start": initial_start.isoformat(),
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
