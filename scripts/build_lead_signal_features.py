from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.lead_signal_data import (
    BINANCE_ARCHIVE_BASE_URL,
    BINANCE_ARCHIVE_FALLBACK_URL,
    BINANCE_ARCHIVE_SPECS,
    BINANCE_SPOT_MICROSECOND_START,
    BinanceArchiveSpec,
    fetch_url_bytes,
    month_bounds,
    parse_checksum_text,
    read_binance_kline_zip,
    sha256_file,
    strict_json_dumps,
    validate_hourly_klines,
    verify_checksum,
    write_daily_csv,
    write_immutable_json,
)
from forecasting.lead_signals import (
    REQUIRED_STREAM_IDS,
    build_lead_signal_daily_features,
    feature_group_columns,
)


def log(message: str) -> None:
    print(message, flush=True)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _fetch_with_fallback(
    urls: tuple[str, ...],
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> tuple[bytes, str]:
    errors: list[str] = []
    for url in urls:
        try:
            payload = fetch_url_bytes(
                url,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
            )
            return payload, url
        except Exception as exc:  # noqa: BLE001 - retain each route failure
            errors.append(f"{url}: {type(exc).__name__}: {exc}")
    raise RuntimeError(" | ".join(errors))


def download_and_validate_archive(
    spec: BinanceArchiveSpec,
    month: str,
    *,
    raw_dir: Path,
    coverage: dict[str, Any],
    max_archive_bytes: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    archive_key = spec.archive_key(month)
    checksum_key = spec.checksum_key(month)
    primary_archive = f"{BINANCE_ARCHIVE_BASE_URL}/{archive_key}"
    fallback_archive = f"{BINANCE_ARCHIVE_FALLBACK_URL}/{archive_key}"
    primary_checksum = f"{BINANCE_ARCHIVE_BASE_URL}/{checksum_key}"
    fallback_checksum = f"{BINANCE_ARCHIVE_FALLBACK_URL}/{checksum_key}"

    checksum_payload, _ = _fetch_with_fallback(
        (primary_checksum, fallback_checksum),
        max_bytes=4096,
        timeout_seconds=timeout_seconds,
    )
    checksum_text = checksum_payload.decode("utf-8")
    expected_digest = parse_checksum_text(checksum_text, spec.filename(month))
    archive_path = raw_dir / "binance" / spec.source_id / spec.filename(month)

    if archive_path.exists():
        try:
            verify_checksum(archive_path, checksum_text)
        except ValueError:
            archive_path.unlink()
    if not archive_path.exists():
        payload, _ = _fetch_with_fallback(
            (primary_archive, fallback_archive),
            max_bytes=max_archive_bytes,
            timeout_seconds=timeout_seconds,
        )
        _write_bytes_atomic(archive_path, payload)

    remote_digest, local_digest = verify_checksum(archive_path, checksum_text)
    frame = read_binance_kline_zip(archive_path)
    month_start, month_end = month_bounds(month)
    expected_start = month_start
    if month == coverage["first_month"]:
        expected_start = pd.Timestamp(coverage["launch_open_time"])
    validation = validate_hourly_klines(
        frame,
        expected_start=expected_start,
        expected_end_exclusive=month_end,
        cutoff=month_end,
    )
    expected_unit = "ms"
    if spec.market == "spot" and month_start >= BINANCE_SPOT_MICROSECOND_START:
        expected_unit = "us"
    if validation["open_time_unit"] != expected_unit:
        validation["status"] = "fail"
        validation["expected_timestamp_unit"] = expected_unit
    if validation["status"] != "pass":
        raise ValueError(
            f"Hourly validation failed for {spec.source_id} {month}: "
            f"{strict_json_dumps(validation).strip()}"
        )
    return {
        "source_id": spec.source_id,
        "month": month,
        "archive_key": archive_key,
        "archive_url": primary_archive,
        "fallback_archive_url": fallback_archive,
        "checksum_url": primary_checksum,
        "fallback_checksum_url": fallback_checksum,
        "remote_sha256": remote_digest,
        "local_sha256": local_digest,
        "expected_sha256": expected_digest,
        "local_path": relative_path(archive_path),
        "size_bytes": archive_path.stat().st_size,
        "cache_validation": "sha256_verified",
        "timestamp_unit": validation["open_time_unit"],
        "row_count": validation["row_count"],
        "first_open_time": validation["first_open_time"],
        "last_open_time": validation["last_open_time"],
        "validation_status": validation["status"],
    }


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _source_months(
    source_manifest: dict[str, Any],
) -> tuple[dict[str, list[str]], str]:
    if source_manifest.get("scope") != "offline_lead_signal_sources":
        raise ValueError("Unexpected lead-signal source manifest scope")
    coverage = source_manifest["sources"]["binance"]["coverage"]
    required = set(REQUIRED_STREAM_IDS)
    if set(coverage) != required:
        raise ValueError("Source manifest does not contain the four required streams")
    common_end = min(coverage[source_id]["last_month"] for source_id in required)
    months = {
        source_id: [
            month for month in coverage[source_id]["months"] if month <= common_end
        ]
        for source_id in REQUIRED_STREAM_IDS
    }
    return months, common_end


def download_all_archives(
    *,
    source_manifest: dict[str, Any],
    raw_dir: Path,
    workers: int,
    max_archive_bytes: int,
    timeout_seconds: float,
) -> list[dict[str, Any]]:
    months_by_source, common_end = _source_months(source_manifest)
    coverage = source_manifest["sources"]["binance"]["coverage"]
    tasks: list[tuple[BinanceArchiveSpec, str]] = []
    for spec in BINANCE_ARCHIVE_SPECS:
        tasks.extend((spec, month) for month in months_by_source[spec.source_id])
    log(f"[binance] validating {len(tasks)} archives through common month {common_end}")

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_and_validate_archive,
                spec,
                month,
                raw_dir=raw_dir,
                coverage=coverage[spec.source_id],
                max_archive_bytes=max_archive_bytes,
                timeout_seconds=timeout_seconds,
            ): (spec.source_id, month)
            for spec, month in tasks
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            source_id, month = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:
                raise RuntimeError(
                    f"Archive failed for {source_id} {month}: {exc}"
                ) from exc
            if completed % 50 == 0 or completed == len(futures):
                log(f"[binance] {completed}/{len(futures)} archives complete")
    return sorted(records, key=lambda item: (item["source_id"], item["month"]))


def load_hourly_streams(
    records: list[dict[str, Any]],
) -> dict[str, pd.DataFrame]:
    streams: dict[str, pd.DataFrame] = {}
    for source_id in REQUIRED_STREAM_IDS:
        source_records = [item for item in records if item["source_id"] == source_id]
        if not source_records:
            raise ValueError(f"No downloaded archives for {source_id}")
        frames = [
            read_binance_kline_zip(PROJECT_ROOT / item["local_path"])
            for item in source_records
        ]
        frame = pd.concat(frames, ignore_index=True)
        if frame["open_time"].duplicated().any():
            raise ValueError(f"Cross-archive duplicate timestamps for {source_id}")
        streams[source_id] = frame.sort_values("open_time").reset_index(drop=True)
    return streams


def load_defillama_sources(
    source_manifest: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], list[dict[str, Any]]]:
    source_records = source_manifest["sources"]["defillama"]["sources"]
    output: dict[str, pd.DataFrame] = {}
    evidence: list[dict[str, Any]] = []
    for name in ("stablecoins", "chain_tvl", "dex_volume"):
        record = source_records[name]
        path = PROJECT_ROOT / record["local_path"]
        observed_hash = sha256_file(path)
        if observed_hash != record["local_sha256"]:
            raise ValueError(f"DefiLlama source hash mismatch: {name}")
        output[name] = pd.read_csv(path)
        evidence.append(
            {
                "source": name,
                "local_path": record["local_path"],
                "local_sha256": observed_hash,
                "payload_sha256": record["payload_sha256"],
                "url": record["url"],
            }
        )
    return output, evidence


def _load_market_daily(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    frame = pd.read_csv(path)
    if "Date" in frame:
        frame = frame.rename(columns={"Date": "date"})
    return frame


def _feature_report(
    *,
    features: pd.DataFrame,
    groups: dict[str, list[str]],
    archive_records: list[dict[str, Any]],
    as_of_date: pd.Timestamp,
    output_sha256: str,
    manifest_sha256: str,
) -> dict[str, Any]:
    bar_columns = [
        f"{prefix}_bar_count"
        for prefix in ("eth_spot", "eth_perp", "btc_spot", "btc_perp")
    ]
    common_mask = features[bar_columns].eq(24).all(axis=1)
    common_dates = features.index[common_mask]
    common_start = pd.Timestamp(common_dates.min())
    common_end = pd.Timestamp(common_dates.max())
    authority_start = common_start + pd.Timedelta(days=730)
    conditions = [
        "DefiLlama historical publication vintages are unavailable; values are lagged one day and remain offline-only",
        "External source terms require review before production use",
        "OKX remains excluded",
        "August 2026 spot hourly data is excluded until its complete monthly archive is available",
    ]
    ready = (
        bool(len(common_dates) >= 730)
        and common_end == as_of_date
        and all(item["validation_status"] == "pass" for item in archive_records)
        and all(groups.values())
    )
    return {
        "schema_version": 1,
        "decision": "pass_for_pr3_offline_evaluation" if ready else "fail",
        "as_of_date": as_of_date.date().isoformat(),
        "feature_table": {
            "row_count": len(features),
            "column_count": len(features.columns),
            "first_date": pd.Timestamp(features.index.min()).date().isoformat(),
            "last_date": pd.Timestamp(features.index.max()).date().isoformat(),
            "sha256": output_sha256,
            "group_column_counts": {
                name: len(columns) for name, columns in groups.items()
            },
        },
        "common_hourly_coverage": {
            "row_count": len(common_dates),
            "first_date": common_start.date().isoformat(),
            "last_date": common_end.date().isoformat(),
            "earliest_authoritative_test_date": authority_start.date().isoformat(),
            "minimum_prior_days": 730,
        },
        "archive_validation": {
            "archive_count": len(archive_records),
            "all_passed": all(
                item["validation_status"] == "pass" for item in archive_records
            ),
            "download_bytes": sum(item["size_bytes"] for item in archive_records),
        },
        "target_contract": {
            "horizon_days": 3,
            "tail_threshold": 0.12,
            "direct": "tail_up_primary = 1[terminal_return_3d >= 0.12]",
            "factorized": "large_move_primary times direction_up with final-score recalibration",
            "multiclass": ["DOWN_TAIL", "NORMAL", "UP_TAIL"],
            "barrier_labels": "diagnostic_only",
        },
        "gate": {
            "offline_model_evaluation_approved": ready,
            "production_use_approved": False,
            "model_training_performed": False,
            "daily_forecast_modified": False,
            "public_contract_modified": False,
            "conditions": conditions,
        },
        "feature_manifest_sha256": manifest_sha256,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build leakage-safe daily lead-signal features from the approved PR 1 "
            "source manifest. This command performs no model training."
        )
    )
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
        default=PROJECT_ROOT / "lake" / "raw" / "lead_signal_full",
    )
    parser.add_argument(
        "--market-daily",
        type=Path,
        default=PROJECT_ROOT / "lake" / "gold" / "eth_master_daily.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "lake" / "gold" / "lead_signal_daily.csv.gz",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "lake" / "manifests" / "lead_signal_features.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT
        / "lake"
        / "reports"
        / "lead_signal_feature_readiness.json",
    )
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--max-archive-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument("--reporting-lag-days", type=int, default=1)
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Explicitly replace immutable feature manifest/report outputs",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 1 <= args.workers <= 8:
        raise ValueError("workers must be between 1 and 8")
    if not 0 < args.timeout_seconds <= 120:
        raise ValueError("timeout-seconds must be in (0, 120]")
    if args.reporting_lag_days < 1:
        raise ValueError("reporting-lag-days must be at least one")

    source_manifest = _read_json(args.source_manifest)
    source_readiness = _read_json(args.source_readiness)
    if source_readiness.get("decision") != "pass_for_pr2_offline":
        raise RuntimeError("PR 1 source readiness does not approve PR 2")
    source_manifest_hash = hashlib.sha256(args.source_manifest.read_bytes()).hexdigest()
    if source_readiness.get("manifest_sha256") != source_manifest_hash:
        raise ValueError("Source readiness and manifest SHA-256 disagree")

    archive_records = download_all_archives(
        source_manifest=source_manifest,
        raw_dir=args.raw_dir,
        workers=args.workers,
        max_archive_bytes=args.max_archive_bytes,
        timeout_seconds=args.timeout_seconds,
    )
    streams = load_hourly_streams(archive_records)
    defillama, defillama_evidence = load_defillama_sources(source_manifest)
    _, common_end_month = _source_months(source_manifest)
    _, month_end_exclusive = month_bounds(common_end_month)
    as_of_date = month_end_exclusive - pd.Timedelta(days=1)

    log(f"[features] aggregating complete UTC days through {as_of_date.date()}")
    features = build_lead_signal_daily_features(
        streams=streams,
        stablecoins=defillama["stablecoins"],
        tvl=defillama["chain_tvl"],
        dex_volume=defillama["dex_volume"],
        as_of_date=as_of_date,
        market_daily=_load_market_daily(args.market_daily),
        reporting_lag_days=args.reporting_lag_days,
    )
    output_sha256 = write_daily_csv(features, args.output)
    groups = feature_group_columns(features)
    manifest = {
        "schema_version": 1,
        "source_manifest_generated_at": source_manifest["generated_at"],
        "scope": "offline_lead_signal_daily_features",
        "immutable": True,
        "as_of_date": as_of_date.date().isoformat(),
        "cutoff_utc": month_end_exclusive.isoformat(),
        "availability_contract": {
            "binance": "UTC day d uses only 24 hourly bars closed by d+1 00:00 UTC",
            "defillama": (
                f"source day d is first eligible on feature day d+{args.reporting_lag_days}; "
                "historical publication vintages unavailable"
            ),
            "forecast_row": "feature date d is available at d+1 00:00 UTC",
        },
        "source_manifest": {
            "path": relative_path(args.source_manifest),
            "sha256": source_manifest_hash,
        },
        "binance_archives": archive_records,
        "defillama_inputs": defillama_evidence,
        "feature_table": {
            "path": relative_path(args.output),
            "sha256": output_sha256,
            "row_count": len(features),
            "column_count": len(features.columns),
            "first_date": pd.Timestamp(features.index.min()).date().isoformat(),
            "last_date": pd.Timestamp(features.index.max()).date().isoformat(),
            "feature_groups": groups,
            "diagnostic_columns": [
                column for column in features if column.endswith("_bar_count")
            ],
        },
        "preprocessing_contract": {
            "daily_changes": [1, 3, 7],
            "fold_local_scaling": True,
            "fold_local_imputation": True,
            "minimum_source_history_before_authoritative_test_days": 730,
            "partial_current_day_allowed": False,
        },
        "gate": {
            "production_use_approved": False,
            "model_training_performed": False,
            "daily_forecast_modified": False,
            "public_contract_modified": False,
        },
    }
    manifest_sha256 = write_immutable_json(
        args.manifest_output,
        manifest,
        replace=args.replace_outputs,
    )
    report = _feature_report(
        features=features,
        groups=groups,
        archive_records=archive_records,
        as_of_date=as_of_date,
        output_sha256=output_sha256,
        manifest_sha256=manifest_sha256,
    )
    write_immutable_json(
        args.report_output,
        report,
        replace=args.replace_outputs,
    )
    log(f"[output] daily features: {relative_path(args.output)}")
    log(f"[output] feature manifest: {relative_path(args.manifest_output)}")
    log(f"[output] readiness: {relative_path(args.report_output)}")
    log(strict_json_dumps(report["gate"]).strip())
    return 0 if report["decision"] == "pass_for_pr3_offline_evaluation" else 2


if __name__ == "__main__":
    raise SystemExit(main())
