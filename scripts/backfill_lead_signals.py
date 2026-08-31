from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlencode

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecasting.lead_signal_data import (
    BINANCE_ARCHIVE_BASE_URL,
    BINANCE_ARCHIVE_FALLBACK_URL,
    BINANCE_ARCHIVE_SPECS,
    BINANCE_S3_LIST_URL,
    BINANCE_SPOT_MICROSECOND_START,
    BinanceArchiveSpec,
    fetch_json,
    fetch_url_bytes,
    month_bounds,
    parse_binance_s3_listing,
    parse_checksum_text,
    parse_defillama_chain_tvl,
    parse_defillama_dex_volume,
    parse_defillama_stablecoins,
    parse_deribit_funding_history,
    parse_okx_funding_history,
    read_binance_kline_zip,
    select_binance_probe_months,
    sha256_bytes,
    strict_json_dumps,
    summarize_binance_listing,
    summarize_funding_intervals,
    validate_daily_history,
    validate_hourly_klines,
    verify_checksum,
    write_daily_csv,
    write_immutable_json,
)

DEFILLAMA_SOURCES = {
    "stablecoins": {
        "url": "https://stablecoins.llama.fi/stablecoincharts/Ethereum",
        "filename": "defillama_ethereum_stablecoins_daily.csv.gz",
        "parser": parse_defillama_stablecoins,
    },
    "chain_tvl": {
        "url": "https://api.llama.fi/v2/historicalChainTvl/Ethereum",
        "filename": "defillama_ethereum_chain_tvl_daily.csv.gz",
        "parser": parse_defillama_chain_tvl,
    },
    "dex_volume": {
        "url": (
            "https://api.llama.fi/overview/dexs/ethereum"
            "?excludeTotalDataChart=false&excludeTotalDataChartBreakdown=true"
        ),
        "filename": "defillama_ethereum_dex_volume_daily.csv.gz",
        "parser": parse_defillama_dex_volume,
    },
}
OKX_API_BASE = "https://www.okx.com/api/v5"
DERIBIT_API_BASE = "https://www.deribit.com/api/v2"


def log(message: str) -> None:
    print(message, flush=True)


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def fetch_json_with_hash(
    url: str,
    *,
    max_bytes: int,
    timeout_seconds: float,
) -> tuple[Any, str]:
    payload = fetch_url_bytes(
        url,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
    )
    return json.loads(payload.decode("utf-8")), sha256_bytes(payload)


def fetch_binance_listing(
    spec: BinanceArchiveSpec,
    *,
    timeout_seconds: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    query = urlencode({"delimiter": "/", "prefix": spec.prefix}, quote_via=quote)
    url = f"{BINANCE_S3_LIST_URL}?{query}"
    payload = fetch_url_bytes(
        url,
        max_bytes=5 * 1024 * 1024,
        timeout_seconds=timeout_seconds,
    )
    items = parse_binance_s3_listing(payload, spec.prefix)
    summary = summarize_binance_listing(spec, items)
    summary["listing_url"] = url
    summary["listing_payload_sha256"] = sha256_bytes(payload)
    return summary, items


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def fetch_binance_archive(
    spec: BinanceArchiveSpec,
    month: str,
    *,
    raw_dir: Path,
    timeout_seconds: float,
    max_archive_bytes: int,
    coverage: dict[str, Any],
) -> dict[str, Any]:
    archive_key = spec.archive_key(month)
    checksum_key = spec.checksum_key(month)
    primary_archive_url = f"{BINANCE_ARCHIVE_BASE_URL}/{archive_key}"
    fallback_archive_url = f"{BINANCE_ARCHIVE_FALLBACK_URL}/{archive_key}"
    primary_checksum_url = f"{BINANCE_ARCHIVE_BASE_URL}/{checksum_key}"
    fallback_checksum_url = f"{BINANCE_ARCHIVE_FALLBACK_URL}/{checksum_key}"

    primary_checksum_payload = fetch_url_bytes(
        primary_checksum_url,
        max_bytes=4096,
        timeout_seconds=timeout_seconds,
    )
    fallback_checksum_payload = fetch_url_bytes(
        fallback_checksum_url,
        max_bytes=4096,
        timeout_seconds=timeout_seconds,
    )
    primary_checksum_text = primary_checksum_payload.decode("utf-8")
    fallback_checksum_text = fallback_checksum_payload.decode("utf-8")
    filename = spec.filename(month)
    primary_digest = parse_checksum_text(primary_checksum_text, filename)
    fallback_digest = parse_checksum_text(fallback_checksum_text, filename)
    if primary_digest != fallback_digest:
        raise ValueError(
            f"Primary/fallback checksum disagreement for {spec.source_id} {month}"
        )

    output_path = raw_dir / "binance" / spec.source_id / filename
    cache_status = "downloaded"
    resolved_archive_url = primary_archive_url
    if output_path.exists():
        try:
            verify_checksum(output_path, primary_checksum_text)
            cache_status = "reused_verified"
        except ValueError:
            output_path.unlink()

    if not output_path.exists():
        archive_errors: list[str] = []
        for url in (primary_archive_url, fallback_archive_url):
            try:
                archive_payload = fetch_url_bytes(
                    url,
                    max_bytes=max_archive_bytes,
                    timeout_seconds=timeout_seconds,
                )
                _write_bytes_atomic(output_path, archive_payload)
                resolved_archive_url = url
                break
            except Exception as exc:  # noqa: BLE001 - retain both route failures
                archive_errors.append(f"{url}: {type(exc).__name__}: {exc}")
        else:
            raise RuntimeError(" | ".join(archive_errors))

    remote_digest, local_digest = verify_checksum(output_path, primary_checksum_text)
    frame = read_binance_kline_zip(output_path)
    month_start, month_end = month_bounds(month)
    expected_start = month_start
    if month == coverage["first_month"]:
        expected_start = pd.Timestamp(frame["open_time"].min())
    validation = validate_hourly_klines(
        frame,
        expected_start=expected_start,
        expected_end_exclusive=month_end,
        cutoff=month_end,
    )
    expected_unit = "ms"
    if spec.market == "spot" and month_start >= BINANCE_SPOT_MICROSECOND_START:
        expected_unit = "us"
    unit_matches_contract = validation["open_time_unit"] == expected_unit
    validation["expected_timestamp_unit"] = expected_unit
    validation["timestamp_unit_matches_contract"] = unit_matches_contract
    if not unit_matches_contract:
        validation["status"] = "fail"

    return {
        "source_id": spec.source_id,
        "month": month,
        "archive_key": archive_key,
        "primary_archive_url": primary_archive_url,
        "fallback_archive_url": fallback_archive_url,
        "resolved_archive_url": resolved_archive_url,
        "primary_checksum_url": primary_checksum_url,
        "fallback_checksum_url": fallback_checksum_url,
        "primary_fallback_checksum_match": primary_digest == fallback_digest,
        "remote_sha256": remote_digest,
        "local_sha256": local_digest,
        "local_path": relative_path(output_path),
        "size_bytes": output_path.stat().st_size,
        "cache_status": cache_status,
        "validation": validation,
    }


def _select_bounded_binance_tasks(
    coverage_by_source: dict[str, dict[str, Any]],
    max_archives: int,
) -> list[tuple[BinanceArchiveSpec, str]]:
    if max_archives < len(BINANCE_ARCHIVE_SPECS):
        raise ValueError(f"max_archives must be at least {len(BINANCE_ARCHIVE_SPECS)}")
    selected_by_source = {
        spec.source_id: select_binance_probe_months(
            spec, coverage_by_source[spec.source_id]["months"]
        )
        for spec in BINANCE_ARCHIVE_SPECS
    }
    tasks: list[tuple[BinanceArchiveSpec, str]] = []

    # Every source receives a latest-month probe before optional boundary cases.
    for spec in BINANCE_ARCHIVE_SPECS:
        months = selected_by_source[spec.source_id]
        tasks.append((spec, months[-1]))
    for spec in BINANCE_ARCHIVE_SPECS:
        months = selected_by_source[spec.source_id]
        if months[0] != months[-1]:
            tasks.append((spec, months[0]))
    for spec in BINANCE_ARCHIVE_SPECS:
        for month in selected_by_source[spec.source_id]:
            task = (spec, month)
            if task not in tasks:
                tasks.append(task)
    return tasks[:max_archives]


def run_binance_preflight(args: argparse.Namespace, raw_dir: Path) -> dict[str, Any]:
    log("[binance] listing four official monthly archive prefixes")
    coverage_by_source: dict[str, dict[str, Any]] = {}
    listing_errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as executor:
        futures = {
            executor.submit(
                fetch_binance_listing,
                spec,
                timeout_seconds=args.timeout_seconds,
            ): spec
            for spec in BINANCE_ARCHIVE_SPECS
        }
        for future in as_completed(futures):
            spec = futures[future]
            try:
                summary, _ = future.result()
                coverage_by_source[spec.source_id] = summary
                log(
                    f"[binance] {spec.source_id}: {summary['first_month']}.."
                    f"{summary['last_month']} ({summary['archive_count']} months)"
                )
            except Exception as exc:  # noqa: BLE001 - source failure becomes evidence
                listing_errors[spec.source_id] = f"{type(exc).__name__}: {exc}"
                log(f"[binance] {spec.source_id}: listing failed: {exc}")

    if listing_errors:
        return {
            "status": "blocked",
            "coverage": coverage_by_source,
            "listing_errors": listing_errors,
            "archive_samples": [],
        }

    tasks = _select_bounded_binance_tasks(coverage_by_source, args.max_binance_archives)
    log(f"[binance] validating {len(tasks)} bounded archive samples")
    samples: list[dict[str, Any]] = []
    sample_errors: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                fetch_binance_archive,
                spec,
                month,
                raw_dir=raw_dir,
                timeout_seconds=args.timeout_seconds,
                max_archive_bytes=args.max_archive_bytes,
                coverage=coverage_by_source[spec.source_id],
            ): (spec, month)
            for spec, month in tasks
        }
        for future in as_completed(futures):
            spec, month = futures[future]
            try:
                result = future.result()
                samples.append(result)
                log(
                    f"[binance] {spec.source_id} {month}: "
                    f"{result['validation']['status']}"
                )
            except Exception as exc:  # noqa: BLE001 - source failure becomes evidence
                sample_errors.append(
                    {
                        "source_id": spec.source_id,
                        "month": month,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                log(f"[binance] {spec.source_id} {month}: failed: {exc}")

    samples.sort(key=lambda item: (item["source_id"], item["month"]))
    for spec in BINANCE_ARCHIVE_SPECS:
        coverage = coverage_by_source[spec.source_id]
        launch_sample = next(
            (
                sample
                for sample in samples
                if sample["source_id"] == spec.source_id
                and sample["month"] == coverage["first_month"]
            ),
            None,
        )
        launch_month_start, _ = month_bounds(coverage["first_month"])
        launch_open_time = (
            launch_sample["validation"]["first_open_time"] if launch_sample else None
        )
        coverage["launch_open_time"] = launch_open_time
        coverage["launch_month_is_partial"] = (
            pd.Timestamp(launch_open_time) > launch_month_start
            if launch_open_time is not None
            else None
        )
        coverage["expected_gap_policy"] = (
            "Hours before launch_open_time in the first listed month are expected; "
            "no other missing month or sampled hourly gap is allowed"
        )
    coverage_ready = all(
        not summary["missing_months"] and not summary["missing_checksum_keys"]
        for summary in coverage_by_source.values()
    )
    every_source_sampled = {sample["source_id"] for sample in samples} == {
        spec.source_id for spec in BINANCE_ARCHIVE_SPECS
    }
    samples_ready = (
        bool(samples)
        and not sample_errors
        and all(
            sample["validation"]["status"] == "pass"
            and sample["primary_fallback_checksum_match"]
            for sample in samples
        )
    )
    status = (
        "ready"
        if coverage_ready and every_source_sampled and samples_ready
        else "blocked"
    )
    return {
        "status": status,
        "coverage": coverage_by_source,
        "archive_samples": samples,
        "sample_errors": sample_errors,
        "route_policy": {
            "primary": BINANCE_ARCHIVE_BASE_URL,
            "fallback": BINANCE_ARCHIVE_FALLBACK_URL,
            "both_checksum_routes_required": True,
        },
        "terms_url": "https://data.binance.vision/terms-of-use.html",
        "production_terms_review": "required_before_non-research_use",
    }


def run_defillama_backfill(
    args: argparse.Namespace, source_table_dir: Path
) -> dict[str, Any]:
    log("[defillama] downloading full Ethereum stablecoin, TVL, and DEX histories")
    source_results: dict[str, dict[str, Any]] = {}

    def fetch_one(name: str, config: dict[str, Any]) -> tuple[str, Any, str]:
        payload, payload_hash = fetch_json_with_hash(
            config["url"],
            max_bytes=100 * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
        )
        return name, payload, payload_hash

    with ThreadPoolExecutor(max_workers=min(args.workers, 3)) as executor:
        futures = {
            executor.submit(fetch_one, name, config): (name, config)
            for name, config in DEFILLAMA_SOURCES.items()
        }
        for future in as_completed(futures):
            name, config = futures[future]
            try:
                _, payload, payload_hash = future.result()
                parser: Callable[..., pd.DataFrame] = config["parser"]
                frame = parser(
                    payload,
                    chain_slug="ethereum",
                    as_of_date=args.as_of_date,
                )
                validation = validate_daily_history(frame)
                output_path = source_table_dir / config["filename"]
                csv_hash = write_daily_csv(frame, output_path)
                source_results[name] = {
                    "status": (
                        "ready"
                        if validation["status"] == "pass"
                        and validation["row_count"] >= 730
                        else "insufficient"
                    ),
                    "url": config["url"],
                    "payload_sha256": payload_hash,
                    "local_path": relative_path(output_path),
                    "local_sha256": csv_hash,
                    "validation": validation,
                }
                log(
                    f"[defillama] {name}: {source_results[name]['status']} "
                    f"({validation['row_count']} rows)"
                )
            except Exception as exc:  # noqa: BLE001 - source failure becomes evidence
                source_results[name] = {
                    "status": "blocked",
                    "url": config["url"],
                    "error": f"{type(exc).__name__}: {exc}",
                }
                log(f"[defillama] {name}: failed: {exc}")

    ready = len(source_results) == len(DEFILLAMA_SOURCES) and all(
        item["status"] == "ready" for item in source_results.values()
    )
    return {
        "status": "ready" if ready else "blocked",
        "sources": source_results,
        "license_review": "public_api_terms_review_before_production",
    }


def _okx_url(path: str, params: dict[str, Any]) -> str:
    return f"{OKX_API_BASE}{path}?{urlencode(params)}"


def run_okx_probe(args: argparse.Namespace) -> dict[str, Any]:
    log("[okx] probing ETH-USDT-SWAP contract and recent funding metadata")
    instrument_url = _okx_url(
        "/public/instruments", {"instType": "SWAP", "instId": "ETH-USDT-SWAP"}
    )
    current_url = _okx_url("/public/funding-rate", {"instId": "ETH-USDT-SWAP"})
    history_url = _okx_url(
        "/public/funding-rate-history", {"instId": "ETH-USDT-SWAP", "limit": 100}
    )
    static_metadata = {
        "historical_download_page": "https://www.okx.com/historical-data",
        "official_archive_coverage_start": "2022-03",
        "api_history_documented_limit": "three months",
        "formula_change_effective": "2026-06-01",
        "post_change_formula": "per-period rate includes 8/N interval factor",
        "supported_settlement_intervals_hours": [1, 2, 4, 8],
        "normalization_contract": "daily_equivalent = per_period_rate * (24 / interval_hours)",
        "l2_download": "deferred",
        "archive_automation": "not_approved_until_download_url_and_terms_are_stable",
        "production_terms_review": "required",
    }
    try:
        instrument_payload = fetch_json(
            instrument_url,
            max_bytes=5 * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
            retries=1,
        )
        current_payload = fetch_json(
            current_url,
            max_bytes=5 * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
            retries=1,
        )
        history_payload = fetch_json(
            history_url,
            max_bytes=10 * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
            retries=1,
        )
        if str(instrument_payload.get("code")) != "0":
            raise ValueError("OKX instrument endpoint did not return code 0")
        instruments = instrument_payload.get("data") or []
        if not instruments:
            raise ValueError("OKX instrument endpoint returned no ETH-USDT-SWAP")
        instrument = instruments[0]
        current_rows = current_payload.get("data") or []
        current = current_rows[0] if current_rows else {}
        funding = parse_okx_funding_history(history_payload)
        interval_summary = summarize_funding_intervals(funding)
        next_interval_hours: float | None = None
        try:
            next_interval_hours = (
                int(current["nextFundingTime"]) - int(current["fundingTime"])
            ) / 3_600_000.0
        except (KeyError, TypeError, ValueError):
            pass
        return {
            "status": "conditional",
            "reason": (
                "Recent public API is reachable, but full archive download URLs and "
                "production terms remain a review gate"
            ),
            "instrument_url": instrument_url,
            "current_funding_url": current_url,
            "history_url": history_url,
            "instrument": {
                key: instrument.get(key)
                for key in (
                    "instId",
                    "instType",
                    "ctType",
                    "ctVal",
                    "ctValCcy",
                    "settleCcy",
                    "state",
                    "listTime",
                )
            },
            "current_interval_hours": next_interval_hours,
            "history": {
                "row_count": len(funding),
                "first_timestamp": (
                    funding.index.min().isoformat() if not funding.empty else None
                ),
                "last_timestamp": (
                    funding.index.max().isoformat() if not funding.empty else None
                ),
                **interval_summary,
            },
            **static_metadata,
        }
    except Exception as exc:  # noqa: BLE001 - source failure becomes evidence
        log(f"[okx] probe blocked: {exc}")
        return {
            "status": "blocked_by_environment",
            "error": f"{type(exc).__name__}: {exc}",
            "instrument_url": instrument_url,
            "current_funding_url": current_url,
            "history_url": history_url,
            **static_metadata,
        }


def _deribit_url(path: str, params: dict[str, Any]) -> str:
    return f"{DERIBIT_API_BASE}{path}?{urlencode(params)}"


def run_deribit_probe(args: argparse.Namespace) -> dict[str, Any]:
    log("[deribit] probing instrument launch and historical funding windows")
    instrument_url = _deribit_url(
        "/public/get_instrument", {"instrument_name": "ETH-PERPETUAL"}
    )
    try:
        instrument_payload = fetch_json(
            instrument_url,
            max_bytes=5 * 1024 * 1024,
            timeout_seconds=args.timeout_seconds,
        )
        instrument = instrument_payload.get("result") or {}
        if not instrument:
            raise ValueError("Deribit instrument endpoint returned no result")

        recent_start = args.as_of_date - pd.Timedelta(days=2)
        probe_windows = (
            ("near_launch", pd.Timestamp("2019-03-15"), pd.Timestamp("2019-03-17")),
            ("old_history", pd.Timestamp("2019-08-01"), pd.Timestamp("2019-08-03")),
            ("mid_history", pd.Timestamp("2023-01-01"), pd.Timestamp("2023-01-03")),
            ("recent", recent_start, args.as_of_date + pd.Timedelta(days=1)),
        )
        windows: dict[str, dict[str, Any]] = {}

        def fetch_window(
            label: str, start: pd.Timestamp, end: pd.Timestamp
        ) -> tuple[str, str, pd.DataFrame]:
            url = _deribit_url(
                "/public/get_funding_rate_history",
                {
                    "instrument_name": "ETH-PERPETUAL",
                    "start_timestamp": int(start.tz_localize("UTC").timestamp() * 1000),
                    "end_timestamp": int(end.tz_localize("UTC").timestamp() * 1000),
                },
            )
            payload = fetch_json(
                url,
                max_bytes=20 * 1024 * 1024,
                timeout_seconds=args.timeout_seconds,
            )
            return label, url, parse_deribit_funding_history(payload)

        with ThreadPoolExecutor(max_workers=min(args.workers, 4)) as executor:
            futures = {
                executor.submit(fetch_window, label, start, end): label
                for label, start, end in probe_windows
            }
            for future in as_completed(futures):
                label = futures[future]
                try:
                    _, url, frame = future.result()
                    windows[label] = {
                        "status": "available" if not frame.empty else "empty",
                        "url": url,
                        "row_count": len(frame),
                        "first_timestamp": (
                            frame.index.min().isoformat() if not frame.empty else None
                        ),
                        "last_timestamp": (
                            frame.index.max().isoformat() if not frame.empty else None
                        ),
                        **summarize_funding_intervals(frame),
                    }
                except Exception as exc:  # noqa: BLE001 - preserve per-window result
                    windows[label] = {
                        "status": "blocked",
                        "error": f"{type(exc).__name__}: {exc}",
                    }

        historical_ready = all(
            windows.get(label, {}).get("status") == "available"
            for label in ("old_history", "mid_history", "recent")
        )
        status = "feasible" if historical_ready else "blocked"
        log(f"[deribit] historical funding: {status}")
        return {
            "status": status,
            "instrument_url": instrument_url,
            "instrument": {
                key: instrument.get(key)
                for key in (
                    "instrument_name",
                    "creation_timestamp",
                    "settlement_period",
                    "is_active",
                    "base_currency",
                    "quote_currency",
                )
            },
            "funding_windows": windows,
            "finding": (
                "Public funding history is available well before the previously "
                "assumed 90-day window; a separate full continuity audit is warranted"
            ),
            "book_summary_history": "prospective_snapshot_only",
            "option_and_future_snapshot_fields": "not_backfillable_from_book_summary",
            "production_terms_review": "required",
        }
    except Exception as exc:  # noqa: BLE001 - source failure becomes evidence
        log(f"[deribit] probe blocked: {exc}")
        return {
            "status": "blocked_by_environment",
            "instrument_url": instrument_url,
            "error": f"{type(exc).__name__}: {exc}",
            "book_summary_history": "prospective_snapshot_only",
        }


def _compact_binance_coverage(payload: dict[str, Any]) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for source_id, item in payload.get("coverage", {}).items():
        compact[source_id] = {
            key: value for key, value in item.items() if key != "months"
        }
    return compact


def build_readiness_report(
    *,
    generated_at: str,
    as_of_date: pd.Timestamp,
    binance: dict[str, Any],
    defillama: dict[str, Any],
    okx: dict[str, Any],
    deribit: dict[str, Any],
    manifest_sha256: str,
) -> dict[str, Any]:
    primary_ready = (
        binance.get("status") == "ready" and defillama.get("status") == "ready"
    )
    blockers: list[str] = []
    conditions: list[str] = []
    if binance.get("status") != "ready":
        blockers.append("Binance spot/perpetual hourly archive preflight did not pass")
    if defillama.get("status") != "ready":
        blockers.append("DefiLlama full-history backfill did not pass")
    if okx.get("status") != "conditional":
        conditions.append("OKX is excluded until API/archive access is reproducible")
    else:
        conditions.append(
            "OKX remains research-only pending archive URL and terms review"
        )
    if deribit.get("status") != "feasible":
        conditions.append("Deribit historical funding remains optional")
    conditions.append("Every external source requires a production terms review")
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "as_of_date": as_of_date.date().isoformat(),
        "scope": "offline_lead_signal_source_feasibility",
        "manifest_sha256": manifest_sha256,
        "decision": "pass_for_pr2_offline" if primary_ready else "stop",
        "gate": {
            "offline_feature_work_approved": primary_ready,
            "production_use_approved": False,
            "model_training_performed": False,
            "daily_forecast_modified": False,
            "public_contract_modified": False,
            "blockers": blockers,
            "conditions": conditions,
        },
        "source_status": {
            "binance_hourly": binance.get("status"),
            "defillama_history": defillama.get("status"),
            "okx_funding": okx.get("status"),
            "deribit_funding": deribit.get("status"),
        },
        "binance_coverage": _compact_binance_coverage(binance),
        "binance_sample_count": len(binance.get("archive_samples", [])),
        "defillama": defillama,
        "okx": okx,
        "deribit": deribit,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline lead-signal source preflight. Downloads bounded Binance samples, "
            "backfills DefiLlama history, and audits OKX/Deribit feasibility."
        )
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "lake" / "raw" / "lead_signal",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=PROJECT_ROOT / "lake" / "manifests" / "lead_signal_sources.json",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=PROJECT_ROOT / "lake" / "reports" / "lead_signal_source_readiness.json",
    )
    parser.add_argument(
        "--source-table-dir",
        type=Path,
        default=PROJECT_ROOT / "lake" / "gold" / "lead_signal_sources",
        help="Tracked compact daily source tables; never read by the daily model in PR 1",
    )
    parser.add_argument(
        "--as-of-date",
        type=pd.Timestamp,
        default=pd.Timestamp.now(tz="UTC").tz_localize(None).floor("D")
        - pd.Timedelta(days=1),
        help="Last complete UTC date included in daily histories",
    )
    parser.add_argument("--timeout-seconds", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-binance-archives", type=int, default=12)
    parser.add_argument("--max-archive-bytes", type=int, default=25 * 1024 * 1024)
    parser.add_argument(
        "--replace-outputs",
        action="store_true",
        help="Explicitly replace an existing immutable manifest/report",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.as_of_date = pd.Timestamp(args.as_of_date)
    if args.as_of_date.tzinfo is not None:
        args.as_of_date = args.as_of_date.tz_convert("UTC").tz_localize(None)
    args.as_of_date = args.as_of_date.floor("D")
    if args.workers < 1 or args.workers > 8:
        raise ValueError("workers must be between 1 and 8")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 120:
        raise ValueError("timeout-seconds must be in (0, 120]")

    generated_at = datetime.now(timezone.utc).isoformat()
    binance = run_binance_preflight(args, args.raw_dir)
    defillama = run_defillama_backfill(args, args.source_table_dir)
    okx = run_okx_probe(args)
    deribit = run_deribit_probe(args)

    manifest = {
        "schema_version": 1,
        "generated_at": generated_at,
        "as_of_date": args.as_of_date.date().isoformat(),
        "scope": "offline_lead_signal_sources",
        "immutable": True,
        "generation": {
            "script": "scripts/backfill_lead_signals.py",
            "max_binance_archives": args.max_binance_archives,
            "max_archive_bytes": args.max_archive_bytes,
            "raw_dir": relative_path(args.raw_dir),
            "source_table_dir": relative_path(args.source_table_dir),
        },
        "sources": {
            "binance": binance,
            "defillama": defillama,
            "okx": okx,
            "deribit": deribit,
        },
    }
    manifest_sha256 = write_immutable_json(
        args.manifest_output,
        manifest,
        replace=args.replace_outputs,
    )
    report = build_readiness_report(
        generated_at=generated_at,
        as_of_date=args.as_of_date,
        binance=binance,
        defillama=defillama,
        okx=okx,
        deribit=deribit,
        manifest_sha256=manifest_sha256,
    )
    write_immutable_json(
        args.report_output,
        report,
        replace=args.replace_outputs,
    )
    log(f"[output] manifest: {relative_path(args.manifest_output)}")
    log(f"[output] readiness: {relative_path(args.report_output)}")
    log(strict_json_dumps(report["gate"]).strip())
    return 0 if report["decision"] == "pass_for_pr2_offline" else 2


if __name__ == "__main__":
    raise SystemExit(main())
