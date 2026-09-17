from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import pandas as pd

BASE_URL = "https://ethsupply.fyi"
HISTORY_URL = BASE_URL + "/api/history?range=retained"
LIVE_URL = BASE_URL + "/api/live"
SOURCE_ID = "ethsupply_fyi"
SOURCE_NAME = "Ethereum Supply / ethsupply.fyi"
LICENSE_STATUS = "rights_unreviewed_public_api"
PUBLIC_REDISTRIBUTION = "blocked_pending_rights_review"
MODEL_USE = "blocked_until_rights_pit_coverage_and_independent_reproduction_pass"
SITE_USE = "blocked"
MAX_RESPONSE_BYTES = 32 * 1024 * 1024
FETCH_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError)
WEI_PER_ETH = Decimal(10) ** 18

HISTORY_FIELDS = (
    "supplyWei",
    "issuanceWei",
    "burnWei",
    "baseFeeBurnWei",
    "blobBaseFeeBurnWei",
    "consensusPenaltiesWei",
    "otherExecutionBurnWei",
    "netWei",
)


def utc(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    if pd.isna(stamp):
        raise ValueError("finite timestamp required")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _selected_headers(response) -> dict[str, str]:
    wanted = {"etag", "x-supply-sha256", "x-supply-generated-at", "x-supply-revision"}
    return {k.lower(): v for k, v in response.headers.items() if k.lower() in wanted}


def _validate_final_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != "ethsupply.fyi":
        raise ValueError("unexpected ethsupply response origin")
    if parsed.path not in {"/api/live", "/api/history"}:
        raise ValueError("unexpected ethsupply response path")


def fetch_json(url: str, *, timeout: int = 35, retries: int = 3, backoff_seconds: float = 1.5) -> tuple[dict[str, Any], bytes, dict[str, str]]:
    _validate_final_url(url)
    headers = {
        "User-Agent": "EtherForecast research collector/1.0 (+https://etherforecast.live)",
        "Accept": "application/json",
    }
    last: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                _validate_final_url(response.geturl())
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                selected = _selected_headers(response)
            if len(raw) > MAX_RESPONSE_BYTES:
                raise ValueError("oversized ethsupply response")
            digest = hashlib.sha256(raw).hexdigest()
            declared = selected.get("x-supply-sha256", "").lower().removeprefix("sha256:")
            if declared and declared != digest:
                raise ValueError("ethsupply response SHA256 mismatch")
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError("ethsupply response is not a JSON object")
            return payload, raw, {**selected, "sha256": digest}
        except FETCH_ERRORS as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= retries - 1:
                break
            time.sleep(backoff_seconds * (attempt + 1))
    assert last is not None
    raise last


def _integer(value: Any, *, signed: bool = False) -> int | None:
    if value is None or value == "":
        return None
    text = str(value)
    pattern = r"-?(?:0|[1-9]\d*)" if signed else r"(?:0|[1-9]\d*)"
    if not re.fullmatch(pattern, text):
        raise ValueError(f"invalid {'signed ' if signed else ''}integer value")
    return int(text)


def _eth(value: int | None) -> str:
    if value is None:
        return ""
    amount = Decimal(value) / WEI_PER_ETH
    return format(amount, "f")


def _data_value(node: Any, *, signed: bool = False) -> tuple[int | None, str, str, str]:
    if not isinstance(node, dict):
        return None, "unavailable", "unknown", ""
    value = _integer(node.get("value"), signed=signed)
    return value, str(node.get("status", "unavailable")), str(node.get("kind", "unknown")), str(node.get("asOf") or "")


def parse_history(payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    if payload.get("schemaVersion") != 1 or payload.get("range") != "retained":
        raise ValueError("unexpected ethsupply retained-history schema")
    revision = int(payload.get("revision"))
    generated_at = utc(pd.to_datetime(int(payload.get("generatedAt")), unit="s", utc=True))
    epochs = payload.get("epochs")
    if not isinstance(epochs, list) or not epochs:
        raise ValueError("ethsupply retained history has no epochs")

    rows: list[dict[str, Any]] = []
    for point in epochs:
        if not isinstance(point, dict):
            raise ValueError("invalid ethsupply history point")
        timestamp = pd.to_datetime(int(point["timestamp"]), unit="s", utc=True)
        row: dict[str, Any] = {
            "timestamp": timestamp.isoformat(),
            "epoch": int(point["epoch"]),
            "attestation_duty_epoch": "" if point.get("attestationDutyEpoch") is None else int(point["attestationDutyEpoch"]),
            "blocks": "" if point.get("blocks") is None else int(point["blocks"]),
        }
        values: dict[str, int | None] = {}
        for field in HISTORY_FIELDS:
            signed = field == "netWei"
            parsed = _integer(point.get(field), signed=signed)
            values[field] = parsed
            snake = re.sub(r"(?<!^)(?=[A-Z])", "_", field).lower()
            row[snake] = "" if parsed is None else str(parsed)
            row[snake.removesuffix("_wei") + "_eth"] = _eth(parsed)
        if values["issuanceWei"] is not None and values["burnWei"] is not None and values["netWei"] is not None:
            row["identity_error_wei"] = str(values["netWei"] - (values["issuanceWei"] - values["burnWei"]))
        else:
            row["identity_error_wei"] = ""
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values(["timestamp", "epoch"], kind="stable").reset_index(drop=True)
    if frame["timestamp"].duplicated().any():
        raise ValueError("duplicate ethsupply history timestamp")
    timestamps = pd.to_datetime(frame["timestamp"], utc=True)
    if timestamps.max() > generated_at + pd.Timedelta(minutes=10):
        raise ValueError("history contains observations after source generation time")
    identity = pd.to_numeric(frame["identity_error_wei"], errors="coerce").dropna()
    meta = {
        "schema_version": 1,
        "source_revision": revision,
        "source_generated_at": generated_at.isoformat(),
        "source_interval": str(payload.get("interval") or "unknown"),
        "history_rows": int(len(frame)),
        "first_timestamp": timestamps.min().isoformat(),
        "last_timestamp": timestamps.max().isoformat(),
        "identity_checked_rows": int(len(identity)),
        "identity_nonzero_rows": int((identity != 0).sum()),
        "partial": bool(payload.get("partial", False)),
    }
    return frame, meta


def parse_live(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schemaVersion") != 1 or payload.get("chainId") != 1:
        raise ValueError("unexpected ethsupply live schema")
    revision = int(payload["revision"])
    generated_at = pd.to_datetime(int(payload["generatedAt"]), unit="s", utc=True)
    supply = payload.get("supply") or {}
    accounting = payload.get("accounting") or {}
    issuance = accounting.get("issuance") or {}
    burn = accounting.get("burn") or {}

    total_wei, total_status, total_kind, total_asof = _data_value(supply.get("totalWei"))
    finalized_wei, finalized_status, finalized_kind, finalized_asof = _data_value(supply.get("finalizedTotalWei"))
    issuance_wei, issuance_status, issuance_kind, issuance_asof = _data_value(issuance.get("totalWei"))
    burn_wei, burn_status, burn_kind, burn_asof = _data_value(burn.get("totalWei"))
    net_wei, net_status, net_kind, net_asof = _data_value(accounting.get("netWei"), signed=True)
    if total_wei is None:
        raise ValueError("live total supply is unavailable")
    identity_error = ""
    if issuance_wei is not None and burn_wei is not None and net_wei is not None:
        identity_error = str(net_wei - (issuance_wei - burn_wei))

    head = payload.get("head") or {}
    finalized = payload.get("finalized") or {}
    as_of = supply.get("asOf") or {}
    row = {
        "source_revision": revision,
        "source_generated_at": generated_at.isoformat(),
        "head_block": head.get("block"), "head_slot": head.get("slot"), "head_timestamp": head.get("timestamp"),
        "finalized_block": finalized.get("block"), "finalized_epoch": finalized.get("epoch"), "finalized_slot": finalized.get("slot"),
        "supply_asof_block": as_of.get("block"), "supply_asof_slot": as_of.get("slot"), "supply_asof_epoch": as_of.get("epoch"), "supply_asof_timestamp": as_of.get("timestamp"),
        "total_supply_wei": str(total_wei), "total_supply_eth": _eth(total_wei), "total_supply_status": total_status, "total_supply_kind": total_kind, "total_supply_asof": total_asof,
        "finalized_supply_wei": "" if finalized_wei is None else str(finalized_wei), "finalized_supply_eth": _eth(finalized_wei), "finalized_supply_status": finalized_status, "finalized_supply_kind": finalized_kind, "finalized_supply_asof": finalized_asof,
        "accounting_interval": str(accounting.get("interval") or ""), "accounting_epoch": accounting.get("epoch"), "accounting_target_epoch": accounting.get("targetEpoch"),
        "issuance_wei": "" if issuance_wei is None else str(issuance_wei), "issuance_eth": _eth(issuance_wei), "issuance_status": issuance_status, "issuance_kind": issuance_kind, "issuance_asof": issuance_asof,
        "burn_wei": "" if burn_wei is None else str(burn_wei), "burn_eth": _eth(burn_wei), "burn_status": burn_status, "burn_kind": burn_kind, "burn_asof": burn_asof,
        "net_wei": "" if net_wei is None else str(net_wei), "net_eth": _eth(net_wei), "net_status": net_status, "net_kind": net_kind, "net_asof": net_asof,
        "identity_error_wei": identity_error,
    }
    return row


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False) if path.exists() and path.stat().st_size else pd.DataFrame()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() and path.stat().st_size else {}


def _same(a: Any, b: Any) -> bool:
    left = "" if a is None else str(a)
    right = "" if b is None else str(b)
    return left == right


def reconcile_history(
    snapshot: pd.DataFrame, *, existing: pd.DataFrame | None, state: dict[str, Any] | None,
    receipt_time: Any, raw_sha256: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, int]]:
    receipt = utc(receipt_time)
    state = dict(state or {})
    initial = not state.get("collector_started_at")
    if initial:
        state.update(schema=1, source_id=SOURCE_ID, collector_started_at=receipt.isoformat(), license_status=LICENSE_STATUS)
    started = utc(state["collector_started_at"])

    key_columns = [c for c in snapshot.columns if c not in {"available_at", "ingested_at", "revision", "vintage_mode", "raw_sha256"}]
    prior: dict[str, dict[str, Any]] = {}
    if existing is not None and not existing.empty:
        prior = {str(row["timestamp"]): row.to_dict() for _, row in existing.iterrows()}
    output = {k: dict(v) for k, v in prior.items()}
    journal: list[dict[str, Any]] = []
    stats = {"new_rows": 0, "revised_rows": 0, "unchanged_rows": 0}

    for _, incoming_row in snapshot.iterrows():
        incoming = incoming_row.to_dict()
        key = str(incoming["timestamp"])
        observed_at = utc(key)
        old = output.get(key)
        if old is None:
            prospective = (not initial) and observed_at >= started
            merged = {
                **incoming,
                "available_at": receipt.isoformat(),
                "ingested_at": receipt.isoformat(),
                "revision": "0",
                "vintage_mode": "observed_vintages" if prospective else "reconstructed",
                "raw_sha256": raw_sha256,
            }
            output[key] = merged
            journal.append({"change_type": "new", **merged})
            stats["new_rows"] += 1
            continue
        changed = any(not _same(old.get(column), incoming.get(column)) for column in key_columns)
        if not changed:
            stats["unchanged_rows"] += 1
            continue
        revision = int(old.get("revision") or 0) + 1
        merged = {
            **old, **incoming,
            "available_at": receipt.isoformat(),
            "ingested_at": receipt.isoformat(),
            "revision": str(revision),
            "vintage_mode": "observed_vintages",
            "raw_sha256": raw_sha256,
        }
        output[key] = merged
        journal.append({"change_type": "revision", **merged})
        stats["revised_rows"] += 1

    columns = list(snapshot.columns) + ["available_at", "ingested_at", "revision", "vintage_mode", "raw_sha256"]
    latest = pd.DataFrame([output[key] for key in sorted(output, key=lambda x: utc(x))])
    latest = latest.reindex(columns=columns)
    events = pd.DataFrame(journal)
    state.update(last_received_at=receipt.isoformat(), last_raw_sha256=raw_sha256, rows=int(len(latest)))
    return latest, events, state, stats


def write_state(root: str | Path, *, receipt_time: Any | None = None) -> dict[str, Any]:
    base = Path(root)
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt = utc(receipt_time or datetime.now(timezone.utc))

    history_payload, history_raw, history_headers = fetch_json(HISTORY_URL)
    live_payload, live_raw, live_headers = fetch_json(LIVE_URL)
    history, history_meta = parse_history(history_payload)
    live = parse_live(live_payload)
    history_hash = history_headers["sha256"]
    live_hash = live_headers["sha256"]

    latest_path = base / "eth_supply_history_intervals.csv"
    journal_path = base / "eth_supply_history_receipts.csv"
    live_path = base / "eth_supply_live_receipts.csv"
    state_path = base / "state.json"
    report_path = base / "report.json"

    latest, events, state, stats = reconcile_history(
        history,
        existing=_read_csv(latest_path),
        state=_read_json(state_path),
        receipt_time=receipt,
        raw_sha256=history_hash,
    )
    latest.to_csv(latest_path, index=False)
    journal = pd.concat([_read_csv(journal_path), events], ignore_index=True, sort=False)
    journal.to_csv(journal_path, index=False)

    live_record = {
        "received_at": receipt.isoformat(),
        "raw_sha256": live_hash,
        "header_revision": live_headers.get("x-supply-revision", ""),
        "header_generated_at": live_headers.get("x-supply-generated-at", ""),
        **live,
    }
    live_receipts = pd.concat([_read_csv(live_path), pd.DataFrame([live_record])], ignore_index=True, sort=False)
    live_receipts.to_csv(live_path, index=False)

    state.update(
        last_history_revision=history_meta["source_revision"],
        last_live_revision=live["source_revision"],
        last_history_generated_at=history_meta["source_generated_at"],
        last_live_generated_at=live["source_generated_at"],
        last_live_raw_sha256=live_hash,
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    (raw_dir / "ethsupply_history_latest.json").write_bytes(history_raw)
    (raw_dir / "ethsupply_live_latest.json").write_bytes(live_raw)

    identity = pd.to_numeric(latest["identity_error_wei"], errors="coerce").dropna()
    report = {
        "schema": 1,
        "generated_at_utc": receipt.isoformat(),
        "status": "research_only_rights_unreviewed",
        "source_id": SOURCE_ID,
        "source_name": SOURCE_NAME,
        "history_url": HISTORY_URL,
        "live_url": LIVE_URL,
        "license_status": LICENSE_STATUS,
        "public_redistribution": PUBLIC_REDISTRIBUTION,
        "model_use": MODEL_USE,
        "site_use": SITE_USE,
        "historical_vintage_mode": "reconstructed_until_first_receipt_then_observed_vintages",
        "historical_available_at_policy": "actual EtherForecast collector receipt; no retroactive availability inference",
        "prospective_available_at_policy": "actual EtherForecast collector receipt",
        "history_sha256": history_hash,
        "live_sha256": live_hash,
        "source_history_revision": history_meta["source_revision"],
        "source_live_revision": live["source_revision"],
        "source_interval": history_meta["source_interval"],
        "history_rows": int(len(latest)),
        "history_receipt_rows": int(len(journal)),
        "live_receipt_rows": int(len(live_receipts)),
        "first_timestamp": str(latest["timestamp"].min()),
        "last_timestamp": str(latest["timestamp"].max()),
        "identity_checked_rows": int(len(identity)),
        "identity_nonzero_rows": int((identity != 0).sum()),
        "live_identity_error_wei": live["identity_error_wei"],
        "live_total_supply_eth": live["total_supply_eth"],
        **stats,
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
