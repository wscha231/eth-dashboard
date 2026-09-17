from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

BASE_URL = "https://ethsupply.fyi"
TRACKED_URL = BASE_URL + "/api/tracked"
LIVE_URL = BASE_URL + "/api/live"
SOURCE_ID = "ethsupply_fyi"
SOURCE_NAME = "Ethereum Supply / ethsupply.fyi"
LICENSE_STATUS = "rights_unreviewed_public_api"
PUBLIC_REDISTRIBUTION = "blocked_pending_rights_review"
MODEL_USE = "blocked_until_rights_prospective_continuity_and_self_derived_history_pass"
SITE_USE = "blocked"
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
FETCH_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError)
WEI_PER_ETH = Decimal(10) ** 18

TRACKED_FIELDS = (
    "issuanceWei",
    "burnWei",
    "baseFeeBurnWei",
    "blobBaseFeeBurnWei",
    "consensusPenaltiesWei",
    "otherExecutionBurnWei",
    "netWei",
)


def _receipt_iso(value: Any | None) -> str:
    if value is None:
        dt = datetime.now(timezone.utc)
    elif isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    elif isinstance(value, str):
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(timezone.utc)
    else:
        raise TypeError("receipt_time must be datetime, ISO string, or None")
    return dt.isoformat()


def _unix_iso(value: Any) -> str:
    integer = _integer(value)
    if integer is None:
        return ""
    return datetime.fromtimestamp(integer, tz=timezone.utc).isoformat()


def _selected_headers(response) -> dict[str, str]:
    wanted = {"etag", "x-supply-sha256", "x-supply-generated-at", "x-supply-revision"}
    return {key.lower(): value for key, value in response.headers.items() if key.lower() in wanted}


def _validate_final_url(value: str) -> None:
    parsed = urlparse(value)
    if parsed.scheme != "https" or parsed.hostname != "ethsupply.fyi":
        raise ValueError("unexpected ethsupply response origin")
    if parsed.path not in {"/api/live", "/api/tracked"}:
        raise ValueError("unexpected ethsupply response path")


def fetch_json(
    url: str,
    *,
    timeout: int = 35,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> tuple[dict[str, Any], bytes, dict[str, str]]:
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
            header_revision = selected.get("x-supply-revision", "")
            if header_revision and payload.get("revision") is not None and int(header_revision) != int(payload["revision"]):
                raise ValueError("ethsupply header/body revision mismatch")
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
    return format(Decimal(value) / WEI_PER_ETH, "f")


def _snake(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def _data_value(node: Any, *, signed: bool = False) -> tuple[int | None, str, str, str]:
    if not isinstance(node, dict):
        return None, "unavailable", "unknown", ""
    value = _integer(node.get("value"), signed=signed)
    return value, str(node.get("status", "unavailable")), str(node.get("kind", "unknown")), str(node.get("asOf") or "")


def parse_tracked(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schemaVersion") != 1 or payload.get("range") != "retained":
        raise ValueError("unexpected ethsupply tracked schema")
    revision = int(payload["revision"])
    generated_at = _unix_iso(payload["generatedAt"])
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("ethsupply tracked summary missing")
    from_slot = _integer(summary.get("fromSlot"))
    to_slot = _integer(summary.get("toSlot"))
    if from_slot is None or to_slot is None or to_slot < from_slot:
        raise ValueError("invalid tracked slot range")

    row: dict[str, Any] = {
        "source_revision": revision,
        "source_generated_at": generated_at,
        "from_slot": from_slot,
        "to_slot": to_slot,
        "from_timestamp": _integer(summary.get("fromTimestamp")),
        "to_timestamp": _integer(summary.get("toTimestamp")),
        "from_time_utc": _unix_iso(summary.get("fromTimestamp")),
        "to_time_utc": _unix_iso(summary.get("toTimestamp")),
        "slots": _integer(summary.get("slots")),
        "blocks": _integer(summary.get("blocks")),
        "finalized_to_slot": "" if payload.get("finalizedToSlot") is None else int(payload["finalizedToSlot"]),
        "live_tail_slots": "" if payload.get("liveTailSlots") is None else int(payload["liveTailSlots"]),
        "warnings": len(payload.get("warnings") or []),
    }
    values: dict[str, int | None] = {}
    for field in TRACKED_FIELDS:
        parsed = _integer(summary.get(field), signed=field == "netWei")
        values[field] = parsed
        snake = _snake(field)
        row[snake] = "" if parsed is None else str(parsed)
        row[snake.removesuffix("_wei") + "_eth"] = _eth(parsed)
    issuance, burn, net = values["issuanceWei"], values["burnWei"], values["netWei"]
    row["identity_error_wei"] = "" if None in {issuance, burn, net} else str(net - (issuance - burn))
    return row


def parse_live(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schemaVersion") != 1 or payload.get("chainId") != 1:
        raise ValueError("unexpected ethsupply live schema")
    revision = int(payload["revision"])
    generated_at = _unix_iso(payload["generatedAt"])
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
    return {
        "source_revision": revision,
        "source_generated_at": generated_at,
        "head_block": head.get("block"),
        "head_slot": head.get("slot"),
        "head_timestamp": head.get("timestamp"),
        "head_time_utc": _unix_iso(head.get("timestamp")),
        "finalized_block": finalized.get("block"),
        "finalized_epoch": finalized.get("epoch"),
        "finalized_slot": finalized.get("slot"),
        "supply_asof_block": as_of.get("block"),
        "supply_asof_slot": as_of.get("slot"),
        "supply_asof_epoch": as_of.get("epoch"),
        "supply_asof_timestamp": as_of.get("timestamp"),
        "supply_asof_time_utc": _unix_iso(as_of.get("timestamp")),
        "total_supply_wei": str(total_wei),
        "total_supply_eth": _eth(total_wei),
        "total_supply_status": total_status,
        "total_supply_kind": total_kind,
        "total_supply_asof": total_asof,
        "finalized_supply_wei": "" if finalized_wei is None else str(finalized_wei),
        "finalized_supply_eth": _eth(finalized_wei),
        "finalized_supply_status": finalized_status,
        "finalized_supply_kind": finalized_kind,
        "finalized_supply_asof": finalized_asof,
        "accounting_interval": str(accounting.get("interval") or ""),
        "accounting_epoch": accounting.get("epoch"),
        "accounting_target_epoch": accounting.get("targetEpoch"),
        "issuance_wei": "" if issuance_wei is None else str(issuance_wei),
        "issuance_eth": _eth(issuance_wei),
        "issuance_status": issuance_status,
        "issuance_kind": issuance_kind,
        "issuance_asof": issuance_asof,
        "burn_wei": "" if burn_wei is None else str(burn_wei),
        "burn_eth": _eth(burn_wei),
        "burn_status": burn_status,
        "burn_kind": burn_kind,
        "burn_asof": burn_asof,
        "net_wei": "" if net_wei is None else str(net_wei),
        "net_eth": _eth(net_wei),
        "net_status": net_status,
        "net_kind": net_kind,
        "net_asof": net_asof,
        "identity_error_wei": identity_error,
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() and path.stat().st_size else {}


def _append_jsonl(path: Path, row: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_receipt_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_state(root: str | Path, *, receipt_time: Any | None = None) -> dict[str, Any]:
    base = Path(root)
    raw_dir = base / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    receipt = _receipt_iso(receipt_time)

    tracked_payload, tracked_raw, tracked_headers = fetch_json(TRACKED_URL)
    live_payload, live_raw, live_headers = fetch_json(LIVE_URL)

    tracked_hash = tracked_headers["sha256"]
    live_hash = live_headers["sha256"]
    (raw_dir / "ethsupply_tracked_latest.json").write_bytes(tracked_raw)
    (raw_dir / "ethsupply_live_latest.json").write_bytes(live_raw)

    tracked = parse_tracked(tracked_payload)
    live = parse_live(live_payload)

    state_path = base / "state.json"
    report_path = base / "report.json"
    tracked_path = base / "eth_supply_tracked_receipts.jsonl"
    live_path = base / "eth_supply_live_receipts.jsonl"
    state = _read_json(state_path)
    if not state.get("collector_started_at"):
        state.update(schema=1, source_id=SOURCE_ID, collector_started_at=receipt, license_status=LICENSE_STATUS)

    tracked_receipt = {
        "received_at": receipt,
        "available_at": receipt,
        "raw_sha256": tracked_hash,
        "header_revision": tracked_headers.get("x-supply-revision", ""),
        "header_generated_at": tracked_headers.get("x-supply-generated-at", ""),
        **tracked,
    }
    live_receipt = {
        "received_at": receipt,
        "available_at": receipt,
        "raw_sha256": live_hash,
        "header_revision": live_headers.get("x-supply-revision", ""),
        "header_generated_at": live_headers.get("x-supply-generated-at", ""),
        **live,
    }
    tracked_count = _append_jsonl(tracked_path, tracked_receipt)
    live_count = _append_jsonl(live_path, live_receipt)

    # Human-inspectable current receipt tables; JSONL remains the append-only source of truth.
    _write_receipt_csv(base / "eth_supply_tracked_receipts.csv", _read_jsonl(tracked_path))
    _write_receipt_csv(base / "eth_supply_live_receipts.csv", _read_jsonl(live_path))

    state.update(
        last_received_at=receipt,
        last_tracked_revision=tracked["source_revision"],
        last_live_revision=live["source_revision"],
        last_tracked_generated_at=tracked["source_generated_at"],
        last_live_generated_at=live["source_generated_at"],
        last_tracked_raw_sha256=tracked_hash,
        last_live_raw_sha256=live_hash,
        tracked_receipt_rows=tracked_count,
        live_receipt_rows=live_count,
    )
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")

    report = {
        "schema": 3,
        "generated_at_utc": receipt,
        "status": "research_only_rights_unreviewed",
        "source_id": SOURCE_ID,
        "source_name": SOURCE_NAME,
        "tracked_url": TRACKED_URL,
        "live_url": LIVE_URL,
        "license_status": LICENSE_STATUS,
        "public_redistribution": PUBLIC_REDISTRIBUTION,
        "model_use": MODEL_USE,
        "site_use": SITE_USE,
        "history_mode": "prospective_receipts_only",
        "historical_backfill": "not_eligible_requires_self_derived_geth_consensus_replay",
        "available_at_policy": "actual EtherForecast collector receipt only",
        "tracked_sha256": tracked_hash,
        "live_sha256": live_hash,
        "source_tracked_revision": tracked["source_revision"],
        "source_live_revision": live["source_revision"],
        "tracked_receipt_rows": tracked_count,
        "live_receipt_rows": live_count,
        "tracked_from_slot": tracked["from_slot"],
        "tracked_to_slot": tracked["to_slot"],
        "tracked_from_time_utc": tracked["from_time_utc"],
        "tracked_to_time_utc": tracked["to_time_utc"],
        "tracked_identity_error_wei": tracked["identity_error_wei"],
        "live_identity_error_wei": live["identity_error_wei"],
        "live_total_supply_eth": live["total_supply_eth"],
        "live_accounting_epoch": live["accounting_epoch"],
        "live_accounting_interval": live["accounting_interval"],
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
