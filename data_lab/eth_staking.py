from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal
import json
from pathlib import Path
from typing import Any

SOURCE_ID = "ethsupply_fyi_staking"
SOURCE_NAME = "Ethereum staking / validator state from ethsupply.fyi live receipt"
LICENSE_STATUS = "rights_unreviewed_public_api"
PUBLIC_REDISTRIBUTION = "blocked_pending_rights_review"
MODEL_USE = "blocked_until_rights_prospective_continuity_and_preregistered_ablation_pass"
SITE_USE = "blocked"
GWEI_PER_ETH = Decimal(10) ** 9

# No post-result feature engineering in the source collector. These are source-native
# state variables only; ratios/deltas belong in a later preregistered MODEL experiment.
FIELD_PATHS = {
    "active_balance_gwei": ("activeBalanceGwei",),
    "effective_balance_gwei": ("effectiveBalanceGwei",),
    "active_validators": ("activeValidators",),
    "pending_deposits_validators": ("pendingDeposits", "validators"),
    "pending_deposits_balance_gwei": ("pendingDeposits", "balanceGwei"),
    "pending_deposits_slot_zero_balance_gwei": ("pendingDeposits", "slotZeroBalanceGwei"),
    "scheduled_activations_validators": ("scheduledActivations", "validators"),
    "scheduled_activations_balance_gwei": ("scheduledActivations", "balanceGwei"),
    "scheduled_exits_validators": ("scheduledExits", "validators"),
    "scheduled_exits_balance_gwei": ("scheduledExits", "balanceGwei"),
    "pending_partial_withdrawals_requests": ("pendingPartialWithdrawals", "requests"),
    "pending_partial_withdrawals_balance_gwei": ("pendingPartialWithdrawals", "balanceGwei"),
    "pending_consolidations_requests": ("pendingConsolidations", "requests"),
    "pending_consolidations_balance_gwei": ("pendingConsolidations", "balanceGwei"),
    "churn_activation_exit_gwei": ("churn", "activationExitGwei"),
    "churn_consolidation_gwei": ("churn", "consolidationGwei"),
}

BALANCE_FIELDS = {name for name in FIELD_PATHS if name.endswith("_gwei")}
COUNT_FIELDS = set(FIELD_PATHS) - BALANCE_FIELDS


def _integer(value: Any) -> int:
    text = str(value)
    if not text or not text.isdigit():
        raise ValueError("staking values must be non-negative base-10 integers")
    return int(text)


def _unix_iso(value: Any) -> str:
    return datetime.fromtimestamp(_integer(value), tz=timezone.utc).isoformat()


def _node(root: dict[str, Any], path: tuple[str, ...]) -> dict[str, Any]:
    value: Any = root
    for key in path:
        if not isinstance(value, dict) or key not in value:
            raise ValueError("missing staking field: " + ".".join(path))
        value = value[key]
    if not isinstance(value, dict):
        raise ValueError("staking metric node must be an object: " + ".".join(path))
    return value


def _metric(node: dict[str, Any]) -> tuple[int, str, str, int, str]:
    value = _integer(node.get("value"))
    status = str(node.get("status") or "")
    kind = str(node.get("kind") or "")
    as_of = _integer(node.get("asOf"))
    sources = node.get("sources")
    if status not in {"exact", "approximate"}:
        raise ValueError("staking metric must expose exact/approximate status")
    if kind not in {"observed", "derived"}:
        raise ValueError("staking metric must expose observed/derived kind")
    if not isinstance(sources, list) or not sources or any(not isinstance(x, str) or not x for x in sources):
        raise ValueError("staking metric must expose non-empty source provenance")
    return value, status, kind, as_of, json.dumps(sources, separators=(",", ":"))


def parse_live_staking(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schemaVersion") != 1 or payload.get("chainId") != 1:
        raise ValueError("unexpected ethsupply live schema")
    staking = payload.get("staking")
    if not isinstance(staking, dict):
        raise ValueError("live staking object missing")

    row: dict[str, Any] = {
        "source_revision": int(payload["revision"]),
        "source_generated_at": _unix_iso(payload["generatedAt"]),
        "head_slot": _integer((payload.get("head") or {}).get("slot")),
        "finalized_slot": _integer((payload.get("finalized") or {}).get("slot")),
    }
    as_of_values: list[int] = []
    for name, path in FIELD_PATHS.items():
        value, status, kind, as_of, sources = _metric(_node(staking, path))
        row[name] = str(value)
        row[name + "_status"] = status
        row[name + "_kind"] = kind
        row[name + "_as_of"] = _unix_iso(as_of)
        row[name + "_sources"] = sources
        as_of_values.append(as_of)
        if name in BALANCE_FIELDS:
            row[name.removesuffix("_gwei") + "_eth"] = format(Decimal(value) / GWEI_PER_ETH, "f")

    # Cross-field sanity only; do not impose economic/model assumptions.
    if _integer(row["effective_balance_gwei"]) > _integer(row["active_balance_gwei"]):
        raise ValueError("effective staking balance exceeds active balance")
    if _integer(row["active_validators"]) <= 0:
        raise ValueError("active validator count must be positive")
    row["staking_as_of_max"] = _unix_iso(max(as_of_values))
    row["staking_as_of_min"] = _unix_iso(min(as_of_values))
    return row


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def append_live_receipt(
    root: str | Path,
    payload: dict[str, Any],
    *,
    received_at: str,
    raw_sha256: str,
    header_revision: str = "",
    header_generated_at: str = "",
) -> dict[str, Any]:
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    parsed = parse_live_staking(payload)
    row = {
        "received_at": received_at,
        "available_at": received_at,
        "raw_sha256": raw_sha256,
        "header_revision": header_revision,
        "header_generated_at": header_generated_at,
        **parsed,
    }
    ledger = base / "eth_staking_live_receipts.jsonl"
    with ledger.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    rows = _read_jsonl(ledger)
    _write_csv(base / "eth_staking_live_receipts.csv", rows)

    report = {
        "schema": 1,
        "generated_at_utc": received_at,
        "status": "research_only_rights_unreviewed",
        "source_id": SOURCE_ID,
        "source_name": SOURCE_NAME,
        "license_status": LICENSE_STATUS,
        "public_redistribution": PUBLIC_REDISTRIBUTION,
        "model_use": MODEL_USE,
        "site_use": SITE_USE,
        "history_mode": "prospective_receipts_only",
        "available_at_policy": "same actual EtherForecast /api/live receipt used by ETH supply collector",
        "live_receipt_rows": len(rows),
        "raw_sha256": raw_sha256,
        "source_revision": parsed["source_revision"],
        "active_validators": parsed["active_validators"],
        "active_balance_eth": parsed["active_balance_eth"],
        "effective_balance_eth": parsed["effective_balance_eth"],
        "pending_deposits_validators": parsed["pending_deposits_validators"],
        "scheduled_activations_validators": parsed["scheduled_activations_validators"],
        "scheduled_exits_validators": parsed["scheduled_exits_validators"],
        "pending_partial_withdrawals_requests": parsed["pending_partial_withdrawals_requests"],
        "pending_consolidations_requests": parsed["pending_consolidations_requests"],
        "staking_as_of_min": parsed["staking_as_of_min"],
        "staking_as_of_max": parsed["staking_as_of_max"],
    }
    (base / "staking_report.json").write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report
