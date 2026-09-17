#!/usr/bin/env python3
"""Fail-closed audit for R3 prospective research-source PIT and rights boundaries."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = {
    "id", "family", "collection_state", "workflow", "artifact", "drive_stream",
    "source_contract", "history_mode", "availability_policy", "pit_status",
    "rights_status", "public_redistribution", "model_use", "site_use",
    "paid_product_use", "historical_alpha_eligibility", "prospective_alpha_eligibility",
}
RESEARCH_STATES = {"research_active", "research_paused", "blocked"}


def _valid_research_state(value: object) -> bool:
    """Accept active/paused states or an explicitly fail-closed blocked reason.

    Detailed blocked states such as ``blocked_missing_authorized_rpc`` remain
    safer and more informative than collapsing every gate failure to the bare
    ``blocked`` token. Arbitrary non-blocked states are still rejected.
    """
    state = str(value or "")
    return state in RESEARCH_STATES or state.startswith("blocked_")


def load(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not isinstance(data.get("sources"), list):
        raise ValueError("invalid R3 research source registry")
    ids = [str(row.get("id", "")) for row in data["sources"]]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError("R3 source ids must be non-empty and unique")
    return data


def audit(registry_path: str | Path, *, root: str | Path = ".") -> dict:
    registry_path = Path(registry_path)
    root = Path(root)
    data = load(registry_path)
    errors: list[str] = []
    rows: list[dict] = []

    production_ids: set[str] = set()
    production_registry = root / "config/data_source_registry.json"
    if production_registry.is_file():
        prod = json.loads(production_registry.read_text(encoding="utf-8"))
        production_ids = {str(row.get("id")) for row in prod.get("sources", [])}

    for source in data["sources"]:
        source_id = str(source.get("id", "<missing>"))
        missing = sorted(REQUIRED - set(source))
        if missing:
            errors.append(f"{source_id}: missing fields {','.join(missing)}")
            continue
        if not _valid_research_state(source["collection_state"]):
            errors.append(f"{source_id}: invalid research collection_state")
        if source_id in production_ids:
            errors.append(f"{source_id}: research id collides with production registry")
        if source["public_redistribution"] != "blocked":
            errors.append(f"{source_id}: public redistribution must remain blocked")
        if source["site_use"] != "blocked":
            errors.append(f"{source_id}: site use must remain blocked")
        if source["paid_product_use"] != "blocked":
            errors.append(f"{source_id}: paid-product use must remain blocked")
        if not str(source["model_use"]).startswith("blocked_"):
            errors.append(f"{source_id}: MODEL use must remain explicitly blocked")
        if not str(source["prospective_alpha_eligibility"]).startswith("blocked_"):
            errors.append(f"{source_id}: prospective alpha eligibility must be blocked")
        if "prospective_receipts_only" in str(source["history_mode"]) and not str(source["historical_alpha_eligibility"]).startswith("blocked_"):
            errors.append(f"{source_id}: prospective-only source cannot claim historical alpha eligibility")

        workflow = root / ".github/workflows" / str(source["workflow"])
        contract = root / str(source["source_contract"])
        if not workflow.is_file():
            errors.append(f"{source_id}: workflow not found: {source['workflow']}")
        if not contract.is_file():
            errors.append(f"{source_id}: source contract not found: {source['source_contract']}")

        rows.append({
            "id": source_id,
            "family": source["family"],
            "collection_state": source["collection_state"],
            "history_mode": source["history_mode"],
            "pit_status": source["pit_status"],
            "rights_status": source["rights_status"],
            "public_redistribution": source["public_redistribution"],
            "model_use": source["model_use"],
            "site_use": source["site_use"],
            "paid_product_use": source["paid_product_use"],
            "drive_stream": source["drive_stream"],
        })

    return {
        "schema": 1,
        "status": "pass" if not errors else "fail",
        "production_effect": "none",
        "research_source_count": len(data["sources"]),
        "active_research_source_count": sum(row.get("collection_state") == "research_active" for row in data["sources"]),
        "all_model_site_public_paid_use_blocked": not any(
            row.get("public_redistribution") != "blocked"
            or row.get("site_use") != "blocked"
            or row.get("paid_product_use") != "blocked"
            or not str(row.get("model_use", "")).startswith("blocked_")
            for row in data["sources"]
        ),
        "errors": errors,
        "sources": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="config/r3_research_source_registry.json")
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    report = audit(args.registry, root=args.root)
    text = json.dumps(report, indent=2, sort_keys=True)
    print(text)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
    return 0 if report["status"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
