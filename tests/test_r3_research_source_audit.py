import json
from pathlib import Path

from scripts.audit_r3_research_sources import audit


ROOT = Path(__file__).resolve().parents[1]


def test_repository_r3_registry_is_research_only_and_fail_closed():
    report = audit(ROOT / "config/r3_research_source_registry.json", root=ROOT)
    assert report["status"] == "pass", report["errors"]
    assert report["production_effect"] == "none"
    assert report["research_source_count"] == 9
    assert report["active_research_source_count"] == 8
    assert report["all_model_site_public_paid_use_blocked"] is True
    ids = {row["id"] for row in report["sources"]}
    assert ids == {
        "eth_etf_flows_farside",
        "eth_supply_ethsupply_fyi",
        "eth_staking_ethsupply_fyi_shared_live",
        "bybit_ethusdt_all_liquidation",
        "r3_p1_bitget_derivatives_history",
        "r3_p1_bitget_derivatives_fast",
        "r3_p1_hyperliquid_funding_history",
        "r3_p1_hyperliquid_derivatives_fast",
        "r3_p1_deribit_eth_dvol_history",
    }


def test_prospective_only_sources_remain_historically_blocked():
    registry = json.loads((ROOT / "config/r3_research_source_registry.json").read_text())
    prospective = [row for row in registry["sources"] if row["history_mode"] == "prospective_receipts_only"]
    assert prospective
    for row in prospective:
        assert row["historical_alpha_eligibility"].startswith("blocked_")
        assert row["prospective_alpha_eligibility"].startswith("blocked_")
        assert row["public_redistribution"] == "blocked"
        assert row["site_use"] == "blocked"
        assert row["paid_product_use"] == "blocked"
        assert row["model_use"].startswith("blocked_")


def test_p1_vendor_derivatives_are_fail_closed_pending_rights_and_private_storage():
    registry = json.loads((ROOT / "config/r3_research_source_registry.json").read_text())
    rows = [row for row in registry["sources"] if row["id"].startswith("r3_p1_")]
    assert len(rows) == 5
    for row in rows:
        assert row["public_redistribution"] == "blocked"
        assert row["site_use"] == "blocked"
        assert row["paid_product_use"] == "blocked"
        assert row["model_use"].startswith("blocked_")
        assert row["historical_alpha_eligibility"].startswith("blocked_")
        assert row["prospective_alpha_eligibility"].startswith("blocked_")
        assert "public_data_branch" in row["storage_exposure"]

    deribit = next(row for row in rows if row["id"] == "r3_p1_deribit_eth_dvol_history")
    assert deribit["collection_state"] == "blocked"
    assert "written_approval" in deribit["rights_status"]


def test_production_registry_does_not_misclassify_r3_ids_as_active_sources():
    production = json.loads((ROOT / "config/data_source_registry.json").read_text())
    research = json.loads((ROOT / "config/r3_research_source_registry.json").read_text())
    production_ids = {row["id"] for row in production["sources"]}
    research_ids = {row["id"] for row in research["sources"]}
    assert production_ids.isdisjoint(research_ids)
    # Broad roadmap placeholders can remain planned until rights/model-facing
    # integration is separately approved; research collection state lives only
    # in the R3 governance registry.
    planned = {row["id"] for row in production["sources"] if row["state"] == "planned"}
    assert {"liquidations_history", "eth_onchain_supply_staking", "eth_etf_flows"} <= planned
