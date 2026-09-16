import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_etf_registry_is_active_research_only_and_private():
    registry = json.loads((ROOT / "config/data_source_registry.json").read_text())
    row = next(item for item in registry["sources"] if item["id"] == "eth_etf_flows")
    assert row["state"] == "active"
    assert row["path"] == "lake/raw/vendor/eth_etf_farside_daily.csv"
    assert row["timestamp_column"] == "date"
    assert row["workflow"] == "daily_forecast.yml"
    assert row["pit_status"] == "reconstructed_d_plus_1_12utc_then_actual_receipt_versions"
    assert row["storage_stream"] == "daily-data"
    assert row["rights_status"] == "unreviewed_no_public_redistribution"
    assert row["public_redistribution_approved"] is False
    assert row["production_use_approved"] is False
    assert row["model_consumer_enabled"] is False
    assert row["public_git_persistence"] is False


def test_daily_workflow_restores_private_versions_and_fails_closed_without_them():
    workflow = (ROOT / ".github/workflows/daily_forecast.yml").read_text()
    assert "Restore private research-only ETF receipts from Drive" in workflow
    assert "gdrive_store.py restore --stream daily-data --target drive-private-research" in workflow
    assert "eth_etf_farside_versions.csv" in workflow
    assert "refusing to re-bootstrap historical availability" in workflow
    assert "collect_eth_etf_flows.py" in workflow
    assert "lake/reports/eth_etf_flows.json" in workflow


def test_etf_numeric_rows_are_not_added_to_model_extras_or_public_worktree():
    workflow = (ROOT / ".github/workflows/daily_forecast.yml").read_text()
    extras_section = workflow.split("Incremental market collection", 1)[1].split("Settle outstanding", 1)[0]
    assert "eth_etf_farside" not in extras_section
    # Public Git persistence is mediated by daily_source_state.py; there is no direct ETF CSV copy.
    public_section = workflow.split("Persist fresh sources for the active hybrid pipeline", 1)[1]
    assert "cp lake/raw/vendor/eth_etf_farside" not in public_section
