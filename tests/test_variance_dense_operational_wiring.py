from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_dense_shadow_reuses_existing_hourly_workflow_without_new_schedule():
    workflow = (ROOT / ".github/workflows/event_hourly.yml").read_text()
    assert 'cron: "8 * * * *"' in workflow
    assert "run_variance_dense_shadow.py --root lake/signals" in workflow
    assert "variance-shadow-dense/issued.db" in workflow
    assert "lake/event-ledger/variance-shadow/active.json" in workflow
    assert workflow.count("schedule:") == 1


def test_dense_state_is_persisted_separately_from_legacy_variance_ledger():
    persistence = (ROOT / "scripts/persist_event_ledger.sh").read_text()
    assert "variance_shadow_dense" in persistence
    assert "variance-shadow-dense" in persistence
    assert "variance-shadow/issued.db" not in persistence


def test_dense_module_has_no_refit_or_6h_issue_path():
    module = (ROOT / "signal_pipeline/variance_dense_shadow.py").read_text()
    assert "HORIZONS = (24, 72)" in module
    assert "fit_checkpoint(" not in module
    assert "obtain_checkpoint(" not in module
    assert "late_or_reconstructed_issue_allowed\": False" in module
    assert "all-clock gate failed; UTC 08-16 failed" in module
