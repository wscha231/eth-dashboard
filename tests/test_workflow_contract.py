from pathlib import Path


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_weekly_eval_publishes_failure_evidence_before_enforcing_gate() -> None:
    workflow = _read(".github/workflows/eth_model_eval.yml")

    assert "id: candidate_gate" in workflow
    assert "forecast_site/public/model_eval_latest.json" in workflow
    assert "forecast_site/public/model_eval_last_pass.json" in workflow
    assert "forecast_site/public/backtest_longrun_candidate_history.json" in workflow
    assert "Enforce candidate promotion gate" in workflow
    assert workflow.index(
        "Publish latest evidence and gated production artifacts"
    ) < workflow.index("Enforce candidate promotion gate")


def test_full_model_eval_label_dispatches_parallel_gate() -> None:
    workflow = _read(".github/workflows/eth_model_eval.yml")

    assert "types: [opened, synchronize, reopened, labeled]" in workflow
    assert workflow.count("full-model-eval") >= 3
    assert "matrix:" in workflow
    assert "horizon: [7, 30]" in workflow
    assert "chunk: [0, 1, 2, 3, 4, 5]" in workflow


def test_compact_h30_gate_is_focused_and_full_run_is_explicit() -> None:
    workflow = _read(".github/workflows/eth_model_eval.yml")

    generic_job = workflow.split("\n  evaluate:\n", maxsplit=1)[1].split(
        "\n  evaluate-full-horizon:\n", maxsplit=1
    )[0]
    compact_job = workflow.split("\n  evaluate-compact-h30:\n", maxsplit=1)[1].split(
        "\n  evaluate-full-gate:\n", maxsplit=1
    )[0]
    assert "github.event.inputs.compact_h30_full != 'true'" in generic_job
    assert "github.event_name == 'pull_request'" in compact_job
    assert "compact_h30_full" in compact_job
    assert 'ETH_ENABLE_COMPACT_H30_REGRESSOR: "1"' in compact_job
    assert "--horizons 30" in compact_job
    assert "--skip-classification" in compact_job
    assert "github.event.inputs.compact_h30_full" in compact_job


def test_daily_workflow_deploys_live_history_and_verifies_site() -> None:
    workflow = _read(".github/workflows/daily_forecast.yml")

    restore_master = workflow.split(
        "\n      - name: Restore deployed master data\n", maxsplit=1
    )[1].split("\n      - name: Refresh market data\n", maxsplit=1)[0]
    assert "origin/data/daily-forecast:lake/gold/eth_master_daily.csv" in restore_master

    assert "forecast_site/public/history.json" in workflow
    assert "forecast_site/public/model_eval_latest.json" in workflow
    assert "Verify deployed freshness" in workflow
    assert "scripts/check_site_freshness.py" in workflow


def test_watchdog_will_not_dispatch_while_daily_run_is_active() -> None:
    workflow = _read(".github/workflows/site_freshness_watchdog.yml")

    assert "active_runs" in workflow
    assert "recovery dispatch skipped" in workflow
    assert "gh workflow run daily_forecast.yml" in workflow


def test_lead_signal_source_preflight_is_bounded_and_isolated() -> None:
    workflow = _read(".github/workflows/lead_signal_source_preflight.yml")

    assert "timeout-minutes: 10" in workflow
    assert "--max-binance-archives 12" in workflow
    assert 'artifact_dir="$RUNNER_TEMP/lead-signal-preflight"' in workflow
    assert "tests/test_lead_signal_data.py" in workflow
    assert "scripts/backfill_lead_signals.py" in workflow
    assert "eth_data_collector.py" not in workflow
    assert "daily_forecast.yml" not in workflow


def test_lead_signal_feature_contract_is_offline_and_time_bounded() -> None:
    workflow = _read(".github/workflows/lead_signal_feature_contract.yml")

    assert "timeout-minutes: 30" in workflow
    assert "scripts/build_lead_signal_features.py" in workflow
    assert "tests/test_lead_signals.py" in workflow
    assert "tests/test_tail_events.py" in workflow
    assert "--workers 8" in workflow
    assert "--replace-outputs" in workflow
    assert "Record committed evidence hashes" in workflow
    assert "Verify deterministic rebuild" in workflow
    assert "lake/gold/lead_signal_daily.csv.gz" in workflow
    assert "lake/manifests/lead_signal_features.json" in workflow
    assert "lake/reports/lead_signal_feature_readiness.json" in workflow
    assert "eth_data_collector.py" not in workflow
    assert "eth_price_forecast.py" not in workflow
    assert "daily_forecast.yml" not in workflow


def test_lead_signal_ablation_workflow_is_offline_and_gate_b_is_manual() -> None:
    workflow = _read(".github/workflows/lead_signal_source_ablation.yml")

    assert "timeout-minutes: 35" in workflow
    assert "scripts/evaluate_lead_signal_ablation.py" in workflow
    assert "origin/data/daily-forecast" in workflow
    assert "default: smoke" in workflow
    assert "- full" in workflow
    assert "continue-on-error: true" in workflow
    assert '.gate.infrastructure_status == "PASS"' in workflow
    assert "eth_data_collector.py" not in workflow
    assert "daily_forecast.yml" not in workflow
    assert "forecast_site/public" not in workflow
