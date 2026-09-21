from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hourly_producer_owns_timely_00utc_variance_issuance():
    workflow=(ROOT/'.github/workflows/event_hourly.yml').read_text()
    assert workflow.count('schedule:') == 1
    assert 'cron: "8 * * * *"' in workflow
    assert workflow.index('id: publication') < workflow.index('id: variance_daily')
    assert workflow.index('id: variance_daily') < workflow.index('id: availability_shadow')
    step=workflow.split('id: variance_daily',1)[1].split('      - name:',1)[0]
    assert 'continue-on-error: true' in step
    assert 'timeout-minutes: 2' in step
    assert 'run_hourly_variance_shadow.sh' in step
    assert "steps.variance_daily.outcome == 'failure'" in workflow


def test_hourly_variance_wrapper_never_refits_or_backfills():
    script=(ROOT/'scripts/run_hourly_variance_shadow.sh').read_text()
    assert '--no-refit --require-current-issue' in script
    assert 'data/event-ledger:refs/remotes/origin/data/event-ledger' in script
    assert 'lake/event-ledger/variance-shadow' in script
    assert 'persist_variance_ledger.sh' in script
    assert 'date -u +%H' in script and 'date -u +%M' in script
    assert '"00"' in script and '"55"' not in script  # shell compares minute -ge 55, not a new schedule
    assert 'minute" -ge 55' in script
    assert 'not_due' in script


def test_dedicated_variance_workflow_remains_refit_research_not_removed():
    workflow=(ROOT/'.github/workflows/variance_shadow.yml').read_text()
    assert '15,30,45 0 * * *' in workflow
    assert 'run_variance_shadow.py --root lake/signals' in workflow
    assert 'persist_variance_ledger.sh' in workflow
