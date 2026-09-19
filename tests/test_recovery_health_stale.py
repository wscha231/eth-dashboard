import json

from scripts.recovery_health import record


def test_failed_stage_cannot_reuse_previous_run_success(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_RUN_ID', 'current-run')
    for name in ('dense_stage_status.json', 'failure_review.json'):
        (tmp_path / name).write_text(json.dumps({'run_id': 'older-run', 'status': 'success'}))
    (tmp_path / 'failure_review.md').write_text('old successful review')
    result = record(tmp_path, 'success', 'failure', 'failure')
    assert result['core_delivery_succeeded'] is True
    assert result['optional_degraded'] is True
    for name in ('dense_stage_status.json', 'failure_review.json'):
        row = json.loads((tmp_path / name).read_text())
        assert row['run_id'] == 'current-run'
        assert row['status'] == 'failed'
        assert 'without_current_run_report' in row['error']
    assert not (tmp_path / 'failure_review.md').exists()


def test_real_current_failure_report_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_RUN_ID', 'current-run')
    path = tmp_path / 'dense_stage_status.json'
    raw = json.dumps({'run_id': 'current-run', 'status': 'failed', 'errors': ['checkpoint checksum mismatch']})
    path.write_text(raw)
    record(tmp_path, 'success', 'failure', 'skipped')
    assert path.read_text() == raw


def test_missing_report_after_failed_stage_is_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv('GITHUB_RUN_ID', 'current-run')
    record(tmp_path, 'failure', 'failure', 'failure')
    for name in ('dense_stage_status.json', 'failure_review.json'):
        assert json.loads((tmp_path / name).read_text())['status'] == 'failed'
