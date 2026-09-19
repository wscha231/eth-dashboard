from pathlib import Path
import hashlib
import json
import sqlite3
import sys
from types import SimpleNamespace

from scripts import run_optional_dense as staged

ROOT = Path(__file__).resolve().parents[1]


def test_dense_shadow_reuses_existing_hourly_workflow_without_new_schedule():
    workflow = (ROOT / ".github/workflows/event_hourly.yml").read_text()
    assert 'cron: "8 * * * *"' in workflow
    assert "run_optional_dense.py --root lake/signals" in workflow
    assert staged.REF == "refs/remotes/origin/data/event-ledger"
    assert staged.BASE_PATH == "lake/event-ledger/variance-shadow"
    assert staged.DENSE_PATH == "lake/event-ledger/variance-shadow-dense"
    assert workflow.index("id: publication") < workflow.index("id: dense_variance")
    assert workflow.count("schedule:") == 1


def test_staged_wrapper_calls_the_original_dense_runner_with_a_bounded_workspace(tmp_path, monkeypatch):
    calls = []
    def execute(args, **kwargs):
        calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stderr="")
    monkeypatch.setattr(staged.subprocess, "run", execute)
    staged.execute_dense(tmp_path)
    args, options = calls[0]
    assert args == [sys.executable, str(staged.REPO / "scripts/run_variance_dense_shadow.py"),
                    "--root", str(tmp_path)]
    assert options["timeout"] == 120
    assert options["cwd"] == staged.REPO


def test_staged_restore_loads_existing_trusted_cohort_and_exact_checkpoint_bytes(tmp_path, monkeypatch):
    prior = tmp_path / "original.db"
    with sqlite3.connect(prior) as db:
        db.execute("CREATE TABLE sentinel(value TEXT)")
        db.execute("INSERT INTO sentinel VALUES ('existing cohort')")
    checkpoint_bytes = {h: json.dumps({"horizon_hours": h}).encode() for h in (24, 72)}
    manifest = {"models": {str(h): {"file": f"h{h}.json",
                    "sha256": hashlib.sha256(checkpoint_bytes[h]).hexdigest()} for h in (24,72)}}
    expected = {staged.BASE_PATH + "/active.json": json.dumps(manifest).encode(),
                staged.DENSE_PATH + "/issued.db": prior.read_bytes()}
    expected.update({staged.BASE_PATH + f"/models/h{h}.json": raw for h, raw in checkpoint_bytes.items()})
    reads = []
    def read(path):
        reads.append(path)
        return expected[path]
    monkeypatch.setattr(staged, "git_bytes", read)
    stage = tmp_path / "stage"
    stage.mkdir()
    staged.restore_git(stage)
    assert set(reads) == set(expected)
    with sqlite3.connect(stage / "prior_dense.db") as db:
        assert db.execute("SELECT value FROM sentinel").fetchone()[0] == "existing cohort"
    for h, raw in checkpoint_bytes.items():
        assert (stage / f"variance_shadow/models/h{h}.json").read_bytes() == raw


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
