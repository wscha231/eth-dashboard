from pathlib import Path
import os
import subprocess

import pytest

from scripts.persist_data_branch_overlay import persist_overlay


def git(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=cwd, check=check, text=True, capture_output=True)


def make_repo(tmp_path: Path) -> tuple[Path, Path]:
    bare = tmp_path / "origin.git"
    git(tmp_path, "init", "--bare", str(bare))
    work = tmp_path / "work"
    git(tmp_path, "clone", str(bare), str(work))
    git(work, "config", "user.name", "test")
    git(work, "config", "user.email", "test@example.com")
    (work / "README.md").write_text("base\n", encoding="utf-8")
    git(work, "add", "README.md")
    git(work, "commit", "-m", "base")
    git(work, "branch", "-M", "main")
    git(work, "push", "-u", "origin", "main")

    git(work, "switch", "-c", "data/daily-forecast")
    (work / "lake/raw/vendor").mkdir(parents=True)
    (work / "lake/raw/vendor/unrelated.csv").write_text("date,x\n2026-01-01,1\n", encoding="utf-8")
    (work / "lake/raw/vendor/preserve_if_missing.csv").write_text("date,x\n2026-01-01,9\n", encoding="utf-8")
    git(work, "add", ".")
    git(work, "commit", "-m", "seed data branch")
    git(work, "push", "-u", "origin", "data/daily-forecast")
    git(work, "switch", "main")
    return bare, work


def show_remote(work: Path, path: str) -> str:
    git(work, "fetch", "origin", "data/daily-forecast")
    return git(work, "show", f"origin/data/daily-forecast:{path}").stdout


def test_overlay_preserves_unrelated_and_missing_target_files(tmp_path):
    _, work = make_repo(tmp_path)
    generated = work / "lake/raw/vendor/generated.csv"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text("date,x\n2026-09-16T00:00:00Z,2\n", encoding="utf-8")

    result = persist_overlay(
        work,
        branch="data/daily-forecast",
        message="persist generated",
        files=["lake/raw/vendor/generated.csv", "lake/raw/vendor/preserve_if_missing.csv"],
        retries=2,
        backoff_seconds=0,
    )

    assert result.status == "pushed"
    assert "2026-09-16T00:00:00Z,2" in show_remote(work, "lake/raw/vendor/generated.csv")
    assert "2026-01-01,1" in show_remote(work, "lake/raw/vendor/unrelated.csv")
    assert "2026-01-01,9" in show_remote(work, "lake/raw/vendor/preserve_if_missing.csv")


def test_identical_overlay_is_noop(tmp_path):
    _, work = make_repo(tmp_path)
    generated = work / "lake/reports/status.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text('{"status":"ok"}\n', encoding="utf-8")
    first = persist_overlay(
        work,
        branch="data/daily-forecast",
        message="persist status",
        files=["lake/reports/status.json"],
        retries=2,
        backoff_seconds=0,
    )
    second = persist_overlay(
        work,
        branch="data/daily-forecast",
        message="persist status",
        files=["lake/reports/status.json"],
        retries=2,
        backoff_seconds=0,
    )
    assert first.status == "pushed"
    assert second.status == "no_changes"


def test_push_rejection_is_retried(tmp_path):
    bare, work = make_repo(tmp_path)
    generated = work / "lake/reports/retry.json"
    generated.parent.mkdir(parents=True, exist_ok=True)
    generated.write_text('{"retry":true}\n', encoding="utf-8")

    hook = bare / "hooks/pre-receive"
    marker = bare / "rejected-once"
    hook.write_text(
        "#!/bin/sh\n"
        f"if [ ! -f '{marker}' ]; then touch '{marker}'; echo reject-once >&2; exit 1; fi\n"
        "exit 0\n",
        encoding="utf-8",
    )
    os.chmod(hook, 0o755)

    result = persist_overlay(
        work,
        branch="data/daily-forecast",
        message="retry persist",
        files=["lake/reports/retry.json"],
        retries=3,
        backoff_seconds=0,
    )
    assert result.status == "pushed"
    assert result.attempts == 2
    assert marker.exists()
    assert '"retry":true' in show_remote(work, "lake/reports/retry.json")


def test_path_traversal_is_rejected(tmp_path):
    _, work = make_repo(tmp_path)
    with pytest.raises(ValueError, match="unsafe repository-relative path"):
        persist_overlay(
            work,
            branch="data/daily-forecast",
            message="bad",
            files=["../outside.csv"],
        )
