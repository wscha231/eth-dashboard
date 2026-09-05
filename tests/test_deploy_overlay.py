"""Reproduce publishing changed code while preserving deploy workflow files."""
from pathlib import Path
import subprocess


def test_overlay_preserves_workflows_even_when_main_added_new_ones(tmp_path):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path, text=True).strip()

    git("init", "-b", "main")
    git("config", "user.name", "Test")
    git("config", "user.email", "test@example.invalid")
    workflows = tmp_path / ".github" / "workflows"
    workflows.mkdir(parents=True)
    (workflows / "daily.yml").write_text("old workflow")
    (tmp_path / "index.html").write_text("old frontend")
    git("add", ".")
    git("commit", "-m", "deployed baseline")
    git("branch", "deployed")
    (workflows / "daily.yml").write_text("new workflow")
    (workflows / "train.yml").write_text("new training workflow")
    (tmp_path / "index.html").write_text("corrected frontend")
    git("add", ".")
    git("commit", "-m", "tested source")
    git("checkout", "deployed")
    git("checkout", "main", "--", ".")
    source = (Path(__file__).resolve().parents[1] / ".github/workflows/daily_forecast.yml").read_text()
    cleanup = next(line.strip() for line in source.splitlines() if "git rm " in line and "--ignore-unmatch" in line)
    subprocess.run(["bash", "-e", "-c", cleanup], cwd=tmp_path, check=True)
    git("checkout", "deployed", "--", ".github/workflows")
    assert git("diff", "--cached", "--", ".github/workflows") == ""
    assert "corrected frontend" in git("diff", "--cached", "--", "index.html")
    assert not (workflows / "train.yml").exists()
