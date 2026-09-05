"""Exercise the real publisher against local git branches without contacting the site."""
from pathlib import Path
import os
import subprocess

import pytest


@pytest.mark.parametrize("complete", [False, True])
def test_hybrid_publish_preserves_deployed_workflows_and_archives(tmp_path, complete):
    source = Path(__file__).resolve().parents[1]
    remote, working, runner = (tmp_path / name for name in ("remote.git", "working", "runner"))
    runner.mkdir()

    def git(*args):
        return subprocess.check_output(["git", *map(str, args)], text=True).strip()

    git("init", "--bare", remote)
    git("init", "-b", "main", working)
    git("-C", working, "config", "user.name", "Test")
    git("-C", working, "config", "user.email", "test@example.invalid")
    git("-C", working, "remote", "add", "origin", remote)

    def write(path, value):
        target = working / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)

    write(".github/workflows/daily.yml", "deployed workflow")
    write("forecast_site/public/index.html", "old frontend")
    write("forecast_site/predictions.db", "immutable archived issues")
    git("-C", working, "add", ".")
    git("-C", working, "commit", "-m", "deployed baseline")
    deployed = git("-C", working, "rev-parse", "HEAD")
    git("-C", working, "push", "origin", "HEAD:refs/heads/data/daily-forecast")
    write(".github/workflows/daily.yml", "new workflow")
    write(".github/workflows/train.yml", "new training workflow")
    write("forecast_site/public/index.html", "corrected frontend")
    git("-C", working, "add", ".")
    git("-C", working, "commit", "-m", "tested source")
    write("lake/hybrid/hybrid_forecast.json", '{"fixture": true}')
    write("lake/hybrid/hybrid_predictions.csv.gz", "new chart data")
    write("lake/hybrid/issued.db", "new immutable issues")

    # Record the final site check while exercising the actual git publisher.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    stub = bin_dir / "python"
    stub.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$RUNNER_TEMP/verification-call"\n')
    stub.chmod(0o755)
    env = dict(os.environ, RUNNER_TEMP=str(runner),
               PATH=f"{bin_dir}:{os.environ['PATH']}",
               HYBRID_PUBLISH_COMPLETE=str(complete).lower())
    subprocess.run(["bash", str(source / "scripts/publish_hybrid.sh")],
                   cwd=working, env=env, check=True, capture_output=True, text=True)

    def remote_git(*args):
        return git("--git-dir", remote, *args)

    data_ref = "refs/heads/data/hybrid-forecast"
    deploy_ref = "refs/heads/data/daily-forecast"
    assert remote_git("diff", "--name-only", deployed, data_ref).splitlines() == [
        "lake/hybrid/hybrid_forecast.json", "lake/hybrid/hybrid_predictions.csv.gz", "lake/hybrid/issued.db"]
    for ref in (data_ref, deploy_ref):
        assert remote_git("diff", deployed, ref, "--", ".github/workflows", "forecast_site/predictions.db") == ""
    if complete:
        assert remote_git("diff", "--name-only", deployed, deploy_ref).splitlines() == [
            "forecast_site/public/hybrid_forecast.json", "forecast_site/public/hybrid_predictions.csv.gz",
            "forecast_site/public/index.html"]
        assert remote_git("show", f"{deploy_ref}:forecast_site/public/index.html") == "corrected frontend"
        assert (runner / "verification-call").read_text().splitlines() == [
            "scripts/verify_hybrid_site.py", "--expected", "lake/hybrid/hybrid_forecast.json"]
    else:
        assert remote_git("rev-parse", deploy_ref) == deployed
        assert not (runner / "verification-call").exists()
