from pathlib import Path
import os
import subprocess

import pytest

from scripts.push_data_branch_retry import push_with_rebase


def git(cwd: Path, *args: str, check: bool = True):
    return subprocess.run(["git",*args],cwd=cwd,check=check,text=True,capture_output=True)


def repo(tmp_path: Path):
    bare=tmp_path/"origin.git";git(tmp_path,"init","--bare",str(bare))
    a=tmp_path/"a";git(tmp_path,"clone",str(bare),str(a))
    git(a,"config","user.name","test");git(a,"config","user.email","test@example.com")
    (a/"base.txt").write_text("base\n");git(a,"add",".");git(a,"commit","-m","base")
    git(a,"branch","-M","main");git(a,"push","-u","origin","main")
    git(a,"switch","-c","data/daily-forecast");git(a,"push","-u","origin","data/daily-forecast")
    b=tmp_path/"b";git(tmp_path,"clone",str(bare),str(b))
    git(b,"config","user.name","test");git(b,"config","user.email","test@example.com")
    git(b,"fetch","origin","data/daily-forecast");git(b,"checkout","--detach","origin/data/daily-forecast")
    return bare,a,b


def test_nonoverlapping_remote_race_rebases_and_pushes(tmp_path):
    _,a,b=repo(tmp_path)
    (b/"daily.txt").write_text("daily\n");git(b,"add",".");git(b,"commit","-m","daily")
    git(a,"switch","data/daily-forecast")
    (a/"status.txt").write_text("status\n");git(a,"add",".");git(a,"commit","-m","status");git(a,"push")
    result=push_with_rebase(b,"data/daily-forecast",retries=3,backoff_seconds=0)
    assert result.status=="pushed" and result.attempts==2
    git(a,"pull")
    assert (a/"daily.txt").read_text()=="daily\n"
    assert (a/"status.txt").read_text()=="status\n"


def test_overlapping_remote_race_fails_closed(tmp_path):
    _,a,b=repo(tmp_path)
    (b/"base.txt").write_text("local\n");git(b,"add",".");git(b,"commit","-m","daily")
    git(a,"switch","data/daily-forecast")
    (a/"base.txt").write_text("remote\n");git(a,"add",".");git(a,"commit","-m","status");git(a,"push")
    with pytest.raises(RuntimeError,match="overlapping"):
        push_with_rebase(b,"data/daily-forecast",retries=3,backoff_seconds=0)
    assert git(b,"status","--porcelain").stdout.strip()==""


def test_non_race_push_failure_is_not_retried(tmp_path):
    bare,a,b=repo(tmp_path)
    (b/"daily.txt").write_text("daily\n");git(b,"add",".");git(b,"commit","-m","daily")
    hook=bare/"hooks/pre-receive"
    hook.write_text("#!/bin/sh\necho policy-denied >&2\nexit 1\n")
    os.chmod(hook,0o755)
    with pytest.raises(RuntimeError,match="without a recoverable"):
        push_with_rebase(b,"data/daily-forecast",retries=3,backoff_seconds=0)


@pytest.mark.parametrize("branch",["","../bad","-oops","has space"])
def test_branch_validation(tmp_path,branch):
    with pytest.raises(ValueError):
        push_with_rebase(tmp_path,branch)
