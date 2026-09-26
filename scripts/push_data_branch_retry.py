#!/usr/bin/env python3
"""Push one prepared data-branch commit with bounded non-fast-forward recovery."""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
import re
import subprocess
import time


@dataclass(frozen=True)
class PushResult:
    status: str
    attempts: int
    commit: str


def _run(args: list[str], *, cwd: Path, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(args, cwd=str(cwd), check=check, text=True, capture_output=True)


def _branch(value: str) -> str:
    value=value.strip().strip("/")
    if not value or value.startswith("-") or ".." in value or any(c.isspace() for c in value):
        raise ValueError("invalid branch")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", value):
        raise ValueError("invalid branch")
    return value


def push_with_rebase(worktree: str | Path, branch: str, *, retries: int = 5, backoff_seconds: float = 1.0) -> PushResult:
    root=Path(worktree).resolve()
    if not (root / ".git").exists() and not (root / ".git").is_file():
        raise ValueError("worktree must be a Git worktree")
    branch=_branch(branch)
    if retries < 1:
        raise ValueError("retries must be >= 1")
    remote_ref=f"refs/remotes/origin/{branch}"
    fetch_spec=f"refs/heads/{branch}:{remote_ref}"
    for attempt in range(1,retries+1):
        head=_run(["git","rev-parse","HEAD"],cwd=root).stdout.strip()
        pushed=_run(["git","push","origin",f"HEAD:refs/heads/{branch}"],cwd=root,check=False)
        if pushed.returncode == 0:
            return PushResult("pushed",attempt,head)
        message=(pushed.stderr or pushed.stdout or "").lower()
        race=("fetch first" in message or "non-fast-forward" in message or "rejected" in message)
        if not race or attempt >= retries:
            raise RuntimeError("data branch push failed without a recoverable non-fast-forward")
        _run(["git","fetch","origin",fetch_spec],cwd=root)
        rebased=_run(["git","rebase",remote_ref],cwd=root,check=False)
        if rebased.returncode != 0:
            _run(["git","rebase","--abort"],cwd=root,check=False)
            raise RuntimeError("data branch changed overlapping files; refusing automatic overwrite")
        time.sleep(max(0.0,backoff_seconds)*attempt)
    raise RuntimeError("data branch push retry exhausted")


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worktree",type=Path,required=True)
    parser.add_argument("--branch",default="data/daily-forecast")
    parser.add_argument("--retries",type=int,default=5)
    parser.add_argument("--backoff-seconds",type=float,default=1.0)
    args=parser.parse_args()
    try:
        result=push_with_rebase(args.worktree,args.branch,retries=args.retries,backoff_seconds=args.backoff_seconds)
    except Exception as exc:
        print("Retry-safe push failed: "+str(exc),flush=True)
        return 1
    print(asdict(result),flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
