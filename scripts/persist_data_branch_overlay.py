#!/usr/bin/env python3
"""Persist generated DATA files onto the latest hot-data branch with retries.

Multiple collectors can finish near the same time. Each attempt starts from the
latest remote branch tip and overlays only the explicitly requested files, so a
non-fast-forward push can be retried without deleting unrelated collector data.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tempfile
import time
from typing import Iterable


@dataclass(frozen=True)
class PersistResult:
    status: str
    attempts: int
    files: tuple[str, ...]
    commit: str | None = None


def _run(
    args: list[str],
    *,
    cwd: Path,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        check=check,
        text=True,
        capture_output=capture,
    )


def _safe_relative(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"unsafe repository-relative path: {value}")
    return path.as_posix()


def _remote_ref(branch: str) -> str:
    clean = branch.strip().strip("/")
    if not clean or clean.startswith("-") or ".." in clean or any(ch.isspace() for ch in clean):
        raise ValueError("invalid target branch")
    return f"refs/remotes/origin/{clean}"


def _cleanup_worktree(source_root: Path, target: Path) -> None:
    subprocess.run(
        ["git", "worktree", "remove", "--force", str(target)],
        cwd=str(source_root),
        check=False,
        text=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=str(source_root),
        check=False,
        text=True,
        capture_output=True,
    )


def persist_overlay(
    source_root: str | Path,
    *,
    branch: str,
    message: str,
    files: Iterable[str],
    retries: int = 5,
    backoff_seconds: float = 1.0,
) -> PersistResult:
    root = Path(source_root).resolve()
    if not (root / ".git").exists():
        raise ValueError("source_root must be a Git working tree")
    if not message.strip():
        raise ValueError("commit message is required")
    if retries < 1:
        raise ValueError("retries must be >= 1")

    relative_files = tuple(dict.fromkeys(_safe_relative(value) for value in files))
    if not relative_files:
        raise ValueError("at least one overlay file is required")

    existing = tuple(path for path in relative_files if (root / path).is_file())
    if not existing:
        return PersistResult("no_files", 0, tuple())

    remote_ref = _remote_ref(branch)
    fetch_spec = f"refs/heads/{branch}:{remote_ref}"
    last_error = ""

    for attempt in range(1, retries + 1):
        _run(["git", "fetch", "origin", fetch_spec], cwd=root)
        with tempfile.TemporaryDirectory(prefix="eth-data-overlay-") as temp:
            target = Path(temp) / "worktree"
            _run(["git", "worktree", "add", "--detach", str(target), remote_ref], cwd=root)
            try:
                staged: list[str] = []
                for relative in existing:
                    source = root / relative
                    destination = target / relative
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, destination)
                    staged.append(relative)

                _run(["git", "config", "user.name", "eth-forecast-bot"], cwd=target)
                _run(["git", "config", "user.email", "eth-forecast-bot@users.noreply.github.com"], cwd=target)
                _run(["git", "add", "-f", "--", *staged], cwd=target)
                diff = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    cwd=str(target),
                    check=False,
                )
                if diff.returncode == 0:
                    return PersistResult("no_changes", attempt, tuple(staged))
                if diff.returncode != 1:
                    raise RuntimeError(f"git diff --cached failed with code {diff.returncode}")

                _run(["git", "commit", "-m", message], cwd=target)
                commit = _run(["git", "rev-parse", "HEAD"], cwd=target).stdout.strip()
                pushed = subprocess.run(
                    ["git", "push", "origin", f"HEAD:refs/heads/{branch}"],
                    cwd=str(target),
                    check=False,
                    text=True,
                    capture_output=True,
                )
                if pushed.returncode == 0:
                    return PersistResult("pushed", attempt, tuple(staged), commit)
                last_error = (pushed.stderr or pushed.stdout or "git push failed").strip()
            finally:
                _cleanup_worktree(root, target)

        if attempt < retries:
            time.sleep(max(0.0, backoff_seconds) * attempt)

    raise RuntimeError(f"failed to persist overlay after {retries} attempts: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", default="data/daily-forecast")
    parser.add_argument("--message", required=True)
    parser.add_argument("--file", action="append", dest="files", required=True)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--backoff-seconds", type=float, default=1.0)
    args = parser.parse_args()

    result = persist_overlay(
        Path.cwd(),
        branch=args.branch,
        message=args.message,
        files=args.files,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
    )
    print(asdict(result))


if __name__ == "__main__":
    main()
