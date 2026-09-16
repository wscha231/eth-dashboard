# DATA hot-branch overlay retry implementation plan

Date: 2026-09-16
Status: implementation approved under the user's instruction to make collection continuous and durable.

## 1. Shared persistence helper

Add `scripts/persist_data_branch_overlay.py`.

CLI inputs:
- target branch, default `data/daily-forecast`
- commit message
- one or more repository-relative output files
- bounded retry count and backoff

Algorithm per attempt:
1. fetch the latest remote target branch;
2. create a fresh detached temporary worktree at that remote tip;
3. copy only existing requested overlay files from the caller checkout into identical repository-relative paths in the worktree;
4. stage only those paths;
5. if no diff, exit successfully;
6. commit and push;
7. on non-fast-forward/rejected push, clean the worktree, back off, and repeat from the new remote tip;
8. on non-retryable git/file errors, fail clearly.

The helper must never delete unrelated files or rewrite an entire raw-data directory.

## 2. Workflow integration

Replace custom worktree/push blocks in:
- `free_source_backfill.yml`
- `fast_derivative_snapshots.yml`
- `data_refresh_guard.yml`

with the helper. Restore independent concurrency groups so GitHub does not cancel pending runs across different workflows. Their schedules remain staggered.

## 3. Tests

Add `tests/test_persist_data_branch_overlay.py` using a temporary bare Git repository. Verify:
- an overlay file is committed to the target branch;
- an unrelated file already on the target branch is preserved;
- optional missing overlay files are skipped rather than deleting target files;
- repeated identical persistence is a no-op;
- repository-relative path traversal is rejected.

Existing full-repository CI remains the merge gate.

## 4. Live stress validation

After merge, the same main push will trigger fast, full-refresh and freshness workflows near-simultaneously. Confirm:
- none is cancelled by a shared concurrency group;
- all complete successfully;
- the target branch contains fast context, refreshed long-source files, and freshness status;
- `unregistered_files` is empty or contains only genuinely new sources;
- Drive archive workflow completes after the source runs and the six-hour sweep remains fallback.

No model or site promotion is part of this plan.
