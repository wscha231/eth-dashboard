# DATA hot-branch overlay retry hardening

Date: 2026-09-16

## New evidence after PR #61

The three new DATA workflows were placed in one GitHub Actions concurrency group with `cancel-in-progress: false`. The first live run showed this is not a sufficient multi-writer queue: one workflow ran, one remained pending, and another pending freshness run was cancelled. GitHub concurrency therefore cannot be relied on as an unlimited writer queue across separate workflows.

The durable solution is to make each data-branch write idempotent and retryable against the latest `data/daily-forecast` tip.

## Required behavior

For every source workflow that writes the hot DATA branch:

1. Keep its own workflow-level concurrency group so repeated instances of the same collector do not overlap.
2. Generate source outputs in the normal checkout.
3. Before persistence, fetch the latest `data/daily-forecast` branch.
4. Create a temporary worktree from that latest tip.
5. Copy only the explicitly generated output files into the worktree; never replace the whole vendor directory.
6. Commit only those overlay files.
7. Push to `data/daily-forecast`.
8. If the push is rejected because another writer advanced the branch, remove the worktree, fetch the new tip, re-apply the same generated output files, and retry with bounded backoff.
9. Preserve unrelated files added by other collectors.

This makes simultaneous push-triggered runs a useful stress test rather than a data-loss risk.

## Registry evidence

The first freshness run also identified nine existing raw CSVs not explicitly represented in the original registry. PR #61 added them through `config/data_source_registry_existing_supplement.json`; the freshness loader merges the base registry and supplement while retaining automatic discovery of future unregistered CSV files.

## Scope boundary

This change modifies persistence coordination only. It does not change source values, infer missing history, promote any feature, alter forecasts, or enable trading.
