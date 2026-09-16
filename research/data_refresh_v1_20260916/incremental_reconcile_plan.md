# Incremental refresh + reconciliation implementation plan

Date: 2026-09-16
Status: implementation approved by the ongoing DATA collection instruction.

## 1. Tail-refresh selector

Add a helper in `scripts/collect_free_source_fallbacks.py` that returns:
- full requested start when cache is absent or `--full-reconcile` is set;
- otherwise latest cached observation minus a configurable overlap.

Default overlap: 45 days for Bitget and Deribit DVOL so 30-day rolling features can be recomputed safely; 14 days for Hyperliquid funding to keep routine requests small.

## 2. FRED incremental windows

Extend the FRED collection helpers with optional observation/realtime starts. Routine refresh requests only the recent window and merges into historical CSVs. Weekly/monthly reconciliation requests the full historical period.

## 3. Hyperliquid pacing

Add a small inter-page delay to long funding pagination. Routine refresh should normally fit in one page; full reconciliation may need dozens of pages and must avoid burst rate limits.

## 4. Reconciliation workflow

Add `.github/workflows/data_reconciliation.yml`:
- weekly Sunday full reconciliation;
- monthly first-day full reconciliation + freshness audit;
- retry-safe overlay persistence;
- artifact receipt;
- direct Drive archive trigger through `gdrive_archive.yml`.

Use a workflow input/mode only where supported; scheduled runs determine weekly vs monthly by date but both may execute full reconciliation safely.

## 5. Tests

Cover:
- cached tail start selection;
- full reconcile start selection;
- FRED recent-window parameter propagation;
- pagination pacing without sleeping in parser tests;
- normal incremental collection preserving old cache rows.

## 6. Verification

After merge:
- normal twice-daily refresh should use incremental windows and avoid a full Hyperliquid replay;
- run should persist on top of current hot branch;
- source report must expose refresh mode and requested sub-ranges;
- Drive archive must succeed after the workflow.
