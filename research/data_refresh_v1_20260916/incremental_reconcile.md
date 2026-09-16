# Incremental refresh + reconciliation design

Date: 2026-09-16

## Live evidence

The first retry-safe continuous run succeeded across freshness, fast snapshots and long refresh. Freshness reported 24 active sources OK, 3 planned, zero hard failures and zero unregistered raw CSVs. The long refresh updated Bitget, Deribit DVOL and FRED/ALFRED through 2026-09-16, but the full-history Hyperliquid funding request ended with HTTP 429 while the previously accumulated historical frame remained intact.

This shows the next sustainability issue: once a long backfill exists, repeatedly downloading the entire history twice per day is unnecessary and increases rate-limit risk.

## Refresh tiers

### Intraday incremental
- Bitget/Hyperliquid current context: 4h actual receipts.
- No historical replay.

### Twice-daily incremental long-source refresh
- Bitget basis/funding: fetch a recent overlap sufficient for 30-day derived features, not the full multi-year history.
- Deribit DVOL: refresh a recent overlap.
- Hyperliquid funding/premium: fetch a recent overlap sized to fit a small number of public API pages.
- FRED current-vintage: recent observations plus the durable prior history.
- FRED/ALFRED initial-release: recent realtime/vintage window plus durable prior initial-release rows.
- Merge by key/index into the historical cache.

### Weekly full reconciliation
Once per Sunday, rerun the full free historical collectors from the configured research start. This can repair historical gaps or provider corrections that a tail refresh would miss. Pace paginated APIs to reduce rate-limit risk. A source failure does not delete cached history.

### Monthly deep audit
On the first day of each month:
- perform full reconciliation;
- regenerate freshness/coverage reports;
- verify unregistered source detection;
- record PIT/revision limitations;
- trigger the existing Drive archive and rely on its hash/integrity checks;
- retain licensing/redistribution review as a required metadata task before commercial/public use.

## Principles

1. Initial backfill remains full if no cache exists.
2. Normal refresh never truncates historical data.
3. A provider error keeps the previous cache and is reported explicitly.
4. Derived rolling features require enough overlap (at least 45 days for 30-day z-scores).
5. Full reconciliation repairs old history; it does not change historical receipt semantics. Backfilled old values remain reconstructed research unless original vintage/receipt evidence exists.
6. MODEL promotion remains separate from collection health.
