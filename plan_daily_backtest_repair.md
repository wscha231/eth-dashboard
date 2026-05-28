# Daily Backtest Repair Plan

## Goal

Make the daily pipeline append one new forecast per day, backfill matured
7-day and 30-day forecasts, and publish visible health diagnostics so the live
site cannot silently lose its track record again.

## Implementation

1. Daily workflow
   - Fetch `origin/data/daily-forecast`.
   - Restore `forecast_site/predictions.db` before running collector/forecast.
   - Keep restoring `eth_forecast_outputs/prediction_history.csv`.
   - Force-add `forecast_site/public/health.json` to the deploy branch.

2. JSON export
   - Continue exporting `latest.json`, `accuracy.json`, and `history.json`.
   - Add `health.json` with DB row counts, latest forecast date, latest
     resolved target date by horizon, and due unresolved counts.

3. Tests
   - Add export-health tests for bootstrap and resolved-history states.
   - Keep existing full pytest suite green.

4. Operations
   - Trigger a manual daily forecast on the repaired branch to republish the
     site immediately.
   - Trigger full model evaluation when code changes affecting model selection
     or backtest output are included.
   - Merge to `main` after checks pass, otherwise the next scheduled run on
     `main` will overwrite the repaired deploy branch.

## Notes

The current remote deploy DB only contains the latest forecast, so old live
forecast history cannot be recovered from that DB alone. If older GitHub
artifacts are still retained, they can be ingested separately, but the durable
fix is preserving the DB on every future daily run.
