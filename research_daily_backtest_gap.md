# Daily Backtest Gap Research

Date: 2026-05-28

## Finding

The daily GitHub Action is running successfully on `main`, and the live
`latest.json` is fresh for 2026-05-28. The live track record is not fresh:

- `https://etherforecast.live/latest.json` generated at `2026-05-28T05:40:20Z`
- `https://etherforecast.live/history.json` is `[]`
- `https://etherforecast.live/accuracy.json` has empty `horizons`
- `https://etherforecast.live/health.json` is not present
- `origin/data/daily-forecast:forecast_site/predictions.db` has:
  - `forecast_runs=1`
  - `forecasts=2`
  - `actuals=0`
  - `accuracy_snapshot=0`
  - `backtest_predictions=43800`

This means the website receives today's forecast and the frozen OOF backtest,
but it does not preserve live daily forecast history. Backfilled actuals cannot
accumulate because the SQLite DB is rebuilt from scratch on each scheduled run.

## Root Cause

The deploy branch is force-rebuilt each run, but the workflow did not restore
the previous deploy-branch `forecast_site/predictions.db` before inserting the
new forecast. Therefore the database starts over daily, `run_id` resets to `1`,
and there are no old forecast rows whose 7-day or 30-day target can mature.

`main` also lacks the newer feedback-history/model-evaluation workflow steps
that exist on `codex/eth-long-cycle-tail-signals`, so scheduled runs are not
using the latest forecast gating work.

## Implication

The "backtest stopped around April" symptom is an operating pipeline issue, not
only a model issue. The frozen OOF backtest is present, but the live resolved
forecast track record is empty after every deploy.

## Required Fix

1. Restore `forecast_site/predictions.db` from `origin/data/daily-forecast`
   before `persist_forecast`.
2. Continue restoring `prediction_history.csv` for model feedback features.
3. Export `forecast_site/public/health.json` so stale live history is visible.
4. Add tests that verify the health payload reports empty or resolved live
   history explicitly.
5. Merge the repaired branch to `main`; scheduled workflows only run from the
   default branch.
