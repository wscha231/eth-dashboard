# R1 exit replay contract

Status: **retrospective/offline only**.

This is the final R1 plumbing step before R2 model competition.

## Inputs

The exporter consumes the exact current shadow reviews referenced by `lake/signals/optimization.json`. A review file is accepted only when its complete public payload matches the corresponding horizon entry; zero or multiple matches fail closed.

Candidate, incumbent and climatology rows must share identical origins and truth. No model is fitted or reselected by the exporter.

## Head scoring on the same origins

- **Center:** candidate simple-return MAE/RMSE/pinball; mandatory no-change baseline; incumbent comparison.
- **Distribution:** candidate WIS80/pinball/coverage/width; incumbent and climatology comparison.
- **Event:** candidate terminal/path Brier, log-loss and ECE; incumbent and climatology comparison.
- **Volatility:** use the already frozen target from `model_lab_hourly_endpoint_v1`: `sum of h squared close-to-close log returns inside path window`. R1 scores only persistence `h * eth_vol_720^2`; challenger skill remains intentionally `null` until R2.

## Output

`lake/signals/v2_replay.json` plus a human-readable `v2_replay.md` are research artifacts retained with the event-research state.

The report is not a promotion certificate. It does not write the production registry, live issuance ledger or website.

## R1 exit criterion

R1 architecture is complete when one workflow run produces the V2 replay successfully for all six horizons with matching origins and no regression in the repository/unit tests. R2 can then preregister and test short-horizon variance/range challengers.
