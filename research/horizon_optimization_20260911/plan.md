# Per-horizon forecast optimization and full diagnostics plan

Status: concrete plan awaiting the user's review under the existing research-before-code workflow. This scope goes beyond the already completed PR #29 availability fix. Research findings and an actual-data chart report accompany this plan.

## 1. Deliver the requested evidence surface

Use the existing English site, current theme and Chart.js; retain the recent availability fix. Add a six-horizon overview with original issue time, reference price, target end, median, 80% range and current/delayed/previous status. Endpoints from different horizons must not be presented as an inferred continuous price path.

Provide two separately labeled views:

- Historical reconstruction: all available chronological replay predictions; original target-end dates, actual prices, median and uncertainty band.
- Actual published forecasts: all timely published versions, with issued/published/input/end times, forecast ID and settlement revision. Pending and missing-truth outcomes remain pending.

The common controls are forecast horizon, date interval, all-history/recent period and existing origin-time market state. Every chart, metric and denominator must follow the same selection. Show each horizon's first/last evaluated date, missing-calendar coverage and nonoverlapping count.

Required charts and metrics:

| Question | Chart / measure |
|---|---|
| How did all six forecasts compare? | Six-horizon overview; median and original range at each distinct target end |
| Did price reach the forecast? | Predicted/actual end-price history, expandable original forecast record |
| How large were errors? | USD MAE, price MAPE, return MAE/RMSE in percentage points; signed-error distribution and time series |
| Did stated probability match reality? | Reliability plots for upper/lower barrier hits and the three end-price classes; sample counts; Brier and log loss |
| Did the model recognize large moves? | Three-class confusion matrix and balanced accuracy; upper/lower event precision, recall, PR-AUC and false-positive rate |
| Was uncertainty realistic? | Target 80% versus observed coverage, interval width and proper interval/pinball scores; distinguish over-wide intervals from accurate predictions |
| Has quality improved? | Rolling settled performance and matched baseline comparison, with model-version change markers and sample counts |

Do not derive an exact CRPS from only three quantiles. Do not translate an uncalibrated score, its rank, or R-squared into a success probability. When a class/event is absent, display insufficient evidence instead of a misleading 0% or 100% headline.

## 2. Export complete histories without slowing hourly issuance

Extend research plot exports with terminal probabilities/classes, saved alert thresholds, selected model/version, baseline predictions and provenance. Export all per-origin rows, not only a recent sample.

Read the immutable issued ledger and generate versioned monthly/horizon history files plus a manifest. Load selected files on demand. Keep current availability and recent fallback records small. The manifest records schema, source/release IDs, counts, dates and file hashes; a page must not combine files from different releases.

Preserve existing SQLite schema and all issued forecast values. An outcome revision creates an updated evaluation view, retaining revision metadata. Actual-performance eligibility requires an original timely verified publication. Late/unpublished records are visible in the audit view but excluded from timely live-performance denominators.

Add exact archive/manifest checks to the existing publication verification. A partially copied archive cannot be reported as complete. Preserve the receipt-before-readiness order implemented in PR #29.

## 3. Correct calibration before optimizing methods

Reconcile the existing frozen-model study. For each candidate, enforce chronological training -> selection -> final fit -> calibration -> untouched evaluation. Every label must end before its next partition, including the horizon and embargo. Calibrate probabilities and alert thresholds on the exact frozen model that issues future predictions; no refit after calibration.

Separate probability calibration from alert-threshold calibration. Compare a simple sigmoid/shrinkage approach first; isotonic calibration requires adequate observations. Report calibration sample size and independent event support. Insufficient data keeps a conservative baseline or disables alerts; it must not reuse mismatched thresholds or silently count overlapping rows as independent events.

Rename the fixed 50:50 mixture accurately in the display. Keep raw predictions, corrected probabilities, selected thresholds and the exact fitted-model hash for reproducibility.

## 4. Optimize outputs separately for every horizon

| Horizon | Initial low-cost comparison | Update policy to evaluate |
|---|---|---|
| 6h | Current compact model vs corrected calibration; shorter versus current training window; past-only volume/volatility features | Hourly inference; weekly candidate review/refit if enough new settled data |
| 1d | Same base candidates; separate point-price and event model selection | Hourly inference; weekly candidate review/refit |
| 3d | Conservative frequency/no-change baselines vs compact model with continuous trend/volatility inputs | Hourly inference; monthly model review, weekly diagnostics |
| 7d | Separate point/interval selection; bounded recent-data weighting | Hourly inference; monthly model review |
| 14d | Longer training and regularized frequency/volatility candidates; stable interval calibration | Hourly inference; monthly model review; delayed-label-aware calibration |
| 30d | No-change/frequency baselines explicitly compete with all learned outputs; shrinkage and calibrated uncertainty first | Hourly inference; monthly model review; only completed 30-day targets update error/calibration evidence |

Select the point-price model on held-out absolute error, probability outputs on proper probability scores and event detection, and interval outputs on pinball/interval score plus coverage and width. A probability winner does not automatically win the median or range. Fallback to a baseline is an explicit recorded choice.

Begin with existing model families and at most two predeclared training-window options per horizon. If a compact alternative is needed, compare one CPU LightGBM candidate before a CNN-LSTM/Transformer. Avoid a large hyperparameter search or splitting the limited long-horizon sample into many independent regime models.

Funding/DVOL inputs from PR #26 are separate ablations on common available dates. Preserve real receipt times and missingness. A successful data fetch is not a model-adoption gate. Do not fabricate historical liquidation, open-interest or ETF inputs.

## 5. Validation and candidate adoption

Freeze candidates, losses, split rules and resource limits before measuring a new run. Keep the present baseline and its full original results. Compare identical origins and complete label windows; report every attempted candidate, including failures.

Historical results already inspected in this project remain development evidence. Use outer chronological evaluation to compare method stability, then accumulate a new, untouched prospective shadow period after the approved experiment is registered. A successful CI/replay job is not a performance approval.

Before an output is eligible for promotion, require:

1. Correct timestamp, input-receipt, target-maturity and exact model/calibrator identity contracts.
2. Improvement against both the incumbent and relevant baseline on the predeclared primary loss, with a paired calendar-block interval favoring the candidate. Block length is at least a full target interval; report how few effective long-horizon samples remain.
3. No unreported deterioration in opposite-direction detection, false alarms, price error, interval coverage/width or the predeclared market-state diagnostics. Optimize on earlier folds, not on the period shown to the user.
4. A separately defined prospective observation requirement appropriate to the horizon. Thirty-day success cannot be certified after a week, and 30-day forecast windows cannot mature faster by issuing more often.
5. Saved protocol, code/data/model/calibrator hashes, tested fallback, and a versioned adoption decision. Failures retain the incumbent or explicit baseline.

The initial calibration correction and all six output comparisons can be implemented together, but a model must remain a shadow candidate until its evidence passes. This plan does not promise improved predictive accuracy or automatically promote a failed candidate.

## 6. Bounded recurring work

- Hourly: collect incremental complete bars, issue from approved saved models, settle newly completed targets, update current cards and the appropriate archive shard. Never fit in the hourly worker.
- Daily: refresh mature-error/calibration diagnostics and verify archive counts. Recomputing diagnostics does not imply retraining.
- Weekly: check drift and errors by horizon; evaluate a bounded short-horizon candidate only when meaningful new outcomes exist. One active research worker; no concurrent duplicate backtests.
- Monthly: run the registered six-horizon retraining/selection study; inspect long-horizon uncertainty before proposing adoption.
- Initial research budget: two CPU threads; checkpoint and stop at 1,500 seconds per job, consistent with the current research budget. Separate candidate outputs from active checkpoints. Resume saved work; do not publish an incomplete replay as complete or let research delay the hourly worker.
- Cache and incremental exports must be measured against present hourly runtime; no expensive model downloads or full-history fitting in a normal forecast refresh.

## 7. Files and acceptance evidence

Expected changes: `signal_pipeline/evaluate.py`, `engine.py`, `models.py`, a separately versioned protocol/configuration, archive exporter, relevant research/hourly workflows, `forecast_site/public/index.html`, `events.js`, publication verification and focused tests. Keep rejected experimental branches and original ledger records intact.

Acceptance:

- Recomputed metrics match the displayed selected rows and denominator for all six horizons.
- Date selection, all-history view, empty/immature outcomes, disconnected charts, delayed fallback and model-version switches behave correctly.
- All individual timely published records are reachable; archive totals reconcile with the ledger, with no missing/duplicated IDs.
- Historical and live performance cannot be mixed; probability kinds and error units are explicit.
- Future-source perturbations cannot change earlier predictions; incomplete labels cannot enter training/calibration/evaluation.
- The exact trained model calibrated is the model issuing probabilities; an artificially changed final model invalidates its calibration.
- Full relevant CI and bounded chronological replay complete before merge. Confirm the deployed HTML/JS/data hashes and six-horizon archive/metric consistency.

Rollback restores the previous production code/model manifest while retaining all forecasts, deliveries and settlement revisions. The existing availability guard remains in place.
