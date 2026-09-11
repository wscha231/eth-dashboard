# Approved implementation and evidence

The user approved implementation, verification and site deployment on 2026-09-11. This extends PR #30's research and plan. It preserves the availability fixes from PR #29.

## Delivered behavior

- A complete evidence workspace compares all six horizons and applies one date/regime selection to its scores and charts. Historical reconstruction, timely published predictions and experimental candidates remain separate sources.
- Content-addressed yearly shards retain original forecast payloads, IDs, publication times, target times and latest settlement revision/evaluation times. The complete append-only revision history remains in the durable SQLite audit. Old records without origin regimes remain explicitly unknown.
- Price ranges, five probability reliability curves, bin counts, terminal confusion matrix, path precision/recall/FPR/AP, USD MAE, price MAPE, return MAE/RMSE, coverage, interval/pinball scores, signed error histogram and monthly errors are available with complete JSON downloads and pagination.
- Public deployment verifies exact HTML, both JavaScript files, replay/CSV equality, every evidence object's hash and row count, then current six-horizon readiness. Published receipts are preserved before readiness failure.

## Candidate protocol

Two fixed training-window choices: 365/730 days for 6–168h; 730/1095 days for 336–720h. Probability, point price and interval heads are selected separately. A no-change point estimate is an eligible price baseline. The final fitted objects end before a separate calibration window; they are never refitted after calibration. Temperature scaling, path logistic calibration and quantile-bound offsets are tied to an exact fitted-object identity. Alert thresholds use a later purged calibration partition and disable an event alert if fewer than 40 negative examples exist. Selection and probability calibration each require at least 60 matured examples. A later 90-calendar-day test remains development evidence because historical data has previously been inspected.

Saved short-horizon candidates refresh weekly; long-horizon candidates reuse the monthly cutoff unless source revisions invalidate its content key. Hourly inference never fits a model. Candidates have a separate immutable ledger and public receipt verification. They cannot overwrite `active.json` or incumbent records. Their measured prospective scores accumulate as targets mature; promotion still requires same-origin prospective evidence, calendar-block uncertainty and no material regression across price, probability and range quality. The current implementation intentionally retains the incumbent pending that evidence; it does not promise continual accuracy gains.

The review has a 1,400-second workflow budget, two CPU threads and content-sensitive completed-horizon caches. Only a complete six-horizon review writes public metadata and its shadow manifest. A failed research candidate step retains the previous complete candidate release and does not block an otherwise complete incumbent replay. Historical checkpoint caches and the original training hash remain compatible.

## Measured local development run

Source: the existing 169,653-row ETH/BTC hourly snapshot, through **2026-09-07 08:00 UTC**. Run time: **48.1 seconds**, six horizons. This snapshot predates the latest GitHub source refresh; deployment uses the fresh workflow result. Full measured summary: `candidate_measurements.json`.

| Horizon | Candidate return MAE (pp) | Incumbent MAE (pp) | Candidate event Brier | Incumbent event Brier |
|---|---:|---:|---:|---:|
| 6h | 0.6924 | 0.6854 | 0.13738 | 0.13669 |
| 24h | 1.8761 | 1.8732 | 0.17748 | 0.17113 |
| 72h | 3.4580 | 3.4508 | 0.15503 | 0.14949 |
| 168h | 6.0448 | 6.0441 | 0.16277 | 0.17637 |
| 336h | 9.3425 | 9.6275 | 0.20359 | 0.16589 |
| 720h | 17.8290 | 14.7249 | 0.15841 | 0.16354 |

There is no uniform improvement. In particular the 30-day candidate has materially worse point error. **No candidate was promoted.** Publishing the measured failures is part of the review process.

## Verification

Local focused regression suite: 79 passed, including issuance deadlines, receipt-before-readiness behavior, ledger preservation, archive integrity/counts, Python/browser metric agreement, filter boundaries, model/calibrator identity and future-target perturbation. Subsequent targeted tests after integration adjustments: 25 passed. The GitHub full regression and fresh replay runs are attached to PR #30. Actual deployment status must be confirmed from the post-merge publication run; this document does not declare the site updated before that check.

## Rollback

Revert the interface/pipeline commit or restore the previous active model manifest. Do not delete or regenerate issued records. Existing historical and published predictions retain their original values. The serving branch retains two evidence generations to support in-flight readers; hashes prevent mixed-generation scores.
