# EtherForecast Recovery Wave 2 — 2026-09-19

Base: main `be7e8070f3ccc542cee519d1648fb483b2e2c172`. Canonical tracker #68, v1.5. Scope and feedback policy frozen in comment 5739737641 before candidate results.

## Operational failure mechanism and change

The original hourly workflow placed dense HAR research before core ledger persistence and public release. A missing checkpoint could therefore prevent valid core forecasts from being published. In addition, `run_variance_dense_shadow.py` prints per-horizon errors without itself exiting nonzero. A process exit code alone is not a semantic success gate.

Wave 2 moves optional dense work AFTER core public delivery, stages it in a temporary workspace, backs up input observations read-only, restores the existing dense ledger and allowed legacy checkpoints from the trusted audit branch, validates hash/JSON/SQLite/append-only contracts, and accepts only complete error-free outputs. A missing existing cohort never falls back to an empty replacement.

Optional failure is NOT hidden: step outcome is recorded; core publication remains separately visible; the final health gate fails only after final evidence retention. The original core model, immutable issuance semantics, exact publication verifier, shared writer lock and rights gates remain unchanged. Only 24h/72h dense authorization is used; 6h remains excluded.

The real effects this patch can claim after testing are isolation of this dependency path and improved failure attribution. It cannot explain every old missing issue slot or promise that GitHub scheduling/CDN delays disappear. Further separation of research refresh/checkpoint installation and shared-lock contention remains follow-up.

## Automated failure review

The actual SQLite ledger is opened read-only. Review uses only forecasts issued by the requested as-of time, real verification before window start, and outcomes available after target maturity. It reports all qualifying windows, not only alert days, with model/target cohorts, descriptive non-overlap counts, missed barrier examples and false alerts.

The historical archive is read by its pinned manifest. Each file's path, SHA256, length, horizon and row count are verified. Historical training/validation cutoffs and h+1 target semantics are checked. A nominal replay-calendar gap is labelled `not_in_pinned_archive`; its cause is not invented. Missing historical replay is never inserted into actual issuance.

`failure_review.json/.md`, `dense_stage_status.json` and `recovery_health.json` are retained with the normal final hourly snapshot. The ordinary hourly job does NOT run the experimental feedback candidate or change current predictions.

## Frozen matured-error feedback diagnostic

One candidate only; no parameter tuning after results:

- for each horizon, use at most 90 latest unique prior origins;
- prior truth is eligible only when `target_end + 1h < new_origin`;
- require at least 60 prior matured labels, otherwise unchanged incumbent;
- `correction = n/(n+90) * median(actual log return - original incumbent q50)`;
- compare corrected q50 with original incumbent and zero-return/no-change on identical valid origins;
- natural-log-return MAE/RMSE, yearly/regime breakdown;
- paired bootstrap using 90-calendar-day blocks, 1,000 draws, fixed seed 20260919;
- historical maturity is a conservative PROXY, not proof of an original data receipt;
- reused history is DIAGNOSTIC, not a newly untouched holdout. No promotion is authorized even if numerically better.

### Result — REJECTED, not activated

Input: verified final artifact 10578885278 from run 35422918168; ZIP SHA256 `0745e1fd2e59c329a8e98b22790739dd55e85334d3b4d855d9cdbc10c78efd9a`. Actual as-of: 2026-09-19T05:07:14Z. The archive remains the separately pinned historical reconstruction, not new real-time forecasts.

| Horizon | Identical valid origins | Original MAE skill vs no-change | Feedback MAE skill vs no-change | Feedback improvement vs original |
|---|---:|---:|---:|---:|
| 6h | 2643 | +0.457% | +0.426% | -0.031% |
| 24h | 2627 | +0.217% | +0.105% | -0.112% |
| 72h | 2601 | -1.100% | -1.210% | -0.108% |
| 7d | 2542 | -1.927% | -2.525% | -0.587% |
| 14d | 2265 | -2.515% | -3.021% | -0.493% |
| 30d | 2107 | -4.177% | -5.539% | -1.307% |

All six candidate comparisons deteriorated. Retain the result and the frozen implementation; do not tune this inspected sample until a pass appears. These natural-log-return losses must not replace earlier reports that used simple-return losses or different model/evaluation cohorts.

### Nominal historical replay gaps

At the source's declared 24-hour replay cadence, between each horizon's first and last stored origin: 6h 419, 24h 369, 72h 393, 7d 381, 14d 295, 30d 346 origins are absent from the pinned archive. This is not a count of known eligible missing forecasts; original exclusion/input/checkpoint reasons remain to be joined. Every actual missing timestamp is retained in the machine-readable report.

### Actual issued-record review at the pinned snapshot

Timely delivered and settled rows: 6h 174; 24h 166; 72h 143; 7d 102; 14d/30d 0. 6h upward barrier hits were caught in 1/32 windows; downward hits 1/36. 24h: upward 0/44, downward 0/26. Windows overlap and are NOT independent events. No improvement in actual prediction skill is claimed by this operational patch.

## Regression coverage / acceptance

Local new tests: 35 passing. They include missing state, timeout, partial writes, output errors despite zero process exit, historical row/outcome mutation, unauthorized horizon, mixed output generations, future-label perturbation invariance, strict maturity, duplicate origins, archive tampering, pipeline ordering, output-status separation and snapshot retention.

Repository-wide CI and the first post-merge public release must be verified separately. A successful first run does not establish the seven-day 99% timely-delivery operating target. No weights/targets/thresholds/rights/promotion gates or old forecasts are changed.

## Research basis for operations

GitHub documents scheduled-run delays/drops under high load and concurrency queuing semantics. Changing a lock name without redesigning shared writers risks lost state; this patch preserves the shared lock. `continue-on-error` changes the step conclusion, so the final gate explicitly uses `steps.<id>.outcome` to retain failure truth.

- https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule
- https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency
- https://github.com/actions/runner/blob/main/docs/adrs/0274-step-outcome-and-conclusion.md

## Next bounded work

Join missing replay origins with original input/checkpoint/eligibility evidence; incrementally recompute only genuinely eligible missing origins as retrospective research. Extend failure isolation to optional pre-issuance refreshes through transactional checkpoint installation. Study a new preregistered regime-conditioned information hypothesis, not repeated median-bias tuning; prospectively evaluate any eligible candidate separately. Source collection, source rights, historical skill and prospective skill stay distinct.
