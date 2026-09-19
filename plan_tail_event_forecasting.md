# ETH Tail-Event Forecasting Implementation Plan

Date: 2026-08-30 (implementation result updated 2026-08-31)

Status: **PR 1 implemented; historical Gate B rejected every core-only candidate**

No daily forecast, database, API, notification, or UI integration is approved
from this result.  The reusable offline evaluation foundation remains useful;
PR 2 and later rollout stages are stopped.

## Goal

Add a leakage-safe, computationally bounded warning for unusually strong ETH
upside moves without destabilizing the existing 7-day and 30-day forecasts.

The first deliverable is not a new central price champion or an automatic trade
signal. It is:

1. a calibrated probability that ETH will rise at least 12% over the next
   three days;
2. a separately validated asymmetric upper-return scenario;
3. a visible `TAIL_WATCH` state that can survive a flat normal-hybrid score but
   remains explicitly non-actionable during shadow validation.

The research basis and failure analysis are in
`research_tail_event_forecasting.md`.

## Approval defaults

Unless changed during review, implementation will use these defaults:

| Decision | Recommended default | Reason |
|---|---|---|
| Primary event | 3-day return `>= +12%` | About 95th-percentile magnitude, 184 rows and ~84 full-history event clusters |
| Robustness event | 3-day return `>= clip(2 * trailing_30d_vol * sqrt(3), 10%, 25%)` | Checks that results are not an artifact of one fixed-volatility era |
| Product behavior | Shadow-only first | Prevents a rare-event experiment from altering point forecasts or trades |
| Alert budget | At most 3 false-alert episodes per 90 OOF days | Makes recall comparable at an explicit operational cost |
| Models | Climatology, heuristic, logistic, CatBoost; optional LightGBM | Low variance and modest compute before deep models |
| New data | Free/official sources first; no paid subscription | Current head must prove value before data spend |
| Promotion evidence | Broad calendar walk-forward OOF plus live shadow | The August rally is already known and cannot be the sole test |

Approval of this plan authorizes code, tests, local evaluation artifacts, and a
reviewable PR. It does not authorize a paid data purchase or an actionable
trading override.

## Non-goals

- Do not replace the current 7-day regression or direction champion.
- Do not replace the 30-day `no_change_anchor` point champion.
- Do not loosen the existing `HIGH_CONFIDENCE` hybrid trading criteria.
- Do not claim that a tail alert predicts an exact price target.
- Do not train Conv-LSTM/VAE/Transformer models in the first implementation.
- Do not add sentiment/news scraping or paid provider dependencies.
- Do not run the normal full 7-day/30-day 36-fold suite daily.

## Output contract

The shadow result should have an independent namespace so it cannot be confused
with the existing classification probability.

Proposed fields:

| Field | Type | Meaning |
|---|---|---|
| `tail_event_mode` | string | `off`, `shadow`, or later `display` |
| `tail_up_horizon_days` | integer | Initially `3` |
| `tail_up_event_threshold_return` | float | Primary cutoff for the current origin |
| `tail_up_probability_raw` | float | Uncalibrated challenger score |
| `tail_up_probability` | float | Temporally calibrated probability |
| `tail_up_alert_threshold` | float | Fold-safe operating threshold |
| `tail_up_alert_state` | string | `NORMAL`, `ELEVATED`, `TAIL_WATCH`, or `UNAVAILABLE` |
| `tail_up_alert_reason` | string | Threshold, model, data readiness, and conflict summary |
| `tail_up_q90_return` | float/null | Conditional upper scenario if the interval gate passes |
| `tail_up_q95_return` | float/null | Conditional upper scenario if the interval gate passes |
| `tail_up_data_readiness` | string | `core_only`, `partial`, `ready`, or `blocked` |
| `tail_up_feature_groups` | list | Coarse feature groups used in the prediction |
| `tail_up_conflicts` | list | Existing top-reversal/downtrend signals shown as context, not silent vetoes |
| `tail_up_evidence_version` | string | Immutable OOF artifact/manifest identifier |

In `shadow` mode these fields may be persisted for evaluation but are not shown
as an actionable banner. Existing central returns, ranges, directions, and
hybrid tiers must remain byte-for-byte unchanged except for additive metadata.

## Target design

### Primary label

For origin date `t`:

```text
r3(t) = close(t + 3 days) / close(t) - 1
tail_up_primary(t) = 1 if r3(t) >= 0.12 else 0
```

The label is known only after `t+3`. Training and evaluation splits must use a
minimum three-day gap and embargo so adjacent outcomes cannot leak across the
boundary.

### Robustness label

```text
sigma30(t) = standard deviation of daily returns ending before t
dynamic_threshold(t) = clip(2 * sigma30(t) * sqrt(3), 0.10, 0.25)
tail_up_dynamic(t) = 1 if r3(t) >= dynamic_threshold(t) else 0
```

The volatility estimate is shifted so day `t` information is treated
consistently with the feature timestamp. No future forward returns may be used
to calculate the threshold at `t`.

The primary label decides the user-facing semantics. The dynamic label is a
robustness audit. A candidate that works only on the fixed label and collapses
on the dynamic label cannot be promoted beyond shadow.

### Episode grouping and weights

Consecutive positive origins around one rally are not independent events.

1. Sort positive origins by date.
2. Group origins separated by no more than three days into the same episode.
3. Give every positive row in an episode weight `1 / episode_row_count`.
4. Apply time decay after episode weighting.
5. Keep all negative rows for probability calibration; use class weights rather
   than random deletion or synthetic oversampling.

Evaluation reports both row-level and episode-level results. An episode is
“caught” if at least one alert was emitted during its eligible three-day origin
window. Adjacent alert rows inside one three-day forecast horizon collapse into
one alert episode. A continuously active alert is renewed every three days; it
cannot remain on for months and consume only one false-alert episode.

## Feature policy

### Core feature set first

The first model should not inherit all ~900 generated candidates. The event
sample is too small. Create a predeclared tail feature-group allowlist and cap
fold-selected features at 64.

Initial groups:

- ETH returns and trend at 1/3/5/7/14/30/60 days;
- realized volatility, volatility ratios, squeeze, range, ATR, and drawdown;
- volume impulse, volume z-score, OBV/CMF, VWAP distance, and gap/range features;
- ETH/BTC relative strength and BTC momentum/drawdown/volatility;
- available risk-on/risk-off macro composites with correct publication lag;
- current DefiLlama stablecoin, TVL, and DEX features only where fold coverage
  passes;
- the existing `eth_tail_event_pressure` as one feature and one standalone
  baseline, never as the target.

Fold ranking and imputation must be fitted only on outer-training rows.
Correlated-feature pruning must also be fold-local.

### Tail-specific source readiness

The normal 60-row vendor minimum is too permissive for this task. A new source
may enter a tail model only when all are true within the outer training fold:

- at least 730 non-null daily rows, unless the source is used only for a
  separately labeled short-history diagnostic;
- latest observation age no more than its declared publication lag plus three
  days;
- at least 20 positive event episodes exist in the source-covered training
  interval;
- no unresolved duplicate, revision, timezone, or as-of timestamp issue;
- coverage remains adequate in each evaluated calendar block.

If a source fails, the core-only model still runs and reports the exclusion.

## Model candidates

### Required baselines

1. **Expanding climatology**: probability from temporally prior labels, with
   optional half-life weighting.
2. **Heuristic baseline**: monotonically map `eth_tail_event_pressure` to a
   score, calibrating the map on prior folds only.
3. **Balanced logistic regression**: standardized, L2-regularized, episode- and
   time-weighted. This is the reference learned model.

### Bounded nonlinear challengers

4. **CatBoost classifier**: shallow trees, strong regularization,
   `SqrtBalanced` or explicit episode-aware weights, fixed seed, no file output,
   one thread cap suitable for CI.
5. **LightGBM classifier**: optional isolated challenger with small leaves and
   class weighting. Its earlier central-regression failure neither promotes nor
   disqualifies it for this different target.

Focal loss is an optional ablation only after ordinary weighted log loss has a
valid baseline. No custom loss should enter the first PR if it prevents stable
probability calibration.

### Probability calibration

- Generate calibration predictions with inner purged time splits inside each
  outer training fold.
- Compare uncalibrated probability with sigmoid/Platt calibration.
- Constrain calibration to preserve ranking. If unconstrained Platt has a
  negative slope, keep slope one and fit only a prior-prevalence intercept;
  reject a calibrated threshold window whose probabilities are constant.
- Do not use isotonic calibration unless the calibration window has enough
  positive episodes; its small-sample step function is a known overfit risk.
- Select the calibration method from training evidence only.
- Persist raw and calibrated scores so calibration can be audited separately
  from ranking.

## Tail interval candidate

The interval experiment is separate from event classification.

1. Train 3-day return quantile models at `q90` and `q95` using quantile loss.
2. Start with HistGradientBoosting or CatBoost quantile regression; do not add a
   deep sequence dependency.
3. Use nested walk-forward residuals to conformalize the upper quantiles.
4. Compare a fixed calibration window, volatility-scaled residuals, and one
   adaptive online conformal update.
5. Report marginal upper coverage, coverage in positive-tail episodes,
   interval width, exceedance size, and weighted interval score.

The `q95` value is initially an **upside scenario**, not a replacement for the
normal prediction interval. Only after its gate passes may the frontend show it
or the range layer take the maximum of its current upper bound and the tail
bound. The lower bound and central point remain untouched.

## Validation design

### Why the normal recent short fold is insufficient

The latest three years contain only about 28 fixed-label positive rows and 12
event clusters. A three-fold daily selection can easily be dominated by one
rally. Tail evaluation needs broad historical calendar coverage while keeping
strict time order.

### Gate A — static and last-six-month smoke

- Unit/leakage tests.
- Six 30-day outer test blocks at the end of history.
- Gap and embargo: three days.
- Models: climatology, heuristic, logistic, and at most one nonlinear model.
- Purpose: reject crashes, inversions, degenerate probabilities, and excessive
  runtime; never promote from this gate.

Stop immediately if:

- any fold has no valid prior-only threshold/calibration path;
- a feature changes when only future source rows are mutated;
- probabilities contain NaN/inf outside an explicitly unavailable state;
- the run exceeds its compute budget.

### Gate B — authoritative broad walk-forward OOF

Use expanding calendar blocks so rare events across different market eras are
represented with a small number of model fits:

- outer tests: calendar-year or year-like blocks from 2019 through the latest
  partial year;
- outer training: all history strictly before each test block, with time decay;
- gap/embargo: at least three days at each boundary;
- inner calibration/threshold selection: purged time splits drawn only from the
  outer training interval;
- report each calendar block separately and an episode-clustered aggregate.

This broad OOF design is the promotion authority for the tail head. The existing
36-fold 7/30-day evidence remains authoritative for the normal products.

### Gate C — data-source ablation

Run the winning algorithm with identical folds and seeds on:

1. core market only;
2. core + macro;
3. core + currently ready crypto-native data;
4. core + each newly backfilled source group;
5. all groups that individually survived.

No source enters the selected manifest merely because its columns are present.
It must add stable OOF value without reducing the usable historical window to a
small recent regime.

### Gate D — live shadow

After historical gates pass:

- persist one shadow probability and upper-tail scenario per daily run;
- backfill the three-day outcome and episode mapping when mature;
- do not alter existing forecast values, hybrid tiers, or notifications;
- expose the result only in diagnostics/model-evaluation JSON at first.

A later actionable integration review requires at least 90 calendar days and
two matured positive episodes. If two episodes do not occur, the model remains
shadow; elapsed time alone is not evidence.

## Metrics and proposed promotion thresholds

These thresholds are deliberately predeclared before model results are seen.

### Probability ranking and calibration

The candidate must satisfy all of the following on Gate B:

- average precision at least 20% above expanding-climatology prevalence;
- average precision not below the heuristic/logistic reference;
- positive Brier skill versus expanding climatology;
- episode recall at least 35% at no more than three false-alert episodes per
  90 evaluated days;
- bootstrap probability of improvement over the best baseline at least 90%
  for shadow entry;
- no probability inversion and no calendar block with an unexplained gross
  calibration failure.

The report must include ROC AUC and balanced accuracy for continuity, but they
are secondary. Primary evidence is average precision, Brier skill, episode
recall, precision, false-alert rate, and lead time.

### Robustness label

On the volatility-normalized label:

- Brier skill must remain non-negative;
- episode recall at the same false-alert budget may fall by no more than 10
  percentage points relative to the primary label;
- a model that reverses rank or becomes overconfident is shadow-only even if
  the primary label gate passes.

### Upper-tail scenario

For a displayed `q95` scenario:

- overall OOF upper coverage must be at least 92%;
- positive-tail upper coverage must improve by at least 15 percentage points
  over the current interval path;
- median upper width may not expand by more than 40% without a weighted interval
  score improvement;
- the block-bootstrap probability of a weighted interval score improvement
  must be at least 90%.

If classification passes and the interval does not, publish only the shadow
probability; leave both the normal range and tail scenario unset.

### Production/display promotion

Promotion from historical evidence to public `TAIL_WATCH` display requires:

- all Gate B/C criteria;
- live shadow calibration within the predeclared tolerance;
- at least 95% bootstrap probability of improvement after the live evidence is
  included;
- zero changes to the existing point forecast in shadow regression tests;
- explicit user approval of the display/notification wording.

Promotion to an actionable hybrid override is a separate future plan and PR.

## Alert semantics and hybrid interaction

The event head must not be vetoed by a heuristic reversal label, because that
would recreate the August suppression. Instead:

- normal hybrid direction and reversal become `tail_up_conflicts` context;
- `TAIL_WATCH` may appear when its calibrated threshold passes even if the
  normal hybrid bias is `FLAT`;
- the alert text states that it is a low-base-rate risk scenario, not a central
  forecast or trade instruction;
- no `HIGH_CONFIDENCE`, buy, sell, or position-size field is changed;
- notification integration is disabled in shadow mode.

Suggested states:

| State | Rule | User meaning |
|---|---|---|
| `UNAVAILABLE` | Evidence/data manifest missing or stale | No tail estimate |
| `NORMAL` | Probability below the lower display threshold | No unusual upside-tail evidence |
| `ELEVATED` | Probability above climatology but below alert threshold | Risk is rising, below alert operating point |
| `TAIL_WATCH` | Calibrated probability passes its fold-safe threshold | Unusual +12%/3-day scenario deserves attention; not a trade signal |

No numeric threshold is hard-coded from the August case. It is selected inside
each training fold subject to the false-alert budget, then frozen in the final
evidence manifest.

## Data-source work plan

Data work is staged so the model question is answered before a large collector
expansion.

### Source phase 1 — Coin Metrics historical on-chain ablation

Candidate columns:

- active address count;
- transaction and transfer counts;
- total fees;
- MVRV;
- exchange inflow/outflow in native units and USD;
- exchange-held supply;
- spot volume.

Required controls:

- confirm CC BY-NC 4.0 compatibility before production use;
- record URL, retrieval time, checksum, schema, and latest source date;
- block live use when stale, even if the history is useful for OOF;
- lag by the observed daily publication schedule;
- treat historical revisions as a source-version change that invalidates cache
  and evaluation manifests.

### Source phase 2 — OKX historical funding

- Backfill ETH perpetual funding from the official March 2022 archive.
- Normalize settlement intervals before daily aggregation.
- Preserve venue and contract identifiers; do not silently merge USDT and USD
  contracts.
- Derive funding level, change, z-score, persistence, and sign-flip features.
- Compare against existing Binance/Deribit funding wherever dates overlap.

### Source phase 3 — Binance futures microstructure proxies

- Use official checksummed daily/monthly futures trades or sub-daily klines.
- Derive intraday realized volatility, high-low range, downside/upside
  semivariance, volume concentration, taker-buy share where available, and
  late-session momentum.
- Aggregate in UTC with all inputs ending before the daily forecast cutoff.
- Do not label this as OI, liquidation, or options data; it is an order-flow and
  realized-volatility proxy.

### Source phase 4 — continue prospective Deribit/DefiLlama history

- Keep existing Deribit funding, future, and option snapshots accumulating.
- Extend free DefiLlama fees/open-interest overview only after schema and history
  checks.
- Re-evaluate readiness after at least 730 days or enough independently sourced
  historical backfill; do not force short-history features into OOF.

Paid Kaiko/Amberdata/Glassnode-class data is out of scope unless free-source
ablation leaves a clear residual need and the user separately approves cost.

## File-level implementation map

The preferred design keeps tail logic out of the already large main script.

### New files

- `forecasting/tail_events.py`
  - label and dynamic-threshold builders;
  - episode grouping/weights;
  - candidate registry and calibration helpers;
  - alert-state/result dataclasses;
  - upper-quantile/conformal helpers.
- `scripts/evaluate_tail_event_candidate.py`
  - Gate A/B/C runner, metrics, bootstrap, gate decision, and strict JSON.
- `tests/phase0/longrun_oof_tail_event.py`
  - resumable broad calendar walk-forward generation.
- `tests/test_tail_events.py`
  - target, episode, split, calibration, alert, and interval unit tests.
- `tests/test_evaluate_tail_event_candidate.py`
  - gate and failure-mode tests.
- `tests/phase0/tail_event_baseline_manifest.json`
  - frozen label/data/split/baseline definition after the first accepted run.

### Existing files to change

- `eth_price_forecast.py`
  - call the tail module only when enabled;
  - append shadow output without changing existing forecast calculations;
  - expose the fixed tail feature allowlist inputs.
- `forecasting/model_registry.py`
  - add isolated tail candidate parameters only if keeping the registry central
    is cleaner than a tail-local registry.
- `eth_data_collector.py`
  - add approved source collectors, immutable raw caches, lags, and
    tail-specific readiness fields.
- `forecast_site/schema.sql`
  - add nullable tail forecast and matured-outcome fields through an idempotent
    migration.
- `forecast_site/persist_forecast.py`
  - persist shadow result/evidence version.
- `forecast_site/backfill_actuals.py`
  - resolve 3-day returns, labels, episode membership, and alert outcomes.
- `forecast_site/export_json.py`
  - publish diagnostics fields; hide display fields while mode is `shadow`.
- `forecast_site/export_candidate_eval.py`
  - expose the tail OOF gate independently of normal model evaluation.
- `.github/workflows/eth_model_eval.yml`
  - add explicit `tail_event_smoke` and manual `tail_event_full` paths.
- `.github/workflows/daily_forecast.yml`
  - run only the accepted lightweight shadow fit/inference and cheap committed
    evidence re-score; never the broad OOF gate.
- `forecast_site/public/index.html`
  - later display-only PR, after live shadow approval.

## Test plan

### Leakage and label tests

- Mutating closes after `t+3` cannot change label/threshold/features at `t`.
- Mutating vendor rows after `t` cannot change features at `t`.
- Dynamic volatility threshold is shifted and uses no forward return.
- Split gap/embargo removes overlapping targets at every boundary.
- Feature selection, imputation, scaling, calibration, and alert thresholds are
  fitted on outer-training data only.

### Episode and metric tests

- Adjacent positive origins collapse into one event episode.
- Positive episode weights sum to one per episode before time decay.
- Adjacent alerts collapse into one alert episode.
- False-alert budget is calculated on negative episodes/time, not raw duplicate
  rows.
- Average precision, Brier skill, episode recall, lead time, and bootstrap are
  stable on synthetic fixtures with known outcomes.

### Calibration and interval tests

- Raw and calibrated scores are both preserved.
- Constant/all-one/all-zero fold edge cases fail explicitly rather than emit
  fake probabilities.
- Upper quantile ordering (`q90 <= q95`) is enforced or the fold fails.
- Conformal calibration never reads the outer test residual.
- Coverage and width gates reject an interval that widens indiscriminately.

### Integration and regression tests

- With `tail_event_mode=off`, existing JSON and forecast values are unchanged.
- With `shadow`, central point/range/direction/hybrid fields are unchanged.
- Missing or stale evidence produces `UNAVAILABLE`, not a default zero risk.
- DB migrations are idempotent and old databases remain readable.
- Three-day actuals backfill exactly once and are restart-safe.
- Resume manifests invalidate when labels, features, model parameters, source
  versions, folds, or calibration settings change.
- Daily workflow never invokes the full tail OOF command.

### Collector tests

- official sample payload/ZIP parsing;
- checksum mismatch and schema drift failure;
- timezone and publication-lag alignment;
- bounded forward fill and stale-source exclusion;
- append-only/idempotent cache behavior;
- source revision report and evidence-manifest invalidation;
- licensing/freshness gate surfaced in the data quality audit.

## Evaluation artifacts

Every run should produce:

- `tail_event_oof_predictions.csv` — origin, target date, primary/dynamic label,
  episode ID, raw/calibrated score, threshold, alert, model, and fold;
- `tail_event_metrics.json` — row, episode, calendar-block, label-robustness,
  calibration, interval, compute, and data-readiness metrics;
- `tail_event_eval_report.md` — readable gate decision and failure reasons;
- `tail_event_feature_ablation.json` — identical-fold source-group comparison;
- `tail_event_manifest.json` — code SHA, source versions/checksums, label,
  features, folds, estimators, calibration, thresholds, and environment;
- PR artifact upload even on failure.

Strict JSON conversion must map non-finite values to `null`; the earlier compact
candidate's NaN serialization failure must not recur.

## Compute budget and stop rules

Target budgets on a 2-core GitHub runner are estimates and become hard timeout
controls during implementation:

| Work | Budget | Frequency |
|---|---:|---|
| Unit/static tests | under 5 minutes | every PR |
| Gate A smoke | under 10 minutes | relevant PRs |
| Gate B broad tail OOF | under 45 minutes | manual/model change only |
| One source ablation | under 20 minutes | manual after readiness |
| Accepted daily shadow head | under 5 additional minutes | daily |

Cost controls:

- run only the 3-day tail task, not 7/30 regression/direction families;
- reuse immutable feature frames and baseline OOF caches when manifests match;
- cap candidate count and 64 selected features;
- stop after Gate A failure;
- stop source ablations individually when their readiness gate fails;
- never auto-launch normal 36-fold evaluation from a tail-only change;
- record wall time and peak memory in the evaluation artifact.

If the daily addition exceeds five minutes or 1.5 GB peak incremental memory,
keep it out of the daily job and use a pre-trained/frozen shadow artifact until
a cheaper inference path is reviewed.

## Rollout sequence

### PR 1 — labels, metrics, and core-only offline gate

- Add `forecasting/tail_events.py` label/episode/metric foundations.
- Add strict leakage tests and Gate A/B offline evaluator.
- Run baselines, logistic, CatBoost, and optional LightGBM on core features.
- Do not touch daily output, DB, or UI.
- Stop if no candidate passes the historical gate.

### PR 2 — asymmetric interval challenger

Only if PR 1 probability evidence passes:

- add `q90/q95` return candidates and conformal calibration;
- compare fixed, scaled, and adaptive residual updates;
- keep scenario hidden when its independent gate fails.

### PR 3 — approved data enrichment

Only if the core head shows skill and the source terms/freshness are acceptable:

- add Coin Metrics and/or OKX backfill;
- add source quality and as-of controls;
- run identical-fold source ablations;
- promote only groups with stable incremental evidence.

### PR 4 — daily shadow persistence

Only after the winning manifest is approved:

- add `off|shadow` runtime mode, DB migration, persistence, actual backfill, and
  diagnostic export;
- enforce zero-delta tests for existing normal forecast fields;
- no UI alert and no Telegram notification.

### PR 5 — optional public display

Only after live shadow criteria and a second user approval:

- add `display` mode and clearly labeled `TAIL_WATCH` card/scenario;
- show conflicts and evidence freshness;
- preserve non-actionable wording;
- retain one-switch rollback to `shadow` or `off`.

## Rollback

- Default `ETH_TAIL_EVENT_MODE=off` until an evidence manifest is committed.
- `shadow` and `display` are additive; turning them off leaves all existing
  forecasts operational.
- A stale/missing/mismatched manifest forces `UNAVAILABLE` automatically.
- DB columns are nullable and additive, so rollback does not require destructive
  migration.
- Collector source failure falls back to the last valid core-only tail model,
  not stale vendor values.
- No central model registry promotion is changed by these PRs.

## Definition of done for the first implementation cycle

The cycle is complete when one of two honest outcomes is reached:

### Pass outcome

- PR 1 historical gates pass for a reproducible core-only candidate;
- all leakage, episode, calibration, and integration tests pass;
- compute stays inside budget;
- artifacts and manifest are reviewable;
- user approves moving to interval/data enrichment or daily shadow.

### Reject outcome

- no candidate clears the predeclared historical gate, or results depend on one
  era/event/source;
- the rejected experiment and metrics are committed as evidence;
- no daily/UI behavior changes;
- further model complexity is not added without a new reviewed hypothesis.

## PR 1 implementation result (2026-08-31)

The reject outcome was reached on production data through 2026-08-29.

Implemented and verified:

- fixed `+12% / 3-day` and shifted-volatility robustness labels;
- episode-normalized/time-decayed sample weights;
- prior-only sigmoid calibration and alert-threshold selection;
- three-day alert renewal, event recall/precision, false-alert cost, AP, Brier
  skill, AUC, balanced accuracy, and moving-block bootstrap;
- six recent 30-day Gate A folds and eight expanding calendar Gate B folds
  (2019 through the 2026 partial year), all with a three-day purge;
- a fixed 64-feature core allowlist and at most one nonlinear model per run;
- strict JSON/Markdown evidence and unit/leakage/gate tests.

Primary-label Gate B results on 2,795 matched OOF dates:

| Model | AP | Brier skill vs climatology | Episode recall | False alerts / 90d |
|---|---:|---:|---:|---:|
| Expanding climatology | 0.05108 | 0.00% | 0.00% | 0.000 |
| Tail-pressure logistic | 0.06055 | +0.29% | 33.85% | 5.088 |
| Episode logistic | **0.07322** | -1.29% | 20.00% | 1.449 |
| CatBoost | 0.05297 | -0.84% | 27.69% | 4.959 |
| LightGBM | 0.05791 | -1.25% | 13.85% | 2.544 |

The AP winner was episode logistic, but it failed positive Brier skill, the 35%
episode-recall floor, and the 90% paired-bootstrap requirement. Its paired
Brier improvement probability versus the tail-pressure reference was only
16.5% (AP-improvement probability 79.5%). On the dynamic robustness label its
Brier skill was -16.91%, so the robustness gate also failed.

CatBoost Gate B completed in 340.7 seconds at about 468 MB peak RSS; LightGBM
completed in 198.3 seconds at about 363 MB. Both stayed well inside the compute
budget. None of the learned candidates alerted on the known 2026-08-16 through
2026-08-18 positive origins under the corrected alert accounting.

Decision: keep only the offline PR 1 framework, commit no accepted model
manifest, do not start the interval/data-enrichment/daily-shadow PRs, and do not
tune more models against the already inspected August event without a new
reviewed hypothesis.

## Approval boundary

The plan was approved for **PR 1 only**. PR 1 changed only offline labels,
episode-aware metrics, tests, and the core-only evaluator. The failed Gate B
keeps forecasting, collector, database, and UI integration outside the approved
boundary; later PRs remain blocked.
