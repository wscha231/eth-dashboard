# ETH Lead-Signal and Adaptive-Interval Implementation Plan

Date: 2026-08-31

Status: **PR 2 implemented offline; full-source contract passed; PR 3 next**

Research basis: `research_lead_signal_forecasting.md`

## Goal

Test whether new leading information can improve three-day ETH upside-tail
warnings and whether adaptive asymmetric intervals can prevent severe upper
range undercoverage during regime shifts.

This plan does not approve an operational model change. It approves only
offline data construction, leakage-safe evaluation, and review artifacts in
the staged order below.

## Default decisions for review

| Decision | Recommended default |
|---|---|
| Primary new data | Binance 1h spot/perpetual klines for ETH and BTC + full DefiLlama history |
| Secondary data | OKX funding; continue Deribit prospective snapshots |
| On-chain exchange flow | Coin Metrics research-only, never production without license approval |
| Primary event | Existing terminal 3-day `>= +12%` for comparability |
| Factorized event | `abs(3-day return) >= 12%` head times a 3-day direction head |
| Interval targets | 3-day and 7-day return `q05/q50/q95` |
| First models | Logistic, CatBoost/HistGB, one LightGBM challenger |
| Deep model | Deferred until source ablation passes |
| Deployment | None; offline only until every gate passes |

## Hard boundaries

- Do not modify the existing 7-day or 30-day promoted model manifests.
- Do not modify the daily forecast JSON contract, database schema, site, or
  notifications in PRs 1-4.
- Do not merge PR #9's rejected tail model into the daily path.
- Do not use future bar fragments, revised macro values without as-of handling,
  or test folds in preprocessing.
- Do not add paid data or accept commercial-use license restrictions without
  separate user approval.
- Do not tune against the August 2026 event.

## Output namespaces

Offline reports should keep predictive probability and interval evidence
separate:

```text
lead_signal/
  data_manifest
  source_readiness
  direct_tail
  factorized_tail
  multiclass_tail
  source_ablation

adaptive_interval/
  quantiles
  conformal_method
  coverage
  tail_exceedance
  regime_breakdown
```

No field enters public forecast output before a later shadow-integration plan.

## PR 1 execution result (2026-08-31)

The source-feasibility implementation is complete on an isolated branch.

| Source | Result | Evidence |
|---|---|---|
| Binance ETH/BTC spot + USD-M 1h | Ready for offline feature work | Four archive listings were continuous through 2026-07; 12 bounded first/latest/2025-transition ZIPs passed upstream/local SHA-256, timestamp, OHLCV, duplicate, and missing-bar checks |
| DefiLlama Ethereum | Ready for offline feature work | TVL 3,260 days, stablecoins 3,197 days, and DEX volume 2,859 days through 2026-08-30; no missing or duplicate daily rows |
| Deribit ETH funding | Feasible secondary source | One-hour funding rows were returned in 2019, 2023, and 2026, disproving the prior assumption that the endpoint retained only about 90 days; snapshot book summaries remain prospective-only |
| OKX funding | Excluded for now | The official page documents history from 2022-03, but the API timed out in the execution environment and stable automated archive URLs/terms remain unresolved |

The strict readiness decision is `pass_for_pr2_offline`, with production use
explicitly false. Evidence is stored in
`lake/manifests/lead_signal_sources.json` and
`lake/reports/lead_signal_source_readiness.json`. No model training, daily
collector wiring, database, site, or public JSON change was made.

## Staged implementation

### PR 1 - source feasibility and immutable manifests

Purpose: prove that historical data can be obtained and aligned before writing
new model code.

Actions:

1. Add a bounded downloader for monthly Binance 1h klines:
   - `ETHUSDT` spot;
   - `ETHUSDT` USD-M futures;
   - `BTCUSDT` spot;
   - `BTCUSDT` USD-M futures.
2. Verify upstream `.CHECKSUM`, then calculate local SHA-256.
3. Detect timestamp-unit changes, duplicate open times, missing bars,
   non-monotonic rows, impossible OHLC, and volume inconsistencies.
4. Record launch dates and expected gaps per symbol/venue.
5. Perform a one-time full-history DefiLlama backfill for Ethereum stablecoin
   supply, chain TVL, and DEX volume.
6. Add an OKX funding feasibility probe with contract and settlement-interval
   metadata; do not download L2 order books yet.
7. Re-audit Deribit funding history limits and keep snapshot-only fields
   explicitly marked prospective.
8. Produce a strict JSON readiness report. No model training in this PR.

Proposed files:

- `scripts/backfill_lead_signals.py` - new offline downloader/validator;
- `forecasting/lead_signal_data.py` - new schemas and alignment helpers;
- `tests/test_lead_signal_data.py` - checksum, timestamp, duplicate, and cutoff tests;
- `lake/reports/lead_signal_source_readiness.json` - generated evidence, not hand-edited;
- `lake/manifests/lead_signal_sources.json` - source versions and checksums.

`eth_data_collector.py` remains unchanged until the offline path proves usable.

Stop PR 1 if:

- GitHub Actions cannot access both a primary and fallback archive route;
- upstream data terms are incompatible with the intended use;
- hourly coverage has unresolved missing/duplicate periods material to event
  windows;
- the derived daily table cannot be reproduced from the same manifest.

### PR 2 - fold-safe feature aggregation and target extension

Purpose: convert validated hourly sources into compact, prior-only daily
features and add the factorized targets.

Hourly-to-daily feature groups:

1. **Order flow**
   - taker-buy quote share;
   - signed taker quote flow `(2 * taker_buy_quote - quote_volume)`;
   - spot/perpetual flow divergence;
   - 1/3/7-day changes and fold-local z-scores.
2. **Leverage/basis**
   - perpetual/spot close basis;
   - basis change and persistence;
   - futures/spot volume and trade-count ratios.
3. **Intraday risk state**
   - hourly realized volatility;
   - upside/downside semivariance;
   - jump-variation proxy;
   - final 4/8/12-hour return and volume share.
4. **Cross-asset leadership**
   - BTC flow and risk state;
   - ETH-minus-BTC flow, momentum, volatility, and basis spreads.
5. **Ethereum liquidity**
   - stablecoin supply growth/acceleration;
   - DEX volume z-score and volume/TVL;
   - TVL growth and TVL/ETH-market-cap.

Target helpers:

- retain `tail_up_primary = 1[r3 >= 0.12]`;
- add `large_move_primary = 1[abs(r3) >= 0.12]`;
- add `direction_up = 1[r3 > 0]`;
- add `UP_TAIL / NORMAL / DOWN_TAIL` multiclass label;
- generate barrier labels for diagnostics only;
- reuse three-day purge/embargo and episode grouping.

All daily features for forecast date `t` must use hourly bars whose close time
is no later than the declared UTC cutoff. Aggregation tests mutate future and
partial current-day bars and require every feature at `t` to remain unchanged.

Proposed files:

- `forecasting/lead_signals.py` - new feature and result dataclasses;
- `forecasting/tail_events.py` - additive target helpers only;
- `tests/test_lead_signals.py`;
- `tests/test_tail_events.py` - additive target/episode cases.

Implementation result (2026-09-03):

- all 374 checksum-pinned monthly archives validated through the common
  complete month `2026-07`;
- the immutable feature table contains 3,269 daily rows and 177 columns from
  `2017-08-18` through `2026-07-31`;
- the four-stream common-hourly window contains 2,388 eligible days from
  `2020-01-01`, making `2021-12-31` the first authoritative PR 3 test date;
- 452 declared historical hourly-grid anomalies affecting 34 UTC dates are
  quarantined as missing market days rather than silently repaired;
- the five disjoint model groups contain 28 order-flow, 14 leverage/basis,
  20 intraday-risk, 88 cross-asset-leadership, and 18 Ethereum-liquidity
  columns;
- the readiness decision is `pass_for_pr3_offline_evaluation`, while
  production use, model training, and public-contract changes remain false.

### PR 3 - source-ablation Gate A and Gate B

Purpose: answer whether the new information adds OOF value. No daily wiring.

Candidate families:

1. existing direct core-only tail baseline;
2. direct model + order-flow group;
3. direct model + Ethereum-liquidity group;
4. direct model + order flow + liquidity;
5. factorized large-move/direction model on the same groups;
6. direct three-class model;
7. optional OKX/Deribit ablations only where coverage passes.

Algorithms:

- expanding climatology/no-signal;
- regularized logistic regression;
- shallow CatBoost or HistGradientBoosting;
- one predeclared LightGBM configuration.

Do not search a large hyperparameter space. At most one fixed configuration
per nonlinear family plus one training-only regularization choice.

#### Gate A - engineering smoke

- last six 30-day blocks;
- three-day purge and embargo;
- source manifests and common-date alignment verified;
- maximum ten minutes and 1 GB peak RSS;
- never promote from this gate.

#### Gate B - authoritative OOF

- expanding calendar-year blocks;
- source-augmented authority starts only after 730 days of prior source history;
- compare every ablation on exactly the same test dates;
- training-only calibration and alert-threshold selection;
- event-clustered block bootstrap;
- calendar and volatility-regime breakdowns.

Primary promotion checks:

```text
AP >= 1.20 * matched_core_AP
Brier skill versus matched core > 0
episode recall >= 0.35
false alerts <= 3 per 90 OOF days
P(paired improvement) >= 0.90
improvement in >= 4 calendar blocks
largest single-block contribution <= 50% of aggregate gain
dynamic-label Brier skill >= 0
```

The factorized probability must be recalibrated as a final score; multiplying
two individually calibrated heads is not assumed to remain calibrated.

Proposed files:

- `scripts/evaluate_lead_signal_candidate.py`;
- `tests/phase0/longrun_oof_lead_signals.py`;
- `tests/test_evaluate_lead_signal_candidate.py`;
- `tests/phase0/lead_signal_gate_result.json`.

If every candidate fails, retain the data layer and evaluator but stop. Do not
move to a deep model to rescue a negative source-ablation result.

### PR 4 - adaptive asymmetric interval challenger

Purpose: improve upper-range reliability independently of directional alpha.

Candidates:

1. current central forecast + current conformal range;
2. `q05/q50/q95` quantile boosting + conformalized quantile residuals;
3. volatility-scaled online conformal;
4. change-point-aware conformal using existing regime probabilities and a
   small state-transition model.

Evaluation horizons:

- 3-day first, because it matches tail-event diagnostics;
- 7-day second if the 3-day implementation is sound;
- 30-day only after the shorter horizons pass, due much smaller effective OOF
  sample and overlap.

Metrics:

- pinball loss per quantile;
- weighted interval score;
- marginal lower/upper coverage;
- average and median width;
- coverage and exceedance size during up-tail/down-tail episodes;
- coverage by calendar year and volatility regime;
- first 1/3/7 observations after detected change points.

Shadow-entry checks:

- lower WIS or q95 pinball loss on matched OOF dates;
- overall upper coverage close to nominal;
- fewer/smaller upside-tail exceedances than the current range;
- no more than 25% median-width increase unless tail-exceedance reduction is
  large and stable;
- paired-bootstrap improvement probability at least 90%;
- no calendar block with unexplained severe undercoverage.

The new upper quantile remains an additive scenario. It cannot replace the
central point or widen the public range in this PR.

Proposed files:

- `forecasting/adaptive_intervals.py`;
- `scripts/evaluate_adaptive_interval_candidate.py`;
- `tests/test_adaptive_intervals.py`;
- `tests/phase0/adaptive_interval_gate_result.json`.

### PR 5 - optional hourly sequence challenger

This PR is authorized only if at least one aggregated source group passes PR 3.

Compare one compact causal 1D-CNN/TCN against the winning tree model using the
same 72-hour source tensors, folds, labels, and dates. Conv-LSTM is an optional
second challenger only if the 1D-CNN demonstrates incremental value.

Controls:

- causal windows only;
- scaler and any representation learning fitted inside each outer fold;
- fixed small architecture and seed set;
- no bidirectional layer;
- no VAE unless used as a separately evaluated anomaly score;
- no SHAP on raw correlated lags; use grouped ablation or grouped permutation;
- maximum 45-minute authoritative run and 2 GB peak RSS.

The deep challenger must beat the simpler model, not merely no-change, and
must pass the same probability/interval gates.

### PR 6 - live shadow, only after historical passage

Prepare a separate plan before this step. Minimum boundaries:

- additive diagnostic persistence only;
- no UI banner, alert, trade tier, or notification;
- mature outcomes backfilled after three/seven days;
- at least 90 live days and two positive episodes before any display review;
- automatic fallback to `UNAVAILABLE` on source or manifest drift.

## Storage and compute design

### Storage

- Do not commit tick trades or L2 order books.
- First stage downloads only monthly 1h klines.
- Persist compact daily feature tables and source manifests.
- Keep archive URLs and upstream/local checksums for reproducibility.
- If raw hourly retention is required, use compressed Parquet outside normal
  Git history; decide the durable location before implementation.

Expected first-stage scale is a few hundred thousand hourly rows, not millions
of ticks.

### Compute budgets

| Job | Budget |
|---|---:|
| Source preflight | <= 10 min |
| Incremental daily aggregation | <= 2 min |
| Gate A | <= 10 min, <= 1 GB RSS |
| Gate B simple models | <= 30 min, <= 1 GB RSS |
| Adaptive interval gate | <= 30 min |
| Optional deep challenger | <= 45 min, <= 2 GB RSS |

The normal daily forecast workflow never runs Gate B or the optional deep
challenger.

## Tests required before any PR is reviewable

### Data integrity

- official and local checksum validation;
- timestamp unit change around 2025 Binance spot archives;
- duplicate and missing hourly bars;
- OHLC and volume invariants;
- upstream archive revision detection;
- funding interval/formula normalization;
- exact UTC cutoff and no partial-day inclusion;
- deterministic daily aggregation from a frozen manifest.

### Leakage

- future rows mutated with no change to earlier features;
- scaler/imputer/ranker fitted only on outer training rows;
- three-day or horizon-specific purge at every boundary;
- calibration and alert thresholds derived only from prior OOF predictions;
- source date versus availability timestamp kept distinct;
- final as-of manifest persisted in every result.

### Evaluation

- common-date baseline enforcement;
- episode counting and continuous-alert renewal;
- factorized final calibration;
- block-bootstrap reproducibility;
- calendar contribution accounting;
- strict JSON serialization and finite probabilities/intervals.

## Review checkpoints

1. **After PR 1:** approve usable sources and storage before feature code.
2. **After PR 2:** approve exact feature/label contract before model runs.
3. **After Gate A:** review only engineering validity; no performance decision.
4. **After Gate B:** decide stop versus interval work and optional deep model.
5. **After PR 4:** decide whether improved uncertainty alone merits live shadow.

## Next approval scope

Review the exact **PR 2** feature and label contract. The next bounded step is
PR 3 matched-date source ablation: Gate A first, then Gate B only if the
engineering gate passes. Deribit remains optional and OKX remains excluded
until its access and terms gates are resolved.

No model, daily forecast, or site behavior changes are authorized by PR 2.
