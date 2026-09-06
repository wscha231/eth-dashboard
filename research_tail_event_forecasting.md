# ETH Tail-Event Forecasting Research

Date: 2026-08-30 (experiment result updated 2026-08-31)

Evidence snapshot: production data through 2026-08-29

## Executive conclusion

The daily pipeline is updating data and refitting the normal 7-day and 30-day
forecast suite. The expensive 36-fold promotion gate is intentionally not run
every day. The recent rally was nevertheless missed in the way that matters to
the user:

- the 7-day direction classifier said `UP` on five of six examined dates;
- the central regression forecast averaged only `+0.43%` while the realized
  return averaged `+30.68%`;
- the average 90th-percentile upper return was only `+10.56%`, with `0/6`
  interval coverage;
- the hybrid gate emitted three `OBSERVE` and three `NO_SIGNAL` decisions, so
  no actionable warning survived.

This is primarily an **objective and integration mismatch**, not proof that one
more generic price model will solve the problem. The current models optimize
ordinary direction and central-return errors; they do not directly estimate the
probability of an upside tail event. The existing `eth_tail_event_pressure` is
a past-only volatility heuristic, not a supervised future-tail target.

The recommended next experiment is therefore an isolated, low-cost
**3-day upside-tail probability head**, paired with an asymmetric upper-tail
quantile forecast. It should run in shadow mode first and must not replace the
7-day/30-day point champions. On the current dataset, a transparent primary
event definition of `3-day return >= +12%` produces 184 positive rows and about
84 overlapping-event clusters over the full history. The recent rally is
positive on 2026-08-16, 17, and 18 under this definition. A 7-day `+20%` target
has only about 44 event clusters, which is too sparse to make the first
production target.

Deep CNN/LSTM/VAE models from the attached papers should remain later-stage
challengers. Their useful ideas are multivariate inputs, quantile/uncertainty
forecasting, and interpretability. Their reported price-level accuracy is not
directly transferable to this repository's leakage-safe 3/7/30-day return task,
and the available independent tail-event sample is too small to justify the
extra capacity or daily compute.

### Subsequent core-only experiment result

The approved first-stage experiment was implemented and rejected by its broad
historical gate. Across 2,795 expanding-calendar OOF dates from 2019 through the
2026 partial year, the best AP came from episode-weighted logistic regression
(0.07322 versus 0.05108 for expanding climatology), but Brier skill was -1.29%
and only 20.0% of 65 primary event episodes were detected. Its paired Brier
improvement probability over the tail-pressure reference was 16.5%, far below
the predeclared 90% requirement. The volatility-normalized label had -16.91%
Brier skill. CatBoost and LightGBM also failed, and none alerted on the known
2026-08-16 through 2026-08-18 positive origins under corrected three-day alert
renewal accounting.

This strengthens the original caution: the current core daily feature set does
not support a reliable production tail warning. The evaluation framework is
retained, but no daily shadow head, interval challenger, data expansion, or UI
alert follows from this evidence.

## Scope and evidence used

This research is based on:

- the current `main`-equivalent repository tree at commit `c528bea`;
- the production `data/daily-forecast` branch through 2026-08-29;
- `eth_price_forecast.py`, `forecasting/model_registry.py`,
  `eth_data_collector.py`, model-evaluation scripts, workflows, and tests;
- the live forecast, prediction history, health, and collector quality reports;
- the three attached papers;
- primary method papers and official data-provider documentation linked below.

Production evidence:

- [latest forecast](https://github.com/wscha231/eth-dashboard/blob/data/daily-forecast/forecast_site/public/latest.json)
- [health report](https://github.com/wscha231/eth-dashboard/blob/data/daily-forecast/forecast_site/public/health.json)
- [prediction history](https://github.com/wscha231/eth-dashboard/blob/data/daily-forecast/eth_forecast_outputs/prediction_history.csv)
- [collector quality audit](https://github.com/wscha231/eth-dashboard/blob/data/daily-forecast/lake/reports/collector_data_quality_audit.json)

The August 2026 rally has already been inspected while forming this proposal.
It is therefore a **known forensic example**, not an untouched holdout that may
be used to claim unbiased performance. Promotion must depend on the full
historical walk-forward evidence and later shadow results.

## What is updated daily

The scheduled workflow does the following:

1. refreshes and merges market, macro, crypto-native, and available vendor data;
2. restores the persistent live forecast/feedback history;
3. refits the normal forecasting suite with the newest training row and its
   short walk-forward checks;
4. persists the new 7-day and 30-day forecasts and backfills matured outcomes;
5. re-scores the already committed long-run OOF evidence for the site.

It does **not** recompute the authoritative 36-fold candidate gate every day.
That expensive job remains in `.github/workflows/eth_model_eval.yml` and is
appropriate only for model changes or scheduled candidate evaluation. This
separation is correct; a tail-event change should preserve it.

## Recent rally forensic

The six 7-day forecasts whose input dates were 2026-08-14 through 2026-08-19
show the failure clearly.

| Input date | Direction | Direction probability | Regression return | Upper return | Realized return | Hybrid result |
|---|---:|---:|---:|---:|---:|---|
| 2026-08-14 | UP | 77.10% | -0.12% | +10.12% | +33.35% | NO_SIGNAL (`bias_flat`) |
| 2026-08-15 | UP | 90.85% | -0.37% | +9.85% | +28.79% | OBSERVE (no consensus) |
| 2026-08-16 | UP | 98.00% | +1.54% | +11.69% | +31.24% | OBSERVE (no consensus) |
| 2026-08-17 | UP | 98.00% | +1.23% | +11.19% | +31.27% | OBSERVE (no consensus) |
| 2026-08-18 | UP | 97.16% | +0.60% | +10.77% | +28.45% | NO_SIGNAL (insufficient confirmation) |
| 2026-08-19 | FLAT | 56.97% | -0.31% | +9.71% | +30.99% | NO_SIGNAL (`bias_flat`) |

The distinction matters:

- **Direction detection was often present.** Replacing every classifier is not
  the first-order fix.
- **Magnitude and uncertainty were badly under-responsive.** A normal central
  regression head and a residual interval calibrated on mostly ordinary days
  had no mechanism dedicated to a `+30%` rally.
- **The hybrid layer intentionally suppressed weakly validated signals.** That
  protection should not be deleted. A separate `TAIL_WATCH` output can surface
  rare-event risk without falsely upgrading it into a high-confidence trade.

## Current objective and integration mismatch

### Direction target

`build_direction_classification_targets()` in `eth_price_forecast.py` creates
an `UP` label when the future return exceeds a volatility-scaled threshold and
a `DOWN` label below its negative counterpart. Neutral rows are excluded. For
7 days, the threshold is clipped between `0.8%` and `4.5%`.

This is useful for separating meaningful direction from noise, but `+5%` and
`+30%` are the same `UP` class. It cannot teach a classifier that the second
outcome is a different operational event.

### Point and interval models

The regression registry contains central estimators such as Ridge,
Extra Trees, HistGradientBoosting, CatBoost with MAE, and isolated LightGBM.
The 30-day point champion is deliberately fixed to `no_change_anchor` because
the long-run OOF gate rejected the learned regressors. There is no promoted
upper-tail quantile head.

The interval logic adds conformal/residual protection around a central model,
then scales scenarios using recent volatility and a heuristic tail-pressure
score. This helps ordinary heteroskedasticity but does not directly optimize
conditional upper-tail coverage. The August result (`0/6`) demonstrates the
gap under a regime shift.

### Hybrid gate

For 7 days, `build_hybrid_forecast()` combines trend, reversal, direction,
central forecast, and fundamentals. `determine_signal_tier()` then requires a
combination of model consensus, positive backtest/holdout evidence, a strong
direction edge, a strong hybrid score, and no regime/reversal conflict. This is
why high `UP` probabilities could still end as `OBSERVE` or `NO_SIGNAL`.

The gate is designed for trade actionability. A rare-event monitor has a
different loss function: missing a large rally is costly, but an alert may be
shown without pretending the central point estimate or trading evidence is
high-confidence. The two decisions should be represented separately.

### Existing tail proxy

`eth_tail_event_pressure` combines recent absolute-return, daily range, volume,
volatility expansion, and volatility-squeeze z-scores. It is past-only and
leakage-safe, but it is an unsupervised heuristic. Several components rise only
after a jump has started, so the feature is not equivalent to forecasting a
future event probability.

## Event population and label choice

Using the continuous ETH close series from 2017-11-09 through 2026-08-29
(3,216 rows), the forward-return distribution is:

| Horizon and cutoff | Positive rows | Row prevalence | Approx. event clusters |
|---|---:|---:|---:|
| 3-day `>= +8%` | 397 | 12.36% | 165 |
| 3-day `>= +10%` | 277 | 8.62% | 120 |
| 3-day `>= +12%` | 184 | 5.73% | 84 |
| 3-day `>= +15%` | 112 | 3.49% | 61 |
| 3-day `>= +20%` | 42 | 1.31% | 26 |
| 7-day `>= +15%` | 345 | 10.75% | 70 |
| 7-day `>= +20%` | 189 | 5.89% | 44 |

“Event clusters” are an audit approximation: positive origin dates separated
by no more than the forecast horizon are grouped into one overlapping episode.
They are not independent observations, which is why reporting 189 positive
rows for the 7-day target would overstate the true evidence.

The 3-day positive-return quantiles are `+9.02%` (90th), `+13.04%` (95th),
`+17.40%` (97.5th), and `+21.64%` (99th). A fixed `+12%` cutoff is therefore
close to the empirical 95th percentile while retaining enough episodes for a
first-stage experiment.

There is also material regime dependence:

- in the latest 3-year window, fixed `+12%` gives only 28 rows and about 12
  event clusters;
- in the latest 5-year window, it gives 56 rows and about 27 clusters;
- over full history, it gives 184 rows and about 84 clusters.

Therefore a tail head cannot reuse the normal 3-year short-horizon training
window without becoming extremely unstable. The research candidate should use
expanding history with time decay and episode weights, while checking a
past-volatility-normalized robustness label. It must not use random shuffling,
random oversampling, or synthetic future-adjacent samples.

Recommended label pair:

1. Primary, interpretable: `r(t, t+3) >= 12%`.
2. Robustness: `r(t, t+3) >= clip(2 * sigma_30(t) * sqrt(3), 10%, 25%)`, where
   `sigma_30(t)` is computed only from returns available before origin `t`.

The recent rally is positive for both definitions on 2026-08-16, 17, and 18.

## Data readiness

The 2026-08-29 collector audit reports `tail_event_readiness=weak`.

| Group | Columns | Coverage | Ready columns | Longest span | Status |
|---|---:|---:|---:|---:|---|
| Core market | 35 | 100.00% | 35 | 3,215 days | OK |
| Macro/liquidity/credit | 21 | 71.93% | 18 | 3,053 days | OK |
| Crypto-native | 24 | 14.44% | 2 | 3,053 days | Partial in practice |
| Derivatives/tail | 18 | 0.64% | 0 | 164 days | Weak |
| On-chain activity | 0 | 0.00% | 0 | 0 days | Weak |

Four vendor columns are explicitly short-history and 31 collected vendor
columns are unlikely to survive the current training coverage filters. A model
can be researched on the core feature set now, but there is not yet enough
point-in-time derivatives history to claim that funding, OI, IV, skew, or
liquidations improve the long-run tail forecast.

## Attached-paper review

### 1. Wu et al., 2024 — deep crypto forecasting review/evaluation

Source: `2405.11431v2.pdf`, arXiv 2405.11431v2, 34 pages.

The paper compares LSTM, bidirectional/encoder-decoder LSTM, CNN, Conv-LSTM,
and Transformer models on univariate and multivariate daily close forecasting.
Conv-LSTM performs best in its reported multivariate setup. The authors also
identify uncertainty quantification, quantile regression for extreme-value
forecasts, hourly data, and multimodal news/social inputs as future work.

Transferable ideas:

- multivariate signals can help more than architecture complexity alone;
- multi-step accuracy degrades as volatility and horizon increase;
- extreme forecasts need a different objective such as quantile loss;
- uncertainty must be evaluated, not inferred from point RMSE.

Limitations for this repository:

- the outcome is primarily normalized price-level RMSE, not rare-event
  probability or calibrated tail coverage;
- one experiment rearranges the initial 70% training data with a shuffle
  strategy, which is not acceptable for this time-ordered task;
- hyperparameters are selected through trial runs/literature/defaults rather
  than this repository's nested purged gate;
- the paper itself reports poorer performance during high-volatility periods.

Decision: do not deploy Conv-LSTM now. Revisit it only after a tree/linear tail
baseline has passed and the number of point-in-time tail features is adequate.

### 2. Badar et al., 2025 — VAE + CNN-LSTM + SHAP

Source: `mathematics-13-01908-v2.pdf`,
[DOI 10.3390/math13121908](https://doi.org/10.3390/math13121908), 22 pages.

The paper proposes a VAE feature extractor, CNN-LSTM price model, and SHAP
explanations, reporting five-fold walk-forward price-level metrics near
`R2=0.99`. It explicitly recommends adding macro, social, high-frequency, and
on-chain features and acknowledges SHAP's computational cost and independence
assumption problem for correlated temporal features.

Transferable ideas:

- explanation output should accompany an alert;
- latent or grouped feature representations may reduce correlated feature
  noise later;
- walk-forward validation is necessary.

Limitations for this repository:

- Bitcoin price-level `R2` is not evidence of ETH tail-event detection;
- the reported high accuracy can be dominated by price persistence and scale;
- VAE/CNN/LSTM capacity is high relative to roughly 84 full-history event
  clusters;
- KernelSHAP-style explanations are costly and can be misleading when temporal
  features are strongly dependent.

Decision: use grouped permutation/ablation importance first. Keep VAE and deep
SHAP as a later challenger, not an initial dependency.

### 3. Bouteska et al., 2024 — ensemble/deep comparison

Source: `1-s2.0-S1057521923005719-main_copy.pdf`,
[DOI 10.1016/j.irfa.2023.103055](https://doi.org/10.1016/j.irfa.2023.103055),
12 pages.

This study evaluates univariate one-day cryptocurrency forecasts with rolling
windows, naive random-walk/white-noise/buy-sell baselines, regression metrics,
and directional metrics (`MDA`, `MDA+`, `MDA-`). It reports strong LightGBM
results for Ethereum in its setting.

Transferable ideas:

- keep naive baselines as first-class competitors;
- report positive- and negative-direction performance separately;
- evaluate operational decisions, not only regression error.

Limitations for this repository:

- it is univariate and one-day, while this project targets 3/7/30-day outcomes;
- it is not a rare-event probability experiment;
- this repository already tested LightGBM under its own 7/30-day purged OOF
  design and rejected it as a central price champion.

Decision: LightGBM may be re-tested only as an isolated tail classifier. The
paper does not justify promoting it as the central 7/30-day regressor.

## Primary method research

### Rare-event metrics

ROC AUC and ordinary accuracy can look acceptable when positives are rare.
Saito and Rehmsmeier show why precision-recall analysis is more informative for
imbalanced classification:
[PLOS ONE, 2015](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432).

For this task, row-level average precision is necessary but not sufficient.
The operational metrics must also count independent rally episodes, alert lead
time, and false-alert episodes per 90 days.

### Class imbalance

Focal loss down-weights easy negatives and concentrates learning on hard
examples ([Lin et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Lin_Focal_Loss_for_ICCV_2017_paper.html)).
It is conceptually relevant but was developed for a much larger detection
problem. With this small event sample, simple class/episode weighting is the
lower-variance first test; focal loss should be an ablation only.

### Asymmetric intervals

Conformalized Quantile Regression combines quantile regression with conformal
calibration and adapts interval width to heteroskedastic inputs:
[Romano, Patterson, and Candes, NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/5103c3584b063c431bd1268e9b5e76fb-Abstract.html).

Vanilla conformal assumptions are strained by financial distribution shift.
Relevant sequential alternatives include:

- [Adaptive Conformal Inference under Distribution Shift, NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html);
- [EnbPI for dynamic time series, ICML 2021](https://proceedings.mlr.press/v139/xu21h.html);
- [online conformal inference under arbitrary shifts, JMLR 2024](https://jmlr.org/papers/v25/22-1218.html).

These papers justify an adaptive/asymmetric interval challenger, not a claim
that a finite-sample exchangeable coverage guarantee automatically holds for
ETH. Coverage must be reported by time block, volatility regime, and upside
tail subset.

## Official data-source audit

| Source | Useful history/features | Verified constraint | Proposed use |
|---|---|---|---|
| [Coin Metrics community archive](https://github.com/coinmetrics/data) | Daily ETH active addresses, fees, MVRV, exchange inflow/outflow, supply, transactions, market data | Archive is CC BY-NC 4.0, schema can change, and the retrieved ETH snapshot ended before the project's live 2026-08-29 date | Historical on-chain ablation only until licensing and freshness gates pass |
| [Raw Coin Metrics ETH file](https://raw.githubusercontent.com/coinmetrics/data/master/csv/eth.csv) | One long daily table beginning in 2015 | Point-in-time revision behavior and current freshness must be recorded on every pull | Backfill candidate with source timestamp and checksum |
| [OKX historical data](https://www.okx.com/historical-data) | Perpetual funding from March 2022; trades from September 2021; candles from July 2023; L2 from March 2023 | Download format and terms must be pinned; funding is not equivalent to OI/liquidations | First historical derivatives backfill candidate |
| [Binance public data](https://github.com/binance/binance-public-data) | Official daily/monthly spot and futures trades/klines with checksums | Official repository documents trades/klines, not a complete historical OI/options archive | Build intraday realized volatility, range, taker-flow, and volume-imbalance proxies |
| [Deribit API](https://docs.deribit.com/) | Live futures/options/perpetual summaries, IV, OI, funding | Existing snapshot collection has short history; public summary snapshots are not a long backfill | Continue prospective collection; do not use in long-run OOF until ready |
| [DefiLlama API](https://api-docs.defillama.com/) | Free historical chain TVL, stablecoins, DEX volume, fees, and open-interest overviews | Endpoint-specific schema/revision checks needed | Extend the already-ready crypto-native history at low cost |

Data must be joined by the timestamp at which it would actually have been
available, not simply by its economic date. Every new source needs:

- raw immutable cache plus checksum/source timestamp;
- explicit publication lag and bounded forward fill;
- coverage, freshness, revision, and duplicate tests;
- train-fold-only feature transforms;
- an ablation showing improvement over the same model without the source.

No paid data subscription is justified before free/core-only baselines show
that the tail head itself can beat climatology.

## Recommended experiment architecture

### Keep existing products stable

- Do not replace the promoted 7-day/30-day point models.
- Do not relax the existing `HIGH_CONFIDENCE` trading gate.
- Do not run another full 36-fold suite every day.

### Add two isolated outputs

1. `tail_up_probability_3d`: calibrated probability of a `+12%` or larger
   three-day rise.
2. `tail_upper_return_q95_3d`: asymmetric upper-return scenario calibrated on
   walk-forward residuals.

The alert should initially be `SHADOW`, then at most `TAIL_WATCH`. It may widen
or add a separate upside scenario after validation, but it must not silently
move the central point forecast or become a trade instruction.

### Start with low-variance challengers

- expanding-window event climatology;
- the current heuristic `eth_tail_event_pressure` as a scored baseline;
- episode-weighted logistic regression;
- regularized CatBoost and optional LightGBM with class weighting;
- upper quantile gradient boosting/CatBoost plus conformal calibration.

Only after these pass should Conv-LSTM, VAE, focal loss, panel transfer from
other coins, news/social language models, or paid derivatives archives be
considered.

## Principal risks

1. **Few independent events.** Full-history row counts overstate the evidence;
   clustered bootstrap and event-level metrics are mandatory.
2. **Regime drift.** Early crypto history contains far more large rallies than
   2023-2026. Use time decay and a volatility-normalized robustness label.
3. **Known-event tuning.** The August rally cannot be treated as an untouched
   test after this forensic analysis.
4. **False alerts.** Optimize recall subject to an explicit false-alert budget,
   not recall alone.
5. **Probability overconfidence.** Calibrate only on temporally prior OOF rows
   and report Brier skill/reliability curves.
6. **Vendor leakage/revisions.** Persist as-of timestamps and audit revisions.
7. **Licensing/freshness.** Coin Metrics community data is not automatically
   cleared for every use and its retrieved snapshot was not live-fresh.
8. **Compute creep.** A dedicated small head is acceptable; daily deep-model
   retraining and repeated full-suite gates are not part of the first phase.

## Research decision

The staged PR 1 probability experiment described in
`plan_tail_event_forecasting.md` was appropriate and has now produced an honest
negative result. Retain its offline evaluation code, but stop before the
asymmetric interval and daily shadow stages. The plan deliberately separates:

- event warning from trade actionability;
- tail probability from the central point forecast;
- cheap daily inference from expensive promotion evidence;
- core-only proof of value from new-source enrichment.
