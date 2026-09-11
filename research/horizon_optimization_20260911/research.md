# ETH horizon optimization and forecast diagnostics

Status: research and a concrete implementation plan. Production model, issued records, schedules and site are unchanged in this proposal. The user's earlier rule requires research/plan approval before production code changes; the latest request defines the desired capabilities but has not yet reviewed this new model protocol.

## Pinned evidence

- Main: `633c4896cf53b0cd27808c9637c0c28e8b046de0`.
- Published data: `0184b101a34fbd3999eb0fa86d53c622b48d2cad`.
- Replay blob: `8d470286612690ef08c02d7be76059b5bab7c668`.
- Historical data through 2026-09-10 22:00 UTC; replay generated 22:59:07 UTC.
- Live forecast payload generated 2026-09-10 23:00:31 UTC.
- `forecast_review.html` is a self-contained report of this snapshot, not a live feed. `evidence.json` records the measurements and source IDs.

## What already works

The model is separately selected for 6, 24, 72, 168, 336 and 720 hours. It compares past-frequency, logistic, CatBoost and a fixed 50:50 CatBoost/frequency mixture. Shorter horizons use two training years and 90 validation days; 14/30-day horizons use three years and 365 validation days. Full label intervals are purged at chronological boundaries.

Live inference uses the current month's saved checkpoint and fresh hourly bars. It does not refit. Scheduled research runs weekly and at the start of each month; unchanged monthly checkpoints are reused. Consequently, an hourly price/probability update does not mean an hourly model improvement.

The site already has one selected-horizon price chart and period/market-state filters. It does not yet provide a six-horizon overview, full individual publication archive, reliability chart, confusion matrix or error distribution.

## Measured weaknesses

| Horizon | Historical rows / nonoverlap rows | Event Brier improvement vs frequency | Price MAE improvement vs no change | Return MAE (percentage points) | 80% interval observed coverage |
|---|---:|---:|---:|---:|---:|
| 6h | 2,636 / 2,636 | +6.39% | +0.46% | 1.35 | 82.55% |
| 1d | 2,620 / 1,313 | +2.41% | +0.22% | 2.96 | 79.54% |
| 3d | 2,594 / 653 | -0.65% | -1.10% | 5.26 | 79.38% |
| 7d | 2,535 / 322 | -1.51% | -1.90% | 8.22 | 77.20% |
| 14d | 2,258 / 156 | -1.03% | -2.50% | 12.42 | 80.12% |
| 30d | 2,100 / 71 | -3.72% | -4.21% | 20.54 | 79.00% |

These comparisons are within each horizon on matched dates. Horizons have different start dates and missing-date coverage. Positive improvement means lower error; return MAE is not price MAPE. For 30 days, the exploratory calendar-block interval for the event Brier difference is positive throughout (0.00321 to 0.01402), favoring the frequency baseline over the selected historical policy.

The three-class balanced accuracy is approximately 33–35%, although ordinary accuracy can be high because the middle class dominates. The three classes mean below/within/above the original volatility-scaled barriers; they are not simply negative/zero/positive returns. Displaying only ordinary accuracy would obscure poor large-move discrimination.

The current monthly choices are CatBoost (6h), the 50:50 mixture (1d, 14d), and frequency (3d, 7d, 30d). Thus some horizons currently cannot change their return distribution with the newest input features, apart from the reference-price conversion. This is a baseline selected by the present validation rule, not a failed hourly inference call.

## Concrete code issues

1. `signal_pipeline/models.py::train_bundle` selects using probabilities from an inner fit, then refits the model and retains thresholds from the earlier fit. Thresholds and final probability distributions can differ. The latest live 3d sample illustrates the issue: all 18 resolved records alert upward while zero upper-barrier events occurred. This is a small sample and not a general performance estimate, but it demonstrates why model/threshold identity matters.
2. `catboost_calibrated` is a fixed 50:50 mixture with prior frequency. It is not a calibrator fitted on disjoint outcomes; its display name should state what it actually is.
3. A combined probability Brier criterion selects the price-quantile model too. A probability winner need not minimize price error. A zero-return point baseline is evaluated but is not a separately selectable point-price candidate.
4. `engine.py::daily` exports only the last 96 verified records, while `prospective_report` aggregates the full timely published ledger. A complete live-history chart cannot be reconstructed from those 96 rows.
5. `evaluate.py::report_replay` exports all selected price/path plot rows, but omits per-row terminal probabilities/classes, thresholds, selected model/version and baseline values from the browser points. Those fields are needed for a complete, auditable confusion/reliability/error interface.
6. The latest live summary has 73/57/18 resolved results for 6h/1d/3d, but only 14/4/1 nonoverlapping outcomes. The 7d/14d/30d outputs have zero resolved results. None supports a claim of proven live superiority.

## Existing research to reuse

[PR #26](https://github.com/wscha231/eth-dashboard/pull/26) supplies funding/DVOL collectors and risk-feature research. It is still unmerged and currently conflicts with main. Data availability is not evidence of predictive gain.

A local prior study, `research/calibration_study_20260908.md`, and `research_pipeline/event_calibration.py` already implement a frozen-model alert-calibration candidate for 6h/24h. Its study document does not contain completed results, and it is not part of the current main. Reconcile that work and its evidence before extending it; do not claim it has passed or silently copy an unmeasured model into production.

## Attached research and applicable evidence

- Wu et al., [Review of deep learning models for crypto price prediction: implementation and evaluation](https://arxiv.org/abs/2405.11431), supplied v2 PDF: compares several deep architectures, multivariate inputs, short multi-step forecasts and different time regimes. It supports testing compact multivariate alternatives; it does not establish a best 30-day ETH probability model. Its model rankings cannot be transferred to our event definition or USD error units.
- Badar et al., [Enhanced Interpretable Forecasting of Cryptocurrency Prices Using Autoencoder Features and a Hybrid CNN-LSTM Model](https://doi.org/10.3390/math13121908), supplied revised PDF: Bitcoin VAE/CNN-LSTM with SHAP and a five-fold walk-forward section. The reported scaled-price MSE/MAE and high R-squared are not ETH event calibration, nor a 99% probability of a correct trade. Reproduce preprocessing within each training partition if this becomes a candidate.
- Bouteska et al., [Cryptocurrency price forecasting – A comparative analysis of ensemble learning and deep learning methods](https://doi.org/10.1016/j.irfa.2023.103055), supplied PDF: includes ETH and distinguishes price errors from directional/trading outcomes. Results differ by method and period; simple recurrent and boosting candidates deserve bounded comparison, not automatic adoption.
- [Scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html): reliability diagrams compare predicted probabilities with observed event frequencies. Brier and log loss also reflect discrimination, so lower Brier alone does not prove better calibration. A calibrator needs predictions on data disjoint from the model's training data.
- Gibbs and Candès, [Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html): motivates evaluating adaptive uncertainty intervals. Its long-run coverage objective is not a guarantee for each ETH forecast, and delayed 30-day labels must be handled explicitly.

Our inference: prioritize consistent calibration, separate output selection and honest diagnostics before a larger neural model. A deep model should face the same horizon-specific, chronological, cost-bounded comparison.

## Report validation

All 14,743 exported historical points were checked for unique origins, mature target windows and exact horizon length. Actual USD prices were reconciled against reference price times realized return. Return MAE, event Brier and interval coverage were independently recomputed and matched all six published summaries. Price charts preserve original target dates and gaps; monthly error charts are explicitly monthly aggregates. Reliability bins retain their sample counts and do not assume independent overlapping observations.

The attached papers were read from the supplied files. No new model fitting, model promotion, production outage injection or live-site deployment was performed for this report.
