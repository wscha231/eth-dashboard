# ETH Forecasting: Next Enhancement Research

Date: 2026-08-31

Status: **research complete; PR 1 source preflight implemented offline**

## Decision

The next enhancement should be **data-first and distribution-aware**, not a
direct jump to CNN-LSTM, VAE, or Transformer models.

The recommended research program has two independent objectives:

1. **Predictive-signal track:** add historically backfilled order-flow,
   derivatives, and Ethereum-liquidity features, then compare the rejected
   direct upside-tail classifier with a factorized large-move/direction model.
2. **Uncertainty track:** add an asymmetric return-quantile challenger with
   change-point-aware conformal calibration so the upper range can remain
   honest during regime shifts even when a rare rally cannot be predicted.

The existing 7-day and 30-day production forecasts, trading gate, database,
and public UI should remain unchanged until an offline challenger passes
predeclared walk-forward gates.

## Source-preflight evidence added after research

The 2026-08-31 PR 1 execution validated the recommended primary sources rather
than assuming they were usable:

- Binance listed continuous ETH/BTC spot history from 2017-08 and USD-M
  perpetual history from 2020-01 through the latest complete monthly archive,
  2026-07. Twelve bounded ZIP samples passed official/local SHA-256 and bar
  integrity checks, including the 2025 spot timestamp-unit transition.
- DefiLlama returned complete daily Ethereum histories through 2026-08-30:
  3,260 TVL rows, 3,197 stablecoin rows, and 2,859 DEX-volume rows.
- Deribit returned one-hour ETH perpetual funding observations in 2019, 2023,
  and 2026. The earlier local assumption of about 90 days of retention was too
  conservative, although book-summary option/future fields remain snapshots.
- OKX remained excluded because the execution environment could not reach its
  public API reliably and its full-archive automation/terms gate is unresolved.

This evidence changes source readiness, not the research conclusion: the next
authorized work is still fold-safe aggregation and source ablation, with no
operational model change.

## Why a new path is necessary

PR #9 evaluated a direct event target: ETH rises at least 12% over the next
three days. On 2,795 out-of-fold dates from 2019 through 2026, the best
core-feature model had:

- average precision `0.0732` versus climatology `0.0511`;
- Brier skill `-1.29%`;
- event recall `20%` at the selected alert threshold;
- only a `16.5%` paired-bootstrap probability of Brier improvement;
- no alert for the known 2026-08-16 through 2026-08-18 origins.

CatBoost and LightGBM also failed. This rules out further tuning of the same
daily price/macro feature set as the next primary experiment. It does not rule
out tree models after genuinely new inputs are added.

The current collector audit also explains the limitation:

| Feature group | Ready columns | Longest history | Assessment |
|---|---:|---:|---|
| Core market | 35 | 3,215 days | Ready |
| Macro/liquidity/credit | 18 | 3,053 days | Ready |
| Crypto-native | 2 | 3,053 days | Mostly sparse |
| Derivatives/tail | 0 | 164 days | Not trainable |
| On-chain activity | 0 | 0 days | Missing |

The 2026-08-29 run also recorded HTTP 451 failures for Binance funding, open
interest, basis, and taker-volume API endpoints. Deribit snapshots had only a
few usable dates, and DefiLlama history began on 2026-07-12 because the source
was introduced after the initial full backfill and normal runs refresh only a
recent window.

## Attached-paper review

### Evidence grading

| Paper | Useful result | Main problem for this project | Decision |
|---|---|---|---|
| Wu et al. (2024), *Review of deep learning models for crypto price prediction* | Multivariate Conv-LSTM performed best in the paper's daily 1-5 step experiments; longer horizons degraded; the authors recommend quantiles and multimodal inputs | Hyperparameters were partly chosen from test performance; one experiment randomly rearranged the initial 70% of a time series; evaluation optimizes normalized-price RMSE rather than trading or tail calibration | Use only as motivation for multivariate inputs and a later bounded sequence challenger |
| Badar et al. (2025), *CNN-LSTM + VAE + SHAP* | Shows a possible representation-learning and explanation pipeline | The text includes `Close` in inputs while `Close` is the target, does not clearly define the target shift, uses nonstationary levels, appears to scale before the split, uses test data for VAE validation and SHAP background, and reports contradictory ablation numbers | Do not use the reported `R2=0.99` as evidence; do not implement this pipeline now |
| Bouteska et al. (2024), ensemble vs deep learning | Uses log-price differences, rolling-window validation, random-walk/white-noise baselines, direction-specific metrics, and investor metrics; LightGBM was strongest for ETH in its univariate setting | One-day univariate task, older 2016-2023 sample, limited source variables, and no modern leakage/transaction-cost audit | Adopt its evaluation discipline, not its winner by assumption |

### Transferable lessons

The consistent result across the stronger evidence is not "deep learning
wins." It is:

- multivariate information matters more than adding depth blindly;
- raw price-level `R2` is a weak success criterion;
- random walk/no-change must remain an explicit baseline;
- upward and downward accuracy must be reported separately;
- multi-step uncertainty and extreme outcomes need their own objectives;
- model architecture should be chosen only after leakage-safe source ablation.

## External research relevant to the next step

### Order flow

Anastasopoulos et al., published in the *Journal of Financial Markets* in
2026, report that lagged world order flow has out-of-sample predictive value
for daily cryptocurrency returns and tends to dominate standard economic
fundamentals in nonlinear models. Their cross-sectional design and proprietary
flow construction are not directly reproducible here, but the result supports
testing signed taker flow and spot/perpetual flow divergence rather than more
OHLC indicators.

Source: [Order flow and cryptocurrency returns](https://www.sciencedirect.com/science/article/pii/S1386418126000029)

### On-chain exchange flows

Chi, Chu, and Hao study 2017-2023 intraday data and report that ETH exchange
net inflows negatively predict future ETH returns and volatility, while USDT
net inflows positively predict short-horizon BTC and ETH returns. Their 1-6
hour horizon is shorter than this project's 3/7/30-day products, so it is a
source hypothesis, not proof of multi-day value.

Source: [Return and Volatility Forecasting Using On-Chain Flows](https://arxiv.org/pdf/2411.06327)

### Distribution shifts and intervals

The NeurIPS 2025 CPTC paper combines a switching-state model with online
conformal prediction so intervals can anticipate predictable change points
instead of reacting only after a miss. It is not validated on ETH, but it
provides a relevant lightweight challenger for the exact failure observed in
August: a central model with a stale residual distribution produced an upper
range that was far too narrow.

Source: [Conformal Prediction for Time-series Forecasting with Change Points](https://proceedings.neurips.cc/paper_files/paper/2025/file/12271b64c483ad8f6192eb6aaa102044-Paper-Conference.pdf)

## Candidate paths considered

| Path | Expected value | Evidence quality | Cost/risk | Rank |
|---|---|---|---|---:|
| Add order flow, funding, basis, and Ethereum liquidity | Directly fills the current information gap and has independent empirical support | Medium-high | Source revisions, venue changes, licensing, timestamp alignment | 1 |
| Factorize large-move occurrence and direction | Increases event evidence and lets volatility and direction use different predictors | Medium; must be tested | Calibration error when multiplying heads | 2 |
| Asymmetric quantiles + adaptive conformal | Can improve honest ranges even if alpha remains weak | High methodological support | May widen ranges without predicting direction | 3 |
| Panel/multitask transfer across coins | Increases rare-event samples and can learn common crypto regimes | Medium | Survivorship bias and ETH-specific domain shift | 4 |
| CNN-LSTM/TCN on hourly sequences | Can capture temporal flow patterns missed by daily aggregates | Conditional | More variance, compute, and leakage surface | 5 |
| VAE feature compression | Possible anomaly score | Low for this use case | Easy to reconstruct price level rather than predictive state | 6 |
| Historical news/LLM catalyst model | Some ETH rallies are event-driven | Conceptually strong, data weak | Timestamped archive and labeling are difficult | Prospective shadow only |

## Target redesign

The direct `3-day return >= +12%` target contains 184 positive origins but
only about 84 overlapping event clusters over the full 2017-2026 history.

Alternative label counts from the same 3,216-row ETH series are:

| Three-day target | Positive rows | Approx. clusters | Use |
|---|---:|---:|---|
| Terminal return `>= +12%` | 184 | 84 | Existing direct control |
| Terminal return `<= -12%` | 157 | 66 | Downside control |
| Absolute terminal return `>= 12%` | 341 | 126 | Large-move head |
| Any future high crosses `+12%` | 314 | 122 | Barrier diagnostic |
| Any future low crosses `-12%` | 319 | 104 | Barrier diagnostic |
| Either barrier is crossed | 627 | 163 | Path-event diagnostic |

The recommended first factorization is conservative and unambiguous:

```text
P(upside tail) = P(abs(3d terminal return) >= 12%)
                 * P(3d return > 0 | available information)
```

This is compared against, not substituted for, the existing direct upside
classifier. Both component probabilities and the final product require
prior-only calibration. A direct three-class `DOWN_TAIL / NORMAL / UP_TAIL`
model is a second challenger. Barrier labels remain diagnostic until hourly
data can determine first passage without daily OHLC ambiguity.

## Recommended source stack

### 1. Binance public hourly archives - first priority

The official archive provides daily/monthly spot and USD-M futures klines,
aggregate trades, and checksums. Hourly klines already include quote volume,
trade count, and taker-buy base/quote volume, so tick downloads are unnecessary
for the first experiment.

Source: [Binance public data](https://github.com/binance/binance-public-data)

Initial contracts:

- `ETHUSDT` spot 1h;
- `ETHUSDT` USD-M perpetual 1h;
- `BTCUSDT` spot 1h;
- `BTCUSDT` USD-M perpetual 1h.

Derived features:

- signed taker-flow ratio and its 1/3/7-day changes;
- spot versus perpetual flow divergence;
- futures/spot volume ratio and trade-count intensity;
- close-to-close perpetual basis and basis change/z-score;
- hourly realized volatility, upside/downside semivariance, and jump ratio;
- final 4/8/12-hour momentum and volume share before the UTC cutoff;
- ETH-minus-BTC flow, volatility, and momentum spreads.

The archive may revise files and has known user-reported missing/duplicate
issues in some futures datasets. Every file needs its official checksum plus
our own schema, continuity, duplicate, and revision checks.

### 2. DefiLlama full-history backfill - first priority

The existing endpoints already support historical Ethereum TVL, stablecoin
supply, and DEX volume. The current 49-day local history is an ingestion-window
problem, not an endpoint-history limit.

Source: [DefiLlama API](https://api-docs.defillama.com/)

Derived features:

- Ethereum stablecoin supply 7/30/90-day change and acceleration;
- Ethereum DEX volume z-score and volume/TVL ratio;
- chain TVL change and TVL/ETH-market-cap ratio;
- stablecoin supply relative to ETH and total crypto market value.

### 3. OKX funding and secondary-venue flow - second priority

OKX publishes perpetual funding from March 2022, tick trades from September
2021, candles from July 2023, and L2 order books from March 2023.

Source: [OKX historical market data](https://www.okx.com/historical-data)

Funding must be normalized to a daily equivalent because settlement intervals
can be 1/2/4/8 hours and OKX revised its formula in June 2026. Venue and
contract identifiers must never be silently merged.

### 4. Deribit - continue prospective collection

Deribit exposes historical perpetual funding and volatility endpoints, while
option IV/open-interest summaries are primarily current snapshots in the
existing collector. Funding backfill should be re-audited; options/basis/OI
remain prospective unless a defensible historical archive is found.

Sources: [funding history](https://docs.deribit.com/api-reference/market-data/public-get_funding_rate_history),
[historical volatility](https://docs.deribit.com/api-reference/market-data/public-get_historical_volatility)

### 5. Coin Metrics - research-only ablation

The community ETH archive begins in 2015 and includes active addresses, fees,
MVRV, exchange inflow/outflow, exchange-held supply, transactions, and spot
volume. However, the archive is CC BY-NC 4.0, warns that schemas may change,
and its ETH file was only current through 2026-05-22 when checked on 2026-08-31.
It should not be a production dependency for a commercial app without separate
license approval.

Source: [Coin Metrics community archive](https://github.com/coinmetrics/data)

## Proposed model architecture

### Track A: predictive signal

Evaluate the following on identical dates and folds:

1. current direct upside-tail baseline;
2. direct upside-tail model plus each new source group;
3. factorized large-move/direction model;
4. three-class tail model;
5. source-group ensemble only if individual ablations pass.

Start with logistic regression, HistGradientBoosting/CatBoost, and one
regularized LightGBM challenger. Deep sequence models are not in the first
gate.

### Track B: uncertainty

Predict 3-day and 7-day return quantiles, initially `q05/q50/q95`, using a
low-variance quantile tree model. Compare:

1. the current empirical/conformal range;
2. conformalized quantile regression;
3. volatility-scaled online conformal;
4. a small change-point-aware conformal challenger using existing regime
   probabilities as state inputs.

This track can pass even if Track A does not. Its output is an uncertainty
scenario, not a trade signal or a silently altered central forecast.

## Validation requirements

### Data gates

- complete UTC cutoff semantics and no partial current-day bar;
- archive URL, exchange symbol, retrieval time, upstream checksum, local SHA;
- explicit source revision and formula-change version;
- at least 730 prior daily observations for a learned source group;
- at least 20 prior independent positive episodes per training fold;
- source coverage and freshness reported per calendar block;
- all transforms fitted inside the outer training fold.

### Model gates

Use the existing expanding annual OOF framework, restricted to common source
coverage for fair ablation. The earliest authoritative test block should have
at least two years of prior source history.

For an upside-tail probability to enter shadow mode, require all of:

- average precision at least 20% above the matched-date core baseline;
- positive Brier skill versus the matched-date core baseline;
- event recall at least 35% within three false-alert episodes per 90 days;
- paired block-bootstrap probability of improvement at least 90%;
- improvement in at least four calendar blocks and no one block responsible
  for more than half of the total gain;
- no collapse on the volatility-normalized label;
- the August 2026 event reported only as a forensic case, never used to choose
  thresholds or features.

For a new upper interval to enter shadow mode, require:

- lower weighted interval score or pinball loss on matched dates;
- nominal overall upper coverage without a material width explosion;
- materially fewer and smaller exceedances in upside-tail episodes;
- calendar- and volatility-regime coverage tables;
- no post-miss information used to choose the current interval.

## What is deliberately deferred

- paid Glassnode/Kaiko/Amberdata/Laevitas data;
- a production news sentiment model;
- live alerts or UI changes;
- daily deep-model retraining;
- VAE features or SHAP explanations on correlated raw time lags;
- tuning to make the August 2026 rally appear predicted.

## Final research conclusion

The best next bet is **hourly order-flow and Ethereum-liquidity backfill,
tested through source ablation and a factorized large-move/direction target**.
In parallel, a **change-point-aware asymmetric interval** addresses the range
failure without pretending to create directional alpha.

CNN-LSTM becomes reasonable only if hourly source groups first show stable OOF
value with simpler models. The VAE paper does not clear the evidentiary bar for
implementation.
