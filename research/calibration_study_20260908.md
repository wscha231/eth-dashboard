# Frozen-model alert calibration and derivative-data ablation

Study protocol recorded on 2026-09-08 before executing the new comparison.
Base code: `baf022d907512d3fdd79c819616095476c6e10b0`.

## Problem and scope

Production chooses the model on earlier validation predictions, refits that model,
then applies thresholds from the earlier fit. A constant frequency model can
therefore alert continuously when its refitted probability exceeds the earlier
threshold. The research candidate reserves a final calibration window, calculates
thresholds using the exact final fitted model, and performs no later refit.
This fixes model/threshold identity, not a guarantee of 5% future false positives.

No existing issued record is changed. This study does not change active production
checkpoints or claim that operational `ready` status establishes predictive skill.

## Fixed comparison

- Horizons: 6 and 24 hours. Origins: daily at 00:00 UTC from 2025-01-01.
- Final source/label cutoff: 2026-09-07 00:00 UTC; incomplete labels excluded.
- Variants: legacy model; frozen calibrated model with existing OHLCV features;
  frozen model plus hourly funding; frozen model plus daily ETH DVOL.
- Every variant uses the same complete input mask, training origins and evaluation
  origins for its method. The correction deliberately changes temporal training
  boundaries; derivative ablations share the corrected boundaries exactly.
- Training stride: six hours. Candidate architectures/hyperparameters unchanged.
- Earlier model selection: 90 days; final calibration: 30 days; training: 730 days.
  Each label interval must end before the following partition minus one hour.
- Funding: sums of disjoint one-hour interest over 6/24 hours, never overlapping
  eight-hour rates. DVOL: previous completed day level and seven-day change.
- Historical input eligibility assumes interval end plus one hour. Actual receipt
  timestamps are retained. This is retrospective reconstruction, not a historical
  point-in-time source archive. No forward fill across missing hours/days.
- Report paired event Brier, 95% calendar-block bootstrap interval, recall/FPR,
  price error/coverage, yearly and origin-defined trend/volatility slices, and
  independent forecast counts. Slices are exploratory, not model-selection rules.
- Trading diagnostic: long/cash; enter only on up alert without down alert; enter
  next hour open, exit target-end close; one position at a time; no leverage.
  Costs: 0, 10, 25 basis points per side, assumed combined fee/spread/slippage.
  Compare ETH buy-and-hold over the entire same execution calendar, including cash
  periods. Drawdown uses hourly closes. Missing execution bars invalidate returns.
- Historical comparisons are development evidence and not prospective performance.
  The decision to study 6/24h follows earlier observed results, so it is not an
  untouched external holdout. No automatic model or portfolio promotion.

## Provenance

Hourly data: same-repository main-branch successful research run
[34101303236](https://github.com/wscha231/eth-dashboard/actions/runs/34101303236),
artifact `10010792872`, SHA256
`69721080fc08c98fc8d6651db544dbdd2e90c75d9cd1e96106bffcf3bca2f655`.
The archive is verified before extracting normalized bars; no external serialized
models are loaded. Original 169,653 ETH/BTC bar rows extend to 2026-09-07 08:00 UTC;
the study truncates at its earlier fixed cutoff.

Funding and DVOL snapshots were obtained in [PR #26](https://github.com/wscha231/eth-dashboard/pull/26).
The collector changes remain separately reviewed; this study uses their local
snapshots and records file hashes/receipt ranges in its output. Provider raw data
are not committed to the public repository.

Before comparing returns, the input audit found ten absent ETH/BTC hours:
2025-10-25 16:00–20:00 UTC and 2026-05-08 02:00–06:00 UTC (bar open times).
The predeclared full-calendar trading diagnostic therefore remains invalid.
An additional coverage diagnostic reports **every** contiguous complete ETH-price
window separately, with a flat initial position and no compounded result across
gaps. Window selection uses availability only, not returns; the strategy and costs
are unchanged. This amendment was recorded before the first new performance
summary or trading result was inspected.

Results and the adoption decision will be appended after the fixed run completes.
