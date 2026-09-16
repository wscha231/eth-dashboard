# R1 shadow compatibility adapter

Status: **offline/shadow compatibility only; no production promotion**.

This increment connects the semantics of the existing `separate_heads_frozen_calibration_v1` research bundle to the explicit `ForecastBundleV2` contract without changing the existing fitter or live inference.

## Mapping

| Existing shadow research | V2 head | R1 treatment |
|---|---|---|
| `choices.point` | Center | compare with mandatory no-change selection MAE; 2% compatibility threshold |
| `choices.interval` | Distribution | compare with legacy climatology interval loss; 2% compatibility threshold |
| `choices.probability` | Event | compare with legacy climatology probability loss; 2% compatibility threshold |
| none | Volatility | **baseline-only** persistence projection; no candidate win is invented |

The thresholds above are compatibility decisions, not production promotion certificates. Full #56 gates still require matched incumbent comparisons, calendar-block stability, paired evidence and the head-specific secondary constraints.

## Evaluation interfaces

`signal_pipeline/v2_evaluate.py` defines model-agnostic metrics:

- Center: simple-return MAE, RMSE, median pinball;
- Volatility: QLIKE and variance RMSE;
- Distribution: one-central-interval WIS, q10/q50/q90 pinball, 80% coverage and width;
- Event: terminal/path Brier, terminal log-loss and worst-component ECE.

The evaluation APIs are intentionally independent so a volatility improvement cannot authorize Center, Distribution or Event promotion.

## Volatility placeholder

R1 projects the existing trailing hourly `sigma` over the forecast horizon as `sigma^2 * h` and marks the head decision as `variance_persistence`. This is a baseline forecast only. R2 will introduce HAR/HAR-RV, EWMA, DVOL and regime-conditioned challengers under the frozen QLIKE gates.

## Hard boundary

This increment does not:

- import V2 into `engine.daily()`;
- modify `optimization.train_candidate()`;
- change `shadow_active.json`;
- write V2 forecasts to the live ledger;
- change the website;
- promote any model.

The next R1 increment is an offline replay/export path that emits V2-compatible records and scores all four heads side-by-side on identical origins. Only then does R2 begin model competition.
