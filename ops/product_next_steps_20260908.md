# Product and prediction priorities

The service goal is useful ETH direction, large-move and trend-change forecasts,
with evidence that a subscriber can understand. A nicer interface is not an
improvement in predictive performance.

## Current delivery

- English outlook with one selected forecast window (6 hours to 30 days).
- The same window controls historical metrics, charts and the published ledger.
- Probability error, rise/fall catch rates, false alarms and price-range coverage
  have distinct explanations. Unfinished outcomes never count as successes.
- Historical simulations and actual published results are shown separately.
- Archived models cannot overwrite the current status or reference price.
- Publication checks compare the served HTML and JavaScript with the source.

## Next implementation order

| Priority | Deliverable | Acceptance evidence |
|---|---|---|
| 1 | Reliable hourly issuance | Collect queue-to-inference and publication timings; give recovery enough time before the :55 issuance deadline; mark deferred runs clearly and verify all 6 horizons by the next valid slot |
| 2 | Consistent fitted model and alert calibration | Calibrate the same fitted model that produces final probabilities; compare on held-out chronological data with settled labels and the existing false-alarm budget |
| 3 | Historical input experiments | Add funding, DVOL and price-volume proxies separately; compare the same origins and dates against the existing model; retain missingness and receipt-time provenance |
| 4 | Horizon and market-condition adaptation | Start with horizon-specific models using past-only market-condition features; require enough independent outcomes before trying separate regime specialists |
| 5 | Subscription evidence | Accumulate fixed, actually published forecasts; track false alarms, missed events, warning lead time, probability quality and interval coverage; evaluate costs and delivery delays before making paid-service claims |

The hourly run on 2026-09-07 at 14:54 UTC was requested before the deadline, but
inference reached 14:55:13. All six forecasts were rejected by the :55 guard.
Its site verification correctly reported `delayed` and `horizons=0`, although
the overall workflow was green because publishing a delay status is allowed.
Do not solve this by removing the issuance guard or backdating forecasts.
Measure a conservative queue/build allowance and defer a late dispatch to the
next valid window. A dependable external trigger is still needed for a paid
service; a successful single recovery is not scheduler reliability evidence.

## Data already recovered, not yet adopted by the live model

[PR #26](https://github.com/wscha231/eth-dashboard/pull/26) has a successful
full test run on its current head. It adds bounded research collectors and
price-volume proxies. The audit recovered 62,254 hourly funding observations
from August 2019 to September 2026 (two hours remain missing), and 1,985 complete
daily ETH DVOL candles from April 2021. Existing unmerged research also contains
longer TVL, stablecoin and DEX histories. These are candidate inputs, not measured
predictive gains. Confirm data-use rights before adopting them in a paid product.

Do not fabricate old open interest, liquidations, or pre-inception ETF flows.
Use genuine archives when available; otherwise keep the long-history base model
and compare an optional-input model only over the common period. Extra hourly
rows do not create independent 30-day events. Synthetic data must never count
as real evaluation outcomes.

## Review cadence

- Hourly: data completeness, input slot, all six outputs and external publication.
- Daily: settle ended windows, refresh the published record, audit missed slots.
- Weekly: compare recent settled performance with the saved baseline; investigate
  data drift and calibration before deciding whether a candidate is needed.
- Monthly: evaluate retraining and model changes using mature labels. Keep the
  previous model and its original forecasts available for comparison.

Long-horizon validation takes time: a 30-day forecast has no final outcome until
its 30-day observation window ends. Neither a quiet week nor a favorable slice
is sufficient evidence for model promotion.

Evidence:

- [Late hourly run and public delay](https://github.com/wscha231/eth-dashboard/actions/runs/34135493934)
- [Recovered-data PR tests](https://github.com/wscha231/eth-dashboard/actions/runs/34129181678)
- [Historical-input research](https://github.com/wscha231/eth-dashboard/pull/26)
