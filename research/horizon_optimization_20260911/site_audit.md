# Site audit — 2026-09-11

Source commit: `5c38c125573719472a3bdb7c0606d329aae6faf1`. Public payload generated 2026-09-11T11:10:02.062339+00:00. External verifier: https://github.com/wscha231/eth-dashboard/actions/runs/34592639518 (11:11:08 UTC). All six incumbents and six shadow predictions were present; no shadow errors. This proves publication, not accuracy. Direct browser access was unavailable in this environment; review used the deployed source, verified public payload, content-addressed evidence manifest and JavaScript interaction tests.

| Window (hours) | Historical return MAE (pp) | Improvement vs no-change | Historical coverage | Live settled / non-overlap | Live MAE / no-change (pp) |
|---|---:|---:|---:|---:|---:|
| 6 | 1.346 | 0.46% | 82.5% | 83 / 16 | 0.591 / 0.596 |
| 24 | 2.963 | 0.21% | 79.5% | 68 / 4 | 0.945 / 0.870 |
| 72 | 5.260 | -1.11% | 79.4% | 28 / 1 | 1.472 / 1.210 |
| 168 | 8.222 | -1.91% | 77.2% | 0 / 0 | Pending |
| 336 | 12.411 | -2.50% | 80.1% | 0 / 0 | Pending |
| 720 | 20.541 | -4.21% | 79.0% | 0 / 0 | Pending |

## Defects corrected

- Long-horizon published charts were empty until the first outcome settled. They now plot original pending medians and ranges, leaving actual prices null until settlement. Late/unverified records remain excluded and pending records never enter metrics.
- All six horizon statuses, fixed prices/ranges, live sample sizes and price baseline comparisons now appear together near the current forecast. The selected card states evidence limitations and identifies a past-frequency model explicitly.
- The complete evidence workspace starts independently of the large legacy replay download.
- Prospective metric normalization now includes each original reference price, restoring dollar MAE for the selected and baseline forecasts. Non-overlap counting sorts origins rather than relying on ledger iteration order.

## Remaining model limits

Historical point MAE is worse than no-change for 3/7/14/30 days. Balanced terminal accuracy is approximately 33–35%. Current live terminal accuracy is dominated by within-threshold outcomes; it is not evidence of directional skill. No long-horizon published forecast has yet settled in this snapshot.

The incumbent alert thresholds are estimated before its final refit. The resulting probability distribution can shift: all 28 settled 3-day observations in this snapshot generated rise alerts despite zero realized rise events. Existing issued alerts and scores are retained honestly. The separate frozen-calibration candidates introduced in PR #30 address this design issue but remain experimental; their point/probability tradeoffs do not justify blanket promotion. A new causal model version and independent prospective validation are needed before replacement. Do not reinterpret original predictions or lower evidence requirements to make results look better.

This change improves availability of evidence and corrects display/reporting defects; it does not claim improved model accuracy.

