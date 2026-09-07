# Period and origin-state evaluation

This extension diagnoses the existing monthly selected models. It does not choose
a new winner from the best-looking historical year or market state.

The replay retains one eligible origin per UTC day at 00:00 for each of 6, 24, 72,
168, 336 and 720 hours. It restores compatible monthly fits and checks that both
training and validation target ends precede that month's cutoff by more than one
hour. The CSV audit records those boundaries and the selected model version.
Historical source revisions and original historical receipt timestamps cannot be
fully recreated; the result remains retrospective development, separate from real
issued and timely published records.

## Fixed diagnostic slices

- Full eligible history, each calendar year, and the last 30/90/365 calendar origin
  days, anchored to **the same source as-of date across all horizons**.
- Trend at the origin: preceding 24-hour ETH log return divided by preceding
  720-hour hourly return volatility times sqrt(24). Above +1 is up, below -1 is
  down, otherwise range. These are observed preceding movements.
- Volatility at the origin: preceding 24-hour / 720-hour hourly return volatility.
  High is at least 1.5; otherwise normal. Missing history is unknown, not normal.
- Each period is crossed with each state filter. Trend and volatility slices
  overlap; they must not be summed or treated as independent experiments.

Only complete, matured target windows enter a comparison. Within a horizon every
model uses the intersection of the same origins and the same realized outcomes.
Duplicated model/origin rows and inconsistent labels fail the evaluation. Recent
30-day forecasts generally have no matured origins inside the last 30 origin days;
the page shows zero eligible observations instead of moving that interval backward.
Initial training history, missing source windows and pending labels are excluded.
Different horizons may have different eligible date ranges and sample counts.

## Metrics and interpretation

- Event Brier averages independent up/down barrier probability losses; lower is
  better. Skill is `1 - candidate Brier / matched frequency Brier`.
- Report up/down recall together with false-positive rates. Thresholds remain
  those fixed in the earlier purged validation, not retuned on each slice.
- Terminal balanced accuracy averages recall for down/flat/up; it is unavailable
  when any of the three realized classes is absent.
- Price MAE is absolute **simple-return** error, compared with predicting no price
  change. Also report nominal 80% interval coverage. The price chart plots the
  prediction and realization at the same target end, using USD or origin-relative
  simple returns. It does not rebase old forecasts to today's price.
- Nonoverlapping origin counts and matched baseline results accompany the dense
  daily-origin scores. Exploratory paired calendar-block bootstrap intervals use
  at least the target interval, and are unavailable with fewer than 60 origins or
  10 calendar blocks. Many slices are inspected; these intervals are not corrected
  discovery tests and cannot establish prospective superiority or authorize promotion.

## Updates and cost

The scheduled weekly research review and monthly checkpoint workflow rebuild these
aggregates using compatible cached models. Hourly collection and inference publish
the most recent complete research result without recomputing its historical scores.
Evaluation/UI-only changes skip unrelated foundation diagnostics. A model-training
implementation, protocol or relevant source change can invalidate its fit cache.

Full model-origin audit rows remain in `event-research-state/replay/h*.csv.gz`.
The site publishes compact plot points plus precomputed matched summaries, and
`signals_segments.csv` provides all period/state/model comparisons for download.
Empty slices appear in that export with zero observations and blank scores.
