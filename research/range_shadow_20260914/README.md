# Short-horizon range-only prospective candidate

The existing forecast remains the production reference. The candidate changes only
the 10th/90th price quantiles for 6, 24 and 72 hours; its median and event
probabilities are copied unchanged from the same incumbent forecast. Longer horizons
are not promoted. The interface labels the three ranges as experimental comparisons.

## Fixed policy and evidence

Use paired incumbent, historical-frequency and optimization-candidate ranges.
Compute their 80% interval scores from mature previous daily origins over 365 days,
with 90-day exponential half-life, at least 90 daily origins and softmax sensitivity
8 relative to the frequency score. Require target end strictly before issuance slot
minus one hour. Live feedback additionally requires actual observation receipt and
outcome evaluation before that cutoff. Use one first-issued pair per UTC day.
Do not widen ranges solely to achieve coverage: score both width and missed outcomes.

The historical bootstrap is a versioned past-loss seed, not fabricated live issuance.
It became available on 2026-09-14 at 06:57:33 UTC. Its source report SHA256 is
2df6c7ab6cdff8d0a670e5c4c31b0dd5a9baaa6de95d7f750f222ace8e991cf7,
from trusted historical run 34595849686, data through 2026-09-11 11:00 UTC.
Protocol, training identity, source counts and original scores were verified before
export. Each issuance retains seed identity, parent forecast IDs, expert quantiles,
weights and mature-feedback cutoff. Later seed revisions cannot rewrite issuance.
The existing adaptive study exports updated seeds; only successful trusted main
artifacts are installed, with version/availability checks and no backward movement.

`retest.json` replays this exact range policy on identical eligible historical
origins, from 2020 through September 2026 after feedback warm-up:

| Horizon | Paired origins | Non-overlapping origins | Incumbent coverage | Candidate coverage | Interval score reduction |
|---|---:|---:|---:|---:|---:|
| 6 hours | 2,082 | 2,082 | 81.41% | 81.99% | 2.85% |
| 24 hours | 2,044 | 1,024 | 79.16% | 80.23% | 1.30% |
| 72 hours | 1,997 | 502 | 78.82% | 79.82% | 2.34% |

Higher coverage alone is not improved directional accuracy or trading profit.
These are previously inspected development data, not an untouched holdout or live
performance. The 30-day 70.1% coverage result belongs to another sample/candidate;
it is not directly comparable to this table. Future-data perturbation at 2024-01-01
left 1,189 / 1,153 / 1,112 earlier candidate outputs unchanged. Per-year and paired
calendar-block uncertainty results are retained in the machine-readable report.

Reproduce with `python scripts/retest_range_candidate.py --root <trusted-study-root>
--output <report.json>`. No full model fitting is needed. Actual live results must
accumulate after candidate publication, with paired and non-overlapping counts.
Missing peer forecasts or incompatible seeds suppress the candidate, not the primary
forecast. No automatic promotion or claimed long-horizon improvement is introduced.
