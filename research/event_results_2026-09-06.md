# Measured hourly ETH research results — 2026-09-06

These are retrospective development results, not prospective performance or a
certificate of the globally best model. Model selection uses earlier purged
validation; historical data were received during this research and cannot recreate
the original publication vintages. The actual-issued ledger starts separately.

Evidence: [completed research and foundation run](https://github.com/wscha231/eth-dashboard/actions/runs/34047874113).
Protocol and operations: [events/README.md](events/README.md).

## Full eligible history

Coinbase ETH-USD and BTC-USD: 169,621 closed hourly bars initially, 568 requests,
89.68 seconds for the initial collection. This is a combined feed row count.
Training samples every six hours; the evaluation samples one origin per UTC day.
Missing windows, initial training and outcomes that have not matured are excluded.

Each outer month selects the candidate using only its earlier validation window.
Event skill is `1 - selected Brier / frequency Brier`; price skill is
`1 - selected log-return MAE / no-change MAE`. Positive values favor the selected
model. Both comparisons use the same eligible origins within each horizon.

| Horizon | Common origins | Event probability skill | Price MAE skill | Up recall | Down recall | Nominal 80% coverage |
|---|---:|---:|---:|---:|---:|---:|
| 6h | 2,632 | +6.37% | +0.45% | 14.36% | 12.06% | 82.52% |
| 24h | 2,616 | +2.41% | +0.22% | 17.38% | 20.35% | 79.51% |
| 72h | 2,590 | -0.65% | -1.11% | 27.03% | 27.12% | 79.34% |
| 7d | 2,531 | -1.51% | -1.92% | 34.97% | 41.98% | 77.16% |
| 14d | 2,254 | -1.03% | -2.51% | 23.02% | 31.91% | 80.08% |
| 30d | 2,096 | -3.73% | -4.24% | 27.46% | 22.18% | 78.96% |

The exploratory paired block-bootstrap intervals for selected-minus-baseline
event Brier exclude zero in favor of 6h and 24h. The 30d interval favors the
baseline. The intermediate horizons do not establish an improvement. Recall is
for path barrier hits, not the earlier daily model's terminal direction target.
Alert false-positive targets are chosen in earlier validation; realized future
false-positive rates differ and are shown alongside recall on the site.

## Foundation and price blend diagnostics

The predeclared diagnostic has at most 100 evenly spaced historical daily origins
per horizon and 512 prior hourly inputs. No future covariates are provided. Unknown
foundation pretraining overlap prevents treating these results as untouched tests.
Checkpoints, licenses and exact revisions are recorded in the artifacts.

| Candidate | 24h price skill (N=100) | 72h price skill (N=99) | 7d price skill (N=95) | Total CPU diagnostic |
|---|---:|---:|---:|---:|
| Chronos-2 | -2.83% | -7.85% | -6.98% | 60.71 s |
| TiRex-2 | +2.88% | +3.39% | -3.27% | 117.71 s |

The TiRex-2 CPU compiler dependency was corrected by matching Triton 3.4.0 to
Torch 2.8. Both final diagnostics completed without model errors. Peak RSS was
about 1.21/1.39 GB (MiB-based runtime report); neither runs in the hourly publisher.

On matched origins shared with the base replay (N=90/89/87), a TiRex-2/base price
blend using only previously matured errors to choose weights 0/0.5/1 achieves
price skill +1.71%/+1.87%/-2.23% for 24h/72h/7d. At least 20 earlier paired outcomes
are required before changing from the base model. Nominal 80% coverage is only
75.56%/73.03%/74.71%. Chronos blending does not improve the matched base result.
These are small price diagnostics, not calibrated path probabilities; the blends
are not automatically promoted to the public event engine.

## Computation and deployment boundary

- Corrected full replay: 535 monthly fits, 1,292.44 s (~21.5 min).
- First unchanged replay: zero fits, 535 reused months, 284.32 s (~4.74 min).
- Final unchanged replay: zero fits, 535 reused months, 290.13 s (~4.84 min).
- Hourly production code prohibits training and restores only active checkpoints.
  Actual hourly inference and external release checks are recorded by the publisher.
- Earlier daily hybrid models stop issuing new forecasts; their issued outcomes
  continue to settle with current truth provenance and remain archived.
- This release deploys a public research beta. Paid access, payment webhooks,
  a production worker/storage SLA and verified prospective edge remain separate
  gates described in the operations README.
