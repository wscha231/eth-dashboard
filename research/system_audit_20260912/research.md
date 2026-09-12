# ETH system audit - 2026-09-12 UTC

Baseline: `e09dd5d0d8ed770fecf96d27d7cd08a4379470cc`, wscha231/eth-dashboard.
Request: recheck each forecast horizon, storage, automated checks and data updates.

## Measured production state

The deployment payload and trusted hourly artifact from run [34713449730](https://github.com/wscha231/eth-dashboard/actions/runs/34713449730) agree: UTC 19:00 slot, generated 19:12:28, ready, all six incumbent and six experimental horizons, no issuance/source/shadow errors. The runner's external verifier confirms the public site at 19:13:44, including all six archive horizons and 15,667 evidence rows. Direct site access from this workspace was unavailable; this is runner verification plus the corresponding GitHub deployment payload, not a browser inspection. Collection: two Coinbase requests / 0.60 seconds; inference: 1.74 seconds. Daily master: 3,229 rows, 118 columns, through 2026-09-11.

| Horizon | Live settled / nonoverlap | Return MAE (pp) | No-change MAE (pp) | 80% interval coverage |
|---|---:|---:|---:|---:|
| 6h | 109 / 20 | 0.8618 | 0.8610 | 89.0% |
| 24h | 95 / 5 | 1.4283 | 1.4207 | 95.8% |
| 72h | 54 / 1 | 1.4760 | 1.4589 | 100.0% |
| 7d | 0 / 0 | pending | pending | pending |
| 14d | 0 / 0 | pending | pending | pending |
| 30d | 0 / 0 | pending | pending | pending |

These are overlapping real published outcomes, not independent trials. All short-window terminal decisions in this snapshot are flat; headline accuracies of 86-96% do not demonstrate directional skill. The 6/24h balanced accuracy is 33.3%; 72h is undefined because a class is absent. Latest same-origin candidate comparisons have only 5/1 independent 6/24h windows. Do not promote any candidate on this evidence. Historical point errors at 3/7/14/30d also failed the no-change comparison in the September 11 audit.

## Actual data and automation topology

- Active engine: `signal_pipeline`, hourly closed ETH-USD + BTC-USD Coinbase bars, 720-hour warmup; all six horizons 6/24/72/168/336/720 hours.
- Current model registry: CatBoost 6h, calibrated CatBoost 24h/14d, past-frequency climatology 3d/7d/30d. Incumbent checkpoint cutoff is the first day of the month. Hourly inference does not train. Weekly research and first-of-month research refresh saved models; experimental heads remain separate.
- Daily `eth_data_collector.py` collects market, macro and crypto source history into a daily master. `research_pipeline` collects daily flow. These auxiliary columns are not inputs to the current hourly model. There is no rclone/Drive synchronization in these active workflows.
- Public site: `data/daily-forecast`; original issuance/settlement/publication receipts: `data/event-ledger`; retired flow and hybrid records have separate data branches.
- Hourly artifact: two-day retention and only 40 days of observations. Daily backup: 30 days, only UTC midnight or research completion. Research artifact: 30 days. Neither artifacts nor caches are permanent storage.
- The enabled account-level hourly recovery automation already checks the artifact-free watchdog. No duplicate automation is needed.
- At 18:33 the watchdog correctly detected a missed slot and requested a new run. The fresh producer verified ready at 18:35. A failed watchdog after dispatch is intentional incident reporting, not a failed repair.

## Confirmed defects

1. Hourly restoration only looks for `event-hourly-state`; `event-daily-backup` is never read. An outage over two days can drop operational observations despite a retained backup.
2. Backup creation depends on a run landing in UTC hour zero. Delayed/missed cron runs can skip that day's backup. Trigger delay is material: many recent hourly runs required watchdog recovery.
3. Research refresh copies observations into hourly state, but hourly source revisions are not archived back. The 40-day pruning eventually removes actual receipt vintages. Re-fetching prices cannot reconstruct what was known at issue time.
4. Daily jobs restore only the gold dataset, not the vendor CSV caches or availability metadata. Snapshot-only sources (e.g. Deribit options) depend on those caches. The code supports incremental histories but runner persistence does not preserve them. Detailed source/schema/exclusion reports are generated but not persisted.
5. Updating homepage HTML unnecessarily triggers the retired daily source job, competing with hourly and search publishers for the one shared pending concurrency slot. Several HTML-only pushes show cancelled hourly runs.
6. README starts with unrelated stock/Colab instructions and describes retired 7/30-day models as the active service.

## Attached papers

Read the three supplied PDFs locally; no external redistribution of their text.
- Wu et al., arXiv:2405.11431v2: compares neural architectures across crypto assets and pre/post-COVID regimes; multivariate convolutional LSTM performed best in their experiment.
- Badar et al., Mathematics 13 (2025) 1908: VAE + CNN/LSTM + SHAP on Bitcoin; normalized MSE/R2 results are not comparable with current ETH live return errors or barrier-event calibration.
- Bouteska et al., IRFA 92 (2024) 103055: compares ensembles and recurrent models across BTC/ETH/XRP/LTC, including naive benchmarks; GRU/RNN/LightGBM are useful candidate families, not a production-promotion argument.

The papers report different winning models and targets. Preserve a common chronological, purged, cost-aware evaluation before changing the deployed family. The next useful experiment is point-in-time derivatives/flow features plus simpler baselines, followed by bounded GRU/LightGBM challengers; a more complex network is not a demonstrated fix for the observed flat decisions.

## Storage decision

Keep the existing code repository and data branches. Store incremental, content-addressed public Coinbase receipts, source-vintage partitions and referenced model checkpoints alongside the durable audit ledger; retain rolling execution snapshots for fast recovery. Daily normalized vendor CSV caches belong with the existing daily data branch. Do not store tokens, manual private files or arbitrary runner directories.

This is a modest-scale continuity repair. Long-term full research archives and growing raw/model blobs should later move to dedicated object storage or an explicitly configured Drive archive, with checksums and tested restoration. Git history grows and 90-day artifacts still expire. No new external service, credentials, paid storage, or perpetual full-research backup is claimed.
