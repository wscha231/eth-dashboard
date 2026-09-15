# EtherForecast DATA → MODEL contract (draft v1)

Status: MODEL consumer proposal, not an authoritative DATA release. Existing production
collectors, shared formulas, Drive archive engine, ledgers, workflows and website are unchanged.

## Ownership and immutable inputs
DATA owns raw observations/revisions, source rights, shared deterministic formulas and releases.
MODEL owns restored-input consumption, labels, fold-local fitting/calibration, experiments and
output validation. The consumer imports `signal_pipeline.data.build_features` rather than copying
a collector or maintaining a parallel common feature definition. Its source SHA256 is checked.

`model_lab/contracts.py` requires release_id, schema_version, dataset_manifest_id/hash,
feature_set_id/version, code_sha, data_cutoff_utc, generated_at_utc, source_versions,
coverage_by_feature, quality_status, vintage_mode, license_status, partition_locations,
checksums, exclusions and breaking_changes. Every input path is relative, checksum-pinned and
confined to the restored root. Canonical release bytes remain fixed even if a latest file changes.
SQLite is opened read-only; source writes, model pickle execution and network calls are absent.

The current consumer receipt is an explicit reconstructed, partial-quality, rights-unreviewed
research input. It must not certify actual live use. DATA must replace/approve it with its own
release and per-feature registry before integration. Passing input checks is not a model edge.

## Per-feature contract still requiring DATA agreement
For each feature provide feature_id, definition/formula_version, source/venue/instrument/unit,
frequency/lookback, event_time, published_at, available_at, ingested_at, revision,
first/last_valid_time, missing_reason, max_staleness, kind, research_status, allowed_use and
evidence_ids. Not-applicable history, source outages and true zero must remain different.
This phase validates the shared function fingerprint and computed coverage; it does not pretend
that the complete per-feature publication/vintage registry already exists.

## Frozen targets and units
Hours are 6/24/72/168/336/720. Coinbase ETH-USD; timestamp means candle OPEN time.
Reference is the close at UTC slot t. Actual issue must be before t+55 minutes.
Window opens at t+1h and ends at t+(h+1)h. Thus endpoint log return spans h+1 hours,
while high/low path events span h candles. Existing `hourly_events_v1` labels and ledgers
are preserved. The added endpoint/variance description is `model_lab_hourly_endpoint_v1`.

Endpoint output: q10/q50/q90 prices, median simple return, separate zero-threshold direction.
Existing terminal down/inside-barriers/up probabilities are NOT sign probabilities; a wide
inside-barriers region is not automatically sideways. Path up/down hits can both occur and
their probabilities must NOT sum to one. Variance is the sum of h squared close-to-close log
returns within the path window; the benchmark Gaussian endpoint conversion explicitly uses
(h+1)/h. Missing intermediate truth bars stay unscored, never negative or zero.

## Time and inference boundaries
Only explicit timezone-aware timestamps are accepted. Source receipts are normalized to UTC
nanoseconds for the existing feature builder. For actual-vintage reads, all candidate revisions
are retained until an exact timestamp filter is applied BEFORE revision selection; this avoids
SQLite julian-day sub-millisecond rounding admitting a future correction.

Training rows must have target_end < next_boundary minus one hour. The one-hour gap retains
the legacy scheduling guard; it is not a claim that overlapping returns become independent.
The study uses strictly forward folds, not random split. All standardization, residual-quantile
calibration and sign-probability calibration are fitted only on prior train/calibration samples.

## Promotion and operations
Every current result is historical development evidence. No output can trigger promotion,
trading, live ledger writes or website replacement. Point MAE, distribution WIS/coverage/width,
variance QLIKE and directional Brier are separate decisions. Inconsistent median/sign outputs
are blocked by the prediction contract. Failed and existing experiment directories are preserved.

Production integration remains a separate reviewed change: DATA release acceptance, richer
same-sample feature tests, pre-registered monthly walk-forward policy, independent prospective
issuance using existing ledgers, mature-label recalibration and public-payload verification.
Continue to use `scripts/gdrive_store.py`, existing Drive streams and existing scheduled jobs;
this PR neither duplicates them nor silently changes their schedules.
