# DATA research release v1

DATA owns this additive package. It does not modify production collectors, common
incumbent formulas, MODEL files, targets, ledgers, workflows or the website. Main's
`signal_pipeline.data.build_features` is imported and its exact source hash checked.

## What is delivered

`python -m data_lab build` accepts a restored input bundle, the expected input-release
SHA256, the original DATA research handoff ZIP and its expected SHA256. It performs
read-only SQLite integrity/OHLCV/revision/UTC checks, verifies all input partitions
and every member of the original handoff manifest, generates per-column coverage,
registers the unchanged 32 incumbent features and 18 **new hourly** OHLCV research
extensions, then writes an immutable DATA release. The new 18 columns are not the
older 18 daily pilot columns. The original 30 research candidate definitions are
preserved as a separate catalog, not advertised as 30 additional ready features.

The release contains a standalone source DB and replay subset, original handoff,
source receipt, registry, coverage/missing/latency audit, exact availability sidecar,
and extension snapshot. It deliberately does not duplicate the incumbent snapshot:
MODEL reuses the original builder. CSV gzip headers are deterministic; numeric
snapshots use 17 significant digits with round-trip reading. Parquet is not required.

`release.json` is compatible with PR53 `PinnedRelease`. All files and their lengths
are pinned by `dataset_manifest.json`. A release is written in a temporary directory,
validated, and installed with an exclusive destination lock. Existing releases and
source DBs are never overwritten. A failed build leaves no published release.
Delete only a *verified abandoned* `.lock` after ensuring no other writer is active.
Existing ef1 Drive blobs/manifests remain the archive engine; this code does not
create another Drive engine, change archive schedules, or claim archival success.

## Authority and limits

This is an authoritative DATA **research delivery**: byte lineage, definitions and
coverage are supplied by DATA. `quality_status=partial`, `vintage_mode=reconstructed`,
`license_status=unreviewed_no_public_redistribution` remain explicit. The builder cannot
turn these flags green. MODEL's prospective authorization must continue to fail.
Historical values received in September 2026 are not original historical receipts.
Receipt-minus-close statistics include backfill and are NOT a live API latency SLA.
The 3,600-second feature staleness limit is a proposed policy, not a measured SLA.
Commercial/public/live rights and full restoration of original Drive-backed SQLite
remain unverified. All validation outcomes are distinct from predictive performance.

## Optional feature consumption

Keep PR53 `load_bars` / `consume_features` unchanged for incumbent reproduction.
To test the optional new columns:

```python
from data_lab.release import load_feature_snapshot
frame = load_feature_snapshot(root, expected_release_sha256, family="extensions")
feature_columns = [c for c in frame if not c.endswith("__available_at")]
```

Snapshot mode is retrospective only. Timestamps are metadata, never model inputs.
Each column has its own complete-window and exact integer-nanosecond receipt maximum.
Never make all research features mandatory and shorten the entire dataset by a joint
`dropna`. Compare additions on matched dates; service evaluation also counts fallback.
No normalization/feature selection/threshold fitting is embedded in these formulas;
MODEL must fit statistical transforms on training folds only. Rolling beta is a
past-only local estimate, with its 721-bar dependence recorded.

The incumbent's float-based rolling time maximum can round up/down by nanoseconds.
`incumbent_availability_exact.csv.gz` preserves exact receipt maxima without silently
changing legacy outputs. This is a timestamp precision issue, not a demonstrated
explanation of poor price forecasts. Existing snapshots remain research-only.

## Auxiliary time alignment and append-only observations

`MetricSpec`, `normalize_observation`, `ObservationStore` and `point_in_time_join`
accept explicit provider/venue/instrument/unit, event/publication/availability/receipt,
revision, value, missing reason, data kind, vintage status, usage rights and raw hash.
Publication may be unknown and remains null. First eligible time includes ingestion.
The journal stores only allowlisted contract fields, rejects conflicting revisions,
and rolls back the whole batch on conflict. Identical retry payloads are idempotent.
A collector must retain first receipts for repeated unchanged values, or issue a new
explicit revision when changed; it must not overwrite an existing revision timestamp.

As-of selection filters eligibility **before** revision selection and picks the latest
eligible *event*, then latest eligible revision. A correction to an old month does not
replace a newer month. The latest null blocks silent carry-forward. Real zero is not
missing. Age is measured from event time, not recent retrieval. Reconstructed,
estimated/synthetic and unapproved-rights records are blocked by the strict reader.
Monthly indicators require a source-specific age threshold tied to their calendar;
this module does not guess release dates or revive expired values.

This normalized journal is a staging input, not a replacement for original raw
receipts or ef1 durable storage. New auxiliary feeds, actual API pagination, source
licensing and automatic collector integration are NOT completed by this delivery.

## Verification

```bash
python -m pip install -r requirements-data-lab.txt
python -m pytest tests/test_data_lab.py -q
python -m data_lab verify --root RESTORED_DATA_RELEASE --sha256 RELEASE_SHA256
```

A full repository installation supplies the existing incumbent builder. Local
validation may use an exact connector-read upstream function extract; disclose this
and verify its fingerprint, do not describe it as a full repository CI run. MODEL
PR53 is optional for standalone DATA unit tests, required for six-horizon integration.

## MODEL acceptance / next requests

D001: DATA release + per-feature registry + original handoff are delivered; MODEL must
acknowledge exact ids/hashes. This is not live-readiness acceptance.
D002: validation, units, missingness, receipt statistics and block rules are supplied;
original publication vintages, rights clearance and live SLA remain open.
D003/D004: funding/OI/basis and macro/on-chain real source expansion remain open.
Use this contract for later source adapters rather than synthesizing missing history.
Preserve long-history base models while evaluating shorter auxiliary datasets.
