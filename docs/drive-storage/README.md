# EtherForecast durable Drive storage

Storage root: `EtherForecast-Storage` (folder ID `1DFboZtT7PhrMHd4JckLUcs7VvSFXBEZP`).
The existing individual OAuth secrets or the single `[gdrive]` config in
`GDRIVE_RCLONE_CONFIG` / `GDRIVE_REFRESH_TOKEN` are reused. No new credentials are needed.

## What is stored

| Stream | Contents | Intended reuse |
|---|---|---|
| event-hourly | Consistent issuance/observations DBs, active models, replay/diagnostics, raw receipts | Automatic fallback when Actions hourly artifacts are unavailable |
| event-ledger | Issued forecasts, outcome revisions, verified delivery records, long-term input vintages and hash-addressed models | Audit and past-decision reconstruction |
| daily-data | Daily CSVs, availability metadata, normalized source caches, retired predictions DB and generated report JSON | Automatic fallback if the daily data git branch cannot be fetched |
| event-research | Successful historical source/replay/model artifacts | Restore for offline research and retraining |
| event-research-partial | Failed/interrupted research checkpoints | Manual diagnosis; never selected as actual hourly state |
| historical-state / historical-report | Resumable candidate progress and completed study results | Continue or compare historical experiments |
| adaptive-report / adaptive-rows | Delayed-feedback study reports and row-level evidence | Recompute metrics without changing issued forecasts |
| hybrid-data / forward-data | Existing archived model families and their settled records | Compare older models against subsequent outcomes |
| foundation-chronos2 / foundation-tirex2 | Available foundation candidate diagnostics | Research comparison only |

Actual issuance and retrospective research stay in separate streams. All original
timestamps, outcome revisions, model version hashes and source receipt times are
preserved as data. Backfilled history is not relabeled as information available in the past.
The archive does not itself change model selection, accuracy or historical evaluation rules.

## Automatic operation

`Durable Google Drive archive` runs after the approved production forecast, daily
collection and research workflows finish (including failures with usable artifacts).
It also sweeps every six hours at UTC minute 19 and supports manual `Run workflow`.
Initial deployment performs a sweep of still-available trusted artifacts and current
data branches. Already expired/deleted historical artifacts cannot be reconstructed.

The hourly workflow saves a final SQLite snapshot after publication, including
delivery receipts. The daily workflow saves a data-only snapshot even if git publication
fails. These staging artifacts last seven days; durable Drive objects have no automatic
expiry. Existing GitHub storage remains operational if Drive is unavailable.

Only completed runs from this repository, `main`, approved event types and exact
workflow paths are accepted. Pull-request artifacts and their code are never executed
with Drive credentials. Data branches are fetched and archived at pinned commit SHAs.

## Storage format and safety

- Each regular data file is split into 4 MiB chunks, deterministically gzip-compressed,
  and stored once under `ef1-blob-<sha256>`. Unchanged chunks are reused across snapshots.
- Uploads use at most four workers and eight pending chunks. Identical in-flight
  chunks share one request; every worker must succeed before the manifest is committed.
- Original logical file paths, full SHA256 checksums, chunk references and provenance
  are recorded in `ef1-manifest-<stream>-<source-key-hash>.json`.
- The manifest is written last, after every referenced upload passes byte-size/MD5
  verification, and then read back. Incomplete uploads cannot become restore points.
- Snapshot objects and manifests are never overwritten, trashed or garbage-collected
  by this implementation. Retrying an interrupted backup reuses completed objects.
- SQLite uses online backup (including committed WAL changes), integrity checks and
  standalone journal mode. Source DBs are not modified by backup.
- Restore checks path safety, size/decompression bounds, chunk and full-file SHA256,
  and SQLite integrity in a temporary directory before installation. It refuses to
  overwrite nonempty destinations. Joblib models are stored/restored as bytes, never
  executed by the archival/recovery workflows.
- OAuth values, token responses, signed download URLs and server error bodies are not
  included in normal logs or manifests. Secret/token/config files and executable source
  files are excluded from captured data.

## Use previous data

Open Actions -> `Restore archived Drive data for research` -> Run workflow, choose the
stream and optionally a UTC cutoff, for example `2026-09-14T00:00:00Z`. The newest committed
snapshot with source snapshot time at or before that cutoff is restored and verified.
The resulting `drive-restored-*` artifact is a seven-day working copy. Drive retains the
original snapshot. Restored research does not replace production ledgers or models.

Equivalent commands in an authenticated runner:

```bash
python scripts/gdrive_store.py list --stream event-research
python scripts/gdrive_store.py restore --stream event-research --target restored-research
python scripts/gdrive_store.py restore --stream event-hourly --as-of 2026-09-14T00:00:00Z --target historical-hourly
```

Snapshot time is provenance for selecting a saved version, not a substitute for the
model's point-in-time source/label availability filters when doing a new backtest.

## Confirm operation and handle failures

Archive job summaries report new snapshots, uploaded/reused chunks and failures by
source. Each archival run performs a full download/SHA/SQLite recovery drill of the
latest hourly snapshot when one exists. A failing archive is visible as a failed
Actions run; it is not treated as a successful backup merely because forecasts worked.

If a backup hits its time budget, rerun it: uploaded content is reused. Scheduled sweeps
catch still-retained artifacts from missed or coalesced workflow events. If the selected
Drive snapshot is corrupt or incomplete, restore fails instead of silently substituting
older data. Existing GitHub recovery remains separate.

No automatic expiry is not a guarantee of infinite capacity or account availability.
Drive quota, revoked OAuth access, or External/Testing token expiry can interrupt writes.
Monitor archive job failures and Drive capacity. Deleting a shared `ef1-blob-*` object
can break multiple historical snapshots, so do not manually prune these objects.
