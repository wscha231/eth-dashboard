# Forecast, public feed and Drive integration

## Durable state

Reuse the existing private `EtherForecast-Storage` connection and credentials.
Preserve closed-bar observations and raw source payloads, frozen models and research
checkpoints, actual issuance and outcome/delivery ledgers, reports and public evidence.
The new short-range candidate has its own `range_shadow/issued.db`; snapshots and
the durable ledger branch retain it and all `range_seeds` revisions. Never replace
actual issuance with reconstructed backtests. Google Drive is durable storage, not
an inference worker or a point-in-time correctness guarantee.

Archive trusted workflow artifacts on completion and run a six-hour catch-up sweep.
Add `data/event-feed` as the `public-feed` stream; retain the other existing streams.
Use immutable manifests and SHA256-addressed chunks, deduplication, SQLite online
backup/integrity validation, and empty-target staged restoration. The existing archive
job performs a full hourly-state restoration drill and writes an exact Drive receipt.
No automatic expiry or deletion is configured. Retention remains subject to the
owner's account, quota and access; it is not a guarantee of permanent service.

Hourly recovery first selects the newest trusted final/hourly/backup artifact, then
uses Drive if disposable artifacts are unavailable. Required active models and replay
must still pass checks. Drive failures are visible and cannot silently certify
successful restoration; archiving does not block an otherwise valid public forecast.

## Update reliability

`data/event-feed` contains approved public JSON, CSV and evidence archives. Its Git
configuration disables deployments on every branch. The website uses same-domain
external rewrites to this public feed. No private Drive links, credentials, databases
or model bundles are exposed. The initial routing/UI change requires a static site
deployment; subsequent hourly data changes do not. Static updates still respect the
existing provider cooldown and are retried by a later normal publication.

Publication compares the actual public payload and interface, verifies every referenced
archive hash/count, and only then records delivery receipts. A stored feed commit
alone is not proof that CDN publication succeeded. GitHub scheduling, upstream cache,
CDN and network availability can still delay updates. The bounded GitHub watchdog
checks worker concurrency, deployment cooldown, deduplication and issuance deadlines
before requesting any recovery. The independent ChatGPT check reads state first and
may rerun only that artifact-free watchdog when genuinely overdue and no watchdog
is active. It must never rerun a producer job with immutable artifact names.

## Range improvement

See [the fixed range policy and retest](../../research/range_shadow_20260914/README.md).
Only 6-hour, 1-day and 3-day ranges are issued as a prospective comparison. Preserve
the incumbent center/probabilities and all longer-horizon behavior. Actual live losses
can update subsequent weights only after outcomes are mature and available. Promotion
requires new forward evidence; it is not an unattended automation action.
