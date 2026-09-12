# Hourly publication recovery

The September 7 incident was an hourly scheduling gap, not slow model inference.
Both the producer and its scheduled watchdog missed several hours; a daily data
refresh did not check hourly publication. A manual watchdog rerun restored all six
horizons. This change addresses delivery reliability; it does not establish better
predictive performance or a paid-service uptime guarantee.

## Triggers and bounded recovery

- The hourly producer remains scheduled at :08 UTC and now also deploys relevant
  publication/workflow changes pushed to `main`.
- The watchdog runs at :38 and on completion of the hourly forecast, daily market
  collection, daily source refresh and archived settlement. It checks the actual
  public JSON, HTML and JavaScript. Only same-repository, non-PR workflow events
  from `main` are accepted; checkout is explicitly `main`.
- A healthy release has all six distinct horizons and distinct issued IDs, an
  aligned slot, matching input cutoffs and valid timezone-aware timestamps. At
  :15, the current UTC hour becomes due. Before :15 the previous slot is allowed.
  The existing 100-minute maximum age and two-minute future clock tolerance are
  hard limits. Publication still verifies the exact expected release and IDs.
- On failure, recovery first checks queued, running, requested, waiting and pending
  work, including **all writers using the shared `daily-forecast` concurrency group**
  and event research. Another dispatch must not replace an existing pending writer.
- The first failed normal run can cause one fresh `workflow_dispatch` on `main`.
  A dispatched hourly attempt imposes a 20-minute cooldown from its creation,
  actual attempt start or latest update. This prevents completion-triggered retry
  loops. No new recovery is dispatched at :50–:59, reserving five minutes before the :55 new-issuance deadline.
- A watchdog that found a problem stays failed, including when recovery was
  requested or deferred. Its job summary says `active_worker`, `cooldown`,
  `issuance_deadline` or `dispatched`. A dispatch is not proof of restored service.
  The new hourly run must complete external release verification.

## Independent account-level backup

The account-level **ETH 갱신 복구 점검** automation uses an hourly ChatGPT condition
watch, separate from GitHub's cron trigger. Its recurring iteration invokes one
completed, artifact-free watchdog job through the connected GitHub job-rerun API.
The repository's deterministic watchdog performs the actual health decision and
bounded recovery. Skip invocation if a watchdog attempt is already active. Report
only a recovery or a remaining incident, and link the actual Actions evidence.

This account automation is configured outside Git. When transferring ownership,
recreate it under the new connected GitHub account and disable the old copy.
If that connection expires or the automation is disabled, the independent backup
stops. It is a second trigger, not a hard real-time scheduler or an uptime SLA.

Do not rerun an old hourly producer job for routine recovery: its already uploaded
immutable artifact name may conflict. Rerun the artifact-free watchdog, which
dispatches a **new** hourly run and uses current `main` code. Do not use transient
Actions tokens outside the runner or load model state from an untrusted run.

## User-visible behavior and incident checks

The page checks its local clock every 30 seconds, independently of fetch success.
Missing current slots become a delay warning after :15. Failed fetches identify
the last received data; the original issued prices remain fixed. A new valid feed
clears the warning. Source, generation and next expected update times stay visible.

1. Read the watchdog reason and the newest hourly/source/research job state.
2. If deferred, allow the active job or 20-minute cooldown to finish. Investigate
   long-running workers rather than cancelling them or resetting their audit data.
3. On a fresh hourly run, inspect source errors, model restoration and publication
   separately. Its log must say `Event site verified`, `ready`, the correct slot and
   `horizons=6`. Inspect the actual served release when available.
4. Preserve issued IDs and their original prices; only externally verified delivery
   receives a receipt. Missed slots cannot be backfilled as real-time predictions.
5. If model/source readiness is the blocker, retain the delay label and diagnose
   it. Recovery does not train models, promote candidates or weaken readiness gates.

Validation covers missing/future/mixed-slot releases, all six horizons, the :15
boundary, every active worker state, retry cooldown including old-run new attempts,
the issuance deadline, offline page aging and recovery of the freshness display.

## Source continuity and routine audit

The hourly worker restores the newest trusted `event-hourly-state` or
`event-daily-backup`, requiring the same repository, main branch and producer
workflow, excluding pull-request runs. Failed publication does not invalidate an
already uploaded consistent snapshot; original delivery receipts are restored from
the durable ledger. A newer successful research run can update checkpoints and
inputs. A recovered complete snapshot can operate without an unexpired research
artifact only while its exact protocol, training hash and current-month checkpoint
remain valid. The inference gate still refuses stale or incompatible models.

At the first actual run of each UTC day, save a 90-day backup if none exists for
that date. Hourly snapshots retain two days. These are expiring recovery copies,
not permanent full research archives. Before public release, the existing ledger
writer also saves `lake/event-inputs` on `data/event-ledger`: compressed partitions
of actual observation receipt dates, raw Coinbase receipts and content-addressed
active checkpoints. Existing partitions are unioned, never replaced by the rolling
40-day subset. Historical backfill retains its actual later receipt time.

`system_audit.json` and `system_audit.md` accompany each hourly snapshot and the
Actions step summary. They report six-window availability, source cutoffs,
settled/non-overlap counts, price MAE versus no-change and pending long windows.
An operationally ready system may still have no demonstrated predictive edge.
Daily macro/on-chain/flow sources remain outside the current hourly feature set.

The daily collector restores/persists normalized market/vendor CSV caches and
availability metadata. No private manual inputs, keys, or arbitrary raw files
are copied. Homepage-only edits no longer start retired daily source collection.
Continue monitoring Git archive growth; use dedicated object storage for sustained
large historical/model retention. No external archive service is configured here.
