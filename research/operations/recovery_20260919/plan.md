# EtherForecast recovery Wave 1 — 2026-09-19

Baseline: main 27602f0fe1ab9b20fb55c8f1ed6180b559455d6e. Canonical tracker: #68.

## Observed problem, not a performance claim

In run 35403023145, collection, inference (1.622 seconds), settlement, dense 24h/72h variance shadow and immutable ledger persistence succeeded. Feed commit a70ec055186b63fe614fd81c54e5c7bdb853df8b was pushed, but the public verifier repeatedly reported `release mismatch`. Existing logs did not record the observed release ID or response cache headers. CDN/upstream propagation is a hypothesis to diagnose, not an established root cause.

The failed run's final artifact 10570969587 has ZIP SHA256 fffd7806d6bc31bef2ece9b9f631c3f90c480cac847f45464c7ef7b47a1fc8ed. Read-only recomputation as of 2026-09-18T22:51:22Z over the explicit/first-observed window 2026-09-06T17:00Z through 2026-09-18T22:00Z found, for each of the six horizons:

- 294 expected hourly slots;
- 192 stored/issued slots;
- 174 slots with verification receipts before window_start;
- 102 missing issue slots;
- zero outcomes overdue by the operational two-hour settlement grace at that snapshot.

192/294 = 65.31% recorded issuance; 174/294 = 59.18% documented timely public coverage. These are service-coverage statistics, NOT forecast hit rates, not current live figures, and not a claim about activity before the first observed ledger record. Unverified publication might have occurred but cannot be counted as proven timely delivery.

Historical replay exists separately: 6h first origin 2018-05-01 and 2,643 common origins; 30d first origin 2019-12-01 and 2,107 common origins in this replay. Eligible-calendar coverage is incomplete. A retrospective reconstruction must never fill a missing prospective slot or be described as an old real-time forecast.

## Cross-chat reconciliation

- #88 all-clock HAR audit is merged, not Draft.
- Existing hourly workflow already contains dense 24h/72h shadow issuance. Do not duplicate it or extend to 6h without its clock-specific authorization.
- #93 self-derived execution-history code is merged. That is not proof that a local node has supplied the complete historical dataset.
- #94 private derivative storage Stage A is merged. Per-stream restore evidence and public cleanup are separate gates.
- Canonical Drive roadmap 18IIVG_EVFxCdHkyZeXRNj7oy6poBaIYf still contains v1.0; #68 contains stale v1.4 items. Synchronize status without changing frozen model criteria.

## Wave 1 implementation

1. Read-only continuity audit from the existing event SQLite ledger. Expose hourly gaps, timeliness, as-of settlement and distinct model/target versions. Default audit window is explicitly left-censored at the first observed record; support a declared --from-slot for rollout audits.
2. Exact public payload preflight before expensive interface/archive checks. At most 420 seconds, further limited by the forecast's start time. Record actual served/upstream release IDs and cache headers. Upstream agreement is diagnostic only, never publication proof.
3. Keep the original full verifier, all freshness/identity/archive checks, real verification timestamps and failure exit status. Run continuity audit on publication success AND failure; final-hourly-state preserves the reports.

This does not change weights, baselines, event thresholds, data rights, targets, prediction content, old ledgers or promotion criteria. A constant climatology return is reported honestly as a baseline, not forcibly made directional.

## Gates and next work

- Deterministic tests: gaps, duplicate versions, missing and late receipts, future information, maturity, constant baseline, bounded retries, identical-ID/different-payload rejection and original exit-code preservation.
- Existing repository regression CI must pass before merge.
- First post-merge public release must independently pass the existing full verifier before calling the outage repaired. Code merge alone is not that proof.
- Then address measured scheduling/propagation gaps; separate optional research failures from core issuance without hiding their failures.
- Continuous history work: preserve old runs, extend leakage-safe walk-forward evidence by matured origin, record excluded dates, compare no-change/persistence/prior on the SAME origins and keep historical reconstruction separate from actual issued forecasts.
- Model work: use already eligible ETH/BTC inputs to preregister one regime/residual or event-alarm challenger; add only individually rights/PIT/continuity-approved external families. Do not wait for every R3 source, relax gates, retry inspected holdouts as new evidence or equate a HAR variance win with directional alpha.
