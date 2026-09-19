# EV2-A — received-input CENTER availability shadow

Frozen before implementation: Issue #68 comment 5743460646, 2026-09-20 KST.
Base: main 42b9848e6fc8be8fb1b99e9fa85d1cca72a0a0a7. This is the bounded follow-on to EV1-A, not an activation of EV1-B/C or a new price-alpha claim.

## Policy

The core ETH+BTC 721-hour continuity rule is unchanged. Only when this continuity condition is not satisfied may the separate shadow emit CENTER-only no-change predictions, provided the most recent 169 ETH bars are contiguous, closed, finite and actually received by the decision clock. This policy diagnoses INPUT availability, not all reasons a core forecast can fail. In particular it does not cover missing models, extreme-feature exclusions, source permission failures, or public-delivery failures.

Every input is selected after filtering receipt time, then resolving revisions. The exact input values must occur in the SHA256-verified gzip raw source response and have a matching Coinbase download receipt. Invalid records are not treated as innocent gaps. Missing recent ETH, stale endpoint, missing raw provenance, integrity failure or a clock at/after minute55 blocks issuance. Current origin only; no operational historical-clock CLI.

The six horizons remain 6/24/72/168/336/720 hours. Existing h+1 target timing is preserved: input origin, next-hour window start, endpoint origin+(h+1). All six center values equal the observed ETH reference price. Event, Distribution and Variance are null. No direction/interval/volatility prediction is fabricated.

## Evidence boundaries

- `availability_shadow/issued.db` is separate from core, dense variance and all existing shadow ledgers.
- Policy, decisions, forecasts and endpoint outcomes are append-only; duplicate origin/horizon attempts cannot replace an earlier prediction.
- Fingerprints include actual receipt/revision/raw identities; trigger gaps have a separate hash.
- Outcomes require the exact endpoint bar plus a genuine receipt no later than evaluation. Revisions append; a nearby endpoint is not substituted.
- No current `signals.json`, website, model manifest, weights, alert thresholds or source-rights registry is changed.
- Shadow issuance is NOT public delivery. It cannot be added to public timeliness, old historical issue counts or established skill cohorts.
- The earlier EV1 82.89%→96.62% estimate was retrospective input coverage and is NOT asserted to be live uptime or accuracy.

## Operational wiring

Use the existing single hourly workflow and shared `daily-forecast` writer lock. The optional stage runs after core publication, including after attempted core inference failure when a valid observation DB remains. It does not run after a failure before inference is attempted. Thus this patch does not repair all scheduling or pre-inference outages.

Restore only the fixed trusted audit subtree `lake/event-ledger/availability-shadow`, preserving a newer local final snapshot by row-set comparison. An existing incomplete/divergent state fails closed. First initialization is allowed only when no subtree and no local state exist; simultaneous total loss cannot be inferred from absence alone and requires external archive recovery.

Only this subtree is committed to `data/event-ledger`; no force-push and no unrelated staged changes. Current-run status records restore/run/persist failures. The optional final export retains its own SQLite backup and manifest with existing `event-final-state`, while a failure cannot prevent the core artifact upload. Optional failures still fail the final health gate after evidence retention. Existing archive machinery can preserve the extra files as part of its existing stream; verify the actual Drive receipt separately.

## Validation performed before PR

- 41 local tests passed: receipt/revision/time boundaries,169/168 bars, missingBTC/ETH, corrupt source/raw/manifest, strict suppression, deadline crossing, duplicates, exact endpoint settlement/revisions, append-only SQL, restore/snapshot regression, shell failure phases, core artifact survival, stale report replacement and workflow invariants.
- Eight fault scenarios on copies of verified artifact10578748686 from run35426198641. Clean input→not_needed/0; missingBTC→eligible/6; ETH gap200h ago→eligible/6; recentETH gap, latestETH missing, lateETH receipt, deadline→blocked/0; corruptETH hash→error/0. These are simulations, not genuine forecasts.
- Original artifact SHA256: 41eaaf798bf82617ec148eeb75869740ab2bef20c1ad9d24146efb1d8bb202a0. Protected observations.db/issued.db/signals.json hashes stayed unchanged in originals and within each treated copy after running the shadow. No simulated ledger is committed or uploaded as live state.
- Existing workflow bytes were reconstructed and verified against original Git blob ecebe1357693a8543a260aedc998ab8c7be19690 before narrowly adding optional steps.

## Merge and post-merge gates

Repository-wide CI must pass. A post-merge main run must separately demonstrate core publication outcome, current-run shadow decision, accepted manifest/audit persistence and final snapshot retention. Clean data can correctly yield zero fallback records. This is not a reason to inject a gap into live data.

Public partial-output/UI/delivery validation and actual future shortage episodes remain separate before public fallback activation. No public activation is claimed here. EV1-B/C and the earlier median-error correction remain unpromoted/rejected. Subsequent predictive research must test a new preregistered information/regime hypothesis, not retune rejected variants on inspected history.
