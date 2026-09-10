# ETH outlook availability implementation

User approval: “어 재발 방지 수정”. Implements the reviewed plan in this directory.

## Changes

- Current and previous cards validate IDs, finite prices/probabilities, UTC input/issue/window timestamps, and unexpired target windows.
- When a current horizon is missing, use its most recent timely, verified publication from recent_issued. Keep all original forecast values and dates. Show Previous published forecast and Update delayed; never promote this record into current.
- Check expiry every 30 seconds, including during connection failure, without repainting an unchanged card on every status timer.
- Exact payload publication compares full JSON as well as the release and ordered forecast IDs. Current records are checked for numeric validity and the complete time contract.
- Publication receipts and the durable ledger are preserved before a final, mandatory --require-ready check. A delivered delayed/partial payload no longer makes the hourly job successful.
- Payload verification logs say Event payload verified. Event site verified is reserved for successful strict readiness checks.
- Recovery dispatch requires five minutes before the existing :55 issuance cutoff. Time is read again after GitHub API calls and immediately before dispatch.
- Production inference reads actual wall time for each issue rather than reusing the function start time. Existing records remain idempotent; a retry does not change an earlier forecast.
- The model training hash and frozen protocol are unchanged. Hourly inference still cannot fit models.

## Local validation

59 focused tests passed in tests/test_event_frontend.py, tests/test_event_recovery.py, and tests/test_event_pipeline.py.
Tests cover first load/reload, all six horizons, empty and partial releases, timely publication eligibility, invalid data, expiry during disconnection, later recovery, and original-value preservation.
They also execute the actual publisher shell flow with isolated external-command stand-ins to verify receipts and ledger persistence occur before readiness failure.
The engine test uses the real issuance ledger with mocked input/model computation, advancing the clock past :55 and across the hour; late new records are rejected without backdating.

The real incident snapshot additionally reproduced six visible previous cards while readiness remained delayed. The real recovered snapshot passed the strengthened strict validator.
JavaScript, shell and Python syntax checks and git diff --check passed.

## Delivery status

Implementation, merge and production verification completed. [PR #29](https://github.com/wscha231/eth-dashboard/pull/29) was merged as fefe57926c5a759e745db0bb9638dcc71d16f8e1.

- The exact PR head passed [269 full pipeline tests](https://github.com/wscha231/eth-dashboard/actions/runs/34538846018) and [hourly event research replay](https://github.com/wscha231/eth-dashboard/actions/runs/34538845967). The 59 focused local tests are a subset, not an additional 59 independent CI tests.
- The [production hourly run](https://github.com/wscha231/eth-dashboard/actions/runs/34539536935) succeeded. At 2026-09-10 22:54:12 UTC (2026-09-11 07:54:12 KST), the runner fetched the public site and checked exact JSON, HTML and JavaScript equality, then passed the final strict readiness check: ready, slot 2026-09-10 22:00 UTC, horizons=6.
- Published data commit: 49bfc39430b4546ec60c6960d94d90f11812dd1d. Release ID: 435690372b07fd7d49887176f8779f3923bdec3d651e3027e9f5c564f6777485.
- Served JavaScript SHA256: 10bb652398d887d62b76485df7ee8c51832f7544d4c516165967788a4b517055. Both external interface checks matched the implementation source.
- The runner preserved publication receipts in ledger commit 5286812f7ccbaeadc20f0e2e863982b800b0da4f before the final readiness check. All six current records match the corresponding previously published ledger records in recent_issued, including original prices and issuance dates. This deployment correctly reused the existing 22:00 issuance; it did not rewrite it as a new forecast.
- At 22:55:38 UTC, the deployed JavaScript and exact published JSON passed all six horizon selections in the Node DOM harness. A local copy with current=[] displayed six verified previous cards and remained delayed; the strict Python validator rejected the same empty release. No outage was injected into production.

Machine-readable checks, receipt comparisons and links are in deployment.json. The public-site check was executed by the GitHub runner; the local six-selector check used the exact deployed code and payload, not an interactive browser screenshot.

GitHub scheduled execution can still be delayed. The change preserves useful, correctly dated forecasts and accurate outage status; it does not create an always-on scheduler or guarantee uninterrupted fresh forecasts.
