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

Local implementation complete. GitHub checks, merge and production verification are recorded in PR #29 and the subsequent deployment record.
This file does not assert that those later steps have already completed.

GitHub scheduled execution can still be delayed. The change preserves useful, correctly dated forecasts and accurate outage status; it does not create an always-on scheduler or guarantee uninterrupted fresh forecasts.
