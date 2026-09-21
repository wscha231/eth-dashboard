import pytest

from scripts.run_variance_shadow import issuance_audit, require_current_issue


def result(at, *, issued=False, errors=False):
    return {
        "generated_at": at,
        "issued_this_run": ([{"forecast_id": "x"}] if issued else []),
        "errors": ([{"reason": "source stale"}] if errors else []),
    }


def test_outside_midnight_window_is_expected_not_a_backfill_candidate():
    audit = issuance_audit(result("2026-09-17T13:30:43+00:00"))
    assert audit["current_window_eligible"] is False
    assert audit["current_window_status"] == "outside_frozen_00utc_origin"
    assert audit["next_eligible_origin"] == "2026-09-18T00:00:00+00:00"
    assert audit["late_or_reconstructed_issue_allowed"] is False
    assert audit["retrospective_origin_match_required"] is True


def test_midnight_window_marks_successful_issue():
    audit = issuance_audit(result("2026-09-18T00:17:00+00:00", issued=True))
    assert audit["current_window_eligible"] is True
    assert audit["current_window_status"] == "current_frozen_origin_issued_or_idempotently_restored"
    assert audit["next_eligible_origin"] == "2026-09-18T00:00:00+00:00"


def test_midnight_window_separates_source_not_ready_from_model_error():
    ready = issuance_audit(result("2026-09-18T00:17:00+00:00"))
    failed = issuance_audit(result("2026-09-18T00:17:00+00:00", errors=True))
    assert ready["current_window_status"] == "current_frozen_origin_source_not_ready"
    assert failed["current_window_status"] == "current_frozen_origin_error"


def test_deadline_passed_never_allows_late_issue():
    audit = issuance_audit(result("2026-09-18T00:56:00+00:00"))
    assert audit["current_window_status"] == "current_frozen_origin_deadline_passed"
    assert audit["next_eligible_origin"] == "2026-09-19T00:00:00+00:00"
    assert audit["late_or_reconstructed_issue_allowed"] is False


def test_strict_current_issue_requires_all_six_horizons():
    report = {
        "generated_at": "2026-09-18T00:17:00+00:00",
        "issued_this_run": [{"horizon_hours": h} for h in (6,24,72,168,336,720)],
        "errors": [],
    }
    audit = issuance_audit(report)
    require_current_issue(report, audit)


def test_strict_current_issue_rejects_partial_or_error():
    base = {
        "generated_at": "2026-09-18T00:17:00+00:00",
        "issued_this_run": [{"horizon_hours": h} for h in (6,24,72,168,336,720)],
        "errors": [],
    }
    partial = dict(base, issued_this_run=base["issued_this_run"][:-1])
    with pytest.raises(ValueError, match="incomplete"):
        require_current_issue(partial, issuance_audit(partial))
    failed = dict(base, errors=[{"horizon_hours":72,"reason":"stale checkpoint"}])
    with pytest.raises(ValueError, match="incomplete"):
        require_current_issue(failed, issuance_audit(failed))


def test_strict_current_issue_rejects_outside_frozen_window():
    report = {
        "generated_at": "2026-09-18T01:17:00+00:00",
        "issued_this_run": [{"horizon_hours": h} for h in (6,24,72,168,336,720)],
        "errors": [],
    }
    with pytest.raises(ValueError, match="not eligible"):
        require_current_issue(report, issuance_audit(report))
