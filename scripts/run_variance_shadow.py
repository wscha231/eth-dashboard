#!/usr/bin/env python3
"""Issue and settle the isolated HAR-RV prospective variance shadow."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from signal_pipeline import variance_shadow
from signal_pipeline.variance_shadow_refit import fit_checkpoint


def _utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("variance-shadow audit timestamp must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def issuance_audit(result: dict) -> dict:
    """Explain prospective issuance without changing the frozen model policy."""
    now = _utc(result["generated_at"])
    origin = now.replace(hour=0, minute=0, second=0, microsecond=0)
    deadline = origin + timedelta(minutes=55)
    in_window = now.hour == 0 and now < deadline
    if in_window:
        next_origin = origin
        if result.get("issued_this_run"):
            status = "current_frozen_origin_issued_or_idempotently_restored"
        elif result.get("errors"):
            status = "current_frozen_origin_error"
        else:
            status = "current_frozen_origin_source_not_ready"
    else:
        next_origin = origin + timedelta(days=1)
        status = "outside_frozen_00utc_origin" if now.hour != 0 else "current_frozen_origin_deadline_passed"
    return {
        "frozen_origin_clock": "00:00 UTC daily",
        "issuance_deadline_minutes_after_origin": 55,
        "current_window_status": status,
        "current_window_eligible": in_window,
        "next_eligible_origin": next_origin.isoformat(),
        "retrospective_origin_match_required": True,
        "late_or_reconstructed_issue_allowed": False,
        "note": "Zero issued rows outside the frozen origin window are expected and are not backfilled prospectively.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    args = parser.parse_args()
    # Operational refits may use a source snapshot at most one completed hour behind
    # wall time. The issuance path itself remains strict and still requires current-slot data.
    variance_shadow.fit_checkpoint = fit_checkpoint
    result = variance_shadow.run(args.root)

    # The frozen HAR-RV policy intentionally uses the same 00:00 UTC origin as the
    # retrospective audit. Surface that operational fact explicitly so a healthy
    # out-of-window run cannot be mistaken for a failed prospective collector.
    audit = issuance_audit(result)
    state = args.root / "variance_shadow"
    report = json.loads((state / "report.json").read_text(encoding="utf-8"))
    report["issuance_audit"] = audit
    variance_shadow.atomic_json(state / "report.json", report)
    public = json.loads((state / "public.json").read_text(encoding="utf-8"))
    public["issuance_audit"] = audit
    variance_shadow.atomic_json(state / "public.json", public)

    print(json.dumps({
        "generated_at": report["generated_at"],
        "source_as_of": report["source_as_of"],
        "issued_this_run": report["issued_this_run"],
        "settled_this_run": report["settled_this_run"],
        "errors": report["errors"],
        "issuance_audit": report["issuance_audit"],
        "prospective": report["prospective"],
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
