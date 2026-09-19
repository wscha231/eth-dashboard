"""Read-only hourly issuance/delivery/settlement audit. Never backfill or promote."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

HORIZONS = (6, 24, 72, 168, 336, 720)
HOUR = timedelta(hours=1)
SETTLEMENT_GRACE = timedelta(hours=2)


def stamp(value):
    if not isinstance(value, (str, datetime)):
        raise ValueError("timezone-aware timestamp required")
    result = datetime.fromisoformat(value.replace("Z", "+00:00")) if isinstance(value, str) else value
    if result.tzinfo is None:
        raise ValueError("timezone-aware timestamp required")
    return result.astimezone(timezone.utc)


def aligned(value):
    result = stamp(value)
    if result != result.replace(minute=0, second=0, microsecond=0):
        raise ValueError("audit slot must align to an hour")
    return result


def due_slot(now):
    now = stamp(now)
    slot = now.replace(minute=0, second=0, microsecond=0)
    return slot - HOUR if now.minute < 15 else slot


def intervals(slots):
    groups = []
    for slot in sorted(slots):
        if groups and slot == groups[-1][1] + HOUR:
            groups[-1][1] = slot
            groups[-1][2] += 1
        else:
            groups.append([slot, slot, 1])
    return [{"first": a.isoformat(), "last": b.isoformat(), "slots": n} for a, b, n in groups]


def _load_rows(path):
    # Opening in mode=ro cannot silently create an empty replacement ledger.
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        names = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"forecasts", "outcomes", "deliveries"} <= names:
            raise ValueError("not an hourly event issuance ledger")
        return {name: [dict(r) for r in db.execute("SELECT * FROM " + name)]
                for name in ("forecasts", "outcomes", "deliveries")}


def audit(root, *, as_of=None, from_slot=None, through_slot=None):
    root = Path(root)
    now = stamp(as_of or datetime.now(timezone.utc))
    path = root / "issued.db"
    if not path.is_file():
        raise FileNotFoundError("hourly issued.db is missing; no empty ledger was created")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    tables = _load_rows(path)
    errors, records, receipts, outcomes = [], [], {}, {}
    for row in tables["deliveries"]:
        try:
            observed = stamp(row["verified_at"])
            if observed <= now and row.get("release_id"):
                receipts.setdefault(row["forecast_id"], []).append(observed)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("invalid delivery receipt: " + str(exc))
    for row in tables["outcomes"]:
        try:
            evaluated = stamp(row["evaluated_at"])
            if evaluated <= now:
                outcomes.setdefault(row["forecast_id"], []).append((evaluated, row))
        except (KeyError, TypeError, ValueError) as exc:
            errors.append("invalid outcome timestamp: " + str(exc))
    for row in tables["forecasts"]:
        try:
            slot, issued = aligned(row["slot"]), stamp(row["issued_at"])
            if slot > now or issued > now:
                continue  # An as-of audit cannot import later forecasts.
            h = row["horizon_seconds"] / 3600
            if h not in HORIZONS:
                raise ValueError("unsupported hourly horizon")
            data = json.loads(row["payload"])
            start, end = stamp(data["window_start"]), stamp(row["target_end"])
            ready = stamp(data["available_at"])
            valid = (data["forecast_id"] == row["forecast_id"] and
                     aligned(data["slot"]) == slot == stamp(data["input_cutoff"]) and
                     stamp(data["issued_at"]) == issued and stamp(data["target_end"]) == end and
                     data["horizon_seconds"] == row["horizon_seconds"] and
                     slot <= ready <= issued < slot + timedelta(minutes=55) and
                     start == slot + HOUR and end == start + h * HOUR)
            if not valid:
                errors.append("invalid issuance timing/identity: " + row["forecast_id"])
            deliveries = sorted(t for t in receipts.get(row["forecast_id"], []) if t >= issued)
            timely = valid and any(t < start for t in deliveries)
            settled = False
            for evaluated, outcome in sorted(outcomes.get(row["forecast_id"], []), reverse=True,
                                              key=lambda item: item[0]):
                truth = json.loads(outcome["payload"])
                truth_at = stamp(truth["truth_available_at"])
                if end <= truth_at <= evaluated:
                    settled = True
                    break
                errors.append("outcome before maturity/receipt: " + row["forecast_id"])
            median_return = None
            q, ref = data.get("price_quantiles"), data.get("reference_price")
            if isinstance(q, list) and len(q) == 3 and isinstance(ref, (int, float)) and ref > 0:
                median_return = round(float(q[1]) / ref - 1, 12)
            records.append({"id": row["forecast_id"], "slot": slot, "h": int(h),
                            "issued": issued, "start": start, "end": end, "valid": valid,
                            "timely": timely, "delivered": bool(deliveries), "settled": settled,
                            "model": data.get("selected_model", "unknown"),
                            "model_version": row.get("model_version"),
                            "target_definition": row.get("target_definition"),
                            "median_return": median_return})
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            errors.append("invalid forecast/outcome row: " + str(row.get("forecast_id")) + ": " + str(exc))
    if from_slot is None and not records:
        raise ValueError("no observable forecasts; supply an explicit --from-slot to audit a launch window")
    first = aligned(from_slot) if from_slot is not None else min(r["slot"] for r in records)
    last = aligned(through_slot) if through_slot is not None else due_slot(now)
    if last > due_slot(now):
        raise ValueError("through-slot is not due yet")
    if first > last or (last - first) / HOUR > 2_000_000:
        raise ValueError("invalid or excessive audit window")
    expected = {first + i * HOUR for i in range(int((last - first) / HOUR) + 1)}
    result = []
    for h in HORIZONS:
        rows = [r for r in records if r["h"] == h and r["slot"] in expected]
        issued_slots = {r["slot"] for r in rows}
        valid_slots = {r["slot"] for r in rows if r["valid"]}
        timely_slots = {r["slot"] for r in rows if r["timely"]}
        missing = expected - issued_slots
        recent = {s for s in expected if s > last - 24 * HOUR}
        medians = {r["median_return"] for r in rows if r["median_return"] is not None}
        result.append({
            "horizon_hours": h, "expected_slots": len(expected), "stored_records": len(rows),
            "issued_slots": len(issued_slots), "timing_valid_slots": len(valid_slots),
            "missing_issue_slots": len(missing), "multiple_version_slots":
                sum(n > 1 for n in Counter(r["slot"] for r in rows).values()),
            "published_before_window_slots": len(timely_slots),
            "timely_service_coverage": len(timely_slots) / len(expected),
            "late_or_unverified_closed_slots": sum(s not in timely_slots and s + HOUR <= now for s in issued_slots),
            "delivery_still_pending_slots": sum(s not in timely_slots and s + HOUR > now for s in issued_slots),
            "matured_records": sum(r["end"] <= now for r in rows),
            "matured_unsettled_records": sum(r["end"] <= now and not r["settled"] for r in rows),
            "overdue_settlement_records": sum(r["end"] + SETTLEMENT_GRACE <= now and not r["settled"] for r in rows),
            "missing_issue_intervals": intervals(missing),
            "recent_24h": {"expected_slots": len(recent), "issued_slots": len(recent & issued_slots),
                           "published_before_window_slots": len(recent & timely_slots)},
            "selected_models": dict(Counter(r["model"] for r in rows)),
            "model_versions": sorted({str(r["model_version"]) for r in rows}),
            "target_definitions": sorted({str(r["target_definition"]) for r in rows}),
            "distinct_median_returns": len(medians),
            "constant_median_return": next(iter(medians)) if len(medians) == 1 and len(rows) > 1 else None,
        })
    replay_path = root / "replay.json"
    historical = {"status": "missing"}
    if replay_path.is_file():
        replay = json.loads(replay_path.read_text())
        historical = {"status": "available", "evidence_kind": "retrospective_reconstruction_not_live_issuance",
                      "generated_at": replay.get("generated_at"), "data_as_of": replay.get("data_as_of"),
                      "sha256": hashlib.sha256(replay_path.read_bytes()).hexdigest(),
                      "horizons": {h: {k: v.get(k) for k in
                                   ("status", "common_origins", "first_origin", "last_origin", "eligible_calendar_coverage")}
                                   for h, v in replay.get("horizons", {}).items()}}
    after = hashlib.sha256(path.read_bytes()).hexdigest()
    if after != before:
        raise RuntimeError("ledger changed during the audit; result is not a pinned snapshot")
    needs_attention = errors or any(r["missing_issue_slots"] or r["late_or_unverified_closed_slots"] or
                                    r["overdue_settlement_records"] for r in result)
    return {"schema_version": 1, "as_of": now.isoformat(), "status": "attention" if needs_attention else "ready",
            "ledger_sha256": before, "read_only": True, "from_slot": first.isoformat(),
            "through_slot": last.isoformat(), "cadence": "hourly event ledger only; not the 00UTC variance cohort",
            "window_basis": "explicit" if from_slot is not None else "first_observed_record_left_censored",
            "settlement_grace_hours": 2, "horizons": result, "historical": historical,
            "errors": errors, "promotion": "not_evaluated_no_model_changes",
            "limitations": ["Missing slots are never reconstructed as prospective records.",
                            "No inference about records before the explicit/first-observed audit window.",
                            "Counts pooled across versions measure service coverage, not model skill.",
                            "Constant climatology returns may be legitimate baselines, not learned directional alpha."]}


def markdown(report):
    lines = ["# Forecast continuity audit", "", f"As of: {report['as_of']}",
             f"Window: {report['from_slot']} through {report['through_slot']}",
             f"Window basis: {report['window_basis']}", "", "| Hours | Expected | Issued slots | Timely public | Missing | Overdue settlement |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in report["horizons"]:
        lines.append("| {horizon_hours} | {expected_slots} | {issued_slots} | {published_before_window_slots} | {missing_issue_slots} | {overdue_settlement_records} |".format(**row))
    lines.extend(["", "Historical replay is separate from actual prospective issuance.",
                  "Service coverage is not predictive skill. No record or model was changed.",
                  "", "Errors: " + ("; ".join(report["errors"]) or "none")])
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    parser.add_argument("--as-of")
    parser.add_argument("--from-slot")
    parser.add_argument("--through-slot")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    report = audit(args.root, as_of=args.as_of, from_slot=args.from_slot, through_slot=args.through_slot)
    (args.root / "continuity_audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.root / "continuity_audit.md").write_text(markdown(report))
    print(markdown(report))
    if args.strict and report["status"] != "ready":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
