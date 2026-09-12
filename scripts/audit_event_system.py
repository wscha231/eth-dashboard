"""Read-only operational and six-horizon evidence report; never promote models."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.verify_event_site import verify


def audit(payload, now=None):
    now = now or datetime.now(timezone.utc)
    issues = []
    try:
        verify(payload, now=now, require_ready=True)
    except (ValueError, KeyError, TypeError) as exc:
        issues.append(str(exc))
    source = payload.get("source") or {}
    if not source.get("ready") or source.get("errors"):
        issues.append("active hourly source is not ready")
    for product in ("ETH-USD", "BTC-USD"):
        if source.get("latest", {}).get(product) != payload.get("expected_slot"):
            issues.append(product + " source cutoff mismatch")
    if payload.get("shadow_errors"):
        issues.append("experimental candidate errors")
    rows = []
    for h in (6, 24, 72, 168, 336, 720):
        records = [r for r in payload.get("current", []) if r.get("horizon_seconds") == h * 3600]
        record = records[0] if len(records) == 1 else {}
        evidence = payload.get("prospective", {}).get(str(h), {})
        metrics = evidence.get("metrics", {})
        rows.append({"horizon_hours": h, "available": bool(record),
                     "model": record.get("selected_model"),
                     "training_target_end": record.get("training_target_end"),
                     "settled": evidence.get("resolved", 0),
                     "nonoverlap": evidence.get("nonoverlap_resolved", 0),
                     "return_mae_pp": metrics.get("return_mae_pp"),
                     "no_change_mae_pp": 100 * metrics["no_change_mae"] if metrics.get("no_change_mae") is not None else None,
                     "coverage80": metrics.get("coverage80"),
                     "balanced_accuracy": metrics.get("terminal_balanced_accuracy"),
                     "assessment": "pending" if not evidence.get("resolved") else
                         "no_change_not_beaten" if metrics.get("mae_skill", 0) <= 0 else "positive_point_estimate_only"})
    return {"generated_at": now.isoformat(), "release_id": payload.get("release_id"),
            "slot": payload.get("expected_slot"), "operation": "attention" if issues else "ready",
            "issues": issues, "horizons": rows,
            "active_inputs": ["Coinbase ETH-USD closed hourly OHLCV", "Coinbase BTC-USD closed hourly OHLCV"],
            "auxiliary_inputs": "Daily macro, on-chain and flow collectors are separate; not used by this hourly model.",
            "storage": "Original ledgers and archived input vintages/checkpoints: data/event-ledger. Rolling snapshots: Actions artifacts, subject to expiry.",
            "interpretation": "Availability is not predictive skill. Overlapping results are not independent evidence; no automatic promotion."}


def markdown(report):
    def fmt(value):
        return "pending" if value is None else f"{value:.4f}"
    lines = [f"ETH system: {report['operation']} | slot {report['slot']}", "",
             "| Hours | Available | Settled / nonoverlap | MAE pp / no-change pp | Assessment |",
             "|---|---|---|---|---|"]
    for r in report["horizons"]:
        lines.append(f"| {r['horizon_hours']} | {r['available']} | {r['settled']} / {r['nonoverlap']} | {fmt(r['return_mae_pp'])} / {fmt(r['no_change_mae_pp'])} | {r['assessment']} |")
    return "\n".join(lines + ["", report["interpretation"], report["auxiliary_inputs"], "Issues: " + ("; ".join(report["issues"]) or "none")]) + "\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("lake/signals"))
    parser.add_argument("--as-of")
    args = parser.parse_args()
    now = datetime.fromisoformat(args.as_of) if args.as_of else None
    result = audit(json.loads((args.root / "signals.json").read_text()), now)
    (args.root / "system_audit.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    (args.root / "system_audit.md").write_text(markdown(result))
    print(markdown(result))
