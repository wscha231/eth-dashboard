"""One-shot driver: persist a longrun OOF JSON to the DB, re-export all site
JSON blobs, and (optionally) compute an A/B diff against a prior longrun.

Why this exists
---------------
Longrun OOFs (36×30 folds, ~12h runtime) finish whenever they finish. The
daily cron (``run_daily.py``) only runs forecast + persist + export for the
*live* forecast pipeline — it does not know to pick up a newly-written
longrun JSON. Without this driver, every longrun freeze requires three
manual steps:

    python -m forecast_site.persist_backtest --freeze-json <path>
    python -m forecast_site.export_backtest_json
    python tests/phase0/diff_longrun_oof.py --old <old> --new <new> ...

This script collapses that into one invocation, and writes the diff JSON
to ``forecast_site/public/backtest_diff_latest.json`` so the frontend can
load it without per-pair filename knowledge.

Usage
-----
Common case — new phase5_nodefi vs existing phase6_production longrun::

    python -m forecast_site.persist_and_export_longrun \\
        --json tests/phase0/phase5_nodefi_longrun_oof_metrics.json \\
        --diff-against tests/phase0/phase6_production_longrun_oof_metrics.json

Persist-and-export only (no diff)::

    python -m forecast_site.persist_and_export_longrun \\
        --json tests/phase0/phase5_nodefi_longrun_oof_metrics.json

Idempotent: persist_backtest uses (model_phase, sha1) as its unique key, so
re-running against the same JSON is a no-op.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from forecast_site import persist_backtest, export_backtest_json  # noqa: E402
from forecast_site.db import DEFAULT_DB_PATH  # noqa: E402

# diff_longrun_oof lives under tests/phase0/ — not a package, so import via
# a path insert. Keep the insert scoped to this function so we don't pollute
# PROJECT_ROOT sys.path with test-suite internals.
def _load_diff_module():
    diff_dir = PROJECT_ROOT / "tests" / "phase0"
    sys.path.insert(0, str(diff_dir))
    try:
        import diff_longrun_oof  # type: ignore
        return diff_longrun_oof
    finally:
        # Leave the path — removing it breaks subsequent imports if the
        # module was already loaded elsewhere. sys.path churn is cheap.
        pass


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "forecast_site" / "public"
DIFF_LATEST_FILE = "backtest_diff_latest.json"


def run_persist(json_path: Path, db_path: Path) -> None:
    """Delegate to persist_backtest.main — handles longrun transaction +
    per-date prediction inserts. Idempotent via (model_phase, sha1) index."""
    if not json_path.exists():
        raise SystemExit(f"Longrun JSON not found: {json_path}")
    print(f"[driver] persisting {json_path} -> {db_path}")
    persist_backtest.main([
        "--freeze-json", str(json_path),
        "--db", str(db_path),
    ])


def run_export(db_path: Path, output_dir: Path) -> None:
    """Delegate to export_backtest_json.main — regenerates versions/backtest/
    per-phase prediction JSONs and the compact longrun_history chart file.

    export_backtest_json.main() parses sys.argv directly (no argv param), so
    we temporarily swap sys.argv to drive it the way the CLI does.
    """
    print(f"[driver] re-exporting site JSONs -> {output_dir}")
    saved = sys.argv
    try:
        sys.argv = [
            "export_backtest_json",
            "--db", str(db_path),
            "--output-dir", str(output_dir),
        ]
        export_backtest_json.main()
    finally:
        sys.argv = saved


def run_diff(old_path: Path, new_path: Path, output_dir: Path) -> dict[str, Any]:
    """Compute A/B diff via diff_longrun_oof.compare_runs and write the
    result to public/backtest_diff_latest.json for the frontend to load.

    Also writes a name-stamped copy so history isn't overwritten when the
    next diff runs.
    """
    if not old_path.exists():
        raise SystemExit(f"Old longrun JSON not found: {old_path}")
    if not new_path.exists():
        raise SystemExit(f"New longrun JSON not found: {new_path}")

    diff_mod = _load_diff_module()
    old_payload = json.loads(old_path.read_text(encoding="utf-8"))
    new_payload = json.loads(new_path.read_text(encoding="utf-8"))

    print(f"[driver] computing diff: old={old_path.name}  new={new_path.name}")
    result = diff_mod.compare_runs(old_payload, new_payload)
    result["meta"] = {
        "old_file": str(old_path),
        "new_file": str(new_path),
        "old_phase": old_payload.get("mode", "").replace("longrun_oof_", ""),
        "new_phase": new_payload.get("mode", "").replace("longrun_oof_", ""),
        "old_mode":  old_payload.get("mode"),
        "new_mode":  new_payload.get("mode"),
        "old_frozen": old_payload.get("frozen_at") or old_payload.get("last_update"),
        "new_frozen": new_payload.get("frozen_at") or new_payload.get("last_update"),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    latest_path = output_dir / DIFF_LATEST_FILE
    latest_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[driver] wrote {latest_path}  ({latest_path.stat().st_size:,} bytes)")

    # Name-stamped archival copy so we can track evolution across phases.
    old_label = (old_payload.get("mode") or "old").replace("longrun_oof_", "").replace("_36x30", "")
    new_label = (new_payload.get("mode") or "new").replace("longrun_oof_", "").replace("_36x30", "")
    archive_path = output_dir / f"backtest_diff_{new_label}_vs_{old_label}.json"
    archive_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[driver] wrote {archive_path}  ({archive_path.stat().st_size:,} bytes)")
    return result


def _fmt_delta(old: float | None, new: float | None, precision: int = 4) -> tuple[str, str]:
    """Return (marker, human-readable delta-string) for old→new."""
    if old is None or new is None:
        return ("·", "—")
    delta = new - old
    if abs(delta) < 10 ** (-precision):
        return ("=", f"±0")
    marker = "↓" if delta < 0 else "↑"
    return (marker, f"{delta:+.{precision}f}")


def print_diff_summary(result: dict[str, Any]) -> None:
    """Terse stdout summary — enough for a CI log + quick Go/No-Go eyeball.

    Field names match compare_runs() output: regression rows carry
    rmse_close_{old,new} and mape_close_{old,new}; classification rows
    carry brier_{old,new} and accuracy_{old,new}. Deltas are computed on
    the fly (the compare_runs "paired_*_mean_diff" is a different
    paired-error statistic, not a headline delta).
    """
    coverage = result.get("coverage", {})
    per_model = result.get("per_model", [])
    meta = result.get("meta", {})
    print()
    print(f"=== A/B summary: new={meta.get('new_phase', '?')}  vs  old={meta.get('old_phase', '?')} ===")
    print(f"   coverage: common={coverage.get('common_keys', '?')} "
          f"old_only={coverage.get('old_only_keys', '?')} "
          f"new_only={coverage.get('new_only_keys', '?')}")

    reg_rows = [r for r in per_model if r.get("head") == "regression"]
    cls_rows = [r for r in per_model if r.get("head") == "classification"]

    if reg_rows:
        print("   Regression Δ price RMSE (lower = better):")
        for r in reg_rows[:8]:
            old = r.get("rmse_close_old")
            new = r.get("rmse_close_new")
            marker, delta = _fmt_delta(old, new, precision=2)
            old_s = f"${old:>7.2f}" if old is not None else "—"
            new_s = f"${new:>7.2f}" if new is not None else "—"
            print(f"     h={r['horizon']:>2} {r['model']:<40} {marker}  {old_s}→{new_s}  Δ={delta}")

    if cls_rows:
        print("   Classification Δ Brier (lower = better):")
        for r in cls_rows[:8]:
            old = r.get("brier_old")
            new = r.get("brier_new")
            marker, delta = _fmt_delta(old, new, precision=4)
            old_s = f"{old:>.4f}" if old is not None else "—"
            new_s = f"{new:>.4f}" if new is not None else "—"
            p = r.get("wilcoxon_p_value")
            p_str = "" if p is None else f"  p={p:.3g}"
            print(f"     h={r['horizon']:>2} {r['model']:<40} {marker}  {old_s}→{new_s}  Δ={delta}{p_str}")
    print()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", type=Path,
        help="Longrun JSON to persist to the DB. Skip if you only want "
             "to re-export or diff.",
    )
    parser.add_argument(
        "--diff-against", type=Path,
        help="Path to an older longrun JSON to diff --json against. "
             "Requires --json.",
    )
    parser.add_argument("--db",         type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--skip-persist", action="store_true",
                        help="Skip the persist step (useful when --json "
                             "was already persisted previously).")
    parser.add_argument("--skip-export",  action="store_true",
                        help="Skip the export step (debugging).")
    args = parser.parse_args(argv)

    if args.diff_against and not args.json:
        raise SystemExit("--diff-against requires --json (the new side of the diff).")

    # 1. Persist (optional)
    if args.json and not args.skip_persist:
        run_persist(args.json, args.db)

    # 2. Export (unconditional unless --skip-export)
    if not args.skip_export:
        run_export(args.db, args.output_dir)

    # 3. Diff (optional)
    if args.diff_against:
        result = run_diff(args.diff_against, args.json, args.output_dir)
        print_diff_summary(result)

    print("[driver] done.")


if __name__ == "__main__":
    main()
