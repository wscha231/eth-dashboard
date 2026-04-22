"""Backtest the LLM analyst against realised ETH returns.

What this does
--------------
Walks forward through a date range, asking the analyst for a view on each
sampled date (weekly by default to keep API cost bounded), and compares
the view's direction + range to the actual 30-day forward return from the
master CSV. Output: per-call rows + a summary report with:

  - Directional accuracy (vs 50% coin flip)
  - Range coverage (fraction of actuals inside [lo, hi])
  - Signed error of range midpoint (bias vs naive=0)
  - Skill score of midpoint vs naive (predict 0% return)
  - Calibration: does confidence track accuracy?

Constraints for honest evaluation
---------------------------------
- **Date gating**: every call sees snapshot + headlines as of the call
  date ONLY. The ``snapshot.build_snapshot`` and
  ``news_store.fetch_headlines_before`` functions enforce this.
- **Forward-only scoring**: a call at date ``T`` is scored against close
  on ``T + horizon`` looked up from the master CSV. If that future date
  isn't yet in the CSV, the row is marked pending (not scored).
- **Hard cost cap**: ``--max-calls`` stops early if hit.

Typical usage
-------------
    # Dry-run: assemble all prompts, don't call the API
    python -m llm_analyst.backtest \\
        --start 2026-01-01 --end 2026-04-01 \\
        --sample-every 7 --dry-run

    # Real backtest (weekly, capped at 20 calls ~ $1)
    python -m llm_analyst.backtest \\
        --start 2026-01-01 --end 2026-04-01 \\
        --sample-every 7 --max-calls 20
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from llm_analyst import analyst, news_store, snapshot


# -----------------------------------------------------------------------------
# Scoring
# -----------------------------------------------------------------------------


def _iter_call_dates(start: date, end: date, sample_every_days: int) -> list[date]:
    current = start
    out: list[date] = []
    while current <= end:
        out.append(current)
        current = current + timedelta(days=sample_every_days)
    return out


def _lookup_close(frame: pd.DataFrame, target: date) -> float | None:
    """Return eth_close at ``target`` (exact date), or None if absent."""
    match = frame[frame["Date"] == pd.Timestamp(target)]
    if match.empty:
        return None
    value = match["eth_close"].iloc[0]
    return float(value) if pd.notna(value) else None


def _nearest_trading_close(frame: pd.DataFrame, target: date, max_backfill_days: int = 4) -> tuple[float | None, date | None]:
    """Walk backward up to ``max_backfill_days`` looking for a close on a non-holiday row."""
    for offset in range(0, max_backfill_days + 1):
        probe = target - timedelta(days=offset)
        value = _lookup_close(frame, probe)
        if value is not None:
            return value, probe
    return None, None


@dataclass
class ScoredCall:
    as_of_date: str
    horizon_days: int
    eth_close_as_of: float | None
    eth_close_forward: float | None
    target_date: str | None
    actual_return: float | None
    direction_actual: str | None  # "up" / "down" / "flat" (threshold +/-2%)
    view: analyst.AnalystView | None
    direction_correct: bool | None
    in_range: bool | None
    midpoint_error: float | None  # midpoint_return - actual_return
    dry_run_prompt_chars: int | None = None
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        r = {
            "as_of_date": self.as_of_date,
            "horizon_days": self.horizon_days,
            "eth_close_as_of": self.eth_close_as_of,
            "eth_close_forward": self.eth_close_forward,
            "target_date": self.target_date,
            "actual_return": self.actual_return,
            "direction_actual": self.direction_actual,
            "direction_correct": self.direction_correct,
            "in_range": self.in_range,
            "midpoint_error": self.midpoint_error,
            "dry_run_prompt_chars": self.dry_run_prompt_chars,
            "error": self.error,
        }
        if self.view:
            r.update(
                {
                    "direction": self.view.direction,
                    "confidence": self.view.confidence,
                    "expected_return_low":  self.view.expected_return_low,
                    "expected_return_high": self.view.expected_return_high,
                    "thesis": self.view.thesis,
                }
            )
        return r


def classify_direction(return_fraction: float, flat_band: float = 0.02) -> str:
    if return_fraction > flat_band:
        return "up"
    if return_fraction < -flat_band:
        return "down"
    return "flat"


def score_call(
    view: analyst.AnalystView,
    actual_return: float | None,
) -> tuple[bool | None, bool | None, float | None, str | None]:
    """Return (direction_correct, in_range, midpoint_error, actual_direction).

    ``direction_correct``: exact match of predicted direction vs actual
    (both classified with the same ±2% flat band).
    """
    if actual_return is None:
        return (None, None, None, None)
    actual_dir = classify_direction(actual_return)
    dir_match = view.direction == actual_dir
    lo = view.expected_return_low
    hi = view.expected_return_high
    in_range = lo <= actual_return <= hi
    mid = (lo + hi) / 2.0
    midpoint_err = mid - actual_return
    return (dir_match, in_range, midpoint_err, actual_dir)


# -----------------------------------------------------------------------------
# Runner
# -----------------------------------------------------------------------------


def run_backtest(
    start: date,
    end: date,
    sample_every_days: int,
    horizon_days: int,
    *,
    master_csv: Path | str,
    news_db_path: Path | str,
    analyst_db_path: Path | str,
    model: str,
    max_calls: int,
    dry_run: bool,
) -> list[ScoredCall]:
    """Run the analyst on each sampled date and return scored calls."""
    frame = pd.read_csv(master_csv)
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)

    call_dates = _iter_call_dates(start, end, sample_every_days)
    if len(call_dates) > max_calls:
        print(f"# {len(call_dates)} sample dates exceeds --max-calls={max_calls}; truncating.")
        call_dates = call_dates[:max_calls]

    scored: list[ScoredCall] = []
    for call_date in call_dates:
        as_of_iso = call_date.isoformat()
        target_date = call_date + timedelta(days=horizon_days)

        eth_as_of, snap_close_date = _nearest_trading_close(frame, call_date)
        eth_forward, forward_close_date = _nearest_trading_close(frame, target_date)
        actual_return = None
        if eth_as_of and eth_forward:
            actual_return = (eth_forward / eth_as_of) - 1.0

        try:
            result = analyst.request_view(
                as_of_iso,
                horizon_days=horizon_days,
                master_csv=master_csv,
                news_db_path=news_db_path,
                model=model,
                dry_run=dry_run,
            )
        except analyst.AnalystError as exc:
            scored.append(
                ScoredCall(
                    as_of_date=as_of_iso,
                    horizon_days=horizon_days,
                    eth_close_as_of=eth_as_of,
                    eth_close_forward=eth_forward,
                    target_date=target_date.isoformat(),
                    actual_return=actual_return,
                    direction_actual=classify_direction(actual_return) if actual_return is not None else None,
                    view=None,
                    direction_correct=None,
                    in_range=None,
                    midpoint_error=None,
                    error=str(exc),
                )
            )
            continue

        if dry_run:
            assert isinstance(result, dict)
            prompt_chars = len(result["user_prompt"])
            scored.append(
                ScoredCall(
                    as_of_date=as_of_iso,
                    horizon_days=horizon_days,
                    eth_close_as_of=eth_as_of,
                    eth_close_forward=eth_forward,
                    target_date=target_date.isoformat(),
                    actual_return=actual_return,
                    direction_actual=classify_direction(actual_return) if actual_return is not None else None,
                    view=None,
                    direction_correct=None,
                    in_range=None,
                    midpoint_error=None,
                    dry_run_prompt_chars=prompt_chars,
                )
            )
            continue

        assert isinstance(result, analyst.AnalystView)
        # Persist every real call so the track record grows.
        analyst.persist_view(result, db_path=analyst_db_path)

        dir_match, in_range, midpoint_err, actual_dir = score_call(result, actual_return)
        scored.append(
            ScoredCall(
                as_of_date=as_of_iso,
                horizon_days=horizon_days,
                eth_close_as_of=eth_as_of,
                eth_close_forward=eth_forward,
                target_date=target_date.isoformat() if target_date else None,
                actual_return=actual_return,
                direction_actual=actual_dir,
                view=result,
                direction_correct=dir_match,
                in_range=in_range,
                midpoint_error=midpoint_err,
            )
        )
    return scored


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------


def _pct(x: float | None, plus: bool = True) -> str:
    if x is None:
        return "n/a"
    sign = "+" if plus and x >= 0 else ""
    return f"{sign}{x*100:.2f}%"


def summarise(scored: list[ScoredCall]) -> dict[str, Any]:
    completed = [s for s in scored if s.direction_correct is not None]
    errors = [s for s in scored if s.error]
    dry_calls = [s for s in scored if s.dry_run_prompt_chars is not None]
    pending = [
        s for s in scored
        if s.view is not None and s.direction_correct is None and not s.error
    ]

    if not completed:
        note = (
            "Dry-run: prompts assembled, no API calls made."
            if dry_calls and not pending
            else "No realised outcomes yet; backtest window is still forward-looking."
        )
        summary = {
            "n_requested": len(scored),
            "n_scored": 0,
            "n_errors": len(errors),
            "n_dry_run": len(dry_calls),
            "n_pending_realisation": len(pending),
            "note": note,
        }
        if dry_calls:
            prompt_sizes = [s.dry_run_prompt_chars for s in dry_calls if s.dry_run_prompt_chars]
            if prompt_sizes:
                summary["dry_run_prompt_chars_mean"] = int(
                    sum(prompt_sizes) / len(prompt_sizes)
                )
                summary["dry_run_prompt_chars_max"] = max(prompt_sizes)
        return summary

    dir_acc = sum(1 for s in completed if s.direction_correct) / len(completed)
    in_range = sum(1 for s in completed if s.in_range) / len(completed)
    midpoint_errors = [s.midpoint_error for s in completed if s.midpoint_error is not None]
    rmse = math.sqrt(sum(e * e for e in midpoint_errors) / len(midpoint_errors)) if midpoint_errors else None
    mean_err = sum(midpoint_errors) / len(midpoint_errors) if midpoint_errors else None

    # Naive "always predict 0% return" benchmark for midpoint RMSE.
    actuals = [s.actual_return for s in completed if s.actual_return is not None]
    naive_rmse = math.sqrt(sum(a * a for a in actuals) / len(actuals)) if actuals else None
    skill = None
    if rmse is not None and naive_rmse and naive_rmse > 0:
        skill = (naive_rmse - rmse) / naive_rmse * 100

    # Bucket by LLM-declared confidence: does higher conf => higher accuracy?
    by_conf: dict[str, list[ScoredCall]] = {"low (<0.4)": [], "mid (0.4-0.7)": [], "high (>=0.7)": []}
    for s in completed:
        assert s.view is not None
        c = s.view.confidence
        bucket = "low (<0.4)" if c < 0.4 else ("mid (0.4-0.7)" if c < 0.7 else "high (>=0.7)")
        by_conf[bucket].append(s)
    conf_breakdown = {
        b: {
            "n": len(items),
            "dir_acc": (sum(1 for s in items if s.direction_correct) / len(items)) if items else None,
        }
        for b, items in by_conf.items()
    }

    # By declared direction: does the model know when it's right?
    by_direction: dict[str, list[ScoredCall]] = {"up": [], "down": [], "flat": []}
    for s in completed:
        assert s.view is not None
        by_direction[s.view.direction].append(s)
    direction_breakdown = {
        d: {
            "n": len(items),
            "dir_acc": (sum(1 for s in items if s.direction_correct) / len(items)) if items else None,
            "avg_actual_return": (sum(s.actual_return for s in items) / len(items)) if items else None,
        }
        for d, items in by_direction.items()
    }

    return {
        "n_requested": len(scored),
        "n_scored": len(completed),
        "n_errors": len(errors),
        "directional_accuracy": dir_acc,
        "range_coverage": in_range,
        "midpoint_rmse_return": rmse,
        "midpoint_mean_error_return": mean_err,
        "naive_rmse_return": naive_rmse,
        "skill_vs_naive_percent": skill,
        "confidence_breakdown": conf_breakdown,
        "direction_breakdown": direction_breakdown,
    }


def print_report(scored: list[ScoredCall], report: dict[str, Any]) -> None:
    print(f"# LLM analyst backtest  n_requested={report['n_requested']}  "
          f"n_scored={report['n_scored']}  n_errors={report['n_errors']}")
    if report["n_scored"] == 0:
        print(f"# {report.get('note', '')}")
        if report.get("n_dry_run"):
            print(f"  dry_run calls: {report['n_dry_run']}  "
                  f"prompt chars mean={report.get('dry_run_prompt_chars_mean')}, "
                  f"max={report.get('dry_run_prompt_chars_max')}")
            print("  per-call preview:")
            for s in scored[:20]:
                actual = _pct(s.actual_return) if s.actual_return is not None else "pending"
                close_as_of = f"${s.eth_close_as_of:,.0f}" if s.eth_close_as_of else "n/a"
                close_fwd = f"${s.eth_close_forward:,.0f}" if s.eth_close_forward else "pending"
                chars = s.dry_run_prompt_chars or 0
                print(f"    {s.as_of_date}  close={close_as_of} -> {close_fwd}  "
                      f"actual={actual}  prompt={chars}chars")
        if report.get("n_pending_realisation"):
            print(f"  pending realisation (view made, outcome not yet known): "
                  f"{report['n_pending_realisation']}")
        return

    print(f"  directional accuracy : {report['directional_accuracy']*100:.1f}% (coin flip = 50%)")
    print(f"  range coverage        : {report['range_coverage']*100:.1f}% (target ~60-70%)")
    print(f"  midpoint RMSE (ret)   : {_pct(report['midpoint_rmse_return'], plus=False)}")
    print(f"  midpoint mean error   : {_pct(report['midpoint_mean_error_return'])} (0 = unbiased)")
    print(f"  naive RMSE (pred=0)   : {_pct(report['naive_rmse_return'], plus=False)}")
    if report["skill_vs_naive_percent"] is not None:
        print(f"  skill vs naive        : {report['skill_vs_naive_percent']:+.1f}%")

    print("  By declared confidence:")
    for bucket, stats in report["confidence_breakdown"].items():
        acc = f"{stats['dir_acc']*100:.1f}%" if stats["dir_acc"] is not None else "n/a"
        print(f"    {bucket:<14s}  n={stats['n']:>3d}  dir_acc={acc}")

    print("  By declared direction:")
    for d, stats in report["direction_breakdown"].items():
        acc = f"{stats['dir_acc']*100:.1f}%" if stats["dir_acc"] is not None else "n/a"
        avg_actual = _pct(stats["avg_actual_return"]) if stats["avg_actual_return"] is not None else "n/a"
        print(f"    {d:<5s}  n={stats['n']:>3d}  dir_acc={acc}  actual_avg={avg_actual}")

    print("\n  per-call (most recent first):")
    for s in sorted(scored, key=lambda x: x.as_of_date, reverse=True)[:10]:
        if s.view is None:
            print(f"    {s.as_of_date}  DRY-RUN / error: {s.error or 'dry'}")
            continue
        outcome = "pending" if s.actual_return is None else (
            "hit " if s.direction_correct else "miss"
        )
        actual = _pct(s.actual_return) if s.actual_return is not None else "n/a"
        pred_lo = _pct(s.view.expected_return_low)
        pred_hi = _pct(s.view.expected_return_high)
        print(
            f"    {s.as_of_date}  pred={s.view.direction:<4s} "
            f"conf={s.view.confidence:.2f}  "
            f"range=[{pred_lo}, {pred_hi}]  "
            f"actual={actual}  outcome={outcome}"
        )


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Backtest the LLM analyst.")
    parser.add_argument("--start", required=True, help="First call date YYYY-MM-DD.")
    parser.add_argument("--end", required=True, help="Last call date YYYY-MM-DD (inclusive).")
    parser.add_argument("--sample-every", type=int, default=7,
                        help="Days between calls. Default 7 (weekly).")
    parser.add_argument("--horizon", type=int, default=30,
                        help="Forward horizon in days. Default 30.")
    parser.add_argument("--csv", default=str(snapshot.DEFAULT_MASTER_CSV))
    parser.add_argument("--news-db", default=str(news_store.DEFAULT_DB_PATH))
    parser.add_argument("--db", default=str(analyst.DEFAULT_DB_PATH))
    parser.add_argument("--model", default=analyst.DEFAULT_MODEL)
    parser.add_argument("--max-calls", type=int, default=50,
                        help="Hard cap on API calls (cost guardrail).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip API calls; validate prompt assembly + date gating.")
    parser.add_argument("--out-csv", default=None,
                        help="Optional path to write per-call rows as CSV.")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    if end < start:
        parser.error("--end must be on or after --start")

    scored = run_backtest(
        start=start,
        end=end,
        sample_every_days=args.sample_every,
        horizon_days=args.horizon,
        master_csv=args.csv,
        news_db_path=args.news_db,
        analyst_db_path=args.db,
        model=args.model,
        max_calls=args.max_calls,
        dry_run=args.dry_run,
    )
    report = summarise(scored)
    print_report(scored, report)

    if args.out_csv:
        rows = [s.to_row() for s in scored]
        if rows:
            fieldnames = sorted({k for row in rows for k in row})
            with open(args.out_csv, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
            print(f"  wrote per-call rows -> {args.out_csv}  (n={len(rows)})")


if __name__ == "__main__":
    _cli()
