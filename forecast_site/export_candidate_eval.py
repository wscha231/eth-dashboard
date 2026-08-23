"""Export the newest OOF candidate as a compact, chart-ready public payload.

Unlike the production backtest archive, this file is published even when the
candidate fails its gate.  The website can therefore show current evidence
without treating the candidate as a promoted production model.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

from forecast_site.export_backtest_json import LONGRUN_CHART_MODEL


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _rmse(errors: list[float]) -> float | None:
    return math.sqrt(sum(error * error for error in errors) / len(errors)) if errors else None


def _phase_name(candidate: dict[str, Any]) -> str:
    mode = str(candidate.get("mode") or "latest_candidate")
    if mode.startswith("longrun_oof_"):
        mode = mode[len("longrun_oof_"):]
    for suffix in ("_36x30", "_longrun_oof"):
        if mode.endswith(suffix):
            mode = mode[: -len(suffix)]
    return f"{mode}_latest_candidate"


def _evaluated_through(candidate: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    for horizon, payload in (candidate.get("horizons") or {}).items():
        if not isinstance(payload, dict):
            continue
        dates = [
            str(row["target_date"])
            for row in (payload.get("predictions") or [])
            if isinstance(row, dict) and row.get("target_date")
        ]
        if dates:
            result[str(horizon)] = max(dates)
    return result


def build_candidate_history(
    candidate: dict[str, Any],
    report: dict[str, Any],
    *,
    model: str = LONGRUN_CHART_MODEL,
    generated_at: str | None = None,
) -> dict[str, Any]:
    phase_name = _phase_name(candidate)
    phase = {
        "model_phase": phase_name,
        "frozen_utc": candidate.get("frozen_at"),
        "mode": candidate.get("mode"),
        "model": model,
        "raw_model": model,
        "chart_selection": "fallback_to_no_change_when_point_rmse_has_no_edge",
        "chart_model_by_horizon": {},
        "points": {},
    }

    for horizon, horizon_payload in sorted(
        (candidate.get("horizons") or {}).items(),
        key=lambda item: int(item[0]),
    ):
        if not isinstance(horizon_payload, dict):
            continue
        rows = [
            row
            for row in (horizon_payload.get("predictions") or [])
            if isinstance(row, dict)
            and row.get("head") == "regression"
            and row.get("model") == model
        ]
        rows.sort(key=lambda row: str(row.get("target_date") or ""))

        model_errors: list[float] = []
        anchor_errors: list[float] = []
        for row in rows:
            actual = _finite(row.get("actual_close"))
            raw_prediction = _finite(row.get("predicted_close"))
            reference = _finite(row.get("reference_close"))
            if actual is None:
                continue
            if raw_prediction is not None:
                model_errors.append(raw_prediction - actual)
            if reference is not None:
                anchor_errors.append(reference - actual)

        model_rmse = _rmse(model_errors)
        anchor_rmse = _rmse(anchor_errors)
        use_anchor = bool(
            anchor_rmse is not None
            and (model_rmse is None or anchor_rmse <= model_rmse)
        )
        chart_model = "no_change_anchor" if use_anchor else model
        point_edge = (
            100.0 * (anchor_rmse - model_rmse) / anchor_rmse
            if model_rmse is not None and anchor_rmse
            else None
        )
        phase["chart_model_by_horizon"][str(horizon)] = {
            "raw_model": model,
            "chart_model": chart_model,
            "model_price_rmse": model_rmse,
            "no_change_price_rmse": anchor_rmse,
            "point_forecast_beats_no_change": bool(
                model_rmse is not None and anchor_rmse is not None and model_rmse < anchor_rmse
            ),
            "point_edge_vs_no_change_pct": point_edge,
        }

        points: list[dict[str, Any]] = []
        for row in rows:
            raw_close = row.get("predicted_close")
            raw_return = row.get("predicted_return")
            selected_close = row.get("reference_close") if use_anchor else raw_close
            selected_return = 0.0 if use_anchor else raw_return
            point = {
                "display_date": row.get("target_date"),
                "target_date": row.get("target_date"),
                "prediction_date": row.get("prediction_date"),
                "reference_close": row.get("reference_close"),
                "predicted_close": selected_close,
                "actual_close": row.get("actual_close"),
                "predicted_return": selected_return,
                "actual_return": row.get("actual_return"),
                "chart_model": chart_model,
                "model_predicted_close": raw_close,
                "model_predicted_return": raw_return,
            }
            if use_anchor:
                point["raw_predicted_close"] = raw_close
                point["raw_predicted_return"] = raw_return
                point["benchmark_predicted_close"] = selected_close
                point["benchmark_predicted_return"] = selected_return
            points.append(point)
        phase["points"][str(horizon)] = points

    evaluated_through = report.get("evaluated_through_by_horizon") or _evaluated_through(candidate)
    return {
        "schema_version": 2,
        "generated_at": generated_at or dt.datetime.now(tz=dt.timezone.utc).isoformat(),
        "evaluation_generated_at": report.get("evaluated_at_utc"),
        "evaluation_status": report.get("gate_status") or ("FAIL" if report.get("failures") else "PASS"),
        "evaluation_failures": report.get("failures") or [],
        "evaluation_warnings": report.get("warnings") or [],
        "candidate_frozen_at": candidate.get("frozen_at"),
        "candidate_checkpoint_utc": candidate.get("last_checkpoint_utc"),
        "evaluated_through_by_horizon": evaluated_through,
        "chart_model": model,
        "raw_chart_model": model,
        "chart_selection": "fallback_to_no_change_when_point_rmse_has_no_edge",
        "phases": {phase_name: phase},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default=LONGRUN_CHART_MODEL)
    args = parser.parse_args(argv)

    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    payload = build_candidate_history(candidate, report, model=args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    print(
        f"[export_candidate_eval] wrote {args.output} "
        f"status={payload['evaluation_status']} phases={len(payload['phases'])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
