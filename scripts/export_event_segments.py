"""Export already computed slices; hourly publication never re-runs a backtest."""
import argparse
import csv
import json
from pathlib import Path


def export(payload, output):
    fields = ["data_as_of", "horizon_hours", "period", "market_state", "period_start", "period_end_exclusive",
              "first_origin", "last_origin", "common_origins", "nonoverlap_origins", "model", "event_brier",
              "event_skill_vs_frequency", "terminal_balanced_accuracy", "up_recall", "up_false_positive_rate",
              "down_recall", "down_false_positive_rate", "price_mae_skill_vs_no_change", "coverage80",
              "selected_brier_difference_low95", "selected_brier_difference_high95"]
    count = 0
    with Path(output).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for horizon, result in payload["horizons"].items():
            segments = result.get("segments")
            if not segments: continue
            periods = {p["id"]: p for p in segments["periods"]}
            for key, summary in segments["results"].items():
                period, state = key.split("|"); dates = periods[period]
                baseline = next((m for m in summary["models"] if m["model"] == "climatology"), {})
                for model in summary["models"] or [{"model": "selected"}]:
                    interval = (summary.get("paired_event_brier") or {}) if model["model"] == "selected" else {}
                    writer.writerow(dict(data_as_of=segments["as_of"], horizon_hours=horizon, period=period,
                        market_state=state, period_start=dates["start"], period_end_exclusive=dates["end"],
                        first_origin=summary.get("first_origin"), last_origin=summary.get("last_origin"),
                        common_origins=summary["common_origins"], nonoverlap_origins=summary["nonoverlapping_selected"]["rows"],
                        model=model["model"], event_brier=model.get("event_brier"),
                        event_skill_vs_frequency=1-model["event_brier"]/baseline["event_brier"] if baseline.get("event_brier") else None,
                        terminal_balanced_accuracy=model.get("terminal_balanced_accuracy"),
                        up_recall=model.get("up", {}).get("recall"), up_false_positive_rate=model.get("up", {}).get("false_positive_rate"),
                        down_recall=model.get("down", {}).get("recall"), down_false_positive_rate=model.get("down", {}).get("false_positive_rate"),
                        price_mae_skill_vs_no_change=model.get("mae_skill"), coverage80=model.get("coverage80"),
                        selected_brier_difference_low95=interval.get("lower95"), selected_brier_difference_high95=interval.get("upper95")))
                    count += 1
    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(); parser.add_argument("input"); parser.add_argument("output")
    args = parser.parse_args()
    print("Segment CSV rows:", export(json.loads(Path(args.input).read_text()), args.output))
