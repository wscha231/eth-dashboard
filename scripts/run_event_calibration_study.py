"""Execute the predeclared study using local normalized provider snapshots.

Never writes active checkpoints, issues forecasts, or changes existing ledgers.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import numpy as np
import pandas as pd

from research_pipeline.event_calibration import (
    VERSION, historical_derivative_features, long_cash_diagnostic, train_calibrated_bundle,
)
from signal_pipeline.data import build_features, feature_columns, utc
from signal_pipeline.evaluate import metrics, paired_block_interval
from signal_pipeline.models import labels, predict_models, train_bundle
from signal_pipeline.segments import origin_market_states


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def rows_from_bundle(bundle, features, outcomes, states, targets, horizon, variant):
    candidates = predict_models(bundle["model"], features.loc[targets])
    names = {variant: bundle["choice"]}
    if variant == "frozen":
        names["frequency"] = "climatology"
    rows = []
    for name, candidate in names.items():
        prediction = candidates[candidate]
        thresholds = bundle["alert_thresholds_by_model"][candidate]
        for j, slot in enumerate(targets):
            y = outcomes.loc[slot]
            rows.append({"slot": slot.isoformat(), "target_end": y.target_end.isoformat(),
                         "horizon_hours": horizon, "variant": name, "selected_model": candidate,
                         "fit_cutoff": bundle["fit_cutoff"], **states.loc[slot].to_dict(),
                         "return": float(y["return"]), "up": int(y.up), "down": int(y.down),
                         "terminal": int(y.terminal), "p_down": float(prediction["terminal"][j, 0]),
                         "p_flat": float(prediction["terminal"][j, 1]),
                         "p_up": float(prediction["terminal"][j, 2]),
                         "hit_up": float(prediction["path"][j, 0]),
                         "hit_down": float(prediction["path"][j, 1]),
                         "threshold_up": thresholds["up"], "threshold_down": thresholds["down"],
                         "q10": float(prediction["quantiles"][j, 0]),
                         "q50": float(prediction["quantiles"][j, 1]),
                         "q90": float(prediction["quantiles"][j, 2])})
    return rows


def summarize(rows, bars, horizon, start, end):
    frame = pd.DataFrame(rows)
    variants = ["legacy", "frozen", "funding", "dvol", "frequency"]
    if frame.empty:
        return {"status": "no_common_origins"}
    if frame.duplicated(["variant", "slot"]).any():
        raise ValueError("duplicate paired origin")
    common = set.intersection(*(set(frame.loc[frame.variant.eq(v), "slot"]) for v in variants))
    if set(frame.slot) != common:
        raise ValueError("all variants must have exactly matched origins")
    truth = ["return", "up", "down", "terminal", "target_end", "trend_state", "volatility_state"]
    if (frame.groupby("slot")[truth].nunique() > 1).any().any():
        raise ValueError("mismatched ground truth")
    groups = {v: frame.loc[frame.variant.eq(v)].sort_values("slot").to_dict("records") for v in variants}
    overall = {v: metrics(g) for v, g in groups.items()}
    for value in overall.values():
        value["event_brier_skill_vs_frequency"] = 1 - value["event_brier"] / overall["frequency"]["event_brier"]
    intervals = {"frozen_minus_legacy": paired_block_interval(groups["frozen"], groups["legacy"], horizon),
                 "frozen_minus_frequency": paired_block_interval(groups["frozen"], groups["frequency"], horizon),
                 "funding_minus_frozen": paired_block_interval(groups["funding"], groups["frozen"], horizon),
                 "dvol_minus_frozen": paired_block_interval(groups["dvol"], groups["frozen"], horizon)}
    independent = []; previous_end = None
    for row in groups["frozen"]:
        if previous_end is None or utc(row["slot"]) >= previous_end:
            independent.append(row["slot"]); previous_end = utc(row["target_end"])
    independent = set(independent)
    slices = {}
    slots = pd.to_datetime(frame.slot, utc=True)
    masks = {"year_2025": slots.dt.year.eq(2025), "year_2026": slots.dt.year.eq(2026),
             "recent_365d": slots >= end - pd.Timedelta(days=365)}
    masks.update({"trend_" + state: frame.trend_state.eq(state) for state in ("up", "down", "range")})
    masks.update({"vol_" + state: frame.volatility_state.eq(state) for state in ("high", "normal")})
    for label, mask in masks.items():
        slices[label] = {v: metrics(frame.loc[mask & frame.variant.eq(v)].to_dict("records")) for v in variants}
    return {"status": "retrospective_development", "common_origins": len(common),
            "first_origin": min(common), "last_origin": max(common), "metrics": overall,
            "paired_event_brier": intervals,
            "nonoverlapping": {v: metrics([r for r in g if r["slot"] in independent]) for v, g in groups.items()},
            "selected_model_counts": {v: dict(pd.Series([r["selected_model"] for r in g]).value_counts().items())
                                      for v, g in groups.items()},
            "segments": slices,
            "trading": {v: [long_cash_diagnostic(g, bars, start, end, cost) for cost in (0, 10, 25)]
                        for v, g in groups.items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hourly", type=Path, required=True)
    parser.add_argument("--funding", type=Path, required=True)
    parser.add_argument("--dvol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", default="2025-01-01T00:00:00Z")
    parser.add_argument("--end", default="2026-09-07T00:00:00Z")
    parser.add_argument("--horizons", type=int, nargs="+", default=[6, 24], choices=[6, 24])
    args = parser.parse_args()
    started = time.monotonic()
    start, end = utc(args.start), utc(args.end)
    if start != start.normalize() or end != end.normalize() or end <= start:
        raise ValueError("whole UTC-day study bounds required")
    args.output.mkdir(parents=True, exist_ok=True)
    bars = pd.read_parquet(args.hourly)
    if bars.duplicated(["product", "close_time"]).any():
        raise ValueError("single normalized source vintage required")
    bars = bars[bars.close_time <= end].copy()
    funding, dvol = pd.read_parquet(args.funding), pd.read_parquet(args.dvol)
    base = build_features(bars)
    extra = historical_derivative_features(base.index, funding, dvol, reconstruction=True)
    common = base[feature_columns(base)].notna().all(axis=1) & extra.notna().all(axis=1)
    # Mask all variants equally even in their earlier training windows.
    base.loc[~common, feature_columns(base)] = np.nan
    variants = {"legacy": base, "frozen": base,
                "funding": base.join(extra.filter(like="funding_")),
                "dvol": base.join(extra.filter(like="dvol_"))}
    states = origin_market_states(base)
    source_paths = {"hourly": args.hourly, "funding": args.funding, "dvol": args.dvol}
    inputs = {k: {"filename": p.name, "sha256": sha256(p)} for k, p in source_paths.items()}
    for key, data, timecol in (("hourly", bars, "close_time"), ("funding", funding, "event_time"), ("dvol", dvol, "event_end")):
        receipt = "observed_at" if key == "hourly" else "received_at"
        inputs[key].update(rows=len(data), first_event=data[timecol].min().isoformat(),
                           last_event=data[timecol].max().isoformat(),
                           first_receipt=data[receipt].min().isoformat(), last_receipt=data[receipt].max().isoformat())
    repo = Path(__file__).resolve().parents[1]
    code = {p: sha256(repo / p) for p in ("signal_pipeline/models.py", "signal_pipeline/data.py",
            "signal_pipeline/protocol.py", "research_pipeline/event_calibration.py", "scripts/run_event_calibration_study.py")}
    cache_key = hashlib.sha256(json.dumps({"inputs": inputs, "code": code}, sort_keys=True).encode()).hexdigest()
    cache = args.output / "checkpoints" / cache_key
    cache.mkdir(parents=True, exist_ok=True)
    payload = {"study_version": VERSION, "base_commit": "baf022d907512d3fdd79c819616095476c6e10b0",
               "generated_at": utc().isoformat(), "start": start.isoformat(), "data_cutoff": end.isoformat(),
               "inputs": inputs, "code_hashes": code,
               "availability": extra.attrs["availability"],
               "scope": "retrospective development; no production or portfolio promotion", "horizons": {}}
    atomic_json(args.output / "study_manifest.json", payload)
    months = pd.date_range(start.replace(day=1), end.replace(day=1), freq="MS")
    for horizon in args.horizons:
        outcomes = labels(bars, base, horizon)
        all_rows = []; audits = []
        for cutoff in months:
            targets = base.index[common & (base.index >= max(start, cutoff))
                                 & (base.index < cutoff + pd.offsets.MonthBegin(1))
                                 & (base.index.hour == 0) & outcomes["return"].notna()
                                 & (outcomes.target_end <= end)]
            if not len(targets):
                continue
            for variant, features in variants.items():
                checkpoint = cache / f"h{horizon}_{cutoff:%Y-%m}_{variant}.joblib"
                if checkpoint.exists():
                    # Only this isolated study creates these local checkpoints.
                    bundle = joblib.load(checkpoint)
                else:
                    trainer = train_bundle if variant == "legacy" else train_calibrated_bundle
                    bundle = trainer(features, outcomes, cutoff, horizon)
                    joblib.dump(bundle, checkpoint)
                audit = {k: v for k, v in bundle.items() if k != "model"}
                audits.append({"variant": variant, **audit})
                all_rows.extend(rows_from_bundle(bundle, features, outcomes, states, targets, horizon, variant))
                print(f"h={horizon} month={cutoff:%Y-%m} variant={variant} model={bundle['choice']} origins={len(targets)} elapsed={time.monotonic()-started:.1f}s", flush=True)
        result = summarize(all_rows, bars, horizon, start, end)
        result["monthly_audits"] = audits
        payload["horizons"][str(horizon)] = result
        pd.DataFrame(all_rows).to_csv(args.output / f"h{horizon}_predictions.csv.gz", index=False,
                                     compression={"method": "gzip", "mtime": 0})
        payload["runtime_seconds"] = time.monotonic() - started
        atomic_json(args.output / "results.json", payload)
        print(json.dumps({"horizon": horizon, "metrics": result.get("metrics"),
                          "paired_event_brier": result.get("paired_event_brier")}), flush=True)


if __name__ == "__main__":
    main()
