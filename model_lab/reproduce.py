"""Re-score pinned existing replay without training, collecting, issuing or trading.

Use the existing Drive restore tool to obtain inputs. This consumer never fetches data.
Historical replay is development evidence, NOT untouched or prospective validation.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd

from .contracts import (HORIZONS, TARGET_SPEC, PinnedRelease, canonical, consume_features,
                        digest_bytes, file_hash, load_bars, nonoverlap_count, target_frame, utc)


MODELS = ("climatology", "logistic", "catboost", "catboost_calibrated", "selected")


def score(rows: pd.DataFrame) -> dict:
    if rows.empty:
        raise ValueError("empty evaluation cohort")
    y = rows["return"].to_numpy(float)
    q = rows[["q10", "q50", "q90"]].to_numpy(float)
    r = np.expm1(y); qr = np.expm1(q)
    if (not np.isfinite(np.r_[y, q.ravel()]).all() or (np.diff(q, axis=1) < 0).any()):
        raise ValueError("invalid quantiles or truth")
    error = np.abs(r - qr[:, 1]); base_error = np.abs(r)
    residual = r[:, None] - qr
    pinball = np.maximum(residual * np.array([.1, .5, .9]),
                         residual * np.array([-.9, -.5, -.1]))
    interval = qr[:, 2] - qr[:, 0] + 10 * np.maximum(qr[:, 0] - r, 0) + 10 * np.maximum(r - qr[:, 2], 0)
    out = {
        "rows": len(rows), "return_mae": float(error.mean()),
        "return_rmse": float(np.sqrt(np.mean((r - qr[:, 1]) ** 2))),
        "no_change_mae": float(base_error.mean()),
        "mae_skill_vs_no_change": float(1 - error.mean() / base_error.mean()) if base_error.mean() else None,
        "pinball_mean": float(pinball.mean()), "interval_score80": float(interval.mean()),
        "wis_q10_q50_q90": float((.5 * error + .1 * interval).mean() / 1.5),
        "coverage80": float(((y >= q[:, 0]) & (y <= q[:, 2])).mean()),
        "mean_interval_width": float((qr[:, 2] - qr[:, 0]).mean()),
    }
    if "reference_price" in rows:
        out["price_mae_usd"] = float((rows.reference_price.to_numpy(float) * error).mean())
    if {"p_down", "p_flat", "p_up", "terminal", "hit_up", "hit_down", "up", "down"}.issubset(rows.columns):
        p = rows[["p_down", "p_flat", "p_up"]].to_numpy(float)
        path = rows[["hit_up", "hit_down"]].to_numpy(float)
        truth = rows[["up", "down"]].to_numpy(float)
        terminal = rows.terminal.to_numpy(int)
        if (not np.isfinite(np.r_[p.ravel(), path.ravel(), truth.ravel()]).all()
                or np.any(p < 0) or np.any(p > 1) or not np.allclose(p.sum(axis=1), 1., atol=1e-8, rtol=0)
                or np.any(path < 0) or np.any(path > 1)
                or not np.isin(truth, [0, 1]).all() or not np.isin(terminal, [0, 1, 2]).all()):
            raise ValueError("invalid legacy probabilities or labels")
        out["legacy_terminal_brier"] = float(((p - np.eye(3)[terminal]) ** 2).sum(axis=1).mean())
        out["path_brier"] = float(((path - truth) ** 2).mean())
    return out


def calendar_skill_interval(rows: pd.DataFrame, horizon: int, *, samples: int = 500) -> dict:
    if samples < 10 or rows.empty:
        raise ValueError("insufficient bootstrap configuration")
    origin = pd.to_datetime(rows.slot, utc=True)
    width = max(7, int(np.ceil((horizon + 1) / 24)))
    # Numeric day indices avoid datetime unit assumptions (ns versus us).
    groups = ((origin - pd.Timestamp("1970-01-01", tz="UTC")) / pd.Timedelta(days=width)).astype(int)
    err = np.abs(np.expm1(rows["return"].to_numpy()) - np.expm1(rows.q50.to_numpy()))
    base = np.abs(np.expm1(rows["return"].to_numpy()))
    stats = pd.DataFrame({"group": groups, "e": err, "b": base, "n": 1}).groupby("group")[["e", "b", "n"]].sum().to_numpy()
    out = {"block_days": width, "blocks": len(stats), "samples": samples,
           "interpretation": "exploratory calendar-block percentile interval; unadjusted for model search"}
    if len(stats) < 10:
        return {**out, "lower95": None, "upper95": None}
    rng = np.random.default_rng(1729)
    resampled = stats[rng.integers(len(stats), size=(samples, len(stats)))].sum(axis=1)
    skills = 1 - resampled[:, 0] / resampled[:, 1]
    return {**out, "lower95": float(np.quantile(skills, .025)), "upper95": float(np.quantile(skills, .975))}


def evaluate(release: PinnedRelease, root: Path, outdir: Path) -> dict:
    """Output directory must be new; failures cannot overwrite prior experiment evidence."""
    if outdir.exists():
        raise FileExistsError("experiment output already exists; use a new experiment ID")
    outdir.mkdir(parents=True)
    try:
        return _evaluate(release, root, outdir)
    except Exception as exc:
        (outdir / "FAILED.json").write_bytes(canonical({"status": "failed", "error": type(exc).__name__, "detail": str(exc), "release_sha256": release.sha256}))
        raise


def _evaluate(release: PinnedRelease, root: Path, outdir: Path) -> dict:
    bars = load_bars(release, root)
    features = consume_features(bars, release)
    cols = [c for c in features if c not in {"reference_price", "sigma", "available_at"}]
    cutoff = utc(release.metadata["data_cutoff_utc"])
    artifact = release.path(root, "research.zip")
    z = zipfile.ZipFile(artifact)
    manifest_paths = [n for n in z.namelist() if n.startswith("archive/historical_manifest_") and n.endswith(".json")]
    if len(manifest_paths) != 1:
        raise ValueError("exactly one frozen historical manifest required")
    old_manifest = json.loads(z.read(manifest_paths[0]))
    input_summary = {}
    for product, group in bars.groupby("product"):
        grid = pd.date_range(group.open_time.min(), cutoff - pd.Timedelta(hours=1), freq="h")
        input_summary[product] = {"rows": len(group), "first_open": group.open_time.min().isoformat(),
                                  "last_open": group.open_time.max().isoformat(),
                                  "missing_hours": len(grid.difference(pd.DatetimeIndex(group.open_time))),
                                  "earliest_receipt": group.observed_at.min().isoformat(),
                                  "latest_receipt": group.observed_at.max().isoformat()}
    coverage = {c: {"valid": int(features[c].notna().sum()), "missing": int(features[c].isna().sum())} for c in cols}
    summary = {"status": "retrospective_development_reproduction", "release_id": release.metadata["release_id"],
               "release_sha256": release.sha256, "data_cutoff": cutoff.isoformat(),
               "target_spec": TARGET_SPEC, "target_spec_sha256": digest_bytes(canonical(TARGET_SPEC)),
               "source": input_summary, "feature_count": len(cols), "coverage_by_feature": coverage,
               "implementation": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__},
               "horizons": {}, "promotion": "none", "new_live_evidence": False,
               "direction_note": "legacy p_flat means inside volatility barriers, not P(return==0); no sign probability inferred"}
    metrics_rows = []; proofs = []
    for h in HORIZONS:
        name = f"replay/h{h}.csv.gz"
        raw = z.read(name)
        f = pd.read_csv(io.BytesIO(gzip.decompress(raw)))
        if set(f.model) != set(MODELS) or f.duplicated(["slot", "model"]).any():
            raise ValueError("unexpected candidate set or duplicate forecast origin")
        origin = pd.to_datetime(f.slot, utc=True); end = pd.to_datetime(f.target_end, utc=True)
        if (not (end == origin + pd.Timedelta(hours=h + 1)).all() or (end > cutoff).any()
                or (pd.to_datetime(f.training_target_end, utc=True) >= pd.to_datetime(f.fit_cutoff, utc=True)).any()
                or (pd.to_datetime(f.validation_target_end, utc=True) >= pd.to_datetime(f.fit_cutoff, utc=True)).any()
                or (pd.to_datetime(f.fit_cutoff, utc=True) > origin).any()):
            raise ValueError("future labels or incompatible historical target times")
        truthcols = ["target_end", "reference_price", "return", "up", "down", "terminal"]
        if f.groupby("slot")[truthcols].nunique().gt(1).any().any():
            raise ValueError("candidate truth mismatch")
        common = set.intersection(*(set(f.loc[f.model.eq(m), "slot"]) for m in MODELS))
        if any(len(common) != f.model.eq(m).sum() for m in MODELS):
            raise ValueError("unequal candidate samples; do not silently drop failures")
        target = target_frame(bars, features, h)
        selected = f[f.model.eq("selected")].sort_values("slot").copy()
        times = pd.DatetimeIndex(pd.to_datetime(selected.slot, utc=True))
        actual = target.loc[times]
        for column in ("return", "up", "down", "terminal", "barrier"):
            if not np.allclose(actual[column].to_numpy(), selected[column].to_numpy(), atol=2e-10, rtol=1e-9, equal_nan=False):
                raise ValueError(f"archived {h}h {column} does not reproduce from pinned source; inspect revisions")
        computed = score(selected)
        previous = old_manifest["horizons"][str(h)]["metrics"]
        for new, old in (("return_mae", "return_mae"), ("coverage80", "coverage80"), ("interval_score80", "interval_score80_pp")):
            scale = 100 if old.endswith("_pp") else 1
            if not np.isclose(computed[new] * scale, previous[old], atol=1e-9, rtol=1e-9):
                raise ValueError(f"archived metric does not reproduce: {h}/{new}")
        details = {}
        for m in MODELS:
            group = f[f.model.eq(m)]
            details[m] = score(group)
            metrics_rows.append({"horizon_hours": h, "model": m, **details[m]})
        noc = selected.copy(); noc[["q10", "q50", "q90"]] = 0.
        noc_score = score(noc)
        # Only point loss is meaningful for a zero-width no-change point baseline.
        details["no_change_point"] = {k: noc_score[k] for k in ("rows", "return_mae", "return_rmse", "price_mae_usd")}
        daily = features.index[features.index.hour == 0]
        scheduled = daily[(daily >= times.min()) & (daily + pd.Timedelta(hours=h + 1) <= cutoff)]
        summary["horizons"][str(h)] = {
            "common_origins": len(times), "first_origin": times.min().isoformat(), "last_origin": times.max().isoformat(),
            "nonoverlap_count": nonoverlap_count(times, actual.target_end),
            "scheduled_daily_origins_in_evaluation_window": len(scheduled),
            "not_replayed_origins": len(scheduled.difference(times)),
            "source_and_target_checks": "pass", "incumbent_metric_reproduction": "pass", "models": details,
            "incumbent_mae_skill_interval": calendar_skill_interval(selected, h),
            "selected_model_counts": {str(k): int(v) for k, v in selected.selected_model.value_counts().items()},
            "status": "baseline_no_proven_price_edge", "candidate_training_this_run": False,
            "note": "Excluded calendar origins reported; replay is not full service uptime or a new holdout."}
        proofs.append({"horizon_hours": h, "replay_file": name, "sha256": digest_bytes(raw), "paired_rows": len(f)})
        # Derived row evidence stays in the private output bundle, not public source control.
        selected.to_csv(outdir / f"incumbent_rows_h{h}.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    summary["replay_proofs"] = proofs
    summary["inputs_unchanged_after_run"] = (file_hash(artifact) == release.metadata["checksums"]["research.zip"]
                                             and file_hash(release.path(root, "observations.db")) == release.metadata["checksums"]["observations.db"])
    if not summary["inputs_unchanged_after_run"]:
        raise ValueError("input mutation detected")
    pd.DataFrame(metrics_rows).to_csv(outdir / "candidate_metrics.csv", index=False)
    (outdir / "summary.json").write_bytes(canonical(summary))
    (outdir / "output_hashes.json").write_bytes(canonical({p.name: file_hash(p) for p in sorted(outdir.iterdir()) if p.is_file()}))
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True); p.add_argument("--release-sha256", required=True)
    p.add_argument("--input-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = evaluate(PinnedRelease.load(a.release, a.release_sha256), a.input_root, a.output)
    print(json.dumps({"status": result["status"], "horizons": list(result["horizons"]), "promotion": "none"}))


if __name__ == "__main__":
    main()
