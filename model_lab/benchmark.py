"""Bounded, fixed-policy development benchmark; never a promotion or production job."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .contracts import (HORIZONS, PinnedRelease, canonical, consume_features, digest_bytes,
                        file_hash, load_bars, purged_indices, target_frame, utc)
from .reproduce import calendar_skill_interval, score

POLICY = {
    "experiment_id": "20260915_matched_development_v1",
    "evidence_grade": "already_inspected_historical_development_not_untouched",
    "folds": ["2024-01-01T00:00:00Z", "2025-01-01T00:00:00Z", "2026-01-01T00:00:00Z"],
    "fold_test_months": 12, "stride_hours": 24, "calibration_days": 180,
    "train_days_short_long": [730, 1095], "embargo_hours": 1,
    "minimum_train": 400, "minimum_calibration": 45,
    "features": "fixed existing 32 DATA OHLCV features; matched ETH-only Ridge ablation",
    "ridge_alpha": 10.,
    "catboost": {"iterations": 100, "depth": 4, "learning_rate": .035, "l2_leaf_reg": 20,
                 "thread_count": 2, "random_seed": 1729, "verbose": False, "allow_writing_files": False},
    "quantiles": [.1, .5, .9],
    "har": "log(total future realized variance/h) on log(prior 24/168/720 hourly sample variance); prior calibration mean-ratio correction",
    "gaussian": "zero log-return center; endpoint variance=(h+1)/h times path-window variance",
    "direction": "binary P(terminal log return>0); zero counted as nonpositive; separate from barriers",
    "direction_calibration": "Platt sigmoid of prior calibration decision scores; fallback to smoothed prior frequency",
    "primary_losses": {"point": "simple-return MAE", "distribution": "WIS q10/q50/q90", "variance": "QLIKE", "direction": "Brier"},
    "promotion": "disabled; annual frozen fits are not the monthly production service",
    "stop": "invalid input, nonfinite output, insufficient prior labels or cohort mismatch => stop and preserve FAILED.json",
    "compute_budget": "local CPU only; 18 CatBoost fits; no external API or paid service",
}


def qlike(actual: np.ndarray, predicted: np.ndarray) -> float:
    if (not np.isfinite(np.r_[actual, predicted]).all() or (actual <= 0).any() or (predicted <= 0).any()):
        raise ValueError("QLIKE needs finite strictly positive realized/predicted variance")
    ratio = actual / predicted
    return float((ratio - np.log(ratio) - 1).mean())


def direction_model(xtrain, ytrain, xcal, ycal, xtest) -> tuple[np.ndarray, float]:
    prior = float((ytrain.sum() + 1) / (len(ytrain) + 2))
    if len(np.unique(ytrain)) < 2:
        return np.repeat(prior, len(xtest)), prior
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=1000, random_state=1729)).fit(xtrain, ytrain)
    if len(np.unique(ycal)) < 2:
        return np.repeat((ycal.sum() + 1) / (len(ycal) + 2), len(xtest)), prior
    cal = LogisticRegression(C=1., max_iter=1000, random_state=1729).fit(model.decision_function(xcal).reshape(-1, 1), ycal)
    return cal.predict_proba(model.decision_function(xtest).reshape(-1, 1))[:, 1], prior


def benchmark(release: PinnedRelease, root: Path, outdir: Path) -> dict:
    if outdir.exists():
        raise FileExistsError("immutable experiment output already exists")
    outdir.mkdir(parents=True)
    (outdir / "policy.json").write_bytes(canonical(POLICY))  # Persist BEFORE looking at new results.
    try:
        return _benchmark(release, root, outdir)
    except Exception as exc:
        (outdir / "FAILED.json").write_bytes(canonical({"error": type(exc).__name__, "detail": str(exc), "promotion": "none"}))
        raise


def _benchmark(release: PinnedRelease, root: Path, outdir: Path) -> dict:
    started = time.monotonic()
    bars = load_bars(release, root); features = consume_features(bars, release)
    cols = [c for c in features if c not in {"reference_price", "sigma", "available_at"}]
    ethcols = [c for c in cols if c.startswith("eth_") and not c.startswith("eth_btc_")]
    cutoff = utc(release.metadata["data_cutoff_utc"])
    allrows = []; folds = []; summaries = {}; saved_models = outdir / "models"; saved_models.mkdir()
    for h in HORIZONS:
        targets = target_frame(bars, features, h)
        valid = features[cols].notna().all(axis=1) & targets["return"].notna() & targets.realized_variance_total.gt(0)
        for foldtime in POLICY["folds"]:
            start = utc(foldtime); end = start + pd.DateOffset(months=12)
            calstart = start - pd.Timedelta(days=180)
            train_days = 1095 if h >= 336 else 730
            train = purged_indices(features, targets, calstart - pd.Timedelta(days=train_days), calstart)
            cal = purged_indices(features, targets, calstart, start)
            train = train[valid.loc[train]]; cal = cal[valid.loc[cal]]
            scheduled = features.index[(features.index.hour == 0) & (features.index >= start) & (features.index < end)
                                       & (features.index + pd.Timedelta(hours=h + 1) <= cutoff)]
            test = scheduled[valid.loc[scheduled]]
            if len(train) < 400 or len(cal) < 45 or not len(test):
                raise ValueError(f"insufficient prior train/calibration/test at {h}/{start}")
            ytrain = targets.loc[train, "return"].to_numpy(); ycal = targets.loc[cal, "return"].to_numpy()
            xt, xc, xv = (features.loc[ix, cols].to_numpy(float) for ix in (train, cal, test))
            sig2 = features.loc[test, "eth_vol_720"].to_numpy() ** 2
            gaussian = np.array([-1.2815515655446004, 0., 1.2815515655446004])
            preds = {"zero_gaussian_rv30": np.sqrt(sig2 * (h + 1))[:, None] * gaussian,
                     "climatology_quantiles": np.tile(np.quantile(ytrain, [.1, .5, .9]), (len(test), 1))}
            fitted_meta = {}
            for name, columns in (("ridge_all", cols), ("ridge_eth_only", ethcols)):
                reg = make_pipeline(StandardScaler(), Ridge(alpha=10.)).fit(features.loc[train, columns], ytrain)
                residual = ycal - reg.predict(features.loc[cal, columns])
                preds[name] = reg.predict(features.loc[test, columns])[:, None] + np.quantile(residual, [.1, .5, .9])
                fitted_meta[name] = {"columns": columns, "coef": reg[-1].coef_.tolist(), "intercept": float(reg[-1].intercept_),
                                     "scale_mean": reg[0].mean_.tolist(), "scale_std": reg[0].scale_.tolist(),
                                     "residual_quantiles": np.quantile(residual, [.1, .5, .9]).tolist()}
            cat = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9", **POLICY["catboost"]).fit(xt, ytrain)
            correction = [np.quantile(ycal - cat.predict(xc)[:, j], q) for j, q in enumerate([.1, .5, .9])]
            preds["catboost_quantiles"] = np.sort(cat.predict(xv) + np.asarray(correction), axis=1)
            modelpath = saved_models / f"catboost_h{h}_{start.year}.cbm"; cat.save_model(str(modelpath))
            fitted_meta["catboost_quantiles"] = {"file": modelpath.name, "sha256": file_hash(modelpath), "correction": correction}
            vcols = ["eth_vol_24", "eth_vol_168", "eth_vol_720"]
            vx = np.log(features[vcols].pow(2).clip(lower=1e-16))
            har = make_pipeline(StandardScaler(), Ridge(alpha=10.)).fit(vx.loc[train], np.log(targets.loc[train, "realized_variance_total"].to_numpy() / h))
            multiplier = float(np.mean((targets.loc[cal, "realized_variance_total"].to_numpy() / h) / np.exp(har.predict(vx.loc[cal]))))
            harvar = h * np.exp(har.predict(vx.loc[test])) * multiplier
            preds["har_variance_gaussian"] = np.sqrt(harvar * (h + 1) / h)[:, None] * gaussian
            probability, prior = direction_model(xt, (ytrain > 0).astype(int), xc, (ycal > 0).astype(int), xv)
            fitted_meta["har"] = {"coef": har[-1].coef_.tolist(), "intercept": float(har[-1].intercept_), "multiplier": multiplier,
                                   "scale_mean": har[0].mean_.tolist(), "scale_std": har[0].scale_.tolist(), "columns": vcols}
            fitted_meta["direction"] = {"artifact_status": "probabilities_only_no_live_model_export", "prior_positive": prior}
            (saved_models / f"model_meta_h{h}_{start.year}.json").write_bytes(canonical(fitted_meta))
            for model, q in preds.items():
                if not np.isfinite(q).all() or (np.diff(q, axis=1) < 0).any():
                    raise ValueError("invalid candidate quantiles")
                frame = pd.DataFrame({"slot": test, "target_end": targets.loc[test, "target_end"].to_numpy(), "horizon_hours": h,
                                      "fold_year": start.year, "model": model, "fit_cutoff": start,
                                      "reference_price": features.loc[test, "reference_price"].to_numpy(),
                                      "return": targets.loc[test, "return"].to_numpy(), "q10": q[:, 0], "q50": q[:, 1], "q90": q[:, 2]})
                trend_z = features.loc[test, "eth_ret_24"].to_numpy() / (features.loc[test, "eth_vol_720"].to_numpy() * np.sqrt(24))
                frame["trend_state"] = np.where(trend_z > 1, "up", np.where(trend_z < -1, "down", "range"))
                frame["volatility_state"] = np.where(features.loc[test, "eth_vol_24"].to_numpy() / features.loc[test, "eth_vol_720"].to_numpy() >= 1.5, "high", "normal")
                if model in {"zero_gaussian_rv30", "har_variance_gaussian"}:
                    frame["variance_total"] = h * sig2 if model == "zero_gaussian_rv30" else harvar
                    frame["realized_variance_total"] = targets.loc[test, "realized_variance_total"].to_numpy()
                if model == "ridge_all":
                    frame["p_positive"] = probability; frame["prior_positive"] = prior
                allrows.append(frame)
            folds.append({"horizon_hours": h, "fold_year": start.year, "training": len(train), "calibration": len(cal), "test": len(test),
                          "eligible_calendar_slots": len(scheduled), "excluded_slots": len(scheduled) - len(test),
                          "train_target_end": targets.loc[train, "target_end"].max().isoformat(),
                          "calibration_target_end": targets.loc[cal, "target_end"].max().isoformat(), "fit_cutoff": start.isoformat()})
            print(f"h={h} fold={start.year} train={len(train)} cal={len(cal)} test={len(test)}", flush=True)
        f = pd.concat([r for r in allrows if r.horizon_hours.iloc[0] == h], ignore_index=True)
        results = {}
        for model, group in f.groupby("model"):
            d = score(group)
            d["mae_skill_interval"] = calendar_skill_interval(group, h)
            d["yearly"] = {str(year): score(g) for year, g in group.groupby("fold_year")}
            d["regimes"] = {str(trend) + "|" + str(vol): score(g) for (trend, vol), g in group.groupby(["trend_state", "volatility_state"])}
            if group.variance_total.notna().all():
                d["variance_qlike"] = qlike(group.realized_variance_total.to_numpy(), group.variance_total.to_numpy())
            if group.p_positive.notna().all():
                truth = (group["return"].to_numpy() > 0).astype(int)
                d["sign_brier"] = float(np.mean((group.p_positive.to_numpy() - truth) ** 2))
                d["sign_prior_brier"] = float(np.mean((group.prior_positive.to_numpy() - truth) ** 2))
                d["sign_note"] = POLICY["direction"]
                # Separate task models are not necessarily one coherent predictive distribution.
                mismatch = ((group.q50.to_numpy() > 0) & (group.p_positive.to_numpy() < .5)) | ((group.q50.to_numpy() < 0) & (group.p_positive.to_numpy() > .5))
                d["median_direction_disagreement_rows"] = int(mismatch.sum())
            results[model] = d
        summaries[str(h)] = {"models": results, "promotion": "none", "status": "development_only"}
    rows = pd.concat(allrows, ignore_index=True)
    rows.to_csv(outdir / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    result = {"policy": POLICY, "policy_sha256": digest_bytes(canonical(POLICY)), "release_id": release.metadata["release_id"],
              "release_sha256": release.sha256, "horizons": summaries, "folds": folds, "runtime_seconds": time.monotonic() - started,
              "code_hashes": {p.name: file_hash(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
              "dependencies": {n: importlib.metadata.version(n) for n in ("numpy", "pandas", "scikit-learn", "catboost")},
              "cost": {"external_api_calls": 0, "paid_services": 0, "catboost_fits": 18},
              "promotion": "disabled; no new prospective evidence; no site or ledger writes"}
    if file_hash(release.path(root, "observations.db")) != release.metadata["checksums"]["observations.db"]:
        raise ValueError("input changed during benchmark")
    (outdir / "summary.json").write_bytes(canonical(result))
    (outdir / "output_hashes.json").write_bytes(canonical({p.relative_to(outdir).as_posix(): file_hash(p) for p in sorted(outdir.rglob("*")) if p.is_file()}))
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True); p.add_argument("--release-sha256", required=True)
    p.add_argument("--input-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); benchmark(PinnedRelease.load(a.release, a.release_sha256), a.input_root, a.output)


if __name__ == "__main__":
    main()
