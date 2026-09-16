"""Operational HAR-RV refit adapter with a bounded source-lag allowance.

The retrospective R2 model, target, train/calibration windows and coefficients are unchanged.
This module only separates model-refit freshness from issuance freshness: a monthly refit may
use a source snapshot ending at most one completed hour before the wall-clock hour. Actual
shadow issuance remains guarded by variance_shadow.run() and requires the current origin row.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import build_features, read_bars, utc
from . import variance_shadow as base

MAX_REFIT_SOURCE_LAG_HOURS = 1


def fit_checkpoint(root, horizon: int, *, now=None) -> dict:
    if horizon not in base.HORIZONS:
        raise ValueError("unsupported variance-shadow horizon")
    fit_time = utc(now)
    boundary = fit_time.floor("h")
    bars = read_bars(root, as_of=fit_time)
    latest = bars.groupby("product").close_time.max().min()
    if pd.isna(latest) or latest > boundary:
        raise ValueError("invalid variance refit source cutoff")
    source_lag = (boundary - latest) / pd.Timedelta(hours=1)
    if source_lag > MAX_REFIT_SOURCE_LAG_HOURS:
        raise ValueError("source snapshot is too stale for variance refit")

    features = build_features(bars)
    targets = base._target_frame(bars, features, horizon)
    calibration_start = boundary - pd.Timedelta(days=base.POLICY["calibration_days"])
    training_start = calibration_start - pd.Timedelta(days=base.POLICY["train_days"])
    train = base._fit_indices(features, targets, training_start, calibration_start)
    calibration = base._fit_indices(features, targets, calibration_start, boundary)
    if len(train) < 400 or len(calibration) < 45:
        raise ValueError("insufficient purged HAR-RV train/calibration history")

    vx = np.log(features[list(base.VOL_COLUMNS)].pow(2).clip(lower=1e-16))
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0)).fit(
        vx.loc[train], np.log(targets.loc[train, "realized_variance_total"].to_numpy() / horizon)
    )
    raw_calibration = np.exp(model.predict(vx.loc[calibration]))
    actual_per_hour = targets.loc[calibration, "realized_variance_total"].to_numpy() / horizon
    multiplier = float(np.mean(actual_per_hour / raw_calibration))
    if not np.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("invalid HAR-RV calibration multiplier")

    checkpoint = {
        "schema": 1,
        "policy": base.POLICY,
        "horizon_hours": horizon,
        "fit_time": fit_time.isoformat(),
        "fit_boundary": boundary.isoformat(),
        "training_source_as_of": latest.isoformat(),
        "refit_source_lag_hours": float(source_lag),
        "training_rows": int(len(train)),
        "calibration_rows": int(len(calibration)),
        "training_target_end": targets.loc[train, "target_end"].max().isoformat(),
        "calibration_target_end": targets.loc[calibration, "target_end"].max().isoformat(),
        "source_snapshot": base._source_hash(bars, boundary),
        "columns": list(base.VOL_COLUMNS),
        "scale_mean": model[0].mean_.tolist(),
        "scale_std": model[0].scale_.tolist(),
        "coef": model[-1].coef_.tolist(),
        "intercept": float(model[-1].intercept_),
        "multiplier": multiplier,
    }
    checkpoint["model_version"] = base.digest(checkpoint)
    return checkpoint
