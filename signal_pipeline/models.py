"""Purged monthly training; event probabilities are learned directly."""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import feature_columns
from .protocol import HORIZONS


def labels(bars, features, horizon):
    eth = bars.loc[bars["product"].eq("ETH-USD")].set_index("close_time").reindex(features.index)
    boundary = np.maximum(np.log1p(HORIZONS[horizon]), features.sigma*np.sqrt(horizon))
    result = pd.DataFrame(index=features.index)
    result["barrier"] = boundary
    result["return"] = np.log(eth.close.shift(-(horizon+1))/features.reference_price)
    result["target_end"] = features.index+pd.Timedelta(hours=horizon+1)
    high = eth.high.rolling(horizon, min_periods=horizon).max().shift(-(horizon+1))
    low = eth.low.rolling(horizon, min_periods=horizon).min().shift(-(horizon+1))
    result["up"] = (np.log(high/features.reference_price) >= boundary).astype(float)
    result["down"] = (np.log(low/features.reference_price) <= -boundary).astype(float)
    result["terminal"] = np.where(result["return"] > boundary, 2,
                                  np.where(result["return"] < -boundary, 0, 1)).astype(float)
    result.loc[high.isna() | low.isna() | result["return"].isna() | boundary.isna(), ["up", "down", "terminal", "return"]] = np.nan
    return result


def fit_models(features, outcomes, indices):
    cols = feature_columns(features)
    x = features.loc[indices, cols].to_numpy(float)
    y = outcomes.loc[indices]
    common = dict(iterations=100, depth=4, learning_rate=.035, l2_leaf_reg=20,
                  thread_count=2, random_seed=1729, verbose=False, allow_writing_files=False)
    terminal = y.terminal.to_numpy(int)
    frequencies = (np.bincount(terminal, minlength=3)+1)/(len(y)+3)
    baseline = {"terminal": frequencies, "path": (y[["up", "down"]].sum().to_numpy()+1)/(len(y)+2),
                "quantiles": np.quantile(y["return"], [.1, .5, .9])}
    model = {"columns": cols, "baseline": baseline, "terminal": None, "path": None,
             "logistic": [], "quantiles": None}
    if len(np.unique(terminal)) > 1:
        model["terminal"] = CatBoostClassifier(loss_function="MultiClass", **common).fit(x, terminal)
    # Some rare-event windows contain a constant label. Keep the smoothed baseline for that output.
    varying = [c for c in ("up", "down") if y[c].nunique() > 1]
    model["path_columns"] = varying
    if varying:
        if len(varying) == 2:
            model["path"] = CatBoostClassifier(loss_function="MultiLogloss", **common).fit(x, y[varying])
        else:
            model["path"] = CatBoostClassifier(loss_function="Logloss", **common).fit(x, y[varying[0]])
    model["quantiles"] = CatBoostRegressor(loss_function="MultiQuantile:alpha=0.1,0.5,0.9", **common).fit(x, y["return"])
    for column in ("terminal", "up", "down"):
        target = y[column].to_numpy(int)
        clf = None
        if len(np.unique(target)) > 1:
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=400)).fit(x, target)
        model["logistic"].append(clf)
    return model


def predict_models(model, features):
    x = features[model["columns"]].to_numpy(float); n = len(x)
    base = {k: np.tile(v, (n, 1)) for k, v in model["baseline"].items()}
    cat = {k: v.copy() for k, v in base.items()}
    linear = {k: v.copy() for k, v in base.items()}
    if model["terminal"] is not None:
        cat["terminal"][:] = 0
        cat["terminal"][:, model["terminal"].classes_.astype(int)] = model["terminal"].predict_proba(x)
    if model["path"] is not None:
        path = np.asarray(model["path"].predict_proba(x))
        if len(model["path_columns"]) == 1:
            path = path[:, 1:2]
        for j, column in enumerate(model["path_columns"]):
            cat["path"][:, ["up", "down"].index(column)] = path[:, j]
    cat["quantiles"] = np.sort(model["quantiles"].predict(x), axis=1)
    for j, clf in enumerate(model["logistic"]):
        if clf is None:
            continue
        if j == 0:
            linear["terminal"][:] = 0
            linear["terminal"][:, clf.classes_.astype(int)] = clf.predict_proba(x)
        else:
            linear["path"][:, j-1] = clf.predict_proba(x)[:, 1]
    mixed = {k: .5*cat[k]+.5*base[k] for k in base}
    return {"climatology": base, "logistic": linear, "catboost": cat, "catboost_calibrated": mixed}


def brier(prediction, y):
    truth = np.eye(3)[y.terminal.to_numpy(int)]
    return float(((prediction["terminal"]-truth)**2).sum(axis=1).mean()/2 +
                 ((prediction["path"]-y[["up", "down"]].to_numpy())**2).mean())


def train_bundle(features, outcomes, cutoff, horizon):
    # All tasks in this horizon share its full label interval. No end-of-bar shortcut.
    cutoff = pd.Timestamp(cutoff)
    validation_days = 365 if horizon >= 336 else 90
    train_days = 1095 if horizon >= 336 else 730
    val_start = cutoff-pd.Timedelta(days=validation_days)
    ready = (features[feature_columns(features)].notna().all(axis=1) & outcomes["return"].notna())
    sampled = features.index.hour % 6 == 0
    train = features.index[ready & sampled & (features.index >= val_start-pd.Timedelta(days=train_days)) &
                           (outcomes.target_end < val_start-pd.Timedelta(hours=1))]
    val = features.index[ready & sampled & (features.index >= val_start) &
                         (outcomes.target_end < cutoff-pd.Timedelta(hours=1))]
    if len(train) < 700 or len(val) < 120:
        raise ValueError("insufficient purged training/validation history")
    inner = fit_models(features, outcomes, train)
    predictions = predict_models(inner, features.loc[val])
    scores = {name: brier(pred, outcomes.loc[val]) for name, pred in predictions.items()}
    choice = min(scores, key=lambda k: (scores[k], list(scores).index(k)))
    # Refit the fixed architecture on all currently matured data; validation has already selected the blend.
    outer = features.index[ready & sampled & (features.index >= cutoff-pd.Timedelta(days=train_days)) &
                           (outcomes.target_end < cutoff-pd.Timedelta(hours=1))]
    fitted = fit_models(features, outcomes, outer)
    # Alert thresholds fixed from prior validation negatives, never the replay/test month.
    selected = predictions[choice]
    thresholds = {}
    for j, event in enumerate(("up", "down")):
        negatives = selected["path"][outcomes.loc[val, event].eq(0).to_numpy(), j]
        thresholds[event] = float(np.quantile(negatives, .95, method="higher")) if len(negatives) >= 40 else 1.
    return {"model": fitted, "choice": choice, "validation_scores": scores,
            "validation_rows": len(val), "training_rows": len(outer),
            "training_target_end": outcomes.loc[outer, "target_end"].max().isoformat(),
            "validation_target_end": outcomes.loc[val, "target_end"].max().isoformat(),
            "fit_cutoff": cutoff.isoformat(), "alert_thresholds": thresholds}
