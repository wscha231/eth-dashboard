#!/usr/bin/env python3
"""Exploratory, leakage-aware screen of newly collected exogenous ETH data.

This is intentionally NOT an operational model evaluator.  It uses a daily
endpoint target (`daily_hypothesis_screen_v1`) only to answer a narrower
research question: does a predeclared source family add information beyond a
fixed past-only ETH/BTC market core on the same historical origins?

Important boundaries
--------------------
* The production event target remains the hourly six-horizon target in
  ``signal_pipeline``.  Scores from this file are not directly comparable to
  the production replay and cannot promote a model.
* Historical vendor backfills are reconstructed research unless the source
  registry says otherwise.  A good historical score does not create an
  original receipt vintage.
* Full-day crypto/DeFi/volatility observations labelled D are shifted to D+1
  before they can enter a feature row.  ALFRED/FRED ``initial_asof`` is already
  indexed by its conservative eligible date and is not shifted again.
* Feature eligibility, median imputation and scaling are fitted inside each
  chronological fold.  There is no random split and no hyperparameter search.
* The five families, three horizons and model classes below are fixed in code;
  all results (including failures) are emitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_SPEC_ID = "daily_hypothesis_screen_v1"
HORIZONS = (7, 14, 30)
TEST_YEARS = (2023, 2024, 2025, 2026)
MIN_TRAIN_ROWS = 540
MIN_FAMILY_TRAIN_COVERAGE = 0.30
EPS = 1e-12
RANDOM_SEED = 20260916


@dataclass(frozen=True)
class FamilySpec:
    name: str
    files: tuple[str, ...]
    shift_days: int
    include_tokens: tuple[str, ...] = ()
    exclude_tokens: tuple[str, ...] = ()
    note: str = ""


FAMILIES = (
    FamilySpec(
        "macro_pit",
        ("fred_initial_asof_daily.csv",),
        0,
        note="ALFRED/FRED initial-release as-of state; already indexed by conservative eligible date",
    ),
    FamilySpec(
        "defi_liquidity",
        (
            "defillama_ethereum_chain_tvl_daily.csv",
            "defillama_ethereum_dex_volume_daily.csv",
            "defillama_ethereum_stablecoins_daily.csv",
        ),
        1,
        note="full-day Ethereum TVL/DEX/stablecoin observations; D becomes eligible D+1 UTC",
    ),
    FamilySpec(
        "dvol",
        ("deribit_eth_dvol_daily.csv",),
        1,
        include_tokens=("dvol",),
        note="Deribit DVOL daily history; reconstructed backfill shifted one day",
    ),
    FamilySpec(
        "bitget_derivatives",
        ("bitget_eth_free_features.csv",),
        1,
        include_tokens=("funding", "basis", "open_interest", "oi_", "volume24h", "turnover24h"),
        exclude_tokens=("mark_price", "index_price"),
        note="funding/basis/leverage context only; absolute mark/index price levels excluded",
    ),
    FamilySpec(
        "hyperliquid_derivatives",
        ("hyperliquid_eth_free_features.csv",),
        1,
        include_tokens=("funding", "premium", "open_interest", "oi_", "day_notional_volume"),
        exclude_tokens=("mark_price", "oracle_price", "mid_price"),
        note="funding/premium/leverage context only; absolute price levels excluded",
    ),
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _date_column(frame: pd.DataFrame) -> str:
    for candidate in ("date", "timestamp", "time", "Date"):
        if candidate in frame.columns:
            return candidate
    return str(frame.columns[0])


def load_daily(path: Path) -> pd.DataFrame:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty input: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"empty input: {path}")
    date_col = _date_column(frame)
    index = pd.to_datetime(frame.pop(date_col), utc=True, errors="coerce")
    frame.index = pd.DatetimeIndex(index)
    frame = frame.loc[frame.index.notna()]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    for column in frame.columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame.index.name = "date"
    return frame.replace([np.inf, -np.inf], np.nan)


def load_master(path: Path) -> pd.DataFrame:
    frame = load_daily(path)
    if "eth_close" not in frame.columns:
        raise ValueError("master data lacks eth_close")
    keep = ["eth_close"]
    if "btc_close" in frame.columns:
        keep.append("btc_close")
    out = frame[keep].copy()
    out["eth_close"] = pd.to_numeric(out["eth_close"], errors="coerce")
    if (out["eth_close"].dropna() <= 0).any():
        raise ValueError("ETH close must be positive")
    return out


def select_family_columns(columns: Iterable[str], spec: FamilySpec) -> list[str]:
    chosen: list[str] = []
    for raw in columns:
        column = str(raw)
        lowered = column.lower()
        if spec.include_tokens and not any(token in lowered for token in spec.include_tokens):
            continue
        if spec.exclude_tokens and any(token in lowered for token in spec.exclude_tokens):
            # Keep a basis column even if the source happened to include a price token.
            if "basis" not in lowered:
                continue
        chosen.append(column)
    return chosen


def load_family(root: Path, spec: FamilySpec) -> tuple[pd.DataFrame, pd.Series, dict]:
    blocks: list[pd.DataFrame] = []
    manifest_files: list[dict] = []
    for relative in spec.files:
        path = root / relative
        block = load_daily(path)
        manifest_files.append(
            {
                "file": relative,
                "sha256": sha256(path),
                "rows": int(len(block)),
                "start": block.index.min().isoformat(),
                "end": block.index.max().isoformat(),
            }
        )
        blocks.append(block)
    frame = pd.concat(blocks, axis=1).sort_index()
    frame = frame.loc[:, ~frame.columns.duplicated(keep="last")]
    columns = select_family_columns(frame.columns, spec)
    if not columns:
        raise ValueError(f"family {spec.name} has no selected columns")
    raw = frame[columns].copy()
    if spec.shift_days:
        raw.index = raw.index + pd.Timedelta(days=spec.shift_days)
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    present = raw.notna().any(axis=1).rename(f"{spec.name}__present")
    return raw, present, {
        "family": spec.name,
        "shift_days": spec.shift_days,
        "note": spec.note,
        "selected_raw_columns": columns,
        "files": manifest_files,
    }


def rolling_z(series: pd.Series, window: int, minimum: int) -> pd.Series:
    mean = series.rolling(window, min_periods=minimum).mean()
    std = series.rolling(window, min_periods=minimum).std().replace(0.0, np.nan)
    return (series - mean) / std


def build_family_features(raw: pd.DataFrame, family: str) -> pd.DataFrame:
    blocks: dict[str, pd.Series] = {}
    for column in raw.columns:
        series = pd.to_numeric(raw[column], errors="coerce")
        base = f"{family}__{column}"
        blocks[f"{base}__level"] = series
        # Age is already a deliberately engineered state variable.  Re-transforming
        # it into percent/z-score features adds no useful semantics.
        if str(column).endswith("__age_days"):
            continue
        blocks[f"{base}__diff1"] = series.diff(1)
        blocks[f"{base}__diff7"] = series.diff(7)
        denom7 = series.shift(7).abs().replace(0.0, np.nan)
        denom30 = series.shift(30).abs().replace(0.0, np.nan)
        blocks[f"{base}__rel7"] = (series.diff(7) / denom7).clip(-10, 10)
        blocks[f"{base}__rel30"] = (series.diff(30) / denom30).clip(-10, 10)
        blocks[f"{base}__z30"] = rolling_z(series, 30, 15)
        blocks[f"{base}__z90"] = rolling_z(series, 90, 45)
    return pd.DataFrame(blocks, index=raw.index).replace([np.inf, -np.inf], np.nan)


def build_core(master: pd.DataFrame) -> pd.DataFrame:
    eth = master["eth_close"].astype(float)
    log_eth = np.log(eth)
    ret = log_eth.diff()
    features: dict[str, pd.Series] = {}
    for lag in (1, 3, 7, 14, 30, 60, 90):
        features[f"core__eth_ret_{lag}"] = log_eth.diff(lag)
    for window in (7, 14, 30, 90):
        features[f"core__eth_vol_{window}"] = ret.rolling(window, min_periods=max(4, window // 2)).std()
        features[f"core__eth_rv_{window}"] = ret.pow(2).rolling(window, min_periods=max(4, window // 2)).sum()
        features[f"core__eth_ma_ratio_{window}"] = eth / eth.rolling(window, min_periods=max(4, window // 2)).mean() - 1.0
    for window in (30, 90):
        features[f"core__eth_drawdown_{window}"] = eth / eth.rolling(window, min_periods=window // 2).max() - 1.0
    if "btc_close" in master.columns:
        btc = master["btc_close"].astype(float).where(lambda x: x > 0)
        log_btc = np.log(btc)
        for lag in (1, 7, 30, 90):
            features[f"core__btc_ret_{lag}"] = log_btc.diff(lag)
            features[f"core__eth_btc_strength_{lag}"] = log_eth.diff(lag) - log_btc.diff(lag)
    return pd.DataFrame(features, index=master.index).replace([np.inf, -np.inf], np.nan)


def build_targets(master: pd.DataFrame, horizon: int) -> pd.DataFrame:
    eth = master["eth_close"].astype(float)
    log_eth = np.log(eth)
    daily_ret = log_eth.diff()
    target = pd.DataFrame(index=master.index)
    target["return"] = log_eth.shift(-horizon) - log_eth
    target["direction"] = (target["return"] > 0).astype(float)
    target.loc[target["return"].isna(), "direction"] = np.nan
    future_rv = pd.Series(0.0, index=master.index)
    valid = pd.Series(True, index=master.index)
    for step in range(1, horizon + 1):
        leg = daily_ret.shift(-step)
        future_rv = future_rv + leg.pow(2).fillna(0.0)
        valid &= leg.notna()
    target["realized_variance"] = future_rv.where(valid)
    target["variance_persistence"] = daily_ret.pow(2).rolling(
        horizon, min_periods=max(4, horizon // 2)
    ).sum()
    target["target_end"] = target.index + pd.Timedelta(days=horizon)
    return target


def make_ridge() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ]
    )


def make_hist() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            (
                "model",
                HistGradientBoostingRegressor(
                    learning_rate=0.04,
                    max_iter=180,
                    max_leaf_nodes=15,
                    min_samples_leaf=30,
                    l2_regularization=10.0,
                    random_state=RANDOM_SEED,
                ),
            ),
        ]
    )


def make_logistic() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=0.1, max_iter=500, random_state=RANDOM_SEED)),
        ]
    )


def eligible_columns(frame: pd.DataFrame, train_index: pd.DatetimeIndex, prefix: str) -> list[str]:
    candidates = [c for c in frame.columns if str(c).startswith(prefix)]
    if not candidates:
        return []
    coverage = frame.loc[train_index, candidates].notna().mean(axis=0)
    variance = frame.loc[train_index, candidates].nunique(dropna=True)
    return [
        str(c)
        for c in candidates
        if float(coverage[c]) >= MIN_FAMILY_TRAIN_COVERAGE and int(variance[c]) > 1
    ]


def mae(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.abs(y - p)))


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y - p))))


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(y - p)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(y)
    if total == 0:
        return float("nan")
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi == 1.0:
            mask = (p >= lo) & (p <= hi)
        else:
            mask = (p >= lo) & (p < hi)
        if not mask.any():
            continue
        value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return float(value)


def qlike(y: np.ndarray, forecast: np.ndarray) -> float:
    y = np.maximum(np.asarray(y, dtype=float), EPS)
    forecast = np.maximum(np.asarray(forecast, dtype=float), EPS)
    ratio = y / forecast
    return float(np.mean(ratio - np.log(ratio) - 1.0))


def skill(loss: float, baseline: float) -> float:
    if not np.isfinite(loss) or not np.isfinite(baseline) or baseline <= 0:
        return float("nan")
    return float(1.0 - loss / baseline)


def select_nonoverlap(index: pd.DatetimeIndex, horizon: int) -> pd.DatetimeIndex:
    chosen: list[pd.Timestamp] = []
    next_allowed: pd.Timestamp | None = None
    for timestamp in pd.DatetimeIndex(index).sort_values():
        if next_allowed is None or timestamp >= next_allowed:
            chosen.append(timestamp)
            next_allowed = timestamp + pd.Timedelta(days=horizon)
    return pd.DatetimeIndex(chosen)


def fit_point_pair(
    model_factory,
    frame: pd.DataFrame,
    core_cols: list[str],
    family_cols: list[str],
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    core_model = model_factory()
    ext_model = model_factory()
    core_model.fit(frame.loc[train, core_cols], y.loc[train])
    ext_model.fit(frame.loc[train, core_cols + family_cols], y.loc[train])
    return (
        np.asarray(core_model.predict(frame.loc[test, core_cols]), dtype=float),
        np.asarray(ext_model.predict(frame.loc[test, core_cols + family_cols]), dtype=float),
    )


def fit_direction_pair(
    frame: pd.DataFrame,
    core_cols: list[str],
    family_cols: list[str],
    train: pd.DatetimeIndex,
    test: pd.DatetimeIndex,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray, float]:
    train_y = y.loc[train].astype(int)
    prior = float(train_y.mean())
    if train_y.nunique() < 2:
        return np.full(len(test), prior), np.full(len(test), prior), prior
    core_model = make_logistic()
    ext_model = make_logistic()
    core_model.fit(frame.loc[train, core_cols], train_y)
    ext_model.fit(frame.loc[train, core_cols + family_cols], train_y)
    return (
        core_model.predict_proba(frame.loc[test, core_cols])[:, 1],
        ext_model.predict_proba(frame.loc[test, core_cols + family_cols])[:, 1],
        prior,
    )


def evaluate_family(
    master: pd.DataFrame,
    core: pd.DataFrame,
    family_raw: pd.DataFrame,
    family_features: pd.DataFrame,
    family_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = core.join(family_features, how="left")
    present = family_raw.notna().any(axis=1).reindex(frame.index).fillna(False)
    rows: list[dict] = []
    predictions: list[dict] = []

    for horizon in HORIZONS:
        target = build_targets(master, horizon).reindex(frame.index)
        for year in TEST_YEARS:
            test_start = pd.Timestamp(f"{year}-01-01", tz="UTC")
            test_end = pd.Timestamp(f"{year + 1}-01-01", tz="UTC")
            matured_before = test_start - pd.Timedelta(days=1)
            usable = target["return"].notna() & present
            train_mask = (
                usable
                & (frame.index < test_start)
                & (target["target_end"] < matured_before)
            )
            test_mask = usable & (frame.index >= test_start) & (frame.index < test_end)
            train = pd.DatetimeIndex(frame.index[train_mask])
            test = pd.DatetimeIndex(frame.index[test_mask])
            if len(train) < MIN_TRAIN_ROWS or len(test) < 20:
                continue

            core_cols = eligible_columns(frame, train, "core__")
            family_cols = eligible_columns(frame, train, f"{family_name}__")
            if not core_cols or not family_cols:
                continue

            y_ret_train = target.loc[train, "return"]
            y_ret_test = target.loc[test, "return"].to_numpy(float)
            fold_prediction: dict[str, np.ndarray] = {}
            for model_name, factory in (("ridge", make_ridge), ("hist", make_hist)):
                core_pred, ext_pred = fit_point_pair(
                    factory, frame, core_cols, family_cols, train, test, target["return"]
                )
                fold_prediction[f"point_core_{model_name}"] = core_pred
                fold_prediction[f"point_ext_{model_name}"] = ext_pred
                base_pred = np.zeros(len(test), dtype=float)
                base_mae = mae(y_ret_test, base_pred)
                core_mae = mae(y_ret_test, core_pred)
                ext_mae = mae(y_ret_test, ext_pred)
                rows.append(
                    {
                        "family": family_name,
                        "horizon_days": horizon,
                        "year": year,
                        "task": "point",
                        "model": model_name,
                        "n": len(test),
                        "nonoverlap_n": len(select_nonoverlap(test, horizon)),
                        "family_feature_count": len(family_cols),
                        "baseline_loss": base_mae,
                        "core_loss": core_mae,
                        "extended_loss": ext_mae,
                        "skill_vs_naive": skill(ext_mae, base_mae),
                        "incremental_vs_core": skill(ext_mae, core_mae),
                        "secondary_baseline": rmse(y_ret_test, base_pred),
                        "secondary_core": rmse(y_ret_test, core_pred),
                        "secondary_extended": rmse(y_ret_test, ext_pred),
                        "metric": "MAE",
                        "secondary_metric": "RMSE",
                    }
                )

            core_prob, ext_prob, prior = fit_direction_pair(
                frame, core_cols, family_cols, train, test, target["direction"]
            )
            direction = target.loc[test, "direction"].to_numpy(int)
            prior_prob = np.full(len(test), prior, dtype=float)
            prior_brier = brier(direction, prior_prob)
            core_brier = brier(direction, core_prob)
            ext_brier = brier(direction, ext_prob)
            rows.append(
                {
                    "family": family_name,
                    "horizon_days": horizon,
                    "year": year,
                    "task": "direction",
                    "model": "logistic",
                    "n": len(test),
                    "nonoverlap_n": len(select_nonoverlap(test, horizon)),
                    "family_feature_count": len(family_cols),
                    "baseline_loss": prior_brier,
                    "core_loss": core_brier,
                    "extended_loss": ext_brier,
                    "skill_vs_naive": skill(ext_brier, prior_brier),
                    "incremental_vs_core": skill(ext_brier, core_brier),
                    "secondary_baseline": float("nan"),
                    "secondary_core": ece(direction, core_prob),
                    "secondary_extended": ece(direction, ext_prob),
                    "metric": "Brier",
                    "secondary_metric": "ECE",
                }
            )
            fold_prediction["direction_core"] = core_prob
            fold_prediction["direction_ext"] = ext_prob

            rv_train_mask = target.loc[train, "realized_variance"].notna()
            rv_test_mask = target.loc[test, "realized_variance"].notna() & target.loc[test, "variance_persistence"].notna()
            rv_train = pd.DatetimeIndex(train[rv_train_mask.to_numpy()])
            rv_test = pd.DatetimeIndex(test[rv_test_mask.to_numpy()])
            if len(rv_train) >= MIN_TRAIN_ROWS and len(rv_test) >= 20:
                rv_target = np.log(target["realized_variance"].clip(lower=EPS))
                core_rv_log, ext_rv_log = fit_point_pair(
                    make_ridge, frame, core_cols, family_cols, rv_train, rv_test, rv_target
                )
                core_rv = np.exp(np.clip(core_rv_log, -30, 10))
                ext_rv = np.exp(np.clip(ext_rv_log, -30, 10))
                actual_rv = target.loc[rv_test, "realized_variance"].to_numpy(float)
                persistence = target.loc[rv_test, "variance_persistence"].to_numpy(float)
                base_q = qlike(actual_rv, persistence)
                core_q = qlike(actual_rv, core_rv)
                ext_q = qlike(actual_rv, ext_rv)
                rows.append(
                    {
                        "family": family_name,
                        "horizon_days": horizon,
                        "year": year,
                        "task": "variance",
                        "model": "ridge_log_rv",
                        "n": len(rv_test),
                        "nonoverlap_n": len(select_nonoverlap(rv_test, horizon)),
                        "family_feature_count": len(family_cols),
                        "baseline_loss": base_q,
                        "core_loss": core_q,
                        "extended_loss": ext_q,
                        "skill_vs_naive": skill(ext_q, base_q),
                        "incremental_vs_core": skill(ext_q, core_q),
                        "secondary_baseline": float("nan"),
                        "secondary_core": float("nan"),
                        "secondary_extended": float("nan"),
                        "metric": "QLIKE",
                        "secondary_metric": "",
                    }
                )

            for offset, timestamp in enumerate(test):
                record = {
                    "family": family_name,
                    "horizon_days": horizon,
                    "year": year,
                    "origin": timestamp.isoformat(),
                    "actual_return": y_ret_test[offset],
                    "actual_direction": int(direction[offset]),
                    "prior_probability_up": prior,
                    "core_probability_up": core_prob[offset],
                    "extended_probability_up": ext_prob[offset],
                }
                for key, values in fold_prediction.items():
                    if len(values) == len(test):
                        record[key] = float(values[offset])
                predictions.append(record)

    return pd.DataFrame(rows), pd.DataFrame(predictions)


def aggregate_fold_metrics(folds: pd.DataFrame) -> pd.DataFrame:
    if folds.empty:
        return pd.DataFrame()
    grouped_rows: list[dict] = []
    keys = ["family", "horizon_days", "task", "model", "metric", "secondary_metric"]
    for key, group in folds.groupby(keys, dropna=False):
        weights = group["n"].to_numpy(float)
        if weights.sum() <= 0:
            continue
        def weighted(column: str) -> float:
            values = group[column].to_numpy(float)
            valid = np.isfinite(values)
            if not valid.any():
                return float("nan")
            return float(np.average(values[valid], weights=weights[valid]))
        baseline = weighted("baseline_loss")
        core = weighted("core_loss")
        extended = weighted("extended_loss")
        row = dict(zip(keys, key))
        row.update(
            {
                "folds": int(len(group)),
                "years": ",".join(str(int(v)) for v in group["year"].tolist()),
                "n": int(group["n"].sum()),
                "nonoverlap_n": int(group["nonoverlap_n"].sum()),
                "family_feature_count_min": int(group["family_feature_count"].min()),
                "family_feature_count_max": int(group["family_feature_count"].max()),
                "baseline_loss": baseline,
                "core_loss": core,
                "extended_loss": extended,
                "skill_vs_naive": skill(extended, baseline),
                "incremental_vs_core": skill(extended, core),
                "improved_vs_naive_folds": int((group["extended_loss"] < group["baseline_loss"]).sum()),
                "improved_vs_core_folds": int((group["extended_loss"] < group["core_loss"]).sum()),
                "secondary_baseline": weighted("secondary_baseline"),
                "secondary_core": weighted("secondary_core"),
                "secondary_extended": weighted("secondary_extended"),
            }
        )
        if row["task"] == "point":
            row["exploratory_signal"] = bool(
                row["skill_vs_naive"] > 0
                and row["incremental_vs_core"] > 0
                and row["improved_vs_core_folds"] >= max(1, math.ceil(row["folds"] / 2))
            )
        elif row["task"] == "variance":
            row["exploratory_signal"] = bool(
                row["skill_vs_naive"] > 0 and row["incremental_vs_core"] > 0
            )
        else:
            row["exploratory_signal"] = bool(
                row["skill_vs_naive"] > 0
                and row["incremental_vs_core"] > 0
                and np.isfinite(row["secondary_extended"])
                and row["secondary_extended"] <= 0.05
            )
        grouped_rows.append(row)
    return pd.DataFrame(grouped_rows).sort_values(
        ["task", "horizon_days", "incremental_vs_core"], ascending=[True, True, False]
    )


def markdown_report(summary: pd.DataFrame, manifests: list[dict], master_manifest: dict) -> str:
    lines = [
        "# Exogenous source-family screen — 2026-09-16",
        "",
        "**Status: exploratory hypothesis screen only; no model promotion.**",
        "",
        f"Target spec: `{TARGET_SPEC_ID}`. Daily endpoint returns at 7/14/30 days are used only to rank source-family hypotheses. This is not the operational hourly event target.",
        "",
        "## Leakage / evidence boundaries",
        "",
        "- Full-day crypto/DeFi/DVOL observations are shifted to the next UTC day before feature construction.",
        "- ALFRED initial-release as-of data are consumed on their already-conservative eligible date.",
        "- Training uses only rows whose target ends before the next test block plus a one-day purge.",
        "- Feature coverage, imputation and scaling are fitted from each training block only.",
        "- Historical backfills remain reconstructed research; no result here is prospective evidence.",
        "- Fixed search count: 5 families × 3 horizons × (2 point models + 1 direction model + 1 variance model) = 60 family/horizon/task-model comparisons. Multiple-testing risk is therefore material.",
        "",
        "## Inputs",
        "",
        f"- Master: `{master_manifest['file']}` SHA256 `{master_manifest['sha256']}`; {master_manifest['rows']} rows.",
    ]
    for manifest in manifests:
        lines.append(
            f"- `{manifest['family']}`: shift_days={manifest['shift_days']}; raw columns={len(manifest['selected_raw_columns'])}; {manifest['note']}"
        )
    lines.extend(["", "## Aggregate results", ""])
    if summary.empty:
        lines.append("No family had enough matched chronological history for the frozen screen.")
        return "\n".join(lines) + "\n"
    display = summary.copy()
    percent_cols = ["skill_vs_naive", "incremental_vs_core"]
    for col in percent_cols:
        display[col] = display[col].map(lambda x: f"{100*x:+.2f}%" if np.isfinite(x) else "NA")
    cols = [
        "family", "horizon_days", "task", "model", "folds", "n", "nonoverlap_n",
        "skill_vs_naive", "incremental_vs_core", "improved_vs_core_folds", "exploratory_signal",
    ]
    lines.append(display[cols].to_markdown(index=False))
    lines.extend(
        [
            "",
            "`exploratory_signal=true` only means the fixed development screen was positive on its own definition. It does not satisfy issue #56, does not create an untouched holdout, and cannot authorize shadow or production use.",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.input_root)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    master_path = root / "eth_master_daily.csv"
    master = load_master(master_path)
    core = build_core(master)
    master_manifest = {
        "file": "eth_master_daily.csv",
        "sha256": sha256(master_path),
        "rows": int(len(master)),
        "start": master.index.min().isoformat(),
        "end": master.index.max().isoformat(),
    }

    all_folds: list[pd.DataFrame] = []
    all_predictions: list[pd.DataFrame] = []
    manifests: list[dict] = []
    for spec in FAMILIES:
        raw, _, manifest = load_family(root, spec)
        family_features = build_family_features(raw, spec.name)
        folds, predictions = evaluate_family(master, core, raw, family_features, spec.name)
        manifests.append(manifest)
        if not folds.empty:
            all_folds.append(folds)
        if not predictions.empty:
            all_predictions.append(predictions)

    fold_metrics = pd.concat(all_folds, ignore_index=True) if all_folds else pd.DataFrame()
    predictions = pd.concat(all_predictions, ignore_index=True) if all_predictions else pd.DataFrame()
    summary = aggregate_fold_metrics(fold_metrics)

    fold_metrics.to_csv(output / "fold_metrics.csv", index=False)
    summary.to_csv(output / "summary.csv", index=False)
    if not predictions.empty:
        predictions.to_csv(output / "predictions.csv.gz", index=False, compression="gzip")
    manifest_payload = {
        "schema": 1,
        "target_spec_id": TARGET_SPEC_ID,
        "generated_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "github_sha": os.getenv("GITHUB_SHA", ""),
        "master": master_manifest,
        "families": manifests,
        "horizons_days": list(HORIZONS),
        "test_years": list(TEST_YEARS),
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_family_train_coverage": MIN_FAMILY_TRAIN_COVERAGE,
        "search_count": 60,
        "promotion_authorized": False,
        "evidence_class": "reconstructed_development_hypothesis_screen",
    }
    (output / "input_manifest.json").write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = markdown_report(summary, manifests, master_manifest)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
