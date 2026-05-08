"""Shared infrastructure for long-running walk-forward OOF freezes.

Unlike the standard freeze scripts (3-fold CV, aggregate metrics only), these
long-run freezes iterate walk-forward folds **one at a time** and persist
per-date out-of-fold predictions to disk between folds. This makes the run:

  1. **Resumable** — a crashed/interrupted run picks up from the last
     successfully-flushed fold via `--resume`. Splits are purely a function
     of ``(n_samples, n_splits, test_size, gap, embargo)`` so fold N on a
     resumed run is byte-identical to fold N on a fresh run.

  2. **Introspectable** — per-date predictions accumulate in the checkpoint
     JSON so you can peek at partial results (first-year OOF at fold 12 /36
     for instance) without waiting for the full 3-year span to complete.

  3. **Visualisable** — the final JSON feeds directly into
     ``backtest_predictions`` in the site DB, which the frontend queries for
     per-day "predicted vs actual" track-record charts.

Leakage guarantees are identical to ``walk_forward_leaderboard`` /
``walk_forward_classification``:

  - Temporal split (train rows always precede test rows).
  - Purge gap = horizon (last ``horizon`` train rows dropped).
  - Embargo buffer.
  - Fold-internal feature selection.
  - Classification threshold picked on nested purged-temporal split.

Module layout:
  - ``FoldRunner``: orchestrates one fold across all regression + classification
    models; mirrors the inner loop of ``walk_forward_leaderboard`` exactly.
  - ``load_checkpoint`` / ``atomic_save_checkpoint``: resume I/O.
  - ``finalize_run``: after the last fold, rebuilds ensemble rows + leaderboard
    using the same efp helpers the standard freezes call.
"""
from __future__ import annotations

import datetime as _dt
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.base import clone

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402


EQUAL_WEIGHT_REGRESSION_MODEL = "trimmed_regression_ensemble_equal"
EQUAL_WEIGHT_CLASSIFICATION_MODEL = "trimmed_classification_ensemble_equal"

STATE_COLS = [
    "target_regime",
    "target_reversal_state",
    "target_bottom_reversal",
    "target_top_reversal",
]


def _dedupe_preserve(seq: Iterable[str]) -> list[str]:
    """Return ``seq`` with duplicates removed, preserving first occurrence.

    Several of the fold-runner utilities (``select_fold_features``, feature
    coverage) assume one-to-one between column-name and DataFrame column. A
    duplicated feature name in ``feature_columns`` causes ``frame[col]`` to
    return a DataFrame instead of a Series, which crashes in subtle places —
    so we sanitize the candidate list at the runner boundary.
    """
    seen: set[str] = set()
    out: list[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


# ---------------------------------------------------------------------------
# Checkpoint I/O — atomic write so a ctrl-C mid-flush never corrupts the JSON.
# ---------------------------------------------------------------------------
def atomic_save_checkpoint(path: Path, payload: dict) -> None:
    """Write JSON via temp file + os.replace() so interruption leaves the
    previous checkpoint intact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(tmp, path)


def load_checkpoint(path: Path) -> dict | None:
    """Return the previous payload or None if no checkpoint exists."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[longrun] checkpoint load failed ({exc}); starting fresh")
        return None


# ---------------------------------------------------------------------------
# Per-fold execution — mirrors walk_forward_leaderboard / walk_forward_
# classification inner loops so the output is numerically identical to running
# the all-fold version with the same (n_splits, test_size, gap, embargo).
# ---------------------------------------------------------------------------
class FoldRunner:
    """Runs one walk-forward fold across all regression + classification models.

    Holds internal OOF DataFrames so ``finalize_run`` can reconstruct the
    standard leaderboard + ensemble rows after the final fold without
    re-training anything.
    """

    def __init__(
        self,
        *,
        dataset: pd.DataFrame,
        feature_columns: list[str],
        sample_weights: pd.Series | None,
        horizon: int,
        n_splits: int,
        test_size: int,
        gap: int,
        embargo: int,
        min_feature_coverage: float = 0.03,
    ):
        self.dataset = dataset
        self.feature_columns = _dedupe_preserve(feature_columns)
        self.horizon = horizon
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.embargo = embargo
        self.min_feature_coverage = min_feature_coverage

        self.aligned_sample_weight = efp.align_sample_weight(
            sample_weights, dataset.index,
        )

        # Pre-materialize splits (deterministic given sizes).
        splitter = efp.purged_time_series_split(
            n_samples=len(dataset), n_splits=n_splits,
            test_size=test_size, gap=gap, embargo=embargo,
        )
        self.splits: list[tuple[np.ndarray, np.ndarray]] = [
            (np.asarray(tr), np.asarray(te))
            for tr, te in splitter.split(dataset)
        ]
        assert len(self.splits) == n_splits, (
            f"expected {n_splits} splits, got {len(self.splits)} — "
            f"likely n_samples={len(dataset)} too small for the requested "
            f"n_splits × test_size + gap"
        )

        # Internal OOF accumulators — one column per model.
        self.reg_oof = pd.DataFrame(index=dataset.index, dtype=float)
        self.cls_oof = pd.DataFrame(index=dataset.index, dtype=float)

        # Model templates cloned per fold.
        self._reg_models = efp.make_models(horizon=horizon)
        self._cls_models = efp.make_classification_models(horizon=horizon)

    # ------------------------------------------------------------------
    # Single-fold execution
    # ------------------------------------------------------------------
    def run_fold(self, fold_index: int) -> list[dict[str, Any]]:
        """Fit + predict all models on one fold; return per-date rows for DB.

        Also updates self.reg_oof / self.cls_oof so ``finalize_run`` can
        compute ensembles and leaderboard at the end without retraining.
        """
        train_idx, test_idx = self.splits[fold_index]
        rows: list[dict[str, Any]] = []

        # --- Regression ---
        y_return = self.dataset["target_return"]
        current_close = self.dataset["eth_close"]
        target_close = self.dataset["target_close"]

        for model_name, template in self._reg_models.items():
            fold_features = self._fold_features(train_idx)
            X_full = self.dataset[fold_features]
            X_train = X_full.iloc[train_idx]
            y_train = y_return.iloc[train_idx]
            X_test = X_full.iloc[test_idx]
            train_sw = (
                self.aligned_sample_weight.iloc[train_idx]
                if self.aligned_sample_weight is not None else None
            )

            model = clone(template)
            efp.fit_model_with_optional_sample_weight(
                model, X_train, y_train, train_sw,
            )
            pred_return = pd.Series(
                model.predict(X_test), index=X_test.index, dtype=float,
            )
            overlay_frame = self.dataset.iloc[test_idx]
            pred_return = efp.apply_regime_response_overlay(
                pred_return, overlay_frame, horizon=self.horizon,
            )

            # Store in internal OOF.
            self.reg_oof.loc[pred_return.index, f"{model_name}_pred_return"] = pred_return

            # Emit per-date rows for checkpoint.
            ref_close = current_close.iloc[test_idx]
            act_close = target_close.iloc[test_idx]
            for date, pr in pred_return.items():
                if not np.isfinite(pr):
                    continue
                ref = float(ref_close.loc[date]) if not pd.isna(ref_close.loc[date]) else None
                act = float(act_close.loc[date]) if not pd.isna(act_close.loc[date]) else None
                actual_return = (
                    (act - ref) / ref if (ref is not None and ref > 0 and act is not None) else None
                )
                rows.append({
                    "horizon_days":    self.horizon,
                    "head":            "regression",
                    "model":           model_name,
                    "prediction_date": date.strftime("%Y-%m-%d"),
                    "target_date":     (date + pd.Timedelta(days=self.horizon)).strftime("%Y-%m-%d"),
                    "fold_index":      int(fold_index),
                    "reference_close": ref,
                    "actual_close":    act,
                    "actual_return":   actual_return,
                    "actual_label":    int(actual_return > 0) if actual_return is not None else None,
                    "predicted_return": float(pr),
                    "predicted_close": float(ref * (1.0 + pr)) if ref is not None else None,
                })

            # Populate reg_oof close column too (matches efp convention).
            self.reg_oof.loc[pred_return.index, f"{model_name}_pred_close"] = (
                current_close.loc[pred_return.index] * (1.0 + pred_return)
            )

            del model
            gc.collect()

        # --- Classification ---
        y_cls = efp.get_direction_classification_target(self.dataset, self.horizon)

        for model_name, template in self._cls_models.items():
            train_positions = np.asarray(train_idx, dtype=int)
            prediction_positions = np.asarray(test_idx, dtype=int)
            train_positions = train_positions[y_cls.iloc[train_positions].notna().to_numpy()]
            if len(train_positions) == 0 or len(prediction_positions) == 0:
                continue
            y_train = y_cls.iloc[train_positions].astype(int)
            if y_train.nunique() < 2:
                continue

            fold_features = self._fold_features(
                train_positions,
                target_column=efp.direction_classification_target_column(self.dataset),
            )
            X_full = self.dataset[fold_features]
            train_sw = (
                self.aligned_sample_weight.iloc[train_positions]
                if self.aligned_sample_weight is not None else None
            )

            model = efp.fit_calibrated_classifier(
                template, X_full.iloc[train_positions], y_train, sample_weight=train_sw,
            )
            prob_up = pd.Series(
                model.predict_proba(X_full.iloc[prediction_positions])[:, 1],
                index=X_full.iloc[prediction_positions].index, dtype=float,
            )
            overlay_frame = self.dataset.iloc[prediction_positions]
            prob_up = efp.apply_direction_regime_overlay(
                prob_up, overlay_frame, horizon=self.horizon,
            )

            self.cls_oof.loc[prob_up.index, f"{model_name}_prob_up"] = prob_up

            ref_close = current_close.iloc[test_idx]
            act_close = target_close.iloc[test_idx]
            for date, prob in prob_up.items():
                if not np.isfinite(prob):
                    continue
                ref = float(ref_close.loc[date]) if not pd.isna(ref_close.loc[date]) else None
                act = float(act_close.loc[date]) if not pd.isna(act_close.loc[date]) else None
                actual_return = (
                    (act - ref) / ref if (ref is not None and ref > 0 and act is not None) else None
                )
                actual_target = y_cls.loc[date] if date in y_cls.index else np.nan
                rows.append({
                    "horizon_days":    self.horizon,
                    "head":            "classification",
                    "model":           model_name,
                    "prediction_date": date.strftime("%Y-%m-%d"),
                    "target_date":     (date + pd.Timedelta(days=self.horizon)).strftime("%Y-%m-%d"),
                    "fold_index":      int(fold_index),
                    "reference_close": ref,
                    "actual_close":    act,
                    "actual_return":   actual_return,
                    "actual_label":    int(actual_target) if pd.notna(actual_target) else None,
                    "probability_up":  float(prob),
                    # predicted_label filled in at finalize (needs threshold).
                    "predicted_label": None,
                })

            del model
            gc.collect()

        return rows

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fold_features(self, train_idx: np.ndarray, target_column: str = "target_return") -> list[str]:
        """Fold-internal feature selection — uses train rows only."""
        selected = efp.select_fold_features(
            dataset=self.dataset,
            candidate_feature_columns=self.feature_columns,
            train_positions=train_idx,
            min_feature_coverage=self.min_feature_coverage,
            horizon=self.horizon,
            target_column=target_column,
        )
        return selected or list(self.feature_columns)

    # ------------------------------------------------------------------
    # Resume: repopulate internal OOF from previously-saved prediction rows
    # so finalize_run can recompute ensembles as if the whole run was done
    # in a single pass.
    # ------------------------------------------------------------------
    def restore_oof_from_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            if row.get("horizon_days") != self.horizon:
                continue
            try:
                date = pd.Timestamp(row["prediction_date"])
            except (KeyError, ValueError):
                continue
            model = row.get("model")
            head = row.get("head")
            if not model or not head:
                continue
            if head == "regression":
                pr = row.get("predicted_return")
                if pr is not None:
                    self.reg_oof.at[date, f"{model}_pred_return"] = float(pr)
                pc = row.get("predicted_close")
                if pc is not None:
                    self.reg_oof.at[date, f"{model}_pred_close"] = float(pc)
            elif head == "classification":
                pu = row.get("probability_up")
                if pu is not None:
                    self.cls_oof.at[date, f"{model}_prob_up"] = float(pu)


# ---------------------------------------------------------------------------
# Post-fold finalization: leaderboard + ensemble rows + classification
# predicted_label (requires global threshold → only runs after all folds).
# ---------------------------------------------------------------------------
def finalize_run(
    runner: FoldRunner,
    predictions_by_horizon: dict[int, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compute leaderboard + ensembles for a single horizon; updates the
    predictions list in place with classification ``predicted_label``.

    Returns the horizon-level payload matching the existing freeze JSON
    shape (regression_leaderboard + classification_leaderboard + ensemble
    member lists).
    """
    horizon = runner.horizon
    dataset = runner.dataset

    # --- Regression leaderboard (identical to walk_forward_leaderboard post-loop) ---
    reg_rows: list[dict[str, float | str]] = []
    reg_models_with_oof = [
        c.replace("_pred_return", "")
        for c in runner.reg_oof.columns
        if c.endswith("_pred_return")
    ]
    for model_name in reg_models_with_oof:
        pred_col = runner.reg_oof[f"{model_name}_pred_return"].dropna()
        if pred_col.empty:
            continue
        matched = dataset.loc[pred_col.index]
        summary = efp.evaluate_predictions(
            current_close=matched["eth_close"].to_numpy(),
            actual_return=matched["target_return"].to_numpy(),
            predicted_return=pred_col.to_numpy(),
        )
        summary["model"] = model_name
        summary["folds"] = float(runner.n_splits)
        reg_rows.append(summary)
    reg_lb = pd.DataFrame(reg_rows).sort_values(
        ["price_rmse", "price_mae"], ascending=True,
    ).reset_index(drop=True) if reg_rows else pd.DataFrame()

    # --- Classification leaderboard (same pattern, but per-model threshold) ---
    cls_rows: list[dict[str, float | str]] = []
    cls_thresholds: dict[str, float] = {}
    cls_models_with_oof = [
        c.replace("_prob_up", "")
        for c in runner.cls_oof.columns
        if c.endswith("_prob_up")
    ]
    y_cls = efp.get_direction_classification_target(dataset, horizon)
    for model_name in cls_models_with_oof:
        prob_col = runner.cls_oof[f"{model_name}_prob_up"].dropna()
        if prob_col.empty:
            continue
        actual_target = y_cls.loc[prob_col.index]
        valid_evaluation = actual_target.notna() & prob_col.notna()
        if not valid_evaluation.any():
            continue
        actual = actual_target.loc[valid_evaluation].astype(int)
        evaluation_probability = prob_col.loc[valid_evaluation]
        threshold, metrics = efp.choose_classification_evaluation_threshold(
            actual_label=actual, probability_up=evaluation_probability, horizon=horizon,
        )
        metrics["model"] = model_name
        metrics["folds"] = float(runner.n_splits)
        metrics["signal_threshold"] = float(threshold)
        cls_rows.append(metrics)
        cls_thresholds[model_name] = float(threshold)

        # Fill predicted_label in internal OOF + in the flat predictions list.
        predicted_label = (runner.cls_oof[f"{model_name}_prob_up"] >= threshold).astype(float)
        predicted_label[runner.cls_oof[f"{model_name}_prob_up"].isna()] = np.nan
        runner.cls_oof[f"{model_name}_pred_label"] = predicted_label

    # Back-fill predicted_label into the flat row list so the DB gets it.
    rows_this_horizon = predictions_by_horizon.get(horizon, [])
    for row in rows_this_horizon:
        if row.get("head") != "classification":
            continue
        model = row.get("model")
        prob = row.get("probability_up")
        if model in cls_thresholds and prob is not None:
            row["predicted_label"] = int(float(prob) >= cls_thresholds[model])

    cls_lb = pd.DataFrame(cls_rows).sort_values(
        ["balanced_accuracy", "f1", "roc_auc", "signal_threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True) if cls_rows else pd.DataFrame()

    # --- Ensemble candidates (existing efp helpers, identical to freezes) ---
    reg_members = efp.select_trimmed_regression_ensemble_members(
        reg_lb, horizon=horizon, oof_predictions=runner.reg_oof,
    )
    cls_members = efp.select_trimmed_classification_ensemble_members(
        cls_lb, horizon=horizon, oof_predictions=runner.cls_oof,
    )
    reg_lb, runner.reg_oof = efp.append_trimmed_regression_ensemble_candidate(
        reg_lb, runner.reg_oof, dataset,
        horizon=horizon, component_models=reg_members,
    )
    cls_lb, runner.cls_oof = efp.append_trimmed_classification_ensemble_candidate(
        cls_lb, runner.cls_oof, dataset,
        horizon=horizon, component_models=cls_members,
    )

    # --- Equal-weight rows (same as existing freeze helpers) ---
    reg_lb = _append_equal_weight_regression_row(
        reg_lb, runner.reg_oof, dataset, reg_members,
    )
    cls_lb = _append_equal_weight_classification_row(
        cls_lb, runner.cls_oof, dataset, horizon, cls_members,
    )

    # --- Emit ensemble OOF rows as prediction rows too, so the DB has them ---
    _emit_ensemble_prediction_rows(
        runner=runner, ensemble_reg_model=EQUAL_WEIGHT_REGRESSION_MODEL,
        ensemble_cls_model=EQUAL_WEIGHT_CLASSIFICATION_MODEL,
        rows_sink=predictions_by_horizon[horizon],
    )

    return {
        "candidate_feature_count": len(runner.feature_columns),
        "training_rows": int(len(dataset)),
        "cv_test_size": runner.test_size,
        "embargo": runner.embargo,
        "n_splits": runner.n_splits,
        "regression_ensemble_members":     reg_members,
        "classification_ensemble_members": cls_members,
        "regression_leaderboard":     _records(reg_lb),
        "classification_leaderboard": _records(cls_lb),
    }


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    if df is None or df.empty:
        return []
    return df.to_dict(orient="records")


def _append_equal_weight_regression_row(leaderboard, oof, dataset, members):
    available = [m for m in members if f"{m}_pred_return" in oof.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof[[f"{m}_pred_return" for m in available]]
    )
    valid = blended.dropna()
    if valid.empty:
        return leaderboard
    matched = dataset.loc[valid.index]
    summary = efp.evaluate_predictions(
        current_close=matched["eth_close"].to_numpy(),
        actual_return=matched["target_return"].to_numpy(),
        predicted_return=valid.to_numpy(),
    )
    summary["model"] = EQUAL_WEIGHT_REGRESSION_MODEL
    summary["folds"] = float(min(len(available), 4))
    summary["component_models"] = "|".join(available)
    # Stash the blended series for later prediction-row emission.
    oof[f"{EQUAL_WEIGHT_REGRESSION_MODEL}_pred_return"] = blended
    oof[f"{EQUAL_WEIGHT_REGRESSION_MODEL}_pred_close"] = dataset["eth_close"] * (1.0 + blended)
    return pd.concat([leaderboard, pd.DataFrame([summary])], ignore_index=True)


def _append_equal_weight_classification_row(leaderboard, oof, dataset, horizon, members):
    available = [m for m in members if f"{m}_prob_up" in oof.columns]
    if len(available) < 2:
        return leaderboard
    blended = efp.trimmed_equal_weight_average(
        oof[[f"{m}_prob_up" for m in available]]
    ).clip(lower=0.0, upper=1.0)
    valid = blended.dropna()
    if valid.empty:
        return leaderboard
    actual_target = efp.get_direction_classification_target(dataset, horizon).loc[valid.index]
    valid_evaluation = actual_target.notna() & valid.notna()
    if not valid_evaluation.any():
        return leaderboard
    actual = actual_target.loc[valid_evaluation].astype(int)
    evaluation_probability = valid.loc[valid_evaluation]
    threshold, metrics = efp.choose_classification_evaluation_threshold(
        actual_label=actual, probability_up=evaluation_probability, horizon=horizon,
    )
    metrics["model"] = EQUAL_WEIGHT_CLASSIFICATION_MODEL
    metrics["folds"] = float(min(len(available), 4))
    metrics["signal_threshold"] = float(threshold)
    metrics["component_models"] = "|".join(available)
    # Stash the blended series + predicted_label for row emission.
    oof[f"{EQUAL_WEIGHT_CLASSIFICATION_MODEL}_prob_up"] = blended
    pred_label = (blended >= threshold).astype(float)
    pred_label[blended.isna()] = np.nan
    oof[f"{EQUAL_WEIGHT_CLASSIFICATION_MODEL}_pred_label"] = pred_label
    oof.attrs[f"{EQUAL_WEIGHT_CLASSIFICATION_MODEL}_threshold"] = float(threshold)
    return pd.concat([leaderboard, pd.DataFrame([metrics])], ignore_index=True)


def _emit_ensemble_prediction_rows(
    *, runner: FoldRunner,
    ensemble_reg_model: str, ensemble_cls_model: str,
    rows_sink: list[dict[str, Any]],
) -> None:
    """Materialise per-date rows for the ensemble pseudo-models so the DB
    track-record chart has an 'ensemble' line alongside individual models."""
    horizon = runner.horizon
    dataset = runner.dataset
    current_close = dataset["eth_close"]
    target_close = dataset["target_close"]
    direction_target = efp.get_direction_classification_target(dataset, horizon)

    # Find fold_index for each test date by scanning existing per-model rows.
    fold_for_date: dict[str, int] = {}
    for row in rows_sink:
        fold_for_date.setdefault(row["prediction_date"], int(row["fold_index"]))

    # Regression ensemble rows.
    reg_col = f"{ensemble_reg_model}_pred_return"
    if reg_col in runner.reg_oof.columns:
        for date, pr in runner.reg_oof[reg_col].dropna().items():
            if not np.isfinite(pr):
                continue
            date_str = date.strftime("%Y-%m-%d")
            ref = float(current_close.loc[date]) if not pd.isna(current_close.loc[date]) else None
            act = float(target_close.loc[date]) if not pd.isna(target_close.loc[date]) else None
            actual_return = (
                (act - ref) / ref if (ref is not None and ref > 0 and act is not None) else None
            )
            rows_sink.append({
                "horizon_days":     horizon,
                "head":             "regression",
                "model":            ensemble_reg_model,
                "prediction_date":  date_str,
                "target_date":      (date + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
                "fold_index":       fold_for_date.get(date_str, -1),
                "reference_close":  ref,
                "actual_close":     act,
                "actual_return":    actual_return,
                "actual_label":     int(actual_return > 0) if actual_return is not None else None,
                "predicted_return": float(pr),
                "predicted_close": float(ref * (1.0 + pr)) if ref is not None else None,
            })

    # Classification ensemble rows.
    prob_col = f"{ensemble_cls_model}_prob_up"
    label_col = f"{ensemble_cls_model}_pred_label"
    if prob_col in runner.cls_oof.columns:
        threshold = runner.cls_oof.attrs.get(f"{ensemble_cls_model}_threshold")
        for date, prob in runner.cls_oof[prob_col].dropna().items():
            if not np.isfinite(prob):
                continue
            date_str = date.strftime("%Y-%m-%d")
            ref = float(current_close.loc[date]) if not pd.isna(current_close.loc[date]) else None
            act = float(target_close.loc[date]) if not pd.isna(target_close.loc[date]) else None
            actual_return = (
                (act - ref) / ref if (ref is not None and ref > 0 and act is not None) else None
            )
            actual_target = direction_target.loc[date] if date in direction_target.index else np.nan
            pred_label = None
            if label_col in runner.cls_oof.columns and not pd.isna(runner.cls_oof.loc[date, label_col]):
                pred_label = int(runner.cls_oof.loc[date, label_col])
            elif threshold is not None:
                pred_label = int(float(prob) >= threshold)
            rows_sink.append({
                "horizon_days":    horizon,
                "head":            "classification",
                "model":           ensemble_cls_model,
                "prediction_date": date_str,
                "target_date":     (date + pd.Timedelta(days=horizon)).strftime("%Y-%m-%d"),
                "fold_index":      fold_for_date.get(date_str, -1),
                "reference_close": ref,
                "actual_close":    act,
                "actual_return":   actual_return,
                "actual_label":    int(actual_target) if pd.notna(actual_target) else None,
                "probability_up":  float(prob),
                "predicted_label": pred_label,
            })


# ---------------------------------------------------------------------------
# High-level driver — a thin wrapper the phase-specific scripts call so they
# only have to prepare (feature_frame, candidates, sample_weights, horizon).
# ---------------------------------------------------------------------------
def run_longrun(
    *,
    checkpoint_path: Path,
    horizon_payloads: dict[int, dict[str, Any]],
    run_metadata: dict[str, Any],
    resume: bool,
    flush_every: int = 1,
    max_folds: int | None = None,
    fold_start: int = 0,
) -> dict[str, Any]:
    """Execute per-fold walk-forward across multiple horizons with resume.

    ``horizon_payloads`` must map ``horizon -> dict`` with keys:
        dataset, feature_columns, sample_weights, n_splits, test_size, gap,
        embargo, min_feature_coverage, extras (any extra metadata to merge
        into the per-horizon output).
    """
    # Build (or restore) runners.
    runners: dict[int, FoldRunner] = {}
    for horizon, payload in horizon_payloads.items():
        runners[horizon] = FoldRunner(
            dataset=payload["dataset"],
            feature_columns=payload["feature_columns"],
            sample_weights=payload.get("sample_weights"),
            horizon=horizon,
            n_splits=payload["n_splits"],
            test_size=payload["test_size"],
            gap=payload["gap"],
            embargo=payload["embargo"],
            min_feature_coverage=payload.get("min_feature_coverage", 0.03),
        )

    # Load checkpoint (if any).
    state: dict[str, Any] = {
        "mode": run_metadata["mode"],
        "frozen_at": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "master_data_csv": run_metadata.get("master_data_csv"),
        "eth_price_forecast_bytes": Path(efp.__file__).stat().st_size,
        "cv_test_size": max(p["test_size"] for p in horizon_payloads.values()),
        "n_splits": max(p["n_splits"] for p in horizon_payloads.values()),
        "partial": True,
        "fold_start": int(max(fold_start, 0)),
        "max_folds_requested": max_folds,
        "folds_completed": {},  # per-horizon
        "folds_target":    {h: p["n_splits"] for h, p in horizon_payloads.items()},
        "horizons": {},          # per-horizon predictions + summary
        **{k: v for k, v in run_metadata.items() if k not in {"mode", "master_data_csv"}},
    }
    predictions_by_horizon: dict[int, list[dict[str, Any]]] = {
        h: [] for h in horizon_payloads
    }

    prev = load_checkpoint(checkpoint_path) if resume else None
    if prev is not None:
        print(f"[longrun] resuming from checkpoint: {checkpoint_path}")
        for h_key, h_payload in prev.get("horizons", {}).items():
            try:
                h = int(h_key)
            except (TypeError, ValueError):
                continue
            if h not in runners:
                continue
            preds = h_payload.get("predictions") or []
            predictions_by_horizon[h] = list(preds)
            runners[h].restore_oof_from_rows(preds)
            done = prev.get("folds_completed", {}).get(str(h)) or prev.get("folds_completed", {}).get(h) or 0
            state["folds_completed"][h] = int(done)
            print(f"  h={h}: restored {len(preds)} rows, folds_completed={done}")

    # Folds are per-horizon independent; iterate each horizon's folds sequentially.
    # For smoke runs, --max-folds is intentionally interpreted per horizon so
    # h=7 and h=30 both get coverage instead of h=7 consuming the whole budget.
    fold_budget_per_horizon = max_folds if max_folds is not None else 10**9
    stopped_by_budget = False

    for horizon, payload in horizon_payloads.items():
        runner = runners[horizon]
        start_fold = max(int(state["folds_completed"].get(horizon, 0)), int(max(fold_start, 0)))
        total_folds = runner.n_splits
        print(f"[longrun] horizon={horizon}: start_fold={start_fold}/{total_folds}")
        folds_used_this_horizon = 0
        for fold_idx in range(start_fold, total_folds):
            if folds_used_this_horizon >= fold_budget_per_horizon:
                print(f"[longrun] horizon={horizon}: hit --max-folds per-horizon budget={max_folds}")
                stopped_by_budget = True
                break
            t0 = time.time()
            rows = runner.run_fold(fold_idx)
            predictions_by_horizon[horizon].extend(rows)
            state["folds_completed"][horizon] = fold_idx + 1
            elapsed = time.time() - t0
            print(
                f"[longrun]   h={horizon} fold {fold_idx+1:>2}/{total_folds}: "
                f"{len(rows)} rows in {elapsed:.1f}s"
            )
            folds_used_this_horizon += 1
            if ((fold_idx + 1) % flush_every == 0) or (fold_idx + 1 == total_folds):
                _flush_checkpoint(
                    checkpoint_path, state, predictions_by_horizon,
                    horizon_payloads, partial=True,
                )

    # All folds done — finalize (leaderboard + ensembles + thresholds).
    completed_all = (not stopped_by_budget) and all(
        int(state["folds_completed"].get(horizon, 0)) >= runners[horizon].n_splits
        for horizon in horizon_payloads
    )
    return _finalize_and_save(
        checkpoint_path, state, runners, predictions_by_horizon,
        horizon_payloads, completed_all_horizons=completed_all,
    )


def _flush_checkpoint(
    checkpoint_path: Path,
    state: dict[str, Any],
    predictions_by_horizon: dict[int, list[dict[str, Any]]],
    horizon_payloads: dict[int, dict[str, Any]],
    *, partial: bool,
) -> None:
    state["last_checkpoint_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    state["partial"] = partial
    state["horizons"] = {
        str(h): {
            **horizon_payloads[h].get("extras", {}),
            "predictions": predictions_by_horizon[h],
        }
        for h in horizon_payloads
    }
    atomic_save_checkpoint(checkpoint_path, state)


def _finalize_and_save(
    checkpoint_path: Path,
    state: dict[str, Any],
    runners: dict[int, FoldRunner],
    predictions_by_horizon: dict[int, list[dict[str, Any]]],
    horizon_payloads: dict[int, dict[str, Any]],
    *, completed_all_horizons: bool,
) -> dict[str, Any]:
    state["last_checkpoint_utc"] = _dt.datetime.now(tz=_dt.timezone.utc).isoformat()
    if not completed_all_horizons:
        state["partial"] = True
        state["horizons"] = {}
        for horizon, runner in runners.items():
            rows = predictions_by_horizon.get(horizon, [])
            horizon_payload: dict[str, Any] = {
                **horizon_payloads[horizon].get("extras", {}),
            }
            if rows:
                horizon_payload.update(finalize_run(runner, predictions_by_horizon))
            horizon_payload["predictions"] = rows
            state["horizons"][str(horizon)] = horizon_payload
        atomic_save_checkpoint(checkpoint_path, state)
        return state
    # Fully done — compute leaderboards + ensembles + thresholds per horizon.
    state["partial"] = False
    state["horizons"] = {}
    for horizon, runner in runners.items():
        summary = finalize_run(runner, predictions_by_horizon)
        state["horizons"][str(horizon)] = {
            **horizon_payloads[horizon].get("extras", {}),
            **summary,
            "predictions": predictions_by_horizon[horizon],
        }
    atomic_save_checkpoint(checkpoint_path, state)
    return state
