"""Phase 0 leakage audit for eth_price_forecast.py.

Findings (as of audit run):

PASS — Verified causal / no-leak:
  1. sklearn Pipeline wraps imputer + scaler + model (lines 2806-3022), so
     imputation and scaling are fit on the TRAIN fold only.
  2. TimeSeriesSplit uses gap=horizon (line 9372), preventing overlap between
     the last train target window and the first test row.
  3. Technical indicators use .rolling / .ewm / .shift(positive) exclusively.
     No bfill found anywhere in the module.
  4. Target columns use shift(-horizon) correctly (line 2706-2707).
  5. State-target labels (build_state_targets, line 2302) use future windows
     only for labels, not for inputs to features.
  6. prediction_history features apply ffill().shift(1) (line 1628), so each
     row only sees predictions strictly prior to it.

FLAG — Potential point-in-time / selection leakage:
  A. finalize_market_data.ffill() at line 784 is applied ONCE on the entire
     merged dataset. Vendor revisions (e.g., FRED CPI revisions) are baked in.
     Severity: low-to-medium.
  B. stale_vendor_columns nullification at line 2393-2396 rewrites history based
     on the LATEST observed staleness state. Historical rows appear as if the
     vendor was already known-stale. Severity: low (biases downward, not up).
  C. rank_feature_candidates (line 1827-1857) computes Spearman correlation
     between each candidate feature and the target over the FULL training mask,
     which includes rows that will later land in the test folds. The top-K
     features are then selected. This is feature-selection leakage.
     Severity: MEDIUM — this is the highest-impact finding.
  D. prune_feature_candidates (line 1888) computes the Spearman correlation
     matrix for de-duplication over the same full training mask. Same severity
     as (C).

Run with:  pytest tests/phase0/test_leakage_audit.py -v
"""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eth_price_forecast as efp


SOURCE_PATH = Path(efp.__file__)
SOURCE_TEXT = SOURCE_PATH.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Static checks — assertions about the source code itself.
# ---------------------------------------------------------------------------


def test_no_bfill_anywhere() -> None:
    """bfill backfills future values into the past — it must never appear."""
    assert ".bfill(" not in SOURCE_TEXT, (
        "Found .bfill( in eth_price_forecast.py — backfilling is leakage."
    )


def test_sklearn_pipelines_keep_imputer_and_scaler_together() -> None:
    """Scaling-sensitive models must have (imputer, scaler, model) in that order."""
    models = efp.make_models(horizon=7)
    scaling_sensitive = {"ridge", "knn_regressor", "mlp_regressor"}
    for name in scaling_sensitive & set(models.keys()):
        steps = [step_name for step_name, _ in models[name].steps]
        assert steps[0] == "imputer", f"{name}: first step must be imputer, got {steps}"
        assert "scaler" in steps, f"{name}: scaler missing, got {steps}"


def test_walk_forward_uses_time_series_split_with_gap() -> None:
    """walk_forward_leaderboard must pass gap=horizon to the purged splitter."""
    source = inspect.getsource(efp.walk_forward_leaderboard)
    assert "purged_time_series_split" in source, (
        "walk_forward_leaderboard must use purged_time_series_split (gap+embargo-aware)"
    )
    assert "gap=gap" in source, "walk_forward_leaderboard must forward the gap parameter"


def test_target_uses_negative_shift_only_for_targets() -> None:
    """shift(-...) (future lookahead) may only appear inside target/state label construction."""
    tree = ast.parse(SOURCE_TEXT)
    allowed_funcs = {
        "build_state_targets",
        "compute_future_window_extrema",
        "build_features",
    }
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name in allowed_funcs:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            if not (isinstance(inner.func, ast.Attribute) and inner.func.attr == "shift"):
                continue
            for arg in inner.args:
                if isinstance(arg, ast.UnaryOp) and isinstance(arg.op, ast.USub):
                    violations.append(f"{node.name}:line {inner.lineno}")
                elif isinstance(arg, ast.Constant) and isinstance(arg.value, int) and arg.value < 0:
                    violations.append(f"{node.name}:line {inner.lineno}")
    assert not violations, f"Unexpected negative shifts outside target construction: {violations}"


# ---------------------------------------------------------------------------
# Dynamic checks — perturbation / future-invariance tests on build_features.
# ---------------------------------------------------------------------------


def test_feature_does_not_see_future_rows(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    """Perturb data at times > k and assert that NON-TARGET features at time k do not change."""
    horizon = 7
    baseline_frame, feature_cols = efp.build_features(synthetic_ohlcv_with_companions, horizon=horizon)

    perturbed_input = synthetic_ohlcv_with_companions.copy()
    k = 600
    perturbed_input.iloc[k + 1 :] = perturbed_input.iloc[k + 1 :] * 5.0

    perturbed_frame, _ = efp.build_features(perturbed_input, horizon=horizon)

    leaks: list[tuple[str, float, float]] = []
    target_cols = {"target_return", "target_close", "target_regime",
                   "target_reversal_state", "target_bottom_reversal", "target_top_reversal"}
    for column in feature_cols:
        if column in target_cols:
            continue
        base_val = baseline_frame[column].iloc[k]
        pert_val = perturbed_frame[column].iloc[k]
        if pd.isna(base_val) and pd.isna(pert_val):
            continue
        if pd.notna(base_val) and pd.notna(pert_val) and np.isclose(base_val, pert_val, rtol=1e-9, atol=1e-12):
            continue
        leaks.append((column, float(base_val) if pd.notna(base_val) else np.nan,
                      float(pert_val) if pd.notna(pert_val) else np.nan))

    assert not leaks, (
        f"{len(leaks)} feature columns changed when FUTURE (t>{k}) data was perturbed. "
        f"This is data leakage. First offenders: {leaks[:10]}"
    )


def test_targets_depend_on_future_as_expected(synthetic_ohlcv_with_companions: pd.DataFrame) -> None:
    """Sanity check: target_return MUST change when future data is perturbed."""
    horizon = 7
    baseline_frame, _ = efp.build_features(synthetic_ohlcv_with_companions, horizon=horizon)

    perturbed_input = synthetic_ohlcv_with_companions.copy()
    k = 600
    perturbed_input.iloc[k + 1 : k + 1 + horizon] = perturbed_input.iloc[k + 1 : k + 1 + horizon] * 2.0
    perturbed_frame, _ = efp.build_features(perturbed_input, horizon=horizon)

    base_target = baseline_frame["target_return"].iloc[k]
    pert_target = perturbed_frame["target_return"].iloc[k]
    assert pd.notna(base_target) and pd.notna(pert_target)
    assert not np.isclose(base_target, pert_target), (
        "target_return at row k did NOT change when future was perturbed — "
        "target is not looking into the future, which is a bug in the labeling logic."
    )


# ---------------------------------------------------------------------------
# Known-leakage markers — these xfail today and become the Phase-1 checklist.
# ---------------------------------------------------------------------------


def test_feature_selection_is_computed_inside_fold() -> None:
    """Phase 1.1 fix: walk-forward functions must re-select features inside each fold.

    Previously rank_feature_candidates / prune_feature_candidates scored over the
    full training mask, which included rows that would later land in the test
    folds. The fix introduces ``select_fold_features`` (wraps
    ``select_usable_feature_columns`` with a fold-only training mask) and calls
    it at the top of each fold in all three walk-forward functions.
    """
    leaderboard_src = inspect.getsource(efp.walk_forward_leaderboard)
    classification_src = inspect.getsource(efp.walk_forward_classification)
    state_src = inspect.getsource(efp.walk_forward_state_classification)
    assert hasattr(efp, "select_fold_features"), "select_fold_features helper is missing"
    assert "select_fold_features" in leaderboard_src, (
        "walk_forward_leaderboard must call select_fold_features inside its fold loop"
    )
    assert "select_fold_features" in classification_src, (
        "walk_forward_classification must call select_fold_features inside its fold loop"
    )
    assert "select_fold_features" in state_src, (
        "walk_forward_state_classification must call select_fold_features inside its fold loop"
    )


def test_market_data_imputation_is_column_aware_and_bounded() -> None:
    """Phase 1.2 fix: finalize_market_data no longer blanket-ffills the merged frame.

    Bulk ``merged.ffill()`` has been replaced with ``apply_point_in_time_imputation``
    which forward-fills only the columns whose semantics justify it and always
    with a bounded ``limit=`` argument:

      - 24/7 crypto (eth_/btc_/...): never ffilled — real NaNs propagate.
      - Traditional markets (spy_/qqq_/...): ffilled up to a small max-age
        to cover weekends / holidays.
      - Vendor macro series: ffilled up to VENDOR_FFILL_MAX_DAYS so quarter-
        old releases don't masquerade as today's value.
    """
    finalize_source = inspect.getsource(efp.finalize_market_data)
    assert ".ffill(" not in finalize_source, (
        "finalize_market_data must not ffill directly; delegate to apply_point_in_time_imputation"
    )
    assert hasattr(efp, "apply_point_in_time_imputation"), "helper missing"
    helper_source = inspect.getsource(efp.apply_point_in_time_imputation)
    assert "limit=" in helper_source, "apply_point_in_time_imputation must use bounded ffill (limit=)"
    assert "CRYPTO_24_7_PREFIXES" in helper_source, "helper must skip 24/7 crypto columns"
    assert "TRADITIONAL_MARKET_PREFIXES" in helper_source, "helper must branch on traditional-market prefixes"
    assert "GENERIC_VENDOR_PREFIXES" in helper_source, "helper must branch on vendor prefixes"


def test_vendor_history_is_not_nullified_by_current_staleness(
    synthetic_ohlcv_with_companions: pd.DataFrame,
) -> None:
    """Phase 1.B fix: build_features must not wipe entire vendor columns when only
    the latest row is stale.

    Reproduces Finding B: fabricate a vendor column whose freshest value sits
    far behind ``data.index.max()`` (i.e. ``latest_age_days`` would trip the
    stale threshold). Historical rows of that column must survive the feature
    build — otherwise the current state is being back-projected onto the past.
    """
    frame = synthetic_ohlcv_with_companions.copy()
    frame["fred_cpi"] = np.nan
    freshness_tail = 60
    populated_slice = frame.index[:-freshness_tail]
    frame.loc[populated_slice, "fred_cpi"] = np.linspace(3.0, 3.5, len(populated_slice))

    feature_frame, _ = efp.build_features(frame, horizon=7)

    historical_fred = feature_frame.loc[populated_slice, "fred_cpi"]
    assert historical_fred.notna().mean() > 0.95, (
        "Historical fred_cpi values were nullified even though they were fresh at the time; "
        "this is the Finding B back-projection bug."
    )


def test_apply_point_in_time_imputation_behaviour() -> None:
    """Direct behaviour check of the Phase 1.2 helper on a synthetic frame.

    Confirms the three branches (24/7 crypto / traditional market / vendor) each
    respect their max-age limit and that unknown-prefix columns are never filled.
    """
    idx = pd.date_range("2024-01-01", periods=12, freq="D")
    frame = pd.DataFrame(index=idx, columns=[], dtype=float)
    frame["eth_close"] = np.arange(12, dtype=float)
    frame.loc[idx[5], "eth_close"] = np.nan  # stays NaN (24/7, no ffill)
    frame["spy_close"] = np.nan
    frame.loc[idx[0], "spy_close"] = 100.0  # should fill 1-5 (5 days), 6-11 stay NaN
    frame["fred_cpi"] = np.nan
    frame.loc[idx[0], "fred_cpi"] = 3.1  # vendor fills 1-11 (45-day limit, well past 11)
    frame["mystery_col"] = np.nan
    frame.loc[idx[0], "mystery_col"] = 42.0  # unknown prefix → stays NaN beyond row 0

    out = efp.apply_point_in_time_imputation(frame)

    assert pd.isna(out["eth_close"].iloc[5]), "24/7 crypto must NOT be ffilled"
    assert out["spy_close"].iloc[5] == 100.0, "traditional-market ffill must cover 5 days"
    assert pd.isna(out["spy_close"].iloc[6]), "traditional-market ffill must stop after max_age"
    assert (out["fred_cpi"].iloc[1:] == 3.1).all(), "vendor ffill should cover the 12-day window"
    assert pd.isna(out["mystery_col"].iloc[1]), "unknown-prefix columns must not be ffilled"


# ---------------------------------------------------------------------------
# Phase 1.3 — Purged walk-forward with explicit gap + embargo.
# ---------------------------------------------------------------------------


def test_walk_forward_functions_use_purged_splitter() -> None:
    """Phase 1.3 fix: all three walk-forward functions must use purged_time_series_split.

    The purged splitter wraps safe_time_series_split with total_gap = gap + embargo,
    so callers can request an explicit embargo buffer without threading gap math.
    """
    assert hasattr(efp, "purged_time_series_split"), "purged_time_series_split helper is missing"
    for fn_name in ("walk_forward_leaderboard", "walk_forward_classification", "walk_forward_state_classification"):
        src = inspect.getsource(getattr(efp, fn_name))
        assert "purged_time_series_split" in src, (
            f"{fn_name} must call purged_time_series_split (got: only safe_time_series_split?)"
        )
        assert "embargo" in src, f"{fn_name} must expose an embargo parameter"


def test_purged_splitter_respects_gap_plus_embargo() -> None:
    """Every fold must have test_idx.min() > train_idx.max() + gap + embargo.

    This is the López de Prado purge + embargo invariant: labels that use
    ``close.shift(-horizon)`` cannot leak into the test period, and a further
    ``embargo`` buffer protects against feature-side spillover.
    """
    n_samples = 400
    gap = 7
    embargo = 5
    splitter = efp.purged_time_series_split(
        n_samples=n_samples, n_splits=3, test_size=40, gap=gap, embargo=embargo,
    )
    fake_X = np.zeros((n_samples, 2))
    folds = list(splitter.split(fake_X))
    assert folds, "splitter produced no folds"
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        assert len(train_idx) > 0 and len(test_idx) > 0
        separation = int(test_idx.min()) - int(train_idx.max())
        assert separation > gap + embargo, (
            f"fold {fold_number}: train.max={train_idx.max()} test.min={test_idx.min()} "
            f"separation={separation} must be > gap+embargo={gap + embargo}"
        )


def test_purged_splitter_with_zero_embargo_matches_safe_splitter() -> None:
    """Backwards-compat: embargo=0 must yield the same folds as safe_time_series_split."""
    kwargs = dict(n_samples=300, n_splits=3, test_size=30, gap=7)
    safe = list(efp.safe_time_series_split(**kwargs).split(np.zeros((kwargs["n_samples"], 1))))
    purged = list(efp.purged_time_series_split(**kwargs, embargo=0).split(np.zeros((kwargs["n_samples"], 1))))
    assert len(safe) == len(purged)
    for (s_tr, s_te), (p_tr, p_te) in zip(safe, purged):
        assert np.array_equal(s_tr, p_tr) and np.array_equal(s_te, p_te)


def test_purged_splitter_embargo_shrinks_train_fold() -> None:
    """Adding embargo must reduce train size (purge steals rows from end of train)."""
    kwargs = dict(n_samples=300, n_splits=3, test_size=30, gap=7)
    no_embargo = list(efp.purged_time_series_split(**kwargs, embargo=0).split(np.zeros((kwargs["n_samples"], 1))))
    with_embargo = list(efp.purged_time_series_split(**kwargs, embargo=10).split(np.zeros((kwargs["n_samples"], 1))))
    for (ne_tr, _), (we_tr, _) in zip(no_embargo, with_embargo):
        assert len(we_tr) <= len(ne_tr), (
            f"embargo should not grow the train fold: no_embargo={len(ne_tr)} with_embargo={len(we_tr)}"
        )


# ---------------------------------------------------------------------------
# Phase 2 — Skill-weighted ensemble blending.
# ---------------------------------------------------------------------------


def test_skill_weighted_mean_matches_equal_weight_when_uniform() -> None:
    """Uniform weights must reproduce trimmed_equal_weight_average exactly."""
    frame = pd.DataFrame(
        {
            "model_a": [1.0, 10.0, 5.0],
            "model_b": [2.0, 20.0, 7.0],
            "model_c": [3.0, 30.0, 9.0],
        }
    )
    equal = efp.trimmed_equal_weight_average(frame)
    uniform = {"model_a": 1.0, "model_b": 1.0, "model_c": 1.0}
    weighted = efp.skill_weighted_trimmed_mean(frame, weights=uniform)
    assert np.allclose(equal.to_numpy(), weighted.to_numpy(), equal_nan=True)


def test_skill_weighted_mean_shifts_toward_higher_weight() -> None:
    """A heavily-weighted model must pull the blend toward its own values."""
    frame = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0], "c": [5.0, 6.0]})
    neutral = efp.skill_weighted_trimmed_mean(frame, weights={"a": 1, "b": 1, "c": 1})
    tilted = efp.skill_weighted_trimmed_mean(frame, weights={"a": 20, "b": 1, "c": 1})
    assert (tilted < neutral).all(), "heavier weight on 'a' (lowest values) must lower the blend"


def test_skill_weighted_mean_handles_nan_rows() -> None:
    """Rows with all NaN stay NaN; partial NaN rows redistribute weight to finite values."""
    frame = pd.DataFrame(
        {
            "a": [np.nan, 1.0, np.nan],
            "b": [np.nan, 2.0, 4.0],
            "c": [np.nan, 3.0, 6.0],
        }
    )
    out = efp.skill_weighted_trimmed_mean(frame, weights={"a": 1, "b": 1, "c": 1})
    assert pd.isna(out.iloc[0])
    assert out.iloc[1] == pytest.approx(2.0)
    assert out.iloc[2] == pytest.approx(5.0)


def test_derive_weights_lower_is_better_favors_low_metric() -> None:
    """Inverse-metric weights must rank the lowest-RMSE model highest and sum to 1."""
    leaderboard = pd.DataFrame(
        {"model": ["alpha", "beta", "gamma"], "price_rmse": [100.0, 200.0, 400.0]}
    )
    weights = efp.derive_ensemble_weights_from_leaderboard(
        leaderboard, ["alpha", "beta", "gamma"], "price_rmse", higher_is_better=False
    )
    assert weights["alpha"] > weights["beta"] > weights["gamma"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_derive_weights_higher_is_better_favors_high_metric() -> None:
    """Higher-better weights must rank the highest-AUC model highest and sum to 1."""
    leaderboard = pd.DataFrame(
        {"model": ["alpha", "beta", "gamma"], "roc_auc": [0.50, 0.60, 0.80]}
    )
    weights = efp.derive_ensemble_weights_from_leaderboard(
        leaderboard, ["alpha", "beta", "gamma"], "roc_auc", higher_is_better=True
    )
    assert weights["gamma"] > weights["beta"] > weights["alpha"]
    assert sum(weights.values()) == pytest.approx(1.0)


def test_derive_weights_fallback_to_uniform_on_missing_metric() -> None:
    """Missing metric column or empty leaderboard must return uniform weights."""
    members = ["alpha", "beta"]
    uniform_from_empty = efp.derive_ensemble_weights_from_leaderboard(
        pd.DataFrame(), members, "price_rmse", higher_is_better=False
    )
    assert uniform_from_empty == {"alpha": 0.5, "beta": 0.5}
    uniform_from_missing_col = efp.derive_ensemble_weights_from_leaderboard(
        pd.DataFrame({"model": members, "other_metric": [1, 2]}),
        members, "price_rmse", higher_is_better=False,
    )
    assert uniform_from_missing_col == {"alpha": 0.5, "beta": 0.5}


def test_ensemble_appenders_use_skill_weighting() -> None:
    """Static check: the OOF-ensemble appenders must reference the skill-weighted blender."""
    for fn_name in ("append_trimmed_regression_ensemble_candidate", "append_trimmed_classification_ensemble_candidate"):
        src = inspect.getsource(getattr(efp, fn_name))
        assert "skill_weighted_trimmed_mean" in src, (
            f"{fn_name} must use skill_weighted_trimmed_mean (regression=price_rmse, classification=brier_score)"
        )
        assert "derive_ensemble_weights_from_leaderboard" in src, (
            f"{fn_name} must derive weights from the leaderboard"
        )


# ---------------------------------------------------------------------------
# Phase 2.5 — Tighter ensemble member selection + best-single fallback.
# ---------------------------------------------------------------------------


def test_regression_member_selection_tolerance_is_tight() -> None:
    """Phase 2.5: tolerance must be <= 1.10 (h<=7) and <= 1.18 (h>7).

    The pre-Phase-2.5 values were 1.18 / 1.30 which let materially worse
    models dilute the blend. Anything above this threshold is a regression.
    """
    src = inspect.getsource(efp.select_trimmed_regression_ensemble_members)
    assert "1.08" in src or "1.05" in src or "1.07" in src or "1.10" in src, (
        "regression member selection tolerance (h<=7) must be tightened below 1.18"
    )
    assert "1.15" in src or "1.12" in src or "1.18" in src, (
        "regression member selection tolerance (h>7) must be tightened below 1.30"
    )


def test_classification_member_selection_uses_brier_filter() -> None:
    """Phase 2.5: classification member filter must inspect brier_score.

    Previously only balanced_accuracy + roc_auc thresholds applied, so models
    with high AUC but poor calibration got into the blend.
    """
    src = inspect.getsource(efp.select_trimmed_classification_ensemble_members)
    assert "brier_score" in src, (
        "classification member selection must filter on brier_score (calibration)"
    )


def test_ensemble_appenders_honour_best_single_fallback() -> None:
    """Phase 2.5: append_*_ensemble_candidate must not record a row worse than best member."""
    for fn_name in ("append_trimmed_regression_ensemble_candidate", "append_trimmed_classification_ensemble_candidate"):
        src = inspect.getsource(getattr(efp, fn_name))
        assert "best_member" in src, (
            f"{fn_name} must compare ensemble OOF metric to best member and skip if worse"
        )


def test_best_single_fallback_skips_dilute_regression_ensemble() -> None:
    """End-to-end: if member predictions differ in quality, the blend's RMSE must
    either match the best member or the ensemble row is omitted entirely."""
    idx = pd.date_range("2024-01-01", periods=20, freq="D")
    # strong member: predicted ~ truth ; weak member: constant zero
    truth = np.linspace(0.01, 0.05, 20)
    strong = truth + np.random.default_rng(0).normal(scale=0.001, size=20)
    weak = np.zeros(20)
    oof = pd.DataFrame(
        {
            "strong_pred_return": strong,
            "strong_pred_close": np.linspace(2000, 2300, 20),
            "weak_pred_return": weak,
            "weak_pred_close": np.linspace(2000, 2000, 20),
        },
        index=idx,
    )
    dataset = pd.DataFrame(
        {"eth_close": np.linspace(2000, 2300, 20), "target_return": truth},
        index=idx,
    )
    leaderboard = pd.DataFrame(
        [
            {"model": "strong", "price_rmse": 10.0, "price_mae": 8.0, "directional_accuracy": 0.9, "folds": 3.0},
            {"model": "weak", "price_rmse": 500.0, "price_mae": 400.0, "directional_accuracy": 0.1, "folds": 3.0},
        ]
    )
    new_lb, _ = efp.append_trimmed_regression_ensemble_candidate(
        leaderboard, oof, dataset, horizon=7, component_models=["strong", "weak"],
    )
    ensemble_row = new_lb.loc[new_lb["model"] == efp.TRIMMED_REGRESSION_ENSEMBLE_MODEL]
    if not ensemble_row.empty:
        ensemble_rmse = float(ensemble_row.iloc[0]["price_rmse"])
        assert ensemble_rmse <= 10.0, (
            f"ensemble RMSE {ensemble_rmse:.2f} should not exceed best member (10.0) once the guard is active"
        )


# ---------------------------------------------------------------------------
# Phase 3: derivatives-regime feature additions
# ---------------------------------------------------------------------------


def _phase3_synthetic_market_data(rows: int = 400) -> pd.DataFrame:
    """Small synthetic gold-style frame with the deribit columns Phase 3 reads."""
    rng = np.random.default_rng(13)
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    eth_close = pd.Series(2000.0 + np.cumsum(rng.normal(0.0, 20.0, rows)), index=idx)
    eth_high = eth_close * (1.0 + np.abs(rng.normal(0.0, 0.01, rows)))
    eth_low = eth_close * (1.0 - np.abs(rng.normal(0.0, 0.01, rows)))
    eth_open = eth_close.shift(1).fillna(eth_close.iloc[0])
    eth_volume = pd.Series(rng.uniform(1e9, 2e9, rows), index=idx)
    btc_close = pd.Series(40000.0 + np.cumsum(rng.normal(0.0, 200.0, rows)), index=idx)
    btc_high = btc_close * 1.01
    btc_low = btc_close * 0.99
    btc_open = btc_close.shift(1).fillna(btc_close.iloc[0])
    btc_volume = pd.Series(rng.uniform(1e10, 2e10, rows), index=idx)
    funding_8h = pd.Series(rng.normal(0.00005, 0.0001, rows), index=idx)
    option_iv = pd.Series(0.6 + 0.1 * np.sin(np.linspace(0, 6, rows)) + rng.normal(0, 0.02, rows), index=idx)
    pcr = pd.Series(0.9 + 0.2 * np.sin(np.linspace(0, 4, rows)) + rng.normal(0, 0.05, rows), index=idx)
    futures_mark = eth_close * (1.0 + rng.normal(0.0, 0.003, rows))
    futures_oi = pd.Series(np.cumsum(rng.normal(1e4, 1e3, rows)) + 1e6, index=idx)
    return pd.DataFrame(
        {
            "eth_close": eth_close, "eth_high": eth_high, "eth_low": eth_low,
            "eth_open": eth_open, "eth_volume": eth_volume,
            "btc_close": btc_close, "btc_high": btc_high, "btc_low": btc_low,
            "btc_open": btc_open, "btc_volume": btc_volume,
            "deribit_eth_funding_interest_8h": funding_8h,
            "deribit_eth_option_mark_iv_mean": option_iv,
            "deribit_eth_option_put_call_oi_ratio": pcr,
            "deribit_eth_future_mark_price_mean": futures_mark,
            "deribit_eth_future_open_interest_sum": futures_oi,
        },
        index=idx,
    )


def test_phase3_derivatives_features_are_emitted() -> None:
    data = _phase3_synthetic_market_data(rows=400)
    feature_frame, candidates = efp.build_features(data, horizon=7)
    expected = {
        "deribit_funding_8h_z_90",
        "deribit_option_iv_z_90",
        "deribit_pcr_z_90",
        "deribit_futures_basis_ratio",
        "deribit_futures_basis_z_30",
        "deribit_future_oi_change_7",
    }
    missing = expected - set(feature_frame.columns)
    assert not missing, f"Phase 3 derivatives features missing: {sorted(missing)}"
    assert expected.issubset(set(candidates)), (
        f"Phase 3 features should be candidate columns; missing {expected - set(candidates)}"
    )


def test_phase3_features_are_fold_safe_at_row_t() -> None:
    """Phase 3 domain features must depend only on past data at row t."""
    data = _phase3_synthetic_market_data(rows=400)
    full_frame, _ = efp.build_features(data, horizon=7)
    # corrupt the future tail; row t values should be unchanged
    truncated = data.iloc[: len(data) - 50].copy()
    partial_frame, _ = efp.build_features(truncated, horizon=7)
    common_index = partial_frame.index.intersection(full_frame.index)
    feature_names = [
        "deribit_funding_8h_z_90",
        "deribit_option_iv_z_90",
        "deribit_pcr_z_90",
        "deribit_futures_basis_ratio",
        "deribit_futures_basis_z_30",
        "deribit_future_oi_change_7",
    ]
    for name in feature_names:
        a = full_frame.loc[common_index, name].astype(float)
        b = partial_frame.loc[common_index, name].astype(float)
        mismatch = (a.fillna(-1e18) - b.fillna(-1e18)).abs()
        assert (mismatch < 1e-9).all(), (
            f"{name} at row t changed when future rows were removed — leakage suspected"
        )


def test_phase3_features_tolerate_missing_deribit_columns() -> None:
    """Older master CSVs without deribit options must still build features."""
    base = _phase3_synthetic_market_data(rows=300)
    without_options = base.drop(
        columns=[
            "deribit_eth_option_mark_iv_mean",
            "deribit_eth_option_put_call_oi_ratio",
            "deribit_eth_future_mark_price_mean",
            "deribit_eth_future_open_interest_sum",
        ]
    )
    feature_frame, candidates = efp.build_features(without_options, horizon=7)
    assert not feature_frame.empty
    for absent in (
        "deribit_option_iv_z_90",
        "deribit_pcr_z_90",
        "deribit_futures_basis_ratio",
        "deribit_future_oi_change_7",
    ):
        assert absent not in feature_frame.columns, (
            f"{absent} should not be emitted when source column is missing"
        )
    # the funding feature should still land because its source column remains
    assert "deribit_funding_8h_z_90" in feature_frame.columns


# ---------------------------------------------------------------------------
# Phase 3A: collector backfill widening
# ---------------------------------------------------------------------------


def test_collector_widens_deribit_funding_backfill_to_90_days() -> None:
    """The collector should fetch a wider funding window than the 3-day overlap
    when local history is shorter than Deribit's ~90-day retention."""
    collector_src = Path(__file__).resolve().parents[2] / "eth_data_collector.py"
    src = collector_src.read_text(encoding="utf-8")
    assert "min_deribit_backfill_days = 90" in src, (
        "Phase 3A: expected the collector to widen Deribit funding backfill to 90d."
    )
    assert "earliest_target" in src and "existing.index.min() > earliest_target" in src, (
        "Phase 3A: backfill widening should key off earliest_target vs existing CSV coverage."
    )


def test_collector_documents_deribit_snapshot_limitation() -> None:
    """Futures/options are snapshot-only; the collector should say so near the loop."""
    collector_src = Path(__file__).resolve().parents[2] / "eth_data_collector.py"
    src = collector_src.read_text(encoding="utf-8")
    snapshot_marker = "deribit_eth_future_snapshot_daily.csv"
    assert snapshot_marker in src
    snapshot_idx = src.index(snapshot_marker)
    preceding_chunk = src[max(0, snapshot_idx - 600):snapshot_idx]
    assert "snapshot-only" in preceding_chunk.lower(), (
        "Phase 3A: options/futures snapshot loop should be preceded by a note "
        "explaining that Deribit has no historical API for these endpoints."
    )


# ---------------------------------------------------------------------------
# Phase 3B: rich-column macro regime composites
# ---------------------------------------------------------------------------


PHASE3B_FEATURES = (
    "fred_yield_curve_change_20",
    "fred_real_yield_10y",
    "fred_real_yield_10y_z_90",
    "fred_credit_stress",
    "fred_credit_stress_change_20",
    "macro_asymmetry_20",
)


def _phase3b_synthetic_market_data(rows: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(29)
    idx = pd.date_range("2024-01-01", periods=rows, freq="D")
    eth_close = pd.Series(2000.0 + np.cumsum(rng.normal(0.0, 20.0, rows)), index=idx)
    eth_high = eth_close * 1.01
    eth_low = eth_close * 0.99
    eth_open = eth_close.shift(1).fillna(eth_close.iloc[0])
    eth_volume = pd.Series(rng.uniform(1e9, 2e9, rows), index=idx)
    btc_close = pd.Series(40000.0 + np.cumsum(rng.normal(0.0, 200.0, rows)), index=idx)
    btc_high = btc_close * 1.01
    btc_low = btc_close * 0.99
    btc_open = btc_close.shift(1).fillna(btc_close.iloc[0])
    btc_volume = pd.Series(rng.uniform(1e10, 2e10, rows), index=idx)
    treasury_10y = pd.Series(4.0 + np.cumsum(rng.normal(0, 0.02, rows)), index=idx)
    curve_10y_2y = pd.Series(np.cumsum(rng.normal(0, 0.01, rows)), index=idx)
    breakeven_10y = pd.Series(2.3 + np.cumsum(rng.normal(0, 0.01, rows)), index=idx)
    hy_oas = pd.Series(3.5 + np.cumsum(rng.normal(0, 0.02, rows)), index=idx)
    bbb_oas = pd.Series(1.5 + np.cumsum(rng.normal(0, 0.01, rows)), index=idx)
    dxy_close = pd.Series(100.0 + np.cumsum(rng.normal(0, 0.1, rows)), index=idx)
    tnx_close = treasury_10y * 10.0
    return pd.DataFrame(
        {
            "eth_close": eth_close, "eth_high": eth_high, "eth_low": eth_low,
            "eth_open": eth_open, "eth_volume": eth_volume,
            "btc_close": btc_close, "btc_high": btc_high, "btc_low": btc_low,
            "btc_open": btc_open, "btc_volume": btc_volume,
            "fred_treasury_10y": treasury_10y,
            "fred_yield_curve_10y_2y": curve_10y_2y,
            "fred_breakeven_10y": breakeven_10y,
            "fred_hy_oas": hy_oas,
            "fred_bbb_oas": bbb_oas,
            "dxy_close": dxy_close,
            "tnx_close": tnx_close,
        },
        index=idx,
    )


def test_phase3b_macro_composites_are_not_emitted() -> None:
    """Regression guard: Phase 3B composites were removed after the ablation
    showed they degrade h=30 classification brier from 0.173 to 0.266 and AUC
    from 0.750 to 0.585 — at the best-single level, not only via ensemble
    selection. Re-adding them as a blanket block would reintroduce the
    regression; any future attempt must be leave-one-out gated first.
    """
    data = _phase3b_synthetic_market_data(rows=400)
    feature_frame, candidates = efp.build_features(data, horizon=7)
    for feature in PHASE3B_FEATURES:
        assert feature not in feature_frame.columns, (
            f"{feature} was removed in Phase 4B; re-adding it without a leave-one-out "
            f"freeze that shows h=30 brier <= 0.175 is a regression."
        )
        assert feature not in candidates


def test_build_features_explains_phase3b_removal() -> None:
    """The removal must be documented in-source so future contributors see why
    adding macro composites as a block is gated on a leave-one-out experiment."""
    src = inspect.getsource(efp.build_features)
    assert "Phase 3B macro composites" in src and "removed" in src.lower(), (
        "build_features should keep a comment explaining why Phase 3B composites were removed"
    )
    assert "leave-one-out" in src.lower(), (
        "removal comment should tell future-us how to retry safely (leave-one-out)"
    )


# ---------------------------------------------------------------------------
# Phase 4 — synergy-aware member selection (OOF-correlation diversification)
# ---------------------------------------------------------------------------


def test_phase4_greedy_diversify_prefers_low_correlation_when_skill_close() -> None:
    """When three candidates have comparable skill, the greedy diversifier must
    skip the near-duplicate of the top pick in favour of an uncorrelated peer."""
    idx = pd.date_range("2024-01-01", periods=120, freq="D")
    base = np.sin(np.linspace(0, 6, 120))
    oof = pd.DataFrame(
        {
            "alpha_pred_return": base + np.random.default_rng(0).normal(scale=0.01, size=120),
            "clone_pred_return": base + np.random.default_rng(1).normal(scale=0.01, size=120),
            "diverse_pred_return": np.cos(np.linspace(0, 6, 120)),
        },
        index=idx,
    )
    candidates = ["alpha", "clone", "diverse"]
    picked = efp._greedy_diversify_members(
        skill_ordered_candidates=candidates,
        oof_predictions=oof,
        prediction_column_suffix="_pred_return",
        max_members=2,
    )
    assert picked[0] == "alpha"
    assert picked[1] == "diverse", (
        f"greedy diversifier should skip the near-duplicate clone, got {picked}"
    )


def test_phase4_greedy_diversify_falls_back_without_oof() -> None:
    """If OOF predictions are unavailable the helper must return skill order."""
    picked = efp._greedy_diversify_members(
        skill_ordered_candidates=["a", "b", "c", "d"],
        oof_predictions=None,
        prediction_column_suffix="_pred_return",
        max_members=3,
    )
    assert picked == ["a", "b", "c"]


def test_phase4_greedy_diversify_falls_back_when_overlap_is_small() -> None:
    """With <min_overlap rows the helper must degrade to skill order rather
    than trust a correlation computed on too-few pairs."""
    idx = pd.date_range("2024-01-01", periods=5, freq="D")
    oof = pd.DataFrame(
        {
            "alpha_pred_return": [0.1, 0.2, 0.15, 0.05, 0.0],
            "clone_pred_return": [0.11, 0.21, 0.16, 0.06, 0.01],
            "diverse_pred_return": [-0.2, -0.1, 0.0, 0.1, 0.2],
        },
        index=idx,
    )
    picked = efp._greedy_diversify_members(
        skill_ordered_candidates=["alpha", "clone", "diverse"],
        oof_predictions=oof,
        prediction_column_suffix="_pred_return",
        max_members=2,
        min_overlap=50,
    )
    assert picked[0] == "alpha"
    assert picked[1] == "clone"  # skill order because overlap is too small


def test_phase4_regression_selector_accepts_oof_predictions() -> None:
    """Public regression selector must accept oof_predictions and diversify."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    base = np.linspace(-0.02, 0.04, 100)
    oof = pd.DataFrame(
        {
            "alpha_pred_return": base + np.random.default_rng(0).normal(scale=0.002, size=100),
            "clone_pred_return": base + np.random.default_rng(1).normal(scale=0.002, size=100),
            "diverse_pred_return": -base + np.random.default_rng(2).normal(scale=0.002, size=100),
        },
        index=idx,
    )
    leaderboard = pd.DataFrame(
        [
            {"model": "alpha", "price_rmse": 10.0, "price_mae": 7.5, "directional_accuracy": 0.62},
            {"model": "clone", "price_rmse": 10.2, "price_mae": 7.6, "directional_accuracy": 0.61},
            {"model": "diverse", "price_rmse": 10.3, "price_mae": 7.7, "directional_accuracy": 0.60},
        ]
    )
    picked = efp.select_trimmed_regression_ensemble_members(
        leaderboard, horizon=7, max_members=2, oof_predictions=oof,
    )
    assert "diverse" in picked, (
        f"diverse (uncorrelated) peer should win diversification slot, got {picked}"
    )


def test_phase4_classification_selector_accepts_oof_predictions() -> None:
    """Public classification selector must accept oof_predictions and diversify."""
    idx = pd.date_range("2024-01-01", periods=100, freq="D")
    base = np.clip(0.5 + 0.4 * np.sin(np.linspace(0, 6, 100)), 0.05, 0.95)
    orthogonal = np.clip(0.5 + 0.4 * np.cos(np.linspace(0, 6, 100)), 0.05, 0.95)
    oof = pd.DataFrame(
        {
            "alpha_prob_up": base,
            "clone_prob_up": np.clip(base + np.random.default_rng(0).normal(scale=0.01, size=100), 0.05, 0.95),
            "diverse_prob_up": orthogonal,
        },
        index=idx,
    )
    leaderboard = pd.DataFrame(
        [
            {"model": "alpha", "balanced_accuracy": 0.62, "roc_auc": 0.68, "f1": 0.60, "brier_score": 0.22, "signal_threshold": 0.5},
            {"model": "clone", "balanced_accuracy": 0.61, "roc_auc": 0.67, "f1": 0.59, "brier_score": 0.23, "signal_threshold": 0.5},
            {"model": "diverse", "balanced_accuracy": 0.60, "roc_auc": 0.66, "f1": 0.58, "brier_score": 0.24, "signal_threshold": 0.5},
        ]
    )
    picked = efp.select_trimmed_classification_ensemble_members(
        leaderboard, horizon=7, max_members=2, oof_predictions=oof,
    )
    assert "diverse" in picked, (
        f"classification diversifier should pick the anti-correlated peer, got {picked}"
    )


def test_phase4_selectors_preserve_backwards_compat_without_oof() -> None:
    """Both selectors must still return skill-order top-N when oof_predictions=None."""
    reg_lb = pd.DataFrame(
        [
            {"model": "a", "price_rmse": 10.0, "price_mae": 7.0, "directional_accuracy": 0.6},
            {"model": "b", "price_rmse": 10.5, "price_mae": 7.5, "directional_accuracy": 0.58},
            {"model": "c", "price_rmse": 11.0, "price_mae": 8.0, "directional_accuracy": 0.56},
        ]
    )
    reg_picked = efp.select_trimmed_regression_ensemble_members(reg_lb, horizon=7, max_members=2)
    assert reg_picked == ["a", "b"]

    cls_lb = pd.DataFrame(
        [
            {"model": "a", "balanced_accuracy": 0.65, "roc_auc": 0.70, "f1": 0.62, "brier_score": 0.20, "signal_threshold": 0.5},
            {"model": "b", "balanced_accuracy": 0.63, "roc_auc": 0.68, "f1": 0.60, "brier_score": 0.21, "signal_threshold": 0.5},
            {"model": "c", "balanced_accuracy": 0.60, "roc_auc": 0.66, "f1": 0.58, "brier_score": 0.22, "signal_threshold": 0.5},
        ]
    )
    cls_picked = efp.select_trimmed_classification_ensemble_members(cls_lb, horizon=7, max_members=2)
    assert cls_picked == ["a", "b"]


def test_phase4_run_horizon_pipeline_passes_oof_to_selectors() -> None:
    """Source audit: run_horizon_pipeline must forward oof_predictions to both selectors."""
    src = inspect.getsource(efp.run_horizon_pipeline)
    assert "oof_predictions=regression_oof_predictions" in src, (
        "run_horizon_pipeline must pass regression_oof_predictions into the regression selector"
    )
    assert "oof_predictions=classification_oof_predictions" in src, (
        "run_horizon_pipeline must pass classification_oof_predictions into the classification selector"
    )


def test_phase4_prediction_headline_is_wired_into_main() -> None:
    """CLI main() must surface a headline forecast block listing 7d/30d predictions."""
    src = inspect.getsource(efp.main)
    assert "Forecast reference" in src, "main() should emit a 'Forecast reference' line"
    assert "d forecast -> target=" in src, (
        "main() should emit a per-horizon headline forecast row"
    )
    assert "hybrid conservative" in src and "hybrid active" in src, (
        "headline should surface both hybrid conservative and active track prices"
    )

