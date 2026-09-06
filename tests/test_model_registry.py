from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, Ridge

import eth_price_forecast as efp
from forecasting import model_registry
from tests.phase0.longrun_oof_common import (
    active_model_registry_manifest,
    checkpoint_model_registry_compatible,
    select_model_subset,
)


@pytest.fixture(autouse=True)
def restore_runtime_options():
    previous = dict(efp.RUNTIME_OPTIONS)
    yield
    efp.RUNTIME_OPTIONS.clear()
    efp.RUNTIME_OPTIONS.update(previous)


def test_refactor_preserves_existing_registry_when_challengers_are_disabled() -> None:
    efp.set_runtime_options(fast_mode=False, challenger_models=False)

    regression_models = efp.make_models(horizon=7)
    classification_models = efp.make_classification_models(horizon=7)

    expected_regression = {
        "ridge",
        "random_forest",
        "extra_trees",
        "hist_gbr",
        "knn_regressor",
        "mlp_regressor",
    }
    expected_classification = {
        "logistic",
        "extra_trees_clf",
        "random_forest_clf",
        "knn_clf",
        "hist_gbc",
        "mlp_clf",
    }
    if efp.CatBoostRegressor is not None:
        expected_regression.add("catboost_regressor")
    if efp.CatBoostClassifier is not None:
        expected_classification.add("catboost_clf")

    assert set(regression_models) == expected_regression
    assert set(classification_models) == expected_classification


@pytest.mark.skipif(
    efp.LGBMRegressor is None or efp.LGBMClassifier is None,
    reason="LightGBM is optional outside the CI/runtime requirements environment",
)
def test_lightgbm_is_opt_in_and_regularized_for_daily_data() -> None:
    efp.set_runtime_options(fast_mode=False, challenger_models=False)
    assert "lightgbm_regressor" not in efp.make_models(horizon=7)
    assert "lightgbm_clf" not in efp.make_classification_models(horizon=7)

    efp.set_runtime_options(challenger_models=True)
    regression = efp.make_models(horizon=30)["lightgbm_regressor"].named_steps["model"]
    classifier = efp.make_classification_models(horizon=30)["lightgbm_clf"].named_steps["model"]
    state_classifier = efp.make_state_classification_models(horizon=30)["lightgbm_clf"].named_steps["model"]

    regression_params = regression.get_params()
    classifier_params = classifier.get_params()
    state_params = state_classifier.get_params()
    assert regression_params["objective"] == "regression_l1"
    assert classifier_params["objective"] == "binary"
    assert state_params["objective"] == "multiclass"
    assert regression_params["num_leaves"] <= 15
    assert regression_params["min_child_samples"] >= 35
    assert classifier_params["class_weight"] == "balanced"
    assert regression_params["deterministic"] is True
    assert classifier_params["deterministic"] is True


@pytest.mark.skipif(
    efp.LGBMRegressor is None or efp.LGBMClassifier is None,
    reason="LightGBM is optional outside the CI/runtime requirements environment",
)
def test_lightgbm_challengers_fit_through_pipeline_and_calibration() -> None:
    rng = np.random.default_rng(42)
    index = pd.date_range("2024-01-01", periods=240, freq="D")
    X = pd.DataFrame(rng.normal(size=(len(index), 8)), index=index)
    y_return = pd.Series(0.02 * X[0] - 0.01 * X[1] + rng.normal(0.0, 0.01, len(index)), index=index)
    y_direction = pd.Series((y_return > 0.0).astype(int), index=index)
    sample_weight = pd.Series(np.linspace(0.25, 1.0, len(index)), index=index)

    efp.set_runtime_options(fast_mode=False, challenger_models=True)
    regression = efp.make_models(horizon=7)["lightgbm_regressor"]
    efp.fit_model_with_optional_sample_weight(regression, X, y_return, sample_weight)
    regression_prediction = regression.predict(X.tail(8))

    classifier = efp.fit_calibrated_classifier(
        efp.make_classification_models(horizon=7)["lightgbm_clf"],
        X,
        y_direction,
        min_calibration_rows=80,
        sample_weight=sample_weight,
        horizon=7,
    )
    probability = classifier.predict_proba(X.tail(8))[:, 1]
    direction_score = efp.classifier_direction_scores(classifier, X.tail(8))

    assert np.isfinite(regression_prediction).all()
    assert np.isfinite(probability).all()
    assert np.isfinite(direction_score).all()
    assert np.all((probability > 0.0) & (probability < 1.0))


def test_catboost_parameter_contract_survives_registry_extraction() -> None:
    assert efp.catboost_regressor_params(7) == model_registry.catboost_regressor_params(7)
    assert efp.catboost_regressor_params(30) == model_registry.catboost_regressor_params(30)
    assert efp.catboost_classifier_params(7) == model_registry.catboost_classifier_params(7)
    assert efp.catboost_classifier_params(30, multiclass=True) == model_registry.catboost_classifier_params(
        30,
        multiclass=True,
    )


def test_compact_h30_selector_keeps_first_192_fold_ranked_features() -> None:
    frame = pd.DataFrame(
        np.arange(4 * 240, dtype=float).reshape(4, 240),
        columns=[f"feature_{index:03d}" for index in range(240)],
    )
    selector = model_registry.LeadingFeatureSelector(max_features=192)

    transformed = selector.fit_transform(frame)

    assert transformed.shape == (4, 192)
    assert list(transformed.columns) == list(frame.columns[:192])
    assert selector.get_feature_names_out().tolist() == list(frame.columns[:192])


def test_compact_h30_regressor_is_strictly_opt_in_and_horizon_scoped() -> None:
    class FakeCatBoostRegressor:
        def __init__(self, **params):
            self.params = params

    disabled = model_registry.build_regression_models(
        horizon=30,
        fast_mode=False,
        imputer_factory=efp._make_median_imputer,
        catboost_regressor_cls=FakeCatBoostRegressor,
        compact_h30_regressor_enabled=False,
    )
    short_horizon = model_registry.build_regression_models(
        horizon=7,
        fast_mode=False,
        imputer_factory=efp._make_median_imputer,
        catboost_regressor_cls=FakeCatBoostRegressor,
        compact_h30_regressor_enabled=True,
    )
    enabled = model_registry.build_regression_models(
        horizon=30,
        fast_mode=False,
        imputer_factory=efp._make_median_imputer,
        catboost_regressor_cls=FakeCatBoostRegressor,
        compact_h30_regressor_enabled=True,
    )

    assert "catboost_compact_h30_regressor" not in disabled
    assert "catboost_compact_h30_regressor" not in short_horizon
    selector = enabled["catboost_compact_h30_regressor"].named_steps["feature_budget"]
    assert selector.max_features == 192


def test_focused_evaluation_model_subset_can_disable_a_head() -> None:
    models = {"ridge": Ridge(alpha=1.0), "ridge_alt": Ridge(alpha=2.0)}

    assert list(select_model_subset(models, ["ridge_alt"], head="regression")) == [
        "ridge_alt"
    ]
    assert select_model_subset(models, [], head="classification") == {}
    with pytest.raises(ValueError, match="Unknown regression model"):
        select_model_subset(models, ["missing"], head="regression")


def test_model_eval_workflows_enable_challengers_but_daily_forecast_does_not() -> None:
    eval_workflow = Path(".github/workflows/eth_model_eval.yml").read_text(encoding="utf-8")
    daily_workflow = Path(".github/workflows/daily_forecast.yml").read_text(encoding="utf-8")

    assert eval_workflow.count('ETH_ENABLE_CHALLENGER_MODELS: "1"') == 3
    assert '- "tests/**"' in eval_workflow
    assert "ETH_ENABLE_CHALLENGER_MODELS" not in daily_workflow
    assert "predict_latest.py" not in daily_workflow
    assert "persist_forecast" not in daily_workflow
    hybrid_daily = Path(".github/workflows/hybrid_daily.yml").read_text(encoding="utf-8")
    assert "scripts/settle_retired_hybrid.py" in hybrid_daily
    assert "scripts/run_hybrid_forecast.py --daily" not in hybrid_daily
    assert 'ETH_ENABLE_COMPACT_H30_REGRESSOR: "1"' in eval_workflow
    assert "ETH_ENABLE_COMPACT_H30_REGRESSOR" not in daily_workflow


def test_summary_resolves_challenger_provenance_at_export_time() -> None:
    efp.set_runtime_options(challenger_models=True, compact_h30_regressor=True)

    payload = efp.summarize_artifacts(
        efp.PipelineArtifacts(raw_data=pd.DataFrame(), horizons={})
    )

    assert bool(payload["optional_models"]["lightgbm_challenger_enabled"]) is True
    assert bool(payload["optional_models"]["lightgbm_available"]) is bool(
        efp.LGBMRegressor is not None and efp.LGBMClassifier is not None
    )
    assert bool(payload["optional_models"]["compact_h30_regressor_enabled"]) is True
    assert payload["optional_models"]["compact_h30_feature_count"] == 192


def test_resume_checkpoint_requires_exact_model_registry_manifest() -> None:
    runner = SimpleNamespace(
        _reg_models={"ridge": Ridge(alpha=1.0)},
        _cls_models={"logistic": LogisticRegression(C=1.0)},
    )
    active = active_model_registry_manifest({7: runner})
    serialized = json.loads(json.dumps(active))

    assert checkpoint_model_registry_compatible(
        {"model_registry": serialized}, active
    )
    assert not checkpoint_model_registry_compatible({}, active)

    changed_runner = SimpleNamespace(
        _reg_models={
            "ridge": Ridge(alpha=1.0),
            "new_challenger": Ridge(alpha=2.0),
        },
        _cls_models={"logistic": LogisticRegression(C=1.0)},
    )
    changed = active_model_registry_manifest({7: changed_runner})
    assert not checkpoint_model_registry_compatible(
        {"model_registry": serialized}, changed
    )
