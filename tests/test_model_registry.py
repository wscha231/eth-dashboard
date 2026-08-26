from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import eth_price_forecast as efp
from forecasting import model_registry


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


def test_model_eval_workflows_enable_challengers_but_daily_forecast_does_not() -> None:
    eval_workflow = Path(".github/workflows/eth_model_eval.yml").read_text(encoding="utf-8")
    daily_workflow = Path(".github/workflows/daily_forecast.yml").read_text(encoding="utf-8")

    assert eval_workflow.count('ETH_ENABLE_CHALLENGER_MODELS: "1"') == 3
    model_eval_job, forecast_job = daily_workflow.split("\n  forecast:\n", maxsplit=1)
    assert 'ETH_ENABLE_CHALLENGER_MODELS: "1"' in model_eval_job
    assert "ETH_ENABLE_CHALLENGER_MODELS" not in forecast_job
