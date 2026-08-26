"""Model factories shared by live forecasts and leakage-safe evaluation.

The public CLI remains in :mod:`eth_price_forecast`.  Keeping estimator
construction here makes the production registry small enough to review and
lets candidate models stay explicitly gated before they reach the daily job.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ImputerFactory = Callable[[], SimpleImputer]


def catboost_regressor_params(horizon: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "MAE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "boosting_type": "Ordered",
        "has_time": True,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if horizon is not None and horizon >= 30:
        params.update(
            {
                "iterations": 650,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.85,
            }
        )
    return params


def catboost_classifier_params(
    horizon: int | None = None,
    multiclass: bool = False,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "MultiClass" if multiclass else "Logloss",
        "iterations": 450,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "boosting_type": "Ordered",
        "has_time": True,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if horizon is not None and horizon >= 30:
        params.update(
            {
                "iterations": 650,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.85,
            }
        )
    if not multiclass and horizon is not None and horizon >= 30:
        params["auto_class_weights"] = "Balanced"
    return params


def lightgbm_regressor_params(horizon: int | None = None) -> dict[str, Any]:
    """Conservative daily-data parameters for an OOF-only challenger."""
    long_horizon = horizon is not None and horizon >= 30
    return {
        "objective": "regression_l1",
        "n_estimators": 650 if long_horizon else 500,
        "learning_rate": 0.02 if long_horizon else 0.025,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 35 if long_horizon else 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0 if long_horizon else 1.5,
        "max_bin": 127,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def lightgbm_classifier_params(
    horizon: int | None = None,
    multiclass: bool = False,
) -> dict[str, Any]:
    """Regularized LightGBM classifier parameters for the small ETH sample."""
    long_horizon = horizon is not None and horizon >= 30
    return {
        "objective": "multiclass" if multiclass else "binary",
        "n_estimators": 600 if long_horizon else 450,
        "learning_rate": 0.02 if long_horizon else 0.025,
        "num_leaves": 15,
        "max_depth": 5,
        "min_child_samples": 35 if long_horizon else 25,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.65,
        "reg_alpha": 0.10,
        "reg_lambda": 2.0 if long_horizon else 1.5,
        "max_bin": 127,
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
        "deterministic": True,
        "force_col_wise": True,
    }


def build_regression_models(
    *,
    horizon: int | None,
    fast_mode: bool,
    imputer_factory: ImputerFactory,
    catboost_regressor_cls: type[Any] | None = None,
    lightgbm_regressor_cls: type[Any] | None = None,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "ridge": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gbr": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=0.05,
                        max_depth=5,
                        max_iter=250,
                        min_samples_leaf=12,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "knn_regressor": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("scaler", StandardScaler()),
                ("model", KNeighborsRegressor(n_neighbors=15, weights="distance")),
            ]
        ),
    }
    if not fast_mode:
        models["mlp_regressor"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
    if (not fast_mode) and catboost_regressor_cls is not None:
        models["catboost_regressor"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("model", catboost_regressor_cls(**catboost_regressor_params(horizon))),
            ]
        )
    if (not fast_mode) and lightgbm_regressor_cls is not None:
        models["lightgbm_regressor"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("model", lightgbm_regressor_cls(**lightgbm_regressor_params(horizon))),
            ]
        )
    return models


def build_classification_models(
    *,
    horizon: int | None,
    fast_mode: bool,
    imputer_factory: ImputerFactory,
    catboost_classifier_cls: type[Any] | None = None,
    lightgbm_classifier_cls: type[Any] | None = None,
) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees_clf": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_clf": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "knn_clf": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=15, weights="distance")),
            ]
        ),
        "hist_gbc": Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=5,
                        max_iter=250,
                        min_samples_leaf=12,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    if not fast_mode:
        models["mlp_clf"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
    if (not fast_mode) and catboost_classifier_cls is not None:
        models["catboost_clf"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("model", catboost_classifier_cls(**catboost_classifier_params(horizon))),
            ]
        )
    if (not fast_mode) and lightgbm_classifier_cls is not None:
        models["lightgbm_clf"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                ("model", lightgbm_classifier_cls(**lightgbm_classifier_params(horizon))),
            ]
        )
    return models


def replace_state_classifiers(
    models: dict[str, Any],
    *,
    horizon: int | None,
    fast_mode: bool,
    imputer_factory: ImputerFactory,
    catboost_classifier_cls: type[Any] | None = None,
    lightgbm_classifier_cls: type[Any] | None = None,
) -> dict[str, Any]:
    """Replace binary optional estimators with their multiclass variants."""
    if fast_mode:
        return models
    if catboost_classifier_cls is not None:
        models["catboost_clf"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    catboost_classifier_cls(
                        **catboost_classifier_params(horizon, multiclass=True)
                    ),
                ),
            ]
        )
    if lightgbm_classifier_cls is not None:
        models["lightgbm_clf"] = Pipeline(
            steps=[
                ("imputer", imputer_factory()),
                (
                    "model",
                    lightgbm_classifier_cls(
                        **lightgbm_classifier_params(horizon, multiclass=True)
                    ),
                ),
            ]
        )
    return models
