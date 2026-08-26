from __future__ import annotations

import math

import pandas as pd

import eth_price_forecast as efp


def test_append_no_change_anchor_adds_first_class_regression_candidate() -> None:
    index = pd.date_range("2024-01-01", periods=4, freq="D")
    dataset = pd.DataFrame(
        {
            "eth_close": [100.0, 100.0, 100.0, 100.0],
            "target_return": [0.10, -0.08, 0.03, -0.02],
        },
        index=index,
    )
    leaderboard = pd.DataFrame(
        [
            {
                "model": "bad_model",
                "price_rmse": 50.0,
                "price_mae": 50.0,
                "directional_accuracy": 0.25,
            }
        ]
    )

    updated_leaderboard, updated_oof = efp.append_no_change_regression_anchor(
        leaderboard,
        pd.DataFrame(index=index),
        dataset,
        folds=3,
    )

    anchor = updated_leaderboard.loc[
        updated_leaderboard["model"] == efp.NO_CHANGE_ANCHOR_MODEL
    ].iloc[0]
    assert anchor["forecast_role"] == "point_anchor"
    assert anchor["price_rmse"] < 10.0
    assert f"{efp.NO_CHANGE_ANCHOR_MODEL}_pred_return" in updated_oof.columns
    assert updated_oof[f"{efp.NO_CHANGE_ANCHOR_MODEL}_pred_return"].dropna().eq(0.0).all()


def test_30d_selection_uses_full_oof_promoted_anchor() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": efp.NO_CHANGE_ANCHOR_MODEL,
                "price_rmse": 100.0,
                "price_mae": 80.0,
                "directional_accuracy": 0.50,
            },
            {
                "model": "hist_gbr",
                "price_rmse": 100.1,
                "price_mae": 79.0,
                "directional_accuracy": 0.55,
            },
        ]
    )

    model_name, basis = efp.select_regression_forecast_model(
        leaderboard,
        horizon=30,
        latest_features=pd.Series(dtype=float),
    )

    assert model_name == efp.NO_CHANGE_ANCHOR_MODEL
    assert basis.startswith("promoted_full_oof_champion[no_change_anchor]")


def test_short_cv_cannot_displace_promoted_30d_point_champion() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": efp.NO_CHANGE_ANCHOR_MODEL,
                "price_rmse": 100.0,
                "price_mae": 80.0,
                "directional_accuracy": 0.0,
            },
            {
                "model": "extra_trees",
                "price_rmse": 50.0,
                "price_mae": 40.0,
                "directional_accuracy": 0.60,
            },
        ]
    )

    model_name, basis = efp.select_regression_forecast_model(
        leaderboard,
        horizon=30,
        latest_features=pd.Series(dtype=float),
    )

    assert model_name == efp.NO_CHANGE_ANCHOR_MODEL
    assert "promoted_full_oof_champion[no_change_anchor]" in basis


def test_interval_model_selection_excludes_the_point_anchor() -> None:
    leaderboard = pd.DataFrame(
        [
            {
                "model": efp.NO_CHANGE_ANCHOR_MODEL,
                "price_rmse": 90.0,
                "price_mae": 75.0,
                "directional_accuracy": 0.50,
            },
            {
                "model": "ridge",
                "price_rmse": 100.0,
                "price_mae": 80.0,
                "directional_accuracy": 0.55,
            },
        ]
    )

    model_name, basis = efp.select_regression_interval_model(
        leaderboard,
        horizon=30,
        latest_features=pd.Series(dtype=float),
    )

    assert model_name == "ridge"
    assert model_name != efp.NO_CHANGE_ANCHOR_MODEL
    assert basis


def test_anchor_forecast_returns_reference_close_with_empirical_interval() -> None:
    index = pd.date_range("2024-01-01", periods=8, freq="D")
    training_dataset = pd.DataFrame(
        {
            "target_return": [-0.12, -0.08, -0.02, 0.00, 0.03, 0.07, 0.11, 0.15],
            "eth_close": [100.0] * 8,
        },
        index=index,
    )
    prediction_frame = pd.DataFrame({"eth_close": [108.0]}, index=[pd.Timestamp("2024-01-09")])
    price_reference = efp.PriceReference(price=110.0, timestamp="2024-01-09 00:00:00", source="test")

    forecast = efp.forecast_next_step(
        training_dataset=training_dataset,
        prediction_frame=prediction_frame,
        feature_columns=[],
        interval="1d",
        horizon=30,
        model_name=efp.NO_CHANGE_ANCHOR_MODEL,
        selection_basis="unit_test",
        price_reference=price_reference,
    )

    assert forecast.model_name == efp.NO_CHANGE_ANCHOR_MODEL
    assert forecast.predicted_return == 0.0
    assert forecast.predicted_close == 110.0
    assert forecast.lower_return_10 <= 0.0 <= forecast.upper_return_90
    assert math.isfinite(forecast.lower_close_10)
    assert math.isfinite(forecast.upper_close_90)


def test_anchor_point_forecast_keeps_an_independent_learned_interval() -> None:
    index = pd.date_range("2024-01-01", periods=100, freq="D")
    feature = pd.Series(range(100), index=index, dtype=float)
    training_dataset = pd.DataFrame(
        {
            "x": feature,
            "target_return": (feature - feature.mean()) / 2000.0,
            "eth_close": 100.0 + feature,
        },
        index=index,
    )
    prediction_frame = pd.DataFrame(
        {"x": [100.0], "eth_close": [200.0]},
        index=[pd.Timestamp("2024-04-10")],
    )
    price_reference = efp.PriceReference(
        price=205.0,
        timestamp="2024-04-10 00:00:00",
        source="test",
    )

    forecast = efp.forecast_next_step(
        training_dataset=training_dataset,
        prediction_frame=prediction_frame,
        feature_columns=["x"],
        interval="1d",
        horizon=30,
        model_name=efp.NO_CHANGE_ANCHOR_MODEL,
        selection_basis="unit_point",
        price_reference=price_reference,
        interval_model_name="ridge",
        interval_selection_basis="unit_interval",
    )

    assert forecast.model_name == efp.NO_CHANGE_ANCHOR_MODEL
    assert forecast.predicted_return == 0.0
    assert forecast.predicted_close == 205.0
    assert "independent_conformal_interval[ridge]" in forecast.selection_basis
    assert forecast.lower_return_10 <= 0.0 <= forecast.upper_return_90
    assert math.isfinite(forecast.lower_close_10)
    assert math.isfinite(forecast.upper_close_90)
