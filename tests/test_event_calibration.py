import numpy as np
import pandas as pd
import pytest

from research_pipeline import event_calibration as study


def training_fixture():
    index = pd.date_range("2023-01-01", "2025-01-02", freq="h", tz="UTC")
    features = pd.DataFrame({"x": np.sin(np.arange(len(index))), "reference_price": 100.,
                             "sigma": .01, "available_at": index}, index=index)
    outcomes = pd.DataFrame({"return": .01, "up": (np.arange(len(index)) // 6) % 3 == 0,
                             "down": (np.arange(len(index)) // 6) % 5 == 0, "terminal": 1,
                             "target_end": index + pd.Timedelta(hours=7)}, index=index)
    return features, outcomes


def fake_models(monkeypatch):
    fits = []

    def fit(features, outcomes, indices):
        # An intentionally constant prediction that changes when the model refits.
        fits.append(indices.copy())
        return {"p": .2 + indices[-1].day / 100, "fit_end": indices[-1]}

    def predict(model, features):
        return {"climatology": {"path": np.full((len(features), 2), model["p"]),
                                 "terminal": np.tile([.2, .6, .2], (len(features), 1))}}

    monkeypatch.setattr(study, "fit_models", fit)
    monkeypatch.setattr(study, "predict_models", predict)
    return fits


def test_final_model_calibration_is_purged_and_constant_probabilities_never_alert(monkeypatch):
    fits = fake_models(monkeypatch)
    x, y = training_fixture()
    bundle = study.train_calibrated_bundle(x, y, "2025-01-01", 6)
    assert len(fits) == 2  # No third fit after calibration.
    for threshold in bundle["alert_thresholds"].values():
        assert threshold == bundle["model"]["p"]
        assert not bundle["model"]["p"] > threshold
    calibration_start = pd.Timestamp(bundle["calibration_start"])
    assert y.loc[fits[1], "target_end"].max() < calibration_start - pd.Timedelta(hours=1)
    assert pd.Timestamp(bundle["validation_target_end"]) < calibration_start - pd.Timedelta(hours=1)
    assert pd.Timestamp(bundle["calibration_target_end"]) < pd.Timestamp("2025-01-01", tz="UTC") - pd.Timedelta(hours=1)
    # Inner and final predictions really differ, recreating the original failure mode.
    assert fits[0][-1].day != fits[1][-1].day


def test_calibration_labels_do_not_select_or_fit_model_and_future_data_cannot_change_bundle(monkeypatch):
    fits = fake_models(monkeypatch)
    x, y = training_fixture()
    original = study.train_calibrated_bundle(x, y, "2025-01-01", 6)
    original_fits = [i.copy() for i in fits]
    modified = y.copy()
    modified.loc[modified.index >= original["calibration_start"], ["up", "down"]] = True
    changed = study.train_calibrated_bundle(x, modified, "2025-01-01", 6)
    assert changed["model"] == original["model"]
    assert changed["choice"] == original["choice"]
    assert changed["validation_scores"] == original["validation_scores"]
    assert changed["alert_thresholds"] == {"up": 1., "down": 1.}
    for first, second in zip(original_fits, fits[2:]):
        assert first.equals(second)
    future = x.copy()
    future.loc[future.index >= "2025-01-01", "x"] = 9999.
    modified = y.copy()
    modified.loc[modified.target_end >= pd.Timestamp("2025-01-01", tz="UTC"), "return"] = -5.
    assert study.train_calibrated_bundle(future, modified, "2025-01-01", 6) == original


def derivative_fixture():
    index = pd.date_range("2024-01-01", "2024-01-15", freq="h", tz="UTC")
    funding = pd.DataFrame({"event_time": index, "interest_1h": .001, "interest_8h": 99.,
                            "received_at": pd.Timestamp("2026-09-07", tz="UTC")})
    days = pd.date_range("2024-01-01", "2024-01-15", freq="D", tz="UTC")
    dvol = pd.DataFrame({"event_end": days, "dvol_close_pct": np.arange(len(days)) + 40.,
                         "received_at": pd.Timestamp("2026-09-07", tz="UTC")})
    return index, funding, dvol


def test_historical_inputs_require_acknowledgement_and_never_backdate_receipts():
    index, funding, dvol = derivative_fixture()
    original = funding.copy(deep=True)
    with pytest.raises(ValueError, match="acknowledged"):
        study.historical_derivative_features(index, funding, dvol)
    out = study.historical_derivative_features(index, funding, dvol, reconstruction=True)
    assert out.loc["2024-01-10T00:00Z", "funding_sum_24h"] == pytest.approx(.024)
    assert out.loc["2024-01-10T00:00Z", "dvol_level"] == 48.
    assert out.loc["2024-01-10T01:00Z", "dvol_level"] == 49.
    pd.testing.assert_frame_equal(funding, original)


def test_missing_hour_or_day_is_not_filled_and_future_intervals_do_not_leak():
    index, funding, dvol = derivative_fixture()
    original = study.historical_derivative_features(index, funding, dvol, reconstruction=True)
    missing = funding[funding.event_time != pd.Timestamp("2024-01-10T00:00Z")]
    result = study.historical_derivative_features(index, missing, dvol, reconstruction=True)
    assert pd.isna(result.loc["2024-01-10T01:00Z", "funding_sum_24h"])
    assert result.loc["2024-01-11T01:00Z", "funding_sum_24h"] == pytest.approx(.024)
    missing_day = dvol[dvol.event_end != pd.Timestamp("2024-01-10T00:00Z")]
    result = study.historical_derivative_features(index, funding, missing_day, reconstruction=True)
    assert pd.isna(result.loc["2024-01-10T01:00Z", "dvol_level"])
    cutoff = pd.Timestamp("2024-01-10T12:00Z")
    funding.loc[funding.event_time >= cutoff, "interest_1h"] = 10.
    dvol.loc[dvol.event_end >= cutoff, "dvol_close_pct"] = 999.
    changed = study.historical_derivative_features(index, funding, dvol, reconstruction=True)
    pd.testing.assert_frame_equal(original.loc[:cutoff], changed.loc[:cutoff])


def execution_fixture():
    index = pd.date_range("2024-01-01T00:00Z", periods=8, freq="h")
    bars = pd.DataFrame({"product": "ETH-USD", "open_time": index,
                         "open": [90., 100., 100., 100., 110., 110., 110., 110.],
                         "close": [100., 105., 95., 110., 110., 110., 110., 110.]})
    row = {"slot": index[0].isoformat(), "target_end": index[4].isoformat(),
           "hit_up": .8, "hit_down": .2, "threshold_up": .7, "threshold_down": .7}
    return index, bars, row


def test_execution_uses_next_hour_open_with_two_sided_costs_and_no_overlapping_positions():
    index, bars, row = execution_fixture()
    overlapping = {**row, "slot": index[1].isoformat(), "target_end": index[5].isoformat()}
    result = study.long_cash_diagnostic([row, overlapping], bars, index[0], index[-1], 25)
    assert result["trades"] == 1
    assert result["invested_hours"] == 3
    assert result["total_return"] == pytest.approx(1.1 * .9975 / 1.0025 - 1)
    assert result["max_drawdown_hourly"] == pytest.approx(95 / 105 - 1)
    assert result["buy_hold_total_return"] == pytest.approx(110 / 90 * .9975 / 1.0025 - 1)


def test_both_alerts_do_not_trade_and_incomplete_execution_history_cannot_report_a_return():
    index, bars, row = execution_fixture()
    result = study.long_cash_diagnostic([{**row, "hit_down": .8}], bars, index[0], index[-1], 10)
    assert result["trades"] == 0
    assert result["total_return"] == 0
    result = study.long_cash_diagnostic([row], bars.drop(index=2), index[0], index[-1], 10)
    assert result["status"] == "incomplete_execution_prices"
    assert "total_return" not in result
