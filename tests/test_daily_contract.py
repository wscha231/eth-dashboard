from __future__ import annotations

import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from forecast_site.backfill_actuals import _resolve_actual_close, resolve_pending_forecasts, refresh_accuracy_snapshots
from forecast_site.export_json import export_accuracy
from forecast_site.persist_forecast import persist_forecast
from forecasting.daily_data import TIME_CONTRACT, closed_daily_rows, model_daily_rows
from forecasting.model_bundle import cached_final_fit, final_model_cache, save_bundle, load_bundle
from forecasting.feature_store import cached_fold_selection, fold_feature_cache


def prices():
    return pd.Series([100.0, 120.0], index=pd.to_datetime(["2026-09-03", "2026-09-05"], utc=True))


def test_current_day_and_gap_do_not_settle():
    assert _resolve_actual_close(prices(), pd.Timestamp("2026-09-05", tz="UTC"), now="2026-09-05T15:00Z") is None
    assert _resolve_actual_close(prices(), pd.Timestamp("2026-09-04", tz="UTC"), now="2026-09-07T15:00Z") is None
    assert _resolve_actual_close(prices(), pd.Timestamp("2026-09-05", tz="UTC"), now="2026-09-06T00:14Z") is None
    assert _resolve_actual_close(prices(), pd.Timestamp("2026-09-05", tz="UTC"), now="2026-09-06T00:15Z") == 120


def test_new_contract_uses_source_day_before_target_end():
    assert _resolve_actual_close(prices(), pd.Timestamp("2026-09-04", tz="UTC"), now="2026-09-05T15:00Z", time_contract=TIME_CONTRACT) == 100


def test_model_rows_shift_once_and_reject_gaps():
    data = pd.DataFrame({"eth_close": [100, 101, 150]}, index=pd.date_range("2026-09-03", periods=3))
    result = model_daily_rows(data, now="2026-09-05T01:00Z")
    assert list(result.eth_close) == [100, 101]
    assert result.index[-1] == pd.Timestamp("2026-09-05")
    assert result.attrs["source_bar_date"] == "2026-09-04"
    with pytest.raises(ValueError, match="gap"):
        model_daily_rows(data.iloc[[0, 2]], now="2026-09-06T01:00Z")


def seed(conn, *, target="2026-09-03", contract=None):
    run = conn.execute("INSERT INTO forecast_runs (run_timestamp_utc,input_timestamp_utc,reference_price,model_phase) VALUES ('2026-08-01','2026-08-01',100,'test')").lastrowid
    forecast = conn.execute("INSERT INTO forecasts (run_id,horizon_days,forecast_target_timestamp_utc,regression_predicted_return,regression_predicted_close,classification_predicted_direction,classification_probability_up,time_contract) VALUES (?,30,?,0,100,'UP',0.8,?)", (run, target, contract)).lastrowid
    return forecast


def test_correction_preserves_forecast_and_is_idempotent(temp_db):
    forecast = seed(temp_db)
    temp_db.execute("INSERT INTO actuals (forecast_id,resolved_at_utc,actual_close,actual_return,direction_actual) VALUES (?,'2026-09-03T01:00Z',90,-0.1,'DOWN')", (forecast,))
    before = tuple(temp_db.execute("SELECT * FROM forecasts").fetchone())
    assert resolve_pending_forecasts(temp_db, prices(), now="2026-09-05T12:00Z") == 1
    row = dict(temp_db.execute("SELECT * FROM actuals").fetchone())
    assert row["actual_close"] == 100
    assert row["brier_contribution"] is None  # incompatible legacy probability event
    archived = json.loads(temp_db.execute("SELECT previous_actual_json FROM actuals_revision").fetchone()[0])
    assert archived["actual_close"] == 90
    assert resolve_pending_forecasts(temp_db, prices(), now="2026-09-05T13:00Z") == 0
    assert temp_db.execute("SELECT COUNT(*) FROM actuals_revision").fetchone()[0] == 1
    assert tuple(temp_db.execute("SELECT * FROM forecasts").fetchone()) == before
    refresh_accuracy_snapshots(temp_db)
    all_rows = export_accuracy(temp_db)["horizons"]["30"]["all"]
    assert all_rows["signal_count"] == 1 and all_rows["resolved_count"] == 1


def test_early_legacy_actual_is_archived_and_excluded(temp_db):
    forecast = seed(temp_db, target="2026-09-05")
    temp_db.execute("INSERT INTO actuals (forecast_id,resolved_at_utc,actual_close,actual_return,direction_actual) VALUES (?,'2026-09-05T01:00Z',120,0.2,'UP')", (forecast,))
    resolve_pending_forecasts(temp_db, prices(), now="2026-09-05T12:00Z")
    assert temp_db.execute("SELECT COUNT(*) FROM actuals").fetchone()[0] == 0
    assert temp_db.execute("SELECT COUNT(*) FROM actuals_revision").fetchone()[0] == 1


def test_partial_summary_never_creates_run(tmp_path):
    csv = tmp_path / "partial.csv"
    pd.DataFrame([{"horizon_steps":7, "forecast_input_timestamp":"2026-09-05", "forecast_target_timestamp":"2026-09-12", "reference_price":100}]).to_csv(csv,index=False)
    db = tmp_path / "partial.db"
    with pytest.raises(SystemExit, match="required horizon"):
        persist_forecast(csv, model_phase="test", db_path=db, required_horizons=(7,30))
    assert not db.exists()


@pytest.mark.parametrize("prices", [[100, float("inf")], [100, -1], [100, None]])
def test_invalid_production_price_never_creates_run(tmp_path, prices):
    csv = tmp_path / "invalid.csv"
    pd.DataFrame([{"horizon_steps": h, "forecast_input_timestamp": "2026-09-05",
                   "forecast_target_timestamp": target, "reference_price": 100,
                   "regression_predicted_close": price}
                  for h, target, price in zip([7, 30], ["2026-09-12", "2026-10-05"], prices)]).to_csv(csv, index=False)
    db = tmp_path / "invalid.db"
    with pytest.raises(SystemExit, match="finite positive"):
        persist_forecast(csv, model_phase="test", db_path=db, required_horizons=(7, 30))
    assert not db.exists()


def test_cached_real_estimator_predictions_roundtrip(tmp_path):
    from sklearn.linear_model import Ridge
    calls = []
    @cached_final_fit
    def fit(dataset, feature_columns, model_name, horizon, sample_weight=None):
        calls.append(1)
        return Ridge().fit(dataset[feature_columns], dataset.target_return)
    data = pd.DataFrame({"x":np.arange(40),"target_return":np.sin(np.arange(40))})
    models = {}
    with final_model_cache("record", models):
        direct = fit(data, ["x"], "ridge", 7).predict(pd.DataFrame({"x":[41,42]}))
    path = tmp_path / "bundle.joblib"
    save_bundle({"horizons":{7:{"models":models},30:{}},"metadata":{}}, path)
    restored = load_bundle(path)
    with final_model_cache("predict", restored["horizons"][7]["models"]):
        np.testing.assert_array_equal(fit(data,["x"],"ridge",7).predict(pd.DataFrame({"x":[41,42]})), direct)
        with pytest.raises(RuntimeError, match="uncached"):
            fit(data.assign(target_return=0),["x"],"ridge",7)
    assert len(calls) == 1
    path.write_bytes(path.read_bytes()+b"modified")
    with pytest.raises(ValueError,match="checksum"):
        load_bundle(path)


def test_fold_cache_uses_training_values_and_target_only():
    calls=[]
    @cached_fold_selection
    def select(dataset,candidate_feature_columns,train_positions,min_feature_coverage,horizon=None,target_column="target_return"):
        calls.append(1)
        return candidate_feature_columns[:1]
    data=pd.DataFrame({"x":[1,2,3],"target_return":[1,2,3]})
    with fold_feature_cache():
        select(data,["x"],np.array([0,1]),.03,7)
        future_changed=data.copy();future_changed.loc[2,"x"]=9999
        select(future_changed,["x"],np.array([0,1]),.03,7)
        assert len(calls)==1
        target_changed=data.copy();target_changed.loc[0,"target_return"]=9999
        select(target_changed,["x"],np.array([0,1]),.03,7)
        assert len(calls)==2
