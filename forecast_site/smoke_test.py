"""End-to-end smoke test for the forecast_site/ persistence pipeline.

Uses a synthetic latest_forecast_summary.csv with target dates in the past,
so backfill_actuals can resolve them immediately against the real master
dataset and we can verify the full flow:

    persist_forecast (synthetic CSV)
        -> forecast_runs + forecasts rows written
    backfill_actuals
        -> actuals rows written + accuracy_snapshot rows written
    verify
        -> print counts, recent forecasts, accuracy numbers

Run:
    python -m forecast_site.smoke_test
"""
from __future__ import annotations

import datetime as _dt
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import eth_price_forecast as efp  # noqa: E402
from forecast_site.backfill_actuals import (  # noqa: E402
    _load_eth_close_lookup, refresh_accuracy_snapshots, resolve_pending_forecasts,
)
from forecast_site.db import connect  # noqa: E402
from forecast_site.persist_forecast import persist_forecast  # noqa: E402

MASTER_CSV = PROJECT_ROOT / "lake" / "gold" / "eth_master_daily.csv"


def _synthetic_row(horizon_days: int, input_ts: pd.Timestamp,
                   reference_price: float, target_ts: pd.Timestamp,
                   pred_return: float, prob_up: float,
                   pred_close: float) -> dict:
    return {
        "horizon_steps": horizon_days,
        "forecast_input_timestamp": input_ts.isoformat(),
        "forecast_target_timestamp": target_ts.isoformat(),
        "reference_price": reference_price,
        "reference_price_source": "smoke_test_synthetic",
        "reference_price_timestamp": input_ts.isoformat(),
        "model_input_close": reference_price,

        "regression_model_name": "smoke_regression",
        "regression_selection_basis": "synthetic",
        "regression_predicted_return": pred_return,
        "regression_predicted_close": pred_close,
        "regression_lower_return_10": pred_return - 0.05,
        "regression_lower_close_10": pred_close * 0.95,
        "regression_upper_return_90": pred_return + 0.05,
        "regression_upper_close_90": pred_close * 1.05,

        "classification_model_name": "smoke_cls",
        "classification_selection_basis": "synthetic",
        "classification_predicted_direction": "UP" if pred_return > 0.006 else (
            "DOWN" if pred_return < -0.006 else "FLAT"),
        "classification_signal_threshold": 0.55,
        "classification_probability_up": prob_up,
        "classification_probability_down": 1.0 - prob_up,
        "classification_confidence": abs(prob_up - 0.5) * 2,

        "regime_model_name": "smoke_regime",
        "regime_selection_basis": "synthetic",
        "regime_predicted_state": "SIDEWAYS",
        "regime_probability_uptrend": 0.3,
        "regime_probability_sideways": 0.5,
        "regime_probability_downtrend": 0.2,

        "reversal_model_name": "smoke_rev",
        "reversal_selection_basis": "synthetic",
        "reversal_predicted_signal": "NONE",
        "reversal_probability_bottom": 0.1,
        "reversal_probability_top": 0.1,

        "hybrid_model_name": "smoke_hybrid",
        "hybrid_predicted_direction": "UP" if pred_return > 0 else "DOWN",
        "hybrid_bias_direction": "FLAT",
        "hybrid_predicted_market_state": "SIDEWAYS",
        "hybrid_predicted_reversal_signal": "NONE",
        "hybrid_score": pred_return,
        "hybrid_confidence": 0.4,
        "hybrid_signal_tier": "NO_SIGNAL",
        "hybrid_high_confidence_signal": False,
        "hybrid_signal_reason": "smoke",
        "hybrid_volatility_scale": 0.04,
        "hybrid_scenario_spread": 0.10,

        "active_predicted_direction": "UP" if pred_return > 0 else "DOWN",
        "active_confidence": 0.4,
        "active_predicted_return": pred_return,
        "active_predicted_close": pred_close,
        "active_bear_predicted_return": pred_return - 0.08,
        "active_bear_predicted_close": pred_close * 0.92,
        "active_bull_predicted_return": pred_return + 0.08,
        "active_bull_predicted_close": pred_close * 1.08,

        "macro_readiness_status": "READY",
        "macro_risk_regime": "NEUTRAL",
        "macro_directional_bias": "FLAT",
        "macro_selection_tag": "smoke",
        "macro_risk_on_score": 0.0,
        "macro_risk_off_score": 0.0,
        "macro_trend_bias_score": 0.0,
        "macro_volatility_stress": 0.0,
        "macro_used_feature_count": 10,
        "macro_missing_feature_count": 2,
    }


def main() -> None:
    # Use a recent input_ts we can resolve: 60 days ago -> h=7 target 53 days
    # ago, h=30 target 30 days ago. Both should be in lake/gold/eth_master_daily.csv.
    today_utc = pd.Timestamp.now(tz="UTC").floor("D")
    input_ts = today_utc - pd.Timedelta(days=60)

    close_lookup = _load_eth_close_lookup(MASTER_CSV)
    if input_ts not in close_lookup.index:
        raise SystemExit(f"Master CSV missing close for input_ts={input_ts.date()}")
    reference_price = float(close_lookup.loc[input_ts])
    print(f"[smoke] input_ts={input_ts.date()} reference_price=${reference_price:,.2f}")

    rows = [
        _synthetic_row(
            horizon_days=7, input_ts=input_ts, reference_price=reference_price,
            target_ts=input_ts + pd.Timedelta(days=7),
            pred_return=0.02, prob_up=0.65, pred_close=reference_price * 1.02,
        ),
        _synthetic_row(
            horizon_days=30, input_ts=input_ts, reference_price=reference_price,
            target_ts=input_ts + pd.Timedelta(days=30),
            pred_return=0.05, prob_up=0.58, pred_close=reference_price * 1.05,
        ),
    ]
    df = pd.DataFrame(rows)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "latest_forecast_summary.csv"
        df.to_csv(csv_path, index=False)

        # Use a throwaway DB so we don't pollute the real one.
        db_path = tmp_path / "smoke_predictions.db"

        print("\n[smoke] -- persist_forecast --")
        run_id = persist_forecast(
            csv_path,
            model_phase="smoke_test",
            db_path=db_path,
            code_source=Path(efp.__file__),
            fast_mode=False,
            notes="pipeline smoke test",
        )
        assert run_id >= 1, run_id

        print("\n[smoke] -- re-persist (idempotency check) --")
        run_id_again = persist_forecast(
            csv_path, model_phase="smoke_test", db_path=db_path,
            code_source=Path(efp.__file__),
        )
        assert run_id == run_id_again, f"expected {run_id}, got {run_id_again}"

        print("\n[smoke] -- backfill_actuals --")
        conn = connect(db_path)
        try:
            resolved = resolve_pending_forecasts(conn, close_lookup)
            print(f"[smoke] resolved: {resolved}")
            assert resolved == 2, f"expected 2 resolved, got {resolved}"
            snapshots = refresh_accuracy_snapshots(conn)
            print(f"[smoke] accuracy snapshots: {snapshots}")

            print("\n[smoke] -- verify --")
            n_runs = conn.execute("SELECT COUNT(*) FROM forecast_runs").fetchone()[0]
            n_forecasts = conn.execute("SELECT COUNT(*) FROM forecasts").fetchone()[0]
            n_actuals = conn.execute("SELECT COUNT(*) FROM actuals").fetchone()[0]
            n_snaps = conn.execute("SELECT COUNT(*) FROM accuracy_snapshot").fetchone()[0]
            print(f"[smoke] rows: runs={n_runs} forecasts={n_forecasts} "
                  f"actuals={n_actuals} snapshots={n_snaps}")
            assert n_runs == 1
            assert n_forecasts == 2
            assert n_actuals == 2

            print("\n[smoke] -- per-horizon accuracy snapshot (window_days=9999) --")
            for row in conn.execute(
                "SELECT horizon_days, window_days, resolved_count, "
                "direction_accuracy, brier_score, price_mape_percent, price_rmse, return_mae "
                "FROM accuracy_snapshot ORDER BY horizon_days, window_days"
            ):
                print(f"  h={row['horizon_days']:>2} window={row['window_days']:>4d} "
                      f"n={row['resolved_count']} "
                      f"dir_acc={row['direction_accuracy']} "
                      f"brier={row['brier_score']} "
                      f"mape={row['price_mape_percent']}")

            print("\n[smoke] -- per-horizon actuals --")
            for row in conn.execute(
                "SELECT f.horizon_days, a.actual_close, a.actual_return, "
                "a.direction_actual, a.direction_correct, "
                "a.price_absolute_error, a.brier_contribution "
                "FROM actuals a JOIN forecasts f ON f.forecast_id = a.forecast_id "
                "ORDER BY f.horizon_days"
            ):
                print(f"  h={row['horizon_days']:>2} close=${row['actual_close']:,.2f} "
                      f"return={row['actual_return']:+.4%} "
                      f"actual_dir={row['direction_actual']} "
                      f"correct={row['direction_correct']} "
                      f"price_ae=${row['price_absolute_error']:,.2f} "
                      f"brier={row['brier_contribution']:.4f}")

        finally:
            conn.close()

    print("\n[smoke] PASS")


if __name__ == "__main__":
    main()
