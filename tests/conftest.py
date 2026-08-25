"""Shared pytest fixtures for the ETH forecast test suite."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def synthetic_ohlcv() -> pd.DataFrame:
    """Deterministic synthetic ETH-like OHLCV, long enough for 200+ rolling windows."""
    rng = np.random.default_rng(42)
    n = 1200
    idx = pd.date_range("2021-01-01", periods=n, freq="D")
    log_returns = rng.normal(0.0005, 0.035, n)
    close = pd.Series(np.exp(np.cumsum(log_returns)) * 2000.0, index=idx, name="eth_close")
    high_mult = 1.0 + np.abs(rng.normal(0.0, 0.015, n))
    low_mult = 1.0 - np.abs(rng.normal(0.0, 0.015, n))
    open_mult = 1.0 + rng.normal(0.0, 0.008, n)
    volume = rng.lognormal(mean=20.0, sigma=0.5, size=n)
    frame = pd.DataFrame(
        {
            "eth_open": (close * open_mult).values,
            "eth_high": (close * high_mult).values,
            "eth_low": (close * low_mult).values,
            "eth_close": close.values,
            "eth_volume": volume,
        },
        index=idx,
    )
    frame["eth_high"] = frame[["eth_high", "eth_close", "eth_open"]].max(axis=1)
    frame["eth_low"] = frame[["eth_low", "eth_close", "eth_open"]].min(axis=1)
    return frame


@pytest.fixture(scope="session")
def synthetic_ohlcv_with_companions(synthetic_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Extend the synthetic OHLCV with BTC/SPY-style companion columns build_features expects."""
    rng = np.random.default_rng(7)
    frame = synthetic_ohlcv.copy()
    n = len(frame)
    for alias, base in (("btc", 40000.0), ("spy", 450.0), ("qqq", 380.0), ("dxy", 104.0), ("tnx", 4.2), ("oil", 80.0)):
        close = np.exp(np.cumsum(rng.normal(0.0, 0.02, n))) * base
        frame[f"{alias}_close"] = close
        frame[f"{alias}_open"] = close * (1.0 + rng.normal(0.0, 0.005, n))
        frame[f"{alias}_high"] = close * (1.0 + np.abs(rng.normal(0.0, 0.01, n)))
        frame[f"{alias}_low"] = close * (1.0 - np.abs(rng.normal(0.0, 0.01, n)))
        frame[f"{alias}_volume"] = rng.lognormal(18.0, 0.4, n)
    return frame


# ---------------------------------------------------------------------------
# DB fixtures for persist_backtest / export_backtest_json tests.
#
# We use tmp_path (function-scoped) rather than :memory: because
# forecast_site.db.connect() runs a PRAGMA journal_mode=WAL that is a no-op
# on :memory: DBs — and we want the test to exercise the real code path a
# production caller hits. File-per-test is cheap (WAL mode + <1MB of DDL).
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_db_path(tmp_path: Path) -> Path:
    """Isolated DB path per test. The file is auto-cleaned by tmp_path.

    Returns the path, not the connection — some tests (persist_backtest CLI,
    full-driver tests) need to pass --db to a subprocess or re-open with a
    fresh connection, and a path is the lowest-common-denominator handle."""
    return tmp_path / "test_backtest.db"


@pytest.fixture
def temp_db(temp_db_path: Path):
    """A live sqlite3.Connection with the v3 schema already applied.

    Yields so teardown closes the connection cleanly (WAL journal files
    get cleaned up on close; a leaked connection would leave -wal/-shm
    siblings in the tmp dir that surprise later tests if tmp_path happens
    to get reused — which it won't because tmp_path is per-test, but
    closing is the right hygiene regardless)."""
    from forecast_site.db import connect
    conn = connect(temp_db_path)
    try:
        yield conn
    finally:
        conn.close()


def _make_tiny_freeze_payload(
    *,
    mode: str = "phase0_test_synthetic",
    frozen_at: str = "2026-01-01T00:00:00+00:00",
    n_models: int = 2,
    include_predictions: bool = False,
    n_pred_dates: int = 4,
) -> dict:
    """Hand-rolled freeze JSON — smaller than the real 13KB baseline fixture
    (100x faster test loop) and lets individual tests toggle longrun shape,
    NaN injection, etc. without coupling to the real freeze pipeline."""
    horizons: dict = {}
    for h in (7, 30):
        reg_rows = [
            {
                "model":                f"model_{i}",
                "folds":                3.0,
                "component_models":     None,
                "return_mae":           0.05 + 0.01 * i,
                "price_mae":            100.0 + 5 * i,
                "price_rmse":           150.0 + 5 * i,
                "price_mape_percent":   5.0 + 0.5 * i,
                "directional_accuracy": 0.55 - 0.02 * i,
            }
            for i in range(n_models)
        ]
        cls_rows = [
            {
                "model":                f"clf_{i}",
                "folds":                3.0,
                "component_models":     None,
                "accuracy":             0.52 + 0.01 * i,
                "balanced_accuracy":    0.51 + 0.01 * i,
                "precision":            0.53,
                "recall":               0.49,
                "f1":                   0.51,
                "roc_auc":              0.54,
                "brier_score":          0.25 - 0.001 * i,
                "signal_threshold":     0.5,
                "threshold_validation_scheme": "oof",
                "validation_leakage_safe": True,
                "directional_accuracy": 0.52,
            }
            for i in range(n_models)
        ]
        payload_h: dict = {
            "regression_leaderboard":     reg_rows,
            "classification_leaderboard": cls_rows,
        }
        if include_predictions:
            payload_h["predictions"] = [
                {
                    "horizon_days":     h,
                    "head":             "regression",
                    "model":            "model_0",
                    "prediction_date":  f"2024-01-{d+1:02d}",
                    "target_date":      f"2024-01-{d+1+h:02d}",
                    "fold_index":       d // 2,
                    "reference_close":  2000.0 + 10 * d,
                    "actual_close":     2020.0 + 10 * d,
                    "actual_return":    0.01,
                    "actual_label":     1,
                    "predicted_return": 0.015,
                    "predicted_close":  2030.0 + 10 * d,
                }
                for d in range(n_pred_dates)
            ] + [
                {
                    "horizon_days":     h,
                    "head":             "classification",
                    "model":            "clf_0",
                    "prediction_date":  f"2024-01-{d+1:02d}",
                    "target_date":      f"2024-01-{d+1+h:02d}",
                    "fold_index":       d // 2,
                    "actual_label":     1,
                    "probability_up":   0.6,
                    "direction_score_up": 0.65,
                    "predicted_label":  1,
                }
                for d in range(n_pred_dates)
            ]
        horizons[str(h)] = payload_h
    return {
        "mode":                       mode,
        "frozen_at":                  frozen_at,
        "master_data_csv":            "lake/gold/test_synthetic.csv",
        "eth_price_forecast_bytes":   123456,
        "cv_test_size":               30 if "longrun" in mode else 60,
        "horizons":                   horizons,
    }


@pytest.fixture
def tiny_freeze_json(tmp_path: Path):
    """Factory that writes a freeze JSON to tmp_path and returns the path.

    Returns a callable so tests can produce multiple variants (different
    phase slugs, longrun vs not, with/without predictions) within the
    same test if needed — the filename naming controls what persist_backtest
    treats as the model_phase slug."""
    import json

    def _make(
        name: str = "phase0_test_metrics.json",
        **kwargs,
    ) -> Path:
        payload = _make_tiny_freeze_payload(**kwargs)
        path = tmp_path / name
        path.write_text(json.dumps(payload), encoding="utf-8")
        return path

    return _make
