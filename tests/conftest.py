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
