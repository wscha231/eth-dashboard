"""Public hot-branch policy for source files with restricted redistribution rights."""
from __future__ import annotations

from pathlib import PurePosixPath

PUBLIC_HOT_BRANCHES = {"data/daily-forecast", "data/event-feed"}
RESTRICTED_DERIVATIVE_CACHES = frozenset({
    "lake/raw/vendor/bitget_eth_context_4h.csv",
    "lake/raw/vendor/bitget_eth_free_features.csv",
    "lake/raw/vendor/hyperliquid_eth_context_4h.csv",
    "lake/raw/vendor/hyperliquid_eth_free_features.csv",
    "lake/raw/vendor/deribit_eth_dvol_daily.csv",
    "lake/raw/vendor/deribit_eth_funding_daily.csv",
    "lake/raw/vendor/deribit_eth_future_snapshot_daily.csv",
    "lake/raw/vendor/deribit_eth_historical_volatility.csv",
    "lake/raw/vendor/deribit_eth_option_snapshot_daily.csv",
})


def normalize(path: str) -> str:
    value = PurePosixPath(path.replace("\\", "/"))
    if value.is_absolute() or not value.parts or any(part in {"", ".", ".."} for part in value.parts):
        raise ValueError(f"unsafe repository-relative path: {path}")
    return value.as_posix()


def blocked_on_public_hot_branch(path: str, branch: str = "data/daily-forecast") -> bool:
    return branch in PUBLIC_HOT_BRANCHES and normalize(path) in RESTRICTED_DERIVATIVE_CACHES
