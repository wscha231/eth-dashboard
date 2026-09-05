"""UTC daily-bar contract shared by collection, inference and settlement.

Source rows are labelled by bar START. Model rows are labelled by bar END.
The grace period is a provider delay allowance, not proof of an exchange finality
flag. Provenance records the exact snapshot and observed availability time.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pandas as pd

TIME_CONTRACT = "utc_bar_end_v2"
GRACE = pd.Timedelta(minutes=15)


def utc_timestamp(value=None) -> pd.Timestamp:
    stamp = pd.Timestamp.now(tz="UTC") if value is None else pd.Timestamp(value)
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def iso_utc(value) -> str:
    return utc_timestamp(value).isoformat().replace("+00:00", "Z")


def closed_daily_rows(frame: pd.DataFrame, *, now=None) -> pd.DataFrame:
    """Keep only elapsed source bars; never interpolate missing ETH dates."""
    result = frame.copy()
    dates = pd.to_datetime(result.index, utc=True)
    if dates.has_duplicates or not (dates == dates.floor("D")).all():
        raise ValueError("Daily source must have unique UTC midnight bar-start labels")
    result = result.loc[dates + pd.Timedelta(days=1) + GRACE <= utc_timestamp(now)].sort_index()
    return result


def model_daily_rows(frame: pd.DataFrame, *, now=None, require_fresh=True) -> pd.DataFrame:
    result = closed_daily_rows(frame, now=now)
    if result.empty or "eth_close" not in result:
        raise ValueError("No closed ETH daily bars")
    dates = pd.to_datetime(result.index, utc=True)
    prices = pd.to_numeric(result["eth_close"], errors="coerce")
    if not prices.map(lambda value: pd.notna(value) and math.isfinite(value) and value > 0).all():
        raise ValueError("ETH close is missing, non-finite or non-positive")
    if len(dates) > 1 and (dates.to_series().diff().dropna() != pd.Timedelta(days=1)).any():
        raise ValueError("ETH daily history contains a gap; repair source data before forecasting")
    origin = dates[-1] + pd.Timedelta(days=1)
    if require_fresh and utc_timestamp(now) - origin > pd.Timedelta(hours=27):
        raise ValueError("Latest closed ETH daily bar is stale")
    result.index = (dates + pd.Timedelta(days=1)).tz_localize(None)
    result.attrs.update(time_contract=TIME_CONTRACT, source_bar_date=str(dates[-1].date()),
                        observed_at_utc=iso_utc(now), origin_time_utc=iso_utc(origin))
    return result


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
