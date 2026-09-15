"""Free, public research sources with explicit timing and units.

This module is deliberately independent from production issuance. It collects
only public market/economic observations and returns deterministic pandas
frames. Callers decide whether/when a source is eligible for model use.

Historical backfills are research evidence unless an original historical
receipt/vintage is available. Economic initial-release dates are stored
separately from observation dates; date-only release timing is made
conservative by exposing the value from the following UTC day.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

BYBIT_BASE = "https://api.bybit.com"
FRED_BASE = "https://api.stlouisfed.org/fred"
DEFAULT_BYBIT_SYMBOL = "ETHUSDT"
DEFAULT_FRED_INITIAL_SERIES = {
    "M2SL": "alfred_m2_initial",
    "CPIAUCSL": "alfred_cpi_initial",
    "UNRATE": "alfred_unemployment_initial",
    "WALCL": "alfred_fed_balance_sheet_initial",
    "WTREGEN": "alfred_tga_initial",
    "RRPONTSYD": "alfred_rrp_initial",
    "SOFR": "alfred_sofr_initial",
    "DFF": "alfred_fed_funds_initial",
    "T10Y2Y": "alfred_yield_curve_10y_2y_initial",
    "BAMLH0A0HYM2": "alfred_hy_oas_initial",
}
SOURCE_FETCH_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError)


def utc(value: Any) -> pd.Timestamp:
    t = pd.Timestamp(value)
    if pd.isna(t):
        raise ValueError("finite timestamp required")
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return t


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    *,
    timeout: int = 30,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> Any:
    final = url
    if params:
        final += ("&" if "?" in final else "?") + urlencode(
            {k: v for k, v in params.items() if v is not None}
        )
    headers = {
        "User-Agent": "etherforecast-data-research/1.0",
        "Accept": "application/json",
    }
    last: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with urlopen(Request(final, headers=headers), timeout=timeout) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError("oversized source response")
            return json.loads(raw)
        except SOURCE_FETCH_ERRORS as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= retries - 1:
                break
            delay = backoff_seconds * (attempt + 1)
            if isinstance(exc, HTTPError):
                retry = exc.headers.get("Retry-After")
                if retry:
                    try:
                        delay = min(30.0, max(delay, float(retry)))
                    except ValueError:
                        pass
            time.sleep(delay)
    if last is not None:
        raise last
    raise RuntimeError("source request failed without an exception")


def _bybit_result(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Bybit response must be an object")
    if int(payload.get("retCode", -1)) != 0:
        raise ValueError("Bybit returned a non-zero retCode")
    result = payload.get("result")
    if not isinstance(result, dict):
        raise ValueError("Bybit result is missing")
    return result


def _numeric_frame(rows: list[dict[str, Any]], *, timestamp_key: str) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    if timestamp_key not in frame:
        raise ValueError(f"missing timestamp field: {timestamp_key}")
    frame.index = pd.to_datetime(
        pd.to_numeric(frame[timestamp_key], errors="raise"), unit="ms", utc=True
    )
    frame = frame.drop(columns=[timestamp_key])
    for c in frame:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    return frame.replace([np.inf, -np.inf], np.nan)


def collect_bybit_funding_history(
    start: Any,
    end: Any,
    *,
    symbol: str = DEFAULT_BYBIT_SYMBOL,
    get_json: Callable[..., Any] = request_json,
    max_pages: int = 200,
) -> pd.DataFrame:
    """Collect Bybit perpetual funding backwards in bounded pages."""
    start_t, end_t = utc(start), utc(end)
    if start_t > end_t:
        raise ValueError("start must not exceed end")
    end_ms = int(end_t.timestamp() * 1000)
    floor_ms = int(start_t.timestamp() * 1000)
    pages: list[pd.DataFrame] = []
    previous_oldest: int | None = None

    for _ in range(max_pages):
        payload = get_json(
            BYBIT_BASE + "/v5/market/funding/history",
            params={
                "category": "linear",
                "symbol": symbol,
                "endTime": end_ms,
                "limit": 200,
            },
        )
        rows = _bybit_result(payload).get("list", [])
        if not isinstance(rows, list) or not rows:
            break
        frame = _numeric_frame(rows, timestamp_key="fundingRateTimestamp")
        if frame.empty:
            break
        frame = frame.loc[(frame.index >= start_t) & (frame.index <= end_t)]
        if not frame.empty:
            pages.append(frame)
        raw_oldest = min(int(r["fundingRateTimestamp"]) for r in rows if "fundingRateTimestamp" in r)
        if previous_oldest is not None and raw_oldest >= previous_oldest:
            raise ValueError("Bybit funding pagination did not move backwards")
        previous_oldest = raw_oldest
        if raw_oldest <= floor_ms or len(rows) < 200:
            break
        end_ms = raw_oldest - 1
    else:
        raise ValueError("Bybit funding pagination exceeded safety limit")

    if not pages:
        return pd.DataFrame()
    raw = pd.concat(pages).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    if "fundingRate" not in raw:
        raise ValueError("Bybit funding response lacks fundingRate")
    daily = pd.DataFrame(index=raw.resample("1D").size().index)
    daily["bybit_eth_funding_rate_mean"] = raw["fundingRate"].resample("1D").mean()
    daily["bybit_eth_funding_rate_last"] = raw["fundingRate"].resample("1D").last()
    daily["bybit_eth_funding_settlements"] = raw["fundingRate"].resample("1D").count()
    return daily.dropna(how="all")


def collect_bybit_open_interest_history(
    start: Any,
    end: Any,
    *,
    symbol: str = DEFAULT_BYBIT_SYMBOL,
    interval: str = "1d",
    get_json: Callable[..., Any] = request_json,
    chunk_days: int = 180,
    max_pages_per_chunk: int = 50,
) -> pd.DataFrame:
    """Collect Bybit OI in base-coin units for linear ETHUSDT."""
    start_t, end_t = utc(start).floor("D"), utc(end).ceil("D")
    if start_t > end_t:
        raise ValueError("start must not exceed end")
    pieces: list[pd.DataFrame] = []
    chunk_start = start_t

    while chunk_start <= end_t:
        chunk_end = min(end_t, chunk_start + pd.Timedelta(days=chunk_days))
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(max_pages_per_chunk):
            params = {
                "category": "linear",
                "symbol": symbol,
                "intervalTime": interval,
                "startTime": int(chunk_start.timestamp() * 1000),
                "endTime": int(chunk_end.timestamp() * 1000),
                "limit": 200,
                "cursor": cursor,
            }
            result = _bybit_result(get_json(BYBIT_BASE + "/v5/market/open-interest", params=params))
            rows = result.get("list", [])
            if not isinstance(rows, list) or not rows:
                break
            frame = _numeric_frame(rows, timestamp_key="timestamp")
            if not frame.empty:
                pieces.append(frame)
            next_cursor = str(result.get("nextPageCursor") or "")
            if not next_cursor:
                break
            if next_cursor in seen:
                raise ValueError("Bybit OI cursor loop detected")
            seen.add(next_cursor)
            cursor = next_cursor
        else:
            raise ValueError("Bybit OI pagination exceeded safety limit")
        chunk_start = chunk_end + pd.Timedelta(milliseconds=1)

    if not pieces:
        return pd.DataFrame()
    raw = pd.concat(pieces).sort_index()
    raw = raw[~raw.index.duplicated(keep="last")]
    keep = [c for c in ("openInterest", "singleOpenInterest") if c in raw]
    if not keep:
        raise ValueError("Bybit OI response lacks OI values")
    daily = pd.DataFrame(index=raw.resample("1D").size().index)
    if "openInterest" in raw:
        daily["bybit_eth_open_interest_eth_last"] = raw["openInterest"].resample("1D").last()
    if "singleOpenInterest" in raw:
        daily["bybit_eth_single_open_interest_eth_last"] = raw["singleOpenInterest"].resample("1D").last()
    return daily.dropna(how="all")


def collect_bybit_premium_index_history(
    start: Any,
    end: Any,
    *,
    symbol: str = DEFAULT_BYBIT_SYMBOL,
    get_json: Callable[..., Any] = request_json,
    max_pages: int = 50,
) -> pd.DataFrame:
    """Collect daily premium-index candles, a free market basis proxy."""
    start_t, end_t = utc(start).floor("D"), utc(end).ceil("D")
    floor_ms = int(start_t.timestamp() * 1000)
    end_ms = int(end_t.timestamp() * 1000)
    rows_out: list[list[Any]] = []
    previous_oldest: int | None = None

    for _ in range(max_pages):
        result = _bybit_result(
            get_json(
                BYBIT_BASE + "/v5/market/premium-index-price-kline",
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "interval": "D",
                    "end": end_ms,
                    "limit": 1000,
                },
            )
        )
        rows = result.get("list", [])
        if not isinstance(rows, list) or not rows:
            break
        valid = [r for r in rows if isinstance(r, (list, tuple)) and len(r) >= 5]
        if not valid:
            break
        rows_out.extend(valid)
        oldest = min(int(r[0]) for r in valid)
        if previous_oldest is not None and oldest >= previous_oldest:
            raise ValueError("Bybit premium pagination did not move backwards")
        previous_oldest = oldest
        if oldest <= floor_ms or len(valid) < 1000:
            break
        end_ms = oldest - 1
    else:
        raise ValueError("Bybit premium pagination exceeded safety limit")

    if not rows_out:
        return pd.DataFrame()
    extra_count = max(0, len(rows_out[0]) - 5)
    frame = pd.DataFrame(
        rows_out,
        columns=["timestamp", "open", "high", "low", "close"] + [f"extra_{i}" for i in range(extra_count)],
    )
    frame.index = pd.to_datetime(pd.to_numeric(frame.timestamp), unit="ms", utc=True)
    frame = frame.drop(columns=[c for c in frame.columns if c == "timestamp" or c.startswith("extra_")])
    for c in ("open", "high", "low", "close"):
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    frame = frame.loc[(frame.index >= start_t) & (frame.index <= end_t)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame(index=frame.index.floor("D"))
    out["bybit_eth_premium_index_open"] = frame["open"].to_numpy()
    out["bybit_eth_premium_index_high"] = frame["high"].to_numpy()
    out["bybit_eth_premium_index_low"] = frame["low"].to_numpy()
    out["bybit_eth_premium_index_close"] = frame["close"].to_numpy()
    out["bybit_eth_premium_index_ohlc_mean"] = frame[["open", "high", "low", "close"]].mean(axis=1).to_numpy()
    return out[~out.index.duplicated(keep="last")].sort_index()


def build_bybit_feature_frame(funding: pd.DataFrame, oi: pd.DataFrame, premium: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic past-only derivative features from free raw series."""
    frames = [f for f in (funding, oi, premium) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]

    if "bybit_eth_funding_rate_mean" in out:
        s = out["bybit_eth_funding_rate_mean"]
        out["bybit_eth_funding_rate_mean_7d"] = s.rolling(7, min_periods=4).mean()
        mean30 = s.rolling(30, min_periods=15).mean()
        std30 = s.rolling(30, min_periods=15).std().replace(0, np.nan)
        out["bybit_eth_funding_z30"] = (s - mean30) / std30

    if "bybit_eth_open_interest_eth_last" in out:
        oi_s = out["bybit_eth_open_interest_eth_last"].where(lambda x: x > 0)
        log_oi = np.log(oi_s)
        out["bybit_eth_oi_log_change_1d"] = log_oi.diff(1)
        out["bybit_eth_oi_log_change_7d"] = log_oi.diff(7)

    if "bybit_eth_premium_index_ohlc_mean" in out:
        p = out["bybit_eth_premium_index_ohlc_mean"]
        out["bybit_eth_premium_index_mean_7d"] = p.rolling(7, min_periods=4).mean()
        pmean = p.rolling(30, min_periods=15).mean()
        pstd = p.rolling(30, min_periods=15).std().replace(0, np.nan)
        out["bybit_eth_premium_z30"] = (p - pmean) / pstd

    if {"bybit_eth_funding_rate_mean", "bybit_eth_oi_log_change_1d"}.issubset(out.columns):
        out["bybit_eth_leverage_pressure"] = out["bybit_eth_funding_rate_mean"] * out["bybit_eth_oi_log_change_1d"]

    out.index = pd.DatetimeIndex(out.index).tz_convert("UTC") if out.index.tz else pd.DatetimeIndex(out.index).tz_localize("UTC")
    out.index.name = "date"
    return out


def availability_sidecar(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["date", "column", "source", "available_at"])
    rows: list[dict[str, Any]] = []
    for t, row in frame.iterrows():
        day = utc(t).floor("D")
        available = day + pd.Timedelta(days=1)
        for c, value in row.items():
            if pd.notna(value):
                rows.append({
                    "date": day.isoformat(),
                    "column": c,
                    "source": source,
                    "available_at": available.isoformat(),
                })
    return pd.DataFrame(rows)


def collect_fred_initial_release_long(
    api_key: str,
    *,
    series: dict[str, str] | None = None,
    get_json: Callable[..., Any] = request_json,
) -> pd.DataFrame:
    """Fetch initial-release observations using FRED/ALFRED output_type=4."""
    if not api_key:
        return pd.DataFrame()
    series = series or DEFAULT_FRED_INITIAL_SERIES
    rows: list[dict[str, Any]] = []
    for series_id, alias in series.items():
        payload = get_json(
            FRED_BASE + "/series/observations",
            params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "output_type": 4,
                "realtime_start": "1776-07-04",
                "realtime_end": "9999-12-31",
                "limit": 100000,
                "sort_order": "asc",
            },
        )
        observations = payload.get("observations", []) if isinstance(payload, dict) else []
        if not isinstance(observations, list):
            raise ValueError("FRED observations payload is invalid")
        for obs in observations:
            try:
                value = float(obs.get("value"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(value):
                continue
            observation_date = pd.to_datetime(obs.get("date"), errors="coerce")
            release_date = pd.to_datetime(obs.get("realtime_start"), errors="coerce")
            if pd.isna(observation_date) or pd.isna(release_date):
                continue
            rows.append({
                "series_id": series_id,
                "alias": alias,
                "observation_date": observation_date.date().isoformat(),
                "initial_release_date": release_date.date().isoformat(),
                "value": value,
            })
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    return frame.sort_values(["series_id", "initial_release_date", "observation_date"]).reset_index(drop=True)


def build_fred_initial_asof_daily(long_frame: pd.DataFrame, start: Any, end: Any) -> pd.DataFrame:
    """Build conservative date-level PIT state from initial releases."""
    start_t, end_t = utc(start).floor("D"), utc(end).floor("D")
    idx = pd.date_range(start_t, end_t, freq="1D", tz="UTC")
    out = pd.DataFrame(index=idx)
    if long_frame is None or long_frame.empty:
        out.index.name = "date"
        return out

    required = {"series_id", "alias", "initial_release_date", "value"}
    if not required.issubset(long_frame.columns):
        raise ValueError("FRED initial-release frame lacks required fields")

    for alias, group in long_frame.groupby("alias"):
        events = group.copy()
        events["release"] = pd.to_datetime(events["initial_release_date"], utc=True) + pd.Timedelta(days=1)
        events["value"] = pd.to_numeric(events["value"], errors="coerce")
        events = events.dropna(subset=["release", "value"]).sort_values("release")
        if "observation_date" in events:
            events["observation_date_dt"] = pd.to_datetime(events["observation_date"], utc=True)
            events = events.sort_values(["release", "observation_date_dt"]).drop_duplicates("release", keep="last")
        else:
            events = events.drop_duplicates("release", keep="last")
        s = events.set_index("release")["value"]
        combined = s.reindex(idx.union(s.index)).sort_index().ffill().reindex(idx)
        out[str(alias)] = combined
        release_series = pd.Series(events["release"].array, index=events["release"])
        last_release = release_series.reindex(idx.union(events["release"])).sort_index().ffill().reindex(idx)
        age = (pd.Series(idx, index=idx) - pd.to_datetime(last_release, utc=True)).dt.total_seconds() / 86400
        out[f"{alias}__age_days"] = age.to_numpy()

    out.index.name = "date"
    return out


@dataclass(frozen=True)
class Coverage:
    rows: int
    columns: int
    start: str | None
    end: str | None


def coverage(frame: pd.DataFrame) -> Coverage:
    if frame is None or frame.empty:
        return Coverage(0, 0, None, None)
    idx = pd.DatetimeIndex(frame.index)
    return Coverage(
        rows=int(len(frame)),
        columns=int(frame.shape[1]),
        start=utc(idx.min()).isoformat(),
        end=utc(idx.max()).isoformat(),
    )
