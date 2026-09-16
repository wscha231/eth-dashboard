from __future__ import annotations
import math
from typing import Any, Callable
import numpy as np
import pandas as pd
from data_lab.free_sources import FRED_BASE, DEFAULT_FRED_INITIAL_SERIES, request_json, utc

BITGET_BASE = "https://api.bitget.com"
SYMBOL = "ETHUSDT"
PRODUCT = "USDT-FUTURES"


def _bg(payload: Any) -> Any:
    if not isinstance(payload, dict) or str(payload.get("code")) != "00000":
        raise ValueError("Bitget non-success response")
    return payload.get("data")


def collect_bitget_funding(start: Any, end: Any, get_json: Callable[..., Any] = request_json, max_pages: int = 100) -> pd.DataFrame:
    start_t, end_t = utc(start), utc(end)
    rows: list[dict[str, Any]] = []
    for page in range(1, max_pages + 1):
        data = _bg(get_json(BITGET_BASE + "/api/v2/mix/market/history-fund-rate", params={
            "symbol": SYMBOL, "productType": "usdt-futures", "pageSize": 100, "pageNo": page,
        }))
        if not isinstance(data, list) or not data:
            break
        rows.extend(data)
        oldest = min(int(x["fundingTime"]) for x in data if x.get("fundingTime"))
        if pd.to_datetime(oldest, unit="ms", utc=True) <= start_t or len(data) < 100:
            break
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(pd.to_numeric(frame["fundingTime"]), unit="ms", utc=True)
    frame["fundingRate"] = pd.to_numeric(frame["fundingRate"], errors="coerce")
    frame = frame.loc[(frame.index >= start_t) & (frame.index <= end_t)].sort_index()
    grouped = frame["fundingRate"].resample("1D")
    out = pd.DataFrame({
        "bitget_eth_funding_rate_mean": grouped.mean(),
        "bitget_eth_funding_rate_last": grouped.last(),
        "bitget_eth_funding_settlements": grouped.count(),
    })
    return out.dropna(how="all")


def _history_candles(path: str, start: Any, end: Any, get_json: Callable[..., Any] = request_json) -> pd.DataFrame:
    start_t, end_t = utc(start).floor("D"), utc(end).floor("D")
    parts: list[pd.DataFrame] = []
    current = start_t
    while current <= end_t:
        stop = min(end_t, current + pd.Timedelta(days=89))
        data = _bg(get_json(BITGET_BASE + path, params={
            "symbol": SYMBOL,
            "productType": "usdt-futures",
            "granularity": "1Dutc",
            "startTime": int(current.timestamp() * 1000),
            "endTime": int((stop + pd.Timedelta(days=1) - pd.Timedelta(milliseconds=1)).timestamp() * 1000),
            "limit": 200,
        }))
        if isinstance(data, list) and data:
            good = [row for row in data if isinstance(row, (list, tuple)) and len(row) >= 5]
            if good:
                frame = pd.DataFrame(good)
                frame.index = pd.to_datetime(pd.to_numeric(frame[0]), unit="ms", utc=True)
                frame["close"] = pd.to_numeric(frame[4], errors="coerce")
                parts.append(frame[["close"]])
        current = stop + pd.Timedelta(days=1)
    if not parts:
        return pd.DataFrame()
    out = pd.concat(parts).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out.loc[(out.index >= start_t) & (out.index <= end_t)]


def collect_bitget_basis(start: Any, end: Any, get_json: Callable[..., Any] = request_json) -> pd.DataFrame:
    mark = _history_candles("/api/v2/mix/market/history-mark-candles", start, end, get_json)
    index = _history_candles("/api/v2/mix/market/history-index-candles", start, end, get_json)
    if mark.empty or index.empty:
        return pd.DataFrame()
    out = mark.rename(columns={"close": "bitget_eth_mark_price"}).join(
        index.rename(columns={"close": "bitget_eth_index_price"}), how="inner"
    )
    out["bitget_eth_mark_index_basis"] = out["bitget_eth_mark_price"] / out["bitget_eth_index_price"] - 1.0
    out["bitget_eth_basis_7d_mean"] = out["bitget_eth_mark_index_basis"].rolling(7, min_periods=4).mean()
    mean30 = out["bitget_eth_mark_index_basis"].rolling(30, min_periods=15).mean()
    std30 = out["bitget_eth_mark_index_basis"].rolling(30, min_periods=15).std().replace(0, np.nan)
    out["bitget_eth_basis_z30"] = (out["bitget_eth_mark_index_basis"] - mean30) / std30
    return out


def collect_bitget_ticker_snapshot(snapshot_day: Any, get_json: Callable[..., Any] = request_json) -> pd.DataFrame:
    data = _bg(get_json(BITGET_BASE + "/api/v3/market/tickers", params={
        "category": "USDT-FUTURES", "symbol": SYMBOL,
    }))
    rows = data if isinstance(data, list) else []
    row = next((r for r in rows if str(r.get("symbol", "")).upper() == SYMBOL), rows[0] if rows else None)
    if not row:
        return pd.DataFrame()
    day = utc(snapshot_day).floor("D")
    def num(key: str) -> float:
        return pd.to_numeric(row.get(key), errors="coerce")
    oi = num("openInterest") if "openInterest" in row else num("holdingAmount")
    return pd.DataFrame([{
        "bitget_eth_open_interest_eth_snapshot": oi,
        "bitget_eth_ticker_funding_rate": num("fundingRate"),
        "bitget_eth_ticker_mark_price": num("markPrice"),
        "bitget_eth_ticker_index_price": num("indexPrice"),
        "bitget_eth_ticker_volume24h_eth": num("volume24h") if "volume24h" in row else num("baseVolume"),
        "bitget_eth_ticker_turnover24h_usdt": num("turnover24h") if "turnover24h" in row else num("quoteVolume"),
    }], index=pd.DatetimeIndex([day], name="date"))


def build_bitget_features(funding: pd.DataFrame, basis: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    frames = [x for x in (funding, basis, snapshot) if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "bitget_eth_funding_rate_mean" in out:
        series = out["bitget_eth_funding_rate_mean"]
        out["bitget_eth_funding_rate_7d_mean"] = series.rolling(7, min_periods=4).mean()
        mean30 = series.rolling(30, min_periods=15).mean()
        std30 = series.rolling(30, min_periods=15).std().replace(0, np.nan)
        out["bitget_eth_funding_z30"] = (series - mean30) / std30
    if "bitget_eth_open_interest_eth_snapshot" in out:
        oi = out["bitget_eth_open_interest_eth_snapshot"].where(lambda x: x > 0)
        out["bitget_eth_oi_log_change_1d"] = np.log(oi).diff()
    out.index.name = "date"
    return out


def collect_fred_current_long(api_key: str, start: Any, end: Any, series: dict[str, str] | None = None, get_json: Callable[..., Any] = request_json) -> tuple[pd.DataFrame, dict[str, Any]]:
    series = series or DEFAULT_FRED_INITIAL_SERIES
    rows: list[dict[str, Any]] = []
    errors: dict[str, Any] = {}
    for series_id, alias in series.items():
        try:
            payload = get_json(FRED_BASE + "/series/observations", params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "output_type": 1,
                "observation_start": utc(start).date().isoformat(),
                "observation_end": utc(end).date().isoformat(),
                "limit": 100000,
                "sort_order": "asc",
            })
            for obs in payload.get("observations", []):
                try:
                    value = float(obs.get("value"))
                except (TypeError, ValueError):
                    continue
                if math.isfinite(value):
                    rows.append({"series_id": series_id, "alias": alias, "observation_date": obs.get("date"), "value": value})
        except Exception as exc:
            errors[series_id] = getattr(exc, "code", type(exc).__name__)
    return pd.DataFrame(rows), errors


def collect_fred_initial_partial(
    api_key: str,
    series: dict[str, str] | None = None,
    get_json: Callable[..., Any] = request_json,
    realtime_start: Any | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    series = series or DEFAULT_FRED_INITIAL_SERIES
    realtime = utc(realtime_start).date().isoformat() if realtime_start is not None else "1776-07-04"
    rows: list[dict[str, Any]] = []
    errors: dict[str, Any] = {}
    for series_id, alias in series.items():
        try:
            payload = get_json(FRED_BASE + "/series/observations", params={
                "series_id": series_id,
                "api_key": api_key,
                "file_type": "json",
                "output_type": 4,
                "realtime_start": realtime,
                "limit": 100000,
                "sort_order": "asc",
            })
            observations = payload.get("observations", []) if isinstance(payload, dict) else []
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
        except Exception as exc:
            errors[series_id] = getattr(exc, "code", type(exc).__name__)
    return pd.DataFrame(rows), errors
