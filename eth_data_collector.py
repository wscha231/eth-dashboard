from __future__ import annotations

import argparse
import json
import os
import socket
import time
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from eth_price_forecast import (
    CACHE_UPDATE_LOOKBACK_ROWS,
    DEFAULT_MASTER_DATA_CSV,
    DEFAULT_STALE_VENDOR_MAX_AGE_DAYS,
    DEFAULT_TICKERS,
    build_external_feature_summary,
    format_timestamp,
    load_external_feature_csvs,
    load_market_data_csv,
    normalize_index,
    sanitize_feature_name,
    save_market_data_csv,
    update_market_data_cache,
)


DEFAULT_COINGECKO_IDS = {
    "eth": "ethereum",
    "btc": "bitcoin",
    "sol": "solana",
    "bnb": "binancecoin",
}
MACRO_SELECTION_CRITICAL_COLUMNS = {
    "market_context": [
        "eth_close",
        "btc_close",
        "spy_close",
        "qqq_close",
        "dxy_close",
        "tnx_close",
        "oil_close",
    ],
    "fred_macro": [
        "fred_bbb_oas",
        "fred_fed_balance_sheet",
        "fred_tga",
        "fred_rrp",
    ],
    "sentiment": [
        "sentiment_fear_greed",
    ],
    "network_activity": [
        "artemis_stablecoin_total_supply",
        "artemis_non_sybil_users",
    ],
}
DEFAULT_ETHERSCAN_ACTIONS = [
    "dailytx",
    "dailygasused",
    "dailynetutilization",
    "dailynewaddress",
    "dailytxnfee",
]
DEFAULT_FRED_SERIES = {
    "treasury_3m": "DGS3MO",
    "treasury_2y": "DGS2",
    "treasury_10y": "DGS10",
    "treasury_30y": "DGS30",
    "yield_curve_10y_2y": "T10Y2Y",
    "sofr": "SOFR",
    "effr": "EFFR",
    "iorb": "IORB",
    "usd_eur": "DEXUSEU",
    "breakeven_10y": "T10YIE",
    "inflation_forward_5y5y": "T5YIFR",
    "hy_oas": "BAMLH0A0HYM2",
    "bbb_oas": "BAMLC0A4CBBB",
    "stl_fsi": "STLFSI4",
    "fed_balance_sheet": "WALCL",
    "tga": "WTREGEN",
    "rrp": "RRPONTSYD",
    "fed_funds": "DFF",
    "unemployment_rate": "UNRATE",
    "cpi": "CPIAUCSL",
    "m2_money_supply": "M2SL",
}
FEAR_GREED_REGIME_MAP = {
    "extreme fear": 0.0,
    "fear": 1.0,
    "neutral": 2.0,
    "greed": 3.0,
    "extreme greed": 4.0,
}
DEFAULT_BINANCE_SYMBOL = "ETHUSDT"
DEFAULT_DERIBIT_CURRENCY = "ETH"
DEFAULT_DAY_BOUNDARY_TZ = "Asia/Seoul"
ACTIVE_DAY_BOUNDARY_TZ = DEFAULT_DAY_BOUNDARY_TZ
DEFAULT_RECENT_REFRESH_DAYS = max(45, int(DEFAULT_STALE_VENDOR_MAX_AGE_DAYS) + 15)
SOURCE_FETCH_EXCEPTIONS = (HTTPError, URLError, TimeoutError, socket.timeout, ValueError, KeyError)
DEFAULT_HTTP_RETRIES = 3
DEFAULT_HTTP_BACKOFF_SECONDS = 1.5
DEFAULT_HTTP_HEADERS = {
    "User-Agent": "eth-data-collector/1.0 (+https://localhost)",
    "Accept": "*/*",
}


@dataclass
class CollectorPaths:
    root: Path
    raw_market_dir: Path
    raw_manual_dir: Path
    raw_vendor_dir: Path
    gold_dir: Path
    reports_dir: Path
    market_cache_csv: Path
    master_data_csv: Path
    schema_csv: Path
    external_feature_summary_csv: Path
    source_status_csv: Path
    collector_summary_json: Path


def log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def set_day_boundary_tz(timezone_name: str | None) -> None:
    global ACTIVE_DAY_BOUNDARY_TZ
    candidate = str(timezone_name or "").strip() or DEFAULT_DAY_BOUNDARY_TZ
    try:
        pd.Timestamp.now(tz=candidate)
    except Exception as exc:
        raise ValueError(f"Unsupported collector timezone: {candidate}") from exc
    ACTIVE_DAY_BOUNDARY_TZ = candidate


def utc_today() -> pd.Timestamp:
    return pd.Timestamp.now(tz=ACTIVE_DAY_BOUNDARY_TZ).tz_localize(None).floor("D")


def to_timestamp(value: str | pd.Timestamp | None, default: pd.Timestamp) -> pd.Timestamp:
    if value is None or str(value).strip() == "":
        return default
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        return timestamp.tz_convert(None)
    return timestamp


def make_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    numeric = frame.copy()
    for column in numeric.columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def drop_all_null_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    return frame.loc[frame.notna().any(axis=1)].copy()


def resample_daily(frame: pd.DataFrame, how: str = "last") -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = normalize_index(frame)
    if how == "mean":
        return ordered.resample("1D").mean()
    if how == "sum":
        return ordered.resample("1D").sum()
    return ordered.resample("1D").last()


def timestamped_pairs_to_frame(
    pairs: list[list[Any]] | list[tuple[Any, Any]],
    column_name: str,
    how: str = "last",
) -> pd.DataFrame:
    if not pairs:
        return pd.DataFrame(columns=[column_name])
    frame = pd.DataFrame(pairs, columns=["timestamp", column_name])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
    frame = frame.set_index("timestamp")
    frame[column_name] = pd.to_numeric(frame[column_name], errors="coerce")
    return resample_daily(frame[[column_name]], how=how)


def merge_history_frame(
    existing: pd.DataFrame,
    incoming: pd.DataFrame,
    overwrite_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    existing = normalize_index(existing) if existing is not None and not existing.empty else pd.DataFrame()
    incoming = normalize_index(incoming) if incoming is not None and not incoming.empty else pd.DataFrame()
    incoming = drop_all_null_rows(incoming)
    if incoming.empty:
        return existing.copy()
    if existing.empty:
        return incoming.copy()

    combined_index = existing.index.union(incoming.index)
    merged = existing.reindex(combined_index).copy()
    patch = incoming.reindex(combined_index)
    if overwrite_start is not None:
        overwrite_start = pd.Timestamp(overwrite_start).floor("D")
        patch = patch.loc[patch.index >= overwrite_start]
    patch = drop_all_null_rows(patch)
    if patch.empty:
        return normalize_index(merged)

    for column in patch.columns:
        incoming_col = pd.to_numeric(patch[column], errors="coerce")
        if column in merged.columns:
            base_col = pd.to_numeric(merged.loc[patch.index, column], errors="coerce")
            merged.loc[patch.index, column] = incoming_col.combine_first(base_col)
        else:
            merged[column] = np.nan
            merged.loc[patch.index, column] = incoming_col
    return normalize_index(merged)


def append_history_csv(
    path: str | Path,
    frame: pd.DataFrame,
    overwrite_start: pd.Timestamp | None = None,
) -> pd.DataFrame:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_market_data_csv(destination)
    combined = merge_history_frame(existing, frame, overwrite_start=overwrite_start)
    save_market_data_csv(combined, destination)
    return combined


def load_cached_history_window(
    path: str | Path,
    start_date: pd.Timestamp | None = None,
    end_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    frame = load_market_data_csv(path)
    if frame.empty:
        return frame

    windowed = normalize_index(frame)
    if start_date is not None:
        windowed = windowed.loc[windowed.index >= pd.Timestamp(start_date).floor("D")]
    if end_date is not None:
        windowed = windowed.loc[windowed.index <= pd.Timestamp(end_date).floor("D")]
    return windowed


def source_status_row(source: str, frame: pd.DataFrame, status: str, note: str = "") -> dict[str, Any]:
    non_empty = frame is not None and not frame.empty
    return {
        "source": source,
        "status": status,
        "rows": int(len(frame)) if non_empty else 0,
        "columns": int(frame.shape[1]) if non_empty else 0,
        "start": format_timestamp(frame.index.min()) if non_empty else "",
        "end": format_timestamp(frame.index.max()) if non_empty else "",
        "note": note,
    }


def request_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = DEFAULT_HTTP_RETRIES,
    backoff_seconds: float = DEFAULT_HTTP_BACKOFF_SECONDS,
) -> Any:
    final_url = url
    if params:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        if query:
            separator = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{separator}{query}"

    merged_headers = {**DEFAULT_HTTP_HEADERS, **(headers or {})}
    last_error: Exception | None = None
    attempts = max(int(retries), 1)
    for attempt in range(attempts):
        request = Request(final_url, headers=merged_headers, method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except SOURCE_FETCH_EXCEPTIONS as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= attempts - 1:
                break
            time.sleep(float(backoff_seconds) * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed without a captured error: {final_url}")


def request_text(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 30,
    retries: int = DEFAULT_HTTP_RETRIES,
    backoff_seconds: float = DEFAULT_HTTP_BACKOFF_SECONDS,
) -> str:
    final_url = url
    if params:
        query = urlencode({key: value for key, value in params.items() if value is not None})
        if query:
            separator = "&" if "?" in final_url else "?"
            final_url = f"{final_url}{separator}{query}"

    merged_headers = {**DEFAULT_HTTP_HEADERS, **(headers or {})}
    last_error: Exception | None = None
    attempts = max(int(retries), 1)
    for attempt in range(attempts):
        request = Request(final_url, headers=merged_headers, method="GET")
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except SOURCE_FETCH_EXCEPTIONS as exc:
            last_error = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= attempts - 1:
                break
            time.sleep(float(backoff_seconds) * float(attempt + 1))
    if last_error is not None:
        raise last_error
    raise RuntimeError(f"Request failed without a captured error: {final_url}")


def coingecko_headers(api_key: str | None, use_pro: bool) -> dict[str, str]:
    if not api_key:
        return {}
    header_name = "x-cg-pro-api-key" if use_pro else "x-cg-demo-api-key"
    return {header_name: api_key}


def collect_coingecko_coin_history(
    coin_id: str,
    alias: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: str | None,
    use_pro: bool,
) -> pd.DataFrame:
    if not use_pro:
        earliest_allowed = utc_today() - pd.Timedelta(days=365)
        start_date = max(start_date, earliest_allowed)
    base_url = "https://pro-api.coingecko.com/api/v3" if use_pro else "https://api.coingecko.com/api/v3"
    payload = request_json(
        f"{base_url}/coins/{coin_id}/market_chart/range",
        params={
            "vs_currency": "usd",
            "from": int(start_date.timestamp()),
            "to": int((end_date + pd.Timedelta(days=1)).timestamp()),
            "precision": "full",
        },
        headers=coingecko_headers(api_key, use_pro),
    )
    blocks = [
        timestamped_pairs_to_frame(payload.get("prices", []), f"cg_{alias}_price_usd", how="last"),
        timestamped_pairs_to_frame(payload.get("market_caps", []), f"cg_{alias}_market_cap_usd", how="last"),
        timestamped_pairs_to_frame(payload.get("total_volumes", []), f"cg_{alias}_total_volume_usd", how="last"),
    ]
    return normalize_index(pd.concat(blocks, axis=1))


def collect_coingecko_global_snapshot(api_key: str | None, use_pro: bool) -> pd.DataFrame:
    base_url = "https://pro-api.coingecko.com/api/v3" if use_pro else "https://api.coingecko.com/api/v3"
    payload = request_json(
        f"{base_url}/global",
        headers=coingecko_headers(api_key, use_pro),
    )
    data = payload.get("data", {})
    market_cap = data.get("total_market_cap", {})
    market_volume = data.get("total_volume", {})
    dominance = data.get("market_cap_percentage", {})
    row = {
        "cg_global_total_market_cap_usd": pd.to_numeric(market_cap.get("usd"), errors="coerce"),
        "cg_global_total_volume_usd": pd.to_numeric(market_volume.get("usd"), errors="coerce"),
        "cg_global_btc_dominance_pct": pd.to_numeric(dominance.get("btc"), errors="coerce"),
        "cg_global_eth_dominance_pct": pd.to_numeric(dominance.get("eth"), errors="coerce"),
        "cg_global_active_cryptocurrencies": pd.to_numeric(data.get("active_cryptocurrencies"), errors="coerce"),
        "cg_global_markets": pd.to_numeric(data.get("markets"), errors="coerce"),
    }
    return pd.DataFrame([row], index=[utc_today()])


def collect_coingecko_defi_snapshot(api_key: str | None, use_pro: bool) -> pd.DataFrame:
    base_url = "https://pro-api.coingecko.com/api/v3" if use_pro else "https://api.coingecko.com/api/v3"
    payload = request_json(
        f"{base_url}/global/decentralized_finance_defi",
        headers=coingecko_headers(api_key, use_pro),
    )
    data = payload.get("data", {})
    row: dict[str, Any] = {}
    for key, value in data.items():
        row[f"cg_defi_{sanitize_feature_name(key)}"] = pd.to_numeric(value, errors="coerce")
    return pd.DataFrame([row], index=[utc_today()])


def collect_coingecko_global_market_chart(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: str | None,
) -> pd.DataFrame:
    if not api_key:
        return pd.DataFrame()
    days = max(2, int((utc_today() - start_date.floor("D")).days) + 2)
    payload = request_json(
        "https://pro-api.coingecko.com/api/v3/global/market_cap_chart",
        params={"days": days},
        headers={"x-cg-pro-api-key": api_key},
    )
    chart = payload.get("market_cap_chart", {})
    blocks = [
        timestamped_pairs_to_frame(chart.get("market_cap", []), "cg_global_market_cap_chart_usd", how="last"),
        timestamped_pairs_to_frame(chart.get("volume", []), "cg_global_market_volume_chart_usd", how="last"),
    ]
    frame = normalize_index(pd.concat(blocks, axis=1))
    return frame.loc[(frame.index >= start_date.floor("D")) & (frame.index <= end_date.floor("D"))]


def collect_fred_macro_daily(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    daily_index = pd.date_range(start_date.floor("D"), end_date.floor("D"), freq="1D")
    daily_frame = pd.DataFrame(index=daily_index)

    for alias, series_id in DEFAULT_FRED_SERIES.items():
        payload = request_text(
            "https://fred.stlouisfed.org/graph/fredgraph.csv",
            params={"id": series_id},
            timeout=30,
        )
        raw = pd.read_csv(StringIO(payload))
        if raw.empty or raw.shape[1] < 2:
            continue

        raw = raw.iloc[:, :2].copy()
        raw.columns = ["date", "value"]
        raw["date"] = pd.to_datetime(raw["date"], errors="coerce").dt.tz_localize(None)
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw = raw.dropna(subset=["date"]).sort_values("date")
        raw = raw.loc[(raw["date"] >= start_date.floor("D")) & (raw["date"] <= end_date.floor("D"))]
        if raw.empty:
            continue

        series = raw.drop_duplicates(subset=["date"], keep="last").set_index("date")["value"]
        daily_frame[f"fred_{alias}"] = series.reindex(daily_index).ffill()

    return normalize_index(daily_frame)


def collect_fear_greed_daily(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    payload = request_json(
        "https://api.alternative.me/fng/",
        params={"limit": 0, "format": "json"},
        timeout=30,
    )
    rows = payload.get("data", [])
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame()

    records: list[dict[str, Any]] = []
    for item in rows:
        try:
            timestamp = pd.to_datetime(int(item["timestamp"]), unit="s", utc=True).tz_convert(None).floor("D")
        except Exception:
            continue
        classification = str(item.get("value_classification", "")).strip().lower()
        records.append(
            {
                "date": timestamp,
                "sentiment_fear_greed": pd.to_numeric(item.get("value"), errors="coerce"),
                "sentiment_fear_greed_regime": FEAR_GREED_REGIME_MAP.get(classification, np.nan),
            }
        )

    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame(records).dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
    frame = frame.set_index("date").sort_index()
    frame = frame.loc[(frame.index >= start_date.floor("D")) & (frame.index <= end_date.floor("D"))]
    return normalize_index(frame)


def collect_binance_frame(path: str, params: dict[str, Any]) -> pd.DataFrame:
    payload = request_json(f"https://fapi.binance.com{path}", params=params)
    if not isinstance(payload, list) or not payload:
        return pd.DataFrame()
    frame = pd.DataFrame(payload)
    if "timestamp" in frame.columns:
        frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        frame = frame.drop(columns=["timestamp"])
    elif "fundingTime" in frame.columns:
        frame.index = pd.to_datetime(frame["fundingTime"], unit="ms", utc=True).dt.tz_convert(None)
        frame = frame.drop(columns=["fundingTime"])
    else:
        return pd.DataFrame()
    return normalize_index(make_numeric_frame(frame))


def collect_binance_funding_history(
    symbol: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    start_ms = int(start_date.timestamp() * 1000)
    end_ms = int((end_date + pd.Timedelta(days=1)).timestamp() * 1000)
    cursor = start_ms
    frames: list[pd.DataFrame] = []
    while cursor <= end_ms:
        batch = collect_binance_frame(
            "/fapi/v1/fundingRate",
            {
                "symbol": symbol,
                "startTime": cursor,
                "endTime": end_ms,
                "limit": 1000,
            },
        )
        if batch.empty:
            break
        frames.append(batch)
        last_timestamp = int(batch.index[-1].timestamp() * 1000)
        if last_timestamp <= cursor:
            break
        cursor = last_timestamp + 1
        if len(batch) < 1000:
            break

    if not frames:
        return pd.DataFrame()

    history = normalize_index(pd.concat(frames, axis=0))
    history = history[~history.index.duplicated(keep="last")]
    daily = pd.DataFrame(index=resample_daily(history[["fundingRate"]], how="mean").index)
    daily["binance_eth_funding_rate_mean"] = resample_daily(history[["fundingRate"]], how="mean")["fundingRate"]
    if "markPrice" in history.columns:
        daily["binance_eth_mark_price_last"] = resample_daily(history[["markPrice"]], how="last")["markPrice"]
    return daily


def collect_binance_recent_feature(
    path: str,
    params: dict[str, Any],
    column_prefix: str,
) -> pd.DataFrame:
    frame = collect_binance_frame(path, params)
    if frame.empty:
        return frame
    renamed = {
        column: f"{column_prefix}_{sanitize_feature_name(column)}"
        for column in frame.columns
    }
    return resample_daily(frame.rename(columns=renamed), how="last")


def collect_deribit_frame(path: str, params: dict[str, Any]) -> pd.DataFrame:
    payload = request_json(f"https://www.deribit.com/api/v2{path}", params=params)
    result = payload.get("result")
    if result is None:
        return pd.DataFrame()
    if isinstance(result, list):
        frame = pd.DataFrame(result)
    else:
        frame = pd.DataFrame([result])
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        frame.index = pd.to_datetime(frame["timestamp"], unit="ms", utc=True).dt.tz_convert(None)
        frame = frame.drop(columns=["timestamp"])
    return normalize_index(make_numeric_frame(frame))


def aggregate_deribit_summary(rows: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()

    result: dict[str, Any] = {}
    if "open_interest" in rows.columns:
        result[f"{prefix}_open_interest_sum"] = pd.to_numeric(rows["open_interest"], errors="coerce").sum()
    if "volume" in rows.columns:
        result[f"{prefix}_volume_sum"] = pd.to_numeric(rows["volume"], errors="coerce").sum()
    if "volume_usd" in rows.columns:
        result[f"{prefix}_volume_usd_sum"] = pd.to_numeric(rows["volume_usd"], errors="coerce").sum()
    for column in ("mark_price", "underlying_price", "mark_iv", "current_funding", "funding_8h"):
        if column in rows.columns:
            result[f"{prefix}_{sanitize_feature_name(column)}_mean"] = pd.to_numeric(
                rows[column],
                errors="coerce",
            ).mean()

    if "instrument_name" in rows.columns and "open_interest" in rows.columns:
        names = rows["instrument_name"].astype(str)
        oi = pd.to_numeric(rows["open_interest"], errors="coerce").fillna(0.0)
        put_oi = oi[names.str.endswith("-P")].sum()
        call_oi = oi[names.str.endswith("-C")].sum()
        if put_oi > 0.0 or call_oi > 0.0:
            result[f"{prefix}_put_call_oi_ratio"] = put_oi / max(call_oi, 1e-9)

    return pd.DataFrame([result], index=[utc_today()])


def collect_deribit_historical_volatility(currency: str) -> pd.DataFrame:
    payload = request_json(
        "https://www.deribit.com/api/v2/public/get_historical_volatility",
        params={"currency": currency},
    )
    result = payload.get("result", [])
    return timestamped_pairs_to_frame(result, f"deribit_{currency.lower()}_historical_volatility", how="last")


def collect_deribit_funding_history(
    instrument_name: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    frame = collect_deribit_frame(
        "/public/get_funding_rate_history",
        params={
            "instrument_name": instrument_name,
            "start_timestamp": int(start_date.timestamp() * 1000),
            "end_timestamp": int((end_date + pd.Timedelta(days=1)).timestamp() * 1000),
        },
    )
    if frame.empty:
        return frame
    renamed = {
        column: f"deribit_eth_funding_{sanitize_feature_name(column)}"
        for column in frame.columns
    }
    return resample_daily(frame.rename(columns=renamed), how="mean")


def collect_etherscan_action(
    action: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    api_key: str,
) -> pd.DataFrame:
    payload = request_json(
        "https://api.etherscan.io/v2/api",
        params={
            "chainid": "1",
            "module": "stats",
            "action": action,
            "startdate": start_date.strftime("%Y-%m-%d"),
            "enddate": end_date.strftime("%Y-%m-%d"),
            "sort": "asc",
            "apikey": api_key,
        },
    )
    result = payload.get("result", [])
    if not isinstance(result, list) or not result:
        return pd.DataFrame()
    frame = pd.DataFrame(result)
    if "UTCDate" in frame.columns:
        frame.index = pd.to_datetime(frame["UTCDate"])
        frame = frame.drop(columns=["UTCDate"])
    elif "unixTimeStamp" in frame.columns:
        frame.index = pd.to_datetime(frame["unixTimeStamp"], unit="s", utc=True).dt.tz_convert(None)
        frame = frame.drop(columns=["unixTimeStamp"])
    else:
        return pd.DataFrame()
    frame = make_numeric_frame(frame)
    renamed = {
        column: f"etherscan_{sanitize_feature_name(action)}_{sanitize_feature_name(column)}"
        for column in frame.columns
    }
    return normalize_index(frame.rename(columns=renamed))


def resolve_collector_paths(
    drive_root: str | Path | None,
    market_cache_csv: str | Path | None,
    master_data_csv: str | Path | None,
    manual_features_dir: str | Path | None,
) -> CollectorPaths:
    root = Path(drive_root) if drive_root else Path.cwd()
    raw_market_dir = root / "lake" / "raw" / "market"
    raw_manual_dir = Path(manual_features_dir) if manual_features_dir else root / "lake" / "raw" / "manual"
    raw_vendor_dir = root / "lake" / "raw" / "vendor"
    gold_dir = root / "lake" / "gold"
    reports_dir = root / "lake" / "reports"

    for path in (raw_market_dir, raw_manual_dir, raw_vendor_dir, gold_dir, reports_dir):
        path.mkdir(parents=True, exist_ok=True)

    market_cache_path = Path(market_cache_csv) if market_cache_csv else raw_market_dir / "market_data_cache.csv"
    master_data_path = Path(master_data_csv) if master_data_csv else gold_dir / DEFAULT_MASTER_DATA_CSV
    return CollectorPaths(
        root=root,
        raw_market_dir=raw_market_dir,
        raw_manual_dir=raw_manual_dir,
        raw_vendor_dir=raw_vendor_dir,
        gold_dir=gold_dir,
        reports_dir=reports_dir,
        market_cache_csv=market_cache_path,
        master_data_csv=master_data_path,
        schema_csv=gold_dir / "eth_master_schema.csv",
        external_feature_summary_csv=gold_dir / "external_feature_summary.csv",
        source_status_csv=reports_dir / "collector_source_status.csv",
        collector_summary_json=reports_dir / "collector_summary.json",
    )


def discover_manual_feature_paths(
    manual_dir: str | Path | None,
    explicit_paths: list[str] | None,
) -> list[str]:
    discovered: list[str] = []
    seen: set[str] = set()

    if manual_dir:
        for source in sorted(Path(manual_dir).rglob("*.csv")):
            resolved = str(source.resolve())
            if resolved not in seen:
                discovered.append(str(source))
                seen.add(resolved)

    for raw_path in explicit_paths or []:
        source = Path(raw_path)
        resolved = str(source.resolve())
        if resolved not in seen:
            discovered.append(str(source))
            seen.add(resolved)

    return discovered


def build_master_schema(master_data: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in master_data.columns:
        series = pd.to_numeric(master_data[column], errors="coerce")
        non_null = series.dropna()
        if column.startswith("cg_"):
            group = "coingecko"
        elif column.startswith("binance_"):
            group = "binance_derivatives"
        elif column.startswith("deribit_"):
            group = "deribit_derivatives"
        elif column.startswith("etherscan_"):
            group = "etherscan_onchain"
        elif column.startswith("fred_"):
            group = "fred_macro"
        elif column.startswith("sentiment_"):
            group = "sentiment"
        elif column.startswith("artemis_"):
            group = "artemis"
        elif column.startswith("external_"):
            group = "manual_external"
        else:
            group = "market_ohlcv"
        rows.append(
            {
                "column": column,
                "column_group": group,
                "non_null_rows": int(non_null.shape[0]),
                "coverage_ratio": float(non_null.shape[0] / len(series)) if len(series) > 0 else 0.0,
                "start": format_timestamp(non_null.index.min()) if not non_null.empty else "",
                "end": format_timestamp(non_null.index.max()) if not non_null.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values(["column_group", "column"]).reset_index(drop=True)


def build_macro_selection_readiness(master_data: pd.DataFrame) -> dict[str, Any]:
    latest_timestamp = master_data.index.max() if not master_data.empty else None
    rows: list[dict[str, Any]] = []
    total_columns = 0
    ready_columns = 0
    stale_columns: list[str] = []
    missing_columns: list[str] = []

    for group, columns in MACRO_SELECTION_CRITICAL_COLUMNS.items():
        for column in columns:
            total_columns += 1
            exists = column in master_data.columns
            latest_value = float("nan")
            latest_age_days = float("nan")
            status = "missing"
            if exists and not master_data.empty:
                series = pd.to_numeric(master_data[column], errors="coerce")
                non_null = series.dropna()
                if not non_null.empty:
                    latest_value = float(series.iloc[-1]) if pd.notna(series.iloc[-1]) else float("nan")
                    if latest_timestamp is not None:
                        latest_age_days = float((pd.Timestamp(latest_timestamp) - pd.Timestamp(non_null.index.max())).days)
                    status = "ready" if pd.notna(series.iloc[-1]) else "stale"
            if status == "ready":
                ready_columns += 1
            elif status == "stale":
                stale_columns.append(column)
            else:
                missing_columns.append(column)
            rows.append(
                {
                    "column_group": group,
                    "column": column,
                    "status": status,
                    "latest_age_days": latest_age_days,
                    "latest_value": latest_value,
                }
            )

    ready_ratio = float(ready_columns / total_columns) if total_columns > 0 else 0.0
    market_context_ready = all(
        row["status"] == "ready"
        for row in rows
        if row["column_group"] == "market_context"
    )
    macro_ready = bool(market_context_ready and ready_ratio >= 0.75)
    if macro_ready:
        summary_status = "ok"
    elif ready_ratio >= 0.55:
        summary_status = "warning"
    else:
        summary_status = "error"

    return {
        "status": summary_status,
        "ready": macro_ready,
        "total_columns": int(total_columns),
        "ready_columns": int(ready_columns),
        "ready_ratio": ready_ratio,
        "market_context_ready": bool(market_context_ready),
        "missing_columns": missing_columns,
        "stale_columns": stale_columns,
        "details": rows,
    }


def apply_stale_vendor_guardrail(
    master_data: pd.DataFrame,
    max_age_days: int = DEFAULT_STALE_VENDOR_MAX_AGE_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    external_summary = build_external_feature_summary(master_data)
    if external_summary.empty:
        return master_data, external_summary, []

    summary = external_summary.copy()
    summary["stale_guardrail_applied"] = False
    summary["stale_guardrail_max_age_days"] = int(max_age_days)
    summary["freshness_status"] = "ok"

    if int(max_age_days) < 0:
        summary["freshness_status"] = "guardrail_disabled"
        return master_data, summary, []

    latest_age_days = pd.to_numeric(summary.get("latest_age_days"), errors="coerce")
    stale_mask = latest_age_days.notna() & (latest_age_days >= float(max_age_days))
    summary.loc[stale_mask, "stale_guardrail_applied"] = True
    summary.loc[stale_mask, "freshness_status"] = "stale_nulled"

    stale_columns = summary.loc[stale_mask, "column"].astype(str).tolist()
    if not stale_columns:
        return master_data, summary, []

    cleaned = master_data.copy()
    for column in stale_columns:
        if column in cleaned.columns:
            cleaned[column] = np.nan
    return cleaned, summary, stale_columns


def collect_sources(
    paths: CollectorPaths,
    period: str,
    interval: str,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    manual_feature_paths: list[str],
    coingecko_api_key: str | None,
    coingecko_use_pro: bool,
    etherscan_api_key: str | None,
    use_coingecko: bool,
    use_binance: bool,
    use_deribit: bool,
    use_etherscan: bool,
    use_fred: bool,
    use_fear_greed: bool,
    cache_lookback_rows: int,
    overwrite_start: pd.Timestamp | None,
    verbose: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_status: list[dict[str, Any]] = []

    log_progress(verbose, "Refreshing market OHLCV cache...")
    market_result = update_market_data_cache(
        DEFAULT_TICKERS,
        dataset_csv=paths.market_cache_csv,
        period=period,
        interval=interval,
        lookback_rows=cache_lookback_rows,
        verbose=verbose,
        return_metadata=True,
    )
    if isinstance(market_result, tuple):
        market_data, market_meta = market_result
    else:
        market_data, market_meta = market_result, {}
    market_status = "ok"
    market_notes: list[str] = []
    fallback_aliases = [str(alias) for alias in market_meta.get("fallback_cached_aliases", [])]
    failed_aliases = [str(alias) for alias in market_meta.get("failed_aliases", [])]
    if bool(market_meta.get("used_cached_only")):
        market_status = "warning"
        market_notes.append("refresh_failed_all_symbols_using_cached_data")
    elif fallback_aliases:
        market_status = "warning"
        market_notes.append(f"cached_fallback_aliases={','.join(fallback_aliases)}")
    if failed_aliases:
        market_notes.append(f"failed_aliases={'; '.join(failed_aliases[:4])}")
    source_status.append(
        source_status_row(
            "market_ohlcv_yfinance",
            market_data,
            market_status,
            note=" | ".join(market_notes),
        )
    )
    feature_frames: list[pd.DataFrame] = [market_data]

    manual_features = load_external_feature_csvs(manual_feature_paths)
    if not manual_features.empty:
        feature_frames.append(manual_features)
        source_status.append(source_status_row("manual_external_csv", manual_features, "ok"))
    else:
        source_status.append(source_status_row("manual_external_csv", pd.DataFrame(), "skipped", "No manual CSV features found"))

    if use_fred:
        try:
            fred_macro = collect_fred_macro_daily(start_date, end_date)
            if not fred_macro.empty:
                fred_macro = append_history_csv(
                    paths.raw_vendor_dir / "fred_daily.csv",
                    fred_macro,
                    overwrite_start=overwrite_start,
                )
                feature_frames.append(fred_macro)
            source_status.append(source_status_row("fred_macro", fred_macro, "ok" if not fred_macro.empty else "skipped"))
        except SOURCE_FETCH_EXCEPTIONS as exc:
            cached_fred = load_cached_history_window(
                paths.raw_vendor_dir / "fred_daily.csv",
                start_date=start_date,
                end_date=end_date,
            )
            if not cached_fred.empty:
                feature_frames.append(cached_fred)
                source_status.append(
                    source_status_row(
                        "fred_macro",
                        cached_fred,
                        "warning",
                        f"live_fetch_failed_using_cached_data: {exc}",
                    )
                )
            else:
                source_status.append(source_status_row("fred_macro", pd.DataFrame(), "error", str(exc)))
    else:
        source_status.append(source_status_row("fred_macro", pd.DataFrame(), "skipped", "Disabled by flag"))

    if use_fear_greed:
        try:
            fear_greed = collect_fear_greed_daily(start_date, end_date)
            fear_greed = (
                append_history_csv(
                    paths.raw_vendor_dir / "fear_greed_daily.csv",
                    fear_greed,
                    overwrite_start=overwrite_start,
                )
                if not fear_greed.empty
                else pd.DataFrame()
            )
            if not fear_greed.empty:
                feature_frames.append(fear_greed)
            source_status.append(source_status_row("sentiment_fear_greed", fear_greed, "ok" if not fear_greed.empty else "skipped"))
        except SOURCE_FETCH_EXCEPTIONS as exc:
            cached_fear_greed = load_cached_history_window(
                paths.raw_vendor_dir / "fear_greed_daily.csv",
                start_date=start_date,
                end_date=end_date,
            )
            if not cached_fear_greed.empty:
                feature_frames.append(cached_fear_greed)
                source_status.append(
                    source_status_row(
                        "sentiment_fear_greed",
                        cached_fear_greed,
                        "warning",
                        f"live_fetch_failed_using_cached_data: {exc}",
                    )
                )
            else:
                source_status.append(source_status_row("sentiment_fear_greed", pd.DataFrame(), "error", str(exc)))
    else:
        source_status.append(source_status_row("sentiment_fear_greed", pd.DataFrame(), "skipped", "Disabled by flag"))

    if use_coingecko:
        for alias, coin_id in DEFAULT_COINGECKO_IDS.items():
            source_name = f"coingecko_{alias}"
            try:
                frame = collect_coingecko_coin_history(
                    coin_id=coin_id,
                    alias=alias,
                    start_date=start_date,
                    end_date=end_date,
                    api_key=coingecko_api_key,
                    use_pro=coingecko_use_pro,
                )
                if not frame.empty:
                    frame = append_history_csv(
                        paths.raw_vendor_dir / f"coingecko_{alias}_daily.csv",
                        frame,
                        overwrite_start=overwrite_start,
                    )
                    feature_frames.append(frame)
                source_status.append(source_status_row(source_name, frame, "ok" if not frame.empty else "skipped"))
            except SOURCE_FETCH_EXCEPTIONS as exc:
                source_status.append(source_status_row(source_name, pd.DataFrame(), "error", str(exc)))

        try:
            global_snapshot = append_history_csv(
                paths.raw_vendor_dir / "coingecko_global_daily.csv",
                collect_coingecko_global_snapshot(coingecko_api_key, coingecko_use_pro),
                overwrite_start=overwrite_start,
            )
            feature_frames.append(global_snapshot)
            source_status.append(source_status_row("coingecko_global_snapshot", global_snapshot, "ok"))
        except SOURCE_FETCH_EXCEPTIONS as exc:
            source_status.append(source_status_row("coingecko_global_snapshot", pd.DataFrame(), "error", str(exc)))

        try:
            defi_snapshot = append_history_csv(
                paths.raw_vendor_dir / "coingecko_defi_daily.csv",
                collect_coingecko_defi_snapshot(coingecko_api_key, coingecko_use_pro),
                overwrite_start=overwrite_start,
            )
            feature_frames.append(defi_snapshot)
            source_status.append(source_status_row("coingecko_defi_snapshot", defi_snapshot, "ok"))
        except SOURCE_FETCH_EXCEPTIONS as exc:
            source_status.append(source_status_row("coingecko_defi_snapshot", pd.DataFrame(), "error", str(exc)))

        try:
            global_chart = collect_coingecko_global_market_chart(
                start_date=start_date,
                end_date=end_date,
                api_key=coingecko_api_key if coingecko_use_pro else None,
            )
            if not global_chart.empty:
                global_chart = append_history_csv(
                    paths.raw_vendor_dir / "coingecko_global_market_chart.csv",
                    global_chart,
                    overwrite_start=overwrite_start,
                )
                feature_frames.append(global_chart)
            note = "" if not global_chart.empty else "Requires CoinGecko Pro for historical global chart"
            source_status.append(
                source_status_row(
                    "coingecko_global_market_chart",
                    global_chart,
                    "ok" if not global_chart.empty else "skipped",
                    note,
                )
            )
        except SOURCE_FETCH_EXCEPTIONS as exc:
            source_status.append(source_status_row("coingecko_global_market_chart", pd.DataFrame(), "error", str(exc)))
    else:
        source_status.append(source_status_row("coingecko", pd.DataFrame(), "skipped", "Disabled by flag"))

    if use_binance:
        try:
            existing = load_market_data_csv(paths.raw_vendor_dir / "binance_eth_funding_daily.csv")
            funding_start = max(start_date, existing.index.max() - pd.Timedelta(days=3)) if not existing.empty else start_date
            funding = collect_binance_funding_history(DEFAULT_BINANCE_SYMBOL, funding_start, end_date)
            funding = (
                append_history_csv(
                    paths.raw_vendor_dir / "binance_eth_funding_daily.csv",
                    funding,
                    overwrite_start=overwrite_start,
                )
                if not funding.empty
                else existing
            )
            if not funding.empty:
                feature_frames.append(funding)
            source_status.append(source_status_row("binance_funding_history", funding, "ok" if not funding.empty else "skipped"))
        except SOURCE_FETCH_EXCEPTIONS as exc:
            source_status.append(source_status_row("binance_funding_history", pd.DataFrame(), "error", str(exc)))

        recent_start = max(start_date, end_date - pd.Timedelta(days=45))
        recent_start_ms = int(recent_start.timestamp() * 1000)
        recent_end_ms = int((end_date + pd.Timedelta(days=1)).timestamp() * 1000)
        recent_configs = [
            (
                "binance_open_interest_hist",
                "/futures/data/openInterestHist",
                {
                    "symbol": DEFAULT_BINANCE_SYMBOL,
                    "period": "1d",
                    "limit": 500,
                    "startTime": recent_start_ms,
                    "endTime": recent_end_ms,
                },
                "binance_eth_oi",
                paths.raw_vendor_dir / "binance_eth_open_interest_daily.csv",
                "Exchange exposes recent history only",
            ),
            (
                "binance_basis",
                "/futures/data/basis",
                {
                    "pair": DEFAULT_BINANCE_SYMBOL,
                    "contractType": "PERPETUAL",
                    "period": "1d",
                    "limit": 500,
                    "startTime": recent_start_ms,
                    "endTime": recent_end_ms,
                },
                "binance_eth_basis",
                paths.raw_vendor_dir / "binance_eth_basis_daily.csv",
                "Exchange exposes recent history only",
            ),
            (
                "binance_taker_buy_sell_volume",
                "/futures/data/takerBuySellVol",
                {
                    "symbol": DEFAULT_BINANCE_SYMBOL,
                    "period": "1d",
                    "limit": 500,
                    "startTime": recent_start_ms,
                    "endTime": recent_end_ms,
                },
                "binance_eth_taker_buy_sell",
                paths.raw_vendor_dir / "binance_eth_taker_buy_sell_daily.csv",
                "Exchange exposes recent history only",
            ),
        ]
        for source_name, path, params, prefix, output_path, note in recent_configs:
            try:
                existing = load_market_data_csv(output_path)
                frame = collect_binance_recent_feature(path, params, prefix)
                frame = append_history_csv(output_path, frame, overwrite_start=overwrite_start) if not frame.empty else existing
                if not frame.empty:
                    feature_frames.append(frame)
                source_status.append(
                    source_status_row(
                        source_name,
                        frame,
                        "ok" if not frame.empty else "skipped",
                        note,
                    )
                )
            except SOURCE_FETCH_EXCEPTIONS as exc:
                source_status.append(source_status_row(source_name, pd.DataFrame(), "error", str(exc)))
    else:
        source_status.append(source_status_row("binance", pd.DataFrame(), "skipped", "Disabled by flag"))

    if use_deribit:
        try:
            hist_vol = collect_deribit_historical_volatility(DEFAULT_DERIBIT_CURRENCY)
            if not hist_vol.empty:
                hist_vol = append_history_csv(
                    paths.raw_vendor_dir / "deribit_eth_historical_volatility.csv",
                    hist_vol,
                    overwrite_start=overwrite_start,
                )
                feature_frames.append(hist_vol)
            source_status.append(
                source_status_row(
                    "deribit_historical_volatility",
                    hist_vol,
                    "ok" if not hist_vol.empty else "skipped",
                )
            )
        except SOURCE_FETCH_EXCEPTIONS as exc:
            source_status.append(source_status_row("deribit_historical_volatility", pd.DataFrame(), "error", str(exc)))

        try:
            existing = load_market_data_csv(paths.raw_vendor_dir / "deribit_eth_funding_daily.csv")
            # Deribit /get_funding_rate_history retains ~90 days. When the local
            # CSV has fewer than that, widen the pull so z_30/z_90 regime
            # features have enough history instead of being all-NaN.
            min_deribit_backfill_days = 90
            if existing.empty:
                funding_start = start_date
            else:
                earliest_target = end_date - pd.Timedelta(days=min_deribit_backfill_days)
                local_tail = existing.index.max() - pd.Timedelta(days=3)
                if existing.index.min() > earliest_target:
                    funding_start = max(start_date, earliest_target)
                else:
                    funding_start = max(start_date, local_tail)
            funding = collect_deribit_funding_history("ETH-PERPETUAL", funding_start, end_date)
            funding = (
                append_history_csv(
                    paths.raw_vendor_dir / "deribit_eth_funding_daily.csv",
                    funding,
                    overwrite_start=overwrite_start,
                )
                if not funding.empty
                else existing
            )
            if not funding.empty:
                feature_frames.append(funding)
            source_status.append(source_status_row("deribit_funding_history", funding, "ok" if not funding.empty else "skipped"))
        except SOURCE_FETCH_EXCEPTIONS as exc:
            source_status.append(source_status_row("deribit_funding_history", pd.DataFrame(), "error", str(exc)))

        # NOTE: Deribit /get_book_summary_by_currency is snapshot-only (no
        # historical parameters). Phase 3 regime features keyed off option IV,
        # PCR, futures basis, and OI therefore only populate from the day
        # snapshots start landing — prior rows remain NaN. Real backfill would
        # require an alternative vendor (e.g. Block Scholes, Amberdata, Laevitas).
        for kind, filename, prefix in (
            ("future", "deribit_eth_future_snapshot_daily.csv", "deribit_eth_future"),
            ("option", "deribit_eth_option_snapshot_daily.csv", "deribit_eth_option"),
        ):
            source_name = f"deribit_{kind}_snapshot"
            try:
                payload = request_json(
                    "https://www.deribit.com/api/v2/public/get_book_summary_by_currency",
                    params={"currency": DEFAULT_DERIBIT_CURRENCY, "kind": kind},
                )
                rows = pd.DataFrame(payload.get("result", []))
                snapshot = aggregate_deribit_summary(rows, prefix=prefix)
                snapshot = (
                    append_history_csv(paths.raw_vendor_dir / filename, snapshot, overwrite_start=overwrite_start)
                    if not snapshot.empty
                    else pd.DataFrame()
                )
                if not snapshot.empty:
                    feature_frames.append(snapshot)
                source_status.append(source_status_row(source_name, snapshot, "ok" if not snapshot.empty else "skipped"))
            except SOURCE_FETCH_EXCEPTIONS as exc:
                source_status.append(source_status_row(source_name, pd.DataFrame(), "error", str(exc)))
    else:
        source_status.append(source_status_row("deribit", pd.DataFrame(), "skipped", "Disabled by flag"))

    if use_etherscan and etherscan_api_key:
        for action in DEFAULT_ETHERSCAN_ACTIONS:
            source_name = f"etherscan_{action}"
            try:
                frame = collect_etherscan_action(action, start_date, end_date, etherscan_api_key)
                output_path = paths.raw_vendor_dir / f"{source_name}.csv"
                frame = append_history_csv(output_path, frame, overwrite_start=overwrite_start) if not frame.empty else pd.DataFrame()
                if not frame.empty:
                    feature_frames.append(frame)
                source_status.append(source_status_row(source_name, frame, "ok" if not frame.empty else "skipped"))
            except SOURCE_FETCH_EXCEPTIONS as exc:
                source_status.append(source_status_row(source_name, pd.DataFrame(), "error", str(exc)))
    elif use_etherscan:
        note = "Etherscan stats need an API key and several useful endpoints are PRO-only"
        source_status.append(source_status_row("etherscan", pd.DataFrame(), "skipped", note))
    else:
        source_status.append(source_status_row("etherscan", pd.DataFrame(), "skipped", "Disabled by flag"))

    master_data = market_data.copy()
    for frame in feature_frames[1:]:
        if frame.empty:
            continue
        master_data = master_data.join(frame, how="left")
    master_data = normalize_index(master_data)
    master_data = master_data.reindex(market_data.index)

    return master_data, pd.DataFrame(source_status)


def _ignored_unknown_cli_arg(value: str) -> bool:
    token = str(value).strip()
    if token == "":
        return True
    lowered = token.lower()
    ignored_prefixes = (
        "--historymanager.",
        "--session.",
        "--ipkernelapp.",
        "--interactiveshellapp.",
        "--ip=",
        "--stdin=",
        "--control=",
        "--hb=",
        "--shell=",
        "--transport=",
        "--iopub=",
    )
    if token == "-f" or any(lowered.startswith(prefix) for prefix in ignored_prefixes):
        return True
    return lowered.endswith(".json") and "kernel" in lowered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect ETH market, derivatives, on-chain, and manual feature data into an ETH master dataset."
    )
    parser.add_argument("--period", default="max", help="yfinance history window for market OHLCV cache")
    parser.add_argument("--interval", default="1d", help="yfinance interval for market OHLCV cache")
    parser.add_argument("--drive-root", default="", help="Google Drive or local root for the ETH data lake")
    parser.add_argument("--market-cache-csv", default="", help="Override the market cache CSV path")
    parser.add_argument("--master-data-csv", default="", help="Override the gold master dataset CSV path")
    parser.add_argument("--manual-features-dir", default="", help="Directory of manually supplied numeric CSVs")
    parser.add_argument(
        "--manual-features-csv",
        nargs="*",
        default=[],
        help="Additional numeric CSV files to merge into the master dataset",
    )
    parser.add_argument("--start-date", default="", help="Optional collection start date in YYYY-MM-DD")
    parser.add_argument("--end-date", default="", help="Optional collection end date in YYYY-MM-DD")
    parser.add_argument("--coingecko-api-key", default=os.getenv("COINGECKO_API_KEY", ""))
    parser.add_argument("--coingecko-pro", action="store_true", help="Use CoinGecko Pro endpoints when the key supports them")
    parser.add_argument("--etherscan-api-key", default=os.getenv("ETHERSCAN_API_KEY", ""))
    parser.add_argument("--cache-lookback-rows", type=int, default=CACHE_UPDATE_LOOKBACK_ROWS)
    parser.add_argument(
        "--day-boundary-tz",
        default=DEFAULT_DAY_BOUNDARY_TZ,
        help="Timezone used to determine today's collection date and daily snapshot boundary.",
    )
    parser.add_argument(
        "--recent-refresh-days",
        type=int,
        default=DEFAULT_RECENT_REFRESH_DAYS,
        help="When a master dataset already exists and --start-date is omitted, refresh only this many recent days.",
    )
    parser.add_argument(
        "--max-stale-age-days",
        type=int,
        default=DEFAULT_STALE_VENDOR_MAX_AGE_DAYS,
        help="Null vendor columns in the saved master dataset when their latest observation is this many days old or more. Use a negative value to disable.",
    )
    parser.add_argument("--no-coingecko", action="store_true")
    parser.add_argument("--no-binance", action="store_true")
    parser.add_argument("--no-deribit", action="store_true")
    parser.add_argument("--no-etherscan", action="store_true")
    parser.add_argument("--no-fred", action="store_true")
    parser.add_argument("--no-fear-greed", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true")
    args, unknown = parser.parse_known_args(argv)
    unexpected = [token for token in unknown if not _ignored_unknown_cli_arg(token)]
    if unexpected:
        parser.error(f"unrecognized arguments: {' '.join(unexpected)}")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.interval != "1d":
        raise SystemExit("eth_data_collector.py currently builds a daily master dataset. Use --interval 1d.")
    paths = resolve_collector_paths(
        drive_root=(args.drive_root or None),
        market_cache_csv=(args.market_cache_csv or None),
        master_data_csv=(args.master_data_csv or None),
        manual_features_dir=(args.manual_features_dir or None),
    )
    verbose = (not args.quiet) and (not args.json)
    set_day_boundary_tz(args.day_boundary_tz)
    existing_master = load_market_data_csv(paths.master_data_csv)
    today = utc_today()
    default_full_start = today - pd.Timedelta(days=365 * 8)
    if str(args.start_date).strip():
        start_date = to_timestamp(args.start_date, default_full_start).floor("D")
    else:
        if existing_master.empty:
            start_date = default_full_start
        else:
            start_date = (today - pd.Timedelta(days=max(int(args.recent_refresh_days), 1))).floor("D")
    end_date = to_timestamp(args.end_date, today).floor("D")

    manual_paths = discover_manual_feature_paths(paths.raw_manual_dir, args.manual_features_csv)
    master_data, source_status = collect_sources(
        paths=paths,
        period=args.period,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date,
        manual_feature_paths=manual_paths,
        coingecko_api_key=(args.coingecko_api_key or None),
        coingecko_use_pro=args.coingecko_pro,
        etherscan_api_key=(args.etherscan_api_key or None),
        use_coingecko=not args.no_coingecko,
        use_binance=not args.no_binance,
        use_deribit=not args.no_deribit,
        use_etherscan=not args.no_etherscan,
        use_fred=not args.no_fred,
        use_fear_greed=not args.no_fear_greed,
        cache_lookback_rows=args.cache_lookback_rows,
        overwrite_start=start_date,
        verbose=verbose,
    )
    if not existing_master.empty:
        master_data = merge_history_frame(existing_master, master_data, overwrite_start=start_date)

    master_data, external_summary, stale_columns = apply_stale_vendor_guardrail(
        master_data,
        max_age_days=int(args.max_stale_age_days),
    )
    schema = build_master_schema(master_data)
    freshness_note = (
        "guardrail_disabled"
        if int(args.max_stale_age_days) < 0
        else (
            f"nulled_stale_vendor_columns={len(stale_columns)}"
            if stale_columns
            else "no_stale_vendor_columns"
        )
    )
    if stale_columns:
        freshness_note = (
            f"{freshness_note} | max_age_days={int(args.max_stale_age_days)} | columns="
            + ", ".join(stale_columns[:8])
        )
    else:
        freshness_note = f"{freshness_note} | max_age_days={int(args.max_stale_age_days)}"
    freshness_row = {
        "source": "freshness_guardrail",
        "status": "warning" if stale_columns else "ok",
        "rows": int(len(master_data)) if not master_data.empty else 0,
        "columns": int(len(stale_columns)),
        "start": format_timestamp(master_data.index.min()) if not master_data.empty else "",
        "end": format_timestamp(master_data.index.max()) if not master_data.empty else "",
        "note": freshness_note,
    }
    source_status = pd.concat([source_status, pd.DataFrame([freshness_row])], ignore_index=True)
    macro_readiness = build_macro_selection_readiness(master_data)
    macro_note_parts: list[str] = [
        f"ready_columns={macro_readiness['ready_columns']}/{macro_readiness['total_columns']}",
        f"ready_ratio={macro_readiness['ready_ratio']:.2f}",
        f"market_context_ready={int(macro_readiness['market_context_ready'])}",
    ]
    if macro_readiness["missing_columns"]:
        macro_note_parts.append("missing=" + ",".join(macro_readiness["missing_columns"][:6]))
    if macro_readiness["stale_columns"]:
        macro_note_parts.append("stale=" + ",".join(macro_readiness["stale_columns"][:6]))
    macro_row = {
        "source": "macro_selection_readiness",
        "status": macro_readiness["status"],
        "rows": int(len(master_data)) if not master_data.empty else 0,
        "columns": int(macro_readiness["ready_columns"]),
        "start": format_timestamp(master_data.index.min()) if not master_data.empty else "",
        "end": format_timestamp(master_data.index.max()) if not master_data.empty else "",
        "note": " | ".join(macro_note_parts),
    }
    source_status = pd.concat([source_status, pd.DataFrame([macro_row])], ignore_index=True)

    save_market_data_csv(master_data, paths.master_data_csv)
    schema.to_csv(paths.schema_csv, index=False)
    external_summary.to_csv(paths.external_feature_summary_csv, index=False)
    source_status.to_csv(paths.source_status_csv, index=False)

    payload = {
        "root": str(paths.root),
        "master_data_csv": str(paths.master_data_csv),
        "market_cache_csv": str(paths.market_cache_csv),
        "schema_csv": str(paths.schema_csv),
        "source_status_csv": str(paths.source_status_csv),
        "raw_rows": int(master_data.shape[0]),
        "column_count": int(master_data.shape[1]),
        "start": format_timestamp(master_data.index.min()) if not master_data.empty else "",
        "end": format_timestamp(master_data.index.max()) if not master_data.empty else "",
        "day_boundary_tz": ACTIVE_DAY_BOUNDARY_TZ,
        "refresh_window_start": format_timestamp(start_date),
        "refresh_window_end": format_timestamp(end_date),
        "recent_refresh_days": int(args.recent_refresh_days),
        "max_stale_age_days": int(args.max_stale_age_days),
        "manual_feature_file_count": len(manual_paths),
        "stale_vendor_column_count": int(len(stale_columns)),
        "stale_vendor_columns": stale_columns,
        "macro_selection_readiness": macro_readiness,
        "sources": source_status.to_dict(orient="records"),
    }
    with paths.collector_summary_json.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("ETH data lake collector")
    print(f"- Root: {paths.root}")
    print(f"- Master dataset: {paths.master_data_csv}")
    print(f"- Market cache: {paths.market_cache_csv}")
    print(f"- Master rows: {payload['raw_rows']}")
    print(f"- Master columns: {payload['column_count']}")
    print(f"- Date span: {payload['start']} -> {payload['end']}")
    print(f"- Day boundary timezone: {payload['day_boundary_tz']}")
    print(f"- Refresh window: {payload['refresh_window_start']} -> {payload['refresh_window_end']}")
    print(f"- Manual CSV files merged: {payload['manual_feature_file_count']}")
    print(
        f"- Freshness guardrail: max_age_days={payload['max_stale_age_days']}, "
        f"nulled_stale_vendor_columns={payload['stale_vendor_column_count']}"
    )
    if payload["stale_vendor_columns"]:
        print(f"  Stale vendor columns nulled: {', '.join(payload['stale_vendor_columns'][:8])}")
    print(
        f"- Macro readiness: status={payload['macro_selection_readiness']['status']}, "
        f"ready={int(payload['macro_selection_readiness']['ready'])}, "
        f"columns={payload['macro_selection_readiness']['ready_columns']}/{payload['macro_selection_readiness']['total_columns']}"
    )
    if payload["macro_selection_readiness"]["missing_columns"]:
        print(
            "  Missing macro columns: "
            + ", ".join(payload["macro_selection_readiness"]["missing_columns"][:8])
        )
    if payload["macro_selection_readiness"]["stale_columns"]:
        print(
            "  Stale macro columns: "
            + ", ".join(payload["macro_selection_readiness"]["stale_columns"][:8])
        )
    print("- Source status")
    for row in payload["sources"]:
        note = f", note={row['note']}" if row["note"] else ""
        print(
            f"  - {row['source']}: status={row['status']}, rows={row['rows']}, columns={row['columns']}{note}"
        )


if __name__ == "__main__":
    main()
