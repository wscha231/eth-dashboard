from __future__ import annotations

import json
import time
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from data_lab.free_sources import request_json, utc

DERIBIT_BASE = "https://www.deribit.com/api/v2"
HYPERLIQUID_INFO = "https://api.hyperliquid.xyz/info"
SOURCE_ERRORS = (HTTPError, URLError, TimeoutError, OSError, ValueError, KeyError)


def request_json_post(
    url: str,
    payload: dict[str, Any],
    *,
    timeout: int = 30,
    retries: int = 3,
    backoff_seconds: float = 1.5,
) -> Any:
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    headers = {
        "User-Agent": "etherforecast-data-research/1.0",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    last: Exception | None = None
    for attempt in range(max(1, int(retries))):
        try:
            with urlopen(Request(url, data=body, headers=headers, method="POST"), timeout=timeout) as response:
                raw = response.read(8_000_001)
            if len(raw) > 8_000_000:
                raise ValueError("oversized source response")
            return json.loads(raw)
        except SOURCE_ERRORS as exc:
            last = exc
            if isinstance(exc, HTTPError) and exc.code not in {408, 425, 429, 500, 502, 503, 504}:
                raise
            if attempt >= retries - 1:
                break
            delay = backoff_seconds * (attempt + 1)
            if isinstance(exc, HTTPError):
                retry_after = exc.headers.get("Retry-After")
                if retry_after:
                    try:
                        delay = min(30.0, max(delay, float(retry_after)))
                    except ValueError:
                        pass
            time.sleep(delay)
    if last is not None:
        raise last
    raise RuntimeError("POST source request failed without an exception")


def collect_deribit_eth_dvol(
    start: Any,
    end: Any,
    *,
    get_json: Callable[..., Any] = request_json,
    max_pages: int = 50,
) -> pd.DataFrame:
    """Collect ETH DVOL candles from Deribit's public volatility-index history."""
    start_t = utc(start).floor("D")
    end_t = utc(end).ceil("D")
    if start_t > end_t:
        raise ValueError("start must not exceed end")
    floor_ms = int(start_t.timestamp() * 1000)
    page_end = int(end_t.timestamp() * 1000)
    rows: list[list[Any]] = []

    for _ in range(max_pages):
        payload = get_json(
            DERIBIT_BASE + "/public/get_volatility_index_data",
            params={
                "currency": "ETH",
                "start_timestamp": floor_ms,
                "end_timestamp": page_end,
                "resolution": "1D",
            },
        )
        result = payload.get("result") if isinstance(payload, dict) else None
        if not isinstance(result, dict):
            raise ValueError("Deribit DVOL result missing")
        data = result.get("data", [])
        if not isinstance(data, list):
            raise ValueError("Deribit DVOL data invalid")
        valid = [r for r in data if isinstance(r, (list, tuple)) and len(r) >= 5]
        rows.extend(valid)
        continuation = result.get("continuation")
        if continuation is None:
            break
        next_end = int(continuation)
        if next_end >= page_end:
            raise ValueError("Deribit DVOL continuation did not move backwards")
        page_end = next_end
        if page_end <= floor_ms:
            break
    else:
        raise ValueError("Deribit DVOL pagination exceeded safety limit")

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close"])
    frame.index = pd.to_datetime(pd.to_numeric(frame.pop("timestamp"), errors="raise"), unit="ms", utc=True)
    for column in ("open", "high", "low", "close"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.loc[(frame.index >= start_t) & (frame.index <= end_t)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    frame = frame.rename(columns={
        "open": "deribit_eth_dvol_open",
        "high": "deribit_eth_dvol_high",
        "low": "deribit_eth_dvol_low",
        "close": "deribit_eth_dvol_close",
    })
    s = frame["deribit_eth_dvol_close"]
    frame["deribit_eth_dvol_change_1d"] = s.diff()
    frame["deribit_eth_dvol_7d_mean"] = s.rolling(7, min_periods=4).mean()
    mean30 = s.rolling(30, min_periods=15).mean()
    std30 = s.rolling(30, min_periods=15).std().replace(0, np.nan)
    frame["deribit_eth_dvol_z30"] = (s - mean30) / std30
    frame.index.name = "date"
    return frame


def collect_hyperliquid_eth_funding(
    start: Any,
    end: Any,
    *,
    post_json: Callable[..., Any] = request_json_post,
    max_pages: int = 200,
) -> pd.DataFrame:
    """Collect ETH perpetual funding history from Hyperliquid's public info API."""
    start_t, end_t = utc(start), utc(end)
    if start_t > end_t:
        raise ValueError("start must not exceed end")
    cursor = int(start_t.timestamp() * 1000)
    end_ms = int(end_t.timestamp() * 1000)
    rows: list[dict[str, Any]] = []
    previous_last: int | None = None

    for _ in range(max_pages):
        payload = post_json(
            HYPERLIQUID_INFO,
            {"type": "fundingHistory", "coin": "ETH", "startTime": cursor, "endTime": end_ms},
        )
        if not isinstance(payload, list) or not payload:
            break
        batch = [r for r in payload if isinstance(r, dict) and r.get("time") is not None]
        if not batch:
            break
        rows.extend(batch)
        last_time = max(int(r["time"]) for r in batch)
        if previous_last is not None and last_time <= previous_last:
            raise ValueError("Hyperliquid funding pagination did not move forward")
        previous_last = last_time
        if last_time >= end_ms or len(batch) < 500:
            break
        cursor = last_time + 1
    else:
        raise ValueError("Hyperliquid funding pagination exceeded safety limit")

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    frame.index = pd.to_datetime(pd.to_numeric(frame["time"], errors="raise"), unit="ms", utc=True)
    frame["fundingRate"] = pd.to_numeric(frame.get("fundingRate"), errors="coerce")
    frame["premium"] = pd.to_numeric(frame.get("premium"), errors="coerce")
    frame = frame.loc[(frame.index >= start_t) & (frame.index <= end_t)]
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    out = pd.DataFrame(index=frame.resample("1D").size().index)
    out["hyperliquid_eth_funding_rate_mean"] = frame["fundingRate"].resample("1D").mean()
    out["hyperliquid_eth_funding_rate_last"] = frame["fundingRate"].resample("1D").last()
    out["hyperliquid_eth_funding_observations"] = frame["fundingRate"].resample("1D").count()
    if frame["premium"].notna().any():
        out["hyperliquid_eth_premium_mean"] = frame["premium"].resample("1D").mean()
        out["hyperliquid_eth_premium_last"] = frame["premium"].resample("1D").last()
    out.index.name = "date"
    return out.dropna(how="all")


def collect_hyperliquid_eth_snapshot(
    snapshot_day: Any,
    *,
    post_json: Callable[..., Any] = request_json_post,
) -> pd.DataFrame:
    """Capture current ETH perp context; OI is accumulated prospectively only."""
    payload = post_json(HYPERLIQUID_INFO, {"type": "metaAndAssetCtxs"})
    if not isinstance(payload, list) or len(payload) != 2:
        raise ValueError("Hyperliquid metaAndAssetCtxs response invalid")
    meta, contexts = payload
    universe = meta.get("universe", []) if isinstance(meta, dict) else []
    if not isinstance(universe, list) or not isinstance(contexts, list):
        raise ValueError("Hyperliquid universe/contexts invalid")
    index = next((i for i, row in enumerate(universe) if isinstance(row, dict) and row.get("name") == "ETH"), None)
    if index is None or index >= len(contexts) or not isinstance(contexts[index], dict):
        raise ValueError("Hyperliquid ETH context missing")
    row = contexts[index]

    def num(key: str) -> float:
        return float(pd.to_numeric(row.get(key), errors="coerce"))

    day = utc(snapshot_day).floor("D")
    out = pd.DataFrame([{
        "hyperliquid_eth_open_interest_native_snapshot": num("openInterest"),
        "hyperliquid_eth_funding_rate_snapshot": num("funding"),
        "hyperliquid_eth_premium_snapshot": num("premium"),
        "hyperliquid_eth_mark_price": num("markPx"),
        "hyperliquid_eth_oracle_price": num("oraclePx"),
        "hyperliquid_eth_mid_price": num("midPx"),
        "hyperliquid_eth_day_notional_volume_usd": num("dayNtlVlm"),
    }], index=pd.DatetimeIndex([day], name="date"))
    return out.replace([np.inf, -np.inf], np.nan)


def build_hyperliquid_features(funding: pd.DataFrame, snapshot: pd.DataFrame) -> pd.DataFrame:
    frames = [f for f in (funding, snapshot) if f is not None and not f.empty]
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, axis=1).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "hyperliquid_eth_funding_rate_mean" in out:
        s = out["hyperliquid_eth_funding_rate_mean"]
        out["hyperliquid_eth_funding_7d_mean"] = s.rolling(7, min_periods=4).mean()
        mean30 = s.rolling(30, min_periods=15).mean()
        std30 = s.rolling(30, min_periods=15).std().replace(0, np.nan)
        out["hyperliquid_eth_funding_z30"] = (s - mean30) / std30
    if "hyperliquid_eth_premium_mean" in out:
        p = out["hyperliquid_eth_premium_mean"]
        out["hyperliquid_eth_premium_7d_mean"] = p.rolling(7, min_periods=4).mean()
    if "hyperliquid_eth_open_interest_native_snapshot" in out:
        oi = out["hyperliquid_eth_open_interest_native_snapshot"].where(lambda x: x > 0)
        out["hyperliquid_eth_oi_log_change_1d"] = np.log(oi).diff()
    out.index.name = "date"
    return out
