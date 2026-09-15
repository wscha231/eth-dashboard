"""Deterministic research extensions; incumbent feature formulas are not changed."""
from __future__ import annotations
from collections import deque
import hashlib
import inspect

import numpy as np
import pandas as pd

INCUMBENT_BUILDER_SHA256 = "7b5b94fa5d26f7df39ab46427301c1834fe8dd2824a357770ccefcd443db69a6"
EXTENSION_VERSION = "ohlcv_extensions_v1"


def incumbent_features(bars: pd.DataFrame) -> pd.DataFrame:
    from signal_pipeline.data import build_features
    if hashlib.sha256(inspect.getsource(build_features).encode()).hexdigest() != INCUMBENT_BUILDER_SHA256:
        raise ValueError("incumbent DATA formula changed; create a reviewed feature version")
    return build_features(bars)


def exact_rolling_max_time(series: pd.Series, n: int) -> pd.Series:
    """Integer nanosecond rolling maximum: never round source receipts into the past."""
    if n < 1:
        raise ValueError("positive rolling window required")
    values = pd.DatetimeIndex(series).as_unit("ns").asi8
    nat = np.iinfo(np.int64).min
    output = np.full(len(values), nat, dtype=np.int64)
    q: deque[int] = deque()
    valid = np.r_[0, np.cumsum(values != nat)]
    for i, v in enumerate(values):
        while q and q[0] <= i - n:
            q.popleft()
        while q and values[q[-1]] <= v:
            q.pop()
        q.append(i)
        if i >= n - 1 and valid[i+1] - valid[i+1-n] == n:
            output[i] = values[q[0]]
    return pd.Series(pd.to_datetime(output, utc=True), index=series.index)


def coverage(frame: pd.DataFrame) -> dict:
    report = {}
    for name in frame:
        if name in {"reference_price", "sigma", "available_at"} or name.endswith("__available_at"):
            continue
        valid = np.isfinite(frame[name].to_numpy(dtype=float))
        boundaries = np.diff(np.r_[False, ~valid, False].astype(int))
        starts, ends = np.flatnonzero(boundaries == 1), np.flatnonzero(boundaries == -1)
        first = int(np.flatnonzero(valid)[0]) if valid.any() else None
        report[name] = {"expected_rows": len(valid), "valid_rows": int(valid.sum()),
            "missing_rows": int((~valid).sum()), "coverage_ratio": float(valid.mean()) if len(valid) else 0.,
            "first_valid_time": frame.index[valid][0].isoformat() if valid.any() else None,
            "last_valid_time": frame.index[valid][-1].isoformat() if valid.any() else None,
            "leading_unavailable_rows": first if first is not None else len(valid),
            "longest_missing_run_rows": int(max(ends-starts, default=0)),
            "reason_policy": "insufficient_required_history_or_source_gap_or_zero_denominator; no fill"}
    return report


def extension_features(bars: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    """18 past-only hourly columns. Each feature has its own availability column.

    Native Coinbase volumes are base-asset volumes, not USD trading flow. Price times
    volume below is an explicitly labeled turnover proxy, not order-book depth.
    """
    grid = pd.date_range(bars.close_time.min(), bars.close_time.max(), freq="h", tz="UTC")
    eth, btc = [bars.loc[bars["product"].eq(p)].set_index("close_time").reindex(grid)
                for p in ("ETH-USD", "BTC-USD")]
    if eth.index.has_duplicates or btc.index.has_duplicates:
        raise ValueError("select source revisions before feature construction")
    le, lb = np.log(eth.close), np.log(btc.close)
    r, rb = le.diff(), lb.diff()
    out = pd.DataFrame(index=grid); registry = []; time_cache = {}

    def add(name, values, required_bars, formula, unit="dimensionless", products=("ETH-USD",)):
        streams = [eth if p == "ETH-USD" else btc for p in products]
        complete = pd.concat([f[["open", "high", "low", "close", "volume"]].notna().all(axis=1)
                              for f in streams], axis=1).all(axis=1)
        valid = complete.rolling(required_bars, min_periods=required_bars).sum().eq(required_bars)
        out[name] = values.where(valid).replace([np.inf, -np.inf], np.nan)
        key = (tuple(products), required_bars)
        if key not in time_cache:
            times = pd.concat([f.observed_at for f in streams], axis=1)
            # Row maximum via integer comparisons, then exact integer rolling maximum.
            ns = np.column_stack([pd.DatetimeIndex(times.iloc[:, j]).as_unit("ns").asi8 for j in range(times.shape[1])])
            row = pd.Series(pd.to_datetime(ns.max(axis=1), utc=True), index=grid).where(complete)
            time_cache[key] = exact_rolling_max_time(row, required_bars)
        out[name + "__available_at"] = time_cache[key].where(out[name].notna())
        registry.append({"feature_id": name, "formula": formula, "formula_version": EXTENSION_VERSION,
            "provider": "coinbase_exchange", "venue": "coinbase_exchange", "instruments": list(products),
            "unit": unit, "frequency": "1h", "lookback_bars": required_bars,
            "kind": "derived", "research_status": "implemented_not_performance_validated",
            "timestamp_rule": "max(actual source receipts in required window); event_time=closed hour",
            "published_at": None, "publication_evidence": "historical publication timestamps not retained",
            "max_staleness_seconds": 3600, "staleness_policy": "proposed consumer limit, not measured SLA",
            "allowed_use": "reconstructed research; public/live rights unreviewed", "evidence_ids": []})

    for h in (24, 168, 720):
        add(f"x_rv_{h}", r.pow(2).rolling(h).sum(), h+1, f"sum of {h} squared hourly log returns", "log_return_squared")
        add(f"x_rel_{h}", le.diff(h)-lb.diff(h), h+1, f"ETH {h}h log return minus BTC {h}h log return", products=("ETH-USD", "BTC-USD"))
    rv24, rv168, rv720 = [r.pow(2).rolling(h).sum() for h in (24,168,720)]
    add("x_rv_ratio_24_720", (rv24/24)/(rv720/720).replace(0,np.nan), 721, "mean hourly RV24 / mean hourly RV720")
    add("x_rv_ratio_168_720", (rv168/168)/(rv720/720).replace(0,np.nan), 721, "mean hourly RV168 / mean hourly RV720")
    for h in (24,168):
        total = r.pow(2).rolling(h).sum()
        add(f"x_downside_share_{h}", r.clip(upper=0).pow(2).rolling(h).sum()/total.replace(0,np.nan), h+1,
            f"negative-return squared sum / total squared sum over {h}h")
        add(f"x_efficiency_{h}", r.rolling(h).sum()/r.abs().rolling(h).sum().replace(0,np.nan), h+1,
            f"signed sum returns / sum absolute returns over {h}h")
    beta = r.rolling(720).cov(rb)/rb.rolling(720).var().replace(0,np.nan)
    add("x_beta_residual_24", le.diff(24)-beta*lb.diff(24), 721,
        "ETH24 - (past720h covariance / BTC variance) * BTC24", products=("ETH-USD", "BTC-USD"))
    v = np.log1p(eth.volume)
    add("x_volume_z_prior168", (v-v.shift(1).rolling(168).mean())/v.shift(1).rolling(168).std().replace(0,np.nan), 169,
        "current log1p base volume standardized using preceding168h only")
    spread = (eth.high-eth.low).replace(0,np.nan)
    location = (2*eth.close-eth.high-eth.low)/spread
    add("x_close_location", location, 1, "(2*close-high-low)/(high-low); flat bars are null")
    add("x_turnover_illiquidity_24", (r.abs()/(eth.close*eth.volume).replace(0,np.nan)).rolling(24).mean(),25,
        "mean(abs(hourly log return)/(close*base volume)) over24h; turnover proxy", "USD_inverse")
    add("x_intrabar_logrange", np.log(eth.high/eth.low),1,"log(high/low)")
    add("x_volume_location", out.x_volume_z_prior168*location,169,"prior168h volume z times current close location")
    return out, registry


def incumbent_registry(columns: list[str], builder_sha: str) -> list[dict]:
    """Machine-readable descriptions of the existing 32 columns; no formula rewrite."""
    registry = []
    for c in columns:
        family = "calendar" if c.startswith("hour_") else "price_volume"
        formula = c
        if "_ret_" in c and not c.startswith("eth_btc"):
            h = int(c.rsplit("_",1)[1]); formula = f"log(close_t)-log(close_t-{h}h)"
        elif "_vol_" in c:
            h = int(c.rsplit("_",1)[1]); formula = f"sample std of {h} hourly log returns"
        elif "_trend_" in c:
            h = int(c.rsplit("_",1)[1]); formula = f"log(close_t)-mean(log(close),{h}h)"
        else:
            formula = {"eth_btc_ret_24":"eth_ret_24-btc_ret_24", "eth_btc_corr":"past168h Pearson correlation of ETH/BTC hourly log returns",
                "hour_sin":"sin(UTC_hour*2*pi/24)","hour_cos":"cos(UTC_hour*2*pi/24)"}.get(c,
                "log(high/low)" if c.endswith("_range") else
                "log1p(base_volume) z-score over168h, including current bar" if c.endswith("_volume_z") else
                "log(close)-max(log(close),168h)")
        registry.append({"feature_id":c,"formula":formula,"formula_version":"sha256:"+builder_sha,
            "provider":"coinbase_exchange","venue":"coinbase_exchange","instruments":["ETH-USD","BTC-USD"],
            "unit":"dimensionless_log_or_ratio","frequency":"1h","lookback_bars":721,
            "lookback_reason":"incumbent global complete-window gate, including calendar features",
            "family":family,"kind":"derived","research_status":"existing_implementation_no_edge_certification",
            "timestamp_rule":"incumbent available_at; separate exact receipt maximum is authoritative audit sidecar",
            "published_at":None,"publication_evidence":"historical publication timestamps not retained",
            "max_staleness_seconds":3600,"staleness_policy":"proposed consumer limit, not measured SLA",
            "allowed_use":"reconstructed research; public/live rights unreviewed","evidence_ids":[]})
    return registry
