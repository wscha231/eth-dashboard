"""Predeclared diagnostic slices; market states use information at the origin."""
import numpy as np
import pandas as pd

SEGMENT_VERSION = "origin_segments_v1"
REGIMES = [
    {"id": "all", "label": "모든 시장 상태"},
    {"id": "trend_up", "label": "상승 움직임"},
    {"id": "trend_down", "label": "하락 움직임"},
    {"id": "trend_range", "label": "횡보 움직임"},
    {"id": "vol_high", "label": "고변동"},
    {"id": "vol_normal", "label": "보통·저변동"},
]


def origin_market_states(features):
    sigma = features.eth_vol_720.clip(lower=1e-8)
    strength = features.eth_ret_24 / (sigma * np.sqrt(24))
    ratio = features.eth_vol_24 / sigma
    result = pd.DataFrame(index=features.index)
    result["trend_state"] = np.where(strength > 1, "up", np.where(strength < -1, "down", "range"))
    result["volatility_state"] = np.where(ratio >= 1.5, "high", "normal")
    result.loc[strength.isna(), "trend_state"] = "unknown"
    result.loc[ratio.isna(), "volatility_state"] = "unknown"
    return result


def periods(first_origin, as_of):
    first, as_of = pd.Timestamp(first_origin), pd.Timestamp(as_of)
    end = as_of.normalize() + pd.Timedelta(days=1)
    result = [{"id": "all", "label": "검증 전구간", "start": first.normalize(), "end": end}]
    result += [{"id": f"recent_{days}", "label": f"최근 {days}일", "start": end-pd.Timedelta(days=days), "end": end}
               for days in (30, 90, 365)]
    result += [{"id": f"year_{year}", "label": f"{year}년", "start": pd.Timestamp(f"{year}-01-01", tz="UTC"),
                "end": min(pd.Timestamp(f"{year+1}-01-01", tz="UTC"), end)}
               for year in range(first.year, as_of.year+1)]
    return [{**p, "start": p["start"].isoformat(), "end": p["end"].isoformat()} for p in result]


def segment_mask(frame, period, regime):
    slots = pd.to_datetime(frame.slot, utc=True)
    mask = (slots >= pd.Timestamp(period["start"])) & (slots < pd.Timestamp(period["end"]))
    if regime.startswith("trend_"):
        mask &= frame.trend_state.eq(regime.removeprefix("trend_"))
    if regime.startswith("vol_"):
        mask &= frame.volatility_state.eq(regime.removeprefix("vol_"))
    return mask
