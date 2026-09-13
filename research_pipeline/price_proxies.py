"""Past-only OHLCV proxies; never represented as measured OI, IV or liquidations."""
import numpy as np
import pandas as pd


def build_price_proxies(bars, *, volume_unit, frequency='h', window=24):
    """Single instrument, close_time index, explicitly declared volume units.

    The caller must preserve source/receipt metadata. Daily inputs produce daily
    proxies, not fabricated hourly measurements. Missing bars invalidate affected
    windows. No backfill, fitted scaler, labels, or future values are used.
    """
    if frequency not in {'h','D'} or window < 2:
        raise ValueError('hourly/daily frequency and window >= 2 required')
    if volume_unit not in {'native','quote'}:
        raise ValueError('declare native-asset or quote-currency volume units')
    if not isinstance(bars.index,pd.DatetimeIndex) or bars.index.has_duplicates:
        raise ValueError('unique DatetimeIndex required')
    if bars.empty:
        return pd.DataFrame()
    original = bars.sort_index()
    if not original.index.equals(original.index.floor(frequency)):
        raise ValueError('timestamps must align with the selected frequency')
    index = pd.date_range(original.index.min(),original.index.max(),freq=frequency)
    frame = original.reindex(index)
    values = frame[['open','high','low','close','volume']]
    bad = ((values[['open','high','low','close']] <= 0).any(axis=1) |
           (values.volume < 0) | (values.high < values[['open','low','close']].max(axis=1)) |
           (values.low > values[['open','high','close']].min(axis=1)))
    if bad.any() or np.isinf(values.to_numpy(float)).any():
        raise ValueError('invalid OHLCV bounds')
    ret = np.log(frame.close).diff()
    rv = ret.pow(2).rolling(window).sum()
    downside = ret.where(ret < 0, 0).where(ret.notna()).pow(2).rolling(window).sum()
    spread = frame.high-frame.low
    location = ((2*frame.close-frame.high-frame.low)/spread.replace(0,np.nan)).where(spread.ne(0),0.)
    volume_sum = frame.volume.rolling(window).sum().replace(0,np.nan)
    out = pd.DataFrame(index=index)
    out['proxy_downside_variance_share'] = (downside/rv.replace(0,np.nan)).where(rv.ne(0),.5)
    out['proxy_close_location_pressure'] = (location*frame.volume).rolling(window).sum()/volume_sum
    quote_volume = frame.volume if volume_unit == 'quote' else frame.close*frame.volume
    out['proxy_price_impact'] = (ret.abs()/quote_volume.replace(0,np.nan)).rolling(window).mean()
    out['proxy_trend_efficiency'] = ret.rolling(window).sum().abs()/ret.abs().rolling(window).sum().replace(0,np.nan)
    out['proxy_realized_volatility'] = np.sqrt(rv/window)
    complete = values.notna().all(axis=1).rolling(window+1).sum().eq(window+1)
    out.loc[~complete] = np.nan
    return out.replace([np.inf,-np.inf],np.nan)
