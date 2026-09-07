import json

import numpy as np
import pandas as pd
import pytest

from research_pipeline.price_proxies import build_price_proxies
from research_pipeline.volatility_history import backfill_dvol


def bars():
    idx=pd.date_range('2020-01-01',periods=90,freq='h',tz='UTC')
    close=100+np.arange(90)*.1+np.sin(np.arange(90))
    return pd.DataFrame({'open':close,'high':close+2,'low':close-2,
                         'close':close,'volume':100.},index=idx)


def test_proxies_do_not_change_when_future_changes():
    frame=bars(); before=build_price_proxies(frame,volume_unit='native')
    frame.loc[frame.index[60]:,['open','high','low','close']] *= 3
    frame.loc[frame.index[60]:,'volume'] *= 10
    after=build_price_proxies(frame,volume_unit='native')
    pd.testing.assert_frame_equal(before.iloc[:60],after.iloc[:60])
    assert after.notna().all(axis=1).sum() > 0


def test_missing_bar_invalidates_entire_affected_window():
    frame=bars().drop(bars().index[40])
    result=build_price_proxies(frame,volume_unit='native')
    assert result.iloc[40:65].isna().all().all()
    assert result.iloc[65].notna().all()


def test_proxy_volume_units_are_explicit_and_equivalent():
    frame=bars(); native=build_price_proxies(frame,volume_unit='native')
    frame.volume *= frame.close
    quoted=build_price_proxies(frame,volume_unit='quote')
    pd.testing.assert_series_equal(native.proxy_price_impact,quoted.proxy_price_impact)
    with pytest.raises(ValueError):build_price_proxies(frame,volume_unit='unknown')


def test_dvol_pagination_keeps_closed_candles_and_receipt_times(tmp_path):
    day=lambda s:int(pd.Timestamp(s,tz='UTC').timestamp()*1000)
    responses=[{'result':{'data':[[day('2021-04-02'),90,100,80,95],
        [day('2021-04-03'),90,100,80,95]],'continuation':day('2021-04-02')}},
        {'result':{'data':[[day('2021-04-01'),90,100,80,95]],'continuation':None}}]
    calls=[]
    def fetch(url,timeout):
        calls.append(url);return json.dumps(responses.pop(0)).encode()
    f,r=backfill_dvol(tmp_path,start='2021-04-01',end='2021-04-03',fetcher=fetch)
    assert r['complete'] and len(f)==2 and len(calls)==2
    assert f.received_at.min() > f.event_end.max()


def test_dvol_empty_or_stuck_pages_do_not_pass(tmp_path):
    empty=lambda *_:json.dumps({'result':{'data':[],'continuation':None}}).encode()
    _,r=backfill_dvol(tmp_path,start='2021-04-01',end='2021-04-03',fetcher=empty)
    assert not r['complete'] and r['missing_days']==2
    stuck=lambda *_:json.dumps({'result':{'data':[],'continuation':1617408000000}}).encode()
    _,r=backfill_dvol(tmp_path,start='2021-04-01',end='2021-04-03',fetcher=stuck)
    assert not r['complete'] and r['errors']
