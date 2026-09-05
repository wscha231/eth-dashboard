"""Validate new archive boundaries and prospective evidence, without network I/O."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
from urllib.error import HTTPError
import zipfile

import numpy as np
import pandas as pd
import pytest

from research_pipeline.collect import collect_incremental, parse_archive
from research_pipeline.forward import connect, features, issue_candidates, nonoverlap_count, report, save_issue, settle, training_targets
from research_pipeline.market_features import REQUIRED_STREAM_IDS, build_market_daily_features
from research_pipeline.protocol import FLOW_COLUMNS, PROTOCOL_HASH


def archive(day, stream, bad=None):
    unit = 'us' if 'spot' in stream and day >= '2025-01-01' else 'ms'
    dates = pd.date_range(day, periods=24, freq='h', tz='UTC')
    scale = 1000 if unit == 'us' else 1_000_000
    rows = [[int(d.value//scale), 100, 102, 99, 101, 10,
             int((d+pd.Timedelta(hours=1)).value//scale)-1, 1000, 20, 6, 600, 0] for d in dates]
    if bad == 'missing': rows.pop(12)
    if bad == 'duplicate': rows[12] = rows[11]
    if bad == 'ohlc': rows[0][2] = 98
    if bad == 'unit': rows[0][0] *= 1000
    text = pd.DataFrame(rows).to_csv(index=False, header=False)
    if 'um' in stream:
        text = 'open_time,open,high,low,close,volume,close_time,quote_volume,count,taker_buy_volume,taker_buy_quote_volume,ignore\n'+text
    buff = io.BytesIO()
    with zipfile.ZipFile(buff, 'w', zipfile.ZIP_DEFLATED) as z:
        z.writestr('bars.csv', text)
    return buff.getvalue()


@pytest.mark.parametrize('stream', REQUIRED_STREAM_IDS)
@pytest.mark.parametrize('day', ['2024-12-31', '2025-01-01'])
def test_archive_timestamp_units_and_checksum(stream, day):
    data = archive(day, stream)
    checksum = hashlib.sha256(data).hexdigest()+'  bars.zip'
    frame = parse_archive(data, checksum, 'bars.zip', day, stream)
    assert len(frame) == 24 and frame.open_time.iloc[-1] == pd.Timestamp(day)+pd.Timedelta(hours=23)
    with pytest.raises(ValueError, match='checksum mismatch'):
        parse_archive(data, '0'*64+' bars.zip', 'bars.zip', day, stream)


@pytest.mark.parametrize('bad', ['missing', 'duplicate', 'ohlc', 'unit'])
def test_invalid_archive_is_rejected(bad):
    data = archive('2025-01-01', REQUIRED_STREAM_IDS[0], bad)
    with pytest.raises((ValueError, OverflowError)):
        parse_archive(data, hashlib.sha256(data).hexdigest()+' bars.zip', 'bars.zip', '2025-01-01', REQUIRED_STREAM_IDS[0])


def flow_fixture(index):
    flow = pd.DataFrame(.1, index=index, columns=FLOW_COLUMNS)
    flow.index.name = 'date'
    flow['market_data_excluded'] = 0
    flow['feature_available_at_utc'] = (index+pd.Timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    flow['observed_at_utc'] = (index+pd.Timedelta(days=1, minutes=20)).strftime('%Y-%m-%dT%H:%M:%SZ')
    flow['provenance_kind'] = 'test_fixture'
    return flow


def test_collector_repairs_interior_gap_and_reuses_verified_buffer(tmp_path):
    seed = flow_fixture(pd.date_range('2025-01-01', '2025-01-11').difference(pd.DatetimeIndex(['2025-01-10'])))
    seed.to_csv(tmp_path/'seed.csv')
    calls = []
    def fetch(url, **kwargs):
        calls.append(url)
        day = url.split('-1h-')[1][:10]
        market = 'um' if '/futures/' in url else 'spot'
        symbol = 'ethusdt' if 'ETHUSDT' in url else 'btcusdt'
        stream = f'binance_{market}_{symbol}_1h'
        payload = archive(day, stream)
        return (hashlib.sha256(payload).hexdigest()+'  '+url.rsplit('/',1)[-1].replace('.CHECKSUM','')).encode() if url.endswith('.CHECKSUM') else payload
    flow, status = collect_incremental(tmp_path/'state', bootstrap=tmp_path/'seed.csv', now='2025-01-13T06:00Z', fetch=fetch)
    assert status['ready_for_current_origin'] and status['missing_day_count'] == 0
    assert pd.Timestamp('2025-01-10') in flow.index
    assert pd.Timestamp('2025-01-13') not in flow.index  # never today's unclosed day
    first_count = len(calls)
    repeated, status2 = collect_incremental(tmp_path/'state', bootstrap=tmp_path/'seed.csv', now='2025-01-13T08:00Z', fetch=fetch)
    assert len(calls) == first_count and status2['downloaded_archives'] == 0
    np.testing.assert_allclose(flow[FLOW_COLUMNS], repeated[FLOW_COLUMNS])


def test_denied_archive_stops_without_an_alternate_host(tmp_path):
    flow_fixture(pd.date_range('2025-01-01', periods=1)).to_csv(tmp_path/'seed.csv')
    calls = []
    def denied(url, **kwargs):
        calls.append(url)
        raise HTTPError(url, 403, 'Denied', {}, None)
    _, status = collect_incremental(tmp_path/'state', bootstrap=tmp_path/'seed.csv', now='2025-02-01T06:00Z', fetch=denied)
    assert status['collection_stopped'] and not status['ready_for_current_origin']
    assert 0 < len(calls) <= 4
    assert all(url.startswith('https://data.binance.vision/') for url in calls)


def test_flow_differences_use_calendar_days_and_ignore_future_bars():
    streams = {}
    days = pd.date_range('2025-01-01', periods=10).difference(pd.DatetimeIndex(['2025-01-03']))
    for stream in REQUIRED_STREAM_IDS:
        frames = []
        for day in days:
            stamp = str(day.date()); data = archive(stamp, stream)
            frame = parse_archive(data, hashlib.sha256(data).hexdigest()+' bars.zip', 'bars.zip', stamp, stream)
            frame['taker_buy_quote_volume'] = 400+day.day
            frames.append(frame)
        streams[stream] = pd.concat(frames)
    result = build_market_daily_features(streams, cutoff='2025-01-11', excluded_dates=('2025-01-03',))
    assert np.isnan(result.loc['2025-01-10','eth_spot_signed_taker_flow_ratio_delta_7d'])
    assert result.loc['2025-01-09','eth_spot_signed_taker_flow_ratio_delta_7d'] == pytest.approx(.014)
    truncated = {s:f[f.open_time < pd.Timestamp('2025-01-09')].copy() for s,f in streams.items()}
    prefix = build_market_daily_features(truncated, cutoff='2025-01-09', excluded_dates=('2025-01-03',))
    pd.testing.assert_frame_equal(result.loc[:'2025-01-08'], prefix)


def test_frozen_monthly_purge_issue_immutability_and_probabilities(tmp_path, synthetic_ohlcv_with_companions):
    master = synthetic_ohlcv_with_companions.copy()
    now = master.index[-1]+pd.Timedelta(days=1, hours=6)
    flow = flow_fixture(master.index)
    x, _, close, returns = features(master, flow, now=now)
    train, _, _, _, cutoff = training_targets(x, close, returns, 30, x.index[-1])
    assert (x.index[train]+pd.Timedelta(days=30) < cutoff).all()
    assert x.index[train][-1] < x.index[-1]-pd.Timedelta(days=30)
    conn = connect(tmp_path/'forward.db')
    result = issue_candidates(conn, master, flow, {'ready_for_current_origin':False}, source_hashes={'test':'fixture'}, now=now)
    assert result['new_issues'] == 4 and len(result['skips']) == 2
    result = issue_candidates(conn, master, flow, {'ready_for_current_origin':True}, source_hashes={'test':'fixture'}, now=now)
    assert result['new_issues'] == 2 and not result['skips']
    original = [r['payload'] for r in conn.execute('SELECT * FROM issued')]
    for raw in original:
        row = json.loads(raw)
        assert sum(row['probability_down_flat_up']) == pytest.approx(1)
        assert pd.Timestamp(row['last_training_target']) < pd.Timestamp(row['training_cutoff'])
    master.iloc[-1, master.columns.get_loc('eth_close')] *= 1.2
    result = issue_candidates(conn, master, flow, {'ready_for_current_origin':True}, source_hashes={'test':'revised'}, now=now)
    assert result['new_issues'] == 0
    assert [r['payload'] for r in conn.execute('SELECT * FROM issued')] == original
    with pytest.raises(sqlite3.IntegrityError, match='immutable'):
        conn.execute("UPDATE issued SET payload='{}'")
    conn.close()


def example_issue():
    return {'protocol_hash':PROTOCOL_HASH, 'origin':'2025-01-01T00:00:00', 'target':'2025-01-08T00:00:00',
            'candidate':'ridge_price', 'horizon':7, 'reference_price':100., 'predicted_return':0.,
            'event_threshold':.05, 'probability_down_flat_up':[.1,.8,.1], 'training_prior_down_flat_up':[.3,.4,.3]}


def test_settlement_requires_exact_closed_day_preserves_revision_and_scores_flat(tmp_path):
    conn = connect(tmp_path/'forward.db'); payload = example_issue(); save_issue(conn, payload)
    master = pd.DataFrame({'eth_close':[101., 102.]}, index=pd.to_datetime(['2025-01-06','2025-01-07']))
    assert settle(conn, master, source_hash='a', now='2025-01-08T00:14Z') == 0
    assert settle(conn, master.iloc[:1], source_hash='a', now='2025-01-08T06:00Z') == 0
    assert settle(conn, master, source_hash='a', now='2025-01-08T06:00Z') == 1
    assert settle(conn, master, source_hash='a', now='2025-01-08T06:00Z') == 0
    result = report(conn, {}, {}, now='2025-01-08T06:00Z')
    metric = result['metrics'][0]
    assert metric['resolved'] == 1 and metric['actual_class_counts']['1'] == 1
    assert metric['multiclass_brier'] == pytest.approx(.06)
    assert not metric['review_eligible'] and metric['mean_mae_improvement_95ci'] is None
    master.iloc[-1, 0] = 103
    assert settle(conn, master, source_hash='b', now='2025-01-09T06:00Z') == 1
    assert conn.execute('SELECT COUNT(*) FROM actual_revisions').fetchone()[0] == 2
    assert json.loads(conn.execute('SELECT payload FROM issued').fetchone()[0]) == payload
    assert nonoverlap_count(pd.date_range('2025-01-01', periods=90), 30) == 3
    conn.close()
