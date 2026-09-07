import json
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse

import pandas as pd
import pytest

from research_pipeline.funding_history import backfill, daily_funding, parse_funding, windows


def payload(start, end):
    return json.dumps({"result": [{"timestamp": int(t.timestamp()*1000),
        "interest_1h": .001, "interest_8h": .008, "index_price": 100.}
        for t in pd.date_range(pd.Timestamp(start)+pd.Timedelta(hours=1), end, freq="h")]}).encode()


def source(url, timeout):
    query = parse_qs(urlparse(url).query)
    return payload(pd.to_datetime(int(query['start_timestamp'][0]), unit='ms', utc=True),
                   pd.to_datetime(int(query['end_timestamp'][0]), unit='ms', utc=True))


def test_requested_windows_are_short_and_cover_interval():
    chunks = list(windows('2020-01-01', '2020-04-01'))
    assert sum((b-a).total_seconds() for a,b in chunks) == 91*86400
    assert all(b-a <= pd.Timedelta(days=28) for a,b in chunks)
    assert all(chunks[i][1] == chunks[i+1][0] for i in range(len(chunks)-1))


def test_partial_response_never_certifies_full_history(tmp_path):
    report = backfill(tmp_path, start='2020-01-01', end='2020-01-04',
        fetcher=lambda *_: payload('2020-01-03', '2020-01-04'))[1]
    assert report['rows'] == 24 and report['missing_hours'] == 48
    assert not report['complete']
    assert not list((tmp_path/'cache').glob('*.json'))


def test_cache_reuse_preserves_actual_receipt_and_integrity(tmp_path):
    first, report = backfill(tmp_path, start='2020-01-01', end='2020-02-01', fetcher=source)
    assert report['complete'] and report['rows'] == 744
    def forbidden(*_):
        raise AssertionError('cached history must not refetch')
    second, resumed = backfill(tmp_path, start='2020-01-01', end='2020-02-01', fetcher=forbidden)
    pd.testing.assert_frame_equal(first, second)
    assert resumed['cached_windows'] == 2
    assert first.received_at.min() > first.event_time.max()
    assert resumed['live_eligible'] is False
    raw = next((tmp_path/'raw').glob('*.gz'))
    raw.write_bytes(b'corrupt')
    with pytest.raises((ValueError, OSError)):
        backfill(tmp_path, start='2020-01-01', end='2020-02-01', fetcher=forbidden)


def test_daily_aggregation_requires_full_disjoint_intervals():
    hourly = parse_funding(payload('2020-01-01', '2020-01-03'),
                           '2020-01-01', '2020-01-03', '2026-09-07')
    daily = daily_funding(hourly)
    assert len(daily) == 2
    assert daily.funding_sum_1h.tolist() == pytest.approx([.024, .024])
    assert str(daily.iloc[0].event_end) == '2020-01-02 00:00:00+00:00'
    assert len(daily_funding(hourly.drop(index=7))) == 1


def test_denied_source_stops_pending_calls(tmp_path):
    calls = []
    def denied(url, timeout):
        calls.append(url)
        raise HTTPError(url, 451, 'Denied', {}, None)
    _, report = backfill(tmp_path, start='2020-01-01', end='2020-04-01',
                         workers=1, fetcher=denied)
    assert len(calls) == 1 and not report['complete']


@pytest.mark.parametrize('mutation', ['duplicate', 'ms_unit', 'future', 'nonfinite'])
def test_invalid_records_are_rejected(mutation):
    data = json.loads(payload('2020-01-01', '2020-01-02'))
    if mutation == 'duplicate': data['result'].append(data['result'][0])
    if mutation == 'ms_unit': data['result'][0]['timestamp'] /= 1000
    if mutation == 'future': data['result'][0]['timestamp'] = 4102444800000
    if mutation == 'nonfinite': data['result'][0]['interest_1h'] = float('nan')
    with pytest.raises(ValueError):
        parse_funding(json.dumps(data).encode(), '2020-01-01', '2020-01-02', '2026-09-07')
