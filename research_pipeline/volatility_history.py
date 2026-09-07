"""Daily ETH DVOL history, with explicit pagination and closed-candle checks."""
import gzip
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd

from .funding_history import fetch_bytes, utc

HOST = "https://www.deribit.com/api/v2/public/get_volatility_index_data"


def backfill_dvol(root, *, start, end, max_requests=8, budget_seconds=120, fetcher=fetch_bytes):
    start, end = utc(start), utc(end)
    if start >= end or start != start.floor('D') or end != end.floor('D'):
        raise ValueError('whole UTC days and start < end required')
    if end > pd.Timestamp.now(tz='UTC').floor('D'):
        raise ValueError('end must exclude the current unfinished day')
    root = Path(root); (root/'raw').mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic()+budget_seconds
    cursor = int(end.timestamp()*1000); rows = []; records = []; errors = []
    finished = False
    for _ in range(max_requests):
        if time.monotonic() >= deadline:
            break
        url = HOST+'?'+urlencode({'currency':'ETH', 'resolution':'1D',
            'start_timestamp':int(start.timestamp()*1000), 'end_timestamp':cursor})
        try:
            raw = fetcher(url, min(20, deadline-time.monotonic()))
            received = pd.Timestamp.now(tz='UTC')
            payload = json.loads(raw)
            result = payload.get('result', {})
            if 'error' in payload or not isinstance(result, dict) or not isinstance(result.get('data'), list):
                raise ValueError('invalid DVOL response')
            sha = hashlib.sha256(raw).hexdigest()
            (root/'raw'/(sha+'.json.gz')).write_bytes(gzip.compress(raw, mtime=0))
            records.append({'url':url,'sha256':sha,'received_at':received.isoformat()})
            for row in result['data']:
                if len(row) != 5 or not np.isfinite(np.asarray(row, dtype=float)).all():
                    raise ValueError('invalid DVOL candle')
                stamp, opening, high, low, close = row
                if stamp % 86_400_000 or min(opening, high, low, close) <= 0:
                    raise ValueError('invalid daily timestamp or DVOL value')
                if high < max(opening, close, low) or low > min(opening, close, high):
                    raise ValueError('invalid DVOL OHLC bounds')
                day = pd.to_datetime(stamp, unit='ms', utc=True)
                if start <= day < end:
                    rows.append({'date':day,'dvol_close_pct':float(close),
                                 'event_end':day+pd.Timedelta(days=1),'received_at':received})
            continuation = result.get('continuation')
            if continuation is None or continuation <= start.timestamp()*1000:
                finished = True
                break
            if not isinstance(continuation, int) or continuation >= cursor:
                raise ValueError('non-progressing DVOL continuation')
            cursor = continuation
        except Exception as exc:
            errors.append({'error':type(exc).__name__,'detail':str(exc)[:180]})
            break
    frame = pd.DataFrame(rows, columns=['date','dvol_close_pct','event_end','received_at'])
    if len(frame):
        if (frame.groupby('date').dvol_close_pct.nunique() > 1).any():
            raise ValueError('conflicting values across DVOL pages')
        frame = frame.drop_duplicates('date').sort_values('date').reset_index(drop=True)
    expected = pd.date_range(start,end-pd.Timedelta(days=1),freq='D')
    actual = pd.DatetimeIndex(pd.to_datetime(frame.date,utc=True))
    missing = expected.difference(actual)
    generated = pd.Timestamp.now(tz='UTC')
    report = {'generated_at':generated.isoformat(),'source':'deribit_eth_dvol_1d',
              'start':start.isoformat(),'end_exclusive':end.isoformat(),'rows':len(frame),
              'expected_days':len(expected),'missing_days':len(missing),
              'missing_sample':[t.isoformat() for t in missing[:30]],
              'complete':finished and not errors and len(missing)==0,
              'pagination_finished':finished,'errors':errors,'records':records,
              'history_kind':'historical_reconstruction_observed_now',
              'live_eligible':False,'commercial_use':'separate_terms_review_required'}
    name='dvol_daily_'+generated.strftime('%Y%m%dT%H%M%S%fZ')+'.parquet'
    frame.to_parquet(root/name,index=False)
    report['snapshot']=name
    report['snapshot_sha256']=hashlib.sha256((root/name).read_bytes()).hexdigest()
    (root/(name.replace('.parquet','.json'))).write_text(json.dumps(report,indent=2))
    (root/'latest_report.json').write_text(json.dumps(report,indent=2))
    return frame,report
