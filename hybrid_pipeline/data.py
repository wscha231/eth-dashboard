"""Features and labels with explicit calendar dates and no future fitting."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from forecasting.daily_data import model_daily_rows
from research_pipeline.protocol import FLOW_COLUMNS
from hybrid_pipeline.protocol import PROTOCOL, PROTOCOL_HASH


def frame_hash(frame):
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    digest.update(json.dumps(list(frame.columns)).encode())
    return digest.hexdigest()


def runtime_hash():
    import catboost, torch, sklearn
    digest = hashlib.sha256(PROTOCOL_HASH.encode())
    for path in sorted(Path(__file__).parent.glob('*.py')):
        digest.update(path.name.encode()+path.read_bytes())
    for path in ['forecasting/daily_data.py', 'research_pipeline/protocol.py', 'research_pipeline/forward.py']:
        digest.update((Path(__file__).parents[1]/path).read_bytes())
    digest.update('|'.join([np.__version__, pd.__version__, sklearn.__version__, catboost.__version__, torch.__version__]).encode())
    return digest.hexdigest()


def features(master, flow, *, now=None):
    market = model_daily_rows(master, now=now)
    columns = [f'{asset}_{field}' for asset in ('eth', 'btc') for field in ('open','high','low','close','volume')]
    raw = market[columns].astype(float)
    if not np.isfinite(raw.to_numpy()).all() or (raw.filter(regex='_(open|high|low|close)$') <= 0).any().any() or (raw.filter(like='volume') < 0).any().any():
        raise ValueError('Incomplete/non-finite OHLCV; do not impute raw prices')
    x = pd.DataFrame(index=raw.index)
    for asset in ('eth','btc'):
        close = raw[f'{asset}_close']; ret = np.log(close).diff()
        for n in (1,3,7,14,30,60,90):
            x[f'{asset}_log_return_{n}'] = np.log(close).diff(n)
        x[f'{asset}_range'] = np.log(raw[f'{asset}_high']/raw[f'{asset}_low'])
        x[f'{asset}_body'] = np.log(close/raw[f'{asset}_open'])
        for n in (7,30,90):
            x[f'{asset}_vol_{n}'] = ret.rolling(n).std()
            x[f'{asset}_ma_{n}'] = np.log(close/close.rolling(n).mean())
        for n in (7,30):
            volume = raw[f'{asset}_volume']
            x[f'{asset}_volume_{n}'] = np.log((volume+1)/(volume.rolling(n).mean()+1))
        for n in (30,90):
            x[f'{asset}_drawdown_{n}'] = np.log(close/close.rolling(n).max())
    for n in (7,30):
        x[f'eth_btc_relative_{n}'] = x[f'eth_log_return_{n}']-x[f'btc_log_return_{n}']
    x['eth_btc_correlation_30'] = x.eth_log_return_1.rolling(30).corr(x.btc_log_return_1)
    available = flow.loc[flow.market_data_excluded.eq(0), FLOW_COLUMNS].astype(float).copy()
    # Source labels are bar starts. +1 is close; +1 extra day allows archive lag.
    available.index = pd.to_datetime(available.index, utc=True).tz_localize(None)+pd.Timedelta(days=2)
    if available.index.has_duplicates:
        raise ValueError('Duplicate flow source dates')
    x = x.join(available)
    x['flow_available_share'] = x[FLOW_COLUMNS].notna().mean(axis=1)
    return raw, x.replace([np.inf,-np.inf],np.nan)


def targets(raw, horizon):
    close = raw.eth_close
    sigma = (np.log(close).diff().rolling(30).std()*np.sqrt(horizon)).clip(lower=.03 if horizon == 7 else .08)
    exact = close.reindex(close.index+pd.Timedelta(days=horizon)).to_numpy()
    log_return = pd.Series(np.log(exact/close.to_numpy()), index=close.index)
    return log_return/sigma, sigma


def training_positions(index, y, month, horizon, kind):
    keep = ((index >= month-pd.DateOffset(years=PROTOCOL['train_years'][str(horizon)][kind]))
            & (index+pd.Timedelta(days=horizon) < month-pd.Timedelta(days=PROTOCOL['embargo_days'][str(horizon)]))
            & (np.arange(len(index)) >= PROTOCOL['sequence_warmup_days']) & y.notna())
    return np.flatnonzero(keep)


def inner_split(index, positions, horizon):
    last = index[positions[-1]]
    validation = positions[index[positions] > last-pd.Timedelta(days=PROTOCOL['inner_validation_days'])]
    cutoff = index[validation[0]]
    train = positions[index[positions]+pd.Timedelta(days=horizon) < cutoff-pd.Timedelta(days=PROTOCOL['embargo_days'][str(horizon)])]
    if len(train) < PROTOCOL['minimum_inner_training_rows']:
        raise ValueError('Insufficient purged inner training rows')
    return train, validation


class PastScaler:
    """Training-only median/scale; an unseen feature stays zero until next fit."""
    def fit(self, values):
        a = np.asarray(values, dtype=float)
        self.seen = np.isfinite(a).sum(axis=0) >= max(20, .5*len(a))
        self.median = np.zeros(a.shape[1]); self.scale = np.ones(a.shape[1])
        for c in np.flatnonzero(self.seen):
            self.median[c] = np.nanmedian(a[:,c])
            self.scale[c] = max(float(np.nanstd(a[:,c])),1e-6)
        return self

    def transform(self, values):
        a = np.asarray(values,dtype=float).copy()
        a = np.where(np.isfinite(a),a,self.median)
        a = np.clip((a-self.median)/self.scale,-8,8)
        a[:,~self.seen] = 0
        return a.astype(np.float32)

    def state(self):
        return {k:getattr(self,k).tolist() for k in ('seen','median','scale')}

    @classmethod
    def from_state(cls, state):
        obj = cls()
        for k,v in state.items():
            setattr(obj,k,np.asarray(v,dtype=bool if k == 'seen' else float))
        return obj
