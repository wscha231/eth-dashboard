"""Resumable monthly fits; a current-month bundle makes daily inference cheap."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

from hybrid_pipeline.data import features, targets, training_positions, frame_hash, runtime_hash
from hybrid_pipeline.models import fit_model, predict_model
from hybrid_pipeline.protocol import BASE_MODELS, PROTOCOL, PROTOCOL_HASH


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n')
    tmp.replace(path)


def input_hashes(master, flow):
    return {'master':hashlib.sha256(Path(master).read_bytes()).hexdigest(),
            'flow':hashlib.sha256(Path(flow).read_bytes()).hexdigest()}


def replay(master, flow, horizon, root, *, part='all', now=None, budget_seconds=1500):
    started = time.monotonic(); root = Path(root); cache = root/'cache'; cache.mkdir(parents=True,exist_ok=True)
    raw,x = features(master,flow,now=now); y,sigma = targets(raw,horizon)
    code = runtime_hash(); rows=[]; folds=[]; fitted=0; reused=0; bundle_reused=0
    latest_month = raw.index[-1].replace(day=1)
    for month in pd.date_range(raw.index[0].replace(day=1),latest_month,freq='MS'):
        if part == 'early' and month >= pd.Timestamp('2023-01-01'):
            continue
        if part == 'late' and month < pd.Timestamp('2023-01-01'):
            continue
        if part == 'current' and month != latest_month:
            continue
        positions = {kind:training_positions(x.index,y,month,horizon,kind) for kind in ('short','long')}
        if min(map(len,positions.values())) < PROTOCOL['minimum_training_rows']:
            continue
        if time.monotonic()-started > budget_seconds:
            raise TimeoutError('Replay budget reached; completed month checkpoints are retained')
        test = np.flatnonzero((x.index >= month)&(x.index < month+pd.offsets.MonthBegin(1)))
        end = x.index[test[-1]]
        past = x.index < month
        train_key = hashlib.sha256((code+str(horizon)+month.isoformat()+frame_hash(raw.loc[past])+frame_hash(x.loc[past])).encode()).hexdigest()
        pred_key = hashlib.sha256((train_key+frame_hash(raw.loc[:end])+frame_hash(x.loc[:end])).encode()).hexdigest()
        path = cache/f'h{horizon}_{month:%Y-%m}.json'
        cached = json.loads(path.read_text()) if path.exists() else None
        bundle_path = root/f'bundle_h{horizon}.pt'
        # A valid prediction checkpoint also needs a bundle for today's month.
        bundle_exists = bundle_path.exists()
        if cached and cached.get('fingerprint') == pred_key and (month != latest_month or bundle_exists):
            result = cached['result']; reused += 1
        else:
            states = None; metadata = None
            if month == latest_month and bundle_exists:
                saved = torch.load(bundle_path,map_location='cpu',weights_only=True)
                if saved['training_key'] == train_key and saved['runtime_hash'] == code:
                    states,metadata = saved['states'],saved['fits']; bundle_reused += 1
            if states is None:
                states={};metadata={}
                # Pass only the recorded past prefix to the fitting function.
                for name in BASE_MODELS:
                    kind = name.rsplit('_',1)[-1]
                    states[name],metadata[name] = fit_model(name,x.loc[past],y.loc[past],positions[kind],horizon)
                fitted += 1
                if month == latest_month:
                    tmp = bundle_path.with_suffix('.tmp')
                    torch.save({'training_key':train_key,'runtime_hash':code,'month':month.isoformat(),
                                'horizon':horizon,'states':states,'fits':metadata},tmp)
                    tmp.replace(bundle_path)
            predictions={name:predict_model(states[name],x.loc[:end],test) for name in BASE_MODELS}
            result={'rows':[{'origin':x.index[i].isoformat(),**{name:float(predictions[name][j]) for name in BASE_MODELS}}
                            for j,i in enumerate(test)],'fits':metadata,'training_key':train_key}
            atomic_json(path,{'fingerprint':pred_key,'result':result})
        rows.extend(result['rows'])
        folds.append({'month':month.isoformat(),'fits':result['fits']})
        print(f'hybrid horizon={horizon} month={month:%Y-%m} cached={bool(cached and cached.get("fingerprint")==pred_key)} elapsed={time.monotonic()-started:.1f}s',flush=True)
    if not rows:
        raise ValueError('No eligible origins for this replay part')
    table=pd.DataFrame(rows,columns=['origin',*BASE_MODELS]);table.origin=pd.to_datetime(table.origin)
    table['horizon']=horizon
    table=settle_table(table,raw)
    return table,{'horizon':horizon,'part':part,'months':len(folds),'refitted_months':fitted,
                  'cached_months':reused,'reused_current_bundles':bundle_reused,'runtime_seconds':time.monotonic()-started,
                  'runtime_hash':code,'protocol_hash':PROTOCOL_HASH,'folds':folds}


def settle_table(table, raw):
    table=table.copy();table.origin=pd.to_datetime(table.origin)
    if table.duplicated(['origin','horizon']).any():
        raise ValueError('Duplicate base forecast origins')
    table['target']=table.origin+pd.to_timedelta(table.horizon,unit='D')
    table['reference_price']=table.origin.map(raw.eth_close)
    table['actual_price']=table.target.map(raw.eth_close)
    table['actual_return']=table.actual_price/table.reference_price-1
    for h in table.horizon.unique():
        _,sigma=targets(raw,int(h));mask=table.horizon.eq(h)
        table.loc[mask,'sigma']=table.loc[mask,'origin'].map(sigma)
    if not np.isfinite(table[['reference_price','sigma',*BASE_MODELS]].to_numpy()).all():
        raise ValueError('Missing origin price/volatility or invalid base prediction')
    expected_resolved=table.target <= raw.index[-1]
    if table.loc[expected_resolved,'actual_return'].isna().any():
        raise ValueError('Missing exact matured target; never interpolate')
    return table.sort_values(['horizon','origin']).reset_index(drop=True)
