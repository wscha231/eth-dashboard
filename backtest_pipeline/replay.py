"""Monthly purged replay with checkpoint reuse and past-only model selection."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

import eth_price_forecast as efp
from forecasting.daily_data import model_daily_rows, utc_timestamp
from forecasting.model_bundle import code_fingerprint, frame_hash
from research_pipeline.forward import features as compact_features, block_interval
from research_pipeline.protocol import FLOW_COLUMNS
from backtest_pipeline.protocol import CANDIDATES, PROTOCOL, PROTOCOL_HASH


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':'))+'\n')
    temporary.replace(path)


def runtime_hash():
    files = sorted(Path(__file__).parent.glob('*.py'))
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in files))
    digest.update((code_fingerprint()+PROTOCOL_HASH+sklearn.__version__+pd.__version__+np.__version__).encode())
    # The compact feature/CI helpers are reused, so changes invalidate caches.
    for name in ('forward.py', 'protocol.py'):
        digest.update((Path(__file__).parents[1]/'research_pipeline'/name).read_bytes())
    return digest.hexdigest()


def train_mask(index, month, horizon):
    return ((index >= month-pd.DateOffset(years=PROTOCOL['train_years'][str(horizon)]))
            & (index+pd.Timedelta(days=horizon) < month-pd.Timedelta(days=PROTOCOL['embargo_days'][str(horizon)])))


def asof_wide_features(market, horizon, month):
    # Production infers vendor cadence from a series. Freeze that inference at
    # the fit cutoff so later test observations cannot choose past transforms.
    original = efp.infer_feature_update_interval_days
    def past_only(series):
        return original(series.loc[series.index < month])
    with patch.object(efp, 'infer_feature_update_interval_days', side_effect=past_only):
        return efp.build_features(market, horizon=horizon)


def fit_month(market, compact, price_columns, horizon, month, test_index):
    started = time.monotonic()
    wide, candidates = asof_wide_features(market.loc[:test_index[-1]], horizon, month)
    train = train_mask(wide.index, month, horizon) & wide.target_return.notna()
    # Matched training origins also make the compact price/flow ablation fair.
    train &= compact[FLOW_COLUMNS].notna().all(axis=1).reindex(wide.index, fill_value=False)
    if int(train.sum()) < PROTOCOL['minimum_training_rows']:
        return None
    positions = np.flatnonzero(train)
    selected = efp.select_fold_features(wide, candidates, positions, .03, horizon)
    if not selected:
        raise ValueError('No causal training features selected')
    y = wide.loc[train, 'target_return']
    weights = efp.build_time_decay_sample_weights(y.index, '1d', horizon)
    prep_seconds = time.monotonic()-started
    values = {'no_change_anchor': np.zeros(len(test_index))}
    fit_times = {}
    templates = efp.make_models(horizon)
    with threadpool_limits(limits=2):
        for name in ('extra_trees', 'random_forest', 'knn_regressor'):
            begin = time.monotonic(); model = clone(templates[name])
            efp.fit_model_with_optional_sample_weight(model, wide.loc[train, selected], y, weights)
            pred = model.predict(wide.loc[test_index, selected])
            values[name] = efp.apply_regime_response_overlay(pred, wide.loc[test_index], horizon).to_numpy()
            fit_times[name] = time.monotonic()-begin
        values['rf_knn_equal'] = (values['random_forest']+values['knn_regressor'])/2
        for name in ('ridge_compact', 'hist_price_compact', 'hist_price_flow_compact'):
            begin = time.monotonic()
            columns = price_columns if name != 'hist_price_flow_compact' else [*price_columns, *FLOW_COLUMNS]
            columns = [c for c in columns if compact.loc[y.index, c].notna().mean() >= .60]
            if name == 'ridge_compact':
                model = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                                      Ridge(alpha=PROTOCOL['compact_models']['ridge_alpha']))
            else:
                params = {k:v for k,v in PROTOCOL['compact_models'].items() if k != 'ridge_alpha'}
                model = HistGradientBoostingRegressor(random_state=PROTOCOL['seed'], **params)
            model.fit(compact.loc[y.index, columns], y)
            values[name] = model.predict(compact.loc[test_index, columns])
            if name == 'hist_price_flow_compact':
                values[name][~compact.loc[test_index, FLOW_COLUMNS].notna().all(axis=1)] = np.nan
            fit_times[name] = time.monotonic()-begin
    rows = []
    for name in CANDIDATES:
        for date, predicted in zip(test_index, values[name]):
            if not np.isfinite(predicted):
                if name == 'hist_price_flow_compact':
                    continue
                raise ValueError(f'Non-finite forecast: {name}')
            if predicted <= -1:
                raise ValueError(f'Non-positive point price: {name}; do not hide failed predictions')
            rows.append({'origin':date.isoformat(), 'horizon':horizon, 'model':name,
                         'predicted_return':float(predicted), 'fit_cutoff':month.isoformat()})
    return {'rows':rows, 'training_rows':len(y), 'last_training_target':(y.index[-1]+pd.Timedelta(days=horizon)).isoformat(),
            'feature_count':len(selected), 'selected_features':selected, 'prepare_seconds':prep_seconds,
            'fit_seconds':fit_times, 'total_seconds':time.monotonic()-started}


def replay_horizon(master, flow, horizon, cache_dir, *, now=None, budget_seconds=1500):
    started = time.monotonic(); market = model_daily_rows(master, now=now)
    compact, price_columns, _, _ = compact_features(master, flow, now=now)
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    last_origin = market.index[-1]-pd.Timedelta(days=horizon)
    fingerprint = runtime_hash(); records = []; folds = []; reused = 0
    for month in pd.date_range(market.index[0].replace(day=1), last_origin, freq='MS'):
        if time.monotonic()-started > budget_seconds:
            raise TimeoutError('Full replay exceeded its declared per-horizon budget; completed months remain cached')
        test_index = market.index[(market.index >= month)&(market.index < month+pd.offsets.MonthBegin(1))&(market.index <= last_origin)]
        if not len(test_index) or train_mask(market.index, month, horizon).sum() < PROTOCOL['minimum_training_rows']:
            continue
        # Prefix ends at the final prediction origin, never at a future target.
        prefix = market.loc[:test_index[-1]]
        flow_prefix = compact.loc[:test_index[-1], FLOW_COLUMNS]
        key = hashlib.sha256((fingerprint+str(horizon)+month.isoformat()+frame_hash(prefix)+frame_hash(flow_prefix)).encode()).hexdigest()
        path = cache_dir/f'h{horizon}_{month:%Y-%m}.json'
        cached = json.loads(path.read_text()) if path.exists() else None
        if cached and cached.get('fingerprint') == key:
            result = cached['result']; reused += 1
        else:
            result = fit_month(market, compact, price_columns, horizon, month, test_index)
            if result is None:
                continue
            atomic_json(path, {'fingerprint':key, 'result':result})
        records.extend(result['rows'])
        folds.append({k:v for k,v in result.items() if k not in ('rows','selected_features')} | {'month':month.isoformat()})
        print(f'replay h={horizon} month={month:%Y-%m} cache={bool(cached and cached.get("fingerprint")==key)} elapsed={time.monotonic()-started:.1f}s', flush=True)
    if not records:
        raise ValueError('No eligible replay origins')
    table = pd.DataFrame(records, columns=['origin','horizon','model','predicted_return','fit_cutoff'])
    table['origin'] = pd.to_datetime(table.origin)
    table['target'] = table.origin+pd.to_timedelta(table.horizon, unit='D')
    table['reference_price'] = table.origin.map(market.eth_close)
    table['actual_price'] = table.target.map(market.eth_close)
    table['actual_return'] = table.actual_price/table.reference_price-1
    cfg = efp.classification_direction_threshold_bounds(horizon)
    threshold = (market.eth_close.pct_change(fill_method=None).rolling(30).std()*np.sqrt(horizon)*cfg[0]).clip(cfg[1],cfg[2])
    table['event_threshold'] = table.origin.map(threshold)
    if table[['reference_price','actual_price','actual_return','event_threshold']].isna().any().any():
        raise ValueError('Missing exact target or threshold; refuse an incomplete chart')
    wide = table.pivot(index='origin', columns='model', values='predicted_return')
    matched = wide.dropna().index
    excluded = len(wide)-len(matched)
    table = table.loc[table.origin.isin(matched)].sort_values(['origin','model']).reset_index(drop=True)
    expected = pd.date_range(table.origin.min(), last_origin)
    missing = expected.difference(matched)
    return table, {'horizon':horizon, 'runtime_seconds':time.monotonic()-started, 'months':len(folds),
                   'cached_months':reused, 'refitted_months':len(folds)-reused, 'folds':folds,
                   'matched_origins':len(matched), 'excluded_origins':excluded,
                   'missing_origin_dates':[d.date().isoformat() for d in missing],
                   'last_source_day':str((market.index[-1]-pd.Timedelta(days=1)).date()),
                   'runtime_hash':fingerprint, 'protocol_hash':PROTOCOL_HASH}


def past_only_selector(table):
    selected = []
    for horizon, data in table.groupby('horizon'):
        history = data.copy()
        history['loss'] = abs(history.predicted_return-history.actual_return)
        for month, current in data.groupby(data.origin.dt.to_period('M')):
            cutoff = month.start_time
            available = history[(history.target < cutoff)&(history.target >= cutoff-pd.Timedelta(days=PROTOCOL['selection_lookback_days']))]
            stats = available.groupby('model').loss.agg(['mean','count'])
            enough = len(stats) == len(CANDIDATES) and stats['count'].min() >= PROTOCOL['selection_min_rows']
            winner = min(CANDIDATES, key=lambda name:(stats.loc[name,'mean'], name != 'no_change_anchor', name)) if enough else 'no_change_anchor'
            chosen = current[current.model == winner].copy()
            chosen['selected_model'] = winner
            chosen['selection_samples'] = int(stats['count'].min()) if len(stats) else 0
            chosen['selection_latest_target'] = available.target.max().isoformat() if len(available) else None
            chosen['model'] = 'past_only_selector'
            selected.append(chosen)
    return pd.concat(selected, ignore_index=True)


def score_group(group, horizon, *, confidence=False):
    actual = group.actual_return.to_numpy(); pred = group.predicted_return.to_numpy()
    loss = abs(pred-actual); base = abs(actual); reference = group.reference_price.to_numpy()
    rmse = float(np.sqrt(np.mean((pred-actual)**2))); baseline_rmse = float(np.sqrt(np.mean(actual**2)))
    threshold = group.event_threshold.to_numpy()
    labels = np.where(actual >= threshold, 2, np.where(actual <= -threshold, 0, 1))
    guesses = np.where(pred >= threshold, 2, np.where(pred <= -threshold, 0, 1))
    return {'rows':len(group), 'origin_start':group.origin.min().date().isoformat(), 'origin_end':group.origin.max().date().isoformat(),
            'target_end':group.target.max().date().isoformat(), 'return_mae':float(loss.mean()), 'return_rmse':rmse,
            'price_mae':float(np.mean(loss*reference)), 'price_rmse':float(np.sqrt(np.mean(((pred-actual)*reference)**2))),
            'mae_skill_vs_no_change':float(1-loss.mean()/base.mean()) if base.mean() else None,
            'rmse_skill_vs_no_change':float(1-rmse/baseline_rmse) if baseline_rmse else None,
            'state_accuracy_all_origins':float(np.mean(labels == guesses)),
            'up_recall':float(np.mean(guesses[labels == 2] == 2)) if (labels == 2).any() else None,
            'down_recall':float(np.mean(guesses[labels == 0] == 0)) if (labels == 0).any() else None,
            'actual_state_counts':{str(k):int(np.sum(labels == k)) for k in (0,1,2)},
            'mean_mae_improvement_95ci':block_interval(base-loss, group.origin, horizon) if confidence else None}


def build_report(table, runtimes, *, source_hashes, now=None):
    table = table.copy(); table.origin = pd.to_datetime(table.origin); table.target = pd.to_datetime(table.target)
    if table.duplicated(['origin','horizon','model']).any():
        raise ValueError('Duplicate OOF origins')
    for h, data in table.groupby('horizon'):
        if set(data.model) != set(CANDIDATES) or not data.groupby('origin').size().eq(len(CANDIDATES)).all():
            raise ValueError('Comparison must use identical origins for every candidate')
        if not (data.target == data.origin+pd.Timedelta(days=h)).all():
            raise ValueError('Target time contract mismatch')
        cutoff = pd.to_datetime(data.fit_cutoff)
        if not (cutoff <= data.origin).all():
            raise ValueError('A model was fitted after the prediction origin')
    selector = past_only_selector(table)
    all_rows = pd.concat([table,selector], ignore_index=True)
    horizons = {}
    for horizon, data in all_rows.groupby('horizon'):
        leaderboard = []
        for model, group in data.groupby('model'):
            leaderboard.append({'model':model, **score_group(group,horizon)})
        fixed = [r for r in leaderboard if r['model'] != 'past_only_selector']
        best = min(fixed,key=lambda r:(r['return_mae'], r['model'] != 'no_change_anchor', r['model']))['model']
        for item in leaderboard:
            if item['model'] in (best,'past_only_selector'):
                group = data[data.model == item['model']]
                item.update(score_group(group,horizon,confidence=True))
        leaderboard.sort(key=lambda r:(r['return_mae'],r['model']))
        recent_start = data.target.max()-pd.Timedelta(days=PROTOCOL['selection_lookback_days'])
        recent = [{'model':name, **score_group(group,horizon)} for name,group in data[data.target > recent_start].groupby('model')]
        yearly = [{'year':int(year),'model':name,**score_group(group,horizon)} for (year,name),group in data.groupby([data.target.dt.year,'model'])]
        recent_best = min((r for r in recent if r['model'] != 'past_only_selector'), key=lambda r:(r['return_mae'],r['model']))['model']
        indexed = data.set_index(['model','origin'])
        base = data[data.model == 'no_change_anchor'].sort_values('origin')
        points = []
        for row in base.itertuples():
            chosen = indexed.loc[('past_only_selector',row.origin)]
            values = {name:float(indexed.loc[(name,row.origin),'predicted_return']) for name in CANDIDATES}
            values['past_only_selector'] = float(chosen.predicted_return)
            points.append({'origin':row.origin.date().isoformat(),'target':row.target.date().isoformat(),
                           'reference_price':float(row.reference_price),'actual_price':float(row.actual_price),
                           'actual_return':float(row.actual_return), 'returns':values,
                           'selected_model':chosen.selected_model,'selection_samples':int(chosen.selection_samples),
                           'selection_latest_target':chosen.selection_latest_target})
        horizons[str(horizon)] = {'best_fixed_model':best,'best_recent_model':recent_best,
                                 'leaderboard':leaderboard,'recent':recent,'yearly':yearly,'points':points,
                                 'matched_origins':len(points),'first_origin':points[0]['origin'],'last_target':points[-1]['target']}
    return {'schema_version':1,'status':'historical_replay','generated_at':utc_timestamp(now).isoformat(),
            'protocol':PROTOCOL,'protocol_hash':PROTOCOL_HASH,'runtime_hash':runtime_hash(),
            'source_hashes':source_hashes,'runtimes':runtimes,'horizons':horizons}, all_rows
