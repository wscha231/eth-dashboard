"""Small, frozen candidates and an append-only prospective forecast ledger."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from forecasting.daily_data import GRACE, model_daily_rows, utc_timestamp
from research_pipeline.protocol import FLOW_COLUMNS, PROTOCOL, PROTOCOL_HASH


def connect(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.executescript("""
      CREATE TABLE IF NOT EXISTS issued (
        id INTEGER PRIMARY KEY, protocol_hash TEXT NOT NULL, origin TEXT NOT NULL,
        horizon INTEGER NOT NULL, candidate TEXT NOT NULL, target TEXT NOT NULL,
        payload TEXT NOT NULL, UNIQUE(protocol_hash,origin,horizon,candidate));
      CREATE TRIGGER IF NOT EXISTS issued_no_update BEFORE UPDATE ON issued
        BEGIN SELECT RAISE(ABORT, 'Issued forecasts are immutable'); END;
      CREATE TRIGGER IF NOT EXISTS issued_no_delete BEFORE DELETE ON issued
        BEGIN SELECT RAISE(ABORT, 'Issued forecasts are immutable'); END;
      CREATE TABLE IF NOT EXISTS actual_revisions (
        id INTEGER PRIMARY KEY, issued_id INTEGER NOT NULL, price REAL NOT NULL,
        observed_at TEXT NOT NULL, source_hash TEXT NOT NULL);
      CREATE INDEX IF NOT EXISTS actual_lookup ON actual_revisions(issued_id,id);
    """)
    return conn


def save_issue(conn, payload):
    # A retry cannot overwrite an earlier forecast, even if inputs were revised.
    with conn:
        cursor = conn.execute("INSERT OR IGNORE INTO issued(protocol_hash,origin,horizon,candidate,target,payload) VALUES(?,?,?,?,?,?)",
                              (payload['protocol_hash'], payload['origin'], payload['horizon'],
                               payload['candidate'], payload['target'], json.dumps(payload, sort_keys=True, allow_nan=False)))
    return cursor.rowcount


def features(master, flow, *, now):
    bars = model_daily_rows(master, now=now)
    close = bars.eth_close.astype(float)
    daily_return = close.pct_change(fill_method=None)
    x = pd.DataFrame(index=bars.index)
    for n in (1, 3, 7, 14, 30, 60, 90):
        x[f'eth_momentum_{n}'] = close.pct_change(n, fill_method=None)
    for n in (7, 30, 90):
        x[f'volatility_{n}'] = daily_return.rolling(n).std()
    for n in (20, 60):
        x[f'distance_ma_{n}'] = close / close.rolling(n).mean()-1
    x['volume_ratio_30'] = bars.eth_volume / bars.eth_volume.rolling(30).mean()
    for n in (7, 30):
        x[f'btc_momentum_{n}'] = bars.btc_close.pct_change(n, fill_method=None)
        x[f'relative_eth_btc_{n}'] = x[f'eth_momentum_{n}'] - x[f'btc_momentum_{n}']
    price_columns = list(x)
    source = flow.loc[flow.market_data_excluded.eq(0), FLOW_COLUMNS].copy()
    source.index = pd.to_datetime(source.index, utc=True).tz_localize(None)+pd.Timedelta(days=1)
    if source.index.has_duplicates:
        raise ValueError('Duplicate flow days')
    x = x.join(source).replace([np.inf, -np.inf], np.nan)
    return x, price_columns, close, daily_return


def training_targets(x, close, daily_return, horizon, origin):
    cutoff = pd.Timestamp(origin).replace(day=1)
    cfg = PROTOCOL['event_threshold'][str(horizon)]
    threshold = (daily_return.rolling(30).std()*np.sqrt(horizon)*cfg['multiplier']).clip(cfg['floor'], cfg['cap'])
    target_return = close.shift(-horizon)/close-1
    labels = pd.Series(np.where(target_return >= threshold, 2, np.where(target_return <= -threshold, 0, 1)), index=x.index)
    train = ((x.index+pd.Timedelta(days=horizon) < cutoff)
             & (x.index >= cutoff-pd.DateOffset(years=PROTOCOL['train_years']))
             & target_return.notna() & threshold.notna() & x[FLOW_COLUMNS].notna().all(axis=1))
    return train, target_return, labels, threshold, cutoff


def issue_candidates(conn, master, flow, source, *, source_hashes, now=None):
    started = time.monotonic()
    current = utc_timestamp(now)
    x, price_columns, close, daily_return = features(master, flow, now=current)
    origin = x.index[-1]
    if origin != current.tz_localize(None).floor('D'):
        raise ValueError('Research needs the current UTC origin; stale forecasts are not backdated')
    skips, saved = [], 0
    code_hash = hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path(__file__).parent.glob('*.py')))).hexdigest()
    with threadpool_limits(limits=2):
        for horizon in PROTOCOL['horizons']:
            train, y, labels, threshold, cutoff = training_targets(x, close, daily_return, horizon, origin)
            training_data_hash = hashlib.sha256(pd.util.hash_pandas_object(
                x.loc[train].join(y.rename('target_return')).join(labels.rename('target_class')), index=True
            ).to_numpy().tobytes()).hexdigest()
            for name in PROTOCOL['candidates']:
                if conn.execute('SELECT 1 FROM issued WHERE protocol_hash=? AND origin=? AND horizon=? AND candidate=?',
                                (PROTOCOL_HASH, origin.isoformat(), horizon, name)).fetchone():
                    continue
                reason = None
                if int(train.sum()) < PROTOCOL['training_min_rows']:
                    reason = 'insufficient matched training history'
                elif not x.loc[origin, price_columns].notna().all():
                    reason = 'current price features incomplete'
                elif name == 'hist_price_flow' and (not source['ready_for_current_origin'] or not x.loc[origin, FLOW_COLUMNS].notna().all()):
                    reason = 'current verified flow archive unavailable'
                elif time.monotonic()-started > PROTOCOL['fit_budget_seconds']:
                    reason = 'declared fit budget reached'
                if reason:
                    skips.append({'horizon': horizon, 'candidate': name, 'reason': reason})
                    continue
                if name == 'hist_price_flow':
                    observed = flow.loc[origin-pd.Timedelta(days=1), 'observed_at_utc']
                    if pd.isna(observed) or utc_timestamp(observed) > current:
                        skips.append({'horizon': horizon, 'candidate': name, 'reason': 'flow retrieval time missing or in the future'})
                        continue
                columns = price_columns if name != 'hist_price_flow' else list(x)
                columns = [c for c in columns if x.loc[train, c].notna().mean() >= PROTOCOL['training_feature_min_coverage']]
                fit_started = time.monotonic()
                if name == 'ridge_price':
                    model = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(), Ridge(alpha=PROTOCOL['ridge_alpha']))
                    classifier = make_pipeline(SimpleImputer(keep_empty_features=True), StandardScaler(),
                                               LogisticRegression(C=PROTOCOL['logistic_C'], max_iter=500, random_state=PROTOCOL['seed']))
                else:
                    model = HistGradientBoostingRegressor(random_state=PROTOCOL['seed'], **PROTOCOL['hgb'])
                    classifier = HistGradientBoostingClassifier(random_state=PROTOCOL['seed'], **PROTOCOL['hgb'])
                model.fit(x.loc[train, columns], y.loc[train])
                classifier.fit(x.loc[train, columns], labels.loc[train])
                pred = float(model.predict(x.loc[[origin], columns])[0])
                probability = np.zeros(3)
                probability[classifier.classes_.astype(int)] = classifier.predict_proba(x.loc[[origin], columns])[0]
                if not np.isfinite(pred) or pred <= -1 or not np.isfinite(probability).all() or not np.isclose(probability.sum(), 1):
                    raise ValueError('Invalid candidate output')
                issued_at = utc_timestamp() if now is None else current
                if issued_at.floor('D').tz_localize(None) != origin:
                    raise ValueError('Issuance crossed UTC day; retry with fresh source')
                prior = np.bincount(labels.loc[train], minlength=3)/int(train.sum())
                payload = {'protocol_hash': PROTOCOL_HASH, 'code_hash': code_hash, 'candidate': name,
                           'origin': origin.isoformat(), 'target': (origin+pd.Timedelta(days=horizon)).isoformat(),
                           'horizon': horizon, 'issued_at': issued_at.isoformat(), 'reference_price': float(close.iloc[-1]),
                           'predicted_return': pred, 'predicted_price': float(close.iloc[-1]*(1+pred)),
                           'probability_down_flat_up': probability.tolist(), 'training_prior_down_flat_up': prior.tolist(),
                           'event_threshold': float(threshold.loc[origin]), 'training_rows': int(train.sum()),
                           'training_cutoff': cutoff.isoformat(),
                           'training_data_hash': training_data_hash,
                           'last_training_target': (x.index[train][-1]+pd.Timedelta(days=horizon)).isoformat(),
                           'source_hashes': source_hashes, 'features_at_issue': x.loc[origin, columns].to_dict(),
                           'flow_observed_at': str(flow.loc[origin-pd.Timedelta(days=1), 'observed_at_utc']) if name == 'hist_price_flow' else None,
                           'fit_seconds': time.monotonic()-fit_started, 'status': 'research_only'}
                saved += save_issue(conn, payload)
    return {'new_issues': saved, 'skips': skips, 'fit_seconds': time.monotonic()-started}


def settle(conn, master, *, source_hash, now=None):
    # Exact source date only. Missing target days never borrow a nearby close.
    prices = master.eth_close.copy()
    prices.index = pd.to_datetime(prices.index, utc=True)
    if prices.index.has_duplicates:
        raise ValueError('Duplicate actual source days')
    current = utc_timestamp(now)
    count = 0
    for row in conn.execute('SELECT * FROM issued').fetchall():
        target = utc_timestamp(row['target'])
        source_day = target-pd.Timedelta(days=1)
        if target+GRACE > current or source_day not in prices.index:
            continue
        price = float(prices.loc[source_day])
        if not np.isfinite(price) or price <= 0:
            continue
        previous = conn.execute('SELECT price FROM actual_revisions WHERE issued_id=? ORDER BY id DESC LIMIT 1', (row['id'],)).fetchone()
        if previous is None or not np.isclose(previous['price'], price, rtol=1e-10, atol=1e-8):
            with conn:
                conn.execute('INSERT INTO actual_revisions(issued_id,price,observed_at,source_hash) VALUES(?,?,?,?)',
                             (row['id'], price, current.isoformat(), source_hash))
            count += 1
    return count


def nonoverlap_count(origins, horizon):
    last = None
    count = 0
    for origin in sorted(pd.to_datetime(origins)):
        if last is None or origin >= last+pd.Timedelta(days=max(30, horizon)):
            count += 1
            last = origin
    return count


def block_interval(gain, origins, horizon):
    # Calendar blocks retain missing issue days; row blocks would conceal gaps.
    series = pd.Series(np.asarray(gain), index=pd.to_datetime(origins)).sort_index()
    series = series.reindex(pd.date_range(series.index.min(), series.index.max()))
    values = series.to_numpy(); block = max(30, horizon)
    rng = np.random.default_rng(PROTOCOL['seed'])
    means = []
    for _ in range(PROTOCOL['bootstrap_replicates']):
        starts = rng.integers(0, len(values), int(np.ceil(len(values)/block)))
        sample = np.concatenate([values[(s+np.arange(block)) % len(values)] for s in starts])[:len(values)]
        if np.isfinite(sample).any():
            means.append(float(np.nanmean(sample)))
    return np.quantile(means, [.025, .975]).tolist()


def report(conn, source, attempt, *, now=None):
    rows = conn.execute('''SELECT i.*, a.price AS actual_price FROM issued i LEFT JOIN actual_revisions a
        ON a.id=(SELECT MAX(id) FROM actual_revisions WHERE issued_id=i.id)
        WHERE i.protocol_hash=? ORDER BY i.origin,i.horizon,i.candidate''', (PROTOCOL_HASH,)).fetchall()
    issued = [dict(json.loads(r['payload']), actual_price=r['actual_price']) for r in rows]
    results, resolved = [], []
    for row in issued:
        if row['actual_price'] is None:
            continue
        actual = row['actual_price']/row['reference_price']-1
        label = 2 if actual >= row['event_threshold'] else 0 if actual <= -row['event_threshold'] else 1
        onehot = np.eye(3)[label]
        resolved.append(dict(row, actual_return=actual, actual_class=label,
                             loss=abs(row['predicted_return']-actual), base_loss=abs(actual),
                             brier=float(np.square(np.array(row['probability_down_flat_up'])-onehot).sum()),
                             prior_brier=float(np.square(np.array(row['training_prior_down_flat_up'])-onehot).sum())))
    for horizon in PROTOCOL['horizons']:
        for candidate in PROTOCOL['candidates']:
            group = [r for r in resolved if r['horizon'] == horizon and r['candidate'] == candidate]
            metrics = {'horizon': horizon, 'candidate': candidate, 'resolved': len(group),
                       'review_eligible': False, 'promotion': 'research_only'}
            if group:
                loss = np.array([r['loss'] for r in group]); base = np.array([r['base_loss'] for r in group])
                actual = np.array([r['actual_return'] for r in group]); pred = np.array([r['predicted_return'] for r in group])
                brier = np.mean([r['brier'] for r in group]); prior = np.mean([r['prior_brier'] for r in group])
                blocks = nonoverlap_count([r['origin'] for r in group], horizon)
                enough = len(group) >= PROTOCOL['review_min_rows'] and blocks >= PROTOCOL['review_min_nonoverlapping_blocks']
                ci = block_interval(base-loss, [r['origin'] for r in group], horizon) if enough else None
                rmse = float(np.sqrt(np.mean((pred-actual)**2))); base_rmse = float(np.sqrt(np.mean(actual**2)))
                metrics.update(return_mae=float(loss.mean()), return_rmse=rmse,
                               mae_skill_vs_no_change=float(1-loss.mean()/base.mean()) if base.mean() else None,
                               multiclass_brier=float(brier), brier_skill_vs_training_prior=float(1-brier/prior) if prior else None,
                               nonoverlapping_blocks=blocks, mean_mae_improvement_95ci=ci,
                               actual_class_counts={str(k):sum(r['actual_class'] == k for r in group) for k in range(3)},
                               review_eligible=bool(enough and ci[0] > 0 and rmse < base_rmse and brier < prior))
            results.append(metrics)
    ablations = []
    for horizon in PROTOCOL['horizons']:
        price = {r['origin']: r for r in resolved if r['horizon'] == horizon and r['candidate'] == 'hist_price'}
        flow = {r['origin']: r for r in resolved if r['horizon'] == horizon and r['candidate'] == 'hist_price_flow'}
        overlap = sorted(price.keys() & flow.keys())
        common = [o for o in overlap if price[o]['reference_price'] == flow[o]['reference_price']
                  and price[o].get('training_data_hash') == flow[o].get('training_data_hash')]
        gain = [price[o]['loss']-flow[o]['loss'] for o in common]
        enough = len(common) >= PROTOCOL['review_min_rows'] and nonoverlap_count(common, horizon) >= PROTOCOL['review_min_nonoverlapping_blocks']
        ablations.append({'horizon': horizon, 'matched_origins': len(common),
                          'excluded_different_input_vintages': len(overlap)-len(common),
                          'flow_mae_improvement': float(np.mean(gain)) if gain else None,
                          '95ci': block_interval(gain, common, horizon) if enough else None})
    latest_origin = max((r['origin'] for r in issued), default=None)
    latest = [{k:v for k,v in r.items() if k not in ('features_at_issue', 'source_hashes')} for r in issued if r['origin'] == latest_origin]
    return {'generated_at': utc_timestamp(now).isoformat(), 'protocol': PROTOCOL, 'protocol_hash': PROTOCOL_HASH,
            'status': 'research_only', 'source': source, 'attempt': attempt, 'issued_count': len(issued),
            'resolved_count': len(resolved), 'pending_count': len(issued)-len(resolved),
            'latest_origin': latest_origin, 'latest': latest, 'metrics': results, 'source_ablation': ablations,
            'comparison_note': 'No-change and training-prior benchmarks use identical origins. Source ablation uses paired origins. Research probabilities include FLAT; production conditional probabilities have a different denominator.'}
