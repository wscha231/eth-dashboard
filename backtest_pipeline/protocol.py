"""Freeze the comparison before inspecting full-period results."""
import hashlib
import json

CANDIDATES = ['no_change_anchor', 'extra_trees', 'random_forest', 'knn_regressor',
              'rf_knn_equal', 'ridge_compact', 'hist_price_compact', 'hist_price_flow_compact']
PROTOCOL = {
    'version': 'full_history_monthly_v1', 'horizons': [7, 30], 'candidates': CANDIDATES,
    'objective': 'matched-origin return MAE; price RMSE reported separately',
    'train_years': {'7': 3, '30': 5}, 'minimum_training_rows': 500,
    'refit': 'calendar month start', 'embargo_days': {'7': 3, '30': 15},
    'purge': 'training origin + horizon < month start - embargo',
    'feature_selection': 'production coverage/rank/prune on training rows only',
    'vendor_cadence': 'infer update cadence from rows before each training cutoff',
    'wide_models': 'existing ExtraTrees/RF/KNN parameters, training-only decay weights and causal regime overlay',
    'compact_models': {'ridge_alpha': 100.0, 'max_iter': 100, 'max_leaf_nodes': 15,
                       'min_samples_leaf': 30, 'learning_rate': .05, 'l2_regularization': 1.0},
    'seed': 42, 'selection_lookback_days': 365, 'selection_min_rows': 90,
    'selector': 'lowest MAE among earlier matured OOF predictions; no-change during warm-up; baseline wins exact ties',
    'uncertainty': '1000 moving calendar-block bootstrap samples, block = max(30,horizon); descriptive, not a promotion gate',
    'latest_period': 'last 365 calendar days of resolved targets',
    'selection_bias': 'Full-period winner is retrospective; candidate shortlist was informed by existing historical evaluations.',
    'input_limit': 'Reconstructed vendor data; historical publication vintages are not verified.',
    'production': 'chart-only research; never replaces issued forecasts or promotes a live model',
    'probabilities': 'point forecasts only; UP/FLAT/DOWN classes are not calibrated probabilities',
    'cache': 'code + protocol + numerical versions + past input fingerprints; target actuals are settled separately',
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
