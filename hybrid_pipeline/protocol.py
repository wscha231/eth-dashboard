"""Versioned development protocol; each replay uses only its recorded past."""
import hashlib
import json

BASE_MODELS = ['cat_short', 'cat_long', 'transformer_short', 'transformer_long']
PROTOCOL = {
    'version': 'cat_patch_transformer_v2', 'horizons': [7, 30], 'seed': 1729,
    'features': 'closed ETH/BTC OHLCV only, plus verified flow lagged one extra calendar day; no macro/on-chain publication vintages',
    'source_limit': 'Reconstructed exchange history, not a claim of original publication-vintage verification.',
    'target': 'direct horizon log return divided by (origin trailing 30-day volatility * sqrt(horizon))',
    'refit': 'calendar month start; daily saved-model inference',
    'minimum_training_rows': 500, 'sequence_warmup_days': 64,
    'embargo_days': {'7': 3, '30': 15},
    'train_years': {'7': {'short': 2, 'long': 3}, '30': {'short': 3, 'long': 5}},
    'purge': 'origin + horizon < fit cutoff - embargo, including inner early-stopping split',
    'inner_validation_days': 120, 'minimum_inner_training_rows': 200,
    'cat': {
        'short': {'depth': 3, 'iterations': 240, 'learning_rate': .04, 'l2_leaf_reg': 20},
        'long': {'depth': 5, 'iterations': 360, 'learning_rate': .035, 'l2_leaf_reg': 40},
    },
    'transformer': {
        'short': {'lookback': 32, 'patch': 4, 'width': 16, 'heads': 2, 'layers': 1, 'epochs': 16},
        'long': {'lookback': 64, 'patch': 4, 'width': 24, 'heads': 4, 'layers': 1, 'epochs': 24},
    },
    'neural_training': {'batch_size': 128, 'learning_rate': .001, 'weight_decay': .02,
                        'dropout': .1, 'patience': 4, 'gradient_clip': 1.0},
    'point_guard': {'quantiles': [.05, .95], 'basis': 'matured outer-training log returns, separately by horizon/window/month; preserve zero inside bounds',
                    'reason': 'v1 development replay exposed extreme volatility-rescaling extrapolation in March 2020; v2 is a disclosed development revision, not a fresh holdout'},
    'objective': 'return MAE on prior matured out-of-sample origins; baseline reported separately',
    'selection_days': {'7': 365, '30': 730}, 'selection_min_rows': 180,
    'blend_weights_cat': [0.0, .25, .5, .75, 1.0], 'amplitudes': [.5, 1.0],
    'selection': 'select CatBoost configuration, Transformer configuration, blend and amplitude using earlier matured OOF results only; equal short-model blend during warmup',
    'safe_selection': 'also compare zero-return reference with the hybrid on prior OOF outcomes',
    'uncertainty': 'empirical normalized OOF residuals from before the monthly cutoff; 10/90 quantiles, no guaranteed conformal coverage',
    'event_threshold': {'7': {'multiplier': .4, 'floor': .008, 'cap': .045},
                        '30': {'multiplier': .65, 'floor': .055, 'cap': .16}},
    'probability': 'unconditional DOWN/FLAT/UP from past residuals, additive-one smoothing',
    'bootstrap': '1000 moving calendar-block samples, block=max(30,horizon); descriptive after historical research',
    'live_gate': 'new family issues as research until prospective evidence is sufficient; no historical result alone asserts predictive edge',
    'retirement': 'legacy RF/KNN/HGB/Ridge candidates leave the active model roster; issued records and benchmarks remain auditable',
    'research_limit': 'Candidate design used previously explored history. Chronological replay is not an untouched prospective holdout.',
}
PROTOCOL_HASH = hashlib.sha256(json.dumps(PROTOCOL, sort_keys=True).encode()).hexdigest()
