-- ETH price forecast service — SQLite schema
-- One database file holds: every pipeline run, every (run × horizon) forecast,
-- backfilled actuals once the target date passes, and rolling accuracy
-- snapshots used by the website.

-- ---------------------------------------------------------------------------
-- forecast_runs: one row per eth_price_forecast.run_suite() invocation.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecast_runs (
    run_id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_timestamp_utc              TEXT    NOT NULL,      -- ISO-8601 UTC, when pipeline finished
    input_timestamp_utc            TEXT    NOT NULL,      -- data cutoff (latest_prediction_input_timestamp)
    reference_price                REAL    NOT NULL,      -- ETH USD close used as forecast anchor
    reference_price_source         TEXT,
    reference_price_timestamp_utc  TEXT,
    model_phase                    TEXT    NOT NULL,      -- e.g. "phase4b_production", "phase5_production"
    code_version                   TEXT,                  -- git sha or bytes-hash of eth_price_forecast.py
    eth_price_forecast_bytes       INTEGER,
    fast_mode                      INTEGER NOT NULL DEFAULT 0,
    notes                          TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_timestamp
    ON forecast_runs(run_timestamp_utc DESC);
CREATE UNIQUE INDEX IF NOT EXISTS idx_runs_input_phase
    ON forecast_runs(input_timestamp_utc, model_phase);

-- ---------------------------------------------------------------------------
-- forecasts: one row per (run_id, horizon_days).
-- Columns mirror build_latest_forecast_summary so we keep everything the UI
-- might want without re-deriving from CSVs.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS forecasts (
    forecast_id                         INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                              INTEGER NOT NULL
        REFERENCES forecast_runs(run_id) ON DELETE CASCADE,
    horizon_days                        INTEGER NOT NULL,
    forecast_target_timestamp_utc       TEXT    NOT NULL,

    -- Regression head
    regression_model                    TEXT,
    regression_selection_basis          TEXT,
    regression_predicted_return         REAL,
    regression_predicted_close          REAL,
    regression_lower_return_10          REAL,
    regression_lower_close_10           REAL,
    regression_upper_return_90          REAL,
    regression_upper_close_90           REAL,

    -- Classification head
    classification_model                TEXT,
    classification_selection_basis      TEXT,
    classification_predicted_direction  TEXT,       -- UP / DOWN / FLAT
    classification_probability_up       REAL,
    classification_probability_down     REAL,
    classification_confidence           REAL,
    classification_signal_threshold     REAL,

    -- Hybrid (ensemble of heads)
    hybrid_model                        TEXT,
    hybrid_predicted_direction          TEXT,
    hybrid_bias_direction               TEXT,
    hybrid_predicted_market_state       TEXT,
    hybrid_predicted_reversal_signal    TEXT,
    hybrid_score                        REAL,
    hybrid_confidence                   REAL,
    hybrid_signal_tier                  TEXT,       -- NO_SIGNAL / WATCH / CONFIRMED etc.
    hybrid_high_confidence_signal       INTEGER,
    hybrid_signal_reason                TEXT,
    hybrid_volatility_scale             REAL,
    hybrid_scenario_spread              REAL,
    forecast_decision_mode              TEXT,
    forecast_actionability              TEXT,
    forecast_point_price_reliable       INTEGER,
    forecast_center_return              REAL,
    forecast_uncertainty_return         REAL,
    forecast_lower_return               REAL,
    forecast_upper_return               REAL,

    -- Hybrid "active" scenario (includes macro tilt)
    active_predicted_direction          TEXT,
    active_confidence                   REAL,
    active_predicted_return             REAL,
    active_predicted_close              REAL,
    active_bear_predicted_return        REAL,
    active_bear_predicted_close         REAL,
    active_bull_predicted_return        REAL,
    active_bull_predicted_close         REAL,

    -- Regime + reversal
    regime_model                        TEXT,
    regime_predicted_state              TEXT,       -- UPTREND / SIDEWAYS / DOWNTREND
    regime_probability_uptrend          REAL,
    regime_probability_sideways         REAL,
    regime_probability_downtrend        REAL,
    reversal_model                      TEXT,
    reversal_predicted_signal           TEXT,
    reversal_probability_bottom         REAL,
    reversal_probability_top            REAL,

    -- Macro readiness + risk scores
    macro_readiness_status              TEXT,
    macro_risk_regime                   TEXT,
    macro_directional_bias              TEXT,
    macro_selection_tag                 TEXT,
    macro_risk_on_score                 REAL,
    macro_risk_off_score                REAL,
    macro_trend_bias_score              REAL,
    macro_volatility_stress             REAL,
    macro_used_feature_count            INTEGER,
    macro_missing_feature_count         INTEGER,

    UNIQUE (run_id, horizon_days)
);

CREATE INDEX IF NOT EXISTS idx_forecasts_horizon_target
    ON forecasts(horizon_days, forecast_target_timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_forecasts_run
    ON forecasts(run_id);

-- ---------------------------------------------------------------------------
-- actuals: realized ETH close + error metrics, backfilled once the target
-- date has passed. One row per forecast_id (populated lazily).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS actuals (
    forecast_id                INTEGER PRIMARY KEY
        REFERENCES forecasts(forecast_id) ON DELETE CASCADE,
    resolved_at_utc            TEXT    NOT NULL,          -- when this row was inserted
    actual_close               REAL    NOT NULL,
    actual_return              REAL    NOT NULL,          -- (actual_close - ref_price) / ref_price
    direction_actual           TEXT    NOT NULL,          -- UP / DOWN / FLAT (based on sign+threshold)
    direction_correct          INTEGER,                    -- 1 if classification pick matched actual
    active_direction_correct   INTEGER,                    -- 1 if active (hybrid+macro) pick matched
    return_absolute_error      REAL,                       -- |predicted_return - actual_return|
    price_absolute_error       REAL,                       -- |predicted_close - actual_close|
    brier_contribution         REAL                        -- (prob_up - (actual==UP))^2
);

-- ---------------------------------------------------------------------------
-- accuracy_snapshot: rolling windows recomputed whenever actuals are
-- backfilled. Fuels the site's "recent accuracy" dashboard.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS accuracy_snapshot (
    snapshot_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_utc         TEXT    NOT NULL,
    horizon_days         INTEGER NOT NULL,
    window_days          INTEGER NOT NULL,             -- 30 / 90 / 180 / all
    resolved_count       INTEGER NOT NULL,
    direction_accuracy   REAL,                          -- fraction 0..1
    brier_score          REAL,                          -- mean brier_contribution
    price_mape_percent   REAL,
    price_rmse           REAL,
    return_mae           REAL
);

CREATE INDEX IF NOT EXISTS idx_accuracy_horizon_window
    ON accuracy_snapshot(horizon_days, window_days, snapshot_utc DESC);

-- ---------------------------------------------------------------------------
-- backtest_runs: one row per frozen walk-forward CV run (freeze_phase*.py).
-- Unlike forecast_runs (which captures live-cron predictions), this table
-- archives the historical 3-year walk-forward evaluation for a given code
-- version. Every version bump (phase1 -> phase2 -> ...) gets a fresh row so
-- the UI can show "how did each version perform on the same 3 years".
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_runs (
    backtest_run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    frozen_utc            TEXT    NOT NULL,     -- freeze_*.py "frozen_at"
    model_phase           TEXT    NOT NULL,     -- "baseline", "phase5_production", ...
    mode                  TEXT,                 -- freeze-script mode string
    code_bytes            INTEGER,              -- size of eth_price_forecast.py at freeze time
    master_data_csv       TEXT,                 -- source dataset path
    cv_test_size          INTEGER,
    freeze_json_path      TEXT,                 -- original JSON for forensic drill-down
    freeze_json_sha1      TEXT,                 -- content hash -> idempotent ingest
    notes                 TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_backtest_runs_phase_hash
    ON backtest_runs(model_phase, freeze_json_sha1);
CREATE INDEX IF NOT EXISTS idx_backtest_runs_frozen
    ON backtest_runs(frozen_utc DESC);

-- ---------------------------------------------------------------------------
-- backtest_metrics: per-(run, horizon, head, model) leaderboard entry.
-- One row per leaderboard cell across both regression + classification heads.
-- Scales to ~20 rows per freeze JSON (2 horizons × 2 heads × ~5 models).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_metrics (
    backtest_run_id       INTEGER NOT NULL
        REFERENCES backtest_runs(backtest_run_id) ON DELETE CASCADE,
    horizon_days          INTEGER NOT NULL,
    head                  TEXT    NOT NULL,     -- "regression" | "classification"
    model                 TEXT    NOT NULL,

    -- Regression-side metrics (NULL on classification rows).
    return_mae            REAL,
    price_mae             REAL,
    price_rmse            REAL,
    price_mape_percent    REAL,

    -- Classification-side metrics (NULL on regression rows).
    accuracy              REAL,
    balanced_accuracy     REAL,
    precision_score       REAL,
    recall_score          REAL,
    f1                    REAL,
    roc_auc               REAL,
    brier_score           REAL,
    signal_threshold      REAL,

    -- Shared.
    directional_accuracy  REAL,                  -- regression-side direction hit-rate
    folds                 REAL,
    component_models      TEXT,                  -- "|" separated ensemble roster
    threshold_validation_scheme TEXT,
    validation_leakage_safe     INTEGER,

    PRIMARY KEY (backtest_run_id, horizon_days, head, model)
);

CREATE INDEX IF NOT EXISTS idx_backtest_metrics_phase_horizon
    ON backtest_metrics(backtest_run_id, horizon_days, head);

-- ---------------------------------------------------------------------------
-- backtest_predictions: per-date out-of-fold prediction for every
-- (run, horizon, head, model, date) combination. Populated by longrun_oof
-- freezes that store the full walk-forward trace (not just summary metrics).
--
-- One row = one model's honest OOF prediction for one target date in one fold.
-- Because splits are purely a function of (n_samples, n_splits, test_size,
-- gap, embargo), the fold_index is deterministic and lets us resume an
-- interrupted run without duplicating work.
--
-- Leakage guarantees (inherited from walk_forward_*):
--   1. Temporal split — test_date always > max(train_dates).
--   2. Purged gap — last `horizon` rows of train dropped so target labels
--      (close.shift(-horizon)) never bleed into test.
--   3. Embargo — additional buffer between train end and test start.
--   4. Fold-internal feature selection — features chosen using train rows
--      only (select_fold_features, min_feature_coverage).
--   5. Purged-temporal threshold (classification) — signal_threshold
--      picked on a held-out purged portion of OOF, not the evaluated set.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS backtest_predictions (
    backtest_run_id   INTEGER NOT NULL
        REFERENCES backtest_runs(backtest_run_id) ON DELETE CASCADE,
    horizon_days      INTEGER NOT NULL,
    head              TEXT    NOT NULL,     -- "regression" | "classification"
    model             TEXT    NOT NULL,
    prediction_date   TEXT    NOT NULL,     -- forecast origin (training cutoff)
    target_date       TEXT    NOT NULL,     -- prediction_date + horizon_days
    fold_index        INTEGER NOT NULL,     -- 0-based walk-forward fold

    -- Anchor + realised truth (populated for every row).
    reference_close   REAL,                  -- eth_close at prediction_date
    actual_close      REAL,                  -- eth_close at target_date
    actual_return     REAL,                  -- (actual_close - reference_close) / reference_close
    actual_label      INTEGER,               -- 1 if actual_return > 0 else 0

    -- Regression payload (NULL on classification rows).
    predicted_return  REAL,
    predicted_close   REAL,

    -- Classification payload (NULL on regression rows).
    probability_up    REAL,
    direction_score_up REAL,                 -- ranking/decision score; may differ from probability_up
    predicted_label   INTEGER,               -- 1/0 after signal_threshold

    PRIMARY KEY (backtest_run_id, horizon_days, head, model, prediction_date)
);

CREATE INDEX IF NOT EXISTS idx_backtest_predictions_lookup
    ON backtest_predictions(backtest_run_id, horizon_days, head, prediction_date);
CREATE INDEX IF NOT EXISTS idx_backtest_predictions_model_date
    ON backtest_predictions(model, prediction_date);

-- ---------------------------------------------------------------------------
-- schema_version: single-row table used by db.init_schema to detect upgrades.
-- v2 added backtest_runs + backtest_metrics tables (2026-04-21).
-- v3 added backtest_predictions table for per-date OOF (2026-04-21).
-- v4 added direction_score_up to preserve classification decisions (2026-08-25).
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER PRIMARY KEY,
    applied_utc  TEXT    NOT NULL
);
INSERT OR IGNORE INTO schema_version (version, applied_utc)
    VALUES (1, CURRENT_TIMESTAMP);
INSERT OR IGNORE INTO schema_version (version, applied_utc)
    VALUES (2, CURRENT_TIMESTAMP);
INSERT OR IGNORE INTO schema_version (version, applied_utc)
    VALUES (3, CURRENT_TIMESTAMP);
INSERT OR IGNORE INTO schema_version (version, applied_utc)
    VALUES (4, CURRENT_TIMESTAMP);
