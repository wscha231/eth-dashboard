from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import subprocess
import sys
import traceback
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from catboost import CatBoostClassifier, CatBoostRegressor
except ImportError:  # pragma: no cover - optional dependency path
    CatBoostClassifier = None
    CatBoostRegressor = None


try:
    import yfinance as yf
except ImportError as exc:  # pragma: no cover - dependency error path
    raise SystemExit(
        "Missing dependency: install `yfinance pandas scikit-learn numpy` and retry."
    ) from exc

try:
    import shap
except ImportError:  # pragma: no cover - optional dependency path
    shap = None


DEFAULT_TICKERS = {
    "eth": "ETH-USD",
    "btc": "BTC-USD",
    "sol": "SOL-USD",
    "bnb": "BNB-USD",
    "spy": "SPY",
    "qqq": "QQQ",
    "gold": "GC=F",
    "oil": "CL=F",
    "dxy": "DX-Y.NYB",
    "vix": "^VIX",
    "tnx": "^TNX",
}
DEFAULT_HORIZONS = [7, 30]
REQUIRED_TICKERS = {"eth", "btc"}
CACHE_UPDATE_LOOKBACK_ROWS = 400
DEFAULT_MASTER_DATA_CSV = "eth_master_daily.csv"
DEFAULT_CLASSIFICATION_SIGNAL_THRESHOLD = 0.55
DEFAULT_CLASSIFICATION_THRESHOLD_GRID = [0.50, 0.55, 0.60, 0.65]
DEFAULT_CLASSIFICATION_MIN_EDGE = 0.03
DEFAULT_REGRESSION_THRESHOLD_GRID = [0.0, 0.005, 0.01, 0.02]
DEFAULT_FEATURE_MIN_COVERAGE = 0.03
DEFAULT_MIN_STANDARD_FEATURE_ROWS = 60
DEFAULT_MIN_HISTORY_FEATURE_ROWS = 12
DEFAULT_FEATURE_CORRELATION_THRESHOLD = 0.985
DEFAULT_MAX_FEATURES_SHORT = 280
DEFAULT_MAX_FEATURES_LONG = 360
DEFAULT_STATE_CONFIDENCE_GRID = [0.40, 0.45, 0.50, 0.55, 0.60]
DEFAULT_STATE_MIN_TOTAL_EVENTS_SHORT = 6
DEFAULT_STATE_MIN_TOTAL_EVENTS_LONG = 5
DEFAULT_MIN_VENDOR_HISTORY_DAYS = 60
DEFAULT_STALE_VENDOR_MAX_AGE_DAYS = 30
DEFAULT_CALIBRATION_MIN_ROWS = 180
DEFAULT_RECENT_HOLDOUT_ROWS = 180
DEFAULT_THRESHOLD_VALIDATION_MIN_TRAIN_ROWS = 80
DEFAULT_THRESHOLD_VALIDATION_MIN_EVAL_ROWS = 60
DEFAULT_THRESHOLD_VALIDATION_EVAL_FRACTION = 0.25
DEFAULT_REFERENCE_FAST_INFO_MAX_GAP_PCT = 0.05
DEFAULT_SHORT_TRAINING_WINDOW_YEARS = 3.0
DEFAULT_LONG_TRAINING_WINDOW_YEARS = 5.0
DEFAULT_SHORT_SAMPLE_WEIGHT_HALF_LIFE_YEARS = 1.0
DEFAULT_LONG_SAMPLE_WEIGHT_HALF_LIFE_YEARS = 1.5
DEFAULT_SAMPLE_WEIGHT_MIN = 0.05
GENERIC_VENDOR_PREFIXES = (
    "external_",
    "artemis_",
    "cg_",
    "binance_",
    "deribit_",
    "etherscan_",
    "fred_",
    "sentiment_",
    # Phase 6 crypto-native on-chain depth. DefiLlama is free + no-key
    # (stablecoin supply, chain TVL, DEX volume); Glassnode requires a paid
    # tier for ETH endpoints and may be absent (skipped gracefully by the
    # collector). Both emit daily columns that slot into the standard vendor
    # transform pipeline (__asof / __age_days / _diff_7 / _pct_30 / _zscore_30).
    # No named regime composite is added yet — gate that behind a LOO pass
    # like we did for the Phase 5 macro re-add.
    "defillama_",
    "glassnode_",
)
CRYPTO_24_7_PREFIXES = ("eth_", "btc_", "sol_", "bnb_")
TRADITIONAL_MARKET_PREFIXES = ("spy_", "qqq_", "gold_", "oil_", "dxy_", "vix_", "tnx_")
TRADITIONAL_MARKET_FFILL_MAX_DAYS = 5
VENDOR_FFILL_MAX_DAYS = 45
OPTIONAL_MODEL_STATUS: dict[str, Any] = {
    "catboost_available": bool(CatBoostRegressor is not None and CatBoostClassifier is not None),
    "catboost_installed_now": False,
    "catboost_error": "",
}
RUNTIME_OPTIONS: dict[str, Any] = {
    "fast_mode": False,
}
REGIME_STATE_LABELS = {0: "DOWNTREND", 1: "SIDEWAYS", 2: "UPTREND"}
REVERSAL_STATE_LABELS = {0: "TOP_REVERSAL", 1: "NONE", 2: "BOTTOM_REVERSAL"}
TRIMMED_REGRESSION_ENSEMBLE_MODEL = "trimmed_equal_weight_regression"
TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL = "trimmed_equal_weight_direction"
MACRO_CONTEXT_FEATURE_COLUMNS = [
    "macro_tightening_20",
    "fred_bbb_oas_z_90",
    "risk_aversion_spread_5",
    "dxy_return_20",
    "tnx_return_20",
    "oil_vol_20",
    "btc_drawdown_60",
    "eth_drawdown_30",
    "equity_risk_on_20",
    "eth_vs_btc_strength_14",
    "fred_net_liquidity_z_90",
    "sentiment_fear_greed_centered",
    "eth_ma_14_ratio",
    "eth_ma_30_ratio",
    "eth_vwap_20_ratio",
    "artemis_stablecoin_total_supply_z_30",
    "artemis_non_sybil_users_z_90",
    "eth_return_7",
    "eth_return_14",
    "eth_return_60",
    "eth_return_90",
    "eth_return_180",
    "eth_vol_7",
    "eth_vol_14",
    "eth_vol_30",
    "eth_vol_90",
    "eth_vol_30_180_ratio",
    "eth_tail_event_pressure",
    "eth_vol_squeeze_20_90",
    "eth_volume_impulse_3_30",
    "eth_range_expansion_z_30",
    "eth_ma_180_ratio",
    "eth_drawdown_180",
    "eth_cycle_position_365_centered",
    "eth_vs_btc_strength_90",
    "btc_drawdown_180",
    "eth_range_pct",
    "eth_close_open_pct",
]


def _make_median_imputer() -> SimpleImputer:
    """Keep all-empty columns from emitting repeated sklearn warnings."""
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:  # pragma: no cover - compatibility with older sklearn
        warnings.filterwarnings(
            "ignore",
            message="Skipping features without any observed values:.*",
            category=UserWarning,
        )
        return SimpleImputer(strategy="median")


def set_runtime_options(*, fast_mode: bool | None = None) -> None:
    if fast_mode is not None:
        RUNTIME_OPTIONS["fast_mode"] = bool(fast_mode)


def runtime_fast_mode_enabled() -> bool:
    return bool(RUNTIME_OPTIONS.get("fast_mode"))

PRICE_FIELD_MAP = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "adj close": "close",
    "adj_close": "close",
    "volume": "volume",
}


@dataclass
class ForecastResult:
    model_name: str
    selection_basis: str
    prediction_timestamp: str
    horizon_steps: int
    model_input_close: float
    last_close: float
    reference_price_source: str
    reference_price_timestamp: str
    predicted_return: float
    predicted_close: float
    lower_return_10: float
    lower_close_10: float
    upper_return_90: float
    upper_close_90: float


@dataclass
class DirectionForecastResult:
    model_name: str
    selection_basis: str
    prediction_timestamp: str
    horizon_steps: int
    model_input_close: float
    last_close: float
    reference_price_source: str
    reference_price_timestamp: str
    predicted_direction: str
    signal_threshold: float
    probability_up: float
    probability_down: float
    confidence: float


@dataclass
class RegimeForecastResult:
    model_name: str
    selection_basis: str
    prediction_timestamp: str
    horizon_steps: int
    model_input_close: float
    last_close: float
    reference_price_source: str
    reference_price_timestamp: str
    predicted_regime: str
    probability_downtrend: float
    probability_sideways: float
    probability_uptrend: float
    confidence: float


@dataclass
class ReversalForecastResult:
    model_name: str
    selection_basis: str
    prediction_timestamp: str
    horizon_steps: int
    model_input_close: float
    last_close: float
    reference_price_source: str
    reference_price_timestamp: str
    predicted_reversal: str
    probability_top_reversal: float
    probability_no_reversal: float
    probability_bottom_reversal: float
    confidence: float


@dataclass
class HybridForecastResult:
    model_name: str
    selection_basis: str
    prediction_timestamp: str
    horizon_steps: int
    model_input_close: float
    last_close: float
    reference_price_source: str
    reference_price_timestamp: str
    predicted_market_state: str
    predicted_reversal_signal: str
    predicted_direction: str
    trend_score: float
    reversal_score: float
    direction_score: float
    forecast_score: float
    fundamental_score: float
    hybrid_score: float
    confidence: float
    bias_direction: str
    signal_tier: str
    high_confidence_signal: bool
    signal_reason: str
    volatility_scale: float
    scenario_spread: float
    macro_readiness_status: str
    macro_risk_regime: str
    macro_directional_bias: str
    macro_selection_tag: str
    macro_risk_on_score: float
    macro_risk_off_score: float
    macro_trend_bias_score: float
    macro_volatility_stress: float
    macro_used_feature_count: int
    macro_missing_feature_count: int
    macro_used_features: str
    macro_missing_features: str
    active_predicted_direction: str
    active_confidence: float
    active_predicted_return: float
    active_predicted_close: float
    active_bear_predicted_return: float
    active_bear_predicted_close: float
    active_bull_predicted_return: float
    active_bull_predicted_close: float
    adjusted_predicted_return: float
    adjusted_predicted_close: float
    bear_predicted_return: float
    bear_predicted_close: float
    bull_predicted_return: float
    bull_predicted_close: float


@dataclass
class PriceReference:
    price: float
    timestamp: str
    source: str


@dataclass
class ExplainabilityArtifacts:
    model_name: str
    native_feature_importance: pd.DataFrame
    permutation_importance: pd.DataFrame
    shap_global_importance: pd.DataFrame
    shap_latest_contributions: pd.DataFrame
    notes: pd.DataFrame


@dataclass
class HorizonArtifacts:
    horizon_steps: int
    regression_forecast: ForecastResult
    classification_forecast: DirectionForecastResult
    regime_forecast: RegimeForecastResult
    reversal_forecast: ReversalForecastResult
    hybrid_forecast: HybridForecastResult
    regression_leaderboard: pd.DataFrame
    classification_leaderboard: pd.DataFrame
    regime_leaderboard: pd.DataFrame
    reversal_leaderboard: pd.DataFrame
    regression_backtest: pd.DataFrame
    classification_backtest: pd.DataFrame
    regime_backtest: pd.DataFrame
    reversal_backtest: pd.DataFrame
    benchmark_backtest: pd.DataFrame
    regression_oof_predictions: pd.DataFrame
    classification_oof_predictions: pd.DataFrame
    regime_oof_predictions: pd.DataFrame
    reversal_oof_predictions: pd.DataFrame
    regression_equity_curves: pd.DataFrame
    classification_equity_curves: pd.DataFrame
    regime_equity_curves: pd.DataFrame
    reversal_equity_curves: pd.DataFrame
    benchmark_equity_curves: pd.DataFrame
    data_window_summary: pd.DataFrame
    feature_coverage_summary: pd.DataFrame
    recent_holdout_report: pd.DataFrame
    feature_snapshot: pd.DataFrame
    regression_explainability: ExplainabilityArtifacts
    classification_explainability: ExplainabilityArtifacts


@dataclass
class PipelineArtifacts:
    raw_data: pd.DataFrame
    horizons: dict[int, HorizonArtifacts]
    failed_horizons: list[dict[str, Any]] = field(default_factory=list)


class CalibratedProbabilityModel(BaseEstimator, ClassifierMixin):
    _estimator_type = "classifier"

    def __init__(
        self,
        base_estimator: Any,
        calibrator: Any | None = None,
        calibration_rows: int = 0,
        calibration_method: str = "none",
    ) -> None:
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.calibration_rows = int(calibration_rows)
        self.calibration_method = calibration_method
        self.classes_ = np.array([0, 1], dtype=int)

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> "CalibratedProbabilityModel":
        y_series = pd.Series(y).astype(int)
        self.base_estimator.fit(X, y_series)
        self.classes_ = np.unique(y_series)

        if self.calibrator is None or y_series.nunique() < 2:
            return self

        raw_probabilities = np.clip(
            self.base_estimator.predict_proba(X)[:, 1],
            1e-6,
            1.0 - 1e-6,
        )
        recalibrator, calibration_method = fit_probability_calibrator(raw_probabilities, y_series)
        if recalibrator is None:
            self.calibrator = None
            self.calibration_rows = 0
            self.calibration_method = "none"
            return self

        self.calibrator = recalibrator
        self.calibration_rows = int(len(y_series))
        self.calibration_method = calibration_method
        return self

    def __sklearn_tags__(self) -> Any:
        tags = super().__sklearn_tags__()
        try:
            tags.estimator_type = "classifier"
        except Exception:
            pass
        return tags

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probabilities = np.clip(self.base_estimator.predict_proba(X)[:, 1], 1e-6, 1.0 - 1e-6)
        if self.calibrator is not None:
            if hasattr(self.calibrator, "predict_proba"):
                probabilities = self.calibrator.predict_proba(probabilities.reshape(-1, 1))[:, 1]
            else:
                probabilities = self.calibrator.predict(probabilities)
        probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - probabilities, probabilities])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def log_progress(enabled: bool, message: str) -> None:
    if enabled:
        print(message)


def fit_probability_calibrator(
    raw_probabilities: pd.Series | np.ndarray,
    actual_labels: pd.Series | np.ndarray,
    sample_weight: pd.Series | np.ndarray | list[float] | None = None,
) -> tuple[Any | None, str]:
    probabilities = np.clip(np.asarray(raw_probabilities, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
    labels = pd.Series(actual_labels).astype(int).to_numpy(dtype=int)
    valid_mask = np.isfinite(probabilities) & np.isfinite(labels)
    if valid_mask.sum() < 25:
        return None, "none"

    probabilities = probabilities[valid_mask]
    labels = labels[valid_mask]
    weight_values = None
    if sample_weight is not None:
        if isinstance(sample_weight, pd.Series):
            raw_weight_values = pd.to_numeric(sample_weight, errors="coerce").to_numpy(dtype=float)
        else:
            raw_weight_values = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(raw_weight_values) == len(valid_mask):
            weight_values = np.nan_to_num(raw_weight_values[valid_mask], nan=1.0, posinf=1.0, neginf=1.0)
            weight_values = np.clip(weight_values, 1e-6, None)

    if len(np.unique(labels)) < 2:
        return None, "none"

    unique_probability_count = len(np.unique(np.round(probabilities, 4)))
    if len(probabilities) >= 80 and unique_probability_count >= 10:
        isotonic = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        try:
            isotonic.fit(probabilities, labels, sample_weight=weight_values)
            return isotonic, "isotonic_last_window"
        except Exception:
            pass

    sigmoid = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    try:
        fit_kwargs: dict[str, Any] = {}
        if weight_values is not None:
            fit_kwargs["sample_weight"] = weight_values
        sigmoid.fit(probabilities.reshape(-1, 1), labels, **fit_kwargs)
        return sigmoid, "sigmoid_last_window"
    except Exception:
        return None, "none"


def bootstrap_optional_model_dependencies(
    auto_install_catboost: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    global CatBoostClassifier, CatBoostRegressor, OPTIONAL_MODEL_STATUS

    status = {
        "catboost_available": bool(CatBoostRegressor is not None and CatBoostClassifier is not None),
        "catboost_installed_now": False,
        "catboost_error": "",
    }
    if status["catboost_available"] or not auto_install_catboost:
        OPTIONAL_MODEL_STATUS = status
        return status

    try:
        log_progress(verbose, "Optional dependency missing: catboost. Attempting install...")
        install_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "catboost"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if install_result.returncode == 0:
            module = importlib.import_module("catboost")
            CatBoostClassifier = getattr(module, "CatBoostClassifier", None)
            CatBoostRegressor = getattr(module, "CatBoostRegressor", None)
            status["catboost_available"] = bool(CatBoostRegressor is not None and CatBoostClassifier is not None)
            status["catboost_installed_now"] = status["catboost_available"]
            if status["catboost_available"]:
                log_progress(verbose, "Optional dependency installed: catboost is active.")
            else:
                status["catboost_error"] = "catboost import failed after install"
        else:
            status["catboost_error"] = f"pip exited with code {install_result.returncode}"
    except Exception as exc:  # pragma: no cover - environment-specific install failure path
        status["catboost_error"] = str(exc)

    OPTIONAL_MODEL_STATUS = status
    return status


def periods_per_year(interval: str) -> int:
    mapping = {
        "1m": 60 * 24 * 365,
        "2m": 30 * 24 * 365,
        "5m": 12 * 24 * 365,
        "15m": 4 * 24 * 365,
        "30m": 2 * 24 * 365,
        "60m": 24 * 365,
        "90m": 16 * 365,
        "1h": 24 * 365,
        "1d": 365,
        "5d": 73,
        "1wk": 52,
        "1mo": 12,
        "3mo": 4,
    }
    return mapping.get(interval, 365)


def training_window_years_for_horizon(horizon: int) -> float:
    return DEFAULT_SHORT_TRAINING_WINDOW_YEARS if horizon <= 7 else DEFAULT_LONG_TRAINING_WINDOW_YEARS


def sample_weight_half_life_years_for_horizon(horizon: int) -> float:
    return (
        DEFAULT_SHORT_SAMPLE_WEIGHT_HALF_LIFE_YEARS
        if horizon <= 7
        else DEFAULT_LONG_SAMPLE_WEIGHT_HALF_LIFE_YEARS
    )


def trim_training_window(
    dataset: pd.DataFrame,
    interval: str,
    horizon: int,
) -> pd.DataFrame:
    if dataset.empty:
        return dataset.copy()

    window_years = training_window_years_for_horizon(horizon)
    latest_timestamp = pd.Timestamp(dataset.index.max())
    cutoff_timestamp = latest_timestamp - pd.Timedelta(days=int(round(window_years * 365.25)))
    trimmed = dataset.loc[dataset.index >= cutoff_timestamp].copy()

    minimum_rows = max(240, horizon * 12, periods_per_year(interval))
    if len(trimmed) < minimum_rows:
        return dataset.copy()
    return trimmed


def align_sample_weight(
    sample_weight: pd.Series | np.ndarray | list[float] | None,
    index: pd.Index,
) -> pd.Series | None:
    if sample_weight is None:
        return None
    if isinstance(sample_weight, pd.Series):
        aligned = pd.to_numeric(sample_weight.reindex(index), errors="coerce")
    else:
        values = np.asarray(sample_weight, dtype=float).reshape(-1)
        if len(values) != len(index):
            return None
        aligned = pd.Series(values, index=index, dtype=float)
    aligned = aligned.fillna(1.0).clip(lower=1e-6)
    return aligned.astype(float)


def build_time_decay_sample_weights(
    index: pd.Index,
    interval: str,
    horizon: int,
) -> pd.Series:
    if len(index) == 0:
        return pd.Series(dtype=float, index=index)

    half_life_years = sample_weight_half_life_years_for_horizon(horizon)
    if half_life_years <= 0.0:
        return pd.Series(1.0, index=index, dtype=float)

    try:
        timestamps = pd.to_datetime(index)
        if getattr(timestamps, "tz", None) is not None:
            timestamps = timestamps.tz_convert(None)
        latest_timestamp = pd.Timestamp(timestamps.max())
        age_years = (latest_timestamp - timestamps) / pd.Timedelta(days=365.25)
        age_years = np.asarray(age_years, dtype=float)
    except Exception:
        age_steps = np.arange(len(index) - 1, -1, -1, dtype=float)
        age_years = age_steps / max(periods_per_year(interval), 1)

    weights = np.power(0.5, age_years / half_life_years)
    weights = np.clip(weights, DEFAULT_SAMPLE_WEIGHT_MIN, 1.0)
    return pd.Series(weights, index=index, dtype=float)


def _fit_signature_accepts_sample_weight(estimator: Any) -> bool:
    try:
        return "sample_weight" in inspect.signature(estimator.fit).parameters
    except (TypeError, ValueError):
        return False


def fit_model_with_optional_sample_weight(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    sample_weight: pd.Series | np.ndarray | list[float] | None = None,
) -> Any:
    aligned_sample_weight = align_sample_weight(sample_weight, X.index)
    if aligned_sample_weight is None:
        model.fit(X, y)
        return model

    if isinstance(model, Pipeline) and model.steps:
        step_name, estimator = model.steps[-1]
        if _fit_signature_accepts_sample_weight(estimator):
            try:
                model.fit(X, y, **{f"{step_name}__sample_weight": aligned_sample_weight.to_numpy(dtype=float)})
                return model
            except Exception:
                pass
    elif _fit_signature_accepts_sample_weight(model):
        try:
            model.fit(X, y, sample_weight=aligned_sample_weight.to_numpy(dtype=float))
            return model
        except Exception:
            pass

    model.fit(X, y)
    return model


def compute_max_drawdown(equity_curve: pd.Series) -> float:
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def flatten_ohlcv_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if not isinstance(frame.columns, pd.MultiIndex):
        renamed = {}
        for column in frame.columns:
            normalized = str(column).strip().lower().replace(" ", "_")
            normalized = PRICE_FIELD_MAP.get(normalized.replace("_", " "), normalized)
            renamed[column] = f"{prefix}_{normalized}"
        return frame.rename(columns=renamed)

    flattened = frame.copy()
    normalized_names: list[str] = []
    for column in flattened.columns.to_flat_index():
        parts = [str(part).strip().lower() for part in column if str(part).strip()]
        price_name = None
        for part in parts:
            if part in PRICE_FIELD_MAP:
                price_name = PRICE_FIELD_MAP[part]
                break
        if price_name is None:
            joined = "_".join(parts).replace(" ", "_")
            normalized_names.append(f"{prefix}_{joined}")
        else:
            normalized_names.append(f"{prefix}_{price_name}")
    flattened.columns = normalized_names
    return flattened


def normalize_index(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized.index = pd.to_datetime(normalized.index)
    if getattr(normalized.index, "tz", None) is not None:
        normalized.index = normalized.index.tz_convert(None)
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    normalized = normalized.sort_index()
    return normalized


def download_symbol_history(
    symbol: str,
    alias: str,
    interval: str,
    period: str | None = None,
    start: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    base_kwargs: dict[str, Any] = {
        "interval": interval,
        "auto_adjust": True,
        "progress": False,
        "threads": False,
    }
    if start is not None:
        base_kwargs["start"] = pd.Timestamp(start).to_pydatetime()
    elif period is not None:
        base_kwargs["period"] = period
    else:
        raise ValueError("Either period or start must be provided to download_symbol_history.")
    try:
        history = yf.download(
            symbol,
            group_by="column",
            multi_level_index=False,
            repair=True,
            **base_kwargs,
        )
    except TypeError:
        history = yf.download(
            symbol,
            group_by="column",
            **base_kwargs,
        )
    if history.empty:
        raise ValueError(f"Could not download data for {symbol}.")

    history = normalize_index(flatten_ohlcv_columns(history, alias))
    required_price_columns = {
        f"{alias}_open",
        f"{alias}_high",
        f"{alias}_low",
        f"{alias}_close",
    }
    missing = required_price_columns - set(history.columns)
    if missing:
        available = [str(column) for column in history.columns.tolist()]
        raise ValueError(
            f"Missing required columns for {symbol}: {sorted(missing)}. Available columns: {available}"
        )
    volume_col = f"{alias}_volume"
    if volume_col not in history.columns:
        history[volume_col] = 0.0
    return history[list(sorted(required_price_columns | {volume_col}))].copy()


def load_market_data_csv(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(source, index_col=0, parse_dates=True)
    return normalize_index(frame)


def save_market_data_csv(frame: pd.DataFrame, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalize_index(frame).to_csv(destination)


def apply_point_in_time_imputation(merged: pd.DataFrame) -> pd.DataFrame:
    """Column-aware bounded forward-fill.

    Replaces the old blanket ``merged.ffill()``. Imputation respects the
    release cadence of each data source so the final frame never silently
    substitutes a year-old vendor observation as if it were today's value:

    - 24/7 crypto OHLCV (eth_/btc_/sol_/bnb_): no ffill; the market never
      pauses, so any NaN is a real missing observation and should propagate to
      per-fold imputation in the sklearn Pipeline.
    - Traditional markets (spy_/qqq_/gold_/oil_/dxy_/vix_/tnx_): ffill up to
      TRADITIONAL_MARKET_FFILL_MAX_DAYS trading days to cover weekends and
      long holidays, but not open-ended gaps.
    - Vendor macro series (GENERIC_VENDOR_PREFIXES): ffill up to
      VENDOR_FFILL_MAX_DAYS calendar days. This matches monthly release
      cadences (CPI, PCE, NFP) without pretending a quarter-old value is still
      current.

    Remaining NaNs survive into feature engineering; the per-fold SimpleImputer
    inside each model Pipeline is responsible for the final fill.
    """
    result = merged.copy()
    for column in result.columns:
        if column.startswith(CRYPTO_24_7_PREFIXES):
            continue
        if column.startswith(TRADITIONAL_MARKET_PREFIXES):
            result[column] = result[column].ffill(limit=TRADITIONAL_MARKET_FFILL_MAX_DAYS)
        elif column.startswith(GENERIC_VENDOR_PREFIXES):
            result[column] = result[column].ffill(limit=VENDOR_FFILL_MAX_DAYS)
    return result


def finalize_market_data(merged: pd.DataFrame) -> pd.DataFrame:
    merged = normalize_index(merged)
    eth_close = merged["eth_close"].dropna()
    if eth_close.empty:
        raise ValueError("ETH close prices are missing after download.")

    merged = merged.reindex(eth_close.index)
    merged = apply_point_in_time_imputation(merged)
    return merged


def load_prediction_history_csv(path: str | Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    source = Path(path)
    if not source.exists():
        return pd.DataFrame()
    frame = pd.read_csv(
        source,
        parse_dates=["run_timestamp", "forecast_input_timestamp", "forecast_target_timestamp"],
    )
    if "horizon_steps" in frame.columns:
        frame["horizon_steps"] = frame["horizon_steps"].astype(int)
    return frame.sort_values(["forecast_input_timestamp", "horizon_steps", "run_timestamp"]).reset_index(drop=True)


def frame_numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype=float)


def frame_text_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column in frame.columns:
        return frame[column].astype(str)
    return pd.Series("", index=frame.index, dtype="object")


def realized_direction_band_for_horizon(horizon: int | float | None) -> float:
    horizon_steps = int(horizon) if horizon is not None and pd.notna(horizon) else 7
    if horizon_steps <= 7:
        return 0.006
    if horizon_steps <= 30:
        return 0.018
    return 0.030


def infer_direction_from_return_value(
    predicted_return: Any,
    horizon: int | float | None,
) -> str:
    value = pd.to_numeric(pd.Series([predicted_return]), errors="coerce").iloc[0]
    if pd.isna(value):
        return ""
    flat_band = realized_direction_band_for_horizon(horizon)
    if float(value) >= flat_band:
        return "UP"
    if float(value) <= -flat_band:
        return "DOWN"
    return "FLAT"


def compute_directional_trade_return(predicted_direction: Any, actual_return: Any) -> float:
    actual_value = pd.to_numeric(pd.Series([actual_return]), errors="coerce").iloc[0]
    if pd.isna(actual_value):
        return float("nan")
    direction = str(predicted_direction).strip().upper()
    if direction == "UP":
        return float(actual_value)
    if direction == "DOWN":
        return float(-actual_value)
    if direction == "FLAT":
        return 0.0
    return float("nan")


def compute_compound_return(returns: pd.Series | np.ndarray | list[float]) -> float:
    series = pd.to_numeric(pd.Series(returns), errors="coerce").dropna()
    if series.empty:
        return float("nan")
    clipped = series.clip(lower=-0.999999)
    return float(np.prod(1.0 + clipped.to_numpy(dtype=float)) - 1.0)


def compute_weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid_mask = numeric_values.notna() & numeric_weights.notna() & (numeric_weights > 0.0)
    if not bool(valid_mask.any()):
        return float("nan")
    return float(np.average(numeric_values.loc[valid_mask], weights=numeric_weights.loc[valid_mask]))


def compute_weighted_rmse(errors: pd.Series, weights: pd.Series) -> float:
    numeric_errors = pd.to_numeric(errors, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce")
    valid_mask = numeric_errors.notna() & numeric_weights.notna() & (numeric_weights > 0.0)
    if not bool(valid_mask.any()):
        return float("nan")
    weighted_mse = np.average(np.square(numeric_errors.loc[valid_mask]), weights=numeric_weights.loc[valid_mask])
    return float(np.sqrt(weighted_mse))


def refresh_prediction_history_with_realizations(
    prediction_history: pd.DataFrame | None,
    market_data: pd.DataFrame | None,
) -> pd.DataFrame:
    if prediction_history is None or prediction_history.empty:
        return pd.DataFrame() if prediction_history is None else prediction_history.copy()
    if market_data is None or market_data.empty or "eth_close" not in market_data.columns:
        return prediction_history.copy()

    history = prediction_history.copy()
    history["forecast_input_timestamp"] = pd.to_datetime(
        history["forecast_input_timestamp"] if "forecast_input_timestamp" in history.columns else pd.Series(pd.NaT, index=history.index),
        errors="coerce",
    )
    history["forecast_target_timestamp"] = pd.to_datetime(
        history["forecast_target_timestamp"] if "forecast_target_timestamp" in history.columns else pd.Series(pd.NaT, index=history.index),
        errors="coerce",
    )
    if "run_timestamp" in history.columns:
        history["run_timestamp"] = pd.to_datetime(history["run_timestamp"], errors="coerce")
    if "horizon_steps" in history.columns:
        history["horizon_steps"] = pd.to_numeric(history["horizon_steps"], errors="coerce").astype("Int64")

    normalized_market = normalize_index(market_data)
    actual_close_series = pd.to_numeric(normalized_market["eth_close"], errors="coerce")
    latest_market_timestamp = actual_close_series.dropna().index.max() if not actual_close_series.dropna().empty else pd.NaT

    target_lookup = history["forecast_target_timestamp"].dt.normalize() if "forecast_target_timestamp" in history.columns else pd.Series(pd.NaT, index=history.index)
    input_lookup = history["forecast_input_timestamp"].dt.normalize() if "forecast_input_timestamp" in history.columns else pd.Series(pd.NaT, index=history.index)

    target_close = actual_close_series.reindex(pd.Index(target_lookup)).to_numpy(dtype=float)
    input_close = actual_close_series.reindex(pd.Index(input_lookup)).to_numpy(dtype=float)
    reference_price = frame_numeric_series(history, "reference_price")
    model_input_close = frame_numeric_series(history, "model_input_close")
    realized_reference_close = reference_price.combine_first(model_input_close)
    realized_reference_close = realized_reference_close.combine_first(pd.Series(input_close, index=history.index, dtype=float))

    matured_mask = (
        history["forecast_target_timestamp"].notna()
        & pd.Series(target_close, index=history.index, dtype=float).notna()
        & realized_reference_close.notna()
        & (realized_reference_close > 0.0)
    )
    if pd.notna(latest_market_timestamp):
        matured_mask = matured_mask & (history["forecast_target_timestamp"] <= pd.Timestamp(latest_market_timestamp))

    actual_return = pd.Series(np.nan, index=history.index, dtype=float)
    actual_return.loc[matured_mask] = (
        pd.Series(target_close, index=history.index, dtype=float).loc[matured_mask]
        / realized_reference_close.loc[matured_mask]
        - 1.0
    )
    history["realized_is_matured"] = matured_mask.astype(bool)
    history["realized_latest_market_timestamp"] = (
        format_timestamp(latest_market_timestamp) if pd.notna(latest_market_timestamp) else ""
    )
    history["realized_actual_close"] = pd.Series(target_close, index=history.index, dtype=float).where(matured_mask)
    history["realized_reference_close"] = realized_reference_close.where(matured_mask)
    history["realized_actual_return"] = actual_return.where(matured_mask)
    history["realized_actual_direction"] = [
        infer_direction_from_return_value(actual_return.iloc[idx], history["horizon_steps"].iloc[idx])
        if bool(matured_mask.iloc[idx])
        else ""
        for idx in range(len(history))
    ]

    evaluation_specs = [
        {
            "task": "regression",
            "model_column": "regression_model_name",
            "predicted_return_column": "regression_predicted_return",
            "predicted_close_column": "regression_predicted_close",
            "predicted_direction_column": None,
            "probability_up_column": None,
            "interval_lower_column": "regression_lower_close_10",
            "interval_upper_column": "regression_upper_close_90",
        },
        {
            "task": "classification",
            "model_column": "classification_model_name",
            "predicted_return_column": None,
            "predicted_close_column": None,
            "predicted_direction_column": "classification_predicted_direction",
            "probability_up_column": "classification_probability_up",
            "interval_lower_column": None,
            "interval_upper_column": None,
        },
        {
            "task": "hybrid",
            "model_column": "hybrid_model_name",
            "predicted_return_column": "hybrid_adjusted_predicted_return",
            "predicted_close_column": "hybrid_adjusted_predicted_close",
            "predicted_direction_column": "hybrid_predicted_direction",
            "probability_up_column": None,
            "interval_lower_column": "hybrid_bear_predicted_close",
            "interval_upper_column": "hybrid_bull_predicted_close",
        },
        {
            "task": "active",
            "model_column": None,
            "predicted_return_column": "active_predicted_return",
            "predicted_close_column": "active_predicted_close",
            "predicted_direction_column": "active_predicted_direction",
            "probability_up_column": None,
            "interval_lower_column": "active_bear_predicted_close",
            "interval_upper_column": "active_bull_predicted_close",
        },
    ]

    for spec in evaluation_specs:
        task = str(spec["task"])
        available_prediction_columns = [
            column_name
            for column_name in (
                spec["predicted_return_column"],
                spec["predicted_close_column"],
                spec["predicted_direction_column"],
                spec["probability_up_column"],
                spec["interval_lower_column"],
                spec["interval_upper_column"],
            )
            if column_name is not None and column_name in history.columns
        ]
        if not available_prediction_columns:
            continue
        if spec["model_column"] is None:
            history[f"{task}_model_name"] = f"{task}_track"
        elif spec["model_column"] not in history.columns:
            continue

        predicted_return = (
            frame_numeric_series(history, spec["predicted_return_column"])
            if spec["predicted_return_column"] is not None
            else pd.Series(np.nan, index=history.index, dtype=float)
        )
        predicted_close = (
            frame_numeric_series(history, spec["predicted_close_column"])
            if spec["predicted_close_column"] is not None
            else pd.Series(np.nan, index=history.index, dtype=float)
        )
        probability_up = (
            frame_numeric_series(history, spec["probability_up_column"])
            if spec["probability_up_column"] is not None
            else pd.Series(np.nan, index=history.index, dtype=float)
        )
        predicted_direction_series = (
            frame_text_series(history, spec["predicted_direction_column"])
            if spec["predicted_direction_column"] is not None
            else pd.Series("", index=history.index, dtype="object")
        )
        inferred_direction = [
            infer_direction_from_return_value(predicted_return.iloc[idx], history["horizon_steps"].iloc[idx])
            if pd.notna(predicted_return.iloc[idx])
            else str(predicted_direction_series.iloc[idx]).strip().upper()
            for idx in range(len(history))
        ]
        inferred_direction_series = pd.Series(inferred_direction, index=history.index, dtype="object").replace({"NAN": "", "NONE": ""})
        realized_direction_hit = pd.Series(np.nan, index=history.index, dtype=float)
        actual_direction_series = history["realized_actual_direction"].astype(str)
        realized_direction_hit.loc[matured_mask] = (
            inferred_direction_series.loc[matured_mask].str.upper()
            == actual_direction_series.loc[matured_mask].str.upper()
        ).astype(float)

        history[f"{task}_realized_direction"] = inferred_direction_series.where(matured_mask, "")
        history[f"{task}_realized_direction_hit"] = realized_direction_hit
        history[f"{task}_realized_strategy_return"] = pd.Series(
            [
                compute_directional_trade_return(inferred_direction_series.iloc[idx], actual_return.iloc[idx])
                if bool(matured_mask.iloc[idx])
                else float("nan")
                for idx in range(len(history))
            ],
            index=history.index,
            dtype=float,
        )
        history[f"{task}_realized_excess_return"] = (
            pd.to_numeric(history[f"{task}_realized_strategy_return"], errors="coerce") - actual_return
        )

        if spec["predicted_return_column"] is not None and spec["predicted_return_column"] in history.columns:
            history[f"{task}_realized_error_return"] = (predicted_return - actual_return).where(matured_mask)
            history[f"{task}_realized_abs_error_return"] = (
                (predicted_return - actual_return).abs().where(matured_mask)
            )
        if spec["predicted_close_column"] is not None and spec["predicted_close_column"] in history.columns:
            actual_close = pd.to_numeric(history["realized_actual_close"], errors="coerce")
            history[f"{task}_realized_error_close"] = (predicted_close - actual_close).where(matured_mask)
            history[f"{task}_realized_abs_error_close"] = (
                (predicted_close - actual_close).abs().where(matured_mask)
            )
        if spec["probability_up_column"] is not None and spec["probability_up_column"] in history.columns:
            actual_up = (actual_return > 0.0).astype(float)
            history[f"{task}_realized_brier_score"] = ((probability_up - actual_up) ** 2).where(matured_mask)
        if spec["interval_lower_column"] is not None and spec["interval_upper_column"] is not None:
            lower_close = frame_numeric_series(history, spec["interval_lower_column"])
            upper_close = frame_numeric_series(history, spec["interval_upper_column"])
            actual_close = pd.to_numeric(history["realized_actual_close"], errors="coerce")
            interval_contains = (
                actual_close.notna()
                & lower_close.notna()
                & upper_close.notna()
                & (actual_close >= lower_close)
                & (actual_close <= upper_close)
            )
            history[f"{task}_realized_interval_coverage"] = interval_contains.where(matured_mask).astype("float")

    return history


def build_prediction_feedback_summary(
    prediction_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if prediction_history is None or prediction_history.empty:
        return pd.DataFrame()

    matured = prediction_history.copy()
    matured = matured.loc[frame_numeric_series(matured, "realized_is_matured").fillna(0.0) > 0.0].copy()
    matured["forecast_target_timestamp"] = pd.to_datetime(
        matured["forecast_target_timestamp"] if "forecast_target_timestamp" in matured.columns else pd.Series(pd.NaT, index=matured.index),
        errors="coerce",
    )
    matured["horizon_steps"] = frame_numeric_series(matured, "horizon_steps")
    matured = matured.dropna(subset=["forecast_target_timestamp", "horizon_steps"])
    if matured.empty:
        return pd.DataFrame()

    latest_target_timestamp = matured["forecast_target_timestamp"].max()
    recency_days = (latest_target_timestamp - matured["forecast_target_timestamp"]).dt.days.clip(lower=0)
    matured["feedback_weight"] = np.power(0.5, recency_days / 90.0)

    task_specs = [
        ("regression", "regression_model_name"),
        ("classification", "classification_model_name"),
        ("hybrid", "hybrid_model_name"),
        ("active", "active_model_name"),
    ]
    rows: list[dict[str, Any]] = []
    for task_name, model_column in task_specs:
        if model_column not in matured.columns:
            continue
        task_frame = matured.copy()
        task_frame["model_name"] = task_frame[model_column].astype(str).str.strip()
        task_frame = task_frame.loc[task_frame["model_name"] != ""].copy()
        if task_frame.empty:
            continue
        for (horizon_value, model_name), group in task_frame.groupby(["horizon_steps", "model_name"], dropna=True):
            weights = pd.to_numeric(group["feedback_weight"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "task": task_name,
                    "horizon_steps": int(horizon_value),
                    "model_name": str(model_name),
                    "feedback_count": int(len(group)),
                    "feedback_weight_sum": float(weights.sum()),
                    "feedback_last_target_timestamp": format_timestamp(group["forecast_target_timestamp"].max()),
                    "feedback_weighted_mean_abs_error_return": compute_weighted_mean(
                        group.get(f"{task_name}_realized_abs_error_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_mean_abs_error_close": compute_weighted_mean(
                        group.get(f"{task_name}_realized_abs_error_close", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_rmse_return": compute_weighted_rmse(
                        group.get(f"{task_name}_realized_error_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_directional_accuracy": compute_weighted_mean(
                        group.get(f"{task_name}_realized_direction_hit", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_mean_strategy_return": compute_weighted_mean(
                        group.get(f"{task_name}_realized_strategy_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_mean_excess_return": compute_weighted_mean(
                        group.get(f"{task_name}_realized_excess_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_brier_score": compute_weighted_mean(
                        group.get(f"{task_name}_realized_brier_score", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_weighted_interval_coverage": compute_weighted_mean(
                        group.get(f"{task_name}_realized_interval_coverage", pd.Series(dtype=float)),
                        weights,
                    ),
                    "feedback_total_strategy_return": compute_compound_return(
                        group.get(f"{task_name}_realized_strategy_return", pd.Series(dtype=float))
                    ),
                    "feedback_total_benchmark_return": compute_compound_return(
                        group.get("realized_actual_return", pd.Series(dtype=float))
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    summary["feedback_total_excess_return"] = (
        pd.to_numeric(summary["feedback_total_strategy_return"], errors="coerce")
        - pd.to_numeric(summary["feedback_total_benchmark_return"], errors="coerce")
    )
    return summary.sort_values(
        ["task", "horizon_steps", "feedback_weight_sum", "feedback_count", "model_name"],
        ascending=[True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_prediction_feedback_state_summary(
    prediction_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if prediction_history is None or prediction_history.empty:
        return pd.DataFrame()

    matured = prediction_history.copy()
    matured = matured.loc[frame_numeric_series(matured, "realized_is_matured").fillna(0.0) > 0.0].copy()
    matured["forecast_target_timestamp"] = pd.to_datetime(
        matured["forecast_target_timestamp"] if "forecast_target_timestamp" in matured.columns else pd.Series(pd.NaT, index=matured.index),
        errors="coerce",
    )
    matured["horizon_steps"] = frame_numeric_series(matured, "horizon_steps")
    matured["macro_risk_regime"] = frame_text_series(matured, "macro_risk_regime").replace("", "UNKNOWN")
    matured = matured.dropna(subset=["forecast_target_timestamp", "horizon_steps"])
    if matured.empty:
        return pd.DataFrame()

    latest_target_timestamp = matured["forecast_target_timestamp"].max()
    recency_days = (latest_target_timestamp - matured["forecast_target_timestamp"]).dt.days.clip(lower=0)
    matured["feedback_weight"] = np.power(0.5, recency_days / 90.0)

    task_specs = [
        ("regression", "regression_model_name"),
        ("classification", "classification_model_name"),
        ("hybrid", "hybrid_model_name"),
        ("active", "active_model_name"),
    ]
    rows: list[dict[str, Any]] = []
    for task_name, model_column in task_specs:
        if model_column not in matured.columns:
            continue
        task_frame = matured.copy()
        task_frame["model_name"] = task_frame[model_column].astype(str).str.strip()
        task_frame = task_frame.loc[task_frame["model_name"] != ""].copy()
        if task_frame.empty:
            continue
        group_keys = ["horizon_steps", "macro_risk_regime", "model_name"]
        for group_values, group in task_frame.groupby(group_keys, dropna=True):
            horizon_value, macro_risk_regime, model_name = group_values
            weights = pd.to_numeric(group["feedback_weight"], errors="coerce").fillna(0.0)
            rows.append(
                {
                    "task": task_name,
                    "horizon_steps": int(horizon_value),
                    "macro_risk_regime": str(macro_risk_regime),
                    "model_name": str(model_name),
                    "state_feedback_count": int(len(group)),
                    "state_feedback_weight_sum": float(weights.sum()),
                    "state_feedback_last_target_timestamp": format_timestamp(group["forecast_target_timestamp"].max()),
                    "state_feedback_weighted_mean_abs_error_return": compute_weighted_mean(
                        group.get(f"{task_name}_realized_abs_error_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "state_feedback_weighted_directional_accuracy": compute_weighted_mean(
                        group.get(f"{task_name}_realized_direction_hit", pd.Series(dtype=float)),
                        weights,
                    ),
                    "state_feedback_weighted_mean_strategy_return": compute_weighted_mean(
                        group.get(f"{task_name}_realized_strategy_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "state_feedback_weighted_mean_excess_return": compute_weighted_mean(
                        group.get(f"{task_name}_realized_excess_return", pd.Series(dtype=float)),
                        weights,
                    ),
                    "state_feedback_weighted_brier_score": compute_weighted_mean(
                        group.get(f"{task_name}_realized_brier_score", pd.Series(dtype=float)),
                        weights,
                    ),
                    "state_feedback_total_strategy_return": compute_compound_return(
                        group.get(f"{task_name}_realized_strategy_return", pd.Series(dtype=float))
                    ),
                    "state_feedback_total_benchmark_return": compute_compound_return(
                        group.get("realized_actual_return", pd.Series(dtype=float))
                    ),
                }
            )

    if not rows:
        return pd.DataFrame()
    summary = pd.DataFrame(rows)
    summary["state_feedback_total_excess_return"] = (
        pd.to_numeric(summary["state_feedback_total_strategy_return"], errors="coerce")
        - pd.to_numeric(summary["state_feedback_total_benchmark_return"], errors="coerce")
    )
    return summary.sort_values(
        ["task", "horizon_steps", "macro_risk_regime", "state_feedback_weight_sum", "state_feedback_count", "model_name"],
        ascending=[True, True, True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def build_prediction_feedback_state_winner_table(
    prediction_feedback_state_summary: pd.DataFrame | None,
) -> pd.DataFrame:
    if prediction_feedback_state_summary is None or prediction_feedback_state_summary.empty:
        return pd.DataFrame()
    winners: list[pd.DataFrame] = []
    summary = prediction_feedback_state_summary.copy()
    summary["winner_rank_score"] = (
        pd.to_numeric(summary["state_feedback_weighted_mean_excess_return"], errors="coerce").fillna(float("-inf")) * 0.45
        + pd.to_numeric(summary["state_feedback_weighted_directional_accuracy"], errors="coerce").fillna(float("-inf")) * 0.35
        + (-pd.to_numeric(summary["state_feedback_weighted_mean_abs_error_return"], errors="coerce").fillna(float("inf")) * 0.20)
    )
    for _, group in summary.groupby(["task", "horizon_steps", "macro_risk_regime"], dropna=True):
        ranked = group.sort_values(
            ["winner_rank_score", "state_feedback_weight_sum", "state_feedback_count"],
            ascending=[False, False, False],
            na_position="last",
        ).head(1)
        winners.append(ranked)
    if not winners:
        return pd.DataFrame()
    winner_table = pd.concat(winners, ignore_index=True)
    return winner_table.drop(columns=["winner_rank_score"], errors="ignore")


def sanitize_feature_name(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in str(value)).strip("_")


def artemis_metric_column_name(value: str) -> str | None:
    metric_name = sanitize_feature_name(value)
    if not metric_name:
        return None
    for chain_prefix in ("ethereum_", "eth_"):
        if metric_name.startswith(chain_prefix):
            metric_name = metric_name[len(chain_prefix) :]
            break
    metric_name = metric_name.strip("_")
    if not metric_name:
        return None
    return f"artemis_{metric_name}"


def merge_external_feature_blocks(
    merged: pd.DataFrame | None,
    incoming: pd.DataFrame,
) -> pd.DataFrame:
    incoming = normalize_index(incoming)
    if merged is None or merged.empty:
        return incoming.copy()

    combined_index = merged.index.union(incoming.index)
    combined = merged.reindex(combined_index).copy()
    incoming = incoming.reindex(combined_index)

    for column in incoming.columns:
        if column in combined.columns:
            combined[column] = pd.to_numeric(incoming[column], errors="coerce").combine_first(
                pd.to_numeric(combined[column], errors="coerce")
            )
        else:
            combined[column] = pd.to_numeric(incoming[column], errors="coerce")
    return normalize_index(combined)


def load_external_feature_csvs(paths: list[str] | None) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame()

    merged: pd.DataFrame | None = None
    for raw_path in paths:
        source = Path(raw_path)
        if not source.exists():
            raise ValueError(f"External feature CSV was not found: {source}")
        frame = pd.read_csv(source, index_col=0, parse_dates=True)
        frame = normalize_index(frame)
        numeric = pd.DataFrame(index=frame.index)
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                coerced = pd.to_numeric(series, errors="coerce")
            else:
                text = series.astype(str).str.strip()
                pct_mask = text.str.endswith("%", na=False)
                cleaned = (
                    text.replace({"": np.nan, "nan": np.nan, "None": np.nan, "null": np.nan, "-": np.nan})
                    .str.replace(",", "", regex=False)
                    .str.replace("%", "", regex=False)
                )
                coerced = pd.to_numeric(cleaned, errors="coerce")
                if bool(pct_mask.any()) and float(pct_mask.mean()) >= 0.80:
                    coerced = coerced / 100.0
            if coerced.notna().any():
                numeric[column] = coerced
        if numeric.empty:
            continue

        stem = sanitize_feature_name(source.stem) or "external"
        artemis_like_columns = [
            column
            for column in numeric.columns
            if str(column).strip().lower().startswith(("ethereum - ", "eth - "))
        ]
        if artemis_like_columns and len(artemis_like_columns) == len(numeric.columns):
            renamed = {
                column: (artemis_metric_column_name(column) or f"external_{stem}_{sanitize_feature_name(column)}")
                for column in numeric.columns
            }
        else:
            renamed = {
                column: f"external_{stem}_{sanitize_feature_name(column)}"
                for column in numeric.columns
            }
        numeric = numeric.rename(columns=renamed)
        merged = merge_external_feature_blocks(merged, numeric)

    if merged is None:
        return pd.DataFrame()
    return normalize_index(merged)


def merge_external_features(market_data: pd.DataFrame, external_features: pd.DataFrame) -> pd.DataFrame:
    if external_features.empty:
        return market_data

    merged = normalize_index(market_data.join(external_features, how="left"))
    merged = merged.reindex(market_data.index)
    return merged


def build_external_feature_summary(market_data: pd.DataFrame) -> pd.DataFrame:
    external_columns = [
        column
        for column in market_data.columns
        if column.startswith(GENERIC_VENDOR_PREFIXES)
    ]
    rows: list[dict[str, Any]] = []
    latest_index_value = market_data.index.max() if len(market_data.index) else None
    for column in external_columns:
        series = pd.to_numeric(market_data[column], errors="coerce")
        non_null = series.dropna()
        latest_age_days = float("nan")
        history_span_days = float("nan")
        if latest_index_value is not None and not non_null.empty:
            latest_age_days = float((pd.Timestamp(latest_index_value) - pd.Timestamp(non_null.index.max())).days)
        if not non_null.empty:
            history_span_days = float((pd.Timestamp(non_null.index.max()) - pd.Timestamp(non_null.index.min())).days)
        short_history_excluded = bool(
            pd.notna(history_span_days) and float(history_span_days) < float(DEFAULT_MIN_VENDOR_HISTORY_DAYS)
        )
        rows.append(
            {
                "column": column,
                "column_group": column.split("_", 1)[0],
                "non_null_rows": int(non_null.shape[0]),
                "coverage_ratio": float(non_null.shape[0] / len(series)) if len(series) > 0 else 0.0,
                "start": format_timestamp(non_null.index.min()) if not non_null.empty else "",
                "end": format_timestamp(non_null.index.max()) if not non_null.empty else "",
                "history_span_days": history_span_days,
                "latest_age_days": latest_age_days,
                "short_history_excluded": short_history_excluded,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "column",
                "column_group",
                "non_null_rows",
                "coverage_ratio",
                "start",
                "end",
                "history_span_days",
                "latest_age_days",
                "short_history_excluded",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["short_history_excluded", "coverage_ratio", "column"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def short_history_vendor_columns(
    market_data: pd.DataFrame,
    min_history_days: int = DEFAULT_MIN_VENDOR_HISTORY_DAYS,
) -> set[str]:
    external_summary = build_external_feature_summary(market_data)
    if external_summary.empty:
        return set()
    short_history = external_summary.loc[
        pd.to_numeric(external_summary.get("history_span_days"), errors="coerce").fillna(-1.0) < float(min_history_days),
        "column",
    ]
    return {str(value) for value in short_history.tolist()}


def stale_vendor_columns(
    market_data: pd.DataFrame,
    max_age_days: int = DEFAULT_STALE_VENDOR_MAX_AGE_DAYS,
) -> set[str]:
    external_summary = build_external_feature_summary(market_data)
    if external_summary.empty:
        return set()
    age_days = pd.to_numeric(external_summary.get("latest_age_days"), errors="coerce")
    stale = external_summary.loc[
        age_days.notna() & (age_days >= float(max_age_days)),
        "column",
    ]
    return {str(value) for value in stale.tolist()}


def resolve_storage_targets(
    output_dir: str | Path,
    dataset_csv: str | Path,
    prediction_history_csv: str | Path | None,
    drive_root: str | Path | None = None,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir)
    dataset_path = Path(dataset_csv)
    history_path = Path(prediction_history_csv) if prediction_history_csv else (output_path / "prediction_history.csv")

    if drive_root is None:
        return output_path, dataset_path, history_path

    root = Path(drive_root)
    root.mkdir(parents=True, exist_ok=True)

    if str(output_dir) == "eth_forecast_outputs":
        output_path = root / "outputs"
    if str(dataset_csv) == "market_data_cache.csv":
        dataset_path = root / "lake" / "raw" / "market" / "market_data_cache.csv"
    if prediction_history_csv is None or str(prediction_history_csv) == "":
        history_path = root / "prediction_history.csv"

    return output_path, dataset_path, history_path


def resolve_master_data_path(
    master_data_csv: str | Path | None,
    drive_root: str | Path | None = None,
) -> Path | None:
    if master_data_csv is None or str(master_data_csv).strip() == "":
        return None

    master_path = Path(master_data_csv)
    if drive_root is not None and str(master_data_csv) == DEFAULT_MASTER_DATA_CSV:
        master_path = Path(drive_root) / "lake" / "gold" / DEFAULT_MASTER_DATA_CSV
    return master_path


def load_or_prepare_market_data(
    period: str,
    interval: str,
    output_dir: str | Path,
    dataset_csv: str | Path,
    prediction_history_csv: str | Path | None,
    external_feature_csvs: list[str] | None,
    drive_root: str | Path | None,
    use_dataset_cache: bool,
    cache_lookback_rows: int,
    master_data_csv: str | Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, Path, Path, Path, Path | None]:
    output_dir_path, dataset_csv_path, history_path = resolve_storage_targets(
        output_dir=output_dir,
        dataset_csv=dataset_csv,
        prediction_history_csv=prediction_history_csv,
        drive_root=drive_root,
    )
    master_data_path = resolve_master_data_path(master_data_csv, drive_root=drive_root)

    if master_data_path is not None and master_data_path.exists():
        log_progress(verbose, f"Loading master data from {master_data_path}...")
        market_data = load_market_data_csv(master_data_path)
        if market_data.empty:
            raise ValueError(f"Master dataset exists but is empty: {master_data_path}")
        return market_data, output_dir_path, dataset_csv_path, history_path, master_data_path

    log_progress(verbose, "Loading market data...")
    if use_dataset_cache:
        market_data = update_market_data_cache(
            DEFAULT_TICKERS,
            dataset_csv=dataset_csv_path,
            period=period,
            interval=interval,
            lookback_rows=cache_lookback_rows,
            verbose=verbose,
        )
    else:
        market_data = download_market_data(DEFAULT_TICKERS, period=period, interval=interval)

    external_features = load_external_feature_csvs(external_feature_csvs)
    if not external_features.empty:
        market_data = merge_external_features(market_data, external_features)

    if master_data_path is not None:
        log_progress(verbose, f"Saving master data to {master_data_path}...")
        save_market_data_csv(market_data, master_data_path)

    return market_data, output_dir_path, dataset_csv_path, history_path, master_data_path


def build_prediction_history_features(
    index: pd.Index,
    prediction_history: pd.DataFrame | None,
) -> pd.DataFrame:
    if prediction_history is None or prediction_history.empty:
        return pd.DataFrame(index=index)

    feature_blocks: list[pd.DataFrame] = []
    history = prediction_history.copy()
    history = history.dropna(subset=["forecast_input_timestamp", "horizon_steps"])
    if history.empty:
        return pd.DataFrame(index=index)

    for horizon_value in sorted(history["horizon_steps"].astype(int).unique()):
        horizon_history = history.loc[history["horizon_steps"] == horizon_value].copy()
        horizon_history = (
            horizon_history.sort_values(["forecast_input_timestamp", "run_timestamp"])
            .drop_duplicates(subset=["forecast_input_timestamp"], keep="last")
            .set_index("forecast_input_timestamp")
            .sort_index()
        )
        if horizon_history.empty:
            continue

        numeric_feature_source = pd.DataFrame(index=horizon_history.index)
        candidate_numeric_columns = [
            "regression_predicted_return",
            "classification_probability_up",
            "classification_confidence",
            "realized_actual_return",
            "classification_realized_direction_hit",
            "hybrid_realized_direction_hit",
            "active_realized_direction_hit",
            "hybrid_realized_abs_error_return",
            "active_realized_abs_error_return",
        ]
        for column in candidate_numeric_columns:
            if column in horizon_history.columns:
                numeric_feature_source[column] = pd.to_numeric(horizon_history[column], errors="coerce")

        aligned = numeric_feature_source.reindex(index).infer_objects(copy=False)
        aligned = aligned.ffill().shift(1)
        tag = f"history_h{horizon_value}"
        pred_return = pd.to_numeric(
            aligned["regression_predicted_return"],
            errors="coerce",
        ) if "regression_predicted_return" in aligned.columns else pd.Series(np.nan, index=index)
        prob_up = pd.to_numeric(
            aligned["classification_probability_up"],
            errors="coerce",
        ) if "classification_probability_up" in aligned.columns else pd.Series(np.nan, index=index)
        confidence = pd.to_numeric(
            aligned["classification_confidence"],
            errors="coerce",
        ) if "classification_confidence" in aligned.columns else pd.Series(np.nan, index=index)
        realized_actual_return = pd.to_numeric(
            aligned["realized_actual_return"],
            errors="coerce",
        ) if "realized_actual_return" in aligned.columns else pd.Series(np.nan, index=index)
        classification_hit = pd.to_numeric(
            aligned["classification_realized_direction_hit"],
            errors="coerce",
        ) if "classification_realized_direction_hit" in aligned.columns else pd.Series(np.nan, index=index)
        hybrid_hit = pd.to_numeric(
            aligned["hybrid_realized_direction_hit"],
            errors="coerce",
        ) if "hybrid_realized_direction_hit" in aligned.columns else pd.Series(np.nan, index=index)
        active_hit = pd.to_numeric(
            aligned["active_realized_direction_hit"],
            errors="coerce",
        ) if "active_realized_direction_hit" in aligned.columns else pd.Series(np.nan, index=index)
        hybrid_abs_error = pd.to_numeric(
            aligned["hybrid_realized_abs_error_return"],
            errors="coerce",
        ) if "hybrid_realized_abs_error_return" in aligned.columns else pd.Series(np.nan, index=index)
        active_abs_error = pd.to_numeric(
            aligned["active_realized_abs_error_return"],
            errors="coerce",
        ) if "active_realized_abs_error_return" in aligned.columns else pd.Series(np.nan, index=index)
        feature_blocks.append(
            pd.DataFrame(
                {
                    f"{tag}_pred_return": pred_return,
                    f"{tag}_prob_up": prob_up,
                    f"{tag}_confidence": confidence,
                    f"{tag}_signal_bias": 2.0 * prob_up - 1.0,
                    f"{tag}_realized_return": realized_actual_return,
                    f"{tag}_classification_hit_rate_5": classification_hit.rolling(5, min_periods=2).mean(),
                    f"{tag}_hybrid_hit_rate_5": hybrid_hit.rolling(5, min_periods=2).mean(),
                    f"{tag}_active_hit_rate_5": active_hit.rolling(5, min_periods=2).mean(),
                    f"{tag}_hybrid_abs_error_return_5": hybrid_abs_error.rolling(5, min_periods=2).mean(),
                    f"{tag}_active_abs_error_return_5": active_abs_error.rolling(5, min_periods=2).mean(),
                },
                index=index,
            )
        )

    if not feature_blocks:
        return pd.DataFrame(index=index)

    history_features = pd.concat(feature_blocks, axis=1)
    for column in list(history_features.columns):
        if column.endswith(("_pred_return", "_prob_up", "_confidence", "_signal_bias")):
            history_features[f"{column}_delta_1"] = history_features[column].diff(1)
            history_features[f"{column}_delta_7"] = history_features[column].diff(7)
    return history_features


def download_market_data(
    symbol_map: dict[str, str],
    period: str,
    interval: str,
) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for alias, symbol in symbol_map.items():
        try:
            current = download_symbol_history(symbol, alias, period=period, interval=interval)
        except Exception as exc:
            if alias in REQUIRED_TICKERS:
                raise
            print(f"Skipping optional symbol {symbol} ({alias}): {exc}")
            continue
        merged = current if merged is None else merged.join(current, how="outer")

    if merged is None or merged.empty:
        raise ValueError("No market data was downloaded.")

    return finalize_market_data(merged)


def build_feature_coverage_summary(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    training_mask: pd.Series,
    min_feature_coverage: float,
) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame(
            columns=[
                "feature",
                "feature_group",
                "non_null_rows",
                "coverage_ratio",
                "unique_values",
                "min_rows_required",
                "kept",
            ]
        )
    training_features = feature_frame.loc[training_mask, feature_columns]
    rows: list[dict[str, Any]] = []
    row_count = len(training_features)
    min_standard_rows = max(DEFAULT_MIN_STANDARD_FEATURE_ROWS, int(row_count * min_feature_coverage))
    min_history_rows = max(DEFAULT_MIN_HISTORY_FEATURE_ROWS, int(row_count * 0.005))

    for column in feature_columns:
        series = training_features[column]
        non_null_rows = int(series.notna().sum())
        unique_values = int(series.nunique(dropna=True))
        column_group = "history" if column.startswith("history_") else "standard"
        min_rows_required = min_history_rows if column_group == "history" else min_standard_rows
        keep = bool(non_null_rows >= min_rows_required and unique_values >= 2)
        rows.append(
            {
                "feature": column,
                "feature_group": column_group,
                "non_null_rows": non_null_rows,
                "coverage_ratio": float(non_null_rows / row_count) if row_count > 0 else 0.0,
                "unique_values": unique_values,
                "min_rows_required": int(min_rows_required),
                "kept": keep,
            }
        )

    summary = pd.DataFrame(rows).sort_values(
        ["kept", "coverage_ratio", "feature"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    return summary


def select_usable_feature_columns(
    feature_frame: pd.DataFrame,
    feature_columns: list[str],
    training_mask: pd.Series,
    min_feature_coverage: float,
    horizon: int | None = None,
    target_column: str = "target_return",
) -> tuple[list[str], pd.DataFrame]:
    coverage_summary = build_feature_coverage_summary(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        training_mask=training_mask,
        min_feature_coverage=min_feature_coverage,
    )
    kept = coverage_summary.loc[coverage_summary["kept"], "feature"].tolist()
    if not kept:
        raise ValueError("No usable feature columns remained after applying coverage and uniqueness filters.")

    selected, selection_meta, correlation_dropped = prune_feature_candidates(
        feature_frame=feature_frame,
        candidate_features=kept,
        training_mask=training_mask,
        target_column=target_column,
        horizon=horizon,
    )
    if not selected:
        selected = kept

    coverage_summary = coverage_summary.merge(selection_meta, on="feature", how="left")
    coverage_summary["pruned_by_correlation"] = coverage_summary["feature"].isin(correlation_dropped)
    coverage_summary["kept"] = coverage_summary["feature"].isin(selected)
    coverage_summary = coverage_summary.sort_values(
        ["kept", "signal_score", "coverage_ratio", "feature"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return selected, coverage_summary


def select_fold_features(
    dataset: pd.DataFrame,
    candidate_feature_columns: list[str],
    train_positions: np.ndarray,
    min_feature_coverage: float,
    horizon: int | None = None,
    target_column: str = "target_return",
) -> list[str]:
    """Re-run coverage + rank + prune using only the walk-forward fold's train rows.

    Phase 1.1 fix: the global select_usable_feature_columns computes Spearman
    correlations over the full training mask, which includes rows that later
    land in test folds. This helper restricts that computation to the integer
    ``train_positions`` produced by ``splitter.split(X)`` so the test folds are
    truly held out.
    """
    available = [column for column in candidate_feature_columns if column in dataset.columns]
    if not available:
        return []
    fold_mask = pd.Series(False, index=dataset.index)
    fold_mask.iloc[np.asarray(train_positions, dtype=int)] = True
    try:
        selected, _ = select_usable_feature_columns(
            feature_frame=dataset,
            feature_columns=available,
            training_mask=fold_mask,
            min_feature_coverage=min_feature_coverage,
            horizon=horizon,
            target_column=target_column,
        )
    except ValueError:
        return available
    return selected


def max_feature_count_for_horizon(horizon: int | None) -> int:
    if horizon is None:
        return DEFAULT_MAX_FEATURES_LONG
    return DEFAULT_MAX_FEATURES_SHORT if horizon <= 7 else DEFAULT_MAX_FEATURES_LONG


def feature_score_adjustment(column: str) -> float:
    adjustment = 0.0
    if column.startswith("eth_"):
        adjustment += 0.05
    if column.startswith("history_"):
        adjustment += 0.01
    if column.startswith(("eth_", "btc_", "eth_vs_btc_")) and (
        "_120" in column or "_180" in column or "_365" in column or "cycle_position" in column
    ):
        adjustment += 0.025
    if column in {
        "eth_tail_event_pressure",
        "eth_cycle_position_365_centered",
        "eth_drawdown_180",
        "eth_return_180",
        "eth_vol_30_180_ratio",
        "eth_vs_btc_strength_90",
    }:
        adjustment += 0.04
    if column.startswith(GENERIC_VENDOR_PREFIXES):
        adjustment -= 0.01
        if column.endswith(("__asof", "__age_days", "__is_missing")):
            adjustment += 0.03
        if column.endswith(("_pct_1", "_pct_7", "_diff_1")):
            adjustment -= 0.08
    return adjustment


def rank_feature_candidates(
    feature_frame: pd.DataFrame,
    candidate_features: list[str],
    training_mask: pd.Series,
    target_column: str,
) -> pd.DataFrame:
    if not candidate_features:
        return pd.DataFrame(columns=["feature", "signal_score", "selection_rank"])

    target = pd.to_numeric(feature_frame.loc[training_mask, target_column], errors="coerce")
    minimum_valid_rows = max(DEFAULT_MIN_STANDARD_FEATURE_ROWS, int(len(target) * 0.05))
    rows: list[dict[str, Any]] = []

    for column in candidate_features:
        series = pd.to_numeric(feature_frame.loc[training_mask, column], errors="coerce")
        valid = series.notna() & target.notna()
        if int(valid.sum()) < minimum_valid_rows:
            score = float("-inf")
        else:
            corr = series.loc[valid].corr(target.loc[valid], method="spearman")
            score = abs(float(corr)) if pd.notna(corr) else 0.0
            score += feature_score_adjustment(column)
        rows.append({"feature": column, "signal_score": float(score)})

    ranking = pd.DataFrame(rows).sort_values(
        ["signal_score", "feature"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    ranking["selection_rank"] = np.arange(1, len(ranking) + 1, dtype=int)
    return ranking


def prune_feature_candidates(
    feature_frame: pd.DataFrame,
    candidate_features: list[str],
    training_mask: pd.Series,
    target_column: str,
    horizon: int | None,
    corr_threshold: float = DEFAULT_FEATURE_CORRELATION_THRESHOLD,
) -> tuple[list[str], pd.DataFrame, set[str]]:
    ranking = rank_feature_candidates(
        feature_frame=feature_frame,
        candidate_features=candidate_features,
        training_mask=training_mask,
        target_column=target_column,
    )
    if ranking.empty:
        return [], ranking, set()

    max_features = min(max_feature_count_for_horizon(horizon), len(candidate_features))
    ranked_features = ranking.loc[np.isfinite(ranking["signal_score"]), "feature"].tolist()
    if not ranked_features:
        ranked_features = candidate_features[:max_features]
        return ranked_features, ranking, set()

    training_values = (
        feature_frame.loc[training_mask, ranked_features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    corr_matrix = training_values.corr(method="spearman").abs()

    selected: list[str] = []
    correlation_dropped: set[str] = set()
    for feature in ranked_features:
        if feature in correlation_dropped:
            continue
        selected.append(feature)
        if len(selected) >= max_features:
            break
        if feature in corr_matrix.columns:
            related = corr_matrix.index[
                (corr_matrix[feature] >= corr_threshold) & (corr_matrix.index != feature)
            ].tolist()
            for other in related:
                if other not in selected:
                    correlation_dropped.add(other)

    if len(selected) < min(max_features, len(ranked_features)):
        for feature in ranked_features:
            if feature not in selected:
                selected.append(feature)
            if len(selected) >= max_features:
                break

    return selected, ranking, correlation_dropped


def update_market_data_cache(
    symbol_map: dict[str, str],
    dataset_csv: str | Path,
    period: str,
    interval: str,
    lookback_rows: int = CACHE_UPDATE_LOOKBACK_ROWS,
    verbose: bool = True,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    cache_path = Path(dataset_csv)
    cached = load_market_data_csv(cache_path)
    metadata: dict[str, Any] = {
        "used_cached_only": False,
        "refresh_start": "",
        "refreshed_aliases": [],
        "fallback_cached_aliases": [],
        "failed_aliases": [],
    }
    if cached.empty:
        log_progress(verbose, f"Creating market dataset cache at {cache_path}...")
        fresh = download_market_data(symbol_map, period=period, interval=interval)
        save_market_data_csv(fresh, cache_path)
        metadata["refreshed_aliases"] = list(symbol_map.keys())
        return (fresh, metadata) if return_metadata else fresh

    latest_timestamp = cached.index.max()
    refresh_start = pd.Timestamp(latest_timestamp) - (interval_to_offset(interval) * lookback_rows)
    metadata["refresh_start"] = format_timestamp(refresh_start)
    log_progress(
        verbose,
        (
            f"Updating market dataset cache from {refresh_start.strftime('%Y-%m-%d %H:%M:%S')} "
            f"using {cache_path}..."
        ),
    )

    refreshed: pd.DataFrame | None = None
    for alias, symbol in symbol_map.items():
        try:
            current = download_symbol_history(symbol, alias, interval=interval, start=refresh_start)
        except Exception as exc:
            existing_alias_columns = [column for column in cached.columns if column.startswith(f"{alias}_")]
            if alias in REQUIRED_TICKERS and not existing_alias_columns:
                raise
            metadata["failed_aliases"].append(f"{alias}:{exc}")
            if existing_alias_columns:
                metadata["fallback_cached_aliases"].append(alias)
            print(f"Keeping cached data for {symbol} ({alias}) because refresh failed: {exc}")
            continue
        metadata["refreshed_aliases"].append(alias)
        refreshed = current if refreshed is None else refreshed.join(current, how="outer")

    if refreshed is None or refreshed.empty:
        log_progress(verbose, "Recent refresh failed for all symbols, using cached dataset as-is.")
        metadata["used_cached_only"] = True
        cached_result = finalize_market_data(cached)
        return (cached_result, metadata) if return_metadata else cached_result

    merged = cached.combine_first(refreshed)
    merged.update(refreshed)
    merged = normalize_index(merged)
    merged = finalize_market_data(merged)
    save_market_data_csv(merged, cache_path)
    return (merged, metadata) if return_metadata else merged


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_gain = up.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = down.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.rolling(window).mean()


def compute_rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    rolling_mean = series.rolling(window).mean()
    rolling_std = series.rolling(window).std()
    return (series - rolling_mean) / rolling_std


def infer_feature_update_interval_days(series: pd.Series) -> float:
    valid_index = pd.to_datetime(series.dropna().index)
    if len(valid_index) < 3:
        return 1.0
    deltas = np.diff(valid_index.asi8).astype(float) / 86_400_000_000_000.0
    deltas = deltas[np.isfinite(deltas) & (deltas > 0.0)]
    if len(deltas) == 0:
        return 1.0
    return float(np.median(deltas))


def prepare_external_source_series(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series, float]:
    numeric = pd.to_numeric(series, errors="coerce")
    index_series = pd.Series(pd.to_datetime(series.index), index=series.index)
    last_valid_timestamp = index_series.where(numeric.notna()).ffill()
    age_days = ((index_series - last_valid_timestamp) / pd.Timedelta(days=1)).astype(float)
    age_days = age_days.where(last_valid_timestamp.notna())
    missing_flag = numeric.isna().astype(float)
    asof_series = numeric.ffill()
    update_interval_days = infer_feature_update_interval_days(numeric)
    return asof_series, missing_flag, age_days, update_interval_days


def compute_drawdown(series: pd.Series, window: int) -> pd.Series:
    rolling_peak = series.rolling(window).max()
    return series / rolling_peak - 1.0


def compute_boolean_streak(mask: pd.Series) -> pd.Series:
    boolean_mask = mask.fillna(False).astype(bool)
    groups = (~boolean_mask).cumsum()
    streak = boolean_mask.groupby(groups).cumsum()
    return streak.where(boolean_mask, 0.0).astype(float)


def compute_stochastic_k(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    rolling_low = low.rolling(window).min()
    rolling_high = high.rolling(window).max()
    return (close - rolling_low) / (rolling_high - rolling_low)


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume.fillna(0.0)).cumsum()


def compute_mfi(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 14,
) -> pd.Series:
    typical_price = (high + low + close) / 3.0
    money_flow = typical_price * volume.fillna(0.0)
    positive_flow = money_flow.where(typical_price.diff() > 0, 0.0).rolling(window).sum()
    negative_flow = money_flow.where(typical_price.diff() < 0, 0.0).abs().rolling(window).sum()
    money_ratio = positive_flow / negative_flow.replace(0.0, np.nan)
    return 100 - (100 / (1 + money_ratio))


def compute_cmf(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.Series:
    denominator = (high - low).replace(0.0, np.nan)
    multiplier = ((close - low) - (high - close)) / denominator
    money_flow_volume = multiplier * volume.fillna(0.0)
    rolling_volume = volume.fillna(0.0).rolling(window).sum().replace(0.0, np.nan)
    return money_flow_volume.rolling(window).sum() / rolling_volume


def compute_downside_volatility(returns: pd.Series, window: int) -> pd.Series:
    negative_returns = returns.where(returns < 0.0, 0.0)
    return negative_returns.rolling(window).std()


def compute_rolling_beta(asset_returns: pd.Series, benchmark_returns: pd.Series, window: int) -> pd.Series:
    covariance = asset_returns.rolling(window).cov(benchmark_returns)
    benchmark_variance = benchmark_returns.rolling(window).var()
    return covariance / benchmark_variance


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0.0), up_move, 0.0),
        index=high.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0.0), down_move, 0.0),
        index=high.index,
    )
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    average_true_range = true_range.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    plus_di = 100.0 * plus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / average_true_range
    minus_di = 100.0 * minus_dm.ewm(alpha=1 / window, min_periods=window, adjust=False).mean() / average_true_range
    dx = 100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, np.nan)
    adx = dx.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    return adx, plus_di, minus_di


def compute_ichimoku(
    high: pd.Series,
    low: pd.Series,
) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    tenkan = (high.rolling(9, min_periods=9).max() + low.rolling(9, min_periods=9).min()) / 2.0
    kijun = (high.rolling(26, min_periods=26).max() + low.rolling(26, min_periods=26).min()) / 2.0
    senkou_a = ((tenkan + kijun) / 2.0).shift(26)
    senkou_b = (
        (high.rolling(52, min_periods=52).max() + low.rolling(52, min_periods=52).min()) / 2.0
    ).shift(26)
    return tenkan, kijun, senkou_a, senkou_b


def compute_rolling_vwap(
    typical_price: pd.Series,
    volume: pd.Series,
    window: int,
) -> pd.Series:
    weighted_sum = (typical_price * volume.fillna(0.0)).rolling(window).sum()
    volume_sum = volume.fillna(0.0).rolling(window).sum().replace(0.0, np.nan)
    return weighted_sum / volume_sum


def compute_volume_profile_levels(
    typical_values: np.ndarray,
    volume_values: np.ndarray,
    low_min: float,
    high_max: float,
    bins: int = 48,
    value_area_pct: float = 0.70,
) -> tuple[float, float, float]:
    if not np.isfinite(low_min) or not np.isfinite(high_max) or high_max <= low_min:
        return np.nan, np.nan, np.nan

    edges = np.linspace(low_min, high_max, bins + 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    histogram = np.zeros(bins, dtype=float)

    for typical_price, volume in zip(typical_values, volume_values):
        if not np.isfinite(typical_price) or not np.isfinite(volume) or volume <= 0.0:
            continue
        bucket = np.searchsorted(edges, typical_price, side="right") - 1
        if 0 <= bucket < bins:
            histogram[bucket] += volume

    total_volume = histogram.sum()
    if total_volume <= 0.0:
        return np.nan, np.nan, np.nan

    poc_index = int(np.nanargmax(histogram))
    chosen: set[int] = set()
    cumulative_volume = 0.0
    for index in np.argsort(histogram)[::-1]:
        chosen.add(int(index))
        cumulative_volume += histogram[index]
        if cumulative_volume / total_volume >= value_area_pct:
            break

    ordered_indices = np.array(sorted(chosen), dtype=int)
    return centers[poc_index], centers[ordered_indices[0]], centers[ordered_indices[-1]]


def compute_rolling_volume_profile(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    window: int,
    bins: int = 48,
    value_area_pct: float = 0.70,
) -> pd.DataFrame:
    typical_price = ((high + low + close) / 3.0).to_numpy(dtype=float)
    volume_values = volume.to_numpy(dtype=float)
    high_values = high.to_numpy(dtype=float)
    low_values = low.to_numpy(dtype=float)
    poc = np.full(len(close), np.nan)
    val = np.full(len(close), np.nan)
    vah = np.full(len(close), np.nan)

    for end_index in range(window - 1, len(close)):
        start_index = end_index - window + 1
        low_min = np.nanmin(low_values[start_index : end_index + 1])
        high_max = np.nanmax(high_values[start_index : end_index + 1])
        profile = compute_volume_profile_levels(
            typical_values=typical_price[start_index : end_index + 1],
            volume_values=volume_values[start_index : end_index + 1],
            low_min=low_min,
            high_max=high_max,
            bins=bins,
            value_area_pct=value_area_pct,
        )
        poc[end_index], val[end_index], vah[end_index] = profile

    return pd.DataFrame({"poc": poc, "val": val, "vah": vah}, index=close.index)


def compute_naked_poc(
    high: pd.Series,
    low: pd.Series,
    poc_series: pd.Series,
) -> pd.Series:
    candidate_poc = poc_series.shift(1)
    active_level = np.nan
    result = np.full(len(poc_series), np.nan)

    for index in range(len(poc_series)):
        new_level = candidate_poc.iloc[index]
        if np.isfinite(new_level):
            active_level = new_level

        if not np.isfinite(active_level):
            continue

        current_high = high.iloc[index]
        current_low = low.iloc[index]
        if np.isfinite(current_high) and np.isfinite(current_low) and current_low <= active_level <= current_high:
            active_level = np.nan
            continue

        result[index] = active_level

    return pd.Series(result, index=poc_series.index)


def add_auxiliary_asset_features(frame: pd.DataFrame, alias: str) -> pd.DataFrame:
    close_col = f"{alias}_close"
    open_col = f"{alias}_open"
    high_col = f"{alias}_high"
    low_col = f"{alias}_low"
    volume_col = f"{alias}_volume"
    required = {close_col, open_col, high_col, low_col, volume_col}
    if not required.issubset(frame.columns):
        return pd.DataFrame(index=frame.index)

    close = frame[close_col]
    high = frame[high_col]
    low = frame[low_col]
    volume = frame[volume_col]
    returns_1 = close.pct_change(1)
    returns_5 = close.pct_change(5)
    returns_20 = close.pct_change(20)
    feature_map: dict[str, pd.Series] = {
        f"{alias}_return_1": returns_1,
        f"{alias}_return_5": returns_5,
        f"{alias}_return_20": returns_20,
        f"{alias}_vol_20": returns_1.rolling(20).std(),
        f"{alias}_ma_20_ratio": close / close.rolling(20).mean() - 1.0,
        f"{alias}_z_20": compute_rolling_zscore(close, 20),
        f"{alias}_drawdown_60": compute_drawdown(close, 60),
        f"{alias}_range_pct": (high - low) / close,
        f"{alias}_rsi_14": compute_rsi(close, 14),
        f"{alias}_stoch_k_14": compute_stochastic_k(high, low, close, 14),
        f"{alias}_obv_z_30": compute_rolling_zscore(compute_obv(close, volume), 30),
    }

    if alias != "eth" and "eth_close" in frame.columns:
        eth_returns_5 = frame["eth_close"].pct_change(5)
        eth_returns_20 = frame["eth_close"].pct_change(20)
        eth_returns_1 = frame["eth_close"].pct_change(1)
        feature_map[f"eth_vs_{alias}_strength_5"] = eth_returns_5 - returns_5
        feature_map[f"eth_vs_{alias}_strength_20"] = eth_returns_20 - returns_20
        feature_map[f"eth_{alias}_corr_30"] = eth_returns_1.rolling(30).corr(returns_1)

    return pd.DataFrame(feature_map, index=frame.index)


def compute_future_window_extrema(series: pd.Series, horizon: int) -> tuple[pd.Series, pd.Series]:
    future = pd.to_numeric(series, errors="coerce").shift(-1)
    future_max = future.iloc[::-1].rolling(horizon, min_periods=horizon).max().iloc[::-1]
    future_min = future.iloc[::-1].rolling(horizon, min_periods=horizon).min().iloc[::-1]
    return future_max, future_min


def build_state_targets(close: pd.Series, horizon: int) -> pd.DataFrame:
    close = pd.to_numeric(close, errors="coerce")
    future_return = close.shift(-horizon) / close - 1.0
    future_high, future_low = compute_future_window_extrema(close, horizon)
    future_max_return = future_high / close - 1.0
    future_min_return = future_low / close - 1.0

    realized_vol = close.pct_change().rolling(30, min_periods=20).std()
    move_threshold = (realized_vol * np.sqrt(max(horizon, 1)) * 1.00).clip(
        lower=0.02 if horizon <= 7 else 0.04,
        upper=0.16 if horizon <= 7 else 0.26,
    )
    guard_threshold = (move_threshold * 0.80).clip(
        lower=0.015 if horizon <= 7 else 0.03,
        upper=0.13 if horizon <= 7 else 0.20,
    )

    lookback = int(np.clip(max(horizon * 3, 20), 20, 90))
    min_periods = max(10, lookback // 2)
    rolling_high = close.rolling(lookback, min_periods=min_periods).max()
    rolling_low = close.rolling(lookback, min_periods=min_periods).min()
    recent_drawdown = close / rolling_high - 1.0
    recent_rebound = close / rolling_low - 1.0

    uptrend = (
        (future_return >= move_threshold)
        & (future_min_return >= -guard_threshold)
    )
    downtrend = (
        (future_return <= -move_threshold)
        & (future_max_return <= guard_threshold)
    )
    regime = pd.Series(1, index=close.index, dtype=float)
    regime.loc[downtrend.fillna(False)] = 0.0
    regime.loc[uptrend.fillna(False)] = 2.0

    bottom_trigger = (move_threshold * 1.25).clip(
        lower=0.06 if horizon <= 7 else 0.08,
        upper=0.22,
    )
    top_trigger = (move_threshold * 1.25).clip(
        lower=0.06 if horizon <= 7 else 0.08,
        upper=0.22,
    )
    reversal_move = (move_threshold * 0.70).clip(
        lower=0.035 if horizon <= 7 else 0.05,
        upper=0.16,
    )
    rebound_cap = (move_threshold * 0.85).clip(
        lower=0.05 if horizon <= 7 else 0.06,
        upper=0.16,
    )
    near_high_guard = (guard_threshold * 1.25).clip(
        lower=0.03 if horizon <= 7 else 0.05,
        upper=0.16,
    )

    bottom_reversal = (
        (recent_drawdown <= -bottom_trigger)
        & (recent_rebound <= rebound_cap)
        & (future_return >= reversal_move)
        & (future_max_return >= reversal_move)
    )
    top_reversal = (
        (recent_rebound >= top_trigger)
        & (recent_drawdown >= -near_high_guard)
        & (future_return <= -reversal_move)
        & (future_min_return <= -reversal_move)
    )
    reversal_state = pd.Series(1, index=close.index, dtype=float)
    reversal_state.loc[top_reversal.fillna(False)] = 0.0
    reversal_state.loc[bottom_reversal.fillna(False)] = 2.0

    return pd.DataFrame(
        {
            "target_regime": regime,
            "target_reversal_state": reversal_state,
            "target_bottom_reversal": bottom_reversal.astype(float),
            "target_top_reversal": top_reversal.astype(float),
        },
        index=close.index,
    )


def build_features(
    data: pd.DataFrame,
    horizon: int,
    prediction_history: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    frame = data.copy()
    raw_columns = set(data.columns)
    # Phase 1.B fix: do NOT nullify entire vendor columns based on TODAY's staleness.
    # Historical rows had their own point-in-time freshness; wiping the whole column
    # back-projects the current state onto the past. Per-row staleness is already
    # exposed via the paired __age_days / __is_missing columns, and the Phase 1.2
    # bounded ffill ensures current-row values don't silently use quarter-old data.
    # Coverage filtering in select_usable_feature_columns will still drop columns
    # that have no usable signal in a given training fold.
    eth_returns_1 = frame["eth_close"].pct_change(1)
    eth_returns_5 = frame["eth_close"].pct_change(5)
    eth_returns_7 = frame["eth_close"].pct_change(7)
    eth_returns_14 = frame["eth_close"].pct_change(14)
    eth_returns_30 = frame["eth_close"].pct_change(30)
    eth_returns_60 = frame["eth_close"].pct_change(60)
    eth_returns_90 = frame["eth_close"].pct_change(90)
    eth_returns_120 = frame["eth_close"].pct_change(120)
    eth_returns_180 = frame["eth_close"].pct_change(180)
    eth_returns_365 = frame["eth_close"].pct_change(365)
    eth_ema_12 = frame["eth_close"].ewm(span=12, adjust=False).mean()
    eth_ema_26 = frame["eth_close"].ewm(span=26, adjust=False).mean()
    eth_macd = eth_ema_12 - eth_ema_26
    eth_macd_signal = eth_macd.ewm(span=9, adjust=False).mean()
    eth_stoch_k_14 = compute_stochastic_k(frame["eth_high"], frame["eth_low"], frame["eth_close"], 14)
    eth_bb_mean = frame["eth_close"].rolling(20).mean()
    eth_bb_std = frame["eth_close"].rolling(20).std()
    eth_obv = compute_obv(frame["eth_close"], frame["eth_volume"])
    eth_typical_price = (frame["eth_high"] + frame["eth_low"] + frame["eth_close"]) / 3.0
    eth_vwap_7 = compute_rolling_vwap(eth_typical_price, frame["eth_volume"], 7)
    eth_vwap_20 = (eth_typical_price * frame["eth_volume"]).rolling(20).sum() / frame["eth_volume"].rolling(20).sum()
    eth_vwap_30 = compute_rolling_vwap(eth_typical_price, frame["eth_volume"], 30)
    eth_vol_7 = eth_returns_1.rolling(7).std()
    eth_vol_14 = eth_returns_1.rolling(14).std()
    eth_vol_30 = eth_returns_1.rolling(30).std()
    eth_vol_20 = eth_returns_1.rolling(20).std()
    eth_vol_90 = eth_returns_1.rolling(90).std()
    eth_vol_120 = eth_returns_1.rolling(120).std()
    eth_vol_180 = eth_returns_1.rolling(180).std()
    eth_vol_365 = eth_returns_1.rolling(365).std()
    eth_range_pct = (frame["eth_high"] - frame["eth_low"]) / frame["eth_close"]
    eth_range_expansion_z_30 = compute_rolling_zscore(eth_range_pct, 30)
    eth_volume_z_30 = (
        (frame["eth_volume"] - frame["eth_volume"].rolling(30).mean())
        / frame["eth_volume"].rolling(30).std()
    )
    eth_volume_impulse_3_30 = frame["eth_volume"].rolling(3).mean() / frame["eth_volume"].rolling(30).mean() - 1.0
    eth_vol_expansion_7_30 = eth_vol_7 / eth_vol_30 - 1.0
    eth_vol_squeeze_20_90 = eth_vol_20 / eth_vol_90 - 1.0
    # Past-only proxy for "tail conditions are building"; it does not use the
    # future target and is meant to widen risk ranges before spike/crash events.
    eth_tail_event_pressure = (
        0.28 * compute_rolling_zscore(eth_returns_1.abs(), 30).clip(lower=0.0, upper=4.0)
        + 0.24 * eth_range_expansion_z_30.clip(lower=0.0, upper=4.0)
        + 0.20 * eth_volume_z_30.clip(lower=0.0, upper=4.0)
        + 0.16 * eth_vol_expansion_7_30.clip(lower=0.0, upper=2.5)
        + 0.12 * (-eth_vol_squeeze_20_90).clip(lower=0.0, upper=2.5)
    )
    eth_rsi_14 = compute_rsi(frame["eth_close"], 14)
    eth_rsi_28 = compute_rsi(frame["eth_close"], 28)
    eth_cmf_20 = compute_cmf(
        frame["eth_high"],
        frame["eth_low"],
        frame["eth_close"],
        frame["eth_volume"],
        20,
    )
    eth_adx_14, eth_plus_di_14, eth_minus_di_14 = compute_adx(
        frame["eth_high"],
        frame["eth_low"],
        frame["eth_close"],
        14,
    )
    eth_tenkan, eth_kijun, eth_senkou_a, eth_senkou_b = compute_ichimoku(
        frame["eth_high"],
        frame["eth_low"],
    )
    eth_profile_7 = compute_rolling_volume_profile(
        frame["eth_high"],
        frame["eth_low"],
        frame["eth_close"],
        frame["eth_volume"],
        7,
        bins=48,
    )
    eth_profile_30 = compute_rolling_volume_profile(
        frame["eth_high"],
        frame["eth_low"],
        frame["eth_close"],
        frame["eth_volume"],
        30,
        bins=64,
    )
    eth_naked_poc_7 = compute_naked_poc(frame["eth_high"], frame["eth_low"], eth_profile_7["poc"])
    eth_naked_poc_30 = compute_naked_poc(frame["eth_high"], frame["eth_low"], eth_profile_30["poc"])
    eth_up_streak = compute_boolean_streak(eth_returns_1 > 0.0)
    eth_down_streak = compute_boolean_streak(eth_returns_1 < 0.0)
    eth_up_day_ratio_7 = (eth_returns_1 > 0.0).rolling(7).mean()
    eth_up_day_ratio_14 = (eth_returns_1 > 0.0).rolling(14).mean()
    eth_low_20 = frame["eth_close"].rolling(20).min()
    eth_low_60 = frame["eth_close"].rolling(60).min()
    eth_low_180 = frame["eth_close"].rolling(180).min()
    eth_low_365 = frame["eth_close"].rolling(365).min()
    eth_high_20 = frame["eth_close"].rolling(20).max()
    eth_high_60 = frame["eth_close"].rolling(60).max()
    eth_high_180 = frame["eth_close"].rolling(180).max()
    eth_high_365 = frame["eth_close"].rolling(365).max()
    eth_cycle_position_180 = (frame["eth_close"] - eth_low_180) / (eth_high_180 - eth_low_180).replace(0.0, np.nan)
    eth_cycle_position_365 = (frame["eth_close"] - eth_low_365) / (eth_high_365 - eth_low_365).replace(0.0, np.nan)
    eth_macd_hist = eth_macd - eth_macd_signal
    eth_adx_trend_bias_14 = (eth_plus_di_14 - eth_minus_di_14) / (eth_plus_di_14 + eth_minus_di_14).replace(0.0, np.nan)

    feature_blocks: list[pd.DataFrame] = [
        pd.DataFrame(
            {
                "eth_return_1": eth_returns_1,
                "eth_return_2": frame["eth_close"].pct_change(2),
                "eth_return_3": frame["eth_close"].pct_change(3),
                "eth_return_5": eth_returns_5,
                "eth_return_7": eth_returns_7,
                "eth_return_14": eth_returns_14,
                "eth_return_21": frame["eth_close"].pct_change(21),
                "eth_return_30": eth_returns_30,
                "eth_return_60": eth_returns_60,
                "eth_return_90": eth_returns_90,
                "eth_return_120": eth_returns_120,
                "eth_return_180": eth_returns_180,
                "eth_return_365": eth_returns_365,
                "eth_log_return_1": np.log(frame["eth_close"]).diff(),
                "eth_close_lag_1": frame["eth_close"].shift(1),
                "eth_close_lag_3": frame["eth_close"].shift(3),
                "eth_close_lag_7": frame["eth_close"].shift(7),
                "eth_range_pct": eth_range_pct,
                "eth_close_open_pct": (frame["eth_close"] - frame["eth_open"]) / frame["eth_open"],
                "eth_up_streak": eth_up_streak,
                "eth_down_streak": eth_down_streak,
                "eth_up_day_ratio_7": eth_up_day_ratio_7,
                "eth_up_day_ratio_14": eth_up_day_ratio_14,
                "eth_rebound_from_20d_low": frame["eth_close"] / eth_low_20 - 1.0,
                "eth_rebound_from_60d_low": frame["eth_close"] / eth_low_60 - 1.0,
                "eth_rebound_from_180d_low": frame["eth_close"] / eth_low_180 - 1.0,
                "eth_rebound_from_365d_low": frame["eth_close"] / eth_low_365 - 1.0,
                "eth_distance_to_20d_high": frame["eth_close"] / eth_high_20 - 1.0,
                "eth_distance_to_60d_high": frame["eth_close"] / eth_high_60 - 1.0,
                "eth_distance_to_180d_high": frame["eth_close"] / eth_high_180 - 1.0,
                "eth_distance_to_365d_high": frame["eth_close"] / eth_high_365 - 1.0,
                "eth_ma_7_ratio": frame["eth_close"] / frame["eth_close"].rolling(7).mean() - 1.0,
                "eth_ma_14_ratio": frame["eth_close"] / frame["eth_close"].rolling(14).mean() - 1.0,
                "eth_ma_30_ratio": frame["eth_close"] / frame["eth_close"].rolling(30).mean() - 1.0,
                "eth_ma_90_ratio": frame["eth_close"] / frame["eth_close"].rolling(90).mean() - 1.0,
                "eth_ma_120_ratio": frame["eth_close"] / frame["eth_close"].rolling(120).mean() - 1.0,
                "eth_ma_180_ratio": frame["eth_close"] / frame["eth_close"].rolling(180).mean() - 1.0,
                "eth_ma_365_ratio": frame["eth_close"] / frame["eth_close"].rolling(365).mean() - 1.0,
                "eth_vol_7": eth_vol_7,
                "eth_vol_14": eth_vol_14,
                "eth_vol_30": eth_vol_30,
                "eth_vol_60": eth_returns_1.rolling(60).std(),
                "eth_vol_90": eth_vol_90,
                "eth_vol_120": eth_vol_120,
                "eth_vol_180": eth_vol_180,
                "eth_vol_365": eth_vol_365,
                "eth_vol_30_180_ratio": eth_vol_30 / eth_vol_180 - 1.0,
                "eth_vol_90_365_ratio": eth_vol_90 / eth_vol_365 - 1.0,
                "eth_downside_vol_14": compute_downside_volatility(eth_returns_1, 14),
                "eth_downside_vol_30": compute_downside_volatility(eth_returns_1, 30),
                "eth_vol_regime_7_30": eth_vol_expansion_7_30,
                "eth_vol_squeeze_20_90": eth_vol_squeeze_20_90,
                "eth_range_expansion_z_30": eth_range_expansion_z_30,
                "eth_volume_impulse_3_30": eth_volume_impulse_3_30,
                "eth_tail_event_pressure": eth_tail_event_pressure,
                "eth_volume_change_1": frame["eth_volume"].pct_change(1),
                "eth_volume_change_7": frame["eth_volume"].pct_change(7),
                "eth_volume_z_30": eth_volume_z_30,
                "eth_rsi_14": eth_rsi_14,
                "eth_rsi_28": eth_rsi_28,
                "eth_rsi_14_change_3": eth_rsi_14.diff(3),
                "eth_ema_12_ratio": eth_ema_12 / frame["eth_close"] - 1.0,
                "eth_ema_26_ratio": eth_ema_26 / frame["eth_close"] - 1.0,
                "eth_macd": eth_macd,
                "eth_macd_signal": eth_macd_signal,
                "eth_macd_hist": eth_macd_hist,
                "eth_macd_hist_change_3": eth_macd_hist.diff(3),
                "eth_atr_14_ratio": compute_atr(
                    frame["eth_high"],
                    frame["eth_low"],
                    frame["eth_close"],
                    14,
                ) / frame["eth_close"],
                "eth_drawdown_30": compute_drawdown(frame["eth_close"], 30),
                "eth_drawdown_90": compute_drawdown(frame["eth_close"], 90),
                "eth_drawdown_180": compute_drawdown(frame["eth_close"], 180),
                "eth_drawdown_365": compute_drawdown(frame["eth_close"], 365),
                "eth_close_z_20": compute_rolling_zscore(frame["eth_close"], 20),
                "eth_close_z_60": compute_rolling_zscore(frame["eth_close"], 60),
                "eth_close_z_180": compute_rolling_zscore(frame["eth_close"], 180),
                "eth_stoch_k_14": eth_stoch_k_14,
                "eth_stoch_d_3": eth_stoch_k_14.rolling(3).mean(),
                "eth_bollinger_width_20": (4.0 * eth_bb_std) / eth_bb_mean,
                "eth_bollinger_z_20": (frame["eth_close"] - eth_bb_mean) / eth_bb_std,
                "eth_obv_z_30": compute_rolling_zscore(eth_obv, 30),
                "eth_mfi_14": compute_mfi(
                    frame["eth_high"],
                    frame["eth_low"],
                    frame["eth_close"],
                    frame["eth_volume"],
                    14,
                ),
                "eth_cmf_20": eth_cmf_20,
                "eth_adx_14": eth_adx_14,
                "eth_adx_trend_bias_14": eth_adx_trend_bias_14,
                "eth_plus_di_14": eth_plus_di_14,
                "eth_minus_di_14": eth_minus_di_14,
                "eth_vwap_20_ratio": frame["eth_close"] / eth_vwap_20 - 1.0,
                "eth_vwap_7_ratio": frame["eth_close"] / eth_vwap_7 - 1.0,
                "eth_vwap_30_ratio": frame["eth_close"] / eth_vwap_30 - 1.0,
                "eth_vwap_7_30_spread": eth_vwap_7 / eth_vwap_30 - 1.0,
                "eth_poc_7_ratio": frame["eth_close"] / eth_profile_7["poc"] - 1.0,
                "eth_val_7_ratio": frame["eth_close"] / eth_profile_7["val"] - 1.0,
                "eth_vah_7_ratio": frame["eth_close"] / eth_profile_7["vah"] - 1.0,
                "eth_value_area_width_7": (eth_profile_7["vah"] - eth_profile_7["val"]) / frame["eth_close"],
                "eth_naked_poc_7_ratio": frame["eth_close"] / eth_naked_poc_7 - 1.0,
                "eth_poc_30_ratio": frame["eth_close"] / eth_profile_30["poc"] - 1.0,
                "eth_val_30_ratio": frame["eth_close"] / eth_profile_30["val"] - 1.0,
                "eth_vah_30_ratio": frame["eth_close"] / eth_profile_30["vah"] - 1.0,
                "eth_value_area_width_30": (eth_profile_30["vah"] - eth_profile_30["val"]) / frame["eth_close"],
                "eth_naked_poc_30_ratio": frame["eth_close"] / eth_naked_poc_30 - 1.0,
                "eth_ichimoku_tenkan_ratio": eth_tenkan / frame["eth_close"] - 1.0,
                "eth_ichimoku_kijun_ratio": eth_kijun / frame["eth_close"] - 1.0,
                "eth_ichimoku_cloud_mid_ratio": ((eth_senkou_a + eth_senkou_b) / 2.0) / frame["eth_close"] - 1.0,
                "eth_ichimoku_cloud_thickness_ratio": (eth_senkou_a - eth_senkou_b).abs() / frame["eth_close"],
                "eth_return_accel_7_30": eth_returns_7 - eth_returns_30,
                "eth_return_accel_30_180": eth_returns_30 - eth_returns_180,
                "eth_skew_30": eth_returns_1.rolling(30).skew(),
                "eth_skew_90": eth_returns_1.rolling(90).skew(),
                "eth_kurt_30": eth_returns_1.rolling(30).kurt(),
                "eth_kurt_90": eth_returns_1.rolling(90).kurt(),
                "eth_range_position_20": (
                    (frame["eth_close"] - frame["eth_low"].rolling(20).min())
                    / (frame["eth_high"].rolling(20).max() - frame["eth_low"].rolling(20).min())
                ),
                "eth_cycle_position_180": eth_cycle_position_180,
                "eth_cycle_position_180_centered": eth_cycle_position_180 - 0.5,
                "eth_cycle_position_365": eth_cycle_position_365,
                "eth_cycle_position_365_centered": eth_cycle_position_365 - 0.5,
            },
            index=frame.index,
        )
    ]

    if {"btc_close", "btc_open", "btc_high", "btc_low", "btc_volume"}.issubset(frame.columns):
        btc_returns_1 = frame["btc_close"].pct_change(1)
        btc_returns_5 = frame["btc_close"].pct_change(5)
        btc_returns_14 = frame["btc_close"].pct_change(14)
        btc_returns_20 = frame["btc_close"].pct_change(20)
        btc_returns_30 = frame["btc_close"].pct_change(30)
        btc_returns_60 = frame["btc_close"].pct_change(60)
        btc_returns_90 = frame["btc_close"].pct_change(90)
        btc_returns_180 = frame["btc_close"].pct_change(180)
        btc_stoch_k_14 = compute_stochastic_k(frame["btc_high"], frame["btc_low"], frame["btc_close"], 14)
        btc_obv = compute_obv(frame["btc_close"], frame["btc_volume"])
        btc_vol_30 = btc_returns_1.rolling(30).std()
        btc_vol_90 = btc_returns_1.rolling(90).std()
        feature_blocks.append(
            pd.DataFrame(
                {
                    "btc_return_1": btc_returns_1,
                    "btc_return_3": frame["btc_close"].pct_change(3),
                    "btc_return_5": btc_returns_5,
                    "btc_return_7": frame["btc_close"].pct_change(7),
                    "btc_return_14": btc_returns_14,
                    "btc_return_20": btc_returns_20,
                    "btc_return_30": btc_returns_30,
                    "btc_return_60": btc_returns_60,
                    "btc_return_90": btc_returns_90,
                    "btc_return_180": btc_returns_180,
                    "btc_vol_14": btc_returns_1.rolling(14).std(),
                    "btc_vol_20": btc_returns_1.rolling(20).std(),
                    "btc_vol_30": btc_vol_30,
                    "btc_vol_90": btc_vol_90,
                    "btc_rsi_14": compute_rsi(frame["btc_close"], 14),
                    "btc_range_pct": (frame["btc_high"] - frame["btc_low"]) / frame["btc_close"],
                    "btc_stoch_k_14": btc_stoch_k_14,
                    "btc_obv_z_30": compute_rolling_zscore(btc_obv, 30),
                    "btc_ma_20_ratio": frame["btc_close"] / frame["btc_close"].rolling(20).mean() - 1.0,
                    "btc_ma_30_ratio": frame["btc_close"] / frame["btc_close"].rolling(30).mean() - 1.0,
                    "btc_ma_90_ratio": frame["btc_close"] / frame["btc_close"].rolling(90).mean() - 1.0,
                    "btc_ma_180_ratio": frame["btc_close"] / frame["btc_close"].rolling(180).mean() - 1.0,
                    "btc_drawdown_60": compute_drawdown(frame["btc_close"], 60),
                    "btc_drawdown_180": compute_drawdown(frame["btc_close"], 180),
                    "btc_z_20": compute_rolling_zscore(frame["btc_close"], 20),
                    "btc_z_90": compute_rolling_zscore(frame["btc_close"], 90),
                    "eth_btc_ratio": frame["eth_close"] / frame["btc_close"],
                    "eth_btc_corr_14": eth_returns_1.rolling(14).corr(btc_returns_1),
                    "eth_btc_corr_30": eth_returns_1.rolling(30).corr(btc_returns_1),
                    "eth_btc_corr_90": eth_returns_1.rolling(90).corr(btc_returns_1),
                    "eth_beta_btc_30": compute_rolling_beta(eth_returns_1, btc_returns_1, 30),
                    "eth_beta_btc_90": compute_rolling_beta(eth_returns_1, btc_returns_1, 90),
                    "eth_vs_btc_strength_5": eth_returns_5 - btc_returns_5,
                    "eth_vs_btc_strength_14": eth_returns_14 - btc_returns_14,
                    "eth_vs_btc_strength_20": frame["eth_close"].pct_change(20) - btc_returns_20,
                    "eth_vs_btc_strength_30": eth_returns_30 - btc_returns_30,
                    "eth_vs_btc_strength_60": eth_returns_60 - btc_returns_60,
                    "eth_vs_btc_strength_90": eth_returns_90 - btc_returns_90,
                    "eth_vs_btc_strength_180": eth_returns_180 - btc_returns_180,
                    "eth_vs_btc_vol_ratio": feature_blocks[0]["eth_vol_30"] / btc_vol_30,
                    "eth_vs_btc_vol_ratio_90": eth_vol_90 / btc_vol_90,
                },
                index=frame.index,
            )
        )

    asset_aliases = sorted({column[:-6] for column in raw_columns if column.endswith("_close")})
    for alias in asset_aliases:
        if alias not in {"eth", "btc"}:
            auxiliary = add_auxiliary_asset_features(frame, alias)
            if not auxiliary.empty:
                feature_blocks.append(auxiliary)

    weekday = frame.index.dayofweek
    month = frame.index.month
    day_of_month = frame.index.day
    calendar_features = pd.DataFrame(
        {
            "dow_sin": np.sin(2 * np.pi * weekday / 7),
            "dow_cos": np.cos(2 * np.pi * weekday / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            "dom_sin": np.sin(2 * np.pi * day_of_month / 31),
            "dom_cos": np.cos(2 * np.pi * day_of_month / 31),
        },
        index=frame.index,
    )
    feature_blocks.append(calendar_features)

    regime_feature_map: dict[str, pd.Series] = {}
    if {"spy_return_20", "qqq_return_20"}.issubset(frame.columns):
        regime_feature_map["equity_risk_on_20"] = 0.5 * (frame["spy_return_20"] + frame["qqq_return_20"])
    if {"gold_return_20", "oil_return_20"}.issubset(frame.columns):
        regime_feature_map["commodity_impulse_20"] = frame["gold_return_20"] + frame["oil_return_20"]
    if {"dxy_return_20", "tnx_return_20"}.issubset(frame.columns):
        regime_feature_map["macro_tightening_20"] = frame["dxy_return_20"] + frame["tnx_return_20"]
    if {"vix_return_5", "spy_return_5"}.issubset(frame.columns):
        regime_feature_map["risk_aversion_spread_5"] = frame["vix_return_5"] - frame["spy_return_5"]
    if {"sol_return_20", "bnb_return_20"}.issubset(frame.columns):
        regime_feature_map["alt_strength_20"] = 0.5 * (frame["sol_return_20"] + frame["bnb_return_20"])
    if "sentiment_fear_greed" in frame.columns:
        regime_feature_map["sentiment_fear_greed_centered"] = (frame["sentiment_fear_greed"] - 50.0) / 50.0
    if {"fred_fed_balance_sheet", "fred_tga", "fred_rrp"}.issubset(frame.columns):
        fred_net_liquidity = frame["fred_fed_balance_sheet"] - frame["fred_tga"] - frame["fred_rrp"]
        regime_feature_map["fred_net_liquidity_raw"] = fred_net_liquidity
        regime_feature_map["fred_net_liquidity_z_90"] = compute_rolling_zscore(fred_net_liquidity, 90)
    if {"cg_eth_market_cap_usd", "cg_btc_market_cap_usd"}.issubset(frame.columns):
        regime_feature_map["cg_eth_btc_market_cap_ratio"] = frame["cg_eth_market_cap_usd"] / frame["cg_btc_market_cap_usd"]
    if {"cg_global_total_market_cap_usd", "cg_eth_market_cap_usd"}.issubset(frame.columns):
        regime_feature_map["cg_eth_share_of_crypto"] = frame["cg_eth_market_cap_usd"] / frame["cg_global_total_market_cap_usd"]
    if {"cg_global_total_market_cap_usd", "cg_global_total_volume_usd"}.issubset(frame.columns):
        regime_feature_map["cg_global_turnover_ratio"] = frame["cg_global_total_volume_usd"] / frame["cg_global_total_market_cap_usd"]
    # Phase 3: domain-specific derivatives-regime features. The generic vendor
    # transform already emits z-scores for raw deribit columns, but these named
    # features capture well-known regimes (crowded funding, vol expansion,
    # hedging pressure, contango/backwardation) that the model otherwise has
    # to re-learn combinatorially.
    if "deribit_eth_funding_interest_8h" in frame.columns:
        regime_feature_map["deribit_funding_8h_z_90"] = compute_rolling_zscore(
            pd.to_numeric(frame["deribit_eth_funding_interest_8h"], errors="coerce"), 90,
        )
    if "deribit_eth_option_mark_iv_mean" in frame.columns:
        regime_feature_map["deribit_option_iv_z_90"] = compute_rolling_zscore(
            pd.to_numeric(frame["deribit_eth_option_mark_iv_mean"], errors="coerce"), 90,
        )
    if "deribit_eth_option_put_call_oi_ratio" in frame.columns:
        regime_feature_map["deribit_pcr_z_90"] = compute_rolling_zscore(
            pd.to_numeric(frame["deribit_eth_option_put_call_oi_ratio"], errors="coerce"), 90,
        )
    if {"deribit_eth_future_mark_price_mean", "eth_close"}.issubset(frame.columns):
        futures_mark = pd.to_numeric(frame["deribit_eth_future_mark_price_mean"], errors="coerce")
        spot = pd.to_numeric(frame["eth_close"], errors="coerce").replace(0.0, np.nan)
        basis_ratio = futures_mark / spot - 1.0
        regime_feature_map["deribit_futures_basis_ratio"] = basis_ratio
        regime_feature_map["deribit_futures_basis_z_30"] = compute_rolling_zscore(basis_ratio, 30)
    if "deribit_eth_future_open_interest_sum" in frame.columns:
        oi = pd.to_numeric(frame["deribit_eth_future_open_interest_sum"], errors="coerce").replace(0.0, np.nan)
        regime_feature_map["deribit_future_oi_change_7"] = oi.pct_change(7, fill_method=None)
    # Phase 5 production (leave-one-out gated re-add of the Phase 3B macro block):
    # Four of six Phase 3B composites were marked CULPRIT by the Phase 5 LOO
    # (fred_yield_curve_change_20, fred_real_yield_10y_z_90,
    # fred_credit_stress_change_20, macro_asymmetry_20). They remain removed —
    # all four are derivatives / z-scores / asymmetries of slow FRED series and
    # the LOO shows differencing monthly-cadence macro data amplifies sampling
    # noise that hurts h=30 brier / AUC.
    #
    # Two composites were marked SAFE and are re-added here as raw levels:
    #   - fred_real_yield_10y = treasury_10y - breakeven_10y
    #   - fred_credit_stress  = 0.5 * (hy_oas + bbb_oas)
    # Combined Phase 5 freeze (2026-04-20) vs Phase 4B baseline:
    #   h=7  cls brier 0.217 -> 0.206, AUC 0.669 -> 0.713  (meaningful lift)
    #   h=30 flat within tolerance (no regression).
    # See tests/phase0/freeze_phase5_production.py + phase5_production_metrics.json.
    # If adding more macro features in the future, run the same leave-one-out
    # gate before merging.
    if {"fred_treasury_10y", "fred_breakeven_10y"}.issubset(frame.columns):
        real_yield_10y = (
            pd.to_numeric(frame["fred_treasury_10y"], errors="coerce")
            - pd.to_numeric(frame["fred_breakeven_10y"], errors="coerce")
        )
        regime_feature_map["fred_real_yield_10y"] = real_yield_10y
    if {"fred_hy_oas", "fred_bbb_oas"}.issubset(frame.columns):
        credit_stress = 0.5 * (
            pd.to_numeric(frame["fred_hy_oas"], errors="coerce")
            + pd.to_numeric(frame["fred_bbb_oas"], errors="coerce")
        )
        regime_feature_map["fred_credit_stress"] = credit_stress
    if regime_feature_map:
        feature_blocks.append(pd.DataFrame(regime_feature_map, index=frame.index))

    external_feature_map: dict[str, pd.Series] = {}
    excluded_short_history_columns = short_history_vendor_columns(frame)
    # Phase 1.B fix: do not exclude columns that happen to be stale at the
    # current tail. A stale vendor today still carries legitimate historical
    # signal, and the resulting __age_days / __is_missing features already tell
    # the model how fresh each row's value is.
    external_columns = [
        column
        for column in raw_columns
        if column.startswith(GENERIC_VENDOR_PREFIXES) and pd.api.types.is_numeric_dtype(frame[column])
    ]
    for column in external_columns:
        if column in excluded_short_history_columns:
            continue
        series = pd.to_numeric(frame[column], errors="coerce")
        asof_series, missing_flag, age_days, update_interval_days = prepare_external_source_series(series)
        safe_series = asof_series.replace(0.0, np.nan)
        slow_moving = bool(update_interval_days >= 5.0)

        external_feature_map[f"{column}__asof"] = asof_series
        external_feature_map[f"{column}__is_missing"] = missing_flag
        external_feature_map[f"{column}__age_days"] = age_days
        external_feature_map[f"{column}_diff_7"] = asof_series.diff(7)
        external_feature_map[f"{column}_pct_30"] = safe_series.pct_change(30, fill_method=None)
        external_feature_map[f"{column}_z_30"] = compute_rolling_zscore(asof_series, 30)
        external_feature_map[f"{column}_z_90"] = compute_rolling_zscore(asof_series, 90)
        external_feature_map[f"{column}_ma_30_ratio"] = asof_series / asof_series.rolling(30).mean() - 1.0

        if not slow_moving:
            external_feature_map[f"{column}_diff_1"] = asof_series.diff(1)
            external_feature_map[f"{column}_pct_1"] = safe_series.pct_change(1, fill_method=None)
            external_feature_map[f"{column}_pct_7"] = safe_series.pct_change(7, fill_method=None)
    if external_feature_map:
        feature_blocks.append(pd.DataFrame(external_feature_map, index=frame.index))

    history_features = build_prediction_history_features(frame.index, prediction_history)
    if not history_features.empty:
        feature_blocks.append(history_features)

    target_block = pd.concat(
        [
            pd.DataFrame(
                {
                    "target_return": frame["eth_close"].shift(-horizon) / frame["eth_close"] - 1.0,
                    "target_close": frame["eth_close"].shift(-horizon),
                },
                index=frame.index,
            ),
            build_state_targets(frame["eth_close"], horizon=horizon),
        ],
        axis=1,
    )

    frame = pd.concat([frame, *feature_blocks, target_block], axis=1).copy()
    if frame.columns.has_duplicates:
        frame = frame.loc[:, ~frame.columns.duplicated(keep="first")].copy()
    frame = frame.replace([np.inf, -np.inf], np.nan)

    feature_columns = []
    for column in frame.columns:
        if column in {
            "target_return",
            "target_close",
            "target_regime",
            "target_reversal_state",
            "target_bottom_reversal",
            "target_top_reversal",
            "eth_close",
        } or column in raw_columns:
            continue
        column_data = frame.loc[:, column]
        if hasattr(column_data, "to_numpy") and column_data.notna().to_numpy().any():
            feature_columns.append(column)
    feature_columns = list(dict.fromkeys(feature_columns))

    return frame, feature_columns


def catboost_regressor_params(horizon: int | None = None) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "MAE",
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "boosting_type": "Ordered",
        "has_time": True,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if horizon is not None and horizon >= 30:
        params.update(
            {
                "iterations": 650,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.85,
            }
        )
    return params


def catboost_classifier_params(horizon: int | None = None, multiclass: bool = False) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "MultiClass" if multiclass else "Logloss",
        "iterations": 450,
        "depth": 6,
        "learning_rate": 0.04,
        "l2_leaf_reg": 5.0,
        "random_strength": 1.0,
        "boosting_type": "Ordered",
        "has_time": True,
        "random_seed": 42,
        "verbose": False,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if horizon is not None and horizon >= 30:
        params.update(
            {
                "iterations": 650,
                "depth": 5,
                "learning_rate": 0.03,
                "l2_leaf_reg": 8.0,
                "random_strength": 1.5,
                "bootstrap_type": "Bernoulli",
                "subsample": 0.85,
            }
        )
    if not multiclass and horizon is not None and horizon >= 30:
        params["auto_class_weights"] = "Balanced"
    return params


def make_models(horizon: int | None = None) -> dict[str, Any]:
    fast_mode = runtime_fast_mode_enabled()
    models: dict[str, Any] = {
        "ridge": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=1.0)),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=200,
                        max_depth=10,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=10,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "hist_gbr": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        loss="absolute_error",
                        learning_rate=0.05,
                        max_depth=5,
                        max_iter=250,
                        min_samples_leaf=12,
                        random_state=42,
                    ),
                ),
            ]
        ),
        "knn_regressor": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsRegressor(
                        n_neighbors=15,
                        weights="distance",
                    ),
                ),
            ]
        ),
    }
    if not fast_mode:
        models["mlp_regressor"] = Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPRegressor(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
    if (not fast_mode) and CatBoostRegressor is not None:
        models["catboost_regressor"] = Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    CatBoostRegressor(**catboost_regressor_params(horizon=horizon)),
                ),
            ]
        )
    return models


def make_classification_models(horizon: int | None = None) -> dict[str, Any]:
    fast_mode = runtime_fast_mode_enabled()
    models: dict[str, Any] = {
        "logistic": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "extra_trees_clf": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        max_depth=8,
                        min_samples_leaf=4,
                        max_features="sqrt",
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "random_forest_clf": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=8,
                        min_samples_leaf=5,
                        max_features="sqrt",
                        class_weight="balanced_subsample",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "knn_clf": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=15,
                        weights="distance",
                    ),
                ),
            ]
        ),
        "hist_gbc": Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        learning_rate=0.05,
                        max_depth=5,
                        max_iter=250,
                        min_samples_leaf=12,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    if not fast_mode:
        models["mlp_clf"] = Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        activation="relu",
                        alpha=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=300,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
    if (not fast_mode) and CatBoostClassifier is not None:
        models["catboost_clf"] = Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    CatBoostClassifier(**catboost_classifier_params(horizon=horizon)),
                ),
            ]
        )
    return models


def make_state_classification_models(horizon: int | None = None) -> dict[str, Any]:
    models = make_classification_models(horizon=horizon)
    if (not runtime_fast_mode_enabled()) and CatBoostClassifier is not None:
        models["catboost_clf"] = Pipeline(
            steps=[
                ("imputer", _make_median_imputer()),
                (
                    "model",
                    CatBoostClassifier(**catboost_classifier_params(horizon=horizon, multiclass=True)),
                ),
            ]
        )
    return models


def fit_calibrated_classifier(
    model_template: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    min_calibration_rows: int = DEFAULT_CALIBRATION_MIN_ROWS,
    sample_weight: pd.Series | np.ndarray | list[float] | None = None,
) -> CalibratedProbabilityModel:
    y_train = y_train.astype(int)
    aligned_sample_weight = align_sample_weight(sample_weight, X_train.index)
    base_model = clone(model_template)
    if y_train.nunique() < 2:
        fit_model_with_optional_sample_weight(base_model, X_train, y_train, aligned_sample_weight)
        return CalibratedProbabilityModel(base_model)

    minimum_rows = max(min_calibration_rows + 80, 160)
    if len(y_train) < minimum_rows:
        fit_model_with_optional_sample_weight(base_model, X_train, y_train, aligned_sample_weight)
        return CalibratedProbabilityModel(base_model)

    calibration_rows = min(
        max(min_calibration_rows, len(y_train) // 5),
        max(len(y_train) - 80, 0),
    )
    if calibration_rows <= 0 or len(y_train) - calibration_rows < 80:
        base_model.fit(X_train, y_train)
        return CalibratedProbabilityModel(base_model)

    X_fit = X_train.iloc[:-calibration_rows]
    y_fit = y_train.iloc[:-calibration_rows]
    X_calibration = X_train.iloc[-calibration_rows:]
    y_calibration = y_train.iloc[-calibration_rows:]
    fit_sample_weight = aligned_sample_weight.iloc[:-calibration_rows] if aligned_sample_weight is not None else None
    calibration_sample_weight = (
        aligned_sample_weight.iloc[-calibration_rows:] if aligned_sample_weight is not None else None
    )
    if y_fit.nunique() < 2 or y_calibration.nunique() < 2:
        fit_model_with_optional_sample_weight(base_model, X_train, y_train, aligned_sample_weight)
        return CalibratedProbabilityModel(base_model)

    fit_model_with_optional_sample_weight(base_model, X_fit, y_fit, fit_sample_weight)
    raw_probabilities = np.clip(base_model.predict_proba(X_calibration)[:, 1], 1e-6, 1.0 - 1e-6)
    calibrator, calibration_method = fit_probability_calibrator(
        raw_probabilities,
        y_calibration,
        sample_weight=calibration_sample_weight,
    )
    if calibrator is None:
        refit_model = clone(model_template)
        fit_model_with_optional_sample_weight(refit_model, X_train, y_train, aligned_sample_weight)
        return CalibratedProbabilityModel(refit_model)

    refit_model = clone(model_template)
    fit_model_with_optional_sample_weight(refit_model, X_train, y_train, aligned_sample_weight)
    return CalibratedProbabilityModel(
        refit_model,
        calibrator=calibrator,
        calibration_rows=calibration_rows,
        calibration_method=calibration_method,
    )


def safe_time_series_split(
    n_samples: int,
    n_splits: int,
    test_size: int,
    gap: int,
) -> TimeSeriesSplit:
    adjusted_splits = max(2, min(n_splits, n_samples - 1))
    max_test_size = max(1, (n_samples - gap - 1) // adjusted_splits)
    adjusted_test_size = max(1, min(test_size, max_test_size))

    try:
        return TimeSeriesSplit(
            n_splits=adjusted_splits,
            test_size=adjusted_test_size,
            gap=gap,
        )
    except TypeError:
        return TimeSeriesSplit(n_splits=adjusted_splits)


def purged_time_series_split(
    n_samples: int,
    n_splits: int,
    test_size: int,
    gap: int,
    embargo: int = 0,
) -> TimeSeriesSplit:
    """Walk-forward splitter with explicit purge + embargo (López de Prado).

    The purge is supplied by ``gap`` — it removes the last ``gap`` rows from
    each train fold and shifts the test start by ``gap``, so training labels
    (which use ``close.shift(-horizon)``) never reach into the test window when
    ``gap >= horizon``.

    The embargo adds a further buffer by effectively inflating the gap by
    ``embargo`` rows. This protects against feature-side spillover when long
    rolling windows at the last train row include data adjacent to the test
    fold. Setting ``embargo = 0`` yields behaviour identical to
    ``safe_time_series_split``.
    """
    total_gap = max(0, int(gap)) + max(0, int(embargo))
    return safe_time_series_split(
        n_samples=n_samples,
        n_splits=n_splits,
        test_size=test_size,
        gap=total_gap,
    )


def format_timestamp(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%Y-%m-%d %H:%M:%S")


def fetch_realtime_reference_price(symbol: str = DEFAULT_TICKERS["eth"]) -> PriceReference | None:
    try:
        ticker = yf.Ticker(symbol)
    except Exception:
        return None

    try:
        intraday = ticker.history(period="2d", interval="1m", auto_adjust=False, prepost=True)
        if not intraday.empty and "Close" in intraday.columns:
            close = pd.to_numeric(intraday["Close"], errors="coerce").dropna()
            if not close.empty:
                return PriceReference(
                    price=float(close.iloc[-1]),
                    timestamp=format_timestamp(close.index[-1]),
                    source="yfinance_history_1m",
                )
    except Exception:
        pass

    try:
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info is not None:
            for key in ("lastPrice", "regularMarketPrice", "last_price", "regular_market_price"):
                value = fast_info.get(key) if hasattr(fast_info, "get") else getattr(fast_info, key, None)
                if value is not None and np.isfinite(value) and float(value) > 0.0:
                    return PriceReference(
                        price=float(value),
                        timestamp="",
                        source=f"yfinance_fast_info:{key}",
                    )
    except Exception:
        pass

    return None


def resolve_price_reference(
    model_input_close: float,
    model_input_timestamp: Any,
    prefer_realtime: bool = True,
    symbol: str = DEFAULT_TICKERS["eth"],
) -> PriceReference:
    if prefer_realtime:
        live_reference = fetch_realtime_reference_price(symbol=symbol)
        if live_reference is not None and np.isfinite(live_reference.price) and live_reference.price > 0.0:
            if np.isfinite(model_input_close) and model_input_close > 0.0:
                live_gap_pct = abs(float(live_reference.price / model_input_close - 1.0))
                if (
                    str(live_reference.source).startswith("yfinance_fast_info:")
                    and live_gap_pct > DEFAULT_REFERENCE_FAST_INFO_MAX_GAP_PCT
                ):
                    return PriceReference(
                        price=float(model_input_close),
                        timestamp=format_timestamp(model_input_timestamp),
                        source=f"daily_feature_close_gap_guard({live_reference.source})",
                    )
            return live_reference
    return PriceReference(
        price=float(model_input_close),
        timestamp=format_timestamp(model_input_timestamp),
        source="daily_feature_close",
    )


def build_shared_price_reference(
    market_data: pd.DataFrame,
    prefer_realtime: bool = True,
) -> PriceReference:
    eth_close = (
        pd.to_numeric(market_data["eth_close"], errors="coerce")
        if "eth_close" in market_data.columns
        else pd.Series(dtype=float)
    ).dropna()
    if eth_close.empty:
        return PriceReference(
            price=float("nan"),
            timestamp="",
            source="daily_feature_close",
        )
    return resolve_price_reference(
        model_input_close=float(eth_close.iloc[-1]),
        model_input_timestamp=eth_close.index[-1],
        prefer_realtime=prefer_realtime,
    )


def get_final_estimator(model: Any) -> Any:
    if isinstance(model, CalibratedProbabilityModel):
        return get_final_estimator(model.base_estimator)
    if isinstance(model, Pipeline):
        return model.steps[-1][1]
    return model


def transform_feature_matrix(
    model: Any,
    X: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    transformed: Any = X.copy()
    if isinstance(model, CalibratedProbabilityModel):
        return transform_feature_matrix(model.base_estimator, X, feature_columns)
    if isinstance(model, Pipeline):
        for _, step in model.steps[:-1]:
            transformed = step.transform(transformed)
    if isinstance(transformed, pd.DataFrame):
        output = transformed.copy()
        output.index = X.index
        if len(output.columns) == len(feature_columns):
            output.columns = feature_columns
        return output
    return pd.DataFrame(transformed, index=X.index, columns=feature_columns)


def build_notes_frame(notes: list[dict[str, Any]]) -> pd.DataFrame:
    if not notes:
        return pd.DataFrame(columns=["analysis", "status", "detail"])
    return pd.DataFrame(notes)


def build_data_window_summary(
    raw_data: pd.DataFrame,
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    candidate_feature_count: int,
    interval: str,
    horizon: int,
    cv_splits: int,
    cv_test_size: int,
    training_window_years: float | None = None,
    sample_weight_half_life_years: float | None = None,
    sample_weights: pd.Series | None = None,
    full_training_rows: int | None = None,
) -> pd.DataFrame:
    splitter = safe_time_series_split(
        len(training_dataset),
        n_splits=cv_splits,
        test_size=cv_test_size,
        gap=horizon,
    )
    splits = list(splitter.split(training_dataset[feature_columns]))
    train_sizes = [len(train_idx) for train_idx, _ in splits]
    test_sizes = [len(test_idx) for _, test_idx in splits]
    practical_min_rows = 730 if horizon <= 7 else 1095
    preferred_rows = 1095 if horizon <= 7 else 1825

    raw_start = raw_data.index.min() if not raw_data.empty else None
    raw_end = raw_data.index.max() if not raw_data.empty else None
    train_start = training_dataset.index.min() if not training_dataset.empty else None
    train_end = training_dataset.index.max() if not training_dataset.empty else None
    latest_prediction_time = prediction_frame.index.max() if not prediction_frame.empty else None
    prediction_target_time = (
        pd.Timestamp(latest_prediction_time) + interval_to_offset(interval) * horizon
        if latest_prediction_time is not None
        else None
    )

    raw_span_days = int((pd.Timestamp(raw_end) - pd.Timestamp(raw_start)).days + 1) if raw_start is not None else 0
    training_span_days = (
        int((pd.Timestamp(train_end) - pd.Timestamp(train_start)).days + 1)
        if train_start is not None and train_end is not None
        else 0
    )
    aligned_sample_weights = align_sample_weight(sample_weights, training_dataset.index)
    original_training_rows = int(full_training_rows) if full_training_rows is not None else int(len(training_dataset))

    return pd.DataFrame(
        [
            {
                "raw_rows": int(len(raw_data)),
                "raw_start": format_timestamp(raw_start),
                "raw_end": format_timestamp(raw_end),
                "raw_span_days": raw_span_days,
                "feature_rows": int(len(prediction_frame)),
                "full_training_rows": original_training_rows,
                "training_rows": int(len(training_dataset)),
                "training_window_rows_dropped": int(max(original_training_rows - len(training_dataset), 0)),
                "rows_lost_to_target_shift": int(max(len(prediction_frame) - len(training_dataset), 0)),
                "candidate_feature_count": int(candidate_feature_count),
                "feature_count": int(len(feature_columns)),
                "dropped_feature_count": int(max(candidate_feature_count - len(feature_columns), 0)),
                "training_start": format_timestamp(train_start),
                "training_end": format_timestamp(train_end),
                "training_span_days": training_span_days,
                "training_window_years_applied": round(float(training_window_years), 2)
                if training_window_years is not None
                else float("nan"),
                "practical_min_rows": int(practical_min_rows),
                "preferred_rows": int(preferred_rows),
                "practical_min_years": round(practical_min_rows / periods_per_year(interval), 2),
                "preferred_years": round(preferred_rows / periods_per_year(interval), 2),
                "meets_practical_minimum": bool(len(training_dataset) >= practical_min_rows),
                "meets_preferred_length": bool(len(training_dataset) >= preferred_rows),
                "sample_weight_half_life_years": round(float(sample_weight_half_life_years), 2)
                if sample_weight_half_life_years is not None
                else float("nan"),
                "sample_weight_min": float(aligned_sample_weights.min())
                if aligned_sample_weights is not None and not aligned_sample_weights.empty
                else float("nan"),
                "sample_weight_max": float(aligned_sample_weights.max())
                if aligned_sample_weights is not None and not aligned_sample_weights.empty
                else float("nan"),
                "requested_cv_splits": int(cv_splits),
                "effective_cv_splits": int(len(splits)),
                "requested_cv_test_size": int(cv_test_size),
                "min_train_rows_per_fold": int(min(train_sizes)) if train_sizes else 0,
                "max_train_rows_per_fold": int(max(train_sizes)) if train_sizes else 0,
                "min_validation_rows_per_fold": int(min(test_sizes)) if test_sizes else 0,
                "max_validation_rows_per_fold": int(max(test_sizes)) if test_sizes else 0,
                "gap_rows_between_train_and_validation": int(horizon),
                "latest_prediction_input_timestamp": format_timestamp(latest_prediction_time),
                "prediction_target_timestamp": format_timestamp(prediction_target_time),
            }
        ]
    )


def build_native_feature_importance(
    model: Any,
    feature_columns: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    estimator = get_final_estimator(model)
    if hasattr(estimator, "feature_importances_"):
        signed_values = np.asarray(estimator.feature_importances_, dtype=float)
        abs_values = np.abs(signed_values)
        method = "native_feature_importance"
    elif hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 2:
            signed_values = np.mean(coefficients, axis=0)
            abs_values = np.mean(np.abs(coefficients), axis=0)
        else:
            signed_values = coefficients.ravel()
            abs_values = np.abs(signed_values)
        method = "absolute_coefficient"
    else:
        return (
            pd.DataFrame(columns=["model", "feature", "importance", "importance_abs", "rank", "method"]),
            {
                "analysis": "native_feature_importance",
                "status": "skipped",
                "detail": f"{model_name} does not expose native importances or coefficients.",
            },
        )

    report = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "importance": signed_values,
            "importance_abs": abs_values,
            "method": method,
        }
    ).sort_values(["importance_abs", "feature"], ascending=[False, True]).reset_index(drop=True)
    report["rank"] = np.arange(1, len(report) + 1)
    return (
        report,
        {
            "analysis": "native_feature_importance",
            "status": "ok",
            "detail": f"Computed from {method} for {model_name}.",
        },
    )


def select_explainability_split(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    preferred_test_size: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    if len(dataset) < max(80, preferred_test_size * 2 + horizon + 5):
        return None

    analysis_test_size = min(
        max(preferred_test_size * 2, horizon * 2, 30),
        max(30, len(dataset) // 5),
    )
    splitter = safe_time_series_split(
        len(dataset),
        n_splits=2,
        test_size=analysis_test_size,
        gap=horizon,
    )
    splits = list(splitter.split(dataset[feature_columns]))
    if not splits:
        return None
    return splits[-1]


def build_permutation_importance_report(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_template: Any,
    model_name: str,
    horizon: int,
    cv_test_size: int,
    task: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    split_indices = select_explainability_split(
        dataset=dataset,
        feature_columns=feature_columns,
        horizon=horizon,
        preferred_test_size=cv_test_size,
    )
    if split_indices is None:
        return (
            pd.DataFrame(
                columns=[
                    "model",
                    "feature",
                    "importance_mean",
                    "importance_std",
                    "rank",
                    "train_rows",
                    "eval_rows",
                    "scoring",
                ]
            ),
            {
                "analysis": "permutation_importance",
                "status": "skipped",
                "detail": "Not enough rows to carve out a recent holdout window for permutation importance.",
            },
        )

    train_idx, test_idx = split_indices
    X = dataset[feature_columns]
    if task == "regression":
        y = dataset["target_return"]
        scoring = "neg_root_mean_squared_error"
    else:
        y = (dataset["target_return"] > 0).astype(int)
        scoring = "roc_auc" if y.iloc[test_idx].nunique() > 1 else "balanced_accuracy"

    if task == "classification" and y.iloc[train_idx].nunique() < 2:
        return (
            pd.DataFrame(
                columns=[
                    "model",
                    "feature",
                    "importance_mean",
                    "importance_std",
                    "rank",
                    "train_rows",
                    "eval_rows",
                    "scoring",
                ]
            ),
            {
                "analysis": "permutation_importance",
                "status": "skipped",
                "detail": f"{model_name} permutation importance skipped because the recent train window had one class.",
            },
        )

    if task == "classification":
        model = fit_calibrated_classifier(
            model_template,
            X.iloc[train_idx],
            y.iloc[train_idx],
        )
    else:
        model = clone(model_template)
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
    result = permutation_importance(
        model,
        X.iloc[test_idx],
        y.iloc[test_idx],
        n_repeats=8,
        random_state=42,
        scoring=scoring,
        n_jobs=-1,
    )
    report = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
            "train_rows": int(len(train_idx)),
            "eval_rows": int(len(test_idx)),
            "scoring": scoring,
        }
    ).sort_values(["importance_mean", "feature"], ascending=[False, True]).reset_index(drop=True)
    report["rank"] = np.arange(1, len(report) + 1)
    return (
        report,
        {
            "analysis": "permutation_importance",
            "status": "ok",
            "detail": (
                f"Computed on a recent holdout window with {len(train_idx)} train rows and "
                f"{len(test_idx)} evaluation rows using {scoring}."
            ),
        },
    )


def extract_shap_matrix(explanation: Any) -> np.ndarray:
    values = getattr(explanation, "values", explanation)
    if isinstance(values, list):
        values = values[1] if len(values) > 1 else values[0]
    values = np.asarray(values)
    if values.ndim == 3:
        class_index = 1 if values.shape[-1] > 1 else 0
        values = values[:, :, class_index]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    return values


def build_shap_reports(
    model: Any,
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    empty_global = pd.DataFrame(columns=["model", "feature", "mean_abs_shap", "rank"])
    empty_latest = pd.DataFrame(columns=["model", "feature", "feature_value", "shap_value", "abs_shap_value", "rank"])

    if shap is None:
        return (
            empty_global,
            empty_latest,
            [
                {
                    "analysis": "shap",
                    "status": "skipped",
                    "detail": "Install `shap` to generate SHAP reports in Colab.",
                }
            ],
        )

    estimator = get_final_estimator(model)
    if not (hasattr(estimator, "feature_importances_") or hasattr(estimator, "coef_")):
        return (
            empty_global,
            empty_latest,
            [
                {
                    "analysis": "shap",
                    "status": "skipped",
                    "detail": f"{model_name} is not a fast tree/linear model, so SHAP was skipped to keep Colab runtime reasonable.",
                }
            ],
        )

    background_rows = min(48, len(training_dataset))
    sample_rows = min(48, len(training_dataset))
    if background_rows == 0 or sample_rows == 0:
        return (
            empty_global,
            empty_latest,
            [
                {
                    "analysis": "shap",
                    "status": "skipped",
                    "detail": "No rows were available for SHAP analysis.",
                }
            ],
        )

    X_background = transform_feature_matrix(
        model,
        training_dataset[feature_columns].tail(background_rows),
        feature_columns,
    )
    X_sample = transform_feature_matrix(
        model,
        training_dataset[feature_columns].tail(sample_rows),
        feature_columns,
    )
    X_latest = transform_feature_matrix(
        model,
        prediction_frame.iloc[[-1]][feature_columns],
        feature_columns,
    )

    try:
        explainer = shap.Explainer(estimator, X_background, feature_names=feature_columns)
        sample_explanation = explainer(X_sample)
        latest_explanation = explainer(X_latest)
    except Exception as exc:  # pragma: no cover - optional dependency/runtime edge case
        return (
            empty_global,
            empty_latest,
            [
                {
                    "analysis": "shap",
                    "status": "failed",
                    "detail": f"SHAP failed for {model_name}: {exc}",
                }
            ],
        )

    sample_matrix = extract_shap_matrix(sample_explanation)
    latest_matrix = extract_shap_matrix(latest_explanation)
    global_report = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "mean_abs_shap": np.abs(sample_matrix).mean(axis=0),
        }
    ).sort_values(["mean_abs_shap", "feature"], ascending=[False, True]).reset_index(drop=True)
    global_report["rank"] = np.arange(1, len(global_report) + 1)

    latest_values = X_latest.iloc[0].to_numpy(dtype=float)
    latest_report = pd.DataFrame(
        {
            "model": model_name,
            "feature": feature_columns,
            "feature_value": latest_values,
            "shap_value": latest_matrix[0],
            "abs_shap_value": np.abs(latest_matrix[0]),
        }
    ).sort_values(["abs_shap_value", "feature"], ascending=[False, True]).reset_index(drop=True)
    latest_report["rank"] = np.arange(1, len(latest_report) + 1)

    return (
        global_report,
        latest_report,
        [
            {
                "analysis": "shap",
                "status": "ok",
                "detail": (
                    f"Computed SHAP using {background_rows} recent background rows and "
                    f"{sample_rows} recent training rows for {model_name}."
                ),
            }
        ],
    )


def empty_explainability_artifacts(model_name: str, reason: str) -> ExplainabilityArtifacts:
    note = {
        "analysis": "explainability",
        "status": "skipped",
        "detail": reason,
    }
    return ExplainabilityArtifacts(
        model_name=model_name,
        native_feature_importance=pd.DataFrame(columns=["model", "feature", "importance", "rank"]),
        permutation_importance=pd.DataFrame(columns=["model", "feature", "importance", "rank"]),
        shap_global_importance=pd.DataFrame(columns=["model", "feature", "mean_abs_shap", "rank"]),
        shap_latest_contributions=pd.DataFrame(
            columns=["model", "feature", "feature_value", "shap_value", "abs_shap_value", "rank"]
        ),
        notes=build_notes_frame([note]),
    )


def build_explainability_artifacts(
    dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    horizon: int,
    cv_test_size: int,
    task: str,
) -> ExplainabilityArtifacts:
    notes: list[dict[str, Any]] = []
    if task == "regression":
        model_template = make_models(horizon=horizon)[model_name]
        target = dataset["target_return"]
    else:
        model_template = make_classification_models(horizon=horizon)[model_name]
        target = (dataset["target_return"] > 0).astype(int)

    if task == "classification":
        fitted_model = fit_calibrated_classifier(
            model_template,
            dataset[feature_columns],
            target,
        )
    else:
        fitted_model = clone(model_template)
        fitted_model.fit(dataset[feature_columns], target)

    native_feature_importance, native_note = build_native_feature_importance(
        model=fitted_model,
        feature_columns=feature_columns,
        model_name=model_name,
    )
    notes.append(native_note)

    permutation_report, permutation_note = build_permutation_importance_report(
        dataset=dataset,
        feature_columns=feature_columns,
        model_template=model_template,
        model_name=model_name,
        horizon=horizon,
        cv_test_size=cv_test_size,
        task=task,
    )
    notes.append(permutation_note)

    shap_global, shap_latest, shap_notes = build_shap_reports(
        model=fitted_model,
        training_dataset=dataset,
        prediction_frame=prediction_frame,
        feature_columns=feature_columns,
        model_name=model_name,
    )
    notes.extend(shap_notes)

    return ExplainabilityArtifacts(
        model_name=model_name,
        native_feature_importance=native_feature_importance,
        permutation_importance=permutation_report,
        shap_global_importance=shap_global,
        shap_latest_contributions=shap_latest,
        notes=build_notes_frame(notes),
    )


def evaluate_predictions(
    current_close: np.ndarray,
    actual_return: np.ndarray,
    predicted_return: np.ndarray,
) -> dict[str, float]:
    actual_close = current_close * (1.0 + actual_return)
    predicted_close = current_close * (1.0 + predicted_return)
    direction = np.sign(actual_return) == np.sign(predicted_return)

    return {
        "return_mae": float(mean_absolute_error(actual_return, predicted_return)),
        "price_mae": float(mean_absolute_error(actual_close, predicted_close)),
        "price_rmse": float(np.sqrt(mean_squared_error(actual_close, predicted_close))),
        "price_mape_percent": float(np.mean(np.abs((actual_close - predicted_close) / actual_close)) * 100),
        "directional_accuracy": float(direction.mean()),
    }


def _feature_series_or_nan(feature_frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in feature_frame.columns:
        return pd.Series(np.nan, index=feature_frame.index, dtype=float)
    return pd.to_numeric(feature_frame[column], errors="coerce")


def _weighted_series_average(feature_frame: pd.DataFrame, items: list[tuple[float, pd.Series]]) -> pd.Series:
    weighted_sum = pd.Series(0.0, index=feature_frame.index, dtype=float)
    total_weight = pd.Series(0.0, index=feature_frame.index, dtype=float)
    for weight, series in items:
        values = pd.to_numeric(series, errors="coerce")
        valid = values.notna()
        if not valid.any():
            continue
        weighted_sum.loc[valid] += weight * values.loc[valid]
        total_weight.loc[valid] += weight
    return weighted_sum.divide(total_weight.replace(0.0, np.nan))


def _truthy_env(name: str) -> bool:
    """Read an env var as a boolean flag. Accepts 1/true/yes/on (case-insensitive)."""
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def overlay_disabled_for_context(context: str) -> bool:
    """Let operators bypass ``apply_regime_response_overlay`` in a specific call context.

    The overlay was designed to stabilise *live* daily forecasts by blending
    tree-model predictions toward recent momentum / trend anchors. On the
    walk-forward OOF / recent-holdout diagnostic paths it cascades with the
    MAE loss and the trimmed ensemble to shrink predicted returns to ~27%
    of realised volatility at h=30 (negative skill vs the naive benchmark).

    To run "raw model" experiments without touching production headline
    outputs, set one of:

        ETH_OVERLAY_DISABLE=1                 # disables in every context
        ETH_OVERLAY_DISABLE_WALK_FORWARD=1    # only walk-forward OOF
        ETH_OVERLAY_DISABLE_HOLDOUT=1         # only the recent-holdout branch

    Live / hybrid / headline prediction paths never check this flag; they
    keep the overlay unconditionally.
    """
    if _truthy_env("ETH_OVERLAY_DISABLE"):
        return True
    return _truthy_env(f"ETH_OVERLAY_DISABLE_{context.upper()}")


def apply_regime_response_overlay(
    predictions: pd.Series | np.ndarray,
    feature_frame: pd.DataFrame,
    horizon: int,
    live_gap: pd.Series | float | None = None,
) -> pd.Series:
    """Tilt regression forecasts toward recent momentum when volatility shocks are elevated."""
    adjusted = pd.Series(np.asarray(predictions, dtype=float), index=feature_frame.index, dtype=float)
    if adjusted.empty:
        return adjusted

    r1 = _feature_series_or_nan(feature_frame, "eth_return_1")
    r2 = _feature_series_or_nan(feature_frame, "eth_return_2")
    r3 = _feature_series_or_nan(feature_frame, "eth_return_3")
    r5 = _feature_series_or_nan(feature_frame, "eth_return_5")
    r7 = _feature_series_or_nan(feature_frame, "eth_return_7")
    r14 = _feature_series_or_nan(feature_frame, "eth_return_14")
    r21 = _feature_series_or_nan(feature_frame, "eth_return_21")
    r30 = _feature_series_or_nan(feature_frame, "eth_return_30")
    ma7 = _feature_series_or_nan(feature_frame, "eth_ma_7_ratio")
    ma14 = _feature_series_or_nan(feature_frame, "eth_ma_14_ratio")
    ma30 = _feature_series_or_nan(feature_frame, "eth_ma_30_ratio")
    ma90 = _feature_series_or_nan(feature_frame, "eth_ma_90_ratio")
    vwap7 = _feature_series_or_nan(feature_frame, "eth_vwap_7_ratio")
    vwap20 = _feature_series_or_nan(feature_frame, "eth_vwap_20_ratio")
    vwap730 = _feature_series_or_nan(feature_frame, "eth_vwap_7_30_spread")
    vol7 = _feature_series_or_nan(feature_frame, "eth_vol_7").abs()
    vol14 = _feature_series_or_nan(feature_frame, "eth_vol_14").abs()
    vol30 = _feature_series_or_nan(feature_frame, "eth_vol_30").abs()
    accel = _feature_series_or_nan(feature_frame, "eth_return_accel_7_30").abs()
    close_open = _feature_series_or_nan(feature_frame, "eth_close_open_pct").abs()
    range_pct = _feature_series_or_nan(feature_frame, "eth_range_pct").abs()
    drawdown30 = _feature_series_or_nan(feature_frame, "eth_drawdown_30")
    live_gap_series = pd.Series(np.nan, index=feature_frame.index, dtype=float)
    if live_gap is not None:
        if np.isscalar(live_gap):
            live_gap_series[:] = float(live_gap)
        else:
            live_gap_series = pd.to_numeric(pd.Series(live_gap), errors="coerce").reindex(feature_frame.index)

    base_vol = vol30.where(vol30 > 1e-6, np.nan).fillna(vol14.where(vol14 > 1e-6, np.nan)).fillna(
        vol7.where(vol7 > 1e-6, np.nan)
    )
    shock_ratio = r1.abs().divide(base_vol).replace([np.inf, -np.inf], np.nan)
    accel_ratio = accel.divide(base_vol).replace([np.inf, -np.inf], np.nan)
    intraday_ratio = close_open.divide(base_vol).replace([np.inf, -np.inf], np.nan)
    range_ratio = range_pct.divide(base_vol).replace([np.inf, -np.inf], np.nan)
    vol_regime = vol7.divide(vol30.where(vol30 > 1e-6, np.nan)).replace([np.inf, -np.inf], np.nan)
    live_ratio = live_gap_series.abs().divide(base_vol).replace([np.inf, -np.inf], np.nan)
    short_burst = _weighted_series_average(
        feature_frame,
        [(0.30, r1), (0.25, r2), (0.20, r3), (0.15, r5), (0.10, r7)],
    )
    medium_burst = _weighted_series_average(
        feature_frame,
        [(0.35, r5), (0.25, r7), (0.20, r14), (0.10, r21), (0.10, r30)],
    )
    trend_stack = _weighted_series_average(
        feature_frame,
        [(0.28, ma7), (0.22, ma14), (0.20, ma30), (0.10, ma90), (0.10, vwap7), (0.10, vwap20)],
    )
    short_burst_ratio = short_burst.abs().divide(base_vol).replace([np.inf, -np.inf], np.nan)
    medium_burst_ratio = medium_burst.abs().divide(base_vol).replace([np.inf, -np.inf], np.nan)
    trend_ratio = trend_stack.abs().divide(base_vol).replace([np.inf, -np.inf], np.nan)
    high_proximity_component = (
        1.0 - drawdown30.abs().clip(lower=0.0, upper=0.30).fillna(0.30).divide(0.30)
    ).clip(lower=0.0, upper=1.0)

    if horizon <= 7:
        momentum_anchor = _weighted_series_average(
            feature_frame,
            [(0.45, r1), (0.35, r3), (0.20, r7)],
        )
        trend_anchor = _weighted_series_average(
            feature_frame,
            [(0.60, ma7), (0.40, vwap7)],
        )
        anchor = _weighted_series_average(
            feature_frame,
            [(0.55, momentum_anchor), (0.15, trend_anchor), (0.30, live_gap_series)],
        )
        max_blend = 0.65
        continuation_anchor = _weighted_series_average(
            feature_frame,
            [(0.55, short_burst), (0.25, trend_stack), (0.20, live_gap_series)],
        )
        max_continuation_pull = 0.38
        continuation_floor_weight = 0.42
        cap_floor = 0.65
    else:
        momentum_anchor = _weighted_series_average(
            feature_frame,
            [(0.45, r7), (0.35, r14), (0.20, r30)],
        )
        shock_anchor = _weighted_series_average(
            feature_frame,
            [(0.60, r1), (0.40, r3)],
        )
        trend_anchor = _weighted_series_average(
            feature_frame,
            [(0.45, ma7), (0.30, ma30), (0.25, vwap730)],
        )
        anchor = _weighted_series_average(
            feature_frame,
            [(0.45, momentum_anchor), (0.20, shock_anchor), (0.20, trend_anchor), (0.15, live_gap_series)],
        )
        max_blend = 0.55
        continuation_anchor = _weighted_series_average(
            feature_frame,
            [(0.45, medium_burst), (0.25, short_burst), (0.20, trend_stack), (0.10, live_gap_series)],
        )
        max_continuation_pull = 0.24
        continuation_floor_weight = 0.22
        cap_floor = 1.25

    shock_component = shock_ratio.clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
    accel_component = accel_ratio.clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
    intraday_component = intraday_ratio.clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
    burst_component = short_burst_ratio.clip(lower=0.0, upper=4.0).fillna(0.0) / 4.0
    medium_component = medium_burst_ratio.clip(lower=0.0, upper=4.0).fillna(0.0) / 4.0
    trend_component = trend_ratio.clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
    range_component = range_ratio.clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
    vol_component = (vol_regime - 1.0).clip(lower=0.0, upper=2.0).fillna(0.0) / 2.0
    live_component = live_ratio.clip(lower=0.0, upper=3.0).fillna(0.0) / 3.0
    blend_strength = (
        0.10
        + 0.17 * shock_component
        + 0.10 * accel_component
        + 0.08 * intraday_component
        + 0.12 * vol_component
        + 0.18 * live_component
    )
    blend_strength = blend_strength.clip(lower=0.10, upper=max_blend)
    if horizon <= 7:
        continuation_component = (
            0.28 * burst_component
            + 0.16 * medium_component
            + 0.18 * trend_component
            + 0.10 * high_proximity_component
            + 0.08 * range_component
            + 0.20 * live_component
        )
    else:
        continuation_component = (
            0.18 * burst_component
            + 0.24 * medium_component
            + 0.20 * trend_component
            + 0.12 * high_proximity_component
            + 0.08 * range_component
            + 0.18 * live_component
        )
    blend_strength = (blend_strength + 0.10 * continuation_component).clip(lower=0.10, upper=max_blend)

    anchor_sign = np.sign(anchor.fillna(0.0))
    base_sign = np.sign(adjusted.fillna(0.0))
    sign_aligned = (anchor_sign == 0.0) | (base_sign == 0.0) | (anchor_sign == base_sign)
    stronger_anchor = anchor.abs() > adjusted.abs()
    blend_strength = blend_strength.where(sign_aligned, blend_strength * 0.45)
    blend_strength = blend_strength.where(stronger_anchor, blend_strength * 0.65)

    adjusted = adjusted + blend_strength * (anchor.fillna(adjusted) - adjusted)

    continuation_sign = np.sign(continuation_anchor.fillna(0.0))
    current_sign = np.sign(adjusted.fillna(0.0))
    live_sign = np.sign(live_gap_series.fillna(0.0))
    continuation_pull = (
        (0.04 if horizon <= 7 else 0.02) + max_continuation_pull * continuation_component
    ).clip(lower=0.0, upper=max_continuation_pull)
    continuation_aligned = (
        (continuation_sign == 0.0)
        | (current_sign == 0.0)
        | (continuation_sign == current_sign)
        | (continuation_component > 0.75)
    )
    continuation_aligned = continuation_aligned & (
        (live_sign == 0.0) | (continuation_sign == 0.0) | (live_sign == continuation_sign)
    )
    continuation_pull = continuation_pull.where(continuation_aligned, continuation_pull * 0.35)
    adjusted = adjusted + continuation_pull * (continuation_anchor.fillna(adjusted) - adjusted)

    continuation_floor = continuation_anchor.abs().fillna(0.0) * continuation_floor_weight
    strong_continuation = (
        (continuation_component > (0.62 if horizon <= 7 else 0.68))
        & continuation_anchor.notna()
        & (continuation_sign != 0.0)
    )
    weak_response = adjusted.abs() < continuation_floor
    use_continuation_floor = strong_continuation & weak_response
    adjusted.loc[use_continuation_floor] = (
        continuation_sign.loc[use_continuation_floor] * continuation_floor.loc[use_continuation_floor]
    )

    live_weight = ((0.45 if horizon <= 7 else 0.18) * (0.55 + 0.45 * live_component)).clip(lower=0.0)
    adjusted = adjusted + live_weight * live_gap_series.fillna(0.0)

    opposing_live = (
        live_gap_series.notna()
        & (np.sign(live_gap_series) != 0.0)
        & (np.sign(adjusted) != 0.0)
        & (np.sign(live_gap_series) != np.sign(adjusted))
    )
    shrink_factor = (0.25 + 0.45 * live_component).clip(lower=0.0, upper=0.70)
    adjusted.loc[opposing_live] = adjusted.loc[opposing_live] * (1.0 - shrink_factor.loc[opposing_live])

    shock_floor = (vol7.fillna(base_vol) * np.sqrt(max(horizon, 1))).clip(lower=0.0, upper=cap_floor) * 0.25
    target_direction = np.sign(anchor.where(anchor.abs() >= adjusted.abs(), adjusted).fillna(adjusted).fillna(0.0))
    floor_value = pd.Series(target_direction, index=feature_frame.index, dtype=float) * shock_floor.fillna(0.0)
    use_floor = (shock_component > 0.70) & (anchor.abs() > adjusted.abs()) & (adjusted.abs() < shock_floor.fillna(0.0))
    adjusted.loc[use_floor] = floor_value.loc[use_floor]

    cap = (base_vol.fillna(vol7).fillna(0.04) * np.sqrt(max(horizon, 1)) * 3.0).clip(
        lower=0.08 if horizon <= 7 else 0.12,
        upper=cap_floor,
    )
    adjusted = adjusted.clip(lower=-cap, upper=cap)
    return adjusted


def apply_direction_regime_overlay(
    probabilities: pd.Series | np.ndarray,
    feature_frame: pd.DataFrame,
    horizon: int,
    live_gap: pd.Series | float | None = None,
) -> pd.Series:
    adjusted = pd.Series(np.asarray(probabilities, dtype=float), index=feature_frame.index, dtype=float)
    if adjusted.empty:
        return adjusted

    def scale_signed(series: pd.Series, scale: float) -> pd.Series:
        if scale <= 0.0:
            return pd.Series(0.0, index=series.index, dtype=float)
        return series.clip(lower=-scale, upper=scale).divide(scale).fillna(0.0)

    def scale_positive(series: pd.Series, scale: float) -> pd.Series:
        if scale <= 0.0:
            return pd.Series(0.0, index=series.index, dtype=float)
        return series.clip(lower=0.0, upper=scale).divide(scale).fillna(0.0)

    r1 = _feature_series_or_nan(feature_frame, "eth_return_1")
    r3 = _feature_series_or_nan(feature_frame, "eth_return_3")
    r5 = _feature_series_or_nan(feature_frame, "eth_return_5")
    r7 = _feature_series_or_nan(feature_frame, "eth_return_7")
    ma7 = _feature_series_or_nan(feature_frame, "eth_ma_7_ratio")
    ma30 = _feature_series_or_nan(feature_frame, "eth_ma_30_ratio")
    vwap20 = _feature_series_or_nan(feature_frame, "eth_vwap_20_ratio")
    drawdown30 = _feature_series_or_nan(feature_frame, "eth_drawdown_30")
    rsi14 = _feature_series_or_nan(feature_frame, "eth_rsi_14")
    rsi14_change_3 = _feature_series_or_nan(feature_frame, "eth_rsi_14_change_3")
    macd_hist_change_3 = _feature_series_or_nan(feature_frame, "eth_macd_hist_change_3")
    up_streak = _feature_series_or_nan(feature_frame, "eth_up_streak")
    up_day_ratio_7 = _feature_series_or_nan(feature_frame, "eth_up_day_ratio_7")
    up_day_ratio_14 = _feature_series_or_nan(feature_frame, "eth_up_day_ratio_14")
    rebound20 = _feature_series_or_nan(feature_frame, "eth_rebound_from_20d_low")
    rebound60 = _feature_series_or_nan(feature_frame, "eth_rebound_from_60d_low")
    distance20 = _feature_series_or_nan(feature_frame, "eth_distance_to_20d_high")
    adx_bias = _feature_series_or_nan(feature_frame, "eth_adx_trend_bias_14")
    range_position = _feature_series_or_nan(feature_frame, "eth_range_position_20")
    live_gap_series = pd.Series(np.nan, index=feature_frame.index, dtype=float)
    if live_gap is not None:
        if np.isscalar(live_gap):
            live_gap_series[:] = float(live_gap)
        else:
            live_gap_series = pd.to_numeric(pd.Series(live_gap), errors="coerce").reindex(feature_frame.index)

    near_high_component = (
        1.0 - distance20.abs().clip(lower=0.0, upper=0.15).fillna(0.15).divide(0.15)
    ).clip(lower=0.0, upper=1.0)
    continuation_score = (
        0.16 * scale_signed(r3, 0.12)
        + 0.14 * scale_signed(r7, 0.20)
        + 0.12 * scale_signed(rebound20, 0.25)
        + 0.08 * scale_signed(rebound60, 0.40)
        + 0.10 * scale_signed(ma7, 0.20)
        + 0.08 * scale_signed(ma30, 0.25)
        + 0.08 * scale_signed(vwap20, 0.20)
        + 0.08 * scale_signed(adx_bias, 1.0)
        + 0.06 * scale_signed(up_day_ratio_7 - 0.5, 0.5)
        + 0.05 * scale_signed(up_day_ratio_14 - 0.5, 0.5)
        + 0.05 * scale_signed(range_position - 0.5, 0.5)
        + 0.10 * scale_signed(live_gap_series, 0.08)
    ).clip(lower=-1.0, upper=1.0)

    exhaustion_score = (
        0.18 * scale_positive(-r1, 0.08)
        + 0.14 * near_high_component
        + 0.12 * scale_positive(rsi14 - 68.0, 22.0)
        + 0.12 * scale_positive(-macd_hist_change_3, 0.05)
        + 0.10 * scale_positive(-rsi14_change_3, 12.0)
        + 0.10 * scale_positive(up_streak - 6.0, 6.0)
        + 0.10 * scale_positive(rebound20, 0.25)
        + 0.08 * scale_positive(rebound60, 0.40)
        + 0.06 * scale_positive(-live_gap_series, 0.08)
    ).clip(lower=0.0, upper=1.0)

    rebound_score = (
        0.16 * scale_positive(r1, 0.08)
        + 0.12 * scale_positive(r3, 0.12)
        + 0.12 * scale_positive(rebound20, 0.25)
        + 0.10 * scale_positive(rebound60, 0.40)
        + 0.08 * scale_positive(-drawdown30, 0.35)
        + 0.08 * scale_positive(adx_bias, 1.0)
        + 0.10 * scale_positive(macd_hist_change_3, 0.05)
        + 0.10 * scale_positive(rsi14_change_3, 12.0)
        + 0.08 * scale_positive(up_day_ratio_7 - 0.5, 0.5)
        + 0.06 * scale_positive(live_gap_series, 0.08)
    ).clip(lower=0.0, upper=1.0)

    if horizon <= 7:
        probability_shift = 0.10 * continuation_score + 0.05 * rebound_score - 0.08 * exhaustion_score
    else:
        probability_shift = 0.08 * continuation_score + 0.04 * rebound_score - 0.06 * exhaustion_score

    adjusted = (adjusted + probability_shift).clip(lower=0.02, upper=0.98)
    return adjusted


def evaluate_direction_classification(
    actual_label: np.ndarray,
    predicted_label: np.ndarray,
    probability_up: np.ndarray,
) -> dict[str, float]:
    probability_up = np.clip(probability_up, 1e-6, 1 - 1e-6)
    labels = [0, 1]
    metrics = {
        "accuracy": float(accuracy_score(actual_label, predicted_label)),
        "balanced_accuracy": float(
            recall_score(actual_label, predicted_label, labels=labels, average="macro", zero_division=0)
        ),
        "precision": float(
            precision_score(actual_label, predicted_label, labels=labels, average="binary", zero_division=0)
        ),
        "recall": float(
            recall_score(actual_label, predicted_label, labels=labels, average="binary", zero_division=0)
        ),
        "f1": float(
            f1_score(actual_label, predicted_label, labels=labels, average="binary", zero_division=0)
        ),
        "brier_score": float(brier_score_loss(actual_label, probability_up)),
    }
    if len(np.unique(actual_label)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(actual_label, probability_up))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def fit_state_classifier(
    model_template: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | np.ndarray | list[float] | None = None,
) -> Any:
    model = clone(model_template)
    aligned_sample_weight = align_sample_weight(sample_weight, X_train.index)
    fit_model_with_optional_sample_weight(model, X_train, y_train.astype(int), aligned_sample_weight)
    return model


def assess_state_target_viability(
    target: pd.Series,
    task_key: str,
    horizon: int,
    test_size: int,
    n_splits: int,
) -> tuple[bool, str]:
    clean_target = pd.to_numeric(target, errors="coerce").dropna().astype(int)
    if clean_target.empty:
        return False, "empty_target"

    counts = clean_target.value_counts().to_dict()
    recent_window = min(
        len(clean_target),
        max(test_size * max(n_splits, 3), horizon * 10, 120 if task_key == "reversal" else 90),
    )
    recent_target = clean_target.iloc[-recent_window:] if recent_window > 0 else clean_target
    recent_counts = recent_target.value_counts().to_dict()
    recent_unique = int(recent_target.nunique())

    if task_key == "reversal":
        min_event_count = DEFAULT_STATE_MIN_TOTAL_EVENTS_SHORT if horizon <= 7 else DEFAULT_STATE_MIN_TOTAL_EVENTS_LONG
        top_total = int(counts.get(0, 0))
        bottom_total = int(counts.get(2, 0))
        recent_events = int(recent_counts.get(0, 0) + recent_counts.get(2, 0))
        if top_total < min_event_count or bottom_total < min_event_count:
            return False, "insufficient_reversal_events"
        if recent_events < max(2, min_event_count // 2):
            return False, "stale_reversal_events"
        if recent_unique < 2:
            return False, "single_recent_reversal_class"
        return True, "ok"

    non_neutral_total = int(counts.get(0, 0) + counts.get(2, 0))
    recent_non_neutral = int(recent_counts.get(0, 0) + recent_counts.get(2, 0))
    if clean_target.nunique() < 2:
        return False, "single_total_regime_class"
    if non_neutral_total < max(12, n_splits * 3):
        return False, "insufficient_regime_extremes"
    if recent_non_neutral < max(4, n_splits):
        return False, "stale_regime_extremes"
    if recent_unique < 2:
        return False, "single_recent_regime_class"
    return True, "ok"


def align_state_probability_frame(
    probability_matrix: np.ndarray,
    classes: np.ndarray | list[Any],
    index: pd.Index,
) -> pd.DataFrame:
    frame = pd.DataFrame(0.0, index=index, columns=[0, 1, 2], dtype=float)
    class_values = list(pd.Index(classes).astype(int)) if len(classes) else []
    for column_index, class_value in enumerate(class_values):
        if class_value in frame.columns and column_index < probability_matrix.shape[1]:
            frame[class_value] = pd.to_numeric(probability_matrix[:, column_index], errors="coerce")
    row_sum = frame.sum(axis=1).replace(0.0, np.nan)
    frame = frame.divide(row_sum, axis=0).fillna(0.0)
    return frame


def evaluate_multistate_classification(
    actual_label: np.ndarray,
    predicted_label: np.ndarray,
    probability_frame: pd.DataFrame,
) -> dict[str, float]:
    actual = np.asarray(actual_label, dtype=int)
    predicted = np.asarray(predicted_label, dtype=int)
    observed_labels = sorted(pd.Index(actual).dropna().astype(int).unique().tolist())
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)) if len(actual) else float("nan"),
        "balanced_accuracy": float("nan"),
        "macro_precision": float("nan"),
        "macro_recall": float("nan"),
        "macro_f1": float("nan"),
    }
    if len(observed_labels) >= 2:
        recalls = recall_score(actual, predicted, labels=observed_labels, average=None, zero_division=0)
        metrics["balanced_accuracy"] = float(np.mean(recalls)) if len(recalls) else float("nan")
        metrics["macro_precision"] = float(
            precision_score(actual, predicted, labels=observed_labels, average="macro", zero_division=0)
        )
        metrics["macro_recall"] = float(
            recall_score(actual, predicted, labels=observed_labels, average="macro", zero_division=0)
        )
        metrics["macro_f1"] = float(
            f1_score(actual, predicted, labels=observed_labels, average="macro", zero_division=0)
        )

    metrics["roc_auc"] = float("nan")
    try:
        if len(observed_labels) >= 3:
            metrics["roc_auc"] = float(
                roc_auc_score(
                    actual,
                    probability_frame.loc[:, observed_labels].to_numpy(dtype=float),
                    multi_class="ovr",
                    labels=observed_labels,
                )
            )
        elif len(observed_labels) == 2:
            positive_label = observed_labels[-1]
            if positive_label in probability_frame.columns:
                metrics["roc_auc"] = float(
                    roc_auc_score((actual == positive_label).astype(int), probability_frame[positive_label].to_numpy(dtype=float))
                )
    except Exception:
        metrics["roc_auc"] = float("nan")
    return metrics


def resolve_classification_signal_thresholds(
    threshold: float,
    min_edge: float = DEFAULT_CLASSIFICATION_MIN_EDGE,
) -> tuple[float, float]:
    requested_upper = float(np.clip(threshold, 0.5, 0.99))
    effective_upper = max(requested_upper, 0.5 + float(min_edge))
    effective_lower = min(1.0 - requested_upper, 0.5 - float(min_edge))
    return effective_lower, effective_upper


def regression_signal_from_predictions(predictions: pd.Series, threshold: float = 0.0) -> pd.Series:
    signal = pd.Series(
        np.where(
            predictions > threshold,
            1.0,
            np.where(predictions < -threshold, -1.0, 0.0),
        ),
        index=predictions.index,
        dtype=float,
    )
    signal[predictions.isna()] = np.nan
    return signal


def classification_signal_from_probabilities(probability_up: pd.Series, threshold: float = 0.55) -> pd.Series:
    lower_threshold, upper_threshold = resolve_classification_signal_thresholds(threshold)
    signal = pd.Series(
        np.where(
            probability_up >= upper_threshold,
            1.0,
            np.where(probability_up <= lower_threshold, -1.0, 0.0),
        ),
        index=probability_up.index,
        dtype=float,
    )
    signal[probability_up.isna()] = np.nan
    return signal


def state_signal_from_probabilities(
    probability_negative: pd.Series,
    probability_neutral: pd.Series,
    probability_positive: pd.Series,
    confidence_threshold: float = 0.50,
    min_edge: float = 0.05,
) -> pd.Series:
    negative = pd.to_numeric(probability_negative, errors="coerce")
    neutral = pd.to_numeric(probability_neutral, errors="coerce")
    positive = pd.to_numeric(probability_positive, errors="coerce")
    probability_frame = pd.concat([negative, neutral, positive], axis=1)
    probability_frame.columns = ["negative", "neutral", "positive"]
    strongest = probability_frame.max(axis=1)
    competitor = pd.concat(
        [
            pd.concat([neutral, positive], axis=1).max(axis=1),
            pd.concat([neutral, negative], axis=1).max(axis=1),
        ],
        axis=1,
    )
    negative_edge = negative - competitor.iloc[:, 0]
    positive_edge = positive - competitor.iloc[:, 1]

    signal = pd.Series(0.0, index=probability_frame.index, dtype=float)
    negative_mask = (
        negative.notna()
        & (negative >= confidence_threshold)
        & (negative >= positive)
        & (negative >= neutral)
        & (negative_edge >= min_edge)
    )
    positive_mask = (
        positive.notna()
        & (positive >= confidence_threshold)
        & (positive >= negative)
        & (positive >= neutral)
        & (positive_edge >= min_edge)
    )
    signal.loc[negative_mask] = -1.0
    signal.loc[positive_mask] = 1.0
    signal.loc[strongest.isna()] = np.nan
    return signal


def backtest_sort_key(metrics: dict[str, float | str]) -> tuple[float, float, float]:
    sharpe = pd.to_numeric(metrics.get("sharpe"), errors="coerce")
    total_return = pd.to_numeric(metrics.get("total_return"), errors="coerce")
    max_drawdown = pd.to_numeric(metrics.get("max_drawdown"), errors="coerce")
    sharpe_value = float(sharpe) if pd.notna(sharpe) else float("-inf")
    total_return_value = float(total_return) if pd.notna(total_return) else float("-inf")
    max_drawdown_value = float(max_drawdown) if pd.notna(max_drawdown) else float("-inf")
    return sharpe_value, total_return_value, max_drawdown_value


def build_purged_threshold_validation_windows(
    reference_index: pd.Index,
    horizon: int,
    *,
    min_train_rows: int = DEFAULT_THRESHOLD_VALIDATION_MIN_TRAIN_ROWS,
    min_eval_rows: int = DEFAULT_THRESHOLD_VALIDATION_MIN_EVAL_ROWS,
    eval_fraction: float = DEFAULT_THRESHOLD_VALIDATION_EVAL_FRACTION,
) -> tuple[pd.Index, pd.Index, int, str]:
    ordered_index = pd.Index(reference_index)
    if hasattr(ordered_index, "is_monotonic_increasing") and not ordered_index.is_monotonic_increasing:
        ordered_index = ordered_index.sort_values()
    total_rows = len(ordered_index)
    embargo_rows = max(int(horizon), 1)
    available_rows = total_rows - embargo_rows
    if total_rows == 0 or available_rows < 24:
        return ordered_index, ordered_index, 0, "full_sample"

    # Scale minimum validation windows to the amount of OOF data that exists so
    # shorter histories can still use a purged split instead of falling back.
    required_train_rows = max(18, min(int(min_train_rows), max(available_rows // 2, 18)))
    required_eval_rows = max(12, min(int(min_eval_rows), max(available_rows // 3, 12)))
    max_eval_rows = total_rows - embargo_rows - required_train_rows
    if max_eval_rows < required_eval_rows:
        required_train_rows = max(12, total_rows - embargo_rows - required_eval_rows)
        max_eval_rows = total_rows - embargo_rows - required_train_rows
    if max_eval_rows < required_eval_rows:
        return ordered_index, ordered_index, 0, "full_sample"

    target_eval_rows = max(required_eval_rows, int(round(available_rows * float(eval_fraction))))
    evaluation_rows = min(max_eval_rows, target_eval_rows)
    selection_end = total_rows - evaluation_rows - embargo_rows
    selection_index = ordered_index[:selection_end]
    evaluation_index = ordered_index[-evaluation_rows:]
    if len(selection_index) < required_train_rows or len(evaluation_index) < required_eval_rows:
        return ordered_index, ordered_index, 0, "full_sample"
    return selection_index, evaluation_index, embargo_rows, "nested_purged_temporal"


def annotate_threshold_validation_metrics(
    metrics: dict[str, Any],
    *,
    scheme: str,
    selection_rows: int,
    evaluation_rows: int,
    embargo_rows: int,
) -> dict[str, Any]:
    annotated = dict(metrics)
    annotated["threshold_validation_scheme"] = str(scheme)
    annotated["threshold_selection_rows"] = int(selection_rows)
    annotated["threshold_evaluation_rows"] = int(evaluation_rows)
    annotated["validation_embargo_rows"] = int(embargo_rows)
    annotated["validation_leakage_safe"] = bool(str(scheme).startswith("nested_purged"))
    return annotated


def choose_threshold_via_nested_backtest(
    *,
    threshold_candidates: list[float],
    signal_builder: Any,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
    reference_index: pd.Index,
    threshold_labeler: Any | None = None,
) -> tuple[float, dict[str, Any], pd.DataFrame]:
    if threshold_labeler is None:
        threshold_labeler = lambda value: float(value)

    selection_index, evaluation_index, embargo_rows, scheme = build_purged_threshold_validation_windows(
        reference_index,
        horizon,
    )
    if len(selection_index) == 0:
        scheme = "full_sample"
        selection_index = pd.Index(reference_index)
        evaluation_index = pd.Index(reference_index)
        embargo_rows = 0

    selection_candidates: list[tuple[float, dict[str, Any]]] = []
    for threshold in threshold_candidates:
        signal = signal_builder(float(threshold))
        selection_signal = signal.reindex(selection_index)
        metrics, _ = run_signal_backtest(
            selection_signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        total_return = pd.to_numeric(metrics.get("total_return"), errors="coerce")
        if pd.notna(total_return):
            selection_candidates.append((float(threshold), metrics))

    if not selection_candidates:
        # Do not fall back to a full-sample threshold search. If the purged
        # selection window produces no tradable signals, keep the validation
        # window intact and report the default threshold as a no-selection case.
        fallback_threshold = float(threshold_candidates[0])
        evaluation_signal = signal_builder(fallback_threshold).reindex(evaluation_index)
        evaluation_metrics, evaluation_curve = run_signal_backtest(
            evaluation_signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        safe_scheme = f"{scheme}_no_selection_signal" if str(scheme).startswith("nested_purged") else scheme
        evaluation_metrics = annotate_threshold_validation_metrics(
            evaluation_metrics,
            scheme=safe_scheme,
            selection_rows=len(selection_index),
            evaluation_rows=len(evaluation_index),
            embargo_rows=embargo_rows,
        )
        evaluation_metrics["signal_threshold"] = float(threshold_labeler(fallback_threshold))
        return fallback_threshold, evaluation_metrics, evaluation_curve

    best_threshold, _ = max(selection_candidates, key=lambda item: backtest_sort_key(item[1]))
    evaluation_signal = signal_builder(float(best_threshold)).reindex(evaluation_index)
    evaluation_metrics, evaluation_curve = run_signal_backtest(
        evaluation_signal,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    evaluation_metrics = annotate_threshold_validation_metrics(
        evaluation_metrics,
        scheme=scheme,
        selection_rows=len(selection_index),
        evaluation_rows=len(evaluation_index),
        embargo_rows=embargo_rows,
    )
    evaluation_metrics["signal_threshold"] = float(threshold_labeler(float(best_threshold)))
    return float(best_threshold), evaluation_metrics, evaluation_curve


def run_signal_backtest(
    signal: pd.Series,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
    fee_bps: float = 5.0,
    enter_on_next_bar: bool = True,
) -> tuple[dict[str, float], pd.DataFrame]:
    valid_signal = signal.dropna()
    if valid_signal.empty:
        empty_curve = pd.DataFrame(
            columns=[
                "signal_timestamp",
                "entry_timestamp",
                "signal",
                "gross_return",
                "benchmark_return",
                "strategy_return",
                "equity",
                "benchmark_equity",
            ]
        )
        return {
            "total_return": float("nan"),
            "benchmark_return": float("nan"),
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "avg_exposure": float("nan"),
            "hit_rate": float("nan"),
            "signal_count": 0.0,
            "turnover": float("nan"),
        }, empty_curve

    close = market_data["eth_close"]
    full_index = close.index
    candidate_rows: list[tuple[pd.Timestamp, pd.Timestamp, int, float]] = []
    entry_offset = 1 if enter_on_next_bar else 0
    for timestamp, signal_value in valid_signal.sort_index().items():
        signal_location = int(full_index.get_indexer([timestamp])[0])
        entry_location = signal_location + entry_offset
        if signal_location < 0 or entry_location < 0 or entry_location + horizon >= len(full_index):
            continue
        candidate_rows.append(
            (
                pd.Timestamp(timestamp),
                pd.Timestamp(full_index[entry_location]),
                entry_location,
                float(signal_value),
            )
        )

    if not candidate_rows:
        empty_curve = pd.DataFrame(
            columns=[
                "signal_timestamp",
                "entry_timestamp",
                "signal",
                "gross_return",
                "benchmark_return",
                "strategy_return",
                "equity",
                "benchmark_equity",
            ]
        )
        return {
            "total_return": float("nan"),
            "benchmark_return": float("nan"),
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "avg_exposure": float("nan"),
            "hit_rate": float("nan"),
            "signal_count": 0.0,
            "turnover": float("nan"),
        }, empty_curve

    equity_value = 1.0
    benchmark_equity_value = 1.0
    invested_rows = 0
    event_rows: list[dict[str, Any]] = []
    cursor = 0

    while cursor < len(candidate_rows):
        signal_timestamp, entry_timestamp, entry_location, signal_value = candidate_rows[cursor]
        if not np.isfinite(signal_value) or signal_value == 0.0:
            cursor += 1
            continue

        exit_location = entry_location + horizon
        gross_return = float(close.iloc[exit_location] / close.iloc[entry_location] - 1.0)
        transaction_cost = abs(signal_value) * (2.0 * fee_bps / 10000.0)
        strategy_return = float(signal_value * gross_return - transaction_cost)

        equity_value *= 1.0 + strategy_return
        benchmark_equity_value *= 1.0 + gross_return
        invested_rows += horizon
        event_rows.append(
            {
                "signal_timestamp": signal_timestamp,
                "entry_timestamp": entry_timestamp,
                "signal": signal_value,
                "gross_return": gross_return,
                "benchmark_return": gross_return,
                "strategy_return": strategy_return,
                "equity": equity_value,
                "benchmark_equity": benchmark_equity_value,
            }
        )

        cursor += 1
        while cursor < len(candidate_rows) and candidate_rows[cursor][2] <= exit_location:
            cursor += 1

    if not event_rows:
        empty_curve = pd.DataFrame(
            columns=[
                "signal_timestamp",
                "entry_timestamp",
                "signal",
                "gross_return",
                "benchmark_return",
                "strategy_return",
                "equity",
                "benchmark_equity",
            ]
        )
        return {
            "total_return": float("nan"),
            "benchmark_return": float("nan"),
            "annual_return": float("nan"),
            "annual_volatility": float("nan"),
            "sharpe": float("nan"),
            "max_drawdown": float("nan"),
            "avg_exposure": float("nan"),
            "hit_rate": float("nan"),
            "signal_count": 0.0,
            "turnover": float("nan"),
        }, empty_curve

    curve = pd.DataFrame(event_rows).set_index(pd.to_datetime([row["entry_timestamp"] for row in event_rows]))
    curve.index.name = "event_timestamp"
    strategy_returns = curve["strategy_return"]
    benchmark_returns = curve["benchmark_return"]
    equity = curve["equity"]
    benchmark_equity = curve["benchmark_equity"]
    events_per_year = max(periods_per_year(interval) / max(horizon, 1), 1.0)
    avg_return = strategy_returns.mean()
    return_std = strategy_returns.std()
    annual_return = float((equity.iloc[-1] ** (events_per_year / len(curve))) - 1.0) if len(curve) > 0 else float("nan")
    annual_vol = float(return_std * np.sqrt(events_per_year)) if pd.notna(return_std) else float("nan")
    sharpe = float((avg_return / return_std) * np.sqrt(events_per_year)) if return_std and return_std > 0 else float("nan")
    evaluation_span_rows = max(candidate_rows[-1][2] + horizon - candidate_rows[0][2], 1)

    metrics = {
        "total_return": float(equity.iloc[-1] - 1.0),
        "benchmark_return": float(benchmark_equity.iloc[-1] - 1.0),
        "annual_return": annual_return,
        "annual_volatility": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": compute_max_drawdown(equity),
        "avg_exposure": float(min(invested_rows / evaluation_span_rows, 1.0)),
        "hit_rate": float((strategy_returns > 0).mean()),
        "signal_count": float(len(curve)),
        "turnover": float((curve["signal"].abs() * 2.0).sum()),
    }
    return metrics, curve


def backtest_regression_models(
    oof_predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    equity_curves = pd.DataFrame()
    model_names = sorted(column.replace("_pred_return", "") for column in oof_predictions.columns if column.endswith("_pred_return"))

    for model_name in model_names:
        predictions = pd.to_numeric(oof_predictions[f"{model_name}_pred_return"], errors="coerce")
        valid_index = predictions.dropna().index
        if len(valid_index) == 0:
            continue
        _, best_metrics, best_curve = choose_threshold_via_nested_backtest(
            threshold_candidates=DEFAULT_REGRESSION_THRESHOLD_GRID,
            signal_builder=lambda threshold, series=predictions: regression_signal_from_predictions(series, threshold=threshold),
            market_data=market_data,
            horizon=horizon,
            interval=interval,
            reference_index=valid_index,
        )
        best_metrics["model"] = model_name
        rows.append(best_metrics)
        if not best_curve.empty:
            if equity_curves.empty:
                equity_curves = best_curve[["benchmark_equity"]].rename(columns={"benchmark_equity": "benchmark_equity"})
            equity_curves[f"{model_name}_equity"] = best_curve["equity"]
            equity_curves[f"{model_name}_signal"] = best_curve["signal"]

    leaderboard = pd.DataFrame(rows).sort_values(
        ["sharpe", "total_return", "max_drawdown"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return leaderboard, equity_curves


def backtest_classification_models(
    oof_predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    equity_curves = pd.DataFrame()
    model_names = sorted(column.replace("_prob_up", "") for column in oof_predictions.columns if column.endswith("_prob_up"))

    for model_name in model_names:
        probabilities = pd.to_numeric(oof_predictions[f"{model_name}_prob_up"], errors="coerce")
        valid_index = probabilities.dropna().index
        if len(valid_index) == 0:
            continue
        _, best_metrics, best_curve = choose_threshold_via_nested_backtest(
            threshold_candidates=DEFAULT_CLASSIFICATION_THRESHOLD_GRID,
            signal_builder=lambda threshold, series=probabilities: classification_signal_from_probabilities(series, threshold=threshold),
            market_data=market_data,
            horizon=horizon,
            interval=interval,
            reference_index=valid_index,
            threshold_labeler=lambda threshold: resolve_classification_signal_thresholds(float(threshold))[1],
        )
        best_metrics["model"] = model_name
        rows.append(best_metrics)
        if not best_curve.empty:
            if equity_curves.empty:
                equity_curves = best_curve[["benchmark_equity"]].rename(columns={"benchmark_equity": "benchmark_equity"})
            equity_curves[f"{model_name}_equity"] = best_curve["equity"]
            equity_curves[f"{model_name}_signal"] = best_curve["signal"]

    leaderboard = pd.DataFrame(rows).sort_values(
        ["sharpe", "total_return", "max_drawdown"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return leaderboard, equity_curves


def walk_forward_state_classification(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    test_size: int,
    gap: int,
    target_column: str,
    task_key: str,
    sample_weight: pd.Series | None = None,
    verbose: bool = False,
    label: str = "",
    fold_feature_selection: bool = False,
    min_feature_coverage: float = DEFAULT_FEATURE_MIN_COVERAGE,
    embargo: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_dataset = dataset.loc[pd.to_numeric(dataset[target_column], errors="coerce").notna()].copy()
    if state_dataset.empty:
        return pd.DataFrame(), pd.DataFrame(index=dataset.index)
    state_viable, _ = assess_state_target_viability(
        state_dataset[target_column],
        task_key=task_key,
        horizon=gap,
        test_size=test_size,
        n_splits=n_splits,
    )
    if not state_viable:
        return pd.DataFrame(), pd.DataFrame(index=dataset.index)

    y = pd.to_numeric(state_dataset[target_column], errors="coerce").astype(int)
    splitter = purged_time_series_split(
        n_samples=len(state_dataset),
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        embargo=embargo,
    )
    models = make_state_classification_models(horizon=gap)
    aligned_sample_weight = align_sample_weight(sample_weight, state_dataset.index)
    oof = pd.DataFrame(index=dataset.index)
    rows: list[dict[str, float | str]] = []
    split_source = state_dataset[feature_columns] if not fold_feature_selection else state_dataset

    for model_name, template in models.items():
        log_progress(verbose, f"{label}[{task_key}] fitting {model_name}")
        probabilities = pd.DataFrame(np.nan, index=state_dataset.index, columns=[0, 1, 2], dtype=float)
        fold_count = 0

        for train_idx, test_idx in splitter.split(split_source):
            y_train = y.iloc[train_idx]
            if y_train.nunique() < 2:
                continue

            if fold_feature_selection:
                fold_features = select_fold_features(
                    dataset=state_dataset,
                    candidate_feature_columns=feature_columns,
                    train_positions=train_idx,
                    min_feature_coverage=min_feature_coverage,
                    horizon=gap,
                    target_column=target_column,
                )
                if not fold_features:
                    fold_features = feature_columns
            else:
                fold_features = feature_columns
            X_fold = state_dataset[fold_features]

            model = fit_state_classifier(
                template,
                X_fold.iloc[train_idx],
                y_train,
                sample_weight=aligned_sample_weight.iloc[train_idx] if aligned_sample_weight is not None else None,
            )
            aligned_probabilities = align_state_probability_frame(
                np.asarray(model.predict_proba(X_fold.iloc[test_idx]), dtype=float),
                getattr(model, "classes_", []),
                X_fold.iloc[test_idx].index,
            )
            probabilities.loc[X_fold.iloc[test_idx].index, [0, 1, 2]] = aligned_probabilities.loc[:, [0, 1, 2]].to_numpy()
            fold_count += 1

        valid_mask = probabilities.notna().any(axis=1)
        if not valid_mask.any() or fold_count == 0:
            continue

        valid_probabilities = probabilities.loc[valid_mask, [0, 1, 2]].copy()
        actual = y.loc[valid_probabilities.index].astype(int)
        viable_summary, _ = assess_state_target_viability(
            actual,
            task_key=task_key,
            horizon=gap,
            test_size=test_size,
            n_splits=max(2, fold_count),
        )
        if not viable_summary:
            continue
        predicted = valid_probabilities.idxmax(axis=1).astype(int)
        metrics = evaluate_multistate_classification(
            actual_label=actual.to_numpy(dtype=int),
            predicted_label=predicted.to_numpy(dtype=int),
            probability_frame=valid_probabilities,
        )
        metrics["model"] = model_name
        metrics["folds"] = float(fold_count)
        rows.append(metrics)

        oof[f"{model_name}__{task_key}_prob_negative"] = probabilities[0].reindex(dataset.index)
        oof[f"{model_name}__{task_key}_prob_neutral"] = probabilities[1].reindex(dataset.index)
        oof[f"{model_name}__{task_key}_prob_positive"] = probabilities[2].reindex(dataset.index)
        predicted_state = pd.Series(np.nan, index=state_dataset.index, dtype=float)
        predicted_state.loc[valid_mask] = valid_probabilities.idxmax(axis=1).astype(float)
        oof[f"{model_name}__{task_key}_pred_state"] = predicted_state.reindex(dataset.index)

    if not rows:
        return pd.DataFrame(), oof

    leaderboard = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "macro_f1", "roc_auc", "accuracy"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return leaderboard, oof


def backtest_state_models(
    oof_predictions: pd.DataFrame,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
    task_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | str]] = []
    equity_curves = pd.DataFrame()
    suffix = f"__{task_key}_prob_negative"
    model_names = sorted(column.replace(suffix, "") for column in oof_predictions.columns if column.endswith(suffix))
    if not model_names:
        return pd.DataFrame(), pd.DataFrame()

    for model_name in model_names:
        negative = pd.to_numeric(oof_predictions[f"{model_name}__{task_key}_prob_negative"], errors="coerce")
        neutral = pd.to_numeric(oof_predictions[f"{model_name}__{task_key}_prob_neutral"], errors="coerce")
        positive = pd.to_numeric(oof_predictions[f"{model_name}__{task_key}_prob_positive"], errors="coerce")
        valid_index = pd.concat([negative, neutral, positive], axis=1).dropna(how="all").index
        if len(valid_index) == 0:
            continue
        _, best_metrics, best_curve = choose_threshold_via_nested_backtest(
            threshold_candidates=DEFAULT_STATE_CONFIDENCE_GRID,
            signal_builder=lambda threshold, neg=negative, neu=neutral, pos=positive: state_signal_from_probabilities(
                neg,
                neu,
                pos,
                confidence_threshold=threshold,
            ),
            market_data=market_data,
            horizon=horizon,
            interval=interval,
            reference_index=valid_index,
        )
        best_metrics["model"] = model_name
        rows.append(best_metrics)
        if not best_curve.empty:
            if equity_curves.empty:
                equity_curves = best_curve[["benchmark_equity"]].rename(columns={"benchmark_equity": "benchmark_equity"})
            equity_curves[f"{model_name}_equity"] = best_curve["equity"]
            equity_curves[f"{model_name}_signal"] = best_curve["signal"]

    if not rows:
        return pd.DataFrame(), equity_curves

    leaderboard = pd.DataFrame(rows).sort_values(
        ["sharpe", "total_return", "max_drawdown"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return leaderboard, equity_curves


def extract_model_signal_thresholds(backtest_frame: pd.DataFrame) -> dict[str, float]:
    if backtest_frame.empty or "model" not in backtest_frame.columns:
        return {}

    thresholds: dict[str, float] = {}
    for _, row in backtest_frame.iterrows():
        model_name = str(row.get("model", "")).strip()
        threshold = pd.to_numeric(row.get("signal_threshold"), errors="coerce")
        if model_name and pd.notna(threshold):
            thresholds[model_name] = float(threshold)
    return thresholds


def build_benchmark_signals(reference_index: pd.Index, market_data: pd.DataFrame) -> dict[str, pd.Series]:
    benchmark_signals: dict[str, pd.Series] = {
        "buy_hold": pd.Series(1.0, index=reference_index, dtype=float),
    }

    eth_momentum_30 = np.sign(market_data["eth_close"].pct_change(30)).replace(0.0, np.nan)
    benchmark_signals["eth_momentum_30"] = eth_momentum_30.reindex(reference_index).astype(float)

    if "btc_close" in market_data.columns:
        eth_btc_strength = market_data["eth_close"].pct_change(30) - market_data["btc_close"].pct_change(30)
        benchmark_signals["eth_btc_strength_30"] = np.sign(eth_btc_strength).replace(0.0, np.nan).reindex(reference_index).astype(float)

    return benchmark_signals


def backtest_benchmark_strategies(
    reference_index: pd.Index,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(reference_index) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict[str, float | str]] = []
    equity_curves = pd.DataFrame()
    for benchmark_name, signal in build_benchmark_signals(reference_index, market_data).items():
        metrics, curve = run_signal_backtest(signal, market_data=market_data, horizon=horizon, interval=interval)
        metrics["model"] = benchmark_name
        rows.append(metrics)
        if not curve.empty:
            if equity_curves.empty:
                equity_curves = curve[["benchmark_equity"]].rename(columns={"benchmark_equity": "benchmark_equity"})
            equity_curves[f"{benchmark_name}_equity"] = curve["equity"]
            equity_curves[f"{benchmark_name}_signal"] = curve["signal"]

    leaderboard = pd.DataFrame(rows).sort_values(
        ["sharpe", "total_return", "max_drawdown"],
        ascending=[False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return leaderboard, equity_curves


def build_recent_holdout_report(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
    cv_test_size: int,
    sample_weight: pd.Series | None = None,
    regression_thresholds: dict[str, float] | None = None,
    classification_thresholds: dict[str, float] | None = None,
    regression_ensemble_members: list[str] | None = None,
    classification_ensemble_members: list[str] | None = None,
) -> pd.DataFrame:
    holdout_rows = min(
        max(DEFAULT_RECENT_HOLDOUT_ROWS, cv_test_size * 4, horizon * 3),
        max(len(dataset) // 5, cv_test_size * 2),
    )
    minimum_train_rows = max(240, holdout_rows + horizon + 30)
    if len(dataset) < minimum_train_rows:
        return pd.DataFrame(
            columns=[
                "task",
                "model",
                "train_rows",
                "evaluation_rows",
                "embargo_rows",
                "validation_scheme",
                "signal_threshold",
                "price_rmse",
                "price_mae",
                "directional_accuracy",
                "balanced_accuracy",
                "f1",
                "roc_auc",
                "brier_score",
                "component_models",
                "total_return",
                "benchmark_return",
                "sharpe",
                "max_drawdown",
            ]
        )

    split_point = len(dataset) - holdout_rows
    holdout_embargo_rows = max(int(horizon), 1)
    train_stop = max(split_point - holdout_embargo_rows, 0)
    train_dataset = dataset.iloc[:train_stop].copy()
    holdout_dataset = dataset.iloc[split_point:].copy()
    if train_dataset.empty or holdout_dataset.empty or len(train_dataset) < max(240, horizon * 12):
        return pd.DataFrame(
            columns=[
                "task",
                "model",
                "train_rows",
                "evaluation_rows",
                "embargo_rows",
                "validation_scheme",
                "signal_threshold",
                "price_rmse",
                "price_mae",
                "directional_accuracy",
                "balanced_accuracy",
                "f1",
                "roc_auc",
                "brier_score",
                "component_models",
                "total_return",
                "benchmark_return",
                "sharpe",
                "max_drawdown",
            ]
        )
    aligned_sample_weight = align_sample_weight(sample_weight, dataset.index)
    train_sample_weight = aligned_sample_weight.iloc[:train_stop] if aligned_sample_weight is not None else None
    rows: list[dict[str, Any]] = []
    regression_holdout_predictions: dict[str, pd.Series] = {}
    classification_holdout_probabilities: dict[str, pd.Series] = {}
    validation_scheme = "recent_holdout_purged"

    for model_name, template in make_models(horizon=horizon).items():
        model = clone(template)
        fit_model_with_optional_sample_weight(
            model,
            train_dataset[feature_columns],
            train_dataset["target_return"],
            train_sample_weight,
        )
        predictions = pd.Series(model.predict(holdout_dataset[feature_columns]), index=holdout_dataset.index)
        if not overlay_disabled_for_context("holdout"):
            predictions = apply_regime_response_overlay(
                predictions,
                holdout_dataset[feature_columns],
                horizon=horizon,
            )
        regression_holdout_predictions[model_name] = predictions.copy()
        metrics = evaluate_predictions(
            current_close=holdout_dataset["eth_close"].to_numpy(),
            actual_return=holdout_dataset["target_return"].to_numpy(),
            predicted_return=predictions.to_numpy(),
        )
        selected_threshold = float((regression_thresholds or {}).get(model_name, 0.0))
        signal = regression_signal_from_predictions(predictions, threshold=selected_threshold)
        best_backtest, _ = run_signal_backtest(
            signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        rows.append(
            {
                "task": "regression",
                "model": model_name,
                "train_rows": int(len(train_dataset)),
                "evaluation_rows": int(len(holdout_dataset)),
                "embargo_rows": int(holdout_embargo_rows),
                "validation_scheme": validation_scheme,
                "signal_threshold": selected_threshold,
                "price_rmse": metrics["price_rmse"],
                "price_mae": metrics["price_mae"],
                "directional_accuracy": metrics["directional_accuracy"],
                "balanced_accuracy": float("nan"),
                "f1": float("nan"),
                "roc_auc": float("nan"),
                "brier_score": float("nan"),
                "component_models": "",
                "total_return": best_backtest["total_return"],
                "benchmark_return": best_backtest["benchmark_return"],
                "sharpe": best_backtest["sharpe"],
                "max_drawdown": best_backtest["max_drawdown"],
            }
        )

    available_regression_members = [
        model_name
        for model_name in (regression_ensemble_members or [])
        if model_name in regression_holdout_predictions
    ]
    if len(available_regression_members) >= 2:
        regression_skill_board = pd.DataFrame(
            [row for row in rows if row.get("task") == "regression"]
        )
        regression_weights = derive_ensemble_weights_from_leaderboard(
            regression_skill_board, available_regression_members, "price_rmse", higher_is_better=False
        )
        ensemble_predictions = skill_weighted_trimmed_mean(
            pd.DataFrame(
                {
                    model_name: regression_holdout_predictions[model_name]
                    for model_name in available_regression_members
                }
            ),
            weights=regression_weights,
        )
        selected_threshold = float((regression_thresholds or {}).get(TRIMMED_REGRESSION_ENSEMBLE_MODEL, 0.0))
        signal = regression_signal_from_predictions(ensemble_predictions, threshold=selected_threshold)
        best_backtest, _ = run_signal_backtest(
            signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        metrics = evaluate_predictions(
            current_close=holdout_dataset["eth_close"].to_numpy(),
            actual_return=holdout_dataset["target_return"].to_numpy(),
            predicted_return=ensemble_predictions.to_numpy(),
        )
        rows.append(
            {
                "task": "regression",
                "model": TRIMMED_REGRESSION_ENSEMBLE_MODEL,
                "train_rows": int(len(train_dataset)),
                "evaluation_rows": int(len(holdout_dataset)),
                "embargo_rows": int(holdout_embargo_rows),
                "validation_scheme": validation_scheme,
                "signal_threshold": selected_threshold,
                "price_rmse": metrics["price_rmse"],
                "price_mae": metrics["price_mae"],
                "directional_accuracy": metrics["directional_accuracy"],
                "balanced_accuracy": float("nan"),
                "f1": float("nan"),
                "roc_auc": float("nan"),
                "brier_score": float("nan"),
                "component_models": "|".join(available_regression_members),
                "total_return": best_backtest["total_return"],
                "benchmark_return": best_backtest["benchmark_return"],
                "sharpe": best_backtest["sharpe"],
                "max_drawdown": best_backtest["max_drawdown"],
            }
        )

    actual_labels = (holdout_dataset["target_return"] > 0).astype(int)
    for model_name, template in make_classification_models(horizon=horizon).items():
        model = fit_calibrated_classifier(
            template,
            train_dataset[feature_columns],
            (train_dataset["target_return"] > 0).astype(int),
            sample_weight=train_sample_weight,
        )
        probabilities = pd.Series(
            model.predict_proba(holdout_dataset[feature_columns])[:, 1],
            index=holdout_dataset.index,
            dtype=float,
        )
        probabilities = apply_direction_regime_overlay(
            probabilities,
            holdout_dataset[feature_columns],
            horizon=horizon,
        )
        classification_holdout_probabilities[model_name] = probabilities.copy()
        selected_threshold = float((classification_thresholds or {}).get(model_name, DEFAULT_CLASSIFICATION_SIGNAL_THRESHOLD))
        _, effective_threshold = resolve_classification_signal_thresholds(selected_threshold)
        classification_metrics = evaluate_direction_classification(
            actual_label=actual_labels.to_numpy(),
            predicted_label=(probabilities >= effective_threshold).astype(int).to_numpy(),
            probability_up=probabilities.to_numpy(),
        )
        signal = classification_signal_from_probabilities(probabilities, threshold=selected_threshold)
        best_backtest, _ = run_signal_backtest(
            signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        rows.append(
            {
                "task": "classification",
                "model": model_name,
                "train_rows": int(len(train_dataset)),
                "evaluation_rows": int(len(holdout_dataset)),
                "embargo_rows": int(holdout_embargo_rows),
                "validation_scheme": validation_scheme,
                "signal_threshold": float(effective_threshold),
                "price_rmse": float("nan"),
                "price_mae": float("nan"),
                "directional_accuracy": float("nan"),
                "balanced_accuracy": classification_metrics["balanced_accuracy"],
                "f1": classification_metrics["f1"],
                "roc_auc": classification_metrics["roc_auc"],
                "brier_score": classification_metrics["brier_score"],
                "component_models": "",
                "total_return": best_backtest["total_return"],
                "benchmark_return": best_backtest["benchmark_return"],
                "sharpe": best_backtest["sharpe"],
                "max_drawdown": best_backtest["max_drawdown"],
            }
        )

    available_classification_members = [
        model_name
        for model_name in (classification_ensemble_members or [])
        if model_name in classification_holdout_probabilities
    ]
    if len(available_classification_members) >= 2:
        classification_skill_board = pd.DataFrame(
            [row for row in rows if row.get("task") == "classification"]
        )
        classification_weights = derive_ensemble_weights_from_leaderboard(
            classification_skill_board,
            available_classification_members,
            "brier_score",
            higher_is_better=False,
        )
        ensemble_probability = skill_weighted_trimmed_mean(
            pd.DataFrame(
                {
                    model_name: classification_holdout_probabilities[model_name]
                    for model_name in available_classification_members
                }
            ),
            weights=classification_weights,
        ).clip(lower=0.0, upper=1.0)
        selected_threshold = float(
            (classification_thresholds or {}).get(
                TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL,
                DEFAULT_CLASSIFICATION_SIGNAL_THRESHOLD,
            )
        )
        _, effective_threshold = resolve_classification_signal_thresholds(selected_threshold)
        classification_metrics = evaluate_direction_classification(
            actual_label=actual_labels.to_numpy(),
            predicted_label=(ensemble_probability >= effective_threshold).astype(int).to_numpy(),
            probability_up=ensemble_probability.to_numpy(),
        )
        signal = classification_signal_from_probabilities(ensemble_probability, threshold=selected_threshold)
        best_backtest, _ = run_signal_backtest(
            signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        rows.append(
            {
                "task": "classification",
                "model": TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL,
                "train_rows": int(len(train_dataset)),
                "evaluation_rows": int(len(holdout_dataset)),
                "embargo_rows": int(holdout_embargo_rows),
                "validation_scheme": validation_scheme,
                "signal_threshold": float(effective_threshold),
                "price_rmse": float("nan"),
                "price_mae": float("nan"),
                "directional_accuracy": float("nan"),
                "balanced_accuracy": classification_metrics["balanced_accuracy"],
                "f1": classification_metrics["f1"],
                "roc_auc": classification_metrics["roc_auc"],
                "brier_score": classification_metrics["brier_score"],
                "component_models": "|".join(available_classification_members),
                "total_return": best_backtest["total_return"],
                "benchmark_return": best_backtest["benchmark_return"],
                "sharpe": best_backtest["sharpe"],
                "max_drawdown": best_backtest["max_drawdown"],
            }
        )

    benchmark_backtest, _ = backtest_benchmark_strategies(
        reference_index=holdout_dataset.index,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    if not benchmark_backtest.empty:
        benchmark_rows = benchmark_backtest.copy()
        benchmark_rows["task"] = "benchmark"
        benchmark_rows["train_rows"] = int(len(train_dataset))
        benchmark_rows["evaluation_rows"] = int(len(holdout_dataset))
        benchmark_rows["embargo_rows"] = int(holdout_embargo_rows)
        benchmark_rows["validation_scheme"] = validation_scheme
        benchmark_rows["signal_threshold"] = float("nan")
        benchmark_rows["price_rmse"] = float("nan")
        benchmark_rows["price_mae"] = float("nan")
        benchmark_rows["directional_accuracy"] = float("nan")
        benchmark_rows["balanced_accuracy"] = float("nan")
        benchmark_rows["f1"] = float("nan")
        benchmark_rows["roc_auc"] = float("nan")
        benchmark_rows["brier_score"] = float("nan")
        benchmark_rows["component_models"] = ""
        rows.extend(benchmark_rows.to_dict(orient="records"))

    report = pd.DataFrame(rows)
    return report.sort_values(
        ["task", "sharpe", "total_return", "model"],
        ascending=[True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _normalize_selection_metric(
    frame: pd.DataFrame,
    column: str,
    *,
    higher_is_better: bool = True,
    absolute: bool = False,
) -> pd.Series:
    scores = pd.Series(0.0, index=frame.index, dtype=float)
    if frame.empty or column not in frame.columns:
        return scores
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if absolute:
        values = values.abs()
    valid = values.notna()
    if not valid.any():
        return scores
    min_value = float(values.loc[valid].min())
    max_value = float(values.loc[valid].max())
    if not np.isfinite(min_value) or not np.isfinite(max_value) or abs(max_value - min_value) < 1e-9:
        scores.loc[valid] = 0.5
        return scores
    normalized = (values.loc[valid] - min_value) / (max_value - min_value)
    if not higher_is_better:
        normalized = 1.0 - normalized
    scores.loc[valid] = normalized.astype(float)
    return scores


def _weighted_available_average(
    items: list[tuple[float, float]],
    *,
    lower: float,
    upper: float,
) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for weight, value in items:
        if pd.isna(value):
            continue
        total_weight += float(weight)
        weighted_sum += float(weight) * float(value)
    if total_weight <= 0.0:
        return 0.0
    return float(np.clip(weighted_sum / total_weight, lower, upper))


def derive_macro_selection_context(
    latest_features: pd.Series | None,
    horizon: int,
) -> dict[str, Any]:
    if latest_features is None or len(latest_features) == 0:
        return {
            "risk_regime": "BALANCED",
            "directional_bias": "FLAT",
            "selection_tag": "macro_balanced",
            "risk_on_score": 0.0,
            "risk_off_score": 0.0,
            "macro_spread": 0.0,
            "trend_bias_score": 0.0,
            "volatility_stress": 0.0,
            "tail_event_pressure": 0.0,
            "risk_on_tilt": 0.0,
            "risk_off_tilt": 0.0,
        }

    latest = pd.to_numeric(pd.Series(latest_features), errors="coerce")

    def signed(column: str, scale: float) -> float:
        value = pd.to_numeric(pd.Series([latest.get(column, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float("nan")
        return float(np.clip(float(value) / max(scale, 1e-6), -1.0, 1.0))

    def positive(column: str, scale: float) -> float:
        value = signed(column, scale)
        if pd.isna(value):
            return float("nan")
        return float(np.clip(value, 0.0, 1.0))

    def negative(column: str, scale: float) -> float:
        value = signed(column, scale)
        if pd.isna(value):
            return float("nan")
        return float(np.clip(-value, 0.0, 1.0))

    def magnitude(column: str, scale: float) -> float:
        value = pd.to_numeric(pd.Series([latest.get(column, np.nan)]), errors="coerce").iloc[0]
        if pd.isna(value):
            return float("nan")
        return float(np.clip(abs(float(value)) / max(scale, 1e-6), 0.0, 1.0))

    risk_off_score = _weighted_available_average(
        [
            (0.15, positive("macro_tightening_20", 1.80)),
            (0.10, positive("fred_bbb_oas_z_90", 2.50)),
            (0.08, positive("risk_aversion_spread_5", 2.50)),
            (0.08, positive("dxy_return_20", 0.06)),
            (0.08, positive("tnx_return_20", 0.08)),
            (0.08, positive("oil_vol_20", 0.10)),
            (0.10, negative("btc_drawdown_60", 0.25)),
            (0.10, negative("eth_drawdown_30", 0.20)),
            (0.08, negative("eth_drawdown_180", 0.45)),
            (0.06, negative("btc_drawdown_180", 0.40)),
            (0.08, negative("equity_risk_on_20", 0.16)),
            (0.07, negative("eth_vs_btc_strength_14", 0.10)),
            (0.05, negative("eth_vs_btc_strength_90", 0.20)),
            (0.05, positive("eth_vol_30_180_ratio", 0.75)),
            (0.08, negative("fred_net_liquidity_z_90", 2.50)),
        ],
        lower=0.0,
        upper=1.0,
    )
    risk_on_score = _weighted_available_average(
        [
            (0.15, positive("equity_risk_on_20", 0.16)),
            (0.12, positive("eth_vs_btc_strength_14", 0.10)),
            (0.10, positive("sentiment_fear_greed_centered", 1.00)),
            (0.10, positive("fred_net_liquidity_z_90", 2.50)),
            (0.10, positive("eth_ma_30_ratio", 0.16)),
            (0.08, positive("eth_ma_180_ratio", 0.35)),
            (0.08, positive("eth_return_90", 0.45)),
            (0.06, positive("eth_return_180", 0.75)),
            (0.06, positive("eth_cycle_position_365_centered", 0.50)),
            (0.06, positive("eth_vs_btc_strength_90", 0.20)),
            (0.08, positive("eth_vwap_20_ratio", 0.18)),
            (0.08, positive("artemis_stablecoin_total_supply_z_30", 2.50)),
            (0.07, positive("artemis_non_sybil_users_z_90", 2.50)),
            (0.05, negative("macro_tightening_20", 1.80)),
            (0.05, negative("fred_bbb_oas_z_90", 2.50)),
        ],
        lower=0.0,
        upper=1.0,
    )
    trend_bias_score = _weighted_available_average(
        [
            (0.18, signed("eth_return_7", 0.12)),
            (0.14, signed("eth_return_14", 0.18)),
            (0.12, signed("eth_return_60", 0.35)),
            (0.10, signed("eth_return_180", 0.75)),
            (0.14, signed("eth_ma_14_ratio", 0.12)),
            (0.12, signed("eth_ma_30_ratio", 0.16)),
            (0.10, signed("eth_ma_180_ratio", 0.35)),
            (0.08, signed("eth_cycle_position_365_centered", 0.50)),
            (0.10, signed("eth_vwap_20_ratio", 0.18)),
            (0.10, signed("eth_vs_btc_strength_14", 0.10)),
            (0.08, signed("eth_vs_btc_strength_90", 0.20)),
            (0.10, signed("equity_risk_on_20", 0.16)),
            (0.07, negative("macro_tightening_20", 1.80)),
            (0.05, negative("btc_drawdown_60", 0.25)),
        ],
        lower=-1.0,
        upper=1.0,
    )
    volatility_stress = _weighted_available_average(
        [
            (0.20, magnitude("eth_vol_7", 0.040)),
            (0.18, magnitude("eth_vol_14", 0.040)),
            (0.18, magnitude("eth_vol_30", 0.050)),
            (0.12, positive("eth_tail_event_pressure", 1.50)),
            (0.08, magnitude("eth_range_expansion_z_30", 2.00)),
            (0.06, magnitude("eth_volume_impulse_3_30", 1.00)),
            (0.14, magnitude("oil_vol_20", 0.100)),
            (0.15, magnitude("eth_range_pct", 0.080)),
            (0.15, magnitude("eth_close_open_pct", 0.050)),
        ],
        lower=0.0,
        upper=1.0,
    )
    tail_event_pressure = _weighted_available_average(
        [
            (0.32, positive("eth_tail_event_pressure", 1.50)),
            (0.20, positive("eth_range_expansion_z_30", 2.00)),
            (0.18, positive("eth_volume_impulse_3_30", 1.00)),
            (0.16, positive("eth_vol_regime_7_30", 0.75)),
            (0.14, negative("eth_vol_squeeze_20_90", 0.45)),
        ],
        lower=0.0,
        upper=1.0,
    )
    macro_spread = float(risk_on_score - risk_off_score)
    if macro_spread >= 0.10:
        risk_regime = "RISK_ON"
    elif macro_spread <= -0.10:
        risk_regime = "RISK_OFF"
    else:
        risk_regime = "BALANCED"

    bias_floor = 0.10 if horizon <= 7 else 0.08
    if trend_bias_score >= bias_floor:
        directional_bias = "UP"
    elif trend_bias_score <= -bias_floor:
        directional_bias = "DOWN"
    else:
        directional_bias = "FLAT"

    risk_off_tilt = float(
        np.clip(max(risk_off_score - risk_on_score, 0.0) + 0.35 * volatility_stress, 0.0, 1.0)
    )
    risk_on_tilt = float(
        np.clip(max(risk_on_score - risk_off_score, 0.0) + 0.20 * max(trend_bias_score, 0.0), 0.0, 1.0)
    )
    selection_tag = {
        "RISK_ON": "macro_risk_on",
        "RISK_OFF": "macro_risk_off",
        "BALANCED": "macro_balanced",
    }[risk_regime]
    return {
        "risk_regime": risk_regime,
        "directional_bias": directional_bias,
        "selection_tag": selection_tag,
        "risk_on_score": risk_on_score,
        "risk_off_score": risk_off_score,
        "macro_spread": macro_spread,
        "trend_bias_score": trend_bias_score,
        "volatility_stress": volatility_stress,
        "tail_event_pressure": tail_event_pressure,
        "risk_on_tilt": risk_on_tilt,
        "risk_off_tilt": risk_off_tilt,
    }


def summarize_effective_macro_context(
    latest_features: pd.Series | None,
    horizon: int,
) -> dict[str, Any]:
    context = derive_macro_selection_context(latest_features, horizon)
    latest = pd.to_numeric(pd.Series(latest_features), errors="coerce") if latest_features is not None else pd.Series(dtype=float)
    used_columns = [
        column for column in MACRO_CONTEXT_FEATURE_COLUMNS if column in latest.index and pd.notna(latest.get(column))
    ]
    missing_columns = [
        column for column in MACRO_CONTEXT_FEATURE_COLUMNS if column not in latest.index or pd.isna(latest.get(column))
    ]
    if len(used_columns) >= max(8, int(len(MACRO_CONTEXT_FEATURE_COLUMNS) * 0.70)):
        readiness_status = "ready"
    elif used_columns:
        readiness_status = "partial"
    else:
        readiness_status = "missing"
    return {
        **context,
        "readiness_status": readiness_status,
        "used_feature_count": int(len(used_columns)),
        "missing_feature_count": int(len(missing_columns)),
        "used_features": "|".join(used_columns),
        "missing_features": "|".join(missing_columns),
    }


def parse_component_models(value: Any) -> list[str]:
    if value is None:
        return []
    raw_text = str(value).strip()
    if not raw_text:
        return []
    return [part.strip() for part in raw_text.split("|") if str(part).strip()]


def trimmed_equal_weight_average(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    averaged = np.full(len(numeric.index), np.nan, dtype=float)
    for row_number, (_, row) in enumerate(numeric.iterrows()):
        clean = row.to_numpy(dtype=float)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            continue
        clean.sort()
        trim_count = 1 if clean.size >= 4 else 0
        if clean.size - 2 * trim_count <= 0:
            trim_count = 0
        trimmed = clean[trim_count: clean.size - trim_count] if trim_count else clean
        averaged[row_number] = float(np.mean(trimmed))
    return pd.Series(averaged, index=numeric.index, dtype=float)


def skill_weighted_trimmed_mean(
    frame: pd.DataFrame,
    weights: Mapping[str, float] | None = None,
) -> pd.Series:
    """Per-row trimmed weighted mean over model prediction columns.

    Uses the same extreme-value trimming as ``trimmed_equal_weight_average``
    (drop 1 top + 1 bottom when >=4 finite values are present). Survivors are
    combined via the supplied per-model weights. Weights are rescaled to sum
    to 1 across the survivors each row, so missing/infinite values in one
    model do not distort the blend. ``weights=None`` yields behaviour
    identical to ``trimmed_equal_weight_average``.
    """
    if frame.empty:
        return pd.Series(index=frame.index, dtype=float)
    numeric = frame.apply(pd.to_numeric, errors="coerce")
    columns = list(numeric.columns)
    if weights is None:
        weight_vector = np.ones(len(columns), dtype=float)
    else:
        weight_vector = np.asarray(
            [float(max(weights.get(col, 0.0), 0.0)) for col in columns], dtype=float
        )
    if not np.any(weight_vector > 0):
        weight_vector = np.ones(len(columns), dtype=float)
    averaged = np.full(len(numeric.index), np.nan, dtype=float)
    values_matrix = numeric.to_numpy(dtype=float, copy=False)
    for row_number in range(len(numeric.index)):
        row_values = values_matrix[row_number]
        finite_mask = np.isfinite(row_values)
        if not finite_mask.any():
            continue
        clean_values = row_values[finite_mask]
        clean_weights = weight_vector[finite_mask]
        order = np.argsort(clean_values)
        clean_values = clean_values[order]
        clean_weights = clean_weights[order]
        trim_count = 1 if clean_values.size >= 4 else 0
        if clean_values.size - 2 * trim_count <= 0:
            trim_count = 0
        if trim_count:
            clean_values = clean_values[trim_count: clean_values.size - trim_count]
            clean_weights = clean_weights[trim_count: clean_weights.size - trim_count]
        weight_sum = float(clean_weights.sum())
        if weight_sum <= 0:
            averaged[row_number] = float(np.mean(clean_values))
        else:
            averaged[row_number] = float(np.sum(clean_values * clean_weights) / weight_sum)
    return pd.Series(averaged, index=numeric.index, dtype=float)


def derive_ensemble_weights_from_leaderboard(
    leaderboard: pd.DataFrame | None,
    member_models: list[str],
    metric_column: str,
    higher_is_better: bool,
) -> dict[str, float]:
    """Map leaderboard metric values to normalized ensemble weights.

    For lower-is-better metrics (price_rmse, brier_score) weight = 1 / (value + eps).
    For higher-is-better metrics (balanced_accuracy, roc_auc) weight is shifted
    so the worst finite member is near zero and the best gets the highest
    weight. Missing / non-finite values fall back to the mean of finite peers,
    and a fully-degenerate leaderboard yields uniform weights.
    """
    if not member_models:
        return {}
    if (
        leaderboard is None
        or leaderboard.empty
        or "model" not in leaderboard.columns
        or metric_column not in leaderboard.columns
    ):
        uniform = 1.0 / len(member_models)
        return {model_name: uniform for model_name in member_models}

    indexed = leaderboard.drop_duplicates(subset=["model"], keep="first").set_index("model")
    raw_values: dict[str, float] = {}
    for model_name in member_models:
        if model_name in indexed.index:
            numeric = pd.to_numeric(indexed.loc[model_name, metric_column], errors="coerce")
        else:
            numeric = float("nan")
        raw_values[model_name] = float(numeric) if pd.notna(numeric) else float("nan")

    finite_values = [value for value in raw_values.values() if np.isfinite(value)]
    if not finite_values:
        uniform = 1.0 / len(member_models)
        return {model_name: uniform for model_name in member_models}

    fallback = float(np.mean(finite_values))
    weights: dict[str, float] = {}
    if higher_is_better:
        floor_value = float(min(finite_values)) - 1e-6
        for model_name, value in raw_values.items():
            if not np.isfinite(value):
                value = fallback
            weights[model_name] = max(float(value - floor_value), 1e-6)
    else:
        for model_name, value in raw_values.items():
            if not np.isfinite(value):
                value = fallback
            weights[model_name] = 1.0 / (float(max(value, 0.0)) + 1e-6)
    total = float(sum(weights.values()))
    if total <= 0:
        uniform = 1.0 / len(member_models)
        return {model_name: uniform for model_name in member_models}
    return {model_name: float(weight / total) for model_name, weight in weights.items()}


def _greedy_diversify_members(
    skill_ordered_candidates: list[str],
    oof_predictions: pd.DataFrame | None,
    prediction_column_suffix: str,
    max_members: int,
    skill_window: int = 3,
    min_overlap: int = 20,
) -> list[str]:
    # Phase 4: once skill has been established (candidates arrive sorted best
    # first), pick subsequent slots to minimise pairwise OOF correlation
    # against already-selected members. Preserves skill primacy - the window
    # only scans the next few candidates by skill rank - but breaks the
    # "stack of near-duplicate models" failure that emerged when newly-added
    # Phase 3 features shifted fold rankings. If OOF predictions are absent
    # or under-overlapping, fall back to pure skill order.
    seen: set[str] = set()
    unique_candidates: list[str] = []
    for name in skill_ordered_candidates:
        if name not in seen:
            seen.add(name)
            unique_candidates.append(name)
    if len(unique_candidates) <= max_members:
        return unique_candidates[:max_members]
    if oof_predictions is None or oof_predictions.empty:
        return unique_candidates[:max_members]

    selected: list[str] = [unique_candidates[0]]
    remaining: list[str] = unique_candidates[1:]
    while remaining and len(selected) < max_members:
        window = remaining[: max(skill_window, 1)]
        selected_cols = [f"{name}{prediction_column_suffix}" for name in selected]
        missing_selected = [col for col in selected_cols if col not in oof_predictions.columns]
        if missing_selected:
            next_pick = window[0]
        else:
            scores: list[tuple[float, int, str]] = []
            for rank_idx, name in enumerate(window):
                candidate_col = f"{name}{prediction_column_suffix}"
                if candidate_col not in oof_predictions.columns:
                    continue
                candidate_series = pd.to_numeric(oof_predictions[candidate_col], errors="coerce")
                corrs: list[float] = []
                for sel_col in selected_cols:
                    aligned = pd.concat(
                        [candidate_series, pd.to_numeric(oof_predictions[sel_col], errors="coerce")],
                        axis=1,
                    ).dropna()
                    if len(aligned) < min_overlap:
                        continue
                    corr = aligned.iloc[:, 0].corr(aligned.iloc[:, 1])
                    if pd.notna(corr):
                        corrs.append(abs(float(corr)))
                mean_corr = float(np.mean(corrs)) if corrs else float("nan")
                scores.append((mean_corr, rank_idx, name))
            if not scores:
                next_pick = window[0]
            else:
                scores.sort(
                    key=lambda item: (
                        item[0] if not pd.isna(item[0]) else float("inf"),
                        item[1],
                    )
                )
                next_pick = scores[0][2]
        remaining.remove(next_pick)
        selected.append(next_pick)
    return selected


def select_trimmed_regression_ensemble_members(
    leaderboard: pd.DataFrame,
    horizon: int,
    max_members: int | None = None,
    oof_predictions: pd.DataFrame | None = None,
) -> list[str]:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return []
    if max_members is None:
        max_members = 3 if horizon <= 7 else 4
    frame = leaderboard.loc[
        leaderboard["model"].astype(str) != TRIMMED_REGRESSION_ENSEMBLE_MODEL
    ].copy()
    if frame.empty:
        return []
    rmse_column = "price_rmse" if "price_rmse" in frame.columns else "cv_price_rmse"
    mae_column = "price_mae" if "price_mae" in frame.columns else "cv_price_mae"
    directional_column = (
        "directional_accuracy" if "directional_accuracy" in frame.columns else "cv_directional_accuracy"
    )
    best_rmse = pd.to_numeric(frame.get(rmse_column), errors="coerce").min()
    best_directional = pd.to_numeric(frame.get(directional_column), errors="coerce").max()
    if pd.notna(best_rmse):
        # Phase 2.5: tighter tolerance so the blend is only built from peers
        # close to the best single model. Previous values (1.18 / 1.30) diluted
        # the ensemble with materially worse members.
        tolerance = 1.08 if horizon <= 7 else 1.15
        frame = frame.loc[
            pd.to_numeric(frame.get(rmse_column), errors="coerce").fillna(float("inf")) <= best_rmse * tolerance
        ].copy()
    if pd.notna(best_directional):
        directional_floor = max(float(best_directional) - (0.08 if horizon <= 7 else 0.12), 0.30)
        frame = frame.loc[
            pd.to_numeric(frame.get(directional_column), errors="coerce").fillna(float("-inf")) >= directional_floor
        ].copy()
    if frame.empty:
        frame = leaderboard.loc[
            leaderboard["model"].astype(str) != TRIMMED_REGRESSION_ENSEMBLE_MODEL
        ].copy()
    frame = frame.sort_values(
        [rmse_column, mae_column, directional_column],
        ascending=[True, True, False],
        na_position="last",
    ).reset_index(drop=True)
    pool_size = max_members + 2
    skill_ordered = [str(model_name) for model_name in frame["model"].tolist()[:pool_size]]
    members = _greedy_diversify_members(
        skill_ordered_candidates=skill_ordered,
        oof_predictions=oof_predictions,
        prediction_column_suffix="_pred_return",
        max_members=max_members,
    )
    if len(members) < 2:
        fallback = leaderboard.loc[
            leaderboard["model"].astype(str) != TRIMMED_REGRESSION_ENSEMBLE_MODEL,
            "model",
        ].astype(str).tolist()[:2]
        members = fallback
    return members if len(members) >= 2 else []


def select_trimmed_classification_ensemble_members(
    leaderboard: pd.DataFrame,
    horizon: int,
    max_members: int | None = None,
    oof_predictions: pd.DataFrame | None = None,
) -> list[str]:
    if leaderboard.empty or "model" not in leaderboard.columns:
        return []
    if max_members is None:
        max_members = 3 if horizon <= 7 else 4
    frame = leaderboard.loc[
        leaderboard["model"].astype(str) != TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL
    ].copy()
    if frame.empty:
        return []
    balanced_column = "balanced_accuracy" if "balanced_accuracy" in frame.columns else "cv_balanced_accuracy"
    roc_column = "roc_auc" if "roc_auc" in frame.columns else "cv_roc_auc"
    f1_column = "f1" if "f1" in frame.columns else "cv_f1"
    brier_column = "brier_score" if "brier_score" in frame.columns else "cv_brier_score"
    threshold_column = "signal_threshold" if "signal_threshold" in frame.columns else "backtest_signal_threshold"
    best_bal = pd.to_numeric(frame.get(balanced_column), errors="coerce").max()
    best_auc = pd.to_numeric(frame.get(roc_column), errors="coerce").max()
    best_brier = pd.to_numeric(frame.get(brier_column), errors="coerce").min()
    if pd.notna(best_bal):
        # Phase 2.5: tighter balanced-accuracy and roc-auc thresholds so ensemble
        # members must track the best single model's direction skill.
        frame = frame.loc[
            pd.to_numeric(frame.get(balanced_column), errors="coerce").fillna(float("-inf"))
            >= max(float(best_bal) - (0.06 if horizon <= 7 else 0.08), 0.40)
        ].copy()
    if pd.notna(best_auc):
        frame = frame.loc[
            pd.to_numeric(frame.get(roc_column), errors="coerce").fillna(float("-inf"))
            >= max(float(best_auc) - (0.10 if horizon <= 7 else 0.14), 0.20)
        ].copy()
    if pd.notna(best_brier):
        # Phase 2.5 addition: also require Brier score within tolerance of the
        # best single — rejects models that happen to hit high AUC while being
        # poorly-calibrated, which drag the probability blend.
        brier_tolerance = 1.12 if horizon <= 7 else 1.20
        frame = frame.loc[
            pd.to_numeric(frame.get(brier_column), errors="coerce").fillna(float("inf"))
            <= float(best_brier) * brier_tolerance
        ].copy()
    if frame.empty:
        frame = leaderboard.loc[
            leaderboard["model"].astype(str) != TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL
        ].copy()
    frame = frame.sort_values(
        [balanced_column, roc_column, f1_column, threshold_column],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    pool_size = max_members + 2
    skill_ordered = [str(model_name) for model_name in frame["model"].tolist()[:pool_size]]
    members = _greedy_diversify_members(
        skill_ordered_candidates=skill_ordered,
        oof_predictions=oof_predictions,
        prediction_column_suffix="_prob_up",
        max_members=max_members,
    )
    if len(members) < 2:
        fallback = leaderboard.loc[
            leaderboard["model"].astype(str) != TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL,
            "model",
        ].astype(str).tolist()[:2]
        members = fallback
    return members if len(members) >= 2 else []


def build_historical_macro_bucket_series(
    feature_frame: pd.DataFrame,
    horizon: int,
) -> pd.Series:
    if feature_frame.empty:
        return pd.Series(dtype=object)
    bucket_values = [
        derive_macro_selection_context(feature_frame.loc[index], horizon).get("risk_regime", "BALANCED")
        for index in feature_frame.index
    ]
    return pd.Series(bucket_values, index=feature_frame.index, dtype="object", name="macro_bucket")


def merge_selection_backtest_metrics(
    base_frame: pd.DataFrame,
    backtest_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if backtest_frame is None or backtest_frame.empty or base_frame.empty:
        return base_frame
    renamed = backtest_frame.rename(
        columns={
            "total_return": "backtest_total_return",
            "benchmark_return": "backtest_benchmark_return",
            "sharpe": "backtest_sharpe",
            "signal_threshold": "backtest_signal_threshold",
            "max_drawdown": "backtest_max_drawdown",
        }
    )
    desired_columns = [
        "model",
        "backtest_total_return",
        "backtest_benchmark_return",
        "backtest_sharpe",
        "backtest_signal_threshold",
        "backtest_max_drawdown",
        "macro_bucket_label",
        "macro_bucket_total_return",
        "macro_bucket_sharpe",
        "macro_bucket_max_drawdown",
        "macro_bucket_signal_count",
        "macro_bucket_benchmark_return",
        "macro_bucket_hit_rate",
    ]
    available_columns = [column for column in desired_columns if column in renamed.columns]
    return base_frame.merge(renamed[available_columns], on="model", how="left")


def merge_prediction_feedback_metrics(
    frame: pd.DataFrame,
    prediction_feedback_summary: pd.DataFrame | None,
    *,
    task_name: str,
    horizon: int | None = None,
) -> pd.DataFrame:
    if frame.empty or prediction_feedback_summary is None or prediction_feedback_summary.empty or "model" not in frame.columns:
        return frame
    feedback = prediction_feedback_summary.copy()
    if "task" not in feedback.columns or "model_name" not in feedback.columns:
        return frame
    feedback = feedback.loc[feedback["task"].astype(str) == str(task_name)].copy()
    if horizon is not None and "horizon_steps" in feedback.columns:
        feedback = feedback.loc[
            pd.to_numeric(feedback["horizon_steps"], errors="coerce") == float(horizon)
        ].copy()
    if feedback.empty:
        return frame
    desired_columns = [
        "model_name",
        "feedback_count",
        "feedback_weight_sum",
        "feedback_last_target_timestamp",
        "feedback_weighted_mean_abs_error_return",
        "feedback_weighted_mean_abs_error_close",
        "feedback_weighted_rmse_return",
        "feedback_weighted_directional_accuracy",
        "feedback_weighted_mean_strategy_return",
        "feedback_weighted_mean_excess_return",
        "feedback_weighted_brier_score",
        "feedback_weighted_interval_coverage",
        "feedback_total_strategy_return",
        "feedback_total_benchmark_return",
        "feedback_total_excess_return",
    ]
    available_columns = [column for column in desired_columns if column in feedback.columns]
    if "model_name" not in available_columns:
        return frame
    renamed_feedback = feedback[available_columns].rename(columns={"model_name": "model"})
    duplicate_columns = [column for column in renamed_feedback.columns if column != "model" and column in frame.columns]
    if duplicate_columns:
        frame = frame.drop(columns=duplicate_columns, errors="ignore")
    return frame.merge(renamed_feedback, on="model", how="left")


def merge_prediction_feedback_state_metrics(
    frame: pd.DataFrame,
    prediction_feedback_state_summary: pd.DataFrame | None,
    *,
    task_name: str,
    horizon: int | None = None,
    macro_context: dict[str, Any] | None = None,
) -> pd.DataFrame:
    if frame.empty or prediction_feedback_state_summary is None or prediction_feedback_state_summary.empty or "model" not in frame.columns:
        return frame
    risk_regime = str((macro_context or {}).get("risk_regime", "")).strip()
    if risk_regime == "":
        return frame
    feedback = prediction_feedback_state_summary.copy()
    required_columns = {"task", "model_name", "macro_risk_regime"}
    if not required_columns.issubset(set(feedback.columns)):
        return frame
    feedback = feedback.loc[
        (feedback["task"].astype(str) == str(task_name))
        & (feedback["macro_risk_regime"].astype(str) == risk_regime)
    ].copy()
    if horizon is not None and "horizon_steps" in feedback.columns:
        feedback = feedback.loc[
            pd.to_numeric(feedback["horizon_steps"], errors="coerce") == float(horizon)
        ].copy()
    if feedback.empty:
        return frame
    desired_columns = [
        "model_name",
        "state_feedback_count",
        "state_feedback_weight_sum",
        "state_feedback_last_target_timestamp",
        "state_feedback_weighted_mean_abs_error_return",
        "state_feedback_weighted_directional_accuracy",
        "state_feedback_weighted_mean_strategy_return",
        "state_feedback_weighted_mean_excess_return",
        "state_feedback_weighted_brier_score",
        "state_feedback_total_strategy_return",
        "state_feedback_total_benchmark_return",
        "state_feedback_total_excess_return",
    ]
    available_columns = [column for column in desired_columns if column in feedback.columns]
    if "model_name" not in available_columns:
        return frame
    renamed_feedback = feedback[available_columns].rename(columns={"model_name": "model"})
    duplicate_columns = [column for column in renamed_feedback.columns if column != "model" and column in frame.columns]
    if duplicate_columns:
        frame = frame.drop(columns=duplicate_columns, errors="ignore")
    return frame.merge(renamed_feedback, on="model", how="left")


def prediction_feedback_underperformance_mask(
    frame: pd.DataFrame,
    *,
    task_name: str,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    feedback_count = frame_numeric_series(frame, "feedback_count").fillna(0.0)
    feedback_excess = frame_numeric_series(frame, "feedback_weighted_mean_excess_return")
    feedback_direction = frame_numeric_series(frame, "feedback_weighted_directional_accuracy")
    feedback_abs_error = frame_numeric_series(frame, "feedback_weighted_mean_abs_error_return")
    feedback_brier = frame_numeric_series(frame, "feedback_weighted_brier_score")
    minimum_count = 4.0
    if task_name == "regression":
        mask = (
            (feedback_count >= minimum_count)
            & feedback_excess.notna()
            & feedback_direction.notna()
            & feedback_abs_error.notna()
            & (feedback_excess <= -0.015)
            & (feedback_direction < 0.46)
            & (feedback_abs_error >= 0.08)
        )
    elif task_name == "classification":
        mask = (
            (feedback_count >= minimum_count)
            & feedback_excess.notna()
            & feedback_direction.notna()
            & (
                ((feedback_excess <= -0.010) & (feedback_direction < 0.48))
                | (feedback_brier.notna() & (feedback_brier >= 0.30) & (feedback_excess <= -0.005))
            )
        )
    else:
        mask = pd.Series(False, index=frame.index, dtype=bool)
    return mask.fillna(False)


def add_relative_performance_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    enriched = frame.copy()
    for value_column, benchmark_column, output_column in (
        ("holdout_total_return", "holdout_benchmark_return", "holdout_excess_return"),
        ("backtest_total_return", "backtest_benchmark_return", "backtest_excess_return"),
        ("macro_bucket_total_return", "macro_bucket_benchmark_return", "macro_bucket_excess_return"),
    ):
        if value_column not in enriched.columns:
            continue
        value_series = pd.to_numeric(enriched[value_column], errors="coerce")
        if benchmark_column in enriched.columns:
            benchmark_series = pd.to_numeric(enriched[benchmark_column], errors="coerce")
        else:
            benchmark_series = pd.Series(np.nan, index=enriched.index, dtype=float)
        enriched[output_column] = value_series - benchmark_series
    return enriched


def benchmark_underperformance_mask(
    frame: pd.DataFrame,
    *,
    horizon: int | None,
    task_name: str,
    require_holdout: bool = False,
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=bool)
    short_horizon = horizon is not None and horizon <= 7
    if task_name == "regression":
        overall_floor = -0.01 if short_horizon else -0.03
        holdout_floor = -0.03 if short_horizon else -0.05
    else:
        overall_floor = -0.015 if short_horizon else -0.035
        holdout_floor = -0.04 if short_horizon else -0.06

    candidate_columns = [column for column in ("holdout_excess_return", "backtest_excess_return", "macro_bucket_excess_return") if column in frame.columns]
    if not candidate_columns:
        return pd.Series(False, index=frame.index, dtype=bool)

    excess_values = pd.concat(
        [pd.to_numeric(frame[column], errors="coerce").rename(column) for column in candidate_columns],
        axis=1,
    )
    available = excess_values.notna().any(axis=1)
    max_excess = excess_values.max(axis=1, skipna=True)
    underperforming = available & (max_excess <= overall_floor)

    if require_holdout and "holdout_excess_return" in frame.columns:
        holdout_excess = pd.to_numeric(frame["holdout_excess_return"], errors="coerce")
        if "backtest_excess_return" in frame.columns:
            backtest_excess = pd.to_numeric(frame["backtest_excess_return"], errors="coerce")
        else:
            backtest_excess = pd.Series(np.nan, index=frame.index, dtype=float)
        underperforming = underperforming | (
            holdout_excess.notna()
            & backtest_excess.notna()
            & (holdout_excess <= holdout_floor)
            & (backtest_excess <= overall_floor)
        )
    return underperforming.fillna(False)


def augment_backtest_with_macro_bucket(
    backtest_frame: pd.DataFrame,
    *,
    signal_builder: Any,
    macro_bucket_series: pd.Series,
    target_bucket_label: str,
    market_data: pd.DataFrame,
    horizon: int,
    interval: str,
) -> pd.DataFrame:
    if backtest_frame.empty or macro_bucket_series.empty or not str(target_bucket_label).strip():
        return backtest_frame
    rows: list[dict[str, Any]] = []
    for _, row in backtest_frame.iterrows():
        model_name = str(row.get("model", "")).strip()
        if not model_name:
            continue
        signal = signal_builder(model_name, row)
        if signal is None:
            continue
        conditioned_signal = pd.Series(signal, copy=True)
        conditioned_signal = conditioned_signal.where(
            macro_bucket_series.reindex(conditioned_signal.index).astype("object") == str(target_bucket_label)
        )
        metrics, _ = run_signal_backtest(
            conditioned_signal,
            market_data=market_data,
            horizon=horizon,
            interval=interval,
        )
        rows.append(
            {
                "model": model_name,
                "macro_bucket_label": str(target_bucket_label),
                "macro_bucket_total_return": metrics.get("total_return"),
                "macro_bucket_benchmark_return": metrics.get("benchmark_return"),
                "macro_bucket_sharpe": metrics.get("sharpe"),
                "macro_bucket_max_drawdown": metrics.get("max_drawdown"),
                "macro_bucket_signal_count": metrics.get("signal_count"),
                "macro_bucket_hit_rate": metrics.get("hit_rate"),
            }
        )
    if not rows:
        return backtest_frame
    conditioned = pd.DataFrame(rows)
    output = backtest_frame.drop(
        columns=[
            "macro_bucket_label",
            "macro_bucket_total_return",
            "macro_bucket_benchmark_return",
            "macro_bucket_sharpe",
            "macro_bucket_max_drawdown",
            "macro_bucket_signal_count",
            "macro_bucket_hit_rate",
        ],
        errors="ignore",
    )
    return output.merge(conditioned, on="model", how="left")


def filter_persistent_losers(
    frame: pd.DataFrame,
    *,
    total_return_columns: list[str],
    sharpe_columns: list[str],
    extra_exclusion: pd.Series | None = None,
) -> tuple[pd.DataFrame, bool]:
    if frame.empty:
        return frame, False
    available_return_columns = [column for column in total_return_columns if column in frame.columns]
    available_sharpe_columns = [column for column in sharpe_columns if column in frame.columns]
    if not available_return_columns and not available_sharpe_columns and extra_exclusion is None:
        return frame, False

    returns_bad = pd.Series(False, index=frame.index, dtype=bool)
    sharpe_bad = pd.Series(False, index=frame.index, dtype=bool)
    if available_return_columns:
        negative_returns = pd.Series(0, index=frame.index, dtype=int)
        positive_returns = pd.Series(0, index=frame.index, dtype=int)
        for column in available_return_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            negative_returns += ((values <= 0.0) & values.notna()).astype(int)
            positive_returns += ((values > 0.0) & values.notna()).astype(int)
        required_negative_returns = min(2, len(available_return_columns))
        returns_bad = (negative_returns >= required_negative_returns) & (positive_returns == 0)
    if available_sharpe_columns:
        negative_sharpes = pd.Series(0, index=frame.index, dtype=int)
        positive_sharpes = pd.Series(0, index=frame.index, dtype=int)
        for column in available_sharpe_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            negative_sharpes += ((values <= 0.0) & values.notna()).astype(int)
            positive_sharpes += ((values > 0.0) & values.notna()).astype(int)
        required_negative_sharpes = min(2, len(available_sharpe_columns))
        sharpe_bad = (negative_sharpes >= required_negative_sharpes) & (positive_sharpes == 0)

    loser_mask = returns_bad & sharpe_bad if available_return_columns and available_sharpe_columns else returns_bad | sharpe_bad
    if extra_exclusion is not None:
        loser_mask = loser_mask | extra_exclusion.reindex(frame.index).fillna(False).astype(bool)
    filtered = frame.loc[~loser_mask].copy()
    if filtered.empty:
        return frame, False
    return filtered, bool(loser_mask.any())


def score_regression_selection_candidates(
    frame: pd.DataFrame,
    *,
    macro_context: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    scored = add_relative_performance_columns(frame)
    holdout_return_score = _normalize_selection_metric(scored, "holdout_total_return")
    holdout_excess_score = _normalize_selection_metric(scored, "holdout_excess_return")
    backtest_return_score = _normalize_selection_metric(scored, "backtest_total_return")
    backtest_excess_score = _normalize_selection_metric(scored, "backtest_excess_return")
    holdout_sharpe_score = _normalize_selection_metric(scored, "holdout_sharpe")
    backtest_sharpe_score = _normalize_selection_metric(scored, "backtest_sharpe")
    cv_rmse_score = _normalize_selection_metric(scored, "cv_price_rmse", higher_is_better=False)
    cv_direction_score = _normalize_selection_metric(scored, "cv_directional_accuracy")
    drawdown_score = _normalize_selection_metric(
        scored,
        "backtest_max_drawdown",
        higher_is_better=False,
        absolute=True,
    )
    macro_bucket_return_score = _normalize_selection_metric(scored, "macro_bucket_total_return")
    macro_bucket_excess_score = _normalize_selection_metric(scored, "macro_bucket_excess_return")
    macro_bucket_sharpe_score = _normalize_selection_metric(scored, "macro_bucket_sharpe")
    macro_bucket_drawdown_score = _normalize_selection_metric(
        scored,
        "macro_bucket_max_drawdown",
        higher_is_better=False,
        absolute=True,
    )
    macro_bucket_signal_count = (
        scored["macro_bucket_signal_count"]
        if "macro_bucket_signal_count" in scored.columns
        else pd.Series(np.nan, index=scored.index, dtype=float)
    )
    macro_bucket_support = pd.to_numeric(macro_bucket_signal_count, errors="coerce").fillna(0.0)
    macro_bucket_support = macro_bucket_support.clip(lower=0.0, upper=4.0) / 4.0
    feedback_support_raw = frame_numeric_series(scored, "feedback_count").fillna(0.0)
    feedback_support = feedback_support_raw.clip(lower=0.0, upper=8.0) / 8.0
    feedback_direction_score = _normalize_selection_metric(scored, "feedback_weighted_directional_accuracy")
    feedback_mae_return_score = _normalize_selection_metric(
        scored,
        "feedback_weighted_mean_abs_error_return",
        higher_is_better=False,
    )
    feedback_excess_score = _normalize_selection_metric(scored, "feedback_weighted_mean_excess_return")
    feedback_strategy_score = _normalize_selection_metric(scored, "feedback_weighted_mean_strategy_return")
    feedback_interval_score = _normalize_selection_metric(scored, "feedback_weighted_interval_coverage")
    state_feedback_support_raw = frame_numeric_series(scored, "state_feedback_count").fillna(0.0)
    state_feedback_support = state_feedback_support_raw.clip(lower=0.0, upper=6.0) / 6.0
    state_feedback_direction_score = _normalize_selection_metric(scored, "state_feedback_weighted_directional_accuracy")
    state_feedback_error_score = _normalize_selection_metric(
        scored,
        "state_feedback_weighted_mean_abs_error_return",
        higher_is_better=False,
    )
    state_feedback_excess_score = _normalize_selection_metric(scored, "state_feedback_weighted_mean_excess_return")
    state_feedback_strategy_score = _normalize_selection_metric(scored, "state_feedback_weighted_mean_strategy_return")
    risk_off_tilt = float(macro_context.get("risk_off_tilt", 0.0))
    risk_on_tilt = float(macro_context.get("risk_on_tilt", 0.0))
    volatility_stress = float(macro_context.get("volatility_stress", 0.0))
    trend_strength = min(abs(float(macro_context.get("trend_bias_score", 0.0))), 1.0)

    weights = {
        "holdout_return": 0.18 + 0.04 * risk_on_tilt - 0.03 * risk_off_tilt,
        "holdout_excess": 0.12 + 0.06 * risk_off_tilt,
        "backtest_return": 0.12 + 0.04 * risk_on_tilt - 0.02 * risk_off_tilt,
        "backtest_excess": 0.10 + 0.06 * risk_off_tilt,
        "holdout_sharpe": 0.18 + 0.06 * risk_off_tilt,
        "backtest_sharpe": 0.14 + 0.05 * risk_off_tilt,
        "cv_rmse": 0.14 + 0.04 * volatility_stress,
        "cv_direction": 0.08 + 0.04 * trend_strength,
        "drawdown": 0.08 + 0.10 * risk_off_tilt,
    }
    macro_bonus = (
        risk_on_tilt * (0.45 * holdout_return_score + 0.30 * holdout_excess_score + 0.25 * backtest_return_score)
        + risk_off_tilt * (0.35 * holdout_excess_score + 0.25 * backtest_excess_score + 0.20 * drawdown_score + 0.20 * backtest_sharpe_score)
    ) * 0.12
    macro_bucket_bonus = macro_bucket_support * (
        0.44 * macro_bucket_excess_score
        + 0.24 * macro_bucket_return_score
        + 0.20 * macro_bucket_sharpe_score
        + 0.12 * macro_bucket_drawdown_score
    ) * 0.24
    feedback_bonus = feedback_support * (
        0.34 * feedback_mae_return_score
        + 0.26 * feedback_direction_score
        + 0.22 * feedback_excess_score
        + 0.10 * feedback_strategy_score
        + 0.08 * feedback_interval_score
    ) * 0.28
    state_feedback_bonus = state_feedback_support * (
        0.34 * state_feedback_error_score
        + 0.28 * state_feedback_direction_score
        + 0.24 * state_feedback_excess_score
        + 0.14 * state_feedback_strategy_score
    ) * 0.32
    score = (
        weights["holdout_return"] * holdout_return_score
        + weights["holdout_excess"] * holdout_excess_score
        + weights["backtest_return"] * backtest_return_score
        + weights["backtest_excess"] * backtest_excess_score
        + weights["holdout_sharpe"] * holdout_sharpe_score
        + weights["backtest_sharpe"] * backtest_sharpe_score
        + weights["cv_rmse"] * cv_rmse_score
        + weights["cv_direction"] * cv_direction_score
        + weights["drawdown"] * drawdown_score
        + macro_bonus
        + macro_bucket_bonus
        + feedback_bonus
        + state_feedback_bonus
    )
    scored["_selection_score"] = score.divide(sum(weights.values()) + 0.96)
    return scored


def score_classification_selection_candidates(
    frame: pd.DataFrame,
    *,
    macro_context: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    scored = add_relative_performance_columns(frame)
    holdout_return_score = _normalize_selection_metric(scored, "holdout_total_return")
    holdout_excess_score = _normalize_selection_metric(scored, "holdout_excess_return")
    backtest_return_score = _normalize_selection_metric(scored, "backtest_total_return")
    backtest_excess_score = _normalize_selection_metric(scored, "backtest_excess_return")
    holdout_sharpe_score = _normalize_selection_metric(scored, "holdout_sharpe")
    backtest_sharpe_score = _normalize_selection_metric(scored, "backtest_sharpe")
    holdout_bal_score = _normalize_selection_metric(scored, "holdout_balanced_accuracy")
    cv_bal_score = _normalize_selection_metric(scored, "cv_balanced_accuracy")
    cv_roc_score = _normalize_selection_metric(scored, "cv_roc_auc")
    conservative_threshold_score = _normalize_selection_metric(scored, "backtest_signal_threshold")
    reactive_threshold_score = _normalize_selection_metric(scored, "backtest_signal_threshold", higher_is_better=False)
    macro_bucket_return_score = _normalize_selection_metric(scored, "macro_bucket_total_return")
    macro_bucket_excess_score = _normalize_selection_metric(scored, "macro_bucket_excess_return")
    macro_bucket_sharpe_score = _normalize_selection_metric(scored, "macro_bucket_sharpe")
    macro_bucket_signal_count = (
        scored["macro_bucket_signal_count"]
        if "macro_bucket_signal_count" in scored.columns
        else pd.Series(np.nan, index=scored.index, dtype=float)
    )
    macro_bucket_support = pd.to_numeric(macro_bucket_signal_count, errors="coerce").fillna(0.0)
    macro_bucket_support = macro_bucket_support.clip(lower=0.0, upper=4.0) / 4.0
    feedback_support_raw = frame_numeric_series(scored, "feedback_count").fillna(0.0)
    feedback_support = feedback_support_raw.clip(lower=0.0, upper=8.0) / 8.0
    feedback_direction_score = _normalize_selection_metric(scored, "feedback_weighted_directional_accuracy")
    feedback_excess_score = _normalize_selection_metric(scored, "feedback_weighted_mean_excess_return")
    feedback_strategy_score = _normalize_selection_metric(scored, "feedback_weighted_mean_strategy_return")
    feedback_brier_score = _normalize_selection_metric(
        scored,
        "feedback_weighted_brier_score",
        higher_is_better=False,
    )
    state_feedback_support_raw = frame_numeric_series(scored, "state_feedback_count").fillna(0.0)
    state_feedback_support = state_feedback_support_raw.clip(lower=0.0, upper=6.0) / 6.0
    state_feedback_direction_score = _normalize_selection_metric(scored, "state_feedback_weighted_directional_accuracy")
    state_feedback_excess_score = _normalize_selection_metric(scored, "state_feedback_weighted_mean_excess_return")
    state_feedback_strategy_score = _normalize_selection_metric(scored, "state_feedback_weighted_mean_strategy_return")
    state_feedback_brier_score = _normalize_selection_metric(
        scored,
        "state_feedback_weighted_brier_score",
        higher_is_better=False,
    )
    risk_off_tilt = float(macro_context.get("risk_off_tilt", 0.0))
    risk_on_tilt = float(macro_context.get("risk_on_tilt", 0.0))
    volatility_stress = float(macro_context.get("volatility_stress", 0.0))

    weights = {
        "holdout_return": 0.13 + 0.04 * risk_on_tilt - 0.03 * risk_off_tilt,
        "holdout_excess": 0.12 + 0.05 * risk_off_tilt,
        "backtest_return": 0.11 + 0.04 * risk_on_tilt - 0.02 * risk_off_tilt,
        "backtest_excess": 0.11 + 0.05 * risk_off_tilt,
        "holdout_sharpe": 0.17 + 0.05 * risk_off_tilt,
        "backtest_sharpe": 0.14 + 0.05 * risk_off_tilt,
        "holdout_bal": 0.14 + 0.06 * risk_off_tilt,
        "cv_bal": 0.10 + 0.04 * volatility_stress,
        "cv_roc": 0.13 + 0.06 * risk_off_tilt,
    }
    macro_bonus = (
        risk_on_tilt * (0.35 * holdout_return_score + 0.25 * holdout_excess_score + 0.20 * reactive_threshold_score + 0.20 * backtest_return_score)
        + risk_off_tilt * (0.28 * holdout_excess_score + 0.22 * backtest_excess_score + 0.20 * holdout_bal_score + 0.18 * cv_roc_score + 0.12 * conservative_threshold_score)
    ) * 0.12
    macro_bucket_bonus = macro_bucket_support * (
        0.42 * macro_bucket_excess_score
        + 0.22 * macro_bucket_return_score
        + 0.22 * macro_bucket_sharpe_score
        + 0.14 * holdout_bal_score
    ) * 0.26
    feedback_bonus = feedback_support * (
        0.32 * feedback_direction_score
        + 0.26 * feedback_excess_score
        + 0.22 * feedback_strategy_score
        + 0.20 * feedback_brier_score
    ) * 0.30
    state_feedback_bonus = state_feedback_support * (
        0.34 * state_feedback_direction_score
        + 0.28 * state_feedback_excess_score
        + 0.20 * state_feedback_strategy_score
        + 0.18 * state_feedback_brier_score
    ) * 0.34
    score = (
        weights["holdout_return"] * holdout_return_score
        + weights["holdout_excess"] * holdout_excess_score
        + weights["backtest_return"] * backtest_return_score
        + weights["backtest_excess"] * backtest_excess_score
        + weights["holdout_sharpe"] * holdout_sharpe_score
        + weights["backtest_sharpe"] * backtest_sharpe_score
        + weights["holdout_bal"] * holdout_bal_score
        + weights["cv_bal"] * cv_bal_score
        + weights["cv_roc"] * cv_roc_score
        + macro_bonus
        + macro_bucket_bonus
        + feedback_bonus
        + state_feedback_bonus
    )
    scored["_selection_score"] = score.divide(sum(weights.values()) + 1.02)
    return scored


def score_state_selection_candidates(
    frame: pd.DataFrame,
    *,
    macro_context: dict[str, Any],
) -> pd.DataFrame:
    if frame.empty:
        return frame
    scored = add_relative_performance_columns(frame)
    backtest_return_score = _normalize_selection_metric(scored, "backtest_total_return")
    backtest_excess_score = _normalize_selection_metric(scored, "backtest_excess_return")
    backtest_sharpe_score = _normalize_selection_metric(scored, "backtest_sharpe")
    balanced_accuracy_score = _normalize_selection_metric(scored, "balanced_accuracy")
    macro_f1_score = _normalize_selection_metric(scored, "macro_f1")
    roc_auc_score = _normalize_selection_metric(scored, "roc_auc")
    macro_bucket_return_score = _normalize_selection_metric(scored, "macro_bucket_total_return")
    macro_bucket_excess_score = _normalize_selection_metric(scored, "macro_bucket_excess_return")
    macro_bucket_sharpe_score = _normalize_selection_metric(scored, "macro_bucket_sharpe")
    macro_bucket_signal_count = (
        scored["macro_bucket_signal_count"]
        if "macro_bucket_signal_count" in scored.columns
        else pd.Series(np.nan, index=scored.index, dtype=float)
    )
    macro_bucket_support = pd.to_numeric(macro_bucket_signal_count, errors="coerce").fillna(0.0)
    macro_bucket_support = macro_bucket_support.clip(lower=0.0, upper=3.0) / 3.0
    risk_off_tilt = float(macro_context.get("risk_off_tilt", 0.0))
    risk_on_tilt = float(macro_context.get("risk_on_tilt", 0.0))
    weights = {
        "backtest_return": 0.14 + 0.05 * risk_on_tilt - 0.03 * risk_off_tilt,
        "backtest_excess": 0.10 + 0.06 * risk_off_tilt,
        "backtest_sharpe": 0.20 + 0.06 * risk_off_tilt,
        "balanced_accuracy": 0.22,
        "macro_f1": 0.20 + 0.05 * risk_off_tilt,
        "roc_auc": 0.18 + 0.05 * risk_off_tilt,
    }
    score = (
        weights["backtest_return"] * backtest_return_score
        + weights["backtest_excess"] * backtest_excess_score
        + weights["backtest_sharpe"] * backtest_sharpe_score
        + weights["balanced_accuracy"] * balanced_accuracy_score
        + weights["macro_f1"] * macro_f1_score
        + weights["roc_auc"] * roc_auc_score
        + macro_bucket_support * (0.46 * macro_bucket_excess_score + 0.30 * macro_bucket_return_score + 0.24 * macro_bucket_sharpe_score) * 0.22
    )
    scored["_selection_score"] = score.divide(sum(weights.values()) + 0.22)
    return scored


def select_regression_forecast_model(
    regression_leaderboard: pd.DataFrame,
    regression_backtest: pd.DataFrame | None = None,
    recent_holdout_report: pd.DataFrame | None = None,
    horizon: int | None = None,
    latest_features: pd.Series | None = None,
    prediction_feedback_summary: pd.DataFrame | None = None,
    prediction_feedback_state_summary: pd.DataFrame | None = None,
) -> tuple[str, str]:
    if regression_leaderboard.empty or "model" not in regression_leaderboard.columns:
        return "", "fallback_heuristic"
    best_cv_rmse = pd.to_numeric(regression_leaderboard["price_rmse"], errors="coerce").min()
    best_cv_directional = pd.to_numeric(regression_leaderboard["directional_accuracy"], errors="coerce").max()
    macro_context = derive_macro_selection_context(latest_features, horizon or 30)
    macro_suffix = f"+{macro_context['selection_tag']}" if macro_context else ""

    if recent_holdout_report is not None and not recent_holdout_report.empty:
        regression_holdout = recent_holdout_report.loc[recent_holdout_report["task"] == "regression"].copy()
        if not regression_holdout.empty:
            best_holdout_rmse = pd.to_numeric(regression_holdout["price_rmse"], errors="coerce").min()
            regression_holdout = regression_holdout.sort_values(
                ["sharpe", "total_return", "directional_accuracy", "price_rmse"],
                ascending=[False, False, False, True],
                na_position="last",
            ).reset_index(drop=True)
            guardrail_candidates: list[dict[str, Any]] = []
            for _, candidate in regression_holdout.iterrows():
                total_return = pd.to_numeric(candidate.get("total_return"), errors="coerce")
                sharpe = pd.to_numeric(candidate.get("sharpe"), errors="coerce")
                candidate_holdout_rmse = pd.to_numeric(candidate.get("price_rmse"), errors="coerce")
                candidate_directional = pd.to_numeric(candidate.get("directional_accuracy"), errors="coerce")
                if not (pd.notna(total_return) and pd.notna(sharpe) and total_return > 0.0 and sharpe > 0.0):
                    continue

                leaderboard_row = regression_leaderboard.loc[regression_leaderboard["model"] == candidate["model"]]
                if leaderboard_row.empty:
                    continue
                candidate_cv_rmse = pd.to_numeric(leaderboard_row.iloc[0].get("price_rmse"), errors="coerce")
                candidate_cv_directional = pd.to_numeric(
                    leaderboard_row.iloc[0].get("directional_accuracy"),
                    errors="coerce",
                )

                rmse_guardrail_ok = (
                    pd.notna(candidate_cv_rmse)
                    and pd.notna(best_cv_rmse)
                    and candidate_cv_rmse <= best_cv_rmse * 1.35
                )
                holdout_guardrail_ok = (
                    pd.notna(candidate_holdout_rmse)
                    and pd.notna(best_holdout_rmse)
                    and candidate_holdout_rmse <= best_holdout_rmse * 1.35
                )
                directional_guardrail_ok = (
                    pd.isna(candidate_cv_directional)
                    or pd.isna(best_cv_directional)
                    or candidate_cv_directional >= best_cv_directional - 0.08
                    or (
                        pd.notna(candidate_directional)
                        and candidate_directional >= 0.50
                    )
                )
                alternative_holdout_guardrail_ok = (
                    pd.notna(candidate_cv_rmse)
                    and pd.notna(best_cv_rmse)
                    and candidate_cv_rmse <= best_cv_rmse * 1.12
                    and (
                        pd.isna(candidate_holdout_rmse)
                        or pd.isna(best_holdout_rmse)
                        or candidate_holdout_rmse <= best_holdout_rmse * 1.60
                    )
                )
                if rmse_guardrail_ok and (holdout_guardrail_ok or alternative_holdout_guardrail_ok) and directional_guardrail_ok:
                    guardrail_candidates.append(
                        {
                            "model": str(candidate["model"]),
                            "holdout_total_return": total_return,
                            "holdout_benchmark_return": pd.to_numeric(candidate.get("benchmark_return"), errors="coerce"),
                            "holdout_sharpe": sharpe,
                            "holdout_price_rmse": candidate_holdout_rmse,
                            "holdout_directional_accuracy": candidate_directional,
                            "cv_price_rmse": candidate_cv_rmse,
                            "cv_directional_accuracy": candidate_cv_directional,
                        }
                    )
            if guardrail_candidates:
                guardrail_frame = pd.DataFrame(guardrail_candidates)
                guardrail_frame = merge_selection_backtest_metrics(guardrail_frame, regression_backtest)
                guardrail_frame = merge_prediction_feedback_metrics(
                    guardrail_frame,
                    prediction_feedback_summary,
                    task_name="regression",
                    horizon=horizon,
                )
                guardrail_frame = merge_prediction_feedback_state_metrics(
                    guardrail_frame,
                    prediction_feedback_state_summary,
                    task_name="regression",
                    horizon=horizon,
                    macro_context=macro_context,
                )
                guardrail_frame = add_relative_performance_columns(guardrail_frame)
                guardrail_frame = guardrail_frame.loc[
                    ~benchmark_underperformance_mask(
                        guardrail_frame,
                        horizon=horizon,
                        task_name="regression",
                        require_holdout=True,
                    )
                ].copy()
                guardrail_frame = guardrail_frame.loc[
                    ~prediction_feedback_underperformance_mask(
                        guardrail_frame,
                        task_name="regression",
                    )
                ].copy()
                if not guardrail_frame.empty:
                    guardrail_frame = score_regression_selection_candidates(
                        guardrail_frame,
                        macro_context=macro_context,
                    ).sort_values(
                        [
                            "_selection_score",
                            "holdout_total_return",
                            "holdout_sharpe",
                            "cv_directional_accuracy",
                            "cv_price_rmse",
                        ],
                        ascending=[False, False, False, False, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    return str(guardrail_frame.loc[0, "model"]), f"recent_holdout_sharpe_positive{macro_suffix}"

            if regression_backtest is not None and not regression_backtest.empty:
                holdout_merge = regression_holdout.rename(
                    columns={
                        "price_rmse": "holdout_price_rmse",
                        "price_mae": "holdout_price_mae",
                        "directional_accuracy": "holdout_directional_accuracy",
                        "total_return": "holdout_total_return",
                        "benchmark_return": "holdout_benchmark_return",
                        "sharpe": "holdout_sharpe",
                    }
                )
                leaderboard_merge = regression_leaderboard.rename(
                    columns={
                        "price_rmse": "cv_price_rmse",
                        "price_mae": "cv_price_mae",
                        "directional_accuracy": "cv_directional_accuracy",
                    }
                )
                merged = leaderboard_merge.merge(
                    holdout_merge[
                        [
                            "model",
                            "holdout_price_rmse",
                            "holdout_price_mae",
                            "holdout_directional_accuracy",
                            "holdout_total_return",
                            "holdout_benchmark_return",
                            "holdout_sharpe",
                        ]
                    ],
                    on="model",
                    how="left",
                )
                merged = merge_selection_backtest_metrics(merged, regression_backtest)
                merged = merge_prediction_feedback_metrics(
                    merged,
                    prediction_feedback_summary,
                    task_name="regression",
                    horizon=horizon,
                )
                merged = merge_prediction_feedback_state_metrics(
                    merged,
                    prediction_feedback_state_summary,
                    task_name="regression",
                    horizon=horizon,
                    macro_context=macro_context,
                )
                merged = add_relative_performance_columns(merged)
                regression_extra_exclusion = pd.Series(False, index=merged.index, dtype=bool)
                if pd.notna(best_cv_rmse):
                    regression_extra_exclusion = regression_extra_exclusion | (
                        pd.to_numeric(merged["cv_price_rmse"], errors="coerce").fillna(float("inf")) > best_cv_rmse * 1.45
                    )
                    if pd.notna(best_cv_directional):
                        regression_extra_exclusion = regression_extra_exclusion & (
                            pd.to_numeric(merged["cv_directional_accuracy"], errors="coerce").fillna(float("-inf"))
                            < max(float(best_cv_directional) - 0.12, 0.35)
                        )
                regression_extra_exclusion = regression_extra_exclusion | benchmark_underperformance_mask(
                    merged,
                    horizon=horizon,
                    task_name="regression",
                    require_holdout=True,
                )
                if horizon is not None and horizon <= 7:
                    holdout_total_return = pd.to_numeric(merged.get("holdout_total_return"), errors="coerce")
                    holdout_sharpe = pd.to_numeric(merged.get("holdout_sharpe"), errors="coerce")
                    holdout_excess = pd.to_numeric(merged.get("holdout_excess_return"), errors="coerce")
                    regression_extra_exclusion = regression_extra_exclusion | (
                        holdout_total_return.notna()
                        & holdout_sharpe.notna()
                        & holdout_excess.notna()
                        & (holdout_total_return <= 0.0)
                        & (holdout_sharpe <= 0.0)
                        & (holdout_excess <= -0.01)
                    )
                regression_extra_exclusion = regression_extra_exclusion | prediction_feedback_underperformance_mask(
                    merged,
                    task_name="regression",
                )
                merged, _ = filter_persistent_losers(
                    merged,
                    total_return_columns=["holdout_total_return", "backtest_total_return"],
                    sharpe_columns=["holdout_sharpe", "backtest_sharpe"],
                    extra_exclusion=regression_extra_exclusion,
                )
                positive_transition_candidates = merged.loc[
                    pd.to_numeric(merged["holdout_total_return"], errors="coerce").fillna(float("-inf")) > 0.0
                ].copy()
                if not positive_transition_candidates.empty:
                    positive_transition_candidates = positive_transition_candidates.loc[
                        pd.to_numeric(
                            positive_transition_candidates["holdout_sharpe"],
                            errors="coerce",
                        ).fillna(float("-inf")) > 0.0
                    ].copy()
                if not positive_transition_candidates.empty:
                    positive_transition_candidates = positive_transition_candidates.loc[
                        pd.to_numeric(
                            positive_transition_candidates["backtest_total_return"],
                            errors="coerce",
                        ).fillna(float("-inf")) > 0.0
                    ].copy()
                if not positive_transition_candidates.empty:
                    positive_transition_candidates = positive_transition_candidates.loc[
                        pd.to_numeric(
                            positive_transition_candidates["backtest_sharpe"],
                            errors="coerce",
                        ).fillna(float("-inf")) > 0.0
                    ].copy()
                if not positive_transition_candidates.empty:
                    positive_transition_candidates = positive_transition_candidates.loc[
                        pd.to_numeric(
                            positive_transition_candidates["cv_price_rmse"],
                            errors="coerce",
                        ).fillna(float("inf")) <= best_cv_rmse * 1.20
                    ].copy()
                if not positive_transition_candidates.empty:
                    direction_floor = (
                        best_cv_directional - 0.10
                        if pd.notna(best_cv_directional)
                        else 0.50
                    )
                    direction_floor = max(direction_floor, 0.50)
                    positive_transition_candidates = positive_transition_candidates.loc[
                        pd.to_numeric(
                            positive_transition_candidates["cv_directional_accuracy"],
                            errors="coerce",
                        ).fillna(float("-inf")) >= direction_floor
                    ].copy()
                if not positive_transition_candidates.empty:
                    positive_transition_candidates = score_regression_selection_candidates(
                        positive_transition_candidates,
                        macro_context=macro_context,
                    ).sort_values(
                        [
                            "_selection_score",
                            "holdout_total_return",
                            "holdout_sharpe",
                            "backtest_sharpe",
                            "cv_directional_accuracy",
                            "cv_price_rmse",
                        ],
                        ascending=[False, False, False, False, False, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    return str(positive_transition_candidates.loc[0, "model"]), f"recent_holdout_return_positive{macro_suffix}"

                robust_candidates = merged.copy()
                robust_candidates = robust_candidates.loc[
                    pd.to_numeric(robust_candidates["cv_price_rmse"], errors="coerce").fillna(float("inf")) <= best_cv_rmse * 1.30
                ].copy()
                if not robust_candidates.empty:
                    direction_floor = (
                        best_cv_directional - 0.15
                        if pd.notna(best_cv_directional)
                        else 0.35
                    )
                    direction_floor = max(direction_floor, 0.35)
                    robust_candidates = robust_candidates.loc[
                        pd.to_numeric(
                            robust_candidates["cv_directional_accuracy"],
                            errors="coerce",
                        ).fillna(float("-inf")) >= direction_floor
                    ].copy()
                if not robust_candidates.empty:
                    robust_candidates = score_regression_selection_candidates(
                        robust_candidates,
                        macro_context=macro_context,
                    ).sort_values(
                        [
                            "_selection_score",
                            "holdout_total_return",
                            "holdout_sharpe",
                            "backtest_total_return",
                            "backtest_sharpe",
                            "cv_price_rmse",
                        ],
                        ascending=[False, False, False, False, False, True],
                        na_position="last",
                    ).reset_index(drop=True)
                    return str(robust_candidates.loc[0, "model"]), f"robust_holdout_rank{macro_suffix}"
    fallback_regression = regression_leaderboard.rename(
        columns={
            "price_rmse": "cv_price_rmse",
            "directional_accuracy": "cv_directional_accuracy",
        }
    )
    fallback_regression = merge_selection_backtest_metrics(fallback_regression, regression_backtest)
    fallback_regression = merge_prediction_feedback_metrics(
        fallback_regression,
        prediction_feedback_summary,
        task_name="regression",
        horizon=horizon,
    )
    fallback_regression = merge_prediction_feedback_state_metrics(
        fallback_regression,
        prediction_feedback_state_summary,
        task_name="regression",
        horizon=horizon,
        macro_context=macro_context,
    )
    fallback_regression = add_relative_performance_columns(fallback_regression)
    fallback_regression, _ = filter_persistent_losers(
        fallback_regression,
        total_return_columns=["backtest_total_return"],
        sharpe_columns=["backtest_sharpe"],
        extra_exclusion=(
            benchmark_underperformance_mask(
                fallback_regression,
                horizon=horizon,
                task_name="regression",
                require_holdout=False,
            )
            | prediction_feedback_underperformance_mask(
                fallback_regression,
                task_name="regression",
            )
        ),
    )
    fallback_regression = score_regression_selection_candidates(
        fallback_regression,
        macro_context=macro_context,
    ).sort_values(
        ["_selection_score", "cv_price_rmse", "cv_directional_accuracy"],
        ascending=[False, True, False],
        na_position="last",
    ).reset_index(drop=True)
    return str(fallback_regression.loc[0, "model"]), f"leaderboard_price_rmse{macro_suffix}"


def select_classification_forecast_model(
    classification_leaderboard: pd.DataFrame,
    classification_backtest: pd.DataFrame,
    recent_holdout_report: pd.DataFrame | None = None,
    horizon: int | None = None,
    latest_features: pd.Series | None = None,
    prediction_feedback_summary: pd.DataFrame | None = None,
    prediction_feedback_state_summary: pd.DataFrame | None = None,
) -> tuple[str, str]:
    if classification_leaderboard.empty or "model" not in classification_leaderboard.columns:
        return "", "fallback_heuristic"
    minimum_cv_balanced_accuracy = 0.50 if horizon is not None and horizon <= 7 else 0.45
    minimum_cv_roc_auc = 0.45 if horizon is not None and horizon <= 7 else 0.35
    minimum_holdout_balanced_accuracy = 0.50 if horizon is not None and horizon <= 7 else 0.55
    macro_context = derive_macro_selection_context(latest_features, horizon or 30)
    macro_suffix = f"+{macro_context['selection_tag']}" if macro_context else ""

    def apply_quality_gate(frame: pd.DataFrame, *, require_holdout: bool) -> pd.DataFrame:
        gated = frame.copy()
        if "cv_balanced_accuracy" in gated.columns:
            gated = gated.loc[
                pd.to_numeric(gated["cv_balanced_accuracy"], errors="coerce").fillna(float("-inf"))
                >= minimum_cv_balanced_accuracy
            ].copy()
        if "cv_roc_auc" in gated.columns:
            gated = gated.loc[
                pd.to_numeric(gated["cv_roc_auc"], errors="coerce").fillna(float("-inf"))
                >= minimum_cv_roc_auc
            ].copy()
        if require_holdout and "holdout_balanced_accuracy" in gated.columns:
            gated = gated.loc[
                pd.to_numeric(gated["holdout_balanced_accuracy"], errors="coerce").fillna(float("-inf"))
                >= minimum_holdout_balanced_accuracy
            ].copy()
        return gated

    if recent_holdout_report is not None and not recent_holdout_report.empty:
        classification_holdout = recent_holdout_report.loc[
            recent_holdout_report["task"] == "classification"
        ].copy()
        if not classification_holdout.empty:
            holdout_merge = classification_holdout.rename(
                columns={
                    "total_return": "holdout_total_return",
                    "benchmark_return": "holdout_benchmark_return",
                    "sharpe": "holdout_sharpe",
                    "balanced_accuracy": "holdout_balanced_accuracy",
                    "roc_auc": "holdout_roc_auc",
                }
            )
            leaderboard_merge = classification_leaderboard.rename(
                columns={
                    "balanced_accuracy": "cv_balanced_accuracy",
                    "f1": "cv_f1",
                    "roc_auc": "cv_roc_auc",
                }
            )
            merged = leaderboard_merge.merge(
                holdout_merge[
                    [
                        "model",
                        "holdout_total_return",
                        "holdout_benchmark_return",
                        "holdout_sharpe",
                        "holdout_balanced_accuracy",
                        "holdout_roc_auc",
                    ]
                ],
                on="model",
                how="left",
            )
            merged = merge_selection_backtest_metrics(merged, classification_backtest)
            merged = merge_prediction_feedback_metrics(
                merged,
                prediction_feedback_summary,
                task_name="classification",
                horizon=horizon,
            )
            merged = merge_prediction_feedback_state_metrics(
                merged,
                prediction_feedback_state_summary,
                task_name="classification",
                horizon=horizon,
                macro_context=macro_context,
            )
            merged = add_relative_performance_columns(merged)
            classification_extra_exclusion = (
                pd.to_numeric(merged.get("cv_balanced_accuracy"), errors="coerce").fillna(float("-inf"))
                < minimum_cv_balanced_accuracy
            ) & (
                pd.to_numeric(merged.get("cv_roc_auc"), errors="coerce").fillna(float("-inf"))
                < minimum_cv_roc_auc
            ) & (
                pd.to_numeric(merged.get("holdout_total_return"), errors="coerce").fillna(float("-inf")) <= 0.0
            )
            classification_extra_exclusion = classification_extra_exclusion | benchmark_underperformance_mask(
                merged,
                horizon=horizon,
                task_name="classification",
                require_holdout=True,
            )
            holdout_excess_floor = -0.01 if horizon is not None and horizon <= 7 else -0.02
            classification_holdout_return = pd.to_numeric(merged.get("holdout_total_return"), errors="coerce")
            classification_holdout_sharpe = pd.to_numeric(merged.get("holdout_sharpe"), errors="coerce")
            classification_holdout_excess = pd.to_numeric(merged.get("holdout_excess_return"), errors="coerce")
            classification_extra_exclusion = classification_extra_exclusion | (
                classification_holdout_return.notna()
                & classification_holdout_sharpe.notna()
                & classification_holdout_excess.notna()
                & (classification_holdout_return <= 0.0)
                & (classification_holdout_sharpe <= 0.0)
                & (classification_holdout_excess <= holdout_excess_floor)
            )
            classification_extra_exclusion = classification_extra_exclusion | prediction_feedback_underperformance_mask(
                merged,
                task_name="classification",
            )
            merged, _ = filter_persistent_losers(
                merged,
                total_return_columns=["holdout_total_return", "backtest_total_return"],
                sharpe_columns=["holdout_sharpe", "backtest_sharpe"],
                extra_exclusion=classification_extra_exclusion,
            )
            holdout_positive_candidates = merged.loc[
                pd.to_numeric(merged["holdout_total_return"], errors="coerce").fillna(float("-inf")) > 0.0
            ].copy()
            if not holdout_positive_candidates.empty:
                holdout_positive_candidates = holdout_positive_candidates.loc[
                    pd.to_numeric(
                        holdout_positive_candidates["holdout_sharpe"],
                        errors="coerce",
                    ).fillna(float("-inf")) > 0.0
                ].copy()
            if not holdout_positive_candidates.empty:
                holdout_positive_candidates = holdout_positive_candidates.loc[
                    pd.to_numeric(
                        holdout_positive_candidates["backtest_total_return"],
                        errors="coerce",
                    ).fillna(float("-inf")) > 0.0
                ].copy()
            if not holdout_positive_candidates.empty:
                holdout_positive_candidates = holdout_positive_candidates.loc[
                    pd.to_numeric(
                        holdout_positive_candidates["backtest_sharpe"],
                        errors="coerce",
                    ).fillna(float("-inf")) > 0.0
                ].copy()
            if not holdout_positive_candidates.empty:
                holdout_positive_candidates = apply_quality_gate(
                    holdout_positive_candidates,
                    require_holdout=True,
                )
            if not holdout_positive_candidates.empty:
                holdout_positive_candidates = score_classification_selection_candidates(
                    holdout_positive_candidates,
                    macro_context=macro_context,
                ).sort_values(
                    [
                        "_selection_score",
                        "holdout_total_return",
                        "holdout_sharpe",
                        "backtest_sharpe",
                        "cv_roc_auc",
                        "cv_balanced_accuracy",
                    ],
                    ascending=[False, False, False, False, False, False],
                    na_position="last",
                ).reset_index(drop=True)
                return str(holdout_positive_candidates.loc[0, "model"]), f"recent_holdout_sharpe_positive{macro_suffix}"

            robust_candidates = apply_quality_gate(merged, require_holdout=True)
            if not robust_candidates.empty:
                robust_candidates = score_classification_selection_candidates(
                    robust_candidates,
                    macro_context=macro_context,
                ).sort_values(
                    [
                        "_selection_score",
                        "holdout_total_return",
                        "holdout_sharpe",
                        "backtest_total_return",
                        "backtest_sharpe",
                        "cv_roc_auc",
                        "cv_balanced_accuracy",
                    ],
                    ascending=[False, False, False, False, False, False, False],
                    na_position="last",
                ).reset_index(drop=True)
                return str(robust_candidates.loc[0, "model"]), f"robust_holdout_rank{macro_suffix}"

    if not classification_backtest.empty:
        fallback_candidates = classification_leaderboard.rename(
            columns={
                "balanced_accuracy": "cv_balanced_accuracy",
                "f1": "cv_f1",
                "roc_auc": "cv_roc_auc",
            }
        )
        fallback_candidates = merge_selection_backtest_metrics(fallback_candidates, classification_backtest)
        fallback_candidates = merge_prediction_feedback_metrics(
            fallback_candidates,
            prediction_feedback_summary,
            task_name="classification",
            horizon=horizon,
        )
        fallback_candidates = merge_prediction_feedback_state_metrics(
            fallback_candidates,
            prediction_feedback_state_summary,
            task_name="classification",
            horizon=horizon,
            macro_context=macro_context,
        )
        fallback_candidates = add_relative_performance_columns(fallback_candidates)
        fallback_extra_exclusion = (
            pd.to_numeric(fallback_candidates.get("cv_balanced_accuracy"), errors="coerce").fillna(float("-inf"))
            < minimum_cv_balanced_accuracy
        ) & (
            pd.to_numeric(fallback_candidates.get("cv_roc_auc"), errors="coerce").fillna(float("-inf"))
            < minimum_cv_roc_auc
        ) & (
            pd.to_numeric(fallback_candidates.get("backtest_total_return"), errors="coerce").fillna(float("-inf")) <= 0.0
        )
        fallback_extra_exclusion = fallback_extra_exclusion | benchmark_underperformance_mask(
            fallback_candidates,
            horizon=horizon,
            task_name="classification",
            require_holdout=False,
        )
        fallback_extra_exclusion = fallback_extra_exclusion | prediction_feedback_underperformance_mask(
            fallback_candidates,
            task_name="classification",
        )
        fallback_candidates, _ = filter_persistent_losers(
            fallback_candidates,
            total_return_columns=["backtest_total_return"],
            sharpe_columns=["backtest_sharpe"],
            extra_exclusion=fallback_extra_exclusion,
        )
        if not fallback_candidates.empty:
            positive_backtest_candidates = fallback_candidates.loc[
                pd.to_numeric(fallback_candidates["backtest_total_return"], errors="coerce").fillna(float("-inf")) > 0.0
            ].copy()
            positive_backtest_candidates = positive_backtest_candidates.loc[
                pd.to_numeric(positive_backtest_candidates["backtest_sharpe"], errors="coerce").fillna(float("-inf")) > 0.0
            ].copy()
            positive_backtest_candidates = apply_quality_gate(
                positive_backtest_candidates,
                require_holdout=False,
            )
            if not positive_backtest_candidates.empty:
                positive_backtest_candidates = score_classification_selection_candidates(
                    positive_backtest_candidates,
                    macro_context=macro_context,
                ).sort_values(
                    [
                        "_selection_score",
                        "backtest_total_return",
                        "backtest_sharpe",
                        "cv_roc_auc",
                        "cv_balanced_accuracy",
                    ],
                    ascending=[False, False, False, False, False],
                    na_position="last",
                ).reset_index(drop=True)
                top_backtest = positive_backtest_candidates.iloc[0]
                total_return = pd.to_numeric(top_backtest.get("backtest_total_return"), errors="coerce")
                sharpe = pd.to_numeric(top_backtest.get("backtest_sharpe"), errors="coerce")
                if pd.notna(total_return) and pd.notna(sharpe) and total_return > 0.0 and sharpe > 0.0:
                    backtest_model = str(top_backtest["model"])
                    leaderboard_row = classification_leaderboard.loc[classification_leaderboard["model"] == backtest_model]
                    if not leaderboard_row.empty:
                        cv_bal = pd.to_numeric(leaderboard_row.iloc[0].get("balanced_accuracy"), errors="coerce")
                        cv_roc = pd.to_numeric(leaderboard_row.iloc[0].get("roc_auc"), errors="coerce")
                        if (
                            pd.notna(cv_bal)
                            and pd.notna(cv_roc)
                            and float(cv_bal) >= minimum_cv_balanced_accuracy
                            and float(cv_roc) >= minimum_cv_roc_auc
                        ):
                            return backtest_model, f"backtest_sharpe_positive{macro_suffix}"
                    return backtest_model, f"backtest_sharpe_positive+cv_weak{macro_suffix}"
            fallback_candidates = score_classification_selection_candidates(
                fallback_candidates,
                macro_context=macro_context,
            ).sort_values(
                ["_selection_score", "cv_roc_auc", "cv_balanced_accuracy"],
                ascending=[False, False, False],
                na_position="last",
            ).reset_index(drop=True)
            top_model = str(fallback_candidates.loc[0, "model"])
            top_row = classification_leaderboard.loc[classification_leaderboard["model"] == top_model].iloc[0]
        else:
            top_model = str(classification_leaderboard.loc[0, "model"])
            top_row = classification_leaderboard.iloc[0]
    else:
        top_model = str(classification_leaderboard.loc[0, "model"])
        top_row = classification_leaderboard.iloc[0]
    top_cv_bal = pd.to_numeric(top_row.get("balanced_accuracy"), errors="coerce")
    top_cv_roc = pd.to_numeric(top_row.get("roc_auc"), errors="coerce")
    selection_basis = "leaderboard_balanced_accuracy"
    if (
        pd.isna(top_cv_bal)
        or pd.isna(top_cv_roc)
        or float(top_cv_bal) < minimum_cv_balanced_accuracy
        or float(top_cv_roc) < minimum_cv_roc_auc
    ):
        selection_basis = f"{selection_basis}+cv_weak"
    return top_model, f"{selection_basis}{macro_suffix}"


def select_state_forecast_model(
    leaderboard: pd.DataFrame,
    backtest: pd.DataFrame,
    task_key: str,
    minimum_balanced_accuracy: float = 0.38,
    latest_features: pd.Series | None = None,
) -> tuple[str, str]:
    if leaderboard.empty:
        return "", "fallback_heuristic"

    macro_context = derive_macro_selection_context(latest_features, 30)
    macro_suffix = f"+{macro_context['selection_tag']}" if macro_context else ""
    viable = leaderboard.copy()
    viable = viable.loc[pd.to_numeric(viable["balanced_accuracy"], errors="coerce").notna()].copy()
    viable = viable.loc[
        pd.to_numeric(viable["balanced_accuracy"], errors="coerce").fillna(float("-inf"))
        >= float(minimum_balanced_accuracy)
    ].copy()
    macro_floor = 0.28 if task_key == "regime" else 0.30
    viable = viable.loc[
        pd.to_numeric(viable["macro_f1"], errors="coerce").fillna(float("-inf")) >= macro_floor
    ].copy()
    if viable.empty:
        return "", "fallback_heuristic"

    if not backtest.empty:
        merged = merge_selection_backtest_metrics(leaderboard, backtest)
        merged = add_relative_performance_columns(merged)
        merged = merged.loc[merged["model"].isin(viable["model"])].copy()
        merged, _ = filter_persistent_losers(
            merged,
            total_return_columns=["backtest_total_return"],
            sharpe_columns=["backtest_sharpe"],
            extra_exclusion=benchmark_underperformance_mask(
                merged,
                horizon=30,
                task_name="classification",
                require_holdout=False,
            ),
        )
        positive_candidates = merged.loc[
            pd.to_numeric(merged["backtest_total_return"], errors="coerce").fillna(float("-inf")) > 0.0
        ].copy()
        if not positive_candidates.empty:
            positive_candidates = positive_candidates.loc[
                pd.to_numeric(positive_candidates["backtest_sharpe"], errors="coerce").fillna(float("-inf")) > 0.0
            ].copy()
        if not positive_candidates.empty:
            positive_candidates = positive_candidates.loc[
                pd.to_numeric(positive_candidates["balanced_accuracy"], errors="coerce").fillna(float("-inf"))
                >= float(minimum_balanced_accuracy)
            ].copy()
        if not positive_candidates.empty:
            positive_candidates = score_state_selection_candidates(
                positive_candidates,
                macro_context=macro_context,
            ).sort_values(
                ["_selection_score", "backtest_sharpe", "backtest_total_return", "macro_f1", "roc_auc", "balanced_accuracy"],
                ascending=[False, False, False, False, False, False],
                na_position="last",
            ).reset_index(drop=True)
            return str(positive_candidates.loc[0, "model"]), f"backtest_sharpe_positive{macro_suffix}"

    viable = score_state_selection_candidates(
        viable,
        macro_context=macro_context,
    ).sort_values(
        ["_selection_score", "balanced_accuracy", "macro_f1", "roc_auc", "accuracy"],
        ascending=[False, False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    return str(viable.loc[0, "model"]), f"leaderboard_balanced_accuracy{macro_suffix}"


def walk_forward_leaderboard(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    test_size: int,
    gap: int,
    sample_weight: pd.Series | None = None,
    verbose: bool = False,
    label: str = "",
    fold_feature_selection: bool = False,
    min_feature_coverage: float = DEFAULT_FEATURE_MIN_COVERAGE,
    embargo: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = dataset["target_return"]
    current_close = dataset["eth_close"]

    splitter = purged_time_series_split(
        n_samples=len(dataset),
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        embargo=embargo,
    )
    models = make_models(horizon=gap)
    aligned_sample_weight = align_sample_weight(sample_weight, dataset.index)
    oof = pd.DataFrame(index=dataset.index)
    rows: list[dict[str, float | str]] = []
    split_source = dataset[feature_columns] if not fold_feature_selection else dataset

    for model_name, template in models.items():
        log_progress(verbose, f"{label}[regression] fitting {model_name}")
        predictions = pd.Series(index=dataset.index, dtype=float)
        fold_metrics: list[dict[str, float]] = []

        for fold_number, (train_idx, test_idx) in enumerate(splitter.split(split_source), start=1):
            if fold_feature_selection:
                fold_features = select_fold_features(
                    dataset=dataset,
                    candidate_feature_columns=feature_columns,
                    train_positions=train_idx,
                    min_feature_coverage=min_feature_coverage,
                    horizon=gap,
                    target_column="target_return",
                )
                if not fold_features:
                    fold_features = feature_columns
            else:
                fold_features = feature_columns
            X_fold = dataset[fold_features]
            model = clone(template)
            X_train = X_fold.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_fold.iloc[test_idx]
            y_test = y.iloc[test_idx]
            current_close_test = current_close.iloc[test_idx]
            train_sample_weight = aligned_sample_weight.iloc[train_idx] if aligned_sample_weight is not None else None

            fit_model_with_optional_sample_weight(model, X_train, y_train, train_sample_weight)
            pred = pd.Series(model.predict(X_test), index=X_test.index, dtype=float)
            overlay_frame = dataset.iloc[test_idx] if fold_feature_selection else X_test
            if not overlay_disabled_for_context("walk_forward"):
                pred = apply_regime_response_overlay(pred, overlay_frame, horizon=gap)
            predictions.loc[X_test.index] = pred
            metrics = evaluate_predictions(
                current_close=current_close_test.to_numpy(),
                actual_return=y_test.to_numpy(),
                predicted_return=pred.to_numpy(),
            )
            metrics["fold"] = float(fold_number)
            fold_metrics.append(metrics)

        valid_predictions = predictions.dropna()
        if valid_predictions.empty:
            continue

        matched = dataset.loc[valid_predictions.index]
        summary = evaluate_predictions(
            current_close=matched["eth_close"].to_numpy(),
            actual_return=matched["target_return"].to_numpy(),
            predicted_return=valid_predictions.to_numpy(),
        )
        summary["model"] = model_name
        summary["folds"] = float(len(fold_metrics))
        rows.append(summary)

        oof[f"{model_name}_pred_return"] = predictions
        oof[f"{model_name}_pred_close"] = dataset["eth_close"] * (1.0 + predictions)

    if not rows:
        return pd.DataFrame(), oof

    leaderboard = pd.DataFrame(rows).sort_values(["price_rmse", "price_mae"], ascending=True).reset_index(drop=True)
    return leaderboard, oof


def walk_forward_classification(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    n_splits: int,
    test_size: int,
    gap: int,
    sample_weight: pd.Series | None = None,
    verbose: bool = False,
    label: str = "",
    fold_feature_selection: bool = False,
    min_feature_coverage: float = DEFAULT_FEATURE_MIN_COVERAGE,
    embargo: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = (dataset["target_return"] > 0).astype(int)
    splitter = purged_time_series_split(
        n_samples=len(dataset),
        n_splits=n_splits,
        test_size=test_size,
        gap=gap,
        embargo=embargo,
    )
    models = make_classification_models(horizon=gap)
    aligned_sample_weight = align_sample_weight(sample_weight, dataset.index)
    oof = pd.DataFrame(index=dataset.index)
    rows: list[dict[str, float | str]] = []
    split_source = dataset[feature_columns] if not fold_feature_selection else dataset

    for model_name, template in models.items():
        log_progress(verbose, f"{label}[classification] fitting {model_name}")
        probabilities = pd.Series(index=dataset.index, dtype=float)
        fold_count = 0

        for train_idx, test_idx in splitter.split(split_source):
            y_train = y.iloc[train_idx]
            if y_train.nunique() < 2:
                continue

            if fold_feature_selection:
                fold_features = select_fold_features(
                    dataset=dataset,
                    candidate_feature_columns=feature_columns,
                    train_positions=train_idx,
                    min_feature_coverage=min_feature_coverage,
                    horizon=gap,
                    target_column="target_return",
                )
                if not fold_features:
                    fold_features = feature_columns
            else:
                fold_features = feature_columns
            X_fold = dataset[fold_features]

            model = fit_calibrated_classifier(
                template,
                X_fold.iloc[train_idx],
                y_train,
                sample_weight=aligned_sample_weight.iloc[train_idx] if aligned_sample_weight is not None else None,
            )
            prob_up = pd.Series(
                model.predict_proba(X_fold.iloc[test_idx])[:, 1],
                index=X_fold.iloc[test_idx].index,
                dtype=float,
            )
            overlay_frame = dataset.iloc[test_idx] if fold_feature_selection else X_fold.iloc[test_idx]
            prob_up = apply_direction_regime_overlay(
                prob_up,
                overlay_frame,
                horizon=gap,
            )
            probabilities.iloc[test_idx] = prob_up.to_numpy()
            fold_count += 1

        valid_probabilities = probabilities.dropna()
        if valid_probabilities.empty or fold_count == 0:
            continue

        actual = y.loc[valid_probabilities.index].astype(int)
        selected_threshold, metrics = choose_classification_evaluation_threshold(
            actual_label=actual,
            probability_up=valid_probabilities,
            horizon=gap,
        )
        predicted_label = pd.Series((probabilities >= selected_threshold).astype(float), index=dataset.index, dtype=float)
        predicted_label[probabilities.isna()] = np.nan
        metrics["model"] = model_name
        metrics["folds"] = float(fold_count)
        metrics["signal_threshold"] = float(selected_threshold)
        rows.append(metrics)

        oof[f"{model_name}_prob_up"] = probabilities
        oof[f"{model_name}_pred_label"] = predicted_label

    if not rows:
        return pd.DataFrame(), oof

    leaderboard = pd.DataFrame(rows).sort_values(
        ["balanced_accuracy", "f1", "roc_auc", "signal_threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return leaderboard, oof


def choose_classification_evaluation_threshold(
    actual_label: pd.Series | np.ndarray,
    probability_up: pd.Series | np.ndarray,
    *,
    horizon: int | None = None,
) -> tuple[float, dict[str, Any]]:
    if isinstance(probability_up, pd.Series):
        probability_series = pd.to_numeric(probability_up, errors="coerce").astype(float)
    else:
        probability_series = pd.Series(np.asarray(probability_up, dtype=float), dtype=float)
    if isinstance(actual_label, pd.Series):
        actual_series = pd.to_numeric(actual_label, errors="coerce").reindex(probability_series.index)
    else:
        actual_series = pd.Series(np.asarray(actual_label, dtype=float), index=probability_series.index, dtype=float)

    valid_mask = probability_series.notna() & actual_series.notna()
    probability_series = probability_series.loc[valid_mask].clip(lower=1e-6, upper=1.0 - 1e-6)
    actual_series = actual_series.loc[valid_mask].astype(int)
    best_threshold = float(resolve_classification_signal_thresholds(DEFAULT_CLASSIFICATION_SIGNAL_THRESHOLD)[1])
    if probability_series.empty or actual_series.empty:
        return best_threshold, {
            "accuracy": float("nan"),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "brier_score": float("nan"),
            "roc_auc": float("nan"),
            "threshold_validation_scheme": "empty",
            "threshold_selection_rows": 0,
            "threshold_evaluation_rows": 0,
            "validation_embargo_rows": 0,
            "validation_leakage_safe": False,
        }

    selection_index, evaluation_index, embargo_rows, scheme = build_purged_threshold_validation_windows(
        probability_series.index,
        horizon or 1,
        min_train_rows=DEFAULT_THRESHOLD_VALIDATION_MIN_TRAIN_ROWS,
        min_eval_rows=max(DEFAULT_THRESHOLD_VALIDATION_MIN_EVAL_ROWS // 2, (horizon or 1) * 4),
    )
    if scheme == "full_sample":
        search_actual = actual_series
        search_probability = probability_series
        evaluation_actual = actual_series
        evaluation_probability = probability_series
    else:
        search_actual = actual_series.loc[selection_index]
        search_probability = probability_series.loc[selection_index]
        evaluation_actual = actual_series.loc[evaluation_index]
        evaluation_probability = probability_series.loc[evaluation_index]
        if search_actual.empty or evaluation_actual.empty:
            scheme = "full_sample"
            embargo_rows = 0
            search_actual = actual_series
            search_probability = probability_series
            evaluation_actual = actual_series
            evaluation_probability = probability_series

    best_metrics = evaluate_direction_classification(
        actual_label=search_actual.to_numpy(dtype=int),
        predicted_label=(search_probability.to_numpy(dtype=float) >= best_threshold).astype(int),
        probability_up=search_probability.to_numpy(dtype=float),
    )
    best_score = (
        float(best_metrics["balanced_accuracy"]),
        float(best_metrics["f1"]),
        float(best_metrics["roc_auc"]) if pd.notna(best_metrics["roc_auc"]) else float("-inf"),
    )

    for requested_threshold in DEFAULT_CLASSIFICATION_THRESHOLD_GRID:
        _, effective_threshold = resolve_classification_signal_thresholds(requested_threshold)
        metrics = evaluate_direction_classification(
            actual_label=search_actual.to_numpy(dtype=int),
            predicted_label=(search_probability.to_numpy(dtype=float) >= effective_threshold).astype(int),
            probability_up=search_probability.to_numpy(dtype=float),
        )
        score = (
            float(metrics["balanced_accuracy"]),
            float(metrics["f1"]),
            float(metrics["roc_auc"]) if pd.notna(metrics["roc_auc"]) else float("-inf"),
        )
        if score > best_score:
            best_threshold = float(effective_threshold)
            best_metrics = metrics
            best_score = score

    best_metrics = evaluate_direction_classification(
        actual_label=evaluation_actual.to_numpy(dtype=int),
        predicted_label=(evaluation_probability.to_numpy(dtype=float) >= best_threshold).astype(int),
        probability_up=evaluation_probability.to_numpy(dtype=float),
    )
    best_metrics = annotate_threshold_validation_metrics(
        best_metrics,
        scheme=scheme,
        selection_rows=len(search_actual),
        evaluation_rows=len(evaluation_actual),
        embargo_rows=embargo_rows,
    )
    return best_threshold, best_metrics


def fit_best_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    horizon: int,
    sample_weight: pd.Series | None = None,
) -> Any:
    model = clone(make_models(horizon=horizon)[model_name])
    fit_model_with_optional_sample_weight(
        model,
        dataset[feature_columns],
        dataset["target_return"],
        sample_weight,
    )
    return model


def fit_best_classifier(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    horizon: int,
    sample_weight: pd.Series | None = None,
) -> Any:
    target = (dataset["target_return"] > 0).astype(int)
    return fit_calibrated_classifier(
        make_classification_models(horizon=horizon)[model_name],
        dataset[feature_columns],
        target,
        sample_weight=sample_weight,
    )


def fit_quantile_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    quantile: float,
    sample_weight: pd.Series | None = None,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("imputer", _make_median_imputer()),
            (
                "model",
                HistGradientBoostingRegressor(
                    loss="quantile",
                    quantile=quantile,
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=200,
                    min_samples_leaf=12,
                    random_state=42,
                ),
            ),
        ]
    )
    fit_model_with_optional_sample_weight(
        model,
        dataset[feature_columns],
        dataset["target_return"],
        sample_weight,
    )
    return model


def estimate_prediction_interval_from_residuals(
    training_dataset: pd.DataFrame,
    feature_columns: list[str],
    point_model: Any,
    model_name: str,
    oof_predictions: pd.DataFrame | None = None,
    residual_oof_prediction: pd.Series | None = None,
    lower_quantile: float = 0.10,
    upper_quantile: float = 0.90,
) -> tuple[float, float]:
    residuals = pd.Series(dtype=float)
    oof_column = f"{model_name}_pred_return"

    if residual_oof_prediction is not None:
        aligned = training_dataset[["target_return"]].join(
            pd.DataFrame({"residual_reference_pred": residual_oof_prediction}),
            how="left",
        )
        residuals = (
            pd.to_numeric(aligned["target_return"], errors="coerce")
            - pd.to_numeric(aligned["residual_reference_pred"], errors="coerce")
        ).dropna()
    elif oof_predictions is not None and oof_column in oof_predictions.columns:
        aligned = training_dataset[["target_return"]].join(oof_predictions[[oof_column]], how="left")
        residuals = (
            pd.to_numeric(aligned["target_return"], errors="coerce")
            - pd.to_numeric(aligned[oof_column], errors="coerce")
        ).dropna()

    if residuals.shape[0] < 80:
        fitted = pd.Series(
            point_model.predict(training_dataset[feature_columns]),
            index=training_dataset.index,
            dtype=float,
        )
        residuals = (
            pd.to_numeric(training_dataset["target_return"], errors="coerce")
            - pd.to_numeric(fitted, errors="coerce")
        ).dropna()

    if residuals.empty:
        return 0.0, 0.0

    calibration_window = min(len(residuals), max(DEFAULT_RECENT_HOLDOUT_ROWS * 2, 240))
    calibrated = residuals.tail(calibration_window).to_numpy(dtype=float)
    lower_offset = float(np.nanquantile(calibrated, lower_quantile))
    upper_offset = float(np.nanquantile(calibrated, upper_quantile))
    central_miscoverage = float(max(1.0 - (upper_quantile - lower_quantile), 0.02))
    conformal_alpha = float(np.clip(central_miscoverage, 0.02, 0.40))
    calibrated_abs = np.abs(calibrated[np.isfinite(calibrated)])
    if calibrated_abs.size > 0:
        quantile_level = min(np.ceil((calibrated_abs.size + 1) * (1.0 - conformal_alpha)) / calibrated_abs.size, 0.995)
        conformal_radius = float(np.nanquantile(calibrated_abs, quantile_level))
        residual_center = float(np.nanmedian(calibrated))
        lower_offset = min(lower_offset, residual_center - conformal_radius)
        upper_offset = max(upper_offset, residual_center + conformal_radius)
    return lower_offset, upper_offset


def append_trimmed_regression_ensemble_candidate(
    leaderboard: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    horizon: int,
    component_models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_models = [
        model_name
        for model_name in component_models
        if f"{model_name}_pred_return" in oof_predictions.columns
    ]
    if len(available_models) < 2:
        return leaderboard, oof_predictions
    weights_by_model = derive_ensemble_weights_from_leaderboard(
        leaderboard, available_models, "price_rmse", higher_is_better=False
    )
    weights_by_column = {
        f"{model_name}_pred_return": weights_by_model.get(model_name, 0.0)
        for model_name in available_models
    }
    ensemble_predictions = skill_weighted_trimmed_mean(
        oof_predictions[[f"{model_name}_pred_return" for model_name in available_models]],
        weights=weights_by_column,
    )
    valid_predictions = ensemble_predictions.dropna()
    if valid_predictions.empty:
        return leaderboard, oof_predictions

    updated_oof = oof_predictions.copy()
    updated_oof[f"{TRIMMED_REGRESSION_ENSEMBLE_MODEL}_pred_return"] = ensemble_predictions
    updated_oof[f"{TRIMMED_REGRESSION_ENSEMBLE_MODEL}_pred_close"] = dataset["eth_close"] * (1.0 + ensemble_predictions)

    matched = dataset.loc[valid_predictions.index]
    summary = evaluate_predictions(
        current_close=matched["eth_close"].to_numpy(),
        actual_return=matched["target_return"].to_numpy(),
        predicted_return=valid_predictions.to_numpy(),
    )
    summary["model"] = TRIMMED_REGRESSION_ENSEMBLE_MODEL
    summary["folds"] = float(min(len(available_models), 4))
    summary["component_models"] = "|".join(available_models)

    # Phase 2.5 best-single guard: if the blend's OOF RMSE is worse than the
    # best member's, the blend is pure noise dilution — surface the single
    # model instead of recording an inferior ensemble row.
    member_rmse = pd.to_numeric(
        leaderboard.set_index("model").reindex(available_models).get("price_rmse"),
        errors="coerce",
    )
    best_member_rmse = float(member_rmse.min()) if member_rmse.notna().any() else float("nan")
    ensemble_rmse = float(summary.get("price_rmse", float("nan")))
    if (
        np.isfinite(best_member_rmse)
        and np.isfinite(ensemble_rmse)
        and ensemble_rmse > best_member_rmse
    ):
        return leaderboard, oof_predictions

    updated_leaderboard = leaderboard.loc[
        leaderboard["model"].astype(str) != TRIMMED_REGRESSION_ENSEMBLE_MODEL
    ].copy()
    updated_leaderboard = pd.concat([updated_leaderboard, pd.DataFrame([summary])], ignore_index=True)
    updated_leaderboard = updated_leaderboard.sort_values(
        ["price_rmse", "price_mae"],
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)
    return updated_leaderboard, updated_oof


def append_trimmed_classification_ensemble_candidate(
    leaderboard: pd.DataFrame,
    oof_predictions: pd.DataFrame,
    dataset: pd.DataFrame,
    horizon: int,
    component_models: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    available_models = [
        model_name
        for model_name in component_models
        if f"{model_name}_prob_up" in oof_predictions.columns
    ]
    if len(available_models) < 2:
        return leaderboard, oof_predictions
    weights_by_model = derive_ensemble_weights_from_leaderboard(
        leaderboard, available_models, "brier_score", higher_is_better=False
    )
    weights_by_column = {
        f"{model_name}_prob_up": weights_by_model.get(model_name, 0.0)
        for model_name in available_models
    }
    ensemble_probability = skill_weighted_trimmed_mean(
        oof_predictions[[f"{model_name}_prob_up" for model_name in available_models]],
        weights=weights_by_column,
    ).clip(lower=0.0, upper=1.0)
    valid_probability = ensemble_probability.dropna()
    if valid_probability.empty:
        return leaderboard, oof_predictions

    updated_oof = oof_predictions.copy()
    actual = (dataset.loc[valid_probability.index, "target_return"] > 0).astype(int)
    selected_threshold, metrics = choose_classification_evaluation_threshold(
        actual_label=actual,
        probability_up=valid_probability,
        horizon=horizon,
    )
    predicted_label = pd.Series(
        (ensemble_probability >= selected_threshold).astype(float),
        index=dataset.index,
        dtype=float,
    )
    predicted_label[ensemble_probability.isna()] = np.nan
    updated_oof[f"{TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL}_prob_up"] = ensemble_probability
    updated_oof[f"{TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL}_pred_label"] = predicted_label

    metrics["model"] = TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL
    metrics["folds"] = float(min(len(available_models), 4))
    metrics["signal_threshold"] = float(selected_threshold)
    metrics["component_models"] = "|".join(available_models)

    # Phase 2.5 best-single guard: skip the blend when its Brier score is
    # worse than every member's. The selector can still pick the best single
    # model; shipping an inferior ensemble row only confuses ranking.
    member_brier = pd.to_numeric(
        leaderboard.set_index("model").reindex(available_models).get("brier_score"),
        errors="coerce",
    )
    best_member_brier = float(member_brier.min()) if member_brier.notna().any() else float("nan")
    ensemble_brier = float(metrics.get("brier_score", float("nan")))
    if (
        np.isfinite(best_member_brier)
        and np.isfinite(ensemble_brier)
        and ensemble_brier > best_member_brier
    ):
        return leaderboard, oof_predictions

    updated_leaderboard = leaderboard.loc[
        leaderboard["model"].astype(str) != TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL
    ].copy()
    updated_leaderboard = pd.concat([updated_leaderboard, pd.DataFrame([metrics])], ignore_index=True)
    updated_leaderboard = updated_leaderboard.sort_values(
        ["balanced_accuracy", "f1", "roc_auc", "signal_threshold"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    return updated_leaderboard, updated_oof


def build_regression_model_score_frame(
    regression_leaderboard: pd.DataFrame | None,
    regression_backtest: pd.DataFrame | None,
    recent_holdout_report: pd.DataFrame | None,
    prediction_feedback_summary: pd.DataFrame | None = None,
    horizon: int | None = None,
) -> pd.DataFrame:
    leaderboard = regression_leaderboard.copy() if regression_leaderboard is not None else pd.DataFrame()
    if leaderboard.empty or "model" not in leaderboard.columns:
        return pd.DataFrame()

    score_frame = leaderboard.rename(
        columns={
            "price_rmse": "cv_price_rmse",
            "price_mae": "cv_price_mae",
            "directional_accuracy": "cv_directional_accuracy",
        }
    )

    if recent_holdout_report is not None and not recent_holdout_report.empty:
        holdout = recent_holdout_report.loc[recent_holdout_report["task"] == "regression"].copy()
        if not holdout.empty:
            holdout = holdout.rename(
                columns={
                    "price_rmse": "holdout_price_rmse",
                    "price_mae": "holdout_price_mae",
                    "directional_accuracy": "holdout_directional_accuracy",
                    "total_return": "holdout_total_return",
                    "benchmark_return": "holdout_benchmark_return",
                    "sharpe": "holdout_sharpe",
                }
            )
            score_frame = score_frame.merge(
                holdout[
                    [
                        "model",
                        "holdout_price_rmse",
                        "holdout_price_mae",
                        "holdout_directional_accuracy",
                        "holdout_total_return",
                        "holdout_benchmark_return",
                        "holdout_sharpe",
                    ]
                ],
                on="model",
                how="left",
            )

    score_frame = merge_selection_backtest_metrics(score_frame, regression_backtest)
    score_frame = merge_prediction_feedback_metrics(
        score_frame,
        prediction_feedback_summary,
        task_name="regression",
        horizon=horizon,
    )
    score_frame = add_relative_performance_columns(score_frame)

    return score_frame


def choose_regression_blend_components(
    score_frame: pd.DataFrame,
    selected_model: str,
    horizon: int,
    max_models: int | None = None,
) -> list[str]:
    if score_frame.empty:
        return [selected_model]

    if max_models is None:
        max_models = 2 if horizon <= 7 else 3

    best_cv_rmse = pd.to_numeric(score_frame["cv_price_rmse"], errors="coerce").min()
    eligible = score_frame.copy()
    if selected_model != TRIMMED_REGRESSION_ENSEMBLE_MODEL:
        eligible = eligible.loc[
            eligible["model"].astype(str) != TRIMMED_REGRESSION_ENSEMBLE_MODEL
        ].copy()
    if pd.notna(best_cv_rmse):
        tolerance = 1.15 if horizon <= 7 else 1.30
        eligible = eligible.loc[
            pd.to_numeric(eligible["cv_price_rmse"], errors="coerce").fillna(float("inf")) <= best_cv_rmse * tolerance
        ].copy()

    eligible = eligible.sort_values(
        [
            "holdout_total_return",
            "holdout_sharpe",
            "backtest_total_return",
            "backtest_sharpe",
            "cv_price_rmse",
        ],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

    components: list[str] = []
    if selected_model:
        components.append(str(selected_model))
    for model_name in eligible["model"].tolist():
        if model_name not in components:
            components.append(str(model_name))
        if len(components) >= max_models:
            break
    return components[:max_models]


def compute_regression_component_weights(
    score_frame: pd.DataFrame,
    component_models: list[str],
) -> dict[str, float]:
    if not component_models:
        return {}

    weights: dict[str, float] = {}
    for model_name in component_models:
        row = score_frame.loc[score_frame["model"] == model_name]
        if row.empty:
            weights[model_name] = 1.0
            continue
        item = row.iloc[0]
        score = 0.0
        for metric_name, scale in (
            ("holdout_total_return", 1.5),
            ("holdout_sharpe", 0.8),
            ("backtest_total_return", 0.8),
            ("backtest_sharpe", 0.5),
            ("feedback_weighted_mean_excess_return", 1.0),
            ("feedback_weighted_mean_strategy_return", 0.7),
            ("state_feedback_weighted_mean_excess_return", 1.2),
            ("state_feedback_weighted_mean_strategy_return", 0.8),
        ):
            value = pd.to_numeric(item.get(metric_name), errors="coerce")
            if pd.notna(value):
                score += scale * max(float(value), 0.0)
        for metric_name, scale in (
            ("feedback_weighted_mean_abs_error_return", 0.18),
            ("state_feedback_weighted_mean_abs_error_return", 0.22),
        ):
            value = pd.to_numeric(item.get(metric_name), errors="coerce")
            if pd.notna(value):
                score += scale * (1.0 / max(float(value), 1e-4))
        cv_rmse = pd.to_numeric(item.get("cv_price_rmse"), errors="coerce")
        if score <= 0.0:
            score = 1.0 / max(float(cv_rmse), 1.0) if pd.notna(cv_rmse) else 1.0
        weights[model_name] = float(score)

    total_weight = sum(max(weight, 0.0) for weight in weights.values())
    if total_weight <= 0.0:
        uniform = 1.0 / len(component_models)
        return {model_name: uniform for model_name in component_models}
    return {model_name: max(weight, 0.0) / total_weight for model_name, weight in weights.items()}


def apply_regression_peer_guardrail(
    base_prediction: float,
    component_predictions: dict[str, float],
    component_weights: dict[str, float],
    training_dataset: pd.DataFrame,
    horizon: int,
) -> tuple[float, bool, bool]:
    values = np.asarray(list(component_predictions.values()), dtype=float)
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return base_prediction, False, False

    weighted_blend = 0.0
    for model_name, prediction in component_predictions.items():
        weighted_blend += float(component_weights.get(model_name, 0.0)) * float(prediction)
    peer_median = float(np.nanmedian(finite_values))
    peer_std = float(np.nanstd(finite_values))

    recent_target = pd.to_numeric(training_dataset["target_return"], errors="coerce").dropna()
    recent_target = recent_target.tail(min(len(recent_target), max(DEFAULT_RECENT_HOLDOUT_ROWS * 2, 240)))
    if recent_target.empty:
        target_low, target_high = -1.0, 1.0
    else:
        target_low = float(np.nanquantile(recent_target.to_numpy(dtype=float), 0.02))
        target_high = float(np.nanquantile(recent_target.to_numpy(dtype=float), 0.98))

    peer_pad = max(0.03 if horizon <= 7 else 0.08, peer_std * 0.75)
    peer_low = float(np.nanmin(finite_values) - peer_pad)
    peer_high = float(np.nanmax(finite_values) + peer_pad)
    clip_low = max(target_low * (1.15 if horizon <= 7 else 1.35), peer_low)
    clip_high = min(target_high * (1.15 if horizon <= 7 else 1.35), peer_high)
    if clip_low >= clip_high:
        clip_low, clip_high = sorted((peer_low, peer_high))

    outlier_threshold = max(0.08 if horizon <= 7 else 0.15, peer_std * 2.25)
    blend_applied = bool(len(component_predictions) >= 2 and abs(base_prediction - peer_median) > outlier_threshold)
    adjusted_prediction = float(base_prediction)
    if blend_applied:
        base_weight = 0.50 if horizon <= 7 else 0.35
        adjusted_prediction = base_weight * base_prediction + (1.0 - base_weight) * weighted_blend

    clipped_prediction = float(np.clip(adjusted_prediction, clip_low, clip_high))
    clip_applied = not np.isclose(clipped_prediction, adjusted_prediction, atol=1e-9)
    return clipped_prediction, blend_applied, clip_applied


def build_blended_regression_oof(
    oof_predictions: pd.DataFrame | None,
    component_weights: dict[str, float],
) -> pd.Series | None:
    if oof_predictions is None or oof_predictions.empty or not component_weights:
        return None

    blended = pd.Series(0.0, index=oof_predictions.index, dtype=float)
    total_weight = 0.0
    for model_name, weight in component_weights.items():
        column = f"{model_name}_pred_return"
        if column not in oof_predictions.columns:
            continue
        series = pd.to_numeric(oof_predictions[column], errors="coerce")
        blended = blended.add(series * float(weight), fill_value=0.0)
        total_weight += float(weight)
    if total_weight <= 0.0:
        return None
    return blended / total_weight


def fit_oof_return_calibration(
    oof_prediction: pd.Series | None,
    actual_return: pd.Series,
    horizon: int,
) -> dict[str, Any]:
    if oof_prediction is None:
        return {"applied": False, "reason": "missing_oof_prediction"}

    aligned = pd.concat(
        [
            pd.to_numeric(oof_prediction, errors="coerce").rename("predicted"),
            pd.to_numeric(actual_return, errors="coerce").rename("actual"),
        ],
        axis=1,
    ).replace([np.inf, -np.inf], np.nan).dropna()

    minimum_rows = max(80, int(horizon) * 4 if horizon else 80)
    if aligned.shape[0] < minimum_rows:
        return {
            "applied": False,
            "reason": "insufficient_oof_rows",
            "rows": int(aligned.shape[0]),
        }

    window_rows = min(aligned.shape[0], max(DEFAULT_RECENT_HOLDOUT_ROWS * 2, 240))
    calibration_window = aligned.tail(window_rows).copy()

    eval_min_rows = max(30, int(horizon) if horizon else 30)
    split_at = int(calibration_window.shape[0] * 0.70)
    if calibration_window.shape[0] - split_at >= eval_min_rows and split_at >= eval_min_rows:
        fit_frame = calibration_window.iloc[:split_at]
        eval_frame = calibration_window.iloc[split_at:]
    else:
        fit_frame = calibration_window
        eval_frame = calibration_window

    def _weighted_linear_params(frame: pd.DataFrame) -> tuple[float, float, float, float]:
        x = frame["predicted"].to_numpy(dtype=float)
        y = frame["actual"].to_numpy(dtype=float)
        if x.size < 2 or np.nanstd(x) <= 1e-10:
            raise ValueError("oof calibration prediction variance is too small")

        if x.size >= 40:
            x_low, x_high = np.nanquantile(x, [0.02, 0.98])
            y_low, y_high = np.nanquantile(y, [0.02, 0.98])
            x = np.clip(x, x_low, x_high)
            y = np.clip(y, y_low, y_high)

        ages = np.arange(x.size - 1, -1, -1, dtype=float)
        half_life = max(x.size / 2.0, 60.0)
        weights = np.power(0.5, ages / half_life)
        weights = weights / max(float(weights.sum()), 1e-12)

        x_mean = float(np.average(x, weights=weights))
        y_mean = float(np.average(y, weights=weights))
        x_centered = x - x_mean
        y_centered = y - y_mean
        variance = float(np.average(np.square(x_centered), weights=weights))
        if variance <= 1e-10:
            raise ValueError("oof calibration prediction variance is too small")

        covariance = float(np.average(x_centered * y_centered, weights=weights))
        raw_slope = covariance / variance
        correlation = (
            float(np.corrcoef(x, y)[0, 1])
            if np.nanstd(x) > 1e-10 and np.nanstd(y) > 1e-10
            else 0.0
        )
        directional_accuracy = float(np.mean(np.sign(x) == np.sign(y)))

        slope_low = 0.12 if horizon <= 7 else 0.08
        slope_high = 1.10 if horizon <= 7 else 0.90
        slope = float(np.clip(raw_slope, slope_low, slope_high))
        if not np.isfinite(correlation) or correlation < 0.05:
            slope *= 0.50
        elif correlation < 0.15:
            slope *= 0.75
        if directional_accuracy < 0.48:
            slope *= 0.70
        elif horizon > 7 and directional_accuracy < 0.52:
            slope *= 0.85

        intercept_cap = 0.018 if horizon <= 7 else 0.045
        intercept = float(np.clip(y_mean - slope * x_mean, -intercept_cap, intercept_cap))
        return slope, intercept, correlation, directional_accuracy

    def _score(frame: pd.DataFrame, slope: float, intercept: float) -> dict[str, float]:
        predicted = frame["predicted"].to_numpy(dtype=float)
        actual = frame["actual"].to_numpy(dtype=float)
        calibrated = predicted * slope + intercept
        base_error = actual - predicted
        calibrated_error = actual - calibrated
        return {
            "base_mae": float(np.nanmean(np.abs(base_error))),
            "calibrated_mae": float(np.nanmean(np.abs(calibrated_error))),
            "base_rmse": float(np.sqrt(np.nanmean(np.square(base_error)))),
            "calibrated_rmse": float(np.sqrt(np.nanmean(np.square(calibrated_error)))),
            "base_directional_accuracy": float(np.mean(np.sign(predicted) == np.sign(actual))),
            "calibrated_directional_accuracy": float(np.mean(np.sign(calibrated) == np.sign(actual))),
        }

    try:
        slope, intercept, correlation, directional_accuracy = _weighted_linear_params(fit_frame)
    except ValueError as exc:
        return {
            "applied": False,
            "reason": str(exc),
            "rows": int(calibration_window.shape[0]),
        }

    validation_score = _score(eval_frame, slope, intercept)
    mae_ok = validation_score["calibrated_mae"] <= validation_score["base_mae"] * (1.015 if horizon <= 7 else 1.025)
    rmse_ok = validation_score["calibrated_rmse"] <= validation_score["base_rmse"] * (1.035 if horizon <= 7 else 1.055)
    direction_ok = (
        validation_score["calibrated_directional_accuracy"]
        >= validation_score["base_directional_accuracy"] - 0.035
    )
    if not (mae_ok and rmse_ok and direction_ok):
        return {
            "applied": False,
            "reason": "validation_gate_rejected",
            "rows": int(calibration_window.shape[0]),
            "slope": float(slope),
            "intercept": float(intercept),
            "correlation": float(correlation),
            "directional_accuracy": float(directional_accuracy),
            **validation_score,
        }

    # Refit on the full recent calibration window after the temporal validation
    # gate passes. This keeps the final transform recent while avoiding a blind
    # in-sample calibration tweak.
    try:
        slope, intercept, correlation, directional_accuracy = _weighted_linear_params(calibration_window)
    except ValueError:
        pass
    validation_score = _score(eval_frame, slope, intercept)
    return {
        "applied": True,
        "reason": "temporal_oof_validation_passed",
        "rows": int(calibration_window.shape[0]),
        "slope": float(slope),
        "intercept": float(intercept),
        "correlation": float(correlation),
        "directional_accuracy": float(directional_accuracy),
        **validation_score,
    }


def apply_oof_return_calibration(
    predicted_return: float,
    residual_oof_prediction: pd.Series | None,
    training_dataset: pd.DataFrame,
    horizon: int,
) -> tuple[float, pd.Series | None, dict[str, Any]]:
    if "target_return" not in training_dataset.columns:
        return predicted_return, residual_oof_prediction, {"applied": False, "reason": "missing_target_return"}

    calibration = fit_oof_return_calibration(
        residual_oof_prediction,
        pd.to_numeric(training_dataset["target_return"], errors="coerce"),
        horizon=horizon,
    )
    if not bool(calibration.get("applied")):
        return predicted_return, residual_oof_prediction, calibration

    slope = float(calibration.get("slope", 1.0))
    intercept = float(calibration.get("intercept", 0.0))
    calibrated_prediction = float(predicted_return * slope + intercept)

    recent_target = pd.to_numeric(training_dataset["target_return"], errors="coerce").dropna()
    recent_target = recent_target.tail(min(len(recent_target), max(DEFAULT_RECENT_HOLDOUT_ROWS * 2, 240)))
    if not recent_target.empty:
        target_low = float(np.nanquantile(recent_target.to_numpy(dtype=float), 0.01))
        target_high = float(np.nanquantile(recent_target.to_numpy(dtype=float), 0.99))
        fallback_cap = 0.22 if horizon <= 7 else 0.55
        fallback_floor = 0.05 if horizon <= 7 else 0.12
        clip_low = min(max(target_low * 1.25, -fallback_cap), -fallback_floor)
        clip_high = max(min(target_high * 1.25, fallback_cap), fallback_floor)
        calibrated_prediction = float(np.clip(calibrated_prediction, clip_low, clip_high))

    calibrated_oof = (
        pd.to_numeric(residual_oof_prediction, errors="coerce") * slope + intercept
        if residual_oof_prediction is not None
        else None
    )
    return calibrated_prediction, calibrated_oof, calibration


def strengthen_classification_threshold(
    base_threshold: float,
    classification_backtest: pd.DataFrame,
    recent_holdout_report: pd.DataFrame | None,
    model_name: str,
    horizon: int,
) -> tuple[float, bool]:
    threshold = float(base_threshold)
    if classification_backtest.empty or not model_name:
        return threshold, False

    backtest_row = classification_backtest.loc[classification_backtest["model"] == model_name]
    holdout_row = (
        recent_holdout_report.loc[
            (recent_holdout_report["task"] == "classification") & (recent_holdout_report["model"] == model_name)
        ]
        if recent_holdout_report is not None and not recent_holdout_report.empty
        else pd.DataFrame()
    )

    backtest_return = pd.to_numeric(backtest_row.iloc[0].get("total_return"), errors="coerce") if not backtest_row.empty else float("nan")
    holdout_return = pd.to_numeric(holdout_row.iloc[0].get("total_return"), errors="coerce") if not holdout_row.empty else float("nan")
    backtest_sharpe = pd.to_numeric(backtest_row.iloc[0].get("sharpe"), errors="coerce") if not backtest_row.empty else float("nan")
    holdout_sharpe = pd.to_numeric(holdout_row.iloc[0].get("sharpe"), errors="coerce") if not holdout_row.empty else float("nan")

    weaken_signal = (
        pd.notna(backtest_return)
        and pd.notna(holdout_return)
        and float(backtest_return) <= 0.0
        and float(holdout_return) <= 0.0
    )
    if pd.notna(backtest_sharpe) and pd.notna(holdout_sharpe):
        weaken_signal = weaken_signal or (float(backtest_sharpe) <= 0.0 and float(holdout_sharpe) <= 0.0)

    if weaken_signal:
        threshold = max(threshold, 0.65 if horizon <= 7 else 0.60)
    return threshold, weaken_signal


def selected_holdout_row(
    recent_holdout_report: pd.DataFrame | None,
    task_name: str,
    model_name: str,
) -> dict[str, Any]:
    if (
        recent_holdout_report is None
        or recent_holdout_report.empty
        or not task_name
        or not model_name
        or "task" not in recent_holdout_report.columns
        or "model" not in recent_holdout_report.columns
    ):
        return {}
    match = recent_holdout_report.loc[
        (recent_holdout_report["task"] == task_name) & (recent_holdout_report["model"] == model_name)
    ]
    return match.iloc[0].to_dict() if not match.empty else {}


def selected_backtest_row(
    backtest_frame: pd.DataFrame | None,
    model_name: str,
) -> dict[str, Any]:
    if backtest_frame is None or backtest_frame.empty or not model_name or "model" not in backtest_frame.columns:
        return {}
    match = backtest_frame.loc[backtest_frame["model"] == model_name]
    return match.iloc[0].to_dict() if not match.empty else {}


def positive_performance(row: dict[str, Any]) -> bool:
    if not row:
        return False
    total_return = safe_float(row.get("total_return"))
    sharpe = safe_float(row.get("sharpe"))
    if pd.notna(total_return) and total_return > 0.0 and pd.notna(sharpe) and sharpe > 0.0:
        return True
    if pd.notna(total_return) and total_return > 0.0 and pd.isna(sharpe):
        return True
    return False


def selected_leaderboard_row(
    leaderboard_frame: pd.DataFrame | None,
    model_name: str,
) -> dict[str, Any]:
    if leaderboard_frame is None or leaderboard_frame.empty or not model_name or "model" not in leaderboard_frame.columns:
        return {}
    match = leaderboard_frame.loc[leaderboard_frame["model"] == model_name]
    return match.iloc[0].to_dict() if not match.empty else {}


def direction_cv_is_weak(leaderboard_row: dict[str, Any]) -> bool:
    if not leaderboard_row:
        return False
    cv_bal = safe_float(leaderboard_row.get("balanced_accuracy"))
    cv_auc = safe_float(leaderboard_row.get("roc_auc"))
    cv_f1 = safe_float(leaderboard_row.get("f1"))
    return (
        pd.notna(cv_bal)
        and cv_bal <= 0.51
        and pd.notna(cv_auc)
        and cv_auc < 0.55
        and pd.notna(cv_f1)
        and cv_f1 <= 0.10
    )


def direction_performance_scale(
    leaderboard_row: dict[str, Any],
    backtest_row: dict[str, Any],
    holdout_row: dict[str, Any],
) -> float:
    def confirmation_scale(row: dict[str, Any]) -> float:
        if not row:
            return 0.55
        total_return = safe_float(row.get("total_return"))
        sharpe = safe_float(row.get("sharpe"))
        if pd.notna(total_return) and total_return > 0.0 and pd.notna(sharpe) and sharpe > 0.0:
            return 1.0
        if pd.notna(total_return) and total_return > 0.0:
            return 0.80
        if pd.notna(total_return) and total_return <= 0.0 and pd.notna(sharpe) and sharpe <= 0.0:
            return 0.30
        if (pd.notna(total_return) and total_return <= 0.0) or (pd.notna(sharpe) and sharpe <= 0.0):
            return 0.40
        return 0.55

    cv_bal = safe_float(leaderboard_row.get("balanced_accuracy"))
    cv_auc = safe_float(leaderboard_row.get("roc_auc"))
    cv_f1 = safe_float(leaderboard_row.get("f1"))
    if (
        (pd.notna(cv_bal) and cv_bal >= 0.53)
        or (pd.notna(cv_auc) and cv_auc >= 0.57)
        or (pd.notna(cv_f1) and cv_f1 >= 0.45)
    ):
        cv_scale = 1.0
    elif direction_cv_is_weak(leaderboard_row):
        cv_scale = 0.35
    elif pd.notna(cv_bal) or pd.notna(cv_auc) or pd.notna(cv_f1):
        cv_scale = 0.55
    else:
        cv_scale = 0.50

    components = [
        confirmation_scale(backtest_row),
        confirmation_scale(holdout_row),
        cv_scale,
    ]
    return float(np.clip(np.mean(components), 0.0, 1.0))


def calibrate_direction_confidence(
    *,
    model_name: str,
    selection_basis: str,
    predicted_direction: str,
    probability_up: float,
    lower_threshold: float,
    upper_threshold: float,
    classification_leaderboard: pd.DataFrame | None,
    classification_backtest: pd.DataFrame | None,
    recent_holdout_report: pd.DataFrame | None,
    horizon: int,
) -> float:
    raw_confidence = float(max(probability_up, 1.0 - probability_up))
    selection_basis_text = str(selection_basis)
    leaderboard_row = selected_leaderboard_row(classification_leaderboard, model_name)
    backtest_row = selected_backtest_row(classification_backtest, model_name)
    holdout_row = selected_holdout_row(recent_holdout_report, "classification", model_name)
    performance_scale = direction_performance_scale(
        leaderboard_row=leaderboard_row,
        backtest_row=backtest_row,
        holdout_row=holdout_row,
    )

    if predicted_direction == "UP":
        margin = np.clip(
            (probability_up - upper_threshold) / max(1.0 - upper_threshold, 1e-6),
            0.0,
            1.0,
        )
        decision_scale = float(0.45 + 0.55 * margin)
    elif predicted_direction == "DOWN":
        margin = np.clip(
            (lower_threshold - probability_up) / max(lower_threshold, 1e-6),
            0.0,
            1.0,
        )
        decision_scale = float(0.45 + 0.55 * margin)
    else:
        decision_scale = 0.25

    confidence = float(min(raw_confidence, performance_scale * (0.30 + 0.70 * decision_scale)))
    if "weak_signal_suppressed" in selection_basis_text:
        confidence = min(confidence, 0.28 if horizon <= 7 else 0.38)
    elif "cv_weak" in selection_basis_text:
        confidence = min(confidence, 0.40 if horizon <= 7 else 0.50)
    return float(np.clip(confidence, 0.0, 1.0))


def consensus_vote_summary(
    selected_model_name: str,
    backtest_frame: pd.DataFrame | None,
    recent_holdout_report: pd.DataFrame | None,
    task_name: str,
) -> dict[str, Any]:
    holdout_leader = ""
    if (
        recent_holdout_report is not None
        and not recent_holdout_report.empty
        and "task" in recent_holdout_report.columns
        and "model" in recent_holdout_report.columns
    ):
        holdout_frame = recent_holdout_report.loc[recent_holdout_report["task"] == task_name].copy()
        if not holdout_frame.empty:
            holdout_frame["total_return_num"] = pd.to_numeric(holdout_frame.get("total_return"), errors="coerce")
            holdout_frame["sharpe_num"] = pd.to_numeric(holdout_frame.get("sharpe"), errors="coerce")
            holdout_frame = holdout_frame.sort_values(
                ["total_return_num", "sharpe_num"],
                ascending=[False, False],
                na_position="last",
            ).reset_index(drop=True)
            holdout_leader = str(holdout_frame.iloc[0].get("model", "")).strip()
    votes = [
        ("selected", str(selected_model_name or "").strip()),
        (
            "backtest",
            str(backtest_frame.iloc[0].get("model", "")).strip()
            if backtest_frame is not None and not backtest_frame.empty and "model" in backtest_frame.columns
            else "",
        ),
        ("holdout", holdout_leader),
    ]
    normalized_votes = [(label, value) for label, value in votes if value]
    vote_counts: dict[str, int] = {}
    for _, vote in normalized_votes:
        vote_counts[vote] = vote_counts.get(vote, 0) + 1
    majority_model = ""
    majority_count = 0
    if vote_counts:
        majority_model = max(vote_counts.items(), key=lambda item: (item[1], item[0]))[0]
        majority_count = vote_counts[majority_model]
    return {
        "votes": normalized_votes,
        "majority_model": majority_model,
        "majority_count": majority_count,
        "backtest_leader": dict(normalized_votes).get("backtest", ""),
        "holdout_leader": dict(normalized_votes).get("holdout", ""),
    }


def determine_signal_tier(
    horizon: int,
    bias_direction: str,
    hybrid_score: float,
    classification_forecast: DirectionForecastResult,
    regime_forecast: RegimeForecastResult,
    reversal_forecast: ReversalForecastResult,
    classification_backtest: pd.DataFrame | None,
    recent_holdout_report: pd.DataFrame | None,
) -> tuple[str, bool, str, str]:
    consensus = consensus_vote_summary(
        selected_model_name=classification_forecast.model_name,
        backtest_frame=classification_backtest,
        recent_holdout_report=recent_holdout_report,
        task_name="classification",
    )
    backtest_row = selected_backtest_row(classification_backtest, classification_forecast.model_name)
    holdout_row = selected_holdout_row(recent_holdout_report, "classification", classification_forecast.model_name)
    backtest_ok = positive_performance(backtest_row)
    holdout_ok = positive_performance(holdout_row)
    classification_direction = str(classification_forecast.predicted_direction or "FLAT")
    probability_edge = float(abs(classification_forecast.probability_up - classification_forecast.probability_down))
    strong_probability = probability_edge >= (0.18 if horizon <= 7 else 0.45)
    strong_hybrid = abs(float(hybrid_score)) >= (0.18 if horizon <= 7 else 0.30)
    majority_count = int(consensus["majority_count"])

    candidate_direction = bias_direction
    direction_override = False
    if (
        candidate_direction == "FLAT"
        and classification_direction in {"UP", "DOWN"}
        and majority_count >= 2
        and backtest_ok
        and holdout_ok
        and strong_probability
    ):
        candidate_direction = classification_direction
        direction_override = True

    if candidate_direction == "FLAT":
        return "NO_SIGNAL", False, "bias_flat", "FLAT"

    regime_opposes = (
        (candidate_direction == "UP" and regime_forecast.predicted_regime == "DOWNTREND")
        or (candidate_direction == "DOWN" and regime_forecast.predicted_regime == "UPTREND")
    )
    if candidate_direction == "UP":
        regime_bias_ok = regime_forecast.probability_uptrend + 0.02 >= regime_forecast.probability_downtrend
        reversal_conflict = reversal_forecast.predicted_reversal == "TOP_REVERSAL"
    else:
        regime_bias_ok = regime_forecast.probability_downtrend + 0.02 >= regime_forecast.probability_uptrend
        reversal_conflict = reversal_forecast.predicted_reversal == "BOTTOM_REVERSAL"

    if (
        direction_override
        and majority_count >= 3
        and backtest_ok
        and holdout_ok
        and strong_probability
        and not regime_opposes
        and not reversal_conflict
    ):
        return "HIGH_CONFIDENCE", True, "direction_consensus_override", candidate_direction

    if (
        direction_override
        and majority_count >= 2
        and backtest_ok
        and holdout_ok
        and strong_probability
        and not regime_opposes
        and not reversal_conflict
    ):
        return "OBSERVE", False, "direction_consensus_override", candidate_direction

    if (
        majority_count >= 2
        and backtest_ok
        and holdout_ok
        and strong_probability
        and strong_hybrid
        and not regime_opposes
        and regime_bias_ok
        and not reversal_conflict
    ):
        return "HIGH_CONFIDENCE", True, "consensus+backtest+holdout+state_aligned", candidate_direction

    if (
        majority_count >= 2
        and backtest_ok
        and strong_probability
        and not regime_opposes
        and not reversal_conflict
    ):
        return "OBSERVE", False, "consensus+backtest_confirmed", "FLAT"

    if holdout_ok and strong_probability and strong_hybrid and not regime_opposes:
        return "OBSERVE", False, "holdout_confirmed_but_no_consensus", "FLAT"

    if backtest_ok and strong_probability and abs(float(hybrid_score)) >= 0.12:
        return "OBSERVE", False, "backtest_confirmed_but_no_consensus", "FLAT"

    return "NO_SIGNAL", False, "insufficient_confirmation", "FLAT"


def interval_to_offset(interval: str) -> pd.tseries.offsets.BaseOffset:
    mapping = {
        "1m": pd.offsets.Minute(1),
        "2m": pd.offsets.Minute(2),
        "5m": pd.offsets.Minute(5),
        "15m": pd.offsets.Minute(15),
        "30m": pd.offsets.Minute(30),
        "60m": pd.offsets.Hour(1),
        "90m": pd.offsets.Minute(90),
        "1h": pd.offsets.Hour(1),
        "1d": pd.offsets.Day(1),
        "5d": pd.offsets.Day(5),
        "1wk": pd.offsets.Week(1),
        "1mo": pd.offsets.MonthBegin(1),
        "3mo": pd.offsets.MonthBegin(3),
    }
    return mapping.get(interval, pd.offsets.Day(1))


def fit_best_state_model(
    dataset: pd.DataFrame,
    feature_columns: list[str],
    model_name: str,
    horizon: int,
    target_column: str,
    sample_weight: pd.Series | None = None,
) -> Any:
    state_dataset = dataset.loc[pd.to_numeric(dataset[target_column], errors="coerce").notna()].copy()
    model = clone(make_state_classification_models(horizon=horizon)[model_name])
    aligned_sample_weight = align_sample_weight(sample_weight, state_dataset.index)
    fit_model_with_optional_sample_weight(
        model,
        state_dataset[feature_columns],
        pd.to_numeric(state_dataset[target_column], errors="coerce").astype(int),
        aligned_sample_weight,
    )
    return model


def predict_state_probabilities(
    model: Any,
    latest_features: pd.DataFrame,
) -> pd.DataFrame:
    probability_matrix = np.asarray(model.predict_proba(latest_features), dtype=float)
    return align_state_probability_frame(
        probability_matrix=probability_matrix,
        classes=getattr(model, "classes_", []),
        index=latest_features.index,
    )


def forecast_regime_state(
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    interval: str,
    horizon: int,
    model_name: str,
    selection_basis: str,
    price_reference: PriceReference,
    sample_weight: pd.Series | None = None,
) -> RegimeForecastResult:
    model = fit_best_state_model(
        training_dataset,
        feature_columns,
        model_name,
        horizon=horizon,
        target_column="target_regime",
        sample_weight=sample_weight,
    )
    latest_features = prediction_frame.iloc[[-1]][feature_columns]
    probabilities = predict_state_probabilities(model, latest_features)
    model_input_close = float(prediction_frame["eth_close"].iloc[-1])
    last_close = float(price_reference.price)
    predicted_label = int(probabilities.iloc[0].idxmax())
    confidence = float(probabilities.iloc[0].max())
    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")
    return RegimeForecastResult(
        model_name=model_name,
        selection_basis=selection_basis,
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=model_input_close,
        last_close=last_close,
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_regime=REGIME_STATE_LABELS.get(predicted_label, "SIDEWAYS"),
        probability_downtrend=float(probabilities.iloc[0][0]),
        probability_sideways=float(probabilities.iloc[0][1]),
        probability_uptrend=float(probabilities.iloc[0][2]),
        confidence=confidence,
    )


def forecast_reversal_state(
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    interval: str,
    horizon: int,
    model_name: str,
    selection_basis: str,
    price_reference: PriceReference,
    sample_weight: pd.Series | None = None,
) -> ReversalForecastResult:
    model = fit_best_state_model(
        training_dataset,
        feature_columns,
        model_name,
        horizon=horizon,
        target_column="target_reversal_state",
        sample_weight=sample_weight,
    )
    latest_features = prediction_frame.iloc[[-1]][feature_columns]
    probabilities = predict_state_probabilities(model, latest_features)
    model_input_close = float(prediction_frame["eth_close"].iloc[-1])
    last_close = float(price_reference.price)
    predicted_label = int(probabilities.iloc[0].idxmax())
    confidence = float(probabilities.iloc[0].max())
    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")
    return ReversalForecastResult(
        model_name=model_name,
        selection_basis=selection_basis,
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=model_input_close,
        last_close=last_close,
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_reversal=REVERSAL_STATE_LABELS.get(predicted_label, "NONE"),
        probability_top_reversal=float(probabilities.iloc[0][0]),
        probability_no_reversal=float(probabilities.iloc[0][1]),
        probability_bottom_reversal=float(probabilities.iloc[0][2]),
        confidence=confidence,
    )


def heuristic_regime_forecast(
    prediction_frame: pd.DataFrame,
    interval: str,
    horizon: int,
    price_reference: PriceReference,
) -> RegimeForecastResult:
    latest = pd.to_numeric(prediction_frame.iloc[-1], errors="coerce")
    def latest_value(column: str) -> float:
        return float(np.nan_to_num(latest.get(column, np.nan), nan=0.0, posinf=0.0, neginf=0.0))
    trend_signal = (
        0.30 * np.clip(latest_value("eth_ma_7_ratio") / 0.15, -1.0, 1.0)
        + 0.25 * np.clip(latest_value("eth_ma_30_ratio") / 0.20, -1.0, 1.0)
        + 0.15 * np.clip(latest_value("eth_vwap_20_ratio") / 0.15, -1.0, 1.0)
        + 0.15 * np.clip(latest_value("eth_adx_trend_bias_14"), -1.0, 1.0)
        + 0.15 * np.clip(latest_value("eth_return_30") / 0.20, -1.0, 1.0)
    )
    trend_signal = float(np.clip(trend_signal, -1.0, 1.0))
    up_prob = float(np.clip(0.33 + 0.30 * max(trend_signal, 0.0), 0.05, 0.90))
    down_prob = float(np.clip(0.33 + 0.30 * max(-trend_signal, 0.0), 0.05, 0.90))
    sideways_prob = float(np.clip(1.0 - up_prob - down_prob, 0.05, 0.90))
    total = up_prob + down_prob + sideways_prob
    up_prob, down_prob, sideways_prob = up_prob / total, down_prob / total, sideways_prob / total
    predicted_label = int(np.argmax([down_prob, sideways_prob, up_prob]))
    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")
    return RegimeForecastResult(
        model_name="heuristic_regime",
        selection_basis="fallback_heuristic",
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=float(prediction_frame["eth_close"].iloc[-1]),
        last_close=float(price_reference.price),
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_regime=REGIME_STATE_LABELS[predicted_label],
        probability_downtrend=down_prob,
        probability_sideways=sideways_prob,
        probability_uptrend=up_prob,
        confidence=float(max(up_prob, down_prob, sideways_prob)),
    )


def heuristic_reversal_forecast(
    prediction_frame: pd.DataFrame,
    interval: str,
    horizon: int,
    price_reference: PriceReference,
) -> ReversalForecastResult:
    latest = pd.to_numeric(prediction_frame.iloc[-1], errors="coerce")
    def latest_value(column: str) -> float:
        return float(np.nan_to_num(latest.get(column, np.nan), nan=0.0, posinf=0.0, neginf=0.0))
    bottom_score = (
        0.28 * np.clip(-latest_value("eth_drawdown_30") / 0.30, 0.0, 1.0)
        + 0.20 * np.clip(latest_value("eth_rebound_from_20d_low") / 0.25, 0.0, 1.0)
        + 0.18 * np.clip(latest_value("eth_rsi_14_change_3") / 12.0, 0.0, 1.0)
        + 0.18 * np.clip(latest_value("eth_macd_hist_change_3") / 0.05, 0.0, 1.0)
        + 0.16 * np.clip(latest_value("eth_adx_trend_bias_14"), 0.0, 1.0)
    )
    top_score = (
        0.28 * np.clip(latest_value("eth_rebound_from_20d_low") / 0.30, 0.0, 1.0)
        + 0.20 * np.clip(-latest_value("eth_return_1") / 0.08, 0.0, 1.0)
        + 0.18 * np.clip(-latest_value("eth_rsi_14_change_3") / 12.0, 0.0, 1.0)
        + 0.18 * np.clip(-latest_value("eth_macd_hist_change_3") / 0.05, 0.0, 1.0)
        + 0.16 * np.clip(1.0 - abs(latest_value("eth_distance_to_20d_high")) / 0.15, 0.0, 1.0)
    )
    none_prob = float(np.clip(1.0 - max(bottom_score, top_score), 0.10, 0.80))
    top_prob = float(np.clip(top_score, 0.05, 0.85))
    bottom_prob = float(np.clip(bottom_score, 0.05, 0.85))
    total = top_prob + none_prob + bottom_prob
    top_prob, none_prob, bottom_prob = top_prob / total, none_prob / total, bottom_prob / total
    predicted_label = int(np.argmax([top_prob, none_prob, bottom_prob]))
    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")
    return ReversalForecastResult(
        model_name="heuristic_reversal",
        selection_basis="fallback_heuristic",
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=float(prediction_frame["eth_close"].iloc[-1]),
        last_close=float(price_reference.price),
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_reversal=REVERSAL_STATE_LABELS[predicted_label],
        probability_top_reversal=top_prob,
        probability_no_reversal=none_prob,
        probability_bottom_reversal=bottom_prob,
        confidence=float(max(top_prob, none_prob, bottom_prob)),
    )


def compute_forecast_score(
    predicted_return: float,
    training_target: pd.Series,
) -> float:
    target = pd.to_numeric(training_target, errors="coerce").dropna()
    if target.empty:
        scale = 0.10
    else:
        scale = float(np.nanquantile(np.abs(target.to_numpy(dtype=float)), 0.80))
        scale = max(scale, 0.04)
    return float(np.clip(predicted_return / scale, -1.0, 1.0))


def compute_slow_fundamental_score(latest_features: pd.Series) -> float:
    latest = pd.to_numeric(latest_features, errors="coerce")

    def scaled(column: str, scale: float = 3.0) -> float:
        value = latest.get(column, np.nan)
        if pd.isna(value):
            return float("nan")
        return float(np.clip(float(value) / scale, -1.0, 1.0))

    contributions: list[tuple[float, float]] = []
    positive_columns = [
        (0.12, "artemis_non_sybil_users_z_90"),
        (0.08, "artemis_returning_users_z_90"),
        (0.10, "artemis_chain_fees_z_30"),
        (0.10, "artemis_chain_tvl_z_30"),
        (0.10, "artemis_filtered_stablecoin_transactions_z_30"),
        (0.08, "artemis_stablecoin_total_supply_z_30"),
        (0.07, "artemis_total_staked__native_z_90"),
        (0.08, "artemis_burns_native_z_30"),
        (0.08, "artemis_net_etf_flow_z_30"),
        (0.05, "artemis_cumulative_etf_flow_z_90"),
    ]
    negative_columns = [
        (0.07, "artemis_gross_emissions__native_z_90"),
        (0.06, "fred_bbb_oas_z_90"),
        (0.05, "macro_tightening_20"),
        (0.05, "risk_aversion_spread_5"),
    ]

    for weight, column in positive_columns:
        value = scaled(column)
        if pd.notna(value):
            contributions.append((weight, value))
    for weight, column in negative_columns:
        value = scaled(column)
        if pd.notna(value):
            contributions.append((weight, -value))

    age_columns = [column for column in latest.index if str(column).startswith("artemis_") and str(column).endswith("__age_days")]
    if age_columns:
        age_values = pd.to_numeric(latest.loc[age_columns], errors="coerce").dropna()
        if not age_values.empty:
            median_age = float(age_values.median())
            age_penalty = float(np.clip(median_age / 45.0, 0.0, 1.0))
            contributions.append((0.08, -age_penalty))

    if not contributions:
        return 0.0

    total_weight = sum(weight for weight, _ in contributions)
    if total_weight <= 0.0:
        return 0.0
    weighted_score = sum(weight * value for weight, value in contributions) / total_weight
    return float(np.clip(weighted_score, -1.0, 1.0))


def resolve_hybrid_volatility_profile(
    training_target: pd.Series,
    latest_features: pd.Series,
    horizon: int,
) -> tuple[float, float, float, float]:
    recent_target = pd.to_numeric(training_target, errors="coerce").dropna()
    recent_target = recent_target.tail(min(len(recent_target), max(DEFAULT_RECENT_HOLDOUT_ROWS * 2, 240)))
    if recent_target.empty:
        target_scale = 0.10
        target_std = 0.06 if horizon <= 7 else 0.14
    else:
        target_values = recent_target.to_numpy(dtype=float)
        target_scale = max(float(np.nanquantile(np.abs(target_values), 0.80)), 0.04)
        target_std = max(float(np.nanstd(target_values)), 0.02 if horizon <= 7 else 0.05)

    latest = pd.to_numeric(pd.Series(latest_features), errors="coerce")
    realized_daily_vol = max(
        safe_float(latest.get("eth_vol_7")) or 0.0,
        safe_float(latest.get("eth_vol_14")) or 0.0,
        safe_float(latest.get("eth_vol_30")) or 0.0,
        0.80 * (safe_float(latest.get("eth_vol_90")) or 0.0),
        0.65 * (safe_float(latest.get("eth_vol_180")) or 0.0),
    )
    realized_move_scale = max(realized_daily_vol * float(np.sqrt(max(horizon, 1))), 0.0)
    tail_event_pressure = max(
        safe_float(latest.get("eth_tail_event_pressure")) or 0.0,
        safe_float(latest.get("eth_range_expansion_z_30")) or 0.0,
        safe_float(latest.get("eth_volume_impulse_3_30")) or 0.0,
        safe_float(latest.get("eth_vol_30_180_ratio")) or 0.0,
        max(-(safe_float(latest.get("eth_vol_squeeze_20_90")) or 0.0), 0.0),
    )
    tail_event_pressure = float(np.clip(tail_event_pressure, 0.0, 2.5))
    baseline_move = 0.08 if horizon <= 7 else 0.18
    volatility_scale = float(
        np.clip(
            max(
                target_scale / baseline_move,
                target_std / (0.06 if horizon <= 7 else 0.14),
                realized_move_scale / baseline_move if realized_move_scale > 0.0 else 0.0,
            ),
            0.85,
            1.75,
        )
    )
    volatility_scale = float(np.clip(volatility_scale * (1.0 + 0.10 * tail_event_pressure), 0.85, 1.95))
    scenario_spread = float(
        np.clip(
            max(
                target_scale * (0.95 if horizon <= 7 else 1.05),
                target_std * (1.35 if horizon <= 7 else 1.55),
                realized_move_scale * (1.15 if horizon <= 7 else 1.25),
                0.05 if horizon <= 7 else 0.12,
            ),
            0.04 if horizon <= 7 else 0.10,
            0.35 if horizon <= 7 else 0.75,
        )
    )
    scenario_spread = float(
        np.clip(
            scenario_spread * (1.0 + (0.14 if horizon <= 7 else 0.20) * tail_event_pressure),
            0.04 if horizon <= 7 else 0.10,
            0.40 if horizon <= 7 else 0.85,
        )
    )
    return target_scale, target_std, volatility_scale, scenario_spread


def build_active_forecast_track(
    *,
    horizon: int,
    price_reference: PriceReference,
    regression_forecast: ForecastResult,
    classification_forecast: DirectionForecastResult,
    regime_forecast: RegimeForecastResult,
    effective_reversal_forecast: ReversalForecastResult,
    hybrid_score: float,
    trend_score: float,
    direction_score: float,
    reversal_score: float,
    fundamental_score: float,
    target_scale: float,
    target_low: float,
    target_high: float,
    scenario_spread: float,
    volatility_scale: float,
    macro_context_summary: dict[str, Any],
    live_gap_penalty: float,
    signal_tier: str = "NO_SIGNAL",
) -> dict[str, float | str]:
    macro_spread = safe_float(macro_context_summary.get("macro_spread")) or 0.0
    trend_bias_score = safe_float(macro_context_summary.get("trend_bias_score")) or 0.0
    risk_on_tilt = safe_float(macro_context_summary.get("risk_on_tilt")) or 0.0
    risk_off_tilt = safe_float(macro_context_summary.get("risk_off_tilt")) or 0.0
    macro_directional_bias = str(macro_context_summary.get("directional_bias", "FLAT"))
    volatility_stress = safe_float(macro_context_summary.get("volatility_stress")) or 0.0

    macro_tilt_score = float(
        np.clip(
            0.45 * macro_spread
            + 0.30 * trend_bias_score
            + 0.15 * (risk_on_tilt - risk_off_tilt)
            + 0.10 * fundamental_score,
            -1.0,
            1.0,
        )
    )
    directional_tilt_score = float(
        np.clip(
            0.42 * direction_score
            + 0.30 * trend_score
            + 0.18 * reversal_score
            + 0.10 * hybrid_score,
            -1.0,
            1.0,
        )
    )
    active_bias = target_scale * (0.58 * macro_tilt_score + 0.42 * directional_tilt_score)
    active_return = float(regression_forecast.predicted_return + 0.55 * active_bias)

    if macro_directional_bias == "UP" and active_return < 0.0:
        active_return *= 0.55
    elif macro_directional_bias == "DOWN" and active_return > 0.0:
        active_return *= 0.55

    if live_gap_penalty > 0.0:
        active_return *= float(np.clip(1.0 - 0.12 * live_gap_penalty, 0.70, 1.0))

    tier = str(signal_tier or "NO_SIGNAL").upper()
    active_clip_multiplier = 1.20 if horizon <= 7 else 1.24
    if horizon <= 7:
        active_clip_cap = 0.24 if tier == "HIGH_CONFIDENCE" else 0.18
        active_clip_floor = 0.08 if tier == "HIGH_CONFIDENCE" else 0.055
    else:
        active_clip_cap = 0.42 if tier == "HIGH_CONFIDENCE" else 0.30
        active_clip_floor = 0.16 if tier == "HIGH_CONFIDENCE" else 0.09
    active_lower = min(max(target_low * active_clip_multiplier, -active_clip_cap), -active_clip_floor)
    active_upper = max(min(target_high * active_clip_multiplier, active_clip_cap), active_clip_floor)
    active_return = float(np.clip(active_return, active_lower, active_upper))

    active_threshold = 0.012 if horizon <= 7 else 0.035
    if active_return >= active_threshold:
        active_direction = "UP"
    elif active_return <= -active_threshold:
        active_direction = "DOWN"
    else:
        active_direction = "FLAT"
        if macro_directional_bias in {"UP", "DOWN"} and abs(active_return) >= active_threshold * 0.55:
            active_direction = macro_directional_bias

    active_spread_scale = float(
        np.clip(
            (1.00 if horizon <= 7 else 1.06) + 0.16 * (volatility_scale - 1.0),
            0.94 if horizon <= 7 else 0.98,
            1.16 if horizon <= 7 else 1.24,
        )
    )
    if tier == "NO_SIGNAL":
        active_spread_scale *= 0.55 if horizon <= 7 else 0.48
        active_spread_cap = 0.085 if horizon <= 7 else 0.16
    elif tier == "OBSERVE":
        active_spread_scale *= 0.72 if horizon <= 7 else 0.64
        active_spread_cap = 0.12 if horizon <= 7 else 0.22
    else:
        active_spread_cap = 0.18 if horizon <= 7 else 0.34
    active_spread_floor = 0.025 if horizon <= 7 else 0.06
    active_bear_spread = float(scenario_spread * active_spread_scale)
    active_bull_spread = float(scenario_spread * active_spread_scale)
    if active_direction == "UP":
        active_bull_spread *= 1.12
        active_bear_spread *= 0.88
    elif active_direction == "DOWN":
        active_bear_spread *= 1.12
        active_bull_spread *= 0.88
    else:
        active_bear_spread *= 0.78 if horizon <= 7 else 0.70
        active_bull_spread *= 0.78 if horizon <= 7 else 0.70
    if live_gap_penalty > 0.0:
        spread_shrink = float(np.clip(1.0 - 0.12 * live_gap_penalty, 0.76, 1.0))
        active_bear_spread *= spread_shrink
        active_bull_spread *= spread_shrink
    active_bear_spread = float(np.clip(active_bear_spread, active_spread_floor, active_spread_cap))
    active_bull_spread = float(np.clip(active_bull_spread, active_spread_floor, active_spread_cap))

    active_bear_return = float(np.clip(active_return - active_bear_spread, active_lower, active_upper))
    active_bull_return = float(np.clip(active_return + active_bull_spread, active_lower, active_upper))
    active_confidence = float(
        np.clip(
            0.34 * abs(hybrid_score)
            + 0.22 * classification_forecast.confidence
            + 0.18 * regime_forecast.confidence
            + 0.16 * abs(macro_spread)
            + 0.10 * (1.0 - volatility_stress),
            0.22,
            0.82 if horizon <= 7 else 0.86,
        )
    )

    return {
        "active_predicted_direction": active_direction,
        "active_confidence": active_confidence,
        "active_predicted_return": active_return,
        "active_predicted_close": float(price_reference.price * (1.0 + active_return)),
        "active_bear_predicted_return": active_bear_return,
        "active_bear_predicted_close": float(price_reference.price * (1.0 + active_bear_return)),
        "active_bull_predicted_return": active_bull_return,
        "active_bull_predicted_close": float(price_reference.price * (1.0 + active_bull_return)),
    }


def build_hybrid_forecast(
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    interval: str,
    horizon: int,
    price_reference: PriceReference,
    regression_forecast: ForecastResult,
    classification_forecast: DirectionForecastResult,
    regime_forecast: RegimeForecastResult,
    reversal_forecast: ReversalForecastResult,
    classification_backtest: pd.DataFrame | None = None,
    recent_holdout_report: pd.DataFrame | None = None,
) -> HybridForecastResult:
    latest_features = prediction_frame.iloc[-1]
    macro_context_summary = summarize_effective_macro_context(pd.Series(latest_features), horizon)
    training_target = pd.to_numeric(training_dataset["target_return"], errors="coerce")
    regime_reliability = 0.65 if str(regime_forecast.selection_basis).startswith("fallback_heuristic") else 1.0
    reversal_is_heuristic = str(reversal_forecast.selection_basis).startswith("fallback_heuristic")
    if reversal_is_heuristic:
        effective_reversal_forecast = ReversalForecastResult(
            model_name=reversal_forecast.model_name,
            selection_basis=reversal_forecast.selection_basis,
            prediction_timestamp=reversal_forecast.prediction_timestamp,
            horizon_steps=reversal_forecast.horizon_steps,
            model_input_close=reversal_forecast.model_input_close,
            last_close=reversal_forecast.last_close,
            reference_price_source=reversal_forecast.reference_price_source,
            reference_price_timestamp=reversal_forecast.reference_price_timestamp,
            predicted_reversal="NONE",
            probability_top_reversal=0.0,
            probability_no_reversal=1.0,
            probability_bottom_reversal=0.0,
            confidence=0.0,
        )
        reversal_reliability = 0.0
    else:
        effective_reversal_forecast = reversal_forecast
        reversal_reliability = 1.0
    trend_score = float(regime_forecast.probability_uptrend - regime_forecast.probability_downtrend) * regime_reliability
    reversal_score = float(
        effective_reversal_forecast.probability_bottom_reversal - effective_reversal_forecast.probability_top_reversal
    ) * reversal_reliability
    direction_score = float(classification_forecast.probability_up - classification_forecast.probability_down)
    forecast_score = compute_forecast_score(regression_forecast.predicted_return, training_target)
    fundamental_score = compute_slow_fundamental_score(pd.Series(latest_features))
    live_gap_pct = 0.0
    if (
        np.isfinite(regression_forecast.last_close)
        and np.isfinite(regression_forecast.model_input_close)
        and regression_forecast.model_input_close > 0.0
    ):
        live_gap_pct = float(regression_forecast.last_close / regression_forecast.model_input_close - 1.0)
    live_gap_abs = abs(live_gap_pct)
    live_gap_threshold = 0.03 if horizon <= 7 else 0.04
    live_gap_penalty = float(
        np.clip((live_gap_abs - live_gap_threshold) / (0.05 if horizon <= 7 else 0.06), 0.0, 1.0)
    )

    if horizon <= 7:
        hybrid_score = (
            0.26 * trend_score
            + 0.28 * reversal_score
            + 0.18 * direction_score
            + 0.20 * forecast_score
            + 0.08 * fundamental_score
        )
    else:
        hybrid_score = (
            0.30 * trend_score
            + 0.22 * reversal_score
            + 0.14 * direction_score
            + 0.20 * forecast_score
            + 0.14 * fundamental_score
        )
    hybrid_score = float(np.clip(hybrid_score, -1.0, 1.0))

    predicted_market_state = "SIDEWAYS"
    if regime_forecast.predicted_regime == "UPTREND" and hybrid_score >= 0.15:
        predicted_market_state = "UPTREND"
    elif regime_forecast.predicted_regime == "DOWNTREND" and hybrid_score <= -0.15:
        predicted_market_state = "DOWNTREND"

    predicted_reversal_signal = effective_reversal_forecast.predicted_reversal
    if effective_reversal_forecast.confidence < (0.48 if horizon <= 7 else 0.45) or predicted_reversal_signal == "NONE":
        predicted_reversal_signal = "NONE"

    if hybrid_score >= (0.14 if horizon <= 7 else 0.16) and classification_forecast.probability_up >= 0.50:
        bias_direction = "UP"
    elif hybrid_score <= -(0.14 if horizon <= 7 else 0.16) and classification_forecast.probability_up <= 0.50:
        bias_direction = "DOWN"
    else:
        bias_direction = "FLAT"

    signal_tier, high_confidence_signal, signal_reason, predicted_direction = determine_signal_tier(
        horizon=horizon,
        bias_direction=bias_direction,
        hybrid_score=hybrid_score,
        classification_forecast=classification_forecast,
        regime_forecast=regime_forecast,
        reversal_forecast=effective_reversal_forecast,
        classification_backtest=classification_backtest,
        recent_holdout_report=recent_holdout_report,
    )
    if live_gap_penalty > 0.0:
        hybrid_score = float(hybrid_score * (1.0 - 0.20 * live_gap_penalty))
        if signal_tier == "HIGH_CONFIDENCE":
            signal_tier = "OBSERVE"
            high_confidence_signal = False
            signal_reason = f"{signal_reason}+live_gap_guardrail"
        elif signal_tier == "OBSERVE":
            signal_tier = "NO_SIGNAL"
            high_confidence_signal = False
            predicted_direction = "FLAT"
            signal_reason = f"{signal_reason}+live_gap_guardrail"
        elif "live_gap_guardrail" not in signal_reason:
            signal_reason = f"{signal_reason}+live_gap_guardrail"

    recent_target = training_target.dropna()
    if recent_target.empty:
        target_low, target_high = (-0.10, 0.10) if horizon <= 7 else (-0.18, 0.18)
    else:
        target_low = float(np.nanquantile(recent_target.to_numpy(dtype=float), 0.02))
        target_high = float(np.nanquantile(recent_target.to_numpy(dtype=float), 0.98))
    target_scale, target_std, volatility_scale, scenario_spread = resolve_hybrid_volatility_profile(
        training_target=training_target,
        latest_features=pd.Series(latest_features),
        horizon=horizon,
    )

    def volatility_linked_multiplier(
        base: float,
        *,
        sensitivity: float,
        lower: float,
        upper: float,
    ) -> float:
        return float(np.clip(base + sensitivity * (volatility_scale - 1.0), lower, upper))

    adjusted_return = float(regression_forecast.predicted_return)
    hybrid_bias = target_scale * 0.35 * hybrid_score
    adjusted_return = 0.72 * adjusted_return + 0.28 * hybrid_bias

    if predicted_direction == "FLAT":
        adjusted_return *= volatility_linked_multiplier(
            0.72 if horizon <= 7 else 0.68,
            sensitivity=0.10,
            lower=0.62 if horizon <= 7 else 0.58,
            upper=0.90,
        )
    if predicted_market_state == "SIDEWAYS":
        adjusted_return *= volatility_linked_multiplier(
            0.68 if horizon <= 7 else 0.60,
            sensitivity=0.16,
            lower=0.55 if horizon <= 7 else 0.50,
            upper=0.92,
        )
    if predicted_reversal_signal == "NONE":
        adjusted_return *= volatility_linked_multiplier(
            0.95 if horizon <= 7 else 0.85,
            sensitivity=0.06,
            lower=0.85 if horizon <= 7 else 0.75,
            upper=1.00,
        )
    if signal_tier == "NO_SIGNAL":
        adjusted_return *= volatility_linked_multiplier(
            0.72 if horizon <= 7 else 0.62,
            sensitivity=0.12,
            lower=0.58 if horizon <= 7 else 0.52,
            upper=0.90,
        )
    elif signal_tier == "OBSERVE":
        adjusted_return *= volatility_linked_multiplier(
            0.88 if horizon <= 7 else 0.78,
            sensitivity=0.08,
            lower=0.75 if horizon <= 7 else 0.66,
            upper=0.98,
        )
    if live_gap_penalty > 0.0:
        adjusted_return *= float(
            np.clip(
                (0.84 if horizon <= 7 else 0.88) - (0.22 if horizon <= 7 else 0.20) * live_gap_penalty,
                0.55,
                0.95,
            )
        )
    if predicted_direction == "UP" and signal_tier in {"OBSERVE", "HIGH_CONFIDENCE"} and adjusted_return < 0.0:
        directional_floor = max(
            target_scale * (0.07 if signal_tier == "OBSERVE" else 0.12),
            scenario_spread * (0.16 if signal_tier == "OBSERVE" else 0.22),
        )
        adjusted_return = max(abs(adjusted_return) * 0.35, directional_floor)
    elif predicted_direction == "DOWN" and signal_tier in {"OBSERVE", "HIGH_CONFIDENCE"} and adjusted_return > 0.0:
        directional_floor = max(
            target_scale * (0.07 if signal_tier == "OBSERVE" else 0.12),
            scenario_spread * (0.16 if signal_tier == "OBSERVE" else 0.22),
        )
        adjusted_return = -max(abs(adjusted_return) * 0.35, directional_floor)
    if predicted_direction == "FLAT" and bias_direction in {"UP", "DOWN"} and signal_tier in {"OBSERVE", "HIGH_CONFIDENCE"}:
        bias_floor = max(
            target_scale * (0.025 if signal_tier == "OBSERVE" else 0.05),
            scenario_spread * (0.06 if signal_tier == "OBSERVE" else 0.10),
            0.0015 if horizon <= 7 else 0.0040,
        )
        if bias_direction == "UP" and adjusted_return < bias_floor:
            adjusted_return = bias_floor
        elif bias_direction == "DOWN" and adjusted_return > -bias_floor:
            adjusted_return = -bias_floor
    if np.sign(adjusted_return) != np.sign(hybrid_score) and abs(hybrid_score) >= 0.12:
        adjusted_return *= volatility_linked_multiplier(
            0.72,
            sensitivity=0.06,
            lower=0.60,
            upper=0.90,
        )
    if predicted_reversal_signal == "BOTTOM_REVERSAL" and adjusted_return < 0.0:
        adjusted_return = max(adjusted_return * 0.35, 0.0) + max(hybrid_bias, 0.0) * 0.40
    elif predicted_reversal_signal == "TOP_REVERSAL" and adjusted_return > 0.0:
        adjusted_return = min(adjusted_return * 0.35, 0.0) + min(hybrid_bias, 0.0) * 0.40
    if predicted_market_state == "SIDEWAYS":
        sideways_cap = max(
            target_scale * (1.10 if horizon <= 7 else 1.20),
            scenario_spread * (0.82 if signal_tier != "NO_SIGNAL" else 0.62),
            0.08 if horizon <= 7 else 0.16,
        )
        adjusted_return = float(np.clip(adjusted_return, -sideways_cap, sideways_cap))

    if signal_tier == "NO_SIGNAL":
        clip_multiplier = 1.10 if horizon <= 7 else 1.15
        clip_cap = 0.16 if horizon <= 7 else 0.28
        clip_floor = 0.06 if horizon <= 7 else 0.09
    elif signal_tier == "OBSERVE":
        clip_multiplier = 1.18 if horizon <= 7 else 1.24
        clip_cap = 0.20 if horizon <= 7 else 0.34
        clip_floor = 0.07 if horizon <= 7 else 0.12
    else:
        clip_multiplier = 1.25 if horizon <= 7 else 1.35
        clip_cap = 0.24 if horizon <= 7 else 0.42
        clip_floor = 0.09 if horizon <= 7 else 0.16
    if live_gap_penalty > 0.0:
        clip_cap = max(clip_floor, clip_cap * (1.0 - 0.16 * live_gap_penalty))
    clip_lower = min(max(target_low * clip_multiplier, -clip_cap), -clip_floor)
    clip_upper = max(min(target_high * clip_multiplier, clip_cap), clip_floor)
    active_track = build_active_forecast_track(
        horizon=horizon,
        price_reference=price_reference,
        regression_forecast=regression_forecast,
        classification_forecast=classification_forecast,
        regime_forecast=regime_forecast,
        effective_reversal_forecast=effective_reversal_forecast,
        hybrid_score=hybrid_score,
        trend_score=trend_score,
        direction_score=direction_score,
        reversal_score=reversal_score,
        fundamental_score=fundamental_score,
        target_scale=target_scale,
        target_low=target_low,
        target_high=target_high,
        scenario_spread=scenario_spread,
        volatility_scale=volatility_scale,
        macro_context_summary=macro_context_summary,
        live_gap_penalty=live_gap_penalty,
        signal_tier=signal_tier,
    )
    adjusted_return = float(
        np.clip(
            adjusted_return,
            clip_lower,
            clip_upper,
        )
    )
    scenario_direction = predicted_direction if predicted_direction != "FLAT" else bias_direction
    bear_spread = float(scenario_spread)
    bull_spread = float(scenario_spread)
    if scenario_direction == "UP":
        bull_spread *= 1.15
        bear_spread *= 0.85
    elif scenario_direction == "DOWN":
        bear_spread *= 1.15
        bull_spread *= 0.85
    if signal_tier == "NO_SIGNAL":
        bear_spread *= 0.72 if horizon <= 7 else 0.65
        bull_spread *= 0.72 if horizon <= 7 else 0.65
    elif signal_tier == "OBSERVE":
        bear_spread *= 0.88 if horizon <= 7 else 0.82
        bull_spread *= 0.88 if horizon <= 7 else 0.82
    elif signal_tier == "HIGH_CONFIDENCE":
        bear_spread *= 1.05
        bull_spread *= 1.05
    if live_gap_penalty > 0.0:
        spread_shrink = float(np.clip(1.0 - 0.18 * live_gap_penalty, 0.72, 1.0))
        bear_spread *= spread_shrink
        bull_spread *= spread_shrink
    bear_return = float(np.clip(adjusted_return - bear_spread, clip_lower, clip_upper))
    bull_return = float(np.clip(adjusted_return + bull_spread, clip_lower, clip_upper))
    confidence = float(
        np.clip(
            0.45 * abs(hybrid_score) + 0.40 * regime_forecast.confidence + 0.15 * effective_reversal_forecast.confidence,
            0.0,
            1.0,
        )
    )
    if live_gap_penalty > 0.0:
        confidence *= float(np.clip(1.0 - 0.30 * live_gap_penalty, 0.60, 1.0))
    if signal_tier == "HIGH_CONFIDENCE":
        confidence = float(np.clip(max(confidence, 0.72 if horizon <= 7 else 0.76), 0.0, 1.0))
    elif signal_tier == "OBSERVE":
        confidence = float(np.clip(max(confidence, 0.52 if horizon <= 7 else 0.56), 0.0, 1.0))
    else:
        confidence = float(np.clip(min(confidence, 0.38 if horizon <= 7 else 0.46), 0.0, 1.0))
    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")
    return HybridForecastResult(
        model_name="hybrid_regime_reversal_gate",
        selection_basis=(
            f"regime={regime_forecast.model_name};"
            f"reversal={effective_reversal_forecast.model_name};"
            f"direction={classification_forecast.model_name};"
            f"regression={regression_forecast.model_name}"
        ),
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=float(prediction_frame["eth_close"].iloc[-1]),
        last_close=float(price_reference.price),
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_market_state=predicted_market_state,
        predicted_reversal_signal=predicted_reversal_signal,
        predicted_direction=predicted_direction,
        trend_score=trend_score,
        reversal_score=reversal_score,
        direction_score=direction_score,
        forecast_score=forecast_score,
        fundamental_score=fundamental_score,
        hybrid_score=hybrid_score,
        confidence=confidence,
        bias_direction=bias_direction,
        signal_tier=signal_tier,
        high_confidence_signal=high_confidence_signal,
        signal_reason=signal_reason,
        volatility_scale=volatility_scale,
        scenario_spread=scenario_spread,
        macro_readiness_status=str(macro_context_summary.get("readiness_status", "missing")),
        macro_risk_regime=str(macro_context_summary.get("risk_regime", "BALANCED")),
        macro_directional_bias=str(macro_context_summary.get("directional_bias", "FLAT")),
        macro_selection_tag=str(macro_context_summary.get("selection_tag", "macro_balanced")),
        macro_risk_on_score=float(macro_context_summary.get("risk_on_score", 0.0)),
        macro_risk_off_score=float(macro_context_summary.get("risk_off_score", 0.0)),
        macro_trend_bias_score=float(macro_context_summary.get("trend_bias_score", 0.0)),
        macro_volatility_stress=float(macro_context_summary.get("volatility_stress", 0.0)),
        macro_used_feature_count=int(macro_context_summary.get("used_feature_count", 0)),
        macro_missing_feature_count=int(macro_context_summary.get("missing_feature_count", 0)),
        macro_used_features=str(macro_context_summary.get("used_features", "")),
        macro_missing_features=str(macro_context_summary.get("missing_features", "")),
        active_predicted_direction=str(active_track["active_predicted_direction"]),
        active_confidence=float(active_track["active_confidence"]),
        active_predicted_return=float(active_track["active_predicted_return"]),
        active_predicted_close=float(active_track["active_predicted_close"]),
        active_bear_predicted_return=float(active_track["active_bear_predicted_return"]),
        active_bear_predicted_close=float(active_track["active_bear_predicted_close"]),
        active_bull_predicted_return=float(active_track["active_bull_predicted_return"]),
        active_bull_predicted_close=float(active_track["active_bull_predicted_close"]),
        adjusted_predicted_return=adjusted_return,
        adjusted_predicted_close=float(price_reference.price * (1.0 + adjusted_return)),
        bear_predicted_return=bear_return,
        bear_predicted_close=float(price_reference.price * (1.0 + bear_return)),
        bull_predicted_return=bull_return,
        bull_predicted_close=float(price_reference.price * (1.0 + bull_return)),
    )


def forecast_next_step(
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    interval: str,
    horizon: int,
    model_name: str,
    selection_basis: str,
    price_reference: PriceReference,
    sample_weight: pd.Series | None = None,
    regression_oof_predictions: pd.DataFrame | None = None,
    regression_leaderboard: pd.DataFrame | None = None,
    regression_backtest: pd.DataFrame | None = None,
    recent_holdout_report: pd.DataFrame | None = None,
    prediction_feedback_summary: pd.DataFrame | None = None,
) -> ForecastResult:
    selection_basis_text = str(selection_basis)
    score_frame = build_regression_model_score_frame(
        regression_leaderboard=regression_leaderboard,
        regression_backtest=regression_backtest,
        recent_holdout_report=recent_holdout_report,
        prediction_feedback_summary=prediction_feedback_summary,
        horizon=horizon,
    )
    latest_features = prediction_frame.iloc[[-1]][feature_columns]
    model_input_close = float(prediction_frame["eth_close"].iloc[-1])
    last_close = float(price_reference.price)
    live_gap = (
        float(last_close / model_input_close - 1.0)
        if np.isfinite(last_close) and np.isfinite(model_input_close) and model_input_close > 0.0
        else 0.0
    )

    blended_oof_prediction = None
    if model_name == TRIMMED_REGRESSION_ENSEMBLE_MODEL:
        score_row = score_frame.loc[score_frame["model"] == model_name] if not score_frame.empty else pd.DataFrame()
        component_models = parse_component_models(score_row.iloc[0].get("component_models")) if not score_row.empty else []
        if len(component_models) < 2:
            component_models = select_trimmed_regression_ensemble_members(
                score_frame,
                horizon=horizon,
                oof_predictions=regression_oof_predictions,
            )
        component_models = [component for component in component_models if component != model_name]
        if len(component_models) < 2:
            raise ValueError("Trimmed regression ensemble was selected without enough component models.")
        component_predictions: dict[str, float] = {}
        point_model = None
        for component_model in component_models:
            fitted_model = fit_best_model(
                training_dataset,
                feature_columns,
                component_model,
                horizon=horizon,
                sample_weight=sample_weight,
            )
            if point_model is None:
                point_model = fitted_model
            component_raw = float(fitted_model.predict(latest_features)[0])
            component_adjusted = float(
                apply_regime_response_overlay(
                    pd.Series([component_raw], index=latest_features.index, dtype=float),
                    latest_features,
                    horizon=horizon,
                    live_gap=live_gap,
                ).iloc[0]
            )
            component_predictions[component_model] = component_adjusted
        regression_weights = derive_ensemble_weights_from_leaderboard(
            score_frame, list(component_models), "price_rmse", higher_is_better=False
        )
        predicted_return = float(
            skill_weighted_trimmed_mean(
                pd.DataFrame([component_predictions], index=latest_features.index),
                weights=regression_weights,
            ).iloc[0]
        )
        blended_oof_prediction = build_blended_regression_oof(
            regression_oof_predictions,
            regression_weights or {
                component_model: 1.0 / len(component_models) for component_model in component_models
            },
        )
        selection_basis_text = (
            f"{selection_basis_text}+skill_weighted[{ '|'.join(component_models) }]"
        )
        if point_model is None:
            raise ValueError("Trimmed regression ensemble did not resolve to a fitted component model.")
    else:
        point_model = fit_best_model(
            training_dataset,
            feature_columns,
            model_name,
            horizon=horizon,
            sample_weight=sample_weight,
        )
        raw_prediction = float(point_model.predict(latest_features)[0])
        predicted_return = float(
            apply_regime_response_overlay(
                pd.Series([raw_prediction], index=latest_features.index, dtype=float),
                latest_features,
                horizon=horizon,
                live_gap=live_gap,
            ).iloc[0]
        )

    component_models = choose_regression_blend_components(
        score_frame=score_frame,
        selected_model=model_name,
        horizon=horizon,
    )
    component_predictions: dict[str, float] = {model_name: predicted_return}
    if model_name != TRIMMED_REGRESSION_ENSEMBLE_MODEL and len(component_models) >= 2:
        for peer_model in component_models:
            if peer_model == model_name:
                continue
            peer_fitted = fit_best_model(
                training_dataset,
                feature_columns,
                peer_model,
                horizon=horizon,
                sample_weight=sample_weight,
            )
            peer_raw = float(peer_fitted.predict(latest_features)[0])
            peer_adjusted = float(
                apply_regime_response_overlay(
                    pd.Series([peer_raw], index=latest_features.index, dtype=float),
                    latest_features,
                    horizon=horizon,
                    live_gap=live_gap,
                ).iloc[0]
            )
            component_predictions[peer_model] = peer_adjusted

        component_weights = compute_regression_component_weights(score_frame, list(component_predictions.keys()))
        predicted_return, blend_applied, clip_applied = apply_regression_peer_guardrail(
            base_prediction=predicted_return,
            component_predictions=component_predictions,
            component_weights=component_weights,
            training_dataset=training_dataset,
            horizon=horizon,
        )
        if blend_applied:
            selection_basis_text = f"{selection_basis_text}+peer_blend"
        if clip_applied:
            selection_basis_text = f"{selection_basis_text}+sanity_clip"
        blended_oof_prediction = build_blended_regression_oof(
            regression_oof_predictions,
            component_weights,
        )

    calibration_oof_prediction = blended_oof_prediction
    if (
        calibration_oof_prediction is None
        and regression_oof_predictions is not None
        and f"{model_name}_pred_return" in regression_oof_predictions.columns
    ):
        calibration_oof_prediction = pd.to_numeric(
            regression_oof_predictions[f"{model_name}_pred_return"],
            errors="coerce",
        )
    predicted_return, calibrated_oof_prediction, calibration_meta = apply_oof_return_calibration(
        predicted_return=predicted_return,
        residual_oof_prediction=calibration_oof_prediction,
        training_dataset=training_dataset,
        horizon=horizon,
    )
    if bool(calibration_meta.get("applied")):
        selection_basis_text = (
            f"{selection_basis_text}+oof_calibrated[slope={float(calibration_meta.get('slope', 1.0)):.2f}]"
        )
        blended_oof_prediction = calibrated_oof_prediction

    lower_offset, upper_offset = estimate_prediction_interval_from_residuals(
        training_dataset=training_dataset,
        feature_columns=feature_columns,
        point_model=point_model,
        model_name=model_name,
        oof_predictions=regression_oof_predictions,
        residual_oof_prediction=blended_oof_prediction,
    )
    selection_basis_text = f"{selection_basis_text}+conformal_interval"
    lower_return = predicted_return + lower_offset
    upper_return = predicted_return + upper_offset
    lower_return, upper_return = sorted((lower_return, upper_return))
    lower_return = min(lower_return, predicted_return)
    upper_return = max(upper_return, predicted_return)

    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")

    return ForecastResult(
        model_name=model_name,
        selection_basis=selection_basis_text,
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=model_input_close,
        last_close=last_close,
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_return=predicted_return,
        predicted_close=last_close * (1.0 + predicted_return),
        lower_return_10=lower_return,
        lower_close_10=last_close * (1.0 + lower_return),
        upper_return_90=upper_return,
        upper_close_90=last_close * (1.0 + upper_return),
    )


def forecast_direction(
    training_dataset: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    feature_columns: list[str],
    interval: str,
    horizon: int,
    model_name: str,
    selection_basis: str,
    signal_threshold: float,
    price_reference: PriceReference,
    sample_weight: pd.Series | None = None,
    classification_leaderboard: pd.DataFrame | None = None,
    classification_backtest: pd.DataFrame | None = None,
    recent_holdout_report: pd.DataFrame | None = None,
) -> DirectionForecastResult:
    latest_features = prediction_frame.iloc[[-1]][feature_columns]
    model_input_close = float(prediction_frame["eth_close"].iloc[-1])
    last_close = float(price_reference.price)
    live_gap = (
        float(last_close / model_input_close - 1.0)
        if np.isfinite(last_close) and np.isfinite(model_input_close) and model_input_close > 0.0
        else 0.0
    )
    if model_name == TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL:
        component_models: list[str] = []
        if classification_leaderboard is not None and not classification_leaderboard.empty:
            score_row = classification_leaderboard.loc[classification_leaderboard["model"] == model_name]
            if not score_row.empty:
                component_models = parse_component_models(score_row.iloc[0].get("component_models"))
        if len(component_models) < 2:
            component_models = select_trimmed_classification_ensemble_members(
                classification_leaderboard if classification_leaderboard is not None else pd.DataFrame(),
                horizon=horizon,
            )
        component_models = [component for component in component_models if component != model_name]
        if len(component_models) < 2:
            raise ValueError("Trimmed direction ensemble was selected without enough component models.")
        component_probabilities: dict[str, float] = {}
        for component_model in component_models:
            component_classifier = fit_best_classifier(
                training_dataset,
                feature_columns,
                component_model,
                horizon=horizon,
                sample_weight=sample_weight,
            )
            component_probability = float(
                apply_direction_regime_overlay(
                    pd.Series(
                        component_classifier.predict_proba(latest_features)[:, 1],
                        index=latest_features.index,
                        dtype=float,
                    ),
                    latest_features,
                    horizon=horizon,
                    live_gap=live_gap,
                ).iloc[0]
            )
            component_probabilities[component_model] = component_probability
        classification_weights = derive_ensemble_weights_from_leaderboard(
            classification_leaderboard, list(component_models), "brier_score", higher_is_better=False
        )
        probability_up = float(
            skill_weighted_trimmed_mean(
                pd.DataFrame([component_probabilities], index=latest_features.index),
                weights=classification_weights,
            ).iloc[0]
        )
        selection_basis = f"{selection_basis}+skill_weighted[{ '|'.join(component_models) }]"
    else:
        model = fit_best_classifier(
            training_dataset,
            feature_columns,
            model_name,
            horizon=horizon,
            sample_weight=sample_weight,
        )
        probability_up = float(
            apply_direction_regime_overlay(
                pd.Series(model.predict_proba(latest_features)[:, 1], index=latest_features.index, dtype=float),
                latest_features,
                horizon=horizon,
                live_gap=live_gap,
            ).iloc[0]
        )
    probability_down = float(1.0 - probability_up)
    lower_threshold, upper_threshold = resolve_classification_signal_thresholds(signal_threshold)
    if probability_up >= upper_threshold:
        predicted_direction = "UP"
    elif probability_up <= lower_threshold:
        predicted_direction = "DOWN"
    else:
        predicted_direction = "FLAT"
    confidence = calibrate_direction_confidence(
        model_name=model_name,
        selection_basis=selection_basis,
        predicted_direction=predicted_direction,
        probability_up=probability_up,
        lower_threshold=lower_threshold,
        upper_threshold=upper_threshold,
        classification_leaderboard=classification_leaderboard,
        classification_backtest=classification_backtest,
        recent_holdout_report=recent_holdout_report,
        horizon=horizon,
    )
    offset = interval_to_offset(interval) * horizon
    prediction_timestamp = (prediction_frame.index[-1] + offset).strftime("%Y-%m-%d %H:%M:%S")

    return DirectionForecastResult(
        model_name=model_name,
        selection_basis=selection_basis,
        prediction_timestamp=prediction_timestamp,
        horizon_steps=horizon,
        model_input_close=model_input_close,
        last_close=last_close,
        reference_price_source=price_reference.source,
        reference_price_timestamp=price_reference.timestamp,
        predicted_direction=predicted_direction,
        signal_threshold=upper_threshold,
        probability_up=probability_up,
        probability_down=probability_down,
        confidence=confidence,
    )


def run_pipeline(
    period: str = "max",
    interval: str = "1d",
    horizon: int = 7,
    cv_splits: int = 3,
    cv_test_size: int = 20,
    output_dir: str | Path = "eth_forecast_outputs",
    dataset_csv: str | Path = "market_data_cache.csv",
    master_data_csv: str | Path | None = None,
    prediction_history_csv: str | Path | None = None,
    external_feature_csvs: list[str] | None = None,
    drive_root: str | Path | None = None,
    use_dataset_cache: bool = True,
    use_prediction_history_features: bool = True,
    cache_lookback_rows: int = CACHE_UPDATE_LOOKBACK_ROWS,
    min_feature_coverage: float = DEFAULT_FEATURE_MIN_COVERAGE,
    use_realtime_reference_price: bool = True,
    build_explainability: bool = True,
    fast_mode: bool = False,
    compact_output: bool = False,
    save_outputs: bool = True,
    verbose: bool = True,
) -> HorizonArtifacts:
    previous_fast_mode = runtime_fast_mode_enabled()
    set_runtime_options(fast_mode=fast_mode)
    try:
        market_data, output_dir_path, _, history_path, _ = load_or_prepare_market_data(
            period=period,
            interval=interval,
            output_dir=output_dir,
            dataset_csv=dataset_csv,
            prediction_history_csv=prediction_history_csv,
            external_feature_csvs=external_feature_csvs,
            drive_root=drive_root,
            use_dataset_cache=use_dataset_cache,
            cache_lookback_rows=cache_lookback_rows,
            master_data_csv=master_data_csv,
            verbose=verbose,
        )
        prediction_history = (
            load_prediction_history_csv(history_path)
            if history_path is not None
            else pd.DataFrame()
        )
        prediction_history = refresh_prediction_history_with_realizations(prediction_history, market_data)
        prediction_feedback_summary = build_prediction_feedback_summary(prediction_history)
        prediction_feedback_state_summary = build_prediction_feedback_state_summary(prediction_history)
        prediction_history_features = prediction_history if use_prediction_history_features else pd.DataFrame()
        shared_price_reference = build_shared_price_reference(
            market_data,
            prefer_realtime=use_realtime_reference_price,
        )
        artifacts = run_horizon_pipeline(
            market_data=market_data,
            interval=interval,
            horizon=horizon,
            cv_splits=cv_splits,
            cv_test_size=cv_test_size,
            prediction_history=prediction_history_features,
            prediction_feedback_summary=prediction_feedback_summary,
            prediction_feedback_state_summary=prediction_feedback_state_summary,
            min_feature_coverage=min_feature_coverage,
            use_realtime_reference_price=use_realtime_reference_price,
            shared_price_reference=shared_price_reference,
            build_explainability=build_explainability,
            verbose=verbose,
        )
        if save_outputs:
            pipeline_artifacts = PipelineArtifacts(
                raw_data=market_data,
                horizons={horizon: artifacts},
            )
            save_artifacts(
                pipeline_artifacts,
                output_dir_path,
                prediction_history_csv=history_path,
                compact_output=compact_output,
            )
        return artifacts
    finally:
        set_runtime_options(fast_mode=previous_fast_mode)


def run_horizon_pipeline(
    market_data: pd.DataFrame,
    interval: str,
    horizon: int,
    cv_splits: int,
    cv_test_size: int,
    prediction_history: pd.DataFrame | None = None,
    prediction_feedback_summary: pd.DataFrame | None = None,
    prediction_feedback_state_summary: pd.DataFrame | None = None,
    min_feature_coverage: float = DEFAULT_FEATURE_MIN_COVERAGE,
    use_realtime_reference_price: bool = True,
    shared_price_reference: PriceReference | None = None,
    build_explainability: bool = True,
    verbose: bool = True,
    cv_embargo: int = 0,
) -> HorizonArtifacts:
    label = f"[{horizon}d] "
    state_target_columns = [
        "target_regime",
        "target_reversal_state",
        "target_bottom_reversal",
        "target_top_reversal",
    ]
    log_progress(verbose, f"{label}building features...")
    feature_frame, feature_columns = build_features(
        market_data,
        horizon=horizon,
        prediction_history=prediction_history,
    )
    candidate_feature_count = len(feature_columns)
    raw_candidate_feature_columns = list(feature_columns)
    full_training_mask = (
        feature_frame["target_return"].notna()
        & feature_frame["target_close"].notna()
        & feature_frame["eth_close"].notna()
    )
    prediction_mask = feature_frame["eth_close"].notna()
    full_training_dataset = feature_frame.loc[
        full_training_mask,
        feature_columns + ["eth_close", "target_return", "target_close", *state_target_columns],
    ].copy()
    full_training_rows = len(full_training_dataset)
    trimmed_training_dataset = trim_training_window(
        full_training_dataset,
        interval=interval,
        horizon=horizon,
    )
    training_mask = feature_frame.index.isin(trimmed_training_dataset.index) & full_training_mask
    feature_columns, feature_coverage_summary = select_usable_feature_columns(
        feature_frame=feature_frame,
        feature_columns=feature_columns,
        training_mask=training_mask,
        min_feature_coverage=min_feature_coverage,
        horizon=horizon,
    )
    training_dataset = feature_frame.loc[
        training_mask,
        raw_candidate_feature_columns + ["eth_close", "target_return", "target_close", *state_target_columns],
    ].copy()
    prediction_frame = feature_frame.loc[prediction_mask, feature_columns + ["eth_close"]].copy()
    training_window_years = training_window_years_for_horizon(horizon)
    sample_weight_half_life_years = sample_weight_half_life_years_for_horizon(horizon)
    sample_weights = build_time_decay_sample_weights(training_dataset.index, interval, horizon)

    if prediction_frame.empty:
        raise ValueError(f"No usable prediction rows were produced for horizon {horizon}.")
    price_reference = (
        shared_price_reference
        if shared_price_reference is not None
        else resolve_price_reference(
            model_input_close=float(prediction_frame["eth_close"].iloc[-1]),
            model_input_timestamp=prediction_frame.index[-1],
            prefer_realtime=use_realtime_reference_price,
        )
    )

    min_required = max(200, cv_splits * cv_test_size + horizon + 60)
    if len(training_dataset) < min_required:
        raise ValueError(
            f"Need at least {min_required} cleaned rows for the current configuration, got {len(training_dataset)}."
        )

    data_window_summary = build_data_window_summary(
        raw_data=market_data,
        training_dataset=training_dataset,
        prediction_frame=prediction_frame,
        feature_columns=feature_columns,
        candidate_feature_count=candidate_feature_count,
        interval=interval,
        horizon=horizon,
        cv_splits=cv_splits,
        cv_test_size=cv_test_size,
        training_window_years=training_window_years,
        sample_weight_half_life_years=sample_weight_half_life_years,
        sample_weights=sample_weights,
        full_training_rows=full_training_rows,
    )

    log_progress(verbose, f"{label}running regression walk-forward validation...")
    regression_leaderboard, regression_oof_predictions = walk_forward_leaderboard(
        dataset=training_dataset,
        feature_columns=raw_candidate_feature_columns,
        n_splits=cv_splits,
        test_size=cv_test_size,
        gap=horizon,
        sample_weight=sample_weights,
        verbose=verbose,
        label=label,
        fold_feature_selection=True,
        min_feature_coverage=min_feature_coverage,
        embargo=cv_embargo,
    )
    log_progress(verbose, f"{label}running classification walk-forward validation...")
    classification_leaderboard, classification_oof_predictions = walk_forward_classification(
        dataset=training_dataset,
        feature_columns=raw_candidate_feature_columns,
        n_splits=cv_splits,
        test_size=cv_test_size,
        gap=horizon,
        sample_weight=sample_weights,
        verbose=verbose,
        label=label,
        fold_feature_selection=True,
        min_feature_coverage=min_feature_coverage,
        embargo=cv_embargo,
    )
    log_progress(verbose, f"{label}running regime-state walk-forward validation...")
    regime_leaderboard, regime_oof_predictions = walk_forward_state_classification(
        dataset=training_dataset,
        feature_columns=raw_candidate_feature_columns,
        n_splits=cv_splits,
        test_size=cv_test_size,
        gap=horizon,
        target_column="target_regime",
        task_key="regime",
        sample_weight=sample_weights,
        verbose=verbose,
        label=label,
        fold_feature_selection=True,
        min_feature_coverage=min_feature_coverage,
        embargo=cv_embargo,
    )
    log_progress(verbose, f"{label}running reversal-state walk-forward validation...")
    reversal_leaderboard, reversal_oof_predictions = walk_forward_state_classification(
        dataset=training_dataset,
        feature_columns=raw_candidate_feature_columns,
        n_splits=cv_splits,
        test_size=cv_test_size,
        gap=horizon,
        target_column="target_reversal_state",
        task_key="reversal",
        sample_weight=sample_weights,
        verbose=verbose,
        label=label,
        fold_feature_selection=True,
        min_feature_coverage=min_feature_coverage,
        embargo=cv_embargo,
    )
    if regression_leaderboard.empty:
        raise ValueError(f"No regression leaderboard was produced for horizon {horizon}.")
    if classification_leaderboard.empty:
        raise ValueError(f"No classification leaderboard was produced for horizon {horizon}.")

    regression_ensemble_members = select_trimmed_regression_ensemble_members(
        regression_leaderboard,
        horizon=horizon,
        oof_predictions=regression_oof_predictions,
    )
    classification_ensemble_members = select_trimmed_classification_ensemble_members(
        classification_leaderboard,
        horizon=horizon,
        oof_predictions=classification_oof_predictions,
    )
    regression_leaderboard, regression_oof_predictions = append_trimmed_regression_ensemble_candidate(
        regression_leaderboard,
        regression_oof_predictions,
        training_dataset,
        horizon,
        regression_ensemble_members,
    )
    classification_leaderboard, classification_oof_predictions = append_trimmed_classification_ensemble_candidate(
        classification_leaderboard,
        classification_oof_predictions,
        training_dataset,
        horizon,
        classification_ensemble_members,
    )

    log_progress(verbose, f"{label}running model backtests...")
    regression_backtest, regression_equity_curves = backtest_regression_models(
        regression_oof_predictions,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    classification_backtest, classification_equity_curves = backtest_classification_models(
        classification_oof_predictions,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    regime_backtest, regime_equity_curves = backtest_state_models(
        regime_oof_predictions,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
        task_key="regime",
    )
    reversal_backtest, reversal_equity_curves = backtest_state_models(
        reversal_oof_predictions,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
        task_key="reversal",
    )
    reference_index = regression_oof_predictions.index[
        regression_oof_predictions.filter(like="_pred_return").notna().any(axis=1)
    ].union(
        classification_oof_predictions.index[
            classification_oof_predictions.filter(like="_prob_up").notna().any(axis=1)
        ]
    ).union(
        regime_oof_predictions.index[
            regime_oof_predictions.filter(like="__regime_prob_negative").notna().any(axis=1)
        ]
    ).union(
        reversal_oof_predictions.index[
            reversal_oof_predictions.filter(like="__reversal_prob_negative").notna().any(axis=1)
        ]
    )
    benchmark_backtest, benchmark_equity_curves = backtest_benchmark_strategies(
        reference_index=reference_index,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    effective_macro_context = summarize_effective_macro_context(prediction_frame.iloc[-1], horizon)
    macro_bucket_label = str(effective_macro_context.get("risk_regime", "BALANCED"))
    historical_macro_bucket = build_historical_macro_bucket_series(
        training_dataset[feature_columns],
        horizon=horizon,
    )
    regression_backtest = augment_backtest_with_macro_bucket(
        regression_backtest,
        signal_builder=lambda model_name, row: regression_signal_from_predictions(
            regression_oof_predictions[f"{model_name}_pred_return"],
            threshold=float(pd.to_numeric(row.get("signal_threshold"), errors="coerce") or 0.0),
        ),
        macro_bucket_series=historical_macro_bucket,
        target_bucket_label=macro_bucket_label,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    classification_backtest = augment_backtest_with_macro_bucket(
        classification_backtest,
        signal_builder=lambda model_name, row: classification_signal_from_probabilities(
            classification_oof_predictions[f"{model_name}_prob_up"],
            threshold=float(
                pd.to_numeric(row.get("signal_threshold"), errors="coerce")
                if pd.notna(pd.to_numeric(row.get("signal_threshold"), errors="coerce"))
                else DEFAULT_CLASSIFICATION_SIGNAL_THRESHOLD
            ),
        ),
        macro_bucket_series=historical_macro_bucket,
        target_bucket_label=macro_bucket_label,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    regime_backtest = augment_backtest_with_macro_bucket(
        regime_backtest,
        signal_builder=lambda model_name, row: state_signal_from_probabilities(
            regime_oof_predictions[f"{model_name}__regime_prob_negative"],
            regime_oof_predictions[f"{model_name}__regime_prob_neutral"],
            regime_oof_predictions[f"{model_name}__regime_prob_positive"],
            confidence_threshold=float(pd.to_numeric(row.get("signal_threshold"), errors="coerce") or 0.40),
        ),
        macro_bucket_series=historical_macro_bucket,
        target_bucket_label=macro_bucket_label,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    reversal_backtest = augment_backtest_with_macro_bucket(
        reversal_backtest,
        signal_builder=lambda model_name, row: state_signal_from_probabilities(
            reversal_oof_predictions[f"{model_name}__reversal_prob_negative"],
            reversal_oof_predictions[f"{model_name}__reversal_prob_neutral"],
            reversal_oof_predictions[f"{model_name}__reversal_prob_positive"],
            confidence_threshold=float(pd.to_numeric(row.get("signal_threshold"), errors="coerce") or 0.40),
        ),
        macro_bucket_series=historical_macro_bucket,
        target_bucket_label=macro_bucket_label,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
    )
    regression_thresholds = extract_model_signal_thresholds(regression_backtest)
    classification_thresholds = extract_model_signal_thresholds(classification_backtest)
    recent_holdout_report = build_recent_holdout_report(
        dataset=training_dataset,
        feature_columns=feature_columns,
        market_data=market_data,
        horizon=horizon,
        interval=interval,
        cv_test_size=cv_test_size,
        sample_weight=sample_weights,
        regression_thresholds=regression_thresholds,
        classification_thresholds=classification_thresholds,
        regression_ensemble_members=regression_ensemble_members,
        classification_ensemble_members=classification_ensemble_members,
    )
    data_window_summary["recent_holdout_rows"] = (
        int(pd.to_numeric(recent_holdout_report.get("evaluation_rows"), errors="coerce").max())
        if not recent_holdout_report.empty and "evaluation_rows" in recent_holdout_report.columns
        else 0
    )
    recent_holdout_embargo_rows = (
        int(pd.to_numeric(recent_holdout_report.get("embargo_rows"), errors="coerce").max())
        if not recent_holdout_report.empty and "embargo_rows" in recent_holdout_report.columns
        else 0
    )
    data_window_summary["recent_holdout_embargo_rows"] = recent_holdout_embargo_rows
    recent_holdout_scheme_values = (
        recent_holdout_report["validation_scheme"].dropna().astype(str).unique().tolist()
        if not recent_holdout_report.empty and "validation_scheme" in recent_holdout_report.columns
        else []
    )
    threshold_scheme_values: list[str] = []
    threshold_embargo_values: list[int] = []
    for backtest_frame in (regression_backtest, classification_backtest):
        if not isinstance(backtest_frame, pd.DataFrame) or backtest_frame.empty:
            continue
        if "threshold_validation_scheme" in backtest_frame.columns:
            threshold_scheme_values.extend(
                [value for value in backtest_frame["threshold_validation_scheme"].dropna().astype(str).tolist() if value]
            )
        if "validation_embargo_rows" in backtest_frame.columns:
            threshold_embargo_values.extend(
                [
                    int(value)
                    for value in pd.to_numeric(backtest_frame["validation_embargo_rows"], errors="coerce").dropna().tolist()
                ]
            )
    unique_threshold_schemes = sorted(set(threshold_scheme_values))
    if not unique_threshold_schemes:
        threshold_validation_scheme = "unknown"
    elif len(unique_threshold_schemes) == 1:
        threshold_validation_scheme = unique_threshold_schemes[0]
    else:
        threshold_validation_scheme = "mixed"
    data_window_summary["threshold_validation_scheme"] = threshold_validation_scheme
    data_window_summary["threshold_validation_embargo_rows"] = (
        max(threshold_embargo_values) if threshold_embargo_values else 0
    )
    threshold_leakage_safe = bool(unique_threshold_schemes) and all(
        str(value).startswith("nested_purged") for value in unique_threshold_schemes
    )
    recent_holdout_leakage_safe = bool(recent_holdout_scheme_values) and all(
        str(value) == "recent_holdout_purged" for value in recent_holdout_scheme_values
    )
    data_window_summary["validation_leakage_safe"] = bool(threshold_leakage_safe and recent_holdout_leakage_safe)

    best_regression_model, regression_selection_basis = select_regression_forecast_model(
        regression_leaderboard,
        regression_backtest=regression_backtest,
        recent_holdout_report=recent_holdout_report,
        horizon=horizon,
        latest_features=prediction_frame.iloc[-1],
        prediction_feedback_summary=prediction_feedback_summary,
        prediction_feedback_state_summary=prediction_feedback_state_summary,
    )
    best_classification_model, classification_selection_basis = select_classification_forecast_model(
        classification_leaderboard,
        classification_backtest,
        recent_holdout_report=recent_holdout_report,
        horizon=horizon,
        latest_features=prediction_frame.iloc[-1],
        prediction_feedback_summary=prediction_feedback_summary,
        prediction_feedback_state_summary=prediction_feedback_state_summary,
    )
    if not regime_leaderboard.empty:
        best_regime_model, regime_selection_basis = select_state_forecast_model(
            regime_leaderboard,
            regime_backtest,
            task_key="regime",
            minimum_balanced_accuracy=0.40 if horizon <= 7 else 0.38,
            latest_features=prediction_frame.iloc[-1],
        )
    else:
        best_regime_model, regime_selection_basis = "", "fallback_heuristic"
    if not reversal_leaderboard.empty:
        best_reversal_model, reversal_selection_basis = select_state_forecast_model(
            reversal_leaderboard,
            reversal_backtest,
            task_key="reversal",
            minimum_balanced_accuracy=0.34,
            latest_features=prediction_frame.iloc[-1],
        )
    else:
        best_reversal_model, reversal_selection_basis = "", "fallback_heuristic"
    classification_threshold = DEFAULT_CLASSIFICATION_SIGNAL_THRESHOLD
    if not classification_backtest.empty:
        selected_backtest_row = classification_backtest.loc[
            classification_backtest["model"] == best_classification_model
        ]
        if not selected_backtest_row.empty:
            threshold_value = pd.to_numeric(selected_backtest_row.iloc[0].get("signal_threshold"), errors="coerce")
            if pd.notna(threshold_value):
                classification_threshold = float(threshold_value)
    classification_threshold, classification_suppressed = strengthen_classification_threshold(
        base_threshold=classification_threshold,
        classification_backtest=classification_backtest,
        recent_holdout_report=recent_holdout_report,
        model_name=best_classification_model,
        horizon=horizon,
    )
    if "cv_weak" in classification_selection_basis:
        classification_suppressed = True
        classification_threshold = max(classification_threshold, 0.65 if horizon <= 7 else 0.60)
    if classification_suppressed:
        classification_selection_basis = f"{classification_selection_basis}+weak_signal_suppressed"
    if build_explainability:
        log_progress(verbose, f"{label}building explainability reports...")
        if best_regression_model == TRIMMED_REGRESSION_ENSEMBLE_MODEL:
            regression_explainability = empty_explainability_artifacts(
                best_regression_model,
                "Trimmed equal-weight regression baseline selected; model-specific explainability was skipped.",
            )
        else:
            try:
                regression_explainability = build_explainability_artifacts(
                    dataset=training_dataset,
                    prediction_frame=prediction_frame,
                    feature_columns=feature_columns,
                    model_name=best_regression_model,
                    horizon=horizon,
                    cv_test_size=cv_test_size,
                    task="regression",
                )
            except Exception as exc:
                regression_explainability = empty_explainability_artifacts(
                    best_regression_model,
                    f"Regression explainability failed and was skipped: {exc}",
                )
        if best_classification_model == TRIMMED_CLASSIFICATION_ENSEMBLE_MODEL:
            classification_explainability = empty_explainability_artifacts(
                best_classification_model,
                "Trimmed equal-weight direction baseline selected; model-specific explainability was skipped.",
            )
        else:
            try:
                classification_explainability = build_explainability_artifacts(
                    dataset=training_dataset,
                    prediction_frame=prediction_frame,
                    feature_columns=feature_columns,
                    model_name=best_classification_model,
                    horizon=horizon,
                    cv_test_size=cv_test_size,
                    task="classification",
                )
            except Exception as exc:
                classification_explainability = empty_explainability_artifacts(
                    best_classification_model,
                    f"Classification explainability failed and was skipped: {exc}",
                )
    else:
        log_progress(verbose, f"{label}skipping explainability reports...")
        regression_explainability = empty_explainability_artifacts(
            best_regression_model,
            "Explainability disabled for this run.",
        )
        classification_explainability = empty_explainability_artifacts(
            best_classification_model,
            "Explainability disabled for this run.",
        )

    log_progress(verbose, f"{label}fitting final forecast models...")
    regression_forecast = forecast_next_step(
        training_dataset=training_dataset,
        prediction_frame=prediction_frame,
        feature_columns=feature_columns,
        interval=interval,
        horizon=horizon,
        model_name=best_regression_model,
        selection_basis=regression_selection_basis,
        price_reference=price_reference,
        sample_weight=sample_weights,
        regression_oof_predictions=regression_oof_predictions,
        regression_leaderboard=regression_leaderboard,
        regression_backtest=regression_backtest,
        recent_holdout_report=recent_holdout_report,
        prediction_feedback_summary=prediction_feedback_summary,
    )
    classification_forecast = forecast_direction(
        training_dataset=training_dataset,
        prediction_frame=prediction_frame,
        feature_columns=feature_columns,
        interval=interval,
        horizon=horizon,
        model_name=best_classification_model,
        selection_basis=classification_selection_basis,
        signal_threshold=classification_threshold,
        price_reference=price_reference,
        sample_weight=sample_weights,
        classification_leaderboard=classification_leaderboard,
        classification_backtest=classification_backtest,
        recent_holdout_report=recent_holdout_report,
    )
    if best_regime_model:
        regime_forecast = forecast_regime_state(
            training_dataset=training_dataset,
            prediction_frame=prediction_frame,
            feature_columns=feature_columns,
            interval=interval,
            horizon=horizon,
            model_name=best_regime_model,
            selection_basis=regime_selection_basis,
            price_reference=price_reference,
            sample_weight=sample_weights,
        )
    else:
        regime_forecast = heuristic_regime_forecast(
            prediction_frame=prediction_frame,
            interval=interval,
            horizon=horizon,
            price_reference=price_reference,
        )
    if best_reversal_model:
        reversal_forecast = forecast_reversal_state(
            training_dataset=training_dataset,
            prediction_frame=prediction_frame,
            feature_columns=feature_columns,
            interval=interval,
            horizon=horizon,
            model_name=best_reversal_model,
            selection_basis=reversal_selection_basis,
            price_reference=price_reference,
            sample_weight=sample_weights,
        )
    else:
        reversal_forecast = heuristic_reversal_forecast(
            prediction_frame=prediction_frame,
            interval=interval,
            horizon=horizon,
            price_reference=price_reference,
        )
    hybrid_forecast = build_hybrid_forecast(
        training_dataset=training_dataset,
        prediction_frame=prediction_frame,
        interval=interval,
        horizon=horizon,
        price_reference=price_reference,
        regression_forecast=regression_forecast,
        classification_forecast=classification_forecast,
        regime_forecast=regime_forecast,
        reversal_forecast=reversal_forecast,
        classification_backtest=classification_backtest,
        recent_holdout_report=recent_holdout_report,
    )

    return HorizonArtifacts(
        horizon_steps=horizon,
        regression_forecast=regression_forecast,
        classification_forecast=classification_forecast,
        regime_forecast=regime_forecast,
        reversal_forecast=reversal_forecast,
        hybrid_forecast=hybrid_forecast,
        regression_leaderboard=regression_leaderboard,
        classification_leaderboard=classification_leaderboard,
        regime_leaderboard=regime_leaderboard,
        reversal_leaderboard=reversal_leaderboard,
        regression_backtest=regression_backtest,
        classification_backtest=classification_backtest,
        regime_backtest=regime_backtest,
        reversal_backtest=reversal_backtest,
        benchmark_backtest=benchmark_backtest,
        regression_oof_predictions=regression_oof_predictions,
        classification_oof_predictions=classification_oof_predictions,
        regime_oof_predictions=regime_oof_predictions,
        reversal_oof_predictions=reversal_oof_predictions,
        regression_equity_curves=regression_equity_curves,
        classification_equity_curves=classification_equity_curves,
        regime_equity_curves=regime_equity_curves,
        reversal_equity_curves=reversal_equity_curves,
        benchmark_equity_curves=benchmark_equity_curves,
        data_window_summary=data_window_summary,
        feature_coverage_summary=feature_coverage_summary,
        recent_holdout_report=recent_holdout_report,
        feature_snapshot=prediction_frame[feature_columns].tail(1).T.rename(
            columns={prediction_frame.index[-1]: "latest_value"}
        ),
        regression_explainability=regression_explainability,
        classification_explainability=classification_explainability,
    )


def save_horizon_artifacts(artifacts: HorizonArtifacts, output_dir: str | Path) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    artifacts.regression_leaderboard.to_csv(destination / "regression_leaderboard.csv", index=False)
    artifacts.classification_leaderboard.to_csv(destination / "classification_leaderboard.csv", index=False)
    artifacts.regime_leaderboard.to_csv(destination / "regime_leaderboard.csv", index=False)
    artifacts.reversal_leaderboard.to_csv(destination / "reversal_leaderboard.csv", index=False)
    artifacts.regression_backtest.to_csv(destination / "regression_backtest.csv", index=False)
    artifacts.classification_backtest.to_csv(destination / "classification_backtest.csv", index=False)
    artifacts.regime_backtest.to_csv(destination / "regime_backtest.csv", index=False)
    artifacts.reversal_backtest.to_csv(destination / "reversal_backtest.csv", index=False)
    artifacts.benchmark_backtest.to_csv(destination / "benchmark_backtest.csv", index=False)
    artifacts.regression_oof_predictions.to_csv(destination / "regression_oof_predictions.csv")
    artifacts.classification_oof_predictions.to_csv(destination / "classification_oof_predictions.csv")
    artifacts.regime_oof_predictions.to_csv(destination / "regime_oof_predictions.csv")
    artifacts.reversal_oof_predictions.to_csv(destination / "reversal_oof_predictions.csv")
    artifacts.regression_equity_curves.to_csv(destination / "regression_equity_curves.csv")
    artifacts.classification_equity_curves.to_csv(destination / "classification_equity_curves.csv")
    artifacts.regime_equity_curves.to_csv(destination / "regime_equity_curves.csv")
    artifacts.reversal_equity_curves.to_csv(destination / "reversal_equity_curves.csv")
    artifacts.benchmark_equity_curves.to_csv(destination / "benchmark_equity_curves.csv")
    artifacts.data_window_summary.to_csv(destination / "data_window_summary.csv", index=False)
    artifacts.feature_coverage_summary.to_csv(destination / "feature_coverage_summary.csv", index=False)
    artifacts.recent_holdout_report.to_csv(destination / "recent_holdout_report.csv", index=False)
    artifacts.feature_snapshot.to_csv(destination / "latest_feature_snapshot.csv")
    artifacts.regression_explainability.native_feature_importance.to_csv(
        destination / "regression_native_feature_importance.csv",
        index=False,
    )
    artifacts.regression_explainability.permutation_importance.to_csv(
        destination / "regression_permutation_importance.csv",
        index=False,
    )
    artifacts.regression_explainability.shap_global_importance.to_csv(
        destination / "regression_shap_global_importance.csv",
        index=False,
    )
    artifacts.regression_explainability.shap_latest_contributions.to_csv(
        destination / "regression_shap_latest_contributions.csv",
        index=False,
    )
    artifacts.regression_explainability.notes.to_csv(
        destination / "regression_explainability_notes.csv",
        index=False,
    )
    artifacts.classification_explainability.native_feature_importance.to_csv(
        destination / "classification_native_feature_importance.csv",
        index=False,
    )
    artifacts.classification_explainability.permutation_importance.to_csv(
        destination / "classification_permutation_importance.csv",
        index=False,
    )
    artifacts.classification_explainability.shap_global_importance.to_csv(
        destination / "classification_shap_global_importance.csv",
        index=False,
    )
    artifacts.classification_explainability.shap_latest_contributions.to_csv(
        destination / "classification_shap_latest_contributions.csv",
        index=False,
    )
    artifacts.classification_explainability.notes.to_csv(
        destination / "classification_explainability_notes.csv",
        index=False,
    )

    with (destination / "regression_forecast.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts.regression_forecast), handle, ensure_ascii=False, indent=2)
    with (destination / "classification_forecast.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts.classification_forecast), handle, ensure_ascii=False, indent=2)
    with (destination / "regime_forecast.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts.regime_forecast), handle, ensure_ascii=False, indent=2)
    with (destination / "reversal_forecast.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts.reversal_forecast), handle, ensure_ascii=False, indent=2)
    with (destination / "hybrid_forecast.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(artifacts.hybrid_forecast), handle, ensure_ascii=False, indent=2)


def extract_model_row(frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
    if frame.empty or "model" not in frame.columns:
        return {}
    match = frame.loc[frame["model"] == model_name]
    return match.iloc[0].to_dict() if not match.empty else {}


def build_latest_forecast_summary(artifacts: PipelineArtifacts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_artifacts in sorted(artifacts.horizons.items()):
        data_window = (
            horizon_artifacts.data_window_summary.iloc[0].to_dict()
            if not horizon_artifacts.data_window_summary.empty
            else {}
        )
        regression_backtest = extract_model_row(
            horizon_artifacts.regression_backtest,
            horizon_artifacts.regression_forecast.model_name,
        )
        classification_backtest = extract_model_row(
            horizon_artifacts.classification_backtest,
            horizon_artifacts.classification_forecast.model_name,
        )
        regime_backtest = extract_model_row(
            horizon_artifacts.regime_backtest,
            horizon_artifacts.regime_forecast.model_name,
        )
        reversal_backtest = extract_model_row(
            horizon_artifacts.reversal_backtest,
            horizon_artifacts.reversal_forecast.model_name,
        )
        rows.append(
            {
                "horizon_steps": int(horizon),
                "forecast_input_timestamp": data_window.get("latest_prediction_input_timestamp", ""),
                "forecast_target_timestamp": horizon_artifacts.regression_forecast.prediction_timestamp,
                "regression_model_name": horizon_artifacts.regression_forecast.model_name,
                "regression_selection_basis": horizon_artifacts.regression_forecast.selection_basis,
                "reference_price": horizon_artifacts.regression_forecast.last_close,
                "reference_price_source": horizon_artifacts.regression_forecast.reference_price_source,
                "reference_price_timestamp": horizon_artifacts.regression_forecast.reference_price_timestamp,
                "model_input_close": horizon_artifacts.regression_forecast.model_input_close,
                "regression_predicted_return": horizon_artifacts.regression_forecast.predicted_return,
                "regression_predicted_close": horizon_artifacts.regression_forecast.predicted_close,
                "regression_lower_return_10": horizon_artifacts.regression_forecast.lower_return_10,
                "regression_lower_close_10": horizon_artifacts.regression_forecast.lower_close_10,
                "regression_upper_return_90": horizon_artifacts.regression_forecast.upper_return_90,
                "regression_upper_close_90": horizon_artifacts.regression_forecast.upper_close_90,
                "classification_model_name": horizon_artifacts.classification_forecast.model_name,
                "classification_selection_basis": horizon_artifacts.classification_forecast.selection_basis,
                "classification_predicted_direction": horizon_artifacts.classification_forecast.predicted_direction,
                "classification_signal_threshold": horizon_artifacts.classification_forecast.signal_threshold,
                "classification_probability_up": horizon_artifacts.classification_forecast.probability_up,
                "classification_probability_down": horizon_artifacts.classification_forecast.probability_down,
                "classification_confidence": horizon_artifacts.classification_forecast.confidence,
                "regime_model_name": horizon_artifacts.regime_forecast.model_name,
                "regime_selection_basis": horizon_artifacts.regime_forecast.selection_basis,
                "regime_predicted_state": horizon_artifacts.regime_forecast.predicted_regime,
                "regime_probability_uptrend": horizon_artifacts.regime_forecast.probability_uptrend,
                "regime_probability_sideways": horizon_artifacts.regime_forecast.probability_sideways,
                "regime_probability_downtrend": horizon_artifacts.regime_forecast.probability_downtrend,
                "reversal_model_name": horizon_artifacts.reversal_forecast.model_name,
                "reversal_selection_basis": horizon_artifacts.reversal_forecast.selection_basis,
                "reversal_predicted_signal": horizon_artifacts.reversal_forecast.predicted_reversal,
                "reversal_probability_bottom": horizon_artifacts.reversal_forecast.probability_bottom_reversal,
                "reversal_probability_top": horizon_artifacts.reversal_forecast.probability_top_reversal,
                "hybrid_model_name": horizon_artifacts.hybrid_forecast.model_name,
                "hybrid_predicted_direction": horizon_artifacts.hybrid_forecast.predicted_direction,
                "hybrid_bias_direction": horizon_artifacts.hybrid_forecast.bias_direction,
                "hybrid_predicted_market_state": horizon_artifacts.hybrid_forecast.predicted_market_state,
                "hybrid_predicted_reversal_signal": horizon_artifacts.hybrid_forecast.predicted_reversal_signal,
                "hybrid_score": horizon_artifacts.hybrid_forecast.hybrid_score,
                "hybrid_confidence": horizon_artifacts.hybrid_forecast.confidence,
                "hybrid_signal_tier": horizon_artifacts.hybrid_forecast.signal_tier,
                "hybrid_high_confidence_signal": horizon_artifacts.hybrid_forecast.high_confidence_signal,
                "hybrid_signal_reason": horizon_artifacts.hybrid_forecast.signal_reason,
                "hybrid_volatility_scale": horizon_artifacts.hybrid_forecast.volatility_scale,
                "hybrid_scenario_spread": horizon_artifacts.hybrid_forecast.scenario_spread,
                "active_predicted_direction": horizon_artifacts.hybrid_forecast.active_predicted_direction,
                "active_confidence": horizon_artifacts.hybrid_forecast.active_confidence,
                "active_predicted_return": horizon_artifacts.hybrid_forecast.active_predicted_return,
                "active_predicted_close": horizon_artifacts.hybrid_forecast.active_predicted_close,
                "active_bear_predicted_return": horizon_artifacts.hybrid_forecast.active_bear_predicted_return,
                "active_bear_predicted_close": horizon_artifacts.hybrid_forecast.active_bear_predicted_close,
                "active_bull_predicted_return": horizon_artifacts.hybrid_forecast.active_bull_predicted_return,
                "active_bull_predicted_close": horizon_artifacts.hybrid_forecast.active_bull_predicted_close,
                "macro_readiness_status": horizon_artifacts.hybrid_forecast.macro_readiness_status,
                "macro_risk_regime": horizon_artifacts.hybrid_forecast.macro_risk_regime,
                "macro_directional_bias": horizon_artifacts.hybrid_forecast.macro_directional_bias,
                "macro_selection_tag": horizon_artifacts.hybrid_forecast.macro_selection_tag,
                "macro_risk_on_score": horizon_artifacts.hybrid_forecast.macro_risk_on_score,
                "macro_risk_off_score": horizon_artifacts.hybrid_forecast.macro_risk_off_score,
                "macro_trend_bias_score": horizon_artifacts.hybrid_forecast.macro_trend_bias_score,
                "macro_volatility_stress": horizon_artifacts.hybrid_forecast.macro_volatility_stress,
                "macro_used_feature_count": horizon_artifacts.hybrid_forecast.macro_used_feature_count,
                "macro_missing_feature_count": horizon_artifacts.hybrid_forecast.macro_missing_feature_count,
                "macro_used_features": horizon_artifacts.hybrid_forecast.macro_used_features,
                "macro_missing_features": horizon_artifacts.hybrid_forecast.macro_missing_features,
                "hybrid_adjusted_predicted_return": horizon_artifacts.hybrid_forecast.adjusted_predicted_return,
                "hybrid_adjusted_predicted_close": horizon_artifacts.hybrid_forecast.adjusted_predicted_close,
                "hybrid_bear_predicted_return": horizon_artifacts.hybrid_forecast.bear_predicted_return,
                "hybrid_bear_predicted_close": horizon_artifacts.hybrid_forecast.bear_predicted_close,
                "hybrid_bull_predicted_return": horizon_artifacts.hybrid_forecast.bull_predicted_return,
                "hybrid_bull_predicted_close": horizon_artifacts.hybrid_forecast.bull_predicted_close,
                "regression_backtest_total_return": regression_backtest.get("total_return"),
                "regression_backtest_sharpe": regression_backtest.get("sharpe"),
                "regression_macro_bucket_label": regression_backtest.get("macro_bucket_label"),
                "regression_macro_bucket_total_return": regression_backtest.get("macro_bucket_total_return"),
                "regression_macro_bucket_sharpe": regression_backtest.get("macro_bucket_sharpe"),
                "regression_macro_bucket_signal_count": regression_backtest.get("macro_bucket_signal_count"),
                "classification_backtest_total_return": classification_backtest.get("total_return"),
                "classification_backtest_sharpe": classification_backtest.get("sharpe"),
                "classification_macro_bucket_label": classification_backtest.get("macro_bucket_label"),
                "classification_macro_bucket_total_return": classification_backtest.get("macro_bucket_total_return"),
                "classification_macro_bucket_sharpe": classification_backtest.get("macro_bucket_sharpe"),
                "classification_macro_bucket_signal_count": classification_backtest.get("macro_bucket_signal_count"),
                "regime_backtest_total_return": regime_backtest.get("total_return"),
                "regime_backtest_sharpe": regime_backtest.get("sharpe"),
                "regime_macro_bucket_label": regime_backtest.get("macro_bucket_label"),
                "regime_macro_bucket_total_return": regime_backtest.get("macro_bucket_total_return"),
                "regime_macro_bucket_sharpe": regime_backtest.get("macro_bucket_sharpe"),
                "regime_macro_bucket_signal_count": regime_backtest.get("macro_bucket_signal_count"),
                "reversal_backtest_total_return": reversal_backtest.get("total_return"),
                "reversal_backtest_sharpe": reversal_backtest.get("sharpe"),
                "reversal_macro_bucket_label": reversal_backtest.get("macro_bucket_label"),
                "reversal_macro_bucket_total_return": reversal_backtest.get("macro_bucket_total_return"),
                "reversal_macro_bucket_sharpe": reversal_backtest.get("macro_bucket_sharpe"),
                "reversal_macro_bucket_signal_count": reversal_backtest.get("macro_bucket_signal_count"),
                "regression_threshold_validation_scheme": regression_backtest.get("threshold_validation_scheme"),
                "regression_threshold_selection_rows": regression_backtest.get("threshold_selection_rows"),
                "regression_threshold_evaluation_rows": regression_backtest.get("threshold_evaluation_rows"),
                "classification_threshold_validation_scheme": classification_backtest.get("threshold_validation_scheme"),
                "classification_threshold_selection_rows": classification_backtest.get("threshold_selection_rows"),
                "classification_threshold_evaluation_rows": classification_backtest.get("threshold_evaluation_rows"),
                "validation_leakage_safe": data_window.get("validation_leakage_safe"),
                "threshold_validation_scheme": data_window.get("threshold_validation_scheme"),
                "threshold_validation_embargo_rows": data_window.get("threshold_validation_embargo_rows"),
                "recent_holdout_rows": data_window.get("recent_holdout_rows"),
                "recent_holdout_embargo_rows": data_window.get("recent_holdout_embargo_rows"),
                "raw_rows": data_window.get("raw_rows"),
                "training_rows": data_window.get("training_rows"),
                "feature_count": data_window.get("feature_count"),
            }
        )
    return pd.DataFrame(rows)


def build_compact_dashboard(artifacts: PipelineArtifacts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for horizon, horizon_artifacts in sorted(artifacts.horizons.items()):
        data_window = (
            horizon_artifacts.data_window_summary.iloc[0].to_dict()
            if not horizon_artifacts.data_window_summary.empty
            else {}
        )
        benchmark_best = (
            horizon_artifacts.benchmark_backtest.iloc[0].to_dict()
            if not horizon_artifacts.benchmark_backtest.empty
            else {}
        )
        regression_backtest = extract_model_row(
            horizon_artifacts.regression_backtest,
            horizon_artifacts.regression_forecast.model_name,
        )
        classification_backtest = extract_model_row(
            horizon_artifacts.classification_backtest,
            horizon_artifacts.classification_forecast.model_name,
        )
        regime_backtest = extract_model_row(
            horizon_artifacts.regime_backtest,
            horizon_artifacts.regime_forecast.model_name,
        )
        reversal_backtest = extract_model_row(
            horizon_artifacts.reversal_backtest,
            horizon_artifacts.reversal_forecast.model_name,
        )

        regression_perm = top_importance_records(
            horizon_artifacts.regression_explainability.permutation_importance,
            "importance_mean",
            limit=3,
        )
        classification_perm = top_importance_records(
            horizon_artifacts.classification_explainability.permutation_importance,
            "importance_mean",
            limit=3,
        )
        rows.append(
            {
                "horizon_steps": int(horizon),
                "forecast_target_timestamp": horizon_artifacts.regression_forecast.prediction_timestamp,
                "raw_rows": data_window.get("raw_rows"),
                "training_rows": data_window.get("training_rows"),
                "candidate_feature_count": data_window.get("candidate_feature_count"),
                "feature_count": data_window.get("feature_count"),
                "dropped_feature_count": data_window.get("dropped_feature_count"),
                "regression_model_name": horizon_artifacts.regression_forecast.model_name,
                "regression_selection_basis": horizon_artifacts.regression_forecast.selection_basis,
                "reference_price": horizon_artifacts.regression_forecast.last_close,
                "reference_price_source": horizon_artifacts.regression_forecast.reference_price_source,
                "reference_price_timestamp": horizon_artifacts.regression_forecast.reference_price_timestamp,
                "model_input_close": horizon_artifacts.regression_forecast.model_input_close,
                "regression_predicted_return": horizon_artifacts.regression_forecast.predicted_return,
                "regression_predicted_close": horizon_artifacts.regression_forecast.predicted_close,
                "regression_lower_return_10": horizon_artifacts.regression_forecast.lower_return_10,
                "regression_lower_close_10": horizon_artifacts.regression_forecast.lower_close_10,
                "regression_upper_return_90": horizon_artifacts.regression_forecast.upper_return_90,
                "regression_upper_close_90": horizon_artifacts.regression_forecast.upper_close_90,
                "regression_backtest_total_return": regression_backtest.get("total_return"),
                "regression_backtest_sharpe": regression_backtest.get("sharpe"),
                "regression_macro_bucket_label": regression_backtest.get("macro_bucket_label"),
                "regression_macro_bucket_total_return": regression_backtest.get("macro_bucket_total_return"),
                "regression_macro_bucket_sharpe": regression_backtest.get("macro_bucket_sharpe"),
                "regression_macro_bucket_signal_count": regression_backtest.get("macro_bucket_signal_count"),
                "regression_threshold_validation_scheme": regression_backtest.get("threshold_validation_scheme"),
                "regression_threshold_selection_rows": regression_backtest.get("threshold_selection_rows"),
                "regression_threshold_evaluation_rows": regression_backtest.get("threshold_evaluation_rows"),
                "classification_model_name": horizon_artifacts.classification_forecast.model_name,
                "classification_selection_basis": horizon_artifacts.classification_forecast.selection_basis,
                "classification_predicted_direction": horizon_artifacts.classification_forecast.predicted_direction,
                "classification_signal_threshold": horizon_artifacts.classification_forecast.signal_threshold,
                "classification_probability_up": horizon_artifacts.classification_forecast.probability_up,
                "classification_probability_down": horizon_artifacts.classification_forecast.probability_down,
                "classification_backtest_total_return": classification_backtest.get("total_return"),
                "classification_backtest_sharpe": classification_backtest.get("sharpe"),
                "classification_macro_bucket_label": classification_backtest.get("macro_bucket_label"),
                "classification_macro_bucket_total_return": classification_backtest.get("macro_bucket_total_return"),
                "classification_macro_bucket_sharpe": classification_backtest.get("macro_bucket_sharpe"),
                "classification_macro_bucket_signal_count": classification_backtest.get("macro_bucket_signal_count"),
                "classification_threshold_validation_scheme": classification_backtest.get("threshold_validation_scheme"),
                "classification_threshold_selection_rows": classification_backtest.get("threshold_selection_rows"),
                "classification_threshold_evaluation_rows": classification_backtest.get("threshold_evaluation_rows"),
                "regime_model_name": horizon_artifacts.regime_forecast.model_name,
                "regime_predicted_state": horizon_artifacts.regime_forecast.predicted_regime,
                "regime_probability_uptrend": horizon_artifacts.regime_forecast.probability_uptrend,
                "regime_probability_sideways": horizon_artifacts.regime_forecast.probability_sideways,
                "regime_backtest_total_return": regime_backtest.get("total_return"),
                "regime_backtest_sharpe": regime_backtest.get("sharpe"),
                "regime_macro_bucket_label": regime_backtest.get("macro_bucket_label"),
                "regime_macro_bucket_total_return": regime_backtest.get("macro_bucket_total_return"),
                "regime_macro_bucket_sharpe": regime_backtest.get("macro_bucket_sharpe"),
                "regime_macro_bucket_signal_count": regime_backtest.get("macro_bucket_signal_count"),
                "reversal_model_name": horizon_artifacts.reversal_forecast.model_name,
                "reversal_predicted_signal": horizon_artifacts.reversal_forecast.predicted_reversal,
                "reversal_probability_bottom": horizon_artifacts.reversal_forecast.probability_bottom_reversal,
                "reversal_probability_top": horizon_artifacts.reversal_forecast.probability_top_reversal,
                "reversal_backtest_total_return": reversal_backtest.get("total_return"),
                "reversal_backtest_sharpe": reversal_backtest.get("sharpe"),
                "reversal_macro_bucket_label": reversal_backtest.get("macro_bucket_label"),
                "reversal_macro_bucket_total_return": reversal_backtest.get("macro_bucket_total_return"),
                "reversal_macro_bucket_sharpe": reversal_backtest.get("macro_bucket_sharpe"),
                "reversal_macro_bucket_signal_count": reversal_backtest.get("macro_bucket_signal_count"),
                "hybrid_model_name": horizon_artifacts.hybrid_forecast.model_name,
                "hybrid_predicted_market_state": horizon_artifacts.hybrid_forecast.predicted_market_state,
                "hybrid_predicted_reversal_signal": horizon_artifacts.hybrid_forecast.predicted_reversal_signal,
                "hybrid_predicted_direction": horizon_artifacts.hybrid_forecast.predicted_direction,
                "hybrid_bias_direction": horizon_artifacts.hybrid_forecast.bias_direction,
                "hybrid_score": horizon_artifacts.hybrid_forecast.hybrid_score,
                "hybrid_confidence": horizon_artifacts.hybrid_forecast.confidence,
                "hybrid_signal_tier": horizon_artifacts.hybrid_forecast.signal_tier,
                "hybrid_high_confidence_signal": horizon_artifacts.hybrid_forecast.high_confidence_signal,
                "hybrid_signal_reason": horizon_artifacts.hybrid_forecast.signal_reason,
                "hybrid_volatility_scale": horizon_artifacts.hybrid_forecast.volatility_scale,
                "hybrid_scenario_spread": horizon_artifacts.hybrid_forecast.scenario_spread,
                "active_predicted_direction": horizon_artifacts.hybrid_forecast.active_predicted_direction,
                "active_confidence": horizon_artifacts.hybrid_forecast.active_confidence,
                "active_predicted_return": horizon_artifacts.hybrid_forecast.active_predicted_return,
                "active_predicted_close": horizon_artifacts.hybrid_forecast.active_predicted_close,
                "active_bear_predicted_return": horizon_artifacts.hybrid_forecast.active_bear_predicted_return,
                "active_bear_predicted_close": horizon_artifacts.hybrid_forecast.active_bear_predicted_close,
                "active_bull_predicted_return": horizon_artifacts.hybrid_forecast.active_bull_predicted_return,
                "active_bull_predicted_close": horizon_artifacts.hybrid_forecast.active_bull_predicted_close,
                "macro_readiness_status": horizon_artifacts.hybrid_forecast.macro_readiness_status,
                "macro_risk_regime": horizon_artifacts.hybrid_forecast.macro_risk_regime,
                "macro_directional_bias": horizon_artifacts.hybrid_forecast.macro_directional_bias,
                "macro_selection_tag": horizon_artifacts.hybrid_forecast.macro_selection_tag,
                "macro_risk_on_score": horizon_artifacts.hybrid_forecast.macro_risk_on_score,
                "macro_risk_off_score": horizon_artifacts.hybrid_forecast.macro_risk_off_score,
                "macro_trend_bias_score": horizon_artifacts.hybrid_forecast.macro_trend_bias_score,
                "macro_volatility_stress": horizon_artifacts.hybrid_forecast.macro_volatility_stress,
                "macro_used_feature_count": horizon_artifacts.hybrid_forecast.macro_used_feature_count,
                "macro_missing_feature_count": horizon_artifacts.hybrid_forecast.macro_missing_feature_count,
                "macro_used_features": horizon_artifacts.hybrid_forecast.macro_used_features,
                "macro_missing_features": horizon_artifacts.hybrid_forecast.macro_missing_features,
                "hybrid_adjusted_predicted_return": horizon_artifacts.hybrid_forecast.adjusted_predicted_return,
                "hybrid_adjusted_predicted_close": horizon_artifacts.hybrid_forecast.adjusted_predicted_close,
                "hybrid_bear_predicted_return": horizon_artifacts.hybrid_forecast.bear_predicted_return,
                "hybrid_bear_predicted_close": horizon_artifacts.hybrid_forecast.bear_predicted_close,
                "hybrid_bull_predicted_return": horizon_artifacts.hybrid_forecast.bull_predicted_return,
                "hybrid_bull_predicted_close": horizon_artifacts.hybrid_forecast.bull_predicted_close,
                "best_benchmark_model": benchmark_best.get("model"),
                "best_benchmark_total_return": benchmark_best.get("total_return"),
                "best_benchmark_sharpe": benchmark_best.get("sharpe"),
                "validation_leakage_safe": data_window.get("validation_leakage_safe"),
                "threshold_validation_scheme": data_window.get("threshold_validation_scheme"),
                "threshold_validation_embargo_rows": data_window.get("threshold_validation_embargo_rows"),
                "recent_holdout_rows": data_window.get("recent_holdout_rows"),
                "recent_holdout_embargo_rows": data_window.get("recent_holdout_embargo_rows"),
                "regression_top_features": ", ".join(row["feature"] for row in regression_perm),
                "classification_top_features": ", ".join(row["feature"] for row in classification_perm),
            }
        )
    return pd.DataFrame(rows)


def build_integrated_report(artifacts: PipelineArtifacts) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    dashboard = build_compact_dashboard(artifacts)
    if not dashboard.empty:
        for row in dashboard.to_dict(orient="records"):
            rows.append({"report_group": "forecast_summary", **row})

    external_summary = build_external_feature_summary(artifacts.raw_data)
    if not external_summary.empty:
        for row in external_summary.to_dict(orient="records"):
            rows.append({"report_group": "external_feature", "horizon_steps": "", **row})

    report_tables = [
        ("regression_backtest", lambda item: item.regression_backtest),
        ("classification_backtest", lambda item: item.classification_backtest),
        ("regime_backtest", lambda item: item.regime_backtest),
        ("reversal_backtest", lambda item: item.reversal_backtest),
        ("benchmark_backtest", lambda item: item.benchmark_backtest),
        ("recent_holdout", lambda item: item.recent_holdout_report),
    ]
    for horizon, horizon_artifacts in sorted(artifacts.horizons.items()):
        for report_group, table_getter in report_tables:
            frame = table_getter(horizon_artifacts)
            if frame.empty:
                continue
            for row in frame.to_dict(orient="records"):
                rows.append({"report_group": report_group, "horizon_steps": int(horizon), **row})

    return pd.DataFrame(rows)


def update_prediction_history(
    artifacts: PipelineArtifacts,
    prediction_history_csv: str | Path,
) -> pd.DataFrame:
    destination = Path(prediction_history_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    existing = load_prediction_history_csv(destination)
    latest = build_latest_forecast_summary(artifacts)
    if latest.empty:
        return existing

    run_timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    latest = latest.copy()
    latest.insert(0, "run_timestamp", run_timestamp)
    latest["forecast_input_timestamp"] = pd.to_datetime(latest["forecast_input_timestamp"], errors="coerce")
    latest["forecast_target_timestamp"] = pd.to_datetime(latest["forecast_target_timestamp"], errors="coerce")

    combined = pd.concat([existing, latest], ignore_index=True)
    combined = combined.sort_values(["forecast_input_timestamp", "horizon_steps", "run_timestamp"])
    combined = combined.drop_duplicates(
        subset=["forecast_input_timestamp", "horizon_steps"],
        keep="last",
    ).reset_index(drop=True)
    combined = refresh_prediction_history_with_realizations(combined, artifacts.raw_data)
    combined.to_csv(destination, index=False)
    feedback_summary = build_prediction_feedback_summary(combined)
    feedback_summary.to_csv(destination.parent / "prediction_feedback_summary.csv", index=False)
    feedback_state_summary = build_prediction_feedback_state_summary(combined)
    feedback_state_summary.to_csv(destination.parent / "prediction_feedback_state_summary.csv", index=False)
    build_prediction_feedback_state_winner_table(feedback_state_summary).to_csv(
        destination.parent / "prediction_feedback_state_winners.csv",
        index=False,
    )
    return combined


def save_artifacts(
    artifacts: PipelineArtifacts,
    output_dir: str | Path,
    prediction_history_csv: str | Path | None = None,
    compact_output: bool = False,
) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    if not compact_output:
        artifacts.raw_data.to_csv(destination / "downloaded_market_data.csv")
        for horizon, horizon_artifacts in artifacts.horizons.items():
            save_horizon_artifacts(horizon_artifacts, destination / f"horizon_{horizon}")
        build_external_feature_summary(artifacts.raw_data).to_csv(
            destination / "external_feature_summary.csv",
            index=False,
        )
    build_latest_forecast_summary(artifacts).to_csv(destination / "latest_forecast_summary.csv", index=False)
    build_compact_dashboard(artifacts).to_csv(destination / "compact_dashboard.csv", index=False)
    build_integrated_report(artifacts).to_csv(destination / "integrated_report.csv", index=False)
    if artifacts.failed_horizons:
        pd.DataFrame(artifacts.failed_horizons).to_csv(destination / "horizon_failures.csv", index=False)
    if prediction_history_csv is not None:
        update_prediction_history(artifacts, prediction_history_csv)
    with (destination / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summarize_artifacts(artifacts), handle, ensure_ascii=False, indent=2)


def run_suite(
    period: str = "max",
    interval: str = "1d",
    horizons: list[int] | None = None,
    cv_splits: int = 3,
    cv_test_size: int = 20,
    output_dir: str | Path = "eth_forecast_outputs",
    dataset_csv: str | Path = "market_data_cache.csv",
    master_data_csv: str | Path | None = None,
    prediction_history_csv: str | Path | None = None,
    external_feature_csvs: list[str] | None = None,
    drive_root: str | Path | None = None,
    use_dataset_cache: bool = True,
    use_prediction_history_features: bool = True,
    cache_lookback_rows: int = CACHE_UPDATE_LOOKBACK_ROWS,
    min_feature_coverage: float = DEFAULT_FEATURE_MIN_COVERAGE,
    use_realtime_reference_price: bool = True,
    auto_install_catboost: bool = True,
    build_explainability: bool | None = None,
    fast_mode: bool = False,
    compact_output: bool = False,
    save_outputs: bool = True,
    continue_on_horizon_error: bool = True,
    verbose: bool = True,
) -> PipelineArtifacts:
    selected_horizons = list(horizons or DEFAULT_HORIZONS)
    previous_fast_mode = runtime_fast_mode_enabled()
    set_runtime_options(fast_mode=fast_mode)
    effective_build_explainability = (
        bool(build_explainability)
        if build_explainability is not None
        else (not compact_output and not fast_mode)
    )
    try:
        optional_model_status = bootstrap_optional_model_dependencies(
            auto_install_catboost=auto_install_catboost and (not fast_mode),
            verbose=verbose,
        )
        log_progress(
            verbose,
            (
                "Optional model status: "
                f"catboost_available={optional_model_status.get('catboost_available')}, "
                f"catboost_installed_now={optional_model_status.get('catboost_installed_now')}, "
                f"catboost_error={optional_model_status.get('catboost_error') or 'none'}"
            ),
        )
        if fast_mode:
            log_progress(
                verbose,
                "Runtime profile: fast_mode=1, slow model families disabled, explainability skipped by default.",
            )
        market_data, output_dir_path, _, history_path, _ = load_or_prepare_market_data(
            period=period,
            interval=interval,
            output_dir=output_dir,
            dataset_csv=dataset_csv,
            prediction_history_csv=prediction_history_csv,
            external_feature_csvs=external_feature_csvs,
            drive_root=drive_root,
            use_dataset_cache=use_dataset_cache,
            cache_lookback_rows=cache_lookback_rows,
            master_data_csv=master_data_csv,
            verbose=verbose,
        )
        prediction_history = (
            load_prediction_history_csv(history_path)
            if history_path is not None
            else pd.DataFrame()
        )
        prediction_history = refresh_prediction_history_with_realizations(prediction_history, market_data)
        prediction_feedback_summary = build_prediction_feedback_summary(prediction_history)
        prediction_feedback_state_summary = build_prediction_feedback_state_summary(prediction_history)
        prediction_history_features = prediction_history if use_prediction_history_features else pd.DataFrame()
        shared_price_reference = build_shared_price_reference(
            market_data,
            prefer_realtime=use_realtime_reference_price,
        )
        horizon_results: dict[int, HorizonArtifacts] = {}
        failed_horizons: list[dict[str, Any]] = []

        for horizon in selected_horizons:
            log_progress(verbose, f"Starting horizon {horizon}d...")
            try:
                horizon_results[horizon] = run_horizon_pipeline(
                    market_data=market_data,
                    interval=interval,
                    horizon=horizon,
                    cv_splits=cv_splits,
                    cv_test_size=cv_test_size,
                    prediction_history=prediction_history_features,
                    prediction_feedback_summary=prediction_feedback_summary,
                    prediction_feedback_state_summary=prediction_feedback_state_summary,
                    min_feature_coverage=min_feature_coverage,
                    use_realtime_reference_price=use_realtime_reference_price,
                    shared_price_reference=shared_price_reference,
                    build_explainability=effective_build_explainability,
                    verbose=verbose,
                )
            except Exception as exc:
                failure = {
                    "horizon_steps": int(horizon),
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback_tail": traceback.format_exc(limit=8),
                }
                failed_horizons.append(failure)
                log_progress(
                    verbose,
                    f"[WARN] Horizon {horizon}d failed and was skipped: {type(exc).__name__}: {exc}",
                )
                if not continue_on_horizon_error:
                    raise

        if not horizon_results:
            failure_summary = "; ".join(
                f"{item.get('horizon_steps')}d={item.get('error_type')}: {item.get('error_message')}"
                for item in failed_horizons
            )
            raise RuntimeError(f"All requested horizons failed: {failure_summary or 'unknown error'}")
        artifacts = PipelineArtifacts(
            raw_data=market_data,
            horizons=horizon_results,
            failed_horizons=failed_horizons,
        )
        if save_outputs:
            save_artifacts(
                artifacts,
                output_dir_path,
                prediction_history_csv=history_path,
                compact_output=compact_output,
            )
        return artifacts
    finally:
        set_runtime_options(fast_mode=previous_fast_mode)


def _ignored_unknown_cli_arg(value: str) -> bool:
    token = str(value).strip()
    if token == "":
        return True
    lowered = token.lower()
    ignored_prefixes = (
        "--historymanager.",
        "--session.",
        "--ipkernelapp.",
        "--interactiveshellapp.",
        "--ip=",
        "--stdin=",
        "--control=",
        "--hb=",
        "--shell=",
        "--transport=",
        "--iopub=",
    )
    if token == "-f" or any(lowered.startswith(prefix) for prefix in ignored_prefixes):
        return True
    return lowered.endswith(".json") and "kernel" in lowered


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ETH-USD multi-horizon forecast suite with point forecast, direction, "
            "regime state, reversal state, and hybrid gating."
        )
    )
    parser.add_argument("--period", default="max", help="yfinance history window, for example 2y, 5y, max")
    parser.add_argument("--interval", default="1d", help="yfinance interval, for example 1d or 1h")
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=DEFAULT_HORIZONS,
        help="Forecast horizons to evaluate together, for example 7 30",
    )
    parser.add_argument("--cv-splits", type=int, default=3, help="Number of time-series validation folds")
    parser.add_argument("--cv-test-size", type=int, default=20, help="Number of rows per validation fold")
    parser.add_argument(
        "--dataset-csv",
        default="market_data_cache.csv",
        help="Local CSV cache for merged market data. First run creates it, later runs update it.",
    )
    parser.add_argument(
        "--master-data-csv",
        default="",
        help="Optional prebuilt master dataset CSV. When present, forecasting loads this instead of rebuilding market plus external features.",
    )
    parser.add_argument(
        "--prediction-history-csv",
        default="",
        help="CSV log of past forecasts. Defaults to <output-dir>/prediction_history.csv when left blank.",
    )
    parser.add_argument(
        "--drive-root",
        default="",
        help="Optional Google Drive root directory. When set, default cache/history/output paths are created under this folder.",
    )
    parser.add_argument(
        "--external-features-csv",
        nargs="*",
        default=[],
        help="Optional numeric CSV files with extra features such as funding, OI, on-chain, ETF flow, or exchange-flow series.",
    )
    parser.add_argument(
        "--cache-lookback-rows",
        type=int,
        default=CACHE_UPDATE_LOOKBACK_ROWS,
        help="How many recent rows to re-download when refreshing the dataset CSV cache.",
    )
    parser.add_argument(
        "--no-dataset-cache",
        action="store_true",
        help="Always download directly from Yahoo Finance instead of using the dataset CSV cache.",
    )
    parser.add_argument(
        "--no-prediction-history-features",
        action="store_true",
        help="Do not use prior saved forecast history as extra meta-features.",
    )
    parser.add_argument(
        "--no-realtime-reference-price",
        action="store_true",
        help="Use the latest daily feature close as the forecast reference price instead of attempting a live ETH price lookup.",
    )
    parser.add_argument(
        "--no-auto-install-catboost",
        action="store_true",
        help="Do not attempt to install CatBoost automatically when it is missing.",
    )
    parser.add_argument(
        "--no-explainability",
        action="store_true",
        help="Skip permutation importance and SHAP generation to reduce runtime.",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="Disable slow model families (MLP/CatBoost) and skip explainability by default.",
    )
    parser.add_argument(
        "--output-dir",
        default="eth_forecast_outputs",
        help="Where CSV and JSON outputs should be saved",
    )
    parser.add_argument(
        "--min-feature-coverage",
        type=float,
        default=DEFAULT_FEATURE_MIN_COVERAGE,
        help="Minimum non-null coverage ratio required for standard features before they are kept.",
    )
    parser.add_argument(
        "--compact-output",
        action="store_true",
        help="Save only compact_dashboard.csv, integrated_report.csv, and summary.json instead of the detailed per-horizon artifact set.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the whole suite immediately if any requested horizon fails.",
    )
    parser.add_argument("--no-save", action="store_true", help="Do not save CSV and JSON artifacts")
    parser.add_argument("--quiet", action="store_true", help="Reduce progress logging")
    parser.add_argument("--json", action="store_true", help="Print the summary as JSON")
    args, unknown = parser.parse_known_args(argv)
    unexpected = [token for token in unknown if not _ignored_unknown_cli_arg(token)]
    if unexpected:
        parser.error(f"unrecognized arguments: {' '.join(unexpected)}")
    return args


def rounded_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    board = frame.copy()
    numeric_cols = board.select_dtypes(include=[np.number]).columns
    board[numeric_cols] = board[numeric_cols].round(6)
    return board.to_dict(orient="records")


def top_importance_records(
    frame: pd.DataFrame,
    score_column: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    if frame.empty or score_column not in frame.columns:
        return []
    ordered = frame.sort_values(score_column, ascending=False).head(limit).copy()
    numeric_cols = ordered.select_dtypes(include=[np.number]).columns
    ordered[numeric_cols] = ordered[numeric_cols].round(6)
    return ordered.to_dict(orient="records")


def normalize_nested_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_nested_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_nested_payload(item) for item in value]
    if isinstance(value, tuple):
        return [normalize_nested_payload(item) for item in value]
    if isinstance(value, (np.floating, float)):
        return round(float(value), 6)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def build_market_diagnostics(horizon_artifacts: HorizonArtifacts) -> dict[str, Any]:
    if horizon_artifacts.feature_snapshot.empty or "latest_value" not in horizon_artifacts.feature_snapshot.columns:
        return {}

    feature_values = pd.to_numeric(horizon_artifacts.feature_snapshot["latest_value"], errors="coerce")

    def value(name: str) -> float | None:
        item = feature_values.get(name, np.nan)
        return None if pd.isna(item) else float(item)

    model_input_close = float(horizon_artifacts.regression_forecast.model_input_close)
    reference_price = float(horizon_artifacts.regression_forecast.last_close)
    live_gap_pct = None
    if np.isfinite(model_input_close) and model_input_close > 0.0 and np.isfinite(reference_price):
        live_gap_pct = float(reference_price / model_input_close - 1.0)

    return {
        "model_input_close": model_input_close,
        "reference_price": reference_price,
        "live_gap_pct": live_gap_pct,
        "eth_return_1": value("eth_return_1"),
        "eth_return_3": value("eth_return_3"),
        "eth_return_5": value("eth_return_5"),
        "eth_return_7": value("eth_return_7"),
        "eth_return_14": value("eth_return_14"),
        "eth_return_30": value("eth_return_30"),
        "eth_ma_7_ratio": value("eth_ma_7_ratio"),
        "eth_ma_14_ratio": value("eth_ma_14_ratio"),
        "eth_ma_30_ratio": value("eth_ma_30_ratio"),
        "eth_ma_90_ratio": value("eth_ma_90_ratio"),
        "eth_vwap_7_ratio": value("eth_vwap_7_ratio"),
        "eth_vwap_20_ratio": value("eth_vwap_20_ratio"),
        "eth_vwap_7_30_spread": value("eth_vwap_7_30_spread"),
        "eth_vol_7": value("eth_vol_7"),
        "eth_vol_14": value("eth_vol_14"),
        "eth_vol_30": value("eth_vol_30"),
        "eth_range_pct": value("eth_range_pct"),
        "eth_close_open_pct": value("eth_close_open_pct"),
        "eth_drawdown_30": value("eth_drawdown_30"),
        "eth_drawdown_90": value("eth_drawdown_90"),
        "eth_return_accel_7_30": value("eth_return_accel_7_30"),
        "eth_up_streak": value("eth_up_streak"),
        "eth_down_streak": value("eth_down_streak"),
        "eth_up_day_ratio_7": value("eth_up_day_ratio_7"),
        "eth_up_day_ratio_14": value("eth_up_day_ratio_14"),
        "eth_rebound_from_20d_low": value("eth_rebound_from_20d_low"),
        "eth_rebound_from_60d_low": value("eth_rebound_from_60d_low"),
        "eth_distance_to_20d_high": value("eth_distance_to_20d_high"),
        "eth_distance_to_60d_high": value("eth_distance_to_60d_high"),
        "eth_adx_trend_bias_14": value("eth_adx_trend_bias_14"),
        "eth_rsi_14_change_3": value("eth_rsi_14_change_3"),
        "eth_macd_hist_change_3": value("eth_macd_hist_change_3"),
    }


def build_feature_quality_summary(
    feature_coverage_summary: pd.DataFrame,
    dropped_limit: int = 12,
    weak_kept_limit: int = 6,
) -> dict[str, Any]:
    if feature_coverage_summary.empty:
        return {}

    summary = feature_coverage_summary.copy()
    summary["kept"] = summary["kept"].astype(bool)
    standard_mask = summary["feature_group"] == "standard"
    history_mask = summary["feature_group"] == "history"
    kept_mask = summary["kept"]
    dropped_mask = ~kept_mask

    weakest_kept = (
        summary.loc[kept_mask]
        .sort_values(["coverage_ratio", "unique_values", "feature"], ascending=[True, True, True])
        .head(weak_kept_limit)
    )
    dropped = (
        summary.loc[dropped_mask]
        .sort_values(["coverage_ratio", "unique_values", "feature"], ascending=[True, True, True])
        .head(dropped_limit)
    )

    def rows_to_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
        if frame.empty:
            return []
        payload = frame[["feature", "feature_group", "coverage_ratio", "unique_values", "min_rows_required"]].copy()
        numeric_cols = payload.select_dtypes(include=[np.number]).columns
        payload[numeric_cols] = payload[numeric_cols].round(6)
        return payload.to_dict(orient="records")

    return {
        "kept_standard_count": int((kept_mask & standard_mask).sum()),
        "total_standard_count": int(standard_mask.sum()),
        "kept_history_count": int((kept_mask & history_mask).sum()),
        "total_history_count": int(history_mask.sum()),
        "dropped_feature_count": int(dropped_mask.sum()),
        "weakest_kept_features": rows_to_records(weakest_kept),
        "dropped_feature_sample": rows_to_records(dropped),
    }


def build_selected_model_diagnostics(horizon_artifacts: HorizonArtifacts) -> dict[str, Any]:
    def find_row(frame: pd.DataFrame, model_name: str) -> dict[str, Any]:
        if frame.empty or "model" not in frame.columns:
            return {}
        match = frame.loc[frame["model"] == model_name]
        return match.iloc[0].to_dict() if not match.empty else {}

    regression_model = horizon_artifacts.regression_forecast.model_name
    classification_model = horizon_artifacts.classification_forecast.model_name
    regime_model = horizon_artifacts.regime_forecast.model_name
    reversal_model = horizon_artifacts.reversal_forecast.model_name
    recent_holdout = horizon_artifacts.recent_holdout_report

    regression_holdout = recent_holdout.loc[
        (recent_holdout["task"] == "regression") & (recent_holdout["model"] == regression_model)
    ] if not recent_holdout.empty else pd.DataFrame()
    classification_holdout = recent_holdout.loc[
        (recent_holdout["task"] == "classification") & (recent_holdout["model"] == classification_model)
    ] if not recent_holdout.empty else pd.DataFrame()

    regression_holdout_row = regression_holdout.iloc[0].to_dict() if not regression_holdout.empty else {}
    classification_holdout_row = classification_holdout.iloc[0].to_dict() if not classification_holdout.empty else {}

    return {
        "regression": {
            "leaderboard": find_row(horizon_artifacts.regression_leaderboard, regression_model),
            "backtest": find_row(horizon_artifacts.regression_backtest, regression_model),
            "recent_holdout": regression_holdout_row,
        },
        "classification": {
            "leaderboard": find_row(horizon_artifacts.classification_leaderboard, classification_model),
            "backtest": find_row(horizon_artifacts.classification_backtest, classification_model),
            "recent_holdout": classification_holdout_row,
        },
        "regime": {
            "leaderboard": find_row(horizon_artifacts.regime_leaderboard, regime_model),
            "backtest": find_row(horizon_artifacts.regime_backtest, regime_model),
            "recent_holdout": {},
        },
        "reversal": {
            "leaderboard": find_row(horizon_artifacts.reversal_leaderboard, reversal_model),
            "backtest": find_row(horizon_artifacts.reversal_backtest, reversal_model),
            "recent_holdout": {},
        },
    }


def build_recent_holdout_highlights(
    recent_holdout_report: pd.DataFrame,
    limit_per_task: int = 3,
) -> dict[str, list[dict[str, Any]]]:
    if recent_holdout_report.empty or "task" not in recent_holdout_report.columns:
        return {}

    highlights: dict[str, list[dict[str, Any]]] = {}
    for task in ("regression", "classification", "benchmark"):
        task_frame = recent_holdout_report.loc[recent_holdout_report["task"] == task].head(limit_per_task).copy()
        if task_frame.empty:
            continue
        numeric_cols = task_frame.select_dtypes(include=[np.number]).columns
        task_frame[numeric_cols] = task_frame[numeric_cols].round(6)
        highlights[task] = task_frame.to_dict(orient="records")
    return highlights


def summarize_artifacts(artifacts: PipelineArtifacts) -> dict[str, Any]:
    horizons: dict[str, Any] = {}
    for horizon, horizon_artifacts in artifacts.horizons.items():
        regression = asdict(horizon_artifacts.regression_forecast)
        classification = asdict(horizon_artifacts.classification_forecast)
        regime = asdict(horizon_artifacts.regime_forecast)
        reversal = asdict(horizon_artifacts.reversal_forecast)
        hybrid = asdict(horizon_artifacts.hybrid_forecast)
        for payload in (regression, classification, regime, reversal, hybrid):
            for key in list(payload):
                if isinstance(payload[key], float):
                    payload[key] = round(payload[key], 6)

        market_diagnostics = build_market_diagnostics(horizon_artifacts)
        selected_model_diagnostics = build_selected_model_diagnostics(horizon_artifacts)
        feature_quality_summary = build_feature_quality_summary(horizon_artifacts.feature_coverage_summary)
        recent_holdout_highlights = build_recent_holdout_highlights(horizon_artifacts.recent_holdout_report)

        horizons[str(horizon)] = {
            "data_window_summary": rounded_records(horizon_artifacts.data_window_summary),
            "feature_coverage_summary": rounded_records(horizon_artifacts.feature_coverage_summary),
            "recent_holdout_report": rounded_records(horizon_artifacts.recent_holdout_report),
            "market_diagnostics": normalize_nested_payload(market_diagnostics),
            "feature_quality_summary": normalize_nested_payload(feature_quality_summary),
            "selected_model_diagnostics": normalize_nested_payload(selected_model_diagnostics),
            "recent_holdout_highlights": normalize_nested_payload(recent_holdout_highlights),
            "regression_forecast": regression,
            "classification_forecast": classification,
            "regime_forecast": regime,
            "reversal_forecast": reversal,
            "hybrid_forecast": hybrid,
            "regression_leaderboard": rounded_records(horizon_artifacts.regression_leaderboard),
            "classification_leaderboard": rounded_records(horizon_artifacts.classification_leaderboard),
            "regime_leaderboard": rounded_records(horizon_artifacts.regime_leaderboard),
            "reversal_leaderboard": rounded_records(horizon_artifacts.reversal_leaderboard),
            "regression_backtest": rounded_records(horizon_artifacts.regression_backtest),
            "classification_backtest": rounded_records(horizon_artifacts.classification_backtest),
            "regime_backtest": rounded_records(horizon_artifacts.regime_backtest),
            "reversal_backtest": rounded_records(horizon_artifacts.reversal_backtest),
            "benchmark_backtest": rounded_records(horizon_artifacts.benchmark_backtest),
            "regression_explainability": {
                "model_name": horizon_artifacts.regression_explainability.model_name,
                "native_feature_importance_top": top_importance_records(
                    horizon_artifacts.regression_explainability.native_feature_importance,
                    "importance_abs",
                ),
                "permutation_importance_top": top_importance_records(
                    horizon_artifacts.regression_explainability.permutation_importance,
                    "importance_mean",
                ),
                "shap_global_importance_top": top_importance_records(
                    horizon_artifacts.regression_explainability.shap_global_importance,
                    "mean_abs_shap",
                ),
                "shap_latest_contributions_top": top_importance_records(
                    horizon_artifacts.regression_explainability.shap_latest_contributions,
                    "abs_shap_value",
                ),
                "notes": rounded_records(horizon_artifacts.regression_explainability.notes),
            },
            "classification_explainability": {
                "model_name": horizon_artifacts.classification_explainability.model_name,
                "native_feature_importance_top": top_importance_records(
                    horizon_artifacts.classification_explainability.native_feature_importance,
                    "importance_abs",
                ),
                "permutation_importance_top": top_importance_records(
                    horizon_artifacts.classification_explainability.permutation_importance,
                    "importance_mean",
                ),
                "shap_global_importance_top": top_importance_records(
                    horizon_artifacts.classification_explainability.shap_global_importance,
                    "mean_abs_shap",
                ),
                "shap_latest_contributions_top": top_importance_records(
                    horizon_artifacts.classification_explainability.shap_latest_contributions,
                    "abs_shap_value",
                ),
                "notes": rounded_records(horizon_artifacts.classification_explainability.notes),
            },
        }

    return {
        "optional_models": normalize_nested_payload(OPTIONAL_MODEL_STATUS),
        "run_status": "partial_success" if artifacts.failed_horizons else "ok",
        "failed_horizons": normalize_nested_payload(artifacts.failed_horizons),
        "horizons": horizons,
        "note": "This is a statistical forecast from historical data, not investment advice.",
    }


def format_console_metric(value: Any, digits: int = 6) -> str:
    if value is None:
        return "n/a"
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.notna(numeric):
        return f"{float(numeric):.{digits}f}"
    return str(value)


def format_feature_record(record: dict[str, Any]) -> str:
    return (
        f"{record.get('feature', 'unknown')}"
        f"(coverage={format_console_metric(record.get('coverage_ratio'))}, "
        f"unique={record.get('unique_values', 'n/a')}, "
        f"group={record.get('feature_group', 'n/a')})"
    )


def safe_float(value: Any) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) else float("nan")


def confidence_bucket(value: Any) -> str:
    numeric = safe_float(value)
    if pd.isna(numeric):
        return "low"
    if numeric >= 0.72:
        return "high"
    if numeric >= 0.55:
        return "medium"
    return "low"


def summarize_market_interpretation(
    horizon_days: int,
    regression: dict[str, Any],
    classification: dict[str, Any],
    regime: dict[str, Any],
    reversal: dict[str, Any],
    hybrid: dict[str, Any],
) -> str:
    adjusted_return = safe_float(hybrid.get("adjusted_predicted_return"))
    market_state = str(hybrid.get("predicted_market_state") or regime.get("predicted_regime") or "SIDEWAYS")
    direction = str(hybrid.get("predicted_direction") or classification.get("predicted_direction") or "FLAT")
    signal_tier = str(hybrid.get("signal_tier") or "NO_SIGNAL")
    reversal_signal = str(hybrid.get("predicted_reversal_signal") or reversal.get("predicted_reversal") or "NONE")

    if pd.isna(adjusted_return):
        return "unclear outlook"

    if market_state == "SIDEWAYS":
        if adjusted_return >= 0.03:
            base = "sideways with mild upside bias"
        elif adjusted_return <= -0.03:
            base = "sideways with mild downside bias"
        else:
            base = "sideways / range-bound"
    elif market_state == "UPTREND":
        base = "uptrend continuation"
    elif market_state == "DOWNTREND":
        base = "downtrend continuation"
    else:
        base = "mixed market state"

    if direction == "UP" and adjusted_return >= 0.02:
        base += ", directional upside active"
    elif direction == "DOWN" and adjusted_return <= -0.02:
        base += ", directional downside active"
    elif direction == "FLAT":
        base += ", no strong directional edge"

    if reversal_signal == "BOTTOM_REVERSAL":
        base += ", bottom reversal watch"
    elif reversal_signal == "TOP_REVERSAL":
        base += ", top reversal watch"
    else:
        base += ", no confirmed reversal"

    if signal_tier == "HIGH_CONFIDENCE":
        base += ", high-confidence signal active"
    elif signal_tier == "OBSERVE":
        base += ", observe-only bias"

    if horizon_days >= 30 and abs(adjusted_return) >= 0.12:
        base += ", high-magnitude scenario"
    return base


def summarize_model_trust(
    task_name: str,
    selection_basis: str,
    diagnostics: dict[str, Any],
) -> str:
    if str(selection_basis).startswith("fallback_heuristic"):
        return "heuristic fallback"

    leaderboard = diagnostics.get("leaderboard", {})
    backtest = diagnostics.get("backtest", {})
    holdout = diagnostics.get("recent_holdout", {})

    score = 0
    details: list[str] = []

    backtest_return = safe_float(backtest.get("total_return"))
    backtest_sharpe = safe_float(backtest.get("sharpe"))
    holdout_return = safe_float(holdout.get("total_return"))
    holdout_sharpe = safe_float(holdout.get("sharpe"))

    if pd.notna(backtest_return) and backtest_return > 0:
        score += 1
        details.append("backtest+")
    if pd.notna(backtest_sharpe) and backtest_sharpe > 0:
        score += 1
    if pd.notna(holdout_return) and holdout_return > 0:
        score += 1
        details.append("holdout+")
    if pd.notna(holdout_sharpe) and holdout_sharpe > 0:
        score += 1

    if task_name == "direction":
        cv_bal = safe_float(leaderboard.get("balanced_accuracy"))
        cv_auc = safe_float(leaderboard.get("roc_auc"))
        if (pd.notna(cv_bal) and cv_bal >= 0.53) or (pd.notna(cv_auc) and cv_auc >= 0.57):
            score += 1
        elif direction_cv_is_weak(leaderboard):
            details.append("cv_weak")
    elif task_name == "regime":
        cv_bal = safe_float(leaderboard.get("balanced_accuracy"))
        cv_f1 = safe_float(leaderboard.get("macro_f1"))
        if (pd.notna(cv_bal) and cv_bal >= 0.52) or (pd.notna(cv_f1) and cv_f1 >= 0.42):
            score += 1
    elif task_name == "regression":
        cv_directional = safe_float(leaderboard.get("directional_accuracy"))
        if pd.notna(cv_directional) and cv_directional >= 0.55:
            score += 1

    if score >= 4:
        label = "high"
    elif score >= 2:
        label = "medium"
    else:
        label = "low"

    if task_name == "direction" and label == "high" and direction_cv_is_weak(leaderboard):
        label = "medium"

    if not details:
        details_text = "no backtest/holdout confirmation"
    else:
        details_text = ", ".join(details)
    return f"{label} ({details_text})"


def best_recent_holdout_model(
    recent_holdout_report: list[dict[str, Any]],
    task_name: str,
) -> str:
    if not recent_holdout_report:
        return ""
    frame = pd.DataFrame(recent_holdout_report)
    if frame.empty or "task" not in frame.columns:
        return ""
    frame = frame.loc[frame["task"] == task_name].copy()
    if frame.empty or "model" not in frame.columns:
        return ""
    frame["total_return_num"] = pd.to_numeric(frame.get("total_return"), errors="coerce")
    frame["sharpe_num"] = pd.to_numeric(frame.get("sharpe"), errors="coerce")
    frame = frame.sort_values(
        ["total_return_num", "sharpe_num"],
        ascending=[False, False],
        na_position="last",
    ).reset_index(drop=True)
    return str(frame.loc[0, "model"]) if not frame.empty else ""


def build_model_guide_text(
    selected_model_name: str,
    backtest_rows: list[dict[str, Any]],
    recent_holdout_report: list[dict[str, Any]],
    task_name: str,
) -> str:
    backtest_leader = ""
    if backtest_rows:
        backtest_leader = str(backtest_rows[0].get("model", "")).strip()
    holdout_leader = best_recent_holdout_model(recent_holdout_report, task_name)
    parts = [f"selected={selected_model_name or 'n/a'}"]
    if backtest_leader:
        parts.append(f"backtest_leader={backtest_leader}")
    if holdout_leader:
        parts.append(f"holdout_leader={holdout_leader}")
    if backtest_leader and holdout_leader and backtest_leader == holdout_leader:
        parts.append("consensus=yes")
    elif backtest_leader or holdout_leader:
        parts.append("consensus=no")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    artifacts = run_suite(
        period=args.period,
        interval=args.interval,
        horizons=args.horizons,
        cv_splits=args.cv_splits,
        cv_test_size=args.cv_test_size,
        output_dir=args.output_dir,
        dataset_csv=args.dataset_csv,
        master_data_csv=(args.master_data_csv or None),
        prediction_history_csv=(args.prediction_history_csv or None),
        external_feature_csvs=(args.external_features_csv or None),
        drive_root=(args.drive_root or None),
        use_dataset_cache=not args.no_dataset_cache,
        use_prediction_history_features=not args.no_prediction_history_features,
        use_realtime_reference_price=not args.no_realtime_reference_price,
        auto_install_catboost=not args.no_auto_install_catboost,
        build_explainability=(not args.no_explainability) and (not args.fast_mode),
        fast_mode=args.fast_mode,
        cache_lookback_rows=args.cache_lookback_rows,
        min_feature_coverage=args.min_feature_coverage,
        compact_output=args.compact_output,
        save_outputs=not args.no_save,
        continue_on_horizon_error=not args.fail_fast,
        verbose=(not args.quiet) and (not args.json),
    )
    external_summary = build_external_feature_summary(artifacts.raw_data)
    payload = summarize_artifacts(artifacts)

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("ETH-USD multi-horizon forecast suite")
    horizons_payload = payload.get("horizons") or {}
    if horizons_payload:
        headline_horizons = sorted(horizons_payload.keys(), key=lambda key: int(key))
        first_reference_price: float | None = None
        first_input_timestamp: Any = None
        for horizon_key in headline_horizons:
            horizon_payload = horizons_payload[horizon_key]
            regression = horizon_payload.get("regression_forecast") or {}
            ref_candidate = regression.get("last_close")
            if ref_candidate is not None:
                first_reference_price = safe_float(ref_candidate)
                data_window_list = horizon_payload.get("data_window_summary") or []
                if data_window_list:
                    first_input_timestamp = data_window_list[0].get("latest_prediction_input_timestamp")
                break
        if first_reference_price is not None and not pd.isna(first_reference_price):
            print(
                "  Forecast reference: "
                f"input_timestamp={first_input_timestamp}, "
                f"reference_price=${first_reference_price:,.2f}"
            )
        for horizon_key in headline_horizons:
            horizon_payload = horizons_payload[horizon_key]
            regression = horizon_payload.get("regression_forecast") or {}
            classification = horizon_payload.get("classification_forecast") or {}
            hybrid = horizon_payload.get("hybrid_forecast") or {}
            data_window_list = horizon_payload.get("data_window_summary") or []
            target_timestamp = (
                data_window_list[0].get("prediction_target_timestamp") if data_window_list else None
            )
            reg_close = safe_float(regression.get("predicted_close"))
            reg_return = safe_float(regression.get("predicted_return"))
            hybrid_close = safe_float(hybrid.get("adjusted_predicted_close"))
            hybrid_bear = safe_float(hybrid.get("bear_predicted_close"))
            hybrid_bull = safe_float(hybrid.get("bull_predicted_close"))
            active_close = safe_float(hybrid.get("active_predicted_close"))
            active_bear = safe_float(hybrid.get("active_bear_predicted_close"))
            active_bull = safe_float(hybrid.get("active_bull_predicted_close"))
            direction = hybrid.get("predicted_direction") or classification.get("predicted_direction") or "n/a"
            prob_up = safe_float(classification.get("probability_up"))
            confidence = safe_float(hybrid.get("confidence"))
            tier = hybrid.get("signal_tier") or "n/a"

            def _fmt_price(value: float) -> str:
                return f"${value:,.2f}" if not pd.isna(value) else "n/a"

            def _fmt_return(value: float) -> str:
                return f"{value * 100:+.2f}%" if not pd.isna(value) else "n/a"

            print(
                f"  {horizon_key}d forecast -> target={target_timestamp}: "
                f"regression={_fmt_price(reg_close)} ({_fmt_return(reg_return)}), "
                f"direction={direction} (prob_up={format_console_metric(prob_up, digits=3)}, "
                f"confidence={format_console_metric(confidence, digits=3)}, tier={tier})"
            )
            print(
                f"    hybrid conservative: base={_fmt_price(hybrid_close)} "
                f"bear={_fmt_price(hybrid_bear)} bull={_fmt_price(hybrid_bull)}"
            )
            print(
                f"    hybrid active:       base={_fmt_price(active_close)} "
                f"bear={_fmt_price(active_bear)} bull={_fmt_price(active_bull)}"
            )
    optional_models = payload.get("optional_models", {})
    print(
        "  Optional models: "
        f"catboost_available={optional_models.get('catboost_available')}, "
        f"catboost_installed_now={optional_models.get('catboost_installed_now')}, "
        f"catboost_error={optional_models.get('catboost_error') or 'none'}"
    )
    failed_horizons = payload.get("failed_horizons", [])
    if failed_horizons:
        print(
            "  Horizon failures: "
            f"count={len(failed_horizons)}, run_status={payload.get('run_status', 'partial_success')}"
        )
        for item in failed_horizons:
            print(
                "    - "
                f"horizon={item.get('horizon_steps')}d, "
                f"error_type={item.get('error_type')}, "
                f"error_message={item.get('error_message')}"
            )
    if not external_summary.empty:
        short_history = external_summary.loc[
            external_summary["short_history_excluded"].fillna(False).astype(bool)
        ].copy()
        stale_external = external_summary.loc[
            pd.to_numeric(external_summary["latest_age_days"], errors="coerce").fillna(0.0) >= 30.0
        ].copy()
        print(
            "  External feature health: "
            f"total_vendor_columns={len(external_summary)}, "
            f"short_history_excluded={len(short_history)}, "
            f"stale_30d_plus={len(stale_external)}"
        )
        if not short_history.empty:
            short_history_text = ", ".join(
                f"{row['column']}(span_days={format_console_metric(row.get('history_span_days'), digits=0)})"
                for row in short_history.head(8).to_dict(orient="records")
            )
            print(f"  Short-history vendor columns skipped: {short_history_text}")
        if not stale_external.empty:
            stale_text = ", ".join(
                f"{row['column']}(age_days={format_console_metric(row.get('latest_age_days'), digits=0)})"
                for row in stale_external.head(8).to_dict(orient="records")
            )
            print(f"  Stale external columns: {stale_text}")
    for horizon_key, horizon_payload in payload["horizons"].items():
        data_window = horizon_payload["data_window_summary"][0] if horizon_payload["data_window_summary"] else {}
        market_diagnostics = horizon_payload.get("market_diagnostics", {})
        feature_quality = horizon_payload.get("feature_quality_summary", {})
        selected_model_diagnostics = horizon_payload.get("selected_model_diagnostics", {})
        recent_holdout_highlights = horizon_payload.get("recent_holdout_highlights", {})
        recent_holdout_report = horizon_payload.get("recent_holdout_report", [])
        regression = horizon_payload["regression_forecast"]
        classification = horizon_payload["classification_forecast"]
        regime = horizon_payload.get("regime_forecast", {})
        reversal = horizon_payload.get("reversal_forecast", {})
        hybrid = horizon_payload.get("hybrid_forecast", {})
        print(f"- Horizon: {horizon_key} days")
        if data_window:
            print(
                "  Data window: "
                f"raw_rows={data_window['raw_rows']}, "
                f"raw_span_days={data_window['raw_span_days']}, "
                f"feature_rows={data_window['feature_rows']}, "
                f"training_rows={data_window['training_rows']}, "
                f"candidate_feature_count={data_window['candidate_feature_count']}, "
                f"feature_count={data_window['feature_count']}, "
                f"dropped_feature_count={data_window['dropped_feature_count']}, "
                f"train_rows_per_fold={data_window['min_train_rows_per_fold']}~{data_window['max_train_rows_per_fold']}, "
                f"validation_rows_per_fold={data_window['min_validation_rows_per_fold']}~{data_window['max_validation_rows_per_fold']}"
            )
            print(
                "  Data sufficiency: "
                f"practical_min_years={data_window['practical_min_years']}, "
                f"preferred_years={data_window['preferred_years']}, "
                f"meets_practical_minimum={data_window['meets_practical_minimum']}, "
                f"meets_preferred_length={data_window['meets_preferred_length']}"
            )
            print(
                "  Timing: "
                f"input_timestamp={data_window.get('latest_prediction_input_timestamp')}, "
                f"target_timestamp={data_window.get('prediction_target_timestamp')}, "
                f"reference_timestamp={regression.get('reference_price_timestamp', 'n/a')}"
            )
            print(
                "  Learning setup: "
                f"training_window_years={format_console_metric(data_window.get('training_window_years_applied'))}, "
                f"full_training_rows={data_window.get('full_training_rows')}, "
                f"window_rows_dropped={data_window.get('training_window_rows_dropped')}, "
                f"sample_weight_half_life_years={format_console_metric(data_window.get('sample_weight_half_life_years'))}, "
                f"sample_weight_range={format_console_metric(data_window.get('sample_weight_min'))}~"
                f"{format_console_metric(data_window.get('sample_weight_max'))}"
            )
            print(
                "  Validation: "
                f"cv_gap_rows={data_window.get('gap_rows_between_train_and_validation')}, "
                f"recent_holdout_rows={data_window.get('recent_holdout_rows')}, "
                f"recent_holdout_embargo_rows={data_window.get('recent_holdout_embargo_rows')}, "
                f"threshold_scheme={data_window.get('threshold_validation_scheme')}, "
                f"leakage_safe={data_window.get('validation_leakage_safe')}"
            )
        if market_diagnostics:
            print(
                "  Market context: "
                f"model_input_close={format_console_metric(market_diagnostics.get('model_input_close'))}, "
                f"reference_price={format_console_metric(market_diagnostics.get('reference_price'))}, "
                f"live_gap_pct={format_console_metric(market_diagnostics.get('live_gap_pct'))}, "
                f"ret_1={format_console_metric(market_diagnostics.get('eth_return_1'))}, "
                f"ret_3={format_console_metric(market_diagnostics.get('eth_return_3'))}, "
                f"ret_5={format_console_metric(market_diagnostics.get('eth_return_5'))}, "
                f"ret_7={format_console_metric(market_diagnostics.get('eth_return_7'))}, "
                f"ret_14={format_console_metric(market_diagnostics.get('eth_return_14'))}, "
                f"ret_30={format_console_metric(market_diagnostics.get('eth_return_30'))}"
            )
            print(
                "  Trend context: "
                f"ma_7_ratio={format_console_metric(market_diagnostics.get('eth_ma_7_ratio'))}, "
                f"ma_14_ratio={format_console_metric(market_diagnostics.get('eth_ma_14_ratio'))}, "
                f"ma_30_ratio={format_console_metric(market_diagnostics.get('eth_ma_30_ratio'))}, "
                f"ma_90_ratio={format_console_metric(market_diagnostics.get('eth_ma_90_ratio'))}, "
                f"vwap_7_ratio={format_console_metric(market_diagnostics.get('eth_vwap_7_ratio'))}, "
                f"vwap_20_ratio={format_console_metric(market_diagnostics.get('eth_vwap_20_ratio'))}, "
                f"drawdown_30={format_console_metric(market_diagnostics.get('eth_drawdown_30'))}, "
                f"drawdown_90={format_console_metric(market_diagnostics.get('eth_drawdown_90'))}"
            )
            print(
                "  Volatility context: "
                f"vol_7={format_console_metric(market_diagnostics.get('eth_vol_7'))}, "
                f"vol_14={format_console_metric(market_diagnostics.get('eth_vol_14'))}, "
                f"vol_30={format_console_metric(market_diagnostics.get('eth_vol_30'))}, "
                f"range_pct={format_console_metric(market_diagnostics.get('eth_range_pct'))}, "
                f"close_open_pct={format_console_metric(market_diagnostics.get('eth_close_open_pct'))}, "
                f"return_accel_7_30={format_console_metric(market_diagnostics.get('eth_return_accel_7_30'))}"
            )
            print(
                "  Transition context: "
                f"up_streak={format_console_metric(market_diagnostics.get('eth_up_streak'))}, "
                f"down_streak={format_console_metric(market_diagnostics.get('eth_down_streak'))}, "
                f"up_day_ratio_7={format_console_metric(market_diagnostics.get('eth_up_day_ratio_7'))}, "
                f"up_day_ratio_14={format_console_metric(market_diagnostics.get('eth_up_day_ratio_14'))}, "
                f"rebound_20d_low={format_console_metric(market_diagnostics.get('eth_rebound_from_20d_low'))}, "
                f"rebound_60d_low={format_console_metric(market_diagnostics.get('eth_rebound_from_60d_low'))}, "
                f"distance_20d_high={format_console_metric(market_diagnostics.get('eth_distance_to_20d_high'))}, "
                f"distance_60d_high={format_console_metric(market_diagnostics.get('eth_distance_to_60d_high'))}, "
                f"adx_bias_14={format_console_metric(market_diagnostics.get('eth_adx_trend_bias_14'))}, "
                f"rsi_14_change_3={format_console_metric(market_diagnostics.get('eth_rsi_14_change_3'))}, "
                f"macd_hist_change_3={format_console_metric(market_diagnostics.get('eth_macd_hist_change_3'))}"
            )
        print(
            "  Regression: "
            f"model={regression['model_name']}, "
            f"selection_basis={regression['selection_basis']}, "
            f"reference_price={regression['last_close']}, "
            f"reference_source={regression.get('reference_price_source', 'daily_feature_close')}, "
            f"predicted_close={regression['predicted_close']}, "
            f"predicted_return={regression['predicted_return']}, "
            f"window=[{regression['lower_close_10']}, {regression['upper_close_90']}]"
        )
        print(
            "  Direction: "
            f"model={classification['model_name']}, "
            f"selection_basis={classification['selection_basis']}, "
            f"reference_price={classification['last_close']}, "
            f"reference_source={classification.get('reference_price_source', 'daily_feature_close')}, "
            f"prediction={classification['predicted_direction']}, "
            f"signal_threshold={classification['signal_threshold']}, "
            f"prob_up={classification['probability_up']}, "
            f"confidence={classification['confidence']}"
        )
        if regime:
            print(
                "  Regime: "
                f"model={regime['model_name']}, "
                f"selection_basis={regime['selection_basis']}, "
                f"state={regime['predicted_regime']}, "
                f"prob_uptrend={regime['probability_uptrend']}, "
                f"prob_sideways={regime['probability_sideways']}, "
                f"prob_downtrend={regime['probability_downtrend']}, "
                f"confidence={regime['confidence']}"
            )
        if reversal:
            print(
                "  Reversal: "
                f"model={reversal['model_name']}, "
                f"selection_basis={reversal['selection_basis']}, "
                f"signal={reversal['predicted_reversal']}, "
                f"prob_bottom={reversal['probability_bottom_reversal']}, "
                f"prob_none={reversal['probability_no_reversal']}, "
                f"prob_top={reversal['probability_top_reversal']}, "
                f"confidence={reversal['confidence']}"
            )
        if hybrid:
            print(
                "  Hybrid: "
                f"model={hybrid['model_name']}, "
                f"market_state={hybrid['predicted_market_state']}, "
                f"reversal_signal={hybrid['predicted_reversal_signal']}, "
                f"direction={hybrid['predicted_direction']}, "
                f"bias_direction={hybrid.get('bias_direction', hybrid['predicted_direction'])}, "
                f"signal_tier={hybrid.get('signal_tier', 'n/a')}, "
                f"hybrid_score={hybrid['hybrid_score']}, "
                f"adjusted_return={hybrid['adjusted_predicted_return']}, "
                f"adjusted_close={hybrid['adjusted_predicted_close']}, "
                f"confidence={hybrid['confidence']}"
            )
            print(
                "  Signal mode: "
                f"high_confidence={hybrid.get('high_confidence_signal', False)}, "
                f"tier={hybrid.get('signal_tier', 'n/a')}, "
                f"reason={hybrid.get('signal_reason', 'n/a')}"
            )
            print(
                "  Scenario range: "
                f"bear_close={hybrid.get('bear_predicted_close')}, "
                f"base_close={hybrid.get('adjusted_predicted_close')}, "
                f"bull_close={hybrid.get('bull_predicted_close')}, "
                f"spread={hybrid.get('scenario_spread')}, "
                f"vol_scale={hybrid.get('volatility_scale')}"
            )
            print(
                "  Dual track: "
                f"conservative_return={hybrid.get('adjusted_predicted_return')}, "
                f"conservative_close={hybrid.get('adjusted_predicted_close')}, "
                f"active_direction={hybrid.get('active_predicted_direction')}, "
                f"active_return={hybrid.get('active_predicted_return')}, "
                f"active_close={hybrid.get('active_predicted_close')}, "
                f"active_confidence={hybrid.get('active_confidence')}"
            )
            print(
                "  Active range: "
                f"bear_close={hybrid.get('active_bear_predicted_close')}, "
                f"base_close={hybrid.get('active_predicted_close')}, "
                f"bull_close={hybrid.get('active_bull_predicted_close')}"
            )
            print(
                "  Macro context: "
                f"status={hybrid.get('macro_readiness_status', 'n/a')}, "
                f"risk_regime={hybrid.get('macro_risk_regime', 'n/a')}, "
                f"directional_bias={hybrid.get('macro_directional_bias', 'n/a')}, "
                f"selection_tag={hybrid.get('macro_selection_tag', 'n/a')}, "
                f"used={hybrid.get('macro_used_feature_count', 'n/a')}, "
                f"missing={hybrid.get('macro_missing_feature_count', 'n/a')}"
            )
        quick_read = summarize_market_interpretation(
            horizon_days=int(horizon_key),
            regression=regression,
            classification=classification,
            regime=regime,
            reversal=reversal,
            hybrid=hybrid,
        )
        regression_selected = selected_model_diagnostics.get("regression", {})
        classification_selected = selected_model_diagnostics.get("classification", {})
        regime_selected = selected_model_diagnostics.get("regime", {})
        reversal_selected = selected_model_diagnostics.get("reversal", {})
        regression_trust = summarize_model_trust(
            "regression",
            regression.get("selection_basis", ""),
            regression_selected,
        )
        direction_trust = summarize_model_trust(
            "direction",
            classification.get("selection_basis", ""),
            classification_selected,
        )
        regime_trust = summarize_model_trust(
            "regime",
            regime.get("selection_basis", ""),
            regime_selected,
        ) if regime else "n/a"
        reversal_trust = summarize_model_trust(
            "reversal",
            reversal.get("selection_basis", ""),
            reversal_selected,
        ) if reversal else "n/a"
        if hybrid:
            hybrid_tier = str(hybrid.get("signal_tier") or "NO_SIGNAL")
            if bool(hybrid.get("high_confidence_signal")):
                hybrid_trust = "high (high-confidence signal)"
            elif hybrid_tier == "OBSERVE":
                hybrid_trust = "medium (observe-only bias)"
            else:
                hybrid_trust = f"{confidence_bucket(hybrid.get('confidence'))} ({hybrid_tier.lower()})"
        else:
            hybrid_trust = "n/a"
        print(f"  Quick read: {quick_read}")
        print(
            "  Trust summary: "
            f"regression={regression_trust}, "
            f"direction={direction_trust}, "
            f"regime={regime_trust}, "
            f"reversal={reversal_trust}, "
            f"hybrid={hybrid_trust}"
        )
        print(
            "  Model guide: "
            f"regression[{build_model_guide_text(regression.get('model_name', ''), horizon_payload.get('regression_backtest', []), recent_holdout_report, 'regression')}]; "
            f"direction[{build_model_guide_text(classification.get('model_name', ''), horizon_payload.get('classification_backtest', []), recent_holdout_report, 'classification')}]"
        )
        regression_selected_leaderboard = regression_selected.get("leaderboard", {})
        regression_selected_backtest = regression_selected.get("backtest", {})
        regression_selected_holdout = regression_selected.get("recent_holdout", {})
        if regression_selected_leaderboard or regression_selected_backtest or regression_selected_holdout:
            print(
                "  Selected regression diagnostics: "
                f"cv_rmse={format_console_metric(regression_selected_leaderboard.get('price_rmse'))}, "
                f"cv_mae={format_console_metric(regression_selected_leaderboard.get('price_mae'))}, "
                f"cv_directional={format_console_metric(regression_selected_leaderboard.get('directional_accuracy'))}, "
                f"backtest_return={format_console_metric(regression_selected_backtest.get('total_return'))}, "
                f"backtest_sharpe={format_console_metric(regression_selected_backtest.get('sharpe'))}, "
                f"holdout_return={format_console_metric(regression_selected_holdout.get('total_return'))}, "
                f"holdout_sharpe={format_console_metric(regression_selected_holdout.get('sharpe'))}, "
                f"holdout_rmse={format_console_metric(regression_selected_holdout.get('price_rmse'))}"
            )
        classification_selected_leaderboard = classification_selected.get("leaderboard", {})
        classification_selected_backtest = classification_selected.get("backtest", {})
        classification_selected_holdout = classification_selected.get("recent_holdout", {})
        if classification_selected_leaderboard or classification_selected_backtest or classification_selected_holdout:
            print(
                "  Selected direction diagnostics: "
                f"cv_balanced_accuracy={format_console_metric(classification_selected_leaderboard.get('balanced_accuracy'))}, "
                f"cv_f1={format_console_metric(classification_selected_leaderboard.get('f1'))}, "
                f"cv_roc_auc={format_console_metric(classification_selected_leaderboard.get('roc_auc'))}, "
                f"backtest_return={format_console_metric(classification_selected_backtest.get('total_return'))}, "
                f"backtest_sharpe={format_console_metric(classification_selected_backtest.get('sharpe'))}, "
                f"holdout_return={format_console_metric(classification_selected_holdout.get('total_return'))}, "
                f"holdout_sharpe={format_console_metric(classification_selected_holdout.get('sharpe'))}, "
                f"holdout_balanced_accuracy={format_console_metric(classification_selected_holdout.get('balanced_accuracy'))}"
            )
        regime_selected_leaderboard = regime_selected.get("leaderboard", {})
        regime_selected_backtest = regime_selected.get("backtest", {})
        if regime_selected_leaderboard or regime_selected_backtest:
            print(
                "  Selected regime diagnostics: "
                f"cv_balanced_accuracy={format_console_metric(regime_selected_leaderboard.get('balanced_accuracy'))}, "
                f"cv_macro_f1={format_console_metric(regime_selected_leaderboard.get('macro_f1'))}, "
                f"cv_roc_auc={format_console_metric(regime_selected_leaderboard.get('roc_auc'))}, "
                f"backtest_return={format_console_metric(regime_selected_backtest.get('total_return'))}, "
                f"backtest_sharpe={format_console_metric(regime_selected_backtest.get('sharpe'))}"
            )
        reversal_selected_leaderboard = reversal_selected.get("leaderboard", {})
        reversal_selected_backtest = reversal_selected.get("backtest", {})
        if reversal_selected_leaderboard or reversal_selected_backtest:
            print(
                "  Selected reversal diagnostics: "
                f"cv_balanced_accuracy={format_console_metric(reversal_selected_leaderboard.get('balanced_accuracy'))}, "
                f"cv_macro_f1={format_console_metric(reversal_selected_leaderboard.get('macro_f1'))}, "
                f"cv_roc_auc={format_console_metric(reversal_selected_leaderboard.get('roc_auc'))}, "
                f"backtest_return={format_console_metric(reversal_selected_backtest.get('total_return'))}, "
                f"backtest_sharpe={format_console_metric(reversal_selected_backtest.get('sharpe'))}"
            )
        if feature_quality:
            print(
                "  Feature quality: "
                f"kept_standard={feature_quality.get('kept_standard_count', 0)}/{feature_quality.get('total_standard_count', 0)}, "
                f"kept_history={feature_quality.get('kept_history_count', 0)}/{feature_quality.get('total_history_count', 0)}, "
                f"dropped_feature_count={feature_quality.get('dropped_feature_count', 0)}"
            )
            weakest_kept = feature_quality.get("weakest_kept_features", [])
            if weakest_kept:
                weakest_kept_text = ", ".join(format_feature_record(row) for row in weakest_kept)
                print(f"  Weakest kept features: {weakest_kept_text}")
            dropped_feature_sample = feature_quality.get("dropped_feature_sample", [])
            if dropped_feature_sample:
                dropped_feature_text = ", ".join(format_feature_record(row) for row in dropped_feature_sample)
                print(f"  Dropped feature sample: {dropped_feature_text}")
        print("  Regression leaderboard")
        for row in horizon_payload["regression_leaderboard"]:
            print(
                "    - "
                f"{row['model']}: price_rmse={row['price_rmse']}, "
                f"price_mae={row['price_mae']}, directional_accuracy={row['directional_accuracy']}"
            )
        print("  Classification leaderboard")
        for row in horizon_payload["classification_leaderboard"]:
            print(
                "    - "
                f"{row['model']}: balanced_accuracy={row['balanced_accuracy']}, "
                f"f1={row['f1']}, roc_auc={row['roc_auc']}"
            )
        if horizon_payload.get("regime_leaderboard"):
            print("  Regime leaderboard")
            for row in horizon_payload["regime_leaderboard"]:
                print(
                    "    - "
                    f"{row['model']}: balanced_accuracy={row['balanced_accuracy']}, "
                    f"macro_f1={row['macro_f1']}, roc_auc={row['roc_auc']}"
                )
        if horizon_payload.get("reversal_leaderboard"):
            print("  Reversal leaderboard")
            for row in horizon_payload["reversal_leaderboard"]:
                print(
                    "    - "
                    f"{row['model']}: balanced_accuracy={row['balanced_accuracy']}, "
                    f"macro_f1={row['macro_f1']}, roc_auc={row['roc_auc']}"
                )
        print("  Regression backtest")
        for row in horizon_payload["regression_backtest"]:
            print(
                "    - "
                f"{row['model']}: total_return={row['total_return']}, "
                f"sharpe={row['sharpe']}, max_drawdown={row['max_drawdown']}, "
                f"signal_threshold={row.get('signal_threshold')}"
            )
        print("  Classification backtest")
        for row in horizon_payload["classification_backtest"]:
            print(
                "    - "
                f"{row['model']}: total_return={row['total_return']}, "
                f"sharpe={row['sharpe']}, max_drawdown={row['max_drawdown']}, "
                f"signal_threshold={row.get('signal_threshold')}"
            )
        if horizon_payload.get("regime_backtest"):
            print("  Regime backtest")
            for row in horizon_payload["regime_backtest"]:
                print(
                    "    - "
                    f"{row['model']}: total_return={row['total_return']}, "
                    f"sharpe={row['sharpe']}, max_drawdown={row['max_drawdown']}, "
                    f"signal_threshold={row.get('signal_threshold')}"
                )
        if horizon_payload.get("reversal_backtest"):
            print("  Reversal backtest")
            for row in horizon_payload["reversal_backtest"]:
                print(
                    "    - "
                    f"{row['model']}: total_return={row['total_return']}, "
                    f"sharpe={row['sharpe']}, max_drawdown={row['max_drawdown']}, "
                    f"signal_threshold={row.get('signal_threshold')}"
                )
        print("  Benchmarks")
        for row in horizon_payload["benchmark_backtest"]:
            print(
                "    - "
                f"{row['model']}: total_return={row['total_return']}, "
                f"sharpe={row['sharpe']}, max_drawdown={row['max_drawdown']}"
            )
        if recent_holdout_highlights:
            print("  Recent holdout highlights")
            for task_name in ("regression", "classification", "benchmark"):
                task_rows = recent_holdout_highlights.get(task_name, [])
                if not task_rows:
                    continue
                task_text = "; ".join(
                    (
                        f"{row['model']}: total_return={row.get('total_return')}, "
                        f"sharpe={row.get('sharpe')}, "
                        f"signal_threshold={row.get('signal_threshold')}"
                    )
                    for row in task_rows
                )
                print(f"    - {task_name}: {task_text}")
        regression_perm = horizon_payload["regression_explainability"]["permutation_importance_top"][:3]
        classification_perm = horizon_payload["classification_explainability"]["permutation_importance_top"][:3]
        regression_shap = horizon_payload["regression_explainability"]["shap_global_importance_top"][:3]
        classification_shap = horizon_payload["classification_explainability"]["shap_global_importance_top"][:3]
        if regression_perm:
            regression_perm_text = ", ".join(
                f"{row['feature']}({row['importance_mean']})" for row in regression_perm
            )
            print(f"  Regression permutation top: {regression_perm_text}")
        if regression_shap:
            regression_shap_text = ", ".join(
                f"{row['feature']}({row['mean_abs_shap']})" for row in regression_shap
            )
            print(f"  Regression SHAP top: {regression_shap_text}")
        if classification_perm:
            classification_perm_text = ", ".join(
                f"{row['feature']}({row['importance_mean']})" for row in classification_perm
            )
            print(f"  Direction permutation top: {classification_perm_text}")
        if classification_shap:
            classification_shap_text = ", ".join(
                f"{row['feature']}({row['mean_abs_shap']})" for row in classification_shap
            )
            print(f"  Direction SHAP top: {classification_shap_text}")
    print(f"- Note: {payload['note']}")


if __name__ == "__main__":
    main()
