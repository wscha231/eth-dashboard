"""Research candidate: calibrate alerts on the exact frozen prediction model.

The production bundle remains a comparator until matched retrospective and
prospective evidence justify a separately versioned checkpoint rollout.
"""
import numpy as np
import pandas as pd

from signal_pipeline.data import feature_columns, utc
from signal_pipeline.models import brier, fit_models, predict_models

VERSION = "frozen_model_alerts_v1"


def train_calibrated_bundle(features, outcomes, cutoff, horizon):
    """Selection, final fit, calibration, then test; no fit after calibration."""
    cutoff = utc(cutoff)
    calibration_days = 90 if horizon >= 336 else 30
    validation_days = 365 if horizon >= 336 else 90
    train_days = 1095 if horizon >= 336 else 730
    calibration_start = cutoff - pd.Timedelta(days=calibration_days)
    selection_start = calibration_start - pd.Timedelta(days=validation_days)
    gap = pd.Timedelta(hours=1)
    ready = (features[feature_columns(features)].notna().all(axis=1)
             & outcomes[["return", "up", "down", "terminal"]].notna().all(axis=1))
    ready &= features.index.hour % 6 == 0

    def indices(start, end):
        return features.index[ready & (features.index >= start)
                              & (outcomes.target_end < end - gap)]

    training = indices(selection_start - pd.Timedelta(days=train_days), selection_start)
    selection = indices(selection_start, calibration_start)
    final_fit = indices(calibration_start - pd.Timedelta(days=train_days), calibration_start)
    calibration = indices(calibration_start, cutoff)
    if len(training) < 700 or len(selection) < 120 or len(final_fit) < 700 or len(calibration) < 60:
        raise ValueError("insufficient purged training/selection/calibration history")

    inner = fit_models(features, outcomes, training)
    predictions = predict_models(inner, features.loc[selection])
    scores = {name: brier(pred, outcomes.loc[selection]) for name, pred in predictions.items()}
    choice = min(scores, key=lambda name: (scores[name], list(scores).index(name)))
    fitted = fit_models(features, outcomes, final_fit)
    # Both thresholds and future inference use this same object, without refit.
    predictions = predict_models(fitted, features.loc[calibration])
    all_thresholds = {}
    negative_counts = {}
    for name, prediction in predictions.items():
        thresholds = {}
        counts = {}
        for j, event in enumerate(("up", "down")):
            negatives = prediction["path"][outcomes.loc[calibration, event].eq(0).to_numpy(), j]
            counts[event] = len(negatives)
            thresholds[event] = (float(np.quantile(negatives, .95, method="higher"))
                                 if len(negatives) >= 40 else 1.)
        all_thresholds[name] = thresholds
        negative_counts[name] = counts
    return {"model": fitted, "choice": choice, "validation_scores": scores,
            "validation_rows": len(selection), "training_rows": len(final_fit),
            "training_target_end": outcomes.loc[final_fit, "target_end"].max().isoformat(),
            "validation_target_end": outcomes.loc[selection, "target_end"].max().isoformat(),
            "fit_cutoff": cutoff.isoformat(), "alert_thresholds": all_thresholds[choice],
            "alert_thresholds_by_model": all_thresholds, "calibration_version": VERSION,
            "selection_start": selection_start.isoformat(),
            "calibration_start": calibration_start.isoformat(),
            "calibration_target_end": outcomes.loc[calibration, "target_end"].max().isoformat(),
            "calibration_rows": len(calibration), "calibration_negatives": negative_counts,
            "claim": "empirical 5% calibration FPR target; future FPR is not guaranteed"}


def historical_derivative_features(index, funding, dvol, *, reconstruction=False):
    """Join closed provider intervals with a one-hour research latency assumption.

    Original receipt timestamps stay in the input snapshots. They are NOT moved
    into the past. These features are ineligible as historical live evidence.
    Missing hours/days remain missing; no synthetic OI or liquidations are made.
    """
    if not reconstruction:
        raise ValueError("historical reconstruction must be explicitly acknowledged")
    for frame, key in ((funding, "event_time"), (dvol, "event_end")):
        if frame[key].duplicated().any() or frame.received_at.isna().any():
            raise ValueError("unique intervals and actual receipt timestamps required")
    index = pd.DatetimeIndex(index)
    if index.tz is None or index.has_duplicates or not index.is_monotonic_increasing:
        raise ValueError("sorted unique timezone-aware origin grid required")
    rates = funding.set_index("event_time").interest_1h.sort_index()
    grid = pd.date_range(min(index.min(), rates.index.min()), index.max(), freq="h")
    rates = rates.reindex(grid)
    out = pd.DataFrame(index=index)
    # Each rate is for the preceding disjoint hour; never sum overlapping 8h rates.
    out["funding_sum_6h"] = rates.rolling(6, min_periods=6).sum().shift(1).reindex(index)
    out["funding_sum_24h"] = rates.rolling(24, min_periods=24).sum().shift(1).reindex(index)
    daily = dvol.set_index("event_end").dvol_close_pct.sort_index()
    daily = daily.reindex(pd.date_range(daily.index.min(), daily.index.max(), freq="D"))
    daily_features = pd.DataFrame({"dvol_level": daily, "dvol_change_7d": daily.diff(7)})
    daily_features.index += pd.Timedelta(hours=1)
    # A missing daily row remains explicit, and every value expires after 24h.
    joined = pd.merge_asof(pd.DataFrame({"origin": index}),
                          daily_features.rename_axis("eligible_at").reset_index(),
                          left_on="origin", right_on="eligible_at", direction="backward",
                          tolerance=pd.Timedelta(hours=23))
    for col in daily_features:
        out[col] = joined[col].to_numpy()
    out.attrs["availability"] = "historical reconstruction; event end + 1h assumed, not observed vintage"
    return out.replace([np.inf, -np.inf], np.nan)


def long_cash_diagnostic(rows, bars, start, end, cost_bps):
    """Predeclared nonoverlapping long/cash diagnostic, next-hour execution.

    Cost is an assumed per-side fee+spread+slippage scenario. Mark-to-market
    drawdown uses hourly closes; it cannot capture intrabar extremes.
    """
    start, end = utc(start), utc(end)
    cost = cost_bps / 10000
    if not 0 <= cost < 1 or end <= start:
        raise ValueError("invalid cost or execution interval")
    eth = bars.loc[bars["product"].eq("ETH-USD")].set_index("open_time").sort_index()
    grid = pd.date_range(start, end - pd.Timedelta(hours=1), freq="h")
    prices = eth.reindex(grid)
    if prices[["open", "close"]].isna().any().any():
        return {"status": "incomplete_execution_prices", "cost_bps_per_side": cost_bps,
                "missing_hours": int(prices[["open", "close"]].isna().any(axis=1).sum())}
    signals = {}
    for row in sorted(rows, key=lambda r: r["slot"]):
        entry = utc(row["slot"]) + pd.Timedelta(hours=1)
        exit_time = utc(row["target_end"])
        if entry >= start and exit_time <= end:
            if entry in signals:
                raise ValueError("duplicate execution origin")
            signals[entry] = row
    cash = 1.; units = 0.; exit_time = None; trades = 0; invested_hours = 0
    equity = [cash]
    for t, price in prices.iterrows():
        signal = signals.get(t)
        if units == 0 and signal and signal["hit_up"] > signal["threshold_up"] and signal["hit_down"] <= signal["threshold_down"]:
            units = cash / (float(price.open) * (1 + cost)); cash = 0.
            exit_time = utc(signal["target_end"]); trades += 1
        if units:
            invested_hours += 1
            if t + pd.Timedelta(hours=1) == exit_time:
                cash = units * float(price.close) * (1 - cost); units = 0.
        equity.append(cash + units * float(price.close))
    assert units == 0, "all diagnostic positions must mature"
    curve = np.asarray(equity)
    hold = prices.close.to_numpy() / (float(prices.open.iloc[0]) * (1 + cost))
    hold[-1] *= 1 - cost
    hold = np.r_[1., hold]
    return {"status": "measured_retrospective", "cost_bps_per_side": cost_bps,
            "trades": trades, "invested_hours": invested_hours, "calendar_hours": len(grid),
            "total_return": float(curve[-1] - 1),
            "max_drawdown_hourly": float(np.min(curve / np.maximum.accumulate(curve) - 1)),
            "buy_hold_total_return": float(hold[-1] - 1),
            "buy_hold_max_drawdown_hourly": float(np.min(hold / np.maximum.accumulate(hold) - 1)),
            "start": start.isoformat(), "end": end.isoformat()}
