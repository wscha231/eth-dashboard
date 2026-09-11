"""Identical-origin comparisons; overlapping forecasts are not independent trials."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .segments import REGIMES, SEGMENT_VERSION, periods, segment_mask


def event_metrics(truth, probability, threshold):
    truth = np.asarray(truth, int); probability = np.asarray(probability)
    alert = probability > np.asarray(threshold)  # ties don't explode the false-alert budget
    tp = np.sum(alert & (truth == 1)); fp = np.sum(alert & (truth == 0))
    return {"brier": float(np.mean((probability-truth)**2)),
            "pr_auc": float(average_precision_score(truth, probability)) if truth.sum() else None,
            "prevalence": float(truth.mean()), "events": int(truth.sum()),
            "recall": float(tp/truth.sum()) if truth.sum() else None,
            "precision": float(tp/alert.sum()) if alert.sum() else None,
            "false_positive_rate": float(fp/(truth == 0).sum()) if (truth == 0).sum() else None,
            "alerts": int(alert.sum()), "false_alerts": int(fp)}


def metrics(rows):
    if not len(rows):
        return {"rows": 0}
    f = pd.DataFrame(rows)
    ret = f["return"].to_numpy(float)
    p = f[["p_down", "p_flat", "p_up"]].to_numpy()
    y = np.eye(3)[f.terminal.astype(int)]
    up = event_metrics(f.up, f.hit_up, f.threshold_up)
    down = event_metrics(f.down, f.hit_down, f.threshold_down)
    mae = float(np.mean(np.abs(np.expm1(ret)-np.expm1(f.q50))))
    baseline_mae = float(np.mean(np.abs(np.expm1(ret))))
    terminal = f.terminal.to_numpy(int)
    predicted = p.argmax(axis=1)
    supports = np.bincount(terminal, minlength=3)
    balanced = float(np.mean([(predicted[terminal == c] == c).mean() for c in range(3)])) if supports.min() else None
    return {"rows": len(f), "terminal_brier": float(((p-y)**2).sum(axis=1).mean()),
            "terminal_accuracy": float((predicted == terminal).mean()),
            "terminal_balanced_accuracy": balanced,
            "terminal_logloss": float(-np.log(np.clip(p[np.arange(len(f)), f.terminal.astype(int)], 1e-8, 1)).mean()),
            "event_brier": (up["brier"]+down["brier"])/2,
            "up": up, "down": down, "return_mae": mae, "no_change_mae": baseline_mae,
            "mae_skill": 1-mae/baseline_mae if baseline_mae else None,
            "coverage80": float(((ret >= f.q10) & (ret <= f.q90)).mean()),
            "mean_interval_width": float((np.expm1(f.q90)-np.expm1(f.q10)).mean()),
            "return_rmse_pp": float(np.sqrt(np.mean((np.expm1(f.q50)-np.expm1(ret))**2))*100),
            "return_mae_pp": mae*100,
            "price_mae_usd": float(np.mean(np.abs(f.reference_price*(np.exp(f.q50)-np.exp(ret))))) if 'reference_price' in f else None,
            "price_mape_pct": float(np.mean(np.abs(np.exp(f.q50-ret)-1))*100),
            "interval_score80_pp": float(np.mean(np.expm1(f.q90)-np.expm1(f.q10)
                +10*np.maximum(np.expm1(f.q10)-np.expm1(ret),0)
                +10*np.maximum(np.expm1(ret)-np.expm1(f.q90),0))*100),
            "confusion_actual_by_predicted": np.bincount(terminal*3+predicted,minlength=9).reshape(3,3).tolist()}


def paired_block_interval(selected, baseline, horizon, samples=500):
    a, b = pd.DataFrame(selected), pd.DataFrame(baseline)
    if len(a) < 60 or len(a) != len(b) or a.slot.tolist() != b.slot.tolist():
        return None
    def losses(f):
        return ((f.hit_up-f.up)**2+(f.hit_down-f.down)**2).to_numpy()/2
    delta = losses(a)-losses(b)
    # Calendar blocks (not adjacent rows across a data outage), at least a full target interval.
    slots = pd.to_datetime(a.slot, utc=True)
    width = max(7, int(np.ceil((horizon+1)/24)))
    groups = (slots.astype("int64")//(86400*10**9*width)).to_numpy()
    blocks = [delta[groups == g] for g in np.unique(groups)]
    if len(blocks) < 10:
        return None
    rng = np.random.default_rng(1729)
    estimates = [np.concatenate([blocks[i] for i in rng.integers(len(blocks), size=len(blocks))]).mean() for _ in range(samples)]
    return {"difference_selected_minus_baseline": float(delta.mean()),
            "lower95": float(np.quantile(estimates, .025)), "upper95": float(np.quantile(estimates, .975)),
            "blocks": len(blocks), "block_days": width,
            "interpretation": "negative favors selected; exploratory interval, not a promotion certificate"}


def comparison(f, horizon):
    if f.empty:
        return {"common_origins": 0, "models": [], "nonoverlapping_selected": {"rows": 0},
                "nonoverlapping_baseline": {"rows": 0}, "paired_event_brier": None, "selected_model_counts": {}}
    models = [{"model": name, **metrics(group.to_dict("records"))} for name, group in f.groupby("model")]
    selected = f[f.model.eq("selected")].sort_values("slot")
    baseline = f[f.model.eq("climatology")].sort_values("slot")
    independent=[];previous_end=None
    for row in selected.to_dict('records'):
        if previous_end is None or pd.Timestamp(row['slot'])>=previous_end:
            independent.append(row);previous_end=pd.Timestamp(row['target_end'])
    slots = {r['slot'] for r in independent}
    return {"models": models, "common_origins": len(selected),
            "first_origin": selected.slot.min(), "last_origin": selected.slot.max(),
            "nonoverlapping_selected": metrics(independent),
            "nonoverlapping_baseline": metrics(baseline[baseline.slot.isin(slots)].to_dict("records")),
            "paired_event_brier": paired_block_interval(selected.to_dict("records"), baseline.to_dict("records"), horizon),
            "selected_model_counts": {str(k): int(v) for k, v in selected.selected_model.value_counts().items()}}


def report_replay(rows, horizon, *, as_of=None):
    if not rows:
        return {"horizon_hours": horizon, "status": "insufficient_history", "models": []}
    f = pd.DataFrame(rows)
    if f.duplicated(["model", "slot"]).any():raise ValueError("duplicate model/origin in replay")
    if not {"selected", "climatology"}.issubset(set(f.model)):
        raise ValueError("paired selected and baseline predictions required")
    expected_models = set(f.model)
    as_of = pd.Timestamp(as_of) if as_of is not None else pd.to_datetime(f.target_end, utc=True).max()
    f = f[pd.to_datetime(f.target_end, utc=True) <= as_of]
    common = set.intersection(*(set(f.loc[f.model.eq(name), "slot"]) for name in expected_models))
    if not common:
        return {"horizon_hours": horizon, "status": "insufficient_history", "models": []}
    f = f[f.slot.isin(common)]
    truth_columns = [c for c in ("target_end", "return", "up", "down", "terminal", "trend_state", "volatility_state") if c in f]
    if (f.groupby("slot")[truth_columns].nunique() > 1).any().any():
        raise ValueError("paired origins have inconsistent truth or market state")
    overall = comparison(f, horizon)
    selected = f[f.model.eq("selected")].sort_values("slot")
    years = [{"year": int(year), **metrics(g.to_dict("records"))} for year, g in selected.groupby(pd.to_datetime(selected.slot, utc=True).dt.year)]
    slices = None
    if {"trend_state", "volatility_state"}.issubset(f.columns):
        definitions = periods(min(common), as_of)
        results = {}
        for period in definitions:
            for regime in REGIMES:
                key = period["id"] + "|" + regime["id"]
                results[key] = overall if key == "all|all" else comparison(f[segment_mask(f, period, regime["id"])], horizon)
        slices = {"version": SEGMENT_VERSION, "as_of": as_of.isoformat(),
                  "periods": definitions, "regimes": REGIMES, "results": results,
                  "definitions": {"trend": "prior 24h log return / (prior 720h hourly volatility * sqrt(24)); up > 1, down < -1, otherwise range",
                                  "volatility": "prior 24h / prior 720h hourly volatility; high >= 1.5",
                                  "membership": "state at prediction origin; never future realized return",
                                  "recent_windows": "calendar origin dates ending on the shared data as-of day; immature outcomes excluded",
                                  "inference": "exploratory overlapping slices; no model selection or promotion on these results"}}
    # Full probabilities, target boundaries and model provenance enable identical filtered scoring.
    point_columns = list(selected.columns)
    points = selected[point_columns].to_dict("records")
    baselines = {r['slot']: r for r in f[f.model.eq('climatology')].to_dict('records')}
    for point in points:
        point['baseline'] = {k: baselines[point['slot']][k] for k in
            ('p_down','p_flat','p_up','hit_up','hit_down','q10','q50','q90','threshold_up','threshold_down')}
    return {"horizon_hours": horizon, "status": "retrospective_research",
            **overall,
            "eligible_calendar_coverage":len(common)/((pd.Timestamp(max(common))-pd.Timestamp(min(common))).days+1),
            "yearly_selected": years,
            "segments": slices,
            "points": points}


def prospective_report(records):
    # Failed/delayed publication is not a timely live forecast. Keep it in the ledger,
    # but don't invent an earlier public release time to make a retrospective hit.
    records=[r for r in records if r.get("published_at") and
             pd.Timestamp(r["published_at"]) < pd.Timestamp(r["window_start"])]
    report = {}
    for horizon in sorted({r["horizon_seconds"]//3600 for r in records}):
        issued = [r for r in records if r["horizon_seconds"] == horizon*3600]
        rows = []; paired_selected=[];baseline_rows=[]; independent_end=None;independent_count=0
        for r in issued:
            if r["outcome"] is None:
                continue
            q = np.log(np.asarray(r["price_quantiles"])/r["reference_price"])
            row={**r["outcome"], "slot":r['slot'],"target_end":r['target_end'],"p_down": r["terminal_down_flat_up"][0],
                         "p_flat": r["terminal_down_flat_up"][1], "p_up": r["terminal_down_flat_up"][2],
                         "hit_up": r["hit_up"], "hit_down": r["hit_down"],
                         "threshold_up": r["alert_thresholds"]["up"], "threshold_down": r["alert_thresholds"]["down"],
                         "q10": q[0], "q50": q[1], "q90": q[2]}
            rows.append(row)
            if r.get('baseline'):
                b=r['baseline'];bq=np.log(np.asarray(b['price_quantiles'])/r['reference_price'])
                baseline_rows.append({**row,'p_down':b['terminal_down_flat_up'][0],'p_flat':b['terminal_down_flat_up'][1],
                                      'p_up':b['terminal_down_flat_up'][2],'hit_up':b['hit_up'],'hit_down':b['hit_down'],
                                      'threshold_up':b.get('alert_thresholds',{}).get('up',1.),'threshold_down':b.get('alert_thresholds',{}).get('down',1.),
                                      'q10':bq[0],'q50':bq[1],'q90':bq[2]})
                paired_selected.append(row)
                if independent_end is None or pd.Timestamp(r['slot'])>=independent_end:
                    independent_count+=1;independent_end=pd.Timestamp(r['target_end'])
        chosen_metrics=metrics(paired_selected);reference_metrics=metrics(baseline_rows)
        degraded=bool(independent_count>=20 and chosen_metrics.get('event_brier',0)>1.1*reference_metrics.get('event_brier',float('inf')))
        report[str(horizon)] = {"issued": len(issued), "resolved": len(rows), "pending": len(issued)-len(rows),
                                "metrics": metrics(rows), "baseline": reference_metrics,
                                "paired_selected":chosen_metrics,"nonoverlap_resolved":independent_count,
                                "performance_watch":'review_required' if degraded else 'insufficient_evidence' if independent_count<20 else 'within_watch_threshold',
                                "promotion": "research_only"}
    return report
