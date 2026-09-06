"""Identical-origin comparisons; overlapping forecasts are not independent trials."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


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
    return {"rows": len(f), "terminal_brier": float(((p-y)**2).sum(axis=1).mean()),
            "terminal_logloss": float(-np.log(np.clip(p[np.arange(len(f)), f.terminal.astype(int)], 1e-8, 1)).mean()),
            "event_brier": (up["brier"]+down["brier"])/2,
            "up": up, "down": down, "return_mae": mae, "no_change_mae": baseline_mae,
            "mae_skill": 1-mae/baseline_mae if baseline_mae else None,
            "coverage80": float(((ret >= f.q10) & (ret <= f.q90)).mean()),
            "mean_interval_width": float((np.expm1(f.q90)-np.expm1(f.q10)).mean())}


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


def report_replay(rows, horizon):
    if not rows:
        return {"horizon_hours": horizon, "status": "insufficient_history", "models": []}
    f = pd.DataFrame(rows)
    common = set.intersection(*(set(g.slot) for _, g in f.groupby("model")))
    f = f[f.slot.isin(common)]
    models = [{"model": name, **metrics(group.to_dict("records"))} for name, group in f.groupby("model")]
    selected = f[f.model.eq("selected")].sort_values("slot")
    baseline = f[f.model.eq("climatology")].sort_values("slot")
    years = [{"year": int(year), **metrics(g.to_dict("records"))} for year, g in selected.groupby(pd.to_datetime(selected.slot, utc=True).dt.year)]
    return {"horizon_hours": horizon, "status": "retrospective_research", "models": models,
            "first_origin": min(common), "last_origin": max(common), "common_origins": len(common),
            "yearly_selected": years,
            "paired_event_brier": paired_block_interval(selected.to_dict("records"), baseline.to_dict("records"), horizon),
            "points": selected.to_dict("records")}


def prospective_report(records):
    report = {}
    for horizon in sorted({r["horizon_seconds"]//3600 for r in records}):
        issued = [r for r in records if r["horizon_seconds"] == horizon*3600]
        rows = []
        for r in issued:
            if r["outcome"] is None:
                continue
            q = np.log(np.asarray(r["price_quantiles"])/r["reference_price"])
            rows.append({**r["outcome"], "p_down": r["terminal_down_flat_up"][0],
                         "p_flat": r["terminal_down_flat_up"][1], "p_up": r["terminal_down_flat_up"][2],
                         "hit_up": r["hit_up"], "hit_down": r["hit_down"],
                         "threshold_up": r["alert_thresholds"]["up"], "threshold_down": r["alert_thresholds"]["down"],
                         "q10": q[0], "q50": q[1], "q90": q[2]})
        report[str(horizon)] = {"issued": len(issued), "resolved": len(rows), "pending": len(issued)-len(rows),
                                "metrics": metrics(rows), "promotion": "research_only"}
    return report
