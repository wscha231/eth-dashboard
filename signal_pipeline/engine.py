"""Bounded historical replay and cached live inference share the same model code."""
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .data import build_features, feature_columns, read_bars, utc
from .evaluate import prospective_report, report_replay
from .ledger import history, issue, settle
from .models import labels, predict_models, train_bundle
from .protocol import DEFAULT_HORIZONS, HORIZONS, PROTOCOL_HASH, SPEC, digest, runtime_hash, training_hash


def atomic_json(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False))
    tmp.replace(path)


def source_hash(bars, cutoff):
    subset = bars.loc[bars.close_time < cutoff].sort_values(["product", "open_time"])
    raw = "\n".join(subset["product"]+":"+subset.open_time.astype(str)+":"+subset.content_hash)
    return hashlib.sha256(raw.encode()).hexdigest()


def obtain_bundle(root, bars, features, outcomes, cutoff, horizon, *, allow_fit):
    root = Path(root); (root/"models").mkdir(parents=True, exist_ok=True)
    if not allow_fit:
        manifest_path=root/"active.json"
        if not manifest_path.exists():raise ValueError("active research release unavailable")
        manifest=json.loads(manifest_path.read_text()).get(str(horizon),{})
        name=manifest.get("file","")
        if not name or Path(name).name!=name:raise ValueError("invalid checkpoint manifest")
        path=root/"models"/name
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest.get("sha256"):
            raise ValueError("checkpoint integrity failure")
        bundle=joblib.load(path)
        if (bundle["protocol_hash"]!=PROTOCOL_HASH or bundle["training_hash"]!=training_hash()
                or bundle["fit_cutoff"]!=cutoff.isoformat()):
            raise ValueError("current monthly checkpoint unavailable; replay required")
        return bundle,False
    code = training_hash(); source = source_hash(bars, cutoff)
    key = digest({"protocol": PROTOCOL_HASH, "code": code, "source": source, "cutoff": str(cutoff), "h": horizon})
    path = root/"models"/f"h{horizon}_{cutoff.strftime('%Y-%m')}_{key[:16]}.joblib"
    if path.exists():
        return joblib.load(path), False
    if not allow_fit:
        raise ValueError("current monthly checkpoint unavailable; replay/retrain required")
    bundle = train_bundle(features, outcomes, cutoff, horizon)
    bundle.update(model_version=key, runtime_hash=runtime_hash(), training_hash=code, protocol_hash=PROTOCOL_HASH, source_snapshot=source)
    temporary = path.with_suffix(".tmp"); joblib.dump(bundle, temporary, compress=3); temporary.replace(path)
    return bundle, True


def replay(root, *, horizons=DEFAULT_HORIZONS, budget_seconds=1200, start=None, end=None):
    started = time.monotonic(); root = Path(root)
    bars = read_bars(root); features = build_features(bars)
    if features.empty:
        raise ValueError("both ETH-USD and BTC-USD history required")
    valid = features[feature_columns(features)].notna().all(axis=1)
    months = pd.date_range(features.index.min().floor("D").replace(day=1), features.index.max(), freq="MS")
    if start: months = months[months >= utc(start)]
    if end: months = months[months <= utc(end)]
    (root/"replay").mkdir(exist_ok=True)
    reports = {}; fits = 0; cached = 0; used_checkpoints=set()
    for h in horizons:
        outcomes = labels(bars, features, h); all_rows = []
        for cutoff in months:
            if time.monotonic()-started > budget_seconds:
                raise TimeoutError("replay budget exhausted; saved monthly checkpoints can resume")
            if (cutoff-features.index.min()).days < 300:
                continue
            next_month = cutoff+pd.offsets.MonthBegin(1)
            targets = features.index[valid & (features.index >= cutoff) & (features.index < next_month) &
                                      (features.index.hour == 0) & outcomes["return"].notna()]
            # Fit this month's checkpoint even when no current target has matured yet.
            latest_month = cutoff == months[-1]
            if not len(targets) and not latest_month:
                continue
            try:
                bundle, fitted = obtain_bundle(root, bars, features, outcomes, cutoff, h, allow_fit=True)
            except ValueError as exc:
                if "insufficient purged" in str(exc):
                    continue
                raise
            fits += fitted; cached += not fitted
            used_checkpoints.add(f"h{h}_{cutoff.strftime('%Y-%m')}_{bundle['model_version'][:16]}.joblib")
            if not len(targets):
                continue
            predictions = predict_models(bundle["model"], features.loc[targets])
            predictions["selected"] = predictions[bundle["choice"]]
            rows = []
            for name, pred in predictions.items():
                thresholds=bundle["alert_thresholds_by_model"][bundle["choice"] if name=="selected" else name]
                for j, slot in enumerate(targets):
                    y = outcomes.loc[slot]
                    rows.append({"slot": slot.isoformat(), "target_end": y.target_end.isoformat(), "horizon_hours": h,
                        "model": name, "selected_model": bundle["choice"], "model_version": bundle["model_version"],
                        "reference_price": float(features.loc[slot, "reference_price"]),
                        "actual_price": float(features.loc[slot, "reference_price"]*np.exp(y["return"])),
                        "return": float(y["return"]), "up": int(y.up), "down": int(y.down), "terminal": int(y.terminal),
                        "barrier": float(y.barrier), "p_down": float(pred["terminal"][j,0]),
                        "p_flat": float(pred["terminal"][j,1]), "p_up": float(pred["terminal"][j,2]),
                        "hit_up": float(pred["path"][j,0]), "hit_down": float(pred["path"][j,1]),
                        "q10": float(pred["quantiles"][j,0]), "q50": float(pred["quantiles"][j,1]), "q90": float(pred["quantiles"][j,2]),
                        "threshold_up": thresholds["up"], "threshold_down": thresholds["down"]})
            all_rows.extend(rows)
            print(f"replay h={h} month={cutoff:%Y-%m} origins={len(targets)} fit={fitted}", flush=True)
        reports[str(h)] = report_replay(all_rows, h)
        pd.DataFrame(all_rows).to_csv(root/"replay"/f"h{h}.csv.gz", index=False, compression={"method":"gzip", "mtime":0})
    payload = {"schema_version": 1, "protocol": SPEC, "protocol_hash": PROTOCOL_HASH, "runtime_hash": runtime_hash(),
               "generated_at": utc().isoformat(), "horizons": reports,
               "runtime": {"seconds": time.monotonic()-started, "new_monthly_fits": fits, "cached_months": cached},
               "claims": "Exploratory historical reconstruction. No prospective superiority established."}
    atomic_json(root/"replay.json", payload)
    # Prior protocol artifacts remain in the previous immutable research release.
    # Don't upload obsolete code families on every weekly refresh.
    for h in horizons:
        for p in (root/"models").glob(f"h{h}_*.joblib"):
            if p.name not in used_checkpoints:p.unlink()
    return payload


def forecast_record(features, slot, h, bundle):
    candidates=predict_models(bundle["model"], features.loc[[slot]])
    pred = candidates[bundle["choice"]]
    ref = float(features.loc[slot, "reference_price"])
    barrier = float(max(np.log1p(HORIZONS[h]), features.loc[slot,"sigma"]*np.sqrt(h)))
    return {"slot": slot.isoformat(), "input_cutoff": slot.isoformat(), "available_at": features.loc[slot,"available_at"].isoformat(),
            "window_start": (slot+pd.Timedelta(hours=1)).isoformat(), "target_end": (slot+pd.Timedelta(hours=h+1)).isoformat(),
            "horizon_seconds": h*3600, "instrument": "ETH-USD", "source": "coinbase_exchange", "reference_price": ref,
            "log_barrier": barrier, "upper_barrier_price": ref*np.exp(barrier), "lower_barrier_price": ref*np.exp(-barrier),
            "terminal_down_flat_up": pred["terminal"][0].tolist(), "hit_up": float(pred["path"][0,0]), "hit_down": float(pred["path"][0,1]),
            "price_quantiles": (ref*np.exp(pred["quantiles"][0])).tolist(), "selected_model": bundle["choice"],
            "alert_thresholds": bundle["alert_thresholds"], "model_version": bundle["model_version"],
            "training_target_end": bundle["training_target_end"], "validation_target_end": bundle["validation_target_end"],
            "source_snapshot": bundle["source_snapshot"], "runtime_hash": runtime_hash(),
            "training_runtime_hash":bundle["runtime_hash"]}


def baseline_record(features,slot,bundle):
    baseline=bundle['model']['baseline'];ref=float(features.loc[slot,'reference_price'])
    return {'terminal_down_flat_up':baseline['terminal'].tolist(), 'hit_up':float(baseline['path'][0]),
            'hit_down':float(baseline['path'][1]), 'price_quantiles':(ref*np.exp(baseline['quantiles'])).tolist(),
            'alert_thresholds':bundle['alert_thresholds_by_model']['climatology']}


def daily(root, *, horizons=DEFAULT_HORIZONS, now=None):
    started = time.monotonic(); root = Path(root); current = utc(now); slot = current.floor("h")
    bars = read_bars(root, as_of=current); features = build_features(bars)
    settle(root, bars, now=current)
    current_records = []; errors = []
    ready = (not features.empty and slot in features.index and
             features.loc[slot, feature_columns(features)].notna().all() and features.loc[slot, "available_at"] <= current)
    for h in horizons:
        if not ready:
            errors.append({"horizon": h, "reason": "latest complete input window unavailable"}); continue
        try:
            cutoff = slot.replace(day=1, hour=0)
            bundle, _ = obtain_bundle(root, bars, features, None, cutoff, h, allow_fit=False)
            record = forecast_record(features, slot, h, bundle)
            record['baseline']=baseline_record(features,slot,bundle)
            # Link every individual inference to the actual complete source snapshot.
            record["input_snapshot"] = source_hash(bars, slot+pd.Timedelta(hours=1))
            current_records.append(issue(root, record, now=current))
        except ValueError as exc:
            errors.append({"horizon": h, "reason": str(exc)})
    records = history(root)
    regime=None
    if ready:
        f=features.loc[slot]
        strength=float(f.eth_ret_24/max(f.eth_vol_720*np.sqrt(24),1e-8))
        regime={"trailing_24h_return":float(np.expm1(f.eth_ret_24)),
                "volatility_ratio_24h_30d":float(f.eth_vol_24/max(f.eth_vol_720,1e-8)),
                "trend_strength":strength,"state":"up" if strength>1 else "down" if strength< -1 else "range",
                "meaning":"past-only current-state detection, not a prediction of an unseen turning point"}
    replay_payload = json.loads((root/"replay.json").read_text()) if (root/"replay.json").exists() else None
    source = json.loads((root/"source_status.json").read_text()) if (root/"source_status.json").exists() else None
    payload = {"schema_version": 1, "product": "ETH event research beta", "generated_at": utc().isoformat(),
               "protocol_hash": PROTOCOL_HASH, "runtime_hash": runtime_hash(), "status": "ready" if len(current_records) == len(horizons) else "delayed",
               "expected_slot": slot.isoformat(), "current": current_records, "errors": errors, "source": source,
               "current_regime":regime,
               "recent_issued": [r for r in records if r.get("published_at")][-96:], "prospective": prospective_report(records),
               "replay_generated_at": replay_payload["generated_at"] if replay_payload else None,
               "runtime_seconds": time.monotonic()-started, "next_expected_update": (slot+pd.Timedelta(hours=1,minutes=8)).isoformat(),
               "service_status": "public_research_beta; paid service and predictive edge not established"}
    payload["release_id"] = digest(payload)
    atomic_json(root/"signals.json", payload)
    return payload
