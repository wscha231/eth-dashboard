"""Versioned, trusted-local model bundles and an explicit inference-only fit cache.

Joblib bundles execute Python on load. Only load bundles produced by this
repository's trusted training workflow, never uploads or PR artifacts.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import subprocess
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from pathlib import Path

import joblib
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[1]
_CACHE = ContextVar("final_model_cache", default=None)
SCHEMA = 1


def code_fingerprint() -> str:
    paths = [ROOT / "eth_price_forecast.py", *sorted((ROOT / "forecasting").glob("*.py"))]
    digest = hashlib.sha256()
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def frame_hash(frame) -> str:
    digest = hashlib.sha256(pd.util.hash_pandas_object(frame, index=True).values.tobytes())
    digest.update(json.dumps(list(frame.columns)).encode())
    return digest.hexdigest()


@contextmanager
def final_model_cache(mode: str, models: dict):
    if mode not in {"record", "predict"}:
        raise ValueError("Unknown final-model cache mode")
    token = _CACHE.set((mode, models))
    try:
        yield
    finally:
        _CACHE.reset(token)


def cached_final_fit(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapper(*args, **kwargs):
        context = _CACHE.get()
        if context is None:
            return function(*args, **kwargs)
        mode, models = context
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        columns = values["feature_columns"]
        # Include complete target/weight data as well as feature order. The same
        # model name alone cannot identify a fitted estimator.
        key = (function.__name__, values["model_name"], values["horizon"],
               values.get("target_column"), tuple(columns), frame_hash(values["dataset"]))
        if values.get("sample_weight") is not None:
            key += (hashlib.sha256(pd.util.hash_pandas_object(values["sample_weight"], index=True).values.tobytes()).hexdigest(),)
        if key in models:
            return models[key]
        if mode == "predict":
            raise RuntimeError("Inference attempted to fit an uncached model; refusing fallback training")
        models[key] = function(*args, **kwargs)
        return models[key]
    return wrapper


def save_bundle(bundle: dict, path: str | Path) -> dict:
    from forecasting.daily_data import iso_utc
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if set(bundle["horizons"]) != {7, 30}:
        raise ValueError("A published bundle requires both horizons")
    metadata = bundle.setdefault("metadata", {})
    metadata.update(schema=SCHEMA, code_fingerprint=code_fingerprint(),
                    python=platform.python_version(), sklearn=sklearn.__version__, created_at_utc=iso_utc(None))
    metadata["code_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    metadata["model_version"] = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:20]
    temp = path.with_suffix(".tmp")
    joblib.dump(bundle, temp, compress=3)
    sha = hashlib.sha256(temp.read_bytes()).hexdigest()
    os.replace(temp, path)
    manifest = {**metadata, "sha256": sha}
    path.with_suffix(".json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_bundle(path: str | Path) -> dict:
    path = Path(path)
    manifest = json.loads(path.with_suffix(".json").read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["sha256"]:
        raise ValueError("Bundle checksum mismatch")
    if manifest["code_fingerprint"] != code_fingerprint() or manifest["sklearn"] != sklearn.__version__:
        raise ValueError("Bundle implementation/dependency mismatch; train a compatible bundle first")
    if manifest["python"].split(".")[:2] != platform.python_version().split(".")[:2]:
        raise ValueError("Bundle Python major/minor mismatch")
    bundle = joblib.load(path)
    if manifest["schema"] != SCHEMA or set(bundle["horizons"]) != {7, 30}:
        raise ValueError("Incomplete or unsupported model bundle")
    return bundle
