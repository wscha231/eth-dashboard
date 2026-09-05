"""Bounded cache for identical fold-local feature selection inputs."""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import inspect

from forecasting.model_bundle import frame_hash

_FOLD_CACHE = ContextVar("fold_feature_cache", default=None)


@contextmanager
def fold_feature_cache():
    token = _FOLD_CACHE.set({})
    try:
        yield
    finally:
        _FOLD_CACHE.reset(token)


def cached_fold_selection(function):
    signature = inspect.signature(function)
    @wraps(function)
    def wrapper(*args, **kwargs):
        cache = _FOLD_CACHE.get()
        if cache is None:
            return function(*args, **kwargs)
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        columns = values["candidate_feature_columns"]
        target = values["target_column"]
        train = values["dataset"].iloc[values["train_positions"]]
        used = list(dict.fromkeys([c for c in columns if c in train] + [target]))
        key = (frame_hash(train[used]), target, values["horizon"], values["min_feature_coverage"], tuple(columns))
        if key not in cache:
            cache[key] = tuple(function(*args, **kwargs))
        return list(cache[key])
    return wrapper
