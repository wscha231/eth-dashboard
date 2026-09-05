"""Stage timings and process high-water RSS; no model-selection side effects."""
from __future__ import annotations

import json
import resource
import time
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

RECORDS: list[dict] = []


@contextmanager
def stage(name: str):
    started = time.perf_counter()
    success = False
    try:
        yield
        success = True
    finally:
        RECORDS.append({"stage": name, "seconds": round(time.perf_counter()-started, 6),
                        "success": success, "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024})


def timed(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        with stage(function.__name__):
            return function(*args, **kwargs)
    return wrapper


def write_metrics(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"stages": RECORDS}, indent=2), encoding="utf-8")
