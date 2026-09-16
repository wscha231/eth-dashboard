from pathlib import Path

import pandas as pd

from data_lab.free_source_fallbacks import collect_fred_initial_partial
from scripts.collect_free_source_fallbacks import (
    cached_tail_start,
    long_tail_start,
    trim_incremental_warmup,
)


def test_cached_tail_start_uses_overlap_and_full_reconcile(tmp_path):
    path = tmp_path / "features.csv"
    pd.DataFrame(
        {"date": ["2026-01-01", "2026-03-01"], "x": [1, 2]}
    ).to_csv(path, index=False)
    requested = pd.Timestamp("2025-01-01", tz="UTC")
    incremental = cached_tail_start(path, requested, overlap_days=60, full_reconcile=False)
    full = cached_tail_start(path, requested, overlap_days=60, full_reconcile=True)
    assert incremental == pd.Timestamp("2025-12-31", tz="UTC")
    assert full == requested


def test_long_tail_start_uses_named_date_column(tmp_path):
    path = tmp_path / "long.csv"
    pd.DataFrame(
        {
            "series_id": ["A", "A"],
            "observation_date": ["2025-01-01", "2026-06-01"],
            "value": [1, 2],
        }
    ).to_csv(path, index=False)
    start = long_tail_start(
        path,
        "observation_date",
        pd.Timestamp("2017-01-01", tz="UTC"),
        overlap_days=180,
        full_reconcile=False,
    )
    assert start == pd.Timestamp("2025-12-03", tz="UTC")


def test_trim_incremental_warmup_preserves_only_recomputed_tail():
    idx = pd.date_range("2026-01-01", periods=70, freq="1D", tz="UTC")
    frame = pd.DataFrame({"x": range(70)}, index=idx)
    trimmed = trim_incremental_warmup(
        frame,
        pd.Timestamp("2026-01-01", tz="UTC"),
        pd.Timestamp("2025-01-01", tz="UTC"),
        full_reconcile=False,
        warmup_days=30,
    )
    assert trimmed.index.min() == pd.Timestamp("2026-01-31", tz="UTC")
    assert trimmed.index.max() == idx.max()


def test_fred_initial_partial_propagates_recent_realtime_start():
    seen = []

    def get_json(url, params=None, **kwargs):
        seen.append(params.copy())
        return {
            "observations": [
                {
                    "date": "2026-01-01",
                    "realtime_start": "2026-02-01",
                    "value": "10",
                }
            ]
        }

    frame, errors = collect_fred_initial_partial(
        "key",
        {"M2SL": "m2"},
        get_json,
        realtime_start="2025-12-01",
    )
    assert not frame.empty
    assert errors == {}
    assert seen[0]["realtime_start"] == "2025-12-01"
