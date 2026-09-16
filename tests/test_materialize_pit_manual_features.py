import numpy as np
import pandas as pd
import pytest

from scripts.materialize_pit_manual_features import (
    first_daily_decision_at_or_after,
    materialize_pit_frame,
)


def _features():
    return pd.DataFrame(
        {
            "funding": [0.001, 0.002],
            "oi": [np.nan, 100.0],
        },
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC"),
    )


def test_daily_value_is_not_visible_on_event_day():
    features = _features()
    sidecar = pd.DataFrame(
        [
            {"date": "2026-01-01T00:00:00Z", "column": "funding", "available_at": "2026-01-02T00:00:00Z"},
            {"date": "2026-01-02T00:00:00Z", "column": "funding", "available_at": "2026-01-03T00:00:00Z"},
            {"date": "2026-01-02T00:00:00Z", "column": "oi", "available_at": "2026-01-03T00:00:00Z"},
        ]
    )

    output = materialize_pit_frame(features, sidecar)

    assert pd.Timestamp("2026-01-01", tz="UTC") not in output.index
    assert output.loc[pd.Timestamp("2026-01-02", tz="UTC"), "funding"] == pytest.approx(0.001)
    assert output.loc[pd.Timestamp("2026-01-03", tz="UTC"), "funding"] == pytest.approx(0.002)
    assert output.loc[pd.Timestamp("2026-01-03", tz="UTC"), "oi"] == pytest.approx(100.0)


def test_intraday_availability_waits_for_next_daily_boundary():
    assert first_daily_decision_at_or_after(
        pd.Timestamp("2026-01-03T12:34:00Z")
    ) == pd.Timestamp("2026-01-04T00:00:00Z")
    assert first_daily_decision_at_or_after(
        pd.Timestamp("2026-01-03T00:00:00Z")
    ) == pd.Timestamp("2026-01-03T00:00:00Z")


def test_missing_sidecar_for_non_null_cell_fails_closed():
    features = pd.DataFrame(
        {"funding": [0.001]},
        index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"),
    )
    sidecar = pd.DataFrame(columns=["date", "column", "available_at"])

    with pytest.raises(ValueError, match="missing availability"):
        materialize_pit_frame(features, sidecar)


def test_null_cell_does_not_require_sidecar_entry():
    features = pd.DataFrame(
        {"funding": [np.nan], "oi": [5.0]},
        index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"),
    )
    sidecar = pd.DataFrame(
        [
            {"date": "2026-01-01T00:00:00Z", "column": "oi", "available_at": "2026-01-02T00:00:00Z"},
        ]
    )

    output = materialize_pit_frame(features, sidecar)
    assert np.isnan(output.loc[pd.Timestamp("2026-01-02", tz="UTC"), "funding"])
    assert output.loc[pd.Timestamp("2026-01-02", tz="UTC"), "oi"] == pytest.approx(5.0)


def test_duplicate_sidecar_entry_fails_closed():
    features = pd.DataFrame(
        {"funding": [0.001]},
        index=pd.DatetimeIndex(["2026-01-01"], tz="UTC"),
    )
    sidecar = pd.DataFrame(
        [
            {"date": "2026-01-01T00:00:00Z", "column": "funding", "available_at": "2026-01-02T00:00:00Z"},
            {"date": "2026-01-01T00:00:00Z", "column": "funding", "available_at": "2026-01-02T01:00:00Z"},
        ]
    )

    with pytest.raises(ValueError, match="duplicate availability"):
        materialize_pit_frame(features, sidecar)


def test_availability_cannot_precede_event_date():
    features = pd.DataFrame(
        {"funding": [0.001]},
        index=pd.DatetimeIndex(["2026-01-02"], tz="UTC"),
    )
    sidecar = pd.DataFrame(
        [
            {"date": "2026-01-02T00:00:00Z", "column": "funding", "available_at": "2026-01-01T23:00:00Z"},
        ]
    )

    with pytest.raises(ValueError, match="precedes event date"):
        materialize_pit_frame(features, sidecar)


def test_same_decision_date_keeps_latest_event_for_column():
    features = pd.DataFrame(
        {"funding": [0.001, 0.002]},
        index=pd.DatetimeIndex(["2026-01-01", "2026-01-02"], tz="UTC"),
    )
    sidecar = pd.DataFrame(
        [
            {"date": "2026-01-01T00:00:00Z", "column": "funding", "available_at": "2026-01-03T01:00:00Z"},
            {"date": "2026-01-02T00:00:00Z", "column": "funding", "available_at": "2026-01-03T22:00:00Z"},
        ]
    )

    output = materialize_pit_frame(features, sidecar)
    assert list(output.index) == [pd.Timestamp("2026-01-04T00:00:00Z")]
    assert output.iloc[0]["funding"] == pytest.approx(0.002)
