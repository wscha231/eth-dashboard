import numpy as np
import pandas as pd

from research.model_lab.exogenous_family_screen import (
    FamilySpec,
    build_core,
    build_family_features,
    build_targets,
    load_family,
    select_nonoverlap,
)


def test_full_day_family_moves_to_next_utc_day(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text(
        "date,x_funding\n"
        "2026-01-01T00:00:00Z,1.0\n"
        "2026-01-02T00:00:00Z,2.0\n",
        encoding="utf-8",
    )
    spec = FamilySpec("x", ("x.csv",), 1, include_tokens=("funding",))
    raw, present, _ = load_family(tmp_path, spec)
    assert pd.Timestamp("2026-01-01T00:00:00Z") not in raw.index
    assert raw.loc[pd.Timestamp("2026-01-02T00:00:00Z"), "x_funding"] == 1.0
    assert bool(present.loc[pd.Timestamp("2026-01-02T00:00:00Z")])


def test_family_transforms_are_past_only_under_future_perturbation():
    index = pd.date_range("2026-01-01", periods=120, freq="D", tz="UTC")
    raw = pd.DataFrame({"x": np.arange(1.0, 121.0)}, index=index)
    before = build_family_features(raw, "f")
    changed = raw.copy()
    changed.loc[index[-10]:, "x"] *= 1000.0
    after = build_family_features(changed, "f")
    pd.testing.assert_frame_equal(before.iloc[:-10], after.iloc[:-10])


def test_core_features_do_not_look_forward():
    index = pd.date_range("2026-01-01", periods=150, freq="D", tz="UTC")
    master = pd.DataFrame(
        {
            "eth_close": np.exp(np.linspace(7.0, 8.0, len(index))),
            "btc_close": np.exp(np.linspace(10.0, 10.5, len(index))),
        },
        index=index,
    )
    before = build_core(master)
    changed = master.copy()
    changed.loc[index[-5]:, "eth_close"] *= 20
    after = build_core(changed)
    pd.testing.assert_frame_equal(before.iloc[:-5], after.iloc[:-5])


def test_target_starts_after_origin_and_variance_is_future_only():
    index = pd.date_range("2026-01-01", periods=80, freq="D", tz="UTC")
    close = np.exp(np.cumsum(np.full(len(index), 0.01)))
    master = pd.DataFrame({"eth_close": close}, index=index)
    target = build_targets(master, 7)
    origin = index[20]
    assert target.loc[origin, "return"] == pytest.approx(0.07)
    assert target.loc[origin, "realized_variance"] == pytest.approx(7 * 0.01**2)
    assert target.loc[origin, "target_end"] == origin + pd.Timedelta(days=7)


def test_nonoverlap_selection_respects_horizon():
    index = pd.date_range("2026-01-01", periods=20, freq="D", tz="UTC")
    chosen = select_nonoverlap(index, 7)
    assert list(chosen) == [index[0], index[7], index[14]]
