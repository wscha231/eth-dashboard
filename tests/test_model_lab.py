"""MODEL contract tests; no collection, model fitting, API calls or production writes."""
import copy
import inspect
import json
from pathlib import Path
import sqlite3

import numpy as np
import pandas as pd
import pytest

from model_lab.contracts import (HORIZONS, RELEASE_FIELDS, PinnedRelease, canonical, consume_features,
                                digest_bytes, file_hash, load_bars, nonoverlap_count, purged_indices,
                                safe_relative, target_frame, utc, validate_prediction)
from model_lab.reproduce import score
from signal_pipeline.data import build_features
from signal_pipeline.models import labels


@pytest.fixture
def release_data():
    return {
        "release_id": "fixture-v1", "schema_version": 1,
        "dataset_manifest_id": "fixture-manifest-v1", "dataset_manifest_hash": "a" * 64,
        "feature_set_id": "ohlcv", "feature_set_version": "v1", "code_sha": "b" * 40,
        "data_cutoff_utc": "2026-01-10T00:00:00Z", "generated_at_utc": "2026-01-11T00:00:00Z",
        "source_versions": {"feature_builder_sha256": digest_bytes(inspect.getsource(build_features).encode())},
        "coverage_by_feature": {"ohlcv": {"status": "test"}}, "quality_status": "partial",
        "vintage_mode": "reconstructed", "license_status": "unreviewed",
        "partition_locations": {"observations.db": "observations.db"},
        "checksums": {"observations.db": "c" * 64}, "exclusions": [], "breaking_changes": []}


def pin(tmp_path, data):
    p = tmp_path / "release.json"; p.write_bytes(canonical(data))
    return PinnedRelease.load(p, file_hash(p))


@pytest.fixture(scope="module")
def market():
    n = 2600; idx = pd.date_range("2025-01-01T00:00:00Z", periods=n, freq="h")
    rng = np.random.default_rng(31); frames = []
    for product, ref in (("ETH-USD", 3000), ("BTC-USD", 60000)):
        close = ref * np.exp(np.cumsum(rng.normal(0, .004, n)))
        frames.append(pd.DataFrame({"product": product, "open_time": idx - pd.Timedelta(hours=1), "close_time": idx,
                                    "observed_at": idx + pd.Timedelta(seconds=2), "open": close,
                                    "high": close * 1.02, "low": close * .98, "close": close,
                                    "volume": rng.uniform(10, 1000, n)}))
    bars = pd.concat(frames, ignore_index=True)
    return bars, build_features(bars)


@pytest.mark.parametrize("field", sorted(RELEASE_FIELDS))
def test_release_requires_all_fields(tmp_path, release_data, field):
    del release_data[field]
    with pytest.raises(ValueError): pin(tmp_path, release_data)


def test_release_checksum_and_frozen_pointer(tmp_path, release_data):
    p = tmp_path / "release.json"; p.write_bytes(canonical(release_data)); h = file_hash(p)
    frozen = PinnedRelease.load(p, h)
    release_data["release_id"] = "replacement-v2"; p.write_bytes(canonical(release_data))
    assert frozen.metadata["release_id"] == "fixture-v1"
    d = frozen.metadata; d["release_id"] = "mutated"
    assert frozen.metadata["release_id"] == "fixture-v1"
    with pytest.raises(ValueError): PinnedRelease.load(p, h)


@pytest.mark.parametrize("path", ["../x", "/tmp/x", "a/../../b", "a\\b", "C:/x", "", ".", "a//b"])
def test_unsafe_paths(path):
    with pytest.raises(ValueError): safe_relative(path)


@pytest.mark.parametrize("time", ["2026-01-01", pd.NaT, "not a date"])
def test_naive_or_invalid_time(time):
    with pytest.raises((ValueError, TypeError)): utc(time)


def test_offset_equivalence():
    assert utc("2026-01-01T09:00:00+09:00") == utc("2026-01-01T00:00:00Z")


def test_reconstructed_release_cannot_certify_live(tmp_path, release_data):
    r = pin(tmp_path, release_data); r.authorize("retrospective")
    with pytest.raises(ValueError): r.authorize("prospective")
    with pytest.raises(ValueError): r.authorize("unknown")


def test_pinned_partition_rejects_change(tmp_path, release_data):
    p = tmp_path / "observations.db"; p.write_bytes(b"original")
    release_data["checksums"]["observations.db"] = file_hash(p)
    r = pin(tmp_path, release_data); assert r.path(tmp_path, "observations.db") == p
    p.write_bytes(b"changed")
    with pytest.raises(ValueError): r.path(tmp_path, "observations.db")


def test_feature_formula_version_is_checked(tmp_path, release_data, market):
    r = pin(tmp_path, release_data)
    assert consume_features(market[0], r).shape[1] == 35
    release_data["source_versions"]["feature_builder_sha256"] = "0" * 64
    with pytest.raises(ValueError): consume_features(market[0], pin(tmp_path, release_data))


@pytest.mark.parametrize("h", HORIZONS)
def test_legacy_targets_are_preserved(market, h):
    bars, features = market
    a = target_frame(bars, features, h); b = labels(bars, features, h)
    for key in ("return", "barrier", "up", "down", "terminal"):
        np.testing.assert_allclose(a[key], b[key], equal_nan=True, atol=1e-12)
    assert (a.target_end == b.target_end).all()
    assert (a.target_end - a.window_start == pd.Timedelta(hours=h)).all()


@pytest.mark.parametrize("h", HORIZONS)
def test_missing_middle_bar_does_not_become_negative(market, h):
    bars, features = market; bars = bars.copy()
    t = features.index[1000]
    mask = bars["product"].eq("ETH-USD") & bars.close_time.eq(t + pd.Timedelta(hours=3))
    bars = bars.loc[~mask]
    result = target_frame(bars, features, h).loc[t]
    assert pd.isna(result["return"]) and pd.isna(result["up"]) and pd.isna(result["terminal"])


@pytest.mark.parametrize("h", HORIZONS)
def test_unmatured_outcomes_stay_missing(market, h):
    a = target_frame(*market, h)
    assert a["return"].iloc[-(h + 1):].isna().all()
    assert a.positive.iloc[-(h + 1):].isna().all()


@pytest.mark.parametrize("h", HORIZONS)
def test_variance_uses_h_not_h_plus_one_returns(market, h):
    bars, features = market; t = features.index[1000]
    a = target_frame(bars, features, h).loc[t]
    closes = bars.loc[bars["product"].eq("ETH-USD")].set_index("close_time").close
    window = closes.loc[t + pd.Timedelta(hours=1):t + pd.Timedelta(hours=h + 1)]
    assert len(window) == h + 1
    assert np.isclose(a.realized_variance_total, np.log(window).diff().pow(2).sum())
    assert np.isclose(a["return"], np.log(window.iloc[-1] / features.loc[t, "reference_price"]))


@pytest.mark.parametrize("h", HORIZONS)
def test_purge_excludes_boundary_crossing_labels(market, h):
    bars, features = market; target = target_frame(bars, features, h)
    boundary = features.index[1800]
    ix = purged_indices(features, target, features.index[700], boundary, stride_hours=6)
    assert len(ix) > 0
    assert (target.loc[ix, "target_end"] < boundary - pd.Timedelta(hours=1)).all()
    assert (ix.hour % 6 == 0).all()


def test_future_perturbation_does_not_change_past_features(market):
    bars, features = market; mutated = bars.copy(); t = features.index[1400]
    future = mutated.close_time > t
    mutated.loc[future, ["open", "high", "low", "close", "volume"]] *= 2
    changed = build_features(mutated)
    pd.testing.assert_frame_equal(features.loc[:t], changed.loc[:t])


def test_zero_and_missing_are_different(market):
    bars, features = market; bars = bars.copy(); features = features.copy()
    t = features.index[1000]; h = 6
    bars.loc[bars["product"].eq("ETH-USD") & bars.close_time.eq(t + pd.Timedelta(hours=h + 1)), "close"] = features.loc[t, "reference_price"]
    a = target_frame(bars, features, h).loc[t]
    assert a["return"] == 0 and a.positive == 0 and a.negative == 0


@pytest.fixture
def record():
    return {"release_id": "r1", "feature_set_version": "f1", "model_version": "m1",
            "target_spec_id": "model_lab_hourly_endpoint_v1", "validation_target_end": "2025-12-31T23:00:00Z", "horizon_hours": 6, "input_cutoff": "2026-01-01T00:00:00Z", "issued_at": "2026-01-01T00:05:00Z",
            "available_at": "2026-01-01T00:01:00Z", "training_target_end": "2025-12-31T23:00:00Z",
            "window_start": "2026-01-01T01:00:00Z", "target_end": "2026-01-01T07:00:00Z",
            "reference_price": 3000., "price_quantiles": [2500., 3100., 3700.],
            "direction_positive_zero_negative": [.6, 0., .4], "path_hit_up_down": [.8, .9],
            "median_simple_return": 3100. / 3000. - 1, "variance_total": .04}


def test_path_events_can_both_happen(record):
    validate_prediction(record)  # sum of path probabilities = 1.7, deliberately allowed.


@pytest.mark.parametrize("field,value", [
    ("price_quantiles", [3000, 2000, 4000]), ("price_quantiles", [-1, 3000, 4000]),
    ("price_quantiles", [2000, np.nan, 4000]), ("path_hit_up_down", [1.01, .8]),
    ("direction_positive_zero_negative", [.9, .9, .1]),
    ("direction_positive_zero_negative", [.1, .0, .9]),
    ("median_simple_return", .5), ("variance_total", -1), ("reference_price", 0),
    ("issued_at", "2026-01-01T00:55:00Z"), ("window_start", "2026-01-01T00:00:00Z"),
    ("target_end", "2026-01-01T06:00:00Z"), ("available_at", "2026-01-01T00:06:00Z"),
    ("training_target_end", "2026-01-01T00:00:00Z"), ("horizon_hours", 7)])
def test_invalid_output_is_blocked(record, field, value):
    record[field] = value
    with pytest.raises(ValueError): validate_prediction(record)


def test_scores_have_correct_units_and_zero_baseline():
    f = pd.DataFrame({"return": np.log([1.1, .9]), "q10": [-.2, -.2], "q50": [0, 0], "q90": [.2, .2], "reference_price": [100, 100]})
    s = score(f)
    assert np.isclose(s["return_mae"], .1)
    assert np.isclose(s["price_mae_usd"], 10.)
    assert np.isclose(s["mae_skill_vs_no_change"], 0.)
    assert s["coverage80"] == 1.


def test_nonoverlap_count_is_not_raw_sample_count():
    ix = pd.date_range("2026-01-01T00:00:00Z", periods=50, freq="h")
    ends = pd.Series(ix + pd.Timedelta(hours=7), index=ix)
    assert nonoverlap_count(ix, ends) == 8


@pytest.mark.parametrize("new_receipt", ["2026-01-01T03:00:00Z", "2026-01-01T02:00:00.000001Z"])
def test_sqlite_revision_filter_precedes_ranking(tmp_path, release_data, new_receipt):
    db = tmp_path / "observations.db"
    with sqlite3.connect(db) as con:
        con.execute("CREATE TABLE bars(product, open_time, observed_at, revision, open,high,low,close,volume,content_hash,raw_hash)")
        con.executemany("INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)", [
            ("ETH-USD", "2026-01-01T00:00:00Z", "2026-01-01T01:01:00Z", 1, 100, 120, 90, 110, 10, "old", "raw1"),
            ("ETH-USD", "2026-01-01T00:00:00Z", new_receipt, 2, 100, 120, 90, 115, 10, "new", "raw2")])
    release_data.update(vintage_mode="observed_vintages", quality_status="pass", license_status="approved_for_intended_use")
    release_data["checksums"]["observations.db"] = file_hash(db)
    r = pin(tmp_path, release_data)
    before = file_hash(db)
    f = load_bars(r, tmp_path, mode="prospective", issued_at="2026-01-01T02:00:00Z")
    assert f.iloc[0].close == 110 and f.iloc[0].revision == 1
    assert file_hash(db) == before
    assert not Path(str(db) + "-wal").exists()


@pytest.mark.parametrize("field", ["release_id", "feature_set_version", "model_version", "target_spec_id"])
def test_output_requires_pinned_provenance(record, field):
    record[field] = "latest"
    with pytest.raises(ValueError): validate_prediction(record)


def test_validation_label_must_mature(record):
    record["validation_target_end"] = record["input_cutoff"]
    with pytest.raises(ValueError): validate_prediction(record)


def test_qlike_identity_and_invalid_inputs():
    from model_lab.benchmark import qlike
    assert np.isclose(qlike(np.array([.1, .2]), np.array([.1, .2])), 0.)
    with pytest.raises(ValueError): qlike(np.array([.1]), np.array([0.]))


def test_failed_experiment_is_not_overwritten(tmp_path):
    from model_lab.reproduce import evaluate
    p = tmp_path / "experiment"; p.mkdir()
    with pytest.raises(FileExistsError): evaluate(None, tmp_path, p)
