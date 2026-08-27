"""Tests for forecast_site.persist_backtest.

Focus areas (by blast radius if broken):
  1. Idempotency — re-running the daily cron on the same freeze JSON must
     not duplicate rows. Without this test a silent bug could pile
     7 × daily = 50 duplicate backtest_runs rows/week into the DB.
  2. Longrun DELETE+INSERT semantics — each longrun checkpoint rewrites
     the JSON with a new sha1. We require "one live row per phase",
     not "one row per checkpoint". Getting this wrong silently doubles
     the metrics counts the frontend reads.
  3. CASCADE delete — when a run row disappears (e.g. longrun refresh),
     its metrics + predictions must also disappear. A bug here leaves
     orphan rows referencing a nonexistent run_id.
  4. NaN float → NULL coercion — the real freeze pipeline emits NaN for
     missing ensemble component_models. sqlite3 silently converts these
     to the string "nan", which poisons aggregates. We explicitly
     coerce to None.

Not tested here (out of scope):
  - CLI argparse behaviour (exercised end-to-end by the persist_and_export
    driver in a separate test).
  - The freeze JSON's own shape — that contract belongs to the freeze
    scripts that emit it.
"""
from __future__ import annotations

import json
import math

import pytest

from forecast_site import persist_backtest


# ---------------------------------------------------------------------------
# Basic ingest
# ---------------------------------------------------------------------------

def test_ingest_creates_run_and_metrics(temp_db, tiny_freeze_json):
    path = tiny_freeze_json(name="phase_test_metrics.json", n_models=3)
    phase, run_id, metrics, predictions = persist_backtest.ingest_file(temp_db, path)
    assert phase == "phase_test"
    assert run_id is not None
    # 3 models × 2 horizons × 2 heads (reg + cls) = 12 rows
    assert metrics == 12
    assert predictions == 0  # non-longrun has no predictions payload

    # Verify the run row actually landed.
    row = temp_db.execute(
        "SELECT model_phase, mode, cv_test_size FROM backtest_runs WHERE backtest_run_id = ?",
        (run_id,),
    ).fetchone()
    assert row["model_phase"] == "phase_test"
    assert row["cv_test_size"] == 60

    # Verify metrics rows are linked to the run.
    metric_count = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_metrics WHERE backtest_run_id = ?",
        (run_id,),
    ).fetchone()[0]
    assert metric_count == 12


def test_ingest_idempotent_on_second_call(temp_db, tiny_freeze_json):
    """Re-ingesting the same file must be a no-op (run_id_or_None == None).

    This is the single most important contract — the daily cron relies on
    it. Without idempotency the DB would grow unboundedly."""
    path = tiny_freeze_json(name="phase_idem_metrics.json")

    _, first_run_id, _, _ = persist_backtest.ingest_file(temp_db, path)
    assert first_run_id is not None

    _, second_run_id, metrics, predictions = persist_backtest.ingest_file(temp_db, path)
    assert second_run_id is None    # "already ingested" sentinel
    assert metrics == 0
    assert predictions == 0

    # DB state must match the first call — exactly one run, no duplicates.
    run_count = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE model_phase = 'phase_idem'"
    ).fetchone()[0]
    assert run_count == 1


def test_ingest_different_sha_creates_new_run(temp_db, tiny_freeze_json):
    """If the freeze JSON is edited (any byte change → new sha1), we want a
    NEW run row — not an upsert. Preserves history for the version timeline."""
    path_v1 = tiny_freeze_json(name="phase_histv_metrics.json", frozen_at="2026-01-01T00:00:00+00:00")
    _, run_id_v1, _, _ = persist_backtest.ingest_file(temp_db, path_v1)

    # Overwrite same filename with different content → different sha.
    path_v2 = tiny_freeze_json(name="phase_histv_metrics.json", frozen_at="2026-01-02T00:00:00+00:00")
    _, run_id_v2, _, _ = persist_backtest.ingest_file(temp_db, path_v2)

    assert run_id_v1 != run_id_v2
    run_count = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE model_phase = 'phase_histv'"
    ).fetchone()[0]
    assert run_count == 2


# ---------------------------------------------------------------------------
# Longrun refresh semantics — the trickiest correctness region.
# ---------------------------------------------------------------------------

def test_longrun_inserts_run_metrics_and_predictions(temp_db, tiny_freeze_json):
    path = tiny_freeze_json(
        name="phase_lr_longrun_oof_metrics.json",
        mode="longrun_oof_phase_lr_36x30",
        include_predictions=True,
        n_pred_dates=5,
    )
    _, run_id, metrics, predictions = persist_backtest.ingest_file(temp_db, path)
    assert run_id is not None
    assert metrics == 8   # 2 models × 2 horizons × 2 heads
    # 5 dates × 2 heads × 2 horizons = 20 prediction rows
    assert predictions == 20

    stored_pred_count = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_predictions WHERE backtest_run_id = ?",
        (run_id,),
    ).fetchone()[0]
    assert stored_pred_count == 20

    classification_row = temp_db.execute(
        """
        SELECT probability_up, direction_score_up, predicted_label
        FROM backtest_predictions
        WHERE backtest_run_id = ? AND head = 'classification'
        LIMIT 1
        """,
        (run_id,),
    ).fetchone()
    assert classification_row["probability_up"] == pytest.approx(0.6)
    assert classification_row["direction_score_up"] == pytest.approx(0.65)
    assert classification_row["predicted_label"] == 1


def test_longrun_second_ingest_replaces_previous_run(temp_db, tiny_freeze_json):
    """Longrun refresh = DELETE+INSERT. After ingesting v1 then v2 of the
    same phase (both with mode='longrun_...'), only v2's row should remain.

    The v1's metrics + predictions must also be gone — orphans would show
    up in export_backtest_json output as stale data."""
    path_v1 = tiny_freeze_json(
        name="phase_lrrefresh_longrun_oof_metrics.json",
        mode="longrun_oof_phase_lrrefresh_36x30",
        include_predictions=True,
        n_pred_dates=3,
        frozen_at="2026-01-01T00:00:00+00:00",
    )
    _, run_id_v1, _, _ = persist_backtest.ingest_file(temp_db, path_v1)
    assert run_id_v1 is not None

    path_v2 = tiny_freeze_json(
        name="phase_lrrefresh_longrun_oof_metrics.json",
        mode="longrun_oof_phase_lrrefresh_36x30",
        include_predictions=True,
        n_pred_dates=6,                       # more folds this round
        frozen_at="2026-01-02T00:00:00+00:00",
    )
    _, run_id_v2, _, _ = persist_backtest.ingest_file(temp_db, path_v2)
    assert run_id_v2 is not None
    assert run_id_v2 != run_id_v1

    # Exactly one run remains for this phase.
    run_count = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_runs WHERE model_phase = 'phase_lrrefresh_longrun_oof'"
    ).fetchone()[0]
    assert run_count == 1

    # v1's metrics/predictions must have cascaded away.
    v1_metrics = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_metrics WHERE backtest_run_id = ?",
        (run_id_v1,),
    ).fetchone()[0]
    v1_preds = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_predictions WHERE backtest_run_id = ?",
        (run_id_v1,),
    ).fetchone()[0]
    assert v1_metrics == 0
    assert v1_preds == 0

    # v2's predictions: 6 dates × 2 heads × 2 horizons = 24.
    v2_preds = temp_db.execute(
        "SELECT COUNT(*) FROM backtest_predictions WHERE backtest_run_id = ?",
        (run_id_v2,),
    ).fetchone()[0]
    assert v2_preds == 24


def test_longrun_detection_by_mode_prefix(temp_db, tiny_freeze_json):
    """_is_longrun_file fires on mode='longrun_...' even when the filename
    doesn't contain 'longrun_oof'. Reason: there's no hard coupling between
    filename and mode; the mode string is authoritative."""
    # Filename deliberately lacks "longrun_oof" to exercise the mode branch.
    path = tiny_freeze_json(
        name="phase_modelronly_metrics.json",
        mode="longrun_oof_phase_modelronly_36x30",
        include_predictions=True,
        n_pred_dates=2,
    )
    _, run_id, metrics, predictions = persist_backtest.ingest_file(temp_db, path)
    # If the longrun path were NOT taken, predictions count would be 0
    # (non-longrun path skips predictions). So a non-zero prediction count
    # here proves the mode-based detection fired.
    assert predictions == 8


def test_longrun_detection_by_filename_suffix(temp_db, tiny_freeze_json):
    """Conversely — filename pattern *_longrun_oof_metrics.json triggers
    longrun semantics even if the mode string doesn't start with 'longrun_'."""
    path = tiny_freeze_json(
        name="phase_fileonly_longrun_oof_metrics.json",
        mode="phase_fileonly_36x30",   # no "longrun_" prefix
        include_predictions=True,
        n_pred_dates=2,
    )
    _, run_id, metrics, predictions = persist_backtest.ingest_file(temp_db, path)
    assert predictions == 8


# ---------------------------------------------------------------------------
# NaN/None handling — coerce to SQL NULL, not the string "nan".
# ---------------------------------------------------------------------------

def test_nan_float_coerced_to_null(temp_db, tmp_path):
    """A freeze JSON with a NaN in a leaderboard float must store as NULL.
    The real-world case is ensemble component_models: NaN for non-ensemble
    rows in pandas → persists as 'nan' string if not coerced → breaks SQL
    aggregations that expect NULLs."""
    # Hand-roll a payload containing NaN. We can't put raw NaN through
    # json.dumps (it emits "NaN" which json.loads accepts as float('nan')
    # but STRICT parsers reject). The real freeze JSONs contain literal
    # "NaN" and python's json module does decode that. We build one.
    payload = {
        "mode":     "phase_nantest",
        "frozen_at": "2026-01-01T00:00:00+00:00",
        "cv_test_size": 60,
        "master_data_csv": "x",
        "eth_price_forecast_bytes": 1,
        "horizons": {
            "7": {
                "regression_leaderboard": [{
                    "model":      "foo",
                    "folds":      3.0,
                    "return_mae": float("nan"),      # NaN → must become NULL
                    "price_rmse": 100.0,
                    "component_models": None,
                }],
                "classification_leaderboard": [],
            },
        },
    }
    path = tmp_path / "phase_nantest_metrics.json"
    # Use allow_nan=True (default) so NaN serialises as literal NaN token.
    path.write_text(json.dumps(payload), encoding="utf-8")

    _, run_id, _, _ = persist_backtest.ingest_file(temp_db, path)
    row = temp_db.execute(
        "SELECT return_mae, price_rmse FROM backtest_metrics WHERE backtest_run_id = ?",
        (run_id,),
    ).fetchone()
    assert row["return_mae"] is None      # coerced from NaN
    assert row["price_rmse"] == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Smoke test against the real baseline fixture.
#
# Doubles as a regression guard: if the real freeze JSON schema drifts
# (e.g. a new leaderboard key appears), this test still passes as long
# as the known columns populate — but will surface if the basic ingest
# contract (run row + metric rows) breaks.
# ---------------------------------------------------------------------------

def test_ingest_real_baseline_fixture(temp_db):
    from pathlib import Path as _P
    real = _P(__file__).parent / "phase0" / "baseline_metrics.json"
    if not real.exists():
        pytest.skip("baseline_metrics.json fixture missing")
    phase, run_id, metrics, predictions = persist_backtest.ingest_file(temp_db, real)
    assert phase == "baseline"
    assert run_id is not None
    assert metrics > 0
    assert predictions == 0  # baseline isn't a longrun
