"""Tests for forecast_site.export_backtest_json.

The frontend (public/index.html) depends on specific keys from each
output file. This suite pins the contract: if someone renames a field
or changes its type, the test fails loudly instead of the website
silently going blank.

Tested contracts:
  - export_versions()          → list of run-metadata dicts (feeds version timeline)
  - export_headline_table()    → best (reg/cls) per (phase, horizon) (feeds version RMSE/Brier charts)
  - export_full_matrix()       → every metric row (feeds drill-down)
  - export_longrun_history()   → ensemble-only predicted-vs-actual (feeds 3-yr chart + regime grid)
  - export_predictions_by_phase() → full per-phase OOF rows (feeds A/B diff + worst-cases)
  - main() → writes all expected files to --output-dir

Empty-DB behaviour is tested separately — the frontend has fallback
"not yet available" states that assume the export functions return
empty structures (not errors) on an empty DB.
"""
from __future__ import annotations

import json
import sys

import pytest

from forecast_site import export_backtest_json, persist_backtest


# ---------------------------------------------------------------------------
# Per-function contract tests against a controlled DB.
# ---------------------------------------------------------------------------

def test_export_versions_shape(temp_db, tiny_freeze_json):
    """Each version dict must carry the fields the version timeline reads."""
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(
        name="phase_alpha_metrics.json", frozen_at="2026-02-01T00:00:00+00:00",
    ))
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(
        name="phase_beta_metrics.json",  frozen_at="2026-02-02T00:00:00+00:00",
    ))

    versions = export_backtest_json.export_versions(temp_db)
    assert len(versions) == 2
    # Ordered by frozen_utc ASC.
    assert [v["model_phase"] for v in versions] == ["phase_alpha", "phase_beta"]
    for v in versions:
        # Keys the version timeline reads. Any rename breaks the frontend.
        assert "backtest_run_id" in v
        assert "model_phase" in v
        assert "frozen_utc" in v


def test_export_headline_picks_best_model_per_horizon(temp_db, tiny_freeze_json):
    """The headline endpoint should return ONE row per (phase, horizon)
    even though the fixture has multiple models per horizon.

    "Best" = lowest price_rmse (regression) and lowest brier_score
    (classification) — the frontend treats these as the canonical numbers."""
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(name="phase_head_metrics.json", n_models=3))
    headline = export_backtest_json.export_headline_table(temp_db)
    # 1 phase × 2 horizons = 2 rows (one per horizon).
    assert len(headline) == 2
    # Fixture: models named "model_0".."model_2", price_rmse=150+5*i → best is "model_0".
    for row in headline:
        assert row["reg_model"] == "model_0"
        # horizon_days and the field names the version chart reads.
        assert row["horizon_days"] in (7, 30)
        assert "reg_price_rmse" in row
        assert "cls_brier" in row


def test_export_full_matrix_one_row_per_model(temp_db, tiny_freeze_json):
    """Flat matrix = every (phase, horizon, head, model) combination."""
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(name="phase_mat_metrics.json", n_models=2))
    matrix = export_backtest_json.export_full_matrix(temp_db)
    # 2 models × 2 horizons × 2 heads = 8 rows
    assert len(matrix) == 8
    # Sanity: each row has model_phase and head set.
    for row in matrix:
        assert row["model_phase"] == "phase_mat"
        assert row["head"] in ("regression", "classification")


def test_export_longrun_history_filters_to_ensemble(temp_db, tmp_path):
    """export_longrun_history takes a ``model=`` argument and filters DB
    rows to that single model. The production payload uses
    ``trimmed_regression_ensemble_equal`` but the function must work for
    any model name the caller passes — the test passes a custom one so
    we don't have to synthesize the real ensemble."""
    # Synthesize a longrun with predictions for multiple models + horizons.
    payload = {
        "mode":                     "longrun_oof_phase_lrhist_36x30",
        "frozen_at":                "2026-02-05T00:00:00+00:00",
        "master_data_csv":          "x",
        "eth_price_forecast_bytes": 1,
        "cv_test_size":             30,
        "horizons": {
            "7":  {"regression_leaderboard": [], "classification_leaderboard": [],
                   "predictions": [
                       {"horizon_days": 7, "head": "regression", "model": m,
                        "prediction_date": f"2024-01-{d+1:02d}",
                        "target_date":     f"2024-01-{d+1+7:02d}",
                        "reference_close": 2000.0, "actual_close": 2020.0,
                        "actual_return": 0.01, "predicted_return": 0.015,
                        "predicted_close": 2030.0, "fold_index": d // 2}
                       for m in ("model_A", "model_B")
                       for d in range(3)
                   ]},
            "30": {"regression_leaderboard": [], "classification_leaderboard": [],
                   "predictions": [
                       {"horizon_days": 30, "head": "regression", "model": "model_A",
                        "prediction_date": f"2024-02-{d+1:02d}",
                        "target_date":     f"2024-02-{d+1+30:02d}" if d+1+30 <= 28
                                           else f"2024-03-{d+1+30-28:02d}",
                        "reference_close": 2100.0, "actual_close": 2200.0,
                        "actual_return": 0.04, "predicted_return": 0.03,
                        "predicted_close": 2163.0, "fold_index": d // 2}
                       for d in range(3)
                   ]},
        },
    }
    p = tmp_path / "phase_lrhist_longrun_oof_metrics.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    persist_backtest.ingest_file(temp_db, p)

    # Ask for model_A only. model_B rows must be excluded.
    history = export_backtest_json.export_longrun_history(temp_db, model="model_A")
    assert "phase_lrhist_longrun_oof" in history
    bundle = history["phase_lrhist_longrun_oof"]
    assert bundle["model"] == "model_A"
    # Both horizons should have points.
    assert set(bundle["points"].keys()) == {"7", "30"}
    # 3 predictions per horizon for model_A.
    assert len(bundle["points"]["7"])  == 3
    assert len(bundle["points"]["30"]) == 3
    # Each point carries the keys the frontend chart reads.
    sample = bundle["points"]["7"][0]
    assert {"target_date", "predicted_close", "actual_close"}.issubset(sample.keys())


def test_export_longrun_history_only_includes_longrun_runs(temp_db, tiny_freeze_json):
    """A non-longrun run that happens to have the requested model name must
    NOT appear in longrun_history. The SQL filter uses
    ``br.mode LIKE 'longrun_oof_%'`` — if that ever regresses, the
    chart would start mixing 3-fold freezes with 36-fold OOFs, which
    visualise completely differently."""
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(name="phase_nonlr_metrics.json"))
    history = export_backtest_json.export_longrun_history(temp_db, model="model_0")
    assert history == {}


def test_export_predictions_omits_phases_without_predictions(temp_db, tiny_freeze_json):
    """Standard freezes (non-longrun) have no `predictions` payload —
    they must NOT show up in the per-phase bundles dict, otherwise the
    frontend manifest would claim a JSON file exists that has zero rows.
    """
    # One standard, one longrun-with-predictions.
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(name="phase_std_metrics.json"))
    persist_backtest.ingest_file(temp_db, tiny_freeze_json(
        name="phase_withpreds_longrun_oof_metrics.json",
        mode="longrun_oof_phase_withpreds_36x30",
        include_predictions=True, n_pred_dates=2,
    ))

    bundles = export_backtest_json.export_predictions_by_phase(temp_db)
    # Only the longrun phase has predictions.
    assert list(bundles.keys()) == ["phase_withpreds_longrun_oof"]


# ---------------------------------------------------------------------------
# Empty-DB behaviour — all export_* functions must return empty containers
# (not raise). The frontend treats empty as "not yet available" and
# renders a graceful fallback.
# ---------------------------------------------------------------------------

def test_all_exports_safe_on_empty_db(temp_db):
    assert export_backtest_json.export_versions(temp_db) == []
    assert export_backtest_json.export_headline_table(temp_db) == []
    assert export_backtest_json.export_full_matrix(temp_db) == []
    assert export_backtest_json.export_longrun_history(temp_db) == {}
    assert export_backtest_json.export_predictions_by_phase(temp_db) == {}


# ---------------------------------------------------------------------------
# Full main() end-to-end: writes all expected files to --output-dir.
#
# export_backtest_json.main() parses sys.argv directly, so we swap argv.
# This mirrors what forecast_site.persist_and_export_longrun does in
# production — if the contract here breaks, the driver breaks too.
# ---------------------------------------------------------------------------

def test_main_writes_expected_files(temp_db_path, tiny_freeze_json, tmp_path, monkeypatch):
    # Arrange: persist a couple of phases, one with longrun predictions.
    from forecast_site.db import connect
    conn = connect(temp_db_path)
    try:
        persist_backtest.ingest_file(conn, tiny_freeze_json(name="phase_main_metrics.json"))
        persist_backtest.ingest_file(conn, tiny_freeze_json(
            name="phase_mainlr_longrun_oof_metrics.json",
            mode="longrun_oof_phase_mainlr_36x30",
            include_predictions=True, n_pred_dates=2,
        ))
    finally:
        conn.close()

    out_dir = tmp_path / "out"
    monkeypatch.setattr(sys, "argv", [
        "export_backtest_json",
        "--db", str(temp_db_path),
        "--output-dir", str(out_dir),
    ])
    export_backtest_json.main()

    # The three core files are always written.
    assert (out_dir / "backtest.json").exists()
    assert (out_dir / "backtest_versions.json").exists()
    assert (out_dir / "backtest_longrun_history.json").exists()

    # Per-phase prediction JSON written only for longrun phases.
    assert (out_dir / "backtest_predictions_phase_mainlr_longrun_oof.json").exists()
    # Non-longrun phase has no predictions → no prediction file.
    assert not (out_dir / "backtest_predictions_phase_main.json").exists()

    # Structural contract: backtest.json carries headline + matrix arrays.
    backtest = json.loads((out_dir / "backtest.json").read_text(encoding="utf-8"))
    assert "headline" in backtest and isinstance(backtest["headline"], list)
    assert "matrix"   in backtest and isinstance(backtest["matrix"],   list)
    assert "prediction_phases" in backtest  # lazy-load manifest
    # generated_at should be ISO8601 UTC (ends with +00:00 or Z).
    assert backtest["generated_at"].endswith(("+00:00", "Z"))

    # Versions manifest.
    versions = json.loads((out_dir / "backtest_versions.json").read_text(encoding="utf-8"))
    assert versions["count"] == 2
    assert len(versions["versions"]) == 2

    # Longrun history: should contain our one longrun phase. With the
    # default model name (trimmed_regression_ensemble_equal) our fixture
    # won't match — so phases may be empty. That's fine; what we're
    # asserting is file existence + valid JSON structure.
    longrun = json.loads((out_dir / "backtest_longrun_history.json").read_text(encoding="utf-8"))
    assert "chart_model" in longrun
    assert "phases" in longrun
