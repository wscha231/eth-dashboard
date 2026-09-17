import numpy as np
import pandas as pd

from research.model.full_horizon_v2_backtest import (
    ALL_HORIZONS,
    LONG_HORIZONS,
    POLICY,
    center_gate,
    event_gate,
    site_recommendation,
)
from research.model.full_horizon_v2_backtest_replay import _frames_from_replay


def review_frames():
    slots = []
    for year in (2024, 2025, 2026):
        slots.extend(pd.date_range(f"{year}-01-01", periods=60, freq="6D", tz="UTC"))
    n = len(slots)
    actual = np.tile(np.array([-0.08, -0.03, 0.02, 0.06, 0.11]), n // 5 + 1)[:n]
    terminal = np.tile(np.array([0, 1, 2]), n // 3 + 1)[:n]
    up = (actual > 0.04).astype(int)
    down = (actual < -0.04).astype(int)

    base = pd.DataFrame({
        "slot": slots,
        "target_end": [str(x + pd.Timedelta(hours=7)) for x in slots],
        "return": actual,
        "terminal": terminal,
        "up": up,
        "down": down,
        "q50": np.zeros(n),
        "p_down": np.full(n, 1 / 3),
        "p_flat": np.full(n, 1 / 3),
        "p_up": np.full(n, 1 / 3),
        "hit_up": np.full(n, 0.5),
        "hit_down": np.full(n, 0.5),
    })
    incumbent = base.copy()
    incumbent["q50"] = actual * 0.25

    candidate = base.copy()
    candidate["q50"] = actual
    probs = np.zeros((n, 3))
    probs[np.arange(n), terminal] = 1.0
    candidate[["p_down", "p_flat", "p_up"]] = probs
    candidate["hit_up"] = up
    candidate["hit_down"] = down
    return candidate, incumbent, base


def test_policy_covers_exact_operational_six_horizons_and_freezes_long_extension():
    assert tuple(POLICY["horizons_hours"]) == tuple(ALL_HORIZONS) == (6, 24, 72, 168, 336, 720)
    assert LONG_HORIZONS == (168, 336, 720)
    assert POLICY["frozen_before_results"] is True
    assert "no post-hoc" in POLICY["variance_distribution_long_policy"]["selection_rule"]


def test_center_gate_can_pass_only_with_real_improvement_across_blocks():
    candidate, incumbent, baseline = review_frames()
    result = center_gate(candidate, incumbent, baseline, 6)
    assert result["passed"] is True
    assert result["mae_skill_vs_no_change"] > 0.9
    assert result["mae_improvement_vs_incumbent"] > 0.9
    assert result["improving_calendar_blocks"] == 3
    assert result["paired_vs_rowwise_better_baseline"]["probability_candidate_better"] > 0.99


def test_event_gate_and_site_policy_hide_failed_legacy_heads():
    candidate, incumbent, baseline = review_frames()
    event = event_gate(candidate, incumbent, baseline, 6)
    assert event["passed"] is True
    assert event["brier_skill_vs_prior"] > 0.9
    assert event["ece"] <= 0.03

    heads = {
        str(h): {
            "center": {"passed": False},
            "event": {"passed": False},
            "distribution": {"passed": False},
            "volatility": {"passed": h in (6, 24, 72)},
        }
        for h in ALL_HORIZONS
    }
    rec = site_recommendation(heads)
    assert rec["remove_legacy_exact_target_price_from_default_surface"] is True
    assert rec["remove_unvalidated_direction_probabilities_from_default_surface"] is True
    assert rec["show_volatility_shadow_where_variance_gate_passes"] is True
    assert rec["preserve_old_forecasts_in_history_audit"] is True


def test_full_replay_extraction_uses_embedded_climatology_at_same_origin():
    payload = {
        "points": [{
            "slot": "2024-01-01T00:00:00+00:00",
            "target_end": "2024-01-01T07:00:00+00:00",
            "return": 0.02,
            "terminal": 2,
            "up": 1,
            "down": 0,
            "q50": 0.015,
            "p_down": 0.1,
            "p_flat": 0.2,
            "p_up": 0.7,
            "hit_up": 0.8,
            "hit_down": 0.1,
            "baseline": {
                "q50": 0.001,
                "p_down": 0.2,
                "p_flat": 0.6,
                "p_up": 0.2,
                "hit_up": 0.3,
                "hit_down": 0.3,
            },
        }]
    }
    candidate, incumbent, baseline = _frames_from_replay(payload)
    assert candidate.loc[0, "q50"] == 0.015
    assert baseline.loc[0, "q50"] == 0.001
    assert incumbent.equals(baseline)
    assert candidate.loc[0, "return"] == baseline.loc[0, "return"] == 0.02
    assert candidate.loc[0, "slot"] == baseline.loc[0, "slot"]
