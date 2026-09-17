import numpy as np

from research.events.r2_range_volatility import POLICY, HORIZONS, qlike, wis80, _distribution_metrics


def test_policy_is_preregistered_short_horizon_only():
    assert HORIZONS == (6, 24, 72)
    assert POLICY["models"] == ["persistence_720", "ewma_168", "har_rv"]
    assert POLICY["variance_gate"]["qlike_skill_vs_persistence"] == 0.05
    assert POLICY["distribution_gate"]["coverage80_min"] == 0.76
    assert POLICY["distribution_gate"]["coverage80_max"] == 0.84


def test_qlike_is_zero_for_perfect_variance():
    x=np.array([0.01,0.02,0.03])
    assert abs(qlike(x,x)) < 1e-12


def test_qlike_rejects_nonpositive_variance():
    import pytest
    with pytest.raises(ValueError):
        qlike(np.array([0.01,0.02]),np.array([0.01,0.0]))


def test_wis_rewards_tighter_correct_interval():
    y=np.array([0.0,0.01,-0.01])
    tight=wis80(y,np.full(3,-0.05),np.zeros(3),np.full(3,0.05))
    wide=wis80(y,np.full(3,-0.20),np.zeros(3),np.full(3,0.20))
    assert tight < wide


def test_distribution_metrics_are_finite():
    out=_distribution_metrics(np.array([0.01,-0.01,0.02]),np.array([0.0004,0.0005,0.0006]),24)
    assert 0 <= out["coverage80"] <= 1
    assert out["mean_width"] > 0
    assert out["wis80"] >= 0
