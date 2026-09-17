import numpy as np
import pandas as pd

from research.events.r2_distribution_gate import HORIZONS, POLICY, metrics, wis80_rows, _paired_probability


def test_distribution_policy_is_frozen():
    assert HORIZONS == (6, 24, 72)
    assert POLICY["candidate"] == "har_norm_conformal80"
    assert POLICY["gate"]["wis_skill_vs_each_baseline"] == 0.02
    assert POLICY["gate"]["coverage80_min"] == 0.76
    assert POLICY["gate"]["coverage80_max"] == 0.84
    assert POLICY["gate"]["width_ratio_vs_better_wis_baseline_max"] == 1.10
    assert POLICY["gate"]["paired_improvement_probability_vs_each_baseline_min"] == 0.90


def test_wis_rows_zero_for_perfect_zero_point_and_zero_width():
    loss=wis80_rows(np.zeros(3),np.zeros(3),np.zeros(3),np.zeros(3))
    assert np.allclose(loss,0)


def test_metrics_reports_coverage_and_width():
    y=np.array([-0.05,0,0.05])
    out=metrics(y,np.full(3,-0.1),np.zeros(3),np.full(3,0.1))
    assert out["coverage80"] == 1.0
    assert np.isclose(out["mean_width"],0.2)
    assert out["rows"] == 3


def test_paired_probability_detects_uniform_improvement():
    slots=pd.date_range("2024-01-01",periods=120,freq="D",tz="UTC")
    out=_paired_probability(np.ones(120),np.full(120,2.0),slots,24,samples=200)
    assert out["probability_candidate_improves"] == 1.0
    assert out["mean_loss_difference"] < 0
