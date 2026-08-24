from scripts.evaluate_eth_candidate import evaluated_through_by_horizon


def test_evaluated_through_by_horizon_uses_latest_target_date() -> None:
    candidate = {
        "horizons": {
            "7": {"predictions": [{"target_date": "2026-01-02"}, {"target_date": "2026-01-09"}]},
            "30": {"predictions": [{"target_date": "2026-01-05"}]},
        }
    }

    assert evaluated_through_by_horizon(candidate) == {
        "7": "2026-01-09",
        "30": "2026-01-05",
    }
