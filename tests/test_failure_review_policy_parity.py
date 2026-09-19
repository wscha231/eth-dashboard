import pytest

from scripts.review_forecast_failures import score
from signal_pipeline.evaluate import event_metrics


def test_failure_review_uses_same_strict_alert_boundary_as_incumbent():
    rows = [dict(q50=0.0, **{'return': 0.01}, up=y, down=1-y,
                 hit_up=p, hit_down=1-p, threshold_up=0.5, threshold_down=0.5)
            for p,y in ((0.5,1),(0.6,1),(0.4,0),(0.5,0))]
    actual = score(rows)
    for side in ('up','down'):
        expected = event_metrics([r[side] for r in rows],
                                 [r['hit_'+side] for r in rows],
                                 [r['threshold_'+side] for r in rows])
        assert actual[side]['alerts'] == expected['alerts'] == 1
        assert actual[side]['events'] == expected['events']
        assert actual[side]['false_alerts'] == expected['false_alerts']
        assert actual[side]['brier'] == pytest.approx(expected['brier'])
        assert actual[side]['precision'] == pytest.approx(expected['precision'])
        assert actual[side]['recall'] == pytest.approx(expected['recall'])
