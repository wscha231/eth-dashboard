from pathlib import Path


INDEX = Path("forecast_site/public/index.html")


def test_frontend_separates_live_history_from_weekly_oof() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert 'loadJSON("history.json")' in html
    assert 'loadJSON("accuracy.json")' in html
    assert '"model_eval_latest.json"' in html
    assert '"model_eval_last_pass.json"' in html
    assert 'id="chart-live-h7"' in html
    assert 'id="chart-live-h30"' in html
    assert 'id="chart-oof-h7"' in html
    assert 'id="chart-oof-h30"' in html


def test_oof_chart_uses_selected_payload_line_before_raw_candidate() -> None:
    html = INDEX.read_text(encoding="utf-8")

    assert "predicted: Number(p.predicted_close)" in html
    assert "predicted: Number(p.model_predicted_close ?? p.raw_predicted_close ?? p.predicted_close)" not in html
    assert 'chartModel: p.chart_model || "model_forecast"' in html
