import json
from pathlib import Path

from scripts.publish_v2_assets import ASSETS, publish


ROOT = Path(__file__).resolve().parents[1]


def test_v2_home_removes_unvalidated_price_claims_from_default_surface():
    html = (ROOT / "forecast_site/public/v2.html").read_text(encoding="utf-8")
    assert "How much might ETH move?" in html
    assert "HAR-RV movement scale" in html
    assert "0 / 6 PASS" in html
    assert "6 / 6 PASS" in html
    assert "Legacy audit" in html
    assert "Exact target price" in html
    assert "80% price range" in html
    assert "volatility.js" in html
    assert 'id="event-horizon"' not in html
    assert "Estimated end price" not in html


def test_v2_javascript_consumes_variance_scale_without_price_quantiles():
    js = (ROOT / "forecast_site/public/volatility.js").read_text(encoding="utf-8")
    assert "variance_shadow.json" in js
    assert "predicted_endpoint_sigma" in js
    assert "historical_qlike_improvement" in js
    assert "minimum_nonoverlap_for_review" in js
    assert "terminal_down_flat_up" in js  # 6h Event-only experiment remains visible.
    assert "price_quantiles" not in js
    assert "q10" not in js and "q90" not in js


def test_vercel_routes_root_to_v2_and_variance_to_immutable_shadow_branch():
    config = json.loads((ROOT / "forecast_site/vercel.json").read_text(encoding="utf-8"))
    rewrites = {row["source"]: row["destination"] for row in config["rewrites"]}
    assert rewrites["/"] == "/v2.html"
    assert rewrites["/variance_shadow.json"].endswith(
        "/data/event-ledger/lake/event-ledger/variance-shadow/public.json"
    )
    assert rewrites["/signals.json"].endswith("/data/event-feed/forecast_site/public/signals.json")
    assert config["git"]["deploymentEnabled"]["data/daily-forecast"] is True


def test_v2_assets_are_copied_and_staged_by_site_publisher(tmp_path):
    target = tmp_path / "target"
    public = target / "forecast_site/public"
    public.mkdir(parents=True)
    copied = publish(target, source=ROOT, stage=False)
    assert tuple(ASSETS) == ("v2.html", "volatility.js")
    assert copied == ["forecast_site/public/v2.html", "forecast_site/public/volatility.js"]
    for name in ASSETS:
        assert (public / name).read_bytes() == (ROOT / "forecast_site/public" / name).read_bytes()


def test_legacy_index_is_preserved_for_audit():
    legacy = (ROOT / "forecast_site/public/index.html").read_text(encoding="utf-8")
    assert 'id="complete-evidence"' in legacy
    assert 'id="legacy-archive"' in legacy
