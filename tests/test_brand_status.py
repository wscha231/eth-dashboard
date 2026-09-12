from pathlib import Path
import json
import shutil
import subprocess
import xml.etree.ElementTree as ET

import pytest


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "forecast_site/public"


def test_status_transitions():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for frontend runtime checks")
    subprocess.run([node, str(ROOT / "tests/brand_status.test.cjs")], check=True)


def test_logo_is_self_contained_and_donation_precedes_wallets():
    svg = ET.parse(PUBLIC / "assets/etherforecast-logo.svg").getroot()
    ids = [node.attrib["id"] for node in svg.iter() if "id" in node.attrib]
    assert len(ids) == len(set(ids))
    for node in svg.iter():
        assert node.tag.rsplit("}", 1)[-1] not in {"script", "foreignObject"}
        for key, value in node.attrib.items():
            assert not key.startswith("on")
            if key.endswith("href"):
                assert value.startswith("data:image/jpeg;base64,")
    html = (PUBLIC / "index.html").read_text()
    assert html.index('id="ef-brand"') < html.index('class="hero"')
    assert html.count('id="event-status"') == 1
    assert html.index('id="support-research"') < html.index('id="donation-wallets"')
    assert "evaluation targets, not expected gains" in html
    assert "$4,826 / year" in html


def test_status_cache_and_workflow_trigger():
    config = json.loads((ROOT / "forecast_site/vercel.json").read_text())
    status = next(item for item in config["headers"] if item["source"] == "/site_status.json")
    assert "no-store" in status["headers"][0]["value"]
    workflow = (ROOT / ".github/workflows/site_search.yml").read_text()
    assert "forecast_site/public/site_status.json" in workflow
