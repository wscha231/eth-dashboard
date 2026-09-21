import copy
import json
from pathlib import Path

import pytest

from scripts.verify_v2_site import validate_availability, validate_static


ROOT = Path(__file__).resolve().parents[1]


def availability(status="not_needed"):
    missing = {"ETH-USD": 0, "BTC-USD": 0} if status == "not_needed" else {"ETH-USD": 0, "BTC-USD": 1}
    return {
        "schema": 1,
        "generated_at": "2026-09-21T11:04:10+00:00",
        "policy": {
            "role": "availability_shadow_only",
            "head": "center",
            "promotion": "none",
            "public_delivery": False,
            "imputation": False,
        },
        "decision": {
            "origin": "2026-09-21T11:00:00+00:00",
            "status": status,
            "reason": "strict_input_condition_satisfied",
            "public_delivery": False,
            "strict_missing_counts": missing,
        },
        "promotion": "none",
        "predictive_skill": "not_claimed",
    }


def test_current_v2_source_contract_is_self_consistent():
    for name, path in (
        ("root", ROOT / "forecast_site/public/v2.html"),
        ("script", ROOT / "forecast_site/public/volatility.js"),
        ("korean", ROOT / "forecast_site/public/ko/index.html"),
    ):
        text = path.read_text(encoding="utf-8")
        validate_static(name, text, text)


@pytest.mark.parametrize("status", ["not_needed", "eligible", "blocked"])
def test_availability_contract_accepts_research_only_states(status):
    validate_availability(availability(status))


@pytest.mark.parametrize(
    "path,value",
    [
        (("policy", "promotion"), "approved"),
        (("policy", "public_delivery"), True),
        (("decision", "public_delivery"), True),
        (("promotion",), "approved"),
        (("predictive_skill",), "proven"),
    ],
)
def test_availability_contract_rejects_promotion_or_public_delivery(path, value):
    row = availability("eligible")
    target = row
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_availability(row)


def test_not_needed_requires_complete_strict_inputs():
    row = availability()
    row["decision"]["strict_missing_counts"]["BTC-USD"] = 1
    with pytest.raises(ValueError, match="strict ETH/BTC"):
        validate_availability(row)


def test_static_verifier_rejects_old_root_and_missing_status_logic():
    root = (ROOT / "forecast_site/public/v2.html").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        validate_static("root", root.replace('id="service"', 'id="old-service"', 1), root)
    script = (ROOT / "forecast_site/public/volatility.js").read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="status logic"):
        validate_static("script", script.replace("availability_shadow.json", "missing.json", 1),
                        script.replace("availability_shadow.json", "missing.json", 1))


def test_workflow_runs_external_v2_verifier_after_privacy_check():
    workflow = (ROOT / ".github/workflows/site_search.yml").read_text(encoding="utf-8")
    assert "scripts/verify_v2_site.py" in workflow
    assert workflow.index("Verify the public privacy policy") < workflow.index("Verify deployed V2 homepage")
