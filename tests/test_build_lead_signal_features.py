from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.build_lead_signal_features import _feature_report, _source_months


def test_pr1_manifest_pins_four_streams_to_one_common_month() -> None:
    manifest = json.loads(
        Path("lake/manifests/lead_signal_sources.json").read_text(encoding="utf-8")
    )
    months, common_end = _source_months(manifest)

    assert common_end == "2026-07"
    assert {name: len(values) for name, values in months.items()} == {
        "binance_spot_ethusdt_1h": 108,
        "binance_um_ethusdt_1h": 79,
        "binance_spot_btcusdt_1h": 108,
        "binance_um_btcusdt_1h": 79,
    }
    assert sum(len(values) for values in months.values()) == 374


def test_feature_readiness_requires_two_year_common_history() -> None:
    index = pd.date_range("2020-01-01", periods=731, freq="D")
    features = pd.DataFrame(
        {
            "eth_spot_bar_count": 24.0,
            "eth_perp_bar_count": 24.0,
            "btc_spot_bar_count": 24.0,
            "btc_perp_bar_count": 24.0,
            "signal": np.arange(len(index), dtype=float),
        },
        index=index,
    )
    groups = {
        "order_flow": ["signal"],
        "leverage_basis": ["signal"],
        "intraday_risk": ["signal"],
        "cross_asset_leadership": ["signal"],
        "ethereum_liquidity": ["signal"],
    }
    report = _feature_report(
        features=features,
        groups=groups,
        archive_records=[{"validation_status": "pass", "size_bytes": 10}],
        as_of_date=index[-1],
        output_sha256="a" * 64,
        manifest_sha256="b" * 64,
    )

    assert report["decision"] == "pass_for_pr3_offline_evaluation"
    assert report["common_hourly_coverage"]["minimum_prior_days"] == 730
    assert report["gate"]["production_use_approved"] is False
    assert report["gate"]["model_training_performed"] is False


def test_generated_feature_artifact_matches_immutable_manifest() -> None:
    feature_path = Path("lake/gold/lead_signal_daily.csv.gz")
    manifest_path = Path("lake/manifests/lead_signal_features.json")
    report_path = Path("lake/reports/lead_signal_feature_readiness.json")
    if not all(path.is_file() for path in (feature_path, manifest_path, report_path)):
        pytest.skip("full source-derived feature artifact has not been generated")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    frame = pd.read_csv(feature_path, parse_dates=["date"])
    observed_hash = hashlib.sha256(feature_path.read_bytes()).hexdigest()

    assert manifest["scope"] == "offline_lead_signal_daily_features"
    assert manifest["feature_table"]["sha256"] == observed_hash
    assert report["feature_table"]["sha256"] == observed_hash
    assert report["decision"] == "pass_for_pr3_offline_evaluation"
    assert report["gate"]["production_use_approved"] is False
    assert len(manifest["binance_archives"]) == 374
    assert all(
        item["remote_sha256"] == item["local_sha256"]
        for item in manifest["binance_archives"]
    )
    assert not frame["date"].duplicated().any()
    assert frame["date"].max() == pd.Timestamp(manifest["as_of_date"])
    assert frame["feature_available_at_utc"].max() == "2026-08-01T00:00:00Z"
