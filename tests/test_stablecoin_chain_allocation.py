from datetime import datetime, timezone
import math

import pandas as pd
import pytest

from data_lab.stablecoin_chain_allocation import (
    OUSDIdentityError,
    build_stablecoin_chain_allocation_features,
    parse_open_standard_ousd_snapshot,
)


def ousd_payload():
    return {
        "peggedAssets": [
            {
                "id": "origin-dollar",
                "name": "Origin Dollar",
                "symbol": "OUSD",
                "pegType": "peggedUSD",
                "circulating": {"peggedUSD": 6_000_000},
                "chains": ["Ethereum"],
                "chainCirculating": [
                    {"chain": "Ethereum", "circulating": 6_000_000},
                ],
            },
            {
                "id": "443",
                "name": "Open USD",
                "symbol": "OUSD",
                "pegType": "peggedUSD",
                "price": 1.0,
                "circulating": 460_000_000,
                "chains": ["Tempo", "Ethereum", "Solana", "Base"],
                "chainCirculating": [
                    {"chain": "Ethereum", "circulating": 10_000_000},
                    {"chain": "Solana", "circulating": 10_000_000},
                    {"chain": "Tempo", "circulating": 425_000_000},
                    {"chain": "Base", "circulating": 15_000_000},
                ],
            },
        ]
    }


def test_open_standard_ousd_identity_does_not_confuse_origin_dollar():
    row = parse_open_standard_ousd_snapshot(
        ousd_payload(), received_at="2026-10-08T12:00:00Z"
    )
    assert row["defillama_asset_id"] == "443"
    assert row["received_at"] == row["available_at"]
    assert row["source_observation_at"] is None
    assert row["total_supply_ousd"] == 460_000_000
    assert row["ethbase_supply_ousd"] == 25_000_000
    assert row["ethbase_share"] == pytest.approx(25 / 460)
    assert row["tempo_share"] == pytest.approx(425 / 460)
    assert row["raw_asset_sha256"] and len(row["raw_asset_sha256"]) == 64



def test_open_standard_ousd_fails_closed_on_provider_id_mismatch():
    corrupted = ousd_payload()
    corrupted["peggedAssets"][1]["id"] = "23"
    with pytest.raises(OUSDIdentityError, match="expected exactly one Open Standard OUSD asset"):
        parse_open_standard_ousd_snapshot(corrupted)

def test_open_standard_ousd_fails_closed_on_missing_or_new_positive_chain():
    missing = ousd_payload()
    missing["peggedAssets"][1]["chainCirculating"] = [
        row for row in missing["peggedAssets"][1]["chainCirculating"] if row["chain"] != "Tempo"
    ]
    with pytest.raises(OUSDIdentityError, match="missing official OUSD chains"):
        parse_open_standard_ousd_snapshot(missing)

    extra = ousd_payload()
    extra["peggedAssets"][1]["chainCirculating"].append(
        {"chain": "NewChain", "circulating": 1}
    )
    extra["peggedAssets"][1]["circulating"] += 1
    with pytest.raises(OUSDIdentityError, match="unreviewed positive OUSD chain supply"):
        parse_open_standard_ousd_snapshot(extra)


def test_ousd_collector_writes_actual_receipt_without_model_materialization(tmp_path, monkeypatch):
    import eth_data_collector
    from scripts.collect_free_research_sources import collect_open_standard_ousd_receipt

    timing = {}

    def completed_response(*args, **kwargs):
        timing["response_returned_at"] = datetime.now(timezone.utc)
        return ousd_payload()

    monkeypatch.setattr(eth_data_collector, "request_json", completed_response)
    result = collect_open_standard_ousd_receipt(tmp_path)
    path = tmp_path / "open_standard_ousd_chain_supply_receipt.csv"
    frame = pd.read_csv(path)

    assert result["status"] == "ok"
    assert result["model_use"] == "blocked"
    assert result["public_hot_branch"] == "blocked"
    assert len(frame) == 1
    assert frame.iloc[0]["source_id"] == "defillama_open_standard_ousd_receipt"
    assert frame.iloc[0]["total_supply_ousd"] == 460_000_000
    assert frame.iloc[0]["received_at"] == frame.iloc[0]["available_at"]
    assert pd.Timestamp(frame.iloc[0]["received_at"]) >= pd.Timestamp(timing["response_returned_at"])


def test_stablecoin_allocation_features_match_frozen_formulas_and_keep_missingness():
    idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    global_supply = pd.Series([100, 110, 121, 133.1, 146.41, 161.051, 177.1561, 194.87171, 214.358881, 235.7947691], index=idx)
    eth = pd.Series([40, 44, 48, 52, 56, 60, 64, 68, 72, 76], index=idx)
    base = pd.Series([10, 11, 12, 13, 14, 15, 16, 17, 18, 19], index=idx)

    out = build_stablecoin_chain_allocation_features(global_supply, eth, base)
    assert out.loc[idx[1], "sc_global_growth_1d"] == pytest.approx(math.log(1.1))
    share0 = 50 / 100
    share1 = 55 / 110
    assert out.loc[idx[1], "sc_ethbase_share_delta_1d"] == pytest.approx(share1 - share0)
    assert pd.notna(out.loc[idx[7], "sc_ethbase_share_delta_7d"])

    base_missing = base.copy()
    base_missing.loc[idx[5]] = float("nan")
    missing = build_stablecoin_chain_allocation_features(global_supply, eth, base_missing)
    assert pd.isna(missing.loc[idx[5], "sc_ethbase_share_delta_1d"])
    assert pd.isna(missing.loc[idx[5], "sc_ethbase_vs_rest_flow_1d"])

def test_defillama_backfill_does_not_report_empty_refresh_as_ok(tmp_path, monkeypatch):
    import eth_data_collector
    from scripts.collect_free_research_sources import collect_defillama_full

    cached_path = tmp_path / "defillama_global_stablecoins_daily.csv"
    pd.DataFrame(
        {"total_supply_usd": [100.0]},
        index=pd.DatetimeIndex(["2026-10-01T00:00:00Z"], name="date"),
    ).to_csv(cached_path, index_label="date")

    good = pd.DataFrame(
        {"value": [1.0]},
        index=pd.DatetimeIndex(["2026-10-02T00:00:00Z"]),
    )

    def stablecoins(chain, start, end):
        if chain in {"all", "Base"}:
            return pd.DataFrame()
        return good.copy()

    monkeypatch.setattr(eth_data_collector, "collect_defillama_stablecoins", stablecoins)
    monkeypatch.setattr(eth_data_collector, "collect_defillama_chain_tvl", lambda *args, **kwargs: good.copy())
    monkeypatch.setattr(eth_data_collector, "collect_defillama_dex_volume", lambda *args, **kwargs: good.copy())

    result = collect_defillama_full(
        tmp_path,
        pd.Timestamp("2026-10-01T00:00:00Z"),
        pd.Timestamp("2026-10-02T00:00:00Z"),
    )

    assert result["defillama_global_stablecoins_daily.csv"]["status"] == "error"
    assert result["defillama_global_stablecoins_daily.csv"]["error"] == "ValueError"
    assert result["defillama_base_stablecoins_daily.csv"]["status"] == "error"
    assert result["defillama_base_stablecoins_daily.csv"]["error"] == "ValueError"
    # A failed current refresh must not erase the retained reconstructed cache.
    cached = pd.read_csv(cached_path)
    assert len(cached) == 1
    assert cached.iloc[0]["total_supply_usd"] == 100.0

