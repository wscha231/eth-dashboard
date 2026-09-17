import json

import pandas as pd

from data_lab import eth_supply


def history_payload(points=None, revision=7, generated=1789617600):
    if points is None:
        points = [
            {
                "epoch": 100,
                "attestationDutyEpoch": 99,
                "timestamp": 1789516800,
                "blocks": 7200,
                "supplyWei": "122000000000000000000000000",
                "issuanceWei": "2800000000000000000000",
                "burnWei": "100000000000000000000",
                "baseFeeBurnWei": "90000000000000000000",
                "blobBaseFeeBurnWei": "10000000000000000000",
                "consensusPenaltiesWei": "0",
                "otherExecutionBurnWei": "0",
                "netWei": "2700000000000000000000",
            },
            {
                "epoch": 325,
                "attestationDutyEpoch": 324,
                "timestamp": 1789603200,
                "blocks": 7201,
                "supplyWei": "122002700000000000000000000",
                "issuanceWei": "2900000000000000000000",
                "burnWei": "200000000000000000000",
                "baseFeeBurnWei": "180000000000000000000",
                "blobBaseFeeBurnWei": "20000000000000000000",
                "consensusPenaltiesWei": "0",
                "otherExecutionBurnWei": "0",
                "netWei": "2700000000000000000000",
            },
        ]
    return {
        "schemaVersion": 1,
        "revision": revision,
        "generatedAt": generated,
        "range": "retained",
        "interval": "day",
        "epochs": points,
        "slots": [],
        "staking": [],
    }


def live_payload(revision=8, generated=1789617600):
    return {
        "schemaVersion": 1,
        "revision": revision,
        "generatedAt": generated,
        "chainId": 1,
        "head": {"block": 25000000, "slot": 15000000, "timestamp": generated - 10},
        "finalized": {"block": 24999900, "epoch": 500000, "slot": 14999900},
        "supply": {
            "totalWei": {"value": "122005400000000000000000000", "status": "exact", "kind": "observed", "asOf": generated - 12},
            "finalizedTotalWei": {"value": "122005399000000000000000000", "status": "exact", "kind": "observed", "asOf": generated - 300},
            "asOf": {"block": 25000000, "slot": 15000000, "epoch": 500001, "timestamp": generated - 12},
        },
        "accounting": {
            "interval": "finalized_epoch",
            "epoch": 500000,
            "targetEpoch": 500000,
            "issuance": {"totalWei": {"value": "1200000000000000000", "status": "exact", "kind": "observed", "asOf": generated - 300}},
            "burn": {"totalWei": {"value": "200000000000000000", "status": "exact", "kind": "observed", "asOf": generated - 300}},
            "netWei": {"value": "1000000000000000000", "status": "exact", "kind": "derived", "asOf": generated - 300},
        },
    }


def test_history_parses_exact_wei_eth_and_identity():
    frame, meta = eth_supply.parse_history(history_payload())
    assert meta["history_rows"] == 2
    assert meta["source_interval"] == "day"
    assert meta["identity_nonzero_rows"] == 0
    assert frame.iloc[0]["issuance_eth"] == "2800"
    assert frame.iloc[0]["burn_eth"] == "100"
    assert frame.iloc[0]["net_eth"] == "2700"
    assert frame.iloc[0]["identity_error_wei"] == "0"


def test_live_parses_supply_accounting_and_identity():
    row = eth_supply.parse_live(live_payload())
    assert row["total_supply_eth"] == "122005400"
    assert row["issuance_eth"] == "1.2"
    assert row["burn_eth"] == "0.2"
    assert row["net_eth"] == "1"
    assert row["identity_error_wei"] == "0"
    assert row["total_supply_status"] == "exact"


def test_first_history_is_reconstructed_then_new_future_point_is_observed():
    snapshot, _ = eth_supply.parse_history(history_payload())
    latest, _, state, stats = eth_supply.reconcile_history(
        snapshot,
        existing=None,
        state=None,
        receipt_time="2026-09-17T03:00:00Z",
        raw_sha256="a" * 64,
    )
    assert stats["new_rows"] == 2
    assert set(latest["vintage_mode"]) == {"reconstructed"}
    assert set(latest["available_at"]) == {"2026-09-17T03:00:00+00:00"}

    point = history_payload()["epochs"][-1].copy()
    point["epoch"] = 550
    point["timestamp"] = int(pd.Timestamp("2026-09-18T00:00:00Z").timestamp())
    future, _ = eth_supply.parse_history(history_payload([point], revision=8, generated=int(pd.Timestamp("2026-09-18T03:00:00Z").timestamp())))
    latest2, _, _, stats2 = eth_supply.reconcile_history(
        future,
        existing=latest,
        state=state,
        receipt_time="2026-09-18T03:00:00Z",
        raw_sha256="b" * 64,
    )
    assert stats2["new_rows"] == 1
    added = latest2[latest2["timestamp"].str.startswith("2026-09-18")].iloc[0]
    assert added["vintage_mode"] == "observed_vintages"
    assert added["available_at"] == "2026-09-18T03:00:00+00:00"


def test_revision_cannot_leak_backwards():
    snapshot, _ = eth_supply.parse_history(history_payload())
    latest, _, state, _ = eth_supply.reconcile_history(
        snapshot, existing=None, state=None, receipt_time="2026-09-17T03:00:00Z", raw_sha256="a" * 64
    )
    changed = snapshot.copy()
    changed.loc[0, "burn_wei"] = "101000000000000000000"
    changed.loc[0, "burn_eth"] = "101"
    changed.loc[0, "net_wei"] = "2699000000000000000000"
    changed.loc[0, "net_eth"] = "2699"
    latest2, events, _, stats = eth_supply.reconcile_history(
        changed, existing=latest, state=state, receipt_time="2026-09-18T04:00:00Z", raw_sha256="b" * 64
    )
    assert stats == {"new_rows": 0, "revised_rows": 1, "unchanged_rows": 1}
    assert len(events) == 1
    revised = latest2.iloc[0]
    assert revised["revision"] == "1"
    assert revised["available_at"] == "2026-09-18T04:00:00+00:00"
    assert revised["raw_sha256"] == "b" * 64


def test_write_state_persists_raw_receipts_and_research_gate(tmp_path, monkeypatch):
    hp = history_payload()
    lp = live_payload()
    hraw = json.dumps(hp, separators=(",", ":")).encode()
    lraw = json.dumps(lp, separators=(",", ":")).encode()

    def fake_fetch(url, **kwargs):
        if "history" in url:
            return hp, hraw, {"sha256": eth_supply.hashlib.sha256(hraw).hexdigest(), "x-supply-revision": "7"}
        return lp, lraw, {"sha256": eth_supply.hashlib.sha256(lraw).hexdigest(), "x-supply-revision": "8"}

    monkeypatch.setattr(eth_supply, "fetch_json", fake_fetch)
    report = eth_supply.write_state(tmp_path, receipt_time="2026-09-17T03:00:00Z")
    assert report["status"] == "research_only_rights_unreviewed"
    assert report["public_redistribution"] == "blocked_pending_rights_review"
    assert report["model_use"].startswith("blocked_")
    assert report["history_rows"] == 2
    assert report["identity_nonzero_rows"] == 0
    assert report["live_identity_error_wei"] == "0"
    assert (tmp_path / "eth_supply_history_intervals.csv").exists()
    assert (tmp_path / "eth_supply_history_receipts.csv").exists()
    assert (tmp_path / "eth_supply_live_receipts.csv").exists()
    assert (tmp_path / "raw" / "ethsupply_history_latest.json").read_bytes() == hraw
    assert (tmp_path / "raw" / "ethsupply_live_latest.json").read_bytes() == lraw
