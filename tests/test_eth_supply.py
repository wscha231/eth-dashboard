import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from data_lab import eth_supply


def data_value(value, generated):
    return {"value": str(value), "status": "exact", "kind": "derived", "asOf": generated - 300, "sources": []}


def tracked_payload(revision=8, generated=1789617600):
    return {
        "schemaVersion": 1,
        "revision": revision,
        "generatedAt": generated,
        "range": "retained",
        "summary": {
            "fromSlot": 1000,
            "toSlot": 2000,
            "fromTimestamp": generated - 100000,
            "toTimestamp": generated - 20,
            "slots": 1001,
            "blocks": 990,
            "issuanceWei": "5000000000000000000000",
            "burnWei": "300000000000000000000",
            "baseFeeBurnWei": "270000000000000000000",
            "blobBaseFeeBurnWei": "30000000000000000000",
            "consensusPenaltiesWei": "50000000000000000000",
            "otherExecutionBurnWei": "20000000000000000000",
            "netWei": "4630000000000000000000",
            "gasTargetTotal": "1",
            "offsetBaseFeeWei": "0",
        },
        "warnings": [],
        "finalizedToSlot": 1990,
        "liveTailSlots": 10,
    }


def live_payload(revision=9, generated=1789617600):
    return {
        "schemaVersion": 1,
        "revision": revision,
        "generatedAt": generated,
        "chainId": 1,
        "head": {"block": 25000000, "slot": 15000000, "timestamp": generated - 10},
        "finalized": {"block": 24999900, "epoch": 500000, "slot": 14999900},
        "supply": {
            "totalWei": {"value": "122005400000000000000000000", "status": "exact", "kind": "observed", "asOf": generated - 12, "sources": []},
            "finalizedTotalWei": {"value": "122005399000000000000000000", "status": "exact", "kind": "observed", "asOf": generated - 300, "sources": []},
            "asOf": {"block": 25000000, "slot": 15000000, "epoch": 500001, "timestamp": generated - 12},
        },
        "accounting": {
            "interval": "finalized_epoch",
            "epoch": 500000,
            "targetEpoch": 500000,
            "issuance": {"totalWei": data_value("1200000000000000000", generated)},
            "burn": {
                "totalWei": data_value("200000000000000000", generated),
                "baseFeeWei": data_value("180000000000000000", generated),
                "blobBaseFeeWei": data_value("10000000000000000", generated),
                "otherExecutionWei": data_value("10000000000000000", generated),
                "consensus": {"totalWei": data_value("50000000000000000", generated)},
            },
            "netWei": data_value("950000000000000000", generated),
        },
    }


class EthSupplyTests(unittest.TestCase):
    def test_tracked_parses_full_protocol_accounting_identity(self):
        row = eth_supply.parse_tracked(tracked_payload())
        self.assertEqual(row["from_slot"], 1000)
        self.assertEqual(row["to_slot"], 2000)
        self.assertEqual(row["issuance_eth"], "5000")
        self.assertEqual(row["burn_eth"], "300")
        self.assertEqual(row["consensus_penalties_eth"], "50")
        self.assertEqual(row["other_execution_burn_eth"], "20")
        self.assertEqual(row["net_eth"], "4630")
        self.assertEqual(row["execution_burn_component_error_wei"], "0")
        self.assertEqual(row["identity_error_wei"], "0")
        self.assertEqual(row["accounting_equation"], eth_supply.TRACKED_EQUATION)

    def test_live_parses_consensus_penalty_separately_from_execution_burn(self):
        row = eth_supply.parse_live(live_payload())
        self.assertEqual(row["total_supply_eth"], "122005400")
        self.assertEqual(row["issuance_eth"], "1.2")
        self.assertEqual(row["burn_eth"], "0.2")
        self.assertEqual(row["consensus_penalties_eth"], "0.05")
        self.assertEqual(row["net_eth"], "0.95")
        self.assertEqual(row["execution_burn_component_error_wei"], "0")
        self.assertEqual(row["identity_error_wei"], "0")
        self.assertEqual(row["accounting_equation"], eth_supply.LIVE_EQUATION)
        self.assertEqual(row["total_supply_status"], "exact")

    def test_write_state_is_prospective_append_only_and_research_blocked(self):
        tp, lp = tracked_payload(), live_payload()
        traw = json.dumps(tp, separators=(",", ":")).encode()
        lraw = json.dumps(lp, separators=(",", ":")).encode()

        def fake_fetch(url, **kwargs):
            if "tracked" in url:
                return tp, traw, {"sha256": hashlib.sha256(traw).hexdigest(), "x-supply-revision": "8"}
            return lp, lraw, {"sha256": hashlib.sha256(lraw).hexdigest(), "x-supply-revision": "9"}

        with tempfile.TemporaryDirectory() as tmp, patch.object(eth_supply, "fetch_json", side_effect=fake_fetch):
            root = Path(tmp)
            first = eth_supply.write_state(root, receipt_time="2026-09-17T03:00:00Z")
            second = eth_supply.write_state(root, receipt_time="2026-09-17T09:00:00Z")

            self.assertEqual(first["history_mode"], "prospective_receipts_only")
            self.assertEqual(first["status"], "research_only_rights_unreviewed")
            self.assertEqual(first["public_redistribution"], "blocked_pending_rights_review")
            self.assertTrue(first["model_use"].startswith("blocked_"))
            self.assertEqual(first["tracked_execution_burn_component_error_wei"], "0")
            self.assertEqual(first["live_execution_burn_component_error_wei"], "0")
            self.assertEqual(first["tracked_identity_error_wei"], "0")
            self.assertEqual(first["live_identity_error_wei"], "0")
            self.assertEqual(second["tracked_receipt_rows"], 2)
            self.assertEqual(second["live_receipt_rows"], 2)

            tracked_lines = [line for line in (root / "eth_supply_tracked_receipts.jsonl").read_text().splitlines() if line]
            live_lines = [line for line in (root / "eth_supply_live_receipts.jsonl").read_text().splitlines() if line]
            self.assertEqual(len(tracked_lines), 2)
            self.assertEqual(len(live_lines), 2)
            self.assertIn('"available_at":"2026-09-17T03:00:00+00:00"', tracked_lines[0])
            self.assertIn('"available_at":"2026-09-17T09:00:00+00:00"', tracked_lines[1])

            self.assertEqual((root / "raw" / "ethsupply_tracked_latest.json").read_bytes(), traw)
            self.assertEqual((root / "raw" / "ethsupply_live_latest.json").read_bytes(), lraw)
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(state["collector_started_at"], "2026-09-17T03:00:00+00:00")
            self.assertEqual(state["last_received_at"], "2026-09-17T09:00:00+00:00")

    def test_nonzero_identity_is_preserved_for_gate_not_silently_fixed(self):
        payload = tracked_payload()
        payload["summary"]["netWei"] = "4630000000000000000001"
        row = eth_supply.parse_tracked(payload)
        self.assertEqual(row["identity_error_wei"], "1")

    def test_execution_burn_component_error_is_preserved(self):
        payload = live_payload()
        payload["accounting"]["burn"]["totalWei"]["value"] = "200000000000000001"
        row = eth_supply.parse_live(payload)
        self.assertEqual(row["execution_burn_component_error_wei"], "1")
        self.assertEqual(row["identity_error_wei"], "1")


if __name__ == "__main__":
    unittest.main()
