import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from data_lab import eth_staking
from scripts.collect_eth_staking import collect


def node(value, *, status="exact", kind="observed", as_of=1789623091, source="test:source"):
    return {"value": str(value), "status": status, "kind": kind, "asOf": as_of, "sources": [source]}


def live_payload():
    return {
        "schemaVersion": 1,
        "revision": 1789623097319,
        "generatedAt": 1789623097,
        "chainId": 1,
        "head": {"slot": 15233255},
        "finalized": {"slot": 15233200},
        "staking": {
            "activeBalanceGwei": node(43399867133096350, status="approximate"),
            "effectiveBalanceGwei": node(43334716000000000),
            "activeValidators": node(911044),
            "pendingDeposits": {
                "validators": node(25000, as_of=1789623008),
                "balanceGwei": node(1836229784343018, kind="derived", as_of=1789623008),
                "slotZeroBalanceGwei": node(18603225822, kind="derived", as_of=1789623008),
            },
            "scheduledActivations": {
                "validators": node(4),
                "balanceGwei": node(356000000000, kind="derived"),
            },
            "scheduledExits": {
                "validators": node(87),
                "balanceGwei": node(19051817993437, kind="derived"),
            },
            "pendingPartialWithdrawals": {
                "requests": node(36),
                "balanceGwei": node(970838708431, kind="derived"),
            },
            "pendingConsolidations": {
                "requests": node(106),
                "balanceGwei": node(3392000000000, kind="derived"),
            },
            "churn": {
                "activationExitGwei": node(256000000000, kind="derived"),
                "consolidationGwei": node(405000000000, kind="derived"),
            },
        },
    }


class EthStakingTests(unittest.TestCase):
    def test_parse_live_staking_preserves_source_native_state(self):
        row = eth_staking.parse_live_staking(live_payload())
        self.assertEqual(row["active_validators"], "911044")
        self.assertEqual(row["active_balance_eth"], "43399867.13309635")
        self.assertEqual(row["effective_balance_eth"], "43334716")
        self.assertEqual(row["active_balance_gwei_status"], "approximate")
        self.assertEqual(row["pending_deposits_validators"], "25000")
        self.assertEqual(row["scheduled_exits_validators"], "87")
        self.assertEqual(row["pending_partial_withdrawals_requests"], "36")
        self.assertEqual(row["pending_consolidations_requests"], "106")
        self.assertEqual(row["staking_as_of_min"], "2026-09-17T05:30:08+00:00")
        self.assertEqual(row["staking_as_of_max"], "2026-09-17T05:31:31+00:00")

    def test_missing_metric_fails_closed(self):
        payload = live_payload()
        del payload["staking"]["scheduledExits"]
        with self.assertRaisesRegex(ValueError, "missing staking field"):
            eth_staking.parse_live_staking(payload)

    def test_effective_balance_cannot_exceed_active_balance(self):
        payload = live_payload()
        payload["staking"]["effectiveBalanceGwei"]["value"] = str(50000000000000000)
        with self.assertRaisesRegex(ValueError, "effective staking balance"):
            eth_staking.parse_live_staking(payload)

    def test_collect_reuses_exact_supply_live_receipt(self):
        payload = live_payload()
        raw = json.dumps(payload, separators=(",", ":")).encode()
        digest = hashlib.sha256(raw).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "ethsupply_live_latest.json").write_bytes(raw)
            supply = {
                "received_at": "2026-09-17T05:31:42.932879+00:00",
                "available_at": "2026-09-17T05:31:42.932879+00:00",
                "raw_sha256": digest,
                "header_revision": "1789623097319",
                "header_generated_at": "2026-09-17T05:31:37Z",
                "source_revision": 1789623097319,
            }
            (root / "eth_supply_live_receipts.jsonl").write_text(json.dumps(supply) + "\n")
            report = collect(root)
            self.assertEqual(report["history_mode"], "prospective_receipts_only")
            self.assertEqual(report["live_receipt_rows"], 1)
            self.assertEqual(report["raw_sha256"], digest)
            self.assertEqual(report["active_validators"], "911044")
            row = json.loads((root / "eth_staking_live_receipts.jsonl").read_text().strip())
            self.assertEqual(row["received_at"], supply["received_at"])
            self.assertEqual(row["available_at"], supply["received_at"])
            self.assertEqual(row["raw_sha256"], digest)

    def test_shared_raw_sha_mismatch_fails_closed(self):
        payload = live_payload()
        raw = json.dumps(payload).encode()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "raw").mkdir()
            (root / "raw" / "ethsupply_live_latest.json").write_bytes(raw)
            supply = {
                "received_at": "2026-09-17T05:31:42+00:00",
                "raw_sha256": "0" * 64,
                "source_revision": 1789623097319,
            }
            (root / "eth_supply_live_receipts.jsonl").write_text(json.dumps(supply) + "\n")
            with self.assertRaisesRegex(ValueError, "SHA256"):
                collect(root)


if __name__ == "__main__":
    unittest.main()
