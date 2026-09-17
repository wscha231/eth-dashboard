import json
from pathlib import Path
import tempfile
import unittest

from data_lab.liquidation_cascade import (
    HISTORY_MODE,
    MODEL_USE,
    PUBLIC_REDISTRIBUTION,
    SITE_USE,
    parse_bybit_message,
    persist_session,
)


class LiquidationCascadeTests(unittest.TestCase):
    def test_subscription_ack(self):
        ack, rows = parse_bybit_message(
            {"op": "subscribe", "success": True, "ret_msg": "", "conn_id": "abc"},
            received_at="2026-09-17T08:00:00+00:00",
        )
        self.assertTrue(ack)
        self.assertEqual(rows, [])

    def test_parse_preserves_bybit_side_semantics_and_receipt_time(self):
        message = {
            "topic": "allLiquidation.ETHUSDT",
            "type": "snapshot",
            "ts": 1789632000500,
            "data": [
                {"T": 1789632000400, "s": "ETHUSDT", "S": "Buy", "v": "2.5", "p": "4010.25"},
                {"T": 1789632000450, "s": "ETHUSDT", "S": "Sell", "v": "1.0", "p": "4020"},
            ],
        }
        ack, rows = parse_bybit_message(message, received_at="2026-09-17T08:00:00.700000+00:00")
        self.assertFalse(ack)
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["liquidated_position"], "long")
        self.assertEqual(rows[1]["liquidated_position"], "short")
        self.assertEqual(rows[0]["available_at"], "2026-09-17T08:00:00.700000+00:00")
        self.assertEqual(rows[0]["received_at"], rows[0]["available_at"])
        self.assertEqual(rows[0]["notional_usdt"], "10025.625")
        self.assertTrue(rows[0]["notional_is_derived"])
        self.assertEqual(len(rows[0]["event_id"]), 64)

    def test_wrong_symbol_fails_closed(self):
        message = {
            "topic": "allLiquidation.ETHUSDT",
            "ts": 1789632000500,
            "data": [{"T": 1789632000400, "s": "BTCUSDT", "S": "Buy", "v": "2", "p": "1"}],
        }
        with self.assertRaisesRegex(ValueError, "unexpected liquidation symbol"):
            parse_bybit_message(message, received_at="2026-09-17T08:00:00+00:00")

    def test_nonpositive_values_fail_closed(self):
        message = {
            "topic": "allLiquidation.ETHUSDT",
            "ts": 1789632000500,
            "data": [{"T": 1789632000400, "s": "ETHUSDT", "S": "Buy", "v": "0", "p": "4010"}],
        }
        with self.assertRaisesRegex(ValueError, "executed size must be positive"):
            parse_bybit_message(message, received_at="2026-09-17T08:00:00+00:00")

    def test_persist_deduplicates_exact_events_and_keeps_empty_session_evidence(self):
        message = {
            "topic": "allLiquidation.ETHUSDT",
            "ts": 1789632000500,
            "data": [{"T": 1789632000400, "s": "ETHUSDT", "S": "Buy", "v": "2", "p": "4000"}],
        }
        _, rows = parse_bybit_message(message, received_at="2026-09-17T08:00:00.700000+00:00")
        session = {
            "schema": 1,
            "source_id": "bybit_v5_all_liquidation_ethusdt",
            "topic": "allLiquidation.ETHUSDT",
            "started_at": "2026-09-17T08:00:00+00:00",
            "ended_at": "2026-09-17T08:01:00+00:00",
            "planned_duration_seconds": 60,
            "actual_duration_seconds": 60.0,
            "subscription_acknowledged": True,
            "complete": True,
            "connections": 1,
            "disconnects": 0,
            "parse_errors": 0,
            "event_count": 1,
        }
        with tempfile.TemporaryDirectory() as tmp:
            report = persist_session(tmp, events=rows + rows, session=session)
            self.assertEqual(report["unique_event_rows"], 1)
            self.assertEqual(report["total_liquidation_notional_usdt"], "8000")
            self.assertEqual(report["long_liquidation_notional_usdt"], "8000")
            self.assertEqual(report["short_liquidation_notional_usdt"], "0")
            report2 = persist_session(tmp, events=[], session={**session, "event_count": 0, "started_at": "2026-09-17T09:00:00+00:00", "ended_at": "2026-09-17T09:01:00+00:00"})
            self.assertEqual(report2["unique_event_rows"], 1)
            self.assertEqual(report2["capture_sessions"], 2)
            self.assertEqual(report2["complete_capture_sessions"], 2)
            sessions = [json.loads(x) for x in Path(tmp, "capture_sessions.jsonl").read_text().splitlines()]
            self.assertEqual(len(sessions), 2)

    def test_research_boundaries_remain_blocked(self):
        self.assertEqual(HISTORY_MODE, "prospective_receipts_only")
        self.assertTrue(PUBLIC_REDISTRIBUTION.startswith("blocked_"))
        self.assertTrue(MODEL_USE.startswith("blocked_"))
        self.assertEqual(SITE_USE, "blocked")


if __name__ == "__main__":
    unittest.main()
