import unittest

from data_lab.eth_execution_network import (
    ContinuityError,
    collect_finalized_window,
    validate_rpc_url,
)


class FakeRpc:
    def __init__(self, start=95, end=104):
        self.blocks = {number: self._block(number) for number in range(start, end + 2)}
        self.finalized = end

    @staticmethod
    def _hash(number):
        return "0x" + f"{number:064x}"

    def _block(self, number):
        return {
            "number": hex(number),
            "hash": self._hash(number),
            "parentHash": self._hash(number - 1),
            "timestamp": hex(1_800_000_000 + number * 12),
            "gasLimit": hex(30_000_000),
            "gasUsed": hex(15_000_000),
            "baseFeePerGas": hex(1_000 + number),
            "blobGasUsed": hex(262_144),
            "excessBlobGas": hex(number * 131_072),
            "transactions": [self._hash(number * 10), self._hash(number * 10 + 1)],
        }

    def call(self, _url, method, params):
        if method == "eth_chainId":
            return "0x1"
        if method == "eth_getBlockByNumber":
            assert params == ["finalized", False]
            return self.blocks[self.finalized]
        if method == "eth_feeHistory":
            count = int(params[0], 16)
            newest = int(params[1], 16)
            oldest = newest - count + 1
            assert params[2] == [10, 50, 90]
            return {
                "oldestBlock": hex(oldest),
                "baseFeePerGas": [hex(1_000 + n) for n in range(oldest, newest + 2)],
                "gasUsedRatio": [0.5 for _ in range(count)],
                "baseFeePerBlobGas": [hex(10 + n) for n in range(oldest, newest + 2)],
                "blobGasUsedRatio": [0.5 for _ in range(count)],
                "reward": [[hex(1), hex(2), hex(3)] for _ in range(count)],
            }
        raise AssertionError(method)

    def batch(self, _url, numbers):
        return [self.blocks[int(number)] for number in numbers]


class EthereumExecutionNetworkTests(unittest.TestCase):
    def test_first_run_bootstrap_is_receipt_time_not_historical_pit(self):
        fake = FakeRpc(end=104)
        result = collect_finalized_window(
            "https://rpc.example",
            max_blocks=3,
            receipt_time="2026-09-18T00:00:00+00:00",
            call_fn=fake.call,
            blocks_fn=fake.batch,
        )
        self.assertEqual(result["start_block"], 102)
        self.assertEqual(result["end_block"], 104)
        self.assertEqual(len(result["rows"]), 3)
        first = result["rows"][0]
        self.assertEqual(first["receipt_kind"], "bootstrap_reconstructed_finalized_window")
        self.assertEqual(first["available_at"], "2026-09-18T00:00:00+00:00")
        self.assertFalse(first["historical_pit_eligible"])
        self.assertFalse(first["prospective_candidate"])
        self.assertEqual(first["gas_used_ratio"], 0.5)
        self.assertEqual(first["execution_base_fee_burn_wei"], (1_000 + 102) * 15_000_000)
        self.assertEqual(first["blob_base_fee_burn_wei"], (10 + 102) * 262_144)
        self.assertEqual(first["priority_fee_p50_wei"], 2)
        self.assertEqual(first["transaction_count"], 2)
        self.assertEqual(len(result["raw_sha256"]), 64)

    def test_continuation_is_strictly_after_prior_finalized_hash(self):
        fake = FakeRpc(end=104)
        result = collect_finalized_window(
            "https://rpc.example",
            prior_block_number=102,
            prior_block_hash=fake.blocks[102]["hash"],
            max_blocks=10,
            receipt_time="2026-09-18T00:10:00Z",
            call_fn=fake.call,
            blocks_fn=fake.batch,
        )
        self.assertEqual([row["block_number"] for row in result["rows"]], [103, 104])
        self.assertTrue(all(row["prospective_candidate"] for row in result["rows"]))
        self.assertEqual(result["rows"][0]["parent_hash"], fake.blocks[102]["hash"])

    def test_prospective_gap_beyond_bound_fails_closed(self):
        fake = FakeRpc(end=104)
        with self.assertRaises(ContinuityError):
            collect_finalized_window(
                "https://rpc.example",
                prior_block_number=95,
                prior_block_hash=fake.blocks[95]["hash"],
                max_blocks=3,
                call_fn=fake.call,
                blocks_fn=fake.batch,
            )

    def test_conflicting_parent_hash_fails(self):
        fake = FakeRpc(end=104)
        fake.blocks[103]["parentHash"] = "0x" + "f" * 64
        with self.assertRaises(ContinuityError):
            collect_finalized_window(
                "https://rpc.example",
                prior_block_number=102,
                prior_block_hash=fake.blocks[102]["hash"],
                max_blocks=10,
                call_fn=fake.call,
                blocks_fn=fake.batch,
            )

    def test_remote_rpc_requires_https(self):
        with self.assertRaises(ValueError):
            validate_rpc_url("http://rpc.example")
        validate_rpc_url("https://rpc.example")
        validate_rpc_url("http://127.0.0.1:8545")


if __name__ == "__main__":
    unittest.main()
