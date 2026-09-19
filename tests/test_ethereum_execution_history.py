import json

import pytest

from data_lab.ethereum_execution_history import (
    CANONICAL,
    HISTORY_MODE,
    ORPHANED,
    aggregate_execution,
    canonicalize_chunk,
    join_execution_rows,
    parse_header_row,
    parse_quantity,
    parse_supply_row,
)


def h(value: int) -> str:
    return "0x" + f"{value:064x}"


def supply(number: int, block_hash: int, parent_hash: int, *, base_burn: int = 200, blob: int = 0, misc: int = 0):
    return parse_supply_row({
        "blockNumber": hex(number),
        "hash": h(block_hash),
        "parentHash": h(parent_hash),
        "issuance": {"genesisAlloc": "0x0", "reward": "0x0", "withdrawals": "0x0"},
        "burn": {"1559": hex(base_burn), "blob": hex(blob), "misc": hex(misc)},
    })


def header(number: int, block_hash: int, parent_hash: int, *, timestamp: int, gas_used: int = 100, base_fee: int = 2):
    return parse_header_row({
        "jsonrpc": "2.0",
        "id": number,
        "result": {
            "number": hex(number),
            "hash": h(block_hash),
            "parentHash": h(parent_hash),
            "timestamp": hex(timestamp),
            "gasUsed": hex(gas_used),
            "gasLimit": hex(1000),
            "baseFeePerGas": hex(base_fee),
            "transactions": [h(number)],
        },
    })


def test_quantities_are_exact_and_floats_are_rejected():
    assert parse_quantity("0xde0b6b3a7640000", "wei") == 10**18
    assert parse_quantity("1000000000000000000", "wei") == 10**18
    with pytest.raises(ValueError, match="floating-point"):
        parse_quantity(1.5, "wei")


def test_supply_parser_preserves_split_burn_components():
    row = supply(100, 1000, 999, base_burn=200, blob=30, misc=7)
    assert row["burn_1559_wei"] == 200
    assert row["burn_blob_wei"] == 30
    assert row["burn_misc_wei"] == 7
    assert row["burn_total_wei"] == 237
    assert row["history_mode"] == HISTORY_MODE


def test_canonicalizer_preserves_reorg_orphans_and_follows_unique_high_tip():
    rows = [
        supply(100, 1000, 999),
        supply(101, 1011, 1000),
        supply(102, 1021, 1011),
        supply(101, 1012, 1000),
        supply(102, 1022, 1012),
        supply(103, 1032, 1022),
    ]
    result = canonicalize_chunk(rows)
    status = {row["block_hash"]: row["canonical_status"] for row in result["rows"]}
    assert result["tip_block_number"] == 103
    assert result["tip_hash"] == h(1032)
    assert result["boundary_parent_hash"] == h(999)
    assert result["canonical_rows"] == 4
    assert result["orphaned_rows"] == 2
    assert status[h(1011)] == ORPHANED
    assert status[h(1021)] == ORPHANED
    assert status[h(1012)] == CANONICAL
    assert status[h(1032)] == CANONICAL


def test_canonicalizer_fails_on_competing_equal_height_tips():
    rows = [
        supply(100, 1000, 999),
        supply(101, 1011, 1000),
        supply(101, 1012, 1000),
    ]
    with pytest.raises(ValueError, match="ambiguous highest-height tips"):
        canonicalize_chunk(rows)


def test_join_requires_full_identity_and_reconciles_eip1559_burn():
    supplies = [supply(100, 1000, 999), supply(101, 1011, 1000)]
    headers = [
        header(100, 1000, 999, timestamp=1_700_000_000),
        header(101, 1011, 1000, timestamp=1_700_000_012),
    ]
    joined = join_execution_rows(supplies, headers)
    assert [row["block_number"] for row in joined] == [100, 101]
    assert all(row["burn_1559_header_reconciliation_wei"] == 0 for row in joined)
    assert all(row["available_at"] is None and row["model_eligible"] is False for row in joined)


def test_join_fails_closed_on_burn_mismatch_or_missing_canonical_header():
    supplies = [supply(100, 1000, 999), supply(101, 1011, 1000)]
    wrong = [
        header(100, 1000, 999, timestamp=1_700_000_000),
        header(101, 1011, 1000, timestamp=1_700_000_012, gas_used=101),
    ]
    with pytest.raises(ValueError, match="EIP-1559 burn mismatch"):
        join_execution_rows(supplies, wrong)
    with pytest.raises(ValueError, match="missing same-node block header"):
        join_execution_rows(supplies, wrong[:1])


def test_hourly_aggregation_keeps_exact_large_integer_sums():
    joined = join_execution_rows(
        [supply(100, 1000, 999, base_burn=10**20), supply(101, 1011, 1000, base_burn=10**20)],
        [
            header(100, 1000, 999, timestamp=1_700_000_000, gas_used=100, base_fee=10**18),
            header(101, 1011, 1000, timestamp=1_700_000_012, gas_used=100, base_fee=10**18),
        ],
    )
    hourly = aggregate_execution(joined, "hour")
    assert len(hourly) == 1
    assert hourly[0]["burn_1559_wei"] == 2 * 10**20
    assert isinstance(hourly[0]["burn_1559_wei"], int)
    assert hourly[0]["block_count"] == 2
    assert hourly[0]["tx_count"] == 2
    assert hourly[0]["available_at"] is None
