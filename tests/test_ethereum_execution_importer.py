import json

import pytest

from scripts.import_ethereum_execution_history import import_chunk


def h(value: int) -> str:
    return "0x" + f"{value:064x}"


def supply_raw(number: int, block_hash: int, parent_hash: int, burn: int = 200):
    return {
        "blockNumber": hex(number),
        "hash": h(block_hash),
        "parentHash": h(parent_hash),
        "burn": {"1559": hex(burn), "blob": "0x0", "misc": "0x0"},
    }


def header_raw(number: int, block_hash: int, parent_hash: int, timestamp: int):
    return {
        "jsonrpc": "2.0",
        "id": number,
        "result": {
            "number": hex(number),
            "hash": h(block_hash),
            "parentHash": h(parent_hash),
            "timestamp": hex(timestamp),
            "gasUsed": hex(100),
            "gasLimit": hex(1000),
            "baseFeePerGas": hex(2),
            "transactions": [h(number)],
        },
    }


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_importer_preserves_orphan_and_blocks_model_use(tmp_path):
    supply = tmp_path / "supply.jsonl"
    headers = tmp_path / "headers.jsonl"
    output = tmp_path / "out"
    write_jsonl(supply, [
        supply_raw(100, 1000, 999),
        supply_raw(101, 1011, 1000),
        supply_raw(101, 1012, 1000),
        supply_raw(102, 1022, 1012),
    ])
    write_jsonl(headers, [
        header_raw(100, 1000, 999, 1_700_000_000),
        header_raw(101, 1012, 1000, 1_700_000_012),
        header_raw(102, 1022, 1012, 1_700_000_024),
    ])
    manifest = import_chunk(supply, headers, output)
    assert manifest["canonical_rows"] == 3
    assert manifest["orphaned_rows"] == 1
    assert manifest["joined_canonical_rows"] == 3
    assert manifest["model_use"] == "blocked"
    assert manifest["availability_contract"] == "not_yet_frozen"
    assert (output / "execution_canonical.jsonl").is_file()
    assert (output / "execution_hourly.jsonl").is_file()
    assert (output / "execution_daily.jsonl").is_file()
    assert "blocked" in (output / "REPORT.md").read_text()
    rows = [json.loads(line) for line in (output / "execution_canonical.jsonl").read_text().splitlines()]
    assert all(row["available_at"] is None for row in rows)
    assert all(row["model_eligible"] is False for row in rows)


def test_importer_output_is_immutable(tmp_path):
    supply = tmp_path / "supply.jsonl"
    headers = tmp_path / "headers.jsonl"
    output = tmp_path / "out"
    write_jsonl(supply, [supply_raw(100, 1000, 999)])
    write_jsonl(headers, [header_raw(100, 1000, 999, 1_700_000_000)])
    import_chunk(supply, headers, output)
    with pytest.raises(FileExistsError, match="immutable"):
        import_chunk(supply, headers, output)
