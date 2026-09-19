import io
import json

import pytest

from scripts import export_ethereum_block_headers as exporter


def h(value: int) -> str:
    return "0x" + f"{value:064x}"


class Response:
    def __init__(self, payload):
        self.payload = json.dumps(payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return self.payload


def fake_urlopen(request, timeout=30):
    body = json.loads(request.data)
    number = int(body["params"][0], 16)
    return Response({
        "jsonrpc": "2.0",
        "id": number,
        "result": {
            "number": hex(number),
            "hash": h(number + 1000),
            "parentHash": h(number + 999),
            "timestamp": hex(1_700_000_000 + number),
            "gasUsed": "0x64",
            "gasLimit": "0x3e8",
            "baseFeePerGas": "0x2",
            "transactions": [],
        },
    })


def test_nonlocal_rpc_is_rejected():
    with pytest.raises(ValueError, match="loopback/local"):
        exporter._require_local_rpc("https://example.com")
    assert exporter._require_local_rpc("http://127.0.0.1:8545").startswith("http://127.0.0.1")
    assert exporter._require_local_rpc("http://localhost:8545").startswith("http://localhost")


def test_bounded_export_is_immutable_and_preserves_jsonrpc(monkeypatch, tmp_path):
    monkeypatch.setattr(exporter, "urlopen", fake_urlopen)
    output = tmp_path / "headers.jsonl"
    report = exporter.export_headers("http://127.0.0.1:8545", 100, 102, output)
    assert report["rows"] == 3
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [int(row["result"]["number"], 16) for row in rows] == [100, 101, 102]
    with pytest.raises(FileExistsError, match="immutable"):
        exporter.export_headers("http://127.0.0.1:8545", 100, 102, output)


def test_failed_export_removes_partial_temp(monkeypatch, tmp_path):
    calls = {"n": 0}

    def fail_second(request, timeout=30):
        calls["n"] += 1
        if calls["n"] == 2:
            raise OSError("rpc interrupted")
        return fake_urlopen(request, timeout)

    monkeypatch.setattr(exporter, "urlopen", fail_second)
    output = tmp_path / "headers.jsonl"
    with pytest.raises(OSError, match="interrupted"):
        exporter.export_headers("http://localhost:8545", 100, 102, output)
    assert not output.exists()
    assert not output.with_suffix(".jsonl.tmp").exists()
