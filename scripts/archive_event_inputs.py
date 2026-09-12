"""Preserve public input vintages and referenced checkpoints beside the audit ledger."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def put_blob(path, data):
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("archive content mismatch: " + path.name)
    else:
        atomic(path, data)


def archive(root, destination):
    root, destination = Path(root), Path(destination)
    database = root / "observations.db"
    if not database.is_file():
        raise ValueError("missing observations database")
    with sqlite3.connect(database.as_uri() + "?mode=ro", uri=True) as con:
        con.row_factory = sqlite3.Row
        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise ValueError("invalid observations database")
        rows = [dict(r) for r in con.execute("SELECT * FROM bars")]
    partitions, raw_hashes = {}, set()
    for row in rows:
        if row["product"] not in ("ETH-USD", "BTC-USD"):
            raise ValueError("unexpected archived product")
        observed = datetime.fromisoformat(row["observed_at"])
        if observed.tzinfo is None:
            raise ValueError("source receipt lacks timezone")
        day = observed.astimezone(timezone.utc).date().isoformat()
        partitions.setdefault(day, []).append(row)
        raw_hashes.add(row["raw_hash"])
    # Validate before committing any references to a missing or corrupt raw receipt.
    for digest in sorted(raw_hashes):
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid raw hash")
        source = root / "raw" / (digest + ".json.gz")
        target = destination / "raw" / source.name
        raw = gzip.decompress(source.read_bytes() if source.is_file() else target.read_bytes())
        if hashlib.sha256(raw).hexdigest() != digest:
            raise ValueError("raw receipt hash mismatch")
        put_blob(target, gzip.compress(raw, mtime=0))
    for day, incoming in partitions.items():
        target = destination / "observations" / (day + ".jsonl.gz")
        previous = [json.loads(line) for line in gzip.decompress(target.read_bytes()).splitlines()] if target.exists() else []
        union = {}
        for row in previous + incoming:
            key = (row["product"], row["open_time"], row["observed_at"], row["content_hash"])
            # Revision numbers can differ between independent research/hourly stores.
            # Keep the original occurrence; values and receipt time must agree.
            value = {k: v for k, v in row.items() if k != "revision"}
            if key in union and union[key] != value:
                raise ValueError("conflicting source vintage")
            union[key] = value
        data = "\n".join(canonical(union[k]) for k in sorted(union)) + "\n"
        atomic(target, gzip.compress(data.encode(), mtime=0))
    models = {}
    for name, folder in (("active.json", "models"), ("shadow_active.json", "shadow_models")):
        source = root / name
        if not source.exists():
            continue
        for horizon, entry in json.loads(source.read_text()).items():
            filename = entry["file"]
            if Path(filename).name != filename or not filename.endswith(".joblib"):
                raise ValueError("invalid model filename")
            data = (root / folder / filename).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("checkpoint hash mismatch")
            target = destination / "models" / (digest + ".joblib")
            put_blob(target, data)
            models[folder + "/" + horizon] = {"source_file": filename, "sha256": digest}
    manifest = {"schema_version": 1, "archived_at": datetime.now(timezone.utc).isoformat(),
                "source": "coinbase_exchange", "snapshot_rows": len(rows),
                "partitions": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                                       "bytes": p.stat().st_size}
                               for p in sorted((destination / "observations").glob("*.jsonl.gz"))},
                "active_models": models,
                "note": "Observed receipts and source values; historical backfill is not historical availability. Full research state remains separately retained."}
    atomic(destination / "manifest.json", (canonical(manifest) + "\n").encode())
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = archive(args.root.resolve(), args.destination.resolve())
    print(f"Source audit preserved: {result['snapshot_rows']} rows, {len(result['partitions'])} receipt dates, {len(result['active_models'])} active models")
