#!/usr/bin/env python3
"""Register R3 research artifacts with the existing immutable Drive archive."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tempfile

# When this file is invoked as `python scripts/archive_r3_research_to_drive.py`,
# Python places `scripts/` rather than the repository root on sys.path. Add the
# checkout root explicitly before importing the package-style `scripts.*` modules.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import gdrive_store as storage

R3_STREAMS = {
    "eth-etf-research",
    "eth-supply-research",
    "eth-liquidation-research",
    "eth-execution-network-research",
    # Keep reconstructed/long derivative history and fast prospective context
    # in separate streams. A stream has one immutable watermark; sharing it
    # across independent workflows could make a newer run hide another source.
    "derivative-history-research",
    "derivative-fast-research",
    # One pre-cleanup snapshot seals every currently public derivative cache,
    # including legacy Deribit files produced outside the two P1 collectors.
    "derivative-public-seed",
    # Distinct legacy Deribit bytes from data/event-feed are preserved under
    # their own source-commit identity.
    "derivative-event-feed-seed",
}
R3_ARTIFACTS = {
    "eth-etf-flow-state": ("eth-etf-research", "eth_etf_flows.yml"),
    "eth-supply-research-state": ("eth-supply-research", "eth_supply_research.yml"),
    "eth-liquidation-research-state": ("eth-liquidation-research", "eth_liquidation_research.yml"),
    "eth-execution-network-research-state": (
        "eth-execution-network-research",
        "eth_execution_network_research.yml",
    ),
    # This artifact intentionally excludes FRED and other mixed fallback data.
    "derivative-history-research-state": (
        "derivative-history-research",
        "free_source_backfill.yml",
    ),
    "derivative-history-reconciliation-state": (
        "derivative-history-research",
        "data_reconciliation.yml",
    ),
    "fast-derivative-state": (
        "derivative-fast-research",
        "fast_derivative_snapshots.yml",
    ),
    "derivative-public-seed-state": (
        "derivative-public-seed",
        "derivative_private_archive_seed.yml",
    ),
}
# This Actions artifact is deliberately manifest-only. It must never be added to
# archive_to_drive.ARTIFACTS because the raw restricted bytes are fetched from
# the exact pinned data/event-feed commit and written straight to private Drive.
R3_PRIVATE_RECEIPTS = {
    "derivative-event-feed-seed-state": (
        "derivative-event-feed-seed",
        "derivative_private_archive_seed.yml",
    ),
}
EVENT_FEED_SEED_ARTIFACT = "derivative-event-feed-seed-state"
EVENT_FEED_SEED_STREAM = "derivative-event-feed-seed"
EVENT_FEED_SEED_WORKFLOW = "derivative_private_archive_seed.yml"
EVENT_FEED_SEED_MANIFEST = "lake/reports/derivative_event_feed_seed_manifest.json"
EVENT_FEED_SEED_FILES = frozenset({
    "lake/raw/vendor/deribit_eth_funding_daily.csv",
    "lake/raw/vendor/deribit_eth_future_snapshot_daily.csv",
    "lake/raw/vendor/deribit_eth_historical_volatility.csv",
    "lake/raw/vendor/deribit_eth_option_snapshot_daily.csv",
})


def register() -> None:
    # Mutate the shared set before archive_to_drive validates stream names.
    storage.STREAMS.update(R3_STREAMS)
    from scripts import archive_to_drive as archive
    archive.STREAMS.update(R3_STREAMS)
    archive.ARTIFACTS.update(R3_ARTIFACTS)


def _event_feed_seed_artifacts(github, source_run: int | None) -> list[dict]:
    if source_run:
        source = github.run(source_run)
        if source.get("path") != ".github/workflows/" + EVENT_FEED_SEED_WORKFLOW:
            return []
        path = f"/actions/runs/{int(source_run)}/artifacts?name={EVENT_FEED_SEED_ARTIFACT}&per_page=100"
    else:
        path = f"/actions/artifacts?name={EVENT_FEED_SEED_ARTIFACT}&per_page=100"
    return github.get(path).get("artifacts", [])


def _validate_event_feed_manifest(document: dict) -> tuple[str, dict[str, dict]]:
    if (
        document.get("schema") != 1
        or document.get("status") != "research_only_private_archive_seed"
        or document.get("source_branch") != "data/event-feed"
        or document.get("file_count") != 4
        or document.get("public_redistribution") != "blocked"
        or document.get("model_use") != "blocked_pending_rights_storage_pit_audit_and_preregistered_gate"
        or document.get("site_use") != "blocked"
        or document.get("paid_product_use") != "blocked"
    ):
        raise storage.CheckError("Invalid event-feed seed receipt contract")
    commit = document.get("source_commit")
    if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise storage.CheckError("Invalid event-feed source commit")
    rows = document.get("files")
    if not isinstance(rows, list) or len(rows) != 4:
        raise storage.CheckError("Invalid event-feed seed file list")
    files = {}
    for row in rows:
        if not isinstance(row, dict):
            raise storage.CheckError("Invalid event-feed seed file row")
        path, size, sha = row.get("path"), row.get("bytes"), row.get("sha256")
        if (
            path not in EVENT_FEED_SEED_FILES
            or path in files
            or not isinstance(size, int)
            or size <= 0
            or not isinstance(sha, str)
            or not re.fullmatch(r"[0-9a-f]{64}", sha)
        ):
            raise storage.CheckError("Invalid event-feed seed file identity")
        files[path] = {"bytes": size, "sha256": sha}
    if set(files) != EVENT_FEED_SEED_FILES:
        raise storage.CheckError("Event-feed seed receipt does not cover the approved four files")
    return commit, files


def archive_event_feed_seed(source_run: int | None = None) -> dict | None:
    """Use a safe manifest artifact to privately archive exact bytes from its pinned Git commit."""
    from scripts import archive_to_drive as archive

    github = archive.GitHub()
    artifacts = _event_feed_seed_artifacts(github, source_run)
    if source_run and not artifacts:
        source = github.run(source_run)
        if source.get("path") == ".github/workflows/" + EVENT_FEED_SEED_WORKFLOW:
            raise storage.CheckError("Trusted event-feed seed receipt artifact is missing")
        return None

    for artifact in artifacts:
        if artifact.get("expired"):
            continue
        run_id = artifact.get("workflow_run", {}).get("id")
        if not run_id:
            continue
        run = github.run(run_id)
        if not archive.trusted(run, EVENT_FEED_SEED_WORKFLOW, github.repo_id) or run.get("conclusion") != "success":
            continue
        with tempfile.TemporaryDirectory(prefix="r3-event-feed-seed-") as temporary:
            temporary = Path(temporary)
            zipped = temporary / "receipt.zip"
            github.download(artifact["id"], zipped)
            extracted = temporary / "receipt"
            extracted.mkdir()
            archive.extract_zip(zipped, extracted)
            present = {
                path.relative_to(extracted).as_posix()
                for path in extracted.rglob("*")
                if path.is_file()
            }
            if present != {EVENT_FEED_SEED_MANIFEST}:
                raise storage.CheckError("Event-feed seed Actions artifact must be manifest-only")
            manifest_path = extracted / EVENT_FEED_SEED_MANIFEST
            raw_manifest = manifest_path.read_bytes()
            document = json.loads(raw_manifest)
            source_commit, expected = _validate_event_feed_manifest(document)

            # Re-read the exact Git object named by the trusted receipt. Restricted
            # bytes never transit a public Actions artifact.
            archive.git("fetch", "--depth=1", "origin", source_commit)
            payload = temporary / "payload"
            for path, info in sorted(expected.items()):
                archive.git("cat-file", "-e", f"{source_commit}:{path}")
                raw = archive.git("show", f"{source_commit}:{path}")
                if len(raw) != info["bytes"] or hashlib.sha256(raw).hexdigest() != info["sha256"]:
                    raise storage.CheckError("Pinned event-feed bytes do not match the seed receipt")
                target = payload / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)

            store = storage.Store(storage.Drive())
            result = store.backup(
                payload,
                EVENT_FEED_SEED_STREAM,
                "git:" + source_commit,
                artifact["created_at"],
                {
                    "run_id": run_id,
                    "artifact_id": artifact["id"],
                    "receipt_sha256": hashlib.sha256(raw_manifest).hexdigest(),
                    "workflow": EVENT_FEED_SEED_WORKFLOW,
                    "workflow_commit": run["head_sha"],
                    "conclusion": run.get("conclusion"),
                    "role": "private-preservation",
                    "source_branch": "data/event-feed",
                    "source_commit": source_commit,
                    "seed_files": {path: info["sha256"] for path, info in sorted(expected.items())},
                    "seed_bytes": {path: info["bytes"] for path, info in sorted(expected.items())},
                },
            )
            return {
                **result,
                "source_commit": source_commit,
                "receipt_artifact_id": artifact["id"],
                "receipt_sha256": hashlib.sha256(raw_manifest).hexdigest(),
            }
    if source_run:
        raise storage.CheckError("No trusted event-feed seed receipt artifact found")
    return None


def main() -> int:
    register()
    from scripts import archive_to_drive as archive

    # Preserve the established direct CLI contract: --help must never require
    # GitHub or Drive credentials merely to render usage.
    if any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        return archive.main()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--source-run", type=int)
    args, _ = parser.parse_known_args()
    try:
        result = archive_event_feed_seed(args.source_run)
        if result:
            storage.report({"event_feed_seed": result})
    except Exception as exc:
        storage.report({
            "event_feed_seed": {
                "status": "failure",
                "error": str(exc) if isinstance(exc, storage.CheckError) else type(exc).__name__,
            }
        })
        return 1
    return archive.main()


if __name__ == "__main__":
    raise SystemExit(main())
