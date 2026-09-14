"""Archive trusted Actions artifacts and pinned data branches without running their code."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.error import HTTPError
from urllib.parse import urlencode, urlparse
from urllib.request import Request, build_opener, urlopen
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.gdrive_store import (CheckError, Drive, MAX_FILES, MAX_TOTAL, NoRedirect,
                                  REPOSITORY, STREAMS, Store, allowed, canonical, digest, report, safe_path)

ARTIFACTS = {
    "daily-source-state": ("daily-data", "daily_forecast.yml"),
    "event-final-state": ("event-hourly", "event_hourly.yml"),
    "event-hourly-state": ("event-hourly", "event_hourly.yml"),
    "event-daily-backup": ("event-hourly", "event_hourly.yml"),
    "event-research-state": ("event-research", "event_research.yml"),
    "event-historical-state": ("historical-state", "event_historical_study.yml"),
    "event-historical-report": ("historical-report", "event_historical_study.yml"),
    "event-adaptive-report": ("adaptive-report", "event_historical_study.yml"),
    "event-adaptive-rows": ("adaptive-rows", "event_historical_study.yml"),
    "event-foundation-chronos2": ("foundation-chronos2", "event_research.yml"),
    "event-foundation-tirex2": ("foundation-tirex2", "event_research.yml"),
}
BRANCHES = {
    "data/event-ledger": ("event-ledger", ["lake/event-ledger", "lake/event-inputs"]),
    "data/daily-forecast": ("daily-data", ["lake/gold", "lake/raw/market", "lake/raw/vendor", "lake/reports",
                                           "forecast_site/predictions.db", "forecast_site/public"]),
    "data/hybrid-forecast": ("hybrid-data", ["lake/hybrid"]),
    "data/forward-research": ("forward-data", ["lake/forward"]),
}


class GitHub:
    def __init__(self):
        if os.environ.get("GITHUB_REPOSITORY") != REPOSITORY:
            raise CheckError("Unexpected source repository")
        self.base = "https://api.github.com/repos/" + REPOSITORY
        self.headers = {"Authorization": "Bearer " + os.environ["GH_TOKEN"],
                        "Accept": "application/vnd.github+json", "X-GitHub-Api-Version": "2022-11-28"}
        self.repo_id = self.get("")["id"]
        self.runs = {}

    def get(self, path):
        try:
            with build_opener(NoRedirect()).open(Request(self.base + path, headers=self.headers), timeout=45) as response:
                return json.load(response)
        except HTTPError as exc:
            status = exc.code
            exc.close()
            raise CheckError("GitHub source HTTP " + str(status)) from None

    def run(self, run_id):
        if run_id not in self.runs:
            self.runs[run_id] = self.get("/actions/runs/" + str(int(run_id)))
        return self.runs[run_id]

    def artifacts(self, source_run=None):
        selectors = [None] if source_run else list(ARTIFACTS)
        for name in selectors:
            page = 1
            while True:
                params = {"per_page": 100, "page": page}
                if name:
                    params["name"] = name
                path = ("/actions/runs/" + str(int(source_run)) if source_run else "") + "/artifacts"
                if not source_run:
                    path = "/actions/artifacts"
                data = self.get(path + "?" + urlencode(params))["artifacts"]
                yield from data
                # Expired artifacts cannot be recovered; pages are newest first.
                if len(data) < 100 or all(a.get("expired") for a in data):
                    break
                page += 1

    def download(self, artifact_id, target):
        req = Request(self.base + "/actions/artifacts/" + str(int(artifact_id)) + "/zip", headers=self.headers)
        try:
            response = build_opener(NoRedirect()).open(req, timeout=45)
        except HTTPError as exc:
            status, location = exc.code, exc.headers.get("Location", "")
            exc.close()
            host = urlparse(location).hostname or ""
            if status != 302 or urlparse(location).scheme != "https" or not (
                    host.endswith(".blob.core.windows.net") or host.endswith(".githubusercontent.com")):
                raise CheckError("Unexpected GitHub artifact download response") from None
            # Signed storage URL receives no GitHub Authorization header.
            response = urlopen(Request(location), timeout=60)
        with response, Path(target).open("wb") as output:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > MAX_TOTAL:
                    raise CheckError("Artifact download exceeds size limit")
                output.write(block)


def trusted(run, workflow, repo_id):
    return (run.get("head_repository", {}).get("id") == repo_id
            and run.get("repository", {}).get("id") == repo_id
            and run.get("head_branch") == "main" and run.get("status") == "completed"
            and run.get("event") in {"push", "schedule", "workflow_dispatch", "workflow_run"}
            and run.get("path") == ".github/workflows/" + workflow)


def extract_zip(source, destination):
    total = 0
    with zipfile.ZipFile(source) as archive:
        members = archive.infolist()
        if len(members) > MAX_FILES:
            raise CheckError("Artifact member count exceeds limit")
        for member in members:
            if member.is_dir():
                continue
            relative = str(safe_path(member.filename))
            if stat.S_ISLNK(member.external_attr >> 16):
                raise CheckError("Artifact contains a symlink")
            total += member.file_size
            if total > MAX_TOTAL:
                raise CheckError("Unpacked artifact exceeds size limit")
            # Keep SQLite sidecars until consistent online backup; never archive them directly.
            if not allowed(relative) and not relative.endswith((".db-wal", ".db-shm")):
                continue
            path = Path(destination) / relative
            if path.exists():
                raise CheckError("Duplicate artifact path")
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as reader, path.open("wb") as writer:
                shutil.copyfileobj(reader, writer, 1024 * 1024)


def git(*args):
    proc = subprocess.run(["git", *args], capture_output=True)
    if proc.returncode:
        raise CheckError("Data branch git operation failed")
    return proc.stdout


def archive_branch(store, branch, temporary):
    stream, prefixes = BRANCHES[branch]
    # Query and pin a commit before fetching/extracting; never checkout branch code.
    refs = git("ls-remote", "--heads", "origin", "refs/heads/" + branch).decode().split()
    if not refs:
        if stream in ("daily-data", "event-ledger"):
            raise CheckError("Required data branch is missing: " + stream)
        return {"stream": stream, "unavailable": True}
    sha = refs[0]
    if not re.fullmatch("[0-9a-f]{40}", sha):
        raise CheckError("Invalid data commit")
    key = "git:" + sha
    if store.exists(stream, key):
        return {"stream": stream, "already_saved": True}
    git("fetch", "--depth=1", "origin", sha)
    at = git("show", "-s", "--format=%cI", sha).decode().strip()
    existing = [p for p in prefixes if git("ls-tree", sha, "--", p)]
    if not existing:
        raise CheckError("Data branch contains no approved paths")
    tarpath = Path(temporary) / "branch.tar"
    with tarpath.open("wb") as writer:
        proc = subprocess.run(["git", "archive", sha, "--", *existing], stdout=writer, stderr=subprocess.PIPE)
    if proc.returncode:
        raise CheckError("Data branch archive failed")
    root = Path(temporary) / "branch"
    root.mkdir()
    total = 0
    with tarfile.open(tarpath) as archive:
        count = 0
        for member in archive:
            if member.isdir():
                continue
            relative = str(safe_path(member.name))
            if not member.isfile():
                raise CheckError("Data branch contains a non-regular file")
            if not allowed(relative):
                continue
            count += 1
            total += member.size
            if total > MAX_TOTAL or count > MAX_FILES:
                raise CheckError("Data branch archive exceeds size limit")
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as reader, path.open("wb") as writer:
                shutil.copyfileobj(reader, writer)
    return store.backup(root, stream, key, at, {"branch": branch, "commit": sha, "role": "issued-and-observed"})


def archive_artifact(store, github, artifact, temporary):
    spec = ARTIFACTS.get(artifact["name"])
    if not spec or artifact.get("expired"):
        return None
    stream, workflow = spec
    run_id = artifact.get("workflow_run", {}).get("id")
    if not run_id:
        return None
    run = github.run(run_id)
    if not trusted(run, workflow, github.repo_id):
        return None
    # Incomplete research is retained separately and can never become production state.
    if stream == "event-research" and run.get("conclusion") != "success":
        stream = "event-research-partial"
    key = "artifact:" + str(artifact["id"])
    if store.exists(stream, key):
        return {"stream": stream, "already_saved": True}
    zip_path = Path(temporary) / "artifact.zip"
    github.download(artifact["id"], zip_path)
    expected = artifact.get("digest", "")
    if expected.startswith("sha256:"):
        with zip_path.open("rb") as reader:
            actual = hashlib.file_digest(reader, "sha256").hexdigest()
        if actual != expected.split(":", 1)[1]:
            raise CheckError("GitHub artifact SHA256 mismatch")
    root = Path(temporary) / "artifact"
    root.mkdir()
    extract_zip(zip_path, root)
    if stream == "event-hourly" and not all((root / p).is_file() for p in ("active.json", "replay.json", "issued.db", "observations.db")):
        raise CheckError("Hourly artifact lacks required recovery files")
    if stream == "daily-data" and not all((root / p).is_file() for p in ("lake/gold/eth_master_daily.csv", "forecast_site/predictions.db")):
        raise CheckError("Daily artifact lacks required recovery files")
    return store.backup(root, stream, key, artifact["created_at"],
                        {"run_id": run_id, "artifact_id": artifact["id"], "commit": run["head_sha"],
                         "workflow": workflow, "conclusion": run.get("conclusion"),
                         "role": "actual-issuance" if stream == "event-hourly" else "observed-and-issued" if stream == "daily-data" else "retrospective-research"})


def automation_status(store, result, recovery, output):
    """Small immutable read interface for connected scheduled assistants."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    run_id = str(int(os.environ["GITHUB_RUN_ID"]))
    attempt = str(int(os.environ.get("GITHUB_RUN_ATTEMPT", "1")))
    document = dict(result, schema=1, repository=REPOSITORY, run_id=run_id, attempt=attempt,
                    generated_at=datetime.now(timezone.utc).isoformat(),
                    source_run=os.environ.get("SOURCE_RUN") or None,
                    recovery_drill=recovery, snapshots={})
    if store is not None:
        for stream in sorted(STREAMS):
            name = store.latest(stream)
            if not name:
                continue
            data = store.drive.get(name)
            manifest = json.loads(data)
            if manifest.get("complete") is not True or manifest.get("repository") != REPOSITORY or manifest.get("stream") != stream:
                raise CheckError("Invalid manifest in automation catalog")
            document["snapshots"][stream] = {
                "name": name, "file_id": store.drive.index[name]["id"], "sha256": digest(data),
                "snapshot_at": manifest["snapshot_at"], "source": manifest["source"],
                "files": len(manifest["files"]),
                "bytes": sum(info["bytes"] for info in manifest["files"].values())}
    data = canonical(document)
    (output / "status.json").write_bytes(data)
    if store is None:
        return None
    name = f"ef1-status-{run_id}-{attempt}.json"
    store.drive.put(name, data, {"ef_kind": "status", "ef_schema": "1"})
    if store.drive.get(name) != data:
        raise CheckError("Automation status readback mismatch")
    receipt = {"run_id": run_id, "attempt": attempt, "file_id": store.drive.index[name]["id"],
               "sha256": digest(data), "name": name}
    (output / "receipt.json").write_bytes(canonical(receipt))
    report({"automation_receipt": receipt})
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run", type=int)
    args = parser.parse_args()
    failed, saved = [], []
    store, recovery = None, None
    started = time.monotonic()
    try:
        store, github = Store(Drive()), GitHub()
        # Daily/ledger data first: they contain published receipts and original input vintages.
        for branch in BRANCHES:
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    result = archive_branch(store, branch, temporary)
                    saved.append(result)
                    report(result)
            except Exception as exc:
                failed.append({"source": branch, "error": str(exc) if isinstance(exc, CheckError) else type(exc).__name__})
        for artifact in github.artifacts(args.source_run):
            if time.monotonic() - started > 2100:
                failed.append({"source": "remaining artifacts", "error": "Archive budget reached; rerun resumes saved objects"})
                break
            try:
                with tempfile.TemporaryDirectory() as temporary:
                    result = archive_artifact(store, github, artifact, temporary)
                    if result:
                        saved.append(result)
                        report(result)
            except Exception as exc:
                failed.append({"source": "artifact:" + str(artifact["id"]),
                               "error": str(exc) if isinstance(exc, CheckError) else type(exc).__name__})
        # End-to-end drill of the latest operational snapshot, without replacing production files.
        with tempfile.TemporaryDirectory() as temporary:
            recovery = store.restore("event-hourly", Path(temporary) / "verified", required=False)
            report({"recovery_drill": recovery})
        if not saved:
            failed.append({"source": "archive", "error": "No eligible source snapshots found"})
    except Exception as exc:
        failed.append({"source": "archive", "error": str(exc) if isinstance(exc, CheckError) else type(exc).__name__})
    result = {"status": "failure" if failed else "success", "processed_snapshots": len(saved),
              "new_snapshots": sum(not x.get("already_saved") and not x.get("unavailable") for x in saved),
              "uploaded_chunks": sum(x.get("uploaded_chunks", 0) for x in saved),
              "reused_chunks": sum(x.get("reused_chunks", 0) for x in saved), "errors": failed,
              "retention": "No automatic expiry or deletion; subject to Drive capacity and account access"}
    try:
        automation_status(store, result, recovery, "drive-archive-status")
    except Exception as exc:
        failed.append({"source": "automation status", "error": str(exc) if isinstance(exc, CheckError) else type(exc).__name__})
        result["status"] = "failure"
        output = Path("drive-archive-status")
        output.mkdir(exist_ok=True)
        (output / "status.json").write_bytes(canonical(result))
    report(result)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
