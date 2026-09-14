"""Immutable, deduplicated Drive snapshots. No deletion or model loading code."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, build_opener

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.check_gdrive_connection import CheckError, NoRedirect, credentials, mask, request

REPOSITORY = "wscha231/eth-dashboard"
API = "https://www.googleapis.com/drive/v3/files"
CHUNK = 4 * 1024 * 1024
MAX_TOTAL = 20 * 1024 ** 3
MAX_FILES = 100000
UPLOAD_WORKERS = 4
UPLOAD_PENDING = 8
STREAMS = {"event-hourly", "event-research", "event-research-partial", "historical-state", "historical-report",
           "adaptive-report", "adaptive-rows", "daily-data", "event-ledger",
           "hybrid-data", "forward-data", "legacy-model", "foundation-chronos2", "foundation-tirex2"}
SUFFIXES = {".db", ".sqlite", ".json", ".jsonl", ".csv", ".parquet", ".gz", ".joblib", ".txt", ".md"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def timestamp(value):
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise CheckError("Snapshot time must include a timezone")
    return dt.astimezone(timezone.utc).isoformat()


def safe_path(name):
    path = PurePosixPath(name)
    if (not name or path.is_absolute() or str(path) != name or "\\" in name or ":" in name
            or any(p in (".", "..") or p.startswith(".") for p in path.parts)):
        raise CheckError("Unsafe snapshot path")
    return path


def allowed(name):
    path = safe_path(name)
    if any(re.search(r"(^|[_-])(secret|token|credential|password|rclone)([_.-]|$)", p, re.I) for p in path.parts):
        return False
    return path.suffix.lower() in SUFFIXES


class Drive:
    """Fixed Google endpoints, bounded responses, no raw errors/credentials in logs."""
    def __init__(self):
        self.values = credentials(os.environ)
        for key, value in self.values.items():
            if not value or any(c.isspace() for c in value):
                raise CheckError("Missing or malformed " + key)
            mask(value, os.environ)
        self.folder = self.values["GDRIVE_FOLDER_ID"]
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.folder):
            raise CheckError("Invalid Drive folder ID")
        self.token = None
        self.token_deadline = 0
        self.token_lock = threading.Lock()
        self.index = {}
        self.refresh()
        folder = self.json("GET", API + "/" + self.folder + "?fields=mimeType,trashed,capabilities(canAddChildren)")
        if folder.get("mimeType") != "application/vnd.google-apps.folder" or folder.get("trashed"):
            raise CheckError("Drive storage folder is unavailable")
        self.scan()

    def refresh(self):
        grant = {"client_id": self.values["GDRIVE_CLIENT_ID"], "client_secret": self.values["GDRIVE_CLIENT_SECRET"],
                 "refresh_token": self.values["GDRIVE_REFRESH_TOKEN"], "grant_type": "refresh_token"}
        data = request("POST", "https://oauth2.googleapis.com/token", body=urlencode(grant).encode(),
                       content_type="application/x-www-form-urlencoded")
        token = data.get("access_token")
        if not isinstance(token, str) or not token or any(c.isspace() for c in token):
            raise CheckError("Drive refresh did not return an access token")
        self.token = token
        self.token_deadline = time.monotonic() + max(1, min(int(data.get("expires_in", 3600)), 3600) - 60)
        mask(token, os.environ)

    def raw(self, method, url, body=None, content_type="application/json", limit=32 * 1024 ** 2):
        if not (url.startswith(API) or url.startswith("https://www.googleapis.com/upload/drive/v3/files")):
            raise CheckError("Unexpected Drive endpoint")
        for attempt in range(3 if method == "GET" else 1):
            if time.monotonic() >= self.token_deadline:
                with self.token_lock:
                    if time.monotonic() >= self.token_deadline:
                        self.refresh()
            req = Request(url, data=body, method=method,
                          headers={"Authorization": "Bearer " + self.token, "Content-Type": content_type})
            try:
                with build_opener(NoRedirect()).open(req, timeout=45) as response:
                    data = response.read(limit + 1)
                if len(data) > limit:
                    raise CheckError("Drive response exceeded size limit")
                return data
            except HTTPError as exc:
                status = exc.code
                exc.close()
                if method == "GET" and attempt < 2 and status in (429, 500, 502, 503, 504):
                    time.sleep(2 ** attempt)
                    continue
                raise CheckError("Drive HTTP " + str(status)) from None
            except (OSError, URLError):
                if method == "GET" and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise CheckError("Drive network request failed; rerun resumes completed objects") from None

    def json(self, method, url, **kwargs):
        return json.loads(self.raw(method, url, **kwargs))

    def scan(self):
        page = None
        while True:
            params = {"q": "'" + self.folder + "' in parents and trashed = false and name contains 'ef1-'",
                      "fields": "nextPageToken,files(id,name,size,md5Checksum,appProperties)", "pageSize": 1000}
            if page:
                params["pageToken"] = page
            data = self.json("GET", API + "?" + urlencode(params))
            for entry in data.get("files", []):
                self.index.setdefault(entry["name"], entry)
            page = data.get("nextPageToken")
            if not page:
                break

    def put(self, name, data, properties):
        md5 = hashlib.md5(data).hexdigest()
        existing = self.index.get(name)
        if existing:
            if existing.get("md5Checksum") != md5 or int(existing["size"]) != len(data):
                raise CheckError("Existing immutable Drive object has conflicting bytes")
            return False
        if len(data) > CHUNK + 65536:
            raise CheckError("Drive object exceeds bounded multipart size")
        boundary = "ef_" + digest(data)
        metadata = canonical({"name": name, "parents": [self.folder], "appProperties": properties,
                              "mimeType": "application/json" if name.endswith(".json") else "application/octet-stream"})
        body = (b"--" + boundary.encode() + b"\r\nContent-Type: application/json; charset=UTF-8\r\n\r\n"
                + metadata + b"\r\n--" + boundary.encode() + b"\r\nContent-Type: application/octet-stream\r\n\r\n"
                + data + b"\r\n--" + boundary.encode() + b"--\r\n")
        entry = self.json("POST", "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart&fields=id,name,size,md5Checksum,appProperties",
                          body=body, content_type="multipart/related; boundary=" + boundary)
        if entry.get("md5Checksum") != md5 or int(entry.get("size", -1)) != len(data):
            raise CheckError("Drive upload integrity verification failed")
        self.index[name] = entry
        return True

    def get(self, name, limit=CHUNK + 65536):
        entry = self.index.get(name)
        if not entry or not re.fullmatch(r"[A-Za-z0-9_-]+", entry["id"]):
            raise CheckError("Referenced Drive object is missing")
        data = self.raw("GET", API + "/" + entry["id"] + "?alt=media", limit=limit)
        if hashlib.md5(data).hexdigest() != entry.get("md5Checksum"):
            raise CheckError("Drive download checksum mismatch")
        return data


def manifest_name(stream, key):
    if stream not in STREAMS or not key or len(key) > 300:
        raise CheckError("Invalid snapshot identity")
    return "ef1-manifest-" + stream + "-" + digest(key.encode()) + ".json"


class Store:
    def __init__(self, drive):
        self.drive = drive

    def exists(self, stream, key):
        return manifest_name(stream, key) in self.drive.index

    def backup(self, root, stream, key, at, source):
        name = manifest_name(stream, key)
        if name in self.drive.index:
            return {"stream": stream, "already_saved": True, "snapshot": name}
        at = timestamp(at)
        files, uploaded, reused, total = {}, 0, 0, 0
        root = Path(root).resolve()
        if not root.is_dir():
            raise CheckError("Snapshot source directory is missing")
        pending = {}

        def finish_one():
            nonlocal uploaded, reused
            key = next(iter(pending))
            added = pending.pop(key).result()
            uploaded += int(added)
            reused += int(not added)

        with tempfile.TemporaryDirectory() as temporary, ThreadPoolExecutor(max_workers=UPLOAD_WORKERS) as pool:
            for directory, dirs, names in os.walk(root):
                for item in dirs:
                    if (Path(directory) / item).is_symlink():
                        raise CheckError("Snapshot source contains a symlink")
                for item in sorted(names):
                    path = Path(directory) / item
                    relative = path.relative_to(root).as_posix()
                    if path.is_symlink() or not path.is_file():
                        raise CheckError("Snapshot source is not a regular file")
                    if not allowed(relative):
                        continue
                    selected = path
                    if path.suffix in (".db", ".sqlite"):
                        selected = Path(temporary) / "consistent.db"
                        selected.unlink(missing_ok=True)
                        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as src, sqlite3.connect(selected) as dst:
                            src.backup(dst)
                            if dst.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                                raise CheckError("SQLite snapshot integrity check failed")
                            dst.execute("PRAGMA journal_mode=DELETE")
                    info = {"bytes": 0, "sha256": "", "chunks": []}
                    sha = hashlib.sha256()
                    with selected.open("rb") as reader:
                        while chunk := reader.read(CHUNK):
                            total += len(chunk)
                            if total > MAX_TOTAL:
                                raise CheckError("Snapshot exceeds configured byte limit")
                            sha.update(chunk)
                            compressed = gzip.compress(chunk, mtime=0)
                            hash_value = digest(compressed)
                            blob = "ef1-blob-" + hash_value
                            if blob in pending:
                                # One request owns each content hash, even across different files.
                                reused += 1
                            elif blob in self.drive.index:
                                self.drive.put(blob, compressed, {"ef_kind": "blob", "ef_schema": "1"})
                                reused += 1
                            else:
                                pending[blob] = pool.submit(self.drive.put, blob, compressed,
                                                            {"ef_kind": "blob", "ef_schema": "1"})
                                if len(pending) >= UPLOAD_PENDING:
                                    finish_one()
                            info["chunks"].append({"sha256": hash_value, "bytes": len(compressed), "raw_bytes": len(chunk)})
                            info["bytes"] += len(chunk)
                    info["sha256"] = sha.hexdigest()
                    files[relative] = info
                    if len(files) > MAX_FILES:
                        raise CheckError("Snapshot file count exceeds limit")
            if not files:
                raise CheckError("Refusing an empty snapshot")
            while pending:
                finish_one()
            manifest = {"schema": 1, "repository": REPOSITORY, "stream": stream, "source_key": key,
                        "snapshot_at": at, "source": source, "files": files,
                        "captured_at": datetime.now(timezone.utc).isoformat(), "complete": True}
            data = canonical(manifest)
            self.drive.put(name, data, {"ef_kind": "manifest", "ef_schema": "1", "ef_stream": stream,
                                       "ef_at": at, "ef_captured": manifest["captured_at"]})
            if self.drive.get(name) != data:
                raise CheckError("Committed manifest readback failed")
        return {"stream": stream, "snapshot": name, "files": len(files), "bytes": total,
                "uploaded_chunks": uploaded, "reused_chunks": reused}

    def latest(self, stream, as_of=None):
        if stream not in STREAMS:
            raise CheckError("Unknown snapshot stream")
        cutoff = timestamp(as_of) if as_of else None
        candidates = []
        for name, entry in self.drive.index.items():
            props = entry.get("appProperties", {})
            if props.get("ef_kind") == "manifest" and props.get("ef_stream") == stream:
                at = timestamp(props["ef_at"])
                if cutoff is None or at <= cutoff:
                    candidates.append((at, props.get("ef_captured", ""), name))
        return max(candidates)[2] if candidates else None

    def restore(self, stream, destination, *, as_of=None, required=True, name=None):
        name = name or self.latest(stream, as_of)
        if not name:
            if required:
                raise CheckError("No committed Drive snapshot for " + stream)
            return {"stream": stream, "restored": False}
        manifest = json.loads(self.drive.get(name))
        if (manifest.get("schema") != 1 or manifest.get("repository") != REPOSITORY
                or manifest.get("stream") != stream or manifest.get("complete") is not True
                or manifest_name(stream, manifest.get("source_key", "")) != name):
            raise CheckError("Invalid snapshot manifest identity")
        if as_of and timestamp(manifest["snapshot_at"]) > timestamp(as_of):
            raise CheckError("Snapshot is newer than requested cutoff")
        files = manifest["files"]
        if not isinstance(files, dict) or not 0 < len(files) <= MAX_FILES:
            raise CheckError("Invalid snapshot file count")
        destination = Path(destination).absolute()
        if destination.is_symlink() or any(p.is_symlink() for p in destination.parents):
            raise CheckError("Restore destination contains a symlink")
        if destination.exists() and any(destination.iterdir()):
            raise CheckError("Restore destination must be empty; production data is never overwritten")
        destination.parent.mkdir(parents=True, exist_ok=True)
        total = 0
        with tempfile.TemporaryDirectory(prefix=".drive-verify-", dir=destination.parent) as temporary:
            staging = Path(temporary) / "payload"
            staging.mkdir()
            for relative, info in files.items():
                if not allowed(relative):
                    raise CheckError("Manifest contains a disallowed file")
                size = info["bytes"]
                if not isinstance(size, int) or size < 0:
                    raise CheckError("Invalid snapshot file size")
                total += size
                if total > MAX_TOTAL:
                    raise CheckError("Restore exceeds configured byte limit")
                if len(info["chunks"]) > (size + CHUNK - 1) // CHUNK:
                    raise CheckError("Invalid chunk count")
                target = staging / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                sha, written = hashlib.sha256(), 0
                with target.open("wb") as writer:
                    for chunk in info["chunks"]:
                        hash_value = chunk["sha256"]
                        if not re.fullmatch("[0-9a-f]{64}", hash_value) or not 0 < chunk["raw_bytes"] <= CHUNK:
                            raise CheckError("Invalid chunk metadata")
                        data = self.drive.get("ef1-blob-" + hash_value)
                        if len(data) != chunk["bytes"] or digest(data) != hash_value:
                            raise CheckError("Stored chunk checksum mismatch")
                        with gzip.GzipFile(fileobj=io.BytesIO(data)) as reader:
                            raw = reader.read(chunk["raw_bytes"] + 1)
                        if len(raw) != chunk["raw_bytes"]:
                            raise CheckError("Chunk decompression size mismatch")
                        sha.update(raw)
                        written += len(raw)
                        writer.write(raw)
                if written != size or sha.hexdigest() != info["sha256"]:
                    raise CheckError("Restored file checksum mismatch")
                if target.suffix in (".db", ".sqlite"):
                    with sqlite3.connect(target.as_uri() + "?mode=ro&immutable=1", uri=True) as con:
                        if con.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                            raise CheckError("Restored SQLite integrity check failed")
            if destination.exists():
                destination.rmdir()
            staging.replace(destination)
        return {"stream": stream, "restored": True, "snapshot": name, "snapshot_at": manifest["snapshot_at"],
                "files": len(files), "bytes": total, "source": manifest["source"]}


def report(result):
    text = json.dumps(result, sort_keys=True)
    print(text, flush=True)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as stream:
            stream.write("\n```json\n" + text + "\n```\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["restore", "list"])
    parser.add_argument("--stream", choices=sorted(STREAMS), required=True)
    parser.add_argument("--target", default="drive-restored")
    parser.add_argument("--as-of")
    parser.add_argument("--optional", action="store_true")
    args = parser.parse_args()
    try:
        store = Store(Drive())
        result = ({"stream": args.stream, "snapshot": store.latest(args.stream, args.as_of)} if args.action == "list"
                  else store.restore(args.stream, args.target, as_of=args.as_of, required=not args.optional))
        report(result)
    except Exception as exc:
        print("Drive storage failed: " + (str(exc) if isinstance(exc, CheckError) else type(exc).__name__), flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
