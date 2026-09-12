import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from scripts.archive_event_inputs import archive
from scripts.audit_event_system import audit
from scripts.daily_source_state import allowed, persist, restore
from signal_pipeline.data import connect, ingest
from tests.test_event_recovery import publication


def source(root, close=105, observed="2026-09-12T01:02:00+00:00"):
    raw = json.dumps([[1789171200, 90, 110, 100, close, 3]]).encode()
    with connect(root) as con:
        ingest(con, root, "ETH-USD", raw, "2026-09-12", "2026-09-13", observed)


def test_archive_preserves_original_receipts_after_rolling_state_pruning(tmp_path):
    root, target = tmp_path / "state", tmp_path / "archive"
    source(root)
    first = archive(root, target)
    source(root, 106, "2026-09-12T01:20:00+00:00")
    second = archive(root, target)
    path = target / "observations/2026-09-12.jsonl.gz"
    rows = [json.loads(x) for x in gzip.decompress(path.read_bytes()).splitlines()]
    assert [x["close"] for x in rows] == [105, 106]
    assert first["snapshot_rows"] == 1 and second["snapshot_rows"] == 2
    before = path.read_bytes()
    archive(root, target)
    assert path.read_bytes() == before
    with connect(root) as con:
        con.execute("DELETE FROM bars WHERE revision=1")
    archive(root, target)
    assert path.read_bytes() == before


def test_archive_rejects_raw_corruption_and_modified_checkpoint(tmp_path):
    root, target = tmp_path / "state", tmp_path / "archive"
    source(root)
    raw = next((root / "raw").glob("*.gz"))
    original = raw.read_bytes()
    raw.write_bytes(gzip.compress(b"[]"))
    with pytest.raises(ValueError, match="raw receipt hash"):
        archive(root, target)
    raw.write_bytes(original)
    (root / "models").mkdir()
    (root / "models/h6.joblib").write_bytes(b"checkpoint")
    manifest = {"6": {"file": "h6.joblib", "sha256": hashlib.sha256(b"checkpoint").hexdigest()}}
    (root / "active.json").write_text(json.dumps(manifest))
    result = archive(root, target)
    assert len(result["active_models"]) == 1
    (root / "models/h6.joblib").write_bytes(b"modified")
    with pytest.raises(ValueError, match="checkpoint hash"):
        archive(root, target)


def test_daily_cache_survives_a_runner_restart_without_private_files(tmp_path):
    root, durable, restarted = [tmp_path / n for n in ("runner", "durable", "restarted")]
    for name in ("lake/raw/vendor/options.csv", "lake/raw/market/market_data_cache.csv",
                 "lake/raw/vendor/token.json", "lake/raw/manual/private.csv"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("date,value\n2026-09-11,12\n2026-09-12,13\n")
    copied = persist(root, durable)
    assert set(copied) == {"lake/raw/vendor/options.csv", "lake/raw/market/market_data_cache.csv"}
    subprocess.run(["git", "init", str(durable)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(durable), "add", "."], check=True)
    subprocess.run(["git", "-C", str(durable), "-c", "user.name=Test", "-c", "user.email=test@example.com",
                    "commit", "-m", "source snapshot"], check=True, capture_output=True)
    subprocess.run(["git", "clone", "--no-checkout", str(durable), str(restarted)], check=True, capture_output=True)
    assert set(restore(restarted, "HEAD")) == set(copied)
    for name in copied:
        assert (restarted / name).read_bytes() == (root / name).read_bytes()
    assert not (durable / "lake/raw/vendor/token.json").exists()
    assert not allowed("lake/raw/vendor/../../secrets.csv")
    assert not allowed("/lake/raw/vendor/file.csv")


def test_system_audit_distinguishes_accuracy_from_publication():
    from datetime import datetime
    payload = publication()
    payload["source"] = {"ready": True, "errors": [], "latest": {p: payload["expected_slot"] for p in ("ETH-USD", "BTC-USD")}}
    payload["prospective"] = {"6": {"resolved": 10, "nonoverlap_resolved": 2,
                                    "metrics": {"return_mae_pp": 2, "no_change_mae": .01, "mae_skill": -1}}}
    report = audit(payload, datetime.fromisoformat("2026-09-07T06:38:00+00:00"))
    assert report["operation"] == "ready"
    assert report["horizons"][0]["assessment"] == "no_change_not_beaten"
    assert report["horizons"][-1]["assessment"] == "pending"
    payload["source"]["latest"]["BTC-USD"] = "2026-09-07T05:00:00+00:00"
    assert audit(payload)["operation"] == "attention"


def select_state(artifacts, runs, monkey_now="2026-09-12T19:00:00Z"):
    document = Path(".github/workflows/event_hourly.yml").read_text()
    block = document.split("id: state\n", 1)[1].split("          script: |\n", 1)[1]
    lines = []
    for line in block.splitlines():
        if not line.startswith("            "):
            break
        lines.append(line[12:])
    harness = r'''
const input=JSON.parse(process.argv[1]),outputs={},lookups=[];
const RealDate=Date;global.Date=class extends RealDate {constructor(...args){super(...(args.length?args:[input.now]))}};
const context={repo:{owner:'wscha231',repo:'eth-dashboard'},payload:{repository:{id:123}}};
const core={setOutput:(k,v)=>outputs[k]=v};
const github={rest:{actions:{
listArtifactsForRepo:async({name})=>({data:{artifacts:input.artifacts[name]||[]}}),
getWorkflowRun:async({run_id})=>{lookups.push(run_id);return {data:input.runs[run_id]}}
}}};
(async()=>{await new (Object.getPrototypeOf(async function(){}).constructor)('github','context','core',input.script)(github,context,core);
console.log(JSON.stringify({outputs,lookups}));})().catch(e=>{console.error(e);process.exit(1)});
'''
    output = subprocess.run(["node", "-e", harness, json.dumps({"script": "\n".join(lines),
                            "artifacts": artifacts, "runs": runs, "now": monkey_now})],
                            check=True, capture_output=True, text=True)
    return json.loads(output.stdout)


def artifact(run, *, day="2026-09-12", expired=False, repo=123):
    return {"created_at": day + "T01:00:00Z", "expired": expired,
            "workflow_run": {"id": run, "head_branch": "main", "head_repository_id": repo}}


@pytest.mark.parametrize("day,due", [("2026-09-12", "false"), ("2026-09-11", "true")])
def test_restore_daily_backup_after_hourly_expiry_and_backup_due_after_midnight(day, due):
    artifacts = {"event-hourly-state": [artifact(1, expired=True)],
                 "event-daily-backup": [artifact(2, day=day)]}
    runs = {2: {"path": ".github/workflows/event_hourly.yml", "event": "schedule", "conclusion": "success"}}
    result = select_state(artifacts, runs)
    assert result["outputs"] == {"hourly": 2, "hourly_name": "event-daily-backup", "backup": due}
    assert 1 not in result["lookups"]


def test_backup_provenance_and_newer_hourly_state_win():
    artifacts = {"event-hourly-state": [artifact(1)],
                 "event-daily-backup": [artifact(9, repo=999), artifact(8), artifact(2, day="2026-09-11")]}
    runs = {n: {"path": ".github/workflows/event_hourly.yml", "event": "pull_request" if n == 8 else "schedule"} for n in (1, 2, 8)}
    result = select_state(artifacts, runs)
    assert result["outputs"]["hourly"] == 1
    assert result["outputs"]["hourly_name"] == "event-hourly-state"
    assert 9 not in result["lookups"]
