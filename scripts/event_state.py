"""Restore research inputs without overwriting prospective issuance or newer vintages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_pipeline.data import connect as observation_db
from signal_pipeline.data import utc
from signal_pipeline.ledger import backup


def merge_research(source, destination):
    source, destination = Path(source), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if not (source/"replay.json").exists():
        raise ValueError("research state is incomplete")
    # Only input observations are merged. Actual issued.db is NEVER imported from research.
    with observation_db(destination) as dst:
        dst.execute("ATTACH DATABASE ? AS research", (str(source/"observations.db"),))
        rows = dst.execute("""SELECT r.* FROM research.bars r LEFT JOIN main.bars b
            ON b.product=r.product AND b.open_time=r.open_time AND b.content_hash=r.content_hash
            WHERE b.content_hash IS NULL ORDER BY r.observed_at,r.product,r.open_time,r.revision""").fetchall()
        for row in rows:
            prior = dst.execute("SELECT revision,observed_at FROM main.bars WHERE product=? AND open_time=? ORDER BY revision DESC LIMIT 1", row[:2]).fetchone()
            if prior and prior[1] >= row[2]:
                continue  # don't substitute older research vintages for newer actual observations
            row=list(row);row[3]=prior[0]+1 if prior else 1
            dst.execute("INSERT INTO main.bars VALUES (?,?,?,?,?,?,?,?,?,?,?)", row)
        dst.execute("INSERT OR IGNORE INTO main.downloads SELECT * FROM research.downloads")
    for name in ("replay.json", "source_status.json"):
        shutil.copy2(source/name, destination/name)
    for p in (source/"raw").glob("*.json.gz"):
        target=destination/"raw"/p.name;target.parent.mkdir(exist_ok=True)
        if not target.exists():shutil.copy2(p,target)
    # Inference needs only the latest monthly bundle per horizon; retain older versions
    # in the research artifact, so an hourly run doesn't carry years of checkpoints.
    (destination/"models").mkdir(exist_ok=True)
    active={}
    for h in (6,24,72,168,336,720):
        candidates=sorted((source/"models").glob(f"h{h}_*.joblib"),key=lambda p:p.name.split('_')[1])
        if candidates:
            month=candidates[-1].name.split('_')[1]
            p=max((p for p in candidates if p.name.split('_')[1]==month),key=lambda p:p.stat().st_mtime)
            shutil.copy2(p,destination/"models"/p.name)
            active[str(h)]={"file":p.name,"sha256":hashlib.sha256(p.read_bytes()).hexdigest()}
    (destination/"active.json").write_text(json.dumps(active))


def snapshot(root, destination):
    root,destination=Path(root),Path(destination);destination.mkdir(parents=True,exist_ok=True)
    backup(root,destination)
    # Full historical sources/checkpoints live in research artifacts. Hourly state is bounded.
    with sqlite3.connect(destination/"observations.db") as con:
        con.execute("DELETE FROM bars WHERE open_time<?", ((utc()-__import__('pandas').Timedelta(days=40)).isoformat(),))
        hashes={r[0] for r in con.execute("SELECT DISTINCT raw_hash FROM bars")}
        con.execute("DELETE FROM downloads WHERE raw_hash NOT IN (SELECT DISTINCT raw_hash FROM bars)")
        con.commit();con.execute("VACUUM")
    active=json.loads((root/"active.json").read_text())
    (destination/"models").mkdir(exist_ok=True);(destination/"raw").mkdir(exist_ok=True)
    for info in active.values():shutil.copy2(root/"models"/info["file"],destination/"models"/info["file"])
    for value in hashes:
        p=root/"raw"/f"{value}.json.gz"
        if p.exists():shutil.copy2(p,destination/"raw"/p.name)
    for name in ("replay.json","source_status.json","signals.json","weekly_review.json","research_run.txt","active.json"):
        if (root/name).exists():shutil.copy2(root/name,destination/name)
    if not (destination/"issued.db").exists():raise ValueError("cannot save hourly state without ledger")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["merge","snapshot"]);p.add_argument("source");p.add_argument("destination")
    a=p.parse_args();(merge_research if a.action=="merge" else snapshot)(a.source,a.destination)
