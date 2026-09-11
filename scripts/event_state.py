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
from scripts.copy_event_evidence import copy_evidence


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
    copy_evidence(source,destination)
    # Upgrade a trusted legacy artifact without retraining or waiting for the next research run.
    replay=json.loads((destination/'replay.json').read_text())
    if not replay.get('archive'):
        import pandas as pd
        from signal_pipeline.diagnostics import export_archive
        for h,report in replay.get('horizons',{}).items():
            audit=source/'replay'/f'h{h}.csv.gz'
            if audit.exists():
                frame=pd.read_csv(audit)
                baseline={r['slot']:r for r in frame[frame.model.eq('climatology')].to_dict('records')}
                points=frame[frame.model.eq('selected')].to_dict('records')
                for point in points:
                    point['baseline']={k:baseline[point['slot']][k] for k in
                        ('p_down','p_flat','p_up','hit_up','hit_down','q10','q50','q90','threshold_up','threshold_down')}
                report['points']=points
            elif report.get('points'):
                raise ValueError('legacy replay upgrade requires full audit rows')
        replay['archive']=export_archive(destination,{h:r.get('points',[]) for h,r in replay.get('horizons',{}).items()},
                                         kind='historical',as_of=replay.get('data_as_of',replay.get('generated_at')))
        (destination/'replay.json').write_text(json.dumps(replay,allow_nan=False))
    for name in ('optimization.json','shadow_active.json'):
        if (source/name).exists():shutil.copy2(source/name,destination/name)
    if (source/'shadow_models').exists():
        shutil.copytree(source/'shadow_models',destination/'shadow_models',dirs_exist_ok=True)
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
    copy_evidence(root,destination)
    for name in ('optimization.json','shadow_active.json'):
        if (root/name).exists():shutil.copy2(root/name,destination/name)
    if (root/'shadow_active.json').exists():
        (destination/'shadow_models').mkdir(exist_ok=True)
        for entry in json.loads((root/'shadow_active.json').read_text()).values():
            shutil.copy2(root/'shadow_models'/entry['file'],destination/'shadow_models'/entry['file'])
    if (root/'shadow/issued.db').exists():
        (destination/'shadow').mkdir(exist_ok=True)
        with sqlite3.connect(root/'shadow/issued.db') as src,sqlite3.connect(destination/'shadow/issued.db') as dst:src.backup(dst)
    if not (destination/"issued.db").exists():raise ValueError("cannot save hourly state without ledger")


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["merge","snapshot"]);p.add_argument("source");p.add_argument("destination")
    a=p.parse_args();(merge_research if a.action=="merge" else snapshot)(a.source,a.destination)
