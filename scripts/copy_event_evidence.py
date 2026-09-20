"""Copy verified evidence objects; retain two public generations for in-flight readers."""
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from signal_pipeline.diagnostics import validate_object


def copy_evidence(source, destination, *, retain_previous=False):
    source,destination=Path(source),Path(destination)
    refs=[]
    for name in ('signals.json','replay.json','historical_study.json'):
        if (source/name).exists():
            d=json.loads((source/name).read_text())
            refs += [(source,r) for r in d.get('evidence_archives',{}).values() if r]
            if d.get('archive'): refs.append((source,d['archive']))
    # A new incumbent research snapshot does not contain the independently produced
    # candidate study. Preserve its verified objects when merging into hourly state.
    # A research refresh must also retain the archives referenced by the actual
    # already-published feed. Research does not replace that feed or its ledger.
    for name in ('historical_study.json', 'signals.json'):
        retained=destination/name
        supplied=source/name
        # Snapshot callers copy report JSON before its archives. Identical
        # reports must resolve objects from the source, not the still-empty
        # destination. Distinct retained live reports still own their objects.
        if (source.resolve()!=destination.resolve() and retained.exists()
                and (not supplied.exists() or supplied.read_bytes()!=retained.read_bytes())):
            d=json.loads(retained.read_text())
            if d.get('archive'):refs.append((destination,d['archive']))
            refs += [(destination,r) for r in d.get('evidence_archives',{}).values() if r]
    names=set(); origins={}
    for origin,ref in refs:
        manifest=validate_object(origin,ref); names.add(ref['path']);origins[ref['path']]=origin
        for horizon in manifest['horizons'].values():
            count=0
            for entry in horizon['shards']:
                obj=validate_object(origin,entry)
                if len(obj['points'])!=entry['rows']:raise ValueError('archive row count mismatch')
                count+=entry['rows'];names.add(entry['path']);origins[entry['path']]=origin
            if count!=horizon['rows']:raise ValueError('manifest total count mismatch')
    (destination/'archive').mkdir(parents=True,exist_ok=True)
    if source.resolve()!=destination.resolve():
        for name in names:
            if origins[name].resolve()!=destination.resolve():shutil.copy2(origins[name]/name,destination/name)
    retention=destination/'archive_retention.json'
    previous=json.loads(retention.read_text()) if retain_previous and retention.exists() else []
    keep=names|set(previous[0] if previous else [])
    for path in (destination/'archive').glob('*.json'):
        if 'archive/'+path.name not in keep:path.unlink()
    retention.write_text(json.dumps([sorted(names),previous[0] if previous else []]))
    return len(names)


if __name__=='__main__':
    print('Verified archive objects copied:',copy_evidence(sys.argv[1],sys.argv[2],retain_previous='--retain-previous' in sys.argv))
