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
    for name in ('signals.json','replay.json'):
        if (source/name).exists():
            d=json.loads((source/name).read_text())
            refs += [r for r in d.get('evidence_archives',{}).values() if r]
            if d.get('archive'): refs.append(d['archive'])
    names=set()
    for ref in refs:
        manifest=validate_object(source,ref); names.add(ref['path'])
        for horizon in manifest['horizons'].values():
            count=0
            for entry in horizon['shards']:
                obj=validate_object(source,entry)
                if len(obj['points'])!=entry['rows']:raise ValueError('archive row count mismatch')
                count+=entry['rows'];names.add(entry['path'])
            if count!=horizon['rows']:raise ValueError('manifest total count mismatch')
    (destination/'archive').mkdir(parents=True,exist_ok=True)
    if source.resolve()!=destination.resolve():
        for name in names:shutil.copy2(source/name,destination/name)
    retention=destination/'archive_retention.json'
    previous=json.loads(retention.read_text()) if retain_previous and retention.exists() else []
    keep=names|set(previous[0] if previous else [])
    for path in (destination/'archive').glob('*.json'):
        if 'archive/'+path.name not in keep:path.unlink()
    retention.write_text(json.dumps([sorted(names),previous[0] if previous else []]))
    return len(names)


if __name__=='__main__':
    print('Verified archive objects copied:',copy_evidence(sys.argv[1],sys.argv[2],retain_previous='--retain-previous' in sys.argv))
