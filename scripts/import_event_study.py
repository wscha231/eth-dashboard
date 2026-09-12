"""Copy only a complete compatible study and its verified immutable archive."""
import argparse
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from signal_pipeline.diagnostics import validate_object
from signal_pipeline.protocol import HORIZONS, PROTOCOL_HASH, training_hash


def import_study(source, destination):
    source, destination = Path(source), Path(destination)
    data = json.loads((source/'historical_study.json').read_text())
    if data.get('status') != 'complete' or set(data.get('horizons', {})) != {str(h) for h in HORIZONS}:
        raise ValueError('complete six-horizon study required')
    if data.get('protocol_hash') != PROTOCOL_HASH or data.get('training_hash') != training_hash():
        raise ValueError('historical study protocol/training mismatch')
    manifest = validate_object(source, data['archive'])
    if manifest['kind'] != 'candidate_history' or set(manifest['horizons']) != {str(h) for h in HORIZONS}:
        raise ValueError('historical candidate archive mismatch')
    paths = [data['archive']['path']]
    for h, info in manifest['horizons'].items():
        count = 0
        for entry in info['shards']:
            value = validate_object(source, entry)
            if value['kind'] != 'candidate_history' or value['horizon_hours'] != int(h) or len(value['points']) != entry['rows']:
                raise ValueError('historical shard mismatch')
            paths.append(entry['path']); count += entry['rows']
        if count != info['rows'] or count != data['horizons'][h].get('origins', 0):
            raise ValueError('historical candidate counts differ')
    # Validate all input before touching an existing report; never copy checkpoints or ledgers.
    destination.mkdir(parents=True, exist_ok=True)
    for name in paths:
        path = destination/name; path.parent.mkdir(parents=True, exist_ok=True)
        if (source/name).resolve() != path.resolve():
            shutil.copy2(source/name, path)
    temporary = destination/'historical_study.tmp'
    temporary.write_text(json.dumps(data, allow_nan=False)); temporary.replace(destination/'historical_study.json')
    return len(paths)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('source'); p.add_argument('destination')
    args = p.parse_args(); print('Verified historical evidence objects:', import_study(args.source, args.destination))
