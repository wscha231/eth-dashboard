from pathlib import Path

from scripts import archive_to_drive as archive
from scripts import gdrive_store as storage
from scripts import archive_r3_research_to_drive as r3


def test_r3_research_streams_extend_existing_archive_without_replacing_core_maps():
    original_streams = set(storage.STREAMS)
    original_artifacts = dict(archive.ARTIFACTS)
    try:
        r3.register()
        assert r3.R3_STREAMS <= storage.STREAMS
        for name, mapping in r3.R3_ARTIFACTS.items():
            assert archive.ARTIFACTS[name] == mapping
        assert original_streams <= storage.STREAMS
        for name, mapping in original_artifacts.items():
            assert archive.ARTIFACTS[name] == mapping
    finally:
        storage.STREAMS.clear()
        storage.STREAMS.update(original_streams)
        archive.ARTIFACTS.clear()
        archive.ARTIFACTS.update(original_artifacts)


def test_r3_drive_workflow_tracks_both_research_workflows_and_isolates_pr_concurrency():
    text = Path('.github/workflows/r3_research_drive_archive.yml').read_text(encoding='utf-8')
    assert 'ETH ETF flow research' in text
    assert 'ETH supply research' in text
    assert "group: gdrive-archive-${{ github.event_name == 'pull_request' && github.event.pull_request.number || 'main' }}" in text
    assert "cancel-in-progress: ${{ github.event_name == 'pull_request' }}" in text
    assert 'archive_r3_research_to_drive.py' in text
