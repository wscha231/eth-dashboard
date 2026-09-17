from pathlib import Path
import unittest

from scripts import archive_to_drive as archive
from scripts import gdrive_store as storage
from scripts import archive_r3_research_to_drive as r3


class R3DriveArchiveTests(unittest.TestCase):
    def test_r3_research_streams_extend_existing_archive_without_replacing_core_maps(self):
        original_streams = set(storage.STREAMS)
        original_artifacts = dict(archive.ARTIFACTS)
        try:
            r3.register()
            self.assertTrue(r3.R3_STREAMS <= storage.STREAMS)
            for name, mapping in r3.R3_ARTIFACTS.items():
                self.assertEqual(archive.ARTIFACTS[name], mapping)
            self.assertTrue(original_streams <= storage.STREAMS)
            for name, mapping in original_artifacts.items():
                self.assertEqual(archive.ARTIFACTS[name], mapping)
        finally:
            storage.STREAMS.clear()
            storage.STREAMS.update(original_streams)
            archive.ARTIFACTS.clear()
            archive.ARTIFACTS.update(original_artifacts)

    def test_r3_drive_workflow_tracks_both_research_workflows_and_isolates_pr_concurrency(self):
        text = Path('.github/workflows/r3_research_drive_archive.yml').read_text(encoding='utf-8')
        self.assertIn('ETH ETF flow research', text)
        self.assertIn('ETH supply research', text)
        self.assertIn("group: gdrive-archive-${{ github.event_name == 'pull_request' && github.event.pull_request.number || 'main' }}", text)
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", text)
        self.assertIn('archive_r3_research_to_drive.py', text)


if __name__ == '__main__':
    unittest.main()
