from pathlib import Path
import subprocess
import sys
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
            self.assertEqual(
                r3.R3_ARTIFACTS['eth-liquidation-research-state'],
                ('eth-liquidation-research', 'eth_liquidation_research.yml'),
            )
            self.assertEqual(
                r3.R3_ARTIFACTS['eth-execution-network-research-state'],
                ('eth-execution-network-research', 'eth_execution_network_research.yml'),
            )
            self.assertEqual(
                r3.R3_ARTIFACTS['derivative-history-research-state'],
                ('derivative-history-research', 'free_source_backfill.yml'),
            )
            self.assertNotIn('free-source-fallback-state', r3.R3_ARTIFACTS)
            self.assertEqual(
                r3.R3_ARTIFACTS['fast-derivative-state'],
                ('derivative-fast-research', 'fast_derivative_snapshots.yml'),
            )
            self.assertNotEqual(
                r3.R3_ARTIFACTS['derivative-history-research-state'][0],
                r3.R3_ARTIFACTS['fast-derivative-state'][0],
            )
            self.assertTrue(original_streams <= storage.STREAMS)
            for name, mapping in original_artifacts.items():
                self.assertEqual(archive.ARTIFACTS[name], mapping)
        finally:
            storage.STREAMS.clear()
            storage.STREAMS.update(original_streams)
            archive.ARTIFACTS.clear()
            archive.ARTIFACTS.update(original_artifacts)

    def test_r3_archiver_can_be_invoked_exactly_like_workflow(self):
        completed = subprocess.run(
            [sys.executable, "scripts/archive_r3_research_to_drive.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--source-run", completed.stdout)

    def test_r3_drive_workflow_tracks_sources_and_has_reconciliation_safety_net(self):
        text = Path('.github/workflows/r3_research_drive_archive.yml').read_text(encoding='utf-8')
        self.assertIn('ETH ETF flow research', text)
        self.assertIn('ETH supply research', text)
        self.assertIn('ETH liquidation research', text)
        self.assertIn('ETH execution network research', text)
        self.assertIn('Free ETH research source backfill', text)
        self.assertIn('Fast ETH derivative snapshots', text)
        self.assertIn('cron: "23,53 * * * *"', text)
        self.assertIn('latest trusted R3 artifact not already covered', text)
        self.assertIn("group: gdrive-archive-${{ github.event_name == 'pull_request' && github.event.pull_request.number || 'main' }}", text)
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", text)
        self.assertIn('archive_r3_research_to_drive.py', text)


if __name__ == '__main__':
    unittest.main()
