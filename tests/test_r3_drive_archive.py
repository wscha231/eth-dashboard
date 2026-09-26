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
                r3.R3_ARTIFACTS['derivative-history-reconciliation-state'],
                ('derivative-history-research', 'data_reconciliation.yml'),
            )
            self.assertEqual(
                r3.R3_ARTIFACTS['fast-derivative-state'],
                ('derivative-fast-research', 'fast_derivative_snapshots.yml'),
            )
            self.assertEqual(
                r3.R3_ARTIFACTS['derivative-public-seed-state'],
                ('derivative-public-seed', 'derivative_private_archive_seed.yml'),
            )
            self.assertEqual(
                len({
                    r3.R3_ARTIFACTS['derivative-history-research-state'][0],
                    r3.R3_ARTIFACTS['fast-derivative-state'][0],
                    r3.R3_ARTIFACTS['derivative-public-seed-state'][0],
                }),
                3,
            )
            self.assertTrue(original_streams <= storage.STREAMS)
            for name, mapping in original_artifacts.items():
                self.assertEqual(archive.ARTIFACTS[name], mapping)
        finally:
            storage.STREAMS.clear()
            storage.STREAMS.update(original_streams)
            archive.ARTIFACTS.clear()
            archive.ARTIFACTS.update(original_artifacts)

    def test_derivative_history_artifact_is_isolated_from_fred_and_retained_for_bridge(self):
        text = Path('.github/workflows/free_source_backfill.yml').read_text(encoding='utf-8')
        marker = 'name: derivative-history-research-state'
        self.assertIn(marker, text)
        package = text.split('name: Package isolated derivative-history research state', 1)[1]
        package = package.split('- uses: actions/upload-artifact@v4', 1)[0]
        self.assertIn('bitget_eth_free_features.csv', package)
        self.assertIn('deribit_eth_dvol_daily.csv', package)
        self.assertIn('hyperliquid_eth_free_features.csv', package)
        self.assertNotIn('fred_current_vintage_long.csv', package)
        self.assertNotIn('fred_initial_release_long.csv', package)
        self.assertIn("'public_redistribution':'blocked'", package)
        upload = text.split(marker, 1)[1]
        self.assertIn('retention-days: 30', upload)

    def test_fast_derivative_artifact_has_bridge_retention(self):
        text = Path('.github/workflows/fast_derivative_snapshots.yml').read_text(encoding='utf-8')
        self.assertIn('name: fast-derivative-state', text)
        self.assertIn('retention-days: 30', text)

    def test_public_derivative_seed_seals_all_current_cleanup_targets(self):
        text = Path('.github/workflows/derivative_private_archive_seed.yml').read_text(encoding='utf-8')
        expected = {
            'bitget_eth_context_4h.csv',
            'bitget_eth_free_features.csv',
            'hyperliquid_eth_context_4h.csv',
            'hyperliquid_eth_free_features.csv',
            'deribit_eth_dvol_daily.csv',
            'deribit_eth_funding_daily.csv',
            'deribit_eth_future_snapshot_daily.csv',
            'deribit_eth_historical_volatility.csv',
            'deribit_eth_option_snapshot_daily.csv',
        }
        for name in expected:
            self.assertIn(name, text)
        self.assertIn("if len(rows) != 9", text)
        self.assertIn("'source_branch':'data/daily-forecast'", text)
        self.assertIn("'public_redistribution':'blocked'", text)
        self.assertIn('name: derivative-public-seed-state', text)
        self.assertIn('retention-days: 30', text)

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
        self.assertIn('DATA weekly monthly reconciliation', text)
        self.assertIn('Derivative private archive seed', text)
        self.assertIn('cron: "23,53 * * * *"', text)
        self.assertIn('latest trusted R3 artifact not already covered', text)
        self.assertIn("group: gdrive-archive-${{ github.event_name == 'pull_request' && github.event.pull_request.number || 'main' }}", text)
        self.assertIn("cancel-in-progress: ${{ github.event_name == 'pull_request' }}", text)
        self.assertIn('archive_r3_research_to_drive.py', text)
        self.assertIn('verify_r3_derivative_recovery.py', text)


    def test_public_hot_branch_workflows_no_longer_persist_restricted_derivative_csvs(self):
        free = Path('.github/workflows/free_source_backfill.yml').read_text(encoding='utf-8')
        fast = Path('.github/workflows/fast_derivative_snapshots.yml').read_text(encoding='utf-8')
        reconcile = Path('.github/workflows/data_reconciliation.yml').read_text(encoding='utf-8')
        public_persist_free = free.split('name: Persist research caches with retryable overlay',1)[1].split('name: Package collection receipt',1)[0]
        public_persist_reconcile = reconcile.split('name: Persist reconciled sources and audit with retryable overlay',1)[1].split('name: Package reconciliation receipt',1)[0]
        for name in ('bitget_eth_free_features.csv','deribit_eth_dvol_daily.csv','hyperliquid_eth_free_features.csv'):
            self.assertNotIn(name, public_persist_free)
            self.assertNotIn(name, public_persist_reconcile)
        self.assertNotIn('Persist prospective context with retryable overlay', fast)
        self.assertIn('restore_r3_research.py --stream derivative-history-research', free)
        self.assertIn('restore_r3_research.py --stream derivative-history-research', reconcile)
        self.assertIn('restore_r3_research.py --stream derivative-fast-research', fast)

    def test_reconciliation_derivative_artifact_is_private_and_complete(self):
        text = Path('.github/workflows/data_reconciliation.yml').read_text(encoding='utf-8')
        self.assertIn('name: derivative-history-reconciliation-state', text)
        for name in ('bitget_eth_free_features.csv','deribit_eth_dvol_daily.csv','hyperliquid_eth_free_features.csv'):
            self.assertIn(name, text)
        self.assertIn("'public_redistribution':'blocked'", text)

    def test_derivative_recovery_verifier_can_be_invoked_exactly_like_workflow(self):
        completed = subprocess.run(
            [sys.executable, "scripts/verify_r3_derivative_recovery.py", "--help"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--output", completed.stdout)


if __name__ == '__main__':
    unittest.main()
