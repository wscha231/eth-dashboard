"""Real contracts with synthetic source responses; all fault records stay in tmp dirs."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
import unittest

from scripts import availability_shadow as a

ORIGIN = datetime(2026, 9, 20, 0, tzinfo=timezone.utc)
NOW = ORIGIN + timedelta(minutes=10)


def add_rows(root, product, opens, received=NOW-timedelta(minutes=1), revision=1, price=2000.):
    points = [[int(t.timestamp()), price-1, price+1, price, price, 10.] for t in opens]
    for i in range(0, len(points), 299):
        batch = points[i:i+299]
        raw = json.dumps(batch).encode()
        sha = hashlib.sha256(raw).hexdigest()
        (root/'raw').mkdir(exist_ok=True)
        (root/'raw'/f'{sha}.json.gz').write_bytes(gzip.compress(raw, mtime=0))
        with closing(sqlite3.connect(root/'observations.db')) as con, con:
            con.execute('INSERT OR IGNORE INTO downloads VALUES (?,?,?,?)',
                        ('https://api.exchange.coinbase.com/products/'+product+'/candles',sha,received.isoformat(),len(batch)))
            for t,lo,hi,o,c,v in batch:
                values = [o,hi,lo,c,v]
                content_hash = hashlib.sha256(json.dumps(values).encode()).hexdigest()
                con.execute('INSERT INTO bars VALUES (?,?,?,?,?,?,?,?,?,?,?)',
                    (product,datetime.fromtimestamp(t,timezone.utc).isoformat(),received.isoformat(),revision,
                     *values,content_hash,sha))


def source(root, eth=721, btc=721):
    root.mkdir(parents=True,exist_ok=True)
    with closing(sqlite3.connect(root/'observations.db')) as con, con:
        con.executescript('''CREATE TABLE bars(product TEXT,open_time TEXT,observed_at TEXT,revision INTEGER,
          open REAL,high REAL,low REAL,close REAL,volume REAL,content_hash TEXT,raw_hash TEXT,
          PRIMARY KEY(product,open_time,revision));
          CREATE TABLE downloads(url TEXT,raw_hash TEXT,observed_at TEXT,rows INTEGER,PRIMARY KEY(url,raw_hash));''')
    for product,n in (('ETH-USD',eth),('BTC-USD',btc)):
        add_rows(root,product,[ORIGIN-i*a.HOUR for i in range(n,0,-1)])
    (root/'issued.db').write_bytes(b'CORE LEDGER UNTOUCHED')
    (root/'signals.json').write_text('{"release_id":"unchanged"}')


class AvailabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'input'
    def build(self,eth=721,btc=721):source(self.root,eth,btc)
    def run_shadow(self,now=NOW):return a.run(self.root,clock=lambda:now)
    def forecasts(self):
        with closing(a.readonly(self.root/a.STATE/'issued.db')) as con:
            return [json.loads(r[0]) for r in con.execute('SELECT payload FROM forecasts ORDER BY horizon')]
    def test_complete_strict_route_suppresses_fallback(self):
        self.build();r=self.run_shadow()
        self.assertEqual(r['decision']['status'],'not_needed')
        self.assertEqual(r['totals']['forecasts'],0)
    def test_missing_btc_permits_exact_center_only(self):
        self.build(btc=0);r=self.run_shadow()
        self.assertEqual(r['decision']['new_forecasts'],6)
        for p,h in zip(self.forecasts(),a.HORIZONS):
            self.assertEqual(p['center_price'],p['reference_price'])
            self.assertEqual(p['center_log_return'],0)
            self.assertIsNone(p['event']);self.assertIsNone(p['variance']);self.assertIsNone(p['distribution'])
            self.assertFalse(p['public_delivery']);self.assertEqual(p['promotion'],'none')
            self.assertEqual(a.utc(p['target_end']),ORIGIN+(h+1)*a.HOUR)
            self.assertLessEqual(a.utc(p['input_available_at']),a.utc(p['issued_at']))
    def test_169_eth_is_sufficient_but_168_is_not(self):
        self.build(eth=169,btc=0);self.assertEqual(self.run_shadow()['totals']['forecasts'],6)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('DELETE FROM bars WHERE open_time=?',((ORIGIN-169*a.HOUR).isoformat(),))
        r=self.run_shadow();self.assertEqual(r['decision']['status'],'blocked')
    def test_old_eth_gap_only_outside169_permits_baseline(self):
        self.build()
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('DELETE FROM bars WHERE product=? AND open_time=?',('ETH-USD',(ORIGIN-200*a.HOUR).isoformat()))
        self.assertEqual(self.run_shadow()['totals']['forecasts'],6)
    def test_missing_latest_eth_is_blocked(self):
        self.build(btc=0)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('DELETE FROM bars WHERE open_time=?',((ORIGIN-a.HOUR).isoformat(),))
        r=self.run_shadow();self.assertEqual(r['decision']['reason'],'incomplete_recent_eth')
        self.assertEqual(r['totals']['forecasts'],0)
    def test_future_receipts_are_not_current_inputs(self):
        self.build(btc=0)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('UPDATE bars SET observed_at=? WHERE open_time=?',((NOW+timedelta(minutes=1)).isoformat(),(ORIGIN-a.HOUR).isoformat()))
        self.assertEqual(self.run_shadow()['totals']['forecasts'],0)
    def test_future_revision_does_not_replace_known_value(self):
        self.build(btc=0)
        add_rows(self.root,'ETH-USD',[ORIGIN-a.HOUR],NOW+timedelta(minutes=1),2,3000.)
        self.run_shadow()
        self.assertEqual(self.forecasts()[0]['center_price'],2000.)
    def test_later_revision_never_changes_original_forecast(self):
        self.build(btc=0);self.run_shadow();before=self.forecasts()
        add_rows(self.root,'ETH-USD',[ORIGIN-a.HOUR],NOW+timedelta(minutes=1),2,3000.)
        r=self.run_shadow(NOW+timedelta(minutes=2))
        self.assertEqual(r['decision']['new_forecasts'],0);self.assertEqual(self.forecasts(),before)
    def test_invalid_ohlcv_or_hash_is_not_missingness(self):
        self.build(btc=0)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('UPDATE bars SET close=-1 WHERE open_time=?',((ORIGIN-a.HOUR).isoformat(),))
        with self.assertRaises(ValueError):self.run_shadow()
        self.assertFalse((self.root/a.STATE).exists())
    def test_corrupt_raw_or_missing_receipt_is_blocked(self):
        self.build(btc=0)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:con.execute('DELETE FROM downloads')
        with self.assertRaisesRegex(ValueError,'receipt'):self.run_shadow()
    def test_forged_raw_hash_path_rejected(self):
        self.build(btc=0)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('UPDATE bars SET raw_hash=?',('../'+'x'*61,))
        with self.assertRaises(ValueError):self.run_shadow()
    def test_raw_integrity_checked(self):
        self.build(btc=0)
        for p in (self.root/'raw').iterdir():p.write_bytes(gzip.compress(b'[]'))
        with self.assertRaisesRegex(ValueError,'integrity'):self.run_shadow()
    def test_unchanged_core_and_source_bytes(self):
        self.build(btc=0)
        before={n:a.file_sha(self.root/n) for n in ('observations.db','issued.db','signals.json')}
        self.run_shadow()
        self.assertEqual(before,{n:a.file_sha(self.root/n) for n in before})
    def test_preclose_receipt_rejected(self):
        self.build(btc=0)
        with closing(sqlite3.connect(self.root/'observations.db')) as con, con:
            con.execute('UPDATE bars SET observed_at=? WHERE open_time=?',((ORIGIN-timedelta(minutes=1)).isoformat(),(ORIGIN-a.HOUR).isoformat()))
        with self.assertRaisesRegex(ValueError,'pre-close'):self.run_shadow()
    def test_deadline_55_is_not_backdated(self):
        self.build(btc=0)
        r=self.run_shadow(ORIGIN+timedelta(minutes=55))
        self.assertEqual(r['totals']['forecasts'],0)
    def test_validation_crosses_deadline(self):
        self.build(btc=0)
        ticks=iter([NOW,ORIGIN+timedelta(minutes=55)])
        r=a.run(self.root,clock=lambda:next(ticks))
        self.assertEqual(r['totals']['forecasts'],0)
    def test_clock_crosses_hour_or_goes_backwards(self):
        self.build(btc=0)
        ticks=iter([NOW,ORIGIN+a.HOUR])
        self.assertEqual(a.run(self.root,clock=lambda:next(ticks))['totals']['forecasts'],0)
        ticks=iter([NOW,NOW-timedelta(seconds=1)])
        with self.assertRaises(ValueError):a.run(self.root,clock=lambda:next(ticks))
    def test_naive_clock_rejected(self):
        with self.assertRaises(ValueError):a.utc('2026-09-20T00:00:00')
    def test_stale_source_next_hour_blocks(self):
        self.build(btc=0)
        self.assertEqual(self.run_shadow(NOW+a.HOUR)['totals']['forecasts'],0)
    def test_no_source_file_is_not_created(self):
        self.root.mkdir()
        with self.assertRaises(sqlite3.OperationalError):self.run_shadow()
        self.assertFalse((self.root/'observations.db').exists())
    def test_manifest_mismatch_rejected_without_replacement(self):
        self.build(btc=0);self.run_shadow()
        p=self.root/a.STATE/'manifest.json';p.write_text('{}')
        sha=a.file_sha(self.root/a.STATE/'issued.db')
        with self.assertRaises((KeyError,ValueError)):self.run_shadow()
        self.assertEqual(sha,a.file_sha(self.root/a.STATE/'issued.db'))
    def test_partial_state_rejected(self):
        self.build(btc=0)
        p=self.root/a.STATE;p.mkdir();(p/'manifest.json').write_text('{}')
        with self.assertRaises(sqlite3.OperationalError):self.run_shadow()
    def test_duplicate_attempts_do_not_duplicate_forecasts(self):
        self.build(btc=0);self.run_shadow();r=self.run_shadow(NOW+timedelta(seconds=1))
        self.assertEqual(r['totals']['forecasts'],6);self.assertEqual(r['decision']['new_forecasts'],0)
    def test_database_forbids_rewriting_prior_records(self):
        self.build(btc=0);self.run_shadow()
        end=ORIGIN+7*a.HOUR
        add_rows(self.root,'ETH-USD',[end-a.HOUR],end+timedelta(minutes=1),1,2200.)
        self.run_shadow(end+timedelta(minutes=2))
        with closing(sqlite3.connect(self.root/a.STATE/'issued.db')) as con, con:
            for table in ('forecasts','outcomes','decisions','metadata'):
                self.assertGreater(con.execute('SELECT count(*) FROM '+table).fetchone()[0],0)
                with self.assertRaises(sqlite3.IntegrityError):con.execute('DELETE FROM '+table)
    def test_exact_endpoint_and_revisions_append_only(self):
        self.build(btc=0);self.run_shadow();prior=self.forecasts()
        # Six-hour target is origin+7h, matching the existing h+1 time convention.
        end=ORIGIN+7*a.HOUR
        add_rows(self.root,'ETH-USD',[end-a.HOUR],end+timedelta(minutes=1),1,2200.)
        self.assertEqual(self.run_shadow(end)['totals']['outcomes'],0) # not received yet
        r=self.run_shadow(end+timedelta(minutes=2))
        self.assertEqual(r['totals']['outcomes'],1)
        add_rows(self.root,'ETH-USD',[end-a.HOUR],end+timedelta(minutes=3),2,2300.)
        self.assertEqual(self.run_shadow(end+timedelta(minutes=4))['totals']['outcomes'],2)
        self.assertEqual(self.forecasts(),prior)
    def test_missing_endpoint_not_replaced_by_neighbour(self):
        self.build(btc=0);self.run_shadow();end=ORIGIN+7*a.HOUR
        add_rows(self.root,'ETH-USD',[end-2*a.HOUR],end+timedelta(minutes=1),1,2200.)
        r=self.run_shadow(end+timedelta(minutes=2))
        self.assertEqual(r['totals']['outcomes'],0)
        self.assertGreater(r['decision']['settlement_pending']['missing_endpoint'],0)
    def test_snapshot_and_restore_bytes(self):
        self.build(btc=0);self.run_shadow()
        dest=Path(self.tmp.name)/'export'
        self.assertTrue(a.snapshot(self.root/a.STATE,dest));a.validate_state(dest)
        self.assertEqual(a.row_sets(self.root/a.STATE),a.row_sets(dest))
    def test_snapshot_rejects_durable_regression(self):
        self.build(btc=0);self.run_shadow()
        older=Path(self.tmp.name)/'old';a.snapshot(self.root/a.STATE,older)
        self.run_shadow(NOW+timedelta(minutes=1))
        with self.assertRaisesRegex(ValueError,'regress'):a.snapshot(older,self.root/a.STATE)
    def test_restore_refuses_partial_established_audit_state(self):
        def git(args,**kwargs):
            if 'rev-parse' in args:return b'commit\n'
            return (a.GIT_PATH+'/manifest.json\n').encode()
        with self.assertRaisesRegex(ValueError,'incomplete'):a.restore(self.root,git=git)
    def test_restore_keeps_newer_local_state(self):
        self.build(btc=0);self.run_shadow()
        older=Path(self.tmp.name)/'old';a.snapshot(self.root/a.STATE,older)
        self.run_shadow(NOW+timedelta(minutes=1))
        def git(args,**kwargs):
            if 'rev-parse' in args:return b'commit\n'
            if 'ls-tree' in args:return ''.join(a.GIT_PATH+'/'+n+'\n' for n in a.FILES).encode()
            return (older/args[-1].rsplit('/',1)[-1]).read_bytes()
        self.assertIn('local_snapshot',a.restore(self.root,git=git))
    def test_restore_from_audit_replays_no_forecasts(self):
        self.build(btc=0);self.run_shadow()
        saved=Path(self.tmp.name)/'saved';a.snapshot(self.root/a.STATE,saved)
        shutil.rmtree(self.root/a.STATE)
        def git(args,**kwargs):
            if 'rev-parse' in args:return b'commit\n'
            if 'ls-tree' in args:return ''.join(a.GIT_PATH+'/'+n+'\n' for n in a.FILES).encode()
            return (saved/args[-1].rsplit('/',1)[-1]).read_bytes()
        self.assertEqual(a.restore(self.root,git=git),'audit_state_restored')
        self.assertEqual(a.row_sets(self.root/a.STATE),a.row_sets(saved))


if __name__=='__main__':unittest.main()
