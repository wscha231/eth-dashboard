"""Fault injection on COPIES of a pinned snapshot; never an actual prospective forecast."""
from contextlib import closing
from datetime import datetime, timedelta, timezone
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts import availability_shadow as a


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    source=args.source
    before={n:a.file_sha(source/n) for n in ('observations.db','issued.db','signals.json')}
    origin=datetime(2026,9,19,6,tzinfo=timezone.utc)
    scenarios=[('clean','not_needed',0),('btc_missing','eligible',6),
               ('eth_old_gap','eligible',6),('eth_recent_gap','blocked',0),
               ('latest_eth_missing','blocked',0),('late_eth_receipt','blocked',0),
               ('deadline','blocked',0),('bad_eth_hash','error',0)]
    results=[]
    for name,expected_status,expected_new in scenarios:
        with tempfile.TemporaryDirectory(prefix='ef-fault-') as tmp:
            root=Path(tmp)/'copy';shutil.copytree(source,root)
            with closing(sqlite3.connect(root/'observations.db')) as con,con:
                if name=='btc_missing':con.execute("DELETE FROM bars WHERE product='BTC-USD' AND open_time=?",((origin-a.HOUR).isoformat(),))
                elif name in ('eth_old_gap','eth_recent_gap','latest_eth_missing'):
                    lag={'eth_old_gap':200,'eth_recent_gap':50,'latest_eth_missing':1}[name]
                    con.execute("DELETE FROM bars WHERE product='ETH-USD' AND open_time=?",((origin-lag*a.HOUR).isoformat(),))
                elif name=='late_eth_receipt':con.execute("UPDATE bars SET observed_at=? WHERE product='ETH-USD' AND open_time=?",((origin+timedelta(minutes=25)).isoformat(),(origin-a.HOUR).isoformat()))
                elif name=='bad_eth_hash':con.execute("UPDATE bars SET content_hash='corrupt' WHERE product='ETH-USD' AND open_time=?",((origin-a.HOUR).isoformat(),))
            copied_before={n:a.file_sha(root/n) for n in before}
            when=origin+timedelta(minutes=55 if name=='deadline' else 20)
            try:
                report=a.run(root,clock=lambda:when)
                status,new=report['decision']['status'],report['decision']['new_forecasts']
                detail=report['decision']['reason']
                duplicate=a.run(root,clock=lambda:when+timedelta(seconds=1))
                assert duplicate['decision']['new_forecasts']==0
                duplicate_ok=True
            except ValueError as exc:
                if name!='bad_eth_hash':raise
                status,new,detail,duplicate_ok='error',0,str(exc),None
            assert status==expected_status,(name,status)
            assert new==expected_new,(name,new)
            unchanged=copied_before=={n:a.file_sha(root/n) for n in copied_before}
            assert unchanged
            results.append({'scenario':name,'status':status,'new_shadow_records':new,
                            'reason':detail,'core_source_bytes_unchanged':unchanged,
                            'duplicate_did_not_reissue':duplicate_ok,
                            'evidence_role':'simulation_on_copied_past_state_not_live'})
    assert before=={n:a.file_sha(source/n) for n in before}
    value={'source':'event-final-state from run35426198641/artifact10578748686',
           'original_artifact_sha256':'41eaaf798bf82617ec148eeb75869740ab2bef20c1ad9d24146efb1d8bb202a0',
           'protected_source_hashes':before,'all_originals_unchanged':True,'scenario_count':len(results),
           'scenarios':results,'promotion':'none','public_delivery':'not_tested',
           'warning':'Synthetic missingness on historical copies is NOT prospective evidence or a live-uptime measurement.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value,indent=2))

if __name__=='__main__':main()
