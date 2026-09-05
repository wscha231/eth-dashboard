"""Cheap scheduler idempotency check, before collecting or loading ML packages."""
from __future__ import annotations
import argparse
import datetime as dt
from pathlib import Path
import sqlite3


def already_published(db, phase, now=None):
    now = now or dt.datetime.now(dt.timezone.utc)
    if now.hour == 0 and now.minute < 15:
        return False
    origin=now.date().isoformat()
    if not Path(db).exists():
        return False
    conn=sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro",uri=True)
    try:
        row=conn.execute("SELECT r.run_id FROM forecast_runs r JOIN forecasts f ON f.run_id=r.run_id WHERE DATE(r.input_timestamp_utc)=? AND r.model_phase=? AND f.time_contract='utc_bar_end_v2' AND f.horizon_days IN (7,30) GROUP BY r.run_id HAVING COUNT(DISTINCT f.horizon_days)=2",(origin,phase)).fetchone()
        return row is not None
    except sqlite3.OperationalError:
        return False  # legacy DB: no v2 run can exist
    finally:
        conn.close()


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",default="forecast_site/predictions.db")
    parser.add_argument("--phase",default="phase6_closed_daily")
    parser.add_argument("--github-output")
    args=parser.parse_args()
    value=already_published(args.db,args.phase)
    if args.github_output:
        with open(args.github_output,"a") as f:
            f.write(f"skip={str(value).lower()}\n")
    print(f"Already published current closed-bar origin: {value}")
