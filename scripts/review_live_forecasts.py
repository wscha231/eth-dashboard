"""Weekly review of immutable issued forecasts; no model fitting or reselection."""
from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd


def review(db):
    started=time.perf_counter()
    conn=sqlite3.connect(f"file:{Path(db).resolve()}?mode=ro",uri=True)
    try:
        frame=pd.read_sql_query("""SELECT f.horizon_days, COALESCE(f.time_contract,'legacy') AS time_contract,
            f.forecast_target_timestamp_utc, f.regression_predicted_return,
            f.regression_lower_close_10, f.regression_upper_close_90,
            a.actual_return,a.actual_close,a.direction_correct,a.brier_contribution,a.evaluation_version
            FROM forecasts f JOIN actuals a ON a.forecast_id=f.forecast_id
            WHERE a.evaluation_version='closed_bar_v2'
            ORDER BY f.forecast_target_timestamp_utc""",conn)
    finally:
        conn.close()
    groups=[]
    for (h,contract),g in frame.groupby(["horizon_days","time_contract"]):
        valid=g.regression_predicted_return.notna() & g.actual_return.notna()
        loss=(g.loc[valid,"regression_predicted_return"]-g.loc[valid,"actual_return"]).abs()
        baseline=g.loc[valid,"actual_return"].abs()
        interval=g.regression_lower_close_10.notna() & g.regression_upper_close_90.notna()
        coverage=((g.actual_close>=g.regression_lower_close_10)&(g.actual_close<=g.regression_upper_close_90))[interval]
        groups.append({"horizon":int(h),"time_contract":contract,"resolved_count":len(g),
            "signal_count":int(g.direction_correct.notna().sum()),"correct_count":int(g.direction_correct.fillna(0).sum()),
            "abstain_count":int(g.direction_correct.isna().sum()),
            "return_mae":float(loss.mean()) if len(loss) else None,
            "mae_skill_vs_no_change":float(1-loss.mean()/baseline.mean()) if len(loss) and baseline.mean()>0 else None,
            "stored_interval_coverage":float(coverage.mean()) if len(coverage) else None,
            "interval_rows":len(coverage),"latest_target":str(g.forecast_target_timestamp_utc.max()),
            "independent_30d_blocks_approx":len(g)//30,
            "promotion":"no automatic promotion from live review"})
    return {"generated_at":dt.datetime.now(dt.timezone.utc).isoformat(),"evaluation_version":"closed_bar_v2",
            "groups":groups,"runtime_seconds":time.perf_counter()-started,
            "note":"Legacy and UTC bar-end forecasts are separate. Overlapping targets are not independent bets."}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db",default="forecast_site/predictions.db")
    parser.add_argument("--output",default="forecast_site/public/live_review.json")
    args=parser.parse_args()
    result=review(args.db)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
