import copy
import json
import sqlite3

import numpy as np
import pandas as pd
import pytest

from signal_pipeline.data import build_features, connect, ingest, parse_candles, read_bars, utc
from signal_pipeline.ledger import backup, history, issue, settle
from signal_pipeline.ledger import mark_verified
from signal_pipeline.evaluate import prospective_report
from signal_pipeline.models import labels, train_bundle, predict_models
from signal_pipeline.protocol import digest


def fixture_bars(hours=2400):
    rng = np.random.default_rng(21)
    opens = pd.date_range("2020-01-01", periods=hours, freq="h", tz="UTC")
    result = []
    for product in ("ETH-USD", "BTC-USD"):
        p = 200*np.exp(np.cumsum(rng.normal(0,.01,hours)))
        f = pd.DataFrame({"product":product, "open_time":opens, "close_time":opens+pd.Timedelta(hours=1),
                          "observed_at":opens+pd.Timedelta(hours=1,minutes=2),
                          "open":p, "high":p*1.01, "low":p*.99, "close":p, "volume":rng.uniform(1,500,hours),
                          "revision":1})
        f["content_hash"] = [digest([float(v)]) for v in p]
        result.append(f)
    return pd.concat(result, ignore_index=True)


def test_source_boundaries_units_revisions_and_original_receipt(tmp_path):
    t = int(utc("2020-01-01").timestamp())
    raw = json.dumps([[t,90,110,100,105,3], [t+3600,90,111,105,109,4]]).encode()
    f = parse_candles(raw,"2020-01-01","2020-01-02","2020-01-01T01:05Z")
    assert len(f) == 1  # unfinished candle is never available
    for timestamp in (t*1000, t*1000000, t+1):
        with pytest.raises((ValueError, OverflowError)):
            parse_candles(json.dumps([[timestamp,90,110,100,105,3]]),"2020-01-01","2020-01-02","2020-01-01T01:05Z")
    with connect(tmp_path) as con:
        ingest(con,tmp_path,"ETH-USD",raw,"2020-01-01","2020-01-02","2020-01-01T01:05Z")
        ingest(con,tmp_path,"ETH-USD",raw,"2020-01-01","2020-01-02","2020-01-01T01:10Z")
        corrected = json.dumps([[t,90,110,100,106,3]]).encode()
        ingest(con,tmp_path,"ETH-USD",corrected,"2020-01-01","2020-01-02","2020-01-01T01:20Z")
    assert len(read_bars(tmp_path)) == 1
    assert read_bars(tmp_path).iloc[0].revision == 2
    assert read_bars(tmp_path,as_of="2020-01-01T01:15Z").iloc[0].close == 105
    assert read_bars(tmp_path,as_of="2020-01-01T01:15Z").iloc[0].observed_at == utc("2020-01-01T01:05Z")


def test_features_are_causal_and_missing_bars_cannot_be_filled():
    bars = fixture_bars(); original = build_features(bars)
    cutoff = original.index[1200]
    changed = bars.copy(); future = changed.close_time > cutoff
    changed.loc[future,["open","high","low","close"]] *= 5
    pd.testing.assert_frame_equal(original.loc[:cutoff],build_features(changed).loc[:cutoff])
    missing = bars.drop(bars[(bars["product"]=="BTC-USD") & (bars.close_time==cutoff)].index)
    assert pd.isna(build_features(missing).loc[cutoff+pd.Timedelta(hours=100),"eth_ret_24"])


def test_path_labels_exclude_preissuance_bar_and_require_full_window():
    bars=fixture_bars(); features=build_features(bars); slot=features.index[1200]
    # A spike in the hour containing issuance is NOT a predicted future hit.
    e=(bars["product"]=="ETH-USD")
    bars.loc[e & (bars.open_time == slot), "high"] = 1e8
    before = labels(bars,features,24).loc[slot]
    bars.loc[e & (bars.open_time == slot+pd.Timedelta(hours=1)),"high"] = 1e8
    after=labels(bars,features,24).loc[slot]
    assert after.up == 1
    assert after.target_end == slot+pd.Timedelta(hours=25)
    assert labels(bars,features,24).iloc[-1][["up","down","terminal","return"]].isna().all()
    missing=bars.drop(bars[e & (bars.open_time == slot+pd.Timedelta(hours=2))].index)
    assert pd.isna(labels(missing,features,24).loc[slot,"up"])


def record(now="2020-04-01T12:05Z"):
    now=utc(now); slot=now.floor("h")
    return {"slot":slot.isoformat(),"input_cutoff":slot.isoformat(),"available_at":(slot+pd.Timedelta(minutes=2)).isoformat(),
            "window_start":(slot+pd.Timedelta(hours=1)).isoformat(),"target_end":(slot+pd.Timedelta(hours=25)).isoformat(),
            "horizon_seconds":86400,"model_version":"test-model","training_target_end":"2020-03-01T00:00:00+00:00",
            "validation_target_end":"2020-03-30T00:00:00+00:00","reference_price":200.,"log_barrier":.03,
            "terminal_down_flat_up":[.2,.5,.3],"hit_up":.6,"hit_down":.4,"price_quantiles":[180.,200.,230.],
            "alert_thresholds":{"up":.8,"down":.8}}


def test_immutable_issue_transaction_pending_truth_revision_and_backup(tmp_path):
    r=record(); a=issue(tmp_path,r,now="2020-04-01T12:05Z")
    changed=copy.deepcopy(r);changed["price_quantiles"]=[190.,205.,240.]
    assert issue(tmp_path,changed,now="2020-04-01T12:06Z") == a
    assert len(history(tmp_path))==1
    with sqlite3.connect(tmp_path/"issued.db") as con:
        assert con.execute("SELECT COUNT(*) FROM outbox").fetchone()[0]==1
    b=fixture_bars();now="2020-04-04T00:00Z"
    assert settle(tmp_path,b,now="2020-04-01T13:00Z")==0
    assert history(tmp_path)[0]["outcome"] is None
    assert settle(tmp_path,b,now=now)==1
    assert settle(tmp_path,b,now=now)==0
    window=(b["product"]=="ETH-USD") & (b.open_time==utc(r["window_start"]))
    b.loc[window,"high"]=1e8; b.loc[window,"content_hash"]="corrected-source"
    assert settle(tmp_path,b,now=now)==1
    assert history(tmp_path)[0]["truth_revision"]==2
    backup(tmp_path,tmp_path/"backup")
    assert history(tmp_path/"backup")==history(tmp_path)


@pytest.mark.parametrize("change", [
    {"available_at":"2020-04-01T12:10Z"}, {"training_target_end":"2020-04-02T00:00Z"},
    {"price_quantiles":[200,190,210]}, {"terminal_down_flat_up":[.5,.5,.5]},
    {"window_start":"2020-04-01T12:00Z"}, {"hit_up":float("nan")}])
def test_invalid_issuance_is_atomic(tmp_path,change):
    with pytest.raises(ValueError): issue(tmp_path,{**record(),**change},now="2020-04-01T12:05Z")
    assert history(tmp_path)==[]


def test_purged_month_selection_is_unchanged_by_future_outcomes():
    bars=fixture_bars(24*420); features=build_features(bars); y=labels(bars,features,24)
    cutoff=utc("2021-01-01")
    a=train_bundle(features,y,cutoff,24)
    assert utc(a["training_target_end"]) < cutoff-pd.Timedelta(hours=1)
    assert utc(a["validation_target_end"]) < cutoff-pd.Timedelta(hours=1)
    mutated=y.copy(); mutated.loc[mutated.target_end>=cutoff,["return","up","down","terminal"]]=[10,1,1,2]
    b=train_bundle(features,mutated,cutoff,24)
    assert a["choice"]==b["choice"] and a["validation_scores"]==b["validation_scores"]
    x=features.loc[[cutoff]]
    pa=predict_models(a["model"],x);pb=predict_models(b["model"],x)
    for name in pa:
        for output in pa[name]: np.testing.assert_array_equal(pa[name][output],pb[name][output])


def test_late_or_failed_publication_is_not_claimed_as_prospective(tmp_path):
    r=record(); issue(tmp_path,r,now="2020-04-01T12:05Z")
    assert prospective_report(history(tmp_path))=={}
    a=history(tmp_path)[0]
    mark_verified(tmp_path,[a["forecast_id"]],"release")  # actually verified now, years after fixture
    assert history(tmp_path)[0]["published_at"] is not None
    assert prospective_report(history(tmp_path))=={}


def test_research_restore_never_replaces_actual_issued_ledger(tmp_path):
    from scripts.event_state import merge_research
    research=tmp_path/'research';live=tmp_path/'live';research.mkdir()
    raw=json.dumps([[1577836800,90,110,100,105,3]]).encode()
    with connect(research) as con:ingest(con,research,'ETH-USD',raw,'2020-01-01','2020-01-02','2020-01-01T01:05Z')
    (research/'replay.json').write_text('{}');(research/'source_status.json').write_text('{}')
    (research/'models').mkdir();(research/'issued.db').write_text('must never be loaded')
    issued=issue(live,record(),now='2020-04-01T12:05Z')
    merge_research(research,live);merge_research(research,live)
    assert history(live)[0]['forecast_id']==issued['forecast_id']
    assert len(read_bars(live))==1


def test_external_release_checks_stale_delayed_or_mismatched_outputs():
    from scripts.verify_event_site import verify
    d={'schema_version':1,'release_id':'one','generated_at':'2020-04-01T12:08:00+00:00','status':'ready','current':[]}
    assert verify(d,d,utc('2020-04-01T12:10Z'))
    with pytest.raises(ValueError):verify(d,{**d,'release_id':'two'},utc('2020-04-01T12:10Z'))
    with pytest.raises(ValueError):verify(d,d,utc('2020-04-01T14:00Z'))
    with pytest.raises(ValueError):verify({**d,'status':'delayed'},now=utc('2020-04-01T12:10Z'),require_ready=True)
