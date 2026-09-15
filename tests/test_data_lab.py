"""DATA tests: contracts, leakage, immutability, feature causality and interoperability."""
import copy
import hashlib
import json
from pathlib import Path
import sqlite3
import zipfile

import numpy as np
import pandas as pd
import pytest

from data_lab.core import (MetricSpec, ObservationStore, canonical, missing_runs, normalize_observation,
    point_in_time_join, read_bars, read_zip, relative, sha256, utc, validate_handoff)
from data_lab.features import extension_features, exact_rolling_max_time, coverage, incumbent_features
from data_lab.release import build_release, validate_release, load_feature_snapshot


@pytest.fixture
def spec():
    return MetricSpec("venue_eth_oi", "venue", "venue", "ETHUSDT-perp", "ETH", 7200)


def observation(spec, *, event="2026-01-01T00:00:00Z", receipt="2026-01-01T00:01:00Z", value=100., revision=1, **kwargs):
    return {"series_id":spec.series_id,"provider":spec.provider,"venue":spec.venue,
        "instrument":spec.instrument,"unit":spec.unit,"event_time":event,"published_at":None,
        "available_at":receipt,"ingested_at":receipt,"revision":revision,"value":value,
        "missing_reason":None if value is not None else "provider_missing","kind":"observed",
        "vintage_mode":"observed_vintages","license_status":"approved_for_intended_use",
        "raw_sha256":"a"*64,**kwargs}


@pytest.fixture(scope="module")
def bars():
    idx = pd.date_range("2025-01-01T00:00:00Z",periods=1600,freq="h")
    rng = np.random.default_rng(72);frames=[]
    for product,base in (("ETH-USD",2500),("BTC-USD",60000)):
        prices=base*np.exp(np.cumsum(rng.normal(0,.003,len(idx))))
        frames.append(pd.DataFrame({"product":product,"open_time":idx-pd.Timedelta(hours=1),
            "close_time":idx,"observed_at":idx+pd.Timedelta(seconds=1,nanoseconds=113),
            "revision":1,"open":prices,"high":prices*1.01,"low":prices*.99,"close":prices,
            "volume":rng.uniform(10,1000,len(idx)),"content_hash":"a"*64,"raw_hash":"b"*64}))
    return pd.concat(frames,ignore_index=True)


@pytest.mark.parametrize("name",["", ".", "../a", "a/../b", "/a", "a\\b", "a//b", "a/", "C:/x", "a\x00b"])
def test_unsafe_paths(name):
    with pytest.raises(ValueError):relative(name)


@pytest.mark.parametrize("value",["2026-01-01",pd.NaT,"wrong"])
def test_explicit_timezone(value):
    with pytest.raises((ValueError,TypeError)):utc(value)


def test_offset_utc_and_nanosecond():
    assert utc("2026-01-01T09:00:00.000000113+09:00")==utc("2026-01-01T00:00:00.000000113Z")


@pytest.mark.parametrize("field,value",[("unit","USD"),("provider","other"),("venue","other"),
    ("instrument","BTCUSDT-perp"),("raw_sha256","bad"),("value",float("nan")),("value",float("inf")),
    ("value",True),("revision",0),("revision",1.5),("revision",True),("kind","unknown"),
    ("vintage_mode","latest"),("license_status","")])
def test_observation_validation(spec,field,value):
    r=observation(spec);r[field]=value
    with pytest.raises(ValueError):normalize_observation(r,spec)


def test_zero_is_real_and_null_is_missing(spec):
    rows=[observation(spec,value=0.)]
    assert point_in_time_join(rows,["2026-01-01T01:00:00Z"],spec).iloc[0].value==0
    r=observation(spec,value=None);r["missing_reason"]=None
    with pytest.raises(ValueError):normalize_observation(r,spec)


def test_future_revision_filtered_before_selection(spec):
    r1=observation(spec,value=1)
    r2=observation(spec,receipt="2026-01-01T01:00:00.000000001Z",value=99,revision=2)
    result=point_in_time_join([r1,r2],["2026-01-01T01:00:00Z","2026-01-01T01:00:00.000000001Z"],spec)
    assert result.value.tolist()==[1,99]


def test_old_revision_does_not_replace_new_event(spec):
    rows=[observation(spec),observation(spec,event="2026-01-01T01:00:00Z",receipt="2026-01-01T01:01:00Z",value=200),
          observation(spec,receipt="2026-01-01T01:30:00Z",value=999,revision=2)]
    result=point_in_time_join(rows,["2026-01-01T01:40:00Z"],spec)
    assert result.iloc[0].value==200


def test_null_latest_not_filled_from_old(spec):
    rows=[observation(spec),observation(spec,event="2026-01-01T01:00:00Z",receipt="2026-01-01T01:01:00Z",value=None)]
    result=point_in_time_join(rows,["2026-01-01T01:40:00Z"],spec)
    assert pd.isna(result.iloc[0].value)
    assert result.iloc[0].missing_reason=="provider_missing"


def test_decision_order_and_duplicates_preserved(spec):
    result=point_in_time_join([observation(spec)],["2026-01-01T01:00:00Z","2025-12-31T23:00:00Z","2026-01-01T01:00:00Z"],spec)
    assert result.value.iloc[0]==result.value.iloc[2]==100
    assert pd.isna(result.value.iloc[1])


@pytest.mark.parametrize("change,reason",[({"kind":"synthetic"},"non_observed_kind_blocked"),
    ({"kind":"estimated"},"non_observed_kind_blocked"),({"vintage_mode":"reconstructed"},"historical_reconstruction_blocked"),
    ({"license_status":"unreviewed"},"rights_unapproved")])
def test_live_blockers(spec,change,reason):
    row=observation(spec,**change)
    assert point_in_time_join([row],["2026-01-01T01:00:00Z"],spec).iloc[0].missing_reason==reason


def test_staleness_is_event_age_not_retrieval_age(spec):
    row=observation(spec,receipt="2026-01-02T00:00:00Z")
    assert point_in_time_join([row],["2026-01-02T00:00:00Z"],spec).iloc[0].missing_reason=="stale_observation"


def test_ingestion_gate_even_when_published_earlier(spec):
    row=observation(spec,receipt="2026-01-01T00:05:00Z",published_at="2026-01-01T00:00:00Z",available_at="2026-01-01T00:00:00Z")
    assert pd.isna(point_in_time_join([row],["2026-01-01T00:02:00Z"],spec).iloc[0].value)


def test_append_idempotent_conflict_and_transaction_rollback(spec,tmp_path):
    store=ObservationStore(tmp_path/"obs.db");one=observation(spec)
    assert store.append([one],spec)=={"added":1,"reused":0}
    assert store.append([one],spec)=={"added":0,"reused":1}
    new=observation(spec,event="2026-01-01T01:00:00Z",receipt="2026-01-01T01:01:00Z")
    bad=observation(spec,value=999)
    with pytest.raises(ValueError):store.append([new,bad],spec)
    assert len(store.read(spec))==1
    assert store.read(spec)[0]["value"]==100


def test_duplicate_revision_rejected(spec):
    with pytest.raises(ValueError):point_in_time_join([observation(spec)]*2,["2026-01-01T01:00:00Z"],spec)


def test_exact_time_rolling_does_not_round_down():
    dates=pd.to_datetime(["2026-01-01T00:00:00.000000113Z","2026-01-01T00:00:00.000000114Z"])
    result=exact_rolling_max_time(pd.Series(dates),2)
    assert result.iloc[-1]==dates[-1]
    assert pd.isna(result.iloc[0])


def test_extension_count_and_prefix_causality(bars):
    full,specs=extension_features(bars)
    t=full.index[1100]
    truncated,_=extension_features(bars.loc[bars.close_time<=t])
    assert len(specs)==18
    pd.testing.assert_frame_equal(full.loc[:t],truncated)
    assert all(s["research_status"]=="implemented_not_performance_validated" for s in specs)


def test_future_price_change_does_not_affect_past(bars):
    initial,_=extension_features(bars)
    t=initial.index[1100];mutated=bars.copy();mask=mutated.close_time>t
    mutated.loc[mask,["open","high","low","close"]]*=10
    later,_=extension_features(mutated)
    pd.testing.assert_frame_equal(initial.loc[:t],later.loc[:t])


def test_gaps_are_not_zero_returns(bars):
    t=bars.close_time.sort_values().unique()[800]
    missing=bars.loc[~(bars["product"].eq("ETH-USD")&bars.close_time.eq(t))]
    frame,_=extension_features(missing)
    assert pd.isna(frame.loc[t,"x_rv_24"])
    assert frame.loc[t:t+pd.Timedelta(hours=24),"x_rv_24"].isna().all()
    assert pd.notna(frame.loc[t+pd.Timedelta(hours=25),"x_rv_24"])


def test_btc_gap_does_not_drop_eth_only_features(bars):
    t=bars.close_time.sort_values().unique()[800]
    missing=bars.loc[~(bars["product"].eq("BTC-USD")&bars.close_time.eq(t))]
    frame,_=extension_features(missing)
    assert pd.isna(frame.loc[t,"x_rel_24"])
    assert pd.notna(frame.loc[t,"x_rv_24"])


def test_feature_formula_variance_and_relative(bars):
    frame,_=extension_features(bars);t=frame.index[-1]
    eth=bars.loc[bars["product"].eq("ETH-USD")].set_index("close_time")
    ret=np.log(eth.close).diff()
    assert frame.loc[t,"x_rv_24"]==pytest.approx(ret.iloc[-24:].pow(2).sum())
    assert frame.loc[t,"x_efficiency_24"]==pytest.approx(ret.iloc[-24:].sum()/ret.iloc[-24:].abs().sum())
    assert 0<=frame.loc[t,"x_downside_share_24"]<=1


def test_zero_denominators_remain_missing(bars):
    b=bars.copy();b.loc[:,["open","high","low","close"]]=100;b["volume"]=0
    f,_=extension_features(b)
    for c in ["x_close_location","x_volume_z_prior168","x_efficiency_24","x_rv_ratio_24_720","x_turnover_illiquidity_24"]:
        assert f[c].isna().all()


def test_coverage_and_exact_gaps():
    i=pd.date_range("2026-01-01T00:00:00Z",periods=6,freq="h")
    g=missing_runs(i[[0,3,5]],i)
    assert [x["rows"] for x in g]==[2,1]
    f=pd.DataFrame({"x":[np.nan,1,2,np.nan,np.nan,3]},index=i)
    c=coverage(f)["x"];assert c["valid_rows"]==3 and c["longest_missing_run_rows"]==2


def test_zip_traversal_and_hash_rejected(tmp_path):
    p=tmp_path/"bad.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr("../secret",b"x")
    with pytest.raises(ValueError):read_zip(p,sha256(p))
    with pytest.raises(ValueError):read_zip(p,"a"*64)


def test_zip_size_budget(tmp_path):
    p=tmp_path/"large.zip"
    with zipfile.ZipFile(p,"w") as z:z.writestr("data",b"xxxx")
    with pytest.raises(ValueError):read_zip(p,sha256(p),max_bytes=3)


def fixture_inputs(tmp_path,bars):
    root=tmp_path/"inputs";root.mkdir()
    with sqlite3.connect(root/"observations.db") as c:
        b=bars.copy().drop(columns="close_time")
        for col in ("open_time","observed_at"):b[col]=b[col].map(lambda t:t.isoformat())
        b.to_sql("bars",c,index=False)
    with zipfile.ZipFile(root/"research.zip","w") as z:z.writestr("fixture.json","{}")
    from data_lab.features import INCUMBENT_BUILDER_SHA256
    data={"release_id":"fixture-source","schema_version":1,"dataset_manifest_id":"fixture","dataset_manifest_hash":"a"*64,
        "feature_set_id":"signal_pipeline.build_features","feature_set_version":"test","code_sha":"b"*40,
        "data_cutoff_utc":bars.close_time.max().isoformat(),"generated_at_utc":bars.observed_at.max().isoformat(),
        "source_versions":{"feature_builder_sha256":INCUMBENT_BUILDER_SHA256},"coverage_by_feature":{"x":{}},
        "quality_status":"partial","vintage_mode":"reconstructed","license_status":"unreviewed",
        "checksums":{p:sha256(root/p) for p in ("observations.db","research.zip")},
        "partition_locations":{p:p for p in ("observations.db","research.zip")},"exclusions":[],"breaking_changes":[]}
    (root/"release.json").write_bytes(canonical(data))
    path=tmp_path/"handoff.zip";raw=canonical({"count":1,"features":[{"id":"candidate","status":"hypothesis"}]})
    name="research_20260915/feature_registry.json"
    with zipfile.ZipFile(path,"w") as z:
        z.writestr(name,raw)
        z.writestr("HANDOFF_MANIFEST.json",canonical({"files":{name:{"bytes":len(raw),"sha256":hashlib.sha256(raw).hexdigest()}}}))
    return root,path


def test_complete_release_repeat_integrity_and_model_contract(tmp_path,bars):
    root,handoff=fixture_inputs(tmp_path,bars)
    args=(root,sha256(root/"release.json"),handoff,sha256(handoff))
    one=build_release(*args,tmp_path/"one",generated_at="2026-09-15T08:00:00Z",builder_commit="b"*40)
    two=build_release(*args,tmp_path/"two",generated_at="2026-09-15T08:00:00Z",builder_commit="b"*40)
    assert one["release_sha256"]==two["release_sha256"]
    assert one["incumbent_features"]==32 and one["extension_features"]==18
    d=validate_release(tmp_path/"one",one["release_sha256"])
    assert d["quality_status"]=="partial" and d["license_status"]=="unreviewed_no_public_redistribution"
    before=sha256(root/"observations.db")
    with pytest.raises(FileExistsError):build_release(*args,tmp_path/"one",generated_at="2026-09-15T08:00:00Z",builder_commit="b"*40)
    assert sha256(root/"observations.db")==before
    f=load_feature_snapshot(tmp_path/"one",one["release_sha256"])
    assert len(f)==len(bars)//2
    with pytest.raises(ValueError):load_feature_snapshot(tmp_path/"one",one["release_sha256"],mode="prospective")
    (tmp_path/"one/extension_features.csv.gz").write_bytes(b"tampered")
    with pytest.raises(ValueError):validate_release(tmp_path/"one",one["release_sha256"])


def test_invalid_source_price_and_sidecar(tmp_path,bars):
    root,_=fixture_inputs(tmp_path,bars)
    path=root/"observations.db"
    kw={"cutoff":bars.close_time.max(),"generated":bars.observed_at.max()}
    _,report=read_bars(path,sha256(path),**kw)
    assert report["products"]["ETH-USD"]["missing_rows"]==0
    Path(str(path)+"-wal").write_bytes(b"uncommitted")
    with pytest.raises(ValueError):read_bars(path,sha256(path),**kw)
    Path(str(path)+"-wal").unlink()
    with sqlite3.connect(path) as con:con.execute("UPDATE bars SET close=-1 WHERE rowid=1")
    with pytest.raises(ValueError):read_bars(path,sha256(path),**kw)


def test_actual_availability_never_precedes_sources(bars):
    f,specs=extension_features(bars)
    for spec in specs:
        name=spec['feature_id'];valid=f[name].notna()
        assert f.loc[valid,name+'__available_at'].notna().all()
        assert (f.loc[valid,name+'__available_at']>=f.index[valid]+pd.Timedelta(seconds=1,nanoseconds=113)).all()


def test_missing_publication_remains_unknown(spec):
    row=normalize_observation(observation(spec),spec)
    assert row['published_at'] is None
    assert row['availability_basis']=='receipt_only_publication_unknown'
    assert row['eligible_at']==row['ingested_at']


@pytest.mark.parametrize('kwargs',[
    {'published_at':'2026-01-01T01:00:00Z'},
    {'available_at':'2025-12-31T23:00:00Z'},
    {'available_at':'2026-01-01T00:02:00Z'},
    {'ingested_at':'2026-01-01'},
])
def test_timestamp_order_rejected(spec,kwargs):
    with pytest.raises(ValueError):normalize_observation(observation(spec,**kwargs),spec)


def test_auxiliary_extra_fields_not_persisted(spec,tmp_path):
    row=observation(spec);row['api_key']='must_not_be_journaled'
    store=ObservationStore(tmp_path/'journal.db');store.append([row],spec)
    assert 'api_key' not in store.read(spec)[0]


def test_handoff_missing_manifest(tmp_path):
    p=tmp_path/'incomplete.zip'
    with zipfile.ZipFile(p,'w') as z:z.writestr('feature_registry.json','{}')
    with pytest.raises(ValueError):validate_handoff(p,sha256(p))
