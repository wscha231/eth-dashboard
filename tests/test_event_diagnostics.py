import copy
import json
import subprocess
from pathlib import Path

import pytest

from signal_pipeline.diagnostics import export_archive, live_point, validate_object
from signal_pipeline.evaluate import metrics


def row():
    return dict(slot='2026-01-01T00:00:00+00:00',target_end='2026-01-01T07:00:00+00:00',
                reference_price=100,actual_price=110,**{'return':__import__('math').log(1.1)},
                q10=-.1,q50=0.,q90=.2,terminal=2,p_down=.1,p_flat=.2,p_up=.7,
                up=1,down=0,hit_up=.8,hit_down=.2,threshold_up=.7,threshold_down=.7)


def test_archive_integrity_full_count_and_pending(tmp_path):
    rows=[{**row(),'slot':f'2026-01-{i:02}T00:00:00+00:00'} for i in range(1,21)]
    rows += [{**row(),'settled':False},{**row(),'eligible':False}]
    ref=export_archive(tmp_path,{'6':rows},kind='published',as_of='2026-02-01T00:00Z')
    manifest=validate_object(tmp_path,ref);h=manifest['horizons']['6']
    assert (h['rows'],h['scored'],h['pending'],h['excluded_publication'])==(22,20,1,1)
    shard=h['shards'][0];assert len(validate_object(tmp_path,shard)['points'])==22
    original=copy.deepcopy(rows)
    assert export_archive(tmp_path,{'6':rows},kind='published',as_of='2026-02-01T00:00Z')==ref
    assert rows==original
    (tmp_path/shard['path']).write_text('{}')
    with pytest.raises(ValueError,match='integrity'):validate_object(tmp_path,shard)
    with pytest.raises(ValueError,match='unsafe'):validate_object(tmp_path,{'path':'../issued.db'})


def test_live_rows_keep_original_and_timeliness():
    record={**row(),'horizon_seconds':21600,'price_quantiles':[90,100,120],
            'terminal_down_flat_up':[.1,.2,.7],'alert_thresholds':{'up':.7,'down':.7},
            'window_start':'2026-01-01T01:00:00+00:00','published_at':'2026-01-01T01:01:00+00:00',
            'outcome':None,'truth_revision':None,'forecast_id':'immutable'}
    before=copy.deepcopy(record);point=live_point(record)
    assert not point['eligible'] and not point['settled'] and point['forecast_id']=='immutable'
    assert record==before


def test_browser_and_python_metrics_agree_with_known_truth():
    rows=[row(),{**row(),'terminal':0,'up':0,'down':1,'return':-.1}]
    js=Path('forecast_site/public/event_diagnostics.js').read_text()
    script="global.window={};global.document={addEventListener(){}};"+js+"\nconsole.log(JSON.stringify(window.EventDiagnosticsMath.score("+json.dumps(rows)+")));"
    s=json.loads(subprocess.check_output(['node','-e',script],text=True));p=metrics(rows)
    assert s['usd']==pytest.approx(p['price_mae_usd'])
    assert s['mae']==pytest.approx(p['return_mae_pp'])
    assert s['rmse']==pytest.approx(p['return_rmse_pp'])
    assert s['interval']==pytest.approx(p['interval_score80_pp'])
    assert s['matrix']==p['confusion_actual_by_predicted']
    assert s['brier']==pytest.approx(p['event_brier'])
    assert s['up']['ap']==pytest.approx(p['up']['pr_auc'])
    assert s['balanced'] is None


def test_filter_inclusive_dates_and_unknown_regime():
    script="global.window={};global.document={addEventListener(){}};"+Path('forecast_site/public/event_diagnostics.js').read_text()+"""
const f=window.EventDiagnosticsMath.filter;
const rows=[{slot:'2026-01-01T23:00:00Z',trend_state:'up'},{slot:'2026-01-02T00:00:00Z',trend_state:'unknown'}];
if(f(rows,'2026-01-01','2026-01-01','all').length!==1)throw Error('inclusive date boundary');
if(f(rows,'','','trend_up').length!==1)throw Error('unknown regime counted');
if(window.EventDiagnosticsMath.score([]).n!==0)throw Error('empty scored');
"""
    subprocess.run(['node','-e',script],check=True)


def test_legacy_artifact_upgrades_full_probabilities_without_touching_ledger(tmp_path):
    import pandas as pd
    from scripts.event_state import merge_research
    from signal_pipeline.data import connect
    source=tmp_path/'source';target=tmp_path/'target';source.mkdir();target.mkdir()
    with connect(source): pass
    (source/'source_status.json').write_text('{}')
    (source/'replay.json').write_text(json.dumps({'horizons':{'6':{'points':[row()]}},'data_as_of':'2026-01-02T00:00Z'}))
    (source/'replay').mkdir()
    pd.DataFrame([{**row(),'model':m} for m in ('selected','climatology')]).to_csv(source/'replay/h6.csv.gz',index=False)
    (target/'issued.db').write_bytes(b'immutable sentinel')
    merge_research(source,target)
    assert (target/'issued.db').read_bytes()==b'immutable sentinel'
    d=json.loads((target/'replay.json').read_text())
    manifest=validate_object(target,d['archive'])
    points=validate_object(target,manifest['horizons']['6']['shards'][0])['points']
    assert len(points)==1 and points[0]['p_up']==.7 and points[0]['baseline']['p_up']==.7


def test_forward_comparison_uses_timely_matching_truth():
    from signal_pipeline.diagnostics import compare_published
    r={**row(),'horizon_seconds':21600,'price_quantiles':[90,100,120],
       'terminal_down_flat_up':[.1,.2,.7],'alert_thresholds':{'up':.7,'down':.7},
       'window_start':'2026-01-01T01:00:00+00:00','published_at':'2026-01-01T00:10:00+00:00',
       'outcome':{k:row()[k] for k in ('actual_price','return','up','down','terminal')},'model_version':'fixed'}
    assert compare_published([r],[r])['6']['paired_rows']==1
    late={**r,'published_at':'2026-01-01T01:01:00+00:00'}
    assert compare_published([r],[late])['6']['paired_rows']==0
    revised={**r,'outcome':{**r['outcome'],'return':.2}}
    assert compare_published([r],[revised])['6']['paired_rows']==0


def test_frontend_renders_verified_archive_and_clears_empty_filter(tmp_path):
    pending={**row(),'slot':'2026-01-02T00:00:00+00:00','target_end':'2026-01-02T07:00:00+00:00',
             'settled':False,'return':None,'actual_price':None,'q50':.1}
    rows=[row(),pending,{**pending,'eligible':False}]
    ref=export_archive(tmp_path,{str(h):rows for h in (6,24,72,168,336,720)},kind='published',as_of='2026-02-01T00:00Z')
    source=Path('forecast_site/public/event_diagnostics.js').read_text()
    script=r'''
const fs=require('fs');global.crypto=require('crypto').webcrypto;
const elements={},listeners=[],plots={};
global.window={};global.document={getElementById(id){return elements[id]??={id,value:id==='diagnostic-regime'?'all':'',textContent:'',innerHTML:'',removeAttribute(k){delete this[k];}};},addEventListener(_,fn){listeners.push(fn);}};
global.Chart=class{constructor(el,config){plots[el.id]=config;}destroy(){}};
global.fetch=async path=>{const b=fs.readFileSync(ROOT+'/'+path);return {ok:true,arrayBuffer:async()=>b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength)}};
'''.replace('ROOT',json.dumps(str(tmp_path)))+source+r'''
(async()=>{
listeners.forEach(f=>f());
window.updateEventDiagnostics(PAYLOAD,6);
elements['diagnostic-source'].onchange({target:{value:'published'}});
await new Promise(r=>setTimeout(r,50));
if(!elements['diagnostic-status'].textContent.includes('1 scored / 3 records'))throw Error(elements['diagnostic-status'].textContent);
const series=plots['diagnostic-price'].data.datasets;
if(series[0].data.length!==2||series[0].data[1]!==null)throw Error('pending actual invented or excluded record plotted');
if(Math.abs(series[1].data[1]-100*Math.exp(.1))>1e-9)throw Error('pending forecast lost or rebased');
if(!elements['diagnostic-confusion'].innerHTML.includes('Higher'))throw Error('confusion missing');
elements['diagnostic-from'].value='2027-01-01';elements['diagnostic-from'].onchange();
await new Promise(r=>setTimeout(r,50));
if(!elements['diagnostic-status'].textContent.includes('0 scored / 0 records'))throw Error('stale metrics');
if(plots['diagnostic-price'].data.datasets[0].data.length!==0)throw Error('stale chart');
})().catch(e=>{console.error(e);process.exitCode=1});
'''.replace('PAYLOAD',json.dumps({'evidence_archives':{'published':ref}}))
    subprocess.run(['node','-e',script],check=True,capture_output=True,text=True)


def test_live_dollar_error_and_independent_count_do_not_depend_on_ledger_order():
    from signal_pipeline.evaluate import prospective_report
    base={**row(),'horizon_seconds':21600,'price_quantiles':[90,100,120],
          'terminal_down_flat_up':[.1,.2,.7],'alert_thresholds':{'up':.7,'down':.7},
          'window_start':'2026-01-01T01:00:00+00:00','published_at':'2026-01-01T00:10:00+00:00',
          'outcome':{k:row()[k] for k in ('actual_price','return','up','down','terminal')}}
    base['baseline']={k:base[k] for k in ('price_quantiles','terminal_down_flat_up','alert_thresholds','hit_up','hit_down')}
    later={**base,'slot':'2026-01-02T00:00:00+00:00','target_end':'2026-01-02T07:00:00+00:00',
           'window_start':'2026-01-02T01:00:00+00:00','published_at':'2026-01-02T00:10:00+00:00'}
    report=prospective_report([later,base])['6']
    assert report['nonoverlap_resolved']==2
    assert report['metrics']['price_mae_usd']==pytest.approx(10.)
    assert report['baseline']['price_mae_usd']==pytest.approx(10.)
