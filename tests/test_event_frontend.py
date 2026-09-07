from pathlib import Path
import shutil
import subprocess

import pytest


def test_event_cards_keep_reference_prices_and_distinguish_two_event_probabilities():
    if not shutil.which('node'):pytest.skip('Node required for frontend checks')
    script=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes={};const element=()=>({children:[],className:'',appendChild(x){this.children.push(x)},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}});
const fixture={schema_version:1,generated_at:new Date().toISOString(),expected_slot:new Date().toISOString(),status:'ready',current:[{horizon_seconds:86400,input_cutoff:new Date().toISOString(),issued_at:new Date().toISOString(),window_start:'2026-09-06T13:00:00Z',target_end:'2026-09-07T13:00:00Z',reference_price:2000,price_quantiles:[1800,2100,2500],hit_up:.8,hit_down:.6,upper_barrier_price:2200,lower_barrier_price:1800,terminal_down_flat_up:[.2,.3,.5],selected_model:'catboost'}],recent_issued:[],prospective:{},replay_generated_at:'v1'};
const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id] ||= element(),createElement:element,addEventListener(){}},fetch:async url=>({ok:true,json:async()=>url==='signals.json'?fixture:{generated_at:'v1',horizons:{}}}),Date,Intl,setInterval(){}});
vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
context.window.loadEventForecasts().then(()=>{
 assert.equal(nodes['ref-price'].textContent,'$2,000');
 const card=nodes['event-current'].children[0];assert.equal(card.children[1].textContent,'$2,100');
 assert.match(card.children[3].children[0].textContent,/80.0%/);assert.match(card.children[3].children[1].textContent,/60.0%/);
 assert.match(nodes['model-phase'].textContent,/연구 베타/);
});
'''
    subprocess.run(['node','-e',script],cwd=Path(__file__).resolve().parents[1],check=True)


def test_publish_jobs_persist_actual_ledger_first_and_do_not_restore_it_from_research():
    workflow=Path('.github/workflows/event_hourly.yml').read_text()
    assert workflow.index('bash scripts/persist_event_ledger.sh') < workflow.index('bash scripts/publish_events.sh')
    assert 'group: daily-forecast' in workflow
    assert 'data/event-ledger:lake/event-ledger/issued.db' in workflow
    assert 'retention-days: 2' in workflow
    publisher=Path('scripts/publish_events.sh').read_text()
    assert publisher.index('verify_event_site.py --expected') < publisher.index('mark_verified(')


def test_loaded_forecasts_age_during_connection_failure_and_recover_without_rewriting_prices():
    script=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let clock=Date.parse('2026-09-07T06:38:00Z'),fail=false;
class Clock extends Date {constructor(...args){super(...(args.length?args:[clock]))}static now(){return clock}}
const nodes={},timers=[];
const element=()=>({children:[],className:'',appendChild(x){this.children.push(x)},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}});
const fixture={schema_version:1,generated_at:'2026-09-07T06:08:00Z',expected_slot:'2026-09-07T06:00:00Z',
 status:'ready',current:[6,24,72,168,336,720].map(h=>({horizon_seconds:h*3600,input_cutoff:'2026-09-07T06:00:00Z',
 issued_at:'2026-09-07T06:08:00Z',window_start:'2026-09-07T07:00:00Z',target_end:'2026-10-07T07:00:00Z',
 reference_price:2000,price_quantiles:[1800,2100,2500],hit_up:.8,hit_down:.6,terminal_down_flat_up:[.2,.3,.5],
 upper_barrier_price:2200,lower_barrier_price:1800,selected_model:'catboost'})),recent_issued:[],prospective:{},replay_generated_at:'v1'};
const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id] ||= element(),createElement:element,addEventListener:(_,fn)=>fn()},
 fetch:async url=>{if(fail)throw new Error('offline');return {ok:true,json:async()=>url==='signals.json'?structuredClone(fixture):{generated_at:'v1',horizons:{}}}},
 Date:Clock,Intl,setInterval:(fn,ms)=>timers.push({fn,ms})});
vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
(async()=>{
 await context.window.loadEventForecasts();assert.equal(nodes['event-status'].textContent,'시간별 연구 예측');
 fail=true;await context.window.loadEventForecasts();assert.match(nodes['event-status'].textContent,/연결 확인/);
 assert.match(nodes['event-updated'].textContent,/마지막 수신/);
 clock=Date.parse('2026-09-07T07:15:00Z');timers.find(t=>t.ms===30000).fn();
 assert.match(nodes['event-status'].textContent,/갱신 지연/);assert.match(nodes['event-status'].className,/warn/);
 assert.match(nodes['event-current'].children[0].children[0].textContent,/갱신 지연/);
 assert.equal(nodes['event-current'].children[0].children[1].textContent,'$2,100');
 fixture.expected_slot='2026-09-07T07:00:00Z';fixture.generated_at='2026-09-07T07:12:00Z';
 fixture.current.forEach(f=>{f.input_cutoff=fixture.expected_slot;});
 fail=false;await context.window.loadEventForecasts();assert.equal(nodes['event-status'].textContent,'시간별 연구 예측');
 assert.doesNotMatch(nodes['event-status'].className,/warn/);
})().catch(e=>{console.error(e);process.exit(1)});
'''
    subprocess.run(['node','-e',script],cwd=Path(__file__).resolve().parents[1],check=True)


def test_period_and_state_controls_update_matched_tables_and_target_dated_price_charts():
    script=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes={},drawn={};
const element=()=>({children:[],appendChild(x){this.children.push(x)},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}});
const summary=(n,brier=.1)=>({common_origins:n,nonoverlapping_selected:{rows:n},models:n?['selected','climatology'].map(model=>({model,event_brier:model==='selected'?brier:.25,
 up:{recall:.4,false_positive_rate:.1},down:{recall:.3,false_positive_rate:.1},terminal_balanced_accuracy:.4,mae_skill:.1,coverage80:.8})):[]});
const periods=[{id:'all',label:'전구간',start:'2025-01-01',end:'2027-01-01'},
 {id:'recent_365',label:'최근 365일',start:'2025-09-08',end:'2026-09-08'},
 {id:'year_2025',label:'2025년',start:'2025-01-01',end:'2026-01-01'},
 {id:'recent_30',label:'최근 30일',start:'2026-08-09',end:'2026-09-08'}];
const results={};for(const p of periods){results[p.id+'|all']=summary(p.id==='recent_30'?0:p.id==='year_2025'?1:3,p.id==='year_2025'?.3:.1);
 results[p.id+'|trend_up']=summary(p.id==='recent_30'?0:p.id==='year_2025'?1:2);}
const points=['2025-11-01','2026-06-01','2026-08-01'].map((date,i)=>({slot:date+'T00:00:00Z',target_end:new Date(Date.parse(date)+25*3600000).toISOString(),
 reference_price:100,return:Math.log(1.2),q10:Math.log(.9),q50:Math.log(1.1),q90:Math.log(1.3),hit_up:.8,hit_down:.2,up:1,down:0,
 trend_state:i===1?'down':'up',volatility_state:'normal'}));
const replay={generated_at:'r1',horizons:{'24':{...summary(3),points,segments:{as_of:'2026-09-07T12:00:00Z',periods,
 regimes:[{id:'all',label:'모든 상태'},{id:'trend_up',label:'상승'}],results}}}};
const payload={schema_version:1,status:'delayed',expected_slot:'2026-09-07T12:00:00Z',generated_at:'2026-09-07T12:08:00Z',current:[],recent_issued:[],replay_generated_at:'r1'};
const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id] ||= {...element(),id},createElement:element,addEventListener:(_,fn)=>fn()},
 fetch:async url=>({ok:true,json:async()=>url==='signals.json'?payload:replay}),Date,Intl,setInterval(){},
 Chart:class {constructor(node,config){this.config=config;drawn[node.id]=this}destroy(){this.destroyed=true}}});
vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
(async()=>{
 await context.window.loadEventForecasts();assert.equal(drawn['event-price-chart'].config.data.labels.length,3);
 nodes['event-period'].onchange({target:{value:'year_2025'}});
 assert.match(nodes['event-replay-status'].textContent,/공통 1개/);assert.match(nodes['event-replay-status'].textContent,/-20.0%/);
 assert.equal(drawn['event-price-chart'].config.data.labels[0],'2025-11-02');
 assert.ok(Math.abs(drawn['event-price-chart'].config.data.datasets[1].data[0]-110)<1e-10);
 const body=nodes['event-model-comparison'].children[0].children[1];assert.equal(body.children[0].children[1].textContent,'0.3000');
 nodes['event-period'].onchange({target:{value:'all'}});nodes['event-market-filter'].onchange({target:{value:'trend_up'}});
 assert.match(nodes['event-replay-status'].textContent,/공통 2개/);assert.equal(drawn['event-path-chart'].config.data.labels.length,2);
 nodes['event-price-unit'].onchange({target:{value:'return'}});
 assert.ok(Math.abs(drawn['event-price-chart'].config.data.datasets[1].data[0]-10)<1e-10);
 nodes['event-period'].onchange({target:{value:'recent_30'}});
 assert.match(nodes['event-replay-status'].textContent,/평가 표본이 없습니다/);assert.equal(drawn['event-price-chart'].destroyed,true);
 assert.equal(nodes['event-model-comparison'].children[0].children[1].children.length,0);
 nodes['event-period'].onchange({target:{value:'all'}});
 nodes['event-horizon'].onchange({target:{value:'720'}});
 assert.match(nodes['event-replay-status'].textContent,/아직 준비되지 않았습니다/);
 assert.equal(drawn['event-price-chart'].destroyed,true);
 assert.equal(nodes['event-period-comparison'].children.length,0);
 assert.equal(nodes['event-segment-download'].hidden,true);
})().catch(e=>{console.error(e);process.exit(1)});
'''
    subprocess.run(['node','-e',script],cwd=Path(__file__).resolve().parents[1],check=True)
