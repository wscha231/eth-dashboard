from pathlib import Path
import shutil
import subprocess

import pytest


def test_event_cards_keep_reference_prices_and_distinguish_two_event_probabilities():
    if not shutil.which('node'):pytest.skip('Node required for frontend checks')
    script=r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
const nodes={};const element=()=>({children:[],className:'',appendChild(x){this.children.push(x)},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}});
const clock=Date.parse('2026-09-06T12:10:00Z');
class Clock extends Date {constructor(...args){super(...(args.length?args:[clock]))}static now(){return clock}}
const fixture={schema_version:1,generated_at:'2026-09-06T12:08:00Z',expected_slot:'2026-09-06T12:00:00Z',status:'ready',current:[{forecast_id:'fixed-24',slot:'2026-09-06T12:00:00Z',available_at:'2026-09-06T12:02:00Z',horizon_seconds:86400,input_cutoff:'2026-09-06T12:00:00Z',issued_at:'2026-09-06T12:08:00Z',window_start:'2026-09-06T13:00:00Z',target_end:'2026-09-07T13:00:00Z',reference_price:2000,price_quantiles:[1800,2100,2500],hit_up:.8,hit_down:.6,upper_barrier_price:2200,lower_barrier_price:1800,terminal_down_flat_up:[.2,.3,.5],selected_model:'catboost'}],recent_issued:[],prospective:{},replay_generated_at:'v1'};
const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id] ||= element(),createElement:element,addEventListener(){}},fetch:async url=>({ok:true,json:async()=>url==='signals.json'?fixture:{generated_at:'v1',horizons:{}}}),Date:Clock,Intl,setInterval(){}});
vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
context.window.loadEventForecasts().then(()=>{
 assert.equal(nodes['ref-price'].textContent,'$2,000');
 const flatten=node=>[node.textContent||'',...node.children.map(flatten)].join(' ');
 const card=flatten(nodes['event-current']);assert.match(card,/\$2,100/);
 assert.match(card,/Touch \$2,200 or higher: 80.0%/);assert.match(card,/Touch \$1,800 or lower: 60.0%/);
 assert.match(card,/more than 100%/);assert.match(card,/not a record of accuracy/);
 assert.doesNotMatch(card,/[가-힣]/);
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
 status:'ready',current:[6,24,72,168,336,720].map(h=>({forecast_id:`fixed-${h}`,slot:'2026-09-07T06:00:00Z',available_at:'2026-09-07T06:02:00Z',horizon_seconds:h*3600,input_cutoff:'2026-09-07T06:00:00Z',
 issued_at:'2026-09-07T06:08:00Z',window_start:'2026-09-07T07:00:00Z',target_end:new Date(Date.parse('2026-09-07T07:00:00Z')+h*3600000).toISOString(),
 reference_price:2000,price_quantiles:[1800,2100,2500],hit_up:.8,hit_down:.6,terminal_down_flat_up:[.2,.3,.5],
 upper_barrier_price:2200,lower_barrier_price:1800,selected_model:'catboost'})),recent_issued:[],prospective:{},replay_generated_at:'v1'};
const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id] ||= element(),createElement:element,addEventListener:(_,fn)=>fn()},
 fetch:async url=>{if(fail)throw new Error('offline');return {ok:true,json:async()=>url==='signals.json'?structuredClone(fixture):{generated_at:'v1',horizons:{}}}},
 Date:Clock,Intl,setInterval:(fn,ms)=>timers.push({fn,ms})});
vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
(async()=>{
 await context.window.loadEventForecasts();assert.equal(nodes['event-status'].textContent,'Up to date');
 fail=true;await context.window.loadEventForecasts();assert.match(nodes['event-status'].textContent,/Connection interrupted/);
 assert.match(nodes['event-updated'].textContent,/last received/);
 clock=Date.parse('2026-09-07T07:15:00Z');timers.find(t=>t.ms===30000).fn();
 assert.match(nodes['event-status'].textContent,/Update delayed/);assert.match(nodes['event-status'].className,/warn/);
 assert.match(nodes['event-current'].children[0].children[0].textContent,/Update delayed/);
 assert.equal(nodes['event-current'].children[0].children[1].children[0].textContent,'$2,100');
 fixture.expected_slot='2026-09-07T07:00:00Z';fixture.generated_at='2026-09-07T07:12:00Z';
 fixture.current.forEach(f=>{f.slot=f.input_cutoff=fixture.expected_slot;f.available_at=f.issued_at=fixture.generated_at;
  f.window_start='2026-09-07T08:00:00Z';f.target_end=new Date(Date.parse(f.window_start)+f.horizon_seconds*1000).toISOString();});
 fail=false;await context.window.loadEventForecasts();assert.equal(nodes['event-status'].textContent,'Up to date');
 assert.doesNotMatch(nodes['event-status'].className,/warn/);
 fixture.current.pop();await context.window.loadEventForecasts();assert.equal(nodes['event-status'].textContent,'Update delayed');
})().catch(e=>{console.error(e);process.exit(1)});
'''
    subprocess.run(['node','-e',script],cwd=Path(__file__).resolve().parents[1],check=True)


def test_previous_verified_forecasts_survive_empty_release_reload_and_expire_without_rebasing():
    script = r'''
const fs=require('node:fs'),vm=require('node:vm'),assert=require('node:assert/strict');
let clock=Date.parse('2026-09-10T14:56:30Z'),offline=false;
class Clock extends Date {constructor(...args){super(...(args.length?args:[clock]))}static now(){return clock}}
const horizons=[6,24,72,168,336,720],slot=Date.parse('2026-09-10T13:00:00Z');
const iso=ms=>new Date(ms).toISOString();
const old=h=>({forecast_id:`original-${h}`,horizon_seconds:h*3600,slot:iso(slot),input_cutoff:iso(slot),
 available_at:iso(slot+2*60000),issued_at:iso(slot+32*60000),published_at:iso(slot+33*60000),
 window_start:iso(slot+3600000),target_end:iso(slot+(h+1)*3600000),reference_price:2000,
 price_quantiles:[1800,2100,2400],terminal_down_flat_up:[.2,.5,.3],hit_up:.6,hit_down:.4,
 lower_barrier_price:1800,upper_barrier_price:2200,selected_model:'catboost'});
const original=horizons.map(old),bytes=JSON.stringify(original);
const payload={schema_version:1,status:'delayed',generated_at:'2026-09-10T14:55:33Z',expected_slot:'2026-09-10T14:00:00Z',
 current:[],recent_issued:structuredClone(original),prospective:{},replay_generated_at:'replay'};
const flatten=n=>[n.textContent||'',...(n.children||[]).map(flatten)].join(' ');
function mount(){
 const nodes={},timers=[];
 const element=()=>({children:[],appendChild(x){this.children.push(x)},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x}});
 const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id]??=element(),createElement:element,addEventListener:(_,fn)=>fn()},
  fetch:async url=>{if(offline)throw Error('offline');return {ok:true,json:async()=>url==='signals.json'?structuredClone(payload):{generated_at:'replay',horizons:{}}}},Date:Clock,Intl,setInterval:(fn,ms)=>timers.push({fn,ms})});
 vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
 return {nodes,timers,load:()=>context.window.loadEventForecasts(),select:h=>nodes['event-horizon'].onchange({target:{value:String(h)}}),card:()=>flatten(nodes['event-current'])};
}
(async()=>{
 let page=mount();await page.load();
 for(const h of horizons){page.select(h);assert.match(page.card(),/Previous published forecast.*Update delayed/);
  assert.match(page.card(),/\$2,100/);assert.match(page.card(),/Published: 10 Sept 2026, 13:33 UTC/);
  assert.equal(page.nodes['generated-at'].textContent,'Reference at 10 Sept 2026, 13:00 UTC');}
 assert.equal(page.nodes['event-status'].textContent,'Update delayed');assert.equal(JSON.stringify(payload.recent_issued),bytes);
 page=mount();await page.load();page.select(720);assert.match(page.card(),/Previous published forecast/);
 offline=true;await page.load();assert.match(page.card(),/\$2,100/);assert.match(page.nodes['event-updated'].textContent,/last received/);
 page.select(6);assert.match(page.card(),/Previous published forecast/);
 clock=Date.parse('2026-09-10T20:00:00Z');page.timers.find(t=>t.ms===30000).fn();
 assert.match(page.card(),/outlook unavailable/);page.select(24);assert.match(page.card(),/Previous published forecast/);
 offline=false;clock=Date.parse('2026-09-10T15:10:00Z');payload.generated_at=iso(clock);payload.expected_slot='2026-09-10T15:00:00Z';
 payload.current=horizons.map(h=>({...old(h),forecast_id:`new-${h}`,slot:payload.expected_slot,input_cutoff:payload.expected_slot,
  available_at:'2026-09-10T15:02:00Z',issued_at:'2026-09-10T15:08:00Z',window_start:'2026-09-10T16:00:00Z',
  target_end:iso(Date.parse('2026-09-10T16:00:00Z')+h*3600000),price_quantiles:[1900,2200,2500]}));
 payload.current.pop();await page.load();page.select(720);assert.match(page.card(),/Previous published forecast/);
 page.select(24);assert.match(page.card(),/\$2,200/);assert.doesNotMatch(page.card(),/Previous published forecast/);
 payload.current.push({...payload.current[0],forecast_id:'new-720',horizon_seconds:720*3600,target_end:iso(Date.parse('2026-09-10T16:00:00Z')+720*3600000)});
 payload.status='ready';await page.load();assert.equal(page.nodes['event-status'].textContent,'Up to date');
 page.select(720);assert.match(page.card(),/Published forecast/);assert.doesNotMatch(page.card(),/Previous published forecast/);
 assert.equal(JSON.stringify(payload.recent_issued),bytes);
 payload.status='delayed';payload.current=[];payload.recent_issued=[old(24)];page.select(24);
 const invalid=[{published_at:null},{published_at:'2026-09-10T14:00:00Z'},{published_at:'2026-09-10T13:31:00Z'},
  {published_at:'2026-09-11T13:33:00Z'},{published_at:'invalid'},{published_at:'2026-09-10T13:33:00'},
  {forecast_id:''},{horizon_seconds:720*3600},{target_end:'invalid'},{target_end:'2026-09-10T14:00:00Z'},
  {available_at:'2026-09-10T13:34:00Z'},{input_cutoff:'2026-09-10T14:00:00Z'},{issued_at:'2026-09-10T13:55:00Z'},
  {price_quantiles:[1800,NaN,2400]},{price_quantiles:[2400,2100,1800]},{reference_price:null},
  {terminal_down_flat_up:[.5,.5,.5]},{hit_up:1.1}];
 for(const change of invalid){payload.recent_issued=[{...old(24),...change}];await page.load();assert.match(page.card(),/outlook unavailable/,JSON.stringify(change));}
 // Do not rely on array order, and do not select an invalid newer publication.
 const earlier={...old(24),forecast_id:'earlier',slot:iso(slot-3600000),input_cutoff:iso(slot-3600000),
  available_at:iso(slot-58*60000),issued_at:iso(slot-28*60000),published_at:iso(slot-27*60000),
  window_start:iso(slot),target_end:iso(slot+24*3600000),price_quantiles:[1700,1990,2300]};
 payload.recent_issued=[old(24),{...old(24),published_at:'invalid',price_quantiles:[1800,2300,2500]},earlier];
 await page.load();assert.match(page.card(),/\$2,100/);assert.doesNotMatch(page.card(),/\$1,990|\$2,300/);
})().catch(e=>{console.error(e);process.exit(1)});
'''
    subprocess.run(['node', '-e', script], cwd=Path(__file__).resolve().parents[1], check=True)


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
const payload={schema_version:1,status:'delayed',expected_slot:'2026-09-07T12:00:00Z',generated_at:'2026-09-07T12:08:00Z',current:[],recent_issued:[],replay_generated_at:'r1',
 prospective:{'24':{issued:25,resolved:10,pending:15,nonoverlap_resolved:2,paired_selected:{event_brier:.15},baseline:{event_brier:.25}},'720':{issued:4,resolved:0,pending:4}}};
const context=vm.createContext({window:{},document:{getElementById:id=>nodes[id] ||= {...element(),id},createElement:element,addEventListener:(_,fn)=>fn()},
 fetch:async url=>({ok:true,json:async()=>url==='signals.json'?payload:replay}),Date,Intl,setInterval(){},
 Chart:class {constructor(node,config){this.config=config;drawn[node.id]=this}destroy(){this.destroyed=true}}});
vm.runInContext(fs.readFileSync('forecast_site/public/events.js','utf8'),context);
(async()=>{
 await context.window.loadEventForecasts();assert.equal(drawn['event-price-chart'].config.data.labels.length,3);
 const flatten=node=>[node.textContent||'',...(node.children||[]).map(flatten)].join(' ');
 assert.match(flatten(nodes['event-scorecards']),/60.0% less error/);
 assert.match(flatten(nodes['event-scorecards']),/40 out of 100 actual large rises/);
 assert.match(flatten(nodes['event-scorecards']),/10.0% of periods without that move/);
 assert.match(flatten(nodes['event-live-scorecards']),/40.0% less error/);
 assert.doesNotMatch(flatten(nodes['event-period']),/[가-힣]/);
 assert.doesNotMatch(flatten(nodes['event-market-filter']),/[가-힣]/);
 nodes['event-period'].onchange({target:{value:'year_2025'}});
 assert.match(nodes['event-replay-status'].textContent,/1 matched observations/);assert.match(nodes['event-replay-status'].textContent,/-20.0%/);
 assert.match(flatten(nodes['event-scorecards']),/20.0% more error/);
 assert.match(nodes['event-verdict'].textContent,/has not beaten/);
 assert.match(nodes['event-prospective-status'].textContent,/25 published, 10 settled, 15 awaiting/);
 assert.equal(drawn['event-price-chart'].config.data.labels[0],'2025-11-02');
 assert.ok(Math.abs(drawn['event-price-chart'].config.data.datasets[1].data[0]-110)<1e-10);
 const body=nodes['event-model-comparison'].children[0].children[1];assert.equal(body.children[0].children[1].textContent,'0.3000');
 nodes['event-period'].onchange({target:{value:'all'}});nodes['event-market-filter'].onchange({target:{value:'trend_up'}});
 assert.match(nodes['event-replay-status'].textContent,/2 matched observations/);assert.equal(drawn['event-path-chart'].config.data.labels.length,2);
 nodes['event-price-unit'].onchange({target:{value:'return'}});
 assert.ok(Math.abs(drawn['event-price-chart'].config.data.datasets[1].data[0]-10)<1e-10);
 nodes['event-period'].onchange({target:{value:'recent_30'}});
 assert.match(nodes['event-replay-status'].textContent,/No settled matching observations/);assert.equal(drawn['event-price-chart'].destroyed,true);
 assert.match(nodes['event-verdict'].textContent,/No settled test results/);
 assert.doesNotMatch(flatten(nodes['event-scorecards']),/80.0%/);
 assert.equal(nodes['event-model-comparison'].children[0].children[1].children.length,0);
 nodes['event-period'].onchange({target:{value:'all'}});
 nodes['event-horizon'].onchange({target:{value:'720'}});
 assert.match(nodes['event-replay-status'].textContent,/not ready/);
 assert.match(nodes['event-prospective-status'].textContent,/4 published, 0 settled, 4 awaiting/);
 assert.match(flatten(nodes['event-live-scorecards']),/Not available/);
 assert.equal(drawn['event-price-chart'].destroyed,true);
 assert.equal(nodes['event-period-comparison'].children.length,0);
 assert.equal(nodes['event-segment-download'].hidden,true);
 context.Chart=undefined;nodes['event-horizon'].onchange({target:{value:'24'}});
 assert.match(nodes['event-chart-status'].textContent,/chart library could not load/);
 assert.match(flatten(nodes['event-scorecards']),/60.0% less error/);
})().catch(e=>{console.error(e);process.exit(1)});
'''
    subprocess.run(['node','-e',script],cwd=Path(__file__).resolve().parents[1],check=True)
