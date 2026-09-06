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
