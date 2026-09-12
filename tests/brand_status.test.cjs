const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

function boot() {
  const elements = Object.fromEntries(['ef-brand','event-status','event-updated'].map(id => [id,{dataset:{},textContent:''}]));
  const listeners = {};
  let answer = {schema_version:1,mode:'auto'};
  const document = {hidden:false,getElementById:id=>elements[id],addEventListener:(event,callback)=>{listeners[event]=callback;}};
  const context = {window:{},document,fetch:async()=>{
    if(answer instanceof Error) throw answer;
    return {ok:true,json:async()=>answer};
  }};
  vm.runInNewContext(fs.readFileSync(path.join(__dirname,'../forecast_site/public/brand-status.js'),'utf8'),context);
  return {api:context.window.EtherForecastBrand, elements, document, listeners, respond:value=>{answer=value;}};
}

(async()=>{
  const t=boot(), good={available:true,delayed:false,fetchFailed:false,detail:'Fresh forecasts'};
  const expect=state=>{
    assert.equal(t.api.getStatus().state,state);
    assert.equal(t.elements['ef-brand'].dataset.state,state);
    assert.equal(t.elements['event-status'].textContent,t.api.getStatus().label);
  };
  t.api.updateForecast(good);expect('checking');
  await t.api.refreshConfig();expect('operating');
  t.api.updateForecast({...good,delayed:true});expect('delayed');
  t.api.updateForecast({...good,available:false});expect('unavailable');
  t.api.updateForecast({...good,fetchFailed:true});expect('connection_error');
  t.respond({schema_version:1,mode:'maintenance',message:'<img src=x onerror=alert(1)>'});
  await t.api.refreshConfig();expect('maintenance');
  assert.equal(t.elements['event-updated'].textContent,'<img src=x onerror=alert(1)>');
  t.respond(new Error('offline'));await t.api.refreshConfig();expect('maintenance');
  t.api.updateForecast(good);expect('maintenance');
  t.respond({schema_version:1,mode:'auto'});await t.api.refreshConfig();expect('operating');
  t.document.hidden=true;t.listeners.visibilitychange();expect('operating');
  assert.equal(t.elements['ef-brand'].dataset.motion,'paused');
  t.document.hidden=false;t.listeners.visibilitychange();
  assert.equal(t.elements['ef-brand'].dataset.motion,'running');
  for(const invalid of [null,{}, {schema_version:2,mode:'auto'}, {schema_version:1,mode:'operating'}, {schema_version:1,mode:'auto',message:{}},new Error('offline')]) {
    t.respond(invalid);await t.api.refreshConfig();expect('unknown');
  }
  const initial=boot();initial.respond(new Error('offline'));await initial.api.refreshConfig();
  assert.equal(initial.api.getStatus().state,'unknown');
  console.log('Brand status precedence, failure recovery, safe message rendering and visibility: passed');
})().catch(error=>{console.error(error);process.exitCode=1;});
