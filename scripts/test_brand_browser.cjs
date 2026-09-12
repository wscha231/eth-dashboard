const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
// Optional browser gate: npm install --no-save playwright; npx playwright install chromium
const {chromium}=require(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES ? path.join(process.env.CODEX_PRIMARY_RUNTIME_NODE_MODULES,'playwright') : 'playwright');
const root=path.resolve(__dirname,'../forecast_site/public');
const out=process.env.BRAND_QA_OUTPUT || path.join(require('node:os').tmpdir(),'etherforecast-brand-qa');fs.mkdirSync(out,{recursive:true});
const slot=Date.parse('2026-09-12T08:00:00Z'),now=slot+30*60000,iso=t=>new Date(t).toISOString();
const fixture=()=>({schema_version:1,status:'ready',generated_at:iso(slot+15*60000),expected_slot:iso(slot),current:[6,24,72,168,336,720].map(h=>({forecast_id:'test-'+h,slot:iso(slot),input_cutoff:iso(slot),available_at:iso(slot+5*60000),issued_at:iso(slot+10*60000),window_start:iso(slot+3600000),target_end:iso(slot+3600000+h*3600000),horizon_seconds:h*3600,terminal_down_flat_up:[.25,.5,.25],price_quantiles:[2300,2500,2700],reference_price:2500,lower_barrier_price:2400,upper_barrier_price:2600,hit_up:.4,hit_down:.35})),prospective:{},recent_issued:[]});
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CHROMIUM_EXECUTABLE || undefined,args:JSON.parse(process.env.CHROMIUM_ARGS || '[]'),headless:true});
 try {
 const page=await browser.newPage({viewport:{width:1365,height:1000}}),errors=[];page.on('pageerror',e=>errors.push(e.message));
 let config={schema_version:1,mode:'auto'},signals=fixture(),failSignals=false,failConfig=false,holdReplay=false;
 await page.clock.install({time:now});
 await page.route('**/*',async route=>{
  const u=new URL(route.request().url());if(u.origin!=='https://etherforecast.test')return route.abort();
  const name=u.pathname==='/'?'index.html':u.pathname.slice(1);
  if(name==='site_status.json')return route.fulfill({status:failConfig?503:200,contentType:'application/json',body:JSON.stringify(config)});
  if(name==='signals.json')return route.fulfill({status:failSignals?503:200,contentType:'application/json',body:JSON.stringify(signals)});
  if(name==='signals_replay.json'){if(holdReplay)return;return route.fulfill({contentType:'application/json',body:'{"horizons":{}}'});}
  const file=path.join(root,name);
  if(!file.startsWith(root)||!fs.existsSync(file))return route.fulfill({status:404,body:'unavailable'});
  return route.fulfill({path:file,contentType:name.endsWith('.svg')?'image/svg+xml':undefined});
 });
 const expect=async state=>{await page.waitForFunction(s=>document.querySelector('#ef-brand').dataset.state===s,state);assert.equal(await page.locator('#event-status').textContent(),await page.evaluate(()=>EtherForecastBrand.getStatus().label));};
 await page.goto('https://etherforecast.test/');await expect('operating');await page.waitForSelector('#ef-brand svg');
 const visual=()=>page.evaluate(()=>({filter:getComputedStyle(document.querySelector('#ef-brand .ef-base')).filter,visible:getComputedStyle(document.querySelector('#ef-brand .ef-effect')).visibility,play:getComputedStyle(document.querySelector('#ef-brand .ef-current')).animationPlayState,offset:getComputedStyle(document.querySelector('#ef-brand .ef-current')).strokeDashoffset}));
 await page.waitForTimeout(500);const v1=await visual();await page.waitForTimeout(400);const v2=await visual();assert.notEqual(v1.offset,v2.offset);assert.equal(v2.play,'running');
 await page.screenshot({path:out+'/operating.png'});
 config={schema_version:1,mode:'maintenance',message:'Scheduled model maintenance'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('maintenance');await page.waitForTimeout(500);
 const off=await visual();assert.equal(off.visible,'hidden');assert.equal(off.play,'paused');assert.match(off.filter,/brightness\(0.26\)/);
 await page.screenshot({path:out+'/maintenance.png'});
 failConfig=true;await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('maintenance');
 failConfig=false;config={schema_version:1,mode:'auto'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('operating');
 failSignals=true;await page.evaluate(()=>loadEventForecasts());await expect('connection_error');failSignals=false;await page.evaluate(()=>loadEventForecasts());await expect('operating');
 signals.current.pop();await page.evaluate(()=>loadEventForecasts());await expect('delayed');signals=fixture();await page.evaluate(()=>loadEventForecasts());await expect('operating');
 await page.clock.runFor(61*60000);await expect('delayed');await page.clock.setSystemTime(now);await page.evaluate(()=>loadEventForecasts());await expect('operating');
 await page.emulateMedia({reducedMotion:'reduce'});assert.equal((await visual()).play,'paused');await expect('operating');assert.equal((await visual()).visible,'visible');await page.emulateMedia({reducedMotion:'no-preference'});
 await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});assert.equal((await visual()).play,'paused');await expect('operating');
 await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'));});
 failConfig=true;await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('unknown');failConfig=false;await page.evaluate(()=>EtherForecastBrand.refreshConfig());
 config={schema_version:1,mode:'maintenance',message:'<img src=x onerror=alert(1)>'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());assert.equal(await page.locator('#event-updated img').count(),0);assert.equal(await page.locator('#event-updated').textContent(),config.message);
 config={schema_version:1,mode:'auto'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('operating');
 await page.locator('#support-research').screenshot({path:out+'/donation.png'});
 for(const width of [320,390]){await page.setViewportSize({width,height:844});await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:out+'/mobile-'+width+'.png'});const overflow=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,items:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&!e.closest('.table-scroll')).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right})).slice(0,15)}));console.log(overflow);assert.equal(overflow.scroll<=width,true,'overflow at '+width);}
 failConfig=true;failSignals=true;await page.reload();await expect('unknown');await page.waitForSelector('#ef-brand svg');assert.equal((await visual()).visible,'hidden');
 failConfig=false;failSignals=false;holdReplay=true;await page.reload();await expect('operating');config={schema_version:1,mode:'maintenance',message:'Maintenance during replay download'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('maintenance');
 assert.deepEqual(errors,[]);console.log('PASS: actual motion, dimmed maintenance, recovery, stale/partial/error/initial-off, reduced motion, hidden tab, safe text, mobile 320/390, no page errors');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
