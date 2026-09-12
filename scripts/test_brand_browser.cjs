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
 let config={schema_version:1,mode:'auto'},signals=fixture(),failSignals=false,failConfig=false,holdReplay=false,failVideo=false;
 await page.clock.install({time:now});
 await page.route('**/*',async route=>{
  const u=new URL(route.request().url());if(u.origin!=='https://etherforecast.test')return route.abort();
  const name=u.pathname==='/'?'index.html':u.pathname.slice(1);
  if(name==='site_status.json')return route.fulfill({status:failConfig?503:200,contentType:'application/json',body:JSON.stringify(config)});
  if(name==='signals.json')return route.fulfill({status:failSignals?503:200,contentType:'application/json',body:JSON.stringify(signals)});
  if(name==='signals_replay.json'){if(holdReplay)return;return route.fulfill({contentType:'application/json',body:'{"horizons":{}}'});}
  if(name==='assets/etherforecast-loop.mp4' && failVideo)return route.fulfill({status:404,body:'unavailable'});
  const file=path.join(root,name);
  if(!file.startsWith(root)||!fs.existsSync(file))return route.fulfill({status:404,body:'unavailable'});
  return route.fulfill({path:file,contentType:name.endsWith('.svg')?'image/svg+xml':name.endsWith('.mp4')?'video/mp4':undefined});
 });
 const expect=async state=>{await page.waitForFunction(s=>document.querySelector('#ef-brand').dataset.state===s,state);assert.equal(await page.locator('#event-status').textContent(),await page.evaluate(()=>EtherForecastBrand.getStatus().label));};
 await page.goto('https://etherforecast.test/');await expect('operating');await page.waitForSelector('#ef-video');
 const visual=()=>page.evaluate(()=>{const v=document.getElementById('ef-video');return {filter:getComputedStyle(document.querySelector('.ef-video-poster')).filter,visible:getComputedStyle(v).opacity,paused:v.paused,time:v.currentTime,ready:v.readyState,error:v.error?.message,width:v.getBoundingClientRect().width};});
 await page.waitForFunction(()=>{const v=document.getElementById('ef-video');return !v.paused && v.currentTime>0.1;});
 const v1=await visual();await page.waitForTimeout(400);const v2=await visual();assert.ok(v2.time>v1.time);assert.equal(v2.visible,'1');assert.ok(v2.width>=220);
 await page.evaluate(()=>{document.getElementById('ef-video').currentTime=4.8;});await page.waitForTimeout(650);assert.ok((await visual()).time<2,'video loops');
 await page.locator('#ef-video-toggle').click();assert.equal((await visual()).paused,true);await page.locator('#ef-video-toggle').click();await page.waitForFunction(()=>!document.getElementById('ef-video').paused);
 await page.screenshot({path:out+'/operating.png'});
 await page.locator('.ef-service').screenshot({path:out+'/top-animation-preview.png'});
 config={schema_version:1,mode:'maintenance',message:'Scheduled model maintenance'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('maintenance');await page.waitForTimeout(500);
 const off=await visual();assert.equal(off.visible,'0');assert.equal(off.paused,true);await page.waitForTimeout(300);assert.equal((await visual()).time,off.time);assert.match(off.filter,/brightness\(0.26\)/);
 await page.screenshot({path:out+'/maintenance.png'});
 failConfig=true;await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('maintenance');
 failConfig=false;config={schema_version:1,mode:'auto'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('operating');
 failSignals=true;await page.evaluate(()=>loadEventForecasts());await expect('connection_error');failSignals=false;await page.evaluate(()=>loadEventForecasts());await expect('operating');
 signals.current.pop();await page.evaluate(()=>loadEventForecasts());await expect('delayed');signals=fixture();await page.evaluate(()=>loadEventForecasts());await expect('operating');
 await page.clock.runFor(61*60000);await expect('delayed');await page.clock.setSystemTime(now);await page.evaluate(()=>loadEventForecasts());await expect('operating');
 await page.emulateMedia({reducedMotion:'reduce'});await page.waitForFunction(()=>document.getElementById('ef-video').paused);assert.equal((await visual()).paused,true);await expect('operating');assert.equal((await visual()).visible,'0');await page.emulateMedia({reducedMotion:'no-preference'});
 await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:true});document.dispatchEvent(new Event('visibilitychange'));});assert.equal((await visual()).paused,true);await expect('operating');
 await page.evaluate(()=>{Object.defineProperty(document,'hidden',{configurable:true,value:false});document.dispatchEvent(new Event('visibilitychange'));});
 failConfig=true;await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('unknown');failConfig=false;await page.evaluate(()=>EtherForecastBrand.refreshConfig());
 config={schema_version:1,mode:'maintenance',message:'<img src=x onerror=alert(1)>'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());assert.equal(await page.locator('#event-updated img').count(),0);assert.equal(await page.locator('#event-updated').textContent(),config.message);
 config={schema_version:1,mode:'auto'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('operating');
 await page.locator('#support-research').screenshot({path:out+'/donation.png'});
 for(const width of [320,390]){await page.setViewportSize({width,height:844});await page.evaluate(()=>scrollTo(0,0));await page.screenshot({path:out+'/mobile-'+width+'.png'});const overflow=await page.evaluate(()=>({width:innerWidth,scroll:document.documentElement.scrollWidth,items:[...document.querySelectorAll('body *')].filter(e=>e.getBoundingClientRect().right>innerWidth+1&&!e.closest('.table-scroll')).map(e=>({tag:e.tagName,id:e.id,cls:e.className,right:e.getBoundingClientRect().right})).slice(0,15)}));console.log(overflow);assert.equal(overflow.scroll<=width,true,'overflow at '+width);}
 failConfig=true;failSignals=true;await page.reload();await expect('unknown');await page.waitForSelector('#ef-video');assert.equal((await visual()).visible,'0');
 failConfig=false;failSignals=false;holdReplay=true;await page.reload();await expect('operating');config={schema_version:1,mode:'maintenance',message:'Maintenance during replay download'};await page.evaluate(()=>EtherForecastBrand.refreshConfig());await expect('maintenance');
 // Block automatic playback, then recover with a real user gesture.
 await page.addInitScript(()=>{window.blockVideo=true;const play=HTMLMediaElement.prototype.play;HTMLMediaElement.prototype.play=function(){return window.blockVideo?Promise.reject(new DOMException('Gesture required','NotAllowedError')):play.call(this);};});
 config={schema_version:1,mode:'auto'};await page.reload();await expect('operating');await page.waitForFunction(()=>document.getElementById('ef-video-toggle').textContent==='Play animation');assert.equal((await visual()).paused,true);
 await page.evaluate(()=>{window.blockVideo=false;});await page.locator('#ef-video-toggle').click();await page.waitForFunction(()=>!document.getElementById('ef-video').paused && document.getElementById('ef-video').currentTime>0.1);
 failVideo=true;await page.reload();await expect('operating');await page.waitForFunction(()=>document.getElementById('ef-video').error!==null);await page.waitForFunction(()=>!document.getElementById('ef-video-error').hidden);assert.equal((await visual()).visible,'0');
 assert.deepEqual(errors,[]);console.log('PASS: native video playback and loop, manual pause/resume, dimmed maintenance, recovery, stale/partial/error/initial-off, reduced motion, hidden tab, safe text, mobile 320/390, no page errors');
 } finally {await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
