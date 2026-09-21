(() => {
  const HORIZONS=[6,24,72,168,336,720];
  const NAMES={6:'6 hours',24:'1 day',72:'3 days',168:'7 days',336:'14 days',720:'30 days'};
  const $=id=>document.getElementById(id);
  const pct=(v,d=1)=>Number.isFinite(Number(v))?`${(100*Number(v)).toFixed(d)}%`:'—';
  const usd=v=>Number.isFinite(Number(v))?new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(Number(v)):'—';
  const date=v=>Number.isFinite(Date.parse(v))?new Date(v).toLocaleString('en-GB',{timeZone:'UTC',day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})+' UTC':'—';
  const finite=v=>Number.isFinite(Number(v));
  let variance=null, signals=null, availability=null;

  async function json(path){const r=await fetch(path,{cache:'no-store'});if(!r.ok)throw new Error(`${path}:${r.status}`);return r.json();}
  async function loadOptional(path){try{return await json(path)}catch(_){return null}}

  function renderValidation(){
    const body=$('validation-table');body.replaceChildren();
    for(const h of HORIZONS){
      const row=variance?.horizons?.[String(h)]||{};
      const p=row.prospective||{};
      const n=p.nonoverlap?.rows||0,min=p.minimum_nonoverlap_for_review||0;
      const tr=document.createElement('tr');
      const progress=min?Math.min(100,100*n/min):0;
      const watch=p.performance_watch==='manual_review_required'?'Review threshold reached':'Building evidence';
      tr.innerHTML=`<td>${NAMES[h]}</td><td><b>${pct(row.historical_qlike_improvement,2)}</b></td><td><span class="pill good">PASS</span></td><td>${n}</td><td>${min}</td><td>${watch}<div class="progress"><i style="width:${progress.toFixed(1)}%"></i></div></td>`;
      body.appendChild(tr);
    }
  }

  function renderCurrent(){
    const h=Number($('horizon').value),row=variance?.horizons?.[String(h)]||{};
    const has=finite(row.predicted_endpoint_sigma)&&finite(row.reference_price);
    $('current-empty').hidden=has;$('current-outlook').hidden=!has;
    if(!has)return;
    $('reference-price').textContent=usd(row.reference_price);
    $('sigma').textContent=pct(row.predicted_endpoint_sigma,1);
    $('sigma-explain').textContent=`HAR-RV 1σ movement scale over ${NAMES[h]}. This is movement magnitude, not direction.`;
    $('sigma-usd').textContent=usd(row.one_sigma_move_usd);
    $('scale-ratio').textContent=finite(row.scale_vs_persistence)?`${Number(row.scale_vs_persistence).toFixed(2)}×`:'—';
    $('target-end').textContent=date(row.target_end);
  }

  function valid6h(f){
    if(!f||Number(f.horizon_seconds)!==21600)return false;
    const p=f.terminal_down_flat_up;
    return Array.isArray(p)&&p.length===3&&p.every(x=>finite(x)&&x>=0&&x<=1)&&Math.abs(p.reduce((a,b)=>a+Number(b),0)-1)<1e-6&&finite(f.hit_up)&&finite(f.hit_down);
  }
  function renderEvent(){
    const current=(signals?.current||[]).find(valid6h);
    $('event-unavailable').hidden=!!current;$('event-content').hidden=!current;
    if(!current)return;
    $('p-down').textContent=pct(current.terminal_down_flat_up[0]);
    $('p-flat').textContent=pct(current.terminal_down_flat_up[1]);
    $('p-up').textContent=pct(current.terminal_down_flat_up[2]);
    $('event-path').textContent=`Path-touch probabilities: upper ${pct(current.hit_up)} · lower ${pct(current.hit_down)}. Historical 6h Event head passed the frozen Brier/ECE gate, but prospective promotion is still disabled.`;
  }

  function renderStatus(){
    const status=$('site-status');
    if(!variance){status.textContent='Variance shadow unavailable';status.className='pill bad';$('generated').textContent='The latest public variance record could not be loaded.';return;}
    const complete=HORIZONS.every(h=>variance.horizons?.[String(h)]?.historical_gate==='pass');
    status.textContent=complete?'6-horizon historical gate passed':'Research state incomplete';
    status.className=`pill ${complete?'good':'warn'}`;
    $('generated').textContent=`Shadow updated ${date(variance.generated_at)} · source ${date(variance.source_as_of)}. No automatic production promotion.`;
  }

  function releaseHealth(now=Date.now()){
    const current=Array.isArray(signals?.current)?signals.current:[];
    const slot=Date.parse(signals?.expected_slot),generated=Date.parse(signals?.generated_at);
    const due=Math.floor((now-15*60000)/3600000)*3600000;
    const counts=new Map();
    for(const row of current){
      const h=Number(row?.horizon_seconds)/3600;
      if(HORIZONS.includes(h))counts.set(h,(counts.get(h)||0)+1);
    }
    const complete=signals?.status==='ready' && typeof signals?.release_id==='string' && !!signals.release_id &&
      Number.isFinite(slot) && slot%3600000===0 && Number.isFinite(generated) && generated<=now+120000 &&
      current.length===6 && HORIZONS.every(h=>counts.get(h)===1);
    const currentEnough=complete && slot>=due && now-generated<=100*60000;
    return {complete,current:currentEnough,slot,generated,release:signals?.release_id};
  }

  function renderService(){
    const release=releaseHealth();
    const releaseState=$('release-state'),releaseDetail=$('release-detail');
    if(release.current){
      releaseState.textContent='Current · 6 / 6';
      releaseState.style.color='var(--good)';
      releaseDetail.textContent=`Release ${String(release.release).slice(0,12)}… · slot ${date(signals.expected_slot)} · generated ${date(signals.generated_at)}.`;
    } else if(release.complete){
      releaseState.textContent='Complete but delayed';
      releaseState.style.color='var(--warn)';
      releaseDetail.textContent=`The last complete six-horizon release is ${date(signals.generated_at)}. Check the published snapshot before treating it as current.`;
    } else {
      releaseState.textContent='Current release unconfirmed';
      releaseState.style.color='var(--bad)';
      releaseDetail.textContent='A complete six-horizon public release could not be verified from the current feed.';
    }

    const inputState=$('input-state'),inputDetail=$('input-detail');
    const decision=availability?.decision;
    const generated=Date.parse(availability?.generated_at);
    const recent=Number.isFinite(generated) && Date.now()-generated<=3*3600000;
    if(!decision || !recent){
      inputState.textContent='Research status unconfirmed';
      inputState.style.color='var(--warn)';
      inputDetail.textContent='The separate received-input fallback status is unavailable or older than three hours.';
    } else if(decision.status==='not_needed'){
      inputState.textContent='Strict inputs complete';
      inputState.style.color='var(--good)';
      inputDetail.textContent=`ETH/BTC strict input continuity was satisfied at ${date(decision.origin)}; the research fallback was correctly suppressed.`;
    } else if(decision.status==='eligible'){
      inputState.textContent='Fallback shadow eligible';
      inputState.style.color='var(--warn)';
      inputDetail.textContent=`A Center-only no-change fallback shadow was eligible at ${date(decision.origin)}. It is research-only and is not a public forecast or proof of accuracy.`;
    } else if(decision.status==='blocked'){
      inputState.textContent='Fallback blocked';
      inputState.style.color='var(--bad)';
      inputDetail.textContent=`The fallback was blocked: ${decision.reason||'input/deadline contract not satisfied'}. No substitute forecast is invented.`;
    } else {
      inputState.textContent='Research status unknown';
      inputState.style.color='var(--warn)';
      inputDetail.textContent='The availability shadow returned an unrecognized state; no public fallback is assumed.';
    }

    const note=$('service-note');
    if(note){
      const publicText=release.current?'The current public feed is complete.':'The current public feed needs attention.';
      const inputText=decision?.status==='eligible'?' The fallback remains shadow-only; the public release is not replaced by it.':'';
      note.textContent=publicText+inputText+' Historical service gaps remain in the audit trail and are never backfilled as if they were real-time publications.';
    }
  }

  async function load(){
    [variance,signals,availability]=await Promise.all([
      loadOptional('variance_shadow.json'),
      loadOptional('signals.json'),
      loadOptional('availability_shadow.json')
    ]);
    renderStatus();renderService();renderValidation();renderCurrent();renderEvent();
  }
  document.addEventListener('DOMContentLoaded',()=>{$('horizon').addEventListener('change',renderCurrent);load();setInterval(load,60000);});
})();
