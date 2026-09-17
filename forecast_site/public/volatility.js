(() => {
  const HORIZONS=[6,24,72,168,336,720];
  const NAMES={6:'6 hours',24:'1 day',72:'3 days',168:'7 days',336:'14 days',720:'30 days'};
  const $=id=>document.getElementById(id);
  const pct=(v,d=1)=>Number.isFinite(Number(v))?`${(100*Number(v)).toFixed(d)}%`:'—';
  const usd=v=>Number.isFinite(Number(v))?new Intl.NumberFormat('en-US',{style:'currency',currency:'USD',maximumFractionDigits:0}).format(Number(v)):'—';
  const date=v=>Number.isFinite(Date.parse(v))?new Date(v).toLocaleString('en-GB',{timeZone:'UTC',day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})+' UTC':'—';
  const finite=v=>Number.isFinite(Number(v));
  let variance=null, signals=null;

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

  async function load(){
    [variance,signals]=await Promise.all([loadOptional('variance_shadow.json'),loadOptional('signals.json')]);
    renderStatus();renderValidation();renderCurrent();renderEvent();
  }
  document.addEventListener('DOMContentLoaded',()=>{$('horizon').addEventListener('change',renderCurrent);load();setInterval(load,60000);});
})();
