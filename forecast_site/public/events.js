/* english-outlook-v1: fixed issued prices, matched historical evidence, separate live record. */
(() => {
  let payload, replay, selectedHorizon = '24', selectedPeriod = 'recent_365', selectedRegime = 'all', priceUnit = 'usd';
  let fetchFailed = false, replayFailed = false, lastDelayed, lastCardKey, loading = false;
  const expectedHorizons = [6,24,72,168,336,720];
  const charts = {};
  const el = id => document.getElementById(id);
  const money = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US', {style:'currency',currency:'USD',maximumFractionDigits:0}).format(value) : '—';
  const pct = value => Number.isFinite(value) ? `${(value*100).toFixed(1)}%` : '—';
  const num = value => Number.isFinite(value) ? value.toFixed(4) : '—';
  const local = value => Number.isFinite(Date.parse(value)) ? new Date(value).toLocaleString('en-GB', {timeZone:'UTC',day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit',hour12:false})+' UTC' : '—';
  const horizonName = hours => Number(hours)<24 ? `${hours} hours` : `${Number(hours)/24} ${Number(hours)===24?'day':'days'}`;
  const names = {selected:'Model selected at the time',climatology:'Past-frequency baseline',logistic:'Logistic regression',catboost:'CatBoost',catboost_calibrated:'CatBoost + frequency calibration'};
  const marketNames = {all:'All conditions',trend_up:'Rising market',trend_down:'Falling market',trend_range:'Sideways market',vol_high:'High volatility',vol_normal:'Normal volatility'};
  const periodName = id => id==='all' ? 'All history' : /^recent_\d+$/.test(id) ? `Last ${id.split('_')[1]} days` : /^year_\d{4}$/.test(id) ? id.slice(5) : 'Available history';
  const stamp = value => typeof value==='string' && /(?:Z|[+-]\d{2}:\d{2})$/i.test(value) ? Date.parse(value) : NaN;
  function validForecast(f, now) {
    if(!f || typeof f.forecast_id!=='string' || !f.forecast_id || !Number.isFinite(f.horizon_seconds) || !expectedHorizons.includes(f.horizon_seconds/3600))return false;
    const slot=stamp(f.input_cutoff),available=stamp(f.available_at),issued=stamp(f.issued_at),start=stamp(f.window_start),end=stamp(f.target_end);
    const p=f.terminal_down_flat_up,q=f.price_quantiles;
    return [slot,available,issued,start,end].every(Number.isFinite) && slot%3600000===0 && stamp(f.slot)===slot &&
      available>=slot && available<=issued && issued>=slot && issued<slot+55*60000 && issued<=now && start===slot+3600000 &&
      end-start===f.horizon_seconds*1000 && end>now &&
      Array.isArray(p) && p.length===3 && p.every(v=>Number.isFinite(v)&&v>=0&&v<=1) && Math.abs(p.reduce((a,b)=>a+b,0)-1)<1e-6 &&
      [f.hit_up,f.hit_down].every(v=>Number.isFinite(v)&&v>=0&&v<=1) &&
      Array.isArray(q) && q.length===3 && q.every(v=>Number.isFinite(v)&&v>0) && q[0]<=q[1] && q[1]<=q[2] &&
      [f.reference_price,f.lower_barrier_price,f.upper_barrier_price].every(v=>Number.isFinite(v)&&v>0) &&
      f.lower_barrier_price<f.reference_price && f.reference_price<f.upper_barrier_price;
  }
  function outlook(now=Date.now()) {
    const matches=f=>f.horizon_seconds/3600===Number(selectedHorizon) && validForecast(f,now);
    const current=payload?.current?.find(f=>f && matches(f));
    if(current)return {record:current,previous:false};
    const previous=(payload?.recent_issued || []).filter(f=>f && matches(f) &&
      Number.isFinite(stamp(f.published_at)) && stamp(f.published_at)>=stamp(f.issued_at) &&
      stamp(f.published_at)<stamp(f.window_start) && stamp(f.published_at)<=now)
      .sort((a,b)=>stamp(b.input_cutoff)-stamp(a.input_cutoff) || stamp(b.published_at)-stamp(a.published_at))[0];
    return {record:previous,previous:!!previous};
  }
  function text(parent,tag,value,className='') {
    const node=document.createElement(tag);node.textContent=value;node.className=className;parent.appendChild(node);return node;
  }
  function table(target,head,rows) {
    const node=document.createElement('table'),thead=document.createElement('thead'),body=document.createElement('tbody');
    const hr=document.createElement('tr');head.forEach(v=>{const th=text(hr,'th',v);th.scope='col';});thead.appendChild(hr);
    rows.forEach(values=>{const tr=document.createElement('tr');values.forEach(v=>text(tr,'td',v));body.appendChild(tr);});
    node.append(thead,body);el(target).replaceChildren(node);
  }
  function delayed() {
    if(!payload || payload.status!=='ready')return true;
    const now=Date.now(),slot=Date.parse(payload.expected_slot),generated=Date.parse(payload.generated_at);
    const dueSlot=Math.floor((now-15*60000)/3600000)*3600000;
    const horizons=(payload.current || []).map(f=>f?.horizon_seconds/3600);
    return !Number.isFinite(slot)||!Number.isFinite(generated)||slot%3600000!==0||slot<dueSlot||slot>now+120000||
      generated>now+120000||now-generated>100*60000||horizons.length!==6||expectedHorizons.some(h=>!horizons.includes(h))||
      new Set(payload.current.map(f=>f?.forecast_id)).size!==6 ||
      payload.current.some(f=>!validForecast(f,now)||Date.parse(f.input_cutoff)!==slot);
  }
  function renderStatus() {
    const stale=delayed();
    el('event-status').textContent=!payload?'Forecast unavailable':stale?'Update delayed':fetchFailed?'Connection interrupted':'Up to date';
    el('event-status').className=`pill ${stale||fetchFailed?'warn':'good'}`;
    el('event-updated').textContent=payload ?
      `${fetchFailed?'Showing the last received data. ':''}Inputs: ${local(payload.expected_slot)} · Last update: ${local(payload.generated_at)}${stale?' · A complete current forecast is not available.':''}` :
      'The latest forecast could not be confirmed. Retrying automatically.';
    // Expiry matters even when the overall delayed status has not changed.
    const chosen=outlook(),cardKey=`${selectedHorizon}:${chosen.record?.forecast_id||''}:${chosen.previous}`;
    if(lastDelayed!==stale || lastCardKey!==cardKey)renderCards();
    lastDelayed=stale;lastCardKey=cardKey;
  }
  function renderCards() {
    const root=el('event-current');root.replaceChildren();
    const {record:f,previous}=outlook();
    el('ref-price').textContent=money(f?.reference_price);
    el('generated-at').textContent=f ? `Reference at ${local(f.input_cutoff)}` : 'No current reference for this forecast window';
    if(!f) {
      const empty=text(root,'article','','panel empty-outlook');
      text(empty,'h2',`${horizonName(selectedHorizon)} outlook unavailable`);
      text(empty,'p','No current forecast is available for this window. Historical test results and previously published records remain available below.');
      return;
    }
    const card=text(root,'article','','panel');
    const stale=delayed()||previous;
    text(card,'p',`${horizonName(selectedHorizon)} outlook · ${previous?'Previous published forecast · Update delayed':stale?'Update delayed':'Published forecast'}`,'eyebrow');
    const top=text(card,'div','','outlook-top');
    text(top,'h2',money(f.price_quantiles?.[1]),'outlook-price');
    text(top,'span','Estimated end price','small');
    text(card,'p',`Intended 80% price range: ${money(f.price_quantiles?.[0])} – ${money(f.price_quantiles?.[2])}`);
    text(card,'p',`Observation window: ${local(f.window_start)} → ${local(f.target_end)}`,'small');
    text(card,'h3','Where might the price finish?');
    const direction=text(card,'div','','direction-grid');
    ['lower','middle','higher'].forEach((kind,i)=>{
      const cell=text(direction,'div','',`direction-cell ${kind}`);
      text(cell,'b',pct(f.terminal_down_flat_up?.[i]));
      text(cell,'span',[`Below ${money(f.lower_barrier_price)}`,'Between thresholds',`Above ${money(f.upper_barrier_price)}`][i]);
    });
    text(card,'p','These three end-price probabilities add to 100%. They are model estimates, not a record of accuracy.','small');
    const detail=text(card,'details','');text(detail,'summary','Could a large move happen along the way?');
    const bands=text(detail,'div','','event-probabilities');
    text(bands,'span',`Touch ${money(f.upper_barrier_price)} or higher: ${pct(f.hit_up)}`,'event-up');
    text(bands,'span',`Touch ${money(f.lower_barrier_price)} or lower: ${pct(f.hit_down)}`,'event-down');
    text(detail,'p','Both thresholds can be touched within the window, so these two probabilities can add to more than 100%.','small');
    text(detail,'p',`Issued: ${local(f.issued_at)} · ${names[f.selected_model] || f.selected_model || 'Research model'}`,'small');
    if(previous)text(card,'p',`Published: ${local(f.published_at)}. The latest update is delayed. This earlier forecast keeps its original prices and observation window.`,'notice');
    else if(stale)text(card,'p','This forecast is out of date or the release is incomplete. The original estimate is preserved; wait for a complete update before treating it as the current outlook.','notice');
  }
  function chart(id,labels,datasets,yLabel) {
    charts[id]?.destroy();delete charts[id];
    if(typeof Chart==='undefined') {
      el('event-chart-status').hidden=false;
      el('event-chart-status').textContent='The chart library could not load. The performance figures and downloadable results are still available.';
      return;
    }
    charts[id]=new Chart(el(id),{
      type:'line',data:{labels,datasets},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:'#dce5d9',font:{size:13}}}},
        scales:{x:{ticks:{color:'#aebcaa',maxTicksLimit:7},grid:{color:'#2b352d'}},
          y:{title:{display:true,text:yLabel,color:'#b8c8ac'},ticks:{color:'#aebcaa'},grid:{color:'#2b352d'},...(id==='event-path-chart'?{min:0,max:1}:{})}}
      }
    });
  }
  function options(id, entries, value) {
    const node=el(id),key=JSON.stringify(entries);
    if(node.optionsKey!==key){node.replaceChildren();entries.forEach(v=>{const option=text(node,'option',v.label);option.value=v.id;});node.optionsKey=key;}
    node.value=value;
  }
  function improvement(chosen,base) {
    return Number.isFinite(chosen?.event_brier) && Number.isFinite(base?.event_brier) && base.event_brier>0 ? 1-chosen.event_brier/base.event_brier : null;
  }
  function skill(summary) {
    return improvement(summary?.models?.find(m=>m.model==='selected'),summary?.models?.find(m=>m.model==='climatology'));
  }
  function errorLabel(value) {
    return !Number.isFinite(value)?'Not available':Math.abs(value)<0.0005?'About the same':`${pct(Math.abs(value))} ${value>0?'less':'more'} error`;
  }
  function score(parent,label,value,note,tone='') {
    const card=text(parent,'div','','score-card');text(card,'h3',label);text(card,'strong',value,`score-value ${tone}`);text(card,'p',note);
  }
  function renderScorecards(summary) {
    const root=el('event-scorecards');root.replaceChildren();
    const chosen=summary?.models?.find(m=>m.model==='selected'),value=skill(summary),n=summary?.common_origins||0;
    const bounds=summary?.paired_event_brier;
    el('event-verdict').textContent=!n ? 'No settled test results for this selection' : !Number.isFinite(value)?'A matched baseline comparison is not available':
      value<=0?'The model has not beaten the simple baseline here.':bounds && bounds.upper95<0?'Lower historical probability error than the baseline.':'Some improvement, but the evidence is not conclusive.';
    el('event-verdict-note').textContent=!n ? 'Try a longer test period. Unfinished forecasts are excluded from the scores.' :
      'The baseline uses past event frequency. Lower error means better probability estimates; it is not the percentage of forecasts that were correct.';
    score(root,'Probability error',errorLabel(value),'Compared with the past-frequency baseline on the same observations.',Number.isFinite(value)?value>0?'up':'down':'');
    for(const [key,label,move] of [['up','Large rises caught','rises'],['down','Large falls caught','falls']]) {
      const metric=chosen?.[key];
      const note=Number.isFinite(metric?.recall)?`About ${Math.round(metric.recall*100)} out of 100 actual large ${move} were flagged. False alarms: ${pct(metric.false_positive_rate)} of periods without that move.`:'Not enough observed events to calculate a catch rate.';
      score(root,label,pct(metric?.recall),note);
    }
    score(root,'Price range hit rate',pct(chosen?.coverage80),Number.isFinite(chosen?.coverage80)?`About ${Math.round(chosen.coverage80*100)} out of 100 final prices fell inside the forecast range. The model aims for 80%.`:'No settled price ranges are available for this selection.');
    el('event-sample-status').textContent=`${horizonName(selectedHorizon)} · ${periodName(selectedPeriod)} · ${marketNames[selectedRegime]||'Available conditions'} · ${n.toLocaleString()} settled observations · ${summary?.nonoverlapping_selected?.rows||0} non-overlapping windows. Historical results do not establish live performance.`;
  }
  function renderProspective() {
    const forward=payload?.prospective?.[selectedHorizon];
    const root=el('event-live-scorecards');root.replaceChildren();
    el('event-live-verdict').textContent=forward?.performance_watch==='review_required'?'Live performance needs review':(forward?.nonoverlap_resolved||0)<20?'Live evidence is still building':'Live research record';
    el('event-prospective-status').textContent=payload ? `${horizonName(selectedHorizon)} · Through the previous verified release: ${forward?.issued||0} published, ${forward?.resolved||0} settled, ${forward?.pending||0} awaiting outcomes. ${forward?.nonoverlap_resolved||0} non-overlapping settled windows. ${forward?.performance_watch==='review_required'?'Probability error exceeded the monitoring threshold.':'A paid-service performance claim is not established.'}` : 'The published record is temporarily unavailable.';
    const value=improvement(forward?.paired_selected,forward?.baseline);
    score(root,'Published',payload?String(forward?.issued||0):'—','Published in time for the observation window.');
    score(root,'Settled',payload?String(forward?.resolved||0):'—','The full window has ended and an outcome is available.');
    score(root,'Awaiting outcome',payload?String(forward?.pending||0):'—','Excluded from performance until settled.');
    score(root,'Live probability error',errorLabel(value),'Compared with the saved baseline on matching settled forecasts.');
  }
  function renderResearch() {
    renderProspective();
    const result=replay?.horizons?.[selectedHorizon];
    if(!result?.models?.length){
      renderScorecards(null);
      el('event-verdict').textContent=replayFailed?'Historical results are temporarily unavailable':'Historical results are not ready for this window';
      el('event-replay-status').textContent='Historical results are not ready for this window.';
      el('event-segment-download').hidden=true;
      ['event-model-comparison','event-period-comparison'].forEach(id=>el(id).replaceChildren());
      ['event-path-chart','event-price-chart'].forEach(id=>{charts[id]?.destroy();delete charts[id];});
      el('event-chart-status').hidden=false;el('event-chart-status').textContent='No historical chart is available for this window.';
      return;
    }
    const segments=result.segments;
    el('event-segment-download').hidden=!segments;
    const periods=(segments?.periods || [{id:'all'}]).map(p=>({...p,label:periodName(p.id)}));
    const regimes=(segments?.regimes || [{id:'all'}]).map(r=>({...r,label:marketNames[r.id]||'Available conditions'}));
    if(!periods.some(p=>p.id===selectedPeriod))selectedPeriod='all';
    if(!regimes.some(r=>r.id===selectedRegime))selectedRegime='all';
    options('event-period',periods,selectedPeriod);options('event-market-filter',regimes,selectedRegime);
    const period=periods.find(p=>p.id===selectedPeriod);
    const summary=segments ? segments.results[selectedPeriod+'|'+selectedRegime] : result;
    renderScorecards(summary);
    if(segments?.as_of)el('event-sample-status').textContent+=` Data as of ${local(segments.as_of)}.`;
    if(replayFailed)el('event-verdict-note').textContent+=' The newest historical results could not load; the previous dated results are shown.';
    const confidence=summary?.paired_event_brier;
    const interval=confidence ? `Exploratory 95% range for the probability-error difference: ${num(confidence.lower95)} to ${num(confidence.upper95)} (negative favors the model).` : 'Too few calendar blocks for an uncertainty range.';
    el('event-replay-status').textContent=`${segments?'Data as of '+local(segments.as_of):'All available results'} · ${period.label} · ${(summary?.common_origins||0).toLocaleString()} matched observations · ${summary?.nonoverlapping_selected?.rows||0} non-overlapping windows · Probability-error improvement: ${pct(skill(summary))}. ${interval} These overlapping slices are exploratory comparisons, not a model-promotion test.${replayFailed?' The newest test file could not be loaded; this is the previous available test.':''}`;
    table('event-model-comparison',['Model','Probability error ↓','Rise catch rate','False-rise rate','Rise alert precision','Fall catch rate','False-fall rate','Fall alert precision','Balanced direction accuracy','Price error improvement','80% range hit rate'],(summary?.models||[]).map(m=>[names[m.model]||m.model,num(m.event_brier),pct(m.up?.recall),pct(m.up?.false_positive_rate),pct(m.up?.precision),pct(m.down?.recall),pct(m.down?.false_positive_rate),pct(m.down?.precision),pct(m.terminal_balanced_accuracy),pct(m.mae_skill),pct(m.coverage80)]));
    table('event-period-comparison',['Test period','Observations','Non-overlapping windows','Model error ↓','Baseline error ↓','Probability-error improvement','Balanced direction accuracy','Price error improvement','80% range hit rate'],periods.map(p=>{
      const s=segments ? segments.results[p.id+'|'+selectedRegime] : result;
      const chosen=s?.models?.find(m=>m.model==='selected'),base=s?.models?.find(m=>m.model==='climatology');
      return [p.label,s?.common_origins||0,s?.nonoverlapping_selected?.rows||0,num(chosen?.event_brier),num(base?.event_brier),pct(skill(s)),pct(chosen?.terminal_balanced_accuracy),pct(chosen?.mae_skill),pct(chosen?.coverage80)];
    }));
    const points=(result.points || []).filter(p=>{
      if(!segments)return true;
      const slot=Date.parse(p.slot);
      return slot>=Date.parse(period.start)&&slot<Date.parse(period.end)&&
        (!selectedRegime.startsWith('trend_')||p.trend_state===selectedRegime.slice(6))&&
        (!selectedRegime.startsWith('vol_')||p.volatility_state===selectedRegime.slice(4));
    });
    el('event-chart-status').hidden=true;
    if(!points.length){
      el('event-replay-status').textContent+=' No settled matching observations for this selection.';
      el('event-chart-status').textContent='No settled matching observations for this selection.';el('event-chart-status').hidden=false;
      ['event-path-chart','event-price-chart'].forEach(id=>{charts[id]?.destroy();delete charts[id];});return;
    }
    const dates=points.map(p=>p.slot.slice(0,10));
    const line=(label,key,color)=>({label,data:points.map(p=>p[key]),borderColor:color,backgroundColor:color,pointRadius:0,borderWidth:1.5});
    chart('event-path-chart',dates,[line('Chance of a large rise','hit_up','#baff70'),line('Chance of a large fall','hit_down','#fc8c87'),{label:'Rise happened',data:points.map(p=>p.up?1:null),showLine:false,pointRadius:2,borderColor:'#baff70',backgroundColor:'#baff70'},{label:'Fall happened',data:points.map(p=>p.down?1:null),showLine:false,pointRadius:2,borderColor:'#fc8c87',backgroundColor:'#fc8c87'}],'Probability (0–1)');
    const value=key=>points.map(p=>priceUnit==='usd'?p.reference_price*Math.exp(p[key]):Math.expm1(p[key])*100);
    const targetDates=points.map(p=>p.target_end.slice(0,10));
    chart('event-price-chart',targetDates,[{label:priceUnit==='usd'?'Actual end price':'Actual return',data:value('return'),borderColor:'#eef5e7',pointRadius:0,borderWidth:1.5},{label:'Forecast midpoint',data:value('q50'),borderColor:'#baff70',pointRadius:0,borderWidth:1.5},{label:'Lower bound (10th percentile)',data:value('q10'),borderColor:'#677e55',pointRadius:0,borderWidth:1},{label:'Upper bound (90th percentile)',data:value('q90'),borderColor:'#677e55',backgroundColor:'rgba(150,200,110,.12)',fill:'-1',pointRadius:0,borderWidth:1}],priceUnit==='usd'?'Price at forecast end (USD)':'Return from reference price (%)');
  }
  function renderLedger() {
    const rows=(payload?.recent_issued || []).filter(r=>r.horizon_seconds/3600===Number(selectedHorizon)).slice().reverse();
    table('event-ledger',['Published (UTC)','Publication','Window','End date (UTC)','Fixed estimate','Actual price','Large rise','Large fall'],rows.slice(0,60).map(r=>[local(r.published_at),r.published_at?(Date.parse(r.published_at)<Date.parse(r.window_start)?'On time':'Late'):'Unverified',horizonName(r.horizon_seconds/3600),local(r.target_end),money(r.price_quantiles?.[1]),money(r.outcome?.actual_price),r.outcome?(r.outcome.up?'Reached':'Not reached'):'Pending',r.outcome?(r.outcome.down?'Reached':'Not reached'):'Pending']));
  }
  window.loadEventForecasts=()=>{
    if(loading)return loading;
    loading=(async()=>{
    try {
      const response=await fetch('signals.json',{cache:'no-store',signal:globalThis.AbortSignal?.timeout?.(15000)});if(!response.ok)throw new Error('unavailable');
      const next=await response.json();if(next.schema_version!==1||!Array.isArray(next.current))throw new Error('schema');
      fetchFailed=false;
      if(payload && new Date(next.generated_at)<new Date(payload.generated_at)){renderStatus();return;}
      payload=next;renderStatus();renderCards();renderLedger();renderProspective();
      const state=payload.current_regime;
      el('event-regime').textContent=state?`Observed market: ${{up:'rising',down:'falling',range:'sideways'}[state.state]||'unclassified'} · Last 24 hours: ${pct(state.trailing_24h_return)} · Volatility: ${Number.isFinite(state.volatility_ratio_24h_30d)?state.volatility_ratio_24h_30d.toFixed(1):'—'}× the 30-day level. This describes past movement, not a forecast.`:'';
      try {
        if(!replay || replay.generated_at!==payload.replay_generated_at){
          const r=await fetch('signals_replay.json',{cache:'no-store',signal:globalThis.AbortSignal?.timeout?.(30000)});
          if(!r.ok)throw new Error('replay unavailable');
          const nextReplay=await r.json();if(!nextReplay.horizons)throw new Error('replay schema');
          replay=nextReplay;
        }
        replayFailed=false;
      } catch (_) {replayFailed=true;}
      renderResearch();
    } catch (_) {
      fetchFailed=true;renderStatus();if(!payload){renderCards();renderResearch();}
    } finally {loading=false;}
    })();
    return loading;
  };
  document.addEventListener('DOMContentLoaded',()=>{
    // Current forecasts are independent of archived JSON and the chart CDN.
    window.loadEventForecasts();
    el('event-horizon').onchange=e=>{selectedHorizon=e.target.value;renderCards();renderResearch();renderLedger();};
    el('event-period').onchange=e=>{selectedPeriod=e.target.value;renderResearch();};
    el('event-market-filter').onchange=e=>{selectedRegime=e.target.value;renderResearch();};
    el('event-price-unit').onchange=e=>{priceUnit=e.target.value;renderResearch();};
    el('event-path-details').ontoggle=()=>charts['event-path-chart']?.resize?.();
    setInterval(window.loadEventForecasts,60000);
    setInterval(renderStatus,30000);
  });
})();
