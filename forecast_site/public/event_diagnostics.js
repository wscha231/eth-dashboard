/* Complete versioned evidence. Historical and published observations never share a score. */
(() => {
  'use strict';
  const H=[6,24,72,168,336,720], $=id=>document.getElementById(id);
  const mean=a=>a.length?a.reduce((s,v)=>s+v,0)/a.length:null;
  const valid=x=>typeof x==='number'&&Number.isFinite(x);
  const fmt=(x,n=2)=>valid(x)?x.toLocaleString('en-US',{maximumFractionDigits:n}):'—';
  const modelName=k=>k==='no_change'?'No-change price':String(k||'').replace(/^(365|730|1095):/,(_,d)=>({365:'1-year ',730:'2-year ',1095:'3-year '}[d])).replace('catboost_calibrated','CatBoost + frequency blend').replace('climatology','past frequency').replace('catboost','CatBoost').replace('logistic','logistic regression');
  const escape=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const matured=r=>r.eligible!==false&&r.settled!==false&&valid(r.return);
  function averagePrecision(rows,key,truth){
    const sorted=rows.slice().sort((a,b)=>b[key]-a[key]),positives=sorted.filter(truth).length;
    if(!positives)return null;
    let tp=0,total=0,area=0,i=0;
    while(i<sorted.length){let j=i,added=0;while(j<sorted.length&&sorted[j][key]===sorted[i][key]){added+=Number(truth(sorted[j]));j++;}tp+=added;total=j;area+=added/positives*tp/total;i=j;}
    return area;
  }
  function event(rows,key,truth,threshold){
    let tp=0,fp=0,positives=0,alerts=0;
    for(const r of rows){const y=Number(truth(r)),a=r[key]>r[threshold];positives+=y;alerts+=a;tp+=a&&y;fp+=a&&!y;}
    return {brier:mean(rows.map(r=>(r[key]-Number(truth(r)))**2)),precision:alerts?tp/alerts:null,
      recall:positives?tp/positives:null,fpr:rows.length>positives?fp/(rows.length-positives):null,
      ap:averagePrecision(rows,key,truth),positives,alerts};
  }
  function reliability(rows,key,truth){
    return Array.from({length:10},(_,i)=>{const a=rows.filter(r=>Math.min(9,Math.floor(r[key]*10))===i);
      const p=mean(a.map(r=>r[key])),rate=mean(a.map(r=>Number(truth(r)))),n=a.length,z=1.96;
      const center=n?(rate+z*z/(2*n))/(1+z*z/n):null;
      const half=n?z*Math.sqrt(rate*(1-rate)/n+z*z/(4*n*n))/(1+z*z/n):null;
      return {x:p,y:rate,n,lower:n?center-half:null,upper:n?center+half:null};});
  }
  function score(input){
    const rows=input.filter(matured),n=rows.length;
    if(!n)return {n:0,independent:0};
    const errors=rows.map(r=>(Math.expm1(r.q50)-Math.expm1(r.return))*100);
    const matrix=Array.from({length:3},()=>[0,0,0]);
    for(const r of rows){const p=[r.p_down,r.p_flat,r.p_up];matrix[r.terminal][p.indexOf(Math.max(...p))]++;}
    let end=null,independent=0;
    for(const r of rows.slice().sort((a,b)=>a.slot.localeCompare(b.slot)))if(end===null||Date.parse(r.slot)>=end){independent++;end=Date.parse(r.target_end);}
    const up=event(rows,'hit_up',r=>r.up,'threshold_up'),down=event(rows,'hit_down',r=>r.down,'threshold_down');
    const supports=matrix.map(a=>a.reduce((s,v)=>s+v,0));
    return {n,independent,errors,matrix,up,down,brier:(up.brier+down.brier)/2,
      terminalBrier:mean(rows.map(r=>[r.p_down,r.p_flat,r.p_up].reduce((s,p,i)=>s+(p-Number(r.terminal===i))**2,0))),
      logloss:mean(rows.map(r=>-Math.log(Math.max(1e-8,[r.p_down,r.p_flat,r.p_up][r.terminal])))),
      accuracy:matrix.reduce((s,a,i)=>s+a[i],0)/n,
      balanced:supports.every(Boolean)?mean(matrix.map((a,i)=>a[i]/supports[i])):null,
      mae:mean(errors.map(Math.abs)),rmse:Math.sqrt(mean(errors.map(e=>e*e))),bias:mean(errors),
      usd:mean(rows.map(r=>Math.abs(r.reference_price*(Math.exp(r.q50)-Math.exp(r.return))))),
      mape:mean(rows.map(r=>Math.abs(Math.exp(r.q50-r.return)-1)*100)),
      noChange:mean(rows.map(r=>Math.abs(Math.expm1(r.return))*100)),
      coverage:mean(rows.map(r=>Number(r.return>=r.q10&&r.return<=r.q90))),
      width:mean(rows.map(r=>(Math.expm1(r.q90)-Math.expm1(r.q10))*100)),
      interval:mean(rows.map(r=>100*(Math.expm1(r.q90)-Math.expm1(r.q10)+10*Math.max(0,Math.expm1(r.q10)-Math.expm1(r.return))+10*Math.max(0,Math.expm1(r.return)-Math.expm1(r.q90))))),
      pinball:mean(rows.flatMap(r=>[.1,.5,.9].map((a,i)=>{const e=100*(Math.expm1(r.return)-Math.expm1(r[['q10','q50','q90'][i]]));return Math.max(a*e,(a-1)*e);}))) };
  }
  function filter(rows,start,end,regime){return rows.filter(r=>(!start||r.slot.slice(0,10)>=start)&&(!end||r.slot.slice(0,10)<=end)&&
    (regime==='all'||(regime.startsWith('trend_')?r.trend_state===regime.slice(6):r.volatility_state===regime.slice(4))));}
  window.EventDiagnosticsMath={score,filter,reliability};
  let payload,source='historical',horizon=24,generation=0,page=0,selected=[],url;
  const cache=new Map(),charts={};
  function table(id,headers,rows){$(id).innerHTML='<table><thead><tr>'+headers.map(h=>'<th>'+escape(h)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+r.map(v=>'<td>'+escape(v)+'</td>').join('')+'</tr>').join('')+'</tbody></table>';}
  async function object(entry){
    if(!entry||!/^archive\/[a-zA-Z0-9_.-]+\.json$/.test(entry.path)||!/^[a-f0-9]{64}$/.test(entry.sha256))throw Error('Archive reference unavailable');
    if(cache.has(entry.sha256))return cache.get(entry.sha256);
    const promise=(async()=>{const response=await fetch(entry.path,{signal:globalThis.AbortSignal?.timeout?.(30000)});if(!response.ok)throw Error('Archive download failed');
      const bytes=await response.arrayBuffer();if(bytes.byteLength!==entry.bytes)throw Error('Archive length mismatch');
      const digest=await crypto.subtle.digest('SHA-256',bytes),sha=Array.from(new Uint8Array(digest),b=>b.toString(16).padStart(2,'0')).join('');
      if(sha!==entry.sha256)throw Error('Archive version mismatch');return JSON.parse(new TextDecoder().decode(bytes));})();
    cache.set(entry.sha256,promise);try{return await promise;}catch(e){cache.delete(entry.sha256);throw e;}
  }
  function chart(id,type,labels,datasets,axis){
    charts[id]?.destroy();delete charts[id];if(typeof Chart!=='function')return;
    charts[id]=new Chart($(id),{type,data:{labels,datasets},options:{responsive:true,maintainAspectRatio:false,animation:false,
      plugins:{legend:{labels:{color:'#b2bdb1'}},tooltip:{callbacks:{afterLabel:ctx=>ctx.raw?.n?`Samples: ${ctx.raw.n}; binomial interval: ${fmt(ctx.raw.lower*100)}–${fmt(ctx.raw.upper*100)}% (overlap unadjusted)`:''}}},
      scales:{x:{ticks:{color:'#b2bdb1',maxTicksLimit:8},grid:{color:'#263128'}},y:{title:{display:!!axis,text:axis,color:'#b2bdb1'},ticks:{color:'#b2bdb1'},grid:{color:'#263128'}}}}});
  }
  const line=(label,data,color,extra={})=>({label,data,borderColor:color,backgroundColor:color,pointRadius:0,borderWidth:1.5,...extra});
  function ledger(){
    const rows=selected.slice().reverse(),max=Math.max(1,Math.ceil(rows.length/50));page=Math.min(page,max-1);
    $('diagnostic-page').textContent=`Page ${page+1} / ${max} · ${rows.length} records`;
    $('diagnostic-prev').disabled=page===0;$('diagnostic-next').disabled=page>=max-1;
    table('diagnostic-ledger',['Origin (UTC)','Target end (UTC)','Model / ID','Issued / published (UTC)','Outcome','Median USD','Actual USD','Truth revision / evaluated'],
      rows.slice(page*50,page*50+50).map(r=>[r.slot,r.target_end,(r.selected_model||'')+' / '+(r.forecast_id||r.model_version||''),
        !['historical','candidate_history'].includes(source)?(r.issued_at||'—')+' / '+(r.published_at||'Unverified'):'Reconstructed',
        r.eligible===false?'Excluded: late / unverified':r.settled===false?'Pending':'Settled',fmt(r.reference_price*Math.exp(r.q50)),fmt(r.actual_price),
        !['historical','candidate_history'].includes(source)?(r.truth_revision||'—')+' / '+(r.outcome_evaluated_at||'—'):'Reconstructed truth']));
  }
  function render(rows,overview,manifest){
    selected=rows;const settled=rows.filter(matured),s=score(rows),p=x=>valid(x)?fmt(x*100)+'%':'—';
    const days=new Set(settled.map(r=>r.slot.slice(0,10))),span=days.size?(Date.parse([...days].sort().at(-1))-Date.parse([...days].sort()[0]))/86400000+1:0;
    $('diagnostic-status').textContent=`${source==='candidate_history'?'Multi-year candidate reconstruction':source==='historical'?'Historical reconstruction':source==='shadow'?'Experimental candidate record':'Published immutable record'} · Data as of ${manifest.as_of} · ${s.n} scored / ${rows.length} records · ${s.independent} non-overlapping windows · ${rows.filter(r=>r.settled===false).length} pending · ${rows.filter(r=>r.eligible===false).length} late or unverified · ${span?span-days.size:0} origin dates without a settled observation inside observed span. Unknown old market states are excluded from specific regime filters.`;
    table('diagnostic-overview',['Window','Scored','Non-overlap','Return MAE (pp)','No-change MAE (pp)','Event Brier','80% coverage'],overview);
    table('diagnostic-metrics',['Measure','Value'],[['Price MAE (USD)',fmt(s.usd)],['Price MAPE',p(valid(s.mape)?s.mape/100:null)],['Return MAE / RMSE (percentage points)',fmt(s.mae)+' / '+fmt(s.rmse)],['Signed return bias (pp)',fmt(s.bias)],['No-change return MAE (pp)',fmt(s.noChange)],['Path / terminal Brier (lower is better)',fmt(s.brier,4)+' / '+fmt(s.terminalBrier,4)],['Terminal log loss',fmt(s.logloss,4)],['Terminal accuracy / balanced accuracy',p(s.accuracy)+' / '+p(s.balanced)],['80% interval observed coverage',p(s.coverage)],['Mean interval width / interval score (pp)',fmt(s.width)+' / '+fmt(s.interval)],['Mean pinball loss at 10%, 50%, 90% (pp)',fmt(s.pinball)]]);
    table('diagnostic-events',['Event','Brier','Precision','Recall','False-positive rate','Average precision','Actual / alerts'],s.n?[['Large rise',s.up],['Large fall',s.down]].map(([label,e])=>[label,fmt(e.brier,4),p(e.precision),p(e.recall),p(e.fpr),fmt(e.ap,4),e.positives+' / '+e.alerts]):[]);
    table('diagnostic-confusion',['Actual ↓ / predicted →','Lower','Within thresholds','Higher'],s.n?s.matrix.map((r,i)=>[['Lower','Within thresholds','Higher'][i],...r]):[]);
    const defs=[['Large rise','hit_up',r=>r.up,'#c9ff7a'],['Large fall','hit_down',r=>r.down,'#fc8c87'],['End lower','p_down',r=>r.terminal===0,'#e0bd6d'],['End within','p_flat',r=>r.terminal===1,'#84bdf5'],['End higher','p_up',r=>r.terminal===2,'#c9a2ff']];
    const reliabilitySets=defs.map(([label,key,truth,color])=>line(label,reliability(settled,key,truth).filter(b=>b.n),color,{showLine:true,pointRadius:4}));
    chart('diagnostic-reliability','scatter',[],[line('Perfect calibration',[{x:0,y:0},{x:1,y:1}],'#607060',{showLine:true,borderDash:[4,4]}),...reliabilitySets],'Observed rate (0–1)');
    table('diagnostic-bins',['Outcome','Predicted probability','Observed rate','Samples'],defs.flatMap(([label,key,truth])=>reliability(settled,key,truth).filter(b=>b.n).map(b=>[label,p(b.x),p(b.y),b.n])));
    // Show genuine pending forecasts as well; only matured outcomes enter scores.
    const plotted=rows.filter(r=>r.eligible!==false).sort((a,b)=>a.target_end.localeCompare(b.target_end));
    const dates=plotted.map(r=>r.target_end),price=k=>plotted.map(r=>valid(r[k])?r.reference_price*Math.exp(r[k]):null);
    const actual=plotted.map(r=>matured(r)?r.reference_price*Math.exp(r.return):null);
    chart('diagnostic-price','line',dates,[line('Actual (settled only)',actual,'#f0f4ed'),line('Original median (includes pending)',price('q50'),'#c9ff7a'),line('10th percentile',price('q10'),'#677e55'),line('90th percentile',price('q90'),'#677e55',{fill:'-1',backgroundColor:'rgba(150,200,110,.12)'})],'Target-end price (USD)');
    const months={};for(const r of settled)(months[r.slot.slice(0,7)]??=[]).push(r);
    const keys=Object.keys(months).sort(),monthly=keys.map(k=>score(months[k]));
    chart('diagnostic-monthly','line',keys,[line('Monthly return MAE (pp)',monthly.map(s=>s.mae),'#c9ff7a'),line('No-change MAE (pp)',monthly.map(s=>s.noChange),'#f0f4ed'),line('Return RMSE (pp)',monthly.map(s=>s.rmse),'#fc8c87')],'Return error (percentage points)');
    const errors=s.errors||[],bound=Math.max(1,...errors.map(Math.abs)),counts=Array(20).fill(0);
    for(const e of errors)counts[Math.min(19,Math.floor((e+bound)/(2*bound)*20))]++;
    chart('diagnostic-errors','bar',counts.map((_,i)=>fmt(-bound+2*bound*i/20,1)+' to '+fmt(-bound+2*bound*(i+1)/20,1)),[{label:'Signed error: predicted − actual (pp)',data:counts,backgroundColor:'#c9ff7a'}],'Settled observations');
    if(url)URL.revokeObjectURL(url);url=URL.createObjectURL(new Blob([JSON.stringify({source,as_of:manifest.as_of,horizon_hours:horizon,filters:{from:$('diagnostic-from').value,to:$('diagnostic-to').value,regime:$('diagnostic-regime').value},points:rows},null,2)],{type:'application/json'}));
    $('diagnostic-download').href=url;$('diagnostic-download').download=`eth-${source}-${horizon}h.json`;
    ledger();
    if(typeof Chart!=='function')$('diagnostic-status').textContent+=' Chart library unavailable; tables and full download remain usable.';
  }
  async function refresh(){
    if(!payload||!$('diagnostic-status'))return;
    const ticket=++generation,start=$('diagnostic-from').value,end=$('diagnostic-to').value,regime=$('diagnostic-regime').value;
    $('diagnostic-status').textContent='Loading verified evidence…';
    try{
      if(start&&end&&start>end)throw Error('Start date must be before end date');
      const manifest=await object(payload.evidence_archives?.[source]);
      if(manifest.kind!==source||manifest.schema_version!==1)throw Error('Archive schema mismatch');
      const groups=await Promise.all(H.map(async h=>{const info=manifest.horizons[String(h)];if(!info)return [h,[]];
        const shards=info.shards.filter(e=>(!start||e.last_origin.slice(0,10)>=start)&&(!end||e.first_origin.slice(0,10)<=end));
        const values=await Promise.all(shards.map(async entry=>{const d=await object(entry);if(d.kind!==source||d.horizon_hours!==h||d.points.length!==entry.rows)throw Error('Archive count mismatch');return d.points;}));
        return [h,filter(values.flat(),start,end,regime)];}));
      if(ticket!==generation)return;
      const overview=groups.map(([h,rows])=>{const s=score(rows);return [h<24?h+' hours':h/24+' days',s.n,s.independent,fmt(s.mae),fmt(s.noChange),fmt(s.brier,4),valid(s.coverage)?fmt(s.coverage*100)+'%':'—'];});
      while(cache.size>256)cache.delete(cache.keys().next().value);
      render(groups.find(([h])=>h===horizon)?.[1]||[],overview,manifest);
    }catch(e){if(ticket===generation){$('diagnostic-status').textContent='Evidence unavailable: '+e.message;selected=[];for(const c of Object.values(charts))c.destroy();for(const key of Object.keys(charts))delete charts[key];['diagnostic-overview','diagnostic-metrics','diagnostic-events','diagnostic-confusion','diagnostic-bins','diagnostic-ledger'].forEach(id=>$(id).textContent='No verified data for this selection.');$('diagnostic-download').removeAttribute('href');}}
  }
  window.updateEventDiagnostics=(next,h)=>{if(!next)return;const changed=payload?.evidence_archives?.[source]?.sha256!==next?.evidence_archives?.[source]?.sha256||horizon!==Number(h);payload=next;horizon=Number(h);if(changed)refresh();
    const historical=next.historical_study;
    $('diagnostic-history-status').textContent=historical?`Monthly chronological candidate evaluation through ${historical.data_as_of}. Each fit uses earlier completed outcomes. Historical reconstruction and actual published evidence remain separate; omitted initial months lack sufficient training/calibration history.`:'A complete multi-year candidate study has not been published yet. Existing historical incumbent results remain available.';
    table('diagnostic-history-table',['Window','First / last origin','Scored / non-overlap','Candidate / incumbent / no-change MAE (pp)','Candidate / incumbent Brier','Candidate 80% coverage','Research decision'],Object.values(historical?.horizons||{}).map(r=>[r.horizon_hours+'h',(r.first_origin||'—').slice(0,10)+' / '+(r.last_origin||'—').slice(0,10),(r.origins||0)+' / '+(r.nonoverlap||0),fmt(r.candidate?.return_mae_pp)+' / '+fmt(r.incumbent?.return_mae_pp)+' / '+fmt(r.no_change?.return_mae_pp),fmt(r.candidate?.event_brier,4)+' / '+fmt(r.incumbent?.event_brier,4),r.candidate?fmt(r.candidate.coverage80*100)+'%':'—',r.decision==='review_candidate_outputs'?'Review individual outputs':r.decision==='retain_incumbent_no_broad_historical_edge'?'No broad historical edge':'Insufficient history']));
    const study=next.optimization;$('diagnostic-optimization').textContent=study?`Candidate review as of ${study.data_as_of}: ${study.status}. ${study.claims}`:'Candidate review has not been published yet. Existing models remain active.';
    table('diagnostic-candidates',['Window','Point / probability / range','Candidate MAE (pp)','Incumbent MAE (pp)','Candidate / incumbent Brier','Promotion'],Object.values(study?.horizons||{}).map(r=>[r.horizon_hours+'h',modelName(r.heads.point)+' / '+modelName(r.heads.probability)+' / '+modelName(r.heads.interval),fmt(r.candidate.return_mae_pp),fmt(r.incumbent.return_mae_pp),fmt(r.candidate.event_brier,4)+' / '+fmt(r.incumbent.event_brier,4),r.promotion==='held_for_prospective_evidence'?'Awaiting live evidence':r.promotion]));
    table('diagnostic-forward',['Window','Matched / non-overlap','Candidate / incumbent MAE (pp)','Candidate / incumbent Brier','Brier difference 95% block interval'],Object.entries(next.candidate_live_comparison||{}).map(([h,r])=>[h+'h',r.paired_rows+' / '+r.nonoverlap,fmt(r.candidate.return_mae_pp)+' / '+fmt(r.incumbent.return_mae_pp),fmt(r.candidate.event_brier,4)+' / '+fmt(r.incumbent.event_brier,4),r.paired_event_brier?fmt(r.paired_event_brier.lower95,4)+' to '+fmt(r.paired_event_brier.upper95,4):'Insufficient independent evidence']));};
  document.addEventListener('DOMContentLoaded',()=>{
    $('diagnostic-source').onchange=e=>{source=e.target.value;page=0;refresh();};
    ['diagnostic-from','diagnostic-to','diagnostic-regime'].forEach(id=>$(id).onchange=()=>{page=0;refresh();});
    $('diagnostic-prev').onclick=()=>{page--;ledger();};$('diagnostic-next').onclick=()=>{page++;ledger();};
  });
})();
