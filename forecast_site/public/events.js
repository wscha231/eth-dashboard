/* Hourly research forecasts. Prices and timestamps stay fixed after issuance. */
(() => {
  let payload, replay, selectedHorizon = '24', selectedPeriod = 'recent';
  const charts = {};
  const el = id => document.getElementById(id);
  const money = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US', {style:'currency',currency:'USD',maximumFractionDigits:0}).format(value) : '—';
  const pct = value => Number.isFinite(value) ? `${(value*100).toFixed(1)}%` : '—';
  const local = value => value ? new Date(value).toLocaleString('ko-KR',{timeZone:'Asia/Seoul',month:'2-digit',day:'2-digit',hour:'2-digit',minute:'2-digit',hour12:false})+' KST' : '—';
  const horizonName = hours => Number(hours)<24 ? `${hours}시간` : `${Number(hours)/24}일`;
  const names = {selected:'당시 검증으로 선택',climatology:'과거 사건 빈도',logistic:'로지스틱',catboost:'CatBoost',catboost_calibrated:'CatBoost + 빈도 보정'};
  function text(parent,tag,value,className='') {
    const node=document.createElement(tag);node.textContent=value;node.className=className;parent.appendChild(node);return node;
  }
  function table(target,head,rows) {
    const node=document.createElement('table'),thead=document.createElement('thead'),body=document.createElement('tbody');
    const hr=document.createElement('tr');head.forEach(v=>text(hr,'th',v));thead.appendChild(hr);
    rows.forEach(values=>{const tr=document.createElement('tr');values.forEach(v=>text(tr,'td',v));body.appendChild(tr);});
    node.append(thead,body);el(target).replaceChildren(node);
  }
  function renderCards() {
    el('event-current').replaceChildren();
    const current=payload.current || [];
    current.forEach(f=>{
      const card=document.createElement('article');card.className='panel';
      const stale=Date.now()-new Date(f.input_cutoff).getTime()>100*60*1000;
      text(card,'p',`${horizonName(f.horizon_seconds/3600)} 전망 · ${stale?'갱신 지연':'실제 발행 기록'}`,'eyebrow');
      text(card,'h2',`${money(f.price_quantiles[1])}`);
      text(card,'p',`만기 가격 명목 80% 범위 ${money(f.price_quantiles[0])} – ${money(f.price_quantiles[2])}`);
      const bands=document.createElement('div');bands.className='event-probabilities';
      text(bands,'span',`상승 도달 ${pct(f.hit_up)}`,'event-up');
      text(bands,'span',`하락 도달 ${pct(f.hit_down)}`,'event-down');card.appendChild(bands);
      text(card,'p',`상승 기준 ${money(f.upper_barrier_price)} / 하락 기준 ${money(f.lower_barrier_price)}`,'small');
      text(card,'p',`만기 상승 ${pct(f.terminal_down_flat_up[2])} · 중립 ${pct(f.terminal_down_flat_up[1])} · 하락 ${pct(f.terminal_down_flat_up[0])}`,'small');
      text(card,'p',`기준 ${money(f.reference_price)} · ${local(f.input_cutoff)}\n관측 시작 ${local(f.window_start)} → 만기 ${local(f.target_end)}`,'small');
      text(card,'p',`${names[f.selected_model] || f.selected_model} · 발행 ${local(f.issued_at)}`,'small');
      const research=replay?.horizons?.[String(f.horizon_seconds/3600)];
      const chosen=research?.models?.find(m=>m.model==='selected');
      const baseline=research?.models?.find(m=>m.model==='climatology');
      if(chosen && baseline?.event_brier) {
        const skill=1-chosen.event_brier/baseline.event_brier;
        text(card,'p',skill>0?`과거 사건 확률 오차 ${pct(skill)} 개선 · 실전 검증 중`:'과거 사건 확률 오차가 기준보다 큽니다 · 연구용 참고치',skill>0?'small':'notice');
      }
      el('event-current').appendChild(card);
    });
    if(!current.length)text(el('event-current'),'p','새 예측을 발행할 완전한 데이터 또는 검증된 월별 모델을 기다리고 있습니다. 과거 발행 기록은 아래에서 확인할 수 있습니다.','notice');
  }
  function chart(id,labels,datasets,yLabel) {
    charts[id]?.destroy();
    charts[id]=new Chart(el(id),{
      type:'line',data:{labels,datasets},
      options:{responsive:true,maintainAspectRatio:false,animation:false,
        interaction:{mode:'index',intersect:false},plugins:{legend:{labels:{color:'#dde4d5'}}},
        scales:{x:{ticks:{color:'#9cac92',maxTicksLimit:7},grid:{color:'#30392b'}},
          y:{title:{display:true,text:yLabel,color:'#b8c8ac'},ticks:{color:'#9cac92'},grid:{color:'#30392b'}}}
      }
    });
  }
  function renderResearch() {
    const result=replay?.horizons?.[selectedHorizon];
    if(!result?.models?.length){el('event-replay-status').textContent='이 기간의 전체 검증 결과가 아직 준비되지 않았습니다.';return;}
    const sel=result.models.find(m=>m.model==='selected'),base=result.models.find(m=>m.model==='climatology');
    const edge=base?.event_brier ? 1-sel.event_brier/base.event_brier : null;
    const confidence=result.paired_event_brier;
    el('event-replay-status').textContent=`${result.first_origin.slice(0,10)} – ${result.last_origin.slice(0,10)} · 공통 ${result.common_origins.toLocaleString()}개 기점 · 사건 확률 오차 개선 ${pct(edge)} · ${confidence?.upper95<0?'과거 구간의 개선 신호가 있습니다.':'기준 빈도 대비 일관된 개선이 확인되지 않았습니다.'} 실제 발행 성과는 별도로 축적합니다.`;
    table('event-model-comparison',['모델','사건 Brier ↓','상승 재현율','상승 오경보율','하락 재현율','하락 오경보율','가격 오차 개선','80% 범위 포함률'],result.models.map(m=>[names[m.model]||m.model,m.event_brier.toFixed(4),pct(m.up.recall),pct(m.up.false_positive_rate),pct(m.down.recall),pct(m.down.false_positive_rate),pct(m.mae_skill),pct(m.coverage80)]));
    const all=result.points || [],last=all.length ? new Date(all.at(-1).slot).getTime() : 0;
    const points=all.filter(p=>selectedPeriod==='all'||new Date(p.slot).getTime()>=last-365*86400000);
    const dates=points.map(p=>p.slot.slice(0,10));
    const line=(label,key,color,extra={})=>({label,data:points.map(p=>p[key]),borderColor:color,backgroundColor:color,pointRadius:0,borderWidth:1.5,...extra});
    chart('event-path-chart',dates,[line('상승 도달 확률','hit_up','#baff70'),line('하락 도달 확률','hit_down','#fc8c87'),{label:'실제 상승 도달',data:points.map(p=>p.up?1:null),showLine:false,pointRadius:2,borderColor:'#baff70',backgroundColor:'#baff70'},{label:'실제 하락 도달',data:points.map(p=>p.down?1:null),showLine:false,pointRadius:2,borderColor:'#fc8c87',backgroundColor:'#fc8c87'}],'확률 (0–1)');
    const ret=key=>points.map(p=>Math.expm1(p[key])*100);
    chart('event-price-chart',dates,[{label:'실현 수익률',data:ret('return'),borderColor:'#eef5e7',pointRadius:0,borderWidth:1.5},{label:'예측 중앙값',data:ret('q50'),borderColor:'#baff70',pointRadius:0,borderWidth:1.5},{label:'하단 10%',data:ret('q10'),borderColor:'#677e55',pointRadius:0,borderWidth:1},{label:'상단 90%',data:ret('q90'),borderColor:'#677e55',backgroundColor:'rgba(150,200,110,.08)',fill:'-1',pointRadius:0,borderWidth:1}],'기준 가격 대비 수익률 (%)');
    const forward=payload.prospective?.[selectedHorizon] || {};
    el('event-prospective-status').textContent=`직전 공개분까지 발행 ${forward.issued||0}건 · 정산 ${forward.resolved||0}건 · 대기 ${forward.pending||0}건 · 겹치지 않는 정산 구간 ${forward.nonoverlap_resolved||0}개. ${forward.performance_watch==='review_required'?'기준 모델보다 성능이 낮아 검토가 필요합니다.':'실전 성능 검증 중입니다.'}`;
  }
  function renderLedger() {
    const rows=(payload.recent_issued || []).slice().reverse();
    table('event-ledger',['발행 시각','기간','고정 예측','실현 가격','상승 도달','하락 도달'],rows.slice(0,60).map(r=>[local(r.issued_at),horizonName(r.horizon_seconds/3600),money(r.price_quantiles[1]),money(r.outcome?.actual_price),r.outcome?(r.outcome.up?'도달':'미도달'):'정산 대기',r.outcome?(r.outcome.down?'도달':'미도달'):'정산 대기']));
  }
  window.loadEventForecasts=async()=>{
    try {
      const response=await fetch('signals.json',{cache:'no-store',signal:globalThis.AbortSignal?.timeout?.(15000)});if(!response.ok)throw new Error('unavailable');
      const next=await response.json();if(next.schema_version!==1)throw new Error('schema');
      if(payload && new Date(next.generated_at)<new Date(payload.generated_at))return;
      payload=next;
      el('event-system').hidden=false;
      const stale=Date.now()-new Date(payload.expected_slot).getTime()>100*60*1000;
      el('event-status').textContent=stale||payload.status!=='ready'?'데이터·예측 갱신 지연':'시간별 연구 예측';
      el('event-status').className=`pill ${stale||payload.status!=='ready'?'warn':''}`;
      el('event-updated').textContent=`자료 ${local(payload.expected_slot)} · 화면 갱신 ${local(payload.generated_at)} · 다음 갱신 예정 ${local(payload.next_expected_update)}`;
      el('model-phase').textContent='ETH 사건 예측 · 연구 베타';el('run-status').textContent=el('event-status').textContent;
      el('eval-status').textContent='실제 성과 축적 중';
      renderCards();renderLedger();
      const state=payload.current_regime;
      if(state)el('event-regime').textContent=`현재 상태: ${{up:'상승 움직임',down:'하락 움직임',range:'뚜렷한 방향 없음'}[state.state]} · 최근 24시간 ${pct(state.trailing_24h_return)} · 변동성 ${state.volatility_ratio_24h_30d.toFixed(1)}배 (최근 30일 대비). 이미 관측한 움직임을 요약한 값입니다.`;
      if(payload.current?.length){el('ref-price').textContent=money(payload.current[0].reference_price);el('generated-at').textContent=`ETH-USD 기준 ${local(payload.current[0].input_cutoff)}`;}
      if(!replay || replay.generated_at!==payload.replay_generated_at){const r=await fetch('signals_replay.json');if(r.ok)replay=await r.json();}
      renderCards();renderResearch();
    } catch (_) {
      el('event-status').textContent='새 예측 자료 확인 중';
      el('event-updated').textContent='새 시간별 데이터의 발행을 확인할 수 없습니다. 아래의 이전 기록은 그대로 유지됩니다.';
    }
  };
  document.addEventListener('DOMContentLoaded',()=>{
    // New forecasts load independently of slow or unavailable archived JSON.
    window.loadEventForecasts();
    el('event-horizon').onchange=e=>{selectedHorizon=e.target.value;renderResearch();};
    el('event-period').onchange=e=>{selectedPeriod=e.target.value;renderResearch();};
    setInterval(window.loadEventForecasts, 60000);
  });
})();
