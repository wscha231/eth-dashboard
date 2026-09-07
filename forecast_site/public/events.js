/* Hourly research forecasts. Prices and timestamps stay fixed after issuance. */
(() => {
  let payload, replay, selectedHorizon = '24', selectedPeriod = 'recent_365', selectedRegime = 'all', priceUnit = 'usd';
  let fetchFailed = false, lastDelayed;
  const expectedHorizons = [6,24,72,168,336,720];
  const charts = {};
  const el = id => document.getElementById(id);
  const money = value => Number.isFinite(value) ? new Intl.NumberFormat('en-US', {style:'currency',currency:'USD',maximumFractionDigits:0}).format(value) : '—';
  const pct = value => Number.isFinite(value) ? `${(value*100).toFixed(1)}%` : '—';
  const num = value => Number.isFinite(value) ? value.toFixed(4) : '—';
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
  function delayed() {
    if(!payload || payload.status!=='ready')return true;
    const now=Date.now(),slot=Date.parse(payload.expected_slot),generated=Date.parse(payload.generated_at);
    const dueSlot=Math.floor((now-15*60000)/3600000)*3600000;
    const horizons=(payload.current || []).map(f=>f.horizon_seconds/3600);
    return !Number.isFinite(slot)||!Number.isFinite(generated)||slot<dueSlot||slot>now+120000||
      generated>now+120000||now-generated>100*60000||horizons.length!==6||expectedHorizons.some(h=>!horizons.includes(h))||
      payload.current.some(f=>Date.parse(f.input_cutoff)!==slot);
  }
  function renderStatus() {
    const stale=delayed();
    el('event-system').hidden=false;
    el('event-status').textContent=stale?'데이터·예측 갱신 지연':fetchFailed?'새 예측 연결 확인 중':'시간별 연구 예측';
    el('event-status').className=`pill ${stale||fetchFailed?'warn':''}`;
    el('event-updated').textContent=payload?
      `${fetchFailed?'연결 실패 · 마지막 수신 자료 표시 중. ':''}자료 ${local(payload.expected_slot)} · 발행 자료 생성 ${local(payload.generated_at)} · 다음 갱신 예정 ${local(payload.next_expected_update)}`:
      '새 시간별 데이터의 발행을 확인할 수 없습니다. 연결 상태를 다시 확인합니다.';
    el('run-status').textContent=el('event-status').textContent;
    if(payload && lastDelayed!==stale)renderCards();
    lastDelayed=stale;
  }
  function renderCards() {
    el('event-current').replaceChildren();
    const current=payload.current || [];
    current.forEach(f=>{
      const card=document.createElement('article');card.className='panel';
      const stale=delayed()||!Number.isFinite(Date.parse(f.input_cutoff))||Date.now()-Date.parse(f.input_cutoff)>100*60*1000;
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
  function options(id, entries, value) {
    const node=el(id),key=JSON.stringify(entries);
    if(node.optionsKey!==key){node.replaceChildren();entries.forEach(v=>{const option=text(node,'option',v.label);option.value=v.id;});node.optionsKey=key;}
    node.value=value;
  }
  function skill(summary) {
    const chosen=summary?.models?.find(m=>m.model==='selected'),base=summary?.models?.find(m=>m.model==='climatology');
    return chosen && base?.event_brier ? 1-chosen.event_brier/base.event_brier : null;
  }
  function renderResearch() {
    const forward=payload?.prospective?.[selectedHorizon] || {};
    el('event-prospective-status').textContent=`직전 공개분까지 발행 ${forward.issued||0}건 · 정산 ${forward.resolved||0}건 · 대기 ${forward.pending||0}건 · 겹치지 않는 정산 구간 ${forward.nonoverlap_resolved||0}개. ${forward.performance_watch==='review_required'?'기준 모델보다 성능이 낮아 검토가 필요합니다.':'실전 성능 검증 중입니다.'}`;
    const result=replay?.horizons?.[selectedHorizon];
    if(!result?.models?.length){
      el('event-replay-status').textContent='이 기간의 전체 검증 결과가 아직 준비되지 않았습니다.';
      el('event-segment-download').hidden=true;
      ['event-model-comparison','event-period-comparison'].forEach(id=>el(id).replaceChildren());
      ['event-path-chart','event-price-chart'].forEach(id=>{charts[id]?.destroy();delete charts[id];});
      return;
    }
    const segments=result.segments;
    el('event-segment-download').hidden=!segments;
    const periods=segments?.periods || [{id:'all',label:'검증 전구간'}];
    const regimes=segments?.regimes || [{id:'all',label:'모든 시장 상태'}];
    if(!periods.some(p=>p.id===selectedPeriod))selectedPeriod='all';
    if(!regimes.some(r=>r.id===selectedRegime))selectedRegime='all';
    options('event-period',periods,selectedPeriod);options('event-market-filter',regimes,selectedRegime);
    const period=periods.find(p=>p.id===selectedPeriod);
    const summary=segments ? segments.results[selectedPeriod+'|'+selectedRegime] : result;
    const confidence=summary?.paired_event_brier;
    const interval=confidence ? `사건 오차 차이 95% 탐색 범위 ${num(confidence.lower95)} ~ ${num(confidence.upper95)} (음수가 유리)` : '통계 범위를 판단할 달력 블록이 부족합니다';
    el('event-replay-status').textContent=`${segments?'자료 기준 '+local(segments.as_of):'구간별 집계 준비 중 · 전체 결과'} · ${period.label} · 공통 ${(summary?.common_origins||0).toLocaleString()}개 기점 · 겹치지 않는 구간 ${summary?.nonoverlapping_selected?.rows||0}개 · 사건 확률 오차 개선 ${pct(skill(summary))}. ${interval}. 구간을 여러 번 비교한 탐색 결과이며 실제 성과는 별도로 축적합니다.`;
    table('event-model-comparison',['모델','사건 Brier ↓','상승 재현율','상승 오경보율','하락 재현율','하락 오경보율','만기 균형정확도','가격 오차 개선','80% 범위 포함률'],(summary?.models||[]).map(m=>[names[m.model]||m.model,num(m.event_brier),pct(m.up.recall),pct(m.up.false_positive_rate),pct(m.down.recall),pct(m.down.false_positive_rate),pct(m.terminal_balanced_accuracy),pct(m.mae_skill),pct(m.coverage80)]));
    table('event-period-comparison',['평가 구간','공통 기점','비중복 구간','모델 Brier ↓','기준 Brier ↓','사건 오차 개선','만기 균형정확도','가격 오차 개선','80% 포함률'],periods.map(p=>{
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
    if(!points.length){
      el('event-replay-status').textContent+=' 이 조건에서 만기가 도래한 공통 평가 표본이 없습니다.';
      ['event-path-chart','event-price-chart'].forEach(id=>{charts[id]?.destroy();delete charts[id];});return;
    }
    const dates=points.map(p=>p.slot.slice(0,10));
    const line=(label,key,color,extra={})=>({label,data:points.map(p=>p[key]),borderColor:color,backgroundColor:color,pointRadius:0,borderWidth:1.5,...extra});
    chart('event-path-chart',dates,[line('상승 도달 확률','hit_up','#baff70'),line('하락 도달 확률','hit_down','#fc8c87'),{label:'실제 상승 도달',data:points.map(p=>p.up?1:null),showLine:false,pointRadius:2,borderColor:'#baff70',backgroundColor:'#baff70'},{label:'실제 하락 도달',data:points.map(p=>p.down?1:null),showLine:false,pointRadius:2,borderColor:'#fc8c87',backgroundColor:'#fc8c87'}],'확률 (0–1)');
    const value=key=>points.map(p=>priceUnit==='usd'?p.reference_price*Math.exp(p[key]):Math.expm1(p[key])*100);
    const targetDates=points.map(p=>p.target_end.slice(0,10));
    chart('event-price-chart',targetDates,[{label:priceUnit==='usd'?'실제 만기 가격':'실현 수익률',data:value('return'),borderColor:'#eef5e7',pointRadius:0,borderWidth:1.5},{label:'예측 중앙값',data:value('q50'),borderColor:'#baff70',pointRadius:0,borderWidth:1.5},{label:'하단 10%',data:value('q10'),borderColor:'#677e55',pointRadius:0,borderWidth:1},{label:'상단 90%',data:value('q90'),borderColor:'#677e55',backgroundColor:'rgba(150,200,110,.08)',fill:'-1',pointRadius:0,borderWidth:1}],priceUnit==='usd'?'만기 시점 가격 (USD)':'기준 가격 대비 수익률 (%)');
  }
  function renderLedger() {
    const rows=(payload.recent_issued || []).slice().reverse();
    table('event-ledger',['발행 시각','기간','고정 예측','실현 가격','상승 도달','하락 도달'],rows.slice(0,60).map(r=>[local(r.issued_at),horizonName(r.horizon_seconds/3600),money(r.price_quantiles[1]),money(r.outcome?.actual_price),r.outcome?(r.outcome.up?'도달':'미도달'):'정산 대기',r.outcome?(r.outcome.down?'도달':'미도달'):'정산 대기']));
  }
  window.loadEventForecasts=async()=>{
    try {
      const response=await fetch('signals.json',{cache:'no-store',signal:globalThis.AbortSignal?.timeout?.(15000)});if(!response.ok)throw new Error('unavailable');
      const next=await response.json();if(next.schema_version!==1)throw new Error('schema');
      fetchFailed=false;
      if(payload && new Date(next.generated_at)<new Date(payload.generated_at)){renderStatus();return;}
      payload=next;renderStatus();
      el('model-phase').textContent='ETH 사건 예측 · 연구 베타';
      el('eval-status').textContent='실제 성과 축적 중';
      renderCards();renderLedger();
      const state=payload.current_regime;
      if(state)el('event-regime').textContent=`현재 상태: ${{up:'상승 움직임',down:'하락 움직임',range:'뚜렷한 방향 없음'}[state.state]} · 최근 24시간 ${pct(state.trailing_24h_return)} · 변동성 ${state.volatility_ratio_24h_30d.toFixed(1)}배 (최근 30일 대비). 이미 관측한 움직임을 요약한 값입니다.`;
      if(payload.current?.length){el('ref-price').textContent=money(payload.current[0].reference_price);el('generated-at').textContent=`ETH-USD 기준 ${local(payload.current[0].input_cutoff)}`;}
      if(!replay || replay.generated_at!==payload.replay_generated_at){const r=await fetch('signals_replay.json');if(r.ok)replay=await r.json();}
      renderCards();renderResearch();
    } catch (_) {
      fetchFailed=true;renderStatus();
    }
  };
  document.addEventListener('DOMContentLoaded',()=>{
    // New forecasts load independently of slow or unavailable archived JSON.
    window.loadEventForecasts();
    el('event-horizon').onchange=e=>{selectedHorizon=e.target.value;renderResearch();};
    el('event-period').onchange=e=>{selectedPeriod=e.target.value;renderResearch();};
    el('event-market-filter').onchange=e=>{selectedRegime=e.target.value;renderResearch();};
    el('event-price-unit').onchange=e=>{priceUnit=e.target.value;renderResearch();};
    setInterval(window.loadEventForecasts, 60000);
    // A frozen/failed HTTP feed must not leave previously loaded cards marked fresh.
    setInterval(renderStatus, 30000);
  });
})();
