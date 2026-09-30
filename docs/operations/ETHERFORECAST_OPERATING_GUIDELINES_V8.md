# EtherForecast 프로젝트 운영지침 v8

작성일: 2026-09-27 KST. 투자 프로젝트 v8의 무결성·검증·비용 원칙을 EtherForecast에 적용한다. 로드맵 #68의 버전과는 별개이며 이 문서 추가는 runtime 변경·모델 승격 승인이 아니다.

## 범위·목표
대상은 wscha231/eth-dashboard, etherforecast.live, EtherForecast-Storage와 관련 예약뿐이다. R1000·KR quant·다른 프로젝트는 수정하지 않는다. 목적은 ETH 가격 중심값·분포·변동성·사건 확률을 적시에 발행하고 실제 사후 결과로 검증하는 서비스다. 수익률 보장·자동매매가 아니다. 기존 6h/24h/72h/7d/14d/30d를 유지한다. 주식 프로젝트의 CAGR 목표·RS 가중치·포트 비중·거래일 달력을 이식하지 않는다.
우선순위: 데이터/권한/PIT → 발행·정산·보관 연속성 → Head별 OOS/prospective → 해석 가능성 → 비용·속도 → 새 기능.

## 시작·공식 상태
모든 작업 시작에 WORK_NEEDED=YES/NO, CODEX_REQUIRED=YES/NO와 이유를 적는다. 최신 main, #68 최신 승인 checkpoint, #56 동결 검증 계약, 관련 PR/Issue, 작은 실제 receipt와 실패/기각 실험부터 확인한다. Issue 본문의 오래된 체크박스나 채팅 기억으로 재구현하지 않는다. 변경이 없으면 저장소/Drive 전체를 반복 조사하지 않는다.
실제 발행 사실은 immutable issuance/delivery/outcome 원본과 검증 receipt가 우선이다. 정책은 사용자의 최신 명시 결정과 계약·승인 기록을 확인한다. 코드 최신성≠데이터 최신성≠운영 정상≠예측력≠승인. GitHub=코드/CI/상태/manifest, Drive=장기 원본/checkpoint/artifact/receipt. 운영 predictor는 signal_pipeline+event_hourly.yml. 구 daily/hybrid/retrain은 의존성 확인 전 삭제하지 않는다.

## 데이터·G0
observation/public_available/received/issue/public_delivery/target 시간을 분리한다. 닫힌 24/7 crypto bar, 거래소·가격/거래량 단위·간격·연속성·수정/vintage를 검증한다. missing/stale/future/conflicting/synthetic/ineligible 입력은 해당 Head/horizon을 차단한다. 결측을 0·중립값·오늘 날짜로 채우지 않는다. 공통 원본 손상은 관련 전체를 차단하되 별도 연구 소스 실패로 유효한 baseline까지 불필요하게 정지하지 않는다.
수집/보관/연구/MODEL/웹/API/유료 이용권을 구분한다. ETF·공급·스테이킹·청산·파생 자료 수집 성공은 모델 승인이 아니다. 재구성 역사는 과거 실수신 receipt가 아니다. Telegram/community는 발견용이며 공식 근거와 source contract 확인 전 운영 확률·BUY에 쓰지 않는다. 온체인 canonical/reorg/finality/단위와 무사건/수집중단을 구분한다.

## 네 Head와 검증
Center / Distribution / Variance / Event를 분리한다. Brier 성과로 q50을, 변동성 성과로 방향·중심값을 승격하지 않는다. #56/#68 gate를 완화하지 않는다.
Center: no-change 대비 MAE skill≥2%, incumbent 대비≥1%; 좋은 baseline 대비 RMSE 악화≤2%; 사전 블록≥2/3 개선; paired≥90%.
Distribution: 양 baseline 대비 WIS 개선≥2%; 80% 포함률76–84%; 폭≤좋은 baseline110%; paired≥90%.
Variance: persistence 대비 QLIKE 개선≥5%; 블록≥2/3 개선; 블록 악화>10% 금지.
Event: prior frequency 대비 Brier skill≥2%; logloss 비악화; ECE≤0.03; paired≥90%.
주요 사전 regime 악화>10%는 승격 차단. 원래 target/loss/단위/표본 계약 전체를 적용하고 simple/log return을 혼용하지 않는다. prospective 비중첩 최소 표본=6h120/24h60/72h40/7d26/14d18/30d12이며 개수만으로 승격하지 않는다. 매시간 겹치는 예측은 독립 표본이 아니다. 장기 만기를 기다리되 다른 적격 연구를 막지 않는다.

## 운영·증거
Historical/development/untouched OOS/prospective shadow/actual public issuance/delivery/matured outcome을 분리한다. 누락 실발행을 소급 생성하지 않는다. 00UTC HAR와 dense24h/72h cohort를 혼합하지 않는다. job success/publication/continuity attention/optional degradation/skill/rights를 구분한다. expected slot 대비 issued·proven-timely·settlement overdue·source freshness·archive lag·restore를 관리한다. 기존 완전한7일≥99% 적시 공개 목표를 별도 평가하며 24h/누적/7일 분모를 혼용하지 않는다. 누락 receipt는 비공개의 확정 증거가 아니다.

## A0·업무분담
A0는 기존 health/operation receipt·registry·workflow를 재사용하는 event-driven control plane이다. 새 parallel orchestrator나 agent별 cron 금지. A1 Data/PIT/권한, A2 정보·사건 발견, A3 네 Head 연구, A4 regime, A5 검증·승격 제안, A6 독립 QA READ_ONLY, A7 운영·승인 공개, A8 실패학습. 논리적 역할이지 아홉 개 AI 상시 실행이 아니다. 직접 상호 호출·자동 승격·투자/매매 권한 확대 금지.
일반 ChatGPT=작은 상태/receipt/diff/CI 확인, 병목·설계·문서·packet·merge 판단. Work=다중파일 구현/복잡한 재현·장애/승인된 대형 연구 때만 수동. 상태 확인·SHA 비교·CI 대기·정상 보고에는 Work 금지. 작은 문서 PR은 일반 채팅에서 처리한다. Codex는 repository 또는 고위험 gate가 요구하는 최종 exact-head 독립검토에만 사용하되 기존 gate는 생략하지 않는다. head 변경 시 필요한 재검증을 연결한다.

## 예약·비용
사용자의 현재 일시중단 결정 유지. EtherForecast 반복 Work=0, ChatGPT 감시 활성=0. 승인 없이 만들거나 재활성화하지 않는다. 과거 문서의 ChatGPT 매시간 점검/watchdog 재실행은 현 정책이 아니다. 기존 GitHub 수집·발행·정산·보관·복구·승인 shadow/갱신은 별도이며 문서 작성만으로 중단·변경하지 않는다.
우선순위 deterministic Actions→검증 artifact 재사용→SKIP_UNCHANGED→일반 ChatGPT→Work→요구된 Codex. task/source/input/dependency/config/model/parameter/runtime/code+origin/horizon+settlement frontier가 같고 receipt·freshness가 유효할 때만 재사용한다. 만료 입력·새 만기·새 origin을 같은 hash라는 이유로 건너뛰지 않는다. 가벼운 선행검사로 불필요한 설치·동일 연구·전체복원을 줄이되 immutable receipt/보관 누락 방지/원본을 보존한다. shared writer lock을 무작정 분리하지 않는다. Work/API/Actions 비용을 분리 계측하고 근거 없이 절감액을 주장하지 않는다.

## 권한·완료·알림
T0 읽기/T1 결정론 계산/T2 검증 보고서 허용. T3 문서/PR은 repository gate 적용. T4 새 모델·threshold·공개 계약 변경은 명시 승인, T5 장부/보호 evidence·삭제/권한/실거래는 명시 승인+독립검증. 기존 승인 범위의 정기 발행은 매회 새 승인을 요구하지 않는다. 대형 fullrun/새 유료 API/노드 구매/승격은 자동 승인하지 않는다.
Task→Work→Evidence→Confidence→Next Action. receipt에 task_key,as_of,availability,source/input/dependency hashes,code/runtime/config/model/feature/target/parameter identity,origin/horizon/cohort,output hash,run/attempt,test/CI,reviewed head,side effects,approval,next/stop을 연결한다. missing/stale/conflicting/hash mismatch면 fail closed. hash/녹색CI/자기보고만으로 DONE 금지.
DOC_PREPARED/PR_OPEN/CI_PASSED/MERGED/DEPLOYED/RUNTIME_VERIFIED/PROSPECTIVE_APPROVED를 구분한다. WAITING_CI/WAITING_REVIEW/SKIP_UNCHANGED/정상반복은 silent. 의미 있는 BLOCKED/CORRECTION_REQUIRED/HUMAN_APPROVAL_REQUIRED/READY_TO_MERGE/DONE/DATA_INTEGRITY_FAILURE/UNEXPECTED_REGRESSION만 알린다. 표본 성숙은 WAITING_MATURITY이며 현재 ChatGPT 예약은 중단이다.

## 실행·Handoff
운영 H1 1건+독립 연구 H2 1건을 기본 WIP 한도로 두며 같은 PR에 섞지 않는다. 이미 병합된 refresh/isolation/receipt/archive를 재구현하지 않는다. 연구는 #95 노드 전체 또는 모든 장기 만기를 기다리지 않는다.
가설/소스 권한·PIT/Head·target/동일-origin baseline/train-calibration-holdout/지표·regime/비용상한/reject·stop을 먼저 동결한다. 한 가설·한 feature family→fixture→최소수정→focused test→승인 bounded backtest→walk-forward/OOS→prospective→독립검토/승인→production→attribution. 이미 본 역사 재튜닝·무한 모델 탐색 금지. 단순 증분 정보를 입증하기 전 Transformer 확대부터 하지 않는다.
Work packet에는 TASK_KEY,CURRENT_MAIN_SHA,WORK_NEEDED,CODEX_REQUIRED,H1/H2,ALLOWED_PATHS,FORBIDDEN_SIDE_EFFECTS,PINNED_INPUTS/HASHES,ONE_BUG/HYPOTHESIS,FIXTURES,TESTS,BUDGET,DOD,REVIEW_GATE,STOP을 넣는다.
[PROJECT_HANDOFF]에 기준시각/데이터 cutoff, 사실·추론, 실제 변경·검증 증거, 미검증 사항, 다음 한 개 작업, blocker·stop, Work/Codex 여부를 남긴다. 완료하지 않은 실행·배포·승인을 완료했다고 쓰지 않는다.

## 2026-09-27 감사 기준과 다음 작업

기준 main: ddcec49e355d3170078de09b7b6caf7de96830b5. 이번 문서 작업은 WORK_NEEDED=NO / CODEX_REQUIRED=NO. 실제 runtime 구현과 보호 evidence 변경은 포함하지 않는다.

run36285201885 attempt1의 event-automation-health artifact10920486814 ZIP SHA256를 API digest와 대조했다: 034b02d064152ad671a1065dea982112725e07a2cbf1b734a9899b243e6ecee1. continuity as_of2026-09-27T01:22:31Z 기준 각 horizon의 recent24h는 expected24/issued14/proven-timely14다. 이번 job/publication 성공과 별개로 health는 attention이다. missing receipt를 모두 실제 비공개라고 단정하지 않고, 현재24h/누적/7일 분모를 분리한다.

EF-OPS-01은 기존 operation_health/continuity/manifest를 재사용해 누락 슬롯의 run→job→input/checkpoint→inference→delivery→archive 원인을 먼저 분류한다. 동일 task 재사용은 expiry·새 origin·새 만기·source-specific completeness를 보존해야 한다. cause map 없이 writer lock을 분리하거나 cron을 일괄 끄지 않는다. 실제 다중파일 수정이 필요할 때만 별도 Work packet으로 실행한다.

EF-R4-01은 이미 적격인 데이터로 하나의 새 6h Event/정보 가설을 사전 등록한다. 기존에 본 quantile/window를 재튜닝하지 않는다. 최신 부정적 HAR prospective 기록을 무시하고 오래된 역사 성과만으로 승격하지 않는다.

#106/#108 파생자료 public-current-tree cleanup은 실제 main의 세 private stream restore/hash 증거와 명시 승인을 확인하기 전 실행하지 않는다. #95 노드 구축은 별도 예산/보안 승인 대상이며 다른 적격 연구의 전역 blocker가 아니다.

근거: [동결 검증 #56](https://github.com/wscha231/eth-dashboard/issues/56), [공식 tracker #68](https://github.com/wscha231/eth-dashboard/issues/68), [최신 전체 감사 checkpoint](https://github.com/wscha231/eth-dashboard/issues/68#issuecomment-5850443317), [health 원본 run](https://github.com/wscha231/eth-dashboard/actions/runs/36285201885).
