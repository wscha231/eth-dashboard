# EtherForecast 상태 연동 로고 — 구현 계획

상태: 사용자가 구현을 승인함(2026-09-12 UTC). 상단 로고·운영 상태와 지갑 위 유료 데이터 비용·후원 이유를 함께 구현한다.

## 목표와 상태 매핑
원본의 구름·비·청록색 회로를 유지하고 사이트의 실제 운영 상태와 동일한 판단으로 동작한다. 웹용은 조절 가능한 인라인 SVG를 사용한다.

| 우선순위/상태 | 판정 | 로고 연출 | 상태 문구 |
|---|---|---|---|
| 1 유지보수 | 유효한 site_status.json의 mode=maintenance | 비·신호·반사광 정지 및 숨김, 밝기 약 30%·채도 감소로 소등 | Maintenance / 유지보수 중 |
| 2 확인 중·미확인 | 최초 로딩 또는 상태 설정 확인 불가 | 소등·정지 | Checking status / 상태 확인 중 |
| 3 연결 실패 | 예측 요청 실패 | 소등·정지 | Connection interrupted / 연결 확인 필요 |
| 4 지연·예측 없음 | payload 없음 또는 기존 delayed()=true | 소등·정지, 지연 상태 배지 | 기존 지연/사용 불가 문구 |
| 5 예측 운영 | mode=auto, 기존 delayed()=false, fetchFailed=false | 비·회로·테두리 신호 반복 | Forecast operating / 예측 운영 중 |

'예측 운영 중'은 현재 유효한 예측이 제공됨을 뜻한다. 추론 프로세스가 지금 GPU에서 계산 중이라는 뜻이나 예측 정확도가 검증되었다는 뜻으로 쓰지 않는다. 부분 구간만 정상인 경우 기존 전체 상태 기준을 그대로 따른다.

## 구현 변경 파일
1. forecast_site/public/index.html: 기존 .mark를 접근 가능한 로고 컨테이너로 교체. 헤더는 약 48px 높이로 제한하고 상태 영역에 약 96px 로고를 배치하여 회로 움직임을 볼 수 있게 구성. 같은 DOM에 중복된 SVG id를 넣지 않는다. 기본 상태는 소등.
2. forecast_site/public/brand-status.css: 기존 사이트와 충돌하지 않는 .ef-brand 범위의 CSS, operating/off 상태, 0.4초 소등 전환, 반응형 크기, prefers-reduced-motion 지원.
3. forecast_site/public/brand-status.js: 원본 기반 SVG 로딩·마운트와 표시 제어. 준비 전·실패 시 정적인 소등 대체 표시. fetch 결과나 DOM 상태 문구를 관찰해서 상태를 추측하지 않고 events.js가 결정한 구조화 상태를 입력받는다.
4. forecast_site/public/assets/etherforecast-logo.svg: 승인받은 원본 기반 도안, 비·회로·조명 그룹을 분리. 고유 id 접두사. 외부 실행 코드 및 외부 리소스 없음. SVG를 인라인으로 주입할 때 동일 출처의 고정 자산만 사용.
5. forecast_site/public/events.js: 기존 검증 함수는 유지. renderStatus에서 하나의 상태 객체를 만들고 상태 배지와 로고에 함께 적용. status 설정은 작은 독립 요청으로 60초마다 확인; 큰 replay 파일 때문에 유지보수 반영이 대기하지 않게 한다. 이전 유지보수 설정 확인 후 통신 실패하면 소등을 유지한다.
6. forecast_site/public/site_status.json: schema_version=1, mode=auto 또는 maintenance, message(선택). 기본 auto. message는 textContent로만 출력. 잘못된 설정/404/요청 실패는 운영 정상으로 강제하지 않는다.
7. scripts/publish_events.sh 및 scripts/publish_hybrid.sh: 새 로고·CSS·JS·상태 설정을 복사하고 stage. 공통 작은 자산 복사 함수를 도입해 두 경로에서 빠짐없이 유지. hourly 생성기가 유지보수 설정을 임의 초기화하지 않도록 main의 설정을 기준으로 배포.
8. forecast_site/vercel.json: site_status.json에 짧은 재검증 캐시 정책. 기존 deployment_policy.py 제한 유지.
9. forecast_site/README.md 및 상태 관련 테스트: 유지보수 진입/해제 절차와 실제 반영 확인 방법 기록.

## 상태 설정 예시
유지보수 시작:
```json
{"schema_version":1,"mode":"maintenance","message":"Forecast service maintenance"}
```
해제:
```json
{"schema_version":1,"mode":"auto","message":""}
```
해제해도 예측 데이터가 오래되었으면 소등을 유지한다. 설정 변경은 정적 사이트 배포가 성공한 뒤에 공개 URL에 반영된다. 브라우저의 60초 폴링은 '배포 후 반영 주기'이며 저장소 수정만으로 즉시 적용된다고 설명하지 않는다.

## 제어 개념
```js
// 계획용 의사코드. 실제 코드는 승인 후 기존 renderStatus에 결합한다.
const state = maintenance ? 'maintenance'
  : !statusConfigConfirmed ? 'unknown'
  : fetchFailed ? 'connection_error'
  : !payload ? 'unavailable'
  : delayed() ? 'delayed'
  : 'operating';
setForecastStatus(state); // 배지와 로고의 단일 입력
```

운영 상태여도 탭이 숨겨졌거나 동작 줄이기가 설정되면 움직임만 정지하고 점등과 '운영 중' 표시는 유지한다. 유지보수는 움직임 정지뿐 아니라 발광을 제거하고 로고 바탕을 어둡게 한다. 처음부터 잘못 켜졌다 꺼지는 현상을 피한다.

## 검증과 완료 기준
- 정상→유지보수→정상, 정상→시간 경과 지연, 정상→연결 실패, 6개 중 1개 구간 누락, 초기 요청 실패를 고정 시각 데이터로 검증.
- mode=auto만으로 비정상 데이터를 정상으로 승격하지 않음 확인.
- 유지보수 조회가 실패해도 켜지지 않음; 정상 상태 복구는 설정과 유효한 예측이 함께 확인된 후에만 가능.
- 320px 모바일/데스크톱 크기, 원본 로고 보존, 소등 시 발광 잔상 없음, 브라우저 애니메이션 정지, 동작 줄이기·탭 숨김 확인.
- hourly/hybrid 배포 결과에 모든 신규 자산이 존재하는지 검사. forecast 입력·이력·확률 계산 결과는 변경하지 않음.
- 적용 직전 최신 main과 충돌 확인. 기존 공급자 cooldown을 존중하고 허용된 배포 경로만 사용.
- 실제 웹사이트의 상태 응답·로고 재생·소등·콘솔 오류를 확인한 뒤에만 '사이트 적용 완료'라고 보고.

## 대안과 선택
MP4/GIF 교체 방식은 소등 상태와 접근성 제어를 따로 관리해야 하고 상태 전환 시 파일 교체가 필요하다. 이번 요구에는 동일 SVG 내부의 빛과 움직임을 직접 제어하는 방식이 적합하다. 별도 WebGL/유료 영상 API/새 서버는 도입하지 않는다.

## 승인 이후 추가 범위
- 원본 정적 로고를 헤더에, 상태 연동 로고를 헤더 바로 아래에 배치. 상태 문구와 마지막 입력 시각을 한곳에서 표시한다.
- 지갑 바로 위에 공식 API 요금·연간 예산·데이터와 활용 가설을 명시한다. 유료 데이터 도입 및 성능 향상은 검증 전이라고 명확히 표시한다. 수치는 성과 약속이 아닌 사전 파일럿 평가 목표로 제시한다.
- 기존 publish_search_assets.py의 공통 자산 목록에 새 자산을 추가하고 hybrid에서도 같은 발행 함수를 사용한다. 이벤트 발행은 이미 이 함수를 호출한다. 중복 복사 함수를 새로 만들지 않는다.
- Vercel 배포 제한 해제 전에는 실제 사이트 적용 완료로 보고하지 않는다.
