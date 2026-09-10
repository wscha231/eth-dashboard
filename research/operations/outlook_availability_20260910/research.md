# ETH 전 구간 예측 비표시 장애 조사

상태: 기존 운영 작업 재실행으로 긴급 복구 완료. 재발 방지 코드는 아직 변경하지 않았다.
조사일: 2026-09-10 UTC / 2026-09-11 KST. 기준 코드: baf022d907512d3fdd79c819616095476c6e10b0.

## 확인한 현상

6시간·1일·3일·7일·14일·30일 화면이 모두 “outlook unavailable”을 표시했다.
배포된 signals.json의 current 배열이 비어 있었으며, 이전 실제 발행 예측은 recent_issued와 별도 원장에 남아 있었다.

| 증거 | 값 |
|---|---|
| 장애 배포 커밋 | 4cdce70085f603f6e3306963fc52ddc59ae44247 |
| 장애 release_id | 178bec029732c15f4d8ff5599b9e1606e40b8720c51d227e82629a415caf0a24 |
| 장애 입력 기준 / 생성 시각 | 2026-09-10 14:00 UTC / 14:55:33 UTC |
| 수집 상태 | ETH·BTC 모두 14:00 UTC 확정봉 확보, source.ready=true, errors=[] |
| 예측 상태 | status=delayed, current=[], 6개 구간 모두 발행 거부 |
| 공통 거부 사유 | only the current hourly slot can be actually issued |
| 직전 정상 발행 | 13:00 UTC 기준, 13:33:16 UTC 공개 확인 |
| 복구 배포 커밋 | 5bacd7e5d772cfad3f92c3db3f162d1b1ce0a1bd |
| 복구 입력 기준 / 생성 시각 | 15:00 UTC / 15:30:47 UTC |
| 복구 사이트 검증 | 15:31:36 UTC = 9월 11일 00:31:36 KST, ready, horizons=6 |

## 실제 원인과 실행 순서

1. Watchdog가 14:54:45 UTC에 복구 작업을 요청했다.
2. 환경 준비·상태 복원·수집 후 실제 추론은 14:55:33 UTC에 실행됐다.
3. signal_pipeline/ledger.py의 validate_forecast는 매시 55분 이후 신규 발행을 거부한다. 아직 관측하지 않은 다음 시간부터 평가창을 시작하고 공개까지 여유를 두기 위한 기존 제한이다.
4. signal_pipeline/engine.py의 daily는 구간별 ValueError를 errors에 모으고 current=[] 및 delayed를 작성한다.
5. scripts/run_event_forecast.py는 delayed 결과를 출력해도 정상 종료한다.
6. scripts/publish_events.sh는 이 빈 결과를 data/daily-forecast에 배포한다. 예측 실패 사실을 정직하게 전달하는 부분이지만, 화면이 이를 처리할 때 직전 유효 발행값을 보여주지 않는다.
7. scripts/verify_event_site.py는 --require-ready 없이 실행되면 지연 상태의 파일도 정확히 배포됐는지만 확인한다. 이 경로의 로그에도 “Event site verified”라는 동일 표현을 사용하여 예측 정상 여부와 혼동하기 쉽다.
8. forecast_site/public/events.js의 renderCards는 payload.current에서만 선택 구간을 검색한다. recent_issued에 공개 확인된 만기 전 기록이 있어도 상단 카드에서는 모두 사라진다.

따라서 이번 장애는 전 구간 모델이 계산 불능이 된 증거가 아니다. 공통 발행 마감과 배포·표시 처리의 결합으로 발생한 운영 장애다.

## 세 가지 대조 검증

### 배포 데이터와 Actions 기록

장애 작업은 GitHub에서 success였지만, 실제 사이트 확인 로그는 다음과 같다.

    Event site verified: ... delayed slot=2026-09-10T14:00:00+00:00 ... horizons=0

Watchdog는 이를 뒤이어 실패로 판정했다. Watchdog가 정상적인 지연 판정을 한 것과, 발행 작업이 성공으로 표시된 것은 서로 다른 검증 경로다.

### 현재 코드의 시간·검증 규칙 재현

운영 원장이나 모델을 수정하지 않고 기존 함수를 직접 실행했다.

| 확인 | 결과 |
|---|---|
| 같은 신규 예측을 14:54:59에 검사 | 허용 |
| 같은 신규 예측을 14:55:00에 검사 | 발행 거부 |
| 실제 장애 payload를 기본 verify로 검사 | True |
| 같은 payload를 require_ready=True로 검사 | hourly forecast delayed |
| 복구 요청 코드, 14:54:45, 다른 작업 없음 | dispatched |
| 복구 요청 코드, 14:55:00, 다른 작업 없음 | issuance_deadline |

복구 요청 제한은 요청 시각만 본다. 실행 준비에 필요한 시간이 반영되지 않아 요청 당시 허용돼도 추론 시각에는 마감을 넘을 수 있다.

### 배포 데이터로 실제 화면 함수 실행

Node VM의 최소 DOM 환경에 현재 events.js와 실제 장애 payload를 넣고 6개 선택 구간을 순서대로 전환했다. 모두 사용자와 같은 “outlook unavailable” 문구를 재현했다.
각 구간의 직전 기록은 공개 확인 시각이 관측창 시작 전이었고, 당시 목표 종료 전이었다.

복구 payload로 같은 화면 함수를 실행하면 6개 모두 가격 카드와 Published forecast를 출력하고 Up to date 상태가 된다.
이는 실제 JS 함수 실행 검증이다. 사용자의 휴대전화 브라우저를 직접 열어 촬영한 검증은 아니다.

## 이번에 수행한 복구

기존 Watchdog 실행 34476590030의 실패 작업만 재실행했다. 이 작업은 중복 실행·공유 작업 잠금·재시도 간격을 검사한 뒤 신규 hourly 실행 34496028599를 요청했다.
새로운 코드를 작성하거나 모델을 재학습하지 않았다.

새 작업 로그에서 다음을 확인했다.

- 공개 HTML 및 events.js가 main 소스와 동일한 해시임을 확인.
- 공개 signals.json의 release_id 및 개별 예측 ID가 생성 결과와 동일함을 확인.
- ready, UTC 15:00 기준, 정확히 6개 구간.
- 기존 replay의 6개 구간 및 2,256개 구간별 CSV 행 일치.
- 15:33:34 UTC에 복구 payload를 현재 시각의 엄격한 require_ready 검사로 다시 검증.
- 과거 발행 예측의 가격·시각·모델 및 관측창은 변경하지 않음.

복구 release_id: fc42abb9401fdc3c1a5af9f206f95ba27573aa79e1b8689e8cf911a5d41d85a6.

## 제한과 남은 문제

- 이번 복구는 현재 실행을 복원한 조치이며 시간별 지속 가용성을 보장하는 영구 수정은 아니다.
- cron 지연과 여러 데이터 작성 작업의 공유 concurrency는 코드에서 확인했다. 이번에 특정 예약 실행이 늦거나 없어진 정확한 GitHub 내부 이유까지 확인한 것은 아니다.
- 검사 환경에서 운영 사이트 직접 HTTP 조회가 502를 반환했다. 공개 사이트 확인은 GitHub runner의 실제 외부 검증 로그와 배포 데이터·소스 해시를 대조한 결과다.
- 기존 pytest 파일을 실행하려 했으나 로컬 의존성 pluggy가 없어 시작하지 못했다. 기존 전체 테스트 통과로 보고하지 않는다. 위 함수·화면 재현 검증은 독립 실행했고 통과했다.
- 모델 우위나 예측 정확도는 이번 가용성 조사에서 평가하지 않았다. 과거 빈도 기준모형(climatology)의 선택은 별도 품질 문제이며, current가 빈 원인과 다르다.
- 첨부 논문은 이번 발행 마감·화면 장애의 원인 규명에 필요하지 않아 새로 분석하지 않았다.

## 근거

- [장애 배포 파일](https://github.com/wscha231/eth-dashboard/blob/4cdce70085f603f6e3306963fc52ddc59ae44247/forecast_site/public/signals.json)
- [장애 발행 작업](https://github.com/wscha231/eth-dashboard/actions/runs/34492172764)
- [복구를 요청한 Watchdog](https://github.com/wscha231/eth-dashboard/actions/runs/34476590030/attempts/5)
- [실제 복구 및 외부 사이트 검증](https://github.com/wscha231/eth-dashboard/actions/runs/34496028599)
- [복구 배포 파일](https://github.com/wscha231/eth-dashboard/blob/5bacd7e5d772cfad3f92c3db3f162d1b1ce0a1bd/forecast_site/public/signals.json)
- [재발 방지 구현 계획](plan.md)
