# EtherForecast 상태 연동 로고 — 코드 조사

## 조사 기준
- 저장소: wscha231/eth-dashboard
- 조사한 main: 4daef5ba5543de636c748843c109058744bc1b4a
- 확인한 배포 브랜치: data/daily-forecast / undefined
- main과 배포 브랜치의 events.js blob SHA 동일: 92f683452dfdc77de7b42a1b1c59e1d4e1e95488. 배포 브랜치 확인은 공개 사이트 응답 검증과 다름.
- 전체 Git tree에 AGENTS.md 없음.

## 실제 상태 판정
forecast_site/public/events.js의 delayed()와 renderStatus()가 주 화면의 최신 예측 상태를 담당한다. 영어 상태는 Up to date / Update delayed / Connection interrupted / Forecast unavailable이다. 한국어 문자열 '예측운영중'을 직접 읽는 방식은 현재 구현과 맞지 않는다.

정상 판정은 signals.json의 status === ready 하나만으로 충분하지 않다. expected_slot, generated_at, 6개 구간(6/24/72/168/336/720시간)의 존재·중복 여부, forecast_id, 발행 시각·입력 기준 시각·목표 시각, 가격·확률 유효성 등을 delayed()/validForecast()에서 검증한다. 이 검증을 그대로 재사용해야 로고와 상태 문구가 불일치하지 않는다.

signals.json은 60초마다 cache:no-store로 다시 요청하고 renderStatus는 30초마다 만료를 재평가한다. 과거 연구 결과의 로딩 실패(replayFailed)는 최신 예측의 연결 실패(fetchFailed)와 분리되어 있다. 예측 성능의 양호 여부도 운영 여부와 분리되어 있어야 한다.

현재 별도의 명시적 유지보수 상태 스위치는 확인되지 않았다. '오래된 예측'이나 '연결 실패'를 곧바로 '유지보수'라고 표시하면 원인을 잘못 전달한다.

## 로고와 배포 경로
forecast_site/public/index.html 헤더에는 28×28 CSS 그라데이션 사각형 .mark만 있다. 이번에 만든 원본 기반 SVG는 약 140KB이며 검정 바탕의 래스터 원본과 SVG 효과가 함께 들어 있다. 단순 animation-play-state:paused만 적용하면 원본의 밝은 구름·회로와 멈춘 발광이 남는다. 소등에는 효과 숨김 및 바탕 이미지 밝기/채도 저하가 함께 필요하다.

scripts/publish_events.sh는 index.html, events.js 등 명시된 목록을 data/daily-forecast로 복사·stage한다. 새 로고/CSS/상태 파일을 이 목록에 추가하지 않으면 main에만 있고 실제 배포에는 빠질 수 있다. scripts/publish_hybrid.sh도 index.html과 events.js를 별도로 복사하므로 동일한 자산 계약을 맞춰야 한다. daily_forecast.yml은 현재 가격/이력/health 등 데이터 중심으로 갱신한다.

forecast_site/vercel.json은 data/daily-forecast만 자동 배포 대상으로 삼는다. scripts/deployment_policy.py와 ops/deployment_cooldown.json에 공급자 배포 제한이 기록되어 있다. 기록상 배포 재개 가능 시각은 2026-09-13T20:10:00Z(한국시간 9월 14일 05:10)이다. 이 정책을 우회하거나 유지보수 UI와 동일시하면 안 된다. 실제 적용 단계에 정책과 서버 상태를 다시 확인한다.

## 기존 산출물 검증 한계
기존 5초 MP4/GIF는 프레임 렌더링과 루프 끝점 비교를 통과했다. 브라우저 실행은 브라우저 다운로드 실패로 검증하지 못했다. 상태 연동 구현 시 실제 브라우저의 전환·정지·접근성 검증이 필요하다.

이번 변경은 조사·계획 문서만 작성하며 애플리케이션 코드 및 배포 설정은 수정하지 않는다.
