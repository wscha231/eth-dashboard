# EtherForecast 상태 연동 로고 — 코드 조사

## 조사 기준
- 저장소: wscha231/eth-dashboard
- 조사한 main: 4daef5ba5543de636c748843c109058744bc1b4a
- 확인한 배포 브랜치: data/daily-forecast (조사 시점의 events.js blob을 비교함)
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

아래 추가 조사를 바탕으로 사용자 승인 후 구현한다.

## 후원 비용 조사 — 2026-09-12 UTC
공식 공개 가격, USD, 세금·초과량·재배포 비용 제외. 미구매 후보 예산이며 현재 지출이 아니다.

| 후보 | 연간 가격 | 데이터 및 주의 |
|---|---:|---|
| CoinGlass Standard | $3,588 (연간 결제) | 상업용. 거래소 통합 미결제약정·펀딩·청산, ETH ETF 흐름 등 엔드포인트 후보. 1시간 이력 최대360일, 1분6일. 모든 항목의 권한·ETH 범위는 구매 전 확인. |
| CoinGecko Analyst | $1,238 (공식 표시 연간 총액) | 시장 가격·시총·거래량, 자산별 최대10년 이력. 상업용 출처표시 필요. 원시 API 접근 재판매는 별도. |
| Santiment Sanbase Max | $2,700 (개인 연간 참고 가격) | 실시간 소셜·온체인 지표, 개인플랜 제한지표2년. 공개 사이트 사용은 Business/API 및 재배포 조건 견적 필요. 합산 예산 제외. |
| Glassnode Professional + API | 견적 필요 | ETH 지원 지표별 온체인 흐름·실현가치·스테이킹 후보. Advanced $49/월을 전체 API 요금으로 오인하지 않음. API·크레딧·사용권 별도 확인. |

출처:
- https://www.coinglass.com/pricing
- https://docs.coinglass.com/reference/endpoint-overview
- https://www.coingecko.com/en/api/pricing
- https://support.coingecko.com/hc/en-us/articles/8203136845081-If-I-wish-to-change-from-one-API-plan-Analyst-Lite-Pro-to-another-how-much-do-I-get-charged
- https://app.santiment.net/pricing
- https://academy.santiment.net/products-and-plans/sanapi-plans/
- https://studio.glassnode.com/pricing
- https://docs.glassnode.com/basic-api/api

CoinGlass + CoinGecko 후보 합계 $4,826/년. 모두 구독해야 한다는 뜻이 아니며 파일럿 검증 뒤 선택한다. 코드에는 Binance/Deribit 펀딩·미결제약정 수집, CoinGecko 및 선택적 Glassnode 수집이 이미 존재한다(eth_data_collector.py). 유료라는 이유만으로 독점 데이터나 개선을 주장하지 않는다. 정제된 다거래소 이력, 시점별 저장, 안정적 수집과 추가 지표가 평가 대상이다.

유료 데이터만 추가한 대조실험 결과는 없다. 세 논문 또는 다른 데이터셋 성과를 이 사이트의 예상 상승률로 전용하지 않는다. 모든 공급자별 효과는 미측정. 제안하는 파일럿 평가 목표는 동일한 고정 기준 모델 대비 가격 MAE3% 또는 Brier5% 상대 감소이며, 기대치가 아니다. 하나의 사전 지정 지표를 선택하고 다른 지표·구간 악화를 같이 점검한다. Brier .20→.19가 상대5% 오차 감소라는 예시이며 정확도5%p 증가가 아니다. 6h/1d/3d/7d/14d/30d별 시계열 분할, 당시 이용 가능 데이터, purge/embargo, 비중첩 표본 및 불확실성 범위를 보고한다. 향상0 또는 악화도 가능하며 근거가 약하면 도입하지 않는다.

## 상단 영상 배치 추가 조사
이전 구현은 112px 인라인 SVG 상태 로고였고 제공한 MP4는 사이트 발행 자산에 포함되지 않았다. 이번 요청은 원본 기반의 실제 영상을 크게 삽입하는 것이다. 기존 MP4는 H.264, 640×700, 24fps, 5초, 무음, 90,341바이트다. 별도 영상 생성이나 유료 API가 필요 없다. video.loop와 상태 기반 play()/pause()로 구현하며, paused 프레임에 빛이 남지 않도록 정지 상태에는 원본 포스터를 덮고 어둡게 한다.
