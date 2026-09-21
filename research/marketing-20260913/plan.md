# EtherForecast 무료 마케팅 자동화 실행계획

상태: 계획 초안. 이 문서는 게시 프로그램, 새 스케줄, SNS 게시를 실행하지 않는다.
대상: https://etherforecast.live/ · wscha231/eth-dashboard
조사 기준: 2026-09-12 UTC (2026-09-13 KST), main 4daef5ba5543de636c748843c109058744bc1b4a.
목표: ETH 관심 방문자의 유입 → 예측 결과 확인 → 재방문 → 자발적 운영비 후원.
첫 30일은 전환 측정과 반복 가능한 운영 확보가 목표이며 방문자/후원 실적을 보장하지 않는다.

## 1. 확인된 상태와 부족한 것

- 소스에 Naver 인증 태그, canonical, 공유 메타정보, 공유 이미지, 한국어 안내, 개인정보처리방침과 후원 안내가 있다.
- 이번 공개 HTTP 조회에서 robots.txt 및 sitemap.xml이 200. 사이트맵은 /, /outlook.html, /ko/, /privacy.html 4개 URL을 담는다.
- signals.json은 ready와 6개 레코드를 반환했다. generated_at=2026-09-12T19:12:28.763173+00:00. 이는 조회 당시 응답이며 모델 성능이나 이후 갱신을 보장하지 않는다.
- scripts/publish_search_assets.py가 배포된 원본 signals.json으로 /outlook.html을 생성한다. 이 주소는 최신 스냅샷으로 덮어쓴다. 날짜별 영구 보고서는 별도 구현 대상이다.
- 이번 main 트리 및 홈페이지 점검에서 RSS 발행기와 명시적인 방문 분석 태그는 확인되지 않았다. 외부 호스팅 통계 또는 비공개 별도 분석 서비스의 존재는 미확인이다.
- Google DNS 토큰 공개와 Naver 태그 배포는 앞선 작업에서 확인했다. 계정 안의 소유확인 완료, 사이트맵 접수, 색인 현황은 아직 이 계획에서 확인하지 않았다.
- SNS 계정, Buffer 연결, 실방문자 수, 현재 후원 수입, 실제 월 운영비는 미확인이다.

## 2. 채널 선택과 무료 범위

| 우선순위 | 채널 | 운영 방식 | 필요한 준비 |
|---|---|---|---|
| 1 | Google·Naver | 사이트맵, 검색용 한국어/영어 보고서, 제목·내부 링크 개선 | 각 계정 소유확인·사이트맵 접수 |
| 1 | X | 짧은 영문 예측 카드와 결과 검증, 주 5회 | 브랜드 계정, Buffer 연결 |
| 2 | Instagram | 수치와 의미를 설명하는 카드, 주 3회 | 프로페셔널 계정, Buffer 연결 |
| 2 | Facebook | 페이지에 한국어 요약과 보고서 링크, 주 3회 | 브랜드 페이지, Buffer 연결 |
| 2 | Bing | 소유확인 및 사이트맵 제출 | Webmaster Tools 접근 |
| 3 | Naver Blog·커뮤니티 | 주 1회 설명글 초안, 게시 규칙에 맞춘 수동 게재 | 계정 및 개별 게시 공간 확인 |
| 3 | RSS | 일간 보고서 구독 및 검색 발견 보조 | 날짜별 보고서 발행기 |

선택: Buffer Free에 X·Instagram·Facebook 3채널을 연결한다. 공식 가격표상 채널당 예약 대기 10개이며 월 총 10개 제한은 아니다. API는 Free에서 1키, 30일 3,000요청이며 별도 일간/단기 제한도 있다. 연결된 3개 채널에서 실제 게시 형식 지원을 첫 시험 때 확인한다.
예비 경로: Meta Business Suite로 Facebook·Instagram 무료 예약 게시. 자동 생성한 파일을 주 1회 직접 올리는 단계부터 운영할 수 있다.
X 직접 API는 사용량 과금이므로 초기 예산에서 제외한다. Buffer는 자체 연결을 통해 게시하며, 사용자에게 별도 X 개발자 API 키를 준비시키지 않는다.
GitHub 공개 저장소의 표준 hosted runner 실행은 무료 범위. 저장공간·대형 runner·외부 서비스·기존 호스팅과 도메인 비용은 별도다.
초기 추가 유료 API·광고비 목표 0원. ChatGPT 이용권이 외부 LLM API 호출비를 포함한다고 가정하지 않는다.

## 3. 콘텐츠 전략

포지셔닝: “이더리움 예측과 실제 결과를 함께 공개하는 연구 대시보드.”
게시 비중 제안: 결과 검증 50%, 최신 전망 30%, 확률·오차 해설 20%.

1. 일간 전망: 하나의 발행 시점에서 1·7·30일 등 제공 가능한 구간을 함께 설명한다. 발행 시각·목표 종료 시각·기준 가격·중앙값·예측 구간·확률 정의를 표시한다.
2. 결과 검증: 과거 실제 공개 기록과 종료 후 실제값을 대응시킨다. 맞힌 사례와 틀린 사례를 모두 포함한다.
3. 주간 리포트: 만기 도래 표본 수, 예측 오차, 구간 포함률, 단순 가격 유지 기준모델 비교. 실시간 공개 실적과 과거 백테스트를 구분한다.
4. 교육글: “80% 예측 구간의 의미”, “상승 확률과 정확도가 다른 이유”, “30일 예측이 어려운 이유”, “급등을 놓친 뒤 무엇을 점검했나”.
5. 업데이트 안내: 기능 변경이 실제 배포된 경우에만 소개한다.

예측 레코드의 lower/middle/higher는 모델 장벽 대비 최종 가격 분류다. upper 확률을 설명 없이 일반적인 상승 확률 또는 적중률로 바꾸지 않는다.
순위 조작을 위한 유사 AI 글 대량 생성은 하지 않는다. 숫자만 바뀌는 매시간 페이지 대신 하루 1개 유용한 보고서와 주간 분석을 만든다.
관련 검색어는 후보로만 사용: 이더리움 예측, ETH 전망, 예측 정확도, Ethereum forecast accuracy. 검색량·난이도는 아직 측정하지 않았다.

## 4. 자동화 작업 흐름

배포 성공 확인 → 원본 공개 데이터 고정 → 유효성 검사 → 날짜별 한·영 보고서 → RSS·사이트맵 → SNS 문안·카드 → 게시 대기열 → 발행 결과 기록 → 주간 개선.

### A. 데이터와 발행 기준
- 마케팅 작업은 배포 검증에 성공한 release_id 및 forecast_id를 참조한다. 별도 모델 실행·재학습은 하지 않는다.
- 기존 readiness, timestamp, 확률합, 가격 분위수 검증을 재사용한다. 첫 구현에서는 현재 100분 freshness 기준과 원본 상태를 존중하고 향후 모델 운영 변경을 따라간다.
- 원본 발행시점이 오래되면 최신 전망 게시를 건너뛴다. 미리 검증된 교육글은 별개 콘텐츠로 발행 가능하다.
- 만기 전 예측에 적중/실패 판정을 붙이지 않는다. 확정된 실제 관측값이 없으면 보류한다.
- 백테스트는 당시 공개 예측처럼 표현하지 않는다. 겹치는 30일 표본을 독립 표본 수로 과장하지 않는다.

### B. 날짜별 보고서와 RSS
- 제안 URL: /reports/YYYY-MM-DD/ko/ 및 /reports/YYYY-MM-DD/en/. 아직 생성된 주소가 아니다.
- 하루 대표 발행시점과 포함 구간을 명시. 최초 발행 기록은 유지하고 결과 추가·정정은 별도 updated_at과 변경 이력을 둔다.
- 날짜만 다른 빈약한 페이지 생성을 억제한다. 유효 데이터가 없으면 새 예측 보고서를 발행하지 않는다.
- /rss.xml 및 /rss-ko.xml, 홈페이지 RSS autodiscovery, 영구 GUID, 실제 발행일과 본문 포함. 최근 30개 보고서 유지.
- 사이트맵에 접근 가능한 canonical 보고서를 추가하고 실제 변경만 lastmod에 기록한다.
- 사이트맵과 RSS는 검색 발견 보조다. 일반 콘텐츠에 Google Indexing API를 오용하거나 동일 URL 색인 요청을 반복하지 않는다.

### C. 문안·이미지
- 최초는 Python 템플릿으로 정확한 원본 수치와 제한 문구를 조합하여 유료 LLM 없이 생성한다.
- 영어 X 짧은 요약, Instagram 3장 카드(전망/불확실성/결과 링크), Facebook 한국어 설명을 같은 report_id에서 생성한다.
- 수치 그래프는 실제 데이터로 렌더링한다. AI로 그린 가짜 가격 차트는 사용하지 않는다.
- PNG 1200×630 공유용, 1080×1350 카드용을 출발 규격으로 삼고 각 채널 업로드 시험을 통과시킨다.
- 이미지 URL은 공개 HTTPS에서 게시 시점까지 유지한다. 이미지에 날짜·출처 링크·기준 구간을 표시한다.
- 시의성이 없는 교육글만 미리 예약한다. 최신 전망은 게시 직전에 새로 검증하고 7일 뒤 게시될 대기열에 넣지 않는다.

### D. 게시와 실패 처리
- 첫 단계는 완성된 샘플 묶음을 검토할 수 있게 만든다. 계정·게시 범위가 확정되면 그 범위의 소유 채널에 자동 발행한다.
- report_id + channel_id + content_type으로 중복 방지. 요청 결과가 불명확하면 원격 게시 상태를 조회하고 재전송한다.
- 예상 상태: generated → validated → queued → published / failed / skipped.
- 매일 게시 결과와 URL을 저장한다. 연결 만료, HTTP 429, 미디어 접근 실패 시 제한된 재시도 후 알린다.
- 채널당 주간 대기열은 7개 이하 목표. 마케팅 요청은 일 30회 이하 목표로 시작하고 무료 API 제한을 넘기지 않는다.
- 예측 배포 실패로 과거 카드가 오늘 카드로 나가지 않도록 신규 전망 발행을 차단한다.
- 게시 채널·빈도 변경 및 유료 결제는 별도 범위로 다룬다. 불특정 사용자 DM, 자동 댓글, 커뮤니티 대량 홍보는 구현 대상이 아니다.

### E. 운영 구조
- 기존 예측 프로세스를 막지 않는 별도 marketing 실행기로 구성한다.
- 사이트 파일 배포는 기존 production writer/concurrency 규칙을 사용한다. SNS 발행 큐는 별도 concurrency를 사용한다.
- SNS 인증·게시 실패가 예측 데이터 발행을 실패 처리하지 않도록 분리한다.
- 비밀값은 GitHub Actions Secrets에만 저장한다. 공개 로그에는 토큰을 출력하지 않는다.
- 대용량 PNG를 매시간 Git history에 누적하지 않는다. 일간 카드만 저장하고 용량을 측정해 필요할 때 저장정책을 바꾼다.
- ChatGPT 정기 점검은 전략·예외 검토에 활용할 수 있지만 이 계획 작성 자체로 예약 작업이 생성되는 것은 아니다.

## 5. 측정과 후원 전환

GA4를 첫 무료 측정 후보로 사용한다. 기존 별도 analytics가 있다면 중복 설치하지 않는다. 추적 도입 시 실제 수집 내용에 맞춰 기존 개인정보처리방침 및 필요한 동의 처리를 검토한다.

이벤트 후보:
- view_report: 보고서 열람
- view_accuracy: 결과 검증 영역 열람
- support_click: 후원 버튼
- copy_wallet: 지갑 주소 복사 성공
- rss_click: 피드 링크 이동 (실제 구독 완료와 구분)

모든 외부 게시 링크에 utm_source, utm_medium, utm_campaign, utm_content를 붙인다. Instagram 프로필 링크의 UTM은 개별 카드 단위 기여도를 확정하지 못한다.
주소 복사는 후원 완료가 아니다. 후원 입금은 거래 hash, 체인, 금액, 확인 상태로 별도 집계하고 자기이체·스팸 토큰을 제외한다. 공개 지갑 거래만으로 방문자 신원이나 SNS 게시물별 기여도를 확정하지 않는다.
원석님이 월 실제 운영비를 제공하면 “월 운영비 중 얼마나 충당됐는지” 표시를 검토한다. 수입이나 목표 금액을 임의로 만들지 않는다. 네트워크·자산을 명확히 표시하고 기존 지갑 주소는 유지한다.

주간 대시보드:
검색 노출/클릭, 채널별 사이트 세션, 보고서→검증 페이지 이동률, 재방문, 후원 버튼 클릭률, 주소 복사, 확인된 후원 건수/금액, 제작시간과 추가비용.
Buffer 공식 도움말상 API analytics는 아직 실험적이다. KPI는 GA4/검색엔진 보고서와 원본 입금 집계에 기반하고 SNS 도달·저장·좋아요는 초기에는 주 1회 화면 또는 export로 확인한다.
GA4 이벤트는 집계 지표이며 방문자 wallet address·개인정보를 이벤트 매개변수로 보내지 않는다.

주간 개선 원칙:
- 높은 노출·낮은 클릭: 제목, 첫 이미지, 링크 설명을 하나씩 바꾼다.
- 방문은 있지만 검증/재방문 부족: 보고서 이해도와 모바일 동선을 바꾼다.
- 후원 클릭은 있지만 입금 부족: 네트워크 안내·운영비 설명·오류 여부를 확인한다.
- 단일 게시물 성공이나 작은 표본으로 승자를 선언하지 않는다. 비슷한 주제·시간대를 비교하고 표본 부족을 표시한다.
- 첫 2주 기준선을 확보한 뒤 다음 2주 가설을 정한다. 팔로워 구매·성과 수치 조작은 하지 않는다.

## 6. 30일 실행 순서

| 단계 | 작업 | 완료 판단 |
|---|---|---|
| 1주 | 검색 등록 상태 확인, 계정 연결, 이벤트 설계·설치, 보고서/RSS/카드 생성기 | 실제 보고서 1개와 각 채널 샘플 1개, 수치·링크 검증 |
| 2주 | 무료 Buffer 연결 시험, 게시 결과 기록, X 주5·Instagram/Facebook 주3 시작 | 중복·오래된 카드 없이 발행, 방문 유입 기록 |
| 3주 | 주간 검증 보고서, 교육글 2개, 댓글 질문을 FAQ로 반영 | 예측 원본과 실제값 추적, FAQ 개선 |
| 4주 | 채널/주제별 비교, 후원 동선 개선, 다음 달 빈도 재배분 | 실제 성과 보고서와 다음 실험 2개 |

제안 운영 시간(Asia/Seoul, 아직 예약되지 않음):
매일 09:00 상태·배포 확인, 최신 자료 확보 뒤 보고서 생성.
월~금 21:00 X 영문 요약을 출발점으로 시험.
월/수/금 20:30 Instagram·Facebook 카드/요약.
일요일 20:00 주간 결과 검증과 다음 주 교육글 편집.
시각은 최적 시간이라는 주장이 아니며 실제 유입으로 조정한다.
SNS 최신 전망을 게시할 때는 오전 보고서가 아닌 게시 직전 검증된 예측인지 다시 확인하거나 명시적으로 오전 시점의 기록이라고 표시한다.

## 7. 원석님이 준비할 항목과 연결 절차

### 필수
1. Google·Naver: 소유확인 완료 및 사이트맵 제출 상태를 화면 또는 결과로 공유. 계정 인증 비밀번호는 공유하지 않는다.
2. X 브랜드 계정 URL, Instagram 프로페셔널 계정 URL, Facebook 페이지 URL. 기존 계정이 있으면 우선 사용하고 이름은 EtherForecast로 통일 권장.
3. Buffer Free 가입 → Channels에서 3계정 연결 → 각 플랫폼의 공식 로그인 화면에서 권한 부여.
4. Buffer Settings의 API에서 personal API key 발급 → GitHub 저장소 Settings → Secrets and variables → Actions → New repository secret에 BUFFER_API_KEY 저장. 값은 채팅·코드·문서에 붙이지 않는다. 연결 후 조직/채널 ID는 읽기 전용 조회로 확인한다.
5. GA4를 선택하면 Google Analytics에서 계정/속성 → 웹 데이터 스트림에 사이트 등록 → G-로 시작하는 측정 ID 공유. 측정 ID는 공개 태그 값이다. 보고서 API 접근은 별도의 읽기 권한 설정이 필요하며 태그 설치만으로 생기지 않는다.
6. 실제 월 운영비와 후원 안내 언어. 카드/간편결제 추가는 원하는 경우에만 별도 검토한다.

### 후속
- Bing Webmaster Tools에서 Google 속성 가져오기 또는 소유확인 후 사이트맵 제출.
- Naver Blog 계정이 있으면 URL 제공. 자동 게시를 약속하지 않고 초안과 카드 공급부터 시작한다.
- 게시 소유 채널과 주간 발행 범위 확정. 외부 커뮤니티는 실제 규칙과 해당 공간의 게시 허용 여부 확인 후 개별 접근한다.
- RSS 발행 후 Naver의 요청→RSS 제출 및 원하는 RSS 리더에 연결. 검색을 위한 RSS는 필수가 아니다.

## 8. 구현 파일 제안 및 인수 조건

제안 파일 (아직 미구현):
- scripts/build_marketing_report.py
- scripts/render_marketing_cards.py
- scripts/publish_marketing_posts.py
- .github/workflows/marketing.yml
- marketing/config.json, marketing/content_calendar.csv, marketing/publication_log.jsonl
- forecast_site/public/reports/, rss.xml, rss-ko.xml
- tests/test_marketing_publication.py (오래된 자료, 중복, 만기 전 결과, 재시도 등 실제 위험 검증)

필수 인수 조건:
원본 수치·발행시각 일치, 한국어/영어 의미 일치, XML 파싱, 영구 URL 200, canonical·내부 링크, 1개 채널 실게시 후 원격 URL 확인, duplicate 재실행 방지, token 오류 시 안전한 보류, 기존 forecast pipeline 영향 없음, 측정 이벤트 실제 수신.
첫 실행은 초안·미리보기 결과를 확인한 다음 확정된 채널 범위만 활성화한다. 이번 요청은 계획 수립이므로 실행이나 활성화를 완료한 것으로 보고하지 않는다.

## 공식 근거

- Buffer Free 가격·예약 제한: https://buffer.com/pricing
- Buffer API 시작/무료 요청 한도: https://buffer.com/api
- Buffer API 지원 채널, 이미지 URL, analytics 한계: https://support.buffer.com/en-us/articles/using-buffers-api-GtIYIQilz5
- Meta Business Suite: https://www.facebook.com/business/tools/meta-business-suite
- X 직접 API 과금: https://docs.x.com/x-api/getting-started/pricing
- GitHub Actions 무료 범위: https://docs.github.com/en/billing/concepts/product-billing/github-actions
- Google sitemap/RSS: https://developers.google.com/search/docs/crawling-indexing/sitemaps/build-sitemap?hl=ko
- Naver sitemap/RSS: https://searchadvisor.naver.com/guide/request-feed
- Google scaled content abuse: https://developers.google.com/search/docs/essentials/spam-policies
- GA4: https://support.google.com/analytics/answer/10089681?hl=ko
