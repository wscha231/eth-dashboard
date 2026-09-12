# 무료 검색 등록과 SNS 운영 절차

코드 배포와 검색엔진 계정 등록은 별도입니다. 검색엔진 등록은 소유권 인증 뒤 완료되며, 등록과 사이트맵 접수가 검색 노출·상위 순위를 보장하지는 않습니다.

## 공통 주소

- 서비스: https://etherforecast.live/
- 한국어 안내: https://etherforecast.live/ko/
- 공개 시점별 예측 요약: https://etherforecast.live/outlook.html
- 사이트맵: https://etherforecast.live/sitemap.xml
- 검색로봇 안내: https://etherforecast.live/robots.txt
- SNS 대표 이미지: https://etherforecast.live/social-preview.png

## 1. 구글 — HTML 태그 방식이 가장 적은 작업

1. https://search.google.com/search-console 에 원석님 구글 계정으로 로그인.
2. 속성 추가 → `URL 접두어` 선택. `https://etherforecast.live/` 입력. `도메인` 방식은 DNS 인증이 필요하므로 이번에는 URL 접두어가 간단합니다.
3. 소유권 확인 화면에서 다른 확인 방법 → `HTML 태그` 열기.
4. `google-site-verification`이 들어 있는 `<meta ...>` 한 줄을 복사해 이 채팅에 전달. 이것은 홈페이지에 공개되는 사이트 소유권 확인 태그입니다. 로그인 비밀번호·인증번호·계정 비밀키는 보내지 않습니다.
5. 담당자가 `forecast_site/public/index.html`의 `<head>`에 해당 태그를 원문 그대로 반영하고 배포. 원석님은 Search Console로 돌아와 `확인` 클릭. 인증 태그는 삭제하지 않습니다.
6. 사이트맵 메뉴 → 새 사이트맵 추가 → `sitemap.xml` 입력(접두어가 없는 입력창이면 전체 주소) → 제출.
7. 상단 URL 검사 → 홈페이지 주소 입력 → 필요하면 `실제 URL 테스트` → `색인 생성 요청`. `/ko/`, `/outlook.html`도 같은 순서로 요청.
8. 이후 페이지 색인 생성과 검색 실적 보고서에서 실제 노출·클릭을 확인. `site:` 검색 결과만으로 전체 색인 상태를 판단하지 않습니다.

선택: 나중에 DNS TXT 인증으로 도메인 속성을 추가하면 http/https·서브도메인을 함께 관리할 수 있습니다. DNS 제공자는 현재 네임서버를 확인한 뒤 선택해야 하며, Vercel 호스팅이라는 이유만으로 DNS도 Vercel이라고 단정하지 않습니다.

공식 문서: https://support.google.com/webmasters/answer/9008080?hl=ko

## 2. 네이버

1. https://searchadvisor.naver.com/ → 네이버 로그인 → 웹마스터 도구.
2. 사이트 관리 → 사이트 등록에 `https://etherforecast.live` 입력.
3. 사이트 소유확인 → `HTML 태그` 선택 → `naver-site-verification` 태그 한 줄을 이 채팅에 전달.
4. 홈페이지 배포 완료 후 소유확인 클릭. 화면에서 별도 보안 확인을 요청하면 직접 완료.
5. 등록된 사이트 선택 → 요청 → 사이트맵 제출 → `https://etherforecast.live/sitemap.xml` 제출.
6. 요청 → 웹 페이지 수집에서 홈페이지·한국어 안내·공개 예측 요약 주소를 제출. 같은 URL을 반복 제출할 필요는 없습니다.
7. 검증 → robots.txt와 웹 페이지 최적화 결과 확인. 보고서에서 수집·색인과 노출 여부 점검.
8. RSS는 실제 피드가 생겼을 때만 제출. sitemap.xml을 RSS로 제출하지 않습니다.

공식 문서: https://searchadvisor.naver.com/guide/request-feed

## 3. Bing과 글로벌 야후

1. https://www.bing.com/webmasters/ 에 로그인.
2. Google Search Console에서 가져오기가 표시되면, 이미 확인된 `etherforecast.live`를 선택해 가져오기. 연결 권한은 화면에서 직접 확인.
3. 직접 등록을 선택한 경우 사이트 주소 입력 → HTML 메타 태그 방식 선택 → `msvalidate.01` 태그를 이 채팅에 전달. 배포 후 확인.
4. Sitemaps 메뉴에서 `https://etherforecast.live/sitemap.xml` 제출.
5. URL Inspection에서 홈페이지와 `/ko/`, `/outlook.html`을 검사하고 URL 제출.
6. 글로벌 Yahoo 검색 유입은 Bing 등록을 우선 활용. Yahoo Japan 등 지역별 서비스는 동일하다고 단정하지 않습니다.

공식 문서: https://www.bing.com/webmasters/help/add-and-verify-site-12184f8b

## 4. 인스타그램

- 직접 계정 생성 또는 기존 계정에서 시작. 사용자 이름은 `etherforecast`처럼 브랜드와 맞추되 실제 사용 가능 여부 확인.
- 프로필 편집 → 이름에 `ETH Forecast | 이더리움 분석`처럼 주제를 명확히 표시.
- 프로필 링크 추가 → `https://etherforecast.live/ko/?utm_source=instagram&utm_medium=social&utm_campaign=profile`.
- 공개 계정으로 운영. 필요하면 프로페셔널 계정으로 전환해 제공되는 인사이트 확인. 정확한 메뉴 이름은 앱 버전에 따라 달라질 수 있습니다.
- 첫 게시물은 이용법, 이번 주 전망, 지난주 실제 오차의 세 가지로 구성. 한 게시물에서 한 가지 질문에 답하고 사이트 링크는 프로필로 안내.
- 카드 3장 구성: 기준 시각이 있는 차트 / 확률·범위 해설 / 실제 결과 및 상세 분석 위치. 잘 맞은 예측만 선택하지 않습니다.
- 게시글 본문 URL을 클릭 가능한 유입 경로로 가정하지 말고 프로필 링크·지원되는 스토리 링크를 사용.
- 자동 대량 댓글, 반복 DM, 무관한 해시태그는 사용하지 않습니다.

## 5. 페이스북

- 페이지 만들기 → 사이트와 같은 브랜드 이름 → 실제 주제에 맞는 카테고리 선택.
- 소개와 웹사이트에 `https://etherforecast.live/?utm_source=facebook&utm_medium=social&utm_campaign=page` 추가.
- 소개 게시물과 성과 확인 방법을 고정. 주간 요약을 링크와 함께 게시하면 사이트의 Open Graph 이미지·제목이 공유 미리보기에 사용될 수 있습니다.
- Meta Business Suite에서 해당 계정에 제공되는 연결·예약 기능을 확인. 인스타그램과 동일 원자료를 사용하되 형식은 조정.
- 관련 그룹은 그룹 규칙을 읽고 운영자 허용 범위에서 참여. 링크만 반복해서 게시하지 않습니다.

## 6. X와 커뮤니티

- 프로필 이름·소개·웹사이트를 통일. X 링크 예: `https://etherforecast.live/?utm_source=x&utm_medium=social&utm_campaign=profile`.
- 고정 게시물에 서비스 소개, 예측 발행 시각, 실측 오차 확인 위치를 안내.
- 주간 발표는 `지난번 예측 → 실제 결과 → 이번 전망과 한계` 순서. 게시 빈도보다 정확성과 일관성을 우선.
- Reddit·가상화폐 카페 등은 커뮤니티 규칙을 읽고 자기 홍보·사이트 링크 허용 범위를 확인. 운영자 또는 개발자임을 명확히 밝힙니다.
- SNS 계정은 각각 별도 가입·운영이 필요합니다. 웹사이트 메타 태그를 넣었다고 SNS 검색에 자동 등록되지는 않습니다.

## 측정과 남은 사항

- 초기 지표: 검색 노출·클릭, 사이트 재방문, 후원 버튼 클릭, 실제 후원 완료.
- UTM은 유입 경로를 표시하는 문자열이며, 그 자체로 방문자 통계를 수집하지 않습니다. 분석 도구 연결·개인정보 안내가 준비된 뒤 사용량을 측정합니다. 이번 변경은 분석 추적기를 추가하지 않습니다.
- 소유권 확인 코드는 사용자별로 다르므로 현재는 미설정. 구글·네이버·Bing의 공개용 HTML 태그를 받은 뒤 추가할 수 있습니다.
- SNS 계정 생성·외부 게시·카드 결제 연결은 이 코드 변경에 포함되지 않습니다.
- IndexNow 갱신 알림은 선택 사항입니다. 구글 색인 요청 대체 수단으로 취급하지 않습니다.
