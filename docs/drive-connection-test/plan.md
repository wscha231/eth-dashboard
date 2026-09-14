# Google Drive 연결 테스트 구현 계획

상태: 계획 검토 대기. 실행 코드 작성 및 실연결 테스트 미실행.

## 변경 파일
- scripts/check_gdrive_connection.py: Python 표준 라이브러리 기반 소규모 연결 검사.
- .github/workflows/gdrive_connection_check.yml: 수동 workflow_dispatch, contents: read, 5분 제한. 예측/배포 workflow와 분리.

## 검사 순서
1. GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN, GDRIVE_FOLDER_ID 환경변수 존재 및 공백/HTTP 헤더/JSON 전체 붙여넣기 등 명백한 형식 오류 확인. 값은 출력하지 않는다.
2. https://oauth2.googleapis.com/token 에 refresh_token grant POST. 응답 access token은 메모리에만 보관하고 즉시 로그 마스킹한다. 응답 본문 및 인증 헤더는 출력하지 않는다.
3. Drive API files.get으로 지정 폴더 MIME type, trashed 및 canAddChildren 확인. drive.file 권한은 기존 임의 폴더 ID만으로 접근이 허용되지 않는다는 점을 오류 안내에 포함.
4. 지정 폴더에 실행 ID와 UUID가 포함된 이름으로 작은 JSON 시험 파일 한 개 생성. 기존 파일은 수정하지 않는다.
5. files.get alt=media로 재다운로드하여 원본과 바이트/SHA-256 일치 검사.
6. 이번 실행이 생성하여 ID를 받은 시험 파일만 finally에서 휴지통으로 이동. 정리 실패는 별도 실패로 기록. 기존 파일 삭제/정리 금지.
7. Actions 요약에 Secret 존재, 토큰 갱신, 폴더 접근, 업로드, 다운로드 일치, 시험 파일 정리의 PASS/FAIL 출력. 모든 항목 성공 시에만 전체 성공.

## 오류 처리
- 네트워크 요청별 30초 timeout, 제한된 재시도. 파일 생성 요청이 불명확하게 실패한 경우 중복 업로드를 막기 위해 무조건 재시도하지 않는다.
- missing secret / invalid_client / invalid_grant / HTTP 403 / HTTP 404는 단계와 정제된 코드만 출력. 비밀값, 토큰 응답, 민감한 폴더 정보는 출력하지 않는다.
- Refresh token의 장기 만료 여부는 단발 테스트 성공으로 보장하지 않는다. Google Cloud에서 External Testing 여부를 별도 확인한다.

## 검증과 실행
- 성공 흐름, 갱신 실패, 바이트 불일치 및 정리 실패를 최소한의 모의 응답으로 확인하고 로그에 토큰이 남지 않도록 검사.
- 사용자의 계획 승인 후 구현하여 검토 가능한 변경으로 제공한다.
- 현재 연결 도구에 신규 Actions 실행 기능이 없으므로 실제 실행은 workflow가 기본 브랜치에 반영된 뒤 사용자가 Run workflow를 한 번 눌러야 할 수 있다. 제공되는 실행 권한/도구가 달라지면 다시 확인한다.
- 이후 실행 링크의 job/로그를 읽어 실검증 결과를 보고한다. Secret 설정이나 workflow 생성만으로 성공이라고 보고하지 않는다.

## 범위
인증과 저장소 왕복 동작만 검사한다. 데이터 이관, 모델 재학습, 자동 백업 운영 연결은 이번 검사에 포함하지 않는다.
