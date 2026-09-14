# Google Drive 연결 테스트 조사

확인일: 2026-09-14 UTC
대상: wscha231/eth-dashboard 기본 브랜치 main

## 직접 확인한 내용
- .github/workflows의 YAML 15개를 읽었으며 GDRIVE, GOOGLE_, drive, oauth 참조가 발견되지 않았다.
- 기본 브랜치 파일 검색 GDRIVE 결과는 0건이었다.
- 전체 파일 트리에서 Drive/OAuth/credential 이름의 전용 모듈은 발견되지 않았다. 파일명 검색만으로 모든 코드의 기능 부재를 단정하지 않는다.
- tests/test_system_storage.py는 로컬 아카이브, git 기반 일별 소스 복원, Actions artifact 복원을 검사한다. Google Drive 실인증 검사가 아니다.
- 최근 Actions 실행 목록에 Drive 연결 테스트는 없었다. 기존 예측 workflow 성공은 Drive 성공 근거가 아니다.

## 검증되지 않은 내용
사용자는 GDRIVE_CLIENT_ID, GDRIVE_CLIENT_SECRET, GDRIVE_REFRESH_TOKEN, GDRIVE_FOLDER_ID 설정을 완료했다고 알렸다. 이 세션 GitHub 도구는 Secrets API를 지원하지 않아 등록 여부 및 값 유효성을 직접 조회하지 못했다. 토큰 갱신, 폴더 접근, 업로드, 다운로드 모두 미실행이다.

## 실행 제약
현재 GitHub 도구에는 기존 실패 job 재실행은 있지만 신규 workflow_dispatch 실행 기능은 없다. CLI gh도 설치되어 있지 않다. 기존 workflow에는 필요한 검사가 없어 재실행으로 검증할 수 없다. 비밀값을 채팅이나 로컬로 추출하지 않고 GitHub Actions 안에서 테스트해야 한다.

## 적용할 사용자 지침
프로젝트 지침은 research.md 및 plan.md를 작성하고 사용자가 계획을 승인한 다음 코드를 작성하도록 요구한다. 이번 커밋은 조사/계획만 포함한다.
