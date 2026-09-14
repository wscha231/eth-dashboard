# Drive 장기 저장 구현 계획

승인 범위: 2026-09-14 사용자 지시 '그렇게하자 저장소 활용을 하여 예측 데이터들을 영구적으로 저장하여할용'. 아래는 그 지시의 세부 구현이다.

1. scripts/gdrive_store.py: 검증된 Secret 파서 재사용. 파일을 고정 크기 chunk로 나누고 deterministic gzip/SHA256으로 중복 제거. 객체 업로드 무결성을 검증한 뒤 마지막에 실행별 불변 manifest를 기록. 기존 객체/manifest 삭제·덮어쓰기 없음. SQLite WAL을 포함한 consistent snapshot, 경로/크기/압축해제 한도, 다운로드 SHA 검증 후 임시 디렉터리에서 복원.
2. scripts/archive_to_drive.py + .github/workflows/gdrive_archive.yml: 예측·시장수집·연구·과거연구 종료마다 별도 archive job. upstream repository/main/event/workflow 경로 확인 후 승인된 artifact만 읽고 데이터 브랜치를 commit 고정하여 보관. raw inputs, actual ledger, historical research, model checkpoints를 다른 stream으로 기록. 중단 후 재실행은 기존 객체를 재사용. 매일 누락 점검과 수동 실행 제공. 초기 push에서 현재 남아 있는 latest artifacts/data branches를 seed.
3. event_hourly.yml: 기존 artifact 복원이 불가능하면 Drive의 검증된 event-hourly snapshot 복원. 발행/장부 정리 후 최종 snapshot도 artifact로 남겨 archive job이 publication receipts까지 저장. 기존 inference/모델선정 로직 변경 없음.
4. daily_forecast.yml: GitHub 일별 source 복원 실패 시 Drive daily-data snapshot으로 복구 후 기존 수집·정산 수행.
5. 독립 recovery workflow: stream/as-of 선택해 과거 버전을 내려받고 SHA/DB 검증하여 연구용 artifact로 제공. 실제 예측 장부나 production 모델 자동 교체 금지.
6. tests: 중복 제거, 중간 실패 manifest 미확정, SQLite WAL, 훼손/경로탈출 거부, 과거시점 선택, 연구/실발행 stream 분리, Secret 미출력, workflow 출처 제한. 기존 관련 테스트 및 CI 통과 후 반영.

보존정책: 자동 만료나 garbage collection 없음. Drive 용량 및 계정 상태의 제약까지 '영구 보장'하지 않는다. 과거 이미 만료된 원본은 복구 불가임을 명시. 운영 파일 보존은 모델 성능 개선 자체와 구분한다. 백업 지연/실패는 Actions 실패와 요약으로 드러낸다.
