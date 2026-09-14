# Drive 장기 저장 연결 조사

기준 main: 823bf7288c1d05b98de0c6c7d7c59fec02e6a14b
사용자는 연결 테스트 PASS 이후 장기 저장 및 다음 실행 재사용을 승인했다.

- Drive 폴더 1DFboZtT7PhrMHd4JckLUcs7VvSFXBEZP를 connector metadata로 확인했다. 연결 테스트 run 34807645021에서 실제 읽기/쓰기/정리 PASS.
- event_hourly.yml: Actions에서 hourly-state(2일), daily-backup(90일), research-state(30일)를 복원한다. snapshots는 observations를 40일로 줄인다.
- persist_event_ledger.sh: issued.db, shadow_issued.db 및 raw/관측 날짜별 partition/모델 hash를 data/event-ledger git 브랜치에 보존한다. publish_events.sh는 발행 확인 후 장부를 다시 저장한다.
- daily_forecast.yml: data/daily-forecast의 일별 CSV, 가용시각, 예측DB, vendor/market CSV를 복원한다.
- event_research.yml: 연구 전체 source/checkpoint는 만료형 artifact에만 보관된다. 연구 issued.db를 실제 장부로 가져오면 안 된다(event_state.py의 merge 규칙).
- event_historical_study.yml: 진행상태 7일, 결과90일, adaptive rows7일. 누락된 과거 artifacts를 새로 만들어 과거 실제 발행으로 취급할 수 없다.
- 기존 Google credential parser는 개별 Secret 및 단일 INI Secret을 모두 지원한다.
- GitHub connector에는 신규 workflow dispatch 기능이 없다. 변경 파일에 한정한 push 자동 실행으로 초기 실제 보관을 검증할 수 있다.

유지할 계약: 실제 발행/결과 수정 이력의 분리, 원시 observed_at, SHA256 모델 버전, SQLite online backup, 검증된 main 출처만 모델 복원, Drive 장애 시 기존 GitHub 경로 유지.

## 첫 실제 이관 관찰 (2026-09-14)

PR #49 반영 뒤 시간별/일별 예측 작업은 성공했다. 초기 Drive archive run 34809033743에서 작은 파일의 순차 POST가 병목이다. 첫 장부 브랜치의 승인된 파일은 874개이며 05:25 UTC에 370개 Drive 객체를 확인했다. 객체 형식은 유지하고 최대 4개 업로드를 병렬 처리해 정기 백업의 대기 시간을 줄인다.
