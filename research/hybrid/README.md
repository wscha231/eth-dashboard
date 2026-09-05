# ETH 7일 / 30일 CatBoost + Transformer 재설계

2026-09-05. PR #19의 기존 후보는 두 기간 모두 무변화 기준선을 이기지 못했다. 사용자 요청에 따라 기존 RF/KNN/HGB/Ridge 운영 후보를 중단하고 CatBoost 및 실제 PyTorch patch Transformer, 그리고 두 계열을 결합하는 정책을 시간순으로 평가한다. 최고의 모델이라는 표현은 **이 고정 탐색 범위에서의 결과**에 한정한다.

## 예측 대상과 입력

7일과 30일의 직접 로그수익률을 각 발행시점의 과거 30일 변동성 × sqrt(horizon)으로 나누어 학습한다. 미래 가격 수준을 입력하지 않는다. 기준가격과 예측 수익률을 곱해 가격 차트를 만든다. 출력 로그수익률은 수치적 폭주에 대비해 사전에 [-3,3]으로 제한하고 제한 횟수를 공개한다.

입력은 종료된 ETH/BTC OHLCV의 과거 변화율·범위·변동성·거래량·상대수익률과 검증된 거래 흐름 17개다. 합계 59개 특징이다. 거시/온체인 역사 수정값은 사용하지 않는다. 거래 흐름은 원본 일봉 시작일 + 2일에 연결해 종료 후 하루를 더 지연시킨다. 선물 자료가 존재하지 않던 초기 구간도 버리지 않고, 학습 시점에 충분히 관측되지 않은 변수는 그 월의 모델에서 비활성화한다. 미래에 값이 생겨도 다음 적합 전까지 활성화하지 않는다.

과거 거래소 데이터는 재구성본이며 모든 당시 공개 시점을 복원했다는 뜻은 아니다. 기존 연구에서 이미 살펴본 역사 자료를 사용하므로, 시간 누수를 통제한 과거 재현과 완전히 새 미래 표본 검증을 구분한다.

## 고정 탐색과 시간 분리

| 구성 | 짧은 설정 | 긴 설정 |
|---|---|---|
| CatBoost | depth 3, 최대 240 trees, L2 20 | depth 5, 최대 360 trees, L2 40 |
| Transformer | 32일 입력, 4일 patch, width 16, 2 heads | 64일 입력, 4일 patch, width 24, 4 heads |
| Transformer 깊이 / 최대 epoch | 1 encoder layer / 16 | 1 encoder layer / 24 |
| 7일 학습창 | 2년 | 3년 |
| 30일 학습창 | 3년 | 5년 |

각 월 초마다 최소 500개 학습행을 확보한다. outer 학습 목표일은 해당 월 초보다 7일 모델은 3일, 30일 모델은 15일 이전이어야 한다. 내부 마지막 120개 날짜로 학습 횟수를 고르며, 내부 학습 목표일도 내부 검증 시작일 이전에 같은 embargo를 적용한다. 내부 scaler와 외부 최종 scaler를 별도로 적합한다. 가장 좋은 내부 학습 횟수로 그 월에 이용 가능한 전체 학습행을 다시 적합한다. 미래 test 구간으로 early stopping하지 않는다.

월별 OOF 예측이 쌓인 후, 이미 만기가 끝난 이전 365일(7일 모델) / 730일(30일 모델)에서 180개 이상의 결과를 확보하면 CatBoost 설정·Transformer 설정·결합 가중치 0/25/50/75/100%·진폭 0.5/1.0을 고른다. 0/100%도 허용해 결합이 단독 모델보다 나쁘면 불필요한 구성의 가중치를 없앤다. 목표는 같은 날짜의 수익률 MAE다. 처음에는 짧은 두 모델의 동일 가중 조합을 사용한다. 이 방식은 in-sample 학습 예측으로 stacking하지 않으며 미래 전체 점수로 과거 가중치를 바꾸지 않는다.

비교 경로는 4개 단독 설정, 동일 가중 hybrid, 과거 성적으로 최적화한 hybrid, hybrid와 무변화 중 과거 성적으로 선택한 안전 경로, 무변화 참고선이다. 홈페이지 기본은 시간순 최적화 hybrid이며 기준선과 성적을 함께 표시한다. 전체 성적을 본 사후 최우수는 실전에서 미리 알고 고른 모델로 표현하지 않는다.

80% 범위와 상승/횡보/하락 확률은 월 초 전에 확정된 OOF의 변동성 정규화 잔차에서 계산한다. 처음에는 이미 만기된 학습수익률 분포로 초기화한다. 이는 경험적 범위이며 CQR의 보장된 coverage라는 주장을 하지 않는다. 가격 MAE/RMSE 외에 상승 recall·precision·오경보율, 무조건부 3상태 Brier, 범위 포함률을 함께 보고한다. 기간이 겹치는 예측은 최소 30일 달력 블록 bootstrap으로 기술적 구간을 표시한다.

## 운영 전환

- 기존 매일 job은 시장 수집과 과거 발행분 실제값 정산만 수행한다. 기존 모델 신규 발행과 RF/KNN champion 재학습을 중단한다.
- 기존 forward job은 거래 흐름 수집·기존 발행분 정산만 수행한다. 예전 HGB/Ridge 신규 발행을 중단한다.
- 기존 전체 후보 재평가와 구형 champion 재학습 스케줄은 수동 재현용으로 남긴다.
- `Hybrid ETH full replay`는 7/30일 × 2023년 이전/이후의 네 작업으로 나눠 같은 원자료 snapshot을 평가한다. 월별 checkpoint를 저장하고 완성된 동일 snapshot만 결합한다. 매주 일요일 07:13 UTC 자동 재검증한다.
- `Daily optimized ETH forecast`는 source refresh 성공 후 또는 07:43 UTC에 실행한다. 현재 월의 저장 모델을 재사용하고, 원자료가 바뀌어 필요한 경우와 월이 바뀐 경우만 적합한다. 7일·30일 예측을 매일 갱신하며 비용이 큰 전 구간 재학습은 하지 않는다.
- `data/hybrid-forecast`에 모델·원시 예측·월별 cache·변경 불가능한 발행 원장을 저장한다. 새로운 `hybrid_forecast.json`과 다운로드를 웹사이트에 게시한다. 과거 모델은 접힌 보관 영역으로 이동한다.
- 오늘 실제 발행한 두 예측만 새 원장에 들어간다. 백테스트 시점을 과거 실발행 기록으로 대량 삽입하지 않는다. 같은 날 다시 실행해도 발행값을 덮어쓰지 않는다. 원자료 정정은 실제값의 새 revision으로 남긴다.
- 신규 계열은 전향 연구 상태로 발행한다. 과거 점수가 좋아져도 검증된 실전 우위를 자동으로 선언하지 않는다. 기준선보다 나쁘면 사이트에서 이를 명시한다.

## 자료와 구현 근거

첨부된 `2405.11431v2`의 모델 비교와 CNN-LSTM/VAE 논문을 검토했으며, 특정 모델 이름만으로 ETH 7/30일 성과를 추정하지 않는다. 이번 범위는 요청한 CatBoost + Transformer의 분리 검증과 결합 효과에 집중한다.

- [PyTorch TransformerEncoder](https://docs.pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html): 실제 attention encoder 사용. window 전체가 origin 이전이므로 window 내 양방향 attention은 미래 목표값을 보지 않는다.
- [CatBoost parameter tuning](https://catboost.ai/docs/en/concepts/parameter-tuning): tree 수의 검증 선택을 과거 inner 구간으로 제한한다.
- [CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression): 정규화 수익률에 MAE를 사용한다.

고정 버전: torch 2.8.0+cpu, CatBoost 1.2.10, numpy 2.3.5, pandas 2.2.3, sklearn 1.8.0. seed 1729, CPU thread 2. 테스트는 미래 입력/목표값 불변성, 실제 두 계열 직렬화/예측, 두 단계 purge, complete cohort, 당일 재실행 불변성, 과거 발행 금지, 차트 단위 변환을 검증한다. 실제 전 구간 결과와 실행시간은 배포 후 PR에 기록한다.
