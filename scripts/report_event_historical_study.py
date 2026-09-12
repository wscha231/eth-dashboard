"""Render the measured multi-year comparison without interpreting it as live skill."""
import argparse
import json
from pathlib import Path


def report(data):
    lines = ['# ETH 모든 기간의 다년 백테스트 결과', '',
             f"데이터 기준: {data['data_as_of']}. 평가: 월별로 학습·선택·보정을 마친 뒤 다음 달의 매일 00:00 UTC 적격 기준시점을 예측하는 시간 순서 재현. 모든 시간대의 시간별 발행을 재현한 결과는 아닙니다.", '',
             '과거 재현은 실제 당시 게시한 예측 기록과 다릅니다. 원래 수집 시점의 데이터 수정 이력을 완전히 복원할 수 없으며, 이미 연구에 사용한 과거 데이터이므로 새로운 미사용 최종 검증표본으로 부르지 않습니다.', '',
             '| 기간 | 평가 시작~끝 | 예측 수 / 비중복 | 가격오차 후보 / 기존 / 변화 없음 (%p) | 확률 Brier 후보 / 기존 | 80% 범위 포함률 |',
             '|---|---|---:|---:|---:|---:|']
    for h, r in data['horizons'].items():
        label = f'{int(h)//24}일' if int(h) >= 24 else f'{h}시간'
        if not r.get('origins'):
            lines.append(f'| {label} | 충분한 학습 이력 없음 | 0 | — | — | — |'); continue
        c, i, n = (r[k] for k in ('candidate', 'incumbent', 'no_change'))
        lines.append(f"| {label} | {r['first_origin'][:10]} ~ {r['last_origin'][:10]} | {r['origins']:,} / {r['nonoverlap']:,} | {c['return_mae_pp']:.3f} / {i['return_mae_pp']:.3f} / {n['return_mae_pp']:.3f} | {c['event_brier']:.4f} / {i['event_brier']:.4f} | {c['coverage80']:.1%} |")
    lines += ['', '가격오차는 수익률 MAE이며 낮을수록 좋습니다. 확률 Brier는 위·아래 장벽 도달 확률 오차의 평균으로 낮을수록 좋습니다. 비중복 표본도 완전한 통계적 독립을 보장하지 않습니다.', '',
              '## 출력별 개선 근거', '', '| 기간 | 가격 | 확률 | 범위 | 후보 / 기존 범위 점수 (%p) |', '|---|---|---|---|---:|']
    for h, r in data['horizons'].items():
        if not r.get('origins'): continue
        flags = ['비교 기준 통과' if r['head_evidence'][k] else '우위 입증 안 됨' for k in ('point', 'probability', 'interval')]
        lines.append(f"| {h}시간 | {' | '.join(flags)} | {r['candidate']['interval_score80_pp']:.3f} / {r['incumbent']['interval_score80_pp']:.3f} |")
    lines += ['', '통과 기준: 후보-비교대상 손실의 95% 달력 블록 구간 상단이 0 미만이어야 합니다. 가격은 기존 모델과 변화 없음, 확률·범위는 기존 모델과 과거빈도 기준선을 모두 비교합니다. 블록 길이는 최소 7일이며 30일 예측에서는 31일입니다. 여러 출력의 탐색적 결과이며 자동 모델 승격이나 실제 서비스 우위를 인증하지 않습니다.', '',
              '## 예측 당시 시장 상태별 가격 오차', '',
              '시장 상태는 예측 기준시각까지의 정보로 구분합니다. 추세 분류와 변동성 분류는 서로 겹치므로 표본 수를 합산하지 않습니다. 특정 사후 구간의 좋은 결과로 새 모델을 선택하지 않습니다.', '',
              '| 기간 | 당시 상태 | 표본 | 후보 / 기존 / 변화 없음 MAE (%p) |', '|---|---|---:|---:|']
    names = {'trend_up': '상승 추세', 'trend_down': '하락 추세', 'trend_range': '횡보',
             'vol_high': '높은 변동성', 'vol_normal': '보통 변동성'}
    for h, r in data['horizons'].items():
        for state, families in r.get('origin_states', {}).items():
            c, i, n = (families[k] for k in ('candidate', 'incumbent', 'no_change'))
            if not c.get('rows'): continue
            lines.append(f"| {h}시간 | {names[state]} | {c['rows']:,} | {c['return_mae_pp']:.3f} / {i['return_mae_pp']:.3f} / {n['return_mae_pp']:.3f} |")
    lines += ['', '## 미래 업데이트에 적용', '',
              '- 원래 월별 활성 모델과 실제 발행 원장은 보존합니다. 개선 후보는 기존 별도 학습·예측 경로로 계속 갱신됩니다.',
              '- 이번 다년 평가를 이후 후보 검토에 함께 보여 줍니다. 가격·확률·범위를 따로 평가하고, 한 항목의 개선을 모델 전체의 개선으로 해석하지 않습니다.',
              '- 다음 연구 실행은 저장된 월별 모델을 재사용하며 새로운 만기 결과와 수정된 입력에 해당하는 월만 다시 평가합니다.',
              '- 학습 이력이 부족한 초기 월, 입력이 빠진 월, 아직 만기가 끝나지 않은 목표는 제외합니다. 30일 정답은 예측 기준시각에서 721시간 뒤 끝나는 전체 관측창이 필요합니다.',
              '- 웹사이트는 완성된 보고서·해시 검증된 차트 자료만 가져옵니다. 사이트 게시 성공은 외부 배포 확인이 별도로 필요합니다.', '',
              f"평가 완료 월: {len(data['monthly_audits'])}. 제외 월: {len(data['excluded_months'])}. 기록 생성: {data['generated_at']}.", '']
    return '\n'.join(lines)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='lake/signals');p.add_argument('--output',required=True);a=p.parse_args()
    data=json.loads((Path(a.root)/'historical_study.json').read_text())
    if data.get('status') != 'complete':raise ValueError('complete study required')
    output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(report(data),encoding='utf-8')
