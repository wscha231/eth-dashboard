"""Run, independently audit and report fixed delayed-feedback candidates."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from signal_pipeline.adaptive_research import run_review, METHODS, EXPERTS
from signal_pipeline.data import utc


def audit(root, data):
    checked = {}
    for h, expected in data['horizons'].items():
        ref = data['files'][h]; path = Path(root)/ref['scored_rows_path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != ref['scored_rows_sha256']:
            raise ValueError('adaptive rows hash mismatch')
        f = pd.read_csv(path); original = f[f.model.eq('incumbent')].sort_values('slot')
        for method in (*EXPERTS, *METHODS):
            v = f[f.model.eq(method)].sort_values('slot')
            if not np.array_equal(v[['slot','target_end','return','up','down','terminal']].to_numpy(),
                                  original[['slot','target_end','return','up','down','terminal']].to_numpy()):
                raise ValueError('adaptive paired truth mismatch')
            y = np.expm1(v['return']); lo, hi = np.expm1(v.q10), np.expm1(v.q90)
            independent = {'return_mae_pp': float(np.abs(np.expm1(v.q50)-y).mean()*100),
                'event_brier': float((((v.hit_up-v.up)**2+(v.hit_down-v.down)**2)/2).mean()),
                'coverage80': float(((v['return'] >= v.q10)&(v['return'] <= v.q90)).mean()),
                'interval_score80_pp': float((hi-lo+10*np.maximum(lo-y,0)+10*np.maximum(y-hi,0)).mean()*100)}
            for key, value in independent.items():
                if not np.isclose(value, expected['metrics'][method][key], rtol=1e-10, atol=1e-10):
                    raise ValueError('independent adaptive metric mismatch: '+key)
        targets = pd.to_datetime(f.target_end, utc=True); slots = pd.to_datetime(f.slot, utc=True)
        if not ((targets-slots).eq(pd.Timedelta(hours=int(h)+1))).all() or not (targets <= utc(data['data_as_of'])).all():
            raise ValueError('incorrect or immature adaptive target')
        a = f[f.model.isin(METHODS[1:])]
        if not (pd.to_datetime(a.feedback_target_end, utc=True) < pd.to_datetime(a.slot, utc=True)-pd.Timedelta(hours=1)).all():
            raise ValueError('future feedback entered prediction')
        c = a[a.calibration_cutoff.notna()]
        if not (pd.to_datetime(c.calibration_target_end,utc=True) < pd.to_datetime(c.calibration_cutoff,utc=True)-pd.Timedelta(hours=1)).all():
            raise ValueError('future calibration outcomes')
        if not (pd.to_datetime(c.calibration_cutoff,utc=True) <= pd.to_datetime(c.slot,utc=True)).all():
            raise ValueError('future fitted calibrator')
        checked[h] = {'origins':len(original),'adaptive_predictions':len(a),'nonoverlap':expected['nonoverlap']}
    return {'status':'passed','horizons':checked,'checks':'row hashes; identical dates/truth; independent MAE/Brier/coverage/interval score; full target maturity; delayed feedback and monthly calibration cutoff',
            'report_sha256':hashlib.sha256((Path(root)/'adaptive_study.json').read_bytes()).hexdigest()}


def markdown(data):
    names = {'incumbent':'기존 모델','candidate':'이전 후보','conservative':'보수 기준',
             'adaptive_mix':'오차 기반 혼합','adaptive_risk':'혼합 + 위험 보정'}
    lines = ['# ETH 개선 후보 재시험 결과','',
        f"자료 기준 {data['data_as_of']}. 과거 월별 예측 기록을 재사용하고, 각 시점 이전에 만기가 완전히 끝난 결과로만 갱신했습니다. 이미 검토한 과거 데이터의 개발 재시험입니다.", '',
        '## 동일 날짜의 출력별 비교','',
        '| 기간 | 방법 | 표본 / 비중복 | 수익률 MAE (%p) | 이벤트 Brier | 80% 범위 포함률 | 범위 점수 (%p) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for h,r in data['horizons'].items():
        for method in names:
            m = r['metrics'][method]
            lines.append(f"| {h}시간 | {names[method]} | {r['origins']} / {r['nonoverlap']} | {m['return_mae_pp']:.3f} | {m['event_brier']:.4f} | {m['coverage80']:.1%} | {m['interval_score80_pp']:.3f} |")
    lines += ['', 'MAE·Brier·범위 점수는 낮을수록 좋습니다. 범위 점수는 폭과 빗나간 결과를 함께 벌점화하므로, 범위를 크게 넓혀 포함률만 올린 것을 개선으로 보지 않습니다. 보수 기준은 가격이 그대로라는 중심 예측과 기존 과거빈도 위험 예측입니다. 가격 오차 감소만으로 급등·급락 예측력이 좋아졌다고 해석하지 않습니다.', '',
              '## 탐색적 비교 기준 통과 여부','', '| 기간 | 방법 | 가격 | 확률 | 범위 |', '|---|---|---|---|---|']
    for h,r in data['horizons'].items():
        for method, flags in r['head_evidence'].items():
            if method == 'conservative': continue
            values = ['통과' if flags[k] else '입증 안 됨' for k in ('point','probability','interval')]
            lines.append(f"| {h}시간 | {names[method]} | {' | '.join(values)} |")
    lines += ['', '각 항목은 기존 모델과 단순 기준선 모두에 대해, 후보-기준 손실의 95% 달력 블록 구간 상단이 0 미만인지 확인합니다. 보수 기준은 비교 기준선이므로 승격 판정에서 제외합니다. 30일 예측의 블록은 31일입니다. 여러 방법·기간을 비교한 탐색 결과이며 다중검정 전체의 유의성을 보장하지 않습니다.', '',
        '## 해석과 다음 단계','',
        '- 먼저 90개(짧은 기간) 또는 365개(14·30일)의 과거 만기 결과를 확보한 뒤 모든 방법을 같은 날짜에서 평가했습니다. 따라서 이전 전체 보고서보다 평가 시작이 늦고 표본 수가 적습니다.',
        '- 가격·확률·범위는 각각 별도의 목표입니다. 높은 중립 클래스 정확도나 적은 경보만으로 예측력이 좋아졌다고 판단하지 않습니다. 전체 JSON에 균형 정확도·상승/하락 재현율·오경보율·연도별·시장상태별 결과를 함께 보존합니다.',
        '- 위험 보정은 과거 실제 생성한 혼합 예측의 오차를 예측 당시 변동성으로 나눈 뒤 현재 변동성에 맞춰 범위를 다시 계산합니다. 확률은 과거 만기 예측을 이용해 매월 보정합니다. 개별 예측의 80% 적중을 보장하지 않습니다.',
        '- 기존 활성 모델과 발행 기록은 유지합니다. 이번 결과는 후속 후보 검토 자료이며 실시간 성과나 자동 교체 결정이 아닙니다.',
        '- 정기 다년 연구가 완성되면 같은 설정의 경량 재시험을 다시 실행합니다. 새 평가 결과를 보고 설정을 바꾸면 별도 버전으로 기록해야 합니다.', '',
        '방법 참고: [시간 변화에 대응하는 예측 범위](https://arxiv.org/abs/2202.07282), [온라인 확률 예측 결합](https://arxiv.org/abs/2109.14309). 본 구현은 지연 정답을 처리하는 별도 경험적 후보로, 논문의 보장을 그대로 주장하지 않습니다.', '']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--root',default='lake/signals')
    p.add_argument('--output',default='adaptive-public'); p.add_argument('--budget-seconds',type=int,default=300)
    p.add_argument('--rows-output',help='Optional folder containing only this completed run\'s verified scored rows')
    a = p.parse_args(); data = run_review(a.root,budget_seconds=a.budget_seconds); verified = audit(a.root,data)
    output = Path(a.output); output.mkdir(parents=True,exist_ok=True)
    (output/'results.json').write_bytes((Path(a.root)/'adaptive_study.json').read_bytes())
    (output/'audit.json').write_text(json.dumps(verified,indent=2)+'\n')
    (output/'results.md').write_text(markdown(data))
    if a.rows_output:
        rows=Path(a.rows_output);rows.mkdir(parents=True,exist_ok=True)
        for ref in data['files'].values():
            source=Path(a.root)/ref['scored_rows_path'];target=rows/ref['scored_rows_path']
            target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        (rows/'manifest.json').write_text(json.dumps(data['files'],indent=2)+'\n')
    print(json.dumps({'status':verified['status'],'runtime_seconds':data['runtime_seconds']}))


if __name__ == '__main__': main()
