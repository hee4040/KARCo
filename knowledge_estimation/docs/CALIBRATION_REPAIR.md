> **보관 기록:** 이 문서의 실험 코드(`knowledge_estimation/simulation/`)와 관련 테스트는 팀 2D 시뮬레이터(`sim/`)로 옮기면서 삭제했다. 코드는 커밋 `cb67c57`(브랜치 `subin/knowledge_estimation`)에 그대로 있다: `git checkout cb67c57 -- knowledge_estimation/simulation`. 현재 연동은 `sim/knowledge.py`를 본다.

# 통과 직후 보정 오류의 원인과 검증

이 문서는 통과 직후 평균 악화의 수정 실험이다. 이후 전체 상황 검증에서 정상 ACK 직후
보정 악화와 미지원이 추가로 드러났다. 최종 상태는 ESTIMATOR_VERIFICATION.md를 따른다.

## 원인

기존 `calibrate.fit_scores`는 N/C/I 각각에 대해 모든 결정 시점의 (raw,K)를 한 PAV에 넣었다.
이 표는 그 기록 분포에서 같은 점수의 평균 보유율을 맞추지만, 다른 증거 이력에 대한
조건부 보유율까지 보장하지 않는다. 실제로 통과 직후와 ACK timeout 이후를 섞으면서
신선한 통과 추정을 과도하게 낮췄다.

기존 360개 실험의 calibration 기록에서 I raw=0.7916667인 사례를 직접 집계했다.

| 상태 (원인 분석 전용) | A가 가진 증거 | 표본 | 실제 K 비율 |
|---|---|---:|---:|
| 통과 직후 | pass_by | 48 | 0.79167 |
| 전송 직전 | pass_by | 48 | 0.79167 |
| 채널 처리 후 | pass_by + ack_timeout | 24 | 0.66667 |
| 비공개 재시작 비교 후 | pass_by + ack_timeout | 24 | 0.33333 |
| TTL 확인 시점 | pass_by + ack_timeout | 24 | 0.33333 |

합치면 108/168=0.642857이다. 기존 보정표가 통과 직후에도 반환한 값이 바로 이것이다.
통과 직후 raw 자체가 이 calibration 집단에서는 0.79167로 맞았는데, 다른 이력의 실패까지
섞으면서 0.64286으로 내려갔다. 반올림/PAV 구현 버그보다 입력 집단 혼합이 주원인이다.
같은 벽이 시야와 채널에 영향을 주므로 timeout이 있는 집단은 통과 집단과 분포도 다르다.
또한 기존 raw는 통과 근거와 0인 미수신 근거를 max로 합쳐 이 차이를 직접 반영하지 못한다.
메모리 재시작도 A에게 보고되지 않으므로 원래 기록만으로 실제 재시작을 알아낼 수는 없다.

## 수정

`--conditioning evidence_sources_v1`은 항목마다 A가 실제로 가진 증거 종류의 집합을
정렬·중복 제거해 별도 PAV를 적합한다. 예: `[]`, `[pass_by]`, `[ack,pass_by]`,
`[ack_timeout,pass_by]`를 구분한다. 소스가 추가돼도 해당 표본이 없으면 미측정으로 거절한다.

- 보정값을 임의로 정하지 않는다. 각 보정 구간은 calibration의 성공 수/표본 수다.
- phase 이름, 실제 reset 여부, 벽, 카메라 방향, B DB, 정답은 모델 입력에서 제외한다.
- 유효성 무효/만료 항목은 기존처럼 0에 고정한다. raw가 미측정이면 None을 유지한다.
- 미관측 context와 지원 범위 밖 raw는 전체 보정표로 대신하지 않는다.
- 런타임 `project`와 오프라인 `evaluate`가 같은 `calibrated_value`를 호출한다.
- 기존 pooled 모델도 계속 읽을 수 있다. 수정안은 명시적 conditioning/선택 모델로 사용한다.

원래 통과 근거를 보고 확률을 추정하는 모델은 유지했다. 이번 수정은 신경망 학습이나
실제 재시작을 알아내는 기능을 추가한 것이 아니라, 증거 이력별 사후 보정이다.
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)의 사후 보정·신뢰도 평가를
참고했으며, 위 증거 종류 분할은 우리 데이터에서 진단한 문제에 대한 자체 설계다.

## 별도 데이터로 검증

`simulation.repair_calibration`은 다음 절차와 조건을 `experiment_plan.json`에 먼저 기록한다.

1. 기존 데이터: 원인 분석에만 사용한다.
2. 새 source 72 에피소드: 새 획득 빈도와 시간 분포를 구해 고정한다.
3. 새 calibration 144 에피소드: pooled와 증거 종류별 PAV를 각각 적합한다.
4. 새 validation 144 에피소드: 통과 직후 N/I 평균 Brier를 1차 기준으로 선택한다.
   동률이면 전체 N/C/I 평균 Brier, 이어 후보 이름 순서로 결정한다.
5. 선택한 model_id·검증 결과·계획 hash를 `selection.json`에 저장한다.
6. 그 뒤에 처음으로 새 final test 144 에피소드를 생성한다. 선택 모델과 기존 방식만 비교한다.

총 504 에피소드다. source/보정/검증/평가의 offset·heading 그리드가 서로 다르고,
이전 실험 그리드도 재사용하지 않았다. ID에는 repair-v1 namespace를 부여했다.
선택 전에는 final test 라벨 파일이 존재하지 않는다. 모든 분할의 그룹 불중복을 검사한다.
이는 이상적인 합성 물체/설계 조건의 분리이며, 실제 미지 물체나 실측 분포의 검증은 아니다.

## 결과와 남은 한계

| 새 final test | 기존 pooled | 증거 종류별 | raw |
|---|---:|---:|---:|
| 통과 직후 N Brier | 0.18263 | 0.14728 | 0.14873 |
| 통과 직후 I Brier | 0.21391 | 0.17593 | 0.18519 |
| 전체 N Brier | 0.18309 | 0.16676 | 0.21436 |
| 전체 C Brier | 0.03125 | 0.03125 | 0.04167 |
| 전체 I Brier | 0.19081 | 0.17464 | 0.21943 |

validation도 같은 방향의 개선을 보였으며 증거 종류별 보정이 선택됐다. test를 보고
모델을 바꾸거나 다시 적합하지 않았다. 결정적 조건 순회이므로 IID 신뢰구간/통계적 유의성을 주장하지 않는다.

통과 직후 평균 악화는 개선했지만 모든 하위 조건이 개선되지는 않았다. 기존 pooled와 비교하면
최근접 거리 0.15m에서 N/I Brier가 각각 약 0.02789/0.02799 더 높고, 벽이 있는 I에서도
약 0.00253 높다. 전체 평균 개선을 거리·방향별 개선으로 확장하지 않는다.
이미 본 test를 다시 최적화하지 않고 이 결과를 잔여 오류로 남겼다.

모델이 실제 가림/방향/재시작을 관측하지 않는 한 그 상태를 정확히 식별할 수 없다.
이후에는 A가 실제로 이용 가능한 추가 관측/경과 시간 모델과 별도 검증 설계를 검토할 수 있다.
아직 규칙 정책의 payload 선택, 실제 집기 성공률, 실제 D455 인식 정확도를 검증한 것은 아니다.

## 파일과 실행

```powershell
python -m knowledge_estimation.simulation.repair_calibration --output knowledge_estimation/data/calibration_repair_new
python -m unittest discover -s knowledge_estimation/tests -q
```

현재 결과는 `data/calibration_repair_v1/`의 RESULTS.md/summary.json에 있다.
diagnosis.json, validation_*.json, test_*.json, test_by_condition.csv로 원인과 세부 조건을 확인한다.
selected_model.json은 같은 폴더의 simulation_params.yaml로 구성한 추정기에만 적용한다.
출처는 simulation이며 실제 정책에 실측 모델로 자동 로드하지 않는다.
