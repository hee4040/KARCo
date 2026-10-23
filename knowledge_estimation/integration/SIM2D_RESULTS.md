# 상대 지식 추정기: 팀 2D 시뮬레이터 연동과 최종 평가

담당: 상대 지식 추정(노션 적응형 통신 설계 v0.3의 `belief/estimator`).
시뮬레이션 환경, 규칙 기반, PPO는 팀원 코드(`#1/ppo`)를 쓴다. 이 문서의 모든 수치는 2D 시뮬레이터 안의 값이며 실측이 아니다.

## 1. 작동 방식

```text
팀원 sim/world.py ── A가 볼 수 있는 사건 ──▶ integration/sim2d.py (연결부)
                                              │ 사건 → 본체 호출
                                              ▼
                                  probabilistic/estimator.py (KnowledgeEstimator)
                                  항목(N·I·C)별 근거 = (출처, P(저장), P(내용이 맞음))
                                  b_raw = 근거 중 가장 강한 것 (독립 가정 없이 max 하한)
                                              ▼
                                  probabilistic/projection.py
                                  b_cal = 보정표(b_raw),  b_safe = min(b_cal, 정책 상한)
                                              ▼
                     결정 직전 기록(b_raw·b_cal·b_safe, 근거, 근거 나이, 지문) → 규칙/PPO가 b_safe를 읽음
```

| 시뮬레이터 사건 | 본체 처리 | 노션 |
|---|---|---|
| TASK·재전송·TASK v2·응답·POSE_UPDATE 전송 | `record_send` (재전송은 같은 msg_id, 다른 시도 번호). 보낸 것만으로는 저장으로 보지 않음 | §6 |
| ACK | `observe_ack`. 저장 확인(R=1), 내용 유효성은 측정 r_crop·r_current | §6 |
| ACK 마감까지 미수신 | `observe_ack_timeout`: d(1−a)/(1−da), 측정 p_recv·p_ack·마감 | §6, ①(b) |
| needs_info, 정보 요청 | `observe_missing(needs_info)`: 해당 항목 무효화 | §6 |
| not_found | 진단 기록만. 원인 불명이라 C를 지우지 않음 | ④(d) |
| B 복귀 보고(경로) | `record_pass_by`: A의 경로 지도 기준, 이상이 알려진 작업대는 정상 통과로 보지 않음 | ③ |
| A가 그 작업대를 다시 봄 | `confirm_pass_by`: 정상이면 사용, 이상이면 폐기 | §8 |
| A의 재검증 성공 | `record_result` + `verify_task`. RESULT ok만으로는 쓰지 않음, 측정 q_learn | ①(a) |
| 오조작 발견 | 외형 무효화, 해당 작업 근거 철회 | ②(d) |
| 기억 epoch 변경 | `observe_status`: 전체 무효화 | ②(b) |
| 위치 재측정 | C의 새 버전 등록, 이전 C 근거 무효 | §5 |

본체 원칙상 지나가며 본 이상은 C의 근거로 쓰지 않는다(버전 있는 보고만 인정).
측정값이 없으면 None이고, 0이나 기본값으로 채우지 않는다. 정책 경계에서만 "모르면 B가 없다고 보고 보낸다"는 보수 규칙으로 0을 넘기며, 그 횟수를 센다. 최종 평가에서는 0회였다.

## 2. 확률의 출처와 데이터 분할 (노션 §11·§12)

| 단계 | seed | 내용 | 산출물 |
|---|---|---|---|
| 파라미터 측정 | 100000–100299 | 시뮬레이터 로그의 성공/시도: p_recv·p_ack·F_recv(채널별), p_acq(항목·작업대·경로별), q_learn, r_crop, r_current | `sim2d_parameters_v1.json` (31개) |
| 점수 보정 | 110000–110199 × 팀 정책 4종 | 결정 직전 (b_raw, B 실제 보유) 11,162건 → PAV | `sim2d_score_calibration_v1.json` |
| 개선안 선택 | 120000–120199 | 단일 보정 / 근거 종류별 / 근거 종류+나이별 비교 → 차이 없음(95% 구간 내), 단일 보정 채택 | — |
| **최종 평가** | **200000–200199** (처음 사용) | 실행 전 추정기·파라미터·보정표 해시 고정, 실행 후 불변 확인 | 아래 표 |

`--scores --candidates`와 `--features sources|sources_age`로 근거별 보정을 다시 만들 수 있다. 노션 §10의 ablation("이득이 출처별 보정에서 오는가")용이다.

## 3. 최종 평가 (seed 200000–200199, 조건 5개 × 200판, 팀 규칙 정책과 Minimal Push+Pull, 결정 26,174번)

결정 직전 추정과 그 순간 B의 실제 보유를 비교했다. 95% 구간은 에피소드 단위 bootstrap이다. τ = 0.8은 미리 고정했다.

### Brier (낮을수록 좋음)

| 추정 | N | I | C |
|---|---:|---:|---:|
| b_raw | 0.098 | 0.250 | 0.383 |
| **b_cal = b_safe** | **0.075** [0.072, 0.078] | **0.166** [0.162, 0.170] | **0.231** [0.228, 0.233] |
| Confirmed Cache (자체 기준선) | 0.272 | 0.616 | 0.384 |
| Confirmed Cache + Result (자체 기준선) | 0.198 | 0.365 | 0.384 |
| 낙관적 추정 (ablation) | 0.084 | 0.169 | 0.336 |

### 높은 추정·미보유 (b ≥ 0.8 중 B가 실제로 없던 비율)

| 추정 | N | I | C |
|---|---:|---:|---:|
| **b_cal** | 2.8% (18,994건) | 10.8% (13,519건) | 2.3% (1,147건) |
| Confirmed Cache | 1.1% (13,898건) | 1.4% (1,636건) | 3.6% |
| Cache + Result | 1.5% | 9.0% | 3.6% |
| 낙관적 추정 | 7.4% | 15.0% | 5.3% |

### 신뢰도 (b_cal 구간별 실제 보유율)

| 항목 | 구간 | 결정 | 평균 추정 | 실제 [Wilson 95%] |
|---|---|---:|---:|---|
| N | 0.2–0.4 | 6,998 | 0.287 | 0.297 [0.287, 0.308] |
| N | 0.8–1.0 | 18,994 | 0.970 | 0.972 [0.970, 0.975] |
| I | 0.4–0.6 | 12,015 | 0.429 | 0.429 [0.420, 0.438] |
| I | 0.8–1.0 | 13,519 | 0.889 | 0.892 [0.887, 0.897] |
| C | 0.2–0.4 | 25,027 | 0.384 | 0.400 [0.394, 0.406] |
| C | 0.8–1.0 | 1,147 | 0.990 | 0.977 [0.967, 0.984] |

### 해석

- 정확도는 세 항목 모두 b_cal이 가장 좋다. 자체 기준선보다 2~4배 정확하고, 조건 5개 모두에서 같다.
- 보정 효과: b_raw → b_cal에서 Brier가 N 23%, I 34%, C 40% 줄었다. 큰 구간의 신뢰도는 거의 일치한다(표본이 50개 미만인 구간은 해석하지 않음).
- 낙관적 추정은 Brier는 비슷하지만 높은 추정·미보유가 약 2.5배다. 노션이 ablation으로 둔 이유와 맞다.
- Confirmed Cache는 위험 비율은 낮지만 높게 추정하는 경우가 적다. 그만큼 다시 보내야 하는 정보가 많다.
- **I의 10.8%는 추정기로 줄일 수 없는 불확실성이다.** 검증 seed 진단에서 그 원인의 76%가 "B가 외형을 잘못 배움", 11%가 "A가 아직 모르는 기억 초기화"였고, 둘 다 A가 관측할 수 없다. 추정값 자체는 잘 보정되어 있다(0.889 대 0.892). 이 위험을 감수할지는 규칙의 기대비용이나 b_safe 상한(정책 파라미터)이 정한다.
- 근거가 없는 결정의 비중이 크다(N 25%, I 45%, C 94%). 이때 b_cal은 평균 보유율 수준이다. C가 대부분 근거 없음인 이유는 정책이 C를 잘 보내지 않고, 본체가 지나가며 본 이상을 C로 쓰지 않기 때문이다.

### 한계

- 시뮬레이터 안의 측정과 평가다. B의 인식은 팀원 모델(프레임당 15%)이며 실측이 아니다.
- seed 0–199(팀원 비교 스크립트의 seed)는 개발 중 진단에 한 번 썼으므로 최종 보고에 쓰지 않는다.
- PPO 결과는 사용자 추정값으로 재학습한 뒤에 비교해야 한다.

## 4. 재현

```powershell
python -m knowledge_estimation.integration.sim2d_calibrate --episodes 300                 # 파라미터
python -m knowledge_estimation.integration.sim2d_calibrate --scores --episodes 200        # b_cal
python -m knowledge_estimation.integration.sim2d_trace --episodes 40                      # 노션 흐름·규칙 검사
python -m knowledge_estimation.integration.sim2d_evaluate --episodes 200 --seed0 200000   # 최종 평가
python -m unittest discover -s knowledge_estimation/tests -t .
```

팀원 파일 수정은 연결 지점만이다: `sim/world.py`(추정기 등록, `_core` 사건 전달, 출동 기록, 자체 검사의 추정기 참조), `sim/env2d.py`·`rule_ppo_compare/calibrated.py`·`imitate.py`·`rl/eval.py`(추정기 이름 기본값), `rule_ppo_compare/compare.py`("Rule (proposed)"가 이 추정기를 읽음, 결과 폴더 옵션).
