# 상대 지식 추정 (knowledge_estimation)

로봇 A가 "로봇 B가 지금 무엇을 알고 있는가"를 확률로 추정한다. 팀의 규칙 기반·PPO 통신 정책은 이 값을 보고 B가 이미 아는 정보는 보내지 않는다. 팀 2D 시뮬레이터(`sim/world.py`)에서 B 지식 추정기는 이것 하나다.

- **추정 대상(노션 §5):** 작업대마다 N(정상 배치), I(물체 외형), C(현재 위치) 세 항목. 각 항목의 값은 "B가 맞는 내용을 가지고 있을 확률"이다.
- **입력은 A가 실제로 아는 것만 쓴다:** 보낸 메시지, 받은 ACK, ACK가 오지 않음, 요청, B의 복귀 보고, A가 다시 본 작업대 상태, A의 재검증, 기억 epoch. B의 실제 기억이나 시뮬레이터의 가림 정보는 넣지 않는다.
- **세 단계 값:** b_raw(근거로 계산한 점수) → b_cal(별도 seed에서 적합한 PAV 보정) → b_safe(정책 상한; 검증 결과 상한 없음). 정책은 b_safe를 읽는다.

결과 요약은 아래 [결과](#결과)에, 전체 수치는 [integration/SIM2D_RESULTS.md](integration/SIM2D_RESULTS.md)에 있다.

## 파일

```text
knowledge_estimation/
  types.py                     항목·버전·근거·추정값 자료형
  parameters.py                측정 파라미터 저장소: 성공/시도로 잰 값만 받고, 없으면 MissingMeasurement
  probabilistic/
    estimator.py               KnowledgeEstimator: 사건별 근거 갱신과 estimate()
    perspective.py             통과 근거 모델: B의 계획 경로를 A의 지도·카메라로 다시 그려 P(저장) 계산
    probabilities.py           무ACK 사후확률 등 공식
    projection.py              b_raw → b_cal PAV 보정 (적합·검증·적용)
  integration/
    sim2d.py                   팀 세계 연결부 CoreKnowledge (세계 사건 → 추정기 호출)
    sim2d_calibrate.py         파라미터 측정(seed 100000+)과 b_cal 적합(110000+)
    sim2d_rule_select.py       b_safe 상한 선택(검증 seed 120000+)
    sim2d_evaluate.py          결정 직전 추정 평가(Brier, 신뢰도, 높은 추정·미보유, 두 변형 짝 비교)
    sim2d_trace.py             한 에피소드 추적과 노션 규칙 검사
    sim2d_parameters_v2.json   측정 파라미터 (seed 100000–100299)
    sim2d_score_calibration_v2.json        b_cal (시점 기반, 기본)
    sim2d_score_calibration_table_v2.json  b_cal (빈도표 변형, 비교용)
    sim2d_policy_v2.json       b_safe 상한 선택 결과
    SIM2D_RESULTS.md           연동 방식, 데이터 분할, 최종 결과
  tests/                       추정기·시점 모델·팀 세계 연동 테스트
  docs/                        논문 근거, 개발 이력
  data/                        실행 산출물 (git 제외)
```

## 사건별 처리

| A가 겪는 사건 | 추정기 처리 | 확률의 출처 |
|---|---|---|
| 메시지 전송·재전송 | 보낸 것만으로는 저장으로 보지 않음 | 채널별 p_recv, F_recv |
| ACK 받음 | 저장 확인(R = 1). I·C는 내용 유효성을 곱함 | r_crop, r_current |
| ACK 마감까지 안 옴 | 수신 사후확률 d(1−a)/(1−da)로 교체 | p_recv, p_ack, 마감 |
| needs_info, 정보 요청 | 그 항목 무효화 | — |
| not_found | 진단 기록만(원인 불명이라 C를 지우지 않음) | — |
| B의 복귀 보고 | 경로 위 작업대마다 통과 기록. **A가 그 작업대를 다시 정상으로 볼 때까지 보류**(노션 §8), 이상이면 폐기 | **시점 기반 모델** (아래) |
| A의 재검증 성공 | 그 작업의 N·I를 근거로 씀(B의 ok 보고만으로는 쓰지 않음) | q_learn(verified) |
| 오조작 발견 | 외형 무효화, 그 작업의 근거 철회 | — |
| 기억 epoch 변경 | 전체 무효화 | — |
| 위치 재측정 | C의 새 버전 등록, 이전 C 근거 무효 | — |

여러 근거는 max로 합친다(독립 가정 없이 쓸 수 있는 하한). 측정값이 없는 근거는 0이 아니라 None이고, None이 섞인 항목은 b_raw도 None이다. 정책 경계에서만 "모르면 B가 없다고 보고 보낸다"는 보수 규칙으로 0을 넘기며, 그 횟수(`unsupported`)를 센다.

## 통과 근거: 시점 기반 모델

B가 작업대 옆을 지나가며 그 작업대를 봤을 확률을 A가 직접 계산한다([perspective.py](probabilistic/perspective.py)).

1. A는 자기가 명령한 경로를 안다. B의 계획 위치·방향을 프레임마다 만든다(B의 D455는 진행 방향 고정).
2. A의 지도(작업대 위치, A가 아는 벽)와 카메라 모델(87°, 0.40–1.20 m)로 작업대가 시야에 든 프레임 수 n을 센다.
3. P(N 저장) = 1 − (1 − p_frame)^n, P(I 맞음) = P(N 저장) × q_learn(pass).

p_frame(시야 안에서 프레임당 인식률)과 q_learn(pass)(본 작업대의 외형을 맞게 기억할 비율)은 경로와 무관한 센서 비율이다. 측정 seed에서 성공/시도로 한 번 잰다. 그래서 새 경로·속도·배치에서도 다시 재지 않고 값을 낼 수 있다. 이전 방식인 경로·작업대별 빈도표(p_acq, 24칸)는 그 경로를 다시 재야 했고, 속도나 배치가 바뀌어도 옛 값을 그대로 냈다.

한계: A의 지도에는 가림막이 없다(팀 설계). 가림막 조건에서는 실제보다 높게 계산하고, 그 차이는 b_cal이 평균적으로만 흡수한다. 프레임마다 인식이 독립이라고 가정한다.

이 모델을 처음 설계하고 검증한 이전 연구 단계는 [docs/HISTORY.md](docs/HISTORY.md)에 한 줄로 남겼다.

## 실행

저장소 루트(`KARCo/`)에서 실행한다. 평가 한 번(1,000–2,000판)은 수십 분 걸린다.

```powershell
python -m unittest discover -s knowledge_estimation/tests -t .                          # 테스트 (약 3초)

# 다시 만들 때만 (결과 파일은 이미 integration/에 있음)
python -m knowledge_estimation.integration.sim2d_calibrate --episodes 300               # 1. 파라미터
python -m knowledge_estimation.integration.sim2d_calibrate --scores --episodes 200      # 2. b_cal (두 변형)
python -m knowledge_estimation.integration.sim2d_rule_select --episodes 200             # 3. b_safe 상한

# 평가 (처음 쓰는 seed로)
python -m knowledge_estimation.integration.sim2d_evaluate --seed0 400000 --episodes 200
python -m knowledge_estimation.integration.sim2d_evaluate --seed0 410000 --episodes 200 --b-speed 0.30
cd rule_ppo_compare; python compare.py --episodes 200 --seed0 500000 --out ../knowledge_estimation/data/rule_final_v2

# 한 에피소드 추적
python -m knowledge_estimation.integration.sim2d_trace --trace base 3
```

## 데이터 분할

| 용도 | seed |
|---|---|
| 파라미터 측정 | 100000–100299 |
| b_cal 적합 | 110000–110199 |
| b_safe 상한 선택 | 120000–120199 |
| 노션 규칙 검사 | 50000+ |
| 최종 추정 평가 | 400000–400199 (속도 시험 410000+, 420000+) |
| 최종 규칙 비교 | 500000–500199 |

seed 0–199(팀 비교 스크립트 기본값)와 200000·300000대(이전 최종 평가)는 개발 중에 봤으므로 새 최종 보고에 쓰지 않는다.

## 결과

처음 쓰는 seed에서 측정한 값이다. 자세한 표와 구간은 [SIM2D_RESULTS.md](integration/SIM2D_RESULTS.md)에 있다.

**추정 정확도** (seed 400000+, 결정 28,216번, Brier: 낮을수록 좋음)

| 추정 | N | I | C |
|---|---:|---:|---:|
| **시점 기반 (기본)** | **0.075** | **0.165** | **0.233** |
| 빈도표 (이전 방식, 비교용) | 0.075 | 0.165 | 0.233 |
| Confirmed Cache (ACK 받은 것만 안다고 봄) | 0.271 | 0.618 | 0.396 |

같은 경로·속도에서는 두 방식이 같다(사전 등록한 비열등 기준 통과). B 속도가 바뀌면 빈도표는 옛 값을 그대로 내고 시점 기반은 다시 계산한다. 그래서 빨라질 때 I 고확신 오류(통과 근거만 있는 결정)가 빈도표 23.7%, 시점 기반 12.8%였다.

**규칙 기반과 연결한 통신 성능** (seed 500000+, 조건마다 200판)

| 비교 대상 | 보상 | 전송량 |
|---|---|---|
| Confirmed Cache | base·유사품 2개 조건에서 +1.4–5.8 (유의), 나머지 2개는 차이 없음 | 5개 조건 모두 4.5–20 KB 적음 |
| 낙관적 추정 | 손실이 큰 limited에서 +1.5 (유의) | 4개 조건에서 적음 |
| Minimal Push+Pull | 유사품 조건에서는 Minimal이 더 높음 | 전송량은 약 절반 |

## 한계와 다음 단계

- 모든 수치는 팀 2D 시뮬레이터 안의 값이다. B의 인식 모델(프레임당 15%, 외형 오학습 10%)은 팀원이 정한 가정이며 실측이 아니다.
- PPO는 이전 추정값으로 학습한 모델이다. 이 추정값으로 다시 학습한 뒤 비교해야 한다.
- 실제 로봇 연결: p_frame과 q_learn을 실측하면 경로마다 다시 잴 필요가 없다. 실측 로그 기록 코드는 [docs/HISTORY.md](docs/HISTORY.md)의 방법으로 되살린다.

## 문서

| 문서 | 내용 |
|---|---|
| [integration/SIM2D_RESULTS.md](integration/SIM2D_RESULTS.md) | 팀 세계 연동 방식, 데이터 분할, 최종 결과 (현재 기준) |
| [docs/RESEARCH_BASIS.md](docs/RESEARCH_BASIS.md) | 논문 근거와 적용 한계 |
| [docs/HISTORY.md](docs/HISTORY.md) | 개발 단계, 지운 문서·코드와 되살리는 방법 |
