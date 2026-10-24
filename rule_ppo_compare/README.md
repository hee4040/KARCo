# 규칙 기반 vs PPO 비교 (적응형 통신 시나리오 기준)

`적응형 통신 시나리오.html`의 시나리오 분기, 비교 방법, 측정 지표를 `sim/` 2D 시뮬레이터에 모두 넣고, 규칙 기반(`rule_based_comm/`, heewoo/rule-based 브랜치 원본)과 PPO를 **같은 사건, 같은 비용**으로 비교한다. 규칙 기반 원본 코드는 수정하지 않고, 설정과 결과 모델만 상속해서 바꾼다.

```bash
python calibrated.py              # 규칙 결과 모델 자체 검사
python imitate.py                 # 규칙을 따라 배운 뒤 PPO (BC, PPO from rule)
python compare.py                 # 조건별 200 에피소드 → 표 + results.csv + episodes.csv
python compare.py --episodes 20   # 빠르게
```
PPO 모델은 `python ../rl/train.py --env sim2d`(PPO), `--no-belief`(PPO no B)로 만든다.

## 시나리오 반영 (`sim/world.py`)

| 분기 | 시뮬레이터 동작 |
|---|---|
| ①(a) | 같은 사건 처리 중이면 보내지 않음(공통 게이트). B는 결과를 보고만 하고, **A가 다음 순찰에서 재검증**한 뒤 지식을 확정 |
| ①(b) | ACK가 2초 안에 안 오면 같은 메시지 재전송(최대 2회). ACK 없을 때 수신 확률 = p(1 − p) / (1 − p²) |
| ②(a) | 확인된 지식은 생략 가능 |
| ②(b) | B 기억 초기화(평균 600초마다), STATUS 50% 유실. ACK에 memory_epoch와 needs_info(N) → A가 전부 무효화하고 N을 출발 전에 보충 |
| ②(c) | 유사품은 **스티커만 다른 쌍**(4 cm 간격). 글(ID)로는 구분 불가, 외형 기억·crop·위치로만 구분 |
| ②(d) | B가 외형을 잘못 배움(10%) → 틀린 물체를 고치고 성공으로 보고 → A 재검증에서 발견 → 그 spot 지식 무효화 → TASK v2(crop + 다른 물체의 N′, C′) |
| ③A | 지나가며 본 정상 근거는 **A가 다음 순찰에서 정상을 확인할 때까지 보류**, 이상이면 제외 |
| ③B | 지나가며 본 외형으로 crop 생략. 가림막 조건에서 못 봤으면 B가 현장에서 요청 → A가 b_I ← 0 |
| ④(a) | σ와 간격으로 crop 필요성 판단. B는 2σ 반경 안 후보만 봄(σ = 거리 RMS 오차) |
| ④(b) | A가 σ를 실제의 1/3로 믿는 경우 20% |
| ④(c) | 유사품 오배치는 A가 스티커를 읽어야 표적 확정(첫 시점 60%, 두 번째 시점 95%). 못 읽으면 보내지 않고 다른 시점에서 재관측 |
| ④(d) | B가 오는 중 물체 재이동(오배치의 15%). A가 보면 POSE_UPDATE, B가 못 찾으면 not_found → A 재관측 → POSE_UPDATE |

## 조건

| 조건 | 내용 |
|---|---|
| base | 유사품 spot 2개, 이상 3종 |
| occluded | base + 가림막(A의 가시성 추정은 모름) |
| limited | base + 대역폭 5 ~ 50 KB/s, 손실 10 ~ 30% |
| lookalike | 모든 spot이 유사품 쌍, 오배치만 (②(c)·③B·④ 집중) |
| lookalike_occluded | lookalike + 가림막 (③B 가림) |

모든 방법이 같은 시드 0 ~ 199를 쓰고, 이상 발생·인식 오차·작업자 실수·재이동·기억 초기화는 통신과 다른 난수로 뽑는다. 그래서 같은 시드면 어떤 방법이든 같은 사건을 겪는다.

## 보상 = 시나리오 문서 기대비용과 같은 가중치

성공(A 재검증) +10, 정보 부족 포기 −3, 미해결 −10, 대기 −1 / 이상 1건 / 5초, 전송 −0.1 / KB, 중복 전송 −1, **요청 왕복 −1, not_found 재관측 −2, 오조작 −12**.

## 방법

| 방법 | 결정 | B 지식 추정 |
|---|---|---|
| Periodic Full | 매번 E + N + C + I (약한 기준선) | — |
| Event-gated Full | 게이트 + E + N + C + I | — |
| Minimal Push+Pull | 게이트 + E만, 부족하면 B가 요청 | — |
| Fixed State | E + ID + C + N 고정 | — |
| Scene-only rule | 기대비용 규칙 | 없음 (0) |
| Rule (Confirmed Cache) | 기대비용 규칙 | ACK로 받은 것만 |
| Rule (Cache+Result) | 기대비용 규칙 | + A가 재검증한 작업 결과 |
| Rule (optimistic) | 기대비용 규칙 | 지나갔으면 안다(1) |
| **Rule (proposed)** | 기대비용 규칙 | 제안 추정(`knowledge_estimation`: 시점 기반 통과 확률, 보류 규칙, 요청·오조작 시 무효화) |
| Rule (p_acq table) | 기대비용 규칙 | 같은 추정기, 통과 확률만 이전 경로별 빈도표 (비교용) |
| Rule-default | 원본 규칙 그대로(원래 확률 가정) | 제안 추정 |
| PPO (no B) / PPO | 처음부터 학습 | 없음 / 제안 추정 |
| BC (rule copy) / **PPO from rule** | Rule (proposed)를 따라 배운 뒤 PPO | 제안 추정 |

"기대비용 규칙"은 원본 구조(요구사항 → 8개 후보 → 기대비용 → 최소)에 시뮬레이터와 같은 크기·비용·결과 모델을 넣은 것이다(`calibrated.py`). 결과 모델은 B의 대상 고르기 순서(지나가며 봄 → 2σ 반경 → crop → 외형 기억 → 유일 후보 → 요청)를 A가 관측 가능한 값으로 계산하고, σ 과신(④(b))과 재이동(④(d))도 Monte Carlo로 반영한다.

## 지표 (시나리오 문서)

| 열 | 뜻 |
|---|---|
| reward, succ, time, late | 보상, 성공률(A 재검증 기준), 탐지→검증 시간, deadline(60초) 초과율 |
| KB, retx, pull, v2 | 총 KB/에피소드, 재전송 KB, 출발 전 N 요청 수, TASK v2 KB |
| msgs, imgs, reqs, dup, wrong | 메시지, 사진, 요청 왕복, 중복 비율, 오조작/에피소드 |
| crop- | crop이 필요했던 결정 중 crop을 뺀 비율 |
| f-ask, f-wr | 생략 실패율: 요청으로 회복 / 미보고 오조작 (결정당) |
| unKB | 불필요 전송 KB/에피소드 (B가 이미 가진 N, 필요 없던 crop) |
| Brier N·I·C, high-est unheld | 결정 직전 추정의 정확도, 추정 ≥ 0.8인데 실제로 없던 비율 |

`compare.py` 끝에는 Rule (proposed) 대비 에피소드별 짝지은 차이(평균, 95% 신뢰구간, p값)를 출력한다.

## 결과 (2026-10-23, 조건별 200 에피소드, PPO 20만 스텝)

> 2026-10-25: 추정기의 통과 확률이 시점 기반 모델로 바뀌었다. 규칙 계열의 새 최종 비교(seed 500000–500199)는 [knowledge_estimation/integration/SIM2D_RESULTS.md §5](../knowledge_estimation/integration/SIM2D_RESULTS.md)에 있다. 아래 표는 seed 0–199에서 이전 추정기로 낸 값이고, PPO는 새 추정값으로 다시 학습해야 한다.

보상 (높을수록 좋음). 괄호는 Rule (proposed) 대비 짝지은 차이, \* = p < 0.05.

| 방법 | base | occluded | limited | lookalike | lookalike_occ |
|---|---|---|---|---|---|
| Periodic Full | −68.0 | −69.7 | −87.0 | −77.1 | −79.6 |
| Event-gated Full | −37.0 | −38.2 | −48.8 | −48.6 | −50.0 |
| Minimal Push+Pull | −18.7 | −19.2 | −28.9 | −25.0 | −25.9 |
| Scene-only rule | −16.8 | −17.6 | −17.5 | −32.3 | −33.9 |
| Rule (Confirmed Cache) | −17.1 | −17.8 | −18.2 | −30.5 (−3.9\*) | −31.5 |
| Rule (Cache+Result) | −17.0 | −17.6 | −18.6 | −27.1 | −29.4 |
| Rule (optimistic) | −16.7 | −17.8 | −20.9 (−2.0\*) | −25.7 | −28.0 |
| **Rule (proposed)** | −16.9 | −17.7 | −18.9 | −26.6 | −28.9 |
| Rule-default | −21.4 | −22.4 | −24.2 | −38.2 | −39.4 |
| PPO (no B) | −23.5 | −23.7 | −28.1 | −28.7 | −29.4 |
| PPO | −16.1 | −16.8 | −18.8 | −25.0 | −25.2 (+3.7\*) |
| BC (rule copy) | −17.2 | −18.0 | −19.8 | −29.5 (−2.9\*) | −30.4 |
| **PPO from rule** | −16.4 | −16.6 | −18.6 | **−23.2 (+3.4\*)** | **−24.2 (+4.7\*)** |

전송량 KB/에피소드 (같은 순서의 주요 방법)

| 방법 | base | limited | lookalike | lookalike_occ |
|---|---|---|---|---|
| Event-gated Full | 201.8 | 306.7 | 188.0 | 188.4 |
| Rule (Confirmed Cache) | 7.8 | 10.7 | 45.8 | 46.6 |
| Rule (proposed) | 6.3 | 8.3 | 31.0 | 35.5 |
| PPO | 9.9 | 9.4 | 51.0 | 58.0 |
| PPO from rule | 11.7 | 17.8 | 61.9 | 67.7 |

전체 수치(지표 전부)는 `results.csv`, 에피소드별 값은 `episodes.csv`.

**읽는 법**
- **B 지식 추정 (H1):** 같은 규칙에서 추정기만 바꾸면, 유사품 조건에서 제안 추정이 Confirmed Cache보다 보상 +3.9(p = 0.009), 전송량 −14.8 KB. base·occluded에서는 차이 없음. PPO도 추정·기록 입력을 빼면(no B) 중복 전송 50%로 크게 나빠짐.
- **낙관적 추정:** 손실이 큰 limited에서 제안보다 −2.0(p = 0.005). 추정을 과신하면 요청 왕복이 늘어 손해. Brier I는 낙관이 낮지만 "높은 추정·미보유" 비율이 0.15 ~ 0.17로 제안(0.09 ~ 0.10)보다 높음.
- **규칙 vs PPO (H2):** 유사품 조건에서 PPO from rule이 보정 규칙보다 보상 +3.4 ~ +4.7(p < 0.05). 대신 사진을 더 보내 전송량은 +31 KB. 규칙의 결과 모델보다 crop을 일찍 보내는 쪽이 시간·실패를 줄여 이득. base·occluded·limited에서는 유의한 차이 없음.
- **처음부터 학습한 PPO**도 이번에는 "보내지 않음"을 배워 규칙 수준에 도달(lookalike_occ +3.7, p = 0.045).
- **Minimal Push+Pull**은 유사품 조건에서 강하지만(−25.0) 손실이 큰 limited에서 크게 나빠짐(−28.9).
