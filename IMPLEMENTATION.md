# 규칙 기반 최소 통신 선택 모듈 구현 정리

## 구현 결과

`적응형 통신 시나리오.html`의 최소 통신 선택 파트를 독립 Python 패키지로 구현했다. 로봇이나 ROS2 없이 사건 정보와 상대 지식 추정값만으로 `E/N/C/I` 8개 후보의 기대비용을 비교하고, 가장 낮은 후보와 선택 근거를 반환한다.

```python
from rule_based_comm import RuleBasedCommunicationPolicy

decision = RuleBasedCommunicationPolicy().select(
    event={
        "event_id": "e3",
        "target_confirmed": True,
        "has_similar_objects": True,
        "object_distance_cm": 4.0,
        "position_uncertainty_cm": 3.0,
    },
    belief={"b_N": 0.72, "b_I": 0.72, "b_C": 0.0},
)

print(decision.selected_items)  # ('E', 'N', 'C')
print(decision.expected_cost)   # 약 1.64
```

결과에는 선택된 항목뿐 아니라 8개 후보 각각의 초기 전송비, 예상 요청비, 후속 전송비, 오조작비, 재관측비, 미해결비가 포함된다. `decision.as_dict()`로 JSON 직렬화 가능한 형태를 얻을 수 있다.

## 파일별 역할

| 파일 | 역할 |
| --- | --- |
| `rule_based_comm/schemas.py` | `Event`, `BeliefState`, 결정 및 비용 결과 스키마와 입력 검증 |
| `rule_based_comm/config.py` | 메시지 크기, 비용 가중치, 결과 예측 확률 |
| `rule_based_comm/requirements.py` | 공통 게이트와 상황별 필요 정보 분석 |
| `rule_based_comm/candidates.py` | E를 포함하는 8개 공통 행동 생성 |
| `rule_based_comm/cost_model.py` | 후보별 결과 확률 및 기대비용 계산 |
| `rule_based_comm/rule_policy.py` | 8개 후보 평가와 최소비용 선택 |
| `rule_based_comm/policy_adapter.py` | PPO와 공유할 행동 ID 0~7 변환 |
| `main.py` | 네 가지 대표 시나리오 실행 예제 |
| `tests/` | 요구사항, 시나리오 분기, 어댑터, 검증 테스트 |

## 합의가 없던 부분에 적용한 결정

### 공통 입력과 행동

- 입력은 `Event`와 `BeliefState`로 분리했다.
- 상대 지식 담당자는 `b_N`, `b_I`, `b_C`만 제공하며 이 모듈은 그 값의 산출 과정을 알지 않는다.
- PPO와 규칙 기반 정책은 동일한 8개 행동을 사용한다.
- `NO_COMM`과 `REOBSERVE`는 HTML의 설명대로 8개 정책 행동 바깥의 공통 사전 게이트로 뒀다. 어댑터는 이때 행동 ID 대신 `None`을 반환한다.
- 알 수 없는 추가 입력 필드는 무시하므로 팀원 모듈의 observation 전체를 그대로 넘길 수 있다. 필수인 `event_id`가 없거나 확률·거리 범위가 잘못되면 즉시 예외를 낸다.

### 작업 요구사항

- 대상 미확정: 전송 후보를 평가하지 않고 `REOBSERVE`.
- 같은 사건 처리 중이며 장면 변화 없음: `NO_COMM`.
- 유사 물체 없음: 작업에 `E`, `N` 필요.
- 유사 물체 있음: 작업에 `E`, `N`, `C`, `I`가 필요. 단, 필요 정보는 반드시 전송할 정보가 아니며 B의 기존 지식과 비용에 따라 생략 가능하다.

### 결과 예측 모델

HTML은 비용 가중치와 일부 예시 결과를 제공하지만 완전한 확률식을 제공하지 않는다. 따라서 다음의 초기 모델을 사용했다.

1. 후보에 N/C가 없으면 `1 - b_N`, `1 - b_C` 확률로 해당 정보를 요청하고 작은 후속 payload를 받는다.
2. I를 보내면 crop 식별 성공률 `r_crop = 0.95`를 사용한다.
3. I를 생략했지만 B가 외형을 보유한 경우 `p_match = 0.9`를 사용한다.
4. B가 외형을 보유하지 않은 경우 `min(1, sigma / distance)` 확률로 모호함을 알아차려 crop을 요청한다.
5. 요청하지 않은 경우 위치만으로 맞출 성공률을 `distance / (distance + 2*sigma)`로 둔다.
6. 기존 외형 매칭 실패 중 `phi = 0.3`은 확신한 오조작, 나머지는 재관측으로 처리하며 재관측 뒤 미해결률은 0.1로 뒀다.
7. crop을 보낸 뒤 남는 5% 실패는 시나리오 ④에 적힌 오조작 약 2.3%, 미해결 약 2.7%로 분리하고 재관측 비용도 반영했다.
8. 후속 요청으로 실제 전송될 N/C/I 바이트도 기대 통신비에 포함했다.

이 식은 `distance=4 cm`, `sigma=3 cm`에서 외형 보유 확률이 0.9이면 약 1.0, 0.72이면 약 1.64의 비용을 만들고, 외형 지식이 없으면 crop 선전송 비용 약 3.67이 crop 생략보다 작아진다. 문서의 0.90, 1.51, 3.68을 상수로 넣은 것은 아니다. 문서 예시보다 조금 큰 이유는 요청으로 뒤늦게 받은 crop의 잔여 실패 위험과 후속 crop 전송량까지 같은 기준으로 비용에 포함했기 때문이다.

`actual_sigma`, 실제 지식 보유 여부, 가림 여부는 정책 입력에 넣지 않았다. 이것들은 평가 환경의 정답이며 정책에 제공하면 정보 누출이 되기 때문이다. A가 같은 `event`와 `belief`를 받으면 실제로 가림이 있었더라도 같은 결정을 한다.

## 확인한 시나리오

| 시나리오 | 입력 요약 | 선택 |
| --- | --- | --- |
| ① 첫 사건 | `b_N=0`, 유사품 없음 | `E + N` (160 B) |
| ② 지식 재사용 | `b_N=1`, 유사품 없음 | `E` (120 B) |
| ② 유사품 등장 | `b_N=1`, `b_I=0.9`, 4 cm/3 cm | `E + C` (160 B) |
| ③ 지나가며 관측 | `b_N=b_I=0.72`, 4 cm/3 cm | `E + N + C` (200 B) |
| ④ 지식 없는 모호 장면 | `b_N=b_I=0`, 4 cm/3 cm | `E + N + C + I` (30,920 B) |
| ④ 오차 과소추정 | 보고 sigma 1 cm, 실제값은 정책에 숨김 | `E + N + C` |

정확한 수치 일치보다 문서가 요구한 선택 분기와 비용 순서를 회귀 테스트로 고정했다.

## 실행 방법

외부 패키지는 필요 없다.

```bash
python -m unittest discover -s tests -v
python main.py observed
python main.py ambiguous
```

## 다른 팀 모듈 연결점

### 상대 지식 추정 모듈

다음 매핑을 그대로 `select()`에 전달하면 된다.

```python
belief = {"b_N": 0.72, "b_I": 0.72, "b_C": 0.0}
```

버전과 epoch의 유효성 판단은 상대 지식 모듈이 끝낸 뒤 유효한 확률만 넘기는 것으로 정했다. 예를 들어 layout version이 바뀌면 그 모듈이 `b_N=0`으로 무효화한다.

### PPO 모듈

`RulePolicyAdapter.predict(observation)`은 `(action_id, decision)`을 반환한다. 행동은 문서와 같은 0~7이고 `ACTIONS` 상수로 공개했다. PPO의 observation에는 아래 필드를 같은 이름으로 두면 된다.

```text
b_N, b_I, b_C, target_confirmed, has_similar_objects,
object_distance_cm, position_uncertainty_cm
```

### 통신/로봇 모듈

통신 담당은 `selected_items`에 든 실제 payload만 조립한다. ACK, 재전송, 직렬화, 실제 byte 측정은 이 모듈의 책임 밖이다. 다만 정책 비교 평가에서는 최초 TASK뿐 아니라 ACK·추가 요청·응답·재전송을 모두 별도 집계해야 한다.

현재 `estimated_bytes`는 HTML의 고정 가정 크기다. 실제 serializer가 정해지면 `MessageSizes` 또는 `ExpectedCostModel.message_size()`를 교체하면 된다.

## 다음 보정 작업

- 실험으로 `crop_match_success`, `appearance_match_success`, `wrong_fraction_on_match_failure`를 교체한다.
- 거리/오차 기반 요청 및 위치 매칭 함수를 실제 측정 곡선이나 lookup table로 교체한다.
- 요청 메시지와 ACK 자체의 byte 수를 후속 통신량 모델에 추가한다. 현재 비용 가중치의 왕복 1회 비용에는 지연과 제어 메시지가 묶여 있다.
- 시뮬레이터에서 정책에 숨긴 `actual_has_N`, `actual_has_I`, `actual_sigma`를 사용해 성공률·오조작률·추가 요청률을 평가한다.
- 팀 공통 serializer와 event/version 스키마가 확정되면 `policy_adapter.py` 한 곳에서 변환한다.

