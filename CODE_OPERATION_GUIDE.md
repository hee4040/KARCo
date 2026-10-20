# 규칙 기반 최소 통신 선택 코드 구현 및 작동 원리

## 1. 이 코드가 하는 일

이 모듈은 Robot A가 Robot B에게 데이터를 직접 전송하는 코드가 아니다. A가 발견한 사건과 A가 추정한 B의 지식 상태를 입력받아, 다음 네 정보 중 무엇을 보내는 것이 가장 효율적인지 결정한다.

| 기호 | 의미 | 기본 크기 |
| --- | --- | ---: |
| `E` | 사건 ID, 장소, 대상, 이상 종류 등의 사건 정보 | 120 B |
| `N` | 대상의 정상 위치와 방향 | 40 B |
| `C` | 대상의 현재 위치와 위치 오차 | 40 B |
| `I` | 대상을 식별하기 위한 crop 이미지 | 30,720 B |

정책의 기본 호출 형태는 다음과 같다.

```python
from rule_based_comm import RuleBasedCommunicationPolicy

policy = RuleBasedCommunicationPolicy()
decision = policy.select(event, belief)
```

정책은 다음 세 행동 중 하나를 반환한다.

- `SEND`: 선택한 정보를 통신 모듈에 전달한다.
- `NO_COMM`: 같은 사건을 이미 처리하고 있고 장면 변화도 없으므로 보내지 않는다.
- `REOBSERVE`: A도 대상을 확정하지 못했으므로 잘못된 정보를 보내지 않고 다시 관측한다.

`SEND`인 경우 E를 반드시 포함하는 8개 조합을 모두 비교한다.

```text
0: E
1: E + N
2: E + C
3: E + I
4: E + N + C
5: E + N + I
6: E + C + I
7: E + N + C + I
```

가장 작은 메시지를 무조건 고르는 것이 아니라, 메시지를 생략했을 때 발생할 추가 요청과 작업 실패 위험까지 비용으로 환산해 총 기대비용이 가장 작은 조합을 고른다.

---

## 2. 전체 처리 흐름

```text
event + belief 입력
        │
        ▼
입력값 검증 및 데이터 객체 변환
        │
        ▼
공통 게이트 검사
 ├─ 처리 중이고 변화 없음 ──> NO_COMM
 ├─ 대상 미확정 ───────────> REOBSERVE
 └─ 전송 판단 가능
        │
        ▼
현재 작업에 필요한 정보 분석
        │
        ▼
E/N/C/I 후보 8개 생성
        │
        ▼
각 후보의 결과 확률 예측
        │
        ▼
각 후보의 총 기대비용 계산
        │
        ▼
비용이 가장 낮은 후보 선택
        │
        ▼
SEND + selected_items + 후보별 비용 반환
```

코드는 이 처리 단계를 파일별로 분리했다. 따라서 이후 확률모델이나 메시지 크기가 바뀌어도 전체 정책을 다시 작성할 필요가 없다.

---

## 3. 입력 데이터 구현

입력과 출력 데이터 구조는 `rule_based_comm/schemas.py`에 정의되어 있다.

### 3.1 사건 정보 `Event`

```python
@dataclass(frozen=True)
class Event:
    event_id: str
    spot_id: str = ""
    object_id: str = ""
    anomaly_type: str = "misplaced"
    target_confirmed: bool = True
    has_similar_objects: bool = False
    similar_object_count: int = 0
    object_distance_cm: float | None = None
    position_uncertainty_cm: float | None = None
    already_processing: bool = False
    scene_changed: bool = False
```

중요 필드는 다음과 같다.

| 필드 | 정책에서 사용하는 방법 |
| --- | --- |
| `target_confirmed` | `False`이면 후보를 계산하지 않고 `REOBSERVE` |
| `has_similar_objects` | `True`이면 현재 위치 C와 식별 정보 I의 필요성을 계산 |
| `object_distance_cm` | 유사 물체 간 거리로 사용 |
| `position_uncertainty_cm` | A가 보고한 위치 오차 `sigma`로 사용 |
| `already_processing` | 같은 사건이 이미 진행 중인지 검사 |
| `scene_changed` | 처리 중인 사건이라도 다시 판단해야 하는 변화가 있었는지 검사 |

거리와 위치 오차가 0 이하이면 잘못된 입력으로 처리한다. 실제 오차인 `actual_sigma`는 정책에 넣지 않는다. 정책은 결정 시점에 A가 알고 있는 보고 오차만 사용할 수 있기 때문이다.

### 3.2 상대 지식 추정값 `BeliefState`

```python
@dataclass(frozen=True)
class BeliefState:
    b_N: float = 0.0
    b_I: float = 0.0
    b_C: float = 0.0
```

각 값은 0에서 1 사이여야 한다.

- `b_N`: B가 유효한 정상 위치를 알고 있을 확률
- `b_I`: B가 유효한 외형 정보를 알고 있을 확률
- `b_C`: B가 유효한 현재 위치를 알고 있을 확률

이 모듈은 이 확률을 직접 계산하지 않는다. 상대 지식 추정 담당자의 출력값을 그대로 사용한다.

`select()`는 dataclass뿐 아니라 일반 dictionary도 받을 수 있다.

```python
event = {
    "event_id": "e3",
    "target_confirmed": True,
    "has_similar_objects": True,
    "object_distance_cm": 4.0,
    "position_uncertainty_cm": 3.0,
}

belief = {
    "b_N": 0.72,
    "b_I": 0.72,
    "b_C": 0.0,
}
```

팀 공통 observation에 다른 필드가 들어 있어도 이 정책에 필요한 필드만 추출한다.

---

## 4. STEP 1: 공통 게이트와 필요 정보 분석

`rule_based_comm/requirements.py`의 `determine_requirements()`가 담당한다.

### 4.1 동일 사건 억제

```python
if event.already_processing and not event.scene_changed:
    return NO_COMM
```

같은 사건이 이미 처리 중이고 현장도 달라지지 않았다면 같은 TASK를 다시 만들 필요가 없다. 이 처리는 규칙 기반 정책과 PPO가 공통으로 사용하는 사건 게이트다.

장면이 변했다면 `scene_changed=True`이므로 처리 중인 사건이라도 다시 후보를 평가한다.

### 4.2 A의 대상 미확정

```python
if not event.target_confirmed:
    return REOBSERVE
```

A가 P와 P′ 중 어느 것이 대상인지 모르는 상태에서 crop을 보내면 잘못된 라벨이 B에게 전달될 수 있다. 따라서 메시지 선택 전에 A가 다른 시점에서 다시 관측하도록 한다.

### 4.3 작업에 필요한 정보 판단

유사 물체가 없는 기본 사건은 다음 정보가 필요하다고 본다.

```text
required = E, N
```

유사 물체가 존재하면 다음처럼 확장한다.

```text
required = E, N, C, I
```

여기서 `required`는 반드시 전송해야 한다는 뜻이 아니다. B가 N이나 I를 이미 가지고 있을 수 있으므로, 실제 전송 여부는 이후 기대비용 계산에서 결정한다.

---

## 5. STEP 2: 메시지 후보 생성

`rule_based_comm/candidates.py`는 E를 필수로 두고 N, C, I의 부분집합을 만든다.

개념적으로 다음 코드와 같다.

```python
for count in range(4):
    for combo in combinations(("N", "C", "I"), count):
        candidate = {"E", *combo}
```

그 결과 정확히 8개 후보가 생성된다. 후보는 `frozenset`으로 표현해 비용 계산 과정에서 실수로 변경되지 않도록 했다.

반환 결과에서는 항상 `E, N, C, I` 순서로 정렬한다. 따라서 set의 내부 순서와 관계없이 통신 모듈과 테스트가 동일한 출력을 받는다.

---

## 6. STEP 3: 메시지 크기와 설정값

모든 조정 가능한 값은 `rule_based_comm/config.py`에 모았다.

```python
MessageSizes(E=120, N=40, C=40, I=30_720)
```

비용 가중치는 다음과 같다.

```python
CostWeights(
    transmission_per_kib=0.1,
    request_roundtrip=1.0,
    wrong_action=12.0,
    reobservation=2.0,
    unresolved=10.0,
)
```

결과 예측에 사용하는 초기 확률은 다음과 같다.

```python
OutcomeParameters(
    crop_match_success=0.95,
    crop_wrong_fraction_on_failure=0.46,
    appearance_match_success=0.90,
    wrong_fraction_on_match_failure=0.30,
    unresolved_after_reobservation=0.10,
)
```

이 값들은 HTML 시나리오에 있는 실측 전 가정값이다. 나중에 로봇 실험 결과가 나오면 이 설정만 교체할 수 있다.

예를 들어 crop 식별 성공률이 실험에서 0.92로 측정되면 다음처럼 별도 설정을 넣을 수 있다.

```python
from rule_based_comm.config import OutcomeParameters, PolicyConfig

config = PolicyConfig(
    outcome=OutcomeParameters(crop_match_success=0.92)
)
policy = RuleBasedCommunicationPolicy(config)
```

---

## 7. STEP 4: 후보를 선택했을 때의 결과 예측

핵심 함수는 `ExpectedCostModel.predict_outcome()`이다. 이 함수는 후보를 선택했을 때 예상되는 다음 값을 계산한다.

```text
expected_requests        예상 추가 요청 횟수
expected_reobservations  예상 재관측 횟수
p_wrong                  잘못된 물체를 조작할 확률
p_unresolved             작업이 미해결로 끝날 확률
expected_followup_bytes  추가 요청 뒤 보내게 될 예상 데이터 크기
```

### 7.1 N과 C가 없는 경우

작업에 N이 필요한데 후보에 N이 없다고 하자.

```python
p_missing_N = 1 - b_N
```

`b_N=0.72`라면 B가 N을 모를 확률은 0.28이다. 따라서 예상 요청 횟수에 0.28을 더하고, 후속 N 응답 40 B도 다음처럼 반영한다.

```text
expected_requests += 0.28
expected_followup_bytes += 0.28 × 40 B
```

C도 같은 방식으로 `1-b_C`를 사용한다.

N이나 C를 후보에 포함했다면 정상적으로 수신한다는 전제에서 해당 정보의 부족 확률은 0이 된다. 패킷 유실과 ACK 재전송은 공통 통신 프로토콜에서 처리한다.

### 7.2 I를 보낸 경우

I를 보냈다고 해서 식별 성공률을 1로 두지는 않는다. HTML의 crop 식별 성공률 `r_crop=0.95`를 사용한다.

```text
crop 실패 확률 = 1 - 0.95 = 0.05
```

시나리오 ④의 설명에 맞춰 이 5%를 다음처럼 나눈다.

```text
오조작 확률 = 약 0.023
미해결 확률 = 약 0.027
재관측 기대 횟수 = 0.05
```

따라서 이미지를 보냈어도 잔여 위험 비용이 남는다.

### 7.3 I를 보내지 않았지만 B가 외형을 아는 경우

B가 외형을 가지고 있는 경로에서는 `appearance_match_success=0.9`를 사용한다.

```text
기존 외형 매칭 실패 확률 = 0.1
```

실패 중 `phi=0.3`은 B가 잘못된 대상을 확신하고 조작한다고 가정한다. 나머지는 재관측으로 넘어가고, 재관측 후에도 일부가 미해결로 남는다.

### 7.4 I를 보내지 않았고 B도 외형을 모르는 경우

이 경우 B가 현장의 모호함을 알아차려 crop을 요청할지, 위치 정보만으로 대상을 선택할지 예측해야 한다.

초기 구현에서는 다음 식을 사용한다.

\[
P_{request}=\min\left(1,\frac{\sigma}{distance}\right)
\]

거리 4 cm, 위치 오차 3 cm라면 다음과 같다.

```text
crop 요청 확률 = 3 / 4 = 0.75
```

요청하지 않고 위치만 사용하는 경우의 성공률은 다음과 같이 두었다.

\[
P_{position}=\frac{distance}{distance+2\sigma}
\]

같은 조건에서는 다음과 같다.

```text
위치만 사용한 성공률 = 4 / (4 + 2×3) = 0.4
```

오차가 1 cm로 보고되면 요청 확률은 0.25, 위치 성공률은 약 0.67이 된다. 이 때문에 정책은 위치만으로도 충분하다고 판단해 I를 생략할 수 있다. 실제 오차가 3 cm였다면 정책이 알 수 없는 오차 과소추정 실패가 발생한다.

이 두 식은 초기 규칙이며 최종 실험에서는 로봇 측정 결과로 만든 lookup table이나 보정 곡선으로 교체해야 한다.

### 7.5 `b_I`를 이용한 두 경로 결합

I를 생략했을 때 결과는 다음 두 경로를 확률적으로 섞어 계산한다.

```text
b_I       × 외형을 보유한 경우의 결과
(1 - b_I) × 외형을 보유하지 않은 경우의 결과
```

예를 들어 `b_I=0.72`라면 기존 외형 매칭 경로를 72%, 외형 부족 경로를 28% 반영한다.

이 구조 때문에 `b_I`가 높아질수록 I 생략 비용이 내려가고, `b_I`가 낮아질수록 crop 선전송이 유리해진다.

---

## 8. STEP 5: 총 기대비용 계산

`ExpectedCostModel.evaluate()`는 예측된 결과를 다음 비용 함수에 넣는다.

\[
J(m)=C_{initial}+C_{followup}+C_{request}+C_{wrong}+C_{reobserve}+C_{unresolved}
\]

각 항목은 코드에서 다음과 같이 계산한다.

```python
initial_tx = 0.1 * initial_bytes / 1024
followup_tx = 0.1 * expected_followup_bytes / 1024
request_cost = expected_requests * 1.0
wrong_cost = p_wrong * 12.0
reobserve_cost = expected_reobservations * 2.0
unresolved_cost = p_unresolved * 10.0

total_cost = (
    initial_tx
    + followup_tx
    + request_cost
    + wrong_cost
    + reobserve_cost
    + unresolved_cost
)
```

중요한 점은 최초 TASK의 전송량만 계산하지 않는다는 것이다. 정보를 생략했다가 B가 추가로 요청하면 나중에 보내게 될 N/C/I의 예상 바이트도 `followup_tx`에 포함한다.

비용 결과는 `CandidateCost`에 다음처럼 분리해 저장한다.

```text
items
initial_bytes
transmission_cost
followup_transmission_cost
request_cost
wrong_cost
reobservation_cost
unresolved_cost
total_cost
```

따라서 어떤 후보가 왜 선택됐는지 확인할 수 있다.

---

## 9. STEP 6: 최소비용 후보 선택

`rule_based_comm/rule_policy.py`의 `select()`가 전체 과정을 조합한다.

```python
costs = tuple(
    cost_model.evaluate(candidate, event, belief, requirements)
    for candidate in generate_candidates()
)

best = min(
    enumerate(costs),
    key=lambda pair: (
        pair[1].total_cost,
        pair[1].initial_bytes,
        pair[0],
    ),
)[1]
```

우선순위는 다음과 같다.

1. 총 기대비용이 가장 작은 후보
2. 비용이 같다면 초기 전송량이 더 작은 후보
3. 비용과 크기가 모두 같다면 고정된 행동 순서에서 앞선 후보

고정된 tie-break 규칙을 사용하므로 같은 입력에는 언제나 같은 결과가 나온다.

모든 후보 비용은 낮은 순서로 정렬해 `candidate_costs`에 넣는다. 최종 선택만 반환하는 것이 아니라 선택 근거까지 남기는 이유는 다음과 같다.

- 실험 중 비용모델 오류를 찾기 쉽다.
- PPO의 선택과 규칙 정책의 선택을 비교할 수 있다.
- 발표나 보고서에서 특정 메시지가 선택된 이유를 설명할 수 있다.

---

## 10. 실제 시나리오의 작동 예

### 10.1 첫 사건

```python
event = {
    "event_id": "e1",
    "target_confirmed": True,
    "has_similar_objects": False,
}
belief = {"b_N": 0.0, "b_I": 0.0}
```

유사 물체가 없으므로 작업 요구사항은 E와 N이다.

- E만 전송: B가 N을 모를 확률이 1이므로 추가 요청 비용 발생
- E+N 전송: N은 40 B라서 전송비가 약 0.0039에 불과함

결과는 다음과 같다.

```text
SEND: E + N
estimated_bytes: 160
```

### 10.2 확인된 정상 위치 재사용

```python
belief = {"b_N": 1.0, "b_I": 0.9}
```

유사 물체가 없고 B가 같은 버전의 N을 확실히 안다면 N 생략 비용이 0이다.

```text
SEND: E
estimated_bytes: 120
```

### 10.3 외형 기억이 있는 유사 물체

```python
belief = {"b_N": 1.0, "b_I": 0.9, "b_C": 0.0}
distance = 4.0
sigma = 3.0
```

- N은 기존 지식을 사용해 생략한다.
- C는 B가 모르므로 작은 40 B를 보낸다.
- I는 B가 외형을 기억할 확률이 높고 30 KiB이므로 생략한다.

```text
SEND: E + C
estimated_bytes: 160
expected_cost: 약 1.0
```

### 10.4 지나가며 관측했을 가능성이 있는 경우

```python
belief = {"b_N": 0.72, "b_I": 0.72, "b_C": 0.0}
distance = 4.0
sigma = 3.0
```

- N은 40 B만 추가하면 0.28의 요청 위험을 제거할 수 있어 전송한다.
- C도 작업에 필요하고 B가 모르므로 전송한다.
- I는 30 KiB를 매번 먼저 보내는 것보다 필요할 때 요청하도록 두는 기대비용이 낮아 생략한다.

```text
SEND: E + N + C
estimated_bytes: 200
expected_cost: 약 1.64
```

HTML의 예시 1.51보다 조금 큰 이유는 이 구현이 요청으로 뒤늦게 받은 crop의 전송량과 crop을 받은 후에도 남는 실패 위험을 함께 계산하기 때문이다.

### 10.5 외형 지식이 없는 모호한 장면

```python
belief = {"b_N": 0.0, "b_I": 0.0, "b_C": 0.0}
distance = 4.0
sigma = 3.0
```

I를 생략하면 B가 crop을 요청할 가능성이 매우 높고, 요청하지 않을 경우 위치만으로 잘못된 대상을 선택할 위험도 있다. 이 비용이 crop 선전송 비용보다 커진다.

```text
SEND: E + N + C + I
estimated_bytes: 30,920
expected_cost: 약 3.67
```

---

## 11. 반환값을 읽는 방법

`PolicyDecision`은 다음 정보를 가진다.

```python
decision.action
decision.selected_items
decision.estimated_bytes
decision.expected_cost
decision.reason
decision.candidate_costs
```

JSON 형태가 필요하면 다음처럼 변환한다.

```python
result = decision.as_dict()
```

예시 출력은 다음과 같다.

```json
{
  "action": "SEND",
  "selected_items": ["E", "N", "C"],
  "estimated_bytes": 200,
  "expected_cost": 1.64,
  "reason": "minimum expected cost among 8 E/N/C/I candidates",
  "candidate_costs": [
    {
      "items": ["E", "N", "C"],
      "initial_bytes": 200,
      "transmission_cost": 0.0195,
      "request_cost": 0.21,
      "wrong_cost": 0.4684,
      "reobservation_cost": 0.1806,
      "unresolved_cost": 0.1365,
      "followup_transmission_cost": 0.63,
      "total_cost": 1.645
    }
  ]
}
```

실제 출력에는 나머지 7개 후보 비용도 모두 포함된다.

---

## 12. 다른 팀 코드와 연결하는 방법

### 12.1 상대 지식 추정 모듈

상대 지식 담당자가 다음 형식으로 확률을 넘기면 된다.

```python
belief = belief_estimator.estimate(event_id, object_id, versions)

decision = policy.select(event, belief)
```

버전이 오래됐는지, memory epoch가 변경됐는지는 상대 지식 모듈에서 판단한다. 예를 들어 layout version이 달라졌다면 유효하지 않은 정상 위치이므로 `b_N=0`으로 만들어 전달한다.

### 12.2 PPO 모듈

`rule_based_comm/policy_adapter.py`는 규칙 정책의 결과를 PPO와 같은 행동 번호로 변환한다.

```python
from rule_based_comm.policy_adapter import RulePolicyAdapter

action_id, decision = RulePolicyAdapter().predict(observation)
```

예를 들면 다음과 같다.

```text
E             -> 0
E + N         -> 1
E + C         -> 2
E + N + C     -> 4
E + N + C + I -> 7
```

`NO_COMM`과 `REOBSERVE`는 공통 게이트이므로 `action_id=None`을 반환한다. PPO도 같은 게이트를 통과한 사건에 대해서만 0~7 중 하나를 고르게 하면 공정하게 비교할 수 있다.

### 12.3 통신 모듈

통신 담당 코드는 `selected_items`만 보고 실제 payload를 조립한다.

```python
decision = policy.select(event, belief)

if decision.action.value == "SEND":
    payload = message_builder.build(
        selected_items=decision.selected_items,
        event=event,
        normal=normal_data,
        current=current_data,
        image=crop_image,
    )
    communicator.send(payload)
```

이 정책은 좌표나 이미지를 직접 생성하지 않는다. 어떤 항목을 메시지에 넣을지만 결정한다.

---

## 13. 테스트 구성

테스트는 외부 라이브러리 없이 Python `unittest`로 작성했다.

### `tests/test_requirements_and_candidates.py`

- 후보가 정확히 8개인지 확인
- 모든 후보가 E를 포함하는지 확인
- 동일 사건 억제 확인
- 대상 미확정 시 재관측 확인
- 유사 물체가 있을 때 작업 요구사항 확인

### `tests/test_scenarios.py`

- 시나리오 ①: E+N
- 시나리오 ②: E
- 시나리오 ② 유사품: E+C
- 시나리오 ③: E+N+C
- 시나리오 ④: E+N+C+I
- 오차 과소추정: E+N+C
- 같은 정책 입력이면 실제 가림 여부와 무관하게 같은 결정
- 8개 후보 비용이 모두 반환되고 정렬되는지 확인

### `tests/test_adapter_and_validation.py`

- PPO 행동 ID 변환 확인
- 공통 게이트가 8개 행동 밖에 있는지 확인
- 잘못된 확률과 거리 입력 거부 확인

테스트 실행 명령은 다음과 같다.

```bash
python -m unittest discover -s tests -v
```

현재 17개 테스트가 모두 통과한다.

대표 시나리오의 전체 후보 비용을 보고 싶으면 다음 명령을 실행한다.

```bash
python main.py first
python main.py reuse
python main.py observed
python main.py ambiguous
```

---

## 14. 현재 모델의 한계

현재 구현은 로봇 실험 전에 정책 구조와 시나리오 분기를 검증하기 위한 초기 모델이다.

1. `sigma/distance` 요청 확률은 실측식이 아니다.
2. 위치만 사용한 식별 성공률도 초기 휴리스틱이다.
3. 메시지를 보냈을 때 정상적으로 수신된다고 가정한다. ACK 유실과 재전송은 공통 프로토콜 영역이다.
4. N, C, I의 실패 위험을 대부분 독립적으로 합산한다. 실제 환경에서는 상관관계가 있을 수 있다.
5. `b_C`는 입력받지만 현재 시나리오에서는 주로 N과 I가 핵심이다.
6. 현재 메시지 크기는 실제 serializer로 측정한 크기가 아니다.
7. 요청 제어 메시지와 ACK 자체의 바이트는 왕복 비용 1에 묶여 있고 별도 byte 합계에는 아직 포함되지 않는다.

이 한계 때문에 기대비용 숫자를 최종 실험 결과로 사용해서는 안 된다. 현재 단계에서 중요한 결과는 다음 두 가지다.

- 8개 후보를 같은 기준으로 비교할 수 있는 구조가 완성됐다.
- 실측 확률과 메시지 크기를 넣을 위치가 코드에서 분리되어 있다.

---

## 15. 실험 데이터가 나온 뒤 수정할 부분

실험 단계에서는 다음 순서로 교체하면 된다.

1. `config.py`의 crop 및 외형 매칭 성공률을 실측값으로 변경한다.
2. `cost_model.py`의 `_missing_appearance_outcome()`에 있는 거리/오차 식을 측정 곡선으로 교체한다.
3. 실제 serializer가 정해지면 `MessageSizes`를 실측 크기로 바꾸거나 `message_size()`를 serializer 측정 함수와 연결한다.
4. ACK, INFO_REQUEST, INFO_RESPONSE, 재전송까지 포함한 실제 통신량을 평가 환경에서 기록한다.
5. `actual_has_N`, `actual_has_I`, `actual_sigma`는 정책에 제공하지 않고 평가 정답으로만 사용한다.
6. 같은 환경과 비용 함수에서 규칙 정책과 PPO 정책을 비교한다.

정책의 외부 인터페이스인 `policy.select(event, belief)`는 유지할 수 있으므로 내부 확률모델을 바꾸더라도 다른 팀 코드의 수정 범위를 최소화할 수 있다.
