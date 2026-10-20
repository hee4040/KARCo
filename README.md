# Rule-Based Minimum Communication Policy

로봇 A가 발견한 사건을 로봇 B에게 전달할 때, **작업 성공에 필요한 정보만 최소 비용으로 선택**하는 규칙 기반 통신 정책입니다.

단순히 가장 작은 메시지를 고르는 것이 아니라 다음 비용을 함께 고려합니다.

- 최초 전송량
- B가 정보 부족으로 다시 요청할 가능성과 후속 전송량
- 잘못된 물체를 조작할 위험
- 재관측 비용
- 끝내 작업을 해결하지 못할 위험

현재 구현은 실제 통신이나 로봇 제어를 수행하지 않습니다. 사건 정보와 B의 지식 상태 추정값을 입력받아 `E/N/C/I` 중 보낼 항목, 예상 비용, 후보별 판단 근거를 반환하는 **의사결정 모듈**입니다.

## 핵심 개념

정책이 선택하는 정보는 다음 네 종류입니다.

| 기호 | 의미 | 기본 크기 |
| --- | --- | ---: |
| `E` | 사건 ID, 장소, 대상, 이상 종류 등의 사건 정보 | 120 B |
| `N` | 대상의 정상 위치와 방향 | 40 B |
| `C` | 대상의 현재 위치와 위치 오차 | 40 B |
| `I` | 대상을 식별하기 위한 crop 이미지 | 30,720 B |

`E`는 모든 전송에 반드시 포함됩니다. 정책은 아래 8개 조합을 전부 평가한 뒤 총 기대비용이 가장 낮은 후보를 고릅니다.

| 행동 ID | 전송 항목 | 행동 ID | 전송 항목 |
| ---: | --- | ---: | --- |
| 0 | `E` | 4 | `E + N + C` |
| 1 | `E + N` | 5 | `E + N + I` |
| 2 | `E + C` | 6 | `E + C + I` |
| 3 | `E + I` | 7 | `E + N + C + I` |

최종 행동은 다음 셋 중 하나입니다.

- `SEND`: 8개 후보 중 하나를 선택해 전송합니다.
- `NO_COMM`: 같은 사건이 이미 처리 중이고 장면 변화가 없어 중복 전송을 막습니다.
- `REOBSERVE`: 송신 로봇도 대상을 확정하지 못해 메시지를 만들기 전에 다시 관측합니다.

`NO_COMM`과 `REOBSERVE`는 8개 전송 행동 바깥의 공통 사전 게이트입니다.

## 처리 흐름

```text
Event + BeliefState
        │
        ▼
입력 검증
        │
        ▼
공통 게이트 ── 대상 미확정 ──────────────> REOBSERVE
        │       처리 중 + 장면 변화 없음 ─> NO_COMM
        ▼
상황별 필요 정보 분석
        │
        ▼
E를 포함한 8개 후보의 결과 확률·비용 계산
        │
        ▼
최소 기대비용 후보 선택
        │
        ▼
SEND + 선택 항목 + 전체 후보별 진단 결과
```

유사 물체가 없는 일반 상황에서는 `E`, `N`이 필요하다고 판단합니다. 유사 물체가 있으면 `C`, `I`도 필요한 정보로 간주합니다. 여기서 "필요"는 항상 전송한다는 뜻이 아닙니다. B가 이미 해당 정보를 알고 있을 확률과 추가 요청·실패 비용을 비교해 생략할 수 있습니다.

## 빠른 시작

### 요구 사항

- Python 3.10 이상
- 런타임 외부 의존성 없음

저장소 루트에서 바로 실행할 수 있습니다.

```bash
python main.py observed
```

지원하는 데모 시나리오는 다음과 같습니다.

```bash
python main.py first       # 첫 사건: E + N
python main.py reuse       # B의 기존 지식 재사용: E
python main.py observed    # 부분적으로 알고 있는 유사 물체: E + N + C
python main.py ambiguous   # 지식 없는 모호한 장면: E + N + C + I
```

패키지 형태로 설치해서 사용할 수도 있습니다.

```bash
python -m pip install -e .
```

## 기본 사용법

```python
from rule_based_comm import RuleBasedCommunicationPolicy

policy = RuleBasedCommunicationPolicy()

decision = policy.select(
    event={
        "event_id": "e3",
        "target_confirmed": True,
        "has_similar_objects": True,
        "object_distance_cm": 4.0,
        "position_uncertainty_cm": 3.0,
    },
    belief={
        "b_N": 0.72,
        "b_I": 0.72,
        "b_C": 0.0,
    },
)

print(decision.action)          # Action.SEND
print(decision.selected_items)  # ('E', 'N', 'C')
print(decision.estimated_bytes) # 200
print(decision.expected_cost)   # 약 1.645
```

`decision.as_dict()`를 호출하면 JSON으로 직렬화 가능한 결과를 얻습니다.

```python
{
    "action": "SEND",
    "selected_items": ["E", "N", "C"],
    "estimated_bytes": 200,
    "expected_cost": 1.6449912500000001,
    "reason": "minimum expected cost among 8 E/N/C/I candidates",
    "candidate_costs": [
        # 비용이 낮은 순서로 8개 후보의 세부 비용과 예상 결과가 들어갑니다.
    ],
}
```

## 입력 스키마

`select()`는 dataclass와 일반 dictionary를 모두 받을 수 있습니다. 팀 공통 observation을 그대로 넘겨도 정의되지 않은 추가 필드는 무시합니다.

### `Event`

| 필드 | 타입 / 기본값 | 설명 |
| --- | --- | --- |
| `event_id` | `str` / 필수 | 사건의 고유 ID. 빈 문자열은 허용하지 않음 |
| `spot_id` | `str` / `""` | 사건 발생 위치 ID |
| `object_id` | `str` / `""` | 대상 물체 ID |
| `anomaly_type` | `str` / `"misplaced"` | 이상 상황 종류 |
| `target_confirmed` | `bool` / `True` | A가 대상을 확정했는지 여부 |
| `has_similar_objects` | `bool` / `False` | 주변에 유사 물체가 있는지 여부 |
| `similar_object_count` | `int` / `0` | 유사 물체 수. 0 이상이어야 함 |
| `object_distance_cm` | `float \| None` | 유사 물체 간 거리. 입력 시 0보다 커야 함 |
| `position_uncertainty_cm` | `float \| None` | 보고된 위치 오차 `sigma`. 입력 시 0보다 커야 함 |
| `already_processing` | `bool` / `False` | 같은 사건이 이미 처리 중인지 여부 |
| `scene_changed` | `bool` / `False` | 처리 도중 장면이 바뀌었는지 여부 |

### `BeliefState`

| 필드 | 기본값 | 설명 |
| --- | ---: | --- |
| `b_N` | `0.0` | B가 유효한 정상 위치를 알고 있을 확률 |
| `b_I` | `0.0` | B가 유효한 외형 정보를 알고 있을 확률 |
| `b_C` | `0.0` | B가 유효한 현재 위치를 알고 있을 확률 |

모든 belief 값은 `0.0` 이상 `1.0` 이하여야 합니다. 이 모듈은 belief를 직접 추정하지 않으며, 상대 지식 추정 모듈이 만든 확률을 입력으로 사용합니다.

## 출력 이해하기

`PolicyDecision`의 주요 필드는 다음과 같습니다.

| 필드 | 의미 |
| --- | --- |
| `action` | `SEND`, `NO_COMM`, `REOBSERVE` 중 하나 |
| `selected_items` | 선택된 `E/N/C/I` 항목 |
| `estimated_bytes` | 최초 전송 payload의 추정 크기 |
| `expected_cost` | 선택 후보의 총 기대비용 |
| `reason` | 해당 행동을 반환한 이유 |
| `candidate_costs` | 8개 후보의 비용을 총비용 오름차순으로 정렬한 결과 |

각 후보 비용에는 최초·후속 전송비, 요청비, 오조작비, 재관측비, 미해결비와 다음 예상 결과가 포함됩니다.

- `expected_requests`: 예상 추가 요청 횟수
- `expected_reobservations`: 예상 재관측 횟수
- `p_wrong`: 잘못된 물체를 조작할 확률
- `p_unresolved`: 작업이 미해결로 끝날 확률
- `expected_followup_bytes`: 요청 뒤 추가로 보낼 것으로 예상되는 데이터 크기

## 비용 모델

후보의 총 기대비용은 아래 항목의 합입니다.

```text
총 기대비용
= 최초 전송비
+ 후속 전송비
+ 예상 요청 횟수 × 요청 왕복 비용
+ 오조작 확률 × 오조작 비용
+ 예상 재관측 횟수 × 재관측 비용
+ 미해결 확률 × 미해결 비용
```

기본 가중치와 메시지 크기는 `rule_based_comm/config.py`에 모여 있습니다. 실험 결과에 맞춰 설정 객체를 바꿔 주입할 수 있습니다.

```python
from rule_based_comm import PolicyConfig, RuleBasedCommunicationPolicy
from rule_based_comm.config import CostWeights, OutcomeParameters

config = PolicyConfig(
    costs=CostWeights(wrong_action=20.0),
    outcome=OutcomeParameters(crop_match_success=0.92),
)

policy = RuleBasedCommunicationPolicy(config)
```

현재 확률식과 가중치는 로봇 실험 전 시나리오를 재현하기 위한 초기 모델입니다. 특히 crop 성공률, 외형 매칭 성공률, 거리/오차 기반 요청 확률은 실제 측정값으로 보정해야 합니다.

## 대표 시나리오

기본 설정에서 회귀 테스트로 고정한 결과입니다.

| 상황 | 핵심 입력 | 선택 결과 |
| --- | --- | --- |
| 첫 사건 | `b_N=0`, 유사 물체 없음 | `E + N` / 160 B |
| 정상 위치 재사용 | `b_N=1`, 유사 물체 없음 | `E` / 120 B |
| 외형 기억이 있는 유사 물체 | `b_N=1`, `b_I=0.9`, 거리 4 cm, 오차 3 cm | `E + C` / 160 B |
| 부분 관측된 유사 물체 | `b_N=b_I=0.72`, 거리 4 cm, 오차 3 cm | `E + N + C` / 200 B |
| 지식 없는 모호한 장면 | `b_N=b_I=0`, 거리 4 cm, 오차 3 cm | `E + N + C + I` / 30,920 B |

## PPO 및 다른 모듈과 연결하기

### PPO 공통 행동 공간

`RulePolicyAdapter`는 규칙 기반 결정 결과를 PPO와 공유하는 행동 ID로 변환합니다.

```python
from rule_based_comm.policy_adapter import RulePolicyAdapter

action_id, decision = RulePolicyAdapter().predict({
    "event_id": "e1",
    "target_confirmed": True,
    "has_similar_objects": False,
    "b_N": 0.0,
    "b_I": 0.0,
    "b_C": 0.0,
})

print(action_id)               # 1
print(decision.selected_items) # ('E', 'N')
```

게이트 결과가 `NO_COMM` 또는 `REOBSERVE`이면 8개 전송 행동에 해당하지 않으므로 `action_id`는 `None`입니다.

### 상대 지식 추정 모듈

상대 지식 추정 모듈은 유효성 판단을 마친 `b_N`, `b_I`, `b_C`만 넘기면 됩니다. 예를 들어 layout version이 바뀌어 정상 위치 지식이 무효화되면 이 정책을 호출하기 전에 `b_N=0`으로 갱신합니다.

### 통신·로봇 모듈

통신 모듈은 `decision.selected_items`에 포함된 실제 payload를 조립하고 전송합니다. 직렬화, ACK, 패킷 재전송, 실제 byte 측정과 로봇 제어는 이 패키지의 책임 범위에 포함되지 않습니다.

## 프로젝트 구조

```text
.
├── main.py                         # 네 가지 대표 시나리오 CLI
├── rule_based_comm/
│   ├── schemas.py                  # 입력·출력 dataclass 및 검증
│   ├── requirements.py             # 공통 게이트와 필요 정보 분석
│   ├── candidates.py               # E를 포함한 8개 후보 생성
│   ├── config.py                   # 메시지 크기·비용·확률 설정
│   ├── cost_model.py               # 결과 확률과 기대비용 계산
│   ├── rule_policy.py              # 후보 평가 및 최종 선택
│   └── policy_adapter.py            # PPO 공통 행동 ID 어댑터
├── tests/                           # 검증·게이트·시나리오 회귀 테스트
├── IMPLEMENTATION.md                # 구현 결정과 가정 요약
└── CODE_OPERATION_GUIDE.md          # 비용 계산의 상세 작동 원리
```

## 테스트

```bash
python -m unittest discover -s tests -v
```

현재 테스트는 다음을 검증합니다.

- 입력값 범위 검증
- `NO_COMM`, `REOBSERVE` 게이트
- 8개 후보 생성과 정렬
- 대표 시나리오별 선택 결과와 예상 비용
- PPO 행동 ID 변환
- 정책에 공개되지 않은 실제 환경 정보가 결정에 누출되지 않는지 여부

## 현재 범위와 다음 작업

- `actual_sigma`, 실제 지식 보유 여부, 가림 여부처럼 정책이 결정 시점에 알 수 없는 정답 정보는 입력에서 제외했습니다.
- 현재 byte 수는 고정 가정입니다. 공통 serializer가 정해지면 `MessageSizes` 또는 `ExpectedCostModel.message_size()`를 교체해야 합니다.
- 거리/오차 기반 확률과 crop·외형 매칭 성공률은 실제 로봇 실험으로 보정해야 합니다.
- 현재 요청 왕복 비용에는 지연과 제어 메시지가 묶여 있습니다. 정밀 평가 시 ACK, 요청 메시지, 재전송량을 별도로 계측해야 합니다.

더 자세한 설계 판단은 [`IMPLEMENTATION.md`](./IMPLEMENTATION.md), 단계별 비용 계산은 [`CODE_OPERATION_GUIDE.md`](./CODE_OPERATION_GUIDE.md)에서 확인할 수 있습니다.
