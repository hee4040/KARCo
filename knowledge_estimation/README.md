# 상대 지식 추정 모듈

Robot A가 **현재 유효한 정보를 Robot B가 가지고 있을 확률**을 계산한다.
제공된 `적응형 통신 시나리오.html` v0.3과 저장소 README를 기준으로 만든
확률 기반 추정기와 실행형 2D 시뮬레이터다. Python 3.10 이상, 표준 라이브러리만 사용한다.

현재 사용 중인 구현은 [`probabilistic/`](probabilistic/README.md)에 있다.
[`learning/`](learning/README.md)은 학습용 데이터 준비 단계이며 모델/가중치는 아직 없다.
논문 원문과 수식·코드 대조 결과는 [확률 모델 검토](probabilistic/RESEARCH_REVIEW.md)에 있다.

```powershell
python -m knowledge_estimation.simulator
python -m knowledge_estimation.simulator --headless --scenario pass_occluded --seconds 90 --seed 7 --output knowledge_estimation/native_run.json
python -m unittest discover -s knowledge_estimation/tests -v
```

모든 명령은 저장소 루트에서 실행한다. ROS, PPO, 네트워크, GPU 설치가 필요 없다.
저장되는 JSON에는 시뮬레이션 상태와 전송 결정 직전의 추정·정답 기록이 들어간다.
다양한 조건의 학습 데이터는 아래 수집기로 생성할 수 있다. 지도학습 모델은 아직 구현하지 않았다.

```powershell
python -m knowledge_estimation.learning.dataset --families 30 --seconds 60 --output knowledge_estimation/data/synthetic_v2
```

30개 환경 묶음 × 4개 시나리오 = 120개 에피소드를 실행한다. 입력과 정답을
서로 다른 파일에 저장하고, 환경 묶음 단위로 학습/검증/시험을 분리한다.
수집 후 자동으로 인과적 입력·정답·분할·체크섬을 검사한다. 기존 출력 경로는
덮어쓰지 않으므로 다시 수집할 때는 `synthetic_v3` 등 새 이름을 지정한다.
필드 정의, 실험 조건과 한계는 [데이터 수집 문서](learning/dataset/README.md)에 있다.

## 별도 실행 창의 2D 시뮬레이터

```powershell
python -m knowledge_estimation.simulator
```

Windows에서는 `knowledge_estimation/run_simulator.bat`를 더블클릭해도 된다.
로봇 이동·관측·패킷 전송을 시간에 따라 실행하고, 지식 추정과 B의 실제 기억을
함께 표시한다. 실행 중 물체 이동·캐시 초기화·가림·수동 정보 전송을 조작할 수 있다.
웹 서버 없이 실행되며, 상세 실행법과 향후 ROS/Gazebo 연결 경계는
[2D 시뮬레이터 문서](simulator/README.md)에 정리했다.

## 파일 구성

| 파일 | 역할 |
|---|---|
| `probabilistic/estimator.py`, `probabilistic/probabilities.py` | 현재 사용 중인 확률 기반 추정기와 수식 |
| `probabilistic/visibility.py` | 이동 경로의 2D 관측 기회 계산 |
| `probabilistic/RESEARCH_REVIEW.md` | 논문 근거, 가정, 코드 대조와 수정 사항 |
| `types.py` | 항목·버전·추정 출력 등 공통 공개 계약 |
| `simulator/engine.py` | 로봇·물체·패킷의 시간 기반 시뮬레이션 |
| `simulator/desktop.py` | 독립 실행 창과 조작 |
| `simulator/bridge.py` | 추정기에 전달할 관측 이벤트와 어댑터 |
| `learning/dataset/` | 향후 학습을 위한 A 특징 추출, 반복 수집, 입력/정답 분리와 감사 |
| `data/` | 공통 생성 데이터. git 제외, v1은 검토 전 기록 / v2는 수정 후 기록 |
| `tests/` | 추정·통신·관측·시뮬레이터 검증 |

루트의 `estimator.py`, `visibility.py`, `dataset/`은 이전 import/명령을 연결하는
짧은 호환 파일이다. 실제 구현을 복제하지 않는다. 새 코드는 위 실제 경로를 사용한다.

## 모듈 경계

```text
A의 통신/작업/경로/관측 기록
         ↓
KnowledgeEstimator (수신 로봇마다 인스턴스 하나)
         ↓
{b_N, b_C, b_I} + 버전/epoch/근거
         ↓
팀원의 규칙 기반 또는 PPO 통신 선택 모듈
```

| 출력 | 정의 | 키/버전 |
|---|---|---|
| b_N | 올바른 정상 위치·방향을 보유할 확률 | 작업대 + 물체 / layout_version |
| b_C | 현재 유효한 위치를 보유할 확률 | 물체 / pose_version |
| b_I | 올바른 외형을 기억하고 사용할 수 있을 확률 | 물체 / appearance_version |

사건 E는 지식 항목이 아니다. E만 받은 ACK는 N/C/I를 올리지 않는다.
메시지 선택, 기대비용, 사건 중복 게이트, 실제 송수신·재전송 스케줄,
물체 식별기, 재검증 센서와 사건 종료 처리는 다른 모듈의 책임이다.
시뮬레이터의 메시지 조합은 설명용으로 지정한 것이며 정책의 최적성을 계산한 결과가 아니다.

## 팀원 코드에서 사용

```python
from knowledge_estimation import (
    Information as Info, KnowledgeEstimator, KnowledgeItem, KnowledgeKey,
)

estimator = KnowledgeEstimator("B", memory_epoch=7)
normal = KnowledgeItem(KnowledgeKey(Info.NORMAL, "P", "S1"), version=1)
appearance = KnowledgeItem(KnowledgeKey(Info.APPEARANCE, "P"), version=1)

# A가 알고 있는 최신 항목/버전 등록 (등록만으로 B가 안다고 판단하지 않음)
estimator.register(normal)
estimator.register(appearance)

# 반드시 전송하기 전에 저장: 이 값이 정책 입력/추정 정확도 평가 대상
decision = estimator.snapshot("P", "S1", now=10.0)
policy_input = decision.as_policy_input()  # {'b_N': 0.0, 'b_C': 0.0, 'b_I': 0.0}

# 팀원 정책이 E + N을 선택했다고 가정
estimator.record_send("msg-1", [normal], now=10.1)
estimator.observe_ack("msg-1", [normal], memory_epoch=7, now=10.2)

# B가 실제로 다룬 항목을 보고해도 A의 재검증 전에는 갱신하지 않음
estimator.record_result("event-1/attempt-1", [appearance], now=20.0)
estimator.verify_task("event-1/attempt-1", success=True, now=21.0)
assert estimator.snapshot("P", "S1").as_policy_input() == {
    'b_N': 1.0, 'b_C': 0.0, 'b_I': 0.9,
}
# decision은 그대로 0,0,0이다. ACK 이후 값으로 소급 평가하지 않는다.
```

`now`는 A가 이벤트를 **처리하는 시각**(시뮬레이션 초 또는 단조 시계)이다.
명시하면 감소하지 않아야 하고, 생략하면 마지막 처리 시각을 사용한다.
도착 순서가 뒤바뀐 센서 측정 시각을 그대로 넣지 말고 처리 시각과 구분한다.
실제 전송 시각/ACK deadline 판단은 통신 모듈에서 담당한다.
한 인스턴스는 순차 호출하며 동시 콜백에서는 외부 잠금을 사용한다.

## 갱신 규칙과 HTML 대응

| HTML 내용 | 호출 / 동작 |
|---|---|
| ① 처음 전송 | `record_send`: 항목·버전·내용 신뢰도·attempt를 기록, 즉시 확정하지 않음 |
| ① ACK | `observe_ack`: ACK에 적힌 저장 항목과 버전만 반영 |
| ① ACK 유실 | deadline 후 `observe_ack_timeout` |
| ① 재전송 | 같은 msg_id·payload, **다른 attempt_id**로 `record_send` |
| ①/②/③/④ 완료 보고 | `record_result`: 외형 학습 증거 보류 |
| A 재검증 성공 | `verify_task`: 명시된 learned_items에 q_learn 증거 추가 |
| ② 기억 초기화 | 수신한 STATUS/ACK의 memory_epoch 증가로 이전 지식 무효화 |
| ② 틀린 기억 재사용 | `revoke_evidence(task_id)`: 그 작업에서 얻은 증거만 제거 |
| ③ 경로 관측 | `record_pass_by`: 통과×시야×저장 확률 |
| ③ 정상 배치 확인 | N은 통과 전 정상 + `confirm_pass_by(normal_after=True)` 뒤 반영 |
| ③ 이상 중 통과 | 정상 확인 전 N은 0 유지, normal_after=False이면 근거 제외 |
| ③ 외형 missing 요청 | `observe_missing(I)`: 현재 I 근거 전부 제거 |
| ③ crop 보충 | 새로운 메시지 기록 후 I 저장 ACK → r_crop |
| ④ 현재 위치 찾기 실패 | `record_result(status='not_found', current_item=C)`: C만 0 |
| ④ A가 재이동을 확인 | 새 C 버전 `register` → 새 POSE_UPDATE 전송/ACK |
| 새 외형/배치 버전 | 새 I/N 버전 `register`: 해당 항목의 이전 버전 근거 제거 |

`verify_task(success=False)`도 해당 작업 증거를 취소한다. 원인이 불명확하면
성공 증거를 추가하지 않고 사건 관리자가 unresolved로 처리한다.
실제 조작한 외형을 저장했다는 정보 없이 무조건 learned_items를 넣지 않는다.
여러 물체 작업은 `items`에 각각의 KnowledgeItem을 넣는다.

### ACK와 조건부 전달 확률

ACK는 **저장 R**을 확인하며 **올바른 사용 K**를 보장하지 않는다.
각 근거는 `p_stored × p_usable_given_stored`를 가진다.
등록된 정확한 N/C의 내용 신뢰도는 기본 1, crop은 기본 0.95다.
A의 ID 판정도 불확실하면 전송 전에 재관측하거나 `content_quality`로
신뢰도를 낮춘다. 위치의 공간 오차 σ와 잘못된 표적 선택 위험은 정책/식별기에서
다루며 여기서 C의 ACK=1은 **그 버전의 정보를 보유함**을 뜻한다.

데이터 도착 d, 조건부 ACK 도착 a를 가정하면:

```text
P(data arrived | no ACK) = d(1-a) / (1-da)
d=a=0.9 → 0.4736842105...
```

이 식은 데이터가 도착하면 검증·저장과 ACK 생성이 성공하며, 각 전송 시도의
전달·ACK 실패가 독립이라는 모델이다. 실제 저장 실패/혼잡·연속 유실이 있으면
실측에 맞게 모델을 보정해야 한다. d=a=1에서 timeout은 불가능하므로 오류다.
같은 attempt의 timeout/ACK 재처리는 근거를 중복 합산하지 않는다.
재전송 timeout은 전달 기회만 합치고 동일 crop의 내용 신뢰도는 올리지 않는다.
같은 항목·버전이 다른 msg_id로 다시 전송돼도 같은 내용 근거 그룹으로 묶는다.
다른 메시지의 불확실한 수신 근거는 이 그룹 안에서 보수적으로 최댓값을 쓴다.

### 출처 결합과 취소

기본 설정은 HTML의 **독립 획득 출처** 가정이다.

```text
b = 1 - ∏(1 - p_source)
pass .72 + verified handling .9 → .972
crop .95 + verified handling .9 → .995
```

HTML은 `.995`를 약 `.99`로 표시하지만 코드는 정확값을 유지한다.
축약된 HTML 재검증 단계는 학습 갱신을 생략한 경우가 있어, 코드에서는
`record_result(learned_items=...)`에 실제로 명시한 항목만 학습시킨다.

같은 관측을 여러 프레임/콜백으로 독립 합산하면 과신한다. 한 번의 통과는
pass_id 하나로 기록한다. 동일한 획득 원인을 공유한 근거는 `dependency_group`을
같게 지정해 최댓값만 사용한다. A/B가 같은 인식기를 쓰는 등의 상관 오류가
큰 경우 `EstimatorConfig(independent_sources=False)`로 모든 출처의 최댓값을
사용할 수 있다. 두 모드 모두 실측된 확률 보정 성능을 확인해야 한다.

N/I가 한 스냅샷에서 함께 생긴 경우 각 주변 확률은 .72다. 이 출력은
**주변 확률**이고 N/I의 결합 독립성을 보장하지 않는다. 정책에서 두 항목의
확률을 무조건 곱하면 안 된다.

각 ItemBelief에는 불변 Evidence 목록이 있다. provenance_ids에는 msg_id,
pass_id 또는 task_id가 담긴다. 예전 작업의 잘못된 외형을 발견하면:

```python
estimator.revoke_evidence("event-1/attempt-1")
```

다른 crop/관측 출처는 유지된다. 현재 버전의 지식 자체가 없거나 쓸 수 없다는
명시적 피드백이면 `observe_missing(item)`으로 해당 항목 전체를 제거한다.
취소된 근거, 부정 피드백 이전의 늦은 ACK/보류된 관측·작업은 되살리지 않는다.
이후 보충은 새 msg_id/pass_id/task_id로 기록한다.

### epoch와 버전 계약

- memory_epoch는 수신 로봇의 영속 캐시 세대이며 **증가하는 정수**다.
  처음에는 STATUS handshake나 알고 있는 epoch로 인스턴스를 초기화한다.
  boot_id만 변경된 재시작은 영속 지식을 지우지 않는다.
- 새 epoch의 ACK는 기존 캐시를 먼저 지우고, 그 ACK가 명시적으로 저장을
  확인한 payload만 새 근거로 반영한다. E-only ACK에는 복원할 지식이 없다.
- 늦은 과거 epoch STATUS/ACK는 무시한다. 독립적인 INFO_REQUEST/RESULT도
  `memory_epoch=`를 전달해 과거 epoch의 피드백을 걸러낼 수 있다.
- 버전도 항목별 증가하는 정수다. 낮은 버전의 ACK/관측/작업으로 최신 버전을
  되돌리지 않는다. appearance_version은 물체 라벨·외형 콘텐츠가 교체될 때
  올리고, 같은 crop 재전송에서는 유지한다.
- task_id는 사건 전체가 아닌 **사건 + 실행 시도**를 식별해야 한다.
  예: `e4/attempt1`, `e4/attempt2`. 실패한 첫 시도의 늦은 검증이 새 학습으로
  반영되지 않게 하는 데 필요하다.
- 최신 버전·초기화·물체 이동을 A가 아직 모르면 갱신할 수 없다. 숨은 상태를
  미리 알려서 HTML의 실패 사례를 제거하지 않는다.

## 경로로 관측 가능성 계산

`probabilistic/visibility.py`는 계획 경로의 2D 카메라 pose 샘플, 타깃 위치, 거리/FOV,
A가 **알고 있는** 직사각형 가림 영역으로 관측 기회를 계산한다.
반복 프레임을 독립 사건으로 합산하지 않고, 시야가 확보된 샘플이 하나라도
있으면 설정된 조건부 시야 확률을 반환한다. 이 값을 `record_pass_by`의
`p_visible`로 사용하고 경로 실현 확률은 `p_pass`로 따로 입력한다.
좌표는 동일 지도 좌표계의 미터, 각도는 라디안이다. 연속 궤적/3D 가림,
체류시간에 따른 검출률과 실제 카메라 인식은 후속 보정 대상이다.

```python
from math import pi
from knowledge_estimation import Pose2D, VisibilityConfig, estimate_visibility

view = estimate_visibility(
    [Pose2D(0, 0, 0), Pose2D(1, 0, 0)],
    target=(2, 0),
    config=VisibilityConfig(max_range_m=3, horizontal_fov_rad=pi / 2),
    known_occluders=[],  # A가 모르는 칸막이는 여기 넣지 않음
)
estimator.record_pass_by(
    "route-1/S1", [normal, appearance], normal_before=True,
    p_visible=view.p_visible, p_pass=1.0,
)
estimator.confirm_pass_by("route-1/S1", normal_after=True)
```

## 다음 학습 단계

현재 추정기는 설명 가능한 확률 기준선이다. `learning/dataset` 수집기는 전송 직전 및
일정한 시간 간격으로 아래 데이터를 같은 시점에 고정해 별도 파일에 저장한다.

| 학습 입력(A가 아는 것) | 평가/학습 정답으로만 쓰는 것 |
|---|---|
| 전송/ACK 이력, 버전, 알려진 epoch, 경로와 관측 가능성, 검증 이력 | B가 그 버전의 N/C/I를 올바르게 보유하는지 0/1 |
| 관측/전송 후 경과 시간, 근거 종류 | 숨은 가림, 알려지지 않은 초기화·재이동 |

잘못된 물체 외형, 낡은 C, 넘어진 모습을 정상으로 저장한 N은 저장됐어도
정답 0이다. 결정 이후 ACK로 바뀐 B 상태를 결정 이전 정답으로 쓰지 않는다.
학습/검증/시험은 행이 아닌 에피소드·경로·배치 단위로 나누고, 통신 선택
규칙/PPO에 같은 추정기를 제공해 공정하게 비교한다.
추정기 자체는 Brier score, 신뢰도 곡선, 높은 추정·미보유 비율로 평가한다.
수집기는 기준선의 Brier score와 구간별 예측/실제 보유 비율도 기록한다.
이 수치는 생성된 합성 데이터에 대한 것이며 실제 로봇의 성능을 뜻하지 않는다.
시간 경과에 따른 망각/위치 노후화는 현재 자동 적용하지 않는다(HTML 재현).
이 기능을 추가할 때는 실측 로그로 학습/보정하고 버전 무효화와 구분한다.

## 참고 연구

[ToM2C 논문](https://arxiv.org/abs/2111.09189),
[저자 공식 코드](https://github.com/UnrealTracking/ToM2C),
[추정기 학습 코드](https://github.com/UnrealTracking/ToM2C/blob/main/train.py).
ToM2C는 상대의 관측·목표를 지도 신호로 추정하는 신경망과 협업 정책을
사용한다. 여기서 필요한 버전별 N/C/I의 유효 지식 확률과는 추정 대상이 다르다.
현재 코드는 HTML 규칙을 독립적으로 구현했으며 해당 연구 코드를 복사하지 않았다.
확률 기반 선행연구와 현재 수식의 정당화 범위는
[논문 대조 문서](probabilistic/RESEARCH_REVIEW.md)에 구분해 기록했다.
