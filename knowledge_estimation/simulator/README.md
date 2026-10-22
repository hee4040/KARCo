# 실행형 2D 시뮬레이터

독립 실행 창에서 로봇의 이동·관측·송수신과 상대 지식 추정을 실시간으로
확인하는 시뮬레이터다. Python 표준 라이브러리 Tkinter를 사용한다.
웹 서버나 브라우저가 필요하지 않다.

저장소 루트에서 실행한다.

```powershell
python -m knowledge_estimation.simulator
```

Windows에서는 상위 폴더의 `run_simulator.bat`를 더블클릭해도 된다.
화면 없이 실행해 검증·기록하는 방법:

```powershell
python -m knowledge_estimation.simulator --headless --scenario pass_occluded --seconds 90 --seed 7 --output knowledge_estimation/native_run.json
```

| 시나리오 | 확인할 현상 |
|---|---|
| `first_event` | 처음 전송한 N의 ACK 확인, 작업 후 외형 학습과 A의 재검증 |
| `ack_loss` | 데이터 수신과 ACK 유실의 구분, timeout과 재전송 |
| `pass_clear` | B가 이동 중 관측한 Q의 지식을 나중 작업에서 재사용 |
| `pass_occluded` | A가 모르는 가림 때문에 잘못 추정하고 추가 I 요청으로 회복 |

실행 창의 재생/일시정지·한 프레임 진행·속도·다시 시작을 사용한다.
아래 조작으로 물체 이동, B 캐시 초기화, 숨은 가림, 수동 N/C/I 전송을
실행할 수 있다. 로그와 전송 중인 패킷, 실제 B의 지식과 A의 추정을 함께 본다.

환경은 센티미터 정확도의 조작기가 아닌 단순화한 평면 환경이다.
정해진 작업 흐름을 따라 움직이며 관측·저장·패킷 손실에는 seed 기반 난수를
사용한다. `ack_loss`는 유실을 관찰하기 위해 ACK 실패를 주입하는 조건이다.
표시되는 통신 bytes는 HTML의 예시 크기이며 실제 네트워크 측정값이 아니다.
메시지 조합은 데모용 작업 흐름으로 지정한다. 팀원의 규칙/PPO 정책으로
교체할 수 있지만 이 데모 자체가 최소 비용 정책을 구현한 것은 아니다.

내부 환경은 0.05초 고정 간격으로 진행하며, GUI는 그 상태를 더 자주 그린다.
‘한 프레임’은 이 환경 간격 하나를 진행한다. 물체를 복구하는 동작도 물리적인
위치 변경이므로 pose_version이 증가한다. B가 보관한 예전 C는 이후 유효 지식으로
세지 않으며, 별도의 최신 C 수신/저장 없이 자동으로 최신 위치를 안다고 가정하지 않는다.

## 화면과 추정기의 분리

```text
engine.py                   desktop.py
물체·로봇·시계·패킷   ───→   Canvas 렌더링 / 조작
       │                         │
       │ A가 실제로 받은 증거     │ step / reset / 이벤트 주입
       ▼                         │
bridge.py  ←─────────────────────┘
관측 이벤트 계약
       │
       ▼
KnowledgeEstimator
N / C / I 유효 지식 확률
```

- `engine.py`는 창 없이 실행할 수 있는 `Simulator`를 제공한다.
- `desktop.py`는 `snapshot()`을 그리며 B의 실제 캐시를 추정기에 넣지 않는다.
- `bridge.py`는 직렬화 가능한 `Observation`과 `EstimatorAdapter`다.
  Tkinter와 ROS를 import하지 않으므로 나중에 수신 콜백에서 사용할 수 있다.
- B의 실제 기억·숨은 가림·A가 아직 모르는 상태는 평가용 world 상태로 유지한다.
  알 수 없는 이벤트를 미리 추정기에 알려서 오류를 없애지 않는다.

시뮬레이션 시간은 초, 좌표는 미터, 각도는 라디안이다.
좌표계는 동일한 지도 좌표계이며 **x는 오른쪽, y는 위쪽**이다.
Tkinter 화면의 아래 방향 y축으로 바꾸는 것은 렌더러에서만 한다.

## ROS/Gazebo로 연결할 때

현재는 ROS 노드나 Gazebo 플러그인을 설치·구현하지 않는다.
추후 다음 입력을 실제 로봇·시뮬레이터의 데이터로 대체할 수 있도록
추정기의 이벤트 계약을 분리했다.

| 2D에서 만드는 증거 | 나중에 연결할 실제 데이터 |
|---|---|
| A/B 로봇의 pose와 경로 | 같은 map 좌표계로 변환한 로봇 위치·이동 계획 |
| 예상 관측 가능성 | 알려진 지도·카메라 방향·거리·가림 정보 |
| 데이터 전송과 ACK | msg_id, 항목·버전, memory_epoch를 담은 수신 기록 |
| `RESULT`와 A 재검증 | B의 보고와 A의 별도 관측 결과 |
| 정상/현재/외형 버전 | layout/pose/appearance 버전 관리 |

`Observation` 예:

```python
from knowledge_estimation import KnowledgeEstimator
from knowledge_estimation.simulator.bridge import EstimatorAdapter, Observation

adapter = EstimatorAdapter(KnowledgeEstimator(memory_epoch=7))
normal = {"kind": "N", "object_id": "P", "site_id": "S1", "version": 1}

adapter.handle(Observation("send", 1.0, {"msg_id": "m1", "items": [normal]}))
adapter.handle(Observation("ack", 1.2, {
    "msg_id": "m1", "stored_items": [normal], "memory_epoch": 7,
}))
print(adapter.belief("P", "S1").as_policy_input())

# JSON 객체로 내보내기 / 복원
event = Observation("status", 2.0, {"memory_epoch": 8})
adapter.handle(Observation.from_dict(event.to_dict()))
```

이 timestamp는 추정기가 처리하는 시각이며 순서대로 전달한다. 센서의 측정
시각, 메시지 지연, 실행기의 처리 시각을 혼용하지 않는다. 실제 ROS 환경에서는
QoS·재전송·좌표 변환·시각 동기화를 별도로 정하고 연결해야 한다.
`truth`, `actual_cache`, `occluded` 같은 평가용 필드는 관측 계약에서 거부한다.

`decision_records`에는 전송 직전의 A 추정, `a_view`와 B의 평가용 정답 및
`label_audit`를 각각 저장한다. `a_view.history_end`는 전체 A 관측 기록에서
그 시점에 사용할 수 있던 정확한 접두 구간이다. 같은 시각의 후속 send/ACK도
그 구간에 없으면 입력으로 쓰지 않는다. 전체 관측은 `export_observations()`로
받으며 화면 표시용 1,000건 제한과 별도로 보존한다.

반복 실행과 학습/검증/시험 데이터 생성은 [데이터 수집기](../learning/dataset/README.md)를
사용한다. 이 수집기는 Q 배치·통과 경유점·B 시작점·카메라 제원과 통신/저장 조건을
바꾸고, 기억 초기화·재이동·추상적인 외형 변경·수동 정보 전송을 주입한다.
외형 변경은 버전 교체일 뿐 실제 RGB 인식 시뮬레이션은 아니다.

현재 시뮬레이터는 `probabilistic.KnowledgeEstimator`를 사용한다. 통과 근거는
A가 아는 거리/FOV상 가능할 때만 추가하며 숨은 가림은 A에게 전달하지 않는다.
같은 통과의 예상 기하는 고정해 이후 물체 이동으로 소급 변경하지 않는다.
