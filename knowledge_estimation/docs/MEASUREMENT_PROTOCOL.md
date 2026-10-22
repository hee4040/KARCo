# 수집과 측정 계약

실제 카메라/로봇 로그가 들어오기 전에는 값을 채우지 않는다. `params.yaml`의 null은
미완료 측정이며, 비슷한 논문의 성공률이나 임의 확률로 대체하지 않는다.
source 확률은 그 정의에 맞게 관측한 모든 기회의 분모가 필요하다.

## 수집 전 정할 것

protocol_id에는 하드웨어 개체, 카메라 intrinsics/extrinsics, 검출기·descriptor 버전,
Tag Frame 정의·변환 검증, 저장 gate(검출 신뢰도·지속시간), TTL과 채널 deadline을
연결한다. receiver_id는 교체된 로봇을 구분할 수 있어야 한다. 다른 하드웨어의
측정값을 같은 역할 이름이라는 이유로 재사용하지 않는다.

검출 gate와 TTL은 논문에서 우리 로봇에 맞는 값을 찾을 수 있는 항목이 아니다.
거리·각도·가림·통과 속도·motion blur·조명·스티커 가시성·오차 허용 범위를 바꿔
측정하거나 과제 허용 오차로 정하고 근거를 남겨야 한다. 정의 전에는 계산을 중단한다.

평가 threshold·구간 경계·confidence level·bootstrap 계획은 평가를 보기 전에 고정한다.
이것은 분석 설계 선택이며 물리적 성공률과 다르다. root params.yaml은 이를 자동 결정하지 않는다.

## 온라인 수집: 정답 없이도 원본을 보관

1. A 이벤트와 현재 알려진 항목 메타데이터를 기록한다.
2. 정책이 전송하기 직전의 A 점수와 B DB 원자적 snapshot을 같은 논리 시각에 고정한다.
3. 전송 후/수신 후 B DB를 별도 snapshot으로 기록한다.
4. 외부 정답을 바로 알 수 없으면 world를 생략한다. labels에는 null로 남고
   manifest에 annotations_complete=false가 기록된다. 0/1을 임의로 붙이지 않는다.
5. 카메라 영상·DB dump·수신/저장 원본·수동 주석을 evidence_id로 연결한다.
6. 미측정 채널 때문에 갱신이 불가능한 이벤트도 journal과 measurement_gaps에 보존한다.
   raw는 미측정 상태이며, 이를 근거 있는 수치로 간주해 정책에 전달하지 않는다.

아래는 변수에 **실제 콜백 값**을 넣는 연결 형태다. 실행 가능한 임의 데이터 예제가 아니다.

```python
from knowledge_estimation.collect import MeasurementRecorder
from knowledge_estimation.probabilistic import KnowledgeEstimator
from knowledge_estimation.parameters import ParameterStore

estimator = KnowledgeEstimator(receiver_id=hardware_receiver_id,
    parameters=ParameterStore.load(measured_parameter_path))
recorder = MeasurementRecorder(new_dataset_path, protocol_id=measurement_protocol_id,
    episode_id=episode_id, estimator=estimator)
recorder.add_evidence(capture_id, raw_capture_relative_path, description=capture_description)
recorder.observe(a_observation)
recorder.record_decision(decision_id=decision_id, meta=group_metadata,
    time=decision_time, object_ref=object_ref, site_id=site_id,
    event_id=event_id, task_version=task_version, attempt_id=attempt_id,
    selected_items=actual_selection, required_items=scene_requirements,
    before_snapshot=atomic_b_db_dump)
recorder.record_transport(direction=direction, message_kind=message_kind,
    encoded_payload=actual_serialized_bytes, decision_id=decision_id,
    category=message_category, time=send_time, is_retry=is_retry)
recorder.record_after_receipt(decision_id, snapshot=post_receive_db_dump)
recorder.finalize()
```

아직 파라미터를 측정하지 못했다면 빈 ParameterStore로 원본을 수집할 수 있다.
저장 사실 R과 미측정 K는 구분되어 남는다. ROS 콜백은 순서대로 직렬화하고,
동시 실행에서는 외부 잠금/트랜잭션을 써야 한다. 이 모듈은 DB나 ROS 노드에 직접 접속하지 않는다.

## A 입력의 메타데이터

`meta`에는 정확히 split, episode_id, session_id, object_instance, route_group_id,
origin, protocol_id만 넣는다. 그룹 ID는 데이터 분할용이며 정책 특징으로 넣지 않는다.
split은 train/calibration/validation/test 중 하나다. origin은 measured 또는 fixture다.

A의 N/C/I는 각 항목의 버전, frame_id, observed_at, 실제 pose/descriptor 참조와
불확실성을 가진다. N에는 승인된 정상 배치 registration_evidence_id가 필요하다.
C에는 과제 프로토콜로 정한 TTL이 필요하다. pose의 단위는 m·rad이며 tuple의
성분 순서/오차 표현은 연결하는 메시지 스키마와 protocol_id에서 고정한다.
값·좌표계·불확실성이 없을 때 0 pose나 0 오차를 넣지 않는다.

입력에는 known epoch/version, journal prefix 길이, 증거/파라미터 ID, source 조건,
raw 의미(lower bound/unmeasured 등)를 남긴다. 실제 B DB·물체 정답은 포함하지 않는다.

## B DB snapshot과 외부 정답

snapshot에는 time, memory_epoch, items가 필요하다. items의 키는 실제 저장된 N/C/I만 둔다.
각 저장 행에는 version, frame_id, observed_at, received_at, evidence_ids를 기록하고,
C는 ttl_seconds도 기록한다. 시계가 동기화되지 않았으면 시간을 임의로 맞추지 않는다.

외부 주석은 별도로 아래를 제공한다.

- world: 같은 시각의 physical_object_id, 현재 N/C/I versions, frame_id,
  assessment_id, assessment_method(external_camera 또는 manual_annotation).
- 저장 행의 physical_object_id와 valid_content: A/B의 자체 성공 플래그가 아니라
  외부 identity·내용 검증으로 판정한다. pose 내용의 유효 기준은 과제 허용 오차와 연결한다.

`R=저장 행 존재`, `K=동일 실제 물체·현재 버전·유효 내용·일치 frame·C TTL 유효`다.
미통지 epoch 변경은 A의 known_epoch와 snapshot epoch를 비교해 사후 진단한다.
평가용 실제 ID·가림막·epoch는 runtime Observation의 허용 필드가 아니다.

## 오프라인 주석 붙이기

annotations.jsonl의 한 행은 decision_id, phase(before/after), world,
item_assessments를 갖는다. item_assessments는 실제 저장 행에 대해서만
physical_object_id와 valid_content를 제공한다. 버전·시각·DB 멤버십은 바꿀 수 없다.
같은 결정의 전후를 모두 주석하며, 원본 입력과 A journal은 그대로 복사한다.

```powershell
python -m knowledge_estimation.annotate measurements/raw --annotations measurements/manual_annotations.jsonl --output measurements/annotated
```

주석이 미완료인 상태로 capture를 보관할 수 있지만 score 보정·평가는 거절한다.
판단이 어려운 실제 사례를 임의 실패나 성공으로 바꿔 표본에 넣지 않는다.
원본을 유지하고 필요한 외부 증거를 추가한다.

## trials.csv의 시도 단위

헤더 순서는 코드의 TRIAL_FIELDS와 일치해야 한다.

```csv
split,episode_id,session_id,object_instance,route_group_id,origin,protocol_id,evidence_id,opportunity_id,source,receiver_id,kind,site_id,route_id,condition_id,success,duration_seconds,deadline_seconds
```

사용하지 않는 칸은 빈 문자열로 두며 success는 외부 검증된 0 또는 1이다.
source와 context별 같은 opportunity를 두 번 넣으면 오류다.

| source | 한 기회/분모 | 성공과 시간의 정의 |
|---|---|---|
| p_acq | 해당 경로·작업대·알려진 조건에서의 사전 지정 통과 기회, N/I 각각 | 기회 종료 때 B가 해당 항목을 올바르게 저장하고 유효하게 보유했는지 |
| q_learn | 직접 작업 후 해당 외형을 저장·재사용하는 시험 | 다음 사용 조건에서 버전·대상에 맞는 외형이 유효한지. 실제 재사용 시험의 evidence 필요 |
| r_crop | 고유 crop/content별 유효성 시험 | 그 content가 해당 대상의 유효한 외형 단서인지. 같은 이미지를 재전송했다고 새 표본으로 세지 않음 |
| r_current | 전달한 현재 위치 내용의 유효성 시험 | 현재 버전·frame·과제 정확도 조건에 맞는지. 식별 P_ident와 별개 |
| p_recv | 정해진 cutoff의 송신·저장 기회 전체 | cutoff까지 앱에서 검증하고 저장했는지. 소켓 도착만으로 성공 처리하지 않음 |
| p_ack | **성공한 p_recv 기회만** | 같은 cutoff까지 ACK가 A에 돌아왔는지. 부모 수신과 같은 opportunity_id로 연결 |
| F_recv | 성공한 p_recv 기회 | 송신부터 저장까지의 측정 시간. 조건부 시간 분포 |
| F_pass | 성공한 p_acq 기회 | 통과 시작부터 유효 저장까지의 측정 시간. p_acq와 곱하는 조건부 분포 |
| F_task | 같은 고정 horizon까지 관찰한 모든 작업 시도 | 완료 시각을 기록. horizon 내 미완료는 분모에 포함하고 success=0. 중간 이탈 시도를 완료된 관측처럼 넣지 않음 |

kind는 p_acq/F_pass에서 N/I, q_learn/r_crop에서 I, r_current에서 C다.
채널/F_task context의 kind·site·route는 빈 문자열이다. p_acq/F_pass는 실제
route_id와 site_id가 필수다. 다른 source의 작업 조건은 condition_id로 명확히 구분한다.
출처/조건이 미측정이면 전체 평균을 끌어와 채우지 않는다.

`condition_id`는 A가 실행 중 알 수 있는 조건만 표현한다. 숨은 칸막이 여부로
runtime 파라미터를 선택하면 정보 누수다. 칸막이 실제 여부는 원시 평가 기록에 둔다.
통과하지 못했거나 검출하지 못한 기회도 분모에서 누락하면 안 된다.
요청이 발생한 사례만 수집하면 선택 편향이 생기므로 모든 예정 기회를 기록한다.

## 분할과 평가

같은 episode/session/물체 instance/route 수행 group이 분할을 넘지 않게 한다.
route_id는 조건 모델의 경로 이름, route_group_id는 겹치면 안 되는 실제 수행/촬영 묶음이다.
완전히 새 route_id처럼 실측 지원이 없는 조건에는 수치를 외삽하지 않는다.

source·score는 calibration에서 적합하고 파라미터 fingerprint를 고정한다.
정책/cap 선택은 validation, 최종 평가는 test로 한다. 수신자별로 보정 모델을 따로 만든다.
측정 source와 score 양쪽의 session/instance/route/episode가 평가에 섞이면 오류다.

evaluate.py에는 다음 분석 계획을 명시해야 한다: high_confidence_threshold,
reliability_bin_edges, confidence_level, bootstrap_repetitions, bootstrap_seed,
bootstrap_unit(episode_id 또는 session_id). 미정인 root 계획으로는 실행하지 않는다.
safe_caps는 별도 정책 ablation이며 지정하지 않으면 cal과 같다.

실제 bytes 합계는 기록된 serializer 결과만 집계한다. 모든 송수신 종류를 계측했는지는
전송 담당자가 보장해야 한다. 로그가 없으면 0 bytes 대신 미측정(null)이다.
오프라인 정책 선택 비교만으로 다른 정책의 전체 bytes/성공률을 만들어내지 않는다.
