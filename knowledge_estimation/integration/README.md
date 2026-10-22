# 팀원 규칙 정책과의 경계

추정기는 N/I/C 키로 b_cal을 전달한다. 팀원의 RulePolicyAdapter가 payload를 선택한다.
P_ident, 후보 장면, 실제 메시지 크기, 비용 가중치는 추정기에서 재구현하지 않는다.

기본 연동 level은 cal이며 고정 보정 모델이 필요하다. raw는 명시적 ablation/점검용이다.
미측정 점수, 보정 지지 범위 밖 점수, 다른 파라미터 fingerprint의 모델은 거절한다.
유사품 등장만으로 b_I를 바꾸지 않는다. 해당 장면 정보는 Event에 별도로 전달한다.

```python
selector = RuleBasedSelector(
    adapter=team_rule_adapter,
    policy_manifest=policy_parameter_evidence,
    model=fixed_receiver_calibration,
)
decision = selector.select(a_observable_event, estimator.snapshot(object_ref, site_id, now=decision_time))
```

policy_manifest는 origin, config, evidence_ids, input_sha256을 갖는다.
config는 실제 adapter.policy.config를 dataclasses.asdict로 변환한 값과 정확히 같아야 한다.
물리적 성공률은 실측 로그, 비용의 선호·분석 가중치는 사전 등록 실험 계획으로 근거를
제공한다. 매니페스트/해시는 출처 추적 장치이며 사람이 작성한 값의 진실성을 자동 증명하지 않는다.

다른 branch의 설명용 기본 PolicyConfig를 아무 설정 없이 생성하는 경로는 차단한다.
테스트에서만 origin=fixture와 allow_fixture=True를 명시하고 팀원의 원본 설정을 입력으로 쓴다.
그 테스트가 정책 성능을 입증한다는 뜻은 아니다.

merge 전 코드 확인용 `prepare_rule_snapshot.py`는 로컬에 받아둔 Git ref에서
원본 패키지를 임시로 읽을 수 있다. fetch·merge·branch 변경은 하지 않는다.
현재 테스트는 origin/heewoo/rule-based의 원본을 임시 폴더에 복사해서 SEND/NO_COMM/
REOBSERVE와 입력 계약을 확인한다. ref가 없으면 이 테스트만 skip된다.

E-only는 TASK 전송이고 NO_COMM과 다르다. 동일 메시지 재시도는 선택된 payload,
content ID, 측정 채널 context를 유지한다. 새로운 측정 context로 바꾸려면 별도 메시지로 기록한다.
업무상 event/version/attempt가 바뀌었는지와 늦은 RESULT 폐기는 공통 사건 관리자에서도 검사해야 한다.

runtime 정책 입력에는 B DB, 실제 물체 ID, 숨은 가림, 평가 정답이 들어가지 않는다.
실행 전후의 추정값과 선택을 MeasurementRecorder에 기록하고 실제 전송은 전송 담당 코드가 수행한다.
