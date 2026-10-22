# 노션 요구사항과 구현 범위

기준은 사용자가 이번 요청에 첨부한 `61fa73cd-e750-4ff3-979a-ad0acf1f9d75/Pasted text.txt`다.
문서의 예시 숫자를 측정값으로 취급하지 않는다. 이 파일은 상대 지식 추정 담당 범위의 대응표다.
첨부 원문 SHA256: `c22001ea11ab4dd68d6bb08e9f66bbcf4bcb1249e6b4940f3eb9c6d84f100c1b`.

| 노션 | 현재 대응 | 범위와 미완료 사항 |
|---|---|---|
| §5 R/K/P_ident 분리 | Evidence.p_stored, ItemBelief.stored_probability, 유효성 점수·오프라인 R/K 라벨 | P_ident와 비용은 팀원 식별/규칙 모듈의 책임 |
| §5 항목 메타데이터 | receiver/epoch, object_ref, 연결 신뢰도, frame, 버전, 관측 시각, 값·불확실성, 증거 ID | 연결 신뢰도 미측정이면 None. 수신 시각은 ACK가 실제로 보고한 경우만 기록 |
| §5 세 가지 값 | b_raw/b_cal/b_safe와 고정 모델 fingerprint | 미측정 raw·미적합 cal은 None. 0으로 대신하지 않음 |
| §6 전송·ACK·무ACK | 측정 p_recv/F_recv, 같은 deadline의 조건부 p_ack, 시도별 posterior 교체 | 채널 프로토콜과 실제 cutoff 기록 필요. 임의 RTT/손실률 없음 |
| §6 결과·재검증 | 7종 RESULT, ok 보류, 명시된 learned_items만 검증 후 사용 | 사건 종료·취소·STATUS_QUERY 상태기계는 사건 담당 모듈에 남김 |
| §6 통과 | 항목·수신자·경로·작업대·조건별 공동 p_acq와 경험 F_pass | F_pass는 성공한 획득의 조건부 저장 시각. 통과 반복·갱신은 같은 근거 교체 |
| §6 정상 배치 조건 | 같은 layout_version의 A 전후 정상 확인 | §8의 '봄' 저장 규칙을 따라 전후 확인 전에는 통과 근거를 사용하지 않음 |
| §6 보관/유효성 | version/epoch, 관측 시각 기반 C TTL, 진단된 frame 오류 | TTL·검출 gate·deadline은 측정/과제 계획으로 외부에서 정해야 함 |
| §6 상관 | dependency_group, 동일 콘텐츠/시도 중복 방지 | 독립 근거가 없으면 raw는 max 하한. 독립 합집합은 명시적 가정으로만 활성화 |
| §8 B 저장 규칙 | 실제 B DB 기록 계약 + 2D 시야/연속 관측 gate | 2D는 D455 명목 기하 모델. 실제 검출기·센서 잡음·물리 조작은 미구현 |
| §8 결정 전/수신 후 DB | 두 snapshot을 분리 저장. 전 상태만 추정 정답으로 사용 | DB 원자적 덤프와 시계 정합은 로봇/ROS 어댑터에서 보장해야 함 |
| §9 ID와 메시지 | msg_id, epoch, item version; decision에 event/version/attempt 기록 | task_id는 event/version/attempt가 구분되는 키로 전달. 업무상 오래된 RESULT 차단은 공통 사건 관리자에서도 수행 |
| §9 실제 bytes | 직렬화된 bytes의 len과 SHA256, 종류/방향/재전송 기록 | ROS/OS 헤더까지 포함한 wire bytes와 별개. 전송기 계측 연결 필요 |
| §10 강한 기준선 | 측정 출처를 요구하는 규칙 어댑터 보존 | Full/Cache/Minimal Push/PPO의 정책별 전체 실험은 아직 하지 않음 |
| §11 평가 | 결정 직전 항목별 Brier/신뢰도·표본 수·구간, 출처·경로 조건별 집계, 생략 실패/불필요 전송 | 실제 성공률/Pareto는 실제 정책별 에피소드 실행이 필요 |
| §11 외부 정답 | 원시 B DB 수집 후 외부 카메라/수동 주석을 별도 attach | A/B 자신이 한 인식의 성공 플래그만으로 최종 정답을 만들지 않음 |
| §11 분할 | session/instance/route_group/episode의 교차 누수 거절 | route_id는 경로 조건의 이름, route_group_id는 분리할 실제 경로 수행 묶음 |
| §12 보정 흐름 | trials.csv → calibrate sources → A-only replay → calibrate scores → evaluate | 실제 CSV는 아직 없음. fixture는 프로그램 계약 검사만 수행 |
| §12 측정 항목 | params.yaml에서 전부 미측정 상태로 선언 | 카메라 종류·내외부 파라미터·조명·실제 B 경로·집기 장치 정보가 없어 수치를 채우지 않음 |

## 이번에 정리한 것

- 자체 설정 확률로 성공/실패를 뽑던 네이티브 2D 엔진과 GUI, 실행 bat.
- 해당 엔진의 무작위 데이터 생성기와 신경망 준비용 미사용/중복 폴더.
- 이전 합성 데이터, Beta 평활 파라미터, 보정 파일, 성능 분석 보고서.
- 그 가정에 종속된 기존 테스트와 오래된 실행 문서.

기존 commit의 기록은 Git에 남지만 현재 구현의 성능 근거로 쓰지 않는다.
현재 `data/d455_v1/`에는 기하 기반 시뮬레이션 실행 결과가 있다. 수작업 fixture와 구분하며 실측 데이터도 아니다.
팀원의 다른 branch와 저장소 공통 README는 수정하지 않았다.

## 수치를 채우지 않은 이유

논문의 로봇·카메라·과제가 우리 실험과 같다고 볼 근거가 없다. 논문에 있는 성공률을
우리 p_acq/q_learn/r_crop/TTL에 옮겨 쓰지 않는다. 확률의 통계적 추정 자체는
연구 목표의 일부지만, 입력 파라미터는 측정 이력·분모·조건·원시 증거와 연결되어야 한다.
0/1 저장 사실, 버전 counter, 확률 범위, 수치해석의 반복 횟수는 경험적 성공률과 구분한다.

## 현재 완료의 의미

완료한 것은 수집·추정·보정·평가 계약, fixture 검사, D455 기준 2D 수집·평가 경로다.
실제 센서 수집, 데이터 기반 보정값, 하드웨어 성능, 전체 팀 시스템의 통합 완료는 아니다.
실측값이 없는 동안 실제 정책에 시뮬레이션 수치를 자동 적용하지 않는다.
