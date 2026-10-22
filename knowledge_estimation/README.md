> 현재 권장 수집기: [균형 수집 설계와 완료 결과](docs/COLLECTION_DESIGN_V2.md). 최신 완료 데이터는 `data/designed_acquisition_v3/`이며 보정은 OFF입니다. 이전 진단·보정 실험은 아래에 보존했습니다.

```powershell
python -m knowledge_estimation.simulation.collect_designed --output knowledge_estimation/data/designed_new --seed 2026102301
```

> 최신 원인 분석: [통과 획득률과 데이터 품질 진단](docs/ACQUISITION_DATA_AUDIT.md). 데이터 구성 차이·고유 장면 수·시간 해상도를 확인했으며 추정기와 원본 데이터는 변경하지 않았습니다.


> 현재 진행 기준: 사후 보정을 끄고 원본 추정기를 먼저 진단했습니다. 최신 결과와 원본·수정본 차이는 [raw 추정기 진단 및 수정](docs/RAW_REVISION.md)을 보세요. 아래 보정 실험은 이전 단계의 기록입니다.

```powershell
python -m knowledge_estimation.simulation.compare_raw --output knowledge_estimation/data/raw_revision_new
```

# 상대 지식 추정 — D455 기준 2D 수집과 실측 로그 연결

사용자가 제공한 노션 설계의 §5·6·8·11·12를 기준으로 상대 지식 추정 담당 부분을
구현한다. 성공률을 미리 넣어 뽑던 이전 생성기 대신, D455 명목 시야·거리와
위치·방향·가림·연속 관측 시간으로 B의 저장 여부를 결정하는 2D 환경을 연결했다.
상대 지식 추정, 로그 수집, 고정 오프라인 보정, 평가, 규칙 정책 경계를 재사용한다.
Python 3.10 이상, 표준 라이브러리를 사용한다.

## 지금 실행하기

최신 수정안은 보정을 선택적으로 적용하며, 근거가 부족한 조건에서는 원래 추정값을 유지한다.
새 ACK의 내용 유효성 계산을 유지해 직후 오류를 줄였고, 통과 없는 전송/대기 상태에서도
원래 값이 있으면 제공한다. **미보고 reset에서는 기존 보정보다 악화되는 한계가 남는다.**
원인별 분석·지원 범위·별도 평가 결과는 [선택적 보정 검증](docs/SELECTIVE_CALIBRATION.md)을 따른다.
최신 모델은 `data/selective_calibration_v2/selected_model.json`, 같은 폴더의 파라미터와 함께 사용한다.
이전 [전체 추정기 검증](docs/ESTIMATOR_VERIFICATION.md)은 수정 전 진단 기록이다.

통과 직후 보정 문제의 원인 분석·수정안 선택·새 최종 평가는 완료했다.
이전 증거 종류별 보정 실험과 결과는 [보정 수정 검증](docs/CALIBRATION_REPAIR.md)에 있다.
`data/calibration_repair_v1/selected_model.json`은 비교 기준으로 보존했다.
기존 실행기는 비교 기준인 pooled 보정을 유지하며, 수정안은 아래 전용 실험에서 검증한다.

```powershell
# 보정 적용/원래 값 유지를 별도 validation에서 결정한 뒤 새 final test에서 비교
python -m knowledge_estimation.simulation.compare_selective --output knowledge_estimation/data/selective_new_run

# 기존 모델을 고정해 새 상황 검증; 학습/재보정 없이 오류와 미지원 비율을 기록
python -m knowledge_estimation.simulation.verify_estimator --output knowledge_estimation/data/estimator_verification_new

# 새 source/calibration/validation/test를 분리하고 검증에서 선택한 뒤 최종 평가
python -m knowledge_estimation.simulation.repair_calibration --output knowledge_estimation/data/calibration_repair_new

# 새 폴더에 수집 → source 빈도 계산 → A-only 재생 → 보정 → 별도 test 평가
python -m knowledge_estimation.simulation.run --output knowledge_estimation/data/d455_new_run

# 속도를 바꿔 수집하고 조건별 표/CSV까지 생성 (속도는 설계 조건)
python -m knowledge_estimation.simulation.run --output knowledge_estimation/data/d455_conditions_new --speeds-mps 0.5 1 2

# 이번 실행 결과를 네이티브 2D 창에서 재생 (브라우저/서버 없음)
python -m knowledge_estimation.simulation.view knowledge_estimation/data/d455_v1
```

이번 결과는 `data/d455_v1/summary.json`, 상세 결과는 `report.json`이다.
추가 조건 실험은 `data/d455_conditions_v1/CONDITIONS.md`와
`acquisition_by_condition.csv`, `estimation_by_condition.csv`에 있다.
기존 120개와 별도로 source 72/보정 144/test 144의 360개 에피소드를 실행했다.
관측한 통과 획득률은 재시작 유무로 중복된 동일 통과를 제거해 계산한다.
기존 폴더를 덮어쓰지 않으므로 재실행 시 새 output 경로를 지정한다.
실험 조건·근거·데이터 구분은 [2D 수집 설명](docs/SIMULATION_PROTOCOL.md)을 따른다.
`origin=simulation`은 기본 실측 로더에서 거절하며, 전용 실행기에서 명시적으로 허용한다.

**실측 데이터와 검증된 성능 수치는 아직 없다.** `params.yaml`의 미측정 항목은
`null`/빈 목록이며, 라이브러리가 예시 확률로 채우지 않는다. 제공 논문은 방법의 근거다.
우리 로봇의 획득률·지연·내용 신뢰도를 대신 측정해 주는 자료가 아니다.

## 파일과 역할

```text
knowledge_estimation/
  params.yaml                 미측정 파라미터·측정/평가 계획
  types.py                    항목·버전·시각·frame·불확실성·증거 계약
  parameters.py               출처 있는 실측 파라미터 조회, 미측정 거절
  protocol.py                 A가 받을 수 있는 이벤트만 허용하는 어댑터
  probabilistic/estimator.py   전송/ACK/통과/작업/epoch/TTL 갱신
  probabilistic/projection.py  b_raw / b_cal / b_safe 분리
  collect.py                  A 이력과 평가 전용 B DB의 분리 수집·A-only 재생
  annotate.py                 외부 정답을 나중에 붙이는 불변 파생 데이터 생성
  calibrate.py                실측 CSV 빈도·경험 CDF·단조 보정 적합
  evaluate.py                 결정 직전 K와 비교, 출처/경로별 평가
  integration/rule_based.py    팀원 규칙 정책과 연결하는 엄격한 경계
  simulation/                 D455 기준 2D 환경·수집 실행기·네이티브 재생 창
  docs/                       노션 대응표·논문 근거·실측 절차
  tests/                      수작업 fixture와 계약 검사
```

사건 관리·B의 실제 검출/조작·통신 전송기·규칙 비용 모델·PPO는 다른 모듈의 책임이다.
이 코드가 그것들을 구현했다고 주장하지 않는다. 전체 대응 범위와 미구현 부분은
[노션 대응표](docs/NOTION_COMPLIANCE.md)를 따른다.

## 실행 중 계산하는 것

- R: B가 해당 버전을 저장했다는 증거. ACK는 R을 확인한다.
- K: 해당 물체·버전·frame·시점에 유효한 정보인지. I/C의 ACK만으로 K=1을 만들지 않는다.
- b_raw: A의 이력과 측정 파라미터로 만든 점수. 근거가 부족하면 `None`이다.
- b_cal: 별도 calibration 분할에서 적합하고 고정한 보정값. 보정 모델이 없으면 `None`이다.
- b_safe: b_cal에 명시한 정책 cap을 적용한 값. cap이 없으면 b_cal을 유지한다.
- P_ident: 현 장면에서 표적을 맞힐 가능성. 규칙/식별 모듈의 별도 입력이며 b_I와 합치지 않는다.

`None`은 0과 다르다. 미측정 값을 0·1·논문 예시값으로 대체하지 않는다.
등록됐지만 획득 증거가 없는 초기 항목의 0은 명시된 초기 지식·획득 모델의 상태다.
그것이 모든 미관측 원인까지 고려한 현실의 확실한 미보유를 뜻하지는 않는다.

출처 간 독립이 입증되지 않은 기본 raw 결합은 `max`라는 합집합 **하한**이다.
이는 잘 보정된 posterior라는 주장이 아니다. 독립 합집합을 사용할 때는
`EstimatorConfig(independent_sources=True, independence_evidence_id=...)`를 명시하고
그 가정을 따로 검증해야 한다. b_cal의 시험 성능도 별도로 평가한다.

### 갱신과 필요한 근거

| 이벤트 | 처리 | 수치의 근거 |
|---|---|---|
| 전송 | 측정 F_recv에 따라 도착 가능성 갱신 | 같은 조건의 p_recv, F_recv |
| ACK | R=1, K는 내용 유효성 반영 | N: 승인된 정상 배치라는 프로토콜 조건. I/C: 측정 r_crop/r_current |
| 무ACK | 같은 시도의 prior를 조건부 확률로 교체 | 같은 deadline의 p_recv와 P(ACK by deadline\|stored) |
| 통과 | 경로·작업대·조건·항목별 p_acq와 측정 F_pass | 중간 시야/저장 계수로 임의 분해하지 않음 |
| 정상 확인 | 노션 §8의 정상 구간 저장 게이트 후 통과 증거 사용 | A가 실제로 확인한 전후 상태, 같은 layout_version |
| RESULT ok | 지식 증거 보류 | B 자신의 판정만으로 완료 확정하지 않음 |
| 재검증 | 명시적으로 보고된 N/I만 반영 | N 등록·저장 프로토콜, I의 실측 q_learn |
| missing/stale_version/expired | 해당 항목 근거 무효화 | 명시적 항목·버전 보고 |
| not_found/ambiguous/물리 실패 | 진단 기록, 원인 미확정 시 보유 추정 유지 | 단일 실패 원인을 임의로 단정하지 않음 |
| epoch 증가 | 이전 epoch 근거 전체 무효화 | 실제 STATUS/ACK 통지 |
| C TTL | 센서 관측 시각 기준 만료 | 과제 정확도·움직임 실험으로 정한 TTL, 기본값 없음 |

같은 crop, msg_id, 통과 기회를 프레임마다 독립 합산하지 않는다. 같은 가림 원인을
공유한 반복 통과는 같은 dependency_group을 사용한다. A의 재관측이나 늦은 ACK는
B가 가진 이전 C의 관측 시각을 바꾸지 않는다. `F_task`는 완료 가능성만 제공하며
RESULT·재검증 없이 사건을 닫지 않는다.

## 실제 데이터를 받는 방법

`MeasurementRecorder`에 로봇/ROS 콜백을 연결한다. 랜덤 데이터 생성기는 없다.

1. 프로토콜 ID, 카메라·좌표 변환·시계·외부 정답 절차·검출/저장 게이트를 정한다.
2. A의 이벤트만 `observe(Observation(...))`로 기록한다.
3. 전송 **직전** A의 점수와 원자적으로 덤프한 B DB를 각각 기록한다.
4. 수신 **후** B DB도 별도 기록한다. 후 상태를 전 상태의 정답으로 쓰지 않는다.
5. 통과·작업·고유 crop·전송 시도별 결과를 `trials.csv`에 기록한다.
6. 실제 직렬화 결과인 `bytes`를 `record_transport`에 넘긴다. 가정한 메시지 크기를 넣지 않는다.

외부 정답을 나중에 얻으면 원본은 미주석 상태로 보존하고 `annotate.py`로 별도 파생
데이터를 만든다. 주석은 물체 연결과 내용 유효성만 평가하며 DB의 버전·저장 여부·시각을
바꿀 수 없다. 수집 완료와 외부 정답 완료를 구분한다.

수집기 출력은 `observations.jsonl`, `inputs.jsonl`, `labels.jsonl`, `trials.csv`,
`transport.jsonl`, `manifest.json`이다. 원시 로그·영상·수동 주석을 evidence_id로 연결하고
파일 SHA256을 남긴다. 해시는 위변조 탐지 수단이며 물리적 측정 진실성의 자동 증명은 아니다.

자세한 입력 스키마와 측정 단위는 [측정 절차](docs/MEASUREMENT_PROTOCOL.md)에 있다.
카메라/로봇이 연결되지 않은 현재는 이 로그를 실제로 얻었다고 주장하지 않는다.

## 실측이 준비된 뒤의 보정 순서

명령은 저장소 루트에서 실행한다. 출력은 기존 파일을 덮어쓰지 않는다.
아래 `measurements/...`는 실제 데이터를 배치할 경로 예시이며 제공된 데이터가 아니다.

```powershell
# 1. calibration의 시도 기록으로 source 확률·CDF 적합
python -m knowledge_estimation.calibrate measurements/run --mode sources --output measurements/measured_params.yaml

# 2. 그 파라미터를 고정하고 각 결정 직전의 A 이력만 재생
python -m knowledge_estimation.collect measurements/run --params measurements/measured_params.yaml --output measurements/replayed

# 3. 재계산한 raw 점수를 calibration의 결정 직전 K에 맞춰 고정 보정
python -m knowledge_estimation.calibrate measurements/replayed --mode scores --output measurements/calibration.json

# 4. 미리 고정한 평가 계획으로 별도 test에서 평가
python -m knowledge_estimation.evaluate measurements/replayed --model measurements/calibration.json --protocol measurements/evaluation_protocol.json --output measurements/test_report.json
```

`params.yaml`은 추가 YAML 패키지를 피하기 위해 JSON 문법(YAML 1.2의 부분집합)으로
저장한다. 현재는 미측정 계획서다. `calibrate.py`는 실제 성공 수/시도 수를 쓰며
Beta 사전분포·가상 성공 횟수·전역 fallback을 넣지 않는다. 표본이 0이면 값을 만들지 않는다.
CDF는 관측한 시간 범위 안에서만 사용하고, 보정 점수가 관측 지지 범위를 벗어나면 거절한다.

같은 촬영 session·물체 instance·경로 수행 group·episode가 분할을 넘으면 오류다.
source 적합과 score 적합이 calibration 분할을 사용하며, 정책 선택은 validation,
최종 보고는 test로 한다. 데이터가 충분하면 source/score 적합을 추가 분리하거나
cross-fitting을 검토하되, 현재의 작은 표본에 대한 개선 보장은 하지 않는다.

평가에는 Brier, 신뢰도 구간별 표본 수·bootstrap 구간, 고확률 미보유, 생략 실패의
요청/미보고 오조작 경로, 불필요 전송을 기록한다. 원인이 입증되지 않은 실패는
별도 미분류로 남긴다. 한 그룹뿐인 bootstrap 구간은 N/A다. 실패 0건을 위험 0으로
해석하지 않도록 그룹 단위 이항 구간도 제공하며 독립 시행 가정을 명시한다.

오프라인 재생은 기록된 같은 결정 시점의 점수 비교다. 새 정책의 에피소드 전체
성공률·총비용을 반사실적으로 만들어내지 않는다.

## 지금 가능한 검사

```powershell
python -m unittest discover -s knowledge_estimation/tests -v
python -m knowledge_estimation.tests.smoke --output knowledge_estimation/data/new_contract_smoke
```

노션 §12가 요청한 가짜 CSV 흐름 검사는 작고 결정적인 **fixture**로만 수행한다.
fixture의 수치는 테스트 입력이며 로봇 확률이 아니다. 기본 로더는 fixture를 거절하고,
검사용 `--allow-fixture`를 명시해야 한다. 보고서에도 `origin=fixture`,
`not_robot_performance=true` 또는 같은 의미의 문구를 남긴다.

팀원 규칙 연동은 [연동 계약](integration/README.md)을 따른다. 기존 팀원 정책의
설명용 비용/성공률 기본값도 자동 사용하지 않는다. 실측/사전 등록 근거와 실제
정책 설정이 일치해야 연동할 수 있다.
현재 규칙 코드에서 확인한 미확정 설정과 연결 순서는
[규칙 연동 준비 상태](docs/RULE_READINESS.md)에 정리했다.

이전 API에서 `simulator.bridge`를 import하던 코드는 `knowledge_estimation.protocol`로
바꿔야 한다. 직접 p_delivery/p_visible/p_store 등을 넘기던 호출은 측정 parameter
store와 condition/route/site ID 조회로 바뀌었다. 오래된 합성 데이터는 새 계약과 호환되지 않는다.

근거 논문과 적용 한계는 [논문·수식 대조](docs/RESEARCH_BASIS.md)에 정리했다.
