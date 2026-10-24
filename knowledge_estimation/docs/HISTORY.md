# 개발 이력과 정리한 문서

현재 상태는 [README](../README.md)와 [SIM2D_RESULTS.md](../integration/SIM2D_RESULTS.md)를 본다. 이 문서는 지금 구조에 이르기까지의 단계와, 그 과정에서 쓰고 정리한 문서·코드를 기록한다. 지운 파일은 git에 그대로 있다.

```powershell
git show 608d56d:knowledge_estimation/docs/<파일 이름>          # 지운 문서 보기
git checkout cb67c57 -- knowledge_estimation/simulation          # 지운 2D 실험 코드 되살리기
```

## 단계

| 단계 | 커밋 | 내용 | 결론 |
|---|---|---|---|
| 1. 추정기 본체 | `af2d134`, `c6c09d5` | 노션 §5·6·8·11·12의 근거 기반 추정기(전송·ACK·무ACK·통과·재검증·epoch·TTL), 측정값만 쓰는 ParameterStore, b_raw/b_cal/b_safe 분리 | 지금도 쓰는 핵심(`probabilistic/estimator.py`, `parameters.py`, `types.py`) |
| 2. D455 기준 결정론적 2D 수집 | `494ba57` | 시야 87°, 깊이 창, 연속 관측 게이트로 B의 저장을 기하로 결정. 빈도표 p_acq, PAV 보정 | 빈도표는 방향·거리·가림을 반영하지 못함 |
| 3. 진단·보정 실험 | `494ba57`–`cb67c57` | raw 추정기 시간 처리 수정, 통과 직후 보정 오류, 증거 종류별·선택적 보정, 균형 수집 설계, 데이터 품질 감사 | 보정을 조건별로 나눠도 근본 원인(획득 모델)은 그대로였다 |
| 4. 확률적 세계와 시점 기반 모델 (사용 종료) | `cb67c57` | 사용자가 만든 별도 2D 세계(물리량에만 잡음, 장면별 참 확률 p\*)에서 A가 B의 통과를 다시 시뮬레이션하는 모델을 설계 | 빈도표 Brier ≈ 0.25(동전 던지기) → 시점 기반 0.022(하한 0.020). 이 세계는 더 쓰지 않고, 모델만 팀 2D 시뮬레이터용으로 다시 썼다(단계 6) |
| 5. 팀 2D 시뮬레이터 연동 | `97ed74d`, `608d56d` | 추정기를 팀 `sim/world.py`의 유일한 B 지식 추정기로 연결. 통과 근거는 이 세계에서 다시 잰 빈도표 | 규칙과 연결해 Confirmed Cache 대비 5개 조건 모두 유의하게 나음. [SIM2D_RESULTS.md §1–4](../integration/SIM2D_RESULTS.md) |
| 6. 시점 기반 모델을 팀 세계에 연결 | (이번 작업) | 빈도표 대신 계획 경로·지도·카메라로 프레임 수를 세고, 경로와 무관한 센서 비율 2개(p_frame, q_learn)만 측정 | [SIM2D_RESULTS.md §6](../integration/SIM2D_RESULTS.md) |

## 단계 3에서 나온 문서 (삭제, `608d56d`에 있음)

| 문서 | 다룬 것 | 남긴 교훈 |
|---|---|---|
| `RAW_REVISION.md` | 중복 ACK/send/RESULT에서 시간 갱신이 늦던 문제 수정 | 수정 전후 별도 데이터에서 개선 0·악화 0. 구현 정확성 문제였지 예측 성능 문제가 아니었다 |
| `CALIBRATION_REPAIR.md` | 모든 결정을 한 PAV에 넣으면 통과 직후 같은 특정 이력에서 악화 | 평균이 맞는 보정이 조건부로도 맞지는 않는다 |
| `ESTIMATOR_VERIFICATION.md` | 고정 모델의 상황별 검증 | 논리 처리는 통과, ACK 직후 보정 악화와 미지원이 드러남 |
| `SELECTIVE_CALIBRATION.md` | 검증에서 개선이 확인된 맥락에만 보정 적용 | 일부 개선, 미보고 기억 초기화에서 악화. 팀 세계에서는 단일 보정과 차이가 없어 단일 보정을 채택(SIM2D_RESULTS §2) |
| `COLLECTION_DESIGN_V2.md` | 방향·거리·벽을 균형 있게 뽑는 수집 설계 | 평균 일치가 장면 구별을 뜻하지 않음 |
| `ACQUISITION_DATA_AUDIT.md` | 획득률이 어긋나는 원인 감사 | 원인은 데이터 구성 차이와 방향·거리를 뺀 확률 모델. 단계 4로 이어짐 |
| `SIMULATION_PROTOCOL.md` | D455에서 가져온 값과 실험자가 정한 값의 구분 | 87° 시야·깊이 창은 명목 사양, 게이트·속도는 설계 조건 |
| `MEASUREMENT_PROTOCOL.md` | 실제 로봇 로그 수집 계약(MeasurementRecorder) | 실측 연결 시 다시 필요. 아래 코드와 함께 되살린다 |
| `NOTION_COMPLIANCE.md` | 노션 항목별 대응표 | 팀 세계 기준 대응은 SIM2D_RESULTS §1 표로 옮김 |
| `RULE_READINESS.md`, `VERIFICATION.md` | 규칙 연동 전 준비 상태, 당시 테스트 기록 | 규칙 연동이 끝나 역할이 끝남 |
| `STOCHASTIC_SIMULATION.md`, `PERSPECTIVE_ESTIMATOR.md` | 단계 4의 확률적 세계와 시점 기반 모델 연구(정보 수준 L0–L3, 카메라 오지정, 분포 밖 거리) | 이 세계를 더 쓰지 않으므로 삭제. 팀 시뮬레이터용 모델과 결과는 SIM2D_RESULTS §6 |

## 이번 정리에서 지운 코드 (`608d56d`에 있음)

| 파일 | 역할 | 지운 이유 |
|---|---|---|
| `protocol.py` | ROS 메시지를 A 관측으로 바꾸는 어댑터 | 호출하는 곳이 없음 |
| `measurement.py`, `calibrate.py` | 실측 로그 기록·빈도 적합·점수 보정 CLI | 짝인 `collect.py`·`evaluate.py`가 이미 지워져 끝까지 실행할 수 없었음. 쓰이던 PAV 함수는 `probabilistic/projection.py`로 옮김 |
| `projection.py`의 선택적·조건부 보정, `sim2d.py`의 근거 나이 구간 | 단계 3의 보정 변형 | 팀 세계 검증에서 단일 보정과 차이가 없어 쓰지 않음 |
| `sim2d_calibrate.py --candidates/--features` | 보정 변형 후보 적합 | 같은 이유 |

실측 로봇을 연결할 때는 `git show 608d56d:knowledge_estimation/measurement.py` 등으로 되살려 지금의 `perspective.py`(p_frame·q_learn 측정)에 맞게 고친다.
