# D455 기준 2D 데이터 수집

수작업 정답 예제(fixture), 시뮬레이터를 실행한 데이터(simulation), 로봇 계측 데이터(measured)를
구분한다. 현재 수집하는 것은 두 번째다. 시뮬레이터의 정답을 A의 입력에 넣지 않는다.

## D455에서 가져온 것과 실험자가 정한 것

| 항목 | 사용값 / 이유 | 근거와 한계 |
|---|---|---|
| 수평 시야각 | 87° | [현 제조사 D450/D455 사양](https://www.realsenseai.com/product-family/450-series/)의 depth 87×58°. RGB-D 공통 수평 영역으로 단순화 |
| 깊이 창 | optical Z 0.6–6m | 제조사 ideal range를 실험의 허용 창으로 선택. 이 범위에서 실제 검출이 보장된다는 의미 아님 |
| 시간 간격 | 1/30초 | 제조사 RGB 30fps를 참고한 시뮬레이션 샘플링 선택. 실제 프레임 드롭/동기화는 재현하지 않음 |
| N/I 저장 gate | 각각 연속 0.3/0.6초 | 설계한 gate의 동작 확인용 조건. D455 성능이나 논문에서 측정한 획득 시간 아님 |
| 환경·속도 | scenario.json의 좌표와 1m/s | 시야/가림 비교를 위한 설계 조건. 실제 로봇 경로나 속도 측정값 아님 |
| 전송 | 양방향 LOS, 단방향 0.1초, cutoff 0.5초 | 기하로 성공/실패가 갈리는 이상적 채널. D455/무선통신 사양과 무관 |
| C TTL | 4초 | 만료 처리를 비교하는 실험 조건. 실제 과제 정확도 요구로 정한 값 아님 |
| 센서 오류 | 모델링하지 않음 | 가시성+gate를 만족하면 정확한 ID/pose/template 저장. 인식 정확도 검증으로 해석 금지 |

사양 확인일은 2026-10-22다. [Intel 기존 D455 페이지](https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html)는
86×57°로 표기하므로 명목 사양 간 차이도 scenario.json에 기록했다. 현재 제조사 87×58°를
채택했으며, 실제 연결 시 장치/스트림 프로파일의 intrinsics로 교체해야 한다.
장착 높이·pitch·수직 시야는 없는 평면 모델이다. 0.6m는 물리적 검출 불가능 경계라고 주장하지 않는다.
제조사 깊이 오차 사양을 성공 확률이나 임의 Gaussian 잡음으로 바꾸지 않았다.

## 실행과 정답 생성

1. B가 직선 경로를 이동한다. 각 프레임에서 물체의 optical Z와 수평 각도, 선분-벽 교차를 검사한다.
2. 연속 관측 시간이 gate에 도달하면 B DB에 N/I를 저장한다. C는 통과만으로 저장하지 않는다.
3. A는 정상 배치 정보와 통과 완료·전후 정상 확인만 받는다. 실제 시야·카메라 방향·가림·DB는 받지 않는다.
4. 수집 정책은 ENCI 1회 전송으로 고정한다. LOS로 수신되면 정확한 내용을 저장하고 ACK한다.
5. 재시작 유무를 별도 실험 요인으로 전부 교차한다. 재시작 시 B DB를 비우고 epoch를 올리지만 A에 STATUS를 보내지 않는다.
6. 초기/통과 후/전송 직전/채널 처리 후/재시작 비교/TTL 만료의 6시점에서 A 추정과 B 정답을 기록한다.
   전송하지 않는 체크포인트는 작업 요구 항목을 비워 두므로 조작 실패로 집계하지 않는다.

확률 성공 여부를 난수로 뽑지 않는다. p_acq는 실제 실행에서 gate를 통과해 저장한 횟수/전체 통과 횟수다.
카메라 방향·가림을 A가 모르는 상태에서 여러 실행의 빈도로 상대 보유를 추정하므로, 기하 정답과 추정식을
같은 확률로 생성하는 순환 검증을 피한다. 다만 환경 규칙은 우리가 정한 것이므로 실제 성능의 증거가 되지는 않는다.

## 분할과 결과

- source: offset 4개 × heading 3개 × 벽 2개 = 24 에피소드. 획득/전송 빈도만 계산.
- calibration: 별도 offset/heading × 벽 2개 × 재시작 2개 = 48 에피소드. 고정 raw에 대한 보정만 적합.
- test: 다시 다른 offset/heading × 벽 2개 × 재시작 2개 = 48 에피소드, 288 평가 시점.
- source·calibration·test의 session/instance/route_group/episode ID는 겹치지 않는다.
  물체는 동일한 이상적 형식이므로 미지 물체 인식 일반화를 검증한 것이 아니다.
- 이 분할과 평가 기준을 scenario.json에서 실행 전에 고정한다. 결과를 보고 조건을 고르면 새 test가 필요하다.
- 설계한 유한 조건의 순회이므로 독립 무작위 시행을 가정하는 bootstrap/이항 신뢰구간은 출력하지 않는다.
- Brier(낮을수록 좋음), 신뢰도 구간별 실제 보유 비율, 고확신 오류와 상수 0/1 기준선을 제공한다.
  규칙/PPO 비교나 실제 집기 성공률은 아직 평가하지 않는다.

추가 조건 실험은 `--speeds-mps 0.5 1 2`로 실행한다. 세 속도는 실측치가 아닌 설계 요인이다.
source 72/보정 144/test 144 = 360 에피소드가 되며 test 평가 시점은 864개다.
기존 120개 결과를 보존하고 `data/d455_conditions_v1/`에 별도로 저장했다.
planned speed는 A가 아는 경로 조건이므로 p_acq/F_pass의 condition_id를 속도별로 구분한다.
실제 카메라 방향·벽·비공개 reset은 A에 추가로 전달하지 않는다.
거리 표의 값은 물체와 경로의 최근접 평면 거리이며 검출 순간의 optical Z와 다르다.
획득률 표는 reset 유무로 중복 실행한 동일 통과를 하나로 센다. 오류 표는 단계별로 분리한다.
CONDITIONS.md, acquisition_by_condition.csv, estimation_by_condition.csv,
decision_condition_rows.csv, condition_report.json이 추가 산출물이다.

`data/d455_v1/`의 `source_runs/`와 `decision_runs/`는 기하 실행 원본이다.
`world_trace.jsonl`은 프레임별 위치/가시성/연속 관측 시간/저장 시각을 담은 평가 전용 증거다.
각 dataset에 scenario와 실행 코드 사본·해시를 남긴다.
`inputs.jsonl`은 A 추정, `labels.jsonl`은 별도 B DB/정답이다.
`replayed/`는 source 빈도를 고정한 A-only 재생 결과다.
`simulation_params.yaml`, `calibration.json`, `report.json`, `summary.json`을 구분해 저장한다.
모든 출처는 simulation이며 기본 실측 로더는 거절한다. root params.yaml은 여전히 미측정이다.

## 논문을 참고한 범위

[ToM2C §3.2와 환경 부록](https://arxiv.org/html/2111.09189v2)은 상대 실제 관측을 학습 정답으로 사용하고
국소 관측으로 추론하는 구조와 2D 실험의 선례다. 우리의 N/C/I DB, dwell gate, 기하 규칙을
그 논문이 검증한 것은 아니다. 신경망 ToM2C를 재현했다고 주장하지 않는다.
[Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)은 예측 점수의 사후 보정과
신뢰도 평가의 근거다. 실제 구현의 PAV와 한계는 RESEARCH_BASIS.md를 따른다.
이번에 제거한 것은 미사용 compatibility import 파일 `estimator.py`와 오래된 생성 출력이다.
수집·주석·보정·평가·규칙 연동·계약 테스트는 재사용하므로 유지했다.
