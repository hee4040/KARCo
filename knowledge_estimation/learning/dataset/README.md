# 2D 지식 추정 데이터 수집

이 수집기는 학습용 입력과 정답을 만든다. 현재 추정기를 신경망으로 바꾸거나
학습된 가중치를 제공하지 않는다. Python 3.10 이상, 표준 라이브러리만 사용한다.

저장소 루트에서 실행한다.

```powershell
python -m knowledge_estimation.learning.dataset --families 30 --seconds 60 --sample-interval 1 --seed 20261022 --output knowledge_estimation/data/synthetic_v2
```

창이나 3D 환경 없이 30개 환경 묶음 × 4개 시나리오를 실행한다. 끝나면 자동
감사를 수행하고 `manifest.json`, `audit_report.json`을 생성한다. 기존 경로는
덮어쓰지 않으므로 다시 실행할 때는 출력 폴더 이름을 바꾼다. 중간에 실패한
폴더는 완료된 데이터셋으로 취급하지 않는다. 완료 표시는 manifest에 있다.
생성 데이터는 `data/`에 두며 git에서 제외했다. 코드와 수집 설정으로 재현한다.

## 입력과 정답을 어떻게 얻는가

예를 들어 A가 Q의 C를 보냈지만 ACK를 받지 못했다고 하자. 입력에는 전송
시각·버전·timeout 등의 관측 이력만 들어간다. 실제 패킷이 도착했는지는 입력에
넣지 않는다. 정답은 같은 시점 B 캐시를 검사해 별도 파일에 저장한다.
데이터 도착/ACK 유실과 데이터 자체 유실은 같은 A 입력에 다른 정답을 만들 수 있다.

| 항목 | 의미 |
|---|---|
| N 정답 | B가 현재 정상 배치 버전의 올바른 정상 정보를 보유하는가 |
| C 정답 | B가 실제 물체의 현재 위치 버전과 일치하는 정보를 보유하는가 |
| I 정답 | B가 현재 외형 버전의 사용 가능한 외형 기억을 보유하는가 |

각 정답은 0/1이다. `stored and usable and stored_version == actual_version`으로
계산한다. A가 아직 변경을 몰라도 실제 B 정보가 낡았으면 정답은 0이다.
정답은 **현재 세계에서의 유효성**이며, A가 알고 있는 예전 버전의 보유 여부와 다르다.
작업 복구도 위치 변경이므로 C 버전이 올라간다. 늦은 ACK가 와도 이전 정답을 바꾸지 않는다.

수집 시점은 두 종류다.

- `periodic`: 시작 시점과 지정한 간격마다 P/Q 각각 수집한다. ACK 수신 전후,
  아무 메시지도 보내지 않는 구간과 숨은 변경 이후도 포함하기 위해서다.
- `TASK`, `INFO_RESPONSE`: 자동 또는 수동 전송 **직전** 수집한다. 이번 전송과
  그 전송의 ACK는 해당 입력/정답보다 뒤에 발생한다. 재전송도 포함한다.

같은 시각의 샘플이 여러 개 있어도 삭제하지 않는다. 이벤트 순서가 다를 수
있으며, `history_end`가 실제로 참조할 관측 접두 구간을 결정한다.

## 파일 구성

```text
synthetic_v2/
  manifest.json                 수집 설정, 특징 순서, 분할, 클래스 통계, SHA-256
  audit_report.json             자동 감사 및 기준선 평가
  inputs/train.jsonl            A 입력만
  inputs/validation.jsonl
  inputs/test.jsonl
  labels/train.jsonl            B 정답과 정답 감사 근거만
  labels/validation.jsonl
  labels/test.jsonl
  observations/e000000.jsonl    해당 에피소드의 전체 A 관측 이벤트
  ...
  episodes.jsonl                재현용 조건·주입 일정·최종 상태 (특권 정보 포함)
```

입력과 정답은 `sample_id`로 연결한다. 두 파일의 순서에 의존하지 않는다.
`inputs` 행은 `time`, `target`, `reason`, `a_view`, `features`를 포함한다.
`a_view`에는 A가 아는 버전·epoch·직전 관측 위치·계획 통과 기하·기준선 확률과
`history_end`가 있다. B 실제 pose·캐시·패킷 전달 결과·숨은 가림은 없다.
기하 정보는 알려진 경로/카메라 제원과 A의 직전 관측으로 계산한 **예상 시야**다.
B가 실제로 보았거나 저장했다는 확인이 아니다.

`labels` 행의 `audit`에는 실제 epoch, 항목별 저장 여부·저장 버전·실제 버전·
사용 가능 여부를 남긴다. 정답 생성 과정을 점검하기 위한 것이며 학습 입력이 아니다.

## 학습에 넣을 값

`features`의 65개 숫자만 입력으로 사용한다. 목록과 순서는 `manifest.feature_names`와
`knowledge_estimation.learning.dataset.FEATURE_NAMES`에 정의되어 있다.

- 알려진 epoch 변경 횟수와 마지막 변경 통지 후 경과 시간
- 계획 통과 시 대상 거리·방향·카메라 범위/FOV 및 예상 범위 포함 여부
- N/C/I별 전송, ACK, timeout, 통과 기회, 정상 확인/거부, 결과 보고,
  재검증, missing, not_found, 근거 취소 횟수
- N/C/I별 마지막 전송·ACK·timeout·통과·재검증·missing·버전 관측 후 경과 시간
- N/C/I별 ACK/timeout으로 정리되지 않은 현재 전송 수

획득 이력은 **현재 알려진 epoch/버전**을 기준으로 센다. 숨은 초기화나
아직 관측하지 않은 물체 변경만으로 이력을 지우지 않는다. 경과 시간은 초이며
`-1`은 해당 관측이 없다는 뜻이다. 음수를 0으로 바꾸면 '방금 관측'과 혼동한다.
특징은 이력의 요약이므로 모든 장기 의존성을 보존하는 표현은 아니다.

`a_view.baseline`은 기존 확률 기준선의 비교 점수다. 기본 학습 입력에 포함하지
않는다. 기준선 보정 모델을 별도로 실험할 때만 의도적으로 사용할 수 있다.
ID, target 문자열, reason, 절대 시각, 시나리오 이름, seed, 정답/audit,
물리 설정과 주입 일정도 기본 특징에 포함하지 않는다.

```python
import json
from knowledge_estimation.learning.dataset import FEATURE_NAMES

with open("knowledge_estimation/data/synthetic_v2/inputs/train.jsonl", encoding="utf-8") as f:
    row = json.loads(next(f))
x = [row["features"][name] for name in FEATURE_NAMES]
# y는 labels/train.jsonl에서 같은 sample_id를 찾아 N,C,I 순서로 읽는다.
```

원시 시계열 모델을 만들 때는 `observations/<episode_id>.jsonl`의
**앞에서 `a_view.history_end`개까지만** 사용한다. 에피소드 전체를 넣거나
timestamp만으로 자르면 미래 정보와 같은 시각의 후속 이벤트가 섞일 수 있다.

## 다양화와 분할 이유

한 환경 묶음은 배치·통과 경로·시작점·카메라 제원·통신/저장 확률·속도·seed와
주입 일정을 공유한다. 그 조건에서 기존 네 시나리오를 실행하고, 네 실행 모두
같은 분할에 넣는다. 가림만 다른 대응 실행이 학습용과 시험용에 나뉘는 누수를 막는다.
환경 묶음을 섞어 약 60/20/20으로 나눈다. 최소 5묶음이며 기본 30묶음에서는
학습 18, 검증 6, 시험 6묶음이다. 행 단위 랜덤 분할은 사용하지 않는다.

| 변화 | 수집 범위 |
|---|---|
| 데이터/ACK 전달 확률 | 각각 0.65–1.0 / 0.55–1.0 |
| 경로 관측 저장/작업 후 학습 성공률 | 각각 0.5–1.0 |
| crop 사용 가능 확률 | 0.65–1.0 |
| 패킷 기본 지연 | 0.1–0.9초 + 기존 0–0.12초 jitter |
| A/B 이동 속도 | 각각 0.8–1.6 / 0.7–1.5 m/s |
| Q 정상 배치·통과 경로 | 기존 S2 주변에서 연속 좌표 변경 |
| B 시작점·카메라 | 시작점 주변, 범위 1.4–3m, FOV 0.65π–1.2π |
| 추가 사건 | 임의 시각의 정보 전송, 통지 유무를 바꾼 기억 초기화, 재이동, 외형 버전 교체 |

주입 일정은 B 정답을 보지 않고 미리 만든다. 환경 seed와 수집 seed를 기록하며,
네트워크와 저장 결과는 서로 다른 난수 스트림을 사용한다. `pass_occluded`와
`ack_loss`는 각각 숨은 장애물/첫 ACK 유실을 보여주기 위한 시나리오 조건이다.
네 시나리오를 같은 수로 수집한 비율은 실제 현장의 발생 빈도를 뜻하지 않는다.

## 검증과 다음 학습 단계

수집 때 자동으로 검사하며, 기존 데이터는 아래 명령으로 다시 검사할 수 있다.

```powershell
python -m knowledge_estimation.learning.dataset.audit knowledge_estimation/data/synthetic_v2
python -m unittest discover -s knowledge_estimation/tests -v
```

검사 항목은 파일 체크섬, sample ID 연결, 환경 묶음 분할, 미래 입력 방지,
원시 관측에서 특징 재계산, 캐시/버전 감사 근거에서 정답 재계산, 클래스 통계다.
별도 테스트로 숨은 가림·미통지 초기화·아직 관측하지 않은 버전 변경이 A 입력에
노출되지 않는지도 확인한다.

`audit_report.json`에는 N/C/I의 양성 수, 기존 기준선 Brier score와 10개 구간의
평균 예측/실제 보유 비율이 있다. Brier score는 `(예측 확률 - 0/1 정답)^2`의
평균이고 작을수록 좋다. periodic/전송 직전 점수를 별도로 제공한다. 학습 모델도
동일 샘플에서 비교하고, 정규화 통계는 train에서만 계산하며 validation으로
모델을 선택한 뒤 test는 최종 평가에 사용한다. 같은 에피소드 내 행은 독립 표본이 아니다.

현재 데이터는 수집 경로를 검증하는 초기 합성 데이터다. 카메라 픽셀·조명·
실제 인식기·오식별 조작은 구현하지 않았으며 I의 사용 가능성은 확률로 샘플링한다.
레이아웃 N의 버전은 고정이고 장기 망각은 없다. 관측 가능성 기준선의 0.8은
A가 아는 거리/FOV상 관측 가능한 경우의 명목값이다. 불가능하면 0이다.
실제 관측 기회는 2D 거리/FOV/숨은 가림에 의해 결정된다.
학습으로 이러한 시뮬레이션 가정을 익힐 수는 있지만 실제 로봇의 확률을 검증할
수는 없다. 이후 Gazebo/실제 로봇에서도 같은 A 관측 계약과 별도 B 정답 기록을
사용해 수집하고, 새 환경에서 다시 평가해야 한다.

생성기 버전 2부터 manifest의 `generator.source_sha256`에 수집·특징 추출·
시뮬레이터·확률 추정기 관련 소스 해시를 기록한다. `synthetic_v1`은 기하 제약
수정 전 기록이며 보존만 한다. 현재 모델 비교에는 `synthetic_v2`를 사용한다.
옛 `python -m knowledge_estimation.dataset` 명령은 새 구현으로 연결된다.
