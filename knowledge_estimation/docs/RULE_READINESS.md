# 조건별 추정 결과를 규칙 선택에 연결하기

사용자가 지정한 `heewoo/rule-based`의 로컬 ref `origin/heewoo/rule-based`를 읽었다.
확인 commit은 `ad056cb7bff058c749372ed125268c73c2d7ffdc`다. 네트워크 fetch를 하지 않았으므로
현재 원격의 최신 commit 여부는 확인하지 않았다. `#1/ppo`는 이번 단계의 구현 대상이 아니다.

## 현재 가능한 것

전체 검증 뒤 선택적 보정으로 ACK 직후 오류와 불필요한 거절을 줄였지만, 미보고 reset에서는
기존 보정보다 악화됐다. SELECTIVE_CALIBRATION.md의 지원 범위/부작용을 먼저 확인한다.
미측정 raw는 계속 None이며 규칙 입력으로 임의 대체하지 않는다.

기존 adapter 계약 검사에서 실제 팀원 코드를 임시로 읽어 N/C/I 키, SEND/NO_COMM/REOBSERVE,
payload 조합과 action_id를 확인한다. 이는 설명용 설정으로 하는 소프트웨어 검사다.
새 D455 조건 실험은 획득 빈도 계산, 고정 파라미터 재생, 별도 보정, untouched test 평가까지 실행했다.
현재 수집 정책은 ENCI 1회 전송이며 규칙이 payload를 선택해 환경을 바꾸는 실행은 아직 아니다.

## 바로 규칙 성능이라고 보고할 수 없는 이유

| 팀원 코드 | 확인 내용 | 연결 전에 필요한 것 |
|---|---|---|
| config.py / MessageSizes | E=120, N=C=40, I=30720의 설명용 크기 | 실제 사용할 표현/이미지와 serializer로 payload별 byte 측정. 현재 2D I 문자열은 실제 영상 crop 크기의 근거가 아님 |
| config.py / OutcomeParameters | crop_match_success=0.95, appearance_match_success=0.90 등은 추후 실측 교체라고 명시 | B 식별/재관측 결과에서 올바름·잘못된 선택·미해결 빈도 수집. 또는 이상적인 2D 식별 규칙을 별도로 정의하고 simulation으로만 평가 |
| cost_model.py / _missing_appearance_outcome | sigma/distance를 요청 확률, distance/(distance+2*sigma)를 위치 식별 성공률로 사용 | 코드 주석도 단순 모델/보정 필요라고 명시. 실제 요청/위치 식별 기록으로 검증하거나 모델을 팀과 수정 |
| config.py / CostWeights | 전송·요청·오조작·재관측·미해결의 가중치 | 팀의 설계 선호로 명시적으로 확정. 가중치 자체를 센서 성공률처럼 실측할 필요는 없음. 시간/byte 같은 비용 입력은 별도 계측 |
| requirements.py / Event | target_confirmed, 유사품 여부, 같은 사건 진행 여부 등에 따라 요구 항목 결정 | A가 관측한 사건 입력을 로그로 제공. 현재 정상 통과 시나리오만으로 유사품 복구 사건을 만들었다고 주장할 수 없음 |

우리 b_I는 유효한 I의 보유 확률이다. 팀원 appearance_match_success는 I를 갖고 있는 경우에도
현재 장면에서 매칭에 성공할 확률이다. 서로 다른 값이므로 같은 숫자로 대신하면 안 된다.
2D의 이상적인 r_crop=1을 실제 crop 매칭 성공률 1로 옮기지 않는다.

## 연결 순서

1. source 수집으로 획득 빈도를 계산하고 고정한다.
2. 별도 calibration 이력에 대한 A-only 추정으로 보정표를 고정한다.
3. 별도 test에서 조건별 획득률/추정 오류를 먼저 확인한다.
4. 확정한 규칙 설정과 A 관측 사건을 이용해 b_cal→RulePolicyAdapter→payload를 기록한다.
5. 선택한 payload를 실제로 B 모델에 적용하고 ACK/요청/DB 변화/실제 serializer bytes를 기록한다.
6. 동일한 외부 조건에서 Full/Confirmed Cache 등 기준선도 별도로 실행해 전체 비용/생략 실패를 비교한다.

4번에서 선택만 재생한 결과로 5~6번의 작업 성공률·총 통신량을 추정해서 보고하지 않는다.
추정·식별 확률은 시뮬레이션 로그/실측으로, 비용 선호는 실험 계획으로 각각 출처를 남긴다.
