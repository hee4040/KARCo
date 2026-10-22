# KARCo · `#1/ppo` 브랜치
### Knowledge-Aware Adaptive Communication for Multi-Robot Collaboration

이 브랜치는 KARCo의 **PPO 기반 적응형 통신 정책**과, 이를 학습·평가하는 **시뮬레이터**, 그리고 Robot A의 **이상 탐지기** 구현을 담고 있습니다.

Robot A가 순찰 중 이상을 발견하면, B가 이미 알고 있을 정보를 추정해서 **B에게 보낼 정보 조합**을 PPO 정책이 고릅니다. B는 해당 spot으로 이동해 노트북에 수리 안내문을 보내고, 사람이 물건을 고치면 B가 확인한 뒤 대기 위치로 돌아갑니다.

> 모든 확률·오차·속도 값은 **실측 전 가정값**입니다. 실제 로봇으로 측정한 뒤 바꿀 예정입니다.

---

## 1. 폴더 구조

```text
project/
├── rl/                   # 강화학습 (PPO)
│   ├── features.py       #   정책 입력(27차원) + B 지식 추정. 시뮬레이터와 실제 Robot A가 같이 씀
│   ├── env.py            #   이벤트 시뮬레이터 (이동·시야 없이 사건 단위)
│   ├── train.py          #   PPO 학습 (CSV 로그, 5만 스텝마다 체크포인트)
│   ├── eval.py           #   통신 방법 비교 평가
│   ├── curve.py          #   수렴 확인 (학습 곡선 + 체크포인트 평가)
│   ├── replay.py         #   이벤트 시뮬레이터 타임라인 출력
│   └── models/           #   학습된 모델(.zip), 학습 로그·그래프(logs/)
├── sim/                  # 2D 시뮬레이터 (현재 주력)
│   ├── world.py          #   배치, 기하, A·B 상태 기계, 작업자, 통신·인식 모델
│   ├── env2d.py          #   Gymnasium 환경 (rl/env.py와 같은 입력·행동·보상)
│   └── view.py           #   matplotlib 애니메이션
├── robot_a/
│   ├── inspector.py      # D455 + AprilTag + 색(HSV)으로 이상 탐지 (학습 모델 없음)
│   └── tag0~3.png        #   AprilTag 36h11 인쇄용 (7 cm)
├── SLAM/                 # D455 + RTAB-Map 초기 실험 (점구름 뷰어)
└── 적응형 통신 시나리오.md  # 팀 시나리오 문서
```

---

## 2. 설치

```bash
conda create -n caps python=3.10
conda activate caps
pip install numpy gymnasium stable-baselines3 matplotlib
pip install pyrealsense2 opencv-contrib-python open3d   # robot_a, SLAM 쓸 때만
```

GPU는 필요 없습니다. 정책 신경망이 작아서(64 × 64 MLP) CPU가 더 빠릅니다.

---

## 3. 실행

```bash
# 2D 시뮬레이터 자체 검사
cd sim
python world.py
python env2d.py

# 시각화: A 순찰, B 이동, 안내문, 메시지 로그
python view.py --policy proposed --condition occluded --seed 0 --speed 3
python view.py --policy full --seed 0          # 비교: 매번 전부 보내기
#   --policy  full | minimal | proposed | unc
#   --condition  base | occluded | limited

# 학습 (rl 폴더)
cd ../rl
python train.py --env sim2d                    # Proposed
python train.py --env sim2d --no-belief        # Uncertainty Only (B 지식 추정 없이)
python train.py --env sim2d --seed 1           # 시드 추가 → 모델 이름 뒤에 _s1

# 수렴 확인, 평가
python curve.py sim2d_proposed_a0p1 sim2d_proposed_a0p1_s1 sim2d_unc_a0p1
python eval.py --env sim2d --episodes 200

# Robot A 이상 탐지 (D455 연결 필요)
cd ../robot_a
python inspector.py teach      # 물건을 정상 위치에 두고 s 키로 저장 (q 종료)
python inspector.py            # 순찰 모드: 이상이 확정되면 A→B 메시지 출력
python inspector.py test       # 카메라 없이 판단 로직 자체 검사
```

`--env` 옵션을 빼면 이벤트 시뮬레이터(`rl/env.py`)로 학습·평가합니다.

---

## 4. 2D 시뮬레이터

| 요소 | 값 |
|---|---|
| 환경 | 2.0 m × 1.6 m |
| 차량 (A, B 동일, PiCar-B 기준) | 20 × 15 cm, 0.15 m/s |
| Robot A | 타원 라인 순찰, 카메라는 안쪽. spot마다 자동 계산된 관측 지점에서 1.5초 정지 |
| Robot B | 대기 위치 → 본선 → spot별 가지. spot 45 cm 앞에서 정지, 후진으로 복귀 |
| 카메라 | D455 기준: 시야 87°, 깊이 0.40 ~ 1.20 m (A, B 공통) |
| spot | AprilTag 7 cm × 4개. 무작위 2개 spot에 물체 2개(4 cm 간격, 같은 물체 / 모양만 다른 물체 무작위) |
| 가림막 | 20 cm, B 경로와 S1 사이 (`occluded` 조건) |

**에피소드 흐름 (300초, 0.2초 단위)**
1. 평균 40초마다 이상 발생: 오배치(4 ~ 10 cm), 전도, 잘못된 부품
2. A가 관측 지점에서 거리·가림을 기하로 계산하고, 이상을 확인하면 **정책이 보낼 정보를 결정**
3. B가 출동. 이동 중 시야에 들어온 spot을 확률적으로 기억함 (정답, A에게는 숨김)
4. B가 대상을 고르고 안내문 출력: `스팟 2에서 오른쪽 파란 물건을 뒤쪽으로 6cm 옮겨주세요`
5. 작업자(코드)가 수정. 20% 확률로 어긋나게 두면 10초 뒤 `물건 위치를 수정해주세요`
6. B가 틀린 물체를 가리켰으면 재확인에서 발견하고, A의 사진을 받아 다시 안내 (TASK v2)
7. 수정 확인 후 복귀. 이동 경로를 A에게 알려 A가 B 지식 추정을 갱신

**실험 조건**

| 조건 | 가림막 | 대역폭 | 손실 |
|---|---|---|---|
| base | 없음 | 200 ~ 1000 KB/s | 0 ~ 2% |
| occluded | 있음 (A의 가시성 추정은 모름) | 200 ~ 1000 KB/s | 0 ~ 2% |
| limited | 없음 | 5 ~ 50 KB/s | 10 ~ 30% |

---

## 5. 강화학습

### 행동 (17개)

`0` = 보내지 않음, `1 ~ 16` = 사건 정보에 아래 항목 중 일부를 붙인 조합.

| 코드 | 팀 문서 기호 | 내용 | 크기 |
|---|---|---|---|
| base | E | 사건 번호, spot, 이상 종류, 수리 요청 | 60 B |
| ID | — | 물체 설명 (색, 모양) | 40 B |
| CUR | C | 현재 위치(태그 기준 cm)와 위치 오차 σ | 60 B |
| NORM | N | 정상 위치 | 60 B |
| IMG | I | 물체 crop 이미지 | 30,000 B |

### 입력 (27차원, `rl/features.py`)

| 범위 | 내용 |
|---|---|
| 0 ~ 7 | A의 인식 상태 (확신도, 가림, 비슷한 물체 수, 위치 오차 ÷ 간격, 같은 물체 여부), 이상 종류 |
| 8 ~ 11 | **B 지식 추정**: 외형·정상 위치·현재 상태를 알 확률, 이미 작업을 받았을 확률 |
| 12 ~ 21 | 이 사건의 통신 기록 (보낸 횟수, ACK, 경과 시간, 결과, 보낸 항목, 요청 이유) |
| 22 ~ 26 | B 상태(근처·바쁨·대기열)와 네트워크(대역폭, 손실) |

B 지식 추정은 A가 볼 수 있는 것만으로 계산합니다: 보낸 기록, B의 ACK·결과·요청, B가 다닌 경로와 배치로 계산한 가시성.

### 보상

| 항목 | 값 |
|---|---|
| 수정 성공 | +10 |
| 정보 부족으로 B가 포기 | −3 |
| 에피소드 끝까지 미해결 | −10 / 건 |
| 대기 | −1 / 이상 1건 / 5초 |
| 전송량 | −0.1 / KB |
| 중복 전송 | −1 / 회 |
| 오조작 (틀린 물체 안내) | −5 / 회 |

### 학습 설정

PPO (Stable-Baselines3), 액터·크리틱 64 × 64 MLP, 환경 8개 병렬, n_steps 256, batch 256, 할인율 0.95, 엔트로피 계수 0.01. 15만 스텝에 약 18분 (CPU).

### 비교 방법

| 방법 | 보내는 것 |
|---|---|
| Full Sharing | 항상 전부 (사진 포함) |
| Fixed Minimal | base만, 나머지는 B가 요청 (Minimal Push + Pull) |
| Fixed Text | base + ID + NORM |
| Fixed State | base + ID + CUR + NORM |
| Uncertainty Only | PPO, B 지식 추정 입력을 가리고 학습 |
| **Proposed** | PPO, 입력 전체 사용 |

---

## 6. 현재 결과 (2026-10-21, 조건별 200 에피소드)

| 조건 | 방법 | 성공률 | KB/에피소드 | 오조작/에피소드 |
|---|---|---|---|---|
| base | Full Sharing | 0.91 | 247 | 0.00 |
| base | Fixed State = Uncertainty Only | 0.90 | 6.1 | 0.04 |
| base | **Proposed** | 0.89 | 7.7 | **0.01** |
| occluded | Fixed State = Uncertainty Only | 0.90 | 7.7 | 0.06 |
| occluded | **Proposed** | 0.91 | 7.9 | **0.04** |
| limited | Fixed State = Uncertainty Only | 0.88 | 6.5 | 0.06 |
| limited | **Proposed** | 0.87 | 7.9 | **0.03** |

- Full Sharing 대비 전송량을 약 97% 줄이면서 성공률은 같습니다. 다만 고정 규칙(Fixed State)으로도 같은 효과가 납니다.
- B 지식 추정을 받은 Proposed만 행동이 달라집니다. 위험할 때 사진을 보내 오조작을 줄입니다.
- 성공률·보상 차이는 아직 없습니다. 오조작 벌점(5)이 사진 비용(약 3)과 비슷해서 이득이 작게 잡힙니다.

그래프: `rl/models/logs/eval_sim2d.png`, `rl/models/logs/curve_sim2d_proposed_a0p1.png`

---

## 7. 남은 작업

- [ ] 오조작 벌점을 팀 문서 비율(사진 3.0 : 오조작 12)에 맞추고 재학습
- [ ] 규칙 기반(기대비용 최소화) 방법을 비교 방법으로 추가 (`heewoo/rule-based` 브랜치와 맞추기)
- [ ] 시나리오 분기 추가: B 기억 초기화, 처리 중 물체 재이동(POSE_UPDATE), ACK 재전송, 다른 시점 재관측
- [ ] 실측값 반영: 위치 오차, 분류 정확도, 사진 매칭률, 차량 속도, PiCar-B 후진·회전 반경
- [ ] B 카메라 확정 (D455 / Pi Camera)에 따라 B 시야 설정 분리
- [ ] `robot_a/inspector.py`: 같은 색 물체 여러 개 처리, 정책과 연결
- [ ] Robot B 구현 (라인 추종, 도착 후 대상 확인, 노트북으로 안내문 전송)

**단순화한 것:** 라인 추종 제어 대신 라인 위를 일정 속도로 이동, A·B 충돌 무시, 카메라 영상 대신 확률 모델 사용.
