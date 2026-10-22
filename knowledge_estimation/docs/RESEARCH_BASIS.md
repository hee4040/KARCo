# 논문·수식·구현의 근거와 적용 한계

노션의 관련 연구 목록 전체가 현재 알고리즘의 근거인 것은 아니다. 아래는 원문/공식
출판 기록에서 확인한 내용과 실제 코드의 연결이다. 논문 속 실험 수치는 이식하지 않았다.

| 1차 자료 | 확인한 내용 | 코드에 반영한 범위와 한계 |
|---|---|---|
| Bühler & Weisswange, **Online Inference of Human Belief for Cooperative Robots**, IROS 2018, §II ([저자 원문](https://d-nb.info/1176107542/34)) | 관측·행동 모델을 사용해 상대의 belief를 간접 추론 | A가 받을 수 있는 증거만 추정 입력으로 사용. 인간의 행동 likelihood나 전체 belief filter를 재현한 것은 아님 |
| Bühler, Adamy & Weisswange, **Theory of Mind Based Assistive Communication in Complex Human Robot Cooperation**, arXiv 2021, §3.1–3.2 ([원문](https://arxiv.org/html/2109.01355)) | 상대 belief의 요인화·Dirichlet 근사와 관측/행동 갱신, 통신 이득·비용을 평가 | '상대 추정 → 별도 통신 선택'의 구조적 근거. 우리 N/C/I의 상관 독립성이나 획득률을 입증하지 않음. Dirichlet moment matching/POMDP는 구현하지 않음 |
| Roth, Simmons & Veloso, **Reasoning About Joint Beliefs for Execution-Time Communication Decisions**, AAMAS 2005 ([CMU 공식 기록](https://publications.ri.cmu.edu/reasoning-about-joint-beliefs-for-execution-time-communication-decisions)) | 실행 중 가능한 joint belief를 고려해 팀에 유용한 관측을 통신 | 추정과 통신 유용성을 함께 다룬 선행연구. 메시지 항목별 보정이나 우리 max 결합을 이 논문에서 가져왔다고 주장하지 않음 |
| Unhelkar, Li & Shah, **Decision-Making for Bidirectional Communication in Sequential Human-Robot Collaborative Tasks**, HRI 2020 ([저자 원문](https://shenlirobot.github.io/docs/20-HRI-Unhelkar-Li-Shah.pdf)) | CommPlan의 통신 계획, 데이터 학습과 수동 지정이 섞인 모델 명세 | 실제 측정/설계 가정과 계획기를 구분하는 비교 근거. 논문의 수동 지정 값을 우리 실측값으로 사용하지 않음 |
| Wang et al., **ToM2C**, ICLR 2022, §3.2·부록 A ([원문](https://arxiv.org/html/2111.09189v2)) | 다른 agent의 실제 목표·관측 관계를 라벨로 ToM network 학습. 2D 관측 환경 사용 | 추정 입력과 타 agent 정답을 분리하는 데이터 설계의 참고. 우리 N/C/I 기억 유효성의 라벨과 대상이 다름. ToM network/A2C는 구현하지 않음 |
| Guo et al., **On Calibration of Modern Neural Networks**, ICML 2017, §2·4.1 ([공식 논문](https://proceedings.mlr.press/v70/guo17a/guo17a.pdf)) | 확률 보정·신뢰도 그림·별도 데이터의 post-hoc 보정, isotonic regression 설명 | 고정 raw 점수의 별도 단조 보정과 평가. 신경망이 아닌 raw 점수에 적용한 선택은 우리 설계이며, 실제 데이터에서 유효성을 평가해야 함 |
| Brier, **Verification of Forecasts Expressed in Terms of Probability**, 1950 ([출판사](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2)) | 확률 예측을 관측 결과와 제곱 오차로 평가 | 노션 정의를 따라 이진 항목마다 mean((b−K)²)를 사용. 원 논문의 범주 전체 합산과 스케일 관례가 다를 수 있음 |
| Clopper & Pearson, **The Use of Confidence or Fiducial Limits Illustrated in the Case of the Binomial**, 1934 ([원문](https://www.barestatistics.nl/uploads/1/1/7/9/11797954/clopper__pearson_1934.pdf), [DOI](https://doi.org/10.1093/biomet/26.4.404)) | 이항 모수의 구간 추정 | 독립 Bernoulli 그룹이라고 볼 수 있는 경우에만 exact 구간을 해석. 한 에피소드의 snapshot 수를 독립 시행 수로 사용하지 않음 |

## 직접 유도한 수식

아래는 프로젝트의 사건 정의에서 유도한 확률 항등식이다. 특정 ToM 논문의 전체
추정기를 구현했다는 뜻이 아니다.

### 무ACK

D를 정해진 cutoff까지 검증·저장된 사건, A를 그 cutoff까지 ACK가 돌아온 사건으로 둔다.
ACK가 D 이후에만 생성되고 진실하며 d=P(D), a=P(A|D)라면:

```text
P(D | not A) = P(D and not A) / P(not A)
             = d(1−a) / (1−da)
```

수치는 같은 cutoff의 측정값을 넣는다. p_ack 분모는 모든 송신이 아니라 성공한
p_recv 기회다. 부모 수신 시도와 ID가 연결되지 않은 ACK 표본은 거절한다.
d=a=1이라면 무ACK는 이 모델과 모순되므로 epsilon을 넣어 숨기지 않는다.
현실에서 이 모순이 관측되면 채널 모델/측정 범위를 다시 정해야 한다.

### 획득과 상관

노션의 p_acq는 해당 경로·작업대·조건에서의 **공동 획득·유효 저장 확률**이다.
중간 검출/저장 라벨이 없으면 개별 계수를 식별하지 않는다.
한 기회에 대한 진행은 p_acq × 측정 F_pass로 표현하고 같은 근거를 교체한다.

독립인 획득 사건이면 합집합은 `1−Π(1−p_j)`다. 상관이 미지이면 일반적으로
`max(p_j) ≤ P(하나 이상 성공)`만 보장된다. 현재의 max는 이 하한 선택이며,
독립성을 자동 학습하거나 완전한 Bayesian posterior를 만든다고 주장하지 않는다.
N과 I가 같은 통과에서 생기면 주변 확률이 같아도 두 항목이 독립인 것은 아니다.

### source 적합·시간 분포

각 관측 기회의 이진 성공 y에 대해 empirical frequency는 Σy/n이다.
Bernoulli likelihood를 최대화하면 이 값이 나오지만, 유한 표본의 실제 확률이
그 값과 정확히 같다는 보장은 없다. 분모·독립 그룹 수·조건·원시 증거를 함께 남긴다.
사전분포나 pseudo-count를 합리화할 별도 근거가 없어 현재는 넣지 않았다.

F_pass/F_recv는 해당 cutoff 안 성공한 기회의 시간 분포다. F_task는 공통 관측
horizon 내의 전체 시도 분모에 완료 시각을 누적한다. 실패/미완료 시도를 분모에서
빼지 않는다. 관측을 중간에 끊은 시도를 완전히 관찰한 실패로 바꾸면 안 된다.
이 구현은 horizon 밖 tail을 외삽하지 않는다.

## 보정의 선택과 한계

PAV는 같은 raw 점수를 묶고, 단조성을 위반하는 이웃 그룹의 성공 수·표본 수를
합쳐 구간 평균을 만든다. raw 추정기와 source 파라미터를 먼저 고정한 뒤
calibration의 결정 직전 라벨에 적합한다. 지원 점수 범위 밖은 미측정으로 처리한다.
작은 표본으로 보정이 나빠질 수 있으므로 별도 validation/test에서 평가해야 한다.

source와 score 적합이 같은 calibration 분할을 쓰는 2단계 방식은 노션의 현 단계
구조다. 충분한 데이터에서 추가 분리/cross-fitting과 다른 보정기를 비교할 필요가 있다.
현재 fixture가 이를 실증적으로 우수하다고 보여준다는 주장은 하지 않는다.

추가 D455 2D 실험에서는 source/calibration/validation/final test를 각각 분리했다.
`evidence_sources_v1`은 동일 raw 값이라도 A가 가진 증거 종류가 다른 경우를 나누어 PAV를
적합하는 자체 수정안이다. Guo et al.의 보정 평가 관점을 참고하되 이 N/C/I 증거 분할을
논문에서 그대로 가져왔다고 주장하지 않는다. 선택 규칙을 먼저 고정하고 validation에서만
선택한 후 새로운 final test에서 평가했다. 자세한 원인과 잔여 오류는 CALIBRATION_REPAIR.md에 있다.

## 모든 수치의 취급

- ACK 저장 여부, 버전 일치, 승인된 N 등록은 프로토콜/검증 조건이다.
- 카메라·로봇·경로·내용 신뢰도·지연·TTL은 실측 또는 명시적 과제 요구가 필요하다.
- confidence level·신뢰도 구간·평가 threshold·bootstrap 횟수는 사전 등록할 분석 계획이다.
  물리적 확률로 혼동하지 않으며 현재 params.yaml에서 미정으로 남겨 두었다.
- 64회 이분법 등은 binary64 수치해석 해상도를 위한 구현 설정이다.
- random 모듈은 관측 데이터를 bootstrap으로 재추출할 때만 쓴다.
  로봇의 성공/실패나 정답을 무작위로 생성하지 않는다.
- tests의 fixture 값은 고정된 검사 입력이다. 실제 측정/성능의 근거로 쓰지 않는다.
