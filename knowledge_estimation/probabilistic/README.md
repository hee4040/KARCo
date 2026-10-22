# 확률 기반 상대 지식 추정 (현재 사용)

N/C/I의 유효 지식 확률을 A가 관측한 전송·ACK·통과·작업/검증 이력에서 계산한다.
신경망 학습이나 학습된 가중치 없이 실행한다. 현재 시뮬레이터도 이 구현을 사용한다.

```python
from knowledge_estimation.probabilistic import KnowledgeEstimator, EstimatorConfig
```

기존 `from knowledge_estimation import KnowledgeEstimator`도 같은 클래스를 반환한다.
상세 호출 예시는 [상위 문서](../README.md)에 있다.

| 파일 | 역할 |
|---|---|
| `estimator.py` | 항목·버전·epoch·출처를 추적하는 추정기 |
| `probabilities.py` | 조건부 전달, 단일 관측 획득, 독립 획득 합집합 수식 |
| `visibility.py` | A가 아는 계획 경로·거리·FOV·가림으로 관측 기회 계산 |
| `RESEARCH_REVIEW.md` | 논문 원문, 수식 유도, 구현 대응, 수정 사항과 한계 |

[논문 대조 및 검토 결과](RESEARCH_REVIEW.md)를 먼저 확인한다. 현재 구현은
논문 전체의 베이지안 필터 재현이 아니라 KARCo HTML에 맞춘 근거 기반 기준선이다.
명목 확률과 출처 독립성은 실제 환경에서 보정/검증해야 한다.

```powershell
python -m knowledge_estimation.simulator
python -m unittest discover -s knowledge_estimation/tests -v
```

`EstimatorConfig(independent_sources=False)`는 출처별 최댓값을 쓰는 하한 근사다.
기본값 `True`는 HTML의 독립 획득 가정을 유지한다. 같은 원인을 공유하는
통과/작업 근거는 `dependency_group`을 같게 지정한다. 상관을 자동 판별하지 않는다.
