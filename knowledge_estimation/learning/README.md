# 학습 기반 상대 지식 추정 준비 (모델 미구현)

현재는 데이터 수집·특징 추출·입력/정답 감사만 구현했다. 신경망 추정기,
학습 루프와 학습된 가중치는 없다. 실제 실행 중인 추정기는
[`probabilistic/`](../probabilistic/README.md)다.

```text
learning/
  dataset/
    collection.py   조건을 다양화한 반복 실행과 입력/정답 분리
    features.py     A 관측 접두 이력에서 숫자 특징 65개 추출
    audit.py        인과성·정답 근거·분할·파일 무결성 점검
```

저장소 루트에서 실행한다. 이미 존재하는 경로는 덮어쓰지 않는다.

```powershell
python -m knowledge_estimation.learning.dataset --families 30 --seconds 60 --output knowledge_estimation/data/synthetic_v2
python -m knowledge_estimation.learning.dataset.audit knowledge_estimation/data/synthetic_v2
```

원시 관측과 N/C/I 정답은 확률 기반 추정기의 평가에도 쓸 수 있으므로 생성 파일은
공통 `knowledge_estimation/data/`에 둔다. 그 폴더는 git에서 제외되어 있다.
`synthetic_v1`은 관측 기하 제약 수정 전 기록이고, 수정 후 비교에는 `synthetic_v2`를 사용한다.
manifest의 생성기 버전/소스 해시로 어떤 코드가 만든 데이터인지 구분한다.

[데이터 형식과 수집 가정](dataset/README.md)에 따라 모델에는 `features`만 넣고,
정답·실제 캐시·숨은 가림·주입 일정은 입력하지 않는다. 기존 확률 추정치는
비교 대상으로 별도 보존한다. 미래의 모델은 같은 시점의 `b_N`, `b_C`, `b_I`를
출력하도록 연결하고 환경 묶음 단위의 학습/검증/시험 분할을 유지한다.
