# 이전 경로 호환용

실제 수집·특징 추출·감사 코드는 [`learning/dataset/`](../learning/dataset/README.md)로
이동했다. 이 폴더의 Python 파일은 기존 import/실행 명령을 새 구현으로 연결할 뿐
추정기나 수집기를 중복 구현하지 않는다.

```powershell
python -m knowledge_estimation.learning.dataset --families 30 --seconds 60 --output knowledge_estimation/data/synthetic_v3
```
