## 요구 사항

- Python 3.9+
- [Ollama](https://ollama.com/download)
- RAM 16GB 이상 권장

## 설치

### 1. Ollama 설치 및 모델 다운로드

```bash
# Ollama 설치 후
ollama pull gemma3:4b
```

### 2. Python 환경 세팅

```bash
python3 -m venv rag-env
source rag-env/bin/activate      # Windows: rag-env\Scripts\activate
pip install -r requirements.txt
```

### 3. 문서 가져오기

```bash
python3 fetch_docs.py    # 위키피디아 테스트 문서
python3 fetch_web.py     # 공식문서 (GitHub raw markdown)
```

## 실행

```bash
source rag-env/bin/activate      # Windows: rag-env\Scripts\activate
python3 rag.py
```

실행하면 터미널에서 질문/답변이 가능합니다. `quit` 입력 시 종료.

## 참고

- `docs/` 폴더에 `.txt` 또는 `.md` 파일을 추가하면 자동으로 인덱싱됩니다.
- 모델을 바꾸려면 `rag.py`에서 `"model": "gemma3:4b"` 부분을 수정하세요.
