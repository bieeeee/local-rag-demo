from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
import requests
import json

# 1. 문서 로드
loader = DirectoryLoader("docs", glob="*.txt", loader_cls=TextLoader)
docs = loader.load()
print(f"문서 {len(docs)}개 로드")

# 2. 청킹
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunks = splitter.split_documents(docs)
print(f"청크 {len(chunks)}개 생성")

# 3. 임베딩 + 벡터 DB 저장
embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-small")
db = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
print("벡터 DB 저장 완료")

# 4. 질문 루프
print("\n=== 온보딩 Q&A (종료: quit) ===\n")
history = []

while True:
    query = input("질문: ")
    if query.strip().lower() == "quit":
        break

    # 이전 대화 맥락을 포함해서 검색 쿼리 보강
    if history:
        search_query = f"{history[-1]['q']} {query}"
    else:
        search_query = query

    # 질문에서 키워드별로 나눠서 검색 후 합치기
    results = db.similarity_search(search_query, k=3)
    # 개별 단어로도 추가 검색해서 다양한 문서 커버
    words = [w for w in query.split() if len(w) > 1]
    for word in words:
        extra = db.similarity_search(word, k=2)
        results.extend(extra)

    # 중복 제거
    seen = set()
    unique_results = []
    for r in results:
        key = r.page_content[:100]
        if key not in seen:
            seen.add(key)
            unique_results.append(r)
    results = unique_results[:7]

    context = "\n\n".join([r.page_content for r in results])

    # 대화 히스토리 포함
    hist_text = ""
    for h in history[-3:]:
        hist_text += f"사용자: {h['q']}\n답변: {h['a']}\n\n"

    prompt = f"""You are an onboarding assistant. Answer based on the documents below.
If the documents don't contain a direct answer but you can infer one by combining information from the documents, go ahead and answer.
Only say "No relevant information found" if the question is completely unrelated to the documents.
You must answer in Korean, concisely.

문서:
{context}

이전 대화:
{hist_text}

질문: {query}
답변:"""

    response = requests.post("http://localhost:11434/api/generate", json={
        "model": "gemma3:4b",
        "prompt": prompt,
        "stream": False
    })
    answer = json.loads(response.text)["response"]
    history.append({"q": query, "a": answer})
    # print(f"\n[검색된 문서 출처]:")
    # for r in results:
    #     print(f"  - {r.metadata['source']} | {r.page_content[:80]}...")
    print(f"\n답변: {answer}\n")

