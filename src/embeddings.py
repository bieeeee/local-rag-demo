"""
Embedding model wrapper.

An embedding turns text into a high-dimensional vector that captures meaning,
and both the indexing step (storing documents) and the retrieval step
(transforming queries) go through the same model.

This file wraps the multilingual-e5 model rather than using it as-is.
The e5 family is trained to expect a role prefix on its input, so
documents to be stored need "passage: " and search queries need "query: "
prepended in order to get proper retrieval quality.
"""

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.embeddings import Embeddings

class E5Embeddings(Embeddings):
    """
    Wraps the multilingual-e5 model with the query:/passage: prefix convention.

    Implements LangChain's Embeddings interface (embed_documents / embed_query),
    so it can be plugged directly into vector stores like Chroma.
    """

    def __init__(self, model_name: str):
        # normalize_embeddings=True: normalizes vectors to unit length.
        # With normalized vectors, the dot product equals cosine similarity,
        # so the vector store's similarity search behaves as intended.
        self._inner = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # Indexing step: these are documents to store, so embed with the "passage: " prefix.
        return self._inner.embed_documents([f"passage: {t}" for t in texts])

    def embed_query(self, text: str) -> list[float]:
        # Retrieval step: this is a user query, so embed with the "query: " prefix.
        return self._inner.embed_query(f"query: {text}")