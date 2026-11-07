from pathlib import Path

import chromadb
from langchain_core.documents import Document

from src.rag_pipeline import SAMPLE_DOCUMENTS, ChromaRAG


class FakeEmbeddings:
    """Deterministic local embeddings for fast, offline unit tests."""

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text):
        tokens = set(text.lower().split())
        vocabulary = [
            "cosine",
            "embedding",
            "chromadb",
            "retrieval",
            "metadata",
            "pdf",
        ]
        return [float(token in tokens) for token in vocabulary]


def test_demo_pack_has_twenty_documents():
    assert len(SAMPLE_DOCUMENTS) == 20
    assert all(len(item) == 3 for item in SAMPLE_DOCUMENTS)


def test_chroma_cosine_collection_and_metadata_filter(tmp_path):
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))

    collection = client.get_or_create_collection(
        name="test_rag",
        metadata={"hnsw:space": "cosine"},
    )

    embedder = FakeEmbeddings()

    docs = [
        Document(
            page_content="cosine similarity compares embedding vectors",
            metadata={"category": "similarity"},
        ),
        Document(
            page_content="PDF pages can be split into retrieval chunks",
            metadata={"category": "pdf"},
        ),
        Document(
            page_content="metadata filtering narrows retrieval results",
            metadata={"category": "retrieval"},
        ),
    ]

    collection.add(
        ids=["a", "b", "c"],
        documents=[doc.page_content for doc in docs],
        metadatas=[doc.metadata for doc in docs],
        embeddings=embedder.embed_documents(
            [doc.page_content for doc in docs]
        ),
    )

    assert collection.count() == 3

    result = collection.query(
        query_embeddings=[embedder.embed_query("cosine embedding")],
        n_results=2,
        where={"category": "similarity"},
        include=["documents", "metadatas", "distances"],
    )

    assert result["ids"][0] == ["a"]
    assert result["metadatas"][0][0]["category"] == "similarity"
    assert 0 <= result["distances"][0][0] <= 2


def test_pdf_chunk_metadata_shape():
    metadata = {
        "category": "pdf",
        "source": "Module-3-PE.pdf",
        "page": "0",
    }

    assert metadata["category"] == "pdf"
    assert metadata["source"].endswith(".pdf")
    assert metadata["page"].isdigit()


def test_search_returns_empty_list_for_unmatched_metadata(tmp_path):
    rag = ChromaRAG.__new__(ChromaRAG)

    client = chromadb.PersistentClient(
        path=str(tmp_path / "empty_filter_db")
    )

    rag.client = client
    rag.collection = client.get_or_create_collection(
        name="empty_filter_test",
        metadata={"hnsw:space": "cosine"},
    )
    rag.embeddings = FakeEmbeddings()

    rag.collection.add(
        ids=["test-01"],
        documents=["Cosine similarity supports semantic retrieval."],
        metadatas=[{"category": "similarity"}],
        embeddings=rag.embeddings.embed_documents(
            ["Cosine similarity supports semantic retrieval."]
        ),
    )

    results = rag.search(
        "cosine similarity",
        top_k=3,
        where={"category": "nonexistent"},
    )

    assert results == []


def test_answer_handles_empty_context_without_calling_llm(monkeypatch):
    from src import rag_pipeline

    rag = ChromaRAG.__new__(ChromaRAG)

    rag.search = lambda *args, **kwargs: []

    def unexpected_llm_call(*args, **kwargs):
        raise AssertionError(
            "LLM should not be called when context is empty"
        )

    monkeypatch.setattr(
        rag_pipeline,
        "ChatOllama",
        unexpected_llm_call,
    )

    answer, matches = rag.answer("What are the guardrails?")

    assert answer == "Information not found in the indexed PDF."
    assert matches == []