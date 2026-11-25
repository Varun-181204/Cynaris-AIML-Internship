import dataclasses
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from haystack import Document, component

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from api import create_app
from haystack_pipeline import (
    HaystackRetrievalSystem,
    resolve_embedding_model,
    resolve_ollama_timeout,
    resolve_ollama_url,
    sanitize_url_for_display,
)


@component
class _FailingDocEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        raise ConnectionError("Ollama unreachable")


@component
class _WorkingDocEmbedder:
    @component.output_types(documents=list[Document])
    def run(self, documents: list[Document]):
        embedded_docs = [
            dataclasses.replace(doc, embedding=[0.1, 0.2, 0.3, 0.4])
            for doc in documents
        ]
        return {"documents": embedded_docs}


@component
class _FailingTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str):
        raise ConnectionError("Ollama unreachable")


@component
class _WorkingTextEmbedder:
    @component.output_types(embedding=list[float])
    def run(self, text: str):
        return {"embedding": [0.1, 0.2, 0.3, 0.4]}


def test_resolve_ollama_url_normalizes_input(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "  https://ollama.example.com/  ")
    assert resolve_ollama_url() == "https://ollama.example.com"

    monkeypatch.setenv("OLLAMA_URL", "   ")
    assert resolve_ollama_url() == "http://localhost:11434"

    assert (
        resolve_ollama_url("http://ollama.railway.internal:11434/")
        == "http://ollama.railway.internal:11434"
    )


def test_resolve_model_and_timeout_helpers(monkeypatch):
    monkeypatch.setenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")
    monkeypatch.setenv("OLLAMA_TIMEOUT", "45")
    assert resolve_embedding_model() == "nomic-embed-text"
    assert resolve_ollama_timeout() == 45

    monkeypatch.setenv("OLLAMA_TIMEOUT", "invalid")
    assert resolve_ollama_timeout() == 30


def test_sanitize_url_strips_credentials():
    assert (
        sanitize_url_for_display("https://user:secret@ollama.example.com:11434")
        == "https://ollama.example.com:11434"
    )
    assert (
        sanitize_url_for_display("http://ollama.railway.internal:11434")
        == "http://ollama.railway.internal:11434"
    )


def test_api_starts_and_serves_bm25_without_ollama(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:19999")
    retrieval_system = HaystackRetrievalSystem(ollama_timeout=1)
    retrieval_system._create_document_embedder = _FailingDocEmbedder
    retrieval_system._create_text_embedder = _FailingTextEmbedder

    test_app = create_app(retrieval_system=retrieval_system)
    with TestClient(test_app) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200
        root_payload = root_response.json()
        assert root_payload["ready"] is True
        assert root_payload["bm25_ready"] is True
        assert root_payload["dense_ready"] is False
        assert root_payload["indexed_chunks"] == 22

        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "ok"
        assert health_payload["ready"] is True
        assert health_payload["bm25_ready"] is True
        assert health_payload["dense_ready"] is False
        assert health_payload["indexed_chunks"] == 22

        query_response = client.post(
            "/query",
            json={
                "question": "How are components connected in a Haystack pipeline?",
                "method": "bm25",
            },
        )
        assert query_response.status_code == 200
        query_payload = query_response.json()
        assert query_payload["method"] == "bm25"
        assert (
            "haystack_pipeline_architecture.pdf" in query_payload["retrieved_sources"]
        )
        assert query_payload["answer"]


def test_dense_query_returns_503_when_ollama_unreachable(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://127.0.0.1:19999")
    retrieval_system = HaystackRetrievalSystem(ollama_timeout=1)
    retrieval_system._create_document_embedder = _FailingDocEmbedder
    retrieval_system._create_text_embedder = _FailingTextEmbedder

    test_app = create_app(retrieval_system=retrieval_system)
    with TestClient(test_app) as client:
        response = client.post(
            "/query",
            json={
                "question": "How does dense retrieval use embeddings?",
                "method": "dense",
            },
        )
        assert response.status_code == 503
        assert "Dense retrieval is unavailable" in response.json()["detail"]


def test_dense_query_recovers_lazily_when_ollama_becomes_available(monkeypatch):
    monkeypatch.setenv("OLLAMA_URL", "http://ollama.railway.internal:11434")
    retrieval_system = HaystackRetrievalSystem(ollama_timeout=5)
    retrieval_system._create_document_embedder = _FailingDocEmbedder
    retrieval_system._create_text_embedder = _WorkingTextEmbedder

    test_app = create_app(retrieval_system=retrieval_system)
    with TestClient(test_app) as client:
        assert client.get("/health").json()["dense_ready"] is False

        retrieval_system._create_document_embedder = _WorkingDocEmbedder
        response = client.post(
            "/query",
            json={
                "question": "How does dense retrieval use embeddings?",
                "method": "dense",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["method"] == "dense"
        assert len(payload["retrieved_sources"]) >= 1
        assert retrieval_system.dense_ready is True
        assert retrieval_system.document_store.count_documents() == 22


def test_query_validation_rejects_invalid_inputs():
    retrieval_system = HaystackRetrievalSystem(ollama_timeout=1)
    retrieval_system._create_document_embedder = _FailingDocEmbedder

    test_app = create_app(retrieval_system=retrieval_system)
    with TestClient(test_app) as client:
        assert (
            client.post(
                "/query", json={"question": "   ", "method": "bm25"}
            ).status_code
            == 422
        )
        assert (
            client.post("/query", json={"question": "hi", "method": "bm25"}).status_code
            == 422
        )
        assert (
            client.post(
                "/query",
                json={"question": "Valid question?", "method": "hybrid"},
            ).status_code
            == 422
        )
