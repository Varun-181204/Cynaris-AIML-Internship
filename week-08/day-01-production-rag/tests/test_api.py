"""Tests for the W8D1 FastAPI service."""

import sys
from pathlib import Path
from unittest.mock import Mock

import requests
from fastapi.testclient import TestClient

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

import api

client = TestClient(api.app)


def test_health_check_success(monkeypatch):
    response_mock = Mock()
    response_mock.raise_for_status.return_value = None

    monkeypatch.setattr(
        api.requests,
        "get",
        lambda *args, **kwargs: response_mock,
    )

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["ollama"] == "connected"


def test_health_check_ollama_unavailable(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("Ollama unavailable")

    monkeypatch.setattr(
        api.requests,
        "get",
        raise_connection_error,
    )

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["detail"] == "Ollama service is unavailable."


def test_ingest_success(monkeypatch):
    monkeypatch.setattr(
        api.rag_system,
        "ingest_documents",
        lambda: 12,
    )

    response = client.post("/ingest")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["chunks_indexed"] == 12


def test_ask_success(monkeypatch):
    expected_result = {
        "answer": "RAG combines retrieval with generation.",
        "contexts": ["RAG fundamentals document."],
        "sources": ["rag_fundamentals.txt"],
        "latency_seconds": 1.25,
    }

    monkeypatch.setattr(
        api.rag_system,
        "ask_question",
        lambda question, use_crew=False: expected_result,
    )

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == expected_result["answer"]
    assert response.json()["sources"] == expected_result["sources"]


def test_ask_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": ""},
    )

    assert response.status_code == 422


def test_ask_handles_empty_index(monkeypatch):
    def raise_runtime_error(*args, **kwargs):
        raise RuntimeError("Index is empty. Run ingestion first.")

    monkeypatch.setattr(
        api.rag_system,
        "ask_question",
        raise_runtime_error,
    )

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 409
    assert "Index is empty" in response.json()["detail"]


def test_ask_handles_ollama_failure(monkeypatch):
    def raise_connection_error(*args, **kwargs):
        raise requests.ConnectionError("Ollama unavailable")

    monkeypatch.setattr(
        api.rag_system,
        "ask_question",
        raise_connection_error,
    )

    response = client.post(
        "/ask",
        json={"question": "What is RAG?"},
    )

    assert response.status_code == 502
