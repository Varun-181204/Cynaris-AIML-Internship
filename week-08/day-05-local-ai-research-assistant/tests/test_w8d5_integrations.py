import pytest
import requests
from src import ingestion, workflow


def test_embed_text_handles_ollama_timeout(monkeypatch):
    def fake_post(*args, **kwargs):
        raise requests.exceptions.Timeout("Ollama timed out")

    monkeypatch.setattr(ingestion.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.Timeout):
        ingestion.embed_text("test research text")


def test_retrieve_limits_results_to_collection_size(monkeypatch):
    class FakeCollection:
        def count(self):
            return 2

        def query(self, **kwargs):
            assert kwargs["n_results"] == 2
            return {
                "documents": [["RAG document", "ML document"]],
                "metadatas": [
                    [
                        {"source": "rag.txt"},
                        {"source": "ml.txt"},
                    ]
                ],
            }

    monkeypatch.setattr(workflow, "get_collection", lambda: FakeCollection())
    monkeypatch.setattr(workflow, "embed_text", lambda question: [0.1, 0.2])

    result = workflow.retrieve({"question": "Explain RAG"})

    assert result["contexts"] == ["RAG document", "ML document"]
    assert result["sources"] == ["ml.txt", "rag.txt"]


def test_retrieve_rejects_empty_collection(monkeypatch):
    class EmptyCollection:
        def count(self):
            return 0

    monkeypatch.setattr(workflow, "get_collection", lambda: EmptyCollection())

    with pytest.raises(RuntimeError, match="No documents indexed"):
        workflow.retrieve({"question": "Explain RAG"})


def test_generate_propagates_ollama_http_failure(monkeypatch):
    class FakeResponse:
        def raise_for_status(self):
            raise requests.exceptions.HTTPError("Ollama server error")

    def fake_post(*args, **kwargs):
        assert kwargs["timeout"] == 180
        return FakeResponse()

    monkeypatch.setattr(workflow.requests, "post", fake_post)

    with pytest.raises(requests.exceptions.HTTPError):
        workflow.generate(
            {
                "question": "Explain RAG",
                "contexts": ["RAG retrieves relevant documents."],
            }
        )
