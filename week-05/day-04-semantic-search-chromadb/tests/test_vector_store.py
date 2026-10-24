"""W5D4 tests for ChromaDB semantic search and metadata filtering."""
import sys
from pathlib import Path
from unittest.mock import patch

import chromadb
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import vector_store


@pytest.fixture
def populated_collection():
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name="w5d4_test_semantic_search",
        metadata={"hnsw:space": "cosine"},
    )
    collection.add(
        ids=["doc-ai", "doc-food", "doc-code"],
        documents=[
            "Machine learning models learn patterns from training data.",
            "A recipe uses vegetables, herbs, and cooking oil.",
            "Python functions organize reusable program logic.",
        ],
        metadatas=[
            {"category": "ai"},
            {"category": "food"},
            {"category": "programming"},
        ],
        embeddings=[
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    )
    yield collection
    client.delete_collection("w5d4_test_semantic_search")


def test_document_corpus_contains_twenty_documents():
    assert len(vector_store.DOCUMENTS) == 20
    assert len({item[0] for item in vector_store.DOCUMENTS}) == 20


def test_similarity_search_returns_nearest_cosine_match(populated_collection):
    with patch.object(vector_store, "ollama_embed", return_value=[[0.98, 0.02, 0.0]]):
        result = vector_store.similarity_search(
            populated_collection, "machine learning", n_results=2
        )

    assert len(result["ids"][0]) == 2
    assert result["ids"][0][0] == "doc-ai"
    assert result["distances"][0][0] <= result["distances"][0][1]


def test_similarity_search_filters_by_metadata(populated_collection):
    with patch.object(vector_store, "ollama_embed", return_value=[[1.0, 0.0, 0.0]]):
        result = vector_store.similarity_search(
            populated_collection,
            "machine learning",
            n_results=3,
            where={"category": "food"},
        )

    assert result["ids"][0] == ["doc-food"]
    assert result["metadatas"][0][0]["category"] == "food"


def test_similarity_search_rejects_empty_query(populated_collection):
    with pytest.raises(ValueError, match="non-empty"):
        vector_store.similarity_search(populated_collection, "  ")


def test_similarity_search_rejects_invalid_result_count(populated_collection):
    with pytest.raises(ValueError, match="at least 1"):
        vector_store.similarity_search(populated_collection, "test", n_results=0)
