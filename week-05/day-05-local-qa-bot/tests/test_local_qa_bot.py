import sys
from pathlib import Path
from unittest.mock import patch

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

import local_qa_bot as qa


@pytest.fixture
def bot(tmp_path):
    instance = qa.LocalQABot(
        storage_dir=tmp_path / "chroma",
        ollama_url="http://localhost:11434",
        collection_name="test_local_qa",
    )

    # Populate a small deterministic collection without contacting Ollama.
    instance.collection.upsert(
        ids=["doc-1", "doc-2"],
        documents=[
            "A vector database stores vector embeddings and supports similarity search.",
            "Semantic search retrieves text based on meaning rather than exact keywords.",
        ],
        metadatas=[
            {"source": "guide.pdf", "page": 1},
            {"source": "guide.pdf", "page": 2},
        ],
        embeddings=[[1.0, 0.0], [0.0, 1.0]],
    )
    return instance


def test_extract_pdf_chunks_from_sample():
    pdf = (
        Path(__file__).resolve().parents[1]
        / "sample_data"
        / "vector_databases_guide.pdf"
    )

    chunks = qa.extract_pdf_chunks(pdf, chunk_size=700, overlap=100)

    assert chunks
    assert all(chunk["text"].strip() for chunk in chunks)
    assert all(chunk["source"] == pdf.name for chunk in chunks)
    assert all(chunk["page"] >= 1 for chunk in chunks)


def test_extract_missing_pdf_raises():
    with pytest.raises(FileNotFoundError):
        qa.extract_pdf_chunks("missing-file.pdf")


def test_empty_question_is_rejected(bot):
    with pytest.raises(ValueError, match="question"):
        bot.retrieve("   ")


@pytest.mark.parametrize("invalid_top_k", [0, -1, 1.5, "2", True])
def test_top_k_must_be_a_positive_integer(bot, invalid_top_k):
    with pytest.raises(ValueError, match="top_k"):
        bot.retrieve("vector database", top_k=invalid_top_k)


def test_retrieve_returns_source_metadata(bot):
    with patch.object(
        qa,
        "embed_texts",
        return_value=[[1.0, 0.0]],
    ):
        results = bot.retrieve("vector database", top_k=1)

    assert len(results) == 1
    assert results[0]["text"]
    assert results[0]["source"] == "guide.pdf"
    assert results[0]["page"] == 1


def test_answer_uses_retrieved_context_and_returns_sources(bot):
    chat_response = {
        "message": {
            "content": "A vector database stores embeddings and supports similarity search."
        }
    }

    with (
        patch.object(qa, "embed_texts", return_value=[[1.0, 0.0]]),
        patch.object(qa, "_post_json", return_value=chat_response) as mock_post,
    ):
        result = bot.answer("What does a vector database do?", top_k=1)

    assert "embeddings" in result["answer"]
    assert result["sources"][0]["source"] == "guide.pdf"
    assert result["sources"][0]["page"] == 1

    request = mock_post.call_args.args
    assert request[1] == "/api/chat"
    payload = request[2]
    assert payload["stream"] is False
    assert "vector database stores vector embeddings" in payload["messages"][1]["content"]


def test_empty_collection_is_reported(tmp_path):
    empty_bot = qa.LocalQABot(
        storage_dir=tmp_path / "empty-chroma",
        collection_name="empty_test_collection",
    )

    with pytest.raises(qa.LocalQABotError, match="No documents are indexed"):
        empty_bot.retrieve("What is semantic search?")


def test_empty_ollama_chat_answer_raises(bot):
    with (
        patch.object(qa, "embed_texts", return_value=[[1.0, 0.0]]),
        patch.object(qa, "_post_json", return_value={"message": {"content": "  "}}),
    ):
        with pytest.raises(qa.LocalQABotError, match="empty chat answer"):
            bot.answer("What is a vector database?", top_k=1)


def test_embed_texts_rejects_empty_input():
    with pytest.raises(ValueError, match="texts"):
        qa.embed_texts([])


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), -float("inf")])
def test_embed_texts_rejects_non_finite_values(invalid_value):
    with (
        patch.object(
            qa,
            "_post_json",
            return_value={"embeddings": [[invalid_value]]},
        ),
        pytest.raises(qa.LocalQABotError, match="non-finite"),
    ):
        qa.embed_texts(["sample text"])

@pytest.mark.parametrize("invalid_texts", [[None], [123], [""], ["   "], ["valid", None]])
def test_embed_texts_rejects_invalid_elements(invalid_texts):
    with pytest.raises(ValueError, match="texts"):
        qa.embed_texts(invalid_texts)
