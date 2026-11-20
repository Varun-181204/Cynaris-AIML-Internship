import sys
import pytest
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from rag_system import chunk_text, build_workflow, ask_question


def test_chunk_text_returns_chunks():
    text = "A" * 1800
    chunks = chunk_text(text, chunk_size=800, overlap=100)

    assert len(chunks) == 3
    assert all(chunks)


def test_chunk_text_preserves_short_text():
    assert chunk_text("RAG is useful.") == ["RAG is useful."]


def test_chunk_text_handles_empty_text():
    assert chunk_text("   ") == []


def test_chunk_text_rejects_invalid_overlap():
    try:
        chunk_text("sample", chunk_size=100, overlap=100)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_workflow_compiles():
    workflow = build_workflow()
    assert workflow is not None


def test_ask_question_rejects_empty_input():
    with pytest.raises(ValueError, match="cannot be empty"):
        ask_question("   ")


def test_ask_question_rejects_overlong_input():
    with pytest.raises(ValueError, match="character limit"):
        ask_question("A" * 1001)
