import pytest
from src.ingestion import split_text
from src.workflow import ask_question


def test_split_text_returns_chunks():
    chunks = split_text("A" * 150, chunk_size=60, overlap=10)
    assert len(chunks) == 3
    assert all(chunks)


def test_split_text_rejects_invalid_overlap():
    with pytest.raises(ValueError):
        split_text("sample", chunk_size=10, overlap=10)


def test_split_text_empty_input():
    assert split_text("") == []


def test_ask_question_rejects_empty_question():
    with pytest.raises(ValueError):
        ask_question("   ")
