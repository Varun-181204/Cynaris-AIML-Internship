"""Unit tests for W5D3 PDF chunking and RAG orchestration."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import pdf_rag


@pytest.fixture
def sample_pdf(tmp_path):
    path = tmp_path / "guide.pdf"
    pdf = canvas.Canvas(str(path), pagesize=letter)
    pdf.drawString(50, 750, "ChromaDB stores embeddings and supports cosine similarity search.")
    pdf.showPage()
    pdf.drawString(50, 750, "RAG retrieves document chunks and passes them to a language model.")
    pdf.showPage()
    pdf.save()
    return path


def test_extract_pdf_chunks_returns_text_and_page_metadata(sample_pdf):
    chunks = pdf_rag.extract_pdf_chunks(sample_pdf, chunk_size=100, overlap=10)
    assert len(chunks) >= 2
    assert all(chunk["text"] for chunk in chunks)
    assert {chunk["page"] for chunk in chunks} == {1, 2}
    assert all(chunk["source"] == "guide.pdf" for chunk in chunks)


def test_chunking_rejects_invalid_overlap(sample_pdf):
    with pytest.raises(ValueError):
        pdf_rag.extract_pdf_chunks(sample_pdf, chunk_size=10, overlap=10)


def test_ask_ollama_rejects_empty_question_context_is_not_relevant():
    # The request function should fail clearly rather than return an empty answer.
    with patch("pdf_rag.urlopen") as mocked:
        mocked.return_value.__enter__.return_value.read.return_value = (
            b'{"message":{"content":"RAG retrieves relevant chunks."}}'
        )
        assert "retrieves" in pdf_rag.ask_ollama("What is RAG?", "RAG retrieves chunks.")
