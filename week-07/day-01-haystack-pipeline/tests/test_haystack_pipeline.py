import json
import sys
from pathlib import Path

import pytest
from haystack import Document

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR / "src"))

from haystack_pipeline import (
    QUESTIONS,
    TOP_K,
    ExtractiveReader,
    HaystackRetrievalSystem,
)

PDF_DIR = BASE_DIR / "data" / "pdfs"
EVIDENCE_PATH = BASE_DIR / "outputs" / "retrieval_evidence.json"


def test_five_pdf_documents_exist():
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    assert len(pdf_files) == 5


def test_ten_evaluation_questions_exist():
    assert len(QUESTIONS) == 10
    assert len({item["id"] for item in QUESTIONS}) == 10


def test_each_question_has_expected_source():
    for item in QUESTIONS:
        assert item["question"].strip()
        assert len(item["relevant_sources"]) >= 1


def test_reader_extracts_answer_and_source():
    reader = ExtractiveReader()

    documents = [
        Document(
            content=(
                "Haystack connects components through named inputs "
                "and outputs. Components form a modular pipeline."
            ),
            meta={"source_name": "haystack_pipeline_architecture.pdf"},
        )
    ]

    result = reader.run(
        query="How are Haystack components connected?",
        documents=documents,
    )

    assert result["answer"]
    assert result["source"] == "haystack_pipeline_architecture.pdf"
    assert result["documents"] == documents


def test_reader_handles_empty_retrieval():
    reader = ExtractiveReader()

    result = reader.run(
        query="Unknown question",
        documents=[],
    )

    assert result["answer"] == "No relevant document was retrieved."
    assert result["source"] == "None"
    assert result["documents"] == []


def test_top_k_deduplicates_repeated_pdf_sources():
    documents = [
        Document(content="Chunk 1", meta={"source_name": "a.pdf"}),
        Document(content="Chunk 2", meta={"source_name": "a.pdf"}),
        Document(content="Chunk 3", meta={"source_name": "b.pdf"}),
        Document(content="Chunk 4", meta={"source_name": "c.pdf"}),
        Document(content="Chunk 5", meta={"source_name": "d.pdf"}),
    ]

    result = HaystackRetrievalSystem.unique_source_results(documents)

    assert len(result) == TOP_K
    assert [doc.meta["source_name"] for doc in result] == [
        "a.pdf", "b.pdf", "c.pdf"
    ]


def test_top_k_returns_fewer_results_when_sources_are_repeated():
    documents = [
        Document(content=f"Chunk {i}", meta={"source_name": "same.pdf"})
        for i in range(5)
    ]

    result = HaystackRetrievalSystem.unique_source_results(documents)

    assert len(result) == 1
    assert result[0] == documents[0]


def test_evidence_file_exists():
    assert EVIDENCE_PATH.exists()


def test_evidence_contains_both_retrieval_methods():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert evidence["pdf_count"] == 5
    assert evidence["indexed_chunks"] > 0

    assert "bm25" in evidence["results"]
    assert "dense" in evidence["results"]


def test_both_methods_evaluated_on_ten_questions():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    assert len(evidence["results"]["bm25"]["queries"]) == 10
    assert len(evidence["results"]["dense"]["queries"]) == 10


def test_precision_scores_are_valid():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    for method in ("bm25", "dense"):
        score = evidence["results"][method]["mean_precision_at_3"]
        assert 0.0 <= score <= 1.0


def test_comparison_is_consistent():
    evidence = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))

    bm25 = evidence["results"]["bm25"]["mean_precision_at_3"]
    dense = evidence["results"]["dense"]["mean_precision_at_3"]

    expected_winner = (
        "Dense" if dense > bm25
        else "BM25" if bm25 > dense
        else "Tie"
    )

    assert evidence["comparison"]["higher_precision"] == expected_winner
