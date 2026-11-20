import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
EVIDENCE = BASE_DIR / "outputs" / "retrieval_evidence.json"


def test_source_documents_exist():
    files = list((BASE_DIR / "data" / "texts").glob("*.txt"))
    assert len(files) >= 5


def test_evidence_has_ten_queries_for_both_engines():
    assert EVIDENCE.exists(), "Run src/run_rag.py before running evaluation tests."
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    assert data["in_memory"]["query_count"] == 10
    assert data["chromadb"]["query_count"] == 10
    assert len(data["in_memory"]["results"]) == 10
    assert len(data["chromadb"]["results"]) == 10


def test_evidence_contains_latency_and_answers():
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for engine in ("in_memory", "chromadb"):
        assert data[engine]["mean_latency_ms"] > 0
        assert data[engine]["median_latency_ms"] > 0
        assert data[engine]["p95_latency_ms"] > 0
        assert data[engine]["stdev_latency_ms"] >= 0
        for result in data[engine]["results"]:
            assert result["latency_ms"] > 0
            assert result["answer"].strip()
            assert result["expected_source"]


def test_expected_source_tracking_is_boolean():
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    for engine in ("in_memory", "chromadb"):
        for result in data[engine]["results"]:
            assert isinstance(result["expected_source_retrieved"], bool)
